from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..train.sft.trainer import CustomSeq2SeqTrainer
from .sampling import TaskBalancedBatchSampler
from .tela import ReptileLoRAEngine, TELAConfig
from .ttald import TTALDConfig, TTALDController


METADATA_KEYS = {"task_id", "task_index", "temporal_stats", "source_index", "split", "domain_labels"}


def split_model_inputs(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = {key: value for key, value in inputs.items() if key in METADATA_KEYS}
    model_inputs = {key: value for key, value in inputs.items() if key not in METADATA_KEYS}
    return model_inputs, metadata


def slice_inputs(inputs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    return {key: value[indices] for key, value in inputs.items() if torch.is_tensor(value)}


class InnovationV2Seq2SeqTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_batch_sampler = None
        self.ttald = None
        self.ttald_resume_strategy = "not_enabled"
        self.tela = None
        if self.finetuning_args.use_ttald:
            self.ttald = TTALDController(
                self.model,
                TTALDConfig(
                    p_min=self.finetuning_args.ttald_p_min,
                    p_max=self.finetuning_args.ttald_p_max,
                    volatility_weight=self.finetuning_args.ttald_volatility_weight,
                    loss_weight=self.finetuning_args.ttald_loss_weight,
                    gap_weight=self.finetuning_args.ttald_gap_weight,
                    ema_decay=self.finetuning_args.ttald_ema_decay,
                    runtime_path=f"{self.args.output_dir}/ttald_runtime.jsonl",
                ),
            )
            self.ttald_resume_strategy = "initialized_empty"
            resume_path = self.args.resume_from_checkpoint
            if isinstance(resume_path, str) and resume_path:
                state_path = Path(resume_path) / "ttald_state.json"
                if state_path.exists():
                    import json

                    self.ttald.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
                    self.ttald_resume_strategy = f"restored:{state_path}"
                else:
                    self.ttald_resume_strategy = f"reinitialized_missing_state:{state_path}"
        if self.finetuning_args.use_tela:
            if self.args.world_size != 1:
                raise ValueError("TELA v2 currently requires single-process training to preserve exact Reptile updates.")
            if self.args.gradient_accumulation_steps != 1:
                raise ValueError(
                    "TELA v2 performs one explicit Reptile meta-update per trainer step and requires "
                    "gradient_accumulation_steps=1."
                )
            self.tela = ReptileLoRAEngine(
                self.model,
                TELAConfig(
                    algorithm=self.finetuning_args.tela_algorithm,
                    tasks_per_meta_batch=self.finetuning_args.tela_tasks_per_meta_batch,
                    support_size=self.finetuning_args.tela_support_size,
                    query_size=self.finetuning_args.tela_query_size,
                    inner_steps=self.finetuning_args.tela_inner_steps,
                    inner_lr=self.finetuning_args.tela_inner_lr,
                    meta_step_size=self.finetuning_args.tela_meta_step_size,
                    adapt_parameters=self.finetuning_args.tela_adapt_parameters,
                ),
                self._tela_loss,
            )

    def get_train_dataloader(self) -> DataLoader:
        if not self.finetuning_args.use_task_balanced_sampling:
            return super().get_train_dataloader()
        if "task_id" not in self.train_dataset.column_names:
            raise ValueError("Task-balanced training requires the processed dataset task_id column.")
        sampler = TaskBalancedBatchSampler(
            self.train_dataset["task_id"],
            mode=self.finetuning_args.task_batch_mode,
            tasks_per_batch=self.finetuning_args.tasks_per_batch,
            samples_per_task=self.finetuning_args.samples_per_task,
            temperature=self.finetuning_args.task_sampling_temperature,
            seed=self.args.seed,
            # Accelerate applies distributed BatchSamplerShard after prepare().
            num_replicas=1,
            rank=0,
        )
        self.task_batch_sampler = sampler
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0,
        )
        return self.accelerator.prepare(dataloader)

    def _tela_loss(self, model: torch.nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        task_id = inputs.get("__task_id")
        temporal_stats = inputs.get("__temporal_stats")
        update_ttald_ema = bool(inputs.get("__update_ttald_ema", False))
        if self.ttald is not None:
            if task_id is None or temporal_stats is None:
                raise ValueError("TELA+TTALD requires per-episode task metadata.")
            self.ttald.before_batch([task_id] * temporal_stats.shape[0], temporal_stats)
        outputs = model(**{key: value for key, value in inputs.items() if not key.startswith("__")})
        if self.ttald is not None:
            if update_ttald_ema:
                self.ttald.update_loss(task_id, float(outputs.loss.detach().cpu()))
            self.ttald.assert_forward_used()
        return outputs.loss

    def compute_loss(self, model, inputs, return_outputs: bool = False, *args, **kwargs):
        model_inputs, metadata = split_model_inputs(inputs)
        use_ttald = self.ttald is not None and model.training
        if use_ttald:
            if "task_id" not in metadata or "temporal_stats" not in metadata:
                raise ValueError("TTALD requires task_id and temporal_stats in every batch.")
            self.ttald.before_batch(metadata["task_id"], metadata["temporal_stats"])
        outputs = model(**model_inputs)
        loss = outputs.loss
        if use_ttald:
            self.ttald.update_loss(metadata["task_id"][0], float(loss.detach().cpu()))
            self.ttald.assert_forward_used()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.tela is None:
            return super().training_step(model, inputs, num_items_in_batch)
        inputs = self._prepare_inputs(inputs)
        model_inputs, metadata = split_model_inputs(inputs)
        task_ids = metadata.get("task_id")
        if not task_ids:
            raise ValueError("TELA requires explicit task_id metadata.")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, task_id in enumerate(task_ids):
            grouped[str(task_id)].append(index)
        if len(grouped) != self.finetuning_args.tela_tasks_per_meta_batch:
            raise ValueError("TELA batch does not contain the configured number of tasks.")
        episodes = []
        needed = self.finetuning_args.tela_support_size + self.finetuning_args.tela_query_size
        for task_id in sorted(grouped):
            indices = grouped[task_id]
            if len(indices) != needed:
                raise ValueError(f"TELA task {task_id} expected {needed} samples, got {len(indices)}.")
            support_indices = indices[: self.finetuning_args.tela_support_size]
            query_indices = indices[self.finetuning_args.tela_support_size :]
            support = slice_inputs(model_inputs, support_indices)
            query = slice_inputs(model_inputs, query_indices)
            if self.ttald is not None:
                support["__task_id"] = task_id
                support["__temporal_stats"] = metadata["temporal_stats"][support_indices]
                support["__update_ttald_ema"] = True
                query["__task_id"] = task_id
                query["__temporal_stats"] = metadata["temporal_stats"][query_indices]
                query["__update_ttald_ema"] = False
            episodes.append(
                ([support], [query])
            )
        result = self.tela.meta_step(episodes)
        return torch.tensor(result.query_loss_after, device=self.args.device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        model_inputs, _ = split_model_inputs(inputs)
        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )

    def save_state(self) -> None:
        super().save_state()
        if self.task_batch_sampler is not None and self.is_world_process_zero():
            import json
            from pathlib import Path

            path = Path(self.args.output_dir) / "task_exposure_counts.json"
            path.write_text(json.dumps(dict(self.task_batch_sampler.exposure_counts), indent=2), encoding="utf-8")
        if self.ttald is not None and self.is_world_process_zero():
            import json
            from pathlib import Path

            path = Path(self.args.output_dir) / "ttald_state.json"
            state = self.ttald.state_dict()
            state["resume_strategy"] = self.ttald_resume_strategy
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
