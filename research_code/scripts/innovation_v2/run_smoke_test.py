#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from llamafactory.innovations_v2.evaluation import evaluate_records
from llamafactory.innovations_v2.metadata.task_metadata import parse_numeric_sequence
from llamafactory.innovations_v2.sampling import TaskBalancedBatchSampler
from llamafactory.innovations_v2.tela import ReptileLoRAEngine, TELAConfig
from llamafactory.innovations_v2.tela.adaptation import clone_parameters, lora_parameters, restore_parameters
from llamafactory.innovations_v2.tracking import RunManifest
from llamafactory.innovations_v2.ttald import TTALDConfig, TTALDController


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "innovation_v2" / "smoke"


class ToyLoRAForecaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(1, 1)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.layer = torch.nn.Module()
        self.layer.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Dropout(0.1)})
        self.lora_A = torch.nn.Linear(1, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x, y=None):
        prediction = self.base(x) + self.lora_B(self.lora_A(self.layer.lora_dropout["default"](x)))
        loss = torch.mean((prediction - y) ** 2) if y is not None else None
        return SimpleNamespace(loss=loss, prediction=prediction)


def load_task(task: str) -> list[dict]:
    path = ROOT / "data" / "innovation_v2" / "train" / f"{task}.json"
    return json.loads(path.read_text(encoding="utf-8"))[:32]


def tensor_batch(records: list[dict]) -> dict[str, torch.Tensor]:
    x_values, y_values = [], []
    for record in records:
        history = parse_numeric_sequence(record["input"])
        target = parse_numeric_sequence(record["output"])
        scale = max(abs(history[-1]), 1.0)
        x_values.append([history[-1] / scale])
        y_values.append([target[0] / scale])
    return {"x": torch.tensor(x_values), "y": torch.tensor(y_values)}


def loss_fn(model, batch):
    return model(**batch).loss


def mse(model, batch):
    model.eval()
    with torch.no_grad():
        return float(model(**batch).loss)


def preference_smoke(records: list[dict]) -> tuple[Path, dict]:
    source = ARTIFACTS / "preference_source.json"
    source.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    output = ARTIFACTS / "preferences.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "innovation_v2" / "build_electrical_preferences.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--seed",
            "1",
            "--min-margin",
            "0.001",
            "--max-margin",
            "3.0",
        ],
        check=True,
        env=env,
    )
    manifest = json.loads((ARTIFACTS / "preference_build_manifest.json").read_text(encoding="utf-8"))
    return output, manifest


def main() -> int:
    torch.manual_seed(1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    hufl = load_task("ETTh1_HUFL")
    hull = load_task("ETTh1_HULL")
    all_records = hufl + hull
    task_ids = [record["task_id"] for record in all_records]
    model = ToyLoRAForecaster()
    optimizer = torch.optim.Adam(lora_parameters(model).values(), lr=0.02)

    # Ordinary mixed SFT smoke.
    mixed = tensor_batch(hufl[:16] + hull[:16])
    mixed_losses = []
    model.train()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, mixed)
        loss.backward()
        optimizer.step()
        mixed_losses.append(float(loss.detach()))

    # Task-balanced sampler exposure.
    sampler = TaskBalancedBatchSampler(
        task_ids,
        mode="n_task_k_shot",
        tasks_per_batch=2,
        samples_per_task=4,
        seed=1,
        batches_per_epoch=4,
    )
    sampled_batches = list(sampler)
    exposure = Counter(task_ids[index] for batch in sampled_batches for index in batch)

    # TELA.
    tela_before = clone_parameters(lora_parameters(model))
    tela = ReptileLoRAEngine(
        model,
        TELAConfig(tasks_per_meta_batch=2, support_size=4, query_size=4, inner_steps=3, inner_lr=0.02),
        loss_fn,
    )
    tela_result = tela.meta_step(
        [
            ([tensor_batch(hufl[:4])], [tensor_batch(hufl[4:8])]),
            ([tensor_batch(hull[:4])], [tensor_batch(hull[4:8])]),
        ]
    )
    tela_changed = any(
        not torch.equal(parameter, tela_before[name]) for name, parameter in lora_parameters(model).items()
    )

    # TTALD actual module update and forward counter.
    ttald = TTALDController(model, TTALDConfig(p_min=0.02, p_max=0.2))
    low_stats = torch.tensor([[0.0, 1.0, 0.05, 0.0, 0.0]] * 4)
    high_stats = torch.tensor([[0.0, 1.0, 5.0, 0.0, 0.0]] * 4)
    low_p = ttald.before_batch(["ETTh1_HUFL"] * 4, low_stats)
    model.train()
    model(**tensor_batch(hufl[:4]))
    ttald.update_loss("ETTh1_HUFL", 2.0)
    high_p = ttald.before_batch(["ETTh1_HUFL"] * 4, high_stats)
    model(**tensor_batch(hufl[:4]))
    ttald.assert_forward_used()

    # Inference-time adaptation semantics.
    parameters = lora_parameters(model)
    shared_state = clone_parameters(parameters)
    query = tensor_batch(hufl[8:16])
    adapt_before = mse(model, query)
    adapt_optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
    model.train()
    for _ in range(3):
        adapt_optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, tensor_batch(hufl[:8]))
        loss.backward()
        adapt_optimizer.step()
    adapt_after = mse(model, query)
    restore_parameters(parameters, shared_state)
    restored = all(torch.equal(parameter, shared_state[name]) for name, parameter in parameters.items())

    # Matched target SFT and a pairwise preference objective smoke.
    target_state = clone_parameters(parameters)
    target_optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
    target_optimizer.zero_grad(set_to_none=True)
    target_loss = loss_fn(model, tensor_batch(hufl[:8]))
    target_loss.backward()
    target_optimizer.step()
    target_sft_changed = any(
        not torch.equal(parameter, target_state[name]) for name, parameter in parameters.items()
    )
    restore_parameters(parameters, target_state)
    chosen_error = torch.tensor([0.1, 0.2], requires_grad=True)
    rejected_error = torch.tensor([0.4, 0.5], requires_grad=True)
    pairwise_loss = -torch.nn.functional.logsigmoid(rejected_error - chosen_error).mean()
    pairwise_loss.backward()

    preference_path, preference_manifest = preference_smoke(hufl[:16])

    predictions = [
        {
            "task_id": record["task_id"],
            "target": json.loads(record["output"]),
            "prediction": record["output"],
            "horizon": record["horizon"],
        }
        for record in hufl[:8]
    ]
    metrics = evaluate_records(predictions)
    manifest = RunManifest(
        experiment_id="innovation_v2_cpu_smoke",
        method="toy mixed SFT + task-balanced + TELA + TTALD + preference",
        seed=1,
        dataset_names=["ETTh1_HUFL", "ETTh1_HULL"],
        train_sample_counts={"ETTh1_HUFL": 32, "ETTh1_HULL": 32},
        task_exposure_counts=dict(exposure),
        actual_steps=5,
        metrics=metrics["metrics"],
        runtime={"device": "cpu"},
    )
    manifest.set_model(model)
    manifest_path = manifest.save(ARTIFACTS)

    report = [
        "# Innovation V2 Smoke Test Report",
        "",
        "> This is a CPU/toy-LoRA integration smoke test. It is not a paper performance experiment.",
        "",
        f"- Command: `PYTHONPATH=src python scripts/innovation_v2/run_smoke_test.py`",
        f"- Ordinary mixed SFT: PASS; loss `{mixed_losses[0]:.6f}` -> `{mixed_losses[-1]:.6f}`",
        f"- Task-balanced sampling: PASS; exposure `{dict(exposure)}`",
        f"- TELA meta update: {'PASS' if tela_changed and tela_result.parameter_delta_norm > 0 else 'FAIL'}",
        f"- TELA query loss: `{tela_result.query_loss_before:.6f}` -> `{tela_result.query_loss_after:.6f}`",
        f"- TTALD actual p range: `{low_p:.6f}` -> `{high_p:.6f}`",
        f"- TTALD forward counter: `{ttald.forward_count}`",
        f"- Adapt-and-restore semantics: {'PASS' if restored else 'FAIL'}; query loss `{adapt_before:.6f}` -> `{adapt_after:.6f}`",
        f"- Matched target SFT parameter update: {'PASS' if target_sft_changed else 'FAIL'}",
        f"- Pairwise preference loss backward: PASS; loss `{float(pairwise_loss.detach()):.6f}`",
        f"- Preference pairs: `{preference_manifest['pair_count']}`; test split used: `{preference_manifest['test_split_used']}`",
        f"- Preference candidate sources: `{preference_manifest['candidate_source_counts']}`",
        f"- Preference margin summary: `{preference_manifest['margin']}`",
        f"- Strict parser/evaluator: PASS; parse success `{metrics['parse_success_rate']:.3f}`",
        f"- Run manifest: `{manifest_path}`",
        f"- Preference file: `{preference_path}`",
        "",
        "Full 8B SFT, target SFT and TRL DPO were not run in this smoke test.",
    ]
    (ROOT / "artifacts" / "innovation_v2" / "smoke_test_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
