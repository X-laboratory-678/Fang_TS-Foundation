import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TTALDConfig:
    p_min: float = 0.02
    p_max: float = 0.20
    volatility_weight: float = 1.0
    loss_weight: float = 0.5
    gap_weight: float = 0.0
    ema_decay: float = 0.95
    enabled: bool = True
    runtime_path: str = ""

    def validate(self) -> None:
        if not 0 <= self.p_min <= self.p_max < 1:
            raise ValueError("TTALD requires 0 <= p_min <= p_max < 1.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("TTALD ema_decay must be in [0, 1).")


class TTALDController:
    def __init__(self, model: torch.nn.Module, config: TTALDConfig) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.task_loss_ema: dict[str, float] = {}
        self.forward_count = 0
        self.step = 0
        self.dropouts = self._find_lora_dropouts()
        if config.enabled and not self.dropouts:
            raise ValueError("TTALD enabled but no actual LoRA dropout modules were found.")
        self._hooks = [dropout.register_forward_hook(self._forward_hook) for dropout in self.dropouts]

    def _find_lora_dropouts(self) -> list[torch.nn.Dropout]:
        found: list[torch.nn.Dropout] = []
        seen: set[int] = set()
        for module in self.model.modules():
            lora_dropout = getattr(module, "lora_dropout", None)
            if lora_dropout is None:
                continue
            values = lora_dropout.values() if hasattr(lora_dropout, "values") else []
            for dropout in values:
                if isinstance(dropout, torch.nn.Dropout) and id(dropout) not in seen:
                    found.append(dropout)
                    seen.add(id(dropout))
        return found

    def _forward_hook(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        self.forward_count += 1

    @staticmethod
    def _normalize_positive(value: float) -> float:
        value = max(0.0, float(value))
        return value / (1.0 + value)

    def compute_probability(self, task_id: str, volatility: float, generalization_gap: float = 0.0) -> float:
        loss_ema = self.task_loss_ema.get(task_id, 0.0)
        score = (
            self.config.volatility_weight * self._normalize_positive(volatility)
            + self.config.loss_weight * self._normalize_positive(loss_ema)
            + self.config.gap_weight * float(generalization_gap)
        )
        sigmoid = 1.0 / (1.0 + math.exp(-score))
        return min(self.config.p_max, max(self.config.p_min, self.config.p_min + (self.config.p_max - self.config.p_min) * sigmoid))

    def before_batch(self, task_ids: list[str], temporal_stats: torch.Tensor | list[list[float]], generalization_gap: float = 0.0) -> float:
        if not self.config.enabled:
            return self.dropouts[0].p if self.dropouts else 0.0
        unique_tasks = set(task_ids)
        if len(unique_tasks) != 1:
            raise ValueError("TTALD requires a task-homogeneous batch.")
        task_id = next(iter(unique_tasks))
        stats = torch.as_tensor(temporal_stats, dtype=torch.float32)
        if stats.ndim != 2 or stats.shape[1] < 3:
            raise ValueError("TTALD temporal_stats must have shape [batch, >=3].")
        volatility = float(stats[:, 2].mean())
        probability = self.compute_probability(task_id, volatility, generalization_gap)
        for dropout in self.dropouts:
            dropout.p = probability
        if any(abs(dropout.p - probability) > 1e-12 for dropout in self.dropouts):
            raise RuntimeError("TTALD failed to update an actual LoRA dropout module.")
        self.step += 1
        self._log(task_id, volatility, probability)
        return probability

    def update_loss(self, task_id: str, loss: float) -> None:
        loss = float(loss)
        previous = self.task_loss_ema.get(task_id, loss)
        self.task_loss_ema[task_id] = self.config.ema_decay * previous + (1 - self.config.ema_decay) * loss

    def assert_forward_used(self) -> None:
        if self.config.enabled and self.forward_count <= 0:
            raise RuntimeError("TTALD changed dropout probabilities but no LoRA dropout forward hook was called.")

    def state_dict(self) -> dict[str, Any]:
        return {"task_loss_ema": self.task_loss_ema, "step": self.step, "forward_count": self.forward_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.task_loss_ema = {str(key): float(value) for key, value in state.get("task_loss_ema", {}).items()}
        self.step = int(state.get("step", 0))
        self.forward_count = int(state.get("forward_count", 0))

    def _log(self, task_id: str, volatility: float, probability: float) -> None:
        if not self.config.runtime_path:
            return
        path = Path(self.config.runtime_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": self.step,
            "task_id": task_id,
            "volatility": volatility,
            "loss_ema": self.task_loss_ema.get(task_id, 0.0),
            "dropout_p": probability,
            "num_updated_lora_layers": len(self.dropouts),
            "forward_count": self.forward_count,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
