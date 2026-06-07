"""PEFT/LoRA configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target: str | list[str] = "all"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LoraSettings":
        return cls(
            rank=int(config.get("rank", cls.rank)),
            alpha=int(config.get("alpha", cls.alpha)),
            dropout=float(config.get("dropout", cls.dropout)),
            target=config.get("target", cls.target),
            bias=str(config.get("bias", cls.bias)),
            task_type=str(config.get("task_type", cls.task_type)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_peft_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target,
            "bias": self.bias,
            "task_type": self.task_type,
        }


def summarize_lora(settings: LoraSettings) -> dict[str, Any]:
    scale = settings.alpha / settings.rank if settings.rank else 0.0
    summary = settings.to_dict()
    summary["scaling"] = scale
    return summary

