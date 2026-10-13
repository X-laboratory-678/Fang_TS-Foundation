from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .adaptation import adapt_lora_parameters, clone_parameters, lora_parameters, restore_parameters
from .config import TELAConfig


@dataclass(frozen=True)
class ReptileStepResult:
    support_tasks: int
    query_loss_before: float
    query_loss_after: float
    parameter_delta_norm: float


class ReptileLoRAEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TELAConfig,
        loss_fn: Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor],
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.meta_steps = 0
        self.parameters = lora_parameters(model)

    def _mean_query_loss(self, query_batches: list[dict[str, Any]]) -> float:
        if not query_batches:
            raise ValueError("TELA query batches cannot be empty.")
        self.model.eval()
        with torch.no_grad():
            losses = [float(self.loss_fn(self.model, batch).detach().cpu()) for batch in query_batches]
        return sum(losses) / len(losses)

    def meta_step(self, episodes: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]) -> ReptileStepResult:
        if len(episodes) != self.config.tasks_per_meta_batch:
            raise ValueError(
                f"TELA expected {self.config.tasks_per_meta_batch} task episodes, received {len(episodes)}."
            )
        meta_state = clone_parameters(self.parameters)
        adapted_states: list[dict[str, torch.Tensor]] = []
        before_losses: list[float] = []
        after_losses: list[float] = []

        for support_batches, query_batches in episodes:
            if not support_batches or not query_batches:
                raise ValueError("Each TELA episode requires non-empty support and query batches.")
            restore_parameters(self.parameters, meta_state)
            before_losses.append(self._mean_query_loss(query_batches))
            adapted_states.append(
                adapt_lora_parameters(
                    self.model,
                    support_batches,
                    self.loss_fn,
                    self.config.inner_steps,
                    self.config.inner_lr,
                )
            )
            after_losses.append(self._mean_query_loss(query_batches))

        restore_parameters(self.parameters, meta_state)
        squared_norm = 0.0
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                mean_adapted = torch.stack([state[name].to(parameter.device) for state in adapted_states]).mean(dim=0)
                delta = mean_adapted - meta_state[name].to(parameter.device)
                parameter.add_(self.config.meta_step_size * delta)
                squared_norm += float(torch.sum(delta.float() ** 2).cpu())
        self.meta_steps += 1
        return ReptileStepResult(
            support_tasks=len(episodes),
            query_loss_before=sum(before_losses) / len(before_losses),
            query_loss_after=sum(after_losses) / len(after_losses),
            parameter_delta_norm=squared_norm**0.5,
        )

