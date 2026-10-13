from collections.abc import Callable
from typing import Any

import torch


def lora_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora" in name.lower()
    }
    if not parameters:
        raise ValueError("TELA requires trainable LoRA parameters; none were found.")
    non_lora_trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora" not in name.lower()
    ]
    if non_lora_trainable:
        raise ValueError(f"TELA requires lora_only adaptation; non-LoRA trainable parameters: {non_lora_trainable[:5]}")
    return parameters


def clone_parameters(parameters: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in parameters.items()}


def restore_parameters(parameters: dict[str, torch.nn.Parameter], state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])


def adapt_lora_parameters(
    model: torch.nn.Module,
    support_batches: list[dict[str, Any]],
    loss_fn: Callable[[torch.nn.Module, dict[str, Any]], torch.Tensor],
    inner_steps: int,
    inner_lr: float,
) -> dict[str, torch.Tensor]:
    parameters = lora_parameters(model)
    optimizer = torch.optim.SGD(parameters.values(), lr=inner_lr)
    model.train()
    for step in range(inner_steps):
        batch = support_batches[step % len(support_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("TELA support loss is NaN or Inf.")
        loss.backward()
        optimizer.step()
    return clone_parameters(parameters)

