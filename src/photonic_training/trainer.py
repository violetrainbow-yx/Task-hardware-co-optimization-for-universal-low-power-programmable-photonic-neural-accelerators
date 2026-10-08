"""A small, task-agnostic training loop for baseline and constrained runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from .regularizers import VoltageRegularizer


def split_model_output(output: Any) -> tuple[Tensor, Tensor | list[Tensor]]:
    """Require models to return ``(prediction, voltages)`` during training."""

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("The public trainer expects model(inputs) -> (prediction, voltages).")
    prediction, voltages = output
    if not isinstance(prediction, Tensor):
        raise TypeError("The first model output must be a torch.Tensor.")
    return prediction, voltages


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    task_loss: Callable[[Tensor, Tensor], Tensor],
    regularizer: VoltageRegularizer,
    device: torch.device | str,
) -> dict[str, float]:
    """Train one epoch and return separately logged task, L1, SM and total loss."""

    model.train()
    sums = {"task": 0.0, "l1": 0.0, "sm": 0.0, "total": 0.0}
    batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction, voltages = split_model_output(model(inputs))
        task = task_loss(prediction, targets)
        penalties = regularizer(voltages)
        total = task + penalties["weighted"]
        total.backward()
        optimizer.step()
        batches += 1
        sums["task"] += task.detach().item()
        sums["l1"] += penalties["l1"].detach().item()
        sums["sm"] += penalties["sm"].detach().item()
        sums["total"] += total.detach().item()
    if batches == 0:
        raise ValueError("The training loader is empty.")
    return {name: value / batches for name, value in sums.items()}
