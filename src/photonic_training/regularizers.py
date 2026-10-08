"""Task-aware L1, L2 and smoothness regularization for voltage tensors.

The baseline is obtained by setting all three coefficients to zero.  L2 is
implemented through AdamW weight decay, matching the training convention used
in this work.  L1 and SM are explicit terms in the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RegularizationConfig:
    """Coefficients for the public task-hardware co-optimization objective.

    ``l2_weight_decay`` belongs in ``torch.optim.AdamW`` rather than in the
    scalar loss.  This prevents accidental double application of L2.
    """

    l1: float = 0.0
    l2_weight_decay: float = 0.0
    sm: float = 0.0

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")


def _as_voltage_list(voltages: Tensor | Sequence[Tensor]) -> list[Tensor]:
    if isinstance(voltages, Tensor):
        return [voltages]
    values = list(voltages)
    if not values or not all(isinstance(value, Tensor) for value in values):
        raise TypeError("voltages must be a Tensor or a non-empty sequence of Tensors.")
    return values


class VoltageRegularizer(nn.Module):
    """Compute the explicit L1 and adjacent-block SM penalties.

    For a voltage tensor shaped ``[n_blocks, 1, n_phase_shifters]``, SM is
    ``||V[1:] - V[:-1]||_2``.  The formulation also supports a list of voltage
    tensors, in which case L1 and SM contributions are summed over the list.
    """

    def __init__(self, config: RegularizationConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

    @staticmethod
    def l1_penalty(voltages: Tensor | Sequence[Tensor]) -> Tensor:
        values = _as_voltage_list(voltages)
        return torch.stack([value.abs().sum() for value in values]).sum()

    @staticmethod
    def smoothness_penalty(voltages: Tensor | Sequence[Tensor]) -> Tensor:
        values = _as_voltage_list(voltages)
        penalties: list[Tensor] = []
        for value in values:
            if value.ndim < 1:
                raise ValueError("Each voltage tensor must have a block dimension.")
            if value.shape[0] < 2:
                penalties.append(value.new_zeros(()))
            else:
                penalties.append(torch.linalg.vector_norm(torch.diff(value, dim=0), ord=2))
        return torch.stack(penalties).sum()

    def forward(self, voltages: Tensor | Sequence[Tensor]) -> dict[str, Tensor]:
        l1 = self.l1_penalty(voltages)
        sm = self.smoothness_penalty(voltages)
        weighted = self.config.l1 * l1 + self.config.sm * sm
        return {"l1": l1, "sm": sm, "weighted": weighted}


def build_adamw(
    parameters: Iterable[nn.Parameter], *, learning_rate: float, config: RegularizationConfig
) -> torch.optim.AdamW:
    """Create the optimizer with decoupled L2 regularization exactly once."""

    config.validate()
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=config.l2_weight_decay)
