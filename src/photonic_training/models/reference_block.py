"""A compact 32×32 voltage-mapped reference model.

It demonstrates the model interface used by the public training loop without
releasing task-specific network wiring or data-processing pipelines.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..device import Chip


class VoltageMappedClassifier(nn.Module):
    """Map vectors through one or more 32×32 CUMEC blocks then classify them."""

    def __init__(self, classes: int = 10, blocks: int = 2) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be at least one.")
        self.chip = Chip()
        # Match the public task models: raw electrical controls are initialized
        # in the 0–7 range and converted to phase by the CUMEC device model.
        self.voltages = nn.Parameter(7.0 * torch.rand(blocks, 1, self.chip.vol_num))
        self.head = nn.Linear(32, classes)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] != 32:
            raise ValueError("Expected inputs with shape [batch, 32].")
        # ``Chip`` stores its own device attribute for allocating intermediate
        # transfer matrices.  PyTorch's parent ``model.to(...)`` moves the
        # tensors recursively but does not update that custom attribute, so
        # synchronize it explicitly before calling ``set_s``.
        if self.chip.device != inputs.device:
            self.chip.to(inputs.device)
        matrices = self.chip.set_s(self.voltages, mode="nophase")[0, :, 0]
        features = inputs
        for matrix in matrices:
            features = torch.relu(features @ matrix.T)
        return self.head(features), self.voltages
