"""CIFAR-10 voltage-mapped network used by the public task example."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from photonic_training.device import Chip


class Cifar(nn.Module):
    """Four-stage CIFAR-10 classifier mapped onto 32×32 CUMEC blocks.

    The model exposes its voltage tensor as the second output so that the
    shared baseline/L1/L2/SM training loop can be used unchanged.
    """

    def __init__(self, num_classes: int = 10, chip_size: int = 32) -> None:
        super().__init__()
        self.chip_size = chip_size
        self.chip = Chip()
        # Match ``code for train and val/cifar/model.py``: raw electrical
        # controls are initialized in the 0–7 range.
        self.weight = nn.Parameter(7.0 * torch.rand(127, 1, self.chip.vol_num))
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _matrices(self, device: torch.device) -> Tensor:
        # ``Chip`` uses a custom device field when allocating intermediate
        # transfer matrices; update it explicitly after ``model.to(device)``.
        if self.chip.device != device:
            self.chip.to(device)
        return self.chip.set_s(self.weight, mode="nophase")[0, :, 0]

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        matrices = self._matrices(inputs.device)
        batch, _, height, width = inputs.shape
        index = 0

        # 3 -> 32, 3×3: one 32×32 block.
        unfolded = F.unfold(F.pad(inputs, (1, 1, 1, 1)), kernel_size=3, stride=1)
        padded = F.pad(unfolded, (0, 0, 0, 5)).view(batch, 1, 32, height * width)
        weights = matrices[index : index + 1].view(1, 1, 32, 32)
        index += 1
        x = torch.einsum("oijk,bikn->bojn", weights, padded).reshape(batch, 32, height, width)
        x = F.relu(self.bn1(x))

        # 32 -> 64, 3×3: two output groups × nine input groups.
        unfolded = F.unfold(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=1)
        weights = matrices[index : index + 18].view(2, 9, 32, 32)
        index += 18
        x = torch.einsum("oijk,bikn->bojn", weights, unfolded.view(batch, 9, 32, height * width))
        x = F.relu(self.bn2(x.reshape(batch, 64, height, width)))
        x = self.pool1(x)

        # 64 -> 64, 3×3: two output groups × eighteen input groups.
        batch, _, height, width = x.shape
        unfolded = F.unfold(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=1)
        weights = matrices[index : index + 36].view(2, 18, 32, 32)
        index += 36
        x = torch.einsum("oijk,bikn->bojn", weights, unfolded.view(batch, 18, 32, height * width))
        x = F.relu(self.bn3(x.reshape(batch, 64, height, width)))

        # 64 -> 128, 3×3: four output groups × eighteen input groups.
        unfolded = F.unfold(F.pad(x, (1, 1, 1, 1)), kernel_size=3, stride=1)
        weights = matrices[index : index + 72].view(4, 18, 32, 32)
        x = torch.einsum("oijk,bikn->bojn", weights, unfolded.view(batch, 18, 32, height * width))
        x = F.relu(self.bn4(x.reshape(batch, 128, height, width)))
        x = self.pool2(x)
        x = self.dropout(x).reshape(batch, -1)
        return self.fc(x), self.weight


# Compatibility alias for earlier public-package revisions.
CIFARVoltageNet = Cifar
