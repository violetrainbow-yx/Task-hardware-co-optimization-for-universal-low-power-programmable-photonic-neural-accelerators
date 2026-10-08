"""Runnable smoke-test for baseline, L1, L2 and SM training modes.

Replace the synthetic loader and reference model with a task-specific loader
and a model that returns ``(prediction, voltages)``.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.photonic_training import RegularizationConfig, VoltageRegularizer, train_one_epoch
from src.photonic_training.models import VoltageMappedClassifier
from src.photonic_training.regularizers import build_adamw


PRESETS = {
    "baseline": RegularizationConfig(),
    "l1": RegularizationConfig(l1=1e-6),
    "l2": RegularizationConfig(l2_weight_decay=1e-2),
    "sm": RegularizationConfig(sm=1e-6),
    "combined": RegularizationConfig(l1=1e-6, l2_weight_decay=1e-2, sm=1e-6),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="sm")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    inputs = torch.randn(32, 32)
    labels = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=8, shuffle=True)
    model = VoltageMappedClassifier().to(device)
    config = PRESETS[args.preset]
    optimizer = build_adamw(model.parameters(), learning_rate=1e-3, config=config)
    regularizer = VoltageRegularizer(config)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, loader, optimizer, criterion, regularizer, device)
        print(f"epoch={epoch + 1} preset={args.preset} {metrics}")


if __name__ == "__main__":
    main()
