"""Train the public CIFAR-10 example with baseline, L1, L2 or SM settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from photonic_training import RegularizationConfig, VoltageRegularizer, train_one_epoch
from photonic_training.regularizers import build_adamw
from photonic_training.examples.cifar10.model import Cifar


PRESETS = {
    # Matches the baseline notebook: AdamW weight_decay = 0.01.
    "baseline": RegularizationConfig(l2_weight_decay=1e-2),
    "l1": RegularizationConfig(l1=1e-4),
    "l2": RegularizationConfig(l2_weight_decay=1e-2),
    "sm": RegularizationConfig(sm=1e-3),
    "combined": RegularizationConfig(l1=1e-4, l2_weight_decay=1e-4, sm=1e-6),
}
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def limited(dataset: torch.utils.data.Dataset, size: int) -> torch.utils.data.Dataset:
    """Use the leading ``size`` samples for a short software smoke test."""

    return dataset if size <= 0 else Subset(dataset, range(min(size, len(dataset))))


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        logits, _ = model(inputs.to(device))
        labels = labels.to(device)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="baseline")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--subset-train", type=int, default=0)
    parser.add_argument("--subset-test", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # Same fixed 40,000 / 10,000 train-validation split as the source notebook.
    train_set_aug = datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=args.download)
    train_set_eval = datasets.CIFAR10(args.data_dir, train=True, transform=test_transform, download=args.download)
    test_set = datasets.CIFAR10(args.data_dir, train=False, transform=test_transform, download=args.download)
    split_generator = torch.Generator().manual_seed(1024)
    indices = torch.randperm(len(train_set_aug), generator=split_generator)
    train_set = Subset(train_set_aug, indices[:40000])
    val_set = Subset(train_set_eval, indices[40000:50000])
    train_loader = DataLoader(limited(train_set, args.subset_train), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(limited(val_set, args.subset_test), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(limited(test_set, args.subset_test), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    config = PRESETS[args.preset]
    model = Cifar().to(device)
    optimizer = build_adamw(model.parameters(), learning_rate=args.learning_rate, config=config)
    regularizer = VoltageRegularizer(config)
    task_loss = torch.nn.CrossEntropyLoss()
    print(f"device={device} preset={args.preset} train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    for epoch in range(args.epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, task_loss, regularizer, device)
        val_accuracy = accuracy(model, val_loader, device)
        test_accuracy = accuracy(model, test_loader, device)
        print(f"epoch={epoch + 1} metrics={metrics} val_accuracy={val_accuracy:.2f}% test_accuracy={test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
