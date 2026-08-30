"""Dataset, transform and DataLoader construction.

CIFAR-10 is the default. Fashion-MNIST is supported as well and is upsampled
to 3x32x32 so a single model factory covers both datasets.

``get_inference_transform`` is deliberately exported: the serving container
must apply exactly the same normalisation as validation did, and duplicating
those constants in ``serve.py`` is how skew gets introduced.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# (mean, std) per dataset, in the channel order the model sees.
DATASET_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fashion_mnist": ((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
}

IMAGE_SIZE = 32


def _normalise(dataset: str) -> transforms.Normalize:
    try:
        mean, std = DATASET_STATS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset {dataset!r}; expected one of {', '.join(DATASET_STATS)}"
        ) from exc
    return transforms.Normalize(mean=mean, std=std)


def get_transforms(train: bool = True, dataset: str = "cifar10") -> transforms.Compose:
    """Augmented pipeline for training, deterministic pipeline otherwise."""
    steps: list[object] = []

    if dataset == "fashion_mnist":
        # 1x28x28 -> 3x32x32 so ``resnet18``/``simple_cnn`` need no changes.
        steps += [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMAGE_SIZE),
        ]

    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
        ]

    steps += [transforms.ToTensor(), _normalise(dataset)]
    return transforms.Compose(steps)


def get_inference_transform(dataset: str = "cifar10") -> transforms.Compose:
    """Validation-time preprocessing, reused verbatim by the serving app."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            _normalise(dataset),
        ]
    )


def _build_split(
    dataset: str,
    data_dir: str,
    train: bool,
    download: bool,
) -> Dataset:
    factory = {"cifar10": datasets.CIFAR10, "fashion_mnist": datasets.FashionMNIST}[dataset]
    return factory(
        root=data_dir,
        train=train,
        download=download,
        transform=get_transforms(train=train, dataset=dataset),
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset: str = "cifar10",
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``.

    The CIFAR-10 test split is used for validation - it is what early stopping
    watches, and this assignment has no separate held-out test set.
    """
    dataset = dataset.strip().lower().replace("-", "_")
    if dataset not in DATASET_STATS:
        raise ValueError(f"unsupported dataset {dataset!r}")

    train_dataset = _build_split(dataset, data_dir, train=True, download=download)
    val_dataset = _build_split(dataset, data_dir, train=False, download=download)

    # ``persistent_workers`` only makes sense when workers actually exist.
    # ``pin_memory`` only helps CUDA: asking for it on MPS (local Apple silicon
    # runs) or on a CPU-only container makes torch emit a UserWarning per loader,
    # which would break the JSON-lines contract of the training logs.
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": num_workers > 0 and torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
