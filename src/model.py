"""Model definitions and factory for the MLOps PyTorch pipeline.

Two architectures are available:

* ``simple_cnn`` - a small VGG-style CNN written from scratch. It converges in
  a couple of epochs and is cheap enough for a CPU-only Kubernetes node, so it
  is the architecture used for smoke runs.
* ``resnet18``   - ``torchvision.models.resnet18`` adapted for 32x32 inputs
  (3x3 stem, no max-pool) because the ImageNet stem throws away too much
  spatial information on CIFAR-sized images.

Both take ``(N, 3, 32, 32)`` tensors and return ``num_classes`` logits.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

FASHION_MNIST_CLASSES: tuple[str, ...] = (
    "t_shirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
)

CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "cifar10": CIFAR10_CLASSES,
    "fashion_mnist": FASHION_MNIST_CLASSES,
}

SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("simple_cnn", "resnet18")


class SimpleCNN(nn.Module):
    """Three conv blocks (32 -> 16 -> 8 -> 4 spatial) then a linear head."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *self._block(in_channels, 32),
            *self._block(32, 64),
            *self._block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> list[nn.Module]:
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _resnet18_for_small_images(num_classes: int, pretrained: bool) -> nn.Module:
    """ResNet-18 with a CIFAR-friendly stem.

    The stock stem (7x7 stride-2 conv + 3x3 stride-2 max-pool) downsamples a
    32x32 image to 8x8 before the first residual block, which costs several
    accuracy points. Swapping in a 3x3 stride-1 conv and dropping the max-pool
    keeps the first stage at full 32x32 resolution.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """Build a model by name.

    ``pretrained`` downloads ImageNet weights, so it defaults to ``False`` -
    containers in the cluster have no reason to reach out to the internet.
    """
    key = architecture.strip().lower().replace("-", "_")
    if key == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if key == "resnet18":
        return _resnet18_for_small_images(num_classes, pretrained)
    raise ValueError(
        f"unsupported architecture {architecture!r}; "
        f"expected one of {', '.join(SUPPORTED_ARCHITECTURES)}"
    )


def model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    """Rebuild the trained model straight from a checkpoint dict.

    ``train.py`` stores the architecture and class count next to the weights so
    the serving container never needs the training config.
    """
    try:
        architecture = checkpoint["architecture"]
        num_classes = checkpoint["num_classes"]
        state_dict = checkpoint["model_state_dict"]
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing required key {exc.args[0]!r}") from exc

    model = get_model(architecture=architecture, num_classes=int(num_classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
