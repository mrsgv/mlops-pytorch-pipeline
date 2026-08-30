"""Unit tests for the model factory and the checkpoint contract."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from model import (
    CIFAR10_CLASSES,
    SUPPORTED_ARCHITECTURES,
    SimpleCNN,
    count_parameters,
    get_model,
    model_from_checkpoint,
)

BATCH = 4
INPUT_SHAPE = (BATCH, 3, 32, 32)


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_forward_pass_shape(architecture: str) -> None:
    model = get_model(architecture, num_classes=10).eval()
    with torch.no_grad():
        logits = model(torch.randn(*INPUT_SHAPE))
    assert logits.shape == (BATCH, 10)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("num_classes", [2, 10, 100])
def test_num_classes_is_respected(num_classes: int) -> None:
    model = get_model("simple_cnn", num_classes=num_classes).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, num_classes)


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="unsupported architecture"):
        get_model("vgg99")


def test_resnet18_stem_is_adapted_for_32x32() -> None:
    """The ImageNet stem would downsample 32x32 to 8x8 before block 1."""
    model = get_model("resnet18", num_classes=10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.fc.out_features == 10


def test_simple_cnn_is_much_smaller_than_resnet18() -> None:
    small = count_parameters(get_model("simple_cnn"))
    large = count_parameters(get_model("resnet18"))
    assert 0 < small < large


def test_architecture_name_is_normalised() -> None:
    assert isinstance(get_model("Simple-CNN"), SimpleCNN)


def test_forward_is_deterministic_under_a_fixed_seed() -> None:
    torch.manual_seed(0)
    first = get_model("simple_cnn").eval()
    torch.manual_seed(0)
    second = get_model("simple_cnn").eval()
    sample = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(first(sample), second(sample))


def test_model_from_checkpoint_round_trip(checkpoint_file) -> None:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    model = model_from_checkpoint(checkpoint)

    assert not model.training  # returned ready for inference
    assert checkpoint["class_names"] == list(CIFAR10_CLASSES)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 10)


def test_model_from_checkpoint_rejects_incomplete_payload() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        model_from_checkpoint({"model_state_dict": {}})
