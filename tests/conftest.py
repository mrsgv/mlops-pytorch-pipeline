"""Shared fixtures: put ``src/`` on the import path and mint a fake checkpoint.

``src/`` is a flat module directory rather than a package because both
containers run their entrypoint as a script (``python src/train.py``), which
puts that directory on ``sys.path`` automatically. Tests replicate that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    """A real, loadable checkpoint from an untrained ``simple_cnn``."""
    import torch

    from model import CIFAR10_CLASSES, get_model

    model = get_model("simple_cnn", num_classes=10)
    path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 1.234,
            "val_accuracy": 0.5678,
            "architecture": "simple_cnn",
            "num_classes": 10,
            "dataset": "cifar10",
            "class_names": list(CIFAR10_CLASSES),
            "saved_at": "2026-01-01T00:00:00+00:00",
        },
        path,
    )
    return path


@pytest.fixture
def png_bytes() -> bytes:
    """A 32x32 RGB PNG, in memory."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 90, 200)).save(buffer, format="PNG")
    return buffer.getvalue()
