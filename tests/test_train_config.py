"""Tests for the configuration layer of ``train.py``.

These cover the part of the training script that the container contract depends
on - config discovery, ConfigMap/env precedence, tolerating a minimal config,
and atomic checkpoint writes - without touching the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import train

MINIMAL_CONFIG = {
    "model": {"architecture": "resnet18", "num_classes": 10},
    "training": {"epochs": 10, "batch_size": 64, "learning_rate": 0.001},
    "data": {"dataset": "cifar10", "data_dir": "/app/data"},
    "output": {"checkpoint_dir": "/app/checkpoints", "model_name": "classifier_v1.pt"},
}


def write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_config_path_env_var_wins(tmp_path, monkeypatch) -> None:
    path = write_config(tmp_path, MINIMAL_CONFIG)
    monkeypatch.setenv("CONFIG_PATH", str(path))
    assert train.resolve_config_path() == path


def test_missing_config_path_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_PATH", "/nope/training_config.yaml")
    with pytest.raises(FileNotFoundError):
        train.resolve_config_path()


def test_falls_back_to_the_config_baked_into_the_image(monkeypatch, repo_root) -> None:
    """With no CONFIG_PATH and no mount, configs/training_config.yaml is used."""
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    resolved = train.resolve_config_path()
    assert resolved.is_file()
    assert resolved.name == "training_config.yaml"
    # Locally that is the repo copy; in the container the /app mount wins.
    assert resolved == repo_root / "configs" / "training_config.yaml"


def test_env_overrides_beat_the_yaml(tmp_path, monkeypatch) -> None:
    config = train.load_config(write_config(tmp_path, MINIMAL_CONFIG))
    monkeypatch.setenv("TRAIN_EPOCHS", "2")
    monkeypatch.setenv("TRAIN_MAX_TRAIN_BATCHES", "5")
    monkeypatch.setenv("CHECKPOINT_DIR", "/tmp/ckpt")

    applied = train.apply_env_overrides(config)

    assert config["training"]["epochs"] == 2
    assert config["training"]["max_train_batches"] == 5
    assert config["output"]["checkpoint_dir"] == "/tmp/ckpt"
    assert config["training"]["batch_size"] == 64  # untouched
    assert any("TRAIN_EPOCHS" in entry for entry in applied)


def test_a_malformed_override_is_rejected_loudly(tmp_path, monkeypatch) -> None:
    config = train.load_config(write_config(tmp_path, MINIMAL_CONFIG))
    monkeypatch.setenv("TRAIN_EPOCHS", "many")
    with pytest.raises(ValueError, match="TRAIN_EPOCHS"):
        train.apply_env_overrides(config)


def test_optional_keys_fall_back_to_defaults() -> None:
    """The ConfigMap in the assignment brief omits these; training must still run."""
    assert train.cfg(MINIMAL_CONFIG, "training.weight_decay", 0.0) == 0.0
    assert train.cfg(MINIMAL_CONFIG, "training.max_train_batches", 0) == 0
    assert train.cfg(MINIMAL_CONFIG, "data.num_workers", 2) == 2
    assert train.cfg({}, "model.architecture", "resnet18") == "resnet18"


def test_explicit_device_preference_is_honoured() -> None:
    assert train.select_device("cpu").type == "cpu"
    assert train.select_device("auto").type in {"cpu", "cuda", "mps"}


def test_checkpoint_write_is_atomic(tmp_path) -> None:
    """No .tmp file survives, so serving never sees a partial checkpoint."""
    path = tmp_path / "nested" / "classifier_v1.pt"
    train.save_checkpoint(path, {"epoch": 1, "model_state_dict": {}})

    assert path.is_file()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert torch.load(path, map_location="cpu", weights_only=True)["epoch"] == 1


def test_log_event_emits_one_json_line(capsys) -> None:
    train.log_event("epoch_metrics", epoch=1, val_loss=0.5)
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    record = json.loads(captured[0])
    assert record["event"] == "epoch_metrics"
    assert record["epoch"] == 1
    assert "ts" in record


def test_one_short_epoch_trains_and_checkpoints(tmp_path, monkeypatch) -> None:
    """End-to-end main() on 8 synthetic images - no dataset download."""
    from torch.utils.data import DataLoader, TensorDataset

    def fake_dataloaders(**_kwargs):
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        return loader, loader

    monkeypatch.setattr(train, "get_dataloaders", fake_dataloaders)

    config = dict(MINIMAL_CONFIG)
    config["model"] = {"architecture": "simple_cnn", "num_classes": 10}
    config["training"] = {"epochs": 1, "batch_size": 4, "learning_rate": 0.001}
    config["output"] = {"checkpoint_dir": str(tmp_path), "model_name": "classifier_v1.pt"}
    monkeypatch.setenv("CONFIG_PATH", str(write_config(tmp_path, config)))
    monkeypatch.setenv("TRAIN_DEVICE", "cpu")

    assert train.main() == 0

    checkpoint = torch.load(tmp_path / "classifier_v1.pt", map_location="cpu", weights_only=True)
    assert checkpoint["architecture"] == "simple_cnn"
    assert checkpoint["class_names"][0] == "airplane"
    assert (tmp_path / "metrics.jsonl").is_file()

    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["best_epoch"] == 1
