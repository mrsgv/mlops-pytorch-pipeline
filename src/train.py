"""Config-driven training entrypoint for the containerised workload.

Configuration precedence, lowest to highest:

1. ``configs/training_config.yaml`` baked into the image (fallback only).
2. The YAML mounted at ``/app/configs/training_config.yaml`` - a bind mount
   locally, a ConfigMap volume in Kubernetes.
3. ``CONFIG_PATH`` pointing anywhere else.
4. ``TRAIN_*`` / ``CHECKPOINT_DIR`` / ``MODEL_NAME`` environment variables.

Every metric line goes to stdout as a single JSON object so that
``kubectl logs job/pytorch-training`` is directly parseable.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from dataset import get_dataloaders
from model import CLASS_NAMES, count_parameters, get_model

CONFIG_SEARCH_PATH = (
    Path("/app/configs/training_config.yaml"),
    Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml",
    Path("configs/training_config.yaml"),
)

# env var -> (dotted config path, cast)
ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "TRAIN_ARCHITECTURE": ("model.architecture", str),
    "TRAIN_NUM_CLASSES": ("model.num_classes", int),
    "TRAIN_EPOCHS": ("training.epochs", int),
    "TRAIN_BATCH_SIZE": ("training.batch_size", int),
    "TRAIN_LEARNING_RATE": ("training.learning_rate", float),
    "TRAIN_WEIGHT_DECAY": ("training.weight_decay", float),
    "TRAIN_EARLY_STOPPING_PATIENCE": ("training.early_stopping_patience", int),
    "TRAIN_MAX_TRAIN_BATCHES": ("training.max_train_batches", int),
    "TRAIN_MAX_VAL_BATCHES": ("training.max_val_batches", int),
    "TRAIN_DEVICE": ("training.device", str),
    "TRAIN_SEED": ("training.seed", int),
    "TRAIN_DATASET": ("data.dataset", str),
    "TRAIN_DATA_DIR": ("data.data_dir", str),
    "TRAIN_NUM_WORKERS": ("data.num_workers", int),
    "TRAIN_DOWNLOAD": ("data.download", lambda v: v.lower() in {"1", "true", "yes"}),
    "CHECKPOINT_DIR": ("output.checkpoint_dir", str),
    "MODEL_NAME": ("output.model_name", str),
}

# Flipped by SIGTERM so a pre-empted Kubernetes pod stops on an epoch boundary
# instead of being killed mid-batch with no checkpoint written.
_stop_requested = False


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON line on stdout."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    print(json.dumps(record, default=str), flush=True)


def _handle_sigterm(signum: int, _frame: Any) -> None:
    global _stop_requested
    _stop_requested = True
    log_event("shutdown_requested", signal=signal.Signals(signum).name)


def resolve_config_path() -> Path:
    explicit = os.environ.get("CONFIG_PATH")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"CONFIG_PATH={explicit} does not exist")
        return path
    for candidate in CONFIG_SEARCH_PATH:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no training config found; set CONFIG_PATH or mount one at "
        "/app/configs/training_config.yaml"
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return config


def _set_nested(config: dict[str, Any], dotted: str, value: Any) -> None:
    section, _, key = dotted.partition(".")
    config.setdefault(section, {})[key] = value


def apply_env_overrides(config: dict[str, Any]) -> list[str]:
    """Override config values from the environment; return what changed."""
    applied: list[str] = []
    for env_var, (dotted, cast) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            _set_nested(config, dotted, cast(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_var}={raw!r} is not a valid value: {exc}") from exc
        applied.append(f"{env_var}->{dotted}={raw}")
    return applied


def cfg(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``section.key`` with a default, so a minimal ConfigMap still works."""
    section, _, key = dotted.partition(".")
    return (config.get(section) or {}).get(key, default)


def select_device(preference: str = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # local runs on Apple silicon
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if total == 0:
        raise RuntimeError("training loader produced no batches")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if total == 0:
        raise RuntimeError("validation loader produced no batches")
    return total_loss / total, correct / total


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write atomically.

    The serving Deployment reads this same PVC while training runs, so a
    half-written file would be picked up as a valid checkpoint. Writing to a
    sibling temp file and renaming makes the swap atomic within the volume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    config_path = resolve_config_path()
    config = load_config(config_path)
    overrides = apply_env_overrides(config)

    architecture = cfg(config, "model.architecture", "resnet18")
    num_classes = int(cfg(config, "model.num_classes", 10))
    pretrained = bool(cfg(config, "model.pretrained", False))

    epochs = int(cfg(config, "training.epochs", 10))
    batch_size = int(cfg(config, "training.batch_size", 64))
    learning_rate = float(cfg(config, "training.learning_rate", 1e-3))
    weight_decay = float(cfg(config, "training.weight_decay", 0.0))
    patience = int(cfg(config, "training.early_stopping_patience", 3))
    min_delta = float(cfg(config, "training.early_stopping_min_delta", 0.0))
    max_train_batches = int(cfg(config, "training.max_train_batches", 0))
    max_val_batches = int(cfg(config, "training.max_val_batches", 0))
    seed = int(cfg(config, "training.seed", 42))

    dataset_name = str(cfg(config, "data.dataset", "cifar10")).lower().replace("-", "_")
    data_dir = str(cfg(config, "data.data_dir", "/app/data"))
    num_workers = int(cfg(config, "data.num_workers", 2))
    download = bool(cfg(config, "data.download", True))

    checkpoint_dir = Path(str(cfg(config, "output.checkpoint_dir", "/app/checkpoints")))
    model_name = str(cfg(config, "output.model_name", "classifier_v1.pt"))

    set_seed(seed)
    device = select_device(str(cfg(config, "training.device", "auto")))

    log_event(
        "training_started",
        config_path=str(config_path),
        env_overrides=overrides,
        device=str(device),
        architecture=architecture,
        dataset=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        torch_version=torch.__version__,
    )

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
    ).to(device)
    log_event("model_built", architecture=architecture, parameters=count_parameters(model))

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=dataset_name,
        download=download,
    )
    log_event(
        "data_ready",
        data_dir=data_dir,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / model_name
    metrics_path = checkpoint_dir / "metrics.jsonl"

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_train_batches
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, max_val_batches)

        metrics = {
            "event": "epoch_metrics",
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "epoch_seconds": round(time.perf_counter() - epoch_start, 1),
        }
        log_event(**metrics)
        with open(metrics_path, "a") as handle:
            handle.write(json.dumps(metrics) + "\n")

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                checkpoint_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    # Metadata so serve.py can rebuild the model on its own.
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "dataset": dataset_name,
                    "class_names": list(CLASS_NAMES.get(dataset_name, ())),
                    # str(): torch.__version__ is a TorchVersion object, which
                    # torch.load(weights_only=True) refuses to unpickle.
                    "torch_version": str(torch.__version__),
                    "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            log_event("checkpoint_saved", path=str(checkpoint_path), epoch=epoch)
        else:
            patience_counter += 1
            log_event(
                "no_improvement",
                epoch=epoch,
                patience_counter=patience_counter,
                patience=patience,
            )
            if patience_counter >= patience:
                stopped_early = True
                log_event("early_stopping", epoch=epoch, best_epoch=best_epoch)
                break

        if _stop_requested:
            log_event("stopping_on_signal", epoch=epoch)
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_accuracy, 4),
        "early_stopped": stopped_early,
        "checkpoint": str(checkpoint_path),
        "total_seconds": round(time.perf_counter() - started_at, 1),
    }
    (checkpoint_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log_event("training_complete", **summary)

    if best_epoch == 0:
        log_event("training_failed", reason="no checkpoint was ever written")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
