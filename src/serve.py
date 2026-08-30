"""FastAPI inference service for the checkpoint written by ``train.py``.

Endpoints
---------
``GET  /health``   200 once a checkpoint is loaded, 503 until then. Both the
                   Kubernetes liveness/readiness probes and the Docker
                   HEALTHCHECK point here.
``POST /predict``  multipart upload (field ``image``) -> class probabilities.
``POST /reload``   re-reads the checkpoint from disk, so a finished training
                   Job can be picked up without restarting the pods.
``GET  /``         service metadata.

Configuration is entirely environment-driven, which is what lets the same
image run under ``docker run`` and as a Kubernetes Deployment with a ConfigMap
and an optional Secret supplying the values.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from dataset import get_inference_transform
from model import CLASS_NAMES, model_from_checkpoint

CHECKPOINT_PATH = Path(
    os.environ.get(
        "CHECKPOINT_PATH",
        str(
            Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
            / os.environ.get("MODEL_NAME", "classifier_v1.pt")
        ),
    )
)
API_KEY = os.environ.get("MODEL_API_KEY", "").strip()
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
SERVE_HOST = os.environ.get("SERVE_HOST", "0.0.0.0")  # noqa: S104 - container port
SERVE_PORT = int(os.environ.get("SERVE_PORT", "8080"))

if os.environ.get("TORCH_NUM_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))


def log_event(event: str, **fields: Any) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    print(json.dumps(record, default=str), flush=True)


@dataclass
class ModelState:
    """Everything the request path needs, swapped in as one object."""

    model: torch.nn.Module | None = None
    class_names: list[str] = field(default_factory=list)
    dataset: str = "cifar10"
    transform: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: str | None = None
    load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


STATE = ModelState()


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint, preferring the safe ``weights_only`` path."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints, or non-tensor objects in the payload
        log_event("checkpoint_weights_only_failed", path=str(path))
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path: Path = CHECKPOINT_PATH) -> ModelState:
    """(Re)load the checkpoint into ``STATE``; never raises."""
    global STATE
    if not path.is_file():
        STATE = ModelState(load_error=f"checkpoint not found at {path}")
        log_event("model_load_failed", reason=STATE.load_error)
        return STATE

    try:
        checkpoint = load_checkpoint(path)
        model = model_from_checkpoint(checkpoint)
        dataset = str(checkpoint.get("dataset", "cifar10"))
        num_classes = int(checkpoint["num_classes"])
        class_names = list(checkpoint.get("class_names") or CLASS_NAMES.get(dataset, ()))
        if len(class_names) != num_classes:
            # Mismatched metadata would silently mislabel predictions.
            class_names = [str(i) for i in range(num_classes)]

        STATE = ModelState(
            model=model,
            class_names=class_names,
            dataset=dataset,
            transform=get_inference_transform(dataset),
            metadata={
                "architecture": checkpoint.get("architecture"),
                "num_classes": checkpoint.get("num_classes"),
                "dataset": dataset,
                "trained_epoch": checkpoint.get("epoch"),
                "val_loss": checkpoint.get("val_loss"),
                "val_accuracy": checkpoint.get("val_accuracy"),
                "checkpoint": str(path),
                "trained_at": checkpoint.get("saved_at"),
            },
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        log_event("model_loaded", **STATE.metadata)
    except Exception as exc:  # a corrupt or half-written file must not crash the pod
        STATE = ModelState(load_error=f"{type(exc).__name__}: {exc}")
        log_event("model_load_failed", reason=STATE.load_error, path=str(path))
    return STATE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log_event("service_starting", checkpoint=str(CHECKPOINT_PATH), api_key_required=bool(API_KEY))
    load_model()
    yield
    log_event("service_stopping")


app = FastAPI(
    title="MLOps PyTorch image classifier",
    version="1.0.0",
    description="Serves the checkpoint produced by the training Job.",
    lifespan=lifespan,
)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op unless ``MODEL_API_KEY`` is supplied (via a Kubernetes Secret)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key"
        )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "mlops-pytorch-pipeline/serve",
        "version": app.version,
        "model_loaded": STATE.is_loaded,
        "endpoints": ["GET /health", "POST /predict", "POST /reload"],
    }


@app.get("/health")
async def health() -> JSONResponse:
    """200 only when a model is in memory - probes depend on that contract."""
    if not STATE.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "model_loaded": False,
                "checkpoint": str(CHECKPOINT_PATH),
                "reason": STATE.load_error,
            },
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "model_loaded": True,
            "loaded_at": STATE.loaded_at,
            "model": STATE.metadata,
        },
    )


@app.post("/reload", dependencies=[Depends(require_api_key)])
async def reload_model() -> JSONResponse:
    state = load_model()
    if not state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=state.load_error
        )
    return JSONResponse(content={"status": "reloaded", "model": state.metadata})


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(
    image: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Classify one uploaded image.

    ``image`` is the documented field name; ``file`` is accepted as an alias
    because that is what most HTTP clients default to.
    """
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="send the image as multipart field 'image'",
        )

    if not STATE.is_loaded:
        # The checkpoint may have appeared after startup (training Job finished).
        load_model()
        if not STATE.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=STATE.load_error or "model not loaded",
            )

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"could not decode {upload.filename!r} as an image: {exc}",
        ) from exc

    started = time.perf_counter()
    tensor = STATE.transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = F.softmax(STATE.model(tensor), dim=1)[0]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    scores = {
        name: round(float(p), 6) for name, p in zip(STATE.class_names, probabilities, strict=True)
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_class, best_score = ranked[0]

    log_event(
        "prediction",
        filename=upload.filename,
        predicted_class=best_class,
        confidence=best_score,
        inference_ms=elapsed_ms,
    )
    return {
        "filename": upload.filename,
        "predicted_class": best_class,
        "predicted_index": STATE.class_names.index(best_class),
        "confidence": best_score,
        "probabilities": scores,
        "top_k": [{"class": name, "probability": score} for name, score in ranked[:TOP_K]],
        "inference_ms": elapsed_ms,
        "model": STATE.metadata,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=SERVE_HOST,
        port=SERVE_PORT,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=os.environ.get("ACCESS_LOG", "false").lower() == "true",
    )
    sys.exit(0)
