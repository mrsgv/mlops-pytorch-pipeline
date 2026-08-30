"""API tests for the serving app: the /health contract, /predict, auth.

``serve`` reads its configuration at import time, so each test reloads the
module with the environment it needs.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _load_serve(checkpoint: Path, monkeypatch: pytest.MonkeyPatch, **env: str):
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("serve", None)
    return importlib.import_module("serve")


def test_health_is_503_until_a_checkpoint_exists(tmp_path, monkeypatch) -> None:
    serve = _load_serve(tmp_path / "missing.pt", monkeypatch)
    with TestClient(serve.app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_health_is_200_once_the_model_loads(checkpoint_file, monkeypatch) -> None:
    serve = _load_serve(checkpoint_file, monkeypatch)
    with TestClient(serve.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model"]["architecture"] == "simple_cnn"
    assert body["model"]["val_accuracy"] == pytest.approx(0.5678)


def test_predict_returns_a_probability_distribution(
    checkpoint_file, png_bytes, monkeypatch
) -> None:
    serve = _load_serve(checkpoint_file, monkeypatch)
    with TestClient(serve.app) as client:
        response = client.post(
            "/predict", files={"image": ("test_image.png", png_bytes, "image/png")}
        )

    assert response.status_code == 200
    body = response.json()
    probabilities = body["probabilities"]
    assert len(probabilities) == 10
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert body["predicted_class"] in probabilities
    assert body["confidence"] == pytest.approx(max(probabilities.values()))
    assert len(body["top_k"]) == 3


def test_predict_accepts_a_non_32x32_image(checkpoint_file, monkeypatch) -> None:
    """Uploads are resized, so an arbitrary photo size still works."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), color=(10, 200, 30)).save(buffer, format="JPEG")

    serve = _load_serve(checkpoint_file, monkeypatch)
    with TestClient(serve.app) as client:
        response = client.post(
            "/predict", files={"image": ("photo.jpg", buffer.getvalue(), "image/jpeg")}
        )
    assert response.status_code == 200


def test_predict_without_a_file_is_400(checkpoint_file, monkeypatch) -> None:
    serve = _load_serve(checkpoint_file, monkeypatch)
    with TestClient(serve.app) as client:
        response = client.post("/predict", data={"nothing": "here"})
    assert response.status_code == 400


def test_predict_rejects_a_non_image_upload(checkpoint_file, monkeypatch) -> None:
    serve = _load_serve(checkpoint_file, monkeypatch)
    with TestClient(serve.app) as client:
        response = client.post(
            "/predict", files={"image": ("notes.txt", b"not an image", "text/plain")}
        )
    assert response.status_code == 400


def test_api_key_is_enforced_when_the_secret_is_present(
    checkpoint_file, png_bytes, monkeypatch
) -> None:
    serve = _load_serve(checkpoint_file, monkeypatch, MODEL_API_KEY="s3cret")
    files = {"image": ("test_image.png", png_bytes, "image/png")}
    with TestClient(serve.app) as client:
        assert client.post("/predict", files=files).status_code == 401
        authorised = client.post("/predict", files=files, headers={"X-API-Key": "s3cret"})
    assert authorised.status_code == 200


def test_reload_picks_up_a_checkpoint_written_after_startup(
    tmp_path, checkpoint_file, monkeypatch
) -> None:
    """Serving can start before the training Job finishes."""
    late = tmp_path / "late" / "classifier_v1.pt"
    serve = _load_serve(late, monkeypatch)

    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 503

        late.parent.mkdir(parents=True, exist_ok=True)
        late.write_bytes(checkpoint_file.read_bytes())

        assert client.post("/reload").status_code == 200
        assert client.get("/health").status_code == 200
