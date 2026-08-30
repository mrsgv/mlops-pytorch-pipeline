"""Static checks on the Kubernetes manifests.

These encode the Part D/E requirements (resources, probes, mount paths, rolling
update policy) so a careless edit to a YAML file fails CI instead of failing in
the cluster. Only PyYAML is needed, so this file runs without torch installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

K8S = Path(__file__).resolve().parent.parent / "k8s"


def load_docs(name: str) -> list[dict[str, Any]]:
    with open(K8S / name) as handle:
        return [doc for doc in yaml.safe_load_all(handle) if doc]


def find(docs: list[dict[str, Any]], kind: str, name: str | None = None) -> dict[str, Any]:
    for doc in docs:
        if doc.get("kind") == kind and (name is None or doc["metadata"]["name"] == name):
            return doc
    raise AssertionError(f"{kind} {name or ''} not found")


def test_every_manifest_parses_and_is_namespaced() -> None:
    files = sorted(p.name for p in K8S.glob("*.yaml"))
    assert files, "no manifests found"
    for name in files:
        for doc in load_docs(name):
            assert doc.get("apiVersion"), f"{name}: missing apiVersion"
            assert doc.get("kind"), f"{name}: missing kind"
            if doc["kind"] != "Namespace":
                assert doc["metadata"].get("namespace") == "ml-training", (
                    f"{name}: {doc['kind']} is not in the ml-training namespace"
                )


def test_namespace() -> None:
    namespace = find(load_docs("namespace.yaml"), "Namespace")
    assert namespace["metadata"]["name"] == "ml-training"
    assert namespace["metadata"]["labels"]["project"] == "mlops-pytorch-pipeline"


def test_training_configmap_holds_a_valid_config() -> None:
    configmap = find(load_docs("configmap.yaml"), "ConfigMap", "training-config")
    config = yaml.safe_load(configmap["data"]["training_config.yaml"])

    assert config["model"]["architecture"] == "resnet18"
    assert config["model"]["num_classes"] == 10
    assert config["training"]["epochs"] == 10
    assert config["training"]["batch_size"] == 64
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["early_stopping_patience"] == 3
    assert config["data"]["dataset"] == "cifar10"
    assert config["data"]["data_dir"] == "/app/data"
    assert config["output"]["checkpoint_dir"] == "/app/checkpoints"
    assert config["output"]["model_name"] == "classifier_v1.pt"


def test_training_job_mounts_and_resources() -> None:
    docs = load_docs("training-job.yaml")
    job = find(docs, "Job", "pytorch-training")
    spec = job["spec"]["template"]["spec"]
    container = spec["containers"][0]

    assert container["image"].startswith("mlops-train")
    assert job["spec"]["template"]["spec"]["restartPolicy"] in {"OnFailure", "Never"}

    mounts = {m["mountPath"]: m["name"] for m in container["volumeMounts"]}
    assert set(mounts) >= {"/app/configs", "/app/data", "/app/checkpoints"}

    volumes = {v["name"]: v for v in spec["volumes"]}
    # ConfigMap volume at /app/configs
    assert volumes[mounts["/app/configs"]]["configMap"]["name"] == "training-config"
    # PVCs for data and checkpoints
    for path in ("/app/data", "/app/checkpoints"):
        assert "persistentVolumeClaim" in volumes[mounts[path]], f"{path} is not a PVC"

    assert container["resources"]["requests"] == {"cpu": "2", "memory": "4Gi"}
    assert container["resources"]["limits"] == {"cpu": "2", "memory": "4Gi"}

    claims = {d["metadata"]["name"] for d in docs if d["kind"] == "PersistentVolumeClaim"}
    assert claims == {"ml-data", "ml-checkpoints"}


def test_gpu_job_requests_a_gpu_and_targets_gpu_nodes() -> None:
    spec = find(load_docs("training-job-gpu.yaml"), "Job")["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert spec["nodeSelector"], "GPU job needs a nodeSelector"
    assert spec["tolerations"], "GPU job needs a toleration for tainted GPU nodes"


def test_serving_deployment_matches_the_spec() -> None:
    deployment = find(load_docs("serving-deployment.yaml"), "Deployment", "model-serving")
    spec = deployment["spec"]
    assert spec["replicas"] == 2

    rolling = spec["strategy"]
    assert rolling["type"] == "RollingUpdate"
    assert rolling["rollingUpdate"] == {"maxSurge": 1, "maxUnavailable": 0}

    container = spec["template"]["spec"]["containers"][0]
    assert container["ports"][0]["containerPort"] == 8080

    liveness = container["livenessProbe"]
    assert liveness["httpGet"]["path"] == "/health"
    assert liveness["periodSeconds"] == 10
    assert liveness["failureThreshold"] == 3

    readiness = container["readinessProbe"]
    assert readiness["httpGet"]["path"] == "/health"
    assert readiness["periodSeconds"] == 5
    assert readiness["initialDelaySeconds"] == 15

    assert container["resources"]["requests"] == {"cpu": "500m", "memory": "1Gi"}
    assert container["resources"]["limits"] == {"cpu": "1", "memory": "2Gi"}

    checkpoints = next(m for m in container["volumeMounts"] if m["mountPath"] == "/app/checkpoints")
    assert checkpoints["readOnly"] is True


def test_serving_service_exposes_80_to_8080() -> None:
    service = find(load_docs("serving-service.yaml"), "Service", "model-serving")
    assert service["spec"]["type"] in {"ClusterIP", "LoadBalancer"}
    port = service["spec"]["ports"][0]
    assert (port["port"], port["targetPort"]) == (80, 8080)
    assert service["spec"]["selector"] == {"app": "model-serving"}


def test_hpa_targets_the_serving_deployment() -> None:
    hpa = find(load_docs("hpa.yaml"), "HorizontalPodAutoscaler")
    assert hpa["spec"]["scaleTargetRef"]["name"] == "model-serving"
    assert hpa["spec"]["minReplicas"] >= 2
    assert hpa["spec"]["maxReplicas"] > hpa["spec"]["minReplicas"]
    assert {m["resource"]["name"] for m in hpa["spec"]["metrics"]} >= {"cpu"}


@pytest.mark.parametrize("manifest", ["training-job.yaml", "serving-deployment.yaml"])
def test_workloads_run_as_the_non_root_image_user(manifest: str) -> None:
    docs = load_docs(manifest)
    workload = next(d for d in docs if d["kind"] in {"Job", "Deployment"})
    security = workload["spec"]["template"]["spec"]["securityContext"]
    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] == 1001  # matches the Dockerfile's mlops user


def test_no_real_secret_is_committed() -> None:
    assert not (K8S / "secret.yaml").exists(), "k8s/secret.yaml must stay out of git"
    example = find(load_docs("secret.example.yaml"), "Secret", "serving-secrets")
    assert "replace-me" in example["stringData"]["MODEL_API_KEY"]
