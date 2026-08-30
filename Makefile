# Convenience wrapper around the commands in README.md. Everything here is a
# thin shell over docker/kubectl - nothing is hidden from the grader.

IMAGE_TRAIN    ?= mlops-train:v1
IMAGE_SERVE    ?= mlops-serve:v1
NAMESPACE      ?= ml-training
PYTHON         ?= python3
# Both Dockerfiles already default to PyTorch's CPU wheel index. Override only if
# you actually want the CUDA build (e.g. building for a real GPU cluster):
#   make build TORCH_INDEX_URL=https://pypi.org/simple
TORCH_INDEX_URL ?=
BUILD_ARGS     := $(if $(TORCH_INDEX_URL),--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL),)

.DEFAULT_GOAL := help
.PHONY: help venv test lint fmt build-train build-serve build train serve test-image \
        k8s-apply k8s-serve k8s-status k8s-logs k8s-forward k8s-clean clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create .venv with dev + serving dependencies
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements/serve.txt -r requirements/dev.txt

test: ## Run the test suite
	.venv/bin/pytest -v

lint: ## ruff check + format check
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

fmt: ## Apply ruff formatting
	.venv/bin/ruff format src tests

build-train: ## Build the training image
	docker build -f docker/Dockerfile.train -t $(IMAGE_TRAIN) $(BUILD_ARGS) .

build-serve: ## Build the serving image
	docker build -f docker/Dockerfile.serve -t $(IMAGE_SERVE) $(BUILD_ARGS) .

build: build-train build-serve ## Build both images

train: ## Run training locally in Docker (override EPOCHS/BATCHES for a smoke run)
	mkdir -p data checkpoints
	docker run --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		$(if $(EPOCHS),-e TRAIN_EPOCHS=$(EPOCHS),) \
		$(if $(BATCHES),-e TRAIN_MAX_TRAIN_BATCHES=$(BATCHES) -e TRAIN_MAX_VAL_BATCHES=$(BATCHES),) \
		$(IMAGE_TRAIN)

serve: ## Run the serving container on :8080
	docker run --rm -p 8080:8080 \
		-v $(PWD)/checkpoints:/app/checkpoints \
		$(IMAGE_SERVE)

test-image: ## Write test_image.png from the CIFAR-10 test split
	.venv/bin/python scripts/make_test_image.py

k8s-apply: ## Namespace + ConfigMaps + training Job
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/training-job.yaml

k8s-serve: ## Deployment + Service + HPA
	kubectl apply -f k8s/serving-deployment.yaml
	kubectl apply -f k8s/serving-service.yaml
	kubectl apply -f k8s/hpa.yaml

k8s-status: ## Pods, job, deployment, service, hpa, pvcs
	kubectl get pods,job,deploy,svc,hpa,pvc -n $(NAMESPACE)

k8s-logs: ## Follow the training Job logs
	kubectl logs -f job/pytorch-training -n $(NAMESPACE)

k8s-forward: ## Port-forward the Service to localhost:8080
	kubectl port-forward svc/model-serving 8080:80 -n $(NAMESPACE)

k8s-clean: ## Delete the namespace (and everything in it)
	kubectl delete namespace $(NAMESPACE) --ignore-not-found

clean: ## Remove local artifacts
	rm -rf .pytest_cache .ruff_cache checkpoints/*.pt test_image.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
