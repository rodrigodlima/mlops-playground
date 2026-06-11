.PHONY: install validate train register promote serve mlflow-ui predict \
        monitor-drift monitor-log \
        docker-build docker-run \
        kfp-setup kfp-build kfp-deploy-mlflow kfp-compile kfp-run kfp-ui kfp-mlflow-ui \
        clean

PORT           ?= 8000
PYTHON         ?= MlFlowStarter/venv/bin/python
PYTHON_KFP     ?= python3.11                    # kfp SDK requires Python < 3.14
UVICORN        ?= MlFlowStarter/venv/bin/uvicorn
MLFLOW         ?= MlFlowStarter/venv/bin/mlflow
KFP_HOST       ?= http://localhost:8080
MLFLOW_K8S_URI ?= http://mlflow-server:5000

# ── Local pipeline (no Kubernetes) ──────────────────────────────────────────

install:
	$(PYTHON) -m pip install -r requirements.txt

validate:
	$(PYTHON) -c "\
from data.loader import load; \
from data.validator import validate; \
d = load(); \
w = validate(d, log_to_mlflow=False); \
print('Warnings:', w if w else 'none')"

train:
	$(PYTHON) -m ml.train

register:
	$(PYTHON) -m ml.register register

promote:
	$(PYTHON) -m ml.register promote

serve:
	$(UVICORN) serve.app:app --host 0.0.0.0 --port $(PORT) --reload

mlflow-ui:
	$(MLFLOW) ui --port 5000

predict:
	curl -s -X POST http://localhost:$(PORT)/predict \
		-H 'Content-Type: application/json' \
		-d '{"instances": [[0, 20.0, 30.0, 1, 0, 7, 0, 0, 0, 0]]}' \
		| $(PYTHON) -m json.tool

monitor-drift:
	$(PYTHON) -m monitor.drift

monitor-log:
	$(PYTHON) -m monitor.logger

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t mlops-playground:latest .

docker-run:
	docker run --rm -p $(PORT):8000 mlops-playground:latest

# ── Kubeflow Pipelines (minikube) ─────────────────────────────────────────────
#
# One-time setup order:
#   1. make kfp-setup        — start minikube + install KFP standalone
#   2. make kfp-build        — build image and load into minikube
#   3. make kfp-deploy-mlflow — deploy MLflow server inside the cluster
#   4. make kfp-compile      — compile pipeline to YAML
#   5. make kfp-run          — submit pipeline run
#   6. make kfp-ui           — open KFP UI  (port-forward :8080)
#   7. make kfp-mlflow-ui    — open MLflow UI (port-forward :5000)

kfp-setup:
	@echo "Starting minikube (4 CPUs, 8 GB RAM)..."
	minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=30g
	@echo "Installing KFP standalone (this takes ~5 min)..."
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=2.2.0"
	@echo "Waiting for KFP pods to be ready (may take 3-5 min)..."
	kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s

kfp-build:
	@echo "Building image and loading into minikube Docker daemon..."
	minikube image build -t mlops-playground:latest .

kfp-deploy-mlflow:
	@echo "Deploying MLflow server inside the cluster..."
	kubectl apply -f k8s/mlflow-deployment.yaml
	kubectl rollout status deployment/mlflow-server --timeout=120s
	@echo "MLflow accessible inside cluster at http://mlflow-server:5000"
	@echo "MLflow NodePort: $$(minikube ip):30500"

kfp-compile:
	$(PYTHON_KFP) -m pipeline.pipeline compile

kfp-run:
	$(PYTHON_KFP) -m pipeline.pipeline run \
		--kfp-host $(KFP_HOST) \
		--mlflow-uri $(MLFLOW_K8S_URI)

kfp-ui:
	@echo "KFP UI → http://localhost:8080"
	kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow

kfp-mlflow-ui:
	@echo "MLflow UI → http://localhost:5000"
	kubectl port-forward svc/mlflow-server 5000:5000

# ─────────────────────────────────────────────────────────────────────────────

clean:
	rm -rf mlruns models data_cache __pycache__ */__pycache__ pipeline/*.yaml
