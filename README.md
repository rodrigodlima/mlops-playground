# mlops-playground

End-to-end MLOps pipeline POC. Five stages: **Data → Train → Register → Serve → Monitor**.

**Use case:** classify the failure mode of a Kubernetes pod from runtime telemetry signals.  
**Dataset:** `kubernetes_failures.csv` — 500 samples, 10 features, 6 classes (balanced).  
**Orchestration:** [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) (minikube) + [MLflow](https://mlflow.org/) for tracking and model registry.

---

## Architecture

```
minikube cluster
├── kubeflow namespace   → KFP control plane (scheduler, UI, artifact store)
└── default namespace
    ├── mlflow-server    → tracking server + model registry (SQLite, NodePort :30500)
    └── pipeline run pods (one per step, spawned by KFP)
          1 · Validate Data
          2 · Train Model  ──→ logs to mlflow-server
          3 · Register → @staging
          4 · Promote → @production
          5 · Detect Drift ──→ logs to mlflow-server
```

---

## Quick start — local (no Kubernetes)

Run the pipeline locally without any cluster for development and testing.

```bash
make install        # install dependencies
make validate       # Stage 1: validate data schema and feature ranges
make train          # Stage 2: train + track with MLflow
make register       # Stage 3: register best run → @staging
make promote        # Stage 3: promote @staging → @production
make serve          # Stage 4: start inference API on :8000
make predict        # Stage 4: call /predict endpoint (healthy pod example)
make monitor-drift  # Stage 5: KS drift test vs training baseline
make monitor-log    # Stage 5: log prediction accuracy to MLflow
make mlflow-ui      # browse all runs and the Model Registry at :5000
```

## Quick start — Kubeflow Pipelines (minikube)

Run the full orchestrated pipeline on a local Kubernetes cluster.

```bash
# One-time cluster setup (~10 min)
make kfp-setup           # start minikube + install KFP standalone
make kfp-build           # build Docker image and load into minikube
make kfp-deploy-mlflow   # deploy MLflow server inside the cluster

# Compile and submit
make kfp-compile         # compile pipeline → pipeline/k8s_failure_pipeline.yaml
make kfp-run             # submit pipeline run to KFP

# Open UIs (each in its own terminal)
make kfp-ui              # KFP pipeline UI   → http://localhost:8080
make kfp-mlflow-ui       # MLflow tracking UI → http://localhost:5000
```

---

## Dataset

File: `kubernetes_failures.csv`

| Feature | Type | Description |
|---|---|---|
| `restart_count` | numeric | Number of container restarts |
| `cpu_usage_pct` | numeric | CPU usage (0–100%) |
| `memory_usage_pct` | numeric | Memory usage (0–100%) |
| `pod_ready` | binary | Pod in Ready state (0 or 1) |
| `last_exit_code` | numeric | Last container exit code (0, 1, 137…) |
| `waiting_reason` | categorical* | Kubernetes waiting reason |
| `oom_killed_count` | numeric | Number of OOMKilled events |
| `image_pull_errors` | numeric | Number of image pull failures |
| `failed_scheduling_events` | numeric | Number of failed scheduling attempts |
| `readiness_probe_failures` | numeric | Number of readiness probe failures |

*`waiting_reason` is ordinal-encoded at load time. Mapping:

| Value | Encoded |
|---|---|
| CrashLoopBackOff | 0 |
| ErrImagePull | 1 |
| Error | 2 |
| ImagePullBackOff | 3 |
| None | 4 |
| OOMKilled | 5 |
| Pending | 6 |
| Running | 7 |
| Unschedulable | 8 |

**Target classes (6, balanced ~83 samples each):**

| Class | Description |
|---|---|
| `crash_loop` | Container crashing repeatedly (CrashLoopBackOff) |
| `healthy` | Pod running normally |
| `image_pull_error` | Cannot pull container image |
| `oom_killed` | Container killed by OOM killer |
| `probe_failure` | Readiness/liveness probe failing |
| `scheduling_failure` | Pod cannot be scheduled to a node |

---

## Project structure

```
mlops-playground/
├── kubernetes_failures.csv      # source dataset (500 rows × 10 features + label)
├── data/
│   ├── loader.py                # load CSV, ordinal-encode waiting_reason, split, baseline save/load
│   └── validator.py             # schema check + feature range validation, logs to MLflow
├── ml/
│   ├── train.py                 # GradientBoosting + 5-fold CV, logs to MLflow
│   ├── register.py              # register run to MLflow Model Registry, manage aliases
│   └── predict.py               # load model from @production alias, run inference
├── serve/
│   └── app.py                   # FastAPI: GET /health, POST /predict
├── monitor/
│   ├── drift.py                 # Kolmogorov-Smirnov test on 10 features vs training baseline
│   └── logger.py                # log prediction batches + ground truth for accuracy tracking
├── pipeline/
│   ├── components.py            # KFP v2 component definitions (one per pipeline stage)
│   ├── pipeline.py              # pipeline assembly, compile, and submission script
│   └── k8s_failure_pipeline.yaml  # generated: compiled KFP pipeline IR
├── k8s/
│   └── mlflow-deployment.yaml   # MLflow server Deployment + NodePort Service
├── data_cache/                  # generated: baseline_X_train.npy, baseline_stats.json
├── mlruns/                      # generated: MLflow tracking store (SQLite, local mode)
├── Dockerfile                   # image used by local serve and KFP pipeline steps
├── Makefile
└── requirements.txt
```

---

## Prerequisites

Python 3.10+ with the packages in `requirements.txt`. The repo ships with a conda environment at `MlFlowStarter/venv/`. All `make` targets default to that interpreter:

```bash
# use the bundled venv (default)
make train

# use a custom interpreter
make train PYTHON=/path/to/python
```

---

## Stage 1 — Data validation

**Goal:** detect schema or distribution problems before a training run starts.

```bash
make validate
```

What it does:

1. Reads `kubernetes_failures.csv` and ordinal-encodes `waiting_reason`.
2. Computes a SHA-256 hash of the raw arrays → **data version** (12-char hex). Same data always produces the same hash.
3. Saves the training split to `data_cache/baseline_X_train.npy` (used by drift detection).
4. Checks 12 rules: correct feature count (10), correct class count (6), expected value ranges for each feature.
5. Prints any warnings. When called from `make train`, warnings are also logged as MLflow params.

Expected output:

```
Warnings: none
```

---

## Stage 2 — Training

**Goal:** train a reproducible model and record everything needed to reproduce or compare runs.

```bash
make train
```

What it does:

1. Loads and validates the dataset (Stage 1 runs automatically inside `train.py`).
2. Starts an MLflow run inside the `k8s-failure-detection` experiment.
3. Logs hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `random_state`, `n_features`, `n_classes`.
4. Logs data metadata: `data_version`, `data_warnings_count`.
5. Runs 5-fold cross-validation on the training split → logs `cv_mean_accuracy`, `cv_std_accuracy`.
6. Trains a `GradientBoostingClassifier` on the full training split.
7. Evaluates on the held-out test split → logs `accuracy`, `f1_macro`.
8. Logs the fitted model as an MLflow artifact.

Expected output:

```
run_id=<hex>
{'accuracy': 1.0, 'f1_macro': 1.0, 'cv_mean_accuracy': 0.975, 'cv_std_accuracy': 0.014}
```

To experiment with hyperparameters:

```python
from ml.train import train
train(n_estimators=200, max_depth=5, learning_rate=0.05)
```

Each call creates a new run in the `k8s-failure-detection` experiment. Compare them in the MLflow UI (`make mlflow-ui`).

---

## Stage 3 — Model Registry

**Goal:** promote a vetted run from the tracking store to a named, versioned artifact store with explicit lifecycle aliases.

### Register the best run

```bash
make register
```

Selects the run with the highest `accuracy` metric, registers its model artifact as a new version of `k8s-failure-classifier`, and assigns the `@staging` alias.

To register a specific run by ID:

```bash
MlFlowStarter/venv/bin/python -m ml.register register --run-id <run_id>
```

### Promote to production

```bash
make promote
```

Moves the `@staging` alias to `@production`. The serving layer always loads `@production`.

To promote a specific version:

```bash
MlFlowStarter/venv/bin/python -m ml.register promote --version 3
```

### Alias model lifecycle

```
New run  →  make register  →  @staging
                          →  make promote  →  @production
                                         ↳  inference loads @production
```

Model URI format: `models:/k8s-failure-classifier@production`

---

## Stage 4 — Serving

**Goal:** expose the production model as an HTTP API.

```bash
make serve       # starts uvicorn on :8000 with --reload
```

The FastAPI app loads the model lazily on the first request. If no model is promoted, `/health` returns HTTP 503.

### Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "k8s-failure-classifier", "alias": "production"}
```

#### `POST /predict`

Accepts one or more rows of 10 features in the order listed in the [Dataset](#dataset) section.

```bash
make predict
```

Manual example — healthy pod (Running, pod_ready=1, all counters zero):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[0, 20.0, 30.0, 1, 0, 7, 0, 0, 0, 0]]}'
```

OOMKilled pod (high memory, exit_code=137, oom_killed_count=3):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[8, 90.0, 99.5, 0, 137, 5, 3, 0, 0, 0]]}'
```

Scheduling failure (pod stuck in Pending, failed_scheduling_events=5):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[0, 5.0, 10.0, 0, 0, 6, 0, 0, 5, 0]]}'
```

Response format:

```json
{
  "predictions": [
    {
      "class_id": 1,
      "class_name": "healthy",
      "proba": {
        "crash_loop": 0.000,
        "healthy": 0.999,
        "image_pull_error": 0.000,
        "oom_killed": 0.000,
        "probe_failure": 0.000,
        "scheduling_failure": 0.000
      }
    }
  ],
  "model_name": "k8s-failure-classifier",
  "model_alias": "production"
}
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Stage 5 — Monitoring

Two independent monitors. Both write runs to the `k8s-failure-detection-monitor` MLflow experiment.

### Feature drift detection

```bash
make monitor-drift
```

Runs a two-sample [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) for each of the 10 features, comparing incoming data against `data_cache/baseline_X_train.npy`.

- **p-value < 0.05** → drift flagged for that feature.
- Results logged to MLflow: `ks_<feature>`, `pval_<feature>`, `n_drifted_features`, `drifted_features`.

Expected output (no drift):

```
Drift check: 0/10 features drifted  →  none
```

To simulate a degraded cluster (high restart counts, high CPU):

```python
import numpy as np
from monitor.drift import detect_drift
from data.loader import load

dataset = load()
X_degraded = dataset["X_test"].copy()
X_degraded[:, 0] *= 10   # restart_count × 10
X_degraded[:, 1] *= 1.5  # cpu_usage_pct × 1.5
detect_drift(X_degraded)
```

### Prediction accuracy logging

```bash
make monitor-log
```

Accepts ground-truth labels alongside predictions and logs `accuracy`, `f1_macro`, and `n_samples` to MLflow. Use this to track model performance on labeled production traffic over time.

```python
from monitor.logger import log_predictions
log_predictions(
    y_true=[0, 1, 4],
    y_pred=[0, 1, 3],
    batch_id="2025-06-09-oncall-batch"
)
```

---

## MLflow UI

```bash
make mlflow-ui    # opens on http://localhost:5000
```

Key views:

- **Experiments → k8s-failure-detection**: all training runs with params and metrics. Use the comparison view to rank runs by `accuracy` or `cv_mean_accuracy`.
- **Experiments → wine-quality-monitor**: drift check runs and prediction accuracy batches.
- **Models → k8s-failure-classifier**: version history, active aliases (`@staging`, `@production`), lineage back to the source training run.

---

## Full pipeline run

```bash
make install
make validate
make train
make register
make promote
make serve &          # background
make predict
make monitor-drift
make monitor-log
make mlflow-ui
```

---

## Kubeflow Pipelines — detailed setup

### Prerequisites

| Tool | Min version | Install |
|---|---|---|
| Docker | 20+ | [docs.docker.com](https://docs.docker.com/get-docker/) |
| minikube | 1.30+ | `brew install minikube` |
| kubectl | 1.26+ | `brew install kubectl` |
| Python | 3.11 | `brew install python@3.11` |
| kfp SDK | 2.9+ | `python3.11 -m pip install "kfp>=2.7,<2.10"` |

> **Note:** kfp SDK requires Python < 3.14. The `make kfp-*` targets use `python3.11` automatically.

### Step 1 — Start cluster + install KFP

```bash
make kfp-setup
```

This runs:
```bash
minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=30g
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=2.2.0"
```

Wait until all pods are ready (3–5 min):
```bash
kubectl get pods -n kubeflow
```

### Step 2 — Build and load Docker image

```bash
make kfp-build
```

Builds `mlops-playground:latest` with all code + dataset and loads it into minikube's Docker daemon. The image is used as the base for every pipeline step.

What's inside the image:
```
/app/
├── data/                # loader.py, validator.py
├── ml/                  # train.py, register.py, predict.py
├── monitor/             # drift.py, logger.py
├── serve/               # app.py
└── kubernetes_failures.csv
```

### Step 3 — Deploy MLflow inside the cluster

```bash
make kfp-deploy-mlflow
```

Applies `k8s/mlflow-deployment.yaml` which creates:
- **Deployment** `mlflow-server` — runs `mlflow server` using the same image
- **Service** `mlflow-server` — ClusterIP at `http://mlflow-server:5000` (visible to all pipeline pods) + NodePort `30500` (visible from host)

### Step 4 — Compile pipeline

```bash
make kfp-compile
```

Generates `pipeline/k8s_failure_pipeline.yaml` — the pipeline definition in KFP IR (Intermediate Representation). This file can be uploaded directly to any KFP cluster via the UI.

Pipeline structure:

```
validate-data-op  ──→  train-model-op  ──→  register-model-op  ──→  promote-model-op  ──→  detect-drift-op
   1 · Validate       2 · Train              3 · Register            4 · Promote           5 · Detect Drift
                      (logs to MLflow)       (→ @staging)            (→ @production)       (logs to MLflow)
```

### Step 5 — Submit and run

```bash
make kfp-run
```

Submits a pipeline run to KFP. The run is visible in the KFP UI under **Experiments → k8s-failure-detection**.

To override hyperparameters:
```bash
make kfp-run MLFLOW_K8S_URI=http://mlflow-server:5000
# or directly:
python3.11 -m pipeline.pipeline run \
  --kfp-host http://localhost:8080 \
  --mlflow-uri http://mlflow-server:5000 \
  --n-estimators 200 \
  --max-depth 5
```

### Step 6 — Open UIs

Open two terminals:

```bash
# Terminal 1 — KFP pipeline UI
make kfp-ui        # → http://localhost:8080

# Terminal 2 — MLflow tracking + model registry
make kfp-mlflow-ui # → http://localhost:5000
```

**KFP UI** shows the pipeline DAG, per-step logs, input/output parameters, and run history.  
**MLflow UI** shows training metrics, model versions, `@staging` and `@production` aliases, and drift check runs.

### How the two tools work together

```
KFP (orchestrator)              MLflow (tracking + registry)
─────────────────               ────────────────────────────
Schedules step 1 ──────────────→ validate: logs data_version param
Schedules step 2 ──────────────→ train: logs params, metrics, model artifact + baseline
Passes run_id to step 3 ───────→ register: creates v1 → @staging
Passes version to step 4 ──────→ promote: moves @staging → @production
Passes run_id to step 5 ───────→ drift: downloads baseline, logs KS stats
```

KFP never stores ML metadata — that's MLflow's job. KFP only coordinates execution and passes scalar outputs between steps.

---

## Docker

```bash
make docker-build    # builds image with all dependencies
make docker-run      # serves on :8000
```

The container expects a promoted model in an MLflow tracking store accessible at runtime. For a self-contained image, mount `mlruns/` or point `MLFLOW_TRACKING_URI` to a remote store.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlflow.db` (SQLite) | MLflow backend store URI |
| `PORT` | `8000` | Port for the inference server |
| `PYTHON` | `MlFlowStarter/venv/bin/python` | Python interpreter used by make targets |
