# mlops-playground

End-to-end MLOps pipeline POC built with MLflow. Five stages: **Data → Train → Register → Serve → Monitor**. Designed to demonstrate how MLOps maps to familiar DevOps concepts.

Dataset: [UCI Wine Recognition](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) — 178 samples, 13 chemical features, 3 classes.

---

## Quick start

```bash
make install        # install dependencies
make validate       # Stage 1: validate data schema
make train          # Stage 2: train + track with MLflow
make register       # Stage 3: register best run → @staging
make promote        # Stage 3: promote @staging → @production
make serve          # Stage 4: start inference API on :8000
make predict        # Stage 4: call /predict endpoint
make monitor-drift  # Stage 5: KS drift test vs training baseline
make monitor-log    # Stage 5: log prediction accuracy to MLflow
make mlflow-ui      # browse all runs and the Model Registry
```

---

## Project structure

```
mlops-playground/
├── data/
│   ├── loader.py         # load_wine(), stratified split, data version hash, baseline save/load
│   └── validator.py      # schema check + feature range validation
├── ml/
│   ├── train.py          # GradientBoosting + cross-val, logs params/metrics/artifact to MLflow
│   ├── register.py       # register best run to Model Registry, manage @staging/@production aliases
│   └── predict.py        # load model from @production alias, run inference
├── serve/
│   └── app.py            # FastAPI: GET /health, POST /predict
├── monitor/
│   ├── drift.py          # Kolmogorov-Smirnov test on 13 features vs training baseline
│   └── logger.py         # log prediction batches + ground truth for online accuracy tracking
├── data_cache/           # generated: baseline_X_train.npy, baseline_stats.json
├── mlruns/               # generated: MLflow tracking store (SQLite)
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

1. Loads the wine dataset via `sklearn.datasets.load_wine`.
2. Computes a SHA-256 hash of the raw arrays → **data version** (12-char hex). Same data always produces the same hash.
3. Saves the training split to `data_cache/baseline_X_train.npy` for downstream drift detection.
4. Checks 15 rules: correct number of features (13), correct number of classes (3), and expected value ranges for each of the 13 features.
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
2. Starts an MLflow run inside the `wine-quality` experiment.
3. Logs hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `random_state`, `n_features`, `n_classes`.
4. Logs data metadata: `data_version`, `data_warnings_count`.
5. Runs 5-fold cross-validation on the training split → logs `cv_mean_accuracy`, `cv_std_accuracy`.
6. Trains a `GradientBoostingClassifier` on the full training split.
7. Evaluates on the held-out test split → logs `accuracy`, `f1_macro`.
8. Logs the fitted model as an MLflow artifact (skops-compatible sklearn model).

Expected output:

```
run_id=<hex>
{'accuracy': 0.944, 'f1_macro': 0.945, 'cv_mean_accuracy': 0.958, 'cv_std_accuracy': 0.040}
```

To experiment with hyperparameters:

```python
from ml.train import train
train(n_estimators=200, max_depth=5, learning_rate=0.05)
```

Each call creates a new run in the `wine-quality` experiment. Compare them in the MLflow UI (`make mlflow-ui`).

---

## Stage 3 — Model Registry

**Goal:** promote a vetted run from the tracking store to a named, versioned artifact store with explicit lifecycle aliases.

### Register the best run

```bash
make register
```

Selects the run with the highest `accuracy` metric, registers its model artifact as a new version of `wine-quality-classifier`, and assigns the `@staging` alias.

To register a specific run by ID:

```bash
MlFlowStarter/venv/bin/python -m ml.register register --run-id <run_id>
```

### Promote to production

```bash
make promote
```

Moves the `@staging` alias to `@production`. The `@staging` alias remains pointing at the same version until the next `make register` call.

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

Aliases replace the deprecated MLflow stages API (`Staging`/`Production` string stages). Model URI format: `models:/wine-quality-classifier@production`.

---

## Stage 4 — Serving

**Goal:** expose the production model as an HTTP API.

```bash
make serve       # starts uvicorn on :8000 with --reload
```

The FastAPI app loads the model lazily on the first request using `models:/wine-quality-classifier@production`. If no model is promoted, the `/health` endpoint returns HTTP 503.

### Endpoints

#### `GET /health`

Returns model status.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "wine-quality-classifier", "alias": "production"}
```

#### `POST /predict`

Accepts one or more rows of 13 features (in the same order as the dataset):

| Index | Feature |
|-------|---------|
| 0 | alcohol |
| 1 | malic_acid |
| 2 | ash |
| 3 | alcalinity_of_ash |
| 4 | magnesium |
| 5 | total_phenols |
| 6 | flavanoids |
| 7 | nonflavanoid_phenols |
| 8 | proanthocyanins |
| 9 | color_intensity |
| 10 | hue |
| 11 | od280_od315_of_diluted_wines |
| 12 | proline |

```bash
make predict
```

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]}'
```

```json
{
  "predictions": [
    {
      "class_id": 0,
      "class_name": "class_0",
      "proba": {"class_0": 0.9999, "class_1": 0.0001, "class_2": 0.0000}
    }
  ],
  "model_name": "wine-quality-classifier",
  "model_alias": "production"
}
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Stage 5 — Monitoring

Two independent monitors, both write runs to the `wine-quality-monitor` MLflow experiment.

### Feature drift detection

```bash
make monitor-drift
```

Runs a two-sample [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) for each of the 13 features, comparing the incoming data distribution against the training baseline saved in `data_cache/baseline_X_train.npy`.

- **p-value < 0.05** → drift flagged for that feature.
- Results logged to MLflow: `ks_<feature>`, `pval_<feature>`, `n_drifted_features`, `drifted_features`.

Expected output (same dataset, no drift):

```
Drift check: 0/13 features drifted  →  none
```

To simulate drift, pass corrupted data:

```python
import numpy as np
from monitor.drift import detect_drift
from data.loader import load

dataset = load()
X_drifted = dataset["X_test"] * 2.5  # shift distribution
detect_drift(X_drifted)
```

### Prediction accuracy logging

```bash
make monitor-log
```

Accepts ground-truth labels alongside predictions and logs `accuracy`, `f1_macro`, and `n_samples` to MLflow. Use this to track model performance on labeled production traffic over time.

```python
from monitor.logger import log_predictions
log_predictions(y_true=[0, 1, 2], y_pred=[0, 1, 1], batch_id="2025-06-01-batch")
```

---

## MLflow UI

```bash
make mlflow-ui    # opens on http://localhost:5000
```

Key views:

- **Experiments → wine-quality**: all training runs with params and metrics. Use the comparison view to rank runs by `accuracy` or `cv_mean_accuracy`.
- **Experiments → wine-quality-monitor**: drift check runs and prediction accuracy batches.
- **Models → wine-quality-classifier**: version history, active aliases (`@staging`, `@production`), lineage back to the source training run.

---

## Full pipeline run

```bash
make install
make validate
make train
make register
make promote
make serve &         # background
make predict
make monitor-drift
make monitor-log
make mlflow-ui
```

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
