"""Load Production model from MLflow Registry and run predictions."""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import mlflow.sklearn
import numpy as np

from data.loader import TARGET_NAMES

MODEL_NAME = "k8s-failure-classifier"
MODEL_ALIAS = "production"


@lru_cache(maxsize=1)
def load_model():
    try:
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot load '{MODEL_NAME}@{MODEL_ALIAS}'. "
            f"Run `make register && make promote` first.\n{e}"
        ) from e


def predict(features: list[list[float]]) -> list[dict]:
    model = load_model()
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    classes = model.predict(X)
    probs = model.predict_proba(X)
    return [
        {
            "class_id": int(c),
            "class_name": TARGET_NAMES[int(c)],
            "proba": {TARGET_NAMES[i]: float(p) for i, p in enumerate(row)},
        }
        for c, row in zip(classes, probs)
    ]


if __name__ == "__main__":
    # healthy pod: running, pod_ready=1, no errors
    # waiting_reason=7 (Running), all counters=0
    raw = sys.stdin.read().strip() or (
        "[[0, 20.0, 30.0, 1, 0, 7, 0, 0, 0, 0]]"
    )
    print(json.dumps(predict(json.loads(raw)), indent=2))
