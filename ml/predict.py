"""Load Production model from MLflow Registry and run predictions."""
from __future__ import annotations

import json
import sys
from functools import lru_cache

import mlflow.sklearn
import numpy as np

from data.loader import TARGET_NAMES

MODEL_NAME = "wine-quality-classifier"
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
    # Sample: first row from wine dataset (class_0)
    raw = sys.stdin.read().strip() or (
        "[[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]"
    )
    print(json.dumps(predict(json.loads(raw)), indent=2))
