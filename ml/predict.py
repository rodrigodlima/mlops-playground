"""Load persisted model and run predictions from CLI or as a library."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.datasets import load_iris

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_rf.joblib"
TARGET_NAMES = load_iris().target_names.tolist()


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m ml.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict(features: Sequence[Sequence[float]]) -> list[dict]:
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
    raw = sys.stdin.read().strip() or "[[5.1, 3.5, 1.4, 0.2]]"
    features = json.loads(raw)
    print(json.dumps(predict(features), indent=2))
