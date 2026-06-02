"""Load and split the wine dataset; stamp a reproducible data version hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
BASELINE_PATH = DATA_CACHE / "baseline_X_train.npy"
BASELINE_STATS_PATH = DATA_CACHE / "baseline_stats.json"

FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]
TARGET_NAMES = ["class_0", "class_1", "class_2"]


def load(test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y = load_wine(return_X_y=True)
    data_version = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:12]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "data_version": data_version,
        "n_features": X.shape[1],
        "n_classes": int(len(np.unique(y))),
    }


def save_baseline(X_train: np.ndarray) -> None:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(BASELINE_PATH, X_train)
    stats = {
        name: {"mean": float(X_train[:, i].mean()), "std": float(X_train[:, i].std())}
        for i, name in enumerate(FEATURE_NAMES)
    }
    BASELINE_STATS_PATH.write_text(json.dumps(stats, indent=2))


def load_baseline() -> np.ndarray:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found at {BASELINE_PATH}. Run `make train` first.")
    return np.load(BASELINE_PATH)
