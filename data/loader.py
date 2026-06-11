"""Load and split the Kubernetes failure dataset; stamp a reproducible data version hash."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

CSV_PATH = Path(__file__).resolve().parent.parent / "kubernetes_failures.csv"
DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
BASELINE_PATH = DATA_CACHE / "baseline_X_train.npy"
BASELINE_STATS_PATH = DATA_CACHE / "baseline_stats.json"

# Alphabetical ordering → deterministic ordinal encoding
WAITING_REASON_CATEGORIES = [
    "CrashLoopBackOff",  # 0
    "ErrImagePull",      # 1
    "Error",             # 2
    "ImagePullBackOff",  # 3
    "None",              # 4
    "OOMKilled",         # 5
    "Pending",           # 6
    "Running",           # 7
    "Unschedulable",     # 8
]
WAITING_REASON_TO_INT = {r: i for i, r in enumerate(WAITING_REASON_CATEGORIES)}

FEATURE_NAMES = [
    "restart_count",
    "cpu_usage_pct",
    "memory_usage_pct",
    "pod_ready",
    "last_exit_code",
    "waiting_reason",
    "oom_killed_count",
    "image_pull_errors",
    "failed_scheduling_events",
    "readiness_probe_failures",
]

TARGET_NAMES = [
    "crash_loop",
    "healthy",
    "image_pull_error",
    "oom_killed",
    "probe_failure",
    "scheduling_failure",
]
LABEL_TO_INT = {label: i for i, label in enumerate(TARGET_NAMES)}


def _load_raw() -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            X_rows.append([
                float(row["restart_count"]),
                float(row["cpu_usage_pct"]),
                float(row["memory_usage_pct"]),
                float(row["pod_ready"]),
                float(row["last_exit_code"]),
                float(WAITING_REASON_TO_INT[row["waiting_reason"]]),
                float(row["oom_killed_count"]),
                float(row["image_pull_errors"]),
                float(row["failed_scheduling_events"]),
                float(row["readiness_probe_failures"]),
            ])
            y_rows.append(LABEL_TO_INT[row["label"]])
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)


def load(test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y = _load_raw()
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
