"""Validate Kubernetes failure dataset schema and feature ranges; log issues to MLflow."""
from __future__ import annotations

import mlflow
import numpy as np

from data.loader import FEATURE_NAMES

SCHEMA = {
    "n_features": 10,
    "n_classes": 6,
    "feature_ranges": {
        "restart_count":             (0.0,   100.0),
        "cpu_usage_pct":             (0.0,   100.0),
        "memory_usage_pct":          (0.0,   100.0),
        "pod_ready":                 (0.0,   1.0),
        "last_exit_code":            (0.0,   255.0),
        "waiting_reason":            (0.0,   8.0),    # ordinal encoded
        "oom_killed_count":          (0.0,   50.0),
        "image_pull_errors":         (0.0,   50.0),
        "failed_scheduling_events":  (0.0,   50.0),
        "readiness_probe_failures":  (0.0,   50.0),
    },
}


def validate(dataset: dict, log_to_mlflow: bool = True) -> list[str]:
    warnings: list[str] = []

    if dataset["n_features"] != SCHEMA["n_features"]:
        warnings.append(f"n_features={dataset['n_features']} expected {SCHEMA['n_features']}")

    if dataset["n_classes"] != SCHEMA["n_classes"]:
        warnings.append(f"n_classes={dataset['n_classes']} expected {SCHEMA['n_classes']}")

    X_all = np.vstack([dataset["X_train"], dataset["X_test"]])
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = SCHEMA["feature_ranges"][name]
        col_min, col_max = float(X_all[:, i].min()), float(X_all[:, i].max())
        if col_min < lo or col_max > hi:
            warnings.append(
                f"{name} out of expected range [{lo}, {hi}]: observed [{col_min:.2f}, {col_max:.2f}]"
            )

    if log_to_mlflow:
        mlflow.log_param("data_version", dataset["data_version"])
        mlflow.log_param("data_warnings_count", len(warnings))
        for idx, w in enumerate(warnings):
            mlflow.log_param(f"data_warning_{idx}", w)

    return warnings
