"""Validate wine dataset schema and feature ranges; log issues to MLflow."""
from __future__ import annotations

import mlflow
import numpy as np

from data.loader import FEATURE_NAMES

SCHEMA = {
    "n_features": 13,
    "n_classes": 3,
    "feature_ranges": {
        "alcohol":                     (10.0, 16.0),
        "malic_acid":                  (0.5,  6.5),
        "ash":                         (1.0,  4.0),
        "alcalinity_of_ash":           (9.0,  34.0),
        "magnesium":                   (60.0, 180.0),
        "total_phenols":               (0.5,  4.5),
        "flavanoids":                  (0.1,  5.5),
        "nonflavanoid_phenols":        (0.05, 0.75),
        "proanthocyanins":             (0.2,  4.5),
        "color_intensity":             (1.0,  14.0),
        "hue":                         (0.4,  2.0),
        "od280_od315_of_diluted_wines":(1.0,  4.5),
        "proline":                     (200.0, 1800.0),
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
