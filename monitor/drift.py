"""Detect feature drift using Kolmogorov-Smirnov test against training baseline."""
from __future__ import annotations

import mlflow
import numpy as np
from scipy import stats

from data.loader import FEATURE_NAMES, load_baseline

DRIFT_THRESHOLD = 0.05  # p-value below this signals drift


def detect_drift(X_new: np.ndarray) -> dict[str, dict]:
    X_baseline = load_baseline()

    mlflow.set_experiment("wine-quality-monitor")
    with mlflow.start_run(run_name="drift-check"):
        results: dict[str, dict] = {}
        drifted: list[str] = []

        for i, name in enumerate(FEATURE_NAMES):
            stat, p_value = stats.ks_2samp(X_baseline[:, i], X_new[:, i])
            is_drifted = p_value < DRIFT_THRESHOLD
            results[name] = {"ks_stat": float(stat), "p_value": float(p_value), "drifted": is_drifted}
            mlflow.log_metric(f"ks_{name}", stat)
            mlflow.log_metric(f"pval_{name}", p_value)
            if is_drifted:
                drifted.append(name)

        mlflow.log_metric("n_drifted_features", len(drifted))
        mlflow.log_param("drifted_features", ",".join(drifted) if drifted else "none")

    status = f"{len(drifted)}/{len(FEATURE_NAMES)} features drifted"
    print(f"Drift check: {status}  →  {drifted or 'none'}")
    return results


if __name__ == "__main__":
    from data.loader import load
    dataset = load()
    detect_drift(dataset["X_test"])
