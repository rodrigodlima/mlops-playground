"""Log prediction batches and ground-truth labels to MLflow for accuracy tracking."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def log_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    batch_id: str | None = None,
) -> dict:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "n_samples": int(len(y_true_arr)),
    }

    run_name = batch_id or datetime.now(timezone.utc).strftime("batch-%Y%m%dT%H%M%S")
    mlflow.set_experiment("wine-quality-monitor")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)
        if batch_id:
            mlflow.log_param("batch_id", batch_id)

    print(f"Logged batch '{run_name}': {metrics}")
    return metrics


if __name__ == "__main__":
    from data.loader import load
    from ml.predict import predict

    dataset = load()
    preds = predict(dataset["X_test"].tolist())
    y_pred = [p["class_id"] for p in preds]
    log_predictions(dataset["y_test"].tolist(), y_pred, batch_id="smoke-test")
