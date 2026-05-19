"""Train iris classifier, log run to MLflow, persist model artifact."""
from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "iris_rf.joblib"
EXPERIMENT = "iris-rf"


def train(n_estimators: int = 100, max_depth: int | None = 5, random_state: int = 42) -> dict:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        })

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, artifact_path="model")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)

        return {"run_id": run.info.run_id, "model_path": str(MODEL_PATH), **metrics}


if __name__ == "__main__":
    result = train()
    print(result)
