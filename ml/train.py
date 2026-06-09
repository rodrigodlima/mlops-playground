"""Train Kubernetes failure classifier, log run to MLflow."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from data.loader import load, save_baseline
from data.validator import validate

EXPERIMENT = "k8s-failure-detection"


def train(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> dict:
    dataset = load(random_state=random_state)
    save_baseline(dataset["X_train"])

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "n_features": dataset["n_features"],
            "n_classes": dataset["n_classes"],
        })

        warnings = validate(dataset, log_to_mlflow=True)
        if warnings:
            print(f"[WARN] Data validation issues: {warnings}")

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        cv_scores = cross_val_score(
            clf, dataset["X_train"], dataset["y_train"], cv=5, scoring="accuracy"
        )
        clf.fit(dataset["X_train"], dataset["y_train"])
        preds = clf.predict(dataset["X_test"])

        metrics = {
            "accuracy": float(accuracy_score(dataset["y_test"], preds)),
            "f1_macro": float(f1_score(dataset["y_test"], preds, average="macro")),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, artifact_path="model", input_example=dataset["X_test"][:2])

        print(f"run_id={run.info.run_id}")
        print(metrics)
        return {"run_id": run.info.run_id, **metrics}


if __name__ == "__main__":
    train()
