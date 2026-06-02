"""Promote the best MLflow run to the Model Registry using aliases (staging/production)."""
from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "wine-quality"
MODEL_NAME = "wine-quality-classifier"
ALIAS_STAGING = "staging"
ALIAS_PRODUCTION = "production"


def _best_run(metric: str = "accuracy") -> mlflow.entities.Run:
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found. Run `make train` first.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found.")
    return runs[0]


def register(run_id: str | None = None) -> str:
    client = MlflowClient()
    if run_id is None:
        run = _best_run()
        run_id = run.info.run_id
        acc = run.data.metrics.get("accuracy", "?")
        print(f"Best run: {run_id}  accuracy={acc:.4f}")

    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    version = result.version
    client.set_registered_model_alias(MODEL_NAME, ALIAS_STAGING, version)
    print(f"Registered '{MODEL_NAME}' v{version} → @{ALIAS_STAGING}")
    return version


def promote(version: str | None = None) -> str:
    client = MlflowClient()
    if version is None:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_STAGING)
            version = mv.version
        except Exception:
            raise RuntimeError(f"No model at @{ALIAS_STAGING}. Run `make register` first.")

    client.set_registered_model_alias(MODEL_NAME, ALIAS_PRODUCTION, version)
    print(f"Promoted '{MODEL_NAME}' v{version} → @{ALIAS_PRODUCTION}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage MLflow Model Registry")
    parser.add_argument("action", choices=["register", "promote"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    if args.action == "register":
        register(run_id=args.run_id)
    else:
        promote(version=args.version)
