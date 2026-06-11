"""KFP v2 component definitions for the k8s-failure MLOps pipeline."""
from kfp import dsl

IMAGE = "mlops-playground:latest"


@dsl.component(base_image=IMAGE)
def validate_data_op(mlflow_uri: str) -> int:
    import mlflow
    from data.loader import load
    from data.validator import validate

    mlflow.set_tracking_uri(mlflow_uri)
    dataset = load()
    warnings = validate(dataset, log_to_mlflow=False)
    print(f"Data version : {dataset['data_version']}")
    print(f"Warnings     : {warnings or 'none'}")
    return len(warnings)


@dsl.component(base_image=IMAGE)
def train_model_op(
    mlflow_uri: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> str:
    """Train model and log baseline artifact to MLflow. Returns run_id."""
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_score

    from data.loader import load, save_baseline
    from data.validator import validate

    mlflow.set_tracking_uri(mlflow_uri)
    dataset = load()
    save_baseline(dataset["X_train"])

    mlflow.set_experiment("k8s-failure-detection")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_features": dataset["n_features"],
            "n_classes": dataset["n_classes"],
        })
        validate(dataset, log_to_mlflow=True)

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
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

        # Log baseline so detect_drift_op can download it by run_id
        mlflow.log_artifact("/app/data_cache/baseline_X_train.npy", artifact_path="baseline")

        run_id = run.info.run_id

    print(f"run_id={run_id}  metrics={metrics}")
    return run_id


@dsl.component(base_image=IMAGE)
def register_model_op(mlflow_uri: str, run_id: str) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    MODEL_NAME = "k8s-failure-classifier"
    mlflow.set_tracking_uri(mlflow_uri)
    result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    version = str(result.version)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "staging", version)
    print(f"Registered '{MODEL_NAME}' v{version} → @staging")
    return version


@dsl.component(base_image=IMAGE)
def promote_model_op(mlflow_uri: str, version: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    MODEL_NAME = "k8s-failure-classifier"
    mlflow.set_tracking_uri(mlflow_uri)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
    print(f"Promoted '{MODEL_NAME}' v{version} → @production")


@dsl.component(base_image=IMAGE)
def detect_drift_op(mlflow_uri: str, train_run_id: str) -> int:
    """Download baseline from train run artifact and run KS drift test."""
    import os
    import shutil
    import tempfile

    import mlflow
    import numpy as np
    from scipy import stats

    from data.loader import FEATURE_NAMES, load

    mlflow.set_tracking_uri(mlflow_uri)

    # Download baseline saved by train_model_op
    with tempfile.TemporaryDirectory() as tmp:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=train_run_id,
            artifact_path="baseline/baseline_X_train.npy",
            dst_path=tmp,
        )
        X_baseline = np.load(local_path)

    dataset = load()
    X_new = dataset["X_test"]
    DRIFT_THRESHOLD = 0.05

    mlflow.set_experiment("k8s-failure-detection-monitor")
    drifted = []

    with mlflow.start_run(run_name="drift-check"):
        for i, name in enumerate(FEATURE_NAMES):
            stat, p_value = stats.ks_2samp(X_baseline[:, i], X_new[:, i])
            mlflow.log_metric(f"ks_{name}", stat)
            mlflow.log_metric(f"pval_{name}", p_value)
            if p_value < DRIFT_THRESHOLD:
                drifted.append(name)
        mlflow.log_metric("n_drifted_features", len(drifted))
        mlflow.log_param("drifted_features", ",".join(drifted) if drifted else "none")

    print(f"Drift: {len(drifted)}/{len(FEATURE_NAMES)} features  →  {drifted or 'none'}")
    return len(drifted)
