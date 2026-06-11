"""K8s failure detection pipeline — definition, compilation, and submission."""
import argparse
from pathlib import Path

from kfp import Client, dsl, compiler

from pipeline.components import (
    detect_drift_op,
    promote_model_op,
    register_model_op,
    train_model_op,
    validate_data_op,
)


PIPELINE_YAML = Path(__file__).resolve().parent / "k8s_failure_pipeline.yaml"


@dsl.pipeline(
    name="k8s-failure-detection",
    description="End-to-end MLOps: validate → train → register → promote → drift detection",
)
def k8s_failure_pipeline(
    mlflow_uri: str = "http://mlflow-server:5000",
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
):
    validate_task = validate_data_op(mlflow_uri=mlflow_uri)
    validate_task.set_display_name("1 · Validate Data")

    train_task = train_model_op(
        mlflow_uri=mlflow_uri,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    ).after(validate_task)
    train_task.set_display_name("2 · Train Model")

    register_task = register_model_op(
        mlflow_uri=mlflow_uri,
        run_id=train_task.output,
    ).after(train_task)
    register_task.set_display_name("3 · Register → @staging")

    promote_task = promote_model_op(
        mlflow_uri=mlflow_uri,
        version=register_task.output,
    ).after(register_task)
    promote_task.set_display_name("4 · Promote → @production")

    drift_task = detect_drift_op(
        mlflow_uri=mlflow_uri,
        train_run_id=train_task.output,
    ).after(promote_task)
    drift_task.set_display_name("5 · Detect Drift")


def compile_pipeline() -> None:
    compiler.Compiler().compile(k8s_failure_pipeline, str(PIPELINE_YAML))
    print(f"Pipeline compiled → {PIPELINE_YAML}")


def run_pipeline(
    kfp_host: str,
    mlflow_uri: str,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    experiment_name: str = "k8s-failure-detection",
) -> None:
    client = Client(host=kfp_host)
    run = client.create_run_from_pipeline_func(
        k8s_failure_pipeline,
        arguments={
            "mlflow_uri": mlflow_uri,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        experiment_name=experiment_name,
        enable_caching=False,
    )
    print(f"Pipeline run submitted: {run.run_id}")
    print(f"Track at: {kfp_host}/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile or submit the KFP pipeline")
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("compile", help="Compile pipeline to YAML")

    run_p = sub.add_parser("run", help="Submit pipeline run to KFP cluster")
    run_p.add_argument("--kfp-host", default="http://localhost:8080")
    run_p.add_argument("--mlflow-uri", default="http://mlflow-server:5000")
    run_p.add_argument("--n-estimators", type=int, default=100)
    run_p.add_argument("--max-depth", type=int, default=3)
    run_p.add_argument("--learning-rate", type=float, default=0.1)

    args = parser.parse_args()

    if args.action == "compile":
        compile_pipeline()
    elif args.action == "run":
        run_pipeline(
            kfp_host=args.kfp_host,
            mlflow_uri=args.mlflow_uri,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    else:
        parser.print_help()
