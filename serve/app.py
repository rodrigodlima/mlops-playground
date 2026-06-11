"""FastAPI inference service — loads model from MLflow Registry (production alias)."""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predict import MODEL_ALIAS, MODEL_NAME, load_model, predict

app = FastAPI(title="k8s-failure-mlops", version="0.1.0")


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ...,
        description=(
            "List of 10-feature rows: [restart_count, cpu_usage_pct, memory_usage_pct, "
            "pod_ready, last_exit_code, waiting_reason (ordinal), oom_killed_count, "
            "image_pull_errors, failed_scheduling_events, readiness_probe_failures]. "
            "waiting_reason encoding: CrashLoopBackOff=0, ErrImagePull=1, Error=2, "
            "ImagePullBackOff=3, None=4, OOMKilled=5, Pending=6, Running=7, Unschedulable=8"
        ),
        examples=[[[0, 20.0, 30.0, 1, 0, 7, 0, 0, 0, 0]]],
    )


class Prediction(BaseModel):
    class_id: int
    class_name: str
    proba: dict[str, float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_name: str = MODEL_NAME
    model_alias: str = MODEL_ALIAS


@app.get("/health")
def health() -> dict:
    try:
        load_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok", "model": MODEL_NAME, "alias": MODEL_ALIAS}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest) -> PredictResponse:
    if not req.instances:
        raise HTTPException(status_code=400, detail="instances must be non-empty")
    try:
        results = predict(req.instances)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(predictions=results)
