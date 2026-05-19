"""FastAPI inference service for the iris classifier POC."""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predict import load_model, predict

app = FastAPI(title="mlops-playground", version="0.1.0")


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ...,
        description="List of feature rows: [sepal_len, sepal_wid, petal_len, petal_wid].",
        examples=[[[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]],
    )


class Prediction(BaseModel):
    class_id: int
    class_name: str
    proba: dict[str, float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]


@app.get("/health")
def health() -> dict:
    try:
        load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest) -> PredictResponse:
    if not req.instances:
        raise HTTPException(status_code=400, detail="instances must be non-empty")
    try:
        results = predict(req.instances)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(predictions=results)
