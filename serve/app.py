"""FastAPI inference service — loads model from MLflow Registry (Production stage)."""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.predict import MODEL_ALIAS, MODEL_NAME, load_model, predict

app = FastAPI(title="wine-quality-mlops", version="0.1.0")


class PredictRequest(BaseModel):
    instances: List[List[float]] = Field(
        ...,
        description="List of 13-feature rows: [alcohol, malic_acid, ash, alcalinity_of_ash, "
                    "magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, "
                    "color_intensity, hue, od280_od315, proline]",
        examples=[[[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]],
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
