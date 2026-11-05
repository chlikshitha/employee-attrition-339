"""Small FastAPI wrapper around the persisted Employee Attrition pipeline."""
from __future__ import annotations

from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.attrition_common import MODEL_PATH
from src.lab10_inference import predict

app = FastAPI(title="Employee Attrition API", version="1.0.0")


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL_PATH.is_file() else "model_not_ready"}


@app.post("/predict")
def predict_attrition(request: PredictionRequest) -> dict[str, list[dict[str, Any]]]:
    try:
        result = predict(pd.DataFrame(request.records))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": result.to_dict(orient="records")}
