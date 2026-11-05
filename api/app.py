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
    records: list[dict[str, Any]] | dict[str, Any] | None = None

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "records": [{
                    "Age": 41,
                    "BusinessTravel": "Travel_Rarely",
                    "DailyRate": 1102,
                    "Department": "Research & Development",
                    "DistanceFromHome": 1,
                    "Education": 3,
                    "EducationField": "Life Sciences",
                    "EnvironmentSatisfaction": 3,
                    "Gender": "Female",
                    "HourlyRate": 94,
                    "JobInvolvement": 3,
                    "JobLevel": 2,
                    "JobRole": "Research Scientist",
                    "JobSatisfaction": 4,
                    "MaritalStatus": "Married",
                    "MonthlyRate": 19479,
                    "NumCompaniesWorked": 1,
                    "OverTime": "No",
                    "PercentSalaryHike": 11,
                    "PerformanceRating": 3,
                    "RelationshipSatisfaction": 3,
                    "StockOptionLevel": 1,
                    "TotalWorkingYears": 8,
                    "TrainingTimesLastYear": 3,
                    "WorkLifeBalance": 3,
                    "YearsAtCompany": 5,
                    "YearsInCurrentRole": 4,
                    "YearsSinceLastPromotion": 0,
                    "YearsWithCurrManager": 5
                }]
            }
        }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if MODEL_PATH.is_file() else "model_not_ready"}


@app.post("/predict")
def predict_attrition(request: PredictionRequest) -> dict[str, list[dict[str, Any]]]:
    records = request.records
    if isinstance(records, dict):
        records = [records]
    elif records is None:
        records = [request.dict(exclude={"records"})]
    if not records or any(not record for record in records):
        raise HTTPException(status_code=422, detail="Provide employee data in 'records' or as a single employee object")
    try:
        result = predict(pd.DataFrame(records))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": result.to_dict(orient="records")}
