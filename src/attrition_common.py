"""Shared, project-local utilities for the Employee Attrition labs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "employee_attrition.csv"
MODEL_PATH = ROOT / "models" / "employee_attrition_pipeline.joblib"
REPORTS = ROOT / "reports"
TARGET = "Attrition"
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours", "MonthlyIncome"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Employee Attrition data not found: {path}")
    frame = pd.read_csv(path)
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    required = {TARGET, "Age", "YearsAtCompany", "YearsInCurrentRole", "Attrition"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required Employee Attrition columns: {missing}")
    if frame.empty:
        raise ValueError("Employee Attrition dataset is empty")
    if frame[TARGET].isna().any():
        raise ValueError("Attrition target contains missing values")
    unexpected = sorted(set(frame[TARGET].dropna().astype(str)) - {"Yes", "No"})
    if unexpected:
        raise ValueError(f"Unexpected Attrition labels: {unexpected}")


def make_estimator(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("balance", RandomOverSampler(random_state=42)),
        ("classifier", LogisticRegression(C=0.01, solver="liblinear", max_iter=2000, random_state=42)),
    ])


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore").copy()
    if {"YearsInCurrentRole", "YearsAtCompany"}.issubset(features.columns):
        features["CurrentRoleTenureRatio"] = features["YearsInCurrentRole"] / (features["YearsAtCompany"] + 1)
    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(features.columns):
        features["PromotionGapRatio"] = features["YearsSinceLastPromotion"] / (features["YearsAtCompany"] + 1)
    if "YearsAtCompany" in features.columns:
        groups = pd.cut(
            features["YearsAtCompany"], bins=[-1, 2, 5, 10, 20, float("inf")],
            labels=["0-2 Years", "3-5 Years", "6-10 Years", "11-20 Years", "20+ Years"],
        )
        features["TenureGroup"] = groups.map({
            "0-2 Years": 0, "3-5 Years": 1, "6-10 Years": 2,
            "11-20 Years": 3, "20+ Years": 4,
        }).astype("float64")
    return features


def features_target(frame: pd.DataFrame):
    y = frame[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    x = model_features(frame)
    return x, y


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
