"""Lab 2: train and evaluate a standalone Employee Attrition baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.attrition_common import ROOT, TARGET, validate_frame, write_json
from src.lab1_data_preparation import PREPARED_DATA_PATH

MODEL_PATH = ROOT / "models" / "lab2_baseline_model.joblib"
METRICS_PATH = ROOT / "reports" / "lab2_baseline_metrics.json"
PREDICTIONS_PATH = ROOT / "outputs" / "lab2_baseline_predictions.csv"


def train_baseline(
    frame: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> dict:
    """Fit a simple decision-tree baseline and save isolated Lab 2 artifacts."""
    validate_frame(frame)
    features = frame.drop(columns=[TARGET]).copy()
    labels = frame[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(transformers, remainder="drop")
    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test).astype("int8")
    probabilities = estimator.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    model_destination = Path(model_path)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_destination)

    metrics_report = {
        "project": "Employee Attrition",
        "lab": 2,
        "model": "DecisionTreeClassifier",
        "parameters": {"max_depth": 5, "random_state": 42},
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": [str(column) for column in features.columns],
        "random_state": 42,
        "test_size": 0.2,
        "metrics": metrics,
        "model_path": str(model_destination),
    }
    write_json(Path(metrics_path), metrics_report)

    prediction_destination = Path(predictions_path)
    prediction_destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "actual_attrition": y_test.to_numpy(dtype="int8"),
            "predicted_attrition": predictions,
            "attrition_probability": probabilities,
        }
    ).to_csv(prediction_destination, index=False)
    return metrics_report


def run(
    prepared_data_path: Path = PREPARED_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
) -> dict:
    source = Path(prepared_data_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {source}. Run python -m src.lab1_data_preparation first."
        )
    report = train_baseline(
        pd.read_csv(source),
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
    report["prepared_data_path"] = str(source)
    write_json(Path(metrics_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PREPARED_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.model, args.metrics, args.predictions), indent=2))


if __name__ == "__main__":
    main()