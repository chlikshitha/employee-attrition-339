import numpy as np
import pandas as pd

from src.lab1_data_preparation import prepare_frame
from src.lab2_baseline_model import train_baseline


def test_lab1_cleans_duplicates_missing_values_and_labels():
    frame = pd.DataFrame(
        {
            "Attrition": [" yes ", "No", "Yes", "No", "Yes", "No", "Yes"],
            "Age": [30, 40, np.nan, 50, 35, 45, 60],
            "Department": ["Sales", "HR", None, "HR", "R&D", "R&D", "Sales"],
            "YearsAtCompany": [2, 8, 1, 15, 3, 9, 6],
            "YearsInCurrentRole": [1, 5, 0, 8, 2, 4, 3],
        }
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    prepared, report = prepare_frame(frame)

    assert len(prepared) == len(frame) - 1
    assert report["duplicate_rows_removed"] == 1
    assert prepared.isna().sum().sum() == 0
    assert set(prepared["Attrition"]) == {"Yes", "No"}
    assert report["missing_values_after"]["Age"] == 0
    assert report["missing_values_after"]["Department"] == 0


def test_lab2_trains_and_saves_model_metrics_and_predictions(tmp_path):
    rows = 60
    frame = pd.DataFrame(
        {
            "Attrition": ["Yes" if index % 2 else "No" for index in range(rows)],
            "Age": [25 + (index % 35) for index in range(rows)],
            "Department": ["Sales" if index % 3 else "Research & Development" for index in range(rows)],
            "YearsAtCompany": [index % 12 for index in range(rows)],
            "YearsInCurrentRole": [index % 6 for index in range(rows)],
        }
    )

    model_path = tmp_path / "lab2_model.joblib"
    metrics_path = tmp_path / "lab2_metrics.json"
    predictions_path = tmp_path / "lab2_predictions.csv"
    report = train_baseline(
        frame,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert predictions_path.is_file()
    assert report["model"] == "DecisionTreeClassifier"
    assert set(report["metrics"]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert len(pd.read_csv(predictions_path)) == report["test_rows"]