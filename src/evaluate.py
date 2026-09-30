import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def run_evaluation():
    print("Starting Employee Attrition Model Evaluation...")

    os.makedirs("outputs", exist_ok=True)

    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")

    model = joblib.load("models/attrition_baseline_model.pkl")

    y_pred = model.predict(X_test_final)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_final)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = 0.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Employee Attrition Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("---------------------------------------------\n")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No", "Actual_Yes"],
        columns=["Predicted_No", "Predicted_Yes"]
    )

    cm_df.to_csv(
        "outputs/confusion_matrix.csv"
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC"
            ],
            "Value": [
                acc,
                prec,
                rec,
                f1,
                roc_auc
            ]
        }
    )

    metrics_df.to_csv(
        "outputs/evaluation_metrics.csv",
        index=False
    )

    try:
        df = pd.read_csv(
            "data/raw/employee_attrition.csv"
        )

        target_column = "Attrition"

        _, X_test_raw = __import__(
            "sklearn.model_selection",
            fromlist=["train_test_split"]
        ).train_test_split(
            df.drop(target_column, axis=1),
            test_size=0.2,
            random_state=42,
            stratify=df[target_column]
        )

        errors_df = X_test_raw.reset_index(drop=True)

        actual_values = pd.Series(y_test).reset_index(drop=True)
        predicted_values = pd.Series(y_pred).reset_index(drop=True)

        errors_df["Actual_Attrition"] = actual_values
        errors_df["Predicted_Attrition"] = predicted_values

        false_negatives = errors_df[
            (errors_df["Actual_Attrition"] == 1)
            & (errors_df["Predicted_Attrition"] == 0)
        ]

        false_positives = errors_df[
            (errors_df["Actual_Attrition"] == 0)
            & (errors_df["Predicted_Attrition"] == 1)
        ]

        false_negatives.to_csv(
            "outputs/false_negatives.csv",
            index=False
        )

        false_positives.to_csv(
            "outputs/false_positives.csv",
            index=False
        )

        print(
            f"False Negatives: {len(false_negatives)}"
        )

        print(
            f"False Positives: {len(false_positives)}"
        )

    except Exception as exc:
        print(
            f"[WARNING] Error analysis export failed: {exc}"
        )

    print(
        "Evaluation complete and output files saved."
    )


if __name__ == "__main__":
    run_evaluation()