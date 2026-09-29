from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_NAME = "Logistic Regression (baseline)"
OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load("models/attrition_baseline_model.pkl")
    X_test = np.load("data/processed/X_test_scaled.npy")
    y_test = np.load("data/processed/y_test.npy")
    test_features = pd.read_csv("data/processed/X_test.csv")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    with (ARTIFACT_DIR / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # X_test.csv preserves the original feature values and row order; the labels
    # and predictions are aligned with it by the same saved train/test split.
    predictions = test_features.copy()
    predictions.insert(0, "test_row", np.arange(len(y_test)))
    predictions["actual"] = y_test
    predictions["predicted"] = y_pred
    predictions["predicted_probability"] = y_prob
    predictions["error_type"] = np.select(
        [(y_test == 0) & (y_pred == 1), (y_test == 1) & (y_pred == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    predictions.loc[predictions["error_type"] == "false_positive"].to_csv(
        OUTPUT_DIR / "false_positives.csv", index=False
    )
    predictions.loc[predictions["error_type"] == "false_negative"].to_csv(
        OUTPUT_DIR / "false_negatives.csv", index=False
    )

    comparison = pd.DataFrame([{"model": MODEL_NAME, **metrics}])
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    ax = comparison.set_index("model")[list(metrics)].plot(
        kind="bar", figsize=(9, 5), ylim=(0, 1),
        title="Lab 3 Baseline Model Metrics"
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close(ax.figure)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No attrition", "Attrition"],
        yticklabels=["No attrition", "Attrition"],
        xlabel="Predicted label", ylabel="Actual label",
        title=f"Confusion Matrix: {MODEL_NAME}",
    )
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Employees")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix_final_model.png", dpi=150)
    plt.close(fig)

    error_counts = {
        "false_positives": int(np.sum((y_test == 0) & (y_pred == 1))),
        "false_negatives": int(np.sum((y_test == 1) & (y_pred == 0))),
        "true_positives": int(np.sum((y_test == 1) & (y_pred == 1))),
        "true_negatives": int(np.sum((y_test == 0) & (y_pred == 0))),
    }
    summary = {
        "model": MODEL_NAME,
        "test_rows": int(len(y_test)),
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["No attrition", "Attrition"],
            "rows_actual_columns_predicted": matrix.tolist(),
        },
        "error_counts": error_counts,
        "false_positive_rate": float(error_counts["false_positives"] / max(error_counts["false_positives"] + error_counts["true_negatives"], 1)),
        "false_negative_rate": float(error_counts["false_negatives"] / max(error_counts["false_negatives"] + error_counts["true_positives"], 1)),
        "error_csvs": ["false_positives.csv", "false_negatives.csv"],
    }
    with (OUTPUT_DIR / "error_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("[SUCCESS] Employee Attrition baseline evaluation completed.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
