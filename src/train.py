import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def train_model():
    print("=" * 70)
    print("EMPLOYEE ATTRITION MODEL TRAINING")
    print("=" * 70)

    train_path = "data/processed/X_train_final.npy"
    test_path = "data/processed/X_test_final.npy"
    y_train_path = "data/processed/y_train.npy"
    y_test_path = "data/processed/y_test.npy"

    required_files = [
        train_path,
        test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required processed file not found: {file_path}"
            )

    print("[INFO] Loading processed training and testing data...")

    X_train = np.load(train_path)
    X_test = np.load(test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    print(
        f"[INFO] Training data shape: {X_train.shape}"
    )

    print(
        f"[INFO] Testing data shape: {X_test.shape}"
    )

    print(
        f"[INFO] Training target shape: {y_train.shape}"
    )

    print(
        f"[INFO] Testing target shape: {y_test.shape}"
    )

    params = {
        "C": 0.01,
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": 42
    }

    print(
        "[INFO] Initializing Logistic Regression model..."
    )

    model = LogisticRegression(
        **params
    )

    print("[INFO] Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("[SUCCESS] Model training completed.")

    print("[INFO] Generating predictions...")

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    }

    print("\n--- Employee Attrition Model Performance ---")
    print(
        f"Accuracy : {accuracy:.4f}"
    )
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall   : {recall:.4f}"
    )
    print(
        f"F1-Score : {f1:.4f}"
    )
    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )
    print("---------------------------------------------")

    print("\n--- Classification Report ---")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Attrition",
                "Attrition"
            ],
            zero_division=0
        )
    )

    print("--- Confusion Matrix ---")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    model_path = (
        "models/attrition_baseline_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"[SUCCESS] Model saved to: {model_path}"
    )

    metrics_path = (
        "outputs/training_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Metrics saved to: {metrics_path}"
    )

    confusion_matrix_path = (
        "outputs/confusion_matrix.npy"
    )

    np.save(
        confusion_matrix_path,
        cm
    )

    print(
        f"[SUCCESS] Confusion matrix saved to: "
        f"{confusion_matrix_path}"
    )

    print("=" * 70)
    print("[SUCCESS] Employee Attrition training pipeline completed.")
    print("=" * 70)

    return model, metrics


if __name__ == "__main__":
    train_model()