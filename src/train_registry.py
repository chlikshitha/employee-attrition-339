import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_and_register_model():
    print(
        "[INFO] --- Starting Employee Attrition "
        "MLflow Model Registry Run ---"
    )

    mlflow.set_experiment(
        "Employee_Attrition_Prediction"
    )

    try:
        X_train = np.load(
            "data/processed/X_train_final.npy"
        )

        X_test = np.load(
            "data/processed/X_test_final.npy"
        )

        y_train = np.load(
            "data/processed/y_train.npy"
        )

        y_test = np.load(
            "data/processed/y_test.npy"
        )

    except FileNotFoundError:
        print(
            "[ERROR] Processed data not found. "
            "Please run the preprocessing pipeline first."
        )
        return

    params = {
        "C": 0.01,
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": 42
    }

    model_name = (
        "Employee_Attrition_Production_Model"
    )

    with mlflow.start_run(
        run_name="LogisticRegression_Registry"
    ) as run:

        print(
            f"[INFO] MLflow Run ID: {run.info.run_id}"
        )

        mlflow.log_params(
            params
        )

        mlflow.log_param(
            "model_family",
            "LogisticRegression"
        )

        mlflow.log_param(
            "dataset",
            "Employee Attrition"
        )

        mlflow.log_param(
            "target",
            "Attrition"
        )

        print(
            "[INFO] Training Logistic Regression model..."
        )

        model = LogisticRegression(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "[INFO] Evaluating registered model..."
        )

        y_pred = model.predict(
            X_test
        )

        y_proba = model.predict_proba(
            X_test
        )[:, 1]

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test,
                y_proba
            )
        }

        mlflow.log_metrics(
            metrics
        )

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(
            metadata_path
        ):
            print(
                "[INFO] Logging dataset metadata..."
            )

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        preprocessor_path = (
            "models/attrition_scaler.pkl"
        )

        if os.path.exists(
            preprocessor_path
        ):
            print(
                "[INFO] Attaching preprocessing "
                "artifact for lineage..."
            )

            mlflow.log_artifact(
                preprocessor_path,
                artifact_path="preprocessing"
            )

        feature_path = (
            "models/feature_columns.pkl"
        )

        if os.path.exists(
            feature_path
        ):
            mlflow.log_artifact(
                feature_path,
                artifact_path="preprocessing"
            )

        print(
            "[INFO] Registering model in MLflow Model Registry..."
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression_model",
            registered_model_name=model_name
        )

        print(
            f"[SUCCESS] Metrics logged: "
            f"F1 = {metrics['f1']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        print(
            f"[SUCCESS] Model successfully registered "
            f"under name: '{model_name}'"
        )

        print(
            f"[INFO] Run ID: {run.info.run_id}"
        )


if __name__ == "__main__":
    train_and_register_model()