import os
import json
import mlflow
from mlflow.tracking import MlflowClient


def generate_registry_report():
    print("[INFO] Generating Employee Attrition Model Registry Report...")

    client = MlflowClient()
    model_name = "Employee_Attrition_Production_Model"

    try:
        registered_models = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not registered_models:
            print(
                f"[ERROR] No registered versions found for {model_name}"
            )
            return

        production_models = [
            mv
            for mv in registered_models
            if mv.current_stage == "Production"
        ]

        if production_models:
            champion = max(
                production_models,
                key=lambda mv: int(mv.version)
            )
        else:
            champion = max(
                registered_models,
                key=lambda mv: int(mv.version)
            )

            print(
                "[WARNING] No Production model found. "
                f"Using latest registered version {champion.version}."
            )

        if not champion.run_id:
            print(
                f"[ERROR] Version {champion.version} has no associated run."
            )
            return

        run = client.get_run(champion.run_id)

        metrics = run.data.metrics
        params = run.data.params

        report = {
            "registry_status": (
                "READY_FOR_DEPLOYMENT"
                if champion.current_stage == "Production"
                else "REGISTERED_FOR_VALIDATION"
            ),
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source
            },
            "performance_metrics": {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc")
            },
            "hyperparameters": params,
            "preprocessing_dependency": {
                "scaler": "models/attrition_scaler.pkl",
                "feature_columns": "models/feature_columns.pkl",
                "preprocessing_pipeline": "src/preprocess.py"
            },
            "deployment_information": {
                "model_type": "Logistic Regression",
                "target_variable": "Attrition",
                "task": "Employee Attrition Classification"
            }
        }

        os.makedirs("artifacts", exist_ok=True)

        report_path = "artifacts/production_model_report.json"

        with open(report_path, "w") as file:
            json.dump(report, file, indent=4)

        print(
            f"[SUCCESS] Registry report generated for "
            f"Version {champion.version}"
        )

        print(
            f"[INFO] Model stage: {champion.current_stage}"
        )

        print(
            f"[INFO] F1 Score: "
            f"{metrics.get('f1', 0.0):.4f}"
        )

        print(
            f"[INFO] ROC-AUC: "
            f"{metrics.get('roc_auc', 0.0):.4f}"
        )

        print(
            f"[INFO] Saved to: {report_path}"
        )

    except Exception as exc:
        print(
            f"[ERROR] Failed to generate registry report: {exc}"
        )


if __name__ == "__main__":
    generate_registry_report()