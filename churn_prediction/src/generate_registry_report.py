import json
import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Telco_Churn_Production_Model"

def generate_registry_report():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    report = {
        "model_name": MODEL_NAME,
        "versions": []
    }

    for version in versions:
        report["versions"].append({
            "version": version.version,
            "run_id": version.run_id,
            "status": version.status,
            "current_stage": version.current_stage
        })

    os.makedirs("artifacts", exist_ok=True)

    report_path = "artifacts/model_registry_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Registry report saved to {report_path}")

if __name__ == "__main__":
    generate_registry_report()