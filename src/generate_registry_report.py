from pathlib import Path
import json
import mlflow
client = mlflow.MlflowClient()
model_name = "Employee_Attrition_Production_Model"
versions = client.search_model_versions(f"name='{model_name}'")
report = {"model_name": model_name, "versions": [{"version": int(v.version), "run_id": v.run_id, "status": v.status, "current_stage": v.current_stage} for v in sorted(versions, key=lambda x: int(x.version))]}
Path("artifacts").mkdir(parents=True, exist_ok=True)
with open("artifacts/model_registry_report.json", "w") as f:
    json.dump(report, f, indent=4)
print("[SUCCESS] Registry report saved to artifacts/model_registry_report.json")
