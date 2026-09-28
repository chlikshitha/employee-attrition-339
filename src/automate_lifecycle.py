import mlflow
client = mlflow.MlflowClient()
model_name = "Employee_Attrition_Production_Model"
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    print("[INFO] No registered model versions found.")
else:
    latest = max(versions, key=lambda v: int(v.version))
    print(f"[INFO] Latest registered version: {latest.version}")
    print(f"[INFO] Current stage: {latest.current_stage}")
    print("[SUCCESS] Model lifecycle check completed.")
