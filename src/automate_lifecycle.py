import json
from pathlib import Path

import mlflow


def automate_champion_challenger():
    """Compare registered Employee Attrition versions without changing stages."""
    client = mlflow.MlflowClient()
    model_name = "Employee_Attrition_Production_Model"
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        report = {"model_name": model_name, "status": "no_registered_versions"}
        print("[INFO] No registered model versions found.")
    else:
        latest = max(versions, key=lambda version: int(version.version))
        production = [version for version in versions if version.current_stage == "Production"]
        champion = max(production, key=lambda version: int(version.version)) if production else None
        latest_run = client.get_run(latest.run_id) if latest.run_id else None
        latest_f1 = latest_run.data.metrics.get("f1") if latest_run else None
        champion_run = client.get_run(champion.run_id) if champion and champion.run_id else None
        champion_f1 = champion_run.data.metrics.get("f1") if champion_run else None
        report = {
            "model_name": model_name,
            "status": "comparison_complete",
            "latest_version": int(latest.version),
            "latest_stage": latest.current_stage,
            "latest_f1": latest_f1,
            "production_version": int(champion.version) if champion else None,
            "production_f1": champion_f1,
            "recommended_action": (
                "review_candidate_for_promotion" if champion is None or
                (latest_f1 is not None and champion_f1 is not None and latest_f1 > champion_f1)
                else "keep_current_production"
            ),
            "registry_changed": False,
        }
        print(f"[INFO] Latest registered version: {latest.version}")
        print(f"[INFO] Current stage: {latest.current_stage}")
        print(f"[INFO] Lifecycle recommendation: {report['recommended_action']}")

    output = Path("artifacts/model_lifecycle_decision.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[SUCCESS] Model lifecycle check completed without changing registry stages.")
    return report


if __name__ == "__main__":
    automate_champion_challenger()
