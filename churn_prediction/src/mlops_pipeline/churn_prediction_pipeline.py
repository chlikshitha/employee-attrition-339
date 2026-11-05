import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


STAGES = [
    ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Reproducibility Validation", "src/mlops_pipeline/reproducibility.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py"),
    ("Failure Injection", "src/mlops_pipeline/10_failure_injection.py"),
]


def run_pipeline(include_failure_tests=True):
    print("[INFO] ==================================================")
    print("[INFO] Starting Churn Prediction MLOps Pipeline")
    print("[INFO] ==================================================")

    stages = STAGES if include_failure_tests else STAGES[:-1]
    run_log = {"pipeline": "Churn Prediction MLOps", "stages": []}

    for stage_name, script_path in stages:
        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        result = subprocess.run([sys.executable, str(os.path.join(ROOT, script_path))], cwd=ROOT)
        status = "PASSED" if result.returncode == 0 else "FAILED"

        run_log["stages"].append(
            {
                "stage": stage_name,
                "status": status,
                "exit_code": result.returncode,
            }
        )

        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            break

    all_passed = len(run_log["stages"]) == len(stages) and all(
        stage["status"] == "PASSED" for stage in run_log["stages"]
    )
    run_log["overall_status"] = "PASSED" if all_passed else "FAILED"

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(ROOT, "logs", "churn_prediction_pipeline_run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)
    with open(os.path.join(ROOT, "logs", "pipeline_run_log.json"), "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if all_passed:
        print("[SUCCESS] Churn Prediction MLOps Pipeline completed successfully.")
    else:
        print("[ERROR] Churn Prediction MLOps Pipeline failed.")
    print(f"[INFO] Run log: {os.path.relpath(log_path, ROOT)}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = run_pipeline(include_failure_tests="--through-lab8" not in sys.argv[1:])
    sys.exit(0 if passed else 1)
