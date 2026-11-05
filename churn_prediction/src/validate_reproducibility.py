import os
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "train_registry.py"


def get_mlflow_config():
    config_path = PROJECT_ROOT / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    return config


def run_training():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        print("[ERROR] Training run failed.")
        sys.exit(1)


def get_latest_run_metrics(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"[ERROR] MLflow experiment not found: {experiment_name}")
        sys.exit(1)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        print("[ERROR] No MLflow runs found.")
        sys.exit(1)

    metrics = {
        key.removeprefix("metrics."): value
        for key, value in runs.iloc[0].to_dict().items()
        if key.startswith("metrics.")
    }

    if not metrics:
        print("[ERROR] No metrics found in the latest MLflow run.")
        sys.exit(1)

    return metrics


def main():
    if not TRAIN_SCRIPT.exists():
        print(f"[ERROR] Training script not found: {TRAIN_SCRIPT}")
        sys.exit(1)

    config = get_mlflow_config()
    experiment_name = config["mlflow"]["experiment_name"]

    print("[INFO] Starting reproducibility validation...")

    print("[INFO] First training run...")
    run_training()
    first_metrics = get_latest_run_metrics(experiment_name)

    print("[INFO] Second training run...")
    run_training()
    second_metrics = get_latest_run_metrics(experiment_name)

    print("[INFO] First run metrics:", first_metrics)
    print("[INFO] Second run metrics:", second_metrics)

    if first_metrics == second_metrics:
        print("[SUCCESS] Metrics are reproducible.")
        sys.exit(0)

    print("[FAILURE] Metrics differ between runs.")
    sys.exit(1)


if __name__ == "__main__":
    import runpy
    runpy.run_path(str(PROJECT_ROOT / "src/mlops_pipeline/reproducibility.py"), run_name="__main__")
