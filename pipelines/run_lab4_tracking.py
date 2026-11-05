import subprocess
import sys


def run_lab4_tracking():
    print("Running Lab 4 Employee Attrition preprocessing and MLflow tracking...")
    for script in ("src/preprocess.py", "src/train_mlflow.py"):
        print(f"[INFO] Executing {script}...")
        result = subprocess.run([sys.executable, script], check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("Lab 4 Employee Attrition MLflow tracking completed.")


if __name__ == "__main__":
    run_lab4_tracking()
