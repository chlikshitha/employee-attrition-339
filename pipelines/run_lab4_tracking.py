import subprocess
import sys
print("Running Lab 4 Employee Attrition MLflow Experiment Tracking...")
result = subprocess.run([sys.executable, "src/train_mlflow.py"])
if result.returncode != 0:
    raise SystemExit(result.returncode)
print("Lab 4 Employee Attrition MLflow tracking completed.")
