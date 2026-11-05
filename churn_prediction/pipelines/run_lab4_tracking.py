import subprocess
import sys

print("Running Lab 4 MLflow Experiment Tracking...")

subprocess.run([sys.executable, "src/train_mlflow.py"], check=True)

print("Lab 4 MLflow tracking completed.")