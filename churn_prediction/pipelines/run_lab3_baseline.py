import subprocess

print("Running Lab 3 Baseline Model...")

subprocess.run(["python", "src/train.py"], check=True)

print("Lab 3 baseline model training completed.")