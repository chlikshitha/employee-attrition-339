import subprocess
import sys
print("Running Lab 3 Employee Attrition Baseline Model...")
for script in ["src/preprocess.py", "src/train.py", "src/evaluate.py"]:
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
print("Lab 3 Employee Attrition baseline model training completed.")
