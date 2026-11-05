import subprocess
import sys
print("[INFO] =========================================")
print("[INFO] Starting Lab 5: Employee Attrition Production Data Pipeline")
print("[INFO] =========================================")
for script in ["src/validate_data.py", "src/preprocess_pipeline.py", "src/validate_outputs.py"]:
    print(f"[INFO] ---> Executing {script}...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
print("[SUCCESS] Lab 5 Employee Attrition Production Pipeline fully executed!")
