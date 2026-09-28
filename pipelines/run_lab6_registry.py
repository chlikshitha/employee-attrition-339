import subprocess
import sys
print("[INFO] =========================================")
print("[INFO] Starting Lab 6: Employee Attrition Model Registry Pipeline")
print("[INFO] =========================================")
for script in ["src/train_registry.py", "src/automate_lifecycle.py", "src/generate_registry_report.py"]:
    print(f"[INFO] ---> Executing {script}...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
print("[SUCCESS] Lab 6 Employee Attrition Model Registry Pipeline fully executed!")
