import subprocess
import sys
result = subprocess.run([sys.executable, "src/preprocess.py"])
if result.returncode != 0:
    raise SystemExit(result.returncode)
print("[SUCCESS] Production preprocessing pipeline completed.")
