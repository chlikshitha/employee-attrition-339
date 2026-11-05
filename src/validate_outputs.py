from pathlib import Path
import json
import numpy as np

required = ["data/processed/X_train_scaled.npy", "data/processed/X_test_scaled.npy", "data/processed/y_train.npy", "data/processed/y_test.npy", "models/attrition_scaler.pkl", "models/feature_columns.pkl"]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise FileNotFoundError(f"Missing pipeline outputs: {missing}")
X_train = np.load(required[0])
X_test = np.load(required[1])
y_train = np.load(required[2])
y_test = np.load(required[3])
report = {"X_train_shape": list(X_train.shape), "X_test_shape": list(X_test.shape), "y_train_shape": list(y_train.shape), "y_test_shape": list(y_test.shape), "feature_count": int(X_train.shape[1])}
Path("artifacts").mkdir(parents=True, exist_ok=True)
with open("artifacts/preprocessing_summary_report.json", "w") as f:
    json.dump(report, f, indent=4)
print("[SUCCESS] Production preprocessing outputs validated.")
print(json.dumps(report, indent=4))
