from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

Path("artifacts").mkdir(parents=True, exist_ok=True)
model = joblib.load("models/attrition_baseline_model.pkl")
X_test = np.load("data/processed/X_test_scaled.npy")
y_test = np.load("data/processed/y_test.npy")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = {"accuracy": float(accuracy_score(y_test, y_pred)), "precision": float(precision_score(y_test, y_pred, zero_division=0)), "recall": float(recall_score(y_test, y_pred, zero_division=0)), "f1": float(f1_score(y_test, y_pred, zero_division=0)), "roc_auc": float(roc_auc_score(y_test, y_prob))}
with open("artifacts/baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("[SUCCESS] Employee Attrition baseline evaluation completed.")
for k in metrics:
    print(f"{k}: {metrics[k]:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))
