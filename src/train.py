from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

Path("models").mkdir(parents=True, exist_ok=True)
X_train = np.load("data/processed/X_train_scaled.npy")
y_train = np.load("data/processed/y_train.npy")
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/attrition_baseline_model.pkl")
print("[SUCCESS] Employee Attrition baseline model trained.")
print("[INFO] Model: Logistic Regression")
print("[INFO] max_iter=2000, random_state=42")
