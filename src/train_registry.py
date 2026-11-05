import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train = np.load("data/processed/X_train_scaled.npy")
X_test = np.load("data/processed/X_test_scaled.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")
mlflow.set_experiment("Employee_Attrition_Prediction")
with mlflow.start_run(run_name="Employee_Attrition_Registry_Run"):
    model = LogisticRegression(C=0.01, solver="liblinear", max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0), "f1": f1_score(y_test, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_test, y_prob)}
    mlflow.log_params({"C": 0.01, "solver": "liblinear", "max_iter": 2000, "random_state": 42})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model", registered_model_name="Employee_Attrition_Production_Model")
    print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
    print("[SUCCESS] Model successfully registered under name: 'Employee_Attrition_Production_Model'")
