from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

Path("models").mkdir(parents=True, exist_ok=True)
X_train = np.load("data/processed/X_train_scaled.npy")
X_test = np.load("data/processed/X_test_scaled.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")
mlflow.set_experiment("Employee_Attrition_Prediction")
for run_name in ["Reproducibility_Run_1", "Reproducibility_Run_2"]:
    print(f"--- Starting MLflow Run: {run_name} ---")
    with mlflow.start_run(run_name=run_name):
        params = {"max_iter": 2000, "random_state": 42}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, zero_division=0), "recall": recall_score(y_test, y_pred, zero_division=0), "f1": f1_score(y_test, y_pred, zero_division=0), "roc_auc": roc_auc_score(y_test, y_prob)}
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "models/attrition_mlflow_model.pkl")
        print(f"Metrics logged: F1 = {metrics['f1']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"Run '{run_name}' successfully tracked!")
print("Lab 4 Employee Attrition MLflow tracking completed.")
