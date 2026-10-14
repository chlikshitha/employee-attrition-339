# Employee Attrition and Customer Churn MLOps

This repository keeps the original Employee Attrition project at the repository root and includes the Customer Churn MLOps project in [`churn_prediction/`](churn_prediction/).

## Employee Attrition

The existing employee attrition data, Labs 3–6 code, notebooks, and project history remain at the root of the repository.

## Customer Churn (Labs 3–9)

Run commands from the Churn project directory:

```powershell
cd churn_prediction
python -m pip install pandas numpy scikit-learn PyYAML joblib mlflow matplotlib xgboost-cpu
python -m unittest discover -s tests -v
python src/mlops_pipeline/churn_prediction_pipeline.py
```

The workflow produces local reports, model artifacts, and predictions. Its CI runs are configured in the repository-root `.github/workflows/` directory because GitHub Actions does not discover workflows nested under a project subdirectory. The existing SQLite MLflow store and generated local artifacts should remain local.
