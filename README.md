# Employee Attrition and Customer Churn MLOps

This repository keeps two independent projects: Employee Attrition at the repository root and the original Customer Churn project under `churn_prediction/`. Each project has its own dataset, model artifacts, scripts, dependencies, and reports. Employee Attrition Labs 3–6 are preserved; Labs 7–13 and their workflows are included at the root.

## Employee Attrition

Uses the IBM HR dataset at `data/raw/employee_attrition.csv` with `Attrition` as the target. Lab 7 aligns with the project research paper's engineered tenure features, tuned Logistic Regression (`C=0.01`, `solver=liblinear`), and class imbalance handling.

| Lab | Implementation | Workflow |
|---|---|---|
| 3 | Baseline preprocessing, training, evaluation | `.github/workflows/lab3_baseline.yml` |
| 4 | MLflow experiment tracking | `.github/workflows/lab4_tracking.yml` |
| 5 | Validated data/preprocessing pipeline | `.github/workflows/lab5_pipeline.yml` |
| 6 | Model registry and lifecycle report | `.github/workflows/lab6_registry.yml` |
| 7 | Reproducible train/evaluation pipeline | `.github/workflows/lab7_pipeline.yml` |
| 8 | CI tests and Python compilation | `.github/workflows/lab8_ci.yml` |
| 9 | Invalid-data failure handling | `.github/workflows/lab9_failure_handling.yml` |
| 10 | Batch inference and FastAPI serving | `.github/workflows/lab10_inference.yml` |
| 11 | Docker image build and Kubernetes manifest | `.github/workflows/lab11_container.yml` |
| 12 | Numeric feature PSI monitoring | `.github/workflows/lab12_monitoring.yml` |
| 13 | End-to-end orchestration and report artifacts | `.github/workflows/lab13_end_to_end.yml` |

The lab scope follows this project's implementation and research paper. Check your course rubric for any additional requirements. Workflow definitions have not been verified by a GitHub Actions run unless stated in the local run reports.

### Setup and run

Run from the repository root with Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python pipelines/run_lab6_registry.py
python -m src.lab7_pipeline
python -m src.lab9_failure_tests
python -m src.lab10_inference
python -m src.lab12_monitoring
python -m src.lab13_end_to_end
python -m pytest -q
```

Linux/macOS activation: `source .venv/bin/activate`.

Lab 7 uses a stratified 80/20 split and fixed random state, records the data SHA-256 and metrics, and saves `models/employee_attrition_pipeline.joblib`. Reports are written under `reports/`. Local MLflow registry state and generated artifacts are environment-specific.

Run the API after creating the Lab 7 model:

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`. Build and run the local image after Lab 7:

```powershell
docker build -t employee-attrition-api:local .
docker run --rm -p 8000:8000 employee-attrition-api:local
```

`k8s/deployment.yaml` is a sample manifest. CI builds the container but does not deploy to a cluster. Lab 12's default reference/current dataset is unchanged input, so it should report no drift; this is not evidence of live production monitoring.

## Customer Churn project

The existing Churn project is retained in [`churn_prediction/`](churn_prediction/README.md). Its GitHub Actions workflows are at the repository root in `.github/workflows/churn_lab8_ci.yml` and `.github/workflows/churn_lab9_failure_tests.yml`.

Run Churn commands from its own directory, with its own environment and dependencies:

```powershell
cd churn_prediction
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/mlops_pipeline/churn_prediction_pipeline.py
```

The Churn project uses its own local MLflow database and outputs. Do not mix its data or generated artifacts with the Employee Attrition project.
