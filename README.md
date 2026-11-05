# Employee Attrition MLOps

This repository extends the existing Employee Attrition project and preserves its Labs 3–6 scripts and artifacts. It uses the IBM HR Employee Attrition data in `data/raw/employee_attrition.csv`; the model target is `Attrition` (`Yes`/`No`). No Telco Customer Churn code or data is part of this implementation.

## Labs

| Lab | Implementation | GitHub Actions workflow |
|---|---|---|
| 3 | Baseline preprocessing, Logistic Regression, evaluation | `.github/workflows/lab3_baseline.yml` |
| 4 | MLflow experiment tracking | `.github/workflows/lab4_tracking.yml` |
| 5 | Input validation and preprocessing output checks | `.github/workflows/lab5_pipeline.yml` |
| 6 | Employee Attrition model registry and lifecycle report | `.github/workflows/lab6_registry.yml` |
| 7 | Deterministic preprocessing/training, metrics, source hash and packaged model | `.github/workflows/lab7_pipeline.yml` |
| 8 | Pull request/push CI: pytest and Python compilation | `.github/workflows/lab8_ci.yml` |
| 9 | Failure injection for missing/empty/invalid Employee Attrition input | `.github/workflows/lab9_failure_handling.yml` |
| 10 | Batch inference and a FastAPI `/predict` endpoint | `.github/workflows/lab10_inference.yml` |
| 11 | Docker image and Kubernetes manifests for the API | `.github/workflows/lab11_container.yml` |
| 12 | Numeric feature PSI monitoring report | `.github/workflows/lab12_monitoring.yml` |
| 13 | End-to-end local orchestration and report artifact upload | `.github/workflows/lab13_end_to_end.yml` |

**Requirement-source note:** The workspace contains the project code, research paper, and a reconstructed project structure, but no Employee Attrition Lab 7–13 handouts or rubric. The lab names and scope above are therefore the implemented project progression, not a claim that an unavailable course rubric was checked. The unrelated churn archive was not used as implementation input.

## Setup

Use Python 3.11 and run commands from this directory (`MLOPS ASSIGNMENT`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS activation: `source .venv/bin/activate`.

## Run

```powershell
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python pipelines/run_lab6_registry.py
python -m src.lab7_pipeline
python -m src.lab9_failure_tests
python -m src.lab10_inference
python -m src.lab12_monitoring
python -m src.lab12_monitoring --current data/raw/employee_attrition.csv
python -m src.lab13_end_to_end
python -m pytest -q
```

Lab 7 uses the Employee Attrition research paper's engineered tenure features and tuned Logistic Regression (`C=0.01`, `solver=liblinear`), handles class imbalance with random oversampling inside the training pipeline, uses a stratified 80/20 split with `random_state=42`, and persists the full preprocessing/model pipeline to `models/employee_attrition_pipeline.joblib`. Metrics and source-data SHA-256 are written to `reports/lab7_pipeline_report.json`. Inference writes `reports/inference_predictions.csv`; the failure, monitoring, and integration runs write corresponding JSON reports in `reports/`.

Run the API locally after training:

```powershell
python -m src.lab7_pipeline
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API schema. The `/predict` request body is `{"records": [{...employee feature fields...}]}`. Do not send employee names or direct identifiers; the training pipeline drops `EmployeeNumber` and other constant/identifier columns.

Build a local image after running Lab 7:

```powershell
docker build -t employee-attrition-api:local .
docker run --rm -p 8000:8000 employee-attrition-api:local
```

`k8s/deployment.yaml` is a sample manifest. It expects the image to be available to the target cluster; no cluster deployment is performed by CI.

## Monitoring and limits

The Lab 12 monitor reports Population Stability Index for numeric features and uses `0.2` as a screening threshold. The default reference/current data are the same checked-in dataset, so an unchanged-data run should report no drift. This is a demo monitor, not evidence of live production drift or a retraining trigger. A labeled current batch can be passed programmatically to the monitor to calculate recall and F1 if it includes `attrition_prediction`.

GitHub Actions use GitHub-hosted Ubuntu runners and do not require paid cloud services. The Lab 12 schedule is a weekly report run; scheduled runs and all other workflows must be observed in GitHub Actions after push before they can be described as successful. Docker build checks build the image but do not deploy it. Kubernetes deployment, production serving, and ongoing monitoring require an operator-provided environment and remain unverified here.

## Data, artifacts and reproducibility

The raw CSV already exists in the project. Lab 7 records the data hash, row counts, split settings, feature names, and evaluation metrics. MLflow is local by default; registry state is environment-specific. Generated reports, model files, and MLflow run state are local outputs unless intentionally committed. Review the data license and repository visibility before publishing employee-level records or derived artifacts.
