# Customer churn prediction MLOps

This repository demonstrates a reproducible customer churn workflow from raw CSV validation through training, MLflow model registration, production inference, quality gates, CI, and failure handling.

## Setup

Use Python 3.11 or newer, install the packages in requirements.txt, and run commands from the repository root. The checked-in config uses a local SQLite MLflow tracking store. Keep the existing mlflow.db to preserve local experiment history. XGBoost uses the CPU-only package variant.

## Run the project

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python src/mlops_pipeline/churn_prediction_pipeline.py

The ordered stages are in src/mlops_pipeline/. The pipeline writes model and preprocessing artifacts, evaluation charts and reports, prediction outputs, and JSON run logs. Lab 9 failure simulations use in-memory corrupted copies and do not replace the source data or model files.

The separate inference sample at data/raw/new_churn.csv is a demonstration batch sampled from the included source dataset with its target removed. Replace it with genuinely new customer records for real inference.

## Structure

- src/: training, validation, inference, lifecycle, and reproducibility code
- pipelines/: Labs 3–6 runners
- notebooks/: exploratory and project implementation notebooks
- docs/: Lab 7–9 reports and MLflow CI setup
- artifacts/, reports/, outputs/, logs/, models/: locally generated deliverables
- .github/workflows/: Lab 8 CI and Lab 9 failure handling workflows

See docs/LAB7_PIPELINE.md, docs/LAB8_CI_REPORT.md, and docs/LAB9_FAILURE_HANDLING_REPORT.md for the lab summaries.
