# Lab 7: End to end pipeline

The pipeline in src/mlops_pipeline/churn_prediction_pipeline.py runs data loading, raw data validation, deterministic preprocessing, processed output checks, model training and MLflow registration, evaluation, the quality gate, lifecycle alias assignment, inference, and Lab 9 failure simulations.

Run it from the repository root with:

    python src/mlops_pipeline/churn_prediction_pipeline.py

The raw feature splits are preserved in data/processed/; the fitted full pipeline is saved at models/churn_full_pipeline.pkl. Evaluation metrics are written to reports/evaluation_report.json, plots and stage reports to artifacts/, and predictions to outputs/inference_predictions.csv.

The pipeline stops on the first failed stage and records each attempted stage in logs/churn_prediction_pipeline_run_log.json.
