# Lab 9: Failure handling and recovery

The failure suite runs three configured repetitions of five cases: missing input, invalid numeric data, schema mismatch, invalid preprocessed values, and production model retrieval. Corruptions are created in memory; the suite does not replace data/raw/churn.csv, processed arrays, model files, or the MLflow database.

Results are written to reports/failure_report.json. Each case records its expected outcome and whether that outcome was observed. The recovery case loads the registered production alias and verifies that it can predict from a held-out raw row.

Run the complete suite after the normal pipeline has produced a production alias:

    python src/mlops_pipeline/10_failure_injection.py
