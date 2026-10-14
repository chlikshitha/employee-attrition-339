# Lab 8: Continuous integration

The lab8_ci.yml workflow validates the source, runs the unit tests, processes the checked-in churn data, trains and evaluates the model, and enforces the configured quality threshold. Its MLflow database is created inside the GitHub runner and is not uploaded as a project artifact.

The gate compares the metric configured in config.yaml with quality_gate.minimum_value. A failure exits nonzero and prevents later stages from running.
