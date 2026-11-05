# MLflow and CI setup

Local runs use the SQLite URI in config.yaml. GitHub Actions uses the same configured local SQLite store on an ephemeral runner. Do not commit or delete an existing mlflow.db when preserving experiment history.

For a remote tracking server, set the MLFLOW_TRACKING_URI repository or environment variable to the tracking server URI and provide its credentials through GitHub Actions secrets. Keep credentials out of config.yaml. The production model is selected with the configured MLflow model alias.

CI creates a fresh tracking database for each run, so registry versions in CI are isolated from local experiment history.
