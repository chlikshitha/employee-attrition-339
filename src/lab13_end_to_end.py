"""Lab 13: orchestrate validation, deterministic training, inference and monitoring."""
from __future__ import annotations

import json

from src.lab7_pipeline import run as run_pipeline
from src.lab9_failure_tests import run as run_failure_checks
from src.lab10_inference import predict
from src.lab12_monitoring import run as run_monitoring
from src.attrition_common import REPORTS, load_data, write_json


def run() -> dict:
    pipeline = run_pipeline()
    failures = run_failure_checks()
    frame = load_data()
    # Smoke-test an unseen-from-training row without reporting in-sample metrics.
    predictions = predict(frame.iloc[:1])
    monitoring = run_monitoring()
    result = {
        "project": "Employee Attrition", "stages": {
            "pipeline": "passed", "failure_checks": "passed" if failures["all_invalid_inputs_rejected"] else "failed",
            "inference": "passed" if len(predictions) == 1 else "failed", "monitoring": "passed",
        },
        "metrics": pipeline["metrics"], "monitoring_drift_detected": monitoring["drift_detected"],
        "deployment_status": "not_deployed; no deployment target configured",
    }
    write_json(REPORTS / "lab13_end_to_end_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
