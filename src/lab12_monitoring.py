"""Lab 12: lightweight drift and model performance monitoring for HR records."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from src.attrition_common import DATA_PATH, DROP_COLUMNS, REPORTS, load_data, validate_frame, write_json


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if not len(reference) or not len(current):
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference)
    b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def run(reference_path=DATA_PATH, current_path=None, current_frame=None) -> dict:
    reference = load_data(reference_path)
    if current_frame is not None:
        current = current_frame.copy()
        validate_frame(current)
    else:
        current = load_data(Path(current_path)) if current_path else reference.copy()
    shared = [c for c in reference.select_dtypes(include="number").columns
              if c in current and c not in DROP_COLUMNS]
    drift = {col: _psi(reference[col], current[col]) for col in shared}
    labels_available = "Attrition" in current and current["Attrition"].notna().all()
    performance = None
    if labels_available and "attrition_prediction" in current:
        y = current["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
        pred = current["attrition_prediction"].astype(int)
        performance = {"recall": float(recall_score(y, pred, zero_division=0)),
                       "f1": float(f1_score(y, pred, zero_division=0))}
    report = {
        "project": "Employee Attrition", "reference_rows": len(reference), "current_rows": len(current),
        "numeric_psi": drift, "max_psi": max(drift.values(), default=0.0),
        "drift_detected": max(drift.values(), default=0.0) >= 0.2,
        "labeled_performance": performance,
        "note": "PSI is a screening signal; this report does not trigger automatic retraining.",
    }
    write_json(REPORTS / "lab12_monitoring_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", help="Optional Employee Attrition CSV batch to compare with the checked-in reference")
    options = parser.parse_args()
    print(json.dumps(run(current_path=options.current), indent=2))
