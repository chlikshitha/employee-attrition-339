"""Lab 7: deterministic end-to-end preprocessing, training, evaluation and packaging."""
from __future__ import annotations

import hashlib
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.attrition_common import DATA_PATH, MODEL_PATH, REPORTS, features_target, load_data, make_estimator, write_json


def run() -> dict:
    frame = load_data()
    x, y = features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_estimator(x_train)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    report = {
        "project": "Employee Attrition",
        "source": str(DATA_PATH.relative_to(DATA_PATH.parents[2])).replace("\\", "/"),
        "source_sha256": source_hash,
        "random_state": 42,
        "test_size": 0.2,
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": list(x.columns),
        "metrics": metrics,
        "model_path": str(MODEL_PATH.relative_to(MODEL_PATH.parents[1])).replace("\\", "/"),
    }
    write_json(REPORTS / "lab7_pipeline_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
