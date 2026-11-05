"""Lab 10: local batch inference from the persisted Employee Attrition pipeline."""
from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd

from src.attrition_common import DATA_PATH, MODEL_PATH, REPORTS, load_data, model_features, write_json


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Trained model is missing; run python -m src.lab7_pipeline first")
    model = joblib.load(MODEL_PATH)
    features = model_features(frame)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame({"attrition_prediction": model.predict(features).astype(int), "attrition_probability": probabilities})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="CSV with Employee Attrition feature columns")
    parser.add_argument("--output", type=str, default=str(REPORTS / "inference_predictions.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input) if args.input else load_data()
    result = predict(frame)
    out = pd.concat([frame.reset_index(drop=True), result], axis=1)
    output = __import__("pathlib").Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    write_json(REPORTS / "lab10_inference_report.json", {
        "project": "Employee Attrition", "input_rows": len(frame),
        "predicted_positive": int(result["attrition_prediction"].sum()), "output": str(output),
    })
    print(json.dumps({"rows": len(out), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
