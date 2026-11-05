"""Lab 9: controlled validation failure scenarios that never alter source data."""
from __future__ import annotations

import json
import pandas as pd

from src.attrition_common import REPORTS, load_data, validate_frame, write_json


def run() -> dict:
    good = load_data()
    scenarios = []
    for label, bad in [
        ("missing_target", good.drop(columns=["Attrition"])),
        ("empty_dataset", good.iloc[0:0]),
        ("invalid_target", good.assign(Attrition="Maybe")),
        ("missing_target_value", good.assign(Attrition=pd.NA)),
    ]:
        try:
            validate_frame(bad)
        except ValueError as exc:
            scenarios.append({"scenario": label, "rejected": True, "reason": str(exc)})
        else:
            scenarios.append({"scenario": label, "rejected": False, "reason": "Input unexpectedly accepted"})
    report = {"project": "Employee Attrition", "scenarios": scenarios,
              "all_invalid_inputs_rejected": all(item["rejected"] for item in scenarios)}
    write_json(REPORTS / "lab9_failure_report.json", report)
    if not report["all_invalid_inputs_rejected"]:
        raise RuntimeError("One or more injected invalid datasets were not rejected")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
