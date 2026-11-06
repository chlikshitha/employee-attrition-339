"""Lab 1: explore, clean, and prepare the Employee Attrition dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.attrition_common import DATA_PATH, ROOT, TARGET, validate_frame, write_json

PREPARED_DATA_PATH = ROOT / "data" / "processed" / "lab1_employee_attrition_cleaned.csv"
REPORT_PATH = ROOT / "reports" / "lab1_data_preparation_report.json"


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy and an exploration/cleaning report."""
    if frame.empty:
        raise ValueError("Employee Attrition dataset is empty")
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")

    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if len(set(cleaned.columns)) != len(cleaned.columns):
        raise ValueError("Column names are duplicated after trimming whitespace")

    shape_before = list(cleaned.shape)
    missing_before = {str(column): int(count) for column, count in cleaned.isna().sum().items()}
    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    for column in cleaned.select_dtypes(include=["object", "string", "category"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    # Labels cannot be imputed; drop rows without a target and normalize Yes/No.
    missing_target_rows = int(cleaned[TARGET].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET]).copy()
    target_values = cleaned[TARGET].astype("string").str.strip().str.casefold()
    cleaned[TARGET] = target_values.replace({"yes": "Yes", "no": "No"})

    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()
    for column in numeric_columns:
        median = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(0 if pd.isna(median) else median)

    categorical_columns = [
        column for column in cleaned.columns
        if column not in numeric_columns
    ]
    for column in categorical_columns:
        if cleaned[column].isna().any():
            modes = cleaned[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)

    validate_frame(cleaned)
    missing_after = {str(column): int(count) for column, count in cleaned.isna().sum().items()}

    numeric_summary = {}
    for column in cleaned.select_dtypes(include="number").columns:
        numeric_summary[str(column)] = {
            str(statistic): float(value)
            for statistic, value in cleaned[column].describe().items()
            if pd.notna(value)
        }
    categorical_summary = {
        str(column): {
            "unique_values": int(cleaned[column].nunique(dropna=True)),
            "top_values": {
                str(value): int(count)
                for value, count in cleaned[column].value_counts(dropna=True).head(5).items()
            },
        }
        for column in cleaned.select_dtypes(exclude="number").columns
        if column != TARGET
    }
    report = {
        "project": "Employee Attrition",
        "lab": 1,
        "input_shape": shape_before,
        "prepared_shape": list(cleaned.shape),
        "duplicate_rows_removed": duplicate_rows,
        "rows_removed_without_target": missing_target_rows,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "target_distribution": {
            str(label): int(count)
            for label, count in cleaned[TARGET].value_counts().sort_index().items()
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "cleaning": [
            "Removed exact duplicate rows",
            "Trimmed whitespace in text columns",
            "Removed rows without an Attrition target",
            "Normalized Attrition labels to Yes/No",
            "Filled numeric missing values with the column median (zero when entirely missing)",
            "Filled categorical missing values with the mode (Unknown when entirely missing)",
        ],
    }
    return cleaned, report


def run(
    input_path: Path = DATA_PATH,
    output_path: Path = PREPARED_DATA_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    source = Path(input_path)
    cleaned, report = prepare_frame(pd.read_csv(source))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)
    report["input_path"] = str(source)
    report["prepared_data_path"] = str(destination)
    write_json(Path(report_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=PREPARED_DATA_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()