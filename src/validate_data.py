import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_employee_attrition_schema():
    return pa.DataFrameSchema({
        "Age": Column(pa.Int, Check.between(18, 100)),
        "Attrition": Column(pa.String, Check.isin(["Yes", "No"])),
        "BusinessTravel": Column(
            pa.String,
            Check.isin([
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ])
        ),
        "DailyRate": Column(pa.Int, Check.ge(0)),
        "Department": Column(
            pa.String,
            Check.isin([
                "Sales",
                "Research & Development",
                "Human Resources"
            ])
        ),
        "DistanceFromHome": Column(pa.Int, Check.ge(0)),
        "Education": Column(pa.Int, Check.between(1, 5)),
        "EducationField": Column(pa.String),
        "EmployeeCount": Column(pa.Int, Check.ge(0)),
        "EmployeeNumber": Column(pa.Int, Check.ge(0)),
        "EnvironmentSatisfaction": Column(pa.Int, Check.between(1, 4)),
        "Gender": Column(
            pa.String,
            Check.isin(["Male", "Female"])
        ),
        "JobInvolvement": Column(pa.Int, Check.between(1, 4)),
        "JobLevel": Column(pa.Int, Check.between(1, 5)),
        "JobRole": Column(pa.String),
        "JobSatisfaction": Column(pa.Int, Check.between(1, 4)),
        "MaritalStatus": Column(
            pa.String,
            Check.isin([
                "Single",
                "Married",
                "Divorced"
            ])
        ),
        "MonthlyIncome": Column(pa.Int, Check.ge(0)),
        "NumCompaniesWorked": Column(pa.Int, Check.ge(0)),
        "Over18": Column(
            pa.String,
            Check.isin(["Y"])
        ),
        "OverTime": Column(
            pa.String,
            Check.isin(["Yes", "No"])
        ),
        "PercentSalaryHike": Column(pa.Int, Check.between(0, 100)),
        "PerformanceRating": Column(pa.Int, Check.between(1, 5)),
        "RelationshipSatisfaction": Column(pa.Int, Check.between(1, 4)),
        "StandardHours": Column(pa.Int, Check.ge(0)),
        "StockOptionLevel": Column(pa.Int, Check.between(0, 3)),
        "TotalWorkingYears": Column(pa.Int, Check.ge(0)),
        "TrainingTimesLastYear": Column(pa.Int, Check.ge(0)),
        "WorkLifeBalance": Column(pa.Int, Check.between(1, 4)),
        "YearsAtCompany": Column(pa.Int, Check.ge(0)),
        "YearsInCurrentRole": Column(pa.Int, Check.ge(0)),
        "YearsSinceLastPromotion": Column(pa.Int, Check.ge(0)),
        "YearsWithCurrManager": Column(pa.Int, Check.ge(0))
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")

    schema = get_employee_attrition_schema()

    try:
        schema.validate(df, lazy=True)

        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[
            [
                "schema_context",
                "column",
                "check",
                "failure_case",
                "index"
            ]
        ]

        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)

        report_path = os.path.join(
            "artifacts",
            output_report_name
        )

        failures.to_csv(
            report_path,
            index=False
        )

        print(
            f"[INFO] Detailed failure report saved to '{report_path}'."
        )

        return False


if __name__ == "__main__":
    data_path = "data/raw/employee_attrition.csv"

    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)

        validate_schema(
            raw_df,
            output_report_name="baseline_validation.csv"
        )
    else:
        print(
            f"[ERROR] Target file not found at: {data_path}"
        )