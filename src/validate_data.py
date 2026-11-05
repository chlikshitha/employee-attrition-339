from pathlib import Path
import pandas as pd

path = Path("data/raw/employee_attrition.csv")
df = pd.read_csv(path)
required = ["Attrition", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df.empty:
    raise ValueError("Dataset is empty")
if df["Attrition"].isna().any():
    raise ValueError("Attrition contains missing values")
print("[SUCCESS] Employee Attrition data validation passed.")
print(f"[INFO] Rows: {len(df)}")
print(f"[INFO] Columns: {len(df.columns)}")
