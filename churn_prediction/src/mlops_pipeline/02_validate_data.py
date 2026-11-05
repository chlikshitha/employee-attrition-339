"""Validate required fields, types, ranges, and target values before processing."""
from pathlib import Path
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"customerID","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"}
def validate(df):
    errors=[]; missing=sorted(EXPECTED-set(df.columns)); extra=sorted(set(df.columns)-EXPECTED)
    if missing: errors.append(f"Missing columns: {missing}")
    if extra: errors.append(f"Unexpected columns: {extra}")
    if df.empty: errors.append("Dataset is empty")
    if "customerID" in df and df.customerID.isna().any(): errors.append("customerID contains nulls")
    if "Churn" in df and not set(df.Churn.dropna().astype(str).str.strip()).issubset({"Yes","No"}): errors.append("Churn must contain only Yes/No")
    if "MonthlyCharges" in df:
        v=pd.to_numeric(df.MonthlyCharges,errors="coerce")
        if v.isna().any() or (v<0).any(): errors.append("MonthlyCharges must be numeric and nonnegative")
    if "tenure" in df:
        v=pd.to_numeric(df.tenure,errors="coerce")
        if v.isna().any() or (v<0).any(): errors.append("tenure must be numeric and nonnegative")
    if "SeniorCitizen" in df and not set(pd.to_numeric(df.SeniorCitizen,errors="coerce").dropna().unique()).issubset({0,1}):
        errors.append("SeniorCitizen must contain only 0 or 1")
    for col in ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]:
        if col in df and df[col].isna().any(): errors.append(f"{col} contains nulls")
    return errors
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); p=ROOT/cfg["data"]["raw_path"]
    if not p.exists(): raise FileNotFoundError(f"Raw dataset not found: {p}")
    errors=validate(pd.read_csv(p)); out=ROOT/"artifacts"/"baseline_validation.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"error":errors}).to_csv(out,index=False)
    if errors: raise ValueError("Data validation failed: " + "; ".join(errors))
    print("Raw data validation passed")
if __name__=="__main__": main()
