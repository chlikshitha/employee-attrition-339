from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def prepare_features(df):
    df = df.copy()
    df["CurrentRoleTenureRatio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["TenureGroup"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 20, float("inf")], labels=["0-2 Years", "3-5 Years", "6-10 Years", "11-20 Years", "20+ Years"])
    df = df.drop(columns=["Over18", "StandardHours", "EmployeeCount"], errors="ignore")
    df = df.drop(columns=["MonthlyIncome"], errors="ignore")
    df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})
    df["TenureGroup"] = df["TenureGroup"].map({"0-2 Years": 0, "3-5 Years": 1, "6-10 Years": 2, "11-20 Years": 3, "20+ Years": 4})
    categorical = df.drop(columns=["Attrition"]).select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    prepared = prepare_features(df)
    X = prepared.drop(columns=["Attrition"])
    y = prepared["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    sampler = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    np.save(PROCESSED_DIR / "X_train_scaled.npy", X_train_scaled)
    np.save(PROCESSED_DIR / "X_test_scaled.npy", X_test_scaled)
    np.save(PROCESSED_DIR / "y_train.npy", y_train_balanced.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())
    X_train_balanced.to_csv(PROCESSED_DIR / "X_train_balanced.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    prepared.to_csv(PROCESSED_DIR / "employee_attrition_encoded.csv", index=False)
    joblib.dump(scaler, MODEL_DIR / "attrition_scaler.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")
    print("[SUCCESS] Employee Attrition preprocessing completed.")
    print(f"[INFO] Training rows before balancing: {len(X_train)}")
    print(f"[INFO] Training rows after balancing: {len(X_train_balanced)}")
    print(f"[INFO] Testing rows: {len(X_test)}")
    print(f"[INFO] Features: {X.shape[1]}")

if __name__ == "__main__":
    main()
