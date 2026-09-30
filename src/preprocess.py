import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler


def run_preprocessing():
    print("[INFO] Starting Employee Attrition Preprocessing Pipeline...")

    data_path = "data/raw/employee_attrition.csv"
    df = pd.read_csv(data_path)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    if "Attrition" not in df.columns:
        raise ValueError(
            "Target column 'Attrition' was not found."
        )

    constant_columns = [
        column
        for column in [
            "Over18",
            "StandardHours",
            "EmployeeCount"
        ]
        if column in df.columns
    ]

    if constant_columns:
        df = df.drop(
            columns=constant_columns
        )

        print(
            f"[INFO] Removed constant columns: "
            f"{constant_columns}"
        )

    if "MonthlyIncome" in df.columns:
        df = df.drop(
            columns=["MonthlyIncome"]
        )

        print(
            "[INFO] Removed MonthlyIncome."
        )

    df["Attrition"] = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0
        })
    )

    if df["Attrition"].isna().any():
        raise ValueError(
            "Invalid values found in Attrition column."
        )

    print(
        "[INFO] Creating engineered features..."
    )

    df["CurrentRoleTenureRatio"] = (
        df["YearsInCurrentRole"]
        / (df["YearsAtCompany"] + 1)
    )

    df["PromotionGapRatio"] = (
        df["YearsSinceLastPromotion"]
        / (df["YearsAtCompany"] + 1)
    )

    df["TenureGroup"] = pd.cut(
        df["YearsAtCompany"],
        bins=[
            -1,
            1,
            3,
            6,
            10,
            np.inf
        ],
        labels=[
            0,
            1,
            2,
            3,
            4
        ]
    ).astype(int)

    X = df.drop(
        "Attrition",
        axis=1
    )

    y = df["Attrition"].astype(
        np.int64
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"[INFO] Training rows before balancing: "
        f"{len(X_train)}"
    )

    print(
        f"[INFO] Testing rows: "
        f"{len(X_test)}"
    )

    cat_cols = X_train.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    ).columns.tolist()

    print(
        f"[INFO] Numerical features: "
        f"{len(num_cols)}"
    )

    print(
        f"[INFO] Categorical features: "
        f"{len(cat_cols)}"
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train[num_cols]
    )

    X_test_scaled = scaler.transform(
        X_test[num_cols]
    )

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = ohe.fit_transform(
        X_train[cat_cols]
    )

    X_test_encoded = ohe.transform(
        X_test[cat_cols]
    )

    X_train_final = np.hstack(
        (
            X_train_scaled,
            X_train_encoded
        )
    )

    X_test_final = np.hstack(
        (
            X_test_scaled,
            X_test_encoded
        )
    )

    print(
        f"[INFO] Features before balancing: "
        f"{X_train_final.shape[1]}"
    )

    ros = RandomOverSampler(
        random_state=42
    )

    X_train_balanced, y_train_balanced = ros.fit_resample(
        X_train_final,
        y_train
    )

    print(
        f"[INFO] Training rows after balancing: "
        f"{len(X_train_balanced)}"
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    np.save(
        "data/processed/X_train_final.npy",
        X_train_balanced
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        np.asarray(
            y_train_balanced,
            dtype=np.int64
        )
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy(
            dtype=np.int64
        )
    )

    joblib.dump(
        scaler,
        "models/attrition_scaler.pkl"
    )

    feature_columns = {
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }

    joblib.dump(
        feature_columns,
        "models/feature_columns.pkl"
    )

    metadata = {
        "dataset_name": "Employee Attrition",
        "target_column": "Attrition",
        "train_shape_before_balancing": [
            int(X_train_final.shape[0]),
            int(X_train_final.shape[1])
        ],
        "train_shape_after_balancing": [
            int(X_train_balanced.shape[0]),
            int(X_train_balanced.shape[1])
        ],
        "test_shape": [
            int(X_test_final.shape[0]),
            int(X_test_final.shape[1])
        ],
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "removed_constant_columns": constant_columns,
        "removed_features": [
            "MonthlyIncome"
        ],
        "feature_engineering": [
            "CurrentRoleTenureRatio",
            "PromotionGapRatio",
            "TenureGroup"
        ],
        "balancing_method": "RandomOverSampler",
        "random_state": 42,
        "test_size": 0.2
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        "[SUCCESS] Employee Attrition preprocessing "
        "completed successfully."
    )

    print(
        "[SUCCESS] Scaler saved as "
        "models/attrition_scaler.pkl"
    )

    print(
        "[SUCCESS] Feature metadata saved as "
        "models/feature_columns.pkl"
    )


if __name__ == "__main__":
    run_preprocessing()