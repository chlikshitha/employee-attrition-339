import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run_preprocessing():
    print("[INFO] Starting Employee Attrition Sklearn Pipeline Preprocessing...")

    data_path = "data/raw/employee_attrition.csv"
    df = pd.read_csv(data_path)

    print(f"[INFO] Raw dataset shape: {df.shape}")

    if "Attrition" not in df.columns:
        raise ValueError(
            "Target column 'Attrition' was not found in the dataset."
        )

    constant_columns = [
        column
        for column in ["Over18", "StandardHours", "EmployeeCount"]
        if column in df.columns
    ]

    if constant_columns:
        df = df.drop(columns=constant_columns)
        print(
            f"[INFO] Removed constant columns: {constant_columns}"
        )

    if "MonthlyIncome" in df.columns:
        df = df.drop(columns=["MonthlyIncome"])
        print("[INFO] Removed MonthlyIncome from the feature set.")

    df["Attrition"] = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if df["Attrition"].isna().any():
        raise ValueError(
            "Invalid values found in Attrition target column."
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
        bins=[-1, 1, 3, 6, 10, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"[INFO] Training rows: {len(X_train)}"
    )

    print(
        f"[INFO] Testing rows: {len(X_test)}"
    )

    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print(
        f"[INFO] Numerical features: {len(num_cols)}"
    )

    print(
        f"[INFO] Categorical features: {len(cat_cols)}"
    )

    num_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                num_pipeline,
                num_cols
            ),
            (
                "cat",
                cat_pipeline,
                cat_cols
            )
        ]
    )

    print(
        "[INFO] Fitting preprocessing pipeline "
        "on training data..."
    )

    X_train_final = preprocessor.fit_transform(X_train)

    print(
        "[INFO] Transforming test data..."
    )

    X_test_final = preprocessor.transform(X_test)

    X_train_final = np.asarray(
        X_train_final,
        dtype=np.float64
    )

    X_test_final = np.asarray(
        X_test_final,
        dtype=np.float64
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
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy(dtype=np.int64)
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy(dtype=np.int64)
    )

    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    feature_columns = (
        num_cols + cat_cols
    )

    joblib.dump(
        feature_columns,
        "models/feature_columns.pkl"
    )

    metadata = {
        "dataset_name": "Employee Attrition",
        "target_column": "Attrition",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "removed_constant_columns": constant_columns,
        "removed_features": ["MonthlyIncome"]
        if "MonthlyIncome" in df.columns
        else [],
        "feature_engineering": [
            "CurrentRoleTenureRatio",
            "PromotionGapRatio",
            "TenureGroup"
        ],
        "test_size": 0.2,
        "random_state": 42
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
        f"[INFO] Final training feature shape: "
        f"{X_train_final.shape}"
    )

    print(
        f"[INFO] Final testing feature shape: "
        f"{X_test_final.shape}"
    )

    print(
        "[SUCCESS] Employee Attrition preprocessing "
        "pipeline completed successfully."
    )

    print(
        "[SUCCESS] Preprocessor saved as "
        "models/preprocessor.pkl"
    )


if __name__ == "__main__":
    run_preprocessing()