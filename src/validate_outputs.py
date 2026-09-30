import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run_deterministic_test():
    print("Running Employee Attrition Reproducibility Validation...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = {
        "C": 0.01,
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": 42
    }

    model_1 = LogisticRegression(**params)

    model_1.fit(
        X_train,
        y_train
    )

    predictions_1 = model_1.predict(
        X_test
    )

    score_1 = f1_score(
        y_test,
        predictions_1,
        zero_division=0
    )

    model_2 = LogisticRegression(**params)

    model_2.fit(
        X_train,
        y_train
    )

    predictions_2 = model_2.predict(
        X_test
    )

    score_2 = f1_score(
        y_test,
        predictions_2,
        zero_division=0
    )

    predictions_match = np.array_equal(
        predictions_1,
        predictions_2
    )

    score_match = score_1 == score_2

    is_reproducible = (
        predictions_match and score_match
    )

    report = {
        "test_name": "Employee Attrition Pipeline Reproducibility Validation",
        "model": "Logistic Regression",
        "parameters": params,
        "execution_1_f1": float(score_1),
        "execution_2_f1": float(score_2),
        "predictions_match": bool(predictions_match),
        "scores_match": bool(score_match),
        "is_strictly_reproducible": bool(is_reproducible),
        "status": "PASSED" if is_reproducible else "FAILED"
    }

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    report_path = (
        "artifacts/reproducibility_report.json"
    )

    with open(
        report_path,
        "w"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"Execution 1 F1: {score_1:.6f}"
    )

    print(
        f"Execution 2 F1: {score_2:.6f}"
    )

    print(
        f"Predictions Match: {predictions_match}"
    )

    if is_reproducible:
        print(
            "SUCCESS: Employee Attrition pipeline is reproducible."
        )
    else:
        print(
            "FAILED: Employee Attrition pipeline is non-deterministic."
        )

    print(
        f"Report saved to: {report_path}"
    )


if __name__ == "__main__":
    run_deterministic_test()