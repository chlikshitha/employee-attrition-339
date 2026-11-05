import asyncio
import httpx
import pandas as pd
import pytest

from src.attrition_common import features_target, make_estimator, validate_frame
from src.lab12_monitoring import _psi
from pipelines.run_lab4_tracking import run_lab4_tracking


def sample_data():
    return pd.DataFrame({
        "Attrition": ["No", "Yes", "No", "Yes", "No", "Yes"],
        "Age": [30, 40, 25, 55, 35, 45],
        "Department": ["Sales", "HR", "Sales", "HR", "R&D", "R&D"],
        "YearsAtCompany": [2, 8, 1, 15, 3, 9],
        "YearsInCurrentRole": [1, 5, 0, 8, 2, 4],
        "YearsSinceLastPromotion": [0, 2, 0, 5, 1, 3],
    })


def test_employee_attrition_schema_and_target_encoding():
    frame = sample_data()
    validate_frame(frame)
    x, y = features_target(frame)
    assert "Attrition" not in x
    assert y.tolist() == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("bad", [
    pd.DataFrame(),
    sample_data().drop(columns="Attrition"),
    sample_data().assign(Attrition="Maybe"),
])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        validate_frame(bad)


def test_pipeline_handles_unseen_categories():
    frame = sample_data()
    x, y = features_target(frame)
    model = make_estimator(x).fit(x.iloc[:4], y.iloc[:4])
    unseen = x.iloc[[4]].copy()
    assert model.predict_proba(unseen).shape == (1, 2)


def test_psi_is_zero_for_same_distribution():
    values = pd.Series([1, 2, 3, 4, 5])
    assert _psi(values, values) == pytest.approx(0.0)


def test_api_returns_employee_attrition_predictions(monkeypatch):
    from api.app import app

    monkeypatch.setattr("api.app.predict", lambda frame: pd.DataFrame({
        "attrition_prediction": [1], "attrition_probability": [0.75]
    }))
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/predict", json={"records": [{"Age": 39}]})

    response = asyncio.run(send_request())
    assert response.status_code == 200
    assert response.json()["predictions"][0]["attrition_prediction"] == 1


def test_lab4_preprocesses_before_mlflow_tracking(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        calls.append(command[-1])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("pipelines.run_lab4_tracking.subprocess.run", fake_run)
    run_lab4_tracking()

    assert calls == ["src/preprocess.py", "src/train_mlflow.py"]


def test_lab4_stops_if_preprocessing_fails(monkeypatch):
    calls = []

    def fake_run(command, check=False):
        calls.append(command[-1])
        return type("Result", (), {"returncode": 7})()

    monkeypatch.setattr("pipelines.run_lab4_tracking.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        run_lab4_tracking()

    assert exc_info.value.code == 7
    assert calls == ["src/preprocess.py"]
