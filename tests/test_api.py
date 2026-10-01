import pytest
from fastapi.testclient import TestClient

from api.main import create_app

READING = {"machine_id": "T1", "machine_type": "M", "air_temperature_k": 300.0,
           "process_temperature_k": 310.0, "rotational_speed_rpm": 1500,
           "torque_nm": 40.0, "tool_wear_min": 100}


@pytest.fixture(scope="module")
def client(trained_bundle_path):
    with TestClient(create_app(trained_bundle_path)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "detail": None}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["model_type"] == "XGBClassifier" and "test_metrics" in body


def test_predict_response_format(client):
    res = client.post("/predict", json=READING)
    assert res.status_code == 200
    body = res.json()
    assert set(["prediction", "failure_probability", "risk_level", "anomaly_status", "top_factors"]) <= set(body)
    assert 0 <= body["failure_probability"] <= 1


@pytest.mark.parametrize("change", [{"machine_type": "X"}, {"torque_nm": -1}, {"rotational_speed_rpm": 0},
                                    {"air_temperature_k": "hot"}])
def test_validation_rejects_bad_input(client, change):
    assert client.post("/predict", json={**READING, **change}).status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in READING.items() if k != "tool_wear_min"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    res = client.post("/predict/batch", json={"readings": [READING, READING]})
    assert res.status_code == 200 and res.json()["count"] == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422


def test_degraded_when_model_missing(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json=READING).status_code == 503
