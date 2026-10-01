import pytest

from src.predict import ModelNotFoundError, Predictor, classify_risk, load_bundle

READING = {"machine_id": "T1", "machine_type": "M", "air_temperature_k": 300.0,
           "process_temperature_k": 310.0, "rotational_speed_rpm": 1500.0,
           "torque_nm": 40.0, "tool_wear_min": 100.0}


def test_load_bundle_missing_file(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


def test_load_bundle_has_expected_keys(trained_bundle_path):
    bundle = load_bundle(trained_bundle_path)
    assert {"model", "anomaly_model", "model_version", "trained_at", "feature_config"} <= set(bundle)


@pytest.mark.parametrize("p,expected", [(0.0, "LOW"), (0.399, "LOW"), (0.40, "MEDIUM"),
                                        (0.699, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")])
def test_risk_thresholds(p, expected):
    assert classify_risk(p, 0.40, 0.70) == expected


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        classify_risk(0.5, 0.8, 0.4)


def test_prediction_format(trained_bundle_path):
    res = Predictor.from_path(trained_bundle_path).predict([READING])[0]
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert res["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert res["anomaly_status"] in {"NORMAL", "ANOMALY"}
    assert isinstance(res["top_factors"], list)


def test_batch_and_missing_field(trained_bundle_path):
    predictor = Predictor.from_path(trained_bundle_path)
    assert len(predictor.predict([READING, READING], explain=False)) == 2
    bad = {k: v for k, v in READING.items() if k != "torque_nm"}
    with pytest.raises(ValueError):
        predictor.predict([bad])
