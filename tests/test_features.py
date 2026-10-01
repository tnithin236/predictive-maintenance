import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_engineered_features, add_time_series_features


def test_engineered_values(sample_df):
    out = add_engineered_features(sample_df)
    r = out.iloc[0]
    assert r["temperature_diff_k"] == pytest.approx(r["process_temperature_k"] - r["air_temperature_k"])
    assert r["power_w"] == pytest.approx(r["torque_nm"] * r["rotational_speed_rpm"] * 2 * np.pi / 60)
    assert r["wear_torque_product"] == pytest.approx(r["tool_wear_min"] * r["torque_nm"])


def test_does_not_mutate_input(sample_df):
    before = sample_df.copy()
    add_engineered_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, before)


def test_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        add_engineered_features(sample_df.drop(columns=["torque_nm"]))


def test_time_series_features_are_per_machine_and_past_only():
    df = pd.DataFrame({
        "machine_id": ["A"] * 4 + ["B"] * 2,
        "timestamp": pd.to_datetime(["2026-01-0%d" % i for i in (1, 2, 3, 4, 1, 2)]),
        "temp": [10, 20, 30, 40, 100, 200],
    })
    out = add_time_series_features(df, ["temp"], lags=(1,), window=2)
    a = out[out.machine_id == "A"].reset_index(drop=True)
    b = out[out.machine_id == "B"].reset_index(drop=True)
    assert np.isnan(a.loc[0, "temp_lag_1"]) and a.loc[1, "temp_lag_1"] == 10
    assert np.isnan(b.loc[0, "temp_lag_1"])          # no bleed from machine A
    assert a.loc[2, "temp_rolling_mean"] == 25       # mean of 20, 30 (no future value)
    assert a.loc[2, "temp_change"] == 10


def test_time_series_requires_time_column(sample_df):
    with pytest.raises(ValueError):
        add_time_series_features(sample_df, ["torque_nm"])
