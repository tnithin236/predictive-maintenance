"""Reusable feature engineering.

Two groups of features live here:

1. ``add_engineered_features``: row-wise features that ARE supported by the
   AI4I 2020 columns (no timestamps, one reading per machine).
2. ``add_time_series_features``: lag / rolling / rate-of-change features for
   datasets that have a machine id and a timestamp. AI4I 2020 has neither a
   time axis nor repeated machines, so these are NOT used by the trained model;
   the function is provided (and unit-tested) for future time-series data.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

RAW_NUMERIC_FEATURES: list[str] = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
ENGINEERED_FEATURES: list[str] = ["temperature_diff_k", "power_w", "wear_torque_product"]
NUMERIC_FEATURES: list[str] = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES: list[str] = ["machine_type"]
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DOCS: dict[str, str] = {
    "temperature_diff_k": "process_temperature_k - air_temperature_k (heat dissipation proxy).",
    "power_w": "torque_nm * rotational_speed_rpm * 2*pi/60 (mechanical power in watts).",
    "wear_torque_product": "tool_wear_min * torque_nm (overstrain proxy: wear under load).",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the documented engineered columns added.

    Raises:
        ValueError: if a required raw sensor column is missing.
    """
    missing = [c for c in RAW_NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")
    out = df.copy()
    out["temperature_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2.0 * np.pi / 60.0
    out["wear_torque_product"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def add_time_series_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    machine_col: str = "machine_id",
    time_col: str = "timestamp",
    lags: Iterable[int] = (1, 3, 5),
    window: int = 5,
) -> pd.DataFrame:
    """Add per-machine lag, rolling mean/std and rate-of-change features.

    Only past and current values are used (no future information). Rows are
    sorted by machine and time; the sorted frame is returned.

    Raises:
        ValueError: if the machine, time or sensor columns are missing.
    """
    needed = [machine_col, time_col, *sensors]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Time-series features need columns {missing}")
    out = df.sort_values([machine_col, time_col]).copy()
    grouped = out.groupby(machine_col, sort=False)
    for sensor in sensors:
        for lag in lags:
            out[f"{sensor}_lag_{lag}"] = grouped[sensor].shift(lag)
        out[f"{sensor}_change"] = grouped[sensor].diff()
        out[f"{sensor}_rolling_mean"] = grouped[sensor].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[f"{sensor}_rolling_std"] = grouped[sensor].transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )
    return out
