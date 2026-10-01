"""Shared fixtures.

``sample_raw_df`` is a SMALL SYNTHETIC TEST FIXTURE shaped like AI4I 2020. It exists
only so unit tests run without the real dataset; it is never used for training the
project model and no reported metric comes from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.train import run_training


def make_sample_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mtype = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    speed = np.clip(rng.normal(1500, 150, n), 1000, None)
    torque = np.clip(rng.normal(40, 10, n), 3, None)
    wear = rng.integers(0, 250, n)
    failure = (((torque * wear) > 8000) | (speed < 1250) | (rng.random(n) < 0.03)).astype(int)
    df = pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "machine_id": [f"{t}{i:05d}" for i, t in enumerate(mtype)],
            "machine_type": mtype,
            "air_temperature_k": air,
            "process_temperature_k": air + 10 + rng.normal(0, 1, n),
            "rotational_speed_rpm": speed,
            "torque_nm": torque,
            "tool_wear_min": wear,
            "machine_failure": failure,
            "twf": 0, "hdf": 0, "pwf": 0, "osf": 0, "rnf": 0,
        }
    )
    return df


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    df = make_sample_df()
    assert df["machine_failure"].sum() >= 30, "fixture needs enough positives"
    return df


@pytest.fixture(scope="session")
def sample_raw_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    inverse = {v: k for k, v in config.COLUMN_RENAME.items()}
    return sample_df.rename(columns=inverse)


@pytest.fixture(scope="session")
def trained_bundle_path(tmp_path_factory, sample_raw_df) -> str:
    """Train a tiny model on the fixture (fast settings, MLflow off)."""
    root = tmp_path_factory.mktemp("model")
    out = run_training(
        raw_df=sample_raw_df, model_dir=root, reports_dir=root / "reports",
        n_iter=2, cv_folds=2, log_to_mlflow=False,
    )
    return out["bundle_path"]
