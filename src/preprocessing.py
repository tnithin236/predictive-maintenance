"""Data validation, cleaning, splitting and the sklearn preprocessor.

The scaler/imputer/encoder live INSIDE sklearn Pipelines so they are fitted on
training data only (also inside each CV fold), which prevents data leakage.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RAW_NUMERIC_FEATURES,
    add_engineered_features,
)

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when the input data does not match the expected schema."""


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw AI4I column names to snake_case names."""
    return df.rename(columns=config.COLUMN_RENAME)


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ``SchemaError`` if required columns are missing."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Coerce types, drop duplicates and invalid rows, and report what changed.

    Missing sensor values are NOT dropped here; they are imputed inside the
    modelling pipeline (median, fitted on training data only). Rows with a
    missing target / id / machine type cannot be used and are dropped.
    """
    report: dict[str, Any] = {"rows_in": int(len(df))}
    out = df.copy()

    for col in RAW_NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[config.TARGET] = pd.to_numeric(out[config.TARGET], errors="coerce")
    report["missing_values"] = {
        c: int(v) for c, v in out[config.REQUIRED_COLUMNS].isna().sum().items()
    }

    before = len(out)
    out = out.dropna(subset=[config.TARGET, "machine_id", "machine_type"])
    report["dropped_missing_target_or_id"] = int(before - len(out))

    dedupe_cols = [c for c in out.columns if c != "udi"]
    before = len(out)
    out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    report["dropped_duplicate_rows"] = int(before - len(out))
    report["duplicate_machine_ids"] = int(out["machine_id"].duplicated().sum())

    invalid = (
        (out["air_temperature_k"] <= 0)
        | (out["process_temperature_k"] <= 0)
        | (out["rotational_speed_rpm"] <= 0)
        | (out["torque_nm"] < 0)
        | (out["tool_wear_min"] < 0)
        | (~out["machine_type"].isin(config.VALID_MACHINE_TYPES))
        | (~out[config.TARGET].isin([0, 1]))
    )
    report["dropped_invalid_values"] = int(invalid.sum())
    out = out.loc[~invalid].copy()
    out[config.TARGET] = out[config.TARGET].astype(int)
    out = out.reset_index(drop=True)
    report["rows_out"] = int(len(out))
    logger.info("Cleaning report: %s", report)
    return out, report


def summarize_dataset(df: pd.DataFrame, cleaning_report: dict[str, Any]) -> dict[str, Any]:
    """Compute documentation statistics from the actual data (nothing hardcoded)."""
    counts = df[config.TARGET].value_counts().to_dict()
    n = int(len(df))
    return {
        "n_records": n,
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target": config.TARGET,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "failure_rate": float(counts.get(1, 0) / n) if n else None,
        "cleaning_report": cleaning_report,
        "describe": df[RAW_NUMERIC_FEATURES].describe().round(4).to_dict(),
    }


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if target / failure-mode columns reached the feature matrix."""
    leaked = [c for c in config.LEAKAGE_COLUMNS if c in features.columns]
    if leaked:
        raise ValueError(f"Target leakage: {leaked} present in features")


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model feature matrix (with engineered features) and target."""
    engineered = add_engineered_features(df)
    features = engineered[MODEL_FEATURES]
    assert_no_leakage(features)
    return features, df[config.TARGET].astype(int)


def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (AI4I has no time axis, so random-stratified)."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config.TEST_SIZE if test_size is None else test_size,
        stratify=df[config.TARGET],
        random_state=config.RANDOM_SEED if seed is None else seed,
    )
    return df.loc[train_idx], df.loc[test_idx]


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Median-impute + standard-scale numerics; impute + one-hot categoricals."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    ).set_output(transform="pandas")
