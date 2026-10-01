import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import MODEL_FEATURES
from src.preprocessing import (
    SchemaError, assert_no_leakage, build_preprocessor, clean_data, get_xy,
    rename_columns, split_data, validate_columns,
)


def test_rename_and_validate(sample_raw_df):
    df = rename_columns(sample_raw_df)
    validate_columns(df)  # should not raise
    assert "machine_id" in df.columns


def test_validate_raises_on_missing_column(sample_df):
    with pytest.raises(SchemaError):
        validate_columns(sample_df.drop(columns=["torque_nm"]))


def test_clean_removes_duplicates_and_invalid(sample_df):
    df = sample_df.head(50).copy()
    dup = df.iloc[[0]].copy()
    dup["udi"] = 9999  # same reading, different row id -> duplicate
    bad = df.iloc[[1]].copy()
    bad["udi"], bad["machine_id"], bad["rotational_speed_rpm"] = 9998, "X1", -5
    cleaned, report = clean_data(pd.concat([df, dup, bad]))
    assert report["dropped_duplicate_rows"] == 1
    assert report["dropped_invalid_values"] == 1
    assert len(cleaned) == 50


def test_missing_sensor_values_are_kept_for_imputation(sample_df):
    df = sample_df.head(20).copy()
    df.loc[3, "torque_nm"] = np.nan
    cleaned, report = clean_data(df)
    assert len(cleaned) == 20 and report["missing_values"]["torque_nm"] == 1


def test_features_exclude_leakage_columns(sample_df):
    X, y = get_xy(sample_df)
    assert list(X.columns) == MODEL_FEATURES
    assert not set(config.LEAKAGE_COLUMNS) & set(X.columns)
    with pytest.raises(ValueError):
        assert_no_leakage(X.assign(twf=0))


def test_split_is_stratified_disjoint_and_reproducible(sample_df):
    train, test = split_data(sample_df)
    assert set(train.index).isdisjoint(test.index)
    assert abs(train["machine_failure"].mean() - test["machine_failure"].mean()) < 0.03
    train2, _ = split_data(sample_df)
    assert train.index.equals(train2.index)


def test_preprocessor_handles_missing_and_unseen_category(sample_df):
    X, _ = get_xy(sample_df)
    pre = build_preprocessor().fit(X)
    probe = X.head(5).copy()
    probe.loc[probe.index[0], "torque_nm"] = np.nan
    probe["machine_type"] = probe["machine_type"].astype(object)
    probe.loc[probe.index[1], "machine_type"] = "UNSEEN"
    out = pre.transform(probe)
    assert not out.isna().any().any()
