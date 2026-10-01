"""Unsupervised anomaly detection with Isolation Forest.

IMPORTANT: anomaly score != failure probability. The score says how unusual a
reading is relative to training data; it does not say a failure will occur.
``anomaly_score`` = -score_samples (higher = more anomalous, roughly 0.5 is the
neutral point). Status comes from the model's own contamination-based cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src import config
from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.preprocessing import build_preprocessor


def build_isolation_forest(
    contamination: float | None = None, seed: int | None = None
) -> Pipeline:
    """Preprocessor + IsolationForest (fitted without labels)."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination=config.ANOMALY_CONTAMINATION if contamination is None else contamination,
                    random_state=config.RANDOM_SEED if seed is None else seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def score_anomalies(
    pipeline: Pipeline, features: pd.DataFrame, machine_ids: pd.Series | list | None = None
) -> pd.DataFrame:
    """Return machine_id, anomaly_score and anomaly_status for each row."""
    scores = -pipeline.score_samples(features)
    flags = pipeline.predict(features)  # -1 anomaly, 1 normal
    return pd.DataFrame(
        {
            "machine_id": list(machine_ids) if machine_ids is not None else [None] * len(features),
            "anomaly_score": np.round(scores.astype(float), 6),
            "anomaly_status": np.where(flags == -1, "ANOMALY", "NORMAL"),
        }
    )


def summarize_against_labels(scored: pd.DataFrame, y_true: pd.Series) -> dict[str, float | None]:
    """Compare anomaly flags with real failure labels (evaluation only)."""
    flagged = (scored["anomaly_status"] == "ANOMALY").to_numpy()
    y = np.asarray(y_true)

    def rate(mask: np.ndarray) -> float | None:
        return float(y[mask].mean()) if mask.any() else None

    return {
        "flagged_rate": float(flagged.mean()),
        "failure_rate_among_flagged": rate(flagged),
        "failure_rate_among_unflagged": rate(~flagged),
        "mean_score_failures": float(scored.loc[y == 1, "anomaly_score"].mean()) if (y == 1).any() else None,
        "mean_score_non_failures": float(scored.loc[y == 0, "anomaly_score"].mean()) if (y == 0).any() else None,
    }
