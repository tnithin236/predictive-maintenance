"""Inference: load the saved bundle and produce predictions (no training here)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config
from src.anomaly_detection import score_anomalies
from src.feature_engineering import RAW_NUMERIC_FEATURES, add_engineered_features

logger = logging.getLogger(__name__)
REQUIRED_INPUTS = [*RAW_NUMERIC_FEATURES, "machine_type"]
BUNDLE_KEYS = {"model", "anomaly_model", "feature_config", "model_version", "trained_at"}


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained model bundle exists yet."""


def load_bundle(path: Path | str | None = None) -> dict[str, Any]:
    """Load the serialized training bundle."""
    path = Path(path or config.MODEL_PATH)
    if not path.exists():
        raise ModelNotFoundError(f"No trained model at {path}. Run `python -m src.train` first.")
    bundle = joblib.load(path)
    missing = BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def classify_risk(
    probability: float,
    medium: float | None = None,
    high: float | None = None,
) -> str:
    """Map a failure probability to LOW / MEDIUM / HIGH using configurable thresholds."""
    medium = config.RISK_MEDIUM_THRESHOLD if medium is None else medium
    high = config.RISK_HIGH_THRESHOLD if high is None else high
    if not 0.0 <= medium <= high <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= medium <= high <= 1")
    if probability >= high:
        return "HIGH"
    if probability >= medium:
        return "MEDIUM"
    return "LOW"


class Predictor:
    """Stateless prediction service around a loaded bundle."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.model = bundle["model"]
        self.anomaly_model = bundle["anomaly_model"]
        self.model_features: list[str] = bundle["feature_config"]["model_features"]
        self._explainer = None

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> "Predictor":
        return cls(load_bundle(path))

    @property
    def explainer(self):
        if self._explainer is None:
            from src.explainability import ShapExplainer

            self._explainer = ShapExplainer(self.model)
        return self._explainer

    def predict(
        self, records: list[dict[str, Any]] | pd.DataFrame, explain: bool = True, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Predict failure probability, risk, anomaly status and top factors."""
        frame = pd.DataFrame(records).copy() if not isinstance(records, pd.DataFrame) else records.copy()
        missing = [c for c in REQUIRED_INPUTS if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing input fields: {missing}")
        if "machine_id" not in frame.columns:
            frame["machine_id"] = None
        features = add_engineered_features(frame)[self.model_features]

        proba = self.model.predict_proba(features)[:, 1]
        anomalies = score_anomalies(self.anomaly_model, features, frame["machine_id"])
        explanations = self.explainer.explain_rows(features, top_k) if explain else [
            {"top_factors": [], "factor_details": []} for _ in range(len(features))
        ]

        results = []
        for i, p in enumerate(proba):
            results.append(
                {
                    "machine_id": None if pd.isna(frame["machine_id"].iloc[i]) else str(frame["machine_id"].iloc[i]),
                    "prediction": int(p >= config.DECISION_THRESHOLD),
                    "failure_probability": round(float(p), 6),
                    "risk_level": classify_risk(float(p)),
                    "anomaly_score": float(anomalies["anomaly_score"].iloc[i]),
                    "anomaly_status": str(anomalies["anomaly_status"].iloc[i]),
                    "top_factors": explanations[i]["top_factors"],
                    "factor_details": explanations[i]["factor_details"],
                    "model_version": self.bundle["model_version"],
                }
            )
        return results

    def model_info(self) -> dict[str, Any]:
        """Metadata for /model/info (all values come from the trained bundle)."""
        b = self.bundle
        return {
            "model_version": b["model_version"],
            "trained_at": b["trained_at"],
            "model_type": type(self.model.named_steps["clf"]).__name__,
            "features": self.model_features,
            "best_params": b.get("best_params"),
            "test_metrics": b.get("test_metrics"),
            "mlflow_run_id": b.get("mlflow_run_id"),
            "thresholds": {
                "decision": config.DECISION_THRESHOLD,
                "risk_medium": config.RISK_MEDIUM_THRESHOLD,
                "risk_high": config.RISK_HIGH_THRESHOLD,
            },
            "library_versions": b.get("library_versions"),
        }
