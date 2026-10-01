"""Central configuration.

Every tunable value can be overridden through an environment variable so the
same code runs locally, in tests and in Docker without edits. No secrets are
stored here; ``DATABASE_URL`` is read from the environment only.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# --- Paths -------------------------------------------------------------
RAW_DATA_DIR = _path("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw")
PROCESSED_DATA_DIR = _path("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
MODEL_DIR = _path("MODEL_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = _path("REPORTS_DIR", MODEL_DIR / "reports")
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "ai4i2020_clean.csv"
MODEL_PATH = _path("MODEL_PATH", MODEL_DIR / "model_bundle.joblib")

# UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset.
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)

# --- Reproducibility / training ----------------------------------------
RANDOM_SEED = _int("RANDOM_SEED", 42)
TEST_SIZE = _float("TEST_SIZE", 0.2)
CV_FOLDS = _int("CV_FOLDS", 3)
TUNING_ITERATIONS = _int("TUNING_ITERATIONS", 12)
ANOMALY_CONTAMINATION = _float("ANOMALY_CONTAMINATION", 0.03)

# --- Decision and risk thresholds (configurable business rules) --------
DECISION_THRESHOLD = _float("DECISION_THRESHOLD", 0.50)
RISK_MEDIUM_THRESHOLD = _float("RISK_MEDIUM_THRESHOLD", 0.40)
RISK_HIGH_THRESHOLD = _float("RISK_HIGH_THRESHOLD", 0.70)

# --- MLflow ------------------------------------------------------------
# MLflow >= 3 no longer supports the plain file store by default, so use SQLite.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive-maintenance")

# --- Serving -----------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_BATCH_SIZE = _int("MAX_BATCH_SIZE", 5000)
DATABASE_URL = os.getenv("DATABASE_URL")  # optional; never hardcode credentials

# --- Dataset schema (AI4I 2020) ----------------------------------------
COLUMN_RENAME = {
    "UDI": "udi",
    "Product ID": "machine_id",
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
TARGET = "machine_failure"
# Failure-mode flags are components of the target -> using them as features
# would be target leakage, so they are excluded from every model input.
FAILURE_MODE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
LEAKAGE_COLUMNS = [TARGET, *FAILURE_MODE_COLUMNS]
VALID_MACHINE_TYPES = ["L", "M", "H"]
REQUIRED_COLUMNS = list(COLUMN_RENAME.values())
