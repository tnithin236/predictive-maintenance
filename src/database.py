"""Modular database layer (SQLAlchemy).

Written against PostgreSQL. Only ``get_engine`` knows the backend; swapping to
Oracle later means installing the ``oracledb`` driver, changing ``DATABASE_URL``
and porting the DDL in database/schema.sql. That port has NOT been done or tested.

CLI:
    python -m src.database seed     # load the cleaned dataset into an EMPTY database
    python -m src.database export   # write CSVs for the psql \\copy path (database/seed.sql)
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine

from src import config

logger = logging.getLogger(__name__)


def get_engine(url: str | None = None) -> Engine:
    """Create an engine from ``DATABASE_URL`` (never hardcoded)."""
    url = url or config.DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return create_engine(url, pool_pre_ping=True)


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned dataset into ``machines`` and ``sensor_readings`` frames."""
    machines = (
        df[["machine_id", "machine_type"]].drop_duplicates("machine_id").reset_index(drop=True)
    )
    readings = df[
        ["machine_id", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
         "torque_nm", "tool_wear_min", config.TARGET]
    ].copy()
    return machines, readings


def seed_database(df: pd.DataFrame, engine: Engine | None = None) -> dict[str, int]:
    """Insert machines and readings. Real dataset rows only; no synthetic data."""
    engine = engine or get_engine()
    machines, readings = build_tables(df)
    with engine.begin() as conn:
        machines.to_sql("machines", conn, if_exists="append", index=False, chunksize=2000, method="multi")
        readings.to_sql("sensor_readings", conn, if_exists="append", index=False, chunksize=2000, method="multi")
    return {"machines": len(machines), "sensor_readings": len(readings)}


def save_predictions(results: list[dict[str, Any]], engine: Engine | None = None) -> int:
    """Persist prediction results. Rows with unknown/absent machine_id are skipped
    (the predictions table has a foreign key to machines)."""
    engine = engine or get_engine()
    rows = [
        {
            "machine_id": r["machine_id"], "failure_probability": r["failure_probability"],
            "risk_level": r["risk_level"], "prediction": r["prediction"],
            "anomaly_score": r["anomaly_score"], "anomaly_status": r["anomaly_status"],
            "model_version": r["model_version"],
        }
        for r in results if r.get("machine_id")
    ]
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    with engine.begin() as conn:
        known = set(pd.read_sql("SELECT machine_id FROM machines", conn)["machine_id"])
        frame = frame[frame["machine_id"].isin(known)]
        if not frame.empty:
            frame.to_sql("predictions", conn, if_exists="append", index=False)
    return len(frame)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seed", "export"])
    args = parser.parse_args()
    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    if args.command == "seed":
        print(seed_database(df))
    else:
        machines, readings = build_tables(df)
        machines.to_csv(config.PROCESSED_DATA_DIR / "db_machines.csv", index=False)
        readings.to_csv(config.PROCESSED_DATA_DIR / "db_sensor_readings.csv", index=False)
        print("Exported db_machines.csv and db_sensor_readings.csv")


if __name__ == "__main__":
    main()
