"""Dataset download and loading.

Usage:
    python -m src.data_loader download   # fetch AI4I 2020 from the UCI repository
    python -m src.data_loader summary     # write dataset statistics computed from the data
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded."""


def download_dataset(dest_dir: Path | None = None, url: str = config.DATASET_URL) -> Path:
    """Download and extract the AI4I 2020 CSV. Returns the CSV path."""
    dest_dir = Path(dest_dir or config.RAW_DATA_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetDownloadError(
            f"Could not download {url}: {exc}. Download 'ai4i2020.csv' manually from "
            "https://archive.ics.uci.edu/dataset/601 and place it in data/raw/."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise DatasetDownloadError("Downloaded archive contains no CSV file.")
        target = dest_dir / "ai4i2020.csv"
        target.write_bytes(archive.read(csv_names[0]))
    logger.info("Dataset saved to %s", target)
    return target


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV. Raises FileNotFoundError with instructions if absent."""
    path = Path(path or config.RAW_DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_loader download` or place "
            "ai4i2020.csv in data/raw/."
        )
    return pd.read_csv(path)


def _write_summary() -> Path:
    from src.preprocessing import clean_data, rename_columns, summarize_dataset, validate_columns

    df = rename_columns(load_raw_data())
    validate_columns(df)
    clean, report = clean_data(df)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "dataset_summary.json"
    out.write_text(json.dumps(summarize_dataset(clean, report), indent=2))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "summary"])
    args = parser.parse_args()
    if args.command == "download":
        print(download_dataset())
    else:
        print(_write_summary())


if __name__ == "__main__":
    main()
