# src/store.py
"""
Storage stage: persist a validated dataframe to the processed data
directory and move the source raw file out of the inbox.

Storage runs only after validation has passed. By the time we get here,
the dataframe is trusted: schema is right, nulls are within thresholds,
distributions are sane. Storage's job is just to write it durably and
manage the lifecycle of the raw file.

Conventions:
  - Processed files are parquet, named by run_date for easy lookup.
  - Successfully-stored raw files move from data/raw/ to data/archive/.
  - Failed-validation raw files move to data/quarantine/ (handled by
    the orchestrator, not here, since this module only runs on success).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def store_processed(
    df: pd.DataFrame,
    run_date: str,
    config: dict[str, Any],
) -> Path:
    """Write the validated dataframe to processed/ as parquet.

    Returns the path to the written file. Overwrites if a file for the
    same run_date already exists (re-runs of the same day produce one
    canonical processed file).
    """
    storage_cfg = config["storage"]
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    filename = storage_cfg["filename_template"].format(run_date=run_date)
    out_path = processed_dir / filename

    df.to_parquet(
        out_path,
        compression=storage_cfg.get("compression", "snappy"),
        index=False,
    )
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def archive_raw(raw_path: str | Path, config: dict[str, Any]) -> Path:
    """Move a successfully-processed raw file to the archive directory.

    Keeping raw files lets you re-run the pipeline against historical
    inputs (e.g., after fixing a validation rule) without losing data.
    """
    raw_path = Path(raw_path)
    archive_dir = Path(config["paths"]["archive_dir"])
    archive_dir.mkdir(parents=True, exist_ok=True)

    dest = archive_dir / raw_path.name
    shutil.move(str(raw_path), str(dest))
    logger.info(f"Archived {raw_path.name} -> {dest}")
    return dest


def quarantine_raw(raw_path: str | Path, config: dict[str, Any]) -> Path:
    """Move a failed raw file to the quarantine directory.

    Called by the orchestrator when validation fails. Keeps the bad file
    around for human inspection — deleting it would lose the evidence.
    """
    raw_path = Path(raw_path)
    quarantine_dir = Path(config["paths"]["quarantine_dir"])
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    dest = quarantine_dir / raw_path.name
    shutil.move(str(raw_path), str(dest))
    logger.info(f"Quarantined {raw_path.name} -> {dest}")
    return dest