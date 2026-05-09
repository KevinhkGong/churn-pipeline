# src/ingest.py
"""
Ingestion stage: read a raw CSV from disk and surface immediate structural
problems before the rest of the pipeline runs.

The philosophy here is "fail fast and obviously." This stage does NOT do
content validation (null rates, value ranges, distributions) — that's the
validator's job. It only catches issues so fundamental that without
addressing them, the rest of the pipeline can't run at all:

  - file doesn't exist or is unreadable
  - file is empty
  - file has zero rows
  - file is missing required columns

Anything that *can* be coerced or cleaned (bad types, weird values, nulls)
is left for downstream stages so it can be reported on rather than crash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.schema import Schema, coerce_dtypes


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception so the orchestrator can distinguish ingest failures
# from validation failures from training failures.
# ---------------------------------------------------------------------------

class IngestError(Exception):
    """Raised when a file cannot be ingested at all (vs. failing validation)."""
    pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_csv(
    file_path: str | Path,
    schema: Schema,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Read a raw CSV, perform structural checks, and coerce dtypes.

    Returns a dataframe ready for the cleaning stage.
    Raises IngestError if the file cannot be ingested at all.
    """
    file_path = Path(file_path)
    logger.info(f"Ingesting file: {file_path}")

    # ---- Existence + readability ----------------------------------------
    if not file_path.exists():
        raise IngestError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise IngestError(f"Path is not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise IngestError(f"File is empty (0 bytes): {file_path}")

    # ---- Read --------------------------------------------------------------
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise IngestError(f"File has no parseable data: {file_path}")
    except pd.errors.ParserError as e:
        raise IngestError(f"CSV parse error in {file_path}: {e}")
    except Exception as e:
        raise IngestError(f"Unexpected read error for {file_path}: {e}")

    if len(df) == 0:
        raise IngestError(f"File parsed but contains zero rows: {file_path}")

    logger.info(f"Read {len(df)} rows, {len(df.columns)} columns")

    # ---- Required-column presence -----------------------------------------
    # This is structural, not content. Missing required columns means the
    # downstream stages literally can't operate, so we fail here rather
    # than letting the validator catch it (it would, but the error would
    # be less clear).
    missing = set(schema.required_columns) - set(df.columns)
    if missing:
        raise IngestError(
            f"Missing required columns in {file_path.name}: {sorted(missing)}"
        )

    # ---- Unexpected columns: warn, don't fail -----------------------------
    # Extra columns aren't dangerous — they just won't be used. Worth
    # logging because they often indicate an upstream change we should
    # know about.
    extra = set(df.columns) - set(schema.column_names)
    if extra:
        logger.warning(
            f"Unexpected columns in {file_path.name} (will be ignored): "
            f"{sorted(extra)}"
        )

    # ---- Type coercion ----------------------------------------------------
    # Coerce now so that downstream stages (clean, validate) work with
    # properly-typed columns. errors='coerce' inside coerce_dtypes means
    # bad values become NaN/NaT — the validator will then count them as
    # null-rate failures rather than crashing here.
    df = coerce_dtypes(df, schema)

    # Subset to only the columns declared in the schema, in schema order.
    # This protects downstream code from extras and keeps column order
    # predictable for parquet output.
    df = df[schema.column_names]

    logger.info(f"Ingestion complete: {len(df)} rows ready for cleaning")
    return df