# src/clean.py
"""
Cleaning stage: deterministic, conservative transformations applied between
ingestion and validation.

Cleaning principles for this pipeline:
  1. Cleaning is for *noise removal*, not for *fixing bad data*.
     Stripping whitespace and lowercasing categoricals is fine.
     Imputing missing values across the board is NOT — that hides data
     quality problems from the validator.
  2. Every cleaning operation is driven by config, not hardcoded.
  3. Cleaning is idempotent: running it twice produces the same result.
  4. Cleaning runs *before* validation so the validator evaluates data
     the way the model will see it.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.utils.schema import Schema


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    schema: Schema,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Apply all cleaning transformations to a freshly-ingested dataframe.

    Returns a new dataframe; does not mutate the input.
    """
    cleaning_cfg = config.get("cleaning", {})
    df = df.copy()
    initial_rows = len(df)

    # ---- Strip whitespace from string-like columns ------------------------
    if cleaning_cfg.get("string_columns_strip_whitespace", False):
        df = _strip_string_whitespace(df)

    # ---- Lowercase specific categorical columns ---------------------------
    lowercase_cols = cleaning_cfg.get("string_columns_lowercase", [])
    if lowercase_cols:
        df = _lowercase_columns(df, lowercase_cols)

    # ---- Targeted missing-value fills (semantic, not blanket imputation) --
    fill_missing = cleaning_cfg.get("fill_missing", {})
    if fill_missing:
        df = _fill_missing_values(df, fill_missing)

    # ---- Drop duplicates on the primary key -------------------------------
    dedup_col = cleaning_cfg.get("drop_duplicates_on")
    if dedup_col:
        df = _drop_duplicates(df, dedup_col)

    # ---- Final summary ----------------------------------------------------
    rows_removed = initial_rows - len(df)
    if rows_removed:
        logger.info(
            f"Cleaning removed {rows_removed} rows "
            f"({rows_removed / initial_rows:.1%} of input)"
        )
    else:
        logger.info("Cleaning kept all rows")

    return df


# ---------------------------------------------------------------------------
# Individual cleaning operations
# ---------------------------------------------------------------------------

def _strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string-typed columns.

    Operates only on columns whose dtype indicates strings, so we don't
    accidentally call .str on numeric columns. Categoricals are handled
    separately because pandas categories aren't directly stringy.
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
        elif isinstance(df[col].dtype, pd.CategoricalDtype):
            # Strip whitespace from category labels, then rebuild the categorical
            categories = df[col].cat.categories
            stripped = pd.Categorical(
                df[col].astype("string").str.strip(),
                categories=[c.strip() for c in categories]
                          if all(isinstance(c, str) for c in categories)
                          else None,
            )
            df[col] = stripped
    return df


def _lowercase_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Lowercase the values of specified columns.

    Used for categoricals where 'Premium', 'PREMIUM', and 'premium' should
    all collapse to one canonical form. We do this BEFORE the validator
    checks the categorical-allowed list, so 'Premium' from a misbehaving
    upstream system passes the check.
    """
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Cannot lowercase: column '{col}' not in dataframe")
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("string").str.lower().astype("category")
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.lower()
        else:
            logger.warning(
                f"Cannot lowercase: column '{col}' has dtype {df[col].dtype}"
            )
    return df


def _fill_missing_values(
    df: pd.DataFrame,
    fill_map: dict[str, Any],
) -> pd.DataFrame:
    """Fill missing values with explicit, semantically-justified defaults.

    This is intentionally narrow. We only fill columns where missing has a
    clear semantic meaning (e.g. num_support_tickets=0 means "no tickets
    filed", not "we don't know how many"). Blanket imputation would hide
    data quality issues from the validator.
    """
    for col, fill_value in fill_map.items():
        if col not in df.columns:
            logger.warning(f"Cannot fill missing: column '{col}' not in dataframe")
            continue
        n_filled = int(df[col].isna().sum())
        if n_filled > 0:
            df[col] = df[col].fillna(fill_value)
            logger.info(
                f"Filled {n_filled} missing values in '{col}' with {fill_value!r}"
            )
    return df


def _drop_duplicates(df: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """Drop rows with duplicate values in the key column.

    Keeps the FIRST occurrence. In a real pipeline you'd want to think
    about which to keep (most recent? highest activity?) but for daily
    snapshots, first-wins is fine and predictable.
    """
    if key_column not in df.columns:
        logger.warning(f"Cannot drop duplicates: key column '{key_column}' missing")
        return df

    n_before = len(df)
    df = df.drop_duplicates(subset=[key_column], keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            f"Dropped {n_dropped} duplicate rows on '{key_column}' "
            f"({n_dropped / n_before:.1%} of input)"
        )
    return df