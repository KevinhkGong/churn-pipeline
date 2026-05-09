# src/utils/schema.py
"""
Schema utilities: load pipeline config, parse the schema definition, and
provide type-coercion helpers used by both the data generator and the
validator. This module is the single source of truth for "what does a
valid row look like."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Kept deliberately thin — no schema-validation-of-the-config-itself.
    For a take-home, that's overkill. In production you'd validate the
    config against a Pydantic model on load.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Schema representation
# ---------------------------------------------------------------------------

# Map our human-friendly dtype names (used in YAML) to pandas/numpy dtypes.
# Keeping this mapping here means the YAML stays readable and the rest of
# the codebase has one place to look for "what does 'date' actually mean".
DTYPE_MAP: dict[str, str] = {
    "string":   "string",
    "int":      "Int64",      # nullable integer — important for validation
    "float":    "float64",
    "bool":     "boolean",    # nullable boolean
    "category": "category",
    "date":     "datetime64[ns]",
}


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single column from the schema."""
    name: str
    dtype: str           # human-readable dtype name from YAML
    required: bool

    @property
    def pandas_dtype(self) -> str:
        return DTYPE_MAP[self.dtype]


@dataclass(frozen=True)
class Schema:
    """Parsed schema for the pipeline's expected dataset."""
    primary_key: str
    target_column: str
    columns: list[ColumnSpec]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def required_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.required]

    def get_column(self, name: str) -> ColumnSpec | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


def parse_schema(config: dict[str, Any]) -> Schema:
    """Build a Schema object from the loaded YAML config."""
    schema_cfg = config["schema"]
    columns = [
        ColumnSpec(name=col_name, dtype=spec["dtype"], required=spec["required"])
        for col_name, spec in schema_cfg["columns"].items()
    ]
    return Schema(
        primary_key=schema_cfg["primary_key"],
        target_column=schema_cfg["target_column"],
        columns=columns,
    )


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def coerce_dtypes(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Coerce a dataframe's columns to the dtypes declared in the schema.

    - Only operates on columns that exist in the dataframe (missing columns
      are caught upstream by the structural validation step).
    - Uses errors='coerce' on numeric/date columns so bad values become NaT/NaN
      rather than raising — the validator will catch them as null-rate failures.
    - Returns a new dataframe; does not mutate the input.
    """
    df = df.copy()

    for col_spec in schema.columns:
        if col_spec.name not in df.columns:
            continue

        dtype = col_spec.dtype

        if dtype == "date":
            df[col_spec.name] = pd.to_datetime(
                df[col_spec.name], errors="coerce", format="mixed"
            )
        elif dtype in ("int", "float"):
            df[col_spec.name] = pd.to_numeric(df[col_spec.name], errors="coerce")
            df[col_spec.name] = df[col_spec.name].astype(col_spec.pandas_dtype)
        elif dtype == "bool":
            # Bools commonly arrive as 'True'/'False' strings or 0/1
            df[col_spec.name] = (
                df[col_spec.name]
                .map({"True": True, "False": False, "true": True, "false": False,
                      True: True, False: False, 1: True, 0: False, "1": True, "0": False})
                .astype("boolean")
            )
        else:
            df[col_spec.name] = df[col_spec.name].astype(col_spec.pandas_dtype)

    return df


# ---------------------------------------------------------------------------
# Convenience: load config + schema in one call
# ---------------------------------------------------------------------------

def load_pipeline_config(
    config_path: str | Path = "config/pipeline_config.yaml",
) -> tuple[dict[str, Any], Schema]:
    """Load the pipeline config and parse the schema in one call."""
    config = load_config(config_path)
    schema = parse_schema(config)
    return config, schema