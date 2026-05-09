# src/validate.py
"""
Validation stage: run all configured data quality checks on a cleaned
dataframe and produce a ValidationReport.

The validator is structured as a series of small check functions, each
of which appends one or more CheckResults to the report. The main
validate() function calls them in order.

Design principles:
  - Validation NEVER mutates the dataframe.
  - Validation NEVER raises on bad data — every problem becomes a
    CheckResult with appropriate severity. Only programmer errors raise.
  - Severity is determined by config (block_training_on list) so the
    business decides what's blocking and what's just a warning.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from src.utils.schema import Schema
from src.validation_report import CheckResult, Severity, ValidationReport


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity resolver
# ---------------------------------------------------------------------------

def _severity_for(category: str, config: dict[str, Any]) -> Severity:
    """Look up whether a given check category is blocking (ERROR) or not.

    Categories listed in validation.block_training_on are ERROR; everything
    else is WARNING. This way, what's blocking is fully owned by config.
    """
    blocking = set(config["validation"].get("block_training_on", []))
    return Severity.ERROR if category in blocking else Severity.WARNING


# ---------------------------------------------------------------------------
# Layer 1: Structural checks
# ---------------------------------------------------------------------------

def _check_row_count(
    df: pd.DataFrame, config: dict[str, Any], report: ValidationReport,
) -> None:
    rc = config["validation"]["row_count"]
    n = len(df)
    in_bounds = rc["min"] <= n <= rc["max"]
    report.add(CheckResult(
        name="row_count",
        category="row_count_out_of_bounds",
        passed=in_bounds,
        severity=Severity.INFO if in_bounds
                 else _severity_for("row_count_out_of_bounds", config),
        message=(f"Row count {n} is within [{rc['min']}, {rc['max']}]"
                 if in_bounds else
                 f"Row count {n} is outside [{rc['min']}, {rc['max']}]"),
        details={"row_count": n, "min": rc["min"], "max": rc["max"]},
    ))


def _check_primary_key_unique(
    df: pd.DataFrame, schema: Schema, config: dict[str, Any],
    report: ValidationReport,
) -> None:
    pk = schema.primary_key
    if pk not in df.columns:
        # Should have been caught at ingest, but defensive
        report.add(CheckResult(
            name=f"primary_key_present_{pk}",
            category="schema_error",
            passed=False,
            severity=_severity_for("schema_error", config),
            message=f"Primary key column '{pk}' is missing",
        ))
        return

    n_dupes = int(df[pk].duplicated().sum())
    passed = n_dupes == 0
    report.add(CheckResult(
        name=f"primary_key_unique_{pk}",
        category="primary_key_violation",
        passed=passed,
        severity=Severity.INFO if passed
                 else _severity_for("primary_key_violation", config),
        message=(f"All {len(df)} values of '{pk}' are unique" if passed
                 else f"Found {n_dupes} duplicate values in '{pk}'"),
        details={"primary_key": pk, "n_duplicates": n_dupes},
    ))


# ---------------------------------------------------------------------------
# Layer 2: Content checks
# ---------------------------------------------------------------------------

def _check_null_rates(
    df: pd.DataFrame, schema: Schema, config: dict[str, Any],
    report: ValidationReport,
) -> None:
    thresholds = config["validation"]["null_rate_thresholds"]
    default = thresholds.get("default", 0.20)

    null_summary: dict[str, float] = {}

    for col_name in schema.column_names:
        if col_name not in df.columns:
            continue
        threshold = thresholds.get(col_name, default)
        null_rate = float(df[col_name].isna().mean())
        null_summary[col_name] = round(null_rate, 4)

        passed = null_rate <= threshold
        severity = (Severity.INFO if passed
                    else _severity_for("null_rate_exceeded", config))
        report.add(CheckResult(
            name=f"null_rate_{col_name}",
            category="null_rate_exceeded",
            passed=passed,
            severity=severity,
            message=(f"Null rate for '{col_name}' is {null_rate:.1%} "
                     f"(threshold {threshold:.1%})"),
            details={
                "column": col_name,
                "null_rate": round(null_rate, 4),
                "threshold": threshold,
            },
        ))

    report.summary["null_rates"] = null_summary


def _check_value_ranges(
    df: pd.DataFrame, config: dict[str, Any], report: ValidationReport,
) -> None:
    """Check that numeric columns stay within their allowed ranges.

    A row is "out of range" if ANY of its checked columns is outside bounds.
    We fail if the fraction of out-of-range rows exceeds the configured
    threshold — single bad rows are noise; systematic out-of-range is a bug.
    """
    ranges = config["validation"]["value_ranges"]
    pct_threshold = config["validation"]["out_of_range_row_pct"]

    if not ranges:
        return

    # Build a row-level mask: True if any checked column is out of range
    out_of_range_mask = pd.Series(False, index=df.index)
    per_column_counts: dict[str, int] = {}

    for col_name, (lo, hi) in ranges.items():
        if col_name not in df.columns:
            continue
        col = df[col_name]
        # Treat nulls as "not out of range" — null rate is a separate check
        col_oor = col.notna() & ((col < lo) | (col > hi))
        per_column_counts[col_name] = int(col_oor.sum())
        out_of_range_mask = out_of_range_mask | col_oor

    n_oor_rows = int(out_of_range_mask.sum())
    pct_oor = n_oor_rows / len(df) if len(df) else 0.0
    passed = pct_oor <= pct_threshold

    report.add(CheckResult(
        name="value_ranges",
        category="out_of_range_exceeded",
        passed=passed,
        severity=Severity.INFO if passed
                 else _severity_for("out_of_range_exceeded", config),
        message=(f"{n_oor_rows} rows ({pct_oor:.2%}) have out-of-range values "
                 f"(threshold {pct_threshold:.2%})"),
        details={
            "n_rows_out_of_range": n_oor_rows,
            "pct_rows_out_of_range": round(pct_oor, 4),
            "threshold": pct_threshold,
            "per_column_counts": per_column_counts,
        },
    ))


def _check_categorical_values(
    df: pd.DataFrame, config: dict[str, Any], report: ValidationReport,
) -> None:
    allowed_map = config["validation"].get("categorical_allowed", {})
    strict = config["validation"].get("categorical_strict", True)

    for col_name, allowed in allowed_map.items():
        if col_name not in df.columns:
            continue
        # Drop nulls before computing observed values — null rate is its
        # own check and shouldn't double-count here.
        observed = set(df[col_name].dropna().astype(str).unique())
        unseen = observed - set(allowed)

        passed = len(unseen) == 0
        category = "unknown_category_strict" if strict else "unknown_category_lenient"
        severity = (Severity.INFO if passed
                    else _severity_for(category, config))
        report.add(CheckResult(
            name=f"categorical_values_{col_name}",
            category=category,
            passed=passed,
            severity=severity,
            message=(f"All values of '{col_name}' are in the allowed set" if passed
                     else f"Unexpected values in '{col_name}': {sorted(unseen)}"),
            details={
                "column": col_name,
                "allowed": list(allowed),
                "unseen": sorted(unseen),
            },
        ))


# ---------------------------------------------------------------------------
# Layer 3: Distribution checks
# ---------------------------------------------------------------------------

def _check_churn_rate(
    df: pd.DataFrame, schema: Schema, config: dict[str, Any],
    report: ValidationReport,
) -> None:
    """Sanity-check the class balance.

    A churn rate way outside the expected range is almost always a bug
    upstream, not a real spike in customer departures.
    """
    target = schema.target_column
    if target not in df.columns:
        return

    rng = config["validation"].get("distribution", {}).get("churn_rate_range")
    if not rng:
        return

    churn_rate = float(df[target].mean())
    lo, hi = rng
    passed = lo <= churn_rate <= hi

    report.add(CheckResult(
        name="churn_rate",
        category="churn_rate_out_of_range",
        passed=passed,
        severity=Severity.INFO if passed
                 else _severity_for("churn_rate_out_of_range", config),
        message=(f"Churn rate is {churn_rate:.2%} "
                 f"(expected range {lo:.0%}-{hi:.0%})"),
        details={"churn_rate": round(churn_rate, 4), "range": [lo, hi]},
    ))

    report.summary["churn_rate"] = round(churn_rate, 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate(
    df: pd.DataFrame,
    schema: Schema,
    config: dict[str, Any],
    file_path: str = "",
    run_date: str | None = None,
) -> ValidationReport:
    """Run all validation checks and return a ValidationReport.

    The orchestrator decides what to do with the report (write it to disk,
    block training, etc.) — this function only produces it.
    """
    if run_date is None:
        run_date = date.today().isoformat()

    report = ValidationReport.new(
        file_path=file_path, run_date=run_date, row_count=len(df),
    )

    # Layer 1: structure
    _check_row_count(df, config, report)
    _check_primary_key_unique(df, schema, config, report)

    # Layer 2: content
    _check_null_rates(df, schema, config, report)
    _check_value_ranges(df, config, report)
    _check_categorical_values(df, config, report)

    # Layer 3: distribution
    _check_churn_rate(df, schema, config, report)

    # Final logging
    if report.passed:
        logger.info(
            f"Validation PASSED: {len(report.checks)} checks, "
            f"{len(report.warnings)} warnings"
        )
    else:
        logger.warning(
            f"Validation FAILED: {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings"
        )
        for err in report.errors:
            logger.warning(f"  ERROR  [{err.category}] {err.message}")
        for warn in report.warnings:
            logger.info(f"  WARN   [{warn.category}] {warn.message}")

    return report