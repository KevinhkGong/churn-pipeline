# tests/test_validate.py
"""
Tests for the validation layer.

Strategy: build small synthetic dataframes that exercise specific failure
modes, and assert the validator catches them with the right severity.

We don't test the full pipeline here — that's an integration test. These
are unit tests on the validator's logic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.schema import load_pipeline_config
from src.validate import validate
from src.validation_report import Severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config_and_schema():
    """Load the real pipeline config so tests reflect actual thresholds."""
    return load_pipeline_config("config/pipeline_config.yaml")


@pytest.fixture
def good_df():
    """Build a small but valid dataframe matching the schema."""
    n = 500
    return pd.DataFrame({
        "user_id": [f"u_{i:06d}" for i in range(n)],
        "signup_date": pd.to_datetime(["2024-01-01"] * n),
        "tenure_days": [365] * n,
        "plan_type": pd.Categorical(["free"] * n,
                                    categories=["free", "basic", "premium"]),
        "monthly_charges": [10.0] * n,
        "total_charges": [120.0] * n,
        "num_logins_last_30d": [15] * n,
        "avg_session_minutes": [12.0] * n,
        "num_support_tickets": [1] * n,
        "days_since_last_login": [3] * n,
        "payment_method": pd.Categorical(["card"] * n,
                                         categories=["card", "bank", "paypal"]),
        "is_active": [True] * n,
        "churned": [0] * (n - 30) + [1] * 30,         # 6% churn
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_clean_data_passes_validation(config_and_schema, good_df):
    config, schema = config_and_schema
    report = validate(good_df, schema, config, run_date="2026-01-01")

    assert report.passed, (
        f"Expected clean data to pass validation. "
        f"Errors: {[e.message for e in report.errors]}"
    )
    assert len(report.errors) == 0


# ---------------------------------------------------------------------------
# Failure cases — one per validation category
# ---------------------------------------------------------------------------

def test_high_null_rate_blocks_training(config_and_schema, good_df):
    """40% nulls on monthly_charges (threshold 5%) should produce an error."""
    config, schema = config_and_schema
    df = good_df.copy()
    df.loc[df.index[:200], "monthly_charges"] = None      # 40% null

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    null_errors = [e for e in report.errors
                   if e.category == "null_rate_exceeded"
                   and "monthly_charges" in e.message]
    assert len(null_errors) == 1
    assert null_errors[0].severity == Severity.ERROR


def test_duplicate_primary_keys_block_training(config_and_schema, good_df):
    config, schema = config_and_schema
    df = good_df.copy()
    df.loc[df.index[-10:], "user_id"] = df.iloc[:10]["user_id"].values

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    pk_errors = [e for e in report.errors
                 if e.category == "primary_key_violation"]
    assert len(pk_errors) == 1


def test_unknown_category_blocks_training(config_and_schema, good_df):
    config, schema = config_and_schema
    df = good_df.copy()
    # Add a brand-new category to plan_type
    df["plan_type"] = df["plan_type"].cat.add_categories(["enterprise"])
    df.loc[df.index[:50], "plan_type"] = "enterprise"

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    cat_errors = [e for e in report.errors
                  if "plan_type" in e.message
                  and "enterprise" in str(e.details.get("unseen", []))]
    assert len(cat_errors) >= 1


def test_churn_rate_spike_blocks_training(config_and_schema, good_df):
    """A churn rate of 70% is way outside the [0.01, 0.30] expected range."""
    config, schema = config_and_schema
    df = good_df.copy()
    df["churned"] = [1] * 350 + [0] * 150              # 70% churn

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    churn_errors = [e for e in report.errors
                    if e.category == "churn_rate_out_of_range"]
    assert len(churn_errors) == 1


def test_out_of_range_values_block_training(config_and_schema, good_df):
    """Set 10% of rows to nonsense; threshold is 2%."""
    config, schema = config_and_schema
    df = good_df.copy()
    df.loc[df.index[:50], "tenure_days"] = -100        # negative, out of range
    df.loc[df.index[:50], "monthly_charges"] = 9999.0  # over the ceiling

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    range_errors = [e for e in report.errors
                    if e.category == "out_of_range_exceeded"]
    assert len(range_errors) == 1


def test_tiny_batch_blocks_training(config_and_schema, good_df):
    config, schema = config_and_schema
    df = good_df.iloc[:50]                              # below min of 100

    report = validate(df, schema, config, run_date="2026-01-01")

    assert not report.passed
    rc_errors = [e for e in report.errors
                 if e.category == "row_count_out_of_bounds"]
    assert len(rc_errors) == 1


# ---------------------------------------------------------------------------
# Report-level invariants
# ---------------------------------------------------------------------------

def test_report_serializes_to_json(config_and_schema, good_df):
    """The report must be JSON-serializable for the audit trail."""
    config, schema = config_and_schema
    report = validate(good_df, schema, config, run_date="2026-01-01")
    json_str = report.to_json()
    assert isinstance(json_str, str)
    assert "checks" in json_str
    assert "passed" in json_str


def test_report_passed_property_only_considers_errors(config_and_schema, good_df):
    """Warnings should not flip passed to False."""
    config, schema = config_and_schema
    report = validate(good_df, schema, config, run_date="2026-01-01")
    # The good fixture should produce zero errors. If there are warnings
    # (e.g. row count info), passed should still be True.
    assert report.passed
    if report.warnings:
        # Sanity: warnings exist but didn't block
        assert all(w.severity == Severity.WARNING for w in report.warnings)