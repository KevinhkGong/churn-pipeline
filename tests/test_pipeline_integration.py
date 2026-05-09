# tests/test_pipeline_integration.py
"""
End-to-end smoke test: generate data, run the full pipeline, verify
outputs land in the right places.

This is a single test that covers the most important property: the
orchestrator routes successful and failed batches to the correct
directories and produces all expected artifacts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.generate_data import generate_clean_batch, inject_problem, write_csv
from src.pipeline import run_pipeline
from src.utils.schema import load_pipeline_config, load_config


@pytest.fixture
def tmp_pipeline_env(tmp_path, monkeypatch):
    """Run the pipeline in an isolated temporary directory.

    Copies the config files to the temp dir, redirects all data paths
    to subdirectories of it, and returns the loaded configs + schema.
    """
    project_root = Path(__file__).parent.parent

    # Copy config files
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(project_root / "config" / "pipeline_config.yaml", cfg_dir)
    shutil.copy(project_root / "config" / "model_config.yaml", cfg_dir)

    # Run inside tmp_path so all relative paths resolve there
    monkeypatch.chdir(tmp_path)

    pipeline_config, schema = load_pipeline_config("config/pipeline_config.yaml")
    model_config = load_config("config/model_config.yaml")
    return pipeline_config, model_config, schema, tmp_path


def test_good_batch_full_pipeline(tmp_pipeline_env):
    """Clean data should pass validation, train, and produce all artifacts."""
    pipeline_config, model_config, schema, tmp_path = tmp_pipeline_env

    from datetime import date
    run_date = "2026-01-15"
    df = generate_clean_batch(n_rows=2000, run_date=date(2026, 1, 15), seed=42)
    raw_path = write_csv(df, date(2026, 1, 15), Path("data/raw"))

    summary = run_pipeline(
        file_path=raw_path,
        run_date=run_date,
        pipeline_config=pipeline_config,
        model_config=model_config,
        schema=schema,
    )

    assert summary.outcome == "success"
    assert summary.validation_passed is True
    assert summary.training_accepted is True
    assert (tmp_path / "data" / "archive" / raw_path.name).exists()
    assert not raw_path.exists()                                   # moved out of raw/
    assert (tmp_path / "data" / "processed" / f"processed_{run_date}.parquet").exists()
    assert (tmp_path / "reports" / f"validation_{run_date}.json").exists()


def test_bad_batch_quarantines_and_does_not_train(tmp_pipeline_env):
    """High null rate should fail validation, quarantine the file, skip training."""
    pipeline_config, model_config, schema, tmp_path = tmp_pipeline_env

    from datetime import date
    run_date = "2026-01-16"
    df = generate_clean_batch(n_rows=2000, run_date=date(2026, 1, 16), seed=42)
    df = inject_problem(df, "high_nulls", seed=42)
    raw_path = write_csv(df, date(2026, 1, 16), Path("data/raw"))

    summary = run_pipeline(
        file_path=raw_path,
        run_date=run_date,
        pipeline_config=pipeline_config,
        model_config=model_config,
        schema=schema,
    )

    assert summary.outcome == "quarantined"
    assert summary.validation_passed is False
    assert summary.training_accepted is None                      # never reached
    assert (tmp_path / "data" / "quarantine" / raw_path.name).exists()
    assert not (tmp_path / "data" / "archive" / raw_path.name).exists()
    # No model should have been written
    models_dir = tmp_path / "models"
    if models_dir.exists():
        assert list(models_dir.glob("*.joblib")) == []
    # But the validation report should exist
    assert (tmp_path / "reports" / f"validation_{run_date}.json").exists()