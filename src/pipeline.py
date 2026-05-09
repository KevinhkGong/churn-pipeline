# src/pipeline.py
"""
Pipeline orchestrator.

Runs the full ETL → validate → store → train flow on a single raw CSV.
This is the entrypoint that gets called per daily file.

Failure handling:
  - IngestError       -> file is fundamentally unusable; quarantine and exit
  - Validation fails  -> file is in the wrong shape; quarantine and exit
                         (do NOT train)
  - Validation passes -> store processed parquet, archive raw, train model
  - Training rejected -> processed file kept, raw archived, no new model
                         artifact (previous model continues to serve)

Every run produces:
  - a validation report at reports/validation_<run_date>.json
  - a log entry summarizing the outcome
  - on success: a parquet at data/processed/, raw moved to data/archive/
  - on failure: raw moved to data/quarantine/
  - on training acceptance: a model artifact at models/

Run from the project root:
    python -m src.pipeline --file data/raw/user_activity_2026-05-09.csv
    python -m src.pipeline --date 2026-05-09          # auto-resolves filename
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.utils.schema import load_pipeline_config, load_config
from src.ingest import ingest_csv, IngestError
from src.clean import clean
from src.validate import validate
from src.store import store_processed, archive_raw, quarantine_raw
from src.train import train_model, TrainingResult
from src.validation_report import ValidationReport


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(logs_dir: str | Path, run_date: str) -> None:
    """Configure logging to both stdout and a date-stamped log file."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"pipeline_{run_date}.log"

    fmt = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
        force=True,                # override any prior basicConfig
    )


logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

@dataclass
class PipelineRunSummary:
    """Top-level summary of a single pipeline invocation."""
    run_date: str
    file_path: str
    started_at: str
    finished_at: str = ""
    stage_reached: str = "ingest"          # ingest | clean | validate | store | train | done
    outcome: str = "unknown"               # success | quarantined | rejected_model | failed
    validation_passed: bool | None = None
    n_validation_errors: int = 0
    n_validation_warnings: int = 0
    training_accepted: bool | None = None
    training_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def log_summary(self) -> None:
        line = (
            f"=== Pipeline run {self.run_date} | "
            f"outcome={self.outcome} | "
            f"stage_reached={self.stage_reached} | "
            f"validation_passed={self.validation_passed} | "
            f"training_accepted={self.training_accepted}"
        )
        logger.info(line)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_input_file(
    args_file: str | None,
    args_date: str | None,
    raw_dir: Path,
) -> tuple[Path, str]:
    """Determine which raw file to process and what run_date to use.

    Either --file or --date must be provided. If --date, we look up
    the conventional filename data/raw/user_activity_<date>.csv.
    """
    if args_file:
        file_path = Path(args_file)
        # Try to extract a date from the filename for run_date defaulting
        stem = file_path.stem                            # user_activity_2026-05-09
        run_date = stem.split("_")[-1] if "_" in stem else date.today().isoformat()
        return file_path, run_date

    if args_date:
        file_path = raw_dir / f"user_activity_{args_date}.csv"
        return file_path, args_date

    # Default: today
    today = date.today().isoformat()
    return raw_dir / f"user_activity_{today}.csv", today


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: Path,
    run_date: str,
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
    schema,
) -> PipelineRunSummary:
    """Run the full pipeline for one file. Returns a structured summary."""
    summary = PipelineRunSummary(
        run_date=run_date,
        file_path=str(file_path),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )

    # ---- INGEST -----------------------------------------------------------
    try:
        df = ingest_csv(file_path, schema, pipeline_config)
    except IngestError as e:
        logger.error(f"Ingest failed: {e}")
        summary.outcome = "quarantined"
        summary.error = str(e)
        if file_path.exists():
            quarantine_raw(file_path, pipeline_config)
            summary.artifacts["quarantined_raw"] = str(
                Path(pipeline_config["paths"]["quarantine_dir"]) / file_path.name
            )
        summary.finished_at = datetime.now().isoformat(timespec="seconds")
        return summary

    summary.stage_reached = "clean"

    # ---- CLEAN ------------------------------------------------------------
    df = clean(df, schema, pipeline_config)
    summary.stage_reached = "validate"

    # ---- VALIDATE ---------------------------------------------------------
    report: ValidationReport = validate(
        df, schema, pipeline_config,
        file_path=str(file_path),
        run_date=run_date,
    )

    # Always write the report — pass or fail
    report_path = report.write(pipeline_config["paths"]["reports_dir"])
    summary.artifacts["validation_report"] = str(report_path)
    summary.validation_passed = report.passed
    summary.n_validation_errors = len(report.errors)
    summary.n_validation_warnings = len(report.warnings)

    if not report.passed:
        logger.error(
            f"Validation failed with {len(report.errors)} errors. "
            f"Quarantining and skipping training."
        )
        if file_path.exists():
            quarantine_raw(file_path, pipeline_config)
            summary.artifacts["quarantined_raw"] = str(
                Path(pipeline_config["paths"]["quarantine_dir"]) / file_path.name
            )
        summary.outcome = "quarantined"
        summary.finished_at = datetime.now().isoformat(timespec="seconds")
        return summary

    summary.stage_reached = "store"

    # ---- STORE ------------------------------------------------------------
    processed_path = store_processed(df, run_date, pipeline_config)
    summary.artifacts["processed_parquet"] = str(processed_path)

    # Move raw to archive — it's been processed successfully
    archived_path = archive_raw(file_path, pipeline_config)
    summary.artifacts["archived_raw"] = str(archived_path)

    summary.stage_reached = "train"

    # ---- TRAIN (only on validation pass) ----------------------------------
    if not pipeline_config["training"].get("enabled", True):
        logger.info("Training disabled in config; skipping.")
        summary.outcome = "success"
        summary.stage_reached = "done"
        summary.finished_at = datetime.now().isoformat(timespec="seconds")
        return summary

    result: TrainingResult = train_model(
        df, run_date, pipeline_config, model_config,
    )
    summary.training_accepted = result.accepted
    summary.training_metrics = result.metrics

    if result.accepted:
        summary.artifacts["model"] = result.artifact_path
        summary.outcome = "success"
    else:
        # Training ran but the model didn't pass acceptance thresholds.
        # The processed data is still kept; the previous model continues
        # to serve.
        summary.outcome = "rejected_model"
        logger.warning(
            f"Training completed but model not promoted: "
            f"{result.rejection_reasons}"
        )

    summary.stage_reached = "done"
    summary.finished_at = datetime.now().isoformat(timespec="seconds")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the churn data pipeline")
    p.add_argument("--file", type=str, default=None,
                   help="Path to a raw CSV. If omitted, --date is used.")
    p.add_argument("--date", type=str, default=None,
                   help="Run date YYYY-MM-DD (used to resolve filename). "
                        "Defaults to today.")
    p.add_argument("--pipeline-config", type=str,
                   default="config/pipeline_config.yaml")
    p.add_argument("--model-config", type=str,
                   default="config/model_config.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    pipeline_config, schema = load_pipeline_config(args.pipeline_config)
    model_config = load_config(args.model_config)

    raw_dir = Path(pipeline_config["paths"]["raw_dir"])
    file_path, run_date = resolve_input_file(args.file, args.date, raw_dir)

    setup_logging(pipeline_config["paths"]["logs_dir"], run_date)
    logger.info(f"Starting pipeline run for {file_path} (run_date={run_date})")

    summary = run_pipeline(
        file_path=file_path,
        run_date=run_date,
        pipeline_config=pipeline_config,
        model_config=model_config,
        schema=schema,
    )
    summary.log_summary()

    # Exit code: 0 on success or rejected_model (pipeline ran cleanly),
    # non-zero on quarantine or hard failure (something needs attention)
    if summary.outcome in {"success", "rejected_model"}:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())