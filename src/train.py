# src/train.py
"""
Training stage: fit a churn classifier on the processed data and persist
the model artifact.

This stage runs ONLY when validation passes. The decision logic lives in
the orchestrator (pipeline.py); this module just trains.

Scope notes for the take-home:
  - The prompt says "trigger model training," not "build the best model."
    A logistic regression with sensible preprocessing is the right scope.
  - We compute test-set metrics so the orchestrator can apply the
    acceptance thresholds in model_config.yaml. A model that fails those
    thresholds is logged but NOT promoted — the previous artifact stays
    in service.
  - In production, this would be a separate job (different compute,
    different schedule). For the take-home, it's an in-process call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Summary of a single training run."""
    run_date: str
    timestamp: str
    model_type: str
    n_train: int
    n_test: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    accepted: bool = False               # passed acceptance thresholds?
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _build_sklearn_pipeline(
    model_cfg: dict[str, Any],
) -> Pipeline:
    """Build the preprocessing + model sklearn Pipeline.

    Imputation lives INSIDE this pipeline, not in our cleaning stage,
    because imputation is a learned transformation that should be fit
    on training data and applied identically to test/serving data.
    """
    numeric_features = model_cfg["features"]["numeric"]
    categorical_features = model_cfg["features"]["categorical"]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    model_type = model_cfg["type"]
    hp = model_cfg["hyperparameters"][model_type]

    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=model_cfg["random_state"], **hp)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(random_state=model_cfg["random_state"], **hp)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    run_date: str,
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
) -> TrainingResult:
    """Fit a model on the processed data and persist it if it meets thresholds.

    Returns a TrainingResult regardless of acceptance — rejected models
    are still logged so we can audit why they didn't ship.
    """
    model_cfg = model_config["model"]
    target = model_cfg["target"]

    # ---- Minimum row sanity check ----------------------------------------
    min_rows = pipeline_config["training"].get("min_training_rows", 100)
    if len(df) < min_rows:
        logger.warning(
            f"Skipping training: {len(df)} rows < min_training_rows {min_rows}"
        )
        return TrainingResult(
            run_date=run_date,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            model_type=model_cfg["type"],
            n_train=0, n_test=0,
            accepted=False,
            rejection_reasons=[f"Insufficient rows: {len(df)} < {min_rows}"],
        )

    # ---- Feature/target split --------------------------------------------
    feature_cols = model_cfg["features"]["numeric"] + model_cfg["features"]["categorical"]
    X = df[feature_cols]
    y = df[target].astype(int)

    # ---- Train/test split ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
        stratify=y,                                  # preserve churn ratio
    )

    # ---- Fit -------------------------------------------------------------
    pipeline = _build_sklearn_pipeline(model_cfg)
    logger.info(
        f"Training {model_cfg['type']} on {len(X_train)} rows "
        f"({len(feature_cols)} features)"
    )
    pipeline.fit(X_train, y_train)

    # ---- Evaluate --------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(f"Test metrics: {metrics}")

    # ---- Acceptance check ------------------------------------------------
    thresholds = model_config.get("acceptance_thresholds", {})
    rejection_reasons: list[str] = []
    if metrics["roc_auc"] < thresholds.get("min_roc_auc", 0.0):
        rejection_reasons.append(
            f"ROC-AUC {metrics['roc_auc']} < {thresholds['min_roc_auc']}"
        )
    if metrics["recall"] < thresholds.get("min_recall_churn_class", 0.0):
        rejection_reasons.append(
            f"Recall {metrics['recall']} < {thresholds['min_recall_churn_class']}"
        )

    accepted = len(rejection_reasons) == 0

    # ---- Persist artifact (only if accepted) -----------------------------
    artifact_path = ""
    if accepted:
        models_dir = Path(pipeline_config["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        filename = model_config["artifact"]["filename_template"].format(
            run_date=run_date
        )
        artifact_path = str(models_dir / filename)
        joblib.dump(pipeline, artifact_path)
        logger.info(f"Model accepted and saved to {artifact_path}")
    else:
        logger.warning(
            f"Model REJECTED — not promoted. Reasons: {rejection_reasons}"
        )

    return TrainingResult(
        run_date=run_date,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model_type=model_cfg["type"],
        n_train=len(X_train),
        n_test=len(X_test),
        metrics=metrics,
        artifact_path=artifact_path,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
    )