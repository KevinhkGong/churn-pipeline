                                                                                                                                                                                                                                                                                                                  # src/generate_data.py
"""
Synthetic churn data generator.

Generates realistic-but-learnable user activity CSVs that match the schema
defined in pipeline_config.yaml. Supports two modes:

  good:  clean data that should pass validation
  bad:   data with deliberately injected problems to demonstrate the
         pipeline's validation layer catching them and blocking training

Usage:
    # Generate one clean daily file
    python -m src.generate_data --mode good --date 2026-05-09

    # Generate a file with injected null-rate problem
    python -m src.generate_data --mode bad --inject high_nulls --date 2026-05-10

    # Generate a week of clean files
    python -m src.generate_data --mode good --days 7 --start-date 2026-05-01
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.schema import load_pipeline_config


# ---------------------------------------------------------------------------
# Core generator: clean realistic data
# ---------------------------------------------------------------------------

def generate_clean_batch(
    n_rows: int,
    run_date: date,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a batch of realistic synthetic churn data.

    Designs the data so that churn has *learnable signal*:
      - users with high days_since_last_login churn more
      - users with many support tickets churn more
      - longer-tenure users churn less
      - premium plans churn less than free
    Plus noise so it's not deterministic.
    """
    rng = np.random.default_rng(seed)

    # User IDs: simple sequential within batch, prefixed with date for global
    # uniqueness across daily files.
    date_str = run_date.strftime("%Y%m%d")
    user_ids = [f"u_{date_str}_{i:06d}" for i in range(n_rows)]

    # Tenure: exponential distribution (most users are newer; long tail of
    # established users). Cap at ~10 years.
    tenure_days = rng.exponential(scale=400, size=n_rows).astype(int)
    tenure_days = np.clip(tenure_days, 1, 3650)

    # Signup date is derived from tenure relative to today
    signup_dates = [run_date - timedelta(days=int(t)) for t in tenure_days]

    # Plan type: weighted toward free, fewer premium
    plan_type = rng.choice(
        ["free", "basic", "premium"],
        size=n_rows,
        p=[0.55, 0.30, 0.15],
    )

    # Monthly charges: depends on plan
    plan_to_charge = {"free": (0, 0), "basic": (10, 25), "premium": (40, 80)}
    monthly_charges = np.array([
        rng.uniform(*plan_to_charge[p]) for p in plan_type
    ]).round(2)

    # Total charges: monthly * tenure_months, with some noise
    tenure_months = tenure_days / 30.0
    total_charges = (monthly_charges * tenure_months
                     * rng.uniform(0.85, 1.15, size=n_rows)).round(2)

    # Engagement features. avg_session_minutes correlates with logins.
    num_logins_last_30d = rng.poisson(lam=15, size=n_rows)
    num_logins_last_30d = np.clip(num_logins_last_30d, 0, 200)

    avg_session_minutes = rng.gamma(shape=2.0, scale=8.0, size=n_rows).round(1)
    avg_session_minutes = np.clip(avg_session_minutes, 0.1, 180.0)

    num_support_tickets = rng.poisson(lam=0.8, size=n_rows)
    num_support_tickets = np.clip(num_support_tickets, 0, 50)

    # Days since last login: most are recent, some are very stale
    days_since_last_login = rng.exponential(scale=5, size=n_rows).astype(int)
    days_since_last_login = np.clip(days_since_last_login, 0, 365)

    payment_method = rng.choice(
        ["card", "bank", "paypal"],
        size=n_rows,
        p=[0.65, 0.20, 0.15],
    )

    # is_active: roughly true if logged in within last 30 days
    is_active = days_since_last_login <= 30

    # ---- The churn label: a logistic function of the features ----
    # Higher churn probability for: stale users, many tickets, free plan,
    # short tenure. Plus noise.
    plan_churn_bias = pd.Series(plan_type).map(
        {"free": 0.5, "basic": 0.0, "premium": -0.6}
    ).to_numpy()

    churn_score = (
        -3.5                                           # baseline (low churn)
        + 0.10 * days_since_last_login                 # main signal — STRONGER
        + 0.6  * num_support_tickets                   # friction — STRONGER
        - 0.002 * tenure_days                          # loyalty — STRONGER
        - 0.04 * num_logins_last_30d                   # engagement — STRONGER
        + plan_churn_bias                              # plan effect (unchanged)
        + rng.normal(0, 0.3, size=n_rows)              # noise — REDUCED
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned = (rng.uniform(size=n_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "user_id": user_ids,
        "signup_date": signup_dates,
        "tenure_days": tenure_days,
        "plan_type": plan_type,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_logins_last_30d": num_logins_last_30d,
        "avg_session_minutes": avg_session_minutes,
        "num_support_tickets": num_support_tickets,
        "days_since_last_login": days_since_last_login,
        "payment_method": payment_method,
        "is_active": is_active,
        "churned": churned,
    })

    return df


# ---------------------------------------------------------------------------
# Bad-data injection: deliberately corrupt a clean batch in specific ways
# ---------------------------------------------------------------------------

INJECTION_TYPES = [
    "high_nulls",
    "schema_drift",
    "out_of_range",
    "duplicate_keys",
    "unknown_category",
    "churn_rate_spike",
    "tiny_batch",
]


def inject_problem(df: pd.DataFrame, problem: str, seed: int | None = None) -> pd.DataFrame:
    """Mutate a clean dataframe to introduce a specific data quality issue.

    Each injection corresponds to a category of failure the validator
    should catch. Useful for end-to-end demos.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if problem == "high_nulls":
        # Blow out the null rate on monthly_charges (threshold is 5%)
        idx = rng.choice(df.index, size=int(len(df) * 0.40), replace=False)
        df.loc[idx, "monthly_charges"] = np.nan

    elif problem == "schema_drift":
        # Drop a required column to simulate an upstream schema change
        df = df.drop(columns=["num_support_tickets"])

    elif problem == "out_of_range":
        # 10% of rows get nonsense values (>2% threshold triggers ERROR)
        idx = rng.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[idx, "tenure_days"] = -50
        df.loc[idx, "monthly_charges"] = 9999.0

    elif problem == "duplicate_keys":
        # Duplicate the first 5% of user_ids by overwriting later rows
        n_dupes = max(5, int(len(df) * 0.05))
        df.iloc[-n_dupes:, df.columns.get_loc("user_id")] = (
            df.iloc[:n_dupes]["user_id"].values
        )

    elif problem == "unknown_category":
        # Inject an unseen plan_type
        idx = rng.choice(df.index, size=int(len(df) * 0.05), replace=False)
        df.loc[idx, "plan_type"] = "enterprise"

    elif problem == "churn_rate_spike":
        # Force the churn rate way out of expected range (1%-30%)
        df["churned"] = (rng.uniform(size=len(df)) < 0.65).astype(int)

    elif problem == "tiny_batch":
        # Truncate to below the row_count minimum
        df = df.iloc[:50]

    else:
        raise ValueError(
            f"Unknown injection type: {problem!r}. "
            f"Valid options: {INJECTION_TYPES}"
        )

    return df


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, run_date: date, raw_dir: Path) -> Path:
    """Write a generated dataframe to data/raw/ with a date-stamped name."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"user_activity_{run_date.strftime('%Y-%m-%d')}.csv"
    out_path = raw_dir / filename
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic churn CSVs")
    p.add_argument("--mode", choices=["good", "bad"], default="good",
                   help="good = clean data, bad = inject a problem")
    p.add_argument("--inject", choices=INJECTION_TYPES, default=None,
                   help="Which problem to inject (required if --mode bad)")
    p.add_argument("--rows", type=int, default=5000,
                   help="Rows per daily file (default 5000)")
    p.add_argument("--date", type=str, default=None,
                   help="Run date YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=1,
                   help="Number of consecutive days to generate")
    p.add_argument("--start-date", type=str, default=None,
                   help="If --days > 1, the start date YYYY-MM-DD")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "bad" and args.inject is None:
        raise SystemExit("--inject is required when --mode is 'bad'")

    config, _schema = load_pipeline_config(args.config)
    raw_dir = Path(config["paths"]["raw_dir"])

    # Determine the date(s) to generate
    if args.days > 1:
        start = (datetime.strptime(args.start_date, "%Y-%m-%d").date()
                 if args.start_date else date.today())
        run_dates = [start + timedelta(days=i) for i in range(args.days)]
    else:
        run_dates = [
            datetime.strptime(args.date, "%Y-%m-%d").date()
            if args.date else date.today()
        ]

    for i, run_date in enumerate(run_dates):
        # Use a different seed offset per day so files aren't identical
        seed = args.seed + i
        df = generate_clean_batch(n_rows=args.rows, run_date=run_date, seed=seed)

        if args.mode == "bad":
            df = inject_problem(df, args.inject, seed=seed)

        out_path = write_csv(df, run_date, raw_dir)
        print(f"[{run_date}] mode={args.mode}"
              f"{f' inject={args.inject}' if args.mode == 'bad' else ''}"
              f" rows={len(df)} -> {out_path}")


if __name__ == "__main__":
    main()