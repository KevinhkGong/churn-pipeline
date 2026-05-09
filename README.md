# Churn Prediction Data Pipeline

A daily ML data pipeline that ingests raw user activity CSVs, validates data
quality, stores processed data, and conditionally triggers model training.

The pipeline is designed around one principle: **training only happens on
data we trust**. Every batch passes through layered validation, and any
batch that fails is quarantined for inspection rather than silently used.

---

## Quick start

```bash
# 1. Set up environment
conda create -n churn python=3.11 -y
conda activate churn
pip install -r requirements.txt

# 2. Run the full demo (clean run + failure run, end to end)
make demo

# 3. Or step through individual stages
make data-good DATE=2026-05-09     # generate a clean daily file
make run DATE=2026-05-09           # run the pipeline on it
make data-bad-nulls DATE=2026-05-10
make run DATE=2026-05-10           # watch validation block training

# 4. Run the tests
make test
```

`make help` lists all available targets.

---

## What the pipeline does

```
raw CSV
   |
   v
[ ingest ]   read file, check structure, coerce dtypes
   |
   v
[ clean ]    deterministic, conservative noise removal
   |
   v
[ validate ] structural + content + distribution checks
   |
   +-- fails --> quarantine raw, write report, STOP
   |
   v passes
[ store ]    write parquet to processed/, archive raw
   |
   v
[ train ]    fit model, evaluate, promote only if metrics pass
   |
   v
model artifact (or rejected_model outcome with previous model still serving)
```

Every run produces a JSON validation report — pass or fail — so an operator
can always answer "why did/didn't training happen on day X?"

---

## Scenario

The challenge specifies a churn prediction pipeline: a new CSV of user activity
arrives daily, and the pipeline cleans, validates, and conditionally retrains.
No data was provided, so this project includes a synthetic data generator
(`src/generate_data.py`) that produces realistic churn data with learnable
signal, plus seven distinct "bad data" injection modes for demonstrating the
validation layer.

### Healthcare framing (ScriptChain Health context)

While the take-home brief is framed around SaaS churn, the same architecture
applies directly to ScriptChain's adherence-prediction use case. Patient
adherence drop-off is structurally identical to subscription churn — both are
binary classification on time-series behavioral data with class imbalance.

Adapting this pipeline to a healthcare context would primarily mean editing
`config/pipeline_config.yaml`:

- Replace the schema (patient_id, vitals, medications, adherence_score)
- Tighten null thresholds — clinical data quality has higher stakes
- Add HIPAA-aware checks (PHI detection, de-identification verification)
- Add clinical range validation (e.g., systolic BP between 60-250 mmHg)
- Switch the target metric to recall on non-adherence (catching at-risk
  patients matters more than precision)

The pipeline code itself wouldn't need to change — that's the point of
config-driven validation.

---

## Project layout

```
churn-pipeline/
├── Makefile                     # demo + utility targets
├── README.md
├── requirements.txt
│
├── config/
│   ├── pipeline_config.yaml     # paths, schema, validation rules, cleaning
│   └── model_config.yaml        # features, hyperparameters, acceptance thresholds
│
├── data/
│   ├── raw/                     # incoming daily CSVs (simulated arrivals)
│   ├── processed/               # validated parquet, versioned by date
│   ├── archive/                 # successfully-processed raw files
│   └── quarantine/              # failed-validation raw files (for inspection)
│
├── src/
│   ├── pipeline.py              # orchestrator
│   ├── ingest.py                # read CSV, structural checks
│   ├── clean.py                 # deterministic transformations
│   ├── validate.py              # data quality checks
│   ├── validation_report.py     # report dataclasses
│   ├── store.py                 # parquet write, raw file lifecycle
│   ├── train.py                 # sklearn pipeline + acceptance gate
│   ├── generate_data.py         # synthetic data generator
│   └── utils/
│       └── schema.py            # config loading, schema parsing, dtype coercion
│
├── tests/
│   ├── test_validate.py         # unit tests on each validation category
│   └── test_pipeline_integration.py  # end-to-end smoke tests
│
├── reports/                     # JSON validation reports (per run)
├── models/                      # accepted model artifacts (joblib)
└── logs/                        # per-run log files
```

---

## Design decisions

### 1. Validation is layered (structural → content → distribution)

The validator runs three categories of checks, each catching a different
class of failure:

| Layer | What it checks | Example |
|---|---|---|
| **Structural** | Can the file even be used? | Required columns present, primary key unique, row count in bounds |
| **Content** | Are the values plausible? | Null rates per column, numeric value ranges, allowed categorical values |
| **Distribution** | Does the batch look like prior batches? | Class balance (churn rate in expected range) |

Each check produces a `CheckResult` with a name, category, severity, and
human-readable message. The validator never raises on bad data — every
problem becomes a structured entry in the report.

### 2. Severity is config-driven, not hardcoded

`config/pipeline_config.yaml` lists which check categories block training:

```yaml
validation:
  block_training_on:
    - schema_error
    - primary_key_violation
    - null_rate_exceeded
    - out_of_range_exceeded
    - unknown_category_strict
    - row_count_out_of_bounds
    - churn_rate_out_of_range
```

Anything not in this list becomes a warning that's logged but doesn't
block the pipeline. The business — not the code — decides what's blocking.

### 3. Cleaning is for noise, not for fixing data

Cleaning operations are limited to deterministic, conservative
transformations: strip whitespace, lowercase categoricals, drop duplicates
on the primary key. The only missing-value fill is `num_support_tickets = 0`,
which is a true semantic mapping ("no tickets filed" rather than "we don't
know").

Other missing values are left alone so the validator can count them. Blanket
imputation in cleaning would hide data quality problems from the validator,
which is a common silent-degradation pattern in production ML.

### 4. Imputation lives inside the sklearn Pipeline, not in cleaning

`SimpleImputer`, `StandardScaler`, and `OneHotEncoder` are all *learned*
transformations: their parameters (medians, means, std deviations, category
sets) are fit on training data and stored with the model. They are inside
the sklearn `Pipeline` so that:

- The same medians used in training fill nulls at inference time
- The model artifact (joblib file) contains the entire transformation chain
- Train-serve skew is eliminated by construction

### 5. Class-imbalance-aware training and gating

Churn is rare (~5% positive rate in the synthetic data). The model uses
`class_weight="balanced"` so the rare class isn't drowned out, and the
acceptance threshold uses recall on the churn class (not overall accuracy)
because catching real churners matters more than precision in this domain.

A trained model is only promoted (saved to `models/`) if it meets the
acceptance thresholds in `model_config.yaml`. Models that fail thresholds
are logged with rejection reasons but not saved — the previously-deployed
model continues to serve.

### 6. Auditability by directory layout

After every run, the filesystem alone tells the story:

- `data/archive/` — files that were successfully processed
- `data/quarantine/` — files that failed validation
- `data/processed/` — trainable parquet outputs
- `models/` — accepted model artifacts
- `reports/` — JSON validation reports for *every* run, pass or fail

No need to grep logs to figure out what happened on a given day.

---

## What I deliberately left out

A take-home should fit its scope. These are extensions a production version
would add:

- **Orchestration**: in production this would run as an Airflow DAG or
  Prefect flow with each stage as a task. The `run_pipeline` function is
  designed to map cleanly onto that.
- **Storage**: `data/processed/` would be S3 or a warehouse table; raw
  archive/quarantine would be S3 prefix moves rather than `shutil.move`.
- **Drift detection**: real distribution monitoring (KS test, PSI, feature
  histogram comparisons against a baseline). The architecture supports
  adding this as another check function in the distribution layer.
- **Champion/challenger model promotion**: instead of fixed acceptance
  thresholds, compare the new model's metrics to the currently-deployed
  one and promote only if it's measurably better.
- **Validation library**: I built the validator from scratch rather than
  using Great Expectations or pandera. For a take-home, this makes the
  logic explicit; in production I'd evaluate those libraries.

---

## Running individual stages

The orchestrator is `src.pipeline`, but every stage is also callable
directly for debugging:

```bash
# Generate data only
python -m src.generate_data --mode good --date 2026-05-09
python -m src.generate_data --mode bad --inject high_nulls --date 2026-05-10

# Run the full pipeline on a specific file
python -m src.pipeline --date 2026-05-09
python -m src.pipeline --file data/raw/some_other_file.csv

# Inspect a validation report
cat reports/validation_2026-05-09.json | python -m json.tool
```

### Bad-data injection modes

The synthetic data generator can inject seven kinds of failures, each
mapped to a validation rule:

| Injection | Catches at | Validation category |
|---|---|---|
| `high_nulls` | validate | `null_rate_exceeded` |
| `schema_drift` | ingest | (missing required column) |
| `out_of_range` | validate | `out_of_range_exceeded` |
| `duplicate_keys` | validate | `primary_key_violation` |
| `unknown_category` | validate | `unknown_category_strict` |
| `churn_rate_spike` | validate | `churn_rate_out_of_range` |
| `tiny_batch` | validate | `row_count_out_of_bounds` |

This gives a one-to-one demonstration that each validation rule actually
fires when its corresponding failure occurs.

---

## Tests

```bash
make test
```

Two test files:

- `tests/test_validate.py` — unit tests on the validator. One test per
  failure category plus report-level invariants.
- `tests/test_pipeline_integration.py` — end-to-end smoke tests that run
  the orchestrator in a temp directory and assert files land in the
  correct destinations on both success and failure.

I focused testing on the validator (the differentiating logic) and the
orchestrator's routing (where stage-fitting bugs would hide). Leaf modules
like the cleaner and storer are thin wrappers over well-tested libraries.

---

## Notes on synthetic data

The data generator produces realistic-but-learnable churn data. The churn
label is a logistic function of the features:

- High `days_since_last_login` → higher churn
- Many `num_support_tickets` → higher churn
- Long `tenure_days` → lower churn (loyalty)
- More `num_logins_last_30d` → lower churn (engagement)
- Plan effect: free > basic > premium (free users churn most)
- Plus Gaussian noise so the relationship isn't deterministic

A logistic regression baseline lands around ROC-AUC 0.71 on this data,
which is in the realistic 0.65–0.85 range typical of churn baselines.
Tuning the synthetic data's signal strength is a deliberate design choice —
without learnable signal, the model would be rejected by the acceptance
gate and the demo wouldn't complete. Real churn data has comparable or
stronger signal.

---

*Built by Hongkun (Kevin) Gong for the ScriptChain Health ML Internship Technical Interview.*