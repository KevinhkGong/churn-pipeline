# Makefile — convenience commands for the churn data pipeline
#
# Usage:
#   make help                   # see all targets
#   make setup                  # install dependencies
#   make data-good              # generate one clean daily file (today's date)
#   make data-bad-nulls         # generate a file with 40% null injection
#   make run                    # run the pipeline on today's file
#   make demo                   # full end-to-end demo: clean run + failure run
#   make test                   # run the test suite
#   make clean-data             # wipe data/ and reports/ for a fresh demo

.DEFAULT_GOAL := help

# ---- Configurable variables -----------------------------------------------
DATE        ?= $(shell date +%Y-%m-%d)
BAD_DATE    ?= $(shell date -d "+1 day" +%Y-%m-%d 2>/dev/null || date -v +1d +%Y-%m-%d)
PYTHON      ?= python

# ---- Help -----------------------------------------------------------------
.PHONY: help
help:
	@echo "Churn Pipeline — available targets:"
	@echo ""
	@echo "  setup              Install dependencies from requirements.txt"
	@echo ""
	@echo "  Data generation:"
	@echo "    data-good        Generate a clean daily CSV (DATE=$(DATE))"
	@echo "    data-bad-nulls   Generate a CSV with 40% null injection"
	@echo "    data-bad-schema  Generate a CSV with a missing column"
	@echo "    data-bad-dupes   Generate a CSV with duplicate user_ids"
	@echo "    data-bad-cat     Generate a CSV with an unknown category"
	@echo "    data-bad-spike   Generate a CSV with a churn rate spike"
	@echo "    data-bad-tiny    Generate a CSV below the row-count minimum"
	@echo "    data-bad-range   Generate a CSV with out-of-range values"
	@echo ""
	@echo "  Pipeline:"
	@echo "    run              Run pipeline on DATE=$(DATE)"
	@echo "    demo             End-to-end: clean run, then failure run"
	@echo ""
	@echo "  Cleanup & test:"
	@echo "    test             Run pytest"
	@echo "    clean-data       Remove generated data/, reports/, models/, logs/"
	@echo "    clean-pyc        Remove Python bytecode caches"
	@echo ""
	@echo "Override DATE on the command line: make run DATE=2026-05-09"

# ---- Setup ----------------------------------------------------------------
.PHONY: setup
setup:
	$(PYTHON) -m pip install -r requirements.txt

# ---- Data generation ------------------------------------------------------
.PHONY: data-good data-bad-nulls data-bad-schema data-bad-dupes \
        data-bad-cat data-bad-spike data-bad-tiny data-bad-range

data-good:
	$(PYTHON) -m src.generate_data --mode good --date $(DATE)

data-bad-nulls:
	$(PYTHON) -m src.generate_data --mode bad --inject high_nulls --date $(DATE)

data-bad-schema:
	$(PYTHON) -m src.generate_data --mode bad --inject schema_drift --date $(DATE)

data-bad-dupes:
	$(PYTHON) -m src.generate_data --mode bad --inject duplicate_keys --date $(DATE)

data-bad-cat:
	$(PYTHON) -m src.generate_data --mode bad --inject unknown_category --date $(DATE)

data-bad-spike:
	$(PYTHON) -m src.generate_data --mode bad --inject churn_rate_spike --date $(DATE)

data-bad-tiny:
	$(PYTHON) -m src.generate_data --mode bad --inject tiny_batch --date $(DATE)

data-bad-range:
	$(PYTHON) -m src.generate_data --mode bad --inject out_of_range --date $(DATE)

# ---- Pipeline -------------------------------------------------------------
.PHONY: run
run:
	$(PYTHON) -m src.pipeline --date $(DATE)

# Self-contained end-to-end demo. Generates a fresh good file and a fresh
# bad file, runs the pipeline against each, and prints a summary.
.PHONY: demo
demo: clean-data
	@echo ""
	@echo "===== DEMO 1: HAPPY PATH (good data, validation passes, training fires) ====="
	@echo ""
	$(PYTHON) -m src.generate_data --mode good --date $(DATE)
	$(PYTHON) -m src.pipeline --date $(DATE)
	@echo ""
	@echo "===== DEMO 2: FAILURE PATH (40% null rate, validation fails, training blocked) ====="
	@echo ""
	$(PYTHON) -m src.generate_data --mode bad --inject high_nulls --date $(BAD_DATE)
	-$(PYTHON) -m src.pipeline --date $(BAD_DATE)
	@echo ""
	@echo "===== FINAL STATE ====="
	@echo "archive/    (successful runs):"; ls data/archive/ 2>/dev/null || echo "  (empty)"
	@echo "quarantine/ (failed runs):";    ls data/quarantine/ 2>/dev/null || echo "  (empty)"
	@echo "processed/  (trainable data):"; ls data/processed/ 2>/dev/null || echo "  (empty)"
	@echo "models/     (accepted models):"; ls models/ 2>/dev/null || echo "  (empty)"
	@echo "reports/    (validation reports for every run):"; ls reports/ 2>/dev/null || echo "  (empty)"

# ---- Test -----------------------------------------------------------------
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

# ---- Cleanup --------------------------------------------------------------
.PHONY: clean-data
clean-data:
	@echo "Wiping data outputs (preserving directory structure)..."
	@find data/raw data/processed data/quarantine data/archive \
	      reports models logs \
	      -type f ! -name '.gitkeep' -delete 2>/dev/null || true

.PHONY: clean-pyc
clean-pyc:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true