# src/validation_report.py
"""
Data structures for validation results.

Kept in a separate module from validate.py because both the validator
(which produces reports) and the orchestrator/report-writer (which
consume them) depend on these types. Splitting them avoids circular
imports and makes the report shape explicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    """Severity levels for individual checks.

    ERROR    -> blocks training
    WARNING  -> logged and surfaced, but training proceeds
    INFO     -> informational only (e.g. summary stats)

    Inheriting from str makes these JSON-serializable as their string value.
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str                           # e.g. "null_rate_monthly_charges"
    category: str                       # e.g. "null_rate_exceeded" — must match
                                        # the names in config's block_training_on
    passed: bool
    severity: Severity
    message: str                        # human-readable explanation
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate result of validating a single batch."""
    file_path: str
    run_date: str                       # ISO date string
    timestamp: str                      # ISO datetime when validation ran
    row_count: int
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    # ---- Derived properties -----------------------------------------------

    @property
    def passed(self) -> bool:
        """True if no ERROR-level checks failed.

        Warnings don't block; only errors do. This is what the orchestrator
        reads to decide whether to trigger training.
        """
        return not any(
            (not c.passed) and c.severity == Severity.ERROR
            for c in self.checks
        )

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks
                if (not c.passed) and c.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks
                if (not c.passed) and c.severity == Severity.WARNING]

    # ---- Mutation ---------------------------------------------------------

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    # ---- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed              # include derived flag
        d["n_errors"] = len(self.errors)
        d["n_warnings"] = len(self.warnings)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write(self, output_dir: str | Path) -> Path:
        """Write the report to <output_dir>/validation_<run_date>.json."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"validation_{self.run_date}.json"
        out_path.write_text(self.to_json())
        return out_path

    # ---- Constructor helper -----------------------------------------------

    @classmethod
    def new(cls, file_path: str, run_date: str, row_count: int) -> "ValidationReport":
        return cls(
            file_path=str(file_path),
            run_date=run_date,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            row_count=row_count,
        )