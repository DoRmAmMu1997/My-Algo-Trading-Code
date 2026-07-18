"""Enforce MAT-110's per-module branch-enabled coverage budgets.

Coverage.py's ``percent_covered`` combines statements and measured branch exits.
The CI run enables branch measurement globally, then this script applies stricter
budgets to the live-money safety core and each broker adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAFETY_THRESHOLDS = {
    "Dependencies/broker_contract.py": 90.0,
    "Dependencies/execution_ledger.py": 90.0,
    "Dependencies/startup_exposure.py": 90.0,
    "Dependencies/trading_lifecycle.py": 90.0,
    "Dependencies/market_data_health.py": 90.0,
    "Dependencies/next_open_entry.py": 90.0,
    "Dependencies/risk_sizing.py": 90.0,
}

BROKER_THRESHOLDS = {
    "Dependencies/Kotak API/kotak_execution.py": 80.0,
    "Dependencies/Shoonya API/shoonya_execution.py": 80.0,
    "Dependencies/Flattrade API/flattrade_execution.py": 80.0,
}


def _portable_path(value: str) -> str:
    """Normalize Coverage.py paths from Windows or POSIX runners."""
    return str(value).replace("\\", "/").lstrip("./")


def evaluate_coverage(
    report: dict[str, Any],
    *,
    safety_thresholds: dict[str, float] | None = None,
    broker_thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Return deterministic failure messages for missing/under-budget modules."""
    files = {
        _portable_path(path): details
        for path, details in (report.get("files") or {}).items()
    }
    thresholds = {
        **(SAFETY_THRESHOLDS if safety_thresholds is None else safety_thresholds),
        **(BROKER_THRESHOLDS if broker_thresholds is None else broker_thresholds),
    }
    failures: list[str] = []
    for raw_path, threshold in sorted(thresholds.items()):
        path = _portable_path(raw_path)
        details = files.get(path)
        if details is None:
            failures.append(f"{path}: missing from coverage report")
            continue
        summary = details.get("summary") or {}
        if int(summary.get("num_branches", 0)) <= 0:
            failures.append(f"{path}: branch data was not measured")
            continue
        measured = float(summary.get("percent_covered", 0.0))
        if measured + 1e-9 < float(threshold):
            failures.append(
                f"{path}: {measured:.2f}% is below the {float(threshold):.1f}% budget"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="coverage.json")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    failures = evaluate_coverage(report)
    if failures:
        print("Coverage policy failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Coverage policy passed for all safety and broker modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
