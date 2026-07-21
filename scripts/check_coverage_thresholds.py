"""Enforce MAT-110's per-module branch-enabled coverage budgets.

Coverage.py's ``percent_covered`` combines statements and measured branch exits.
The CI run enables branch measurement globally, then this script applies stricter
budgets to the live-money safety core and each broker adapter.

Why this script exists at all: Coverage.py has exactly ONE global
``fail_under`` setting (pyproject pins it to the repository's 54.7% baseline),
so the stricter 90%/80% per-module budgets have to be enforced from the JSON
report by hand.  The split is deliberate -- the global floor stops overall
erosion, while these budgets stop a specific safety module from quietly losing
its tests.
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
    # Websocket tick-to-bar helpers feed the same live frames/LTP cache as the
    # REST producer, so they carry the same data-safety budget.
    "Dependencies/tick_bar_builder.py": 90.0,
    "Dependencies/next_open_entry.py": 90.0,
    "Dependencies/risk_sizing.py": 90.0,
    # Exchange freeze-limit arithmetic: an off-by-one here means a live order
    # the exchange rejects, which this runner would treat as a paper fallback.
    "Dependencies/order_splitting.py": 90.0,
    # MAT-108's redaction layer is data-safety code: a regression here leaks
    # credentials into logs, so it carries the same 90% budget.
    "Dependencies/secret_redaction.py": 90.0,
}

# EVERY live execution adapter belongs in this dict. A broker added without a
# row here silently escapes the "80% per broker adapter" policy that
# CLAUDE.md/AGENTS.md promise (exactly what happened when the Dhan adapter
# first landed), so treat this list as part of adding a broker.
BROKER_THRESHOLDS = {
    "Dependencies/Kotak API/kotak_execution.py": 80.0,
    "Dependencies/Shoonya API/shoonya_execution.py": 80.0,
    "Dependencies/Flattrade API/flattrade_execution.py": 80.0,
    "Dependencies/Dhan API/dhan_execution.py": 80.0,
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
            # A renamed/deleted module must FAIL the gate, not silently drop
            # out of it -- otherwise moving a file would retire its budget.
            failures.append(f"{path}: missing from coverage report")
            continue
        summary = details.get("summary") or {}
        if int(summary.get("num_branches", 0)) <= 0:
            # Percent alone can look fine while branch measurement was never
            # switched on (a broken [tool.coverage.run] branch=true would
            # silently weaken every budget). Zero measured branches in modules
            # this size is only ever a measurement failure.
            failures.append(f"{path}: branch data was not measured")
            continue
        measured = float(summary.get("percent_covered", 0.0))
        # The epsilon keeps an exactly-on-budget module passing despite float
        # representation (e.g. a true 80.0 stored as 79.999...).
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
