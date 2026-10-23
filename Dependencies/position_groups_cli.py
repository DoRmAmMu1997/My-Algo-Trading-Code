"""
``python algo.py group-legs`` -- fetch carried legs into the position-groups file.

Two modes:

    python algo.py group-legs --broker zerodha     # add new broker legs to "unassigned"
    python algo.py group-legs --check              # validate the file, print the groups

Fetching logs in with today's token (see ``setup-broker-token``), reads the net
positions, and appends every option leg not already in the file to its
``unassigned`` list.  It never edits anything already there: a changed
quantity or a leg the broker no longer holds is printed for you to fix by hand.
Then open the file, move legs from ``unassigned`` into a group, and set that
group's ``target`` (and ``booked_pnl`` if the strategy has already booked some).

Read-only towards the broker.  The only file written is the groups file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEPENDENCIES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPENDENCIES_DIR.parent
for _import_root in (_REPO_ROOT, _DEPENDENCIES_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import position_brokers  # noqa: E402
from position_groups import (  # noqa: E402
    GroupConfigError,
    load_config,
    merge_broker_legs,
    parse_config,
    read_document,
    write_document_atomic,
)

DEFAULT_FILE = _DEPENDENCIES_DIR / "position_groups.json"


def _check(path: Path) -> int:
    try:
        config = load_config(path)
    except (GroupConfigError, OSError) as exc:
        problems = exc.problems if isinstance(exc, GroupConfigError) else [str(exc)]
        print(f"INVALID: {path}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {path} -- {len(config.groups)} group(s), {len(config.unassigned)} unassigned leg(s).")
    for group in config.groups:
        print(
            f"  {group.name} [{group.broker}] target {group.target:,.0f}, "
            f"booked {group.booked_pnl:,.0f}, {len(group.legs)} leg(s)"
        )
        for leg in group.legs:
            print(f"    {leg.quantity:>+7} {leg.symbol} @ {leg.entry_price:g}")
    return 0


def _fetch(path: Path, broker: str, client: object | None = None) -> int:
    try:
        document = read_document(path)
        parse_config(document)
    except (GroupConfigError, OSError, ValueError) as exc:
        print(f"Refusing to touch {path}: it is not valid right now ({exc}).")
        print("Fix it (python algo.py group-legs --check lists the problems) and retry.")
        return 1
    if client is None:
        client = position_brokers.POSITION_CLIENT_CLASSES[broker]()
    result = client.list_position_details()  # type: ignore[attr-defined]
    if result.is_indeterminate:
        print(f"ERROR: could not read {broker} positions: {result.reason}")
        return 1
    report = merge_broker_legs(document, broker, result.items)
    if report.added:
        write_document_atomic(path, document)
    print(f"{len(report.added)} new leg(s) added to 'unassigned' in {path}:")
    for leg in report.added:
        print(f"  {leg.quantity:>+7} {leg.symbol} @ {leg.entry_price:g} (check this entry price)")
    for symbol in report.unidentified:
        print(f"  skipped {symbol}: not an NFO/BFO option the broker's instrument list knows")
    for line in report.quantity_changed:
        print(f"  QUANTITY DIFFERS (left as-is, fix by hand): {line}")
    for symbol in report.no_longer_held:
        print(f"  NO LONGER HELD at {broker} (left as-is, remove by hand): {symbol}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch carried legs into the position-groups file.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--broker",
        choices=sorted(name.lower() for name in position_brokers.POSITION_CLIENT_CLASSES),
    )
    mode.add_argument("--check", action="store_true", help="Validate the file and print it.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    if args.check:
        return _check(args.file)
    return _fetch(args.file, args.broker.upper())


if __name__ == "__main__":
    sys.exit(main())
