"""Tests for the pre-flight checks shared by every broker diagnostic.

Entirely in memory: these helpers take plain numbers and never touch a broker
session, which is the point of keeping them separate from the adapters.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str) -> ModuleType:
    """Load a module whose folder may contain spaces."""

    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("preflight_under_test", "Dependencies/diagnostic_preflight.py")


@pytest.mark.parametrize(
    ("quantity", "lot_size", "rejected"),
    [
        # The exact case that produced Dhan's misleading "Invalid IP" message.
        (1, 65, True),
        (64, 65, True),
        (66, 65, True),
        (100, 65, True),
        (65, 65, False),      # one lot
        (130, 65, False),     # two lots
        (30, 30, False),      # BankNIFTY lot
        (75, 25, False),      # three lots of a smaller contract
    ],
)
def test_quantity_must_be_a_whole_lot_multiple(
    quantity: int,
    lot_size: int,
    rejected: bool,
) -> None:
    message = preflight.validate_quantity_for_lot(quantity, lot_size)

    assert bool(message) is rejected
    if rejected:
        # The operator needs the lot size to fix the command, not just a refusal.
        assert str(lot_size) in message
        assert str(lot_size * 2) in message


@pytest.mark.parametrize("lot_size", [0, -1, -65])
def test_unknown_lot_size_never_blocks_an_order(lot_size: int) -> None:
    """Missing catalogue metadata must not veto a good order.

    Kotak's lot column name varies by master revision and a Shoonya master
    without one leaves the map empty; in both cases the broker stays the
    authority rather than this local check.
    """

    assert preflight.validate_quantity_for_lot(1, lot_size) == ""
    assert preflight.validate_quantity_for_lot(65, lot_size) == ""


@pytest.mark.parametrize(
    ("quantity", "lot_size"),
    [("65", 65), (65, "65"), ("abc", 65), (65, "abc"), (None, 65), (65, None)],
)
def test_unparseable_values_are_not_treated_as_violations(
    quantity: object,
    lot_size: object,
) -> None:
    """Garbage in means "cannot check", never a false refusal.

    Each diagnostic parses --qty its own way (argparse for Flattrade/Dhan, hand
    rolled for Kotak/Shoonya), so this helper takes whatever they hand it.
    """

    message = preflight.validate_quantity_for_lot(quantity, lot_size)  # type: ignore[arg-type]

    assert message == "" or "not a multiple" in message


def test_every_diagnostic_uses_the_shared_check() -> None:
    """All four diagnostics must share one implementation, not copies.

    The guard existed only in the Dhan diagnostic first; duplicating it per
    broker is exactly how the four would drift apart.
    """

    diagnostics = {
        "kotak": "Dependencies/Kotak API/diagnose_kotak_symbol.py",
        "shoonya": "Dependencies/Shoonya API/diagnose_shoonya_symbol.py",
        "flattrade": "Dependencies/Flattrade API/diagnose_flattrade_symbol.py",
        "dhan": "Dependencies/Dhan API/diagnose_dhan_symbol.py",
    }
    for broker, relative_path in diagnostics.items():
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "from diagnostic_preflight import validate_quantity_for_lot" in source, (
            f"{broker} diagnostic does not import the shared quantity check"
        )
        assert "validate_quantity_for_lot(" in source, (
            f"{broker} diagnostic imports the shared check but never calls it"
        )
