"""MAT-104 regressions for hard per-trade risk-budget sizing."""

from __future__ import annotations

import math

import pytest

from Dependencies.risk_sizing import SizingDecision


def test_floor_sizing_never_exceeds_budget() -> None:
    decision = SizingDecision.from_risk_budget(
        entry=22_500.0,
        stop=22_480.0,
        lot_size=50,
        budget=2_500.0,
        max_lots=5,
    )

    assert decision.accepted is True
    assert decision.one_lot_risk == pytest.approx(1_000.0)
    assert decision.lots == 2
    assert decision.quantity == 100
    assert decision.total_risk == pytest.approx(2_000.0)
    assert decision.total_risk <= decision.budget


def test_one_lot_over_budget_is_rejected_instead_of_forced_to_one() -> None:
    decision = SizingDecision.from_risk_budget(
        entry=22_500.0,
        stop=22_440.0,
        lot_size=50,
        budget=2_500.0,
        max_lots=5,
    )

    assert decision.accepted is False
    assert decision.lots == 0
    assert decision.quantity == 0
    assert decision.one_lot_risk == pytest.approx(3_000.0)
    assert "one lot" in decision.reason.lower()


@pytest.mark.parametrize(
    ("entry", "stop"),
    [
        (22_500.0, 22_500.0),
        (math.nan, 22_450.0),
        (22_500.0, math.nan),
        (math.inf, 22_450.0),
        (22_500.0, -math.inf),
    ],
)
def test_nonfinite_or_nonpositive_risk_is_rejected(entry: float, stop: float) -> None:
    decision = SizingDecision.from_risk_budget(
        entry=entry,
        stop=stop,
        lot_size=50,
        budget=2_500.0,
        max_lots=5,
    )

    assert decision.accepted is False
    assert decision.lots == 0
    assert decision.quantity == 0


def test_tiny_stop_is_capped_at_five_lots() -> None:
    decision = SizingDecision.from_risk_budget(
        entry=22_500.0,
        stop=22_499.9,
        lot_size=50,
        budget=2_500.0,
        max_lots=5,
    )

    assert decision.accepted is True
    assert decision.lots == 5
    assert decision.total_risk == pytest.approx(25.0)


def test_namespaced_max_lots_can_tighten_the_global_default() -> None:
    decision = SizingDecision.from_risk_budget(
        entry=22_500.0,
        stop=22_499.0,
        lot_size=50,
        budget=2_500.0,
        max_lots=3,
    )

    assert decision.accepted is True
    assert decision.lots == 3


@pytest.mark.parametrize(
    ("lot_size", "budget", "max_lots"),
    [
        (0, 2_500.0, 5),
        (50, 0.0, 5),
        (50, math.inf, 5),
        (50, 2_500.0, 0),
    ],
)
def test_invalid_sizing_inputs_fail_closed(
    lot_size: int,
    budget: float,
    max_lots: int,
) -> None:
    decision = SizingDecision.from_risk_budget(
        entry=22_500.0,
        stop=22_490.0,
        lot_size=lot_size,
        budget=budget,
        max_lots=max_lots,
    )

    assert decision.accepted is False
    assert decision.quantity == 0
