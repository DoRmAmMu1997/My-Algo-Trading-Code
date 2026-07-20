"""Fail-closed position sizing for strategies with rupee risk budgets.

The old strategy-specific helpers used ``ceil`` and always forced at least one
lot.  Both behaviours can exceed the configured budget.  ``SizingDecision``
is the single authority used by the master runner and the standalone SL
Hunting executor: invalid inputs and one-lot-over-budget setups are explicit
rejections, while accepted trades use ``floor`` with a configurable lot cap.

Beginner's walkthrough of why floor-and-reject, with a worked example
(budget Rs.2500, NIFTY lot 75, entry 24300, stop 24290 -> 10-point risk):

- one lot risks 10 * 75 = Rs.750, so floor(2500 / 750) = **3 lots**
  (Rs.2250 worst case, inside the budget).
- the old ``ceil`` gave 4 lots = Rs.3000 worst case -- 20% OVER the budget
  the operator configured, on every single trade with an awkward stop.
- with a 40-point stop, one lot risks Rs.3000 > budget.  The old "minimum
  one lot" still traded it; this module REJECTS it, because a budget that
  silently stretches for wide stops is not a budget at all.  A skipped
  setup costs an opportunity; an oversized one costs real money.

The rupee-risk proxy is deliberately simple: ``|entry - stop| * lot_size``
measures the UNDERLYING's move as if the option gained/lost point-for-point
(delta ~= 1).  Real ATM option deltas are nearer 0.5, so this proxy usually
OVERSTATES the risk -- conservative in exactly the direction a hard limit
should be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _finite_number(value: Any) -> float | None:
    """Return a finite float, rejecting booleans and conversion failures."""

    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True, slots=True)
class SizingDecision:
    """Immutable explanation of an accepted or rejected position size.

    Every field the sizing maths used is carried on the decision itself, so a
    log line or Telegram alert can show WHY a trade was sized (or skipped)
    without re-deriving anything.  ``accepted=False`` always comes with
    ``lots=0``/``quantity=0`` -- callers never need to double-check.
    """

    accepted: bool
    lots: int
    lot_size: int
    quantity: int
    risk_points: float
    one_lot_risk: float
    total_risk: float
    budget: float
    max_lots: int
    reason: str

    @classmethod
    def _rejected(
        cls,
        reason: str,
        *,
        lot_size: int = 0,
        risk_points: float = 0.0,
        one_lot_risk: float = 0.0,
        budget: float = 0.0,
        max_lots: int = 0,
    ) -> SizingDecision:
        """Build a zero-quantity rejection without raising into a worker loop."""

        return cls(
            accepted=False,
            lots=0,
            lot_size=max(0, int(lot_size)) if type(lot_size) is int else 0,
            quantity=0,
            risk_points=float(risk_points),
            one_lot_risk=float(one_lot_risk),
            total_risk=0.0,
            budget=float(budget),
            max_lots=max(0, int(max_lots)) if type(max_lots) is int else 0,
            reason=str(reason),
        )

    @classmethod
    def fixed(cls, *, lots: int, lot_size: int) -> SizingDecision:
        """Represent a validated static lot setting used by non-budget strategies."""

        if type(lots) is not int or lots <= 0:
            return cls._rejected("Static lots must be a positive integer.")
        if type(lot_size) is not int or lot_size <= 0:
            return cls._rejected(
                "Exchange lot size must be a positive integer.",
                max_lots=lots,
            )
        return cls(
            accepted=True,
            lots=lots,
            lot_size=lot_size,
            quantity=lots * lot_size,
            risk_points=0.0,
            one_lot_risk=0.0,
            total_risk=0.0,
            budget=0.0,
            max_lots=lots,
            reason="Static strategy lot setting accepted.",
        )

    @classmethod
    def from_risk_budget(
        cls,
        *,
        entry: float,
        stop: float,
        lot_size: int,
        budget: float,
        max_lots: int = 5,
    ) -> SizingDecision:
        """Size with ``min(max_lots, floor(budget / one_lot_risk))``.

        The spot-distance proxy is conservative and intentionally unchanged:
        ``one_lot_risk = abs(entry - stop) * lot_size``.  What changes is the
        hard-limit policy—there is no fallback lot and no rounding upward.

        The validation ladder below runs most-fundamental-first (budget, then
        lot size, then cap, then prices, then stop distance) so the rejection
        ``reason`` always names the FIRST thing the operator must fix, and
        each rejection carries whatever figures were already validated.
        """

        normalized_budget = _finite_number(budget)
        if normalized_budget is None or normalized_budget <= 0.0:
            return cls._rejected("Risk budget must be finite and positive.")
        if type(lot_size) is not int or lot_size <= 0:
            return cls._rejected(
                "Exchange lot size must be a positive integer.",
                budget=normalized_budget,
            )
        if type(max_lots) is not int or max_lots <= 0:
            return cls._rejected(
                "Maximum lots must be a positive integer.",
                lot_size=lot_size,
                budget=normalized_budget,
            )

        normalized_entry = _finite_number(entry)
        normalized_stop = _finite_number(stop)
        if normalized_entry is None or normalized_stop is None:
            return cls._rejected(
                "Entry and stop must be finite numbers.",
                lot_size=lot_size,
                budget=normalized_budget,
                max_lots=max_lots,
            )

        risk_points = abs(normalized_entry - normalized_stop)
        if not math.isfinite(risk_points) or risk_points <= 0.0:
            return cls._rejected(
                "Stop distance must be finite and positive.",
                lot_size=lot_size,
                risk_points=0.0,
                budget=normalized_budget,
                max_lots=max_lots,
            )

        one_lot_risk = risk_points * lot_size
        if not math.isfinite(one_lot_risk) or one_lot_risk <= 0.0:
            return cls._rejected(
                "One-lot risk must be finite and positive.",
                lot_size=lot_size,
                risk_points=risk_points,
                budget=normalized_budget,
                max_lots=max_lots,
            )
        if one_lot_risk > normalized_budget:
            return cls._rejected(
                "One lot exceeds the configured risk budget.",
                lot_size=lot_size,
                risk_points=risk_points,
                one_lot_risk=one_lot_risk,
                budget=normalized_budget,
                max_lots=max_lots,
            )

        affordable_lots = math.floor(normalized_budget / one_lot_risk)
        lots = min(max_lots, affordable_lots)
        if lots <= 0:
            return cls._rejected(
                "No whole lot fits within the configured risk budget.",
                lot_size=lot_size,
                risk_points=risk_points,
                one_lot_risk=one_lot_risk,
                budget=normalized_budget,
                max_lots=max_lots,
            )

        total_risk = lots * one_lot_risk
        return cls(
            accepted=True,
            lots=lots,
            lot_size=lot_size,
            quantity=lots * lot_size,
            risk_points=risk_points,
            one_lot_risk=one_lot_risk,
            total_risk=total_risk,
            budget=normalized_budget,
            max_lots=max_lots,
            reason="Risk-budget sizing accepted.",
        )
