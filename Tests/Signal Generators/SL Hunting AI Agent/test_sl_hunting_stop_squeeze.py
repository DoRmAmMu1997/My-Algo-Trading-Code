"""SLH-025: a stop refused for sizing may not be squeezed to fit (operator, 21 Oct).

On 2026-10-21 the order tool refused a LONG at the first-candle-low stop 22,981
for sizing, then at 23,008, and the agent tightened to 23,011 to get one lot --
three attempts in sixteen seconds, inside one decision pass. v5t and v5u already
said "if the real stop does not fit the budget, HOLD"; the prose did not bind.
Since July this happened three times (two of them lost).

Two halves, both pinned here:

* the NUDGE: a sizing refusal tells the model, in the result it reads, not to
  tighten the stop to fit -- and is tagged ``refusal_kind: "sizing"`` so the tool
  never has to parse message text;
* the GUARD: after a sizing refusal, the order tool refuses any further ENTER in
  the same direction with a TIGHTER stop for the rest of that decision pass.

Scope is deliberately one pass, as proposed: the opposite direction, a non-sizing
refusal and the next bar's fresh context are not touched by it.
"""

from __future__ import annotations

import pandas as pd
from sl_hunting_executor import SIZING_REFUSAL_GUIDANCE, MasterWorkerExecutor, StandaloneExecutor
from sl_hunting_tools import SLHuntingToolContext


def _candles(n=30, start_price=25000.0):
    ts = pd.date_range(start="2026-06-26 09:15", periods=n, freq="5min")
    closes = [start_price + i for i in range(n)]
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 2, min(prev, c) - 2, c))
        prev = c
    return pd.DataFrame({
        "timestamp": ts,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * n,
    })


# The synthetic series ends at 25029. With the default budget one 75-unit lot
# allows a stop about 33 points away: 49 points is refused, 29 points fits.
WIDE_STOP = 24980.0
TIGHT_STOP = 25000.0
TARGET = 25100.0


# --------------------------------------------------------------------------
# The nudge: sizing refusals say so, and are tagged
# --------------------------------------------------------------------------

def test_a_paper_sizing_refusal_is_tagged_and_says_hold():
    result = StandaloneExecutor().enter("LONG", WIDE_STOP, TARGET, "first-candle low", price=25029.0)

    assert result["accepted"] is False
    assert result["refusal_kind"] == "sizing"
    assert SIZING_REFUSAL_GUIDANCE in result["reason"]
    assert "Do not tighten the stop" in SIZING_REFUSAL_GUIDANCE
    assert "HOLD" in SIZING_REFUSAL_GUIDANCE


class _Pos:
    active = False


class _RefusingWorker:
    def __init__(self, refusal):
        self.pos = _Pos()
        self.last_entry_refusal = refusal

    def enter_position(self, direction, entry_underlying, stop_underlying=0.0, target_underlying=0.0):
        return False


def test_a_master_sizing_refusal_is_tagged_and_says_hold():
    worker = _RefusingWorker("sizing: risk per lot 4260.75 exceeds budget. nothing was placed.")
    result = MasterWorkerExecutor(worker).enter("LONG", 22981.0, 23100.0, "first-candle low", price=23046.55)

    assert result["accepted"] is False
    assert result["refusal_kind"] == "sizing"
    assert result["reason"].startswith("worker refused the entry: sizing:")
    assert SIZING_REFUSAL_GUIDANCE in result["reason"]


def test_a_master_non_sizing_refusal_is_not_tagged():
    worker = _RefusingWorker("spread gate: quote too wide")
    result = MasterWorkerExecutor(worker).enter("LONG", 22981.0, 23100.0, "first-candle low", price=23046.55)

    assert result["accepted"] is False
    assert "refusal_kind" not in result
    assert SIZING_REFUSAL_GUIDANCE not in result["reason"]


# --------------------------------------------------------------------------
# The guard: no tighter same-direction stop after a sizing refusal, this pass
# --------------------------------------------------------------------------

def test_a_squeezed_stop_after_a_sizing_refusal_is_refused_this_pass():
    ex = StandaloneExecutor()
    ctx = SLHuntingToolContext.build(_candles(), ex)

    first = ctx.do_order("ENTER_LONG", stop=WIDE_STOP, target=TARGET, reason="hammer at the first-candle low")
    squeezed = ctx.do_order("ENTER_LONG", stop=TIGHT_STOP, target=TARGET, reason="stop tightened to fit the budget")

    assert first["accepted"] is False and first["refusal_kind"] == "sizing"
    assert squeezed["accepted"] is False
    assert "SLH-025" in squeezed["reason"]
    assert "HOLD" in squeezed["reason"]
    assert ex.snapshot()["in_position"] is False
    assert ctx.state.value == "ACTIVE"


def test_the_guard_does_not_touch_the_opposite_direction():
    ex = StandaloneExecutor()
    ctx = SLHuntingToolContext.build(_candles(), ex)

    ctx.do_order("ENTER_LONG", stop=WIDE_STOP, target=TARGET, reason="hammer at the first-candle low")
    short = ctx.do_order("ENTER_SHORT", stop=25050, target=24950, reason="rejection at the high instead")

    assert short["accepted"] is True
    assert ex.snapshot()["direction"] == "SHORT"


def test_a_non_sizing_refusal_does_not_arm_the_guard():
    ex = StandaloneExecutor()
    ctx = SLHuntingToolContext.build(_candles(), ex)

    bad_levels = ctx.do_order("ENTER_LONG", stop=25040, target=TARGET, reason="stop on the wrong side")
    fixed = ctx.do_order("ENTER_LONG", stop=TIGHT_STOP, target=TARGET, reason="hammer at the pivot")

    assert bad_levels["accepted"] is False
    assert fixed["accepted"] is True


def test_the_guard_is_scoped_to_one_pass():
    ex = StandaloneExecutor()
    first_pass = SLHuntingToolContext.build(_candles(), ex)
    first_pass.do_order("ENTER_LONG", stop=WIDE_STOP, target=TARGET, reason="hammer at the first-candle low")

    next_pass = SLHuntingToolContext.build(_candles(), ex)
    fresh = next_pass.do_order("ENTER_LONG", stop=TIGHT_STOP, target=TARGET, reason="new higher low printed")

    assert fresh["accepted"] is True


def test_an_executor_refusal_for_another_reason_does_not_arm_the_guard():
    """Only a refusal TAGGED as sizing arms it: a cooldown or a quote problem says
    nothing about whether the stop fits, so a tighter retry stays the model's call."""

    class _CooldownThenAccept(StandaloneExecutor):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def enter(self, direction, stop, target, reason, price):
            self.calls += 1
            if self.calls == 1:
                return {"accepted": False, "reason": "post-exit cooldown: 30s still to run"}
            return super().enter(direction, stop, target, reason, price)

    ex = _CooldownThenAccept()
    ctx = SLHuntingToolContext.build(_candles(), ex)

    refused = ctx.do_order("ENTER_LONG", stop=WIDE_STOP, target=TARGET, reason="hammer at the first-candle low")
    retried = ctx.do_order("ENTER_LONG", stop=TIGHT_STOP, target=TARGET, reason="hammer at the pivot")

    assert refused["accepted"] is False and "refusal_kind" not in refused
    assert retried["accepted"] is True
