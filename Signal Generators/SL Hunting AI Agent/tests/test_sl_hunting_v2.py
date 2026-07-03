"""v2 tests: BankNIFTY cross-confirmation + dynamic (Rs.2500) position sizing."""

from __future__ import annotations

import json

import pandas as pd
from sl_hunting_agent import AgentRunResult, SLHuntingAgent
from sl_hunting_executor import StandaloneExecutor, risk_based_lots, stop_or_target_hit
from sl_hunting_indicators import cross_index_signal
from sl_hunting_tools import SLHuntingToolContext


def _two_day(prev_high, prev_low, prev_close, today_closes):
    """Build a 2-day 5-min OHLC frame (prev day sets H/L/C; today is a close path)."""
    prev_rows = [
        (prev_low, prev_high, prev_low, prev_high),
        (prev_high, prev_high, prev_low, prev_close),
    ]
    today_rows = []
    p = prev_close
    for c in today_closes:
        today_rows.append((p, max(p, c) + 1, min(p, c) - 1, c))
        p = c
    rows = prev_rows + today_rows
    ts = list(pd.date_range("2026-06-25 09:15", periods=len(prev_rows), freq="5min")) + \
        list(pd.date_range("2026-06-26 09:15", periods=len(today_rows), freq="5min"))
    return pd.DataFrame({
        "timestamp": ts,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * len(rows),
    })


# Both indices sitting on support (last just above prev-day low; pivot above).
_NIFTY_AT_SUPPORT = _two_day(25100, 24900, 25000, [24960, 24950, 24940, 24930, 24920, 24910, 24905, 24902])
_BNF_AT_SUPPORT = _two_day(55100, 54900, 55000, [54960, 54950, 54940, 54930, 54920, 54910, 54905, 54902])
# Both indices broke down through pivot (rose above pivot, then closed clearly below
# it — BankNIFTY needs a larger drop since its 0.15% "at-level" band is ~2x NIFTY's).
_NIFTY_BREAKDOWN = _two_day(25100, 24900, 25000, [25010, 25030, 25020, 24990, 24970, 24950, 24940])
_BNF_BREAKDOWN = _two_day(55100, 54900, 55000, [55010, 55030, 55020, 54980, 54940, 54900, 54870])


# --------------------------------------------------------------------------
# cross_index_signal
# --------------------------------------------------------------------------

def test_cross_index_unavailable_without_bnf():
    out = cross_index_signal(_NIFTY_AT_SUPPORT, None)
    assert out["available"] is False
    assert "NIFTY alone" in out["reason"]


def test_cross_index_both_at_support_biases_down():
    out = cross_index_signal(_NIFTY_AT_SUPPORT, _BNF_AT_SUPPORT)
    assert out["available"] is True
    assert out["nifty"]["at_support"] is True and out["bank_nifty"]["at_support"] is True
    assert out["alignment"] == "both_at_support"
    assert out["bias"] == "down"


def test_cross_index_both_breakdown_biases_up():
    out = cross_index_signal(_NIFTY_BREAKDOWN, _BNF_BREAKDOWN)
    assert out["available"] is True
    assert out["nifty"]["broke_pivot_down"] is True and out["bank_nifty"]["broke_pivot_down"] is True
    assert out["alignment"] == "both_breakdown"
    assert out["bias"] == "up"


# --------------------------------------------------------------------------
# bank_nifty / cross_index tool payloads via the context
# --------------------------------------------------------------------------

def test_tool_payloads_available_with_bnf_and_unavailable_without():
    with_bnf = SLHuntingToolContext.build(_NIFTY_AT_SUPPORT, StandaloneExecutor(), bnf_candles=_BNF_AT_SUPPORT)
    assert with_bnf.bank_nifty_payload()["available"] is True
    assert with_bnf.cross_index_payload()["available"] is True

    no_bnf = SLHuntingToolContext.build(_NIFTY_AT_SUPPORT, StandaloneExecutor(), bnf_candles=None)
    assert no_bnf.bank_nifty_payload()["available"] is False
    assert no_bnf.cross_index_payload()["available"] is False


# --------------------------------------------------------------------------
# decide() passes bnf_candles through (fake runner)
# --------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self, text):
        self.text = text

    async def __call__(self, prompt, *, system_prompt, model, max_turns, tool_context=None):
        return AgentRunResult(text=self.text, cost_usd=0.0)


def test_decide_accepts_bnf_candles():
    canned = json.dumps({"action": "HOLD", "stop": 0, "target": 0, "confidence": 2,
                         "setup": "none", "reasoning": "waiting", "model_used": "x"})
    agent = SLHuntingAgent(model="test-model", runner=_FakeRunner(canned))
    decision = agent.decide(_NIFTY_AT_SUPPORT, StandaloneExecutor(), bnf_candles=_BNF_AT_SUPPORT)
    assert decision.action == "HOLD"
    assert decision.model_used == "test-model"


# --------------------------------------------------------------------------
# Dynamic position sizing (~Rs.2500 risk per trade)
# --------------------------------------------------------------------------

def test_risk_based_lots_targets_budget():
    # 15-pt stop, lot 75 -> risk/lot 1125; ceil(2500/1125)=3.
    assert risk_based_lots(25000, 24985, 75, 2500) == 3
    # 40-pt stop -> risk/lot 3000; ceil(2500/3000)=1 (never zero).
    assert risk_based_lots(25000, 24960, 75, 2500) == 1
    # 10-pt stop -> risk/lot 750; ceil(2500/750)=4.
    assert risk_based_lots(25000, 24990, 75, 2500) == 4


def test_risk_based_lots_falls_back_on_zero_distance():
    assert risk_based_lots(25000, 25000, 75, 2500, fallback_lots=2) == 2


def test_stop_or_target_hit():
    # LONG: spot at/below stop -> AI_STOP; at/above target -> AI_TARGET.
    assert stop_or_target_hit("LONG", 24985, 25080, 24980) == "AI_STOP"
    assert stop_or_target_hit("LONG", 24985, 25080, 25090) == "AI_TARGET"
    assert stop_or_target_hit("LONG", 24985, 25080, 25000) is None
    # SHORT: spot at/above stop -> AI_STOP; at/below target -> AI_TARGET.
    assert stop_or_target_hit("SHORT", 25050, 24900, 25060) == "AI_STOP"
    assert stop_or_target_hit("SHORT", 25050, 24900, 24890) == "AI_TARGET"
    assert stop_or_target_hit("SHORT", 25050, 24900, 25000) is None
    # Zero levels mean "not set" -> never trigger.
    assert stop_or_target_hit("LONG", 0, 0, 1) is None


def test_standalone_executor_sizes_dynamically():
    ex = StandaloneExecutor(lot_size=75, risk_budget=2500)
    r = ex.enter("LONG", stop=24985, target=25060, reason="t", price=25000)
    assert r["accepted"] is True
    assert r["lots"] == 3 and r["quantity"] == 3 * 75
    out = ex.exit("target", price=25020)  # +20 pts
    assert out["lots"] == 3
    assert out["pnl_proxy"] == 20 * 3 * 75
