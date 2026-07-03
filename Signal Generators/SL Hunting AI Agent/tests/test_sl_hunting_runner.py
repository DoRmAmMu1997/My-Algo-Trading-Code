"""Tests for the standalone runner's non-trivial helpers."""

from __future__ import annotations

import pandas as pd
from sl_hunting_executor import StandaloneExecutor
from sl_hunting_runner import _manage_open_position, resample_1m_to_n


def _one_min(n=20, start_price=25000.0):
    ts = pd.date_range(start="2026-06-26 09:15", periods=n, freq="1min")
    closes = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [10] * n,
        }
    )


def test_resample_1m_to_5m_aggregates_ohlc():
    df = _one_min(n=20)
    out = resample_1m_to_n(df, 5)
    # 20 one-minute bars → 4 five-minute bars.
    assert len(out) == 4
    # First 5-min bar's high should be the max of its constituent 1-min highs.
    assert out.iloc[0]["high"] == df.iloc[:5]["high"].max()
    assert {"timestamp", "open", "high", "low", "close"}.issubset(out.columns)


def test_resample_drops_incomplete_tail_bucket():
    # 12 one-minute bars -> two COMPLETE 5-min bars (09:15, 09:20); the 09:25 bucket
    # has only 2 of 5 rows and must be dropped (matches the master's resampler).
    out = resample_1m_to_n(_one_min(n=12), 5)
    assert len(out) == 2


def test_manage_open_position_fills_long_target_and_stop():
    ex = StandaloneExecutor()
    ex.enter("LONG", stop=24950, target=25100, reason="t", price=25000)
    # A bar whose high reaches the target fills the target.
    bar = pd.Series({"high": 25120.0, "low": 25010.0})
    _manage_open_position(ex, bar)
    assert ex.snapshot()["in_position"] is False
    assert ex.closed_trades[-1]["exit_reason"] == "target_hit"

    ex2 = StandaloneExecutor()
    ex2.enter("LONG", stop=24950, target=25100, reason="t", price=25000)
    bar2 = pd.Series({"high": 25010.0, "low": 24940.0})  # low breaches stop
    _manage_open_position(ex2, bar2)
    assert ex2.snapshot()["in_position"] is False
    assert ex2.closed_trades[-1]["exit_reason"] == "stop_hit"


def test_runner_journal_helpers_open_and_close(tmp_path):
    import json

    from sl_hunting_agent import SLHuntingDecision
    from sl_hunting_executor import StandaloneExecutor
    from sl_hunting_journal import TradeJournal
    from sl_hunting_runner import _close_journal_row, _open_journal_row

    ex = StandaloneExecutor(lot_size=75, risk_budget=2500)
    ex.enter("LONG", stop=24985, target=25060, reason="t", price=25000)  # 15-pt stop -> 3 lots
    j = TradeJournal(str(tmp_path / "j.jsonl"))
    dec = SLHuntingDecision(action="ENTER_LONG", stop=24985, target=25060, confidence=7,
                            setup="pivot_support", reasoning="r")
    oid = _open_journal_row(j, dec, _one_min(n=12), None, ex)
    assert j.open_count == 1

    ex.exit("target_hit", price=25030)
    _close_journal_row(j, oid, ex)
    assert j.open_count == 0
    row = json.loads((tmp_path / "j.jsonl").read_text(encoding="utf-8").strip())
    assert row["direction"] == "LONG" and row["lots"] == 3
    assert row["outcome"]["exit_reason"] == "target_hit"


def test_open_journal_row_uses_executor_state_not_decision(tmp_path):
    """Regression (Codex P2): journal the EXECUTED position, not a mismatched decision.

    The order tool opened a LONG during decide(), but the model's final JSON came back
    as a safe HOLD (malformed / disagreeing). The row must reflect the executor's
    actual position (LONG + its stop/target), not the decision. The old code derived
    direction from `decision.action` (a HOLD fell into the `else` and was mislabelled
    SHORT) and stored decision.stop/target (0/0), corrupting the row.
    """
    import json

    from sl_hunting_agent import SLHuntingDecision
    from sl_hunting_journal import TradeJournal
    from sl_hunting_runner import _close_journal_row, _open_journal_row

    ex = StandaloneExecutor(lot_size=75, risk_budget=2500)
    ex.enter("LONG", stop=24985, target=25060, reason="t", price=25000)
    j = TradeJournal(str(tmp_path / "j2.jsonl"))
    # Decision DISAGREES with the executed trade (what decide() returns on bad JSON).
    dec = SLHuntingDecision(action="HOLD", stop=0, target=0, confidence=0,
                            setup="agent_error", reasoning="holding")
    oid = _open_journal_row(j, dec, _one_min(n=12), None, ex)
    # The row is flushed to disk on close; the entry half (direction/stop/target) is
    # what the fix is about, and it is unchanged by the exit.
    ex.exit("manual_close", price=25030)
    _close_journal_row(j, oid, ex)
    row = json.loads((tmp_path / "j2.jsonl").read_text(encoding="utf-8").strip())
    assert row["direction"] == "LONG"    # from ex.pos (old code would have said SHORT)
    assert row["stop"] == 24985          # ex.pos.stop (old code used decision.stop = 0)
    assert row["target"] == 25060        # ex.pos.target (old code used decision.target = 0)
    assert row["setup"] == "agent_error"  # rationale still comes from the decision


def test_manage_open_position_fills_short():
    ex = StandaloneExecutor()
    ex.enter("SHORT", stop=25050, target=24900, reason="t", price=25000)
    bar = pd.Series({"high": 25010.0, "low": 24880.0})  # low reaches target
    _manage_open_position(ex, bar)
    assert ex.closed_trades[-1]["exit_reason"] == "target_hit"
