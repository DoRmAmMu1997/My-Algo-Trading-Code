"""Phase 1 tests: the trade journal + entry-context snapshot."""

from __future__ import annotations

import json

import pandas as pd

from sl_hunting_journal import (
    TradeJournal,
    build_entry_context,
    followed_method,
    make_entry_record,
)


def _two_day():
    """Minimal 2-day 5-min frame so pivot/levels resolve."""
    rows = [
        (24900, 25100, 24900, 25100), (25100, 25100, 24900, 25000),  # prev day
        (25000, 25010, 24990, 25000), (25000, 25020, 24995, 25010),  # today
        (25010, 25030, 25005, 25020), (25020, 25025, 25000, 25005),
    ]
    ts = list(pd.date_range("2026-06-25 09:15", periods=2, freq="5min")) + \
        list(pd.date_range("2026-06-26 09:15", periods=4, freq="5min"))
    return pd.DataFrame({
        "timestamp": ts,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [100] * len(rows),
    })


def test_journal_open_close_writes_one_complete_row(tmp_path):
    j = TradeJournal(str(tmp_path / "j.jsonl"))
    entry = {
        "direction": "LONG", "setup": "pivot_support_hammer", "confidence": 7,
        "stop": 24985, "target": 25060, "reasoning": "x", "entry_underlying": 25000,
        "lots": 3, "context": {}, "followed_method": True,
    }
    tid = j.open_trade(entry)
    assert j.open_count == 1
    row = j.close_trade(tid, {
        "exit_underlying": 25030, "exit_reason": "target_hit", "points": 30,
        "option_pnl": 4500, "lots": 3,
    })
    assert j.open_count == 0
    # R = points / |entry - stop| = 30 / 15 = 2.0
    assert row["outcome"]["r_multiple"] == 2.0
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["direction"] == "LONG"
    assert parsed["trade_id"] == tid
    assert parsed["outcome"]["exit_reason"] == "target_hit"


def test_close_unknown_id_is_noop(tmp_path):
    j = TradeJournal(str(tmp_path / "j.jsonl"))
    assert j.close_trade("does-not-exist", {"points": 1}) is None
    assert not (tmp_path / "j.jsonl").exists() or (tmp_path / "j.jsonl").read_text(encoding="utf-8") == ""


def test_followed_method_matches_direction():
    ctx_bull = {"confirmed_patterns": [{"type": "bullish_engulfing", "direction": "bullish", "bars_ago": 1}]}
    assert followed_method(ctx_bull, "LONG") is True
    assert followed_method(ctx_bull, "SHORT") is False
    assert followed_method({"confirmed_patterns": []}, "LONG") is False


def test_build_entry_context_and_make_record():
    df = _two_day()
    ctx = build_entry_context(df, None)
    assert "pivot" in ctx and "cross_index" in ctx and "confirmed_patterns" in ctx
    assert ctx["cross_index"] == {"available": False}  # no BNF passed

    rec = make_entry_record(
        direction="LONG", setup="s", confidence=6, stop=24985, target=25060,
        reasoning="r", entry_underlying=25000, lots=2, nifty_df=df, bnf_df=None,
    )
    assert rec["direction"] == "LONG" and rec["lots"] == 2
    assert "context" in rec and "followed_method" in rec
    assert rec["stop"] == 24985.0 and rec["target"] == 25060.0
