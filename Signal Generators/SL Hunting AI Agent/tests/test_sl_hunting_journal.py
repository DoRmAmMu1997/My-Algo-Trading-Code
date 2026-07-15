"""Phase 1 tests: the trade journal + entry-context snapshot."""

from __future__ import annotations

import json

import pandas as pd
from sl_hunting_journal import (
    TradeJournal,
    build_entry_context,
    followed_method,
    make_entry_record,
    resolve_entry_narrative,
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


def test_timeout_entry_journals_order_reason_not_placeholder(tmp_path):
    """A real ENTER whose SDK call then times out AFTER firing must journal the order
    tool's real reason and levels -- not the fail-soft 'Agent call timed out; holding.'
    placeholder (2026-07-15 journal-fidelity fix; see journal row 20)."""
    from types import SimpleNamespace

    # The agent returned the fail-soft placeholder (the call timed out AFTER the order
    # tool had already placed the SHORT this bar).
    placeholder = SimpleNamespace(
        action="HOLD", confidence=0, setup="agent_error",
        reasoning="Agent call timed out; holding.", model_used="m")
    # ...but the executor captured what the order tool actually sent this bar.
    order_payload = {
        "action": "ENTER_SHORT",
        "reason": "Gap-up-and-go rejected the psych level; hunting trapped buyers.",
        "stop": 24230, "target": 24120,
    }
    setup, confidence, reasoning = resolve_entry_narrative(placeholder, order_payload)
    assert reasoning == order_payload["reason"]
    assert "Agent call timed out" not in reasoning
    assert setup != "agent_error"          # a distinct sentinel, not the no-op-hold one
    assert confidence == 0                  # never returned by the model; stays 0

    # End-to-end through the journal (rows persist on close): the finished row carries
    # the real rationale/levels, exactly as the master worker's _journal_open_row wires
    # resolve_entry_narrative into make_entry_record.
    df = _two_day()
    j = TradeJournal(str(tmp_path / "j.jsonl"))
    entry = make_entry_record(
        direction="SHORT", setup=setup, confidence=confidence, reasoning=reasoning,
        stop=order_payload["stop"], target=order_payload["target"],
        entry_underlying=24195, lots=2, nifty_df=df, bnf_df=None,
    )
    tid = j.open_trade(entry)
    j.close_trade(tid, {"exit_underlying": 24187, "exit_reason": "AI_EXIT", "points": 8})
    lines = (tmp_path / "j.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["trade_id"] == tid
    assert parsed["reasoning"] == order_payload["reason"]
    assert "Agent call timed out" not in parsed["reasoning"]
    assert parsed["setup"] != "agent_error"
    assert parsed["stop"] == 24230.0 and parsed["target"] == 24120.0


def test_resolve_entry_narrative_leaves_genuine_decisions_untouched():
    """The fallback fires ONLY for the placeholder-with-order case: a real parsed
    decision is returned unchanged even if an order payload is present, and a genuine
    fail-soft hold (no order this bar) keeps its placeholder text."""
    from types import SimpleNamespace

    real = SimpleNamespace(setup="pivot_support_hammer", confidence=7,
                           reasoning="Hammer at pivot support with BNF confirmation.")
    payload = {"action": "ENTER_LONG", "reason": "different tool reason", "stop": 1, "target": 2}
    assert resolve_entry_narrative(real, payload) == (
        "pivot_support_hammer", 7, "Hammer at pivot support with BNF confirmation.")

    # Placeholder but NO order fired this bar (a true hold) -> left exactly as-is.
    hold = SimpleNamespace(setup="agent_error", confidence=0,
                           reasoning="Agent call timed out; holding.")
    assert resolve_entry_narrative(hold, None) == (
        "agent_error", 0, "Agent call timed out; holding.")


def test_decision_log_records_holds_and_actions(tmp_path):
    """The separate decision log captures EVERY decision -- including HOLDs, which the
    trade journal never records (no trade)."""
    from types import SimpleNamespace

    from sl_hunting_journal import append_decision, make_decision_record

    df = _two_day()
    hold = SimpleNamespace(action="HOLD", confidence=2, setup="none", stop=0, target=0,
                           reasoning="No confirmed setup at a level; waiting.", model_used="m")
    rec = make_decision_record(hold, df, None)
    assert rec["action"] == "HOLD" and rec["confidence"] == 2 and rec["setup"] == "none"
    assert rec["reasoning"].startswith("No confirmed setup")
    assert "decided_at" in rec and "pivot" in rec["context"]  # context snapshot attached

    path = str(tmp_path / "decisions.jsonl")
    append_decision(path, rec)
    enter = SimpleNamespace(action="ENTER_LONG", confidence=7, setup="pivot_support",
                            stop=24985, target=25060, reasoning="hammer at pivot", model_used="m")
    append_decision(path, make_decision_record(enter, df, None))

    lines = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "HOLD"               # the HOLD is captured
    second = json.loads(lines[1])
    assert second["action"] == "ENTER_LONG" and second["stop"] == 24985.0
