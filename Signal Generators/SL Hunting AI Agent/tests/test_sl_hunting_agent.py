"""Agent + executor + tool-context tests (no SDK / CLI / network).

The agentic loop is driven by an injected ``_FakeRunner`` so these tests never
spawn the Claude CLI — mirroring the Streamlit Scanner App's testing seam.
"""

from __future__ import annotations

import json

import pandas as pd
from sl_hunting_agent import AgentRunResult, SLHuntingAgent, SLHuntingDecision
from sl_hunting_executor import (
    MasterWorkerExecutor,
    StandaloneExecutor,
    execution_tool_name,
)
from sl_hunting_tools import SLHuntingToolContext


def _candles(n=30, start_price=25000.0):
    ts = pd.date_range(start="2026-06-26 09:15", periods=n, freq="5min")
    closes = [start_price + i for i in range(n)]
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 2, min(prev, c) - 2, c))
        prev = c
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100] * n,
        }
    )


class _FakeRunner:
    """Stand-in for the Claude Agent SDK runner; returns canned final text."""

    def __init__(self, text: str, *, cost_usd: float | None = 0.01):
        self.text = text
        self.cost_usd = cost_usd
        self.calls = 0
        self.last_model = None

    async def __call__(self, prompt, *, system_prompt, model, max_turns, tool_context=None):
        self.calls += 1
        self.last_model = model
        return AgentRunResult(text=self.text, cost_usd=self.cost_usd)


# --------------------------------------------------------------------------
# execution_tool_name — the env "picks one of three tools" mapping
# --------------------------------------------------------------------------

def test_execution_tool_name_mapping():
    assert execution_tool_name(False, None) == "place_paper_order"
    assert execution_tool_name(False, "KOTAK") == "place_paper_order"
    assert execution_tool_name(True, "KOTAK") == "place_kotak_order"
    assert execution_tool_name(True, "shoonya") == "place_shoonya_order"
    # Unknown broker fails closed to paper.
    assert execution_tool_name(True, "ZERODHA") == "place_paper_order"


# --------------------------------------------------------------------------
# StandaloneExecutor guards + P&L proxy
# --------------------------------------------------------------------------

def test_standalone_executor_enter_exit_and_guards():
    ex = StandaloneExecutor(lots=1, lot_size=75)
    assert ex.exit("flat", price=25000)["accepted"] is False  # cannot exit when flat

    r = ex.enter("LONG", stop=24950, target=25100, reason="test", price=25000)
    assert r["accepted"] is True
    assert ex.snapshot()["in_position"] is True

    # cannot enter again while in a position
    assert ex.enter("SHORT", 25050, 24900, "again", 25000)["accepted"] is False

    out = ex.exit("target", price=25075)
    assert out["accepted"] is True
    assert out["points"] == 75.0
    assert out["pnl_proxy"] == 75.0 * 75
    assert ex.snapshot()["in_position"] is False
    assert len(ex.closed_trades) == 1


def test_standalone_executor_rejects_bad_direction():
    ex = StandaloneExecutor()
    assert ex.enter("SIDEWAYS", 1, 2, "x", 100)["accepted"] is False


# --------------------------------------------------------------------------
# Tool context routes orders through the executor
# --------------------------------------------------------------------------

def test_tool_context_do_order_routes_to_executor():
    ex = StandaloneExecutor()
    ctx = SLHuntingToolContext.build(_candles(), ex, live_active=False, broker=None)
    assert ctx.action_tool_name() == "place_paper_order"

    res = ctx.do_order("ENTER_LONG", stop=24990, target=25080, reason="pivot bounce")
    assert res["accepted"] is True
    assert ex.snapshot()["in_position"] is True
    assert ex.pos.direction == "LONG"

    # context tool payloads are JSON-serialisable dicts
    assert ctx.pivot_and_levels_payload()["available"] is True
    assert ctx.position_state_payload()["in_position"] is True


# --------------------------------------------------------------------------
# Agent.decide with the fake runner
# --------------------------------------------------------------------------

def test_agent_decide_parses_canned_json_and_stamps_model():
    canned = json.dumps(
        {
            "action": "HOLD",
            "stop": 0,
            "target": 0,
            "confidence": 3,
            "setup": "none",
            "reasoning": "No confirmed setup at a level; waiting.",
            "model_used": "ignored-will-be-overwritten",
        }
    )
    agent = SLHuntingAgent(model="test-model", runner=_FakeRunner(canned))
    decision = agent.decide(_candles(), StandaloneExecutor())
    assert isinstance(decision, SLHuntingDecision)
    assert decision.action == "HOLD"
    assert decision.confidence == 3
    assert decision.model_used == "test-model"  # stamped by the agent


def test_agent_decide_holds_on_malformed_output():
    agent = SLHuntingAgent(model="test-model", runner=_FakeRunner("this is not json"))
    decision = agent.decide(_candles(), StandaloneExecutor())
    assert decision.action == "HOLD"
    assert decision.setup == "agent_error"


def test_agent_decide_on_empty_candles_holds():
    agent = SLHuntingAgent(model="test-model", runner=_FakeRunner("{}"))
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    decision = agent.decide(empty, StandaloneExecutor())
    assert decision.action == "HOLD"
    assert decision.setup == "no_data"


# --------------------------------------------------------------------------
# _raise_for_error_result — classify the CLI's failed-API-call result by HTTP status
# (the "is_error=True, subtype=success" case that otherwise surfaces opaquely as
#  "Claude Code returned an error result: success").
# --------------------------------------------------------------------------

class _FakeResult:
    """Minimal stand-in for the SDK ResultMessage (is_error + api_error_status)."""

    def __init__(self, is_error: bool, api_error_status=None):
        self.is_error = is_error
        self.api_error_status = api_error_status


def test_raise_for_error_result_classifies_api_status():
    import pytest
    from sl_hunting_agent import (
        SLHuntingAgentError,
        SLHuntingAuthError,
        SLHuntingUsageLimitError,
        _raise_for_error_result,
    )

    # 401 / 403 -> auth error, with an actionable, content-free message.
    with pytest.raises(SLHuntingAuthError) as ei:
        _raise_for_error_result(_FakeResult(True, 401))
    assert "401" in str(ei.value) and "claude" in str(ei.value).lower()
    with pytest.raises(SLHuntingAuthError):
        _raise_for_error_result(_FakeResult(True, 403))

    # 429 -> usage limit
    with pytest.raises(SLHuntingUsageLimitError):
        _raise_for_error_result(_FakeResult(True, 429))

    # Other HTTP errors -> generic agent error (NOT the auth/usage subclasses).
    with pytest.raises(SLHuntingAgentError) as ei2:
        _raise_for_error_result(_FakeResult(True, 500))
    assert not isinstance(ei2.value, (SLHuntingAuthError, SLHuntingUsageLimitError))
    assert "500" in str(ei2.value)

    # Not an error result, or no result at all -> no raise (returns None).
    assert _raise_for_error_result(_FakeResult(False, None)) is None
    assert _raise_for_error_result(None) is None


def test_agent_decide_logs_actionable_auth_error(caplog):
    """A 401 from the SDK path makes decide() HOLD AND log the fix, not an opaque trace."""
    import logging

    from sl_hunting_agent import SLHuntingAuthError

    class _AuthFailRunner:
        async def __call__(self, prompt, *, system_prompt, model, max_turns, tool_context=None):
            raise SLHuntingAuthError(401)

    agent = SLHuntingAgent(model="test-model", runner=_AuthFailRunner())
    with caplog.at_level(logging.WARNING):
        decision = agent.decide(_candles(), StandaloneExecutor())
    assert decision.action == "HOLD" and decision.setup == "agent_error"  # fail-soft held
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "401" in logged and "claude" in logged.lower()  # actionable fix in the log


# --------------------------------------------------------------------------
# Fast mode ("disable extended thinking") must build a VALID thinking config.
# Regression: ThinkingConfigDisabled is a TypedDict, so a bare ThinkingConfigDisabled()
# is {} (no "type" key). Passing that to the SDK makes _build_command raise
# `KeyError: 'type'` on `thinking["type"]` -- i.e. fast mode never worked.
# --------------------------------------------------------------------------

def test_disabled_thinking_config_carries_type_key():
    import pytest

    sdk = pytest.importorskip("claude_agent_sdk")
    from sl_hunting_agent import _disabled_thinking_config

    cfg = _disabled_thinking_config(sdk)
    assert cfg is not None
    # The SDK does `thinking["type"]`; a config without it KeyErrors in _build_command.
    assert cfg["type"] == "disabled"

    # Guard against regressing to the bare call: ThinkingConfigDisabled() is the {} trap.
    assert dict(sdk.ThinkingConfigDisabled()) == {}


def test_disabled_thinking_config_none_when_sdk_lacks_it():
    from sl_hunting_agent import _disabled_thinking_config

    class _OldSdk:  # too old to expose ThinkingConfigDisabled -> caller warns, no thinking
        pass

    assert _disabled_thinking_config(_OldSdk()) is None


# --------------------------------------------------------------------------
# MasterWorkerExecutor duck-types the worker
# --------------------------------------------------------------------------

class _FakePos:
    def __init__(self):
        self.active = False
        self.direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0


class _FakeWorker:
    def __init__(self):
        self.pos = _FakePos()
        self.entries = []

    def enter_position(self, direction, entry_underlying, stop_underlying=0.0, target_underlying=0.0):
        self.entries.append((direction, entry_underlying, stop_underlying, target_underlying))
        self.pos.active = True
        self.pos.direction = direction
        self.pos.entry_underlying = entry_underlying
        self.pos.stop_underlying = stop_underlying
        self.pos.target_underlying = target_underlying
        return True

    def exit_position(self, reason):
        self.pos = _FakePos()

    def _get_open_position_pnl(self):
        return 123.0


def test_master_worker_executor_delegates():
    w = _FakeWorker()
    ex = MasterWorkerExecutor(w)
    assert ex.snapshot() == {"in_position": False}

    r = ex.enter("LONG", stop=24950, target=25100, reason="x", price=25000)
    assert r["accepted"] is True
    assert w.entries == [("LONG", 25000.0, 24950.0, 25100.0)]
    snap = ex.snapshot()
    assert snap["in_position"] is True and snap["unrealized_pnl"] == 123.0

    # cannot enter again while in a position
    assert ex.enter("SHORT", 1, 2, "y", 25000)["accepted"] is False
    assert ex.exit("done", price=25050)["accepted"] is True
    assert ex.snapshot()["in_position"] is False
