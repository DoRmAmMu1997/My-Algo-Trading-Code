"""MODIFY_TARGET: the order tool can RAISE an open position's target (operator, 21 Oct).

v5w tells the agent not to book a run that has not pulled back before its target,
and the operator asked that the agent be able to move that target further out
rather than only wait for it. The target is a MECHANICAL level the worker checks
against spot on every poll -- no broker order rests there -- so raising it is a
local state change. These tests pin the contract:

* raise-only: a LONG's target can only go up, a SHORT's only down -- never closer,
  never removed, and never on the wrong side of the current price;
* bounded like an entry (within ~10% of the current price);
* an executor or worker that cannot do it refuses, it never half-applies;
* it is one accepted side effect, like ENTER and EXIT;
* the decision the agent records afterwards is an honest HOLD.

No SDK, CLI or network: the agent loop is driven by an injected runner.
"""

from __future__ import annotations

import json
import math

import pandas as pd
from sl_hunting_agent import AgentRunResult, SLHuntingAgent
from sl_hunting_executor import MasterWorkerExecutor, StandaloneExecutor, target_raise_error
from sl_hunting_tools import SLHuntingToolContext, order_tool_description


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


# The synthetic series ends at 25029 -- the order tool's current price.
LAST = 25029.0


# --------------------------------------------------------------------------
# The shared validation helper
# --------------------------------------------------------------------------

def test_a_long_target_can_be_raised_beyond_the_current_price():
    assert target_raise_error("LONG", current_target=25080.0, new_target=25120.0, price=LAST) is None


def test_a_long_target_can_never_be_lowered_or_kept():
    lower = target_raise_error("LONG", current_target=25080.0, new_target=25060.0, price=LAST)
    same = target_raise_error("LONG", current_target=25080.0, new_target=25080.0, price=LAST)
    assert lower is not None and "only be RAISED" in lower
    assert same is not None and "only be RAISED" in same


def test_a_long_target_must_stay_above_the_current_price():
    # Raised relative to an old target that sat below price is still refused:
    # a target the market has already passed would exit on the next poll.
    problem = target_raise_error("LONG", current_target=25000.0, new_target=25020.0, price=LAST)
    assert problem is not None and "above the current price" in problem


def test_a_short_target_is_the_mirror_image():
    assert target_raise_error("SHORT", current_target=24980.0, new_target=24940.0, price=LAST) is None
    higher = target_raise_error("SHORT", current_target=24980.0, new_target=24990.0, price=LAST)
    wrong_side = target_raise_error("SHORT", current_target=25040.0, new_target=25035.0, price=LAST)
    assert higher is not None and "only be LOWERED" in higher
    assert wrong_side is not None and "below the current price" in wrong_side


def test_a_raise_stays_inside_the_entry_bound():
    far = target_raise_error("LONG", current_target=25080.0, new_target=LAST * 1.11, price=LAST)
    assert far is not None and "10%" in far


def test_a_raise_rejects_nonsense_numbers():
    for bad in (0.0, -5.0, math.nan, math.inf):
        assert target_raise_error("LONG", 25080.0, bad, LAST) is not None
    assert target_raise_error("SIDEWAYS", 25080.0, 25100.0, LAST) is not None


# --------------------------------------------------------------------------
# StandaloneExecutor (paper)
# --------------------------------------------------------------------------

def test_standalone_raise_moves_the_paper_target():
    ex = StandaloneExecutor()
    assert ex.enter("LONG", stop=25000, target=25080, reason="pivot bounce", price=LAST)["accepted"]

    result = ex.modify_target(25120.0, "run has not pulled back", price=LAST)

    assert result["accepted"] is True
    assert result["action"] == "MODIFY_TARGET"
    assert result["old_target"] == 25080.0 and result["target"] == 25120.0
    assert ex.snapshot()["target"] == 25120.0


def test_standalone_refuses_when_flat_or_when_lowering():
    ex = StandaloneExecutor()
    assert ex.modify_target(25120.0, "nothing open", price=LAST)["accepted"] is False

    ex.enter("LONG", stop=25000, target=25080, reason="pivot bounce", price=LAST)
    refused = ex.modify_target(25060.0, "closer target", price=LAST)

    assert refused["accepted"] is False
    assert ex.snapshot()["target"] == 25080.0


# --------------------------------------------------------------------------
# MasterWorkerExecutor (live-capable)
# --------------------------------------------------------------------------

class _Pos:
    def __init__(self, active=True, direction="LONG", target=25080.0):
        self.active = active
        self.direction = direction
        self.entry_underlying = 25010.0
        self.stop_underlying = 24990.0
        self.target_underlying = target


class _RaisingWorker:
    """A worker exposing the hook; it records what it was asked to do."""

    def __init__(self, pos):
        self.pos = pos
        self.calls = []

    def raise_target_underlying(self, new_target, reason):
        self.calls.append((new_target, reason))
        self.pos.target_underlying = new_target
        return True


class _OldWorker:
    """A worker that predates the hook."""

    def __init__(self, pos):
        self.pos = pos


def test_master_executor_delegates_a_valid_raise_to_the_worker():
    worker = _RaisingWorker(_Pos())
    result = MasterWorkerExecutor(worker).modify_target(25120.0, "run has not pulled back", price=LAST)

    assert result["accepted"] is True
    assert result["old_target"] == 25080.0 and result["target"] == 25120.0
    assert worker.calls == [(25120.0, "run has not pulled back")]


def test_master_executor_validates_before_touching_the_worker():
    worker = _RaisingWorker(_Pos())
    result = MasterWorkerExecutor(worker).modify_target(25060.0, "closer target", price=LAST)

    assert result["accepted"] is False
    assert worker.calls == []
    assert worker.pos.target_underlying == 25080.0


def test_master_executor_refuses_when_the_nifty_leg_is_flat():
    worker = _RaisingWorker(_Pos(active=False))
    result = MasterWorkerExecutor(worker).modify_target(25120.0, "nothing open", price=LAST)

    assert result["accepted"] is False
    assert worker.calls == []


def test_master_executor_fails_closed_on_a_worker_without_the_hook():
    worker = _OldWorker(_Pos())
    result = MasterWorkerExecutor(worker).modify_target(25120.0, "run has not pulled back", price=LAST)

    assert result["accepted"] is False
    assert worker.pos.target_underlying == 25080.0


# --------------------------------------------------------------------------
# The order tool
# --------------------------------------------------------------------------

def _context_with_open_long():
    ex = StandaloneExecutor()
    ex.enter("LONG", stop=25000, target=25080, reason="pivot bounce", price=LAST)
    return ex, SLHuntingToolContext.build(_candles(), ex)


def test_order_tool_raises_the_target_and_consumes_the_pass():
    ex, ctx = _context_with_open_long()

    result = ctx.do_order("MODIFY_TARGET", stop=0, target=25120, reason="run has not pulled back; v5w")

    assert result["accepted"] is True
    assert ex.snapshot()["target"] == 25120.0
    assert ctx.state.value == "CONSUMED"
    # One side effect per pass, exactly like ENTER and EXIT.
    assert ctx.do_order("EXIT", stop=0, target=0, reason="second action")["accepted"] is False


def test_order_tool_sends_a_bad_raise_back_for_correction():
    ex, ctx = _context_with_open_long()

    refused = ctx.do_order("MODIFY_TARGET", stop=0, target=25060, reason="closer target")

    assert refused["accepted"] is False
    assert ctx.state.value == "ACTIVE"
    assert ex.snapshot()["target"] == 25080.0


def test_order_tool_requires_a_real_reason_for_a_raise():
    ex, ctx = _context_with_open_long()

    refused = ctx.do_order("MODIFY_TARGET", stop=0, target=25120, reason="")

    assert refused["accepted"] is False
    assert "reason" in refused["reason"]
    assert ex.snapshot()["target"] == 25080.0


def test_order_tool_refuses_cleanly_on_an_executor_without_modify_target():
    class _EntryExitOnly:
        def enter(self, direction, stop, target, reason, price):
            return {"accepted": True}

        def exit(self, reason, price, leg="BOTH"):
            return {"accepted": True}

        def snapshot(self):
            return {"in_position": True, "direction": "LONG", "target": 25080.0}

    ctx = SLHuntingToolContext.build(_candles(), _EntryExitOnly())
    refused = ctx.do_order("MODIFY_TARGET", stop=0, target=25120, reason="run has not pulled back")

    assert refused["accepted"] is False
    assert ctx.state.value == "ACTIVE"


def test_order_tool_description_documents_the_raise_only_action():
    text = order_tool_description("PAPER")
    assert "MODIFY_TARGET" in text
    assert "only RAISES" in text
    assert "never lowers or removes" in text


# --------------------------------------------------------------------------
# The decision the agent records afterwards
# --------------------------------------------------------------------------

def _raising_runner(final_text):
    async def _runner(prompt, *, system_prompt, model, max_turns, tool_context=None):
        result = tool_context.do_order(
            "MODIFY_TARGET", stop=0, target=25120, reason="run has not pulled back; v5w"
        )
        assert result["accepted"] is True
        return AgentRunResult(text=final_text)
    return _runner


def test_a_raise_is_recorded_as_a_hold():
    ex = StandaloneExecutor()
    ex.enter("LONG", stop=25000, target=25080, reason="pivot bounce", price=LAST)
    final = json.dumps({"action": "HOLD", "confidence": 6, "setup": "target_raised",
                        "reasoning": "Run has not pulled back; target raised to 25120."})

    decision = SLHuntingAgent(model="m", runner=_raising_runner(final)).decide(_candles(), ex)

    assert decision.action == "HOLD"
    assert decision.setup == "target_raised"
    assert ex.snapshot()["target"] == 25120.0


def test_a_raise_with_an_unparseable_final_answer_is_still_an_honest_hold():
    ex = StandaloneExecutor()
    ex.enter("LONG", stop=25000, target=25080, reason="pivot bounce", price=LAST)

    decision = SLHuntingAgent(model="m", runner=_raising_runner("not json")).decide(_candles(), ex)

    assert decision.action == "HOLD"
    assert decision.setup == "tool_execution"
    assert "run has not pulled back" in decision.reasoning
    assert ex.snapshot()["target"] == 25120.0
