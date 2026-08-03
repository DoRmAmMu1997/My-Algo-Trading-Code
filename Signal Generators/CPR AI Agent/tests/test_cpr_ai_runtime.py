"""Specify the isolated four-tool Codex runtime and host safety policy.

Every response in this module is a local fake.  These tests deliberately
exercise the real host policy, never an SDK or broker, because a model proposal
is untrusted input in a live-money system.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cpr_ai_codex_runner as codex_runner
import cpr_ai_codex_subprocess as codex_child
import pytest
from cpr_ai_agent import CPRAgent, CPRAgentRunResult, CPRHostPolicy, CPRToolCallRecord
from cpr_ai_codex_runner import build_codex_thread_config, safe_subprocess_environment
from cpr_ai_decision_log import CPRDecisionLogger
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_runner import _fake_runner as smoke_fake_runner
from cpr_ai_runner import main as runner_main
from cpr_ai_schema import CPRAgentDecision
from cpr_ai_tools import EXPECTED_TOOL_NAMES


def _context(*, is_flat: bool = True, direction: str | None = None) -> dict[str, dict[str, object]]:
    """Return a hand-authored frozen bar with every required host fact.

    The values make a long continuation valid: close 100, stop 95, the next
    buffered R1 milestone at 108, and a final R2 target at 118.
    """

    return {
        "session_levels": {
            "current_close": 100.0,
            "levels": {"r1": 110.0, "r2": 120.0, "s1": 90.0, "s2": 80.0},
            "next_levels": {
                "buffer_points": 2.0,
                "upside": {"name": "r1", "price": 110.0},
                "downside": {"name": "s1", "price": 90.0},
            },
        },
        "momentum_vwap": {
            "rsi14": 55.0,
            "stochastic_rsi": {
                "cross_up_in_oversold": True,
                "cross_down_in_overbought": True,
            },
            "vwap": {
                "sequence_evidence": {
                    "all_recent_above": True,
                    "all_recent_below": False,
                    "reclaimed": True,
                    "lost": False,
                },
                "entry_candle": {"body_fraction_above": 0.6, "body_fraction_below": 0.0},
            },
            "ema": {"order": "EMA5_ABOVE_EMA20", "ema5_slope": 1.0, "ema20_slope": 0.5},
            "candle": {"low": 95.0, "high": 105.0, "close": 100.0},
        },
        "market_structure": {
            "swings": {"lows": [{"price": 94.0}], "highs": [{"price": 106.0}]},
            "r1_scale_in_candidate": {"eligible": True, "direction": "LONG"},
        },
        "position_state": {
            "is_flat": is_flat,
            "direction": direction,
            "premise": "TRENDING_VWAP_CONTINUATION",
            "scale_in_eligible": True,
            "scale_in_count": 0,
        },
    }


def _proposal(action: str, regime: str, setup: str) -> CPRAgentDecision:
    """Build a schema-valid model classification with no execution data."""

    return CPRAgentDecision(
        action=action,
        regime=regime,
        setup=setup,
        confidence=8,
        reasoning="Synthetic test proposal.",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )


def _calls(*, missing: str | None = None, failed: str | None = None, unexpected: bool = False):
    """Create complete or intentionally defective MCP-call evidence."""

    records = [
        CPRToolCallRecord(tool=name, status="failed" if name == failed else "completed")
        for name in EXPECTED_TOOL_NAMES
        if name != missing
    ]
    if unexpected:
        records.append(CPRToolCallRecord(tool="shell", status="completed"))
    return tuple(records)


def _runner(proposal: CPRAgentDecision, *, calls=None, delay: float = 0.0):
    """Return an injected, deterministic SDK stand-in."""

    def run(**_kwargs):
        if delay:
            time.sleep(delay)
        return CPRAgentRunResult(
            final_response=proposal.model_dump_json(),
            tool_calls=tuple(calls if calls is not None else _calls()),
            token_usage={"total_tokens": 17},
        )

    return run


@pytest.mark.parametrize(
    ("missing", "failed", "unexpected", "code"),
    [
        ("market_structure", None, False, "missing_tool_call"),
        (None, "position_state", False, "failed_tool_call"),
        (None, None, True, "unexpected_agent_action"),
    ],
)
def test_agent_rejects_incomplete_or_unexpected_four_tool_evidence(missing, failed, unexpected, code):
    """The host accepts exactly four successful read-only MCP calls, nothing else."""

    agent = CPRAgent(
        runner=_runner(
            _proposal("HOLD", "UNDECIDED", "NONE"), calls=_calls(missing=missing, failed=failed, unexpected=unexpected)
        )
    )

    outcome = agent.decide(_context(), bar_signature="bar-1")

    assert outcome.accepted is False
    assert outcome.validation_code == code
    assert outcome.action == "HOLD"


def test_runtime_configuration_is_read_only_and_sanitizes_credentials(tmp_path):
    """A child gets no execution surface or trading/API credentials."""

    safe = safe_subprocess_environment(
        {
            "PATH": "safe",
            "SYSTEMROOT": "windows",
            "DHAN_ACCESS_TOKEN": "secret",
            "OPENAI_API_KEY": "key",
            "LIVE_TRADING_ENABLED": "true",
        }
    )
    config = build_codex_thread_config(str(tmp_path / "snapshot.json"), "python.exe", str(tmp_path))

    assert safe == {"PATH": "safe", "SYSTEMROOT": "windows"}
    assert config["features"]["shell_tool"] is False
    assert config["features"]["unified_exec"] is False
    assert config["features"]["multi_agent"] is False
    assert config["web_search"] == "disabled"
    assert config["mcp_servers"]["cpr_ai"]["enabled_tools"] == list(EXPECTED_TOOL_NAMES)


def test_real_runner_uses_a_sanitized_subprocess_and_parses_only_structured_evidence(monkeypatch):
    """The host must not import an SDK or give a child ambient credentials directly."""

    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "final_response": _proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
                    "tool_calls": [{"tool": name, "status": "completed"} for name in EXPECTED_TOOL_NAMES],
                    "token_usage": {"total_tokens": 2},
                    "unexpected_actions": [],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_runner, "safe_subprocess_environment", lambda: {"PATH": "safe"})

    result = codex_runner.run_codex_turn(
        context=_context(), prompt="prompt", model="gpt-5.6-terra", reasoning_effort="medium", output_schema={}
    )

    assert observed["env"] == {"PATH": "safe"}
    assert observed["shell"] is False
    assert result.token_usage == {"total_tokens": 2}
    assert tuple(call.tool for call in result.tool_calls) == EXPECTED_TOOL_NAMES


def test_host_policy_derives_long_geometry_and_rejects_bad_trending_evidence():
    """A model direction is insufficient without every frozen 40%/RSI/EMA gate."""

    proposal = _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION")
    accepted = CPRHostPolicy().validate(_context(), proposal)
    rejected_context = _context()
    rejected_context["momentum_vwap"]["vwap"]["entry_candle"]["body_fraction_above"] = 0.39
    rejected = CPRHostPolicy().validate(rejected_context, proposal)

    assert accepted.accepted is True
    assert accepted.entry_price == 100.0
    assert accepted.stop_price == 95.0
    assert accepted.milestone_price == 108.0
    assert accepted.final_target_price == 118.0
    assert rejected.accepted is False
    assert rejected.validation_code == "vwap_body_fraction_rejected"


@pytest.mark.parametrize(
    ("action", "regime", "setup", "is_flat", "direction", "accepted", "code"),
    [
        ("ENTER_LONG", "SIDEWAYS", "SIDEWAYS_SRSI", True, None, True, "accepted_entry"),
        ("ENTER_SHORT", "SIDEWAYS", "SIDEWAYS_SRSI", True, None, True, "accepted_entry"),
        ("EXIT", "UNDECIDED", "PREMISE_EXIT", False, "LONG", True, "accepted_exit"),
        ("SCALE_IN", "TRENDING", "R1_SCALE_IN", False, "LONG", True, "accepted_scale_in"),
        ("SCALE_IN", "TRENDING", "R1_SCALE_IN", True, None, False, "flat_action_rejected"),
    ],
)
def test_host_policy_enforces_flat_open_matrix_and_exit_scale_in_contract(
    action, regime, setup, is_flat, direction, accepted, code
):
    """Only the documented flat/open action families may cross the host boundary."""

    context = _context(is_flat=is_flat, direction=direction)
    if action == "ENTER_SHORT":
        context["momentum_vwap"]["stochastic_rsi"] = {"cross_up_in_oversold": False, "cross_down_in_overbought": True}
        context["market_structure"]["swings"]["highs"] = [{"price": 106.0}]
    outcome = CPRHostPolicy().validate(context, _proposal(action, regime, setup))

    assert outcome.accepted is accepted
    assert outcome.validation_code == code
    if action == "EXIT" and accepted:
        assert outcome.entry_price is None and outcome.stop_price is None
    if action == "SCALE_IN" and accepted:
        assert outcome.scale_in_permitted is True


def test_agent_rejects_schema_model_prompt_staleness_timeout_and_second_same_bar():
    """A bad echo, late result, or duplicate completed bar cannot produce an order plan."""

    good = _proposal("HOLD", "SIDEWAYS", "NONE")
    bad_model = good.model_copy(update={"model_used": "other"})
    mismatch = CPRAgent(runner=_runner(bad_model), model="gpt-5.6-terra").decide(_context(), bar_signature="bar")
    stale = CPRAgent(runner=_runner(good)).decide(_context(), bar_signature="bar", current_signature=lambda: "new")
    slow = CPRAgent(runner=_runner(good, delay=0.05), timeout_seconds=0.01).decide(_context(), bar_signature="bar")
    once = CPRAgent(runner=_runner(good))
    first = once.decide(_context(), bar_signature="one")
    second = once.decide(_context(), bar_signature="one")

    assert mismatch.validation_code == "model_mismatch"
    assert stale.validation_code == "stale_bar_signature"
    assert slow.validation_code == "timeout"
    assert first.validation_code == "accepted_hold"
    assert second.validation_code == "duplicate_bar"


def test_timeout_returns_before_a_late_sdk_thread_finishes():
    """A late SDK call must not hold the market worker past its 10ms deadline."""

    started = time.monotonic()
    outcome = CPRAgent(runner=_runner(_proposal("HOLD", "UNDECIDED", "NONE"), delay=0.1), timeout_seconds=0.01).decide(
        _context(), bar_signature="late"
    )

    assert outcome.validation_code == "timeout"
    assert time.monotonic() - started < 0.06


def test_late_sdk_thread_blocks_a_second_concurrent_bar_until_it_finishes():
    """A timeout must not let a second model turn overlap the late first turn."""

    release = threading.Event()
    calls = []

    def slow_runner(**_kwargs):
        calls.append("called")
        release.wait(timeout=0.3)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(),
        )

    agent = CPRAgent(runner=slow_runner, timeout_seconds=0.01)
    first = agent.decide(_context(), bar_signature="first")
    second = agent.decide(_context(), bar_signature="second")
    release.set()

    assert first.validation_code == "timeout"
    assert second.validation_code == "inference_in_progress"
    assert calls == ["called"]


def test_decision_log_and_order_free_smokes_keep_only_sanitized_host_evidence(tmp_path, capsys):
    """Audit rows support later execution without leaking secrets or creating orders."""

    path = tmp_path / "decisions.jsonl"
    outcome = CPRHostPolicy().validate(_context(), _proposal("HOLD", "SIDEWAYS", "NONE"))
    CPRDecisionLogger(str(path)).write(
        frozen_context={**_context(), "position_state": {"is_flat": True, "access_token": "secret"}},
        proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
        outcome=outcome,
        latency_ms=12,
        token_usage={"total_tokens": 17},
        tool_evidence=[record.__dict__ for record in _calls()],
    )

    raw = path.read_text(encoding="utf-8")
    row = json.loads(raw)
    assert "secret" not in raw
    assert row["validation"]["code"] == "accepted_hold"
    assert row["execution"] == {"mode": "ORDER_FREE", "submitted": False}
    assert runner_main(["--synthetic", "--fake"]) == 0
    assert runner_main(["--synthetic", "--authenticated"], authenticated_runner=smoke_fake_runner) == 0
    assert capsys.readouterr().out.count("NO ORDER") == 2


def test_geometry_uses_next_directional_level_not_hard_coded_r1_s1():
    """Already beyond R1/S1, the next frozen milestone must be used instead."""

    long_context = _context()
    long_context["session_levels"]["levels"].update({"r1": 90.0, "r2": 120.0})
    long_context["session_levels"]["next_levels"]["upside"] = {"name": "r2", "price": 120.0}
    short_context = _context()
    short_context["session_levels"]["current_close"] = 85.0
    short_context["session_levels"]["levels"].update({"s1": 90.0, "s2": 70.0})
    short_context["session_levels"]["next_levels"]["downside"] = {"name": "s2", "price": 70.0}
    short_context["momentum_vwap"].update(
        {
            "rsi14": 60.0,
            "ema": {"order": "EMA5_BELOW_EMA20", "ema5_slope": -1.0, "ema20_slope": -1.0},
            "candle": {"high": 90.0},
        }
    )
    short_context["momentum_vwap"]["vwap"] = {
        "sequence_evidence": {"all_recent_below": True},
        "entry_candle": {"body_fraction_below": 0.5},
    }

    long = CPRHostPolicy().validate(long_context, _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION"))
    short = CPRHostPolicy().validate(short_context, _proposal("ENTER_SHORT", "TRENDING", "TRENDING_VWAP_CONTINUATION"))

    assert long.milestone_price == 118.0
    assert short.milestone_price == 72.0


def test_valid_boundary_persists_regime_even_when_host_rejects_entry_and_boundary_failure_does_not():
    """Regime is advisory state, independent from deterministic execution permission."""

    rejected_context = _context()
    rejected_context["momentum_vwap"]["vwap"]["entry_candle"]["body_fraction_above"] = 0.0
    decision = _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION")
    persisted = CPRAgent(runner=_runner(decision)).decide(rejected_context, bar_signature="valid-boundary")
    malformed = CPRAgent(runner=lambda **_kwargs: CPRAgentRunResult("{", _calls())).decide(
        _context(), bar_signature="bad"
    )
    broken_context = _context()
    broken_context["position_state"] = {"is_flat": "unknown"}
    context_failure = CPRAgent(runner=_runner(decision)).decide(broken_context, bar_signature="bad-context")

    assert persisted.validation_code == "vwap_body_fraction_rejected"
    assert persisted.accepted_regime == "TRENDING"
    assert malformed.accepted_regime is None
    assert context_failure.validation_code == "invalid_position_state"
    assert context_failure.accepted_regime is None


def test_agent_rejects_malformed_schema_and_prompt_version_mismatch():
    """A syntactically valid response must still be schema and prompt pinned."""

    malformed_schema = CPRAgent(runner=lambda **_kwargs: CPRAgentRunResult('{"action":"HOLD"}', _calls())).decide(
        _context(), bar_signature="schema"
    )
    wrong_prompt = _proposal("HOLD", "UNDECIDED", "NONE").model_copy(update={"prompt_version": "old"})
    mismatch = CPRAgent(runner=_runner(wrong_prompt)).decide(_context(), bar_signature="prompt")

    assert malformed_schema.validation_code == "malformed_output"
    assert mismatch.validation_code == "prompt_version_mismatch"


def test_child_uses_one_authoritative_config_and_public_sdk_item_contract(monkeypatch, tmp_path):
    """The child passes the pure isolation config and exact public turn arguments."""

    observed = {}

    class Thread:
        def run(self, prompt, **kwargs):
            observed["run"] = (prompt, kwargs)
            items = [
                SimpleNamespace(root=SimpleNamespace(type="agentMessage")),
                SimpleNamespace(root=SimpleNamespace(type="reasoning")),
                *[
                    SimpleNamespace(root=SimpleNamespace(type="mcpToolCall", tool=name, status="completed"))
                    for name in EXPECTED_TOOL_NAMES
                ],
            ]
            return SimpleNamespace(
                final_response="{}", items=items, usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5)
            )

    class Codex:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def thread_start(self, **kwargs):
            observed["start"] = kwargs
            return Thread()

    sdk = SimpleNamespace(
        Codex=Codex, Sandbox=SimpleNamespace(read_only="read"), ApprovalMode=SimpleNamespace(deny_all="deny")
    )
    monkeypatch.setitem(sys.modules, "openai_codex", sdk)
    request = {
        "snapshot_path": str(tmp_path / "snapshot.json"),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "prompt": "prompt",
        "output_schema": {"type": "object"},
    }

    response = codex_child._run_request(request)

    assert observed["start"]["config"] == codex_child.build_isolated_thread_config(str(tmp_path / "snapshot.json"))
    assert observed["start"]["approval_mode"] == "deny"
    assert observed["run"][1] == {
        "approval_mode": "deny",
        "output_schema": {"type": "object"},
        "model_reasoning_effort": "medium",
    }
    assert [call["tool"] for call in response["tool_calls"]] == list(EXPECTED_TOOL_NAMES)
    assert response["token_usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
