"""Specify the isolated four-tool Codex runtime and host safety policy.

Every model response in this module is a local deterministic fake. The tests
exercise the real schema, tool-evidence gate, host policy, deadline behavior,
auth-only profile builder, and JSONL sanitizer without authenticating an SDK or
touching a broker. This matters because model output and child-process output
are both untrusted inputs in a live-money system.

The subprocess import is used only with a monkeypatched ``subprocess.run`` that
returns a local ``CompletedProcess``. No test launches a user- or model-selected
command, makes a network request, or creates an order.
"""

from __future__ import annotations

import itertools
import json
import subprocess  # nosec B404
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import cpr_ai_agent
import cpr_ai_codex_runner as codex_runner
import cpr_ai_codex_subprocess as codex_child
import cpr_ai_mcp_server as mcp_server
import pytest
from cpr_ai_agent import (
    CPRAgent,
    CPRAgentRunResult,
    CPRHostPolicy,
    CPRToolCallRecord,
    CPRTurnRequestKind,
)
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
    buffered R1 milestone at 108, and a final R2 target at 118. Tests mutate
    only the boundary they name so rejection codes remain diagnostic.
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
    """Build a schema-valid advisory classification with no execution fields."""

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
    """Create exact or deliberately defective four-tool evidence.

    ``missing`` omits an expected read, ``failed`` preserves its name with a
    failed status, and ``unexpected`` appends a forbidden shell attempt. This
    isolates the host's three distinct evidence failures.
    """

    records = [
        CPRToolCallRecord(tool=name, status="failed" if name == failed else "completed")
        for name in EXPECTED_TOOL_NAMES
        if name != missing
    ]
    if unexpected:
        records.append(CPRToolCallRecord(tool="shell", status="completed"))
    return tuple(records)


def _runner(proposal: CPRAgentDecision, *, calls=None, delay: float = 0.0):
    """Return a deterministic SDK stand-in with optional latency/evidence.

    The closure accepts the same keyword surface as the real adapter but returns
    only local JSON and call records, so cadence and timeout behavior are tested
    without an authenticated model call.
    """

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


@pytest.mark.parametrize(
    "first_attempt_calls",
    [
        (),
        _calls(failed="position_state"),
    ],
    ids=["missing-all-tools", "failed-one-tool"],
)
def test_agent_retries_one_incomplete_frozen_tool_attempt_within_the_same_deadline(first_attempt_calls):
    """A transient MCP-evidence miss gets one same-snapshot retry, never authority.

    The first local result models the two production failure shapes: Codex may
    skip every MCP read, or one MCP read may fail.  The second result is valid.
    A regression that removes the retry, changes the frozen bar between
    attempts, or resets the configured deadline will fail this test.
    """

    attempts = []

    def runner(**kwargs):
        attempts.append(kwargs)
        calls = first_attempt_calls if len(attempts) == 1 else _calls()
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=tuple(calls),
            token_usage={"total_tokens": 10 + len(attempts)},
        )

    frozen = _context()
    outcome = CPRAgent(runner=runner, timeout_seconds=17.5).decide(
        frozen,
        bar_signature="same-frozen-bar",
    )

    assert outcome.validation_code == "accepted_hold"
    assert outcome.inference_attempts == 2
    assert outcome.token_usage == {"total_tokens": 23}
    assert len(attempts) == 2
    assert attempts[0]["context"] is frozen
    assert attempts[1]["context"] is frozen
    assert attempts[0]["bar_signature"] == "same-frozen-bar"
    assert attempts[1]["bar_signature"] == "same-frozen-bar"
    assert attempts[0]["request_kind"] is CPRTurnRequestKind.NORMAL
    assert attempts[1]["request_kind"] is CPRTurnRequestKind.TOOL_REPAIR
    assert 0 < attempts[1]["timeout_seconds"] <= attempts[0]["timeout_seconds"] <= 17.5
    assert [(item.attempt_number, item.request_kind, item.evidence_code) for item in outcome.attempt_evidence] == [
        (1, "normal", "missing_tool_call" if not first_attempt_calls else "failed_tool_call"),
        (2, "tool_repair", None),
    ]


def test_agent_keeps_frozen_context_signature_and_remaining_deadline_for_failed_tool_repair():
    """A failed frozen read gets one repair turn against precisely the original bar."""

    attempts = []

    def runner(**kwargs):
        attempts.append(kwargs)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(failed="position_state") if len(attempts) == 1 else _calls(),
            token_usage={"input_tokens": 7, "total_tokens": 11, "model_context_window": 128000},
        )

    frozen = _context()
    outcome = CPRAgent(runner=runner, timeout_seconds=13.0).decide(frozen, bar_signature="failed-read-bar")

    assert outcome.validation_code == "accepted_hold"
    assert [attempt["request_kind"] for attempt in attempts] == [
        CPRTurnRequestKind.NORMAL,
        CPRTurnRequestKind.TOOL_REPAIR,
    ]
    assert all(attempt["context"] is frozen for attempt in attempts)
    assert [attempt["bar_signature"] for attempt in attempts] == ["failed-read-bar", "failed-read-bar"]
    assert 0 < attempts[1]["timeout_seconds"] <= attempts[0]["timeout_seconds"] <= 13.0
    assert outcome.token_usage == {"input_tokens": 14, "total_tokens": 22, "model_context_window": 128000}
    assert [item.tool_records for item in outcome.attempt_evidence] == [
        _calls(failed="position_state"),
        _calls(),
    ]


def test_repair_runtime_error_preserves_completed_first_attempt_audit_without_exception_text():
    """A second-turn adapter failure must not erase the first rejected evidence."""

    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("operator secret must never reach an audit row")
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(missing="market_structure"),
            token_usage={"input_tokens": 5, "total_tokens": 8, "model_context_window": 128000},
        )

    outcome = CPRAgent(runner=runner).decide(_context(), bar_signature="repair-runtime-error")

    assert outcome.action == "HOLD"
    assert outcome.validation_code == "runtime_error"
    assert "secret" not in outcome.validation_reason
    assert outcome.inference_attempts == 2
    assert outcome.tool_evidence == _calls(missing="market_structure")
    assert outcome.token_usage == {"input_tokens": 5, "total_tokens": 8, "model_context_window": 128000}
    recorded_attempts = [
        (item.request_kind, item.evidence_code, item.tool_records, item.token_usage)
        for item in outcome.attempt_evidence
    ]
    assert recorded_attempts == [
        (CPRTurnRequestKind.NORMAL, "missing_tool_call", _calls(missing="market_structure"), outcome.token_usage),
        (CPRTurnRequestKind.TOOL_REPAIR, "runtime_error", (), {}),
    ]
    assert len(calls) == 2


def test_repair_deadline_exhaustion_preserves_completed_first_attempt_audit(monkeypatch):
    """A spent total deadline records a second repair timeout without inventing evidence."""

    # The first sample starts the shared five-second budget. The second shows
    # that budget fully consumed before a repair child can be launched.
    clock = itertools.chain((100.0, 105.0), itertools.repeat(105.0))
    monkeypatch.setattr(cpr_ai_agent, "monotonic", lambda: next(clock))
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(failed="position_state"),
            token_usage={"total_tokens": 13},
        )

    outcome = CPRAgent(runner=runner, timeout_seconds=5.0).decide(_context(), bar_signature="repair-deadline")

    assert outcome.action == "HOLD"
    assert outcome.validation_code == "timeout"
    assert outcome.inference_attempts == 2
    assert outcome.tool_evidence == _calls(failed="position_state")
    assert outcome.token_usage == {"total_tokens": 13}
    recorded_attempts = [
        (item.request_kind, item.evidence_code, item.tool_records, item.token_usage)
        for item in outcome.attempt_evidence
    ]
    assert recorded_attempts == [
        (CPRTurnRequestKind.NORMAL, "failed_tool_call", _calls(failed="position_state"), {"total_tokens": 13}),
        (CPRTurnRequestKind.TOOL_REPAIR, "timeout", (), {}),
    ]
    assert len(calls) == 1


def test_outer_normal_deadline_timeout_records_started_normal_attempt_without_invented_evidence():
    """The parent deadline must retain a normal-turn diagnostic while its thread unwinds."""

    release = threading.Event()
    started = threading.Event()

    def runner(**_kwargs):
        started.set()
        release.wait(timeout=0.3)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(),
            token_usage={"total_tokens": 99},
        )

    try:
        outcome = CPRAgent(runner=runner, timeout_seconds=0.01).decide(
            _context(), bar_signature="normal-outer-timeout"
        )
    finally:
        release.set()

    assert started.is_set()
    assert outcome.action == "HOLD"
    assert outcome.validation_code == "timeout"
    assert outcome.inference_attempts == 1
    recorded_attempts = [
        (item.request_kind, item.evidence_code, item.tool_records, item.token_usage)
        for item in outcome.attempt_evidence
    ]
    assert recorded_attempts == [(CPRTurnRequestKind.NORMAL, "timeout", (), {})]
    assert outcome.tool_evidence == ()
    assert outcome.token_usage == {}


def test_normal_runner_timeout_error_records_typed_timeout_without_raw_exception_text():
    """A child timeout before its first response stays fail-closed and auditable."""

    def runner(**_kwargs):
        raise subprocess.TimeoutExpired(cmd="private child command", timeout=1.0)

    outcome = CPRAgent(runner=runner).decide(_context(), bar_signature="normal-inner-timeout")

    assert outcome.action == "HOLD"
    assert outcome.validation_code == "timeout"
    assert "private" not in outcome.validation_reason
    assert outcome.inference_attempts == 1
    recorded_attempts = [
        (item.request_kind, item.evidence_code, item.tool_records, item.token_usage)
        for item in outcome.attempt_evidence
    ]
    assert recorded_attempts == [(CPRTurnRequestKind.NORMAL, "timeout", (), {})]
    assert outcome.tool_evidence == ()
    assert outcome.token_usage == {}


@pytest.mark.parametrize(
    ("kind", "runner", "expected_code", "expected_attempts"),
    [
        pytest.param(
            "outer_timeout",
            None,
            "timeout",
            [(CPRTurnRequestKind.NORMAL, "timeout", (), {})],
            id="outer-timeout",
        ),
        pytest.param(
            "inner_timeout",
            lambda **_kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd="private timeout path", timeout=1.0)
            ),
            "timeout",
            [(CPRTurnRequestKind.NORMAL, "timeout", (), {})],
            id="inner-timeout",
        ),
        pytest.param(
            "first_runtime",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("credential private response")),
            "runtime_error",
            [(CPRTurnRequestKind.NORMAL, "runtime_error", (), {})],
            id="first-runtime",
        ),
    ],
)
def test_terminal_first_attempt_outcomes_capture_one_validation_signature(
    kind, runner, expected_code, expected_attempts
):
    """Terminal first-turn outcomes audit one host signature without raw errors."""

    release = threading.Event()
    started = threading.Event()
    if kind == "outer_timeout":

        def runner(**_kwargs):
            started.set()
            release.wait(timeout=0.3)
            return CPRAgentRunResult("{}", (), {})

        agent = CPRAgent(runner=runner, timeout_seconds=0.01)
    else:
        agent = CPRAgent(runner=runner)
    signature_samples = []

    try:
        outcome = agent.decide(
            _context(),
            bar_signature="frozen",
            current_signature=lambda: signature_samples.append("terminal-current") or "terminal-current",
        )
    finally:
        release.set()

    assert outcome.validation_code == expected_code
    assert outcome.validation_current_signature == "terminal-current"
    assert signature_samples == ["terminal-current"]
    assert [
        (item.request_kind, item.evidence_code, item.tool_records, item.token_usage)
        for item in outcome.attempt_evidence
    ] == expected_attempts
    assert "private" not in outcome.validation_reason


@pytest.mark.parametrize("repair_error", [TimeoutError(), RuntimeError("broker secret path")])
def test_terminal_repair_failures_capture_one_validation_signature(repair_error):
    """A repair terminal result retains its one final host-generation sample."""

    calls = []

    def runner(**_kwargs):
        calls.append("attempt")
        if len(calls) == 1:
            return CPRAgentRunResult(
                _proposal("HOLD", "SIDEWAYS", "NONE").model_dump_json(),
                _calls(missing="market_structure"),
                {"total_tokens": 3},
            )
        raise repair_error

    samples = []
    current_signatures = iter(("frozen", "repair-current"))
    outcome = CPRAgent(runner=runner).decide(
        _context(),
        bar_signature="frozen",
        current_signature=lambda: samples.append(next(current_signatures)) or samples[-1],
    )

    assert outcome.validation_code == ("timeout" if isinstance(repair_error, TimeoutError) else "runtime_error")
    assert outcome.validation_current_signature == "repair-current"
    assert samples == ["frozen", "repair-current"]
    assert [(item.request_kind, item.evidence_code) for item in outcome.attempt_evidence] == [
        (CPRTurnRequestKind.NORMAL, "missing_tool_call"),
        (CPRTurnRequestKind.TOOL_REPAIR, outcome.validation_code),
    ]


def test_decision_log_drops_untrusted_tool_and_status_text_from_every_evidence_shape(tmp_path):
    """Only canonical tool/status pairs may enter final or per-attempt JSONL."""

    unsafe = "C:/private auth broker order venue symbol quantity response"
    outcome = CPRHostPolicy().validate(_context(), _proposal("HOLD", "SIDEWAYS", "NONE"))
    outcome.attempt_evidence = (
        cpr_ai_agent.CPRAttemptEvidence(
            attempt_number=1,
            request_kind=CPRTurnRequestKind.NORMAL,
            evidence_code="unexpected_agent_action",
            tool_records=(CPRToolCallRecord(tool=unsafe, status=unsafe),),
            token_usage={},
        ),
    )
    path = tmp_path / "decisions.jsonl"
    CPRDecisionLogger(str(path)).write(
        frozen_context=_context(),
        proposal=None,
        outcome=outcome,
        latency_ms=1,
        token_usage={},
        tool_evidence=[{"tool": unsafe, "status": unsafe}],
    )

    raw = path.read_text(encoding="utf-8")
    row = json.loads(raw)
    assert unsafe not in raw
    assert row["tool_evidence"] == []
    assert row["attempt_evidence"][0]["tool_records"] == []


@pytest.mark.parametrize("first_calls", [_calls(missing="market_structure"), _calls(failed="position_state")])
def test_second_incomplete_tool_repair_remains_hold_after_exactly_two_attempts(first_calls):
    """A repair may not cascade into a third inference or a permissive result."""

    attempts = []

    def runner(**kwargs):
        attempts.append(kwargs)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=first_calls,
        )

    outcome = CPRAgent(runner=runner).decide(_context(), bar_signature="second-failure")

    assert outcome.action == "HOLD"
    assert outcome.validation_code in {"missing_tool_call", "failed_tool_call"}
    assert outcome.inference_attempts == 2
    assert len(attempts) == 2
    assert [item.request_kind for item in outcome.attempt_evidence] == ["normal", "tool_repair"]


@pytest.mark.parametrize(
    "defective_calls",
    [
        (*_calls(), CPRToolCallRecord(tool="session_levels", status="completed")),
        _calls(unexpected=True),
    ],
    ids=["duplicate-tool", "unexpected-capability"],
)
def test_duplicate_or_unexpected_evidence_never_starts_corrective_retry(defective_calls):
    """Capability-contract violations are final HOLDs, not repairable omissions."""

    attempts = []

    def runner(**kwargs):
        attempts.append(kwargs)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=defective_calls,
        )

    outcome = CPRAgent(runner=runner).decide(_context(), bar_signature="capability-violation")

    assert outcome.validation_code == "unexpected_agent_action"
    assert len(attempts) == 1
    assert outcome.inference_attempts == 1
    assert [(item.request_kind, item.evidence_code) for item in outcome.attempt_evidence] == [
        ("normal", "unexpected_agent_action")
    ]


def test_runtime_configuration_is_read_only_and_sanitizes_credentials(tmp_path):
    """A child gets no execution surface or trading/API credentials."""

    isolated_home = tmp_path / "isolated-codex-home"
    profile_home = tmp_path / "isolated-profile"
    safe = safe_subprocess_environment(
        {
            "PATH": "safe",
            "SYSTEMROOT": "windows",
            "HOME": "operator-home",
            "USERPROFILE": "operator-profile",
            "APPDATA": "operator-appdata",
            "LOCALAPPDATA": "operator-local-appdata",
            "CODEX_HOME": "operator-codex-home",
            "DHAN_ACCESS_TOKEN": "secret",
            "OPENAI_API_KEY": "key",
            "LIVE_TRADING_ENABLED": "true",
        },
        codex_home=isolated_home,
        profile_home=profile_home,
    )
    config = build_codex_thread_config(str(tmp_path / "snapshot.json"), "python.exe", str(tmp_path))

    assert safe == {
        "SYSTEMROOT": "windows",
        "CODEX_HOME": str(isolated_home),
        "HOME": str(profile_home),
        "USERPROFILE": str(profile_home),
        "APPDATA": str(profile_home / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(profile_home / "AppData" / "Local"),
        "TEMP": str(profile_home / "Temp"),
        "TMP": str(profile_home / "Temp"),
    }
    assert set(config["features"]) == {
        "apps",
        "browser_use",
        "browser_use_external",
        "collab",
        "collaboration_modes",
        "computer_use",
        "connectors",
        "enable_mcp_apps",
        "in_app_browser",
        "multi_agent",
        "plugin_sharing",
        "plugins",
        "remote_plugin",
        "shell_tool",
        "skill_mcp_dependency_install",
        "skill_search",
        "tool_search",
        "unified_exec",
    }
    assert all(value is False for value in config["features"].values())
    assert config["web_search"] == "disabled"
    assert config["mcp_servers"]["cpr_ai"]["enabled_tools"] == list(EXPECTED_TOOL_NAMES)


def test_auth_only_codex_home_copies_one_artifact_and_excludes_ambient_capabilities(tmp_path):
    """Config, MCP, plugin, skill, app, and rule sentinels cannot enter the child home."""

    source_home = tmp_path / "operator-codex-home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"tokens":"sentinel-auth"}', encoding="utf-8")
    (source_home / "config.toml").write_text('[mcp_servers.sentinel]', encoding="utf-8")
    for name in ("mcp", "plugins", "skills", "apps", "rules"):
        directory = source_home / name
        directory.mkdir()
        (directory / "sentinel.txt").write_text(name, encoding="utf-8")

    isolated_home = codex_runner.create_auth_only_codex_home(
        source_home,
        tmp_path / "runtime-root",
    )

    assert isolated_home != source_home
    assert [path.relative_to(isolated_home) for path in isolated_home.rglob("*")] == [Path("auth.json")]
    assert (isolated_home / "auth.json").read_text(encoding="utf-8") == '{"tokens":"sentinel-auth"}'


def test_process_auth_home_is_copy_once_and_preserves_child_refresh(tmp_path, monkeypatch):
    """Later turns reuse a child refresh instead of recopying stale operator auth."""

    source_home = tmp_path / "operator-codex-home"
    source_home.mkdir()
    (source_home / "auth.json").write_text('{"state":"operator-original"}', encoding="utf-8")
    monkeypatch.setattr(codex_runner, "_operator_codex_home", lambda: source_home)
    codex_runner._cleanup_process_codex_home()
    try:
        first_home = codex_runner.process_isolated_codex_home()
        (first_home / "auth.json").write_text('{"state":"child-refreshed"}', encoding="utf-8")

        second_home = codex_runner.process_isolated_codex_home()

        assert second_home == first_home
        assert (second_home / "auth.json").read_text(encoding="utf-8") == '{"state":"child-refreshed"}'
        assert (source_home / "auth.json").read_text(encoding="utf-8") == '{"state":"operator-original"}'
    finally:
        codex_runner._cleanup_process_codex_home()


def test_missing_auth_fails_before_the_isolated_subprocess_can_launch(tmp_path, monkeypatch):
    """No ambient profile fallback is permitted when subscription auth cannot be isolated."""

    source_home = tmp_path / "missing-auth-home"
    source_home.mkdir()
    with pytest.raises(RuntimeError, match="authentication"):
        codex_runner.create_auth_only_codex_home(source_home, tmp_path / "runtime-root")

    launched = False

    def fake_run(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("subprocess must not launch")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        codex_runner,
        "process_isolated_codex_home",
        lambda: (_ for _ in ()).throw(RuntimeError("authentication isolation unavailable")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="authentication"):
        codex_runner.run_codex_turn(
            context=_context(),
            prompt="prompt",
            model="gpt-5.6-terra",
            reasoning_effort="medium",
            output_schema={},
            timeout_seconds=5,
        )
    assert launched is False


def test_generated_mcp_command_reaches_the_real_server_parser(tmp_path, monkeypatch):
    """Generated MCP arguments must launch the four-tool frozen server."""

    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_context()), encoding="utf-8")
    config = build_codex_thread_config(
        str(snapshot_path), sys.executable, str(tmp_path)
    )
    command = config["mcp_servers"]["cpr_ai"]
    observed = {}

    def fake_run(server, *, transport):
        observed["transport"] = transport
        observed["tools"] = tuple(server._tool_manager._tools)

    monkeypatch.setattr("mcp.server.mcpserver.MCPServer.run", fake_run)

    assert mcp_server.main(command["args"][1:]) == 0
    assert observed == {
        "transport": "stdio",
        "tools": EXPECTED_TOOL_NAMES,
    }


def test_real_runner_uses_a_sanitized_subprocess_and_parses_only_structured_evidence(
    monkeypatch, tmp_path
):
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
    isolated_home = tmp_path / "codex-home"
    isolated_home.mkdir()
    (isolated_home / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        codex_runner,
        "process_isolated_codex_home",
        lambda: isolated_home,
    )
    monkeypatch.setattr(
        codex_runner,
        "safe_subprocess_environment",
        lambda **_kwargs: {"PATH": "safe"},
    )

    result = codex_runner.run_codex_turn(
        context=_context(), prompt="prompt", model="gpt-5.6-terra", reasoning_effort="medium", output_schema={}
    )

    assert observed["env"] == {"PATH": "safe"}
    assert observed["shell"] is False
    assert result.token_usage == {"total_tokens": 2}
    assert tuple(call.tool for call in result.tool_calls) == EXPECTED_TOOL_NAMES


def test_parent_serializes_enum_backed_tool_repair_kind_without_arbitrary_request_text(monkeypatch, tmp_path):
    """A closed enum -- never caller prose -- selects the fixed repair request."""

    requests = []

    def fake_run(command, **kwargs):
        requests.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "final_response": _proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
                    "tool_calls": [{"tool": name, "status": "completed"} for name in EXPECTED_TOOL_NAMES],
                    "token_usage": {},
                    "unexpected_actions": [],
                }
            ),
            stderr="",
        )

    isolated_home = tmp_path / "codex-home"
    isolated_home.mkdir()
    (isolated_home / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_runner, "process_isolated_codex_home", lambda: isolated_home)

    codex_runner.run_codex_turn(
        context=_context(),
        prompt="unchanged prompt",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        output_schema={},
        request_kind=CPRTurnRequestKind.TOOL_REPAIR,
    )

    assert requests[0]["request_kind"] == "tool_repair"
    assert set(requests[0]) == {
        "snapshot_path",
        "model",
        "reasoning_effort",
        "prompt",
        "output_schema",
        "request_kind",
    }


@pytest.mark.parametrize("invalid_kind", ["repair with these words", "unknown", 1, None])
def test_parent_rejects_non_enum_request_kind_before_child_launch(invalid_kind):
    """Caller/model text must never become a child repair instruction."""

    with pytest.raises(ValueError, match="request kind"):
        codex_runner.run_codex_turn(context=_context(), request_kind=invalid_kind)


@pytest.mark.parametrize("configured_timeout", [0.25, 135.0])
def test_real_runner_uses_the_configured_subprocess_timeout(configured_timeout, monkeypatch, tmp_path):
    """The child deadline follows the validated agent setting at lower and upper values."""

    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "final_response": _proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
                    "tool_calls": [{"tool": name, "status": "completed"} for name in EXPECTED_TOOL_NAMES],
                    "token_usage": {},
                    "unexpected_actions": [],
                }
            ),
            stderr="",
        )

    isolated_home = tmp_path / "codex-home"
    isolated_home.mkdir()
    (isolated_home / "auth.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(codex_runner, "process_isolated_codex_home", lambda: isolated_home, raising=False)

    codex_runner.run_codex_turn(
        context=_context(),
        prompt="prompt",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        output_schema={},
        timeout_seconds=configured_timeout,
    )

    assert observed["timeout"] == configured_timeout


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


def test_agent_validates_positive_timeout_and_passes_it_to_the_runtime():
    """Invalid deadlines fail at construction and valid ones reach the isolated runner."""

    with pytest.raises(ValueError, match="positive"):
        CPRAgent(timeout_seconds=0)
    observed = {}

    def runner(**kwargs):
        observed.update(kwargs)
        return CPRAgentRunResult(
            final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
            tool_calls=_calls(),
        )

    outcome = CPRAgent(runner=runner, timeout_seconds=17.5).decide(_context(), bar_signature="timeout-forwarded")

    assert outcome.validation_code == "accepted_hold"
    assert observed["timeout_seconds"] == 17.5
    assert "model_used must exactly equal gpt-5.6-terra" in observed["prompt"]
    confidence_schema = observed["output_schema"]["properties"]["confidence"]
    assert confidence_schema["minimum"] == 0
    assert confidence_schema["maximum"] == 10


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
    frozen = _context()
    frozen["session_levels"]["next_levels"]["ordered"] = [{"name": "r1", "price": 110.0}]
    frozen["position_state"] = {
        "is_flat": True,
        "access_token": "secret",
        "accessTokens": ["secret-plural"],
        "auth": "secret-auth",
        "broker": "secret-broker",
        "order_id": "secret-order",
        "venue": "secret-venue",
    }
    CPRDecisionLogger(str(path)).write(
        frozen_context=frozen,
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
    assert row["inference_attempts"] == 1
    assert row["frozen_context"]["session_levels"]["next_levels"]["ordered"] == [
        {"name": "r1", "price": 110.0}
    ]
    assert row["authoritative_geometry"]["action"] == "HOLD"
    assert row["token_usage"] == {"total_tokens": 17}
    assert row["execution"] == {"mode": "ORDER_FREE", "submitted": False}
    assert runner_main(["--synthetic", "--fake"]) == 0
    assert runner_main(["--synthetic", "--authenticated"], authenticated_runner=smoke_fake_runner) == 0
    assert capsys.readouterr().out.count("NO ORDER") == 2


def test_decision_log_removes_plural_sensitive_keys_but_keeps_near_matches(tmp_path):
    """Plural credentials/execution keys must not leak into the JSONL audit."""

    path = tmp_path / "decisions.jsonl"
    frozen = _context()
    outcome = CPRHostPolicy().validate(frozen, _proposal("HOLD", "SIDEWAYS", "NONE"))

    # These similarly spelled fields prove that the filter is token-aware.  It
    # must remove true credentials/execution metadata without deleting useful
    # strategy evidence merely because a word contains "order" or "auth".
    frozen["position_state"] = {
        "is_flat": True,
        "api_keys": ["SECRET-SNAKE-KEYS"],
        "apiKeys": ["SECRET-CAMEL-KEYS"],
        "brokers": ["SECRET-BROKER-LIST"],
        "orders": [{"id": "SECRET-ORDER-LIST"}],
        "passwords": ["SECRET-PASSWORD-LIST"],
        "venues": ["SECRET-VENUE-LIST"],
        "ordered": [{"name": "r1", "price": 110.0}],
        "authoritative_geometry": {"preserved": True},
    }
    CPRDecisionLogger(str(path)).write(
        frozen_context=frozen,
        proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
        outcome=outcome,
        latency_ms=12,
        token_usage={"total_tokens": 17},
        tool_evidence=[record.__dict__ for record in _calls()],
    )

    raw = path.read_text(encoding="utf-8")
    position_state = json.loads(raw)["frozen_context"]["position_state"]
    assert "SECRET-" not in raw
    assert position_state == {
        "is_flat": True,
        "ordered": [{"name": "r1", "price": 110.0}],
        "authoritative_geometry": {"preserved": True},
    }


def test_decision_log_records_ist_bar_coverage_and_typed_attempt_evidence_without_secrets(tmp_path):
    """An audit row must preserve host facts without retaining unsafe model text.

    Removing the timestamp, a bar-generation field, retry evidence, or the
    recursive redaction of a nested diagnostic must make this test fail.  The
    expected values are hand-authored so this does not mirror logger helpers.
    """

    calls = iter(
        [
            CPRAgentRunResult(
                _proposal("HOLD", "SIDEWAYS", "NONE").model_dump_json(),
                _calls(missing="market_structure"),
                {"input_tokens": 3, "total_tokens": 5},
            ),
            CPRAgentRunResult(
                _proposal("HOLD", "SIDEWAYS", "NONE").model_dump_json(),
                _calls(),
                {"output_tokens": 2, "total_tokens": 4},
            ),
        ]
    )
    validation_signatures = iter(["frozen-0930", "frozen-0930"])
    outcome = CPRAgent(runner=lambda **_kwargs: next(calls)).decide(
        _context(),
        bar_signature="frozen-0930",
        current_signature=lambda: next(validation_signatures),
    )
    path = tmp_path / "decisions.jsonl"
    metadata = {
        "bar_timestamp": "2026-08-13T09:30:00+05:30",
        "frozen_signature": "frozen-0930",
        "required_official_minutes": [
            "2026-08-13T09:30:00+05:30",
            "2026-08-13T09:31:00+05:30",
            "2026-08-13T09:32:00+05:30",
            "2026-08-13T09:33:00+05:30",
            "2026-08-13T09:34:00+05:30",
        ],
        "present_official_minutes": [
            "2026-08-13T09:30:00+05:30",
            "2026-08-13T09:31:00+05:30",
            "2026-08-13T09:32:00+05:30",
            "2026-08-13T09:33:00+05:30",
            "2026-08-13T09:34:00+05:30",
        ],
        "official_coverage": True,
        "nested": {
            "auth": "DO-NOT-LOG-AUTH",
            "local_path": "C:/private/decision.jsonl",
            "broker": "DO-NOT-LOG-BROKER",
            "order": "DO-NOT-LOG-ORDER",
            "venue": "DO-NOT-LOG-VENUE",
            "symbol": "DO-NOT-LOG-SYMBOL",
            "quantity": "DO-NOT-LOG-QUANTITY",
        },
    }
    logger = CPRDecisionLogger(str(path))
    logger.write(
        frozen_context=_context(),
        proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
        outcome=outcome,
        latency_ms=12,
        token_usage=outcome.token_usage,
        tool_evidence=[record.__dict__ for record in outcome.tool_evidence],
        audit_stage="PRE_ACTION",
        bar_metadata=metadata,
    )
    logger.write(
        frozen_context=_context(),
        proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
        outcome=outcome,
        latency_ms=12,
        token_usage=outcome.token_usage,
        tool_evidence=[record.__dict__ for record in outcome.tool_evidence],
        audit_stage="POST_ACTION",
        bar_metadata=metadata,
    )

    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines()]
    assert outcome.validation_current_signature == "frozen-0930"
    assert [row["audit_stage"] for row in rows] == ["PRE_ACTION", "POST_ACTION"]
    assert all(
        __import__("datetime").datetime.fromisoformat(row["recorded_at"]).utcoffset().total_seconds() == 19800
        for row in rows
    )
    assert rows[0]["bar"] == {
        "bar_timestamp": "2026-08-13T09:30:00+05:30",
        "frozen_signature": "frozen-0930",
        "validation_current_signature": "frozen-0930",
        "required_official_minutes": metadata["required_official_minutes"],
        "present_official_minutes": metadata["present_official_minutes"],
        "official_coverage": True,
        "nested": {},
    }
    assert rows[0]["attempt_evidence"] == [
        {
            "attempt_number": 1,
            "request_kind": "normal",
            "evidence_code": "missing_tool_call",
            "tool_records": [
                {"tool": "session_levels", "status": "completed"},
                {"tool": "momentum_vwap", "status": "completed"},
                {"tool": "position_state", "status": "completed"},
            ],
            "token_usage": {"input_tokens": 3, "total_tokens": 5},
        },
        {
            "attempt_number": 2,
            "request_kind": "tool_repair",
            "evidence_code": None,
            "tool_records": [
                {"tool": name, "status": "completed"} for name in EXPECTED_TOOL_NAMES
            ],
            "token_usage": {"output_tokens": 2, "total_tokens": 4},
        },
    ]
    assert "Synthetic test proposal." not in raw
    assert "DO-NOT-LOG" not in raw


def test_decision_log_direct_call_keeps_a_safe_default_stage_and_disabled_logging_is_a_noop(tmp_path):
    """Legacy direct callers must remain append-compatible and optional logging inert."""

    path = tmp_path / "decisions.jsonl"
    outcome = CPRHostPolicy().validate(_context(), _proposal("HOLD", "SIDEWAYS", "NONE"))
    CPRDecisionLogger(str(path)).write(
        frozen_context=_context(),
        proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
        outcome=outcome,
        latency_ms=1,
        token_usage={},
        tool_evidence=[],
    )
    CPRDecisionLogger(str(tmp_path / "disabled.jsonl"), enabled=False).write(
        frozen_context=_context(),
        proposal=None,
        outcome=outcome,
        latency_ms=1,
        token_usage={},
        tool_evidence=[],
    )

    assert json.loads(path.read_text(encoding="utf-8"))["audit_stage"] == "DIRECT"
    assert not (tmp_path / "disabled.jsonl").exists()


@pytest.mark.parametrize(
    ("calls", "signatures", "code", "expected_signature"),
    [
        pytest.param(
            _calls(missing="position_state"),
            ("new-before-retry", "new-at-validation"),
            "missing_tool_call",
            "new-at-validation",
            id="missing-tool",
        ),
        pytest.param(
            _calls(),
            ("new-at-validation",),
            "stale_bar_signature",
            "new-at-validation",
            id="stale-bar",
        ),
    ],
)
def test_agent_retains_the_exact_validation_signature_for_nonexecuting_outcomes(
    calls, signatures, code, expected_signature
):
    """The audit must use the validation sample, not an earlier retry check."""

    samples = iter(signatures)
    outcome = CPRAgent(
        runner=_runner(_proposal("HOLD", "SIDEWAYS", "NONE"), calls=calls)
    ).decide(
        _context(),
        bar_signature="frozen",
        current_signature=lambda: next(samples),
    )

    assert outcome.validation_code == code
    assert outcome.validation_current_signature == expected_signature


def test_decision_log_keeps_normal_and_failed_repair_attempt_evidence_parseable(tmp_path):
    """Every attempt shape remains useful when recovery still fails closed."""

    normal = CPRAgent(runner=_runner(_proposal("HOLD", "SIDEWAYS", "NONE"))).decide(
        _context(), bar_signature="normal-only"
    )
    attempts = iter(
        [
            CPRAgentRunResult(
                _proposal("HOLD", "SIDEWAYS", "NONE").model_dump_json(),
                _calls(missing="market_structure"),
                {"total_tokens": 3},
            ),
            CPRAgentRunResult(
                _proposal("HOLD", "SIDEWAYS", "NONE").model_dump_json(),
                _calls(failed="position_state"),
                {"total_tokens": 4},
            ),
        ]
    )
    failed_repair = CPRAgent(runner=lambda **_kwargs: next(attempts)).decide(
        _context(), bar_signature="failed-repair", current_signature=lambda: "failed-repair"
    )
    path = tmp_path / "decisions.jsonl"
    logger = CPRDecisionLogger(str(path))
    for outcome in (normal, failed_repair):
        logger.write(
            frozen_context=_context(),
            proposal=_proposal("HOLD", "SIDEWAYS", "NONE"),
            outcome=outcome,
            latency_ms=1,
            token_usage=outcome.token_usage,
            tool_evidence=[],
        )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["attempt_evidence"] == [
        {
            "attempt_number": 1,
            "request_kind": "normal",
            "evidence_code": None,
            "tool_records": [
                {"tool": name, "status": "completed"} for name in EXPECTED_TOOL_NAMES
            ],
            "token_usage": {"total_tokens": 17},
        }
    ]
    assert [(item["request_kind"], item["evidence_code"]) for item in rows[1]["attempt_evidence"]] == [
        ("normal", "missing_tool_call"),
        ("tool_repair", "failed_tool_call"),
    ]


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

    class McpToolCallStatus(Enum):
        """Mirror the public SDK enum whose printable form is not its value."""

        completed = "completed"

    class TokenUsageBreakdown:
        """Mirror the nested public per-turn token object."""

        def model_dump(self):
            return {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}

    class ThreadTokenUsage:
        """Mirror the public SDK aggregate usage shape."""

        def model_dump(self):
            return {
                "last": TokenUsageBreakdown(),
                "total": TokenUsageBreakdown(),
                "model_context_window": 128000,
            }

    class Thread:
        def run(self, prompt, *, approval_mode, output_schema, effort):
            """Accept only current public SDK turn parameters."""

            observed["run"] = (
                prompt,
                {"approval_mode": approval_mode, "output_schema": output_schema, "effort": effort},
            )
            items = [
                # The real SDK includes the submitted turn input in
                # ``TurnResult.items``.  It is bookkeeping, not a capability
                # used by Codex, so the host must not classify it as an action.
                SimpleNamespace(root=SimpleNamespace(type="userMessage")),
                SimpleNamespace(root=SimpleNamespace(type="agentMessage")),
                SimpleNamespace(root=SimpleNamespace(type="reasoning")),
                *[
                    SimpleNamespace(
                        root=SimpleNamespace(type="mcpToolCall", tool=name, status=McpToolCallStatus.completed)
                    )
                    for name in EXPECTED_TOOL_NAMES
                ],
            ]
            return SimpleNamespace(
                final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
                items=items,
                usage=ThreadTokenUsage(),
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
        "request_kind": "normal",
    }

    response = codex_child._run_request(request)

    assert observed["start"]["config"] == codex_child.build_isolated_thread_config(str(tmp_path / "snapshot.json"))
    assert observed["start"]["approval_mode"] == "deny"
    assert observed["start"]["developer_instructions"] == "prompt"
    assert observed["run"][0] == (
        "Before deciding, call each frozen MCP tool exactly once: session_levels, "
        "momentum_vwap, market_structure, and position_state. Wait for all four "
        "calls to complete, then evaluate the frozen CPR context and return one decision."
    )
    assert observed["run"][1] == {"approval_mode": "deny", "output_schema": {"type": "object"}, "effort": "medium"}
    assert [call["tool"] for call in response["tool_calls"]] == list(EXPECTED_TOOL_NAMES)
    assert [call["status"] for call in response["tool_calls"]] == ["completed"] * 4
    assert response["token_usage"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "model_context_window": 128000,
    }
    assert response["unexpected_actions"] == []
    assert all(value is False for value in observed["start"]["config"]["features"].values())
    assert set(observed["start"]["config"]["features"]) >= {
        "shell_tool",
        "unified_exec",
        "collab",
        "multi_agent",
        "apps",
        "plugins",
        "connectors",
        "browser_use",
        "tool_search",
    }
    assert "workspace_write" not in observed["start"]["config"]["features"]
    host = CPRAgent(
        runner=lambda **_kwargs: CPRAgentRunResult(
            final_response=response["final_response"],
            tool_calls=tuple(CPRToolCallRecord(**call) for call in response["tool_calls"]),
            token_usage=response["token_usage"],
        )
    )
    assert host.decide(_context(), bar_signature="enum-status").validation_code == "accepted_hold"


def test_child_repair_kind_uses_constant_corrective_request_with_unchanged_read_only_config(monkeypatch, tmp_path):
    """The repair is a new read-only thread, not model-supplied corrective text."""

    observed = {}

    class Thread:
        def run(self, prompt, *, approval_mode, output_schema, effort):
            observed["run"] = (prompt, approval_mode, output_schema, effort)
            return SimpleNamespace(
                final_response=_proposal("HOLD", "UNDECIDED", "NONE").model_dump_json(),
                items=[
                    SimpleNamespace(root=SimpleNamespace(type="mcpToolCall", tool=name, status="completed"))
                    for name in EXPECTED_TOOL_NAMES
                ],
                usage={},
            )

    class Codex:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def thread_start(self, **kwargs):
            observed["start"] = kwargs
            return Thread()

    monkeypatch.setitem(
        sys.modules,
        "openai_codex",
        SimpleNamespace(
            Codex=Codex,
            Sandbox=SimpleNamespace(read_only="read"),
            ApprovalMode=SimpleNamespace(deny_all="deny"),
        ),
    )

    response = codex_child._run_request(
        {
            "snapshot_path": str(tmp_path / "snapshot.json"),
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "prompt": "same system prompt",
            "output_schema": {"type": "object"},
            "request_kind": "tool_repair",
        }
    )

    assert observed["run"][0] == codex_child._TOOL_REPAIR_TURN_REQUEST
    assert "prior attempt" in observed["run"][0].lower()
    assert all(observed["run"][0].count(name) == 1 for name in EXPECTED_TOOL_NAMES)
    assert observed["start"]["developer_instructions"] == "same system prompt"
    assert observed["start"]["config"] == codex_child.build_isolated_thread_config(str(tmp_path / "snapshot.json"))
    assert observed["start"]["config"]["mcp_servers"]["cpr_ai"]["enabled_tools"] == list(EXPECTED_TOOL_NAMES)
    assert response["unexpected_actions"] == []


@pytest.mark.parametrize(
    "request_kind",
    [
        pytest.param(None, id="missing"),
        pytest.param("unknown", id="unknown"),
        pytest.param("arbitrary repair", id="arbitrary"),
    ],
)
def test_child_request_shape_requires_known_request_kind(request_kind, monkeypatch):
    """The JSON boundary rejects absent/unknown kinds before the SDK import."""

    request = {
        "snapshot_path": "snapshot.json",
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "prompt": "prompt",
        "output_schema": {},
    }
    if request_kind is not None:
        request["request_kind"] = request_kind
    monkeypatch.setattr(codex_child, "_run_request", lambda _request: pytest.fail("SDK path must not run"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(read=lambda: json.dumps(request)))
    captured = []
    monkeypatch.setattr(codex_child.json, "dump", lambda payload, _stream: captured.append(payload))

    assert codex_child.main() == 2
    assert captured == [{"ok": False, "error": "ValueError"}]
