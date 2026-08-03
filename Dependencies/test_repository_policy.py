"""Regression tests for MAT-110 dependency and CI policy.

These checks keep the safety controls reviewable in ordinary pytest runs. They
do not contact package indexes or GitHub; they only validate committed policy.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import yaml
from check_env_config import audit, env_keys_read_by, source_files

ROOT = Path(__file__).resolve().parent.parent


def _requirement_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_optional_dependency_sets_are_exact_and_kotak_uses_official_tag():
    core = _requirement_lines("requirements.txt")
    ai = _requirement_lines("requirements-ai.txt")
    codex_ai = _requirement_lines("requirements-codex-ai.txt")
    brokers = _requirement_lines("requirements-brokers.txt")

    assert "requests==2.34.2" in core
    assert "python-dotenv==1.2.2" in core
    # The full quality job imports the vendored Shoonya client while measuring
    # broker-adapter coverage, so its import-time WebSocket dependency belongs
    # in the core test/runtime environment as well as the isolated broker set.
    assert "websocket-client==1.8.0" in core
    # dhanhq.marketfeed (the websocket market data producer) hard-imports the
    # async `websockets` library at package import time, so the exact version
    # must be pinned in core rather than left to transitive resolution.
    assert "websockets==16.0" in core
    assert "claude-agent-sdk==0.2.123" in ai
    assert "pydantic==2.13.4" in ai
    assert all("==" in line for line in ai)
    # The independent CPR agent is an optional, subscription-authenticated
    # runtime. Keep its small compatibility set exact and reviewable.
    assert codex_ai == [
        "openai-codex==0.144.4",
        "mcp==1.28.1",
        "pydantic==2.13.4",
    ]
    assert "pyotp==2.9.0" in brokers
    assert "websocket-client==1.8.0" in brokers
    assert any(
        line == (
            "neo_api_client @ git+https://github.com/Kotak-Neo/"
            "Kotak-neo-api-v2.git@v2.0.1"
        )
        for line in brokers
    )
    assert all("==" in line or " @ git+" in line for line in brokers)


def test_ci_runs_audit_branch_coverage_and_every_exact_dependency_set():
    workflow = (ROOT / ".github/workflows/quality-and-security.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    core_job = workflow.split("\n  broker-dependencies:", maxsplit=1)[0]

    assert set(parsed["jobs"]) == {"verify", "broker-dependencies"}
    assert "requirements-ai.txt" in workflow
    assert "requirements-codex-ai.txt" in core_job
    assert "requirements-brokers.txt" in workflow
    assert "broker-dependencies:" in workflow
    assert "requirements-brokers.txt" not in core_job
    # Hosted verification is deliberately order-free and authentication-free.
    assert "--authenticated" not in workflow
    assert "python -m pip_audit" in workflow
    assert "python -m coverage run" in workflow
    assert "scripts/check_coverage_thresholds.py" in workflow


def test_dependabot_updates_python_and_github_actions_weekly():
    config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    ecosystems = {
        item["package-ecosystem"]: item["schedule"]["interval"]
        for item in config["updates"]
    }

    assert ecosystems == {"pip": "weekly", "github-actions": "weekly"}


def test_coverage_config_is_branch_enabled_and_preserves_overall_baseline():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["coverage"]["run"]["branch"] is True
    assert config["tool"]["coverage"]["report"]["fail_under"] == 54.7


def test_mypy_covers_the_complete_identifier_named_cpr_ai_runtime():
    """A new importable CPR module must not silently fall outside mypy."""

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    cpr_ai_files = {
        "Signal Generators/CPR AI Agent/cpr_ai_agent.py",
        "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py",
        "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py",
        "Signal Generators/CPR AI Agent/cpr_ai_context.py",
        "Signal Generators/CPR AI Agent/cpr_ai_decision_log.py",
        "Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py",
        "Signal Generators/CPR AI Agent/cpr_ai_prompt.py",
        "Signal Generators/CPR AI Agent/cpr_ai_runner.py",
        "Signal Generators/CPR AI Agent/cpr_ai_schema.py",
        "Signal Generators/CPR AI Agent/cpr_ai_signals.py",
        "Signal Generators/CPR AI Agent/cpr_ai_tools.py",
    }

    assert cpr_ai_files <= set(mypy["files"])
    assert "Signal Generators/CPR AI Agent" in mypy["mypy_path"]
    assert "Nifty Multi Strategy Front Test - Master File.py" not in mypy["files"]
    assert any(
        "openai_codex.*" in override.get("module", [])
        and override.get("ignore_missing_imports") is True
        for override in mypy["overrides"]
    )


def test_cpr_ai_env_defaults_match_the_independent_host_contract():
    """Catch stale arbiter knobs and undocumented mechanical host invariants."""

    env_text = (ROOT / "Dependencies/env.example").read_text(encoding="utf-8")
    env_values = {
        key.strip(): value.strip()
        for line in env_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }
    expected = {
        "CPR_AI_ENABLED": "false",
        "CPR_AI_MODEL": "gpt-5.6-terra",
        "CPR_AI_REASONING_EFFORT": "medium",
        "CPR_AI_SDK_TIMEOUT_SECONDS": "90",
        "CPR_AI_LOTS": "1",
        "CPR_AI_MAX_LOSS": "5500",
        "CPR_AI_SIZE_MULTIPLIER": "1",
        "CPR_AI_POLL_SECONDS": "5",
        "CPR_AI_TRADING_START_HOUR": "9",
        "CPR_AI_TRADING_START_MINUTE": "30",
        "CPR_AI_ENTRY_CUTOFF_HOUR": "15",
        "CPR_AI_ENTRY_CUTOFF_MINUTE": "0",
        "CPR_AI_SQUARE_OFF_HOUR": "15",
        "CPR_AI_SQUARE_OFF_MINUTE": "15",
        "CPR_AI_DECISION_LOGGING_ENABLED": "true",
        "CPR_AI_DECISION_LOG_PATH": "Backtest Outputs/cpr_ai_decisions.jsonl",
        "CPR_AI_VIRTUAL_TRADING": "true",
        "CPR_AI_LIVE_TRADING": "false",
    }

    assert {key: env_values.get(key) for key in expected} == expected
    assert "CPR_AI_ITM_OFFSET" not in env_text
    cpr_ai_section = env_text.split(
        "# CPR Codex AI Agent", maxsplit=1
    )[1].split("# Supertrend Bullish strategy", maxsplit=1)[0]
    lower = cpr_ai_section.lower()
    for required_explanation in (
        "completed five-minute candles",
        "one equal-size add",
        "30 nifty points",
        "2 nifty points",
        "0.40",
        "rsi 14 / stochastic 14 / k 3 / d 3 / zones 20 and 80",
        "live_trading_enabled=true",
        "cpr_ai_live_trading=true",
        "independent positions and p&l",
    ):
        assert required_explanation in lower
    assert "paper only" not in lower


def test_cpr_ai_documentation_rejects_obsolete_arbiter_and_worker_disable_guidance():
    """Keep operator instructions aligned with the final independent worker."""

    repository_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    focused_readme = (ROOT / "Signal Generators/CPR AI Agent/README.md").read_text(
        encoding="utf-8"
    )
    cpr_ai_summary = "\n".join(
        line
        for line in repository_readme.splitlines()
        if "CPR Codex AI Agent" in line or "CPRAIWorker" in line
    )
    combined = f"{cpr_ai_summary}\n{focused_readme}"
    lower = combined.lower()

    for obsolete in (
        "cpr codex ai agent groundwork",
        "paper-only",
        "paper only",
        "computes the existing algo 1/2/3",
        "select one triggered strategy",
        "cpr_virtual_trading=false",
        "cpr_algo3_virtual_trading=false",
        "must be disabled",
        "can never be live",
    ):
        assert obsolete not in lower

    for tool_name in (
        "session_levels",
        "momentum_vwap",
        "market_structure",
        "position_state",
    ):
        assert tool_name in focused_readme
    assert "LIVE_TRADING_ENABLED=true" in focused_readme
    assert "CPR_AI_LIVE_TRADING=true" in focused_readme
    assert "independent positions" in lower
    assert "independent p&l" in lower
    assert (
        'python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake'
        in focused_readme
    )
    assert (
        'python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --authenticated'
        in focused_readme
    )
    assert "automated verification" in lower
    assert "no billed/model/broker call" in lower


def test_agent_architecture_docs_stay_in_sync_and_cover_the_optional_cpr_agent():
    """The two agent guides share one runtime source of truth."""

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    marker = "## What this project is"
    agents_runtime = agents.split(marker, maxsplit=1)[1]
    claude_runtime = claude.split(marker, maxsplit=1)[1]

    assert agents_runtime == claude_runtime
    lower = agents_runtime.lower()
    assert "cpr codex ai agent" in lower
    assert "four frozen" in lower
    assert "double gate" in lower


def test_every_env_setting_the_code_reads_is_documented_in_env_example():
    """A new `.env` knob must ship with its `env.example` entry.

    `env.example` is the ONLY discovery surface for configuration -- the real
    `.env` is gitignored, so a key that never reaches the template is invisible
    to the operator and silently runs on whatever in-code default it was born
    with. This gate closes that gap at the point it opens: twelve keys had
    already drifted out of the operator's file before it was added.

    One direction only (code -> template). The reverse would flag the ~200
    per-strategy `<PREFIX>_*` knobs that `_signal_gen_ops` builds from
    f-strings, which are real settings the AST cannot see.
    """
    # Same helpers the `python algo.py check-env` diagnostic uses, so the gate
    # and the operator-facing tool can never disagree about what "documented"
    # means.
    read: set[str] = set()
    for path in source_files(ROOT):
        read |= env_keys_read_by(path)

    # Sanity check: if the AST walk silently stopped matching (a helper was
    # renamed, say), this test would "pass" while checking nothing at all.
    assert len(read) > 300, f"env-key extraction looks broken: found only {len(read)}"

    undocumented = audit(ROOT)["undocumented"]
    assert not undocumented, (
        "these env settings are read by the code but missing from "
        "Dependencies/env.example: " + ", ".join(undocumented)
    )


def test_coverage_threshold_checker_enforces_safety_and_broker_budgets():
    path = ROOT / "scripts/check_coverage_thresholds.py"
    spec = importlib.util.spec_from_file_location("check_coverage_thresholds", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    safety_path = next(iter(module.SAFETY_THRESHOLDS))
    broker_path = next(iter(module.BROKER_THRESHOLDS))
    report = {
        "files": {
            safety_path: {
                "summary": {"percent_covered": 89.99, "num_branches": 2},
            },
            broker_path.replace("/", "\\"): {
                "summary": {"percent_covered": 80.0, "num_branches": 2},
            },
        }
    }

    failures = module.evaluate_coverage(
        report,
        safety_thresholds={safety_path: 90.0},
        broker_thresholds={broker_path: 80.0},
    )

    assert len(failures) == 1
    assert safety_path in failures[0]
