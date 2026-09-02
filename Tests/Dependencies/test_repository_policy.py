"""Regression tests for MAT-110 dependency and CI policy.

These checks keep the safety controls reviewable in ordinary pytest runs. They
do not contact package indexes or GitHub; they only validate committed policy.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
import urllib.parse
from pathlib import Path

import yaml
from check_env_config import audit, env_keys_read_by, source_files

# Tests/Dependencies/<this file> -> the repository root is two levels up.
ROOT = Path(__file__).resolve().parents[2]


def _requirement_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_optional_dependency_sets_are_exact_and_kotak_uses_official_tag():
    core = _requirement_lines("requirements.txt")
    ai = _requirement_lines("requirements-ai.txt")
    brokers = _requirement_lines("requirements-brokers.txt")

    assert "requests==2.34.2" in core
    # 1.2.2 -> 1.2.3 (2026-08-24, PR #135). Patch release on the .env loader;
    # the runner reads every setting through it at startup, so a break would
    # be immediate and total rather than subtle.
    assert "python-dotenv==1.2.3" in core
    # The full quality job imports the vendored Shoonya client while measuring
    # broker-adapter coverage, so its import-time WebSocket dependency belongs
    # in the core test/runtime environment as well as the isolated broker set.
    assert "websocket-client==1.8.0" in core
    # dhanhq.marketfeed (the websocket market data producer) hard-imports the
    # async `websockets` library at package import time, so the exact version
    # must be pinned in core rather than left to transitive resolution.
    # Bumping this assertion is deliberate: dhanhq only declares
    # websockets>=12.0.1, so the pin is the only thing standing between a feed
    # regression and a live session. Move it together with requirements.txt, and
    # only once a PAPER session has confirmed the feed still ticks on the new
    # version -- CI never opens a real socket, so a green build proves nothing
    # about the transport.
    #
    # 17.0 -> 17.0.1 (2026-08-11, PR #119). The upstream tag-to-tag diff looks
    # alarming -- it removes `Server.wrap()` and reworks the asyncio server --
    # but every one of those is SERVER-side and marketfeed is a client. Its
    # entire surface is three names, all confirmed present in 17.0.1:
    # `websockets.connect`, `websockets.ConnectionClosed`, and
    # `websockets.protocol.State.CLOSED`. That rules out the import-time
    # "runner fails to start" failure this pin exists to prevent; it does NOT
    # prove the transport behaves, which only a session on the real socket can.
    # Operator decision: validate on the next run rather than ahead of the
    # merge, because MARKET_DATA_SOURCE=WEBSOCKET is active but every strategy
    # is PAPER (LIVE_TRADING_ENABLED=false), so a bad feed costs a session and
    # not money. If the feed does not tick, revert this pin first.
    # 17.0.1 -> 17.1 (2026-09-01, PR #144). Upstream calls 17.1 purely additive
    # and the changelog confirms it: the backward-incompatible release was 17.0,
    # where the new asyncio implementation became the default, and this pin is
    # already past that. Everything new in 17.1 — reconnect(), redirect-following,
    # alternate host/port — lands in the THREADING implementation, which
    # dhanhq.marketfeed does not use; it is an asyncio client. marketfeed's entire
    # surface is still the same three names, none of them touched: connect,
    # ConnectionClosed, protocol.State.CLOSED. The one behavioural change that
    # reaches any client is "connections are garbage collected immediately once
    # closed", which cannot break a feed that reconnects on ConnectionClosed.
    # Same operator decision as before: MARKET_DATA_SOURCE=WEBSOCKET is active
    # but every strategy is PAPER, so a bad feed costs a session, not money.
    # If the feed does not tick at 09:15, revert this pin first.
    assert "websockets==17.1" in core
    # 2.3.3.260113 -> 3.0.5.260730 (2026-09-01, PR #150). A correction more than
    # an upgrade. `pandas` is pinned at 3.0.5 and is NOT in the Dependabot ignore
    # list, so it moved to the 3.x line while a semver-major ignore pinned the
    # stubs to the 2.3.3 line -- the mypy gate has been typing pandas 3 with
    # pandas 2 stubs ever since. pandas-stubs tracks the pandas release it
    # describes, so matching majors IS the point of this pin rather than a risk
    # to it, and the ignore has been retired accordingly. Unlike the transport
    # pins above this one was verified BEFORE the merge, because a stubs change
    # is typing-only and cannot reach the trading path: mypy reports no issues in
    # 54 source files against 3.0.5.260730. That is also the version the
    # operator's box already had installed while this file still asked for
    # 2.3.3, so `pip install -r requirements.txt` would have DOWNGRADED it.
    assert "pandas-stubs==3.0.5.260730" in core
    # Same reasoning for the agent transport: SL_HUNTING_ENABLED=true, so this
    # is an active path, but paper-only until the next session confirms it.
    #
    # 0.2.132 -> 0.2.137 (2026-08-17, PR #125). Upstream flags a BREAKING change
    # in 0.2.137: "the `Message` union was widened; exhaustive matching code
    # needs updates" (a new `ConversationResetMessage`, plus an `origin` field on
    # `UserMessage`/`ResultMessage` and two resume options). We are not exhaustive
    # matchers -- `sl_hunting_agent._run_query` and `sl_hunting_coach` both walk
    # the stream with an `isinstance(ResultMessage) / elif isinstance(
    # AssistantMessage)` chain and no else-raise, so an unrecognised member is
    # ignored rather than fatal. That is what makes this bump safe to take.
    # 0.2.133-0.2.136 are bundled-CLI bumps only (2.1.224 -> 2.1.228).
    #
    # 0.2.137 -> 0.2.143 (2026-08-24, PR #135). Five of the six releases are
    # bundled-CLI bumps only (2.1.232 -> 2.1.238). The one with substance is
    # 0.2.140, reviewed in three parts:
    #   * It adds `ConversationResetMessage`, breaking EXHAUSTIVE matchers. We
    #     are not one -- `_run_query` and `sl_hunting_coach` walk the stream with
    #     an isinstance chain and no else-raise, so an unknown member is ignored.
    #     Same reasoning that cleared 0.2.137.
    #   * It WIDENS its MCP requirement to mcp>=1.23.0,<3.0.0. That is a
    #     relaxation, and our shared pin (mcp==1.29.0, used by BOTH agents) sits
    #     inside it. Nothing to do; do not chase MCP 2.x on this alone.
    #   * It adds a `ResultError` exception. It cannot escape: `_run_query` ends
    #     in a broad `except Exception` that converts any SDK failure into a
    #     structured error, and the worker turns that into a fail-soft HOLD.
    # Unchanged from before: CI never spawns the bundled CLI, so confirm on the
    # next PAPER session that decisions still return ("SLHuntingAgent decision
    # cost ~$..." in the log). If they stop, revert this pin first.
    # 0.2.143 -> 0.2.145 (2026-09-01, PR #144). The narrowest SDK bump so far:
    # both 0.2.144 and 0.2.145 are CLI-bundle-only releases with NO user-facing
    # Python changes — query(), ClaudeAgentOptions, the in-process MCP server and
    # tool definitions are all untouched, so none of the reasoning above needs
    # revisiting. What DID change is the bundled Claude CLI (2.1.245 -> 2.1.247),
    # and that is the part actually running the agent. CI never spawns it, so
    # this build proves nothing about it. Confirm on the next PAPER session that
    # decisions still return ("SLHuntingAgent decision cost ~$..." in the log,
    # alongside the SLH-013 latency line). If they stop, revert this pin first.
    # 0.2.145 -> 0.2.148 (2026-09-02, PR #151). Three releases, all bundled-CLI
    # only (2.1.247 -> 2.1.251) with no user-facing Python changes, so none of
    # the reasoning above needs revisiting. Same standing check as before: CI
    # never spawns the bundled CLI, so confirm on the next PAPER session that
    # decisions still return ("SLHuntingAgent decision cost ~$..." in the log).
    assert "claude-agent-sdk==0.2.148" in ai
    # 2.13.4 -> 2.13.5 (2026-09-02, PR #151). Patch. Still inside every window
    # that matters: mcp 1.29.1 wants pydantic>=2.11.0,<3.0.0 and openai-codex
    # wants >=2.12, and the strict models both agents rely on are unaffected.
    assert "pydantic==2.13.5" in ai
    assert all("==" in line for line in ai)
    # The independent CPR Codex agent is an optional, subscription-authenticated
    # runtime and now shares this file. Both AI agents run inside the SAME
    # master process, so Python can only install one version of what they share
    # -- which is exactly why the two sets were merged (PR #125). Keeping them
    # apart meant `mcp` and `pydantic` were pinned twice and had to be kept
    # equal by hand.
    # 0.144.4 -> 0.147.0 (2026-08-24, PR #135). THE LEAST VERIFIABLE PIN IN THIS
    # FILE, and the reason is structural rather than a judgement about the
    # release: NOTHING in the automated gates can detect a break in it.
    #   * `Tests/Signal Generators/CPR AI Agent/test_cpr_ai_runtime.py`
    #     monkeypatches a FAKE `openai_codex` into sys.modules, so the runtime
    #     tests exercise our code against a stub, never the SDK.
    #   * mypy has ignore_missing_imports for `openai_codex.*` (pyproject).
    #   * CI never runs the authenticated smoke command, by design.
    #   * The import is lazy, inside `_run_isolated_turn`, so even importing the
    #     module is not attempted until a real turn runs.
    # What was reviewed by hand instead: our entire surface is three names and
    # one call shape -- `Codex()`, `thread_start(model/config/cwd/
    # developer_instructions/ephemeral/sandbox/approval_mode)`, `thread.run(
    # prompt/approval_mode/output_schema/effort)`, and the result fields
    # `final_response`, `items`, `usage`. 0.147.0 still declares only
    # pydantic>=2.12 (our 2.13.4 satisfies it) and pins its own CLI binary.
    # A three-minor jump on a young SDK could still rename a kwarg silently.
    # CPR_AI_ENABLED is true and the worker trades paper daily, so a break
    # surfaces as CPR AI errors in the next session -- that session, not this
    # build, is the actual test.
    assert "openai-codex==0.147.0" in ai
    # 1.29.0 -> 1.29.1 (2026-09-01, PR #144). A patch, and it stays inside the
    # window BOTH agents require — claude-agent-sdk declares mcp>=1.23.0,<3.0.0
    # and openai-codex is satisfied too, so the shared single-version constraint
    # that forced these into one file still holds. Nothing to validate beyond a
    # clean resolve, which CI does perform.
    # 1.29.1 -> 2.1.1 (2026-09-02, PR #151). A MAJOR, taken together with the
    # port it required, in the same commit. Held first in this same PR, then
    # ported once the v2 surface had been read rather than assumed.
    # WHAT ACTUALLY BROKE: 2.x deletes `mcp.server.fastmcp` and renames FastMCP
    # to MCPServer, so cpr_ai_mcp_server failed at import and took
    # test_cpr_ai_context and test_cpr_ai_runtime with it -- the exact two tests
    # cited in PR #150 as the coverage that made retiring the mcp-major ignore
    # safe. They caught it the first week it could reach a PR.
    # WHAT DID NOT CHANGE, verified against 2.1.1 in an isolated venv rather
    # than inferred from the migration guide: the `@server.tool(name=,
    # description=)` decorator signature, `run(transport="stdio")`, the
    # constructor's `log_level`, and the tool objects the tests read
    # (`_tool_manager._tools`, `get_tool`, `.parameters["properties"] == {}`,
    # `.fn()`). So the port is the import and the class name, and the four
    # frozen no-argument tools are registered exactly as before.
    # THE OTHER AGENT WAS CHECKED TOO, because one process can hold only one
    # mcp: claude-agent-sdk imports the LOW-LEVEL `mcp.server.Server` plus
    # mcp.shared.memory / mcp.shared.message / mcp.types, and every one of those
    # paths still resolves on 2.1.1, so SL Hunting is unaffected.
    assert "mcp==2.1.1" in ai
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


def test_precommit_ruff_rev_matches_the_requirements_dev_pin():
    """The commit hook must enforce the SAME ruff ruleset as the CI gate.

    These are pinned in two unrelated files -- `.pre-commit-config.yaml` carries
    a git TAG (`v0.16.3`) and `requirements.txt` carries a PyPI version
    (`ruff==0.16.3`) -- and nothing but a comment kept them together. They
    drifted to v0.15.1 against a 0.16.2 pin, which is worse than having no hook
    at all: ruff gains and changes rules every minor release, so a commit could
    pass locally and still fail CI's `Run Ruff static checks` on a rule the hook
    had never heard of. Dependabot bumps the requirement and cannot see the
    hook, so this only stays fixed if a test says so.
    """
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hook_revs = {
        repo["repo"].rstrip("/").rsplit("/", maxsplit=1)[-1]: repo["rev"]
        for repo in config["repos"]
    }

    pinned = [
        line for line in _requirement_lines("requirements.txt")
        if line.startswith("ruff==")
    ]
    assert len(pinned) == 1, f"expected exactly one ruff pin, found {pinned}"
    required_version = pinned[0].split("==", maxsplit=1)[1]

    assert hook_revs["ruff-pre-commit"] == f"v{required_version}", (
        f".pre-commit-config.yaml pins ruff-pre-commit at "
        f"{hook_revs['ruff-pre-commit']} but requirements.txt pins "
        f"ruff=={required_version}. Bump BOTH together so the commit hook "
        f"enforces the same ruleset as CI."
    )


def test_core_requirements_carry_both_the_runtime_and_the_dev_toolchain():
    """requirements-dev.txt was merged into requirements.txt (PR #125).

    Guards the merge in both directions: the runtime pins DEPS-001 protects
    must still be there, and the dev toolchain must not quietly drift back out
    into a second file that CI would then stop installing.
    """
    core = _requirement_lines("requirements.txt")

    assert not (ROOT / "requirements-dev.txt").exists(), (
        "requirements-dev.txt is back. If splitting it out again is deliberate, "
        "update the CI install step, the README gate block, and this test "
        "together -- CI installs requirements.txt only."
    )
    # numpy joined this list on 2026-09-01 (PR #150) when the MINOR half of its
    # Dependabot ignore was retired. Until then nothing asserted the numpy pin a
    # second time, so a bump could have gone green with no human reading it --
    # the same hole that was open on pandas-stubs. MAT-106's determinism
    # snapshot verifies numpy's BEHAVIOUR to 8 decimal places, which is the
    # stronger check; this assertion is the separate guarantee that the VERSION
    # cannot move without someone editing this line. Majors remain ignored:
    # TA-Lib ships a compiled extension built against the numpy 2.x C ABI and
    # declares a bare `numpy` with no ceiling, so nothing in the metadata would
    # stop pip resolving numpy 3.x against a wheel that cannot survive it.
    # numpy 2.4.6 -> 2.5.2 (2026-09-02, PR #151) -- the FIRST bump admitted by
    # retiring the minor half of numpy's Dependabot ignore, and the evidence that
    # retirement rested on held: MAT-106's determinism snapshot pins EMA, ATR,
    # ADX, SMA, Supertrend, MACD and Renko to 8 decimal places, and it passed
    # UNCHANGED on 2.5.2 on both 3.12 and 3.13. Nothing in the indicator pipeline
    # moved. Majors remain ignored (TA-Lib's compiled extension, no numpy ceiling
    # declared anywhere).
    for runtime_pin in ("dhanhq==2.2.0", "pandas==3.0.5", "TA-Lib==0.6.8", "numpy==2.5.2"):
        assert runtime_pin in core
    # mypy 1.20.2 -> 2.3.1 (2026-09-02, PR #151): a MAJOR, and the first one
    # admitted by retiring that ignore. It is exactly the case the retirement
    # argued for -- mypy cannot reach the trading path, so a bad major can only
    # turn the build red. Note the gate ORDER when reading this PR's history: the
    # test step runs BEFORE mypy, so the run that failed on mcp never reached
    # mypy at all and proved nothing about it. The green run on this commit is
    # what actually type-checks 2.3.1, on both 3.12 and 3.13.
    for tool_pin in ("pytest==9.1.1", "mypy==2.3.1", "bandit==1.9.4", "pip-audit==2.10.1"):
        assert tool_pin in core
    assert all("==" in line for line in core)


def test_every_precommit_hook_rev_is_a_pinned_tag():
    """No floating refs: a hook that tracks a branch is not a reproducible gate."""
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))

    for repo in config["repos"]:
        rev = repo["rev"]
        assert re.fullmatch(r"v?\d+\.\d+\.\d+", rev), (
            f"{repo['repo']} is pinned to {rev!r}, which is not an exact "
            f"version tag. Floating refs make the hook unreproducible."
        )


def test_ci_runs_audit_branch_coverage_and_every_exact_dependency_set():
    workflow = (ROOT / ".github/workflows/quality-and-security.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    core_job = workflow.split("\n  broker-dependencies:", maxsplit=1)[0]

    assert set(parsed["jobs"]) == {"verify", "broker-dependencies"}
    assert "requirements-ai.txt" in workflow
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
    # 68.0 -> 70.0 (2026-09-02). CI measured 70.2% on five consecutive main
    # runs, so the floor was raised into the headroom it had accumulated. The
    # remaining margin is ~0.2pp by design; per pyproject's own rule this only
    # ever moves UP.
    assert config["tool"]["coverage"]["report"]["fail_under"] == 70.0


def test_mypy_covers_the_complete_identifier_named_cpr_ai_runtime():
    """A new importable CPR module must not silently fall outside mypy."""

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = config["tool"]["mypy"]
    agent_dir = ROOT / "Signal Generators/CPR AI Agent"
    runtime_files = {
        path.relative_to(ROOT).as_posix()
        for path in agent_dir.glob("cpr_ai_*.py")
        if path.is_file()
    }
    configured_files = {
        path
        for path in mypy["files"]
        if path.startswith("Signal Generators/CPR AI Agent/cpr_ai_")
    }

    assert configured_files == runtime_files
    assert "Signal Generators/CPR AI Agent" in mypy["mypy_path"]
    # The master is STILL outside mypy, but the reason changed on 2026-09-02.
    # It used to be structural -- a spaced filename cannot be a mypy module.
    # ADR-0014 tier 2a renamed it, so the exclusion is now purely a backlog:
    # mypy reports 288 errors on it (193 attr-defined). This assertion is
    # therefore TEMPORARY and must be INVERTED, not deleted, once tier 2b has
    # worked those down -- at which point the master joins `files` and this
    # line should assert its PRESENCE.
    assert "nifty_multi_strategy_master.py" not in mypy["files"]
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
    assert "inherits no environment variables" not in focused_readme
    assert "strict allowlist" in focused_readme
    assert "trading and api secrets" in focused_readme.lower()

    master = (ROOT / "nifty_multi_strategy_master.py").read_text(
        encoding="utf-8"
    )
    assert "optional CPR arbiter" not in master


def test_current_architecture_docs_distinguish_core_from_optional_agents():
    """A 27-core description must not masquerade as the enabled total."""

    architecture_files = (
        ROOT / "README.md",
        ROOT / "Signal Generators/Readme.md",
        ROOT / "AGENTS.md",
        ROOT / "CLAUDE.md",
        ROOT / "nifty_multi_strategy_master.py",
        # The committed HLD is a whole-system overview, so the same rule applies:
        # a reader must be able to see both optional agents and must not mistake
        # the core count for the enabled total (docs/adr/0011 follow-up).
        ROOT / "docs/hld/system-overview.md",
    )
    failures: list[str] = []
    for path in architecture_files:
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        # Both agents are outside the core roster and can be independently
        # omitted. Every current architecture overview must make both visible.
        if "sl hunting" not in lower or "cpr codex ai" not in lower:
            failures.append(
                f"{path.relative_to(ROOT).as_posix()}: optional agents are incomplete"
            )

        for line_number, line in enumerate(text.splitlines(), start=1):
            normalized = line.lower().replace("twenty-seven", "27")
            has_27_roster_claim = re.search(
                r"(?<!\d)(?:~|approximately\s+)?27(?!\d)", normalized
            ) and re.search(
                r"\b(?:strategyworker|workers?|consumers?|strateg(?:y|ies))\b",
                normalized,
            )
            # Regime Adaptive legitimately makes the core approximately 27,
            # but two optional agents mean 27 can no longer describe the
            # complete configured or running worker total.
            if has_27_roster_claim and "core" not in normalized:
                failures.append(
                    f"{path.relative_to(ROOT).as_posix()}:{line_number}: {line.strip()}"
                )

        if re.search(
            r"\b(?:26|twenty[- ]six)\s+(?:strategyworker|workers?|consumers?|strateg(?:y|ies))\b",
            lower,
        ):
            failures.append(f"{path.relative_to(ROOT).as_posix()}: stale 26-worker roster")
        if re.search(r"\b27\s+(?:workers?|strateg(?:y|ies))\s+total\b", lower):
            failures.append(f"{path.relative_to(ROOT).as_posix()}: stale 27-worker total")

    assert not failures, "ambiguous current worker-roster claims:\n" + "\n".join(failures)
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "see the latest addition at the top of this list for the current total" not in root_readme


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


def _committed_design_documents() -> list[Path]:
    """Every tracked design document under docs/, excluding the scratchpad.

    ``docs/superpowers/`` is gitignored session working material, not product
    documentation, so it is deliberately outside every gate here (docs/adr/0011).
    """

    return sorted(
        path
        for folder in ("adr", "lld")
        for path in (ROOT / "docs" / folder).glob("*.md")
        if path.is_file()
    )


def test_every_committed_design_document_is_linked_from_the_docs_index():
    """A new ADR or LLD must reach the index, and the index must not rot.

    ``docs/README.md`` is the only navigation surface for the committed
    architecture set. A document that never gets linked is invisible -- it will
    not be read, will not be maintained, and will quietly go stale. The reverse
    is just as bad: a link left behind by a renamed or deleted file sends a
    reader to a 404 and makes the whole index untrustworthy.

    Checked in BOTH directions for that reason.
    """

    index_path = ROOT / "docs/README.md"
    index = index_path.read_text(encoding="utf-8")

    linked = {
        match.group(1)
        for match in re.finditer(r"\((?:\./)?((?:adr|lld)/[^)#]+\.md)[^)]*\)", index)
    }
    on_disk = {
        path.relative_to(ROOT / "docs").as_posix() for path in _committed_design_documents()
    }

    unlinked = sorted(on_disk - linked)
    assert not unlinked, (
        "these design documents exist but are not linked from docs/README.md: "
        + ", ".join(unlinked)
    )

    dangling = sorted(linked - on_disk)
    assert not dangling, (
        "docs/README.md links these documents, which do not exist: " + ", ".join(dangling)
    )


def test_relative_links_inside_the_committed_docs_resolve():
    """No broken cross-reference anywhere in the committed docs set.

    The HLD, the LLDs and the ADRs reference each other and the source tree
    constantly. Renaming a file is the normal way those break, and a broken link
    is invisible until somebody follows it -- so it is checked mechanically
    rather than by review.

    Only relative targets are resolved. External URLs are not fetched: this
    suite must stay network-free.
    """

    docs_root = ROOT / "docs"
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    broken: list[str] = []

    for document in sorted(docs_root.rglob("*.md")):
        # The Superpowers scratchpad is gitignored and may reference paths that
        # only existed during one session.
        if "superpowers" in document.parts:
            continue
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for target in link_pattern.findall(line):
                target = target.strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                # Strip any "#section" anchor, then undo the %20 escaping the
                # spaced-name folders need in markdown links.
                relative = urllib.parse.unquote(target.split("#", maxsplit=1)[0])
                if not relative:
                    continue
                if not (document.parent / relative).resolve().exists():
                    broken.append(
                        f"{document.relative_to(ROOT).as_posix()}:{line_number} -> {target}"
                    )

    assert not broken, "broken relative links in the committed docs:\n" + "\n".join(broken)


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


def _test_function_bodies(source: str) -> dict[str, str]:
    """Split a test module into ``{function name: body}``.

    Needed because a file-wide substring count proves nothing here: this module
    has several tests that use a logger, so "at least N logger calls exist"
    stays true even after the one line that matters has been deleted. The
    assertion has to be made INSIDE the function that owns the line.
    """
    bodies: dict[str, str] = {}
    for match in re.finditer(r"^def (test_\w+)\(", source, re.MULTILINE):
        name = match.group(1)
        nxt = re.search(r"^def test_\w+\(", source[match.end() :], re.MULTILINE)
        end = match.end() + (nxt.start() if nxt else len(source))
        bodies[name] = source[match.start() : end]
    return bodies


def test_secret_redaction_canary_coverage_survives_codeql_pressure():
    """The canary-logging redaction tests must not be "fixed" away (ADR-0013).

    CodeQL flags three lines of ``Tests/Dependencies/test_secret_redaction.py``
    as ``py/clear-text-logging-sensitive-data``. That is EXPECTED and the alerts
    are dismissed: each line logs a fake CANARY secret through a handler that
    has ``RedactingFilter`` installed, into an in-memory buffer, and the test
    then asserts the secret never reached the output.

    The logging call is not incidental to those tests -- it IS the test. It is
    the only way to prove the filter scrubs a secret before it reaches a
    handler, and that filter exists because dhanhq's marketfeed puts a live
    access token in its websocket URL, so a connect error would otherwise write
    a working credential into a log operators routinely share.

    So the obvious way to quiet the scanner -- delete the logging call -- would
    remove the protection while leaving a green Security tab. This guard makes
    that edit fail loudly instead. If you are here because CodeQL complained
    again, re-dismiss the alert and read ADR-0013; do not patch the test.
    """
    source = (ROOT / "Tests/Dependencies/test_secret_redaction.py").read_text(
        encoding="utf-8"
    )
    bodies = _test_function_bodies(source)

    # The flagged lines, by the test that owns each and the FILTER PATH it
    # covers. Checked per path, not per test: the first test carries two of the
    # three alerts, so "this test still logs something" stays true after one of
    # them is deleted. CLAUDE.md names both paths -- "lazy %s args and exception
    # tracebacks included" -- and losing either is a real hole.
    required_paths = {
        "test_debug_and_exception_records_never_emit_canary_secrets": (
            (r"logger\.debug\([^)]*secret", "a lazy %s/dict argument"),
            (r"logger\.exception\([^)]*secret", "an exception traceback"),
        ),
        "test_install_redaction_filter_covers_logger_and_existing_handlers": (
            (r"logger\.debug\([^)]*secret", "a logger installed via install_redaction_filter"),
        ),
    }

    for name, paths in required_paths.items():
        body = bodies.get(name)
        assert body, (
            f"{name} has been removed or renamed. It is one of only two tests "
            "proving the redaction filter scrubs a secret before it reaches a "
            "handler -- see ADR-0013 before changing it."
        )
        for pattern, path_description in paths:
            assert re.search(pattern, body), (
                f"{name} no longer pushes the canary secret through {path_description}. "
                "That is exactly the CodeQL-silencing edit ADR-0013 rejects: the "
                "alert disappears and so does the coverage."
            )
        assert "assert secret not in output" in body, (
            f"{name} no longer asserts the secret is absent from the captured "
            "output. Logging a canary proves nothing without that check."
        )

    # The non-logging redaction test is not under CodeQL pressure, but it is the
    # only coverage for nested containers, so keep it from drifting away too.
    assert "test_redaction_reaches_secrets_inside_tuples_and_sets" in bodies
    assert "CANARY" in source, "the canary marker strings have been removed"
