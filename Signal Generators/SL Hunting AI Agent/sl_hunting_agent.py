"""The SL Hunting AI Agent — a Claude-Agent-SDK trader for NIFTY index options.

What this is (beginner note)
----------------------------
This is the "brain". On each completed bar it is handed the recent NIFTY candles
and a `TradeExecutor`, and it runs ONE agentic pass: Claude calls the read-only
context tools (pivot/levels, fibo, candle patterns, structure, position state),
decides whether a confirmed SL-Hunting setup is present, and — if so — calls its
single order tool to act. Its final message is a strict `SLHuntingDecision` JSON
object recording what it did (for logging/provenance).

It mirrors the house pattern in
`../Streamlit Scanner App/backend/technical/technical_agent.py`:
- the Claude Agent SDK is imported LAZILY inside the runner, so this module imports
  cleanly even when `claude-agent-sdk` is not installed (tests inject a fake runner);
- the agentic loop is `query(prompt, options=ClaudeAgentOptions(...))`;
- a Windows-safe async→sync bridge runs it from threaded/sync code;
- `runner=` is an injectable seam so unit tests never spawn the Claude CLI;
- output is a strict Pydantic model validated with `parse_with_retry`.

Subscription billing note: keep ``ANTHROPIC_API_KEY`` UNSET so the SDK draws on your
Claude plan's Agent SDK credit instead of per-token API billing (same as the
Scanner app). The agent NEVER raises into the trading loop — any failure returns a
safe HOLD decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import Field, ValidationError, field_validator, model_validator
from sl_hunting_ai_validation import StrictAIModel
from sl_hunting_executor import TradeExecutor
from sl_hunting_indicators import SLHuntingIndicatorConfig, prepare_candles
from sl_hunting_knowledge import FINAL_OUTPUT_INSTRUCTION, build_system_prompt
from sl_hunting_tools import SLHuntingToolContext, build_sl_hunting_mcp_server

logger = logging.getLogger(__name__)

# How many recent candles to show the model for orientation (it gets precise facts
# from the tools; this is just context). Keep small to bound tokens/cost.
_SNAPSHOT_BARS = 40
SL_HUNTING_PROMPT_VERSION = "sl-hunting-v1"


@dataclass
class AgentRunResult:
    """One agentic loop's final text plus its (optional) reported cost."""

    text: str
    cost_usd: float | None = None


class SLHuntingAgentError(RuntimeError):
    """Infrastructure failure (SDK missing, CLI error, bad final output)."""


class SLHuntingUsageLimitError(SLHuntingAgentError):
    """The Claude subscription's Agent SDK usage/rate limit was hit (HTTP 429)."""


class SLHuntingTimeoutError(SLHuntingAgentError):
    """The SDK call exceeded its time budget and was abandoned (SLH-001).

    decide() runs on the strategy worker's OWN thread; while it blocks, the
    worker's per-poll stop/target check, max-loss and 15:15 square-off are all
    frozen. Bounding the call means a hung Claude CLI costs one bar's decision
    (a fail-soft HOLD), not the position's entire mechanical safety net.
    """

    def __init__(self, timeout_seconds: float, thread: threading.Thread | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        # The still-running worker thread we abandoned. decide() keeps a reference
        # so it can gate new calls until this one finishes -- otherwise a CLI that
        # hangs every bar would pile up daemon threads + subprocesses.
        self.thread = thread
        super().__init__(
            f"Claude Agent SDK call exceeded {self.timeout_seconds:.0f}s and was "
            "abandoned; this bar's order tool has been disarmed."
        )


class SLHuntingAuthError(SLHuntingAgentError):
    """The bundled Claude CLI could not authenticate to Anthropic.

    The Claude SUBSCRIPTION token is missing or expired in the runner's environment.
    A headless runner (unlike a Claude Code/IDE session) has nothing to refresh an
    expired OAuth login, so the spawned CLI fails on every call — either as an HTTP
    401/403 from the API, or (SLH-004) as a LOCAL failure with no HTTP status at all
    when the CLI finds its stored token expired and unrefreshable. Fix by running
    `claude setup-token` (preferred for an unattended runner) or `claude login` in the
    terminal that launches the runner, with ANTHROPIC_API_KEY UNSET, then restart.
    """

    def __init__(self, status: int | None = None) -> None:
        # None = the CLI reported the failure locally (expired/unrefreshable stored
        # OAuth token) with no HTTP status attached.
        self.status = int(status) if status is not None else None
        cause = f"HTTP {self.status}" if self.status is not None else "expired local OAuth token"
        super().__init__(
            f"Claude subscription authentication failed ({cause}). The "
            "spawned `claude` CLI has no valid token in the runner's environment. Run "
            "`claude setup-token` (or `claude login`) in that terminal with "
            "ANTHROPIC_API_KEY unset, then restart the runner."
        )


def _raise_for_error_result(result_message: Any) -> None:
    """Classify a failed-API-call ResultMessage into a typed, actionable error.

    The SDK sets ``is_error=True`` with ``subtype="success"`` and an HTTP
    ``api_error_status`` when the CLI's call to Anthropic failed (this is exactly what
    surfaces otherwise as the opaque "Claude Code returned an error result: success").
    Map the status to a specific error so the log explains itself. Only the status
    CODE is used — it carries no model/message content, so it is safe to log. No-op
    when the result is absent or not an error.
    """
    if result_message is None or not getattr(result_message, "is_error", False):
        return
    status = getattr(result_message, "api_error_status", None)
    if status in (401, 403):
        raise SLHuntingAuthError(status)
    if status == 429:
        raise SLHuntingUsageLimitError()
    if status is None:
        # SLH-004: the CLI ALSO uses this error-result shape for LOCAL failures
        # carrying no HTTP status. The canonical one is an expired OAuth token it
        # cannot refresh headlessly (seen live 2026-07-13; result text "Failed to
        # authenticate: OAuth session expired and could not be refreshed").
        # Classify by the result text, but raise typed errors whose messages are
        # CANNED — nothing from the result body reaches the logs.
        text = str(getattr(result_message, "result", "") or "")
        if _mentions_usage_limit(text):
            raise SLHuntingUsageLimitError()
        if re.search(r"authenticat|oauth|log ?in|api key", text, re.IGNORECASE):
            raise SLHuntingAuthError()
    detail = f"HTTP {status}" if status else "no api_error_status"
    raise SLHuntingAgentError(f"Claude Agent SDK returned an error result ({detail}).")


# A runner drives one full agentic loop and returns the final message text.
# Production uses `SLHuntingAgent._default_run`; tests inject a fake.
RunnerFn = Callable[..., Awaitable[AgentRunResult]]


# ---------------------------------------------------------------------------
# Strict output schema
# ---------------------------------------------------------------------------

Action = Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"]
# Which leg an EXIT applies to. The NIFTY leg and its mechanical BankNIFTY mirror
# are TIED for hard risk (stop/target/max-loss/square-off close both), but the
# agent may cut them INDEPENDENTLY on premise-invalidation via this selector.
ExitLeg = Literal["NIFTY", "BNF", "BOTH"]


class SLHuntingDecision(StrictAIModel):
    """The agent's structured decision for one bar.

    `confidence` is validated with a `@field_validator` (not `Field(ge=, le=)`) so
    the JSON schema we describe to Claude stays free of `minimum`/`maximum`, which
    Claude rejects on integer types — same trick as the Scanner's verdict models.
    """

    action: Action = Field(description="ENTER_LONG, ENTER_SHORT, EXIT or HOLD.")
    stop: float = Field(default=0.0, description="Underlying stop level for an entry; 0 for EXIT/HOLD.")
    target: float = Field(default=0.0, description="Underlying target level for an entry; 0 for EXIT/HOLD.")
    exit_leg: ExitLeg = Field(
        default="BOTH",
        description="For EXIT only: which basket leg to close — NIFTY, BNF, or BOTH (default).",
    )
    confidence: int = Field(description="0-10 confidence in the decision (10 = textbook).")
    setup: str = Field(default="none", description="Short name of the setup acted on, or 'none'.")
    reasoning: str = Field(description="2-4 sentences: level, pattern+confirmation, stop/target, why now.")
    model_used: str = Field(default="", description="Which model produced this decision.")

    @field_validator("confidence")
    @classmethod
    def _validate_confidence_range(cls, value: int) -> int:
        if not 0 <= value <= 10:
            raise ValueError(f"confidence must be between 0 and 10 inclusive, got {value}")
        return value

    @field_validator("stop", "target")
    @classmethod
    def _validate_level_sanity(cls, value: float) -> float:
        """Reject hallucinated stop/target garbage (SLH-002).

        These are UNDERLYING price levels the mechanical per-poll exit uses
        verbatim -- a negative or absurd stop silently disables the stop for
        the whole trade. 0.0 stays valid as the documented EXIT/HOLD
        placeholder. Bounds live here (not `Field(ge=/le=)`) for the same
        reason as `confidence`: Claude rejects `minimum`/`maximum` keys in
        the JSON schema we describe to it. Price-aware checks (side, distance
        band) happen in the order tool, which knows the current price.
        """
        if not math.isfinite(value):
            raise ValueError(f"stop/target must be a finite number, got {value!r}")
        if value < 0:
            raise ValueError(f"stop/target must be >= 0, got {value}")
        if value > 10_000_000:
            raise ValueError(f"stop/target is implausibly large for an index level: {value}")
        return value

    @model_validator(mode="after")
    def _require_levels_for_entries(self) -> SLHuntingDecision:
        """An ENTER decision must carry a positive stop AND target (SLH-002 / Codex).

        The order tool already rejects zero/omitted levels when the agent CALLS
        it, but the final JSON is a separate self-report: without this, an
        ENTER_LONG/ENTER_SHORT with the 0.0 defaults would still validate and be
        recorded as a real entry with no levels, corrupting the decision journal
        and defeating the tool-side guard. EXIT/HOLD keep their 0.0 placeholders.
        """
        if self.action in ("ENTER_LONG", "ENTER_SHORT") and not (self.stop > 0 and self.target > 0):
            raise ValueError(
                f"{self.action} requires a positive stop and target, got "
                f"stop={self.stop}, target={self.target}"
            )
        return self


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the SLHuntingDecision JSON object out of the model's final message.

    Tolerant of a stray ```json fence or a leading sentence: looks for a fenced
    block first, then falls back to the outermost {...} span. Returns None when
    nothing parses (mirrors the Scanner's extractor).
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _mentions_usage_limit(*texts: str | None) -> bool:
    """True if any error text looks like a subscription usage/rate limit (vs a real bug).

    Lets us map those specific failures to `SLHuntingUsageLimitError` so the caller can
    treat "out of plan credit" differently from a genuine crash.
    """
    blob = " ".join(t for t in texts if t).lower()
    return any(k in blob for k in ("usage limit", "rate limit", "429", "quota"))


def _disabled_thinking_config(sdk_module: Any) -> dict[str, Any] | None:
    """Build the SDK ``thinking`` option that DISABLES extended thinking (fast mode).

    Returns ``{"type": "disabled"}`` (via the SDK's ``ThinkingConfigDisabled`` type),
    or ``None`` when the installed SDK is too old to expose that type (the caller then
    warns and falls back to the default thinking).

    Why this helper exists: ``ThinkingConfigDisabled`` is a *TypedDict*, so a bare
    ``ThinkingConfigDisabled()`` evaluates to an empty ``{}`` with NO ``"type"`` key.
    The SDK's ``_build_command`` reads ``thinking["type"]`` unconditionally, so that
    empty dict makes it raise ``KeyError: 'type'`` before the CLI is ever launched --
    i.e. `SL_HUNTING_FAST_MODE=true` crashed the agent. The fix is to build the dict
    WITH its required ``type="disabled"`` field.
    """
    thinking_disabled = getattr(sdk_module, "ThinkingConfigDisabled", None)
    if thinking_disabled is None:
        return None
    return thinking_disabled(type="disabled")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SLHuntingAgent:
    """Per-bar SL-Hunting agent backed by the Claude Agent SDK + Claude subscription.

    One instance is reused across bars. The agentic loop is driven by an injectable
    `runner` so unit tests avoid spawning the CLI; production uses `_default_run`,
    which lazily imports `claude_agent_sdk`.
    """

    MAX_TURNS = 12  # up to 5 context tools + 1 order tool + reasoning + final JSON

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerFn | None = None,
        fast_mode: bool = False,
        indicator_config: SLHuntingIndicatorConfig | None = None,
        lessons_block: str = "",
        sdk_timeout_seconds: float = 90.0,
    ) -> None:
        if not model:
            raise ValueError("SLHuntingAgent: model is required.")
        self._model = model
        self._runner = runner
        self._fast_mode = bool(fast_mode)
        self._cfg = indicator_config or SLHuntingIndicatorConfig()
        # SLH-001: hard budget for one SDK call. decide() blocks the worker
        # thread that also enforces stop/target/max-loss/square-off, so a hung
        # CLI call must cost one bar (fail-soft HOLD), not the safety net.
        # <= 0 disables the bound (not recommended for the live runner).
        self._sdk_timeout_seconds: float | None = (
            float(sdk_timeout_seconds) if float(sdk_timeout_seconds) > 0 else None
        )
        # SLH-001 / Codex: a worker thread from a PRIOR timed-out call that is
        # still alive. While set, decide() gates new calls (no fresh thread or
        # subprocess) so a persistently hung CLI can accumulate at most one.
        self._abandoned_call: threading.Thread | None = None
        # Optional v3 LEARNED LESSONS block (loaded + formatted by the caller, gated by
        # SL_HUNTING_LESSONS_ENABLED). Injected ONCE here, before the output contract,
        # so the system prefix stays stable per session and prompt caching is preserved.
        learned = ("\n\n" + lessons_block.strip()) if lessons_block and lessons_block.strip() else ""
        self._system_prompt = build_system_prompt() + learned + FINAL_OUTPUT_INSTRUCTION
        # SLH-003: cache for the system-prompt temp FILE handed to the SDK by
        # `_default_run` (written once per unique text, reused every bar) — see
        # `_system_prompt_as_file` for why a string system prompt cannot be used.
        self._system_prompt_file_path: str | None = None
        self._system_prompt_file_text: str | None = None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _snapshot_csv(self, candles: pd.DataFrame) -> str:
        """Render the last ``_SNAPSHOT_BARS`` candles as a tiny CSV for the prompt.

        This is ORIENTATION only (a glance at recent price) — the agent gets exact
        numbers from its tools, so we keep this short to bound tokens/cost.
        """
        recent = candles.tail(_SNAPSHOT_BARS)
        lines = ["time,open,high,low,close"]
        for row in recent.itertuples(index=False):
            ts = str(getattr(row, "timestamp", ""))[:19]
            lines.append(f"{ts},{float(row.open):.2f},{float(row.high):.2f},{float(row.low):.2f},{float(row.close):.2f}")
        return "\n".join(lines)

    def _build_user_prompt(self, candles: pd.DataFrame, position: dict[str, Any]) -> str:
        """Compose the per-bar USER message: instrument, current price, position, candles, task."""
        last_price = float(candles["close"].iloc[-1]) if not candles.empty else 0.0
        pos_line = (
            f"You currently HOLD a {position.get('direction')} position "
            f"(entry {position.get('entry')}, stop {position.get('stop')}, target {position.get('target')})."
            if position.get("in_position")
            else "You are currently FLAT (no open position)."
        )
        return (
            f"Instrument: NIFTY index (options). Current underlying price: {last_price:.2f}.\n"
            f"{pos_line}\n\n"
            f"Recent candles (CSV, for orientation only — get precise facts from your tools):\n"
            f"{self._snapshot_csv(candles)}\n\n"
            "Call your tools (pivot_and_levels, candle_patterns, fibo_levels, "
            "market_structure, position_state) to gather the facts, apply the "
            "SL-Hunting method, and decide. If — and only if — a confirmed setup with "
            "a tight stop and a worthwhile target is present, call your single order "
            f"tool to act. Set model_used to '{self._model}'. Then output the "
            "SLHuntingDecision JSON exactly per the FINAL OUTPUT FORMAT."
        )

    # ------------------------------------------------------------------
    # Default runner (Claude Agent SDK)
    # ------------------------------------------------------------------

    def _system_prompt_as_file(self, system_prompt: str) -> dict[str, str]:
        """Persist ``system_prompt`` to a temp file and return the SDK's file form.

        SLH-003: the SDK passes a *string* system prompt to the spawned ``claude``
        CLI as a literal ``--system-prompt`` argv entry, and Windows caps a child
        process command line at 32,767 characters. ``build_system_prompt()`` alone
        is ~37k chars (since the v3f-v3h knowledge additions), so a string spawn
        dies with WinError 206 before any model call — misreported by the SDK as
        ``CLINotFoundError``. The file form (``--system-prompt-file``) keeps the
        command line tiny no matter how much knowledge is added. The file is
        written once per unique text and reused across bars, so the system prefix
        stays byte-identical per session and prompt caching is preserved.
        """
        if (
            self._system_prompt_file_path is None
            or self._system_prompt_file_text != system_prompt
        ):
            fd, path = tempfile.mkstemp(prefix="sl_hunting_system_prompt_", suffix=".md")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(system_prompt)
            self._system_prompt_file_path = path
            self._system_prompt_file_text = system_prompt
        return {"type": "file", "path": self._system_prompt_file_path}

    async def _default_run(
        self,
        prompt: str,
        *,
        system_prompt: str,
        model: str,
        max_turns: int,
        tool_context: SLHuntingToolContext,
    ) -> AgentRunResult:
        """Run one agentic loop on the Claude Agent SDK and return final text.

        Imports `claude_agent_sdk` lazily so this module imports without the SDK.
        Registers the in-process MCP server for THIS bar and restricts the agent to
        our tools only (`allowed_tools` + `permission_mode="dontAsk"`).
        """
        try:
            import claude_agent_sdk as claude_sdk  # type: ignore[import-not-found, unused-ignore]
            from claude_agent_sdk import (  # type: ignore[import-not-found, unused-ignore]
                AssistantMessage,
                ClaudeAgentOptions,
                CLINotFoundError,
                ProcessError,
                ResultMessage,
                query,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SLHuntingAgentError(
                "claude-agent-sdk is not installed. Run `pip install claude-agent-sdk` "
                "and sign in once with the bundled Claude CLI (using your Claude "
                "subscription) to enable the SL Hunting AI Agent. Keep ANTHROPIC_API_KEY "
                "UNSET so it bills your plan, not per-token API usage."
            ) from exc

        mcp_servers, allowed_tools = build_sl_hunting_mcp_server(tool_context)

        options_kwargs: dict[str, Any] = {
            "model": model,
            # File form, NOT the raw string — a string is passed on the CLI
            # command line and blows Windows' 32,767-char cap (SLH-003).
            "system_prompt": self._system_prompt_as_file(system_prompt),
            "max_turns": max_turns,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
            # "dontAsk" denies any tool not in allowed_tools, so the agent can never
            # reach the built-in filesystem/bash tools in a headless run.
            "permission_mode": "dontAsk",
            "setting_sources": [],
        }
        if self._fast_mode:
            thinking_cfg = _disabled_thinking_config(claude_sdk)
            if thinking_cfg is not None:
                options_kwargs["thinking"] = thinking_cfg
            else:
                logger.warning("SL Hunting fast mode requested but ThinkingConfigDisabled "
                               "is unavailable; using default thinking.")
        options = ClaudeAgentOptions(**options_kwargs)

        final_text = ""
        cost_usd: float | None = None
        result_message: Any = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_message = message
                    cost_usd = getattr(message, "total_cost_usd", None)
                    if getattr(message, "result", None):
                        final_text = message.result or ""
                elif isinstance(message, AssistantMessage):
                    for block in getattr(message, "content", None) or []:
                        block_text = getattr(block, "text", None)
                        if block_text:
                            final_text = block_text
        except CLINotFoundError as exc:
            # The SDK raises CLINotFoundError for ANY FileNotFoundError from the
            # spawn — including WinError 206, where the CLI exists but the spawn's
            # COMMAND LINE exceeded Windows' 32,767-char cap. Telling the operator
            # to reinstall for that case sends them down the wrong path (SLH-003).
            if getattr(exc.__cause__, "winerror", None) == 206:
                raise SLHuntingAgentError(
                    "Claude CLI spawn failed: the command line exceeded Windows' "
                    "32,767-character limit (WinError 206). An oversized option is "
                    "being passed as a CLI argument (a string system prompt is the "
                    "usual culprit — it must go through _system_prompt_as_file)."
                ) from exc
            raise SLHuntingAgentError(
                "The bundled Claude CLI could not be found. Reinstall with "
                "`pip install --force-reinstall claude-agent-sdk`."
            ) from exc
        except ProcessError as exc:
            if _mentions_usage_limit(str(exc), getattr(exc, "stderr", None)):
                raise SLHuntingUsageLimitError() from exc
            raise SLHuntingAgentError(f"Claude Agent SDK process error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface the structured error instead
            # When the CLI returns an error result, SDK 0.2.x raises a GENERIC Exception
            # ("Claude Code returned an error result: ...") from inside the stream — but it
            # yields the structured ResultMessage to us FIRST, so classify by that (HTTP
            # 401 auth / 429 usage limit / etc.) for an actionable error rather than the
            # opaque text. Falls through to a generic error if we captured no result.
            _raise_for_error_result(result_message)
            raise SLHuntingAgentError(f"Claude Agent SDK error: {type(exc).__name__}") from exc

        # The loop can also complete normally with an error result (CLI exit 0); classify it.
        _raise_for_error_result(result_message)
        return AgentRunResult(text=final_text, cost_usd=cost_usd)

    # ------------------------------------------------------------------
    # Sync bridge (Windows-safe; lets threaded master code drive the async SDK)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_sync(
        coro: Awaitable[AgentRunResult], *, timeout_seconds: float | None = None
    ) -> AgentRunResult:
        """Run an async coroutine to completion from sync/threaded code.

        Uses a dedicated worker thread with its OWN event loop so we never collide
        with any running loop. On Windows a ProactorEventLoop is required because
        the Agent SDK launches the Claude CLI as a subprocess (identical to the
        Scanner app's bridge).

        `timeout_seconds` bounds the wait (SLH-001). The worker is a DAEMON
        thread: when the budget is exceeded we stop waiting and raise
        `SLHuntingTimeoutError`, abandoning the thread (a daemon can never block
        process exit, and a hung CLI subprocess dies with the process). The
        caller must then treat this bar's tool context as expired -- the
        abandoned loop may wake later and must not be allowed to act.
        """
        # ("ok", result) or ("err", exception), appended by the worker thread.
        outcome: list[tuple[str, Any]] = []

        def _runner() -> None:
            if sys.platform == "win32":
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                outcome.append(("ok", loop.run_until_complete(coro)))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread below
                outcome.append(("err", exc))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        thread = threading.Thread(target=_runner, name="sl-hunting-sdk-call", daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            # Hand the still-running thread back so the caller can gate future
            # calls on it instead of spawning another abandoned worker each bar.
            raise SLHuntingTimeoutError(timeout_seconds or 0.0, thread=thread)
        if not outcome:  # pragma: no cover - the runner always records one entry
            raise SLHuntingAgentError("SDK call thread ended without recording a result.")
        kind, value = outcome[0]
        if kind == "err":
            raise value
        return value

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _previous_call_still_running(self) -> bool:
        """True while a PRIOR timed-out SDK call's thread is still alive.

        SLH-001 / Codex (PR #41). A hung CLI would otherwise let each bar spawn
        another abandoned daemon thread + subprocess, piling up resources and
        stale agent loops. We gate new calls until the previous one finishes, so
        at most one abandoned call exists at a time. Clears the reference once
        the thread dies, resuming normal operation on the next bar.
        """
        thread = self._abandoned_call
        if thread is None:
            return False
        if thread.is_alive():
            return True
        self._abandoned_call = None
        return False

    def decide(
        self,
        candles: pd.DataFrame,
        executor: TradeExecutor,
        *,
        bnf_candles: pd.DataFrame | None = None,
        live_active: bool = False,
        broker: str | None = None,
    ) -> SLHuntingDecision:
        """Run one agentic pass for the latest bar and return the decision.

        The agent ACTS by calling its order tool during the loop (the tool drives
        `executor`), so by the time this returns, `executor` already reflects any
        trade. The returned decision is the agent's own record of what it did.

        This NEVER raises into the trading loop: on any failure it returns a safe
        HOLD decision with `setup="agent_error"` so the worker simply does nothing.
        """
        prepared = prepare_candles(candles)
        if prepared.empty:
            return SLHuntingDecision(
                action="HOLD", confidence=0, setup="no_data",
                reasoning="No candles available.", model_used=self._model,
            )

        tool_context = SLHuntingToolContext.build(
            prepared, executor, cfg=self._cfg, live_active=live_active, broker=broker,
            bnf_candles=bnf_candles,
        )
        position = executor.snapshot()
        prompt = self._build_user_prompt(prepared, position)
        runner = self._runner or self._default_run

        def _run_once() -> str:
            run_result = self._run_sync(
                runner(
                    prompt,
                    system_prompt=self._system_prompt,
                    model=self._model,
                    max_turns=self.MAX_TURNS,
                    tool_context=tool_context,
                ),
                timeout_seconds=self._sdk_timeout_seconds,
            )
            if run_result.cost_usd is not None:
                logger.info("SLHuntingAgent decision cost ~$%.4f", run_result.cost_usd)
            return run_result.text

        def _parse(text: str) -> SLHuntingDecision:
            payload = _extract_json_object(text)
            if payload is None:
                raise SLHuntingAgentError("Agent did not return a parseable SLHuntingDecision JSON object.")
            decision = SLHuntingDecision.model_validate(payload)
            return decision.model_copy(update={"model_used": self._model})

        def _hold(reasoning: str) -> SLHuntingDecision:
            return SLHuntingDecision(
                action="HOLD", confidence=0, setup="agent_error",
                reasoning=reasoning, model_used=self._model,
            )

        # Codex (PR #41): if a prior call timed out and its worker thread is still
        # hung, do NOT start another one -- gate to a fail-soft HOLD until it
        # finishes so we never accumulate abandoned threads/subprocesses.
        if self._previous_call_still_running():
            logger.warning(
                "SL Hunting agent holding — a previous SDK call is still running past its "
                "timeout; skipping this bar to avoid stacking hung agent loops."
            )
            return _hold("Previous agent call still running; holding.")

        # IMPORTANT: do NOT retry the agentic loop. Unlike the read-only Scanner
        # agents, our order tool has SIDE EFFECTS — the agent ENTERS/EXITS via the
        # executor DURING the loop. Re-running the loop after a malformed final JSON
        # could double-fire an order or act against already-changed state. So we run
        # exactly ONCE: the executor state is the source of truth for any trade, and
        # a bad/missing final JSON simply yields a safe HOLD record (no further action).
        try:
            text = _run_once()
        except SLHuntingTimeoutError as exc:
            # SLH-001: the call was ABANDONED, not finished -- the agentic loop may
            # still be alive on its daemon thread. Disarm this bar's order tool so
            # a late-waking loop cannot fire a zombie order, then fail-soft HOLD.
            tool_context.expire("sdk_timeout")
            # Track the still-running thread so the NEXT bar gates on it instead of
            # spawning another (Codex PR #41). Cleared once it finally finishes.
            self._abandoned_call = exc.thread
            logger.warning("SL Hunting agent holding — %s", exc)
            return _hold("Agent call timed out; holding.")
        except SLHuntingUsageLimitError as exc:
            # Actionable, content-free message (e.g. "...usage limit was hit (HTTP 429)").
            logger.warning("SL Hunting agent holding — %s", exc)
            return _hold("Agent usage-limited; holding.")
        except SLHuntingAuthError as exc:
            # Tells the operator exactly how to fix it (run `claude setup-token`).
            logger.warning("SL Hunting agent holding — %s", exc)
            return _hold("Agent auth failed; holding.")
        except Exception as exc:  # noqa: BLE001 - other infra failure (SDK/CLI)
            logger.warning("SLHuntingAgent run failed (%s); holding.", type(exc).__name__, exc_info=True)
            return _hold(f"Agent unavailable ({type(exc).__name__}); holding.")
        try:
            return _parse(text)
        except (SLHuntingAgentError, ValidationError) as exc:
            logger.warning("SLHuntingAgent returned invalid output (%s); holding.", type(exc).__name__)
            return _hold(f"Agent returned invalid output ({type(exc).__name__}); holding.")
