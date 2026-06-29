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
import concurrent.futures
import json
import logging
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import Field, ValidationError, field_validator

from sl_hunting_ai_validation import StrictAIModel
from sl_hunting_knowledge import FINAL_OUTPUT_INSTRUCTION, build_system_prompt
from sl_hunting_indicators import SLHuntingIndicatorConfig, prepare_candles
from sl_hunting_tools import SLHuntingToolContext, build_sl_hunting_mcp_server
from sl_hunting_executor import TradeExecutor

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
    """The Claude subscription's Agent SDK usage limit was hit."""


# A runner drives one full agentic loop and returns the final message text.
# Production uses `SLHuntingAgent._default_run`; tests inject a fake.
RunnerFn = Callable[..., Awaitable[AgentRunResult]]


# ---------------------------------------------------------------------------
# Strict output schema
# ---------------------------------------------------------------------------

Action = Literal["ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"]


class SLHuntingDecision(StrictAIModel):
    """The agent's structured decision for one bar.

    `confidence` is validated with a `@field_validator` (not `Field(ge=, le=)`) so
    the JSON schema we describe to Claude stays free of `minimum`/`maximum`, which
    Claude rejects on integer types — same trick as the Scanner's verdict models.
    """

    action: Action = Field(description="ENTER_LONG, ENTER_SHORT, EXIT or HOLD.")
    stop: float = Field(default=0.0, description="Underlying stop level for an entry; 0 for EXIT/HOLD.")
    target: float = Field(default=0.0, description="Underlying target level for an entry; 0 for EXIT/HOLD.")
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
    blob = " ".join(t for t in texts if t).lower()
    return any(k in blob for k in ("usage limit", "rate limit", "429", "quota"))


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
    ) -> None:
        if not model:
            raise ValueError("SLHuntingAgent: model is required.")
        self._model = model
        self._runner = runner
        self._fast_mode = bool(fast_mode)
        self._cfg = indicator_config or SLHuntingIndicatorConfig()
        # Optional v3 LEARNED LESSONS block (loaded + formatted by the caller, gated by
        # SL_HUNTING_LESSONS_ENABLED). Injected ONCE here, before the output contract,
        # so the system prefix stays stable per session and prompt caching is preserved.
        learned = ("\n\n" + lessons_block.strip()) if lessons_block and lessons_block.strip() else ""
        self._system_prompt = build_system_prompt() + learned + FINAL_OUTPUT_INSTRUCTION

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _snapshot_csv(self, candles: pd.DataFrame) -> str:
        recent = candles.tail(_SNAPSHOT_BARS)
        lines = ["time,open,high,low,close"]
        for row in recent.itertuples(index=False):
            ts = str(getattr(row, "timestamp", ""))[:19]
            lines.append(f"{ts},{float(row.open):.2f},{float(row.high):.2f},{float(row.low):.2f},{float(row.close):.2f}")
        return "\n".join(lines)

    def _build_user_prompt(self, candles: pd.DataFrame, position: dict[str, Any]) -> str:
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

        ThinkingConfigDisabled = getattr(claude_sdk, "ThinkingConfigDisabled", None)
        mcp_servers, allowed_tools = build_sl_hunting_mcp_server(tool_context)

        options_kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "mcp_servers": mcp_servers,
            "allowed_tools": allowed_tools,
            # "dontAsk" denies any tool not in allowed_tools, so the agent can never
            # reach the built-in filesystem/bash tools in a headless run.
            "permission_mode": "dontAsk",
            "setting_sources": [],
        }
        if self._fast_mode and ThinkingConfigDisabled is not None:
            options_kwargs["thinking"] = ThinkingConfigDisabled()
        elif self._fast_mode:
            logger.warning("SL Hunting fast mode requested but ThinkingConfigDisabled is unavailable; using default thinking.")
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
                        final_text = message.result
                elif isinstance(message, AssistantMessage):
                    for block in getattr(message, "content", None) or []:
                        block_text = getattr(block, "text", None)
                        if block_text:
                            final_text = block_text
        except CLINotFoundError as exc:
            raise SLHuntingAgentError(
                "The bundled Claude CLI could not be found. Reinstall with "
                "`pip install --force-reinstall claude-agent-sdk`."
            ) from exc
        except ProcessError as exc:
            if _mentions_usage_limit(str(exc), getattr(exc, "stderr", None)):
                raise SLHuntingUsageLimitError() from exc
            raise SLHuntingAgentError(f"Claude Agent SDK process error: {exc}") from exc

        if result_message is not None and getattr(result_message, "is_error", False):
            if getattr(result_message, "api_error_status", None) == 429:
                raise SLHuntingUsageLimitError()
            raise SLHuntingAgentError(
                f"SL Hunting agent run failed: {str(getattr(result_message, 'result', '') or '')[:300]}".strip()
            )
        return AgentRunResult(text=final_text, cost_usd=cost_usd)

    # ------------------------------------------------------------------
    # Sync bridge (Windows-safe; lets threaded master code drive the async SDK)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_sync(coro: Awaitable[AgentRunResult]) -> AgentRunResult:
        """Run an async coroutine to completion from sync/threaded code.

        Uses a dedicated worker thread with its OWN event loop so we never collide
        with any running loop. On Windows a ProactorEventLoop is required because
        the Agent SDK launches the Claude CLI as a subprocess (identical to the
        Scanner app's bridge).
        """

        def _runner() -> AgentRunResult:
            if sys.platform == "win32":
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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
            return SLHuntingDecision(action="HOLD", confidence=0, setup="no_data", reasoning="No candles available.", model_used=self._model)

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
                )
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

        # IMPORTANT: do NOT retry the agentic loop. Unlike the read-only Scanner
        # agents, our order tool has SIDE EFFECTS — the agent ENTERS/EXITS via the
        # executor DURING the loop. Re-running the loop after a malformed final JSON
        # could double-fire an order or act against already-changed state. So we run
        # exactly ONCE: the executor state is the source of truth for any trade, and
        # a bad/missing final JSON simply yields a safe HOLD record (no further action).
        try:
            text = _run_once()
        except Exception as exc:  # noqa: BLE001 - infra failure (SDK/CLI/usage limit)
            logger.warning("SLHuntingAgent run failed (%s); holding.", type(exc).__name__, exc_info=True)
            return _hold(f"Agent unavailable ({type(exc).__name__}); holding.")
        try:
            return _parse(text)
        except (SLHuntingAgentError, ValidationError) as exc:
            logger.warning("SLHuntingAgent returned invalid output (%s); holding.", type(exc).__name__)
            return _hold(f"Agent returned invalid output ({type(exc).__name__}); holding.")
