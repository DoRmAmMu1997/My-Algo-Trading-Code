"""In-process MCP tool server for the SL Hunting AI Agent.

Why the agent gets tools (beginner note)
----------------------------------------
Instead of pre-chewing every fact into the prompt, the Claude agent is given
*tools* it calls to fetch precise, deterministic facts about the current NIFTY
chart, plus ONE tool to act:

Read-only context tools (no arguments — each closes over the per-call context):
- ``pivot_and_levels`` → day pivot, prev-day OHLC, today O/H/L, first-candle hi/lo,
                         psych levels, previous close, and distances.
- ``fibo_levels``      → 50/61/78 retracement + 161/261 extension of the latest swing.
- ``candle_patterns``  → reversal patterns on recent candles + confirmation status.
- ``market_structure`` → swings, trend, fast/slow, trendline points, W/M, doubles.
- ``position_state``   → the current open position (or flat).
- ``bank_nifty``       → BankNIFTY's own pivot/levels/structure/patterns (advisory).
- ``cross_index``      → NIFTY-vs-BankNIFTY alignment verdict (NF/BNF rules).

Action tool (exactly ONE, named by the environment):
- ``place_paper_order`` / ``place_kotak_order`` / ``place_shoonya_order`` — only the
  one the configuration selected is registered (see `executor.execution_tool_name`),
  so the agent can never choose paper-vs-real or the broker.

This mirrors the house pattern in
`../Streamlit Scanner App/backend/technical/tools.py`: the SDK is imported lazily
inside `build_sl_hunting_mcp_server` (so this module imports without the SDK), tool
results are wrapped as ``{"content": [{"type": "text", "text": json}]}``, and tool
names are namespaced ``mcp__<server>__<tool>``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd
from sl_hunting_executor import (
    MAX_ORDER_REASON_CHARS,
    TradeExecutor,
    execution_tool_name,
)
from sl_hunting_indicators import (
    SLHuntingIndicatorConfig,
    candle_patterns,
    cross_index_signal,
    fibo_levels,
    market_structure,
    pivot_and_levels,
    prepare_candles,
)

SERVER_NAME = "slhunting"

# SLH-002: sanity bands for the agent's ENTRY stop/target, as fractions of the
# current underlying price. Generous on purpose -- real SL-Hunting stops are a
# few dozen points (<<1% of NIFTY), so these only catch hallucinated garbage
# (wrong-instrument levels, sign flips, absurd numbers), never a real setup.
MAX_STOP_DISTANCE_FRACTION = 0.03
MAX_TARGET_DISTANCE_FRACTION = 0.10


def _entry_levels_error(direction: str, stop: float, target: float, price: float) -> str | None:
    """Explain why these entry levels are unusable, or return None when sane.

    The mechanical per-poll exit uses these UNDERLYING levels verbatim: a stop on
    the wrong side either fires instantly or never fires at all, and a non-finite
    or far-away level silently disables the stop, leaving only max-loss and the
    15:15 square-off. Checked at the order tool because only it knows the price.
    """
    if price <= 0:
        return "no current underlying price available to validate the levels"
    if not (math.isfinite(stop) and math.isfinite(target)):
        return "stop and target must be finite numbers"
    if stop <= 0 or target <= 0:
        return "an entry requires a positive stop AND target on the underlying"
    if direction == "LONG" and not (stop < price < target):
        return f"a LONG needs stop below and target above the current price {price:.2f}"
    if direction == "SHORT" and not (target < price < stop):
        return f"a SHORT needs stop above and target below the current price {price:.2f}"
    if abs(stop - price) > MAX_STOP_DISTANCE_FRACTION * price:
        return f"stop is more than {MAX_STOP_DISTANCE_FRACTION:.0%} away from the current price"
    if abs(target - price) > MAX_TARGET_DISTANCE_FRACTION * price:
        return f"target is more than {MAX_TARGET_DISTANCE_FRACTION:.0%} away from the current price"
    return None


# SLH-007. The `reason` the model passes here is the PERMANENT record of the trade:
# it flows to `exit_position(reason)` -> the journal's `outcome.exit_reason`, the log
# line and the Telegram alert, and the reflection coach later reads it to propose
# lessons. The model's own final-JSON reasoning is produced ~25s later and never
# reaches the journal, so a throwaway here is information LOST, not a cosmetic
# blemish. Four of the first 46 journal rows recorded "placeholder" (or "placeholder
# - will not call, holding") as the permanent reason for a real trade.
MIN_REASON_CHARS = 12
# One bound, defined next to the executor that ultimately records the string.
# Importing it (rather than repeating the number) keeps the tool's trimming and the
# executor's trimming from ever disagreeing about what was written down.
MAX_REASON_CHARS = MAX_ORDER_REASON_CHARS
_PLACEHOLDER_REASONS = frozenset(
    {"placeholder", "n/a", "na", "tbd", "none", "-", "test", "todo", "reason", "exit", "entry"}
)
LOGGER = logging.getLogger(__name__)

NO_REASON_SENTINEL = "AI_EXIT (no reason supplied by the model)"


def order_tool_description(venue: str, *, note_active: bool = False) -> str:
    """The order tool's description, as a function so tests can assert on it.

    Kept out of `build_sl_hunting_mcp_server` because that call needs the Claude
    Agent SDK installed; the wording is the fix for SLH-007 and must be checkable
    without it.
    """
    return (
        f"Place an SL-Hunting order on NIFTY ATM options via the {venue} venue (the "
        "configuration chose this venue; you cannot change it). CALLING THIS TOOL IS "
        "ITSELF THE ACTION -- it places a real order at the moment you call it. There "
        "is no HOLD action and there is no dry run: if your decision is to HOLD, or to "
        "keep an open position, DO NOT CALL THIS TOOL AT ALL. Note especially that an "
        "EXIT is executed immediately and is NEVER rejected -- unlike an entry it is "
        "not validated and cannot be corrected or undone, so an EXIT sent by mistake "
        "closes the position for good. action is ENTER_LONG "
        "(buy CALL), ENTER_SHORT (buy PUT) or EXIT. stop and target are NIFTY "
        "UNDERLYING price levels (required for entries; ignored for EXIT). They must "
        "BRACKET the current price on the correct side (LONG: stop below, target "
        "above; SHORT: reversed), with the stop within ~3% and the target within "
        "~10% of the current price -- otherwise the order is rejected. "
        "reason is the PERMANENT record of this trade: it is written verbatim to the "
        "trade journal, the log and the operator's alert, and it is what the reflection "
        "coach reads later to work out what worked. The reasoning in your final JSON is "
        "NOT saved there, so this is your only chance to say why you acted. Give a real "
        f"one-line justification naming the setup and the trigger; it is trimmed to "
        f"{MAX_REASON_CHARS} printable characters, so put the point FIRST. A placeholder "
        "or an empty string is REJECTED on entries (resend a real one), and on an exit it "
        "is replaced by a sentinel recording that you supplied nothing. "
        "exit_leg (EXIT only) is NIFTY, BNF or BOTH "
        "(default BOTH): every NIFTY entry is mirrored by an equal-lot BankNIFTY ATM leg, "
        "and you may cut ONE leg on premise-invalidation while the other runs — but hard "
        "risk (stop/target/max-loss/square-off) always closes both. Returns whether "
        "the order was accepted or rejected."
        + (
            " note_check is REQUIRED on entries today because a pre-open analyst note "
            "applies to this session. Start it with AGREES, CONTRADICTS or "
            "NOT_APPLICABLE, then a colon and the branch of the note you are applying "
            "-- for example 'CONTRADICTS: the note says a gap-up means follow the gap, "
            "but I am fading it because the opening spike trapped fresh buyers'. The "
            "note is ADVISORY and CONTRADICTS is a perfectly good answer: your own read "
            "wins. What is not allowed is ignoring it silently."
            if note_active
            else ""
        )
    )


def _printable_one_line(reason: str) -> str:
    """Collapse any input to one printable, whitespace-normalised line.

    Control characters become spaces rather than surviving into the journal, the
    log or an operator's Telegram alert, where they could reflow or colour text.
    """

    raw = "".join(char if char.isprintable() else " " for char in str(reason or ""))
    return " ".join(raw.split())


def _reason_meaning_problem(reason: str) -> str | None:
    """Explain why this justification says NOTHING, else None.

    Deliberately separate from length. A blank, placeholder or two-word reason is
    a defect the model must FIX, so entries are rejected and it gets to retry
    within the same pass. Being verbose is not a defect of meaning -- it is
    trimmed (see `bounded_reason`), never bounced. Measured on the shipped
    journal, 38 of 39 model-written reasons ran past 120 characters, so rejecting
    on length would have cost an extra SDK round-trip on essentially every entry
    inside a one-minute bar.
    """

    cleaned = _printable_one_line(reason)
    if not cleaned:
        return "reason must not be blank"
    if cleaned.lower().rstrip(".") in _PLACEHOLDER_REASONS:
        return f"{cleaned!r} is a placeholder, not a justification"
    if cleaned.lower().startswith("placeholder"):
        return f"{cleaned!r} is a placeholder, not a justification"
    if len(cleaned) < MIN_REASON_CHARS:
        return f"reason is too short to be a justification (min {MIN_REASON_CHARS} characters)"
    return None


def bounded_reason(reason: str) -> str:
    """The exact string that will be recorded: printable, one line, length-capped."""

    return _printable_one_line(reason)[:MAX_REASON_CHARS].strip()


def _reason_problem(reason: str) -> str | None:
    """Every way a reason can be unusable, including length.

    Retained for callers that want the strict question ("is this string already
    fit to record verbatim?"). The order path does NOT gate entries on this --
    see `_reason_meaning_problem`.
    """

    meaning = _reason_meaning_problem(reason)
    if meaning is not None:
        return meaning
    cleaned = str(reason or "").strip()
    if len(cleaned) > MAX_REASON_CHARS:
        return f"reason must be at most {MAX_REASON_CHARS} characters"
    if any(not char.isprintable() for char in cleaned):
        return "reason must be one printable line"
    return None


def _safe_exit_reason(reason: str) -> str:
    """Return a bounded printable exit record without ever blocking the exit."""

    cleaned = bounded_reason(reason)
    if _reason_meaning_problem(cleaned) is not None:
        return NO_REASON_SENTINEL
    return cleaned

# Read-only context tools are always present; the action tool's name is decided at
# build time by the environment, so it is appended in `build_sl_hunting_mcp_server`.
CONTEXT_TOOL_NAMES = [
    f"mcp__{SERVER_NAME}__pivot_and_levels",
    f"mcp__{SERVER_NAME}__fibo_levels",
    f"mcp__{SERVER_NAME}__candle_patterns",
    f"mcp__{SERVER_NAME}__market_structure",
    f"mcp__{SERVER_NAME}__position_state",
    f"mcp__{SERVER_NAME}__bank_nifty",
    f"mcp__{SERVER_NAME}__cross_index",
]


class ToolContextState(StrEnum):
    """Lifecycle of the one side-effect capability granted to an agent pass.

    The transitions, in plain English:

    - ``ACTIVE``    : the pass may still place its one order.  Rejected
                      validation attempts return here so the model can
                      correct its levels within the same pass.
    - ``EXECUTING`` : an order is inside the executor right now -- a second
                      call arriving mid-flight is refused, never queued.
    - ``CONSUMED``  : an ACCEPTED side effect happened.  Terminal: one pass
                      gets at most one accepted ENTER or EXIT, ever.
    - ``EXPIRED``   : the decision window closed (deadline, stale generation,
                      mechanical exit, or the pass simply finished) before an
                      accepted action.  Terminal: a late tool call from an
                      abandoned SDK loop finds a dead capability, not a
                      market order.
    """

    ACTIVE = "ACTIVE"
    EXECUTING = "EXECUTING"
    CONSUMED = "CONSUMED"
    EXPIRED = "EXPIRED"


@dataclass
class SLHuntingToolContext:
    """Everything the tools need for ONE per-bar decision about NIFTY.

    Built fresh for each `decide(...)` call so concurrent workers never share
    mutable state. Holds the prepared candle window, the detector config, the
    injected executor, and the environment's venue selection (live + broker) used
    only to NAME the single action tool.
    """

    candles: pd.DataFrame
    cfg: SLHuntingIndicatorConfig
    executor: TradeExecutor
    live_active: bool = False
    broker: str | None = None
    last_price: float = field(default=0.0)
    bnf_candles: pd.DataFrame | None = None
    # SLH-009: True when a pre-open note applies to TODAY. The note stays ADVISORY
    # -- it can never veto a trade -- but when one exists the order tool refuses an
    # ENTRY until the model says which branch of it applies and whether it is acting
    # with or against it. On 2026-08-03 the agent faded a gap-up on a day its own
    # note said gap-up means do NOT target buyers, never mentioned the note at all,
    # and was stopped in 27 seconds. Code cannot check COMPLIANCE (which branch
    # applies depends on the open, which only the model can read), but it can make
    # silently ignoring the note impossible.
    premarket_note_active: bool = False
    # Set via `expire()` when the agent abandons this bar's SDK call (SLH-001,
    # e.g. a timed-out CLI call). The abandoned agentic loop keeps running on
    # its daemon thread and could otherwise still call the order tool MINUTES
    # later, against a market that has moved on; once expired, `do_order`
    # refuses instead of touching the executor.
    expired_reason: str | None = field(default=None)
    generation: int = 0
    deadline_monotonic: float | None = None
    generation_is_current: Callable[[int], bool] | None = field(default=None, repr=False)
    state: ToolContextState = field(default=ToolContextState.ACTIVE, init=False)
    _execution_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _execution_lock: Any = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def build(
        cls,
        candles: pd.DataFrame,
        executor: TradeExecutor,
        *,
        cfg: SLHuntingIndicatorConfig | None = None,
        live_active: bool = False,
        broker: str | None = None,
        bnf_candles: pd.DataFrame | None = None,
        generation: int = 0,
        generation_is_current: Callable[[int], bool] | None = None,
        deadline_seconds: float | None = None,
        execution_lock: Any | None = None,
        premarket_note_active: bool = False,
    ) -> SLHuntingToolContext:
        """Prepare the per-bar context: clean the candles, cache the last price, attach BNF.

        Normalises the NIFTY (and optional BankNIFTY) frames once here so each tool call
        reuses the same prepared data, and records the venue selection (live + broker)
        used only to NAME the single order tool.
        """
        cfg = cfg or SLHuntingIndicatorConfig()
        prepared = prepare_candles(candles)
        last_price = float(prepared["close"].iloc[-1]) if not prepared.empty else 0.0
        # BankNIFTY is optional (advisory cross-confirmation); prepare it if given.
        prepared_bnf = None
        if bnf_candles is not None:
            prepared_bnf = prepare_candles(bnf_candles)
            if prepared_bnf.empty:
                prepared_bnf = None
        return cls(
            candles=prepared,
            cfg=cfg,
            executor=executor,
            live_active=bool(live_active),
            broker=broker,
            last_price=last_price,
            bnf_candles=prepared_bnf,
            premarket_note_active=bool(premarket_note_active),
            generation=int(generation),
            generation_is_current=generation_is_current,
            deadline_monotonic=(
                time.monotonic() + float(deadline_seconds)
                if deadline_seconds is not None
                else None
            ),
            _execution_lock=execution_lock or threading.RLock(),
        )

    # --- tool payload builders (plain functions; easy to unit test) ---------

    def pivot_and_levels_payload(self) -> dict[str, Any]:
        return pivot_and_levels(self.candles, self.cfg)

    def fibo_levels_payload(self) -> dict[str, Any]:
        return fibo_levels(self.candles, self.cfg)

    def candle_patterns_payload(self) -> dict[str, Any]:
        return candle_patterns(self.candles, self.cfg)

    def market_structure_payload(self) -> dict[str, Any]:
        return market_structure(self.candles, self.cfg)

    def position_state_payload(self) -> dict[str, Any]:
        return self.executor.snapshot()

    def bank_nifty_payload(self) -> dict[str, Any]:
        """BankNIFTY's own pivot/levels, structure and recent patterns (advisory)."""
        if self.bnf_candles is None or self.bnf_candles.empty:
            return {"available": False, "reason": "BankNIFTY data unavailable; judge on NIFTY alone."}
        return {
            "available": True,
            "pivot_and_levels": pivot_and_levels(self.bnf_candles, self.cfg),
            "market_structure": market_structure(self.bnf_candles, self.cfg),
            "candle_patterns": candle_patterns(self.bnf_candles, self.cfg),
        }

    def cross_index_payload(self) -> dict[str, Any]:
        """NIFTY-vs-BankNIFTY alignment verdict (the doc's NF/BNF rules)."""
        return cross_index_signal(self.candles, self.bnf_candles, self.cfg)

    def action_tool_name(self) -> str:
        return execution_tool_name(self.live_active, self.broker)

    def expire(self, reason: str) -> None:
        """Disarm this bar's order tool (the decision window is over -- SLH-001)."""
        with self._execution_lock:
            if self.state is ToolContextState.CONSUMED:
                return
            self.expired_reason = str(reason)
            self.state = ToolContextState.EXPIRED

    @property
    def execution_result(self) -> dict[str, Any] | None:
        """Copy of the accepted side effect, used as the authoritative decision."""
        with self._execution_lock:
            return dict(self._execution_result) if self._execution_result is not None else None

    def _refusal_inside_lock(self) -> dict[str, Any] | None:
        """Recheck pass freshness at the final side-effect boundary."""
        if self.state is not ToolContextState.ACTIVE:
            reason = self.expired_reason or f"tool context is {self.state.value.lower()}"
            return {"accepted": False, "reason": f"decision window expired/unavailable ({reason}); order refused"}
        if self.generation_is_current is not None and not self.generation_is_current(self.generation):
            self.expired_reason = "stale generation"
            self.state = ToolContextState.EXPIRED
            return {"accepted": False, "reason": "decision generation is stale; order refused"}
        if self.deadline_monotonic is not None and time.monotonic() > self.deadline_monotonic:
            self.expired_reason = "deadline elapsed"
            self.state = ToolContextState.EXPIRED
            return {"accepted": False, "reason": "decision deadline elapsed; order refused"}
        return None

    # SLH-009. Accepted answers for an ENTRY while a pre-open note applies. The note
    # is ADVISORY, so CONTRADICTS is a legitimate answer that lets the order through
    # -- what is forbidden is not answering at all.
    NOTE_ALIGNMENTS = ("AGREES", "CONTRADICTS", "NOT_APPLICABLE")

    def _premarket_note_problem(self, note_check: str) -> str | None:
        """Explain why this note acknowledgement is unusable, or None when fine."""

        if not self.premarket_note_active:
            return None
        cleaned = _printable_one_line(note_check).strip()
        if not cleaned:
            return (
                "a pre-open analyst note applies to TODAY and you have not addressed it. "
                "Resend this entry with note_check set to AGREES, CONTRADICTS or "
                "NOT_APPLICABLE, followed by the branch of the note you are applying "
                "(e.g. 'CONTRADICTS: note says gap-up means follow the gap; I am fading "
                "it because ...'). The note is ADVISORY -- CONTRADICTS is allowed and "
                "your own read wins -- but it may not be passed over in silence."
            )
        verdict = cleaned.split(":", 1)[0].strip().upper().replace(" ", "_")
        if verdict not in self.NOTE_ALIGNMENTS:
            return (
                f"note_check must START with one of {', '.join(self.NOTE_ALIGNMENTS)} "
                f"(got {verdict!r}). Follow it with the branch you are applying."
            )
        if verdict != "NOT_APPLICABLE" and len(cleaned) <= len(verdict) + 1:
            return (
                f"note_check says {verdict} but does not say WHICH branch of the note "
                "you are applying. Add it after a colon."
            )
        return None

    def do_order(
        self,
        action: str,
        stop: float,
        target: float,
        reason: str,
        exit_leg: str = "BOTH",
        note_check: str = "",
    ) -> dict[str, Any]:
        """Route an agent order through the injected executor (single source of truth).

        `exit_leg` applies only to EXIT: NIFTY / BNF / BOTH (default). It lets the agent
        cut one leg of the BankNIFTY-mirror basket on premise-invalidation while leaving
        the other running; hard risk (stop/target/max-loss/square-off) still ties both.

        SLH-002: entry stop/target are validated against the CURRENT price here --
        this is the one price-aware choke point for the agent's numbers. A rejected
        entry goes back to the model mid-loop as `accepted: false` with the reason,
        so it can correct its levels; the executor is never touched.
        """
        action = (action or "").strip().upper()
        executor_call: Callable[[], dict[str, Any]]
        if action in ("ENTER_LONG", "ENTER_SHORT"):
            direction = "LONG" if action == "ENTER_LONG" else "SHORT"
            stop = float(stop or 0.0)
            target = float(target or 0.0)
            problem = _entry_levels_error(direction, stop, target, self.last_price)
            if problem:
                return {"accepted": False, "reason": f"invalid entry levels: {problem}"}
            # SLH-007: an ENTRY may be refused so the model rewrites its justification
            # within the same pass -- exactly how the levels check above works. Nothing
            # is at risk yet, because no position exists. Only MEANING is grounds for
            # refusal; an over-long or non-printable reason is trimmed rather than
            # bounced, because bouncing it would cost an SDK round-trip on nearly
            # every entry without protecting anything.
            reason_problem = _reason_meaning_problem(reason)
            if reason_problem:
                return {
                    "accepted": False,
                    "reason": (
                        f"unusable reason: {reason_problem}. This string is the PERMANENT "
                        "record of the trade (journal, log and alert) -- resend the order "
                        "with a one-line justification naming the setup and why you are acting."
                    ),
                }
            # SLH-009: when a pre-open note applies today, an ENTRY must say which
            # branch of it applies and whether this trade agrees with it. The note
            # remains ADVISORY -- "CONTRADICTS" is an accepted answer and the order
            # goes through -- but it can no longer be passed over in silence.
            note_problem = self._premarket_note_problem(note_check)
            if note_problem:
                return {"accepted": False, "reason": note_problem}
            entry_reason = bounded_reason(reason)
            recorded_reason = entry_reason
            def executor_call() -> dict[str, Any]:
                return self.executor.enter(
                    direction, stop, target, entry_reason, self.last_price
                )
        elif action == "EXIT":
            leg = (exit_leg or "BOTH").strip().upper()
            if leg not in ("NIFTY", "BNF", "BOTH"):
                return {"accepted": False, "reason": f"invalid exit_leg {exit_leg!r}; expected NIFTY, BNF or BOTH"}
            # SLH-007: an EXIT is NEVER refused over its wording. Blocking it would
            # strand an open position -- the one thing the risk rules forbid. Record an
            # explicit sentinel instead, so the journal and the coach are told plainly
            # that no justification was given rather than being handed a lie.
            exit_reason = _safe_exit_reason(reason)
            recorded_reason = exit_reason
            if exit_reason == NO_REASON_SENTINEL:
                # SLH-010: the exit still goes through -- SLH-007 above is not
                # being reversed -- but a deliberate exit always carries a
                # justification, because the tool description tells the model
                # that string is the PERMANENT record. An EXIT with nothing in
                # it is therefore the strongest signal available that the call
                # was not intended, which is exactly what happened on
                # 2026-08-19 09:38: the model later reported "the order tool
                # was called with EXIT instead of intended HOLD". That day the
                # fact sat at INFO inside a long trade line and was noticed
                # only because the model confessed on the next bar. Say it
                # plainly instead, so the operator sees it the same session.
                LOGGER.warning(
                    "SL Hunting: EXIT (leg=%s) executed with NO reason from the model. "
                    "A deliberate exit always carries one, so treat this as a probable "
                    "UNINTENDED order and check whether the position should have been held.",
                    leg,
                )
            def executor_call() -> dict[str, Any]:
                return self.executor.exit(exit_reason, self.last_price, leg=leg)
        else:
            return {"accepted": False, "reason": f"unknown action {action!r}; expected ENTER_LONG, ENTER_SHORT or EXIT"}

        # The final gate.  The freshness checks (state, generation, deadline)
        # run INSIDE the same lock that serializes the executor call, because
        # checking first and locking second would leave a gap: the mechanical
        # risk loop could invalidate this pass between the check and the
        # order.  Inside the lock, what was checked is what executes.  This is
        # also the lock the worker's stop/target/square-off paths hold while
        # THEY act, so an agent order and a mechanical exit can never race.
        with self._execution_lock:
            refusal = self._refusal_inside_lock()
            if refusal is not None:
                return refusal
            self.state = ToolContextState.EXECUTING
            try:
                result = executor_call()
            except BaseException:
                # An executor that blew up mid-order leaves unknown exposure;
                # this pass must not get a second try at it.
                self.expired_reason = "executor raised"
                self.state = ToolContextState.EXPIRED
                raise
            if bool(result.get("accepted")):
                # SLH-011: keep the model's OWN justification on the result. If this
                # pass later fails to return a parseable decision, the journal can
                # record why the order was placed rather than a placeholder -- and
                # for an ENTRY that string is guaranteed meaningful, because
                # `_reason_meaning_problem` above rejects placeholders on entries.
                self._execution_result = {**result, "model_reason": recorded_reason}
                self.state = ToolContextState.CONSUMED
            else:
                # Rejected validation/execution may be corrected once within the
                # same pass; only an accepted side effect consumes the context.
                self.state = ToolContextState.ACTIVE
            return result


def _as_tool_text(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload dict in the MCP tool-result envelope the SDK expects."""
    return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}


def build_sl_hunting_mcp_server(context: SLHuntingToolContext):
    """Build the in-process MCP server: 5 read-only tools + 1 env-named order tool.

    Returns ``(mcp_servers, allowed_tool_names)`` ready for `ClaudeAgentOptions`.
    Imports the Claude Agent SDK lazily so importing this module never requires it
    (mirrors the Scanner app's `build_technical_mcp_server`).
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # type: ignore[import-not-found, unused-ignore]

    @tool("pivot_and_levels",
          "Day pivot, previous-day OHLC, today's O/H/L, first-candle hi/lo, psych levels "
          "and the previous close, with distances from current price.", {})
    async def _pivot(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.pivot_and_levels_payload())

    @tool("fibo_levels",
          "Fibonacci 50/61/78 retracement and 161/261 extension levels of the most recent "
          "significant swing, and where price sits relative to them.", {})
    async def _fibo(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.fibo_levels_payload())

    @tool("candle_patterns",
          "Reversal candlestick patterns (hammer/doji, engulfing, inside-bar, reversal-bar, "
          "star) on recent candles, each tagged with whether a confirmation candle has "
          "already printed.", {})
    async def _patterns(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.candle_patterns_payload())

    @tool("market_structure",
          "Swing points, trend, fast/slow read, trendline points, and W/M / double "
          "top-bottom detection.", {})
    async def _structure(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.market_structure_payload())

    @tool("position_state",
          "Your current open position (direction, entry, stop, target, unrealised P&L) "
          "or 'flat'.", {})
    async def _position(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.position_state_payload())

    @tool("bank_nifty",
          "BankNIFTY's OWN pivot/levels, market structure and recent candle patterns (for "
          "cross-confirmation). Returns available:false when BankNIFTY data is unavailable.", {})
    async def _bank_nifty(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.bank_nifty_payload())

    @tool("cross_index",
          "NIFTY-vs-BankNIFTY alignment verdict per the NF/BNF rules (both at support -> "
          "bias down; both breakdown -> bias up; one fails -> the holder wins; opposite "
          "sides of pivot -> wait). Advisory. available:false when BankNIFTY data is "
          "unavailable.", {})
    async def _cross_index(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.cross_index_payload())

    order_name = context.action_tool_name()
    venue_by_tool = {
        "place_paper_order": "PAPER",
        "place_kotak_order": "KOTAK (LIVE)",
        "place_shoonya_order": "SHOONYA (LIVE)",
        "place_flattrade_order": "FLATTRADE (LIVE)",
    }
    venue = venue_by_tool[order_name]

    @tool(
        order_name,
        order_tool_description(venue, note_active=context.premarket_note_active),
        {
            "action": str,
            "stop": float,
            "target": float,
            "reason": str,
            "exit_leg": str,
            "note_check": str,
        },
    )
    async def _order(args: dict[str, Any]) -> dict[str, Any]:
        result = context.do_order(
            args.get("action", ""),
            args.get("stop", 0.0),
            args.get("target", 0.0),
            args.get("reason", ""),
            args.get("exit_leg", "BOTH"),
            args.get("note_check", ""),
        )
        return _as_tool_text(result)

    server = create_sdk_mcp_server(
        name=SERVER_NAME,
        version="1.0.0",
        tools=[_pivot, _fibo, _patterns, _structure, _position, _bank_nifty, _cross_index, _order],
    )
    allowed = [*CONTEXT_TOOL_NAMES, f"mcp__{SERVER_NAME}__{order_name}"]
    return {SERVER_NAME: server}, allowed
