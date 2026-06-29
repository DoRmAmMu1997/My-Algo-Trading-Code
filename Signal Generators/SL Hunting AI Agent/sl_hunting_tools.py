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
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sl_hunting_indicators import (
    SLHuntingIndicatorConfig,
    candle_patterns,
    cross_index_signal,
    fibo_levels,
    market_structure,
    pivot_and_levels,
    prepare_candles,
)
from sl_hunting_executor import TradeExecutor, execution_tool_name

SERVER_NAME = "slhunting"

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
    ) -> "SLHuntingToolContext":
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

    def do_order(self, action: str, stop: float, target: float, reason: str) -> dict[str, Any]:
        """Route an agent order through the injected executor (single source of truth)."""
        action = (action or "").strip().upper()
        if action == "ENTER_LONG":
            return self.executor.enter("LONG", float(stop or 0.0), float(target or 0.0), reason, self.last_price)
        if action == "ENTER_SHORT":
            return self.executor.enter("SHORT", float(stop or 0.0), float(target or 0.0), reason, self.last_price)
        if action == "EXIT":
            return self.executor.exit(reason, self.last_price)
        return {"accepted": False, "reason": f"unknown action {action!r}; expected ENTER_LONG, ENTER_SHORT or EXIT"}


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

    @tool("pivot_and_levels", "Day pivot, previous-day OHLC, today's O/H/L, first-candle hi/lo, psych levels and the previous close, with distances from current price.", {})
    async def _pivot(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.pivot_and_levels_payload())

    @tool("fibo_levels", "Fibonacci 50/61/78 retracement and 161/261 extension levels of the most recent significant swing, and where price sits relative to them.", {})
    async def _fibo(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.fibo_levels_payload())

    @tool("candle_patterns", "Reversal candlestick patterns (hammer/doji, engulfing, inside-bar, reversal-bar, star) on recent candles, each tagged with whether a confirmation candle has already printed.", {})
    async def _patterns(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.candle_patterns_payload())

    @tool("market_structure", "Swing points, trend, fast/slow read, trendline points, and W/M / double top-bottom detection.", {})
    async def _structure(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.market_structure_payload())

    @tool("position_state", "Your current open position (direction, entry, stop, target, unrealised P&L) or 'flat'.", {})
    async def _position(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.position_state_payload())

    @tool("bank_nifty", "BankNIFTY's OWN pivot/levels, market structure and recent candle patterns (for cross-confirmation). Returns available:false when BankNIFTY data is unavailable.", {})
    async def _bank_nifty(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.bank_nifty_payload())

    @tool("cross_index", "NIFTY-vs-BankNIFTY alignment verdict per the NF/BNF rules (both at support -> bias down; both breakdown -> bias up; one fails -> the holder wins; opposite sides of pivot -> wait). Advisory. available:false when BankNIFTY data is unavailable.", {})
    async def _cross_index(_args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_text(context.cross_index_payload())

    order_name = context.action_tool_name()
    venue = "PAPER" if order_name == "place_paper_order" else ("KOTAK (LIVE)" if order_name == "place_kotak_order" else "SHOONYA (LIVE)")

    @tool(
        order_name,
        f"Place an SL-Hunting order on NIFTY ATM options via the {venue} venue (the "
        "configuration chose this venue; you cannot change it). action is ENTER_LONG "
        "(buy CALL), ENTER_SHORT (buy PUT) or EXIT. stop and target are NIFTY "
        "UNDERLYING price levels (required for entries; ignored for EXIT). reason is "
        "a one-line justification. Returns whether the order was accepted or rejected.",
        {"action": str, "stop": float, "target": float, "reason": str},
    )
    async def _order(args: dict[str, Any]) -> dict[str, Any]:
        result = context.do_order(
            args.get("action", ""),
            args.get("stop", 0.0),
            args.get("target", 0.0),
            args.get("reason", ""),
        )
        return _as_tool_text(result)

    server = create_sdk_mcp_server(
        name=SERVER_NAME,
        version="1.0.0",
        tools=[_pivot, _fibo, _patterns, _structure, _position, _bank_nifty, _cross_index, _order],
    )
    allowed = list(CONTEXT_TOOL_NAMES) + [f"mcp__{SERVER_NAME}__{order_name}"]
    return {SERVER_NAME: server}, allowed
