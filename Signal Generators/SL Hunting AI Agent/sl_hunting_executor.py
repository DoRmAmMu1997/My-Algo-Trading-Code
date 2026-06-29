"""Trade-execution seam for the SL Hunting AI Agent.

Beginner note (why this file exists)
------------------------------------
The agent decides WHAT to do (enter long/short, exit). WHERE that order goes —
paper bookkeeping, or a real Kotak / Shoonya order through the master's shared
broker session — is a separate concern. This module is that seam.

The agent core never imports a broker. Instead it is handed ONE `TradeExecutor`
(dependency injection, exactly like the Streamlit Scanner App injects a `runner`):

- `StandaloneExecutor`  — paper-only bookkeeping for the folder's standalone runner
  (`sl_hunting_runner.py`). P&L is a simple proxy on the NIFTY underlying, which is
  fine for validating the agent's direction/timing without an option chain.
- `MasterWorkerExecutor` — used inside the master front-test. It delegates to the
  worker's existing, safe `enter_position` / `exit_position` methods, which already
  resolve the ATM option, place paper OR real orders (via `_place_real_leg`),
  enforce max-loss / square-off, and publish Telegram alerts. The live-vs-paper and
  Kotak-vs-Shoonya decision therefore stays exactly where the rest of the system
  makes it — nothing new is invented here.

SAFETY: the agent is only ever given the SINGLE execution tool the configuration
selected (see `execution_tool_name`). It can never choose paper-vs-real or the
broker — the environment does. Every executor guards state: you cannot ENTER while
already in a position, and cannot EXIT while flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

# Valid directions the agent may request. LONG → buy CE (expect up); SHORT → buy PE.
VALID_DIRECTIONS = ("LONG", "SHORT")

# Default rupee risk budget per trade (used by dynamic position sizing). The agent
# does NOT choose the lot count — it is computed so the worst-case underlying-points
# loss is ~= this budget, exactly like the repo's Profit Shooter sizer.
DEFAULT_RISK_BUDGET = 2500.0


def risk_based_lots(
    entry: float,
    stop: float,
    lot_size: int,
    risk_budget: float = DEFAULT_RISK_BUDGET,
    fallback_lots: int = 1,
) -> int:
    """Lots sized so the worst-case underlying-points loss ~= ``risk_budget``.

    ``risk_points`` is the agent's UNDERLYING (spot) stop distance; the rupee risk
    proxy is ``risk_points * lot_size`` per lot (a conservative delta~=1 proxy, the
    same one Profit Shooter / Goldmine / Money Machine use). ``math.ceil`` over a
    positive ratio guarantees a minimum of 1 lot, so a trade is never sized to zero.
    Falls back to ``fallback_lots`` when the stop distance is unusable.
    """
    risk_points = abs(float(entry) - float(stop))
    if risk_points <= 0 or int(lot_size) <= 0:
        return max(1, int(fallback_lots))
    return max(1, math.ceil(float(risk_budget) / (risk_points * int(lot_size))))


def stop_or_target_hit(direction: str, stop: float, target: float, price: float) -> str | None:
    """Return 'AI_STOP' / 'AI_TARGET' / None for a spot `price` vs the open position's
    underlying stop/target levels.

    Used by the master worker to ENFORCE the agent's stop/target on every poll — the
    agent only re-decides once per completed bar, so without this an option position
    would stay open (only max-loss / square-off / a later AI EXIT would close it) and
    the ~Rs.2500 risk budget would not be bounded. Zero levels mean "not set" (skip).
    """
    direction = (direction or "").strip().upper()
    price = float(price)
    s, t = float(stop or 0.0), float(target or 0.0)
    if direction == "LONG":
        if s and price <= s:
            return "AI_STOP"
        if t and price >= t:
            return "AI_TARGET"
    elif direction == "SHORT":
        if s and price >= s:
            return "AI_STOP"
        if t and price <= t:
            return "AI_TARGET"
    return None


def execution_tool_name(live_active: bool, broker: str | None) -> str:
    """Return the single order-tool name the configuration selects.

    This is the literal realisation of "three tools, the env picks one":
    - not live                     → ``place_paper_order``
    - live + LIVE_BROKER=KOTAK     → ``place_kotak_order``
    - live + LIVE_BROKER=SHOONYA   → ``place_shoonya_order``

    Only this one name is ever registered, so the agent cannot pick the venue.
    """
    if not live_active:
        return "place_paper_order"
    broker_up = (broker or "").strip().upper()
    if broker_up == "KOTAK":
        return "place_kotak_order"
    if broker_up == "SHOONYA":
        return "place_shoonya_order"
    # Fail closed: unknown broker → paper (mirrors the master's broker fail-closed).
    return "place_paper_order"


@runtime_checkable
class TradeExecutor(Protocol):
    """The minimal surface the execution tool needs. Implemented two ways below."""

    def enter(self, direction: str, stop: float, target: float, reason: str, price: float) -> dict[str, Any]:
        """Open a position (or reject if already in one). Returns a result dict."""
        ...

    def exit(self, reason: str, price: float) -> dict[str, Any]:
        """Close the open position (or reject if flat). Returns a result dict."""
        ...

    def snapshot(self) -> dict[str, Any]:
        """Return the current position state for the `position_state` tool."""
        ...


# ---------------------------------------------------------------------------
# Standalone (paper) executor
# ---------------------------------------------------------------------------

@dataclass
class _PaperPosition:
    active: bool = False
    direction: str = ""
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    entry_time: str = ""
    reason: str = ""
    lots: int = 0


class StandaloneExecutor:
    """Paper bookkeeping for the standalone runner.

    P&L is a proxy on the underlying: ``(last - entry)`` for LONG and
    ``(entry - last)`` for SHORT, in NIFTY points × lots × lot_size. This is
    deliberately simple — it validates the agent's *decisions*, not option pricing.
    The closed-trade log is kept so the runner can print a session summary.
    """

    def __init__(self, *, lots: int = 1, lot_size: int = 75, risk_budget: float = DEFAULT_RISK_BUDGET) -> None:
        self.lots = int(lots)  # fallback when the stop distance is unusable
        self.lot_size = int(lot_size)
        self.risk_budget = float(risk_budget)
        self.pos = _PaperPosition()
        self.closed_trades: list[dict[str, Any]] = []

    def enter(self, direction: str, stop: float, target: float, reason: str, price: float) -> dict[str, Any]:
        """Open a paper position at ``price`` (rejects a bad direction or a double-entry).

        Records the entry/stop/target and the dynamically-sized lot count, then returns
        an ``accepted`` result dict the order tool relays back to the agent.
        """
        direction = (direction or "").strip().upper()
        if direction not in VALID_DIRECTIONS:
            return {"accepted": False, "reason": f"invalid direction {direction!r}; expected LONG or SHORT"}
        if self.pos.active:
            return {"accepted": False, "reason": "already in a position; EXIT first before entering again"}
        # Dynamic sizing: lots chosen so worst-case risk ~= self.risk_budget (Rs.2500).
        lots = risk_based_lots(float(price), float(stop), self.lot_size, self.risk_budget, self.lots)
        self.pos = _PaperPosition(
            active=True,
            direction=direction,
            entry=round(float(price), 2),
            stop=round(float(stop), 2),
            target=round(float(target), 2),
            entry_time=datetime.now().isoformat(timespec="seconds"),
            reason=str(reason)[:300],
            lots=lots,
        )
        return {"accepted": True, "action": f"ENTER_{direction}", "entry": self.pos.entry, "stop": self.pos.stop, "target": self.pos.target, "lots": lots, "quantity": lots * self.lot_size}

    def exit(self, reason: str, price: float) -> dict[str, Any]:
        """Close the open position at ``price``, append it to the trade log, go flat.

        P&L is the simple underlying-points proxy (LONG profits as price rises, SHORT as
        it falls) × lots × lot_size. Rejects if there is nothing open to close.
        """
        if not self.pos.active:
            return {"accepted": False, "reason": "no open position to exit"}
        last = float(price)
        # LONG makes money when price rises above entry; SHORT when it falls below entry.
        pts = (last - self.pos.entry) if self.pos.direction == "LONG" else (self.pos.entry - last)
        lots = self.pos.lots or self.lots
        pnl = round(pts * lots * self.lot_size, 2)
        trade = {
            "direction": self.pos.direction,
            "entry": self.pos.entry,
            "exit": round(last, 2),
            "points": round(pts, 2),
            "lots": lots,
            "pnl_proxy": pnl,
            "entry_time": self.pos.entry_time,
            "exit_time": datetime.now().isoformat(timespec="seconds"),
            "exit_reason": str(reason)[:300],
        }
        self.closed_trades.append(trade)
        self.pos = _PaperPosition()
        return {"accepted": True, "action": "EXIT", **trade}

    def snapshot(self) -> dict[str, Any]:
        """Return the current position as a dict for the `position_state` tool (or flat)."""
        if not self.pos.active:
            return {"in_position": False}
        return {
            "in_position": True,
            "direction": self.pos.direction,
            "entry": self.pos.entry,
            "stop": self.pos.stop,
            "target": self.pos.target,
            "entry_time": self.pos.entry_time,
        }

    def mark(self, price: float) -> dict[str, Any]:
        """Return the live unrealised P&L proxy for the open position (helper)."""
        if not self.pos.active:
            return {"in_position": False}
        pts = (price - self.pos.entry) if self.pos.direction == "LONG" else (self.pos.entry - price)
        lots = self.pos.lots or self.lots
        return {"in_position": True, "points": round(pts, 2), "lots": lots, "pnl_proxy": round(pts * lots * self.lot_size, 2)}


# ---------------------------------------------------------------------------
# Master-worker executor (live-capable; delegates to the worker)
# ---------------------------------------------------------------------------

class MasterWorkerExecutor:
    """Delegates to an `AtmSingleLegStrategyWorker` instance in the master runner.

    The worker already owns the safe path: `enter_position(direction,
    entry_underlying, stop_underlying, target_underlying)` resolves the ATM
    option and places a paper OR real order (per `worker.live_trading` +
    `LIVE_BROKER`), and `exit_position(reason)` closes it — both with max-loss /
    square-off / Telegram handled. We duck-type the worker so this module never
    imports the master.
    """

    def __init__(self, worker: Any) -> None:
        self._w = worker

    def enter(self, direction: str, stop: float, target: float, reason: str, price: float) -> dict[str, Any]:
        """Delegate the entry to the master worker's safe `enter_position`.

        The worker resolves the ATM option and places the paper/real order; we only
        guard against an invalid direction or entering while already in a position, and
        translate the worker's True/False into the result dict the order tool expects.
        """
        direction = (direction or "").strip().upper()
        if direction not in VALID_DIRECTIONS:
            return {"accepted": False, "reason": f"invalid direction {direction!r}; expected LONG or SHORT"}
        pos = getattr(self._w, "pos", None)
        if pos is not None and getattr(pos, "active", False):
            return {"accepted": False, "reason": "already in a position; EXIT first before entering again"}
        ok = bool(self._w.enter_position(
            direction,
            float(price),
            stop_underlying=float(stop),
            target_underlying=float(target),
        ))
        if not ok:
            return {"accepted": False, "reason": "worker rejected the entry (e.g. could not resolve option / outside trading window)"}
        return {"accepted": True, "action": f"ENTER_{direction}", "entry": round(float(price), 2), "stop": round(float(stop), 2), "target": round(float(target), 2)}

    def exit(self, reason: str, price: float) -> dict[str, Any]:
        """Delegate the close to the worker's `exit_position` (which handles P&L/alerts)."""
        pos = getattr(self._w, "pos", None)
        if pos is None or not getattr(pos, "active", False):
            return {"accepted": False, "reason": "no open position to exit"}
        self._w.exit_position(str(reason)[:300] or "AI_EXIT")
        return {"accepted": True, "action": "EXIT", "reason": str(reason)[:300]}

    def snapshot(self) -> dict[str, Any]:
        """Read the worker's open position (duck-typed) into the `position_state` shape.

        The worker stores entry/stop/target on its ``pos`` object as ``*_underlying``
        attributes; we copy those across and best-effort attach unrealised P&L if the
        worker exposes a getter for it.
        """
        pos = getattr(self._w, "pos", None)
        if pos is None or not getattr(pos, "active", False):
            return {"in_position": False}
        snap: dict[str, Any] = {
            "in_position": True,
            "direction": getattr(pos, "direction", ""),
            "entry": round(float(getattr(pos, "entry_underlying", 0.0) or 0.0), 2),
            "stop": round(float(getattr(pos, "stop_underlying", 0.0) or 0.0), 2),
            "target": round(float(getattr(pos, "target_underlying", 0.0) or 0.0), 2),
        }
        getter = getattr(self._w, "_get_open_position_pnl", None)
        if callable(getter):
            try:
                snap["unrealized_pnl"] = round(float(getter()), 2)
            except Exception:  # noqa: BLE001 - best-effort enrichment only
                pass
        return snap
