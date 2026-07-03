"""Trade journal for the SL Hunting AI Agent (v3 — learn from mistakes).

Records one structured row per COMPLETED trade: the ENTRY context (the agent's
decision + a deterministic snapshot of pivot/levels, fibo, the confirmed candle
pattern, and the NF/BNF cross-index read) and the EXIT outcome (reason, underlying
points, R-multiple, option P&L). The reflection coach (`sl_hunting_coach.py`) reads
this JSONL to propose lessons.

Driven by the worker (master) and the runner (standalone) — both already hold the
NIFTY/BNF frames and the returned decision — so the agent itself stays
journal-agnostic. **Never raises into the trading loop**: all I/O is best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from sl_hunting_indicators import (
    SLHuntingIndicatorConfig,
    candle_patterns,
    cross_index_signal,
    fibo_levels,
    pivot_and_levels,
)

logger = logging.getLogger(__name__)


def build_entry_context(
    nifty_df: pd.DataFrame,
    bnf_df: pd.DataFrame | None = None,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """A compact, deterministic snapshot of the facts at entry (reuses the detectors).

    Kept small (not the full tool payloads) so a journal row stays readable: pivot +
    distances, the nearest fibo retracement, the recent confirmed candle patterns, and
    the NF/BNF cross-index verdict. This is the *context the agent acted in* — what the
    coach correlates with outcomes to find what works and what doesn't.
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    pl = pivot_and_levels(nifty_df, cfg)
    fib = fibo_levels(nifty_df, cfg)
    pat = candle_patterns(nifty_df, cfg)
    xidx = cross_index_signal(nifty_df, bnf_df, cfg) if bnf_df is not None else {"available": False}
    confirmed = [
        {"type": p.get("type"), "direction": p.get("direction"), "bars_ago": p.get("bars_ago")}
        for p in (pat.get("confirmed_patterns") or [])
    ][-3:]
    dist = pl.get("distance_to") or {}
    return {
        "last_price": pl.get("last_price"),
        "pivot": pl.get("pivot"),
        "vs_pivot": dist.get("pivot"),
        "nearest_psych": dist.get("nearest_psych"),
        "nearest_fibo": fib.get("nearest_retracement") if fib.get("available") else None,
        "confirmed_patterns": confirmed,
        "cross_index": (
            {"alignment": xidx.get("alignment"), "bias": xidx.get("bias")}
            if xidx.get("available") else {"available": False}
        ),
    }


def followed_method(context: dict[str, Any], direction: str) -> bool:
    """Process check: was a confirmed pattern present in the trade's direction at entry?

    Separates *process* from *outcome* so the coach doesn't mislabel a sound,
    rule-following setup that simply lost (variance) as a "mistake".
    """
    want = "bullish" if (direction or "").strip().upper() == "LONG" else "bearish"
    return any(p.get("direction") == want for p in context.get("confirmed_patterns", []))


def make_entry_record(
    *,
    direction: str,
    setup: str,
    confidence: int,
    stop: float,
    target: float,
    reasoning: str,
    entry_underlying: float,
    lots: int,
    nifty_df: pd.DataFrame,
    bnf_df: pd.DataFrame | None = None,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Assemble the ENTRY half of a journal row (shared by worker + runner)."""
    ctx = build_entry_context(nifty_df, bnf_df, cfg)
    return {
        "direction": direction,
        "setup": setup,
        "confidence": confidence,
        "stop": round(float(stop or 0.0), 2),
        "target": round(float(target or 0.0), 2),
        "reasoning": str(reasoning)[:500],
        "entry_underlying": round(float(entry_underlying or 0.0), 2),
        "lots": int(lots or 0),
        "context": ctx,
        "followed_method": followed_method(ctx, direction),
    }


def make_decision_record(
    decision: Any,
    nifty_df: pd.DataFrame,
    bnf_df: pd.DataFrame | None = None,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Build one AUDIT row for a SINGLE agent decision — HOLD included.

    For a SEPARATE decisions log, NOT the trade journal: it captures what the agent
    decided this bar (action / confidence / setup / reasoning + the deterministic
    context it saw), so an operator can review every decision, including the HOLDs.
    The trade journal stays trade-only so the reflection coach's win/loss stats are
    never diluted by the ~70 no-trade bars a day.
    """
    ctx = build_entry_context(nifty_df, bnf_df, cfg)
    return {
        "decided_at": datetime.now().isoformat(timespec="seconds"),
        "action": getattr(decision, "action", ""),
        "confidence": getattr(decision, "confidence", None),
        "setup": getattr(decision, "setup", ""),
        "stop": round(float(getattr(decision, "stop", 0.0) or 0.0), 2),
        "target": round(float(getattr(decision, "target", 0.0) or 0.0), 2),
        "reasoning": str(getattr(decision, "reasoning", ""))[:500],
        "model_used": getattr(decision, "model_used", ""),
        "context": ctx,
    }


def append_decision(path: str, record: dict[str, Any]) -> None:
    """Append one decision row to the decisions JSONL (best-effort; never raises).

    A plain append (not via TradeJournal) because a decision has no open/close
    lifecycle — it is one complete row written immediately, every bar.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.warning("Could not append to decisions log %s", path, exc_info=True)


class TradeJournal:
    """Append-only JSONL journal — one complete row per closed trade.

    Open records are held in memory and the full (entry + outcome) row is written on
    close, so each line is one finished trade. A crash mid-trade loses only that one
    pending journal row (never the position itself, which the worker still manages).
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._open: dict[str, dict[str, Any]] = {}
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except OSError:
            logger.warning("Could not create journal directory for %s", self.path, exc_info=True)

    def open_trade(self, entry: dict[str, Any]) -> str:
        """Record a new open trade; returns its id (pass to `close_trade`)."""
        trade_id = uuid.uuid4().hex[:12]
        record = dict(entry)
        record["trade_id"] = trade_id
        record.setdefault("opened_at", datetime.now().isoformat(timespec="seconds"))
        self._open[trade_id] = record
        return trade_id

    def close_trade(self, trade_id: str, outcome: dict[str, Any]) -> dict[str, Any] | None:
        """Merge the outcome, compute the R-multiple, and append the finished row."""
        entry = self._open.pop(trade_id, None)
        if entry is None:
            logger.warning("TradeJournal.close_trade: unknown trade_id %r", trade_id)
            return None
        out = dict(outcome)
        out.setdefault("closed_at", datetime.now().isoformat(timespec="seconds"))
        # R-multiple = profit/loss measured in UNITS OF RISK: points won/lost ÷ the
        # entry's stop distance. +2R means we made twice what we risked; -1R means the
        # stop was hit. It normalises trades of different sizes so the coach can compare
        # them fairly (a +30pt win on a 10pt stop = +3R beats a +30pt win on a 30pt stop).
        stop_dist = abs(float(entry.get("entry_underlying") or 0.0) - float(entry.get("stop") or 0.0))
        pts = out.get("points")
        if stop_dist > 0 and pts is not None:
            out["r_multiple"] = round(float(pts) / stop_dist, 2)
        row = {**entry, "outcome": out}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            logger.warning("Could not append to journal %s", self.path, exc_info=True)
        return row

    @property
    def open_count(self) -> int:
        """How many trades are opened-but-not-yet-closed (held in memory)."""
        return len(self._open)
