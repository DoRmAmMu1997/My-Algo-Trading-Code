"""Detect the Trend-Day Rider candidate from completed five-minute bars.

This is the deterministic core of the CPR AI strategy. The host proposes a
trade only when this module says the newest completed bar qualifies; Codex may
then accept or veto it, but never loosen it. The same functions drive the live
context, the host policy, and ``cpr_ai_trend_day_backtest.py``, so the backtest
is an honest baseline for the agent.

The thesis, measured on five years of NIFTY one-minute data and real weekly
option premiums: by late morning, a session that has already out-ranged its
recent average and is pinned at one extreme, on the trend side of VWAP, is a
trend day -- and trend days tend to close near their extremes. The strategy
sells the ATM option on the side the market is leaving (bullish -> sell PE,
bearish -> sell CE), with the entry bar's VWAP as a hard spot stop.

The module is deliberately pure: pandas/numpy only, no pydantic, broker, clock,
or environment access. That keeps it importable by the backtest without the
optional AI stack and trivially unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

LONG = "LONG"
SHORT = "SHORT"


@dataclass(frozen=True)
class TrendDayConfig:
    """Hold the backtested constants that define a trend-day candidate.

    These are code-owned invariants, not ``.env`` knobs: the backtest numbers
    only describe this exact definition. Window bounds are *bar start* times
    and both are inclusive (11:00 through 13:30 means the bars that close at
    11:05 through 13:35). ``range_atr_multiple`` is a strict inequality.
    """

    window_start: time = time(11, 0)
    window_end: time = time(13, 30)
    range_atr_multiple: float = 1.0
    location_threshold: float = 0.85
    atr_sessions: int = 5
    min_atr_sessions: int = 3
    extension_atr_multiple: float = 0.35
    long_min_confluence: int = 2
    short_min_confluence: int = 0

    def __post_init__(self) -> None:
        """Reject impossible settings before they can shape a live candidate."""

        if self.window_start > self.window_end:
            raise ValueError("window_start must not be after window_end.")
        for name in ("range_atr_multiple", "extension_atr_multiple"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if not 0.5 < self.location_threshold < 1.0:
            raise ValueError("location_threshold must be between 0.5 and 1.0.")
        if not 1 <= self.min_atr_sessions <= self.atr_sessions:
            raise ValueError("min_atr_sessions must be between 1 and atr_sessions.")
        for name in ("long_min_confluence", "short_min_confluence"):
            if not 0 <= getattr(self, name) <= 3:
                raise ValueError(f"{name} must be between 0 and 3.")


DEFAULT_TREND_DAY_CONFIG = TrendDayConfig()


@dataclass(frozen=True)
class TrendDayAssessment:
    """Describe whether the newest bar is a candidate, and why.

    Every measurable fact is filled in even when the bar is rejected, because
    the model reads them as context and the decision log keeps them for later
    counterfactual review. ``eligible`` is the only field that authorizes the
    host to consider an entry; ``reason`` names the first gate that failed.
    """

    eligible: bool
    reason: str
    direction: str | None
    bar_start: str | None
    in_window: bool
    entry: float | None
    stop: float | None
    risk_points: float | None
    session_high: float | None
    session_low: float | None
    session_range: float | None
    atr: float | None
    atr_sessions_used: int
    range_atr: float | None
    location: float | None
    vwap: float | None
    distance_from_vwap_atr: float | None
    r1: float | None
    s1: float | None
    beyond_r1_s1: bool
    gap_in_direction: bool
    extended_from_vwap: bool
    confluence_score: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy for the frozen context and audit log."""

        return asdict(self)


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/NaN/infinite input."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def session_vwap(session_bars: pd.DataFrame) -> pd.Series:
    """Return the session VWAP after each completed bar.

    Volume weighting is used when the feed supplies any volume. NIFTY index
    candles carry none, so the normal path is the expanding mean of typical
    price -- the same proxy the rest of the CPR AI context has always used.
    """

    typical = (session_bars["high"] + session_bars["low"] + session_bars["close"]) / 3.0
    volume = (
        pd.to_numeric(session_bars["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "volume" in session_bars
        else pd.Series(0.0, index=session_bars.index)
    )
    if float(volume.sum()) > 0.0:
        return (typical * volume).cumsum() / volume.cumsum()
    return typical.expanding().mean()


def session_atr(
    prior_ranges: Sequence[float],
    config: TrendDayConfig = DEFAULT_TREND_DAY_CONFIG,
) -> tuple[float | None, int]:
    """Average the most recent prior-session high-low ranges.

    ``prior_ranges`` is oldest-first. Only positive finite ranges count, the
    newest ``atr_sessions`` of them are averaged, and fewer than
    ``min_atr_sessions`` yields ``None`` (no candidate can form). The live
    store keeps about five sessions, which is why five is the default.
    """

    usable = [value for value in (_finite(item) for item in prior_ranges) if value is not None and value > 0]
    recent = usable[-config.atr_sessions :]
    if len(recent) < config.min_atr_sessions:
        return None, len(recent)
    return float(np.mean(recent)), len(recent)


def evaluate_trend_day_candidate(
    session_bars: pd.DataFrame,
    *,
    prior_high: float,
    prior_low: float,
    prior_close: float,
    atr: float | None,
    atr_sessions_used: int = 0,
    config: TrendDayConfig = DEFAULT_TREND_DAY_CONFIG,
) -> TrendDayAssessment:
    """Assess the newest completed bar of the current session.

    ``session_bars`` holds only today's completed five-minute bars, oldest
    first, with ``timestamp`` (bar start) and OHLC columns. Gates run in a
    fixed order -- window, ATR, range expansion, extreme location on the VWAP
    side, then the bullish-only confluence requirement -- and the first
    failure names the rejection. Entry is the bar close and the stop is that
    bar's VWAP, so a long always has VWAP below it and a short above it.
    """

    empty = TrendDayAssessment(
        eligible=False, reason="no_session_bars", direction=None, bar_start=None, in_window=False,
        entry=None, stop=None, risk_points=None, session_high=None, session_low=None,
        session_range=None, atr=_finite(atr), atr_sessions_used=atr_sessions_used, range_atr=None,
        location=None, vwap=None, distance_from_vwap_atr=None, r1=None, s1=None,
        beyond_r1_s1=False, gap_in_direction=False, extended_from_vwap=False, confluence_score=0,
    )
    if session_bars.empty:
        return empty
    bars = session_bars.reset_index(drop=True)
    newest = bars.iloc[-1]
    bar_start = pd.Timestamp(newest["timestamp"])
    in_window = config.window_start <= bar_start.time() <= config.window_end
    close = float(newest["close"])
    high = float(bars["high"].max())
    low = float(bars["low"].min())
    session_range = high - low
    vwap = float(session_vwap(bars).iloc[-1])
    atr_value = _finite(atr)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    r1 = 2.0 * pivot - prior_low
    s1 = 2.0 * pivot - prior_high
    location = (close - low) / session_range if session_range > 0 else None
    range_atr = session_range / atr_value if atr_value else None
    distance_atr = abs(close - vwap) / atr_value if atr_value else None

    # Direction comes from location plus the VWAP side together; either alone
    # is not a trend-day signature.
    direction: str | None = None
    if location is not None and location >= config.location_threshold and close > vwap:
        direction = LONG
    elif location is not None and location <= 1.0 - config.location_threshold and close < vwap:
        direction = SHORT

    session_open = float(bars.iloc[0]["open"])
    beyond = bool(direction == LONG and close > r1) or bool(direction == SHORT and close < s1)
    gap = bool(direction == LONG and session_open > prior_close) or bool(
        direction == SHORT and session_open < prior_close
    )
    extended = bool(direction is not None and distance_atr is not None and distance_atr > config.extension_atr_multiple)
    score = int(beyond) + int(gap) + int(extended)

    reason = "eligible"
    if not in_window:
        reason = "outside_window"
    elif atr_value is None:
        reason = "atr_unavailable"
    elif not session_range > config.range_atr_multiple * atr_value:
        reason = "range_not_expanded"
    elif direction is None:
        reason = "not_at_trend_extreme"
    elif score < (config.long_min_confluence if direction == LONG else config.short_min_confluence):
        reason = "confluence_too_low"
    eligible = reason == "eligible"
    return TrendDayAssessment(
        eligible=eligible,
        reason=reason,
        direction=direction,
        bar_start=bar_start.isoformat(),
        in_window=in_window,
        entry=close,
        stop=vwap if direction is not None else None,
        risk_points=abs(close - vwap) if direction is not None else None,
        session_high=high,
        session_low=low,
        session_range=session_range,
        atr=atr_value,
        atr_sessions_used=atr_sessions_used,
        range_atr=range_atr,
        location=location,
        vwap=vwap,
        distance_from_vwap_atr=distance_atr,
        r1=r1,
        s1=s1,
        beyond_r1_s1=beyond,
        gap_in_direction=gap,
        extended_from_vwap=extended,
        confluence_score=score,
    )


__all__ = [
    "DEFAULT_TREND_DAY_CONFIG",
    "LONG",
    "SHORT",
    "TrendDayAssessment",
    "TrendDayConfig",
    "evaluate_trend_day_candidate",
    "session_atr",
    "session_vwap",
]
