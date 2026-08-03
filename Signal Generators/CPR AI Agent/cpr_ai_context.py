"""Compute a self-contained, deterministic CPR/SRSI/VWAP context.

This module intentionally does not import the older CPR Strategy package.  It
accepts ordinary one-minute OHLC history, retains prior sessions for level
math, and exposes facts rather than choosing an agent regime or trade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from cpr_ai_schema import validate_position_state

_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
_LEVEL_BUFFER_POINTS = 2.0


def _as_float(value: Any) -> float | None:
    """Return a JSON-safe finite float, or ``None`` when a value is unavailable."""

    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _prepared_minutes(one_minute_candles: pd.DataFrame) -> pd.DataFrame:
    """Copy, type-check, and chronologically sort one-minute OHLC observations."""

    missing = [column for column in _REQUIRED_COLUMNS if column not in one_minute_candles.columns]
    if missing:
        raise ValueError(f"One-minute CPR context is missing columns: {missing}")
    frame = one_minute_candles.loc[:, [*one_minute_candles.columns]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def build_completed_five_minute_bars(one_minute_candles: pd.DataFrame) -> pd.DataFrame:
    """Build only exact five-observation OHLC buckets from one-minute candles.

    A bucket containing fewer than five one-minute records is deliberately
    discarded.  This prevents an in-progress five-minute candle from being
    mistaken for final market evidence.
    """

    minutes = _prepared_minutes(one_minute_candles)
    completed: list[pd.DataFrame] = []
    # Resample per calendar session so a 15:29 observation cannot be paired
    # with a new day's 09:15 observation in an accidental cross-day bucket.
    for _, session in minutes.groupby(minutes["timestamp"].dt.date, sort=True):
        indexed = session.set_index("timestamp")
        buckets = indexed.resample("5min", label="left", closed="left", origin="start_day")
        result = buckets.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            _count=("close", "count"),
        )
        completed.append(result.loc[result["_count"] == 5].drop(columns="_count").reset_index())
    if not completed:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.concat(completed, ignore_index=True)


def _rsi_wilder(closes: pd.Series, length: int = 14) -> pd.Series:
    """Calculate TradingView-style Wilder RSI from a close series."""

    changes = closes.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    # Wilder's RMA is an EMA with alpha=1/length, rather than the usual EMA
    # alpha=2/(length+1).  ``min_periods`` avoids claiming early values exist.
    average_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    average_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    # Flat loss-free periods conventionally read as 100; fully flat periods are
    # neutral 50 instead of a misleading overbought signal.
    return rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0).mask(
        (average_loss == 0.0) & (average_gain == 0.0), 50.0
    )


def _stochastic_rsi(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return RSI14 and TradingView-style Stochastic RSI K/D (14, 3, 3)."""

    rsi = _rsi_wilder(closes, 14)
    lowest = rsi.rolling(14, min_periods=14).min()
    highest = rsi.rolling(14, min_periods=14).max()
    denominator = highest - lowest
    raw = ((rsi - lowest) / denominator.replace(0.0, np.nan)) * 100.0
    k = raw.rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()
    return rsi, k, d


def _opening_facts(session_bars: pd.DataFrame, count: int) -> dict[str, Any]:
    """Return a completed first-N-minute OHLC corridor or a clear incomplete marker."""

    bars_needed = count // 5
    window = session_bars.head(bars_needed)
    if len(window) != bars_needed:
        return {"complete": False, "minutes": count}
    return {
        "complete": True,
        "minutes": count,
        "open": _as_float(window.iloc[0]["open"]),
        "high": _as_float(window["high"].max()),
        "low": _as_float(window["low"].min()),
        "close": _as_float(window.iloc[-1]["close"]),
        "range": _as_float(window["high"].max() - window["low"].min()),
    }


def _level_view(name: str, price: float, current_close: float) -> dict[str, Any]:
    """Expose a named level with signed and absolute distance from the close."""

    return {
        "name": name,
        "price": price,
        "distance_from_close": price - current_close,
        "abs_distance": abs(price - current_close),
    }


def _session_levels(minutes: pd.DataFrame, session_bars: pd.DataFrame) -> dict[str, Any]:
    """Calculate prior-day CPR, support/resistance, opening, and next-level facts."""

    sessions = sorted(minutes["timestamp"].dt.date.unique())
    if len(sessions) < 2:
        raise ValueError("CPR context needs one complete prior session and one current session.")
    current_session = sessions[-1]
    previous = minutes.loc[minutes["timestamp"].dt.date == sessions[-2]]
    current_close = float(session_bars.iloc[-1]["close"])
    prior_high, prior_low, prior_close = (
        float(previous["high"].max()),
        float(previous["low"].min()),
        float(previous.iloc[-1]["close"]),
    )
    pivot = (prior_high + prior_low + prior_close) / 3.0
    bc_raw = (prior_high + prior_low) / 2.0
    tc_raw = (2.0 * pivot) - bc_raw
    cpr_lower, cpr_upper = sorted((bc_raw, tc_raw))
    levels = {
        "pivot": pivot,
        "bc": bc_raw,
        "tc": tc_raw,
        "cpr_lower": cpr_lower,
        "cpr_upper": cpr_upper,
        "r1": (2.0 * pivot) - prior_low,
        "r2": pivot + (prior_high - prior_low),
        "s1": (2.0 * pivot) - prior_high,
        "s2": pivot - (prior_high - prior_low),
    }
    ordered = sorted(
        (_level_view(name, price, current_close) for name, price in levels.items()), key=lambda item: item["price"]
    )
    above = [item for item in ordered if item["price"] > current_close + _LEVEL_BUFFER_POINTS]
    below = [item for item in ordered if item["price"] < current_close - _LEVEL_BUFFER_POINTS]
    return {
        "session_date": str(current_session),
        "current_close": current_close,
        "previous_day": {"high": prior_high, "low": prior_low, "close": prior_close},
        "levels": levels,
        "opening": {
            "opening_corridor": _opening_facts(session_bars, 5),
            "first_15_minutes": _opening_facts(session_bars, 15),
            "first_30_minutes": _opening_facts(session_bars, 30),
        },
        "distances_from_current_close": {name: price - current_close for name, price in levels.items()},
        "next_levels": {
            "buffer_points": _LEVEL_BUFFER_POINTS,
            "upside": above[0] if above else None,
            "downside": below[-1] if below else None,
            "ordered": ordered,
        },
    }


def _momentum_vwap(session_bars: pd.DataFrame) -> dict[str, Any]:
    """Calculate SRSI, RSI, EMA, candle, and session VWAP evidence from bars."""

    bars = session_bars.copy()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    volume = bars["volume"].fillna(0.0).clip(lower=0.0)
    if float(volume.sum()) > 0.0:
        vwap = (typical * volume).cumsum() / volume.cumsum()
        vwap_method = "volume_weighted"
    else:
        vwap = typical.expanding().mean()
        vwap_method = "equal_weight_typical_price"
    rsi, k, d = _stochastic_rsi(bars["close"])
    ema5, ema20 = bars["close"].ewm(span=5, adjust=False).mean(), bars["close"].ewm(span=20, adjust=False).mean()
    current = bars.iloc[-1]
    current_k, previous_k, current_d, previous_d = (
        _as_float(k.iloc[-1]),
        _as_float(k.iloc[-2]),
        _as_float(d.iloc[-1]),
        _as_float(d.iloc[-2]),
    )
    cross_up = (
        all(value is not None for value in (current_k, previous_k, current_d, previous_d))
        and current_k > current_d
        and previous_k <= previous_d
    )
    cross_down = (
        all(value is not None for value in (current_k, previous_k, current_d, previous_d))
        and current_k < current_d
        and previous_k >= previous_d
    )
    relation = np.where(bars["close"] > vwap, "ABOVE", np.where(bars["close"] < vwap, "BELOW", "AT"))
    last_relations = relation[-3:].tolist()
    ema_order = (
        "EMA5_ABOVE_EMA20"
        if ema5.iloc[-1] > ema20.iloc[-1]
        else "EMA5_BELOW_EMA20"
        if ema5.iloc[-1] < ema20.iloc[-1]
        else "EQUAL"
    )
    body = abs(float(current["close"] - current["open"]))
    candle_range = float(current["high"] - current["low"])
    return {
        "rsi14": _as_float(rsi.iloc[-1]),
        "stochastic_rsi": {
            "rsi_length": 14,
            "stochastic_length": 14,
            "k_sma_length": 3,
            "d_sma_length": 3,
            "oversold": 20.0,
            "overbought": 80.0,
            "current_k": current_k,
            "previous_k": previous_k,
            "current_d": current_d,
            "previous_d": previous_d,
            "cross_up": bool(cross_up),
            "cross_down": bool(cross_down),
            "cross_up_in_oversold": bool(cross_up and max(current_k, previous_k) <= 20.0),
            "cross_down_in_overbought": bool(cross_down and min(current_k, previous_k) >= 80.0),
        },
        "vwap": {
            "method": vwap_method,
            "value": _as_float(vwap.iloc[-1]),
            "distance_from_close": _as_float(float(vwap.iloc[-1] - current["close"])),
            "fraction_above": float((bars["close"] > vwap).mean()),
            "fraction_below": float((bars["close"] < vwap).mean()),
            "sequence_evidence": {
                "relations": last_relations,
                "all_recent_above": all(value == "ABOVE" for value in last_relations),
                "all_recent_below": all(value == "BELOW" for value in last_relations),
                "reclaimed": len(last_relations) >= 2 and last_relations[-2:] == ["BELOW", "ABOVE"],
                "lost": len(last_relations) >= 2 and last_relations[-2:] == ["ABOVE", "BELOW"],
            },
        },
        "ema": {
            "ema5": _as_float(ema5.iloc[-1]),
            "ema20": _as_float(ema20.iloc[-1]),
            "ema5_slope": _as_float(ema5.iloc[-1] - ema5.iloc[-2]) if len(ema5) > 1 else None,
            "ema20_slope": _as_float(ema20.iloc[-1] - ema20.iloc[-2]) if len(ema20) > 1 else None,
            "order": ema_order,
        },
        "candle": {
            "colour": "BULLISH"
            if current["close"] > current["open"]
            else "BEARISH"
            if current["close"] < current["open"]
            else "DOJI",
            "range": candle_range,
            "body": body,
        },
        "recent_candles": [
            {
                "timestamp": str(row.timestamp),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
            for row in bars.tail(5).itertuples(index=False)
        ],
    }


def _swing_points(bars: pd.DataFrame, field: str, *, window: int, higher_is_swing: bool) -> list[dict[str, Any]]:
    """Confirm fractal swing points only after ``window`` bars on both sides."""

    values = bars[field].to_numpy(dtype=float)
    points: list[dict[str, Any]] = []
    for index in range(window, len(values) - window):
        neighbours = np.concatenate((values[index - window : index], values[index + 1 : index + window + 1]))
        is_swing = values[index] > neighbours.max() if higher_is_swing else values[index] < neighbours.min()
        if is_swing:
            points.append({"timestamp": str(bars.iloc[index]["timestamp"]), "price": float(values[index])})
    return points[-5:]


def _market_structure(session_bars: pd.DataFrame, levels: Mapping[str, Any], *, swing_window: int) -> dict[str, Any]:
    """Return objective swing comparisons plus the one allowed long R1 pattern."""

    highs = _swing_points(session_bars, "high", window=swing_window, higher_is_swing=True)
    lows = _swing_points(session_bars, "low", window=swing_window, higher_is_swing=False)
    high_comparison = "INSUFFICIENT" if len(highs) < 2 else "HH" if highs[-1]["price"] > highs[-2]["price"] else "LH"
    low_comparison = "INSUFFICIENT" if len(lows) < 2 else "HL" if lows[-1]["price"] > lows[-2]["price"] else "LL"
    r1 = float(levels["levels"]["r1"])
    previous, current = session_bars.iloc[-2], session_bars.iloc[-1]
    bearish_touch = (
        previous["close"] < previous["open"]
        and previous["low"] <= r1 + _LEVEL_BUFFER_POINTS
        and previous["high"] >= r1 - _LEVEL_BUFFER_POINTS
    )
    bullish_reclaim = (
        current["close"] > current["open"] and current["close"] >= r1 and current["low"] >= r1 - _LEVEL_BUFFER_POINTS
    )
    return {
        "swing_window": swing_window,
        "swings": {"highs": highs, "lows": lows},
        "comparisons": {
            "highs": high_comparison,
            "lows": low_comparison,
            "higher_high": high_comparison == "HH",
            "lower_high": high_comparison == "LH",
            "higher_low": low_comparison == "HL",
            "lower_low": low_comparison == "LL",
        },
        "r1_scale_in_candidate": {
            "eligible": bool(bearish_touch and bullish_reclaim),
            "direction": "LONG",
            "r1": r1,
            "buffer_points": _LEVEL_BUFFER_POINTS,
            "bearish_touch": bool(bearish_touch),
            "bullish_reclaim": bool(bullish_reclaim),
        },
    }


def build_cpr_context(
    one_minute_candles: pd.DataFrame, *, position_state: Mapping[str, Any] | None = None, swing_window: int = 2
) -> dict[str, dict[str, Any]]:
    """Freeze-ready CPR context for the latest session's last complete five-minute bar.

    The returned mapping has exactly the four public MCP sections.  It contains
    no inferred regime and no execution object, credential, venue, or broker.
    """

    if swing_window < 1:
        raise ValueError("swing_window must be at least one bar on each side.")
    minutes = _prepared_minutes(one_minute_candles)
    bars = build_completed_five_minute_bars(minutes)
    if bars.empty:
        raise ValueError("CPR context needs at least one complete five-minute bar.")
    current_day = bars.iloc[-1]["timestamp"].date()
    latest_input_day = minutes.iloc[-1]["timestamp"].date()
    if current_day != latest_input_day:
        # A newly opened session can have only one to four observations.  Do
        # not quietly offer the prior session's frozen context for that bar.
        raise ValueError("CPR context needs a completed five-minute bar in the latest input session.")
    session_bars = bars.loc[bars["timestamp"].dt.date == current_day].reset_index(drop=True)
    if len(session_bars) < 2:
        raise ValueError("CPR context needs two complete current-session bars for pattern evidence.")
    levels = _session_levels(minutes, session_bars)
    return {
        "session_levels": levels,
        "momentum_vwap": _momentum_vwap(session_bars),
        "market_structure": _market_structure(session_bars, levels, swing_window=swing_window),
        "position_state": validate_position_state(position_state),
    }


__all__ = ["build_completed_five_minute_bars", "build_cpr_context"]
