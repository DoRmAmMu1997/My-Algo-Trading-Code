"""Turn raw one-minute candles into the CPR agent's deterministic evidence.

This file owns the part of the strategy that must give the same answer every
time: completed five-minute bars, prior-session CPR levels, the recent-range
ATR, session VWAP, confirmed swing points, the host's Trend-Day Rider
candidate, and position facts. It intentionally does not import the older CPR
Strategy package, so those strategies and this agent evolve independently.

The important boundary for a new maintainer is that this module describes the
market; it does not interpret it. The candidate itself comes from
``cpr_ai_trend_day`` -- the same code the backtest replays -- and the host
policy trusts only that frozen verdict. Codex may accept or veto it and judge
premise exits; prices and risk belong to the host. Keeping those
responsibilities separate prevents model prose from quietly becoming
executable trading data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from cpr_ai_schema import validate_position_state
from cpr_ai_trend_day import evaluate_trend_day_candidate, session_atr, session_vwap

if TYPE_CHECKING:
    # mypy_path exposes Dependencies by its bare module name. The importlib
    # production loader starts from the repository root instead.
    from market_data_health import (
        complete_minute_bucket_mask,
        newest_completed_minute_timestamp,
    )
else:
    from Dependencies.market_data_health import (
        complete_minute_bucket_mask,
        newest_completed_minute_timestamp,
    )


_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
_IST = ZoneInfo("Asia/Kolkata")
_RECENT_CANDLES = 6


def _as_float(value: Any) -> float | None:
    """Return a JSON-safe finite float, or ``None`` when data is unavailable.

    JSON does not have portable representations for pandas ``NA``, ``NaN``,
    or infinity.  Converting them to ``None`` keeps the frozen MCP snapshot
    valid and makes missing evidence explicit to the host validator.
    """

    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _prepared_minutes(one_minute_candles: pd.DataFrame) -> pd.DataFrame:
    """Copy, type-check, and chronologically sort one-minute OHLC observations.

    Duplicate timestamps deliberately remain present here.  The exact-minute
    completeness check below must see them; silently keeping one revision could
    let a duplicated/missing websocket sequence masquerade as a complete bar.
    """

    missing = [column for column in _REQUIRED_COLUMNS if column not in one_minute_candles.columns]
    if missing:
        raise ValueError(f"One-minute CPR context is missing columns: {missing}")
    frame = one_minute_candles.loc[:, [*one_minute_candles.columns]].copy()
    parsed_timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["timestamp"] = [
        (
            pd.Timestamp(value).tz_localize(_IST)
            if pd.Timestamp(value).tzinfo is None
            else pd.Timestamp(value).tz_convert(_IST)
        ).tz_localize(None)
        for value in parsed_timestamps
    ]
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("timestamp", kind="stable")
    return frame.reset_index(drop=True)


def _completed_minutes(
    one_minute_candles: pd.DataFrame,
    *,
    as_of: datetime | None,
) -> pd.DataFrame:
    """Return only start-stamped one-minute candles whose interval has closed.

    The shared feed may already contain the 09:19 row while the 09:19-09:20
    interval is still changing.  The health helper applies that repository-wide
    start-stamp convention, so an agent never treats that forming minute as
    completed evidence.
    """

    minutes = _prepared_minutes(one_minute_candles)
    newest = newest_completed_minute_timestamp(minutes, now=as_of)
    if newest is None:
        return minutes.iloc[0:0].copy()
    cutoff = pd.Timestamp(newest).tz_convert(_IST).tz_localize(None)
    return minutes.loc[minutes["timestamp"] <= cutoff].reset_index(drop=True)


def build_completed_five_minute_bars(
    one_minute_candles: pd.DataFrame,
    *,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Build five-minute bars only from the five exact expected minute slots.

    Merely counting five rows is unsafe: a duplicated 09:17 row plus a missing
    09:18 row would still total five observations.  The completeness mask
    verifies the actual minute identities before resampling.  Partial/forming
    buckets are discarded so completed-bar actions cannot fire intrabar.
    """

    minutes = _completed_minutes(one_minute_candles, as_of=as_of)
    completed: list[pd.DataFrame] = []
    # Resample per calendar session so a 15:29 observation cannot be paired
    # with a new day's 09:15 observation in an accidental cross-day bucket.
    for _, session in minutes.groupby(minutes["timestamp"].dt.date, sort=True):
        indexed = session.set_index("timestamp")
        exact_minute_mask = complete_minute_bucket_mask(
            pd.DatetimeIndex(indexed.index),
            5,
        )
        indexed = indexed.loc[exact_minute_mask.to_numpy()]
        if indexed.empty:
            continue
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



def _opening_facts(session_bars: pd.DataFrame, count: int) -> dict[str, Any]:
    """Return the first-N-minute corridor, or explicitly mark it incomplete.

    Returning ``complete=False`` is safer than constructing a shorter opening
    range: the model can see that the fact is unavailable, and the host cannot
    accidentally validate a setup against a half-built window.
    """

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


def _swing_points(bars: pd.DataFrame, field: str, *, window: int, higher_is_swing: bool) -> list[dict[str, Any]]:
    """Confirm fractal swings only after ``window`` bars exist on both sides.

    Requiring later bars prevents a current extreme from being labeled a swing
    before it is confirmed.  That delay is intentional because sideways stops
    use the latest returned swing as authoritative geometry.
    """

    values = bars[field].to_numpy(dtype=float)
    points: list[dict[str, Any]] = []
    for index in range(window, len(values) - window):
        neighbours = np.concatenate((values[index - window : index], values[index + 1 : index + window + 1]))
        is_swing = values[index] > neighbours.max() if higher_is_swing else values[index] < neighbours.min()
        if is_swing:
            points.append({"timestamp": str(bars.iloc[index]["timestamp"]), "price": float(values[index])})
    return points[-5:]


def _vwap_method(session_bars: pd.DataFrame) -> str:
    """Name the VWAP flavour so the model can tell a proxy from true VWAP."""

    volume = pd.to_numeric(session_bars["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return "volume_weighted" if float(volume.sum()) > 0.0 else "equal_weight_typical_price"


def _session_levels(
    previous: pd.DataFrame,
    session_bars: pd.DataFrame,
    *,
    session_date: str,
    prior_accepted_regime: str | None,
    atr: float | None,
    atr_sessions_used: int,
    prior_ranges: list[float],
) -> dict[str, Any]:
    """Calculate prior-day CPR levels, the opening gap, and the recent ATR.

    ``prior_accepted_regime`` is only continuity context for the next model
    turn. It never locks a regime or manufactures an entry. The ATR is the
    same mean of recent prior-session ranges the candidate gate uses, so the
    model and the host describe expansion with one number.
    """

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
    session_open = float(session_bars.iloc[0]["open"])
    gap_points = session_open - prior_close
    return {
        "session_date": session_date,
        "prior_accepted_regime": prior_accepted_regime,
        "current_close": current_close,
        "previous_day": {"high": prior_high, "low": prior_low, "close": prior_close},
        "levels": levels,
        "distances_from_current_close": {name: price - current_close for name, price in levels.items()},
        "gap": {
            "session_open": session_open,
            "points": gap_points,
            "direction": "UP" if gap_points > 0 else "DOWN" if gap_points < 0 else "FLAT",
        },
        "atr": {
            "value": _as_float(atr),
            "sessions_used": atr_sessions_used,
            "prior_session_ranges": [float(value) for value in prior_ranges[-5:]],
        },
        "opening": {
            "opening_corridor": _opening_facts(session_bars, 5),
            "first_15_minutes": _opening_facts(session_bars, 15),
            "first_30_minutes": _opening_facts(session_bars, 30),
        },
    }


def _momentum_vwap(session_bars: pd.DataFrame, *, atr: float | None) -> dict[str, Any]:
    """Describe session VWAP, the newest candle, and the recent candles.

    Everything here resets at the session boundary, so prior-day prices can
    never distort VWAP or candle facts. NIFTY index candles carry no volume,
    so VWAP is normally the equal-weight typical-price proxy; ``method`` says
    which one was used.
    """

    bars = session_bars.reset_index(drop=True)
    vwap = session_vwap(bars)
    current = bars.iloc[-1]
    close = float(current["close"])
    current_vwap = float(vwap.iloc[-1])
    relation = np.where(bars["close"] > vwap, "ABOVE", np.where(bars["close"] < vwap, "BELOW", "AT"))
    candle_range = float(current["high"] - current["low"])
    return {
        "vwap": {
            "method": _vwap_method(bars),
            "value": _as_float(current_vwap),
            "close_minus_vwap": _as_float(close - current_vwap),
            "distance_atr": _as_float(abs(close - current_vwap) / atr) if atr else None,
            "fraction_of_bars_above": float((bars["close"] > vwap).mean()),
            "fraction_of_bars_below": float((bars["close"] < vwap).mean()),
            "recent_relations": relation[-_RECENT_CANDLES:].tolist(),
        },
        "candle": {
            "timestamp": str(current["timestamp"]),
            "colour": "BULLISH"
            if current["close"] > current["open"]
            else "BEARISH"
            if current["close"] < current["open"]
            else "DOJI",
            "open": _as_float(current["open"]),
            "high": _as_float(current["high"]),
            "low": _as_float(current["low"]),
            "close": _as_float(close),
            "range": candle_range,
            "body": abs(float(current["close"] - current["open"])),
            "range_atr": _as_float(candle_range / atr) if atr else None,
        },
        "recent_candles": [
            {
                "timestamp": str(row.timestamp),
                # pandas-stubs gives named-tuple cells a deliberately broad
                # scalar union; preparation above has already made these four
                # columns numeric, so the casts document that proven boundary.
                "open": float(cast(Any, row.open)),
                "high": float(cast(Any, row.high)),
                "low": float(cast(Any, row.low)),
                "close": float(cast(Any, row.close)),
                "vwap": float(cast(Any, vwap_value)),
            }
            for row, vwap_value in zip(
                bars.tail(_RECENT_CANDLES).itertuples(index=False),
                vwap.tail(_RECENT_CANDLES),
                strict=True,
            )
        ],
    }


def _market_structure(
    session_bars: pd.DataFrame,
    *,
    swing_window: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Return confirmed swings, extreme recency, and the host's candidate.

    HH/HL/LH/LL are facts derived from confirmed points; this function does
    not convert them into a regime. ``trend_day_candidate`` is the frozen
    verdict of the deterministic gate -- the only evidence the host policy
    accepts for an entry.
    """

    highs = _swing_points(session_bars, "high", window=swing_window, higher_is_swing=True)
    lows = _swing_points(session_bars, "low", window=swing_window, higher_is_swing=False)
    high_comparison = "INSUFFICIENT" if len(highs) < 2 else "HH" if highs[-1]["price"] > highs[-2]["price"] else "LH"
    low_comparison = "INSUFFICIENT" if len(lows) < 2 else "HL" if lows[-1]["price"] > lows[-2]["price"] else "LL"
    last_index = len(session_bars) - 1
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
        "extremes": {
            "bars_since_session_high": int(last_index - int(np.argmax(session_bars["high"].to_numpy()))),
            "bars_since_session_low": int(last_index - int(np.argmin(session_bars["low"].to_numpy()))),
        },
        "trend_day_candidate": candidate,
    }


def build_cpr_context(
    one_minute_candles: pd.DataFrame,
    *,
    position_state: Mapping[str, Any] | None = None,
    prior_accepted_regime: str | None = None,
    swing_window: int = 2,
    as_of: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the four-section snapshot for the latest completed five-minute bar.

    The result is ready to freeze and expose through the read-only MCP tools.
    It contains no inferred regime (apart from the explicitly labelled prior
    judgment), execution object, credential, venue, broker, or order method.
    Position input is normalized before inclusion so both Codex and the host
    validate the same facts.
    """

    if swing_window < 1:
        raise ValueError("swing_window must be at least one bar on each side.")
    if prior_accepted_regime not in {None, "SIDEWAYS", "TRENDING", "UNDECIDED"}:
        raise ValueError(
            "prior_accepted_regime must be SIDEWAYS, TRENDING, UNDECIDED, or None."
        )
    # Apply the completion cutoff before resampling and again inside the public
    # bar builder.  The second application is harmless and keeps that public
    # helper safe when it is called independently.
    minutes = _completed_minutes(one_minute_candles, as_of=as_of)
    bars = build_completed_five_minute_bars(minutes, as_of=as_of)
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
    prior_sessions = [
        group for day, group in minutes.groupby(minutes["timestamp"].dt.date, sort=True) if day < current_day
    ]
    if not prior_sessions:
        raise ValueError("CPR context needs one complete prior session and one current session.")
    # Oldest first, as the candidate gate expects. The live store keeps about
    # five sessions, which is exactly what the ATR5 definition needs.
    prior_ranges = [float(group["high"].max() - group["low"].min()) for group in prior_sessions]
    atr, atr_sessions_used = session_atr(prior_ranges)
    previous = prior_sessions[-1]
    candidate = evaluate_trend_day_candidate(
        session_bars,
        prior_high=float(previous["high"].max()),
        prior_low=float(previous["low"].min()),
        prior_close=float(previous.iloc[-1]["close"]),
        atr=atr,
        atr_sessions_used=atr_sessions_used,
    ).to_dict()
    return {
        "session_levels": _session_levels(
            previous,
            session_bars,
            session_date=str(current_day),
            prior_accepted_regime=prior_accepted_regime,
            atr=atr,
            atr_sessions_used=atr_sessions_used,
            prior_ranges=prior_ranges,
        ),
        "momentum_vwap": _momentum_vwap(session_bars, atr=atr),
        "market_structure": _market_structure(session_bars, swing_window=swing_window, candidate=candidate),
        "position_state": validate_position_state(position_state),
    }


__all__ = ["build_completed_five_minute_bars", "build_cpr_context"]
