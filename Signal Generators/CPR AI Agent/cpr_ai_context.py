"""Turn raw one-minute candles into the CPR agent's deterministic evidence.

This file owns the part of the strategy that must give the same answer every
time: completed five-minute bars, prior-session CPR levels, indicators,
confirmed swing points, and position facts.  It intentionally does not import
the older CPR Strategy package, so those strategies and this agent can evolve
and run independently.

The important boundary for a new maintainer is that this module describes the
market; it does not interpret it.  Regime classification and premise judgment
belong to Codex, while order permission, prices, and risk checks belong to the
host.  Keeping those responsibilities separate prevents model prose from
quietly becoming executable trading data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from cpr_ai_schema import validate_position_state

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
_LEVEL_BUFFER_POINTS = 2.0
_IST = ZoneInfo("Asia/Kolkata")


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


def _rsi_wilder(closes: pd.Series, length: int = 14) -> pd.Series:
    """Calculate TradingView-style Wilder RSI from a close series.

    Wilder's method is not pandas' default exponentially weighted mean.  It
    starts with a simple average of the first ``length`` gains/losses and then
    applies the recursive ``(old * (length - 1) + new) / length`` update.  The
    explicit implementation keeps our values aligned with charting evidence.
    """

    if length <= 0:
        raise ValueError("Wilder RSI length must be positive.")
    values = pd.to_numeric(closes, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= length:
        return result

    changes = values.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    # Seed both averages with the first complete window.  Starting the
    # recurrence at the first price change would produce a different RSI path.
    average_gain = float(gains.iloc[1 : length + 1].mean())
    average_loss = float(losses.iloc[1 : length + 1].mean())

    def rsi_value(gain: float, loss: float) -> float:
        """Convert averaged gains/losses into RSI, including flat-market edges."""

        if loss == 0.0:
            return 50.0 if gain == 0.0 else 100.0
        return 100.0 - (100.0 / (1.0 + gain / loss))

    result.iloc[length] = rsi_value(average_gain, average_loss)
    for position in range(length + 1, len(values)):
        average_gain = (
            average_gain * (length - 1) + float(gains.iloc[position])
        ) / length
        average_loss = (
            average_loss * (length - 1) + float(losses.iloc[position])
        ) / length
        result.iloc[position] = rsi_value(average_gain, average_loss)
    return result


def _stochastic_rsi(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return RSI14 and TradingView-style Stochastic RSI K/D (14, 3, 3).

    Stochastic RSI normalizes RSI within its own 14-value range, then smooths
    that raw oscillator with a three-period K average and a three-period D
    average.  The host later checks crosses and 20/80 zones from these facts.
    """

    rsi = _rsi_wilder(closes, 14)
    lowest = rsi.rolling(14, min_periods=14).min()
    highest = rsi.rolling(14, min_periods=14).max()
    denominator = highest - lowest
    raw = ((rsi - lowest) / denominator.replace(0.0, np.nan)) * 100.0
    k = raw.rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()
    return rsi, k, d


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


def _level_view(name: str, price: float, current_close: float) -> dict[str, Any]:
    """Expose a named level with signed and absolute distance from the close."""

    return {
        "name": name,
        "price": price,
        "distance_from_close": price - current_close,
        "abs_distance": abs(price - current_close),
    }


def _session_levels(
    minutes: pd.DataFrame,
    session_bars: pd.DataFrame,
    *,
    prior_accepted_regime: str | None,
) -> dict[str, Any]:
    """Calculate prior-day CPR, support/resistance, and opening-range facts.

    ``prior_accepted_regime`` is only continuity context for the next model
    turn.  It does not lock the new regime and is never used here to manufacture
    an entry.  The model may change its judgment when fresh evidence warrants.
    """

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
    # The next milestone must be beyond the two-point buffer.  A level already
    # being touched is not advertised as future reward for the 1R geometry gate.
    ordered = sorted(
        (_level_view(name, price, current_close) for name, price in levels.items()), key=lambda item: item["price"]
    )
    above = [item for item in ordered if item["price"] > current_close + _LEVEL_BUFFER_POINTS]
    below = [item for item in ordered if item["price"] < current_close - _LEVEL_BUFFER_POINTS]
    return {
        "session_date": str(current_session),
        "prior_accepted_regime": prior_accepted_regime,
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


def _momentum_vwap(
    session_bars: pd.DataFrame,
    indicator_bars: pd.DataFrame,
) -> dict[str, Any]:
    """Combine continuous momentum indicators with session-reset VWAP facts.

    ``indicator_bars`` contains every completed five-minute bar retained in the
    frozen snapshot. RSI, Stochastic RSI, and EMA therefore arrive at the new
    session already warmed, matching a continuous intraday chart. The overnight
    move from the prior close to the current open remains part of that history.

    ``session_bars`` contains only today's completed bars. VWAP, its recent
    relationships, the current candle, and recent-candle evidence must reset at
    the session boundary, so prior-day prices can never distort those facts.

    Volume is preferred when the feed supplies it.  This repository's index
    feed may carry no usable volume, so the deterministic fallback is the
    expanding mean of typical price.  Its method name is included in the
    snapshot so the agent can distinguish a proxy from true volume VWAP.
    """

    bars = session_bars.copy()
    indicators = indicator_bars.copy()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    volume = bars["volume"].fillna(0.0).clip(lower=0.0)
    if float(volume.sum()) > 0.0:
        vwap = (typical * volume).cumsum() / volume.cumsum()
        vwap_method = "volume_weighted"
    else:
        # Do not invent index volume.  Equal-weight typical price is explicit,
        # reproducible, and exposes its limitation through ``vwap_method``.
        vwap = typical.expanding().mean()
        vwap_method = "equal_weight_typical_price"
    rsi, k, d = _stochastic_rsi(indicators["close"])
    ema5 = indicators["close"].ewm(span=5, adjust=False).mean()
    ema20 = indicators["close"].ewm(span=20, adjust=False).mean()
    current = bars.iloc[-1]
    current_k, previous_k, current_d, previous_d = (
        _as_float(k.iloc[-1]),
        _as_float(k.iloc[-2]),
        _as_float(d.iloc[-1]),
        _as_float(d.iloc[-2]),
    )
    cross_up = (
        current_k is not None
        and previous_k is not None
        and current_d is not None
        and previous_d is not None
        and current_k > current_d
        and previous_k <= previous_d
    )
    cross_down = (
        current_k is not None
        and previous_k is not None
        and current_d is not None
        and previous_d is not None
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
    current_vwap = float(vwap.iloc[-1])

    def side_fraction(lower: float, upper: float, boundary: float, *, above: bool) -> float:
        """Return the fraction of a price segment strictly on one VWAP side.

        The host uses the completed entry candle rather than a session-wide
        close count.  A zero-length doji body deliberately yields zero on both
        sides, preventing it from satisfying a directional 40% body gate.
        """

        width = upper - lower
        if width <= 0.0:
            return 0.0
        portion = upper - max(lower, boundary) if above else min(upper, boundary) - lower
        return float(max(0.0, min(width, portion)) / width)

    body_lower, body_upper = sorted((float(current["open"]), float(current["close"])))
    range_lower, range_upper = float(current["low"]), float(current["high"])
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
            "cross_up_in_oversold": bool(
                cross_up
                and current_k is not None
                and previous_k is not None
                and max(current_k, previous_k) <= 20.0
            ),
            "cross_down_in_overbought": bool(
                cross_down
                and current_k is not None
                and previous_k is not None
                and min(current_k, previous_k) >= 80.0
            ),
        },
        "vwap": {
            "method": vwap_method,
            "value": _as_float(current_vwap),
            "distance_from_close": _as_float(float(current_vwap - current["close"])),
            "fraction_above": float((bars["close"] > vwap).mean()),
            "fraction_below": float((bars["close"] < vwap).mean()),
            "entry_candle": {
                "body_fraction_above": side_fraction(body_lower, body_upper, current_vwap, above=True),
                "body_fraction_below": side_fraction(body_lower, body_upper, current_vwap, above=False),
                "range_fraction_above": side_fraction(range_lower, range_upper, current_vwap, above=True),
                "range_fraction_below": side_fraction(range_lower, range_upper, current_vwap, above=False),
            },
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
            "open": _as_float(current["open"]),
            "high": _as_float(current["high"]),
            "low": _as_float(current["low"]),
            "close": _as_float(current["close"]),
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
            }
            for row in bars.tail(5).itertuples(index=False)
        ],
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


def _market_structure(session_bars: pd.DataFrame, levels: Mapping[str, Any], *, swing_window: int) -> dict[str, Any]:
    """Return objective swing comparisons plus the one allowed long R1 add.

    HH/HL/LH/LL are facts derived from confirmed points; this function does not
    convert them into a regime.  The scale-in candidate is deliberately only
    the documented red-then-green long pattern at R1.  No unapproved short/S1
    mirror is inferred.
    """

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
    levels = _session_levels(
        minutes,
        session_bars,
        prior_accepted_regime=prior_accepted_regime,
    )
    return {
        "session_levels": levels,
        "momentum_vwap": _momentum_vwap(session_bars, bars),
        "market_structure": _market_structure(session_bars, levels, swing_window=swing_window),
        "position_state": validate_position_state(position_state),
    }


__all__ = ["build_completed_five_minute_bars", "build_cpr_context"]
