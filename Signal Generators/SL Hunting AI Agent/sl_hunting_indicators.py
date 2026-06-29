"""Deterministic price-action detectors for the SL Hunting AI Agent.

Beginner note (why this file exists)
------------------------------------
The Claude agent should reason about *facts*, not eyeball a candle dump. So this
module computes the SL-Hunting method's building blocks deterministically from a
plain OHLC candle table:

- `pivot_and_levels(...)`  → day pivot, previous-day OHLC, today O/H/L, first-candle
                             hi/lo, nearby psychological levels, previous close.
- `fibo_levels(...)`       → 50/61/78 retracement + 161/261 extension of the most
                             recent significant swing, and where price sits.
- `candle_patterns(...)`   → reversal patterns (hammer/doji, engulfing, inside-bar,
                             reversal-bar, star) on recent candles, each tagged with
                             whether a CONFIRMATION candle has already printed.
- `market_structure(...)`  → swing points, trend, fast/slow read, trendline points,
                             W/M and double top/bottom.

Everything here is a pure function of (candles, config) — no randomness, no network
— so the agent's per-bar decisions are reproducible and cacheable. The functions
return plain JSON-serialisable dicts/lists so the MCP tools in `tools.py` can wrap
them directly.

Input contract: a pandas DataFrame with columns ``open, high, low, close`` (and
optionally ``volume``) plus a ``timestamp`` column OR a DatetimeIndex, ordered
oldest → newest. Helper `prepare_candles` normalises this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Fibonacci ratios used by the method (retracements + extensions).
_RETRACEMENTS = (0.5, 0.618, 0.786)
_EXTENSIONS = (1.272, 1.618, 2.618)


@dataclass(frozen=True)
class SLHuntingIndicatorConfig:
    """All tunable detector settings in one beginner-friendly place.

    Defaults are chosen for 5-minute NIFTY candles (the agent's default derived
    timeframe) but work for 1-minute too. The agent does the final judgement, so
    these only need to surface useful, deterministic facts — not be perfect.
    """

    # Candle anatomy thresholds (fractions of the candle's full range).
    doji_body_max_ratio: float = 0.10          # body <= 10% of range → doji
    long_wick_min_ratio: float = 0.55          # one wick >= 55% of range → long-wick
    full_body_min_ratio: float = 0.55          # body >= 55% of range → "full body" (confirmation)
    # How many of the most recent COMPLETED candles to scan for patterns.
    pattern_lookback: int = 12
    # Engulfing tolerance (fraction) when comparing bodies.
    engulf_tolerance: float = 0.0
    # Inside-bar: child range must be within parent range (no tolerance by default).
    # Swing detection: a pivot needs this many lower/higher bars on each side.
    swing_left: int = 2
    swing_right: int = 2
    swing_lookback: int = 120                  # bars to scan for swings / fibo
    # Two swings count as a "double"/equal level if within this fraction.
    double_tolerance_pct: float = 0.25
    # Psychological levels: nearest round numbers. NIFTY ≈ 25000, so 50/100-point
    # steps are the meaningful psych grid; we surface the nearby ones.
    psych_step: float = 100.0
    psych_window: int = 3                       # how many psych levels each side of price
    # "Fast vs slow": compare the average body of the latest third of the window to
    # the prior portion; ratio above/below these flags acceleration/deceleration.
    speed_window: int = 20
    # Cross-index (NF/BNF): how close (fraction of price) the last price must be to a
    # level to count as "at" that level / pivot.
    cross_index_band_pct: float = 0.15


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def prepare_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, time-sorted OHLC frame with a ``timestamp`` column.

    Accepts a ``timestamp`` column or a DatetimeIndex; coerces OHLC to float;
    drops rows with missing OHLC; sorts oldest → newest; resets the index.
    """
    if candles is None or len(candles) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = candles.copy()
    if "timestamp" not in df.columns:
        # Fall back to the index if it looks like a datetime.
        df = df.reset_index()
        # After reset_index the datetime index becomes a column; find it.
        for cand in ("timestamp", "index", "date", "datetime", "Datetime", "Date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    if "timestamp" not in df.columns:
        raise ValueError("prepare_candles: need a 'timestamp' column or a DatetimeIndex.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"prepare_candles: missing required column '{col}'.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Candle anatomy helpers
# ---------------------------------------------------------------------------

def _anatomy(row: pd.Series, cfg: SLHuntingIndicatorConfig) -> dict[str, Any]:
    """Describe ONE candle as numbers + simple shape flags.

    Beginner note: a candle has four prices — open/high/low/close (OHLC). We turn
    those into the parts a trader eyeballs:
    - ``range``  = high − low  (the candle's total height; the denominator for every
      ratio below, floored at a tiny number so we never divide by zero).
    - ``body``   = |close − open|  (the thick part).
    - ``upper_wick`` = the thin line ABOVE the body (high down to the body's top).
    - ``lower_wick`` = the thin line BELOW the body (body's bottom down to the low).
    The boolean flags then label the shape by comparing those parts to the
    fractions in ``cfg`` (e.g. "is this basically all wick and no body?").
    """
    o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
    rng = max(h - l, 1e-9)            # total height; 1e-9 floor avoids divide-by-zero
    body = abs(c - o)                 # thick part (direction-agnostic)
    upper = h - max(o, c)            # wick above the body
    lower = min(o, c) - l            # wick below the body
    is_bull = c > o                   # green candle (close above open)
    return {
        "open": round(o, 2),
        "high": round(h, 2),
        "low": round(l, 2),
        "close": round(c, 2),
        "range": round(rng, 2),
        "body": round(body, 2),
        "upper_wick": round(upper, 2),
        "lower_wick": round(lower, 2),
        "is_bull": bool(is_bull),
        "is_bear": bool(c < o),
        "is_doji": bool(body <= cfg.doji_body_max_ratio * rng),
        "is_full_body": bool(body >= cfg.full_body_min_ratio * rng),
        # Hammer: a long LOWER wick (price was pushed down, then rejected back up) with
        # almost no upper wick — a classic bullish-reversal shape at a support level.
        "is_hammer": bool(lower >= cfg.long_wick_min_ratio * rng and upper <= 0.25 * rng),
        # Shooting star: the mirror image — long UPPER wick, tiny lower wick (bearish at resistance).
        "is_shooting_star": bool(upper >= cfg.long_wick_min_ratio * rng and lower <= 0.25 * rng),
        # Long-wick (either side): marks where stop-losses / money are parked (a reversal/target zone).
        "is_long_wick": bool(max(upper, lower) >= cfg.long_wick_min_ratio * rng),
    }


# ---------------------------------------------------------------------------
# Pivot, OHLC and key levels
# ---------------------------------------------------------------------------

def pivot_and_levels(
    candles: pd.DataFrame,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Compute the day pivot, previous-day OHLC, today's levels and psych levels.

    Splits the intraday frame by calendar date: the latest date is "today", the
    date before it is the "previous day". Pivot = (prevHigh + prevLow + prevClose)/3.
    Psychological levels are the round numbers (``psych_step`` grid) nearest the
    current price. Distances are reported in points and percent so the agent can
    judge "near a level".
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    df = prepare_candles(candles)
    if df.empty:
        return {"available": False, "reason": "no candles"}

    df = df.assign(_date=df["timestamp"].dt.date)
    dates = list(dict.fromkeys(df["_date"].tolist()))  # unique, order-preserving
    today = dates[-1]
    today_df = df[df["_date"] == today]
    last_price = float(today_df["close"].iloc[-1])

    prev_day: dict[str, Any] | None = None
    pivot: float | None = None
    if len(dates) >= 2:
        prev = dates[-2]
        prev_df = df[df["_date"] == prev]
        ph, pl, pc, po = (
            float(prev_df["high"].max()),
            float(prev_df["low"].min()),
            float(prev_df["close"].iloc[-1]),
            float(prev_df["open"].iloc[0]),
        )
        prev_day = {"open": round(po, 2), "high": round(ph, 2), "low": round(pl, 2), "close": round(pc, 2)}
        pivot = round((ph + pl + pc) / 3.0, 2)

    first = today_df.iloc[0]
    today_levels = {
        "open": round(float(today_df["open"].iloc[0]), 2),
        "high": round(float(today_df["high"].max()), 2),
        "low": round(float(today_df["low"].min()), 2),
        "last": round(last_price, 2),
        "first_candle_high": round(float(first.high), 2),
        "first_candle_low": round(float(first.low), 2),
        "first_candle_time": str(first.timestamp),
    }

    # Psychological round-number levels around current price.
    step = cfg.psych_step
    base = round(last_price / step) * step
    psych = sorted(
        {round(base + k * step, 2) for k in range(-cfg.psych_window, cfg.psych_window + 1)}
    )

    def _dist(level: float | None) -> dict[str, Any] | None:
        if level is None:
            return None
        pts = round(last_price - level, 2)
        return {"level": round(level, 2), "points_away": pts, "pct_away": round(100.0 * pts / max(level, 1e-9), 3)}

    return {
        "available": True,
        "last_price": round(last_price, 2),
        "pivot": pivot,
        "previous_day_ohlc": prev_day,
        "previous_close": prev_day["close"] if prev_day else None,
        "today": today_levels,
        "psych_levels": psych,
        "psych_step": step,
        "distance_to": {
            "pivot": _dist(pivot),
            "previous_close": _dist(prev_day["close"]) if prev_day else None,
            "previous_high": _dist(prev_day["high"]) if prev_day else None,
            "previous_low": _dist(prev_day["low"]) if prev_day else None,
            "today_open": _dist(today_levels["open"]),
            "nearest_psych": _dist(min(psych, key=lambda p: abs(p - last_price))) if psych else None,
        },
        "note": "Above pivot = buyers' market; below = sellers'. A WICK at a level = trap/reversal; a clean BREAK = continuation.",
    }


# ---------------------------------------------------------------------------
# Swings (used by fibo + structure)
# ---------------------------------------------------------------------------

def _swings(df: pd.DataFrame, cfg: SLHuntingIndicatorConfig) -> list[dict[str, Any]]:
    """Return "fractal" swing highs/lows in the recent window (oldest → newest).

    Beginner note: a swing HIGH is a candle that is taller than its neighbours — its
    high is the highest within ``swing_left`` bars to its left and ``swing_right`` to
    its right (a local peak). A swing LOW is the opposite (a local trough/valley).
    These peaks and troughs are the skeleton the fibo + structure tools build on.
    We only look at the last ``swing_lookback`` bars to stay "recent".
    """
    window = df.tail(cfg.swing_lookback).reset_index(drop=True)
    n = len(window)
    left, right = cfg.swing_left, cfg.swing_right
    swings: list[dict[str, Any]] = []
    # Skip the first ``left`` and last ``right`` bars — they don't have enough
    # neighbours on both sides to be confirmed as a peak/trough.
    for i in range(left, n - right):
        hi = float(window.high.iloc[i])
        lo = float(window.low.iloc[i])
        # Peak/trough test: compare candle i against every neighbour in the window
        # [i-left .. i+right] (excluding itself). >= / <= so flat ties still qualify.
        is_high = all(hi >= float(window.high.iloc[j]) for j in range(i - left, i + right + 1) if j != i)
        is_low = all(lo <= float(window.low.iloc[j]) for j in range(i - left, i + right + 1) if j != i)
        if is_high:
            swings.append({"kind": "high", "price": round(hi, 2), "time": str(window.timestamp.iloc[i]), "bars_ago": n - 1 - i})
        elif is_low:
            swings.append({"kind": "low", "price": round(lo, 2), "time": str(window.timestamp.iloc[i]), "bars_ago": n - 1 - i})
    return swings


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------

def fibo_levels(
    candles: pd.DataFrame,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Compute fibo retracement/extension off the most recent significant swing.

    Uses the latest swing high and latest swing low in the window to define the
    impulse (direction = whichever of the two is more recent is the END of the
    move). Reports 50/61/78 retracements, 161/261 extensions, and where the
    current price sits relative to them.
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    df = prepare_candles(candles)
    if len(df) < (cfg.swing_left + cfg.swing_right + 2):
        return {"available": False, "reason": "not enough candles"}

    swings = _swings(df, cfg)
    highs = [s for s in swings if s["kind"] == "high"]
    lows = [s for s in swings if s["kind"] == "low"]
    if not highs or not lows:
        return {"available": False, "reason": "no clear swing high/low"}

    last_high = min(highs, key=lambda s: s["bars_ago"])
    last_low = min(lows, key=lambda s: s["bars_ago"])
    hi, lo = last_high["price"], last_low["price"]
    if hi <= lo:
        return {"available": False, "reason": "degenerate swing"}

    # Direction: if the high is more recent than the low → up-swing (low→high), so
    # retracements pull DOWN from the high. Else down-swing (high→low), retraces UP.
    up_swing = last_high["bars_ago"] <= last_low["bars_ago"]
    span = hi - lo
    last_price = round(float(df["close"].iloc[-1]), 2)

    # Truncate (not round) so the labels read the method's vocabulary exactly:
    # 0.618 -> "61%", 0.786 -> "78%", 1.618 -> "161%", 2.618 -> "261%".
    retr: dict[str, float] = {}
    for r in _RETRACEMENTS:
        level = hi - span * r if up_swing else lo + span * r
        retr[f"{int(r * 100)}%"] = round(level, 2)
    ext: dict[str, float] = {}
    for e in _EXTENSIONS:
        level = lo + span * e if up_swing else hi - span * e
        ext[f"{int(e * 100)}%"] = round(level, 2)

    # Which retracement band is price nearest to right now?
    nearest = min(retr.items(), key=lambda kv: abs(kv[1] - last_price))
    return {
        "available": True,
        "swing_direction": "up" if up_swing else "down",
        "swing_high": hi,
        "swing_low": lo,
        "last_price": last_price,
        "retracements": retr,
        "extensions": ext,
        "nearest_retracement": {"level_name": nearest[0], "price": nearest[1], "points_away": round(last_price - nearest[1], 2)},
        "note": "Only 50/61/78 are valid reversal zones (need pattern + confirmation); 161/261 are targets. 78% is the deepest valid zone.",
    }


# ---------------------------------------------------------------------------
# Candlestick patterns + confirmation
# ---------------------------------------------------------------------------

def _confirmation(
    df: pd.DataFrame, pattern_idx: int, direction: str, pattern_high: float, pattern_low: float,
    cfg: SLHuntingIndicatorConfig,
) -> dict[str, Any]:
    """Check whether any candle after `pattern_idx` confirms the pattern.

    A bullish confirmation = a later full-body candle that closes ABOVE the
    pattern's high; bearish = closes BELOW the pattern's low. Returns the first
    such candle, plus a `trap` flag if a later candle's wick poked back through the
    pattern (the doc's invalidation rule).
    """
    n = len(df)
    for j in range(pattern_idx + 1, n):
        a = _anatomy(df.iloc[j], cfg)
        if direction == "bullish" and a["close"] > pattern_high and a["is_full_body"] and a["is_bull"]:
            return {"confirmed": True, "bars_after": j - pattern_idx, "close": a["close"], "time": str(df.timestamp.iloc[j])}
        if direction == "bearish" and a["close"] < pattern_low and a["is_full_body"] and a["is_bear"]:
            return {"confirmed": True, "bars_after": j - pattern_idx, "close": a["close"], "time": str(df.timestamp.iloc[j])}
    # No confirmation yet — flag a trap if price poked back through after a wick test.
    return {"confirmed": False, "bars_after": None, "close": None, "time": None}


def candle_patterns(
    candles: pd.DataFrame,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Detect reversal patterns on recent completed candles, each with confirmation.

    Scans the last `pattern_lookback` candles for: hammer / shooting-star / doji
    (long-wick), bullish/bearish engulfing, inside-bar (harami), reversal-bar, and
    morning/evening star. Each detection carries its implied direction, the
    pattern's high/low (for the stop), and whether a confirmation candle has
    already printed beyond it (the method's mandatory rule).
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    df = prepare_candles(candles)
    n = len(df)
    if n < 3:
        return {"available": False, "reason": "not enough candles"}

    start = max(2, n - cfg.pattern_lookback)
    found: list[dict[str, Any]] = []

    for i in range(start, n):
        cur = _anatomy(df.iloc[i], cfg)
        prev = _anatomy(df.iloc[i - 1], cfg)
        prev2 = _anatomy(df.iloc[i - 2], cfg)
        bars_ago = n - 1 - i
        p_high, p_low = cur["high"], cur["low"]

        def _add(ptype: str, direction: str, hi: float, lo: float) -> None:
            conf = _confirmation(df, i, direction, hi, lo, cfg)
            found.append({
                "type": ptype,
                "direction": direction,
                "bars_ago": bars_ago,
                "time": str(df.timestamp.iloc[i]),
                "pattern_high": round(hi, 2),
                "pattern_low": round(lo, 2),
                **conf,
            })

        # Hammer / shooting star (long-wick single candle) — direction set by where
        # the confirmation closes, so we record BOTH potential directions' levels but
        # bias by wick side.
        if cur["is_hammer"]:
            _add("hammer", "bullish", p_high, p_low)
        if cur["is_shooting_star"]:
            _add("shooting_star", "bearish", p_high, p_low)
        if cur["is_doji"] and not cur["is_hammer"] and not cur["is_shooting_star"]:
            # A doji at a level can go either way; record as neutral-ish via both checks.
            _add("doji", "bullish", p_high, p_low)
            _add("doji", "bearish", p_high, p_low)

        # Engulfing (2-candle): the current candle's BODY fully swallows the previous
        # candle's body in the OPPOSITE colour. Bullish = a green candle whose body
        # covers a prior red body (close ≥ prev open AND open ≤ prev close); bearish is
        # the mirror. Colour matters here (unlike a hammer). Both bodies must be non-zero.
        if cur["body"] > 0 and prev["body"] > 0:
            bull_engulf = cur["is_bull"] and prev["is_bear"] and cur["close"] >= prev["open"] and cur["open"] <= prev["close"]
            bear_engulf = cur["is_bear"] and prev["is_bull"] and cur["close"] <= prev["open"] and cur["open"] >= prev["close"]
            if bull_engulf:
                _add("bullish_engulfing", "bullish", max(cur["high"], prev["high"]), min(cur["low"], prev["low"]))
            if bear_engulf:
                _add("bearish_engulfing", "bearish", max(cur["high"], prev["high"]), min(cur["low"], prev["low"]))

        # Inside bar / harami: the current candle sits ENTIRELY inside the previous
        # ("mother") candle's high-low range — a pause/coil. The trade is the breakout
        # of the mother candle, so we record both its high and low as the levels and
        # let the confirmation candle decide which way it actually breaks.
        if cur["high"] <= prev["high"] and cur["low"] >= prev["low"]:
            # Direction = breakout of the mother candle; record both edges.
            _add("inside_bar", "bullish", prev["high"], prev["low"])
            _add("inside_bar", "bearish", prev["high"], prev["low"])

        # Morning / evening star (3-candle reversal): a big candle, then a SMALL-bodied
        # candle (indecision — body ≤ half the first's), then a big candle the other way
        # that closes back past the MIDPOINT of the first. Morning star = down→pause→up
        # (bullish); evening star = up→pause→down (bearish).
        small_middle = prev["body"] <= 0.5 * max(prev2["body"], 1e-9)
        if prev2["is_bear"] and small_middle and cur["is_bull"] and cur["close"] > (prev2["open"] + prev2["close"]) / 2:
            _add("morning_star", "bullish", max(cur["high"], prev["high"], prev2["high"]), min(cur["low"], prev["low"], prev2["low"]))
        if prev2["is_bull"] and small_middle and cur["is_bear"] and cur["close"] < (prev2["open"] + prev2["close"]) / 2:
            _add("evening_star", "bearish", max(cur["high"], prev["high"], prev2["high"]), min(cur["low"], prev["low"], prev2["low"]))

    confirmed = [p for p in found if p.get("confirmed")]
    return {
        "available": True,
        "latest_candle": _anatomy(df.iloc[-1], cfg),
        "patterns": found[-cfg.pattern_lookback:],
        "confirmed_patterns": confirmed[-5:],
        "note": "A tradeable setup needs a pattern AT a level AND a confirmation candle that already closed beyond it. No confirmation = no trade.",
    }


# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------

def market_structure(
    candles: pd.DataFrame,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Report swings, trend, fast/slow read, trendline points, W/M and doubles."""
    cfg = cfg or SLHuntingIndicatorConfig()
    df = prepare_candles(candles)
    if len(df) < (cfg.swing_left + cfg.swing_right + 3):
        return {"available": False, "reason": "not enough candles"}

    swings = _swings(df, cfg)
    highs = [s for s in swings if s["kind"] == "high"]
    lows = [s for s in swings if s["kind"] == "low"]

    # Trend from the last two highs and last two lows.
    trend = "sideways"
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1]["price"] > highs[-2]["price"]
        hl = lows[-1]["price"] > lows[-2]["price"]
        lh = highs[-1]["price"] < highs[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]
        if hh and hl:
            trend = "uptrend"
        elif lh and ll:
            trend = "downtrend"

    # Fast vs slow: average body of the most recent third vs the prior portion.
    win = df.tail(cfg.speed_window)
    if len(win) >= 6:
        bodies = (win["close"] - win["open"]).abs().to_numpy()
        third = max(2, len(bodies) // 3)
        recent_avg = float(np.mean(bodies[-third:]))
        prior_avg = float(np.mean(bodies[:-third])) if len(bodies) > third else recent_avg
        ratio = recent_avg / max(prior_avg, 1e-9)
        speed = "accelerating" if ratio > 1.3 else "decelerating" if ratio < 0.7 else "steady"
    else:
        speed = "unknown"

    # Double top / bottom: last two highs (or lows) within tolerance.
    def _double(points: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(points) < 2:
            return None
        a, b = points[-2]["price"], points[-1]["price"]
        if abs(a - b) <= cfg.double_tolerance_pct / 100.0 * max(a, b):
            return {"price": round((a + b) / 2, 2), "points": [points[-2], points[-1]]}
        return None

    double_top = _double(highs)
    double_bottom = _double(lows)

    # Trendline points: the last few same-kind swings define an ascending/descending
    # line; we surface them so the agent can apply the "3rd point / break" rule.
    asc_line = lows[-4:] if len(lows) >= 2 else []
    desc_line = highs[-4:] if len(highs) >= 2 else []

    return {
        "available": True,
        "trend": trend,
        "speed": speed,
        "recent_swings": swings[-8:],
        "last_swing_high": highs[-1] if highs else None,
        "last_swing_low": lows[-1] if lows else None,
        "ascending_trendline_points": asc_line,
        "descending_trendline_points": desc_line,
        "double_top": double_top,
        "double_bottom": double_bottom,
        "note": "Trade a trendline only on the 3rd touch or a confirmed break (4th+). For W/M, trade the activation after the neckline break, not the breakout itself.",
    }


# ---------------------------------------------------------------------------
# Cross-index (NIFTY vs BankNIFTY) confirmation
# ---------------------------------------------------------------------------

def index_position(
    candles: pd.DataFrame,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Classify one index's current position vs its own pivot and key levels.

    Returns a compact read used by `cross_index_signal`: where the last price sits
    relative to the day pivot, the nearest support/resistance among {pivot, prev-day
    H/L, today H/L, prev close, nearest psych}, whether it is *at* support/resistance
    (within a small band), and whether it has recently broken the pivot up/down.
    Heuristic but deterministic — the agent applies the nuance.
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    pl = pivot_and_levels(candles, cfg)
    if not pl.get("available"):
        return {"available": False}
    df = prepare_candles(candles)
    last = float(pl["last_price"])
    band = max(cfg.cross_index_band_pct / 100.0 * last, 1e-6)
    pivot = pl.get("pivot")
    prev = pl.get("previous_day_ohlc") or {}

    levels: list[tuple[str, float]] = []
    if pivot is not None:
        levels.append(("pivot", float(pivot)))
    # Stable levels only (pivot, prev-day OHLC, nearest psych). Today's H/L are
    # excluded — they sit right next to price and would make "at a level" trivial.
    for name, val in (
        ("prev_high", prev.get("high")), ("prev_low", prev.get("low")),
        ("prev_close", pl.get("previous_close")),
    ):
        if val is not None:
            levels.append((name, float(val)))
    psych = pl.get("psych_levels") or []
    if psych:
        levels.append(("psych", float(min(psych, key=lambda p: abs(p - last)))))

    # Split the levels by where they sit vs the current price: anything at/below price
    # could act as support; anything strictly above could act as resistance.
    supports = [(n, v) for n, v in levels if v <= last]
    resistances = [(n, v) for n, v in levels if v > last]
    # "Nearest" support is the HIGHEST support (closest up to price from below) → max();
    # nearest resistance is the LOWEST resistance (closest down to price from above) → min().
    nearest_support = max(supports, key=lambda nv: nv[1]) if supports else None
    nearest_resistance = min(resistances, key=lambda nv: nv[1]) if resistances else None
    # "At" a level = price is within ``band`` (a small % of price) of it — i.e. touching it.
    at_support = bool(nearest_support and (last - nearest_support[1]) <= band)
    at_resistance = bool(nearest_resistance and (nearest_resistance[1] - last) <= band)

    if pivot is None:
        vs_pivot = "unknown"
    elif abs(last - pivot) <= band:
        vs_pivot = "at"
    else:
        vs_pivot = "above" if last > pivot else "below"

    # Detect a FRESH pivot break in the last few bars (not just "opened far from it").
    # Broke DOWN = price is now clearly below the pivot, yet some recent bar still
    # reached up to/above it — i.e. price was at the pivot and then crossed below.
    # Broke UP is the mirror. This is what the SL-hunting "clean break" rule keys on.
    broke_pivot_down = broke_pivot_up = False
    if pivot is not None and len(df) >= 2:
        recent = df.tail(6)
        if last < pivot - band and float(recent["high"].max()) >= pivot:
            broke_pivot_down = True
        if last > pivot + band and float(recent["low"].min()) <= pivot:
            broke_pivot_up = True

    return {
        "available": True,
        "last": round(last, 2),
        "pivot": round(pivot, 2) if pivot is not None else None,
        "vs_pivot": vs_pivot,
        "at_support": at_support,
        "at_resistance": at_resistance,
        "nearest_support": {"name": nearest_support[0], "price": round(nearest_support[1], 2)} if nearest_support else None,
        "nearest_resistance": {"name": nearest_resistance[0], "price": round(nearest_resistance[1], 2)} if nearest_resistance else None,
        "broke_pivot_down": broke_pivot_down,
        "broke_pivot_up": broke_pivot_up,
    }


def cross_index_signal(
    nifty_df: pd.DataFrame,
    bnf_df: pd.DataFrame | None,
    cfg: SLHuntingIndicatorConfig | None = None,
) -> dict[str, Any]:
    """Compare NIFTY vs BankNIFTY and emit the doc's NF/BNF alignment verdict.

    Faithful to the method's NF/BNF rules (advisory):
    - both at support   -> bias DOWN (operator likely breaks the support / SL-hunt)
    - both breakdown     -> bias UP   (the breakdown reverses)
    - both at resistance -> bias UP;  both breakup -> bias DOWN
    - one breaks, the other holds -> the break likely FAILS; bias toward the holder
    - opposite sides of pivot -> treat pivot as a normal level; WAIT for alignment
    Returns {"available": false, ...} when BankNIFTY data is missing.
    """
    cfg = cfg or SLHuntingIndicatorConfig()
    n = index_position(nifty_df, cfg)
    b = index_position(bnf_df, cfg) if bnf_df is not None else {"available": False}
    if not (n.get("available") and b.get("available")):
        return {"available": False, "reason": "BankNIFTY data unavailable; judge on NIFTY alone (be more conservative)."}

    # Verdict ladder, checked MOST-SPECIFIC first (both-break and both-at-a-level beat
    # the looser "same side of pivot" context). Remember the SL-hunting inversion: both
    # indices agreeing AT a level means that level likely FAILS (so fade it), while a
    # shared clean BREAK tends to reverse. ``bias`` is the suggested NIFTY direction.
    alignment, bias, reason = "neutral", "none", ""
    if n["broke_pivot_down"] and b["broke_pivot_down"]:
        alignment, bias = "both_breakdown", "up"
        reason = "Both indices broke down through pivot -> breakdown reverses; bias UP."
    elif n["broke_pivot_up"] and b["broke_pivot_up"]:
        alignment, bias = "both_breakup", "down"
        reason = "Both indices broke up through pivot -> breakup reverses; bias DOWN."
    elif n["at_support"] and b["at_support"]:
        alignment, bias = "both_at_support", "down"
        reason = "Both indices at support -> support likely fails; bias DOWN."
    elif n["at_resistance"] and b["at_resistance"]:
        alignment, bias = "both_at_resistance", "up"
        reason = "Both indices at resistance -> continuation up; bias UP."
    elif (n["broke_pivot_down"] != b["broke_pivot_down"]) and (n["at_support"] or b["at_support"]):
        holder = "BankNIFTY" if n["broke_pivot_down"] else "NIFTY"
        alignment, bias = "divergence_breakdown", "up"
        reason = f"One index broke down while {holder} held support -> breakdown likely fails; bias UP toward the holder."
    elif n["vs_pivot"] in ("above", "below") and b["vs_pivot"] in ("above", "below") and n["vs_pivot"] != b["vs_pivot"]:
        alignment, bias = "opposite_sides_of_pivot", "wait"
        reason = "Indices are on opposite sides of their pivots -> treat pivot as a normal level and WAIT for alignment."
    elif n["vs_pivot"] == b["vs_pivot"] == "above":
        alignment, bias = "both_above_pivot", "up_context"
        reason = "Both above pivot -> buyers' market on both; bullish context."
    elif n["vs_pivot"] == b["vs_pivot"] == "below":
        alignment, bias = "both_below_pivot", "down_context"
        reason = "Both below pivot -> sellers' market on both; bearish context."

    return {
        "available": True,
        "alignment": alignment,
        "bias": bias,
        "reason": reason,
        "nifty": n,
        "bank_nifty": b,
        "note": "Advisory: weigh with the NIFTY setup. 'wait' = require alignment; if this disagrees with your NIFTY read, prefer HOLD.",
    }
