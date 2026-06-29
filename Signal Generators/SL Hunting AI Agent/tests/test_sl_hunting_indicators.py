"""Tests for the deterministic SL-Hunting detectors."""

from __future__ import annotations

import pandas as pd

from sl_hunting_indicators import (
    SLHuntingIndicatorConfig,
    candle_patterns,
    fibo_levels,
    market_structure,
    pivot_and_levels,
    prepare_candles,
)


def _ohlc(rows, start="2026-06-26 09:15"):
    """Build an OHLC frame from (o,h,l,c) tuples at 5-min steps."""
    ts = pd.date_range(start=start, periods=len(rows), freq="5min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100] * len(rows),
        }
    )


def _closes_to_ohlc(closes, start="2026-06-26 09:15"):
    rows = []
    prev = closes[0]
    for c in closes:
        o = prev
        hi = max(o, c) + 1.0
        lo = min(o, c) - 1.0
        rows.append((o, hi, lo, c))
        prev = c
    return _ohlc(rows, start=start)


def test_prepare_candles_accepts_datetime_index():
    df = _ohlc([(100, 101, 99, 100)]).set_index("timestamp")
    prepared = prepare_candles(df)
    assert "timestamp" in prepared.columns
    assert len(prepared) == 1


def test_pivot_uses_previous_day_ohlc():
    # Previous day (2026-06-25): high 110, low 90, close 100 -> pivot = 100.0
    prev = _ohlc(
        [(95, 110, 90, 100), (100, 105, 98, 100)],
        start="2026-06-25 09:15",
    )
    today = _ohlc([(101, 103, 100, 102)], start="2026-06-26 09:15")
    df = pd.concat([prev, today], ignore_index=True)

    result = pivot_and_levels(df)
    assert result["available"] is True
    assert result["pivot"] == round((110 + 90 + 100) / 3.0, 2)
    assert result["previous_day_ohlc"] == {"open": 95.0, "high": 110.0, "low": 90.0, "close": 100.0}
    assert result["today"]["first_candle_high"] == 103.0
    assert result["previous_close"] == 100.0


def test_fibo_levels_available_with_swings():
    closes = (
        list(range(100, 116))      # up to ~115
        + list(range(115, 105, -1))  # pull back to ~106
        + list(range(106, 122))    # up to ~121
        + list(range(121, 112, -1))  # pull back
    )
    df = _closes_to_ohlc([float(c) for c in closes])
    result = fibo_levels(df)
    assert result["available"] is True
    assert set(result["retracements"]) == {"50%", "61%", "78%"}
    assert result["swing_direction"] in ("up", "down")


def test_candle_patterns_detects_confirmed_bullish_engulfing():
    rows = [
        (100, 101, 99, 100.0),   # filler
        (100, 101, 99, 100.0),   # filler
        (100, 101, 95, 96.0),    # bearish (engulf target)
        (95, 103, 94, 102.0),    # bullish engulfing of the prior body
        (103, 109, 102.5, 108.0),  # full-body bullish confirmation closing above pattern high
    ]
    df = _ohlc(rows)
    result = candle_patterns(df)
    assert result["available"] is True
    kinds = {p["type"] for p in result["patterns"]}
    assert "bullish_engulfing" in kinds
    confirmed = [p for p in result["confirmed_patterns"] if p["type"] == "bullish_engulfing"]
    assert confirmed and confirmed[0]["confirmed"] is True


def test_market_structure_reports_trend_and_speed():
    closes = [float(c) for c in (list(range(100, 130)) )]
    df = _closes_to_ohlc(closes)
    result = market_structure(df)
    assert result["available"] is True
    assert result["trend"] in ("uptrend", "downtrend", "sideways")
    assert result["speed"] in ("accelerating", "decelerating", "steady", "unknown")


def test_empty_frame_is_handled():
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    assert pivot_and_levels(empty)["available"] is False
    assert fibo_levels(empty)["available"] is False
