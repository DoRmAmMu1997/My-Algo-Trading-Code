"""Tests for the deterministic SL-Hunting detectors."""

from __future__ import annotations

import pandas as pd
from sl_hunting_indicators import (
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


# ---------------------------------------------------------------------------
# SLH-017: the open is CLASSIFIED in code, not derived by the model
# ---------------------------------------------------------------------------

def _two_sessions(prev_close, today_open):
    """Prior session ending at `prev_close`, today opening at `today_open`."""
    prev = _ohlc(
        [(prev_close, prev_close + 5, prev_close - 5, prev_close)],
        start="2026-09-15 09:15",
    )
    today = _ohlc(
        [(today_open, today_open + 2, today_open - 2, today_open)],
        start="2026-09-16 09:15",
    )
    return pd.concat([prev, today], ignore_index=True)


def test_the_open_classification_measures_the_gap_against_the_prior_session_close():
    """SLH-017. The arithmetic the model kept doing by hand, done once in code.

    On 2026-09-16 the model computed "+82pts (0.36%)" correctly and then called
    it a gap-up, which flipped the pre-open note's branch and produced four
    losing longs. v4t already calibrated the threshold; nothing applied it.
    """
    result = pivot_and_levels(_two_sessions(23118.60, 23200.60))

    oc = result["open_classification"]
    assert oc["available"] is True
    assert oc["reference_level"] == 23118.60
    assert oc["today_open"] == 23200.60
    assert oc["gap_points"] == 82.0
    assert oc["gap_pct"] == round(100.0 * 82.0 / 23118.60, 3)


def test_a_third_of_a_percent_is_classified_FLAT_not_a_gap():
    """The 16 Sep open itself: +0.355%, below the half-percent line -> FLAT."""
    oc = pivot_and_levels(_two_sessions(23118.60, 23200.60))["open_classification"]

    assert oc["gap_pct"] < 0.5
    assert oc["verdict"] == "FLAT"


def test_half_a_percent_or_more_earns_the_gap_reading():
    """v4t: "something in the region of half a percent" is where a gap starts."""
    up = pivot_and_levels(_two_sessions(20000.0, 20100.0))["open_classification"]
    assert up["gap_pct"] == 0.5
    assert up["verdict"] == "GAP_UP"

    down = pivot_and_levels(_two_sessions(20000.0, 19900.0))["open_classification"]
    assert down["gap_pct"] == -0.5
    assert down["verdict"] == "GAP_DOWN"


def test_just_under_the_threshold_still_reads_FLAT_on_both_sides():
    """Hesitation resolves to flat, so the boundary is inclusive of the gap side
    and everything below it is flat -- in BOTH directions."""
    up = pivot_and_levels(_two_sessions(20000.0, 20099.0))["open_classification"]
    assert up["verdict"] == "FLAT"

    down = pivot_and_levels(_two_sessions(20000.0, 19901.0))["open_classification"]
    assert down["verdict"] == "FLAT"


def test_the_classification_names_its_reference_and_its_threshold():
    """A verdict the model cannot audit is one it will re-derive its own way."""
    oc = pivot_and_levels(_two_sessions(23118.60, 23200.60))["open_classification"]

    assert oc["threshold_pct"] == 0.5
    # v4u / SLH-023: the reference is the prior session's 3:15 level, not the
    # 15:29 auction print or the official 15:30 close, because settlement
    # prints are not positioning.
    assert "3:15" in oc["reference"]
    assert "15:30" in oc["reference"]


def test_the_gap_is_measured_from_the_OPEN_and_does_not_drift_intraday():
    """Caught by mutation testing: every other fixture here opens and closes at
    the same price, so swapping the OPEN for the last price passed them all.

    It matters because the classification is a statement about the OPEN. If it
    read the live price instead, a flat open would quietly become a "gap up" by
    mid-morning and re-select the branch of the pre-open plan hours after the
    fact -- the exact failure this block exists to prevent, arriving by a
    different door.
    """
    prev = _ohlc([(20000.0, 20005.0, 19995.0, 20000.0)], start="2026-09-15 09:15")
    # Opens +0.05% (flat), then rallies 1.5% during the session.
    today = _ohlc(
        [
            (20010.0, 20015.0, 20005.0, 20012.0),
            (20012.0, 20310.0, 20010.0, 20300.0),
        ],
        start="2026-09-16 09:15",
    )

    oc = pivot_and_levels(pd.concat([prev, today], ignore_index=True))["open_classification"]

    assert oc["today_open"] == 20010.0
    assert oc["gap_points"] == 10.0
    assert oc["verdict"] == "FLAT"


def test_a_single_session_cannot_classify_its_own_open():
    """With no prior day there is no reference, and the block says so rather
    than guessing from today's first candle."""
    today = _ohlc([(100, 103, 99, 102)], start="2026-09-16 09:15")

    oc = pivot_and_levels(today)["open_classification"]

    assert oc["available"] is False
    assert oc["verdict"] is None


# ---------------------------------------------------------------------------
# SLH-023: the 15:15-onward settlement prints are not trading
# ---------------------------------------------------------------------------

def _bars(day, rows):
    """1-min bars on `day` from (HH:MM, o, h, l, c) tuples."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"{day} {r[0]}" for r in rows]),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [100] * len(rows),
        }
    )


def _settled_session(day, close_315, auction_close):
    """A session that trades into 15:14, prints flat to 15:28 and jumps at 15:29,
    as the real NIFTY feed does (12 Oct 2026: flat at 22,316.85, then 22,363.35)."""
    rows = [
        ("15:13", close_315 - 4, close_315 + 1, close_315 - 6, close_315 - 2),
        ("15:14", close_315 - 2, close_315 + 3, close_315 - 3, close_315),
    ]
    rows += [(f"15:{m:02d}", close_315, close_315, close_315, close_315) for m in range(15, 29)]
    rows.append(("15:29", close_315, auction_close, close_315, auction_close))
    return _bars(day, rows)


def test_the_gap_is_measured_from_the_3_15_level_not_the_closing_auction():
    """SLH-023, the 13 Oct case. The reference used to be the prior session's
    LAST bar -- the 15:29 auction print -- which turned a +0.21% open into
    "+0.8 points". The 3:15 level is what v4u and the code's own comment meant."""
    prev = _settled_session("2026-10-12", 22316.90, 22363.35)
    today = _bars("2026-10-13", [("09:15", 22364.15, 22370.0, 22288.0, 22355.45)])

    oc = pivot_and_levels(pd.concat([prev, today], ignore_index=True))["open_classification"]

    assert oc["reference_level"] == 22316.90
    assert oc["gap_points"] == 47.25
    assert "3:15" in oc["reference"] and "auction" in oc["reference"]


def test_an_auction_jump_can_no_longer_hide_a_gap():
    """The worst five-year case, 5 Aug 2026: +0.84% on the 3:15 level read FLAT
    because the prior auction print had jumped 0.62%. Measured over five years,
    the two references disagreed on the verdict in 69 of 1,227 sessions."""
    prev = _settled_session("2026-08-04", 20000.0, 20124.0)
    today = _bars("2026-08-05", [("09:15", 20168.0, 20180.0, 20150.0, 20170.0)])

    oc = pivot_and_levels(pd.concat([prev, today], ignore_index=True))["open_classification"]

    assert oc["gap_pct"] == 0.84
    assert oc["verdict"] == "GAP_UP"


def test_a_session_with_no_bar_before_15_15_falls_back_to_its_last_bar():
    prev = _bars(
        "2026-10-12",
        [("15:20", 100.0, 100.0, 100.0, 100.0), ("15:29", 100.0, 101.0, 100.0, 101.0)],
    )
    today = _bars("2026-10-13", [("09:15", 101.2, 102.0, 101.0, 101.5)])

    oc = pivot_and_levels(pd.concat([prev, today], ignore_index=True))["open_classification"]

    assert oc["available"] is True
    assert oc["reference_level"] == 101.0


def test_candle_patterns_ignore_an_earlier_sessions_settlement_prints():
    """SLH-023, the other half. Seven decisions on 13 Oct had to discard
    'confirmed' patterns that were the prior session's flat settlement prints."""
    prev = _settled_session("2026-10-12", 22316.90, 22363.35)
    today = _bars(
        "2026-10-13",
        [
            ("09:15", 22364.15, 22370.0, 22288.0, 22355.45),
            ("09:16", 22355.0, 22390.0, 22350.0, 22385.0),
        ],
    )

    result = candle_patterns(pd.concat([prev, today], ignore_index=True))

    assert result["available"] is True
    assert result["patterns"]
    # Today's two bars are bars_ago 1 and 0. Every pattern must sit on one of
    # them: before SLH-023 the lookback reached back into the flat 15:15-15:29
    # prints and reported inside bars and dojis there.
    assert all(p["bars_ago"] <= 1 for p in result["patterns"]), result["patterns"]


def test_the_latest_session_keeps_its_own_late_bars():
    """Only EARLIER sessions lose their settlement prints; today's bars are what
    the agent trades, so a 15:15 bar of the latest session is still scanned."""
    today = _bars(
        "2026-10-13",
        [
            ("15:12", 100.0, 103.0, 99.0, 102.0),
            ("15:13", 102.0, 104.0, 98.0, 99.0),
            ("15:14", 99.0, 102.0, 98.5, 101.0),
            ("15:15", 101.0, 101.0, 101.0, 101.0),
        ],
    )

    result = candle_patterns(today)

    # The flat 15:15 bar is still the latest candle, and the scan read it.
    assert result["latest_candle"]["high"] == result["latest_candle"]["low"] == 101.0
    assert any(p["type"] == "inside_bar" and p["bars_ago"] == 0 for p in result["patterns"])
