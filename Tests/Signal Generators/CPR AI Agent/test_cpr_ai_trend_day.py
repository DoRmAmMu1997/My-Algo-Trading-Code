"""Specify the deterministic Trend-Day Rider candidate gate.

Every fixture is hand-built so its session range, close location, and VWAP are
known exactly: a first bar that sets one extreme, flat middle bars at 100, and
a final bar at a chosen time and close. That makes each boundary in the plan --
window edges, the strict range > ATR test, the 0.85/0.15 location cut, the VWAP
side, and the bullish-only confluence rule -- a one-number change.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd
import pytest
from cpr_ai_trend_day import (
    TrendDayConfig,
    evaluate_trend_day_candidate,
    session_atr,
    session_vwap,
)

SESSION = datetime(2026, 9, 24, 9, 15)


OHLC = tuple[float, float, float, float]


def _bars(first: OHLC, last: OHLC, *, last_start: time) -> pd.DataFrame:
    """Build 09:15..``last_start`` five-minute bars: an extreme-setting first bar, flat 100s, a final bar."""

    count = (datetime.combine(SESSION.date(), last_start) - SESSION) // timedelta(minutes=5) + 1
    rows = []
    for index in range(count):
        ohlc = first if index == 0 else last if index == count - 1 else (100.0, 100.0, 100.0, 100.0)
        rows.append(
            {
                "timestamp": SESSION + timedelta(minutes=5 * index),
                "open": ohlc[0],
                "high": ohlc[1],
                "low": ohlc[2],
                "close": ohlc[3],
                "volume": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _bull(close: float = 170.0, *, last_start: time = time(11, 0)) -> pd.DataFrame:
    """Session low 0 (first bar), high 200 (last bar): location = close / 200."""

    return _bars((100.0, 100.0, 0.0, 100.0), (100.0, 200.0, 100.0, close), last_start=last_start)


def _bear(close: float = 30.0) -> pd.DataFrame:
    """Session high 200 (first bar), low 0 (last bar): location = close / 200."""

    return _bars((100.0, 200.0, 100.0, 100.0), (100.0, 100.0, 0.0, close), last_start=time(11, 0))


# Prior day chosen so R1 is far above and S1 far below the fixtures: only the
# gap and VWAP-extension confluence factors are in play unless a test says so.
FAR_LEVELS = {"prior_high": 1000.0, "prior_low": 0.0}


def _assess(bars: pd.DataFrame, *, prior_close: float = 90.0, atr: float | None = 100.0, **levels: float):
    """Evaluate with far CPR levels by default; ``prior_close`` 90 means a gap up from open 100."""

    merged = {**FAR_LEVELS, **levels}
    return evaluate_trend_day_candidate(bars, prior_close=prior_close, atr=atr, atr_sessions_used=5, **merged)


def test_bullish_trend_day_is_a_candidate_with_vwap_stop() -> None:
    result = _assess(_bull())
    vwap = float(session_vwap(_bull()).iloc[-1])

    assert result.eligible and result.reason == "eligible"
    assert result.direction == "LONG"
    assert result.entry == 170.0
    assert result.stop == pytest.approx(vwap)
    assert result.risk_points == pytest.approx(170.0 - vwap)
    assert result.location == pytest.approx(0.85)
    assert result.range_atr == pytest.approx(2.0)
    assert (result.gap_in_direction, result.extended_from_vwap, result.beyond_r1_s1) == (True, True, False)
    assert result.confluence_score == 2


@pytest.mark.parametrize(
    ("last_start", "in_window"),
    [(time(10, 55), False), (time(11, 0), True), (time(13, 30), True), (time(13, 35), False)],
)
def test_window_is_inclusive_bar_start_11_00_through_13_30(last_start: time, in_window: bool) -> None:
    result = _assess(_bull(last_start=last_start))

    assert result.in_window is in_window
    assert result.eligible is in_window
    assert result.reason == ("eligible" if in_window else "outside_window")


def test_range_must_strictly_exceed_atr() -> None:
    # R1 below the close keeps confluence at 2 once the larger ATR removes
    # the VWAP-extension factor, so only the range gate is being tested.
    near_r1 = {"prior_high": 150.0, "prior_low": 100.0}
    assert _assess(_bull(), atr=200.0, **near_r1).reason == "range_not_expanded"
    assert _assess(_bull(), atr=199.0, **near_r1).eligible


def test_missing_atr_never_forms_a_candidate() -> None:
    result = _assess(_bull(), atr=None)

    assert not result.eligible and result.reason == "atr_unavailable"


def test_bullish_location_boundary_is_0_85() -> None:
    assert _assess(_bull(170.0)).eligible
    below = _assess(_bull(169.8))
    assert not below.eligible and below.reason == "not_at_trend_extreme" and below.direction is None


def test_bearish_location_boundary_is_0_15_and_needs_no_confluence() -> None:
    # atr 199 keeps the VWAP extension just under 0.35 ATR; open 100 is above
    # prior close 90 (no gap down) and S1 is far below: score 0.
    result = _assess(_bear(30.0), atr=199.0)

    assert result.eligible and result.direction == "SHORT"
    assert result.confluence_score == 0
    assert result.stop == pytest.approx(float(session_vwap(_bear(30.0)).iloc[-1]))
    assert _assess(_bear(30.2), atr=199.0).reason == "not_at_trend_extreme"


def test_close_on_the_wrong_side_of_vwap_is_not_a_trend_extreme() -> None:
    # Location 0.9 but the flat bars sit at 190, so VWAP is above the close.
    bars = _bars((190.0, 190.0, 0.0, 190.0), (190.0, 200.0, 180.0, 180.0), last_start=time(11, 0))
    bars.loc[1:len(bars) - 2, ["open", "high", "low", "close"]] = 190.0

    result = _assess(bars)

    assert result.location == pytest.approx(0.9)
    assert result.direction is None and result.reason == "not_at_trend_extreme"


def test_bullish_needs_two_confluence_factors() -> None:
    no_gap = _assess(_bull(), prior_close=110.0)  # open 100 below prior close
    assert (no_gap.gap_in_direction, no_gap.extended_from_vwap) == (False, True)
    assert no_gap.confluence_score == 1
    assert not no_gap.eligible and no_gap.reason == "confluence_too_low"

    # Moving R1 below the close restores a second factor without the gap.
    beyond_r1 = _assess(_bull(), prior_close=110.0, prior_high=150.0, prior_low=100.0)
    assert beyond_r1.beyond_r1_s1 and beyond_r1.confluence_score == 2 and beyond_r1.eligible


def test_session_atr_uses_the_newest_five_valid_prior_ranges() -> None:
    atr, used = session_atr([1000.0, 900.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    assert (atr, used) == (30.0, 5)

    atr, used = session_atr([float("nan"), 0.0, -5.0, 60.0, 30.0])
    assert (atr, used) == (None, 2)

    atr, used = session_atr([30.0, 60.0, 90.0])
    assert (atr, used) == (60.0, 3)


def test_vwap_is_volume_weighted_only_when_volume_exists() -> None:
    bars = _bull()
    assert float(session_vwap(bars).iloc[-1]) == pytest.approx(float(((bars.high + bars.low + bars.close) / 3).mean()))

    weighted = bars.assign(volume=0.0)
    weighted.loc[len(weighted) - 1, "volume"] = 10.0
    assert float(session_vwap(weighted).iloc[-1]) == pytest.approx((200.0 + 100.0 + 170.0) / 3)


def test_empty_session_is_never_eligible() -> None:
    result = evaluate_trend_day_candidate(
        pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"]),
        prior_high=1.0, prior_low=0.0, prior_close=0.5, atr=1.0,
    )
    assert not result.eligible and result.reason == "no_session_bars"


@pytest.mark.parametrize(
    "overrides",
    [
        {"window_start": time(14, 0), "window_end": time(11, 0)},
        {"range_atr_multiple": 0.0},
        {"extension_atr_multiple": float("nan")},
        {"location_threshold": 0.5},
        {"min_atr_sessions": 6},
        {"long_min_confluence": 4},
    ],
)
def test_config_rejects_impossible_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrendDayConfig(**overrides)  # type: ignore[arg-type]


def test_assessment_serializes_for_the_frozen_context() -> None:
    payload = _assess(_bull()).to_dict()

    assert payload["eligible"] is True and payload["direction"] == "LONG"
    assert payload["bar_start"] == "2026-09-24T11:00:00"
