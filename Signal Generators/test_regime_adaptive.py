"""Tests for the regime-adaptive port: session helpers, branches, and routing.

The generic port contract (config validation, decision shape, determinism) is
covered by the parametrized suite in `test_trading_bot_ports.py`. What is tested
here is the behaviour that suite cannot see: that the opening range is honest
about when it exists, that VWAP degrades visibly rather than silently, and that
the router refuses to guess a regime it cannot measure.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SIGNAL_DIR = Path(__file__).resolve().parent
if str(SIGNAL_DIR) not in sys.path:
    sys.path.insert(0, str(SIGNAL_DIR))

import regime_common  # noqa: E402
from regime_candidates import (  # noqa: E402
    attach_breakout_columns,
    attach_mean_reversion_columns,
)


def _load_router():
    """Load the spaced-filename generator the way the master file does."""
    path = SIGNAL_DIR / "Nifty Regime Adaptive Signal Generator.py"
    spec = importlib.util.spec_from_file_location("regime_adaptive_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROUTER = _load_router()


def _session(day: str, bars: int, *, start_price: float = 25000.0, step: float = 1.0,
             minutes_step: int = 1, volume: float | None = None) -> pd.DataFrame:
    base = datetime.fromisoformat(f"{day} 09:15:00")
    rows = []
    price = start_price
    for i in range(bars):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=i * minutes_step),
                "open": price,
                "high": price + 5.0,
                "low": price - 5.0,
                "close": price + step,
                **({"volume": volume} if volume is not None else {}),
            }
        )
        price += step
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Opening range
# ---------------------------------------------------------------------------

def test_opening_range_is_hidden_until_the_window_closes():
    """A half-built range must never be publishable — you cannot break a level
    that is still being formed."""
    frame = regime_common.attach_session_date(_session("2026-08-03", 30))
    result = regime_common.attach_opening_range(frame, minutes=15)

    inside = result.iloc[:15]
    after = result.iloc[15:]
    assert inside["or_high"].isna().all(), "range leaked while still forming"
    assert inside["or_low"].isna().all()
    assert not inside["or_complete"].any()
    assert after["or_high"].notna().all()
    assert after["or_complete"].all()


def test_opening_range_matches_the_window_extremes():
    frame = regime_common.attach_session_date(_session("2026-08-03", 30))
    result = regime_common.attach_opening_range(frame, minutes=15)
    window = frame.iloc[:15]
    assert result.iloc[20]["or_high"] == pytest.approx(window["high"].max())
    assert result.iloc[20]["or_low"] == pytest.approx(window["low"].min())


def test_opening_range_never_leaks_across_sessions():
    """Yesterday's range surviving into today would be silently catastrophic."""
    day_one = _session("2026-08-03", 30, start_price=25000.0)
    day_two = _session("2026-08-04", 30, start_price=24000.0)
    frame = regime_common.attach_session_date(pd.concat([day_one, day_two], ignore_index=True))
    result = regime_common.attach_opening_range(frame, minutes=15)

    second_day = result[result["session_date"] == pd.Timestamp("2026-08-04").date()]
    assert second_day.iloc[:15]["or_high"].isna().all()
    # Day two's range must come from day two's own prices, nowhere near 25000.
    assert second_day.iloc[-1]["or_high"] < 24100.0


def test_opening_range_rejects_a_nonsense_window():
    frame = regime_common.attach_session_date(_session("2026-08-03", 5))
    with pytest.raises(ValueError, match="positive"):
        regime_common.attach_opening_range(frame, minutes=0)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def test_vwap_falls_back_to_the_equal_weight_proxy_and_says_so():
    """The live feed carries no volume. The fallback must be visible, not silent."""
    frame = regime_common.attach_session_date(_session("2026-08-03", 10))
    result = regime_common.attach_session_vwap(frame)

    typical = (result["high"] + result["low"] + result["close"]) / 3.0
    assert result["vwap"].tolist() == pytest.approx(typical.expanding().mean().tolist())
    assert result["vwap_is_proxy"].all(), "proxy VWAP must be flagged"


def test_vwap_uses_volume_when_the_frame_has_it():
    frame = regime_common.attach_session_date(_session("2026-08-03", 10, volume=1000.0))
    result = regime_common.attach_session_vwap(frame)
    assert not result["vwap_is_proxy"].any()


def test_vwap_resets_each_session():
    day_one = _session("2026-08-03", 10, start_price=25000.0)
    day_two = _session("2026-08-04", 10, start_price=20000.0)
    frame = regime_common.attach_session_date(pd.concat([day_one, day_two], ignore_index=True))
    result = regime_common.attach_session_vwap(frame)
    # The first bar of day two anchors on day two's own price, not a carried mean.
    first_of_day_two = result.iloc[10]
    assert first_of_day_two["vwap"] < 21000.0


# ---------------------------------------------------------------------------
# Candidate branches — fail-closed behaviour
# ---------------------------------------------------------------------------

def _branch_frame() -> pd.DataFrame:
    frame = regime_common.prepare_session_frame(_session("2026-08-03", 40), 15)
    frame["atr"] = 10.0
    frame["adx"] = 15.0
    return frame


def test_breakout_produces_no_setup_without_atr():
    frame = _branch_frame()
    frame["atr"] = np.nan
    result = attach_breakout_columns(
        frame, buffer_atr_mult=0.05, buffer_min_points=2.0, stop_loss_pct=0.015, target_pct=0.03
    )
    assert not result["br_long_setup"].any()
    assert not result["br_short_setup"].any()


def test_mean_reversion_stands_down_in_a_trend_and_without_adx():
    frame = _branch_frame()
    # Force a large stretch below VWAP so only the regime gate can block it.
    frame["vwap"] = frame["close"] + 100.0

    trending = frame.copy()
    trending["adx"] = 40.0
    blocked = attach_mean_reversion_columns(
        trending, atr_mult=1.5, adx_ceiling=25.0, stop_loss_pct=0.015, target_pct=0.03
    )
    assert not blocked["mr_long_setup"].any(), "faded a trending market"

    missing = frame.copy()
    missing["adx"] = np.nan
    unknown = attach_mean_reversion_columns(
        missing, atr_mult=1.5, adx_ceiling=25.0, stop_loss_pct=0.015, target_pct=0.03
    )
    assert not unknown["mr_long_setup"].any(), "traded without knowing the regime"

    ranging = frame.copy()
    ranging["adx"] = 10.0
    allowed = attach_mean_reversion_columns(
        ranging, atr_mult=1.5, adx_ceiling=25.0, stop_loss_pct=0.015, target_pct=0.03
    )
    assert allowed["mr_long_setup"].any(), "fade never fires even in a clear range"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def test_router_holds_when_adx_is_unknown():
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    frame.loc[frame.index[-1], "regime"] = ROUTER.NO_BRANCH
    decision = engine.evaluate_candle(frame)
    assert decision.action == "HOLD"
    assert decision.debug.get("reason") == "regime_unknown"


def test_router_selects_the_branch_from_adx():
    config = ROUTER.RegimeAdaptiveConfig(adx_trend_threshold=20.0)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    adx = frame["adx"].astype(float)
    trending_rows = frame.loc[adx.notna() & (adx >= 20.0), "regime"]
    ranging_rows = frame.loc[adx.notna() & (adx < 20.0), "regime"]
    assert (trending_rows == ROUTER.BREAKOUT_BRANCH).all()
    assert (ranging_rows == ROUTER.MEAN_REVERSION_BRANCH).all()
    assert (frame.loc[adx.isna(), "regime"] == ROUTER.NO_BRANCH).all()


def test_router_exit_uses_stop_and_target_regardless_of_regime():
    """A regime flip mid-trade must never strand an open position."""
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    frame.loc[frame.index[-1], "regime"] = ROUTER.NO_BRANCH  # regime became unknown
    last = frame.iloc[-1]

    stopped = engine.evaluate_candle(
        frame,
        position=ROUTER.RegimeAdaptivePositionContext(
            direction="LONG",
            entry_underlying=float(last["close"]),
            stop_underlying=float(last["high"]) + 1.0,  # already through the stop
            target_underlying=float(last["close"]) * 2.0,
        ),
    )
    assert stopped.action == "EXIT"
    assert stopped.exit_reason == "STOP"


def _short_position(frame, *, stop: float, target: float):
    return ROUTER.RegimeAdaptivePositionContext(
        direction="SHORT",
        entry_underlying=float(frame.iloc[-1]["close"]),
        stop_underlying=stop,
        target_underlying=target,
    )


def test_short_exit_honours_stop_then_target_then_opposite_signal():
    """The SHORT side is a mirror, not a copy — its comparisons are inverted, so
    it needs its own coverage or a flipped inequality would ship unnoticed."""
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    last = frame.iloc[-1]
    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])

    # Stop first: for a SHORT the stop sits ABOVE and triggers on `high >= stop`.
    stopped = engine.evaluate_candle(frame, position=_short_position(frame, stop=low - 1.0, target=0.0))
    assert stopped.action == "EXIT"
    assert stopped.exit_reason == "STOP"

    # Target: unreachable stop, target touched from below (`low <= target`).
    hit = engine.evaluate_candle(frame, position=_short_position(frame, stop=high + 1000.0, target=close))
    assert hit.action == "EXIT"
    assert hit.exit_reason == "TARGET"

    # Opposite signal: neither level touched, but the frame now says go long.
    frame.loc[frame.index[-1], "long_setup"] = True
    flipped = engine.evaluate_candle(frame, position=_short_position(frame, stop=high + 1000.0, target=0.0))
    assert flipped.action == "EXIT"
    assert flipped.exit_reason == "OPPOSITE_SIGNAL"

    # Nothing triggered at all is a plain hold, not an exit.
    frame.loc[frame.index[-1], "long_setup"] = False
    held = engine.evaluate_candle(frame, position=_short_position(frame, stop=high + 1000.0, target=0.0))
    assert held.action == "HOLD"


def test_long_exit_honours_target_and_opposite_signal():
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    last = frame.iloc[-1]
    low = float(last["low"])
    close = float(last["close"])

    def long_position(*, stop: float, target: float):
        return ROUTER.RegimeAdaptivePositionContext(
            direction="LONG", entry_underlying=close, stop_underlying=stop, target_underlying=target
        )

    hit = engine.evaluate_candle(frame, position=long_position(stop=low - 1000.0, target=close))
    assert hit.action == "EXIT"
    assert hit.exit_reason == "TARGET"

    frame.loc[frame.index[-1], "short_setup"] = True
    flipped = engine.evaluate_candle(frame, position=long_position(stop=low - 1000.0, target=0.0))
    assert flipped.action == "EXIT"
    assert flipped.exit_reason == "OPPOSITE_SIGNAL"


def test_unknown_direction_holds_rather_than_guessing():
    """A corrupt direction must not be silently treated as one side or the other."""
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    decision = engine.evaluate_candle(
        frame,
        position=ROUTER.RegimeAdaptivePositionContext(
            direction="SIDEWAYS", entry_underlying=1.0, stop_underlying=1.0, target_underlying=1.0
        ),
    )
    assert decision.action == "HOLD"
    assert decision.debug.get("reason") == "unknown_direction"


def test_entries_read_the_collapsed_columns_for_both_sides():
    """Both ENTER paths must carry real levels through to the master's
    enter_position -- a zero stop would size the trade off nothing."""
    config = ROUTER.RegimeAdaptiveConfig()
    engine = ROUTER.RegimeAdaptiveSignalEngine(config)
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)
    index = frame.index[-1]
    frame.loc[index, "regime"] = ROUTER.MEAN_REVERSION_BRANCH

    for side, action in (("long", "ENTER_LONG"), ("short", "ENTER_SHORT")):
        other = "short" if side == "long" else "long"
        frame.loc[index, f"{side}_setup"] = True
        frame.loc[index, f"{other}_setup"] = False
        frame.loc[index, f"{side}_entry_price"] = 25000.0
        frame.loc[index, f"{side}_stop_from_setup"] = 24800.0
        frame.loc[index, f"{side}_target_from_setup"] = 25300.0

        decision = engine.evaluate_candle(frame)
        assert decision.action == action
        assert decision.signal_triggered
        assert decision.entry_underlying == pytest.approx(25000.0)
        assert decision.stop_underlying == pytest.approx(24800.0)
        assert decision.target_underlying == pytest.approx(25300.0)
        assert decision.reason == f"{ROUTER.MEAN_REVERSION_BRANCH}_{side}"
        assert decision.debug["regime"] == ROUTER.MEAN_REVERSION_BRANCH


def test_collapsed_levels_stay_numeric_not_boolean():
    """Regression: `long_stop_from_setup` ends in "setup" too. Collapsing the
    branches by suffix turned the stop and target into booleans, so every entry
    handed the master a stop of 0.0 or 1.0 — real money sized off nothing."""
    config = ROUTER.RegimeAdaptiveConfig()
    frame = ROUTER.build_regime_adaptive_with_indicators(_session("2026-08-03", 80), config)

    for column in ("long_setup", "short_setup"):
        assert frame[column].dtype == bool, f"{column} should be a flag"
    for column in (
        "long_entry_price",
        "short_entry_price",
        "long_stop_from_setup",
        "short_stop_from_setup",
        "long_target_from_setup",
        "short_target_from_setup",
    ):
        assert frame[column].dtype == float, f"{column} must stay a price level"


def test_empty_frame_holds():
    engine = ROUTER.RegimeAdaptiveSignalEngine(ROUTER.RegimeAdaptiveConfig())
    assert engine.evaluate_candle(pd.DataFrame()).action == "HOLD"
    assert ROUTER.build_regime_adaptive_with_indicators(pd.DataFrame(), None).empty


def test_config_rejects_non_positive_periods_and_distances():
    with pytest.raises(ValueError, match="positive"):
        ROUTER.RegimeAdaptiveConfig(adx_period=0)
    with pytest.raises(ValueError, match="greater than zero"):
        ROUTER.RegimeAdaptiveConfig(stop_loss_pct=0.0)
    with pytest.raises(ValueError, match="ADX thresholds"):
        ROUTER.RegimeAdaptiveConfig(adx_trend_threshold=0.0)
    with pytest.raises(ValueError, match="fade distance"):
        ROUTER.RegimeAdaptiveConfig(meanrev_atr_mult=0.0)


def test_minimum_history_bars_is_sane():
    engine = ROUTER.RegimeAdaptiveSignalEngine(ROUTER.RegimeAdaptiveConfig())
    bars = engine.minimum_history_bars()
    assert isinstance(bars, int)
    assert 0 < bars < 5000


def test_config_rejects_percentages_at_or_above_one():
    with pytest.raises(ValueError, match="percentage"):
        ROUTER.RegimeAdaptiveConfig(stop_loss_pct=1.0)
