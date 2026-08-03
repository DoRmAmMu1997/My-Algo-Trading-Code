"""Shared session-aware helpers for the regime-adaptive strategy family.

WHAT THIS IS FOR
----------------
The regime-adaptive port needs three things the repo did not already have:
a session date, a session VWAP, and a session OPENING RANGE. Everything else it
needs (ATR, ADX, EMA, frame normalisation, config validation) already lives in
`misc_strategy_common.py` and is reused from there rather than reimplemented.

Kept in an underscore-named module (not a spaced "... Signal Generator.py" file)
so it can be imported normally and type-checked by mypy.

A NOTE ON VWAP THAT MATTERS FOR ANYONE READING THE SIGNALS
----------------------------------------------------------
The live runner never sees volume. `normalize_dhan_intraday_response` in the
master file emits only timestamp/open/high/low/close, so a true volume-weighted
VWAP is impossible during a live or paper session. `attach_session_vwap` falls
back to an equal-weight average of the typical price since the session open and
sets a `vwap_is_proxy` column to True so the distinction is visible downstream
instead of silently assumed. Backtest frames from `Data Extractors/` DO carry
volume, and there the real volume-weighted VWAP is used.

This mirrors the existing precedent in `CPR Strategy/cpr_strategy_logic.py`
(volume-weighted with an equal-weight fallback); it is not a new invention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from misc_strategy_common import normalize_ohlc_frame, require_columns

__all__ = [
    "attach_opening_range",
    "attach_session_date",
    "attach_session_vwap",
    "prepare_session_frame",
]


def attach_session_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a `session_date` column so per-day grouping never leaks across days.

    Every helper below groups by this column. Without it a cumulative maximum
    would happily carry yesterday's opening-range high into today.
    """
    require_columns(frame, ["timestamp"])
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
    result["session_date"] = timestamps.dt.date
    return result


def attach_session_vwap(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `vwap` and `vwap_is_proxy`, resetting at every session open.

    Volume-weighted when a usable `volume` column exists; otherwise the
    equal-weight cumulative mean of the typical price since the session open.
    `vwap_is_proxy` records which of the two produced each row, so a strategy
    that behaves differently on proxy VWAP can be audited after the fact.
    """
    require_columns(frame, ["high", "low", "close", "session_date"])
    result = frame.copy()

    typical_price = (
        result["high"].astype(float) + result["low"].astype(float) + result["close"].astype(float)
    ) / 3.0
    sessions = result["session_date"]

    # Equal-weight fallback: the running mean of typical price within the day.
    equal_weight = typical_price.groupby(sessions).expanding().mean()
    equal_weight = equal_weight.reset_index(level=0, drop=True).sort_index()

    volume_vwap = pd.Series(np.nan, index=result.index, dtype="float64")
    if "volume" in result.columns:
        volume = pd.to_numeric(result["volume"], errors="coerce").fillna(0.0).astype(float)
        cumulative_volume = volume.groupby(sessions).cumsum()
        cumulative_value = (typical_price * volume).groupby(sessions).cumsum()
        # Zero cumulative volume (index candles) leaves NaN, which the fillna
        # below replaces with the equal-weight figure.
        volume_vwap = cumulative_value / cumulative_volume.replace(0.0, np.nan)

    result["vwap"] = volume_vwap.fillna(equal_weight)
    result["vwap_is_proxy"] = volume_vwap.isna()
    return result


def attach_opening_range(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Add `or_high`, `or_low` and `or_complete` for a session opening range.

    Three properties this deliberately guarantees, because getting any of them
    wrong would let the strategy trade a range that does not exist yet:

    1. The range is built ONLY from bars whose timestamp falls inside the first
       `minutes` of that session.
    2. `or_high` / `or_low` stay NaN on every bar inside the window. A partially
       formed range is never published, so no bar can break a range that is
       still being built.
    3. Grouping is per `session_date`, so yesterday's range can never survive
       into today.

    NOTE: this is NOT what `Nifty Opening Range Breakout Signal Generator.py`
    computes. That file puts an ATR band around each candle's own open, which is
    a different idea despite the similar name.
    """
    require_columns(frame, ["timestamp", "high", "low", "session_date"])
    if int(minutes) <= 0:
        raise ValueError("opening-range minutes must be positive")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
    sessions = result["session_date"]

    session_open_ts = timestamps.groupby(sessions).transform("min")
    minutes_elapsed = (timestamps - session_open_ts).dt.total_seconds() / 60.0
    inside_window = minutes_elapsed < float(minutes)

    # Mask to the window, take the running extreme, then forward-fill so bars
    # after the window keep carrying the finished range.
    windowed_high = result["high"].astype(float).where(inside_window)
    windowed_low = result["low"].astype(float).where(inside_window)
    running_high = windowed_high.groupby(sessions).cummax().groupby(sessions).ffill()
    running_low = windowed_low.groupby(sessions).cummin().groupby(sessions).ffill()

    # Hide the range until the window has actually closed (property 2 above).
    result["or_high"] = running_high.where(~inside_window)
    result["or_low"] = running_low.where(~inside_window)
    result["or_complete"] = result["or_high"].notna() & result["or_low"].notna()
    return result


def prepare_session_frame(ohlc: pd.DataFrame, opening_range_minutes: int) -> pd.DataFrame:
    """Normalise, then attach session date, VWAP and opening range in one pass.

    The worker factory calls exactly ONE `build_*_with_indicators`, so every
    session-derived column both router branches need has to be produced here.
    """
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame
    frame = attach_session_date(frame)
    frame = attach_session_vwap(frame)
    frame = attach_opening_range(frame, opening_range_minutes)
    return frame
