"""
Shared CPR strategy logic for 5-minute OHLC candles.

This file is the reusable "brain" for the CPR strategy. It does three jobs:
1. Convert raw OHLC data into a clean, time-sorted candle table.
2. Add CPR levels, EMA/VWAP/RSI, swing, zone, and divergence columns.
3. Read the latest candle and return one simple trading decision.

The PDF contains a few chart-reading phrases like "minimal wick", "swing",
"support/resistance zone", and "RSI divergence". Those are translated into
explicit configurable defaults in `CPRStrategyConfig`, so the backtest and any
future front-test use the same repeatable rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except ImportError:  # pragma: no cover - fallback path is used when TA-Lib is absent
    talib = None


OHLC_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class CPRStrategyConfig:
    """All tunable CPR strategy settings live in one beginner-friendly place."""

    # Indicator periods from the PDF.
    ema_fast_period: int = 5
    ema_slow_period: int = 20
    rsi_period: int = 14
    rsi_ema_period: int = 20

    # "Minimal upper/lower wick" is coded as a maximum wick/range ratio.
    max_entry_wick_ratio: float = 0.25

    # Algo 1 condition 2 uses the PDF's 0.5% directional move idea.
    trend_move_pct: float = 0.005

    # Swing highs/lows are confirmed fractals. A value of 3 means the center
    # candle must be higher/lower than 3 candles on each side.
    swing_lookback: int = 3

    # Algo 2 support/resistance zones.
    zone_width_points: float = 10.0
    zone_lookback: int = 60
    zone_min_touches: int = 2

    # RSI divergence between two swing highs or two swing lows.
    divergence_min_separation: int = 12
    divergence_requires_trend_move: bool = True

    # Minimum history before any entry is allowed.
    min_history_bars: int = 30

    # Optional metadata used by wrappers/debugging.
    enabled_algorithms: tuple[str, ...] = field(
        default=("ALGO1", "ALGO2_ZONE", "RSI_DIVERGENCE")
    )

    def __post_init__(self) -> None:
        positive_ints = {
            "ema_fast_period": self.ema_fast_period,
            "ema_slow_period": self.ema_slow_period,
            "rsi_period": self.rsi_period,
            "rsi_ema_period": self.rsi_ema_period,
            "swing_lookback": self.swing_lookback,
            "zone_lookback": self.zone_lookback,
            "zone_min_touches": self.zone_min_touches,
            "divergence_min_separation": self.divergence_min_separation,
            "min_history_bars": self.min_history_bars,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if float(self.max_entry_wick_ratio) < 0.0:
            raise ValueError("max_entry_wick_ratio must be non-negative.")
        if float(self.trend_move_pct) < 0.0:
            raise ValueError("trend_move_pct must be non-negative.")
        if float(self.zone_width_points) <= 0.0:
            raise ValueError("zone_width_points must be greater than zero.")


@dataclass
class CPRPositionContext:
    """Small summary of the currently open trade."""

    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float
    strategy_name: str = ""


@dataclass
class CPRDecision:
    """
    Standard answer returned by the signal engine.

    `action` can be HOLD, ENTER_LONG, ENTER_SHORT, or EXIT.
    """

    action: str = "HOLD"
    strategy_name: str = ""
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    target_underlying: float = 0.0
    exit_underlying: float = 0.0
    exit_reason: str = ""
    signal_triggered: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def _find_first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """Find a column name case-insensitively."""
    col_map = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _require_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    """Fail early when the caller has not supplied the needed columns."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _normalize_ohlc_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Accept common CSV column names and return lowercase OHLC columns."""
    if data is None or data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ts_col = _find_first_col(data, ["timestamp", "datetime", "date", "time"])
    o_col = _find_first_col(data, ["open", "Open"])
    h_col = _find_first_col(data, ["high", "High"])
    l_col = _find_first_col(data, ["low", "Low"])
    c_col = _find_first_col(data, ["close", "Close"])
    v_col = _find_first_col(data, ["volume", "Volume"])

    if not all([ts_col, o_col, h_col, l_col, c_col]):
        raise ValueError("Input data must contain timestamp/date plus open/high/low/close columns.")

    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data[ts_col], errors="coerce"),
            "open": pd.to_numeric(data[o_col], errors="coerce"),
            "high": pd.to_numeric(data[h_col], errors="coerce"),
            "low": pd.to_numeric(data[l_col], errors="coerce"),
            "close": pd.to_numeric(data[c_col], errors="coerce"),
        }
    )

    if v_col is not None:
        result["volume"] = pd.to_numeric(data[v_col], errors="coerce").fillna(0.0)
    else:
        result["volume"] = 0.0

    result = result.dropna(subset=["timestamp", "open", "high", "low", "close"])
    result = result.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return result


def prepare_cpr_ohlc_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return complete 5-minute candles.

    The project mostly stores 1-minute data, but the CPR rules expect 5-minute
    candles. This helper accepts both:
    - true 5-minute input: returned unchanged after cleaning
    - 1-minute input: resampled into complete 5-minute bars
    """
    frame = _normalize_ohlc_frame(data)
    if len(frame) < 2:
        return frame

    deltas = frame["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return frame
    median_minutes = float(positive_deltas.median())

    if 4.5 <= median_minutes <= 5.5:
        return frame
    if not (0.5 <= median_minutes <= 1.5):
        raise ValueError(
            "CPR strategy expects either 1-minute or 5-minute OHLC data; "
            f"detected median spacing of {median_minutes:.2f} minutes."
        )

    source = frame.set_index("timestamp")
    resampled = source.resample("5min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Keep only complete 5-minute candles. If the latest 1-minute snapshot is
    # still forming, that partial bar is dropped so signals never repaint.
    counts = source["close"].resample("5min", label="left", closed="left").count()
    resampled = resampled[counts == 5].dropna(subset=OHLC_COLUMNS)
    return resampled.reset_index()


def classify_daily_cpr_width(high: float, low: float, close: float) -> str:
    """Classify CPR width using the daily thresholds from the PDF."""
    values = [high, low, close]
    if not all(np.isfinite(value) for value in values):
        return "unknown"

    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    r1 = 2.0 * pivot - low
    if pivot <= 0.0 or close <= 0.0:
        return "unknown"

    r1_minus_pivot = r1 - pivot
    pivot_minus_bc = pivot - bc
    if (r1_minus_pivot < pivot * 0.01) and (pivot_minus_bc < close * 0.009):
        return "narrow"
    if (r1_minus_pivot < pivot * 0.02) and (pivot_minus_bc < close * 0.018):
        return "medium"
    return "wide"


def _ema(values: pd.Series, period: int) -> pd.Series:
    """Calculate EMA with TA-Lib when available, otherwise pandas."""
    if talib is not None:
        return pd.Series(
            talib.EMA(values.to_numpy(dtype=float), timeperiod=int(period)),  # type: ignore[union-attr]
            index=values.index,
        )
    return values.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculate Wilder-style RSI."""
    if talib is not None:
        return pd.Series(
            talib.RSI(close.to_numpy(dtype=float), timeperiod=int(period)),  # type: ignore[union-attr]
            index=close.index,
        )

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean()
    avg_loss = loss.ewm(alpha=1.0 / int(period), adjust=False, min_periods=int(period)).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    result = result.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return result


def _add_daily_cpr(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-session CPR levels to every intraday candle."""
    result = frame.copy()
    result["session_date"] = result["timestamp"].dt.date

    daily = (
        result.groupby("session_date", sort=True)
        .agg(prev_high=("high", "max"), prev_low=("low", "min"), prev_close=("close", "last"))
        .reset_index()
    )
    daily[["prev_high", "prev_low", "prev_close"]] = daily[
        ["prev_high", "prev_low", "prev_close"]
    ].shift(1)

    result = result.merge(daily, on="session_date", how="left")

    high = result["prev_high"]
    low = result["prev_low"]
    close = result["prev_close"]
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * pivot - bc

    result["pivot"] = pivot
    result["bc"] = bc
    result["tc"] = tc
    result["cpr_lower"] = pd.concat([bc, tc], axis=1).min(axis=1)
    result["cpr_upper"] = pd.concat([bc, tc], axis=1).max(axis=1)
    result["r1"] = 2.0 * pivot - low
    result["r2"] = pivot + high - low
    result["r3"] = high + 2.0 * (pivot - low)
    result["r4"] = result["r3"] + high - low
    result["s1"] = 2.0 * pivot - high
    result["s2"] = pivot - (high - low)
    result["s3"] = low - 2.0 * (high - pivot)
    result["s4"] = result["s3"] - (high - low)
    result["cpr_width"] = [
        classify_daily_cpr_width(h, l, c)
        for h, l, c in zip(result["prev_high"], result["prev_low"], result["prev_close"])
    ]
    return result


def _add_session_indicators(frame: pd.DataFrame, config: CPRStrategyConfig) -> pd.DataFrame:
    """Attach indicators that reset by trading session where needed."""
    result = frame.copy()
    close = result["close"].astype(float)
    result["ema5"] = _ema(close, config.ema_fast_period)
    result["ema20"] = _ema(close, config.ema_slow_period)
    result["rsi"] = _rsi(close, config.rsi_period)
    result["rsi_ema20"] = _ema(result["rsi"], config.rsi_ema_period)

    typical_price = (result["high"] + result["low"] + result["close"]) / 3.0
    volume = result["volume"].fillna(0.0).astype(float)
    weighted_value = typical_price * volume
    cumulative_volume = volume.groupby(result["session_date"]).cumsum()
    cumulative_value = weighted_value.groupby(result["session_date"]).cumsum()
    volume_vwap = cumulative_value / cumulative_volume.replace(0.0, np.nan)

    # Index candles often have zero volume. In that case, use the equal-weight
    # session VWAP proxy: cumulative average of typical price since session open.
    equal_weight_vwap = typical_price.groupby(result["session_date"]).expanding().mean()
    equal_weight_vwap = equal_weight_vwap.reset_index(level=0, drop=True).sort_index()
    result["vwap"] = volume_vwap.fillna(equal_weight_vwap)

    grouped = result.groupby("session_date", sort=False)
    result["session_open"] = grouped["open"].transform("first")
    result["session_high_so_far"] = grouped["high"].cummax()
    result["session_low_so_far"] = grouped["low"].cummin()
    result["bullish_half_pct_move_seen"] = (
        result["session_high_so_far"] >= result["session_open"] * (1.0 + config.trend_move_pct)
    )
    result["bearish_half_pct_move_seen"] = (
        result["session_low_so_far"] <= result["session_open"] * (1.0 - config.trend_move_pct)
    )
    return result


def _add_candle_anatomy(frame: pd.DataFrame, config: CPRStrategyConfig) -> pd.DataFrame:
    """Add body/wick measurements used by the entry filters."""
    result = frame.copy()
    result["candle_range"] = (result["high"] - result["low"]).abs()
    result["candle_body"] = (result["close"] - result["open"]).abs()
    result["upper_wick"] = result["high"] - result[["open", "close"]].max(axis=1)
    result["lower_wick"] = result[["open", "close"]].min(axis=1) - result["low"]
    safe_range = result["candle_range"].replace(0.0, np.nan)
    result["upper_wick_ratio"] = result["upper_wick"] / safe_range
    result["lower_wick_ratio"] = result["lower_wick"] / safe_range
    result["bullish_candle"] = result["close"] > result["open"]
    result["bearish_candle"] = result["close"] < result["open"]
    result["minimal_upper_wick"] = result["upper_wick_ratio"] <= config.max_entry_wick_ratio
    result["minimal_lower_wick"] = result["lower_wick_ratio"] <= config.max_entry_wick_ratio
    return result


def _add_swings_and_divergence(frame: pd.DataFrame, config: CPRStrategyConfig) -> pd.DataFrame:
    """Confirm swing highs/lows and mark RSI divergence on confirmation bars."""
    result = frame.copy()
    n = len(result)
    lookback = int(config.swing_lookback)

    confirmed_high = np.zeros(n, dtype=bool)
    confirmed_low = np.zeros(n, dtype=bool)
    swing_high_price = np.full(n, np.nan)
    swing_low_price = np.full(n, np.nan)
    swing_high_index = np.full(n, np.nan)
    swing_low_index = np.full(n, np.nan)
    bearish_divergence = np.zeros(n, dtype=bool)
    bullish_divergence = np.zeros(n, dtype=bool)
    divergence_stop = np.full(n, np.nan)
    divergence_target = np.full(n, np.nan)

    highs = result["high"].to_numpy(dtype=float)
    lows = result["low"].to_numpy(dtype=float)
    rsi_values = result["rsi"].to_numpy(dtype=float)
    bullish_move_seen = result["bullish_half_pct_move_seen"].to_numpy(dtype=bool)
    bearish_move_seen = result["bearish_half_pct_move_seen"].to_numpy(dtype=bool)

    last_high: Optional[tuple[int, float, float]] = None
    last_low: Optional[tuple[int, float, float]] = None
    last_confirmed_low_price = np.nan
    last_confirmed_high_price = np.nan

    for confirm_idx in range(lookback * 2, n):
        center_idx = confirm_idx - lookback
        left_start = center_idx - lookback
        right_end = center_idx + lookback + 1
        if left_start < 0 or right_end > n:
            continue

        center_high = highs[center_idx]
        center_low = lows[center_idx]
        center_rsi = rsi_values[center_idx]
        high_window = highs[left_start:right_end]
        low_window = lows[left_start:right_end]

        is_swing_high = center_high == np.nanmax(high_window) and np.sum(high_window == center_high) == 1
        is_swing_low = center_low == np.nanmin(low_window) and np.sum(low_window == center_low) == 1

        if is_swing_low:
            confirmed_low[confirm_idx] = True
            swing_low_price[confirm_idx] = center_low
            swing_low_index[confirm_idx] = center_idx
            if last_low is not None and np.isfinite(center_rsi) and np.isfinite(last_low[2]):
                separation = center_idx - last_low[0]
                trend_ok = (not config.divergence_requires_trend_move) or bearish_move_seen[confirm_idx]
                if (
                    separation >= config.divergence_min_separation
                    and center_low < last_low[1]
                    and center_rsi > last_low[2]
                    and trend_ok
                    and np.isfinite(last_confirmed_high_price)
                ):
                    bullish_divergence[confirm_idx] = True
                    divergence_stop[confirm_idx] = center_low
                    divergence_target[confirm_idx] = last_confirmed_high_price
            if np.isfinite(center_rsi):
                last_low = (center_idx, center_low, center_rsi)
            last_confirmed_low_price = center_low

        if is_swing_high:
            confirmed_high[confirm_idx] = True
            swing_high_price[confirm_idx] = center_high
            swing_high_index[confirm_idx] = center_idx
            if last_high is not None and np.isfinite(center_rsi) and np.isfinite(last_high[2]):
                separation = center_idx - last_high[0]
                trend_ok = (not config.divergence_requires_trend_move) or bullish_move_seen[confirm_idx]
                if (
                    separation >= config.divergence_min_separation
                    and center_high > last_high[1]
                    and center_rsi < last_high[2]
                    and trend_ok
                    and np.isfinite(last_confirmed_low_price)
                ):
                    bearish_divergence[confirm_idx] = True
                    divergence_stop[confirm_idx] = center_high
                    divergence_target[confirm_idx] = last_confirmed_low_price
            if np.isfinite(center_rsi):
                last_high = (center_idx, center_high, center_rsi)
            last_confirmed_high_price = center_high

    result["confirmed_swing_high"] = confirmed_high
    result["confirmed_swing_low"] = confirmed_low
    result["confirmed_swing_high_price"] = swing_high_price
    result["confirmed_swing_low_price"] = swing_low_price
    result["confirmed_swing_high_index"] = swing_high_index
    result["confirmed_swing_low_index"] = swing_low_index
    result["last_swing_high"] = pd.Series(swing_high_price, index=result.index).ffill()
    result["last_swing_low"] = pd.Series(swing_low_price, index=result.index).ffill()
    result["last_swing_high_before_bar"] = result["last_swing_high"].shift(1)
    result["last_swing_low_before_bar"] = result["last_swing_low"].shift(1)
    result["broken_upper_swing"] = result["close"] > result["last_swing_high_before_bar"]
    result["broken_lower_swing"] = result["close"] < result["last_swing_low_before_bar"]
    result["bullish_swing_break_seen"] = result.groupby("session_date")["broken_upper_swing"].cummax()
    result["bearish_swing_break_seen"] = result.groupby("session_date")["broken_lower_swing"].cummax()
    result["bullish_rsi_divergence"] = bullish_divergence
    result["bearish_rsi_divergence"] = bearish_divergence
    result["rsi_divergence_stop"] = divergence_stop
    result["rsi_divergence_target"] = divergence_target
    return result


def _pick_first_above(row: pd.Series, columns: Sequence[str]) -> float:
    """Pick the first resistance level above the current close."""
    close = float(row["close"])
    for column in columns:
        value = row.get(column, np.nan)
        if np.isfinite(value) and float(value) > close:
            return float(value)
    return np.nan


def _pick_first_below(row: pd.Series, columns: Sequence[str]) -> float:
    """Pick the first support level below the current close."""
    close = float(row["close"])
    for column in columns:
        value = row.get(column, np.nan)
        if np.isfinite(value) and float(value) < close:
            return float(value)
    return np.nan


def _add_algo1_setups(frame: pd.DataFrame) -> pd.DataFrame:
    """Add PDF Algo 1 trend-entry setup columns."""
    result = frame.copy()
    active = result["cpr_width"].isin(["narrow", "medium"])

    rsi_bullish_45 = (result["rsi"] > result["rsi_ema20"]) & (result["rsi_ema20"] > 45.0)
    rsi_bullish_50 = (result["rsi"] > result["rsi_ema20"]) & (result["rsi_ema20"] > 50.0)
    rsi_bearish_60 = (result["rsi"] < result["rsi_ema20"]) & (result["rsi_ema20"] < 60.0)
    rsi_bearish_50 = (result["rsi"] < result["rsi_ema20"]) & (result["rsi_ema20"] < 50.0)

    crossed_above_neutral = (result["low"] <= result["cpr_upper"]) & (result["close"] > result["cpr_upper"])
    crossed_below_neutral = (result["high"] >= result["cpr_lower"]) & (result["close"] < result["cpr_lower"])

    result["algo1_c1_long_setup"] = (
        active
        & crossed_above_neutral
        & result["bullish_candle"]
        & result["minimal_upper_wick"]
        & (result["ema20"] > result["vwap"])
        & (result["ema5"] > result["ema20"])
        & rsi_bullish_45
    )
    result["algo1_c1_short_setup"] = (
        active
        & crossed_below_neutral
        & result["bearish_candle"]
        & result["minimal_lower_wick"]
        & (result["ema20"] < result["vwap"])
        & (result["ema5"] < result["ema20"])
        & rsi_bearish_60
    )

    retrace_to_ema20_long = (
        (result["low"] <= result["ema20"])
        & (result["high"] >= result["ema20"])
        & (result["close"] >= result["ema20"])
    )
    retrace_to_ema20_short = (
        (result["low"] <= result["ema20"])
        & (result["high"] >= result["ema20"])
        & (result["close"] <= result["ema20"])
    )
    long_context = result["bullish_half_pct_move_seen"] | result["bullish_swing_break_seen"]
    short_context = result["bearish_half_pct_move_seen"] | result["bearish_swing_break_seen"]

    result["algo1_c2_long_setup"] = (
        active
        & long_context
        & retrace_to_ema20_long
        & (result["ema20"] > result["vwap"])
        & (result["ema20"] > result["tc"])
        & (result["vwap"] > result["tc"])
        & (result["ema5"] > result["ema20"])
        & rsi_bullish_50
    )
    result["algo1_c2_short_setup"] = (
        active
        & short_context
        & retrace_to_ema20_short
        & (result["ema20"] < result["vwap"])
        & (result["ema20"] < result["bc"])
        & (result["vwap"] < result["bc"])
        & (result["ema5"] < result["ema20"])
        & rsi_bearish_50
    )

    result["algo1_c1_long_target"] = result.apply(
        lambda row: _pick_first_above(row, ["r2", "r3", "r4"]), axis=1
    )
    result["algo1_c1_short_target"] = result.apply(
        lambda row: _pick_first_below(row, ["s2", "s3", "s4"]), axis=1
    )
    result["algo1_c1_long_stop"] = result["cpr_lower"]
    result["algo1_c1_short_stop"] = result["cpr_upper"]
    result["algo1_c2_long_target"] = result[["last_swing_high", "session_high_so_far"]].max(axis=1)
    result["algo1_c2_short_target"] = result[["last_swing_low", "session_low_so_far"]].min(axis=1)
    result["algo1_c2_long_stop"] = result[["last_swing_low", "session_low_so_far"]].min(axis=1)
    result["algo1_c2_short_stop"] = result[["last_swing_high", "session_high_so_far"]].max(axis=1)
    return result


def _add_sideways_zones(frame: pd.DataFrame, config: CPRStrategyConfig) -> pd.DataFrame:
    """Add Algo 2 sideways support/resistance zone setup columns."""
    result = frame.copy()
    width = float(config.zone_width_points)
    lookback = int(config.zone_lookback)
    min_touches = int(config.zone_min_touches)

    resistance_low = np.full(len(result), np.nan)
    resistance_high = np.full(len(result), np.nan)
    support_low = np.full(len(result), np.nan)
    support_high = np.full(len(result), np.nan)
    resistance_touches = np.zeros(len(result), dtype=int)
    support_touches = np.zeros(len(result), dtype=int)

    highs = result["high"].to_numpy(dtype=float)
    lows = result["low"].to_numpy(dtype=float)

    for index in range(len(result)):
        start = max(0, index - lookback + 1)
        high_window = highs[start : index + 1]
        low_window = lows[start : index + 1]

        high_anchor = np.nanmax(high_window)
        low_anchor = np.nanmin(low_window)
        res_floor = np.floor(high_anchor / width) * width
        sup_floor = np.floor(low_anchor / width) * width

        resistance_low[index] = res_floor
        resistance_high[index] = res_floor + width
        support_low[index] = sup_floor
        support_high[index] = sup_floor + width
        resistance_touches[index] = int(
            np.sum((high_window >= resistance_low[index]) & (high_window <= resistance_high[index]))
        )
        support_touches[index] = int(
            np.sum((low_window >= support_low[index]) & (low_window <= support_high[index]))
        )

    result["resistance_zone_low"] = resistance_low
    result["resistance_zone_high"] = resistance_high
    result["support_zone_low"] = support_low
    result["support_zone_high"] = support_high
    result["resistance_touch_count"] = resistance_touches
    result["support_touch_count"] = support_touches

    active = result["cpr_width"].isin(["wide", "medium"])
    result["algo2_sideways_short_setup"] = (
        active
        & (result["resistance_touch_count"] >= min_touches)
        & (result["high"] >= result["resistance_zone_low"])
        & (result["high"] <= result["resistance_zone_high"])
        & result["bearish_candle"]
    )
    result["algo2_sideways_long_setup"] = (
        active
        & (result["support_touch_count"] >= min_touches)
        & (result["low"] >= result["support_zone_low"])
        & (result["low"] <= result["support_zone_high"])
        & result["bullish_candle"]
    )
    result["sideways_long_target"] = result["resistance_zone_low"]
    result["sideways_long_stop"] = result["support_zone_low"]
    result["sideways_short_target"] = result["support_zone_high"]
    result["sideways_short_stop"] = result["resistance_zone_high"]
    return result


def build_cpr_with_indicators(
    ohlc: pd.DataFrame,
    config: Optional[CPRStrategyConfig] = None,
) -> pd.DataFrame:
    """
    Normalize OHLC candles and add all CPR strategy indicator/setup columns.

    The returned DataFrame is safe to inspect directly in notebooks or CSVs.
    It never mutates the caller's original DataFrame.
    """
    config = config or CPRStrategyConfig()
    frame = prepare_cpr_ohlc_input(ohlc)
    _require_columns(frame, ["timestamp", "open", "high", "low", "close"])
    if frame.empty:
        return frame

    frame = _add_daily_cpr(frame)
    frame = _add_session_indicators(frame, config)
    frame = _add_candle_anatomy(frame, config)
    frame = _add_swings_and_divergence(frame, config)
    frame = _add_algo1_setups(frame)
    frame = _add_sideways_zones(frame, config)

    frame["rsi_divergence_long_setup"] = frame["bullish_rsi_divergence"]
    frame["rsi_divergence_short_setup"] = frame["bearish_rsi_divergence"]
    frame["rsi_divergence_long_target"] = frame["rsi_divergence_target"]
    frame["rsi_divergence_long_stop"] = frame["rsi_divergence_stop"]
    frame["rsi_divergence_short_target"] = frame["rsi_divergence_target"]
    frame["rsi_divergence_short_stop"] = frame["rsi_divergence_stop"]
    return frame


class CPRSignalEngine:
    """Evaluate the latest enriched CPR candle and return one decision."""

    def __init__(
        self,
        config: Optional[CPRStrategyConfig] = None,
        enabled_algorithms: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or CPRStrategyConfig()
        self.enabled_algorithms = tuple(enabled_algorithms or self.config.enabled_algorithms)

    @staticmethod
    def _hold(reason: str = "", debug: Optional[dict[str, Any]] = None) -> CPRDecision:
        return CPRDecision(action="HOLD", exit_reason=reason, debug=debug or {})

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def minimum_history_bars(self) -> int:
        """Return how many candles should exist before entries are considered."""
        indicator_need = max(
            self.config.ema_fast_period,
            self.config.ema_slow_period,
            self.config.rsi_period + self.config.rsi_ema_period,
            self.config.swing_lookback * 2 + 1,
        )
        return max(self.config.min_history_bars, indicator_need)

    def _evaluate_exit(self, current: pd.Series, position: CPRPositionContext) -> CPRDecision:
        """Use current candle range to decide whether an open trade hit stop/target."""
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if np.isfinite(stop) and low <= stop:
                return CPRDecision(
                    action="EXIT",
                    strategy_name=position.strategy_name,
                    exit_underlying=stop,
                    exit_reason="STOP",
                    debug={"close": close},
                )
            if np.isfinite(target) and high >= target:
                return CPRDecision(
                    action="EXIT",
                    strategy_name=position.strategy_name,
                    exit_underlying=target,
                    exit_reason="TARGET",
                    debug={"close": close},
                )

        if direction == "SHORT":
            if np.isfinite(stop) and high >= stop:
                return CPRDecision(
                    action="EXIT",
                    strategy_name=position.strategy_name,
                    exit_underlying=stop,
                    exit_reason="STOP",
                    debug={"close": close},
                )
            if np.isfinite(target) and low <= target:
                return CPRDecision(
                    action="EXIT",
                    strategy_name=position.strategy_name,
                    exit_underlying=target,
                    exit_reason="TARGET",
                    debug={"close": close},
                )

        return self._hold()

    @staticmethod
    def _valid_long(entry: float, stop: float, target: float) -> bool:
        return all(np.isfinite(v) for v in (entry, stop, target)) and stop < entry < target

    @staticmethod
    def _valid_short(entry: float, stop: float, target: float) -> bool:
        return all(np.isfinite(v) for v in (entry, stop, target)) and target < entry < stop

    def _candidate(
        self,
        row: pd.Series,
        condition: str,
        direction: str,
        strategy_name: str,
        stop_column: str,
        target_column: str,
        priority: int,
    ) -> Optional[dict[str, Any]]:
        """Return a normalized entry candidate if the setup is valid."""
        if not bool(row.get(condition, False)):
            return None

        entry = float(row["close"])
        stop = float(row.get(stop_column, np.nan))
        target = float(row.get(target_column, np.nan))
        if direction == "LONG" and not self._valid_long(entry, stop, target):
            return None
        if direction == "SHORT" and not self._valid_short(entry, stop, target):
            return None
        return {
            "direction": direction,
            "strategy_name": strategy_name,
            "entry": entry,
            "stop": stop,
            "target": target,
            "priority": priority,
        }

    def _entry_candidates(self, current: pd.Series) -> list[dict[str, Any]]:
        """Collect all same-candle entry candidates in priority order."""
        specs: list[tuple[str, str, str, str, str, int, str]] = []
        enabled = set(self.enabled_algorithms)

        if "ALGO1" in enabled:
            specs.extend(
                [
                    (
                        "algo1_c1_long_setup",
                        "LONG",
                        "ALGO1_CONDITION1",
                        "algo1_c1_long_stop",
                        "algo1_c1_long_target",
                        10,
                        "Algo 1 condition 1 bullish neutral-zone breakout",
                    ),
                    (
                        "algo1_c1_short_setup",
                        "SHORT",
                        "ALGO1_CONDITION1",
                        "algo1_c1_short_stop",
                        "algo1_c1_short_target",
                        10,
                        "Algo 1 condition 1 bearish neutral-zone breakdown",
                    ),
                    (
                        "algo1_c2_long_setup",
                        "LONG",
                        "ALGO1_CONDITION2",
                        "algo1_c2_long_stop",
                        "algo1_c2_long_target",
                        20,
                        "Algo 1 condition 2 bullish EMA20 retrace",
                    ),
                    (
                        "algo1_c2_short_setup",
                        "SHORT",
                        "ALGO1_CONDITION2",
                        "algo1_c2_short_stop",
                        "algo1_c2_short_target",
                        20,
                        "Algo 1 condition 2 bearish EMA20 retrace",
                    ),
                ]
            )

        if "ALGO2_ZONE" in enabled:
            specs.extend(
                [
                    (
                        "algo2_sideways_long_setup",
                        "LONG",
                        "ALGO2_SIDEWAYS_ZONE",
                        "sideways_long_stop",
                        "sideways_long_target",
                        30,
                        "Algo 2 support-zone long",
                    ),
                    (
                        "algo2_sideways_short_setup",
                        "SHORT",
                        "ALGO2_SIDEWAYS_ZONE",
                        "sideways_short_stop",
                        "sideways_short_target",
                        30,
                        "Algo 2 resistance-zone short",
                    ),
                ]
            )

        if "RSI_DIVERGENCE" in enabled:
            specs.extend(
                [
                    (
                        "rsi_divergence_long_setup",
                        "LONG",
                        "RSI_DIVERGENCE",
                        "rsi_divergence_long_stop",
                        "rsi_divergence_long_target",
                        40,
                        "Bullish RSI divergence reversal",
                    ),
                    (
                        "rsi_divergence_short_setup",
                        "SHORT",
                        "RSI_DIVERGENCE",
                        "rsi_divergence_short_stop",
                        "rsi_divergence_short_target",
                        40,
                        "Bearish RSI divergence reversal",
                    ),
                ]
            )

        candidates: list[dict[str, Any]] = []
        for condition, direction, name, stop_col, target_col, priority, label in specs:
            candidate = self._candidate(current, condition, direction, name, stop_col, target_col, priority)
            if candidate is not None:
                candidate["label"] = label
                candidates.append(candidate)
        return sorted(candidates, key=lambda item: int(item["priority"]))

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[CPRPositionContext] = None,
    ) -> CPRDecision:
        """Evaluate only the newest candle in an enriched CPR DataFrame."""
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        _require_columns(candles_with_indicators, ["timestamp", "open", "high", "low", "close"])

        current = candles_with_indicators.iloc[-1]
        if position is not None:
            return self._evaluate_exit(current, position)

        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        if str(current.get("cpr_width", "unknown")) == "unknown":
            return self._hold("Previous-day CPR is not available yet.")

        candidates = self._entry_candidates(current)
        if not candidates:
            return self._hold("No CPR setup on latest candle.")

        directions = {candidate["direction"] for candidate in candidates}
        if len(directions) > 1:
            return self._hold(
                "Conflict: both long and short CPR setups are valid on the same candle.",
                debug={"candidates": candidates},
            )

        selected = candidates[0]
        action = "ENTER_LONG" if selected["direction"] == "LONG" else "ENTER_SHORT"
        return CPRDecision(
            action=action,
            strategy_name=str(selected["strategy_name"]),
            entry_underlying=float(selected["entry"]),
            stop_underlying=float(selected["stop"]),
            target_underlying=float(selected["target"]),
            signal_triggered=True,
            debug={
                "label": selected["label"],
                "cpr_width": current.get("cpr_width", ""),
                "timestamp": current.get("timestamp"),
            },
        )


class CPRSignalGenerator:
    """
    Convenience wrapper that can generate a full signal history or latest signal.
    """

    def __init__(
        self,
        config: Optional[CPRStrategyConfig] = None,
        enabled_algorithms: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or CPRStrategyConfig()
        self.engine = CPRSignalEngine(self.config, enabled_algorithms=enabled_algorithms)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return enriched candles plus action/backtest stream columns."""
        frame = build_cpr_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        strategy_names: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: Optional[CPRPositionContext] = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            strategy_names.append(decision.strategy_name)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = CPRPositionContext(
                    direction="LONG",
                    entry_underlying=decision.entry_underlying,
                    stop_underlying=decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                    strategy_name=decision.strategy_name,
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = CPRPositionContext(
                    direction="SHORT",
                    entry_underlying=decision.entry_underlying,
                    stop_underlying=decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                    strategy_name=decision.strategy_name,
                )
            elif decision.action == "EXIT":
                stream.append(2)
                position = None
            else:
                stream.append(0)

        result = frame.copy()
        result["signalAction"] = actions
        result["strategyName"] = strategy_names
        result["entryUnderlying"] = entries
        result["stopUnderlying"] = stops
        result["targetUnderlying"] = targets
        result["backtestStream"] = stream
        return result

    def latest_signal(
        self,
        data: pd.DataFrame,
        position: Optional[CPRPositionContext] = None,
    ) -> CPRDecision:
        """Return only the newest CPR decision."""
        frame = build_cpr_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_cpr_signals(
    data: pd.DataFrame,
    config: Optional[CPRStrategyConfig] = None,
    enabled_algorithms: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Functional wrapper for full-history signal generation."""
    return CPRSignalGenerator(config=config, enabled_algorithms=enabled_algorithms).generate(data)


def get_latest_cpr_signal(
    data: pd.DataFrame,
    config: Optional[CPRStrategyConfig] = None,
    position: Optional[CPRPositionContext] = None,
    enabled_algorithms: Optional[Sequence[str]] = None,
) -> CPRDecision:
    """Functional wrapper for latest-candle signal generation."""
    return CPRSignalGenerator(config=config, enabled_algorithms=enabled_algorithms).latest_signal(
        data,
        position=position,
    )


__all__ = [
    "CPRStrategyConfig",
    "CPRPositionContext",
    "CPRDecision",
    "CPRSignalEngine",
    "CPRSignalGenerator",
    "build_cpr_with_indicators",
    "classify_daily_cpr_width",
    "prepare_cpr_ohlc_input",
    "generate_cpr_signals",
    "get_latest_cpr_signal",
]
