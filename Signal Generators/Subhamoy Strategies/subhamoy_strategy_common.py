"""
Shared helpers for Subhamoy strategy signal generators and backtests.

This file deliberately does NOT resample data. The front-test/data-fetch layer
is responsible for preparing 5-minute candles before calling these strategies.

Beginner mental model:
1. Normalize the caller's OHLC table into predictable lowercase columns.
2. Calculate standard indicators through the repository's pinned TA-Lib build.
3. Keep small reusable candle helpers here so Goldmine and Money Machine do not
   each carry their own copy of the same boilerplate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from typing import Any, cast

import numpy as np
import pandas as pd
import talib

OHLC_COLUMNS = ["open", "high", "low", "close"]


def validate_finite_config(config: Any) -> None:
    """Reject NaN/infinity in numeric strategy configuration fields."""

    invalid = [
        field.name
        for field in fields(config)
        if not isinstance((value := getattr(config, field.name)), bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and not np.isfinite(float(value))
    ]
    if invalid:
        raise ValueError(
            "Strategy configuration values must be finite. Invalid: "
            + ", ".join(invalid)
        )


def find_first_col(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    """Find the first matching column name using case-insensitive comparison."""
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def require_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise a clear beginner-friendly error when required columns are absent."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def normalize_ohlc_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return clean, time-sorted candles with lowercase OHLC column names.

    Accepted timestamp inputs:
    - a `timestamp`, `datetime`, `date`, or `time` column
    - a datetime-like DataFrame index

    This helper drops invalid rows but never changes candle timeframe.
    """
    if data is None or not isinstance(data, pd.DataFrame):
        raise ValueError("OHLC input must be a pandas DataFrame.")
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ts_col = find_first_col(data, ["timestamp", "datetime", "date", "time"])
    o_col = find_first_col(data, ["open"])
    h_col = find_first_col(data, ["high"])
    l_col = find_first_col(data, ["low"])
    c_col = find_first_col(data, ["close"])
    v_col = find_first_col(data, ["volume", "vol"])

    missing = []
    if o_col is None:
        missing.append("open")
    if h_col is None:
        missing.append("high")
    if l_col is None:
        missing.append("low")
    if c_col is None:
        missing.append("close")
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = pd.DataFrame(index=data.index)
    if ts_col is not None:
        result["timestamp"] = pd.to_datetime(data[ts_col], errors="coerce")
    else:
        result["timestamp"] = pd.to_datetime(pd.Series(data.index, index=data.index), errors="coerce")

    if result["timestamp"].isna().all():
        raise ValueError("A valid timestamp/datetime column or datetime-like index is required.")

    result["open"] = pd.to_numeric(data[o_col], errors="coerce")
    result["high"] = pd.to_numeric(data[h_col], errors="coerce")
    result["low"] = pd.to_numeric(data[l_col], errors="coerce")
    result["close"] = pd.to_numeric(data[c_col], errors="coerce")
    if v_col is not None:
        result["volume"] = pd.to_numeric(data[v_col], errors="coerce").fillna(0.0)
    else:
        result["volume"] = 0.0

    result = result.dropna(subset=["timestamp", "open", "high", "low", "close"])
    result = result.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return result


def validate_five_minute_spacing(frame: pd.DataFrame) -> None:
    """
    Confirm that data is already approximately 5-minute OHLC.

    A 1-minute file is rejected here on purpose. Resampling belongs in the
    user's front-test/data-preparation workflow, not inside these strategy files.
    """
    if frame is None or frame.empty or len(frame) < 2:
        return
    require_columns(frame, ["timestamp"])
    # pandas-stubs types .diff() on an untyped column as Series[float]; the
    # timestamp column holds datetimes, so tell mypy the gaps are Timedeltas
    # before using the .dt accessor (no runtime effect).
    gaps = cast("pd.Series[pd.Timedelta]", frame["timestamp"].diff().dropna())
    deltas = gaps.dt.total_seconds() / 60.0
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return
    median_minutes = float(positive_deltas.median())
    if not (4.5 <= median_minutes <= 5.5):
        raise ValueError(
            "Subhamoy strategy backtests expect already-prepared 5-minute OHLC data; "
            f"detected median spacing of {median_minutes:.2f} minutes. "
            "Please resample in the front-test/data-preparation file before running this backtest."
        )


def sma(values: pd.Series, period: int) -> pd.Series:
    """Calculate simple moving average with the pinned TA-Lib backend."""
    return pd.Series(
        talib.SMA(values.to_numpy(dtype="float64"), timeperiod=int(period)),
        index=values.index,
    )


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Calculate ATR with the pinned TA-Lib backend."""
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    return pd.Series(
        talib.ATR(
            np.asarray(high, dtype=np.float64),
            np.asarray(low, dtype=np.float64),
            np.asarray(close, dtype=np.float64),
            timeperiod=int(period),
        ),
        index=frame.index,
    )


def add_candle_anatomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Add body, wick, and range columns used by both candle-pattern strategies."""
    result = frame.copy()
    result["candle_range"] = (result["high"] - result["low"]).abs()
    result["body"] = (result["close"] - result["open"]).abs()
    result["upper_wick"] = result["high"] - result[["open", "close"]].max(axis=1)
    result["lower_wick"] = result[["open", "close"]].min(axis=1) - result["low"]
    return result


def rising_over_lookback(values: pd.Series, lookback: int) -> pd.Series:
    """True when the latest value is above the value `lookback` bars ago."""
    return values > values.shift(int(lookback))


def falling_over_lookback(values: pd.Series, lookback: int) -> pd.Series:
    """True when the latest value is below the value `lookback` bars ago."""
    return values < values.shift(int(lookback))


def finite(value: object) -> bool:
    """Return True only for real finite numbers."""
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
