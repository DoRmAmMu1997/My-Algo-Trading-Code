"""
================================================================================
Supertrend Signal Generator  (beginner-friendly, heavily commented)
================================================================================

WHAT THIS FILE DOES
-------------------
This file is an import-able module. You give it a DataFrame of candles
(with open/high/low/close columns) and it returns the same DataFrame plus
extra columns that describe Supertrend state and trade signals.

STRATEGY IN ONE LINE
--------------------
Go bullish (for example by selling a put option) when the Supertrend flips
from red to green on a closed candle, and exit when it flips back from
green to red on a closed candle.

QUICK START
-----------
    from importlib import import_module
    # If the module path works normally for you:
    #     from SupertrendStrategies.Supertrend_Signal_Generator import (
    #         generate_supertrend_signals, get_latest_supertrend_signal,
    #     )
    # Or load it directly with importlib.util if the filename contains spaces.

    signals = generate_supertrend_signals(ohlc_df)           # full history
    latest  = get_latest_supertrend_signal(ohlc_df)          # newest decision

WHAT IS SUPERTREND  (beginner refresher)
----------------------------------------
Supertrend is a trend-following indicator built on top of ATR (Average True
Range). It draws a single line under price when the trend is up (colored
GREEN by convention) and above price when the trend is down (colored RED).
When price closes on the other side of this line, the line "flips" to the
other color. Traders use these flips as simple trend-change signals.

DEFAULT PARAMETERS (as requested in the strategy spec)
------------------------------------------------------
- ATR length: 14
- Factor    : 3

WHY WE USE TA-LIB FOR ATR
-------------------------
TA-Lib is a widely-used C library with a Python wrapper called `talib`. Its
ATR implementation is fast and matches TradingView's standard ATR because
both use "Wilder's smoothing" (also known as RMA). Using `talib.ATR`:
  1. Keeps results aligned with popular charting platforms.
  2. Is faster than a pure-Python/pandas loop on large histories.

IMPORTANT: TA-Lib DOES NOT provide a built-in Supertrend function. So we
still have to compute the Supertrend bands and direction ourselves. This
file uses `talib.ATR` when available, and falls back to a pandas-based RMA
ATR if TA-Lib is not installed on the machine.
"""

from __future__ import annotations

# --- Standard library imports ------------------------------------------------
# `dataclass` lets us define small "struct-like" classes for settings and
# decisions without writing `__init__` boilerplate.
from dataclasses import dataclass
# `IntEnum` is a tidy way to attach human-readable names (LONG, SHORT, ...)
# to small integer flags. The underlying values are still plain integers, so
# they can be compared against numpy arrays cheaply.
from enum import IntEnum
from typing import Iterable, Optional

# --- Third-party imports -----------------------------------------------------
# numpy is used for fast array math (no Python for-loops over raw numbers).
import numpy as np
# pandas is the tabular-data library we accept as input and return as output.
import pandas as pd

# --- Optional TA-Lib import --------------------------------------------------
# We try to import TA-Lib. If the user's environment does not have it, we
# simply set `_TALIB_AVAILABLE = False` and compute ATR manually later.
# This "try/except ImportError" pattern keeps the module import-safe even
# when optional native libraries are missing.
try:
    import talib  # type: ignore
    _TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised when TA-Lib missing
    talib = None  # type: ignore
    _TALIB_AVAILABLE = False


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class Direction(IntEnum):
    """
    Direction labels used throughout the signal engine.

    Integer values are chosen on purpose:
    -  +1 for LONG (Supertrend is GREEN, price is above the line)
    -  -1 for SHORT (Supertrend is RED, price is below the line)
    -   0 for NEUTRAL (only used as an "undefined" placeholder)

    Integers are convenient because we can store the whole direction series
    as a plain numpy int array.
    """

    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass(slots=True)
class SupertrendSettings:
    """
    Tunable inputs for the Supertrend calculation.

    - `atr_length`: Number of candles used to compute ATR. A higher number
      smooths the indicator and reduces whipsaws, but also makes it slower
      to react to real trend changes.
    - `factor`: Multiplier applied to ATR when building the bands. A higher
      factor keeps the Supertrend farther from price, so the trend stays
      green (or red) longer before it flips. A smaller factor makes the
      indicator more sensitive.

    These defaults match the strategy spec:
        ATR length = 14, Factor = 3
    """

    atr_length: int = 14
    factor: float = 3.0


@dataclass(slots=True)
class SupertrendDecision:
    """
    Snapshot of the newest bar's decision.

    This is what `get_latest_supertrend_signal(...)` returns. It is designed
    to be easy to inspect inside a live trading loop where only the latest
    candle matters.

    Fields:
    - `timestamp`        : timestamp of the evaluated candle
    - `close`            : close price of that candle
    - `supertrend`       : current Supertrend line value
    - `direction`        : +1 if line is green, -1 if line is red
    - `actions`          : tuple of action strings such as ('ENTER_LONG',)
    - `start_long_trade` : True when this bar flipped from red to green
    - `end_long_trade`   : True when this bar flipped from green to red
    - `flipped_to_green` : True when today's direction changed red -> green
    - `flipped_to_red`   : True when today's direction changed green -> red
    """

    timestamp: object
    close: float
    supertrend: float
    direction: int
    actions: tuple[str, ...]
    start_long_trade: bool
    end_long_trade: bool
    flipped_to_green: bool
    flipped_to_red: bool


# =============================================================================
# INPUT PREPARATION HELPERS
# =============================================================================
def _find_first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """
    Return the first column in `frame` whose (case-insensitive, trimmed)
    name is in `names`, or None if nothing matches.

    Why this exists:
    - Different data sources call the same column different things.
      e.g. "Open", "OPEN", "open", "Date", "timestamp", "DateTime"...
    - We want the caller to pass raw broker / CSV data without forcing them
      to rename columns first.
    """
    col_map = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _prepare_ohlc_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize the input DataFrame.

    After this function runs, the returned frame is guaranteed to have:
    - columns named exactly: open, high, low, close
    - optional: timestamp column (parsed as datetime when possible)
    - no NaN values in the O/H/L/C columns
    - a fresh 0..N-1 index for easy row-based access later

    We make a copy (`data.copy()`) so that we never mutate the user's input.
    """
    if data is None or data.empty:
        raise ValueError("Input OHLC data is empty.")

    # Try to locate the OHLC columns even if they are written in any case
    # variation or include extra whitespace. Timestamp is optional.
    open_col = _find_first_col(data, ["open"])
    high_col = _find_first_col(data, ["high"])
    low_col = _find_first_col(data, ["low"])
    close_col = _find_first_col(data, ["close"])
    ts_col = _find_first_col(data, ["timestamp", "datetime", "date", "time"])

    # If any required column is missing, fail early with a clear message.
    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError("Input data must contain open/high/low/close columns.")

    frame = data.copy()
    out = pd.DataFrame(index=frame.index)
    if ts_col is not None:
        out["timestamp"] = frame[ts_col]
    # `pd.to_numeric` converts strings like "1234.5" into floats.
    # `errors="coerce"` turns un-parseable entries into NaN instead of raising.
    out["open"] = pd.to_numeric(frame[open_col], errors="coerce")
    out["high"] = pd.to_numeric(frame[high_col], errors="coerce")
    out["low"] = pd.to_numeric(frame[low_col], errors="coerce")
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce")

    # Any row where OHLC could not be parsed is discarded. The indicator
    # would otherwise corrupt from that row onward.
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    if "timestamp" in out.columns:
        # Try to parse to datetime. If parsing fails (e.g. the column is
        # already datetime64 or holds unparseable values), keep the column
        # as-is.
        #
        # We used to pass `errors="ignore"` here, but pandas deprecated that
        # option (FutureWarning on every call). The try/except below gives
        # the same behaviour without the warning.
        try:
            out["timestamp"] = pd.to_datetime(out["timestamp"])
        except (ValueError, TypeError):
            pass
    # Reset to a clean integer index so `.iloc[-1]`, `.iloc[-2]` etc. behave
    # predictably regardless of the original index.
    return out.reset_index(drop=True)


# =============================================================================
# ATR CALCULATION (TA-LIB PRIMARY, PANDAS FALLBACK)
# =============================================================================
def _rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's moving average, also known as RMA or SMMA.

    Many indicators (ATR, RSI, ADX) historically use Wilder's smoothing
    instead of a simple moving average. The math is a special form of
    exponential moving average where:
        alpha = 1 / period

    Beginner intuition:
    - RMA weights the most recent bar by 1/period (small weight).
    - Older bars keep most of the value.
    - This makes the output smoother and slower-reacting than a simple EMA
      of the same period, but still responsive.

    We use `Series.ewm(..., adjust=False)` to match TradingView / TA-Lib.
    """
    period = max(int(period), 1)
    return series.ewm(alpha=1.0 / float(period), adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range (TR) per bar.

    For each candle, TR = max of three things:
    1. High - Low                    (today's full range)
    2. |High - Previous Close|       (gap up from yesterday's close)
    3. |Low  - Previous Close|       (gap down from yesterday's close)

    By taking the max of those three, TR captures both intraday range AND
    overnight/weekend gaps. That is why ATR is used instead of just high-low.
    """
    prev_close = close.shift(1)
    # `pd.concat(..., axis=1)` places the three candidate values as columns,
    # then `.max(axis=1)` takes the row-wise maximum.
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr_manual(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Pandas-only ATR used as the fallback path when TA-Lib is not installed.

    We intentionally mirror TA-Lib's warm-up behavior so both code paths
    produce numerically identical output:
      1. The first `period` bars return NaN (not enough data for a full TR).
      2. The value at bar index `period` is the simple average of the first
         `period` True Range values (this is TA-Lib's seed).
      3. Every subsequent bar uses Wilder's recursive formula:
             ATR[i] = (ATR[i-1] * (period - 1) + TR[i]) / period
    """
    period = max(int(period), 1)
    tr = _true_range(high, low, close)
    tr_values = tr.to_numpy(dtype=float)
    n = len(tr_values)
    atr_values = np.full(n, np.nan, dtype=float)

    # Need at least `period` True Range values (bars 1..period, since TR at
    # bar 0 is undefined because it needs a previous close).
    if n <= period:
        return pd.Series(atr_values, index=tr.index)

    # Seed at bar index `period`: simple mean of TR[1..period] inclusive.
    # Any NaN in that window invalidates the seed, matching TA-Lib's behavior.
    seed_window = tr_values[1 : period + 1]
    if np.isnan(seed_window).any():
        return pd.Series(atr_values, index=tr.index)
    atr_values[period] = float(np.mean(seed_window))

    # Apply Wilder's recursive smoothing from bar `period + 1` onward.
    for i in range(period + 1, n):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr_values[i]) / period

    return pd.Series(atr_values, index=tr.index)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Compute ATR using TA-Lib when available, otherwise use the pandas version.

    Why TA-Lib first:
    - `talib.ATR` is implemented in C, so it is faster on large datasets.
    - It uses Wilder smoothing by definition, which matches what charting
      platforms like TradingView display.

    Why a fallback:
    - TA-Lib is a native library (not a pure-Python package) and can be
      fiddly to install on some machines. We do not want that to be a hard
      dependency of this signal generator.
    - Both paths produce numerically equivalent results because they both
      apply Wilder's smoothing on the same True Range series.
    """
    period = max(int(period), 1)

    if _TALIB_AVAILABLE:
        # TA-Lib works on plain numpy float arrays, not pandas Series, so we
        # convert explicitly. `dtype=float` ensures no integer rounding.
        high_arr = high.to_numpy(dtype=float)
        low_arr = low.to_numpy(dtype=float)
        close_arr = close.to_numpy(dtype=float)
        try:
            atr_values = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
            # Rewrap as a pandas Series with the same index as the inputs so
            # downstream code can keep treating it like a normal Series.
            return pd.Series(atr_values, index=high.index)
        except Exception:  # pragma: no cover - safety net for odd inputs
            # If TA-Lib raises for any reason (shape mismatch, all-NaN input,
            # etc.), we gracefully fall back to the manual path rather than
            # breaking the signal generator.
            pass

    return _atr_manual(high, low, close, period)


# =============================================================================
# SUPERTREND CORE CALCULATION
# =============================================================================
def _compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_length: int,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Supertrend line values and direction labels.

    ALGORITHM (matches the Pine Script reference used by TradingView)
    -----------------------------------------------------------------
    Step 1: Midpoint of the candle
        hl2 = (high + low) / 2

    Step 2: Raw bands using ATR
        upper_basic = hl2 + factor * atr
        lower_basic = hl2 - factor * atr

    Step 3: "Final" bands that only move in one direction until a reset
        lower := close[-1] > lower[-1]
                    ? max(lower_basic, lower[-1])   # ratchet up while in uptrend
                    : lower_basic
        upper := close[-1] < upper[-1]
                    ? min(upper_basic, upper[-1])   # ratchet down while in downtrend
                    : upper_basic

    Step 4: Direction (trend) flips when close crosses the opposite prior band
        direction[-1] == SHORT AND close > upper[-1]  ->  flip to LONG
        direction[-1] == LONG  AND close < lower[-1]  ->  flip to SHORT
        else keep previous direction

    Step 5: Supertrend line value shown on the chart
        supertrend = lower if direction == LONG else upper

    WHY THIS "LADDER" BEHAVIOR
    --------------------------
    The raw upper/lower bands can wiggle with every candle. If we drew them
    directly, they would produce constant whipsaws. The "final" bands only
    move in the direction of the trend (lower band ratchets up during an
    uptrend, upper band ratchets down during a downtrend). This way the
    indicator only flips when price genuinely breaks through the opposite
    band, not when ATR changes by a tiny amount.

    Returns
    -------
    supertrend : np.ndarray[float]
        Supertrend line value for every bar.
    direction  : np.ndarray[int]
        +1 when the trend is bullish (line is green), -1 when bearish (red).
    """
    n = len(close)
    supertrend = np.zeros(n, dtype=float)
    direction = np.zeros(n, dtype=int)
    if n == 0:
        # Empty input -> return empty arrays so the caller can handle it.
        return supertrend, direction

    # Step 1: midpoints.
    hl2 = (high.to_numpy(dtype=float) + low.to_numpy(dtype=float)) / 2.0

    # Step 2: ATR (uses TA-Lib when available).
    atr_values = _atr(high, low, close, atr_length).to_numpy(dtype=float)
    close_values = close.to_numpy(dtype=float)

    # Raw bands. These can shift up and down every candle.
    upper_basic = hl2 + float(factor) * atr_values
    lower_basic = hl2 - float(factor) * atr_values

    # Final bands. These ratchet in one direction until a trend flip.
    final_upper = np.zeros(n, dtype=float)
    final_lower = np.zeros(n, dtype=float)

    # We walk bar-by-bar because final bands depend on their previous value.
    # This is one of the few cases where a Python for-loop is unavoidable
    # (unless we reimplement in Cython / numba).
    for i in range(n):
        if i == 0 or not np.isfinite(atr_values[i]):
            # Warm-up bars: ATR has no value yet (returns NaN for the first
            # `atr_length` bars). We seed the final bands with the raw bands
            # and start in an arbitrary LONG state. The direction will correct
            # itself automatically once real ATR values start coming in.
            final_upper[i] = upper_basic[i] if np.isfinite(upper_basic[i]) else 0.0
            final_lower[i] = lower_basic[i] if np.isfinite(lower_basic[i]) else 0.0
            direction[i] = int(Direction.LONG)
            supertrend[i] = final_lower[i]
            continue

        prev_close = close_values[i - 1]

        # Final lower band:
        #   If the previous close was above the previous final_lower (trend
        #   still up), we ratchet the band up by taking the max of today's
        #   raw lower band and the previous final lower. Otherwise we reset
        #   to today's raw lower band.
        final_lower[i] = (
            max(lower_basic[i], final_lower[i - 1])
            if prev_close > final_lower[i - 1]
            else lower_basic[i]
        )
        # Final upper band: mirror logic for a downtrend.
        final_upper[i] = (
            min(upper_basic[i], final_upper[i - 1])
            if prev_close < final_upper[i - 1]
            else upper_basic[i]
        )

        # Direction flip rules, using the PREVIOUS bar's final band as the
        # break level (this matches TradingView's Supertrend exactly).
        prev_direction = direction[i - 1]
        if prev_direction == int(Direction.SHORT) and close_values[i] > final_upper[i - 1]:
            direction[i] = int(Direction.LONG)
        elif prev_direction == int(Direction.LONG) and close_values[i] < final_lower[i - 1]:
            direction[i] = int(Direction.SHORT)
        else:
            # No break => carry the direction forward.
            direction[i] = prev_direction

        # The Supertrend line shown on the chart is whichever band is
        # "active" for the current direction: lower when bullish (line is
        # below price, colored green), upper when bearish (line is above
        # price, colored red).
        supertrend[i] = final_lower[i] if direction[i] == int(Direction.LONG) else final_upper[i]

    return supertrend, direction


# =============================================================================
# PUBLIC SIGNAL GENERATOR CLASS
# =============================================================================
class SupertrendSignalGenerator:
    """
    Reusable Supertrend signal engine.

    TRADING RULES IMPLEMENTED (from the strategy spec)
    --------------------------------------------------
    - Enter a bullish position (for example by selling a put) when a candle
      CLOSES and the Supertrend changes from RED to GREEN on that candle.
    - Exit the bullish position when a candle CLOSES and the Supertrend
      changes from GREEN to RED.

    Important: the signals are evaluated on CLOSED candles only. That means
    the entry/exit flags for a given bar are known *at or after* that bar's
    close, never mid-candle. This prevents repainting and aligns with how
    the same rules would be traded live.

    WHY "SELLING A PUT" IS A BULLISH ACTION
    ---------------------------------------
    A short put (selling a put) profits when the underlying stays flat or
    rises. So when the Supertrend flips bullish and we "sell a put", that
    is equivalent to expressing a long / bullish view on the underlying.
    """

    def __init__(self, settings: Optional[SupertrendSettings] = None) -> None:
        # If the caller does not pass settings, use the class's default
        # configuration (ATR 14, Factor 3) from `SupertrendSettings`.
        self.settings = settings or SupertrendSettings()

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Supertrend + signals for the full history.

        Returns a new DataFrame (does not mutate the input) with these
        extra columns on top of the original OHLC columns:

        - atr                : ATR used by this run of the indicator
        - supertrend         : Supertrend line value
        - direction          : +1 (green/long) or -1 (red/short)
        - color              : 'green' or 'red' for human readability
        - flippedToGreen     : True on the candle where direction went -1 -> +1
        - flippedToRed       : True on the candle where direction went +1 -> -1
        - startLongTrade     : same as flippedToGreen (alias for clarity)
        - endLongTrade       : same as flippedToRed   (alias for clarity)
        - startLongTradePrice: close price on entry candles, NaN otherwise
        - endLongTradePrice  : close price on exit candles, NaN otherwise
        - backtestStream     : integer stream: 1 = entry, 2 = exit, 0 = hold
        """
        # Step 1: clean and normalize the input.
        frame = _prepare_ohlc_frame(data)
        if frame.empty:
            # Nothing to analyze. We still return a DataFrame (not None) so
            # downstream code can safely call .empty / len() on the result.
            return frame

        # Step 2: pull config values out of settings and sanity-check them.
        atr_length = int(self.settings.atr_length)
        factor = float(self.settings.factor)
        if atr_length < 1:
            raise ValueError("atr_length must be at least 1.")
        if factor <= 0:
            raise ValueError("factor must be positive.")

        # Step 3: run the core Supertrend math.
        supertrend, direction = _compute_supertrend(
            frame["high"],
            frame["low"],
            frame["close"],
            atr_length,
            factor,
        )

        # Step 4: detect flips between the previous bar and the current bar.
        #
        # We build `prev_direction` by prepending a NEUTRAL value and then
        # dropping the last element. This creates a "shift by 1" of the
        # direction array without needing pandas for it. The very first bar
        # therefore has NEUTRAL as its "previous", which means it cannot
        # ever be counted as a flip -- exactly what we want for warm-up.
        prev_direction = np.concatenate(([int(Direction.NEUTRAL)], direction[:-1]))

        # A flip is a strict change between the previous and current direction.
        flipped_to_green = (prev_direction == int(Direction.SHORT)) & (
            direction == int(Direction.LONG)
        )
        flipped_to_red = (prev_direction == int(Direction.LONG)) & (
            direction == int(Direction.SHORT)
        )

        # The strategy trades bullish only (sells puts on green flip), so:
        # - An "entry" is when Supertrend just turned green.
        # - An "exit" is when Supertrend just turned red.
        start_long_trade = flipped_to_green
        end_long_trade = flipped_to_red

        # A compact integer "event stream" that backtest engines can read:
        #   1 -> enter bullish
        #   2 -> exit bullish
        #   0 -> do nothing / hold
        # `np.select` chooses the first matching condition per row.
        backtest_stream = np.select(
            [start_long_trade, end_long_trade],
            [1, 2],
            default=0,
        )

        # Step 5: attach all computed columns to the result frame.
        result = frame.copy()
        result["atr"] = _atr(frame["high"], frame["low"], frame["close"], atr_length).to_numpy(
            dtype=float
        )
        result["supertrend"] = supertrend
        result["direction"] = direction
        # Store a string color for easy eyeballing when inspecting output.
        result["color"] = np.where(direction == int(Direction.LONG), "green", "red")
        result["flippedToGreen"] = flipped_to_green
        result["flippedToRed"] = flipped_to_red
        result["startLongTrade"] = start_long_trade
        result["endLongTrade"] = end_long_trade
        # `np.where(mask, value_if_true, value_if_false)` fills a column
        # with close prices on event bars and NaN elsewhere. This matches
        # the "start trade price" / "end trade price" pattern used in the
        # other signal generators in this project.
        result["startLongTradePrice"] = np.where(start_long_trade, result["close"], np.nan)
        result["endLongTradePrice"] = np.where(end_long_trade, result["close"], np.nan)
        result["backtestStream"] = backtest_stream
        return result

    def latest_signal(self, data: pd.DataFrame) -> SupertrendDecision:
        """
        Return a `SupertrendDecision` for the newest bar only.

        Use this in live trading / front-testing loops where you only care
        about what to do right now, not the full signal history.
        """
        generated = self.generate(data)
        if generated.empty:
            raise ValueError("No signal could be generated from the provided data.")

        # `.iloc[-1]` is the last row (newest bar) of the signal frame.
        last_row = generated.iloc[-1]

        # Translate boolean flags into a tuple of action strings.
        # We collect them in a list first, then freeze to a tuple so callers
        # cannot accidentally mutate the returned decision object.
        actions: list[str] = []
        if bool(last_row["startLongTrade"]):
            actions.append("ENTER_LONG")
        if bool(last_row["endLongTrade"]):
            actions.append("EXIT_LONG")
        if not actions:
            actions.append("HOLD")

        # If the input had a timestamp column, use it; otherwise fall back
        # to the DataFrame's own index value for this row.
        timestamp = (
            last_row["timestamp"]
            if "timestamp" in generated.columns
            else generated.index[-1]
        )
        return SupertrendDecision(
            timestamp=timestamp,
            close=float(last_row["close"]),
            supertrend=float(last_row["supertrend"]),
            direction=int(last_row["direction"]),
            actions=tuple(actions),
            start_long_trade=bool(last_row["startLongTrade"]),
            end_long_trade=bool(last_row["endLongTrade"]),
            flipped_to_green=bool(last_row["flippedToGreen"]),
            flipped_to_red=bool(last_row["flippedToRed"]),
        )


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================
# These tiny wrappers exist so callers who don't want to instantiate the
# class can just import a single function and go. They use default settings
# unless the caller overrides them.
def generate_supertrend_signals(
    data: pd.DataFrame,
    settings: Optional[SupertrendSettings] = None,
) -> pd.DataFrame:
    """Functional alias for `SupertrendSignalGenerator(...).generate(data)`."""
    generator = SupertrendSignalGenerator(settings=settings)
    return generator.generate(data)


def get_latest_supertrend_signal(
    data: pd.DataFrame,
    settings: Optional[SupertrendSettings] = None,
) -> SupertrendDecision:
    """Functional alias for `SupertrendSignalGenerator(...).latest_signal(data)`."""
    generator = SupertrendSignalGenerator(settings=settings)
    return generator.latest_signal(data)


# `__all__` controls what `from module import *` exposes. Listing public
# names here also serves as a quick table-of-contents for readers.
__all__ = [
    "Direction",
    "SupertrendSettings",
    "SupertrendDecision",
    "SupertrendSignalGenerator",
    "generate_supertrend_signals",
    "get_latest_supertrend_signal",
]
