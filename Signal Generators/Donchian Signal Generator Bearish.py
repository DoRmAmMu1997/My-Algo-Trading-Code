"""
================================================================================
Donchian Bearish Signal Generator  (beginner-friendly, heavily commented)
================================================================================

WHAT THIS FILE DOES
-------------------
This file is an import-able module. You give it a DataFrame of candles
(with open/high/low/close columns) and it returns the same DataFrame plus
extra columns that describe the Donchian Channel state and a *bearish*
entry signal.

It is the bearish counterpart of `Supertrend Signal Generator Bullish.py`
in the same folder. Both files share the same I/O contract:

    * Input  : a pandas DataFrame with O/H/L/C columns (any case).
    * Output : the same DataFrame with extra computed columns appended.

STRATEGY IN ONE LINE
--------------------
Go bearish (for example by SELLING A CALL OPTION) when the most recent
two CLOSED candles each closed higher than the one before -- a small rally
-- WHILE the upper Donchian Channel was flat-or-falling just before that
rally. Intuition: a tiny rally is running into a stalling upper boundary,
so the rally is more likely to fail than to break out.

ENTRY RULES (exactly as supplied by the strategy spec)
------------------------------------------------------
Live framing: the very latest candle is *forming* (still incomplete). We
IGNORE that one and only judge the CLOSED candles. Counting back from
"the last" bar:

    last        ->  the live, forming bar (IGNORED)
    second-last ->  the most recent CLOSED bar
    third-last  ->  the bar just before that (also CLOSED)
    fourth-last ->  the bar just before that (also CLOSED)
    fifth-last  ->  the bar just before that (also CLOSED)

Conditions, ALL must be true on the same evaluation:

    1)  close[second-last]   >  close[third-last]
    2)  close[third-last]    >  close[fourth-last]
    3)  upper_donchian[fourth-last]  <=  upper_donchian[fifth-last]
        (the rolling-high line slopes DOWN or stays FLAT between bars 5
         and 4 from the end -- it must NOT be rising)

If all three hold, we mark `startShortTrade = True` on the second-last bar.

EXIT RULE
---------
NOT YET DEFINED. The exit logic for the bearish strategy will be supplied
later. For now this module emits ONLY entry signals (`startShortTrade`)
and keeps `endShortTrade = False` everywhere so callers can still rely on
the column being present (matching the schema of the bullish file).

WHERE THE SIGNAL IS PLACED IN THE OUTPUT (important convention)
---------------------------------------------------------------
This file follows the SAME convention as the bullish file: the entry
signal is placed at the bar where the rule's TRIGGER condition is first
true. Concretely, in the output DataFrame:

    startShortTrade[i] = True
        if and only if ALL of these hold:
            close[i]    >  close[i-1]            # rule (1)
            close[i-1]  >  close[i-2]            # rule (2)
            upper[i-2]  <= upper[i-3]            # rule (3)

Why? Because at decision time, "bar i" plays the role of the spec's
"second-last" bar (the most-recent-closed one). The spec's "ignored last"
bar is therefore "bar i+1", which is the bar where you would actually
place a live order after observing bar i's close.

QUICK START
-----------
    # If your project supports normal package paths:
    #     from SupertrendStrategies.DonchianSignalGeneratorBearish import (
    #         generate_donchian_bearish_signals,
    #         get_latest_donchian_bearish_signal,
    #     )
    # Or load it directly with importlib.util when the filename has spaces.

    signals = generate_donchian_bearish_signals(ohlc_df)        # full history
    latest  = get_latest_donchian_bearish_signal(ohlc_df)       # newest decision

WHAT IS A DONCHIAN CHANNEL  (beginner refresher)
------------------------------------------------
A Donchian Channel is one of the simplest indicators that exists. With a
length of N bars, it draws three lines:

    upper  = highest high  of the last N bars
    middle = (upper + lower) / 2
    lower  = lowest  low   of the last N bars

That is literally all there is to it. There is no smoothing, no exponential
math, no ATR -- just rolling max and rolling min. A FLAT or DECLINING upper
line therefore tells you "the rolling high stopped making new highs in the
last few bars," which is a small but useful "supply zone is intact" hint.

DEFAULT PARAMETERS (as requested in the strategy spec)
------------------------------------------------------
- Donchian length: 20

WHERE DOES THE INDICATOR LIVE?  (library vs fallback)
-----------------------------------------------------
The user requested that we check whether the indicator is available in any
standard library, prefer the library if present, but ALWAYS code our own
fallback. Here is the lay of the land for Donchian:

    * `pandas_ta.donchian(high, low, lower_length, upper_length)`
        - YES, ships it directly. We try this FIRST because the column
          labels and warm-up handling match TradingView the closest.

    * TA-Lib (the `talib` package)
        - DOES NOT ship a `Donchian` function, BUT it ships `talib.MAX`
          and `talib.MIN`, which are exactly the building blocks
          (`upper = MAX(high, length)` and `lower = MIN(low, length)`).
          We use those as our SECOND-CHOICE path because the rolling
          max/min in TA-Lib is implemented in C and is faster than the
          pandas equivalent on long histories.

    * Pure pandas
        - `Series.rolling(window=length).max()` / `.min()` is the universal
          fallback. We keep this as the LAST-RESORT path so the module
          always works, even on a machine with neither library installed.

All three paths produce numerically identical results (within floating-
point rounding) because they compute the same rolling max/min/average.
"""

from __future__ import annotations

# --- Standard library imports ------------------------------------------------
# `dataclass` lets us define small "struct-like" classes without writing the
# `__init__` boilerplate ourselves.
from dataclasses import dataclass
# `IntEnum` attaches human-readable names (LONG, SHORT, ...) to small integer
# flags. The underlying values stay plain integers, so they can be compared
# against numpy arrays cheaply.
from enum import IntEnum
from typing import Iterable, Optional

# --- Third-party imports -----------------------------------------------------
# numpy is used for fast array math (no Python for-loops over raw numbers).
import numpy as np
# pandas is the tabular-data library we accept as input and return as output.
import pandas as pd

# --- Optional TA-Lib import --------------------------------------------------
# Same try/except pattern used in the bullish file. TA-Lib is a native (C)
# library and may not be installed everywhere. For Donchian we only need
# `talib.MAX` and `talib.MIN`. TA-Lib's import is fast so we do it eagerly.
try:
    import talib  # type: ignore
    _TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised when TA-Lib missing
    talib = None  # type: ignore
    _TALIB_AVAILABLE = False

# --- Optional pandas_ta import (LAZY) ----------------------------------------
# `pandas_ta` ships a direct `donchian` function with TradingView-compatible
# labels, but its module-level import can be VERY slow on some machines
# (tens of seconds of CPU-bound init for numba caches etc.). Because the
# front-test runner imports this file at module load via `importlib`, a
# slow pandas_ta would block the whole program's startup and the user sees
# nothing in the console until it finally finishes.
#
# We therefore defer the import. `pta` starts as None, and the first time
# we actually need the pandas_ta backend (which in the current priority
# order is only after TA-Lib has failed) we try to import it exactly once
# and cache the outcome. On an ordinary session with TA-Lib installed,
# pandas_ta is NEVER imported at all.
pta = None  # populated lazily on first use, never at module load
_PANDAS_TA_AVAILABLE: Optional[bool] = None  # None = "not probed yet"


def _ensure_pandas_ta() -> bool:
    """
    Lazily import pandas_ta on first need and cache the outcome.

    Returns True if the import succeeded, False otherwise. Subsequent calls
    return the cached result without re-attempting the import.

    Why this exists: see the comment block above. Keeping the import out of
    module load time means the front-test runner's startup is fast and any
    pandas_ta weirdness only surfaces if a caller really falls through to
    the pandas_ta backend.
    """
    global pta, _PANDAS_TA_AVAILABLE
    if _PANDAS_TA_AVAILABLE is not None:
        return bool(_PANDAS_TA_AVAILABLE)
    try:
        import pandas_ta as _pta  # type: ignore
    except Exception:  # noqa: BLE001 - some pandas_ta versions raise non-ImportError
        pta = None  # type: ignore
        _PANDAS_TA_AVAILABLE = False
        return False
    pta = _pta  # type: ignore
    _PANDAS_TA_AVAILABLE = True
    return True


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class Direction(IntEnum):
    """
    Direction labels used throughout the signal engine.

    Mirror of the bullish file's `Direction` enum so that callers who keep a
    portfolio mixing bullish and bearish signal generators can use a single
    integer code-set for both.

    -  +1 for LONG (bullish)
    -  -1 for SHORT (bearish)
    -   0 for NEUTRAL ("undefined" placeholder)
    """

    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass(slots=True)
class DonchianSettings:
    """
    Tunable inputs for the Donchian Channel calculation.

    - `length`: Number of candles used to compute the rolling highest-high
      and lowest-low. The default (20) matches the strategy spec and is
      also TradingView's default for the "Donchian Channels" study.

    Note: we use a single `length` here for both the upper and the lower
    line, because that is what the strategy spec asks for. `pandas_ta`
    accepts independent `lower_length` and `upper_length`, but we always
    pass the same value to both for a "balanced" Donchian.
    """

    length: int = 20


@dataclass(slots=True)
class BearishDecision:
    """
    Snapshot of the newest bar's bearish decision.

    This is what `get_latest_donchian_bearish_signal(...)` returns. It is
    designed to be easy to inspect inside a live trading loop where only
    the latest candle matters.

    Fields:
    - `timestamp`            : timestamp of the evaluated candle
    - `close`                : close price of that candle
    - `upper_donchian`       : current upper Donchian line value
    - `middle_donchian`      : current middle Donchian line value
    - `lower_donchian`       : current lower Donchian line value
    - `actions`              : tuple of action strings such as ('ENTER_SHORT',)
    - `start_short_trade`    : True when this bar's rules fired for entry
    - `end_short_trade`      : reserved for the future exit rule (always False
                               for now)
    - `rule_close_pair_a`    : True if close[i]   > close[i-1]
    - `rule_close_pair_b`    : True if close[i-1] > close[i-2]
    - `rule_donchian_slope`  : True if upper[i-2] <= upper[i-3]
    """

    timestamp: object
    close: float
    upper_donchian: float
    middle_donchian: float
    lower_donchian: float
    actions: tuple[str, ...]
    start_short_trade: bool
    end_short_trade: bool
    rule_close_pair_a: bool
    rule_close_pair_b: bool
    rule_donchian_slope: bool


# =============================================================================
# INPUT PREPARATION HELPERS  (mirror of the bullish file's helpers)
# =============================================================================
# These two helpers are intentionally COPIED (not imported) from the bullish
# file. Keeping each signal generator self-contained means a caller can use
# `importlib.util` to load either file in isolation -- without having to
# resolve cross-file imports for the helper functions. The duplication is a
# few dozen lines and pays for itself in decoupling.

def _find_first_col(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """
    Return the first column in `frame` whose (case-insensitive, trimmed)
    name is in `names`, or None if nothing matches.

    Why this exists:
    - Different data sources spell the same column differently (e.g. "Open",
      "OPEN", "open", "Date", "timestamp", "DateTime"...).
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
    - a fresh 0..N-1 index for safe `.iloc[-k]` access later

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
    # `pd.to_numeric` converts strings like "1234.5" into floats. The
    # `errors="coerce"` flag turns unparseable entries into NaN instead of
    # raising, which we then drop below.
    out["open"] = pd.to_numeric(frame[open_col], errors="coerce")
    out["high"] = pd.to_numeric(frame[high_col], errors="coerce")
    out["low"] = pd.to_numeric(frame[low_col], errors="coerce")
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce")

    # Any row where OHLC could not be parsed is discarded. The Donchian
    # rolling max/min would otherwise propagate that NaN through the window.
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
    # Reset to a clean integer index so `.iloc[-1]`, `.iloc[-2]`, etc. behave
    # predictably regardless of the original index.
    return out.reset_index(drop=True)


# =============================================================================
# DONCHIAN CHANNEL CALCULATION
# (PANDAS_TA PRIMARY  ->  TA-LIB SECONDARY  ->  PURE-PANDAS FALLBACK)
# =============================================================================
def _donchian_pure_pandas(
    high: pd.Series,
    low: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Universal pandas-only Donchian (the always-works fallback).

    For each bar `i`:
        upper[i]  = max( high[i-length+1 .. i] )
        lower[i]  = min( low [i-length+1 .. i] )
        middle[i] = (upper[i] + lower[i]) / 2

    Notes:
    - `min_periods=length` makes sure the first `length-1` rows return NaN,
      matching the warm-up behavior of TA-Lib and pandas_ta. Without this,
      pandas would still emit a partial value with fewer-than-N bars.
    - The middle line is just the arithmetic mean of upper and lower, which
      is the standard Donchian midline (also known as the "Donchian basis").
    """
    length = max(int(length), 1)
    upper = high.rolling(window=length, min_periods=length).max()
    lower = low.rolling(window=length, min_periods=length).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def _donchian_via_talib(
    high: pd.Series,
    low: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian via TA-Lib's `MAX` and `MIN` (the SECOND-choice path).

    TA-Lib does not ship `Donchian` as a single call, but its `MAX(arr, n)`
    returns the rolling maximum of the last `n` bars (and `MIN` does the
    mirror). We just glue them together. The output dtype is float to match
    pandas_ta's output.

    Why this is preferable over pure pandas when TA-Lib is installed:
    - TA-Lib's rolling max/min is implemented in C, so it is significantly
      faster than the pandas-Python equivalent on long histories.
    """
    length = max(int(length), 1)
    high_arr = high.to_numpy(dtype=float)
    low_arr = low.to_numpy(dtype=float)
    upper_values = talib.MAX(high_arr, timeperiod=length)
    lower_values = talib.MIN(low_arr, timeperiod=length)
    middle_values = (upper_values + lower_values) / 2.0
    upper = pd.Series(upper_values, index=high.index, dtype=float)
    lower = pd.Series(lower_values, index=low.index, dtype=float)
    middle = pd.Series(middle_values, index=high.index, dtype=float)
    return upper, middle, lower


def _donchian_via_pandas_ta(
    high: pd.Series,
    low: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian via `pandas_ta.donchian` (the FIRST-choice path).

    `pandas_ta` returns a DataFrame with columns named like
    `DCL_<lower>_<upper>`, `DCM_<lower>_<upper>`, `DCU_<lower>_<upper>`.

    We pick those three columns BY PREFIX (`DCU`, `DCM`, `DCL`) instead of
    by exact name, so that small label changes between pandas_ta versions
    do not break us.
    """
    length = max(int(length), 1)
    df = pta.donchian(high=high, low=low, lower_length=length, upper_length=length)
    if df is None or df.empty:
        # If pandas_ta declines to compute (e.g. too few rows), bubble up
        # so the caller can fall back to TA-Lib or pure pandas.
        raise RuntimeError("pandas_ta.donchian returned an empty result.")

    upper_col = next((c for c in df.columns if str(c).startswith("DCU")), None)
    middle_col = next((c for c in df.columns if str(c).startswith("DCM")), None)
    lower_col = next((c for c in df.columns if str(c).startswith("DCL")), None)
    if not all([upper_col, middle_col, lower_col]):
        raise RuntimeError(
            f"pandas_ta.donchian returned unexpected columns: {list(df.columns)}"
        )

    upper = df[upper_col].astype(float)
    middle = df[middle_col].astype(float)
    lower = df[lower_col].astype(float)
    return upper, middle, lower


def _compute_donchian(
    high: pd.Series,
    low: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Top-level Donchian dispatcher.

    Tries each available implementation in priority order:
        1. TA-Lib      (fast, pure-C rolling max/min - our preferred path
                        because its import is quick and the result is
                        numerically identical to pandas_ta's)
        2. pandas_ta   (lazily imported on first need; skipped entirely if
                        TA-Lib handled the request)
        3. Pure pandas (always works, slowest)

    Each path is wrapped in its own try/except so a single library hiccup
    (e.g., a buggy pandas_ta version on a particular pandas) does not bring
    the whole module down.

    Why TA-Lib is first:
    Previously pandas_ta sat at the top of the chain, which also meant its
    slow module-level init ran at import time. On some environments that
    init takes 10+ seconds, which made the multi-threaded front-test
    runner look hung during startup. Putting TA-Lib first, plus making
    pandas_ta a lazy import, means module load is instant whenever TA-Lib
    is installed (our usual case).
    """
    length = max(int(length), 1)

    if _TALIB_AVAILABLE:
        try:
            return _donchian_via_talib(high, low, length)
        except Exception:  # noqa: BLE001 - log-and-fall-through
            pass

    # pandas_ta backend is attempted only if TA-Lib is missing or failed.
    # `_ensure_pandas_ta` imports pandas_ta on first call and caches the
    # outcome so later calls are cheap regardless of how slow the initial
    # import was.
    if _ensure_pandas_ta():
        try:
            return _donchian_via_pandas_ta(high, low, length)
        except Exception:  # noqa: BLE001 - log-and-fall-through
            pass

    return _donchian_pure_pandas(high, low, length)


# =============================================================================
# PUBLIC SIGNAL GENERATOR CLASS
# =============================================================================
class DonchianBearishSignalGenerator:
    """
    Reusable Donchian-bearish signal engine.

    TRADING RULES IMPLEMENTED (from the strategy spec)
    --------------------------------------------------
    See the module docstring for the full prose. Compactly:

        startShortTrade[i] = True
            when ALL three of these hold at bar i:
                close[i]    >  close[i-1]
                close[i-1]  >  close[i-2]
                upper[i-2]  <= upper[i-3]

    Important: the signal is evaluated on CLOSED candles only -- exactly
    like the bullish file. In live trading, when bar `i` closes True, the
    trader takes the bearish position on bar `i+1` (the bar that the spec
    calls the "ignored last" bar at decision time).

    WHY "SELLING A CALL OPTION" IS A BEARISH ACTION
    -----------------------------------------------
    A short call (selling a call) profits when the underlying stays flat
    or falls. So when our rules fire and we "sell a call", we are
    expressing a short / bearish view on the underlying. This is the
    natural bearish counterpart to the bullish file's "sell a put".
    """

    def __init__(self, settings: Optional[DonchianSettings] = None) -> None:
        # If the caller does not pass settings, fall back to the default
        # configuration from `DonchianSettings` (length = 20).
        self.settings = settings or DonchianSettings()

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Donchian Channel + bearish entry signals for the full history.

        Returns a NEW DataFrame (does not mutate `data`) with these extra
        columns appended to the cleaned OHLC columns:

        - upperDonchian               : rolling-max line (length bars)
        - middleDonchian              : average of upper and lower
        - lowerDonchian               : rolling-min line (length bars)
        - ruleCloseSecondAboveThird   : True where close[i]   > close[i-1]
        - ruleCloseThirdAboveFourth   : True where close[i-1] > close[i-2]
        - ruleDonchianUpperNonRising  : True where upper[i-2] <= upper[i-3]
        - startShortTrade             : True where ALL THREE rules hold
        - endShortTrade               : placeholder, always False (exit TBD)
        - startShortTradePrice        : close[i] on entry rows, NaN elsewhere
        - endShortTradePrice          : NaN (no exit logic yet)
        - backtestStream              : 1 = entry, 2 = exit (unused), 0 = hold
        """
        # -- Step 1: clean and normalize the input ---------------------------
        frame = _prepare_ohlc_frame(data)
        if frame.empty:
            # Nothing to analyze. We still return a DataFrame (not None) so
            # downstream code can safely call `.empty` / `len()` on the result.
            return frame

        # -- Step 2: validate settings ---------------------------------------
        length = int(self.settings.length)
        if length < 1:
            raise ValueError("Donchian length must be at least 1.")

        # -- Step 3: compute the Donchian channel ----------------------------
        # `_compute_donchian` returns three pandas Series that share the same
        # index as `frame`. NaN values appear during the warm-up period
        # (the first `length - 1` rows).
        upper, middle, lower = _compute_donchian(frame["high"], frame["low"], length)

        # -- Step 4: build per-bar rule masks (vectorized) -------------------
        #
        # Trick: `Series.shift(k)` returns a copy of the series shifted
        # forward by `k` rows. Position `i` of `series.shift(2)` therefore
        # holds `series[i - 2]`. This lets us express the per-bar rules as
        # whole-array comparisons, without writing a Python `for` loop.
        #
        # We need:
        #   close[i]    -> `close`
        #   close[i-1]  -> `close.shift(1)`
        #   close[i-2]  -> `close.shift(2)`
        #   upper[i-2]  -> `upper.shift(2)`
        #   upper[i-3]  -> `upper.shift(3)`
        close = frame["close"].astype(float)
        close_prev1 = close.shift(1)        # close[i-1] aligned to row i
        close_prev2 = close.shift(2)        # close[i-2] aligned to row i

        upper_prev2 = upper.shift(2)        # upper[i-2] aligned to row i
        upper_prev3 = upper.shift(3)        # upper[i-3] aligned to row i

        # `.fillna(False)` is defensive. On plain float Series, pandas
        # already returns False for any comparison involving NaN; but with
        # newer nullable dtypes it can return `pd.NA`. Forcing False keeps
        # the boolean array deterministic and free of "tri-state" entries.
        rule_close_a = (close > close_prev1).fillna(False).to_numpy(dtype=bool)
        rule_close_b = (close_prev1 > close_prev2).fillna(False).to_numpy(dtype=bool)
        rule_donchian = (upper_prev2 <= upper_prev3).fillna(False).to_numpy(dtype=bool)

        # -- Step 5: combine into the entry signal ---------------------------
        # All three rules must be true on the SAME bar `i` for a signal to
        # fire there. The bitwise `&` is the right operator for boolean
        # numpy arrays.
        start_short = rule_close_a & rule_close_b & rule_donchian

        # Exit logic is intentionally NOT defined yet (per the spec). We
        # keep the column shape so downstream code can stay schema-stable
        # and the bearish file's output line up perfectly with the bullish.
        end_short = np.zeros(len(frame), dtype=bool)

        # The integer event stream mirrors the bullish file:
        #   1 -> enter bearish, 2 -> exit bearish, 0 -> do nothing
        # `np.select` returns the value associated with the FIRST matching
        # condition per row; `default` covers rows where none match.
        backtest_stream = np.select(
            [start_short, end_short],
            [1, 2],
            default=0,
        )

        # -- Step 6: assemble the result DataFrame ---------------------------
        result = frame.copy()
        result["upperDonchian"] = upper.to_numpy(dtype=float)
        result["middleDonchian"] = middle.to_numpy(dtype=float)
        result["lowerDonchian"] = lower.to_numpy(dtype=float)
        result["ruleCloseSecondAboveThird"] = rule_close_a
        result["ruleCloseThirdAboveFourth"] = rule_close_b
        result["ruleDonchianUpperNonRising"] = rule_donchian
        result["startShortTrade"] = start_short
        result["endShortTrade"] = end_short
        # `np.where(mask, value_if_true, value_if_false)` fills a column
        # with close prices on event bars and NaN elsewhere -- same pattern
        # the bullish file uses for `startLongTradePrice` etc.
        result["startShortTradePrice"] = np.where(start_short, result["close"], np.nan)
        result["endShortTradePrice"] = np.where(end_short, result["close"], np.nan)
        result["backtestStream"] = backtest_stream
        return result

    def latest_signal(self, data: pd.DataFrame) -> BearishDecision:
        """
        Return a `BearishDecision` for the newest bar only.

        Use this in live trading / front-testing loops where you only care
        about what to do right now, not the full signal history.
        """
        generated = self.generate(data)
        if generated.empty:
            raise ValueError("No signal could be generated from the provided data.")

        # `.iloc[-1]` is the last row (newest bar) of the signal frame.
        last_row = generated.iloc[-1]

        # Translate boolean flags into a tuple of action strings.
        # We collect them in a list first, then freeze to a tuple so that
        # callers cannot accidentally mutate the returned decision object.
        actions: list[str] = []
        if bool(last_row["startShortTrade"]):
            actions.append("ENTER_SHORT")
        if bool(last_row["endShortTrade"]):
            actions.append("EXIT_SHORT")
        if not actions:
            actions.append("HOLD")

        # If the input had a timestamp column, use it; otherwise fall back
        # to the DataFrame's own index value for this row.
        timestamp = (
            last_row["timestamp"]
            if "timestamp" in generated.columns
            else generated.index[-1]
        )

        # Helper: convert NaN-ish values to a plain float NaN for the
        # dataclass. Donchian values may be NaN during the warm-up period.
        def _nan_safe_float(value: object) -> float:
            try:
                num = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return float("nan")
            return num

        return BearishDecision(
            timestamp=timestamp,
            close=_nan_safe_float(last_row["close"]),
            upper_donchian=_nan_safe_float(last_row["upperDonchian"]),
            middle_donchian=_nan_safe_float(last_row["middleDonchian"]),
            lower_donchian=_nan_safe_float(last_row["lowerDonchian"]),
            actions=tuple(actions),
            start_short_trade=bool(last_row["startShortTrade"]),
            end_short_trade=bool(last_row["endShortTrade"]),
            rule_close_pair_a=bool(last_row["ruleCloseSecondAboveThird"]),
            rule_close_pair_b=bool(last_row["ruleCloseThirdAboveFourth"]),
            rule_donchian_slope=bool(last_row["ruleDonchianUpperNonRising"]),
        )


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================
# These tiny wrappers exist so callers who do not want to instantiate the
# class can just import a single function and go. They use default settings
# unless the caller overrides them.
def generate_donchian_bearish_signals(
    data: pd.DataFrame,
    settings: Optional[DonchianSettings] = None,
) -> pd.DataFrame:
    """Functional alias for `DonchianBearishSignalGenerator(...).generate(data)`."""
    generator = DonchianBearishSignalGenerator(settings=settings)
    return generator.generate(data)


def get_latest_donchian_bearish_signal(
    data: pd.DataFrame,
    settings: Optional[DonchianSettings] = None,
) -> BearishDecision:
    """Functional alias for `DonchianBearishSignalGenerator(...).latest_signal(data)`."""
    generator = DonchianBearishSignalGenerator(settings=settings)
    return generator.latest_signal(data)


# `__all__` controls what `from module import *` exposes. Listing public
# names here also serves as a quick table-of-contents for readers.
__all__ = [
    "Direction",
    "DonchianSettings",
    "BearishDecision",
    "DonchianBearishSignalGenerator",
    "generate_donchian_bearish_signals",
    "get_latest_donchian_bearish_signal",
]
