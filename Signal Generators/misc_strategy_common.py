"""
Shared helpers for the "Misc Signal Generators" family.

These thirteen signal generators are this repo's re-implementations of the
strategies in the public TradingBot reference project. They deliberately follow
the same conventions as the existing strategy-logic modules in this codebase
(see ``Subhamoy Strategies/subhamoy_strategy_common.py`` and
``goldmine_strategy_logic.py``):

1. Normalize the caller's OHLC table into predictable lowercase columns.
2. Calculate indicators through TA-Lib first whenever it is available.
3. Fall back to pure pandas/numpy so the files remain usable on machines where
   the native TA-Lib package has not been installed yet.
4. Keep one copy of every shared indicator here so the thirteen generators stay
   small and consistent instead of each carrying its own boilerplate.

This module does NOT resample data. The front-test/data-fetch layer (or, inside
the master runner, ``resample_ohlc_from_1m``) is responsible for preparing the
candle timeframe before calling these strategies.

Beginner orientation
--------------------
- "OHLC" = the Open, High, Low, Close prices of one candle (one time bucket).
- An "indicator" is just a formula applied to those prices to summarise something
  (trend, momentum, volatility, ...). Every indicator function below takes price
  data in and returns a pandas Series (one number per candle) you can read.
- "TA-Lib first, pandas fallback" means: if the fast C library `talib` is
  installed we use it; otherwise we compute the same thing in plain pandas so the
  code still runs. Both paths aim to produce the same numbers.
- Warm-up: most indicators need a minimum number of candles before their value is
  meaningful, so the first few rows come back as NaN (Not a Number). That is
  expected, and the strategies skip those rows.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

try:
    # TA-Lib is the preferred indicator backend in this repo because it matches
    # the standard library indicator implementations used by the other folders.
    import talib
except ImportError:  # pragma: no cover - used only when TA-Lib is absent
    # The assignment error only exists where TA-Lib (with stubs) is installed;
    # on stub-less machines the ignore is unused, hence the dual code.
    talib = None  # type: ignore[assignment, unused-ignore]


OHLC_COLUMNS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Column / frame helpers (shared with the Subhamoy convention)
# ---------------------------------------------------------------------------
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


def add_candle_anatomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Add body, wick, and range columns used by several candle-pattern checks."""
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


def _as_float_array(values: pd.Series) -> np.ndarray:
    """Convert a Series to a contiguous float64 numpy array for TA-Lib."""
    return np.asarray(values, dtype="float64")


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(values: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average: the plain average of the last `period` values.

    Beginner view: it smooths the jagged price line into a single trend line.
    A rising SMA means price has been drifting up on average; a falling SMA means
    it has been drifting down. Bigger `period` = smoother but slower to react.
    (TA-Lib first, then a pandas rolling-mean fallback.)
    """
    period = int(period)
    if talib is not None:
        return pd.Series(talib.SMA(_as_float_array(values), timeperiod=period), index=values.index)
    return values.rolling(window=period, min_periods=period).mean()


def ema(values: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average: like the SMA, but recent candles count for more.

    Beginner view: because it weights the latest prices most heavily, the EMA
    "hugs" price more closely and turns faster than an SMA of the same length.
    That responsiveness is why crossover strategies prefer it. (TA-Lib first,
    then a pandas exponentially-weighted-mean fallback.)
    """
    period = int(period)
    if talib is not None:
        return pd.Series(talib.EMA(_as_float_array(values), timeperiod=period), index=values.index)
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """
    Average True Range: the average size of a candle over `period` bars.

    Beginner view: ATR measures volatility ("how far price typically travels in
    one candle"), NOT direction. A large ATR means big, jumpy candles; a small
    ATR means a quiet, tight market. Strategies use it to size stops/targets to
    current conditions (e.g. "stop = 1.5 x ATR away"). (TA-Lib first, then a
    Wilder-smoothed pandas fallback.)
    """
    period = int(period)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    if talib is not None:
        return pd.Series(
            talib.ATR(_as_float_array(high), _as_float_array(low), _as_float_array(close), timeperiod=period),
            index=frame.index,
        )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    """
    True Range: one candle's full travel, including any overnight/gap move.

    It is the largest of: this candle's high-low, |high - previous close|, and
    |low - previous close|. ATR (above) is just a smoothed average of this. Used
    here by the ADX fallback.
    """
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    previous_close = frame["close"].astype(float).shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------
def rsi(values: pd.Series, period: int) -> pd.Series:
    """
    Relative Strength Index: a 0-100 momentum gauge of recent up- vs down-moves.

    Beginner view: RSI compares the size of recent gains to recent losses.
    - Near 100 -> price has been rising hard ("overbought", possibly stretched).
    - Near 0   -> price has been falling hard ("oversold").
    - 50       -> balanced. The classic 30/70 lines flag oversold/overbought.
    Note RSI depends on the up/down *ratio*, not the size of the move, so a long
    one-directional run pins it near 0 or 100. (TA-Lib first, then a Wilder-style
    pandas fallback.)
    """
    period = int(period)
    if talib is not None:
        return pd.Series(talib.RSI(_as_float_array(values), timeperiod=period), index=values.index)
    delta = values.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero the market only rose, so RSI saturates at 100.
    result = result.where(avg_loss != 0.0, 100.0)
    return result


def macd(
    values: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence): a momentum/trend indicator.

    Beginner view, three outputs:
    - macd line   = fast EMA minus slow EMA (positive = short-term above long-term).
    - signal line = an EMA of the macd line (a smoothed version of it).
    - histogram   = macd minus signal. Histogram > 0 means upward momentum is
      building; < 0 means downward momentum. Strategies here mostly read the
      histogram's sign. (TA-Lib first, then a pandas EWM fallback.)
    """
    fast_period, slow_period, signal_period = int(fast_period), int(slow_period), int(signal_period)
    if talib is not None:
        macd_line, signal_line, hist = talib.MACD(
            _as_float_array(values),
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )
        return (
            pd.Series(macd_line, index=values.index),
            pd.Series(signal_line, index=values.index),
            pd.Series(hist, index=values.index),
        )
    fast_ema = values.ewm(span=fast_period, adjust=False).mean()
    slow_ema = values.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(
    frame: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Slow Stochastic Oscillator returning (%K, %D), each on a 0-100 scale.

    Beginner view: it asks "where is the close sitting inside the recent
    high-low range?" Near 100 = closing at the top of the range (strong);
    near 0 = closing at the bottom (weak). %K is the raw line, %D is a slower
    average of %K; a %K-crossing-%D event is the usual trade trigger, and the
    20/80 levels mark oversold/overbought.

    Mirrors TA-Lib STOCH semantics: raw %K is smoothed by `smooth_k` to make the
    "slow %K", and %D is the `d_period` SMA of slow %K. (TA-Lib first, then a
    pandas fallback.)
    """
    k_period, d_period, smooth_k = int(k_period), int(d_period), int(smooth_k)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    if talib is not None:
        slowk, slowd = talib.STOCH(
            _as_float_array(high),
            _as_float_array(low),
            _as_float_array(close),
            fastk_period=k_period,
            slowk_period=smooth_k,
            slowk_matype=0,  # type: ignore[arg-type, unused-ignore]
            slowd_period=d_period,
            slowd_matype=0,  # type: ignore[arg-type, unused-ignore]
        )
        return pd.Series(slowk, index=frame.index), pd.Series(slowd, index=frame.index)
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
    slow_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    slow_d = slow_k.rolling(window=d_period, min_periods=d_period).mean()
    return slow_k, slow_d


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index: trend *strength* on a 0-100 scale (not direction).

    Beginner view: ADX answers "is there a real trend right now, or just chop?"
    - Above ~25 -> a strong trend is in progress (worth trend-following).
    - Below ~20 -> weak/sideways; breakout and trend signals fail more often.
    It says nothing about up vs down - only how strong the move is. Strategies use
    it as a filter (e.g. only take a Parabolic SAR flip when ADX is high enough).
    (TA-Lib first, then a Wilder-smoothed pandas fallback.)
    """
    period = int(period)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    if talib is not None:
        return pd.Series(
            talib.ADX(_as_float_array(high), _as_float_array(low), _as_float_array(close), timeperiod=period),
            index=frame.index,
        )
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=frame.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=frame.index)
    tr = true_range(frame)
    atr_wilder = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_wilder
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_wilder
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Bands / channels
# ---------------------------------------------------------------------------
def bollinger_bands(
    values: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands: a moving average with a volatility "envelope" around it.

    Beginner view: the middle band is an SMA; the upper/lower bands sit `num_std`
    standard deviations above/below it. Price spends most of its time between the
    bands, so a tag of the lower band can signal "stretched low" (bounce candidate)
    and the upper band "stretched high". The bands widen when volatility rises and
    pinch together when it falls. Returns (upper, middle, lower). (TA-Lib first,
    then a pandas fallback; both use population std to match.)
    """
    period = int(period)
    num_std = float(num_std)
    if talib is not None:
        upper, middle, lower = talib.BBANDS(
            _as_float_array(values),
            timeperiod=period,
            nbdevup=num_std,
            nbdevdn=num_std,
            matype=0,  # type: ignore[arg-type, unused-ignore]
        )
        return (
            pd.Series(upper, index=values.index),
            pd.Series(middle, index=values.index),
            pd.Series(lower, index=values.index),
        )
    middle = values.rolling(window=period, min_periods=period).mean()
    # TA-Lib BBANDS uses the population standard deviation (ddof=0); match it so
    # the fallback and the native backend agree.
    rolling_std = values.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


def keltner_channels(
    frame: pd.DataFrame,
    period: int = 20,
    atr_period: int = 20,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels: an EMA with an ATR-based envelope (a volatility channel).

    Beginner view: similar idea to Bollinger Bands, but the band width is based on
    ATR (candle size) instead of standard deviation. The middle is an EMA of close;
    the bands sit `multiplier` x ATR above/below it. The "Keltner squeeze" strategy
    compares these against Bollinger Bands: when the (std-based) Bollinger Bands sit
    *inside* the (ATR-based) Keltner Channels, volatility is unusually low - a
    "squeeze" that often precedes a breakout. Returns (upper, middle, lower).

    There is no TA-Lib primitive for Keltner, so this is always computed from the
    shared ema()/atr() helpers (which are themselves TA-Lib-first).
    """
    middle = ema(frame["close"].astype(float), period)
    channel_atr = atr(frame, atr_period)
    upper = middle + float(multiplier) * channel_atr
    lower = middle - float(multiplier) * channel_atr
    return upper, middle, lower


def rolling_zscore(values: pd.Series, period: int) -> pd.Series:
    """
    Rolling z-score: how many standard deviations price is from its recent mean.

    Beginner view: z = (price - rolling average) / rolling standard deviation.
    z = 0 means price is exactly average; z = +2 means it is 2 std-devs *above*
    average (unusually high); z = -2 means 2 std-devs below (unusually low). Mean-
    reversion strategies fade these extremes, betting price drifts back toward 0.
    (The tiny epsilon avoids divide-by-zero on a perfectly flat window.)
    """
    period = int(period)
    mean = values.rolling(window=period, min_periods=period).mean()
    std = values.rolling(window=period, min_periods=period).std(ddof=0)
    return (values - mean) / (std + 1e-10)


# ---------------------------------------------------------------------------
# Trend-following lines (no TA-Lib primitive -> manual, deterministic)
# ---------------------------------------------------------------------------
def parabolic_sar(
    frame: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """
    Parabolic SAR ("Stop And Reverse"): a trailing dot that flips with the trend.

    Beginner view: SAR plots a dot each candle. While the dot is *below* price the
    trend is up (direction +1); when price crosses it, the dot jumps *above* price
    and the trend flips to down (direction -1), and vice-versa. Traders use the dot
    as a trailing stop and each flip as a reversal signal. The acceleration factor
    makes the dot speed up the longer a trend runs. Returns (sar_value, direction),
    where direction is +1 (up) or -1 (down), NaN during warm-up.

    TA-Lib's SAR only exposes a single acceleration knob, so when
    `af_start == af_step` we use it; otherwise we fall back to the classic
    iterative algorithm (which supports an independent start vs. step).
    """
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    n = len(frame)
    if n == 0:
        empty = pd.Series([], dtype="float64", index=frame.index)
        return empty, empty

    if talib is not None and float(af_start) == float(af_step):
        sar_values = talib.SAR(_as_float_array(high), _as_float_array(low),
                               acceleration=float(af_start), maximum=float(af_max))
        sar_series = pd.Series(sar_values, index=frame.index)
        # SAR below the candle => uptrend (+1); SAR above => downtrend (-1).
        direction = pd.Series(np.where(sar_series <= low.to_numpy(), 1, -1), index=frame.index)
        direction = direction.mask(sar_series.isna())
        return sar_series, direction

    highs = high.to_numpy()
    lows = low.to_numpy()
    sar = np.full(n, np.nan, dtype="float64")
    trend = np.zeros(n, dtype="float64")

    # Seed the first bar from the first two candles.
    long_trend = highs[min(1, n - 1)] >= highs[0]
    af = float(af_start)
    ep = highs[0] if long_trend else lows[0]
    sar[0] = lows[0] if long_trend else highs[0]
    trend[0] = 1.0 if long_trend else -1.0

    for i in range(1, n):
        prior_sar = sar[i - 1]
        sar[i] = prior_sar + af * (ep - prior_sar)
        if long_trend:
            # SAR can never penetrate the prior two lows in an uptrend.
            sar[i] = min(sar[i], lows[i - 1], lows[max(0, i - 2)])
            if lows[i] < sar[i]:
                long_trend = False
                sar[i] = ep
                ep = lows[i]
                af = float(af_start)
            elif highs[i] > ep:
                ep = highs[i]
                af = min(af + float(af_step), float(af_max))
        else:
            sar[i] = max(sar[i], highs[i - 1], highs[max(0, i - 2)])
            if highs[i] > sar[i]:
                long_trend = True
                sar[i] = ep
                ep = highs[i]
                af = float(af_start)
            elif lows[i] < ep:
                ep = lows[i]
                af = min(af + float(af_step), float(af_max))
        trend[i] = 1.0 if long_trend else -1.0

    return pd.Series(sar, index=frame.index), pd.Series(trend, index=frame.index)


def supertrend(
    frame: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend: an ATR-based trailing line that sits below price in up-trends and
    above price in down-trends, returning (supertrend_line, direction).

    Beginner view: it draws a single line that "trails" price by `multiplier` x ATR.
    While price holds above the line the trend is up (direction +1) and the line
    acts as a rising stop; when price closes through it, the line flips to the other
    side and the trend turns down (direction -1). Each flip is a reversal signal.
    Bigger `multiplier` = looser line (fewer, later flips).

    Computed iteratively with the standard band carry-forward rules; ATR comes from
    the shared TA-Lib-first helper. `direction` is NaN during the ATR warm-up.
    """
    atr_period = int(atr_period)
    multiplier = float(multiplier)
    n = len(frame)
    if n == 0:
        empty = pd.Series([], dtype="float64", index=frame.index)
        return empty, empty

    high = frame["high"].astype(float).to_numpy()
    low = frame["low"].astype(float).to_numpy()
    close = frame["close"].astype(float).to_numpy()
    atr_values = atr(frame, atr_period).to_numpy()

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr_values
    basic_lower = hl2 - multiplier * atr_values

    final_upper = np.full(n, np.nan, dtype="float64")
    final_lower = np.full(n, np.nan, dtype="float64")
    supertrend_line = np.full(n, np.nan, dtype="float64")
    # Direction stays NaN during the ATR warm-up; only resolved bars get +/-1.
    direction = np.full(n, np.nan, dtype="float64")

    for i in range(n):
        if np.isnan(atr_values[i]):
            # Still inside the ATR warm-up window: nothing to assert yet.
            continue
        if np.isnan(final_upper[i - 1]) if i > 0 else True:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            direction[i] = 1.0 if close[i] >= hl2[i] else -1.0
            supertrend_line[i] = final_lower[i] if direction[i] > 0 else final_upper[i]
            continue

        final_upper[i] = (
            basic_upper[i]
            if (basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if (basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        prior_dir = direction[i - 1]
        if prior_dir >= 0:
            direction[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > final_upper[i] else -1.0
        supertrend_line[i] = final_lower[i] if direction[i] > 0 else final_upper[i]

    return pd.Series(supertrend_line, index=frame.index), pd.Series(direction, index=frame.index)


# ---------------------------------------------------------------------------
# Swing-point detection (used by the divergence strategy)
# ---------------------------------------------------------------------------
def find_swing_lows(values: pd.Series, window: int) -> list[int]:
    """
    Return positional indices of confirmed swing lows ("local bottoms").

    Beginner view: a swing low is a candle that is lower than the `window` candles
    on BOTH sides of it - a little valley in the line. Because we need to see
    `window` candles afterwards to confirm it, a swing low is only known `window`
    bars later (this is deliberate - it avoids "peeking" into the future). The RSI
    divergence strategy compares the last two swing lows of price vs. RSI.
    Indices returned are positional (0..len-1).
    """
    window = int(window)
    series = values.reset_index(drop=True)
    swings: list[int] = []
    n = len(series)
    for i in range(window, n - window):
        center = series.iloc[i]
        if pd.isna(center):
            continue
        left = series.iloc[i - window:i]
        right = series.iloc[i + 1:i + window + 1]
        if center <= left.min() and center <= right.min():
            swings.append(i)
    return swings


def find_swing_highs(values: pd.Series, window: int) -> list[int]:
    """
    Return positional indices of confirmed swing highs ("local tops").

    The exact mirror of find_swing_lows: a candle higher than the `window` candles
    on both sides (a little peak). Same `window`-bar confirmation delay applies.
    """
    window = int(window)
    series = values.reset_index(drop=True)
    swings: list[int] = []
    n = len(series)
    for i in range(window, n - window):
        center = series.iloc[i]
        if pd.isna(center):
            continue
        left = series.iloc[i - window:i]
        right = series.iloc[i + 1:i + window + 1]
        if center >= left.max() and center >= right.max():
            swings.append(i)
    return swings


__all__ = [
    "OHLC_COLUMNS",
    "add_candle_anatomy",
    "adx",
    "atr",
    "bollinger_bands",
    "ema",
    "falling_over_lookback",
    "find_first_col",
    "find_swing_highs",
    "find_swing_lows",
    "finite",
    "keltner_channels",
    "macd",
    "normalize_ohlc_frame",
    "parabolic_sar",
    "require_columns",
    "rising_over_lookback",
    "rolling_zscore",
    "rsi",
    "sma",
    "stochastic",
    "supertrend",
    "true_range",
]
