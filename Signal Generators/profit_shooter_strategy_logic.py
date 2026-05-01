"""
Shared Profit Shooter strategy logic driven by normal OHLC candles.

What this file does:
1. Build indicator and candle-structure columns needed by the strategy.
2. Detect bullish and bearish pin bars using configurable thresholds.
3. Detect full long/short setup candles.
4. Evaluate next-candle breakout entries and trade-management exits.

Quick-start:
1. Call `build_profit_shooter_with_indicators()` on OHLC candles.
2. Keep one `ProfitShooterSignalEngine()` instance alive across evaluations.
3. When flat, call `evaluate_candle(...)` on each newly completed candle.
4. When in a trade, pass `ProfitShooterPositionContext(...)`.
5. If the engine returns `HOLD`, also persist its returned trailing-state flags.

Important note about trailing exits:
- This strategy exits from EMA9 trailing mode on the *next candle open*
  after a completed candle closes beyond EMA9.
- Because of that, the returned decision also carries:
  - `trailing_active`
  - `pending_trailing_exit`
- The caller must keep those values in its open-position state, similar to how
  the Renko strategy keeps RR-armed state.

Indicator note:
- TA-Lib is used wherever it makes sense for built-in indicators:
  - SMA
  - EMA
  - ATR
- If TA-Lib is not installed, the file falls back to pandas-based
  implementations so the strategy can still run.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    # TA-Lib is the preferred path because it gives us the standard,
    # battle-tested indicator implementations traders usually expect from
    # charting software and broker platforms.
    import talib
except ImportError:  # pragma: no cover - fallback only used when TA-Lib is unavailable
    # We intentionally keep a fallback path so the strategy does not become
    # unusable in environments where TA-Lib is not installed yet.
    talib = None


@dataclass(frozen=True)
class ProfitShooterConfig:
    """All strategy parameters are configurable from one place."""

    # Trend filter periods:
    # - SMA20 is the faster structure filter.
    # - SMA200 is the higher-timeframe directional filter.
    sma_fast_period: int = 20
    sma_slow_period: int = 200

    # Volatility and trailing periods:
    # - ATR14 is used for proximity checks and stop-loss buffering.
    # - EMA9 is used only after 1.5R is reached, for trailing exits.
    atr_period: int = 14
    trailing_ema_period: int = 9

    # "SMA20 is rising/falling over the last 3 completed candles" is
    # implemented through `trend_lookback`.
    trend_lookback: int = 3

    # Pullback detection:
    # "At least 2 of the last 3 completed candles" is represented by:
    # - `pullback_lookback = 3`
    # - `pullback_min_count = 2`
    pullback_lookback: int = 3
    pullback_min_count: int = 2

    # Entry and risk parameters from the brief.
    proximity_atr_multiple: float = 0.5
    tick_size: float = 0.05
    sl_atr_buffer: float = 0.2
    target_r_multiple: float = 1.5

    # Pin-bar geometry rules.
    pin_bar_body_max_ratio: float = 0.30
    pin_bar_long_wick_min_ratio: float = 0.60
    pin_bar_wick_to_body_multiple: float = 2.0
    pin_bar_opposite_wick_max_ratio: float = 0.15
    pin_bar_end_zone_ratio: float = 0.40

    def __post_init__(self) -> None:
        # Validate all "count/period" values up front so the rest of the code
        # can safely assume the config is mathematically valid.
        period_fields = {
            "sma_fast_period": self.sma_fast_period,
            "sma_slow_period": self.sma_slow_period,
            "atr_period": self.atr_period,
            "trailing_ema_period": self.trailing_ema_period,
            "trend_lookback": self.trend_lookback,
            "pullback_lookback": self.pullback_lookback,
            "pullback_min_count": self.pullback_min_count,
        }
        invalid_periods = [name for name, value in period_fields.items() if int(value) <= 0]
        if invalid_periods:
            raise ValueError(
                "All period/count values must be positive. "
                f"Invalid: {', '.join(invalid_periods)}"
            )

        # The strategy cannot ask for "at least 5 pullback candles in the last
        # 3 candles", so this relationship must also be checked.
        if self.pullback_min_count > self.pullback_lookback:
            raise ValueError("`pullback_min_count` cannot be larger than `pullback_lookback`.")

        # Ratios like 0.30 or 0.60 represent percentages of the candle range,
        # so they must stay between 0 and 1.
        bounded_ratios = {
            "pin_bar_body_max_ratio": self.pin_bar_body_max_ratio,
            "pin_bar_long_wick_min_ratio": self.pin_bar_long_wick_min_ratio,
            "pin_bar_opposite_wick_max_ratio": self.pin_bar_opposite_wick_max_ratio,
            "pin_bar_end_zone_ratio": self.pin_bar_end_zone_ratio,
        }
        invalid_ratios = [
            name for name, value in bounded_ratios.items() if not (0.0 <= float(value) <= 1.0)
        ]
        if invalid_ratios:
            raise ValueError(
                "The following ratios must be between 0.0 and 1.0: "
                f"{', '.join(invalid_ratios)}"
            )

        # The remaining values are allowed to be zero or positive depending on
        # their purpose, but negative values would make no trading sense.
        if float(self.pin_bar_wick_to_body_multiple) < 0.0:
            raise ValueError("`pin_bar_wick_to_body_multiple` must be non-negative.")
        if float(self.proximity_atr_multiple) < 0.0:
            raise ValueError("`proximity_atr_multiple` must be non-negative.")
        if float(self.tick_size) < 0.0:
            raise ValueError("`tick_size` must be non-negative.")
        if float(self.sl_atr_buffer) < 0.0:
            raise ValueError("`sl_atr_buffer` must be non-negative.")
        if float(self.target_r_multiple) <= 0.0:
            raise ValueError("`target_r_multiple` must be greater than 0.")


@dataclass
class ProfitShooterPositionContext:
    """
    Minimal snapshot of an already-open trade.

    The signal engine needs these values to manage stop, 1.5R activation, and
    EMA9 trailing exits.
    """

    # LONG or SHORT.
    direction: str
    # Actual underlying entry price used by the strategy.
    entry_underlying: float
    # Fixed protective stop price.
    stop_underlying: float
    # Initial 1.5R target price. Optional because it can be rebuilt from entry
    # and stop if the caller does not persist it explicitly.
    target_underlying: float = 0.0
    # Becomes True once price reaches the initial 1.5R target.
    trailing_active: bool = False
    # Becomes True after a completed candle closes beyond EMA9 in trailing
    # mode. The actual exit is then taken at the next candle open.
    pending_trailing_exit: bool = False


@dataclass
class ProfitShooterDecision:
    """
    Standard response object returned by the signal engine.

    `action` can be:
    - HOLD
    - ENTER_LONG
    - ENTER_SHORT
    - EXIT
    """

    # Final decision for the caller.
    action: str = "HOLD"
    # Entry payload fields, used only on ENTER_* decisions.
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    target_underlying: float = 0.0
    # Exit payload field, used only on EXIT decisions.
    exit_underlying: float = 0.0
    # Machine-friendly exit label like STOP or EMA9_TRAIL_EXIT.
    exit_reason: str = ""
    # True means a strategy setup/trigger happened, even if final validation
    # later rejected the entry.
    signal_triggered: bool = False
    # Updated trade-management state that the caller should persist.
    trailing_active: bool = False
    pending_trailing_exit: bool = False


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Fail early with a clear message when OHLC columns are missing."""
    # This prevents later KeyError-style failures from showing up in confusing
    # places deeper inside the strategy.
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _resolve_timestamp(ohlc: pd.DataFrame) -> pd.Series:
    """Accept either a `timestamp` column or a datetime-like index."""
    # Some callers keep time in a proper column while others use the DataFrame
    # index. This helper supports both shapes so upstream code stays simple.
    if "timestamp" in ohlc.columns:
        timestamp = pd.to_datetime(ohlc["timestamp"], errors="coerce")
    else:
        timestamp = pd.to_datetime(pd.Series(ohlc.index, index=ohlc.index), errors="coerce")

    # If every value is invalid, we cannot safely sort candles in time order.
    if timestamp.isna().all():
        raise ValueError("A valid 'timestamp' column or datetime-like index is required.")
    return timestamp


def _fallback_sma(values: pd.Series, period: int) -> pd.Series:
    """SMA fallback used only when TA-Lib is unavailable."""
    # A simple moving average is just the mean of the last `period` values.
    return values.rolling(window=period, min_periods=period).mean()


def _fallback_ema(values: pd.Series, period: int) -> pd.Series:
    """EMA fallback used only when TA-Lib is unavailable."""
    # `adjust=False` keeps the recursive EMA behavior most traders expect.
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def _fallback_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR fallback matching the standard True Range definition."""
    # ATR first needs True Range, which captures both the candle's internal
    # range and any gap versus the previous close.
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _strict_monotonic(series: pd.Series, lookback: int, direction: str) -> pd.Series:
    """
    Check whether values are strictly rising or falling over `lookback` candles.

    Example for `lookback=3`:
    - rising  => s[t] > s[t-1] > s[t-2]
    - falling => s[t] < s[t-1] < s[t-2]
    """
    if lookback < 2:
        raise ValueError("`lookback` must be at least 2 for monotonic checks.")

    # For a 3-candle rising check, we build:
    # - current > previous
    # - previous > previous-1
    # and then combine them with AND.
    comparisons = []
    for offset in range(lookback - 1):
        left = series.shift(offset)
        right = series.shift(offset + 1)
        if direction == "up":
            comparisons.append(left > right)
        elif direction == "down":
            comparisons.append(left < right)
        else:
            raise ValueError("`direction` must be either 'up' or 'down'.")

    result = comparisons[0]
    for comparison in comparisons[1:]:
        result = result & comparison
    return result


def _build_bullish_pin_bar(frame: pd.DataFrame, config: ProfitShooterConfig) -> pd.Series:
    """Return True when a candle matches the bullish pin-bar definition."""
    # `upper_zone_floor` means:
    # "open and close must be in the upper X% of the candle range".
    # With the default 40% rule, both open and close must sit in the top 40%
    # portion of the candle.
    candle_range = frame["candle_range"]
    range_positive = candle_range > 0.0
    upper_zone_floor = frame["low"] + (1.0 - config.pin_bar_end_zone_ratio) * candle_range

    return (
        range_positive
        & (frame["body"] <= config.pin_bar_body_max_ratio * candle_range)
        & (frame["lower_wick"] >= config.pin_bar_long_wick_min_ratio * candle_range)
        & (frame["lower_wick"] >= config.pin_bar_wick_to_body_multiple * frame["body"])
        & (frame["upper_wick"] <= config.pin_bar_opposite_wick_max_ratio * candle_range)
        & (frame["open"] >= upper_zone_floor)
        & (frame["close"] >= upper_zone_floor)
    )


def _build_bearish_pin_bar(frame: pd.DataFrame, config: ProfitShooterConfig) -> pd.Series:
    """Return True when a candle matches the bearish pin-bar definition."""
    # `lower_zone_ceiling` means:
    # "open and close must be in the lower X% of the candle range".
    candle_range = frame["candle_range"]
    range_positive = candle_range > 0.0
    lower_zone_ceiling = frame["low"] + config.pin_bar_end_zone_ratio * candle_range

    return (
        range_positive
        & (frame["body"] <= config.pin_bar_body_max_ratio * candle_range)
        & (frame["upper_wick"] >= config.pin_bar_long_wick_min_ratio * candle_range)
        & (frame["upper_wick"] >= config.pin_bar_wick_to_body_multiple * frame["body"])
        & (frame["lower_wick"] <= config.pin_bar_opposite_wick_max_ratio * candle_range)
        & (frame["open"] <= lower_zone_ceiling)
        & (frame["close"] <= lower_zone_ceiling)
    )


def _build_long_setup(frame: pd.DataFrame, config: ProfitShooterConfig) -> pd.Series:
    """Combine every bullish setup filter from the strategy brief."""
    # This is the full "ready" candle, not the final entry trigger.
    # Actual entry still waits for the *next* candle to break above the setup
    # candle's high.
    return (
        (frame["close"] > frame["sma200"])
        & (frame["sma20"] > frame["sma200"])
        & frame["sma20_rising"]
        & frame["near_sma20"]
        & frame["recent_long_pullback"]
        & frame["bullish_pin_bar"]
    )


def _build_short_setup(frame: pd.DataFrame, config: ProfitShooterConfig) -> pd.Series:
    """Combine every bearish setup filter from the strategy brief."""
    # Same idea as long setup, but mirrored for bearish conditions.
    return (
        (frame["close"] < frame["sma200"])
        & (frame["sma20"] < frame["sma200"])
        & frame["sma20_falling"]
        & frame["near_sma20"]
        & frame["recent_short_pullback"]
        & frame["bearish_pin_bar"]
    )


def build_profit_shooter_with_indicators(
    ohlc: pd.DataFrame,
    config: Optional[ProfitShooterConfig] = None,
) -> pd.DataFrame:
    """
    Enrich OHLC candles with the indicators and derived columns used by the strategy.

    Output columns added:
    - timestamp
    - sma20, sma200, ema9, atr
    - candle_range, body, upper_wick, lower_wick
    - bullish_pin_bar, bearish_pin_bar
    - sma20_rising, sma20_falling
    - near_sma20
    - recent_long_pullback, recent_short_pullback
    - long_setup, short_setup
    - long_breakout_level, short_breakout_level
    - long_entry_trigger, short_entry_trigger
    - long_entry_price, short_entry_price
    - long_stop_from_setup, short_stop_from_setup
    - long_target_from_setup, short_target_from_setup
    """
    # If caller does not pass a custom config, use the strategy defaults.
    config = config or ProfitShooterConfig()
    _require_columns(ohlc, ["open", "high", "low", "close"])

    # Work on a copy so we never mutate the caller's original DataFrame.
    frame = ohlc.copy()
    frame["timestamp"] = _resolve_timestamp(frame)
    # Strategy rules must be evaluated in true time order. Duplicate timestamps
    # are removed so no candle is counted twice.
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Convert price columns to numeric safely. Bad strings become NaN instead
    # of breaking the strategy mid-run.
    numeric_columns = [column for column in ("open", "high", "low", "close") if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # A candle without OHLC prices is not usable, so we drop such rows.
    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    # Indicator stage:
    # Prefer TA-Lib whenever available. This is exactly what the user asked
    # for. If TA-Lib is missing, fall back to pandas implementations so the
    # file remains portable.
    if talib is not None:
        close_values = close.to_numpy(dtype="float64")
        high_values = high.to_numpy(dtype="float64")
        low_values = low.to_numpy(dtype="float64")

        # TA-Lib path:
        # - SMA20 and SMA200 for trend structure
        # - EMA9 for trailing exit logic
        # - ATR14 for proximity and stop buffering
        frame["sma20"] = talib.SMA(close_values, timeperiod=config.sma_fast_period)
        frame["sma200"] = talib.SMA(close_values, timeperiod=config.sma_slow_period)
        frame["ema9"] = talib.EMA(close_values, timeperiod=config.trailing_ema_period)
        frame["atr"] = talib.ATR(
            high_values,
            low_values,
            close_values,
            timeperiod=config.atr_period,
        )
    else:
        # Fallback path used only when TA-Lib is unavailable.
        frame["sma20"] = _fallback_sma(close, config.sma_fast_period)
        frame["sma200"] = _fallback_sma(close, config.sma_slow_period)
        frame["ema9"] = _fallback_ema(close, config.trailing_ema_period)
        frame["atr"] = _fallback_atr(frame, config.atr_period)

    # Candle anatomy stage:
    # These are the exact measurements required by the pin-bar definition.
    frame["candle_range"] = (frame["high"] - frame["low"]).abs()
    frame["body"] = (frame["close"] - frame["open"]).abs()
    frame["upper_wick"] = frame["high"] - frame[["open", "close"]].max(axis=1)
    frame["lower_wick"] = frame[["open", "close"]].min(axis=1) - frame["low"]

    # Pin-bar classification stage.
    frame["bullish_pin_bar"] = _build_bullish_pin_bar(frame, config)
    frame["bearish_pin_bar"] = _build_bearish_pin_bar(frame, config)

    # Trend-quality stage:
    # Example default bullish rule:
    # sma20[t] > sma20[t-1] > sma20[t-2]
    frame["sma20_rising"] = _strict_monotonic(frame["sma20"], config.trend_lookback, "up")
    frame["sma20_falling"] = _strict_monotonic(frame["sma20"], config.trend_lookback, "down")

    # Proximity stage:
    # "Price is close to SMA20" means the close must sit within a configurable
    # ATR-based distance from SMA20.
    frame["near_sma20"] = (
        (frame["atr"] > 0.0)
        & ((frame["close"] - frame["sma20"]).abs() <= config.proximity_atr_multiple * frame["atr"])
    )

    # Pullback stage:
    # - Long setup wants at least `pullback_min_count` down-closing candles in
    #   the last `pullback_lookback` candles.
    # - Short setup wants the mirror image with up-closing candles.
    down_close = frame["close"] < frame["close"].shift(1)
    up_close = frame["close"] > frame["close"].shift(1)
    frame["recent_long_pullback"] = (
        down_close.rolling(window=config.pullback_lookback, min_periods=config.pullback_lookback).sum()
        >= config.pullback_min_count
    )
    frame["recent_short_pullback"] = (
        up_close.rolling(window=config.pullback_lookback, min_periods=config.pullback_lookback).sum()
        >= config.pullback_min_count
    )

    # Full setup candles:
    # These are the pin-bar candles that satisfy every structural filter.
    frame["long_setup"] = _build_long_setup(frame, config)
    frame["short_setup"] = _build_short_setup(frame, config)

    # Next-candle trigger stage:
    # The actual trade does not happen on the setup candle itself.
    # It happens only on the next candle if that candle breaks the setup
    # candle's high/low by at least `tick_size`.
    prev_high = frame["high"].shift(1)
    prev_low = frame["low"].shift(1)
    prev_open = frame["open"].shift(1)
    prev_close = frame["close"].shift(1)
    prev_atr = frame["atr"].shift(1)
    prev_long_setup = frame["long_setup"].shift(1).eq(True)
    prev_short_setup = frame["short_setup"].shift(1).eq(True)

    # Breakout levels are based on the previous candle because the previous
    # candle is the actual pin-bar setup candle.
    frame["long_breakout_level"] = prev_high + config.tick_size
    frame["short_breakout_level"] = prev_low - config.tick_size
    frame["long_entry_trigger"] = prev_long_setup & (frame["high"] >= frame["long_breakout_level"])
    frame["short_entry_trigger"] = prev_short_setup & (frame["low"] <= frame["short_breakout_level"])

    # Entry price rule from the brief:
    # - Long  = max(next candle open, pin-bar high + tick_size)
    # - Short = min(next candle open, pin-bar low  - tick_size)
    frame["long_entry_price"] = np.where(
        frame["long_entry_trigger"],
        np.maximum(frame["open"], frame["long_breakout_level"]),
        np.nan,
    )
    frame["short_entry_price"] = np.where(
        frame["short_entry_trigger"],
        np.minimum(frame["open"], frame["short_breakout_level"]),
        np.nan,
    )

    # Initial stop-loss and 1.5R target are both derived from the setup candle
    # and that candle's ATR value.
    frame["long_stop_from_setup"] = prev_low - config.sl_atr_buffer * prev_atr
    frame["short_stop_from_setup"] = prev_high + config.sl_atr_buffer * prev_atr
    long_risk = frame["long_entry_price"] - frame["long_stop_from_setup"]
    short_risk = frame["short_stop_from_setup"] - frame["short_entry_price"]
    frame["long_target_from_setup"] = frame["long_entry_price"] + config.target_r_multiple * long_risk
    frame["short_target_from_setup"] = frame["short_entry_price"] - config.target_r_multiple * short_risk

    # Keep a few shifted source values for easy debugging when inspecting rows.
    frame["setup_high"] = prev_high
    frame["setup_low"] = prev_low
    frame["setup_open"] = prev_open
    frame["setup_close"] = prev_close
    return frame


class ProfitShooterSignalEngine:
    """
    Decision engine for the Profit Shooter pin-bar strategy.

    This class reads the indicator-enriched DataFrame built by
    `build_profit_shooter_with_indicators()` and returns one clean decision.
    """

    def __init__(self, config: Optional[ProfitShooterConfig] = None):
        # Store config on the engine so every caller shares the same rule set.
        self.config = config or ProfitShooterConfig()

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        # Normalize values like "long", " LONG ", etc.
        return str(direction).strip().upper()

    @staticmethod
    def _hold(
        *,
        signal_triggered: bool = False,
        trailing_active: bool = False,
        pending_trailing_exit: bool = False,
        target_underlying: float = 0.0,
    ) -> ProfitShooterDecision:
        # Central helper for "do nothing right now" responses.
        # We still pass state back because trailing-related flags can change
        # even when the final action is HOLD.
        return ProfitShooterDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            trailing_active=trailing_active,
            pending_trailing_exit=pending_trailing_exit,
            target_underlying=target_underlying,
        )

    def _minimum_history_bars(self) -> int:
        # We need enough history for:
        # 1. the longest moving average,
        # 2. ATR warm-up,
        # 3. EMA9 warm-up,
        # 4. lookback-based rising/falling and pullback checks,
        # 5. and one extra candle because entries depend on the prior setup bar.
        return (
            max(
                self.config.sma_fast_period,
                self.config.sma_slow_period,
                self.config.atr_period,
                self.config.trailing_ema_period,
            )
            + max(self.config.trend_lookback, self.config.pullback_lookback)
            + 1
        )

    def _resolve_target(self, position: ProfitShooterPositionContext) -> float:
        """Use stored target if present; otherwise rebuild it from entry and stop."""
        # Rebuilding the target makes the engine more forgiving if a caller only
        # persists entry and stop but forgets to persist the target.
        explicit_target = float(position.target_underlying)
        if explicit_target > 0.0:
            return explicit_target

        entry = float(position.entry_underlying)
        stop = float(position.stop_underlying)
        direction = self._normalize_direction(position.direction)
        if direction == "LONG":
            risk = entry - stop
            return entry + self.config.target_r_multiple * risk
        if direction == "SHORT":
            risk = stop - entry
            return entry - self.config.target_r_multiple * risk
        return 0.0

    def _evaluate_exit(
        self,
        position: ProfitShooterPositionContext,
        current_candle: pd.Series,
    ) -> ProfitShooterDecision:
        """
        Evaluate exits for an already-open position.

        Exit rules:
        1. Stop loss is active for the full trade life.
        2. Before 1.5R, only stop or target activation matter.
        3. If stop and target are both touched before trailing activates,
           assume stop was hit first.
        4. After trailing activates, a completed close beyond EMA9 arms an exit
           at the next candle open.
        """
        # Read the current candle into simple float values so the logic below
        # stays easy to follow.
        direction = self._normalize_direction(position.direction)
        open_px = float(current_candle["open"])
        high = float(current_candle["high"])
        low = float(current_candle["low"])
        close = float(current_candle["close"])
        ema9 = float(current_candle["ema9"])
        stop = float(position.stop_underlying)
        target = self._resolve_target(position)
        trailing_active = bool(position.trailing_active)
        pending_trailing_exit = bool(position.pending_trailing_exit)

        # If a previous candle already armed a trailing exit, today's job is
        # simple: exit immediately at the current candle open.
        if pending_trailing_exit:
            return ProfitShooterDecision(
                action="EXIT",
                exit_underlying=open_px,
                exit_reason="EMA9_TRAIL_EXIT",
                target_underlying=target,
                trailing_active=True,
                pending_trailing_exit=False,
            )

        if direction == "LONG":
            # Stop loss is always checked first. This also automatically honors
            # the user's conservative rule: if stop and target both happen
            # inside the same candle before trailing activation, stop wins.
            stop_hit = low <= stop
            if stop_hit:
                return ProfitShooterDecision(
                    action="EXIT",
                    exit_underlying=stop,
                    exit_reason="STOP",
                    target_underlying=target,
                    trailing_active=trailing_active,
                    pending_trailing_exit=False,
                )

            if not trailing_active:
                # Before 1.5R is reached, only two things matter:
                # 1. stop loss
                # 2. initial target hit
                target_hit = high >= target
                if target_hit:
                    # Once target is touched, we do NOT exit immediately.
                    # Instead, the strategy switches into EMA9 trailing mode.
                    # If this same completed candle already closes below EMA9,
                    # we arm a next-open trailing exit right away.
                    return self._hold(
                        trailing_active=True,
                        pending_trailing_exit=close < ema9,
                        target_underlying=target,
                    )
                return self._hold(
                    trailing_active=False,
                    pending_trailing_exit=False,
                    target_underlying=target,
                )

            # Trailing mode is already active here.
            # A completed close below EMA9 means "exit next candle open".
            if close < ema9:
                return self._hold(
                    trailing_active=True,
                    pending_trailing_exit=True,
                    target_underlying=target,
                )
            return self._hold(
                trailing_active=True,
                pending_trailing_exit=False,
                target_underlying=target,
            )

        if direction == "SHORT":
            # Mirror image of the long logic.
            stop_hit = high >= stop
            if stop_hit:
                return ProfitShooterDecision(
                    action="EXIT",
                    exit_underlying=stop,
                    exit_reason="STOP",
                    target_underlying=target,
                    trailing_active=trailing_active,
                    pending_trailing_exit=False,
                )

            if not trailing_active:
                target_hit = low <= target
                if target_hit:
                    # After 1.5R is touched on a short, trailing mode activates.
                    # If the same completed candle already closes above EMA9,
                    # the next candle open will be used as the exit.
                    return self._hold(
                        trailing_active=True,
                        pending_trailing_exit=close > ema9,
                        target_underlying=target,
                    )
                return self._hold(
                    trailing_active=False,
                    pending_trailing_exit=False,
                    target_underlying=target,
                )

            # In trailing mode for a short, a close above EMA9 arms the exit.
            if close > ema9:
                return self._hold(
                    trailing_active=True,
                    pending_trailing_exit=True,
                    target_underlying=target,
                )
            return self._hold(
                trailing_active=True,
                pending_trailing_exit=False,
                target_underlying=target,
            )

        return self._hold(
            trailing_active=trailing_active,
            pending_trailing_exit=pending_trailing_exit,
            target_underlying=target,
        )

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[ProfitShooterPositionContext] = None,
    ) -> ProfitShooterDecision:
        """
        Evaluate the latest completed candle and return one strategy decision.

        Flat-state entry logic:
        - Previous candle must be a full long/short setup.
        - Current candle is the next candle.
        - Current candle must break through the setup candle trigger level.

        In-trade logic:
        - Stop loss remains active throughout the trade.
        - 1.5R activates EMA9 trailing mode.
        - EMA9 trailing exit happens on the next candle open after a completed
          close beyond EMA9.
        """
        # These are the columns the engine expects the builder function to have
        # created already.
        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "ema9",
            "atr",
            "long_setup",
            "short_setup",
            "long_entry_trigger",
            "short_entry_trigger",
            "long_entry_price",
            "short_entry_price",
            "long_stop_from_setup",
            "short_stop_from_setup",
            "long_target_from_setup",
            "short_target_from_setup",
        ]
        if candles_with_indicators is None:
            # If there is no data, the safest action is to do nothing while
            # preserving whatever trailing state the caller already has.
            return self._hold(
                trailing_active=bool(position.trailing_active) if position else False,
                pending_trailing_exit=bool(position.pending_trailing_exit) if position else False,
            )
        _require_columns(candles_with_indicators, required_columns)

        # Warm-up guard:
        # We avoid trading until all major indicators and lookback checks have
        # enough history to be trustworthy.
        if len(candles_with_indicators) < self._minimum_history_bars():
            return self._hold(
                trailing_active=bool(position.trailing_active) if position else False,
                pending_trailing_exit=bool(position.pending_trailing_exit) if position else False,
            )

        current = candles_with_indicators.iloc[-1]

        # Even if the DataFrame is long enough overall, the latest row could
        # still contain NaN values. We refuse to trade on incomplete data.
        needed_values = pd.Series(
            [
                current["open"],
                current["high"],
                current["low"],
                current["close"],
                current["ema9"],
            ]
        )
        if needed_values.isna().any():
            return self._hold(
                trailing_active=bool(position.trailing_active) if position else False,
                pending_trailing_exit=bool(position.pending_trailing_exit) if position else False,
            )

        if position is not None:
            # If a trade is already open, entry logic is ignored completely.
            # The only question is whether we should keep holding or exit now.
            return self._evaluate_exit(position, current)

        if bool(current["long_entry_trigger"]):
            # Long entry always happens on the breakout candle, not the setup
            # candle itself. Entry/stop/target values were precomputed by the
            # builder using the previous setup candle.
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target) and stop < entry < target:
                return ProfitShooterDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    trailing_active=False,
                    pending_trailing_exit=False,
                )
            # A signal occurred, but its derived prices were invalid, so we
            # mark `signal_triggered=True` while still refusing the trade.
            return self._hold(signal_triggered=True)

        if bool(current["short_entry_trigger"]):
            # Short logic is the exact mirror of the long path.
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target) and target < entry < stop:
                return ProfitShooterDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    trailing_active=False,
                    pending_trailing_exit=False,
                )
            return self._hold(signal_triggered=True)

        # No setup, no trigger, no exit => do nothing.
        return self._hold()
