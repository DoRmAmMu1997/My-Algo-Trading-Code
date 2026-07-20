"""
Shared Money Machine strategy logic for already-prepared 5-minute OHLC candles.

Money Machine uses the same SMA20/SMA200 trend context as Goldmine, but its
trigger is different:
1. Price compresses in a narrow range for several candles.
2. A strong Marubozu/Hulk candle breaks out of that compression.
3. Execution is planned for the next candle open/market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from subhamoy_strategy_common import (
    add_candle_anatomy,
    atr,
    falling_over_lookback,
    finite,
    normalize_ohlc_frame,
    require_columns,
    rising_over_lookback,
    sma,
    validate_finite_config,
)


@dataclass(frozen=True)
class MoneyMachineStrategyConfig:
    """
    All tunable Money Machine settings live in one beginner-friendly object.

    The video leaves several phrases qualitative, such as "narrow range" and
    "very small wick." These fields convert those chart-reading phrases into
    explicit numbers that can be tested and tuned.
    """

    sma_fast_period: int = 20
    sma_slow_period: int = 200
    atr_period: int = 14
    slope_lookback: int = 3
    near_sma_atr_multiple: float = 0.5
    compression_window: int = 3
    compression_range_atr_multiple: float = 0.75
    marubozu_body_min_ratio: float = 0.80
    marubozu_wick_max_ratio: float = 0.15
    require_breakout_close: bool = True
    target_atr_multiple: float = 2.0

    def __post_init__(self) -> None:
        """
        Validate all Money Machine parameters before calculations begin.

        Beginner note:
        - Period/window fields must be positive because rolling indicators need
          at least one candle.
        - Ratio fields must stay between 0 and 1 because they represent a
          fraction of candle range.
        - Target multiple must be positive because target distance cannot be
          zero or negative.
        """
        validate_finite_config(self)
        positive_ints = {
            "sma_fast_period": self.sma_fast_period,
            "sma_slow_period": self.sma_slow_period,
            "atr_period": self.atr_period,
            "slope_lookback": self.slope_lookback,
            "compression_window": self.compression_window,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if int(self.sma_fast_period) >= int(self.sma_slow_period):
            raise ValueError("sma_fast_period must be smaller than sma_slow_period.")
        bounded_ratios = {
            "marubozu_body_min_ratio": self.marubozu_body_min_ratio,
            "marubozu_wick_max_ratio": self.marubozu_wick_max_ratio,
        }
        invalid_ratios = [
            name for name, value in bounded_ratios.items() if not (0.0 <= float(value) <= 1.0)
        ]
        if invalid_ratios:
            raise ValueError(f"Ratios must be between 0 and 1: {', '.join(invalid_ratios)}")
        if float(self.near_sma_atr_multiple) < 0.0:
            raise ValueError("near_sma_atr_multiple must be non-negative.")
        if float(self.compression_range_atr_multiple) < 0.0:
            raise ValueError("compression_range_atr_multiple must be non-negative.")
        if float(self.target_atr_multiple) <= 0.0:
            raise ValueError("target_atr_multiple must be greater than zero.")


@dataclass
class MoneyMachinePositionContext:
    """
    Small snapshot of an already-open Money Machine trade.

    Passing this object into the engine tells it: "we are managing a trade now,
    so ignore fresh entries and only check stop/target exits."
    """

    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class MoneyMachineDecision:
    """
    Standard decision returned by the Money Machine signal engine.

    The same decision shape is used for entries, exits, and holds so callers do
    not have to switch between different return types.
    """

    action: str = "HOLD"
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    target_underlying: float = 0.0
    exit_underlying: float = 0.0
    exit_reason: str = ""
    signal_triggered: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def _price_near_sma20(frame: pd.DataFrame, config: MoneyMachineStrategyConfig) -> pd.Series:
    """Accept candles that touch SMA20 or close within an ATR-based buffer."""
    candle_touches_sma = (frame["low"] <= frame["sma20"]) & (frame["high"] >= frame["sma20"])
    close_near_sma = (
        (frame["atr"] > 0.0)
        & ((frame["close"] - frame["sma20"]).abs() <= float(config.near_sma_atr_multiple) * frame["atr"])
    )
    return candle_touches_sma | close_near_sma


def _build_hulk_masks(frame: pd.DataFrame, config: MoneyMachineStrategyConfig) -> tuple[pd.Series, pd.Series]:
    """Return bullish and bearish Marubozu/Hulk masks."""
    # Replace zero ranges with NaN so a flat/bad candle cannot accidentally pass
    # the ratio tests through division by zero.
    range_safe = frame["candle_range"].replace(0.0, np.nan)
    body_ratio = frame["body"] / range_safe
    upper_ratio = frame["upper_wick"] / range_safe
    lower_ratio = frame["lower_wick"] / range_safe

    # A Hulk candle needs two things at once:
    # 1. A large real body.
    # 2. Tiny upper and lower wicks.
    strong_body = body_ratio >= float(config.marubozu_body_min_ratio)
    tiny_wicks = (
        upper_ratio <= float(config.marubozu_wick_max_ratio)
    ) & (
        lower_ratio <= float(config.marubozu_wick_max_ratio)
    )
    bullish = (frame["close"] > frame["open"]) & strong_body & tiny_wicks
    bearish = (frame["close"] < frame["open"]) & strong_body & tiny_wicks
    return bullish, bearish


def build_money_machine_with_indicators(
    ohlc: pd.DataFrame,
    config: MoneyMachineStrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Return OHLC candles enriched with Money Machine indicators and setup columns.

    The intermediate columns are intentionally verbose. A user can open the
    DataFrame and see exactly which rule failed: trend, near-SMA, compression,
    Hulk geometry, or breakout.
    """
    config = config or MoneyMachineStrategyConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    # Indicator stage:
    # SMA20/SMA200 define the trend and trading zone. ATR gives us a volatility
    # scale for both "near SMA20" and "previous range is narrow."
    close = frame["close"].astype(float)
    frame["sma20"] = sma(close, config.sma_fast_period)
    frame["sma200"] = sma(close, config.sma_slow_period)
    frame["atr"] = atr(frame, config.atr_period)
    frame = add_candle_anatomy(frame)

    frame["sma20_rising"] = rising_over_lookback(frame["sma20"], config.slope_lookback)
    frame["sma20_falling"] = falling_over_lookback(frame["sma20"], config.slope_lookback)
    frame["near_sma20"] = _price_near_sma20(frame, config)
    # Start with explicit False columns so the output has predictable columns
    # even if a future edit changes the helper implementation.
    frame["bullish_hulk"] = False
    frame["bearish_hulk"] = False
    frame["bullish_hulk"], frame["bearish_hulk"] = _build_hulk_masks(frame, config)

    # Compression stage:
    # The rolling range is built from the candles BEFORE the signal candle. The
    # signal candle itself should be the breakout from that range, not part of it.
    previous_high = frame["high"].shift(1).rolling(
        window=config.compression_window,
        min_periods=config.compression_window,
    ).max()
    previous_low = frame["low"].shift(1).rolling(
        window=config.compression_window,
        min_periods=config.compression_window,
    ).min()
    previous_range = previous_high - previous_low

    # The compression range is compared to ATR available on the signal candle.
    # This keeps the rule simple and causal: no future data is used.
    frame["compression_high"] = previous_high
    frame["compression_low"] = previous_low
    frame["compression_range"] = previous_range
    frame["compression_ok"] = (
        (frame["atr"] > 0.0)
        & (previous_range <= float(config.compression_range_atr_multiple) * frame["atr"])
    )

    if config.require_breakout_close:
        # Close-based breakout is stricter and avoids signals where price poked
        # outside the range intrabar but closed back inside.
        frame["bullish_compression_breakout"] = frame["close"] > previous_high
        frame["bearish_compression_breakout"] = frame["close"] < previous_low
    else:
        # High/low breakout is more permissive and is available for experiments.
        frame["bullish_compression_breakout"] = frame["high"] > previous_high
        frame["bearish_compression_breakout"] = frame["low"] < previous_low

    # Final setup stage:
    # Money Machine requires trend, trading-zone location, compression, Hulk
    # candle geometry, and breakout direction to agree on the same candle.
    frame["long_setup"] = (
        (frame["close"] > frame["sma200"])
        & (frame["sma20"] > frame["sma200"])
        & frame["sma20_rising"]
        & frame["near_sma20"]
        & frame["compression_ok"]
        & frame["bullish_hulk"]
        & frame["bullish_compression_breakout"]
    )
    frame["short_setup"] = (
        (frame["close"] < frame["sma200"])
        & (frame["sma20"] < frame["sma200"])
        & frame["sma20_falling"]
        & frame["near_sma20"]
        & frame["compression_ok"]
        & frame["bearish_hulk"]
        & frame["bearish_compression_breakout"]
    )

    # Risk stage:
    # The video places the stop at the Hulk candle extreme and the target at
    # 2 x ATR from entry. The backtest later recomputes target from actual next
    # open, but these values are still useful for front-test signal payloads.
    frame["long_entry_price"] = frame["close"]
    frame["short_entry_price"] = frame["close"]
    frame["long_stop_from_setup"] = frame["low"]
    frame["short_stop_from_setup"] = frame["high"]
    frame["long_target_from_setup"] = frame["long_entry_price"] + float(config.target_atr_multiple) * frame["atr"]
    frame["short_target_from_setup"] = frame["short_entry_price"] - float(config.target_atr_multiple) * frame["atr"]
    frame["signal_atr"] = frame["atr"]
    return frame


class MoneyMachineSignalEngine:
    """Decision engine for Money Machine entries and exits."""

    def __init__(self, config: MoneyMachineStrategyConfig | None = None) -> None:
        """
        Store one config object for all later evaluations.

        This avoids mixing indicator settings and signal settings when a caller
        evaluates many candles over time.
        """
        self.config = config or MoneyMachineStrategyConfig()

    def minimum_history_bars(self) -> int:
        """Return the warm-up length needed before entries are trusted."""
        return (
            max(self.config.sma_fast_period, self.config.sma_slow_period, self.config.atr_period)
            + max(self.config.slope_lookback, self.config.compression_window)
            + 1
        )

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> MoneyMachineDecision:
        """Return a consistent HOLD decision."""
        return MoneyMachineDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        """Normalize direction strings like 'short' or ' SHORT '."""
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: MoneyMachinePositionContext) -> MoneyMachineDecision:
        """Evaluate stop and target for an open Money Machine trade."""
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            # Stop is checked before target so same-candle stop+target touches
            # are handled conservatively from OHLC data.
            if low <= stop:
                return MoneyMachineDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return MoneyMachineDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()

        if direction == "SHORT":
            # Short exit logic is the mirror image of long exit logic.
            if high >= stop:
                return MoneyMachineDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return MoneyMachineDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()

        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: MoneyMachinePositionContext | None = None,
    ) -> MoneyMachineDecision:
        """Evaluate the newest candle and return HOLD, ENTER_LONG, ENTER_SHORT, or EXIT."""
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            # Open position means entry checks are skipped. The engine should
            # only manage the existing trade until it exits.
            return self._evaluate_exit(current, position)

        required = [
            "sma20",
            "sma200",
            "atr",
            "long_setup",
            "short_setup",
            "long_entry_price",
            "short_entry_price",
            "long_stop_from_setup",
            "short_stop_from_setup",
            "long_target_from_setup",
            "short_target_from_setup",
        ]
        require_columns(candles_with_indicators, required)
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        if bool(current["long_setup"]) and bool(current["short_setup"]):
            return self._hold("Conflict: both long and short Money Machine setups are true.")

        if bool(current["long_setup"]):
            # The signal payload uses the completed setup close, while
            # debug["entry_timing"] tells the caller to execute on next open.
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return MoneyMachineDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    debug={
                        "entry_timing": "NEXT_OPEN",
                        "timestamp": current.get("timestamp"),
                        "pattern": "BULLISH_HULK",
                        "signal_atr": float(current.get("signal_atr", np.nan)),
                    },
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if bool(current["short_setup"]):
            # This branch mirrors the long branch for readability and easier
            # future tuning.
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return MoneyMachineDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    debug={
                        "entry_timing": "NEXT_OPEN",
                        "timestamp": current.get("timestamp"),
                        "pattern": "BEARISH_HULK",
                        "signal_atr": float(current.get("signal_atr", np.nan)),
                    },
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No Money Machine setup on latest candle.")


class MoneyMachineSignalGenerator:
    """Convenience wrapper for full-history and latest-candle Money Machine signals."""

    def __init__(self, config: MoneyMachineStrategyConfig | None = None) -> None:
        """
        Create one reusable engine for this generator instance.

        The wrapper is intentionally small: all trading rules stay in the shared
        engine, while this class only adds full-history iteration.
        """
        self.config = config or MoneyMachineStrategyConfig()
        self.engine = MoneyMachineSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return enriched candles plus repo-style signal stream columns."""
        frame = build_money_machine_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: MoneyMachinePositionContext | None = None

        for index in range(len(frame)):
            # Full-history generation keeps one simple simulated position so
            # output streams do not show overlapping entries.
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = MoneyMachinePositionContext(
                    direction="LONG",
                    entry_underlying=decision.entry_underlying,
                    stop_underlying=decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = MoneyMachinePositionContext(
                    direction="SHORT",
                    entry_underlying=decision.entry_underlying,
                    stop_underlying=decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                )
            elif decision.action == "EXIT":
                stream.append(2)
                position = None
            else:
                stream.append(0)

        result = frame.copy()
        result["signalAction"] = actions
        result["entryUnderlying"] = entries
        result["stopUnderlying"] = stops
        result["targetUnderlying"] = targets
        result["backtestStream"] = stream
        return result

    def latest_signal(
        self,
        data: pd.DataFrame,
        position: MoneyMachinePositionContext | None = None,
    ) -> MoneyMachineDecision:
        """Return only the newest Money Machine decision."""
        frame = build_money_machine_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_money_machine_signals(
    data: pd.DataFrame,
    config: MoneyMachineStrategyConfig | None = None,
) -> pd.DataFrame:
    """Functional wrapper for full-history Money Machine signals."""
    return MoneyMachineSignalGenerator(config=config).generate(data)


def get_latest_money_machine_signal(
    data: pd.DataFrame,
    config: MoneyMachineStrategyConfig | None = None,
    position: MoneyMachinePositionContext | None = None,
) -> MoneyMachineDecision:
    """Functional wrapper for the latest Money Machine decision."""
    return MoneyMachineSignalGenerator(config=config).latest_signal(data, position=position)


__all__ = [
    "MoneyMachineDecision",
    "MoneyMachinePositionContext",
    "MoneyMachineSignalEngine",
    "MoneyMachineSignalGenerator",
    "MoneyMachineStrategyConfig",
    "build_money_machine_with_indicators",
    "generate_money_machine_signals",
    "get_latest_money_machine_signal",
]
