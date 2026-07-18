"""
Shared Goldmine strategy logic for already-prepared 5-minute OHLC candles.

What this module does:
1. Calculates SMA20, SMA200, and ATR with the pinned TA-Lib build.
2. Finds the Goldmine pullback + engulfing setup on completed candles.
3. Returns a simple decision object that front-tests and backtests can share.

Important:
- This file does not resample. Pass 5-minute OHLC candles from your front-test
  or data-preparation layer.
- A signal is emitted on the completed setup candle, but `debug["entry_timing"]`
  is `NEXT_OPEN` because the planned execution is the next candle open/market.
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
class GoldmineStrategyConfig:
    """
    All tunable Goldmine settings live in one beginner-friendly object.

    A config dataclass makes the strategy easier to experiment with because the
    user can change periods, thresholds, and target behavior in one place
    without hunting through the signal engine.
    """

    sma_fast_period: int = 20
    sma_slow_period: int = 200
    atr_period: int = 14
    slope_lookback: int = 3
    pullback_lookback: int = 3
    pullback_min_count: int = 2
    near_sma_atr_multiple: float = 0.5
    engulf_tolerance: float = 0.05
    target_atr_multiple: float = 2.0
    max_bars_in_trade: int = 6

    def __post_init__(self) -> None:
        """
        Validate the config immediately after dataclass construction.

        Beginner note:
        - Indicator periods and lookbacks must be positive integers.
        - Ratio/multiple values must be non-negative where they represent a
          tolerance, or strictly positive where they represent a target.
        - Catching mistakes here gives the caller a clear error before any
          candle calculations start.
        """
        validate_finite_config(self)
        positive_ints = {
            "sma_fast_period": self.sma_fast_period,
            "sma_slow_period": self.sma_slow_period,
            "atr_period": self.atr_period,
            "slope_lookback": self.slope_lookback,
            "pullback_lookback": self.pullback_lookback,
            "pullback_min_count": self.pullback_min_count,
            "max_bars_in_trade": self.max_bars_in_trade,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if int(self.pullback_min_count) > int(self.pullback_lookback):
            raise ValueError("pullback_min_count cannot be larger than pullback_lookback.")
        if int(self.sma_fast_period) >= int(self.sma_slow_period):
            raise ValueError("sma_fast_period must be smaller than sma_slow_period.")
        if float(self.near_sma_atr_multiple) < 0.0:
            raise ValueError("near_sma_atr_multiple must be non-negative.")
        if float(self.engulf_tolerance) < 0.0:
            raise ValueError("engulf_tolerance must be non-negative.")
        if float(self.engulf_tolerance) > 1.0:
            raise ValueError("engulf_tolerance must not exceed 1.0.")
        if float(self.target_atr_multiple) <= 0.0:
            raise ValueError("target_atr_multiple must be greater than zero.")


@dataclass
class GoldminePositionContext:
    """
    Small snapshot of an already-open Goldmine trade.

    The engine receives this object when a trade is already live. It then skips
    entry logic and checks only exit rules, which prevents accidental duplicate
    entries while a position is open.
    """

    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float
    bars_in_trade: int = 0


@dataclass
class GoldmineDecision:
    """
    Standard decision returned by the Goldmine signal engine.

    `action` is intentionally simple:
    - HOLD means do nothing.
    - ENTER_LONG / ENTER_SHORT means a setup candle just completed.
    - EXIT means an existing position should be closed.

    Numeric fields are left at 0.0 when they do not apply to the action.
    """

    action: str = "HOLD"
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    target_underlying: float = 0.0
    exit_underlying: float = 0.0
    exit_reason: str = ""
    signal_triggered: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def _recent_count(mask: pd.Series, lookback: int, min_count: int) -> pd.Series:
    """
    Count recent pullback candles before the current signal candle.

    The final `.shift(1)` matters: the current engulfing candle should not be
    counted as part of the pullback that came before it.
    """
    return mask.rolling(window=int(lookback), min_periods=int(lookback)).sum().shift(1) >= int(min_count)


def _build_bullish_engulfing(frame: pd.DataFrame, tolerance: float) -> pd.Series:
    """Bullish engulfing body pattern from the extracted Goldmine rules."""
    # The previous candle must show selling pressure, and the current candle
    # must reverse that pressure with a green body.
    previous_red = frame["close"].shift(1) < frame["open"].shift(1)
    current_green = frame["close"] > frame["open"]
    # We compare candle bodies, not full high/low ranges, because the video
    # definition talks about the real body engulfing the previous real body.
    body_engulfs = (
        frame["open"] <= frame["close"].shift(1) + float(tolerance)
    ) & (
        frame["close"] >= frame["open"].shift(1) - float(tolerance)
    )
    return previous_red & current_green & body_engulfs


def _build_bearish_engulfing(frame: pd.DataFrame, tolerance: float) -> pd.Series:
    """Bearish engulfing body pattern from the extracted Goldmine rules."""
    # This is the exact mirror image of the bullish engulfing check:
    # prior candle green, current candle red, and current real body wraps the
    # previous real body.
    previous_green = frame["close"].shift(1) > frame["open"].shift(1)
    current_red = frame["close"] < frame["open"]
    body_engulfs = (
        frame["open"] >= frame["close"].shift(1) - float(tolerance)
    ) & (
        frame["close"] <= frame["open"].shift(1) + float(tolerance)
    )
    return previous_green & current_red & body_engulfs


def _price_near_sma20(frame: pd.DataFrame, config: GoldmineStrategyConfig) -> pd.Series:
    """
    Convert "price is close to SMA20" into repeatable code.

    A candle is accepted when either:
    - its range touches SMA20, or
    - its close is within the configured ATR distance.
    """
    candle_touches_sma = (frame["low"] <= frame["sma20"]) & (frame["high"] >= frame["sma20"])
    close_near_sma = (
        (frame["atr"] > 0.0)
        & ((frame["close"] - frame["sma20"]).abs() <= float(config.near_sma_atr_multiple) * frame["atr"])
    )
    return candle_touches_sma | close_near_sma


def build_goldmine_with_indicators(
    ohlc: pd.DataFrame,
    config: GoldmineStrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Return OHLC candles enriched with Goldmine indicators and setup columns.

    The returned DataFrame is designed for two uses:
    1. A signal engine can read the newest row and make one decision.
    2. A tester can inspect intermediate columns like `near_sma20` or
       `bullish_engulfing` to understand why a signal did or did not happen.
    """
    config = config or GoldmineStrategyConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    # Indicator stage:
    # SMA20/SMA200 describe trend and trading zone; ATR gives the volatility
    # scale for "near SMA20" and the 2 x ATR target.
    close = frame["close"].astype(float)
    frame["sma20"] = sma(close, config.sma_fast_period)
    frame["sma200"] = sma(close, config.sma_slow_period)
    frame["atr"] = atr(frame, config.atr_period)
    frame = add_candle_anatomy(frame)

    # Trend-zone stage:
    # The strategy wants the 20-SMA moving in the trade direction and price
    # close enough to that SMA to count as a pullback entry zone.
    frame["sma20_rising"] = rising_over_lookback(frame["sma20"], config.slope_lookback)
    frame["sma20_falling"] = falling_over_lookback(frame["sma20"], config.slope_lookback)
    frame["near_sma20"] = _price_near_sma20(frame, config)

    # Pullback stage:
    # The signal candle itself is an engulfing reversal, so the pullback count
    # intentionally looks at the candles before the signal candle.
    down_close = frame["close"] < frame["close"].shift(1)
    up_close = frame["close"] > frame["close"].shift(1)
    frame["recent_long_pullback"] = _recent_count(
        down_close,
        config.pullback_lookback,
        config.pullback_min_count,
    )
    frame["recent_short_pullback"] = _recent_count(
        up_close,
        config.pullback_lookback,
        config.pullback_min_count,
    )

    frame["bullish_engulfing"] = _build_bullish_engulfing(frame, config.engulf_tolerance)
    frame["bearish_engulfing"] = _build_bearish_engulfing(frame, config.engulf_tolerance)

    # Final setup stage:
    # Every condition must be true on the same completed candle before the
    # engine is allowed to emit an entry decision.
    frame["long_setup"] = (
        (frame["close"] > frame["sma200"])
        & (frame["sma20"] > frame["sma200"])
        & frame["sma20_rising"]
        & frame["near_sma20"]
        & frame["recent_long_pullback"]
        & frame["bullish_engulfing"]
    )
    frame["short_setup"] = (
        (frame["close"] < frame["sma200"])
        & (frame["sma20"] < frame["sma200"])
        & frame["sma20_falling"]
        & frame["near_sma20"]
        & frame["recent_short_pullback"]
        & frame["bearish_engulfing"]
    )

    # Risk stage:
    # Goldmine uses the full two-candle engulfing pattern for stop placement,
    # which is a little more conservative than using only the signal candle.
    pattern_low = pd.concat([frame["low"], frame["low"].shift(1)], axis=1).min(axis=1)
    pattern_high = pd.concat([frame["high"], frame["high"].shift(1)], axis=1).max(axis=1)
    frame["long_entry_price"] = frame["close"]
    frame["short_entry_price"] = frame["close"]
    frame["long_stop_from_setup"] = pattern_low
    frame["short_stop_from_setup"] = pattern_high
    frame["long_target_from_setup"] = frame["long_entry_price"] + float(config.target_atr_multiple) * frame["atr"]
    frame["short_target_from_setup"] = frame["short_entry_price"] - float(config.target_atr_multiple) * frame["atr"]
    frame["signal_atr"] = frame["atr"]
    return frame


class GoldmineSignalEngine:
    """Decision engine for Goldmine entries and exits."""

    def __init__(self, config: GoldmineStrategyConfig | None = None) -> None:
        """
        Store one config object for all later evaluations.

        Keeping the config on the engine prevents subtle drift where a caller
        builds indicators with one set of values but evaluates entries with
        another set.
        """
        self.config = config or GoldmineStrategyConfig()

    def minimum_history_bars(self) -> int:
        """Return the warm-up length needed before entries are trusted."""
        return (
            max(self.config.sma_fast_period, self.config.sma_slow_period, self.config.atr_period)
            + max(self.config.slope_lookback, self.config.pullback_lookback)
            + 1
        )

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> GoldmineDecision:
        """Return a consistent HOLD decision."""
        return GoldmineDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        """Normalize direction strings like 'long' or ' LONG '."""
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: GoldminePositionContext) -> GoldmineDecision:
        """Evaluate stop, target, and time exit for an open Goldmine trade."""
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            # Conservative ordering: if a candle touches both stop and target,
            # the stop branch wins because we cannot know intrabar sequence from
            # OHLC alone.
            if low <= stop:
                return GoldmineDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return GoldmineDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if int(position.bars_in_trade) >= int(self.config.max_bars_in_trade):
                return GoldmineDecision(action="EXIT", exit_underlying=close, exit_reason="TIME_EXIT")
            return self._hold()

        if direction == "SHORT":
            # Short exits mirror the long exits: stop is above price, target is
            # below price, and the same stop-first rule is used.
            if high >= stop:
                return GoldmineDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return GoldmineDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if int(position.bars_in_trade) >= int(self.config.max_bars_in_trade):
                return GoldmineDecision(action="EXIT", exit_underlying=close, exit_reason="TIME_EXIT")
            return self._hold()

        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: GoldminePositionContext | None = None,
    ) -> GoldmineDecision:
        """Evaluate the newest candle and return HOLD, ENTER_LONG, ENTER_SHORT, or EXIT."""
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            # Position supplied means the caller is already in a trade. Entry
            # setups are ignored until that trade exits.
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
            return self._hold("Conflict: both long and short Goldmine setups are true.")

        if bool(current["long_setup"]):
            # Entry price is the setup close for the signal payload. The
            # backtest/front-test uses debug["entry_timing"] to execute on the
            # next candle open instead of pretending the setup candle was tradable.
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return GoldmineDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    debug={
                        "entry_timing": "NEXT_OPEN",
                        "timestamp": current.get("timestamp"),
                        "pattern": "BULLISH_ENGULFING",
                        "signal_atr": float(current.get("signal_atr", np.nan)),
                    },
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if bool(current["short_setup"]):
            # The short branch is deliberately shaped like the long branch so a
            # beginner can compare them side by side.
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return GoldmineDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    signal_triggered=True,
                    debug={
                        "entry_timing": "NEXT_OPEN",
                        "timestamp": current.get("timestamp"),
                        "pattern": "BEARISH_ENGULFING",
                        "signal_atr": float(current.get("signal_atr", np.nan)),
                    },
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No Goldmine setup on latest candle.")


class GoldmineSignalGenerator:
    """Convenience wrapper for full-history and latest-candle Goldmine signals."""

    def __init__(self, config: GoldmineStrategyConfig | None = None) -> None:
        """
        Create one reusable signal engine for this generator instance.

        Full-history generation walks candle by candle, so reusing the engine
        keeps the public API simple and mirrors the other strategy folders.
        """
        self.config = config or GoldmineStrategyConfig()
        self.engine = GoldmineSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return enriched candles plus repo-style signal stream columns."""
        frame = build_goldmine_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: GoldminePositionContext | None = None

        for index in range(len(frame)):
            # The generator simulates a very simple single-position stream:
            # once a position is open, later candles are passed as position
            # management checks until an EXIT resets the state.
            if position is not None:
                position.bars_in_trade += 1
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = GoldminePositionContext(
                    direction="LONG",
                    entry_underlying=decision.entry_underlying,
                    stop_underlying=decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = GoldminePositionContext(
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
        position: GoldminePositionContext | None = None,
    ) -> GoldmineDecision:
        """Return only the newest Goldmine decision."""
        frame = build_goldmine_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_goldmine_signals(
    data: pd.DataFrame,
    config: GoldmineStrategyConfig | None = None,
) -> pd.DataFrame:
    """Functional wrapper for full-history Goldmine signals."""
    return GoldmineSignalGenerator(config=config).generate(data)


def get_latest_goldmine_signal(
    data: pd.DataFrame,
    config: GoldmineStrategyConfig | None = None,
    position: GoldminePositionContext | None = None,
) -> GoldmineDecision:
    """Functional wrapper for the latest Goldmine decision."""
    return GoldmineSignalGenerator(config=config).latest_signal(data, position=position)


__all__ = [
    "GoldmineDecision",
    "GoldminePositionContext",
    "GoldmineSignalEngine",
    "GoldmineSignalGenerator",
    "GoldmineStrategyConfig",
    "build_goldmine_with_indicators",
    "generate_goldmine_signals",
    "get_latest_goldmine_signal",
]
