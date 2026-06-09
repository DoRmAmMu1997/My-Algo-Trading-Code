"""
NIFTY Stochastic Oscillator Signal Generator.

Stochastic %K/%D crossovers inside the oversold/overbought zones, filtered by a
longer trend SMA. Ported from the public TradingBot reference repo.

The idea (plain English)
------------------------
The stochastic oscillator (see ``stochastic`` in misc_strategy_common) asks where
the close sits inside the recent high-low range. Two lines, %K (fast) and %D
(slow), wiggle between 0 and 100. The trade trigger is a %K-crosses-%D event while
the oscillator is still down in the oversold zone (for longs) or up in the
overbought zone (for shorts) - i.e. momentum is just turning back up/down from an
extreme. A long trend SMA keeps us trading with the broader trend only.

Signals
-------
- ENTER_LONG  when %K crosses ABOVE %D while near the oversold zone AND price is
  above the trend SMA.
- ENTER_SHORT the mirror: %K crosses BELOW %D near the overbought zone, below the
  trend SMA.
- EXIT        on a fixed % stop or % target.

Using this file
---------------
- build_stochastic_oscillator_with_indicators(ohlc): adds %K/%D/SMA + setup columns.
- StochasticOscillatorSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_stochastic_oscillator_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from misc_strategy_common import finite, normalize_ohlc_frame, require_columns, sma, stochastic


@dataclass(frozen=True)
class StochasticOscillatorConfig:
    """Tunable settings for the stochastic oscillator strategy."""

    k_period: int = 14            # lookback for the raw %K range calculation
    d_period: int = 3             # smoothing for the %D (signal) line
    smooth_k: int = 3             # smoothing applied to %K to make the "slow %K"
    oversold: float = 20.0        # classic oversold threshold
    overbought: float = 80.0      # classic overbought threshold
    zone_buffer: float = 10.0     # how far past the threshold still counts as "near" the zone
    trend_filter_period: int = 50 # long SMA: only trade in the direction of this trend
    stop_loss_pct: float = 0.015  # protective stop, 1.5% from entry
    target_pct: float = 0.03      # profit target, 3% from entry

    def __post_init__(self) -> None:
        positive_ints = {
            "k_period": self.k_period,
            "d_period": self.d_period,
            "smooth_k": self.smooth_k,
            "trend_filter_period": self.trend_filter_period,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if not (0.0 < float(self.oversold) < float(self.overbought) < 100.0):
            raise ValueError("Require 0 < oversold < overbought < 100.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class StochasticOscillatorPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class StochasticOscillatorDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_stochastic_oscillator_with_indicators(
    ohlc: pd.DataFrame,
    config: Optional[StochasticOscillatorConfig] = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with %K/%D, trend SMA, crossover flags, and risk levels."""
    config = config or StochasticOscillatorConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    stoch_k, stoch_d = stochastic(frame, config.k_period, config.d_period, config.smooth_k)
    frame["stoch_k"] = stoch_k
    frame["stoch_d"] = stoch_d
    frame["sma_trend"] = sma(close, config.trend_filter_period)

    prev_k = frame["stoch_k"].shift(1)
    prev_d = frame["stoch_d"].shift(1)
    cross_up = (prev_k <= prev_d) & (frame["stoch_k"] > frame["stoch_d"])
    cross_down = (prev_k >= prev_d) & (frame["stoch_k"] < frame["stoch_d"])

    in_oversold = frame["stoch_k"] < (config.oversold + config.zone_buffer)
    in_overbought = frame["stoch_k"] > (config.overbought - config.zone_buffer)
    uptrend = close > frame["sma_trend"]
    downtrend = close < frame["sma_trend"]

    # All three must agree on the same candle: the %K/%D cross, being inside the
    # oversold/overbought zone, and the longer-term trend direction.
    frame["long_setup"] = (cross_up & in_oversold & uptrend).fillna(False)
    frame["short_setup"] = (cross_down & in_overbought & downtrend).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class StochasticOscillatorSignalEngine:
    """Decision engine for stochastic oscillator entries and exits."""

    def __init__(self, config: Optional[StochasticOscillatorConfig] = None) -> None:
        self.config = config or StochasticOscillatorConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.trend_filter_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> StochasticOscillatorDecision:
        return StochasticOscillatorDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: StochasticOscillatorPositionContext) -> StochasticOscillatorDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return StochasticOscillatorDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return StochasticOscillatorDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return StochasticOscillatorDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return StochasticOscillatorDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[StochasticOscillatorPositionContext] = None,
    ) -> StochasticOscillatorDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both stochastic setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return StochasticOscillatorDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="STOCH_BULLISH_CROSS",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return StochasticOscillatorDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="STOCH_BEARISH_CROSS",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No stochastic setup on latest candle.")


class StochasticOscillatorSignalGenerator:
    """Convenience wrapper for full-history and latest-candle stochastic signals."""

    def __init__(self, config: Optional[StochasticOscillatorConfig] = None) -> None:
        self.config = config or StochasticOscillatorConfig()
        self.engine = StochasticOscillatorSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_stochastic_oscillator_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: Optional[StochasticOscillatorPositionContext] = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = StochasticOscillatorPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = StochasticOscillatorPositionContext(
                    "SHORT", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
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
        position: Optional[StochasticOscillatorPositionContext] = None,
    ) -> StochasticOscillatorDecision:
        frame = build_stochastic_oscillator_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_stochastic_oscillator_signals(
    data: pd.DataFrame,
    config: Optional[StochasticOscillatorConfig] = None,
) -> pd.DataFrame:
    return StochasticOscillatorSignalGenerator(config=config).generate(data)


def get_latest_stochastic_oscillator_signal(
    data: pd.DataFrame,
    config: Optional[StochasticOscillatorConfig] = None,
    position: Optional[StochasticOscillatorPositionContext] = None,
) -> StochasticOscillatorDecision:
    return StochasticOscillatorSignalGenerator(config=config).latest_signal(data, position=position)


NiftyStochasticOscillatorSignalGenerator = StochasticOscillatorSignalGenerator
generate_nifty_stochastic_oscillator_signals = generate_stochastic_oscillator_signals
get_latest_nifty_stochastic_oscillator_signal = get_latest_stochastic_oscillator_signal


__all__ = [
    "StochasticOscillatorConfig",
    "StochasticOscillatorPositionContext",
    "StochasticOscillatorDecision",
    "build_stochastic_oscillator_with_indicators",
    "StochasticOscillatorSignalEngine",
    "StochasticOscillatorSignalGenerator",
    "generate_stochastic_oscillator_signals",
    "get_latest_stochastic_oscillator_signal",
    "NiftyStochasticOscillatorSignalGenerator",
    "generate_nifty_stochastic_oscillator_signals",
    "get_latest_nifty_stochastic_oscillator_signal",
]
