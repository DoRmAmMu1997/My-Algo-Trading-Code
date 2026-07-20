"""
NIFTY Multi Timeframe Signal Generator.

Trend-aligned EMA crossover with an RSI filter, ported from the public TradingBot
reference repo.

The idea (plain English)
------------------------
Only take trades that agree with the bigger trend. A long, slow SMA (default 50)
stands in for the "higher timeframe" trend: we only go long when price is above
it, and only short when below it. Within that bias we time entries with a fast/slow
EMA crossover, and we require RSI to be in a healthy band (not already exhausted).
Three filters lining up = a higher-quality, less-noisy trend entry.

Signals
-------
- ENTER_LONG  when price is above the trend SMA AND the fast EMA crosses above the
  slow EMA AND RSI is inside the bullish band (default 40-70).
- ENTER_SHORT the exact mirror: below trend SMA, bearish EMA cross, RSI 30-60.
- EXIT        on a fixed % stop or % target.

Using this file
---------------
- build_multi_timeframe_with_indicators(ohlc): adds trend/EMA/RSI + setup columns.
- MultiTimeframeSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_multi_timeframe_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import ema, finite, normalize_ohlc_frame, require_columns, rsi, sma, validate_finite_config


@dataclass(frozen=True)
class MultiTimeframeConfig:
    """Tunable settings for the multi-timeframe trend strategy."""

    trend_sma_period: int = 50    # long SMA = the "higher timeframe" trend filter
    ema_fast: int = 9             # fast EMA for the entry-timing crossover
    ema_slow: int = 21            # slow EMA for the entry-timing crossover
    rsi_period: int = 14          # RSI lookback for the momentum filter
    rsi_buy_min: float = 40.0     # for longs, RSI must be at least this (not too weak)
    rsi_buy_max: float = 70.0     # ...and at most this (not already overbought)
    rsi_sell_min: float = 30.0    # for shorts, RSI must be at least this (not already oversold)
    rsi_sell_max: float = 60.0    # ...and at most this (not too strong)
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        validate_finite_config(self)
        positive_ints = {
            "trend_sma_period": self.trend_sma_period,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "rsi_period": self.rsi_period,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if int(self.ema_fast) >= int(self.ema_slow):
            raise ValueError("ema_fast must be smaller than ema_slow.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class MultiTimeframePositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class MultiTimeframeDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_multi_timeframe_with_indicators(
    ohlc: pd.DataFrame,
    config: MultiTimeframeConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with trend SMA, EMA crossover flags, RSI, and risk levels."""
    config = config or MultiTimeframeConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    frame["sma_trend"] = sma(close, config.trend_sma_period)
    frame["ema_fast"] = ema(close, config.ema_fast)
    frame["ema_slow"] = ema(close, config.ema_slow)
    frame["rsi"] = rsi(close, config.rsi_period)

    prev_fast = frame["ema_fast"].shift(1)
    prev_slow = frame["ema_slow"].shift(1)
    cross_up = (frame["ema_fast"] > frame["ema_slow"]) & (prev_fast <= prev_slow)
    cross_down = (frame["ema_fast"] < frame["ema_slow"]) & (prev_fast >= prev_slow)

    uptrend = close > frame["sma_trend"]
    downtrend = close < frame["sma_trend"]
    rsi_buy_ok = (frame["rsi"] >= config.rsi_buy_min) & (frame["rsi"] <= config.rsi_buy_max)
    rsi_sell_ok = (frame["rsi"] >= config.rsi_sell_min) & (frame["rsi"] <= config.rsi_sell_max)

    # A setup needs all three filters true on the same candle: trend bias (price
    # vs trend SMA) AND the EMA crossover AND RSI inside its allowed band.
    frame["long_setup"] = (uptrend & cross_up & rsi_buy_ok).fillna(False)
    frame["short_setup"] = (downtrend & cross_down & rsi_sell_ok).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class MultiTimeframeSignalEngine:
    """Decision engine for multi-timeframe trend entries and exits."""

    def __init__(self, config: MultiTimeframeConfig | None = None) -> None:
        self.config = config or MultiTimeframeConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.trend_sma_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> MultiTimeframeDecision:
        return MultiTimeframeDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: MultiTimeframePositionContext) -> MultiTimeframeDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return MultiTimeframeDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return MultiTimeframeDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return MultiTimeframeDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return MultiTimeframeDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: MultiTimeframePositionContext | None = None,
    ) -> MultiTimeframeDecision:
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
            return self._hold("Conflict: both trend setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return MultiTimeframeDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="TREND_EMA_CROSS_UP",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return MultiTimeframeDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="TREND_EMA_CROSS_DOWN",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No multi-timeframe setup on latest candle.")


class MultiTimeframeSignalGenerator:
    """Convenience wrapper for full-history and latest-candle multi-timeframe signals."""

    def __init__(self, config: MultiTimeframeConfig | None = None) -> None:
        self.config = config or MultiTimeframeConfig()
        self.engine = MultiTimeframeSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_multi_timeframe_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: MultiTimeframePositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = MultiTimeframePositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = MultiTimeframePositionContext(
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
        position: MultiTimeframePositionContext | None = None,
    ) -> MultiTimeframeDecision:
        frame = build_multi_timeframe_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_multi_timeframe_signals(
    data: pd.DataFrame,
    config: MultiTimeframeConfig | None = None,
) -> pd.DataFrame:
    return MultiTimeframeSignalGenerator(config=config).generate(data)


def get_latest_multi_timeframe_signal(
    data: pd.DataFrame,
    config: MultiTimeframeConfig | None = None,
    position: MultiTimeframePositionContext | None = None,
) -> MultiTimeframeDecision:
    return MultiTimeframeSignalGenerator(config=config).latest_signal(data, position=position)


NiftyMultiTimeframeSignalGenerator = MultiTimeframeSignalGenerator
generate_nifty_multi_timeframe_signals = generate_multi_timeframe_signals
get_latest_nifty_multi_timeframe_signal = get_latest_multi_timeframe_signal


__all__ = [
    "MultiTimeframeConfig",
    "MultiTimeframeDecision",
    "MultiTimeframePositionContext",
    "MultiTimeframeSignalEngine",
    "MultiTimeframeSignalGenerator",
    "NiftyMultiTimeframeSignalGenerator",
    "build_multi_timeframe_with_indicators",
    "generate_multi_timeframe_signals",
    "generate_nifty_multi_timeframe_signals",
    "get_latest_multi_timeframe_signal",
    "get_latest_nifty_multi_timeframe_signal",
]
