"""
NIFTY RSI Reversal Signal Generator.

Contrarian RSI entries ported from the public TradingBot reference repo.

The idea (plain English)
------------------------
RSI is a 0-100 momentum gauge (see ``rsi`` in misc_strategy_common). When RSI
drops below 30 the market is "oversold" (it has fallen hard and may be due for a
bounce), and when it climbs above 70 it is "overbought". This strategy bets on the
snap-back: buy into oversold, sell into overbought. It is a fade / counter-trend
strategy, so it works best in ranging markets and can struggle in strong trends.

Signals
-------
- ENTER_LONG  when RSI crosses DOWN through the oversold line (30).
- ENTER_SHORT when RSI crosses UP through the overbought line (70).
- EXIT        on a fixed % stop or % target, or - if `exit_at_mean` is on - when
  RSI reverts back through the 50 mid-line (momentum has normalised).

Using this file
---------------
- build_rsi_reversal_with_indicators(ohlc): adds RSI + setup + risk columns.
- RSIReversalSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_rsi_reversal_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import finite, normalize_ohlc_frame, require_columns, rsi, validate_finite_config


@dataclass(frozen=True)
class RSIReversalConfig:
    """Tunable settings for the RSI reversal strategy."""

    rsi_period: int = 14       # RSI lookback (candles)
    oversold: float = 30.0     # below this RSI is "too cheap" -> look to buy
    overbought: float = 70.0   # above this RSI is "too hot" -> look to sell
    exit_at_mean: bool = True  # also exit when RSI returns to the 50 mid-line
    mean_level: float = 50.0   # the neutral RSI level used for the mean-reversion exit
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        validate_finite_config(self)
        if int(self.rsi_period) <= 0:
            raise ValueError("rsi_period must be positive.")
        if not (0.0 < float(self.oversold) < float(self.overbought) < 100.0):
            raise ValueError("Require 0 < oversold < overbought < 100.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class RSIReversalPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class RSIReversalDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, RSI_MEAN_REVERSION
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_rsi_reversal_with_indicators(
    ohlc: pd.DataFrame,
    config: RSIReversalConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with RSI, oversold/overbought cross flags, and risk levels."""
    config = config or RSIReversalConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    frame["rsi"] = rsi(close, config.rsi_period)
    prev_rsi = frame["rsi"].shift(1)

    # Trigger only on the candle RSI first ENTERS the zone (crosses the line),
    # not on every candle it stays there - so we trade the fresh signal once.
    frame["long_setup"] = ((frame["rsi"] < config.oversold) & (prev_rsi >= config.oversold)).fillna(False)
    frame["short_setup"] = ((frame["rsi"] > config.overbought) & (prev_rsi <= config.overbought)).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class RSIReversalSignalEngine:
    """Decision engine for RSI reversal entries and exits."""

    def __init__(self, config: RSIReversalConfig | None = None) -> None:
        self.config = config or RSIReversalConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.rsi_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> RSIReversalDecision:
        return RSIReversalDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(
        self,
        current: pd.Series,
        prev: pd.Series | None,
        position: RSIReversalPositionContext,
    ) -> RSIReversalDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)
        cur_rsi = float(current.get("rsi")) if finite(current.get("rsi")) else None
        prev_rsi = float(prev.get("rsi")) if prev is not None and finite(prev.get("rsi")) else None
        mean = float(self.config.mean_level)

        if direction == "LONG":
            if low <= stop:
                return RSIReversalDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return RSIReversalDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if (
                self.config.exit_at_mean
                and cur_rsi is not None
                and prev_rsi is not None
                and cur_rsi >= mean
                and prev_rsi < mean
            ):
                return RSIReversalDecision(action="EXIT", exit_underlying=close, exit_reason="RSI_MEAN_REVERSION")
            return self._hold()

        if direction == "SHORT":
            if high >= stop:
                return RSIReversalDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return RSIReversalDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if (
                self.config.exit_at_mean
                and cur_rsi is not None
                and prev_rsi is not None
                and cur_rsi <= mean
                and prev_rsi > mean
            ):
                return RSIReversalDecision(action="EXIT", exit_underlying=close, exit_reason="RSI_MEAN_REVERSION")
            return self._hold()

        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: RSIReversalPositionContext | None = None,
    ) -> RSIReversalDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]
        prev = candles_with_indicators.iloc[-2] if len(candles_with_indicators) >= 2 else None

        if position is not None:
            return self._evaluate_exit(current, prev, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "rsi"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both RSI setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return RSIReversalDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="RSI_OVERSOLD",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return RSIReversalDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="RSI_OVERBOUGHT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No RSI reversal on latest candle.")


class RSIReversalSignalGenerator:
    """Convenience wrapper for full-history and latest-candle RSI reversal signals."""

    def __init__(self, config: RSIReversalConfig | None = None) -> None:
        self.config = config or RSIReversalConfig()
        self.engine = RSIReversalSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_rsi_reversal_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: RSIReversalPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = RSIReversalPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = RSIReversalPositionContext(
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
        position: RSIReversalPositionContext | None = None,
    ) -> RSIReversalDecision:
        frame = build_rsi_reversal_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_rsi_reversal_signals(
    data: pd.DataFrame,
    config: RSIReversalConfig | None = None,
) -> pd.DataFrame:
    return RSIReversalSignalGenerator(config=config).generate(data)


def get_latest_rsi_reversal_signal(
    data: pd.DataFrame,
    config: RSIReversalConfig | None = None,
    position: RSIReversalPositionContext | None = None,
) -> RSIReversalDecision:
    return RSIReversalSignalGenerator(config=config).latest_signal(data, position=position)


NiftyRSIReversalSignalGenerator = RSIReversalSignalGenerator
generate_nifty_rsi_reversal_signals = generate_rsi_reversal_signals
get_latest_nifty_rsi_reversal_signal = get_latest_rsi_reversal_signal


__all__ = [
    "NiftyRSIReversalSignalGenerator",
    "RSIReversalConfig",
    "RSIReversalDecision",
    "RSIReversalPositionContext",
    "RSIReversalSignalEngine",
    "RSIReversalSignalGenerator",
    "build_rsi_reversal_with_indicators",
    "generate_nifty_rsi_reversal_signals",
    "generate_rsi_reversal_signals",
    "get_latest_nifty_rsi_reversal_signal",
    "get_latest_rsi_reversal_signal",
]
