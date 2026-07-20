"""
NIFTY Mean Reversion Z-Score Signal Generator.

Z-score mean reversion ported from the public TradingBot reference repo.

The idea (plain English)
------------------------
A z-score measures how far price has strayed from its own recent average, counted
in standard deviations (see ``rolling_zscore`` in misc_strategy_common). z = -2
means "unusually cheap vs the last N candles", z = +2 means "unusually expensive".
The bet is that such stretches snap back toward the average, so we buy deep-negative
z-scores and sell deep-positive ones, aiming to exit as price returns to its mean.
This is a fade / mean-reversion strategy.

Signals
-------
- ENTER_LONG  when the z-score crosses DOWN below -entry_z (deeply oversold).
- ENTER_SHORT when the z-score crosses UP above +entry_z (deeply overbought).
- EXIT        at the rolling mean (profit target), when the z-score reverts back
  through the exit band, or on a fixed % stop.

Using this file
---------------
- build_mean_reversion_zscore_with_indicators(ohlc): adds mean/z-score + setup columns.
- MeanReversionZscoreSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_mean_reversion_zscore_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import (
    finite,
    normalize_ohlc_frame,
    require_columns,
    rolling_zscore,
    sma,
    validate_finite_config,
)


@dataclass(frozen=True)
class MeanReversionZscoreConfig:
    """Tunable settings for the z-score mean reversion strategy."""

    lookback_period: int = 20  # window for the rolling mean and standard deviation
    entry_z: float = 2.0       # enter when |z| is at least this (how "stretched" to require)
    exit_z: float = 0.0        # exit when z reverts back to this band (0 = back to the mean)
    stop_loss_pct: float = 0.03  # protective stop, 3% from entry (target is the mean)

    def __post_init__(self) -> None:
        validate_finite_config(self)
        if int(self.lookback_period) <= 0:
            raise ValueError("lookback_period must be positive.")
        if float(self.entry_z) <= 0.0:
            raise ValueError("entry_z must be greater than zero.")
        if float(self.exit_z) < 0.0:
            raise ValueError("exit_z must be non-negative.")
        if float(self.stop_loss_pct) <= 0.0:
            raise ValueError("stop_loss_pct must be greater than zero.")


@dataclass
class MeanReversionZscorePositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class MeanReversionZscoreDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price = the rolling mean here
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, ZSCORE_REVERT
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_mean_reversion_zscore_with_indicators(
    ohlc: pd.DataFrame,
    config: MeanReversionZscoreConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with rolling mean, z-score, cross flags, and risk levels."""
    config = config or MeanReversionZscoreConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    frame["mean_sma"] = sma(close, config.lookback_period)
    frame["zscore"] = rolling_zscore(close, config.lookback_period)
    prev_z = frame["zscore"].shift(1)

    # Trigger on the candle the z-score first pushes past the entry band (a cross),
    # so we act once on a fresh extreme rather than every bar it stays stretched.
    frame["long_setup"] = ((frame["zscore"] < -config.entry_z) & (prev_z >= -config.entry_z)).fillna(False)
    frame["short_setup"] = ((frame["zscore"] > config.entry_z) & (prev_z <= config.entry_z)).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    # Profit target is reversion to the rolling mean.
    frame["long_target_from_setup"] = frame["mean_sma"]
    frame["short_target_from_setup"] = frame["mean_sma"]
    return frame


class MeanReversionZscoreSignalEngine:
    """Decision engine for z-score mean reversion entries and exits."""

    def __init__(self, config: MeanReversionZscoreConfig | None = None) -> None:
        self.config = config or MeanReversionZscoreConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.lookback_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> MeanReversionZscoreDecision:
        return MeanReversionZscoreDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(
        self, current: pd.Series, position: MeanReversionZscorePositionContext
    ) -> MeanReversionZscoreDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)
        cur_z = float(current.get("zscore")) if finite(current.get("zscore")) else None
        exit_z = float(self.config.exit_z)

        if direction == "LONG":
            if low <= stop:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if finite(target) and high >= target:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_z is not None and cur_z >= exit_z:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=close, exit_reason="ZSCORE_REVERT")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if finite(target) and low <= target:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_z is not None and cur_z <= -exit_z:
                return MeanReversionZscoreDecision(action="EXIT", exit_underlying=close, exit_reason="ZSCORE_REVERT")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: MeanReversionZscorePositionContext | None = None,
    ) -> MeanReversionZscoreDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "zscore", "mean_sma"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both z-score setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return MeanReversionZscoreDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="ZSCORE_OVERSOLD",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return MeanReversionZscoreDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="ZSCORE_OVERBOUGHT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No z-score setup on latest candle.")


class MeanReversionZscoreSignalGenerator:
    """Convenience wrapper for full-history and latest-candle z-score signals."""

    def __init__(self, config: MeanReversionZscoreConfig | None = None) -> None:
        self.config = config or MeanReversionZscoreConfig()
        self.engine = MeanReversionZscoreSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_mean_reversion_zscore_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: MeanReversionZscorePositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = MeanReversionZscorePositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = MeanReversionZscorePositionContext(
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
        position: MeanReversionZscorePositionContext | None = None,
    ) -> MeanReversionZscoreDecision:
        frame = build_mean_reversion_zscore_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_mean_reversion_zscore_signals(
    data: pd.DataFrame,
    config: MeanReversionZscoreConfig | None = None,
) -> pd.DataFrame:
    return MeanReversionZscoreSignalGenerator(config=config).generate(data)


def get_latest_mean_reversion_zscore_signal(
    data: pd.DataFrame,
    config: MeanReversionZscoreConfig | None = None,
    position: MeanReversionZscorePositionContext | None = None,
) -> MeanReversionZscoreDecision:
    return MeanReversionZscoreSignalGenerator(config=config).latest_signal(data, position=position)


NiftyMeanReversionZscoreSignalGenerator = MeanReversionZscoreSignalGenerator
generate_nifty_mean_reversion_zscore_signals = generate_mean_reversion_zscore_signals
get_latest_nifty_mean_reversion_zscore_signal = get_latest_mean_reversion_zscore_signal


__all__ = [
    "MeanReversionZscoreConfig",
    "MeanReversionZscoreDecision",
    "MeanReversionZscorePositionContext",
    "MeanReversionZscoreSignalEngine",
    "MeanReversionZscoreSignalGenerator",
    "NiftyMeanReversionZscoreSignalGenerator",
    "build_mean_reversion_zscore_with_indicators",
    "generate_mean_reversion_zscore_signals",
    "generate_nifty_mean_reversion_zscore_signals",
    "get_latest_mean_reversion_zscore_signal",
    "get_latest_nifty_mean_reversion_zscore_signal",
]
