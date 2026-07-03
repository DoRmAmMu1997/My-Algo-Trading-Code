"""
NIFTY Bollinger Bands Signal Generator.

Mean-reversion off the Bollinger Bands, ported from the public TradingBot
reference repo into this repo's standard signal-engine style.

The idea (plain English)
------------------------
Bollinger Bands wrap a moving average in a volatility envelope (see
``bollinger_bands`` in misc_strategy_common). Price usually stays inside the
envelope, so when it pokes the LOWER band and then turns back up, it is often
"too cheap" and snaps back toward the middle - a long. The mirror happens at the
UPPER band - a short. This is a fade / mean-reversion strategy, the opposite
mindset to trend-following.

Signals
-------
- ENTER_LONG  when the previous candle closed at/below the lower band and this
  candle closes back above it (a bounce up off the lower band).
- ENTER_SHORT when the previous candle closed at/above the upper band and this
  candle closes back below it (a rejection down off the upper band).
- EXIT        at the middle band (the "mean" we expect price to revert to), or on
  a fixed % stop if it keeps going the wrong way.

Using this file
---------------
- build_bollinger_bands_with_indicators(ohlc): adds band + setup + risk columns.
- BollingerBandsSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_bollinger_bands_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import bollinger_bands, finite, normalize_ohlc_frame, require_columns


@dataclass(frozen=True)
class BollingerBandsConfig:
    """Tunable settings for the Bollinger Bands mean-reversion strategy."""

    bb_period: int = 20        # SMA length for the middle band (candles)
    bb_std: float = 2.0        # band width in standard deviations (2 = the classic setting)
    stop_loss_pct: float = 0.02  # protective stop, 2% from entry (target is the middle band)

    def __post_init__(self) -> None:
        if int(self.bb_period) <= 0:
            raise ValueError("bb_period must be positive.")
        if float(self.bb_std) <= 0.0:
            raise ValueError("bb_std must be greater than zero.")
        if float(self.stop_loss_pct) <= 0.0:
            raise ValueError("stop_loss_pct must be greater than zero.")


@dataclass
class BollingerBandsPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class BollingerBandsDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, OPPOSITE_SIGNAL
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_bollinger_bands_with_indicators(
    ohlc: pd.DataFrame,
    config: BollingerBandsConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with Bollinger Bands, bounce flags, and risk levels."""
    config = config or BollingerBandsConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    upper, middle, lower = bollinger_bands(close, config.bb_period, config.bb_std)
    frame["bb_upper"] = upper
    frame["bb_middle"] = middle
    frame["bb_lower"] = lower

    prev_close = close.shift(1)
    prev_lower = frame["bb_lower"].shift(1)
    prev_upper = frame["bb_upper"].shift(1)

    # Bounce detection: last candle was outside/at the band, this candle is back
    # inside. That "poke and reverse" is the mean-reversion trigger.
    frame["long_setup"] = ((prev_close <= prev_lower) & (close > frame["bb_lower"])).fillna(False)
    frame["short_setup"] = ((prev_close >= prev_upper) & (close < frame["bb_upper"])).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    # The profit target for both sides is reversion to the middle band.
    frame["long_target_from_setup"] = frame["bb_middle"]
    frame["short_target_from_setup"] = frame["bb_middle"]
    return frame


class BollingerBandsSignalEngine:
    """Decision engine for Bollinger Bands mean-reversion entries and exits."""

    def __init__(self, config: BollingerBandsConfig | None = None) -> None:
        self.config = config or BollingerBandsConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.bb_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> BollingerBandsDecision:
        return BollingerBandsDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: BollingerBandsPositionContext) -> BollingerBandsDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return BollingerBandsDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return BollingerBandsDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return BollingerBandsDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return BollingerBandsDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: BollingerBandsPositionContext | None = None,
    ) -> BollingerBandsDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "bb_middle"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both Bollinger setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return BollingerBandsDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="LOWER_BAND_BOUNCE",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return BollingerBandsDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="UPPER_BAND_REJECTION",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No Bollinger setup on latest candle.")


class BollingerBandsSignalGenerator:
    """Convenience wrapper for full-history and latest-candle Bollinger signals."""

    def __init__(self, config: BollingerBandsConfig | None = None) -> None:
        self.config = config or BollingerBandsConfig()
        self.engine = BollingerBandsSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_bollinger_bands_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: BollingerBandsPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = BollingerBandsPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = BollingerBandsPositionContext(
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
        position: BollingerBandsPositionContext | None = None,
    ) -> BollingerBandsDecision:
        frame = build_bollinger_bands_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_bollinger_bands_signals(
    data: pd.DataFrame,
    config: BollingerBandsConfig | None = None,
) -> pd.DataFrame:
    return BollingerBandsSignalGenerator(config=config).generate(data)


def get_latest_bollinger_bands_signal(
    data: pd.DataFrame,
    config: BollingerBandsConfig | None = None,
    position: BollingerBandsPositionContext | None = None,
) -> BollingerBandsDecision:
    return BollingerBandsSignalGenerator(config=config).latest_signal(data, position=position)


NiftyBollingerBandsSignalGenerator = BollingerBandsSignalGenerator
generate_nifty_bollinger_bands_signals = generate_bollinger_bands_signals
get_latest_nifty_bollinger_bands_signal = get_latest_bollinger_bands_signal


__all__ = [
    "BollingerBandsConfig",
    "BollingerBandsDecision",
    "BollingerBandsPositionContext",
    "BollingerBandsSignalEngine",
    "BollingerBandsSignalGenerator",
    "NiftyBollingerBandsSignalGenerator",
    "build_bollinger_bands_with_indicators",
    "generate_bollinger_bands_signals",
    "generate_nifty_bollinger_bands_signals",
    "get_latest_bollinger_bands_signal",
    "get_latest_nifty_bollinger_bands_signal",
]
