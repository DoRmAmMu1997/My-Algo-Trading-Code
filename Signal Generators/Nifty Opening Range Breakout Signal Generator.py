"""
NIFTY Opening Range Breakout Signal Generator.

ATR-scaled breakout around each candle's open, ported from the public TradingBot
reference repo.

The idea (plain English)
------------------------
Draw a small "range" around the candle's open price, sized by current volatility:
open +/- (atr_multiplier x ATR). If price then closes outside that range it has
broken out with conviction, so we trade in the breakout direction. Using ATR to
size the range means the band auto-widens in fast markets and tightens in calm
ones, instead of using a fixed number of points.

  upper = open + atr_multiplier * ATR
  lower = open - atr_multiplier * ATR

Signals
-------
- ENTER_LONG  when close crosses ABOVE the upper band.
- ENTER_SHORT when close crosses BELOW the lower band.
- EXIT        on a fixed % stop or % target.

Using this file
---------------
- build_opening_range_breakout_with_indicators(ohlc): adds ATR/band + setup columns.
- OpeningRangeBreakoutSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_opening_range_breakout_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from misc_strategy_common import atr, finite, normalize_ohlc_frame, require_columns


@dataclass(frozen=True)
class OpeningRangeBreakoutConfig:
    """Tunable settings for the opening range breakout strategy."""

    atr_period: int = 14          # ATR lookback used to size the opening range
    atr_multiplier: float = 0.3   # how wide the range is, in ATRs, around the open
    stop_loss_pct: float = 0.015  # protective stop, 1.5% from entry
    target_pct: float = 0.03      # profit target, 3% from entry

    def __post_init__(self) -> None:
        if int(self.atr_period) <= 0:
            raise ValueError("atr_period must be positive.")
        if float(self.atr_multiplier) <= 0.0:
            raise ValueError("atr_multiplier must be greater than zero.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class OpeningRangeBreakoutPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class OpeningRangeBreakoutDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_opening_range_breakout_with_indicators(
    ohlc: pd.DataFrame,
    config: Optional[OpeningRangeBreakoutConfig] = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with ATR-scaled opening-range levels and cross flags."""
    config = config or OpeningRangeBreakoutConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    frame["atr"] = atr(frame, config.atr_period)
    frame["or_upper"] = open_ + config.atr_multiplier * frame["atr"]
    frame["or_lower"] = open_ - config.atr_multiplier * frame["atr"]

    # Breakout = previous close was inside the band and this close is outside it
    # (a genuine cross of the band, evaluated candle-over-candle).
    prev_close = close.shift(1)
    frame["long_setup"] = ((close > frame["or_upper"]) & (prev_close <= frame["or_upper"].shift(1))).fillna(False)
    frame["short_setup"] = ((close < frame["or_lower"]) & (prev_close >= frame["or_lower"].shift(1))).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class OpeningRangeBreakoutSignalEngine:
    """Decision engine for opening range breakout entries and exits."""

    def __init__(self, config: Optional[OpeningRangeBreakoutConfig] = None) -> None:
        self.config = config or OpeningRangeBreakoutConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.atr_period) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> OpeningRangeBreakoutDecision:
        return OpeningRangeBreakoutDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: OpeningRangeBreakoutPositionContext) -> OpeningRangeBreakoutDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return OpeningRangeBreakoutDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return OpeningRangeBreakoutDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return OpeningRangeBreakoutDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return OpeningRangeBreakoutDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[OpeningRangeBreakoutPositionContext] = None,
    ) -> OpeningRangeBreakoutDecision:
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
            return self._hold("Conflict: both breakout setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return OpeningRangeBreakoutDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="OR_UPPER_BREAKOUT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return OpeningRangeBreakoutDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="OR_LOWER_BREAKOUT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No opening range breakout on latest candle.")


class OpeningRangeBreakoutSignalGenerator:
    """Convenience wrapper for full-history and latest-candle ORB signals."""

    def __init__(self, config: Optional[OpeningRangeBreakoutConfig] = None) -> None:
        self.config = config or OpeningRangeBreakoutConfig()
        self.engine = OpeningRangeBreakoutSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_opening_range_breakout_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: Optional[OpeningRangeBreakoutPositionContext] = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = OpeningRangeBreakoutPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = OpeningRangeBreakoutPositionContext(
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
        position: Optional[OpeningRangeBreakoutPositionContext] = None,
    ) -> OpeningRangeBreakoutDecision:
        frame = build_opening_range_breakout_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_opening_range_breakout_signals(
    data: pd.DataFrame,
    config: Optional[OpeningRangeBreakoutConfig] = None,
) -> pd.DataFrame:
    return OpeningRangeBreakoutSignalGenerator(config=config).generate(data)


def get_latest_opening_range_breakout_signal(
    data: pd.DataFrame,
    config: Optional[OpeningRangeBreakoutConfig] = None,
    position: Optional[OpeningRangeBreakoutPositionContext] = None,
) -> OpeningRangeBreakoutDecision:
    return OpeningRangeBreakoutSignalGenerator(config=config).latest_signal(data, position=position)


NiftyOpeningRangeBreakoutSignalGenerator = OpeningRangeBreakoutSignalGenerator
generate_nifty_opening_range_breakout_signals = generate_opening_range_breakout_signals
get_latest_nifty_opening_range_breakout_signal = get_latest_opening_range_breakout_signal


__all__ = [
    "OpeningRangeBreakoutConfig",
    "OpeningRangeBreakoutPositionContext",
    "OpeningRangeBreakoutDecision",
    "build_opening_range_breakout_with_indicators",
    "OpeningRangeBreakoutSignalEngine",
    "OpeningRangeBreakoutSignalGenerator",
    "generate_opening_range_breakout_signals",
    "get_latest_opening_range_breakout_signal",
    "NiftyOpeningRangeBreakoutSignalGenerator",
    "generate_nifty_opening_range_breakout_signals",
    "get_latest_nifty_opening_range_breakout_signal",
]
