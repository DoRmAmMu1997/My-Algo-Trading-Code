"""
NIFTY Volatility Breakout Signal Generator.

Larry Williams style volatility breakout, ported from the public TradingBot
reference repo.

The idea (plain English)
------------------------
Take the previous candle's range (its high minus low) as a yardstick for "a
normal move". If today's price travels more than a fraction (`k_factor`) of that
range beyond the previous close, something has changed - a breakout is underway -
so we jump in the breakout's direction. Up through the upper trigger = long; down
through the lower trigger = short. It is a momentum/breakout strategy.

  upper_trigger = previous_close + k_factor * previous_range
  lower_trigger = previous_close - k_factor * previous_range

Signals
-------
- ENTER_LONG  when price crosses ABOVE the upper trigger.
- ENTER_SHORT when price crosses BELOW the lower trigger.
- EXIT        on a fixed % stop or % target.

Using this file
---------------
- build_volatility_breakout_with_indicators(ohlc): adds trigger + setup columns.
- VolatilityBreakoutSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_volatility_breakout_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import finite, normalize_ohlc_frame, require_columns


@dataclass(frozen=True)
class VolatilityBreakoutConfig:
    """Tunable settings for the volatility breakout strategy."""

    k_factor: float = 0.5      # how far beyond prev close (in prev-range units) to trigger
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        if float(self.k_factor) <= 0.0:
            raise ValueError("k_factor must be greater than zero.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class VolatilityBreakoutPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class VolatilityBreakoutDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_volatility_breakout_with_indicators(
    ohlc: pd.DataFrame,
    config: VolatilityBreakoutConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with breakout trigger levels and cross flags."""
    config = config or VolatilityBreakoutConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    prev_range = (frame["high"].astype(float).shift(1) - frame["low"].astype(float).shift(1))
    frame["upper_trigger"] = prev_close + config.k_factor * prev_range
    frame["lower_trigger"] = prev_close - config.k_factor * prev_range

    # Breakout = price was on the calm side of the trigger last candle and has
    # punched through it this candle (a fresh cross, not a sustained state).
    frame["long_setup"] = (
        (close > frame["upper_trigger"]) & (prev_close <= frame["upper_trigger"].shift(1))
    ).fillna(False)
    frame["short_setup"] = (
        (close < frame["lower_trigger"]) & (prev_close >= frame["lower_trigger"].shift(1))
    ).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class VolatilityBreakoutSignalEngine:
    """Decision engine for volatility breakout entries and exits."""

    def __init__(self, config: VolatilityBreakoutConfig | None = None) -> None:
        self.config = config or VolatilityBreakoutConfig()

    def minimum_history_bars(self) -> int:
        return 5

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> VolatilityBreakoutDecision:
        return VolatilityBreakoutDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(
        self, current: pd.Series, position: VolatilityBreakoutPositionContext
    ) -> VolatilityBreakoutDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return VolatilityBreakoutDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return VolatilityBreakoutDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return VolatilityBreakoutDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return VolatilityBreakoutDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: VolatilityBreakoutPositionContext | None = None,
    ) -> VolatilityBreakoutDecision:
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
                return VolatilityBreakoutDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="UPPER_BREAKOUT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return VolatilityBreakoutDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="LOWER_BREAKOUT",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No volatility breakout on latest candle.")


class VolatilityBreakoutSignalGenerator:
    """Convenience wrapper for full-history and latest-candle breakout signals."""

    def __init__(self, config: VolatilityBreakoutConfig | None = None) -> None:
        self.config = config or VolatilityBreakoutConfig()
        self.engine = VolatilityBreakoutSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_volatility_breakout_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: VolatilityBreakoutPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = VolatilityBreakoutPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = VolatilityBreakoutPositionContext(
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
        position: VolatilityBreakoutPositionContext | None = None,
    ) -> VolatilityBreakoutDecision:
        frame = build_volatility_breakout_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_volatility_breakout_signals(
    data: pd.DataFrame,
    config: VolatilityBreakoutConfig | None = None,
) -> pd.DataFrame:
    return VolatilityBreakoutSignalGenerator(config=config).generate(data)


def get_latest_volatility_breakout_signal(
    data: pd.DataFrame,
    config: VolatilityBreakoutConfig | None = None,
    position: VolatilityBreakoutPositionContext | None = None,
) -> VolatilityBreakoutDecision:
    return VolatilityBreakoutSignalGenerator(config=config).latest_signal(data, position=position)


NiftyVolatilityBreakoutSignalGenerator = VolatilityBreakoutSignalGenerator
generate_nifty_volatility_breakout_signals = generate_volatility_breakout_signals
get_latest_nifty_volatility_breakout_signal = get_latest_volatility_breakout_signal


__all__ = [
    "NiftyVolatilityBreakoutSignalGenerator",
    "VolatilityBreakoutConfig",
    "VolatilityBreakoutDecision",
    "VolatilityBreakoutPositionContext",
    "VolatilityBreakoutSignalEngine",
    "VolatilityBreakoutSignalGenerator",
    "build_volatility_breakout_with_indicators",
    "generate_nifty_volatility_breakout_signals",
    "generate_volatility_breakout_signals",
    "get_latest_nifty_volatility_breakout_signal",
    "get_latest_volatility_breakout_signal",
]
