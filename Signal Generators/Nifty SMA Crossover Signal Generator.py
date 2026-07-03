"""
NIFTY SMA Crossover Signal Generator.

Re-implementation of the classic fast/slow simple-moving-average crossover from
the public TradingBot reference repo, written in this codebase's standard
signal-engine style (see ``goldmine_strategy_logic.py``).

The idea (plain English)
------------------------
Keep two moving averages of price: a short/fast one (reacts quickly) and a
long/slow one (reacts slowly). When the fast average rises through the slow one,
momentum has turned up, so we go long; when it falls through, momentum has turned
down, so we go short. It is one of the oldest trend-following ideas there is.

Signals
-------
- ENTER_LONG  when the fast SMA crosses ABOVE the slow SMA ("golden cross").
- ENTER_SHORT when the fast SMA crosses BELOW the slow SMA ("death cross").
- EXIT        on a fixed % stop or % target, or if an opposite crossover appears.

Using this file
---------------
- build_sma_crossover_with_indicators(ohlc): returns the candles plus the SMA,
  crossover-flag, and entry/stop/target columns.
- SMACrossoverSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long entry, -1 = short entry, 2 = exit, 0 = nothing).
- get_latest_sma_crossover_signal(df, position=...): just the newest decision,
  which is what the live master runner calls each candle.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import finite, normalize_ohlc_frame, require_columns, sma


@dataclass(frozen=True)
class SMACrossoverConfig:
    """Tunable settings for the SMA crossover strategy."""

    short_window: int = 9      # fast SMA length (candles) - the quick-reacting line
    long_window: int = 21      # slow SMA length (candles) - the trend line
    stop_loss_pct: float = 0.015  # protective stop, 1.5% from entry
    target_pct: float = 0.03      # profit target, 3% from entry

    def __post_init__(self) -> None:
        positive_ints = {"short_window": self.short_window, "long_window": self.long_window}
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if int(self.short_window) >= int(self.long_window):
            raise ValueError("short_window must be smaller than long_window.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class SMACrossoverPositionContext:
    """Snapshot of an open SMA crossover trade so the engine can manage exits."""

    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class SMACrossoverDecision:
    """Standard decision object returned by the engine."""

    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, OPPOSITE_SIGNAL
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_sma_crossover_with_indicators(
    ohlc: pd.DataFrame,
    config: SMACrossoverConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with fast/slow SMAs, crossover flags, and risk levels."""
    config = config or SMACrossoverConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    frame["sma_short"] = sma(close, config.short_window)
    frame["sma_long"] = sma(close, config.long_window)

    # A "cross" is detected by comparing this candle with the previous one
    # (`.shift(1)` = the value one candle ago). A bullish cross means the fast
    # line was at/below the slow line last candle but is above it now.
    prev_short = frame["sma_short"].shift(1)
    prev_long = frame["sma_long"].shift(1)
    frame["cross_above"] = (frame["sma_short"] > frame["sma_long"]) & (prev_short <= prev_long)
    frame["cross_below"] = (frame["sma_short"] < frame["sma_long"]) & (prev_short >= prev_long)

    frame["long_setup"] = frame["cross_above"].fillna(False)
    frame["short_setup"] = frame["cross_below"].fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class SMACrossoverSignalEngine:
    """Decision engine for SMA crossover entries and exits."""

    def __init__(self, config: SMACrossoverConfig | None = None) -> None:
        self.config = config or SMACrossoverConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.long_window) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> SMACrossoverDecision:
        return SMACrossoverDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: SMACrossoverPositionContext) -> SMACrossoverDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return SMACrossoverDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return SMACrossoverDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if bool(current.get("short_setup", False)):
                return SMACrossoverDecision(action="EXIT", exit_underlying=close, exit_reason="OPPOSITE_SIGNAL")
            return self._hold()

        if direction == "SHORT":
            if high >= stop:
                return SMACrossoverDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return SMACrossoverDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if bool(current.get("long_setup", False)):
                return SMACrossoverDecision(action="EXIT", exit_underlying=close, exit_reason="OPPOSITE_SIGNAL")
            return self._hold()

        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: SMACrossoverPositionContext | None = None,
    ) -> SMACrossoverDecision:
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
            return self._hold("Conflict: both crossover setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return SMACrossoverDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SMA_BULLISH_CROSS",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return SMACrossoverDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SMA_BEARISH_CROSS",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No SMA crossover on latest candle.")


class SMACrossoverSignalGenerator:
    """Convenience wrapper for full-history and latest-candle SMA crossover signals."""

    def __init__(self, config: SMACrossoverConfig | None = None) -> None:
        self.config = config or SMACrossoverConfig()
        self.engine = SMACrossoverSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_sma_crossover_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: SMACrossoverPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = SMACrossoverPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = SMACrossoverPositionContext(
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
        position: SMACrossoverPositionContext | None = None,
    ) -> SMACrossoverDecision:
        frame = build_sma_crossover_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_sma_crossover_signals(
    data: pd.DataFrame,
    config: SMACrossoverConfig | None = None,
) -> pd.DataFrame:
    """Functional wrapper for full-history SMA crossover signals."""
    return SMACrossoverSignalGenerator(config=config).generate(data)


def get_latest_sma_crossover_signal(
    data: pd.DataFrame,
    config: SMACrossoverConfig | None = None,
    position: SMACrossoverPositionContext | None = None,
) -> SMACrossoverDecision:
    """Functional wrapper for the latest SMA crossover decision."""
    return SMACrossoverSignalGenerator(config=config).latest_signal(data, position=position)


# Nifty-flavored aliases keep the public naming style used across the repo.
NiftySMACrossoverSignalGenerator = SMACrossoverSignalGenerator
generate_nifty_sma_crossover_signals = generate_sma_crossover_signals
get_latest_nifty_sma_crossover_signal = get_latest_sma_crossover_signal


__all__ = [
    "NiftySMACrossoverSignalGenerator",
    "SMACrossoverConfig",
    "SMACrossoverDecision",
    "SMACrossoverPositionContext",
    "SMACrossoverSignalEngine",
    "SMACrossoverSignalGenerator",
    "build_sma_crossover_with_indicators",
    "generate_nifty_sma_crossover_signals",
    "generate_sma_crossover_signals",
    "get_latest_nifty_sma_crossover_signal",
    "get_latest_sma_crossover_signal",
]
