"""
NIFTY Supertrend Signal Generator (Misc family).

ATR-band Supertrend trend-following, ported from the public TradingBot reference
repo. (The repo already has a separate hedged Supertrend family under
``Supertrend Strategies/``; this is the standalone single-leg flip version.)

The idea (plain English)
------------------------
Supertrend (see ``supertrend`` in misc_strategy_common) draws one line that
trails price by a few ATRs. While price stays above the line the trend is up and
the line is a rising stop; when price closes through it, the line flips to the
other side and the trend turns down. We ride the trend: go long on an up-flip, go
short on a down-flip, and use the line itself as the stop.

Signals
-------
- ENTER_LONG  when the Supertrend direction flips from down (-1) to up (+1).
- ENTER_SHORT when it flips from up (+1) to down (-1).
- EXIT        on the opposite flip, on the Supertrend-line stop, or on an optional
  fixed % target (disabled by default so the trade can ride the trend to the flip).

Note on behaviour
-----------------
Supertrend is an "always in the market" idea (every flip is both an exit and a
reverse entry). This repo uses a single-position, no-auto-reverse model, so an
opposite flip while in a trade is taken as an EXIT only; the next same-direction
flip then re-enters. The long_setup/short_setup columns themselves are fully
symmetric - only the simulated position stream is one-at-a-time.

Using this file
---------------
- build_supertrend_with_indicators(ohlc): adds the line/direction/flip + risk columns.
- SupertrendSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_supertrend_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import finite, normalize_ohlc_frame, require_columns, supertrend


@dataclass(frozen=True)
class SupertrendConfig:
    """Tunable settings for the Supertrend strategy."""

    atr_period: int = 10       # ATR lookback that sets the Supertrend band width
    multiplier: float = 3.0    # band width in ATRs (bigger = looser line, fewer flips)
    target_pct: float = 0.0    # 0 disables the fixed target (ride the trend to the flip)

    def __post_init__(self) -> None:
        if int(self.atr_period) <= 0:
            raise ValueError("atr_period must be positive.")
        if float(self.multiplier) <= 0.0:
            raise ValueError("multiplier must be greater than zero.")
        if float(self.target_pct) < 0.0:
            raise ValueError("target_pct must be non-negative (0 disables it).")


@dataclass
class SupertrendPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float = 0.0


@dataclass
class SupertrendDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop = the Supertrend line at entry
    target_underlying: float = 0.0  # profit-target price (0.0 = disabled; ride to flip)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, SUPERTREND_FLIP
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_supertrend_with_indicators(
    ohlc: pd.DataFrame,
    config: SupertrendConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with the Supertrend line, direction, flip flags, and risk levels."""
    config = config or SupertrendConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    st_line, st_dir = supertrend(frame, config.atr_period, config.multiplier)
    frame["supertrend"] = st_line
    frame["supertrend_dir"] = st_dir

    # A flip = the direction column changed between the previous candle and this one.
    prev_dir = frame["supertrend_dir"].shift(1)
    frame["long_setup"] = ((frame["supertrend_dir"] == 1) & (prev_dir == -1)).fillna(False)
    frame["short_setup"] = ((frame["supertrend_dir"] == -1) & (prev_dir == 1)).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    # The Supertrend line itself is the protective stop on each side.
    frame["long_stop_from_setup"] = frame["supertrend"]
    frame["short_stop_from_setup"] = frame["supertrend"]
    if config.target_pct > 0.0:
        frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
        frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    else:
        frame["long_target_from_setup"] = 0.0
        frame["short_target_from_setup"] = 0.0
    return frame


class SupertrendSignalEngine:
    """Decision engine for Supertrend flip entries and exits."""

    def __init__(self, config: SupertrendConfig | None = None) -> None:
        self.config = config or SupertrendConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.atr_period) + 5

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> SupertrendDecision:
        return SupertrendDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: SupertrendPositionContext) -> SupertrendDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)
        cur_dir = current.get("supertrend_dir")
        cur_dir = int(cur_dir) if finite(cur_dir) else 0

        if direction == "LONG":
            if low <= stop:
                return SupertrendDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if target > 0.0 and high >= target:
                return SupertrendDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_dir == -1:
                return SupertrendDecision(action="EXIT", exit_underlying=close, exit_reason="SUPERTREND_FLIP")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return SupertrendDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if target > 0.0 and low <= target:
                return SupertrendDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_dir == 1:
                return SupertrendDecision(action="EXIT", exit_underlying=close, exit_reason="SUPERTREND_FLIP")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: SupertrendPositionContext | None = None,
    ) -> SupertrendDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "supertrend"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both Supertrend setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if finite(entry) and finite(stop) and stop < entry and (target == 0.0 or entry < target):
                return SupertrendDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SUPERTREND_FLIP_BULLISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if finite(entry) and finite(stop) and stop > entry and (target == 0.0 or target < entry):
                return SupertrendDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SUPERTREND_FLIP_BEARISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No Supertrend flip on latest candle.")


class SupertrendSignalGenerator:
    """Convenience wrapper for full-history and latest-candle Supertrend signals."""

    def __init__(self, config: SupertrendConfig | None = None) -> None:
        self.config = config or SupertrendConfig()
        self.engine = SupertrendSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_supertrend_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: SupertrendPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = SupertrendPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = SupertrendPositionContext(
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
        position: SupertrendPositionContext | None = None,
    ) -> SupertrendDecision:
        frame = build_supertrend_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_supertrend_signals(
    data: pd.DataFrame,
    config: SupertrendConfig | None = None,
) -> pd.DataFrame:
    return SupertrendSignalGenerator(config=config).generate(data)


def get_latest_supertrend_signal(
    data: pd.DataFrame,
    config: SupertrendConfig | None = None,
    position: SupertrendPositionContext | None = None,
) -> SupertrendDecision:
    return SupertrendSignalGenerator(config=config).latest_signal(data, position=position)


NiftySupertrendMiscSignalGenerator = SupertrendSignalGenerator
generate_nifty_supertrend_misc_signals = generate_supertrend_signals
get_latest_nifty_supertrend_misc_signal = get_latest_supertrend_signal


__all__ = [
    "NiftySupertrendMiscSignalGenerator",
    "SupertrendConfig",
    "SupertrendDecision",
    "SupertrendPositionContext",
    "SupertrendSignalEngine",
    "SupertrendSignalGenerator",
    "build_supertrend_with_indicators",
    "generate_nifty_supertrend_misc_signals",
    "generate_supertrend_signals",
    "get_latest_nifty_supertrend_misc_signal",
    "get_latest_supertrend_signal",
]
