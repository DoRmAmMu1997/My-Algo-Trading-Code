"""
NIFTY Parabolic SAR Signal Generator.

Parabolic SAR trend flips with an ADX strength filter, ported from the public
TradingBot reference repo.

The idea (plain English)
------------------------
Parabolic SAR (see ``parabolic_sar`` in misc_strategy_common) prints a trailing
dot that sits below price in up-trends and above price in down-trends; when price
crosses the dot, the dot "stops and reverses" to the other side - a trend flip.
We enter in the new trend's direction on each flip. To avoid whipsaws in flat
markets, we only act when ADX (trend-strength, see ``adx``) confirms a real trend
is in force.

Signals
-------
- ENTER_LONG  when SAR flips from down to up (dot moves below price) AND ADX >= adx_min.
- ENTER_SHORT when SAR flips from up to down (dot moves above price) AND ADX >= adx_min.
- EXIT        on the opposite SAR flip, or on a fixed % stop / % target.

Using this file
---------------
- build_parabolic_sar_with_indicators(ohlc): adds SAR/direction/ADX + setup columns.
- ParabolicSARSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_parabolic_sar_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import (
    adx,
    finite,
    normalize_ohlc_frame,
    parabolic_sar,
    require_columns,
    validate_finite_config,
)


@dataclass(frozen=True)
class ParabolicSARConfig:
    """Tunable settings for the Parabolic SAR strategy."""

    af_start: float = 0.02     # SAR acceleration factor at the start of a trend
    af_step: float = 0.02      # how much the acceleration factor grows each new extreme
    af_max: float = 0.2        # cap on the acceleration factor (limits how fast the dot moves)
    adx_period: int = 14       # ADX lookback for the trend-strength filter
    adx_min: float = 20.0      # only trade flips when ADX is at least this (real trend present)
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        validate_finite_config(self)
        if int(self.adx_period) <= 0:
            raise ValueError("adx_period must be positive.")
        if not (0.0 < float(self.af_start) <= float(self.af_max)):
            raise ValueError("Require 0 < af_start <= af_max.")
        if float(self.af_step) <= 0.0:
            raise ValueError("af_step must be greater than zero.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class ParabolicSARPositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class ParabolicSARDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET, SAR_FLIP
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_parabolic_sar_with_indicators(
    ohlc: pd.DataFrame,
    config: ParabolicSARConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with SAR, SAR direction, ADX, flip flags, and risk levels."""
    config = config or ParabolicSARConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    sar, sar_dir = parabolic_sar(frame, config.af_start, config.af_step, config.af_max)
    frame["sar"] = sar
    frame["sar_dir"] = sar_dir
    frame["adx"] = adx(frame, config.adx_period)

    # A "flip" is a change in SAR direction from one candle to the next: bullish
    # flip = was -1 (down) last candle, is +1 (up) now. `strong` is the ADX gate.
    prev_dir = frame["sar_dir"].shift(1)
    flip_bull = (frame["sar_dir"] == 1) & (prev_dir == -1)
    flip_bear = (frame["sar_dir"] == -1) & (prev_dir == 1)
    strong = frame["adx"] >= config.adx_min

    frame["long_setup"] = (flip_bull & strong).fillna(False)
    frame["short_setup"] = (flip_bear & strong).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class ParabolicSARSignalEngine:
    """Decision engine for Parabolic SAR entries and exits."""

    def __init__(self, config: ParabolicSARConfig | None = None) -> None:
        self.config = config or ParabolicSARConfig()

    def minimum_history_bars(self) -> int:
        return 2 * int(self.config.adx_period) + 5

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> ParabolicSARDecision:
        return ParabolicSARDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: ParabolicSARPositionContext) -> ParabolicSARDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)
        cur_dir = current.get("sar_dir")
        cur_dir = int(cur_dir) if finite(cur_dir) else 0

        if direction == "LONG":
            if low <= stop:
                return ParabolicSARDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return ParabolicSARDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_dir == -1:
                return ParabolicSARDecision(action="EXIT", exit_underlying=close, exit_reason="SAR_FLIP")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return ParabolicSARDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return ParabolicSARDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if cur_dir == 1:
                return ParabolicSARDecision(action="EXIT", exit_underlying=close, exit_reason="SAR_FLIP")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: ParabolicSARPositionContext | None = None,
    ) -> ParabolicSARDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "sar_dir"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both SAR setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return ParabolicSARDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SAR_FLIP_BULLISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return ParabolicSARDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SAR_FLIP_BEARISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No SAR flip on latest candle.")


class ParabolicSARSignalGenerator:
    """Convenience wrapper for full-history and latest-candle Parabolic SAR signals."""

    def __init__(self, config: ParabolicSARConfig | None = None) -> None:
        self.config = config or ParabolicSARConfig()
        self.engine = ParabolicSARSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_parabolic_sar_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: ParabolicSARPositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = ParabolicSARPositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = ParabolicSARPositionContext(
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
        position: ParabolicSARPositionContext | None = None,
    ) -> ParabolicSARDecision:
        frame = build_parabolic_sar_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_parabolic_sar_signals(
    data: pd.DataFrame,
    config: ParabolicSARConfig | None = None,
) -> pd.DataFrame:
    return ParabolicSARSignalGenerator(config=config).generate(data)


def get_latest_parabolic_sar_signal(
    data: pd.DataFrame,
    config: ParabolicSARConfig | None = None,
    position: ParabolicSARPositionContext | None = None,
) -> ParabolicSARDecision:
    return ParabolicSARSignalGenerator(config=config).latest_signal(data, position=position)


NiftyParabolicSARSignalGenerator = ParabolicSARSignalGenerator
generate_nifty_parabolic_sar_signals = generate_parabolic_sar_signals
get_latest_nifty_parabolic_sar_signal = get_latest_parabolic_sar_signal


__all__ = [
    "NiftyParabolicSARSignalGenerator",
    "ParabolicSARConfig",
    "ParabolicSARDecision",
    "ParabolicSARPositionContext",
    "ParabolicSARSignalEngine",
    "ParabolicSARSignalGenerator",
    "build_parabolic_sar_with_indicators",
    "generate_nifty_parabolic_sar_signals",
    "generate_parabolic_sar_signals",
    "get_latest_nifty_parabolic_sar_signal",
    "get_latest_parabolic_sar_signal",
]
