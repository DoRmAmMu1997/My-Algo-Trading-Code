"""
NIFTY Keltner Squeeze Signal Generator.

TTM-style "squeeze" breakout, ported from the public TradingBot reference repo.

The idea (plain English)
------------------------
Two volatility envelopes are compared: Bollinger Bands (width from standard
deviation) and Keltner Channels (width from ATR). When the Bollinger Bands shrink
*inside* the Keltner Channels, the market is unusually quiet - coiled like a
spring. That is the "squeeze". Quiet periods tend to be followed by a sharp move,
so when the squeeze RELEASES (Bollinger pops back outside Keltner) we enter in the
direction the MACD histogram is pointing (its sign = which way momentum leans).

Signals
-------
- ENTER_LONG  on a squeeze release while the MACD histogram is positive.
- ENTER_SHORT on a squeeze release while the MACD histogram is negative.
- EXIT        on a fixed % stop or % target.

Using this file
---------------
- build_keltner_squeeze_with_indicators(ohlc): adds BB/KC/MACD + squeeze + setup columns.
- KeltnerSqueezeSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_keltner_squeeze_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from misc_strategy_common import (
    bollinger_bands,
    finite,
    keltner_channels,
    macd,
    normalize_ohlc_frame,
    require_columns,
)


@dataclass(frozen=True)
class KeltnerSqueezeConfig:
    """Tunable settings for the Keltner squeeze strategy."""

    bb_period: int = 20        # Bollinger Band SMA length
    bb_std: float = 2.0        # Bollinger Band width in standard deviations
    kc_period: int = 20        # Keltner Channel EMA length
    kc_atr_period: int = 20    # ATR lookback for the Keltner Channel width
    kc_multiplier: float = 1.5 # Keltner Channel width in ATRs
    macd_fast: int = 12        # MACD fast EMA (momentum direction)
    macd_slow: int = 26        # MACD slow EMA
    macd_signal: int = 9       # MACD signal EMA
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        positive_ints = {
            "bb_period": self.bb_period,
            "kc_period": self.kc_period,
            "kc_atr_period": self.kc_atr_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if int(self.macd_fast) >= int(self.macd_slow):
            raise ValueError("macd_fast must be smaller than macd_slow.")
        if float(self.bb_std) <= 0.0 or float(self.kc_multiplier) <= 0.0:
            raise ValueError("bb_std and kc_multiplier must be greater than zero.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class KeltnerSqueezePositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class KeltnerSqueezeDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def build_keltner_squeeze_with_indicators(
    ohlc: pd.DataFrame,
    config: KeltnerSqueezeConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with BB/KC bands, MACD histogram, squeeze-release flags, and risk levels."""
    config = config or KeltnerSqueezeConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    bb_upper, _bb_mid, bb_lower = bollinger_bands(close, config.bb_period, config.bb_std)
    kc_upper, _kc_mid, kc_lower = keltner_channels(frame, config.kc_period, config.kc_atr_period, config.kc_multiplier)
    _, _, macd_hist = macd(close, config.macd_fast, config.macd_slow, config.macd_signal)

    frame["bb_upper"] = bb_upper
    frame["bb_lower"] = bb_lower
    frame["kc_upper"] = kc_upper
    frame["kc_lower"] = kc_lower
    frame["macd_hist"] = macd_hist

    # Squeeze ON = the Bollinger Bands sit entirely INSIDE the Keltner Channels
    # (lower BB above lower KC AND upper BB below upper KC) -> unusually low volatility.
    squeeze_on = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).fillna(False).astype(bool)
    frame["squeeze_on"] = squeeze_on
    # A "release" is the candle where the squeeze turns OFF (was on last candle,
    # off now). `shift(fill_value=False)` keeps the boolean dtype (no downcasting).
    release = squeeze_on.shift(1, fill_value=False) & (~squeeze_on)

    # On release, the MACD histogram's sign picks the breakout direction.
    frame["long_setup"] = (release & (frame["macd_hist"] > 0.0)).fillna(False)
    frame["short_setup"] = (release & (frame["macd_hist"] < 0.0)).fillna(False)

    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class KeltnerSqueezeSignalEngine:
    """Decision engine for Keltner squeeze entries and exits."""

    def __init__(self, config: KeltnerSqueezeConfig | None = None) -> None:
        self.config = config or KeltnerSqueezeConfig()

    def minimum_history_bars(self) -> int:
        return (
            max(self.config.bb_period, self.config.kc_period, self.config.kc_atr_period, self.config.macd_slow)
            + int(self.config.macd_signal)
            + 2
        )

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> KeltnerSqueezeDecision:
        return KeltnerSqueezeDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: KeltnerSqueezePositionContext) -> KeltnerSqueezeDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return KeltnerSqueezeDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return KeltnerSqueezeDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return KeltnerSqueezeDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return KeltnerSqueezeDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: KeltnerSqueezePositionContext | None = None,
    ) -> KeltnerSqueezeDecision:
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("No candles supplied.")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        require_columns(candles_with_indicators, ["long_setup", "short_setup", "macd_hist"])
        if len(candles_with_indicators) < self.minimum_history_bars():
            return self._hold("Insufficient history.")

        long_setup = bool(current["long_setup"])
        short_setup = bool(current["short_setup"])
        if long_setup and short_setup:
            return self._hold("Conflict: both squeeze setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return KeltnerSqueezeDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SQUEEZE_RELEASE_BULLISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return KeltnerSqueezeDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="SQUEEZE_RELEASE_BEARISH",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No squeeze release on latest candle.")


class KeltnerSqueezeSignalGenerator:
    """Convenience wrapper for full-history and latest-candle squeeze signals."""

    def __init__(self, config: KeltnerSqueezeConfig | None = None) -> None:
        self.config = config or KeltnerSqueezeConfig()
        self.engine = KeltnerSqueezeSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_keltner_squeeze_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: KeltnerSqueezePositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = KeltnerSqueezePositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = KeltnerSqueezePositionContext(
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
        position: KeltnerSqueezePositionContext | None = None,
    ) -> KeltnerSqueezeDecision:
        frame = build_keltner_squeeze_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_keltner_squeeze_signals(
    data: pd.DataFrame,
    config: KeltnerSqueezeConfig | None = None,
) -> pd.DataFrame:
    return KeltnerSqueezeSignalGenerator(config=config).generate(data)


def get_latest_keltner_squeeze_signal(
    data: pd.DataFrame,
    config: KeltnerSqueezeConfig | None = None,
    position: KeltnerSqueezePositionContext | None = None,
) -> KeltnerSqueezeDecision:
    return KeltnerSqueezeSignalGenerator(config=config).latest_signal(data, position=position)


NiftyKeltnerSqueezeSignalGenerator = KeltnerSqueezeSignalGenerator
generate_nifty_keltner_squeeze_signals = generate_keltner_squeeze_signals
get_latest_nifty_keltner_squeeze_signal = get_latest_keltner_squeeze_signal


__all__ = [
    "KeltnerSqueezeConfig",
    "KeltnerSqueezeDecision",
    "KeltnerSqueezePositionContext",
    "KeltnerSqueezeSignalEngine",
    "KeltnerSqueezeSignalGenerator",
    "NiftyKeltnerSqueezeSignalGenerator",
    "build_keltner_squeeze_with_indicators",
    "generate_keltner_squeeze_signals",
    "generate_nifty_keltner_squeeze_signals",
    "get_latest_keltner_squeeze_signal",
    "get_latest_nifty_keltner_squeeze_signal",
]
