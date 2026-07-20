"""
NIFTY RSI Divergence Signal Generator.

Classic price/RSI divergence, ported from the public TradingBot reference repo.

The idea (plain English)
------------------------
"Divergence" is when price and momentum disagree, hinting the current move is
running out of steam:
- Bullish divergence: price grinds to a LOWER low, but RSI makes a HIGHER low -
  selling pressure is fading even though price dipped, so a bounce is likely.
- Bearish divergence: price pushes to a HIGHER high, but RSI makes a LOWER high -
  buying pressure is fading, so a drop is likely.
We detect "swing" lows/highs (little valleys/peaks, see ``find_swing_lows`` /
``find_swing_highs``) on both the price and the RSI line, then compare the last
two of each.

Signals
-------
- ENTER_LONG  on a confirmed bullish divergence.
- ENTER_SHORT on a confirmed bearish divergence.
- EXIT        on a fixed % stop or % target.

No look-ahead: a swing is only "confirmed" `swing_window` candles after it forms
(we must see the candles on both sides), so a divergence is emitted on the
confirmation candle, never back-dated onto the swing itself.

Using this file
---------------
- build_rsi_divergence_with_indicators(ohlc): adds RSI + divergence setup columns.
- RSIDivergenceSignalGenerator().generate(df): full-history signals; adds a
  `backtestStream` column (1 = long, -1 = short, 2 = exit, 0 = nothing).
- get_latest_rsi_divergence_signal(df, position=...): newest decision only.

This module does not resample - feed it already-prepared OHLC candles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from misc_strategy_common import (
    find_swing_highs,
    find_swing_lows,
    finite,
    normalize_ohlc_frame,
    require_columns,
    rsi,
    validate_finite_config,
)


@dataclass(frozen=True)
class RSIDivergenceConfig:
    """Tunable settings for the RSI divergence strategy."""

    rsi_period: int = 14       # RSI lookback (candles)
    swing_window: int = 5      # candles required on EACH side to confirm a swing high/low
    stop_loss_pct: float = 0.02   # protective stop, 2% from entry
    target_pct: float = 0.04      # profit target, 4% from entry

    def __post_init__(self) -> None:
        validate_finite_config(self)
        if int(self.rsi_period) <= 0:
            raise ValueError("rsi_period must be positive.")
        if int(self.swing_window) <= 0:
            raise ValueError("swing_window must be positive.")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")


@dataclass
class RSIDivergencePositionContext:
    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class RSIDivergenceDecision:
    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0   # underlying price to enter at (ENTER_* actions)
    stop_underlying: float = 0.0    # protective stop price for the trade
    target_underlying: float = 0.0  # profit-target price (0.0 when not used)
    exit_underlying: float = 0.0    # price the exit triggered at (EXIT action)
    exit_reason: str = ""           # why we exited: e.g. STOP, TARGET
    reason: str = ""                # plain-English reason we entered
    signal_triggered: bool = False  # True if a valid signal fired (even if no order placed)
    debug: dict[str, Any] = field(default_factory=dict)  # optional extra diagnostics


def _last_two_at_or_before(positions: list[int], limit: int) -> tuple[int, int] | None:
    """Return the last two swing positions that are confirmed by `limit`."""
    eligible = [p for p in positions if p <= limit]
    if len(eligible) < 2:
        return None
    return eligible[-2], eligible[-1]


def build_rsi_divergence_with_indicators(
    ohlc: pd.DataFrame,
    config: RSIDivergenceConfig | None = None,
) -> pd.DataFrame:
    """Enrich OHLC candles with RSI, divergence flags, and risk levels."""
    config = config or RSIDivergenceConfig()
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame

    close = frame["close"].astype(float)
    frame["rsi"] = rsi(close, config.rsi_period)
    window = int(config.swing_window)
    n = len(frame)

    close_vals = close.to_numpy()
    rsi_vals = frame["rsi"].to_numpy()
    price_lows = find_swing_lows(close, window)
    price_highs = find_swing_highs(close, window)
    rsi_lows = find_swing_lows(frame["rsi"], window)
    rsi_highs = find_swing_highs(frame["rsi"], window)

    long_setup = np.zeros(n, dtype=bool)
    short_setup = np.zeros(n, dtype=bool)

    # Bullish divergence: price lower low + RSI higher low, emitted at the bar
    # that confirms the more recent price swing low.
    for idx in range(1, len(price_lows)):
        p_prev, p = price_lows[idx - 1], price_lows[idx]
        confirm_bar = p + window
        if confirm_bar >= n:
            continue
        rsi_pair = _last_two_at_or_before(rsi_lows, p)
        if rsi_pair is None:
            continue
        r_prev, r = rsi_pair
        if (
            close_vals[p] < close_vals[p_prev]
            and rsi_vals[r] > rsi_vals[r_prev]
            and np.isfinite(rsi_vals[r])
            and np.isfinite(rsi_vals[r_prev])
        ):
            long_setup[confirm_bar] = True

    # Bearish divergence: price higher high + RSI lower high.
    for idx in range(1, len(price_highs)):
        p_prev, p = price_highs[idx - 1], price_highs[idx]
        confirm_bar = p + window
        if confirm_bar >= n:
            continue
        rsi_pair = _last_two_at_or_before(rsi_highs, p)
        if rsi_pair is None:
            continue
        r_prev, r = rsi_pair
        if (
            close_vals[p] > close_vals[p_prev]
            and rsi_vals[r] < rsi_vals[r_prev]
            and np.isfinite(rsi_vals[r])
            and np.isfinite(rsi_vals[r_prev])
        ):
            short_setup[confirm_bar] = True

    frame["long_setup"] = long_setup
    frame["short_setup"] = short_setup
    frame["long_entry_price"] = close
    frame["short_entry_price"] = close
    frame["long_stop_from_setup"] = close * (1.0 - config.stop_loss_pct)
    frame["long_target_from_setup"] = close * (1.0 + config.target_pct)
    frame["short_stop_from_setup"] = close * (1.0 + config.stop_loss_pct)
    frame["short_target_from_setup"] = close * (1.0 - config.target_pct)
    return frame


class RSIDivergenceSignalEngine:
    """Decision engine for RSI divergence entries and exits."""

    def __init__(self, config: RSIDivergenceConfig | None = None) -> None:
        self.config = config or RSIDivergenceConfig()

    def minimum_history_bars(self) -> int:
        return int(self.config.rsi_period) + 4 * int(self.config.swing_window) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> RSIDivergenceDecision:
        return RSIDivergenceDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(self, current: pd.Series, position: RSIDivergencePositionContext) -> RSIDivergenceDecision:
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return RSIDivergenceDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if high >= target:
                return RSIDivergenceDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        if direction == "SHORT":
            if high >= stop:
                return RSIDivergenceDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if low <= target:
                return RSIDivergenceDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            return self._hold()
        return self._hold("Unknown position direction.")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: RSIDivergencePositionContext | None = None,
    ) -> RSIDivergenceDecision:
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
            return self._hold("Conflict: both divergence setups true.")

        if long_setup:
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and stop < entry < target:
                return RSIDivergenceDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="BULLISH_DIVERGENCE",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid long entry prices.", signal_triggered=True)

        if short_setup:
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            if all(finite(value) for value in [entry, stop, target]) and target < entry < stop:
                return RSIDivergenceDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason="BEARISH_DIVERGENCE",
                    signal_triggered=True,
                    debug={"entry_timing": "NEXT_OPEN", "timestamp": current.get("timestamp")},
                )
            return self._hold("Invalid short entry prices.", signal_triggered=True)

        return self._hold("No RSI divergence on latest candle.")


class RSIDivergenceSignalGenerator:
    """Convenience wrapper for full-history and latest-candle RSI divergence signals."""

    def __init__(self, config: RSIDivergenceConfig | None = None) -> None:
        self.config = config or RSIDivergenceConfig()
        self.engine = RSIDivergenceSignalEngine(self.config)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = build_rsi_divergence_with_indicators(data, self.config)
        if frame.empty:
            return frame

        actions: list[str] = []
        entries: list[float] = []
        stops: list[float] = []
        targets: list[float] = []
        stream: list[int] = []
        position: RSIDivergencePositionContext | None = None

        for index in range(len(frame)):
            decision = self.engine.evaluate_candle(frame.iloc[: index + 1], position=position)
            actions.append(decision.action)
            entries.append(decision.entry_underlying)
            stops.append(decision.stop_underlying)
            targets.append(decision.target_underlying)

            if decision.action == "ENTER_LONG":
                stream.append(1)
                position = RSIDivergencePositionContext(
                    "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
                )
            elif decision.action == "ENTER_SHORT":
                stream.append(-1)
                position = RSIDivergencePositionContext(
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
        position: RSIDivergencePositionContext | None = None,
    ) -> RSIDivergenceDecision:
        frame = build_rsi_divergence_with_indicators(data, self.config)
        return self.engine.evaluate_candle(frame, position=position)


def generate_rsi_divergence_signals(
    data: pd.DataFrame,
    config: RSIDivergenceConfig | None = None,
) -> pd.DataFrame:
    return RSIDivergenceSignalGenerator(config=config).generate(data)


def get_latest_rsi_divergence_signal(
    data: pd.DataFrame,
    config: RSIDivergenceConfig | None = None,
    position: RSIDivergencePositionContext | None = None,
) -> RSIDivergenceDecision:
    return RSIDivergenceSignalGenerator(config=config).latest_signal(data, position=position)


NiftyRSIDivergenceSignalGenerator = RSIDivergenceSignalGenerator
generate_nifty_rsi_divergence_signals = generate_rsi_divergence_signals
get_latest_nifty_rsi_divergence_signal = get_latest_rsi_divergence_signal


__all__ = [
    "NiftyRSIDivergenceSignalGenerator",
    "RSIDivergenceConfig",
    "RSIDivergenceDecision",
    "RSIDivergencePositionContext",
    "RSIDivergenceSignalEngine",
    "RSIDivergenceSignalGenerator",
    "build_rsi_divergence_with_indicators",
    "generate_nifty_rsi_divergence_signals",
    "generate_rsi_divergence_signals",
    "get_latest_nifty_rsi_divergence_signal",
    "get_latest_rsi_divergence_signal",
]
