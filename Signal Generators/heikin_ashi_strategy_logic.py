"""
Shared Heikin Ashi + Bollinger signal generator.

This module separates the Heikin Ashi strategy logic from any front-test or
paper-trade orchestration code.

Quick-start:
1. Pass normal OHLC candles to `build_heikin_ashi_with_bollinger(...)`.
2. Keep one `HeikinAshiSignalEngine()` instance alive across evaluations.
3. Call `evaluate_candle(...)` on each newly completed candle.
4. After a successful entry, call `consume_long_setup()` or
   `consume_short_setup()` so the latch state matches the original strategy.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


DEFAULT_BOLL_PERIOD = 20
DEFAULT_BOLL_STD = 2.0


@dataclass
class HeikinAshiPositionContext:
    """Minimal snapshot of an already-open trade."""

    direction: str


@dataclass
class HeikinAshiDecision:
    """
    Standard response object returned by the signal engine.

    `action` can be:
    - HOLD
    - ENTER_LONG
    - ENTER_SHORT
    - REVERSE_TO_LONG
    - REVERSE_TO_SHORT
    """

    action: str = "HOLD"
    entry_underlying: float = 0.0
    exit_reason: str = ""
    signal_triggered: bool = False


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Fail early when required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _resolve_timestamp(ohlc: pd.DataFrame) -> pd.Series:
    """Accept either a `timestamp` column or a datetime-like index."""
    if "timestamp" in ohlc.columns:
        timestamp = pd.to_datetime(ohlc["timestamp"], errors="coerce")
    else:
        timestamp = pd.to_datetime(pd.Series(ohlc.index, index=ohlc.index), errors="coerce")

    if timestamp.isna().all():
        raise ValueError("A valid 'timestamp' column or datetime-like index is required.")
    return timestamp


def build_heikin_ashi(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normal OHLC candles to Heikin Ashi candles.

    Formulas:
    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (Prev HA Open + Prev HA Close) / 2
    - HA High = max(High, HA Open, HA Close)
    - HA Low  = min(Low, HA Open, HA Close)
    """
    _require_columns(ohlc, ["open", "high", "low", "close"])

    frame = ohlc.copy()
    frame["timestamp"] = _resolve_timestamp(frame)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    numeric_columns = [column for column in ("open", "high", "low", "close") if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if frame.empty:
        return frame

    ha = frame.copy()
    ha_close = (ha["open"] + ha["high"] + ha["low"] + ha["close"]) / 4.0
    ha_open = [0.0] * len(ha)
    ha_open[0] = (float(ha.iloc[0]["open"]) + float(ha.iloc[0]["close"])) / 2.0

    for index in range(1, len(ha)):
        ha_open[index] = (ha_open[index - 1] + float(ha_close.iloc[index - 1])) / 2.0

    ha["ha_open"] = ha_open
    ha["ha_close"] = ha_close
    ha["ha_high"] = pd.concat([ha["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"] = pd.concat([ha["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha


def build_heikin_ashi_with_bollinger(
    ohlc: pd.DataFrame,
    boll_period: int = DEFAULT_BOLL_PERIOD,
    boll_std: float = DEFAULT_BOLL_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands on Heikin Ashi close."""
    if int(boll_period) <= 0:
        raise ValueError("`boll_period` must be positive.")
    if float(boll_std) <= 0:
        raise ValueError("`boll_std` must be positive.")

    ha = build_heikin_ashi(ohlc)
    if ha.empty:
        return ha

    ha["bb_mid"] = ha["ha_close"].rolling(int(boll_period)).mean()
    rolling_std = ha["ha_close"].rolling(int(boll_period)).std(ddof=0)
    ha["bb_upper"] = ha["bb_mid"] + (float(boll_std) * rolling_std)
    ha["bb_lower"] = ha["bb_mid"] - (float(boll_std) * rolling_std)
    return ha.dropna(subset=["bb_mid", "bb_upper", "bb_lower"]).reset_index(drop=True)


class HeikinAshiSignalEngine:
    """
    Stateful signal engine for the Heikin Ashi + Bollinger strategy.

    Memory kept across candles:
    - `lower_touch_latched`: lower-band touch seen, waiting for bullish bounce
    - `upper_touch_latched`: upper-band touch seen, waiting for bearish bounce
    """

    def __init__(self):
        self.lower_touch_latched = False
        self.upper_touch_latched = False

    @staticmethod
    def _hold(signal_triggered: bool = False) -> HeikinAshiDecision:
        return HeikinAshiDecision(action="HOLD", signal_triggered=signal_triggered)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        return str(direction).strip().upper()

    def consume_long_setup(self) -> None:
        """Reset latches after a successful long entry."""
        self.lower_touch_latched = False
        self.upper_touch_latched = False

    def consume_short_setup(self) -> None:
        """Reset latches after a successful short entry."""
        self.upper_touch_latched = False
        self.lower_touch_latched = False

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: Optional[HeikinAshiPositionContext] = None,
    ) -> HeikinAshiDecision:
        """
        Evaluate the latest candle and return one decision.

        Flat-state rules:
        - lower-band touch then middle-band recovery -> ENTER_LONG
        - upper-band touch then middle-band rejection -> ENTER_SHORT

        In-position rules:
        - long + short signal -> REVERSE_TO_SHORT
        - short + long signal -> REVERSE_TO_LONG
        """
        required_columns = [
            "ha_close",
            "ha_low",
            "ha_high",
            "bb_mid",
            "bb_lower",
            "bb_upper",
        ]
        if candles_with_indicators is None:
            return self._hold()

        _require_columns(candles_with_indicators, required_columns)
        if len(candles_with_indicators) < 2:
            return self._hold()

        current = candles_with_indicators.iloc[-1]
        previous = candles_with_indicators.iloc[-2]

        current_values = pd.Series(
            [
                current["ha_close"],
                current["ha_low"],
                current["ha_high"],
                current["bb_mid"],
                current["bb_lower"],
                current["bb_upper"],
                previous["ha_close"],
                previous["bb_mid"],
            ]
        )
        if current_values.isna().any():
            return self._hold()

        current_ha_close = float(current["ha_close"])
        current_bb_mid = float(current["bb_mid"])
        previous_ha_close = float(previous["ha_close"])
        previous_bb_mid = float(previous["bb_mid"])
        current_ha_low = float(current["ha_low"])
        current_ha_high = float(current["ha_high"])
        current_bb_lower = float(current["bb_lower"])
        current_bb_upper = float(current["bb_upper"])

        if current_ha_low <= current_bb_lower:
            self.lower_touch_latched = True
        if current_ha_high >= current_bb_upper:
            self.upper_touch_latched = True

        long_signal = (
            self.lower_touch_latched
            and previous_ha_close < previous_bb_mid
            and current_ha_close >= current_bb_mid
        )
        short_signal = (
            self.upper_touch_latched
            and previous_ha_close > previous_bb_mid
            and current_ha_close <= current_bb_mid
        )

        if long_signal and short_signal:
            if position is not None:
                direction = self._normalize_direction(position.direction)
                if direction == "LONG":
                    long_signal = False
                elif direction == "SHORT":
                    short_signal = False
                else:
                    return self._hold(signal_triggered=True)
            else:
                return self._hold(signal_triggered=True)

        if position is None:
            if long_signal:
                return HeikinAshiDecision(
                    action="ENTER_LONG",
                    entry_underlying=current_ha_close,
                    signal_triggered=True,
                )
            if short_signal:
                return HeikinAshiDecision(
                    action="ENTER_SHORT",
                    entry_underlying=current_ha_close,
                    signal_triggered=True,
                )
            return self._hold()

        direction = self._normalize_direction(position.direction)
        if direction == "LONG" and short_signal:
            return HeikinAshiDecision(
                action="REVERSE_TO_SHORT",
                entry_underlying=current_ha_close,
                exit_reason="REVERSAL_TO_SHORT",
                signal_triggered=True,
            )

        if direction == "SHORT" and long_signal:
            return HeikinAshiDecision(
                action="REVERSE_TO_LONG",
                entry_underlying=current_ha_close,
                exit_reason="REVERSAL_TO_LONG",
                signal_triggered=True,
            )

        return self._hold()
