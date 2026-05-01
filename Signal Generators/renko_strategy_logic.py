"""
Shared Renko strategy logic used by both front-test and backtest scripts.

Beginner quick-start:
1. `build_renko_with_indicators()` converts normal OHLC candles into Renko + EMA columns.
2. Create one `RenkoSignalEngine()` object and keep it alive across candles.
3. For each newly formed Renko candle, call `evaluate_candle(...)`.
4. If a trade is open, pass a `RenkoPositionContext`; otherwise pass `None`.
5. Use returned `RenkoDecision.action` to decide whether to enter, exit, or hold.

Why this file exists:
1. Keep one single source of truth for Renko signal rules.
2. Avoid drift between front-test and backtest behavior.
3. Make strategy rules easier to read, test, and maintain.

What this file contains:
1. Indicator utilities (ATR).
2. Renko builder utilities.
3. A stateful signal engine that returns clean entry/exit decisions.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class RenkoPositionContext:
    """
    Minimal snapshot of an already-open trade.

    The engine only needs these values to evaluate exits:
    - `direction`: LONG or SHORT
    - `entry_underlying`: entry price on underlying
    - `stop_underlying`: current fixed stop on underlying
    - `rr_armed`: whether RR-based color exit is armed
    """

    direction: str
    entry_underlying: float
    stop_underlying: float
    rr_armed: bool = False


@dataclass
class RenkoDecision:
    """
    Standard response object returned by the signal engine.

    `action` can be:
    - HOLD
    - ENTER_LONG
    - ENTER_SHORT
    - EXIT
    """

    # Final decision for the current evaluation step.
    action: str = "HOLD"
    # Suggested entry/stop values on underlying (used only for entry actions).
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    # Short machine-friendly reason string for exits (STOP / EMA_EXIT / RR_*).
    exit_reason: str = ""
    # Updated RR state (engine can arm it after sufficient profit move).
    rr_armed: bool = False
    # True when a signal condition happened, even if entry validation later failed.
    signal_triggered: bool = False


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR using a simple moving average of True Range.

    True Range for each row is the maximum of:
    1. High - Low
    2. abs(High - Previous Close)
    3. abs(Low - Previous Close)
    """
    # Previous close is needed to capture overnight or candle-to-candle gaps.
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def build_renko_from_close(
    df: pd.DataFrame,
    box_size: float
) -> pd.DataFrame:
    """
    Build Renko bricks from close prices with explicit trigger levels.

    Trigger rules (same as front-test):
    - Print next green brick only if price >= previous brick high + box_size
    - Print next red brick only if price <= previous brick low - box_size

    Important:
    - One source candle can print multiple Renko bricks if price moves a lot.
    - Output columns match downstream strategy expectations.
    """
    closes = df["close"].tolist()
    times = df["timestamp"].tolist()
    if not closes:
        return pd.DataFrame()

    # Before any printed brick, we anchor to the first close.
    # Think of this as the initial reference level from which first brick emerges.
    last_brick_open = closes[0]
    last_brick_close = closes[0]
    rows = []

    for i in range(1, len(closes)):
        price = closes[i]
        ts = times[i]

        # Keep printing bricks until the current price no longer crosses
        # the next trigger level.
        while True:
            # Previous brick high/low create the next valid trigger levels.
            prev_high = max(last_brick_open, last_brick_close)
            prev_low = min(last_brick_open, last_brick_close)
            up_trigger = prev_high + box_size
            down_trigger = prev_low - box_size

            if price >= up_trigger:
                # New green brick starts from previous high and closes one box above it.
                brick_open = prev_high
                brick_close = prev_high + box_size
                rows.append(
                    {
                        "timestamp": ts,
                        "open": brick_open,
                        "high": max(brick_open, brick_close),
                        "low": min(brick_open, brick_close),
                        "close": brick_close,
                        "color": "green",
                    }
                )
                last_brick_open = brick_open
                last_brick_close = brick_close
                continue

            if price <= down_trigger:
                # New red brick starts from previous low and closes one box below it.
                brick_open = prev_low
                brick_close = prev_low - box_size
                rows.append(
                    {
                        "timestamp": ts,
                        "open": brick_open,
                        "high": max(brick_open, brick_close),
                        "low": min(brick_open, brick_close),
                        "close": brick_close,
                        "color": "red",
                    }
                )
                last_brick_open = brick_open
                last_brick_close = brick_close
                continue

            # Stop when current price is inside trigger band.
            break

    return pd.DataFrame(rows)


def build_renko_with_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLC candles into Renko candles plus EMA indicators.

    Steps:
    1. Compute ATR(14).
    2. Use latest ATR value as dynamic Renko box size.
    3. Build Renko bricks from close prices.
    4. Add EMA(5), EMA(21), EMA(44) on Renko close.
    """
    # ATR is used for dynamic Renko box size.
    atr_series = atr(ohlc, 14)
    # box_size = float(atr_series.iloc[-1])
    box_size = 12.5
    if pd.isna(box_size) or box_size <= 0:
        # ATR may be NaN in early warm-up or in malformed data.
        return pd.DataFrame()

    # Build Renko from source OHLC into each printed brick.
    renko = build_renko_from_close(ohlc, box_size)
    if renko.empty:
        return renko

    renko["ema5"] = renko["close"].ewm(span=5, adjust=False).mean()
    renko["ema21"] = renko["close"].ewm(span=21, adjust=False).mean()
    renko["ema44"] = renko["close"].ewm(span=44, adjust=False).mean()
    renko["box_size"] = box_size
    return renko


class RenkoSignalEngine:
    """
    Stateful signal engine for entries, re-entries, and exits.

    This engine carries memory across candles:
    - Pullback flags for EMA5/EMA21 zones.
    - Direction of the previously closed trade (for re-entry gating).
    """

    def __init__(self):
        # Bullish pullback flags (legacy CE naming retained to match old logic).
        self.ce_pullback_ema5 = False
        self.ce_pullback_ema21 = False
        # Bearish pullback flags (legacy PE naming retained to match old logic).
        self.pe_pullback_ema5 = False
        self.pe_pullback_ema21 = False
        # Last closed direction used to gate re-entry direction.
        # Example:
        # - if previous closed trade is LONG, only long re-entry is allowed.
        # - fresh entries are still allowed both sides by design.
        self.previous_trade_direction = ""

    def update_previous_trade_direction(self, direction: str) -> None:
        """
        Store last closed direction for re-entry gating.

        Only LONG/SHORT are accepted to avoid accidental bad state.
        """
        direction_txt = str(direction).strip().upper()
        if direction_txt in ("LONG", "SHORT"):
            self.previous_trade_direction = direction_txt

    def reset_reentry_flags(self) -> None:
        """Clear all pullback memory flags."""
        self.ce_pullback_ema5 = False
        self.ce_pullback_ema21 = False
        self.pe_pullback_ema5 = False
        self.pe_pullback_ema21 = False

    def _update_reentry_flags(
        self,
        c: float,
        ema5: float,
        ema21: float,
        has_active_position: bool,
    ) -> None:
        """
        Update pullback memory while flat.

        Rule:
        - If close falls below EMA5/EMA21, arm bullish pullback flags.
        - If close rises above EMA5/EMA21, arm bearish pullback flags.
        """
        # Pullback flags are only tracked while flat.
        # Once in a trade, these memory flags are irrelevant and are not updated.
        if has_active_position:
            return

        if c < ema5:
            self.ce_pullback_ema5 = True
        if c < ema21:
            self.ce_pullback_ema21 = True
        if c > ema5:
            self.pe_pullback_ema5 = True
        if c > ema21:
            self.pe_pullback_ema21 = True

    @staticmethod
    def _hold(rr_armed: bool = False) -> RenkoDecision:
        """Convenience helper for a no-action decision."""
        return RenkoDecision(action="HOLD", rr_armed=rr_armed)

    def _evaluate_exit(
        self,
        position: RenkoPositionContext,
        c: float,
        color: str,
        ema5: float,
        ema21: float,
        ema44: float,
    ) -> RenkoDecision:
        """
        Evaluate exit conditions for an already-open position.

        Exit triggers:
        1. Stop hit
        2. EMA trend invalidation
        3. Opposite color candle after RR >= 1.5 has been armed
        """
        # Keep local copy so method can return updated state without mutating input object.
        rr_armed = bool(position.rr_armed)

        if position.direction == "LONG":
            # Risk-per-unit is entry minus stop for long positions.
            risk = float(position.entry_underlying) - float(position.stop_underlying)
            rr = (c - float(position.entry_underlying)) / risk if risk > 0 else -999.0
            if rr >= 1.5:
                rr_armed = True

            stop_hit = c <= float(position.stop_underlying)
            trend_exit = c < ema5 and c < ema21 and c < ema44
            rr_exit = rr_armed and color == "red"
            if stop_hit or trend_exit or rr_exit:
                reason = "STOP" if stop_hit else ("EMA_EXIT" if trend_exit else "RR_RED_CANDLE")
                return RenkoDecision(action="EXIT", exit_reason=reason, rr_armed=rr_armed)
            return self._hold(rr_armed=rr_armed)

        if position.direction == "SHORT":
            # Risk-per-unit is stop minus entry for short positions.
            risk = float(position.stop_underlying) - float(position.entry_underlying)
            rr = (float(position.entry_underlying) - c) / risk if risk > 0 else -999.0
            if rr >= 1.5:
                rr_armed = True

            stop_hit = c >= float(position.stop_underlying)
            trend_exit = c > ema5 and c > ema21 and c > ema44
            rr_exit = rr_armed and color == "green"
            if stop_hit or trend_exit or rr_exit:
                reason = "STOP" if stop_hit else ("EMA_EXIT" if trend_exit else "RR_GREEN_CANDLE")
                return RenkoDecision(action="EXIT", exit_reason=reason, rr_armed=rr_armed)
            return self._hold(rr_armed=rr_armed)

        return self._hold(rr_armed=rr_armed)

    def evaluate_candle(
        self,
        renko: pd.DataFrame,
        position: Optional[RenkoPositionContext] = None,
    ) -> RenkoDecision:
        """
        Evaluate the latest Renko candle and return one strategy decision.

        Input:
        - `renko`: DataFrame with close/color/ema5/ema21/ema44 columns
        - `position`: open-position context; pass None when flat

        Output:
        - `RenkoDecision` with action and relevant payload fields
        """
        # Need at least 3 Renko rows:
        # - current candle
        # - previous candle for stop anchoring
        # - minimal history buffer for stable logic
        if renko is None or len(renko) < 3:
            return self._hold(rr_armed=bool(position.rr_armed) if position else False)

        cur = renko.iloc[-1]
        # Previous printed Renko candle used for initial stop anchoring.
        prev2 = renko.iloc[-2]

        c = float(cur["close"])
        color = str(cur["color"])
        ema5 = float(cur["ema5"])
        ema21 = float(cur["ema21"])
        ema44 = float(cur["ema44"])

        # Keep re-entry memory updated before checking decision rules.
        # This ensures pullback signals are remembered and can be consumed later.
        self._update_reentry_flags(c, ema5, ema21, has_active_position=position is not None)

        # If a position is active, only exit logic is evaluated.
        if position is not None:
            return self._evaluate_exit(position, c, color, ema5, ema21, ema44)

        # Fresh trend-following entries.
        ce_fresh = color == "green" and c > ema5 and c > ema21 and c > ema44
        pe_fresh = color == "red" and c < ema5 and c < ema21 and c < ema44

        # Pullback re-entries (flags must be armed earlier while flat).
        # These are "continuation after pullback" style entries.
        ce_re_ema5 = self.ce_pullback_ema5 and color == "green" and c > ema5
        ce_re_ema21 = self.ce_pullback_ema21 and color == "green" and c > ema21
        pe_re_ema5 = self.pe_pullback_ema5 and color == "red" and c < ema5
        pe_re_ema21 = self.pe_pullback_ema21 and color == "red" and c < ema21

        long_reentry = ce_re_ema5 or ce_re_ema21
        short_reentry = pe_re_ema5 or pe_re_ema21

        # Re-entry is direction-gated by the previous closed trade direction.
        # Fresh entries are NOT blocked by this gate.
        allow_long_reentry = long_reentry and self.previous_trade_direction in ("", "LONG")
        allow_short_reentry = short_reentry and self.previous_trade_direction in ("", "SHORT")

        long_entry_trigger = ce_fresh or allow_long_reentry
        short_entry_trigger = pe_fresh or allow_short_reentry

        if long_entry_trigger:
            stop = float(prev2["low"])
            if stop < c:
                # Entry consumed bullish pullback flags.
                self.ce_pullback_ema5 = False
                self.ce_pullback_ema21 = False
                return RenkoDecision(
                    action="ENTER_LONG",
                    entry_underlying=c,
                    stop_underlying=stop,
                    signal_triggered=True,
                )
            # Signal happened, but stop validation rejected actual entry.
            return RenkoDecision(signal_triggered=True)

        if short_entry_trigger:
            stop = float(prev2["high"])
            if stop > c:
                # Entry consumed bearish pullback flags.
                self.pe_pullback_ema5 = False
                self.pe_pullback_ema21 = False
                return RenkoDecision(
                    action="ENTER_SHORT",
                    entry_underlying=c,
                    stop_underlying=stop,
                    signal_triggered=True,
                )
            # Signal happened, but stop validation rejected actual entry.
            return RenkoDecision(signal_triggered=True)

        return self._hold()
