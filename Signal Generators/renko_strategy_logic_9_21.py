"""
Standalone Renko strategy logic variant using only 2 EMAs for signal generation:
- EMA 9
- EMA 21

This file is intentionally self-contained.
It does not import trading logic from `renko_strategy_logic.py`.

Public API parity with the original file:
1. The class names are the same.
2. The top-level function names are the same.
3. The key signal-engine method names are the same.
4. This lets caller files switch imports with minimal code changes.

Beginner quick-start:
1. Pass normal OHLC candles to `build_renko_with_indicators(...)`.
2. This file will build Renko candles and add `ema9` and `ema21`.
3. Create one `RenkoSignalEngine()` object and keep it alive across candles.
4. For each newly formed Renko candle, call `evaluate_candle(...)`.
5. If a trade is open, pass `RenkoPositionContext`; otherwise pass `None`.
6. Use the returned `RenkoDecision.action` to decide whether to enter, exit, or hold.

Why this file exists:
1. Keep the 9/21 EMA strategy as a separate variant.
2. Let you import it independently in backtest, front-test, or live files.
3. Avoid accidental coupling with the original 5/21/44 EMA strategy file.

What this file contains:
1. ATR utility.
2. Renko builder utility.
3. Renko + indicator builder using ATR(21) as box size.
4. Stateful signal engine for entries, re-entries, and exits.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class RenkoPositionContext:
    """
    Minimal snapshot of an already-open trade.

    The signal engine only needs these values to evaluate exit conditions:
    - `direction`: LONG or SHORT
    - `entry_underlying`: underlying price at entry
    - `stop_underlying`: fixed stop on underlying
    - `rr_armed`: whether RR-based opposite-color exit is already armed
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

    # Final action the strategy wants to take on the latest Renko candle.
    action: str = "HOLD"
    # Underlying entry price suggested by the signal engine.
    entry_underlying: float = 0.0
    # Underlying stop price suggested by the signal engine.
    stop_underlying: float = 0.0
    # Exit reason is useful for logs and backtest analysis.
    exit_reason: str = ""
    # Updated RR state returned back to caller.
    rr_armed: bool = False
    # True when a valid signal was seen, even if actual order was later rejected.
    signal_triggered: bool = False


def atr(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """
    Calculate ATR using a simple moving average of True Range.

    Why ATR is useful here:
    - ATR measures recent volatility.
    - A higher ATR produces larger Renko box sizes.
    - A lower ATR produces smaller Renko box sizes.

    True Range for each row is the maximum of:
    1. High - Low
    2. abs(High - Previous Close)
    3. abs(Low - Previous Close)

    In this variant, `build_renko_with_indicators()` uses the latest ATR(21)
    value as the Renko box size.
    """
    # Previous close is needed because price can gap between candles.
    # Without this, ATR would underestimate volatility after large jumps.
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
    box_size: float,
) -> pd.DataFrame:
    """
    Build Renko bricks from close prices using explicit trigger levels.

    Trigger rules:
    - Print next green brick only if price >= previous brick high + box_size
    - Print next red brick only if price <= previous brick low - box_size

    Important beginner note:
    - One normal candle can print multiple Renko bricks if price moves a lot.
    - The output format is designed to be easy for the signal engine to consume.
    """
    # Convert to simple lists because the brick-printing loop is easier to reason
    # about as sequential values than as vectorized DataFrame logic.
    closes = df["close"].tolist()
    times = df["timestamp"].tolist()
    if not closes:
        return pd.DataFrame()

    # Before the first printed brick exists, we anchor Renko to the first close.
    # This gives the builder a starting reference even though no real brick
    # has been printed yet.
    last_brick_open = closes[0]
    last_brick_close = closes[0]
    rows = []

    for i in range(1, len(closes)):
        # `price` is the current source close we are testing against Renko triggers.
        price = closes[i]
        # Every brick printed from this move gets the same timestamp as the source row.
        ts = times[i]

        # Keep printing bricks until the current price no longer crosses
        # the next up/down trigger.
        while True:
            # The previous brick's high and low define the next valid breakout points.
            # Price must fully cross one of these thresholds before a new brick is printed.
            prev_high = max(last_brick_open, last_brick_close)
            prev_low = min(last_brick_open, last_brick_close)
            up_trigger = prev_high + box_size
            down_trigger = prev_low - box_size

            if price >= up_trigger:
                # A green brick starts at previous high and closes one box above it.
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
                # A red brick starts at previous low and closes one box below it.
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

            # Stop printing when current price is inside the trigger band.
            break

    return pd.DataFrame(rows)


def build_renko_with_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLC candles into Renko candles plus EMA indicators.

    Steps:
    1. Compute ATR(21).
    2. Use the latest ATR value as Renko box size.
    3. Build Renko bricks from close prices.
    4. Add EMA(9) and EMA(21) on Renko close.

    Output columns include:
    - timestamp
    - open, high, low, close
    - color
    - ema9
    - ema21
    - box_size
    """
    # ATR(21) is the defining difference in this variant.
    # We use the most recent ATR value as today's Renko box size.
    atr_series = atr(ohlc, 21)
    # box_size = float(atr_series.iloc[-1])
    box_size = 12.5
    if pd.isna(box_size) or box_size <= 0:
        # ATR can be invalid during warm-up or when incoming data is malformed.
        return pd.DataFrame()

    # Build Renko candles first, then calculate EMAs on Renko closes.
    # This is important: the EMAs are based on Renko candles, not raw OHLC candles.
    renko = build_renko_from_close(ohlc, box_size)
    if renko.empty:
        return renko

    # These two EMAs are the only trend filters used by this variant.
    renko["ema9"] = renko["close"].ewm(span=9, adjust=False).mean()
    renko["ema21"] = renko["close"].ewm(span=21, adjust=False).mean()
    renko["box_size"] = box_size
    return renko


class RenkoSignalEngine:
    """
    Stateful signal engine for entries, re-entries, and exits.

    This engine carries memory across candles:
    - Pullback flags for EMA9/EMA21 zones
    - Direction of the previously closed trade for re-entry gating

    Why state matters:
    - Fresh entries depend only on the current candle.
    - Re-entries depend on what happened earlier during the pullback.
    - RR-based exits depend on whether the trade had already reached RR >= 1.5.
    """

    def __init__(self):
        # Bullish pullback flags.
        # These are armed while flat if price pulls below EMA 9 or EMA 21.
        self.ce_pullback_ema9 = False
        self.ce_pullback_ema21 = False

        # Bearish pullback flags.
        # These are armed while flat if price pulls above EMA 9 or EMA 21.
        self.pe_pullback_ema9 = False
        self.pe_pullback_ema21 = False

        # Last closed trade direction used to control re-entry direction.
        # This does not block fresh trend entries. It only affects re-entries.
        self.previous_trade_direction = ""

    def update_previous_trade_direction(self, direction: str) -> None:
        """
        Store the last closed trade direction for re-entry gating.

        Only LONG and SHORT are accepted to avoid bad internal state.
        """
        direction_txt = str(direction).strip().upper()
        if direction_txt in ("LONG", "SHORT"):
            self.previous_trade_direction = direction_txt

    def reset_reentry_flags(self) -> None:
        """
        Clear all bullish and bearish pullback memory flags.

        Callers usually use this only when they want to reset strategy state,
        for example after a risk halt or at a session reset.
        """
        self.ce_pullback_ema9 = False
        self.ce_pullback_ema21 = False
        self.pe_pullback_ema9 = False
        self.pe_pullback_ema21 = False

    def _update_reentry_flags(
        self,
        c: float,
        ema9: float,
        ema21: float,
        has_active_position: bool,
    ) -> None:
        """
        Update pullback memory while flat.

        Rule:
        - If close falls below EMA 9 or EMA 21, arm bullish pullback flags.
        - If close rises above EMA 9 or EMA 21, arm bearish pullback flags.
        - If a trade is already open, do not touch these flags.

        Beginner intuition:
        - A bullish pullback means an uptrend dipped down first, then may resume.
        - A bearish pullback means a downtrend bounced up first, then may resume.
        """
        if has_active_position:
            return

        if c < ema9:
            self.ce_pullback_ema9 = True
        if c < ema21:
            self.ce_pullback_ema21 = True
        if c > ema9:
            self.pe_pullback_ema9 = True
        if c > ema21:
            self.pe_pullback_ema21 = True

    @staticmethod
    def _hold(rr_armed: bool = False) -> RenkoDecision:
        """Return a standard no-action decision."""
        return RenkoDecision(action="HOLD", rr_armed=rr_armed)

    def _evaluate_exit(
        self,
        position: RenkoPositionContext,
        c: float,
        color: str,
        ema9: float,
        ema21: float,
    ) -> RenkoDecision:
        """
        Evaluate exit conditions for an already-open position.

        Exit triggers:
        1. Stop hit
        2. EMA trend invalidation
        3. Opposite color candle after RR >= 1.5 has been armed
        """
        # Keep a local copy and return the updated state back to the caller.
        # The engine itself does not mutate the external position object directly.
        rr_armed = bool(position.rr_armed)

        if position.direction == "LONG":
            # For a long trade, risk is entry minus stop.
            risk = float(position.entry_underlying) - float(position.stop_underlying)
            # RR tells us how many units of original risk the trade has made.
            rr = (c - float(position.entry_underlying)) / risk if risk > 0 else -999.0
            if rr >= 1.5:
                rr_armed = True

            # Long trade becomes invalid if price breaks stop,
            # loses both EMAs, or prints opposite color after RR arm.
            stop_hit = c <= float(position.stop_underlying)
            trend_exit = c < ema9 and c < ema21
            rr_exit = rr_armed and color == "red"
            if stop_hit or trend_exit or rr_exit:
                reason = "STOP" if stop_hit else ("EMA_EXIT" if trend_exit else "RR_RED_CANDLE")
                return RenkoDecision(action="EXIT", exit_reason=reason, rr_armed=rr_armed)
            return self._hold(rr_armed=rr_armed)

        if position.direction == "SHORT":
            # For a short trade, risk is stop minus entry.
            risk = float(position.stop_underlying) - float(position.entry_underlying)
            # RR tells us how many units of original risk the trade has made.
            rr = (float(position.entry_underlying) - c) / risk if risk > 0 else -999.0
            if rr >= 1.5:
                rr_armed = True

            # Short trade becomes invalid if price breaks stop,
            # regains both EMAs, or prints opposite color after RR arm.
            stop_hit = c >= float(position.stop_underlying)
            trend_exit = c > ema9 and c > ema21
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
        - `renko`: DataFrame with close/color/ema9/ema21 columns
        - `position`: open-position context; pass None when flat

        Output:
        - `RenkoDecision` with action and relevant payload fields
        """
        # We need at least 3 rows so stop anchoring and state logic are safe.
        # If the caller passes too little data, safest output is HOLD.
        if renko is None or len(renko) < 3:
            return self._hold(rr_armed=bool(position.rr_armed) if position else False)

        cur = renko.iloc[-1]
        # The previous printed Renko candle is used for initial stop anchoring.
        # The variable name `prev2` is kept to stay aligned with the original file's style.
        prev2 = renko.iloc[-2]

        c = float(cur["close"])
        color = str(cur["color"])
        ema9 = float(cur["ema9"])
        ema21 = float(cur["ema21"])

        # Update pullback memory before evaluating decision rules.
        # This ensures the engine remembers whether a pullback happened earlier.
        self._update_reentry_flags(c, ema9, ema21, has_active_position=position is not None)

        # If a position is already open, this method should only evaluate exit logic.
        if position is not None:
            return self._evaluate_exit(position, c, color, ema9, ema21)

        # Fresh trend-following entries.
        # Long requires green candle and close above both EMAs.
        # Short requires red candle and close below both EMAs.
        ce_fresh = color == "green" and c > ema9 and c > ema21
        pe_fresh = color == "red" and c < ema9 and c < ema21

        # Pullback re-entries.
        # Example long idea:
        # - price had dipped below an EMA earlier while flat
        # - now a green Renko candle closes back above that EMA
        ce_re_ema9 = self.ce_pullback_ema9 and color == "green" and c > ema9
        ce_re_ema21 = self.ce_pullback_ema21 and color == "green" and c > ema21
        pe_re_ema9 = self.pe_pullback_ema9 and color == "red" and c < ema9
        pe_re_ema21 = self.pe_pullback_ema21 and color == "red" and c < ema21

        long_reentry = ce_re_ema9 or ce_re_ema21
        short_reentry = pe_re_ema9 or pe_re_ema21

        # Re-entry is direction-gated by the most recently closed trade.
        # Fresh entries are not blocked by this rule.
        allow_long_reentry = long_reentry and self.previous_trade_direction in ("", "LONG")
        allow_short_reentry = short_reentry and self.previous_trade_direction in ("", "SHORT")

        long_entry_trigger = ce_fresh or allow_long_reentry
        short_entry_trigger = pe_fresh or allow_short_reentry

        if long_entry_trigger:
            # Long stop is anchored to the low of the previously printed Renko candle.
            stop = float(prev2["low"])
            if stop < c:
                # Once the long entry is taken, clear bullish pullback memory.
                self.ce_pullback_ema9 = False
                self.ce_pullback_ema21 = False
                return RenkoDecision(
                    action="ENTER_LONG",
                    entry_underlying=c,
                    stop_underlying=stop,
                    signal_triggered=True,
                )
            # The signal condition happened, but stop validation rejected actual entry.
            return RenkoDecision(signal_triggered=True)

        if short_entry_trigger:
            # Short stop is anchored to the high of the previously printed Renko candle.
            stop = float(prev2["high"])
            if stop > c:
                # Once the short entry is taken, clear bearish pullback memory.
                self.pe_pullback_ema9 = False
                self.pe_pullback_ema21 = False
                return RenkoDecision(
                    action="ENTER_SHORT",
                    entry_underlying=c,
                    stop_underlying=stop,
                    signal_triggered=True,
                )
            # The signal condition happened, but stop validation rejected actual entry.
            return RenkoDecision(signal_triggered=True)

        return self._hold()
