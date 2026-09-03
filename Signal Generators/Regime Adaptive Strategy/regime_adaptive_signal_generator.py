"""Regime-adaptive router: opening-range breakout in trend, VWAP fade in range.

WHAT IT DOES
------------
One strategy that changes its mind about WHICH rule applies, based on ADX:

    ADX missing            -> HOLD          (fail closed; never guess the regime)
    ADX >= trend threshold -> BREAKOUT      (opening-range break + VWAP alignment)
    ADX <  trend threshold -> MEAN REVERSION (fade an extension away from VWAP)

Both branches BUY options only: a bullish decision buys an ATM CE, a bearish one
an ATM PE. Neither ever sells.

HOW THE ROUTING IS MADE VISIBLE
-------------------------------
The worker factory calls exactly ONE `build_*_with_indicators` and ONE
`evaluate_candle`, so the builder computes BOTH branches every bar and then
collapses the active one into the repo's standard column contract
(`long_setup` / `short_setup` / `*_entry_price` / `*_stop_from_setup` /
`*_target_from_setup`). The chosen branch is left in the frame as `regime` and
`active_branch`, which means the router's decision is inspectable in the data
and testable without running the engine at all.

DELIBERATE DEVIATIONS FROM THE UPSTREAM PROJECT
-----------------------------------------------
- The upstream router has a third branch (OI/liquidity momentum). It is not
  ported: it needs the option chain, which this factory contract cannot pass to
  an engine, and the chain endpoint is already rate-budgeted elsewhere.
- The upstream bid/ask-spread veto IS implemented, but in the WORKER, not here:
  it needs the option chain, and this contract only ever receives a DataFrame.
  See `_spread_gate_allows_entry` in the master file.
- The upstream liquidity, India VIX and breadth vetoes are still absent — by
  CHOICE, not for want of data: the source runs on Dhan too and reads them from
  ordinary REST calls. See REGIME_PORTING_NOTES.md for where each one comes from.
- VWAP is an equal-weight session proxy in live/paper sessions, because the live
  feed carries no volume. `vwap_is_proxy` records this per bar.

Adapted from the MIT-licensed
`workratananmol-hub/nifty-options-paper-trading-bot` (`src/strategies/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from regime_candidates import attach_breakout_columns, attach_mean_reversion_columns

# Every shared indicator comes through `regime_common`, which owns this folder's
# single sys.path bootstrap -- see its module docstring. These are the same
# objects as `misc_strategy_common.adx` / `.atr`, not local reimplementations.
from regime_common import adx as adx_indicator
from regime_common import atr as atr_indicator
from regime_common import finite, prepare_session_frame, require_columns, validate_finite_config

BREAKOUT_BRANCH = "BREAKOUT"
MEAN_REVERSION_BRANCH = "MEANREV"
NO_BRANCH = "NONE"


@dataclass(frozen=True)
class RegimeAdaptiveConfig:
    """Tunable settings for the regime-adaptive router."""

    adx_period: int = 14                    # ADX lookback (regime detector)
    atr_period: int = 14                    # ATR lookback (breakout buffer + fade distance)
    opening_range_minutes: int = 15         # first N minutes that define the session range
    adx_trend_threshold: float = 20.0       # at/above this -> breakout branch
    adx_meanrev_ceiling: float = 25.0       # at/above this the fade stands down entirely
    breakout_buffer_atr_mult: float = 0.05  # break buffer as a multiple of ATR
    breakout_buffer_min_points: float = 2.0 # ...with this floor in index points
    # Fade distance: max(atr * mult, min_points). Both values match the upstream
    # `max(ctx.atr * 0.6, 15.0)` -- see REGIME_PORTING_NOTES.md, which records the
    # earlier 1.5-with-no-floor divergence and why it was corrected.
    meanrev_atr_mult: float = 0.6           # how far from VWAP counts as "extended"
    meanrev_min_points: float = 15.0        # ...with this floor in index points
    stop_loss_pct: float = 0.015            # protective stop, 1.5% from entry
    target_pct: float = 0.03                # profit target, 3% from entry

    def __post_init__(self) -> None:
        validate_finite_config(self)
        positive_ints = {
            "adx_period": self.adx_period,
            "atr_period": self.atr_period,
            "opening_range_minutes": self.opening_range_minutes,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Config values must be positive: {', '.join(invalid)}")
        if float(self.stop_loss_pct) <= 0.0 or float(self.target_pct) <= 0.0:
            raise ValueError("stop_loss_pct and target_pct must be greater than zero.")
        if float(self.stop_loss_pct) >= 1.0 or float(self.target_pct) >= 1.0:
            raise ValueError("stop_loss_pct and target_pct are percentages below 1.0.")
        if float(self.adx_trend_threshold) <= 0.0 or float(self.adx_meanrev_ceiling) <= 0.0:
            raise ValueError("ADX thresholds must be greater than zero.")
        if float(self.breakout_buffer_min_points) <= 0.0 or float(self.meanrev_atr_mult) <= 0.0:
            raise ValueError("Breakout buffer floor and fade distance must be greater than zero.")
        if float(self.meanrev_min_points) <= 0.0:
            raise ValueError("Fade distance floor must be greater than zero.")


@dataclass
class RegimeAdaptivePositionContext:
    """Snapshot of an open trade so the engine can manage its exit.

    NOTE the field set is fixed by the worker factory: there is nowhere to record
    WHICH branch opened the position. Exits are therefore branch-agnostic — see
    `_evaluate_exit`.
    """

    direction: str
    entry_underlying: float
    stop_underlying: float
    target_underlying: float


@dataclass
class RegimeAdaptiveDecision:
    """Standard decision object returned by the engine."""

    action: str = "HOLD"            # HOLD, ENTER_LONG, ENTER_SHORT, or EXIT
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    target_underlying: float = 0.0
    exit_underlying: float = 0.0
    exit_reason: str = ""           # STOP, TARGET, OPPOSITE_SIGNAL
    reason: str = ""
    signal_triggered: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def build_regime_adaptive_with_indicators(
    ohlc: pd.DataFrame,
    config: RegimeAdaptiveConfig | None = None,
) -> pd.DataFrame:
    """Compute both branches, pick the active one, and publish standard columns."""
    config = config or RegimeAdaptiveConfig()
    frame = prepare_session_frame(ohlc, config.opening_range_minutes)
    if frame.empty:
        return frame

    frame["atr"] = atr_indicator(frame, config.atr_period)
    frame["adx"] = adx_indicator(frame, config.adx_period)

    frame = attach_breakout_columns(
        frame,
        buffer_atr_mult=config.breakout_buffer_atr_mult,
        buffer_min_points=config.breakout_buffer_min_points,
        stop_loss_pct=config.stop_loss_pct,
        target_pct=config.target_pct,
    )
    frame = attach_mean_reversion_columns(
        frame,
        atr_mult=config.meanrev_atr_mult,
        min_points=config.meanrev_min_points,
        adx_ceiling=config.adx_meanrev_ceiling,
        stop_loss_pct=config.stop_loss_pct,
        target_pct=config.target_pct,
    )

    # The regime decision itself. A missing ADX is NO_BRANCH, not a default —
    # that is the upstream "missing ADX -> no-trade" rule.
    adx_values = frame["adx"].astype(float)
    trending = adx_values >= float(config.adx_trend_threshold)
    frame["regime"] = NO_BRANCH
    frame.loc[adx_values.notna() & trending, "regime"] = BREAKOUT_BRANCH
    frame.loc[adx_values.notna() & ~trending, "regime"] = MEAN_REVERSION_BRANCH
    frame["active_branch"] = frame["regime"]

    # Collapse the active branch into the contract every engine in this repo
    # reads, so the routing lives in the data rather than in engine branching.
    #
    # NOTE the boolean columns are named EXPLICITLY rather than detected by
    # suffix: `long_stop_from_setup` also ends in "setup", so a suffix test would
    # cast the stop and target levels to booleans and hand `enter_position` a
    # stop of 0.0 or 1.0.
    use_breakout = frame["regime"] == BREAKOUT_BRANCH
    use_meanrev = frame["regime"] == MEAN_REVERSION_BRANCH
    boolean_columns = {"long_setup", "short_setup"}
    for suffix in (
        "long_setup",
        "short_setup",
        "long_entry_price",
        "short_entry_price",
        "long_stop_from_setup",
        "short_stop_from_setup",
        "long_target_from_setup",
        "short_target_from_setup",
    ):
        is_flag = suffix in boolean_columns
        collapsed = pd.Series(False if is_flag else 0.0, index=frame.index, dtype=object)
        collapsed = collapsed.where(~use_breakout, frame[f"br_{suffix}"])
        collapsed = collapsed.where(~use_meanrev, frame[f"mr_{suffix}"])
        frame[suffix] = collapsed.astype(bool) if is_flag else collapsed.astype(float)
    return frame


class RegimeAdaptiveSignalEngine:
    """Decision engine: routes on regime, then acts on the collapsed columns."""

    def __init__(self, config: RegimeAdaptiveConfig | None = None) -> None:
        self.config = config or RegimeAdaptiveConfig()

    def minimum_history_bars(self) -> int:
        # ADX needs roughly twice its period to settle; add the opening-range
        # window so the first evaluable bar always has a finished range too.
        return int(self.config.adx_period) * 2 + int(self.config.opening_range_minutes) + 2

    @staticmethod
    def _hold(reason: str = "", *, signal_triggered: bool = False) -> RegimeAdaptiveDecision:
        return RegimeAdaptiveDecision(
            action="HOLD",
            signal_triggered=signal_triggered,
            debug={"reason": reason} if reason else {},
        )

    @staticmethod
    def _direction(direction: str) -> str:
        return str(direction).strip().upper()

    def _evaluate_exit(
        self, current: pd.Series, position: RegimeAdaptivePositionContext
    ) -> RegimeAdaptiveDecision:
        """Stop, then target, then an opposing setup on the CURRENTLY active branch.

        Branch-agnostic by necessity: the position context cannot carry which
        branch opened the trade. If the regime flips mid-trade, the stop and
        target still govern, so a flip can never strand a position.
        """
        direction = self._direction(position.direction)
        high = float(current["high"])
        low = float(current["low"])
        close = float(current["close"])
        stop = float(position.stop_underlying)
        target = float(position.target_underlying)

        if direction == "LONG":
            if low <= stop:
                return RegimeAdaptiveDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if target > 0.0 and high >= target:
                return RegimeAdaptiveDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if bool(current.get("short_setup", False)):
                return RegimeAdaptiveDecision(
                    action="EXIT", exit_underlying=close, exit_reason="OPPOSITE_SIGNAL"
                )
            return self._hold()

        if direction == "SHORT":
            if high >= stop:
                return RegimeAdaptiveDecision(action="EXIT", exit_underlying=stop, exit_reason="STOP")
            if target > 0.0 and low <= target:
                return RegimeAdaptiveDecision(action="EXIT", exit_underlying=target, exit_reason="TARGET")
            if bool(current.get("long_setup", False)):
                return RegimeAdaptiveDecision(
                    action="EXIT", exit_underlying=close, exit_reason="OPPOSITE_SIGNAL"
                )
            return self._hold()

        return self._hold("unknown_direction")

    def evaluate_candle(
        self,
        candles_with_indicators: pd.DataFrame,
        position: RegimeAdaptivePositionContext | None = None,
    ) -> RegimeAdaptiveDecision:
        """Evaluate the newest completed bar."""
        if candles_with_indicators is None or candles_with_indicators.empty:
            return self._hold("empty_frame")
        require_columns(candles_with_indicators, ["open", "high", "low", "close"])
        current = candles_with_indicators.iloc[-1]

        if position is not None:
            return self._evaluate_exit(current, position)

        regime = str(current.get("regime", NO_BRANCH))
        if regime == NO_BRANCH:
            # Missing ADX -> we do not know the regime -> we do not trade.
            return self._hold("regime_unknown")

        debug = {
            "regime": regime,
            "adx": current.get("adx"),
            "atr": current.get("atr"),
            "vwap_is_proxy": bool(current.get("vwap_is_proxy", False)),
        }

        if bool(current.get("long_setup", False)):
            entry = float(current["long_entry_price"])
            stop = float(current["long_stop_from_setup"])
            target = float(current["long_target_from_setup"])
            # Same gate the other ported strategies apply: the master sizes the
            # trade off the entry-to-stop distance, so a NaN or a mis-ordered
            # level must refuse the trade rather than reach `enter_position`.
            # It is not merely defensive -- a fade whose VWAP has converged onto
            # the close produces target == entry, a zero-width trade.
            if all(finite(value) for value in (entry, stop, target)) and stop < entry < target:
                return RegimeAdaptiveDecision(
                    action="ENTER_LONG",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason=f"{regime}_long",
                    signal_triggered=True,
                    debug=debug,
                )
            # The signal DID fire; only execution was refused. Saying so keeps a
            # rejected level visible in the logs instead of looking like no setup.
            return self._hold(f"{regime}_long_invalid_levels", signal_triggered=True)
        if bool(current.get("short_setup", False)):
            entry = float(current["short_entry_price"])
            stop = float(current["short_stop_from_setup"])
            target = float(current["short_target_from_setup"])
            # Mirrored: a short's stop sits ABOVE entry and its target BELOW.
            if all(finite(value) for value in (entry, stop, target)) and target < entry < stop:
                return RegimeAdaptiveDecision(
                    action="ENTER_SHORT",
                    entry_underlying=entry,
                    stop_underlying=stop,
                    target_underlying=target,
                    reason=f"{regime}_short",
                    signal_triggered=True,
                    debug=debug,
                )
            return self._hold(f"{regime}_short_invalid_levels", signal_triggered=True)
        return self._hold(f"{regime}_no_setup")
