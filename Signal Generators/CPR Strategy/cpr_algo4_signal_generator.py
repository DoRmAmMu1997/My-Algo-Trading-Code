"""
NIFTY CPR Algo 4 Signal Generator -- the deterministic "Intraday SRSI VWAP" playbook.

============================== What Algo 4 trades ==============================
The operator's "Intraday SRSI VWAP" document splits every day in two:

1. DAY TYPE, decided ONCE at 09:30 from the close of the 09:25 5-minute candle
   (the close of the first 15 minutes):
     - the "zone" runs from MIN(S1, PDL) up to MAX(R1, PDH) (previous-session
       levels -- PDH/PDL are the previous day's high/low);
     - close ABOVE the zone  -> TRENDING, direction UP   (we buy calls);
     - close BELOW the zone  -> TRENDING, direction DOWN (we buy puts);
     - close inside the zone -> SIDEWAYS.
   The day type is fixed for the rest of the session. On a trending day only the
   DIRECTION can change later (see "structure flips" below).

2. SIDEWAYS days trade Stochastic RSI (the "blue" %K and "red" %D lines):
     - %K crosses above %D inside the oversold zone (<= 20) -> buy a CALL,
       stop = the latest confirmed swing low of today's session;
     - %K crosses below %D inside the overbought zone (>= 80) -> buy a PUT,
       stop = the latest confirmed swing high.
   The trade also exits when Stochastic RSI crosses back the other way inside
   the opposite zone.

3. TRENDING days trade VWAP pullbacks in the day's direction ("continuation"):
     - UP: a candle closes below VWAP, then the next candle closes back above it
       with at least 40% of its body above VWAP (or: a red candle supported on
       VWAP followed by a green candle supported on VWAP) -> buy a CALL, stop =
       the entry candle's low. DOWN is the mirror image.
     - Filters: RSI > 45 for calls (< 65 for puts), EMA5 above EMA20 with both
       rising for calls (the mirror for puts).
   STRUCTURE FLIPS: when confirmed swings print a lower high AND a lower low on
   an UP day (or a higher high AND a higher low on a DOWN day), the direction
   flips. An open position is cut, and the next trade needs the "reversal"
   sequence in the new direction: (for a new DOWN) a close below VWAP, then a
   green candle closing above VWAP, then a red candle closing below VWAP with at
   least 40% of its body below. After that reversal trade, continuation setups
   resume.

4. RISK AND TARGETS for every trade (the doc's general rules):
     - skip the trade if the stop is more than 30 NIFTY points away;
     - skip it if the next CPR level (less a 2-point buffer: "cut the trade 2
       points short of the next level") is closer than 1:1;
     - TARGET exit mode books at the earlier of 1:1 and that buffered level
       (sideways trades may also book at the first-30-minute high/low when the
       operator turns that option on). Because of the 1:1 entry rule, the next
       level is never nearer than 1:1, so in practice TARGET mode books at 1R;
     - TRAIL exit mode instead moves the stop to breakeven at the first
       milestone and then trails candle by candle (exit when a candle closes
       below the previous candle's low, or above its high for puts). Reversal
       trades trail in two stages: breakeven first, then a locked 1R once
       2R (or the following level) is reached;
     - R2 (calls) / S2 (puts), less the 2-point buffer, always books the trade;
     - no new entries on a candle that ends at or after 15:00.
   On a trending long, a red candle at R1 followed by a green candle that
   reclaims R1 allows ONE equal-size add ("increase the quantity").

============================== How the pieces fit ==============================
This module is PURE: no broker, no clock, no environment variables. The live
worker in the front-test master and the backtest both drive exactly this code:

  - `build_cpr_algo4_frame` turns 1-minute candles into completed 5-minute rows
    carrying CPR levels, VWAP, RSI, EMAs (all from the shared
    `cpr_strategy_logic` builder that CPR / CPR Algo 3 already use) plus the
    Stochastic RSI this strategy adds.
  - `CPRAlgo4Engine.on_bar` is fed EVERY completed 5-minute row in order and
    answers ENTER_LONG / ENTER_SHORT / EXIT / SCALE_IN / HOLD. It keeps the
    per-session memory (day type, trend direction, swing points, the reversal
    sequence) so the caller does not have to.
  - `check_intrabar_exit` is the stop / target / final-level check. The live
    worker calls it on every poll with the spot price; the backtest calls it
    with each candle's high and low.

It deliberately does NOT import the CPR AI agent package (that one needs the
optional AI dependencies); the few lines of Stochastic RSI maths live here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any

import numpy as np
import pandas as pd
from cpr_strategy_logic import CPRStrategyConfig, build_cpr_with_indicators

# ---------------------------------------------------------------------------
# Vocabulary shared with the worker, the backtest and the tests.
# ---------------------------------------------------------------------------
EXIT_MODES = ("TARGET", "TRAIL")

REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_TRENDING_UP = "TRENDING_UP"
REGIME_TRENDING_DOWN = "TRENDING_DOWN"
REGIME_NO_TRADE = "NO_TRADE"

PREMISE_SIDEWAYS = "SIDEWAYS_SRSI"
PREMISE_CONTINUATION = "TREND_CONTINUATION"
PREMISE_REVERSAL = "TREND_REVERSAL"

EXIT_STOP = "CPR_ALGO4_STOP"
EXIT_TARGET = "CPR_ALGO4_TARGET"
EXIT_FINAL = "CPR_ALGO4_FINAL_TARGET"
EXIT_SRSI = "CPR_ALGO4_SRSI_EXIT"
EXIT_STRUCTURE = "CPR_ALGO4_STRUCTURE_FLIP"
EXIT_TRAIL = "CPR_ALGO4_TRAIL"

# Previous-session levels that can serve as "the next CPR level", low -> high.
# PDH/PDL are included because the doc's day-type zone is built from them.
LEVEL_LADDER_COLUMNS = ("s2", "s1", "prev_low", "cpr_lower", "pivot", "cpr_upper", "prev_high", "r1", "r2")


@dataclass(frozen=True)
class CPRAlgo4Config:
    """Every Algo 4 setting in one beginner-friendly place.

    The numbers come straight from the operator's document. Only `exit_mode`,
    `first30_target`, `scale_in_enabled` and `entry_cutoff` are exposed as
    `.env` knobs by the master; the rest are fixed rules of the playbook and
    are dataclass fields so the backtest can experiment with them.
    """

    # RSI / EMA / VWAP / CPR maths come from the shared CPR config, so Algo 4
    # computes those indicators exactly like CPR and CPR Algo 3 do.
    indicator_config: CPRStrategyConfig = field(default_factory=CPRStrategyConfig)
    # TradingView Stochastic RSI defaults (the RSI length is indicator_config.rsi_period).
    srsi_stoch_period: int = 14
    srsi_k_period: int = 3
    srsi_d_period: int = 3
    srsi_oversold: float = 20.0
    srsi_overbought: float = 80.0
    # A swing is a bar whose high (low) beats `swing_window` bars on each side.
    swing_window: int = 2
    max_stop_points: float = 30.0
    level_buffer_points: float = 2.0
    min_vwap_body_fraction: float = 0.40
    rsi_long_min: float = 45.0
    rsi_short_max: float = 65.0
    r1_touch_buffer: float = 2.0
    bar_minutes: int = 5
    # The bar whose close decides the day type (09:25 bar = 09:25-09:30).
    regime_bar: time = time(9, 25)
    # Last bar of the first 30 minutes (09:40 bar = 09:40-09:45).
    first30_last_bar: time = time(9, 40)
    # No entry or add on a bar that ENDS at or after this time.
    entry_cutoff: time = time(15, 0)
    exit_mode: str = "TARGET"
    first30_target: bool = False
    scale_in_enabled: bool = True

    def __post_init__(self) -> None:
        numbers = {
            "srsi_oversold": self.srsi_oversold,
            "srsi_overbought": self.srsi_overbought,
            "max_stop_points": self.max_stop_points,
            "level_buffer_points": self.level_buffer_points,
            "min_vwap_body_fraction": self.min_vwap_body_fraction,
            "rsi_long_min": self.rsi_long_min,
            "rsi_short_max": self.rsi_short_max,
            "r1_touch_buffer": self.r1_touch_buffer,
        }
        bad = [name for name, value in numbers.items() if not math.isfinite(float(value))]
        if bad:
            raise ValueError("CPR Algo 4 configuration values must be finite: " + ", ".join(bad))
        periods = {
            "srsi_stoch_period": self.srsi_stoch_period,
            "srsi_k_period": self.srsi_k_period,
            "srsi_d_period": self.srsi_d_period,
            "swing_window": self.swing_window,
            "bar_minutes": self.bar_minutes,
        }
        bad = [name for name, value in periods.items() if int(value) <= 0]
        if bad:
            raise ValueError("CPR Algo 4 periods must be positive: " + ", ".join(bad))
        if float(self.max_stop_points) <= 0.0:
            raise ValueError("max_stop_points must be greater than zero.")
        if float(self.level_buffer_points) < 0.0 or float(self.r1_touch_buffer) < 0.0:
            raise ValueError("Level buffers cannot be negative.")
        if not (0.0 <= float(self.min_vwap_body_fraction) <= 1.0):
            raise ValueError("min_vwap_body_fraction must be between 0 and 1.")
        if not (0.0 <= float(self.srsi_oversold) < float(self.srsi_overbought) <= 100.0):
            raise ValueError("Require 0 <= srsi_oversold < srsi_overbought <= 100.")
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}, got {self.exit_mode!r}.")
        if not (self.regime_bar < self.first30_last_bar and self.regime_bar < self.entry_cutoff):
            raise ValueError("The regime bar must come before the first-30 window end and the entry cutoff.")


@dataclass
class CPRAlgo4TradePlan:
    """The spot-price geometry of one open Algo 4 trade.

    Created by `build_trade_plan` at entry. The engine mutates the trailing
    fields in place as completed bars arrive; `current_stop` only ever moves
    toward profit (see `ratchet_stop`). The caller owns quantities, contracts
    and fills -- this object is prices only.
    """

    direction: str  # "LONG" (bought a CALL) or "SHORT" (bought a PUT)
    premise: str
    entry: float
    risk: float
    original_stop: float
    current_stop: float
    first_milestone: float
    following_milestone: float
    target: float  # NaN in TRAIL mode (no fixed target)
    final_target: float  # buffered R2 / S2: always books
    exit_mode: str
    entry_bar_ts: pd.Timestamp | None = None
    trail_stage: str = "NONE"
    trail_armed: bool = False
    scale_in_used: bool = False

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    def ratchet_stop(self, candidate: float) -> None:
        """Move the stop toward profit only; never loosen protection."""
        if self.is_long:
            self.current_stop = max(self.current_stop, float(candidate))
        else:
            self.current_stop = min(self.current_stop, float(candidate))


@dataclass
class CPRAlgo4Decision:
    """One answer from the engine for one completed bar."""

    action: str = "HOLD"  # HOLD | ENTER_LONG | ENTER_SHORT | EXIT | SCALE_IN
    reason: str = ""
    plan: CPRAlgo4TradePlan | None = None
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Indicators and the 5-minute frame.
# ---------------------------------------------------------------------------
def stochastic_rsi(
    rsi: pd.Series,
    *,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """TradingView-style Stochastic RSI: returns (%K "blue", %D "red").

    raw = where RSI sits inside its own `stoch_period` high/low range (0-100);
    %K = SMA(raw, k_period); %D = SMA(%K, d_period). A perfectly flat RSI window
    has no range, so its value is left undefined (NaN) rather than faked as 0.
    """
    values = rsi.astype(float)
    lowest = values.rolling(stoch_period).min()
    highest = values.rolling(stoch_period).max()
    spread = (highest - lowest).replace(0.0, np.nan)
    raw = (values - lowest) / spread * 100.0
    k = raw.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return k, d


def build_cpr_algo4_frame(ohlc: pd.DataFrame, config: CPRAlgo4Config | None = None) -> pd.DataFrame:
    """Completed 5-minute rows with CPR levels, VWAP, RSI, EMAs and Stochastic RSI.

    Accepts the same 1-minute (or 5-minute) OHLC the other CPR strategies take.
    The shared builder keeps only COMPLETE 5-minute buckets; the live worker
    additionally drops the still-forming minute before calling this, so a
    bucket cannot be treated as complete during its fifth minute.
    Stochastic RSI runs over the whole multi-day history, so it is already
    warmed up at the open.
    """
    config = config or CPRAlgo4Config()
    frame = build_cpr_with_indicators(ohlc, config.indicator_config)
    if frame.empty:
        return frame
    frame["srsi_k"], frame["srsi_d"] = stochastic_rsi(
        frame["rsi"],
        stoch_period=config.srsi_stoch_period,
        k_period=config.srsi_k_period,
        d_period=config.srsi_d_period,
    )
    return frame


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------
def _num(row: Mapping[str, Any], key: str) -> float:
    """Read one numeric field as a float; missing or non-finite -> NaN."""
    value = row.get(key)
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def body_fraction_beyond_vwap(open_: float, close: float, vwap: float, *, above: bool) -> float:
    """Share (0-1) of the candle BODY that sits above (or below) VWAP.

    The doc's "40-50% of the candle above VWAP". A doji has no body, so it
    returns 0 and can never qualify.
    """
    if not _finite(open_, close, vwap):
        return 0.0
    top, bottom = max(open_, close), min(open_, close)
    body = top - bottom
    if body <= 0.0:
        return 0.0
    part = top - max(bottom, vwap) if above else min(top, vwap) - bottom
    return min(max(part, 0.0), body) / body


def classify_regime(close_0925: float, levels: Mapping[str, Any]) -> str:
    """Day type from the 09:25 close against the [MIN(S1,PDL), MAX(R1,PDH)] zone.

    The zone edges themselves count as inside (SIDEWAYS). Missing levels (for
    example, no previous session in the data) fail closed as NO_TRADE.
    """
    r1, pdh, s1, pdl = (_num(levels, key) for key in ("r1", "prev_high", "s1", "prev_low"))
    if not _finite(close_0925, r1, pdh, s1, pdl):
        return REGIME_NO_TRADE
    zone_top, zone_bottom = max(r1, pdh), min(s1, pdl)
    if close_0925 > zone_top:
        return REGIME_TRENDING_UP
    if close_0925 < zone_bottom:
        return REGIME_TRENDING_DOWN
    return REGIME_SIDEWAYS


def _earliest(long: bool, values: list[float]) -> float:
    """The level price reaches first: the lowest for a long, the highest for a short."""
    return min(values) if long else max(values)


def build_trade_plan(
    *,
    direction: str,
    premise: str,
    entry: float,
    stop: float,
    levels: Mapping[str, Any],
    config: CPRAlgo4Config,
    bar_ts: pd.Timestamp | None = None,
    first30_extreme: float | None = None,
) -> tuple[CPRAlgo4TradePlan | None, str]:
    """Apply the doc's risk rules and derive every price of a new trade.

    Returns `(plan, "")` when the trade is allowed, or `(None, reason)` when a
    rule rejects it:
      - `invalid_stop_geometry`: the stop is not on the losing side of entry;
      - `risk_too_wide`: the stop is more than `max_stop_points` away;
      - `no_next_level` / `next_level_under_one_r`: the next CPR level, cut
        `level_buffer_points` short, is missing or closer than 1:1;
      - `final_target_behind_entry`: buffered R2 (S2) is not ahead of entry.
    """
    long = direction == "LONG"
    if not _finite(entry, stop):
        return None, "missing_price"
    risk = entry - stop if long else stop - entry
    if risk <= 0.0:
        return None, "invalid_stop_geometry"
    if risk > config.max_stop_points:
        return None, "risk_too_wide"

    buffer = config.level_buffer_points
    ladder = [value for value in (_num(levels, key) for key in LEVEL_LADDER_COLUMNS) if math.isfinite(value)]
    # Each level is "cut" 2 points short; a level already inside that buffer
    # counts as reached and the one after it becomes the next level.
    if long:
        ahead = [level - buffer for level in ladder if level - buffer > entry]
    else:
        ahead = [level + buffer for level in ladder if level + buffer < entry]
    if not ahead:
        return None, "no_next_level"
    milestone = _earliest(long, ahead)
    if abs(milestone - entry) < risk:
        return None, "next_level_under_one_r"

    final_raw = _num(levels, "r2" if long else "s2")
    final_target = final_raw - buffer if long else final_raw + buffer
    if not math.isfinite(final_target) or (final_target <= entry if long else final_target >= entry):
        return None, "final_target_behind_entry"

    sign = 1.0 if long else -1.0
    one_r = entry + sign * risk
    two_r = entry + sign * 2.0 * risk
    first_milestone = _earliest(long, [one_r, milestone])
    beyond_first = [level for level in ahead if (level > first_milestone if long else level < first_milestone)]
    following_level = _earliest(long, beyond_first) if beyond_first else final_target
    following_milestone = _earliest(long, [two_r, following_level])

    target = math.nan
    if config.exit_mode == "TARGET":
        candidates = [one_r, milestone]
        # The optional first-30-minute extreme only counts when it lies ahead of entry.
        first30_ahead = (
            first30_extreme is not None
            and math.isfinite(first30_extreme)
            and (first30_extreme > entry if long else first30_extreme < entry)
        )
        if first30_ahead and first30_extreme is not None:
            candidates.append(float(first30_extreme))
        target = _earliest(long, candidates)

    plan = CPRAlgo4TradePlan(
        direction="LONG" if long else "SHORT",
        premise=premise,
        entry=float(entry),
        risk=float(risk),
        original_stop=float(stop),
        current_stop=float(stop),
        first_milestone=float(first_milestone),
        following_milestone=float(following_milestone),
        target=float(target),
        final_target=float(final_target),
        exit_mode=config.exit_mode,
        entry_bar_ts=bar_ts,
    )
    return plan, ""


def check_intrabar_exit(plan: CPRAlgo4TradePlan, *, high: float, low: float) -> tuple[str, float] | None:
    """Stop / target / final-level check for one price range.

    The stop is tested FIRST: when one range touches both, assume the loss
    (the conservative choice a candle cannot disprove). Returns
    `(reason, level)` or None. Live code passes the spot price as both `high`
    and `low`; the backtest passes the candle's range.
    """
    if plan.is_long:
        if low <= plan.current_stop:
            return EXIT_STOP, plan.current_stop
        if math.isfinite(plan.target) and high >= plan.target:
            return EXIT_TARGET, plan.target
        if high >= plan.final_target:
            return EXIT_FINAL, plan.final_target
        return None
    if high >= plan.current_stop:
        return EXIT_STOP, plan.current_stop
    if math.isfinite(plan.target) and low <= plan.target:
        return EXIT_TARGET, plan.target
    if low <= plan.final_target:
        return EXIT_FINAL, plan.final_target
    return None


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------
_BAR_FIELDS = ("open", "high", "low", "close", "vwap", "rsi", "ema5", "ema20", "srsi_k", "srsi_d")


class CPRAlgo4Engine:
    """Per-session state machine for the Intraday SRSI VWAP playbook.

    Feed it every completed 5-minute row, in time order, through `on_bar`.
    Pass the open trade's plan while a position is open (None when flat). Tell
    it about fills and exits with `on_entry_filled` / `on_exit` so the reversal
    sequence and the "no re-entry on the exit bar" rule stay correct.

    Everything resets when a row from a new session date arrives, so a restart
    only has to replay today's rows to rebuild the exact same state.
    """

    def __init__(self, config: CPRAlgo4Config | None = None) -> None:
        self.config = config or CPRAlgo4Config()
        self._reset_session(None)

    # -- session memory -------------------------------------------------------
    def _reset_session(self, session: date | None) -> None:
        self.session_date: date | None = session
        self.regime: str | None = None
        self.trend_dir: str | None = None  # "UP" / "DOWN" on trending days
        self.reversal_pending = False
        self.reversal_step = 0
        self.last_bar_ts: pd.Timestamp | None = None
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._swing_highs: list[float] = []
        self._swing_lows: list[float] = []
        self._prev: dict[str, float] | None = None
        self._first30_high = math.nan
        self._first30_low = math.nan
        self._first30_complete = False
        self._blocked_entry_bar: pd.Timestamp | None = None

    def on_entry_filled(self, plan: CPRAlgo4TradePlan) -> None:
        """Record a filled entry. A reversal fill re-enables continuation setups."""
        if plan.premise == PREMISE_REVERSAL:
            self.reversal_pending = False
            self.reversal_step = 0

    def on_exit(self, exit_bar_ts: Any) -> None:
        """Record the bar during which an exit happened: no new entry on that bar."""
        self._blocked_entry_bar = pd.Timestamp(exit_bar_ts)

    # -- per-bar entry point --------------------------------------------------
    def on_bar(self, row: Mapping[str, Any], plan: CPRAlgo4TradePlan | None = None) -> CPRAlgo4Decision:
        """Update the session state with one completed bar and return a decision.

        With `plan` given (a position is open) the answer is EXIT, SCALE_IN or
        HOLD; trailing-stop moves are written into `plan` directly. Without a
        plan the answer is ENTER_LONG, ENTER_SHORT or HOLD.
        """
        ts = pd.Timestamp(row["timestamp"])
        if ts.date() != self.session_date:
            self._reset_session(ts.date())
        bar = {name: _num(row, name) for name in _BAR_FIELDS}
        levels = {name: _num(row, name) for name in LEVEL_LADDER_COLUMNS}
        prev = self._prev

        self._update_regime(ts, bar["close"], levels)
        self._update_first30(ts, bar)
        flipped = self._update_structure(bar)

        if plan is not None:
            decision = self._manage(ts, bar, prev, levels, plan, flipped)
        else:
            reversal_ready = self._advance_reversal(bar, flipped)
            decision = self._entry(ts, bar, prev, levels, flipped, reversal_ready)
        if decision.action == "EXIT":
            self._blocked_entry_bar = ts

        self._prev = bar
        self.last_bar_ts = ts
        return decision

    # -- state updates ----------------------------------------------------------
    def _update_regime(self, ts: pd.Timestamp, close: float, levels: Mapping[str, float]) -> None:
        if self.regime is not None:
            return
        bar_time = ts.time()
        if bar_time == self.config.regime_bar:
            self.regime = classify_regime(close, levels)
            if self.regime == REGIME_TRENDING_UP:
                self.trend_dir = "UP"
            elif self.regime == REGIME_TRENDING_DOWN:
                self.trend_dir = "DOWN"
        elif bar_time > self.config.regime_bar:
            # The deciding bar is missing (data hole or late start): fail closed.
            self.regime = REGIME_NO_TRADE

    def _update_first30(self, ts: pd.Timestamp, bar: Mapping[str, float]) -> None:
        if self._first30_complete:
            return
        if ts.time() > self.config.first30_last_bar:
            self._first30_complete = True
            return
        high, low = bar["high"], bar["low"]
        if math.isfinite(high):
            self._first30_high = high if math.isnan(self._first30_high) else max(self._first30_high, high)
        if math.isfinite(low):
            self._first30_low = low if math.isnan(self._first30_low) else min(self._first30_low, low)
        if ts.time() == self.config.first30_last_bar:
            self._first30_complete = True

    def _update_structure(self, bar: Mapping[str, float]) -> str | None:
        """Confirm swings (no look-ahead) and return the new direction on a flip.

        A swing needs `swing_window` session bars on BOTH sides, so it is only
        confirmed `swing_window` bars after it prints. A flip needs BOTH swing
        comparisons to agree: lower high + lower low turns UP into DOWN, higher
        high + higher low turns DOWN into UP.
        """
        self._highs.append(bar["high"])
        self._lows.append(bar["low"])
        window = self.config.swing_window
        center = len(self._highs) - 1 - window
        if center >= window:
            highs = self._highs[center - window : center + window + 1]
            lows = self._lows[center - window : center + window + 1]
            pivot_high, pivot_low = self._highs[center], self._lows[center]
            others_high = highs[:window] + highs[window + 1 :]
            others_low = lows[:window] + lows[window + 1 :]
            if _finite(pivot_high, *others_high) and pivot_high > max(others_high):
                self._swing_highs.append(pivot_high)
            if _finite(pivot_low, *others_low) and pivot_low < min(others_low):
                self._swing_lows.append(pivot_low)

        if self.trend_dir is None or len(self._swing_highs) < 2 or len(self._swing_lows) < 2:
            return None
        higher = self._swing_highs[-1] > self._swing_highs[-2] and self._swing_lows[-1] > self._swing_lows[-2]
        lower = self._swing_highs[-1] < self._swing_highs[-2] and self._swing_lows[-1] < self._swing_lows[-2]
        new_dir = "DOWN" if self.trend_dir == "UP" and lower else "UP" if self.trend_dir == "DOWN" and higher else None
        if new_dir is not None:
            self.trend_dir = new_dir
            self.reversal_pending = True
            self.reversal_step = 0
        return new_dir

    def _advance_reversal(self, bar: Mapping[str, float], flipped: str | None) -> bool:
        """Walk the post-flip sequence; True when its final step completes on this bar.

        For a new DOWN direction: (1) a close below VWAP, then (2) a green
        candle closing above VWAP, then (3) a red candle closing below VWAP with
        enough body below it. Steps must happen in that order on later bars, but
        need not be consecutive. The flip bar itself never counts as step 1.
        """
        if not self.reversal_pending or flipped is not None or self.trend_dir is None:
            return False
        open_, close, vwap = bar["open"], bar["close"], bar["vwap"]
        if not _finite(open_, close, vwap):
            return False
        down = self.trend_dir == "DOWN"
        beyond = close < vwap if down else close > vwap
        back_across = (close > open_ and close > vwap) if down else (close < open_ and close < vwap)
        trend_colour = close < open_ if down else close > open_
        if self.reversal_step == 0 and beyond:
            self.reversal_step = 1
            return False
        if self.reversal_step == 1 and back_across:
            self.reversal_step = 2
            return False
        if self.reversal_step == 2 and trend_colour and beyond:
            fraction = body_fraction_beyond_vwap(open_, close, vwap, above=not down)
            return fraction >= self.config.min_vwap_body_fraction
        return False

    # -- entries ------------------------------------------------------------------
    def _bar_before_cutoff(self, ts: pd.Timestamp) -> bool:
        bar_end = ts + pd.Timedelta(minutes=self.config.bar_minutes)
        return bar_end.date() == ts.date() and bar_end.time() < self.config.entry_cutoff

    def _entry(
        self,
        ts: pd.Timestamp,
        bar: Mapping[str, float],
        prev: Mapping[str, float] | None,
        levels: Mapping[str, float],
        flipped: str | None,
        reversal_ready: bool,
    ) -> CPRAlgo4Decision:
        if self.regime in (None, REGIME_NO_TRADE) or ts.time() <= self.config.regime_bar:
            return CPRAlgo4Decision(reason="no_regime")
        if prev is None or flipped is not None:
            return CPRAlgo4Decision(reason="no_setup")
        if self._blocked_entry_bar is not None and ts == self._blocked_entry_bar:
            return CPRAlgo4Decision(reason="exit_bar")
        if not self._bar_before_cutoff(ts):
            return CPRAlgo4Decision(reason="after_entry_cutoff")
        if self.regime == REGIME_SIDEWAYS:
            return self._sideways_entry(ts, bar, prev, levels)
        return self._trending_entry(ts, bar, prev, levels, reversal_ready)

    def _sideways_entry(
        self,
        ts: pd.Timestamp,
        bar: Mapping[str, float],
        prev: Mapping[str, float],
        levels: Mapping[str, float],
    ) -> CPRAlgo4Decision:
        k, d, prev_k, prev_d = bar["srsi_k"], bar["srsi_d"], prev["srsi_k"], prev["srsi_d"]
        if not _finite(k, d, prev_k, prev_d):
            return CPRAlgo4Decision(reason="srsi_unavailable")
        cross_up = prev_k <= prev_d and k > d and max(prev_k, k) <= self.config.srsi_oversold
        cross_down = prev_k >= prev_d and k < d and min(prev_k, k) >= self.config.srsi_overbought
        if not (cross_up or cross_down):
            return CPRAlgo4Decision(reason="no_setup")
        long = cross_up
        swings = self._swing_lows if long else self._swing_highs
        if not swings:
            return CPRAlgo4Decision(reason="missing_swing_stop")
        first30 = None
        if self.config.first30_target and self._first30_complete:
            first30 = self._first30_high if long else self._first30_low
        return self._enter(
            ts, "LONG" if long else "SHORT", PREMISE_SIDEWAYS, bar["close"], swings[-1], levels, first30
        )

    def _trending_entry(
        self,
        ts: pd.Timestamp,
        bar: Mapping[str, float],
        prev: Mapping[str, float],
        levels: Mapping[str, float],
        reversal_ready: bool,
    ) -> CPRAlgo4Decision:
        long = self.trend_dir == "UP"
        if self.reversal_pending:
            if not reversal_ready:
                return CPRAlgo4Decision(reason="waiting_for_reversal_sequence")
            premise = PREMISE_REVERSAL
        else:
            if not self._continuation_pattern(bar, prev, long):
                return CPRAlgo4Decision(reason="no_setup")
            premise = PREMISE_CONTINUATION
        if not self._trend_filters_pass(bar, prev, long):
            return CPRAlgo4Decision(reason="trend_filters_rejected")
        stop = bar["low"] if long else bar["high"]
        return self._enter(ts, "LONG" if long else "SHORT", premise, bar["close"], stop, levels, None)

    def _continuation_pattern(self, bar: Mapping[str, float], prev: Mapping[str, float], long: bool) -> bool:
        """Close back across VWAP with enough body, or the two-candle VWAP support."""
        o, h, lo, c, v = bar["open"], bar["high"], bar["low"], bar["close"], bar["vwap"]
        po, ph, plo, pc, pv = prev["open"], prev["high"], prev["low"], prev["close"], prev["vwap"]
        if not _finite(o, h, lo, c, v, po, ph, plo, pc, pv):
            return False
        minimum = self.config.min_vwap_body_fraction
        if long:
            crossed = pc < pv and c > v and body_fraction_beyond_vwap(o, c, v, above=True) >= minimum
            supported = pc < po and plo <= pv < pc and c > o and lo <= v < c
        else:
            crossed = pc > pv and c < v and body_fraction_beyond_vwap(o, c, v, above=False) >= minimum
            supported = pc > po and ph >= pv > pc and c < o and h >= v > c
        return crossed or supported

    def _trend_filters_pass(self, bar: Mapping[str, float], prev: Mapping[str, float], long: bool) -> bool:
        """RSI > 45 (< 65 for puts) and EMA5/EMA20 ordered and both sloping the trade's way."""
        rsi, ema5, ema20 = bar["rsi"], bar["ema5"], bar["ema20"]
        prev_ema5, prev_ema20 = prev["ema5"], prev["ema20"]
        if not _finite(rsi, ema5, ema20, prev_ema5, prev_ema20):
            return False
        slope5, slope20 = ema5 - prev_ema5, ema20 - prev_ema20
        if long:
            return rsi > self.config.rsi_long_min and ema5 > ema20 and slope5 > 0.0 and slope20 > 0.0
        return rsi < self.config.rsi_short_max and ema5 < ema20 and slope5 < 0.0 and slope20 < 0.0

    def _enter(
        self,
        ts: pd.Timestamp,
        direction: str,
        premise: str,
        entry: float,
        stop: float,
        levels: Mapping[str, float],
        first30: float | None,
    ) -> CPRAlgo4Decision:
        plan, reason = build_trade_plan(
            direction=direction,
            premise=premise,
            entry=entry,
            stop=stop,
            levels=levels,
            config=self.config,
            bar_ts=ts,
            first30_extreme=first30,
        )
        if plan is None:
            return CPRAlgo4Decision(reason=reason, debug={"premise": premise, "direction": direction})
        return CPRAlgo4Decision(
            action="ENTER_LONG" if direction == "LONG" else "ENTER_SHORT",
            reason=premise,
            plan=plan,
            debug={"regime": self.regime, "trend_dir": self.trend_dir},
        )

    # -- open-position management ------------------------------------------------
    def _manage(
        self,
        ts: pd.Timestamp,
        bar: Mapping[str, float],
        prev: Mapping[str, float] | None,
        levels: Mapping[str, float],
        plan: CPRAlgo4TradePlan,
        flipped: str | None,
    ) -> CPRAlgo4Decision:
        long = plan.is_long
        # 1. Premise exits: the reason for being in the trade has gone.
        if plan.premise == PREMISE_SIDEWAYS:
            if prev is not None and self._srsi_reversal_against(bar, prev, long):
                return CPRAlgo4Decision(action="EXIT", reason=EXIT_SRSI)
        elif flipped is not None and flipped != ("UP" if long else "DOWN"):
            return CPRAlgo4Decision(action="EXIT", reason=EXIT_STRUCTURE)

        # 2. TRAIL mode: breach first (against the PREVIOUS bar), then advance
        #    stages -- otherwise a losing close would reset its own reference.
        close = bar["close"]
        if plan.exit_mode == "TRAIL" and math.isfinite(close):
            if plan.trail_armed and prev is not None:
                reference = prev["low"] if long else prev["high"]
                if math.isfinite(reference) and (close < reference if long else close > reference):
                    return CPRAlgo4Decision(action="EXIT", reason=EXIT_TRAIL)
            self._advance_trail(plan, close)

        # 3. The one R1 add on a trending long.
        if self._scale_in_ready(ts, bar, prev, levels, plan):
            return CPRAlgo4Decision(action="SCALE_IN", reason="R1_RED_THEN_GREEN")
        return CPRAlgo4Decision(debug={"current_stop": plan.current_stop, "trail_stage": plan.trail_stage})

    def _srsi_reversal_against(self, bar: Mapping[str, float], prev: Mapping[str, float], long: bool) -> bool:
        k, d, prev_k, prev_d = bar["srsi_k"], bar["srsi_d"], prev["srsi_k"], prev["srsi_d"]
        if not _finite(k, d, prev_k, prev_d):
            return False
        if long:
            return prev_k >= prev_d and k < d and min(prev_k, k) >= self.config.srsi_overbought
        return prev_k <= prev_d and k > d and max(prev_k, k) <= self.config.srsi_oversold

    @staticmethod
    def _advance_trail(plan: CPRAlgo4TradePlan, close: float) -> None:
        """Staged ratchets on a completed close (TRAIL mode only).

        Normal trades: at the first milestone -> stop to breakeven, trail armed.
        Reversal trades: first milestone -> breakeven; following milestone ->
        stop locks 1R and the trail arms.
        """

        def reached(level: float) -> bool:
            return close >= level if plan.is_long else close <= level

        if plan.premise == PREMISE_REVERSAL:
            if plan.trail_stage == "NONE" and reached(plan.first_milestone):
                plan.ratchet_stop(plan.entry)
                plan.trail_stage = "BREAKEVEN"
            if plan.trail_stage == "BREAKEVEN" and reached(plan.following_milestone):
                locked = plan.entry + plan.risk if plan.is_long else plan.entry - plan.risk
                plan.ratchet_stop(locked)
                plan.trail_stage = "R1_LOCKED"
                plan.trail_armed = True
        elif plan.trail_stage == "NONE" and reached(plan.first_milestone):
            plan.ratchet_stop(plan.entry)
            plan.trail_stage = "TRAILING"
            plan.trail_armed = True

    def _scale_in_ready(
        self,
        ts: pd.Timestamp,
        bar: Mapping[str, float],
        prev: Mapping[str, float] | None,
        levels: Mapping[str, float],
        plan: CPRAlgo4TradePlan,
    ) -> bool:
        """Red candle touching R1, then a green candle reclaiming it (long, trend, once)."""
        if (
            not self.config.scale_in_enabled
            or plan.scale_in_used
            or not plan.is_long
            or plan.premise not in (PREMISE_CONTINUATION, PREMISE_REVERSAL)
            or prev is None
            or not self._bar_before_cutoff(ts)
        ):
            return False
        r1, buffer = levels["r1"], self.config.r1_touch_buffer
        po, ph, plo, pc = prev["open"], prev["high"], prev["low"], prev["close"]
        o, lo, c = bar["open"], bar["low"], bar["close"]
        if not _finite(r1, po, ph, plo, pc, o, lo, c):
            return False
        red_touch = pc < po and plo <= r1 + buffer and ph >= r1 - buffer
        green_reclaim = c > o and c >= r1 and lo >= r1 - buffer
        return red_touch and green_reclaim
