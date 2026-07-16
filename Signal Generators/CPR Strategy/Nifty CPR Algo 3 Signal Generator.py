"""
NIFTY CPR Algo 3 Signal Generator (multi-instrument).

============================== What makes Algo 3 different ====================
Algo 1 and Algo 2 look at ONE chart (the NIFTY index/spot) and decide a trade.
Algo 3 from the "CPR basic setup" PDF is different: it looks at THREE charts at
the same moment and only trades when all three agree -

  1. the SPOT (NIFTY index),
  2. a ~100-point IN-THE-MONEY CALL (CE), and
  3. a ~100-point IN-THE-MONEY PUT (PE).

(Picking the actual 100-ITM strikes - and avoiding 50-point strikes by stepping
one more strike in - is the *caller's* job; this file just receives the three
candle streams.)

A CALL trade (buy the CE) fires only when EVERY one of these is true on the same
5-minute candle:
  - VWAP:  CE close above its VWAP, PE close below its VWAP, spot close above VWAP.
  - CPR :  CE close above its CPR band, PE close below its CPR band, spot close
           above its CPR band. (Being above the band also means spot is NOT
           inside the band, which is the PDF's "no entry inside CPR" rule.)
  - RSI :  on SPOT, RSI > ARSI and ARSI > 45.  ("ARSI" = the 20-period EMA of RSI,
           which the shared engine already computes as the `rsi_ema20` column;
           the repo's CPR.pdf uses exactly this RSI > EMA(RSI,20) rule.)
A PUT trade (buy the PE) is the mirror image (… below VWAP, below CPR band, and on
spot RSI < ARSI and ARSI < 60).

The TARGET is the next spot pivot level in the trade's direction, taken from the
ladder S2 / S1 / PrevDayLow / BottomCPR / TopCPR / R1 / PrevDayHigh / R2 (the PDF's
zone table). The stop is the CPR-band edge the price just broke out of.

============================== How it reuses the engine =======================
All the heavy indicator maths (CPR levels, VWAP, RSI, ARSI=`rsi_ema20`, the
previous-day pivots) already lives in `cpr_strategy_logic.build_cpr_with_indicators`.
We just run that builder on each of the three frames and read the columns - so Algo
3 stays in lock-step with how Algo 1/2 compute the very same indicators.

NOTE ON SHAPE: because Algo 3 needs three charts, its public functions take THREE
DataFrames (spot, ce, pe) instead of the single `data` frame Algo 1/2 take. The
output is the same `CPRDecision` the other generators return, with
`action == "ENTER_LONG"` meaning "buy the CE" and `action == "ENTER_SHORT"` meaning
"buy the PE" (the same convention the front-test master already uses).

NOTE ON WIRING: this generator is standalone for now. The live front-test and the
backtests still feed only spot data, so Algo 3 is not wired into them yet - doing so
needs synchronized 1-minute OHLC for the chosen ITM CE & PE (a separate follow-up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from cpr_strategy_logic import (
    CPRDecision,
    CPRPositionContext,
    CPRStrategyConfig,
    build_cpr_with_indicators,
)


@dataclass
class CPRAlgo3Config:
    """Tunable settings for Algo 3, kept in one beginner-friendly place."""

    # Spot RSI/ARSI thresholds from the PDF (45 for calls, 60 for puts).
    call_arsi_min: float = 45.0
    put_arsi_max: float = 60.0
    # Indicator periods (RSI/EMA/VWAP/CPR) come from the shared CPR config so all
    # three algos compute indicators identically.
    indicator_config: CPRStrategyConfig = field(default_factory=CPRStrategyConfig)


# Spot pivot columns that form the target "ladder", listed low -> high. The target
# for a CE is the next one ABOVE the spot close; for a PE, the next one BELOW.
_SPOT_LEVEL_COLUMNS = ("s2", "s1", "prev_low", "bc", "tc", "r1", "prev_high", "r2")

# Indicator columns every instrument's candle must have (non-NaN) before we can
# judge it. They are blank until the previous session's data exists, so requiring
# them also enforces "wait for enough history".
_REQUIRED_COLUMNS = ("close", "vwap", "cpr_upper", "cpr_lower", "rsi", "rsi_ema20")


def _row_ready(row: pd.Series) -> bool:
    """True only when every indicator we need on this candle is a real number."""
    return all(
        pd.notna(row.get(col)) and math.isfinite(float(row.get(col)))
        for col in _REQUIRED_COLUMNS
    )


def _next_level(row: pd.Series, reference: float, direction: str) -> float:
    """
    Next spot pivot level above (`direction='up'`) or below (`'down'`) `reference`.

    Returns NaN if there is no further level in that direction.
    """
    levels = [row.get(col) for col in _SPOT_LEVEL_COLUMNS]
    levels = sorted(
        float(value)
        for value in levels
        if pd.notna(value) and math.isfinite(float(value))
    )
    if direction == "up":
        higher = [v for v in levels if v > reference]
        return higher[0] if higher else float("nan")
    lower = [v for v in levels if v < reference]
    return lower[-1] if lower else float("nan")


class NiftyCPRAlgo3SignalGenerator:
    """Multi-instrument CPR Algo 3 generator (spot + ITM CE + ITM PE)."""

    def __init__(self, config: CPRAlgo3Config | None = None) -> None:
        self.config = config or CPRAlgo3Config()

    def _enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add CPR levels, VWAP, RSI and ARSI to one instrument's candles."""
        return build_cpr_with_indicators(frame, self.config.indicator_config)

    def _evaluate(self, spot: pd.Series, ce: pd.Series, pe: pd.Series) -> CPRDecision:
        """Apply Algo 3's four-way alignment to one aligned (spot, ce, pe) candle."""
        if not (_row_ready(spot) and _row_ready(ce) and _row_ready(pe)):
            return CPRDecision(action="HOLD", strategy_name="CPR_ALGO3",
                               exit_reason="insufficient indicator history")
        cfg = self.config

        # --- CALL alignment (buy the CE) ---
        call_vwap = ce["close"] > ce["vwap"] and pe["close"] < pe["vwap"] and spot["close"] > spot["vwap"]
        call_cpr = ce["close"] > ce["cpr_upper"] and pe["close"] < pe["cpr_lower"] and spot["close"] > spot["cpr_upper"]
        call_rsi = spot["rsi"] > spot["rsi_ema20"] and spot["rsi_ema20"] > cfg.call_arsi_min
        call = bool(call_vwap and call_cpr and call_rsi)

        # --- PUT alignment (buy the PE) ---
        put_vwap = pe["close"] > pe["vwap"] and ce["close"] < ce["vwap"] and spot["close"] < spot["vwap"]
        put_cpr = pe["close"] > pe["cpr_upper"] and ce["close"] < ce["cpr_lower"] and spot["close"] < spot["cpr_lower"]
        put_rsi = spot["rsi"] < spot["rsi_ema20"] and spot["rsi_ema20"] < cfg.put_arsi_max
        put = bool(put_vwap and put_cpr and put_rsi)

        debug = {"call_vwap": call_vwap, "call_cpr": call_cpr, "call_rsi": call_rsi,
                 "put_vwap": put_vwap, "put_cpr": put_cpr, "put_rsi": put_rsi}

        if call and put:
            # Opposite setups at once should be impossible (they need spot on both
            # sides of VWAP/CPR); guard anyway rather than guess, like the engine does.
            return CPRDecision(action="HOLD", strategy_name="CPR_ALGO3",
                               exit_reason="conflicting call/put alignment", debug=debug)
        spot_close = float(spot["close"])
        if call:
            target = _next_level(spot, spot_close, "up")
            stop = float(spot["cpr_upper"])
            if not (
                math.isfinite(target)
                and target > spot_close
                and math.isfinite(stop)
                and 0.0 < stop < spot_close
            ):
                return CPRDecision(
                    action="HOLD",
                    strategy_name="CPR_ALGO3",
                    exit_reason="invalid long target/stop geometry",
                    debug=debug,
                )
            return CPRDecision(
                action="ENTER_LONG", strategy_name="CPR_ALGO3", signal_triggered=True,
                entry_underlying=spot_close,
                target_underlying=target,
                stop_underlying=stop,  # invalidates if it falls back into the band
                debug=debug,
            )
        if put:
            target = _next_level(spot, spot_close, "down")
            stop = float(spot["cpr_lower"])
            if not (
                math.isfinite(target)
                and 0.0 < target < spot_close
                and math.isfinite(stop)
                and stop > spot_close
            ):
                return CPRDecision(
                    action="HOLD",
                    strategy_name="CPR_ALGO3",
                    exit_reason="invalid short target/stop geometry",
                    debug=debug,
                )
            return CPRDecision(
                action="ENTER_SHORT", strategy_name="CPR_ALGO3", signal_triggered=True,
                entry_underlying=spot_close,
                target_underlying=target,
                stop_underlying=stop,
                debug=debug,
            )
        return CPRDecision(action="HOLD", strategy_name="CPR_ALGO3", debug=debug)

    def _aligned(self, spot: pd.DataFrame, ce: pd.DataFrame, pe: pd.DataFrame):
        """Enrich each frame and return the three indexed by their shared timestamps."""
        s = self._enrich(spot).set_index("timestamp")
        c = self._enrich(ce).set_index("timestamp")
        p = self._enrich(pe).set_index("timestamp")
        common = s.index.intersection(c.index).intersection(p.index)
        return s, c, p, common

    def generate(self, spot: pd.DataFrame, ce: pd.DataFrame, pe: pd.DataFrame) -> pd.DataFrame:
        """Return one Algo 3 decision row per candle the three charts share."""
        s, c, p, common = self._aligned(spot, ce, pe)
        rows = []
        for ts in common:
            decision = self._evaluate(s.loc[ts], c.loc[ts], p.loc[ts])
            rows.append({
                "timestamp": ts,
                "action": decision.action,
                "signal": {"ENTER_LONG": "BUY_CE", "ENTER_SHORT": "BUY_PE"}.get(decision.action, "NO_SIGNAL"),
                "entry": decision.entry_underlying,
                "target": decision.target_underlying,
                "stop": decision.stop_underlying,
            })
        return pd.DataFrame(rows)

    def latest_signal(
        self,
        spot: pd.DataFrame,
        ce: pd.DataFrame,
        pe: pd.DataFrame,
        position: CPRPositionContext | None = None,  # accepted for API parity; entries only
    ) -> CPRDecision:
        """Return only the newest Algo 3 decision (the last shared candle)."""
        s, c, p, common = self._aligned(spot, ce, pe)
        if len(common) == 0:
            return CPRDecision(action="HOLD", strategy_name="CPR_ALGO3",
                               exit_reason="no aligned candles across the three charts")
        latest = (s.index[-1], c.index[-1], p.index[-1])
        if not (latest[0] == latest[1] == latest[2]):
            return CPRDecision(
                action="HOLD",
                strategy_name="CPR_ALGO3",
                exit_reason="cross-index timestamps are not current/aligned",
            )
        ts = common[-1]
        return self._evaluate(s.loc[ts], c.loc[ts], p.loc[ts])


def generate_nifty_cpr_algo3_signals(
    spot: pd.DataFrame,
    ce: pd.DataFrame,
    pe: pd.DataFrame,
    config: CPRAlgo3Config | None = None,
) -> pd.DataFrame:
    """Full-history Algo 3 signals across the candles the three charts share."""
    return NiftyCPRAlgo3SignalGenerator(config=config).generate(spot, ce, pe)


def get_latest_nifty_cpr_algo3_signal(
    spot: pd.DataFrame,
    ce: pd.DataFrame,
    pe: pd.DataFrame,
    config: CPRAlgo3Config | None = None,
    position: CPRPositionContext | None = None,
) -> CPRDecision:
    """Only the newest Algo 3 decision."""
    return NiftyCPRAlgo3SignalGenerator(config=config).latest_signal(spot, ce, pe, position=position)


__all__ = [
    "CPRAlgo3Config",
    "NiftyCPRAlgo3SignalGenerator",
    "generate_nifty_cpr_algo3_signals",
    "get_latest_nifty_cpr_algo3_signal",
]
