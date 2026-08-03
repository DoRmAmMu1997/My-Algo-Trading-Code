"""The two candidate rules the regime-adaptive router dispatches between.

WHY THESE ARE PLAIN FUNCTIONS AND NOT ENGINES
---------------------------------------------
In the upstream project these are standalone strategies. Here the operator chose
a ROUTER-ONLY topology: `regime_adaptive` is the single worker, and these two are
library code behind it. If they were also registered as workers, the router and a
candidate could take the SAME signal in the same session — real double exposure
that the Google Sheet would not reveal, because each row would look like an
ordinary independent strategy.

So they deliberately expose no `Config` / `Engine` / `PositionContext` classes and
are NOT listed in `test_trading_bot_ports.py`'s PORTS table. They are column
producers: each returns the setup flags and levels for its branch, and the router
decides which branch is live on any given bar.

Both rules are LONG-OPTIONS-ONLY in effect: a bullish setup buys a CE, a bearish
one buys a PE. Neither ever sells an option.

Adapted from the MIT-licensed `workratananmol-hub/nifty-options-paper-trading-bot`
(`src/strategies/opening_range.py`, `src/strategies/vwap_mean_reversion.py`).
See REGIME_PORTING_NOTES.md for everything that was deliberately NOT ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "attach_breakout_columns",
    "attach_mean_reversion_columns",
]


def attach_breakout_columns(
    frame: pd.DataFrame,
    *,
    buffer_atr_mult: float,
    buffer_min_points: float,
    stop_loss_pct: float,
    target_pct: float,
) -> pd.DataFrame:
    """Opening-range break confirmed by VWAP alignment. Prefix: `br_`.

    A long needs price above the opening-range high by a buffer AND above VWAP;
    a short is the mirror. The buffer is `max(atr * mult, min_points)`, matching
    the upstream `max(atr*0.05, 2.0)` — it stops a one-tick poke through the
    range counting as a break.

    FAIL CLOSED: ATR is mandatory upstream, so a bar with no ATR (or with no
    completed opening range, or no VWAP) produces no setup at all rather than a
    setup computed from a partial picture.
    """
    result = frame.copy()
    close = result["close"].astype(float)
    atr = result["atr"].astype(float)
    vwap = result["vwap"].astype(float)

    buffer = np.maximum(atr * float(buffer_atr_mult), float(buffer_min_points))
    usable = result["or_complete"].fillna(False) & atr.notna() & vwap.notna()

    broke_up = close > (result["or_high"].astype(float) + buffer)
    broke_down = close < (result["or_low"].astype(float) - buffer)

    result["br_long_setup"] = (usable & broke_up & (close > vwap)).fillna(False)
    result["br_short_setup"] = (usable & broke_down & (close < vwap)).fillna(False)

    result["br_long_entry_price"] = close
    result["br_short_entry_price"] = close
    result["br_long_stop_from_setup"] = close * (1.0 - float(stop_loss_pct))
    result["br_long_target_from_setup"] = close * (1.0 + float(target_pct))
    result["br_short_stop_from_setup"] = close * (1.0 + float(stop_loss_pct))
    result["br_short_target_from_setup"] = close * (1.0 - float(target_pct))
    return result


def attach_mean_reversion_columns(
    frame: pd.DataFrame,
    *,
    atr_mult: float,
    adx_ceiling: float,
    stop_loss_pct: float,
    target_pct: float,
) -> pd.DataFrame:
    """Fade an extension away from VWAP, range regimes only. Prefix: `mr_`.

    Price stretched BELOW VWAP by `atr_mult * ATR` is faded LONG (expecting a
    pull back up to VWAP); stretched above is faded SHORT.

    FAIL CLOSED twice over, exactly as upstream: ATR and ADX are both mandatory,
    and the rule stands down entirely once `adx >= adx_ceiling`, because fading
    a genuine trend is how a mean-reversion rule bleeds.
    """
    result = frame.copy()
    close = result["close"].astype(float)
    atr = result["atr"].astype(float)
    adx = result["adx"].astype(float)
    vwap = result["vwap"].astype(float)

    distance = close - vwap
    threshold = atr * float(atr_mult)
    # ADX must be PRESENT and below the ceiling. `adx.notna()` is what makes a
    # missing ADX a no-trade rather than a silently permissive comparison.
    range_regime = adx.notna() & (adx < float(adx_ceiling))
    usable = range_regime & atr.notna() & vwap.notna()

    result["mr_long_setup"] = (usable & (distance <= -threshold)).fillna(False)
    result["mr_short_setup"] = (usable & (distance >= threshold)).fillna(False)

    result["mr_long_entry_price"] = close
    result["mr_short_entry_price"] = close
    result["mr_long_stop_from_setup"] = close * (1.0 - float(stop_loss_pct))
    # A fade targets the mean it is fading back towards, capped by the ordinary
    # percentage target so a VWAP sitting far away cannot invent a huge target.
    result["mr_long_target_from_setup"] = np.minimum(vwap, close * (1.0 + float(target_pct)))
    result["mr_short_stop_from_setup"] = close * (1.0 + float(stop_loss_pct))
    result["mr_short_target_from_setup"] = np.maximum(vwap, close * (1.0 - float(target_pct)))
    return result
