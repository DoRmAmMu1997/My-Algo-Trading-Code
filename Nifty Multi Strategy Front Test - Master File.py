"""
================================================================================
Nifty Multi Strategy Front Test - Master File
================================================================================

WHAT THIS FILE IS (read this first if you have never seen this code base)
------------------------------------------------------------------------
This is a single multi-threaded paper-trading runner that combines SEVEN
NIFTY strategies behind ONE shared market-data fetcher. It started as
the union of two earlier files plus one Greeks-driven addition:

  1. "Nifty Multi Strategy Front Test - DhanHQ ATM Options.py"
       - 4 strategies that trade single-leg ATM options of the
         next-next NIFTY expiry, BUY-only:
           * Renko          (1-min source)
           * EMA Trend      (1-min source, locally resampled to 5-min)
           * Heikin Ashi    (1-min source)
           * Profit Shooter (1-min source)

  2. "Nifty Supertrend Front Test - DhanHQ Hedged Puts.py"
       - 2 strategies that trade two-leg HEDGED option spreads of the
         current-week NIFTY expiry. Strike selection here is by
         "live premium closest to a target":
           * Supertrend Bullish: SELL ~Rs.160 PE + BUY ~Rs.10 PE
                                 (locally resampled to 3-min)
           * Donchian   Bearish: SELL ~Rs.160 CE + BUY ~Rs.10 CE
                                 (locally resampled to 5-min)

  3. Delta-0.2 Hedged Spread (this file only)
       - 1 strategy that runs TWO independent hedged spreads at once
         (one CE side, one PE side) of the current-week expiry. Strike
         selection is by Greek (delta), not by premium:
           * At 09:20 IST it pulls the live option chain, picks the CE
             with delta closest to +0.20 and the PE with delta closest
             to -0.20, and stores their LTPs as reference premiums.
           * A side enters when its monitored leg's LTP drops 5% below
             the reference: SELL the monitored strike, BUY the strike
             4 strikes further OTM (CE: higher; PE: lower) as a hedge.
           * Per-side exit at 3x the reference; strategy-wide max-loss
             of Rs.5000/lot; force exit at 15:20.

EXPIRY RULES (the explicit if-else the user asked for)
------------------------------------------------------
Different strategies pick different expiries. Rather than burying this
in a single overloaded method, every worker is explicit about which
expiry rule it uses:

    if strategy belongs to the "Hedged Puts" family
        -> use OptionsContractResolver.get_current_week_expiry()
           (the FIRST expiry on or after today)
    else
        -> use OptionsContractResolver.get_target_expiry()
           (the SECOND expiry on or after today, i.e. "next-next")

Both methods live on the same resolver so the choice is one line at
the call site. That keeps the rule visible in the code and easy to
audit later without chasing a hidden flag.

STRIKE RULES (also explicit per family)
---------------------------------------
- ATM family (the 4 ATM workers) -> resolver.get_atm_option(spot, dir).
  Picks the strike whose round-50 value is nearest to live spot.
- Hedged family (Supertrend Bullish + Donchian Bearish) -> the strike
  selection lives INSIDE each dedicated worker
  (`_pick_hedged_puts` / `_pick_hedged_calls`). Those workers use
  resolver primitives (`list_puts_for_expiry`, `list_calls_for_expiry`,
  `pick_put_by_target_premium`, `pick_call_by_target_premium`) but the
  end-to-end "pick a hedged pair" recipe is per-strategy, because each
  hedged strategy could plausibly want different premium thresholds
  later (e.g. a different target premium for a different volatility
  regime).

EXECUTION MODEL CHEAT SHEET (paper)
-----------------------------------
Single-leg ATM trades (the 4 ATM workers):
    PnL = (exit_option_price - entry_option_price) * qty
    Both LONG and SHORT signals open as BUY legs (CE for LONG, PE for
    SHORT). PnL is therefore always (live - entry) * qty.

Hedged spreads (the 2 Hedged workers):
    PnL_main  = (entry_premium - exit_premium) * qty   (we SOLD this leg)
    PnL_hedge = (exit_premium - entry_premium) * qty   (we BOUGHT this leg)
    Total     = PnL_main + PnL_hedge

THREAD ARCHITECTURE
-------------------
- One `CentralMarketDataFetcher` thread:
    * Pulls live 1-minute NIFTY OHLC every poll.
    * Pulls LTPs for NIFTY spot AND every option leg currently held by
      ANY worker (single batched ticker_data call).
    * Publishes everything into a thread-safe `SharedMarketDataStore`.

- Six `*StrategyWorker` threads:
    * Read the 1-minute OHLC from the shared store.
    * Resample locally if their strategy timeframe is higher than 1m.
    * Run their own signal generator (one of six logic modules).
    * Manage their own paper position (single-leg OR hedged).
    * Never call the broker for OHLC directly. They only do tiny
      direct LTP fallback fetches when the cache is cold.

This central-fetcher design keeps API usage low and makes the data
deterministic across all six strategies.

WHY ONE FILE INSTEAD OF SEPARATE FILES
--------------------------------------
- Single fetch budget. One DhanHQ ticker_data call covers spot plus
  every active option leg from every worker simultaneously.
- One log destination. All six strategies share LOG_FILE so a single
  audit trail captures the day.
- One process to start, one Ctrl+C to stop everything cleanly.

CLASS HIERARCHY OVERVIEW
------------------------
    BasePaperStrategyWorker            (abstract - run loop + helpers)
        |
        +-- AtmSingleLegStrategyWorker (single-leg BUY-only ATM)
        |       |
        |       +-- RenkoStrategyWorker
        |       +-- EMATrendStrategyWorker
        |       +-- HeikinAshiStrategyWorker
        |       +-- ProfitShooterStrategyWorker
        |
        +-- SupertrendBullishWorker    (hedged PE spread)
        +-- DonchianBearishWorker      (hedged CE spread)
        +-- Delta20HedgedSpreadWorker  (dual-side hedged spread, Greeks-driven)

The split is intentional: only the four ATM workers share an
`enter_position` / `exit_position` flow, so it is hosted on
`AtmSingleLegStrategyWorker`. The hedged workers each implement
their own enter / exit because the position shape, leg count, and
PnL math differ - and the Delta-0.2 worker further differs in that
it carries TWO simultaneous hedged positions (one CE, one PE) and
ignores OHLC entirely (its triggers are option-leg LTPs against a
09:20 reference).
"""

from __future__ import annotations

# --- Standard library imports ------------------------------------------------
import glob
import importlib.util
import logging
import os
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

# --- Third-party imports -----------------------------------------------------
import pandas as pd
# `DhanContext` wraps (client_id, access_token) so the rest of the SDK can
# share one authenticated session. `DhanLogin` is the auth helper class we
# use only for `user_profile(...)` startup validation -- the OAuth dance
# itself happens in `Dependencies/dhan_token_setup.py`, not here.
from dhanhq import DhanContext, DhanLogin, dhanhq

try:
    # `python-dotenv` is optional. If installed, values such as the DhanHQ
    # credentials and the EMA tuning parameters can be auto-loaded from a
    # local `.env` file at the repo root or in the EMA Trend Strategy folder.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


# =============================================================================
# STATIC FILE-LEVEL CONFIGURATION
# =============================================================================
# Constants that never come from `.env` because they are tied to the project
# layout or are pure visual / architectural identifiers.
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT_DIR / "Dependencies" / "log_files" / "nifty_multi_strategy_master_front_test_dhanhq.log"
LOGGER_NAME = "nifty_multi_strategy_master_front_test_dhanhq"

INSTRUMENT_MASTER_GLOB = str(ROOT_DIR / "Dependencies" / "all_instrument *.csv")

# The fetcher publishes only one centralized stream: 1-minute OHLC.
# Higher timeframes (3-min for Supertrend, 5-min for EMA / Donchian) are
# derived inside the consuming workers. This keeps the fetch budget tiny.
REQUIRED_TIMEFRAMES = ("1",)

# ANSI color codes for PnL formatting in console output.
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"


# -----------------------------------------------------------------------------
# .env loading (must happen BEFORE any `_env_*` reads below)
# -----------------------------------------------------------------------------
# Single source of truth for all tunable knobs in this runner:
#   Multithreading/Dependencies/.env
#
# That file lives next to this trading script (one folder over). Existing
# shell variables win (`override=False`); the `.env` only fills the gap
# when nothing is set in the shell.
ENV_FILE = Path(__file__).resolve().parent / "Dependencies" / ".env"
if load_dotenv is not None and ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)


# -----------------------------------------------------------------------------
# Small env helpers (defined before the env-resolved config block so the
# Tier 2 constants below can call into them).
# -----------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    """
    Read a string from the environment, falling back to `default` if unset.

    Surrounding quotes are stripped because users sometimes habitually
    quote `.env` values; we treat `KEY="abc"` and `KEY=abc` as equivalent.
    """
    raw = os.getenv(name, "")
    if raw is None:
        return str(default)
    raw = raw.strip()
    if raw.startswith(('"', "'")) and raw.endswith(('"', "'")):
        raw = raw[1:-1]
    return raw if raw else str(default)


def _env_float(name: str, default: float) -> float:
    """
    Read a float from the environment without crashing on bad input.

    A typo in `.env` should not crash the runner; we silently fall back to
    the supplied default and continue.
    """
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment. Same forgiving pattern as `_env_float`."""
    raw = os.getenv(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


# =============================================================================
# DHANHQ CREDENTIALS (read STRICTLY from .env; no in-code defaults)
# =============================================================================
# All four values are populated by the user / by `Dependencies/dhan_token_setup.py`.
# The runner refuses to start if `CLIENT_CODE` or `ACCESS_TOKEN` is missing -
# see `main()` for the validation message.
#
# DHAN_CLIENT_CODE  : 10-digit dhanClientId (e.g. "1102601655").
# DHAN_API_KEY      : long-lived "app_id" used by the OAuth setup script.
# DHAN_API_SECRET   : long-lived "app_secret" pair to DHAN_API_KEY.
# DHAN_ACCESS_TOKEN : 12-month token produced by the setup script. This is
#                     what the dhanhq SDK actually authenticates with.
CLIENT_CODE = _env_str("DHAN_CLIENT_CODE", "")
API_KEY = _env_str("DHAN_API_KEY", "")
API_SECRET = _env_str("DHAN_API_SECRET", "")
ACCESS_TOKEN = _env_str("DHAN_ACCESS_TOKEN", "")


# =============================================================================
# TIER 3 ENV-DRIVEN CONFIGURATION
# =============================================================================
# Every constant below is read from `.env` at module import time. The
# numeric / string literals in the second argument of each `_env_*(...)`
# call are ONLY in-code fallbacks for when the env var is absent (e.g.
# someone runs the script without `.env` to get a quick smoke test).
#
# To tune any of these, edit the corresponding key in
# `Multithreading/Dependencies/.env` -- changes propagate to all three
# trading scripts on the next run.

# -----------------------------------------------------------------------------
# Operational settings (apply to the runner as a whole)
# -----------------------------------------------------------------------------
UNDERLYING = _env_str("UNDERLYING", "NIFTY").upper().strip() or "NIFTY"

# Minimum number of 1-minute bars before any strategy is allowed to evaluate
# signals. 120 is enough warm-up for ATR/EMA/Donchian/Supertrend across all
# six strategies even after one of them resamples 1m -> 5m (still ~24 bars).
MIN_BARS = _env_int("MIN_BARS", 120)

# How often the fetcher re-polls 1-minute OHLC and the LTP batch (seconds).
FETCH_POLL_SECONDS = _env_int("FETCH_POLL_SECONDS", 2)

# Bounded join timeout for each thread on shutdown.
SHUTDOWN_JOIN_SECONDS = _env_float("SHUTDOWN_JOIN_SECONDS", 6.0)

# How many calendar days of intraday history to request every fetch.
# DhanHQ supports up to 90 days; we use a small rolling window because all we
# need is enough bars to satisfy MIN_BARS plus margin for weekends/holidays.
INTRADAY_LOOKBACK_DAYS = _env_int("INTRADAY_LOOKBACK_DAYS", 7)


# -----------------------------------------------------------------------------
# DhanHQ wire identifiers (rarely tuned; exposed for completeness)
# -----------------------------------------------------------------------------
# Change these only if you swap underlying (e.g. BANKNIFTY) or DhanHQ
# changes its segment naming. Reference:
#   https://dhanhq.co/docs/v2/annexure/
NIFTY_INDEX_SECURITY_ID = _env_int("NIFTY_INDEX_SECURITY_ID", 13)
NIFTY_INDEX_EXCHANGE_SEGMENT = _env_str("NIFTY_INDEX_EXCHANGE_SEGMENT", "IDX_I")
NIFTY_INDEX_INSTRUMENT_TYPE = _env_str("NIFTY_INDEX_INSTRUMENT_TYPE", "INDEX")
OPTION_EXCHANGE_SEGMENT = _env_str("OPTION_EXCHANGE_SEGMENT", "NSE_FNO")

# NIFTY listed strikes step in 50-point increments. The ATM rule rounds the
# live spot to the nearest multiple of this step.
ATM_STRIKE_STEP = _env_float("ATM_STRIKE_STEP", 50.0)


# =============================================================================
# RENKO STRATEGY CONSTANTS (Tier 3)
# =============================================================================
RENKO_LOTS = _env_int("RENKO_LOTS", 1)
RENKO_MAX_LOSS = _env_float("RENKO_MAX_LOSS", 5500.0)
RENKO_POLL_SECONDS = _env_int("RENKO_POLL_SECONDS", 2)

# Trading window: entries before TRADING_START are ignored; positions are
# force-closed at SQUARE_OFF.
RENKO_TRADING_START_HOUR = _env_int("RENKO_TRADING_START_HOUR", 9)
RENKO_TRADING_START_MINUTE = _env_int("RENKO_TRADING_START_MINUTE", 15)
RENKO_SQUARE_OFF_HOUR = _env_int("RENKO_SQUARE_OFF_HOUR", 15)
RENKO_SQUARE_OFF_MINUTE = _env_int("RENKO_SQUARE_OFF_MINUTE", 15)

# Renko's midday "no new entries" window (e.g. lunchtime chop avoidance).
# Existing positions keep running through the window; only fresh entries
# are blocked. Default 12:00 - 13:00 IST.
RENKO_NO_TRADE_START_HOUR = _env_int("RENKO_NO_TRADE_START_HOUR", 12)
RENKO_NO_TRADE_START_MINUTE = _env_int("RENKO_NO_TRADE_START_MINUTE", 0)
RENKO_NO_TRADE_END_HOUR = _env_int("RENKO_NO_TRADE_END_HOUR", 13)
RENKO_NO_TRADE_END_MINUTE = _env_int("RENKO_NO_TRADE_END_MINUTE", 0)


# =============================================================================
# EMA TREND STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Sizing/risk + trading window. Indicator periods are tuned in
# `EMA_STRATEGY_CONFIG` further down (also env-driven).
EMA_LOTS = _env_int("EMA_LOTS", 1)
EMA_MAX_LOSS = _env_float("EMA_MAX_LOSS", 5500.0)
EMA_POLL_SECONDS = _env_int("EMA_POLL_SECONDS", 2)

EMA_TRADING_START_HOUR = _env_int("EMA_TRADING_START_HOUR", 9)
EMA_TRADING_START_MINUTE = _env_int("EMA_TRADING_START_MINUTE", 25)
EMA_SQUARE_OFF_HOUR = _env_int("EMA_SQUARE_OFF_HOUR", 15)
EMA_SQUARE_OFF_MINUTE = _env_int("EMA_SQUARE_OFF_MINUTE", 15)

# Number of minutes per derived EMA candle. The 1-min source data is
# resampled into N-minute candles before EMA evaluation.
EMA_DERIVED_TIMEFRAME_MINUTES = _env_int("EMA_DERIVED_TIMEFRAME_MINUTES", 5)


# =============================================================================
# HEIKIN ASHI STRATEGY CONSTANTS (Tier 3)
# =============================================================================
HEIKIN_LOTS = _env_int("HEIKIN_LOTS", 1)
HEIKIN_MAX_LOSS = _env_float("HEIKIN_MAX_LOSS", 5500.0)
HEIKIN_POLL_SECONDS = _env_int("HEIKIN_POLL_SECONDS", 5)

HEIKIN_TRADING_START_HOUR = _env_int("HEIKIN_TRADING_START_HOUR", 9)
HEIKIN_TRADING_START_MINUTE = _env_int("HEIKIN_TRADING_START_MINUTE", 15)
HEIKIN_SQUARE_OFF_HOUR = _env_int("HEIKIN_SQUARE_OFF_HOUR", 15)
HEIKIN_SQUARE_OFF_MINUTE = _env_int("HEIKIN_SQUARE_OFF_MINUTE", 15)

# Bollinger period and stddev used by the Heikin Ashi signal builder.
HEIKIN_BOLL_PERIOD = _env_int("HEIKIN_BOLL_PERIOD", 20)
HEIKIN_BOLL_STD = _env_float("HEIKIN_BOLL_STD", 2.0)


# =============================================================================
# PROFIT SHOOTER STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Sizing/risk + trading window. Indicator/pin-bar tuning lives further down
# inside `PROFIT_SHOOTER_STRATEGY_CONFIG` (also env-driven).
PROFIT_SHOOTER_POLL_SECONDS = _env_int("PROFIT_SHOOTER_POLL_SECONDS", 2)

PROFIT_SHOOTER_TRADING_START_HOUR = _env_int("PROFIT_SHOOTER_TRADING_START_HOUR", 9)
PROFIT_SHOOTER_TRADING_START_MINUTE = _env_int("PROFIT_SHOOTER_TRADING_START_MINUTE", 25)
PROFIT_SHOOTER_SQUARE_OFF_HOUR = _env_int("PROFIT_SHOOTER_SQUARE_OFF_HOUR", 15)
PROFIT_SHOOTER_SQUARE_OFF_MINUTE = _env_int("PROFIT_SHOOTER_SQUARE_OFF_MINUTE", 15)

# Profit Shooter never opens a new trade from a setup printed at or after
# this time, because the daily 15:15 cutoff would close it almost
# immediately. Keep this within ~1 minute of the square-off time.
PROFIT_SHOOTER_LAST_SETUP_HOUR = _env_int("PROFIT_SHOOTER_LAST_SETUP_HOUR", 15)
PROFIT_SHOOTER_LAST_SETUP_MINUTE = _env_int("PROFIT_SHOOTER_LAST_SETUP_MINUTE", 14)


# =============================================================================
# SUPERTREND BULLISH (HEDGED PE SPREAD) CONSTANTS (Tier 3)
# =============================================================================
# Strategy timeframe: the 1-minute source data is resampled locally into
# N-minute candles before Supertrend is evaluated.
SUPERTREND_TIMEFRAME_MINUTES = _env_int("SUPERTREND_TIMEFRAME_MINUTES", 3)

# Supertrend indicator parameters (ATR length and band factor).
SUPERTREND_ATR_LENGTH = _env_int("SUPERTREND_ATR_LENGTH", 14)
SUPERTREND_FACTOR = _env_float("SUPERTREND_FACTOR", 3.0)

# Target premiums (INR) for the bullish hedged setup.
MAIN_PUT_TARGET_PREMIUM = _env_float("MAIN_PUT_TARGET_PREMIUM", 160.0)
HEDGE_PUT_TARGET_PREMIUM = _env_float("HEDGE_PUT_TARGET_PREMIUM", 10.0)

# Trading window for the Supertrend bullish strategy.
SUPERTREND_TRADING_START_HOUR = _env_int("SUPERTREND_TRADING_START_HOUR", 9)
SUPERTREND_TRADING_START_MINUTE = _env_int("SUPERTREND_TRADING_START_MINUTE", 18)
SUPERTREND_SQUARE_OFF_HOUR = _env_int("SUPERTREND_SQUARE_OFF_HOUR", 15)
SUPERTREND_SQUARE_OFF_MINUTE = _env_int("SUPERTREND_SQUARE_OFF_MINUTE", 14)
SUPERTREND_POLL_SECONDS = _env_int("SUPERTREND_POLL_SECONDS", 2)

# After every exit, no new entry is even evaluated for this many minutes.
POST_EXIT_COOLDOWN_MINUTES = _env_int("POST_EXIT_COOLDOWN_MINUTES", 5)

# Sizing and risk for the bullish leg.
BULLISH_LOTS = _env_int("BULLISH_LOTS", 1)
BULLISH_MAX_LOSS = _env_float("BULLISH_MAX_LOSS", 5500.0)


# =============================================================================
# DONCHIAN BEARISH (HEDGED CE SPREAD) CONSTANTS (Tier 3)
# =============================================================================
# 5-minute Donchian. Same locally-resampled-from-1m pattern as Supertrend.
BEARISH_TIMEFRAME_MINUTES = _env_int("BEARISH_TIMEFRAME_MINUTES", 5)

# Donchian Channel length (in BEARISH_TIMEFRAME_MINUTES-minute bars).
DONCHIAN_LENGTH = _env_int("DONCHIAN_LENGTH", 20)

# Target premiums (INR) for the bearish hedged setup.
MAIN_CALL_TARGET_PREMIUM = _env_float("MAIN_CALL_TARGET_PREMIUM", 160.0)
HEDGE_CALL_TARGET_PREMIUM = _env_float("HEDGE_CALL_TARGET_PREMIUM", 10.0)

# Bearish entry window. (See note in the original Hedged Puts file: the
# spec said "10:59 PM" but Indian markets close at 15:30, so we read it
# as 10:59 AM.) After the cutoff no NEW entries are taken; positions
# already opened keep running until SL / target / 15:14 fires.
BEARISH_TRADING_START_HOUR = _env_int("BEARISH_TRADING_START_HOUR", 9)
BEARISH_TRADING_START_MINUTE = _env_int("BEARISH_TRADING_START_MINUTE", 25)
BEARISH_ENTRY_CUTOFF_HOUR = _env_int("BEARISH_ENTRY_CUTOFF_HOUR", 10)
BEARISH_ENTRY_CUTOFF_MINUTE = _env_int("BEARISH_ENTRY_CUTOFF_MINUTE", 59)
BEARISH_POLL_SECONDS = _env_int("BEARISH_POLL_SECONDS", 2)

# SL and target for the bearish leg are expressed as percentages of the
# NIFTY spot at entry, NOT as percentages of the option premium.
#   * SL fires when spot >= entry_spot * (1 + BEARISH_SL_UP_PCT)
#   * TP fires when spot <= entry_spot * (1 - BEARISH_TARGET_DOWN_PCT)
BEARISH_SL_UP_PCT = _env_float("BEARISH_SL_UP_PCT", 0.0020)
BEARISH_TARGET_DOWN_PCT = _env_float("BEARISH_TARGET_DOWN_PCT", 0.0040)

# Sizing and risk for the bearish leg.
BEARISH_LOTS = _env_int("BEARISH_LOTS", 1)
BEARISH_MAX_LOSS = _env_float("BEARISH_MAX_LOSS", 5500.0)


# =============================================================================
# DELTA-0.2 HEDGED SPREAD STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Background for readers new to options trading:
#
# - "Delta" is one of the option Greeks. For a CALL, delta ranges from 0
#   (deep OTM, market expects strike won't be reached) to ~1 (deep ITM).
#   For a PUT, delta is reported as a negative number from -1 (deep ITM)
#   to 0 (deep OTM). A common shorthand is "20-delta strike" meaning
#   |delta| ~= 0.20 - i.e. the option roughly has a 20% chance of
#   finishing in-the-money. These are typically a few strikes OTM.
#
# - "Premium" = the price of the option contract (its LTP).
#
# - "OTM" = Out-of-the-money. For CE it means strike > spot; for PE it
#   means strike < spot. Hedging here means BUYING a further-OTM option
#   so a runaway move against us has a hard cap on the loss.
#
# - This is a CREDIT spread: we SELL the monitored strike (collecting
#   premium) and BUY a cheaper further-OTM strike (paying a small
#   premium). Net = positive credit. We profit if both legs expire
#   worthless or if the monitored leg's price decays.
#
# Strategy spec (the user's brief; one CE side + one PE side, INDEPENDENT):
#
#   1. At DELTA20_TRADING_START_*:* (default 09:20), pull the live option
#      chain for the current-week expiry and pick:
#        * the CE strike with delta closest to +DELTA20_TARGET_DELTA
#        * the PE strike with delta closest to -DELTA20_TARGET_DELTA
#      Snapshot each leg's premium at that moment as the "reference".
#
#   2. ENTRY (per side, independent of the other side):
#      When the monitored CE/PE leg's LTP falls to <= ref * (1 - drop)
#      (default: 5% drop), open the spread:
#        * SELL the monitored strike (this is the "main" leg)
#        * BUY the strike DELTA20_HEDGE_STRIKES_OTM further OTM
#          (CE side: higher strike; PE side: lower strike) for cap-loss
#          protection.
#      A side may enter even if the other side is already running.
#
#   3. EXIT conditions (per side, independent):
#      a. The monitored strike's LTP reaches DELTA20_EXIT_MULTIPLIER x
#         the reference premium (default: 3x). Closes ONLY that side.
#         Reading: "the option got 3x more expensive => the market is
#         pushing toward our short strike => bail out before assignment."
#      b. STRATEGY-WIDE max-loss: realized + open PnL across BOTH sides
#         <= -(DELTA20_MAX_LOSS_PER_LOT * lots). Closes everything AND
#         stops the worker for the day.
#      c. Time cutoff at DELTA20_SQUARE_OFF_*:* (default 15:20). Closes
#         anything still open.
#
#   Once a side has exited it is NOT re-opened the same day. The
#   monitor + hedge LTP subscriptions for that side are released so we
#   stop spending API budget watching them.
DELTA20_LOTS = _env_int("DELTA20_LOTS", 1)
# Per-lot stoploss as the user phrased it; the worker translates that into
# an absolute INR cap by multiplying with DELTA20_LOTS at risk-check time.
DELTA20_MAX_LOSS_PER_LOT = _env_float("DELTA20_MAX_LOSS_PER_LOT", 5000.0)
DELTA20_MAX_LOSS = DELTA20_MAX_LOSS_PER_LOT * DELTA20_LOTS
DELTA20_POLL_SECONDS = _env_int("DELTA20_POLL_SECONDS", 2)

# Reference-capture time and force-exit cutoff.
DELTA20_TRADING_START_HOUR = _env_int("DELTA20_TRADING_START_HOUR", 9)
DELTA20_TRADING_START_MINUTE = _env_int("DELTA20_TRADING_START_MINUTE", 20)
DELTA20_SQUARE_OFF_HOUR = _env_int("DELTA20_SQUARE_OFF_HOUR", 15)
DELTA20_SQUARE_OFF_MINUTE = _env_int("DELTA20_SQUARE_OFF_MINUTE", 20)

# The "20-delta" target. Stored as an unsigned magnitude; the worker compares
# CE delta against +TARGET and PE delta against -TARGET when picking strikes.
DELTA20_TARGET_DELTA = _env_float("DELTA20_TARGET_DELTA", 0.20)

# Entry trigger: the monitored strike's LTP must drop to <= ref * (1 - drop).
DELTA20_ENTRY_DROP_PCT = _env_float("DELTA20_ENTRY_DROP_PCT", 0.05)

# Exit target: the monitored strike's LTP rising to >= ref * multiplier.
DELTA20_EXIT_MULTIPLIER = _env_float("DELTA20_EXIT_MULTIPLIER", 3.0)

# Hedge offset, expressed in number of strike steps (each step = ATM_STRIKE_STEP).
# 4 strikes on NIFTY = 200 points further OTM than the monitored strike.
DELTA20_HEDGE_STRIKES_OTM = _env_int("DELTA20_HEDGE_STRIKES_OTM", 4)

# Backoff between retries when the 09:20 reference capture fails (e.g. the
# DhanHQ option_chain endpoint is rate-limited or returns an empty payload).
DELTA20_CAPTURE_RETRY_SECONDS = _env_int("DELTA20_CAPTURE_RETRY_SECONDS", 5)


logger = logging.getLogger(LOGGER_NAME)


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    """
    Configure the shared root logger for the whole runner.

    Why we configure the *root* logger (not just our named logger):
    - This file starts a fetcher thread plus six worker threads.
    - Every thread creates a child logger like
      `nifty_multi_strategy_master_front_test_dhanhq.renko`.
    - Attaching handlers once at the root makes every child's output flow
      through the same file + console pipeline automatically.

    This function is idempotent: if handlers are already attached (for
    example, if it is called twice from a notebook), it returns early
    instead of stacking duplicate handlers.
    """
    configured_logger = logging.getLogger()
    if configured_logger.handlers:
        return logging.getLogger(LOGGER_NAME)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    configured_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    configured_logger.addHandler(file_handler)
    configured_logger.addHandler(stream_handler)
    return logging.getLogger(LOGGER_NAME)


# =============================================================================
# DYNAMIC MODULE IMPORT (for files whose paths contain spaces)
# =============================================================================
def load_module(module_name: str, file_path: Path):
    """
    Import a local Python file by its absolute path.

    Why this helper exists:
    - Several strategy folders contain spaces in their names, which break
      the standard `import package.module` syntax.
    - Some strategy logic files sit in folders that are not on Python's
      normal import path when this script starts.
    - We only want to import the reusable *signal/indicator logic* modules
      from those folders, not the bigger front-test controllers next to
      them.

    Steps:
    1. Temporarily add the file's directory to `sys.path`.
    2. Build an import spec from the file path.
    3. Execute the module once and cache it in `sys.modules`.
    4. Clean up the temporary `sys.path` entry.
    """
    inserted_path = False
    parent_dir = str(file_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        inserted_path = True

    try:
        if module_name in sys.modules:
            return sys.modules[module_name]

        module_spec = importlib.util.spec_from_file_location(module_name, file_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not build import spec for {file_path}")

        module = importlib.util.module_from_spec(module_spec)
        # Insert into `sys.modules` BEFORE exec so dataclass-with-slots and
        # other features that look up the module mid-execution work.
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module
    finally:
        if inserted_path and sys.path and sys.path[0] == parent_dir:
            sys.path.pop(0)


# Reusable signal/indicator logic modules.
# These are the ONLY external strategy code that the master file pulls in.
# All trade management lives in this file's worker classes.
RENKO_LOGIC = load_module(
    "master_renko_strategy_logic_9_21",
    ROOT_DIR / "Renko Strategy" / "renko_strategy_logic_9_21.py",
)
EMA_LOGIC = load_module(
    "master_ema_trend_strategy_logic",
    ROOT_DIR / "EMA Trend Strategy" / "ema_trend_strategy_logic.py",
)
HEIKIN_LOGIC = load_module(
    "master_heikin_ashi_strategy_logic",
    ROOT_DIR / "Heikin Ashi Strategy" / "heikin_ashi_strategy_logic.py",
)
PROFIT_SHOOTER_LOGIC = load_module(
    "master_profit_shooter_strategy_logic",
    ROOT_DIR / "Profit Shooter Strategy" / "profit_shooter_strategy_logic.py",
)
SUPERTREND_LOGIC = load_module(
    "master_supertrend_signal_generator_bullish",
    ROOT_DIR / "Supertrend Strategies" / "Supertrend Signal Generator Bullish.py",
)
DONCHIAN_BEARISH_LOGIC = load_module(
    "master_donchian_signal_generator_bearish",
    ROOT_DIR / "Supertrend Strategies" / "Donchian Signal Generator Bearish.py",
)


# =============================================================================
# STRATEGY CONFIG OBJECTS
# =============================================================================
# The EMA Trend logic uses a config dataclass for its tunable parameters.
# Building it here (with `_env_*` overrides) keeps every parameter visible in
# one block and lets ops tune behaviour without editing strategy code.
EMA_STRATEGY_CONFIG = EMA_LOGIC.EMATrendConfig(
    ema_fast_period=_env_int("EMA_TREND_FAST_PERIOD", 4),
    ema_mid_period=_env_int("EMA_TREND_MID_PERIOD", 11),
    ema_slow_period=_env_int("EMA_TREND_SLOW_PERIOD", 18),
    atr_period=_env_int("EMA_TREND_ATR_PERIOD", 14),
    adx_period=_env_int("EMA_TREND_ADX_PERIOD", 14),
    slope_lookback=_env_int("EMA_TREND_SLOPE_LOOKBACK", 3),
    adx_threshold=_env_float("EMA_TREND_ADX_THRESHOLD", 20.0),
    distance_atr_multiplier=_env_float("EMA_TREND_DISTANCE_ATR_MULT", 0.5),
    ema11_slope_atr_multiplier=_env_float("EMA_TREND_EMA11_SLOPE_ATR_MULT", 0.3),
    ema18_slope_atr_multiplier=_env_float("EMA_TREND_EMA18_SLOPE_ATR_MULT", 0.2),
)

# Profit Shooter also uses a config dataclass. Same idea: every tunable goes
# through `_env_*` so the env can override defaults without editing code.
PROFIT_SHOOTER_STRATEGY_CONFIG = PROFIT_SHOOTER_LOGIC.ProfitShooterConfig(
    sma_fast_period=_env_int("PROFIT_SHOOTER_SMA_FAST_PERIOD", 20),
    sma_slow_period=_env_int("PROFIT_SHOOTER_SMA_SLOW_PERIOD", 200),
    atr_period=_env_int("PROFIT_SHOOTER_ATR_PERIOD", 14),
    trailing_ema_period=_env_int("PROFIT_SHOOTER_TRAILING_EMA_PERIOD", 9),
    trend_lookback=_env_int("PROFIT_SHOOTER_TREND_LOOKBACK", 3),
    pullback_lookback=_env_int("PROFIT_SHOOTER_PULLBACK_LOOKBACK", 3),
    pullback_min_count=_env_int("PROFIT_SHOOTER_PULLBACK_MIN_COUNT", 2),
    proximity_atr_multiple=_env_float("PROFIT_SHOOTER_PROXIMITY_ATR_MULT", 0.5),
    tick_size=_env_float("PROFIT_SHOOTER_TICK_SIZE", 0.05),
    sl_atr_buffer=_env_float("PROFIT_SHOOTER_SL_ATR_BUFFER", 0.2),
    target_r_multiple=_env_float("PROFIT_SHOOTER_TARGET_R_MULT", 1.5),
    pin_bar_body_max_ratio=_env_float("PROFIT_SHOOTER_PIN_BAR_BODY_MAX_RATIO", 0.30),
    pin_bar_long_wick_min_ratio=_env_float("PROFIT_SHOOTER_PIN_BAR_LONG_WICK_MIN_RATIO", 0.60),
    pin_bar_wick_to_body_multiple=_env_float("PROFIT_SHOOTER_PIN_BAR_WICK_TO_BODY_MULT", 2.0),
    pin_bar_opposite_wick_max_ratio=_env_float("PROFIT_SHOOTER_PIN_BAR_OPPOSITE_WICK_MAX_RATIO", 0.15),
    pin_bar_end_zone_ratio=_env_float("PROFIT_SHOOTER_PIN_BAR_END_ZONE_RATIO", 0.40),
)
# Operational sizing/risk for Profit Shooter (separate from signal config).
PROFIT_SHOOTER_LOTS = _env_int("PROFIT_SHOOTER_LOTS", 1)
PROFIT_SHOOTER_STARTING_CAPITAL = _env_float("PROFIT_SHOOTER_STARTING_CAPITAL", 600000.0)
PROFIT_SHOOTER_DAILY_MAX_LOSS_PCT = _env_float("PROFIT_SHOOTER_DAILY_MAX_LOSS_PCT", 0.03)
PROFIT_SHOOTER_MAX_LOSS = PROFIT_SHOOTER_STARTING_CAPITAL * PROFIT_SHOOTER_DAILY_MAX_LOSS_PCT
PROFIT_SHOOTER_MIN_BARS = (
    max(
        PROFIT_SHOOTER_STRATEGY_CONFIG.sma_fast_period,
        PROFIT_SHOOTER_STRATEGY_CONFIG.sma_slow_period,
        PROFIT_SHOOTER_STRATEGY_CONFIG.atr_period,
        PROFIT_SHOOTER_STRATEGY_CONFIG.trailing_ema_period,
    )
    + max(
        PROFIT_SHOOTER_STRATEGY_CONFIG.trend_lookback,
        PROFIT_SHOOTER_STRATEGY_CONFIG.pullback_lookback,
    )
    + 2
)

# Settings objects for the two Hedged Puts strategies.
SUPERTREND_SETTINGS = SUPERTREND_LOGIC.SupertrendSettings(
    atr_length=SUPERTREND_ATR_LENGTH,
    factor=SUPERTREND_FACTOR,
)
DONCHIAN_BEARISH_SETTINGS = DONCHIAN_BEARISH_LOGIC.DonchianSettings(
    length=DONCHIAN_LENGTH,
)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class MarketSnapshot:
    """
    One snapshot of a centrally fetched timeframe.

    Fields:
    - `timeframe`        : source timeframe string ("1" for the 1-minute pool).
    - `frame`            : normalized OHLC DataFrame.
    - `source_candle_ts` : timestamp of the latest candle in `frame`.
    - `candle_signature` : lightweight fingerprint of the latest row's state.
    - `fetched_at`       : wall-clock time when the fetch completed.

    Why we keep both `source_candle_ts` and `candle_signature`:
    - During a live 1-minute candle, the timestamp does not change but the
      high / low / close keep updating second by second.
    - `source_candle_ts` is enough to detect a NEW candle.
    - `candle_signature` is needed to detect content changes WITHIN the
      currently-forming candle. Workers use it to decide whether the strategy
      logic actually needs to re-run.
    """

    timeframe: str
    frame: pd.DataFrame
    source_candle_ts: Optional[pd.Timestamp]
    candle_signature: Optional[tuple]
    fetched_at: datetime


@dataclass
class PaperPosition:
    """
    Runtime state for ONE single-leg paper trade (used by the four ATM
    workers).

    Each ATM strategy worker owns exactly one of these. When the strategy is
    flat, the object's `active` flag is False. When in a trade, the object
    holds the option leg's identifiers, the entry fill, and any extra
    strategy-specific bookkeeping (e.g. trailing flags for Profit Shooter).
    """

    active: bool = False
    direction: str = ""
    symbol: str = ""
    quantity: int = 0
    entry_order_id: str = ""
    exit_order_id: str = ""
    # Strategy-side prices (NIFTY spot levels), used for stop/target rules.
    entry_underlying: float = 0.0
    stop_underlying: float = 0.0
    # Profit Shooter keeps its original 1.5R target around because trailing
    # mode arms only after that target is touched.
    target_underlying: float = 0.0
    # Renko-only flag.
    rr_armed: bool = False
    # Profit Shooter trailing-mode flags. Other strategies leave these False.
    trailing_active: bool = False
    pending_trailing_exit: bool = False
    # Actual paper-fill price on the option leg (used for PnL).
    entry_trade_price: float = 0.0
    # Option-leg metadata, locked at entry and reused on exit.
    option_security_id: int = 0
    option_exchange_segment: str = ""
    option_right: str = ""
    option_strike: float = 0.0
    option_expiry: Optional[date] = None
    option_lot_size: int = 0


@dataclass
class HedgedPaperPosition:
    """
    Runtime state for a TWO-leg hedged paper trade (used by the two Hedged
    Puts workers).

    Holds the main (sold) leg and the hedge (bought) leg side-by-side. A flat
    worker holds a default-constructed instance with `active=False`.

    Why a separate dataclass instead of overloading PaperPosition:
    - The single-leg flow has a different set of fields (rr_armed, trailing
      flags) that simply do not apply to a hedged position.
    - The hedged flow has TWO of every leg field; folding both into
      PaperPosition would force every read site to know which leg is which.
    - Keeping the shapes separate makes each worker's state machine
      type-checkable and unambiguous.
    """

    active: bool = False
    direction: str = ""

    # Snapshot at entry, used for logging / audit / SL-target anchoring.
    entry_underlying: float = 0.0
    entry_timestamp: Optional[datetime] = None

    # -- Main leg (SELL PE for bullish, SELL CE for bearish) ----------------
    main_symbol: str = ""
    main_side: str = ""
    main_security_id: int = 0
    main_exchange_segment: str = ""
    main_right: str = ""
    main_strike: float = 0.0
    main_expiry: Optional[date] = None
    main_lot_size: int = 0
    main_quantity: int = 0
    main_entry_price: float = 0.0
    main_entry_order_id: str = ""

    # -- Hedge leg (BUY PE for bullish, BUY CE for bearish) -----------------
    hedge_symbol: str = ""
    hedge_side: str = ""
    hedge_security_id: int = 0
    hedge_exchange_segment: str = ""
    hedge_right: str = ""
    hedge_strike: float = 0.0
    hedge_expiry: Optional[date] = None
    hedge_lot_size: int = 0
    hedge_quantity: int = 0
    hedge_entry_price: float = 0.0
    hedge_entry_order_id: str = ""


@dataclass
class LTPSnapshot:
    """
    Cached last-traded-price snapshot for one (segment, security_id) pair.

    The fetcher batches LTP requests and writes results into the shared store
    using one of these per (segment, sec_id). Workers read from the cache;
    they do not call the broker each time they need a price.
    """

    segment: str
    security_id: int
    ltp: float
    fetched_at: datetime


@dataclass
class OptionSubscription:
    """
    Metadata for one option leg the fetcher should keep polling LTPs for.

    Workers register a leg right after entering a paper trade and unregister
    it on exit. The fetcher reads the subscription set every poll and includes
    those security_ids in its batched ticker_data call so all live MTM/exit
    pricing stays current.
    """

    security_id: int
    exchange_segment: str
    trading_symbol: str
    right: str
    strike: float
    expiry: Optional[date]


# =============================================================================
# SHARED THREAD-SAFE DATA STORE
# =============================================================================
class SharedMarketDataStore:
    """
    Thread-safe storage for centrally fetched OHLC and LTP values.

    This is the single rendezvous between the producer thread (the fetcher)
    and the consumer threads (the six workers). One `threading.Lock` guards
    every mutation, so a reader can never see a half-written snapshot.

    Three independent pools live inside:
    1. `_snapshots`          - source-timeframe OHLC (we only use "1" here).
    2. `_ltp_snapshots`      - latest LTPs keyed by (segment, security_id).
    3. `_option_subscriptions` - option legs the fetcher should poll.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._ltp_snapshots: dict[tuple[str, int], LTPSnapshot] = {}
        self._option_subscriptions: dict[tuple[str, int], OptionSubscription] = {}

    # ------------------------------------------------------------------
    # OHLC pool
    # ------------------------------------------------------------------
    def update(self, timeframe: str, frame: pd.DataFrame) -> MarketSnapshot:
        """
        Atomically replace the stored snapshot for `timeframe`.

        The new snapshot is built BEFORE the lock is taken; only the swap
        happens under lock so the critical section stays small.
        """
        source_candle_ts = None
        if frame is not None and not frame.empty and "timestamp" in frame.columns:
            source_candle_ts = pd.to_datetime(frame.iloc[-1]["timestamp"])
        candle_signature = build_last_row_signature(frame)

        snapshot = MarketSnapshot(
            timeframe=str(timeframe),
            frame=frame.copy(),
            source_candle_ts=source_candle_ts,
            candle_signature=candle_signature,
            fetched_at=datetime.now(),
        )
        with self._lock:
            self._snapshots[str(timeframe)] = snapshot
        return snapshot

    def get(self, timeframe: str) -> Optional[MarketSnapshot]:
        """
        Return a fresh `MarketSnapshot` whose `frame` is a pandas copy.

        Workers can mutate their local copy freely without affecting the
        shared store.
        """
        with self._lock:
            snapshot = self._snapshots.get(str(timeframe))
        if snapshot is None:
            return None

        return MarketSnapshot(
            timeframe=snapshot.timeframe,
            frame=snapshot.frame.copy(),
            source_candle_ts=snapshot.source_candle_ts,
            candle_signature=snapshot.candle_signature,
            fetched_at=snapshot.fetched_at,
        )

    # ------------------------------------------------------------------
    # LTP pool
    # ------------------------------------------------------------------
    def update_ltp_map(self, ltp_map: dict[tuple[str, int], float]) -> None:
        """
        Bulk update the LTP cache from a dict of {(segment, sec_id): price}.

        Non-positive prices are ignored. Each call is one atomic block under
        the lock so concurrent readers always see a coherent set.
        """
        if not ltp_map:
            return
        now = datetime.now()
        with self._lock:
            for (segment, sec_id), price in ltp_map.items():
                if float(price) <= 0:
                    continue
                key = (str(segment), int(sec_id))
                self._ltp_snapshots[key] = LTPSnapshot(
                    segment=str(segment),
                    security_id=int(sec_id),
                    ltp=float(price),
                    fetched_at=now,
                )

    def get_ltp_by_secid(
        self,
        segment: str,
        security_id: int,
        fallback: float = 0.0,
    ) -> float:
        """
        Read the newest cached LTP for one (segment, security_id) pair.

        This method NEVER calls the broker; it just returns whatever the
        fetcher has already cached, falling back to `fallback` when nothing
        positive is available yet.
        """
        key = (str(segment), int(security_id))
        with self._lock:
            snapshot = self._ltp_snapshots.get(key)
            if snapshot is not None and snapshot.ltp > 0:
                return snapshot.ltp
        return float(fallback)

    # ------------------------------------------------------------------
    # Option-leg subscription pool
    # ------------------------------------------------------------------
    def register_option_subscription(self, subscription: OptionSubscription) -> None:
        """Tell the fetcher to keep polling LTP for this option leg."""
        key = (str(subscription.exchange_segment), int(subscription.security_id))
        with self._lock:
            self._option_subscriptions[key] = subscription

    def unregister_option_subscription(self, segment: str, security_id: int) -> None:
        """Drop a leg once a paper trade has been closed."""
        key = (str(segment), int(security_id))
        with self._lock:
            self._option_subscriptions.pop(key, None)

    def snapshot_option_subscriptions(self) -> list[OptionSubscription]:
        """Return a list copy of all currently subscribed option legs."""
        with self._lock:
            return list(self._option_subscriptions.values())


# =============================================================================
# SMALL UTILITY HELPERS
# =============================================================================
def _safe_float(value, default=0.0) -> float:
    """Convert anything to float without crashing; fall back to `default`."""
    try:
        return float(value)
    except Exception:
        return default


def _to_int_safe(value, default=0) -> int:
    """Convert anything to int without crashing; fall back to `default`."""
    try:
        return int(float(value))
    except Exception:
        return default


def _first_existing_col(df: pd.DataFrame, names) -> Optional[str]:
    """
    Case-insensitively look up the first column from `names` that exists in
    `df`. Used for instrument-master parsing where column names can vary
    slightly between CSV snapshots.
    """
    col_map = {str(column).strip().lower(): column for column in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _infer_epoch_unit(values: pd.Series) -> str:
    """
    Guess whether a numeric timestamp series is in seconds, milliseconds,
    or microseconds based on the magnitude of its largest value.

    DhanHQ historical responses can return any of the three units depending
    on endpoint version; we infer per-call rather than hard-code one unit.
    """
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return "ms"

    max_value = float(nums.max())
    if max_value > 1e14:
        return "us"
    if max_value > 1e11:
        return "ms"
    return "s"


def normalize_dhan_intraday_response(resp) -> pd.DataFrame:
    """
    Normalize a `dhanhq.intraday_minute_data` response into a standard
    OHLC DataFrame with columns: timestamp, open, high, low, close.

    The DhanHQ wire shape is:
        {
            "status": "success",
            "data":   {"timestamp": [...], "open": [...], "high": [...],
                       "low": [...], "close": [...], "volume": [...]},
            "remarks": ...
        }

    All quirks (UTC vs IST, second/ms/us epoch, etc.) are flattened here so
    the rest of the pipeline can assume one consistent schema.
    """
    if not isinstance(resp, dict):
        raise ValueError(f"Unexpected dhanhq response type: {type(resp).__name__}")

    status = str(resp.get("status", "")).strip().lower()
    if status and status != "success":
        remarks = resp.get("remarks") or resp.get("data") or resp.get("message")
        raise ValueError(f"dhanhq API failed: status={status}, remarks={remarks}")

    data = resp.get("data")
    if data is None:
        raise ValueError("dhanhq response has no 'data' key")

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("dhanhq response data is empty")

    col_map = {str(column).lower(): column for column in df.columns}

    ts_col = None
    for candidate in ["timestamp", "start_time", "starttime", "datetime", "date", "time"]:
        if candidate in col_map:
            ts_col = col_map[candidate]
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {list(df.columns)}")

    o_col = col_map.get("open")
    h_col = col_map.get("high")
    l_col = col_map.get("low")
    c_col = col_map.get("close")
    if not all([o_col, h_col, l_col, c_col]):
        raise ValueError(f"OHLC columns missing in {list(df.columns)}")

    ts_numeric = pd.to_numeric(df[ts_col], errors="coerce")
    if ts_numeric.notna().sum() >= max(1, len(df) // 2):
        unit = _infer_epoch_unit(ts_numeric)
        ts = pd.to_datetime(ts_numeric, unit=unit, errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    # Convert UTC timestamps into Indian market time. Every strategy in the
    # project reasons in Asia/Kolkata local time.
    ts_local = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": ts_local,
            "open": pd.to_numeric(df[o_col], errors="coerce"),
            "high": pd.to_numeric(df[h_col], errors="coerce"),
            "low": pd.to_numeric(df[l_col], errors="coerce"),
            "close": pd.to_numeric(df[c_col], errors="coerce"),
        }
    ).dropna()

    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(out) < MIN_BARS:
        raise ValueError(f"Need >= {MIN_BARS} bars, got {len(out)}")
    return out


def build_last_row_signature(frame: pd.DataFrame) -> Optional[tuple]:
    """
    Fingerprint the latest row of an OHLC table.

    Workers use this fingerprint to decide whether the latest candle changed
    enough to warrant re-running the strategy. Without this guard, the
    fetcher's continuous polling would force the strategy logic to repeat
    every 2 seconds even when nothing changed.

    Composition: (row_count, latest_ts, last_open, last_high, last_low,
    last_close). Simple but strong enough to detect intra-minute high/low
    updates as well as new candle arrivals.
    """
    if frame is None or frame.empty:
        return None

    last_row = frame.iloc[-1]
    last_ts = None
    if "timestamp" in frame.columns:
        try:
            last_ts = pd.to_datetime(last_row["timestamp"], errors="coerce")
        except Exception:
            last_ts = last_row["timestamp"]

    values = [len(frame), last_ts]
    for column in ("open", "high", "low", "close"):
        if column in frame.columns:
            values.append(round(_safe_float(last_row[column], 0.0), 8))
        else:
            values.append(None)
    return tuple(values)


def resample_ohlc_from_1m(ohlc: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Resample 1-minute OHLC into a higher timeframe, KEEPING ONLY COMPLETE BARS.

    Worked example: with `timeframe_minutes=5`, a 5-minute bar is kept only
    if all 5 underlying 1-minute candles exist. That prevents any worker
    from ever evaluating a partially formed bar, which is the whole point
    of the strategies' "evaluate on closed candles only" rule.

    Steps:
    1. Sort by timestamp.
    2. Resample with first/max/min/last aggregation.
    3. Count how many 1-min rows fed each resampled bucket.
    4. Drop buckets that did not get the full count.
    """
    if int(timeframe_minutes) <= 1:
        return ohlc.copy()

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing = [column for column in required_columns if column not in ohlc.columns]
    if missing:
        raise ValueError(f"Missing required columns for resampling: {', '.join(missing)}")

    frame = ohlc.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        return frame

    rule = f"{int(timeframe_minutes)}min"
    indexed = frame.set_index("timestamp")

    resampled = indexed.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    resampled["source_bar_count"] = (
        indexed["close"].resample(rule, label="left", closed="left").count()
    )

    # Only fully formed bars are safe for strategy logic.
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled[resampled["source_bar_count"] == int(timeframe_minutes)]

    if resampled.empty:
        return pd.DataFrame(columns=required_columns)

    resampled = resampled.drop(columns=["source_bar_count"]).reset_index()
    return resampled


# =============================================================================
# DHAN BROKER CLIENT WRAPPER
# =============================================================================
class DhanBrokerClient:
    """
    Thin wrapper around the DhanHQ Python SDK.

    Responsibilities:
    - Hold one authenticated `dhanhq` client.
    - Expose only the two calls this runner needs:
        1. Fetch 1-minute intraday OHLC for the NIFTY spot index.
        2. Fetch a batch of LTPs for a (segment -> [security_id]) dict.

    Funnelling everything through this wrapper means future SDK changes
    require edits in just one place.
    """

    def __init__(self, client_code: str, access_token: str):
        # In dhanhq >= 2.1, the SDK constructor takes a `DhanContext` object
        # instead of raw (client_id, access_token) positional arguments.
        # Building the context up front lets us reuse the same authenticated
        # session for every call (intraday OHLC, ticker LTPs, etc.).
        self._dhan_context = DhanContext(client_code, access_token)
        self.dhan = dhanhq(self._dhan_context)

    def fetch_index_1m_ohlc(
        self,
        security_id: int,
        exchange_segment: str,
        instrument_type: str,
        lookback_days: int = INTRADAY_LOOKBACK_DAYS,
    ) -> pd.DataFrame:
        """Pull a rolling window of 1-minute OHLC candles for the given index."""
        today = datetime.now().date()
        from_date = (today - timedelta(days=max(1, int(lookback_days)))).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        resp = self.dhan.intraday_minute_data(
            security_id=str(security_id),
            exchange_segment=str(exchange_segment),
            instrument_type=str(instrument_type),
            from_date=from_date,
            to_date=to_date,
            interval=1,
        )
        return normalize_dhan_intraday_response(resp)

    def fetch_ltp_map(
        self,
        securities_by_segment: dict[str, list[int]],
    ) -> dict[tuple[str, int], float]:
        """
        Fetch LTPs for a batch of (segment -> [security_id,...]) entries.

        Expected request shape:
            {"IDX_I": [13], "NSE_FNO": [49081, 49082, ...]}

        Expected response shape (abridged):
            {"status": "success",
             "data": {"IDX_I": {"13": {"last_price": 22450.1}}, ...}}

        DhanHQ sometimes nests the segment map one level deeper under an
        inner `"data"` key; we unwrap that defensively.

        Returns a flat dict {(segment, security_id): price} with non-positive
        entries dropped silently.
        """
        result: dict[tuple[str, int], float] = {}
        cleaned: dict[str, list[int]] = {}
        for segment, ids in (securities_by_segment or {}).items():
            if not ids:
                continue
            filtered = [int(sid) for sid in ids if int(sid) > 0]
            if filtered:
                cleaned[str(segment)] = sorted(set(filtered))
        if not cleaned:
            return result

        resp = self.dhan.ticker_data(cleaned)
        if not isinstance(resp, dict):
            return result

        status = str(resp.get("status", "")).strip().lower()
        if status and status != "success":
            return result

        payload = resp.get("data")
        if (
            isinstance(payload, dict)
            and "data" in payload
            and isinstance(payload["data"], dict)
            and all(isinstance(v, dict) for v in payload["data"].values())
        ):
            payload = payload["data"]
        if not isinstance(payload, dict):
            return result

        for segment, sec_map in payload.items():
            if not isinstance(sec_map, dict):
                continue
            for sec_id_str, entry in sec_map.items():
                try:
                    sec_id = int(sec_id_str)
                except (TypeError, ValueError):
                    continue
                price = 0.0
                if isinstance(entry, dict):
                    for key in ("last_price", "ltp", "LTP", "close", "price"):
                        if key in entry:
                            price = _safe_float(entry[key], 0.0)
                            if price > 0:
                                break
                elif isinstance(entry, (int, float)):
                    price = float(entry)
                elif isinstance(entry, str):
                    price = _safe_float(entry, 0.0)
                if price > 0:
                    result[(str(segment), int(sec_id))] = float(price)
        return result

    def fetch_option_chain(
        self,
        under_security_id: int,
        under_exchange_segment: str,
        expiry: date,
    ) -> dict:
        """
        Fetch the live option chain (with Greeks) for one underlying expiry.

        DhanHQ's `/optionchain` endpoint returns LTP, IV, OI, and per-leg
        Greeks (delta, gamma, theta, vega) for every listed strike. The
        Delta-0.2 Hedged Spread strategy uses this ONCE at 09:20 to pick
        its monitored CE/PE strikes; thereafter it relies on the cheaper
        `ticker_data` LTP batch (much higher rate-limit) for ongoing
        premium tracking. So even though this endpoint is "expensive"
        on the rate-limit budget, we only call it about once per session.

        Args:
            under_security_id: The DhanHQ security_id of the underlying.
                For NIFTY index this is 13 (see NIFTY_INDEX_SECURITY_ID).
            under_exchange_segment: Segment string for the underlying.
                For NIFTY index this is "IDX_I".
            expiry: A `date` instance for the target expiry. We format it
                as "YYYY-MM-DD" before sending so the wire format matches
                DhanHQ's documented schema.

        Returns:
            The raw response dict. Typical shape (abridged):
                {
                  "status": "success",
                  "data": {
                    "last_price": 22411.85,
                    "oc": {
                      "22000.000000": {
                        "ce": {
                          "last_price": 481.7,
                          "implied_volatility": 12.5,
                          "greeks": {
                            "delta": 0.74,
                            "gamma": 0.001,
                            "theta": -8.5,
                            "vega":  6.8
                          }
                        },
                        "pe": {...}
                      },
                      ...
                    }
                  }
                }

        Note: this endpoint has its own (tighter) rate-limit budget
        separate from the LTP batch endpoint. Callers should retry with
        backoff on failure rather than polling tightly. The Delta-0.2
        worker uses DELTA20_CAPTURE_RETRY_SECONDS (default 5s) for that.
        """
        # The dhanhq SDK accepts the expiry as a string. We normalize it
        # here so callers don't have to think about it.
        expiry_str = (
            expiry.strftime("%Y-%m-%d")
            if isinstance(expiry, date)
            else str(expiry)
        )
        return self.dhan.option_chain(
            under_security_id=int(under_security_id),
            under_exchange_segment=str(under_exchange_segment),
            expiry=expiry_str,
        )


# =============================================================================
# OPTIONS CONTRACT RESOLVER (unified - serves both ATM and Hedged families)
# =============================================================================
class OptionsContractResolver:
    """
    One resolver that knows how to look up NIFTY option contracts under
    BOTH expiry rules and BOTH strike rules used in this file.

    EXPIRY RULES (the if-else the user asked for, exposed as two methods):
    - `get_target_expiry()`        -> "next-next" expiry (used by ATM family).
    - `get_current_week_expiry()`  -> "current-week" expiry, i.e. the FIRST
                                       expiry on or after today (used by the
                                       Hedged Puts family).

    STRIKE RULES:
    - `get_atm_option(spot, dir)`  -> ATM strike rounded to nearest 50, paired
                                       with CE for LONG / PE for SHORT. Used
                                       by the four ATM workers.
    - `list_puts_for_expiry(exp)` + `pick_put_by_target_premium(...)`
                                    -> primitives the BULLISH hedged worker
                                       uses to pick a PE leg whose live LTP
                                       is closest to a target premium.
    - `list_calls_for_expiry(exp)` + `pick_call_by_target_premium(...)`
                                    -> mirror primitives for the BEARISH
                                       hedged worker (CE legs).

    Caching:
    - The full option chain is cached per trading day so we do not re-read
      the instrument master CSV on every entry decision.
    """

    def __init__(self, underlying: str, instrument_master_glob: str, log: logging.Logger):
        self.underlying = str(underlying).upper().strip()
        self.instrument_master_glob = instrument_master_glob
        self.log = log
        self._option_chain_cache: Optional[pd.DataFrame] = None
        self._cache_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_latest_instrument_master_path(self) -> str:
        """Pick the newest file matching the instrument-master glob."""
        matches = glob.glob(self.instrument_master_glob)
        if not matches:
            return ""
        matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        return matches[0]

    def _load_option_chain(self) -> pd.DataFrame:
        """
        Load, normalize, and cache the NIFTY options slice of the master CSV.

        Resulting columns:
            security_id, trading_symbol, custom_symbol, strike, option_type,
            expiry, lot_size

        The result keeps both PE and CE rows for every weekly expiry on or
        after today. The two strike-picking flows on this resolver each
        slice the cached DataFrame as needed.
        """
        today = datetime.now().date()
        if self._option_chain_cache is not None and self._cache_date == today:
            return self._option_chain_cache

        instrument_path = self._get_latest_instrument_master_path()
        if not instrument_path:
            raise ValueError(
                f"Instrument master not found for pattern '{self.instrument_master_glob}'. "
                "Cannot resolve option contracts."
            )

        df = pd.read_csv(instrument_path, dtype=str, low_memory=False)
        if df.empty:
            raise ValueError(f"Instrument master file is empty: {instrument_path}")

        exch_col = _first_existing_col(df, ["SEM_EXM_EXCH_ID"])
        seg_col = _first_existing_col(df, ["SEM_SEGMENT"])
        ins_col = _first_existing_col(df, ["SEM_INSTRUMENT_NAME"])
        ts_col = _first_existing_col(df, ["SEM_TRADING_SYMBOL"])
        cs_col = _first_existing_col(df, ["SEM_CUSTOM_SYMBOL"])
        exp_col = _first_existing_col(df, ["SEM_EXPIRY_DATE"])
        lot_col = _first_existing_col(df, ["SEM_LOT_UNITS"])
        sec_col = _first_existing_col(df, ["SEM_SMST_SECURITY_ID"])
        strike_col = _first_existing_col(df, ["SEM_STRIKE_PRICE"])
        opt_type_col = _first_existing_col(df, ["SEM_OPTION_TYPE"])
        sm_col = _first_existing_col(df, ["SM_SYMBOL_NAME"])

        required = [exch_col, seg_col, ins_col, ts_col, exp_col, sec_col, strike_col, opt_type_col]
        if any(column is None for column in required):
            raise ValueError(
                f"Instrument file missing required columns for option resolution: {instrument_path}"
            )

        work = pd.DataFrame(
            {
                "exchange": df[exch_col].fillna("").astype(str).str.upper().str.strip(),
                "segment": df[seg_col].fillna("").astype(str).str.upper().str.strip(),
                "instrument": df[ins_col].fillna("").astype(str).str.upper().str.strip(),
                "trading_symbol": df[ts_col].fillna("").astype(str).str.strip(),
                "custom_symbol": (df[cs_col].fillna("").astype(str).str.strip() if cs_col else ""),
                "security_id_raw": df[sec_col].fillna("").astype(str).str.strip(),
                "strike_raw": df[strike_col].fillna("").astype(str).str.strip(),
                "option_type": df[opt_type_col].fillna("").astype(str).str.upper().str.strip(),
                "expiry_raw": df[exp_col].fillna("").astype(str).str.strip(),
                "lot_raw": (df[lot_col].fillna("").astype(str).str.strip() if lot_col else ""),
                "sm_symbol": (df[sm_col].fillna("").astype(str).str.upper().str.strip() if sm_col else ""),
            }
        )
        work["expiry"] = pd.to_datetime(work["expiry_raw"], errors="coerce").dt.date
        work["strike"] = pd.to_numeric(work["strike_raw"], errors="coerce")
        work["security_id"] = pd.to_numeric(work["security_id_raw"], errors="coerce")
        work["lot_size"] = pd.to_numeric(work["lot_raw"], errors="coerce")

        # Keep only NSE index-option rows for our underlying that expire today
        # or later. The extra string filters reject BANKNIFTY/FINNIFTY/etc.
        # rows that share prefix patterns in some master CSV layouts.
        underlying_prefix = f"{self.underlying}-"
        opt_mask = (
            (work["exchange"] == "NSE")
            & (work["segment"] == "D")
            & (work["instrument"] == "OPTIDX")
            & work["option_type"].isin(["CE", "PE"])
            & (
                (work["sm_symbol"] == self.underlying)
                | work["trading_symbol"].str.upper().str.startswith(underlying_prefix)
            )
            & work["expiry"].notna()
            & work["strike"].notna()
            & work["security_id"].notna()
            & (work["expiry"] >= today)
        )
        opt_mask &= work["trading_symbol"].str.upper().str.startswith(underlying_prefix)
        opt_mask &= ~work["trading_symbol"].str.upper().str.startswith("BANKNIFTY-")
        opt_mask &= ~work["trading_symbol"].str.upper().str.startswith("FINNIFTY-")
        opt_mask &= ~work["trading_symbol"].str.upper().str.startswith("MIDCPNIFTY-")
        opt_mask &= ~work["trading_symbol"].str.upper().str.startswith("NIFTYNXT50-")

        options = work.loc[
            opt_mask,
            [
                "security_id",
                "trading_symbol",
                "custom_symbol",
                "strike",
                "option_type",
                "expiry",
                "lot_size",
            ],
        ].copy()

        if options.empty:
            raise ValueError(f"No valid {self.underlying} option rows found in {instrument_path}")

        options["security_id"] = options["security_id"].astype(int)
        options = options.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)

        self._option_chain_cache = options
        self._cache_date = today
        self.log.info(
            "Loaded %s option chain | File=%s | Rows=%s | Expiries=%s",
            self.underlying,
            os.path.basename(instrument_path),
            len(options),
            sorted(options["expiry"].unique().tolist())[:5],
        )
        return options

    # ------------------------------------------------------------------
    # Expiry rules (the explicit if-else)
    # ------------------------------------------------------------------
    def get_target_expiry(self) -> date:
        """
        Return the "next-next" expiry: the SECOND expiry on or after today.

        Used by the four ATM workers (Renko, EMA, Heikin, Profit Shooter).

        Steps:
        1. Load the option chain (cached per day).
        2. Collect unique expiries >= today.
        3. Sort ascending.
        4. Return the second entry.
        """
        options = self._load_option_chain()
        today = datetime.now().date()
        expiries = sorted({exp for exp in options["expiry"].tolist() if exp is not None and exp >= today})
        if len(expiries) < 2:
            raise ValueError(
                f"Need at least two future {self.underlying} expiries to pick the next-next expiry, "
                f"found {len(expiries)}."
            )
        return expiries[1]

    def get_current_week_expiry(self) -> date:
        """
        Return the "current-week" expiry: the FIRST expiry on or after today.

        Used by the two Hedged Puts workers (Supertrend Bullish + Donchian
        Bearish). For NIFTY weekly options this is typically the upcoming
        Thursday. If today is itself the Thursday expiry, this returns
        today's expiry.
        """
        options = self._load_option_chain()
        today = datetime.now().date()
        expiries = sorted({exp for exp in options["expiry"].tolist() if exp is not None and exp >= today})
        if not expiries:
            raise ValueError(f"No future {self.underlying} expiry found in instrument master.")
        return expiries[0]

    # ------------------------------------------------------------------
    # ATM strike rule (used by the 4 ATM workers)
    # ------------------------------------------------------------------
    def get_atm_option(self, spot_price: float, direction: str) -> dict:
        """
        Resolve the ATM option row for the given direction and the
        next-next expiry.

        Direction mapping:
        - "LONG"  -> CE
        - "SHORT" -> PE

        ATM rule:
        - Round spot to the nearest 50.
        - If that exact strike is not listed, pick the closest available
          strike (smallest absolute difference, with the lower strike
          breaking ties).
        """
        direction_upper = str(direction).strip().upper()
        if direction_upper == "LONG":
            right = "CE"
        elif direction_upper == "SHORT":
            right = "PE"
        else:
            raise ValueError(f"Unsupported direction for option resolution: {direction!r}")

        spot = _safe_float(spot_price, 0.0)
        if spot <= 0:
            raise ValueError(f"Invalid spot price for ATM resolution: {spot_price!r}")

        options = self._load_option_chain()
        target_expiry = self.get_target_expiry()

        atm_strike = round(spot / ATM_STRIKE_STEP) * ATM_STRIKE_STEP

        subset = options[(options["expiry"] == target_expiry) & (options["option_type"] == right)].copy()
        if subset.empty:
            raise ValueError(
                f"No {right} rows found for {self.underlying} expiry {target_expiry}."
            )

        subset["strike_diff"] = (subset["strike"] - atm_strike).abs()
        subset = subset.sort_values(["strike_diff", "strike"])
        row = subset.iloc[0]

        today = datetime.now().date()
        expiry_date = row["expiry"]
        days_to_expiry = (expiry_date - today).days if expiry_date is not None else None

        return {
            "security_id": int(row["security_id"]),
            "exchange_segment": OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": str(row["trading_symbol"]).strip(),
            "custom_symbol": str(row["custom_symbol"]).strip(),
            "strike": float(row["strike"]),
            "option_type": right,
            "expiry_date": expiry_date,
            "days_to_expiry": days_to_expiry,
            "lot_size": _to_int_safe(row["lot_size"], 0),
            "spot_reference": float(spot),
            "atm_strike_rounded": float(atm_strike),
        }

    # ------------------------------------------------------------------
    # PE primitives (used by the BULLISH hedged worker)
    # ------------------------------------------------------------------
    def list_puts_for_expiry(self, expiry: date) -> pd.DataFrame:
        """
        Return every PE row for `expiry`, sorted by strike ascending.

        The BULLISH hedged worker batch-fetches LTPs for these security_ids
        and feeds the result to `pick_put_by_target_premium`.
        """
        options = self._load_option_chain()
        subset = options[
            (options["expiry"] == expiry) & (options["option_type"] == "PE")
        ].copy()
        subset = subset.sort_values("strike").reset_index(drop=True)
        return subset

    def pick_put_by_target_premium(
        self,
        puts: pd.DataFrame,
        ltp_map: dict[tuple[str, int], float],
        target_premium: float,
        exclude_security_ids: Optional[set[int]] = None,
    ) -> dict:
        """
        From a PE-rows DataFrame and a (segment, sec_id) -> LTP map, return
        the row whose LTP is closest to `target_premium`.

        - `exclude_security_ids` lets callers skip strikes already chosen
          for another leg (e.g. so the hedge leg never accidentally lands
          on the same strike as the main leg in a tie).
        - Ties prefer the lower strike because the input is sorted ascending.
        """
        if puts is None or puts.empty:
            raise ValueError("No candidate PE rows supplied.")
        if target_premium <= 0:
            raise ValueError(f"Invalid target premium: {target_premium!r}")
        exclude = exclude_security_ids or set()

        best_row = None
        best_ltp = 0.0
        best_diff = float("inf")

        for _, row in puts.iterrows():
            sec_id = int(row["security_id"])
            if sec_id in exclude:
                continue
            key = (OPTION_EXCHANGE_SEGMENT, sec_id)
            ltp = _safe_float(ltp_map.get(key, 0.0), 0.0)
            if ltp <= 0:
                continue
            diff = abs(ltp - float(target_premium))
            if diff < best_diff:
                best_diff = diff
                best_row = row
                best_ltp = ltp

        if best_row is None:
            raise ValueError(
                f"No PE with a valid live LTP found near target premium {target_premium:.2f}."
            )

        return {
            "security_id": int(best_row["security_id"]),
            "exchange_segment": OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": str(best_row["trading_symbol"]).strip(),
            "custom_symbol": str(best_row["custom_symbol"]).strip(),
            "strike": float(best_row["strike"]),
            "option_type": "PE",
            "expiry_date": best_row["expiry"],
            "lot_size": _to_int_safe(best_row["lot_size"], 0),
            "entry_ltp": float(best_ltp),
        }

    # ------------------------------------------------------------------
    # CE primitives (used by the BEARISH hedged worker)
    # ------------------------------------------------------------------
    # These mirror the PE methods exactly. The bodies are duplicated rather
    # than refactored into a shared helper so that an edit to PE behaviour
    # cannot accidentally change CE behaviour, and vice versa.
    def list_calls_for_expiry(self, expiry: date) -> pd.DataFrame:
        """Return every CE row for `expiry`, sorted by strike ascending."""
        options = self._load_option_chain()
        subset = options[
            (options["expiry"] == expiry) & (options["option_type"] == "CE")
        ].copy()
        subset = subset.sort_values("strike").reset_index(drop=True)
        return subset

    def pick_call_by_target_premium(
        self,
        calls: pd.DataFrame,
        ltp_map: dict[tuple[str, int], float],
        target_premium: float,
        exclude_security_ids: Optional[set[int]] = None,
    ) -> dict:
        """
        Mirror of `pick_put_by_target_premium` for CE rows.
        See that method for the full contract.
        """
        if calls is None or calls.empty:
            raise ValueError("No candidate CE rows supplied.")
        if target_premium <= 0:
            raise ValueError(f"Invalid target premium: {target_premium!r}")
        exclude = exclude_security_ids or set()

        best_row = None
        best_ltp = 0.0
        best_diff = float("inf")

        for _, row in calls.iterrows():
            sec_id = int(row["security_id"])
            if sec_id in exclude:
                continue
            key = (OPTION_EXCHANGE_SEGMENT, sec_id)
            ltp = _safe_float(ltp_map.get(key, 0.0), 0.0)
            if ltp <= 0:
                continue
            diff = abs(ltp - float(target_premium))
            if diff < best_diff:
                best_diff = diff
                best_row = row
                best_ltp = ltp

        if best_row is None:
            raise ValueError(
                f"No CE with a valid live LTP found near target premium {target_premium:.2f}."
            )

        return {
            "security_id": int(best_row["security_id"]),
            "exchange_segment": OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": str(best_row["trading_symbol"]).strip(),
            "custom_symbol": str(best_row["custom_symbol"]).strip(),
            "strike": float(best_row["strike"]),
            "option_type": "CE",
            "expiry_date": best_row["expiry"],
            "lot_size": _to_int_safe(best_row["lot_size"], 0),
            "entry_ltp": float(best_ltp),
        }

    # ------------------------------------------------------------------
    # Exact-strike lookup (used by the Delta-0.2 hedged spread worker)
    # ------------------------------------------------------------------
    def get_option_for_strike(
        self,
        expiry: date,
        strike: float,
        right: str,
    ) -> Optional[dict]:
        """
        Return the option-chain row for an exact (expiry, strike, right) tuple.

        Why this helper exists:
        - DhanHQ's live option_chain endpoint reports per-strike Greeks
          but does NOT include our broker-side `security_id`.
        - To place a paper trade (or subscribe to LTP updates) we need
          the security_id from the instrument-master CSV.
        - This helper bridges the two: given an (expiry, strike, right),
          it returns the matching CSV row's identifiers.

        Used by `Delta20HedgedSpreadWorker` to:
          1. Map a strike chosen via the live option_chain Greeks back
             into our instrument-master row (which has security_id and
             lot_size).
          2. Resolve the hedge leg (4 strikes further OTM) once the
             monitor leg's strike is known.

        Args:
            expiry: A `date` matching one of the listed weekly expiries.
            strike: The requested strike level (e.g. 22500.0). We tolerate
                tiny float drift (e.g. 22500.000001 vs 22500.0) by
                rejecting matches whose absolute strike-diff exceeds 0.5.
            right: "CE" or "PE" (case-insensitive).

        Returns:
            A dict with the same shape that `get_atm_option` produces,
            minus the spot/diff fields. Returns `None` if the strike is
            not listed for the given expiry/right.
        """
        right_upper = str(right).strip().upper()
        if right_upper not in ("CE", "PE"):
            raise ValueError(f"Unsupported option right: {right!r}")

        # Pull the cached option chain (already filtered to NIFTY rows
        # for expiries today-or-later) and slice down to the right side.
        options = self._load_option_chain()
        subset = options[
            (options["expiry"] == expiry) & (options["option_type"] == right_upper)
        ].copy()
        if subset.empty:
            return None

        # Find the row whose strike is closest to the requested value.
        # Sort by strike_diff first (best match), strike second (stable
        # tie-breaking when two equally-distant strikes exist - very rare).
        subset["strike_diff"] = (subset["strike"] - float(strike)).abs()
        subset = subset.sort_values(["strike_diff", "strike"])
        best = subset.iloc[0]
        if float(best["strike_diff"]) > 0.5:
            # No listed strike close enough to the requested level. This
            # typically means the request is out of the listed range
            # (e.g. asking for a hedge 200 points beyond the deepest OTM
            # listed strike). Caller treats `None` as "skip this side".
            return None

        return {
            "security_id": int(best["security_id"]),
            "exchange_segment": OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": str(best["trading_symbol"]).strip(),
            "custom_symbol": str(best["custom_symbol"]).strip(),
            "strike": float(best["strike"]),
            "option_type": right_upper,
            "expiry_date": best["expiry"],
            "lot_size": _to_int_safe(best["lot_size"], 0),
        }


# =============================================================================
# TIME HELPERS
# =============================================================================
def is_before_time(hour: int, minute: int) -> bool:
    """Return True when local wall-clock time is before HH:MM."""
    now = datetime.now()
    threshold = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now < threshold


def is_after_time(hour: int, minute: int) -> bool:
    """Return True when local wall-clock time is at or after HH:MM."""
    now = datetime.now()
    threshold = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now >= threshold


# =============================================================================
# CENTRAL MARKET-DATA FETCHER
# =============================================================================
class CentralMarketDataFetcher(threading.Thread):
    """
    Producer thread: pulls OHLC and LTPs and publishes them into the shared
    store.

    Per poll cycle:
    1. Fetch 1-minute NIFTY spot OHLC and `store.update(...)` it.
    2. Build a batched LTP request that always includes the NIFTY spot, plus
       every option leg currently subscribed by ANY worker (PE legs from
       bullish, CE legs from bearish, ATM CE/PE from the four ATM workers).
       Fetch and `store.update_ltp_map(...)` them.
    3. Sleep `FETCH_POLL_SECONDS` using `stop_event.wait(...)` so shutdown
       stays responsive.

    Logging is intentionally quiet: candle log fires only when the latest
    1-minute candle timestamp changes, and the index-LTP log fires only when
    the cached spot changes. The full per-poll detail still goes to the
    workers via the store.
    """

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(name="MarketDataFetcher", daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.broker = broker
        self.log = logging.getLogger(f"{LOGGER_NAME}.fetcher")
        self.last_logged_candle_ts: dict[str, Optional[pd.Timestamp]] = {}
        self.last_logged_index_ltp: Optional[float] = None

    def fetch_ohlc(self, timeframe: str) -> pd.DataFrame:
        """Fetch 1-min NIFTY spot OHLC. Higher TFs are derived per-worker."""
        if str(timeframe) != "1":
            raise ValueError(f"Unsupported source timeframe for dhanhq fetcher: {timeframe}")
        return self.broker.fetch_index_1m_ohlc(
            security_id=NIFTY_INDEX_SECURITY_ID,
            exchange_segment=NIFTY_INDEX_EXCHANGE_SEGMENT,
            instrument_type=NIFTY_INDEX_INSTRUMENT_TYPE,
        )

    def refresh_index_and_option_ltps(self) -> None:
        """
        Refresh the NIFTY spot LTP plus every subscribed option LTP in a
        single batched ticker_data call.
        """
        subscriptions = self.store.snapshot_option_subscriptions()
        securities: dict[str, list[int]] = {
            NIFTY_INDEX_EXCHANGE_SEGMENT: [NIFTY_INDEX_SECURITY_ID],
        }
        for sub in subscriptions:
            segment = str(sub.exchange_segment)
            securities.setdefault(segment, [])
            if int(sub.security_id) not in securities[segment]:
                securities[segment].append(int(sub.security_id))

        try:
            ltp_map = self.broker.fetch_ltp_map(securities)
        except Exception as exc:
            self.log.exception("Error while refreshing LTP batch: %s", exc)
            return

        if ltp_map:
            self.store.update_ltp_map(ltp_map)

        index_ltp = ltp_map.get((NIFTY_INDEX_EXCHANGE_SEGMENT, NIFTY_INDEX_SECURITY_ID), 0.0)
        if index_ltp > 0 and self.last_logged_index_ltp != index_ltp:
            self.last_logged_index_ltp = index_ltp
            self.log.info(
                "Cached NIFTY index LTP | SecurityId=%s | LTP=%.2f | SubscribedOptions=%s",
                NIFTY_INDEX_SECURITY_ID,
                index_ltp,
                len(subscriptions),
            )

    def run(self) -> None:
        self.log.info("Starting central market data fetcher (dhanhq backend).")
        while not self.stop_event.is_set():
            for timeframe in REQUIRED_TIMEFRAMES:
                if self.stop_event.is_set():
                    break
                try:
                    frame = self.fetch_ohlc(timeframe)
                    snapshot = self.store.update(timeframe, frame)
                    if self.last_logged_candle_ts.get(timeframe) != snapshot.source_candle_ts:
                        self.last_logged_candle_ts[timeframe] = snapshot.source_candle_ts
                        self.log.info(
                            "Fetched timeframe=%s | Rows=%s | LastCandle=%s",
                            timeframe,
                            len(frame),
                            snapshot.source_candle_ts,
                        )
                except Exception as exc:
                    self.log.exception("Fetch error for timeframe=%s: %s", timeframe, exc)
            try:
                self.refresh_index_and_option_ltps()
            except Exception as exc:
                self.log.exception("Fetch error while refreshing LTPs: %s", exc)
            self.stop_event.wait(FETCH_POLL_SECONDS)
        self.log.info("Central market data fetcher stopped.")


# =============================================================================
# BASE PAPER STRATEGY WORKER (abstract)
# =============================================================================
class BasePaperStrategyWorker(threading.Thread):
    """
    Abstract base class for every strategy worker thread.

    What the base does for everyone:
    - Owns the main `run()` loop (risk -> time cutoff -> pre-open wait ->
      snapshot read -> signature gate -> dispatch).
    - Provides `_get_option_ltp` and `_get_underlying_spot` helpers that
      read the shared cache first and only fall back to a direct broker
      call if the cache is empty.
    - Supplies max-loss and time-cutoff shutdown wrappers that close any
      open paper trade and halt the worker for the day.
    - Defines the synthetic paper-order-id generator.

    What the base does NOT do:
    - It does not know how a specific strategy converts OHLC into signals.
    - It does not know whether the worker uses single-leg or hedged
      execution.
    Both of those are the subclass's job.

    Subclass contract:
    - `build_strategy_frame(ohlc) -> pd.DataFrame`
        Transform 1-minute OHLC into the strategy's working DataFrame
        (resample if needed, attach indicator columns).
    - `process_strategy_frame(strategy_frame) -> None`
        Inspect the latest row and decide whether to enter / exit.
    - `exit_position(reason)` and `_get_open_position_pnl()`
        Close the current paper position and report MTM PnL. Default stubs
        do nothing so the cutoff handlers never crash on flat workers.
    """

    strategy_name = "Base"
    timeframe = "1"
    poll_seconds = 2
    lots = 1
    max_loss = 0.0
    trading_start_hour = 9
    trading_start_minute = 15
    square_off_hour = 15
    square_off_minute = 15

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(name=f"{self.strategy_name}Thread", daemon=True)
        self.store = store
        self.stop_event = stop_event
        self.broker = broker
        self.log = logging.getLogger(f"{LOGGER_NAME}.{self.strategy_name.lower().replace(' ', '_')}")

        # The active paper position. Subclasses replace this default with a
        # PaperPosition (single-leg) or HedgedPaperPosition (two-leg) as
        # appropriate. The base only checks `self.pos.active` so either type
        # works without further coordination.
        self.pos = PaperPosition()

        # Latest strategy-frame fingerprint. Workers only re-run when the
        # latest strategy candle changed. Without this gate the fetcher's
        # 2-second polling would force every signal to recompute every poll.
        self.last_processed_frame_signature = None

        # Synthetic order-id counter (paper-only). Used purely for log audit.
        self.paper_order_counter = 0
        self.completed_trades = 0
        self.realized_pnl = 0.0

        # Each worker gets its own resolver instance. The CSV cache is
        # per-instance because we never share it across threads.
        self.contract_resolver = OptionsContractResolver(
            underlying=UNDERLYING,
            instrument_master_glob=INSTRUMENT_MASTER_GLOB,
            log=self.log,
        )

        # Guards so we never run the same shutdown / pre-open log twice.
        self.cutoff_handled = False
        self.preopen_wait_logged = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _color_pnl_text(pnl_value: float) -> str:
        """Color a PnL number green / red for console readability."""
        text = f"{pnl_value:.2f}"
        if pnl_value > 0:
            return f"{ANSI_GREEN}{text}{ANSI_RESET}"
        if pnl_value < 0:
            return f"{ANSI_RED}{text}{ANSI_RESET}"
        return text

    def _next_paper_order_id(self, side: str) -> str:
        """
        Build a unique synthetic order id for a paper trade leg.

        Format: `PAPER-<SIDE>-<YYYYMMDDhhmmss>-<NNNN>`. The timestamp gives
        chronological ordering when scanning logs by eye; the trailing
        counter guarantees uniqueness when two legs are placed inside the
        same wall-clock second (which happens with hedged spreads).
        """
        self.paper_order_counter += 1
        return f"PAPER-{side}-{datetime.now():%Y%m%d%H%M%S}-{self.paper_order_counter:04d}"

    def _get_option_ltp(self, segment: str, security_id: int, fallback: float) -> float:
        """
        Latest option LTP, with a 3-step preference:
        1. Centrally cached value (the fetcher refreshes it every cycle).
        2. One-shot direct broker fetch if the cache is cold.
        3. Caller-supplied fallback if both above fail.
        """
        cached = self.store.get_ltp_by_secid(segment, security_id, fallback=0.0)
        if cached > 0:
            return cached

        try:
            ltp_map = self.broker.fetch_ltp_map({segment: [security_id]})
        except Exception as exc:
            self.log.warning(
                "Direct LTP fetch failed for segment=%s security_id=%s: %s",
                segment,
                security_id,
                exc,
            )
            return float(fallback)

        price = ltp_map.get((str(segment), int(security_id)), 0.0)
        if price > 0:
            self.store.update_ltp_map({(str(segment), int(security_id)): price})
            return float(price)
        return float(fallback)

    def _get_underlying_spot(self, fallback: float) -> float:
        """Latest NIFTY spot LTP, with the same cache -> direct -> fallback chain."""
        spot = self.store.get_ltp_by_secid(
            NIFTY_INDEX_EXCHANGE_SEGMENT,
            NIFTY_INDEX_SECURITY_ID,
            fallback=0.0,
        )
        if spot > 0:
            return spot
        try:
            ltp_map = self.broker.fetch_ltp_map(
                {NIFTY_INDEX_EXCHANGE_SEGMENT: [NIFTY_INDEX_SECURITY_ID]}
            )
        except Exception as exc:
            self.log.warning("Direct NIFTY spot LTP fetch failed: %s", exc)
            return float(fallback)
        price = ltp_map.get((NIFTY_INDEX_EXCHANGE_SEGMENT, NIFTY_INDEX_SECURITY_ID), 0.0)
        if price > 0:
            self.store.update_ltp_map(
                {(NIFTY_INDEX_EXCHANGE_SEGMENT, NIFTY_INDEX_SECURITY_ID): price}
            )
            return float(price)
        return float(fallback)

    # ------------------------------------------------------------------
    # Risk / lifecycle (shared)
    # ------------------------------------------------------------------
    def _get_open_position_pnl(self) -> float:
        """
        Default: return zero. Subclasses override with their leg-specific
        MTM math (single-leg ATM or hedged-pair).
        """
        return 0.0

    def is_max_loss_breached(self):
        """
        Check session risk and report whether the daily loss limit was hit.

        Sign convention:
        - `realized_pnl` and `open_pnl` are signed (negative = losing).
        - `self.max_loss` is stored as a POSITIVE number representing the
          maximum tolerable drawdown for the day.
        - The breach condition is therefore `total_pnl <= -max_loss`
          (i.e. the loss has dipped to or below the negative limit).

        Returns a 3-tuple `(breached, total_pnl, open_pnl)` so the caller
        can log both totals at once when the breach handler fires.

        If `max_loss` is zero or negative, risk shutdown is disabled.
        """
        open_pnl = self._get_open_position_pnl()
        total_pnl = self.realized_pnl + open_pnl
        if self.max_loss <= 0:
            return False, total_pnl, open_pnl
        return total_pnl <= (-1.0 * float(self.max_loss)), total_pnl, open_pnl

    def summary_text(self) -> str:
        """
        One-line end-of-day session summary string.

        Strategies that track extra counters (signals, entries, exits)
        override this to extend the line. The base provides the minimum:
        completed trades and realized PnL.
        """
        return f"Trades={self.completed_trades} | RealizedPnL={self.realized_pnl:.2f}"

    def minimum_strategy_rows(self) -> int:
        """
        Minimum size of the strategy frame before signal logic should run.

        Defaults to `MIN_BARS` (the 1-min source minimum). Strategies that
        need more warm-up override this (e.g. Profit Shooter needs SMA200
        history; Heikin Ashi needs Bollinger warm-up).
        """
        return MIN_BARS

    def wait_for_next_poll(self) -> None:
        """
        Sleep until the next strategy tick, returning early on shutdown.

        Using `stop_event.wait(seconds)` instead of `time.sleep(seconds)`
        means the worker wakes up immediately when the supervisor sets the
        stop event during Ctrl+C, keeping `join` timeouts meaningful.
        """
        self.stop_event.wait(self.poll_seconds)

    def handle_max_loss_and_stop(self, total_pnl: float, open_pnl: float) -> None:
        """One-time shutdown sequence triggered by a risk breach."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.error(
            "MAX_LOSS breached | Limit=%.2f | SessionPnL=%.2f | OpenPnL=%.2f. "
            "Stopping strategy and closing paper positions.",
            self.max_loss,
            total_pnl,
            open_pnl,
        )
        try:
            if self.pos.active:
                self.exit_position("MAX_LOSS_BREACH")
        except Exception as exc:
            self.log.exception("Error while exiting active position on max-loss breach: %s", exc)
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day due to max-loss shutdown.")

    def handle_square_off_and_stop(self) -> None:
        """One-time shutdown sequence triggered by the daily time cutoff."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.info(
            "%02d:%02d cutoff reached. Closing any open positions and halting trading for the day.",
            self.square_off_hour,
            self.square_off_minute,
        )
        try:
            if self.pos.active:
                self.exit_position("TIME_CUTOFF")
        except Exception as exc:
            self.log.exception("Error while exiting active position at cutoff: %s", exc)
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day.")

    # ------------------------------------------------------------------
    # Subclass contract (must be implemented by every concrete worker)
    # ------------------------------------------------------------------
    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        raise NotImplementedError

    def exit_position(self, reason: str) -> None:
        """Subclasses override. Default is a safe no-op for stubs."""
        return

    def after_exit(self, closed_position, reason: str) -> None:
        """Optional hook invoked at the end of `exit_position`."""
        return

    # ------------------------------------------------------------------
    # Main loop (shared by every concrete worker, except the bearish
    # Donchian worker which overrides `run` to also poll SL/target)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Standard main loop, used by 5 of the 6 workers as-is.

        Loop flow per iteration:
        1. Risk check first  -> max-loss breach stops trading immediately.
        2. Time cutoff second -> daily square-off stops trading even if
           there is still strategy activity to process.
        3. Pre-open wait third -> no analysis before the strategy start
           time so we never enter off a half-formed morning bar.
        4. Read the latest 1-min snapshot from the shared store. Bail out
           politely if the fetcher has not published enough bars yet.
        5. Build the strategy-specific frame (resampled + indicators) and
           bail out if it is too small.
        6. Signature-gate it: if the latest row is byte-identical to the
           last one we processed, skip — there is nothing new.
        7. Dispatch to the subclass's `process_strategy_frame`.

        The bearish Donchian worker overrides this loop to also run
        `_check_sl_target_and_exit()` on every poll. See its docstring
        for the rationale.
        """
        self.log.info("Starting %s strategy worker.", self.strategy_name)
        while not self.stop_event.is_set():
            try:
                breached, total_pnl, open_pnl = self.is_max_loss_breached()
                if breached:
                    self.handle_max_loss_and_stop(total_pnl, open_pnl)
                    break

                if is_after_time(self.square_off_hour, self.square_off_minute):
                    self.handle_square_off_and_stop()
                    break

                if is_before_time(self.trading_start_hour, self.trading_start_minute):
                    if not self.preopen_wait_logged:
                        self.log.info(
                            "Before strategy start (%02d:%02d). Waiting to begin processing.",
                            self.trading_start_hour,
                            self.trading_start_minute,
                        )
                        self.preopen_wait_logged = True
                    self.wait_for_next_poll()
                    continue
                self.preopen_wait_logged = False

                snapshot = self.store.get(self.timeframe)
                if snapshot is None or snapshot.frame.empty:
                    # Fetcher has not published anything yet (typical at
                    # process start). Wait politely instead of crashing.
                    self.wait_for_next_poll()
                    continue
                if len(snapshot.frame) < MIN_BARS:
                    # Have data, but not enough warm-up history yet.
                    self.wait_for_next_poll()
                    continue

                strategy_frame = self.build_strategy_frame(snapshot.frame)
                if strategy_frame is None or strategy_frame.empty:
                    self.wait_for_next_poll()
                    continue
                if len(strategy_frame) < self.minimum_strategy_rows():
                    # Strategy-specific extra warm-up requirement not met.
                    self.wait_for_next_poll()
                    continue

                frame_signature = build_last_row_signature(strategy_frame)
                if frame_signature is None:
                    self.wait_for_next_poll()
                    continue
                if self.last_processed_frame_signature == frame_signature:
                    # Nothing changed since the last poll — skip the work.
                    self.wait_for_next_poll()
                    continue

                self.last_processed_frame_signature = frame_signature
                self.process_strategy_frame(strategy_frame)
            except Exception as exc:
                # We never want one bad iteration to kill the whole worker
                # for the day. Log the trace and continue on the next poll.
                self.log.exception("Main loop error: %s", exc)

            self.wait_for_next_poll()

        self.log.info("%s strategy worker exited.", self.strategy_name)


# =============================================================================
# ATM SINGLE-LEG STRATEGY WORKER (concrete intermediate class)
# =============================================================================
class AtmSingleLegStrategyWorker(BasePaperStrategyWorker):
    """
    Concrete intermediate base for the four ATM strategies.

    What this layer adds on top of `BasePaperStrategyWorker`:
    - `enter_position(direction, ...)` that resolves the ATM CE/PE of the
      next-next expiry and BUYS one paper leg.
    - `exit_position(reason)` that SELLS the same leg back and realizes PnL.
    - `_get_open_position_pnl()` that returns `(live - entry) * qty` for
      the open BUY leg.

    Why this lives here and not on the base:
    - The two hedged workers below need a different position shape and
      different math, so single-leg ATM behaviour is shared only among
      strategies that actually use it.
    """

    def _get_open_position_pnl(self) -> float:
        """
        MTM PnL for the open BUY leg.

        Every ATM paper trade is on the BUY side (CE for LONG, PE for SHORT)
        so PnL is always (live - entry) * qty regardless of strategy direction.
        """
        if not self.pos.active:
            return 0.0
        live_price = self._get_option_ltp(
            self.pos.option_exchange_segment,
            self.pos.option_security_id,
            fallback=self.pos.entry_trade_price,
        )
        return (live_price - self.pos.entry_trade_price) * self.pos.quantity

    def enter_position(
        self,
        direction: str,
        entry_underlying: float,
        stop_underlying: float = 0.0,
        target_underlying: float = 0.0,
        trailing_active: bool = False,
        pending_trailing_exit: bool = False,
    ) -> bool:
        """
        Open a new paper position on the ATM option of the next-next expiry.

        Steps:
        1. Read the latest NIFTY spot from the central LTP cache.
        2. Resolve the ATM CE/PE for the target expiry.
        3. Use the option's LTP as the realistic paper entry fill.
        4. Subscribe the leg with the fetcher so its LTP keeps refreshing.
        5. Persist all entry fields (including any strategy-specific flags
           like target/trailing for Profit Shooter) in `self.pos`.
        """
        spot = self._get_underlying_spot(fallback=_safe_float(entry_underlying, 0.0))
        if spot <= 0:
            self.log.warning("Skipping %s entry because NIFTY spot LTP was unavailable.", direction)
            return False

        try:
            contract = self.contract_resolver.get_atm_option(spot, direction)
        except Exception as exc:
            self.log.warning("Skipping %s entry because ATM option resolution failed: %s", direction, exc)
            return False

        option_sec_id = int(contract["security_id"])
        option_segment = str(contract["exchange_segment"])
        trading_symbol = str(contract["trading_symbol"])
        option_right = str(contract["option_type"])
        option_strike = float(contract["strike"])
        expiry_date = contract["expiry_date"]
        days_to_expiry = contract["days_to_expiry"]
        lot_size = _to_int_safe(contract["lot_size"], 0)

        if not trading_symbol or option_sec_id <= 0:
            self.log.warning("Skipping %s entry because ATM option identifiers were missing.", direction)
            return False
        if lot_size <= 0:
            self.log.warning(
                "Skipping %s entry because invalid lot size %s for %s.",
                direction,
                lot_size,
                trading_symbol,
            )
            return False

        option_ltp = self._get_option_ltp(option_segment, option_sec_id, fallback=0.0)
        if option_ltp <= 0:
            self.log.warning(
                "Skipping %s entry because option LTP was not available for %s (security_id=%s).",
                direction,
                trading_symbol,
                option_sec_id,
            )
            return False

        quantity = lot_size * self.lots
        entry_side = "BUY"  # Both LONG (CE) and SHORT (PE) open as BUY legs.
        order_id = self._next_paper_order_id(entry_side)

        # Keep the fetcher refreshing this option's LTP for the trade lifetime.
        self.store.register_option_subscription(
            OptionSubscription(
                security_id=option_sec_id,
                exchange_segment=option_segment,
                trading_symbol=trading_symbol,
                right=option_right,
                strike=option_strike,
                expiry=expiry_date,
            )
        )

        self.pos = PaperPosition(
            active=True,
            direction=direction,
            symbol=trading_symbol,
            quantity=quantity,
            entry_order_id=str(order_id),
            entry_underlying=float(entry_underlying),
            stop_underlying=float(stop_underlying),
            target_underlying=float(target_underlying),
            rr_armed=False,
            trailing_active=bool(trailing_active),
            pending_trailing_exit=bool(pending_trailing_exit),
            entry_trade_price=float(option_ltp),
            option_security_id=option_sec_id,
            option_exchange_segment=option_segment,
            option_right=option_right,
            option_strike=option_strike,
            option_expiry=expiry_date,
            option_lot_size=lot_size,
        )

        expiry_txt = expiry_date.isoformat() if expiry_date else "NA"
        dte_txt = str(days_to_expiry) if days_to_expiry is not None else "NA"
        self.log.info(
            "ENTRY %s | Side=%s | OptionSymbol=%s | Right=%s | Strike=%.2f | ExpiryDate=%s | "
            "DaysToExpiry=%s | Qty=%s | Spot=%.2f | EntryUnderlying=%.2f | StopUnderlying=%.2f | "
            "EntryOptPx=%.2f | PaperRef=%s",
            direction,
            entry_side,
            trading_symbol,
            option_right,
            option_strike,
            expiry_txt,
            dte_txt,
            quantity,
            spot,
            entry_underlying,
            stop_underlying,
            option_ltp,
            order_id,
        )
        return True

    def exit_position(self, reason: str) -> None:
        """
        Close the current paper position and realize PnL.

        Steps:
        1. Read the latest cached option LTP (fallback to direct fetch).
        2. PnL = (exit - entry) * qty (always BUY side here).
        3. Update realized PnL.
        4. Log the exit and unsubscribe the leg from the fetcher.
        5. Run any strategy-specific post-exit hook.
        6. Reset `self.pos` to a flat default.
        """
        if not self.pos.active:
            return

        closed_position = self.pos
        exit_side = "SELL"
        order_id = self._next_paper_order_id(exit_side)
        exit_trade_price = self._get_option_ltp(
            closed_position.option_exchange_segment,
            closed_position.option_security_id,
            fallback=closed_position.entry_trade_price,
        )

        pnl = (exit_trade_price - closed_position.entry_trade_price) * closed_position.quantity

        pnl_colored = self._color_pnl_text(pnl)
        self.completed_trades += 1
        self.realized_pnl += pnl

        expiry_txt = (
            closed_position.option_expiry.isoformat()
            if closed_position.option_expiry is not None
            else "NA"
        )
        self.log.info(
            "EXIT %s | OptionSymbol=%s | Right=%s | Strike=%.2f | ExpiryDate=%s | Qty=%s | "
            "Reason=%s | ExitSide=%s | EntryOptPx=%.2f | ExitOptPx=%.2f | P&L=%.2f | "
            "CumPnL=%.2f | PaperRef=%s",
            closed_position.direction,
            closed_position.symbol,
            closed_position.option_right,
            closed_position.option_strike,
            expiry_txt,
            closed_position.quantity,
            reason,
            exit_side,
            closed_position.entry_trade_price,
            exit_trade_price,
            pnl,
            self.realized_pnl,
            order_id,
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | {self.strategy_name} EXIT {closed_position.direction} | "
            f"OptionSymbol={closed_position.symbol} | Qty={closed_position.quantity} | Reason={reason} | "
            f"P&L={pnl_colored} | PaperRef={order_id}"
        )

        self.store.unregister_option_subscription(
            closed_position.option_exchange_segment,
            closed_position.option_security_id,
        )

        self.after_exit(closed_position, reason)
        self.pos = PaperPosition()


# =============================================================================
# RENKO STRATEGY WORKER (1-min source, ATM single-leg)
# =============================================================================
class RenkoStrategyWorker(AtmSingleLegStrategyWorker):
    """
    Renko strategy on the 1-minute centralized OHLC.

    Specifics:
    - Builds Renko bricks via the shared Renko logic module.
    - Uses Renko-specific state (`rr_armed`, previous-trade direction memory).
    - Refuses fresh entries inside the midday no-trade window (12:00-13:00).
    """

    strategy_name = "Renko"
    timeframe = "1"
    # All per-strategy knobs come from `.env` via the module-level
    # constants above. Setting them as class attributes (not instance)
    # keeps the existing base-class lookup pattern unchanged.
    poll_seconds = RENKO_POLL_SECONDS
    lots = RENKO_LOTS
    max_loss = RENKO_MAX_LOSS
    trading_start_hour = RENKO_TRADING_START_HOUR
    trading_start_minute = RENKO_TRADING_START_MINUTE
    square_off_hour = RENKO_SQUARE_OFF_HOUR
    square_off_minute = RENKO_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = RENKO_LOGIC.RenkoSignalEngine()

    def minimum_strategy_rows(self) -> int:
        # Need enough Renko history for stable indicator behaviour.
        return 60

    def is_in_midday_no_trade_window(self) -> bool:
        """True during the Renko midday no-new-entry window."""
        now = datetime.now()
        no_trade_start = now.replace(
            hour=RENKO_NO_TRADE_START_HOUR,
            minute=RENKO_NO_TRADE_START_MINUTE,
            second=0,
            microsecond=0,
        )
        no_trade_end = now.replace(
            hour=RENKO_NO_TRADE_END_HOUR,
            minute=RENKO_NO_TRADE_END_MINUTE,
            second=0,
            microsecond=0,
        )
        return no_trade_start <= now < no_trade_end

    def after_exit(self, closed_position: PaperPosition, reason: str) -> None:
        # Renko's re-entry rules depend on the most recent closed trade
        # direction, so we keep that memory in the signal engine.
        self.signal_engine.update_previous_trade_direction(closed_position.direction)

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        return RENKO_LOGIC.build_renko_with_indicators(ohlc)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Interpret the latest Renko brick and decide enter / exit / hold.

        Flow:
        1. If currently in a trade, build a Renko position context.
        2. Ask the engine for one decision.
        3. In trade -> only honor exits.
        4. Flat -> apply midday filter, then enter on signal.
        """
        if len(strategy_frame) < 3:
            return

        position_ctx = None
        if self.pos.active:
            position_ctx = RENKO_LOGIC.RenkoPositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
                rr_armed=self.pos.rr_armed,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            self.pos.rr_armed = bool(decision.rr_armed)
            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason)
            return

        if self.is_in_midday_no_trade_window():
            if decision.action in ("ENTER_LONG", "ENTER_SHORT"):
                self.log.info(
                    "Midday no-trade window active (12:00-13:00). Skipping fresh %s entry.",
                    "LONG" if decision.action == "ENTER_LONG" else "SHORT",
                )
            return

        if decision.action == "ENTER_LONG":
            self.enter_position("LONG", decision.entry_underlying, decision.stop_underlying)
            return
        if decision.action == "ENTER_SHORT":
            self.enter_position("SHORT", decision.entry_underlying, decision.stop_underlying)
            return


# =============================================================================
# EMA TREND STRATEGY WORKER (1-min source, locally resampled to 5-min)
# =============================================================================
class EMATrendStrategyWorker(AtmSingleLegStrategyWorker):
    """
    EMA Trend strategy. Reads 1-minute data from the shared store, locally
    resamples it into complete 5-minute candles, then runs the EMA / ATR /
    ADX indicator pipeline.
    """

    strategy_name = "EMA"
    timeframe = "1"
    poll_seconds = EMA_POLL_SECONDS
    lots = EMA_LOTS
    max_loss = EMA_MAX_LOSS
    trading_start_hour = EMA_TRADING_START_HOUR
    trading_start_minute = EMA_TRADING_START_MINUTE
    square_off_hour = EMA_SQUARE_OFF_HOUR
    square_off_minute = EMA_SQUARE_OFF_MINUTE
    derived_timeframe_minutes = EMA_DERIVED_TIMEFRAME_MINUTES

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = EMA_LOGIC.EMATrendSignalEngine(EMA_STRATEGY_CONFIG)
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0

    def summary_text(self) -> str:
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | RealizedPnL={self.realized_pnl:.2f}"
        )

    def after_exit(self, closed_position: PaperPosition, reason: str) -> None:
        self.exit_count += 1

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Resample shared 1-minute OHLC into complete 5-minute candles, then
        attach EMA / ATR / ADX columns from the EMA Trend logic module.
        """
        ohlc_5m = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        return EMA_LOGIC.build_ema_trend_with_indicators(ohlc_5m, EMA_STRATEGY_CONFIG)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Interpret the latest EMA Trend 5-minute candle.

        Flow:
        1. If currently in a trade, build a position context for the engine.
        2. Ask the engine for one decision on the latest candle.
        3. In trade -> only honor exits on this candle.
        4. Flat -> count signal triggers and submit fresh entries.
        """
        position_ctx = None
        if self.pos.active:
            position_ctx = EMA_LOGIC.EMATrendPositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason)
            return

        if decision.signal_triggered:
            self.signal_count += 1

        if decision.action == "ENTER_LONG":
            if self.enter_position("LONG", decision.entry_underlying, decision.stop_underlying):
                self.entry_submit_count += 1
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position("SHORT", decision.entry_underlying, decision.stop_underlying):
                self.entry_submit_count += 1
            return


# =============================================================================
# HEIKIN ASHI STRATEGY WORKER (1-min source, ATM single-leg)
# =============================================================================
class HeikinAshiStrategyWorker(AtmSingleLegStrategyWorker):
    """
    Heikin Ashi strategy. Builds HA candles plus Bollinger Bands and uses
    the shared Heikin Ashi signal engine (which has latch state for
    band-touch + confirmation logic).
    """

    strategy_name = "HeikinAshi"
    timeframe = "1"
    poll_seconds = HEIKIN_POLL_SECONDS
    lots = HEIKIN_LOTS
    max_loss = HEIKIN_MAX_LOSS
    trading_start_hour = HEIKIN_TRADING_START_HOUR
    trading_start_minute = HEIKIN_TRADING_START_MINUTE
    square_off_hour = HEIKIN_SQUARE_OFF_HOUR
    square_off_minute = HEIKIN_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = HEIKIN_LOGIC.HeikinAshiSignalEngine()

    def minimum_strategy_rows(self) -> int:
        return max(HEIKIN_BOLL_PERIOD + 2, 30)

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        return HEIKIN_LOGIC.build_heikin_ashi_with_bollinger(
            ohlc,
            boll_period=HEIKIN_BOLL_PERIOD,
            boll_std=HEIKIN_BOLL_STD,
        )

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Heikin Ashi can also REVERSE: exit current side and enter the
        opposite side on the same signal. After a successful entry we
        consume the engine's latch state so the same setup is not reused.
        """
        position_ctx = None
        if self.pos.active:
            position_ctx = HEIKIN_LOGIC.HeikinAshiPositionContext(direction=self.pos.direction)

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if decision.action == "ENTER_LONG":
            if self.enter_position("LONG", decision.entry_underlying):
                self.signal_engine.consume_long_setup()
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position("SHORT", decision.entry_underlying):
                self.signal_engine.consume_short_setup()
            return
        if decision.action == "REVERSE_TO_LONG":
            self.exit_position(decision.exit_reason or "REVERSAL_TO_LONG")
            if self.enter_position("LONG", decision.entry_underlying):
                self.signal_engine.consume_long_setup()
            return
        if decision.action == "REVERSE_TO_SHORT":
            self.exit_position(decision.exit_reason or "REVERSAL_TO_SHORT")
            if self.enter_position("SHORT", decision.entry_underlying):
                self.signal_engine.consume_short_setup()
            return


# =============================================================================
# PROFIT SHOOTER STRATEGY WORKER (1-min source, ATM single-leg)
# =============================================================================
class ProfitShooterStrategyWorker(AtmSingleLegStrategyWorker):
    """
    Profit Shooter pin-bar strategy.

    What is unusual here:
    - The strategy is multi-stage. Once 1.5R is hit, the trade switches to
      EMA9 trailing mode. After that, a completed close beyond EMA9 only
      ARMS the exit; the actual exit happens at the next candle open.
    - That means we must persist trailing flags between loop cycles, and
      also remember the original target.
    """

    strategy_name = "ProfitShooter"
    timeframe = "1"
    poll_seconds = PROFIT_SHOOTER_POLL_SECONDS
    lots = PROFIT_SHOOTER_LOTS
    max_loss = PROFIT_SHOOTER_MAX_LOSS
    trading_start_hour = PROFIT_SHOOTER_TRADING_START_HOUR
    trading_start_minute = PROFIT_SHOOTER_TRADING_START_MINUTE
    square_off_hour = PROFIT_SHOOTER_SQUARE_OFF_HOUR
    square_off_minute = PROFIT_SHOOTER_SQUARE_OFF_MINUTE
    # Profit Shooter refuses to open a new trade from a setup printed at
    # or after this time, because the daily square-off would close it
    # almost immediately. Built from env-driven hour/minute constants.
    last_setup_cutoff = dt_time(
        PROFIT_SHOOTER_LAST_SETUP_HOUR,
        PROFIT_SHOOTER_LAST_SETUP_MINUTE,
    )

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = PROFIT_SHOOTER_LOGIC.ProfitShooterSignalEngine(PROFIT_SHOOTER_STRATEGY_CONFIG)
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0

    def summary_text(self) -> str:
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | RealizedPnL={self.realized_pnl:.2f}"
        )

    def after_exit(self, closed_position: PaperPosition, reason: str) -> None:
        self.exit_count += 1

    def minimum_strategy_rows(self) -> int:
        return PROFIT_SHOOTER_MIN_BARS

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        return PROFIT_SHOOTER_LOGIC.build_profit_shooter_with_indicators(ohlc, PROFIT_SHOOTER_STRATEGY_CONFIG)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        latest_candle_time = pd.to_datetime(strategy_frame.iloc[-1]["timestamp"]).time()

        position_ctx = None
        if self.pos.active:
            position_ctx = PROFIT_SHOOTER_LOGIC.ProfitShooterPositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
                target_underlying=self.pos.target_underlying,
                trailing_active=self.pos.trailing_active,
                pending_trailing_exit=self.pos.pending_trailing_exit,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            # Even a HOLD decision can change trailing-mode state, so we
            # always persist the returned flags before checking exits.
            if float(decision.target_underlying) > 0:
                self.pos.target_underlying = float(decision.target_underlying)
            self.pos.trailing_active = bool(decision.trailing_active)
            self.pos.pending_trailing_exit = bool(decision.pending_trailing_exit)

            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason or "SIGNAL_EXIT")
            return

        if decision.signal_triggered:
            self.signal_count += 1

        if latest_candle_time > self.last_setup_cutoff:
            if decision.action in ("ENTER_LONG", "ENTER_SHORT"):
                self.log.info(
                    "Last setup cutoff passed (%s). Skipping fresh %s entry on candle %s.",
                    self.last_setup_cutoff.strftime("%H:%M"),
                    "LONG" if decision.action == "ENTER_LONG" else "SHORT",
                    latest_candle_time.strftime("%H:%M"),
                )
            return

        if decision.action == "ENTER_LONG":
            if self.enter_position(
                "LONG",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
                trailing_active=decision.trailing_active,
                pending_trailing_exit=decision.pending_trailing_exit,
            ):
                self.entry_submit_count += 1
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position(
                "SHORT",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
                trailing_active=decision.trailing_active,
                pending_trailing_exit=decision.pending_trailing_exit,
            ):
                self.entry_submit_count += 1
            return


# =============================================================================
# SUPERTREND BULLISH WORKER (3-min, hedged PE spread)
# =============================================================================
class SupertrendBullishWorker(BasePaperStrategyWorker):
    """
    Hedged-PE bullish strategy.

    SPEC RECAP
    ----------
    - ENTRY: 3-minute Supertrend flips RED -> GREEN on a closed candle.
        * MAIN  (SELL) : current-week PE with live LTP closest to Rs.160.
        * HEDGE (BUY)  : current-week PE with live LTP closest to Rs.10.
    - EXIT : 3-minute Supertrend flips GREEN -> RED on a closed candle.
              Both legs close simultaneously.
    - COOL-OFF: 5 minutes after every exit, no entry is even evaluated.
    - WINDOW : entries 09:18 -> 15:14, force close at 15:14.

    EXPIRY RULE: this is the Hedged Puts family, so we use
    `OptionsContractResolver.get_current_week_expiry()`, NOT the next-next
    expiry rule used by the four ATM workers.

    STRIKE RULE: the strike-picking pipeline lives in this worker
    (`_pick_hedged_puts`), per the user's design hint. The resolver only
    supplies the primitives.
    """

    strategy_name = "SupertrendBullish"
    timeframe = "1"
    poll_seconds = SUPERTREND_POLL_SECONDS
    lots = BULLISH_LOTS
    max_loss = BULLISH_MAX_LOSS
    trading_start_hour = SUPERTREND_TRADING_START_HOUR
    trading_start_minute = SUPERTREND_TRADING_START_MINUTE
    square_off_hour = SUPERTREND_SQUARE_OFF_HOUR
    square_off_minute = SUPERTREND_SQUARE_OFF_MINUTE

    derived_timeframe_minutes = SUPERTREND_TIMEFRAME_MINUTES
    # ATR + a small buffer so the flip detection has at least a couple of
    # bars of warm-up beyond the indicator's own minimum.
    min_strategy_rows = SUPERTREND_ATR_LENGTH + 5

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        # Two-leg position shape replaces the default single-leg PaperPosition.
        self.pos = HedgedPaperPosition()
        # Wall-clock time of the most recent paper-trade exit. The cool-off
        # gate compares (now - last_exit) against POST_EXIT_COOLDOWN_MINUTES.
        self.last_exit_datetime: Optional[datetime] = None
        self.post_exit_cooldown = timedelta(minutes=POST_EXIT_COOLDOWN_MINUTES)
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        # Prevents the cool-off "skip" log line from spamming every poll.
        self._cooldown_log_bar_signature = None

    def minimum_strategy_rows(self) -> int:
        return int(self.min_strategy_rows)

    def summary_text(self) -> str:
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | "
            f"RealizedPnL={self.realized_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Strategy-frame plumbing
    # ------------------------------------------------------------------
    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        ohlc_3m = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        if ohlc_3m is None or ohlc_3m.empty:
            return pd.DataFrame()
        if len(ohlc_3m) < self.minimum_strategy_rows():
            # Return the resampled frame unchanged so the base-class signature
            # gate still ticks forward as new 3-minute bars complete.
            return ohlc_3m
        return SUPERTREND_LOGIC.generate_supertrend_signals(ohlc_3m, SUPERTREND_SETTINGS)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Decision tree:
        1. In a position -> exit only on GREEN -> RED flip.
        2. Flat and inside cool-off -> ignore signals (log once per candle).
        3. Flat and outside cool-off -> enter on RED -> GREEN flip.
        """
        if strategy_frame is None or strategy_frame.empty:
            return
        if (
            "startLongTrade" not in strategy_frame.columns
            or "endLongTrade" not in strategy_frame.columns
        ):
            return

        last_row = strategy_frame.iloc[-1]
        start_long = bool(last_row.get("startLongTrade", False))
        end_long = bool(last_row.get("endLongTrade", False))
        close_price = _safe_float(last_row.get("close", 0.0), 0.0)
        latest_candle_ts = pd.to_datetime(last_row.get("timestamp"), errors="coerce")

        if self.pos.active:
            if end_long:
                self.exit_position("SUPERTREND_FLIP_TO_RED")
            return

        if self._is_in_post_exit_cooldown():
            if start_long:
                candle_key = (latest_candle_ts, start_long)
                if self._cooldown_log_bar_signature != candle_key:
                    self._cooldown_log_bar_signature = candle_key
                    elapsed = datetime.now() - (self.last_exit_datetime or datetime.now())
                    self.log.info(
                        "Post-exit cool-off active (elapsed=%.1fs). Skipping bullish entry at candle %s.",
                        elapsed.total_seconds(),
                        latest_candle_ts,
                    )
            return

        if start_long:
            self.signal_count += 1
            if close_price <= 0:
                self.log.warning(
                    "Bullish flip detected but candle close looks invalid (%.4f). Skipping.",
                    close_price,
                )
                return
            self._enter_hedged_bullish_position(close_price, latest_candle_ts)

    def _is_in_post_exit_cooldown(self) -> bool:
        if self.last_exit_datetime is None:
            return False
        return (datetime.now() - self.last_exit_datetime) < self.post_exit_cooldown

    # ------------------------------------------------------------------
    # Hedged entry (PE legs - strike selection lives HERE per the user's hint)
    # ------------------------------------------------------------------
    def _pick_hedged_puts(self, spot_price: float) -> Optional[tuple[dict, dict, date]]:
        """
        Pick the main (SELL ~Rs.160) and hedge (BUY ~Rs.10) PE legs for the
        CURRENT-WEEK expiry.

        Per-step:
        1. Resolve the current-week expiry (Hedged Puts rule).
        2. List every PE row for that expiry.
        3. Batch-fetch LTPs for all candidate security_ids.
        4. Cache the LTPs in the shared store so any later read benefits.
        5. Pick main by closest LTP to ~160, then hedge by closest LTP to ~10
           (excluding the main leg's security_id so a tie cannot pick the
           same strike twice).
        """
        try:
            expiry = self.contract_resolver.get_current_week_expiry()
        except Exception as exc:
            self.log.warning("Cannot resolve current-week expiry: %s", exc)
            return None

        try:
            puts = self.contract_resolver.list_puts_for_expiry(expiry)
        except Exception as exc:
            self.log.warning("Cannot list PE rows for expiry %s: %s", expiry, exc)
            return None
        if puts.empty:
            self.log.warning("No PE options available for expiry %s.", expiry)
            return None

        sec_ids = [int(sid) for sid in puts["security_id"].tolist() if int(sid) > 0]
        if not sec_ids:
            self.log.warning("PE list for expiry %s has no valid security_ids.", expiry)
            return None

        try:
            ltp_map = self.broker.fetch_ltp_map({OPTION_EXCHANGE_SEGMENT: sec_ids})
        except Exception as exc:
            self.log.warning("PE LTP batch fetch failed for expiry %s: %s", expiry, exc)
            return None

        if not ltp_map:
            self.log.warning(
                "PE LTP batch fetch returned no prices for expiry %s (spot=%.2f).",
                expiry,
                spot_price,
            )
            return None

        self.store.update_ltp_map(ltp_map)

        try:
            main = self.contract_resolver.pick_put_by_target_premium(
                puts, ltp_map, MAIN_PUT_TARGET_PREMIUM
            )
        except Exception as exc:
            self.log.warning(
                "Cannot pick main put near %.2f for expiry %s: %s",
                MAIN_PUT_TARGET_PREMIUM,
                expiry,
                exc,
            )
            return None

        try:
            hedge = self.contract_resolver.pick_put_by_target_premium(
                puts,
                ltp_map,
                HEDGE_PUT_TARGET_PREMIUM,
                exclude_security_ids={int(main["security_id"])},
            )
        except Exception as exc:
            self.log.warning(
                "Cannot pick hedge put near %.2f for expiry %s: %s",
                HEDGE_PUT_TARGET_PREMIUM,
                expiry,
                exc,
            )
            return None

        return main, hedge, expiry

    def _enter_hedged_bullish_position(self, entry_underlying: float, latest_candle_ts) -> bool:
        """Open a new hedged bullish paper position (SELL Rs.160 PE + BUY Rs.10 PE)."""
        spot = self._get_underlying_spot(fallback=entry_underlying)
        if spot <= 0:
            self.log.warning("Skipping bullish entry because NIFTY spot LTP was unavailable.")
            return False

        picked = self._pick_hedged_puts(spot)
        if picked is None:
            return False
        main, hedge, expiry = picked

        main_lot_size = _to_int_safe(main.get("lot_size"), 0)
        hedge_lot_size = _to_int_safe(hedge.get("lot_size"), 0)
        if main_lot_size <= 0 or hedge_lot_size <= 0:
            self.log.warning(
                "Skipping bullish entry: invalid lot size (main=%s hedge=%s).",
                main_lot_size,
                hedge_lot_size,
            )
            return False

        main_qty = main_lot_size * self.lots
        hedge_qty = hedge_lot_size * self.lots
        main_entry_price = _safe_float(main.get("entry_ltp"), 0.0)
        hedge_entry_price = _safe_float(hedge.get("entry_ltp"), 0.0)
        if main_entry_price <= 0 or hedge_entry_price <= 0:
            self.log.warning(
                "Skipping bullish entry: invalid entry premium(s) main=%.2f hedge=%.2f.",
                main_entry_price,
                hedge_entry_price,
            )
            return False

        # Subscribe both legs so the fetcher keeps refreshing their LTPs.
        self.store.register_option_subscription(
            OptionSubscription(
                security_id=int(main["security_id"]),
                exchange_segment=str(main["exchange_segment"]),
                trading_symbol=str(main["trading_symbol"]),
                right=str(main["option_type"]),
                strike=float(main["strike"]),
                expiry=main["expiry_date"],
            )
        )
        self.store.register_option_subscription(
            OptionSubscription(
                security_id=int(hedge["security_id"]),
                exchange_segment=str(hedge["exchange_segment"]),
                trading_symbol=str(hedge["trading_symbol"]),
                right=str(hedge["option_type"]),
                strike=float(hedge["strike"]),
                expiry=hedge["expiry_date"],
            )
        )

        main_order_id = self._next_paper_order_id("SELL")
        hedge_order_id = self._next_paper_order_id("BUY")

        self.pos = HedgedPaperPosition(
            active=True,
            direction="LONG",
            entry_underlying=float(entry_underlying),
            entry_timestamp=datetime.now(),
            main_symbol=str(main["trading_symbol"]),
            main_side="SELL",
            main_security_id=int(main["security_id"]),
            main_exchange_segment=str(main["exchange_segment"]),
            main_right=str(main["option_type"]),
            main_strike=float(main["strike"]),
            main_expiry=main["expiry_date"],
            main_lot_size=main_lot_size,
            main_quantity=main_qty,
            main_entry_price=main_entry_price,
            main_entry_order_id=main_order_id,
            hedge_symbol=str(hedge["trading_symbol"]),
            hedge_side="BUY",
            hedge_security_id=int(hedge["security_id"]),
            hedge_exchange_segment=str(hedge["exchange_segment"]),
            hedge_right=str(hedge["option_type"]),
            hedge_strike=float(hedge["strike"]),
            hedge_expiry=hedge["expiry_date"],
            hedge_lot_size=hedge_lot_size,
            hedge_quantity=hedge_qty,
            hedge_entry_price=hedge_entry_price,
            hedge_entry_order_id=hedge_order_id,
        )

        self.entry_submit_count += 1
        net_credit = (main_entry_price * main_qty) - (hedge_entry_price * hedge_qty)
        expiry_txt = expiry.isoformat() if expiry is not None else "NA"
        self.log.info(
            "ENTRY LONG (bullish) | Candle=%s | Expiry=%s | "
            "MainSELL %s Strike=%.2f Qty=%s @ %.2f | "
            "HedgeBUY %s Strike=%.2f Qty=%s @ %.2f | "
            "Spot=%.2f | EntryUnderlying=%.2f | NetCredit=%.2f | "
            "MainPaperRef=%s | HedgePaperRef=%s",
            latest_candle_ts,
            expiry_txt,
            self.pos.main_symbol,
            self.pos.main_strike,
            main_qty,
            main_entry_price,
            self.pos.hedge_symbol,
            self.pos.hedge_strike,
            hedge_qty,
            hedge_entry_price,
            spot,
            entry_underlying,
            net_credit,
            main_order_id,
            hedge_order_id,
        )
        return True

    # ------------------------------------------------------------------
    # Exit and PnL
    # ------------------------------------------------------------------
    def _get_open_position_pnl(self) -> float:
        """
        Live MTM on the open hedged position.

        Main leg (SOLD PE)   -> (entry - live) * qty.
        Hedge leg (BOUGHT PE) -> (live - entry) * qty.
        """
        if not self.pos.active:
            return 0.0
        main_live = self._get_option_ltp(
            self.pos.main_exchange_segment,
            self.pos.main_security_id,
            fallback=self.pos.main_entry_price,
        )
        hedge_live = self._get_option_ltp(
            self.pos.hedge_exchange_segment,
            self.pos.hedge_security_id,
            fallback=self.pos.hedge_entry_price,
        )
        main_pnl = (self.pos.main_entry_price - main_live) * self.pos.main_quantity
        hedge_pnl = (hedge_live - self.pos.hedge_entry_price) * self.pos.hedge_quantity
        return float(main_pnl + hedge_pnl)

    def exit_position(self, reason: str) -> None:
        """Close both PE legs of the bullish hedged trade and start cool-off."""
        if not self.pos.active:
            return

        closed = self.pos
        main_exit_price = self._get_option_ltp(
            closed.main_exchange_segment,
            closed.main_security_id,
            fallback=closed.main_entry_price,
        )
        hedge_exit_price = self._get_option_ltp(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
            fallback=closed.hedge_entry_price,
        )

        main_pnl = (closed.main_entry_price - main_exit_price) * closed.main_quantity
        hedge_pnl = (hedge_exit_price - closed.hedge_entry_price) * closed.hedge_quantity
        total_pnl = main_pnl + hedge_pnl

        self.completed_trades += 1
        self.exit_count += 1
        self.realized_pnl += total_pnl
        # Stamp the cool-off clock the moment the exit lands.
        self.last_exit_datetime = datetime.now()

        total_pnl_colored = self._color_pnl_text(total_pnl)
        main_order_id = self._next_paper_order_id("BUY")   # Buy-to-close main.
        hedge_order_id = self._next_paper_order_id("SELL") # Sell-to-close hedge.

        expiry_txt = closed.main_expiry.isoformat() if closed.main_expiry is not None else "NA"
        self.log.info(
            "EXIT %s | Reason=%s | Expiry=%s | "
            "MainBUY-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "HedgeSELL-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "TotalPnL=%.2f | CumPnL=%.2f | MainPaperRef=%s | HedgePaperRef=%s",
            closed.direction,
            reason,
            expiry_txt,
            closed.main_symbol,
            closed.main_strike,
            closed.main_quantity,
            closed.main_entry_price,
            main_exit_price,
            main_pnl,
            closed.hedge_symbol,
            closed.hedge_strike,
            closed.hedge_quantity,
            closed.hedge_entry_price,
            hedge_exit_price,
            hedge_pnl,
            total_pnl,
            self.realized_pnl,
            main_order_id,
            hedge_order_id,
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | {self.strategy_name} EXIT {closed.direction} | "
            f"Reason={reason} | TotalP&L={total_pnl_colored} | "
            f"Main={closed.main_symbol}@{closed.main_entry_price:.2f}->{main_exit_price:.2f} | "
            f"Hedge={closed.hedge_symbol}@{closed.hedge_entry_price:.2f}->{hedge_exit_price:.2f}"
        )

        self.store.unregister_option_subscription(
            closed.main_exchange_segment,
            closed.main_security_id,
        )
        self.store.unregister_option_subscription(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
        )

        self.after_exit(closed, reason)
        self.pos = HedgedPaperPosition()


# =============================================================================
# DONCHIAN BEARISH WORKER (5-min, hedged CE spread)
# =============================================================================
class DonchianBearishWorker(BasePaperStrategyWorker):
    """
    Hedged-CE bearish strategy.

    SPEC RECAP
    ----------
    - ENTRY: on a CLOSED 5-minute candle where the Donchian Bearish signal
      generator emits `startShortTrade == True` AND the wall-clock is inside
      the bearish entry window (09:25 - 10:59).
        * MAIN  (SELL): current-week CE with live LTP closest to Rs.160.
        * HEDGE (BUY) : current-week CE with live LTP closest to Rs.10.
    - EXIT : the signal generator does NOT emit exit signals. Exits come
      from three sources, ALL checked on every poll (not only on new
      5-min candle closes):
        * SL     : NIFTY spot >= entry_spot * (1 + 0.0020) -> EXIT_SL
        * TARGET : NIFTY spot <= entry_spot * (1 - 0.0040) -> EXIT_TARGET
        * 15:14  : universal time cutoff -> TIME_CUTOFF_1514
    - WINDOW : NO new entries after 10:59. Positions opened during the
      entry window keep running until SL/target/15:14 fires.

    EXPIRY RULE: this is the Hedged Puts family, so we use
    `OptionsContractResolver.get_current_week_expiry()`.

    STRIKE RULE: lives in this worker (`_pick_hedged_calls`).

    RUN-LOOP OVERRIDE: the base loop only dispatches when the candle
    signature changes, which is correct for ENTRIES (closed-candle
    semantics) but wrong for EXITS (must respond to live spot moves).
    We override `run` to run SL/target every poll and dispatch entries
    on the signature gate.
    """

    strategy_name = "DonchianBearish"
    timeframe = "1"
    poll_seconds = BEARISH_POLL_SECONDS
    lots = BEARISH_LOTS
    max_loss = BEARISH_MAX_LOSS
    trading_start_hour = BEARISH_TRADING_START_HOUR
    trading_start_minute = BEARISH_TRADING_START_MINUTE
    # Universal square-off so this worker aligns with Supertrend at 15:14.
    square_off_hour = SUPERTREND_SQUARE_OFF_HOUR
    square_off_minute = SUPERTREND_SQUARE_OFF_MINUTE

    derived_timeframe_minutes = BEARISH_TIMEFRAME_MINUTES
    # Donchian needs `length + a few` rows for a meaningful latest signal.
    min_strategy_rows = DONCHIAN_LENGTH + 5

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        # Two-leg shape replaces the default single-leg PaperPosition.
        self.pos = HedgedPaperPosition()
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        # Used to log the "outside entry window" message at most once
        # per 5-minute candle.
        self._cutoff_log_candle = None

    def minimum_strategy_rows(self) -> int:
        return int(self.min_strategy_rows)

    def summary_text(self) -> str:
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | "
            f"RealizedPnL={self.realized_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Strategy-frame plumbing
    # ------------------------------------------------------------------
    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        ohlc_5m = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        if ohlc_5m is None or ohlc_5m.empty:
            return pd.DataFrame()
        if len(ohlc_5m) < self.minimum_strategy_rows():
            return ohlc_5m
        return DONCHIAN_BEARISH_LOGIC.generate_donchian_bearish_signals(
            ohlc_5m, DONCHIAN_BEARISH_SETTINGS
        )

    # ------------------------------------------------------------------
    # Entry window
    # ------------------------------------------------------------------
    def _is_inside_entry_window(self) -> bool:
        """True between BEARISH_TRADING_START_*:* and BEARISH_ENTRY_CUTOFF_*:*."""
        if is_before_time(self.trading_start_hour, self.trading_start_minute):
            return False
        if is_after_time(BEARISH_ENTRY_CUTOFF_HOUR, BEARISH_ENTRY_CUTOFF_MINUTE):
            return False
        return True

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        ENTRY-only logic. Exits for SL/target are handled in `run()` so they
        fire on every poll regardless of new candle closes.
        """
        if strategy_frame is None or strategy_frame.empty:
            return
        if "startShortTrade" not in strategy_frame.columns:
            return

        last_row = strategy_frame.iloc[-1]
        start_short = bool(last_row.get("startShortTrade", False))
        close_price = _safe_float(last_row.get("close", 0.0), 0.0)
        latest_candle_ts = pd.to_datetime(last_row.get("timestamp"), errors="coerce")

        if self.pos.active:
            return  # Existing position -> no entry path on the candle close.

        if not start_short:
            return

        self.signal_count += 1

        if not self._is_inside_entry_window():
            if self._cutoff_log_candle != latest_candle_ts:
                self._cutoff_log_candle = latest_candle_ts
                self.log.info(
                    "Bearish signal at candle %s ignored: outside entry window "
                    "(%02d:%02d -> %02d:%02d).",
                    latest_candle_ts,
                    self.trading_start_hour,
                    self.trading_start_minute,
                    BEARISH_ENTRY_CUTOFF_HOUR,
                    BEARISH_ENTRY_CUTOFF_MINUTE,
                )
            return

        if close_price <= 0:
            self.log.warning(
                "Bearish signal fired but candle close looks invalid (%.4f). Skipping.",
                close_price,
            )
            return

        self._enter_hedged_bearish_position(close_price, latest_candle_ts)

    # ------------------------------------------------------------------
    # Hedged entry (CE legs - strike selection lives HERE per the user's hint)
    # ------------------------------------------------------------------
    def _pick_hedged_calls(self, spot_price: float) -> Optional[tuple[dict, dict, date]]:
        """
        Pick the main (SELL ~Rs.160) and hedge (BUY ~Rs.10) CE legs for the
        CURRENT-WEEK expiry. Mirror of `_pick_hedged_puts` with PE swapped
        for CE.
        """
        try:
            expiry = self.contract_resolver.get_current_week_expiry()
        except Exception as exc:
            self.log.warning("Cannot resolve current-week expiry: %s", exc)
            return None

        try:
            calls = self.contract_resolver.list_calls_for_expiry(expiry)
        except Exception as exc:
            self.log.warning("Cannot list CE rows for expiry %s: %s", expiry, exc)
            return None
        if calls.empty:
            self.log.warning("No CE options available for expiry %s.", expiry)
            return None

        sec_ids = [int(sid) for sid in calls["security_id"].tolist() if int(sid) > 0]
        if not sec_ids:
            self.log.warning("CE list for expiry %s has no valid security_ids.", expiry)
            return None

        try:
            ltp_map = self.broker.fetch_ltp_map({OPTION_EXCHANGE_SEGMENT: sec_ids})
        except Exception as exc:
            self.log.warning("CE LTP batch fetch failed for expiry %s: %s", expiry, exc)
            return None
        if not ltp_map:
            self.log.warning(
                "CE LTP batch fetch returned no prices for expiry %s (spot=%.2f).",
                expiry,
                spot_price,
            )
            return None

        self.store.update_ltp_map(ltp_map)

        try:
            main = self.contract_resolver.pick_call_by_target_premium(
                calls, ltp_map, MAIN_CALL_TARGET_PREMIUM
            )
        except Exception as exc:
            self.log.warning(
                "Cannot pick main call near %.2f for expiry %s: %s",
                MAIN_CALL_TARGET_PREMIUM,
                expiry,
                exc,
            )
            return None

        try:
            hedge = self.contract_resolver.pick_call_by_target_premium(
                calls,
                ltp_map,
                HEDGE_CALL_TARGET_PREMIUM,
                exclude_security_ids={int(main["security_id"])},
            )
        except Exception as exc:
            self.log.warning(
                "Cannot pick hedge call near %.2f for expiry %s: %s",
                HEDGE_CALL_TARGET_PREMIUM,
                expiry,
                exc,
            )
            return None

        return main, hedge, expiry

    def _enter_hedged_bearish_position(self, entry_underlying: float, latest_candle_ts) -> bool:
        """Open a new hedged bearish paper position (SELL Rs.160 CE + BUY Rs.10 CE)."""
        spot = self._get_underlying_spot(fallback=entry_underlying)
        if spot <= 0:
            self.log.warning("Skipping bearish entry because NIFTY spot LTP was unavailable.")
            return False

        picked = self._pick_hedged_calls(spot)
        if picked is None:
            return False
        main, hedge, expiry = picked

        main_lot_size = _to_int_safe(main.get("lot_size"), 0)
        hedge_lot_size = _to_int_safe(hedge.get("lot_size"), 0)
        if main_lot_size <= 0 or hedge_lot_size <= 0:
            self.log.warning(
                "Skipping bearish entry: invalid lot size (main=%s hedge=%s).",
                main_lot_size,
                hedge_lot_size,
            )
            return False

        main_qty = main_lot_size * self.lots
        hedge_qty = hedge_lot_size * self.lots
        main_entry_price = _safe_float(main.get("entry_ltp"), 0.0)
        hedge_entry_price = _safe_float(hedge.get("entry_ltp"), 0.0)
        if main_entry_price <= 0 or hedge_entry_price <= 0:
            self.log.warning(
                "Skipping bearish entry: invalid entry premium(s) main=%.2f hedge=%.2f.",
                main_entry_price,
                hedge_entry_price,
            )
            return False

        self.store.register_option_subscription(
            OptionSubscription(
                security_id=int(main["security_id"]),
                exchange_segment=str(main["exchange_segment"]),
                trading_symbol=str(main["trading_symbol"]),
                right=str(main["option_type"]),
                strike=float(main["strike"]),
                expiry=main["expiry_date"],
            )
        )
        self.store.register_option_subscription(
            OptionSubscription(
                security_id=int(hedge["security_id"]),
                exchange_segment=str(hedge["exchange_segment"]),
                trading_symbol=str(hedge["trading_symbol"]),
                right=str(hedge["option_type"]),
                strike=float(hedge["strike"]),
                expiry=hedge["expiry_date"],
            )
        )

        main_order_id = self._next_paper_order_id("SELL")
        hedge_order_id = self._next_paper_order_id("BUY")

        # `entry_underlying` = spot at entry (anchor for SL/target).
        self.pos = HedgedPaperPosition(
            active=True,
            direction="SHORT",
            entry_underlying=float(spot),
            entry_timestamp=datetime.now(),
            main_symbol=str(main["trading_symbol"]),
            main_side="SELL",
            main_security_id=int(main["security_id"]),
            main_exchange_segment=str(main["exchange_segment"]),
            main_right=str(main["option_type"]),
            main_strike=float(main["strike"]),
            main_expiry=main["expiry_date"],
            main_lot_size=main_lot_size,
            main_quantity=main_qty,
            main_entry_price=main_entry_price,
            main_entry_order_id=main_order_id,
            hedge_symbol=str(hedge["trading_symbol"]),
            hedge_side="BUY",
            hedge_security_id=int(hedge["security_id"]),
            hedge_exchange_segment=str(hedge["exchange_segment"]),
            hedge_right=str(hedge["option_type"]),
            hedge_strike=float(hedge["strike"]),
            hedge_expiry=hedge["expiry_date"],
            hedge_lot_size=hedge_lot_size,
            hedge_quantity=hedge_qty,
            hedge_entry_price=hedge_entry_price,
            hedge_entry_order_id=hedge_order_id,
        )

        self.entry_submit_count += 1
        net_credit = (main_entry_price * main_qty) - (hedge_entry_price * hedge_qty)
        sl_level = spot * (1.0 + BEARISH_SL_UP_PCT)
        tp_level = spot * (1.0 - BEARISH_TARGET_DOWN_PCT)
        expiry_txt = expiry.isoformat() if expiry is not None else "NA"
        self.log.info(
            "ENTRY SHORT (bearish) | Candle=%s | Expiry=%s | "
            "MainSELL %s Strike=%.2f Qty=%s @ %.2f | "
            "HedgeBUY %s Strike=%.2f Qty=%s @ %.2f | "
            "Spot=%.2f | SL(spot>=)=%.2f | Target(spot<=)=%.2f | "
            "NetCredit=%.2f | MainPaperRef=%s | HedgePaperRef=%s",
            latest_candle_ts,
            expiry_txt,
            self.pos.main_symbol,
            self.pos.main_strike,
            main_qty,
            main_entry_price,
            self.pos.hedge_symbol,
            self.pos.hedge_strike,
            hedge_qty,
            hedge_entry_price,
            spot,
            sl_level,
            tp_level,
            net_credit,
            main_order_id,
            hedge_order_id,
        )
        return True

    # ------------------------------------------------------------------
    # SL / target intra-poll exit check
    # ------------------------------------------------------------------
    def _check_sl_target_and_exit(self) -> bool:
        """
        Compare live spot against SL / target levels anchored at entry_spot.
        Returns True if an exit fired so the caller can skip the entry path
        for this poll cycle.
        """
        if not self.pos.active:
            return False
        entry_spot = float(self.pos.entry_underlying)
        if entry_spot <= 0:
            # Defensive: should never happen if entry populated the field.
            return False

        spot = self._get_underlying_spot(fallback=0.0)
        if spot <= 0:
            return False  # No live spot this poll - try again next time.

        sl_level = entry_spot * (1.0 + BEARISH_SL_UP_PCT)
        tp_level = entry_spot * (1.0 - BEARISH_TARGET_DOWN_PCT)

        if spot >= sl_level:
            self.log.info(
                "Bearish SL hit | EntrySpot=%.2f | LiveSpot=%.2f | SL=%.2f",
                entry_spot,
                spot,
                sl_level,
            )
            self.exit_position("EXIT_SL")
            return True
        if spot <= tp_level:
            self.log.info(
                "Bearish target hit | EntrySpot=%.2f | LiveSpot=%.2f | Target=%.2f",
                entry_spot,
                spot,
                tp_level,
            )
            self.exit_position("EXIT_TARGET")
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop override: SL/target every poll, entries on signature change
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Same shape as `BasePaperStrategyWorker.run` but with an extra step:
        `_check_sl_target_and_exit()` runs every poll while a position is
        open. If an exit fires, we skip the entry path for that poll.
        """
        self.log.info("Starting %s strategy worker.", self.strategy_name)
        while not self.stop_event.is_set():
            try:
                breached, total_pnl, open_pnl = self.is_max_loss_breached()
                if breached:
                    self.handle_max_loss_and_stop(total_pnl, open_pnl)
                    break

                if is_after_time(self.square_off_hour, self.square_off_minute):
                    self.handle_square_off_and_stop()
                    break

                if is_before_time(self.trading_start_hour, self.trading_start_minute):
                    if not self.preopen_wait_logged:
                        self.log.info(
                            "Before strategy start (%02d:%02d). Waiting to begin processing.",
                            self.trading_start_hour,
                            self.trading_start_minute,
                        )
                        self.preopen_wait_logged = True
                    self.wait_for_next_poll()
                    continue
                self.preopen_wait_logged = False

                # SL / target check runs every poll while a position is open.
                if self._check_sl_target_and_exit():
                    self.wait_for_next_poll()
                    continue

                snapshot = self.store.get(self.timeframe)
                if snapshot is None or snapshot.frame.empty:
                    self.wait_for_next_poll()
                    continue
                if len(snapshot.frame) < MIN_BARS:
                    self.wait_for_next_poll()
                    continue

                strategy_frame = self.build_strategy_frame(snapshot.frame)
                if strategy_frame is None or strategy_frame.empty:
                    self.wait_for_next_poll()
                    continue
                if len(strategy_frame) < self.minimum_strategy_rows():
                    self.wait_for_next_poll()
                    continue

                frame_signature = build_last_row_signature(strategy_frame)
                if frame_signature is None:
                    self.wait_for_next_poll()
                    continue
                if self.last_processed_frame_signature == frame_signature:
                    self.wait_for_next_poll()
                    continue

                self.last_processed_frame_signature = frame_signature
                self.process_strategy_frame(strategy_frame)
            except Exception as exc:
                self.log.exception("Main loop error: %s", exc)

            self.wait_for_next_poll()

        self.log.info("%s strategy worker exited.", self.strategy_name)

    # ------------------------------------------------------------------
    # Exit and PnL (CE legs - same math as bullish, opposite right)
    # ------------------------------------------------------------------
    def _get_open_position_pnl(self) -> float:
        """
        Live MTM on the open bearish hedged position.

        Main leg (SOLD CE)    -> (entry - live) * qty.
        Hedge leg (BOUGHT CE) -> (live - entry) * qty.
        Identical math to the bullish version; only the option right (CE
        vs PE) differs because the legs being valued are different.
        """
        if not self.pos.active:
            return 0.0
        main_live = self._get_option_ltp(
            self.pos.main_exchange_segment,
            self.pos.main_security_id,
            fallback=self.pos.main_entry_price,
        )
        hedge_live = self._get_option_ltp(
            self.pos.hedge_exchange_segment,
            self.pos.hedge_security_id,
            fallback=self.pos.hedge_entry_price,
        )
        main_pnl = (self.pos.main_entry_price - main_live) * self.pos.main_quantity
        hedge_pnl = (hedge_live - self.pos.hedge_entry_price) * self.pos.hedge_quantity
        return float(main_pnl + hedge_pnl)

    def exit_position(self, reason: str) -> None:
        """Close both CE legs of the bearish hedged trade. No cool-off here."""
        if not self.pos.active:
            return

        closed = self.pos
        main_exit_price = self._get_option_ltp(
            closed.main_exchange_segment,
            closed.main_security_id,
            fallback=closed.main_entry_price,
        )
        hedge_exit_price = self._get_option_ltp(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
            fallback=closed.hedge_entry_price,
        )

        main_pnl = (closed.main_entry_price - main_exit_price) * closed.main_quantity
        hedge_pnl = (hedge_exit_price - closed.hedge_entry_price) * closed.hedge_quantity
        total_pnl = main_pnl + hedge_pnl

        self.completed_trades += 1
        self.exit_count += 1
        self.realized_pnl += total_pnl

        total_pnl_colored = self._color_pnl_text(total_pnl)
        main_order_id = self._next_paper_order_id("BUY")   # Buy-to-close main.
        hedge_order_id = self._next_paper_order_id("SELL") # Sell-to-close hedge.

        expiry_txt = closed.main_expiry.isoformat() if closed.main_expiry is not None else "NA"
        self.log.info(
            "EXIT %s | Reason=%s | Expiry=%s | "
            "MainBUY-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "HedgeSELL-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "TotalPnL=%.2f | CumPnL=%.2f | MainPaperRef=%s | HedgePaperRef=%s",
            closed.direction,
            reason,
            expiry_txt,
            closed.main_symbol,
            closed.main_strike,
            closed.main_quantity,
            closed.main_entry_price,
            main_exit_price,
            main_pnl,
            closed.hedge_symbol,
            closed.hedge_strike,
            closed.hedge_quantity,
            closed.hedge_entry_price,
            hedge_exit_price,
            hedge_pnl,
            total_pnl,
            self.realized_pnl,
            main_order_id,
            hedge_order_id,
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | {self.strategy_name} EXIT {closed.direction} | "
            f"Reason={reason} | TotalP&L={total_pnl_colored} | "
            f"Main={closed.main_symbol}@{closed.main_entry_price:.2f}->{main_exit_price:.2f} | "
            f"Hedge={closed.hedge_symbol}@{closed.hedge_entry_price:.2f}->{hedge_exit_price:.2f}"
        )

        self.store.unregister_option_subscription(
            closed.main_exchange_segment,
            closed.main_security_id,
        )
        self.store.unregister_option_subscription(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
        )

        self.after_exit(closed, reason)
        self.pos = HedgedPaperPosition()


# =============================================================================
# DELTA-0.2 HEDGED SPREAD WORKER (option-chain Greeks driven, dual-side)
# =============================================================================
class Delta20HedgedSpreadWorker(BasePaperStrategyWorker):
    """
    Hedged spread driven by option-chain Greeks, with two INDEPENDENT sides.

    BEGINNER-FRIENDLY OVERVIEW
    --------------------------
    Think of this strategy as TWO completely separate trades that just
    happen to share the same worker thread:
        * "CE side" : a CALL credit spread anchored at a 20-delta strike.
        * "PE side" : a PUT  credit spread anchored at a 20-delta strike.
    Each side has its own entry trigger, its own exit target, and its
    own active/flat state. They do not block or wait on each other.

    The "20-delta" anchor:
        At 09:20, we ask the broker for the live option chain (which
        includes Greeks). The CE we monitor is the call whose delta is
        closest to +0.20; the PE we monitor is the put whose delta is
        closest to -0.20. The premium of each at 09:20 is the "reference"
        we'll compare against for the rest of the session.

    Why these specific entry / exit triggers?
        * Entry on a 5% drop in the monitored premium = "the option is
          getting cheaper, the market is moving away from this strike,
          good time to be a seller". Selling the monitored strike +
          buying a further-OTM hedge gives us a defined-risk credit
          spread.
        * Exit on a 3x rise of the monitored premium = "the option got
          much more expensive, the market is rushing toward our short
          strike, time to close before assignment risk".
        * Strategy-wide max-loss (Rs.5000/lot) is a hard backstop in case
          the spread goes against us before the 3x trigger fires.
        * 15:20 force-close avoids carrying intraday positions overnight.

    SPEC RECAP (the user's brief, in code terms)
    --------------------------------------------
    - At 09:20 IST: pull the live option chain for the current-week
      expiry, pick:
        * the CE strike whose delta is closest to +DELTA20_TARGET_DELTA
        * the PE strike whose delta is closest to -DELTA20_TARGET_DELTA
      Snapshot each leg's premium at that moment as the reference.
    - ENTRY trigger (independent per side): when the monitored CE/PE LTP
      falls to <= ref * (1 - DELTA20_ENTRY_DROP_PCT) (default 5% drop):
        * SELL the monitored strike (main leg)
        * BUY the strike DELTA20_HEDGE_STRIKES_OTM further OTM (hedge leg)
      The CE and PE entries are NOT mutually exclusive - either side can
      enter while the other is still flat or already running.
    - EXIT conditions (per side):
        1. Per-side target: monitored strike's LTP >= ref *
           DELTA20_EXIT_MULTIPLIER (default 3x). Closes only that side.
        2. Strategy-wide max-loss: total realized + open PnL <=
           -(DELTA20_MAX_LOSS_PER_LOT * lots). Closes BOTH sides AND ends
           the worker for the day.
        3. Time cutoff at DELTA20_SQUARE_OFF_*:* (default 15:20). Closes
           anything still open and ends the worker for the day.
      Once a side has exited, it is NOT re-entered the same day. The
      monitor + hedge LTP subscriptions for that side are released.

    EXPIRY RULE
    -----------
    Current-week expiry (`OptionsContractResolver.get_current_week_expiry`),
    consistent with the other hedged-spread strategies in this runner.
    Short-dated weeklies decay fast, which is the seller's friend.

    WHY OVERRIDE `run()` INSTEAD OF USING THE BASE LOOP
    ---------------------------------------------------
    This strategy does not consume OHLC at all. The base loop's "candle
    signature" gate (which only dispatches when the latest 1-min candle
    changes) would gate our triggers as well, even though our triggers
    are driven purely by option-leg LTPs that the central fetcher
    refreshes every poll. Overriding `run` keeps the same risk / cutoff /
    pre-open scaffolding but skips the OHLC plumbing entirely.

    NON-OBVIOUS DESIGN POINTS
    -------------------------
    - `self.pos` (the base-class single-position handle) is intentionally
      left as a flat default. We carry two real positions in `self.ce_pos`
      and `self.pe_pos`. The base risk handlers are overridden to operate
      on both rather than on the unused `self.pos`.
    - Reference capture uses an absolute backoff timer instead of relying
      on the poll cadence, because DhanHQ's `/optionchain` endpoint has
      a tighter rate-limit than the LTP batch and we should not hammer
      it on every 2-second poll.
    - Each side's max_loss is shared, not per-side. The user said
      "5000 per lot for the strategy", so we treat both sides as a
      single risk pool: if the COMBINED PnL breaches the limit, we shut
      down everything.
    """

    strategy_name = "Delta20Hedged"
    timeframe = "1"
    poll_seconds = DELTA20_POLL_SECONDS
    lots = DELTA20_LOTS
    max_loss = DELTA20_MAX_LOSS
    trading_start_hour = DELTA20_TRADING_START_HOUR
    trading_start_minute = DELTA20_TRADING_START_MINUTE
    square_off_hour = DELTA20_SQUARE_OFF_HOUR
    square_off_minute = DELTA20_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)

        # ----- Position state (per side) --------------------------------
        # Two independent hedged positions. The base-class `self.pos`
        # (single-position) stays at a flat default and is unused; we
        # override the risk handlers below to consult ce_pos and pe_pos
        # directly so both sides flow through the shared risk plumbing.
        self.ce_pos = HedgedPaperPosition()
        self.pe_pos = HedgedPaperPosition()

        # ----- 09:20 reference-capture state ----------------------------
        # `reference_captured` is the master gate. Until it flips True,
        # no entries are evaluated and no LTP subscriptions exist for the
        # monitored legs. Once True, the rest of the day is pure
        # premium-based monitoring.
        self.reference_captured = False
        # Counter for how many times we tried (and failed) to capture the
        # reference. Used by `_log_capture_failure` to throttle log spam.
        self.reference_capture_failed_count = 0
        # Wall-clock time of the most recent capture attempt. We use this
        # to enforce a backoff between retries so we do not spam DhanHQ's
        # `/optionchain` endpoint when something is wrong.
        self.last_capture_attempt_at: Optional[datetime] = None
        self.capture_backoff = timedelta(seconds=DELTA20_CAPTURE_RETRY_SECONDS)
        # The expiry we captured against. Stored only for log clarity.
        self.reference_expiry: Optional[date] = None

        # ----- Per-side metadata snapshot at 09:20 ----------------------
        # Each *_meta is a dict with keys:
        #   security_id, exchange_segment, trading_symbol, custom_symbol,
        #   strike, option_type, expiry_date, lot_size
        # The "_monitor" leg is the strike we will SELL (chosen by delta).
        # The "_hedge"   leg is the strike we will BUY  (4 strikes
        # further OTM than the monitor leg).
        # All four are subscribed with the central fetcher at capture
        # time, so their LTPs stay fresh in the shared store throughout
        # the session without us needing to make extra broker calls.
        self.ce_monitor_meta: Optional[dict] = None
        self.ce_hedge_meta: Optional[dict] = None
        self.pe_monitor_meta: Optional[dict] = None
        self.pe_hedge_meta: Optional[dict] = None
        # The "reference" premium = the monitor leg's LTP at 09:20. This
        # is the yardstick: entries fire on a 5% drop below it; exits
        # fire on a 3x rise above it. The deltas are stored only for
        # logging clarity - they do not affect the trigger math.
        self.ce_ref_premium = 0.0
        self.pe_ref_premium = 0.0
        self.ce_ref_delta = 0.0
        self.pe_ref_delta = 0.0

        # ----- Per-side "already exited today" flags --------------------
        # Per the user's spec a side trades AT MOST ONCE per day. After
        # an exit (or if max-loss / time cutoff fires while the side was
        # open), we set the corresponding *_done flag so the entry
        # checker cannot re-fire that side later in the same session.
        self.ce_side_done = False
        self.pe_side_done = False

        # ----- Telemetry counters ---------------------------------------
        # `signal_count` increments every time an ENTRY trigger fires
        # (whether or not the entry actually went through; both fires
        # count as "the strategy saw a setup"). `entry_submit_count` and
        # `exit_count` track successful state transitions.
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0

    # ------------------------------------------------------------------
    # Lifecycle / risk overrides (we use ce_pos+pe_pos, not self.pos)
    # ------------------------------------------------------------------
    def _get_open_position_pnl(self) -> float:
        """
        Live mark-to-market PnL summed across BOTH sides.

        Used by the base class's `is_max_loss_breached` to decide
        whether the strategy-wide stoploss has been hit. The base class
        only knows about a single `self.pos`, so we override this to
        reach into our two real positions.

        Per side (CE or PE):
            main was SOLD  -> profit when live < entry
                              -> (entry - live) * qty
            hedge was BOUGHT -> profit when live > entry
                              -> (live - entry) * qty
        Total PnL = sum across both sides for legs currently active.
        Inactive sides contribute zero.
        """
        total = 0.0
        for side_pos in (self.ce_pos, self.pe_pos):
            if not side_pos.active:
                continue
            main_live = self._get_option_ltp(
                side_pos.main_exchange_segment,
                side_pos.main_security_id,
                fallback=side_pos.main_entry_price,
            )
            hedge_live = self._get_option_ltp(
                side_pos.hedge_exchange_segment,
                side_pos.hedge_security_id,
                fallback=side_pos.hedge_entry_price,
            )
            main_pnl = (side_pos.main_entry_price - main_live) * side_pos.main_quantity
            hedge_pnl = (hedge_live - side_pos.hedge_entry_price) * side_pos.hedge_quantity
            total += main_pnl + hedge_pnl
        return float(total)

    def exit_position(self, reason: str) -> None:
        """
        Close every open side at once.

        This is the single entry point the base-class risk and
        time-cutoff handlers expect. By overriding it we ensure both
        sides flow through the same exit pathway when the day ends,
        with the same `reason` string surfaced in both EXIT log lines.

        Note: this method is also reachable from `_check_exit` -> ...
        actually NO. The per-side 3x target uses `_exit_side` directly
        (so only that one side closes). This override is only used for
        STRATEGY-wide events (max-loss breach, time cutoff).
        """
        if self.ce_pos.active:
            self._exit_side("CE", reason)
        if self.pe_pos.active:
            self._exit_side("PE", reason)

    def handle_max_loss_and_stop(self, total_pnl: float, open_pnl: float) -> None:
        """One-time shutdown when the per-lot max loss is breached."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.error(
            "MAX_LOSS breached | Limit=%.2f (%.0f * %d lots) | "
            "SessionPnL=%.2f | OpenPnL=%.2f. Closing both sides and stopping.",
            self.max_loss,
            DELTA20_MAX_LOSS_PER_LOT,
            self.lots,
            total_pnl,
            open_pnl,
        )
        try:
            self.exit_position("MAX_LOSS_BREACH")
        except Exception as exc:
            self.log.exception("Error while exiting on max-loss breach: %s", exc)
        self._unsubscribe_unentered_legs()
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day due to max-loss shutdown.")

    def handle_square_off_and_stop(self) -> None:
        """One-time shutdown at the daily 15:20 cutoff."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.info(
            "%02d:%02d cutoff reached. Closing any open sides and halting trading for the day.",
            self.square_off_hour,
            self.square_off_minute,
        )
        try:
            self.exit_position("TIME_CUTOFF")
        except Exception as exc:
            self.log.exception("Error while exiting at cutoff: %s", exc)
        self._unsubscribe_unentered_legs()
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day.")

    def summary_text(self) -> str:
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | "
            f"RealizedPnL={self.realized_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Run loop (no OHLC dependency)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Main worker loop. Runs every DELTA20_POLL_SECONDS (default 2s)
        until either a stop_event arrives or one of the strategy-ending
        events fires (max-loss breach or 15:20 time cutoff).

        Per iteration, in order:
            1. Risk check    -> max-loss breach -> close everything, exit.
            2. Time cutoff   -> 15:20 reached -> close everything, exit.
            3. Pre-open wait -> before 09:20 -> sleep and retry.
            4. Reference capture -> at/after 09:20, fetch the option chain
               ONCE and pick the monitored CE/PE strikes. With backoff on
               failure (rate-limit / pre-market). Once captured, this step
               is a no-op for the rest of the session.
            5. Per-side EXIT checks (independent: CE first, then PE).
               Done before entries so a side that just hit 3x closes
               cleanly without confusing the entry path.
            6. Per-side ENTRY checks (independent: CE first, then PE).
               Each side's entry only fires if it is currently flat AND
               has not already exited once today.

        The loop is wrapped in a try/except so a single bad iteration
        (e.g. a transient network error) cannot kill the worker for the
        whole day - we log the trace and continue on the next poll.
        """
        self.log.info("Starting %s strategy worker.", self.strategy_name)
        while not self.stop_event.is_set():
            try:
                # 1. Risk check first. If the combined PnL across both
                #    sides has dipped below -(per_lot * lots), we shut
                #    everything down and break out of the loop.
                breached, total_pnl, open_pnl = self.is_max_loss_breached()
                if breached:
                    self.handle_max_loss_and_stop(total_pnl, open_pnl)
                    break

                # 2. Daily time cutoff (15:20 by default). Close anything
                #    still open and stop the worker for the day.
                if is_after_time(self.square_off_hour, self.square_off_minute):
                    self.handle_square_off_and_stop()
                    break

                # 3. Pre-open wait. We do nothing before 09:20 - that's
                #    when the reference snapshot is taken. The
                #    `preopen_wait_logged` flag prevents the "waiting"
                #    log line from spamming every poll.
                if is_before_time(self.trading_start_hour, self.trading_start_minute):
                    if not self.preopen_wait_logged:
                        self.log.info(
                            "Before strategy start (%02d:%02d). Waiting to capture delta-%.2f reference strikes.",
                            self.trading_start_hour,
                            self.trading_start_minute,
                            DELTA20_TARGET_DELTA,
                        )
                        self.preopen_wait_logged = True
                    self.wait_for_next_poll()
                    continue
                self.preopen_wait_logged = False

                # 4. Reference capture. The first time we run this branch
                #    we may need a few attempts (backoff-throttled) before
                #    DhanHQ returns a usable option chain. On success the
                #    flag flips True and this branch is skipped forever.
                if not self.reference_captured:
                    self._maybe_capture_reference()
                    if not self.reference_captured:
                        # Capture didn't succeed this poll. Try again
                        # after the next backoff window. Without this
                        # `continue`, we'd evaluate entry/exit using
                        # uninitialized reference data.
                        self.wait_for_next_poll()
                        continue

                # 5. Exits checked first: if a per-side 3x target is
                #    already met, taking it before evaluating new entries
                #    avoids any ambiguity about which leg is being touched
                #    this poll. (Defensive ordering; the two branches
                #    operate on different legs anyway, so they cannot
                #    interfere - but explicit is better than implicit.)
                self._check_exit("CE")
                self._check_exit("PE")

                # 6. Entries are independent per side and may both fire
                #    in the same poll if both monitored premiums dropped
                #    enough.
                self._check_entry("CE")
                self._check_entry("PE")
            except Exception as exc:
                # Never let one bad iteration kill the worker. Log the
                # full traceback (so the issue is visible in the log
                # file) and continue on the next poll.
                self.log.exception("Main loop error: %s", exc)

            self.wait_for_next_poll()

        self.log.info("%s strategy worker exited.", self.strategy_name)

    # ------------------------------------------------------------------
    # 09:20 reference capture (delta-0.2 monitored strikes + their hedges)
    # ------------------------------------------------------------------
    def _maybe_capture_reference(self) -> None:
        """
        Backoff-throttled wrapper around `_capture_reference`.

        Why throttle? DhanHQ's `/optionchain` endpoint has a tighter
        rate-limit than the LTP batch endpoint. The worker poll cadence
        (~2 seconds) would exceed that limit if we called option_chain
        on every poll. By keeping a `last_capture_attempt_at` timestamp
        and a `capture_backoff` (default 5s), we ensure at most one
        attempt per backoff window.

        On success this branch is never reached again because `run`
        gates it with `self.reference_captured`.
        """
        now = datetime.now()
        if (
            self.last_capture_attempt_at is not None
            and (now - self.last_capture_attempt_at) < self.capture_backoff
        ):
            # Still inside the backoff window after a recent attempt.
            # Skip silently - we'll retry on the next eligible poll.
            return
        self.last_capture_attempt_at = now
        self._capture_reference()

    def _capture_reference(self) -> None:
        """
        Build the day's CE/PE monitor + hedge metadata and store the
        reference premiums. Idempotent on success: once captured, this
        method is no-op'd by the `reference_captured` flag in `run`.

        Step-by-step (each step gates the next):
            1. Resolve the current-week expiry (Friday-of-this-week
               typically; the helper picks the FIRST listed expiry on
               or after today).
            2. Call DhanHQ `/optionchain` for that expiry.
            3. Flatten the response into {strike: {ce/pe: {delta, ltp}}}.
            4. Pick the CE strike with delta closest to +0.20 and the
               PE strike with delta closest to -0.20.
            5. Map both strikes (and their hedges 4 strikes further OTM)
               back to instrument-master rows so we have security_ids
               and lot sizes for trade execution / LTP subscription.
            6. Validate that both reference premiums look real (>0).
            7. Persist all of the above as instance state, subscribe the
               4 legs with the central fetcher, and flip the
               `reference_captured` flag.

        Any failure short-circuits the function (the flag stays False)
        so the run loop will retry after the backoff window.
        """
        try:
            expiry = self.contract_resolver.get_current_week_expiry()
        except Exception as exc:
            self._log_capture_failure(f"current-week expiry resolution failed: {exc}")
            return

        try:
            resp = self.broker.fetch_option_chain(
                NIFTY_INDEX_SECURITY_ID,
                NIFTY_INDEX_EXCHANGE_SEGMENT,
                expiry,
            )
        except Exception as exc:
            self._log_capture_failure(f"option_chain fetch raised: {exc}")
            return

        parsed = self._parse_option_chain_for_deltas(resp)
        if not parsed:
            self._log_capture_failure(
                "option_chain returned no usable strikes (rate-limited or pre-market?)"
            )
            return

        ce_pick = self._pick_strike_by_delta(parsed, +float(DELTA20_TARGET_DELTA), "ce")
        pe_pick = self._pick_strike_by_delta(parsed, -float(DELTA20_TARGET_DELTA), "pe")
        if ce_pick is None or pe_pick is None:
            self._log_capture_failure(
                f"could not find CE near +{DELTA20_TARGET_DELTA:.2f} "
                f"and PE near -{DELTA20_TARGET_DELTA:.2f}"
            )
            return

        ce_strike, ce_leg = ce_pick
        pe_strike, pe_leg = pe_pick

        try:
            ce_monitor = self.contract_resolver.get_option_for_strike(expiry, ce_strike, "CE")
            pe_monitor = self.contract_resolver.get_option_for_strike(expiry, pe_strike, "PE")
            ce_hedge_strike = float(ce_strike) + DELTA20_HEDGE_STRIKES_OTM * ATM_STRIKE_STEP
            pe_hedge_strike = float(pe_strike) - DELTA20_HEDGE_STRIKES_OTM * ATM_STRIKE_STEP
            ce_hedge = self.contract_resolver.get_option_for_strike(expiry, ce_hedge_strike, "CE")
            pe_hedge = self.contract_resolver.get_option_for_strike(expiry, pe_hedge_strike, "PE")
        except Exception as exc:
            self._log_capture_failure(f"contract resolution raised: {exc}")
            return

        missing = [
            label
            for label, meta in (
                ("ce_monitor", ce_monitor),
                ("ce_hedge", ce_hedge),
                ("pe_monitor", pe_monitor),
                ("pe_hedge", pe_hedge),
            )
            if meta is None
        ]
        if missing:
            self._log_capture_failure(
                f"could not resolve contract metadata for {missing} "
                f"(monitor CE={ce_strike} hedge={ce_hedge_strike}; "
                f"monitor PE={pe_strike} hedge={pe_hedge_strike})"
            )
            return

        ce_ref = float(ce_leg.get("ltp", 0.0))
        pe_ref = float(pe_leg.get("ltp", 0.0))
        if ce_ref <= 0 or pe_ref <= 0:
            self._log_capture_failure(
                f"reference premium invalid (CE={ce_ref:.2f}, PE={pe_ref:.2f})"
            )
            return

        self.reference_expiry = expiry
        self.ce_monitor_meta = ce_monitor
        self.ce_hedge_meta = ce_hedge
        self.pe_monitor_meta = pe_monitor
        self.pe_hedge_meta = pe_hedge
        self.ce_ref_premium = ce_ref
        self.pe_ref_premium = pe_ref
        self.ce_ref_delta = float(ce_leg.get("delta", 0.0))
        self.pe_ref_delta = float(pe_leg.get("delta", 0.0))

        # Subscribe all four legs so the central fetcher keeps refreshing
        # their LTPs from now on. Without this, monitoring (and entry
        # detection) would have to do its own per-poll LTP fetches.
        for meta in (ce_monitor, ce_hedge, pe_monitor, pe_hedge):
            self.store.register_option_subscription(
                OptionSubscription(
                    security_id=int(meta["security_id"]),
                    exchange_segment=str(meta["exchange_segment"]),
                    trading_symbol=str(meta["trading_symbol"]),
                    right=str(meta["option_type"]),
                    strike=float(meta["strike"]),
                    expiry=meta["expiry_date"],
                )
            )

        self.reference_captured = True
        self.log.info(
            "Reference captured | Time=%s | Expiry=%s | "
            "CE Monitor=%s Strike=%.2f Delta=%.4f Premium=%.2f / Hedge=%s Strike=%.2f | "
            "PE Monitor=%s Strike=%.2f Delta=%.4f Premium=%.2f / Hedge=%s Strike=%.2f | "
            "EntryTriggerDrop=%.2f%% ExitMultiplier=%.2fx",
            datetime.now().strftime("%H:%M:%S"),
            expiry.isoformat(),
            ce_monitor["trading_symbol"],
            ce_monitor["strike"],
            self.ce_ref_delta,
            self.ce_ref_premium,
            ce_hedge["trading_symbol"],
            ce_hedge["strike"],
            pe_monitor["trading_symbol"],
            pe_monitor["strike"],
            self.pe_ref_delta,
            self.pe_ref_premium,
            pe_hedge["trading_symbol"],
            pe_hedge["strike"],
            DELTA20_ENTRY_DROP_PCT * 100.0,
            DELTA20_EXIT_MULTIPLIER,
        )

    def _log_capture_failure(self, reason: str) -> None:
        """
        Log capture failures with rate-limited verbosity.

        The first 3 attempts are noisy (so an early misconfiguration is
        immediately visible in the log), then it backs off to once every
        30 attempts (~ every 2.5 minutes at the default 5-second backoff).
        """
        self.reference_capture_failed_count += 1
        n = self.reference_capture_failed_count
        if n <= 3 or (n % 30) == 0:
            self.log.warning(
                "Reference capture failed (attempt %d): %s. Will retry after backoff.",
                n,
                reason,
            )

    @staticmethod
    def _parse_option_chain_for_deltas(resp) -> dict:
        """
        Flatten DhanHQ's `/optionchain` payload to a simple dict shape.

        Output shape:
            {
              22000.0: {"ce": {"delta": 0.74, "ltp": 481.7},
                        "pe": {"delta": -0.10, "ltp": 22.1}},
              22050.0: {...},
              ...
            }

        Wire shape (abridged) we are simplifying:
            {
              "status": "success",
              "data": {
                "last_price": <spot>,
                "oc": {
                  "22000.000000": {
                    "ce": {"last_price": 481.7, "greeks": {"delta": 0.74, ...}, ...},
                    "pe": {"last_price": 22.1,  "greeks": {"delta": -0.21, ...}, ...}
                  },
                  ...
                }
              }
            }

        Defensive parsing because the DhanHQ wire format is occasionally
        loose (mixed case keys, nested data wrappers, missing greeks
        sub-dict, illiquid strikes with zero LTP, etc.). Strikes whose
        CE or PE is missing entirely (or has zero LTP) are dropped on
        the corresponding side, so the caller only sees liquid
        candidates and never picks a "20-delta" strike that has no
        actual market depth.
        """
        # Top-level guards: anything that is not a dict-shaped payload
        # gets normalized to an empty result so callers can simply check
        # `if not parsed:` rather than wrapping every read in try/except.
        if not isinstance(resp, dict):
            return {}
        status = str(resp.get("status", "")).strip().lower()
        if status and status != "success":
            return {}
        payload = resp.get("data")
        if not isinstance(payload, dict):
            return {}
        # The strike map is keyed under "oc" (DhanHQ shorthand for
        # "option chain"). We accept the all-caps form too just in case.
        oc = payload.get("oc") or payload.get("OC") or {}
        if not isinstance(oc, dict):
            return {}

        out: dict[float, dict] = {}
        for strike_str, leg_map in oc.items():
            if not isinstance(leg_map, dict):
                continue
            # DhanHQ keys strikes as strings like "22000.000000". Convert
            # to float so the picker can do numeric comparisons.
            try:
                strike = float(strike_str)
            except (TypeError, ValueError):
                continue

            # Process CE and PE legs separately. We accept the most
            # likely casing variants for robustness against minor wire
            # changes between DhanHQ versions.
            record: dict = {}
            for right_label, leg_keys in (
                ("ce", ("ce", "CE", "Ce")),
                ("pe", ("pe", "PE", "Pe")),
            ):
                leg = None
                for key in leg_keys:
                    candidate = leg_map.get(key)
                    if isinstance(candidate, dict):
                        leg = candidate
                        break
                if leg is None:
                    continue
                greeks = leg.get("greeks")
                if not isinstance(greeks, dict):
                    greeks = {}
                # DhanHQ sometimes uses "last_price" and sometimes "ltp".
                # We prefer the former and fall back to the latter.
                ltp = _safe_float(
                    leg.get("last_price")
                    if leg.get("last_price") is not None
                    else leg.get("ltp"),
                    0.0,
                )
                if ltp <= 0:
                    # Skip illiquid / non-quoted strikes entirely. We
                    # never want to pick a "20-delta" strike whose
                    # premium is zero - the entry math relies on a
                    # positive reference.
                    continue
                record[right_label] = {
                    "delta": _safe_float(greeks.get("delta"), 0.0),
                    "ltp": ltp,
                }
            if record:
                out[strike] = record
        return out

    @staticmethod
    def _pick_strike_by_delta(
        parsed_chain: dict,
        target_delta: float,
        right: str,
    ) -> Optional[tuple[float, dict]]:
        """
        Return (strike, leg_dict) whose delta is closest to `target_delta`.

        For the CE side we pass `target_delta=+DELTA20_TARGET_DELTA`
        (e.g. +0.20). For the PE side we pass the SIGNED target,
        i.e. `-DELTA20_TARGET_DELTA` (e.g. -0.20), because put deltas
        are negative. The picker compares signed deltas with `abs(diff)`,
        so the closest absolute distance wins.

        Strikes that lack a record on the given side or have an invalid
        (zero) LTP are skipped automatically by
        `_parse_option_chain_for_deltas`. Returns `None` if no candidate
        survives - callers treat this as "could not capture reference,
        try again on the next eligible poll".

        Args:
            parsed_chain: Output of `_parse_option_chain_for_deltas`.
            target_delta: Signed target (e.g. +0.20 for CE, -0.20 for PE).
            right: "ce" or "pe" (case-insensitive).

        Returns:
            (best_strike, leg_dict_with_delta_and_ltp) or None.
        """
        right_key = str(right).strip().lower()
        if right_key not in ("ce", "pe"):
            return None
        best_strike = None
        best_record = None
        # Track the smallest absolute distance seen so far. We start
        # with infinity so the first candidate always wins.
        best_diff = float("inf")
        for strike, leg_map in parsed_chain.items():
            leg = leg_map.get(right_key)
            if not leg:
                continue
            diff = abs(float(leg.get("delta", 0.0)) - float(target_delta))
            if diff < best_diff:
                best_diff = diff
                best_strike = float(strike)
                best_record = leg
        if best_strike is None or best_record is None:
            return None
        return (best_strike, best_record)

    # ------------------------------------------------------------------
    # Entry / exit (per-side, using captured monitor + hedge metadata)
    # ------------------------------------------------------------------
    def _check_entry(self, side: str) -> None:
        """
        Evaluate the entry trigger for one side ("CE" or "PE").

        Trigger: monitored leg's live LTP <= ref * (1 - DELTA20_ENTRY_DROP_PCT).
        In words: "the option that we plan to SELL has dropped at least
        5% in price since 09:20". A premium drop without spot moving
        through the strike usually means time-decay or a small move
        AWAY from the strike - both favorable for a credit seller.

        Sides that already have an open position, that already exited
        once today, or that lack reference metadata are skipped silently
        (these are normal mid-session states, not error conditions).
        """
        # Unpack just the fields we need for this side. The trailing
        # underscores are unused fields we deliberately ignore to keep
        # the call site short.
        side_pos, _, monitor_meta, hedge_meta, ref_premium, side_done = self._side_state(side)
        # Already trading or already done for the day - nothing to do.
        if side_pos.active or side_done:
            return
        # Reference data missing means capture failed and we shouldn't
        # be evaluating entries yet. Defensive guard - run() should
        # already gate on this.
        if monitor_meta is None or hedge_meta is None or ref_premium <= 0:
            return

        # Read the monitored leg's latest LTP from the central cache
        # (the fetcher subscribes it for us at capture time, so this is
        # essentially free). `fallback=0.0` lets us detect "LTP not
        # available yet" cleanly.
        monitor_live = self._get_option_ltp(
            str(monitor_meta["exchange_segment"]),
            int(monitor_meta["security_id"]),
            fallback=0.0,
        )
        if monitor_live <= 0:
            # No fresh LTP this poll. Try again on the next iteration.
            return

        # Compute the price at which we would enter. e.g. if ref=100
        # and drop=0.05, we enter when the live LTP touches 95.
        trigger_price = ref_premium * (1.0 - DELTA20_ENTRY_DROP_PCT)
        if monitor_live > trigger_price:
            # Premium hasn't dropped enough yet. Keep monitoring.
            return

        # Trigger fired - go open the spread.
        self._enter_side(side, ref_premium, monitor_live)

    def _check_exit(self, side: str) -> None:
        """
        Evaluate the per-side 3x target.

        Trigger: monitored leg's live LTP >= ref * DELTA20_EXIT_MULTIPLIER.
        In words: "the option we are short has tripled in price". That
        usually means the underlying is rushing toward our short strike
        (or implied volatility just spiked) - either way it's a clear
        signal to close before assignment risk grows further.

        Note: the OTHER two exit conditions in the user's spec
        (strategy-wide max-loss and 15:20 time-cutoff) are NOT handled
        here. Those flow through the base risk handlers and our
        `exit_position` override, which closes BOTH sides at once and
        also halts the worker for the day. This method only handles
        the per-side soft target.
        """
        side_pos, _, _, _, ref_premium, _ = self._side_state(side)
        if not side_pos.active or ref_premium <= 0:
            return

        # `fallback=side_pos.main_entry_price` keeps the math sane even
        # if the cache is briefly empty (e.g. first poll after entry):
        # we'd compare entry to entry, which fails the >= 3x test
        # cleanly without crashing.
        monitor_live = self._get_option_ltp(
            side_pos.main_exchange_segment,
            side_pos.main_security_id,
            fallback=side_pos.main_entry_price,
        )
        if monitor_live <= 0:
            return

        target_price = ref_premium * DELTA20_EXIT_MULTIPLIER
        if monitor_live >= target_price:
            self.log.info(
                "%s side 3x target hit | RefPremium=%.2f | LiveMonitorPx=%.2f | TargetPx=%.2f",
                side,
                ref_premium,
                monitor_live,
                target_price,
            )
            self._exit_side(side, "TARGET_3X")

    def _enter_side(self, side: str, ref_premium: float, monitor_live: float) -> bool:
        """
        Open a new hedged position on the given side at the current LTP.

        Spread shape (the same for both CE and PE sides):
            Main leg  : SELL the monitored strike at `monitor_live`.
            Hedge leg : BUY  the strike 4 steps further OTM at `hedge_live`.

        Net effect: we collect `monitor_live - hedge_live` per share as
        a credit upfront. Maximum loss is capped at the strike width
        (200 points by default = 4 strikes * 50 pts per strike) minus
        the credit collected. If both legs expire OTM (the typical
        outcome for sellers when the market doesn't move much), we keep
        the full credit.

        Args:
            side: "CE" or "PE".
            ref_premium: The 09:20 reference premium for this side.
                Used here only for log clarity / drop-pct calculation.
            monitor_live: The current LTP that triggered the entry. We
                use this as the paper-fill price for the main (sold) leg.

        Returns:
            True on a successful entry, False on any guard failure
            (missing hedge LTP, invalid lot size, etc.).
        """
        # Pull this side's metadata. We deliberately ignore the position
        # / done flags here - they were already checked by `_check_entry`.
        _, _, monitor_meta, hedge_meta, _, _ = self._side_state(side)
        if monitor_meta is None or hedge_meta is None:
            return False

        # The hedge leg's LTP is needed to record a realistic paper fill
        # price. If it isn't available yet we skip the entry rather than
        # entering with stale / synthetic data.
        hedge_live = self._get_option_ltp(
            str(hedge_meta["exchange_segment"]),
            int(hedge_meta["security_id"]),
            fallback=0.0,
        )
        if hedge_live <= 0:
            self.log.warning(
                "Skipping %s entry: hedge LTP unavailable (sym=%s sec=%s).",
                side,
                hedge_meta.get("trading_symbol"),
                hedge_meta.get("security_id"),
            )
            return False

        # Lot sizes come from the instrument-master CSV. NIFTY weekly
        # options usually report 75 here (one contract = 75 quantity).
        # Real-world trade quantity = lot_size * lots-multiplier.
        monitor_lot_size = _to_int_safe(monitor_meta.get("lot_size"), 0)
        hedge_lot_size = _to_int_safe(hedge_meta.get("lot_size"), 0)
        if monitor_lot_size <= 0 or hedge_lot_size <= 0:
            self.log.warning(
                "Skipping %s entry: invalid lot size (monitor=%s hedge=%s).",
                side,
                monitor_lot_size,
                hedge_lot_size,
            )
            return False

        monitor_qty = monitor_lot_size * self.lots
        hedge_qty = hedge_lot_size * self.lots
        # Synthetic order ids for paper-trade audit trail. These are
        # unique within the worker and timestamp-encoded so a human
        # scanning the log can correlate ENTRY and EXIT lines easily.
        main_order_id = self._next_paper_order_id("SELL")
        hedge_order_id = self._next_paper_order_id("BUY")
        # Snapshot the underlying spot at entry. Used purely for log
        # clarity (it lets us verify post-mortem that the spread was
        # opened sensibly relative to where NIFTY actually was).
        spot = self._get_underlying_spot(fallback=0.0)

        new_pos = HedgedPaperPosition(
            active=True,
            direction=side,  # "CE" or "PE" (used purely for log labelling)
            entry_underlying=float(spot),
            entry_timestamp=datetime.now(),
            main_symbol=str(monitor_meta["trading_symbol"]),
            main_side="SELL",
            main_security_id=int(monitor_meta["security_id"]),
            main_exchange_segment=str(monitor_meta["exchange_segment"]),
            main_right=str(monitor_meta["option_type"]),
            main_strike=float(monitor_meta["strike"]),
            main_expiry=monitor_meta["expiry_date"],
            main_lot_size=monitor_lot_size,
            main_quantity=monitor_qty,
            main_entry_price=float(monitor_live),
            main_entry_order_id=main_order_id,
            hedge_symbol=str(hedge_meta["trading_symbol"]),
            hedge_side="BUY",
            hedge_security_id=int(hedge_meta["security_id"]),
            hedge_exchange_segment=str(hedge_meta["exchange_segment"]),
            hedge_right=str(hedge_meta["option_type"]),
            hedge_strike=float(hedge_meta["strike"]),
            hedge_expiry=hedge_meta["expiry_date"],
            hedge_lot_size=hedge_lot_size,
            hedge_quantity=hedge_qty,
            hedge_entry_price=float(hedge_live),
            hedge_entry_order_id=hedge_order_id,
        )
        if side == "CE":
            self.ce_pos = new_pos
        else:
            self.pe_pos = new_pos

        self.signal_count += 1
        self.entry_submit_count += 1
        net_credit = (monitor_live * monitor_qty) - (hedge_live * hedge_qty)
        expiry_txt = (
            new_pos.main_expiry.isoformat() if new_pos.main_expiry is not None else "NA"
        )
        self.log.info(
            "ENTRY %s | RefPremium=%.2f | TriggerLTP=%.2f (drop=%.2f%% from ref) | "
            "Expiry=%s | MainSELL %s Strike=%.2f Qty=%s @ %.2f | "
            "HedgeBUY %s Strike=%.2f Qty=%s @ %.2f | Spot=%.2f | "
            "NetCredit=%.2f | MainPaperRef=%s | HedgePaperRef=%s",
            side,
            ref_premium,
            monitor_live,
            (1.0 - monitor_live / ref_premium) * 100.0 if ref_premium > 0 else 0.0,
            expiry_txt,
            new_pos.main_symbol,
            new_pos.main_strike,
            monitor_qty,
            monitor_live,
            new_pos.hedge_symbol,
            new_pos.hedge_strike,
            hedge_qty,
            hedge_live,
            spot,
            net_credit,
            main_order_id,
            hedge_order_id,
        )
        return True

    def _exit_side(self, side: str, reason: str) -> None:
        """
        Close the hedged position on the given side and book PnL.

        PnL math (paper):
            Main leg was SOLD at entry, so we BUY-TO-CLOSE on exit.
                main_pnl  = (entry - exit) * qty
                Positive when the option got cheaper (good for sellers).
            Hedge leg was BOUGHT at entry, so we SELL-TO-CLOSE on exit.
                hedge_pnl = (exit - entry) * qty
                Positive when the option got more expensive.
            Total side PnL = main_pnl + hedge_pnl.

        After the exit:
        - Both legs are unsubscribed from the central LTP fetcher (we
          stop wasting API budget watching legs we no longer hold).
        - The side's `*_done` flag flips True so the entry trigger
          cannot re-fire today.

        `reason` is a free-form short string ("TARGET_3X", "MAX_LOSS_BREACH",
        "TIME_CUTOFF") used in the EXIT log line for post-mortem analysis.
        """
        side_pos = self.ce_pos if side == "CE" else self.pe_pos
        if not side_pos.active:
            return

        # Snapshot the position so we can reset `self.ce_pos` /
        # `self.pe_pos` to a flat default at the end without losing the
        # data we need for logging and PnL math.
        closed = side_pos
        # Read the latest LTPs for both legs. The fallback to entry
        # price keeps the math defined even on the rare poll where the
        # LTP cache is briefly empty - a no-op PnL for that leg, which
        # an operator can easily spot in the log.
        main_exit_price = self._get_option_ltp(
            closed.main_exchange_segment,
            closed.main_security_id,
            fallback=closed.main_entry_price,
        )
        hedge_exit_price = self._get_option_ltp(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
            fallback=closed.hedge_entry_price,
        )
        # Main was SOLD -> profit if exit < entry  ->  (entry - exit) * qty
        # Hedge was BOUGHT -> profit if exit > entry -> (exit - entry) * qty
        main_pnl = (closed.main_entry_price - main_exit_price) * closed.main_quantity
        hedge_pnl = (hedge_exit_price - closed.hedge_entry_price) * closed.hedge_quantity
        total_pnl = main_pnl + hedge_pnl

        self.completed_trades += 1
        self.exit_count += 1
        self.realized_pnl += total_pnl

        total_pnl_colored = self._color_pnl_text(total_pnl)
        main_order_id = self._next_paper_order_id("BUY")    # Buy-to-close main.
        hedge_order_id = self._next_paper_order_id("SELL")  # Sell-to-close hedge.
        expiry_txt = closed.main_expiry.isoformat() if closed.main_expiry is not None else "NA"
        self.log.info(
            "EXIT %s | Reason=%s | Expiry=%s | "
            "MainBUY-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "HedgeSELL-TO-CLOSE %s Strike=%.2f Qty=%s | EntryPx=%.2f ExitPx=%.2f LegPnL=%.2f | "
            "TotalPnL=%.2f | CumPnL=%.2f | MainPaperRef=%s | HedgePaperRef=%s",
            side,
            reason,
            expiry_txt,
            closed.main_symbol,
            closed.main_strike,
            closed.main_quantity,
            closed.main_entry_price,
            main_exit_price,
            main_pnl,
            closed.hedge_symbol,
            closed.hedge_strike,
            closed.hedge_quantity,
            closed.hedge_entry_price,
            hedge_exit_price,
            hedge_pnl,
            total_pnl,
            self.realized_pnl,
            main_order_id,
            hedge_order_id,
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | {self.strategy_name} EXIT {side} | "
            f"Reason={reason} | TotalP&L={total_pnl_colored} | "
            f"Main={closed.main_symbol}@{closed.main_entry_price:.2f}->{main_exit_price:.2f} | "
            f"Hedge={closed.hedge_symbol}@{closed.hedge_entry_price:.2f}->{hedge_exit_price:.2f}"
        )

        # Once a side closes, drop its leg subscriptions and lock the side
        # so the entry trigger does not re-fire later in the day.
        self.store.unregister_option_subscription(
            closed.main_exchange_segment,
            closed.main_security_id,
        )
        self.store.unregister_option_subscription(
            closed.hedge_exchange_segment,
            closed.hedge_security_id,
        )
        if side == "CE":
            self.ce_pos = HedgedPaperPosition()
            self.ce_side_done = True
        else:
            self.pe_pos = HedgedPaperPosition()
            self.pe_side_done = True

    def _side_state(self, side: str):
        """
        Bundled accessor for one side's runtime state.

        Why bundle: every per-side helper (`_check_entry`, `_check_exit`,
        `_enter_side`) needs the SAME six fields, and writing the
        `if side == "CE": ... else: ...` ladder six times would hide
        the actual logic behind boilerplate. This helper centralizes
        the dispatch so the call sites read like normal local variables.

        Returns a 6-tuple in a fixed order:
            (position, ref_delta, monitor_meta, hedge_meta,
             ref_premium, side_done)

        Callers typically destructure with `_` placeholders for the
        fields they don't need (e.g. ref_delta is only useful for
        log lines, not trigger math).
        """
        if side == "CE":
            return (
                self.ce_pos,
                self.ce_ref_delta,
                self.ce_monitor_meta,
                self.ce_hedge_meta,
                self.ce_ref_premium,
                self.ce_side_done,
            )
        if side == "PE":
            return (
                self.pe_pos,
                self.pe_ref_delta,
                self.pe_monitor_meta,
                self.pe_hedge_meta,
                self.pe_ref_premium,
                self.pe_side_done,
            )
        raise ValueError(f"Unsupported side: {side!r}")

    def _unsubscribe_unentered_legs(self) -> None:
        """
        Drop any monitor/hedge LTP subscriptions for sides that never
        opened a position today.

        Why this matters: on a normal day the captured legs stay
        subscribed all session so the entry trigger can read fresh LTPs.
        When the day ends (max-loss or 15:20 cutoff) and a side never
        traded, the four legs would otherwise leak as orphan
        subscriptions, making the central fetcher poll LTPs we no
        longer care about. This cleanup runs once during the
        end-of-day shutdown handlers and removes those orphan
        subscriptions.

        Note: legs that DID open a position are unsubscribed inside
        `_exit_side` at the moment of exit, not here.
        """
        if not self.reference_captured:
            # Reference capture never succeeded - there is nothing to
            # clean up.
            return
        # Only unsubscribe a side's legs if that side never traded
        # today (active==False AND done==False). If the side did trade
        # and is now flat (active==False AND done==True), `_exit_side`
        # already unsubscribed them.
        if not self.ce_pos.active and not self.ce_side_done:
            for meta in (self.ce_monitor_meta, self.ce_hedge_meta):
                if meta is not None:
                    self.store.unregister_option_subscription(
                        str(meta["exchange_segment"]),
                        int(meta["security_id"]),
                    )
        if not self.pe_pos.active and not self.pe_side_done:
            for meta in (self.pe_monitor_meta, self.pe_hedge_meta):
                if meta is not None:
                    self.store.unregister_option_subscription(
                        str(meta["exchange_segment"]),
                        int(meta["security_id"]),
                    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Wire up the central fetcher plus the six strategy worker threads.

    Architecture summary:
    1. Configure logging (root logger so every thread benefits).
    2. Change working directory to the repo root so relative paths
       (instrument master, log files) stay valid no matter where the
       script is launched from.
    3. Build the shared store and shared stop signal.
    4. Build one fetcher and the six workers.
    5. Start the fetcher first so workers see fresh data on their first poll.
    6. Supervise: wait until workers exit or a Ctrl+C arrives.
    7. Signal shutdown and join every thread with a bounded timeout.

    The MAIN thread does NOT trade. It is purely a supervisor.
    """
    setup_logging()
    os.chdir(ROOT_DIR)

    # Credentials must come from `.env` (or the shell env). We never carry
    # in-code defaults for these because they are secrets.
    if not CLIENT_CODE or not ACCESS_TOKEN:
        raise ValueError(
            "DHAN_CLIENT_CODE and DHAN_ACCESS_TOKEN must be set in "
            "Multithreading/Dependencies/.env. If you have an API Key + "
            "API Secret but no access token yet, run:\n"
            "    python Multithreading/Dependencies/dhan_token_setup.py"
        )

    # Fail fast if the access token is invalid / expired. The user_profile
    # endpoint is cheap and gives us a clear early error instead of a
    # confusing failure inside an OHLC fetch later.
    try:
        DhanLogin(CLIENT_CODE).user_profile(ACCESS_TOKEN)
    except Exception as exc:
        raise ValueError(
            "DHAN_ACCESS_TOKEN failed validation against /v2/profile: "
            f"{exc}\nIf the token has expired, regenerate it with:\n"
            "    python Multithreading/Dependencies/dhan_token_setup.py"
        ) from exc

    logger.info(
        "Starting NIFTY Multi Strategy MASTER paper runner (dhanhq) | "
        "ATM family (4): Renko 1m, EMA 5m, HeikinAshi 1m, ProfitShooter 1m. "
        "Hedged Puts family (2): Supertrend 3m PE, Donchian 5m CE. "
        "Delta-0.2 family (1): Delta20 09:20 reference, dual-side."
    )

    broker = DhanBrokerClient(CLIENT_CODE, ACCESS_TOKEN)
    store = SharedMarketDataStore()
    stop_event = threading.Event()

    # One producer (fetcher) + seven consumers (4 ATM + 2 Hedged + 1 Delta20).
    fetcher = CentralMarketDataFetcher(store, stop_event, broker)
    workers = [
        # ATM family: each picks ATM CE/PE of the next-next expiry.
        RenkoStrategyWorker(store, stop_event, broker),
        EMATrendStrategyWorker(store, stop_event, broker),
        HeikinAshiStrategyWorker(store, stop_event, broker),
        ProfitShooterStrategyWorker(store, stop_event, broker),
        # Hedged Puts family: each picks current-week CE/PE by target premium.
        SupertrendBullishWorker(store, stop_event, broker),
        DonchianBearishWorker(store, stop_event, broker),
        # Delta-0.2 family: dual-side hedged spread driven by option-chain Greeks.
        Delta20HedgedSpreadWorker(store, stop_event, broker),
    ]

    fetcher.start()
    for worker in workers:
        worker.start()

    try:
        # The main thread is a passive supervisor; it does not trade.
        while any(worker.is_alive() for worker in workers):
            for worker in workers:
                worker.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Signaling all threads to stop.")
        stop_event.set()
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=SHUTDOWN_JOIN_SECONDS)
        fetcher.join(timeout=SHUTDOWN_JOIN_SECONDS)

        alive_threads = [thread.name for thread in [fetcher, *workers] if thread.is_alive()]
        if alive_threads:
            logger.warning("Some threads did not exit within the shutdown window: %s", alive_threads)
        else:
            logger.info("All threads exited cleanly.")


if __name__ == "__main__":
    main()
