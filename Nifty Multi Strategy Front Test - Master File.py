"""
================================================================================
Nifty Multi Strategy Front Test - Master File
================================================================================

WHAT THIS FILE IS (read this first if you have never seen this code base)
------------------------------------------------------------------------
This is a single multi-threaded paper-trading runner that combines TWENTY-SIX
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

Since that union, FOUR more single-leg ATM strategies were added directly to
this master file -- Goldmine, Money Machine (both 5-min, Subhamoy folder),
Opening Strike PCR/VWAP/ATR (5-min), and CPR (5-min) -- which brought the runner
to eleven strategies (8 ATM single-leg + 2 Hedged Puts + 1 Delta-0.2).

Most recently, THIRTEEN more single-leg ATM strategies were added -- ports of the
public TradingBot project (kept in the "Signal Generators" folder), built
from one shared factory: SMA Crossover, Bollinger Bands, Keltner Squeeze, Mean
Reversion Z-Score, ML Ensemble, Multi-Timeframe, Opening Range Breakout, Parabolic
SAR, RSI Divergence, RSI Reversal, Stochastic, Supertrend, and Volatility
Breakout. They are the SAME ATM single-leg family (a LONG buys the ATM CE, a SHORT
the ATM PE). (ML Ensemble needs scikit-learn.)

Two more were then added. CPR Algo 3 is a multi-instrument "CPR basic setup" ATM
worker: it watches the spot PLUS a ~ITM CE and a ~ITM PE of the current-week expiry
and only fires when VWAP and the CPR band align across all three, but it still
trades the ATM CE/PE of the next-next expiry like the rest of the ATM family. The
Long Strangle worker is a time-based two-leg BUY of the OTM1 weekly CE+PE, with
independent per-leg trailing stops and momentum re-entry. Together they bring the
runner to TWENTY-SIX strategies total: 22 ATM single-leg + 2 Hedged Puts +
1 Delta-0.2 + 1 Long Strangle.

One OPTIONAL, opt-in 27th worker can also be loaded: the SL Hunting AI Agent
(under "Signal Generators/SL Hunting AI Agent/"). Unlike every other strategy it
is LLM-driven -- a Claude agent (claude-agent-sdk, on your Claude subscription)
decides once per completed 1-min bar (the method's native timeframe) via in-process
tools -- with optional BankNIFTY cross-confirmation fetched per bar (like CPR Algo 3)
and dynamic ~Rs.2500 risk-based sizing -- then acts through this runner's own
enter_position/exit_position (so it is just another ATM single-leg member of the
family). It stops opening NEW positions after noon (SL_HUNTING_NO_NEW_ENTRY_HOUR,
default 12:00); that is NOT a square-off -- open positions, their stop/target, and the
15:15 square-off are all unaffected, and when flat past the cutoff it skips the LLM
call entirely. It is OFF by default; SL_HUNTING_ENABLED=true includes it, and its
optional deps (claude-agent-sdk, pydantic) are imported behind try/except so a missing
dep simply disables it without touching the rest of the runner. It can also learn from
its own trades (v3): a per-trade journal feeds an off-loop reflection coach that
proposes lessons, which the operator promotes into lessons.json and the agent injects
only when SL_HUNTING_LESSONS_ENABLED (human-gated, paper-first, off by default).

EXPIRY RULES (the explicit if-else the user asked for)
------------------------------------------------------
Different strategies pick different expiries. Rather than burying this
in a single overloaded method, every worker is explicit about which
expiry rule it uses:

    if strategy belongs to the "Hedged Puts" family, or is the Long Strangle
        -> use OptionsContractResolver.get_current_week_expiry()
           (the FIRST expiry on or after today)
    else
        -> use OptionsContractResolver.get_target_expiry()
           (the SECOND expiry on or after today, i.e. "next-next")

(CPR Algo 3 is "else" -- it TRADES the next-next ATM -- even though its read-only
observation legs use the current-week expiry.)

Both methods live on the same resolver so the choice is one line at
the call site. That keeps the rule visible in the code and easy to
audit later without chasing a hidden flag.

STRIKE RULES (also explicit per family)
---------------------------------------
- ATM family (the 22 ATM workers) -> resolver.get_atm_option(spot, dir).
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
Single-leg ATM trades (the 22 ATM workers):
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

- Twenty-six `*StrategyWorker` threads:
    * Read the 1-minute OHLC from the shared store.
    * Resample locally if their strategy timeframe is higher than 1m.
    * Run their own signal generator (the Delta-0.2 worker is the
      exception and reads option-chain Greeks instead of OHLC).
    * Manage their own paper position (single-leg OR hedged).
    * Never call the broker for OHLC directly. They only do tiny
      direct LTP fallback fetches when the cache is cold.

This central-fetcher design keeps API usage low and makes the data
deterministic across all twenty-six strategies.

WHY ONE FILE INSTEAD OF SEPARATE FILES
--------------------------------------
- Single fetch budget. One DhanHQ ticker_data call covers spot plus
  every active option leg from every worker simultaneously.
- One log destination. All twenty-six strategies share LOG_FILE so a single
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
        |       +-- GoldmineStrategyWorker
        |       +-- MoneyMachineStrategyWorker
        |       +-- OpeningStrikePCRVWAPATRWorker
        |       +-- CPRStrategyWorker
        |       +-- CPRAlgo3StrategyWorker      (multi-instrument: spot + ITM CE + ITM PE)
        |       +-- (+ 13 TradingBot ports, built via _build_signal_gen_worker_class:
        |             SMA Crossover, Bollinger Bands, Keltner Squeeze, Mean Reversion
        |             Z-Score, ML Ensemble, Multi-Timeframe, Opening Range Breakout,
        |             Parabolic SAR, RSI Divergence, RSI Reversal, Stochastic,
        |             Supertrend, Volatility Breakout)
        |       +-- SLHuntingAIWorker         (OPTIONAL opt-in 27th: LLM/Claude-agent driven)
        |
        +-- SupertrendBullishWorker    (hedged PE spread)
        +-- DonchianBearishWorker      (hedged CE spread)
        +-- Delta20HedgedSpreadWorker  (dual-side hedged spread, Greeks-driven)
        +-- LongStrangleWorker         (time-based two-leg BUY OTM1 CE+PE, momentum re-entry)

The split is intentional: only the ATM workers (the 9 core + 13 ports) share an
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
import html
import importlib.util
import logging
import math
import os
import queue
import re
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

# --- Third-party imports -----------------------------------------------------
import pandas as pd
# `requests` is only used by the end-of-day instrument-master refresh helper at
# the bottom of this file. We keep it next to `pandas` rather than lazy-loading
# it inside the helper because the rest of the repo (e.g. the swing strategy
# data fetcher) already depends on `requests`, so it is guaranteed to be present.
import requests
# `DhanContext` wraps (client_id, access_token) so the rest of the SDK can
# share one authenticated session. `DhanLogin` is the auth helper class we
# use only for `user_profile(...)` startup validation -- the OAuth dance
# itself happens in `Dependencies/dhan_token_setup.py`, not here.
from dhanhq import DhanContext, DhanLogin, dhanhq

try:
    # `python-dotenv` is optional. If installed, values such as the DhanHQ
    # credentials and the strategy tuning parameters can be auto-loaded from the
    # local `.env` file in the Dependencies/ folder.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


# =============================================================================
# STATIC FILE-LEVEL CONFIGURATION
# =============================================================================
# Constants that never come from `.env` because they are tied to the project
# layout or are pure visual / architectural identifiers.
#
# This master file lives at the repo root and the strategy logic lives under
# `Signal Generators/`, so ROOT_DIR is simply the master file's own directory.
ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "Dependencies" / "log_files" / "nifty_multi_strategy_master_front_test_dhanhq.log"
LOGGER_NAME = "nifty_multi_strategy_master_front_test_dhanhq"

INSTRUMENT_MASTER_GLOB = str(ROOT_DIR / "Dependencies" / "all_instrument *.csv")

# Public DhanHQ scrip master used by the end-of-day refresh helper at the
# bottom of this file. The DETAILED variant is required because the option
# resolver (`_load_option_chain`) depends on the detailed option columns
# (`EXCH_ID`, `SEGMENT`, `INSTRUMENT`, `SYMBOL_NAME`, `DISPLAY_NAME`,
# `SM_EXPIRY_DATE`, `LOT_SIZE`, `SECURITY_ID`, `STRIKE_PRICE`, `OPTION_TYPE`,
# `UNDERLYING_SYMBOL`). The shorter `api-scrip-master.csv` does not carry the
# same complete option metadata and would break the resolver on the next run.
DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

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


def _env_bool(name: str, default: bool = False) -> bool:
    """
    Read a yes/no setting from the .env file and return a real True/False.

    .env values are always text, so this turns words like "true"/"yes"/"on"/"1"
    into True (case doesn't matter, surrounding quotes are ignored). If the key is
    missing or blank it returns `default`; anything else counts as False.
    Example: LIVE_TRADING_ENABLED=true  ->  _env_bool("LIVE_TRADING_ENABLED")
    """
    raw = os.getenv(name, "")
    if raw is None:
        return bool(default)
    raw = raw.strip().strip('"').strip("'").lower()
    if raw == "":
        return bool(default)
    return raw in ("1", "true", "yes", "on")


# =============================================================================
# DHANHQ CREDENTIALS (read STRICTLY from .env; no in-code defaults)
# =============================================================================
# All four values are populated by the user / by `Dependencies/dhan_token_setup.py`.
# The runner refuses to start if `CLIENT_CODE` or `ACCESS_TOKEN` is missing -
# see `main()` for the validation message.
#
# DHAN_CLIENT_CODE  : 10-digit dhanClientId (e.g. "1100000000").
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
# strategies even after the 5-min resamplers downsample 1m -> 5m
# (still ~24 bars).
MIN_BARS = _env_int("MIN_BARS", 120)

# How often the fetcher re-polls 1-minute OHLC and the LTP batch (seconds).
FETCH_POLL_SECONDS = _env_int("FETCH_POLL_SECONDS", 2)

# Bounded join timeout for each thread on shutdown.
SHUTDOWN_JOIN_SECONDS = _env_float("SHUTDOWN_JOIN_SECONDS", 6.0)

# Telegram trade-notification settings. See Dependencies/.env for the one-time
# bot/channel setup. When disabled (or token/chat blank) the notifier thread is
# never started and workers' publish_trade_event() calls are cheap no-ops.
TELEGRAM_ENABLED = _env_str("TELEGRAM_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
TELEGRAM_BOT_TOKEN = _env_str("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = _env_str("TELEGRAM_CHAT_ID", "").strip()

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
# BankNIFTY index identity (DhanHQ security_id 25 on the IDX_I segment). Only the
# optional SL Hunting AI Agent uses these today, to fetch BankNIFTY 1-min OHLC for
# its NF/BNF cross-confirmation (the same fetch_index_1m_ohlc path the fetcher and
# CPR Algo 3 use). Env-overridable like the NIFTY constants above.
BANKNIFTY_INDEX_SECURITY_ID = _env_int("BANKNIFTY_INDEX_SECURITY_ID", 25)
BANKNIFTY_INDEX_EXCHANGE_SEGMENT = _env_str("BANKNIFTY_INDEX_EXCHANGE_SEGMENT", "IDX_I")
BANKNIFTY_INDEX_INSTRUMENT_TYPE = _env_str("BANKNIFTY_INDEX_INSTRUMENT_TYPE", "INDEX")
OPTION_EXCHANGE_SEGMENT = _env_str("OPTION_EXCHANGE_SEGMENT", "NSE_FNO")
# Instrument type for index OPTIONS in DhanHQ's intraday/historical APIs. Used to
# pull 1-minute OHLC for a specific option strike (e.g. CPR Algo 3's observation
# legs), exactly like the index fetch but on the NSE_FNO segment.
OPTION_INSTRUMENT_TYPE = _env_str("OPTION_INSTRUMENT_TYPE", "OPTIDX")

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
RENKO_NO_TRADE_START_MINUTE = _env_int("RENKO_NO_TRADE_START_MINUTE", 30)
RENKO_NO_TRADE_END_HOUR = _env_int("RENKO_NO_TRADE_END_HOUR", 12)
RENKO_NO_TRADE_END_MINUTE = _env_int("RENKO_NO_TRADE_END_MINUTE", 30)


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
# CPR STRATEGY CONSTANTS (Central Pivot Range, 5-min via internal resample)
# =============================================================================
CPR_LOTS = _env_int("CPR_LOTS", 1)
CPR_MAX_LOSS = _env_float("CPR_MAX_LOSS", 5500.0)
CPR_POLL_SECONDS = _env_int("CPR_POLL_SECONDS", 5)

# CPR enters from 9:25 (after the opening range), squares off at 15:15.
CPR_TRADING_START_HOUR = _env_int("CPR_TRADING_START_HOUR", 9)
CPR_TRADING_START_MINUTE = _env_int("CPR_TRADING_START_MINUTE", 25)
CPR_SQUARE_OFF_HOUR = _env_int("CPR_SQUARE_OFF_HOUR", 15)
CPR_SQUARE_OFF_MINUTE = _env_int("CPR_SQUARE_OFF_MINUTE", 15)


# =============================================================================
# CPR ALGO 3 STRATEGY CONSTANTS (multi-instrument: spot + ITM CE + ITM PE)
# =============================================================================
# Algo 3 watches the spot AND a ~ITM CE + ~ITM PE of the CURRENT-week expiry
# (observation only). A signal still BUYS the ATM CE/PE of the next-next expiry,
# exactly like the other ATM workers, so it shares all of CPR's risk knobs.
CPR_ALGO3_LOTS = _env_int("CPR_ALGO3_LOTS", 1)
CPR_ALGO3_MAX_LOSS = _env_float("CPR_ALGO3_MAX_LOSS", 5500.0)
CPR_ALGO3_POLL_SECONDS = _env_int("CPR_ALGO3_POLL_SECONDS", 5)
CPR_ALGO3_TRADING_START_HOUR = _env_int("CPR_ALGO3_TRADING_START_HOUR", 9)
CPR_ALGO3_TRADING_START_MINUTE = _env_int("CPR_ALGO3_TRADING_START_MINUTE", 25)
CPR_ALGO3_SQUARE_OFF_HOUR = _env_int("CPR_ALGO3_SQUARE_OFF_HOUR", 15)
CPR_ALGO3_SQUARE_OFF_MINUTE = _env_int("CPR_ALGO3_SQUARE_OFF_MINUTE", 15)
# How deep ITM (in points) the two observation options sit. The PDF uses ~100
# points ITM (stepping past the 50-point strike), so CE strike = ATM - offset and
# PE strike = ATM + offset, rounded to the listed strike step.
CPR_ALGO3_ITM_OFFSET = _env_float("CPR_ALGO3_ITM_OFFSET", 100.0)


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
# GOLDMINE STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Goldmine is a 5-minute SMA20/SMA200 trend + pullback + engulfing strategy.
# Timing/timeframe knobs live here; sizing/risk and the indicator config object
# live further down (also env-driven).
GOLDMINE_POLL_SECONDS = _env_int("GOLDMINE_POLL_SECONDS", 5)
# The shared 1-min OHLC is locally resampled into N-minute candles before the
# Goldmine indicator pipeline runs.
GOLDMINE_DERIVED_TIMEFRAME_MINUTES = _env_int("GOLDMINE_DERIVED_TIMEFRAME_MINUTES", 5)

GOLDMINE_TRADING_START_HOUR = _env_int("GOLDMINE_TRADING_START_HOUR", 9)
GOLDMINE_TRADING_START_MINUTE = _env_int("GOLDMINE_TRADING_START_MINUTE", 25)
GOLDMINE_SQUARE_OFF_HOUR = _env_int("GOLDMINE_SQUARE_OFF_HOUR", 15)
GOLDMINE_SQUARE_OFF_MINUTE = _env_int("GOLDMINE_SQUARE_OFF_MINUTE", 15)


# =============================================================================
# MONEY MACHINE STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Money Machine is a 5-minute SMA20/SMA200 trend + compression + Marubozu
# breakout strategy. Same knob layout as Goldmine.
MONEY_MACHINE_POLL_SECONDS = _env_int("MONEY_MACHINE_POLL_SECONDS", 5)
MONEY_MACHINE_DERIVED_TIMEFRAME_MINUTES = _env_int("MONEY_MACHINE_DERIVED_TIMEFRAME_MINUTES", 5)

MONEY_MACHINE_TRADING_START_HOUR = _env_int("MONEY_MACHINE_TRADING_START_HOUR", 9)
MONEY_MACHINE_TRADING_START_MINUTE = _env_int("MONEY_MACHINE_TRADING_START_MINUTE", 25)
MONEY_MACHINE_SQUARE_OFF_HOUR = _env_int("MONEY_MACHINE_SQUARE_OFF_HOUR", 15)
MONEY_MACHINE_SQUARE_OFF_MINUTE = _env_int("MONEY_MACHINE_SQUARE_OFF_MINUTE", 15)


# =============================================================================
# OPENING STRIKE PCR VWAP ATR STRATEGY CONSTANTS (Tier 3)
# =============================================================================
# Sizing/risk + trading window. Indicator periods and PCR thresholds are
# tuned in `OPENING_STRIKE_STRATEGY_CONFIG` further down (also env-driven).
OPENING_STRIKE_LOTS = _env_int("OPENING_STRIKE_LOTS", 1)
OPENING_STRIKE_MAX_LOSS = _env_float("OPENING_STRIKE_MAX_LOSS", 5500.0)
OPENING_STRIKE_POLL_SECONDS = _env_int("OPENING_STRIKE_POLL_SECONDS", 10)

# The strategy uses VWAP/ATR on the latest closed candle of the chosen
# derived timeframe. 5-min is the common default for opening-range / PCR
# style intraday setups in Indian markets, so the 1-min source data is
# resampled into 5-min candles before evaluation.
OPENING_STRIKE_DERIVED_TIMEFRAME_MINUTES = _env_int("OPENING_STRIKE_DERIVED_TIMEFRAME_MINUTES", 5)

# The strategy needs the opening strike fixed at session start; we wait
# until ~10 minutes after market open so the first 1-2 derived candles
# have closed before any signal fires.
OPENING_STRIKE_TRADING_START_HOUR = _env_int("OPENING_STRIKE_TRADING_START_HOUR", 9)
OPENING_STRIKE_TRADING_START_MINUTE = _env_int("OPENING_STRIKE_TRADING_START_MINUTE", 25)
OPENING_STRIKE_SQUARE_OFF_HOUR = _env_int("OPENING_STRIKE_SQUARE_OFF_HOUR", 15)
OPENING_STRIKE_SQUARE_OFF_MINUTE = _env_int("OPENING_STRIKE_SQUARE_OFF_MINUTE", 15)

# No new trades from a setup printed at or after this time. With a default
# 5-min timeframe, 15:00 still leaves three candles of runway before the
# 15:15 square-off, which is the bare minimum for a 1R move on the option.
OPENING_STRIKE_LAST_SETUP_HOUR = _env_int("OPENING_STRIKE_LAST_SETUP_HOUR", 15)
OPENING_STRIKE_LAST_SETUP_MINUTE = _env_int("OPENING_STRIKE_LAST_SETUP_MINUTE", 0)

# How often the worker is allowed to call the (rate-limited) option_chain
# endpoint when the strategy is flat. The standard run loop already
# signature-gates per derived candle (~once every 5 min) so 30s is a
# conservative floor that double-protects the API budget during fast
# intra-candle re-evaluations.
OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS = _env_int(
    "OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS", 30
)


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


# =============================================================================
# LONG STRANGLE (AlgoTest port) - time-based, dual-leg, BUY-only
# =============================================================================
# Strategy in one paragraph (full notes live in Dependencies/env.example):
#   At STRANGLE_ENTRY_*:* (default 09:30) we BUY one lot of the OTM1 weekly CE
#   AND one lot of the OTM1 weekly PE - a "long strangle" that profits from a
#   big intraday move in EITHER direction and bleeds slowly on a quiet day.
#   The two legs are managed INDEPENDENTLY (AlgoTest "Square Off: Partial"):
#   each has its own 5% premium stop-loss with a "1:1" trailing stop (every +1%
#   the premium rises, the stop trails up by 1% of the entry premium). Anything
#   still open is force-closed at STRANGLE_SQUARE_OFF_*:* (default 15:15).
#   Entry is purely time-based (no indicator / signal generator), so the worker
#   holds the entire rule set itself - mirroring Delta20HedgedSpreadWorker.
#   Phase 2 (not yet implemented) will add momentum re-entry after a stop-out.
STRANGLE_LOTS = _env_int("STRANGLE_LOTS", 1)
# Strategy-wide safety backstop in INR across BOTH legs. The AlgoTest config
# had the overall stop-loss switched OFF, but the live runner always keeps a
# hard cap so a pathological day cannot run unbounded. Set 0 to disable (the
# per-leg stop and the time cutoff still apply).
STRANGLE_MAX_LOSS = _env_float("STRANGLE_MAX_LOSS", 10000.0)
STRANGLE_POLL_SECONDS = _env_int("STRANGLE_POLL_SECONDS", 2)

# Entry time (both legs open here, once) and the daily force-close cutoff.
STRANGLE_ENTRY_HOUR = _env_int("STRANGLE_ENTRY_HOUR", 9)
STRANGLE_ENTRY_MINUTE = _env_int("STRANGLE_ENTRY_MINUTE", 30)
STRANGLE_SQUARE_OFF_HOUR = _env_int("STRANGLE_SQUARE_OFF_HOUR", 15)
STRANGLE_SQUARE_OFF_MINUTE = _env_int("STRANGLE_SQUARE_OFF_MINUTE", 15)

# How many strikes out-of-the-money each leg is. 1 = OTM1 (CE = ATM+1 step,
# PE = ATM-1 step), where one step = ATM_STRIKE_STEP (50 for NIFTY).
STRANGLE_OTM_STEPS = _env_int("STRANGLE_OTM_STEPS", 1)

# Per-leg stop-loss as a fraction of the entry premium (0.05 = 5%). A leg
# exits when its premium drops to <= entry * (1 - STRANGLE_SL_PCT)...
STRANGLE_SL_PCT = _env_float("STRANGLE_SL_PCT", 0.05)
# ...but the stop TRAILS up as the premium rises. "1:1" = for every
# STRANGLE_TRAIL_TRIGGER_PCT (%) the premium gains over entry, lift the stop by
# STRANGLE_TRAIL_STEP_PCT (%) of the entry premium. With both at 1.0 the stop
# sits a constant 5% under the high-water premium once the trade is in profit.
STRANGLE_TRAIL_TRIGGER_PCT = _env_float("STRANGLE_TRAIL_TRIGGER_PCT", 1.0)
STRANGLE_TRAIL_STEP_PCT = _env_float("STRANGLE_TRAIL_STEP_PCT", 1.0)

# Momentum re-entry (AlgoTest "RE MOMENTUM"). After a leg is stopped out we do
# NOT re-buy immediately; we keep watching the SAME strike and re-enter only
# once its premium rebounds STRANGLE_REENTRY_MOMENTUM_PCT (default 5%) above the
# stop-out price - confirmation the move resumed in the buyer's favour. Repeats
# up to STRANGLE_MAX_REENTRIES (default 10) times per leg per day. Enabled by
# default because the source AlgoTest strategy had it on; set 0/false to make a
# stopped-out leg stay flat for the rest of the day.
STRANGLE_REENTRY_ENABLED = _env_bool("STRANGLE_REENTRY_ENABLED", True)
STRANGLE_REENTRY_MOMENTUM_PCT = _env_float("STRANGLE_REENTRY_MOMENTUM_PCT", 0.05)
STRANGLE_MAX_REENTRIES = _env_int("STRANGLE_MAX_REENTRIES", 10)


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
    ROOT_DIR / "Signal Generators" / "renko_strategy_logic_9_21.py",
)
EMA_LOGIC = load_module(
    "master_ema_trend_strategy_logic",
    ROOT_DIR / "Signal Generators" / "ema_trend_strategy_logic.py",
)
HEIKIN_LOGIC = load_module(
    "master_heikin_ashi_strategy_logic",
    ROOT_DIR / "Signal Generators" / "heikin_ashi_strategy_logic.py",
)
PROFIT_SHOOTER_LOGIC = load_module(
    "master_profit_shooter_strategy_logic",
    ROOT_DIR / "Signal Generators" / "Subhamoy Strategies" / "profit_shooter_strategy_logic.py",
)
GOLDMINE_LOGIC = load_module(
    "master_goldmine_strategy_logic",
    ROOT_DIR / "Signal Generators" / "Subhamoy Strategies" / "goldmine_strategy_logic.py",
)
MONEY_MACHINE_LOGIC = load_module(
    "master_money_machine_strategy_logic",
    ROOT_DIR / "Signal Generators" / "Subhamoy Strategies" / "money_machine_strategy_logic.py",
)
SUPERTREND_LOGIC = load_module(
    "master_supertrend_signal_generator_bullish",
    ROOT_DIR / "Signal Generators" / "Supertrend Signal Generator Bullish.py",
)
DONCHIAN_BEARISH_LOGIC = load_module(
    "master_donchian_signal_generator_bearish",
    ROOT_DIR / "Signal Generators" / "Donchian Signal Generator Bearish.py",
)
OPENING_STRIKE_LOGIC = load_module(
    "master_nifty_opening_strike_pcr_vwap_atr_signal_generator",
    ROOT_DIR / "Signal Generators" / "Nifty Opening Strike PCR VWAP ATR Signal Generator.py",
)
CPR_LOGIC = load_module(
    "master_cpr_strategy_logic",
    ROOT_DIR / "Signal Generators" / "CPR Strategy" / "cpr_strategy_logic.py",
)
# CPR Algo 3 (multi-instrument) reuses the engine above via `from cpr_strategy_logic
# import ...`. Alias the already-loaded engine under that bare name so loading the
# Algo 3 file binds to THIS instance instead of importing a second copy of it.
sys.modules.setdefault("cpr_strategy_logic", CPR_LOGIC)
CPR_ALGO3_LOGIC = load_module(
    "master_cpr_algo3_signal_generator",
    ROOT_DIR / "Signal Generators" / "CPR Strategy" / "Nifty CPR Algo 3 Signal Generator.py",
)

# -----------------------------------------------------------------------------
# Live-execution layer (optional) - Kotak Neo, Shoonya, or Flattrade.
# -----------------------------------------------------------------------------
# This is how real (non-paper) orders reach the broker. All three helpers expose
# the SAME surface (ensure_logged_in /
# preload_scrip_master / resolve_option_symbol / place_market_order /
# extract_order_id / logout / is_logged_in), so the runner uses whichever one
# LIVE_BROKER selects via a single generic `execution_client`. Each import is
# wrapped in try/except on purpose: if a broker's SDK/deps are missing, that
# client is set to None and the runner keeps working (the OTHER broker, or
# paper-only). main() forces any "live" strategy back to paper when the selected
# client is None.
try:
    _kotak_execution_module = load_module(
        "master_kotak_execution",
        Path(__file__).resolve().parent / "Dependencies" / "Kotak API" / "kotak_execution.py",
    )
    kotak_execution_client = _kotak_execution_module.kotak_execution_client
except Exception as _kotak_import_exc:  # ImportError when SDK folder/deps missing
    kotak_execution_client = None
    logging.getLogger(LOGGER_NAME).warning(
        "Kotak execution layer unavailable (%s); Kotak live trading disabled.",
        _kotak_import_exc,
    )

try:
    _shoonya_execution_module = load_module(
        "master_shoonya_execution",
        Path(__file__).resolve().parent / "Dependencies" / "Shoonya API" / "shoonya_execution.py",
    )
    shoonya_execution_client = _shoonya_execution_module.shoonya_execution_client
except Exception as _shoonya_import_exc:  # ImportError when SDK folder/deps missing
    shoonya_execution_client = None
    logging.getLogger(LOGGER_NAME).warning(
        "Shoonya execution layer unavailable (%s); Shoonya live trading disabled.",
        _shoonya_import_exc,
    )

try:
    _flattrade_execution_module = load_module(
        "master_flattrade_execution",
        Path(__file__).resolve().parent / "Dependencies" / "Flattrade API" / "flattrade_execution.py",
    )
    flattrade_execution_client = _flattrade_execution_module.flattrade_execution_client
except Exception as _flattrade_import_exc:
    flattrade_execution_client = None
    logging.getLogger(LOGGER_NAME).warning(
        "Flattrade execution layer unavailable (%s); Flattrade live trading disabled.",
        _flattrade_import_exc,
    )


def _select_execution_client(broker_name: str):
    """Return client, exchange, and product for one explicit broker selection.

    Keeping this decision in one small function makes the fail-closed behaviour
    easy to test.  A typo must never route real-money orders to another broker.
    """
    broker = str(broker_name).upper().strip()
    if broker == "KOTAK":
        product = _env_str("KOTAK_PRODUCT_TYPE", "INTRADAY").upper().strip() or "INTRADAY"
        return kotak_execution_client, "nse_fo", product
    if broker == "SHOONYA":
        product = _env_str("SHOONYA_PRODUCT_TYPE", "INTRADAY").upper().strip() or "INTRADAY"
        return shoonya_execution_client, "NFO", product
    if broker == "FLATTRADE":
        product = _env_str("FLATTRADE_PRODUCT_TYPE", "INTRADAY").upper().strip() or "INTRADAY"
        return flattrade_execution_client, "NFO", product
    logging.getLogger(LOGGER_NAME).error(
        "Unknown LIVE_BROKER=%r (expected KOTAK, SHOONYA, or FLATTRADE); "
        "live trading DISABLED (paper only).",
        broker_name,
    )
    return None, "", "INTRADAY"


# Pick the active broker from .env (default KOTAK). The rest of the runner only
# touches these three generic values. INTRADAY is same-day; NORMAL is carry-forward.
LIVE_BROKER = _env_str("LIVE_BROKER", "KOTAK").upper().strip() or "KOTAK"
execution_client, LIVE_EXCHANGE_SEGMENT, LIVE_PRODUCT_TYPE = _select_execution_client(
    LIVE_BROKER
)

# =============================================================================
# SIGNAL GENERATOR PORTS (ATM single-leg strategies from the TradingBot repo)
# =============================================================================
# Thirteen extra ATM single-leg strategies re-implemented from the public
# TradingBot project, kept alongside the other strategies under Signal Generators/.
# They are the SAME execution family as Renko/Goldmine/CPR (a LONG buys the ATM
# CE, a SHORT the ATM PE of the next-next expiry); each is namespaced by its own
# name so every knob is independently tunable from .env, exactly like the
# strategies above. (Pairs Trading is excluded - it needs two instruments;
# ML Ensemble requires scikit-learn.)
SIGNAL_GEN_DIR = ROOT_DIR / "Signal Generators"

SMA_CROSSOVER_LOGIC = load_module(
    "master_sma_crossover", SIGNAL_GEN_DIR / "Nifty SMA Crossover Signal Generator.py"
)
BOLLINGER_BANDS_LOGIC = load_module(
    "master_bollinger_bands", SIGNAL_GEN_DIR / "Nifty Bollinger Bands Signal Generator.py"
)
KELTNER_SQUEEZE_LOGIC = load_module(
    "master_keltner_squeeze", SIGNAL_GEN_DIR / "Nifty Keltner Squeeze Signal Generator.py"
)
MEAN_REVERSION_ZSCORE_LOGIC = load_module(
    "master_mean_reversion_zscore", SIGNAL_GEN_DIR / "Nifty Mean Reversion Zscore Signal Generator.py"
)
ML_ENSEMBLE_LOGIC = load_module(
    "master_ml_ensemble", SIGNAL_GEN_DIR / "Nifty ML Ensemble Signal Generator.py"
)
MULTI_TIMEFRAME_LOGIC = load_module(
    "master_multi_timeframe", SIGNAL_GEN_DIR / "Nifty Multi Timeframe Signal Generator.py"
)
OPENING_RANGE_BREAKOUT_LOGIC = load_module(
    "master_opening_range_breakout", SIGNAL_GEN_DIR / "Nifty Opening Range Breakout Signal Generator.py"
)
PARABOLIC_SAR_LOGIC = load_module(
    "master_parabolic_sar", SIGNAL_GEN_DIR / "Nifty Parabolic SAR Signal Generator.py"
)
RSI_DIVERGENCE_LOGIC = load_module(
    "master_rsi_divergence", SIGNAL_GEN_DIR / "Nifty RSI Divergence Signal Generator.py"
)
RSI_REVERSAL_LOGIC = load_module(
    "master_rsi_reversal", SIGNAL_GEN_DIR / "Nifty RSI Reversal Signal Generator.py"
)
STOCHASTIC_LOGIC = load_module(
    "master_stochastic_oscillator", SIGNAL_GEN_DIR / "Nifty Stochastic Oscillator Signal Generator.py"
)
SUPERTREND_PORT_LOGIC = load_module(
    "master_supertrend_port", SIGNAL_GEN_DIR / "Nifty Supertrend Signal Generator.py"
)
VOLATILITY_BREAKOUT_LOGIC = load_module(
    "master_volatility_breakout", SIGNAL_GEN_DIR / "Nifty Volatility Breakout Signal Generator.py"
)

# =============================================================================
# SL HUNTING AI AGENT (optional, LLM-driven strategy)
# =============================================================================
# A Claude-Agent-SDK strategy that trades the discretionary "SL Hunting" price-
# action method on NIFTY ATM options. It is OPT-IN via SL_HUNTING_ENABLED and an
# OPTIONAL component: its extra deps (claude-agent-sdk, pydantic) are imported
# here behind try/except exactly like the broker layer above, so a missing dep
# just disables this one worker and never breaks the rest of the runner or the
# unittest suite. The heavy Claude Agent SDK is imported lazily at decision time
# (inside the agent), so only `pydantic` must be importable for the worker to load.
#
# The folder has cross-importing modules all prefixed `sl_hunting_*`, so we add it
# to sys.path once and import by name (cleaner than load_module for a multi-file
# folder; the unique prefix avoids any sys.modules collision).
SL_HUNTING_ENABLED = _env_bool("SL_HUNTING_ENABLED", False)
SL_HUNTING_DIR = SIGNAL_GEN_DIR / "SL Hunting AI Agent"
SL_HUNTING_AVAILABLE = False
SL_HUNTING_AGENT_MODULE = None
SL_HUNTING_EXECUTOR_MODULE = None
SL_HUNTING_JOURNAL_MODULE = None
SL_HUNTING_LESSONS_MODULE = None
SL_HUNTING_INDICATOR_CONFIG = None
if SL_HUNTING_ENABLED:
    try:
        if str(SL_HUNTING_DIR) not in sys.path:
            sys.path.insert(0, str(SL_HUNTING_DIR))
        SL_HUNTING_AGENT_MODULE = importlib.import_module("sl_hunting_agent")
        SL_HUNTING_EXECUTOR_MODULE = importlib.import_module("sl_hunting_executor")
        SL_HUNTING_JOURNAL_MODULE = importlib.import_module("sl_hunting_journal")
        SL_HUNTING_LESSONS_MODULE = importlib.import_module("sl_hunting_lessons")
        SL_HUNTING_INDICATOR_CONFIG = importlib.import_module(
            "sl_hunting_indicators"
        ).SLHuntingIndicatorConfig()
        SL_HUNTING_AVAILABLE = True
        logging.getLogger(LOGGER_NAME).info(
            "SL Hunting AI Agent loaded (model=%s); add SL_HUNTING_LIVE_TRADING + "
            "LIVE_TRADING_ENABLED for real orders.",
            _env_str("SL_HUNTING_MODEL", "claude-opus-4-8"),
        )
    except Exception as _sl_hunting_import_exc:  # missing pydantic / SDK / import error
        logging.getLogger(LOGGER_NAME).warning(
            "SL Hunting AI Agent unavailable (%s); that strategy is disabled. "
            "Install its deps with `pip install claude-agent-sdk pydantic`.",
            _sl_hunting_import_exc,
        )

# Each port is fully namespaced by its own prefix (<STRATEGY>_<FIELD>), so every
# operational knob (poll/lots/timing/risk) and every indicator field is an
# independent .env override - the same pattern Goldmine/Money Machine use. The
# operational knobs are built per strategy by _signal_gen_ops() (defined next to
# the worker factory below); the indicator configs are built explicitly here so
# each field stays greppable.
SMA_CROSSOVER_CONFIG = SMA_CROSSOVER_LOGIC.SMACrossoverConfig(
    short_window=_env_int("SMA_CROSSOVER_SHORT_WINDOW", 9),
    long_window=_env_int("SMA_CROSSOVER_LONG_WINDOW", 21),
    stop_loss_pct=_env_float("SMA_CROSSOVER_STOP_LOSS_PCT", 0.015),
    target_pct=_env_float("SMA_CROSSOVER_TARGET_PCT", 0.03),
)
BOLLINGER_BANDS_CONFIG = BOLLINGER_BANDS_LOGIC.BollingerBandsConfig(
    bb_period=_env_int("BOLLINGER_BANDS_BB_PERIOD", 20),
    bb_std=_env_float("BOLLINGER_BANDS_BB_STD", 2.0),
    stop_loss_pct=_env_float("BOLLINGER_BANDS_STOP_LOSS_PCT", 0.02),
)
KELTNER_SQUEEZE_CONFIG = KELTNER_SQUEEZE_LOGIC.KeltnerSqueezeConfig(
    bb_period=_env_int("KELTNER_SQUEEZE_BB_PERIOD", 20),
    bb_std=_env_float("KELTNER_SQUEEZE_BB_STD", 2.0),
    kc_period=_env_int("KELTNER_SQUEEZE_KC_PERIOD", 20),
    kc_atr_period=_env_int("KELTNER_SQUEEZE_KC_ATR_PERIOD", 20),
    kc_multiplier=_env_float("KELTNER_SQUEEZE_KC_MULTIPLIER", 1.5),
    macd_fast=_env_int("KELTNER_SQUEEZE_MACD_FAST", 12),
    macd_slow=_env_int("KELTNER_SQUEEZE_MACD_SLOW", 26),
    macd_signal=_env_int("KELTNER_SQUEEZE_MACD_SIGNAL", 9),
    stop_loss_pct=_env_float("KELTNER_SQUEEZE_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("KELTNER_SQUEEZE_TARGET_PCT", 0.04),
)
MEAN_REVERSION_ZSCORE_CONFIG = MEAN_REVERSION_ZSCORE_LOGIC.MeanReversionZscoreConfig(
    lookback_period=_env_int("MEAN_REVERSION_ZSCORE_LOOKBACK_PERIOD", 20),
    entry_z=_env_float("MEAN_REVERSION_ZSCORE_ENTRY_Z", 2.0),
    exit_z=_env_float("MEAN_REVERSION_ZSCORE_EXIT_Z", 0.0),
    stop_loss_pct=_env_float("MEAN_REVERSION_ZSCORE_STOP_LOSS_PCT", 0.03),
)
ML_ENSEMBLE_CONFIG = ML_ENSEMBLE_LOGIC.MLEnsembleConfig(
    rsi_fast=_env_int("ML_ENSEMBLE_RSI_FAST", 7),
    rsi_slow=_env_int("ML_ENSEMBLE_RSI_SLOW", 14),
    macd_fast=_env_int("ML_ENSEMBLE_MACD_FAST", 12),
    macd_slow=_env_int("ML_ENSEMBLE_MACD_SLOW", 26),
    macd_signal=_env_int("ML_ENSEMBLE_MACD_SIGNAL", 9),
    bb_period=_env_int("ML_ENSEMBLE_BB_PERIOD", 20),
    bb_std=_env_float("ML_ENSEMBLE_BB_STD", 2.0),
    atr_period=_env_int("ML_ENSEMBLE_ATR_PERIOD", 14),
    vol_window=_env_int("ML_ENSEMBLE_VOL_WINDOW", 20),
    training_window=_env_int("ML_ENSEMBLE_TRAINING_WINDOW", 200),
    retrain_every=_env_int("ML_ENSEMBLE_RETRAIN_EVERY", 20),
    forward_bars=_env_int("ML_ENSEMBLE_FORWARD_BARS", 5),
    buy_threshold=_env_float("ML_ENSEMBLE_BUY_THRESHOLD", 0.6),
    sell_threshold=_env_float("ML_ENSEMBLE_SELL_THRESHOLD", 0.4),
    n_estimators=_env_int("ML_ENSEMBLE_N_ESTIMATORS", 100),
    max_depth=_env_int("ML_ENSEMBLE_MAX_DEPTH", 5),
    min_training_rows=_env_int("ML_ENSEMBLE_MIN_TRAINING_ROWS", 50),
    random_state=_env_int("ML_ENSEMBLE_RANDOM_STATE", 42),
    stop_loss_pct=_env_float("ML_ENSEMBLE_STOP_LOSS_PCT", 0.025),
    target_pct=_env_float("ML_ENSEMBLE_TARGET_PCT", 0.05),
)
MULTI_TIMEFRAME_CONFIG = MULTI_TIMEFRAME_LOGIC.MultiTimeframeConfig(
    trend_sma_period=_env_int("MULTI_TIMEFRAME_TREND_SMA_PERIOD", 50),
    ema_fast=_env_int("MULTI_TIMEFRAME_EMA_FAST", 9),
    ema_slow=_env_int("MULTI_TIMEFRAME_EMA_SLOW", 21),
    rsi_period=_env_int("MULTI_TIMEFRAME_RSI_PERIOD", 14),
    rsi_buy_min=_env_float("MULTI_TIMEFRAME_RSI_BUY_MIN", 40.0),
    rsi_buy_max=_env_float("MULTI_TIMEFRAME_RSI_BUY_MAX", 70.0),
    rsi_sell_min=_env_float("MULTI_TIMEFRAME_RSI_SELL_MIN", 30.0),
    rsi_sell_max=_env_float("MULTI_TIMEFRAME_RSI_SELL_MAX", 60.0),
    stop_loss_pct=_env_float("MULTI_TIMEFRAME_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("MULTI_TIMEFRAME_TARGET_PCT", 0.04),
)
OPENING_RANGE_BREAKOUT_CONFIG = OPENING_RANGE_BREAKOUT_LOGIC.OpeningRangeBreakoutConfig(
    atr_period=_env_int("OPENING_RANGE_BREAKOUT_ATR_PERIOD", 14),
    atr_multiplier=_env_float("OPENING_RANGE_BREAKOUT_ATR_MULTIPLIER", 0.3),
    stop_loss_pct=_env_float("OPENING_RANGE_BREAKOUT_STOP_LOSS_PCT", 0.015),
    target_pct=_env_float("OPENING_RANGE_BREAKOUT_TARGET_PCT", 0.03),
)
PARABOLIC_SAR_CONFIG = PARABOLIC_SAR_LOGIC.ParabolicSARConfig(
    af_start=_env_float("PARABOLIC_SAR_AF_START", 0.02),
    af_step=_env_float("PARABOLIC_SAR_AF_STEP", 0.02),
    af_max=_env_float("PARABOLIC_SAR_AF_MAX", 0.2),
    adx_period=_env_int("PARABOLIC_SAR_ADX_PERIOD", 14),
    adx_min=_env_float("PARABOLIC_SAR_ADX_MIN", 20.0),
    stop_loss_pct=_env_float("PARABOLIC_SAR_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("PARABOLIC_SAR_TARGET_PCT", 0.04),
)
RSI_DIVERGENCE_CONFIG = RSI_DIVERGENCE_LOGIC.RSIDivergenceConfig(
    rsi_period=_env_int("RSI_DIVERGENCE_RSI_PERIOD", 14),
    swing_window=_env_int("RSI_DIVERGENCE_SWING_WINDOW", 5),
    stop_loss_pct=_env_float("RSI_DIVERGENCE_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("RSI_DIVERGENCE_TARGET_PCT", 0.04),
)
RSI_REVERSAL_CONFIG = RSI_REVERSAL_LOGIC.RSIReversalConfig(
    rsi_period=_env_int("RSI_REVERSAL_RSI_PERIOD", 14),
    oversold=_env_float("RSI_REVERSAL_OVERSOLD", 30.0),
    overbought=_env_float("RSI_REVERSAL_OVERBOUGHT", 70.0),
    exit_at_mean=bool(_env_int("RSI_REVERSAL_EXIT_AT_MEAN", 1)),
    mean_level=_env_float("RSI_REVERSAL_MEAN_LEVEL", 50.0),
    stop_loss_pct=_env_float("RSI_REVERSAL_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("RSI_REVERSAL_TARGET_PCT", 0.04),
)
STOCHASTIC_CONFIG = STOCHASTIC_LOGIC.StochasticOscillatorConfig(
    k_period=_env_int("STOCHASTIC_K_PERIOD", 14),
    d_period=_env_int("STOCHASTIC_D_PERIOD", 3),
    smooth_k=_env_int("STOCHASTIC_SMOOTH_K", 3),
    oversold=_env_float("STOCHASTIC_OVERSOLD", 20.0),
    overbought=_env_float("STOCHASTIC_OVERBOUGHT", 80.0),
    zone_buffer=_env_float("STOCHASTIC_ZONE_BUFFER", 10.0),
    trend_filter_period=_env_int("STOCHASTIC_TREND_FILTER_PERIOD", 50),
    stop_loss_pct=_env_float("STOCHASTIC_STOP_LOSS_PCT", 0.015),
    target_pct=_env_float("STOCHASTIC_TARGET_PCT", 0.03),
)
SUPERTREND_PORT_CONFIG = SUPERTREND_PORT_LOGIC.SupertrendConfig(
    atr_period=_env_int("SUPERTREND_PORT_ATR_PERIOD", 10),
    multiplier=_env_float("SUPERTREND_PORT_MULTIPLIER", 3.0),
    target_pct=_env_float("SUPERTREND_PORT_TARGET_PCT", 0.0),
)
VOLATILITY_BREAKOUT_CONFIG = VOLATILITY_BREAKOUT_LOGIC.VolatilityBreakoutConfig(
    k_factor=_env_float("VOLATILITY_BREAKOUT_K_FACTOR", 0.5),
    stop_loss_pct=_env_float("VOLATILITY_BREAKOUT_STOP_LOSS_PCT", 0.02),
    target_pct=_env_float("VOLATILITY_BREAKOUT_TARGET_PCT", 0.04),
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

# CPR uses its own config dataclass. The defaults already encode the PDF's
# indicator periods and entry filters, so we keep them as-is (all three
# sub-algos -- ALGO1, ALGO2 zone, RSI divergence -- stay enabled by default).
CPR_STRATEGY_CONFIG = CPR_LOGIC.CPRStrategyConfig()

# CPR Algo 3 wraps the same indicator config (so its CPR/VWAP/RSI/ARSI match the
# other CPR algos) plus the PDF's 45/60 RSI-vs-ARSI thresholds (its own defaults).
CPR_ALGO3_CONFIG = CPR_ALGO3_LOGIC.CPRAlgo3Config()

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
# Per-trade rupee risk budget used by Profit Shooter's dynamic position sizer.
# The worker overrides the static lot count and instead picks the smallest
# whole-lot quantity whose underlying-points risk fits within this budget.
PROFIT_SHOOTER_RISK_BUDGET = _env_float("PROFIT_SHOOTER_RISK_BUDGET", 2500.0)
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

# Goldmine uses a config dataclass for its indicator/pattern tunables. Every
# field is env-driven so ops can tune behaviour without editing strategy code.
GOLDMINE_STRATEGY_CONFIG = GOLDMINE_LOGIC.GoldmineStrategyConfig(
    sma_fast_period=_env_int("GOLDMINE_SMA_FAST_PERIOD", 20),
    sma_slow_period=_env_int("GOLDMINE_SMA_SLOW_PERIOD", 200),
    atr_period=_env_int("GOLDMINE_ATR_PERIOD", 14),
    slope_lookback=_env_int("GOLDMINE_SLOPE_LOOKBACK", 3),
    pullback_lookback=_env_int("GOLDMINE_PULLBACK_LOOKBACK", 3),
    pullback_min_count=_env_int("GOLDMINE_PULLBACK_MIN_COUNT", 2),
    near_sma_atr_multiple=_env_float("GOLDMINE_NEAR_SMA_ATR_MULT", 0.5),
    engulf_tolerance=_env_float("GOLDMINE_ENGULF_TOLERANCE", 0.05),
    target_atr_multiple=_env_float("GOLDMINE_TARGET_ATR_MULT", 2.0),
    max_bars_in_trade=_env_int("GOLDMINE_MAX_BARS_IN_TRADE", 6),
)
# Operational sizing/risk for Goldmine (Profit-Shooter-style dynamic sizing).
# GOLDMINE_LOTS is only the fallback used when risk-based sizing cannot be
# computed; normal entries size off GOLDMINE_RISK_BUDGET.
GOLDMINE_LOTS = _env_int("GOLDMINE_LOTS", 1)
GOLDMINE_RISK_BUDGET = _env_float("GOLDMINE_RISK_BUDGET", 2500.0)
GOLDMINE_STARTING_CAPITAL = _env_float("GOLDMINE_STARTING_CAPITAL", 600000.0)
GOLDMINE_DAILY_MAX_LOSS_PCT = _env_float("GOLDMINE_DAILY_MAX_LOSS_PCT", 0.03)
GOLDMINE_MAX_LOSS = GOLDMINE_STARTING_CAPITAL * GOLDMINE_DAILY_MAX_LOSS_PCT

# Money Machine config dataclass. Same env-driven approach as Goldmine.
MONEY_MACHINE_STRATEGY_CONFIG = MONEY_MACHINE_LOGIC.MoneyMachineStrategyConfig(
    sma_fast_period=_env_int("MONEY_MACHINE_SMA_FAST_PERIOD", 20),
    sma_slow_period=_env_int("MONEY_MACHINE_SMA_SLOW_PERIOD", 200),
    atr_period=_env_int("MONEY_MACHINE_ATR_PERIOD", 14),
    slope_lookback=_env_int("MONEY_MACHINE_SLOPE_LOOKBACK", 3),
    near_sma_atr_multiple=_env_float("MONEY_MACHINE_NEAR_SMA_ATR_MULT", 0.5),
    compression_window=_env_int("MONEY_MACHINE_COMPRESSION_WINDOW", 3),
    compression_range_atr_multiple=_env_float("MONEY_MACHINE_COMPRESSION_RANGE_ATR_MULT", 0.75),
    marubozu_body_min_ratio=_env_float("MONEY_MACHINE_MARUBOZU_BODY_MIN_RATIO", 0.80),
    marubozu_wick_max_ratio=_env_float("MONEY_MACHINE_MARUBOZU_WICK_MAX_RATIO", 0.15),
    require_breakout_close=bool(_env_int("MONEY_MACHINE_REQUIRE_BREAKOUT_CLOSE", 1)),
    target_atr_multiple=_env_float("MONEY_MACHINE_TARGET_ATR_MULT", 2.0),
)
# Operational sizing/risk for Money Machine (same dynamic sizing as Goldmine).
MONEY_MACHINE_LOTS = _env_int("MONEY_MACHINE_LOTS", 1)
MONEY_MACHINE_RISK_BUDGET = _env_float("MONEY_MACHINE_RISK_BUDGET", 2500.0)
MONEY_MACHINE_STARTING_CAPITAL = _env_float("MONEY_MACHINE_STARTING_CAPITAL", 600000.0)
MONEY_MACHINE_DAILY_MAX_LOSS_PCT = _env_float("MONEY_MACHINE_DAILY_MAX_LOSS_PCT", 0.03)
MONEY_MACHINE_MAX_LOSS = MONEY_MACHINE_STARTING_CAPITAL * MONEY_MACHINE_DAILY_MAX_LOSS_PCT

# Settings objects for the two Hedged Puts strategies.
SUPERTREND_SETTINGS = SUPERTREND_LOGIC.SupertrendSettings(
    atr_length=SUPERTREND_ATR_LENGTH,
    factor=SUPERTREND_FACTOR,
)
DONCHIAN_BEARISH_SETTINGS = DONCHIAN_BEARISH_LOGIC.DonchianSettings(
    length=DONCHIAN_LENGTH,
)

# Opening Strike PCR/VWAP/ATR uses a frozen dataclass for its tunables.
# Same pattern as EMA / Profit Shooter -- every parameter is `_env_*`
# driven so ops can tune behaviour without editing strategy code.
OPENING_STRIKE_STRATEGY_CONFIG = OPENING_STRIKE_LOGIC.NiftyOpeningStrikePCRVWAPATRConfig(
    strike_step=int(ATM_STRIKE_STEP),
    strike_window_n=_env_int("OPENING_STRIKE_STRIKE_WINDOW_N", 3),
    pcr_bullish_threshold=_env_float("OPENING_STRIKE_PCR_BULLISH_THRESHOLD", 1.2),
    pcr_bearish_threshold=_env_float("OPENING_STRIKE_PCR_BEARISH_THRESHOLD", 0.8),
    option_moneyness="ATM",
    atr_period=_env_int("OPENING_STRIKE_ATR_PERIOD", 14),
    vwap_near_atr_multiplier=_env_float("OPENING_STRIKE_VWAP_NEAR_ATR_MULT", 0.5),
    enable_rsi_filter=bool(_env_int("OPENING_STRIKE_ENABLE_RSI_FILTER", 0)),
    rsi_period=_env_int("OPENING_STRIKE_RSI_PERIOD", 14),
    buy_rsi_min=_env_float("OPENING_STRIKE_BUY_RSI_MIN", 50.0),
    sell_rsi_max=_env_float("OPENING_STRIKE_SELL_RSI_MAX", 50.0),
    initial_sl_points=_env_float("OPENING_STRIKE_INITIAL_SL_POINTS", 20.0),
    expiry_selection=_env_str("OPENING_STRIKE_EXPIRY_SELECTION", "NEXT_WEEKLY"),
    # Single entry per day matches the signal-generator default and the
    # "fix the opening strike, fire once" intent of the strategy.
    allow_multiple_entries=bool(_env_int("OPENING_STRIKE_ALLOW_MULTIPLE_ENTRIES", 0)),
)

# Minimum number of derived (e.g. 5-min) candles before the engine can
# evaluate. `_calculate_atr` needs `atr_period` samples; we add 2 bars of
# slack so the first usable signal lands AFTER warmup, not on it.
OPENING_STRIKE_MIN_BARS = int(OPENING_STRIKE_STRATEGY_CONFIG.atr_period) + 2
if OPENING_STRIKE_STRATEGY_CONFIG.enable_rsi_filter:
    OPENING_STRIKE_MIN_BARS = max(
        OPENING_STRIKE_MIN_BARS,
        int(OPENING_STRIKE_STRATEGY_CONFIG.rsi_period) + 2,
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
    Runtime state for ONE single-leg paper trade (used by the eight ATM
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
    # Goldmine-only: completed-candle counter that drives its time exit. Other
    # strategies leave this at 0.
    bars_in_trade: int = 0
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
    and the consumer threads (the eleven workers). One `threading.Lock` guards
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
        resp = self.dhan.option_chain(
            under_security_id=int(under_security_id),
            under_exchange_segment=str(under_exchange_segment),
            expiry=expiry_str,
        )
        # The SDK wraps every API response in an outer envelope of the form
        # {"status", "remarks", "data": <api_response>}. Unwrap it so callers
        # see the /optionchain shape documented above (status + data.oc)
        # rather than a doubly-nested dict where data.oc lives at data.data.
        # On SDK-level failure (network error etc.) `data` is "" not a dict,
        # so the original envelope falls through and the parser will see the
        # "failure" status and short-circuit cleanly.
        if isinstance(resp, dict) and isinstance(resp.get("data"), dict):
            return resp["data"]
        return resp


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
                                       by the eight ATM workers.
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

        exch_col = _first_existing_col(df, ["EXCH_ID"])
        seg_col = _first_existing_col(df, ["SEGMENT"])
        ins_col = _first_existing_col(df, ["INSTRUMENT"])
        ts_col = _first_existing_col(df, ["SYMBOL_NAME"])
        cs_col = _first_existing_col(df, ["DISPLAY_NAME"])
        exp_col = _first_existing_col(df, ["SM_EXPIRY_DATE"])
        lot_col = _first_existing_col(df, ["LOT_SIZE"])
        sec_col = _first_existing_col(df, ["SECURITY_ID"])
        strike_col = _first_existing_col(df, ["STRIKE_PRICE"])
        opt_type_col = _first_existing_col(df, ["OPTION_TYPE"])
        sm_col = _first_existing_col(df, ["UNDERLYING_SYMBOL"])

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

        Used by the eight ATM workers (Renko, EMA, Heikin Ashi, Profit Shooter,
        Goldmine, Money Machine, Opening Strike, CPR).

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
    # ATM strike rule (used by the 8 ATM workers)
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

    def get_otm_option(
        self,
        spot_price: float,
        right: str,
        otm_steps: int = 1,
        expiry: Optional[date] = None,
    ) -> dict:
        """
        Resolve an OUT-OF-THE-MONEY option row, `otm_steps` strikes from ATM.

        Used by the Long Strangle worker, which buys the OTM1 CE and OTM1 PE.

        Right + OTM direction:
        - "CE" -> a call ABOVE the money: target = ATM + otm_steps * step
        - "PE" -> a put  BELOW the money: target = ATM - otm_steps * step
          where `step` is ATM_STRIKE_STEP (50 for NIFTY) and ATM is the spot
          rounded to the nearest step (same rounding as `get_atm_option`).
        `otm_steps=0` degenerates to the ATM strike for that right.

        Expiry:
        - Defaults to the CURRENT-WEEK expiry (the strangle trades weeklies),
          matching the hedged-puts family. Pass `expiry` to override.

        Strike pick:
        - If the exact target strike is listed we take it; otherwise the
          closest available strike (smallest absolute difference, lower strike
          breaking ties) - identical to `get_atm_option`.
        """
        right_upper = str(right).strip().upper()
        if right_upper not in ("CE", "PE"):
            raise ValueError(f"Unsupported option right for OTM resolution: {right!r}")

        spot = _safe_float(spot_price, 0.0)
        if spot <= 0:
            raise ValueError(f"Invalid spot price for OTM resolution: {spot_price!r}")

        steps = int(otm_steps)
        if steps < 0:
            raise ValueError(f"otm_steps must be >= 0, got {otm_steps!r}")

        options = self._load_option_chain()
        target_expiry = expiry if expiry is not None else self.get_current_week_expiry()

        atm_strike = round(spot / ATM_STRIKE_STEP) * ATM_STRIKE_STEP
        if right_upper == "CE":
            target_strike = atm_strike + steps * ATM_STRIKE_STEP
        else:
            target_strike = atm_strike - steps * ATM_STRIKE_STEP

        subset = options[
            (options["expiry"] == target_expiry) & (options["option_type"] == right_upper)
        ].copy()
        if subset.empty:
            raise ValueError(
                f"No {right_upper} rows found for {self.underlying} expiry {target_expiry}."
            )

        subset["strike_diff"] = (subset["strike"] - target_strike).abs()
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
            "option_type": right_upper,
            "expiry_date": expiry_date,
            "days_to_expiry": days_to_expiry,
            "lot_size": _to_int_safe(row["lot_size"], 0),
            "spot_reference": float(spot),
            "atm_strike_rounded": float(atm_strike),
            "target_strike": float(target_strike),
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
       bullish, CE legs from bearish, ATM CE/PE from the eight ATM workers).
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

        # Optional trade-event sink (a queue.Queue) consumed by the Telegram
        # notifier. main() injects it after construction; it stays None when
        # notifications are disabled, in which case publish_trade_event no-ops.
        self.trade_event_queue = None

        # Per-strategy execution mode. Defaults to paper; main() flips this to
        # True after construction when LIVE_TRADING_ENABLED and this strategy's
        # <PREFIX>_LIVE_TRADING are both on. When True, the take-trade logic places
        # real orders on the active broker (see `_place_real_leg`) in addition to
        # the usual paper bookkeeping.
        self.live_trading = False

    def _place_real_leg(self, side: str, leg: dict) -> bool:
        """
        Send ONE real buy/sell order to the active broker for a single option ("leg").

        This is the bridge between "paper" and "real": in paper mode it does
        nothing and just returns True, so the rest of the trade code can call it
        without caring which mode we're in. In live mode it actually places the
        order via the selected `execution_client` (Kotak, Shoonya, or Flattrade). A "leg" is
        one option contract (a spread has two legs).

        `leg` is a dict describing the option:
            {"option_type": "CE"/"PE", "strike": float, "expiry": date,
             "quantity": int, "dhan_symbol": str}
        The broker order symbol is built/resolved by the client's
        `resolve_option_symbol` from the contract details.

        Returns True on success. When this worker is not in live mode this is a
        no-op that returns True, so every call site can invoke it unconditionally.

        Failure policy: on any error (incl. an unresolvable symbol) we log a
        warning and return False WITHOUT raising. Callers then fall back to paper
        bookkeeping (the position is still recorded/closed as a paper trade) per
        the user's chosen policy.
        """
        if not self.live_trading:
            return True
        quantity = leg["quantity"]
        dhan_symbol = leg.get("dhan_symbol", "")
        if execution_client is None:
            self.log.warning(
                "REAL ORDER SKIPPED (no %s client) | %s %s qty=%s dhan=%s | falling back to paper.",
                LIVE_BROKER, self.strategy_name, side, quantity, dhan_symbol,
            )
            return False
        # Resolve the broker's order symbol for this contract before ordering.
        broker_symbol = execution_client.resolve_option_symbol(
            underlying=UNDERLYING,
            expiry=leg.get("expiry"),
            option_type=leg.get("option_type", ""),
            strike=leg.get("strike", 0.0),
        )
        if not broker_symbol:
            self.log.warning(
                "REAL ORDER SKIPPED (%s symbol not found) | %s %s strike=%s %s dhan=%s | "
                "falling back to paper.",
                LIVE_BROKER, self.strategy_name, side, leg.get("strike"),
                leg.get("option_type"), dhan_symbol,
            )
            return False
        try:
            resp = execution_client.place_market_order(
                symbol=broker_symbol,
                side=side,
                quantity=quantity,
                exchange_segment=LIVE_EXCHANGE_SEGMENT,
                product_type=LIVE_PRODUCT_TYPE,
            )
            self.log.info(
                "REAL ORDER OK | %s %s %s qty=%s sym=%s (dhan=%s) | OrderId=%s",
                LIVE_BROKER, self.strategy_name, side, quantity, broker_symbol, dhan_symbol,
                execution_client.extract_order_id(resp),
            )
            return True
        except Exception as exc:
            self.log.warning(
                "REAL ORDER FAILED (falling back to paper) | %s %s %s qty=%s sym=%s (dhan=%s) | %s",
                LIVE_BROKER, self.strategy_name, side, quantity, broker_symbol, dhan_symbol, exc,
            )
            return False

    def _exec_mode_tag(self, real_ok: bool) -> str:
        """
        Return a short label describing how a trade was actually executed, used in
        logs and Telegram messages so you can tell real fills from paper ones:
        - "PAPER"          : strategy is in paper mode.
        - "LIVE"           : live mode and the real order(s) succeeded.
        - "PAPER_FALLBACK" : live mode but the real order failed, so it was
                             recorded/closed as paper instead.
        """
        if not self.live_trading:
            return "PAPER"
        return "LIVE" if real_ok else "PAPER_FALLBACK"

    def _place_real_hedged_entry(self, main_leg: dict, hedge_leg: dict) -> bool:
        """
        Open a real two-leg "hedged" position (sell one option, buy a cheaper one
        for protection). Each leg is a dict like in `_place_real_leg`.

        We BUY the protective hedge FIRST, then SELL the main leg. Why that order?
        If only one leg fills, being left holding a bought option (defined, small
        risk) is far safer than being left with a naked short (unlimited risk).
        Returns True only if BOTH legs filled. If just the hedge filled, we close
        it back out and return False so the caller treats the whole thing as paper.
        No-op returning True in paper mode.
        """
        if not self.live_trading:
            return True
        if not self._place_real_leg("BUY", hedge_leg):
            return False
        if not self._place_real_leg("SELL", main_leg):
            # Main leg failed after the hedge filled -> unwind the hedge so we
            # aren't left holding a stray live leg, then fall back to paper.
            unwound = self._place_real_leg("SELL", hedge_leg)
            if unwound:
                self.log.warning(
                    "Hedged entry PARTIAL fill (hedge filled, main failed) for %s; "
                    "hedge unwound, falling back to paper.",
                    self.strategy_name,
                )
            else:
                # Double failure: the hedge BUY filled but neither the main SELL
                # nor the unwind SELL did. A LIVE long leg is open and untracked.
                self.log.error(
                    "MANUAL ACTION NEEDED | %s hedged entry: hedge filled but main "
                    "AND unwind failed -> a LIVE BUY %s strike=%s qty=%s is OPEN and "
                    "untracked. Square it off manually.",
                    self.strategy_name, hedge_leg.get("option_type"),
                    hedge_leg.get("strike"), hedge_leg.get("quantity"),
                )
                self.publish_trade_event({
                    "action": "UNHEDGED_LEG_OPEN",
                    "mode": "LIVE",
                    "leg": "hedge",
                    "option_type": hedge_leg.get("option_type"),
                    "strike": hedge_leg.get("strike"),
                    "quantity": hedge_leg.get("quantity"),
                })
            return False
        return True

    def _place_real_hedged_exit(self, main_leg: dict, hedge_leg: dict) -> bool:
        """
        Close a real two-leg hedged position: BUY back the main leg we sold, and
        SELL the hedge leg we bought ("BUY-to-close" / "SELL-to-close").

        We try BOTH legs no matter what, to close as much as possible. The caller
        always flattens its own paper books afterward either way. Returns True
        only if both legs closed cleanly. No-op returning True in paper mode.
        """
        if not self.live_trading:
            return True
        main_ok = self._place_real_leg("BUY", main_leg)
        hedge_ok = self._place_real_leg("SELL", hedge_leg)
        return main_ok and hedge_ok

    def publish_trade_event(self, event: dict) -> None:
        """
        Best-effort hand-off of a trade event to the Telegram notifier queue.

        Used by every worker (single-leg and hedged) on each entry/exit. It
        never raises and never blocks: trading must continue even if the queue
        is absent (notifications off), full, or otherwise unhappy. The consumer
        (TelegramMessageWorker) owns all formatting and network I/O, so the
        trading threads are never exposed to Telegram latency or failures.
        """
        event_queue = getattr(self, "trade_event_queue", None)
        if event_queue is None:
            return
        try:
            event.setdefault("strategy", self.strategy_name)
            event.setdefault("ts", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            event.setdefault("mode", "PAPER")
            event_queue.put_nowait(event)
        except Exception:
            # A failed enqueue must never disturb the trading loop.
            pass

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
    Concrete intermediate base for the eight ATM strategies.

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

    def _compute_entry_lots(
        self,
        entry_underlying: float,
        stop_underlying: float,
        lot_size: int,
    ) -> int:
        """
        Number of lots to use for this entry.

        Default: the static class-level `self.lots`. Subclasses (e.g. Profit
        Shooter) override this to size dynamically off the underlying SL.
        """
        return self.lots

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

        lots_for_entry = self._compute_entry_lots(
            float(entry_underlying), float(stop_underlying), lot_size
        )
        quantity = lot_size * lots_for_entry
        entry_side = "BUY"  # Both LONG (CE) and SHORT (PE) open as BUY legs.
        order_id = self._next_paper_order_id(entry_side)

        # Real execution (live mode only; no-op returning True in paper mode).
        # On failure we fall back to paper: the position is still recorded below
        # so the strategy keeps tracking it as a paper trade.
        real_ok = self._place_real_leg(entry_side, {
            "option_type": option_right, "strike": option_strike,
            "expiry": expiry_date, "quantity": quantity, "dhan_symbol": trading_symbol,
        })
        exec_mode = self._exec_mode_tag(real_ok)

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
        self.publish_trade_event(
            {
                "action": "ENTRY",
                "mode": exec_mode,
                "direction": direction,
                "lots": lots_for_entry,
                "lot_size": lot_size,
                "quantity": quantity,
                "spot": spot,
                "expiry": expiry_txt,
                "legs": [
                    {
                        "symbol": trading_symbol,
                        "side": entry_side,
                        "right": option_right,
                        "strike": option_strike,
                        "entry_price": option_ltp,
                    }
                ],
            }
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

        # Real execution (live mode only; no-op True in paper mode).
        real_ok = self._place_real_leg(exit_side, {
            "option_type": closed_position.option_right, "strike": closed_position.option_strike,
            "expiry": closed_position.option_expiry, "quantity": closed_position.quantity,
            "dhan_symbol": closed_position.symbol,
        })
        exec_mode = self._exec_mode_tag(real_ok)

        # If a LIVE exit did not confirm a fill, the real broker position is still
        # open. Do NOT flatten the books - keep the position active so the worker
        # retries the exit on its next cycle, and alert for manual square-off.
        if self.live_trading and not real_ok:
            self.log.error(
                "LIVE EXIT NOT CONFIRMED | %s %s | OptionSymbol=%s | Qty=%s | Reason=%s "
                "| position kept OPEN for retry/manual square-off.",
                self.strategy_name, closed_position.direction, closed_position.symbol,
                closed_position.quantity, reason,
            )
            self.publish_trade_event({
                "action": "EXIT_FAILED",
                "mode": exec_mode,
                "direction": closed_position.direction,
                "reason": reason,
                "quantity": closed_position.quantity,
                "legs": [{
                    "symbol": closed_position.symbol, "side": exit_side,
                    "right": closed_position.option_right, "strike": closed_position.option_strike,
                }],
            })
            return

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

        self.publish_trade_event(
            {
                "action": "EXIT",
                "mode": exec_mode,
                "exit_order_failed": bool(self.live_trading and not real_ok),
                "direction": closed_position.direction,
                "reason": reason,
                "lot_size": closed_position.option_lot_size,
                "quantity": closed_position.quantity,
                "pnl": pnl,
                "expiry": expiry_txt,
                "legs": [
                    {
                        "symbol": closed_position.symbol,
                        "side": exit_side,
                        "right": closed_position.option_right,
                        "strike": closed_position.option_strike,
                        "entry_price": closed_position.entry_trade_price,
                        "exit_price": exit_trade_price,
                    }
                ],
            }
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

    def _compute_entry_lots(
        self,
        entry_underlying: float,
        stop_underlying: float,
        lot_size: int,
    ) -> int:
        """
        Profit-Shooter-only risk-based sizing.

        Pick the smallest whole-lot quantity whose worst-case underlying-points
        loss (`risk_points * lot_size`) is within `PROFIT_SHOOTER_RISK_BUDGET`.
        `math.ceil` over a positive ratio guarantees a minimum of 1 lot, so a
        trade is never sized to zero -- a setup either trades at >=1 lot or
        not at all.
        """
        risk_points = abs(float(entry_underlying) - float(stop_underlying))
        if risk_points <= 0 or lot_size <= 0:
            self.log.warning(
                "Profit Shooter dynamic sizing fell back to static lots=%s "
                "(entry=%.2f, stop=%.2f, lot_size=%s).",
                self.lots, entry_underlying, stop_underlying, lot_size,
            )
            return self.lots
        lots = math.ceil(PROFIT_SHOOTER_RISK_BUDGET / (risk_points * lot_size))
        self.log.info(
            "Profit Shooter dynamic sizing: risk_points=%.2f | lot_size=%s | "
            "risk_budget=%.2f -> lots=%s, qty=%s.",
            risk_points,
            lot_size,
            PROFIT_SHOOTER_RISK_BUDGET,
            lots,
            lots * lot_size,
        )
        return int(lots)

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
# GOLDMINE STRATEGY WORKER (1-min source, locally resampled to 5-min)
# =============================================================================
class GoldmineStrategyWorker(AtmSingleLegStrategyWorker):
    """
    Goldmine strategy. Reads 1-minute data from the shared store, locally
    resamples it into complete 5-minute candles, then runs the SMA20/SMA200
    trend + pullback + engulfing pipeline.

    Specifics:
    - Profit-Shooter-style dynamic sizing: each entry is sized so the
      worst-case underlying-points loss fits GOLDMINE_RISK_BUDGET.
    - Goldmine has a TIME exit. `GoldminePositionContext.bars_in_trade` is
      incremented once per completed 5-min candle so the engine can emit a
      TIME_EXIT after `max_bars_in_trade` bars. The resampler drops partial
      bars, so each `process_strategy_frame` call is exactly one new candle.
    """

    strategy_name = "Goldmine"
    timeframe = "1"
    poll_seconds = GOLDMINE_POLL_SECONDS
    lots = GOLDMINE_LOTS
    max_loss = GOLDMINE_MAX_LOSS
    trading_start_hour = GOLDMINE_TRADING_START_HOUR
    trading_start_minute = GOLDMINE_TRADING_START_MINUTE
    square_off_hour = GOLDMINE_SQUARE_OFF_HOUR
    square_off_minute = GOLDMINE_SQUARE_OFF_MINUTE
    derived_timeframe_minutes = GOLDMINE_DERIVED_TIMEFRAME_MINUTES

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = GOLDMINE_LOGIC.GoldmineSignalEngine(GOLDMINE_STRATEGY_CONFIG)
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
        # Goldmine needs SMA200 warm-up; the engine reports its own minimum.
        return self.signal_engine.minimum_history_bars()

    def _compute_entry_lots(
        self,
        entry_underlying: float,
        stop_underlying: float,
        lot_size: int,
    ) -> int:
        """
        Risk-based sizing (same model as Profit Shooter).

        Pick the smallest whole-lot quantity whose worst-case underlying-points
        loss (`risk_points * lot_size`) is within `GOLDMINE_RISK_BUDGET`.
        `math.ceil` over a positive ratio guarantees at least 1 lot, so a setup
        either trades at >=1 lot or not at all.
        """
        risk_points = abs(float(entry_underlying) - float(stop_underlying))
        if risk_points <= 0 or lot_size <= 0:
            self.log.warning(
                "Goldmine dynamic sizing fell back to static lots=%s "
                "(entry=%.2f, stop=%.2f, lot_size=%s).",
                self.lots, entry_underlying, stop_underlying, lot_size,
            )
            return self.lots
        lots = math.ceil(GOLDMINE_RISK_BUDGET / (risk_points * lot_size))
        self.log.info(
            "Goldmine dynamic sizing: risk_points=%.2f | lot_size=%s | "
            "risk_budget=%.2f -> lots=%s, qty=%s.",
            risk_points, lot_size, GOLDMINE_RISK_BUDGET, lots, lots * lot_size,
        )
        return int(lots)

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-min OHLC into complete 5-min candles, then attach Goldmine indicators."""
        ohlc_5m = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        return GOLDMINE_LOGIC.build_goldmine_with_indicators(ohlc_5m, GOLDMINE_STRATEGY_CONFIG)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Interpret the latest completed 5-min candle.

        Flow:
        1. In a trade -> bump the bars-in-trade counter (one new closed candle
           per call), build a position context, and honor only exits.
        2. Flat -> count signal triggers and submit fresh entries.
        """
        position_ctx = None
        if self.pos.active:
            # One process call == one new closed 5-min candle (partial bars are
            # dropped by the resampler), so a single increment mirrors the
            # backtest generator's per-candle bars_in_trade bump.
            self.pos.bars_in_trade += 1
            position_ctx = GOLDMINE_LOGIC.GoldminePositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
                target_underlying=self.pos.target_underlying,
                bars_in_trade=self.pos.bars_in_trade,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason or "SIGNAL_EXIT")
            return

        if decision.signal_triggered:
            self.signal_count += 1

        if decision.action == "ENTER_LONG":
            if self.enter_position(
                "LONG",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position(
                "SHORT",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return


# =============================================================================
# MONEY MACHINE STRATEGY WORKER (1-min source, locally resampled to 5-min)
# =============================================================================
class MoneyMachineStrategyWorker(AtmSingleLegStrategyWorker):
    """
    Money Machine strategy. Reads 1-minute data from the shared store, locally
    resamples it into complete 5-minute candles, then runs the SMA20/SMA200
    trend + compression + Marubozu-breakout pipeline.

    Specifics:
    - Profit-Shooter-style dynamic sizing off MONEY_MACHINE_RISK_BUDGET.
    - No time exit: the engine manages stop/target only, so the position
      context is just direction + entry/stop/target.
    """

    strategy_name = "MoneyMachine"
    timeframe = "1"
    poll_seconds = MONEY_MACHINE_POLL_SECONDS
    lots = MONEY_MACHINE_LOTS
    max_loss = MONEY_MACHINE_MAX_LOSS
    trading_start_hour = MONEY_MACHINE_TRADING_START_HOUR
    trading_start_minute = MONEY_MACHINE_TRADING_START_MINUTE
    square_off_hour = MONEY_MACHINE_SQUARE_OFF_HOUR
    square_off_minute = MONEY_MACHINE_SQUARE_OFF_MINUTE
    derived_timeframe_minutes = MONEY_MACHINE_DERIVED_TIMEFRAME_MINUTES

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = MONEY_MACHINE_LOGIC.MoneyMachineSignalEngine(MONEY_MACHINE_STRATEGY_CONFIG)
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
        # Money Machine needs SMA200 warm-up; the engine reports its own minimum.
        return self.signal_engine.minimum_history_bars()

    def _compute_entry_lots(
        self,
        entry_underlying: float,
        stop_underlying: float,
        lot_size: int,
    ) -> int:
        """Risk-based sizing (same model as Profit Shooter / Goldmine)."""
        risk_points = abs(float(entry_underlying) - float(stop_underlying))
        if risk_points <= 0 or lot_size <= 0:
            self.log.warning(
                "Money Machine dynamic sizing fell back to static lots=%s "
                "(entry=%.2f, stop=%.2f, lot_size=%s).",
                self.lots, entry_underlying, stop_underlying, lot_size,
            )
            return self.lots
        lots = math.ceil(MONEY_MACHINE_RISK_BUDGET / (risk_points * lot_size))
        self.log.info(
            "Money Machine dynamic sizing: risk_points=%.2f | lot_size=%s | "
            "risk_budget=%.2f -> lots=%s, qty=%s.",
            risk_points, lot_size, MONEY_MACHINE_RISK_BUDGET, lots, lots * lot_size,
        )
        return int(lots)

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-min OHLC into complete 5-min candles, then attach Money Machine indicators."""
        ohlc_5m = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        return MONEY_MACHINE_LOGIC.build_money_machine_with_indicators(ohlc_5m, MONEY_MACHINE_STRATEGY_CONFIG)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Interpret the latest completed 5-min candle.

        Flow:
        1. In a trade -> build a position context and honor only stop/target exits.
        2. Flat -> count signal triggers and submit fresh entries.
        """
        position_ctx = None
        if self.pos.active:
            position_ctx = MONEY_MACHINE_LOGIC.MoneyMachinePositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
                target_underlying=self.pos.target_underlying,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason or "SIGNAL_EXIT")
            return

        if decision.signal_triggered:
            self.signal_count += 1

        if decision.action == "ENTER_LONG":
            if self.enter_position(
                "LONG",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position(
                "SHORT",
                decision.entry_underlying,
                decision.stop_underlying,
                target_underlying=decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return


# =============================================================================
# OPENING STRIKE PCR VWAP ATR WORKER (5-min, ATM single-leg)
# =============================================================================
class OpeningStrikePCRVWAPATRWorker(AtmSingleLegStrategyWorker):
    """
    Opening Strike PCR / VWAP / ATR strategy on a 5-minute derived series.

    PLAIN-ENGLISH WALKTHROUGH (for first-time readers)
    ---------------------------------------------------
    What this strategy is trying to do, conceptually:

    1. Right after the market opens, this worker remembers ONE special
       strike: the NIFTY ATM strike based on the day's opening price. That
       single strike is called the "opening strike" and is FIXED for the
       rest of the day. Every PCR (Put/Call ratio) calculation downstream
       uses a small window of strikes centred on this anchor.

    2. The worker watches the option-chain OPEN-INTEREST (OI) "change"
       around that opening strike. OI change is just "how much new OI got
       added today" -- put OI growing fast means option WRITERS believe
       NIFTY won't fall (bullish), call OI growing fast means writers
       believe NIFTY won't rise (bearish).

    3. On every closed 5-minute candle, ask the signal engine: "Given the
       current OI flow + this candle's VWAP/ATR/price action, do we want
       to BUY_CALL, BUY_PUT, or do nothing?". The engine combines PCR,
       candle colour, VWAP position, and (optionally) RSI to answer.

    4. On a BUY_CALL signal: buy the ATM Call option. On BUY_PUT: buy the
       ATM Put. We always BUY (never sell) options here, so the worst-case
       loss per trade is bounded at the premium paid.

    5. Manage the trade with a "ratcheting" stop loss applied on the
       OPTION'S PREMIUM (NOT on NIFTY spot):
         * initial SL = entry_premium - 20 points
         * once the option moves +2R in our favour (where 1R = 20 pts),
           ratchet SL up to entry + 1R (locks in 1R profit).
         * at +3R reward, ratchet to entry + 2R, and so on.
       The "high-water" rule means SL only moves UP, never back down.

    6. At 15:15 IST sharp the worker force-closes any running trade and
       stops for the day. This happens whether or not a signal ever fired
       -- the daily session cutoff is unconditional. See "SQUARE-OFF
       SAFETY NET" below for details.

    WHY THIS WORKER IS A LITTLE DIFFERENT FROM THE OTHER ATM WORKERS
    ----------------------------------------------------------------
    Renko / EMA / HeikinAshi / ProfitShooter look at NIFTY OHLC only.
    THIS worker ALSO needs live option-chain data, because the PCR signal
    is what the engine actually keys off. So we:

    - Call `broker.fetch_option_chain` every ~30 seconds (rate-limited so
      we never exceed DhanHQ's per-second budget for that endpoint).
    - Snapshot OI per strike on the FIRST successful fetch of the session
      and treat that as our "baseline". Every later fetch computes the
      OI CHANGE as `current_oi - baseline_oi`. This is the intraday
      definition of OI change (today's flow only), which is the same view
      that opening-range / PCR traders watch in Sensibull, Opstra, etc.
    - The exit rule is on OPTION premium, not on NIFTY spot. We pass
      `stop_underlying=0.0` to the parent's `enter_position` (the parent
      records this for audit only; it does not enforce it) and run our
      own premium-based ratchet inside `_check_premium_trailing_sl_and_exit`.

    SQUARE-OFF SAFETY NET (READ THIS IF YOU ARE WORRIED ABOUT 15:15)
    ----------------------------------------------------------------
    Three different time-based guards work together to make sure we are
    flat by the end of the session no matter what state the worker is in:

    1. `square_off_hour` / `square_off_minute` (= 15:15 by default).
       The BasePaperStrategyWorker's main `run()` loop checks
       `is_after_time(self.square_off_hour, self.square_off_minute)` on
       EVERY single poll iteration. The moment we cross that wall-clock
       time, the loop runs `handle_square_off_and_stop()` which:
            (a) sets a one-shot `cutoff_handled` flag so it only runs once,
            (b) if `self.pos.active` is True -> calls `exit_position(
                "TIME_CUTOFF")` to close the open option leg at the live
                LTP and record a normal PnL,
            (c) logs the day's paper summary,
            (d) `break`s out of the run loop so the thread exits cleanly.
       Every scenario is covered:
         - In a trade at 15:15 -> the trade is exited at TIME_CUTOFF.
         - Flat at 15:15 with no signal having fired all day -> the worker
           still hits the cutoff branch, logs the summary, and stops the
           thread (so it won't waste CPU polling after market close).
         - Flat at 15:15 but holding a "we already entered once" flag on
           the signal engine -> same as above, the thread exits.

    2. `last_setup_cutoff` (= 15:00 by default). This is SOFTER than the
       hard 15:15 stop. After this time we refuse to OPEN a fresh trade,
       because a 5-min strategy with only 15 minutes of runway is very
       unlikely to make a clean R move before the hard cutoff kicks in.
       Trades already open keep running and follow the trailing SL.

    3. `max_loss` (=Rs.5500 by default). Independent of clock time. If
       realized + open PnL ever dips below `-max_loss` the base loop
       force-closes the active trade and stops the worker, just like the
       time-cutoff path. Set max_loss=0 in .env to disable this.

    EDITABLE KNOBS (all live in `.env` under "OPENING_STRIKE_*")
    -----------------------------------------------------------
    Every numeric / boolean parameter the strategy uses -- sizing, the
    trading window, PCR thresholds, ATR/RSI periods, the SL points, the
    derived timeframe, the option-chain refresh budget -- is loaded via
    `_env_*` calls at the top of this file. So you never need to edit
    this class to tune behaviour; just change the corresponding key in
    `Multithreading/Dependencies/.env` and restart the runner.
    """

    # --- Class-level attributes -------------------------------------------
    # These are read by the BasePaperStrategyWorker run loop. They have to
    # be class attributes (not instance attributes) because the base class
    # accesses them via `self.poll_seconds` etc. before any instance
    # initialization can race with the supervisor thread.

    # Short label used in log lines like "OpeningStrike ENTRY LONG | ..."
    # and in the thread name. Keep it short and unique across all workers.
    strategy_name = "OpeningStrike"

    # The shared central fetcher publishes only one timeframe ("1" min).
    # Workers that need a higher timeframe (this one needs 5-min) resample
    # locally in `build_strategy_frame`. So this stays "1" for ALL workers.
    timeframe = "1"

    # How often the run loop wakes up to check things. The strategy fires
    # only on closed 5-min candles, so 10 seconds is plenty -- it lets us
    # react to a fresh candle within 10s of its close without burning the
    # broker's API rate-limit on the option-chain endpoint.
    poll_seconds = OPENING_STRIKE_POLL_SECONDS

    # Number of NIFTY option lots to BUY per entry. 1 lot = 75 units of
    # NIFTY today (per the NSE schedule). The instrument master CSV is the
    # source of truth for the current lot size.
    lots = OPENING_STRIKE_LOTS

    # Daily INR drawdown cap. Once realized+open PnL <= -max_loss, the run
    # loop closes the trade and stops the worker for the day. Set to 0 in
    # .env to disable this safety net entirely (not recommended live).
    max_loss = OPENING_STRIKE_MAX_LOSS

    # Trading WINDOW. The base class refuses to even evaluate signals
    # before `(trading_start_hour, trading_start_minute)`. 09:25 gives the
    # opening 10 minutes of volatility time to settle AND ensures at least
    # one full 5-min candle has closed before we look at anything.
    trading_start_hour = OPENING_STRIKE_TRADING_START_HOUR
    trading_start_minute = OPENING_STRIKE_TRADING_START_MINUTE

    # Hard cutoff. At THIS wall-clock time the base class force-closes any
    # open option leg and stops the worker (see "SQUARE-OFF SAFETY NET"
    # in the class docstring). 15:15 matches every other option-buying
    # worker in this runner (Renko, EMA, HeikinAshi, ProfitShooter).
    square_off_hour = OPENING_STRIKE_SQUARE_OFF_HOUR
    square_off_minute = OPENING_STRIKE_SQUARE_OFF_MINUTE

    # Minutes per derived candle. 1-min source bars get resampled into
    # 5-min bars in `build_strategy_frame`. Change in .env if you want a
    # 1-min or 3-min variant.
    derived_timeframe_minutes = OPENING_STRIKE_DERIVED_TIMEFRAME_MINUTES

    # Soft cutoff: after this time we don't OPEN a new trade because the
    # 15:15 hard stop would close it almost immediately. Existing trades
    # keep running. `dt_time(h, m)` is Python's built-in time-of-day type;
    # comparing it to `candle.time()` does the right thing.
    last_setup_cutoff = dt_time(
        OPENING_STRIKE_LAST_SETUP_HOUR,
        OPENING_STRIKE_LAST_SETUP_MINUTE,
    )

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        # Standard worker plumbing: shared data store, the supervisor's
        # stop-event for Ctrl+C handling, and the authenticated broker
        # client. The parent class hooks these into the run loop.
        super().__init__(store, stop_event, broker)

        # The signal engine is the brain that decides BUY_CALL/BUY_PUT/
        # NO_SIGNAL on each candle. It is STATEFUL -- it remembers the
        # opening strike from its first call and tracks whether it has
        # already fired today. We keep exactly ONE engine per worker.
        self.signal_engine = OPENING_STRIKE_LOGIC.NiftyOpeningStrikePCRVWAPATRSignalGenerator(
            OPENING_STRIKE_STRATEGY_CONFIG
        )

        # Three simple counters for the end-of-day summary line.
        # `signal_count` increments each time the engine returns a
        # genuine BUY signal (NO_SIGNAL responses don't count).
        # `entry_submit_count` only counts signals that we actually
        # managed to TURN INTO a paper entry (some signals fail at the
        # entry stage, e.g. if the ATM contract cannot be resolved).
        # `exit_count` increments whenever any exit fires (SL, trailing
        # SL, time cutoff, or max-loss).
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0

        # --- Option-chain state ----------------------------------------
        # `self._oi_baseline` holds the OPEN-INTEREST value we saw on the
        # FIRST successful option-chain fetch of the session. Every later
        # fetch computes "today's OI change" as `current_oi - baseline`.
        # It stays empty until the first fetch lands, then is never
        # overwritten (so changes always reflect THIS session, not some
        # mid-session reset).
        # Shape: {strike_float: {"ce": ce_oi_float, "pe": pe_oi_float}}.
        self._oi_baseline: dict[float, dict[str, float]] = {}
        self._oi_baseline_captured_at: Optional[datetime] = None

        # `self._last_option_chain_fetch_at` is a simple "remember the
        # last time we called the option_chain API" timestamp. Combined
        # with `_last_oi_change_frame` (the cached result), it lets us
        # avoid hammering the rate-limited endpoint when the run loop
        # ticks faster than OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS.
        self._last_option_chain_fetch_at: Optional[datetime] = None
        self._last_oi_change_frame: Optional[pd.DataFrame] = None

        # `self._high_water_R` tracks the BEST reward this trade has
        # touched so far, measured in "R units" (1R = OPENING_STRIKE_
        # INITIAL_SL_POINTS rupees of OPTION premium). The trailing-SL
        # ratchet uses this number, NOT the current live reward, so a
        # quick spike up to +2.4R locks in the 1R-locked-in tier even if
        # the price immediately pulls back. Reset to 0 in `after_exit`.
        self._high_water_R: float = 0.0

    # ------------------------------------------------------------------
    # Status / lifecycle
    # ------------------------------------------------------------------
    def summary_text(self) -> str:
        """
        One-line end-of-day report. Logged once at 15:15 (or whenever a
        max-loss / Ctrl+C cutoff fires). Format is identical to the other
        ATM workers so the log file is easy to grep across strategies.
        """
        return (
            f"Signals={self.signal_count} | Entries={self.entry_submit_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | RealizedPnL={self.realized_pnl:.2f}"
        )

    def after_exit(self, closed_position: PaperPosition, reason: str) -> None:
        """
        Hook called by the parent's `exit_position` AFTER PnL is realized
        and the option subscription is unregistered. Two jobs here:

        - Bump the exit counter (for the end-of-day summary).
        - Reset the trailing high-water mark so that IF the operator has
          turned on `OPENING_STRIKE_ALLOW_MULTIPLE_ENTRIES=1`, the NEXT
          trade starts with a fresh ratchet (otherwise it would carry
          over a stale "max reward seen" from the previous trade).
        """
        self.exit_count += 1
        self._high_water_R = 0.0

    def minimum_strategy_rows(self) -> int:
        """
        How many derived (5-min) candles must exist before the engine is
        allowed to evaluate. ATR-14 needs 14 candles, so we add a small
        buffer to avoid a "borderline NaN" first call. See the
        OPENING_STRIKE_MIN_BARS computation near the config block.
        """
        return OPENING_STRIKE_MIN_BARS

    # ------------------------------------------------------------------
    # Frame building (1m source -> 5m derived + session VWAP)
    # ------------------------------------------------------------------
    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Turn the shared 1-min NIFTY snapshot into a 5-min DataFrame with
        a session VWAP column attached. The signal engine consumes the
        result on each `process_strategy_frame` call.

        WHY WE COMPUTE VWAP HERE INSTEAD OF READING IT FROM THE FETCHER:
        The central fetcher only publishes O/H/L/C (no volume column),
        because the NIFTY *index* itself reports zero volume on most
        data sources -- it is a synthetic level computed from constituent
        stocks, not a tradable instrument with order flow. The signal
        generator's VWAP check, however, only needs a "fair value" line
        to compare candle closes against, so we use the equal-weight
        proxy `cumulative mean of (H+L+C)/3 since session open`. When
        every bar carries equal volume this proxy IS mathematically the
        true VWAP, and it is the same fallback Indian-market index
        traders use when a real volume series isn't available.
        """
        # Step 1: resample 1-min source into derived (5-min) bars. The
        # helper keeps only COMPLETE bars -- i.e. a 5-min bar made of
        # exactly 5 underlying 1-min rows -- so we never evaluate on a
        # half-formed candle.
        ohlc_n = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
        if ohlc_n.empty:
            return ohlc_n

        # Step 2: filter to today's session only. The central snapshot
        # spans the last several days (the lookback default is 7) so we
        # must drop yesterday's tail before we compute "since-open VWAP"
        # or read "today's opening price". Without this filter, the
        # opening strike would still anchor on the FIRST bar we ever
        # received, which could be from a previous day.
        today = datetime.now().date()
        ohlc_n = ohlc_n[pd.to_datetime(ohlc_n["timestamp"]).dt.date == today].reset_index(drop=True)
        if ohlc_n.empty:
            # Today's first 5-min bar has not closed yet -- the engine
            # will see an empty frame and skip; the run loop will retry.
            return ohlc_n

        # Step 3: compute the equal-weight session VWAP proxy. `expanding`
        # gives a running window that starts at row 0 and grows by one
        # row each step, so `expanding().mean()` is a cumulative average
        # from session open to "now".
        typical_price = (ohlc_n["high"] + ohlc_n["low"] + ohlc_n["close"]) / 3.0
        ohlc_n["vwap"] = typical_price.expanding().mean()
        return ohlc_n

    # ------------------------------------------------------------------
    # Option-chain OI capture
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_option_chain_for_oi(resp) -> dict:
        """
        Flatten DhanHQ's `/optionchain` response into one tidy dict keyed
        by strike. Output shape:

            {22000.0: {"ce_oi": 12345.0, "pe_oi": 67890.0}, ...}

        The DhanHQ wire format is verbose (`{"status": ..., "data":
        {"oc": {"22000.000000": {"ce": {...}, "pe": {...}}}}}`) and the
        SDK has historically shifted casing between minor releases. This
        parser is intentionally defensive: anything that doesn't look
        like a `dict` at each nesting level is skipped silently rather
        than raising, so a single malformed strike never poisons the
        whole evaluation.

        We DROP strikes where:
            - either CE or PE leg is missing entirely (a half-strike
              would skew the PCR sum), OR
            - both CE_OI and PE_OI are zero (typical of far-OTM strikes
              that have never traded -- they would carry pure noise).
        """
        # Guard against entirely malformed responses (network error,
        # empty body, non-dict payload). Returning `{}` lets callers
        # check truthiness instead of try/except'ing every call.
        if not isinstance(resp, dict):
            return {}

        # DhanHQ sets `status="success"` on a good response. Anything
        # else (e.g. "failure" / "error") means the API rejected us;
        # treat it like an empty payload.
        status = str(resp.get("status", "")).strip().lower()
        if status and status != "success":
            return {}

        payload = resp.get("data")
        if not isinstance(payload, dict):
            return {}

        # The strike map lives under the `oc` key (DhanHQ's shorthand
        # for "option chain"). We accept the all-caps variant too just
        # in case a future SDK update changes the casing.
        oc = payload.get("oc") or payload.get("OC") or {}
        if not isinstance(oc, dict):
            return {}

        out: dict[float, dict[str, float]] = {}
        for strike_str, leg_map in oc.items():
            # Each value in `oc` should itself be a dict {"ce": {...},
            # "pe": {...}}. Skip anything that isn't.
            if not isinstance(leg_map, dict):
                continue

            # Strikes are keyed as strings like "22000.000000". Convert
            # to float so we can do numeric comparisons elsewhere.
            try:
                strike = float(strike_str)
            except (TypeError, ValueError):
                continue

            # Pull CE OI then PE OI. We try a couple of casing variants
            # for each side -- some DhanHQ versions return "ce"/"pe",
            # some return "CE"/"PE", and occasionally "Ce"/"Pe".
            record: dict[str, float] = {}
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
                    # Side missing entirely -- can't compute PCR for this
                    # strike, so skip the side. The strike will be
                    # dropped by the "both sides required" check below.
                    continue
                # `_safe_float` returns a float on any input, defaulting
                # to 0.0 for None / empty / non-numeric values.
                record[f"{right_label}_oi"] = _safe_float(leg.get("oi"), 0.0)

            # Only keep strikes that have BOTH CE and PE OI reported,
            # and where at least one side has non-zero OI.
            if "ce_oi" in record and "pe_oi" in record:
                if record["ce_oi"] > 0 or record["pe_oi"] > 0:
                    out[strike] = record
        return out

    def _build_option_chain_oi_change(self) -> Optional[pd.DataFrame]:
        """
        Produce the per-strike OI-change DataFrame the signal engine
        expects. Columns:

            strike (int) | call_oi_change (float) | put_oi_change (float)

        The returned DataFrame is the engine's "OI flow snapshot": each
        row says "at this strike, today's call OI is up by X and put OI
        is up by Y since the session-open baseline". The engine sums
        these around the opening strike to get its PCR number.

        CACHING / RATE LIMITING:
        DhanHQ's option_chain endpoint is one of the tighter rate-limited
        APIs (~1 request/sec budget). We hold a cached copy and only
        re-fetch when more than OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS
        have elapsed since the last successful fetch. That way the run
        loop can poll quickly without blowing through the API budget.

        ERROR HANDLING:
        If the fetch fails (network error, rate limit, empty response),
        we return the LAST cached frame (which may be `None` if no fetch
        has ever succeeded). The caller treats `None`/empty as "skip
        this evaluation" and tries again on the next poll.
        """
        now = datetime.now()

        # If a cached frame is fresh enough, use it as-is to avoid
        # spending an API call.
        if (
            self._last_option_chain_fetch_at is not None
            and self._last_oi_change_frame is not None
            and (now - self._last_option_chain_fetch_at).total_seconds()
            < OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS
        ):
            return self._last_oi_change_frame

        # Step 1: resolve which expiry's chain to query. Same "next-next
        # weekly" rule as the other ATM workers -- this keeps the OI
        # flow we monitor aligned with the option we will eventually
        # BUY when a signal triggers.
        try:
            expiry = self.contract_resolver.get_target_expiry()
        except Exception as exc:
            self.log.warning("Option-chain fetch skipped: expiry resolution failed: %s", exc)
            return self._last_oi_change_frame

        # Step 2: call the broker for the option chain. Wrapped in
        # try/except so a transient network error never crashes the
        # worker -- we just log and let the next poll retry.
        try:
            resp = self.broker.fetch_option_chain(
                NIFTY_INDEX_SECURITY_ID,
                NIFTY_INDEX_EXCHANGE_SEGMENT,
                expiry,
            )
        except Exception as exc:
            self.log.warning("Option-chain fetch raised: %s", exc)
            return self._last_oi_change_frame

        # Step 3: flatten the response into our tidy strike->OI map. If
        # the parser couldn't extract anything (pre-market, rate-limit
        # response, etc.), bail out and let the next poll retry.
        current_oi = self._parse_option_chain_for_oi(resp)
        if not current_oi:
            self.log.warning(
                "Option-chain returned no usable strikes (rate-limited or pre-market?)"
            )
            return self._last_oi_change_frame

        # Step 4: if this is the FIRST successful fetch of the day,
        # capture an OI baseline. This is the snapshot we will subtract
        # from every future fetch to get "today's OI change".
        if not self._oi_baseline:
            self._oi_baseline = {
                strike: {"ce": meta.get("ce_oi", 0.0), "pe": meta.get("pe_oi", 0.0)}
                for strike, meta in current_oi.items()
            }
            self._oi_baseline_captured_at = now
            self.log.info(
                "Captured intraday OI baseline for %d strikes at %s (expiry=%s).",
                len(self._oi_baseline),
                now.strftime("%H:%M:%S"),
                expiry.isoformat(),
            )

        # Step 5: walk every strike in the current snapshot and compute
        # "current OI minus baseline OI" for each side.
        rows = []
        for strike, meta in current_oi.items():
            baseline = self._oi_baseline.get(strike)
            if baseline is None:
                # This strike wasn't in the baseline (e.g. NSE listed a
                # new far-OTM strike mid-session). Treat its baseline as
                # equal to the current value so its first reported
                # "change" is zero rather than a huge spike that would
                # skew our PCR average.
                self._oi_baseline[strike] = {
                    "ce": meta.get("ce_oi", 0.0),
                    "pe": meta.get("pe_oi", 0.0),
                }
                baseline = self._oi_baseline[strike]
            rows.append(
                {
                    "strike": int(round(strike)),
                    "call_oi_change": float(meta.get("ce_oi", 0.0) - baseline.get("ce", 0.0)),
                    "put_oi_change": float(meta.get("pe_oi", 0.0) - baseline.get("pe", 0.0)),
                }
            )

        df = pd.DataFrame(rows)

        # Cache the result + fetch timestamp so the next call within
        # OPENING_STRIKE_OPTION_CHAIN_REFRESH_SECONDS uses this instead
        # of hitting the API again.
        self._last_option_chain_fetch_at = now
        self._last_oi_change_frame = df
        return df

    # ------------------------------------------------------------------
    # Trade management
    # ------------------------------------------------------------------
    def _check_premium_trailing_sl_and_exit(self) -> None:
        """
        Run the premium-based trailing-SL check ONCE. Called on every
        run-loop iteration while we hold an open option leg.

        THE LADDER (also published as `risk_metadata.rr_trailing_rule`
        by the signal generator, so the worker and the engine agree):

            "R" = OPENING_STRIKE_INITIAL_SL_POINTS rupees of option
                  PREMIUM. With the default 20, 1R = 20 INR.

            Reward state                 SL location
            ------------                 -----------
            best reward seen <  2R   ->  entry - 1R (initial loss tol)
            best reward seen >= 2R   ->  entry + 1R (locks in +1R)
            best reward seen >= 3R   ->  entry + 2R (locks in +2R)
            best reward seen >= 4R   ->  entry + 3R (locks in +3R)
            ...and so on for higher tiers.

        The "best reward seen" is `self._high_water_R`, a high-water
        mark that only ratchets UP. A quick spike up to +2.4R locks in
        the +1R tier even if the trade immediately pulls back to +0.5R
        -- the SL never relaxes back to the initial level.
        """
        # 1R in option-premium points. Pulled from the strategy config so
        # tuning it in .env (OPENING_STRIKE_INITIAL_SL_POINTS) is enough.
        R = float(OPENING_STRIKE_STRATEGY_CONFIG.initial_sl_points)
        if R <= 0:
            # Operator set the SL to 0 -- effectively disables the
            # premium-based stop. The time cutoff and max_loss are still
            # in play, so the trade is not unbounded.
            return

        # Look up the current option LTP. We pass `entry_trade_price` as
        # the fallback so a momentarily missing tick doesn't accidentally
        # trigger a "current_premium=0 looks like a crash" exit.
        current_premium = self._get_option_ltp(
            self.pos.option_exchange_segment,
            self.pos.option_security_id,
            fallback=self.pos.entry_trade_price,
        )
        if current_premium <= 0:
            # Truly bad tick (cache empty, direct fetch failed, fallback
            # was zero). Skip this poll; the next poll will likely have
            # a fresh tick.
            return

        # Express the current reward as a multiple of R. Positive means
        # we are in profit, negative means we are in drawdown.
        reward_R = (current_premium - self.pos.entry_trade_price) / R

        # Ratchet the high-water mark: it only goes UP, never down.
        if reward_R > self._high_water_R:
            self._high_water_R = reward_R

        # Decide which SL tier we are on based on the high-water mark.
        if self._high_water_R >= 2.0:
            # Trailing tier active. math.floor(2.4) = 2 -> SL at +1R.
            # math.floor(3.1) = 3 -> SL at +2R. And so on.
            tier_R = math.floor(self._high_water_R) - 1
            effective_sl = self.pos.entry_trade_price + tier_R * R
            sl_label = f"TRAILING_SL_{int(tier_R)}R"
        else:
            # Still in the initial-loss zone. SL stays at entry minus 1R.
            effective_sl = self.pos.entry_trade_price - R
            sl_label = "INITIAL_SL"

        # If the current premium has dropped to (or below) the SL, exit.
        # `exit_position` handles the SELL leg, PnL accounting, and the
        # post-exit hook (which resets `_high_water_R` to 0).
        if current_premium <= effective_sl:
            self.log.info(
                "%s hit | CurrentPremium=%.2f <= SL=%.2f | EntryPremium=%.2f | HighWaterR=%.2f",
                sl_label,
                current_premium,
                effective_sl,
                self.pos.entry_trade_price,
                self._high_water_R,
            )
            self.exit_position(sl_label)

    # ------------------------------------------------------------------
    # Strategy loop body
    # ------------------------------------------------------------------
    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Body of the main poll-iteration. Called by the BasePaperStrategy
        Worker.run() loop AFTER every per-iteration check has passed:
        max-loss not breached, BEFORE the 15:15 cutoff, AFTER the 09:25
        start, source data is fresh, derived frame is large enough, AND
        the latest derived candle changed since last time.

        Flow:
        - If we are already in a trade -> just check the premium
          trailing SL. The signal engine is single-shot per day, so we
          never re-enter on top of an existing position.
        - If we are flat and the engine has already fired today AND the
          operator has not enabled multi-entry mode -> bail before even
          touching the rate-limited option_chain endpoint.
        - If we are flat and past the 15:00 "last setup" cutoff -> bail
          (the 15:15 hard stop would close the trade almost immediately).
        - Otherwise: fetch the latest OI-change snapshot, hand it +
          today's candles + the day's opening price to the signal engine,
          and submit a CE/PE BUY paper order on BUY_CALL / BUY_PUT.
        """
        # ---- In a trade: only manage the trailing SL --------------------
        # While we hold an open option leg the entire job here is risk
        # management. The 15:15 force-close is handled OUTSIDE this
        # method by the base class's run() loop, so we don't have to
        # check the clock here.
        if self.pos.active:
            self._check_premium_trailing_sl_and_exit()
            return

        # ---- Fast-bail: engine already fired today ----------------------
        # The signal engine is stateful: once it returns BUY_CALL or
        # BUY_PUT, it stores that fact and returns NO_SIGNAL on every
        # subsequent call (unless `allow_multiple_entries=True`). We
        # mirror that check here so we don't waste a rate-limited
        # option_chain fetch on a request the engine would reject anyway.
        if (
            not OPENING_STRIKE_STRATEGY_CONFIG.allow_multiple_entries
            and getattr(self.signal_engine, "_entry_signal_sent", False)
        ):
            return

        # ---- Soft cutoff: no fresh entries after 15:00 ------------------
        # `latest_candle_time` is a Python `time` object (just HH:MM:SS,
        # no date). Comparing it against `dt_time(15, 0)` does the right
        # thing for the "is the latest candle from after 15:00?" check.
        latest_candle_time = pd.to_datetime(strategy_frame.iloc[-1]["timestamp"]).time()
        if latest_candle_time > self.last_setup_cutoff:
            return

        # ---- Build OI-change snapshot ---------------------------------
        # This is the rate-limited call (~ once every 30s). If it fails
        # or comes back empty we skip this iteration; the next poll will
        # try again.
        oi_change_df = self._build_option_chain_oi_change()
        if oi_change_df is None or oi_change_df.empty:
            return

        # ---- Tell the engine today's opening price --------------------
        # The first row of `strategy_frame` is today's first 5-min bar
        # (we filtered to today's session in `build_strategy_frame`), so
        # its `open` IS the day's opening price. The engine uses this to
        # fix the opening strike on its very first call of the day.
        opening_price = float(strategy_frame.iloc[0]["open"]) if not strategy_frame.empty else None

        # ---- Ask the engine for a decision ----------------------------
        decision = self.signal_engine.evaluate(
            strategy_frame,
            oi_change_df,
            opening_price=opening_price,
        )

        # Count signal-quality events for the end-of-day summary.
        if decision.signal_triggered:
            self.signal_count += 1

        # ---- Act on the decision --------------------------------------
        # The signal generator emits at most one of BUY_CALL / BUY_PUT /
        # NO_SIGNAL. For BUY_CALL we go LONG (buy ATM CE); for BUY_PUT
        # we go SHORT directionally (buy ATM PE). We never SELL options
        # in this worker, so the worst-case loss per trade is bounded
        # by the premium we paid -- the parent's `enter_position`
        # records `stop_underlying=0.0` because our SL is on premium,
        # not on NIFTY spot.
        if decision.action == "BUY_CALL":
            if self.enter_position("LONG", decision.entry_underlying):
                self.entry_submit_count += 1
            return
        if decision.action == "BUY_PUT":
            if self.enter_position("SHORT", decision.entry_underlying):
                self.entry_submit_count += 1
            return


# =============================================================================
# CPR STRATEGY WORKER (1-min source, resampled to 5-min, ATM single-leg)
# =============================================================================
class CPRStrategyWorker(AtmSingleLegStrategyWorker):
    """
    CPR (Central Pivot Range) strategy. Reads 1-minute data from the shared
    store and hands it straight to the CPR logic module, which internally
    resamples to complete 5-minute candles and attaches previous-day CPR
    levels plus EMA / VWAP / RSI / swing / zone / divergence columns.

    CPR runs three sub-algos (neutral-zone breakout + EMA20 retrace, sideways
    support/resistance zones, RSI divergence reversals) behind one
    `evaluate_candle` decision. Each LONG/SHORT signal is expressed as a BUY
    of the ATM CE/PE on the next-next expiry, exactly like the other directional
    ATM workers. Exits are driven by the engine when the latest 5-minute candle
    crosses the trade's stop or target, so the stop AND target are persisted on
    the position and fed back into the engine via the position context.
    """

    strategy_name = "CPR"
    timeframe = "1"
    poll_seconds = CPR_POLL_SECONDS
    lots = CPR_LOTS
    max_loss = CPR_MAX_LOSS
    trading_start_hour = CPR_TRADING_START_HOUR
    trading_start_minute = CPR_TRADING_START_MINUTE
    square_off_hour = CPR_SQUARE_OFF_HOUR
    square_off_minute = CPR_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.signal_engine = CPR_LOGIC.CPRSignalEngine(CPR_STRATEGY_CONFIG)
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
        # The engine needs indicator + swing warm-up AND a prior session for
        # the previous-day CPR levels. Its own minimum covers the indicators;
        # we floor it so we never run on a single sparse session.
        return max(self.signal_engine.minimum_history_bars(), 40)

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Hand the shared 1-minute OHLC straight to the CPR builder. CPR's
        `prepare_cpr_ohlc_input` resamples 1-minute data into complete
        5-minute candles internally, so no local resample is needed here.
        """
        return CPR_LOGIC.build_cpr_with_indicators(ohlc, CPR_STRATEGY_CONFIG)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Interpret the latest CPR 5-minute candle.

        Flow:
        1. If currently in a trade, build a position context (with target) for
           the engine so it can detect stop/target hits on this candle.
        2. Ask the engine for one decision on the latest candle.
        3. In trade -> only honor exits on this candle.
        4. Flat -> count signal triggers and submit fresh entries, passing the
           stop AND target so the position carries them for later exit checks.
        """
        position_ctx = None
        if self.pos.active:
            position_ctx = CPR_LOGIC.CPRPositionContext(
                direction=self.pos.direction,
                entry_underlying=self.pos.entry_underlying,
                stop_underlying=self.pos.stop_underlying,
                target_underlying=self.pos.target_underlying,
            )

        decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

        if self.pos.active:
            if decision.action == "EXIT":
                self.exit_position(decision.exit_reason)
            return

        if decision.signal_triggered:
            self.signal_count += 1

        if decision.action == "ENTER_LONG":
            if self.enter_position(
                "LONG",
                decision.entry_underlying,
                decision.stop_underlying,
                decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return
        if decision.action == "ENTER_SHORT":
            if self.enter_position(
                "SHORT",
                decision.entry_underlying,
                decision.stop_underlying,
                decision.target_underlying,
            ):
                self.entry_submit_count += 1
            return


# =============================================================================
# CPR ALGO 3 WORKER (multi-instrument: spot + ITM CE + ITM PE observation)
# =============================================================================
class CPRAlgo3StrategyWorker(AtmSingleLegStrategyWorker):
    """
    CPR Algo 3 ("CPR basic setup") — the multi-instrument CPR strategy.

    Unlike the single-chart CPR worker above, Algo 3 watches THREE charts at once:
    the NIFTY spot plus a ~ITM CE and a ~ITM PE of the CURRENT-week expiry. It only
    fires when VWAP and the CPR band line up across all three (with RSI/ARSI and the
    target read off the spot). See `Nifty CPR Algo 3 Signal Generator.py`.

    The two ITM options are OBSERVATION ONLY: a signal still BUYS the ATM CE/PE of
    the next-next expiry through the shared, battle-tested `enter_position` path, so
    Algo 3 trades exactly like the other directional ATM workers. The observation
    strikes are picked ONCE per session (fixed) so each option's VWAP/CPR/RSI stay
    continuous, and their 1-minute OHLC is fetched on demand from the broker.

    Exits: the generator is entry-only, so this worker drives its own exit by
    watching the spot close cross the stored target / stop (plus the base loop's
    max-loss and daily square-off).
    """

    strategy_name = "CPRAlgo3"
    timeframe = "1"
    poll_seconds = CPR_ALGO3_POLL_SECONDS
    lots = CPR_ALGO3_LOTS
    max_loss = CPR_ALGO3_MAX_LOSS
    trading_start_hour = CPR_ALGO3_TRADING_START_HOUR
    trading_start_minute = CPR_ALGO3_TRADING_START_MINUTE
    square_off_hour = CPR_ALGO3_SQUARE_OFF_HOUR
    square_off_minute = CPR_ALGO3_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        self.config = CPR_ALGO3_CONFIG
        # Observation legs, chosen once per session by `_ensure_observation_strikes`.
        self.itm_ce: Optional[dict] = None
        self.itm_pe: Optional[dict] = None
        self.observation_expiry = None
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
        # 5-minute rows after the internal resample: enough for the indicator
        # warm-up AND a prior session for the previous-day CPR levels.
        return 40

    def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the SPOT 1-minute frame with CPR levels / VWAP / RSI / ARSI.

        This drives the base loop's 5-minute signature gate (so we only act on a new
        spot candle) and gives the latest spot close for exit checks. The generator
        re-derives indicators on the raw frames itself, so `process_strategy_frame`
        passes RAW OHLC to it, not this enriched frame.
        """
        return CPR_LOGIC.build_cpr_with_indicators(ohlc, self.config.indicator_config)

    def _ensure_observation_strikes(self) -> bool:
        """Pick the ~ITM CE/PE of the current-week expiry once per session."""
        if self.itm_ce is not None and self.itm_pe is not None:
            return True
        spot = self._get_underlying_spot(fallback=0.0)
        if spot <= 0:
            return False
        atm = round(spot / ATM_STRIKE_STEP) * ATM_STRIKE_STEP
        ce_strike = atm - CPR_ALGO3_ITM_OFFSET  # CE is ITM at lower strikes
        pe_strike = atm + CPR_ALGO3_ITM_OFFSET  # PE is ITM at higher strikes
        try:
            expiry = self.contract_resolver.get_current_week_expiry()
            ce = self.contract_resolver.get_option_for_strike(expiry, ce_strike, "CE")
            pe = self.contract_resolver.get_option_for_strike(expiry, pe_strike, "PE")
        except Exception as exc:
            self.log.warning("Algo3 observation-strike resolution failed: %s", exc)
            return False
        if not ce or not pe:
            self.log.warning(
                "Algo3 could not resolve observation legs (atm=%.0f, ce=%.0f, pe=%.0f); will retry.",
                atm, ce_strike, pe_strike,
            )
            return False
        self.itm_ce, self.itm_pe, self.observation_expiry = ce, pe, expiry
        self.log.info(
            "Algo3 observation legs fixed | Expiry=%s | CE=%s (%.0f) | PE=%s (%.0f)",
            expiry.isoformat(), ce["trading_symbol"], ce_strike, pe["trading_symbol"], pe_strike,
        )
        return True

    def _fetch_option_1m(self, contract: dict) -> pd.DataFrame:
        """1-minute OHLC for one option strike (same intraday API as the index)."""
        return self.broker.fetch_index_1m_ohlc(
            security_id=int(contract["security_id"]),
            exchange_segment=str(contract["exchange_segment"]),
            instrument_type=OPTION_INSTRUMENT_TYPE,
        )

    def _check_spot_target_stop_and_exit(self, spot_close: float) -> None:
        """Exit the open trade when the spot close crosses the stored target/stop."""
        stop = self.pos.stop_underlying
        target = self.pos.target_underlying
        stop_ok = pd.notna(stop) and stop > 0
        target_ok = pd.notna(target) and target > 0
        reason = None
        if self.pos.direction == "LONG":
            if stop_ok and spot_close <= stop:
                reason = "ALGO3_STOP"
            elif target_ok and spot_close >= target:
                reason = "ALGO3_TARGET"
        else:  # SHORT
            if stop_ok and spot_close >= stop:
                reason = "ALGO3_STOP"
            elif target_ok and spot_close <= target:
                reason = "ALGO3_TARGET"
        if reason:
            self.exit_position(reason)

    def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
        """
        Once per new 5-minute spot candle:
        - ensure the observation legs are chosen (retry until resolvable),
        - if in a trade, only check the spot target/stop exit,
        - if flat, fetch the two option OHLC series, ask Algo 3 for a decision on the
          latest aligned candle, and BUY the ATM CE/PE (next-next expiry) on a signal.
        """
        if not self._ensure_observation_strikes():
            return

        if self.pos.active:
            self._check_spot_target_stop_and_exit(float(strategy_frame.iloc[-1]["close"]))
            return

        try:
            ce_ohlc = self._fetch_option_1m(self.itm_ce)
            pe_ohlc = self._fetch_option_1m(self.itm_pe)
        except Exception as exc:
            self.log.warning("Algo3 observation OHLC fetch failed: %s", exc)
            return
        if ce_ohlc is None or ce_ohlc.empty or pe_ohlc is None or pe_ohlc.empty:
            return

        spot_snapshot = self.store.get(self.timeframe)
        if spot_snapshot is None or spot_snapshot.frame.empty:
            return

        decision = CPR_ALGO3_LOGIC.get_latest_nifty_cpr_algo3_signal(
            spot_snapshot.frame, ce_ohlc, pe_ohlc, config=self.config
        )
        if decision.signal_triggered:
            self.signal_count += 1
        if decision.action == "ENTER_LONG":
            if self.enter_position(
                "LONG", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
            ):
                self.entry_submit_count += 1
        elif decision.action == "ENTER_SHORT":
            if self.enter_position(
                "SHORT", decision.entry_underlying, decision.stop_underlying, decision.target_underlying
            ):
                self.entry_submit_count += 1


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
    expiry rule used by the eight ATM workers.

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

        # Real execution (live mode only; no-op True in paper mode). On failure
        # we fall back to paper and still record the position below.
        real_ok = self._place_real_hedged_entry(
            main_leg={"option_type": str(main["option_type"]), "strike": float(main["strike"]),
                      "expiry": main["expiry_date"], "quantity": main_qty,
                      "dhan_symbol": str(main["trading_symbol"])},
            hedge_leg={"option_type": str(hedge["option_type"]), "strike": float(hedge["strike"]),
                       "expiry": hedge["expiry_date"], "quantity": hedge_qty,
                       "dhan_symbol": str(hedge["trading_symbol"])},
        )
        exec_mode = self._exec_mode_tag(real_ok)

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
        self.publish_trade_event(
            {
                "action": "ENTRY",
                "mode": exec_mode,
                "direction": self.pos.direction,
                "lots": self.lots,
                "lot_size": self.pos.main_lot_size,
                "quantity": self.pos.main_quantity,
                "expiry": (
                    self.pos.main_expiry.isoformat() if self.pos.main_expiry is not None else "NA"
                ),
                "legs": [
                    {
                        "symbol": self.pos.main_symbol,
                        "side": self.pos.main_side,
                        "right": self.pos.main_right,
                        "strike": self.pos.main_strike,
                        "entry_price": self.pos.main_entry_price,
                    },
                    {
                        "symbol": self.pos.hedge_symbol,
                        "side": self.pos.hedge_side,
                        "right": self.pos.hedge_right,
                        "strike": self.pos.hedge_strike,
                        "entry_price": self.pos.hedge_entry_price,
                    },
                ],
            }
        )
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

        # Real execution (live mode only; no-op True in paper mode).
        real_ok = self._place_real_hedged_exit(
            main_leg={"option_type": closed.main_right, "strike": closed.main_strike,
                      "expiry": closed.main_expiry, "quantity": closed.main_quantity,
                      "dhan_symbol": closed.main_symbol},
            hedge_leg={"option_type": closed.hedge_right, "strike": closed.hedge_strike,
                       "expiry": closed.hedge_expiry, "quantity": closed.hedge_quantity,
                       "dhan_symbol": closed.hedge_symbol},
        )
        exec_mode = self._exec_mode_tag(real_ok)

        # If a LIVE hedged exit did not confirm fills on BOTH legs, real exposure
        # is still open. Do NOT flatten the books - keep the position so the worker
        # retries the exit next cycle, and alert for manual square-off.
        if self.live_trading and not real_ok:
            self.log.error(
                "LIVE HEDGED EXIT NOT CONFIRMED | %s %s | Reason=%s | position kept OPEN "
                "for retry/manual square-off (main=%s, hedge=%s).",
                self.strategy_name, closed.direction, reason, closed.main_symbol, closed.hedge_symbol,
            )
            self.publish_trade_event({
                "action": "EXIT_FAILED",
                "mode": exec_mode,
                "direction": closed.direction,
                "reason": reason,
                "legs": [
                    {"symbol": closed.main_symbol, "side": "BUY",
                     "right": closed.main_right, "strike": closed.main_strike},
                    {"symbol": closed.hedge_symbol, "side": "SELL",
                     "right": closed.hedge_right, "strike": closed.hedge_strike},
                ],
            })
            return

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

        self.publish_trade_event(
            {
                "action": "EXIT",
                "mode": exec_mode,
                "exit_order_failed": bool(self.live_trading and not real_ok),
                "direction": closed.direction,
                "reason": reason,
                "lots": self.lots,
                "lot_size": closed.main_lot_size,
                "quantity": closed.main_quantity,
                "pnl": total_pnl,
                "expiry": expiry_txt,
                "legs": [
                    {
                        "symbol": closed.main_symbol,
                        "side": closed.main_side,
                        "right": closed.main_right,
                        "strike": closed.main_strike,
                        "entry_price": closed.main_entry_price,
                        "exit_price": main_exit_price,
                    },
                    {
                        "symbol": closed.hedge_symbol,
                        "side": closed.hedge_side,
                        "right": closed.hedge_right,
                        "strike": closed.hedge_strike,
                        "entry_price": closed.hedge_entry_price,
                        "exit_price": hedge_exit_price,
                    },
                ],
            }
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

        # Real execution (live mode only; no-op True in paper mode). On failure
        # we fall back to paper and still record the position below.
        real_ok = self._place_real_hedged_entry(
            main_leg={"option_type": str(main["option_type"]), "strike": float(main["strike"]),
                      "expiry": main["expiry_date"], "quantity": main_qty,
                      "dhan_symbol": str(main["trading_symbol"])},
            hedge_leg={"option_type": str(hedge["option_type"]), "strike": float(hedge["strike"]),
                       "expiry": hedge["expiry_date"], "quantity": hedge_qty,
                       "dhan_symbol": str(hedge["trading_symbol"])},
        )
        exec_mode = self._exec_mode_tag(real_ok)

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
        self.publish_trade_event(
            {
                "action": "ENTRY",
                "mode": exec_mode,
                "direction": self.pos.direction,
                "lots": self.lots,
                "lot_size": self.pos.main_lot_size,
                "quantity": self.pos.main_quantity,
                "expiry": (
                    self.pos.main_expiry.isoformat() if self.pos.main_expiry is not None else "NA"
                ),
                "legs": [
                    {
                        "symbol": self.pos.main_symbol,
                        "side": self.pos.main_side,
                        "right": self.pos.main_right,
                        "strike": self.pos.main_strike,
                        "entry_price": self.pos.main_entry_price,
                    },
                    {
                        "symbol": self.pos.hedge_symbol,
                        "side": self.pos.hedge_side,
                        "right": self.pos.hedge_right,
                        "strike": self.pos.hedge_strike,
                        "entry_price": self.pos.hedge_entry_price,
                    },
                ],
            }
        )
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

        # Real execution (live mode only; no-op True in paper mode).
        real_ok = self._place_real_hedged_exit(
            main_leg={"option_type": closed.main_right, "strike": closed.main_strike,
                      "expiry": closed.main_expiry, "quantity": closed.main_quantity,
                      "dhan_symbol": closed.main_symbol},
            hedge_leg={"option_type": closed.hedge_right, "strike": closed.hedge_strike,
                       "expiry": closed.hedge_expiry, "quantity": closed.hedge_quantity,
                       "dhan_symbol": closed.hedge_symbol},
        )
        exec_mode = self._exec_mode_tag(real_ok)

        # If a LIVE hedged exit did not confirm fills on BOTH legs, real exposure
        # is still open. Do NOT flatten the books - keep the position so the worker
        # retries the exit next cycle, and alert for manual square-off.
        if self.live_trading and not real_ok:
            self.log.error(
                "LIVE HEDGED EXIT NOT CONFIRMED | %s %s | Reason=%s | position kept OPEN "
                "for retry/manual square-off (main=%s, hedge=%s).",
                self.strategy_name, closed.direction, reason, closed.main_symbol, closed.hedge_symbol,
            )
            self.publish_trade_event({
                "action": "EXIT_FAILED",
                "mode": exec_mode,
                "direction": closed.direction,
                "reason": reason,
                "legs": [
                    {"symbol": closed.main_symbol, "side": "BUY",
                     "right": closed.main_right, "strike": closed.main_strike},
                    {"symbol": closed.hedge_symbol, "side": "SELL",
                     "right": closed.hedge_right, "strike": closed.hedge_strike},
                ],
            })
            return

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

        self.publish_trade_event(
            {
                "action": "EXIT",
                "mode": exec_mode,
                "exit_order_failed": bool(self.live_trading and not real_ok),
                "direction": closed.direction,
                "reason": reason,
                "lots": self.lots,
                "lot_size": closed.main_lot_size,
                "quantity": closed.main_quantity,
                "pnl": total_pnl,
                "expiry": expiry_txt,
                "legs": [
                    {
                        "symbol": closed.main_symbol,
                        "side": closed.main_side,
                        "right": closed.main_right,
                        "strike": closed.main_strike,
                        "entry_price": closed.main_entry_price,
                        "exit_price": main_exit_price,
                    },
                    {
                        "symbol": closed.hedge_symbol,
                        "side": closed.hedge_side,
                        "right": closed.hedge_right,
                        "strike": closed.hedge_strike,
                        "entry_price": closed.hedge_entry_price,
                        "exit_price": hedge_exit_price,
                    },
                ],
            }
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

        # Real execution (live mode only; no-op True in paper mode). On failure
        # we fall back to paper and still record the position below.
        real_ok = self._place_real_hedged_entry(
            main_leg={"option_type": str(monitor_meta["option_type"]), "strike": float(monitor_meta["strike"]),
                      "expiry": monitor_meta["expiry_date"], "quantity": monitor_qty,
                      "dhan_symbol": str(monitor_meta["trading_symbol"])},
            hedge_leg={"option_type": str(hedge_meta["option_type"]), "strike": float(hedge_meta["strike"]),
                       "expiry": hedge_meta["expiry_date"], "quantity": hedge_qty,
                       "dhan_symbol": str(hedge_meta["trading_symbol"])},
        )
        exec_mode = self._exec_mode_tag(real_ok)
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
        self.publish_trade_event(
            {
                "action": "ENTRY",
                "mode": exec_mode,
                "direction": side,
                "lots": self.lots,
                "lot_size": new_pos.main_lot_size,
                "quantity": new_pos.main_quantity,
                "expiry": (
                    new_pos.main_expiry.isoformat() if new_pos.main_expiry is not None else "NA"
                ),
                "legs": [
                    {
                        "symbol": new_pos.main_symbol,
                        "side": new_pos.main_side,
                        "right": new_pos.main_right,
                        "strike": new_pos.main_strike,
                        "entry_price": new_pos.main_entry_price,
                    },
                    {
                        "symbol": new_pos.hedge_symbol,
                        "side": new_pos.hedge_side,
                        "right": new_pos.hedge_right,
                        "strike": new_pos.hedge_strike,
                        "entry_price": new_pos.hedge_entry_price,
                    },
                ],
            }
        )
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

        # Real execution (live mode only; no-op True in paper mode).
        real_ok = self._place_real_hedged_exit(
            main_leg={"option_type": closed.main_right, "strike": closed.main_strike,
                      "expiry": closed.main_expiry, "quantity": closed.main_quantity,
                      "dhan_symbol": closed.main_symbol},
            hedge_leg={"option_type": closed.hedge_right, "strike": closed.hedge_strike,
                       "expiry": closed.hedge_expiry, "quantity": closed.hedge_quantity,
                       "dhan_symbol": closed.hedge_symbol},
        )
        exec_mode = self._exec_mode_tag(real_ok)

        # If a LIVE hedged exit did not confirm fills on BOTH legs, real exposure
        # is still open. Do NOT flatten the books - keep the position so the worker
        # retries the exit next cycle, and alert for manual square-off.
        if self.live_trading and not real_ok:
            self.log.error(
                "LIVE HEDGED EXIT NOT CONFIRMED | %s %s | Reason=%s | position kept OPEN "
                "for retry/manual square-off (main=%s, hedge=%s).",
                self.strategy_name, closed.direction, reason, closed.main_symbol, closed.hedge_symbol,
            )
            self.publish_trade_event({
                "action": "EXIT_FAILED",
                "mode": exec_mode,
                "direction": closed.direction,
                "reason": reason,
                "legs": [
                    {"symbol": closed.main_symbol, "side": "BUY",
                     "right": closed.main_right, "strike": closed.main_strike},
                    {"symbol": closed.hedge_symbol, "side": "SELL",
                     "right": closed.hedge_right, "strike": closed.hedge_strike},
                ],
            })
            return
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
        self.publish_trade_event(
            {
                "action": "EXIT",
                "mode": exec_mode,
                "exit_order_failed": bool(self.live_trading and not real_ok),
                "direction": side,
                "reason": reason,
                "lots": self.lots,
                "lot_size": closed.main_lot_size,
                "quantity": closed.main_quantity,
                "pnl": total_pnl,
                "expiry": expiry_txt,
                "legs": [
                    {
                        "symbol": closed.main_symbol,
                        "side": closed.main_side,
                        "right": closed.main_right,
                        "strike": closed.main_strike,
                        "entry_price": closed.main_entry_price,
                        "exit_price": main_exit_price,
                    },
                    {
                        "symbol": closed.hedge_symbol,
                        "side": closed.hedge_side,
                        "right": closed.hedge_right,
                        "strike": closed.hedge_strike,
                        "entry_price": closed.hedge_entry_price,
                        "exit_price": hedge_exit_price,
                    },
                ],
            }
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
# LONG STRANGLE WORKER (AlgoTest port; time-based, dual-leg, BUY-only)
# =============================================================================
class LongStrangleWorker(BasePaperStrategyWorker):
    """
    Intraday long OTM strangle on NIFTY weekly options (ported from AlgoTest).

    BEGINNER-FRIENDLY OVERVIEW
    --------------------------
    A "long strangle" is two BOUGHT options held at once:
        * BUY one OTM call (CE) - profits if NIFTY rallies hard.
        * BUY one OTM put  (PE) - profits if NIFTY falls hard.
    Net, it is a bet that the index makes a BIG intraday move in EITHER
    direction; it bleeds slowly (premium decay) on a quiet, range-bound day.

    Like Delta20HedgedSpreadWorker, the two legs are TWO INDEPENDENT trades
    sharing one worker thread: each has its own stop-loss and its own
    active/flat state, and one leg's exit does not touch the other
    (AlgoTest's "Square Off: Partial").

    SPEC RECAP (the AlgoTest config, in code terms)
    -----------------------------------------------
    - ENTRY (once per day, time-based): at STRANGLE_ENTRY_*:* (default 09:30)
      BUY 1 lot of the OTM1 weekly CE and 1 lot of the OTM1 weekly PE. "OTM1" =
      one strike out of the money (CE = ATM+1 step, PE = ATM-1 step; one step =
      ATM_STRIKE_STEP). There is NO indicator and NO signal generator - the only
      entry trigger is the wall-clock.
    - PER-LEG STOP-LOSS: a leg exits when its premium falls to
      <= entry * (1 - STRANGLE_SL_PCT) (default 5% below entry)...
    - PER-LEG TRAILING STOP ("1:1"): ...but the stop trails UP as the premium
      rises. For every STRANGLE_TRAIL_TRIGGER_PCT (%) the premium gains over
      entry, the stop is lifted by STRANGLE_TRAIL_STEP_PCT (%) of the entry
      premium. With both at 1.0 the stop sits a constant 5% below the
      high-water premium once the trade is in profit. The high-water mark only
      ratchets up, so a spike-then-pullback still locks in the gain. (See
      `_compute_trailing_sl` for the exact ladder + a worked example.)
    - EXIT (force): at STRANGLE_SQUARE_OFF_*:* (default 15:15) close whatever is
      still open and stop for the day.
    - MOMENTUM RE-ENTRY (AlgoTest "RE MOMENTUM"): a stopped-out leg is NOT
      re-bought immediately. We keep watching the SAME strike and re-enter only
      once its premium rebounds STRANGLE_REENTRY_MOMENTUM_PCT (default +5%) above
      the stop-out price - confirmation the move resumed in the buyer's favour.
      Repeats up to STRANGLE_MAX_REENTRIES (default 10) times per leg per day.
      Toggle with STRANGLE_REENTRY_ENABLED. (See `_exit_leg` for arming and
      `_manage_reentry` for the trigger.)
    - SAFETY: STRANGLE_MAX_LOSS is a strategy-wide INR backstop across both legs
      (the AlgoTest overall-SL was off, but the live runner always keeps a hard
      cap).

    EXPIRY RULE: current-week weekly (OptionsContractResolver.get_otm_option
    defaults to get_current_week_expiry), matching the AlgoTest "Weekly" setting.

    WHY OVERRIDE run() INSTEAD OF USING THE BASE LOOP
    -------------------------------------------------
    This strategy never consumes OHLC. The base loop's candle-signature gate
    would needlessly hold back our purely time-and-LTP-driven logic, so we
    override run() to keep the shared risk / cutoff / pre-open scaffolding but
    skip the OHLC plumbing entirely (same rationale as Delta20).

    NON-OBVIOUS DESIGN POINTS
    -------------------------
    - self.pos (the base single-position handle) is intentionally left flat and
      unused. We carry two real positions in self.ce_pos and self.pe_pos and
      override the base risk / exit handlers to operate on both.
    - Each leg's stop-loss is independent, but STRANGLE_MAX_LOSS is a single
      shared pool: if the COMBINED PnL breaches it, both legs close and the
      worker stops for the day.
    """

    strategy_name = "LongStrangle"
    timeframe = "1"
    poll_seconds = STRANGLE_POLL_SECONDS
    lots = STRANGLE_LOTS
    max_loss = STRANGLE_MAX_LOSS
    # The base pre-open wait uses trading_start_*; for us that IS the entry time.
    trading_start_hour = STRANGLE_ENTRY_HOUR
    trading_start_minute = STRANGLE_ENTRY_MINUTE
    square_off_hour = STRANGLE_SQUARE_OFF_HOUR
    square_off_minute = STRANGLE_SQUARE_OFF_MINUTE

    def __init__(
        self,
        store: SharedMarketDataStore,
        stop_event: threading.Event,
        broker: DhanBrokerClient,
    ):
        super().__init__(store, stop_event, broker)
        # Two independent single-leg BUY positions. The base self.pos stays
        # flat and unused (see class docstring).
        self.ce_pos = PaperPosition()
        self.pe_pos = PaperPosition()
        # High-water premium per leg, seeded at entry and ratcheted up only.
        # Drives the 1:1 trailing-stop ladder in _compute_trailing_sl.
        self.ce_high_water_premium = 0.0
        self.pe_high_water_premium = 0.0
        # Single-shot INITIAL entry guards. Each leg opens its FIRST position
        # once (the first poll past the entry time it can resolve + fill);
        # `entered_today` is the convenience "both initial legs are on" gate the
        # run loop checks. Crucially, once a leg's initial entry is done it is
        # NEVER fresh-entered again - any later re-fill comes ONLY from the
        # momentum re-entry path (_manage_reentry). This keeps the partial-entry
        # retry (one leg failed to fill at open) from accidentally re-opening a
        # leg that has since been stopped out.
        self.ce_initial_done = False
        self.pe_initial_done = False
        self.entered_today = False

        # ----- Momentum re-entry state (per leg) ------------------------
        # When a leg is stopped out we (optionally) arm it: keep its LTP
        # subscription alive and watch the SAME strike for a +momentum rebound
        # above the stop-out price. These hold that watch.
        self.ce_awaiting_reentry = False
        self.pe_awaiting_reentry = False
        # Premium at which the leg was last stopped out - the reference the
        # rebound is measured against.
        self.ce_stop_out_price = 0.0
        self.pe_stop_out_price = 0.0
        # How many times each leg has already re-entered today (capped by
        # STRANGLE_MAX_REENTRIES).
        self.ce_reentry_count = 0
        self.pe_reentry_count = 0
        # The exact contract last held on each leg (normalized 7-key dict), so a
        # re-entry re-buys the SAME strike rather than re-deriving OTM1 from a
        # since-moved spot.
        self.ce_last_contract: Optional[dict] = None
        self.pe_last_contract: Optional[dict] = None

        # Telemetry for the end-of-day summary line.
        self.entry_submit_count = 0
        self.exit_count = 0
        self.reentry_count = 0

    # ------------------------------------------------------------------
    # Per-side state accessor (keeps the per-leg helpers readable)
    # ------------------------------------------------------------------
    def _leg_pos(self, side: str) -> PaperPosition:
        if side == "CE":
            return self.ce_pos
        if side == "PE":
            return self.pe_pos
        raise ValueError(f"Unsupported side: {side!r}")

    def _reentry_state(self, side: str):
        """
        Bundled read of one leg's momentum-re-entry state, returned in a fixed
        order: (awaiting, stop_out_price, last_contract, reentry_count). Keeps
        the re-entry helpers readable without a six-line if/else at each call.
        """
        if side == "CE":
            return (self.ce_awaiting_reentry, self.ce_stop_out_price,
                    self.ce_last_contract, self.ce_reentry_count)
        if side == "PE":
            return (self.pe_awaiting_reentry, self.pe_stop_out_price,
                    self.pe_last_contract, self.pe_reentry_count)
        raise ValueError(f"Unsupported side: {side!r}")

    # ------------------------------------------------------------------
    # Risk / lifecycle overrides (we use ce_pos + pe_pos, not self.pos)
    # ------------------------------------------------------------------
    def _get_open_position_pnl(self) -> float:
        """
        Live mark-to-market PnL summed across both legs.

        Both legs are BUY options, so each leg's PnL is simply
        (live - entry) * qty. Inactive legs contribute zero. The base class's
        is_max_loss_breached() calls this to enforce STRANGLE_MAX_LOSS.
        """
        total = 0.0
        for leg_pos in (self.ce_pos, self.pe_pos):
            if not leg_pos.active:
                continue
            live = self._get_option_ltp(
                leg_pos.option_exchange_segment,
                leg_pos.option_security_id,
                fallback=leg_pos.entry_trade_price,
            )
            total += (live - leg_pos.entry_trade_price) * leg_pos.quantity
        return float(total)

    def exit_position(self, reason: str) -> None:
        """
        Close BOTH legs at once. This is the single entry point the base risk /
        time-cutoff handlers call for strategy-wide events (max-loss breach,
        15:15 cutoff). Per-leg stop-losses call _exit_leg directly so only the
        stopped leg closes.
        """
        if self.ce_pos.active:
            self._exit_leg("CE", reason)
        if self.pe_pos.active:
            self._exit_leg("PE", reason)

    def handle_max_loss_and_stop(self, total_pnl: float, open_pnl: float) -> None:
        """One-time shutdown when the strategy-wide max loss is breached."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.error(
            "MAX_LOSS breached | Limit=%.2f | SessionPnL=%.2f | OpenPnL=%.2f. "
            "Closing both legs and stopping.",
            self.max_loss, total_pnl, open_pnl,
        )
        try:
            self.exit_position("MAX_LOSS_BREACH")
        except Exception as exc:
            self.log.exception("Error while exiting on max-loss breach: %s", exc)
        self._unsubscribe_armed_legs()
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day due to max-loss shutdown.")

    def handle_square_off_and_stop(self) -> None:
        """One-time shutdown at the daily square-off cutoff (default 15:15)."""
        if self.cutoff_handled:
            return
        self.cutoff_handled = True
        self.log.info(
            "%02d:%02d cutoff reached. Closing any open legs and halting trading for the day.",
            self.square_off_hour, self.square_off_minute,
        )
        try:
            self.exit_position("TIME_CUTOFF")
        except Exception as exc:
            self.log.exception("Error while exiting at cutoff: %s", exc)
        self._unsubscribe_armed_legs()
        self.log.info("Paper summary | %s", self.summary_text())
        self.log.info("Strategy stopped for the day.")

    def summary_text(self) -> str:
        return (
            f"Entries={self.entry_submit_count} | Re-entries={self.reentry_count} | "
            f"Exits={self.exit_count} | Trades={self.completed_trades} | "
            f"RealizedPnL={self.realized_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Run loop (no OHLC dependency)
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Main worker loop, every STRANGLE_POLL_SECONDS until stop / shutdown.

        Per iteration, in order:
            1. Risk check    -> max-loss breach -> close everything, exit.
            2. Time cutoff   -> 15:15 reached   -> close everything, exit.
            3. Pre-open wait -> before entry time -> sleep and retry.
            4. Entry         -> at/after entry time, open BOTH legs once.
            5. Manage stops  -> run each open leg's stop-loss / trailing stop.
            6. Manage re-entry -> for each stopped-out, armed leg, re-buy the
               same strike once its premium rebounds past the momentum trigger.

        Wrapped in try/except so one bad poll (e.g. a transient network error)
        cannot kill the worker for the whole day.
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
                            "Before entry time (%02d:%02d). Waiting to open the strangle.",
                            self.trading_start_hour, self.trading_start_minute,
                        )
                        self.preopen_wait_logged = True
                    self.wait_for_next_poll()
                    continue
                self.preopen_wait_logged = False

                # 4. Open both legs once, the first poll at/after entry time.
                if not self.entered_today:
                    self._enter_both_legs()

                # 5. Manage each open leg's stop independently (also harmless
                #    immediately after entry: premium ~= entry, no exit fires).
                self._manage_leg("CE")
                self._manage_leg("PE")

                # 6. Re-buy a stopped-out leg once its premium rebounds past the
                #    momentum trigger (no-op for legs that are flat-and-done or
                #    still in a trade).
                self._manage_reentry("CE")
                self._manage_reentry("PE")
            except Exception as exc:
                self.log.exception("Main loop error: %s", exc)

            self.wait_for_next_poll()

        self.log.info("%s strategy worker exited.", self.strategy_name)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _enter_both_legs(self) -> None:
        """
        Open each leg's INITIAL position once. If only one resolves this poll
        (e.g. a transient LTP gap on the other), its `*_initial_done` flag flips
        and the next poll retries ONLY the still-pending leg. A leg whose initial
        entry is already done is never fresh-entered here again, so a leg that
        has since been stopped out is left to the momentum re-entry path rather
        than re-opened by this retry. `entered_today` (= both initial done)
        closes the gate for good.
        """
        spot = self._get_underlying_spot(fallback=0.0)
        if spot <= 0:
            self.log.warning("Strangle entry deferred: NIFTY spot LTP unavailable.")
            return
        if not self.ce_initial_done:
            self.ce_initial_done = self._enter_leg("CE", spot)
        if not self.pe_initial_done:
            self.pe_initial_done = self._enter_leg("PE", spot)
        self.entered_today = self.ce_initial_done and self.pe_initial_done

    def _enter_leg(self, side: str, spot: float) -> bool:
        """
        Resolve the OTM1 strike for `side` from the live spot and BUY it (the
        INITIAL entry). No-op (returns True) if the leg is already open. The
        order placement / bookkeeping is shared with the re-entry path through
        `_buy_leg`.
        """
        leg_pos = self._leg_pos(side)
        if leg_pos.active:
            return True

        try:
            contract = self.contract_resolver.get_otm_option(
                spot, side, otm_steps=STRANGLE_OTM_STEPS
            )
        except Exception as exc:
            self.log.warning(
                "Strangle %s entry skipped: OTM option resolution failed: %s", side, exc
            )
            return False

        return self._buy_leg(side, contract, spot=spot, is_reentry=False)

    def _buy_leg(self, side: str, contract: dict, spot: float, is_reentry: bool) -> bool:
        """
        BUY one option leg from a resolved `contract` dict and record it as a
        paper position. Shared by the initial entry (`_enter_leg`) and the
        momentum re-entry (`_manage_reentry`). Mirrors
        AtmSingleLegStrategyWorker.enter_position for a single BUY leg but writes
        into ce_pos / pe_pos.

        `contract` must carry: security_id, exchange_segment, trading_symbol,
        option_type, strike, expiry_date, lot_size (both `get_otm_option` and the
        stashed `_last_contract` provide these). `is_reentry` only changes the
        log/telemetry wording and bumps the per-leg re-entry counter.
        """
        option_sec_id = int(contract["security_id"])
        option_segment = str(contract["exchange_segment"])
        trading_symbol = str(contract["trading_symbol"])
        option_right = str(contract["option_type"])
        option_strike = float(contract["strike"])
        expiry_date = contract["expiry_date"]
        lot_size = _to_int_safe(contract["lot_size"], 0)

        if not trading_symbol or option_sec_id <= 0:
            self.log.warning("Strangle %s entry skipped: option identifiers missing.", side)
            return False
        if lot_size <= 0:
            self.log.warning(
                "Strangle %s entry skipped: invalid lot size %s for %s.",
                side, lot_size, trading_symbol,
            )
            return False

        option_ltp = self._get_option_ltp(option_segment, option_sec_id, fallback=0.0)
        if option_ltp <= 0:
            self.log.warning(
                "Strangle %s entry skipped: option LTP unavailable for %s (security_id=%s).",
                side, trading_symbol, option_sec_id,
            )
            return False

        quantity = lot_size * self.lots
        order_id = self._next_paper_order_id("BUY")

        # Real execution (live mode only; no-op returning True in paper mode).
        real_ok = self._place_real_leg("BUY", {
            "option_type": option_right, "strike": option_strike,
            "expiry": expiry_date, "quantity": quantity, "dhan_symbol": trading_symbol,
        })
        exec_mode = self._exec_mode_tag(real_ok)

        # Keep the fetcher refreshing this leg's LTP for its whole lifetime. On a
        # re-entry the subscription is already live (kept since the stop-out);
        # re-registering is harmless/idempotent.
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

        new_pos = PaperPosition(
            active=True,
            direction=side,
            symbol=trading_symbol,
            quantity=quantity,
            entry_order_id=str(order_id),
            entry_trade_price=float(option_ltp),
            option_security_id=option_sec_id,
            option_exchange_segment=option_segment,
            option_right=option_right,
            option_strike=option_strike,
            option_expiry=expiry_date,
            option_lot_size=lot_size,
        )
        # Normalized contract snapshot, reused verbatim if this leg is stopped
        # out and later re-entered on the SAME strike.
        normalized = {
            "security_id": option_sec_id,
            "exchange_segment": option_segment,
            "trading_symbol": trading_symbol,
            "option_type": option_right,
            "strike": option_strike,
            "expiry_date": expiry_date,
            "lot_size": lot_size,
        }
        if side == "CE":
            self.ce_pos = new_pos
            self.ce_high_water_premium = float(option_ltp)
            self.ce_last_contract = normalized
            self.ce_awaiting_reentry = False
            if is_reentry:
                self.ce_reentry_count += 1
        else:
            self.pe_pos = new_pos
            self.pe_high_water_premium = float(option_ltp)
            self.pe_last_contract = normalized
            self.pe_awaiting_reentry = False
            if is_reentry:
                self.pe_reentry_count += 1

        self.entry_submit_count += 1
        if is_reentry:
            self.reentry_count += 1
        entry_kind = "RE-ENTRY" if is_reentry else "ENTRY"
        expiry_txt = expiry_date.isoformat() if expiry_date else "NA"
        self.log.info(
            "%s %s | Side=BUY | OptionSymbol=%s | Right=%s | Strike=%.2f | Expiry=%s | "
            "Qty=%s | Spot=%.2f | EntryOptPx=%.2f | PaperRef=%s",
            entry_kind, side, trading_symbol, option_right, option_strike, expiry_txt,
            quantity, spot, option_ltp, order_id,
        )
        self.publish_trade_event({
            "action": "ENTRY",
            "mode": exec_mode,
            "reentry": is_reentry,
            "direction": side,
            "lots": self.lots,
            "lot_size": lot_size,
            "quantity": quantity,
            "spot": spot,
            "expiry": expiry_txt,
            "legs": [{
                "symbol": trading_symbol, "side": "BUY",
                "right": option_right, "strike": option_strike,
                "entry_price": option_ltp,
            }],
        })
        return True

    # ------------------------------------------------------------------
    # Momentum re-entry (AlgoTest "RE MOMENTUM")
    # ------------------------------------------------------------------
    def _manage_reentry(self, side: str) -> None:
        """
        Re-buy a stopped-out leg once its premium rebounds past the momentum
        trigger. No-op unless the leg is flat, armed (`_awaiting_reentry`), and
        under the re-entry cap.

        Trigger: re-enter when the SAME strike's live premium rises to
        >= stop_out_price * (1 + STRANGLE_REENTRY_MOMENTUM_PCT). The strike is
        re-bought from the stashed `_last_contract`, not re-derived from a
        since-moved spot.
        """
        if not STRANGLE_REENTRY_ENABLED:
            return
        leg_pos = self._leg_pos(side)
        if leg_pos.active:
            return  # Already back in a trade.

        awaiting, stop_out_price, contract, reentry_count = self._reentry_state(side)
        if not awaiting or contract is None or stop_out_price <= 0:
            return
        if reentry_count >= STRANGLE_MAX_REENTRIES:
            return  # Cap reached (defensive; arming already enforces this).

        current = self._get_option_ltp(
            str(contract["exchange_segment"]), int(contract["security_id"]), fallback=0.0
        )
        if current <= 0:
            return  # Bad tick; try again next poll.

        trigger = stop_out_price * (1.0 + STRANGLE_REENTRY_MOMENTUM_PCT)
        if current >= trigger:
            self.log.info(
                "%s momentum re-entry | Current=%.2f >= %.2f (stop=%.2f +%.1f%%) | re-entry %d/%d",
                side, current, trigger, stop_out_price,
                STRANGLE_REENTRY_MOMENTUM_PCT * 100.0,
                reentry_count + 1, STRANGLE_MAX_REENTRIES,
            )
            # Spot is only for the entry log line; the strike comes from contract.
            spot = self._get_underlying_spot(fallback=0.0)
            self._buy_leg(side, contract, spot=spot, is_reentry=True)

    def _unsubscribe_armed_legs(self) -> None:
        """
        Drop LTP subscriptions for legs that are flat but still ARMED for
        momentum re-entry (kept subscribed during the session so we could watch
        for the rebound). Called once during end-of-day shutdown. Active legs
        are unsubscribed by `_exit_leg`; legs that never entered were never
        subscribed.
        """
        for side in ("CE", "PE"):
            leg_pos = self._leg_pos(side)
            awaiting, _, contract, _ = self._reentry_state(side)
            if not leg_pos.active and awaiting and contract is not None:
                self.store.unregister_option_subscription(
                    str(contract["exchange_segment"]), int(contract["security_id"])
                )

    # ------------------------------------------------------------------
    # Per-leg stop-loss / trailing-stop management
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_trailing_sl(
        entry_premium: float,
        high_water_premium: float,
        sl_pct: float,
        trail_trigger_pct: float,
        trail_step_pct: float,
    ) -> float:
        """
        Effective stop-loss premium for one leg, given its entry price and the
        high-water premium seen so far. Pure function (no I/O) so it is unit
        tested directly.

        The ladder, with entry E, SL% (e.g. 0.05), trigger X% and step Y%:
            gain%  = max(0, (high_water - E) / E * 100)
            steps  = floor(gain% / X)
            SL     = E * (1 - SL%) + steps * (Y/100) * E

        Worked example (E=100, SL%=5%, X=1, Y=1):
            high_water=100 -> steps 0  -> SL 95   (initial 5% stop)
            high_water=105 -> steps 5  -> SL 100  (breakeven)
            high_water=110 -> steps 10 -> SL 105  (locks +5)
        i.e. once in profit the stop trails a constant 5% under the high-water
        mark, stepped in 1% increments.
        """
        if entry_premium <= 0:
            return 0.0
        gain_pct = max(0.0, (high_water_premium - entry_premium) / entry_premium * 100.0)
        steps = math.floor(gain_pct / trail_trigger_pct) if trail_trigger_pct > 0 else 0
        return entry_premium * (1.0 - sl_pct) + steps * (trail_step_pct / 100.0) * entry_premium

    def _manage_leg(self, side: str) -> None:
        """
        Run the stop-loss / trailing-stop check once for one leg. Called every
        poll while the leg is open; exits the leg if its premium has fallen to
        or below the effective (trailed) stop.
        """
        leg_pos = self._leg_pos(side)
        if not leg_pos.active:
            return

        current = self._get_option_ltp(
            leg_pos.option_exchange_segment,
            leg_pos.option_security_id,
            fallback=leg_pos.entry_trade_price,
        )
        if current <= 0:
            return  # Bad tick; try again next poll.

        # Ratchet the per-leg high-water mark upward only.
        if side == "CE":
            self.ce_high_water_premium = max(self.ce_high_water_premium, current)
            high_water = self.ce_high_water_premium
        else:
            self.pe_high_water_premium = max(self.pe_high_water_premium, current)
            high_water = self.pe_high_water_premium

        effective_sl = self._compute_trailing_sl(
            leg_pos.entry_trade_price, high_water,
            STRANGLE_SL_PCT, STRANGLE_TRAIL_TRIGGER_PCT, STRANGLE_TRAIL_STEP_PCT,
        )
        if current <= effective_sl:
            # Label distinguishes the initial stop from a trailed one in logs.
            label = "TRAILING_SL" if high_water > leg_pos.entry_trade_price else "STOP_LOSS"
            self.log.info(
                "%s %s | CurrentPremium=%.2f <= SL=%.2f | Entry=%.2f | HighWater=%.2f",
                side, label, current, effective_sl, leg_pos.entry_trade_price, high_water,
            )
            # Per-leg stop-outs may arm a momentum re-entry; strategy-wide exits
            # (cutoff / max-loss) go through exit_position and never arm.
            self._exit_leg(side, label, allow_reentry=True)

    # ------------------------------------------------------------------
    # Per-leg exit
    # ------------------------------------------------------------------
    def _exit_leg(self, side: str, reason: str, allow_reentry: bool = False) -> None:
        """
        SELL-to-close one leg and book its PnL. Mirrors
        AtmSingleLegStrategyWorker.exit_position for a single BUY leg, but
        operates on ce_pos / pe_pos.

        `allow_reentry` (True only for per-leg stop-outs) decides whether the
        leg is ARMED for a momentum re-entry: if armed (and under the re-entry
        cap), the LTP subscription is KEPT alive and the stop-out price recorded
        so `_manage_reentry` can watch for the rebound; otherwise the leg is
        unsubscribed and left flat for the day.
        """
        leg_pos = self._leg_pos(side)
        if not leg_pos.active:
            return

        closed = leg_pos
        order_id = self._next_paper_order_id("SELL")
        exit_price = self._get_option_ltp(
            closed.option_exchange_segment,
            closed.option_security_id,
            fallback=closed.entry_trade_price,
        )

        # Real execution (live mode only; no-op True in paper mode).
        real_ok = self._place_real_leg("SELL", {
            "option_type": closed.option_right, "strike": closed.option_strike,
            "expiry": closed.option_expiry, "quantity": closed.quantity,
            "dhan_symbol": closed.symbol,
        })
        exec_mode = self._exec_mode_tag(real_ok)

        # If a LIVE exit did not confirm a fill, keep the leg OPEN for retry /
        # manual square-off rather than flattening the books (same safety rule
        # as the single-leg ATM worker).
        if self.live_trading and not real_ok:
            self.log.error(
                "LIVE EXIT NOT CONFIRMED | %s %s | OptionSymbol=%s | Qty=%s | Reason=%s "
                "| leg kept OPEN for retry/manual square-off.",
                self.strategy_name, side, closed.symbol, closed.quantity, reason,
            )
            self.publish_trade_event({
                "action": "EXIT_FAILED", "mode": exec_mode, "direction": side, "reason": reason,
                "quantity": closed.quantity,
                "legs": [{"symbol": closed.symbol, "side": "SELL",
                          "right": closed.option_right, "strike": closed.option_strike}],
            })
            return

        pnl = (exit_price - closed.entry_trade_price) * closed.quantity
        self.completed_trades += 1
        self.exit_count += 1
        self.realized_pnl += pnl

        pnl_colored = self._color_pnl_text(pnl)
        expiry_txt = closed.option_expiry.isoformat() if closed.option_expiry is not None else "NA"
        self.log.info(
            "EXIT %s | OptionSymbol=%s | Right=%s | Strike=%.2f | Expiry=%s | Qty=%s | "
            "Reason=%s | EntryOptPx=%.2f | ExitOptPx=%.2f | P&L=%.2f | CumPnL=%.2f | PaperRef=%s",
            side, closed.symbol, closed.option_right, closed.option_strike, expiry_txt,
            closed.quantity, reason, closed.entry_trade_price, exit_price, pnl,
            self.realized_pnl, order_id,
        )
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | {self.strategy_name} EXIT {side} | "
            f"OptionSymbol={closed.symbol} | Qty={closed.quantity} | Reason={reason} | "
            f"P&L={pnl_colored} | PaperRef={order_id}"
        )

        self.publish_trade_event({
            "action": "EXIT", "mode": exec_mode,
            "exit_order_failed": bool(self.live_trading and not real_ok),
            "direction": side, "reason": reason,
            "lot_size": closed.option_lot_size, "quantity": closed.quantity,
            "pnl": pnl, "expiry": expiry_txt,
            "legs": [{"symbol": closed.symbol, "side": "SELL",
                      "right": closed.option_right, "strike": closed.option_strike,
                      "entry_price": closed.entry_trade_price, "exit_price": exit_price}],
        })

        # Decide whether to ARM this leg for a momentum re-entry. Only per-leg
        # stop-outs (allow_reentry=True) arm; strategy-wide exits (cutoff /
        # max-loss) never do. Arming keeps the LTP subscription alive so
        # _manage_reentry can watch the same strike for the rebound.
        _, _, _, prior_reentries = self._reentry_state(side)
        reentry_armed = (
            allow_reentry
            and STRANGLE_REENTRY_ENABLED
            and prior_reentries < STRANGLE_MAX_REENTRIES
        )
        if reentry_armed:
            self.log.info(
                "%s armed for momentum re-entry | stop=%.2f | re-buy when premium >= %.2f "
                "(+%.1f%%) | re-entries so far %d/%d",
                side, exit_price, exit_price * (1.0 + STRANGLE_REENTRY_MOMENTUM_PCT),
                STRANGLE_REENTRY_MOMENTUM_PCT * 100.0, prior_reentries, STRANGLE_MAX_REENTRIES,
            )
        else:
            # No re-entry: stop watching this strike.
            self.store.unregister_option_subscription(
                closed.option_exchange_segment, closed.option_security_id,
            )

        if side == "CE":
            self.ce_pos = PaperPosition()
            self.ce_high_water_premium = 0.0
            self.ce_awaiting_reentry = reentry_armed
            self.ce_stop_out_price = exit_price if reentry_armed else 0.0
        else:
            self.pe_pos = PaperPosition()
            self.pe_high_water_premium = 0.0
            self.pe_awaiting_reentry = reentry_armed
            self.pe_stop_out_price = exit_price if reentry_armed else 0.0


# =============================================================================
# SIGNAL GENERATOR PORT WORKERS (ATM single-leg; TradingBot ports)
# =============================================================================
# The thirteen ported strategies share one identical worker lifecycle: resample
# the shared 1-min OHLC to the strategy's derived timeframe, build the strategy
# frame, evaluate the latest candle, and translate ENTER_LONG / ENTER_SHORT /
# EXIT into ATM CE/PE paper trades. Because they differ only by which logic module
# + config + env prefix they use, we build them from a single factory + table
# instead of thirteen copy-pasted classes. Each is fully namespaced by its own env
# prefix (like Goldmine), so every operational knob is independently tunable.
def _signal_gen_ops(prefix: str) -> dict:
    """
    Build one ported strategy's operational knobs from its env prefix.

    Returns the per-strategy poll cadence, resample timeframe, trading window,
    lot size, and daily max-loss cap. Every key is <PREFIX>_<NAME> in .env (e.g.
    SMA_CROSSOVER_POLL_SECONDS), so each strategy is tuned independently. The
    defaults are identical across all thirteen; only the prefix differs.
    """
    starting_capital = _env_float(f"{prefix}_STARTING_CAPITAL", 600000.0)
    return {
        "poll_seconds": _env_int(f"{prefix}_POLL_SECONDS", 5),
        "derived_timeframe_minutes": _env_int(f"{prefix}_DERIVED_TIMEFRAME_MINUTES", 5),
        "trading_start_hour": _env_int(f"{prefix}_TRADING_START_HOUR", 9),
        "trading_start_minute": _env_int(f"{prefix}_TRADING_START_MINUTE", 25),
        "square_off_hour": _env_int(f"{prefix}_SQUARE_OFF_HOUR", 15),
        "square_off_minute": _env_int(f"{prefix}_SQUARE_OFF_MINUTE", 15),
        "lots": _env_int(f"{prefix}_LOTS", 1),
        "max_loss": starting_capital * _env_float(f"{prefix}_DAILY_MAX_LOSS_PCT", 0.03),
    }


def _build_signal_gen_worker_class(
    class_name: str,
    display_name: str,
    logic_module,
    engine_attr: str,
    build_attr: str,
    position_attr: str,
    config,
    env_prefix: str,
):
    """
    Return a concrete AtmSingleLegStrategyWorker subclass for one ported strategy.

    A "worker" is one trading thread. Every poll it does the same four steps via
    the base class: read the shared 1-min candles, build this strategy's frame,
    ask the strategy engine for a decision, and act on it (buy/sell an ATM option).
    All thirteen do those steps identically and differ only by which logic
    module/config/env-prefix they plug in - which is what this factory captures,
    so we avoid thirteen near-identical copy-pasted classes.

    Parameters:
    - class_name / display_name: the worker's Python class name and its log/UI name
      (display_name is what shows on Telegram, e.g. "SMA Crossover").
    - logic_module: the loaded strategy module (e.g. SMA_CROSSOVER_LOGIC).
    - engine_attr / build_attr / position_attr: names of the engine class, the
      indicator-builder function, and the position-context class inside that module.
    - config: the strategy's frozen indicator config object.
    - env_prefix: the .env namespace for this strategy's operational knobs
      (e.g. "SMA_CROSSOVER").
    """
    ops = _signal_gen_ops(env_prefix)

    class _SignalGenWorker(AtmSingleLegStrategyWorker):
        strategy_name = display_name
        timeframe = "1"
        poll_seconds = ops["poll_seconds"]
        lots = ops["lots"]
        max_loss = ops["max_loss"]
        trading_start_hour = ops["trading_start_hour"]
        trading_start_minute = ops["trading_start_minute"]
        square_off_hour = ops["square_off_hour"]
        square_off_minute = ops["square_off_minute"]
        derived_timeframe_minutes = ops["derived_timeframe_minutes"]

        def __init__(self, store, stop_event, broker):
            super().__init__(store, stop_event, broker)
            self.signal_engine = getattr(logic_module, engine_attr)(config)

        def minimum_strategy_rows(self) -> int:
            # Each engine reports its own warm-up requirement.
            return self.signal_engine.minimum_history_bars()

        def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
            ohlc_n = resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)
            return getattr(logic_module, build_attr)(ohlc_n, config)

        def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
            # If we already hold a position, tell the engine about it so it only
            # checks EXIT rules (it will not open a second, overlapping trade).
            position_ctx = None
            if self.pos.active:
                position_ctx = getattr(logic_module, position_attr)(
                    direction=self.pos.direction,
                    entry_underlying=self.pos.entry_underlying,
                    stop_underlying=self.pos.stop_underlying,
                    target_underlying=self.pos.target_underlying,
                )

            # Ask the strategy engine what to do with the latest candle.
            decision = self.signal_engine.evaluate_candle(strategy_frame, position=position_ctx)

            # In a trade -> only honour an EXIT and then stop for this candle.
            if self.pos.active:
                if decision.action == "EXIT":
                    self.exit_position(decision.exit_reason or "SIGNAL_EXIT")
                return

            # Flat -> a LONG buys an ATM CE, a SHORT buys an ATM PE (see enter_position).
            if decision.action == "ENTER_LONG":
                self.enter_position(
                    "LONG",
                    decision.entry_underlying,
                    decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                )
            elif decision.action == "ENTER_SHORT":
                self.enter_position(
                    "SHORT",
                    decision.entry_underlying,
                    decision.stop_underlying,
                    target_underlying=decision.target_underlying,
                )

    _SignalGenWorker.__name__ = class_name
    _SignalGenWorker.__qualname__ = class_name
    return _SignalGenWorker


# (class name, display/log name, logic module, engine attr, build attr,
#  position attr, config, env prefix)
_SIGNAL_GEN_WORKER_SPECS = [
    ("SMACrossoverWorker", "SMA Crossover", SMA_CROSSOVER_LOGIC,
     "SMACrossoverSignalEngine", "build_sma_crossover_with_indicators",
     "SMACrossoverPositionContext", SMA_CROSSOVER_CONFIG, "SMA_CROSSOVER"),
    ("BollingerBandsWorker", "Bollinger Bands", BOLLINGER_BANDS_LOGIC,
     "BollingerBandsSignalEngine", "build_bollinger_bands_with_indicators",
     "BollingerBandsPositionContext", BOLLINGER_BANDS_CONFIG, "BOLLINGER_BANDS"),
    ("KeltnerSqueezeWorker", "Keltner Squeeze", KELTNER_SQUEEZE_LOGIC,
     "KeltnerSqueezeSignalEngine", "build_keltner_squeeze_with_indicators",
     "KeltnerSqueezePositionContext", KELTNER_SQUEEZE_CONFIG, "KELTNER_SQUEEZE"),
    ("MeanReversionZscoreWorker", "Mean Reversion Zscore", MEAN_REVERSION_ZSCORE_LOGIC,
     "MeanReversionZscoreSignalEngine", "build_mean_reversion_zscore_with_indicators",
     "MeanReversionZscorePositionContext", MEAN_REVERSION_ZSCORE_CONFIG, "MEAN_REVERSION_ZSCORE"),
    ("MLEnsembleWorker", "ML Ensemble", ML_ENSEMBLE_LOGIC,
     "MLEnsembleSignalEngine", "build_ml_ensemble_with_indicators",
     "MLEnsemblePositionContext", ML_ENSEMBLE_CONFIG, "ML_ENSEMBLE"),
    ("MultiTimeframeWorker", "Multi Timeframe", MULTI_TIMEFRAME_LOGIC,
     "MultiTimeframeSignalEngine", "build_multi_timeframe_with_indicators",
     "MultiTimeframePositionContext", MULTI_TIMEFRAME_CONFIG, "MULTI_TIMEFRAME"),
    ("OpeningRangeBreakoutWorker", "Opening Range Breakout", OPENING_RANGE_BREAKOUT_LOGIC,
     "OpeningRangeBreakoutSignalEngine", "build_opening_range_breakout_with_indicators",
     "OpeningRangeBreakoutPositionContext", OPENING_RANGE_BREAKOUT_CONFIG, "OPENING_RANGE_BREAKOUT"),
    ("ParabolicSARWorker", "Parabolic SAR", PARABOLIC_SAR_LOGIC,
     "ParabolicSARSignalEngine", "build_parabolic_sar_with_indicators",
     "ParabolicSARPositionContext", PARABOLIC_SAR_CONFIG, "PARABOLIC_SAR"),
    ("RSIDivergenceWorker", "RSI Divergence", RSI_DIVERGENCE_LOGIC,
     "RSIDivergenceSignalEngine", "build_rsi_divergence_with_indicators",
     "RSIDivergencePositionContext", RSI_DIVERGENCE_CONFIG, "RSI_DIVERGENCE"),
    ("RSIReversalWorker", "RSI Reversal", RSI_REVERSAL_LOGIC,
     "RSIReversalSignalEngine", "build_rsi_reversal_with_indicators",
     "RSIReversalPositionContext", RSI_REVERSAL_CONFIG, "RSI_REVERSAL"),
    ("StochasticOscillatorWorker", "Stochastic Oscillator", STOCHASTIC_LOGIC,
     "StochasticOscillatorSignalEngine", "build_stochastic_oscillator_with_indicators",
     "StochasticOscillatorPositionContext", STOCHASTIC_CONFIG, "STOCHASTIC"),
    ("SupertrendPortWorker", "Supertrend", SUPERTREND_PORT_LOGIC,
     "SupertrendSignalEngine", "build_supertrend_with_indicators",
     "SupertrendPositionContext", SUPERTREND_PORT_CONFIG, "SUPERTREND_PORT"),
    ("VolatilityBreakoutWorker", "Volatility Breakout", VOLATILITY_BREAKOUT_LOGIC,
     "VolatilityBreakoutSignalEngine", "build_volatility_breakout_with_indicators",
     "VolatilityBreakoutPositionContext", VOLATILITY_BREAKOUT_CONFIG, "VOLATILITY_BREAKOUT"),
]

# Concrete worker classes, ready to instantiate in main().
SIGNAL_GEN_WORKERS = [_build_signal_gen_worker_class(*spec) for spec in _SIGNAL_GEN_WORKER_SPECS]


# Links each strategy to the prefix used for ITS .env settings.
#
# Each strategy's knobs in .env share a prefix, e.g. Renko uses RENKO_LOTS,
# RENKO_MAX_LOSS, and (the one that matters here) RENKO_LIVE_TRADING. This dict
# answers "given a running worker, which prefix do I look up?" so we can find the
# right <PREFIX>_LIVE_TRADING flag for it. Key = the worker's strategy_name.
#
# The first 11 entries are typed out by hand (their strategy_name is set on the
# class); the 13 ported strategies are added automatically from their build
# specs (spec[1] is the name, spec[7] is the prefix) so the two lists can never
# drift out of sync.
STRATEGY_ENV_PREFIX = {
    "Renko": "RENKO",
    "EMA": "EMA",
    "HeikinAshi": "HEIKIN",
    "ProfitShooter": "PROFIT_SHOOTER",
    "Goldmine": "GOLDMINE",
    "MoneyMachine": "MONEY_MACHINE",
    "OpeningStrike": "OPENING_STRIKE",
    "CPR": "CPR",
    "CPRAlgo3": "CPR_ALGO3",
    "SupertrendBullish": "BULLISH",
    "DonchianBearish": "BEARISH",
    "Delta20Hedged": "DELTA20",
    "LongStrangle": "STRANGLE",
    **{spec[1]: spec[7] for spec in _SIGNAL_GEN_WORKER_SPECS},
}


# -----------------------------------------------------------------------------
# SL Hunting AI Agent worker (optional, LLM brain)
# -----------------------------------------------------------------------------
# Unlike the deterministic ATM workers above, this one asks a Claude agent for a
# decision once per completed N-min bar. The agent ACTS by calling its single
# env-selected order tool, which routes through a MasterWorkerExecutor back into
# this worker's own safe enter_position / exit_position -- so paper-vs-live, the
# broker choice (LIVE_BROKER), max-loss, square-off and Telegram all behave
# exactly like every other strategy. Defined only when the optional agent loaded;
# otherwise SLHuntingAIWorker stays None and main() simply skips it.
SLHuntingAIWorker = None
if SL_HUNTING_AVAILABLE:
    _sl_hunting_ops = _signal_gen_ops("SL_HUNTING")
    # Dynamic position sizing: ~Rs.2500 worst-case risk per trade (the same house
    # convention/default as Profit Shooter). The agent does NOT choose lots; this is
    # computed from the agent's underlying stop distance.
    SL_HUNTING_RISK_BUDGET = _env_float("SL_HUNTING_RISK_BUDGET", 2500.0)
    # Trade journal (v3): record each trade's entry context + exit outcome to a
    # gitignored JSONL the reflection coach reads. Best-effort; never blocks trading.
    SL_HUNTING_JOURNAL_ENABLED = _env_bool("SL_HUNTING_JOURNAL_ENABLED", True)
    SL_HUNTING_JOURNAL_PATH = _env_str(
        "SL_HUNTING_JOURNAL_PATH", str(ROOT_DIR / "Backtest Outputs" / "sl_hunting_journal.jsonl")
    )
    # Decision log: record EVERY agent decision (HOLD included) to a SEPARATE gitignored
    # JSONL so the operator can review what the agent decided each bar. Kept apart from
    # the trade journal above so the coach's win/loss stats aren't diluted by no-trade
    # bars. Best-effort; never blocks trading.
    SL_HUNTING_DECISIONS_ENABLED = _env_bool("SL_HUNTING_DECISIONS_ENABLED", True)
    SL_HUNTING_DECISIONS_PATH = _env_str(
        "SL_HUNTING_DECISIONS_PATH", str(ROOT_DIR / "Backtest Outputs" / "sl_hunting_decisions.jsonl")
    )
    # Learned lessons (v3): inject APPROVED lessons into the agent's prompt ONLY when
    # explicitly enabled (default OFF) -- human-gated + paper-first. The live store
    # lives in the agent folder so it ships with the strategy.
    SL_HUNTING_LESSONS_ENABLED = _env_bool("SL_HUNTING_LESSONS_ENABLED", False)
    SL_HUNTING_LESSONS_PATH = _env_str("SL_HUNTING_LESSONS_PATH", str(SL_HUNTING_DIR / "lessons.json"))

    class SLHuntingAIWorker(AtmSingleLegStrategyWorker):  # noqa: F811 - optional definition
        """ATM single-leg worker whose entries/exits come from the SL Hunting agent."""

        strategy_name = "SL Hunting AI"
        timeframe = "1"
        poll_seconds = _sl_hunting_ops["poll_seconds"]
        lots = _sl_hunting_ops["lots"]
        max_loss = _sl_hunting_ops["max_loss"]
        trading_start_hour = _sl_hunting_ops["trading_start_hour"]
        trading_start_minute = _sl_hunting_ops["trading_start_minute"]
        square_off_hour = _sl_hunting_ops["square_off_hour"]
        square_off_minute = _sl_hunting_ops["square_off_minute"]
        # SL Hunting runs on the 1-MINUTE timeframe by design (the method's "daily trades"
        # are 1-min; 5-min is only the opening-setup nuance), so override the shared
        # 5-min default. NOTE: the agent makes one LLM/subscription call PER completed bar,
        # so 1-min is ~5x the calls/usage of 5-min — raise this if usage/cost is a concern.
        derived_timeframe_minutes = _env_int("SL_HUNTING_DERIVED_TIMEFRAME_MINUTES", 1)
        # Stop opening NEW positions at/after this time (default 12:00), mirroring the
        # manual "no fresh trades after noon" rule. This does NOT square off open
        # positions -- only the existing square_off_* gate (15:15) force-closes; exits
        # (stop/target, AI exit, max-loss) keep working. See process_strategy_frame.
        no_new_entry_hour = _env_int("SL_HUNTING_NO_NEW_ENTRY_HOUR", 12)
        no_new_entry_minute = _env_int("SL_HUNTING_NO_NEW_ENTRY_MINUTE", 0)

        def __init__(self, store, stop_event, broker):
            super().__init__(store, stop_event, broker)
            # Optional learned-lessons block (loaded ONCE at construction so the prompt
            # prefix stays stable per session — preserves prompt caching). Gated + off
            # by default; only APPROVED lessons are in the live store.
            lessons_block = ""
            if SL_HUNTING_LESSONS_ENABLED:
                try:
                    lessons_block = SL_HUNTING_LESSONS_MODULE.format_lessons(
                        SL_HUNTING_LESSONS_MODULE.load_lessons(SL_HUNTING_LESSONS_PATH)
                    )
                    if lessons_block:
                        self.log.info("SL Hunting: injected %d learned lesson chars.", len(lessons_block))
                except Exception as exc:  # noqa: BLE001 - lessons are advisory, never fatal
                    self.log.warning("SL Hunting lessons load failed: %s", exc)
            self.agent = SL_HUNTING_AGENT_MODULE.SLHuntingAgent(
                model=_env_str("SL_HUNTING_MODEL", "claude-opus-4-8"),
                fast_mode=_env_bool("SL_HUNTING_FAST_MODE", False),
                indicator_config=SL_HUNTING_INDICATOR_CONFIG,
                lessons_block=lessons_block,
            )
            # The order tool routes through this executor into our own safe methods.
            self._executor = SL_HUNTING_EXECUTOR_MODULE.MasterWorkerExecutor(self)
            # Throttle: consult the LLM only once per NEW completed bar (not per poll).
            self._last_decision_bar = None
            # Optional BankNIFTY cross-confirmation, fetched per bar via self.broker
            # exactly like CPR Algo 3 fetches its observation legs.
            self._use_bnf = _env_bool("SL_HUNTING_USE_BNF", True)
            # Trade journal (v3): open a row on entry here, close it in after_exit
            # (the universal post-close hook, so EVERY exit path is captured).
            self._journal = (
                SL_HUNTING_JOURNAL_MODULE.TradeJournal(SL_HUNTING_JOURNAL_PATH)
                if SL_HUNTING_JOURNAL_ENABLED else None
            )
            self._open_trade_id = None
            self._entry_realized_pnl = 0.0
            # Separate per-bar decision log (HOLD included); None disables it.
            self._decisions_path = (
                SL_HUNTING_DECISIONS_PATH if SL_HUNTING_DECISIONS_ENABLED else None
            )

        def _journal_open_row(self, decision, nifty_df, bnf_df) -> None:
            """Open a journal row for a trade the agent just entered (best-effort)."""
            try:
                lot_size = max(int(getattr(self.pos, "option_lot_size", 0)), 1)
                entry = SL_HUNTING_JOURNAL_MODULE.make_entry_record(
                    direction=getattr(self.pos, "direction", ""),
                    setup=decision.setup, confidence=decision.confidence,
                    reasoning=decision.reasoning,
                    stop=float(getattr(self.pos, "stop_underlying", 0.0)),
                    target=float(getattr(self.pos, "target_underlying", 0.0)),
                    entry_underlying=float(getattr(self.pos, "entry_underlying", 0.0)),
                    lots=int(getattr(self.pos, "quantity", 0)) // lot_size,
                    nifty_df=nifty_df, bnf_df=bnf_df,
                )
                self._open_trade_id = self._journal.open_trade(entry)
                self._entry_realized_pnl = self.realized_pnl
            except Exception as exc:  # noqa: BLE001 - journaling must never disturb trading
                self.log.warning("SL Hunting journal open failed: %s", exc)

        def after_exit(self, closed_position, reason: str) -> None:
            """Universal post-close hook (fires for AI exit, stop/target, max-loss,
            square-off). Close the open journal row with the underlying outcome."""
            super().after_exit(closed_position, reason)
            if self._journal is None or self._open_trade_id is None:
                return
            try:
                entry_u = float(getattr(closed_position, "entry_underlying", 0.0))
                direction = getattr(closed_position, "direction", "")
                spot = self._get_underlying_spot(fallback=entry_u)
                points = (spot - entry_u) if direction == "LONG" else (entry_u - spot)
                lot_size = max(int(getattr(closed_position, "option_lot_size", 0)), 1)
                self._journal.close_trade(self._open_trade_id, {
                    "exit_underlying": round(spot, 2),
                    "exit_reason": reason,
                    "points": round(points, 2),
                    "option_pnl": round(self.realized_pnl - self._entry_realized_pnl, 2),
                    "lots": int(getattr(closed_position, "quantity", 0)) // lot_size,
                })
            except Exception as exc:  # noqa: BLE001
                self.log.warning("SL Hunting journal close failed: %s", exc)
            finally:
                self._open_trade_id = None

        def minimum_strategy_rows(self) -> int:
            # Enough derived bars for swings / fibo / structure, with a margin.
            return max(40, int(getattr(SL_HUNTING_INDICATOR_CONFIG, "swing_lookback", 120)) // 2)

        def _compute_entry_lots(self, entry_underlying: float, stop_underlying: float, lot_size: int) -> int:
            """Dynamic risk-based sizing (~SL_HUNTING_RISK_BUDGET, default Rs.2500),
            from the agent's UNDERLYING stop distance. Shares ONE implementation with
            the standalone executor via `risk_based_lots`, and matches the Profit
            Shooter sizer (ceil, minimum 1 lot — a setup trades at >=1 lot or not at all).
            """
            lots = SL_HUNTING_EXECUTOR_MODULE.risk_based_lots(
                entry_underlying, stop_underlying, lot_size, SL_HUNTING_RISK_BUDGET, self.lots
            )
            self.log.info(
                "SL Hunting dynamic sizing: entry=%.2f stop=%.2f lot_size=%s risk_budget=%.0f -> lots=%s qty=%s.",
                entry_underlying, stop_underlying, lot_size, SL_HUNTING_RISK_BUDGET, lots, lots * lot_size,
            )
            return lots

        def build_strategy_frame(self, ohlc: pd.DataFrame) -> pd.DataFrame:
            return resample_ohlc_from_1m(ohlc, self.derived_timeframe_minutes)

        def process_strategy_frame(self, strategy_frame: pd.DataFrame) -> None:
            if strategy_frame is None or strategy_frame.empty:
                return

            # Enforce the agent's UNDERLYING stop/target on EVERY poll (not just once
            # per bar), so the ~Rs.2500 risk is actually bounded between the agent's
            # bar-cadence decisions -- the agent only re-decides per completed bar, so
            # we cannot wait for it to call EXIT. Mirrors CPR Algo 3's spot stop/target
            # check; _get_underlying_spot reads the live LTP cache.
            if self.pos.active:
                spot = self._get_underlying_spot(fallback=0.0)
                if spot > 0:
                    hit = SL_HUNTING_EXECUTOR_MODULE.stop_or_target_hit(
                        self.pos.direction, self.pos.stop_underlying, self.pos.target_underlying, spot
                    )
                    if hit:
                        self.log.info("SL Hunting %s hit at spot=%.2f; exiting.", hit, spot)
                        self.exit_position(hit)
                        return

            # No NEW positions at/after the entry cutoff (default 12:00). If we are FLAT
            # past the cutoff, don't consult the agent at all -- it could only ENTER, and
            # skipping it also saves the per-bar LLM call all afternoon. An OPEN position
            # is unaffected: the per-poll stop/target above, a discretionary AI exit
            # below, max-loss, and the 15:15 square-off all still run, so exits and the
            # force-square-off gate are NEVER blocked by this.
            if not self.pos.active and is_after_time(self.no_new_entry_hour, self.no_new_entry_minute):
                return

            # Only call the agent when a NEW bar has completed since last time.
            latest_bar = strategy_frame["timestamp"].iloc[-1]
            if latest_bar == self._last_decision_bar:
                return
            self._last_decision_bar = latest_bar

            # Optional BankNIFTY for the agent's NF/BNF cross-confirmation. Fetched
            # on-demand via the shared broker (same fetch_index_1m_ohlc path as the
            # central fetcher / CPR Algo 3). Any failure -> NIFTY-only this bar.
            bnf_candles = None
            if self._use_bnf:
                try:
                    bnf_1m = self.broker.fetch_index_1m_ohlc(
                        BANKNIFTY_INDEX_SECURITY_ID,
                        BANKNIFTY_INDEX_EXCHANGE_SEGMENT,
                        BANKNIFTY_INDEX_INSTRUMENT_TYPE,
                    )
                    bnf_candles = resample_ohlc_from_1m(bnf_1m, self.derived_timeframe_minutes)
                except Exception as exc:
                    self.log.warning(
                        "BankNIFTY fetch failed (%s); SL Hunting goes NIFTY-only this bar.", exc
                    )

            # The agent acts via the order tool DURING decide() (it calls our
            # enter_position / exit_position through the executor); the returned
            # decision is the agent's own record of what it did, for the log.
            was_active = self.pos.active
            decision = self.agent.decide(
                strategy_frame,
                self._executor,
                bnf_candles=bnf_candles,
                live_active=bool(self.live_trading),
                broker=LIVE_BROKER,
            )
            if decision.action != "HOLD":
                self.log.info(
                    "SL Hunting AI: %s (conf=%d, setup=%s) stop=%s target=%s :: %s",
                    decision.action, decision.confidence, decision.setup,
                    decision.stop, decision.target, decision.reasoning,
                )
            # Decision log: persist EVERY decision (HOLD included) to its own JSONL so the
            # operator can review what the agent decided each bar. Best-effort.
            if self._decisions_path is not None:
                try:
                    SL_HUNTING_JOURNAL_MODULE.append_decision(
                        self._decisions_path,
                        SL_HUNTING_JOURNAL_MODULE.make_decision_record(
                            decision, strategy_frame, bnf_candles
                        ),
                    )
                except Exception as exc:  # never let logging disturb trading
                    self.log.warning("SL Hunting decision-log failed: %s", exc)
            # Journal a fresh entry (flat -> active during decide). Exits are journaled
            # in after_exit, so EVERY close path is captured there.
            if (self._journal is not None and self._open_trade_id is None
                    and not was_active and self.pos.active):
                self._journal_open_row(decision, strategy_frame, bnf_candles)

    SLHuntingAIWorker.__name__ = "SLHuntingAIWorker"
    SLHuntingAIWorker.__qualname__ = "SLHuntingAIWorker"
    # Wire the live-trading flag lookup (same pattern as every other strategy).
    STRATEGY_ENV_PREFIX["SL Hunting AI"] = "SL_HUNTING"


# =============================================================================
# END-OF-DAY HELPERS
# =============================================================================
def _refresh_instrument_master_for_next_day() -> None:
    """
    Pull a fresh DhanHQ scrip master and save it date-stamped for tomorrow.

    Lifecycle:
    1. Build the target path `Dependencies/all_instrument <tomorrow>.csv`.
       Tomorrow's date (calendar day, not next trading day) is used so the
       file is in place before the next scheduled run, whichever day of the
       week that turns out to be.
    2. Stream the download to a sibling `*.part` file. We never write
       directly to the final path because a half-finished download whose
       process died would otherwise look like a complete file to the next
       run's glob.
    3. `os.replace(tmp, final)` to publish atomically.
    4. Sweep every OTHER `all_instrument *.csv` in the same folder so the
       glob only sees one master going forward. The just-written file is
       excluded from the sweep by an explicit resolved-path comparison.

    Failure handling: any exception is swallowed and logged. We must NOT
    crash `main()` here -- the trading session has already finished by the
    time we run -- and we must NOT have already deleted the previous file,
    which is why deletion happens last, after the successful `os.replace`.
    """
    try:
        next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        deps_dir = ROOT_DIR / "Dependencies"
        deps_dir.mkdir(parents=True, exist_ok=True)
        new_path = deps_dir / f"all_instrument {next_day}.csv"
        tmp_path = new_path.with_name(new_path.name + ".part")

        logger.info(
            "Refreshing instrument master from %s -> %s",
            DHAN_SCRIP_MASTER_URL,
            new_path.name,
        )

        # Stream so a ~30-50 MB CSV does not balloon memory. (15s connect,
        # 120s per-chunk read) covers a slow link without hanging forever.
        with requests.get(
            DHAN_SCRIP_MASTER_URL,
            timeout=(15.0, 120.0),
            stream=True,
        ) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)

        os.replace(tmp_path, new_path)

        # Sweep every other all_instrument *.csv. Compare resolved paths so
        # case differences and relative/absolute mixing cannot accidentally
        # delete the file we just wrote.
        new_resolved = new_path.resolve()
        for old in glob.glob(str(deps_dir / "all_instrument *.csv")):
            if Path(old).resolve() == new_resolved:
                continue
            try:
                os.remove(old)
                logger.info("Removed stale instrument master: %s", os.path.basename(old))
            except OSError as exc:
                logger.warning("Could not delete old instrument master %s: %s", old, exc)

        logger.info("Instrument master refresh complete: %s", new_path.name)
    except Exception as exc:
        logger.warning(
            "Instrument master refresh failed (existing file preserved): %s", exc
        )


# =============================================================================
# TELEGRAM TRADE NOTIFIER
# =============================================================================
def _format_inr(value) -> str:
    """Format a rupee amount with an explicit sign and thousands separators."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}₹{abs(amount):,.2f}"


def _format_eod_summary(event: dict) -> str:
    """
    Build the end-of-day cumulative P&L message: one line per strategy worker
    (sorted best to worst) plus a grand total. Sent once, after every worker
    has exited cleanly for the day.
    """
    mode = html.escape(str(event.get("mode", "PAPER")))
    ts = html.escape(str(event.get("ts", "")))
    rows = event.get("rows") or []
    lines = [f"\U0001F4CA <b>End-of-Day P&amp;L</b> [{mode}]"]
    if ts:
        lines.append(f"<i>{ts}</i>")
    for row in sorted(rows, key=lambda r: _safe_float(r.get("pnl", 0.0), 0.0), reverse=True):
        strat = html.escape(str(row.get("strategy", "?")))
        trades = _to_int_safe(row.get("trades", 0), 0)
        lines.append(f"{strat}: <b>{_format_inr(row.get('pnl', 0.0))}</b> ({trades} trade(s))")
    lines.append("──────────")
    lines.append(
        f"<b>Total: {_format_inr(event.get('total_pnl', 0.0))} "
        f"({_to_int_safe(event.get('total_trades', 0), 0)} trade(s))</b>"
    )
    return "\n".join(lines)


def format_trade_message(event: dict) -> str:
    """
    Build the Telegram message body for one trade event.

    Handles both single-leg (one element in `legs`) and hedged (two elements)
    trades from a single unified schema. Every message carries the fields asked
    for the group feed: the exact option instrument(s), the lot size / quantity,
    the exact entry price, the exact exit price (on exits), the realized P&L
    (on exits) and the strategy name.
    """
    action = str(event.get("action", "")).upper()
    if action == "EOD_SUMMARY":
        return _format_eod_summary(event)
    strategy = html.escape(str(event.get("strategy", "?")))
    direction = html.escape(str(event.get("direction", "")))
    mode = html.escape(str(event.get("mode", "PAPER")))
    legs = event.get("legs") or []

    if action == "ENTRY":
        # Rocket = entry-specific. The green/red circles are reserved for the
        # exit P&L sign (below), so entries never use a coloured dot.
        header = f"\U0001F680 <b>ENTRY</b> · {strategy}"
    elif action == "EXIT":
        # Chequered flag = exit-specific. Profit/loss is signalled separately by
        # the coloured dot on the P&L line.
        header = f"\U0001F3C1 <b>EXIT</b> · {strategy}"
        reason = html.escape(str(event.get("reason", "")))
        if reason:
            header += f" ({reason})"
    else:
        header = f"<b>{html.escape(action)}</b> · {strategy}"

    lines = [header]
    if direction:
        lines.append(f"Direction: <b>{direction}</b>")

    for leg in legs:
        symbol = html.escape(str(leg.get("symbol", "?")))
        side = html.escape(str(leg.get("side", "")))
        prefix = f"{side} " if side else ""
        lines.append(f"Instrument: <b>{prefix}{symbol}</b>")
        entry_price = leg.get("entry_price")
        exit_price = leg.get("exit_price")
        if entry_price is not None and exit_price is not None:
            lines.append(
                f"  Entry: ₹{float(entry_price):.2f} → Exit: ₹{float(exit_price):.2f}"
            )
        elif entry_price is not None:
            lines.append(f"  Entry: ₹{float(entry_price):.2f}")

    quantity = event.get("quantity")
    lot_size = event.get("lot_size")
    lots = event.get("lots")
    if quantity is not None:
        size_line = f"Qty: <b>{quantity}</b>"
        if lots is not None and lot_size:
            size_line += f" ({lots} lot(s) × {lot_size})"
        elif lot_size:
            size_line += f" (lot size {lot_size})"
        lines.append(size_line)

    if action == "EXIT" and event.get("pnl") is not None:
        # Coloured dot reserved for the exit outcome: green = profit, red =
        # loss, white = breakeven. Entry/exit themselves use the rocket and
        # chequered-flag emojis in the header, never a coloured dot.
        pnl_value = _safe_float(event["pnl"], 0.0)
        if pnl_value > 0:
            pnl_dot = "\U0001F7E2"
        elif pnl_value < 0:
            pnl_dot = "\U0001F534"
        else:
            pnl_dot = "\U000026AA"
        lines.append(f"{pnl_dot} P&amp;L: <b>{_format_inr(event['pnl'])}</b>")

    ts = html.escape(str(event.get("ts", "")))
    lines.append(f"<i>[{mode}] {ts}</i>")
    return "\n".join(lines)


class TelegramMessageWorker(threading.Thread):
    """
    Posts trade events to a Telegram group/channel, one message per entry/exit.

    Design notes:
    - Runs on its own daemon thread so Telegram's network latency or downtime
      never touches the trading threads. Workers only enqueue plain dicts via
      `BasePaperStrategyWorker.publish_trade_event`.
    - Polls the queue with a short timeout so Ctrl+C is honoured promptly, then
      flushes whatever is still queued before exiting so the final exit
      messages of the day are not lost.
    - If the token / chat id is blank it logs once and drops messages instead
      of crashing, so the runner still works with notifications half-configured.
    """

    def __init__(
        self,
        event_queue: "queue.Queue",
        stop_event: threading.Event,
        bot_token: str,
        chat_id: str,
    ):
        super().__init__(name="TelegramThread", daemon=True)
        self.event_queue = event_queue
        self.stop_event = stop_event
        self.bot_token = str(bot_token or "")
        self.chat_id = str(chat_id or "")
        self.log = logging.getLogger(f"{LOGGER_NAME}.telegram")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.sent_count = 0
        self.failed_count = 0

    def run(self) -> None:
        if not self.bot_token or not self.chat_id:
            self.log.warning(
                "Telegram notifier started without a bot token / chat id; messages "
                "will be dropped. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in "
                "Dependencies/.env to enable."
            )
        self.log.info("Starting Telegram notifier worker (chat_id=%s).", self.chat_id or "UNSET")
        while not self.stop_event.is_set():
            try:
                event = self.event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._handle_event(event)
        # Flush anything still queued so end-of-day exits are not lost.
        self._drain_remaining()
        self.log.info(
            "Telegram notifier exited. Sent=%d Failed=%d.", self.sent_count, self.failed_count
        )

    def _drain_remaining(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_event(event)

    def _handle_event(self, event: dict) -> None:
        try:
            text = format_trade_message(event)
        except Exception as exc:
            self.failed_count += 1
            self.log.warning("Could not format trade event %s: %s", event, exc)
            return
        self._send(text)

    def _send(self, text: str) -> None:
        if not self.bot_token or not self.chat_id:
            return
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        for attempt in range(1, 4):
            try:
                response = requests.post(self.api_url, json=payload, timeout=10)
                if response.status_code == 200:
                    self.sent_count += 1
                    return
                self.log.warning(
                    "Telegram API returned HTTP %s on attempt %d: %s",
                    response.status_code,
                    attempt,
                    response.text[:300],
                )
            except Exception as exc:
                self.log.warning("Telegram send error on attempt %d: %s", attempt, exc)
            if attempt < 3:
                self.stop_event.wait(1.5 * attempt)
        self.failed_count += 1


def _publish_eod_summary(workers, event_queue) -> None:
    """
    Log the cumulative paper P&L per strategy worker and, when Telegram is
    enabled, enqueue an end-of-day summary for the notifier to post.

    Called once on a clean end of day (every worker has exited on its own via
    square-off / max-loss, so each `realized_pnl` is final). The Telegram
    worker is still alive at this point, so it consumes the event before the
    shutdown join flushes it. Delivery is a no-op when `event_queue` is None
    (notifications disabled); the local log line is still written.
    """
    rows = []
    total_pnl = 0.0
    total_trades = 0
    for worker in workers:
        pnl = _safe_float(getattr(worker, "realized_pnl", 0.0), 0.0)
        trades = _to_int_safe(getattr(worker, "completed_trades", 0), 0)
        rows.append({"strategy": worker.strategy_name, "pnl": pnl, "trades": trades})
        total_pnl += pnl
        total_trades += trades

    logger.info(
        "END-OF-DAY cumulative paper P&L across %d workers = %.2f over %d trade(s).",
        len(workers),
        total_pnl,
        total_trades,
    )
    if event_queue is None:
        return
    try:
        event_queue.put_nowait(
            {
                "action": "EOD_SUMMARY",
                "rows": rows,
                "total_pnl": total_pnl,
                "total_trades": total_trades,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "PAPER",
            }
        )
    except Exception:
        logger.warning("Could not enqueue the end-of-day Telegram summary (queue full?).")


# =============================================================================
# END-OF-DAY GOOGLE SHEET P&L (per-strategy, with month backfill)
# =============================================================================
# After every worker has exited on a clean end of day, parse the day's log for
# each strategy's realised P&L and write it into the tracker Google Sheet: rows
# are strategies (column A), columns are calendar days of the month. Today's cell
# is (over)written; earlier-this-month cells that are still blank are backfilled
# from the (append-mode) log, so a missed day fills itself in on a later run.
#
# Auth is OAuth "user token" via gspread: set GSHEET_ID and point
# GSHEET_OAUTH_CLIENT_FILE at a Desktop OAuth client JSON; the first run opens a
# browser for consent and caches the token at GSHEET_OAUTH_TOKEN_FILE. The whole
# step is a guarded no-op (logs a warning) if unconfigured or on any error, so it
# never disturbs shutdown.

# Worker strategy_name -> exact sheet row label (column A). A strategy whose row
# is absent is skipped with a warning (e.g. split the sheet's "Stochastic
# Supertrend" row into the two labels below so all strategies map 1:1).
_PNL_SHEET_ROW_LABELS = {
    "Renko": "Renko Strategy",
    "EMA": "EMA Strategy",
    "HeikinAshi": "Heikin Ashi Strategy",
    "ProfitShooter": "Profit Shooter Strategy",
    "Goldmine": "Goldmine Strategy",
    "MoneyMachine": "Money Machine Strategy",
    "OpeningStrike": "Opening Strike PCR VWAP ATR Strategy",
    "CPR": "CPR Strategy",
    "CPRAlgo3": "CPR Algo 3 Strategy",
    "SMA Crossover": "SMA Crossover Strategy",
    "Bollinger Bands": "Bollinger Bands Strategy",
    "Keltner Squeeze": "Keltner Squeeze Strategy",
    "Mean Reversion Zscore": "Mean Reversion Z-score Strategy",
    "ML Ensemble": "ML Ensemble Strategy",
    "Multi Timeframe": "Multi-Timeframe Strategy",
    "Opening Range Breakout": "Opening Range Breakout Strategy",
    "Parabolic SAR": "Parabolic SAR Strategy",
    "RSI Divergence": "RSI Divergence Strategy",
    "RSI Reversal": "RSI Reversal Strategy",
    "Stochastic Oscillator": "Stochastic Oscillator Strategy",
    "Supertrend": "Supertrend Strategy",
    "Volatility Breakout": "Volatility Breakout Strategy",
    "SupertrendBullish": "Supertrend Bullish Strategy",
    "DonchianBearish": "Donchian Bearish Strategy",
    "Delta20Hedged": "Delta 0.2 Hedged Spread Strategy",
    "LongStrangle": "Long Strangle Strategy",
    # Optional opt-in 27th worker (SL_HUNTING_ENABLED); maps only when it runs.
    "SL Hunting AI": "SL Hunting AI Agent Strategy",
}

_PNL_LOG_LINE_RE = re.compile(r"RealizedPnL=(-?\d+(?:\.\d+)?)")
# Only Paper-summary lines inside this window are used for the sheet. Covers
# max-loss during the session (from 9:15) through the last square-off (15:20)
# while ignoring after-market test runs that log fresh summaries at 17:00+.
_PNL_LOG_WINDOW_START = (9, 15)
_PNL_LOG_WINDOW_END = (15, 21)


def _normalize_pnl_strategy_name(name: str) -> str:
    """Strip legacy 'Misc ' prefix from signal-generator thread names."""
    # Older runs logged the ported strategies as "Misc SMA Crossover", etc.;
    # newer runs drop that prefix. Removing it here means both old and new log
    # lines resolve to the same key in _PNL_SHEET_ROW_LABELS.
    return name[5:].strip() if name.startswith("Misc ") else name


def _asctime_in_pnl_window(asctime: str) -> bool:
    """True when the log timestamp falls in the trading-day P&L window (inclusive)."""
    try:
        # A log timestamp looks like "2026-06-09 15:18:07,420". Characters 11..15
        # are exactly the "HH:MM" portion, so slice those out and split on ":".
        time_part = asctime.strip()[11:16]
        hour, minute = (int(part) for part in time_part.split(":", 1))
    except (ValueError, IndexError):
        # If the timestamp is malformed we can't place it, so treat it as outside
        # the window (safer to skip a line than to mis-date a P&L figure).
        return False
    # Turn the window edges and the line's own time into "minutes since midnight"
    # so the in-window test is a single, easy integer range comparison.
    start_minutes = _PNL_LOG_WINDOW_START[0] * 60 + _PNL_LOG_WINDOW_START[1]
    end_minutes = _PNL_LOG_WINDOW_END[0] * 60 + _PNL_LOG_WINDOW_END[1]
    current_minutes = hour * 60 + minute
    return start_minutes <= current_minutes <= end_minutes


def _parse_eod_pnl_by_day(log_path) -> dict:
    """
    Parse the runner's log for each strategy's end-of-day realised P&L per day.

    Reads the per-strategy "Paper summary | ... RealizedPnL=<pnl>" lines that
    every worker logs at square-off / max-loss. The log format is
    "<asctime> | <level> | <threadName> | <message>", and threadName is
    "<strategy_name>Thread", so we recover the date (from asctime), the strategy
    (from the thread name), and the figure. Returns {"YYYY-MM-DD":
    {strategy_name: pnl}} (last line wins per strategy/day). The log is opened in
    append mode, so it spans multiple days -- which is what enables month backfill.

    Only lines whose timestamp falls in 9:15-15:21 are considered, so after-market
    test runs do not overwrite real session P&L. Legacy thread names prefixed with
    "Misc " are normalized before lookup in _PNL_SHEET_ROW_LABELS.
    """
    result: dict[str, dict[str, float]] = {}
    try:
        # A missing log (e.g. a brand-new machine) just means "no figures yet".
        if not Path(log_path).exists():
            return result
        # errors="replace" stops one stray non-UTF-8 byte from aborting the parse.
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                # Cheap pre-filter: skip everything that isn't a P&L summary line
                # before doing the costlier split/regex work below.
                if "Paper summary" not in line or "RealizedPnL=" not in line:
                    continue
                # Break into the 4 log fields. maxsplit=3 keeps the message intact
                # even if the message itself contains " | ".
                parts = line.split(" | ", 3)
                if len(parts) < 4:
                    continue
                asctime, _level, thread_name, message = parts
                # Ignore lines logged outside trading hours (e.g. an after-market
                # test run) so they can't overwrite the real session's figures.
                if not _asctime_in_pnl_window(asctime):
                    continue
                # The date is the first 10 chars of asctime: "YYYY-MM-DD".
                date_str = asctime.strip()[:10]
                if len(date_str) != 10 or date_str[4] != "-":
                    continue
                # threadName is "<strategy_name>Thread"; drop the suffix to get the
                # strategy, then strip any legacy "Misc " prefix so it matches a row.
                thread_name = thread_name.strip()
                if not thread_name.endswith("Thread"):
                    continue
                strategy = _normalize_pnl_strategy_name(
                    thread_name[: -len("Thread")].strip()
                )
                # Pull the number out of "...RealizedPnL=1234.5..." via the regex.
                match = _PNL_LOG_LINE_RE.search(message)
                if not match:
                    continue
                try:
                    # Last write wins: a later summary for the same strategy/day
                    # (e.g. a re-entry's final line) overwrites the earlier one.
                    result.setdefault(date_str, {})[strategy] = float(match.group(1))
                except ValueError:
                    continue
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not parse P&L from the log file: %s", exc)
    return result


def _compute_pnl_sheet_updates(values, pnl_by_day, today_str):
    """
    Pure helper: decide which sheet cells to write (no I/O).

    INPUT:
      values     - the worksheet's current contents as a list of row-lists.
      pnl_by_day - {"YYYY-MM-DD": {strategy_name: pnl}} (from the log).
      today_str  - today's date "YYYY-MM-DD".

    Layout: row 0 holds the date column headers; column 0 holds strategy labels.
    Rule: only the current calendar month is touched; today's cell is always
    (over)written; an earlier day's cell is written only when it is currently
    blank (so manual edits and already-filled days are never clobbered).

    Returns (updates, unmatched): updates is a list of (row_idx, col_idx, value)
    with 0-based indices; unmatched is the sorted strategy names with no sheet row.
    """
    # An empty sheet (no header row) means there's nothing to line up against.
    if not values:
        return [], []
    # Build "date -> column index" from the header row, accepting only cells that
    # look exactly like YYYY-MM-DD so stray header text is ignored.
    header = values[0]
    date_to_col = {}
    for col_idx, cell in enumerate(header):
        text = str(cell).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            date_to_col[text] = col_idx
    # Build "row label -> row index" from column A. First occurrence wins, so a
    # duplicate label further down can't hijack the row we write to.
    label_to_row = {}
    for row_idx, row in enumerate(values):
        if not row:
            continue
        label = str(row[0]).strip()
        if label and label not in label_to_row:
            label_to_row[label] = row_idx

    updates = []
    unmatched = set()
    # Restrict everything to the current month so backfill can't wander into a
    # previous month that happens to share day-of-month columns.
    current_month = today_str[:7]
    for date_str, per_strategy in pnl_by_day.items():
        # Skip days outside this month, or days the sheet has no column for.
        if date_str[:7] != current_month or date_str not in date_to_col:
            continue
        col_idx = date_to_col[date_str]
        for strategy, pnl in per_strategy.items():
            # Map the code's strategy name to the sheet's exact row label.
            label = _PNL_SHEET_ROW_LABELS.get(strategy)
            if label is None or label not in label_to_row:
                # No mapping, or the sheet is missing that row -> report it.
                unmatched.add(strategy)
                continue
            row_idx = label_to_row[label]
            # Read whatever is already in the target cell (a short row counts as
            # blank, since the cell hasn't been created yet).
            existing = ""
            if col_idx < len(values[row_idx]):
                existing = str(values[row_idx][col_idx]).strip()
            # Today's cell is always written; earlier days are only *backfilled*
            # when blank, so a value already on the sheet is never overwritten.
            if date_str == today_str or existing == "":
                updates.append((row_idx, col_idx, round(float(pnl), 2)))
    return updates, sorted(unmatched)


def _update_pnl_google_sheet() -> None:
    """
    End-of-day: write each strategy's realised P&L into the current month's tab of
    the tracker Google Sheet (today's column, plus blank earlier-this-month cells
    backfilled from the log).

    Guarded no-op: if GSHEET_ID is unset, gspread is missing, the log has no
    figures, or anything goes wrong, it logs a warning and returns without
    disturbing the shutdown sequence.
    """
    # Step 1: bail out early (and quietly) when the feature isn't configured.
    sheet_id = _env_str("GSHEET_ID", "")
    if not sheet_id:
        logger.info("Google Sheet P&L update skipped (GSHEET_ID not set in .env).")
        return

    # Step 2: turn the log into {day: {strategy: pnl}}. No figures -> nothing to do.
    pnl_by_day = _parse_eod_pnl_by_day(LOG_FILE)
    if not pnl_by_day:
        logger.info("Google Sheet P&L update skipped (no per-strategy P&L found in the log).")
        return

    # Step 3: gspread is an optional dependency, so import it lazily and treat a
    # missing install as a skip rather than a crash.
    try:
        import gspread
    except ImportError:
        logger.warning(
            "Google Sheet P&L update skipped: gspread not installed "
            "(pip install gspread). The figures are still in the log."
        )
        return

    try:
        # Step 4: authenticate with the cached OAuth user token. The very first
        # run opens a browser for consent and writes the token file; later runs
        # reuse (and silently refresh) it.
        client_file = _env_str(
            "GSHEET_OAUTH_CLIENT_FILE",
            str(ROOT_DIR / "Dependencies" / "gsheet_oauth_client.json"),
        )
        token_file = _env_str(
            "GSHEET_OAUTH_TOKEN_FILE",
            str(ROOT_DIR / "Dependencies" / "gsheet_oauth_token.json"),
        )
        gc = gspread.oauth(
            credentials_filename=client_file,
            authorized_user_filename=token_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        # Step 5: the tracker keeps one tab per month ("May 2026", "June 2026", ...),
        # so select the current month's tab by name (not just the first sheet) and
        # read it in full, letting the pure helper line figures up against its own
        # rows/columns. If this month's tab doesn't exist yet, skip with a clear note.
        spreadsheet = gc.open_by_key(sheet_id)
        month_tab = datetime.now().strftime("%B %Y")
        try:
            worksheet = spreadsheet.worksheet(month_tab)
        except gspread.WorksheetNotFound:
            logger.warning(
                "Google Sheet P&L update skipped: no '%s' tab in the spreadsheet "
                "(add a tab for the new month).",
                month_tab,
            )
            return
        values = worksheet.get_all_values()
        today_str = datetime.now().strftime("%Y-%m-%d")
        # Step 6: work out the exact cells to write (no network I/O in here).
        updates, unmatched = _compute_pnl_sheet_updates(values, pnl_by_day, today_str)
        if unmatched:
            # A strategy with no row on the sheet is reported, never invented.
            logger.warning(
                "Google Sheet P&L: no matching row for %s -- add the corresponding "
                "strategy label in column A of the sheet (see _PNL_SHEET_ROW_LABELS).",
                ", ".join(unmatched),
            )
        if not updates:
            logger.info("Google Sheet P&L update: nothing to write.")
            return
        # Step 7: gspread's Cell is 1-based, our tuples are 0-based, so add 1 to
        # each. One batched update_cells call is far fewer API hits than writing
        # cells one by one; USER_ENTERED makes Sheets store the value as a number.
        cells = [gspread.Cell(row + 1, col + 1, value) for (row, col, value) in updates]
        worksheet.update_cells(cells, value_input_option="USER_ENTERED")
        logger.info(
            "Google Sheet P&L updated: %d cell(s) written (today=%s).", len(cells), today_str
        )
    except Exception as exc:
        # gspread turns some API failures (e.g. a 403 when the Google Sheets API
        # is disabled for the project, or the account can't access the sheet)
        # into a *bare* exception whose str() is empty -- which would log a
        # blank reason. Use %r + exc_info so the type and the full chained
        # traceback (which carries Google's own message and fix URL) are shown.
        logger.warning(
            "Google Sheet P&L update failed (non-fatal): %r", exc, exc_info=True
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Wire up the central fetcher plus all the strategy worker threads.

    Architecture summary:
    1. Configure logging (root logger so every thread benefits).
    2. Change working directory to the repo root so relative paths
       (instrument master, log files) stay valid no matter where the
       script is launched from.
    3. Build the shared store and shared stop signal.
    4. Build one fetcher and all the workers.
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
        "ATM single-leg family (22): 9 core - Renko 1m, EMA 5m, HeikinAshi 1m, "
        "ProfitShooter 1m, Goldmine 5m, MoneyMachine 5m, OpeningStrike 5m "
        "PCR/VWAP/ATR, CPR 5m, CPR Algo 3 5m (multi-instrument); + 13 TradingBot ports (%dm) - SMA Crossover, "
        "Bollinger Bands, Keltner Squeeze, Mean Reversion Z-Score, ML Ensemble, "
        "Multi-Timeframe, Opening Range Breakout, Parabolic SAR, RSI Divergence, "
        "RSI Reversal, Stochastic, Supertrend, Volatility Breakout. "
        "Hedged Puts family (2): Supertrend 3m PE, Donchian 5m CE. "
        "Delta-0.2 family (1): Delta20 09:20 reference, dual-side. "
        "Long Strangle family (1): 09:30 OTM1 CE+PE, two independent legs.",
        SIGNAL_GEN_WORKERS[0].derived_timeframe_minutes,
    )

    broker = DhanBrokerClient(CLIENT_CODE, ACCESS_TOKEN)
    store = SharedMarketDataStore()
    stop_event = threading.Event()

    # One producer (fetcher) + 26 consumers (22 ATM single-leg + 2 Hedged +
    # 1 Delta-0.2 + 1 Long Strangle), plus the optional SL Hunting AI agent
    # (appended below when SL_HUNTING_ENABLED).
    fetcher = CentralMarketDataFetcher(store, stop_event, broker)

    # ----- ATM single-leg family: each BUYS the ATM CE (LONG) / PE (SHORT) of the
    #       next-next expiry. All of these subclass AtmSingleLegStrategyWorker. -----
    workers = [
        # 8 "core" ATM strategies authored directly in this repo.
        RenkoStrategyWorker(store, stop_event, broker),
        EMATrendStrategyWorker(store, stop_event, broker),
        HeikinAshiStrategyWorker(store, stop_event, broker),
        ProfitShooterStrategyWorker(store, stop_event, broker),
        # Goldmine & Money Machine: sibling Subhamoy 5-min strategies, both with
        # Profit-Shooter-style dynamic (risk-budget) position sizing.
        GoldmineStrategyWorker(store, stop_event, broker),
        MoneyMachineStrategyWorker(store, stop_event, broker),
        # Opening-Strike PCR/VWAP/ATR also runs ATM single-leg, but it is
        # additionally driven by intraday option-chain OI flow.
        OpeningStrikePCRVWAPATRWorker(store, stop_event, broker),
        # CPR (Central Pivot Range): directional ATM single-leg on 5-min candles.
        CPRStrategyWorker(store, stop_event, broker),
        # CPR Algo 3: multi-instrument (spot + ITM CE + ITM PE observation); trades
        # the ATM CE/PE next-next expiry like the rest of this family.
        CPRAlgo3StrategyWorker(store, stop_event, broker),
    ]
    # Same family, different source: the 13 TradingBot ports are ALSO
    # AtmSingleLegStrategyWorker subclasses (built from a shared factory), so they
    # belong here next to the core ATM strategies rather than in a family of their own.
    workers.extend(worker_cls(store, stop_event, broker) for worker_cls in SIGNAL_GEN_WORKERS)

    workers += [
        # ----- Hedged Puts family: each picks current-week CE/PE by target premium. -----
        SupertrendBullishWorker(store, stop_event, broker),
        DonchianBearishWorker(store, stop_event, broker),
        # ----- Delta-0.2 family: dual-side hedged spread driven by option-chain Greeks. -----
        Delta20HedgedSpreadWorker(store, stop_event, broker),
        # ----- Long Strangle family: time-based, BUY OTM1 CE+PE, two independent legs. -----
        LongStrangleWorker(store, stop_event, broker),
    ]

    # ----- SL Hunting AI agent (optional): LLM-driven ATM single-leg. Appended
    #       only when the agent module loaded AND SL_HUNTING_ENABLED is set. Stays
    #       paper unless SL_HUNTING_LIVE_TRADING + LIVE_TRADING_ENABLED are both on
    #       (the live-wiring below applies the same double-gate as every strategy).
    if SLHuntingAIWorker is not None:
        workers.append(SLHuntingAIWorker(store, stop_event, broker))

    # -------------------------------------------------------------------------
    # Decide which strategies trade for real vs on paper.
    # -------------------------------------------------------------------------
    # A strategy trades LIVE only if TWO switches are both on:
    #   1) the global master switch LIVE_TRADING_ENABLED, and
    #   2) that strategy's own <PREFIX>_LIVE_TRADING flag.
    # This double-gate is a safety net: one switch can't accidentally send
    # everything live. We work it out ONCE here (not inside the threads) and set
    # each worker's `live_trading` flag. Default stays paper; if a strategy has no
    # known prefix it is left on paper and an error is logged so it's noticed.
    # Real orders go to whichever broker LIVE_BROKER selected (see execution_client).
    master_live = _env_bool("LIVE_TRADING_ENABLED", False)
    live_count = 0  # how many strategies ended up live (for the summary log)
    for worker in workers:
        prefix = STRATEGY_ENV_PREFIX.get(worker.strategy_name)
        per_strategy = _env_bool(f"{prefix}_LIVE_TRADING", False) if prefix else False
        worker.live_trading = bool(master_live and per_strategy)
        if worker.live_trading:
            live_count += 1
            logger.warning(
                "LIVE TRADING ENABLED for %s -> real %s orders (product=%s).",
                worker.strategy_name, LIVE_BROKER, LIVE_PRODUCT_TYPE,
            )
        elif prefix is None:
            logger.error(
                "No env prefix mapped for strategy %s; forcing PAPER.", worker.strategy_name
            )

    # Startup guard: never attempt live trading without a working broker client.
    if live_count > 0 and execution_client is None:
        logger.error(
            '%d strateg(ies) flagged for LIVE trading but the %s execution layer is '
            "unavailable (SDK/deps missing or failed to import). "
            "Forcing ALL strategies back to PAPER.",
            live_count, LIVE_BROKER,
        )
        for worker in workers:
            worker.live_trading = False
        live_count = 0

    # Eagerly establish the broker session NOW (at startup) so worker threads never
    # block on login mid-session (Kotak uses TOTP, Shoonya may auto-generate it,
    # and Flattrade uses browser authorization). If login fails here, force every
    # strategy to PAPER rather than failing one order at a time later.
    if live_count > 0 and execution_client is not None:
        logger.warning("Logging in to %s for LIVE trading now.", LIVE_BROKER)
        if not execution_client.ensure_logged_in():
            logger.error("%s login failed at startup; forcing ALL strategies back to PAPER.", LIVE_BROKER)
            for worker in workers:
                worker.live_trading = False
            live_count = 0
        else:
            # One-time scrip/symbol-master download so live orders can be resolved
            # without paying the download cost mid-session.
            logger.info("%s login OK; downloading NSE F&O scrip master (one-time)...", LIVE_BROKER)
            if not execution_client.preload_scrip_master():
                logger.warning(
                    "Scrip master preload failed; symbols will be resolved lazily on first order."
                )

    if master_live:
        logger.warning(
            "LIVE_TRADING_ENABLED=true (broker=%s): %d of %d strategies will place REAL orders.",
            LIVE_BROKER, live_count, len(workers),
        )
    else:
        logger.info(
            "Live trading master switch OFF (LIVE_TRADING_ENABLED=false); all strategies PAPER."
        )

    # Optional Telegram notifier: a queue-based consumer thread that posts a
    # message on every entry/exit from ANY worker. Wired only when the .env
    # flag is on AND credentials are present; otherwise workers keep their
    # default `trade_event_queue = None` and publish_trade_event() is a no-op.
    telegram_worker = None
    trade_event_queue = None
    if TELEGRAM_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        trade_event_queue = queue.Queue(maxsize=1000)
        for worker in workers:
            worker.trade_event_queue = trade_event_queue
        telegram_worker = TelegramMessageWorker(
            trade_event_queue, stop_event, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        )
        logger.info("Telegram notifications ENABLED -> chat_id=%s", TELEGRAM_CHAT_ID)
    elif TELEGRAM_ENABLED:
        logger.warning(
            "TELEGRAM_ENABLED=true but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID is blank; "
            "running WITHOUT Telegram notifications."
        )
    else:
        logger.info("Telegram notifications disabled (TELEGRAM_ENABLED=false).")

    fetcher.start()
    if telegram_worker is not None:
        telegram_worker.start()
    for worker in workers:
        worker.start()

    try:
        # The main thread is a passive supervisor; it does not trade.
        while any(worker.is_alive() for worker in workers):
            for worker in workers:
                worker.join(timeout=1.0)
        # Every worker has exited on its own (square-off / max-loss) -> clean
        # end of day. Send the cumulative per-strategy P&L summary through the
        # Telegram worker, which is still alive here (stop_event is set below,
        # in finally). On Ctrl+C this line is skipped, so partial/forced
        # shutdowns do not emit a misleading summary.
        _publish_eod_summary(workers, trade_event_queue)
        # All workers have exited cleanly -> write each strategy's day-end P&L
        # into the tracker Google Sheet (and backfill any blank days this month).
        _update_pnl_google_sheet()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Signaling all threads to stop.")
        stop_event.set()
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=SHUTDOWN_JOIN_SECONDS)
        fetcher.join(timeout=SHUTDOWN_JOIN_SECONDS)
        if telegram_worker is not None:
            telegram_worker.join(timeout=SHUTDOWN_JOIN_SECONDS)

        supervised_threads = [fetcher, telegram_worker, *workers]
        alive_threads = [
            thread.name for thread in supervised_threads if thread is not None and thread.is_alive()
        ]
        if alive_threads:
            logger.warning("Some threads did not exit within the shutdown window: %s", alive_threads)
        else:
            logger.info("All threads exited cleanly.")

        # Politely log out of the broker on the way out, but only if we ever logged
        # in (pure-paper runs never do). Wrapped in try/except so a logout hiccup
        # can't stop the rest of shutdown.
        if execution_client is not None and getattr(execution_client, "is_logged_in", False):
            try:
                execution_client.logout()
                logger.info("%s execution session logged out.", LIVE_BROKER)
            except Exception as exc:
                logger.warning("%s logout failed: %s", LIVE_BROKER, exc)

    # End-of-day instrument master refresh. Runs unconditionally (even after
    # Ctrl+C or a partial shutdown) because the scheduler that re-launches
    # this script tomorrow morning needs the fresh CSV regardless of how
    # today's session ended. The helper is fail-soft: a failed download is
    # logged and the existing CSV stays on disk as a fallback.
    _refresh_instrument_master_for_next_day()


if __name__ == "__main__":
    main()
