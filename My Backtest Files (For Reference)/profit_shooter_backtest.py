"""
Backtest the Profit Shooter strategy using backtesting.py.

High-level flow:
1. Read 1-minute OHLC CSV.
2. Build SMA / EMA / ATR and pin-bar columns using the shared strategy file.
3. Detect setup candles from the latest completed candle.
4. Place a one-bar stop-entry order so only the *next* candle can trigger entry.
5. Manage the open trade using the shared signal engine for:
   - 1.5R activation
   - EMA9 trailing logic
6. Keep a broker-side stop-loss attached to the trade throughout its life.
7. Enforce daily max-loss guard and intraday square-off.
8. Save trades, daily equity, and raw stats for later review.

Dataset selection flow:
- You can now choose whether the backtest should use NIFTY, BANKNIFTY, or
  FINNIFTY OHLC data.
- If you pass `--data`, that explicit CSV path wins.
- If you do not pass `--data`, the script automatically picks the default CSV
  path for the selected `--dataset`.
- Output files also get dataset-specific names so one run does not overwrite
  another run from a different index.

Important modeling note:
- The shared Profit Shooter signal file defines trailing exit as:
  "exit at the next candle open after a completed candle closes beyond EMA9".
- To model that more faithfully, this backtest uses:
  `trade_on_close=False`
- That means market exits requested at the end of a candle are executed on the
  next available candle open.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import time as dt_time
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from dotenv import load_dotenv

# Anchor everything to the repo root (parent of `My Backtest Files (For
# Reference)/`) so paths resolve regardless of cwd. `__file__` lives at
# <repo>/My Backtest Files (For Reference)/..., so dirname(dirname(__file__))
# == <repo_root>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Strategy logic lives in the sibling `Signal Generators/` folder. Adding it
# to sys.path here lets the import below resolve regardless of cwd.
_SIGNAL_GENERATORS_DIR = os.path.join(_REPO_ROOT, "Signal Generators")
if _SIGNAL_GENERATORS_DIR not in sys.path:
    sys.path.insert(0, _SIGNAL_GENERATORS_DIR)

from profit_shooter_strategy_logic import (
    ProfitShooterConfig,
    ProfitShooterPositionContext,
    ProfitShooterSignalEngine,
    build_profit_shooter_with_indicators,
)


# -----------------------------
# User configuration
# -----------------------------
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Backtest Outputs")

# These are the project-level default CSV paths for each index.
# The backtest will use one of these automatically when `--data` is not passed.
DATASET_DEFAULT_PATHS = {
    "nifty": os.path.join(_REPO_ROOT, "Backtest Outputs", "nifty_renko_futures_5y_1min_data.csv"),
    "banknifty": os.path.join(
        _REPO_ROOT, "Backtest Outputs", "banknifty_renko_futures_5y_1min_data.csv"
    ),
    "finnifty": os.path.join(
        _REPO_ROOT, "Backtest Outputs", "finnifty_renko_futures_5y_1min_data.csv"
    ),
}

# Friendly labels are used in logs so the output is easier to read.
DATASET_DISPLAY_NAMES = {
    "nifty": "NIFTY",
    "banknifty": "BANKNIFTY",
    "finnifty": "FINNIFTY",
}

# Allow a few human-friendly aliases, so inputs like `bank_nifty` or
# `fin-nifty` still resolve to the correct internal dataset key.
DATASET_KEY_ALIASES = {
    "nifty": "nifty",
    "nifty50": "nifty",
    "banknifty": "banknifty",
    "bank_nifty": "banknifty",
    "bank-nifty": "banknifty",
    "finnifty": "finnifty",
    "fin_nifty": "finnifty",
    "fin-nifty": "finnifty",
}

STARTING_CAPITAL = 600000

# Position sizing is intentionally kept separate from dataset selection.
# In other words:
# - `--dataset` changes which OHLC candles are loaded
# - these values still decide how much quantity the backtest trades
#
# So if you switch from NIFTY data to BANKNIFTY or FINNIFTY data, review these
# numbers and adjust them to whatever contract sizing you want to simulate.
LOT_SIZE = 65
LOTS = 3
POSITION_SIZE = LOT_SIZE * LOTS
MARGIN_REQUIREMENT = 0.15
AUTO_ADJUST_MARGIN = True
MIN_MARGIN_FLOOR = 0.02

ENTRY_START_TIME = dt_time(9, 25)
LAST_SETUP_TIME = dt_time(15, 14)
SQUARE_OFF_TIME = dt_time(15, 15)
DAILY_MAX_LOSS_PCT = 0.03

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


def normalize_dataset_key(value: str) -> str:
    """
    Normalize a user-supplied dataset name into one supported internal key.

    Example:
    - `BANKNIFTY` -> `banknifty`
    - `bank_nifty` -> `banknifty`
    - `fin-nifty` -> `finnifty`

    This keeps the CLI friendly while ensuring the rest of the code works with
    one consistent set of dictionary keys.
    """
    key = str(value or "").strip().lower()
    normalized = DATASET_KEY_ALIASES.get(key, key)
    if normalized not in DATASET_DEFAULT_PATHS:
        valid = ", ".join(sorted(DATASET_DEFAULT_PATHS))
        raise ValueError(f"Unsupported dataset `{value}`. Choose from: {valid}")
    return normalized


try:
    DEFAULT_DATASET_KEY = normalize_dataset_key(os.getenv("PROFIT_SHOOTER_DATASET", "banknifty"))
except ValueError:
    # Fall back to NIFTY if the environment variable contains a typo.
    DEFAULT_DATASET_KEY = "nifty"


def _env_float(name: str, default: float) -> float:
    """Read float config from environment safely."""
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    """Read integer config from environment safely."""
    raw = os.getenv(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


STRATEGY_CONFIG = ProfitShooterConfig(
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

MIN_BARS = (
    max(
        STRATEGY_CONFIG.sma_fast_period,
        STRATEGY_CONFIG.sma_slow_period,
        STRATEGY_CONFIG.atr_period,
        STRATEGY_CONFIG.trailing_ema_period,
    )
    + max(STRATEGY_CONFIG.trend_lookback, STRATEGY_CONFIG.pullback_lookback)
    + 2
)

BROKERAGE = _env_float("BROKERAGE", _env_float("BROGERAGE", 80.0))


def get_dataset_display_name(dataset_key: str) -> str:
    """Return the human-readable label used in logs and filenames."""
    return DATASET_DISPLAY_NAMES[normalize_dataset_key(dataset_key)]


def resolve_data_path(dataset_key: str, explicit_path: str = "") -> str:
    """
    Decide which CSV path the backtest should load.

    Precedence:
    1. `--data` explicit path from the user
    2. default path mapped to `--dataset`
    """
    if str(explicit_path or "").strip():
        return str(explicit_path).strip()
    return DATASET_DEFAULT_PATHS[normalize_dataset_key(dataset_key)]


def build_output_paths(dataset_key: str) -> dict:
    """
    Build dataset-specific output file paths.

    This is important because a BANKNIFTY run and a NIFTY run should not both
    write to the same `trades.csv` or `stats.txt` file.
    """
    normalized_key = normalize_dataset_key(dataset_key)
    prefix = f"{normalized_key}_profit_shooter_futures"
    return {
        "output_dir": OUTPUT_DIR,
        "log_file": os.path.join(OUTPUT_DIR, f"{prefix}_5y_backtest.log"),
        "trades_path": os.path.join(OUTPUT_DIR, f"{prefix}_5y_trades.csv"),
        "daily_equity_path": os.path.join(OUTPUT_DIR, f"{prefix}_5y_daily_equity.csv"),
        "stats_path": os.path.join(OUTPUT_DIR, f"{prefix}_5y_stats.txt"),
        "daily_loss_path": os.path.join(OUTPUT_DIR, f"{prefix}_daily_max_loss.csv"),
    }


def setup_logging(log_path: str) -> None:
    """Log to both console and file for easy debugging and audit."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _find_first_col(df: pd.DataFrame, names):
    """Return first matching column name (case-insensitive)."""
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def load_ohlc_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw OHLC CSV, clean it, and convert it to backtesting.py format.

    Output format:
    - Datetime index
    - Columns: Open, High, Low, Close, Volume
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"Data file is empty: {csv_path}")

    ts_col = _find_first_col(raw, ["timestamp", "datetime", "date", "time"])
    o_col = _find_first_col(raw, ["open"])
    h_col = _find_first_col(raw, ["high"])
    l_col = _find_first_col(raw, ["low"])
    c_col = _find_first_col(raw, ["close"])
    if not all([ts_col, o_col, h_col, l_col, c_col]):
        raise ValueError(
            "Input CSV must contain timestamp/date + open/high/low/close columns."
        )

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col], errors="coerce"),
            "open": pd.to_numeric(raw[o_col], errors="coerce"),
            "high": pd.to_numeric(raw[h_col], errors="coerce"),
            "low": pd.to_numeric(raw[l_col], errors="coerce"),
            "close": pd.to_numeric(raw[c_col], errors="coerce"),
        }
    ).dropna()

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS:
        raise ValueError(f"Need at least {MIN_BARS} rows, got {len(df)}")

    bt_df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    bt_df = bt_df.set_index("timestamp")
    bt_df["Volume"] = 0
    return bt_df


@dataclass
class PendingEntryOrder:
    """Minimal state needed for one-bar breakout entry orders."""

    direction: str
    order: object
    setup_timestamp: pd.Timestamp
    breakout_underlying: float
    stop_underlying: float


class ProfitShooterFuturesStrategy(Strategy):
    """
    backtesting.py strategy class for the Profit Shooter futures strategy.

     This class is responsible for 4 main jobs:
    1. Rebuild indicator history from candles seen so far.
    2. Place next-candle breakout stop-entry orders from setup candles.
    3. Let the shared strategy engine manage target activation and EMA9 trailing.
    4. Enforce day-level safety rules like max loss and square-off.
    """

    lot_size = POSITION_SIZE

    def init(self):
        """Initialize per-run state used by the backtest loop."""
        self.last_processed_candle_ts = None
        self.strategy_engine = ProfitShooterSignalEngine(STRATEGY_CONFIG)

        # Active trade helper state.
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0
        self.trailing_active = False
        self.pending_trailing_exit = False
        self.close_request_reason = ""

        # Pending next-candle breakout order state.
        self.pending_entry: Optional[PendingEntryOrder] = None

        # Diagnostic counters.
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        self.square_off_count = 0
        self.daily_loss_halt_count = 0
        self.margin_skip_count = 0
        self.pending_entry_cancel_count = 0

        # Used to detect new auto-closed trades such as stop-loss exits.
        self.last_closed_trade_count = 0
        self.margin_skip_logged_dates = set()

        # Per-day risk tracking.
        self.current_day = None
        self.day_start_equity = None
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        self.day_trading_blocked = False
        self.daily_loss_tracker = {}

    def _reset_trade_state(self) -> None:
        """Clear active-trade helper variables after a trade is closed."""
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0
        self.trailing_active = False
        self.pending_trailing_exit = False
        self.close_request_reason = ""

    def _clear_pending_entry(self, *, cancel_order: bool) -> None:
        """
        Clear the remembered pending breakout order.

        `cancel_order=True` also cancels the live order object inside
        backtesting.py because the "only next candle" window has expired.
        """
        if self.pending_entry is None:
            return

        if cancel_order:
            try:
                self.pending_entry.order.cancel()
            except Exception:
                pass
            self.pending_entry_cancel_count += 1

        self.pending_entry = None

    def _compute_target(self, direction: str, entry_price: float, stop_price: float) -> float:
        """Convert entry + stop into the strategy's initial 1.5R target."""
        direction_txt = str(direction).strip().upper()
        if direction_txt == "LONG":
            risk = float(entry_price) - float(stop_price)
            return float(entry_price) + STRATEGY_CONFIG.target_r_multiple * risk
        if direction_txt == "SHORT":
            risk = float(stop_price) - float(entry_price)
            return float(entry_price) - STRATEGY_CONFIG.target_r_multiple * risk
        return 0.0

    def _can_afford_entry(self, entry_price: float):
        """
        Check if a new entry is affordable using current running equity.

        Returns:
            (can_enter, effective_margin, required_margin, equity, notional)
        """
        equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        if notional <= 0:
            return False, MARGIN_REQUIREMENT, 0.0, equity, notional

        if AUTO_ADJUST_MARGIN:
            affordable_margin = (equity * 0.98) / notional
            effective_margin = min(
                MARGIN_REQUIREMENT,
                max(MIN_MARGIN_FLOOR, affordable_margin),
            )
        else:
            effective_margin = MARGIN_REQUIREMENT

        required_margin = notional * effective_margin
        can_enter = required_margin <= equity
        return can_enter, effective_margin, required_margin, equity, notional

    def _current_ohlc(self, n: int) -> pd.DataFrame:
        """Build OHLC DataFrame up to current index `n` from engine arrays."""
        idx = pd.DatetimeIndex(self.data.index[:n])
        return pd.DataFrame(
            {
                "timestamp": idx,
                "open": np.asarray(self.data.Open[:n], dtype=float),
                "high": np.asarray(self.data.High[:n], dtype=float),
                "low": np.asarray(self.data.Low[:n], dtype=float),
                "close": np.asarray(self.data.Close[:n], dtype=float),
            }
        )

    def _sync_closed_trade_state(self) -> None:
        """
        Detect trades that got closed by the framework since the previous bar.

        This is especially important because attached stop-loss orders can close
        the trade intrabar without any explicit `self.position.close()` call.
        """
        closed_count = len(self.closed_trades)
        if closed_count <= self.last_closed_trade_count:
            return

        newly_closed = closed_count - self.last_closed_trade_count
        self.exit_count += newly_closed
        self.last_closed_trade_count = closed_count

        self._reset_trade_state()
        if self.pending_entry is not None:
            self._clear_pending_entry(cancel_order=False)

    def _sync_open_trade_state(self) -> None:
        """
        Detect when a pending breakout order filled and turn it into trade state.
        """
        if not self.position or self.trade_direction or self.pending_entry is None:
            return

        active_trade = self.trades[-1] if self.trades else None
        actual_entry_price = (
            float(active_trade.entry_price)
            if active_trade is not None
            else float(self.pending_entry.breakout_underlying)
        )
        self.trade_direction = str(self.pending_entry.direction).strip().upper()
        self.entry_underlying = actual_entry_price
        self.stop_underlying = float(self.pending_entry.stop_underlying)
        self.target_underlying = self._compute_target(
            self.trade_direction,
            self.entry_underlying,
            self.stop_underlying,
        )
        self.trailing_active = False
        self.pending_trailing_exit = False
        self.close_request_reason = ""
        self._clear_pending_entry(cancel_order=False)

    def _expire_old_pending_entry(self, bar_ts: pd.Timestamp) -> None:
        """
        Cancel the one-bar breakout order after its allowed candle has passed.

        Example:
        - candle T is the setup pin bar
        - the stop-entry order is placed at the end of candle T
        - candle T+1 is the only candle allowed to trigger it
        - when we reach the end of T+1 and still have no position, cancel it
        """
        if self.pending_entry is None or self.position:
            return

        if pd.Timestamp(bar_ts) > pd.Timestamp(self.pending_entry.setup_timestamp):
            logging.info(
                "Pending entry expired without trigger | setup_ts=%s | side=%s",
                self.pending_entry.setup_timestamp,
                self.pending_entry.direction,
            )
            self._clear_pending_entry(cancel_order=True)

    def _place_setup_order(
        self,
        *,
        direction: str,
        setup_timestamp: pd.Timestamp,
        breakout_price: float,
        stop_price: float,
        bar_date,
        bar_time,
    ) -> None:
        """
        Submit a one-bar stop-entry order from the latest setup candle.

        Long:
        - buy stop at pin-bar high + tick
        - attached stop-loss at pin-bar low - ATR buffer

        Short:
        - sell stop at pin-bar low - tick
        - attached stop-loss at pin-bar high + ATR buffer
        """
        can_enter, eff_margin, req_margin, equity_now, _ = self._can_afford_entry(breakout_price)
        if not can_enter:
            self.margin_skip_count += 1
            if bar_date not in self.margin_skip_logged_dates:
                logging.warning(
                    "Setup skipped (insufficient current equity margin) | "
                    "date=%s | time=%s | side=%s | equity=%.2f | required=%.2f | "
                    "effective_margin=%.4f | configured_margin=%.4f",
                    bar_date,
                    bar_time,
                    direction,
                    equity_now,
                    req_margin,
                    eff_margin,
                    MARGIN_REQUIREMENT,
                )
                self.margin_skip_logged_dates.add(bar_date)
            return

        direction_txt = str(direction).strip().upper()
        if direction_txt == "LONG":
            order = self.buy(
                size=self.lot_size,
                stop=float(breakout_price),
                sl=float(stop_price),
                tag="PROFIT_SHOOTER_LONG_ENTRY",
            )
        else:
            order = self.sell(
                size=self.lot_size,
                stop=float(breakout_price),
                sl=float(stop_price),
                tag="PROFIT_SHOOTER_SHORT_ENTRY",
            )

        self.entry_submit_count += 1
        self.pending_entry = PendingEntryOrder(
            direction=direction_txt,
            order=order,
            setup_timestamp=pd.Timestamp(setup_timestamp),
            breakout_underlying=float(breakout_price),
            stop_underlying=float(stop_price),
        )
        logging.info(
            "Pending setup order placed | setup_ts=%s | side=%s | trigger=%.2f | stop=%.2f",
            setup_timestamp,
            direction_txt,
            breakout_price,
            stop_price,
        )

    def next(self):
        """
        Main strategy loop called by backtesting.py for every new bar.

        Sequence:
        - Sync framework state (new fills and auto-closed trades)
        - Enforce intraday time rules
        - Enforce per-day max loss
        - Warm-up check
        - Rebuild indicator columns from history so far
        - Use latest setup candle to place one-bar breakout orders
        - Use shared engine to manage open-trade trailing logic
        """
        bar_ts = pd.Timestamp(self.data.index[-1])
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        self._sync_closed_trade_state()
        self._sync_open_trade_state()
        self._expire_old_pending_entry(bar_ts)

        if self.current_day != bar_date:
            self.current_day = bar_date
            self.day_start_equity = float(self.equity)
            self.day_trading_blocked = False
            self.daily_loss_tracker[bar_date] = 0.0

        if self.day_start_equity is None:
            self.day_start_equity = float(self.equity)
        current_equity = float(self.equity)
        day_loss = max(0.0, self.day_start_equity - current_equity)
        prev_day_max_loss = float(self.daily_loss_tracker.get(bar_date, 0.0))
        if day_loss > prev_day_max_loss:
            self.daily_loss_tracker[bar_date] = float(day_loss)

        if (not self.day_trading_blocked) and day_loss >= self.day_loss_limit:
            self.day_trading_blocked = True
            self.daily_loss_halt_count += 1
            logging.warning(
                "Daily loss cap hit | date=%s | time=%s | day_loss=%.2f | cap=%.2f",
                bar_date,
                bar_time,
                day_loss,
                self.day_loss_limit,
            )

        if self.day_trading_blocked:
            if self.pending_entry is not None:
                self._clear_pending_entry(cancel_order=True)
            if self.position and not self.close_request_reason:
                self.position.close()
                self.close_request_reason = "DAILY_MAX_LOSS"
            return

        if bar_time >= SQUARE_OFF_TIME:
            if self.pending_entry is not None:
                self._clear_pending_entry(cancel_order=True)
            if self.position and not self.close_request_reason:
                self.position.close()
                self.close_request_reason = "TIME_CUTOFF_1515"
                self.square_off_count += 1
            return

        if bar_time < ENTRY_START_TIME:
            return

        if self.position and self.close_request_reason:
            return

        n = len(self.data.Close)
        if n < MIN_BARS:
            return

        ohlc = self._current_ohlc(n)
        candles = build_profit_shooter_with_indicators(ohlc, STRATEGY_CONFIG)
        if candles.empty:
            return

        last_ts = candles.iloc[-1]["timestamp"]
        if self.last_processed_candle_ts == last_ts:
            return
        self.last_processed_candle_ts = last_ts

        current = candles.iloc[-1]

        if self.position:
            position_ctx = ProfitShooterPositionContext(
                direction=self.trade_direction,
                entry_underlying=self.entry_underlying,
                stop_underlying=self.stop_underlying,
                target_underlying=self.target_underlying,
                trailing_active=self.trailing_active,
                pending_trailing_exit=self.pending_trailing_exit,
            )
            decision = self.strategy_engine.evaluate_candle(candles, position=position_ctx)

            # Keep local state updated even when the action is HOLD because the
            # shared engine may have just activated trailing mode.
            self.target_underlying = float(decision.target_underlying or self.target_underlying)
            self.trailing_active = bool(decision.trailing_active)
            self.pending_trailing_exit = bool(decision.pending_trailing_exit)

            if decision.action == "EXIT":
                self.position.close()
                self.close_request_reason = decision.exit_reason or "ENGINE_EXIT"
                return

            if decision.pending_trailing_exit:
                # "If a completed candle closes beyond EMA9, exit at next candle open."
                self.position.close()
                self.close_request_reason = "EMA9_TRAIL_EXIT"
                return

            return

        # If a pending entry still exists here, it means we just placed it on
        # this same bar and should wait for the next bar to decide whether it fills.
        if self.pending_entry is not None:
            return

        if not (ENTRY_START_TIME <= bar_time <= LAST_SETUP_TIME):
            return

        if bool(current["long_setup"]):
            self.signal_count += 1
            breakout_price = float(current["high"]) + STRATEGY_CONFIG.tick_size
            stop_price = float(current["low"]) - STRATEGY_CONFIG.sl_atr_buffer * float(current["atr"])
            if np.isfinite(breakout_price) and np.isfinite(stop_price) and stop_price < breakout_price:
                self._place_setup_order(
                    direction="LONG",
                    setup_timestamp=last_ts,
                    breakout_price=breakout_price,
                    stop_price=stop_price,
                    bar_date=bar_date,
                    bar_time=bar_time,
                )
            return

        if bool(current["short_setup"]):
            self.signal_count += 1
            breakout_price = float(current["low"]) - STRATEGY_CONFIG.tick_size
            stop_price = float(current["high"]) + STRATEGY_CONFIG.sl_atr_buffer * float(current["atr"])
            if np.isfinite(breakout_price) and np.isfinite(stop_price) and breakout_price < stop_price:
                self._place_setup_order(
                    direction="SHORT",
                    setup_timestamp=last_ts,
                    breakout_price=breakout_price,
                    stop_price=stop_price,
                    bar_date=bar_date,
                    bar_time=bar_time,
                )
            return


def save_outputs(stats, output_paths: dict, strategy_obj=None) -> None:
    """
    Write trades, daily equity, daily max-loss tracker, and raw stats snapshot to disk.

    `output_paths` comes from `build_output_paths(...)`, so the saved filenames
    automatically match the chosen dataset. For example:
    - NIFTY run -> `nifty_profit_shooter_futures_...`
    - BANKNIFTY run -> `banknifty_profit_shooter_futures_...`
    """
    output_dir = output_paths["output_dir"]
    trades_path = output_paths["trades_path"]
    daily_equity_path = output_paths["daily_equity_path"]
    stats_path = output_paths["stats_path"]
    daily_loss_path = output_paths["daily_loss_path"]
    os.makedirs(output_dir, exist_ok=True)

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        trades_to_save = trades.copy()
        if "PnL" in trades_to_save.columns:
            pnl_values = pd.to_numeric(trades_to_save["PnL"], errors="coerce")
            trades_to_save["Brokerage"] = float(BROKERAGE)
            trades_to_save["FinalPnL"] = pnl_values - float(BROKERAGE)
            trades_to_save["CumulativeFinalPnL"] = trades_to_save["FinalPnL"].cumsum()
            logging.info(
                "Brokerage-adjusted trade PnL | brokerage_per_trade=%.2f | trades=%s | "
                "gross_total=%.2f | final_total=%.2f",
                BROKERAGE,
                len(trades_to_save),
                float(pnl_values.sum(skipna=True)),
                float(trades_to_save["FinalPnL"].sum(skipna=True)),
            )
        else:
            logging.warning("Trades output missing `PnL` column; brokerage adjustment not applied.")
        trades_to_save.to_csv(trades_path, index=False)

    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, errors="coerce")
        eq = eq[eq.index.notna()]
        daily = eq[["Equity"]].resample("D").last().dropna()
        daily.to_csv(daily_equity_path, index_label="date")

    with open(stats_path, "w", encoding="ascii", errors="ignore") as file_obj:
        file_obj.write(str(stats))

    if strategy_obj is not None:
        tracker = getattr(strategy_obj, "daily_loss_tracker", None)
        if isinstance(tracker, dict) and tracker:
            daily_loss_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(list(tracker.keys())),
                    "max_intraday_loss": [float(value) for value in tracker.values()],
                    "daily_loss_cap": STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
                }
            ).sort_values("date")
            daily_loss_df.to_csv(daily_loss_path, index=False, date_format="%Y-%m-%d")
            logging.info("Daily max-loss tracker saved: %s", daily_loss_path)

    logging.info("Trades saved: %s", trades_path)
    logging.info("Daily equity saved: %s", daily_equity_path)
    logging.info("Stats saved: %s", stats_path)


def run_backtest(data_path: str, dataset_key: str = "nifty"):
    """
    Load data, configure Backtest engine, run strategy, and return stats + strategy object.

    The strategy logic itself is dataset-agnostic. It only needs OHLC candles.
    The `dataset_key` is mainly used for clearer logging so you know which
    instrument the current run is meant to represent.
    """
    dataset_label = get_dataset_display_name(dataset_key)
    logging.info("Starting backtest | dataset=%s | csv=%s", dataset_label, data_path)

    data = load_ohlc_data(data_path)
    data_start = pd.Timestamp(data.index.min())
    data_end = pd.Timestamp(data.index.max())
    logging.info(
        "Backtest Period (from CSV data) | start=%s | end=%s",
        data_start,
        data_end,
    )
    logging.info(
        "Loaded data | rows=%s | start=%s | end=%s",
        len(data),
        data_start,
        data_end,
    )

    if AUTO_ADJUST_MARGIN:
        effective_margin = MIN_MARGIN_FLOOR
        logging.info(
            "Dynamic margin auto-adjust enabled | engine_margin_floor=%.4f | configured_margin=%.4f",
            effective_margin,
            MARGIN_REQUIREMENT,
        )
    else:
        effective_margin = MARGIN_REQUIREMENT
        max_close = float(data["Close"].max())
        max_notional = max_close * POSITION_SIZE
        required_margin_at_config = max_notional * MARGIN_REQUIREMENT
        if required_margin_at_config > STARTING_CAPITAL:
            logging.warning(
                "Configured margin %.4f needs %.2f (> start cash %.2f). "
                "Entries may be rejected due to insufficient margin.",
                MARGIN_REQUIREMENT,
                required_margin_at_config,
                STARTING_CAPITAL,
            )

    bt = Backtest(
        data,
        ProfitShooterFuturesStrategy,
        cash=STARTING_CAPITAL,
        margin=effective_margin,
        commission=0.0,
        trade_on_close=False,
        exclusive_orders=True,
        hedging=False,
        finalize_trades=True,
    )
    stats = bt.run()
    strategy_obj = stats.get("_strategy", None)
    if strategy_obj is not None:
        logging.info(
            "Strategy counters | signals=%s | entry_submissions=%s | exits=%s | "
            "day_square_offs=%s | day_loss_halts=%s | margin_skips=%s | "
            "pending_entry_cancels=%s",
            getattr(strategy_obj, "signal_count", "NA"),
            getattr(strategy_obj, "entry_submit_count", "NA"),
            getattr(strategy_obj, "exit_count", "NA"),
            getattr(strategy_obj, "square_off_count", "NA"),
            getattr(strategy_obj, "daily_loss_halt_count", "NA"),
            getattr(strategy_obj, "margin_skip_count", "NA"),
            getattr(strategy_obj, "pending_entry_cancel_count", "NA"),
        )
    if "# Trades" in stats.index and int(stats["# Trades"]) == 0:
        logging.warning("Backtest finished with 0 closed trades. Check entry conditions/data regime.")
    return stats, strategy_obj


def main():
    """
    CLI entrypoint.

    Usage:
    `python "profit_shooter_backtest.py" --dataset banknifty`
    `python "profit_shooter_backtest.py" --dataset finnifty`
    `python "profit_shooter_backtest.py" --dataset nifty --data "custom_path.csv"`

    Important precedence rule:
    - `--data` overrides the dataset default CSV path
    - `--dataset` still controls the log/output file naming unless you change
      the code further
    """
    parser = argparse.ArgumentParser(
        description="Profit Shooter futures backtest using backtesting.py"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_KEY,
        choices=sorted(DATASET_DEFAULT_PATHS),
        help=(
            "Which default OHLC dataset to use when --data is not passed. "
            "Choices: nifty, banknifty, finnifty."
        ),
    )
    parser.add_argument(
        "--data",
        default="",
        help=(
            "Optional explicit path to OHLC CSV with "
            "timestamp/open/high/low/close columns. If this is passed, it "
            "overrides the dataset's default CSV path."
        ),
    )
    args = parser.parse_args()

    dataset_key = normalize_dataset_key(args.dataset)
    dataset_label = get_dataset_display_name(dataset_key)
    data_path = resolve_data_path(dataset_key, explicit_path=args.data)
    output_paths = build_output_paths(dataset_key)

    setup_logging(output_paths["log_file"])
    logging.info(
        "Config | dataset=%s | data_path=%s | capital=%s | lot_size=%s | lots=%s | qty=%s | margin=%s | "
        "auto_adjust_margin=%s | daily_max_loss_pct=%s | daily_max_loss_amt=%s | "
        "brokerage_per_trade=%s | sma_fast=%s | sma_slow=%s | atr_period=%s | "
        "trailing_ema=%s | trend_lookback=%s | pullback_lookback=%s | "
        "pullback_min_count=%s | proximity_atr_mult=%s | tick_size=%s | "
        "sl_atr_buffer=%s | target_r_mult=%s",
        dataset_label,
        data_path,
        STARTING_CAPITAL,
        LOT_SIZE,
        LOTS,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        AUTO_ADJUST_MARGIN,
        DAILY_MAX_LOSS_PCT,
        STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
        BROKERAGE,
        STRATEGY_CONFIG.sma_fast_period,
        STRATEGY_CONFIG.sma_slow_period,
        STRATEGY_CONFIG.atr_period,
        STRATEGY_CONFIG.trailing_ema_period,
        STRATEGY_CONFIG.trend_lookback,
        STRATEGY_CONFIG.pullback_lookback,
        STRATEGY_CONFIG.pullback_min_count,
        STRATEGY_CONFIG.proximity_atr_multiple,
        STRATEGY_CONFIG.tick_size,
        STRATEGY_CONFIG.sl_atr_buffer,
        STRATEGY_CONFIG.target_r_multiple,
    )

    stats, strategy_obj = run_backtest(data_path, dataset_key=dataset_key)
    save_outputs(stats, output_paths, strategy_obj=strategy_obj)

    summary_keys = [
        "Start",
        "End",
        "Duration",
        "Equity Final [$]",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Max. Drawdown [%]",
        "Win Rate [%]",
        "# Trades",
        "Profit Factor",
        "Sharpe Ratio",
    ]
    for key in summary_keys:
        if key in stats.index:
            logging.info("%s: %s", key, stats[key])


if __name__ == "__main__":
    main()
