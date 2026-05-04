"""
Backtest the Nifty EMA trend strategy using backtesting.py.

Flow:
1. Read 1-minute OHLC CSV.
2. Detect start/end date from the CSV itself.
3. Build EMA/ATR/ADX indicator columns using the shared EMA logic file.
4. Ask the shared signal engine for entry/exit decisions.
5. Restrict fresh entries to the allowed intraday window (09:15 to before 15:15).
6. Force square-off any open position at/after 15:15 each day.
7. Enforce daily max-loss cap (3% of capital); halt for that day if breached.
8. Save stats, trades, and daily equity.

Beginner mental model:
- `load_ohlc_data()` prepares clean historical candles.
- `NiftyEMATrendFutures5YStrategy.next()` is called bar-by-bar by the framework.
- Shared EMA logic decides entries and EMA11 exits.
- `run_backtest()` wires config + engine together and executes the run.
- `save_outputs()` exports results for later spreadsheet review.

Important beginner note about this file:
- It does not contain the EMA rules themselves.
- Those rules live in `ema_trend_strategy_logic.py`.
- This file is mostly about "when should we ask for a signal?" and
  "how should we simulate entries, exits, capital, and reporting?"
"""

import argparse
import logging
import os
import sys
from datetime import time as dt_time

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

from ema_trend_strategy_logic import (
    EMATrendConfig,
    EMATrendPositionContext,
    EMATrendSignalEngine,
    build_ema_trend_with_indicators,
)


# -----------------------------
# User configuration
# -----------------------------
# Reuses the same 1-minute OHLC dataset used by the Renko backtest by default.
DATA_PATH = os.path.join(_REPO_ROOT, "Backtest Outputs", "nifty_renko_futures_5y_1min_data.csv")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Backtest Outputs")
LOG_FILE = os.path.join(OUTPUT_DIR, "nifty_ema_trend_futures_5y_backtest.log")

# Starting capital: 6 lakh INR
STARTING_CAPITAL = 600000
# Lot size and quantity (3 lots).
LOT_SIZE = 65
LOTS = 3
POSITION_SIZE = LOT_SIZE * LOTS
# Margin is used by backtesting.py to allow leveraged exposure.
MARGIN_REQUIREMENT = 0.15
# If enabled, entry affordability is auto-adjusted per trade using current equity.
AUTO_ADJUST_MARGIN = True
# Small floor to avoid unrealistic infinite leverage (also used as engine margin floor).
MIN_MARGIN_FLOOR = 0.02
# Warm-up threshold before strategy can operate reliably.
MIN_BARS = 120

# Intraday session controls:
# - Fresh entries are allowed only during this window.
# - Any open position is force-closed at/after square-off time.
ENTRY_START_TIME = dt_time(9, 25)
SQUARE_OFF_TIME = dt_time(15, 15)

# Daily risk control:
# If day loss reaches this % of capital, stop trading for that day.
DAILY_MAX_LOSS_PCT = 0.03


# Load .env values from this strategy folder only (keeps config local and explicit).
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


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


STRATEGY_CONFIG = EMATrendConfig(
    # These environment overrides let you tune the shared EMA strategy without
    # editing Python code. If no env value is supplied, the strategy falls back
    # to the default values from the original brief.
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


# Per closed trade, this fixed amount is deducted from gross trade PnL.
# Supports typo alias `BROGERAGE` for backward compatibility.
BROKERAGE = _env_float("BROKERAGE", _env_float("BROGERAGE", 80.0))


def setup_logging(log_path: str) -> None:
    """Log to both console and file for easy debugging and audit."""
    # Logging is especially useful in backtests because it lets you inspect
    # risk halts, margin skips, and summary stats after a long run finishes.
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _find_first_col(df: pd.DataFrame, names):
    """Return first matching column name (case-insensitive)."""
    # Historical CSVs often come from different brokers or export tools, and
    # column names may vary slightly. This helper makes the loader more tolerant.
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def load_ohlc_data(csv_path: str) -> pd.DataFrame:
    """
    Load raw OHLC CSV, clean it, and convert to backtesting.py format.

    Output format:
    - Datetime index
    - Columns: Open, High, Low, Close, Volume
    """
    # Fail fast if the input path is wrong so the user gets a clean error
    # message before the backtest engine is even created.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"Data file is empty: {csv_path}")

    # We detect columns flexibly because different CSV exports may use names
    # like `datetime`, `timestamp`, `date`, or `time`.
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

    # Sorting and de-duplicating ensures the backtest walks through candles in
    # the same clean order a live strategy would see them.
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS:
        raise ValueError(f"Need at least {MIN_BARS} rows, got {len(df)}")

    bt_df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    bt_df = bt_df.set_index("timestamp")
    # Volume is unused by this strategy, but backtesting.py is happy when the
    # common OHLCV shape is present.
    bt_df["Volume"] = 0
    return bt_df


class NiftyEMATrendFutures5YStrategy(Strategy):
    """
    backtesting.py strategy class for the EMA trend futures strategy.

    Notes:
    - `next()` runs once per incoming bar.
    - All EMA signal rules come from shared `EMATrendSignalEngine`.
    - LONG path = bullish logic, SHORT path = bearish logic.
    - Only one position at a time.
    - Daily loss guard can halt trading for the remaining bars of the day.
    - Entry affordability check uses current equity (dynamic auto-adjust margin).
    """

    lot_size = POSITION_SIZE

    def init(self):
        """
        Initialize per-run state used by the backtest loop.

        This method is called once when backtesting.py creates the strategy.
        Everything stored on `self` here becomes state that can be reused on
        every later call to `next()`.
        """
        # We only want to process each completed candle once, even though the
        # framework calls `next()` on every raw bar update.
        self.last_processed_candle_ts = None
        self.strategy_engine = EMATrendSignalEngine(STRATEGY_CONFIG)

        # Trade-state fields mirror the currently open strategy trade.
        # They are kept separate from the framework object so our own logging
        # and signal context remain simple and explicit.
        self.trade_direction = ""
        self.entry_underlying = 0.0
        # Stored for logs/debugging only; actual exit is dynamic when the candle
        # range breaches EMA11 against the trade.
        self.stop_underlying = 0.0

        # These counters are not required for trading, but they are extremely
        # helpful for sanity-checking behavior after a run completes.
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        self.square_off_count = 0
        self.daily_loss_halt_count = 0
        self.margin_skip_count = 0

        # This dictionary stores the worst intraday drawdown seen for each date.
        self.daily_loss_tracker = {}
        self.margin_skip_logged_dates = set()
        self.square_off_requested_date = None

        # These fields reset as the backtest moves from one trading day to the next.
        self.current_day = None
        self.day_start_equity = None
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        self.day_trading_blocked = False

    def _reset_trade_state(self):
        """Clear active-trade helper variables after exit."""
        # Keeping this reset logic in one place reduces the chance that some
        # field gets forgotten after a close.
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0

    def _can_afford_entry(self, entry_price: float):
        """
        Check if a new entry is affordable using current running equity.

        Returns:
            (can_enter, effective_margin, required_margin, equity, notional)
        """
        # `self.equity` is the current account value inside the backtest,
        # including the mark-to-market effect of open positions.
        equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        if notional <= 0:
            return False, MARGIN_REQUIREMENT, 0.0, equity, notional

        if AUTO_ADJUST_MARGIN:
            # This branch makes entries more realistic by checking what margin
            # current equity can actually support, instead of assuming the start
            # capital is always fully available forever.
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
        # backtesting.py exposes price history as array-like objects.
        # We convert the data back into a normal DataFrame because the shared
        # EMA logic expects a standard OHLC table with a timestamp column.
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

    def next(self):
        """
        Main strategy loop called by backtesting.py for every new bar.

        Sequence:
        - Enforce intraday time rules (entry window and 3:15 square-off)
        - Enforce per-day max loss cap
        - Warm-up check
        - Rebuild indicators from history so far
        - Process one fresh candle only once
        - Ask the shared signal engine for one decision
        - If in trade: only process exit decisions
        - If flat: process fresh entry decisions
        """
        # Safety cleanup:
        # If the framework no longer has an open position but our custom state
        # still thinks a trade exists, bring our state back in sync.
        if not self.position and self.trade_direction:
            self._reset_trade_state()

        # `bar_ts` is the timestamp of the latest raw candle currently being
        # processed by backtesting.py. Time-based rules are built from this.
        bar_ts = pd.Timestamp(self.data.index[-1])
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        # At the first bar of each new day, reset day-specific guards and
        # record the account equity we will use as that day's loss baseline.
        if self.current_day != bar_date:
            self.current_day = bar_date
            self.day_start_equity = float(self.equity)
            self.day_trading_blocked = False
            self.square_off_requested_date = None
            self.daily_loss_tracker[bar_date] = 0.0

        # Day loss is measured as "how far current equity has dropped from the
        # start-of-day equity". This naturally includes realized and unrealized
        # PnL because both are reflected in `self.equity`.
        if self.day_start_equity is None:
            self.day_start_equity = float(self.equity)
        current_equity = float(self.equity)
        day_loss = max(0.0, self.day_start_equity - current_equity)
        prev_day_max_loss = float(self.daily_loss_tracker.get(bar_date, 0.0))
        if day_loss > prev_day_max_loss:
            self.daily_loss_tracker[bar_date] = float(day_loss)

        # Once the daily loss cap is hit, the strategy stops initiating or
        # holding trades for the rest of that day.
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
            # If a trade is already open when the loss limit is hit, we close it
            # immediately and skip the rest of the day's logic.
            if self.position:
                self.position.close()
                self.exit_count += 1
                self._reset_trade_state()
            return

        # Square-off is enforced every day so the system stays intraday and
        # never carries a position overnight in the backtest.
        if bar_time >= SQUARE_OFF_TIME:
            if self.position and self.square_off_requested_date != bar_date:
                self.position.close()
                self.exit_count += 1
                self.square_off_count += 1
                self._reset_trade_state()
                self.square_off_requested_date = bar_date
            return

        # The strategy should not trade before the official session start.
        if bar_time < ENTRY_START_TIME:
            return

        # `self.data` grows by one bar at a time. Before enough bars exist,
        # the indicator engine would only be working with warm-up values.
        n = len(self.data.Close)
        if n < MIN_BARS:
            return

        # Rebuild all indicators from the candles seen so far. This mirrors
        # how a live system would make decisions using only available history.
        ohlc = self._current_ohlc(n)
        candles = build_ema_trend_with_indicators(ohlc, STRATEGY_CONFIG)
        if candles.empty:
            return

        # Only act once per latest completed candle. Without this guard, the
        # same signal could be processed repeatedly on the same timestamp.
        last_ts = candles.iloc[-1]["timestamp"]
        if self.last_processed_candle_ts == last_ts:
            return
        self.last_processed_candle_ts = last_ts

        # The shared engine only needs a very small summary of the open trade,
        # not the full backtesting.py position object.
        position_ctx = None
        if self.position:
            position_ctx = EMATrendPositionContext(
                direction=self.trade_direction,
                entry_underlying=self.entry_underlying,
                stop_underlying=self.stop_underlying,
            )
        decision = self.strategy_engine.evaluate_candle(candles, position=position_ctx)

        # If a trade is active, this strategy does not consider new entries.
        # It only asks whether the current trade should be exited.
        if self.position:
            if decision.action == "EXIT":
                self.position.close()
                self.exit_count += 1
                self._reset_trade_state()
            return

        # Entry logic below this point runs only while flat and only during the
        # allowed intraday session window.
        if not (ENTRY_START_TIME <= bar_time < SQUARE_OFF_TIME):
            return

        # `signal_triggered` counts valid strategy setups, even if an order is
        # later skipped because of capital or time constraints.
        if decision.signal_triggered:
            self.signal_count += 1

        if decision.action == "ENTER_LONG":
            entry_price = float(decision.entry_underlying)
            stop = float(decision.stop_underlying)
            can_enter, eff_margin, req_margin, equity_now, _ = self._can_afford_entry(entry_price)
            if can_enter:
                # No hard `sl` order is placed here because the shared strategy
                # logic handles EMA11 breach exits from candle data itself.
                self.buy(size=self.lot_size)
                self.entry_submit_count += 1
                self.trade_direction = "LONG"
                self.entry_underlying = entry_price
                self.stop_underlying = stop
            else:
                self.margin_skip_count += 1
                if bar_date not in self.margin_skip_logged_dates:
                    logging.warning(
                        "Entry skipped (insufficient current equity margin) | "
                        "date=%s | time=%s | side=LONG | equity=%.2f | required=%.2f | "
                        "effective_margin=%.4f | configured_margin=%.4f",
                        bar_date,
                        bar_time,
                        equity_now,
                        req_margin,
                        eff_margin,
                        MARGIN_REQUIREMENT,
                    )
                    self.margin_skip_logged_dates.add(bar_date)
            return

        if decision.action == "ENTER_SHORT":
            entry_price = float(decision.entry_underlying)
            stop = float(decision.stop_underlying)
            can_enter, eff_margin, req_margin, equity_now, _ = self._can_afford_entry(entry_price)
            if can_enter:
                self.sell(size=self.lot_size)
                self.entry_submit_count += 1
                self.trade_direction = "SHORT"
                self.entry_underlying = entry_price
                self.stop_underlying = stop
            else:
                self.margin_skip_count += 1
                if bar_date not in self.margin_skip_logged_dates:
                    logging.warning(
                        "Entry skipped (insufficient current equity margin) | "
                        "date=%s | time=%s | side=SHORT | equity=%.2f | required=%.2f | "
                        "effective_margin=%.4f | configured_margin=%.4f",
                        bar_date,
                        bar_time,
                        equity_now,
                        req_margin,
                        eff_margin,
                        MARGIN_REQUIREMENT,
                    )
                    self.margin_skip_logged_dates.add(bar_date)


def save_outputs(stats, output_dir: str, strategy_obj=None) -> None:
    """
    Write trades, daily equity, daily max-loss tracker, and raw stats snapshot to disk.
    """
    # All output files are written to one folder so post-run analysis is easy.
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "nifty_ema_trend_futures_5y_trades.csv")
    daily_equity_path = os.path.join(output_dir, "nifty_ema_trend_futures_5y_daily_equity.csv")
    stats_path = os.path.join(output_dir, "nifty_ema_trend_futures_5y_stats.txt")
    daily_loss_path = os.path.join(output_dir, "nifty_ema_trend_futures_daily_max_loss.csv")

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        # Copy before modifying so we never mutate the framework-owned object.
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
        # Resample to daily last equity so spreadsheet review is easier.
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


def run_backtest(data_path: str):
    """
    Load data, configure Backtest engine, run strategy, and return stats + strategy object.
    """
    # Step 1: turn a raw CSV into the shape expected by backtesting.py.
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
        # The engine-level margin is kept low here because the real entry
        # affordability check happens inside the strategy on each signal.
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

    # Step 2: create the actual backtesting.py engine with strategy class + data.
    bt = Backtest(
        data,
        NiftyEMATrendFutures5YStrategy,
        cash=STARTING_CAPITAL,
        margin=effective_margin,
        commission=0.0,
        trade_on_close=True,
        exclusive_orders=True,
        hedging=False,
        finalize_trades=True,
    )
    # Step 3: run the full historical simulation.
    stats = bt.run()
    strategy_obj = stats.get("_strategy", None)
    if strategy_obj is not None:
        logging.info(
            "Strategy counters | signals=%s | entry_submissions=%s | exits=%s | "
            "day_square_offs=%s | day_loss_halts=%s | margin_skips=%s",
            getattr(strategy_obj, "signal_count", "NA"),
            getattr(strategy_obj, "entry_submit_count", "NA"),
            getattr(strategy_obj, "exit_count", "NA"),
            getattr(strategy_obj, "square_off_count", "NA"),
            getattr(strategy_obj, "daily_loss_halt_count", "NA"),
            getattr(strategy_obj, "margin_skip_count", "NA"),
        )
    if "# Trades" in stats.index and int(stats["# Trades"]) == 0:
        logging.warning("Backtest finished with 0 closed trades. Check entry conditions/data regime.")
    return stats, strategy_obj


def main():
    """
    CLI entrypoint.

    Usage:
    `python "Nifty EMA Trend Strategy Backtest.py" --data "path_to_csv.csv"`
    """
    parser = argparse.ArgumentParser(
        description="Nifty EMA trend futures backtest using backtesting.py"
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to OHLC CSV with timestamp/open/high/low/close columns.",
    )
    args = parser.parse_args()

    # Logging is configured here so everything from the run gets captured in the file.
    setup_logging(LOG_FILE)
    logging.info(
        "Config | capital=%s | lot_size=%s | lots=%s | qty=%s | margin=%s | "
        "auto_adjust_margin=%s | daily_max_loss_pct=%s | daily_max_loss_amt=%s | "
        "brokerage_per_trade=%s | slope_lookback=%s | adx_threshold=%s",
        STARTING_CAPITAL,
        LOT_SIZE,
        LOTS,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        AUTO_ADJUST_MARGIN,
        DAILY_MAX_LOSS_PCT,
        STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
        BROKERAGE,
        STRATEGY_CONFIG.slope_lookback,
        STRATEGY_CONFIG.adx_threshold,
    )

    # This is the actual backtest execution line.
    stats, strategy_obj = run_backtest(args.data)
    save_outputs(stats, OUTPUT_DIR, strategy_obj=strategy_obj)

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
