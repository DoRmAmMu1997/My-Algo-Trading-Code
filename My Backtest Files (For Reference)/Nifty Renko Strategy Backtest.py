"""
Backtest a Nifty Renko strategy using backtesting.py.

Flow:
1. Read 1-minute OHLC CSV.
2. Detect start/end date from the CSV itself.
3. Build Renko candles + EMA indicators using shared logic file.
4. Ask shared signal engine for entry/exit decisions.
5. Restrict fresh entries to the allowed intraday window (09:15 to before 15:15).
6. Block fresh entries during the midday no-trade window (11:00 AM to 1:30 PM).
7. Force square-off any open position at/after 15:15 each day.
8. Enforce daily max-loss cap (2% of capital); halt for that day if breached.
9. Save stats, trades, and daily equity.

Beginner mental model:
- `load_ohlc_data()` prepares clean historical candles.
- `NiftyRenkoFutures5YStrategy.next()` is called bar-by-bar by the framework.
- Shared Renko logic file decides entries/exits.
- `run_backtest()` wires config + engine together and executes the run.
- `save_outputs()` exports results for analysis in spreadsheets or notebooks.
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

# Strategy logic lives in the sibling `Signal Generators/` folder. Adding it
# to sys.path here (computed from __file__, not cwd) lets the import below
# resolve regardless of which directory the backtest is launched from.
_SIGNAL_GENERATORS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Signal Generators")
)
if _SIGNAL_GENERATORS_DIR not in sys.path:
    sys.path.insert(0, _SIGNAL_GENERATORS_DIR)

from renko_strategy_logic_9_21 import (
    RenkoPositionContext,
    RenkoSignalEngine,
    build_renko_with_indicators,
)


# -----------------------------
# User configuration
# -----------------------------
DATA_PATH = os.path.join("Backtest Outputs", "nifty_renko_futures_5y_1min_data.csv")
OUTPUT_DIR = "Backtest Outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "nifty_renko_futures_5y_backtest.log")

# Starting capital: 2.5 lakh INR
STARTING_CAPITAL = 600000
# Lot size and quantity (1 lot)
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
# - Fresh entries are blocked again during the midday pause inside this window.
# - Any open position is force-closed at/after square-off time.
ENTRY_START_TIME = dt_time(9, 15)
MIDDAY_NO_TRADE_START_TIME = dt_time(11, 30)
MIDDAY_NO_TRADE_END_TIME = dt_time(13, 00)
SQUARE_OFF_TIME = dt_time(15, 15)

# Daily risk control:
# If day loss reaches this % of capital, stop trading for that day.
DAILY_MAX_LOSS_PCT = 0.03


# Load .env values from this strategy folder only (keeps config local and explicit).
# Why absolute path from `__file__`:
# - script can be run from any working directory
# - config resolution remains stable and predictable
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _env_float(name: str, default: float) -> float:
    """
    Read numeric config from environment safely.

    Beginner note:
    - If variable is missing or invalid text, we return `default`.
    - This avoids backtest crashes due to config typos.
    """
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# Per closed trade, this fixed amount is deducted from gross trade PnL.
# Example:
# - raw trade PnL = 450
# - BROKERAGE = 80
# - final trade PnL = 370
# Supports typo alias `BROGERAGE` for backward compatibility.
BROKERAGE = _env_float("BROKERAGE", _env_float("BROGERAGE", 80.0))


def setup_logging(log_path: str) -> None:
    """Log to both console and file for easy debugging and audit."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def _find_first_col(df: pd.DataFrame, names):
    """
    Return first matching column name (case-insensitive).

    Useful when CSV exports use variants like `DateTime`, `datetime`, or `date`.
    """
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

    # Parse types safely. Rows with invalid values are dropped.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col], errors="coerce"),
            "open": pd.to_numeric(raw[o_col], errors="coerce"),
            "high": pd.to_numeric(raw[h_col], errors="coerce"),
            "low": pd.to_numeric(raw[l_col], errors="coerce"),
            "close": pd.to_numeric(raw[c_col], errors="coerce"),
        }
    ).dropna()

    # Keep data sorted and unique by timestamp.
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS:
        raise ValueError(f"Need at least {MIN_BARS} rows, got {len(df)}")

    # backtesting.py expects these exact OHLC column names.
    bt_df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    bt_df = bt_df.set_index("timestamp")
    # Volume is not used in this strategy, but the framework supports it.
    bt_df["Volume"] = 0
    return bt_df


class NiftyRenkoFutures5YStrategy(Strategy):
    """
    backtesting.py strategy class for Renko futures backtesting.

    Notes:
    - `next()` runs once per incoming bar.
    - All Renko signal rules come from shared `RenkoSignalEngine`.
    - LONG path = bullish logic, SHORT path = bearish logic.
    - Only one position at a time.
    - Daily loss guard can halt trading for the remaining bars of the day.
    - Entry affordability check uses current equity (dynamic auto-adjust margin).
    """
    lot_size = POSITION_SIZE

    def init(self):
        """
        Initialize per-run state used by the backtest loop.

        This includes:
        - Shared strategy engine state
        - Active trade helper values
        - Diagnostic counters
        - Per-day risk tracking values
        """
        # Guards to avoid processing same Renko candle more than once.
        self.last_processed_candle_ts = None
        self.strategy_engine = RenkoSignalEngine()

        # Active trade state used by exit logic.
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.rr_armed = False
        # Diagnostic counters for post-run sanity checks.
        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        self.square_off_count = 0
        self.daily_loss_halt_count = 0
        self.margin_skip_count = 0
        # Stores max intraday loss observed for each date.
        self.daily_loss_tracker = {}
        self.margin_skip_logged_dates = set()
        # Track date for which 15:15 square-off was already requested.
        self.square_off_requested_date = None
        # Per-day loss tracking state.
        self.current_day = None
        self.day_start_equity = None
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        self.day_trading_blocked = False

    def _reset_trade_state(self):
        """Clear active-trade helper variables after exit."""
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.rr_armed = False

    def _reset_reentry_flags(self):
        """Clear pullback memory flags."""
        self.strategy_engine.reset_reentry_flags()

    def _mark_closed_trade_direction(self):
        """Update engine with most recent closed trade direction."""
        if self.trade_direction in ("LONG", "SHORT"):
            self.strategy_engine.update_previous_trade_direction(self.trade_direction)

    def _can_afford_entry(self, entry_price: float):
        """
        Check if a new 1-lot entry is affordable using current running equity.

        Returns:
            (can_enter, effective_margin, required_margin, equity, notional)
        """
        equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        if notional <= 0:
            return False, MARGIN_REQUIREMENT, 0.0, equity, notional

        if AUTO_ADJUST_MARGIN:
            # Dynamic margin auto-adjust is computed at entry-time from current equity.
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

    def next(self):
        """
        Main strategy loop called by backtesting.py for every new bar.

        Sequence:
        - Enforce intraday time rules (entry window, midday entry block, and 3:15 square-off)
        - Enforce per-day max loss cap (2% of capital)
        - Warm-up check
        - Rebuild Renko + indicators from history so far (shared utility)
        - Process one fresh Renko candle only once
        - Ask shared signal engine for one decision
        - If in trade: only process exit decisions
        - If flat: process entry decisions only outside 11:00-13:30 (subject to margin rules)
        """
        # Safety: if engine is flat but local state wasn't cleared, reset it.
        if not self.position and self.trade_direction:
            self._mark_closed_trade_direction()
            self._reset_trade_state()

        # Current engine bar timestamp used for time-based rules.
        bar_ts = pd.Timestamp(self.data.index[-1])
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        # New trading day: reset day-level guards and baseline equity.
        if self.current_day != bar_date:
            self.current_day = bar_date
            # Baseline equity for daily loss computation.
            self.day_start_equity = float(self.equity)
            # Re-enable trading each new day unless a new halt condition is hit.
            self.day_trading_blocked = False
            self.square_off_requested_date = None
            self.daily_loss_tracker[bar_date] = 0.0

        # Day loss is measured from start-of-day equity (mark-to-market).
        # This includes both realized and unrealized PnL impact in equity.
        if self.day_start_equity is None:
            self.day_start_equity = float(self.equity)
        current_equity = float(self.equity)
        day_loss = max(0.0, self.day_start_equity - current_equity)
        # Keep a running record of worst intraday drawdown for each date.
        prev_day_max_loss = float(self.daily_loss_tracker.get(bar_date, 0.0))
        if day_loss > prev_day_max_loss:
            self.daily_loss_tracker[bar_date] = float(day_loss)

        # If daily loss cap is breached, halt trading for the rest of the day.
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
            # Clear pullback memory to avoid carrying stale context after risk halt.
            self._reset_reentry_flags()

        if self.day_trading_blocked:
            # If a position is open when cap is breached, close it immediately.
            if self.position:
                self.position.close()
                self.exit_count += 1
                self._mark_closed_trade_direction()
                self._reset_trade_state()
            return

        # Force square-off at/after 15:15 every day.
        if bar_time >= SQUARE_OFF_TIME:
            if self.position and self.square_off_requested_date != bar_date:
                self.position.close()
                self.exit_count += 1
                self.square_off_count += 1
                self._mark_closed_trade_direction()
                self._reset_trade_state()
                self.square_off_requested_date = bar_date
            return

        # No entries/signals before market start.
        # We return early so this strategy remains strictly 09:15-15:15 intraday.
        if bar_time < ENTRY_START_TIME:
            return

        # `self.data` grows one bar at a time as framework iterates the dataset.
        n = len(self.data.Close)
        if n < MIN_BARS:
            return

        # Rebuild Renko from all bars up to this point.
        ohlc = self._current_ohlc(n)
        renko = build_renko_with_indicators(ohlc)
        if renko.empty or len(renko) < 3:
            return

        # Only act once per newly formed Renko candle.
        last_ts = renko.iloc[-1]["timestamp"]
        if self.last_processed_candle_ts == last_ts:
            return
        self.last_processed_candle_ts = last_ts

        position_ctx = None
        if self.position:
            position_ctx = RenkoPositionContext(
                direction=self.trade_direction,
                entry_underlying=self.entry_underlying,
                stop_underlying=self.stop_underlying,
                rr_armed=self.rr_armed,
            )
        # Exactly one decision is returned per newly formed Renko candle.
        decision = self.strategy_engine.evaluate_candle(renko, position=position_ctx)

        if self.position:
            self.rr_armed = bool(decision.rr_armed)
            if decision.action == "EXIT":
                self.position.close()
                self.exit_count += 1
                self._mark_closed_trade_direction()
                self._reset_trade_state()
            return

        # Entry window is [09:15, 15:15). At 15:15 we always square-off instead.
        if not (ENTRY_START_TIME <= bar_time < SQUARE_OFF_TIME):
            return

        # `signal_triggered` counts raw strategy signals, even when trade is rejected
        # later due to time filters, invalid stop, or insufficient margin.
        if decision.signal_triggered:
            self.signal_count += 1

        # Midday block applies only to fresh entries.
        # Open-position exit handling already happened above and is unaffected.
        if MIDDAY_NO_TRADE_START_TIME <= bar_time < MIDDAY_NO_TRADE_END_TIME:
            return

        if decision.action == "ENTER_LONG":
            c = float(decision.entry_underlying)
            stop = float(decision.stop_underlying)
            can_enter, eff_margin, req_margin, equity_now, _ = self._can_afford_entry(c)
            if can_enter:
                self.buy(size=self.lot_size, sl=stop)
                self.entry_submit_count += 1
                self.trade_direction = "LONG"
                self.entry_underlying = c
                self.stop_underlying = stop
                self.rr_armed = False
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
            c = float(decision.entry_underlying)
            stop = float(decision.stop_underlying)
            can_enter, eff_margin, req_margin, equity_now, _ = self._can_afford_entry(c)
            if can_enter:
                self.sell(size=self.lot_size, sl=stop)
                self.entry_submit_count += 1
                self.trade_direction = "SHORT"
                self.entry_underlying = c
                self.stop_underlying = stop
                self.rr_armed = False
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

    Files created:
    - trades CSV (one row per closed trade)
    - daily equity CSV (end-of-day equity points)
    - daily max-loss CSV (worst intraday loss per day, if available)
    - text dump of full `stats` object
    """
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "nifty_renko_futures_5y_trades.csv")
    daily_equity_path = os.path.join(output_dir, "nifty_renko_futures_5y_daily_equity.csv")
    stats_path = os.path.join(output_dir, "nifty_renko_futures_5y_stats.txt")
    daily_loss_path = os.path.join(output_dir, "nifty_renko_futures_daily_max_loss.csv")

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        # Never mutate framework-owned table directly; write a derived copy.
        trades_to_save = trades.copy()
        if "PnL" in trades_to_save.columns:
            # Parse PnL safely so numeric math is robust even if CSV/object has mixed types.
            pnl_values = pd.to_numeric(trades_to_save["PnL"], errors="coerce")
            # Keep brokerage explicit per row so each trade record is self-explanatory.
            trades_to_save["Brokerage"] = float(BROKERAGE)
            # Net trade result after fixed brokerage deduction.
            trades_to_save["FinalPnL"] = pnl_values - float(BROKERAGE)
            # Running cumulative net result after brokerage-adjusted trade outcomes.
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
            logging.warning(
                "Trades output missing `PnL` column; brokerage adjustment not applied."
            )
        trades_to_save.to_csv(trades_path, index=False)

    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, errors="coerce")
        eq = eq[eq.index.notna()]
        # Save daily last equity for easy curve review in spreadsheet tools.
        daily = eq[["Equity"]].resample("D").last().dropna()
        daily.to_csv(daily_equity_path, index_label="date")

    with open(stats_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(str(stats))

    if strategy_obj is not None:
        tracker = getattr(strategy_obj, "daily_loss_tracker", None)
        if isinstance(tracker, dict) and tracker:
            daily_loss_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(list(tracker.keys())),
                    "max_intraday_loss": [float(v) for v in tracker.values()],
                    "daily_loss_cap": STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
                }
            ).sort_values("date")
            daily_loss_df.to_csv(daily_loss_path, index=False, date_format="%Y-%m-%d")
            logging.info("Daily max-loss tracker saved: %s", daily_loss_path)

    logging.info(f"Trades saved: {trades_path}")
    logging.info(f"Daily equity saved: {daily_equity_path}")
    logging.info(f"Stats saved: {stats_path}")


def run_backtest(data_path: str):
    """
    Load data, configure Backtest engine, run strategy, and return stats + strategy object.

    Returning the strategy object allows callers to read custom counters after run completion.
    """
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

    # Engine-level margin is static in backtesting.py.
    # For dynamic (current-equity-based) entry affordability, strategy-level checks are used.
    # To avoid engine rejecting orders before dynamic checks, keep engine margin at floor.
    if AUTO_ADJUST_MARGIN:
        effective_margin = MIN_MARGIN_FLOOR
        logging.info(
            "Dynamic margin auto-adjust enabled | engine_margin_floor=%.4f | "
            "configured_margin=%.4f",
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
        NiftyRenkoFutures5YStrategy,
        cash=STARTING_CAPITAL,
        margin=effective_margin,
        commission=0.0,
        trade_on_close=True,
        exclusive_orders=True,
        hedging=False,
        finalize_trades=True,
    )
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
        logging.warning(
            "Backtest finished with 0 closed trades. Check entry conditions/data regime."
        )
    return stats, strategy_obj


def main():
    """
    CLI entrypoint.

    Usage:
    `python "Nifty Renko Strategy Backtest.py" --data "path_to_csv.csv"`
    """
    parser = argparse.ArgumentParser(
        description="Nifty Renko futures backtest using backtesting.py"
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to OHLC CSV with timestamp/open/high/low/close columns.",
    )
    args = parser.parse_args()

    setup_logging(LOG_FILE)
    logging.info(
        "Config | capital=%s | lot_size=%s | lots=%s | qty=%s | margin=%s | "
        "auto_adjust_margin=%s | daily_max_loss_pct=%s | daily_max_loss_amt=%s | "
        "brokerage_per_trade=%s",
        STARTING_CAPITAL,
        LOT_SIZE,
        LOTS,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        AUTO_ADJUST_MARGIN,
        DAILY_MAX_LOSS_PCT,
        STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
        BROKERAGE,
    )

    stats, strategy_obj = run_backtest(args.data)
    save_outputs(stats, OUTPUT_DIR, strategy_obj=strategy_obj)

    # Print the most useful top-level stats.
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
