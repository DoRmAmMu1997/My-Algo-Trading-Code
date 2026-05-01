"""
Backtest Nifty Heikin Ashi + Bollinger strategy (mirrors front-test logic)
using backtesting.py.

Flow:
1. Read 1-minute OHLC CSV.
2. Detect start/end date from the CSV itself.
3. Restrict entries to intraday window (09:15 to before 15:15).
4. Force square-off any open position at/after 15:15 each day.
5. Enforce daily max-loss cap (2% of capital); halt for that day if breached.
6. Build Heikin Ashi candles + Bollinger Bands and apply touch/reversal rules.
"""

import argparse
import logging
import os
from datetime import time as dt_time

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


# -----------------------------
# User configuration
# -----------------------------
DATA_PATH = os.path.join("Backtest Outputs", "nifty_renko_futures_5y_1min_data.csv")
OUTPUT_DIR = "Backtest Outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "nifty_heiken_ashi_futures_5y_backtest.log")

# Starting capital: 2 lakh INR
STARTING_CAPITAL = 200000
# Lot size and quantity (1 lot)
LOT_SIZE = 65
LOTS = 1
POSITION_SIZE = LOT_SIZE * LOTS

# Futures-style margin approximation so 1-lot exposure is feasible with 2L capital.
MARGIN_REQUIREMENT = 0.15
# If configured margin cannot support 1 lot with given capital, auto-adjust.
AUTO_ADJUST_MARGIN = True
# Small floor to avoid unrealistic infinite leverage.
MIN_MARGIN_FLOOR = 0.02

# Strategy parameters copied from front-test file.
BOLL_PERIOD = 20
BOLL_STD = 2.0
MIN_BARS = 120

# Intraday session controls:
# - Fresh entries are allowed only during this window.
# - Any open position is force-closed at/after square-off time.
ENTRY_START_TIME = dt_time(9, 15)
SQUARE_OFF_TIME = dt_time(15, 15)

# Daily risk control:
# If day loss reaches this % of capital, stop trading for that day.
DAILY_MAX_LOSS_PCT = 0.02


def setup_logging(log_path: str) -> None:
    """Log to both console and file for easier analysis."""
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
    Load OHLC CSV, clean it, and convert to backtesting.py format.

    Important:
    - No hard-coded 5-year filter is applied here.
    - Whatever period exists in the CSV becomes the tested period.
      The exact start/end timestamps are logged inside `run_backtest()`.

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
    v_col = _find_first_col(raw, ["volume"])
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
            "volume": pd.to_numeric(raw[v_col], errors="coerce") if v_col else 0.0,
        }
    ).dropna(subset=["timestamp", "open", "high", "low", "close"])

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) < MIN_BARS:
        raise ValueError(f"Need at least {MIN_BARS} rows, got {len(df)}")

    bt_df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    bt_df = bt_df.set_index("timestamp")
    bt_df["Volume"] = bt_df["Volume"].fillna(0.0)
    return bt_df


def build_heikin_ashi_arrays(open_, high_, low_, close_):
    """
    Convert OHLC vectors to Heikin Ashi vectors using classic formulas.
    """
    ha_close = (open_ + high_ + low_ + close_) / 4.0
    ha_open = np.zeros_like(ha_close, dtype=float)
    if len(ha_open) == 0:
        return ha_open, ha_close, ha_open, ha_open

    ha_open[0] = (open_[0] + close_[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([high_, ha_open, ha_close])
    ha_low = np.minimum.reduce([low_, ha_open, ha_close])
    return ha_open, ha_close, ha_high, ha_low


class NiftyHeikenAshiFutures5YStrategy(Strategy):
    """
    backtesting.py strategy class implementing the front-test HA+Bollinger logic.

    Notes:
    - Entries are only within 09:15 <= t < 15:15.
    - Any open position is squared off at/after 15:15.
    - If daily loss reaches 2% of capital, trading halts for that day.
    - Entry affordability auto-adjust uses current running equity, not fixed
      starting capital.
    """

    lot_size = POSITION_SIZE

    def init(self):
        """
        Precompute HA + Bollinger arrays and initialize runtime state.

        We precompute arrays once in `init()` for performance. The strategy then
        reads values by current index in `next()`.
        """
        open_ = np.asarray(self.data.Open, dtype=float)
        high_ = np.asarray(self.data.High, dtype=float)
        low_ = np.asarray(self.data.Low, dtype=float)
        close_ = np.asarray(self.data.Close, dtype=float)

        (
            self.ha_open_arr,
            self.ha_close_arr,
            self.ha_high_arr,
            self.ha_low_arr,
        ) = build_heikin_ashi_arrays(open_, high_, low_, close_)

        ha_close_series = pd.Series(self.ha_close_arr)
        bb_mid = ha_close_series.rolling(BOLL_PERIOD).mean()
        bb_std = ha_close_series.rolling(BOLL_PERIOD).std(ddof=0)
        self.bb_mid_arr = bb_mid.to_numpy()
        self.bb_upper_arr = (bb_mid + (BOLL_STD * bb_std)).to_numpy()
        self.bb_lower_arr = (bb_mid - (BOLL_STD * bb_std)).to_numpy()

        # Signal memory (same behavior as front-test):
        # once a band touch is seen, wait for middle-band confirmation.
        self.lower_touch_latched = False
        self.upper_touch_latched = False

        # Day/session state for intraday trading controls.
        self.current_day = None
        # Mark-to-market equity snapshot taken at first bar of each day.
        self.day_start_equity = None
        # Absolute INR cap for intraday loss (2% of configured capital).
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        # If True, strategy is blocked for remaining bars of current day.
        self.day_trading_blocked = False
        # Per-date audit trail of worst intraday drawdown encountered.
        self.daily_loss_tracker = {}
        # Prevent duplicate square-off close requests in same day.
        self.square_off_requested_date = None
        # Diagnostics for runtime affordability auto-adjust behavior.
        self.runtime_margin_adjust_count = 0
        self.runtime_margin_reject_count = 0

    def _consume_long_setup(self):
        """Reset latches after successful long entry."""
        self.lower_touch_latched = False
        self.upper_touch_latched = False

    def _consume_short_setup(self):
        """Reset latches after successful short entry."""
        self.upper_touch_latched = False
        self.lower_touch_latched = False

    def _resolve_runtime_margin(self, entry_price: float):
        """
        Resolve entry affordability using current running equity.

        Why this exists:
        - User requested auto-adjust to use *current* capital at entry time.
        - backtesting.py margin is static for the full run, so we enforce this
          check in strategy logic using live `self.equity`.
        """
        if entry_price <= 0:
            return False, MARGIN_REQUIREMENT, 0.0, float(self.equity), False

        current_equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        required_default = notional * MARGIN_REQUIREMENT

        # Preferred path: configured margin is affordable with current equity.
        if required_default <= current_equity:
            return True, MARGIN_REQUIREMENT, required_default, current_equity, False

        # If auto-adjust is disabled, reject this entry.
        if not AUTO_ADJUST_MARGIN:
            return False, MARGIN_REQUIREMENT, required_default, current_equity, False

        # Auto-adjust path:
        # pick a dynamic margin based on current equity (with safety buffer),
        # bounded by minimum floor.
        affordable_margin = (current_equity * 0.98) / notional if notional > 0 else 0.0
        dynamic_margin = max(MIN_MARGIN_FLOOR, affordable_margin)
        required_dynamic = notional * dynamic_margin
        can_trade = required_dynamic <= current_equity
        return can_trade, dynamic_margin, required_dynamic, current_equity, True

    def _place_entry_if_affordable(self, direction: str, entry_price: float) -> bool:
        """
        Submit one-lot entry only if affordable under runtime capital rules.
        """
        can_trade, used_margin, req_margin, equity_now, was_adjusted = self._resolve_runtime_margin(
            entry_price
        )
        if not can_trade:
            self.runtime_margin_reject_count += 1
            logging.info(
                "Skipping %s entry | price=%.2f | equity=%.2f | required_margin=%.2f",
                direction,
                float(entry_price),
                equity_now,
                req_margin,
            )
            return False

        if was_adjusted and used_margin < MARGIN_REQUIREMENT:
            self.runtime_margin_adjust_count += 1
            logging.info(
                "Runtime margin auto-adjust | direction=%s | price=%.2f | equity=%.2f | "
                "configured_margin=%.4f | used_margin=%.4f | required_margin=%.2f",
                direction,
                float(entry_price),
                equity_now,
                MARGIN_REQUIREMENT,
                used_margin,
                req_margin,
            )

        if direction == "LONG":
            self.buy(size=self.lot_size)
            return True

        if direction == "SHORT":
            self.sell(size=self.lot_size)
            return True

        return False

    def next(self):
        """
        Main bar-by-bar engine:
        - day state reset
        - daily max-loss check and halt
        - daily square-off and entry-window gate
        - band-touch latch updates
        - middle-band confirmation
        - flat entry / active reversal
        """
        n = len(self.data.Close)
        i = n - 1
        if n < max(MIN_BARS, BOLL_PERIOD + 2):
            return

        ts = pd.Timestamp(self.data.index[-1])
        bar_date = ts.date()
        bar_time = ts.time()

        # New day:
        # 1) reset setup latches (day-scoped behavior),
        # 2) reset daily risk guard and capture day-start equity baseline,
        # 3) initialize that day's max-loss tracker value.
        if self.current_day != bar_date:
            self.current_day = bar_date
            self.lower_touch_latched = False
            self.upper_touch_latched = False
            self.day_start_equity = float(self.equity)
            self.day_trading_blocked = False
            self.square_off_requested_date = None
            self.daily_loss_tracker[bar_date] = 0.0

        # Daily max-loss is measured from day-start mark-to-market equity.
        # This captures both realized and unrealized drawdown during the day.
        if self.day_start_equity is None:
            self.day_start_equity = float(self.equity)
        current_equity = float(self.equity)
        day_loss = max(0.0, self.day_start_equity - current_equity)

        # Track worst intraday loss for each date.
        prev_day_max_loss = float(self.daily_loss_tracker.get(bar_date, 0.0))
        if day_loss > prev_day_max_loss:
            self.daily_loss_tracker[bar_date] = float(day_loss)

        # Breach loss cap -> stop for this day.
        # If a position is open, it is closed immediately and no new entries
        # are allowed until the next trading day.
        if (not self.day_trading_blocked) and day_loss >= self.day_loss_limit:
            self.day_trading_blocked = True
            logging.warning(
                "Daily loss cap hit | date=%s | time=%s | day_loss=%.2f | cap=%.2f",
                bar_date,
                bar_time,
                day_loss,
                self.day_loss_limit,
            )

        if self.day_trading_blocked:
            if self.position:
                self.position.close()
            return

        # Time policy:
        # - at/after 15:15: force close any open trade once and stop processing.
        # - before 09:15: do nothing.
        # Net effect: entries are only possible in [09:15, 15:15).
        if bar_time >= SQUARE_OFF_TIME:
            if self.position and self.square_off_requested_date != bar_date:
                self.position.close()
                self.square_off_requested_date = bar_date
            return

        # Keep strategy strictly intraday.
        if bar_time < ENTRY_START_TIME:
            return

        cur_ha_close = float(self.ha_close_arr[i])
        cur_ha_low = float(self.ha_low_arr[i])
        cur_ha_high = float(self.ha_high_arr[i])
        cur_bb_mid = float(self.bb_mid_arr[i])
        cur_bb_lower = float(self.bb_lower_arr[i])
        cur_bb_upper = float(self.bb_upper_arr[i])

        prev_ha_close = float(self.ha_close_arr[i - 1])
        prev_bb_mid = float(self.bb_mid_arr[i - 1])

        if (
            np.isnan(cur_bb_mid)
            or np.isnan(cur_bb_lower)
            or np.isnan(cur_bb_upper)
            or np.isnan(prev_bb_mid)
        ):
            return

        # Latch setup states when bands are touched.
        # Latches persist until consumed by a valid entry/reversal.
        if cur_ha_low <= cur_bb_lower:
            self.lower_touch_latched = True
        if cur_ha_high >= cur_bb_upper:
            self.upper_touch_latched = True

        # Middle-band confirmation logic from front-test:
        # - Long: after lower-touch context, cross back above middle band.
        # - Short: after upper-touch context, cross back below middle band.
        long_signal = (
            self.lower_touch_latched
            and prev_ha_close < prev_bb_mid
            and cur_ha_close >= cur_bb_mid
        )
        short_signal = (
            self.upper_touch_latched
            and prev_ha_close > prev_bb_mid
            and cur_ha_close <= cur_bb_mid
        )

        if long_signal and short_signal:
            if self.position and self.position.is_long:
                long_signal = False
            elif self.position and self.position.is_short:
                short_signal = False
            else:
                return

        if not self.position:
            if long_signal:
                if self._place_entry_if_affordable("LONG", cur_ha_close):
                    self._consume_long_setup()
                return
            if short_signal:
                if self._place_entry_if_affordable("SHORT", cur_ha_close):
                    self._consume_short_setup()
                return
            return

        if self.position.is_long and short_signal:
            self.position.close()
            if self._place_entry_if_affordable("SHORT", cur_ha_close):
                self._consume_short_setup()
            return

        if self.position.is_short and long_signal:
            self.position.close()
            if self._place_entry_if_affordable("LONG", cur_ha_close):
                self._consume_long_setup()
            return


def save_outputs(stats, output_dir: str) -> None:
    """
    Write backtest artifacts to disk.

    Files generated:
    - trades CSV
    - daily close-equity CSV
    - raw stats text dump
    - daily max-loss tracker CSV (for risk-cap audit)
    """
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "nifty_heiken_ashi_futures_5y_trades.csv")
    daily_equity_path = os.path.join(
        output_dir, "nifty_heiken_ashi_futures_5y_daily_equity.csv"
    )
    stats_path = os.path.join(output_dir, "nifty_heiken_ashi_futures_5y_stats.txt")
    daily_loss_path = os.path.join(output_dir, "nifty_heiken_ashi_futures_daily_max_loss.csv")

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(trades_path, index=False)

    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, errors="coerce")
        eq = eq[eq.index.notna()]
        daily = eq[["Equity"]].resample("D").last().dropna()
        daily.to_csv(daily_equity_path, index_label="date")

    with open(stats_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(str(stats))

    # Pull strategy instance from stats to export per-day max-loss diagnostics.
    strategy_obj = stats.get("_strategy", None)
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

    logging.info("Trades saved: %s", trades_path)
    logging.info("Daily equity saved: %s", daily_equity_path)
    logging.info("Stats saved: %s", stats_path)


def run_backtest(data_path: str) -> pd.Series:
    """
    Load data, configure Backtest, run strategy, and return stats.

    Includes a margin configuration step compatible with runtime auto-adjust:
    affordability is evaluated on each entry using current `self.equity`.
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

    # Engine margin setup:
    # - When runtime auto-adjust is ON, keep backtesting engine margin at the
    #   minimum floor and enforce true affordability inside strategy on each bar
    #   using current running equity.
    # - When OFF, keep configured static margin and require starting capital to
    #   support at least one-lot margin at the maximum observed close.
    if AUTO_ADJUST_MARGIN:
        effective_margin = MIN_MARGIN_FLOOR
        logging.info(
            "Runtime auto-adjust enabled | engine_margin_floor=%.4f | "
            "entry affordability uses running equity",
            effective_margin,
        )
    else:
        effective_margin = MARGIN_REQUIREMENT
        max_close = float(data["Close"].max())
        max_notional = max_close * POSITION_SIZE
        required_margin_at_config = max_notional * MARGIN_REQUIREMENT
        if required_margin_at_config > STARTING_CAPITAL:
            raise ValueError(
                "No trades possible with AUTO_ADJUST_MARGIN=False: configured margin is too high "
                f"for 1-lot size and starting cash. Required={required_margin_at_config:.2f}, "
                f"cash={STARTING_CAPITAL:.2f}"
            )

    bt = Backtest(
        data,
        NiftyHeikenAshiFutures5YStrategy,
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
            "Runtime margin counters | auto_adjust_entries=%s | rejected_entries=%s",
            getattr(strategy_obj, "runtime_margin_adjust_count", "NA"),
            getattr(strategy_obj, "runtime_margin_reject_count", "NA"),
        )
    return stats


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Nifty Heikin Ashi futures backtest using backtesting.py"
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
        "auto_adjust_margin=%s | entry_start=%s | square_off=%s | "
        "daily_max_loss_pct=%s | daily_max_loss_amt=%s",
        STARTING_CAPITAL,
        LOT_SIZE,
        LOTS,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        AUTO_ADJUST_MARGIN,
        ENTRY_START_TIME,
        SQUARE_OFF_TIME,
        DAILY_MAX_LOSS_PCT,
        STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
    )

    stats = run_backtest(args.data)
    save_outputs(stats, OUTPUT_DIR)

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
