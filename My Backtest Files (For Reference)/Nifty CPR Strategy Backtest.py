"""
Backtest the NIFTY CPR strategy using backtesting.py.

Beginner flow:
1. Read a raw OHLC CSV.
2. Accept either 1-minute or 5-minute candles.
3. Convert the data to complete 5-minute CPR candles.
4. Precompute CPR levels, indicators, zones, swings, and divergence markers.
5. Walk bar-by-bar and ask the shared CPR engine for entries/exits.
6. Save trades, equity, daily loss tracker, stats, and logs.

This backtest trades the underlying futures directionally:
- ENTER_LONG  -> buy futures
- ENTER_SHORT -> sell futures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

# The backtest lives in `My Backtest Files (For Reference)`, while the CPR
# signal logic lives in `Signal Generators/CPR Strategy`. Add that folder to
# sys.path so the import below works no matter where this script is run from.
ROOT_DIR = Path(__file__).resolve().parent.parent
CPR_SIGNAL_GENERATOR_DIR = ROOT_DIR / "Signal Generators" / "CPR Strategy"
if str(CPR_SIGNAL_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(CPR_SIGNAL_GENERATOR_DIR))

from cpr_strategy_logic import (
    CPRPositionContext,
    CPRSignalEngine,
    CPRStrategyConfig,
    build_cpr_with_indicators,
    prepare_cpr_ohlc_input,
)


OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"

DEFAULT_DATA_PATHS = {
    "nifty": OUTPUT_DIR / "nifty_renko_futures_5y_1min_data.csv",
    "banknifty": OUTPUT_DIR / "banknifty_renko_futures_5y_1min_data.csv",
    "finnifty": OUTPUT_DIR / "finnifty_renko_futures_5y_1min_data.csv",
}

STARTING_CAPITAL = 600000
LOT_SIZE = 65
LOTS = 3
POSITION_SIZE = LOT_SIZE * LOTS
MARGIN_REQUIREMENT = 0.15
AUTO_ADJUST_MARGIN = True
MIN_MARGIN_FLOOR = 0.02
ENTRY_START_TIME = dt_time(9, 25)
SQUARE_OFF_TIME = dt_time(15, 15)
DAILY_MAX_LOSS_PCT = 0.03


def _env_float(name: str, default: float) -> float:
    """Read a float from environment variables without crashing on typos."""
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    """Read an integer from environment variables without crashing on typos."""
    raw = os.getenv(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


STRATEGY_CONFIG = CPRStrategyConfig(
    ema_fast_period=_env_int("CPR_EMA_FAST_PERIOD", 5),
    ema_slow_period=_env_int("CPR_EMA_SLOW_PERIOD", 20),
    rsi_period=_env_int("CPR_RSI_PERIOD", 14),
    rsi_ema_period=_env_int("CPR_RSI_EMA_PERIOD", 20),
    max_entry_wick_ratio=_env_float("CPR_MAX_ENTRY_WICK_RATIO", 0.25),
    trend_move_pct=_env_float("CPR_TREND_MOVE_PCT", 0.005),
    swing_lookback=_env_int("CPR_SWING_LOOKBACK", 3),
    zone_width_points=_env_float("CPR_ZONE_WIDTH_POINTS", 10.0),
    zone_lookback=_env_int("CPR_ZONE_LOOKBACK", 60),
    zone_min_touches=_env_int("CPR_ZONE_MIN_TOUCHES", 2),
    divergence_min_separation=_env_int("CPR_DIVERGENCE_MIN_SEPARATION", 12),
    divergence_requires_trend_move=bool(_env_int("CPR_DIVERGENCE_REQUIRES_TREND_MOVE", 1)),
)

# Supports the older typo alias just like other project backtests.
BROKERAGE = _env_float("BROKERAGE", _env_float("BROGERAGE", 80.0))


def normalize_dataset_key(value: str) -> str:
    """Map CLI dataset names to stable internal keys."""
    key = str(value or "nifty").strip().lower()
    aliases = {
        "nifty50": "nifty",
        "bank-nifty": "banknifty",
        "bank_nifty": "banknifty",
        "fin-nifty": "finnifty",
        "fin_nifty": "finnifty",
    }
    key = aliases.get(key, key)
    if key not in DEFAULT_DATA_PATHS:
        raise ValueError(f"Unsupported dataset `{value}`. Choose from: {', '.join(DEFAULT_DATA_PATHS)}")
    return key


def resolve_data_path(dataset_key: str, explicit_path: str = "") -> Path:
    """Use --data when supplied, otherwise use the dataset default."""
    if explicit_path:
        return Path(explicit_path)
    return DEFAULT_DATA_PATHS[dataset_key]


def build_output_paths(dataset_key: str) -> dict[str, Path]:
    """Keep all output filenames consistent and easy to find."""
    prefix = f"{dataset_key}_cpr_strategy_futures_5y"
    return {
        "log": OUTPUT_DIR / f"{prefix}_backtest.log",
        "trades": OUTPUT_DIR / f"{prefix}_trades.csv",
        "daily_equity": OUTPUT_DIR / f"{prefix}_daily_equity.csv",
        "stats": OUTPUT_DIR / f"{prefix}_stats.txt",
        "daily_loss": OUTPUT_DIR / f"{dataset_key}_cpr_strategy_futures_daily_max_loss.csv",
    }


def setup_logging(log_path: Path) -> None:
    """Log to both file and console."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_ohlc_data(csv_path: Path) -> pd.DataFrame:
    """
    Load raw CSV and return backtesting.py OHLCV format.

    Output:
    - DatetimeIndex
    - Open, High, Low, Close, Volume columns
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    candles = prepare_cpr_ohlc_input(raw)
    if candles.empty:
        raise ValueError(f"No usable OHLC candles found in {csv_path}")

    bt_df = candles.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    bt_df = bt_df.set_index("timestamp")
    bt_df.index = pd.DatetimeIndex(bt_df.index)
    if "Volume" not in bt_df.columns:
        bt_df["Volume"] = 0.0
    return bt_df[["Open", "High", "Low", "Close", "Volume"]]


class NiftyCPRStrategyBacktest(Strategy):
    """backtesting.py strategy class for the CPR futures strategy."""

    lot_size = POSITION_SIZE

    def init(self) -> None:
        """Prepare reusable state for the whole backtest run."""
        self.signal_engine = CPRSignalEngine(STRATEGY_CONFIG)
        self.last_processed_candle_ts = None
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0
        self.strategy_name = ""

        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        self.square_off_count = 0
        self.daily_loss_halt_count = 0
        self.margin_skip_count = 0

        self.current_day = None
        self.day_start_equity = None
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        self.day_trading_blocked = False
        self.square_off_requested_date = None
        self.daily_loss_tracker: dict[object, float] = {}
        self.margin_skip_logged_dates = set()

        # Precomputing is safe because every indicator in cpr_strategy_logic is
        # causal: it uses only current/past data or delayed swing confirmation.
        source = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(self.data.index),
                "open": np.asarray(self.data.Open, dtype=float),
                "high": np.asarray(self.data.High, dtype=float),
                "low": np.asarray(self.data.Low, dtype=float),
                "close": np.asarray(self.data.Close, dtype=float),
                "volume": np.asarray(self.data.Volume, dtype=float),
            }
        )
        self.cpr_candles = build_cpr_with_indicators(source, STRATEGY_CONFIG)

    def _reset_trade_state(self) -> None:
        """Clear custom state after the framework position closes."""
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0
        self.strategy_name = ""

    def _can_afford_entry(self, entry_price: float):
        """Check if current equity can support the new futures position."""
        equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        if notional <= 0.0:
            return False, MARGIN_REQUIREMENT, 0.0, equity

        if AUTO_ADJUST_MARGIN:
            affordable_margin = (equity * 0.98) / notional
            effective_margin = min(MARGIN_REQUIREMENT, max(MIN_MARGIN_FLOOR, affordable_margin))
        else:
            effective_margin = MARGIN_REQUIREMENT

        required_margin = notional * effective_margin
        return required_margin <= equity, effective_margin, required_margin, equity

    def _position_context(self) -> CPRPositionContext:
        """Build the small position object expected by the CPR signal engine."""
        return CPRPositionContext(
            direction=self.trade_direction,
            entry_underlying=self.entry_underlying,
            stop_underlying=self.stop_underlying,
            target_underlying=self.target_underlying,
            strategy_name=self.strategy_name,
        )

    def next(self) -> None:
        """Main bar-by-bar backtest loop."""
        if not self.position and self.trade_direction:
            self._reset_trade_state()

        bar_ts = pd.Timestamp(self.data.index[-1])
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        if self.current_day != bar_date:
            self.current_day = bar_date
            self.day_start_equity = float(self.equity)
            self.day_trading_blocked = False
            self.square_off_requested_date = None
            self.daily_loss_tracker[bar_date] = 0.0

        if self.day_start_equity is None:
            self.day_start_equity = float(self.equity)
        day_loss = max(0.0, float(self.day_start_equity) - float(self.equity))
        if day_loss > float(self.daily_loss_tracker.get(bar_date, 0.0)):
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
            if self.position:
                self.position.close()
                self.exit_count += 1
                self._reset_trade_state()
            return

        if bar_time >= SQUARE_OFF_TIME:
            if self.position and self.square_off_requested_date != bar_date:
                self.position.close()
                self.exit_count += 1
                self.square_off_count += 1
                self._reset_trade_state()
                self.square_off_requested_date = bar_date
            return

        if bar_time < ENTRY_START_TIME:
            return

        current_len = len(self.data.Close)
        candles = self.cpr_candles.iloc[:current_len]
        if candles.empty:
            return

        latest_ts = candles.iloc[-1]["timestamp"]
        if self.last_processed_candle_ts == latest_ts:
            return
        self.last_processed_candle_ts = latest_ts

        if self.position:
            decision = self.signal_engine.evaluate_candle(candles, position=self._position_context())
            if decision.action == "EXIT":
                self.position.close()
                self.exit_count += 1
                self._reset_trade_state()
            return

        if not (ENTRY_START_TIME <= bar_time < SQUARE_OFF_TIME):
            return

        decision = self.signal_engine.evaluate_candle(candles)
        if decision.signal_triggered:
            self.signal_count += 1
        if decision.action not in ("ENTER_LONG", "ENTER_SHORT"):
            return

        entry_price = float(decision.entry_underlying)
        can_enter, eff_margin, req_margin, equity_now = self._can_afford_entry(entry_price)
        if not can_enter:
            self.margin_skip_count += 1
            if bar_date not in self.margin_skip_logged_dates:
                logging.warning(
                    "Entry skipped due to margin | date=%s | time=%s | side=%s | "
                    "equity=%.2f | required=%.2f | effective_margin=%.4f",
                    bar_date,
                    bar_time,
                    decision.action,
                    equity_now,
                    req_margin,
                    eff_margin,
                )
                self.margin_skip_logged_dates.add(bar_date)
            return

        if decision.action == "ENTER_LONG":
            self.buy(size=self.lot_size)
            self.trade_direction = "LONG"
        else:
            self.sell(size=self.lot_size)
            self.trade_direction = "SHORT"

        self.entry_submit_count += 1
        self.entry_underlying = entry_price
        self.stop_underlying = float(decision.stop_underlying)
        self.target_underlying = float(decision.target_underlying)
        self.strategy_name = decision.strategy_name


def save_outputs(stats, output_paths: dict[str, Path], strategy_obj=None) -> None:
    """Write backtest artifacts to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        trades_to_save = trades.copy()
        if "PnL" in trades_to_save.columns:
            pnl_values = pd.to_numeric(trades_to_save["PnL"], errors="coerce")
            trades_to_save["Brokerage"] = float(BROKERAGE)
            trades_to_save["FinalPnL"] = pnl_values - float(BROKERAGE)
            trades_to_save["CumulativeFinalPnL"] = trades_to_save["FinalPnL"].cumsum()
        trades_to_save.to_csv(output_paths["trades"], index=False)

    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        eq = equity.copy()
        eq.index = pd.to_datetime(eq.index, errors="coerce")
        eq = eq[eq.index.notna()]
        eq[["Equity"]].resample("D").last().dropna().to_csv(
            output_paths["daily_equity"],
            index_label="date",
        )

    with open(output_paths["stats"], "w", encoding="ascii", errors="ignore") as file_obj:
        file_obj.write(str(stats))

    if strategy_obj is not None:
        tracker = getattr(strategy_obj, "daily_loss_tracker", None)
        if isinstance(tracker, dict) and tracker:
            daily_loss = pd.DataFrame(
                {
                    "date": pd.to_datetime(list(tracker.keys())),
                    "max_intraday_loss": [float(value) for value in tracker.values()],
                    "daily_loss_cap": STARTING_CAPITAL * DAILY_MAX_LOSS_PCT,
                }
            ).sort_values("date")
            daily_loss.to_csv(output_paths["daily_loss"], index=False, date_format="%Y-%m-%d")

    logging.info("Trades saved: %s", output_paths["trades"])
    logging.info("Daily equity saved: %s", output_paths["daily_equity"])
    logging.info("Stats saved: %s", output_paths["stats"])


def run_backtest(data_path: Path, dataset_key: str):
    """Load data, run the CPR strategy, and return stats plus strategy object."""
    data = load_ohlc_data(data_path)
    logging.info(
        "Loaded CPR data | rows=%s | start=%s | end=%s",
        len(data),
        data.index.min(),
        data.index.max(),
    )

    effective_margin = MIN_MARGIN_FLOOR if AUTO_ADJUST_MARGIN else MARGIN_REQUIREMENT
    bt = Backtest(
        data,
        NiftyCPRStrategyBacktest,
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
            "Strategy counters | signals=%s | entries=%s | exits=%s | square_offs=%s | "
            "daily_loss_halts=%s | margin_skips=%s",
            getattr(strategy_obj, "signal_count", "NA"),
            getattr(strategy_obj, "entry_submit_count", "NA"),
            getattr(strategy_obj, "exit_count", "NA"),
            getattr(strategy_obj, "square_off_count", "NA"),
            getattr(strategy_obj, "daily_loss_halt_count", "NA"),
            getattr(strategy_obj, "margin_skip_count", "NA"),
        )
    if "# Trades" in stats.index and int(stats["# Trades"]) == 0:
        logging.warning("Backtest finished with 0 closed trades. Check CPR rules/data regime.")
    return stats, strategy_obj


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="NIFTY CPR strategy futures backtest")
    parser.add_argument("--dataset", default="nifty", help="nifty, banknifty, or finnifty")
    parser.add_argument("--data", default="", help="Optional explicit OHLC CSV path")
    args = parser.parse_args()

    dataset_key = normalize_dataset_key(args.dataset)
    data_path = resolve_data_path(dataset_key, args.data)
    output_paths = build_output_paths(dataset_key)
    setup_logging(output_paths["log"])

    logging.info(
        "Config | dataset=%s | data=%s | capital=%s | qty=%s | margin=%s | "
        "entry_start=%s | square_off=%s | brokerage=%s",
        dataset_key,
        data_path,
        STARTING_CAPITAL,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        ENTRY_START_TIME,
        SQUARE_OFF_TIME,
        BROKERAGE,
    )

    stats, strategy_obj = run_backtest(data_path, dataset_key)
    save_outputs(stats, output_paths, strategy_obj=strategy_obj)

    for key in [
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
    ]:
        if key in stats.index:
            logging.info("%s: %s", key, stats[key])


if __name__ == "__main__":
    main()
