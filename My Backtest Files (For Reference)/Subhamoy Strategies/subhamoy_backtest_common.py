"""
Shared backtest utilities for Subhamoy strategy backtests.

These helpers keep Goldmine and Money Machine backtests consistent:
- same 5-minute CSV validation
- same output file naming
- same brokerage-adjusted artifacts
- same risk/margin defaults
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SIGNAL_GENERATOR_DIR = ROOT_DIR / "Signal Generators" / "Subhamoy Strategies"
if str(SIGNAL_GENERATOR_DIR) not in sys.path:
    # Repo-local backtests live in `My Backtest Files (For Reference)`, while
    # their reusable signal modules live in `Signal Generators`. Adding the
    # signal folder here keeps CLI runs and tests from needing manual PYTHONPATH.
    sys.path.insert(0, str(SIGNAL_GENERATOR_DIR))

from subhamoy_strategy_common import normalize_ohlc_frame, validate_five_minute_spacing

OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"

# These constants intentionally mirror the existing project backtests so that a
# Goldmine/Money Machine run feels familiar when compared with CPR, EMA, Renko,
# or Profit Shooter results.
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


def env_float(name: str, default: float) -> float:
    """
    Read a float from environment variables without crashing on typos.

    This lets the user tune a backtest through environment variables while still
    getting a sensible default if an env var is missing or malformed.
    """
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    """
    Read an integer from environment variables without crashing on typos.

    `int(float(raw))` accepts common values like `3` and `3.0`, which is useful
    when config is copied from spreadsheets or `.env` files.
    """
    raw = os.getenv(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


BROKERAGE = env_float("BROKERAGE", env_float("BROGERAGE", 80.0))


def normalize_dataset_key(value: str) -> str:
    """Map friendly CLI dataset names to stable internal output-name keys."""
    # Dataset does not choose a default input file for these new backtests. It
    # is only used to label output artifacts, because the user explicitly passes
    # the already-resampled 5-minute CSV with `--data`.
    key = str(value or "nifty").strip().lower()
    aliases = {
        "nifty50": "nifty",
        "bank-nifty": "banknifty",
        "bank_nifty": "banknifty",
        "fin-nifty": "finnifty",
        "fin_nifty": "finnifty",
    }
    key = aliases.get(key, key)
    allowed = {"nifty", "banknifty", "finnifty"}
    if key not in allowed:
        raise ValueError(f"Unsupported dataset `{value}`. Choose from: {', '.join(sorted(allowed))}")
    return key


def build_output_paths(dataset_key: str, strategy_slug: str) -> dict[str, Path]:
    """Build consistent output paths for one strategy/dataset pair."""
    dataset = normalize_dataset_key(dataset_key)
    # Keep filenames descriptive enough that multiple strategy runs can coexist
    # in `Backtest Outputs` without overwriting each other.
    prefix = f"{dataset}_{strategy_slug}_futures_5y"
    return {
        "log": OUTPUT_DIR / f"{prefix}_backtest.log",
        "trades": OUTPUT_DIR / f"{prefix}_trades.csv",
        "daily_equity": OUTPUT_DIR / f"{prefix}_daily_equity.csv",
        "stats": OUTPUT_DIR / f"{prefix}_stats.txt",
        "daily_loss": OUTPUT_DIR / f"{dataset}_{strategy_slug}_futures_daily_max_loss.csv",
    }


def setup_logging(log_path: Path) -> None:
    """Log to both the console and a strategy-specific file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove existing root handlers so importing/running multiple backtests in
    # one Python process does not duplicate every log line.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def load_ohlc_data(csv_path: Path | str) -> pd.DataFrame:
    """
    Load a 5-minute OHLC CSV and return backtesting.py OHLCV format.

    This function validates timeframe only. It never resamples.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw = pd.read_csv(path)
    candles = normalize_ohlc_frame(raw)
    if candles.empty:
        raise ValueError(f"No usable OHLC candles found in {path}")
    # The user asked for no resampling in these modules. This check rejects
    # 1-minute files with a direct message instead of quietly changing data.
    validate_five_minute_spacing(candles)

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
        # backtesting.py expects a Volume column, even though these strategies
        # do not use volume in their signal rules.
        bt_df["Volume"] = 0.0
    return bt_df[["Open", "High", "Low", "Close", "Volume"]]


def save_outputs(stats: Any, output_paths: dict[str, Path], strategy_obj=None) -> None:
    """Write trades, equity, stats, and daily max-loss artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = stats.get("_trades", pd.DataFrame())
    if isinstance(trades, pd.DataFrame):
        trades_to_save = trades.copy()
        if "PnL" in trades_to_save.columns:
            # Keep raw framework PnL and add final PnL after per-trade brokerage
            # so the output remains easy to audit.
            pnl_values = pd.to_numeric(trades_to_save["PnL"], errors="coerce")
            trades_to_save["Brokerage"] = float(BROKERAGE)
            trades_to_save["FinalPnL"] = pnl_values - float(BROKERAGE)
            trades_to_save["CumulativeFinalPnL"] = trades_to_save["FinalPnL"].cumsum()
        trades_to_save.to_csv(output_paths["trades"], index=False)

    equity = stats.get("_equity_curve", pd.DataFrame())
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        # Daily equity is lighter to inspect than the full per-bar equity curve
        # and matches the output style used by existing strategy backtests.
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
            # The strategy object records the maximum intraday drawdown seen for
            # each date. Saving it makes daily loss cap behavior visible later.
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


def log_selected_stats(stats: Any) -> None:
    """Print the most useful backtest stats into the configured logger."""
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
