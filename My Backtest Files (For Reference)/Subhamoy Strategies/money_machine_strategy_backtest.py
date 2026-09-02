"""
Backtest the NIFTY Money Machine strategy using backtesting.py.

Beginner flow:
1. Load an already-prepared 5-minute OHLC CSV.
2. Validate that the file really is 5-minute data. No resampling happens here.
3. Precompute Money Machine indicators and setup candles.
4. Submit market entries after a completed Hulk/Marubozu setup candle.
5. Sync the actual next-open fill, then attach stop/target levels.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
SIGNAL_GENERATOR_DIR = ROOT_DIR / "Signal Generators" / "Subhamoy Strategies"
if str(SIGNAL_GENERATOR_DIR) not in sys.path:
    # The repo stores signal generators and backtests in different top-level
    # folders. This path bridge lets the backtest import the shared Money
    # Machine engine no matter where the command is launched from.
    sys.path.insert(0, str(SIGNAL_GENERATOR_DIR))

from money_machine_strategy_logic import (
    MoneyMachinePositionContext,
    MoneyMachineSignalEngine,
    MoneyMachineStrategyConfig,
    build_money_machine_with_indicators,
)
from subhamoy_backtest_common import (
    AUTO_ADJUST_MARGIN,
    DAILY_MAX_LOSS_PCT,
    ENTRY_START_TIME,
    MARGIN_REQUIREMENT,
    MIN_MARGIN_FLOOR,
    POSITION_SIZE,
    SQUARE_OFF_TIME,
    STARTING_CAPITAL,
    build_output_paths,
    env_float,
    env_int,
    load_ohlc_data,
    log_selected_stats,
    normalize_dataset_key,
    save_outputs,
    setup_logging,
)

STRATEGY_CONFIG = MoneyMachineStrategyConfig(
    sma_fast_period=env_int("MONEY_MACHINE_SMA_FAST_PERIOD", 20),
    sma_slow_period=env_int("MONEY_MACHINE_SMA_SLOW_PERIOD", 200),
    atr_period=env_int("MONEY_MACHINE_ATR_PERIOD", 14),
    slope_lookback=env_int("MONEY_MACHINE_SLOPE_LOOKBACK", 3),
    near_sma_atr_multiple=env_float("MONEY_MACHINE_NEAR_SMA_ATR_MULT", 0.5),
    compression_window=env_int("MONEY_MACHINE_COMPRESSION_WINDOW", 3),
    compression_range_atr_multiple=env_float("MONEY_MACHINE_COMPRESSION_RANGE_ATR_MULT", 0.75),
    marubozu_body_min_ratio=env_float("MONEY_MACHINE_MARUBOZU_BODY_MIN_RATIO", 0.80),
    marubozu_wick_max_ratio=env_float("MONEY_MACHINE_MARUBOZU_WICK_MAX_RATIO", 0.15),
    require_breakout_close=bool(env_int("MONEY_MACHINE_REQUIRE_BREAKOUT_CLOSE", 1)),
    target_atr_multiple=env_float("MONEY_MACHINE_TARGET_ATR_MULT", 2.0),
)


class NiftyMoneyMachineStrategyBacktest(Strategy):
    """
    backtesting.py strategy class for the Money Machine futures strategy.

    The shared Money Machine module decides whether a candle is a valid setup.
    This class handles the practical backtest mechanics around that decision:
    next-open fills, attached stop/target orders, daily risk limits, and output
    counters.
    """

    lot_size = POSITION_SIZE

    def init(self) -> None:
        """Prepare strategy state for the whole backtest run."""
        # The signal engine is intentionally kept separate from execution
        # simulation so strategy rules can be reused by front-tests later.
        self.signal_engine = MoneyMachineSignalEngine(STRATEGY_CONFIG)
        self.last_processed_candle_ts = None

        # Pending entry means: "a signal happened, a market order was submitted,
        # and we are waiting for backtesting.py to fill it on the next open."
        self.pending_entry: dict[str, object] | None = None
        # These fields are a plain-language mirror of the active framework trade.
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0

        self.signal_count = 0
        self.entry_submit_count = 0
        self.exit_count = 0
        self.square_off_count = 0
        self.daily_loss_halt_count = 0
        self.margin_skip_count = 0
        self.last_closed_trade_count = 0

        # Daily risk state. This mirrors the other repo backtests and makes the
        # saved daily max-loss output possible.
        self.current_day = None
        self.day_start_equity = None
        self.day_loss_limit = STARTING_CAPITAL * DAILY_MAX_LOSS_PCT
        self.day_trading_blocked = False
        self.square_off_requested_date = None
        self.daily_loss_tracker: dict[object, float] = {}
        self.margin_skip_logged_dates = set()

        # Precompute indicators once. In `next()` we slice this table to the
        # current visible candle count, so future rows are never evaluated.
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
        self.money_machine_candles = build_money_machine_with_indicators(source, STRATEGY_CONFIG)

    def _reset_trade_state(self) -> None:
        """Clear custom state after a trade closes."""
        # The framework position is already gone; reset our local mirror so the
        # next valid setup can be traded.
        self.pending_entry = None
        self.trade_direction = ""
        self.entry_underlying = 0.0
        self.stop_underlying = 0.0
        self.target_underlying = 0.0

    def _can_afford_entry(self, entry_price: float):
        """Check if current equity can support the new futures position."""
        equity = float(self.equity)
        notional = float(entry_price) * float(self.lot_size)
        if notional <= 0.0:
            return False, MARGIN_REQUIREMENT, 0.0, equity

        if AUTO_ADJUST_MARGIN:
            # Same margin behavior as the other strategy backtests: use the
            # configured margin when possible, otherwise reduce it within a
            # controlled floor so the simulation can continue after drawdown.
            affordable_margin = (equity * 0.98) / notional
            effective_margin = min(MARGIN_REQUIREMENT, max(MIN_MARGIN_FLOOR, affordable_margin))
        else:
            effective_margin = MARGIN_REQUIREMENT

        required_margin = notional * effective_margin
        return required_margin <= equity, effective_margin, required_margin, equity

    def _sync_closed_trade_state(self) -> None:
        """Detect stop/target/framework closures that happened since last bar."""
        # Attached stop/target orders can close a position without an explicit
        # call from this class. Counting framework closed trades keeps our local
        # mirror synchronized.
        closed_count = len(self.closed_trades)
        if closed_count <= self.last_closed_trade_count:
            return
        self.exit_count += closed_count - self.last_closed_trade_count
        self.last_closed_trade_count = closed_count
        self._reset_trade_state()

    def _sync_open_trade_state(self) -> None:
        """Turn a pending next-open order into active trade state after it fills."""
        if not self.position or self.trade_direction or self.pending_entry is None:
            return

        # Money Machine targets are based on actual entry plus/minus 2 x signal
        # ATR. Using actual_entry captures next-open gaps realistically.
        active_trade = self.trades[-1] if self.trades else None
        actual_entry = float(active_trade.entry_price) if active_trade is not None else float(self.data.Open[-1])
        signal_atr = float(self.pending_entry["signal_atr"])
        direction = str(self.pending_entry["direction"]).strip().upper()
        stop = float(self.pending_entry["stop_underlying"])
        if direction == "LONG":
            target = actual_entry + float(STRATEGY_CONFIG.target_atr_multiple) * signal_atr
        else:
            target = actual_entry - float(STRATEGY_CONFIG.target_atr_multiple) * signal_atr

        self.trade_direction = direction
        self.entry_underlying = actual_entry
        self.stop_underlying = stop
        self.target_underlying = target
        self.pending_entry = None

        if active_trade is not None:
            # Attach the protective stop and profit target only after entry
            # exists. backtesting.py then handles intrabar touches from here.
            if (direction == "LONG" and stop < actual_entry < target) or (
                direction == "SHORT" and target < actual_entry < stop
            ):
                active_trade.sl = stop
                active_trade.tp = target
            else:
                # If the next-open gap breaks the planned risk shape, exit
                # immediately instead of carrying a trade with inverted levels.
                self.position.close()

    def _position_context(self) -> MoneyMachinePositionContext:
        """Build the position object expected by the shared signal engine."""
        return MoneyMachinePositionContext(
            direction=self.trade_direction,
            entry_underlying=self.entry_underlying,
            stop_underlying=self.stop_underlying,
            target_underlying=self.target_underlying,
        )

    def _submit_next_open_entry(self, decision, bar_date, bar_time) -> None:
        """Submit a market entry that backtesting.py fills at the next bar open."""
        # The signal candle close is used only for a margin estimate. The true
        # fill is synchronized from backtesting.py on the next bar.
        can_enter, eff_margin, req_margin, equity_now = self._can_afford_entry(decision.entry_underlying)
        if not can_enter:
            self.margin_skip_count += 1
            if bar_date not in self.margin_skip_logged_dates:
                logging.warning(
                    "Money Machine entry skipped due to margin | date=%s | time=%s | side=%s | "
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

        direction = "LONG" if decision.action == "ENTER_LONG" else "SHORT"
        signal_atr = float(decision.debug.get("signal_atr", 0.0))
        if not np.isfinite(signal_atr) or signal_atr <= 0.0:
            # ATR drives target distance. No valid ATR means no reliable trade.
            return

        if direction == "LONG":
            self.buy(size=self.lot_size, tag="MONEY_MACHINE_LONG_ENTRY")
        else:
            self.sell(size=self.lot_size, tag="MONEY_MACHINE_SHORT_ENTRY")

        self.entry_submit_count += 1
        self.pending_entry = {
            "direction": direction,
            "stop_underlying": float(decision.stop_underlying),
            "signal_atr": signal_atr,
        }

    def next(self) -> None:
        """Main bar-by-bar backtest loop."""
        # Synchronize framework events first so entries/exits from the new bar
        # open are reflected before evaluating the latest completed candle.
        self._sync_closed_trade_state()
        self._sync_open_trade_state()

        bar_ts = pd.Timestamp(self.data.index[-1])
        bar_date = bar_ts.date()
        bar_time = bar_ts.time()

        if self.current_day != bar_date:
            # New trading day, new loss baseline, and trading can resume.
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
            # Once the daily cap is hit, stop trading for the rest of the day.
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
            self.pending_entry = None
            if self.position:
                self.position.close()
            return

        if bar_time >= SQUARE_OFF_TIME:
            self.pending_entry = None
            if self.position and self.square_off_requested_date != bar_date:
                self.position.close()
                self.square_off_count += 1
                self.square_off_requested_date = bar_date
            return

        if bar_time < ENTRY_START_TIME:
            return

        current_len = len(self.data.Close)
        # Use only history visible to the current bar. The indicators were
        # precomputed for speed, but this slice keeps evaluation causal.
        candles = self.money_machine_candles.iloc[:current_len]
        if candles.empty:
            return

        latest_ts = candles.iloc[-1]["timestamp"]
        if self.last_processed_candle_ts == latest_ts:
            return
        self.last_processed_candle_ts = latest_ts

        if self.position and self.trade_direction:
            # While in a trade, skip new entries and ask the shared engine only
            # whether the current candle hit stop/target conditions.
            decision = self.signal_engine.evaluate_candle(candles, position=self._position_context())
            if decision.action == "EXIT":
                self.position.close()
            return

        if self.pending_entry is not None:
            # A signal order is waiting for next-open fill; do not stack another
            # order before the first one resolves.
            return

        if not (ENTRY_START_TIME <= bar_time < SQUARE_OFF_TIME):
            return

        decision = self.signal_engine.evaluate_candle(candles)
        if decision.signal_triggered:
            self.signal_count += 1
        if decision.action in ("ENTER_LONG", "ENTER_SHORT"):
            self._submit_next_open_entry(decision, bar_date, bar_time)


def run_backtest(data_path: Path | str, dataset_key: str = "nifty"):
    """Load data, run the Money Machine strategy, and return stats plus strategy object."""
    data = load_ohlc_data(data_path)
    logging.info(
        "Loaded Money Machine data | rows=%s | start=%s | end=%s",
        len(data),
        data.index.min(),
        data.index.max(),
    )

    effective_margin = MIN_MARGIN_FLOOR if AUTO_ADJUST_MARGIN else MARGIN_REQUIREMENT
    # trade_on_close=False models the planned next-open entry behavior.
    bt = Backtest(
        data,
        NiftyMoneyMachineStrategyBacktest,
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
            "Money Machine counters | signals=%s | entries=%s | exits=%s | square_offs=%s | "
            "daily_loss_halts=%s | margin_skips=%s",
            getattr(strategy_obj, "signal_count", "NA"),
            getattr(strategy_obj, "entry_submit_count", "NA"),
            getattr(strategy_obj, "exit_count", "NA"),
            getattr(strategy_obj, "square_off_count", "NA"),
            getattr(strategy_obj, "daily_loss_halt_count", "NA"),
            getattr(strategy_obj, "margin_skip_count", "NA"),
        )
    return stats, strategy_obj


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="NIFTY Money Machine strategy futures backtest")
    parser.add_argument("--data", required=True, help="Path to an already-prepared 5-minute OHLC CSV")
    parser.add_argument("--dataset", default="nifty", help="Output label: nifty, banknifty, or finnifty")
    args = parser.parse_args()

    dataset_key = normalize_dataset_key(args.dataset)
    output_paths = build_output_paths(dataset_key, "money_machine_strategy")
    setup_logging(output_paths["log"])

    logging.info(
        "Config | strategy=Money Machine | dataset=%s | data=%s | capital=%s | qty=%s | "
        "margin=%s | entry_start=%s | square_off=%s",
        dataset_key,
        args.data,
        STARTING_CAPITAL,
        POSITION_SIZE,
        MARGIN_REQUIREMENT,
        ENTRY_START_TIME,
        SQUARE_OFF_TIME,
    )
    stats, strategy_obj = run_backtest(args.data, dataset_key)
    save_outputs(stats, output_paths, strategy_obj=strategy_obj)
    log_selected_stats(stats)


if __name__ == "__main__":
    main()
