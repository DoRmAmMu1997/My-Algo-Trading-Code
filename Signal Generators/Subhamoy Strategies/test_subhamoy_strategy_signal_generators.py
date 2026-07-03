import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STRATEGY_DIR.parents[1]
BACKTEST_DIR = REPO_ROOT / "My Backtest Files (For Reference)" / "Subhamoy Strategies"


def load_module(path: Path, module_name: str):
    """Load a module from a file path, including filenames with spaces."""
    assert path.exists(), f"Expected module at {path}"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_base_rows(rows=40, start_price=100.0, step=0.12):
    """Create simple 5-minute OHLC rows with a mild trend."""
    start = pd.Timestamp("2026-06-01 09:15")
    candles = []
    for index in range(rows):
        close = start_price + index * step
        candles.append(
            {
                "timestamp": start + pd.Timedelta(minutes=5 * index),
                "open": close - 0.05,
                "high": close + 0.35,
                "low": close - 0.35,
                "close": close,
                "volume": 1000 + index,
            }
        )
    return candles


def build_goldmine_long_frame():
    """Make the last candle a bullish engulfing setup near the rising SMA20."""
    rows = build_base_rows(rows=40, start_price=100.0, step=0.12)
    rows[-4].update({"open": 104.40, "high": 104.55, "low": 104.05, "close": 104.20})
    rows[-3].update({"open": 104.20, "high": 104.35, "low": 103.80, "close": 104.00})
    rows[-2].update({"open": 104.05, "high": 104.12, "low": 103.70, "close": 103.82})
    rows[-1].update({"open": 103.78, "high": 104.45, "low": 103.65, "close": 104.25})
    return pd.DataFrame(rows)


def build_goldmine_short_frame():
    """Make the last candle a bearish engulfing setup near the falling SMA20."""
    rows = build_base_rows(rows=40, start_price=110.0, step=-0.12)
    rows[-4].update({"open": 105.60, "high": 105.95, "low": 105.40, "close": 105.78})
    rows[-3].update({"open": 105.78, "high": 106.10, "low": 105.60, "close": 105.95})
    rows[-2].update({"open": 105.92, "high": 106.25, "low": 105.85, "close": 106.15})
    rows[-1].update({"open": 106.20, "high": 106.35, "low": 105.45, "close": 105.75})
    return pd.DataFrame(rows)


def build_money_machine_long_frame():
    """Make the last candle a bullish Hulk breakout after compression."""
    rows = build_base_rows(rows=40, start_price=100.0, step=0.12)
    rows[-4].update({"open": 104.00, "high": 104.25, "low": 103.95, "close": 104.10})
    rows[-3].update({"open": 104.08, "high": 104.28, "low": 103.98, "close": 104.18})
    rows[-2].update({"open": 104.16, "high": 104.30, "low": 104.00, "close": 104.08})
    rows[-1].update({"open": 104.02, "high": 105.20, "low": 104.00, "close": 105.12})
    return pd.DataFrame(rows)


def build_money_machine_short_frame():
    """Make the last candle a bearish Hulk breakdown after compression."""
    rows = build_base_rows(rows=40, start_price=110.0, step=-0.12)
    rows[-4].update({"open": 105.95, "high": 106.10, "low": 105.78, "close": 105.88})
    rows[-3].update({"open": 105.90, "high": 106.08, "low": 105.75, "close": 105.82})
    rows[-2].update({"open": 105.84, "high": 106.00, "low": 105.72, "close": 105.92})
    rows[-1].update({"open": 105.98, "high": 106.00, "low": 104.78, "close": 104.86})
    return pd.DataFrame(rows)


class TestSubhamoyStrategySignalGenerators(unittest.TestCase):
    def test_goldmine_generates_long_signal_from_bullish_engulfing_pullback(self):
        module = load_module(STRATEGY_DIR / "goldmine_strategy_logic.py", "goldmine_strategy_logic_long")
        config = module.GoldmineStrategyConfig(
            sma_fast_period=5,
            sma_slow_period=20,
            atr_period=5,
            slope_lookback=3,
            pullback_lookback=3,
            pullback_min_count=2,
            near_sma_atr_multiple=2.0,
        )

        enriched = module.build_goldmine_with_indicators(build_goldmine_long_frame(), config)
        decision = module.GoldmineSignalEngine(config).evaluate_candle(enriched)

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertTrue(decision.signal_triggered)
        self.assertEqual(decision.debug["entry_timing"], "NEXT_OPEN")
        self.assertLess(decision.stop_underlying, decision.entry_underlying)
        self.assertGreater(decision.target_underlying, decision.entry_underlying)

    def test_goldmine_generates_short_signal_from_bearish_engulfing_pullback(self):
        module = load_module(STRATEGY_DIR / "goldmine_strategy_logic.py", "goldmine_strategy_logic_short")
        config = module.GoldmineStrategyConfig(
            sma_fast_period=5,
            sma_slow_period=20,
            atr_period=5,
            slope_lookback=3,
            pullback_lookback=3,
            pullback_min_count=2,
            near_sma_atr_multiple=2.0,
        )

        enriched = module.build_goldmine_with_indicators(build_goldmine_short_frame(), config)
        decision = module.GoldmineSignalEngine(config).evaluate_candle(enriched)

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertTrue(decision.signal_triggered)
        self.assertEqual(decision.debug["entry_timing"], "NEXT_OPEN")
        self.assertGreater(decision.stop_underlying, decision.entry_underlying)
        self.assertLess(decision.target_underlying, decision.entry_underlying)

    def test_goldmine_stop_wins_when_stop_and_target_hit_same_candle(self):
        module = load_module(STRATEGY_DIR / "goldmine_strategy_logic.py", "goldmine_strategy_logic_exit")
        config = module.GoldmineStrategyConfig(max_bars_in_trade=6)
        candles = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-06-01 10:00"),
                    "open": 100.0,
                    "high": 103.0,
                    "low": 97.0,
                    "close": 101.0,
                    "atr": 1.0,
                }
            ]
        )
        position = module.GoldminePositionContext(
            direction="LONG",
            entry_underlying=100.0,
            stop_underlying=98.0,
            target_underlying=102.0,
            bars_in_trade=1,
        )

        decision = module.GoldmineSignalEngine(config).evaluate_candle(candles, position=position)

        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.exit_reason, "STOP")
        self.assertEqual(decision.exit_underlying, 98.0)

    def test_goldmine_exits_after_configured_time_limit(self):
        module = load_module(STRATEGY_DIR / "goldmine_strategy_logic.py", "goldmine_strategy_logic_time_exit")
        config = module.GoldmineStrategyConfig(max_bars_in_trade=6)
        candles = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-06-01 10:30"),
                    "open": 100.8,
                    "high": 101.0,
                    "low": 100.2,
                    "close": 100.6,
                    "atr": 1.0,
                }
            ]
        )
        position = module.GoldminePositionContext(
            direction="LONG",
            entry_underlying=100.0,
            stop_underlying=98.0,
            target_underlying=103.0,
            bars_in_trade=6,
        )

        decision = module.GoldmineSignalEngine(config).evaluate_candle(candles, position=position)

        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.exit_reason, "TIME_EXIT")
        self.assertEqual(decision.exit_underlying, 100.6)

    def test_money_machine_generates_long_signal_from_hulk_breakout(self):
        module = load_module(STRATEGY_DIR / "money_machine_strategy_logic.py", "money_machine_strategy_logic_long")
        config = module.MoneyMachineStrategyConfig(
            sma_fast_period=5,
            sma_slow_period=20,
            atr_period=5,
            slope_lookback=3,
            near_sma_atr_multiple=2.0,
            compression_range_atr_multiple=1.5,
        )

        enriched = module.build_money_machine_with_indicators(build_money_machine_long_frame(), config)
        decision = module.MoneyMachineSignalEngine(config).evaluate_candle(enriched)

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertTrue(decision.signal_triggered)
        self.assertEqual(decision.debug["entry_timing"], "NEXT_OPEN")
        self.assertLess(decision.stop_underlying, decision.entry_underlying)
        self.assertGreater(decision.target_underlying, decision.entry_underlying)

    def test_money_machine_generates_short_signal_from_hulk_breakdown(self):
        module = load_module(STRATEGY_DIR / "money_machine_strategy_logic.py", "money_machine_strategy_logic_short")
        config = module.MoneyMachineStrategyConfig(
            sma_fast_period=5,
            sma_slow_period=20,
            atr_period=5,
            slope_lookback=3,
            near_sma_atr_multiple=2.0,
            compression_range_atr_multiple=1.5,
        )

        enriched = module.build_money_machine_with_indicators(build_money_machine_short_frame(), config)
        decision = module.MoneyMachineSignalEngine(config).evaluate_candle(enriched)

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertTrue(decision.signal_triggered)
        self.assertEqual(decision.debug["entry_timing"], "NEXT_OPEN")
        self.assertGreater(decision.stop_underlying, decision.entry_underlying)
        self.assertLess(decision.target_underlying, decision.entry_underlying)

    def test_money_machine_holds_when_latest_candle_is_not_marubozu(self):
        module = load_module(STRATEGY_DIR / "money_machine_strategy_logic.py", "money_machine_strategy_logic_hold")
        data = build_money_machine_long_frame()
        data.loc[data.index[-1], ["open", "high", "low", "close"]] = [104.20, 105.20, 103.90, 104.35]
        config = module.MoneyMachineStrategyConfig(
            sma_fast_period=5,
            sma_slow_period=20,
            atr_period=5,
            slope_lookback=3,
            near_sma_atr_multiple=2.0,
        )

        enriched = module.build_money_machine_with_indicators(data, config)
        decision = module.MoneyMachineSignalEngine(config).evaluate_candle(enriched)

        self.assertEqual(decision.action, "HOLD")
        self.assertFalse(decision.signal_triggered)

    def test_missing_ohlc_columns_raise_clear_error(self):
        goldmine = load_module(STRATEGY_DIR / "goldmine_strategy_logic.py", "goldmine_strategy_logic_missing")

        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            goldmine.build_goldmine_with_indicators(pd.DataFrame({"timestamp": [pd.Timestamp("2026-06-01")]}))

    def test_wrapper_files_import_and_expose_latest_functions(self):
        goldmine_wrapper = load_module(
            STRATEGY_DIR / "Nifty Goldmine Signal Generator.py",
            "nifty_goldmine_signal_generator",
        )
        money_wrapper = load_module(
            STRATEGY_DIR / "Nifty Money Machine Signal Generator.py",
            "nifty_money_machine_signal_generator",
        )

        self.assertTrue(hasattr(goldmine_wrapper, "get_latest_nifty_goldmine_signal"))
        self.assertTrue(hasattr(money_wrapper, "get_latest_nifty_money_machine_signal"))


class TestSubhamoyStrategyBacktests(unittest.TestCase):
    def _write_csv(self, rows, directory: Path, name: str) -> Path:
        path = directory / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_goldmine_backtest_loader_accepts_five_minute_data_and_rejects_one_minute_data(self):
        module = load_module(
            BACKTEST_DIR / "Nifty Goldmine Strategy Backtest.py",
            "nifty_goldmine_strategy_backtest_loader",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            five_minute_csv = self._write_csv(build_base_rows(rows=30), tmp_path, "five_minute.csv")
            one_minute_rows = build_base_rows(rows=30)
            start = pd.Timestamp("2026-06-01 09:15")
            for index, row in enumerate(one_minute_rows):
                row["timestamp"] = start + pd.Timedelta(minutes=index)
            one_minute_csv = self._write_csv(one_minute_rows, tmp_path, "one_minute.csv")

            loaded = module.load_ohlc_data(five_minute_csv)
            self.assertEqual(list(loaded.columns), ["Open", "High", "Low", "Close", "Volume"])
            with self.assertRaisesRegex(ValueError, "5-minute"):
                module.load_ohlc_data(one_minute_csv)

    def test_backtest_modules_run_smoke_backtests_on_synthetic_five_minute_data(self):
        goldmine = load_module(
            BACKTEST_DIR / "Nifty Goldmine Strategy Backtest.py",
            "nifty_goldmine_strategy_backtest_smoke",
        )
        money = load_module(
            BACKTEST_DIR / "Nifty Money Machine Strategy Backtest.py",
            "nifty_money_machine_strategy_backtest_smoke",
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(build_base_rows(rows=260, start_price=100.0, step=0.02), Path(tmp), "smoke.csv")

            goldmine_stats, _ = goldmine.run_backtest(csv_path, "nifty")
            money_stats, _ = money.run_backtest(csv_path, "nifty")

        self.assertIn("# Trades", goldmine_stats.index)
        self.assertIn("# Trades", money_stats.index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
