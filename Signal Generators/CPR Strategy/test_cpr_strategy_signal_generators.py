import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

STRATEGY_DIR = Path(__file__).resolve().parent
LOGIC_PATH = STRATEGY_DIR / "cpr_strategy_logic.py"
BACKTEST_PATH = STRATEGY_DIR / "Nifty CPR Strategy Backtest.py"
ALGO3_PATH = STRATEGY_DIR / "Nifty CPR Algo 3 Signal Generator.py"


def load_module(path: Path, name: str):
    """Load a module from a file path, including files whose names contain spaces."""
    assert path.exists(), f"Expected module at {path}"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_day(date_text, start_price, rows=75, step=0.0):
    """Create one simple 5-minute trading session."""
    start = pd.Timestamp(f"{date_text} 09:15")
    candles = []
    for index in range(rows):
        base = start_price + index * step
        candles.append(
            {
                "timestamp": start + pd.Timedelta(minutes=5 * index),
                "open": base,
                "high": base + 0.4,
                "low": base - 0.4,
                "close": base,
                "volume": 1000,
            }
        )
    return candles


def make_strategy_frame(current_day_rows):
    """Build two previous days plus caller-supplied current-day candles."""
    rows = []
    rows.extend(build_day("2026-05-01", 100.0, rows=75, step=0.0))
    rows.extend(build_day("2026-05-04", 100.5, rows=75, step=0.0))
    rows.extend(current_day_rows)
    return pd.DataFrame(rows)


class TestCPRStrategySignalGenerators(unittest.TestCase):
    def test_calculates_previous_day_daily_cpr_levels(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_math")
        rows = []
        rows.extend(build_day("2026-05-01", 90.0, rows=2))
        rows[-2].update({"high": 101.0, "low": 99.0, "close": 100.0})
        rows[-1].update({"high": 103.0, "low": 98.0, "close": 102.0})
        rows.extend(build_day("2026-05-04", 105.0, rows=2))

        enriched = module.build_cpr_with_indicators(pd.DataFrame(rows))
        first_current = enriched[enriched["session_date"] == pd.Timestamp("2026-05-04").date()].iloc[0]

        expected_pivot = (103.0 + 98.0 + 102.0) / 3.0
        expected_bc = (103.0 + 98.0) / 2.0
        self.assertAlmostEqual(first_current["pivot"], expected_pivot)
        self.assertAlmostEqual(first_current["bc"], expected_bc)
        self.assertAlmostEqual(first_current["tc"], 2.0 * expected_pivot - expected_bc)
        self.assertAlmostEqual(first_current["r1"], 2.0 * expected_pivot - 98.0)
        self.assertAlmostEqual(first_current["s1"], 2.0 * expected_pivot - 103.0)

    def test_classifies_daily_cpr_width_as_narrow_medium_and_wide(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_width")

        narrow = module.classify_daily_cpr_width(101.0, 100.0, 100.5)
        medium = module.classify_daily_cpr_width(102.5, 100.0, 101.0)
        wide = module.classify_daily_cpr_width(130.0, 100.0, 110.0)

        self.assertEqual(narrow, "narrow")
        self.assertEqual(medium, "medium")
        self.assertEqual(wide, "wide")

    def test_resamples_complete_one_minute_bars_to_five_minute_candles(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_resample")
        start = pd.Timestamp("2026-05-05 09:15")
        rows = []
        for index in range(11):
            price = 100.0 + index
            rows.append(
                {
                    "timestamp": start + pd.Timedelta(minutes=index),
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.25,
                    "volume": 10,
                }
            )

        result = module.prepare_cpr_ohlc_input(pd.DataFrame(rows))

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["timestamp"]), [start, start + pd.Timedelta(minutes=5)])
        self.assertEqual(result.iloc[0]["open"], 100.0)
        self.assertEqual(result.iloc[0]["close"], 104.25)
        self.assertEqual(result.iloc[1]["open"], 105.0)
        self.assertEqual(result.iloc[1]["close"], 109.25)

    def test_generates_bullish_algo1_condition1_signal(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_bull_c1")
        current = build_day("2026-05-05", 100.6, rows=40, step=0.02)
        current[-2].update({"open": 101.4, "high": 101.8, "low": 100.6, "close": 101.0})
        current[-1].update({"open": 100.7, "high": 102.1, "low": 100.2, "close": 101.95})
        config = module.CPRStrategyConfig(rsi_period=3, rsi_ema_period=3, ema_fast_period=3, ema_slow_period=5)
        engine = module.CPRSignalEngine(config=config)

        decision = engine.evaluate_candle(module.build_cpr_with_indicators(make_strategy_frame(current), config))

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.strategy_name, "ALGO1_CONDITION1")
        self.assertGreater(decision.target_underlying, decision.entry_underlying)
        self.assertLess(decision.stop_underlying, decision.entry_underlying)

    def test_generates_bearish_algo1_condition1_signal(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_bear_c1")
        current = build_day("2026-05-05", 100.4, rows=40, step=-0.02)
        current[-1].update({"open": 100.2, "high": 100.5, "low": 98.8, "close": 99.05})
        config = module.CPRStrategyConfig(rsi_period=3, rsi_ema_period=3, ema_fast_period=3, ema_slow_period=5)
        engine = module.CPRSignalEngine(config=config)

        decision = engine.evaluate_candle(module.build_cpr_with_indicators(make_strategy_frame(current), config))

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.strategy_name, "ALGO1_CONDITION1")
        self.assertLess(decision.target_underlying, decision.entry_underlying)
        self.assertGreater(decision.stop_underlying, decision.entry_underlying)

    def test_generates_algo1_condition2_retrace_signal(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_c2")
        current = build_day("2026-05-05", 100.8, rows=45, step=0.08)
        current[-3].update({"open": 104.0, "high": 104.4, "low": 103.8, "close": 104.2})
        current[-2].update({"open": 104.3, "high": 104.6, "low": 103.5, "close": 103.6})
        current[-1].update({"open": 104.0, "high": 104.58, "low": 103.4, "close": 104.25})
        config = module.CPRStrategyConfig(rsi_period=3, rsi_ema_period=3, ema_fast_period=3, ema_slow_period=5)
        engine = module.CPRSignalEngine(config=config)

        decision = engine.evaluate_candle(module.build_cpr_with_indicators(make_strategy_frame(current), config))

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.strategy_name, "ALGO1_CONDITION2")

    def test_generates_sideways_zone_signal(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_zone")
        rows = []
        rows.extend(build_day("2026-05-01", 100.0, rows=75, step=0.0))
        wide_prev = build_day("2026-05-04", 110.0, rows=75, step=0.0)
        wide_prev[0].update({"high": 130.0, "low": 100.0, "close": 110.0})
        rows.extend(wide_prev)
        current = build_day("2026-05-05", 112.0, rows=40, step=0.0)
        for idx in (5, 20, 39):
            current[idx].update({"high": 120.0, "low": 111.0, "open": 118.0, "close": 116.0})
        for idx in (8, 25):
            current[idx].update({"high": 116.0, "low": 105.0, "open": 107.0, "close": 108.0})
        rows.extend(current)
        config = module.CPRStrategyConfig(rsi_period=3, rsi_ema_period=3, zone_lookback=40)
        engine = module.CPRSignalEngine(config=config)

        decision = engine.evaluate_candle(module.build_cpr_with_indicators(pd.DataFrame(rows), config))

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.strategy_name, "ALGO2_SIDEWAYS_ZONE")

    def test_generates_rsi_divergence_signal(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_divergence")
        current = build_day("2026-05-05", 100.8, rows=55, step=0.03)
        current[15].update({"high": 104.0, "low": 101.0, "open": 102.0, "close": 103.8})
        current[16].update({"high": 102.0, "low": 100.2, "open": 101.8, "close": 100.5})
        current[35].update({"high": 105.2, "low": 101.0, "open": 104.8, "close": 101.2})
        current[36].update({"high": 102.0, "low": 100.1, "open": 101.3, "close": 100.4})
        config = module.CPRStrategyConfig(
            rsi_period=3,
            rsi_ema_period=3,
            swing_lookback=2,
            divergence_min_separation=12,
            divergence_requires_trend_move=False,
        )
        engine = module.CPRSignalEngine(config=config)

        enriched = module.build_cpr_with_indicators(make_strategy_frame(current), config)
        divergence_rows = enriched[enriched["bearish_rsi_divergence"]]
        self.assertFalse(divergence_rows.empty)
        decision = engine.evaluate_candle(enriched.iloc[: divergence_rows.index[-1] + 1])

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.strategy_name, "RSI_DIVERGENCE")

    def test_conflicting_opposite_signals_return_hold(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_conflict")
        current = build_day("2026-05-05", 100.6, rows=40, step=0.02)
        current[-2].update({"open": 101.4, "high": 101.8, "low": 100.6, "close": 101.0})
        current[-1].update({"open": 100.7, "high": 102.1, "low": 100.2, "close": 101.95})
        config = module.CPRStrategyConfig(rsi_period=3, rsi_ema_period=3, ema_fast_period=3, ema_slow_period=5)
        enriched = module.build_cpr_with_indicators(make_strategy_frame(current), config)
        enriched.loc[enriched.index[-1], "algo2_sideways_short_setup"] = True
        enriched.loc[enriched.index[-1], "sideways_short_target"] = 99.0
        enriched.loc[enriched.index[-1], "sideways_short_stop"] = 103.0
        engine = module.CPRSignalEngine(config=config)

        decision = engine.evaluate_candle(enriched)

        self.assertEqual(decision.action, "HOLD")
        self.assertIn("conflict", decision.exit_reason.lower())

    def test_rejects_missing_required_columns(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_missing")

        with self.assertRaises(ValueError):
            module.build_cpr_with_indicators(pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-05")], "open": [1.0]}))

    def test_insufficient_history_returns_hold(self):
        module = load_module(LOGIC_PATH, "cpr_strategy_logic_hold")
        config = module.CPRStrategyConfig()
        engine = module.CPRSignalEngine(config=config)
        enriched = module.build_cpr_with_indicators(pd.DataFrame(build_day("2026-05-05", 100.0, rows=5)), config)

        decision = engine.evaluate_candle(enriched)

        self.assertEqual(decision.action, "HOLD")
        self.assertFalse(decision.signal_triggered)


def algo3_instrument(prev_base, current_start, current_step, rows=40, prev2_updates=None):
    """
    One OPTION leg's three-session frame for Algo 3 tests.

    Two flat prior days set the CPR band (the 2nd can be widened via `prev2_updates`),
    then a monotonic current day (`current_step` > 0 rises, < 0 falls). All instruments
    built with the same `rows` share the same 5-minute timestamps, so Algo 3 can align
    them. (Option legs only need to sit above/below their own CPR/VWAP, so a clean trend
    is fine - their RSI is irrelevant; the spot RSI is the one Algo 3 reads.)
    """
    out = []
    out.extend(build_day("2026-05-01", prev_base, rows=rows, step=0.0))
    prev2 = build_day("2026-05-04", prev_base, rows=rows, step=0.0)
    for idx, update in (prev2_updates or {}).items():
        prev2[idx].update(update)
    out.extend(prev2)
    out.extend(build_day("2026-05-05", current_start, rows=rows, step=current_step))
    return pd.DataFrame(out)


def trending_spot(direction, rows=40, prev2_updates=None):
    """
    SPOT frame whose current day DRIFTS `direction` ('up'/'down') while oscillating.

    A clean monotonic ramp saturates RSI to 100/0 and ties it with its own EMA (ARSI),
    which would make the strict `RSI vs ARSI` test ambiguous. A gentle zig-zag keeps RSI
    mid-range and clearly on the right side of ARSI. Two flat prior days (the 2nd widened
    so r1/prev_high/r2 sit above the band and s1/prev_low/s2 below it, leaving room for a
    target) set the CPR band and pivot levels.
    """
    parts = []
    parts.extend(build_day("2026-05-01", 100.0, rows=rows, step=0.0))
    prev2 = build_day("2026-05-04", 100.0, rows=rows, step=0.0)
    for idx, update in (prev2_updates or {}).items():
        prev2[idx].update(update)
    parts.extend(prev2)
    start = pd.Timestamp("2026-05-05 09:15")
    sign = 1.0 if direction == "up" else -1.0
    price = 100.0 + sign * 1.0
    closes = [price]
    for index in range(rows - 1):
        price += (sign * 0.6 if index % 2 == 0 else -sign * 0.3)  # two steps forward, one back
        closes.append(price)
    for index, close in enumerate(closes):
        parts.append({"timestamp": start + pd.Timedelta(minutes=5 * index),
                      "open": close, "high": close + 0.4, "low": close - 0.4, "close": close, "volume": 1000})
    return pd.DataFrame(parts)


class TestCPRAlgo3SignalGenerator(unittest.TestCase):
    """Multi-instrument CPR Algo 3 (spot + ITM CE + ITM PE)."""

    # A widened previous day for spot, so r1/prev_high/r2 sit above the CPR band and
    # there is room for an upward target (and s1/prev_low/s2 below for a downward one).
    WIDE_SPOT_PREV = {0: {"high": 110.0}, 1: {"low": 90.0}}

    def _config(self, module):
        # Short indicator periods so a synthetic session moves RSI/ARSI decisively.
        return module.CPRAlgo3Config(
            indicator_config=module.CPRStrategyConfig(
                rsi_period=3, rsi_ema_period=3, ema_fast_period=3, ema_slow_period=5
            )
        )

    def test_call_alignment_enters_long_to_buy_ce(self):
        module = load_module(ALGO3_PATH, "cpr_algo3_call")
        spot = trending_spot("up", prev2_updates=self.WIDE_SPOT_PREV)  # spot up over CPR, RSI > ARSI
        ce = algo3_instrument(200.0, 201.0, 0.3)   # call rises above its CPR/VWAP
        pe = algo3_instrument(200.0, 199.0, -0.3)  # put falls below its CPR/VWAP

        decision = module.get_latest_nifty_cpr_algo3_signal(spot, ce, pe, config=self._config(module))

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.strategy_name, "CPR_ALGO3")
        self.assertGreater(decision.target_underlying, decision.entry_underlying)

    def test_put_alignment_enters_short_to_buy_pe(self):
        module = load_module(ALGO3_PATH, "cpr_algo3_put")
        spot = trending_spot("down", prev2_updates=self.WIDE_SPOT_PREV)  # spot down under CPR, RSI < ARSI
        ce = algo3_instrument(200.0, 199.0, -0.3)  # call falls below its CPR/VWAP
        pe = algo3_instrument(200.0, 201.0, 0.3)   # put rises above its CPR/VWAP

        decision = module.get_latest_nifty_cpr_algo3_signal(spot, ce, pe, config=self._config(module))

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.strategy_name, "CPR_ALGO3")
        self.assertLess(decision.target_underlying, decision.entry_underlying)

    def test_misaligned_options_hold(self):
        module = load_module(ALGO3_PATH, "cpr_algo3_hold")
        spot = trending_spot("up", prev2_updates=self.WIDE_SPOT_PREV)  # spot would-be call
        ce = algo3_instrument(200.0, 200.0, 0.0)   # but the options are flat - not above/below their CPR
        pe = algo3_instrument(200.0, 200.0, 0.0)

        decision = module.get_latest_nifty_cpr_algo3_signal(spot, ce, pe, config=self._config(module))

        self.assertEqual(decision.action, "HOLD")
        self.assertFalse(decision.signal_triggered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
