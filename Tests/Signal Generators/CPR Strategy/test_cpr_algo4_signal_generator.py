"""Unit tests for the deterministic CPR Algo 4 ("Intraday SRSI VWAP") module.

Most tests drive `CPRAlgo4Engine` with hand-built 5-minute rows instead of
running the full indicator pipeline. That keeps every scenario readable: each
row states exactly the close, VWAP, RSI, EMA and Stochastic RSI values the rule
under test looks at, so a failure points at one rule rather than at indicator
warm-up. The frame builder and the Stochastic RSI maths get their own tests.
"""

import importlib.util
import math
import sys
import unittest
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

# Tests/Signal Generators/CPR Strategy/<this file> -> repository root is three up.
STRATEGY_DIR = Path(__file__).resolve().parents[3] / "Signal Generators" / "CPR Strategy"
ALGO4_PATH = STRATEGY_DIR / "cpr_algo4_signal_generator.py"


def load_module(path: Path, name: str):
    """Load a module from a file path (the strategy folder has a space in it)."""
    assert path.exists(), f"Expected module at {path}"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


algo4 = load_module(ALGO4_PATH, "cpr_algo4_signal_generator_under_test")

DAY = "2026-05-06"

# Previous-session levels for SIDEWAYS scenarios. Zone = [min(S1, PDL),
# max(R1, PDH)] = [21880, 22120].
SIDEWAYS_LEVELS = {
    "s2": 21800.0,
    "s1": 21900.0,
    "prev_low": 21880.0,
    "cpr_lower": 21995.0,
    "pivot": 22000.0,
    "cpr_upper": 22005.0,
    "prev_high": 22120.0,
    "r1": 22100.0,
    "r2": 22200.0,
}

# Previous-session levels for TRENDING scenarios. Zone = [21780, 22020].
TREND_LEVELS = {
    "s2": 21700.0,
    "s1": 21800.0,
    "prev_low": 21780.0,
    "cpr_lower": 21895.0,
    "pivot": 21900.0,
    "cpr_upper": 21905.0,
    "prev_high": 22020.0,
    "r1": 22000.0,
    "r2": 22100.0,
}


def bar(clock, o, h, lo, c, levels=None, **extra):
    """One completed 5-minute row. Indicators default to neutral values."""
    row = {
        "timestamp": pd.Timestamp(f"{DAY} {clock}"),
        "open": float(o),
        "high": float(h),
        "low": float(lo),
        "close": float(c),
        "vwap": float(c),
        "rsi": 50.0,
        "ema5": float(c),
        "ema20": float(c),
        "srsi_k": 50.0,
        "srsi_d": 50.0,
    }
    row.update(levels if levels is not None else SIDEWAYS_LEVELS)
    row.update(extra)
    return row


def feed(engine, rows, plan=None):
    """Feed rows in order and return the decision for the last one."""
    decision = None
    for row in rows:
        decision = engine.on_bar(row, plan)
    return decision


def make_plan(**overrides):
    """A trending long plan used by management tests."""
    values = {
        "direction": "LONG",
        "premise": algo4.PREMISE_CONTINUATION,
        "entry": 22040.0,
        "risk": 10.0,
        "original_stop": 22030.0,
        "current_stop": 22030.0,
        "first_milestone": 22050.0,
        "following_milestone": 22060.0,
        "target": float("nan"),
        "final_target": 22098.0,
        "exit_mode": "TRAIL",
    }
    values.update(overrides)
    return algo4.CPRAlgo4TradePlan(**values)


# --- Scenario builders ----------------------------------------------------------


def sideways_long_rows(swing_low=21915.0):
    """SIDEWAYS day: a confirmed swing low, then an oversold SRSI cross up at 09:45."""
    return [
        bar("09:15", 21940, 21941, 21935, 21940),
        bar("09:20", 21938, 21942, 21930, 21936),
        bar("09:25", 21936, 21940, swing_low, 21932),  # regime bar, inside the zone
        bar("09:30", 21932, 21938, 21925, 21930),
        bar("09:35", 21930, 21936, 21928, 21931, srsi_k=10.0, srsi_d=15.0),  # confirms the swing
        bar("09:40", 21931, 21937, 21929, 21932, srsi_k=12.0, srsi_d=16.0),
        bar("09:45", 21932, 21936, 21927, 21930, srsi_k=18.0, srsi_d=14.0),  # cross up in oversold
    ]


def sideways_short_rows():
    """SIDEWAYS day: a confirmed swing high, then an overbought SRSI cross down."""
    return [
        bar("09:15", 22060, 22065, 22055, 22060),
        bar("09:20", 22062, 22070, 22058, 22064),
        bar("09:25", 22064, 22085, 22060, 22068),
        bar("09:30", 22068, 22075, 22062, 22070),
        bar("09:35", 22069, 22072, 22064, 22069, srsi_k=90.0, srsi_d=85.0),
        bar("09:40", 22069, 22073, 22065, 22070, srsi_k=88.0, srsi_d=84.0),
        bar("09:45", 22070, 22074, 22066, 22070, srsi_k=82.0, srsi_d=86.0),  # cross down in overbought
    ]


def trend_regime_rows(close_0925=22032.0):
    """Three opening bars whose 09:25 close sits above the trend zone (UP)."""
    return [
        bar("09:15", 22025, 22030, 22020, 22028, TREND_LEVELS),
        bar("09:20", 22028, 22034, 22024, 22030, TREND_LEVELS),
        bar("09:25", 22030, 22036, 22026, close_0925, TREND_LEVELS),
    ]


def continuation_long_rows(entry_vwap=22032.0, entry_open=22030.0, **entry_overrides):
    """UP day: a close below VWAP, then a close back above it with the filters met."""
    entry = {"vwap": entry_vwap, "rsi": 58.0, "ema5": 22033.0, "ema20": 22029.0}
    entry.update(entry_overrides)
    return [
        *trend_regime_rows(),
        bar("09:30", 22032, 22034, 22028, 22030, TREND_LEVELS, vwap=22031.0, ema5=22031.0, ema20=22028.0),
        bar("09:35", entry_open, 22042, 22029, 22040, TREND_LEVELS, **entry),
    ]


def zigzag_rows():
    """UP day whose confirmed swings print LH then LL, flipping the trend DOWN at 10:05."""
    shape = [
        ("09:15", 22150, 22140),
        ("09:20", 22160, 22150),
        ("09:25", 22170, 22160),  # regime bar: close 22165 > 22020 -> TRENDING_UP; swing high H1
        ("09:30", 22160, 22150),
        ("09:35", 22150, 22135),  # swing low L1
        ("09:40", 22158, 22145),
        ("09:45", 22165, 22150),  # swing high H2 (lower high)
        ("09:50", 22150, 22140),
        ("09:55", 22140, 22125),  # swing low L2 (lower low)
        ("10:00", 22145, 22130),
        ("10:05", 22148, 22135),  # confirms L2 -> LH + LL -> flip DOWN
    ]
    rows = []
    for clock, high, low in shape:
        mid = (high + low) / 2.0
        rows.append(bar(clock, mid, high, low, mid, TREND_LEVELS))
    return rows


def reversal_short_rows():
    """After the DOWN flip: close below VWAP, green close above, red close below."""
    return [
        bar("10:10", 22135, 22138, 22128, 22130, TREND_LEVELS, vwap=22133.0),  # step A
        bar("10:15", 22130, 22140, 22129, 22137, TREND_LEVELS, vwap=22134.0, ema5=22131.0, ema20=22135.0),  # step B
        bar(
            "10:20", 22137, 22139, 22120, 22124, TREND_LEVELS,
            vwap=22133.0, rsi=40.0, ema5=22128.0, ema20=22134.0,
        ),  # step C -> ENTER_SHORT
    ]


# --- Stochastic RSI and the frame builder -------------------------------------


class TestStochasticRsi(unittest.TestCase):
    def test_matches_hand_computed_values(self):
        rsi = pd.Series([10.0, 20.0, 30.0, 20.0, 10.0, 40.0, 50.0])
        k, d = algo4.stochastic_rsi(rsi, stoch_period=3, k_period=2, d_period=2)
        # raw = [nan, nan, 100, 0, 0, 100, 100]; K = SMA2(raw); D = SMA2(K).
        expected_k = [np.nan, np.nan, np.nan, 50.0, 0.0, 50.0, 100.0]
        expected_d = [np.nan, np.nan, np.nan, np.nan, 25.0, 25.0, 75.0]
        np.testing.assert_allclose(k.to_numpy(), expected_k, equal_nan=True)
        np.testing.assert_allclose(d.to_numpy(), expected_d, equal_nan=True)

    def test_flat_rsi_window_is_undefined_not_zero(self):
        k, _ = algo4.stochastic_rsi(pd.Series([50.0] * 6), stoch_period=3, k_period=1, d_period=1)
        self.assertTrue(k.isna().all())


def one_minute_days(days=4, seed=7):
    """Deterministic 1-minute NIFTY-like sessions (09:15-15:29) for builder tests."""
    rng = np.random.default_rng(seed)
    frames = []
    price = 22000.0
    for offset in range(days):
        start = pd.Timestamp("2026-05-04 09:15") + pd.Timedelta(days=offset)
        stamps = pd.date_range(start, periods=375, freq="1min")
        steps = rng.normal(0.0, 3.0, len(stamps)).cumsum()
        close = price + steps
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": stamps,
                    "open": close - 0.5,
                    "high": close + 2.0,
                    "low": close - 2.0,
                    "close": close,
                    "volume": 0.0,
                }
            )
        )
        price = float(close[-1])
    return pd.concat(frames, ignore_index=True)


class TestBuildFrame(unittest.TestCase):
    def test_adds_stochastic_rsi_columns_in_range(self):
        frame = algo4.build_cpr_algo4_frame(one_minute_days())
        for column in ("srsi_k", "srsi_d", "vwap", "rsi", "ema5", "ema20", "r1", "s1", "prev_high", "prev_low"):
            self.assertIn(column, frame.columns)
        valid = frame["srsi_k"].dropna()
        self.assertFalse(valid.empty)
        self.assertTrue(((valid >= 0.0) & (valid <= 100.0)).all())

    def test_no_look_ahead_in_engine_columns(self):
        data = one_minute_days()
        full = algo4.build_cpr_algo4_frame(data)
        cut = len(data) - 500
        prefix = algo4.build_cpr_algo4_frame(data.iloc[:cut])
        columns = ["open", "high", "low", "close", "vwap", "rsi", "ema5", "ema20", "srsi_k", "srsi_d", "r1", "s1"]
        merged = prefix[["timestamp", *columns]].merge(
            full[["timestamp", *columns]], on="timestamp", suffixes=("_p", "_f")
        )
        self.assertEqual(len(merged), len(prefix))
        for column in columns:
            np.testing.assert_allclose(
                merged[f"{column}_p"].to_numpy(dtype=float),
                merged[f"{column}_f"].to_numpy(dtype=float),
                equal_nan=True,
                err_msg=column,
            )


# --- Configuration ------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_rejects_unknown_exit_mode(self):
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(exit_mode="SOMETIMES")

    def test_rejects_non_finite_or_non_positive_values(self):
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(max_stop_points=float("nan"))
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(max_stop_points=0.0)
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(min_vwap_body_fraction=1.5)
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(srsi_oversold=90.0, srsi_overbought=80.0)

    def test_rejects_cutoff_before_regime_bar(self):
        with self.assertRaises(ValueError):
            algo4.CPRAlgo4Config(entry_cutoff=time(9, 20))


# --- Regime --------------------------------------------------------------------------


class TestRegime(unittest.TestCase):
    def regime_after(self, close_0925, open_0915=21950.0, levels=SIDEWAYS_LEVELS):
        engine = algo4.CPRAlgo4Engine()
        feed(
            engine,
            [
                bar("09:15", open_0915, 22000, 21900, 21950, levels),
                bar("09:20", 21950, 22000, 21900, 21950, levels),
                bar("09:25", 21950, 22200, 21800, close_0925, levels),
            ],
        )
        return engine

    def test_close_above_zone_is_trending_up(self):
        engine = self.regime_after(22121.0)
        self.assertEqual(engine.regime, algo4.REGIME_TRENDING_UP)
        self.assertEqual(engine.trend_dir, "UP")

    def test_close_below_zone_is_trending_down(self):
        engine = self.regime_after(21879.0)
        self.assertEqual(engine.regime, algo4.REGIME_TRENDING_DOWN)
        self.assertEqual(engine.trend_dir, "DOWN")

    def test_close_inside_or_on_the_zone_edge_is_sideways(self):
        self.assertEqual(self.regime_after(22000.0).regime, algo4.REGIME_SIDEWAYS)
        self.assertEqual(self.regime_after(22120.0).regime, algo4.REGIME_SIDEWAYS)
        self.assertEqual(self.regime_after(21880.0).regime, algo4.REGIME_SIDEWAYS)

    def test_only_the_0925_close_counts_not_the_open(self):
        # A gap-up open far above the zone that closes back inside by 09:25.
        self.assertEqual(self.regime_after(22000.0, open_0915=22300.0).regime, algo4.REGIME_SIDEWAYS)

    def test_zone_uses_max_r1_pdh_and_min_s1_pdl(self):
        levels = dict(SIDEWAYS_LEVELS, r1=22150.0, prev_high=22120.0, s1=21850.0, prev_low=21880.0)
        # 22130 is above PDH but below R1 -> still inside [21850, 22150].
        self.assertEqual(self.regime_after(22130.0, levels=levels).regime, algo4.REGIME_SIDEWAYS)

    def test_missing_0925_bar_means_no_trade(self):
        engine = algo4.CPRAlgo4Engine()
        feed(engine, [bar("09:15", 21950, 21960, 21940, 21950), bar("09:30", 21950, 21960, 21940, 22300)])
        self.assertEqual(engine.regime, algo4.REGIME_NO_TRADE)

    def test_missing_levels_mean_no_trade(self):
        levels = dict(SIDEWAYS_LEVELS, r1=float("nan"))
        self.assertEqual(self.regime_after(22000.0, levels=levels).regime, algo4.REGIME_NO_TRADE)

    def test_new_session_resets_state(self):
        engine = self.regime_after(22121.0)
        engine.on_bar(bar("09:15", 21950, 21960, 21940, 21950) | {"timestamp": pd.Timestamp("2026-05-07 09:15")})
        self.assertIsNone(engine.regime)
        self.assertIsNone(engine.trend_dir)


# --- Trade-plan geometry -------------------------------------------------------------


class TestTradePlan(unittest.TestCase):
    def plan(self, **kwargs):
        values = {
            "direction": "LONG",
            "premise": algo4.PREMISE_SIDEWAYS,
            "entry": 21930.0,
            "stop": 21915.0,
            "levels": SIDEWAYS_LEVELS,
            "config": algo4.CPRAlgo4Config(),
        }
        values.update(kwargs)
        return algo4.build_trade_plan(**values)

    def test_long_target_mode_books_at_one_r(self):
        plan, reason = self.plan()
        self.assertEqual(reason, "")
        self.assertEqual(plan.risk, 15.0)
        self.assertEqual(plan.target, 21945.0)  # 1R; next level 21995-2 is further
        self.assertEqual(plan.first_milestone, 21945.0)
        # Following = earlier of 2R (21960) and the level after 21993 (22000-2).
        self.assertEqual(plan.following_milestone, 21960.0)
        self.assertEqual(plan.final_target, 22198.0)
        self.assertEqual(plan.current_stop, 21915.0)

    def test_trail_mode_has_no_fixed_target(self):
        plan, _ = self.plan(config=algo4.CPRAlgo4Config(exit_mode="TRAIL"))
        self.assertTrue(math.isnan(plan.target))
        self.assertEqual(plan.exit_mode, "TRAIL")

    def test_short_geometry_mirrors(self):
        plan, reason = self.plan(direction="SHORT", entry=22070.0, stop=22085.0)
        self.assertEqual(reason, "")
        self.assertEqual(plan.target, 22055.0)
        self.assertEqual(plan.final_target, 21802.0)

    def test_rejects_stop_on_wrong_side(self):
        self.assertEqual(self.plan(stop=21935.0), (None, "invalid_stop_geometry"))

    def test_rejects_risk_wider_than_thirty_points(self):
        self.assertEqual(self.plan(stop=21899.0), (None, "risk_too_wide"))

    def test_rejects_next_level_under_one_r(self):
        # Next level 21995 - 2 = 21993 is only 8 points from 21985; risk is 15.
        self.assertEqual(self.plan(entry=21985.0, stop=21970.0), (None, "next_level_under_one_r"))

    def test_level_inside_the_buffer_is_skipped(self):
        # 21995 - 2 = 21993 is not beyond 21994, so the next usable level is 22000 - 2.
        plan, reason = self.plan(entry=21994.0, stop=21990.0)
        self.assertEqual(reason, "")
        self.assertEqual(plan.first_milestone, 21998.0)

    def test_rejects_when_final_target_is_behind_entry(self):
        # A malformed ladder whose R2 sits below the entry: the booking level is behind us.
        levels = dict(SIDEWAYS_LEVELS, r2=21920.0)
        self.assertEqual(self.plan(levels=levels, entry=21930.0, stop=21925.0)[1], "final_target_behind_entry")

    def test_first30_extreme_can_book_earlier(self):
        plan, _ = self.plan(first30_extreme=21940.0)
        self.assertEqual(plan.target, 21940.0)

    def test_first30_extreme_behind_entry_is_ignored(self):
        plan, _ = self.plan(first30_extreme=21925.0)
        self.assertEqual(plan.target, 21945.0)


# --- Sideways entries ------------------------------------------------------------


class TestSidewaysEntries(unittest.TestCase):
    def test_oversold_cross_up_buys_with_swing_low_stop(self):
        decision = feed(algo4.CPRAlgo4Engine(), sideways_long_rows())
        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.plan.premise, algo4.PREMISE_SIDEWAYS)
        self.assertEqual(decision.plan.entry, 21930.0)
        self.assertEqual(decision.plan.original_stop, 21915.0)
        self.assertEqual(decision.plan.target, 21945.0)

    def test_overbought_cross_down_buys_puts_with_swing_high_stop(self):
        decision = feed(algo4.CPRAlgo4Engine(), sideways_short_rows())
        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.plan.original_stop, 22085.0)
        self.assertEqual(decision.plan.target, 22055.0)

    def test_cross_outside_the_oversold_zone_holds(self):
        rows = sideways_long_rows()
        rows[-1]["srsi_k"], rows[-1]["srsi_d"] = 25.0, 14.0
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), rows).reason, "no_setup")

    def test_no_confirmed_swing_holds(self):
        rows = sideways_long_rows()
        # Make the lows keep falling so no swing low is confirmed by 09:45.
        for row, low in zip(rows, (21935, 21930, 21925, 21920, 21915, 21910, 21905), strict=True):
            row["low"] = float(low)
        decision = feed(algo4.CPRAlgo4Engine(), rows)
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.reason, "missing_swing_stop")

    def test_swing_stop_wider_than_thirty_points_holds(self):
        decision = feed(algo4.CPRAlgo4Engine(), sideways_long_rows(swing_low=21890.0))
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.reason, "risk_too_wide")

    def test_first30_target_only_when_enabled(self):
        config = algo4.CPRAlgo4Config(first30_target=True)
        decision = feed(algo4.CPRAlgo4Engine(config), sideways_long_rows())
        # First-30-minute high (09:15-09:40 bars) is 21942 < 1R at 21945.
        self.assertEqual(decision.plan.target, 21942.0)

    def test_trending_filters_do_not_apply_to_sideways(self):
        rows = sideways_long_rows()
        rows[-1]["rsi"] = 30.0  # would fail the trending RSI > 45 filter
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), rows).action, "ENTER_LONG")


# --- Trending entries ---------------------------------------------------------------


class TestTrendingEntries(unittest.TestCase):
    def test_continuation_long_after_close_back_above_vwap(self):
        decision = feed(algo4.CPRAlgo4Engine(), continuation_long_rows())
        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.plan.premise, algo4.PREMISE_CONTINUATION)
        self.assertEqual(decision.plan.original_stop, 22029.0)  # entry candle low
        self.assertEqual(decision.plan.target, 22051.0)

    def test_body_fraction_boundary(self):
        # Body 22030-22040; VWAP 22036 puts exactly 40% above -> allowed.
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), continuation_long_rows(entry_vwap=22036.0)).action, "ENTER_LONG")
        # VWAP 22036.1 leaves 39% above -> rejected.
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), continuation_long_rows(entry_vwap=22036.1)).reason, "no_setup")

    def test_doji_never_qualifies(self):
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), continuation_long_rows(entry_open=22040.0)).reason, "no_setup")

    def test_rsi_filter(self):
        decision = feed(algo4.CPRAlgo4Engine(), continuation_long_rows(rsi=45.0))
        self.assertEqual(decision.reason, "trend_filters_rejected")

    def test_ema_order_filter(self):
        self.assertEqual(
            feed(algo4.CPRAlgo4Engine(), continuation_long_rows(ema5=22027.0, ema20=22029.0)).reason,
            "trend_filters_rejected",
        )

    def test_ema_slope_filter(self):
        # EMA20 falls from 22028 (prior bar) to 22027.5 -> not rising.
        self.assertEqual(
            feed(algo4.CPRAlgo4Engine(), continuation_long_rows(ema20=22027.5)).reason, "trend_filters_rejected"
        )

    def test_vwap_support_alternative(self):
        rows = [
            *trend_regime_rows(),
            # Red candle supported on VWAP (low <= VWAP < close), closing ABOVE VWAP,
            # so the close-below-then-above pattern cannot be what fires.
            bar("09:30", 22036, 22037, 22030, 22033, TREND_LEVELS, vwap=22031.0, ema5=22031.0, ema20=22028.0),
            bar(
                "09:35", 22033, 22042, 22031, 22040, TREND_LEVELS,
                vwap=22032.0, rsi=58.0, ema5=22033.0, ema20=22029.0,
            ),
        ]
        decision = feed(algo4.CPRAlgo4Engine(), rows)
        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.plan.original_stop, 22031.0)

    def test_continuation_short_on_down_day(self):
        rows = [
            bar("09:15", 21775, 21780, 21770, 21772, TREND_LEVELS),
            bar("09:20", 21772, 21776, 21766, 21770, TREND_LEVELS),
            bar("09:25", 21770, 21774, 21764, 21768, TREND_LEVELS),  # below 21780 -> DOWN
            bar("09:30", 21768, 21772, 21766, 21770, TREND_LEVELS, vwap=21769.0, ema5=21769.0, ema20=21772.0),
            bar(
                "09:35", 21770, 21771, 21758, 21760, TREND_LEVELS,
                vwap=21768.0, rsi=42.0, ema5=21766.0, ema20=21771.0,
            ),
        ]
        decision = feed(algo4.CPRAlgo4Engine(), rows)
        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.plan.original_stop, 21771.0)

    def test_no_entry_on_or_after_the_cutoff_bar(self):
        def rows(prev_clock, entry_clock):
            return [
                *trend_regime_rows(),
                bar(prev_clock, 22032, 22034, 22028, 22030, TREND_LEVELS, vwap=22031.0, ema5=22031.0, ema20=22028.0),
                bar(
                    entry_clock, 22030, 22042, 22029, 22040, TREND_LEVELS,
                    vwap=22032.0, rsi=58.0, ema5=22033.0, ema20=22029.0,
                ),
            ]

        # The 14:50 bar completes at 14:55 -> allowed; the 14:55 bar completes at 15:00 -> blocked.
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), rows("14:45", "14:50")).action, "ENTER_LONG")
        self.assertEqual(feed(algo4.CPRAlgo4Engine(), rows("14:50", "14:55")).reason, "after_entry_cutoff")

    def test_no_reentry_on_the_bar_an_exit_happened(self):
        engine = algo4.CPRAlgo4Engine()
        rows = continuation_long_rows()
        feed(engine, rows[:-1])
        engine.on_exit(rows[-1]["timestamp"])
        self.assertEqual(engine.on_bar(rows[-1]).reason, "exit_bar")


# --- Structure flips and reversals --------------------------------------------------


class TestStructureAndReversal(unittest.TestCase):
    def test_lower_high_and_lower_low_flip_the_trend_and_cut_a_long(self):
        engine = algo4.CPRAlgo4Engine()
        rows = zigzag_rows()
        feed(engine, rows[:-1])
        self.assertEqual(engine.trend_dir, "UP")
        decision = engine.on_bar(rows[-1], make_plan(exit_mode="TARGET", target=22200.0))
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.reason, algo4.EXIT_STRUCTURE)
        self.assertEqual(engine.trend_dir, "DOWN")
        self.assertTrue(engine.reversal_pending)

    def test_flip_while_flat_just_changes_direction(self):
        engine = algo4.CPRAlgo4Engine()
        decision = feed(engine, zigzag_rows())
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(engine.trend_dir, "DOWN")
        self.assertTrue(engine.reversal_pending)

    def test_reversal_sequence_enters_the_new_direction(self):
        engine = algo4.CPRAlgo4Engine()
        decision = feed(engine, zigzag_rows() + reversal_short_rows())
        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.plan.premise, algo4.PREMISE_REVERSAL)
        self.assertEqual(decision.plan.original_stop, 22139.0)  # entry candle high

    def test_reversal_needs_the_green_pullback_first(self):
        engine = algo4.CPRAlgo4Engine()
        step_a, _, step_c = reversal_short_rows()
        feed(engine, [*zigzag_rows(), step_a])
        step_c = dict(step_c, timestamp=pd.Timestamp(f"{DAY} 10:15"))
        self.assertEqual(engine.on_bar(step_c).reason, "waiting_for_reversal_sequence")

    def test_continuation_pattern_is_blocked_while_a_reversal_is_pending(self):
        engine = algo4.CPRAlgo4Engine()
        feed(engine, zigzag_rows())
        rows = [
            bar("10:10", 22120, 22126, 22118, 22124, TREND_LEVELS, vwap=22122.0, ema5=22121.0, ema20=22130.0),
            bar(
                "10:15", 22118, 22120, 22110, 22112, TREND_LEVELS,
                vwap=22121.0, rsi=40.0, ema5=22118.0, ema20=22128.0,
            ),
        ]
        decision = feed(engine, rows)
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.reason, "waiting_for_reversal_sequence")

    def test_after_the_reversal_fill_continuation_resumes(self):
        engine = algo4.CPRAlgo4Engine()
        entry = feed(engine, zigzag_rows() + reversal_short_rows())
        engine.on_entry_filled(entry.plan)
        self.assertFalse(engine.reversal_pending)
        engine.on_exit(pd.Timestamp(f"{DAY} 10:25"))
        rows = [
            bar("10:25", 22124, 22130, 22119, 22126, TREND_LEVELS),
            bar("10:30", 22120, 22126, 22118, 22124, TREND_LEVELS, vwap=22122.0, ema5=22121.0, ema20=22130.0),
            # Stop 22120 (risk 8) keeps R2+2 = 22102 at least 1:1 away (10 points).
            bar(
                "10:35", 22118, 22120, 22110, 22112, TREND_LEVELS,
                vwap=22121.0, rsi=40.0, ema5=22118.0, ema20=22128.0,
            ),
        ]
        decision = feed(engine, rows)
        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.plan.premise, algo4.PREMISE_CONTINUATION)


# --- Open-position management ---------------------------------------------------


class TestManagement(unittest.TestCase):
    def sideways_engine(self):
        engine = algo4.CPRAlgo4Engine()
        feed(engine, sideways_long_rows()[:3])
        return engine

    def test_sideways_long_exits_on_overbought_cross_down(self):
        engine = self.sideways_engine()
        plan = make_plan(premise=algo4.PREMISE_SIDEWAYS, exit_mode="TARGET", target=21945.0)
        engine.on_bar(bar("09:30", 21932, 21938, 21925, 21930, srsi_k=85.0, srsi_d=80.0), plan)
        decision = engine.on_bar(bar("09:35", 21930, 21936, 21928, 21931, srsi_k=82.0, srsi_d=84.0), plan)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.reason, algo4.EXIT_SRSI)

    def test_srsi_exit_does_not_apply_to_trend_trades(self):
        engine = self.sideways_engine()
        plan = make_plan(exit_mode="TARGET", target=22200.0)
        engine.on_bar(bar("09:30", 21932, 21938, 21925, 21930, srsi_k=85.0, srsi_d=80.0), plan)
        decision = engine.on_bar(bar("09:35", 21930, 21936, 21928, 21931, srsi_k=82.0, srsi_d=84.0), plan)
        self.assertEqual(decision.action, "HOLD")

    def trail_rows(self):
        return [
            *trend_regime_rows(22030.0),
            bar("09:30", 22040, 22052, 22039, 22051, TREND_LEVELS),  # close >= first milestone 22050
            bar("09:35", 22051, 22058, 22050, 22056, TREND_LEVELS),
            bar("09:40", 22056, 22063, 22055, 22061, TREND_LEVELS),  # close >= following milestone 22060
            bar("09:45", 22061, 22062, 22048, 22049, TREND_LEVELS),  # closes below the prior low
        ]

    def test_trail_normal_trade_breakeven_then_prior_low_trail(self):
        engine = algo4.CPRAlgo4Engine(algo4.CPRAlgo4Config(exit_mode="TRAIL"))
        plan = make_plan()
        rows = self.trail_rows()
        feed(engine, rows[:3])
        engine.on_bar(rows[3], plan)
        self.assertEqual(plan.current_stop, 22040.0)  # breakeven
        self.assertTrue(plan.trail_armed)
        self.assertEqual(engine.on_bar(rows[4], plan).action, "HOLD")
        self.assertEqual(engine.on_bar(rows[5], plan).action, "HOLD")
        decision = engine.on_bar(rows[6], plan)
        self.assertEqual(decision.action, "EXIT")
        self.assertEqual(decision.reason, algo4.EXIT_TRAIL)

    def test_trail_reversal_trade_is_staged(self):
        engine = algo4.CPRAlgo4Engine(algo4.CPRAlgo4Config(exit_mode="TRAIL"))
        plan = make_plan(premise=algo4.PREMISE_REVERSAL)
        rows = self.trail_rows()
        feed(engine, rows[:3])
        engine.on_bar(rows[3], plan)
        self.assertEqual(plan.current_stop, 22040.0)  # stage 1: breakeven only
        self.assertFalse(plan.trail_armed)
        engine.on_bar(rows[4], plan)
        engine.on_bar(rows[5], plan)
        self.assertEqual(plan.current_stop, 22050.0)  # stage 2: lock 1R
        self.assertTrue(plan.trail_armed)
        self.assertEqual(engine.on_bar(rows[6], plan).reason, algo4.EXIT_TRAIL)

    def test_stop_never_loosens(self):
        engine = algo4.CPRAlgo4Engine(algo4.CPRAlgo4Config(exit_mode="TRAIL"))
        plan = make_plan(current_stop=22045.0)
        rows = self.trail_rows()
        feed(engine, rows[:3])
        engine.on_bar(rows[3], plan)
        self.assertEqual(plan.current_stop, 22045.0)

    def test_target_mode_never_moves_the_stop(self):
        engine = algo4.CPRAlgo4Engine()
        plan = make_plan(exit_mode="TARGET", target=22050.0)
        rows = self.trail_rows()
        feed(engine, rows[:3])
        for row in rows[3:]:
            self.assertEqual(engine.on_bar(row, plan).action, "HOLD")
        self.assertEqual(plan.current_stop, 22030.0)


class TestIntrabarExit(unittest.TestCase):
    def test_long_stop_is_checked_before_target(self):
        plan = make_plan(exit_mode="TARGET", target=22050.0)
        self.assertEqual(algo4.check_intrabar_exit(plan, high=22051.0, low=22029.0), (algo4.EXIT_STOP, 22030.0))
        self.assertEqual(algo4.check_intrabar_exit(plan, high=22051.0, low=22035.0), (algo4.EXIT_TARGET, 22050.0))
        self.assertIsNone(algo4.check_intrabar_exit(plan, high=22049.0, low=22031.0))

    def test_trail_mode_books_only_the_final_level(self):
        plan = make_plan()
        self.assertIsNone(algo4.check_intrabar_exit(plan, high=22097.0, low=22035.0))
        self.assertEqual(algo4.check_intrabar_exit(plan, high=22098.0, low=22035.0), (algo4.EXIT_FINAL, 22098.0))

    def test_short_mirror(self):
        plan = make_plan(
            direction="SHORT", entry=21760.0, original_stop=21770.0, current_stop=21770.0,
            target=21750.0, final_target=21702.0, exit_mode="TARGET",
        )
        self.assertEqual(algo4.check_intrabar_exit(plan, high=21770.0, low=21749.0), (algo4.EXIT_STOP, 21770.0))
        self.assertEqual(algo4.check_intrabar_exit(plan, high=21765.0, low=21750.0), (algo4.EXIT_TARGET, 21750.0))


class TestScaleIn(unittest.TestCase):
    def r1_rows(self):
        return [
            *trend_regime_rows(),
            bar("09:30", 22008, 22010, 21998, 22001, TREND_LEVELS),  # red candle touching R1 (22000)
            bar("09:35", 22001, 22009, 21999, 22006, TREND_LEVELS),  # green candle reclaiming R1
        ]

    def run_rows(self, plan, config=None):
        engine = algo4.CPRAlgo4Engine(config)
        return feed(engine, self.r1_rows(), plan)

    def test_red_then_green_at_r1_adds_to_a_trending_long(self):
        self.assertEqual(self.run_rows(make_plan(exit_mode="TARGET", target=22200.0)).action, "SCALE_IN")

    def test_only_once(self):
        plan = make_plan(exit_mode="TARGET", target=22200.0, scale_in_used=True)
        self.assertEqual(self.run_rows(plan).action, "HOLD")

    def test_never_for_shorts_or_sideways_or_when_disabled(self):
        short = make_plan(
            direction="SHORT", current_stop=22300.0, original_stop=22300.0, first_milestone=22030.0,
            following_milestone=22020.0, target=21900.0, final_target=21702.0, exit_mode="TARGET",
        )
        self.assertEqual(self.run_rows(short).action, "HOLD")
        sideways = make_plan(premise=algo4.PREMISE_SIDEWAYS, exit_mode="TARGET", target=22200.0)
        self.assertEqual(self.run_rows(sideways).action, "HOLD")
        disabled = algo4.CPRAlgo4Config(scale_in_enabled=False)
        self.assertEqual(self.run_rows(make_plan(exit_mode="TARGET", target=22200.0), disabled).action, "HOLD")


if __name__ == "__main__":
    unittest.main()
