import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import unittest
import warnings
from contextlib import ExitStack
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import pandas as pd

from Dependencies.broker_contract import BrokerQueryResult, OrderResult, OrderStatus

# =============================================================================
# DYNAMIC MODULE IMPORT
# =============================================================================
# Python usually expects file names without spaces (e.g. "my_script.py").
# Since the original strategy file has spaces in its name, we must use
# 'importlib' to manually load the file as a module so we can test its contents.
file_path = Path(__file__).parent / "Nifty Multi Strategy Front Test - Master File.py"
spec = importlib.util.spec_from_file_location("master_file", file_path)
master_file = importlib.util.module_from_spec(spec)
sys.modules["master_file"] = master_file

# =============================================================================
# MOCKING (FAKE DATA) FOR SAFE TESTING
# =============================================================================
# We do not want our tests to accidentally place real trades or connect to
# the DhanHQ API. So we "mock" (fake) the API connections.
# The patch of 'dhanhq.dhanhq' completely disables the real SDK while the
# script loads.
with (
    patch.dict(os.environ, {"DHAN_CLIENT_CODE": "test", "DHAN_TOKEN_ID": "test"}),
    patch("dhanhq.dhanhq"),
):
    try:
        # We execute the file so that all classes and functions become available
        spec.loader.exec_module(master_file)
    except Exception as e:
        print(f"Failed to load master_file for testing: {e}")


# The Flattrade helper is loaded separately so its low-level REST behaviour can
# be tested without starting the master runner or making any network requests.
flattrade_file_path = Path(__file__).parent / "Dependencies" / "Flattrade API" / "flattrade_execution.py"
flattrade_module = None
if flattrade_file_path.is_file():
    flattrade_spec = importlib.util.spec_from_file_location(
        "flattrade_execution_under_test", flattrade_file_path
    )
    flattrade_module = importlib.util.module_from_spec(flattrade_spec)
    sys.modules["flattrade_execution_under_test"] = flattrade_module
    flattrade_spec.loader.exec_module(flattrade_module)

flattrade_diagnostic_path = (
    Path(__file__).parent
    / "Dependencies"
    / "Flattrade API"
    / "diagnose_flattrade_symbol.py"
)
flattrade_diagnostic_module = None
if flattrade_diagnostic_path.is_file():
    flattrade_diagnostic_spec = importlib.util.spec_from_file_location(
        "diagnose_flattrade_symbol_under_test", flattrade_diagnostic_path
    )
    flattrade_diagnostic_module = importlib.util.module_from_spec(
        flattrade_diagnostic_spec
    )
    sys.modules["diagnose_flattrade_symbol_under_test"] = flattrade_diagnostic_module
    # The diagnostic imports its sibling module by name, just like a standalone
    # invocation. Temporarily expose the already-loaded, network-free test copy.
    with patch.dict(sys.modules, {"flattrade_execution": flattrade_module}):
        flattrade_diagnostic_spec.loader.exec_module(flattrade_diagnostic_module)


# =============================================================================
# TEST SUITE: UTILITIES
# =============================================================================
class TestMasterFileUtilities(unittest.TestCase):
    """
    This class tests the small "helper" functions in the strategy file.
    Helper functions do basic jobs like safely converting text to numbers.
    """

    def test_safe_float(self):
        """Verify that strings are safely converted to decimals (floats), falling back on error."""
        self.assertEqual(master_file._safe_float("123.45"), 123.45)
        self.assertEqual(master_file._safe_float("abc", 10.0), 10.0)
        self.assertEqual(master_file._safe_float(None, 0.0), 0.0)

    def test_to_int_safe(self):
        """Verify that strings are safely converted to whole numbers (integers), falling back on error."""
        self.assertEqual(master_file._to_int_safe("123.45"), 123)
        self.assertEqual(master_file._to_int_safe("abc", 10), 10)
        self.assertEqual(master_file._to_int_safe(None, 0), 0)

    def test_infer_epoch_unit(self):
        """Ensure the system can guess if a timestamp is in seconds, ms, or us based on its size."""
        self.assertEqual(master_file._infer_epoch_unit(pd.Series([1672531200])), "s")
        self.assertEqual(master_file._infer_epoch_unit(pd.Series([1672531200000])), "ms")
        self.assertEqual(master_file._infer_epoch_unit(pd.Series([1672531200000000])), "us")

    def test_build_last_row_signature(self):
        """Test the fingerprinting mechanism used to detect if a candle's price has changed."""
        df = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00"],
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0]
        })
        sig = master_file.build_last_row_signature(df)
        self.assertIsNotNone(sig)
        self.assertEqual(sig[0], 1)
        self.assertEqual(sig[2], 100.0)
        self.assertEqual(sig[5], 102.0)

        self.assertIsNone(master_file.build_last_row_signature(None))
        self.assertIsNone(master_file.build_last_row_signature(pd.DataFrame()))


# =============================================================================
# TEST SUITE: SHARED MARKET DATA STORE
# =============================================================================
class TestSharedMarketDataStore(unittest.TestCase):
    """
    This tests the 'SharedMarketDataStore', which is like a central bulletin board.
    One thread fetches the data and pins it to the board, while the strategy threads
    read from it. We need to make sure the board works correctly.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()

    def test_update_and_get_ohlc(self):
        """Verify that simulated Open/High/Low/Close data is correctly saved and fetched from the store."""
        df = pd.DataFrame({
            "timestamp": ["2023-01-01 10:00:00"],
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0]
        })
        snapshot = self.store.update("1", df)
        self.assertEqual(snapshot.timeframe, "1")

        fetched = self.store.get("1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.timeframe, "1")
        self.assertEqual(fetched.frame.iloc[-1]["close"], 102.0)

    def test_ltp_cache(self):
        """Verify that the Last Traded Price (LTP) cache stores prices properly and ignores glitches."""
        self.store.update_ltp_map({("NSE_FNO", 1234): 150.5, ("IDX_I", 13): 20000.0})

        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 1234), 150.5)
        self.assertEqual(self.store.get_ltp_by_secid("IDX_I", 13), 20000.0)
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 9999, fallback=10.0), 10.0)

        self.store.update_ltp_map({("NSE_FNO", 1234): -50.0})
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 1234), 150.5)

        self.store.update_ltp_map({("NSE_FNO", 1234): float("inf")})
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 1234), 150.5)

        self.store.update_ltp_map({("NSE_FNO", 1234): "not-a-price"})
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 1234), 150.5)

    def test_invalid_ohlc_does_not_replace_last_good_snapshot(self):
        """Publication is atomic: invalid replacement data leaves the prior snapshot intact."""
        good = pd.DataFrame({
            "timestamp": ["2026-05-15 10:00:00"],
            "open": [100.0], "high": [105.0], "low": [95.0], "close": [102.0],
        })
        self.store.update("1", good)
        invalid = good.copy()
        invalid.loc[0, "close"] = float("inf")

        with self.assertRaises(master_file.MarketDataValidationError):
            self.store.update("1", invalid)

        self.assertEqual(self.store.get("1").frame.iloc[-1]["close"], 102.0)

    def test_subscriptions(self):
        """Check if option subscriptions can be correctly registered and unregistered."""
        sub = master_file.OptionSubscription(
            security_id=123, exchange_segment="NSE_FNO", trading_symbol="OPT1",
            right="CE", strike=20000.0, expiry=date(2023, 1, 26)
        )
        self.store.register_option_subscription(sub)

        subs = self.store.snapshot_option_subscriptions()
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].security_id, 123)

        self.store.unregister_option_subscription("NSE_FNO", 123)
        self.assertEqual(len(self.store.snapshot_option_subscriptions()), 0)


class TestWorkerMarketDataSafety(unittest.TestCase):
    """The feed gate and stale-feed unwind protect REAL money: live workers only.

    Operator decision (2026-07-17): a paper worker keeps entering and keeps its
    virtual position on the last-good snapshot -- the gate had blocked every
    paper strategy through the 17 Jul opening window while the feed warmed up.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.worker = master_file.AtmSingleLegStrategyWorker(
            self.store,
            threading.Event(),
            MagicMock(),
        )

    def test_live_entry_is_blocked_until_feed_recovers(self):
        self.worker.live_trading = True
        self.store.begin_market_data_monitoring()
        with patch.object(self.worker, "_get_underlying_spot") as get_spot:
            self.assertFalse(self.worker.enter_position("LONG", 22500.0))
        get_spot.assert_not_called()

    def test_paper_entry_allowed_while_feed_unhealthy(self):
        """A paper (virtual) worker sails through the feed gate: the entry
        attempt must reach the spot lookup instead of being blocked."""
        self.assertFalse(self.worker.live_trading)
        self.store.begin_market_data_monitoring()
        with patch.object(self.worker, "_get_underlying_spot", return_value=0.0) as get_spot:
            # Returns False further down (no spot LTP in this synthetic setup),
            # but the market-data gate itself must have been passed.
            self.assertFalse(self.worker.enter_position("LONG", 22500.0))
        get_spot.assert_called_once()

    def test_thirty_second_unhealthy_state_invokes_square_off_for_live(self):
        self.worker.live_trading = True
        health = MagicMock(
            monitoring=True,
            entry_allowed=False,
            liquidation_required=True,
            healthy_streak=0,
            unhealthy_seconds=31.0,
            reasons=("LTP IDX_I/13 is stale (41.0s)",),
        )
        self.worker.pos.active = True
        self.worker.exit_position = MagicMock()
        self.worker._flatten_additional_positions = MagicMock()
        self.worker._sweep_orphan_live_legs = MagicMock()

        with patch.object(self.store.market_data_health, "snapshot", return_value=health):
            consumed = self.worker._handle_market_data_health()

        self.assertTrue(consumed)
        self.worker.exit_position.assert_called_once_with("MARKET_DATA_UNHEALTHY")
        self.worker._flatten_additional_positions.assert_called_once_with(
            "MARKET_DATA_UNHEALTHY"
        )
        self.worker._sweep_orphan_live_legs.assert_called_once_with(force=True)

    def test_paper_worker_skips_market_data_square_off(self):
        """A paper worker's virtual position must survive a 30s+ feed outage:
        no forced close, no orphan sweep, and the poll is NOT consumed (the
        strategy keeps running on the last-good snapshot)."""
        self.assertFalse(self.worker.live_trading)
        health = MagicMock(
            monitoring=True,
            entry_allowed=False,
            liquidation_required=True,
            healthy_streak=0,
            unhealthy_seconds=31.0,
            reasons=("LTP IDX_I/13 is stale (41.0s)",),
        )
        self.worker.pos.active = True
        self.worker.exit_position = MagicMock()
        self.worker._flatten_additional_positions = MagicMock()
        self.worker._sweep_orphan_live_legs = MagicMock()

        with patch.object(self.store.market_data_health, "snapshot", return_value=health):
            consumed = self.worker._handle_market_data_health()

        self.assertFalse(consumed)
        self.worker.exit_position.assert_not_called()
        self.worker._flatten_additional_positions.assert_not_called()
        self.worker._sweep_orphan_live_legs.assert_not_called()
        # The 30s+ outage is still logged once for the operator's audit trail.
        self.assertTrue(self.worker._market_data_liquidation_logged)

    def test_unhealthy_primary_close_error_cannot_skip_other_exposure(self):
        self.worker.live_trading = True
        health = MagicMock(
            monitoring=True,
            entry_allowed=False,
            liquidation_required=True,
            healthy_streak=0,
            unhealthy_seconds=31.0,
            reasons=("newest completed one-minute bar is stale",),
        )
        self.worker.pos.active = True
        self.worker.exit_position = MagicMock(side_effect=RuntimeError("primary close failed"))
        self.worker._flatten_additional_positions = MagicMock()
        self.worker._sweep_orphan_live_legs = MagicMock()

        with patch.object(self.store.market_data_health, "snapshot", return_value=health):
            consumed = self.worker._handle_market_data_health()

        self.assertTrue(consumed)
        self.worker._flatten_additional_positions.assert_called_once_with(
            "MARKET_DATA_UNHEALTHY"
        )
        self.worker._sweep_orphan_live_legs.assert_called_once_with(force=True)


# =============================================================================
# TEST SUITE: DATA CLASSES
# =============================================================================
class TestDataclasses(unittest.TestCase):
    """
    Dataclasses are like blueprints for storing structured information.
    Here we test if the blueprints assemble the objects correctly.
    """

    def test_paper_position(self):
        """Test that the PaperPosition container correctly stores properties of a single ongoing trade."""
        pos = master_file.PaperPosition(active=True, direction="LONG", quantity=50, entry_trade_price=105.5)
        self.assertTrue(pos.active)
        self.assertEqual(pos.direction, "LONG")
        self.assertEqual(pos.quantity, 50)
        self.assertEqual(pos.entry_trade_price, 105.5)

    def test_hedged_paper_position(self):
        """Test that HedgedPaperPosition correctly stores info about a spread trade with two legs."""
        pos = master_file.HedgedPaperPosition(
            active=True, direction="BULLISH",
            main_quantity=50, main_entry_price=160.0,
            hedge_quantity=50, hedge_entry_price=10.0
        )
        self.assertTrue(pos.active)
        self.assertEqual(pos.main_quantity, 50)
        self.assertEqual(pos.hedge_entry_price, 10.0)


# =============================================================================
# TEST SUITE: PURE HELPER FUNCTIONS (extends TestMasterFileUtilities)
# =============================================================================
class TestPureHelpers(unittest.TestCase):
    """
    Tests the small deterministic helpers in the master file that don't need
    any broker, store, or worker setup: time gates, env loaders, OHLC
    resampling, and column-name resolution.
    """

    def test_live_mirror_requires_confirmed_live_nifty_fill(self):
        """A paper fallback must never trigger a real BankNIFTY mirror order."""

        self.assertTrue(master_file._should_open_bnf_mirror(False, False))
        self.assertTrue(master_file._should_open_bnf_mirror(True, True))
        self.assertFalse(master_file._should_open_bnf_mirror(True, False))

    def test_indeterminate_alert_includes_escaped_broker_evidence(self):
        """Operators must see the exact exposure evidence in the Telegram alert."""

        message = master_file.format_trade_message(
            {
                "action": "INDETERMINATE_EXPOSURE",
                "strategy": "Unsafe & Test",
                "mode": "LIVE_INDETERMINATE",
                "side": "BUY",
                "symbol": "NIFTY<CE>",
                "order_id": "ORD<1>",
                "requested_quantity": 50,
                "filled_quantity": 20,
                "remaining_quantity": 30,
                "status": "PARTIAL",
                "broker_state": "OPEN&PENDING",
                "reason": "response <lost>",
            }
        )

        self.assertIn("NIFTY&lt;CE&gt;", message)
        self.assertIn("ORD&lt;1&gt;", message)
        self.assertIn("20 / 50", message)
        self.assertIn("remaining 30", message)
        self.assertIn("OPEN&amp;PENDING", message)
        self.assertIn("response &lt;lost&gt;", message)

    def test_exit_failed_alert_includes_escaped_reason(self):
        """A failed close alert is actionable only when its reason is visible."""

        message = master_file.format_trade_message(
            {
                "action": "EXIT_FAILED",
                "strategy": "Risk & Test",
                "mode": "LIVE_REJECTED",
                "direction": "LONG",
                "reason": "broker <rejected> & position remains open",
            }
        )

        self.assertIn("broker &lt;rejected&gt; &amp; position remains open", message)

    def test_env_str_strips_quotes_and_uses_default(self):
        """`_env_str` strips surrounding quotes and falls back when unset."""
        with patch.dict(os.environ, {"DUMMY_KEY": '"abc"'}, clear=False):
            self.assertEqual(master_file._env_str("DUMMY_KEY", "fallback"), "abc")
        with patch.dict(os.environ, {"DUMMY_KEY": "'xyz'"}, clear=False):
            self.assertEqual(master_file._env_str("DUMMY_KEY", "fallback"), "xyz")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DUMMY_KEY", None)
            self.assertEqual(master_file._env_str("DUMMY_KEY", "fallback"), "fallback")

    def test_env_float_handles_bad_input(self):
        """`_env_float` returns the default on garbage rather than crashing."""
        with patch.dict(os.environ, {"DUMMY_F": "1.5"}, clear=False):
            self.assertEqual(master_file._env_float("DUMMY_F", 9.9), 1.5)
        with patch.dict(os.environ, {"DUMMY_F": "not-a-number"}, clear=False):
            self.assertEqual(master_file._env_float("DUMMY_F", 9.9), 9.9)

    def test_env_int_handles_bad_input(self):
        """`_env_int` accepts float strings and falls back on garbage."""
        with patch.dict(os.environ, {"DUMMY_I": "12.7"}, clear=False):
            self.assertEqual(master_file._env_int("DUMMY_I", 0), 12)
        with patch.dict(os.environ, {"DUMMY_I": "junk"}, clear=False):
            self.assertEqual(master_file._env_int("DUMMY_I", 5), 5)

    def test_first_existing_col_case_insensitive(self):
        """`_first_existing_col` looks up case-insensitively and returns the first match."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        self.assertEqual(master_file._first_existing_col(df, ["open"]), "Open")
        self.assertEqual(master_file._first_existing_col(df, ["CLOSE"]), "Close")
        self.assertEqual(
            master_file._first_existing_col(df, ["missing", "high"]), "High"
        )
        self.assertIsNone(master_file._first_existing_col(df, ["volume"]))

    def test_is_before_time_and_after_time(self):
        """Time gates compare wall-clock against the supplied HH:MM threshold."""
        # Mock now() to 12:30 -> before 13:00, after 11:00.
        fake_now = datetime(2026, 5, 15, 12, 30, 0)
        with patch.object(master_file, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(master_file.is_before_time(13, 0))
            self.assertFalse(master_file.is_before_time(11, 0))
            self.assertTrue(master_file.is_after_time(11, 0))
            self.assertFalse(master_file.is_after_time(13, 0))

    def test_time_gates_convert_utc_now_to_ist(self):
        """Market cutoffs use Asia/Kolkata even when the host clock is UTC."""

        # 07:30 UTC is 13:00 IST. A host-local comparison would incorrectly
        # report that noon has not arrived yet.
        utc_now = datetime(2026, 5, 15, 7, 30, tzinfo=UTC)
        with patch.object(master_file, "datetime") as mock_dt:
            mock_dt.now.return_value = utc_now
            self.assertTrue(master_file.is_after_time(12, 0))
            self.assertFalse(master_file.is_before_time(12, 0))

    def test_resample_ohlc_from_1m_passthrough(self):
        """1-minute resampling is a no-op."""
        ohlc = pd.DataFrame({
            "timestamp": pd.date_range("2026-05-15 09:15", periods=3, freq="1min"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low":  [99, 100, 101],
            "close": [101, 102, 103],
        })
        result = master_file.resample_ohlc_from_1m(ohlc, 1)
        self.assertEqual(len(result), 3)

    def test_resample_ohlc_from_1m_drops_incomplete_buckets(self):
        """Only fully-formed 5-min buckets (5 source bars) survive."""
        # 7 bars - one complete 5-min bucket (rows 0-4), one partial (rows 5-6).
        ts = pd.date_range("2026-05-15 09:15", periods=7, freq="1min")
        ohlc = pd.DataFrame({
            "timestamp": ts,
            "open":  [100, 101, 102, 103, 104, 105, 106],
            "high":  [101, 102, 103, 104, 105, 106, 107],
            "low":   [99,  100, 101, 102, 103, 104, 105],
            "close": [101, 102, 103, 104, 105, 106, 107],
        })
        result = master_file.resample_ohlc_from_1m(ohlc, 5)
        # Only the complete bucket survives.
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["open"], 100)
        self.assertEqual(result.iloc[0]["high"], 105)
        self.assertEqual(result.iloc[0]["low"], 99)
        self.assertEqual(result.iloc[0]["close"], 105)

    def test_resample_ohlc_raises_on_missing_columns(self):
        """Missing required OHLC columns is an error, not a silent skip."""
        ohlc = pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-15 09:15")]})
        with self.assertRaises(ValueError):
            master_file.resample_ohlc_from_1m(ohlc, 5)

    def test_resample_rejects_count_complete_but_slot_incomplete_bucket(self):
        """A duplicate minute cannot hide a missing minute in a five-row bucket."""
        timestamps = pd.to_datetime([
            "2026-05-15 09:15", "2026-05-15 09:15", "2026-05-15 09:17",
            "2026-05-15 09:18", "2026-05-15 09:19",
        ])
        ohlc = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100, 100, 102, 103, 104],
            "high": [101, 101, 103, 104, 105],
            "low": [99, 99, 101, 102, 103],
            "close": [100.5, 100.5, 102.5, 103.5, 104.5],
        })
        result = master_file.resample_ohlc_from_1m(ohlc, 5)
        self.assertTrue(result.empty)

    def test_color_pnl_text_static(self):
        """`_color_pnl_text` colors positive green, negative red, zero plain."""
        pos = master_file.BasePaperStrategyWorker._color_pnl_text(12.5)
        neg = master_file.BasePaperStrategyWorker._color_pnl_text(-3.2)
        zero = master_file.BasePaperStrategyWorker._color_pnl_text(0.0)
        self.assertIn("12.50", pos)
        self.assertIn(master_file.ANSI_GREEN, pos)
        self.assertIn("-3.20", neg)
        self.assertIn(master_file.ANSI_RED, neg)
        self.assertEqual(zero, "0.00")


# =============================================================================
# TEST SUITE: DHANHQ INTRADAY RESPONSE NORMALIZATION
# =============================================================================
class TestNormalizeDhanResponse(unittest.TestCase):
    """
    `normalize_dhan_intraday_response` parses every OHLC response from the
    broker. The downstream pipeline cannot survive a malformed shape so the
    parser is the first line of defense.
    """

    def _make_dict_resp(self, n_bars):
        """Build a valid dhanhq-style response with `n_bars` 1-min candles."""
        base_ts = int(datetime(2026, 5, 15, 9, 15, tzinfo=None).timestamp())
        return {
            "status": "success",
            "data": {
                "timestamp": [base_ts + i * 60 for i in range(n_bars)],
                "open":   [100.0 + i for i in range(n_bars)],
                "high":   [101.0 + i for i in range(n_bars)],
                "low":    [99.0 + i for i in range(n_bars)],
                "close":  [100.5 + i for i in range(n_bars)],
                "volume": [1000 for _ in range(n_bars)],
            },
        }

    def test_normalize_happy_path(self):
        """Well-formed response is normalized into a sorted, IST-localized frame."""
        resp = self._make_dict_resp(master_file.MIN_BARS + 10)
        out = master_file.normalize_dhan_intraday_response(resp)
        self.assertEqual(
            list(out.columns), ["timestamp", "open", "high", "low", "close"]
        )
        self.assertGreaterEqual(len(out), master_file.MIN_BARS)
        # Sorted ascending.
        self.assertTrue(out["timestamp"].is_monotonic_increasing)
        # Timezone has been stripped (naive Asia/Kolkata time).
        self.assertTrue(
            all(ts.tzinfo is None for ts in pd.DatetimeIndex(out["timestamp"]))
        )

    def test_normalize_rejects_non_dict(self):
        """A non-dict response is unrecoverable - raise rather than corrupt."""
        with self.assertRaises(ValueError):
            master_file.normalize_dhan_intraday_response("not a dict")

    def test_normalize_rejects_failure_status(self):
        """Status != success raises with a remarks-bearing message."""
        with self.assertRaises(ValueError) as ctx:
            master_file.normalize_dhan_intraday_response(
                {"status": "failure", "remarks": "rate limit"}
            )
        self.assertIn("rate limit", str(ctx.exception))

    def test_normalize_rejects_missing_data(self):
        """Response with no `data` key is rejected."""
        with self.assertRaises(ValueError):
            master_file.normalize_dhan_intraday_response({"status": "success"})

    def test_normalize_rejects_missing_ohlc_columns(self):
        """A timestamp-only payload has no OHLC and must raise."""
        with self.assertRaises(ValueError):
            master_file.normalize_dhan_intraday_response(
                {"status": "success", "data": {"timestamp": [1672531200]}}
            )

    def test_normalize_rejects_too_few_bars(self):
        """Below-MIN_BARS frames are rejected to avoid corrupt warm-up."""
        resp = self._make_dict_resp(master_file.MIN_BARS - 5)
        with self.assertRaises(ValueError):
            master_file.normalize_dhan_intraday_response(resp)

    def test_normalize_rejects_non_finite_and_impossible_ohlc(self):
        """Infinity and impossible high/low geometry fail closed at ingestion."""
        resp = self._make_dict_resp(master_file.MIN_BARS + 1)
        resp["data"]["close"][3] = float("inf")
        with self.assertRaises(master_file.MarketDataValidationError):
            master_file.normalize_dhan_intraday_response(resp)

        resp = self._make_dict_resp(master_file.MIN_BARS + 1)
        resp["data"]["high"][3] = resp["data"]["close"][3] - 1.0
        with self.assertRaises(master_file.MarketDataValidationError):
            master_file.normalize_dhan_intraday_response(resp)

    def test_normalize_rejects_mixed_epoch_units(self):
        """One millisecond value among second epochs cannot corrupt the whole timeline."""
        resp = self._make_dict_resp(master_file.MIN_BARS + 1)
        resp["data"]["timestamp"][3] *= 1000
        with self.assertRaises(master_file.MarketDataValidationError):
            master_file.normalize_dhan_intraday_response(resp)


# =============================================================================
# TEST SUITE: OPTION-CHAIN PARSERS (PCR/VWAP + Delta-0.2 workers)
# =============================================================================
class TestOptionChainParsers(unittest.TestCase):
    """
    Static parsers for the DhanHQ option-chain payload. Both are `@staticmethod`
    so they're trivial to test without instantiating their worker classes.
    """

    def _chain_resp(self):
        """A minimal but realistic /optionchain payload."""
        return {
            "status": "success",
            "data": {
                "last_price": 22411.85,
                "oc": {
                    "22000.000000": {
                        "ce": {"last_price": 481.7, "oi": 1000.0,
                               "greeks": {"delta": 0.74}},
                        "pe": {"last_price": 22.1, "oi": 2500.0,
                               "greeks": {"delta": -0.10}},
                    },
                    "22500.000000": {
                        "ce": {"last_price": 100.0, "oi": 1500.0,
                               "greeks": {"delta": 0.20}},
                        "pe": {"last_price": 80.0,  "oi": 3000.0,
                               "greeks": {"delta": -0.40}},
                    },
                    # Strike with both legs at zero OI - dropped from OI parser.
                    "23000.000000": {
                        "ce": {"last_price": 5.0, "oi": 0.0,
                               "greeks": {"delta": 0.05}},
                        "pe": {"last_price": 0.0,  "oi": 0.0,
                               "greeks": {"delta": -0.95}},
                    },
                },
            },
        }

    def test_parse_oi_happy_path(self):
        """OI parser flattens `oc` to {strike: {ce_oi, pe_oi}} and drops zero-OI strikes."""
        parsed = master_file.OpeningStrikePCRVWAPATRWorker._parse_option_chain_for_oi(
            self._chain_resp()
        )
        self.assertIn(22000.0, parsed)
        self.assertIn(22500.0, parsed)
        # 23000 has zero OI on both legs -> dropped.
        self.assertNotIn(23000.0, parsed)
        self.assertEqual(parsed[22000.0]["ce_oi"], 1000.0)
        self.assertEqual(parsed[22000.0]["pe_oi"], 2500.0)

    def test_parse_oi_rejects_bad_payloads(self):
        """Non-dict / failure-status / non-dict data all return {} cleanly."""
        parse = master_file.OpeningStrikePCRVWAPATRWorker._parse_option_chain_for_oi
        self.assertEqual(parse(None), {})
        self.assertEqual(parse({"status": "failure"}), {})
        self.assertEqual(parse({"status": "success", "data": "not a dict"}), {})
        # Status missing is treated as success (empty `oc`).
        self.assertEqual(parse({"data": {"oc": "not a dict"}}), {})

    def test_parse_deltas_happy_path(self):
        """Delta parser returns {strike: {ce: {delta, ltp}, pe: {delta, ltp}}}."""
        parsed = (
            master_file.Delta20HedgedSpreadWorker
            ._parse_option_chain_for_deltas(self._chain_resp())
        )
        self.assertEqual(parsed[22500.0]["ce"]["delta"], 0.20)
        self.assertEqual(parsed[22500.0]["ce"]["ltp"], 100.0)
        self.assertEqual(parsed[22500.0]["pe"]["delta"], -0.40)
        # Zero-LTP legs are dropped: 23000 PE has ltp=0 so its 'pe' key is absent.
        self.assertNotIn("pe", parsed.get(23000.0, {}))

    def test_pick_strike_by_delta_chooses_closest(self):
        """Picker returns the (strike, leg) with delta closest to the target."""
        parsed = {
            22000.0: {"ce": {"delta": 0.74, "ltp": 481.7}},
            22500.0: {"ce": {"delta": 0.20, "ltp": 100.0}},
            23000.0: {"ce": {"delta": 0.05, "ltp": 5.0}},
        }
        pick = master_file.Delta20HedgedSpreadWorker._pick_strike_by_delta(
            parsed, target_delta=0.20, right="ce"
        )
        self.assertEqual(pick[0], 22500.0)
        self.assertEqual(pick[1]["ltp"], 100.0)

    def test_pick_strike_by_delta_rejects_bad_right(self):
        """An invalid `right` returns None cleanly."""
        self.assertIsNone(
            master_file.Delta20HedgedSpreadWorker._pick_strike_by_delta(
                {}, target_delta=0.2, right="bogus"
            )
        )

    def test_pick_strike_by_delta_empty_chain(self):
        """No candidates -> None."""
        self.assertIsNone(
            master_file.Delta20HedgedSpreadWorker._pick_strike_by_delta(
                {}, target_delta=0.2, right="ce"
            )
        )


class TestOpeningStrikeEntryAcknowledgement(unittest.TestCase):
    """The one-shot setup belongs to a successful entry, not an emitted signal."""

    def setUp(self):
        self.worker = master_file.OpeningStrikePCRVWAPATRWorker(
            master_file.SharedMarketDataStore(),
            threading.Event(),
            MagicMock(),
        )
        self.worker.signal_engine = MagicMock()
        self.worker.signal_engine._entry_signal_sent = False
        self.worker.signal_engine.evaluate.return_value = (
            master_file.OPENING_STRIKE_LOGIC.NiftyOpeningStrikePCRVWAPATRDecision(
                action="BUY_CALL",
                signal_triggered=True,
                entry_underlying=25000.0,
            )
        )
        self.worker._build_option_chain_oi_change = MagicMock(
            return_value=pd.DataFrame({"strike": [25000.0]})
        )
        self.frame = pd.DataFrame(
            {
                "timestamp": [datetime(2026, 7, 16, 10, 0)],
                "open": [24990.0],
                "close": [25000.0],
            }
        )

    def test_failed_entry_does_not_consume_one_shot_signal(self):
        self.worker.enter_position = MagicMock(return_value=False)

        self.worker.process_strategy_frame(self.frame)

        self.worker.signal_engine.acknowledge_entry.assert_not_called()

    def test_successful_entry_consumes_one_shot_signal(self):
        self.worker.enter_position = MagicMock(return_value=True)

        self.worker.process_strategy_frame(self.frame)

        self.worker.signal_engine.acknowledge_entry.assert_called_once_with()


# =============================================================================
# TEST SUITE: OPTIONS CONTRACT RESOLVER
# =============================================================================
class TestOptionsContractResolver(unittest.TestCase):
    """
    Resolver tests use a small synthetic instrument-master CSV in a temp dir.
    The CSV's required columns (EXCH_ID, SEGMENT, INSTRUMENT, SYMBOL_NAME,
    SM_EXPIRY_DATE, SECURITY_ID, STRIKE_PRICE, OPTION_TYPE) match the real
    DhanHQ master schema.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls.tmpdir.name) / "all_instrument 1.csv"
        # Use far-future expiries so the resolver's `expiry >= today` filter
        # always passes regardless of when the test is run.
        cls.exp1 = (date.today() + timedelta(days=7)).isoformat()
        cls.exp2 = (date.today() + timedelta(days=14)).isoformat()
        cls.exp3 = (date.today() + timedelta(days=21)).isoformat()
        rows = []
        sec = 10000
        for exp in (cls.exp1, cls.exp2, cls.exp3):
            for strike in (22000, 22500, 23000, 23500):
                for right in ("CE", "PE"):
                    rows.append({
                        "EXCH_ID": "NSE",
                        "SEGMENT": "D",
                        "INSTRUMENT": "OPTIDX",
                        "SYMBOL_NAME": f"NIFTY-{exp}-{strike}-{right}",
                        "DISPLAY_NAME": f"NIFTY {exp} {strike} {right}",
                        "SM_EXPIRY_DATE": exp,
                        "LOT_SIZE": "50",
                        "SECURITY_ID": str(sec),
                        "STRIKE_PRICE": str(strike),
                        "OPTION_TYPE": right,
                        "UNDERLYING_SYMBOL": "NIFTY",
                    })
                    sec += 1
        pd.DataFrame(rows).to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _make_resolver(self):
        import logging
        log = logging.getLogger("test_resolver")
        glob_pattern = str(Path(self.tmpdir.name) / "all_instrument *.csv")
        return master_file.OptionsContractResolver(
            underlying="NIFTY",
            instrument_master_glob=glob_pattern,
            log=log,
        )

    def test_get_target_expiry_returns_second_expiry(self):
        """next-next expiry = the second future expiry."""
        resolver = self._make_resolver()
        self.assertEqual(resolver.get_target_expiry().isoformat(), self.exp2)

    def test_get_current_week_expiry_returns_first_expiry(self):
        """Current-week expiry = the first future expiry."""
        resolver = self._make_resolver()
        self.assertEqual(resolver.get_current_week_expiry().isoformat(), self.exp1)

    def test_get_atm_option_long_returns_ce(self):
        """LONG -> CE at the strike nearest to spot."""
        resolver = self._make_resolver()
        contract = resolver.get_atm_option(spot_price=22510.0, direction="LONG")
        self.assertEqual(contract["option_type"], "CE")
        self.assertEqual(contract["strike"], 22500.0)
        self.assertEqual(contract["lot_size"], 50)
        self.assertEqual(contract["exchange_segment"], master_file.OPTION_EXCHANGE_SEGMENT)

    def test_get_atm_option_short_returns_pe(self):
        """SHORT -> PE at the strike nearest to spot."""
        resolver = self._make_resolver()
        contract = resolver.get_atm_option(spot_price=22510.0, direction="SHORT")
        self.assertEqual(contract["option_type"], "PE")
        self.assertEqual(contract["strike"], 22500.0)

    def test_get_atm_option_rejects_invalid_direction(self):
        """Direction must be LONG or SHORT - anything else raises."""
        resolver = self._make_resolver()
        with self.assertRaises(ValueError):
            resolver.get_atm_option(spot_price=22500.0, direction="UPWARD")

    def test_get_atm_option_rejects_invalid_spot(self):
        """Non-positive spot is rejected."""
        resolver = self._make_resolver()
        with self.assertRaises(ValueError):
            resolver.get_atm_option(spot_price=0, direction="LONG")

    def test_list_puts_for_expiry(self):
        """`list_puts_for_expiry` returns PE rows for the given expiry only."""
        resolver = self._make_resolver()
        exp = resolver.get_current_week_expiry()
        puts = resolver.list_puts_for_expiry(exp)
        self.assertEqual(len(puts), 4)  # 4 strikes
        self.assertTrue((puts["option_type"] == "PE").all())
        # Sorted ascending by strike.
        self.assertTrue(puts["strike"].is_monotonic_increasing)

    def test_pick_put_by_target_premium(self):
        """Pick the PE whose LTP is closest to the target premium."""
        resolver = self._make_resolver()
        exp = resolver.get_current_week_expiry()
        puts = resolver.list_puts_for_expiry(exp)
        ltp_map = {
            (master_file.OPTION_EXCHANGE_SEGMENT, int(row["security_id"])): ltp
            for row, ltp in zip(
                [r for _, r in puts.iterrows()],
                [10.0, 50.0, 160.0, 350.0],
                strict=False,
            )
        }
        pick = resolver.pick_put_by_target_premium(
            puts, ltp_map, target_premium=160.0
        )
        self.assertEqual(pick["option_type"], "PE")
        self.assertAlmostEqual(pick["entry_ltp"], 160.0)

    def test_pick_call_by_target_premium_excludes_used_strike(self):
        """`exclude_security_ids` prevents picking the same strike twice."""
        resolver = self._make_resolver()
        exp = resolver.get_current_week_expiry()
        calls = resolver.list_calls_for_expiry(exp)
        ltp_map = {
            (master_file.OPTION_EXCHANGE_SEGMENT, int(row["security_id"])): ltp
            for row, ltp in zip(
                [r for _, r in calls.iterrows()],
                [350.0, 160.0, 50.0, 10.0],
                strict=False,
            )
        }
        # First pick - landed at the 160-Rs strike.
        first = resolver.pick_call_by_target_premium(
            calls, ltp_map, target_premium=160.0
        )
        # Exclude that sec_id; the next-closest should win.
        second = resolver.pick_call_by_target_premium(
            calls,
            ltp_map,
            target_premium=160.0,
            exclude_security_ids={int(first["security_id"])},
        )
        self.assertNotEqual(first["security_id"], second["security_id"])

    def test_get_option_for_strike(self):
        """Exact (expiry, strike, right) lookup returns the matching row."""
        resolver = self._make_resolver()
        exp = resolver.get_current_week_expiry()
        result = resolver.get_option_for_strike(exp, 22500.0, "CE")
        self.assertIsNotNone(result)
        self.assertEqual(result["strike"], 22500.0)
        self.assertEqual(result["option_type"], "CE")

    def test_get_option_for_strike_out_of_range(self):
        """A strike far outside the listed range returns None, not a raise."""
        resolver = self._make_resolver()
        exp = resolver.get_current_week_expiry()
        self.assertIsNone(resolver.get_option_for_strike(exp, 50000.0, "CE"))


# =============================================================================
# TEST SUITE: RESOLVER PER-UNDERLYING FILTER + MIRROR EXPIRY RULE (BNF-001)
# =============================================================================
class TestOptionsContractResolverUnderlyings(unittest.TestCase):
    """
    BNF-001 regression suite. The instrument-master filter used to hardcode
    `~startswith("BANKNIFTY-")`-style exclusions regardless of the resolver's
    own underlying, so a BANKNIFTY resolver could never see a single row
    ("No valid BANKNIFTY option rows found" even though the CSV had them).
    These tests pin the fixed behaviour: each resolver sees ONLY its own
    underlying's rows, and the SL Hunting mirror's monthly-rollover expiry
    picker chooses the current expiry unless it is about to expire.
    """

    @staticmethod
    def _option_rows(underlying: str, expiries, strikes, lot_size: str, sec_start: int):
        """Build DhanHQ-master-schema rows for one underlying (CE+PE per strike)."""
        rows = []
        sec = sec_start
        for exp in expiries:
            for strike in strikes:
                for right in ("CE", "PE"):
                    rows.append({
                        "EXCH_ID": "NSE",
                        "SEGMENT": "D",
                        "INSTRUMENT": "OPTIDX",
                        "SYMBOL_NAME": f"{underlying}-{exp}-{strike}-{right}",
                        "DISPLAY_NAME": f"{underlying} {exp} {strike} {right}",
                        "SM_EXPIRY_DATE": str(exp),
                        "LOT_SIZE": lot_size,
                        "SECURITY_ID": str(sec),
                        "STRIKE_PRICE": str(strike),
                        "OPTION_TYPE": right,
                        "UNDERLYING_SYMBOL": underlying,
                    })
                    sec += 1
        return rows

    def _write_master(self, rows) -> tempfile.TemporaryDirectory:
        """Write one synthetic instrument master; caller owns the tempdir."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        pd.DataFrame(rows).to_csv(Path(tmpdir.name) / "all_instrument 1.csv", index=False)
        return tmpdir

    def _resolver(self, tmpdir, underlying: str):
        import logging
        return master_file.OptionsContractResolver(
            underlying=underlying,
            instrument_master_glob=str(Path(tmpdir.name) / "all_instrument *.csv"),
            log=logging.getLogger("test_resolver_underlyings"),
        )

    def _mixed_master(self):
        """One CSV holding NIFTY + BANKNIFTY + FINNIFTY rows over two expiries."""
        self.exp_first = date.today() + timedelta(days=10)
        self.exp_second = date.today() + timedelta(days=40)
        expiries = (self.exp_first.isoformat(), self.exp_second.isoformat())
        rows = (
            self._option_rows("NIFTY", expiries, (24200, 24300, 24400), "75", 20000)
            + self._option_rows("BANKNIFTY", expiries, (57800, 57900, 58000), "35", 30000)
            + self._option_rows("FINNIFTY", expiries, (26300, 26400), "65", 40000)
        )
        return self._write_master(rows)

    def test_banknifty_resolver_returns_banknifty_atm_contract(self):
        """A BANKNIFTY resolver must resolve a BANKNIFTY contract from a mixed master."""
        tmpdir = self._mixed_master()
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        contract = resolver.get_atm_option(spot_price=57910.0, direction="LONG")
        self.assertTrue(str(contract["trading_symbol"]).startswith("BANKNIFTY-"))
        self.assertEqual(contract["option_type"], "CE")
        self.assertEqual(contract["strike"], 57900.0)
        self.assertEqual(contract["lot_size"], 35)

    def test_banknifty_chain_contains_only_banknifty_rows(self):
        """The BANKNIFTY chain must include every BNF row and nothing else."""
        tmpdir = self._mixed_master()
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        chain = resolver._load_option_chain()
        self.assertEqual(len(chain), 12)  # 2 expiries x 3 strikes x CE/PE
        self.assertTrue(chain["trading_symbol"].str.startswith("BANKNIFTY-").all())

    def test_nifty_resolver_excludes_other_underlyings(self):
        """The NIFTY chain must still exclude BANKNIFTY/FINNIFTY rows (regression lock)."""
        tmpdir = self._mixed_master()
        resolver = self._resolver(tmpdir, "NIFTY")
        chain = resolver._load_option_chain()
        self.assertEqual(len(chain), 12)  # 2 expiries x 3 strikes x CE/PE
        self.assertTrue(chain["trading_symbol"].str.startswith("NIFTY-").all())

    def test_get_atm_option_accepts_explicit_expiry(self):
        """An explicit expiry overrides the default next-next rule (mirror wiring)."""
        tmpdir = self._mixed_master()
        resolver = self._resolver(tmpdir, "NIFTY")
        contract = resolver.get_atm_option(
            spot_price=24310.0, direction="LONG", expiry_date=self.exp_first
        )
        self.assertEqual(contract["expiry_date"], self.exp_first)
        # And the default still lands on the next-next expiry.
        default_contract = resolver.get_atm_option(spot_price=24310.0, direction="LONG")
        self.assertEqual(default_contract["expiry_date"], self.exp_second)

    def _bnf_two_expiry_master(self, days_to_first: int, days_to_second: int = 40):
        self.exp_near = date.today() + timedelta(days=days_to_first)
        self.exp_far = date.today() + timedelta(days=days_to_second)
        rows = self._option_rows(
            "BANKNIFTY",
            (self.exp_near.isoformat(), self.exp_far.isoformat()),
            (57900,),
            "35",
            30000,
        )
        return self._write_master(rows)

    def test_rollover_keeps_current_expiry_when_far(self):
        """>= min_days to the current expiry -> stay on the current expiry."""
        tmpdir = self._bnf_two_expiry_master(days_to_first=10)
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        self.assertEqual(resolver.get_monthly_rollover_expiry(7), self.exp_near)

    def test_rollover_rolls_to_next_expiry_when_near(self):
        """< min_days to the current expiry -> roll to the next expiry."""
        tmpdir = self._bnf_two_expiry_master(days_to_first=3)
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        self.assertEqual(resolver.get_monthly_rollover_expiry(7), self.exp_far)

    def test_rollover_boundary_day_keeps_current_expiry(self):
        """Exactly min_days away is NOT 'fewer than' -> keep the current expiry."""
        tmpdir = self._bnf_two_expiry_master(days_to_first=7)
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        self.assertEqual(resolver.get_monthly_rollover_expiry(7), self.exp_near)

    def test_rollover_returns_only_expiry_when_single(self):
        """With a single listed expiry there is nothing to roll to -> return it."""
        self.exp_only = date.today() + timedelta(days=3)
        rows = self._option_rows("BANKNIFTY", (self.exp_only.isoformat(),), (57900,), "35", 30000)
        tmpdir = self._write_master(rows)
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        self.assertEqual(resolver.get_monthly_rollover_expiry(7), self.exp_only)

    def test_banknifty_atm_uses_100_point_strike_step(self):
        """BankNIFTY strikes are 100-point. A 57,960 spot must resolve to the
        nearest listed 58,000 -- the old global 50-point seed rounded to 57,950,
        which ties 57,900/58,000 and the lower-strike tie-break wrongly bought
        57,900 (Codex P2 on PR #40)."""
        tmpdir = self._mixed_master()  # BANKNIFTY strikes 57800/57900/58000
        resolver = self._resolver(tmpdir, "BANKNIFTY")
        contract = resolver.get_atm_option(spot_price=57960.0, direction="LONG")
        self.assertEqual(contract["strike"], 58000.0)
        self.assertEqual(contract["atm_strike_rounded"], 58000.0)

    def test_nifty_atm_keeps_50_point_strike_step(self):
        """NIFTY must be untouched: a 50-point step still seeds the ATM strike,
        so a spot 40 points above a strike rounds up to it."""
        exp = (date.today() + timedelta(days=10)).isoformat()
        rows = self._option_rows("NIFTY", (exp,), (24950, 25000, 25050), "75", 20000)
        tmpdir = self._write_master(rows)
        resolver = self._resolver(tmpdir, "NIFTY")
        # 24,990 rounds to 25,000 on a 50-step; a 100-step would wrongly seed 25,000
        # too here, so use 24,940 -> 50-step seeds 24,950 (its nearest listed strike).
        contract = resolver.get_atm_option(spot_price=24940.0, direction="LONG", expiry_date=date.fromisoformat(exp))
        self.assertEqual(contract["strike"], 24950.0)


# =============================================================================
# TEST SUITE: DHAN BROKER CLIENT (mocked dhanhq SDK)
# =============================================================================
class TestDhanBrokerClient(unittest.TestCase):
    """
    Tests the thin wrapper around the `dhanhq` SDK. We don't hit a real
    network; the underlying SDK methods (`intraday_minute_data`, `ticker_data`,
    `option_chain`) are replaced with MagicMocks that return canned dicts.
    """

    def _make_broker(self):
        broker = master_file.DhanBrokerClient.__new__(master_file.DhanBrokerClient)
        broker.dhan = MagicMock()
        broker._dhan_context = MagicMock()
        return broker

    def test_fetch_index_1m_ohlc_normalizes_response(self):
        """The wrapper delegates to `normalize_dhan_intraday_response`."""
        broker = self._make_broker()
        base_ts = int(datetime(2026, 5, 15, 9, 15).timestamp())
        n = master_file.MIN_BARS + 5
        broker.dhan.intraday_minute_data.return_value = {
            "status": "success",
            "data": {
                "timestamp": [base_ts + i * 60 for i in range(n)],
                "open":   [100.0 + i for i in range(n)],
                "high":   [101.0 + i for i in range(n)],
                "low":    [99.0 + i for i in range(n)],
                "close":  [100.5 + i for i in range(n)],
            },
        }
        out = broker.fetch_index_1m_ohlc(
            security_id=13, exchange_segment="IDX_I", instrument_type="INDEX",
            lookback_days=2,
        )
        self.assertGreaterEqual(len(out), master_file.MIN_BARS)
        broker.dhan.intraday_minute_data.assert_called_once()

    def test_fetch_ltp_map_flattens_response(self):
        """The wrapper flattens nested dict shape into (segment, sec_id) -> price."""
        broker = self._make_broker()
        broker.dhan.ticker_data.return_value = {
            "status": "success",
            "data": {
                "NSE_FNO": {"49081": {"last_price": 150.5}},
                "IDX_I":   {"13":    {"last_price": 22500.0}},
            },
        }
        result = broker.fetch_ltp_map({
            "NSE_FNO": [49081],
            "IDX_I":   [13],
        })
        self.assertEqual(result[("NSE_FNO", 49081)], 150.5)
        self.assertEqual(result[("IDX_I", 13)], 22500.0)

    def test_fetch_ltp_map_unwraps_double_nested_data(self):
        """DhanHQ sometimes wraps response in {data: {data: {...}}}; unwrap it."""
        broker = self._make_broker()
        broker.dhan.ticker_data.return_value = {
            "status": "success",
            "data": {
                "data": {
                    "NSE_FNO": {"49081": {"last_price": 150.5}},
                },
            },
        }
        result = broker.fetch_ltp_map({"NSE_FNO": [49081]})
        self.assertEqual(result[("NSE_FNO", 49081)], 150.5)

    def test_fetch_ltp_map_drops_negative_and_zero_prices(self):
        """Non-positive prices are silently dropped (data quality guard)."""
        broker = self._make_broker()
        broker.dhan.ticker_data.return_value = {
            "status": "success",
            "data": {
                "NSE_FNO": {
                    "49081": {"last_price": 150.5},
                    "49082": {"last_price": 0.0},
                    "49083": {"last_price": -10.0},
                },
            },
        }
        result = broker.fetch_ltp_map({"NSE_FNO": [49081, 49082, 49083]})
        self.assertIn(("NSE_FNO", 49081), result)
        self.assertNotIn(("NSE_FNO", 49082), result)
        self.assertNotIn(("NSE_FNO", 49083), result)

    def test_fetch_ltp_map_empty_request_short_circuits(self):
        """No ids in request -> no API call, empty result."""
        broker = self._make_broker()
        result = broker.fetch_ltp_map({"NSE_FNO": []})
        self.assertEqual(result, {})
        broker.dhan.ticker_data.assert_not_called()

    def test_fetch_ltp_map_failure_status_returns_empty(self):
        """A failure-status response is treated as 'no data' rather than raising."""
        broker = self._make_broker()
        broker.dhan.ticker_data.return_value = {"status": "failure"}
        self.assertEqual(broker.fetch_ltp_map({"NSE_FNO": [49081]}), {})

    def test_fetch_option_chain_unwraps_envelope(self):
        """SDK wraps {status, data: <api_response>}; wrapper returns the inner data."""
        broker = self._make_broker()
        inner = {"status": "success", "data": {"last_price": 22500.0, "oc": {}}}
        broker.dhan.option_chain.return_value = {"status": "success", "data": inner}
        out = broker.fetch_option_chain(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            expiry=date.today() + timedelta(days=7),
        )
        # The wrapper returns the inner dict (one level peeled).
        self.assertEqual(out, inner)


# =============================================================================
# TEST SUITE: CENTRAL MARKET DATA FETCHER
# =============================================================================
class TestCentralMarketDataFetcher(unittest.TestCase):
    """
    Fetcher tests use a mocked broker so no real API calls are made. We
    verify that the fetcher builds the right LTP-request batch and pushes
    snapshots into the store correctly.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.fetcher = master_file.CentralMarketDataFetcher(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )

    def test_fetch_ohlc_rejects_non_1min_timeframe(self):
        """Source data is always 1-min; higher TFs are derived per-worker."""
        with self.assertRaises(ValueError):
            self.fetcher.fetch_ohlc("5")

    def test_fetch_ohlc_delegates_to_broker(self):
        """`fetch_ohlc('1')` calls `broker.fetch_index_1m_ohlc` once."""
        expected_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-05-15 09:15", periods=1, freq="1min"),
             "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]}
        )
        self.broker.fetch_index_1m_ohlc.return_value = expected_df
        result = self.fetcher.fetch_ohlc("1")
        self.broker.fetch_index_1m_ohlc.assert_called_once()
        self.assertTrue(result.equals(expected_df))

    def test_refresh_index_and_option_ltps_includes_subscriptions(self):
        """All subscribed option sec_ids are batched into one ticker_data call."""
        sub_ce = master_file.OptionSubscription(
            security_id=49081, exchange_segment="NSE_FNO",
            trading_symbol="OPT_CE", right="CE", strike=22500.0,
            expiry=date.today() + timedelta(days=7),
        )
        sub_pe = master_file.OptionSubscription(
            security_id=49082, exchange_segment="NSE_FNO",
            trading_symbol="OPT_PE", right="PE", strike=22500.0,
            expiry=date.today() + timedelta(days=7),
        )
        self.store.register_option_subscription(sub_ce)
        self.store.register_option_subscription(sub_pe)
        self.broker.fetch_ltp_map.return_value = {
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
             master_file.NIFTY_INDEX_SECURITY_ID): 22500.0,
            ("NSE_FNO", 49081): 100.0,
            ("NSE_FNO", 49082): 80.0,
        }
        self.fetcher.refresh_index_and_option_ltps()

        # The single call should include both subscribed legs PLUS the index.
        args, _ = self.broker.fetch_ltp_map.call_args
        request = args[0]
        self.assertIn(master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, request)
        self.assertIn("NSE_FNO", request)
        self.assertCountEqual(request["NSE_FNO"], [49081, 49082])

        # Returned LTPs were pushed into the store.
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 49081), 100.0)
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 49082), 80.0)

    def test_refresh_swallows_broker_exceptions(self):
        """A broker failure must not propagate or kill the fetcher loop."""
        self.broker.fetch_ltp_map.side_effect = RuntimeError("network down")
        try:
            self.fetcher.refresh_index_and_option_ltps()
        except RuntimeError:
            self.fail("refresh_index_and_option_ltps should swallow broker errors")


# =============================================================================
# TEST SUITE: MARKET DATA SOURCE SELECTOR
# =============================================================================
class TestMarketDataSourceSelector(unittest.TestCase):
    """
    The MARKET_DATA_SOURCE env flag picks the producer class. Anything that is
    not exactly "WEBSOCKET" must FAIL CLOSED to the battle-tested REST poller.
    """

    def test_rest_selects_central_fetcher(self):
        with patch.object(master_file, "MARKET_DATA_SOURCE", "REST"):
            self.assertIs(
                master_file._select_market_data_fetcher_class(),
                master_file.CentralMarketDataFetcher,
            )

    def test_websocket_selects_ws_fetcher(self):
        with patch.object(master_file, "MARKET_DATA_SOURCE", "WEBSOCKET"):
            self.assertIs(
                master_file._select_market_data_fetcher_class(),
                master_file.WebSocketMarketDataFetcher,
            )

    def test_unknown_value_fails_closed_to_rest(self):
        for bad_value in ("WEBSOKET", "ws", "", "TICKS"):
            with patch.object(master_file, "MARKET_DATA_SOURCE", bad_value):
                self.assertIs(
                    master_file._select_market_data_fetcher_class(),
                    master_file.CentralMarketDataFetcher,
                    bad_value,
                )


# =============================================================================
# TEST SUITE: STORE LTP FRESHNESS TOUCH (websocket health support)
# =============================================================================
class TestSharedMarketDataStoreLtpFreshnessTouch(unittest.TestCase):
    """
    `touch_ltp_freshness` re-stamps cached LTPs as fresh while the websocket
    connection is alive. It must only ever touch keys that already hold a
    positive price -- it can never invent a price for an unknown instrument.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()

    def test_restamps_existing_positive_key(self):
        key = ("NSE_FNO", 111)
        self.store.update_ltp_map({key: 55.5})
        stale = self.store._ltp_snapshots[key].fetched_at - timedelta(seconds=60)
        self.store._ltp_snapshots[key].fetched_at = stale
        self.store.touch_ltp_freshness({key})
        self.assertGreater(self.store._ltp_snapshots[key].fetched_at, stale)
        # The price itself must be untouched.
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 111), 55.5)

    def test_never_creates_missing_keys(self):
        self.store.touch_ltp_freshness({("NSE_FNO", 999)})
        self.assertNotIn(("NSE_FNO", 999), self.store._ltp_snapshots)
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 999, fallback=0.0), 0.0)


# =============================================================================
# TEST SUITE: WEBSOCKET MARKET DATA FETCHER
# =============================================================================
class TestWebSocketMarketDataFetcher(unittest.TestCase):
    """
    Websocket producer tests drive the fetcher's internals directly with fake
    marketfeed packets and a MagicMock broker -- no threads, no sockets. The
    wall clock is injected (`now_ist=...`) so results do not depend on when
    the suite runs.
    """

    # A mid-session instant matching the packets below (naive IST).
    NOW = datetime(2026, 5, 15, 10, 17, 34)

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.fetcher = master_file.WebSocketMarketDataFetcher(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        self.index_key = (
            master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
            master_file.NIFTY_INDEX_SECURITY_ID,
        )

    def _index_tick(self, ltp="24238.50", ltt="10:17:33"):
        """Ticker packet in the exact shape dhanhq 2.2.0 emits for NIFTY."""
        return {
            "type": "Ticker Data",
            "exchange_segment": 0,
            "security_id": master_file.NIFTY_INDEX_SECURITY_ID,
            "LTP": ltp,
            "LTT": ltt,
        }

    def _option_tick(self, security_id=49081, ltp="139.45", ltt="10:17:33"):
        return {
            "type": "Ticker Data",
            "exchange_segment": 2,
            "security_id": security_id,
            "LTP": ltp,
            "LTT": ltt,
        }

    def _register_option(self, security_id=49081):
        self.store.register_option_subscription(
            master_file.OptionSubscription(
                security_id=security_id, exchange_segment="NSE_FNO",
                trading_symbol="OPT_CE", right="CE", strike=22500.0,
                expiry=date.today() + timedelta(days=7),
            )
        )

    def test_thread_name_matches_rest_fetcher(self):
        """Log/EOD tooling keys on the thread name; both producers share it."""
        self.assertEqual(self.fetcher.name, "MarketDataFetcher")

    def test_handle_packet_updates_ltp_cache_and_confirms(self):
        self.fetcher._handle_packet(self._index_tick(), now_ist=self.NOW)
        self.assertEqual(
            self.store.get_ltp_by_secid(*self.index_key), 24238.50
        )
        self.assertIn(self.index_key, self.fetcher._confirmed_keys)
        self.assertIsNotNone(self.fetcher._last_packet_monotonic)

    def test_previous_close_confirms_without_price(self):
        packet = {
            "type": "Previous Close",
            "exchange_segment": 2,
            "security_id": 49081,
            "prev_close": "216.95",
            "prev_OI": 0,
        }
        self.fetcher._handle_packet(packet, now_ist=self.NOW)
        self.assertIn(("NSE_FNO", 49081), self.fetcher._confirmed_keys)
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 49081), 0.0)

    def test_index_ticks_build_bars_and_published_frame_validates(self):
        self.fetcher._handle_packet(self._index_tick("24238.50", "10:17:33"), now_ist=self.NOW)
        self.fetcher._publish_frame_if_changed()
        snapshot = self.store.get("1")
        self.assertIsNotNone(snapshot)
        # The forming minute must be present as the last row.
        self.assertEqual(
            snapshot.frame.iloc[-1]["timestamp"], pd.Timestamp("2026-05-15 10:17:00")
        )
        self.assertEqual(snapshot.frame.iloc[-1]["close"], 24238.50)

    def test_option_ticks_never_become_bars(self):
        self.fetcher._handle_packet(self._option_tick(), now_ist=self.NOW)
        self.assertTrue(self.fetcher.aggregator.tick_bars_frame().empty)
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 49081), 139.45)

    def test_stale_snapshot_tick_feeds_ltp_but_not_bars(self):
        """The subscribe-replay tick carries an old LTT; cache it, never bar it."""
        self.fetcher._handle_packet(
            self._index_tick(ltp="24334.30", ltt="15:29:59"), now_ist=self.NOW
        )
        self.assertEqual(self.store.get_ltp_by_secid(*self.index_key), 24334.30)
        self.assertTrue(self.fetcher.aggregator.tick_bars_frame().empty)

    def test_minute_rollover_publishes_through_throttle(self):
        self.fetcher._handle_packet(self._index_tick("24238.50", "10:17:33"), now_ist=self.NOW)
        self.fetcher._publish_frame_if_changed()
        self.assertEqual(len(self.store.get("1").frame), 1)

        # Same-minute update inside the throttle window: not republished.
        self.fetcher._last_publish_monotonic = time.monotonic()
        self.fetcher._handle_packet(self._index_tick("24240.00", "10:17:35"),
                                    now_ist=datetime(2026, 5, 15, 10, 17, 36))
        self.fetcher._publish_frame_if_changed()
        self.assertEqual(self.store.get("1").frame.iloc[-1]["close"], 24238.50)

        # A minute rollover must bypass the throttle and publish immediately.
        self.fetcher._handle_packet(self._index_tick("24241.00", "10:18:01"),
                                    now_ist=datetime(2026, 5, 15, 10, 18, 2))
        self.fetcher._publish_frame_if_changed()
        frame = self.store.get("1").frame
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[-1]["timestamp"], pd.Timestamp("2026-05-15 10:18:00"))

    def test_desired_instruments_cover_index_and_subscriptions(self):
        self._register_option(49081)
        desired = self.fetcher._desired_instruments()
        self.assertIn(self.index_key, desired)
        self.assertIn(("NSE_FNO", 49081), desired)
        # Feed tuples use marketfeed codes and STRING security ids.
        self.assertEqual(desired[("NSE_FNO", 49081)][0], 2)
        self.assertEqual(desired[("NSE_FNO", 49081)][1], "49081")

    def test_sync_subscriptions_adds_removes_and_protects_index(self):
        feed = MagicMock()
        self.fetcher._feed = feed
        self.fetcher._subscribed_keys = {self.index_key}

        self._register_option(49081)
        self.fetcher._sync_subscriptions()
        feed.subscribe_symbols.assert_called_once()
        added = feed.subscribe_symbols.call_args[0][0]
        self.assertEqual([(t[0], t[1]) for t in added], [(2, "49081")])

        self.store.unregister_option_subscription("NSE_FNO", 49081)
        self.fetcher._sync_subscriptions()
        feed.unsubscribe_symbols.assert_called_once()
        removed = feed.unsubscribe_symbols.call_args[0][0]
        self.assertEqual([(t[0], t[1]) for t in removed], [(2, "49081")])
        # The index leg must never be unsubscribed.
        for call in feed.unsubscribe_symbols.call_args_list:
            for entry in call[0][0]:
                self.assertNotEqual(entry[1], str(master_file.NIFTY_INDEX_SECURITY_ID))

    def test_sync_subscriptions_skips_unknown_segment(self):
        feed = MagicMock()
        self.fetcher._feed = feed
        self.fetcher._subscribed_keys = {self.index_key}
        self.store.register_option_subscription(
            master_file.OptionSubscription(
                security_id=777, exchange_segment="MCX_WEIRD",
                trading_symbol="ODD", right="CE", strike=1.0, expiry=None,
            )
        )
        self.fetcher._sync_subscriptions()
        feed.subscribe_symbols.assert_not_called()

    def test_true_up_overwrites_completed_bar_keeps_forming(self):
        completed = pd.Timestamp("2026-05-15 10:16:00")
        forming = pd.Timestamp("2026-05-15 10:17:00")
        self.fetcher.aggregator.add_tick(completed, 100.2)
        self.fetcher.aggregator.add_tick(completed, 100.4)
        self.fetcher.aggregator.add_tick(forming, 100.6)
        self.broker.fetch_index_1m_ohlc.return_value = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-15 10:15:00"), completed],
                "open": [99.8, 100.0], "high": [100.4, 101.0],
                "low": [99.6, 99.5], "close": [100.1, 100.5],
            }
        )
        self.fetcher._run_true_up("test", now_ist=datetime(2026, 5, 15, 10, 17, 40))
        frame = self.store.get("1").frame
        self.assertEqual(len(frame), 3)
        by_ts = frame.set_index("timestamp")
        # Completed minute now carries the OFFICIAL candle...
        self.assertEqual(by_ts.loc[completed]["open"], 100.0)
        self.assertEqual(by_ts.loc[completed]["close"], 100.5)
        # ...while the forming minute keeps its tick-built values.
        self.assertEqual(by_ts.loc[forming]["close"], 100.6)

    def test_true_up_prunes_trued_minutes_so_divergence_is_per_cycle(self):
        """Once official candles cover a minute, its tick bar must leave the
        aggregator -- otherwise every later true-up re-reports the same old
        mismatches forever (observed in the 2026-07-21 paper session)."""
        completed = pd.Timestamp("2026-05-15 10:16:00")
        forming = pd.Timestamp("2026-05-15 10:17:00")
        self.fetcher.aggregator.add_tick(completed, 100.2)
        self.fetcher.aggregator.add_tick(forming, 100.6)
        self.broker.fetch_index_1m_ohlc.return_value = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-15 10:15:00"), completed],
                "open": [99.8, 100.0], "high": [100.4, 101.0],
                "low": [99.6, 99.5], "close": [100.1, 100.5],
            }
        )
        self.fetcher._run_true_up("test", now_ist=datetime(2026, 5, 15, 10, 17, 40))
        # Only the still-tick-owned forming minute survives in the aggregator.
        remaining = self.fetcher.aggregator.tick_bars_frame()
        self.assertEqual(list(remaining["timestamp"]), [forming])
        # The published merge is unaffected: official rows + the forming bar.
        self.assertEqual(len(self.store.get("1").frame), 3)

    def test_true_up_rest_failure_keeps_tick_bars(self):
        forming = pd.Timestamp("2026-05-15 10:17:00")
        self.fetcher.aggregator.add_tick(forming, 100.6)
        self.fetcher._publish_frame_if_changed()
        self.broker.fetch_index_1m_ohlc.side_effect = RuntimeError("REST down")
        try:
            self.fetcher._run_true_up("test", now_ist=datetime(2026, 5, 15, 10, 17, 40))
        except RuntimeError:
            self.fail("_run_true_up must swallow REST errors and keep tick bars")
        frame = self.store.get("1").frame
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["close"], 100.6)

    def test_health_touches_only_confirmed_keys_while_alive(self):
        self._register_option(49081)
        self.store.touch_ltp_freshness = MagicMock()
        self.fetcher._confirmed_keys = {self.index_key}
        self.fetcher._last_packet_monotonic = time.monotonic()
        self.fetcher._record_health_if_due()
        self.store.touch_ltp_freshness.assert_called_once_with({self.index_key})

    def test_health_never_touches_when_socket_silent(self):
        self._register_option(49081)
        self.store.touch_ltp_freshness = MagicMock()
        self.fetcher._confirmed_keys = {self.index_key, ("NSE_FNO", 49081)}
        self.fetcher._last_packet_monotonic = (
            time.monotonic() - master_file.WS_CONN_LIVENESS_SECONDS - 5.0
        )
        self.fetcher._record_health_if_due()
        self.store.touch_ltp_freshness.assert_not_called()

    def test_health_publishes_required_keys_on_cadence(self):
        self._register_option(49081)
        self.store.record_market_data_refresh = MagicMock(
            return_value=MagicMock(reasons=[])
        )
        self.fetcher._record_health_if_due()
        self.fetcher._record_health_if_due()  # Inside the cadence window: skipped.
        self.assertEqual(self.store.record_market_data_refresh.call_count, 1)
        kwargs = self.store.record_market_data_refresh.call_args[1]
        self.assertEqual(
            kwargs["required_ltp_keys"], {self.index_key, ("NSE_FNO", 49081)}
        )

    def test_warmup_retries_until_success(self):
        good = pd.DataFrame(
            {"timestamp": [pd.Timestamp("2026-05-15 10:16:00")],
             "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]}
        )
        self.broker.fetch_index_1m_ohlc.side_effect = [RuntimeError("boom"), good]
        self.fetcher.WARMUP_RETRY_SECONDS = 0.01
        self.assertTrue(self.fetcher._warmup_official_history())
        self.assertEqual(len(self.fetcher.official_frame), 1)
        self.assertIsNotNone(self.store.get("1"))

    def test_pump_rebuilds_fresh_feed_with_full_desired_set(self):
        self._register_option(49081)
        feed_one = MagicMock()
        feed_one.get_data.side_effect = RuntimeError("socket died")
        feed_two = MagicMock()

        def _stop_then_raise():
            self.stop_event.set()
            raise RuntimeError("shutdown")

        feed_two.get_data.side_effect = _stop_then_raise
        self.broker.make_market_feed.side_effect = [feed_one, feed_two]
        self.fetcher.RECONNECT_BACKOFF_INITIAL_SECONDS = 0.01
        self.fetcher._pump_main()

        self.assertEqual(self.broker.make_market_feed.call_count, 2)
        for call in self.broker.make_market_feed.call_args_list:
            instruments = call[0][0]
            ids = {entry[1] for entry in instruments}
            self.assertIn(str(master_file.NIFTY_INDEX_SECURITY_ID), ids)
            self.assertIn("49081", ids)
        # A (re)connect requests an immediate true-up (the gap backfill).
        self.assertIsNotNone(self.fetcher._trueup_reason)

    def test_close_feed_swallows_sdk_errors(self):
        feed = MagicMock()
        feed.close_connection.side_effect = RuntimeError("loop not running")
        self.fetcher._feed = feed
        try:
            self.fetcher._close_feed()
        except RuntimeError:
            self.fail("_close_feed must swallow SDK shutdown errors")


# =============================================================================
# TEST SUITE: DHANHQ SDK DEPRECATION-WARNING FILTER
# =============================================================================
class TestDhanhqDeprecationWarningFilter(unittest.TestCase):
    """Loading the master must silence dhanhq 2.2.0's per-tick
    `utcfromtimestamp()` DeprecationWarning -- and ONLY that warning.

    The filter is scoped by message AND module so a deprecation raised by our
    own code (or any other dependency) still reaches the operator's console.
    """

    def test_scoped_ignore_filter_is_installed_at_import(self):
        matching = [
            entry
            for entry in warnings.filters
            if entry[0] == "ignore"
            and entry[2] is DeprecationWarning
            and entry[1] is not None
            and "utcfromtimestamp" in entry[1].pattern
        ]
        self.assertTrue(
            matching,
            "master import must install the dhanhq marketfeed warning filter",
        )
        for entry in matching:
            self.assertIsNotNone(entry[3], "filter must be module-scoped")
            self.assertIn("dhanhq", entry[3].pattern)

    def test_filter_suppresses_only_the_sdk_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            # `simplefilter` wipes the global list inside this context, so
            # re-install the master's filter in front of an "always" baseline
            # -- the same precedence it has in the real process.
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore",
                message=r"datetime\.datetime\.utcfromtimestamp\(\) is deprecated",
                category=DeprecationWarning,
                module=r"dhanhq\.marketfeed",
            )
            sdk_message = (
                "datetime.datetime.utcfromtimestamp() is deprecated and "
                "scheduled for removal in a future version. Use timezone-aware "
                "objects to represent datetimes in UTC: "
                "datetime.datetime.fromtimestamp(timestamp, datetime.UTC)."
            )
            # The exact warning the SDK's utc_time helper triggers ...
            warnings.warn_explicit(
                sdk_message,
                DeprecationWarning,
                filename="dhanhq/marketfeed.py",
                lineno=523,
                module="dhanhq.marketfeed",
            )
            # ... the same message from ANY other module must stay visible ...
            warnings.warn_explicit(
                sdk_message,
                DeprecationWarning,
                filename="somewhere/else.py",
                lineno=1,
                module="somewhere.else",
            )
            # ... and a different deprecation from the SDK module must too.
            warnings.warn_explicit(
                "some other deprecation",
                DeprecationWarning,
                filename="dhanhq/marketfeed.py",
                lineno=1,
                module="dhanhq.marketfeed",
            )
        messages = [str(item.message) for item in caught]
        self.assertEqual(len(messages), 2, messages)
        self.assertIn(sdk_message, messages)
        self.assertIn("some other deprecation", messages)


# =============================================================================
# TEST SUITE: ADDITIONAL DATACLASS COVERAGE
# =============================================================================
class TestAdditionalDataclasses(unittest.TestCase):
    """Constructs the dataclasses that weren't tested by `TestDataclasses`."""

    def test_market_snapshot(self):
        """MarketSnapshot holds timeframe, frame, candle ts, signature, and fetched_at."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-05-15 09:15", periods=1, freq="1min"),
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
        })
        snap = master_file.MarketSnapshot(
            timeframe="1",
            frame=df,
            source_candle_ts=df["timestamp"].iloc[-1],
            candle_signature=master_file.build_last_row_signature(df),
            fetched_at=datetime.now(),
        )
        self.assertEqual(snap.timeframe, "1")
        self.assertEqual(snap.frame.iloc[-1]["close"], 100.5)
        self.assertIsNotNone(snap.candle_signature)

    def test_ltp_snapshot(self):
        """LTPSnapshot identifies a leg and its latest price + fetched time."""
        snap = master_file.LTPSnapshot(
            segment="NSE_FNO",
            security_id=49081,
            ltp=150.5,
            fetched_at=datetime.now(),
        )
        self.assertEqual(snap.ltp, 150.5)
        self.assertEqual(snap.security_id, 49081)

    def test_option_subscription(self):
        """OptionSubscription carries everything needed to identify and refresh a leg."""
        sub = master_file.OptionSubscription(
            security_id=49081,
            exchange_segment="NSE_FNO",
            trading_symbol="NIFTY-22500-CE",
            right="CE",
            strike=22500.0,
            expiry=date(2026, 5, 22),
        )
        self.assertEqual(sub.security_id, 49081)
        self.assertEqual(sub.right, "CE")
        self.assertEqual(sub.strike, 22500.0)


# =============================================================================
# TEST SUITE: BASE PAPER STRATEGY WORKER
# =============================================================================
class TestBasePaperStrategyWorker(unittest.TestCase):
    """
    Base-class behaviour (LTP lookups, max-loss gate, cutoff handlers).
    We instantiate `AtmSingleLegStrategyWorker` because the base is abstract;
    the methods under test all live on the base.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        # AtmSingleLegStrategyWorker inherits everything from the base and
        # provides concrete `_get_open_position_pnl` / `exit_position`.
        self.worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        self.worker.max_loss = 1000.0

    def test_get_option_ltp_prefers_cache(self):
        """If the cache has a positive price, no broker call is made."""
        self.store.update_ltp_map({("NSE_FNO", 49081): 150.5})
        price = self.worker._get_option_ltp("NSE_FNO", 49081, fallback=99.0)
        self.assertEqual(price, 150.5)
        self.broker.fetch_ltp_map.assert_not_called()

    def test_get_option_ltp_falls_back_to_broker(self):
        """Cold cache -> direct broker fetch -> warm up the cache."""
        self.broker.fetch_ltp_map.return_value = {("NSE_FNO", 49081): 175.0}
        price = self.worker._get_option_ltp("NSE_FNO", 49081, fallback=99.0)
        self.assertEqual(price, 175.0)
        # Direct hit should also populate the cache for next time.
        self.assertEqual(self.store.get_ltp_by_secid("NSE_FNO", 49081), 175.0)

    def test_get_option_ltp_returns_fallback_on_broker_error(self):
        """Broker exception is caught - fallback is returned instead."""
        self.broker.fetch_ltp_map.side_effect = RuntimeError("network down")
        price = self.worker._get_option_ltp("NSE_FNO", 49081, fallback=99.0)
        self.assertEqual(price, 99.0)

    def test_get_underlying_spot_uses_cache_first(self):
        """Same cache-first preference for the NIFTY index spot."""
        self.store.update_ltp_map(
            {(master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
              master_file.NIFTY_INDEX_SECURITY_ID): 22500.0}
        )
        self.assertEqual(self.worker._get_underlying_spot(fallback=0.0), 22500.0)
        self.broker.fetch_ltp_map.assert_not_called()

    def test_is_max_loss_breached_at_threshold(self):
        """Total PnL at -max_loss triggers the breach gate."""
        self.worker.max_loss = 1000.0
        self.worker.realized_pnl = -1000.0
        breached, total, _ = self.worker.is_max_loss_breached()
        self.assertTrue(breached)
        self.assertEqual(total, -1000.0)

    def test_is_max_loss_breached_disabled_when_zero(self):
        """max_loss <= 0 disables the gate entirely."""
        self.worker.max_loss = 0.0
        self.worker.realized_pnl = -1e9
        breached, _, _ = self.worker.is_max_loss_breached()
        self.assertFalse(breached)

    def test_paper_order_id_format(self):
        """Synthetic order ids follow `PAPER-<SIDE>-<YYYYMMDDhhmmss>-<NNNN>`."""
        oid1 = self.worker._next_paper_order_id("BUY")
        oid2 = self.worker._next_paper_order_id("BUY")
        self.assertTrue(oid1.startswith("PAPER-BUY-"))
        # Counter increments per call.
        self.assertNotEqual(oid1, oid2)
        self.assertTrue(oid2.endswith("-0002"))

    def test_session_execution_mode_tracks_live_and_paper_fallbacks_without_telegram(self):
        """Mode telemetry is trading state, not a side effect of enabling Telegram."""
        self.assertEqual(self.worker.session_execution_mode(), "PAPER")

        self.worker.live_trading = True
        self.worker.publish_trade_event({"action": "ENTRY", "mode": "LIVE"})
        self.assertEqual(self.worker.session_execution_mode(), "LIVE")

        self.worker.publish_trade_event({"action": "ENTRY", "mode": "PAPER_FALLBACK"})
        self.assertEqual(self.worker.session_execution_mode(), "MIXED")

    def test_stop_event_flattens_before_worker_reaches_stopped(self):
        """A pre-set terminal event is translated into flatten-then-stop."""

        self.worker.pos.active = True
        self.stop_event.set()

        def close_position(_reason):
            self.worker.pos.active = False

        with patch.object(self.worker, "exit_position", side_effect=close_position) as close:
            self.assertTrue(self.worker._run_shutdown_cycle_if_requested())

        close.assert_called_once_with("STOP_EVENT")
        self.assertEqual(
            self.worker.lifecycle.snapshot().state,
            master_file.LifecycleState.STOPPED,
        )

    def test_transient_close_failure_retries_on_one_second_backoff(self):
        """A failed first close keeps ownership and a later retry reaches flat."""

        clock = [100.0]
        self.worker.lifecycle = master_file.TradingLifecycle(monotonic=lambda: clock[0])
        self.worker.pos.active = True
        attempts = []

        def close_position(reason):
            attempts.append(reason)
            if len(attempts) == 2:
                self.worker.pos.active = False

        with patch.object(self.worker, "exit_position", side_effect=close_position):
            self.worker.handle_square_off_and_stop()
            waiting = self.worker.lifecycle.snapshot()
            self.assertEqual(waiting.state, master_file.LifecycleState.RECONCILING)
            self.assertEqual(waiting.next_retry_at, 101.0)

            # Before the deadline, no duplicate close is submitted.
            self.assertTrue(self.worker._run_shutdown_cycle_if_requested())
            self.assertEqual(attempts, ["TIME_CUTOFF"])

            clock[0] = 101.0
            self.assertTrue(self.worker._run_shutdown_cycle_if_requested())

        self.assertEqual(attempts, ["TIME_CUTOFF", "TIME_CUTOFF"])
        self.assertEqual(
            self.worker.lifecycle.snapshot().state,
            master_file.LifecycleState.STOPPED,
        )

    def test_permanent_close_failure_never_reports_stopped(self):
        """Unresolved exposure remains in degraded reconciliation indefinitely."""

        self.worker.pos.active = True
        with patch.object(self.worker, "exit_position") as close:
            self.worker.handle_max_loss_and_stop(-1000.0, -1000.0)

        close.assert_called_once_with("MAX_LOSS_BREACH")
        snapshot = self.worker.lifecycle.snapshot()
        self.assertEqual(snapshot.state, master_file.LifecycleState.RECONCILING)
        self.assertFalse(snapshot.entry_allowed)


# =============================================================================
# TEST SUITE: ATM SINGLE-LEG STRATEGY WORKER
# =============================================================================
class TestAtmSingleLegStrategyWorker(unittest.TestCase):
    """
    Tests `enter_position` -> `_get_open_position_pnl` -> `exit_position`
    end-to-end for an ATM single-leg paper trade. The contract resolver is
    mocked to return a canned option contract so no CSV is needed.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        # Mock the resolver - we don't want to hit the instrument-master CSV.
        self.worker.contract_resolver = MagicMock()
        self.worker.contract_resolver.get_atm_option.return_value = {
            "security_id": 49081,
            "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": "NIFTY-22500-CE",
            "custom_symbol": "NIFTY 22500 CE",
            "strike": 22500.0,
            "option_type": "CE",
            "expiry_date": date.today() + timedelta(days=7),
            "days_to_expiry": 7,
            "lot_size": 50,
            "spot_reference": 22500.0,
            "atm_strike_rounded": 22500.0,
        }
        # Seed LTP cache: spot + option price.
        self.store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
             master_file.NIFTY_INDEX_SECURITY_ID): 22500.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 49081): 100.0,
        })

    def test_compute_entry_lots_default(self):
        """Default sizing returns the class-level `lots` attribute unchanged."""
        self.worker.lots = 3
        self.assertEqual(
            self.worker._compute_entry_lots(22500.0, 22400.0, lot_size=50), 3
        )

    def test_enter_position_opens_paper_trade(self):
        """`enter_position` resolves ATM, fills at LTP, and persists position state."""
        result = self.worker.enter_position(
            direction="LONG",
            entry_underlying=22500.0,
            stop_underlying=22400.0,
            target_underlying=22700.0,
        )
        self.assertTrue(result)
        self.assertTrue(self.worker.pos.active)
        self.assertEqual(self.worker.pos.direction, "LONG")
        self.assertEqual(self.worker.pos.option_security_id, 49081)
        self.assertEqual(self.worker.pos.entry_trade_price, 100.0)
        self.assertEqual(self.worker.pos.quantity, 50 * self.worker.lots)
        # Subscription was registered with the fetcher.
        subs = self.store.snapshot_option_subscriptions()
        self.assertTrue(any(s.security_id == 49081 for s in subs))

    def test_enter_position_skips_when_spot_unavailable(self):
        """No spot LTP -> no entry. The broker fallback also returns 0."""
        self.broker.fetch_ltp_map.return_value = {}
        # Wipe the cached spot to force a broker call.
        self.store._ltp_snapshots.clear()  # type: ignore[attr-defined]
        result = self.worker.enter_position(
            direction="LONG", entry_underlying=0.0,
        )
        self.assertFalse(result)
        self.assertFalse(self.worker.pos.active)

    def test_enter_position_skips_when_option_ltp_unavailable(self):
        """Option LTP missing -> entry refused."""
        # Wipe option price but keep spot.
        self.store._ltp_snapshots.pop(  # type: ignore[attr-defined]
            (master_file.OPTION_EXCHANGE_SEGMENT, 49081), None
        )
        self.broker.fetch_ltp_map.return_value = {}
        result = self.worker.enter_position(
            direction="LONG", entry_underlying=22500.0,
        )
        self.assertFalse(result)

    def test_rejected_sizing_decision_skips_entry_before_order_routing(self):
        decision = master_file.SizingDecision.from_risk_budget(
            entry=22500.0,
            stop=22400.0,
            lot_size=50,
            budget=1000.0,
            max_lots=5,
        )
        self.assertFalse(decision.accepted)
        with (
            patch.object(
                self.worker,
                "_compute_entry_sizing",
                return_value=decision,
            ),
            patch.object(self.worker, "_place_real_leg") as route_order,
        ):
            result = self.worker.enter_position(
                direction="LONG",
                entry_underlying=22500.0,
                stop_underlying=22400.0,
            )

        self.assertFalse(result)
        self.assertFalse(self.worker.pos.active)
        route_order.assert_not_called()

    def test_get_open_position_pnl_marks_to_market(self):
        """Open MTM = (live - entry) * qty for the BUY leg."""
        self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        # Move the option price up by 20.
        self.store.update_ltp_map(
            {(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 120.0}
        )
        expected = (120.0 - 100.0) * (50 * self.worker.lots)
        self.assertAlmostEqual(self.worker._get_open_position_pnl(), expected)

    def test_exit_position_realizes_pnl(self):
        """`exit_position` closes, accumulates realized PnL, unsubscribes leg."""
        self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.store.update_ltp_map(
            {(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 130.0}
        )
        self.worker.exit_position("TEST_EXIT")

        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.worker.completed_trades, 1)
        self.assertAlmostEqual(
            self.worker.realized_pnl, (130.0 - 100.0) * (50 * self.worker.lots)
        )
        # Subscription was cleaned up.
        self.assertFalse(self.store.snapshot_option_subscriptions())

    def test_exit_position_noop_when_flat(self):
        """Exit called on a flat worker is a no-op (defensive)."""
        self.worker.exit_position("TEST_EXIT")
        self.assertEqual(self.worker.completed_trades, 0)
        self.assertEqual(self.worker.realized_pnl, 0.0)


# =============================================================================
# TEST SUITE: PROFIT SHOOTER DYNAMIC LOT SIZING
# =============================================================================
class TestProfitShooterStrategyWorker(unittest.TestCase):
    """
    Tests the Profit Shooter override that picks lots dynamically based on
    the distance between entry and stop on the underlying.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.ProfitShooterStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )

    def test_compute_entry_lots_returns_positive_integer(self):
        """For a reasonable SL distance, the override must return >= 1 lot."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0,
            stop_underlying=22450.0,
            lot_size=50,
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)

    def test_compute_entry_lots_handles_zero_distance(self):
        """Zero SL distance fails closed instead of inventing fallback risk."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0,
            stop_underlying=22500.0,
            lot_size=50,
        )
        self.assertIsInstance(lots, int)
        self.assertEqual(lots, 0)

    def test_one_lot_over_budget_is_skipped(self):
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0,
            stop_underlying=22449.0,
            lot_size=50,
        )

        self.assertEqual(lots, 0)

    def test_tiny_stop_never_exceeds_namespaced_five_lot_cap(self):
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0,
            stop_underlying=22499.9,
            lot_size=50,
        )

        self.assertEqual(lots, master_file.PROFIT_SHOOTER_MAX_LOTS)

    def test_build_strategy_frame_resamples_to_five_minutes(self):
        """Profit Shooter is a 5-minute method: the strategy frame must be built
        from the 1-min source resampled to 5-min candles, not raw 1-min bars."""
        n = 400
        ts = pd.date_range("2026-05-15 09:15", periods=n, freq="1min")
        close = pd.Series([22500.0 + i * 0.5 for i in range(n)])
        ohlc = pd.DataFrame({
            "timestamp": ts,
            "open":  close.shift(1).fillna(22500.0).values,
            "high":  (close + 1.5).values,
            "low":   (close - 1.5).values,
            "close": close.values,
        })
        frame = self.worker.build_strategy_frame(ohlc)
        # 400 complete 1-min bars -> 80 complete 5-min buckets.
        self.assertEqual(len(frame), n // 5)
        spacing = pd.to_datetime(frame["timestamp"]).diff().dropna().unique()
        self.assertEqual(len(spacing), 1)
        self.assertEqual(pd.Timedelta(spacing[0]), pd.Timedelta(minutes=5))

    def test_open_position_bypasses_entry_indicator_warmup(self):
        """An existing trade's hard stop cannot wait for 200 entry bars."""

        self.worker.pos.active = True

        self.assertEqual(self.worker.minimum_strategy_rows(), 1)
        self.assertEqual(self.worker.minimum_source_rows(), 1)


# =============================================================================
# TEST SUITE: GOLDMINE DYNAMIC LOT SIZING
# =============================================================================
class _NextOpenWorkerTestMixin:
    """Shared acceptance tests for one-bar ``NEXT_OPEN`` worker intents."""

    worker = None
    decision_type = None

    def _next_open_decision(
        self,
        *,
        action="ENTER_LONG",
        entry=100.0,
        stop=95.0,
        target=110.0,
        signal_at=datetime(2026, 7, 16, 10, 0),
    ):
        return self.decision_type(
            action=action,
            entry_underlying=entry,
            stop_underlying=stop,
            target_underlying=target,
            signal_triggered=True,
            debug={"entry_timing": "NEXT_OPEN", "timestamp": signal_at},
        )

    def test_next_open_signal_is_queued_instead_of_entered_immediately(self):
        decision = self._next_open_decision()
        self.worker.signal_engine.evaluate_candle = MagicMock(return_value=decision)
        self.worker.enter_position = MagicMock(return_value=True)

        self.worker.process_strategy_frame(pd.DataFrame({"close": [100.0]}))

        self.worker.enter_position.assert_not_called()
        self.assertIsNotNone(self.worker._pending_next_open)
        self.assertEqual(
            self.worker._pending_next_open.expected_open_at,
            datetime(2026, 7, 16, 10, 5),
        )

    def test_long_gap_rebases_stop_and_target_from_observed_next_open(self):
        decision = self._next_open_decision()
        self.assertTrue(self.worker._queue_next_open_decision("LONG", decision))
        self.worker.enter_position = MagicMock(return_value=True)

        consumed = self.worker.process_pending_entry(
            pd.DataFrame(
                {
                    "timestamp": [datetime(2026, 7, 16, 10, 5)],
                    "open": [120.0],
                }
            )
        )

        self.assertTrue(consumed)
        self.worker.enter_position.assert_called_once_with(
            "LONG",
            120.0,
            115.0,
            target_underlying=130.0,
        )
        self.assertEqual(self.worker.entry_submit_count, 1)
        self.assertIsNone(self.worker._pending_next_open)

    def test_short_gap_rebases_stop_and_target_from_observed_next_open(self):
        decision = self._next_open_decision(
            action="ENTER_SHORT",
            entry=100.0,
            stop=105.0,
            target=90.0,
        )
        self.assertTrue(self.worker._queue_next_open_decision("SHORT", decision))
        self.worker.enter_position = MagicMock(return_value=True)

        consumed = self.worker.process_pending_entry(
            pd.DataFrame(
                {
                    "timestamp": [datetime(2026, 7, 16, 10, 5)],
                    "open": [80.0],
                }
            )
        )

        self.assertTrue(consumed)
        self.worker.enter_position.assert_called_once_with(
            "SHORT",
            80.0,
            85.0,
            target_underlying=70.0,
        )
        self.assertIsNone(self.worker._pending_next_open)

    def test_missing_expected_open_expires_after_one_bar(self):
        decision = self._next_open_decision()
        self.assertTrue(self.worker._queue_next_open_decision("LONG", decision))
        self.worker.enter_position = MagicMock(return_value=True)

        consumed = self.worker.process_pending_entry(
            pd.DataFrame(
                {
                    "timestamp": [datetime(2026, 7, 16, 10, 10)],
                    "open": [120.0],
                }
            )
        )

        self.assertTrue(consumed)
        self.worker.enter_position.assert_not_called()
        self.assertIsNone(self.worker._pending_next_open)

    def test_pending_intent_waits_until_expected_open_slot(self):
        decision = self._next_open_decision()
        self.assertTrue(self.worker._queue_next_open_decision("LONG", decision))
        self.worker.enter_position = MagicMock(return_value=True)

        consumed = self.worker.process_pending_entry(
            pd.DataFrame(
                {
                    "timestamp": [datetime(2026, 7, 16, 10, 4)],
                    "open": [101.0],
                }
            )
        )

        self.assertFalse(consumed)
        self.worker.enter_position.assert_not_called()
        self.assertIsNotNone(self.worker._pending_next_open)


class TestGoldmineStrategyWorker(_NextOpenWorkerTestMixin, unittest.TestCase):
    """Goldmine reuses Profit Shooter's risk-based `_compute_entry_lots`."""

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.GoldmineStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        self.decision_type = master_file.GOLDMINE_LOGIC.GoldmineDecision

    def test_compute_entry_lots_returns_positive_integer(self):
        """For a reasonable SL distance, the sizer returns >= 1 lot."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22450.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)

    def test_compute_entry_lots_handles_zero_distance(self):
        """Zero SL distance is an explicit no-trade decision."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22500.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertEqual(lots, 0)

    def test_namespaced_max_lots_caps_tiny_stop(self):
        with patch.object(master_file, "GOLDMINE_MAX_LOTS", 2):
            lots = self.worker._compute_entry_lots(
                entry_underlying=22500.0,
                stop_underlying=22499.9,
                lot_size=50,
            )

        self.assertEqual(lots, 2)


# =============================================================================
# TEST SUITE: MONEY MACHINE DYNAMIC LOT SIZING
# =============================================================================
class TestMoneyMachineStrategyWorker(_NextOpenWorkerTestMixin, unittest.TestCase):
    """Money Machine reuses the same risk-based `_compute_entry_lots`."""

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.MoneyMachineStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        self.decision_type = master_file.MONEY_MACHINE_LOGIC.MoneyMachineDecision

    def test_compute_entry_lots_returns_positive_integer(self):
        """For a reasonable SL distance, the sizer returns >= 1 lot."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22450.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)

    def test_compute_entry_lots_handles_zero_distance(self):
        """Zero SL distance is an explicit no-trade decision."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22500.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertEqual(lots, 0)

    def test_namespaced_max_lots_caps_tiny_stop(self):
        with patch.object(master_file, "MONEY_MACHINE_MAX_LOTS", 3):
            lots = self.worker._compute_entry_lots(
                entry_underlying=22500.0,
                stop_underlying=22499.9,
                lot_size=50,
            )

        self.assertEqual(lots, 3)


# =============================================================================
# TEST SUITE: PER-STRATEGY build_strategy_frame SMOKE TESTS
# =============================================================================
class TestStrategyFrameBuilders(unittest.TestCase):
    """
    Smoke tests for each worker's `build_strategy_frame`. We feed a synthetic
    OHLC frame large enough to satisfy each strategy's warm-up requirements
    and verify the call returns a non-None DataFrame without raising.

    Deep signal-logic correctness is out of scope for these smoke tests -
    those tests would essentially re-implement each strategy's indicator
    library. The goal here is to catch shape / interface regressions.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        # 400 1-min bars: enough warm-up for SMA200, ATR, Supertrend, etc.
        n = 400
        ts = pd.date_range("2026-05-15 09:15", periods=n, freq="1min")
        # Slight uptrend + small noise so indicators are well-defined.
        close = pd.Series(
            [22500.0 + i * 0.5 + ((i * 7) % 13 - 6) * 0.3 for i in range(n)]
        )
        self.ohlc = pd.DataFrame({
            "timestamp": ts,
            "open":  close.shift(1).fillna(22500.0).values,
            "high":  (close + 1.5).values,
            "low":   (close - 1.5).values,
            "close": close.values,
        })

    def _smoke(self, worker_cls):
        """Run `build_strategy_frame(self.ohlc)` and assert it returns a DataFrame."""
        worker = worker_cls(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        result = worker.build_strategy_frame(self.ohlc)
        self.assertIsInstance(result, pd.DataFrame)

    def test_renko_build_strategy_frame(self):
        self._smoke(master_file.RenkoStrategyWorker)

    def test_ema_build_strategy_frame(self):
        self._smoke(master_file.EMATrendStrategyWorker)

    def test_heikin_ashi_build_strategy_frame(self):
        self._smoke(master_file.HeikinAshiStrategyWorker)

    def test_profit_shooter_build_strategy_frame(self):
        self._smoke(master_file.ProfitShooterStrategyWorker)

    def test_goldmine_build_strategy_frame(self):
        self._smoke(master_file.GoldmineStrategyWorker)

    def test_money_machine_build_strategy_frame(self):
        self._smoke(master_file.MoneyMachineStrategyWorker)

    def test_pcr_vwap_atr_build_strategy_frame(self):
        self._smoke(master_file.OpeningStrikePCRVWAPATRWorker)

    def test_supertrend_bullish_build_strategy_frame(self):
        self._smoke(master_file.SupertrendBullishWorker)

    def test_donchian_bearish_build_strategy_frame(self):
        self._smoke(master_file.DonchianBearishWorker)


# =============================================================================
# TEST SUITE: SHOONYA LIVE-TRADING TOGGLE
# =============================================================================
class _FakeShoonya:
    """A stand-in for the active `execution_client` that records calls instead of
    hitting the broker.
    - `fail_on(symbol, side)` simulates an order reject on a specific leg.
    - `resolve_returns` overrides the resolved Shoonya symbol; pass "" to simulate
      a symbol-master resolution miss."""

    def __init__(self, fail_on=None, resolve_returns=None, result_status=OrderStatus.FILLED):
        self.calls = []      # (shoonya_symbol, side, quantity)
        self.order_tags = []
        self.resolved = []   # (underlying, option_type, strike)
        self._fail_on = fail_on or (lambda symbol, side: False)
        self._resolve_returns = resolve_returns
        self._result_status = result_status

    def resolve_option_symbol(self, underlying, expiry, option_type, strike,
                              exchange_segment="NFO"):
        self.resolved.append((underlying, option_type, float(strike)))
        if self._resolve_returns is not None:
            return self._resolve_returns
        return f"SHOONYA-{underlying}-{int(strike)}-{option_type}"

    def place_market_order(self, symbol, side, quantity,
                           exchange_segment="NFO", product_type="INTRADAY",
                           *, order_tag=""):
        self.calls.append((symbol, side, quantity))
        self.order_tags.append(order_tag)
        status = (
            OrderStatus.REJECTED
            if self._fail_on(symbol, side)
            else self._result_status
        )
        if callable(status):
            status = status(symbol, side)
        filled = {
            OrderStatus.FILLED: int(quantity),
            OrderStatus.PARTIAL: max(1, int(quantity) // 2),
            OrderStatus.REJECTED: 0,
            OrderStatus.UNKNOWN: 0,
        }[status]
        return OrderResult(
            order_id=f"ORD-{len(self.calls)}",
            requested_quantity=int(quantity),
            filled_quantity=filled,
            remaining_quantity=int(quantity) - filled,
            status=status,
            broker_state=status.value,
            reason=f"simulated {status.value.lower()} outcome",
        )

    def extract_order_id(self, resp):
        return resp.order_id if isinstance(resp, OrderResult) else ""


class TestEnvBool(unittest.TestCase):
    """`_env_bool` parses .env booleans forgivingly."""

    def test_truthy_values(self):
        for raw in ("1", "true", "TRUE", "Yes", "on", '"true"'):
            with patch.dict(os.environ, {"X_FLAG": raw}):
                self.assertTrue(master_file._env_bool("X_FLAG", False), raw)

    def test_falsy_and_default(self):
        for raw in ("0", "false", "no", "off", "garbage"):
            with patch.dict(os.environ, {"X_FLAG": raw}):
                self.assertFalse(master_file._env_bool("X_FLAG", True), raw)
        # Blank / unset falls back to the supplied default.
        with patch.dict(os.environ, {"X_FLAG": ""}):
            self.assertTrue(master_file._env_bool("X_FLAG", True))
        os.environ.pop("X_FLAG", None)
        self.assertFalse(master_file._env_bool("X_FLAG", False))


class TestSelectExecutionClient(unittest.TestCase):
    """`_select_execution_client` routes brokers and fails closed on a typo.

    Its own docstring calls this out as the reason the decision lives in one
    small function: a typo must never route real-money orders to another
    broker, and an unrecognised name must disable live trading entirely.
    """

    def _select(self, broker, **clients):
        """Run the selector with every broker client patched to a sentinel."""
        defaults = {
            "kotak_execution_client": "KOTAK-CLIENT",
            "shoonya_execution_client": "SHOONYA-CLIENT",
            "flattrade_execution_client": "FLATTRADE-CLIENT",
            "dhan_execution_client": "DHAN-CLIENT",
        }
        defaults.update(clients)
        with ExitStack() as stack:
            for name, value in defaults.items():
                stack.enter_context(patch.object(master_file, name, value))
            return master_file._select_execution_client(broker)

    def test_each_broker_routes_to_its_own_client_and_segment(self):
        # The exchange-segment string is broker-specific and is passed straight
        # through to the order call, so a wrong value would reject every order.
        expected = {
            "KOTAK": ("KOTAK-CLIENT", "nse_fo"),
            "SHOONYA": ("SHOONYA-CLIENT", "NFO"),
            "FLATTRADE": ("FLATTRADE-CLIENT", "NFO"),
            "DHAN": ("DHAN-CLIENT", "NSE_FNO"),
        }
        for broker, (client, segment) in expected.items():
            with self.subTest(broker=broker), patch.dict(os.environ, {}, clear=False):
                got_client, got_segment, got_product = self._select(broker)
                self.assertEqual(got_client, client)
                self.assertEqual(got_segment, segment)
                self.assertEqual(got_product, "INTRADAY")

    def test_broker_name_is_case_and_whitespace_insensitive(self):
        client, segment, _product = self._select("  dhan  ")
        self.assertEqual(client, "DHAN-CLIENT")
        self.assertEqual(segment, "NSE_FNO")

    def test_product_type_comes_from_the_brokers_own_env_key(self):
        with patch.dict(os.environ, {"DHAN_PRODUCT_TYPE": "normal"}):
            _client, _segment, product = self._select("DHAN")
        self.assertEqual(product, "NORMAL")

    def test_unknown_broker_fails_closed(self):
        for broker in ("ZERODHA", "", "dhann", None):
            with self.subTest(broker=broker):
                self.assertEqual(
                    self._select(broker),
                    (None, "", "INTRADAY"),
                )

    def test_missing_client_yields_none_so_startup_forces_paper(self):
        # A broker whose SDK failed to import is None; the selector still
        # returns it so `_configure_startup_live_trading` disables live mode.
        client, segment, _product = self._select("DHAN", dhan_execution_client=None)
        self.assertIsNone(client)
        self.assertEqual(segment, "NSE_FNO")


class TestTimezoneAssumption(unittest.TestCase):
    """MAT-103: trading windows are pinned to Asia/Kolkata explicitly."""

    def test_ist_offset_produces_no_warning(self):
        offset = timedelta(hours=5, minutes=30)
        self.assertIsNone(master_file._timezone_assumption_warning(offset))

    def test_non_ist_host_offset_no_longer_needs_a_warning(self):
        for offset in (timedelta(0), timedelta(hours=-5), timedelta(hours=5, minutes=45)):
            self.assertIsNone(master_file._timezone_assumption_warning(offset), offset)

    def test_default_uses_system_offset(self):
        # Kept as a compatibility helper for callers from older scripts.
        system_offset = datetime.now().astimezone().utcoffset()
        self.assertIsNone(master_file._timezone_assumption_warning())
        self.assertIsNone(master_file._timezone_assumption_warning(system_offset))

    def test_ist_clock_is_timezone_aware(self):
        now = master_file._ist_now()
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.utcoffset(), timedelta(hours=5, minutes=30))


class TestGoogleSheetRetry(unittest.TestCase):
    """OPS-001: one transient gspread/network hiccup used to skip the day's P&L
    write entirely. The writer now takes a few slow retries before giving up
    (still strictly non-fatal)."""

    def _fake_gspread(self, fail_times: int):
        import types as _types

        calls = {"oauth": 0, "updates": []}

        class _WorksheetNotFound(Exception):
            pass

        class _FakeWorksheet:
            def get_all_values(self):
                return [["Strategy", "2026-07-07"], ["Renko", ""]]

            def update_cells(self, cells, value_input_option=""):
                calls["updates"].append(list(cells))

        class _FakeSpreadsheet:
            def worksheet(self, name):
                return _FakeWorksheet()

        class _FakeClient:
            def open_by_key(self, key):
                return _FakeSpreadsheet()

        def _oauth(**kwargs):
            calls["oauth"] += 1
            if calls["oauth"] <= fail_times:
                raise ConnectionError("simulated transient Google outage")
            return _FakeClient()

        module = _types.ModuleType("gspread")
        module.oauth = _oauth
        module.WorksheetNotFound = _WorksheetNotFound
        module.Cell = lambda row, col, value: (row, col, value)
        return module, calls

    def _run_writer(self, fake_module):
        with (
            patch.dict(sys.modules, {"gspread": fake_module}),
            patch.dict(os.environ, {"GSHEET_ID": "sheet-id"}),
            patch.object(master_file, "_parse_eod_pnl_by_day",
                         return_value={"2026-07-07": {"Renko": 123.0}}),
            patch.object(master_file, "_compute_pnl_sheet_updates",
                         return_value=([(1, 1, 123.0)], [])),
            patch.object(master_file.time, "sleep"),   # retries must not slow tests
        ):
            master_file._update_pnl_google_sheet()

    def test_transient_failure_is_retried_then_writes(self):
        fake, calls = self._fake_gspread(fail_times=1)
        self._run_writer(fake)
        self.assertEqual(calls["oauth"], 2)            # failed once, then succeeded
        self.assertEqual(len(calls["updates"]), 1)     # the day's P&L was written

    def test_persistent_failure_stays_bounded_and_nonfatal(self):
        fake, calls = self._fake_gspread(fail_times=99)
        self._run_writer(fake)                          # must not raise
        self.assertEqual(calls["oauth"], 3)             # bounded attempts
        self.assertEqual(calls["updates"], [])


class TestExecutionModeResults(unittest.TestCase):
    """End-of-day logs, Telegram, and Sheet rows must retain execution provenance."""

    @staticmethod
    def _worker(name: str, mode: str, pnl: float, trades: int):
        worker = MagicMock()
        worker.strategy_name = name
        worker.realized_pnl = pnl
        worker.completed_trades = trades
        worker.session_execution_mode.return_value = mode
        return worker

    def test_eod_summary_reports_mixed_mode_and_per_strategy_modes(self):
        event_queue = master_file.queue.Queue()
        workers = [
            self._worker("Renko", "LIVE", 100.0, 1),
            self._worker("EMA", "PAPER", -25.0, 2),
        ]

        master_file._publish_eod_summary(workers, event_queue)

        event = event_queue.get_nowait()
        self.assertEqual(event["mode"], "MIXED")
        self.assertEqual([row["mode"] for row in event["rows"]], ["LIVE", "PAPER"])
        self.assertIn("[MIXED]", master_file.format_trade_message(event))

    def test_log_parser_keeps_new_modes_and_legacy_paper_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runner.log"
            path.write_text(
                "2026-07-16 15:15:00,000 | INFO | RenkoThread | "
                "Result summary | Mode=LIVE | Trades=1 | RealizedPnL=100.00\n"
                "2026-07-16 15:16:00,000 | INFO | EMAThread | "
                "Result summary | Mode=MIXED | Trades=2 | RealizedPnL=-25.00\n"
                "2026-07-15 15:15:00,000 | INFO | RenkoThread | "
                "Paper summary | Trades=1 | RealizedPnL=50.00\n",
                encoding="utf-8",
            )

            parsed = master_file._parse_eod_pnl_by_day(path)

        self.assertEqual(parsed["2026-07-16"]["Renko"], {"pnl": 100.0, "mode": "LIVE"})
        self.assertEqual(parsed["2026-07-16"]["EMA"], {"pnl": -25.0, "mode": "MIXED"})
        self.assertEqual(parsed["2026-07-15"]["Renko"], {"pnl": 50.0, "mode": "PAPER"})

    def test_sheet_uses_mode_specific_labels_without_turning_pnl_into_text(self):
        values = [
            ["Strategy", "2026-07-16"],
            ["Renko Strategy [LIVE]", ""],
            ["EMA Strategy [MIXED]", ""],
            ["Renko Strategy", ""],
        ]
        pnl_by_day = {
            "2026-07-16": {
                "Renko": {"pnl": 100.0, "mode": "LIVE"},
                "EMA": {"pnl": -25.0, "mode": "MIXED"},
            }
        }

        updates, unmatched = master_file._compute_pnl_sheet_updates(
            values, pnl_by_day, "2026-07-16"
        )

        self.assertEqual(updates, [(1, 1, 100.0), (2, 1, -25.0)])
        self.assertEqual(unmatched, [])


class TestStrategyEnvPrefixMap(unittest.TestCase):
    """Every worker's `strategy_name` must map to an env prefix, or it can never
    be switched live (it would silently stay paper)."""

    def test_all_worker_strategy_names_are_mapped(self):
        core = [
            master_file.RenkoStrategyWorker, master_file.EMATrendStrategyWorker,
            master_file.HeikinAshiStrategyWorker, master_file.ProfitShooterStrategyWorker,
            master_file.GoldmineStrategyWorker, master_file.MoneyMachineStrategyWorker,
            master_file.OpeningStrikePCRVWAPATRWorker, master_file.CPRStrategyWorker,
            master_file.CPRAlgo3StrategyWorker,
            master_file.SupertrendBullishWorker, master_file.DonchianBearishWorker,
            master_file.Delta20HedgedSpreadWorker, master_file.LongStrangleWorker,
        ]
        for cls in core + list(master_file.SIGNAL_GEN_WORKERS):
            self.assertIn(
                cls.strategy_name, master_file.STRATEGY_ENV_PREFIX,
                f"{cls.__name__} ({cls.strategy_name}) missing from STRATEGY_ENV_PREFIX",
            )

    def test_no_empty_prefixes(self):
        for name, prefix in master_file.STRATEGY_ENV_PREFIX.items():
            self.assertTrue(prefix, f"empty env prefix for strategy {name}")

    def test_effective_flag_requires_master_and_strategy(self):
        """Effective live = master switch AND per-strategy toggle (mirrors main())."""
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "false",
                                     "RENKO_LIVE_TRADING": "true"}):
            master = master_file._env_bool("LIVE_TRADING_ENABLED", False)
            per = master_file._env_bool("RENKO_LIVE_TRADING", False)
            self.assertFalse(master and per)
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "true",
                                     "RENKO_LIVE_TRADING": "true"}):
            master = master_file._env_bool("LIVE_TRADING_ENABLED", False)
            per = master_file._env_bool("RENKO_LIVE_TRADING", False)
            self.assertTrue(master and per)


class TestVirtualTradingToggle(unittest.TestCase):
    """The per-strategy virtual (paper) gate: a strategy runs unless its
    `<PREFIX>_VIRTUAL_TRADING` is explicitly false. Default is everything runs,
    and there is deliberately NO global master switch."""

    def test_default_is_enabled_when_key_absent(self):
        os.environ.pop("RENKO_VIRTUAL_TRADING", None)
        self.assertTrue(master_file._strategy_virtual_trading_enabled("Renko"))

    def test_explicit_false_disables(self):
        with patch.dict(os.environ, {"RENKO_VIRTUAL_TRADING": "false"}):
            self.assertFalse(master_file._strategy_virtual_trading_enabled("Renko"))

    def test_explicit_true_enables(self):
        with patch.dict(os.environ, {"RENKO_VIRTUAL_TRADING": "true"}):
            self.assertTrue(master_file._strategy_virtual_trading_enabled("Renko"))

    def test_unmapped_strategy_fails_open(self):
        """A strategy name with no env prefix must never be silently disabled."""
        self.assertTrue(master_file._strategy_virtual_trading_enabled("NoSuchStrategy"))

    def test_toggle_is_independent_per_strategy(self):
        """Disabling one strategy does not affect another."""
        with patch.dict(os.environ, {"RENKO_VIRTUAL_TRADING": "false"}):
            self.assertFalse(master_file._strategy_virtual_trading_enabled("Renko"))
            self.assertTrue(master_file._strategy_virtual_trading_enabled("EMA"))

    def test_sl_hunting_prefix_respected(self):
        """The optional agent maps to SL_HUNTING; its virtual gate must work too."""
        if "SL Hunting AI" not in master_file.STRATEGY_ENV_PREFIX:
            self.skipTest("SL Hunting worker not loaded in this environment.")
        with patch.dict(os.environ, {"SL_HUNTING_VIRTUAL_TRADING": "false"}):
            self.assertFalse(master_file._strategy_virtual_trading_enabled("SL Hunting AI"))


# =============================================================================
# TEST SUITE: LONG STRANGLE WORKER
# =============================================================================
class TestLongStrangleWorker(unittest.TestCase):
    """
    OTM1 strike resolution, the trailing-stop ladder (pure function),
    independent per-leg exits, registry coverage, and paper-by-default.
    """

    @classmethod
    def setUpClass(cls):
        # The shared resolver test uses 500-point strike gaps; the strangle
        # needs adjacent 50-point strikes to exercise the OTM1 offset, so we
        # build a dedicated synthetic instrument master here.
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.csv_path = Path(cls.tmpdir.name) / "all_instrument 1.csv"
        cls.exp1 = (date.today() + timedelta(days=7)).isoformat()
        cls.exp2 = (date.today() + timedelta(days=14)).isoformat()
        rows = []
        sec = 50000
        for exp in (cls.exp1, cls.exp2):
            for strike in (22400, 22450, 22500, 22550, 22600):
                for right in ("CE", "PE"):
                    rows.append({
                        "EXCH_ID": "NSE", "SEGMENT": "D", "INSTRUMENT": "OPTIDX",
                        "SYMBOL_NAME": f"NIFTY-{exp}-{strike}-{right}",
                        "DISPLAY_NAME": f"NIFTY {exp} {strike} {right}",
                        "SM_EXPIRY_DATE": exp, "LOT_SIZE": "75",
                        "SECURITY_ID": str(sec), "STRIKE_PRICE": str(strike),
                        "OPTION_TYPE": right, "UNDERLYING_SYMBOL": "NIFTY",
                    })
                    sec += 1
        pd.DataFrame(rows).to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _make_resolver(self):
        import logging
        glob_pattern = str(Path(self.tmpdir.name) / "all_instrument *.csv")
        return master_file.OptionsContractResolver(
            underlying="NIFTY",
            instrument_master_glob=glob_pattern,
            log=logging.getLogger("test_strangle_resolver"),
        )

    # ----- get_otm_option resolver -----------------------------------------
    def test_get_otm_option_ce_is_one_strike_above_atm(self):
        """CE OTM1 sits one step ABOVE the ATM strike, current-week expiry."""
        resolver = self._make_resolver()
        # spot 22510 -> ATM 22500 -> OTM1 CE = 22550.
        contract = resolver.get_otm_option(spot_price=22510.0, right="CE", otm_steps=1)
        self.assertEqual(contract["option_type"], "CE")
        self.assertEqual(contract["strike"], 22550.0)
        self.assertEqual(contract["lot_size"], 75)
        self.assertEqual(contract["expiry_date"].isoformat(), self.exp1)

    def test_get_otm_option_pe_is_one_strike_below_atm(self):
        """PE OTM1 sits one step BELOW the ATM strike."""
        resolver = self._make_resolver()
        # spot 22510 -> ATM 22500 -> OTM1 PE = 22450.
        contract = resolver.get_otm_option(spot_price=22510.0, right="PE", otm_steps=1)
        self.assertEqual(contract["option_type"], "PE")
        self.assertEqual(contract["strike"], 22450.0)

    def test_get_otm_option_rejects_bad_right(self):
        resolver = self._make_resolver()
        with self.assertRaises(ValueError):
            resolver.get_otm_option(spot_price=22500.0, right="XX", otm_steps=1)

    def test_get_otm_option_rejects_invalid_spot(self):
        resolver = self._make_resolver()
        with self.assertRaises(ValueError):
            resolver.get_otm_option(spot_price=0, right="CE", otm_steps=1)

    # ----- _compute_trailing_sl ladder (pure function) ---------------------
    def test_trailing_sl_initial_is_five_pct_below_entry(self):
        sl = master_file.LongStrangleWorker._compute_trailing_sl(
            entry_premium=100.0, high_water_premium=100.0,
            sl_pct=0.05, trail_trigger_pct=1.0, trail_step_pct=1.0,
        )
        self.assertAlmostEqual(sl, 95.0)

    def test_trailing_sl_reaches_breakeven_at_five_pct_gain(self):
        sl = master_file.LongStrangleWorker._compute_trailing_sl(
            entry_premium=100.0, high_water_premium=105.0,
            sl_pct=0.05, trail_trigger_pct=1.0, trail_step_pct=1.0,
        )
        self.assertAlmostEqual(sl, 100.0)

    def test_trailing_sl_locks_profit_above_breakeven(self):
        sl = master_file.LongStrangleWorker._compute_trailing_sl(
            entry_premium=100.0, high_water_premium=110.0,
            sl_pct=0.05, trail_trigger_pct=1.0, trail_step_pct=1.0,
        )
        self.assertAlmostEqual(sl, 105.0)

    def test_trailing_sl_safe_on_zero_entry(self):
        sl = master_file.LongStrangleWorker._compute_trailing_sl(
            entry_premium=0.0, high_water_premium=0.0,
            sl_pct=0.05, trail_trigger_pct=1.0, trail_step_pct=1.0,
        )
        self.assertEqual(sl, 0.0)

    # ----- Worker behaviour with a fake store + mocked resolver ------------
    def _make_worker(self):
        store = master_file.SharedMarketDataStore()
        broker = MagicMock()
        # Any cache-miss LTP lookup returns "no price" rather than a MagicMock,
        # so a deliberately-absent leg resolves cleanly to its fallback.
        broker.fetch_ltp_map.return_value = {}
        worker = master_file.LongStrangleWorker(
            store=store, stop_event=threading.Event(), broker=broker
        )

        # Canned OTM contracts so the worker never touches a CSV.
        def fake_otm(spot, right, otm_steps=1, expiry=None):
            if right == "CE":
                return {
                    "security_id": 1001, "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
                    "trading_symbol": "NIFTY-CE", "custom_symbol": "NIFTY CE",
                    "strike": 22550.0, "option_type": "CE", "expiry_date": date.today(),
                    "days_to_expiry": 2, "lot_size": 75, "spot_reference": spot,
                    "atm_strike_rounded": 22500.0, "target_strike": 22550.0,
                }
            return {
                "security_id": 2002, "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
                "trading_symbol": "NIFTY-PE", "custom_symbol": "NIFTY PE",
                "strike": 22450.0, "option_type": "PE", "expiry_date": date.today(),
                "days_to_expiry": 2, "lot_size": 75, "spot_reference": spot,
                "atm_strike_rounded": 22500.0, "target_strike": 22450.0,
            }

        worker.contract_resolver = MagicMock()
        worker.contract_resolver.get_otm_option.side_effect = fake_otm
        return worker, store

    def test_enter_both_legs_opens_two_independent_positions(self):
        worker, store = self._make_worker()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._enter_both_legs()
        self.assertTrue(worker.ce_pos.active)
        self.assertTrue(worker.pe_pos.active)
        self.assertTrue(worker.entered_today)
        self.assertEqual(worker.ce_pos.entry_trade_price, 100.0)
        self.assertEqual(worker.pe_pos.entry_trade_price, 90.0)
        self.assertEqual(worker.ce_pos.quantity, 75)  # 1 lot * 75

    def test_stopping_ce_leaves_pe_active(self):
        """Independent legs: a CE stop-out must not disturb the PE leg."""
        worker, store = self._make_worker()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._enter_both_legs()
        # Drop the CE premium below its 5% stop (100 -> 90); PE unchanged.
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._manage_leg("CE")
        self.assertFalse(worker.ce_pos.active)   # CE stopped out
        self.assertTrue(worker.pe_pos.active)    # PE untouched
        self.assertEqual(worker.exit_count, 1)
        self.assertEqual(worker.completed_trades, 1)

    def test_paper_by_default(self):
        worker, _ = self._make_worker()
        self.assertFalse(worker.live_trading)

    def test_partial_entry_is_force_closed_by_strangle_cutoff_override(self):
        """The custom cutoff path must sweep a live leg with no ce_pos owner."""

        class TerminalPartialThenCloseFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if side == "BUY":
                    filled, status, broker_state = 20, OrderStatus.PARTIAL, "CANCELLED"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"STRANGLE-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted strangle cutoff recovery",
                )

        worker, store = self._make_worker()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
             master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
        })
        worker.live_trading = True
        fake = TerminalPartialThenCloseFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            self.assertFalse(worker._enter_leg("CE", 22510.0))
            self.assertFalse(worker.ce_pos.active)
            self.assertEqual(len(worker._orphan_live_legs), 1)
            worker.handle_square_off_and_stop()

        self.assertEqual(
            [(side, quantity) for _symbol, side, quantity in fake.calls],
            [("BUY", 75), ("SELL", 20)],
        )
        self.assertEqual(worker._orphan_live_legs, [])

    def test_pnl_sheet_label_present(self):
        self.assertIn("LongStrangle", master_file._PNL_SHEET_ROW_LABELS)

    # ----- Phase 2: momentum re-entry --------------------------------------
    def _enter_and_stop_ce(self, worker, store):
        """Helper: open both legs then stop the CE leg out (100 -> 90)."""
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._enter_both_legs()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0,   # CE hits its 5% stop
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._manage_leg("CE")

    def test_stop_out_arms_momentum_reentry(self):
        """A per-leg stop-out arms the leg and records the stop-out price."""
        worker, store = self._make_worker()
        self._enter_and_stop_ce(worker, store)
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.ce_awaiting_reentry)
        self.assertAlmostEqual(worker.ce_stop_out_price, 90.0)
        self.assertEqual(worker.ce_reentry_count, 0)  # armed, not yet re-entered

    def test_momentum_rebound_triggers_reentry(self):
        """Re-entry fires only once the premium rebounds +5% above the stop."""
        worker, store = self._make_worker()
        self._enter_and_stop_ce(worker, store)  # stop_out_price = 90 -> trigger 94.5

        # Still below the +5% trigger: no re-entry.
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 1001): 93.0})
        worker._manage_reentry("CE")
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.ce_awaiting_reentry)

        # Rebounds past +5%: re-enter the SAME strike at the new premium.
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 1001): 95.0})
        worker._manage_reentry("CE")
        self.assertTrue(worker.ce_pos.active)
        self.assertFalse(worker.ce_awaiting_reentry)
        self.assertEqual(worker.ce_reentry_count, 1)
        self.assertEqual(worker.reentry_count, 1)
        self.assertEqual(worker.ce_pos.entry_trade_price, 95.0)
        self.assertEqual(worker.ce_high_water_premium, 95.0)  # high-water reset
        self.assertTrue(worker.pe_pos.active)  # PE untouched throughout

    def test_reentry_respects_max_cap(self):
        """At the re-entry cap, a stop-out does not arm again."""
        worker, store = self._make_worker()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._enter_both_legs()
        worker.ce_reentry_count = master_file.STRANGLE_MAX_REENTRIES  # cap reached
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._manage_leg("CE")
        self.assertFalse(worker.ce_pos.active)
        self.assertFalse(worker.ce_awaiting_reentry)  # not re-armed

    def test_reentry_disabled_leaves_leg_flat(self):
        """With re-entry disabled, a stopped-out leg is not armed."""
        worker, store = self._make_worker()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        worker._enter_both_legs()
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        with patch.object(master_file, "STRANGLE_REENTRY_ENABLED", False):
            worker._manage_leg("CE")
        self.assertFalse(worker.ce_pos.active)
        self.assertFalse(worker.ce_awaiting_reentry)

    def test_partial_initial_entry_does_not_refresh_stopped_leg(self):
        """A pending initial leg must not cause a fresh re-open of the OTHER
        leg after it has been stopped out (re-fills are momentum-only)."""
        worker, store = self._make_worker()
        # CE fills at 100; PE LTP deliberately absent -> PE initial entry defers.
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
        })
        worker._enter_both_legs()
        self.assertTrue(worker.ce_pos.active)
        self.assertTrue(worker.ce_initial_done)
        self.assertFalse(worker.pe_pos.active)
        self.assertFalse(worker.pe_initial_done)
        self.assertFalse(worker.entered_today)

        # CE stops out -> armed, flat.
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0})
        worker._manage_leg("CE")
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.ce_awaiting_reentry)

        # Re-call the initial-entry path (PE still pending). CE must stay flat -
        # its initial entry is done, so it is never fresh-entered again.
        worker._enter_both_legs()
        self.assertFalse(worker.ce_pos.active)
        self.assertEqual(worker.ce_reentry_count, 0)

    def test_paper_fallback_leg_then_exit_sends_no_real_order_but_flattens(self):
        """P1 (LongStrangle sibling of PR #42 / HEDGE-001 and the single-leg guard):
        a LIVE worker whose leg BUY fell back to paper (rejected order / symbol-master
        miss) opened no real leg at the broker, so closing that leg must NOT send a
        real SELL -- that would be a naked short of an OTM option we never bought.
        The exit still flattens the paper books (nothing real is open to keep for
        retry). Mirrors TestLiveOrderRouting.<same name for the single-leg worker>."""
        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        # Every real order is rejected -> both legs are recorded as paper (no live leg).
        reject_fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        with patch.object(master_file, "execution_client", reject_fake):
            worker._enter_both_legs()
        self.assertTrue(worker.ce_pos.active)               # tracked as paper
        self.assertTrue(worker.pe_pos.active)
        self.assertIsNone(worker.ce_pos.live_leg)           # ...but no real leg opened
        self.assertIsNone(worker.pe_pos.live_leg)

        # A fresh client for the exits must receive ZERO orders...
        exit_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", exit_fake):
            worker._exit_leg("CE", "TEST_EXIT")
            worker._exit_leg("PE", "TEST_EXIT")
        self.assertEqual(exit_fake.calls, [])               # no phantom naked short
        self.assertFalse(worker.ce_pos.active)              # ...yet both legs flattened
        self.assertFalse(worker.pe_pos.active)
        self.assertEqual(worker.completed_trades, 2)

    def test_confirmed_live_leg_marks_legs_open_and_exit_sells(self):
        """The invariant's other side (non-regression): a confirmed live leg BUY marks
        an entry-complete live-leg snapshot, and closing it sends one real SELL.
        Also covers the shared `_buy_leg` path used by momentum re-entries."""
        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        entry_fake = _FakeShoonya()                          # every order fills
        with patch.object(master_file, "execution_client", entry_fake):
            worker._enter_both_legs()
        self.assertTrue(worker.ce_pos.live_leg.entry_complete)  # both legs really open
        self.assertTrue(worker.pe_pos.live_leg.entry_complete)

        # Closing the CE leg sends exactly one real SELL for that leg; PE untouched.
        exit_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", exit_fake):
            worker._exit_leg("CE", "TEST_EXIT")
        self.assertEqual([s for (_sym, s, _q) in exit_fake.calls], ["SELL"])
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.pe_pos.active)

    def test_initial_live_legs_share_correlation_but_have_distinct_roles(self):
        """The CE/PE basket is traceable as one correlation without conflating legs."""
        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })

        with patch.object(master_file, "execution_client", _FakeShoonya()):
            worker._enter_both_legs()

        self.assertIsNotNone(worker.ce_pos.live_leg)
        self.assertIsNotNone(worker.pe_pos.live_leg)
        self.assertEqual(worker.ce_pos.live_leg.spec.role, "C")
        self.assertEqual(worker.pe_pos.live_leg.spec.role, "P")
        self.assertEqual(
            worker.ce_pos.live_leg.spec.correlation_id,
            worker.pe_pos.live_leg.spec.correlation_id,
        )

    def test_reentry_gets_new_correlation_after_prior_leg_is_confirmed_flat(self):
        """A stopped leg's new life never reuses its broker-audit identity."""
        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        client = _FakeShoonya()
        with patch.object(master_file, "execution_client", client):
            worker._enter_both_legs()
            old_live_leg = worker.ce_pos.live_leg

            store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 1001): 90.0})
            worker._manage_leg("CE")
            self.assertTrue(store.execution_ledger.get(old_live_leg.exposure_id).broker_confirmed_flat)

            store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 1001): 95.0})
            worker._manage_reentry("CE")

        self.assertTrue(worker.ce_pos.live_leg.entry_complete)
        self.assertNotEqual(
            worker.ce_pos.live_leg.spec.correlation_id,
            old_live_leg.spec.correlation_id,
        )

    def test_partial_close_keeps_leg_active_then_retries_only_remaining_quantity(self):
        """A terminal partial SELL preserves 50 open units and retries exactly 50."""

        class PartialCloseClient(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.sell_count = 0

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if side == "SELL":
                    self.sell_count += 1
                    if self.sell_count == 1:
                        return OrderResult(
                            order_id="SELL-PARTIAL",
                            requested_quantity=int(quantity),
                            filled_quantity=25,
                            remaining_quantity=int(quantity) - 25,
                            status=OrderStatus.PARTIAL,
                            broker_state="CANCELLED",
                            reason="terminal partial close",
                        )
                return OrderResult(
                    order_id=f"ORD-{len(self.calls)}",
                    requested_quantity=int(quantity),
                    filled_quantity=int(quantity),
                    remaining_quantity=0,
                    status=OrderStatus.FILLED,
                    broker_state="FILLED",
                    reason="simulated fill",
                )

        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        client = PartialCloseClient()
        with patch.object(master_file, "execution_client", client):
            worker._enter_both_legs()
            worker._exit_leg("CE", "TEST_PARTIAL")

            self.assertTrue(worker.ce_pos.active)
            self.assertIsNotNone(worker.ce_pos.live_leg)
            self.assertEqual(worker.ce_pos.live_leg.confirmed_live_quantity, 50)

            worker._exit_leg("CE", "TEST_RETRY")

        sell_quantities = [qty for (_symbol, side, qty) in client.calls if side == "SELL"]
        self.assertEqual(sell_quantities, [75, 50])
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.pe_pos.active)

    def test_partial_initial_leg_stays_tracked_and_retries_only_remaining_quantity(self):
        """A partial CE retries 50, then its planned PE companion may open."""

        class PartialEntryClient(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.buy_count = 0

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                self.buy_count += 1
                if self.buy_count > 1:
                    return OrderResult(
                        order_id="BUY-REMAINDER",
                        requested_quantity=int(quantity),
                        filled_quantity=int(quantity),
                        remaining_quantity=0,
                        status=OrderStatus.FILLED,
                        broker_state="FILLED",
                        reason="remaining entry filled",
                    )
                return OrderResult(
                    order_id="BUY-PARTIAL",
                    requested_quantity=int(quantity),
                    filled_quantity=25,
                    remaining_quantity=int(quantity) - 25,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial entry",
                )

        worker, store = self._make_worker()
        worker.live_trading = True
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        client = PartialEntryClient()
        with patch.object(master_file, "execution_client", client):
            worker._enter_both_legs()

            active = store.execution_ledger.active_states()
            self.assertFalse(worker.ce_pos.active)
            self.assertFalse(worker.ce_initial_done)
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].spec.role, "C")
            self.assertEqual(active[0].confirmed_live_quantity, 25)
            self.assertEqual(active[0].remaining_quantity, 50)

            worker._enter_both_legs()

        buy_quantities = [qty for (_symbol, side, qty) in client.calls if side == "BUY"]
        self.assertEqual(buy_quantities, [75, 50, 75])
        self.assertTrue(worker.ce_pos.active)
        self.assertTrue(worker.ce_pos.live_leg.entry_complete)
        self.assertTrue(worker.pe_pos.active)
        self.assertTrue(worker.pe_pos.live_leg.entry_complete)
        self.assertEqual(
            worker.ce_pos.live_leg.spec.correlation_id,
            worker.pe_pos.live_leg.spec.correlation_id,
        )

    def test_paper_fallback_leg_exit_is_tagged_paper_fallback(self):
        """Codex on PR #47 (propagated to this stacked PR): a paper-fallback
        LongStrangle leg exit sends no broker order, so its EXIT event must read
        PAPER_FALLBACK, not LIVE."""
        worker, store = self._make_worker()
        worker.live_trading = True
        events = MagicMock()
        worker.trade_event_queue = events
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22510.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 2002): 90.0,
        })
        with patch.object(master_file, "execution_client", _FakeShoonya(fail_on=lambda s, side: True)):
            worker._enter_both_legs()
        self.assertIsNone(worker.ce_pos.live_leg)
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            worker._exit_leg("CE", "TEST_EXIT")
        exit_modes = [c.args[0].get("mode") for c in events.put_nowait.call_args_list
                      if c.args[0].get("action") == "EXIT"]
        self.assertEqual(exit_modes, ["PAPER_FALLBACK"])


@unittest.skipIf(
    getattr(master_file, "SLHuntingAIWorker", None) is None,
    "SL Hunting worker unavailable (claude-agent-sdk / pydantic absent)",
)
class TestSLHuntingBnfMirror(unittest.TestCase):
    """
    The Intraday-Hunter-style BankNIFTY mirror: every NIFTY entry opens an
    equal-LOT-count BankNIFTY ATM leg, and every NIFTY exit closes it (one
    basket). The mirror is fail-soft and must never disturb the NIFTY leg.
    """

    NIFTY_CONTRACT = {
        "security_id": 1001, "exchange_segment": None,  # segment filled in setUp
        "trading_symbol": "NIFTY-24300-CE", "custom_symbol": "NIFTY CE",
        "strike": 24300.0, "option_type": "CE", "expiry_date": None,
        "days_to_expiry": 2, "lot_size": 75, "spot_reference": 24300.0,
        "atm_strike_rounded": 24300.0,
    }
    BNF_CONTRACT = {
        "security_id": 3003, "exchange_segment": None,
        "trading_symbol": "BANKNIFTY-57900-CE", "custom_symbol": "BANKNIFTY CE",
        "strike": 57900.0, "option_type": "CE", "expiry_date": None,
        "days_to_expiry": 20, "lot_size": 35, "spot_reference": 57900.0,
        "atm_strike_rounded": 57900.0,
    }

    def _make_worker(self):
        store = master_file.SharedMarketDataStore()
        broker = MagicMock()
        broker.fetch_ltp_map.return_value = {}
        worker = master_file.SLHuntingAIWorker(
            store=store, stop_event=threading.Event(), broker=broker
        )
        worker._mirror_enabled = True
        nifty_c = dict(self.NIFTY_CONTRACT,
                       exchange_segment=master_file.OPTION_EXCHANGE_SEGMENT,
                       expiry_date=date.today())
        bnf_c = dict(self.BNF_CONTRACT,
                     exchange_segment=master_file.OPTION_EXCHANGE_SEGMENT,
                     expiry_date=date.today() + timedelta(days=20))
        worker.contract_resolver = MagicMock()
        worker.contract_resolver.get_atm_option.return_value = nifty_c
        worker._bnf_resolver = MagicMock()
        worker._bnf_resolver.get_atm_option.return_value = bnf_c
        worker._last_bnf_close = 57910.0
        store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 24300.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 100.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 3003): 500.0,
        })
        return worker, store

    def _expected_nifty_lots(self):
        """Expected lots for this class's canonical 10-pt-stop entry.

        Derived from the master's EFFECTIVE constants (already scaled by
        SL_HUNTING_SIZE_MULTIPLIER at import), so these tests hold under
        whatever multiplier the operator's .env sets, not just the default.
        At defaults: min(floor(2500 / (10*75)), 5) = 3 NIFTY lots.
        The sizing formula itself is covered by risk_sizing's own tests;
        this class only checks the mirror's bookkeeping around it.
        """
        return min(
            int(master_file.SL_HUNTING_RISK_BUDGET // (10 * 75)),
            master_file.SL_HUNTING_MAX_LOTS,
        )

    def test_mirror_opens_with_same_lot_count(self):
        worker, _ = self._make_worker()
        # 10-pt stop -> floor(budget / (10*75)) lots (3 at the default 2500).
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        nifty_lots = worker.pos.quantity // 75
        self.assertEqual(nifty_lots, self._expected_nifty_lots())
        self.assertTrue(worker._mirror_pos.active)
        self.assertEqual(worker._mirror_pos.quantity, nifty_lots * 35)
        self.assertEqual(worker._mirror_pos.option_right, "CE")
        # The mirror asked BankNIFTY's resolver with the BNF spot + direction,
        # pinning the expiry to the monthly-rollover rule (BNF-001).
        worker._bnf_resolver.get_atm_option.assert_called_once_with(
            57910.0, "LONG",
            expiry_date=worker._bnf_resolver.get_monthly_rollover_expiry.return_value,
        )

    def test_hung_inference_cannot_delay_square_off_and_does_not_stack(self):
        """The LLM runs off-loop: cutoff closes exposure while one pass is hung."""
        worker, _ = self._make_worker()
        worker._mirror_enabled = False
        worker._use_bnf = False
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))

        release = threading.Event()
        calls = {"count": 0}

        def _hung_decide(*args, **kwargs):
            calls["count"] += 1
            release.wait(5)
            return MagicMock(action="HOLD", confidence=0, setup="none", stop=0, target=0)

        worker.agent.decide = _hung_decide
        frame = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-07-16 10:00:00")],
                "open": [24300.0], "high": [24305.0], "low": [24295.0], "close": [24300.0],
            }
        )

        poll = threading.Thread(target=worker.process_strategy_frame, args=(frame,))
        poll.start()
        poll.join(timeout=0.5)
        self.assertFalse(poll.is_alive())

        later = frame.copy()
        later["timestamp"] = pd.Timestamp("2026-07-16 10:01:00")
        worker.process_strategy_frame(later)
        self.assertEqual(calls["count"], 1)

        square_off = threading.Thread(target=worker.handle_square_off_and_stop)
        square_off.start()
        square_off.join(timeout=0.5)
        self.assertFalse(square_off.is_alive())
        self.assertFalse(worker.pos.active)
        release.set()

    def test_basket_exits_together_and_pnl_includes_both_legs(self):
        worker, store = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        nifty_qty = worker.pos.quantity
        bnf_qty = worker._mirror_pos.quantity
        # NIFTY option +10, BNF option +20.
        store.update_ltp_map({
            (master_file.OPTION_EXCHANGE_SEGMENT, 1001): 110.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 3003): 520.0,
        })
        expected = 10.0 * nifty_qty + 20.0 * bnf_qty
        self.assertAlmostEqual(worker._get_open_position_pnl(), expected)
        worker.exit_position("AI_TARGET")
        self.assertFalse(worker.pos.active)
        self.assertFalse(worker._mirror_pos.active)
        self.assertAlmostEqual(worker.realized_pnl, expected)

    def test_mirror_failure_never_blocks_the_nifty_leg(self):
        worker, _ = self._make_worker()
        worker._bnf_resolver.get_atm_option.side_effect = ValueError("no BNF chain")
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        self.assertTrue(worker.pos.active)
        self.assertFalse(worker._mirror_pos.active)

    def test_mirror_skipped_without_bnf_spot(self):
        worker, _ = self._make_worker()
        worker._last_bnf_close = 0.0
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        self.assertFalse(worker._mirror_pos.active)

    def test_mirror_disabled_flag_trades_nifty_only(self):
        worker, _ = self._make_worker()
        worker._mirror_enabled = False
        self.assertTrue(worker.enter_position("SHORT", 24300.0, 24310.0, 24200.0))
        self.assertTrue(worker.pos.active)
        self.assertFalse(worker._mirror_pos.active)
        worker._bnf_resolver.get_atm_option.assert_not_called()

    def test_real_leg_uses_banknifty_underlying_for_mirror(self):
        """_place_real_leg must resolve the mirror's broker symbol as BANKNIFTY."""
        worker, _ = self._make_worker()
        captured = []
        original_place_real_leg = worker._place_real_leg

        def capturing_real_leg(side, leg, *, opens_exposure):
            captured.append((side, dict(leg)))
            return original_place_real_leg(
                side,
                leg,
                opens_exposure=opens_exposure,
            )

        worker._place_real_leg = capturing_real_leg
        # Exit orders are now gated on live_legs_open, which is only set True when the
        # entry ran live and confirmed. Run this worker live so both legs' exits fire
        # and we can assert the BANKNIFTY underlying on every real call.
        worker.live_trading = True
        with (
            patch.object(master_file, "execution_client", _FakeShoonya()),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
            worker.exit_position("AI_TARGET")
        mirror_legs = [(s, leg) for s, leg in captured if leg.get("underlying") == "BANKNIFTY"]
        self.assertEqual([s for s, _ in mirror_legs], ["BUY", "SELL"])
        # The NIFTY legs carry no underlying override (default applies).
        nifty_legs = [(s, leg) for s, leg in captured if "underlying" not in leg]
        self.assertEqual([s for s, _ in nifty_legs], ["BUY", "SELL"])

    # ----- Per-leg exit independence (premise-invalidation) -----------------
    def test_exit_nifty_leg_only_keeps_mirror(self):
        """Cutting the NIFTY leg alone must leave the BankNIFTY mirror running."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        self.assertTrue(worker._mirror_pos.active)
        worker.exit_nifty_leg_only("nifty_premise_dead")
        self.assertFalse(worker.pos.active)
        self.assertTrue(worker._mirror_pos.active)     # mirror rides on
        self.assertFalse(worker._suppress_mirror_close)  # flag reset

    def test_exit_bnf_mirror_only_keeps_nifty(self):
        """Cutting the mirror alone must leave the NIFTY leg running."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        worker.exit_bnf_mirror_only("bnf_premise_dead")
        self.assertFalse(worker._mirror_pos.active)
        self.assertTrue(worker.pos.active)             # NIFTY rides on

    def test_executor_routes_exit_leg(self):
        """The MasterWorkerExecutor routes NIFTY/BNF/BOTH to the right leg."""
        ex_mod = master_file.SL_HUNTING_EXECUTOR_MODULE
        if ex_mod is None:
            self.skipTest("SL Hunting executor module not loaded in this environment.")
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        ex = ex_mod.MasterWorkerExecutor(worker)
        # BNF leg only.
        res = ex.exit("bnf gone", 24300.0, leg="BNF")
        self.assertTrue(res["accepted"])
        self.assertFalse(worker._mirror_pos.active)
        self.assertTrue(worker.pos.active)
        # BNF again with no mirror -> clean reject.
        res2 = ex.exit("bnf gone", 24300.0, leg="BNF")
        self.assertFalse(res2["accepted"])
        # BOTH now closes the surviving NIFTY leg.
        res3 = ex.exit("all done", 24300.0, leg="BOTH")
        self.assertTrue(res3["accepted"])
        self.assertFalse(worker.pos.active)

    def test_square_off_sweeps_lone_mirror(self):
        """After a NIFTY-only cut, the 15:15 square-off must still close the orphan mirror."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        worker.exit_nifty_leg_only("nifty_premise_dead")
        self.assertTrue(worker._mirror_pos.active)
        worker.handle_square_off_and_stop()
        self.assertFalse(worker._mirror_pos.active)

    def test_max_loss_sweeps_lone_mirror(self):
        """Same orphan sweep on a max-loss breach."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        worker.exit_nifty_leg_only("nifty_premise_dead")
        worker.handle_max_loss_and_stop(-9999.0, -9999.0)
        self.assertFalse(worker._mirror_pos.active)

    def test_mirror_snapshot_exposes_the_leg(self):
        """position_state must be able to see the mirror as its own leg."""
        worker, _ = self._make_worker()
        self.assertIsNone(worker.mirror_snapshot())
        worker.enter_position("SHORT", 24300.0, 24310.0, 24200.0)
        snap = worker.mirror_snapshot()
        self.assertEqual(snap["underlying"], "BANKNIFTY")
        self.assertEqual(snap["direction"], "SHORT")
        self.assertIn("unrealized_pnl", snap)

    # ----- P1: a lone mirror must not read as "flat" ------------------------
    def _executor(self, worker):
        ex_mod = master_file.SL_HUNTING_EXECUTOR_MODULE
        if ex_mod is None:
            self.skipTest("SL Hunting executor module not loaded in this environment.")
        return ex_mod.MasterWorkerExecutor(worker)

    def test_lone_mirror_reads_as_in_position(self):
        """After a NIFTY-only cut, position_state must report in_position (not flat)."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        worker.exit_nifty_leg_only("nifty_premise_dead")
        snap = self._executor(worker).snapshot()
        self.assertTrue(snap["in_position"])
        self.assertIn("mirror", snap)
        self.assertTrue(snap.get("nifty_leg_flat"))

    def test_entry_rejected_while_lone_mirror_open(self):
        """A fresh entry must be refused while a lone BankNIFTY mirror is still open."""
        worker, _ = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        worker.exit_nifty_leg_only("nifty_premise_dead")
        res = self._executor(worker).enter("LONG", 24290.0, 24400.0, "new setup", 24300.0)
        self.assertFalse(res["accepted"])
        self.assertIn("mirror", res["reason"].lower())
        self.assertTrue(worker._mirror_pos.active)  # unchanged

    # ----- P2: journal close defers until BOTH legs are flat ----------------
    def test_journal_close_deferred_until_mirror_closes(self):
        """A NIFTY-only cut must NOT close the journal row until the mirror closes, so
        option_pnl reflects the whole basket (both legs)."""
        worker, store = self._make_worker()
        worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        # Simulate an open journal row (process_strategy_frame opens it live).
        worker._journal = MagicMock()
        worker._open_trade_id = "t1"
        worker._entry_realized_pnl = 0.0
        worker.exit_nifty_leg_only("nifty_premise_dead")
        worker._journal.close_trade.assert_not_called()   # deferred
        self.assertIsNotNone(worker._pending_journal_exit)
        self.assertEqual(worker._open_trade_id, "t1")
        # Mirror option +20 -> a positive mirror leg; then close it.
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 520.0})
        worker.exit_bnf_mirror_only("bnf_premise_dead")
        worker._journal.close_trade.assert_called_once()
        _tid, payload = worker._journal.close_trade.call_args[0]
        self.assertEqual(_tid, "t1")
        self.assertGreater(payload["option_pnl"], 0.0)    # mirror P&L is captured
        self.assertIsNone(worker._pending_journal_exit)
        self.assertIsNone(worker._open_trade_id)

    # ----- P1: a paper-fallback mirror must not phantom-short (sibling of PR #42) --
    def test_mirror_paper_fallback_close_sends_no_real_order(self):
        """If the BankNIFTY mirror BUY fell back to paper (rejected / symbol-master
        miss) the mirror opened no real leg, so closing it must NOT send a real
        SELL -- that would be a phantom BankNIFTY short of an option never bought.
        The NIFTY leg (really open) still sells; the mirror books flatten silently."""
        worker, store = self._make_worker()
        worker.live_trading = True
        # Only the BankNIFTY mirror leg is rejected at the broker; the NIFTY leg fills.
        entry_fake = _FakeShoonya(fail_on=lambda symbol, side: "BANKNIFTY" in symbol)
        with patch.object(master_file, "execution_client", entry_fake):
            worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        self.assertTrue(worker.pos.live_legs_open)           # NIFTY leg really open
        self.assertTrue(worker._mirror_pos.active)           # mirror tracked (paper)
        self.assertFalse(worker._mirror_pos.live_legs_open)  # ...but no real BNF leg

        exit_fake = _FakeShoonya()
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 520.0})
        with patch.object(master_file, "execution_client", exit_fake):
            worker.exit_position("AI_TARGET")
        self.assertFalse(worker._mirror_pos.active)          # mirror books flattened
        self.assertFalse(worker.pos.active)
        bnf_orders = [c for c in exit_fake.calls if "BANKNIFTY" in c[0]]
        self.assertEqual(bnf_orders, [])                     # no phantom BNF short
        # The real NIFTY leg still sold exactly once.
        self.assertEqual([s for (_sym, s, _q) in exit_fake.calls], ["SELL"])

    def test_confirmed_live_mirror_close_sends_real_sell(self):
        """Non-regression: when BOTH legs opened live, closing the basket still
        SELLs the BankNIFTY mirror leg exactly once."""
        worker, _ = self._make_worker()
        worker.live_trading = True
        ok_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", ok_fake):
            worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        self.assertTrue(worker._mirror_pos.live_legs_open)
        exit_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", exit_fake):
            worker.exit_position("AI_TARGET")
        bnf_sells = [c for c in exit_fake.calls if "BANKNIFTY" in c[0] and c[1] == "SELL"]
        self.assertEqual(len(bnf_sells), 1)

    def test_live_mirror_shares_correlation_with_nifty_and_uses_distinct_roles(self):
        """The two broker legs must be recognizable as one strategy basket."""

        worker, _ = self._make_worker()
        worker.live_trading = True
        fake = _FakeShoonya()

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))

        self.assertIsNotNone(worker.pos.live_leg)
        self.assertIsNotNone(worker._mirror_pos.live_leg)
        self.assertEqual(
            worker.pos.live_leg.spec.correlation_id,
            worker._mirror_pos.live_leg.spec.correlation_id,
        )
        self.assertEqual(worker.pos.live_leg.spec.role, "N")
        self.assertEqual(worker._mirror_pos.live_leg.spec.role, "B")

    def test_partial_mirror_entry_stays_tracked_and_closes_only_confirmed_quantity(self):
        """An asymmetric BNF fill must remain reachable by the worker's exit path."""

        class PartialMirrorEntryFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self._partial_bnf_entry_sent = False
                self.status_queries = []

            def place_market_order(
                self,
                symbol,
                side,
                quantity,
                exchange_segment="NFO",
                product_type="INTRADAY",
                *,
                order_tag="",
            ):
                if side == "BUY" and "BANKNIFTY" in symbol and not self._partial_bnf_entry_sent:
                    self._partial_bnf_entry_sent = True
                    self.calls.append((symbol, side, quantity))
                    self.order_tags.append(order_tag)
                    return OrderResult(
                        order_id="BNF-ENTRY-1",
                        requested_quantity=int(quantity),
                        filled_quantity=35,
                        remaining_quantity=int(quantity) - 35,
                        status=OrderStatus.PARTIAL,
                        broker_state="OPEN",
                        reason="simulated asymmetric mirror entry",
                    )
                return super().place_market_order(
                    symbol,
                    side,
                    quantity,
                    exchange_segment,
                    product_type,
                    order_tag=order_tag,
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=int(requested_quantity),
                    filled_quantity=35,
                    remaining_quantity=int(requested_quantity) - 35,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal asymmetric mirror entry",
                )

        worker, _ = self._make_worker()
        worker.live_trading = True
        fake = PartialMirrorEntryFake()

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
            self.assertTrue(worker.pos.active)
            self.assertTrue(worker._mirror_pos.active)
            # MAT-104 floor sizing: same lot count as the NIFTY leg, in BNF
            # units (3 lots x 35 = 105 at the default config); only the fake's
            # single BNF lot (35) is confirmed live.
            mirror_qty = self._expected_nifty_lots() * 35
            self.assertEqual(worker._mirror_pos.quantity, mirror_qty)
            self.assertEqual(worker._mirror_pos.live_leg.confirmed_live_quantity, 35)
            self.assertEqual(worker._mirror_pos.live_leg.risk_quantity, mirror_qty)

            worker.exit_position("ASYMMETRIC_ENTRY_RECOVERY")

        bnf_orders = [
            (side, quantity)
            for symbol, side, quantity in fake.calls
            if "BANKNIFTY" in symbol
        ]
        self.assertEqual(bnf_orders, [("BUY", mirror_qty), ("SELL", 35)])
        self.assertEqual(fake.status_queries, [("BNF-ENTRY-1", mirror_qty)])
        self.assertFalse(worker._mirror_pos.active)

    def test_unknown_mirror_entry_uses_conservative_risk_quantity_for_mtm(self):
        """Zero confirmed units cannot make a possibly filled mirror look harmless."""

        class UnknownMirrorEntryFake(_FakeShoonya):
            def place_market_order(
                self,
                symbol,
                side,
                quantity,
                exchange_segment="NFO",
                product_type="INTRADAY",
                *,
                order_tag="",
            ):
                if side == "BUY" and "BANKNIFTY" in symbol:
                    self.calls.append((symbol, side, quantity))
                    self.order_tags.append(order_tag)
                    return OrderResult(
                        order_id="BNF-UNKNOWN-1",
                        requested_quantity=int(quantity),
                        filled_quantity=0,
                        remaining_quantity=int(quantity),
                        status=OrderStatus.UNKNOWN,
                        broker_state="OPEN",
                        reason="mirror acknowledgement lost",
                    )
                return super().place_market_order(
                    symbol,
                    side,
                    quantity,
                    exchange_segment,
                    product_type,
                    order_tag=order_tag,
                )

        worker, store = self._make_worker()
        worker.live_trading = True
        fake = UnknownMirrorEntryFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))

        mirror = worker._mirror_pos
        self.assertTrue(mirror.active)
        self.assertEqual(mirror.live_leg.confirmed_live_quantity, 0)
        # MAT-104 floor sizing: same lot count as the NIFTY leg, in BNF units
        # (3 lots x 35 = 105 at the default config).
        mirror_qty = self._expected_nifty_lots() * 35
        self.assertEqual(mirror.live_leg.risk_quantity, mirror_qty)
        self.assertEqual(mirror.quantity, mirror_qty)
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 490.0})
        # Adverse MTM counts the FULL intended quantity at -10/unit.
        self.assertEqual(worker._mirror_leg_pnl(), -10.0 * mirror_qty)
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            worker.exit_bnf_mirror_only("UNKNOWN_ENTRY_RECOVERY")
        self.assertTrue(worker._mirror_pos.active)
        self.assertEqual(worker._mirror_pos.quantity, mirror_qty)
        self.assertFalse(
            any(
                side == "SELL" and "BANKNIFTY" in symbol
                for symbol, side, _quantity in fake.calls
            )
        )
        # Possible-but-unconfirmed quantity is counted for adverse MTM only;
        # phantom upside must never mask a max-loss breach on the NIFTY leg.
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 510.0})
        self.assertEqual(worker._mirror_leg_pnl(), 0.0)

    def test_partial_mirror_close_retries_only_remaining_and_defers_journal(self):
        """A partial BNF sell stays owned until only its confirmed remainder closes."""

        class PartialMirrorCloseFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self._partial_bnf_sell_sent = False
                self.status_queries = []

            def place_market_order(
                self,
                symbol,
                side,
                quantity,
                exchange_segment="NFO",
                product_type="INTRADAY",
                *,
                order_tag="",
            ):
                if side == "SELL" and "BANKNIFTY" in symbol and not self._partial_bnf_sell_sent:
                    self._partial_bnf_sell_sent = True
                    self.calls.append((symbol, side, quantity))
                    self.order_tags.append(order_tag)
                    filled = 35
                    return OrderResult(
                        order_id="BNF-EXIT-1",
                        requested_quantity=int(quantity),
                        filled_quantity=filled,
                        remaining_quantity=int(quantity) - filled,
                        status=OrderStatus.PARTIAL,
                        broker_state="OPEN",
                        reason="simulated partial mirror close",
                    )
                return super().place_market_order(
                    symbol,
                    side,
                    quantity,
                    exchange_segment,
                    product_type,
                    order_tag=order_tag,
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=int(requested_quantity),
                    filled_quantity=35,
                    remaining_quantity=int(requested_quantity) - 35,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial mirror close",
                )

        worker, store = self._make_worker()
        worker.live_trading = True
        fake = PartialMirrorCloseFake()
        worker._journal = MagicMock()
        worker._open_trade_id = "partial-mirror"
        worker._entry_realized_pnl = 0.0

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
            original_quantity = worker._mirror_pos.quantity
            exposure_id = worker._mirror_pos.live_leg.exposure_id
            store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 520.0})

            worker.exit_position("AI_TARGET")

            self.assertFalse(worker.pos.active)
            self.assertTrue(worker._mirror_pos.active)
            self.assertEqual(worker._mirror_pos.quantity, original_quantity - 35)
            self.assertEqual(
                worker._mirror_pos.live_leg.confirmed_live_quantity,
                original_quantity - 35,
            )
            worker._journal.close_trade.assert_not_called()
            self.assertIsNotNone(worker._pending_journal_exit)

            worker.exit_bnf_mirror_only("RETRY_PARTIAL_CLOSE")
            worker.exit_bnf_mirror_only("IDEMPOTENT_RETRY")

        bnf_sell_quantities = [
            quantity
            for symbol, side, quantity in fake.calls
            if side == "SELL" and "BANKNIFTY" in symbol
        ]
        self.assertEqual(bnf_sell_quantities, [original_quantity, original_quantity - 35])
        self.assertEqual(fake.status_queries, [("BNF-EXIT-1", original_quantity)])
        self.assertFalse(worker._mirror_pos.active)
        self.assertTrue(store.execution_ledger.get(exposure_id).broker_confirmed_flat)
        worker._journal.close_trade.assert_called_once()
        self.assertIsNone(worker._pending_journal_exit)
        self.assertIsNone(worker._open_trade_id)

    def test_mirror_paper_fallback_close_is_tagged_paper_fallback(self):
        """Codex on PR #47: a paper-fallback mirror close sends no broker SELL, so
        its MIRROR EXIT event must read PAPER_FALLBACK, not LIVE."""
        worker, store = self._make_worker()
        worker.live_trading = True
        events = MagicMock()
        worker.trade_event_queue = events
        with patch.object(master_file, "execution_client",
                          _FakeShoonya(fail_on=lambda symbol, side: "BANKNIFTY" in symbol)):
            worker.enter_position("LONG", 24300.0, 24290.0, 24400.0)
        self.assertFalse(worker._mirror_pos.live_legs_open)
        store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 3003): 520.0})
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            worker.exit_position("AI_TARGET")

        def _is_mirror_exit(ev):
            legs = ev.get("legs") or [{}]
            return ev.get("action") == "EXIT" and "BANKNIFTY" in str(legs[0].get("symbol", ""))

        mirror_exit_modes = [c.args[0].get("mode") for c in events.put_nowait.call_args_list
                             if _is_mirror_exit(c.args[0])]
        self.assertEqual(mirror_exit_modes, ["PAPER_FALLBACK"])

    # ----- BNF-001: the REAL resolver must be able to open the mirror -------
    def test_mirror_opens_with_real_resolver_and_rollover_expiry(self):
        """Integration lock for BNF-001: with a real OptionsContractResolver over
        a synthetic instrument master, the mirror must open a BANKNIFTY contract
        at the current monthly expiry -- rolled to the NEXT month here because
        the near expiry is inside the SL_HUNTING_BNF_MIRROR_ROLLOVER_DAYS window.
        (The other mirror tests mock `_bnf_resolver`, which is exactly how the
        original always-empty-chain bug slipped through.)"""
        import logging

        worker, store = self._make_worker()
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        near = date.today() + timedelta(days=3)    # < 7 days -> must roll
        far = date.today() + timedelta(days=40)
        rows = []
        sec = 30000
        for exp in (near.isoformat(), far.isoformat()):
            for strike in (57800, 57900, 58000):
                for right in ("CE", "PE"):
                    rows.append({
                        "EXCH_ID": "NSE", "SEGMENT": "D", "INSTRUMENT": "OPTIDX",
                        "SYMBOL_NAME": f"BANKNIFTY-{exp}-{strike}-{right}",
                        "DISPLAY_NAME": f"BANKNIFTY {exp} {strike} {right}",
                        "SM_EXPIRY_DATE": exp, "LOT_SIZE": "35",
                        "SECURITY_ID": str(sec), "STRIKE_PRICE": str(strike),
                        "OPTION_TYPE": right, "UNDERLYING_SYMBOL": "BANKNIFTY",
                    })
                    sec += 1
        csv_path = Path(tmpdir.name) / "all_instrument 1.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        worker._bnf_resolver = master_file.OptionsContractResolver(
            underlying="BANKNIFTY",
            instrument_master_glob=str(Path(tmpdir.name) / "all_instrument *.csv"),
            log=logging.getLogger("test_bnf_mirror_real_resolver"),
        )
        # Give every synthetic BNF option a live LTP so whichever row the
        # resolver picks has a price in the shared cache.
        store.update_ltp_map({
            (master_file.OPTION_EXCHANGE_SEGMENT, int(row["SECURITY_ID"])): 500.0
            for row in rows
        })

        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        self.assertTrue(worker._mirror_pos.active)
        self.assertTrue(worker._mirror_pos.symbol.startswith("BANKNIFTY-"))
        self.assertEqual(worker._mirror_pos.option_strike, 57900.0)  # ATM for 57910 spot
        self.assertEqual(worker._mirror_pos.option_right, "CE")
        self.assertEqual(worker._mirror_pos.option_expiry, far)      # rolled past `near`


class TestLiveOrderRouting(unittest.TestCase):
    """
    Drives the single-leg take-trade methods with `live_trading=True` and a fake
    Shoonya client, asserting real orders are routed correctly AND that the paper
    bookkeeping is preserved. Only explicit zero-fill rejection falls back to
    paper; partial or unknown exposure freezes live entry.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        self.worker.contract_resolver = MagicMock()
        self.worker.contract_resolver.get_atm_option.return_value = {
            "security_id": 49081,
            "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": "NIFTY-22500-CE",
            "custom_symbol": "NIFTY 22500 CE",
            "strike": 22500.0,
            "option_type": "CE",
            "expiry_date": date.today() + timedelta(days=7),
            "days_to_expiry": 7,
            "lot_size": 50,
            "spot_reference": 22500.0,
            "atm_strike_rounded": 22500.0,
        }
        self.store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT,
             master_file.NIFTY_INDEX_SECURITY_ID): 22500.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 49081): 100.0,
        })

    def test_paper_mode_places_no_real_order(self):
        """Default (paper) worker never calls the Shoonya client."""
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            self.assertFalse(self.worker.live_trading)
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertEqual(fake.calls, [])
        self.assertTrue(self.worker.pos.active)

    @staticmethod
    def _sides(fake):
        """(side, quantity) pairs in call order, ignoring the resolved symbol."""
        return [(side, qty) for (_sym, side, qty) in fake.calls]

    def test_live_entry_resolves_symbol_places_buy_and_records_position(self):
        fake = _FakeShoonya()
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(ok)
        # The contract was resolved to a Shoonya symbol (CE @ 22500) ...
        self.assertIn(("CE", 22500.0), [(ot, st) for (_u, ot, st) in fake.resolved])
        # ... and the order used that resolved symbol, BUY side, correct qty.
        self.assertEqual(len(fake.calls), 1)
        sym, side, qty = fake.calls[0]
        self.assertTrue(sym.startswith("SHOONYA-"))
        self.assertEqual((side, qty), ("BUY", 50 * self.worker.lots))
        # Paper bookkeeping is preserved in live mode.
        self.assertTrue(self.worker.pos.active)
        self.assertEqual(self.worker.pos.entry_trade_price, 100.0)

    def test_shutdown_request_blocks_entry_before_broker_submission(self):
        """The final execution lock rechecks lifecycle entry permission."""

        fake = _FakeShoonya()
        self.worker.live_trading = True
        self.worker.lifecycle.request_shutdown("Ctrl+C")

        with patch.object(master_file, "execution_client", fake):
            opened = self.worker.enter_position(
                direction="LONG",
                entry_underlying=22500.0,
            )

        self.assertFalse(opened)
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(fake.calls, [])

    def test_live_exit_places_sell_and_realizes_pnl(self):
        fake = _FakeShoonya()
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
            self.store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 130.0})
            self.worker.exit_position("TEST_EXIT")
        self.assertIn(("SELL", 50 * self.worker.lots), self._sides(fake))
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.worker.completed_trades, 1)
        self.assertAlmostEqual(
            self.worker.realized_pnl, (130.0 - 100.0) * (50 * self.worker.lots)
        )

    def test_entry_falls_back_to_paper_on_order_failure(self):
        """An explicit zero-fill rejection still records a paper fallback."""
        fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(ok)             # not skipped
        self.assertTrue(self.worker.pos.active)   # recorded as paper

    def test_partial_live_entry_freezes_and_never_becomes_paper(self):
        """A partial fill freezes every worker and starts broker reconciliation."""

        class ReconciliationFake(_FakeShoonya):
            def __init__(self):
                super().__init__(result_status=OrderStatus.PARTIAL)
                self.status_queries = []
                self.reconciliation_complete = threading.Event()

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=max(1, requested_quantity // 2),
                    remaining_quantity=requested_quantity - max(1, requested_quantity // 2),
                    status=OrderStatus.PARTIAL,
                    broker_state="PARTIAL",
                    reason="still partially filled",
                )

            def list_open_orders(self):
                return BrokerQueryResult.success(())

            def list_open_positions(self):
                self.reconciliation_complete.set()
                return BrokerQueryResult.success(())

        fake = ReconciliationFake()
        events = MagicMock()
        self.worker.live_trading = True
        self.worker.trade_event_queue = events
        second_worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store,
            stop_event=self.stop_event,
            broker=self.broker,
        )
        second_worker.contract_resolver = self.worker.contract_resolver
        second_worker.live_trading = True

        with patch.object(master_file, "execution_client", fake):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
            second_ok = second_worker.enter_position(
                direction="LONG", entry_underlying=22500.0
            )
            self.assertTrue(fake.reconciliation_complete.wait(1.0))

        self.assertFalse(ok)
        self.assertFalse(second_ok)
        self.assertFalse(self.worker.pos.active)
        self.assertTrue(self.worker._live_execution_frozen)
        self.assertEqual(len(fake.calls), 1)
        indeterminate = [
            call.args[0]
            for call in events.put_nowait.call_args_list
            if call.args[0].get("action") == "INDETERMINATE_EXPOSURE"
        ]
        self.assertEqual(len(indeterminate), 1)
        self.assertEqual(indeterminate[0]["status"], "PARTIAL")
        self.assertEqual(fake.status_queries, [("ORD-1", 50 * self.worker.lots)])
        frozen, reason = self.store.execution_safety.entry_freeze_snapshot()
        self.assertTrue(frozen)
        self.assertEqual(reason, "simulated partial outcome")

    def test_terminal_partial_entry_retries_only_the_unfinished_quantity(self):
        """A cancelled partial fill may continue, but only for the exact remainder."""

        class PartialThenFillFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if len(self.calls) == 1:
                    return OrderResult(
                        order_id="ENTRY-1",
                        requested_quantity=quantity,
                        filled_quantity=20,
                        remaining_quantity=quantity - 20,
                        status=OrderStatus.PARTIAL,
                        broker_state="OPEN",
                        reason="entry still working",
                    )
                return OrderResult(
                    order_id="ENTRY-2",
                    requested_quantity=quantity,
                    filled_quantity=quantity,
                    remaining_quantity=0,
                    status=OrderStatus.FILLED,
                    broker_state="COMPLETE",
                    reason="remainder filled",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=20,
                    remaining_quantity=requested_quantity - 20,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial entry",
                )

        fake = PartialThenFillFake()
        self.worker.live_trading = True
        leg = self._leg("CE", 22500.0, 50, "NIFTY-22500-CE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            first = self.worker._place_real_leg("BUY", leg, opens_exposure=True)
            state = leg["live_leg"]
            second = self.worker._place_real_leg("BUY", leg, opens_exposure=True)

        self.assertEqual(first.status, OrderStatus.PARTIAL)
        self.assertEqual(second.status, OrderStatus.FILLED)
        self.assertEqual(fake.status_queries, [("ENTRY-1", 50)])
        self.assertEqual(self._sides(fake), [("BUY", 50), ("BUY", 30)])
        state = self.store.execution_ledger.get(state.exposure_id)
        self.assertEqual(state.confirmed_live_quantity, 50)
        self.assertTrue(state.entry_complete)
        self.assertEqual(len(set(fake.order_tags)), 2)

    def test_terminal_partial_entry_is_force_closed_at_cutoff_without_new_signal(self):
        """A partial entry must retain a shutdown owner even when its signal vanishes."""

        class TerminalPartialThenCloseFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if side == "BUY":
                    filled = 20
                    status = OrderStatus.PARTIAL
                    broker_state = "CANCELLED"
                else:
                    filled = quantity
                    status = OrderStatus.FILLED
                    broker_state = "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted cutoff recovery",
                )

        fake = TerminalPartialThenCloseFake()
        self.worker.live_trading = True
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.assertFalse(
                self.worker.enter_position("LONG", entry_underlying=22500.0)
            )
            self.assertFalse(self.worker.pos.active)
            self.assertEqual(len(self.worker._orphan_live_legs), 1)
            self.worker.handle_square_off_and_stop()

        self.assertEqual(self._sides(fake), [("BUY", 50), ("SELL", 20)])
        self.assertEqual(self.worker._orphan_live_legs, [])
        self.assertEqual(self.store.execution_ledger.active_states(), ())

    def test_rebuilt_entry_uses_original_ledger_target_when_sizing_changes(self):
        """A later smaller signal cannot shrink bookkeeping for an older live leg."""

        class PartialThenFillFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                first = len(self.calls) == 1
                filled = 20 if first else quantity
                return OrderResult(
                    order_id=f"ENTRY-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=OrderStatus.PARTIAL if first else OrderStatus.FILLED,
                    broker_state="CANCELLED" if first else "COMPLETE",
                    reason="scripted target normalization",
                )

        fake = PartialThenFillFake()
        self.worker.live_trading = True
        self.worker.lots = 5
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.assertFalse(
                self.worker.enter_position("LONG", entry_underlying=22500.0)
            )
            self.worker.lots = 1
            self.assertTrue(
                self.worker.enter_position("LONG", entry_underlying=22500.0)
            )

        self.assertEqual(self._sides(fake), [("BUY", 250), ("BUY", 230)])
        self.assertEqual(self.worker.pos.live_leg.spec.target_quantity, 250)
        self.assertEqual(self.worker.pos.quantity, 250)
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_terminal_partial_close_retries_only_confirmed_remaining_exposure(self):
        """A partial close subtracts fills and never re-sends the original quantity."""

        class EntryThenPartialCloseFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if side == "BUY":
                    filled = quantity
                    status = OrderStatus.FILLED
                    state = "COMPLETE"
                elif len([call for call in self.calls if call[1] == "SELL"]) == 1:
                    filled = 12
                    status = OrderStatus.PARTIAL
                    state = "OPEN"
                else:
                    filled = quantity
                    status = OrderStatus.FILLED
                    state = "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=state,
                    reason="simulated quantity-aware close",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=12,
                    remaining_quantity=requested_quantity - 12,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial close",
                )

        fake = EntryThenPartialCloseFake()
        self.worker.live_trading = True
        entry_leg = self._leg("CE", 22500.0, 50, "NIFTY-22500-CE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.worker._place_real_leg("BUY", entry_leg, opens_exposure=True)
            state = entry_leg["live_leg"]
            close_leg = dict(entry_leg)
            close_leg["live_leg"] = state
            first = self.worker._place_real_leg("SELL", close_leg, opens_exposure=False)
            second = self.worker._place_real_leg("SELL", close_leg, opens_exposure=False)

        self.assertEqual(first.status, OrderStatus.PARTIAL)
        self.assertEqual(second.status, OrderStatus.FILLED)
        self.assertEqual(fake.status_queries, [("ORDER-2", 50)])
        self.assertEqual(
            self._sides(fake),
            [("BUY", 50), ("SELL", 50), ("SELL", 38)],
        )
        state = self.store.execution_ledger.get(state.exposure_id)
        self.assertEqual(state.confirmed_live_quantity, 0)
        self.assertTrue(state.broker_confirmed_flat)

    def test_public_single_leg_flow_retains_quantity_through_entry_and_close(self):
        """Strategy-owned state survives rebuilt leg dictionaries and partial retries."""

        class ScriptedQuantityFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                call_number = len(self.calls)
                scripted = {
                    1: ("ENTRY-1", 20, OrderStatus.PARTIAL, "OPEN"),
                    2: ("ENTRY-2", quantity, OrderStatus.FILLED, "COMPLETE"),
                    3: ("EXIT-1", 12, OrderStatus.PARTIAL, "OPEN"),
                    4: ("EXIT-2", quantity, OrderStatus.FILLED, "COMPLETE"),
                }
                order_id, filled, status, broker_state = scripted[call_number]
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason=f"scripted call {call_number}",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                filled = 20 if order_id == "ENTRY-1" else 12
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=filled,
                    remaining_quantity=requested_quantity - filled,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal scripted partial",
                )

        fake = ScriptedQuantityFake()
        self.worker.live_trading = True
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.assertFalse(
                self.worker.enter_position("LONG", entry_underlying=22500.0)
            )
            pending = self.store.execution_ledger.active_states()
            self.assertEqual(len(pending), 1)
            exposure_id = pending[0].exposure_id
            self.assertEqual(pending[0].confirmed_live_quantity, 20)

            self.assertTrue(
                self.worker.enter_position("LONG", entry_underlying=22500.0)
            )
            self.assertTrue(self.worker.pos.active)
            self.assertEqual(self.worker.pos.live_leg.exposure_id, exposure_id)
            self.assertEqual(self.worker.pos.live_leg.confirmed_live_quantity, 50)

            self.worker.exit_position("SCRIPTED_PARTIAL_CLOSE")
            self.assertTrue(self.worker.pos.active)
            self.assertEqual(self.worker.pos.live_leg.confirmed_live_quantity, 38)

            self.worker.exit_position("SCRIPTED_CLOSE_RETRY")

        self.assertFalse(self.worker.pos.active)
        final_state = self.store.execution_ledger.get(exposure_id)
        self.assertTrue(final_state.broker_confirmed_flat)
        self.assertEqual(
            self._sides(fake),
            [("BUY", 50), ("BUY", 30), ("SELL", 50), ("SELL", 38)],
        )
        self.assertEqual(
            fake.status_queries,
            [("ENTRY-1", 50), ("EXIT-1", 50)],
        )

    def test_concurrent_worker_cannot_queue_past_shared_entry_freeze(self):
        """A queued entry must recheck the shared gate after ambiguity is known."""

        class RaceFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.second_submitted = threading.Event()
                self._calls_lock = threading.Lock()

            def place_market_order(self, symbol, side, quantity, **kwargs):
                with self._calls_lock:
                    call_number = len(self.calls) + 1
                    self.calls.append((symbol, side, quantity))
                if call_number == 1:
                    self.first_started.set()
                    self.release_first.wait(1.0)
                    status = OrderStatus.PARTIAL
                    filled = max(1, int(quantity) // 2)
                else:
                    self.second_submitted.set()
                    status = OrderStatus.FILLED
                    filled = int(quantity)
                return OrderResult(
                    order_id=f"RACE-{call_number}",
                    requested_quantity=int(quantity),
                    filled_quantity=filled,
                    remaining_quantity=int(quantity) - filled,
                    status=status,
                    broker_state=status.value,
                    reason=f"race {status.value.lower()}",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                filled = max(1, requested_quantity // 2)
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=filled,
                    remaining_quantity=requested_quantity - filled,
                    status=OrderStatus.PARTIAL,
                    broker_state="PARTIAL",
                    reason="still partial",
                )

            def list_open_orders(self):
                return BrokerQueryResult.success(())

            def list_open_positions(self):
                return BrokerQueryResult.success(())

        second_worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store,
            stop_event=self.stop_event,
            broker=self.broker,
        )
        second_worker.contract_resolver = self.worker.contract_resolver
        self.worker.live_trading = True
        second_worker.live_trading = True
        fake = RaceFake()
        results = {}

        def enter(name, worker):
            results[name] = worker.enter_position(
                direction="LONG",
                entry_underlying=22500.0,
            )

        with patch.object(master_file, "execution_client", fake):
            first = threading.Thread(target=enter, args=("first", self.worker))
            second = threading.Thread(target=enter, args=("second", second_worker))
            first.start()
            self.assertTrue(fake.first_started.wait(0.5))
            second.start()
            try:
                self.assertFalse(fake.second_submitted.wait(0.1))
            finally:
                fake.release_first.set()
                first.join(timeout=1)
                second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(results, {"first": False, "second": False})
        self.assertEqual(len(fake.calls), 1)
        self.assertFalse(self.worker.pos.active)
        self.assertFalse(second_worker.pos.active)

    def test_shutdown_after_attempt_staging_aborts_entry_before_broker_submission(self):
        """The final broker boundary must recheck a concurrently requested stop."""

        fake = _FakeShoonya()
        self.worker.live_trading = True
        attempt_staged = threading.Event()
        release_attempt = threading.Event()
        original_start_attempt = self.store.execution_ledger.start_attempt

        def stage_then_pause(*args, **kwargs):
            handle = original_start_attempt(*args, **kwargs)
            attempt_staged.set()
            release_attempt.wait(1.0)
            return handle

        result = {}

        def enter():
            result["entered"] = self.worker.enter_position(
                direction="LONG",
                entry_underlying=22500.0,
            )

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(
                self.store.execution_ledger,
                "start_attempt",
                side_effect=stage_then_pause,
            ),
        ):
            thread = threading.Thread(target=enter)
            thread.start()
            self.assertTrue(attempt_staged.wait(0.5))
            self.worker.request_worker_shutdown("TEST_SHUTDOWN_RACE")
            release_attempt.set()
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, {"entered": False})
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.store.execution_ledger.active_states(), ())

    def test_worker_shutdown_blocks_only_its_own_entries(self):
        """One strategy's shutdown must not freeze the shared account gate.

        A paper strategy's max-loss stop (or the earliest 15:15 square-off)
        blocks new entries for THAT worker through its own lifecycle gate; the
        other live strategies keep trading. The shared freeze stays reserved
        for genuinely account-wide conditions (indeterminate exposure, a
        failed startup audit).
        """

        self.worker.live_trading = True
        self.worker.request_worker_shutdown("MAX_LOSS_BREACH")

        # The shared account-wide gate is untouched...
        frozen, _reason = self.store.execution_safety.entry_freeze_snapshot()
        self.assertFalse(frozen)

        # ...but this worker's own entries are refused at the order boundary.
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            entered = self.worker.enter_position(
                direction="LONG",
                entry_underlying=22500.0,
            )
        self.assertFalse(entered)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.worker.pos.active)

    def test_live_exit_still_reduces_after_entry_gate_is_disabled(self):
        """Turning off future entries must never suppress a known live close."""

        fake = _FakeShoonya()
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            self.assertTrue(
                self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
            )
            self.worker.live_trading = False
            self.worker.exit_position("LIVE_GATE_DISABLED")

        self.assertEqual(
            self._sides(fake),
            [("BUY", 50 * self.worker.lots), ("SELL", 50 * self.worker.lots)],
        )
        self.assertFalse(self.worker.pos.active)

    def test_failed_live_exit_after_gate_disable_keeps_position_open(self):
        """A disabled entry gate must not let a rejected close erase exposure."""

        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            self.assertTrue(
                self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
            )
        self.worker.live_trading = False
        reject_exit = _FakeShoonya(fail_on=lambda _symbol, side: side == "SELL")

        with patch.object(master_file, "execution_client", reject_exit):
            self.worker.exit_position("LIVE_GATE_DISABLED")

        self.assertEqual(self._sides(reject_exit), [("SELL", 50 * self.worker.lots)])
        self.assertTrue(self.worker.pos.active)
        self.assertEqual(self.worker.completed_trades, 0)

    def test_unknown_or_legacy_truthy_result_freezes_without_paper_fallback(self):
        """Response loss and old truthy payloads both fail closed as UNKNOWN."""

        class LegacyTruthyClient(_FakeShoonya):
            def place_market_order(self, *args, **kwargs):
                self.calls.append((kwargs["symbol"], kwargs["side"], kwargs["quantity"]))
                return {"stat": "Ok", "norenordno": "LEGACY"}

        for fake in (
            _FakeShoonya(result_status=OrderStatus.UNKNOWN),
            LegacyTruthyClient(),
        ):
            with self.subTest(fake=type(fake).__name__):
                worker = master_file.AtmSingleLegStrategyWorker(
                    store=self.store,
                    stop_event=self.stop_event,
                    broker=self.broker,
                )
                worker.contract_resolver = self.worker.contract_resolver
                worker.live_trading = True
                with patch.object(master_file, "execution_client", fake):
                    ok = worker.enter_position(
                        direction="LONG",
                        entry_underlying=22500.0,
                    )
                self.assertFalse(ok)
                self.assertFalse(worker.pos.active)
                self.assertTrue(worker._live_execution_frozen)

    def test_entry_falls_back_to_paper_on_symbol_resolution_miss(self):
        """If the Shoonya symbol can't be resolved, no order is sent; paper fallback."""
        fake = _FakeShoonya(resolve_returns="")  # scrip-master miss
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(ok)
        self.assertEqual(fake.calls, [])          # never attempted an order
        self.assertTrue(self.worker.pos.active)   # recorded as paper

    def test_failed_live_exit_keeps_position_open(self):
        """A failed/unconfirmed LIVE exit must NOT flatten - the real position is
        still open, so we keep it active for retry rather than going flat."""
        ok_fake = _FakeShoonya()
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", ok_fake):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(self.worker.pos.active)
        fail_fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        with patch.object(master_file, "execution_client", fail_fake):
            self.worker.exit_position("TEST_EXIT")
        self.assertTrue(self.worker.pos.active)            # kept OPEN for retry
        self.assertEqual(self.worker.completed_trades, 0)  # not booked as completed

    def test_unknown_live_exit_is_not_resubmitted_before_reconciliation(self):
        """Response loss on an exit must not trigger a duplicate full-size SELL."""

        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        unknown = _FakeShoonya(result_status=OrderStatus.UNKNOWN)

        with patch.object(master_file, "execution_client", unknown):
            self.worker.exit_position("TEST_EXIT")
            self.worker.exit_position("TEST_EXIT_RETRY")

        self.assertTrue(self.worker.pos.active)
        self.assertEqual(self._sides(unknown), [("SELL", 50 * self.worker.lots)])

    def test_heikin_reversal_waits_for_confirmed_flat_exit(self):
        """A rejected reversal SELL must not be followed by an opposite BUY."""

        worker = master_file.HeikinAshiStrategyWorker(
            store=self.store,
            stop_event=self.stop_event,
            broker=self.broker,
        )
        worker.contract_resolver = self.worker.contract_resolver
        worker.live_trading = True
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            self.assertTrue(
                worker.enter_position(direction="LONG", entry_underlying=22500.0)
            )

        worker.signal_engine = MagicMock()
        worker.signal_engine.evaluate_candle.return_value = MagicMock(
            action="REVERSE_TO_SHORT",
            exit_reason="REVERSAL_TO_SHORT",
            entry_underlying=22500.0,
        )
        reject_exit = _FakeShoonya(
            fail_on=lambda _symbol, side: side == "SELL"
        )
        with patch.object(master_file, "execution_client", reject_exit):
            worker.process_strategy_frame(pd.DataFrame({"close": [22500.0]}))

        self.assertTrue(worker.pos.active)
        self.assertEqual(self._sides(reject_exit), [("SELL", 50 * worker.lots)])

    def test_heikin_reversal_retries_close_remainder_before_opposite_entry(self):
        """A partial reversal closes exactly the remainder before opening opposite."""

        worker = master_file.HeikinAshiStrategyWorker(
            store=self.store,
            stop_event=self.stop_event,
            broker=self.broker,
        )
        worker.contract_resolver = self.worker.contract_resolver
        worker.live_trading = True
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            self.assertTrue(worker.enter_position("LONG", 22500.0))
        old_exposure_id = worker.pos.live_leg.exposure_id

        class PartialReversalFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                call_number = len(self.calls)
                filled = 20 if call_number == 1 else quantity
                status = OrderStatus.PARTIAL if call_number == 1 else OrderStatus.FILLED
                return OrderResult(
                    order_id=f"REV-{call_number}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state="OPEN" if call_number == 1 else "COMPLETE",
                    reason=f"reversal call {call_number}",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=20,
                    remaining_quantity=requested_quantity - 20,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial reversal close",
                )

            def list_open_orders(self):
                return BrokerQueryResult.success(())

            def list_open_positions(self):
                return BrokerQueryResult.success(())

            def recover_after_reconciliation(self):
                return True

        worker.signal_engine = MagicMock()
        worker.signal_engine.evaluate_candle.return_value = MagicMock(
            action="REVERSE_TO_SHORT",
            exit_reason="REVERSAL_TO_SHORT",
            entry_underlying=22500.0,
        )
        fake = PartialReversalFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            worker.process_strategy_frame(pd.DataFrame({"close": [22500.0]}))
            self.assertTrue(worker.pos.active)
            self.assertEqual(worker.pos.direction, "LONG")
            self.assertEqual(worker.pos.live_leg.confirmed_live_quantity, 30)
            worker.signal_engine.consume_short_setup.assert_not_called()

            worker.process_strategy_frame(pd.DataFrame({"close": [22500.0]}))

        self.assertEqual(
            self._sides(fake),
            [("SELL", 50), ("SELL", 30), ("BUY", 50)],
        )
        self.assertTrue(
            self.store.execution_ledger.get(old_exposure_id).broker_confirmed_flat
        )
        self.assertTrue(worker.pos.active)
        self.assertEqual(worker.pos.direction, "SHORT")
        worker.signal_engine.consume_short_setup.assert_called_once()

    def test_paper_exit_always_flattens(self):
        """Paper mode (or live success) still flattens normally."""
        self.worker.live_trading = False
        self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.worker.exit_position("TEST_EXIT")
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.worker.completed_trades, 1)

    def test_missing_client_falls_back_to_paper(self):
        """live_trading True but client is None -> paper fallback, no crash."""
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", None):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(ok)
        self.assertTrue(self.worker.pos.active)

    def test_paper_fallback_entry_then_exit_sends_no_real_order_but_flattens(self):
        """P1 (single-leg sibling of PR #42 / HEDGE-001): a LIVE worker whose ENTRY
        fell back to paper (rejected order / symbol-master miss) opened no real leg,
        so its exit must NOT send a real SELL -- that would be a naked short of an
        option we never bought at the broker. The exit still flattens the paper
        books (the position is not a real one to keep open for retry)."""
        # Entry rejected at the broker -> position recorded as paper (no live leg).
        reject_fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", reject_fake):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(self.worker.pos.active)             # tracked as paper
        self.assertFalse(self.worker.pos.live_legs_open)    # but no real leg opened

        # A fresh client for the exit must receive ZERO orders...
        exit_fake = _FakeShoonya()
        self.store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 130.0})
        with patch.object(master_file, "execution_client", exit_fake):
            self.worker.exit_position("TEST_EXIT")
        self.assertEqual(exit_fake.calls, [])               # no phantom naked short
        self.assertFalse(self.worker.pos.active)            # ...yet books flattened
        self.assertEqual(self.worker.completed_trades, 1)

    def test_confirmed_live_entry_marks_legs_open_and_exit_sells(self):
        """The invariant's other side (non-regression): a confirmed live entry marks
        live_legs_open True and its exit DOES send the real SELL."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
            self.assertTrue(self.worker.pos.live_legs_open)
            self.store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 130.0})
            self.worker.exit_position("TEST_EXIT")
        self.assertIn(("SELL", 50 * self.worker.lots), self._sides(fake))
        self.assertFalse(self.worker.pos.active)

    def test_paper_fallback_single_leg_exit_is_tagged_paper_fallback(self):
        """Codex on PR #47: a paper-fallback single-leg exit sends no broker order,
        so its EXIT event must read PAPER_FALLBACK, not LIVE."""
        self.worker.live_trading = True
        events = MagicMock()
        self.worker.trade_event_queue = events
        with patch.object(master_file, "execution_client", _FakeShoonya(fail_on=lambda s, side: True)):
            self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.store.update_ltp_map({(master_file.OPTION_EXCHANGE_SEGMENT, 49081): 130.0})
        with patch.object(master_file, "execution_client", _FakeShoonya()):
            self.worker.exit_position("TEST_EXIT")
        exit_modes = [c.args[0].get("mode") for c in events.put_nowait.call_args_list
                      if c.args[0].get("action") == "EXIT"]
        self.assertEqual(exit_modes, ["PAPER_FALLBACK"])

    # --- Hedged helpers: leg dicts as built by the call sites. ---
    def _leg(self, option_type, strike, qty, dhan):
        return {"option_type": option_type, "strike": strike,
                "expiry": date.today() + timedelta(days=2), "quantity": qty,
                "dhan_symbol": dhan}

    def test_hedged_entry_buys_hedge_then_sells_main(self):
        """Hedged entry order: BUY hedge first, then SELL main."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50)])
        self.assertTrue(main_leg["live_leg"].entry_complete)
        self.assertTrue(hedge_leg["live_leg"].entry_complete)
        self.assertEqual(main_leg["live_leg"].spec.role, "M")
        self.assertEqual(hedge_leg["live_leg"].spec.role, "H")
        self.assertEqual(
            main_leg["live_leg"].spec.correlation_id,
            hedge_leg["live_leg"].spec.correlation_id,
        )

    def test_completed_partial_hedge_can_authorize_its_main_companion(self):
        """A resolved role H retry may finish the planned role M basket leg."""

        class PartialHedgeThenFillFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                first = len(self.calls) == 1
                filled = 10 if first else quantity
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=OrderStatus.PARTIAL if first else OrderStatus.FILLED,
                    broker_state="OPEN" if first else "COMPLETE",
                    reason="scripted companion authorization",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=10,
                    remaining_quantity=requested_quantity - 10,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal first hedge partial",
                )

        fake = PartialHedgeThenFillFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            first = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            second = self.worker._place_real_hedged_entry(main_leg, hedge_leg)

        self.assertEqual(first.status, OrderStatus.PARTIAL)
        self.assertEqual(second.status, OrderStatus.FILLED)
        self.assertEqual(
            self._sides(fake),
            [("BUY", 25), ("BUY", 15), ("SELL", 50)],
        )
        self.assertTrue(main_leg["live_leg"].entry_complete)
        self.assertTrue(hedge_leg["live_leg"].entry_complete)
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_companion_exception_cannot_bypass_another_baskets_freeze(self):
        """Freeze attribution, not merely terminality, controls companion orders."""

        class CrossBasketFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                is_other_partial = "22500" in symbol
                filled = 20 if is_other_partial else quantity
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=OrderStatus.PARTIAL if is_other_partial else OrderStatus.FILLED,
                    broker_state="CANCELLED" if is_other_partial else "COMPLETE",
                    reason="scripted cross-basket freeze",
                )

        other = master_file.AtmSingleLegStrategyWorker(
            store=self.store,
            stop_event=self.stop_event,
            broker=self.broker,
        )
        self.worker.live_trading = True
        other.live_trading = True
        anchor = self._leg("PE", 21000.0, 25, "HEDGE")
        anchor.update({"role": "H", "correlation_id": "BASKET01"})
        other_leg = self._leg("CE", 22500.0, 50, "OTHER")
        other_leg.update({"role": "N", "correlation_id": "OTHER001"})
        companion = self._leg("PE", 22000.0, 50, "MAIN")
        companion.update({"role": "M", "correlation_id": "BASKET01"})
        fake = CrossBasketFake()

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
            patch.object(other, "_start_execution_reconciliation"),
        ):
            self.assertEqual(
                self.worker._place_real_leg("BUY", anchor, opens_exposure=True).status,
                OrderStatus.FILLED,
            )
            self.assertEqual(
                other._place_real_leg("BUY", other_leg, opens_exposure=True).status,
                OrderStatus.PARTIAL,
            )
            result = self.worker._place_real_leg(
                "SELL",
                companion,
                opens_exposure=True,
            )

        self.assertEqual(result.status, OrderStatus.UNKNOWN)
        self.assertEqual(
            self._sides(fake),
            [("BUY", 25), ("BUY", 50)],
        )
        self.assertNotIn("live_leg", companion)

    def test_hedged_entry_partial_main_keeps_protective_hedge(self):
        """A partially opened short keeps its confirmed protective long hedge."""

        fake = _FakeShoonya(
            result_status=lambda symbol, side: (
                OrderStatus.PARTIAL
                if side == "SELL" and "22000" in symbol
                else OrderStatus.FILLED
            )
        )
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
        self.assertEqual(result.status, OrderStatus.PARTIAL)
        self.assertTrue(self.worker._live_execution_frozen)
        # Do not unwind protection while an unknown amount of the short is live.
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50)])

    def test_hedged_entry_retries_only_terminal_partial_main_remainder(self):
        """A full hedge is never rebought while the short finishes its remainder."""

        class TerminalPartialMainFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                main_calls = [call for call in self.calls if call[1] == "SELL"]
                if side == "SELL" and len(main_calls) == 1:
                    filled, status, broker_state = 20, OrderStatus.PARTIAL, "OPEN"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted asymmetric hedge entry",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=20,
                    remaining_quantity=requested_quantity - 20,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial main",
                )

        fake = TerminalPartialMainFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            first = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            second = self.worker._place_real_hedged_entry(main_leg, hedge_leg)

        self.assertEqual(first.status, OrderStatus.PARTIAL)
        self.assertEqual(second.status, OrderStatus.FILLED)
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50), ("SELL", 30)])
        self.assertEqual(fake.status_queries, [("ORDER-2", 50)])
        self.assertTrue(main_leg["live_leg"].entry_complete)
        self.assertEqual(main_leg["live_leg"].confirmed_live_quantity, 50)
        self.assertEqual(hedge_leg["live_leg"].confirmed_live_quantity, 25)

    def test_rejected_main_remainder_cannot_become_paper_fallback(self):
        """A later zero-fill reject cannot hide the short quantity filled earlier."""

        class PartialThenRejectFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                main_calls = [call for call in self.calls if call[1] == "SELL"]
                if side == "SELL" and len(main_calls) == 1:
                    filled, status, broker_state = 20, OrderStatus.PARTIAL, "OPEN"
                elif side == "SELL":
                    filled, status, broker_state = 0, OrderStatus.REJECTED, "REJECTED"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted main remainder rejection",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=20,
                    remaining_quantity=requested_quantity - 20,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal initial partial",
                )

        fake = PartialThenRejectFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)

        self.assertNotEqual(result.status, OrderStatus.REJECTED)
        self.assertFalse(self.worker._entry_outcome_allows_position(result))
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50), ("SELL", 30)])
        self.assertEqual(main_leg["live_leg"].confirmed_live_quantity, 20)
        self.assertEqual(hedge_leg["live_leg"].confirmed_live_quantity, 25)

    def test_rejected_hedge_remainder_cannot_become_paper_fallback(self):
        """A later reject cannot erase a protective hedge's earlier partial fill."""

        class PartialThenRejectHedgeFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if len(self.calls) == 1:
                    filled, status, broker_state = 10, OrderStatus.PARTIAL, "OPEN"
                else:
                    filled, status, broker_state = 0, OrderStatus.REJECTED, "REJECTED"
                return OrderResult(
                    order_id=f"HEDGE-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted hedge remainder rejection",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=10,
                    remaining_quantity=requested_quantity - 10,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal initial hedge partial",
                )

        fake = PartialThenRejectHedgeFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)

        self.assertNotEqual(result.status, OrderStatus.REJECTED)
        self.assertFalse(self.worker._entry_outcome_allows_position(result))
        self.assertEqual(self._sides(fake), [("BUY", 25), ("BUY", 15)])
        self.assertEqual(hedge_leg["live_leg"].confirmed_live_quantity, 10)
        self.assertNotIn("live_leg", main_leg)

    def test_hedged_entry_rejected_main_unwinds_hedge(self):
        """A known zero-fill main rejection safely unwinds the filled hedge."""

        fake = _FakeShoonya(
            fail_on=lambda symbol, side: side == "SELL" and "22000" in symbol
        )
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        # BUY hedge (filled) -> SELL main (rejected) -> SELL hedge (filled unwind).
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50), ("SELL", 25)])

    def test_hedged_entry_missing_main_symbol_unwinds_hedge(self):
        """A locally unsubmitted main leg must not strand the filled hedge."""

        class MissingMainSymbolFake(_FakeShoonya):
            def resolve_option_symbol(
                self, underlying, expiry, option_type, strike, exchange_segment="NFO"
            ):
                self.resolved.append((underlying, option_type, float(strike)))
                if float(strike) == 22000.0:
                    return ""
                return f"SHOONYA-{underlying}-{int(strike)}-{option_type}"

        fake = MissingMainSymbolFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")

        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(main_leg, hedge_leg)

        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 25)])
        self.assertTrue(hedge_leg["live_leg"].broker_confirmed_flat)
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_hedged_exit_buys_main_sells_hedge(self):
        """Hedged exit: BUY-to-close main, SELL-to-close hedge (real legs open)."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            fake.calls.clear()
            result = self.worker._place_real_hedged_exit(main_leg, hedge_leg)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(self._sides(fake), [("BUY", 50), ("SELL", 25)])
        self.assertTrue(main_leg["live_leg"].broker_confirmed_flat)
        self.assertTrue(hedge_leg["live_leg"].broker_confirmed_flat)

    def test_hedged_exit_partial_hedge_does_not_rebuy_closed_main(self):
        """A terminal partial hedge retries its remainder without rebuying main."""

        class PartialHedgeExitFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []
                self.exit_started = False

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if self.exit_started and side == "SELL" and quantity == 25:
                    filled, status, broker_state = 10, OrderStatus.PARTIAL, "OPEN"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted asymmetric hedge exit",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=10,
                    remaining_quantity=requested_quantity - 10,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial hedge close",
                )

        fake = PartialHedgeExitFake()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")

        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.worker._place_real_hedged_entry(main_leg, hedge_leg)
            fake.calls.clear()
            fake.exit_started = True
            first = self.worker._place_real_hedged_exit(main_leg, hedge_leg)
            second = self.worker._place_real_hedged_exit(main_leg, hedge_leg)

        self.assertEqual(first.status, OrderStatus.PARTIAL)
        self.assertEqual(second.status, OrderStatus.FILLED)
        self.assertEqual(self._sides(fake), [("BUY", 50), ("SELL", 25), ("SELL", 15)])
        self.assertEqual(fake.status_queries, [("ORDER-2", 25)])
        self.assertTrue(main_leg["live_leg"].broker_confirmed_flat)
        self.assertTrue(hedge_leg["live_leg"].broker_confirmed_flat)

    def test_hedged_exit_sends_no_orders_for_paper_fallback_position(self):
        """A live worker whose entry fell back to paper (live_legs_open False) must
        NOT send closing orders -- there are no real legs, and a BUY main / SELL
        hedge here would open phantom exposure (P1, Codex on PR #42)."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_exit(main_leg, hedge_leg)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(fake.calls, [])  # but nothing was sent to the broker


class TestOrphanHedgeLegRecovery(unittest.TestCase):
    """
    HEDGE-001: when a hedged entry DOUBLE-fails live (the hedge BUY filled, then
    the main SELL and the unwind SELL both failed), a real bought option is open
    with no paper record. Recovery used to be a log line + Telegram alert only.
    The worker must REMEMBER the orphan and keep trying to close it itself -- on
    a slow poll cadence, and one final forced attempt at the daily shutdown --
    so an unnoticed alert cannot leave real money bleeding all afternoon.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.worker = master_file.AtmSingleLegStrategyWorker(
            store=self.store, stop_event=threading.Event(), broker=MagicMock()
        )
        self.worker.live_trading = True
        self.main_leg = {"option_type": "PE", "strike": 22000.0,
                         "expiry": date.today() + timedelta(days=2), "quantity": 50,
                         "dhan_symbol": "MAIN"}
        self.hedge_leg = {"option_type": "PE", "strike": 21000.0,
                          "expiry": date.today() + timedelta(days=2), "quantity": 25,
                          "dhan_symbol": "HEDGE"}

    @staticmethod
    def _sides(fake):
        return [(side, qty) for (_sym, side, qty) in fake.calls]

    def _double_fail_entry(self):
        """Drive the double failure: hedge BUY fills, every SELL is rejected."""
        fake = _FakeShoonya(fail_on=lambda symbol, side: side == "SELL")
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)
        self.assertEqual(result.status, OrderStatus.UNKNOWN)
        return fake

    def test_double_failure_records_the_orphaned_hedge_leg(self):
        self._double_fail_entry()
        self.assertEqual(len(self.worker._orphan_live_legs), 1)
        orphan = self.worker._orphan_live_legs[0]
        self.assertEqual(orphan["quantity"], 25)       # the bought hedge, not the main
        self.assertEqual(orphan["strike"], 21000.0)
        self.assertEqual(orphan["live_leg"].confirmed_live_quantity, 25)
        self.assertFalse(orphan["live_leg"].broker_confirmed_flat)

    def test_orphan_reconciliation_never_reuses_the_filled_entry_order_id(self):
        """A rejected close with no ID must not poll the earlier filled BUY.

        The reconciliation probe is bound to the latest ledger attempt.  If its
        synthetic result carries the hedge entry's order ID, a FILLED status for
        that BUY can be applied to the rejected SELL attempt and falsely subtract
        the entire live hedge from local exposure.
        """

        class MissingRejectionIdsFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                del kwargs
                self.calls.append((symbol, side, quantity))
                is_hedge_entry = side == "BUY"
                return OrderResult(
                    order_id="HEDGE-ENTRY" if is_hedge_entry else "",
                    requested_quantity=quantity,
                    filled_quantity=quantity if is_hedge_entry else 0,
                    remaining_quantity=0 if is_hedge_entry else quantity,
                    status=OrderStatus.FILLED if is_hedge_entry else OrderStatus.REJECTED,
                    broker_state="COMPLETE" if is_hedge_entry else "REJECTED",
                    reason="scripted missing rejection order id",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=requested_quantity,
                    remaining_quantity=0,
                    status=OrderStatus.FILLED,
                    broker_state="COMPLETE",
                    reason="filled hedge entry status",
                )

        class ImmediateThread:
            def __init__(self, *, target, **kwargs):
                del kwargs
                self._target = target

            def start(self):
                self._target()

        fake = MissingRejectionIdsFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(master_file.threading, "Thread", ImmediateThread),
        ):
            result = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)

        self.assertEqual(result.status, OrderStatus.UNKNOWN)
        self.assertEqual(result.order_id, "")
        self.assertEqual(fake.status_queries, [])
        orphan = self.worker._orphan_live_legs[0]["live_leg"]
        self.assertEqual(orphan.confirmed_live_quantity, 25)
        self.assertFalse(orphan.broker_confirmed_flat)

    def test_single_failure_with_clean_unwind_records_nothing(self):
        # Only the MAIN SELL fails (strike 22000); the unwind SELL succeeds.
        fake = _FakeShoonya(fail_on=lambda symbol, side: side == "SELL" and "22000" in symbol)
        with patch.object(master_file, "execution_client", fake):
            result = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_sweep_retries_and_closes_orphan_when_broker_recovers(self):
        self._double_fail_entry()
        events = MagicMock()
        self.worker.trade_event_queue = events
        ok_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", ok_fake):
            self.worker._sweep_orphan_live_legs(force=True)
        self.assertEqual(self.worker._orphan_live_legs, [])
        self.assertEqual(self._sides(ok_fake), [("SELL", 25)])  # closed the bought leg
        actions = [c.args[0].get("action") for c in events.put_nowait.call_args_list]
        self.assertIn("UNHEDGED_LEG_CLOSED", actions)

    def test_sweep_is_rate_limited_between_retries(self):
        self._double_fail_entry()
        fail_fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        with patch.object(master_file, "execution_client", fail_fake):
            self.worker._sweep_orphan_live_legs(force=True)    # one attempt, fails
            attempts_after_first = len(fail_fake.calls)
            self.worker._sweep_orphan_live_legs()              # inside the cadence -> skipped
        self.assertEqual(len(fail_fake.calls), attempts_after_first)
        self.assertEqual(len(self.worker._orphan_live_legs), 1)   # still tracked

    def test_terminal_partial_orphan_unwind_retries_only_remainder(self):
        """A 10/25 close can retry 15 after cancellation, never the original 25."""

        class PartialThenFillFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if len(self.calls) == 1:
                    filled, status, broker_state = 10, OrderStatus.PARTIAL, "OPEN"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"ORPHAN-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted orphan recovery",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=10,
                    remaining_quantity=requested_quantity - 10,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal partial orphan close",
                )

        self._double_fail_entry()
        partial_fake = PartialThenFillFake()
        self.worker._next_orphan_retry_ts = 0.0

        with (
            patch.object(master_file, "execution_client", partial_fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            self.worker._sweep_orphan_live_legs(force=True)
            self.worker._sweep_orphan_live_legs(force=True)

        self.assertEqual(self._sides(partial_fake), [("SELL", 25), ("SELL", 15)])
        self.assertEqual(partial_fake.status_queries, [("ORPHAN-1", 25)])
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_forced_sweep_keeps_hedge_when_partial_short_cannot_close(self):
        """A failed main BUY-to-close must never expose a naked short basket."""

        class PartialMainCloseRejectFake(_FakeShoonya):
            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if "21000" in symbol and side == "BUY":
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                elif "22000" in symbol and side == "SELL":
                    filled, status, broker_state = 20, OrderStatus.PARTIAL, "CANCELLED"
                elif "22000" in symbol and side == "BUY":
                    filled, status, broker_state = 0, OrderStatus.REJECTED, "REJECTED"
                else:
                    # This is the unsafe protective-hedge SELL the regression
                    # test must prove is never submitted.
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"ORDER-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted correlated orphan recovery",
                )

        fake = PartialMainCloseRejectFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(self.worker, "_start_execution_reconciliation"),
        ):
            result = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)
            self.assertEqual(result.status, OrderStatus.PARTIAL)
            self.assertEqual(len(self.worker._orphan_live_legs), 2)
            self.worker.handle_square_off_and_stop()

        self.assertEqual(
            self._sides(fake),
            [("BUY", 25), ("SELL", 50), ("BUY", 20)],
        )
        states = {
            leg["live_leg"].spec.role: leg["live_leg"]
            for leg in self.worker._orphan_live_legs
        }
        self.assertEqual(states["M"].confirmed_live_quantity, 20)
        self.assertEqual(states["H"].confirmed_live_quantity, 25)
        self.assertEqual(len(self.worker._orphan_live_legs), 2)

    def test_wait_for_next_poll_sweeps_orphans_each_cadence(self):
        self._double_fail_entry()
        self.worker.poll_seconds = 0
        self.worker._next_orphan_retry_ts = 0.0                # cadence elapsed
        ok_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", ok_fake):
            self.worker.wait_for_next_poll()
        self.assertEqual(self.worker._orphan_live_legs, [])

    def test_square_off_shutdown_takes_a_final_forced_attempt(self):
        self._double_fail_entry()
        ok_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", ok_fake):
            self.worker.handle_square_off_and_stop()
        self.assertEqual(self.worker._orphan_live_legs, [])


class TestHedgedPaperFallbackExit(unittest.TestCase):
    """P1 (Codex on PR #42): a live hedged worker whose entry fell back to paper
    (live_legs_open False -- explicit zero-fill rejection or a locally skipped
    order) must NOT send real closing orders at exit. A BUY for the never-opened
    main leg plus a SELL for the already-closed hedge would open phantom live
    exposure. The exit still flattens the paper books."""

    def _make_bullish_worker(self, *, live_legs_open: bool):
        store = master_file.SharedMarketDataStore()
        worker = master_file.SupertrendBullishWorker(
            store=store, stop_event=threading.Event(), broker=MagicMock()
        )
        worker.live_trading = True
        seg = master_file.OPTION_EXCHANGE_SEGMENT
        main_live_leg = None
        hedge_live_leg = None
        if live_legs_open:
            main_leg = {
                "option_type": "PE", "strike": 22000.0,
                "expiry": date.today() + timedelta(days=2), "quantity": 50,
                "dhan_symbol": "NIFTY-22000-PE",
            }
            hedge_leg = {
                "option_type": "PE", "strike": 21000.0,
                "expiry": date.today() + timedelta(days=2), "quantity": 50,
                "dhan_symbol": "NIFTY-21000-PE",
            }
            with patch.object(master_file, "execution_client", _FakeShoonya()):
                worker._place_real_hedged_entry(main_leg, hedge_leg)
            main_live_leg = main_leg["live_leg"]
            hedge_live_leg = hedge_leg["live_leg"]
        worker.pos = master_file.HedgedPaperPosition(
            active=True, direction="LONG",
            main_live_leg=main_live_leg, hedge_live_leg=hedge_live_leg,
            entry_underlying=22500.0,
            main_symbol="NIFTY-22000-PE", main_side="SELL", main_security_id=5001,
            main_exchange_segment=seg, main_right="PE", main_strike=22000.0,
            main_quantity=50, main_entry_price=160.0,
            hedge_symbol="NIFTY-21000-PE", hedge_side="BUY", hedge_security_id=5002,
            hedge_exchange_segment=seg, hedge_right="PE", hedge_strike=21000.0,
            hedge_quantity=50, hedge_entry_price=10.0,
        )
        store.update_ltp_map({(seg, 5001): 120.0, (seg, 5002): 8.0})
        return worker

    def test_paper_fallback_exit_sends_no_real_orders_but_flattens(self):
        worker = self._make_bullish_worker(live_legs_open=False)
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            worker.exit_position("TIME_CUTOFF")
        self.assertEqual(fake.calls, [])            # no phantom broker orders
        self.assertFalse(worker.pos.active)         # paper books flattened

    def test_confirmed_live_exit_still_sends_both_closing_orders(self):
        worker = self._make_bullish_worker(live_legs_open=True)
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            worker.exit_position("TIME_CUTOFF")
        sides = [(side, qty) for (_sym, side, qty) in fake.calls]
        self.assertEqual(sides, [("BUY", 50), ("SELL", 50)])  # BUY main, SELL hedge
        self.assertFalse(worker.pos.active)

    def test_paper_fallback_exit_is_tagged_paper_fallback_not_live(self):
        """Codex on PR #47: the EXIT event for a paper-fallback position must NOT
        claim `LIVE` -- no broker order was sent -- or an operator would think a
        real leg was closed. It must read PAPER_FALLBACK."""
        worker = self._make_bullish_worker(live_legs_open=False)
        events = MagicMock()
        worker.trade_event_queue = events
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            worker.exit_position("TIME_CUTOFF")
        modes = [c.args[0].get("mode") for c in events.put_nowait.call_args_list
                 if c.args[0].get("action") == "EXIT"]
        self.assertEqual(modes, ["PAPER_FALLBACK"])

    def test_confirmed_live_exit_is_tagged_live(self):
        worker = self._make_bullish_worker(live_legs_open=True)
        events = MagicMock()
        worker.trade_event_queue = events
        fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", fake):
            worker.exit_position("TIME_CUTOFF")
        modes = [c.args[0].get("mode") for c in events.put_nowait.call_args_list
                 if c.args[0].get("action") == "EXIT"]
        self.assertEqual(modes, ["LIVE"])

    def test_partial_hedge_close_updates_position_and_retries_only_remainder(self):
        """Public exit keeps asymmetric state until both broker legs are flat."""

        class PartialThenFinishFake(_FakeShoonya):
            def __init__(self):
                super().__init__()
                self.status_queries = []

            def place_market_order(self, symbol, side, quantity, **kwargs):
                self.calls.append((symbol, side, quantity))
                self.order_tags.append(kwargs.get("order_tag", ""))
                if side == "SELL" and len(self.calls) == 2:
                    filled, status, broker_state = 20, OrderStatus.PARTIAL, "OPEN"
                else:
                    filled, status, broker_state = quantity, OrderStatus.FILLED, "COMPLETE"
                return OrderResult(
                    order_id=f"EXIT-{len(self.calls)}",
                    requested_quantity=quantity,
                    filled_quantity=filled,
                    remaining_quantity=quantity - filled,
                    status=status,
                    broker_state=broker_state,
                    reason="scripted public hedged exit",
                )

            def get_order_status(self, order_id, requested_quantity=0):
                self.status_queries.append((order_id, requested_quantity))
                return OrderResult(
                    order_id=order_id,
                    requested_quantity=requested_quantity,
                    filled_quantity=20,
                    remaining_quantity=requested_quantity - 20,
                    status=OrderStatus.PARTIAL,
                    broker_state="CANCELLED",
                    reason="terminal public hedge partial",
                )

        worker = self._make_bullish_worker(live_legs_open=True)
        fake = PartialThenFinishFake()
        with (
            patch.object(master_file, "execution_client", fake),
            patch.object(worker, "_start_execution_reconciliation"),
        ):
            worker.exit_position("FIRST_ATTEMPT")
            self.assertTrue(worker.pos.active)
            self.assertTrue(worker.pos.main_live_leg.broker_confirmed_flat)
            self.assertEqual(worker.pos.hedge_live_leg.confirmed_live_quantity, 30)
            worker.exit_position("SECOND_ATTEMPT")

        self.assertFalse(worker.pos.active)
        self.assertEqual(
            [(side, quantity) for _symbol, side, quantity in fake.calls],
            [("BUY", 50), ("SELL", 50), ("SELL", 30)],
        )
        self.assertEqual(fake.status_queries, [("EXIT-2", 50)])


class TestExecModeTag(unittest.TestCase):
    """`_exec_mode_tag` labels how a trade actually executed (Codex PR #47)."""

    def _worker(self, *, live: bool):
        w = master_file.AtmSingleLegStrategyWorker(
            store=master_file.SharedMarketDataStore(),
            stop_event=threading.Event(), broker=MagicMock(),
        )
        w.live_trading = live
        return w

    @staticmethod
    def _result(status: OrderStatus) -> OrderResult:
        filled = 50 if status is OrderStatus.FILLED else 0
        return OrderResult(
            order_id="TEST",
            requested_quantity=50,
            filled_quantity=filled,
            remaining_quantity=50 - filled,
            status=status,
            broker_state=status.value,
            reason="test result",
        )

    def test_paper_worker_is_always_paper(self):
        w = self._worker(live=False)
        filled = self._result(OrderStatus.FILLED)
        self.assertEqual(w._exec_mode_tag(filled), "PAPER")
        self.assertEqual(w._exec_mode_tag(filled, live_legs_open=False), "PAPER")

    def test_live_worker_labels(self):
        w = self._worker(live=True)
        filled = self._result(OrderStatus.FILLED)
        rejected = self._result(OrderStatus.REJECTED)
        unknown = self._result(OrderStatus.UNKNOWN)
        self.assertEqual(w._exec_mode_tag(filled), "LIVE")
        self.assertEqual(w._exec_mode_tag(rejected), "PAPER_FALLBACK")
        self.assertEqual(w._exec_mode_tag(unknown), "LIVE_INDETERMINATE")
        self.assertEqual(w._exec_mode_tag(rejected, is_exit=True), "LIVE_REJECTED")
        # No live legs open means no broker close was needed.
        self.assertEqual(
            w._exec_mode_tag(filled, live_legs_open=False, is_exit=True),
            "PAPER_FALLBACK",
        )

    def test_disabled_entry_gate_still_labels_known_live_exit_as_live(self):
        """Execution telemetry follows actual exposure, not a mutable entry flag."""

        worker = self._worker(live=False)
        self.assertEqual(
            worker._exec_mode_tag(
                self._result(OrderStatus.FILLED),
                live_legs_open=True,
                is_exit=True,
            ),
            "LIVE",
        )


class TestShoonyaOrderAck(unittest.TestCase):
    """
    NorenApi.place_order preserves both success and error dictionaries. The
    wrapper's `_is_order_ack` must tell a real acknowledgement apart from a
    rejection or indeterminate response.
    """

    def setUp(self):
        self.client = master_file.shoonya_execution_client
        if self.client is None:
            self.skipTest("Shoonya execution layer not importable in this environment.")

    def test_acknowledged_payloads(self):
        self.assertTrue(self.client._is_order_ack(
            {"stat": "Ok", "norenordno": "250612000123"}))
        self.assertTrue(self.client._is_order_ack(
            {"stat": "ok", "norenordno": "ABC123", "request_time": "..."}))

    def test_rejected_or_error_payloads(self):
        for bad in (
            {"stat": "Not_Ok", "emsg": "RMS rejected"},
            {"norenordno": "ABC123"},          # missing stat == Ok
            {"stat": "Ok"},                    # missing norenordno
            {},
            None,
            "oops",
        ):
            self.assertFalse(self.client._is_order_ack(bad), bad)

    def test_generate_totp_blank_secret_returns_empty(self):
        """No secret -> "" (caller then prompts / aborts), never a crash."""
        self.assertEqual(type(self.client)._generate_totp(""), "")
        self.assertEqual(type(self.client)._generate_totp("   "), "")


class _StubNoren:
    """Minimal stand-in for the NorenApi client exposing only single_order_history."""

    def __init__(self, history):
        self._history = history

    def single_order_history(self, orderno=None):
        return self._history


class TestShoonyaFillConfirmation(unittest.TestCase):
    """
    Shoonya's place_order only ACKNOWLEDGES an order; the wrapper must confirm a
    real fill via single_order_history before reporting success.
    """

    def setUp(self):
        # Same guard as TestShoonyaOrderAck: without the optional Shoonya deps
        # the master binds shoonya_execution_client to None, and type(None)()
        # below would blow up instead of skipping.
        if master_file.shoonya_execution_client is None:
            self.skipTest("Shoonya execution layer not importable in this environment.")

    def _client(self, history):
        c = type(master_file.shoonya_execution_client)()  # fresh instance, not the singleton
        c.client = _StubNoren(history)
        c.is_logged_in = True
        return c

    @staticmethod
    def _hist(state, fld=75, qty=75, rej="--"):
        # single_order_history returns a LIST of rows (newest first).
        return [{"status": state, "fillshares": fld, "qty": qty, "rejreason": rej}]

    def test_order_status_parses_latest_row(self):
        c = self._client(self._hist("COMPLETE", fld=50, qty=75))
        state, filled, qty, _reason = c._order_status("ORD1")
        self.assertEqual((state, filled, qty), ("complete", 50, 75))

    def test_confirm_fill_returns_on_complete(self):
        c = self._client(self._hist("COMPLETE", fld=75, qty=75))
        result = c._confirm_fill("ORD1", 75)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.filled_quantity, 75)

    def test_confirm_fill_returns_explicit_rejection(self):
        c = self._client(self._hist("REJECTED", fld=0, qty=75, rej="RMS: margin"))
        result = c._confirm_fill("ORD1", 75)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertEqual(result.filled_quantity, 0)
        self.assertIn("margin", result.reason)

    def test_confirm_fill_timeout_is_unknown(self):
        c = self._client(self._hist("OPEN", fld=0, qty=75))
        mod = type(c).__module__
        smod = sys.modules[mod]
        with patch.object(smod, "_FILL_TIMEOUT_SECONDS", 0.05), \
             patch.object(smod, "_FILL_POLL_INTERVAL", 0.01):
            result = c._confirm_fill("ORD1", 75)
        self.assertEqual(result.status, OrderStatus.UNKNOWN)
        self.assertIn("indeterminate", result.reason.lower())


class TestShoonyaSymbolResolution(unittest.TestCase):
    """
    Shoonya's place_order needs a tsym shaped <UNDERLYING><DDMMMYY><C|P><STRIKE>.
    The resolver BUILDS that string and, when the NFO symbol master is loaded,
    validates membership (returning "" on a miss).
    """

    def _client(self):
        client = master_file.shoonya_execution_client
        if client is None:
            self.skipTest("Shoonya execution layer not importable in this environment.")
        return type(client)()  # fresh instance, not the singleton (clean cache)

    def test_builds_tsym_when_master_not_loaded(self):
        c = self._client()  # _symbol_set is None -> trust the construction
        self.assertEqual(
            c.resolve_option_symbol("NIFTY", date(2026, 6, 25), "CE", 22500),
            "NIFTY25JUN26C22500",
        )
        self.assertEqual(
            c.resolve_option_symbol("NIFTY", date(2026, 6, 25), "PE", 22500),
            "NIFTY25JUN26P22500",
        )

    def test_validates_against_loaded_master(self):
        c = self._client()
        c._symbol_set = {"NIFTY25JUN26C22500"}  # only this contract exists
        self.assertEqual(
            c.resolve_option_symbol("NIFTY", date(2026, 6, 25), "CE", 22500),
            "NIFTY25JUN26C22500",
        )

    def test_resolution_miss_returns_empty(self):
        c = self._client()
        c._symbol_set = {"NIFTY25JUN26C23000"}  # wanted strike absent
        self.assertEqual(
            c.resolve_option_symbol("NIFTY", date(2026, 6, 25), "CE", 22500), ""
        )


class _FakeHttpResponse:
    """Small requests.Response stand-in used by the offline Flattrade tests."""

    def __init__(self, payload, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClock:
    """Controllable monotonic clock so rate-limit tests never really sleep."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TestFlattradeAuthentication(unittest.TestCase):
    """Flattrade login must validate a token or perform the documented code exchange."""

    def setUp(self):
        self.assertIsNotNone(
            flattrade_module,
            "Dependencies/Flattrade API/flattrade_execution.py has not been implemented",
        )
        self.client = flattrade_module.FlattradeExecutionClient()
        self.session = MagicMock()
        self.client.client = self.session

    def test_existing_access_token_is_validated_with_user_details(self):
        self.session.post.return_value = _FakeHttpResponse(
            {"stat": "Ok", "actid": "FT123", "exarr": ["NFO"]}
        )
        env = {
            "FLATTRADE_CLIENT_ID": "FT123",
            "FLATTRADE_ACCESS_TOKEN": "daily-token",
            "FLATTRADE_API_KEY": "",
            "FLATTRADE_API_SECRET": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertTrue(self.client.ensure_logged_in())

        self.assertTrue(self.client.is_logged_in)
        self.assertEqual(self.client._access_token, "daily-token")
        _, kwargs = self.session.post.call_args
        self.assertEqual(kwargs["timeout"], flattrade_module._API_TIMEOUT_SECONDS)
        posted = parse_qs(kwargs["data"])
        self.assertEqual(json.loads(posted["jData"][0]), {"uid": "FT123"})
        self.assertEqual(posted["jKey"], ["daily-token"])
        self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

    def test_browser_request_code_is_hashed_exchanged_and_validated(self):
        self.session.post.side_effect = [
            _FakeHttpResponse({"status": "Ok", "token": "issued-token", "client": "FT123"}),
            _FakeHttpResponse({"stat": "Ok", "actid": "FT123", "exarr": ["NFO"]}),
        ]
        env = {
            "FLATTRADE_CLIENT_ID": "FT123",
            "FLATTRADE_ACCESS_TOKEN": "",
            "FLATTRADE_API_KEY": "public-key",
            "FLATTRADE_API_SECRET": "raw-secret",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(flattrade_module.webbrowser, "open", return_value=True) as open_browser, \
             patch("builtins.input", return_value="request-code"):
            self.assertTrue(self.client.ensure_logged_in())

        expected_hash = hashlib.sha256(
            b"public-keyrequest-coderaw-secret"
        ).hexdigest()
        open_browser.assert_called_once_with(
            "https://auth.flattrade.in/?app_key=public-key"
        )
        token_call = self.session.post.call_args_list[0]
        self.assertEqual(token_call.args[0], flattrade_module._TOKEN_URL)
        self.assertEqual(
            token_call.kwargs["json"],
            {
                "api_key": "public-key",
                "request_code": "request-code",
                "api_secret": expected_hash,
            },
        )
        self.assertEqual(self.client._access_token, "issued-token")

    def test_missing_credentials_fail_closed_without_network(self):
        env = {
            "FLATTRADE_CLIENT_ID": "",
            "FLATTRADE_ACCESS_TOKEN": "",
            "FLATTRADE_API_KEY": "",
            "FLATTRADE_API_SECRET": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertFalse(self.client.ensure_logged_in())
        self.session.post.assert_not_called()

    def test_failed_exchange_logs_no_secret_or_request_code(self):
        self.session.post.return_value = _FakeHttpResponse(
            {"status": "Not_Ok", "emsg": "invalid input"}
        )
        env = {
            "FLATTRADE_CLIENT_ID": "FT123",
            "FLATTRADE_ACCESS_TOKEN": "",
            "FLATTRADE_API_KEY": "public-key",
            "FLATTRADE_API_SECRET": "raw-secret",
        }
        with patch.dict(os.environ, env, clear=False), \
             patch.object(flattrade_module.webbrowser, "open", return_value=True), \
             patch("builtins.input", return_value="request-code"), \
             self.assertLogs(flattrade_module.log, level="ERROR") as captured:
            self.assertFalse(self.client.ensure_logged_in())
        joined = "\n".join(captured.output)
        self.assertNotIn("raw-secret", joined)
        self.assertNotIn("request-code", joined)


class TestFlattradeRateLimits(unittest.TestCase):
    """The client must wait for short bursts and fail before stale minute waits."""

    def setUp(self):
        self.assertIsNotNone(flattrade_module)

    def test_short_per_second_limit_waits_for_a_slot(self):
        clock = _FakeClock()
        limiter = flattrade_module._RollingWindowRateLimiter(
            per_second=2,
            per_minute=10,
            max_wait_seconds=2.0,
            clock=clock.monotonic,
            sleeper=clock.sleep,
            label="test",
        )
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        self.assertTrue(clock.sleeps)
        self.assertGreaterEqual(clock.now, 1.0)

    def test_exhausted_minute_limit_raises_before_http_request(self):
        clock = _FakeClock()
        limiter = flattrade_module._RollingWindowRateLimiter(
            per_second=10,
            per_minute=2,
            max_wait_seconds=0.0,
            clock=clock.monotonic,
            sleeper=clock.sleep,
            label="test",
        )
        limiter.acquire()
        limiter.acquire()

        client = flattrade_module.FlattradeExecutionClient()
        client.client = MagicMock()
        client._access_token = "token"
        client._client_id = "FT123"
        client._api_limiter = limiter
        with self.assertRaises(RuntimeError):
            client._post_api("UserDetails", {"uid": "FT123"})
        client.client.post.assert_not_called()


class TestFlattradeSymbolResolution(unittest.TestCase):
    """Option resolution must use exact rows from Flattrade's official CSV schema."""

    def setUp(self):
        self.assertIsNotNone(flattrade_module)
        self.client = flattrade_module.FlattradeExecutionClient()
        raw = pd.DataFrame(
            [
                {
                    "Exchange": "NFO",
                    "Token": "51377",
                    "Lotsize": "65",
                    "Symbol": "NIFTY",
                    "Tradingsymbol": "NIFTY14JUL26C24150",
                    "Instrument": "OPTIDX",
                    "Expiry": "14-JUL-2026",
                    "Strike": "24150.00",
                    "Optiontype": "CE",
                }
            ]
        )
        self.client._scrip_df = self.client._prepare_scrip_master(raw)

    def test_exact_contract_resolves_and_is_cached(self):
        with patch.object(self.client, "ensure_logged_in", return_value=True):
            symbol = self.client.resolve_option_symbol(
                "NIFTY", date(2026, 7, 14), "CE", 24150
            )
            self.client._scrip_df = pd.DataFrame()
            cached = self.client.resolve_option_symbol(
                "NIFTY", date(2026, 7, 14), "CE", 24150
            )
        self.assertEqual(symbol, "NIFTY14JUL26C24150")
        self.assertEqual(cached, symbol)

    def test_nonexistent_contract_returns_empty(self):
        with patch.object(self.client, "ensure_logged_in", return_value=True):
            self.assertEqual(
                self.client.resolve_option_symbol(
                    "NIFTY", date(2026, 7, 14), "PE", 24150
                ),
                "",
            )

    def test_malformed_master_is_rejected(self):
        with self.assertRaises(ValueError):
            self.client._prepare_scrip_master(pd.DataFrame({"Symbol": ["NIFTY"]}))


class TestFlattradeOrders(unittest.TestCase):
    """Market orders must use Flattrade mappings and confirm the complete fill."""

    def setUp(self):
        self.assertIsNotNone(flattrade_module)
        self.client = flattrade_module.FlattradeExecutionClient()
        self.client._client_id = "FT123"
        self.client._account_id = "FT123"
        self.client._access_token = "token"
        self.client.is_logged_in = True

    def test_market_order_payload_uses_documented_codes(self):
        ack = {"stat": "Ok", "norenordno": "260703000001"}
        filled = OrderResult(
            order_id="260703000001",
            requested_quantity=65,
            filled_quantity=65,
            remaining_quantity=0,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
            reason="simulated fill",
        )
        with patch.dict(os.environ, {"FLATTRADE_MARKET_PROTECTION": "5"}), \
             patch.object(self.client, "ensure_logged_in", return_value=True), \
             patch.object(self.client, "_post_api", return_value=ack) as post_api, \
             patch.object(self.client, "_confirm_fill", return_value=filled) as confirm:
            result = self.client.place_market_order(
                symbol="NIFTY14JUL26C24150",
                side="BUY",
                quantity=65,
                exchange_segment="NFO",
                product_type="INTRADAY",
            )

        self.assertEqual(result, filled)
        endpoint, payload = post_api.call_args.args
        self.assertEqual(endpoint, "PlaceOrder")
        self.assertEqual(
            payload,
            {
                "uid": "FT123",
                "actid": "FT123",
                "exch": "NFO",
                "tsym": "NIFTY14JUL26C24150",
                "qty": "65",
                "prc": "0",
                "dscqty": "0",
                "prd": "I",
                "trantype": "B",
                "prctyp": "MKT",
                "ret": "DAY",
                "ordersource": "API",
                "mkt_protection": "5",
            },
        )
        self.assertTrue(post_api.call_args.kwargs["is_order"])
        confirm.assert_called_once_with("260703000001", 65)

    def test_invalid_order_inputs_raise_before_submission(self):
        with patch.object(self.client, "_post_api") as post_api:
            for kwargs in (
                {"symbol": "X", "side": "HOLD", "quantity": 65},
                {"symbol": "X", "side": "BUY", "quantity": 0},
                {
                    "symbol": "X",
                    "side": "BUY",
                    "quantity": 65,
                    "exchange_segment": "NSE",
                },
                {
                    "symbol": "X",
                    "side": "BUY",
                    "quantity": 65,
                    "product_type": "DELIVERY",
                },
            ):
                with self.assertRaises(ValueError):
                    self.client.place_market_order(**kwargs)
        post_api.assert_not_called()

    def test_acknowledgement_requires_ok_and_order_id(self):
        self.assertTrue(
            self.client._is_order_ack({"stat": "Ok", "norenordno": "ORD1"})
        )
        for bad in (
            {"stat": "Not_Ok", "norenordno": "ORD1"},
            {"stat": "Ok"},
            {"norenordno": "ORD1"},
            None,
        ):
            self.assertFalse(self.client._is_order_ack(bad))

    def test_order_status_parses_single_order_history(self):
        history = [
            {
                "stat": "Ok",
                "status": "COMPLETE",
                "fillshares": "65",
                "qty": "65",
                "rejreason": "",
            }
        ]
        with patch.object(self.client, "_post_api", return_value=history):
            self.assertEqual(
                self.client._order_status("ORD1"),
                ("complete", 65, 65, ""),
            )

    def test_fill_confirmation_handles_complete_rejected_and_timeout(self):
        with patch.object(
            self.client, "_order_status", return_value=("complete", 65, 65, "")
        ):
            complete = self.client._confirm_fill("ORD1", 65)
        self.assertEqual(complete.status, OrderStatus.FILLED)

        with patch.object(
            self.client,
            "_order_status",
            return_value=("rejected", 0, 65, "RMS: margin"),
        ):
            rejected = self.client._confirm_fill("ORD2", 65)
        self.assertEqual(rejected.status, OrderStatus.REJECTED)

        with patch.object(
            self.client, "_order_status", return_value=("open", 0, 65, "")
        ), patch.object(flattrade_module, "_FILL_TIMEOUT_SECONDS", 0.02), \
             patch.object(flattrade_module, "_FILL_POLL_INTERVAL", 0.005):
            unknown = self.client._confirm_fill("ORD3", 65)
        self.assertEqual(unknown.status, OrderStatus.UNKNOWN)
        self.assertIn("indeterminate", unknown.reason.lower())

    def test_recursive_order_id_extraction_and_local_logout(self):
        self.assertEqual(
            self.client.extract_order_id({"data": [{"norenordno": "ORD9"}]}),
            "ORD9",
        )
        self.client.client = MagicMock()
        response = self.client.logout()
        self.assertEqual(response["stat"], "Ok")
        self.assertFalse(self.client.is_logged_in)
        self.assertEqual(self.client._access_token, "")
        self.client.client.close.assert_called_once()


class TestFlattradeDiagnostic(unittest.TestCase):
    """The diagnostic is read-only unless the operator explicitly confirms."""

    def setUp(self):
        self.assertIsNotNone(
            flattrade_diagnostic_module,
            "diagnose_flattrade_symbol.py has not been implemented",
        )
        raw = pd.DataFrame(
            [
                {
                    "Exchange": "NFO",
                    "Token": "1",
                    "Lotsize": "65",
                    "Symbol": "NIFTY",
                    "Tradingsymbol": "NIFTY14JUL26C24150",
                    "Instrument": "OPTIDX",
                    "Expiry": "14-JUL-2026",
                    "Strike": "24150.00",
                    "Optiontype": "CE",
                },
                {
                    "Exchange": "NFO",
                    "Token": "2",
                    "Lotsize": "65",
                    "Symbol": "NIFTY",
                    "Tradingsymbol": "NIFTY21JUL26C24150",
                    "Instrument": "OPTIDX",
                    "Expiry": "21-JUL-2026",
                    "Strike": "24150.00",
                    "Optiontype": "CE",
                },
            ]
        )
        self.client = flattrade_module.FlattradeExecutionClient()
        self.client._scrip_df = self.client._prepare_scrip_master(raw)

    def test_nearest_matching_expiry_and_lot_size_are_selected(self):
        expiry, symbol, lot_size = flattrade_diagnostic_module.select_contract(
            self.client,
            underlying="NIFTY",
            option_type="CE",
            strike=24150,
            requested_expiry=None,
            today=date(2026, 7, 3),
        )
        self.assertEqual(expiry, date(2026, 7, 14))
        self.assertEqual(symbol, "NIFTY14JUL26C24150")
        self.assertEqual(lot_size, 65)

    def test_explicit_expiry_selects_exact_contract(self):
        expiry, symbol, lot_size = flattrade_diagnostic_module.select_contract(
            self.client,
            underlying="NIFTY",
            option_type="CE",
            strike=24150,
            requested_expiry=date(2026, 7, 21),
            today=date(2026, 7, 3),
        )
        self.assertEqual((expiry, symbol, lot_size), (
            date(2026, 7, 21), "NIFTY21JUL26C24150", 65
        ))

    def test_round_trip_aborts_without_exact_yes(self):
        fake_client = MagicMock()
        with patch("builtins.input", return_value="yes"):
            placed = flattrade_diagnostic_module.place_round_trip_test_order(
                fake_client, "NIFTY14JUL26C24150", 65
            )
        self.assertFalse(placed)
        fake_client.place_market_order.assert_not_called()

    def test_real_order_requires_explicit_quantity_before_login(self):
        """The diagnostic never guesses a live quantity from a changing lot size."""

        with patch.object(
            flattrade_diagnostic_module.fe,
            "flattrade_execution_client",
        ) as client:
            result = flattrade_diagnostic_module.main(
                ["CE", "24150", "--place-order"]
            )

        self.assertEqual(result, 2)
        client.preload_scrip_master.assert_not_called()

    def test_round_trip_buys_then_sells_after_confirmation(self):
        fake_client = MagicMock()
        fake_client.place_market_order.side_effect = [
            OrderResult(
                order_id="BUY1", requested_quantity=65, filled_quantity=65,
                remaining_quantity=0, status=OrderStatus.FILLED,
                broker_state="COMPLETE", reason="simulated entry fill",
            ),
            OrderResult(
                order_id="SELL1", requested_quantity=65, filled_quantity=65,
                remaining_quantity=0, status=OrderStatus.FILLED,
                broker_state="COMPLETE", reason="simulated exit fill",
            ),
        ]
        with patch("builtins.input", return_value="YES"):
            placed = flattrade_diagnostic_module.place_round_trip_test_order(
                fake_client, "NIFTY14JUL26C24150", 65
            )
        self.assertTrue(placed)
        self.assertEqual(
            [call.kwargs["side"] for call in fake_client.place_market_order.call_args_list],
            ["BUY", "SELL"],
        )

    def test_unconfirmed_entry_warns_that_a_live_position_may_exist(self):
        fake_client = MagicMock()
        fake_client.place_market_order.side_effect = TimeoutError("fill unknown")
        with patch("builtins.input", return_value="YES"), \
             patch("builtins.print") as printed:
            placed = flattrade_diagnostic_module.place_round_trip_test_order(
                fake_client, "NIFTY14JUL26C24150", 65
            )
        self.assertFalse(placed)
        output = "\n".join(" ".join(map(str, call.args)) for call in printed.call_args_list)
        self.assertIn("MAY BE OPEN", output)


class TestFlattradeMasterIntegration(unittest.TestCase):
    """The master and friendly CLI must expose Flattrade without network access."""

    def test_master_guarded_loads_flattrade_singleton(self):
        self.assertIsNotNone(
            getattr(master_file, "flattrade_execution_client", None)
        )

    def test_master_selector_uses_flattrade_nfo_product_settings(self):
        self.assertTrue(hasattr(master_file, "_select_execution_client"))
        with patch.dict(
            os.environ, {"FLATTRADE_PRODUCT_TYPE": "NORMAL"}, clear=False
        ):
            client, exchange, product = master_file._select_execution_client(
                "FLATTRADE"
            )
        self.assertIs(client, master_file.flattrade_execution_client)
        self.assertEqual((exchange, product), ("NFO", "NORMAL"))

    def test_master_selector_fails_closed_for_unknown_broker(self):
        self.assertTrue(hasattr(master_file, "_select_execution_client"))
        with self.assertLogs(master_file.logging.getLogger(master_file.LOGGER_NAME), level="ERROR"):
            selected = master_file._select_execution_client("FLAT-TYPO")
        self.assertEqual(selected, (None, "", "INTRADAY"))

    def test_algo_cli_maps_flattrade_diagnostic(self):
        import algo

        self.assertEqual(
            algo.BROKER_DIAGNOSTICS.get("flattrade"),
            "Dependencies/Flattrade API/diagnose_flattrade_symbol.py",
        )


# =============================================================================
# CUSTOM TEST RUNNER — per-test pass/fail log + final pretty summary
# =============================================================================
# Goal: when running this file (either via `python file.py` OR `python -m
# unittest file`) always show one line per test with PASS/FAIL/ERROR, and a
# final summary table that counts the totals.
#
# Why this exists:
# - `unittest.main(verbosity=2)` only prints verbose output when the file is
#   run directly. `python -m unittest` defaults to verbosity 1 (dots) unless
#   the user remembers to pass `-v`. The runner below forces verbosity 2 in
#   both invocation modes.
# - The standard `Ran N tests / OK` line is fine, but a per-test status list
#   (saved to disk too) is much more useful when scanning a long run.

class _LoggingTestResult(unittest.TextTestResult):
    """
    `TextTestResult` subclass that also records each test's outcome in a list
    so we can render a final summary table. Verbose per-test output is still
    delegated to the parent class (it prints the `... ok` / `... FAIL` line
    using `verbosity=2`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each entry: {"name": "TestClass.test_method", "status": "PASS" | "FAIL" | "ERROR" | "SKIP"}
        self.outcomes: list[dict] = []

    def _record(self, test, status: str) -> None:
        self.outcomes.append({"name": test.id(), "status": status})

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL")

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP")


def _print_final_summary(result: _LoggingTestResult) -> None:
    """Render a counts table + per-test status list at the end of a run."""
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
    for outcome in result.outcomes:
        counts[outcome["status"]] += 1
    total = sum(counts.values())

    print("\n" + "=" * 72)
    print("PER-TEST RESULTS")
    print("=" * 72)
    for outcome in result.outcomes:
        # Right-pad the status so the test names align in a column.
        print(f"  {outcome['status']:<6} {outcome['name']}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Total tests run : {total}")
    print(f"  Passed          : {counts['PASS']}")
    print(f"  Failed          : {counts['FAIL']}")
    print(f"  Errored         : {counts['ERROR']}")
    print(f"  Skipped         : {counts['SKIP']}")
    print(f"  Overall         : {'OK' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 72)


def _run_with_logging() -> bool:
    """Run every test in this module with verbose output and a summary."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, resultclass=_LoggingTestResult)
    result = runner.run(suite)
    _print_final_summary(result)
    return result.wasSuccessful()


class TestCPRAlgo3StrategyWorker(unittest.TestCase):
    """
    CPR Algo 3 worker WIRING: observation-strike selection, signal dispatch into
    the shared ATM `enter_position` path, "no fetch while in a position", and the
    spot target/stop exit. Algo 3's multi-instrument decision LOGIC is covered by
    the generator's own suite (Signal Generators/CPR Strategy/), so the generator
    decision is stubbed here.
    """

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.CPRAlgo3StrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )
        # Mock the resolver so no instrument-master CSV is touched.
        self.worker.contract_resolver = MagicMock()
        self.worker.contract_resolver.get_current_week_expiry.return_value = (
            date.today() + timedelta(days=3)
        )

        def fake_strike(expiry, strike, right):
            if right == "CE":
                return {
                    "security_id": 1001, "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
                    "trading_symbol": "NIFTY-22400-CE", "custom_symbol": "NIFTY 22400 CE",
                    "strike": float(strike), "option_type": "CE",
                    "expiry_date": expiry, "lot_size": 50,
                }
            return {
                "security_id": 2002, "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
                "trading_symbol": "NIFTY-22600-PE", "custom_symbol": "NIFTY 22600 PE",
                "strike": float(strike), "option_type": "PE",
                "expiry_date": expiry, "lot_size": 50,
            }

        self.worker.contract_resolver.get_option_for_strike.side_effect = fake_strike
        # The trade itself buys the ATM CE/PE (next-next expiry) via the shared path.
        self.worker.contract_resolver.get_atm_option.return_value = {
            "security_id": 49081, "exchange_segment": master_file.OPTION_EXCHANGE_SEGMENT,
            "trading_symbol": "NIFTY-22500-CE", "custom_symbol": "NIFTY 22500 CE",
            "strike": 22500.0, "option_type": "CE",
            "expiry_date": date.today() + timedelta(days=10), "days_to_expiry": 10,
            "lot_size": 50, "spot_reference": 22500.0, "atm_strike_rounded": 22500.0,
        }
        # Seed spot + ATM-option LTPs.
        self.store.update_ltp_map({
            (master_file.NIFTY_INDEX_EXCHANGE_SEGMENT, master_file.NIFTY_INDEX_SECURITY_ID): 22500.0,
            (master_file.OPTION_EXCHANGE_SEGMENT, 49081): 100.0,
        })
        # A non-empty 1-min OHLC frame so the option fetch + spot snapshot are truthy.
        self._dummy = pd.DataFrame([{
            "timestamp": pd.Timestamp("2026-06-25 09:20"),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1,
        }])
        self.broker.fetch_index_1m_ohlc.return_value = self._dummy
        self.store.update("1", self._dummy)

    def test_observation_strikes_picked_around_atm(self):
        self.assertTrue(self.worker._ensure_observation_strikes())
        self.assertEqual(self.worker.itm_ce["security_id"], 1001)
        self.assertEqual(self.worker.itm_pe["security_id"], 2002)
        # CE ~100 ITM below ATM(22500); PE ~100 above. (Spot 22500, offset 100.)
        ce_call = next(c for c in self.worker.contract_resolver.get_option_for_strike.call_args_list
                       if c.args[2] == "CE")
        pe_call = next(c for c in self.worker.contract_resolver.get_option_for_strike.call_args_list
                       if c.args[2] == "PE")
        self.assertEqual(ce_call.args[1], 22400.0)
        self.assertEqual(pe_call.args[1], 22600.0)

    def test_enter_long_dispatches_to_atm_entry(self):
        decision = master_file.CPR_ALGO3_LOGIC.CPRDecision(
            action="ENTER_LONG", strategy_name="CPR_ALGO3", signal_triggered=True,
            entry_underlying=22500.0, stop_underlying=22450.0, target_underlying=22700.0,
        )
        with patch.object(master_file.CPR_ALGO3_LOGIC, "get_latest_nifty_cpr_algo3_signal",
                          return_value=decision):
            self.worker.process_strategy_frame(self._dummy)
        self.assertTrue(self.worker.pos.active)
        self.assertEqual(self.worker.pos.direction, "LONG")
        self.assertEqual(self.worker.pos.option_security_id, 49081)  # bought the ATM leg
        self.assertEqual(self.worker.entry_submit_count, 1)

    def test_no_option_fetch_while_in_position(self):
        """While holding, the worker only checks the exit - it must not re-fetch options."""
        self.worker._ensure_observation_strikes()
        self.broker.fetch_index_1m_ohlc.reset_mock()
        self.worker.enter_position("LONG", 22500.0, 22450.0, 22700.0)
        # A fresh candle whose close has NOT hit the target or stop.
        frame = pd.DataFrame([{
            "timestamp": pd.Timestamp("2026-06-25 09:25"),
            "open": 22500.0, "high": 22510.0, "low": 22490.0, "close": 22500.0, "volume": 1,
        }])
        self.worker.process_strategy_frame(frame)
        self.broker.fetch_index_1m_ohlc.assert_not_called()
        self.assertTrue(self.worker.pos.active)

    def test_spot_target_hit_exits_long(self):
        self.worker.enter_position("LONG", 22500.0, 22450.0, 22700.0)
        self.assertTrue(self.worker.pos.active)
        self.worker._check_spot_target_stop_and_exit(22750.0)  # spot >= target
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.worker.exit_count, 1)
        self.assertEqual(self.worker.completed_trades, 1)

    def test_spot_stop_hit_exits_long(self):
        self.worker.enter_position("LONG", 22500.0, 22450.0, 22700.0)
        self.worker._check_spot_target_stop_and_exit(22440.0)  # spot <= stop
        self.assertFalse(self.worker.pos.active)
        self.assertEqual(self.worker.exit_count, 1)

    def test_paper_by_default(self):
        self.assertFalse(self.worker.live_trading)


class TestStartupLiveExposureWiring(unittest.TestCase):
    """Prove that startup cannot enable a worker before both books are safe."""

    class _Worker:
        def __init__(self, strategy_name: str) -> None:
            self.strategy_name = strategy_name
            # Starting true makes the test prove that the startup helper first
            # clears stale state rather than relying on constructor defaults.
            self.live_trading = True
            self.lots = 1
            self.max_loss = 5500.0
            self.trading_start_hour = 9
            self.trading_start_minute = 15
            self.square_off_hour = 15
            self.square_off_minute = 15

    class _Client:
        def __init__(
            self,
            workers,
            *,
            orders,
            positions,
            login_ok: bool = True,
            preload_ok: bool = True,
        ) -> None:
            self.workers = workers
            self.orders = orders
            self.positions = positions
            self.login_ok = login_ok
            self.preload_ok = preload_ok
            self.calls = []
            self.mutations = []

        def _record_read(self, name: str) -> None:
            # Login, preload and both reconciliation reads must all happen while
            # every strategy is still paper-only.
            if any(worker.live_trading for worker in self.workers):
                raise AssertionError(f"{name} ran after a live worker was enabled")
            self.calls.append(name)

        def ensure_logged_in(self):
            self._record_read("login")
            return self.login_ok

        def preload_scrip_master(self):
            self._record_read("preload")
            return self.preload_ok

        def list_open_orders(self):
            self._record_read("orders")
            return self.orders

        def list_open_positions(self):
            self._record_read("positions")
            return self.positions

        def place_market_order(self, *args, **kwargs):
            self.mutations.append("place_market_order")
            raise AssertionError("startup must not place orders")

        def cancel_order(self, *args, **kwargs):
            self.mutations.append("cancel_order")
            raise AssertionError("startup must not cancel orders")

        def recover(self, *args, **kwargs):
            self.mutations.append("recover")
            raise AssertionError("startup must not recover broker state")

    def _workers_and_store(self):
        workers = [self._Worker("Renko"), self._Worker("EMA")]
        return workers, master_file.SharedMarketDataStore()

    def _live_env(self, name, default=False):
        del default
        return name == "RENKO_LIVE_TRADING"

    def test_clean_books_enable_only_intended_candidates_after_preload(self):
        workers, store = self._workers_and_store()

        def checked_env(name, default=False):
            self.assertTrue(all(not worker.live_trading for worker in workers))
            return self._live_env(name, default)

        client = self._Client(
            workers,
            orders=BrokerQueryResult.success(()),
            positions=BrokerQueryResult.success(()),
            # A symbol-master failure is non-fatal, but the attempt must still
            # finish before either broker book is audited.
            preload_ok=False,
        )
        with patch.object(master_file, "_env_bool", side_effect=checked_env):
            live_count, audit = master_file._configure_startup_live_trading(
                workers,
                store,
                master_live=True,
                client=client,
            )

        self.assertEqual(client.calls, ["login", "preload", "orders", "positions"])
        self.assertEqual([worker.live_trading for worker in workers], [True, False])
        self.assertEqual(live_count, 1)
        self.assertTrue(audit.safe_to_enable_live)
        self.assertIs(store.startup_exposure_audit, audit)
        self.assertEqual(client.mutations, [])

    def test_open_order_blocks_every_candidate_and_queues_safe_alert(self):
        from Dependencies.broker_contract import OpenOrder

        workers, store = self._workers_and_store()
        client = self._Client(
            workers,
            orders=BrokerQueryResult.success(
                (
                    OpenOrder(
                        order_id="SECRET-ORDER-ID",
                        symbol="NIFTY16JUL2622500CE",
                        side="BUY",
                        requested_quantity=50,
                        filled_quantity=10,
                        remaining_quantity=40,
                        broker_state="OPEN",
                    ),
                )
            ),
            positions=BrokerQueryResult.success(()),
        )
        with patch.object(master_file, "_env_bool", side_effect=self._live_env):
            live_count, audit = master_file._configure_startup_live_trading(
                workers,
                store,
                master_live=True,
                client=client,
            )

        self.assertEqual(live_count, 0)
        self.assertTrue(all(not worker.live_trading for worker in workers))
        self.assertFalse(audit.safe_to_enable_live)
        self.assertEqual(client.calls, ["login", "preload", "orders", "positions"])
        self.assertEqual(client.mutations, [])
        self.assertTrue(store.execution_safety.entry_freeze_snapshot()[0])

        event_queue = master_file.queue.Queue()
        self.assertTrue(master_file._enqueue_startup_exposure_alert(audit, event_queue))
        alert = event_queue.get_nowait()
        self.assertEqual(alert["action"], "STARTUP_LIVE_BLOCKED")
        self.assertIn("Broker reported 1 open order.", alert["reason"])
        self.assertNotIn("SECRET-ORDER-ID", repr(alert))

    def test_indeterminate_book_blocks_live_without_broker_mutation(self):
        workers, store = self._workers_and_store()
        client = self._Client(
            workers,
            orders=BrokerQueryResult.indeterminate("token=DO-NOT-LOG"),
            positions=BrokerQueryResult.success(()),
        )
        with patch.object(master_file, "_env_bool", side_effect=self._live_env):
            live_count, audit = master_file._configure_startup_live_trading(
                workers,
                store,
                master_live=True,
                client=client,
            )

        self.assertEqual(live_count, 0)
        self.assertFalse(audit.safe_to_enable_live)
        self.assertEqual(client.calls, ["login", "preload", "orders", "positions"])
        self.assertEqual(client.mutations, [])
        self.assertNotIn("DO-NOT-LOG", repr(audit))

    def test_missing_client_or_failed_login_stays_paper(self):
        for case in ("missing", "login_failed"):
            with self.subTest(case=case):
                workers, store = self._workers_and_store()
                client = None
                if case == "login_failed":
                    client = self._Client(
                        workers,
                        orders=BrokerQueryResult.success(()),
                        positions=BrokerQueryResult.success(()),
                        login_ok=False,
                    )
                with patch.object(master_file, "_env_bool", side_effect=self._live_env):
                    live_count, audit = master_file._configure_startup_live_trading(
                        workers,
                        store,
                        master_live=True,
                        client=client,
                    )

                self.assertEqual(live_count, 0)
                self.assertTrue(all(not worker.live_trading for worker in workers))
                self.assertIsNotNone(audit)
                self.assertFalse(audit.safe_to_enable_live)
                self.assertIs(store.startup_exposure_audit, audit)
                self.assertTrue(store.execution_safety.entry_freeze_snapshot()[0])
                if client is not None:
                    self.assertEqual(client.calls, ["login"])
                    self.assertEqual(client.mutations, [])

    def test_invalid_numeric_live_config_skips_broker_login_and_stays_paper(self):
        workers = [self._Worker("Renko")]
        store = master_file.SharedMarketDataStore()
        client = self._Client(
            workers,
            orders=BrokerQueryResult.success(()),
            positions=BrokerQueryResult.success(()),
        )
        with (
            patch.dict(os.environ, {"RENKO_MAX_LOSS": "not-a-number"}),
            patch.object(master_file, "_env_bool", side_effect=self._live_env),
        ):
            live_count, audit = master_file._configure_startup_live_trading(
                workers,
                store,
                master_live=True,
                client=client,
            )

        self.assertEqual(live_count, 0)
        self.assertFalse(workers[0].live_trading)
        self.assertEqual(client.calls, [])
        self.assertIsNotNone(audit)
        self.assertFalse(audit.safe_to_enable_live)
        self.assertIn("RENKO_MAX_LOSS", " ".join(audit.evidence))

    def test_one_invalid_requested_strategy_blocks_every_live_candidate(self):
        workers = [self._Worker("Renko"), self._Worker("EMA")]
        store = master_file.SharedMarketDataStore()
        client = self._Client(
            workers,
            orders=BrokerQueryResult.success(()),
            positions=BrokerQueryResult.success(()),
        )
        with (
            patch.dict(os.environ, {"RENKO_MAX_LOSS": "invalid"}),
            patch.object(
                master_file,
                "_env_bool",
                side_effect=lambda name, default=False: name
                in {"RENKO_LIVE_TRADING", "EMA_LIVE_TRADING"},
            ),
        ):
            live_count, audit = master_file._configure_startup_live_trading(
                workers,
                store,
                master_live=True,
                client=client,
            )

        self.assertEqual(live_count, 0)
        self.assertTrue(all(not worker.live_trading for worker in workers))
        self.assertEqual(client.calls, [])
        self.assertIsNotNone(audit)
        self.assertFalse(audit.safe_to_enable_live)

    def test_nonpositive_lots_and_bad_cutoff_disable_only_live_mode(self):
        worker = self._Worker("Renko")
        worker.lots = 0
        worker.square_off_hour = 25

        errors = master_file._live_config_errors(worker, "RENKO")

        self.assertTrue(any("lots" in error.lower() for error in errors))
        self.assertTrue(any("cutoff" in error.lower() for error in errors))
        # Virtual/paper worker construction is intentionally unaffected.
        self.assertTrue(worker.live_trading)

    def test_shared_hedged_clock_env_names_are_strictly_validated(self):
        for strategy_name, prefix in (
            ("SupertrendBullish", "BULLISH"),
            ("DonchianBearish", "BEARISH"),
        ):
            with self.subTest(strategy_name=strategy_name):
                worker = self._Worker(strategy_name)
                with patch.dict(
                    os.environ,
                    {"SUPERTREND_SQUARE_OFF_HOUR": "not-an-hour"},
                ):
                    errors = master_file._live_config_errors(worker, prefix)

                self.assertIn(
                    "SUPERTREND_SQUARE_OFF_HOUR is not numeric",
                    errors,
                )

    def test_malformed_raw_trading_start_cannot_hide_behind_default(self):
        worker = self._Worker("Renko")
        with patch.dict(os.environ, {"RENKO_TRADING_START_HOUR": "bad"}):
            errors = master_file._live_config_errors(worker, "RENKO")

        self.assertIn("RENKO_TRADING_START_HOUR is not numeric", errors)

    def test_risk_budget_max_lots_must_be_a_positive_integer_for_live(self):
        worker = self._Worker("ProfitShooter")
        with patch.dict(os.environ, {"PROFIT_SHOOTER_MAX_LOTS": "2.5"}):
            errors = master_file._live_config_errors(worker, "PROFIT_SHOOTER")

        self.assertIn(
            "PROFIT_SHOOTER_MAX_LOTS must be a positive integer",
            errors,
        )

    def test_resolved_risk_budget_max_lots_fails_closed_for_live(self):
        worker = self._Worker("MoneyMachine")
        with patch.object(master_file, "MONEY_MACHINE_MAX_LOTS", 0):
            errors = master_file._live_config_errors(worker, "MONEY_MACHINE")

        self.assertIn(
            "resolved MONEY_MACHINE_MAX_LOTS must be a positive integer",
            errors,
        )

    def test_malformed_size_multiplier_blocks_live_trading(self):
        """The multiplier scales lots, budget, cap AND max-loss together, so a
        malformed one must REFUSE live rather than fall back to 1 and trade a
        size the operator did not configure. (Paper still falls back to 1.)"""
        worker = self._Worker("Renko")
        for raw in ("0", "-1", "2.5", "26", "two"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"RENKO_SIZE_MULTIPLIER": raw}
            ):
                errors = master_file._live_config_errors(worker, "RENKO")

            self.assertTrue(
                any("RENKO_SIZE_MULTIPLIER" in error for error in errors),
                f"{raw!r} must block live trading, got {errors}",
            )

    def test_valid_and_absent_size_multipliers_are_accepted_for_live(self):
        worker = self._Worker("Renko")
        for raw in ("1", "2", "25"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"RENKO_SIZE_MULTIPLIER": raw}
            ):
                errors = master_file._live_config_errors(worker, "RENKO")

            self.assertFalse(
                any("SIZE_MULTIPLIER" in error for error in errors),
                f"{raw!r} is valid but was rejected: {errors}",
            )


class TestStrategySizeMultiplier(unittest.TestCase):
    """`<PREFIX>_SIZE_MULTIPLIER` scales one strategy's whole size/risk set.

    The multiplier exists so position size can grow with the account by editing
    ONE number per strategy. Because it moves real-money size in both
    directions, the tests below pin three separate properties: the default is a
    no-op, a valid multiplier scales every size-bearing knob exactly once, and
    a malformed one can never quietly resize a live strategy.
    """

    @staticmethod
    def _without(*names):
        """Patch os.environ with the given keys guaranteed ABSENT."""
        patcher = patch.dict(os.environ)
        patcher.start()
        for name in names:
            os.environ.pop(name, None)
        return patcher

    def test_absent_multiplier_is_a_no_op(self):
        """The safety-critical case: an unset multiplier must leave every knob
        byte-identical to the pre-feature behaviour."""
        patcher = self._without("RENKO_SIZE_MULTIPLIER", "GOLDMINE_SIZE_MULTIPLIER")
        self.addCleanup(patcher.stop)

        self.assertEqual(master_file._strategy_size_multiplier("RENKO"), 1)
        self.assertEqual(
            master_file._scaled_int("RENKO", "RENKO_LOTS", 1),
            master_file._env_int("RENKO_LOTS", 1),
        )
        self.assertEqual(
            master_file._scaled_float("GOLDMINE", "GOLDMINE_RISK_BUDGET", 2500.0),
            master_file._env_float("GOLDMINE_RISK_BUDGET", 2500.0),
        )

    def test_whole_number_multiplier_scales_lots_budget_cap_and_max_loss(self):
        """The operator's own figures: 5 lots -> 10, Rs.5,500 -> Rs.11,000."""
        with patch.dict(
            os.environ,
            {
                "GOLDMINE_SIZE_MULTIPLIER": "2",
                "RENKO_SIZE_MULTIPLIER": "2",
            },
        ):
            self.assertEqual(master_file._strategy_size_multiplier("GOLDMINE"), 2)
            # Lot cap 5 -> 10.
            self.assertEqual(
                master_file._scaled_int("GOLDMINE", "GOLDMINE_MAX_LOTS", 5), 10
            )
            # Per-trade budget 2500 -> 5000.
            self.assertEqual(
                master_file._scaled_float("GOLDMINE", "GOLDMINE_RISK_BUDGET", 2500.0),
                5000.0,
            )
            # Daily kill-switch 5500 -> 11000.
            self.assertEqual(
                master_file._scaled_float("RENKO", "RENKO_MAX_LOSS", 5500.0),
                11000.0,
            )

    def test_multiplier_is_scoped_to_its_own_strategy(self):
        """Per-strategy only: scaling one worker must not touch any other."""
        with patch.dict(os.environ, {"RENKO_SIZE_MULTIPLIER": "3"}):
            self.assertEqual(master_file._strategy_size_multiplier("RENKO"), 3)
            self.assertEqual(master_file._strategy_size_multiplier("EMA"), 1)

    def test_signal_gen_ops_scales_lots_and_daily_max_loss(self):
        """One helper feeds 14 workers (13 ported + SL Hunting), so its two
        size-bearing values must both scale -- and the max-loss must scale
        exactly ONCE despite being a capital x percentage product."""
        env = {
            "SMA_CROSSOVER_LOTS": "2",
            "SMA_CROSSOVER_STARTING_CAPITAL": "600000",
            "SMA_CROSSOVER_DAILY_MAX_LOSS_PCT": "0.03",
        }
        with patch.dict(os.environ, {**env, "SMA_CROSSOVER_SIZE_MULTIPLIER": "1"}):
            base = master_file._signal_gen_ops("SMA_CROSSOVER")
        with patch.dict(os.environ, {**env, "SMA_CROSSOVER_SIZE_MULTIPLIER": "3"}):
            scaled = master_file._signal_gen_ops("SMA_CROSSOVER")

        self.assertEqual(base["lots"], 2)
        self.assertEqual(scaled["lots"], 6)
        self.assertAlmostEqual(base["max_loss"], 18000.0)
        self.assertAlmostEqual(scaled["max_loss"], 54000.0)
        # Non-size knobs are untouched.
        self.assertEqual(base["poll_seconds"], scaled["poll_seconds"])
        self.assertEqual(base["square_off_hour"], scaled["square_off_hour"])

    def test_malformed_values_fall_back_to_one_for_paper(self):
        """Forgiving like _env_int/_env_float: a typo must not crash a paper
        run, and must never resolve to a SMALLER-or-larger surprise size."""
        for raw in ("0", "-1", "2.5", "30", "two", "", "  "):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"RENKO_SIZE_MULTIPLIER": raw}
            ):
                self.assertEqual(master_file._strategy_size_multiplier("RENKO"), 1)

    def test_ceiling_is_twenty_five_inclusive(self):
        """25 is accepted; 26 falls back to 1 rather than sizing at 26x."""
        with patch.dict(os.environ, {"RENKO_SIZE_MULTIPLIER": "25"}):
            self.assertEqual(master_file._strategy_size_multiplier("RENKO"), 25)
        with patch.dict(os.environ, {"RENKO_SIZE_MULTIPLIER": "26"}):
            self.assertEqual(master_file._strategy_size_multiplier("RENKO"), 1)
        self.assertEqual(master_file.MAX_SIZE_MULTIPLIER, 25)

    def test_delta20_absolute_cap_is_never_double_scaled(self):
        """DELTA20_MAX_LOSS is PER_LOT x LOTS, so it inherits the multiplier
        through LOTS. Scaling PER_LOT as well would square it (M^2)."""
        self.assertAlmostEqual(
            master_file.DELTA20_MAX_LOSS,
            master_file.DELTA20_MAX_LOSS_PER_LOT * master_file.DELTA20_LOTS,
        )

    def test_every_size_knob_is_wired_through_the_scaled_helpers(self):
        """Drift guard: a strategy added later must not read a size knob with
        the raw _env_* helpers, or its multiplier would silently do nothing.

        Deliberately excluded: *_MAX_LOSS_PER_LOT (a per-lot figure whose
        absolute cap already scales via LOTS) and *_STARTING_CAPITAL /
        *_DAILY_MAX_LOSS_PCT (their PRODUCT carries the multiplier instead).
        """
        source = file_path.read_text(encoding="utf-8")
        unscaled = re.findall(
            r"^[A-Z][A-Z0-9_]*_(?:LOTS|MAX_LOTS|RISK_BUDGET|MAX_LOSS) = _env_(?:int|float)\(.*$",
            source,
            flags=re.MULTILINE,
        )
        self.assertEqual(
            unscaled,
            [],
            "these size knobs bypass the size multiplier: " + "; ".join(unscaled),
        )

    def test_risk_budget_sizing_doubles_the_accepted_lots(self):
        """End-to-end through the real sizing authority: a setup that takes 3
        lots unscaled takes 6 at 2x, and still respects the scaled budget."""
        entry, stop, lot_size = 24300.0, 24290.0, 75  # one lot risks Rs.750

        base = master_file.SizingDecision.from_risk_budget(
            entry=entry, stop=stop, lot_size=lot_size, budget=2500.0, max_lots=5
        )
        scaled = master_file.SizingDecision.from_risk_budget(
            entry=entry, stop=stop, lot_size=lot_size, budget=5000.0, max_lots=10
        )

        self.assertEqual(base.lots, 3)
        self.assertEqual(scaled.lots, 6)
        self.assertEqual(scaled.quantity, 2 * base.quantity)
        self.assertLessEqual(scaled.total_risk, 5000.0)

    def test_scaled_budget_may_exceed_a_pure_multiple_but_stays_inside_it(self):
        """Documented consequence: floor(M*b/r) >= M*floor(b/r). A 2x of a
        Rs.2,500 budget against a Rs.1,500 one-lot risk gives 3 lots, not 2 --
        more than a pure doubling, yet still strictly within the scaled budget."""
        entry, stop, lot_size = 24300.0, 24280.0, 75  # one lot risks Rs.1500

        base = master_file.SizingDecision.from_risk_budget(
            entry=entry, stop=stop, lot_size=lot_size, budget=2500.0, max_lots=5
        )
        scaled = master_file.SizingDecision.from_risk_budget(
            entry=entry, stop=stop, lot_size=lot_size, budget=5000.0, max_lots=10
        )

        self.assertEqual(base.lots, 1)
        self.assertEqual(scaled.lots, 3)
        self.assertGreater(scaled.lots, 2 * base.lots)
        self.assertLessEqual(scaled.total_risk, 5000.0)


class TestCoordinatedShutdownSupervisor(unittest.TestCase):
    """Process finalization is forbidden until local and broker books are flat."""

    class _RuntimeWorker:
        def __init__(self, *, start_error=None, interrupt_shutdown_once=False):
            self.start_error = start_error
            self.interrupt_shutdown_once = interrupt_shutdown_once
            self.started = False
            self.alive = False
            self.shutdown_requests = []

        def start(self):
            if self.start_error is not None:
                raise self.start_error
            self.started = True
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            if self.shutdown_requests:
                self.alive = False

        def request_worker_shutdown(self, reason):
            if self.interrupt_shutdown_once:
                self.interrupt_shutdown_once = False
                raise KeyboardInterrupt
            self.shutdown_requests.append(reason)

    class _ShutdownClient:
        def __init__(self, audits):
            self.is_logged_in = True
            self._audits = list(audits)
            self.logout_calls = 0

        def list_open_orders(self):
            outcome = self._audits[0]
            return outcome[0]

        def list_open_positions(self):
            outcome = self._audits.pop(0) if len(self._audits) > 1 else self._audits[0]
            return outcome[1]

        def logout(self):
            self.logout_calls += 1

    @staticmethod
    def _flat_books():
        return (BrokerQueryResult.success(()), BrokerQueryResult.success(()))

    @staticmethod
    def _indeterminate_books():
        return (
            BrokerQueryResult.indeterminate("status timeout"),
            BrokerQueryResult.success(()),
        )

    @staticmethod
    def _store_with_tracked_leg():
        """Build a store whose execution ledger tracks one unresolved live leg."""
        from Dependencies.execution_ledger import LegSpec

        store = master_file.SharedMarketDataStore()
        spec = LegSpec(
            strategy="Renko",
            correlation_id="ABCD1234",
            role="N",
            underlying="NIFTY",
            symbol="NIFTY16JUL2622500CE",
            option_type="CE",
            strike=22500.0,
            expiry=None,
            opening_side="BUY",
            target_quantity=75,
        )
        state = store.execution_ledger.register(spec)
        return store, state

    @staticmethod
    def _flatten_tracked_leg(store, state):
        """Resolve the tracked leg with a terminal zero-fill rejection (flat)."""
        from Dependencies.broker_contract import OrderResult, OrderStatus
        from Dependencies.execution_ledger import OrderIntent

        handle = store.execution_ledger.start_attempt(
            state.exposure_id, OrderIntent.OPEN, 75
        )
        store.execution_ledger.apply_result(
            handle,
            OrderResult(
                order_id="TEST-FLAT-1",
                requested_quantity=75,
                filled_quantity=0,
                remaining_quantity=75,
                status=OrderStatus.REJECTED,
                broker_state="REJECTED",
                reason="test rejection",
            ),
        )

    def test_ctrl_c_requests_worker_shutdown_without_setting_terminal_event(self):
        workers = [MagicMock(), MagicMock()]
        terminal_event = threading.Event()

        count = master_file._request_worker_shutdown(workers, "KEYBOARD_INTERRUPT")

        self.assertEqual(count, 2)
        self.assertFalse(terminal_event.is_set())
        for worker in workers:
            worker.request_worker_shutdown.assert_called_once_with("KEYBOARD_INTERRUPT")

    def test_partial_thread_start_failure_still_coordinates_started_worker(self):
        fetcher = MagicMock()
        first = self._RuntimeWorker()
        second = self._RuntimeWorker(start_error=RuntimeError("start failed"))

        natural_eod = master_file._start_and_supervise_runtime_threads(
            fetcher,
            None,
            [first, second],
        )

        self.assertFalse(natural_eod)
        self.assertTrue(first.started)
        self.assertFalse(first.alive)
        self.assertEqual(first.shutdown_requests, ["SUPERVISOR_EXCEPTION"])

    def test_interrupt_during_shutdown_request_is_ignored_until_worker_stops(self):
        fetcher = MagicMock()
        worker = self._RuntimeWorker(interrupt_shutdown_once=True)

        def interrupt_after_start():
            worker.started = True
            worker.alive = True
            raise KeyboardInterrupt

        worker.start = interrupt_after_start

        natural_eod = master_file._start_and_supervise_runtime_threads(
            fetcher,
            None,
            [worker],
        )

        self.assertFalse(natural_eod)
        self.assertFalse(worker.alive)
        self.assertEqual(worker.shutdown_requests, ["KEYBOARD_INTERRUPT"])

    def test_wait_retries_until_runner_ledger_confirms_flat(self):
        """Unresolved RUNNER exposure blocks; account books never block here."""

        client = self._ShutdownClient([self._indeterminate_books()])
        store, state = self._store_with_tracked_leg()
        sleeps = []

        def flattening_sleep(delay):
            sleeps.append(delay)
            self._flatten_tracked_leg(store, state)

        flat = master_file._wait_for_shutdown_account_flat(
            store,
            client,
            sleep=flattening_sleep,
            max_attempts=3,
        )

        self.assertTrue(flat)
        self.assertEqual(sleeps, [1.0])

    def test_permanent_runner_exposure_never_allows_clean_finalization(self):
        client = self._ShutdownClient([self._flat_books()])
        store, _state = self._store_with_tracked_leg()
        sleeps = []

        flat = master_file._wait_for_shutdown_account_flat(
            store,
            client,
            sleep=sleeps.append,
            max_attempts=4,
        )

        self.assertFalse(flat)
        self.assertEqual(sleeps, [1.0, 2.0, 5.0])

    def test_missing_client_after_live_session_is_not_treated_as_flat(self):
        store = master_file.SharedMarketDataStore()
        store.live_session_started = True

        audit = master_file._advisory_account_audit(store, None)

        self.assertFalse(audit.safe_to_enable_live)
        self.assertIn("unavailable", " ".join(audit.reasons).lower())

    def test_shutdown_audit_recovers_logged_out_live_session_before_query(self):
        class RecoveringClient(self._ShutdownClient):
            def __init__(self):
                super().__init__([TestCoordinatedShutdownSupervisor._flat_books()])
                self.is_logged_in = False
                self.ensure_calls = 0

            def ensure_logged_in(self):
                self.ensure_calls += 1
                self.is_logged_in = True
                return True

        store = master_file.SharedMarketDataStore()
        store.live_session_started = True
        client = RecoveringClient()

        audit = master_file._advisory_account_audit(store, client)

        self.assertTrue(audit.safe_to_enable_live)
        self.assertEqual(client.ensure_calls, 1)

    def test_shutdown_audit_keeps_failed_session_recovery_indeterminate(self):
        client = self._ShutdownClient([self._flat_books()])
        client.is_logged_in = False
        client.ensure_logged_in = MagicMock(return_value=False)
        store = master_file.SharedMarketDataStore()
        store.live_session_started = True

        audit = master_file._advisory_account_audit(store, client)

        self.assertFalse(audit.safe_to_enable_live)
        client.ensure_logged_in.assert_called_once_with()

    def test_additional_interrupt_cannot_bypass_final_runner_reconciliation(self):
        client = self._ShutdownClient([self._flat_books()])
        store, state = self._store_with_tracked_leg()
        sleeps = []

        def interrupted_flattening_sleep(delay):
            sleeps.append(delay)
            self._flatten_tracked_leg(store, state)
            raise KeyboardInterrupt

        flat = master_file._wait_for_shutdown_account_flat(
            store,
            client,
            sleep=interrupted_flattening_sleep,
            max_attempts=3,
        )

        self.assertTrue(flat)
        self.assertEqual(sleeps, [1.0])

    def test_logout_results_and_refresh_are_blocked_while_runner_exposure_open(self):
        client = self._ShutdownClient([self._flat_books()])
        store, _state = self._store_with_tracked_leg()
        workers = [MagicMock()]
        with (
            patch.object(master_file, "_publish_eod_summary") as summary,
            patch.object(master_file, "_update_pnl_google_sheet") as sheet,
            patch.object(master_file, "_refresh_instrument_master_for_next_day") as refresh,
        ):
            finalized = master_file._finalize_flat_session(
                workers,
                store,
                client,
                trade_event_queue=None,
                natural_eod=True,
            )

        self.assertFalse(finalized)
        self.assertEqual(client.logout_calls, 0)
        summary.assert_not_called()
        sheet.assert_not_called()
        refresh.assert_not_called()

    def test_manual_account_exposure_warns_but_does_not_block_finalization(self):
        """The operator's own positions alert loudly; results/logout proceed."""
        from Dependencies.broker_contract import OpenPosition

        client = self._ShutdownClient(
            [
                (
                    BrokerQueryResult.success(()),
                    BrokerQueryResult.success(
                        (
                            OpenPosition(
                                symbol="NIFTY16JUL2622500CE",
                                quantity=75,
                                product_type="NRML",
                            ),
                        )
                    ),
                )
            ]
        )
        store = master_file.SharedMarketDataStore()
        store.live_session_started = True
        event_queue = master_file.queue.Queue()
        with (
            patch.object(master_file, "_publish_eod_summary") as summary,
            patch.object(master_file, "_update_pnl_google_sheet") as sheet,
            patch.object(master_file, "_refresh_instrument_master_for_next_day") as refresh,
        ):
            finalized = master_file._finalize_flat_session(
                [MagicMock()],
                store,
                client,
                trade_event_queue=event_queue,
                natural_eod=True,
            )

        self.assertTrue(finalized)
        self.assertEqual(client.logout_calls, 1)
        summary.assert_called_once()
        sheet.assert_called_once_with()
        refresh.assert_called_once_with()
        alert = event_queue.get_nowait()
        self.assertEqual(alert["action"], "SHUTDOWN_ACCOUNT_WARNING")
        self.assertIn("position", alert["reason"].lower())
        # Fixed vocabulary only: the operator's symbol never reaches the alert.
        self.assertNotIn("NIFTY16JUL2622500CE", repr(alert))

    def test_interrupt_during_final_flat_audit_blocks_finalization(self):
        client = self._ShutdownClient([self._flat_books()])
        with (
            patch.object(
                master_file,
                "_runner_exposure_audit",
                side_effect=KeyboardInterrupt,
            ),
            patch.object(master_file, "_publish_eod_summary") as summary,
            patch.object(master_file, "_update_pnl_google_sheet") as sheet,
            patch.object(master_file, "_refresh_instrument_master_for_next_day") as refresh,
        ):
            finalized = master_file._finalize_flat_session(
                [MagicMock()],
                master_file.SharedMarketDataStore(),
                client,
                trade_event_queue=None,
                natural_eod=True,
            )

        self.assertFalse(finalized)
        self.assertEqual(client.logout_calls, 0)
        summary.assert_not_called()
        sheet.assert_not_called()
        refresh.assert_not_called()

    def test_flat_session_may_logout_and_refresh_after_final_audit(self):
        client = self._ShutdownClient([self._flat_books()])
        workers = [MagicMock()]
        with (
            patch.object(master_file, "_publish_eod_summary") as summary,
            patch.object(master_file, "_update_pnl_google_sheet") as sheet,
            patch.object(master_file, "_refresh_instrument_master_for_next_day") as refresh,
        ):
            finalized = master_file._finalize_flat_session(
                workers,
                master_file.SharedMarketDataStore(),
                client,
                trade_event_queue=None,
                natural_eod=False,
            )

        self.assertTrue(finalized)
        self.assertEqual(client.logout_calls, 1)
        summary.assert_not_called()
        sheet.assert_not_called()
        refresh.assert_called_once_with()


if __name__ == "__main__":
    # Use the custom runner so both `python file.py` and any other direct
    # invocation produce verbose per-test output PLUS the final summary.
    sys.exit(0 if _run_with_logging() else 1)
