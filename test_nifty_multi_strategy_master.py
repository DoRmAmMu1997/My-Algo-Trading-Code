import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import pandas as pd

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
        """Zero SL distance must not divide-by-zero - falls back to at least 1."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0,
            stop_underlying=22500.0,
            lot_size=50,
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)


# =============================================================================
# TEST SUITE: GOLDMINE DYNAMIC LOT SIZING
# =============================================================================
class TestGoldmineStrategyWorker(unittest.TestCase):
    """Goldmine reuses Profit Shooter's risk-based `_compute_entry_lots`."""

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.GoldmineStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )

    def test_compute_entry_lots_returns_positive_integer(self):
        """For a reasonable SL distance, the sizer returns >= 1 lot."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22450.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)

    def test_compute_entry_lots_handles_zero_distance(self):
        """Zero SL distance must not divide-by-zero - falls back to at least 1."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22500.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)


# =============================================================================
# TEST SUITE: MONEY MACHINE DYNAMIC LOT SIZING
# =============================================================================
class TestMoneyMachineStrategyWorker(unittest.TestCase):
    """Money Machine reuses the same risk-based `_compute_entry_lots`."""

    def setUp(self):
        self.store = master_file.SharedMarketDataStore()
        self.broker = MagicMock()
        self.stop_event = threading.Event()
        self.worker = master_file.MoneyMachineStrategyWorker(
            store=self.store, stop_event=self.stop_event, broker=self.broker
        )

    def test_compute_entry_lots_returns_positive_integer(self):
        """For a reasonable SL distance, the sizer returns >= 1 lot."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22450.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)

    def test_compute_entry_lots_handles_zero_distance(self):
        """Zero SL distance must not divide-by-zero - falls back to at least 1."""
        lots = self.worker._compute_entry_lots(
            entry_underlying=22500.0, stop_underlying=22500.0, lot_size=50
        )
        self.assertIsInstance(lots, int)
        self.assertGreaterEqual(lots, 1)


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

    def __init__(self, fail_on=None, resolve_returns=None):
        self.calls = []      # (shoonya_symbol, side, quantity)
        self.resolved = []   # (underlying, option_type, strike)
        self._fail_on = fail_on or (lambda symbol, side: False)
        self._resolve_returns = resolve_returns

    def resolve_option_symbol(self, underlying, expiry, option_type, strike,
                              exchange_segment="NFO"):
        self.resolved.append((underlying, option_type, float(strike)))
        if self._resolve_returns is not None:
            return self._resolve_returns
        return f"SHOONYA-{underlying}-{int(strike)}-{option_type}"

    def place_market_order(self, symbol, side, quantity,
                           exchange_segment="NFO", product_type="INTRADAY"):
        self.calls.append((symbol, side, quantity))
        if self._fail_on(symbol, side):
            raise RuntimeError("simulated broker reject")
        return {"stat": "Ok", "norenordno": f"ORD-{len(self.calls)}"}

    def extract_order_id(self, resp):
        return resp.get("norenordno", "") if isinstance(resp, dict) else ""


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


class TestTimezoneAssumption(unittest.TestCase):
    """OPS-001: every trading-window gate compares against the LOCAL wall clock
    (naive datetime.now()), assuming the box runs in IST. A wrong system
    timezone silently shifts market-open, entry cutoffs and the 15:15
    square-off, so startup must call it out loudly."""

    def test_ist_offset_produces_no_warning(self):
        offset = timedelta(hours=5, minutes=30)
        self.assertIsNone(master_file._timezone_assumption_warning(offset))

    def test_non_ist_offset_produces_actionable_warning(self):
        for offset in (timedelta(0), timedelta(hours=-5), timedelta(hours=5, minutes=45)):
            warning = master_file._timezone_assumption_warning(offset)
            self.assertIsNotNone(warning, offset)
            self.assertIn("IST", warning)
            self.assertIn("timezone", warning.lower())

    def test_default_uses_system_offset(self):
        # Whatever this machine's offset is, the helper must not crash and must
        # answer consistently with an explicit call using the same offset.
        system_offset = datetime.now().astimezone().utcoffset()
        self.assertEqual(
            master_file._timezone_assumption_warning(),
            master_file._timezone_assumption_warning(system_offset),
        )


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
        self.assertFalse(worker.ce_pos.live_legs_open)      # ...but no real leg opened
        self.assertFalse(worker.pe_pos.live_legs_open)

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
        live_legs_open True, and closing that leg DOES send the real SELL exactly once.
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
        self.assertTrue(worker.ce_pos.live_legs_open)        # both legs really open
        self.assertTrue(worker.pe_pos.live_legs_open)

        # Closing the CE leg sends exactly one real SELL for that leg; PE untouched.
        exit_fake = _FakeShoonya()
        with patch.object(master_file, "execution_client", exit_fake):
            worker._exit_leg("CE", "TEST_EXIT")
        self.assertEqual([s for (_sym, s, _q) in exit_fake.calls], ["SELL"])
        self.assertFalse(worker.ce_pos.active)
        self.assertTrue(worker.pe_pos.active)


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

    def test_mirror_opens_with_same_lot_count(self):
        worker, _ = self._make_worker()
        # 10-pt stop -> risk_based_lots => ceil(2500 / (10*75)) = 4 NIFTY lots.
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        nifty_lots = worker.pos.quantity // 75
        self.assertTrue(worker._mirror_pos.active)
        self.assertEqual(worker._mirror_pos.quantity, nifty_lots * 35)
        self.assertEqual(worker._mirror_pos.option_right, "CE")
        # The mirror asked BankNIFTY's resolver with the BNF spot + direction,
        # pinning the expiry to the monthly-rollover rule (BNF-001).
        worker._bnf_resolver.get_atm_option.assert_called_once_with(
            57910.0, "LONG",
            expiry_date=worker._bnf_resolver.get_monthly_rollover_expiry.return_value,
        )

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

        def fake_real_leg(side, leg):
            captured.append((side, dict(leg)))
            return True

        worker._place_real_leg = fake_real_leg
        # Exit orders are now gated on live_legs_open, which is only set True when the
        # entry ran live and confirmed. Run this worker live so both legs' exits fire
        # and we can assert the BANKNIFTY underlying on every real call.
        worker.live_trading = True
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
    bookkeeping is preserved (and that failures fall back to paper).
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
        """A rejected real entry still records the position (paper fallback)."""
        fake = _FakeShoonya(fail_on=lambda symbol, side: True)
        self.worker.live_trading = True
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker.enter_position(direction="LONG", entry_underlying=22500.0)
        self.assertTrue(ok)             # not skipped
        self.assertTrue(self.worker.pos.active)   # recorded as paper

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
            ok = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
        self.assertTrue(ok)
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50)])

    def test_hedged_entry_partial_fill_unwinds_hedge(self):
        """If the main (SELL) leg fails after the hedge filled, unwind the hedge."""
        fake = _FakeShoonya(fail_on=lambda symbol, side: side == "SELL")
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker._place_real_hedged_entry(main_leg, hedge_leg)
        self.assertFalse(ok)
        # BUY hedge (filled) -> SELL main (rejected) -> SELL hedge (unwind).
        self.assertEqual(self._sides(fake), [("BUY", 25), ("SELL", 50), ("SELL", 25)])

    def test_hedged_exit_buys_main_sells_hedge(self):
        """Hedged exit: BUY-to-close main, SELL-to-close hedge (real legs open)."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker._place_real_hedged_exit(main_leg, hedge_leg, live_legs_open=True)
        self.assertTrue(ok)
        self.assertEqual(self._sides(fake), [("BUY", 50), ("SELL", 25)])

    def test_hedged_exit_sends_no_orders_for_paper_fallback_position(self):
        """A live worker whose entry fell back to paper (live_legs_open False) must
        NOT send closing orders -- there are no real legs, and a BUY main / SELL
        hedge here would open phantom exposure (P1, Codex on PR #42)."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker._place_real_hedged_exit(main_leg, hedge_leg, live_legs_open=False)
        self.assertTrue(ok)             # caller flattens its paper books normally
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
            ok = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)
        self.assertFalse(ok)
        return fake

    def test_double_failure_records_the_orphaned_hedge_leg(self):
        self._double_fail_entry()
        self.assertEqual(len(self.worker._orphan_live_legs), 1)
        orphan = self.worker._orphan_live_legs[0]
        self.assertEqual(orphan["quantity"], 25)       # the bought hedge, not the main
        self.assertEqual(orphan["strike"], 21000.0)

    def test_single_failure_with_clean_unwind_records_nothing(self):
        # Only the MAIN SELL fails (strike 22000); the unwind SELL succeeds.
        fake = _FakeShoonya(fail_on=lambda symbol, side: side == "SELL" and "22000" in symbol)
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker._place_real_hedged_entry(self.main_leg, self.hedge_leg)
        self.assertFalse(ok)
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
    (live_legs_open False -- rejected/partial fill, or the double-failure orphan
    path) must NOT send real closing orders at exit. A BUY for the never-opened
    main leg plus a SELL for the already-closed hedge would open phantom live
    exposure. The exit still flattens the paper books."""

    def _make_bullish_worker(self, *, live_legs_open: bool):
        store = master_file.SharedMarketDataStore()
        worker = master_file.SupertrendBullishWorker(
            store=store, stop_event=threading.Event(), broker=MagicMock()
        )
        worker.live_trading = True
        seg = master_file.OPTION_EXCHANGE_SEGMENT
        worker.pos = master_file.HedgedPaperPosition(
            active=True, direction="LONG", live_legs_open=live_legs_open,
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


class TestExecModeTag(unittest.TestCase):
    """`_exec_mode_tag` labels how a trade actually executed (Codex PR #47)."""

    def _worker(self, *, live: bool):
        w = master_file.AtmSingleLegStrategyWorker(
            store=master_file.SharedMarketDataStore(),
            stop_event=threading.Event(), broker=MagicMock(),
        )
        w.live_trading = live
        return w

    def test_paper_worker_is_always_paper(self):
        w = self._worker(live=False)
        self.assertEqual(w._exec_mode_tag(True), "PAPER")
        self.assertEqual(w._exec_mode_tag(True, live_legs_open=False), "PAPER")

    def test_live_worker_labels(self):
        w = self._worker(live=True)
        self.assertEqual(w._exec_mode_tag(True), "LIVE")                       # confirmed real order
        self.assertEqual(w._exec_mode_tag(False), "PAPER_FALLBACK")            # real order failed
        # No real legs open -> no broker order sent -> PAPER_FALLBACK even though real_ok is a vacuous True.
        self.assertEqual(w._exec_mode_tag(True, live_legs_open=False), "PAPER_FALLBACK")


class TestShoonyaOrderAck(unittest.TestCase):
    """
    NorenApi.place_order returns None on failure and a dict with stat == "Ok" and
    a norenordno on success. The wrapper's `_is_order_ack` must tell a real
    acknowledgement apart from a failure so failures fall back to paper.
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
        c._confirm_fill("ORD1", 75)  # must not raise

    def test_confirm_fill_raises_on_rejected(self):
        c = self._client(self._hist("REJECTED", fld=0, qty=75, rej="RMS: margin"))
        with self.assertRaises(Exception):
            c._confirm_fill("ORD1", 75)

    def test_confirm_fill_times_out_when_never_filled(self):
        c = self._client(self._hist("OPEN", fld=0, qty=75))
        mod = type(c).__module__
        smod = sys.modules[mod]
        with patch.object(smod, "_FILL_TIMEOUT_SECONDS", 0.05), \
             patch.object(smod, "_FILL_POLL_INTERVAL", 0.01), self.assertRaises(Exception):
            c._confirm_fill("ORD1", 75)


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
        with patch.dict(os.environ, {"FLATTRADE_MARKET_PROTECTION": "5"}), \
             patch.object(self.client, "ensure_logged_in", return_value=True), \
             patch.object(self.client, "_post_api", return_value=ack) as post_api, \
             patch.object(self.client, "_confirm_fill") as confirm:
            result = self.client.place_market_order(
                symbol="NIFTY14JUL26C24150",
                side="BUY",
                quantity=65,
                exchange_segment="NFO",
                product_type="INTRADAY",
            )

        self.assertEqual(result, ack)
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
            self.client._confirm_fill("ORD1", 65)

        with patch.object(
            self.client,
            "_order_status",
            return_value=("rejected", 0, 65, "RMS: margin"),
        ), self.assertRaises(RuntimeError):
            self.client._confirm_fill("ORD2", 65)

        with patch.object(
            self.client, "_order_status", return_value=("open", 0, 65, "")
        ), patch.object(flattrade_module, "_FILL_TIMEOUT_SECONDS", 0.02), \
             patch.object(flattrade_module, "_FILL_POLL_INTERVAL", 0.005), \
             self.assertRaises(TimeoutError):
            self.client._confirm_fill("ORD3", 65)

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

    def test_round_trip_buys_then_sells_after_confirmation(self):
        fake_client = MagicMock()
        fake_client.place_market_order.side_effect = [
            {"stat": "Ok", "norenordno": "BUY1"},
            {"stat": "Ok", "norenordno": "SELL1"},
        ]
        fake_client.extract_order_id.side_effect = ["BUY1", "SELL1"]
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


if __name__ == "__main__":
    # Use the custom runner so both `python file.py` and any other direct
    # invocation produce verbose per-test output PLUS the final summary.
    sys.exit(0 if _run_with_logging() else 1)
