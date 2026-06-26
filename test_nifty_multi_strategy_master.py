import unittest
import importlib.util
import tempfile
import threading
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
from unittest.mock import MagicMock, patch
import os
import sys

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
with patch.dict(os.environ, {"DHAN_CLIENT_CODE": "test", "DHAN_TOKEN_ID": "test"}):
    # This completely disables the 'dhanhq' module while the script loads
    with patch("dhanhq.dhanhq"):
        try:
            # We execute the file so that all classes and functions become available
            spec.loader.exec_module(master_file)
        except Exception as e:
            print(f"Failed to load master_file for testing: {e}")


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
        """Hedged exit: BUY-to-close main, SELL-to-close hedge."""
        fake = _FakeShoonya()
        self.worker.live_trading = True
        main_leg = self._leg("PE", 22000.0, 50, "MAIN")
        hedge_leg = self._leg("PE", 21000.0, 25, "HEDGE")
        with patch.object(master_file, "execution_client", fake):
            ok = self.worker._place_real_hedged_exit(main_leg, hedge_leg)
        self.assertTrue(ok)
        self.assertEqual(self._sides(fake), [("BUY", 50), ("SELL", 25)])


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
        state, filled, qty, reason = c._order_status("ORD1")
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
             patch.object(smod, "_FILL_POLL_INTERVAL", 0.01):
            with self.assertRaises(Exception):
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
