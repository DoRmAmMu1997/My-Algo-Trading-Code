"""Focused safety tests for shared live market-data validation and health state."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from Dependencies.market_data_health import (
    MarketDataHealth,
    MarketDataValidationError,
    complete_minute_bucket_mask,
    validate_ohlc_frame,
)

IST = ZoneInfo("Asia/Kolkata")


def _frame(timestamps: list[object]) -> pd.DataFrame:
    """Build a small, geometrically valid OHLC fixture."""

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [102.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [101.0] * len(timestamps),
        }
    )


class TestOHLCValidation(unittest.TestCase):
    """Bad candles must never be published as a usable live snapshot."""

    def setUp(self) -> None:
        self.now = datetime(2026, 7, 16, 10, 0, 30, tzinfo=IST)

    def test_rejects_non_finite_and_non_positive_prices(self) -> None:
        frame = _frame([datetime(2026, 7, 16, 9, 59)])
        frame.loc[0, "close"] = float("inf")
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(frame, now=self.now)

        frame.loc[0, "close"] = 0.0
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(frame, now=self.now)

    def test_rejects_impossible_candle_geometry(self) -> None:
        frame = _frame([datetime(2026, 7, 16, 9, 59)])
        frame.loc[0, "high"] = 100.5
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(frame, now=self.now)

    def test_rejects_future_and_unordered_timestamps(self) -> None:
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(
                _frame([datetime(2026, 7, 16, 10, 1)]),
                now=self.now,
            )

        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(
                _frame(
                    [
                        datetime(2026, 7, 16, 9, 59),
                        datetime(2026, 7, 16, 9, 58),
                    ]
                ),
                now=self.now,
            )

    def test_collapses_identical_duplicates_but_rejects_conflicts(self) -> None:
        ts = datetime(2026, 7, 16, 9, 59)
        identical = validate_ohlc_frame(_frame([ts, ts]), now=self.now)
        self.assertEqual(len(identical), 1)

        conflicting = _frame([ts, ts])
        conflicting.loc[1, "close"] = 100.5
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(conflicting, now=self.now)

    def test_normalizes_aware_timestamps_to_naive_ist(self) -> None:
        utc_ts = pd.Timestamp("2026-07-16 04:29:00+00:00")
        result = validate_ohlc_frame(_frame([utc_ts]), now=self.now)
        self.assertEqual(result.iloc[0]["timestamp"], pd.Timestamp("2026-07-16 09:59:00"))
        self.assertIsNone(result.iloc[0]["timestamp"].tzinfo)


class TestCompleteMinuteBuckets(unittest.TestCase):
    """A row count cannot stand in for exact one-minute slot coverage."""

    def test_requires_each_exact_minute_once(self) -> None:
        complete = pd.DatetimeIndex(pd.date_range("2026-07-16 09:15", periods=5, freq="1min"))
        self.assertEqual(complete_minute_bucket_mask(complete, 5).tolist(), [True] * 5)

        wrong_slots = pd.DatetimeIndex(
            [
                "2026-07-16 09:15",
                "2026-07-16 09:15",
                "2026-07-16 09:17",
                "2026-07-16 09:18",
                "2026-07-16 09:19",
            ]
        )
        self.assertEqual(complete_minute_bucket_mask(wrong_slots, 5).tolist(), [False] * 5)


class TestMarketDataHealth(unittest.TestCase):
    """Live entry and liquidation state must follow explicit freshness rules."""

    def setUp(self) -> None:
        self.now = datetime(2026, 7, 16, 10, 0, 30, tzinfo=IST)
        self.key = ("IDX_I", 13)
        self.health = MarketDataHealth()
        self.health.begin_monitoring(now=self.now)

    def _record_healthy(self, now: datetime) -> None:
        self.health.record_refresh(
            ohlc_ok=True,
            newest_completed_bar=now - timedelta(seconds=60),
            ltp_fetched_at={self.key: now},
            required_ltp_keys={self.key},
            now=now,
        )

    def test_requires_three_healthy_refreshes_before_entry(self) -> None:
        self._record_healthy(self.now)
        self.assertFalse(self.health.snapshot(now=self.now).entry_allowed)
        self._record_healthy(self.now + timedelta(seconds=2))
        self.assertFalse(self.health.snapshot(now=self.now + timedelta(seconds=2)).entry_allowed)
        self._record_healthy(self.now + timedelta(seconds=4))
        snapshot = self.health.snapshot(now=self.now + timedelta(seconds=4))
        self.assertTrue(snapshot.entry_allowed)
        self.assertEqual(snapshot.healthy_streak, 3)

    def test_stale_ltp_blocks_entry_and_triggers_liquidation_after_30_seconds(self) -> None:
        for offset in (0, 2, 4):
            self._record_healthy(self.now + timedelta(seconds=offset))

        stale_at = self.now + timedelta(seconds=15)
        stale = self.health.snapshot(now=stale_at)
        self.assertFalse(stale.entry_allowed)
        self.assertFalse(stale.liquidation_required)
        self.assertTrue(any("LTP" in reason for reason in stale.reasons))

        liquidate = self.health.snapshot(now=stale_at + timedelta(seconds=30))
        self.assertTrue(liquidate.liquidation_required)

    def test_stale_completed_bar_blocks_entry(self) -> None:
        self.health.record_refresh(
            ohlc_ok=True,
            newest_completed_bar=self.now - timedelta(seconds=91),
            ltp_fetched_at={self.key: self.now},
            required_ltp_keys={self.key},
            now=self.now,
        )
        snapshot = self.health.snapshot(now=self.now)
        self.assertFalse(snapshot.entry_allowed)
        self.assertTrue(any("bar" in reason.lower() for reason in snapshot.reasons))

    def test_monitoring_is_disabled_by_default_for_offline_consumers(self) -> None:
        health = MarketDataHealth()
        snapshot = health.snapshot(now=self.now)
        self.assertFalse(snapshot.monitoring)
        self.assertTrue(snapshot.entry_allowed)


if __name__ == "__main__":
    unittest.main()
