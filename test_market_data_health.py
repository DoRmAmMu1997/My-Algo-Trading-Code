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
    newest_completed_minute_timestamp,
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

    def test_rejects_empty_or_incomplete_snapshots(self) -> None:
        for frame in (None, pd.DataFrame()):
            with self.subTest(frame=frame), self.assertRaises(
                MarketDataValidationError, msg="empty"
            ):
                validate_ohlc_frame(frame, now=self.now)  # type: ignore[arg-type]

        with self.assertRaises(MarketDataValidationError, msg="missing"):
            validate_ohlc_frame(pd.DataFrame({"timestamp": [self.now]}), now=self.now)

    def test_rejects_invalid_or_misaligned_timestamps(self) -> None:
        for timestamp in (pd.NaT, datetime(2026, 7, 16, 9, 59, 1)):
            with self.subTest(timestamp=timestamp), self.assertRaises(
                MarketDataValidationError
            ):
                validate_ohlc_frame(_frame([timestamp]), now=self.now)

    def test_rejects_nonnumeric_price_and_each_geometry_violation(self) -> None:
        nonnumeric = _frame([datetime(2026, 7, 16, 9, 59)])
        nonnumeric["open"] = pd.Series(["not-a-price"], dtype=object)
        with self.assertRaises(MarketDataValidationError):
            validate_ohlc_frame(nonnumeric, now=self.now)

        mutations = (
            ("low", 101.5),   # low above open
            ("low", 101.5),   # low above close
            ("high", 99.5),   # high below open/close
            ("low", 103.0),   # low above high
        )
        for column, value in mutations:
            frame = _frame([datetime(2026, 7, 16, 9, 59)])
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(
                MarketDataValidationError
            ):
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

    def test_rejects_invalid_timeframes_and_handles_empty_or_partial_buckets(self) -> None:
        with self.assertRaises(ValueError, msg="positive"):
            complete_minute_bucket_mask(pd.DatetimeIndex([]), 0)

        self.assertEqual(
            complete_minute_bucket_mask(pd.DatetimeIndex([]), 5).tolist(),
            [],
        )
        partial = pd.DatetimeIndex(["2026-07-16 09:15", "2026-07-16 09:16"])
        self.assertEqual(complete_minute_bucket_mask(partial, 5).tolist(), [False, False])


class TestCompletedMinuteTimestamp(unittest.TestCase):
    """Only completed one-minute bars may drive freshness decisions."""

    def test_returns_none_for_missing_inputs(self) -> None:
        now = datetime(2026, 7, 16, 10, 0, 30, tzinfo=IST)
        self.assertIsNone(newest_completed_minute_timestamp(None, now=now))  # type: ignore[arg-type]
        self.assertIsNone(newest_completed_minute_timestamp(pd.DataFrame(), now=now))
        self.assertIsNone(
            newest_completed_minute_timestamp(pd.DataFrame({"open": [100]}), now=now)
        )

    def test_excludes_current_minute_and_returns_latest_completed_bar(self) -> None:
        now = datetime(2026, 7, 16, 10, 0, 30, tzinfo=IST)
        frame = _frame(
            [
                datetime(2026, 7, 16, 9, 58),
                datetime(2026, 7, 16, 9, 59),
                datetime(2026, 7, 16, 10, 0),
            ]
        )
        self.assertEqual(
            newest_completed_minute_timestamp(frame, now=now),
            datetime(2026, 7, 16, 9, 59, tzinfo=IST),
        )


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
            newest_completed_bar=self.now - timedelta(seconds=151),
            ltp_fetched_at={self.key: self.now},
            required_ltp_keys={self.key},
            now=self.now,
        )
        snapshot = self.health.snapshot(now=self.now)
        self.assertFalse(snapshot.entry_allowed)
        self.assertTrue(any("bar" in reason.lower() for reason in snapshot.reasons))

    def test_healthy_minute_cycle_bar_ages_are_never_stale(self) -> None:
        """Regression: bar ages are measured from the bar's START minute, so a
        normally-advancing feed's newest completed bar cycles from 60s to just
        under 120s old (plus publish latency) within EVERY minute.  The old 90s
        threshold sat inside that cycle and false-alarmed from second :31 of
        each minute; these exact in-cycle ages must always read as healthy."""
        for bar_age_seconds in (91, 119, 125):
            with self.subTest(bar_age_seconds=bar_age_seconds):
                snapshot = self.health.record_refresh(
                    ohlc_ok=True,
                    newest_completed_bar=self.now - timedelta(seconds=bar_age_seconds),
                    ltp_fetched_at={self.key: self.now},
                    required_ltp_keys={self.key},
                    now=self.now,
                )
                self.assertEqual(snapshot.reasons, ())

    def test_monitoring_is_disabled_by_default_for_offline_consumers(self) -> None:
        health = MarketDataHealth()
        snapshot = health.snapshot(now=self.now)
        self.assertFalse(snapshot.monitoring)
        self.assertTrue(snapshot.entry_allowed)

    def test_stop_monitoring_restores_offline_entry_behavior(self) -> None:
        self.health.stop_monitoring()
        snapshot = self.health.snapshot(now=self.now)
        self.assertFalse(snapshot.monitoring)
        self.assertTrue(snapshot.entry_allowed)

    def test_failed_refresh_and_missing_inputs_report_each_reason(self) -> None:
        snapshot = self.health.record_refresh(
            ohlc_ok=False,
            newest_completed_bar=None,
            ltp_fetched_at={},
            required_ltp_keys={self.key},
            now=self.now,
        )
        self.assertFalse(snapshot.entry_allowed)
        self.assertEqual(snapshot.healthy_streak, 0)
        self.assertTrue(any("refresh failed" in reason for reason in snapshot.reasons))
        self.assertTrue(any("unavailable" in reason for reason in snapshot.reasons))

    def test_future_bar_and_future_ltp_are_unhealthy(self) -> None:
        snapshot = self.health.record_refresh(
            ohlc_ok=True,
            newest_completed_bar=self.now + timedelta(seconds=6),
            ltp_fetched_at={self.key: self.now + timedelta(seconds=6)},
            required_ltp_keys={self.key},
            now=self.now,
        )
        self.assertTrue(any("bar is in the future" in reason for reason in snapshot.reasons))
        self.assertTrue(any("future-dated" in reason for reason in snapshot.reasons))

    def test_unhealthy_clock_starts_when_snapshot_first_observes_silence(self) -> None:
        for offset in (0, 2, 4):
            self._record_healthy(self.now + timedelta(seconds=offset))

        first_stale = self.now + timedelta(seconds=20)
        first = self.health.snapshot(now=first_stale)
        later = self.health.snapshot(now=first_stale + timedelta(seconds=31))
        self.assertEqual(first.unhealthy_seconds, 0.0)
        self.assertGreaterEqual(later.unhealthy_seconds, 31.0)
        self.assertTrue(later.liquidation_required)


if __name__ == "__main__":
    unittest.main()
