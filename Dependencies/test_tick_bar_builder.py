"""Unit tests for the pure websocket tick-to-bar helpers.

These cover the dhanhq-free logic the websocket market data producer relies
on: packet parsing, subscription confirmation, minute resolution (including
the stale-snapshot and off-session rejections verified against the real feed
on 2026-07-21), tick aggregation, the official-candle merge, and the per-minute
true-up divergence statistics.  Everything here runs offline.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pandas as pd
import pytest

from Dependencies.tick_bar_builder import (
    SEGMENT_CODE_TO_NAME,
    SEGMENT_NAME_TO_FEED_CODE,
    TickBarAggregator,
    TickEvent,
    TrueUpDivergence,
    divergence_stats,
    merge_official_and_tick_frames,
    packet_confirms_subscription,
    parse_marketfeed_packet,
    resolve_tick_minute,
)

# A mid-session IST wall-clock instant used across the tests (naive IST,
# matching the runner's frame timestamps).
NOW = datetime(2026, 7, 21, 10, 17, 34)


def _ticker(segment: int = 0, security_id: int = 13, ltp: str = "24238.50",
            ltt: str = "10:17:33") -> dict:
    """Real Ticker packet shape captured from dhanhq 2.2.0 marketfeed."""

    return {
        "type": "Ticker Data",
        "exchange_segment": segment,
        "security_id": security_id,
        "LTP": ltp,
        "LTT": ltt,
    }


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (minute, o, h, l, c) tuples."""

    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(ts) for ts, *_ in rows],
            "open": [o for _, o, *_ in rows],
            "high": [h for _, _, h, *_ in rows],
            "low": [low for *_, low, _ in rows],
            "close": [c for *_, c in rows],
        }
    )


class TestSegmentMaps:
    def test_maps_are_inverse_of_each_other(self):
        for code, name in SEGMENT_CODE_TO_NAME.items():
            assert SEGMENT_NAME_TO_FEED_CODE[name] == code

    def test_known_segments_present(self):
        assert SEGMENT_CODE_TO_NAME[0] == "IDX_I"
        assert SEGMENT_CODE_TO_NAME[2] == "NSE_FNO"
        assert SEGMENT_CODE_TO_NAME[8] == "BSE_FNO"


class TestParseMarketfeedPacket:
    def test_index_ticker_packet_parses(self):
        event = parse_marketfeed_packet(_ticker(), NOW)
        assert event == TickEvent(
            segment="IDX_I",
            security_id=13,
            ltp=24238.50,
            ltt_raw="10:17:33",
            received_at=NOW,
        )

    def test_option_ticker_packet_parses(self):
        event = parse_marketfeed_packet(
            _ticker(segment=2, security_id=57346, ltp="139.45"), NOW
        )
        assert event is not None
        assert event.segment == "NSE_FNO"
        assert event.security_id == 57346
        assert event.ltp == pytest.approx(139.45)

    def test_quote_and_full_packets_parse(self):
        for packet_type in ("Quote Data", "Full Data"):
            packet = _ticker()
            packet["type"] = packet_type
            event = parse_marketfeed_packet(packet, NOW)
            assert event is not None, packet_type

    def test_non_dict_payloads_rejected(self):
        assert parse_marketfeed_packet(None, NOW) is None
        assert parse_marketfeed_packet("Markets Open", NOW) is None
        assert parse_marketfeed_packet(42, NOW) is None

    def test_non_price_packet_types_rejected(self):
        for packet_type in ("Previous Close", "Market Depth", "OI Data", ""):
            packet = _ticker()
            packet["type"] = packet_type
            assert parse_marketfeed_packet(packet, NOW) is None, packet_type
        assert parse_marketfeed_packet({"LTP": "1.00"}, NOW) is None

    def test_bad_prices_rejected(self):
        for ltp in ("0.00", "-1.25", "abc", "", None, float("nan"), float("inf")):
            assert parse_marketfeed_packet(_ticker(ltp=ltp), NOW) is None, ltp
        packet = _ticker()
        del packet["LTP"]
        assert parse_marketfeed_packet(packet, NOW) is None

    def test_unknown_segment_rejected(self):
        assert parse_marketfeed_packet(_ticker(segment=5), NOW) is None

    def test_bad_security_id_rejected(self):
        assert parse_marketfeed_packet(_ticker(security_id="abc"), NOW) is None
        packet = _ticker()
        del packet["security_id"]
        assert parse_marketfeed_packet(packet, NOW) is None

    def test_missing_ltt_still_yields_event_for_ltp_use(self):
        packet = _ticker()
        del packet["LTT"]
        event = parse_marketfeed_packet(packet, NOW)
        assert event is not None
        assert event.ltt_raw is None


class TestPacketConfirmsSubscription:
    def test_price_packets_confirm(self):
        assert packet_confirms_subscription(_ticker()) == ("IDX_I", 13)
        quote = _ticker(segment=2, security_id=57346)
        quote["type"] = "Quote Data"
        assert packet_confirms_subscription(quote) == ("NSE_FNO", 57346)

    def test_previous_close_confirms_without_a_trade(self):
        packet = {
            "type": "Previous Close",
            "exchange_segment": 2,
            "security_id": 57346,
            "prev_close": "216.95",
            "prev_OI": 1240977408,
        }
        assert packet_confirms_subscription(packet) == ("NSE_FNO", 57346)

    def test_other_payloads_do_not_confirm(self):
        assert packet_confirms_subscription(None) is None
        assert packet_confirms_subscription("Markets Open") is None
        depth = _ticker()
        depth["type"] = "Market Depth"
        assert packet_confirms_subscription(depth) is None
        assert packet_confirms_subscription(_ticker(segment=5)) is None
        bad_id = _ticker(security_id="abc")
        assert packet_confirms_subscription(bad_id) is None


class TestResolveTickMinute:
    def test_live_tick_uses_ltt_minute(self):
        assert resolve_tick_minute("10:17:33", NOW) == pd.Timestamp("2026-07-21 10:17:00")

    def test_boundary_skew_within_tolerance_trusted(self):
        now = datetime(2026, 7, 21, 10, 18, 20)
        assert resolve_tick_minute("10:17:59", now) == pd.Timestamp("2026-07-21 10:17:00")

    def test_stale_snapshot_ltt_rejected(self):
        # On subscribe Dhan replays the last known tick; pre-open that carries
        # the prior session's 15:29:59 stamp and must never become a bar.
        now = datetime(2026, 7, 21, 9, 16, 0)
        assert resolve_tick_minute("15:29:59", now) is None

    def test_off_session_recompute_rejected_even_when_fresh(self):
        # Index values are recomputed after hours (observed LTT 19:26:03);
        # fresh-but-off-session ticks must not create bars.
        now = datetime(2026, 7, 21, 19, 26, 5)
        assert resolve_tick_minute("19:26:03", now) is None

    def test_pre_open_rejected(self):
        now = datetime(2026, 7, 21, 9, 15, 1)
        assert resolve_tick_minute("09:14:59", now) is None

    def test_session_first_and_last_minutes_accepted(self):
        now_open = datetime(2026, 7, 21, 9, 15, 2)
        assert resolve_tick_minute("09:15:00", now_open) == pd.Timestamp("2026-07-21 09:15:00")
        now_close = datetime(2026, 7, 21, 15, 29, 59)
        assert resolve_tick_minute("15:29:58", now_close) == pd.Timestamp("2026-07-21 15:29:00")

    def test_after_session_end_rejected(self):
        now = datetime(2026, 7, 21, 15, 30, 2)
        assert resolve_tick_minute("15:30:01", now) is None

    def test_unparsable_ltt_rejected(self):
        for raw in (None, "", "garbage", "25:99:99", "12:34"):
            assert resolve_tick_minute(raw, NOW) is None, raw

    def test_tolerance_is_configurable(self):
        now = datetime(2026, 7, 21, 10, 20, 0)
        assert resolve_tick_minute("10:17:00", now, tolerance_seconds=300.0) == pd.Timestamp(
            "2026-07-21 10:17:00"
        )
        assert resolve_tick_minute("10:17:00", now, tolerance_seconds=90.0) is None


class TestTickBarAggregator:
    def test_first_tick_opens_bar_and_reports_rollover(self):
        agg = TickBarAggregator()
        minute = pd.Timestamp("2026-07-21 10:17:00")
        assert agg.add_tick(minute, 100.0) is True
        frame = agg.tick_bars_frame()
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["timestamp"] == minute
        assert row["open"] == row["high"] == row["low"] == row["close"] == 100.0

    def test_subsequent_ticks_update_high_low_close_not_open(self):
        agg = TickBarAggregator()
        minute = pd.Timestamp("2026-07-21 10:17:00")
        agg.add_tick(minute, 100.0)
        assert agg.add_tick(minute, 102.0) is False
        assert agg.add_tick(minute, 99.0) is False
        assert agg.add_tick(minute, 101.0) is False
        row = agg.tick_bars_frame().iloc[0]
        assert row["open"] == 100.0
        assert row["high"] == 102.0
        assert row["low"] == 99.0
        assert row["close"] == 101.0

    def test_new_minute_reports_rollover(self):
        agg = TickBarAggregator()
        agg.add_tick(pd.Timestamp("2026-07-21 10:17:00"), 100.0)
        assert agg.add_tick(pd.Timestamp("2026-07-21 10:18:00"), 100.5) is True

    def test_late_tick_updates_older_bar_without_rollover(self):
        agg = TickBarAggregator()
        earlier = pd.Timestamp("2026-07-21 10:17:00")
        agg.add_tick(earlier, 100.0)
        agg.add_tick(pd.Timestamp("2026-07-21 10:18:00"), 100.5)
        assert agg.add_tick(earlier, 103.0) is False
        row = agg.tick_bars_frame().set_index("timestamp").loc[earlier]
        assert row["high"] == 103.0

    def test_gap_minute_created_without_rollover(self):
        agg = TickBarAggregator()
        agg.add_tick(pd.Timestamp("2026-07-21 10:18:00"), 100.5)
        assert agg.add_tick(pd.Timestamp("2026-07-21 10:16:00"), 99.5) is False
        assert len(agg.tick_bars_frame()) == 2

    def test_frame_sorted_with_canonical_columns(self):
        agg = TickBarAggregator()
        agg.add_tick(pd.Timestamp("2026-07-21 10:18:00"), 100.5)
        agg.add_tick(pd.Timestamp("2026-07-21 10:16:00"), 99.5)
        frame = agg.tick_bars_frame()
        assert list(frame.columns) == ["timestamp", "open", "high", "low", "close"]
        assert frame["timestamp"].is_monotonic_increasing

    def test_empty_frame_has_canonical_columns(self):
        frame = TickBarAggregator().tick_bars_frame()
        assert list(frame.columns) == ["timestamp", "open", "high", "low", "close"]
        assert frame.empty

    def test_signature_changes_on_mutation_and_is_stable_when_idle(self):
        agg = TickBarAggregator()
        first = agg.signature()
        assert agg.signature() == first
        agg.add_tick(pd.Timestamp("2026-07-21 10:17:00"), 100.0)
        second = agg.signature()
        assert second != first
        assert agg.signature() == second
        agg.add_tick(pd.Timestamp("2026-07-21 10:17:00"), 101.0)
        assert agg.signature() != second

    def test_max_minutes_cap_drops_oldest(self):
        agg = TickBarAggregator(max_minutes=2)
        agg.add_tick(pd.Timestamp("2026-07-21 10:16:00"), 99.0)
        agg.add_tick(pd.Timestamp("2026-07-21 10:17:00"), 100.0)
        agg.add_tick(pd.Timestamp("2026-07-21 10:18:00"), 101.0)
        frame = agg.tick_bars_frame()
        assert len(frame) == 2
        assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-07-21 10:17:00")

    def test_prune_older_than_removes_stale_bars(self):
        agg = TickBarAggregator()
        agg.add_tick(pd.Timestamp("2026-07-18 10:17:00"), 99.0)
        agg.add_tick(pd.Timestamp("2026-07-21 10:17:00"), 100.0)
        agg.prune_older_than(pd.Timestamp("2026-07-21 00:00:00"))
        frame = agg.tick_bars_frame()
        assert len(frame) == 1
        assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-07-21 10:17:00")


class TestMergeOfficialAndTickFrames:
    def test_official_wins_on_collision(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        tick = _bars([("2026-07-21 10:16:00", 100.2, 100.9, 99.9, 100.4)])
        merged = merge_official_and_tick_frames(official, tick)
        assert len(merged) == 1
        assert merged.iloc[0]["open"] == 100.0

    def test_tick_bars_extend_beyond_official_tail(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        tick = _bars([("2026-07-21 10:17:00", 100.5, 100.8, 100.2, 100.6)])
        merged = merge_official_and_tick_frames(official, tick)
        assert list(merged["timestamp"]) == [
            pd.Timestamp("2026-07-21 10:16:00"),
            pd.Timestamp("2026-07-21 10:17:00"),
        ]

    def test_tick_bars_fill_official_holes(self):
        official = _bars(
            [
                ("2026-07-21 10:15:00", 100.0, 101.0, 99.5, 100.5),
                ("2026-07-21 10:17:00", 100.6, 100.9, 100.3, 100.7),
            ]
        )
        tick = _bars([("2026-07-21 10:16:00", 100.5, 100.8, 100.2, 100.6)])
        merged = merge_official_and_tick_frames(official, tick)
        assert list(merged["timestamp"]) == [
            pd.Timestamp("2026-07-21 10:15:00"),
            pd.Timestamp("2026-07-21 10:16:00"),
            pd.Timestamp("2026-07-21 10:17:00"),
        ]

    def test_empty_inputs(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        empty = _bars([])
        assert merge_official_and_tick_frames(official, empty).equals(
            official.reset_index(drop=True)
        )
        assert len(merge_official_and_tick_frames(empty, official)) == 1
        both = merge_official_and_tick_frames(empty, empty)
        assert both.empty
        assert list(both.columns) == ["timestamp", "open", "high", "low", "close"]

    def test_extra_columns_dropped_and_inputs_unmutated(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        official["volume"] = [123]
        tick = _bars([("2026-07-21 10:17:00", 100.5, 100.8, 100.2, 100.6)])
        before_official = official.copy()
        before_tick = tick.copy()
        merged = merge_official_and_tick_frames(official, tick)
        assert list(merged.columns) == ["timestamp", "open", "high", "low", "close"]
        pd.testing.assert_frame_equal(official, before_official)
        pd.testing.assert_frame_equal(tick, before_tick)


class TestDivergenceStats:
    def test_identical_frames_have_no_mismatch(self):
        bars = _bars(
            [
                ("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5),
                ("2026-07-21 10:17:00", 100.6, 100.9, 100.3, 100.7),
            ]
        )
        stats = divergence_stats(bars, bars.copy())
        assert stats == TrueUpDivergence(overlapping=2, mismatched=0, max_abs_delta=0.0)

    def test_single_field_difference_counted_once(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        tick = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 101.75)])
        stats = divergence_stats(official, tick)
        assert stats.overlapping == 1
        assert stats.mismatched == 1
        assert stats.max_abs_delta == pytest.approx(1.25)

    def test_disjoint_frames_report_zero_overlap(self):
        official = _bars([("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5)])
        tick = _bars([("2026-07-21 10:17:00", 100.5, 100.8, 100.2, 100.6)])
        assert divergence_stats(official, tick) == TrueUpDivergence(0, 0, 0.0)

    def test_forming_minute_excluded(self):
        forming = pd.Timestamp("2026-07-21 10:17:00")
        official = _bars(
            [
                ("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5),
                ("2026-07-21 10:17:00", 100.6, 100.9, 100.3, 100.7),
            ]
        )
        tick = _bars(
            [
                ("2026-07-21 10:16:00", 100.0, 101.0, 99.5, 100.5),
                ("2026-07-21 10:17:00", 100.6, 105.0, 100.3, 104.0),
            ]
        )
        stats = divergence_stats(official, tick, forming_minute=forming)
        assert stats == TrueUpDivergence(overlapping=1, mismatched=0, max_abs_delta=0.0)

    def test_divergence_dataclass_is_frozen(self):
        stats = TrueUpDivergence(0, 0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.mismatched = 5  # type: ignore[misc]
