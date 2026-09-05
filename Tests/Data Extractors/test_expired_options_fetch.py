"""Tests for the expired-options download engine.

Nothing here touches the network: the DhanHQ client is a `SimpleNamespace` duck,
matching the style already used in `test_index_fetch_construction.py`.

The cases worth their weight are the ones encoding something the API taught us
the hard way -- expiryCode being 1-based, a `str` vs `dict` `remarks` meaning
completely different things, and the parallel-array response shape.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Tests/Data Extractors/<this file> -> the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = _REPO_ROOT / "Data Extractors" / "expired_options_fetch_dhan_common.py"
spec = importlib.util.spec_from_file_location("expired_options_fetch_dhan_common", MODULE_PATH)
engine = importlib.util.module_from_spec(spec)
sys.modules["expired_options_fetch_dhan_common"] = engine
spec.loader.exec_module(engine)


DEFAULTS = engine.ExpiredOptionsDefaults(
    display_name="NIFTY",
    security_id=13,
    default_output_dir="out",
)


def make_args(**overrides):
    """Parse a realistic argv, then apply overrides."""
    argv = ["--client-id", "CLIENT123", *overrides.pop("argv", [])]
    args = engine.parse_args(DEFAULTS, argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def epoch(text: str) -> int:
    """IST wall-clock string -> the epoch seconds the API would send."""
    return int(pd.Timestamp(text, tz="Asia/Kolkata").timestamp())


def payload_for(side: str, stamps: list[int], **fields):
    """Build a response block in the API's column-per-field shape."""
    block = {"timestamp": stamps}
    for name in engine.REQUIRED_DATA_FIELDS:
        block[name] = fields.get(name, [1.0] * len(stamps))
    other = "pe" if side == "ce" else "ce"
    return {side: block, other: None}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_strike_labels_are_symmetric_around_atm():
    assert engine.strike_labels(2) == ["ATM-2", "ATM-1", "ATM", "ATM+1", "ATM+2"]
    assert engine.strike_labels(0) == ["ATM"]
    assert len(engine.strike_labels(10)) == 21


def test_series_plan_pairs_every_label_with_every_option_type():
    plan = engine.series_plan(1, ["CALL", "PUT"])
    assert len(plan) == 6
    assert ("ATM", "CALL") in plan and ("ATM+1", "PUT") in plan


def test_chunk_ranges_cover_the_whole_span_without_gaps():
    windows = engine.chunk_ranges(date(2025, 1, 1), date(2025, 3, 31), 29)

    assert windows[0][0] == date(2025, 1, 1)
    # toDate is non-inclusive, so the last window ends one day past the range.
    assert windows[-1][1] == date(2025, 4, 1)
    for earlier, later in itertools.pairwise(windows):
        assert earlier[1] == later[0], "windows must abut exactly"
    assert all((end - start).days <= 29 for start, end in windows)


def test_chunk_ranges_handles_a_single_day():
    assert engine.chunk_ranges(date(2025, 1, 6), date(2025, 1, 6), 29) == [
        (date(2025, 1, 6), date(2025, 1, 7))
    ]


def test_chunk_ranges_rejects_a_reversed_or_degenerate_range():
    with pytest.raises(ValueError, match="is after end"):
        engine.chunk_ranges(date(2025, 2, 1), date(2025, 1, 1), 29)
    with pytest.raises(ValueError, match="chunk_days must be"):
        engine.chunk_ranges(date(2025, 1, 1), date(2025, 2, 1), 0)


def test_expiry_code_zero_is_rejected_because_the_api_reads_it_as_absent():
    # The live API answers expiryCode=0 with "DH-905 expiryCode is required",
    # so 1 is the near expiry here even though Dhan's annexure says otherwise.
    assert DEFAULTS.expiry_code == 1
    problems = engine.validate_args(make_args(expiry_code=0))
    assert any("expiry-code" in p for p in problems), problems


def test_validate_args_reports_every_problem_at_once():
    problems = engine.validate_args(
        make_args(chunk_days=99, strike_range=25, requests_per_second=0, option_types="CALL,BOTH")
    )
    joined = " ".join(problems)
    assert "--chunk-days" in joined
    assert "--strike-range" in joined
    assert "--requests-per-second" in joined
    assert "BOTH" in joined


def test_valid_args_produce_no_complaints():
    assert engine.validate_args(make_args()) == []


def test_describe_plan_counts_every_call_including_the_calendar_pass():
    args = make_args(strike_range=1, chunk_days=29, calendar_csv="")
    text = engine.describe_plan(args, date(2025, 1, 1), date(2025, 1, 10))

    # 3 labels x 2 types x 1 window, plus 2 calendar calls.
    assert "TOTAL API CALLS: 8" in text
    assert "series         : 6 files" in text


def test_series_csv_path_names_the_file_after_the_series():
    args = make_args()
    path = engine.series_csv_path(Path("out"), DEFAULTS, args, "ATM+3", "CALL")
    assert path.name == "nifty_1m_WEEK_ATM+3_CALL.csv"


# ---------------------------------------------------------------------------
# Envelope handling
# ---------------------------------------------------------------------------


def test_dict_remarks_is_a_real_refusal_and_is_not_retried():
    remarks = {"error_code": "DH-905", "error_type": "Input_Exception", "error_message": "bad field"}
    assert engine.classify_failure(remarks) == "fatal"


def test_str_remarks_is_indeterminate_and_is_retried():
    # The SDK stringifies transport errors into the same envelope shape as a
    # server refusal. A read is safe to repeat, so this one retries.
    assert engine.classify_failure("HTTPSConnectionPool: read timed out") == "retryable"


def test_a_rate_limit_refusal_is_retried_even_though_it_is_structured():
    remarks = {"error_code": "DH-908", "error_message": "Rate Limit exceeded"}
    assert engine.classify_failure(remarks) == "retryable"


def test_empty_window_is_recognised_from_either_remarks_shape():
    assert engine.is_empty_window({"error_message": "No Data available"})
    assert engine.is_empty_window("no records found")
    assert not engine.is_empty_window({"error_message": "Invalid_Authentication"})


def test_extract_payload_unwraps_the_doubled_data_key():
    inner = {"ce": {"timestamp": [1]}, "pe": None}
    assert engine.extract_payload({"data": inner}) is inner
    # A flat body (should Dhan ever stop double-wrapping) still works.
    assert engine.extract_payload(inner) is inner
    assert engine.extract_payload({"unrelated": 1}) is None
    assert engine.extract_payload("") is None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_retries_a_transport_failure_then_succeeds():
    stamps = [epoch("2025-01-06 09:15")]
    calls = []
    responses = [
        {"status": "failure", "remarks": "read timed out", "data": ""},
        {"status": "success", "remarks": "", "data": {"data": payload_for("ce", stamps)}},
    ]

    def fake(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    slept = []
    payload = engine.fetch_rolling_chunk(
        SimpleNamespace(expired_options_data=fake),
        args=make_args(max_retries=3),
        strike_label="ATM",
        option_type="CALL",
        interval=1,
        from_date=date(2025, 1, 6),
        to_date=date(2025, 1, 7),
        sleeper=slept.append,
    )

    assert len(calls) == 2
    assert slept == [1.0]
    assert payload is not None and "ce" in payload
    assert calls[0]["expiry_code"] == 1
    assert calls[0]["strike"] == "ATM"
    assert set(calls[0]["required_data"]) == set(engine.REQUIRED_DATA_FIELDS)


def test_fetch_does_not_retry_a_structured_refusal():
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return {"status": "failure", "remarks": {"error_code": "DH-905"}, "data": ""}

    with pytest.raises(RuntimeError, match="API refused"):
        engine.fetch_rolling_chunk(
            SimpleNamespace(expired_options_data=fake),
            args=make_args(max_retries=3),
            strike_label="ATM",
            option_type="CALL",
            interval=1,
            from_date=date(2025, 1, 6),
            to_date=date(2025, 1, 7),
            sleeper=lambda _s: None,
        )

    assert len(calls) == 1, "a refusal must not be repeated"


def test_fetch_returns_none_for_a_window_that_simply_has_no_data():
    fake = SimpleNamespace(
        expired_options_data=lambda **_k: {"status": "failure", "remarks": {"error_message": "No Data"}, "data": ""}
    )
    assert (
        engine.fetch_rolling_chunk(
            fake,
            args=make_args(),
            strike_label="ATM-10",
            option_type="PUT",
            interval=1,
            from_date=date(2025, 1, 6),
            to_date=date(2025, 1, 7),
            sleeper=lambda _s: None,
        )
        is None
    )


def test_fetch_gives_up_after_the_retry_budget():
    fake = SimpleNamespace(expired_options_data=lambda **_k: {"status": "failure", "remarks": "timeout", "data": ""})
    with pytest.raises(RuntimeError, match="API failed 3x"):
        engine.fetch_rolling_chunk(
            fake,
            args=make_args(max_retries=2),
            strike_label="ATM",
            option_type="CALL",
            interval=1,
            from_date=date(2025, 1, 6),
            to_date=date(2025, 1, 7),
            sleeper=lambda _s: None,
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_builds_ist_rows_from_the_parallel_arrays():
    stamps = [epoch("2025-01-06 09:15"), epoch("2025-01-06 09:16")]
    payload = payload_for(
        "ce", stamps, close=[142.5, 143.8], strike=[21500.0, 21500.0], spot=[21503.2, 21510.0], oi=[1250, 1300]
    )

    frame = engine.normalize_rolling_payload(payload, "CALL", "ATM+3")

    assert list(frame.columns) == list(engine.OUTPUT_COLUMNS)
    assert len(frame) == 2
    assert str(frame["timestamp"].iloc[0]) == "2025-01-06 09:15:00"
    assert frame["strike_price"].iloc[0] == 21500.0
    assert frame["spot"].iloc[1] == 21510.0
    assert frame["strike_label"].iloc[0] == "ATM+3"
    assert frame["option_type"].iloc[0] == "CALL"


def test_normalize_reads_the_pe_block_for_a_put():
    stamps = [epoch("2025-01-06 09:15")]
    payload = payload_for("pe", stamps, close=[88.0])

    assert engine.normalize_rolling_payload(payload, "PUT", "ATM").iloc[0]["close"] == 88.0
    # The CE side is null in that same payload, so a CALL request finds nothing.
    assert engine.normalize_rolling_payload(payload, "CALL", "ATM").empty


def test_normalize_refuses_to_zip_misaligned_arrays():
    # A short field would silently pair prices with the wrong timestamps.
    payload = {"ce": {"timestamp": [1, 2, 3], "close": [10.0, 11.0]}, "pe": None}
    with pytest.raises(engine.MarketDataValidationError, match="has 2 values for 3 timestamps"):
        engine.normalize_rolling_payload(payload, "CALL", "ATM")


def test_normalize_tolerates_a_field_the_server_left_empty():
    stamps = [epoch("2025-01-06 09:15")]
    block = {"timestamp": stamps, "close": [10.0], "open": [10.0], "high": [10.0], "low": [10.0], "iv": []}
    frame = engine.normalize_rolling_payload({"ce": block, "pe": None}, "CALL", "ATM")

    assert len(frame) == 1
    assert pd.isna(frame["iv"].iloc[0])


def test_normalize_handles_empty_and_missing_payloads():
    assert engine.normalize_rolling_payload(None, "CALL", "ATM").empty
    assert engine.normalize_rolling_payload({"ce": None, "pe": None}, "CALL", "ATM").empty
    assert engine.normalize_rolling_payload({"ce": {"timestamp": []}, "pe": None}, "CALL", "ATM").empty


def test_normalize_rejects_a_chunk_mixing_epoch_units():
    stamps = [epoch("2025-01-06 09:15"), epoch("2025-01-06 09:16") * 1000]
    with pytest.raises(engine.MarketDataValidationError, match="mixes epoch units"):
        engine.normalize_rolling_payload(payload_for("ce", stamps), "CALL", "ATM")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def frame_from(rows):
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_validation_accepts_a_zero_priced_deep_otm_candle():
    # market_data_health.validate_ohlc_frame would reject this outright, which
    # is why this module has its own validator.
    frame = frame_from(
        [{"timestamp": "2025-01-06 09:15", "open": 0.0, "high": 0.05, "low": 0.0, "close": 0.05,
          "volume": 0, "oi": 0}]
    )
    engine.validate_options_frame(frame, date(2025, 1, 6), date(2025, 1, 7), "ATM-10_PUT")


def test_validation_rejects_a_candle_outside_the_requested_window():
    frame = frame_from(
        [{"timestamp": "2025-01-09 09:15", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
          "volume": 1, "oi": 1}]
    )
    with pytest.raises(engine.MarketDataValidationError, match="outside the requested window"):
        engine.validate_options_frame(frame, date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")


def test_validation_rejects_duplicate_and_unsorted_timestamps():
    one_bar = {
        "timestamp": "2025-01-06 09:15", "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1, "oi": 1,
    }
    dupe = frame_from([one_bar, dict(one_bar)])
    with pytest.raises(engine.MarketDataValidationError, match="duplicate timestamps"):
        engine.validate_options_frame(dupe, date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")

    backwards = frame_from(
        [
            {"timestamp": "2025-01-06 09:16", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "oi": 1},
            {"timestamp": "2025-01-06 09:15", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "oi": 1},
        ]
    )
    with pytest.raises(engine.MarketDataValidationError, match="not ascending"):
        engine.validate_options_frame(backwards, date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")


def test_validation_rejects_impossible_candle_geometry_and_negatives():
    inverted = frame_from(
        [{"timestamp": "2025-01-06 09:15", "open": 5.0, "high": 1.0, "low": 9.0, "close": 5.0,
          "volume": 1, "oi": 1}]
    )
    with pytest.raises(engine.MarketDataValidationError, match="high < low"):
        engine.validate_options_frame(inverted, date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")

    negative = frame_from(
        [{"timestamp": "2025-01-06 09:15", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
          "volume": -5, "oi": 1}]
    )
    with pytest.raises(engine.MarketDataValidationError, match="negative volume"):
        engine.validate_options_frame(negative, date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")


def test_validation_says_nothing_about_an_empty_chunk():
    engine.validate_options_frame(pd.DataFrame(), date(2025, 1, 6), date(2025, 1, 7), "ATM_CALL")


# ---------------------------------------------------------------------------
# Expiry labelling
# ---------------------------------------------------------------------------


def test_labelling_attaches_expiry_and_drops_what_it_cannot_label():
    frame = frame_from(
        [
            {"timestamp": "2025-01-06 09:15", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
             "volume": 1, "oi": 1},
            {"timestamp": "2025-01-20 09:15", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
             "volume": 1, "oi": 1},
        ]
    )
    frame["expiry_date"] = pd.NA
    frame["days_to_expiry"] = pd.NA

    labelled = engine.label_with_expiry(frame, {date(2025, 1, 6): date(2025, 1, 9)})

    # The 20-Jan bar has no mapping, and an unlabelled bar is worse than none.
    assert len(labelled) == 1
    assert labelled["expiry_date"].iloc[0] == "2025-01-09"
    assert labelled["days_to_expiry"].iloc[0] == 3


def test_labelling_leaves_an_empty_frame_alone():
    assert engine.label_with_expiry(pd.DataFrame(), {}).empty


# ---------------------------------------------------------------------------
# Resumable writing
# ---------------------------------------------------------------------------


def written_frame(stamps: list[str]) -> pd.DataFrame:
    """A frame shaped exactly like one the engine would write.

    Built column-first so that an EMPTY stamps list still carries the full
    column set. An empty frame with no columns at all is not something the
    engine ever hands to `append_chunk`, so testing against one proves nothing.
    """
    frame = pd.DataFrame({name: [1.0] * len(stamps) for name in engine.OUTPUT_COLUMNS})
    frame["timestamp"] = pd.to_datetime(pd.Series(stamps, dtype="object"))
    return frame[list(engine.OUTPUT_COLUMNS)]


def test_append_writes_one_header_and_then_only_new_rows(tmp_path):
    target = tmp_path / "series.csv"

    rows, size, newest = engine.append_chunk(
        target, written_frame(["2025-01-06 09:15", "2025-01-06 09:16"]), last_timestamp=None
    )
    assert rows == 2 and size > 0 and newest == "2025-01-06 09:16:00"

    rows, _size, newest = engine.append_chunk(
        target, written_frame(["2025-01-06 09:16", "2025-01-06 09:17"]), last_timestamp=newest
    )
    # The overlapping boundary bar is dropped, not duplicated.
    assert rows == 1 and newest == "2025-01-06 09:17:00"

    text = target.read_text(encoding="utf-8")
    assert text.count("timestamp,open") == 1
    assert text.count("2025-01-06 09:16:00") == 1


def test_append_of_an_empty_chunk_is_a_no_op(tmp_path):
    target = tmp_path / "series.csv"
    rows, size, newest = engine.append_chunk(target, written_frame([]), last_timestamp="2025-01-06 09:15:00")

    assert (rows, size, newest) == (0, 0, "2025-01-06 09:15:00")
    assert not target.exists()


def test_truncate_discards_a_half_written_chunk(tmp_path):
    target = tmp_path / "series.csv"
    # Bytes, not text. On Windows a text-mode write turns each \n into \r\n, so
    # a byte offset computed from the text would not match the file on disk --
    # and byte offsets agreeing with the file is the entire point of truncation.
    target.write_bytes(b"header\ngood\nPARTIAL")
    good_length = len(b"header\ngood\n")

    engine.truncate_to(target, good_length)

    assert target.read_bytes() == b"header\ngood\n"


def test_truncate_leaves_a_shorter_or_missing_file_alone(tmp_path):
    target = tmp_path / "series.csv"
    target.write_text("short", encoding="utf-8")
    engine.truncate_to(target, 9999)
    assert target.read_text(encoding="utf-8") == "short"

    engine.truncate_to(tmp_path / "absent.csv", 10)  # must not raise


def test_manifest_round_trips_and_treats_damage_as_start_over(tmp_path):
    path = tmp_path / "_manifest.json"
    assert engine.load_manifest(path) == {}

    engine.save_manifest(path, {"ATM_CALL": {"rows": 5, "bytes": 120}})
    assert engine.load_manifest(path)["ATM_CALL"]["rows"] == 5
    assert json.loads(path.read_text(encoding="utf-8"))["ATM_CALL"]["bytes"] == 120

    path.write_text("{ this is not json", encoding="utf-8")
    assert engine.load_manifest(path) == {}


# ---------------------------------------------------------------------------
# Calendar plumbing and verification
# ---------------------------------------------------------------------------


def test_trading_days_are_read_out_of_the_downloaded_frames():
    frame = written_frame(["2025-01-06 09:15", "2025-01-06 15:29", "2025-01-07 09:15"])
    assert engine.trading_days_from_frames([frame]) == [date(2025, 1, 6), date(2025, 1, 7)]
    assert engine.trading_days_from_frames([pd.DataFrame()]) == []


def test_calendar_csv_round_trips(tmp_path):
    days = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 9)]
    expiry_map = dict.fromkeys(days, date(2025, 1, 9))
    path = tmp_path / engine.CALENDAR_FILENAME

    frame = engine.write_calendar_csv(path, days, expiry_map)
    assert list(frame.columns) == list(engine.CALENDAR_COLUMNS)
    assert frame["days_to_expiry"].tolist() == [3, 2, 0]
    assert frame["trading_days_to_expiry"].tolist() == [2, 1, 0]

    assert engine.read_calendar_csv(path) == expiry_map


def test_reading_a_calendar_missing_its_columns_says_which(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("trade_date\n2025-01-06\n", encoding="utf-8")
    with pytest.raises(ValueError, match="weekly_expiry_date"):
        engine.read_calendar_csv(path)


def test_verification_passes_when_the_straddle_collapses_on_expiry_day():
    calls = written_frame(["2025-01-09 15:29"])
    calls.loc[0, "close"] = 8.0
    puts = written_frame(["2025-01-09 15:29"])
    puts.loc[0, "close"] = 6.0

    assert engine.verify_expiries(calls, puts, [date(2025, 1, 9)]) == []


def test_verification_flags_a_day_that_still_carries_time_value():
    # A calendar that is off by a day looks exactly like this.
    calls = written_frame(["2025-01-08 15:29"])
    calls.loc[0, "close"] = 90.0
    puts = written_frame(["2025-01-08 15:29"])
    puts.loc[0, "close"] = 85.0

    complaints = engine.verify_expiries(calls, puts, [date(2025, 1, 8)])
    assert len(complaints) == 1
    assert "may not be an expiry day" in complaints[0]


def test_verification_reports_a_derived_expiry_with_no_data_at_all():
    calls = puts = written_frame(["2025-01-09 15:29"])
    complaints = engine.verify_expiries(calls, puts, [date(2025, 1, 16)])
    assert "has no ATM close" in complaints[0]

    assert engine.verify_expiries(pd.DataFrame(), pd.DataFrame(), [date(2025, 1, 9)]) == [
        "no ATM data available to verify expiries against"
    ]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_token_is_environment_only_and_never_a_command_line_flag(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "env-only-token")
    monkeypatch.delenv("DHAN_TOKEN_ID", raising=False)

    assert engine.parse_args(DEFAULTS, ["--client-id", "C1"]).access_token == "env-only-token"

    with pytest.raises(SystemExit):
        engine.parse_args(DEFAULTS, ["--access-token", "cli-token"])


def test_build_client_refuses_to_start_without_credentials(monkeypatch):
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_TOKEN_ID", raising=False)
    args = engine.parse_args(DEFAULTS, ["--client-id", "C1"])

    with pytest.raises(ValueError, match="Missing credentials"):
        engine.build_client(args)


# ---------------------------------------------------------------------------
# Downloading one series, and resuming it
# ---------------------------------------------------------------------------


def one_day_client(close_by_day: dict[str, float]):
    """A fake client that serves one bar at 09:15 for each day it is asked about."""

    def fake(**kwargs):
        start = date.fromisoformat(kwargs["from_date"])
        end = date.fromisoformat(kwargs["to_date"])
        stamps, closes = [], []
        for day, close in close_by_day.items():
            when = date.fromisoformat(day)
            if start <= when < end:
                stamps.append(epoch(f"{day} 09:15"))
                closes.append(close)
        if not stamps:
            return {"status": "failure", "remarks": {"error_message": "No Data"}, "data": ""}
        # Flat candles (o == h == l == c). Anything else here would trip the
        # engine's own geometry check, which is the correct behaviour but not
        # what these tests are about.
        block = payload_for("ce", stamps, close=closes, open=closes, high=closes, low=closes)
        return {"status": "success", "remarks": "", "data": {"data": block}}

    return SimpleNamespace(expired_options_data=fake)


EXPIRY_MAP = {
    date(2025, 1, 6): date(2025, 1, 9),
    date(2025, 1, 7): date(2025, 1, 9),
    date(2025, 1, 8): date(2025, 1, 9),
    date(2025, 1, 9): date(2025, 1, 9),
}


def test_download_series_writes_rows_and_records_its_progress(tmp_path):
    client = one_day_client({"2025-01-06": 100.0, "2025-01-08": 120.0})
    manifest: dict = {}
    manifest_path = tmp_path / engine.MANIFEST_FILENAME
    windows = [(date(2025, 1, 6), date(2025, 1, 8)), (date(2025, 1, 8), date(2025, 1, 10))]

    rows = engine.download_series(
        client, make_args(), DEFAULTS, tmp_path, "ATM", "CALL",
        windows, EXPIRY_MAP, manifest, manifest_path, None,
    )

    assert rows == 2
    written = pd.read_csv(tmp_path / "nifty_1m_WEEK_ATM_CALL.csv")
    assert written["close"].tolist() == [100.0, 120.0]
    assert written["expiry_date"].tolist() == ["2025-01-09", "2025-01-09"]
    assert written["days_to_expiry"].tolist() == [3, 1]

    # The manifest must survive on disk, not just in memory: it is what a
    # restarted run reads.
    assert engine.load_manifest(manifest_path)["ATM_CALL"]["rows"] == 2


def test_download_series_resumes_without_refetching_finished_windows(tmp_path):
    windows = [(date(2025, 1, 6), date(2025, 1, 8)), (date(2025, 1, 8), date(2025, 1, 10))]
    calls: list = []

    def counting_client(**kwargs):
        calls.append(kwargs["from_date"])
        return one_day_client({"2025-01-08": 120.0}).expired_options_data(**kwargs)

    # First window already done, per the manifest.
    csv_path = tmp_path / "nifty_1m_WEEK_ATM_CALL.csv"
    header = ",".join(engine.OUTPUT_COLUMNS) + "\n"
    first_row = "2025-01-06 09:15:00,1,1,1,100.0,1,1,1,21500,21500,ATM,CALL,2025-01-09,3\n"
    csv_path.write_bytes((header + first_row).encode("utf-8"))
    manifest = {
        "ATM_CALL": {
            "last_to_date": "2025-01-08",
            "last_timestamp": "2025-01-06 09:15:00",
            "bytes": csv_path.stat().st_size,
            "rows": 1,
        }
    }

    rows = engine.download_series(
        SimpleNamespace(expired_options_data=counting_client),
        make_args(), DEFAULTS, tmp_path, "ATM", "CALL",
        windows, EXPIRY_MAP, manifest, tmp_path / engine.MANIFEST_FILENAME, None,
    )

    assert calls == ["2025-01-08"], "the finished window must not be requested again"
    assert rows == 2, "the resumed count continues from the manifest, not from zero"
    written = pd.read_csv(csv_path)
    assert written["close"].tolist() == [100.0, 120.0]


def test_download_series_discards_a_chunk_left_half_written_by_a_crash(tmp_path):
    csv_path = tmp_path / "nifty_1m_WEEK_ATM_CALL.csv"
    header = ",".join(engine.OUTPUT_COLUMNS) + "\n"
    first_row = "2025-01-06 09:15:00,1,1,1,100.0,1,1,1,21500,21500,ATM,CALL,2025-01-09,3\n"
    good_bytes = len((header + first_row).encode("utf-8"))
    # Simulate the crash: bytes on disk that the manifest never acknowledged.
    csv_path.write_bytes((header + first_row + "2025-01-07 09:15:00,1,1,1,GARB").encode("utf-8"))

    manifest = {
        "ATM_CALL": {
            "last_to_date": "2025-01-08",
            "last_timestamp": "2025-01-06 09:15:00",
            "bytes": good_bytes,
            "rows": 1,
        }
    }

    engine.download_series(
        one_day_client({"2025-01-08": 120.0}),
        make_args(), DEFAULTS, tmp_path, "ATM", "CALL",
        [(date(2025, 1, 8), date(2025, 1, 10))], EXPIRY_MAP, manifest,
        tmp_path / engine.MANIFEST_FILENAME, None,
    )

    written = pd.read_csv(csv_path)
    assert written["close"].tolist() == [100.0, 120.0], "the torn row must be gone"


def test_no_resume_starts_the_series_from_scratch(tmp_path):
    csv_path = tmp_path / "nifty_1m_WEEK_ATM_CALL.csv"
    csv_path.write_bytes(b"stale contents that must not survive\n")
    manifest = {"ATM_CALL": {"last_to_date": "2025-01-08", "last_timestamp": None, "bytes": 5, "rows": 99}}

    rows = engine.download_series(
        one_day_client({"2025-01-06": 100.0}),
        make_args(no_resume=True), DEFAULTS, tmp_path, "ATM", "CALL",
        [(date(2025, 1, 6), date(2025, 1, 8))], EXPIRY_MAP, manifest,
        tmp_path / engine.MANIFEST_FILENAME, None,
    )

    assert rows == 1
    assert "stale contents" not in csv_path.read_text(encoding="utf-8")


def test_dry_run_prints_the_plan_and_makes_no_calls(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setattr(
        engine, "build_client", lambda _a: pytest.fail("--dry-run must not build a client")
    )

    engine.run_expired_options_fetcher(
        DEFAULTS,
        [
            "--client-id", "C1",
            "--start-date", "2025-01-06", "--end-date", "2025-01-10",
            "--strike-range", "1", "--output-dir", str(tmp_path), "--dry-run",
        ],
    )

    printed = capsys.readouterr().out
    assert "TOTAL API CALLS: 8" in printed
    assert not list(tmp_path.iterdir()), "--dry-run must not write anything"


def test_bad_arguments_stop_the_run_before_any_network_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setattr(engine, "build_client", lambda _a: pytest.fail("must not reach the client"))

    with pytest.raises(ValueError, match="Invalid arguments"):
        engine.run_expired_options_fetcher(
            DEFAULTS,
            ["--client-id", "C1", "--strike-range", "99", "--output-dir", str(tmp_path)],
        )
