"""The resume point must never claim a session that is not over.

On 2026-10-06 a 09:01 IST run fetched the final chunk, which ended on that day,
and recorded `last_to_date = 2026-10-06` although the file's newest bar was
2026-10-05 15:29 -- the market had not opened. The 16:00 run then skipped every
chunk as "already fetched" and reported success without the day's session.

These tests drive the same two runs through a settable clock. Nothing here
touches the network; `fetch_chunk` is always replaced.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

# Tests/Data Extractors/<this file> -> the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = _REPO_ROOT / "Data Extractors" / "index_1m_5y_data_fetch_dhan_common.py"
spec = importlib.util.spec_from_file_location("index_1m_5y_data_fetch_dhan_common", MODULE_PATH)
fetcher = importlib.util.module_from_spec(spec)
sys.modules["index_1m_5y_data_fetch_dhan_common"] = fetcher
spec.loader.exec_module(fetcher)

DEFAULTS = SimpleNamespace(display_name="NIFTY")

# Saturday 10 Jan 2026 would be no use as "today"; Friday 9 Jan is a session.
SESSION_DAY = date(2026, 1, 9)
BEFORE_THE_OPEN = datetime(2026, 1, 9, 9, 1)
DURING_THE_SESSION = datetime(2026, 1, 9, 11, 0)
AFTER_THE_CLOSE = datetime(2026, 1, 9, 16, 0)


class _Clock(datetime):
    """`datetime` with a settable `now()`; the fetcher's IST zone is ignored."""

    now_value = BEFORE_THE_OPEN

    @classmethod
    def now(cls, tz=None):
        return cls.now_value


def _lookback_args(tmp_path: Path):
    """A rolling 7-day lookback in 5-day chunks: 01-02..01-06 and 01-07..01-09."""

    # Built from a dict (not keyword args) so Bandit's B106 does not read the
    # dummy `access_token` literal as a hardcoded password.
    fields = {
        "client_id": "CLIENT123", "access_token": "dummy-token",
        "exchange_segment": "IDX_I", "security_id": 13, "instrument_type": "INDEX",
        "interval": 1, "chunk_days": 5, "sleep_seconds": 0,
        "output": str(tmp_path / "index.csv"),
        "start_date": "", "end_date": "", "lookback": "7d", "no_resume": False,
    }
    return SimpleNamespace(**fields)


def _exchange(now: datetime):
    """A `fetch_chunk` that serves only bars that exist at `now`.

    Every weekday gets three bars from 09:15, except that bars on `now`'s own
    date exist only up to `now` (none at all before the open).
    """

    seen: list[tuple[date, date]] = []

    def respond(**kwargs):
        start, end = kwargs["chunk_start"], kwargs["chunk_end"]
        seen.append((start, end))
        stamps = []
        day = start
        while day <= end:
            if day.weekday() < 5:
                base = pd.Timestamp(f"{day.isoformat()} 09:15:00")
                stamps += [base + pd.Timedelta(minutes=i) for i in range(3)]
            day += timedelta(days=1)
        stamps = [stamp for stamp in stamps if stamp <= pd.Timestamp(now)]
        count = len(stamps)
        return pd.DataFrame(
            {
                "timestamp": stamps,
                "open": [100.0] * count, "high": [101.0] * count,
                "low": [99.0] * count, "close": [100.5] * count, "volume": [0] * count,
            }
        )

    return respond, seen


def _run_at(now: datetime, args) -> list[tuple[date, date]]:
    _Clock.now_value = now
    respond, seen = _exchange(now)
    with (
        patch.object(fetcher, "datetime", _Clock),
        patch.object(fetcher, "parse_args", return_value=args),
        patch.object(fetcher, "DhanContext"),
        patch.object(fetcher, "dhanhq"),
        patch.object(fetcher, "fetch_chunk", side_effect=respond),
    ):
        fetcher.run_index_fetcher(DEFAULTS)
    return seen


def _progress(args) -> dict:
    manifest = Path(args.output + ".manifest.json").read_text(encoding="utf-8")
    return json.loads(manifest)["progress"]


def _session_bars(args) -> pd.Series:
    stamps = pd.to_datetime(pd.read_csv(args.output)["timestamp"])
    return stamps[stamps.dt.date == SESSION_DAY]


def test_a_pre_open_run_does_not_hide_that_days_session_from_the_evening_run(tmp_path):
    """The 2026-10-06 sequence: 09:01 run, then the 16:00 run on the same day."""

    args = _lookback_args(tmp_path)
    _run_at(BEFORE_THE_OPEN, args)
    assert _session_bars(args).empty, "the market had not opened"
    assert _progress(args)["last_to_date"] == "2026-01-08", "capped at yesterday"

    seen = _run_at(AFTER_THE_CLOSE, args)

    assert any(start <= SESSION_DAY <= end for start, end in seen), "day D was requested"
    assert len(_session_bars(args)) == 3
    saved = pd.read_csv(args.output)
    assert saved["timestamp"].is_monotonic_increasing
    assert not saved["timestamp"].duplicated().any()


def test_an_intraday_run_does_not_mark_a_partial_session_complete(tmp_path):
    """A run during market hours leaves the rest of the session to the next run."""

    args = _lookback_args(tmp_path)
    _run_at(DURING_THE_SESSION, args)
    assert len(_session_bars(args)) == 3
    assert _progress(args)["last_to_date"] == "2026-01-08"

    seen = _run_at(AFTER_THE_CLOSE, args)

    assert seen == [(date(2026, 1, 7), SESSION_DAY)], "only the chunk holding day D"
    assert not pd.read_csv(args.output)["timestamp"].duplicated().any()


def test_a_rebuild_on_day_d_does_not_claim_day_d_either(tmp_path):
    """`rebuild_atomically` writes its own manifest and must apply the same cap."""

    args = _lookback_args(tmp_path)
    _run_at(BEFORE_THE_OPEN, args)
    Path(args.output + ".manifest.json").unlink()  # forces the atomic rebuild

    _run_at(BEFORE_THE_OPEN, args)

    assert _progress(args)["last_to_date"] == "2026-01-08"


def test_a_chunk_entirely_in_the_past_keeps_its_own_end():
    """Holiday chunks with no bars must still move the resume point forward."""

    _Clock.now_value = AFTER_THE_CLOSE
    with patch.object(fetcher, "datetime", _Clock):
        assert fetcher.completed_through(date(2026, 1, 1)) == date(2026, 1, 1)
        assert fetcher.completed_through(SESSION_DAY) == date(2026, 1, 8)
        assert fetcher.completed_through(date(2026, 1, 20)) == date(2026, 1, 8)
