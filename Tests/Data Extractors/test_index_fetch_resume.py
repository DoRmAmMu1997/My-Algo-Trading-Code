"""Resume behaviour of the shared index fetcher.

A five-year pull is ~21 requests over ~10 minutes, and before this the chunks
were accumulated in memory and written once at the end -- so a failure in the
last request threw away every earlier one. These tests pin the resume contract
that replaced that, and in particular the two ways a resume can be WRONG rather
than merely slow: honouring progress that describes a different run (which would
silently skip years), and appending to a file there is no last timestamp to
de-duplicate against (which would write the history twice).

Nothing here touches the network; `fetch_chunk` is always replaced.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

# Tests/Data Extractors/<this file> -> the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = _REPO_ROOT / "Data Extractors" / "index_1m_5y_data_fetch_dhan_common.py"
spec = importlib.util.spec_from_file_location("index_1m_5y_data_fetch_dhan_common", MODULE_PATH)
fetcher = importlib.util.module_from_spec(spec)
sys.modules["index_1m_5y_data_fetch_dhan_common"] = fetcher
spec.loader.exec_module(fetcher)

DEFAULTS = SimpleNamespace(display_name="NIFTY")


def _args(tmp_path: Path, **overrides):
    """A parsed-argument stand-in. Two chunks: 01-01..01-05 and 01-06..01-10."""

    # Built from a dict (not keyword args) so Bandit's B106 does not read the
    # dummy `access_token` literal as a hardcoded password.
    fields = {
        "client_id": "CLIENT123", "access_token": "dummy-token",
        "exchange_segment": "IDX_I", "security_id": 13, "instrument_type": "INDEX",
        "interval": 1, "chunk_days": 5, "sleep_seconds": 0,
        "output": str(tmp_path / "index.csv"),
        "start_date": "2026-01-01", "end_date": "2026-01-10",
        "lookback": "5y", "no_resume": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _frame(day: date, count: int = 3) -> pd.DataFrame:
    """A chunk of `count` one-minute bars stamped on `day`."""

    base = pd.Timestamp(f"{day.isoformat()} 09:15:00")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(minutes=i) for i in range(count)],
            "open": [100.0 + i for i in range(count)],
            "high": [101.0 + i for i in range(count)],
            "low": [99.0 + i for i in range(count)],
            "close": [100.5 + i for i in range(count)],
            "volume": [0] * count,
        }
    )


def _responder(*, fail_on: date | None = None, empty: bool = False):
    """A `fetch_chunk` replacement that records the windows it was asked for."""

    seen: list[date] = []

    def respond(**kwargs):
        start = kwargs["chunk_start"]
        seen.append(start)
        if fail_on is not None and start == fail_on:
            raise RuntimeError("network died")
        return pd.DataFrame() if empty else _frame(start)

    return respond, seen


def _run(args, respond) -> None:
    with (
        patch.object(fetcher, "parse_args", return_value=args),
        patch.object(fetcher, "DhanContext"),
        patch.object(fetcher, "dhanhq"),
        patch.object(fetcher, "fetch_chunk", side_effect=respond),
    ):
        fetcher.run_index_fetcher(DEFAULTS)


def _manifest(args) -> dict:
    return json.loads(Path(args.output + ".manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the point


def test_an_interrupted_run_resumes_from_the_last_completed_chunk(tmp_path):
    """The whole reason this exists: chunk 1 survives chunk 2 failing."""

    args = _args(tmp_path)

    respond, first_seen = _responder(fail_on=date(2026, 1, 6))
    with pytest.raises(RuntimeError):
        _run(args, respond)

    # The completed chunk is on disk, and the manifest says how far it got.
    saved = pd.read_csv(args.output)
    assert len(saved) == 3
    assert _manifest(args)["progress"]["last_to_date"] == "2026-01-05"
    assert first_seen == [date(2026, 1, 1), date(2026, 1, 6)]

    # The re-run asks for the failed chunk ONLY -- the first is not downloaded
    # again, which is the entire saving.
    respond, second_seen = _responder()
    _run(args, respond)

    assert second_seen == [date(2026, 1, 6)]
    saved = pd.read_csv(args.output)
    assert len(saved) == 6
    assert saved["timestamp"].is_monotonic_increasing
    assert not saved["timestamp"].duplicated().any()


def test_a_completed_run_re_requests_nothing(tmp_path):
    """Re-running a finished pull is free, not a second download."""

    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)

    respond, seen = _responder()
    _run(args, respond)

    assert seen == []
    assert len(pd.read_csv(args.output)) == 6


# ------------------------------------------------- the ways it could be wrong


def test_a_manifest_from_a_different_run_is_discarded_and_the_file_restarted(tmp_path):
    """Progress from another run would silently skip history, so it is refused.

    The dangerous shape is a narrow earlier run followed by a wide one: every
    chunk ending before the stored point would be skipped and the command would
    report success over a file missing most of its history.
    """

    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)

    # Same output, different interval -- the rows would MEAN something else.
    changed = _args(tmp_path, interval=5)
    respond, seen = _responder()
    _run(changed, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)], "every chunk must be fetched again"
    assert len(pd.read_csv(changed.output)) == 6, "the file was restarted, not appended to"


def test_a_manifest_whose_csv_vanished_is_ignored(tmp_path):
    """Progress describing a file that is gone must not be believed."""

    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)
    Path(args.output).unlink()

    respond, seen = _responder()
    _run(args, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)]
    assert len(pd.read_csv(args.output)) == 6


def test_a_csv_with_no_manifest_is_restarted_not_appended_to(tmp_path):
    """Without a last timestamp there is nothing to de-duplicate against."""

    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)
    Path(args.output + ".manifest.json").unlink()

    respond, seen = _responder()
    _run(args, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)]
    saved = pd.read_csv(args.output)
    assert len(saved) == 6, "the history must not be written twice"
    assert not saved["timestamp"].duplicated().any()


def test_empty_chunks_still_move_the_resume_point(tmp_path):
    """A run of holidays must not be re-requested on the next attempt."""

    args = _args(tmp_path)
    respond, _ = _responder(empty=True)
    _run(args, respond)

    assert _manifest(args)["progress"]["last_to_date"] == "2026-01-10"
    assert _manifest(args)["progress"]["rows"] == 0

    respond, seen = _responder(empty=True)
    _run(args, respond)
    assert seen == []


# ------------------------------------------------------------- the mechanics


def test_no_resume_writes_one_atomic_file_and_leaves_no_manifest(tmp_path):
    """The escape hatch keeps the original all-at-once behaviour."""

    args = _args(tmp_path, no_resume=True)
    respond, seen = _responder()
    _run(args, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)]
    assert len(pd.read_csv(args.output)) == 6
    assert not Path(args.output + ".manifest.json").exists()


def test_append_drops_rows_at_or_before_the_last_timestamp(tmp_path):
    """What makes a re-run idempotent and absorbs a duplicated boundary bar."""

    target = tmp_path / "index.csv"
    frame = _frame(date(2026, 1, 1))

    written, _, newest = fetcher.append_chunk(target, frame, last_timestamp=None)
    assert written == 3

    written_again, _, newest_again = fetcher.append_chunk(
        target, frame, last_timestamp=newest
    )
    assert written_again == 0
    assert newest_again == newest
    assert len(pd.read_csv(target)) == 3


def test_truncate_rolls_back_a_partly_written_chunk(tmp_path):
    """The manifest is written after the rows, so the file can run ahead of it."""

    target = tmp_path / "index.csv"
    good = b"timestamp,close\n2026-01-01 09:15:00,100.0\n"
    target.write_bytes(good + b"2026-01-01 09:16:00,10")

    fetcher.truncate_to(target, len(good))

    assert target.read_bytes() == good


def test_a_damaged_manifest_means_start_over_not_a_crash(tmp_path):
    """Anything unreadable is treated as no progress at all."""

    args = _args(tmp_path)
    Path(args.output + ".manifest.json").write_text("{not json", encoding="utf-8")

    respond, seen = _responder()
    _run(args, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)]
    assert len(pd.read_csv(args.output)) == 6


def test_a_manifest_with_nonsense_types_does_not_crash_the_run(tmp_path):
    """Manifest values are `object` until proved otherwise."""

    assert fetcher.manifest_int("not a number") == 0
    assert fetcher.manifest_int(True) == 0, "a bool is not a row count"
    assert fetcher.manifest_int(7) == 7
    assert fetcher.manifest_text(7) is None
    assert fetcher.manifest_text("2026-01-05") == "2026-01-05"


def test_a_manifest_describing_a_different_file_is_refused(tmp_path):
    """The byte count says how MUCH; the first row says WHICH.

    Same filename, different contents -- a restore, a hand edit, a rewrite by
    another tool -- and the stored length still "fits". Truncating on it would
    cut at a boundary that means nothing, and the resume would then append from
    the stored last timestamp, straight past whatever was lost. The result
    still looks ascending and de-duplicated, which is what makes it worth
    catching.
    """

    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)

    # Same name, different file.
    Path(args.output).write_text(
        "timestamp,open,high,low,close,volume\n2020-01-01 09:15:00,1,2,0.5,1.5,0\n",
        encoding="utf-8",
    )

    respond, seen = _responder()
    _run(args, respond)

    assert seen == [date(2026, 1, 1), date(2026, 1, 6)], "every chunk must be fetched again"
    saved = pd.read_csv(args.output)
    assert len(saved) == 6
    assert "2020-01-01 09:15:00" not in set(saved["timestamp"]), "the foreign file is gone"


def test_the_manifest_records_which_file_it_describes(tmp_path):
    args = _args(tmp_path)
    respond, _ = _responder()
    _run(args, respond)

    progress = _manifest(args)["progress"]
    with Path(args.output).open(encoding="utf-8") as handle:
        handle.readline()
        first_data_row = handle.readline().strip()

    assert progress["first_row"] == first_data_row
