"""Tests for the crash-durable session state (`Dependencies/session_state.py`).

The module exists because a session that dies mid-day loses its books, so the
tests concentrate on the properties that make recovery trustworthy:

* a trade event reaches the disk immediately, not at shutdown;
* the file is never observed half-written;
* nothing that could invent exposure is ever offered back for resume -- not a
  stale date, not a clean shutdown, not a live strategy; and
* a persistence failure degrades reporting, never trading.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

# Bare import: this folder's conftest.py puts the SOURCE `Dependencies/` on
# sys.path, which is the same resolution the runtime performs.
from session_state import (
    MAX_RESUMABLE_MARKS_AGE_SECONDS,
    SCHEMA_VERSION,
    SessionStateStore,
    _marks_path_for,
    load_session_state,
    recorded_realized_pnl,
    resumable_open_positions,
    serialize_position,
)


@dataclass
class _FakePosition:
    """Stand-in with the same field shape the runner's PaperPosition uses."""

    active: bool = True
    direction: str = "BULLISH"
    symbol: str = "NIFTY-Aug2026-24550-CE"
    quantity: int = 75
    entry_trade_price: float = 112.35
    entry_underlying: float = 24561.2
    stop_underlying: float = 24510.0
    target_underlying: float = 24640.0
    option_security_id: int = 43210
    option_exchange_segment: str = "NSE_FNO"
    option_right: str = "CE"
    option_strike: float = 24550.0
    option_expiry: date | None = date(2026, 8, 11)
    option_lot_size: int = 75
    option_opening_side: str = "SELL"
    live_leg: object | None = None


@pytest.fixture()
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "session_state.json"


def _store(state_path: Path, **kwargs) -> SessionStateStore:
    kwargs.setdefault("session_date", date(2026, 8, 10))
    return SessionStateStore(state_path, **kwargs)


# ---------------------------------------------------------------------------
# Position serialization
# ---------------------------------------------------------------------------
def test_serialize_position_keeps_the_entry_price_and_the_mark():
    """Entry price is the whole point: P&L is computed against it."""
    record = serialize_position(
        _FakePosition(),
        leg_marks={"option": 98.1},
        unrealized_pnl=-1068.75,
    )
    assert record is not None
    assert record["entry_trade_price"] == 112.35
    assert record["last_mark_ltp"] == 98.1
    assert record["leg_marks"] == {"option": 98.1}
    assert record["unrealized_pnl"] == -1068.75
    # The stop/target the resumed worker has to keep managing.
    assert record["stop_underlying"] == 24510.0
    assert record["target_underlying"] == 24640.0
    assert record["quantity"] == 75
    assert record["option_opening_side"] == "SELL"
    assert record["position_type"] == "_FakePosition"


def test_serialize_position_converts_dates_and_survives_json():
    record = serialize_position(_FakePosition())
    assert record is not None
    assert record["option_expiry"] == "2026-08-11"
    # Must round-trip: an unserializable field would break the whole file.
    assert json.loads(json.dumps(record))["option_expiry"] == "2026-08-11"


def test_serialize_position_returns_none_when_flat():
    assert serialize_position(_FakePosition(active=False)) is None
    assert serialize_position(None) is None


def test_serialize_position_never_persists_broker_leg_state():
    """live_leg goes stale the instant the process dies; only a flag is kept."""
    record = serialize_position(_FakePosition(live_leg=object()))
    assert record is not None
    assert "live_leg" not in record
    assert record["had_live_leg"] is True


def test_serialize_position_drops_non_finite_marks():
    """NaN would make json.dump emit invalid JSON and poison the whole file."""
    record = serialize_position(
        _FakePosition(),
        leg_marks={"option": float("nan")},
        unrealized_pnl=float("inf"),
    )
    assert record is not None
    assert "last_mark_ltp" not in record
    assert "leg_marks" not in record
    assert "unrealized_pnl" not in record
    json.dumps(record)  # must not raise


def test_serialize_position_handles_multi_leg_marks():
    @dataclass
    class _Hedged:
        active: bool = True
        main_security_id: int = 1
        hedge_security_id: int = 2

    record = serialize_position(_Hedged(), leg_marks={"main": 40.0, "hedge": 12.5})
    assert record is not None
    assert record["leg_marks"] == {"main": 40.0, "hedge": 12.5}
    # No `option` leg, so no top-level single-leg convenience field.
    assert "last_mark_ltp" not in record


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_trade_event_is_written_immediately(state_path: Path):
    """The crash this module guards against happens between trades."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})

    on_disk = load_session_state(state_path)
    assert on_disk is not None
    assert on_disk["trades"][0]["pnl"] == -929.5
    assert on_disk["strategies"]["Renko"]["recorded_pnl"] == -929.5
    assert on_disk["strategies"]["Renko"]["recorded_trades"] == 1


def test_only_pnl_bearing_events_move_the_rollup(state_path: Path):
    store = _store(state_path)
    store.record_trade_event({"action": "ENTRY", "strategy": "Renko"})
    store.record_trade_event({"action": "EXIT_FAILED", "strategy": "Renko"})
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 100.0})

    state = store.snapshot()
    # Every event is kept for forensics...
    assert len(state["trades"]) == 3
    # ...but only the EXIT counts toward realized P&L.
    assert state["strategies"]["Renko"]["recorded_trades"] == 1
    assert state["strategies"]["Renko"]["recorded_pnl"] == 100.0


def test_realized_pnl_accumulates_across_trades(state_path: Path):
    store = _store(state_path)
    for pnl in (-5195.25, 848.25, 442.0):
        store.record_trade_event({"action": "EXIT", "strategy": "SL Hunting", "pnl": pnl})
    assert recorded_realized_pnl(store.snapshot())["SL Hunting"] == -3905.0


def test_snapshot_is_throttled_but_force_overrides(state_path: Path):
    store = _store(state_path, snapshot_interval_seconds=3600.0)
    assert store.update_worker_snapshot([{"strategy": "Renko"}], force=True) is True
    # Second call inside the interval must not write.
    assert store.update_worker_snapshot([{"strategy": "Renko"}]) is False
    assert store.update_worker_snapshot([{"strategy": "Renko"}], force=True) is True


def test_snapshot_removes_a_closed_position(state_path: Path):
    """A stale open_position left behind would look resumable."""
    store = _store(state_path)
    position = serialize_position(_FakePosition())
    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": position}], force=True
    )
    assert "open_position" in store.snapshot()["strategies"]["Renko"]

    store.update_worker_snapshot([{"strategy": "Renko", "open_position": None}], force=True)
    assert "open_position" not in store.snapshot()["strategies"]["Renko"]


def test_snapshot_ignores_entries_without_a_strategy_name(state_path: Path):
    store = _store(state_path)
    store.update_worker_snapshot([{"strategy": "  "}, {"realized_pnl": 1.0}], force=True)
    assert store.snapshot()["strategies"] == {}


def test_trade_records_are_capped_keeping_the_newest(state_path: Path):
    store = _store(state_path, max_trade_records=3)
    for index in range(6):
        store.record_trade_event({"action": "ENTRY", "strategy": "Renko", "seq": index})
    kept = [record["seq"] for record in store.snapshot()["trades"]]
    assert kept == [3, 4, 5]


def test_write_is_atomic_and_leaves_no_temp_file(state_path: Path):
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 1.0})
    assert state_path.exists()
    assert not state_path.with_name(state_path.name + ".tmp").exists()
    # A complete, parseable document -- never a truncated one.
    json.loads(state_path.read_text(encoding="utf-8"))


def test_concurrent_writers_do_not_corrupt_the_file(state_path: Path):
    """~29 worker threads share one store; the file must stay valid JSON."""
    store = _store(state_path)

    def write(index: int) -> None:
        for _ in range(20):
            store.record_trade_event(
                {"action": "EXIT", "strategy": f"S{index}", "pnl": 1.0}
            )

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state = load_session_state(state_path)
    assert state is not None
    assert len(state["trades"]) == 160
    assert all(entry["recorded_pnl"] == 20.0 for entry in state["strategies"].values())


def test_persistence_failure_never_raises(tmp_path: Path):
    """A full disk or a bad path is a reporting problem, not a trading one."""
    # A directory where the file should be: every write attempt fails.
    bad_path = tmp_path / "state"
    bad_path.mkdir()
    store = SessionStateStore(bad_path, session_date=date(2026, 8, 10))

    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 1.0})
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    store.mark_clean_shutdown()
    # The write really did fail (guards against this test passing vacuously)...
    assert store._write_failure_logged is True
    # ...but in-memory bookkeeping still worked; only the file did not.
    assert store.snapshot()["strategies"]["Renko"]["recorded_pnl"] == 1.0


def test_unserializable_event_payload_is_dropped_not_fatal(state_path: Path):
    store = _store(state_path)
    store.record_trade_event(
        {"action": "EXIT", "strategy": "Renko", "pnl": 5.0, "handle": object()}
    )
    state = load_session_state(state_path)
    assert state is not None
    assert "handle" not in state["trades"][0]
    assert state["strategies"]["Renko"]["recorded_pnl"] == 5.0


def test_mark_clean_shutdown_flips_the_flag(state_path: Path):
    store = _store(state_path)
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    assert load_session_state(state_path)["clean_shutdown"] is False
    store.mark_clean_shutdown(results_published=False)
    state = load_session_state(state_path)
    assert state["clean_shutdown"] is True
    assert state["results_published"] is False


def test_restart_archives_old_file_and_carries_same_day_trade_book(state_path: Path):
    """A normal restart must not erase the crash evidence it was built to save.

    The first store represents a process that banked a loss and then died with
    a paper position open.  Constructing the replacement store is the exact
    startup boundary that used to replace that file with an empty document.
    """

    crashed = _store(state_path)
    crashed.record_trade_event(
        {"action": "EXIT", "strategy": "Renko", "pnl": -1250.0}
    )
    crashed.update_worker_snapshot(
        [
            {
                "strategy": "Renko",
                "realized_pnl": -1250.0,
                "completed_trades": 1,
                "open_position": serialize_position(_FakePosition()),
            }
        ],
        force=True,
    )

    replacement = _store(state_path)
    replacement.update_worker_snapshot(
        [{"strategy": "Renko", "realized_pnl": -1250.0, "completed_trades": 1}],
        force=True,
    )

    current = load_session_state(state_path)
    assert current is not None
    assert recorded_realized_pnl(current) == {"Renko": -1250.0}
    assert [trade["pnl"] for trade in current["trades"]] == [-1250.0]
    # An old open position is forensic evidence, not an automatically invented
    # position in the replacement process.  The exact old document is archived.
    assert "open_position" not in current["strategies"]["Renko"]
    assert replacement.previous_state is not None
    assert replacement.archive_path is not None
    archived = load_session_state(replacement.archive_path)
    assert archived is not None
    assert archived["strategies"]["Renko"]["open_position"]["active"] is True


def test_restart_does_not_carry_a_different_trading_days_book(state_path: Path):
    old = _store(state_path)
    old.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -50.0})

    replacement = SessionStateStore(
        state_path,
        session_date=date(2026, 8, 11),
    )

    assert replacement.previous_state is not None
    assert replacement.archive_path is not None
    assert replacement.snapshot()["trades"] == []
    assert replacement.snapshot()["strategies"] == {}


def test_failed_snapshot_marks_old_position_unsafe_to_resume(state_path: Path):
    """A stale position may remain for forensics but can never be resumed."""

    store = _store(state_path)
    store.update_worker_snapshot(
        [
            {
                "strategy": "Renko",
                "live_trading": False,
                "execution_mode": "PAPER",
                "open_position": serialize_position(_FakePosition()),
            }
        ],
        force=True,
    )
    store.update_worker_snapshot(
        [{"strategy": "Renko", "snapshot_valid": False}],
        force=True,
    )

    state = store.snapshot()
    assert "open_position" in state["strategies"]["Renko"]
    assert state["strategies"]["Renko"]["snapshot_valid"] is False
    assert resumable_open_positions(state, session_date=TODAY) == {}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_load_missing_file_is_not_an_error(tmp_path: Path):
    assert load_session_state(tmp_path / "nope.json") is None


def test_load_corrupt_file_returns_none(tmp_path: Path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert load_session_state(path) is None


def test_load_non_object_json_returns_none(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_session_state(path) is None


# ---------------------------------------------------------------------------
# Resume eligibility -- the safety-critical half
# ---------------------------------------------------------------------------
def _crashed_state(**overrides) -> dict:
    state = {
        "schema_version": SCHEMA_VERSION,
        "session_date": "2026-08-10",
        "clean_shutdown": False,
        # A real merged document always carries this (load_session_state computes
        # it from the two files' timestamps). Fresh marks = no lag.
        "marks_age_seconds": 0.0,
        "strategies": {
            "Renko": {
                "live_trading": False,
                "execution_mode": "PAPER",
                "realized_pnl": -929.5,
                "completed_trades": 2,
                "open_position": serialize_position(
                    _FakePosition(), leg_marks={"option": 98.1}
                ),
            }
        },
        "trades": [],
    }
    state.update(overrides)
    return state


TODAY = date(2026, 8, 10)


def test_paper_position_from_today_is_resumable():
    resumable = resumable_open_positions(_crashed_state(), session_date=TODAY)
    assert set(resumable) == {"Renko"}
    assert resumable["Renko"]["entry_trade_price"] == 112.35
    assert resumable["Renko"]["last_mark_ltp"] == 98.1


def test_yesterdays_position_is_never_resumable():
    """Resuming a stale position would invent exposure that does not exist."""
    state = _crashed_state(session_date="2026-08-07")
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_tomorrow_dated_state_is_not_resumable():
    state = _crashed_state(session_date=(TODAY + timedelta(days=1)).isoformat())
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_clean_shutdown_is_never_resumable():
    """A clean end means the position was squared off and already published."""
    state = _crashed_state(clean_shutdown=True)
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_schema_mismatch_is_never_resumable():
    state = _crashed_state(schema_version=SCHEMA_VERSION + 1)
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_live_position_is_withheld_by_default():
    """In live trading the broker account is the authority, not this file."""
    state = _crashed_state()
    state["strategies"]["Renko"]["live_trading"] = True
    assert resumable_open_positions(state, session_date=TODAY) == {}
    # ...and the escape hatch exists but must be asked for explicitly.
    assert set(resumable_open_positions(state, session_date=TODAY, allow_live=True)) == {"Renko"}


def test_mixed_execution_mode_is_withheld():
    """MIXED means at least one real fill happened; treat it as live."""
    state = _crashed_state()
    state["strategies"]["Renko"]["execution_mode"] = "MIXED"
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_inactive_or_missing_position_is_skipped():
    state = _crashed_state()
    state["strategies"]["Renko"]["open_position"]["active"] = False
    state["strategies"]["Flat"] = {"live_trading": False, "realized_pnl": 0.0}
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_malformed_state_shapes_are_survivable():
    assert resumable_open_positions(None, session_date=TODAY) == {}
    assert resumable_open_positions({}, session_date=TODAY) == {}
    assert resumable_open_positions(
        _crashed_state(strategies="not-a-mapping"), session_date=TODAY
    ) == {}
    state = _crashed_state()
    state["strategies"]["Broken"] = "not-a-mapping"
    assert set(resumable_open_positions(state, session_date=TODAY)) == {"Renko"}


def test_recorded_realized_pnl_ignores_malformed_entries():
    state = _crashed_state()
    state["strategies"]["Renko"]["recorded_pnl"] = -929.5
    state["strategies"]["Bad"] = {"recorded_pnl": "oops"}
    state["strategies"]["Worse"] = "not-a-mapping"
    assert recorded_realized_pnl(state) == {"Renko": -929.5}
    assert recorded_realized_pnl(None) == {}
    assert recorded_realized_pnl({"strategies": "nope"}) == {}


def test_end_to_end_crash_then_recover(state_path: Path):
    """The 2026-08-10 scenario: trades bank, a position is open, the box dies."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})
    store.update_worker_snapshot(
        [
            {
                "strategy": "Renko",
                "realized_pnl": -929.5,
                "completed_trades": 1,
                "live_trading": False,
                "execution_mode": "PAPER",
                "open_position": serialize_position(
                    _FakePosition(), leg_marks={"option": 98.1}
                ),
            }
        ],
        force=True,
    )
    # No mark_clean_shutdown(): the process "died" here.
    del store

    recovered = load_session_state(state_path)
    assert recovered is not None
    assert recorded_realized_pnl(recovered) == {"Renko": -929.5}
    resumable = resumable_open_positions(recovered, session_date=TODAY)
    assert resumable["Renko"]["entry_trade_price"] == 112.35
    assert resumable["Renko"]["last_mark_ltp"] == 98.1
    assert os.path.exists(state_path)


# ---------------------------------------------------------------------------
# Durable / marks split -- the fsync-contention fix
# ---------------------------------------------------------------------------
# The supervisor rewrote the whole document every 30s WITH fsync, which on
# 2026-08-11 produced 210 slow-write warnings (median 0.83s, max 8.73s) against
# 13 on trading threads. Dropping fsync from that path is only safe if the
# snapshot stops touching the durable document at all: `os.replace` is atomic
# for the NAME but not the DATA, so a hard kill during an un-fsynced rewrite can
# publish a present-but-garbage file -- which would destroy the trades and the
# P&L rollup that ADR-0012 exists to protect.


def test_trade_events_fsync_but_snapshots_do_not(state_path: Path, monkeypatch):
    """The whole point of the split: one path pays fsync, the other never does."""
    calls: list[str] = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        "session_state.os.fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1]
    )

    store = _store(state_path)
    calls.clear()  # construction establishes the durable file; ignore that write.

    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    assert calls == [], "the 30s snapshot loop must never fsync"

    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})
    assert calls == ["fsync"], "a realized-P&L event must be durable before returning"


def test_a_snapshot_never_rewrites_the_durable_file(state_path: Path):
    """The safety property. A torn snapshot must not be able to eat the books."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})
    durable_before = state_path.read_bytes()
    mtime_before = state_path.stat().st_mtime_ns

    for _ in range(5):
        store.update_worker_snapshot(
            [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
            force=True,
        )

    assert state_path.read_bytes() == durable_before
    assert state_path.stat().st_mtime_ns == mtime_before
    # ...and the marks landed in their own file.
    assert _marks_path_for(state_path).exists()


def test_durable_file_exists_from_construction(state_path: Path):
    """A session that dies before its first trade must still leave a document.

    Without this the session date, the shutdown flags and any trade book carried
    forward from a same-day restart would all be missing from recovery.
    """
    _store(state_path)
    state = load_session_state(state_path)
    assert state is not None
    assert state["session_date"] == "2026-08-10"
    assert state["clean_shutdown"] is False


def test_positions_and_pnl_merge_back_into_one_document(state_path: Path):
    """Readers keep seeing the single-document shape they always have."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})
    store.update_worker_snapshot(
        [
            {
                "strategy": "Renko",
                "realized_pnl": -929.5,
                "completed_trades": 1,
                "live_trading": False,
                "execution_mode": "PAPER",
                "open_position": serialize_position(
                    _FakePosition(), leg_marks={"option": 98.1}
                ),
            }
        ],
        force=True,
    )

    merged = load_session_state(state_path)
    assert merged is not None
    assert recorded_realized_pnl(merged) == {"Renko": -929.5}
    entry = merged["strategies"]["Renko"]
    assert entry["completed_trades"] == 1          # from marks
    assert entry["recorded_pnl"] == -929.5         # from durable
    assert entry["open_position"]["entry_trade_price"] == 112.35
    assert resumable_open_positions(merged, session_date=TODAY)["Renko"]


def test_a_corrupt_marks_file_costs_positions_but_never_the_pnl(state_path: Path):
    """The asymmetry that justifies skipping fsync on the marks path."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -929.5})
    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
        force=True,
    )
    # Simulate the torn write that skipping fsync makes possible.
    _marks_path_for(state_path).write_text("{ not json", encoding="utf-8")

    recovered = load_session_state(state_path)
    assert recovered is not None
    assert recorded_realized_pnl(recovered) == {"Renko": -929.5}
    assert resumable_open_positions(recovered, session_date=TODAY) == {}


def test_a_missing_marks_file_is_not_an_error(state_path: Path):
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 5.0})
    _marks_path_for(state_path).unlink(missing_ok=True)

    recovered = load_session_state(state_path)
    assert recovered is not None
    assert recorded_realized_pnl(recovered) == {"Renko": 5.0}


def test_marks_from_another_session_are_ignored(state_path: Path):
    """A stale marks file must never graft yesterday's positions onto today."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 5.0})
    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
        force=True,
    )
    stale = json.loads(_marks_path_for(state_path).read_text(encoding="utf-8"))
    stale["session_date"] = "2020-01-01"
    _marks_path_for(state_path).write_text(json.dumps(stale), encoding="utf-8")

    recovered = load_session_state(state_path)
    assert recovered is not None
    assert recorded_realized_pnl(recovered) == {"Renko": 5.0}
    assert resumable_open_positions(recovered, session_date=TODAY) == {}


def test_a_corrupt_durable_file_is_still_a_total_loss(state_path: Path):
    """Unchanged contract: the durable half has no fallback, which is why it fsyncs."""
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 5.0})
    state_path.write_text("{ not json", encoding="utf-8")
    assert load_session_state(state_path) is None


def test_restart_archives_both_files_under_one_timestamp(state_path: Path):
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": -100.0})
    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
        force=True,
    )

    replacement = _store(state_path)
    archive = replacement.archive_path
    assert archive is not None and archive.exists()
    assert _marks_path_for(archive).exists(), "the marks file must be archived too"
    # The live marks path was moved aside rather than left pointing at old state.
    assert not _marks_path_for(state_path).exists()
    # Same-day P&L still carried into the replacement session.
    assert recorded_realized_pnl(replacement.snapshot()) == {"Renko": -100.0}


def test_slow_marks_write_warns_about_supervision_not_trading(state_path: Path, caplog):
    """The old message blamed the trading loop for a supervisor-thread write."""
    import session_state as module

    store = _store(state_path)
    with caplog.at_level("WARNING"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "SLOW_MARKS_WRITE_WARNING_SECONDS", 0.0)
        store.update_worker_snapshot([{"strategy": "Renko"}], force=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any("marks write took" in m and "not a trading decision" in m for m in messages)
    # And it must not blame the trading loop, which was the old message's error.
    assert not any("delaying the caller's trading loop" in m for m in messages)


# ---------------------------------------------------------------------------
# Snapshot-loop liveness and the stale-marks guard
# ---------------------------------------------------------------------------
# On 2026-08-12 the supervisor stopped writing marks at 12:30 while trading ran
# to 15:10. No exception, no partial file, no log line -- the freeze was only
# found by comparing file mtimes hours later. The marks on disk described 11
# positions where 23 were genuinely open, and nothing would have stopped a
# resume from restoring that stale set as a current book.


def test_health_reports_a_live_snapshot_loop(state_path: Path):
    store = _store(state_path, snapshot_interval_seconds=1.0)
    before = store.health()
    assert before["marks_writes"] == 0

    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
        force=True,
    )
    after = store.health()
    assert after["marks_writes"] == 1
    assert after["marks_age_seconds"] < 5
    assert after["marks_stalled"] is False
    assert after["open_positions"] == 1


def test_health_flags_a_stalled_snapshot_loop(state_path: Path, monkeypatch):
    """The exact 2026-08-12 signature: writes simply stop, silently."""
    store = _store(state_path, snapshot_interval_seconds=30.0)
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    assert store.health()["marks_stalled"] is False

    # Jump the monotonic clock past MARKS_STALL_INTERVALS * interval.
    import session_state as module

    base = time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 30.0 * 4 + 1)
    assert store.health()["marks_stalled"] is True


def test_marks_stall_warns_once_per_episode(state_path: Path, monkeypatch, caplog):
    import session_state as module

    store = _store(state_path, snapshot_interval_seconds=30.0)
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)

    base = time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 200.0)
    with caplog.at_level("ERROR"):
        assert store.warn_if_marks_stalled() is True
        assert store.warn_if_marks_stalled() is True  # still stalled...
    messages = [r.getMessage() for r in caplog.records]
    assert sum("have not been written" in m for m in messages) == 1, "must log once"
    # The message must be clear about what IS still safe.
    assert any("Realized P&L is unaffected" in m for m in messages)


def test_a_successful_write_clears_the_stall_latch(state_path: Path, monkeypatch, caplog):
    import session_state as module

    store = _store(state_path, snapshot_interval_seconds=30.0)
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    base = time.monotonic()
    monkeypatch.setattr(module.time, "monotonic", lambda: base + 200.0)
    store.warn_if_marks_stalled()

    monkeypatch.undo()
    store.update_worker_snapshot([{"strategy": "Renko"}], force=True)
    assert store.health()["marks_stalled"] is False
    with caplog.at_level("ERROR"):
        assert store.warn_if_marks_stalled() is False


def test_load_records_how_far_the_marks_lag_the_durable_file(state_path: Path):
    store = _store(state_path)
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 5.0})
    store.update_worker_snapshot(
        [{"strategy": "Renko", "open_position": serialize_position(_FakePosition())}],
        force=True,
    )
    merged = load_session_state(state_path)
    assert merged is not None
    assert merged["marks_age_seconds"] is not None
    assert abs(float(merged["marks_age_seconds"])) < 30


def test_stale_marks_are_never_resumable(state_path: Path):
    """The 2026-08-12 scenario end to end: a book that is hours out of date."""
    state = _crashed_state(marks_age_seconds=2.0 * 60 * 60 + 40 * 60)  # 2h40m
    assert resumable_open_positions(state, session_date=TODAY) == {}


def test_marks_lag_just_inside_the_limit_still_resumes():
    state = _crashed_state(marks_age_seconds=MAX_RESUMABLE_MARKS_AGE_SECONDS - 1)
    assert set(resumable_open_positions(state, session_date=TODAY)) == {"Renko"}


def test_unknown_marks_lag_fails_closed():
    """An unparseable or absent timestamp must refuse, not assume zero."""
    state = _crashed_state(marks_age_seconds=None)
    assert resumable_open_positions(state, session_date=TODAY) == {}
    del state["marks_age_seconds"]
    assert resumable_open_positions(state, session_date=TODAY) == {}


# ---------------------------------------------------------------------------
# Optional off-thread durable writer
# ---------------------------------------------------------------------------
# The synchronous write blocks the trading thread that published the event. On
# 2026-08-11, 86% of websocket feed disconnects landed within 20s of a write
# stall (`keepalive ping timeout` dominant) -- a blocked event loop drops the
# market feed. These tests pin the contract of moving it off-thread.


def _slow_store(state_path: Path, delay: float = 0.25):
    """A store whose durable write is artificially slow, like the real disk."""
    store = _store(state_path)
    real = store._write_document

    def slow(target, document, *, durable):
        if durable:
            time.sleep(delay)
        return real(target, document, durable=durable)

    store._write_document = slow  # type: ignore[method-assign]
    return store


def test_record_returns_immediately_once_the_writer_is_running(state_path: Path):
    """The whole point: a trading thread must not wait on fsync."""
    store = _slow_store(state_path, delay=0.4)
    store.start_durable_writer()
    try:
        began = time.monotonic()
        store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 1.0})
        elapsed = time.monotonic() - began
        assert elapsed < 0.2, f"caller blocked for {elapsed:.3f}s"
    finally:
        assert store.stop_durable_writer(timeout=10.0) is True

    on_disk = load_session_state(state_path)
    assert on_disk is not None
    assert on_disk["strategies"]["Renko"]["recorded_pnl"] == 1.0


def test_synchronous_path_is_unchanged_without_a_writer(state_path: Path):
    """Default behaviour must remain the original ADR-0012 contract."""
    store = _slow_store(state_path, delay=0.3)
    began = time.monotonic()
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 1.0})
    elapsed = time.monotonic() - began
    assert elapsed >= 0.3, "without a writer the caller must pay the fsync"
    assert load_session_state(state_path)["strategies"]["Renko"]["recorded_pnl"] == 1.0


def test_a_burst_is_coalesced_into_fewer_writes(state_path: Path):
    """Coalescing is what makes the queued-loss window small AND cuts fsyncs."""
    store = _slow_store(state_path, delay=0.15)
    writes = []
    real = store._write_document

    def counting(target, document, *, durable):
        if durable:
            writes.append(1)
        return real(target, document, durable=durable)

    store._write_document = counting  # type: ignore[method-assign]
    store.start_durable_writer()
    try:
        for i in range(12):
            store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 1.0, "n": i})
    finally:
        assert store.stop_durable_writer(timeout=10.0) is True

    # Every event must be present...
    state = load_session_state(state_path)
    assert len(state["trades"]) == 12
    assert state["strategies"]["Renko"]["recorded_trades"] == 12
    # ...but they must NOT have cost 12 separate durable writes.
    assert len(writes) < 12, f"no coalescing happened ({len(writes)} writes)"


def test_stop_drains_everything_recorded(state_path: Path):
    """An orderly shutdown must be exactly as durable as the sync path."""
    store = _slow_store(state_path, delay=0.05)
    store.start_durable_writer()
    for i in range(6):
        store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 2.0, "n": i})
    assert store.stop_durable_writer(timeout=10.0) is True

    state = load_session_state(state_path)
    assert len(state["trades"]) == 6
    assert state["strategies"]["Renko"]["recorded_pnl"] == 12.0


def test_stop_is_idempotent_and_safe_without_a_writer(state_path: Path):
    store = _store(state_path)
    assert store.stop_durable_writer() is True  # never started
    store.start_durable_writer()
    assert store.stop_durable_writer(timeout=10.0) is True
    assert store.stop_durable_writer() is True  # already stopped


def test_starting_twice_does_not_spawn_a_second_writer(state_path: Path):
    store = _store(state_path)
    store.start_durable_writer()
    first = store._writer
    store.start_durable_writer()
    try:
        assert store._writer is first
    finally:
        store.stop_durable_writer(timeout=10.0)


def test_a_failing_write_does_not_kill_the_writer_or_lose_the_event(state_path: Path):
    """A transient disk error must be retried, not swallowed silently."""
    store = _store(state_path)
    real = store._write_document
    calls = {"n": 0}

    def flaky(target, document, *, durable):
        if durable:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk hiccup")
        return real(target, document, durable=durable)

    store._write_document = flaky  # type: ignore[method-assign]
    store.start_durable_writer()
    store.record_trade_event({"action": "EXIT", "strategy": "Renko", "pnl": 7.0})
    try:
        # The writer re-marks the state dirty, so a later cycle persists it.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = load_session_state(state_path)
            if state and state.get("strategies", {}).get("Renko", {}).get("recorded_pnl") == 7.0:
                break
            time.sleep(0.05)
    finally:
        store.stop_durable_writer(timeout=10.0)

    state = load_session_state(state_path)
    assert state["strategies"]["Renko"]["recorded_pnl"] == 7.0
    assert calls["n"] >= 2, "the failed write must have been retried"
