"""Crash-durable per-session state for the multi-strategy runner.

Why this module exists
----------------------
The runner only writes its per-strategy results to the Google Sheet at the END
of a clean session.  That is fine when the process exits normally, but on
2026-08-10 the machine hung mid-session: the process died with thirteen open
positions and never wrote a summary, so a whole morning of realized P&L had to
be reconstructed by hand from the log file afterwards.

The existing "trades count" guard in the Sheet writer cannot help there.  It
stops a ``Trades=0`` summary from OVERWRITING real figures, but a crash writes
no summary at all -- there is nothing for the guard to reject.

So this module keeps a small JSON file up to date DURING the session:

* every trade event is appended the moment it happens (a few dozen writes a
  day, not a hot path), so realized P&L survives a hard kill; and
* every open position is snapshotted on a slow cadence together with its last
  known mark, so a position that was open when the machine died can be
  inspected -- and, in paper mode, resumed -- instead of vanishing.

Design rules this module follows
--------------------------------
1. **It must never break trading.**  Every public method swallows its own
   exceptions and logs them.  A full disk or a locked file is a reporting
   problem, never a reason for a strategy thread to die.
2. **Writes are atomic.**  State is written to a sibling ``.tmp`` file and then
   published with :func:`os.replace`, mirroring the instrument-master refresh
   in the master file.  A reader therefore sees either the whole previous file
   or the whole new one -- never a half-written one, which is exactly the
   failure mode a power cut would otherwise produce.
3. **State is date-scoped.**  The session date is recorded in the file and
   re-checked on load, so yesterday's open position can never be resumed into
   today's session.
4. **Live positions are never resumed from this file.**  In live trading the
   BROKER is the source of truth and the runner already has a reconciliation
   path for it (``recover_after_reconciliation`` / the startup exposure audit).
   A JSON file that disagrees with the account is worse than no file at all, so
   :func:`resumable_open_positions` refuses any record that was not pure paper.
   Restoring the realized-P&L BOOKKEEPING is safe in both modes and is not
   subject to that restriction.

What a position record keeps
----------------------------
Everything the position dataclass holds that can be represented in JSON --
which crucially includes ``entry_trade_price`` (the actual fill price the P&L
is computed against) and the strategy-side ``entry_underlying`` /
``stop_underlying`` / ``target_underlying`` levels -- plus a ``last_mark_ltp``
taken from the shared LTP cache at snapshot time.  Those two together are what
make a resumed position able to continue managing its own stop and target.

The one field deliberately NOT persisted is ``live_leg``.  It describes broker
exposure, it goes stale the instant the process dies, and resume is paper-only
anyway; the record keeps a plain ``had_live_leg`` boolean for forensics instead.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Bumped whenever the on-disk shape changes incompatibly. A file written by a
# different version is read for forensics but never resumed from.
SCHEMA_VERSION = 1

# A busy session produces well under a thousand events. The cap only exists so
# a runaway loop cannot grow the file without bound; the OLDEST records are
# dropped first because the newest ones are the ones a recovery needs.
DEFAULT_MAX_TRADE_RECORDS = 5000

# A durable write is deliberately synchronous: returning before ``fsync``
# would re-introduce the exact hard-kill window this module closes.  Local-disk
# writes normally finish in milliseconds, so crossing this threshold is an
# operational warning that the storage device may be delaying a trading worker.
SLOW_WRITE_WARNING_SECONDS = 0.25

# The marks file is written from the supervisor thread and never fsyncs, so a
# slow one delays shutdown supervision rather than a trading decision.  It gets
# a looser threshold on purpose: warning at the durable threshold produced 210
# lines in one session (2026-08-11) and buried the 13 that actually mattered.
SLOW_MARKS_WRITE_WARNING_SECONDS = 2.0

# Per-strategy keys that live in the MARKS file rather than the durable one.
# All of them are refreshed wholesale by every snapshot, so losing the newest
# 30 seconds of them in a crash is the documented trade-off (ADR-0012); none is
# needed to answer "what had this strategy banked when the process died".
_MARKS_ONLY_KEYS = (
    "open_position",
    "snapshot_valid",
    "snapshot_error_at",
    "completed_trades",
    "realized_pnl",
    "execution_mode",
    "live_trading",
)

# Position attributes that are deliberately never written out. ``live_leg`` is
# broker exposure state (see the module docstring); the private/dunder filter
# below removes bookkeeping attributes that are not part of the position shape.
_POSITION_FIELD_DENYLIST = frozenset({"live_leg"})

# Event actions that carry realized P&L. Recorded like any other event, but
# called out here because the reconciliation path sums exactly these.
PNL_BEARING_ACTIONS = frozenset({"EXIT"})


def _now_ist() -> datetime:
    """Timezone-aware 'now' in the market's own timezone."""
    return datetime.now(IST)


def _jsonable(value: Any) -> Any:
    """Coerce one value into something :func:`json.dump` accepts.

    Returns the sentinel :data:`_UNSERIALIZABLE` for anything that cannot be
    represented, so the caller can drop the field entirely rather than write a
    misleading ``null`` that a reader might mistake for a real zero/absent
    value.

    ``date``/``datetime`` become ISO strings because the position dataclasses
    store ``option_expiry`` as a :class:`datetime.date`.
    """
    if value is None or isinstance(value, bool | int | float | str):
        # Reject NaN/inf: json.dump would emit bare NaN/Infinity, which is not
        # valid JSON and would make the file unreadable by a strict parser.
        if isinstance(value, float) and not _is_finite(value):
            return _UNSERIALIZABLE
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        coerced = {}
        for key, item in value.items():
            item_value = _jsonable(item)
            if item_value is not _UNSERIALIZABLE:
                coerced[str(key)] = item_value
        return coerced
    if isinstance(value, list | tuple | set | frozenset):
        items = [_jsonable(item) for item in value]
        return [item for item in items if item is not _UNSERIALIZABLE]
    return _UNSERIALIZABLE


class _Unserializable:
    """Sentinel type: distinguishes 'cannot serialize' from a real ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<UNSERIALIZABLE>"


_UNSERIALIZABLE = _Unserializable()


def _is_finite(value: float) -> bool:
    """True when ``value`` is a real number (not NaN and not +/-inf)."""
    return value == value and value not in (float("inf"), float("-inf"))


def _position_attributes(position: Any) -> dict[str, Any]:
    """Read a position object's fields regardless of which dataclass it is.

    The runner has several position shapes (``PaperPosition``,
    ``HedgedPaperPosition``, and the agent workers' own variants), and more may
    be added.  Rather than teach this module about each one, read whatever
    public attributes the object actually has.  A new worker family is then
    covered automatically instead of silently persisting nothing.
    """
    if is_dataclass(position) and not isinstance(position, type):
        raw = vars(position)
    else:
        raw = getattr(position, "__dict__", {}) or {}
    return {
        name: value
        for name, value in raw.items()
        if not name.startswith("_") and name not in _POSITION_FIELD_DENYLIST
    }


def serialize_position(
    position: Any,
    *,
    leg_marks: Mapping[str, float] | None = None,
    unrealized_pnl: float | None = None,
) -> dict[str, Any] | None:
    """Turn a live position object into a JSON-safe record.

    Returns ``None`` when the worker is flat, so a flat strategy simply has no
    ``open_position`` key rather than an ``active: false`` stub that a reader
    has to remember to check.

    ``leg_marks`` maps a leg's field prefix to its most recent cached price --
    ``{"option": 98.1}`` for the single-leg family, ``{"main": ..., "hedge":
    ...}`` for a hedged pair -- and ``unrealized_pnl`` is the position's
    mark-to-market at those prices.  Both are supplied by the RUNNER because
    reading a price is its job; this module must never touch the broker or the
    shared market-data store.

    ``last_mark_ltp`` is additionally set from the ``option`` leg when present.
    That is the single-leg ATM family, which is the shape resume supports, so
    keeping it as a top-level field saves every reader from digging into the
    per-leg map for the common case.
    """
    if position is None or not getattr(position, "active", False):
        return None

    record: dict[str, Any] = {"position_type": type(position).__name__}
    for name, value in _position_attributes(position).items():
        coerced = _jsonable(value)
        if coerced is not _UNSERIALIZABLE:
            record[name] = coerced

    # Broker-leg state is intentionally reduced to a boolean; see the module
    # docstring for why the full LiveLegState is not persisted.
    record["had_live_leg"] = getattr(position, "live_leg", None) is not None

    marks = {
        str(name): round(float(price), 2)
        for name, price in (leg_marks or {}).items()
        if isinstance(price, int | float) and _is_finite(float(price))
    }
    if marks:
        record["leg_marks"] = marks
        record["last_mark_at"] = _now_ist().isoformat()
        if "option" in marks:
            record["last_mark_ltp"] = marks["option"]
    if unrealized_pnl is not None and _is_finite(float(unrealized_pnl)):
        record["unrealized_pnl"] = round(float(unrealized_pnl), 2)
    return record


class SessionStateStore:
    """Owns one session's JSON state file and every write to it.

    One instance is created by ``main()`` and shared by all worker threads, so
    every method that touches ``_state`` holds ``_lock``.  The lock is held
    across the file write too: the writes are small and infrequent, and letting
    two threads publish different snapshots concurrently would be a far worse
    trade than a few milliseconds of contention.

    Constructing a replacement store never destroys the prior run.  If the
    configured path already exists, its exact bytes are first moved to a
    timestamped ``*.recovery.json`` sibling.  Compatible same-day trade and P&L
    bookkeeping is then copied into the new in-memory session, while prior open
    positions stay only in ``previous_state`` and the archive until the runner
    explicitly validates and resumes them.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        session_date: date | None = None,
        snapshot_interval_seconds: float = 30.0,
        max_trade_records: int = DEFAULT_MAX_TRADE_RECORDS,
        log: logging.Logger | None = None,
    ) -> None:
        self.path = Path(path)
        # Open positions and their marks live in a SEPARATE file, written on the
        # snapshot cadence without fsync. They are deliberately not in the
        # durable document: the supervisor rewrites them every 30 seconds, and
        # `os.replace` is atomic for the NAME but not for the DATA -- a hard kill
        # during an un-fsynced rewrite can publish a present-but-garbage file. If
        # that file also held the trades and the P&L rollup, one torn snapshot
        # would destroy the very record ADR-0012 exists to protect.
        self.marks_path = _marks_path_for(self.path)
        self.session_date = session_date or _now_ist().date()
        self.snapshot_interval_seconds = max(1.0, float(snapshot_interval_seconds))
        self.max_trade_records = max(1, int(max_trade_records))
        self.log = log or logger

        # Read and move the old file before creating any new state.  If the
        # move fails, construction raises and main() disables this optional
        # subsystem; importantly, it does NOT overwrite the recovery evidence.
        self.previous_state, self.archive_path = self._archive_previous_file()

        self._lock = threading.Lock()
        # Monotonic, not wall-clock: the throttle must be immune to a clock
        # correction (NTP step, DST) landing mid-session.
        self._last_snapshot_at = 0.0
        # Set once after the first write failure so a broken path (bad
        # permissions, missing drive) logs one error instead of one per trade.
        self._write_failure_logged = False

        self._state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_date": self.session_date.isoformat(),
            "started_at": _now_ist().isoformat(),
            "updated_at": _now_ist().isoformat(),
            # Flipped true only by mark_clean_shutdown(). A file still showing
            # false is the signature of the crash this module exists for.
            "clean_shutdown": False,
            # A clean process exit and a successful external Sheet write are
            # separate facts.  Keeping both avoids calling an orderly Ctrl+C
            # shutdown "published" when no EOD export actually happened.
            "results_published": False,
            "strategies": {},
            "trades": [],
        }
        self._carry_forward_same_day_bookkeeping()
        # Establish the durable file immediately. Everything else only writes it
        # on a trade event or at shutdown, so without this a session that dies
        # before its first trade would leave NO durable document -- losing the
        # session date, the shutdown flags and any trade book carried forward
        # from a same-day restart, all of which a recovery needs.
        try:
            with self._lock:
                self._flush_durable_locked()
        except Exception:  # noqa: BLE001 - persistence must never stop a session
            self._log_write_failure("establish the durable state file")

    def _archive_previous_file(self) -> tuple[dict[str, Any] | None, Path | None]:
        """Move an existing state file aside and return its parsed document.

        ``os.replace`` is used rather than copy-then-delete.  A crash between
        this move and the first write of the replacement session therefore
        leaves the complete old file in the recovery archive, never half in
        each location.
        """

        if not self.path.exists():
            return None, None
        if not self.path.is_file():
            # Leave an invalid target (for example, a directory at the file
            # path) untouched.  The first write will fail through the normal
            # best-effort path and set the degraded-persistence warning.
            return None, None

        previous = load_session_state(self.path)
        timestamp = _now_ist().strftime("%Y%m%dT%H%M%S%f")
        suffix = self.path.suffix
        stem = self.path.name[: -len(suffix)] if suffix else self.path.name
        archive = self.path.with_name(f"{stem}.{timestamp}.recovery{suffix}")
        os.replace(self.path, archive)
        # The marks file is archived alongside it under the SAME timestamp, so a
        # recovery pair stays identifiable. Its absence is normal (a session that
        # died before its first snapshot has none), so this never fails the move.
        if self.marks_path.is_file():
            marks_archive = _marks_path_for(archive)
            try:
                os.replace(self.marks_path, marks_archive)
            except OSError:
                self.log.warning(
                    "Could not archive the previous marks file at %s; it will be "
                    "overwritten by this session.", self.marks_path,
                )
        self.log.warning(
            "Archived the previous session state before starting a new file: %s",
            archive,
        )
        return previous, archive

    def _carry_forward_same_day_bookkeeping(self) -> None:
        """Seed today's new session with durable trades, but no old exposure."""

        previous = self.previous_state
        if not isinstance(previous, Mapping):
            return
        if int(previous.get("schema_version", -1)) != SCHEMA_VERSION:
            return
        if str(previous.get("session_date", "")) != self.session_date.isoformat():
            return

        old_trades = previous.get("trades")
        if isinstance(old_trades, list):
            copied_trades = _jsonable(old_trades)
            if isinstance(copied_trades, list):
                self._state["trades"] = copied_trades[-self.max_trade_records :]

        old_strategies = previous.get("strategies")
        if not isinstance(old_strategies, Mapping):
            return
        for strategy, raw_entry in old_strategies.items():
            if not isinstance(raw_entry, Mapping):
                continue
            copied_entry = _jsonable(dict(raw_entry))
            if not isinstance(copied_entry, dict):
                continue
            # Exposure never carries implicitly.  The runner validates the old
            # record separately and its first successful snapshot writes back
            # only positions that genuinely resumed.
            copied_entry.pop("open_position", None)
            copied_entry.pop("snapshot_valid", None)
            copied_entry.pop("snapshot_error_at", None)
            self._state["strategies"][str(strategy)] = copied_entry

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def record_trade_event(self, event: Mapping[str, Any]) -> None:
        """Append one trade event and publish the file immediately.

        Called from the runner's single ``publish_trade_event`` choke point, so
        every entry and exit of every worker family arrives here without each
        exit path needing its own hook.  The write is immediate and unthrottled
        on purpose: these events are what a crashed session loses, and there
        are only a few dozen of them per day.
        """
        try:
            record = _jsonable(dict(event))
            if record is _UNSERIALIZABLE or not isinstance(record, dict):
                return
            with self._lock:
                trades = self._state["trades"]
                trades.append(record)
                if len(trades) > self.max_trade_records:
                    # Drop oldest first -- a recovery cares about the newest.
                    del trades[: len(trades) - self.max_trade_records]
                self._apply_pnl_bearing_event_locked(record)
                self._flush_durable_locked()
        except Exception:  # noqa: BLE001 - reporting must never break trading
            self._log_write_failure("record trade event")

    def _apply_pnl_bearing_event_locked(self, record: Mapping[str, Any]) -> None:
        """Fold a realized-P&L event into that strategy's running totals.

        Keeping a per-strategy rollup alongside the raw event list means a
        recovery does not have to re-sum the events to answer "what did this
        strategy make today" -- and the two can be cross-checked against each
        other, which is exactly the sanity check a manual reconciliation wants.
        """
        if str(record.get("action", "")) not in PNL_BEARING_ACTIONS:
            return
        strategy = str(record.get("strategy", "")).strip()
        if not strategy:
            return
        pnl = record.get("pnl")
        if not isinstance(pnl, int | float) or not _is_finite(float(pnl)):
            return
        entry = self._state["strategies"].setdefault(strategy, {})
        entry["recorded_trades"] = int(entry.get("recorded_trades", 0)) + 1
        entry["recorded_pnl"] = round(float(entry.get("recorded_pnl", 0.0)) + float(pnl), 2)

    def update_worker_snapshot(
        self,
        snapshots: Sequence[Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> bool:
        """Refresh every strategy's counters and open position.

        ``snapshots`` is a sequence of plain dicts built by the RUNNER (one per
        worker), so this module never reaches into worker objects or the shared
        market-data store.  Each dict carries at least ``strategy``; the
        recognised optional keys are ``completed_trades``, ``realized_pnl``,
        ``execution_mode``, ``live_trading`` and ``open_position``.

        Throttled to ``snapshot_interval_seconds`` unless ``force`` is set,
        because it is driven from the supervisor loop that ticks once a second.
        Returns True when a write actually happened.
        """
        try:
            now = time.monotonic()
            with self._lock:
                due = force or (now - self._last_snapshot_at) >= self.snapshot_interval_seconds
                if not due:
                    return False
                self._last_snapshot_at = now
                for snapshot in snapshots:
                    strategy = str(snapshot.get("strategy", "")).strip()
                    if not strategy:
                        continue
                    entry = self._state["strategies"].setdefault(strategy, {})
                    snapshot_valid = bool(snapshot.get("snapshot_valid", True))
                    entry["snapshot_valid"] = snapshot_valid
                    if not snapshot_valid:
                        # Preserve the last record for human forensics, but mark
                        # it unsafe so resume cannot mistake stale state for a
                        # current worker observation.
                        entry["snapshot_error_at"] = _now_ist().isoformat()
                        continue
                    entry.pop("snapshot_error_at", None)
                    for key in (
                        "completed_trades",
                        "realized_pnl",
                        "execution_mode",
                        "live_trading",
                    ):
                        if key in snapshot:
                            coerced = _jsonable(snapshot[key])
                            if coerced is not _UNSERIALIZABLE:
                                entry[key] = coerced
                    position = snapshot.get("open_position")
                    if position:
                        coerced_position = _jsonable(position)
                        if coerced_position is not _UNSERIALIZABLE:
                            entry["open_position"] = coerced_position
                    else:
                        # A closed position must be REMOVED, not left behind --
                        # a stale record here would look resumable.
                        entry.pop("open_position", None)
                # Marks only: the snapshot loop must never rewrite the durable
                # document, or one torn write could destroy the day's books.
                self._flush_marks_locked()
            return True
        except Exception:  # noqa: BLE001 - reporting must never break trading
            self._log_write_failure("update worker snapshot")
            return False

    def mark_clean_shutdown(self, *, results_published: bool = False) -> None:
        """Record that the session ended normally.

        A state file whose ``clean_shutdown`` is still false is the signal that
        the process died unexpectedly -- which is what makes the open positions
        in it worth looking at.  ``results_published`` is recorded separately:
        orderly flattening can succeed during Ctrl+C even though the EOD Sheet
        export was intentionally skipped or failed.
        """
        try:
            with self._lock:
                self._state["clean_shutdown"] = True
                self._state["results_published"] = bool(results_published)
                # Both halves: the shutdown flags are crash-critical, and the
                # final marks write leaves the pair consistent for a reader.
                self._flush_durable_locked()
                self._flush_marks_locked()
        except Exception:  # noqa: BLE001 - reporting must never break shutdown
            self._log_write_failure("mark clean shutdown")

    def _write_document(self, target: Path, document: Mapping[str, Any], *, durable: bool) -> float:
        """Atomically publish one document. Returns the elapsed seconds.

        Writes a sibling ``.tmp`` then :func:`os.replace`s it over the target,
        so a reader can only ever see a complete file under normal operation.

        ``durable`` adds ``flush`` + ``fsync`` before the replace, and that is
        the whole difference between the two files. Without it the rename can
        reach the disk before the data does, so a hard kill can publish a
        present-but-garbage file. The durable document therefore always pays the
        fsync; the marks document never does, because re-deriving it costs at
        most one snapshot interval and it is written 200+ times a session.
        """
        started_at = time.monotonic()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=False)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        return time.monotonic() - started_at

    def _durable_document_locked(self) -> dict[str, Any]:
        """The crash-critical half: everything except the volatile marks."""
        document = {
            key: value for key, value in self._state.items() if key != "strategies"
        }
        strategies: dict[str, Any] = {}
        for strategy, entry in self._state.get("strategies", {}).items():
            if isinstance(entry, Mapping):
                strategies[str(strategy)] = {
                    key: value
                    for key, value in entry.items()
                    if key not in _MARKS_ONLY_KEYS
                }
        document["strategies"] = strategies
        return document

    def _marks_document_locked(self) -> dict[str, Any]:
        """The volatile half: open positions, their marks, and live counters."""
        strategies: dict[str, Any] = {}
        for strategy, entry in self._state.get("strategies", {}).items():
            if not isinstance(entry, Mapping):
                continue
            carried = {
                key: value for key, value in entry.items() if key in _MARKS_ONLY_KEYS
            }
            if carried:
                strategies[str(strategy)] = carried
        return {
            "schema_version": SCHEMA_VERSION,
            "session_date": self._state.get("session_date"),
            "updated_at": self._state.get("updated_at"),
            "strategies": strategies,
        }

    def _flush_durable_locked(self) -> None:
        """Publish the durable document with fsync. Caller must hold the lock.

        Called only from the trade-event and shutdown paths -- a few dozen times
        a session -- so the fsync cost is paid rarely and always for a record
        that a crash would otherwise make expensive to rebuild by hand.
        """
        self._state["updated_at"] = _now_ist().isoformat()
        elapsed = self._write_document(
            self.path, self._durable_document_locked(), durable=True
        )
        if elapsed >= SLOW_WRITE_WARNING_SECONDS:
            self.log.warning(
                "Session state durable write took %.3fs (path=%s); the local disk "
                "is delaying the caller's trading loop.",
                elapsed,
                self.path,
            )

    def _flush_marks_locked(self) -> None:
        """Publish the marks document without fsync. Caller must hold the lock.

        Runs on the supervisor thread, which never trades, and skips fsync
        because a torn marks file costs one snapshot interval of mark data and
        nothing else -- the trades and the P&L rollup are in the durable file
        this method does not touch.
        """
        self._state["updated_at"] = _now_ist().isoformat()
        elapsed = self._write_document(
            self.marks_path, self._marks_document_locked(), durable=False
        )
        if elapsed >= SLOW_MARKS_WRITE_WARNING_SECONDS:
            self.log.warning(
                "Session state marks write took %.3fs (path=%s); this delays "
                "supervision, not a trading decision.",
                elapsed,
                self.marks_path,
            )

    def _log_write_failure(self, action: str) -> None:
        """Log the first persistence failure loudly, then stay quiet."""
        if self._write_failure_logged:
            return
        self._write_failure_logged = True
        self.log.exception(
            "Session state persistence failed while trying to %s (path=%s). "
            "Trading continues; crash recovery for this session is degraded.",
            action,
            self.path,
        )

    # ------------------------------------------------------------------
    # Reading (used by tests and by the operator-facing recovery report)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a deep-ish copy of the current state, for tests/diagnostics."""
        with self._lock:
            return json.loads(json.dumps(self._state))


def _marks_path_for(path: str | Path) -> Path:
    """Sibling marks path for a durable state path (``x.json`` -> ``x.marks.json``)."""
    state_path = Path(path)
    suffix = state_path.suffix
    stem = state_path.name[: -len(suffix)] if suffix else state_path.name
    return state_path.with_name(f"{stem}.marks{suffix}")


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Read one JSON object, or ``None`` if it is missing or unusable."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        logger.warning("Session state at %s is not a JSON object; ignoring it.", path)
        return None
    return document


def load_session_state(path: str | Path) -> dict[str, Any] | None:
    """Read a session back as ONE merged document, or ``None`` if unusable.

    The state is stored as two files -- a durable one holding the trades and the
    P&L rollup, and a best-effort ``*.marks.json`` holding open positions and
    their last marks (see :class:`SessionStateStore`).  This merges them so that
    every reader (`resumable_open_positions`, `recorded_realized_pnl`, the
    runner's resume path) sees the same single-document shape it always has.

    Deliberately forgiving: a missing file is the normal first-run case, and a
    truncated or hand-edited one must not stop the runner from starting.

    The asymmetry is the point.  A corrupt DURABLE file means no recovery, and
    is reported as ``None``.  A corrupt or missing MARKS file costs only the
    open positions: the P&L is still returned, because that is the record whose
    loss motivated ADR-0012 and it was never written by the snapshot loop.
    """
    try:
        state_path = Path(path)
        state = _read_json_object(state_path)
        if state is None:
            return None
    except Exception:  # noqa: BLE001 - a bad state file must not stop startup
        logger.exception("Could not read session state from %s; continuing without recovery.", path)
        return None

    try:
        marks = _read_json_object(_marks_path_for(state_path))
    except Exception:  # noqa: BLE001 - marks are best-effort by design
        logger.exception(
            "Could not read the session marks beside %s; continuing with P&L only "
            "(no open positions will be offered for resume).", state_path,
        )
        return state

    if marks is None:
        return state
    if str(marks.get("session_date", "")) != str(state.get("session_date", "")):
        logger.warning(
            "Session marks at %s are from a different session than the durable "
            "state; ignoring them.", _marks_path_for(state_path),
        )
        return state

    mark_strategies = marks.get("strategies")
    if not isinstance(mark_strategies, Mapping):
        return state
    strategies = state.setdefault("strategies", {})
    if not isinstance(strategies, dict):
        return state
    for strategy, entry in mark_strategies.items():
        if not isinstance(entry, Mapping):
            continue
        target = strategies.setdefault(str(strategy), {})
        if isinstance(target, dict):
            target.update(entry)
    return state


def resumable_open_positions(
    state: Mapping[str, Any] | None,
    *,
    session_date: date,
    allow_live: bool = False,
) -> dict[str, dict[str, Any]]:
    """Select the open positions from ``state`` that may safely be resumed.

    A record has to clear every one of these before it is offered back:

    * the file's ``schema_version`` matches this module's;
    * the file's ``session_date`` is TODAY -- yesterday's open position is not
      a position, it is a stale record, and resuming it would invent exposure;
    * the session did NOT end cleanly, because a clean end means the position
      was squared off and its P&L already published; and
    * the strategy was trading PAPER.  Live records are withheld unless the
      caller explicitly passes ``allow_live`` (which the runner does not),
      because in live trading the broker -- not this file -- is the authority
      on what is open.

    Returns a ``{strategy_name: position_record}`` mapping, empty whenever
    nothing qualifies.
    """
    if not state:
        return {}
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        logger.warning(
            "Session state schema %s does not match %s; not resuming any position.",
            state.get("schema_version"),
            SCHEMA_VERSION,
        )
        return {}
    if str(state.get("session_date", "")) != session_date.isoformat():
        return {}
    if bool(state.get("clean_shutdown", False)):
        return {}

    resumable: dict[str, dict[str, Any]] = {}
    strategies = state.get("strategies")
    if not isinstance(strategies, Mapping):
        return {}
    for strategy, entry in strategies.items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("snapshot_valid", True) is False:
            logger.warning(
                "Not resuming %s: its last worker snapshot failed, so the persisted "
                "open position may be stale.",
                strategy,
            )
            continue
        position = entry.get("open_position")
        if not isinstance(position, Mapping) or not position.get("active"):
            continue
        was_live = bool(entry.get("live_trading", False)) or str(
            entry.get("execution_mode", "PAPER")
        ).upper() not in ("PAPER", "")
        if was_live and not allow_live:
            logger.warning(
                "Not resuming %s from session state: it traded LIVE, so the broker "
                "account -- not this file -- is the authority on what is open.",
                strategy,
            )
            continue
        resumable[str(strategy)] = dict(position)
    return resumable


def recorded_realized_pnl(state: Mapping[str, Any] | None) -> dict[str, float]:
    """Per-strategy realized P&L as recorded during the session.

    This is the figure that survives a crash.  It is summed from the recorded
    EXIT events rather than from the workers' in-memory counters, so it is
    exactly what the runner had actually banked at the moment it died.
    """
    if not state:
        return {}
    strategies = state.get("strategies")
    if not isinstance(strategies, Mapping):
        return {}
    totals: dict[str, float] = {}
    for strategy, entry in strategies.items():
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("recorded_pnl")
        if isinstance(value, int | float) and _is_finite(float(value)):
            totals[str(strategy)] = round(float(value), 2)
    return totals
