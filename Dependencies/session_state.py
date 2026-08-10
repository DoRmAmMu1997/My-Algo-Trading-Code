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
        self.session_date = session_date or _now_ist().date()
        self.snapshot_interval_seconds = max(1.0, float(snapshot_interval_seconds))
        self.max_trade_records = max(1, int(max_trade_records))
        self.log = log or logger

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
            "strategies": {},
            "trades": [],
        }

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
                self._flush_locked()
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
                self._flush_locked()
            return True
        except Exception:  # noqa: BLE001 - reporting must never break trading
            self._log_write_failure("update worker snapshot")
            return False

    def mark_clean_shutdown(self) -> None:
        """Record that the session ended normally.

        A state file whose ``clean_shutdown`` is still false is the signal that
        the process died unexpectedly -- which is what makes the open positions
        in it worth looking at.  Called after the end-of-day results have been
        published, so a file marked clean is one whose figures already reached
        the Sheet.
        """
        try:
            with self._lock:
                self._state["clean_shutdown"] = True
                self._flush_locked()
        except Exception:  # noqa: BLE001 - reporting must never break shutdown
            self._log_write_failure("mark clean shutdown")

    def _flush_locked(self) -> None:
        """Publish the in-memory state atomically. Caller must hold the lock.

        Writes a sibling ``.tmp`` then :func:`os.replace`s it over the target,
        so a reader (or a crash) can only ever see a complete file.  ``flush``
        plus ``fsync`` before the replace matter here: without them the rename
        can reach the disk before the data does, which on a power cut leaves a
        published-but-empty file -- precisely the outcome this module exists to
        prevent.
        """
        self._state["updated_at"] = _now_ist().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)

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


def load_session_state(path: str | Path) -> dict[str, Any] | None:
    """Read a state file back, or return ``None`` when it is unusable.

    Deliberately forgiving: a missing file is the normal first-run case, and a
    truncated or hand-edited file must not stop the runner from starting.  Both
    are reported as ``None`` and the caller simply proceeds without recovery.
    """
    try:
        state_path = Path(path)
        if not state_path.exists():
            return None
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if not isinstance(state, dict):
            logger.warning("Session state at %s is not a JSON object; ignoring it.", state_path)
            return None
        return state
    except Exception:  # noqa: BLE001 - a bad state file must not stop startup
        logger.exception("Could not read session state from %s; continuing without recovery.", path)
        return None


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
