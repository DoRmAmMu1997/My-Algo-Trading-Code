"""Thread-safe state for a coordinated flatten-then-stop shutdown.

Stopping a worker thread is not the same as safely stopping a live trading
process.  Once shutdown begins, new entries must stay blocked while the runner
closes every tracked leg and asks the broker whether the account is flat.  A
failed close keeps the process alive in reconciliation instead of allowing it
to report a clean shutdown.

This module deliberately performs no broker calls and never sleeps.  The
runner drives each transition and can inspect :meth:`retry_due` from its normal
loop.  Accepting a monotonic clock keeps retry scheduling deterministic in
tests and immune to wall-clock adjustments in production.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

_RETRY_DELAYS_SECONDS = (1.0, 2.0, 5.0)


class LifecycleState(Enum):
    """Ordered phases of a safe live-trading shutdown."""

    RUNNING = "RUNNING"
    ENTRY_BLOCKED = "ENTRY_BLOCKED"
    FLATTENING = "FLATTENING"
    RECONCILING = "RECONCILING"
    FLAT = "FLAT"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    """Immutable, coherent view of the lifecycle at one instant."""

    state: LifecycleState
    shutdown_reason: str
    flatten_attempts: int
    next_retry_at: float | None

    @property
    def entry_allowed(self) -> bool:
        """Whether a strategy may submit a new opening order."""

        return self.state is LifecycleState.RUNNING


class TradingLifecycle:
    """Serialize shutdown transitions and retry scheduling across threads."""

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        if not callable(monotonic):
            raise ValueError("monotonic must be callable")
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._state = LifecycleState.RUNNING
        self._shutdown_reason = ""
        self._flatten_attempts = 0
        self._next_retry_at: float | None = None

    def _snapshot_locked(self) -> LifecycleSnapshot:
        """Copy lock-owned fields while they form one coherent state."""

        return LifecycleSnapshot(
            state=self._state,
            shutdown_reason=self._shutdown_reason,
            flatten_attempts=self._flatten_attempts,
            next_retry_at=self._next_retry_at,
        )

    def snapshot(self) -> LifecycleSnapshot:
        """Return an immutable view safe for callers on any worker thread."""

        with self._lock:
            return self._snapshot_locked()

    def request_shutdown(self, reason: str) -> LifecycleSnapshot:
        """Block new entries and preserve the first reason for shutdown.

        Several triggers can race at market close, such as Ctrl+C, max-loss,
        and the scheduled square-off.  The first request performs the state
        transition; later requests are harmless and cannot overwrite its
        diagnostic reason.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("shutdown reason must be a non-empty string")
        normalized_reason = reason.strip()
        with self._lock:
            if self._state is LifecycleState.RUNNING:
                self._shutdown_reason = normalized_reason
                self._state = LifecycleState.ENTRY_BLOCKED
            return self._snapshot_locked()

    def start_flattening(self) -> LifecycleSnapshot:
        """Begin the first close attempt or a retry whose deadline has arrived."""

        with self._lock:
            if self._state is LifecycleState.ENTRY_BLOCKED:
                pass
            elif self._state is LifecycleState.RECONCILING:
                if self._next_retry_at is None:
                    raise RuntimeError(
                        "start_flattening requires a scheduled RECONCILING retry"
                    )
                if self._monotonic() < self._next_retry_at:
                    raise RuntimeError("flatten retry is not due")
            else:
                raise RuntimeError(
                    "start_flattening requires ENTRY_BLOCKED or a due "
                    "RECONCILING retry"
                )

            self._state = LifecycleState.FLATTENING
            self._flatten_attempts += 1
            self._next_retry_at = None
            return self._snapshot_locked()

    def start_reconciling(self) -> LifecycleSnapshot:
        """Finish one close pass and begin checking broker exposure."""

        with self._lock:
            if self._state is not LifecycleState.FLATTENING:
                raise RuntimeError("start_reconciling requires FLATTENING")
            self._state = LifecycleState.RECONCILING
            return self._snapshot_locked()

    def record_reconciliation(self, *, broker_flat: bool) -> LifecycleSnapshot:
        """Record broker evidence, reaching ``FLAT`` or scheduling a retry.

        Repeated non-flat polls keep the original deadline.  This prevents a
        frequent reconciliation loop from continually postponing liquidation.
        A later flat result can still finish shutdown while a retry is pending.
        """

        if type(broker_flat) is not bool:
            raise ValueError("broker_flat must be a bool")
        with self._lock:
            if self._state is not LifecycleState.RECONCILING:
                raise RuntimeError("record_reconciliation requires RECONCILING")
            if broker_flat:
                self._state = LifecycleState.FLAT
                self._next_retry_at = None
            elif self._next_retry_at is None:
                # Capped backoff: attempt 1 retries after 1s, attempt 2 after
                # 2s, attempt 3 AND EVERY LATER attempt after 5s.  The cap is
                # deliberate -- unlike a network backoff this loop is trying to
                # CLOSE live exposure, so it must never back off into minutes.
                delay_index = min(
                    self._flatten_attempts - 1,
                    len(_RETRY_DELAYS_SECONDS) - 1,
                )
                delay = _RETRY_DELAYS_SECONDS[delay_index]
                self._next_retry_at = self._monotonic() + delay
            return self._snapshot_locked()

    def retry_due(self) -> bool:
        """Return whether another flatten pass may start now."""

        with self._lock:
            return (
                self._state is LifecycleState.RECONCILING
                and self._next_retry_at is not None
                and self._monotonic() >= self._next_retry_at
            )

    def mark_stopped(self) -> LifecycleSnapshot:
        """Report shutdown complete only after the broker confirmed flat."""

        with self._lock:
            if self._state is LifecycleState.STOPPED:
                return self._snapshot_locked()
            if self._state is not LifecycleState.FLAT:
                raise RuntimeError("mark_stopped requires broker-confirmed FLAT")
            self._state = LifecycleState.STOPPED
            return self._snapshot_locked()
