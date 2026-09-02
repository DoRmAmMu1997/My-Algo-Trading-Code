"""One rolling-window rate limiter, shared by every broker adapter that needs it.

Hoisted from the Flattrade and Dhan adapters on 2026-09-02, where the class was
duplicated. The two copies were confirmed EQUIVALENT before the move: with
docstrings and comments stripped and the broker name normalised, their parsed
bodies were identical, so this is a de-duplication and not a behaviour change.
The wording kept here is Dhan's, because it was already broker-neutral.

Only the budgets differ between brokers, and they are constructor arguments:
Flattrade allows 40/second and 200/minute for general API calls (10/40 for
orders); Dhan allows 20/second and 800/minute (10/250 for orders). Kotak and
Shoonya publish no such limits and use no limiter at all.

The class is deliberately self-contained -- it reads no module-level constant,
so a caller's deadline policy stays the caller's business.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from collections.abc import Callable

# Broker adapters run both as repo modules (``Dependencies.broker_rate_limit``)
# and as standalone diagnostic scripts that import ``broker_rate_limit`` after
# putting the Dependencies directory on ``sys.path``. Point both names at this
# one module so a second copy of the class can never be created -- the same
# reasoning, and the same fix, as broker_contract.py.
sys.modules.setdefault("broker_rate_limit", sys.modules[__name__])
sys.modules.setdefault("Dependencies.broker_rate_limit", sys.modules[__name__])


class RollingWindowRateLimiter:
    """Enforce per-second and per-minute request budgets across worker threads.

    Short bursts wait for a slot.  A wait longer than ``max_wait_seconds`` raises
    before the request is sent; a stale live order is more dangerous than a
    clear indeterminate result that freezes new live entry.
    """

    def __init__(
        self,
        per_second: int,
        per_minute: int,
        max_wait_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        label: str = "API",
    ) -> None:
        self.per_second = int(per_second)
        self.per_minute = int(per_minute)
        self.max_wait_seconds = float(max_wait_seconds)
        self._clock = clock
        self._sleep = sleeper
        self.label = label
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, deadline: float | None = None) -> None:
        """Reserve one request slot or raise before a long/stale wait.

        The deque stores only timestamps from the last minute.  We separately
        count its last one-second slice because both limits apply at once.  A
        short wait is acceptable; a long wait would make a trading decision
        stale, so it raises *before* any broker request is sent.
        """
        started = self._clock()
        if deadline is None:
            acquired = self._lock.acquire()
        else:
            remaining = deadline - started
            acquired = remaining > 0 and self._lock.acquire(timeout=remaining)
        if not acquired:
            raise TimeoutError(
                f"{self.label} deadline expired waiting for the limiter lock"
            )
        try:
            while True:
                now = self._clock()
                # Discard calls that are older than the longest (60-second)
                # window.  Keeping the deque small also makes each check cheap.
                minute_cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] <= minute_cutoff:
                    self._timestamps.popleft()

                # The minute deque includes the one-second window, so select its
                # recent tail instead of maintaining a second source of truth.
                recent_second = [
                    stamp for stamp in self._timestamps if stamp > now - 1.0
                ]
                second_full = len(recent_second) >= self.per_second
                minute_full = len(self._timestamps) >= self.per_minute

                if not second_full and not minute_full:
                    # Reserving the timestamp before returning prevents two
                    # strategy threads from claiming the same final slot.
                    self._timestamps.append(now)
                    return

                waits = []
                if second_full:
                    waits.append(1.0 - (now - recent_second[0]))
                if minute_full:
                    waits.append(60.0 - (now - self._timestamps[0]))
                wait_seconds = max(0.0, max(waits))
                elapsed = now - started
                exceeds_call_deadline = (
                    deadline is not None and now + wait_seconds > deadline
                )
                if elapsed + wait_seconds > self.max_wait_seconds or exceeds_call_deadline:
                    raise RuntimeError(
                        f"{self.label} rate limit exhausted; "
                        f"safe slot needs {wait_seconds:.2f}s"
                    )
                self._sleep(wait_seconds)
        finally:
            self._lock.release()

    def reset(self) -> None:
        """Forget local request history after a session is explicitly closed."""
        with self._lock:
            self._timestamps.clear()
