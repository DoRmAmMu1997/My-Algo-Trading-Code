"""Plumbing for the dashboard's per-strategy LIVE/PAPER toggle (ADR-0021).

The dashboard can ask for one strategy to switch between paper and live while
the runner is running. This module holds the three pieces that do NOT need the
runner's internals, so they can be tested on their own:

* :func:`parse_toggle_request` -- validates the browser's request body. The
  body is untrusted input, so every field is checked and anything extra is
  refused.
* :class:`LiveModeSlot` -- a small lock-guarded mailbox on each worker. The
  controller writes a REQUEST into it; only the worker's own thread ever acts on
  it, at a safe point in its loop. Nothing here ever writes
  ``worker.live_trading``.
* :class:`LiveModeController` -- one daemon thread that takes requests off a
  bounded queue and runs the runner's handler for each. A paper-to-live
  request may log in to the broker, which can take up to ten seconds, so it
  runs here and never on an HTTP thread.

**Why the worker applies the change itself.** The order path reads
``live_trading`` several times while it places one order (lease check, then
paper short-circuit). If another thread flipped the flag between two of those
reads, a live order could go out without the shared entry lease. So the flag
is only ever changed by the thread that also places the orders, and only when
that worker is flat.

Configuration note: like ``dashboard_server``, this module reads NO environment
variables. The master reads every knob through its ``_env_*`` helpers.
"""

from __future__ import annotations

import hmac
import json
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

#: The two modes a request can ask for. Upper-case on the wire, exactly.
TARGET_LIVE = "LIVE"
TARGET_PAPER = "PAPER"
TARGETS = frozenset({TARGET_LIVE, TARGET_PAPER})

#: A request's life on the dashboard: waiting for the worker to be flat, done,
#: or refused with a reason.
STATUS_PENDING = "PENDING"
STATUS_APPLIED = "APPLIED"
STATUS_REFUSED = "REFUSED"

#: A real request is about 120 bytes. Anything far larger is not ours.
MAX_REQUEST_BYTES = 1024

#: The exact keys a request carries -- no more, no fewer.
_REQUEST_KEYS = frozenset({"strategy", "target", "confirm", "token"})


@dataclass(frozen=True)
class LiveModeRequest:
    """One validated ask: switch ``strategy`` to ``target`` (LIVE or PAPER)."""

    strategy: str
    target: str


def parse_toggle_request(
    body: bytes,
    *,
    strategies: frozenset[str],
    token: str,
) -> LiveModeRequest | str:
    """Validate a toggle request body; return the request, or why it was refused.

    The browser sends ``{"strategy", "target", "confirm", "token"}``:

    * ``token`` must equal the server's per-process token, compared in constant
      time. A page on another website cannot read that token, so it cannot
      forge this request even though it can make the browser send one.
    * ``confirm`` must equal ``strategy`` exactly -- that is the operator typing
      the strategy's name in the confirmation box.
    * ``strategy`` must be one of the strategies this runner is running, and
      ``target`` must be exactly ``LIVE`` or ``PAPER``.

    The refusal message is fixed text that never echoes the request back.
    """

    if not token:
        return "Live toggle is not available."
    if len(body) > MAX_REQUEST_BYTES:
        return "Request body is too large."
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return "Request body must be a JSON object."
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        return "Request must carry exactly strategy, target, confirm and token."

    supplied_token = payload["token"]
    if not isinstance(supplied_token, str) or not hmac.compare_digest(
        supplied_token.encode("utf-8"), token.encode("utf-8")
    ):
        return "Request token is missing or stale; reload the page."

    strategy = payload["strategy"]
    if not isinstance(strategy, str) or strategy not in strategies:
        return "Unknown strategy."
    target = payload["target"]
    if not isinstance(target, str) or target not in TARGETS:
        return "Target must be LIVE or PAPER."
    if payload["confirm"] != strategy:
        return "Confirmation did not match the strategy name."
    return LiveModeRequest(strategy=strategy, target=target)


class LiveModeSlot:
    """One worker's toggle mailbox, plus what the dashboard shows about it.

    The controller calls :meth:`request` or :meth:`refuse`; the worker's own
    thread calls :meth:`pending` and then :meth:`resolve`. The dashboard's
    builder thread calls :meth:`view`. Every method takes the same small lock,
    holds it only for a few attribute writes, and never calls out.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # True = live requested, False = paper requested, None = nothing waiting.
        self._requested: bool | None = None
        # The last target asked for, kept after resolution so the dashboard can
        # say "LIVE -> APPLIED" rather than just "APPLIED".
        self._last_target: bool | None = None
        self._status = ""
        self._reason = ""
        self._at = ""

    def request(self, live: bool) -> None:
        """Record a request the worker should apply once it is safe to."""

        with self._lock:
            self._requested = bool(live)
            self._last_target = bool(live)
            self._status = STATUS_PENDING
            self._reason = ""
            self._at = _clock()

    def refuse(self, live: bool, reason: str) -> None:
        """Record a refused request without disturbing one already waiting."""

        with self._lock:
            if self._requested is None:
                self._last_target = bool(live)
                self._status = STATUS_REFUSED
                self._reason = str(reason)
                self._at = _clock()

    def pending(self) -> bool | None:
        """The waiting request, or None. Read by the worker's own thread."""

        with self._lock:
            return self._requested

    def resolve(self, seen: bool | None, status: str, reason: str = "") -> bool:
        """Close the request the worker READ; return False if a newer one replaced it.

        ``seen`` is what :meth:`pending` returned. If the operator asked for
        something else in the meantime, that newer request stays pending and
        this call changes nothing.
        """

        with self._lock:
            if seen is None or self._requested is not seen:
                return False
            self._requested = None
            self._status = status
            self._reason = str(reason)
            self._at = _clock()
            return True

    def view(self) -> dict[str, object]:
        """A plain copy for the dashboard document."""

        with self._lock:
            requested_mode = None
            if self._last_target is not None:
                requested_mode = TARGET_LIVE if self._last_target else TARGET_PAPER
            return {
                "requested_mode": requested_mode,
                "toggle_status": self._status,
                "toggle_reason": self._reason,
                "toggle_at": self._at,
            }


class LiveModeController(threading.Thread):
    """Runs toggle requests one at a time, off the HTTP threads.

    One thread means two requests can never race each other through the
    broker login and account audit. The queue is bounded and :meth:`submit`
    never blocks, so a burst of clicks cannot stall the HTTP thread that
    received them.
    """

    def __init__(
        self,
        handler: Callable[[LiveModeRequest], None],
        *,
        maxsize: int = 16,
    ) -> None:
        super().__init__(name="LiveModeController", daemon=True)
        self._handler = handler
        self.requests: queue.Queue[LiveModeRequest] = queue.Queue(maxsize=max(1, int(maxsize)))
        self._stop_event = threading.Event()

    def submit(self, request: LiveModeRequest) -> bool:
        """Queue a request; False when stopped or full. Never blocks."""

        if self._stop_event.is_set():
            return False
        try:
            self.requests.put_nowait(request)
        except queue.Full:
            return False
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self.requests.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._handler(request)
            except Exception:  # noqa: BLE001 - one bad request must not end the thread
                logger.exception(
                    "Live toggle request for %s could not be processed.", request.strategy
                )


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")
