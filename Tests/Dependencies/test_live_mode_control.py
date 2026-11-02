"""Tests for the dashboard's live/paper toggle plumbing (`Dependencies/live_mode_control.py`).

Three things are asserted here, because each one is a safety property rather
than a convenience:

* **Parsing refuses anything that is not exactly one well-formed request.** The
  body comes from a browser, so the token, the typed confirmation, the strategy
  name and the target are each validated, and anything extra is rejected.
* **A slot never loses a newer request to an older one.** The worker resolves
  the request it READ; if the operator changed their mind in between, the newer
  request must survive.
* **The controller thread survives a handler that raises** and drops requests
  rather than blocking the HTTP thread when its queue is full.
"""

from __future__ import annotations

import json
import queue
import threading

import pytest

# Bare import: this folder's conftest.py puts the SOURCE `Dependencies/` on
# sys.path, which is the same resolution the runtime performs.
from live_mode_control import (
    MAX_REQUEST_BYTES,
    STATUS_APPLIED,
    STATUS_PENDING,
    STATUS_REFUSED,
    TARGET_LIVE,
    TARGET_PAPER,
    LiveModeController,
    LiveModeRequest,
    LiveModeSlot,
    parse_toggle_request,
)

STRATEGIES = frozenset({"Renko", "CPR"})
TOKEN = "a" * 43
# A named blank, so the "no token" cases read as intent rather than a typo.
NO_TOKEN = ""


def _body(**overrides) -> bytes:
    payload = {"strategy": "Renko", "target": "LIVE", "confirm": "Renko", "token": TOKEN}
    payload.update(overrides)
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# parse_toggle_request
# ---------------------------------------------------------------------------
def test_a_well_formed_request_parses():
    parsed = parse_toggle_request(_body(), strategies=STRATEGIES, token=TOKEN)
    assert parsed == LiveModeRequest(strategy="Renko", target=TARGET_LIVE)


def test_paper_is_an_accepted_target():
    parsed = parse_toggle_request(_body(target="PAPER"), strategies=STRATEGIES, token=TOKEN)
    assert parsed == LiveModeRequest(strategy="Renko", target=TARGET_PAPER)


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[1, 2]",
        json.dumps({"strategy": "Renko"}).encode(),
        _body(token="b" * 43),
        _body(token=NO_TOKEN),
        _body(token=123),
        _body(strategy="Unknown", confirm="Unknown"),
        _body(target="live"),
        _body(target="MIXED"),
        _body(confirm="renko"),
        _body(confirm=""),
        _body(extra="field"),
        b"\xff\xfe",
    ],
)
def test_anything_else_is_refused_with_a_message(body):
    parsed = parse_toggle_request(body, strategies=STRATEGIES, token=TOKEN)
    assert isinstance(parsed, str) and parsed


def test_an_oversized_body_is_refused_before_it_is_parsed():
    body = _body(strategy="Renko" + " " * MAX_REQUEST_BYTES)
    parsed = parse_toggle_request(body, strategies=STRATEGIES, token=TOKEN)
    assert isinstance(parsed, str)


def test_an_empty_server_token_refuses_everything():
    """A server that never minted a token must not accept a blank one."""
    parsed = parse_toggle_request(_body(token=NO_TOKEN), strategies=STRATEGIES, token=NO_TOKEN)
    assert isinstance(parsed, str)


# ---------------------------------------------------------------------------
# LiveModeSlot
# ---------------------------------------------------------------------------
def test_a_fresh_slot_has_nothing_pending():
    slot = LiveModeSlot()
    assert slot.pending() is None
    assert slot.view() == {
        "requested_mode": None,
        "toggle_status": "",
        "toggle_reason": "",
        "toggle_at": "",
    }


def test_a_request_is_pending_until_resolved():
    slot = LiveModeSlot()
    slot.request(True)
    assert slot.pending() is True
    assert slot.view()["requested_mode"] == TARGET_LIVE
    assert slot.view()["toggle_status"] == STATUS_PENDING
    assert slot.view()["toggle_at"]

    assert slot.resolve(True, STATUS_APPLIED) is True
    assert slot.pending() is None
    assert slot.view()["toggle_status"] == STATUS_APPLIED
    assert slot.view()["requested_mode"] == TARGET_LIVE


def test_a_newer_request_survives_resolving_an_older_one():
    slot = LiveModeSlot()
    slot.request(True)
    seen = slot.pending()
    slot.request(False)  # operator changed their mind before the worker acted

    assert slot.resolve(seen, STATUS_APPLIED) is False
    assert slot.pending() is False
    assert slot.view()["toggle_status"] == STATUS_PENDING


def test_a_refusal_records_its_reason_and_clears_nothing_else():
    slot = LiveModeSlot()
    slot.refuse(True, "Broker login failed.")
    assert slot.pending() is None
    view = slot.view()
    assert view["toggle_status"] == STATUS_REFUSED
    assert view["toggle_reason"] == "Broker login failed."
    assert view["requested_mode"] == TARGET_LIVE


def test_a_refusal_cancels_an_older_pending_request():
    """The latest click wins: the page must show the refusal, not a stale PENDING."""
    slot = LiveModeSlot()
    slot.request(True)
    slot.refuse(True, "frozen")
    assert slot.pending() is None
    assert slot.view()["toggle_status"] == STATUS_REFUSED


# ---------------------------------------------------------------------------
# LiveModeController
# ---------------------------------------------------------------------------
def test_the_controller_hands_each_request_to_the_handler_in_order():
    handled: list[LiveModeRequest] = []
    done = threading.Event()

    def handler(request):
        handled.append(request)
        if len(handled) == 2:
            done.set()

    controller = LiveModeController(handler)
    controller.start()
    try:
        assert controller.submit(LiveModeRequest("Renko", TARGET_LIVE))
        assert controller.submit(LiveModeRequest("CPR", TARGET_PAPER))
        assert done.wait(5)
    finally:
        controller.stop()
        controller.join(5)
    assert handled == [
        LiveModeRequest("Renko", TARGET_LIVE),
        LiveModeRequest("CPR", TARGET_PAPER),
    ]
    assert not controller.is_alive()


def test_a_raising_handler_does_not_kill_the_controller():
    calls = []
    done = threading.Event()

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("boom")
        done.set()

    controller = LiveModeController(handler)
    controller.start()
    try:
        controller.submit(LiveModeRequest("Renko", TARGET_LIVE))
        controller.submit(LiveModeRequest("Renko", TARGET_PAPER))
        assert done.wait(5)
    finally:
        controller.stop()
        controller.join(5)
    assert len(calls) == 2


def test_a_full_queue_refuses_instead_of_blocking():
    controller = LiveModeController(lambda request: None, maxsize=1)
    # Not started, so nothing drains the queue.
    assert controller.submit(LiveModeRequest("Renko", TARGET_LIVE)) is True
    assert controller.submit(LiveModeRequest("Renko", TARGET_PAPER)) is False


def test_a_stopped_controller_refuses_new_requests():
    controller = LiveModeController(lambda request: None)
    controller.stop()
    assert controller.submit(LiveModeRequest("Renko", TARGET_LIVE)) is False


def test_the_controller_is_a_daemon():
    """Shutdown must never wait on it."""
    assert LiveModeController(lambda request: None).daemon is True


def test_queue_type_is_bounded():
    controller = LiveModeController(lambda request: None, maxsize=3)
    assert isinstance(controller.requests, queue.Queue)
    assert controller.requests.maxsize == 3
