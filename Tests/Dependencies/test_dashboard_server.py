"""Tests for the dashboard's transport (`Dependencies/dashboard_server.py`).

Two properties matter more than the rest, and both are asserted directly:

* **Read-only.** GET is the only verb; everything else must answer 405. There
  must be no path arithmetic a traversal could exploit, and no CORS header.
* **Silent.** Nothing may reach `sys.stderr`, and nothing may attach a handler
  to the root logger. The runner's `setup_logging` early-returns when root
  already has handlers, so a monitoring module that touches root would take
  out the log file the end-of-day Google Sheet write parses.

The HTTP cases drive a real socket bound to an ephemeral loopback port, which
covers the header and status plumbing a mock would silently paper over.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Bare import: this folder's conftest.py puts the SOURCE `Dependencies/` on
# sys.path, which is the same resolution the runtime performs.
from check_env_config import env_keys_read_by
from dashboard_server import (
    ASSET_ROUTES,
    ASSETS_DIR,
    DASHBOARD_BIND_HOST,
    DashboardBuilderThread,
    DashboardEventSink,
    DashboardPublisher,
    load_assets,
    start_dashboard,
)

QUIET = logging.getLogger("test_dashboard_server")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False


def _render(document):
    return json.dumps(document).encode("utf-8")


@pytest.fixture()
def server():
    """A live dashboard on an ephemeral loopback port, torn down after."""

    state = {"n": 0}

    def builder():
        state["n"] += 1
        return {
            "generated_at": f"10:00:{state['n']:02d}",
            "chart": {"series_version": 1},
        }

    running = start_dashboard(
        port=0,
        refresh_seconds=30.0,  # one build; the tests drive changes explicitly
        builder=builder,
        renderer=_render,
        chart_builder=lambda: {"series_version": 1, "bars": [{"time": 1, "close": 2}]},
        log=QUIET,
    )
    assert running is not None
    try:
        yield running
    finally:
        running.stop(timeout=2.0)


def get(url, *, method="GET", headers=None):
    """Return (status, headers, body) without raising on 3xx/4xx."""

    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        # The URL is always this test's own loopback server, built from the
        # ephemeral port the fixture bound: no external or caller-supplied
        # scheme exists here for B310 to audit.
        with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


# ---------------------------------------------------------------------------
# The event sink
# ---------------------------------------------------------------------------
def test_the_sink_copies_events_so_a_later_mutation_cannot_rewrite_history():
    sink = DashboardEventSink(maxlen=10)
    event = {"action": "ENTRY", "pnl": 1.0}
    sink.record(event)
    event["pnl"] = 999.0
    assert sink.events() == [{"action": "ENTRY", "pnl": 1.0}]


def test_the_sink_is_bounded_and_keeps_the_newest():
    sink = DashboardEventSink(maxlen=3)
    for index in range(6):
        sink.record({"seq": index})
    assert [event["seq"] for event in sink.events()] == [3, 4, 5]


def test_seeding_ignores_anything_that_is_not_an_event():
    sink = DashboardEventSink(maxlen=10)
    sink.seed([{"action": "ENTRY"}, None, "nope", 7])
    assert sink.events() == [{"action": "ENTRY"}]


# ---------------------------------------------------------------------------
# Publisher and builder
# ---------------------------------------------------------------------------
def test_the_publisher_round_trips_a_payload_and_its_etag():
    publisher = DashboardPublisher()
    assert publisher.current() == (b"", "")
    publisher.publish(b"{}", etag='W/"1"', generated_at="10:00:00")
    assert publisher.current() == (b"{}", 'W/"1"')
    assert publisher.generated_at == "10:00:00"


def test_a_builder_that_raises_keeps_the_last_good_payload_on_screen():
    """Blanking a live dashboard mid-session would be worse than freezing it."""
    publisher = DashboardPublisher()
    failing = {"now": False}

    def builder():
        if failing["now"]:
            raise RuntimeError("worker exploded")
        return {"generated_at": "10:00:00"}

    thread = DashboardBuilderThread(
        builder=builder, renderer=_render, publisher=publisher,
        refresh_seconds=30.0, log=QUIET,
    )
    assert thread.build_once() is True
    good, good_etag = publisher.current()

    failing["now"] = True
    assert thread.build_once() is False
    assert publisher.current() == (good, good_etag)


def test_the_chart_series_is_republished_only_when_its_version_moves():
    publisher = DashboardPublisher()
    version = {"n": 1}
    calls = {"n": 0}

    def chart_builder():
        calls["n"] += 1
        return {"series_version": version["n"], "bars": []}

    thread = DashboardBuilderThread(
        builder=lambda: {"chart": {"series_version": version["n"]}},
        renderer=_render, publisher=publisher, refresh_seconds=30.0, log=QUIET,
        chart_builder=chart_builder,
    )
    thread.build_once()
    thread.build_once()
    thread.build_once()
    assert calls["n"] == 1, "an unchanged minute must not re-serialize the candles"

    version["n"] = 2
    thread.build_once()
    assert calls["n"] == 2


def test_the_builder_thread_shadows_nothing_that_threading_owns():
    """Regression guard for a bug only Python 3.12 surfaced.

    `DashboardBuilderThread` originally kept its stop flag in `self._stop`.
    `threading.Thread` already has a private `_stop()` METHOD, which CPython
    3.12's `join()` calls through `_wait_for_tstate_lock` -- so every join
    raised "'Event' object is not callable". Python 3.13 no longer takes that
    path, so the local run and the 3.13 CI leg passed while 3.12 failed, and
    `dashboard.stop()` would have failed the same way in production.

    Checking every attribute, rather than re-testing `_stop` alone, means the
    next accidental collision fails here on any version.
    """

    thread = DashboardBuilderThread(
        builder=lambda: {}, renderer=_render, publisher=DashboardPublisher(),
        refresh_seconds=30.0, log=QUIET,
    )
    # Reflection over the RUNNING interpreter is not enough on its own: 3.13
    # deleted `Thread._stop`, so on 3.13 there is nothing left to collide with
    # and the original bug would sail straight through. The union with names
    # that existed in a supported-but-older version is what makes this guard
    # work on the interpreter that is not failing.
    removed_in_newer_pythons = {"_stop", "_wait_for_tstate_lock", "_reset_internal_locks"}
    reserved = {
        name
        for name in dir(threading.Thread)
        if callable(getattr(threading.Thread, name, None))
    } | removed_in_newer_pythons

    shadowed = sorted(set(vars(thread)) & reserved)
    assert shadowed == [], (
        "these instance attributes shadow threading.Thread methods and will "
        f"break join() or start() on at least one supported Python: {shadowed}"
    )


def test_joining_the_builder_thread_after_it_exits_does_not_raise():
    """The exact call `DashboardServer.stop` makes, which 3.12 broke on."""

    thread = DashboardBuilderThread(
        builder=lambda: {"generated_at": "x"}, renderer=_render,
        publisher=DashboardPublisher(), refresh_seconds=30.0, log=QUIET,
    )
    thread.start()
    thread.stop()
    thread.join(timeout=2.0)
    thread.join(timeout=2.0)  # a second join must be a no-op, not a TypeError
    assert not thread.is_alive()


def test_the_builder_thread_stops_promptly_when_asked():
    publisher = DashboardPublisher()
    thread = DashboardBuilderThread(
        builder=lambda: {"generated_at": "x"}, renderer=_render, publisher=publisher,
        refresh_seconds=30.0, log=QUIET,
    )
    thread.start()
    thread.stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "stop must not wait out the refresh interval"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def test_the_state_endpoint_serves_json_with_an_etag(server):
    status, headers, body = get(server.url + "api/state")
    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["ETag"]
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(body)["generated_at"].startswith("10:00:")


def test_an_unchanged_document_answers_304(server):
    _, headers, _ = get(server.url + "api/state")
    status, _, body = get(server.url + "api/state", headers={"If-None-Match": headers["ETag"]})
    assert status == 304
    assert body == b""


def test_the_chart_endpoint_serves_the_candle_array(server):
    status, _, body = get(server.url + "api/chart")
    assert status == 200
    assert json.loads(body)["bars"] == [{"time": 1, "close": 2}]


def test_the_index_page_and_its_assets_are_served(server):
    for route, (_filename, content_type) in ASSET_ROUTES.items():
        status, headers, body = get(server.url.rstrip("/") + route)
        assert status == 200, route
        assert headers["Content-Type"] == content_type
        assert body


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_every_verb_but_get_is_refused(server, method):
    """The read-only guarantee, asserted rather than asserted-in-a-docstring."""
    status, headers, _ = get(server.url, method=method)
    assert status == 405
    assert headers["Allow"] == "GET"


def test_an_unknown_path_is_404(server):
    assert get(server.url + "nope")[0] == 404


def test_traversal_cannot_reach_a_repository_file(server):
    """There is no path arithmetic to exploit -- routes are a frozen whitelist."""
    for path in (
        "../nifty_multi_strategy_master.py",
        "..%2F..%2FDependencies%2F.env",
        "vendor/../../session_state.py",
    ):
        assert get(server.url + path)[0] == 404, path


def test_a_foreign_host_header_is_refused(server):
    """DNS-rebinding defence: a page on the internet must not read this."""
    status, _, _ = get(server.url + "api/state", headers={"Host": "evil.example"})
    assert status == 403


@pytest.mark.parametrize("host_suffix", ["", "localhost"])
def test_the_expected_hosts_are_accepted(server, host_suffix):
    headers = {"Host": f"{host_suffix}:{server.port}"} if host_suffix else None
    assert get(server.url + "api/state", headers=headers)[0] == 200


def test_every_response_carries_the_security_headers_and_no_cors(server):
    _, headers, _ = get(server.url)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    assert headers.get("Access-Control-Allow-Origin") is None


# ---------------------------------------------------------------------------
# Silence
# ---------------------------------------------------------------------------
def test_serving_writes_nothing_to_stderr(server, monkeypatch):
    """`http.server` logs every request to stderr by default.

    Stray stderr writes pollute the log the end-of-day Google Sheet parse
    reads, and would bypass the runner's secret-redaction filter.
    """
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)
    get(server.url + "api/state")
    get(server.url + "definitely-not-a-route")
    assert captured.getvalue() == ""


def test_importing_and_running_the_server_never_touches_the_root_logger(server):
    """`setup_logging()` early-returns if root already has handlers.

    A monitoring module that attached one would silently disable the runner's
    log file AND its secret redaction.
    """
    before = list(logging.getLogger().handlers)
    get(server.url)
    assert logging.getLogger().handlers == before


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def test_a_port_already_in_use_disables_the_dashboard_rather_than_raising():
    """A second runner must FAIL to bind, not quietly share the socket and
    show the older process's numbers."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind((DASHBOARD_BIND_HOST, 0))
    holder.listen(1)
    try:
        assert start_dashboard(
            port=holder.getsockname()[1],
            refresh_seconds=1.0,
            builder=lambda: {},
            renderer=_render,
            log=QUIET,
        ) is None
    finally:
        holder.close()


def test_every_declared_asset_route_resolves_to_a_file_that_exists():
    """Catches a rename that would leave the page serving a 404 for its own JS."""
    missing = [
        filename for filename, _ in ASSET_ROUTES.values()
        if not (ASSETS_DIR / filename).is_file()
    ]
    assert missing == []


def test_a_missing_asset_directory_still_yields_a_usable_server(tmp_path: Path):
    """A dashboard with no chart library still shows the tables."""
    assets = load_assets(tmp_path)
    assert assets == {}


def test_the_bind_host_is_loopback_and_not_configurable():
    """The safety property: no `.env` value can expose this off-box.

    Both dashboard modules must also read no configuration of their own. All
    of it is read in the master, which is where `check_env_config` audits
    against `env.example`; a knob read here would escape that audit -- and a
    host knob specifically would turn a reviewed code change into a one-line
    `.env` edit.
    """

    assert DASHBOARD_BIND_HOST == "127.0.0.1"
    dependencies = Path(__file__).resolve().parents[2] / "Dependencies"
    for module in ("dashboard_server.py", "dashboard_snapshot.py"):
        path = dependencies / module
        # The repo's own AST auditor, so this test and `algo.py check-env`
        # can never disagree about what counts as reading a setting.
        assert env_keys_read_by(path) == set(), module
        # `os` is the only route to an unaudited read, and neither module has
        # any other use for it.
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "os" not in imported, module


def test_the_server_stops_without_leaving_its_threads_running():
    running = start_dashboard(
        port=0, refresh_seconds=0.1, builder=lambda: {"generated_at": "x"},
        renderer=_render, log=QUIET,
    )
    assert running is not None
    running.stop(timeout=2.0)
    names = {thread.name for thread in threading.enumerate()}
    assert "DashboardBuilder" not in names
    assert "DashboardHTTP" not in names
