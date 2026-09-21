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
import re
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


def test_every_element_the_page_script_looks_up_exists_in_the_markup():
    """A wired id missing from the HTML blanks the whole page, silently.

    The control-wiring loop in `dashboard.js` does `const box = el(id)` with no
    null guard, so one absent id throws inside the IIFE and NOTHING on the page
    renders -- not a missing checkbox, a blank dashboard. Nothing else in the
    repository reads these assets, so that would ship unnoticed.

    Two lookup shapes are checked: a literal `el("x")` / `getElementById("x")`,
    and the `["element-id", "prefKey"]` pairs the checkbox loop iterates, where
    the id never appears next to `el(` at all.

    The pair shape is read from INSIDE the wiring loop rather than from any
    two-string array in the file. Matching those wherever they appeared made an
    ordinary `for (const name of ["stoch_k", "stoch_d"])` look like a wired
    element id, and failed the test for something that is not one.
    """

    assets = ASSETS_DIR
    script = (assets / "dashboard.js").read_text(encoding="utf-8")
    markup = (assets / "index.html").read_text(encoding="utf-8")

    wiring = re.search(r"for \(const \[id, key\] of \[(.*?)\]\) \{", script, re.DOTALL)
    assert wiring, "the control-wiring loop was not found; this test is checking nothing"

    looked_up = set(
        re.findall(r'(?:el|getElementById)\(\s*"([A-Za-z0-9_-]+)"\s*\)', script)
    ) | set(
        re.findall(
            r'\[\s*"([A-Za-z0-9_-]+)"\s*,\s*"[A-Za-z0-9_]+"\s*\]', wiring.group(1)
        )
    )
    declared = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', markup))

    # Sanity check: if the patterns stop matching, this test would "pass" while
    # checking nothing at all.
    assert len(looked_up) > 20, f"id extraction looks broken: found only {len(looked_up)}"
    assert looked_up <= declared, (
        "dashboard.js looks up element ids that index.html does not define: "
        f"{sorted(looked_up - declared)}"
    )


def _render_cpr_lines_source() -> str:
    """The body of `renderCprLines`, read out of the page script.

    Brace counting rather than a regex: the function holds object literals,
    template strings and block comments, and a lazy match would stop at the
    first `}` inside any of them.
    """

    script = (ASSETS_DIR / "dashboard.js").read_text(encoding="utf-8")
    start = script.index("function renderCprLines(")
    opening = script.index("{", start)
    depth = 0
    for offset in range(opening, len(script)):
        if script[offset] == "{":
            depth += 1
        elif script[offset] == "}":
            depth -= 1
            if depth == 0:
                return script[start : offset + 1]
    raise AssertionError("renderCprLines is not balanced; this test is checking nothing")


def test_the_cpr_levels_are_drawn_as_steps_and_never_rely_on_whitespace():
    """A whitespace point cannot break a CPR line, so the levels must step.

    Lightweight-charts 5.2.1 filters valueless rows out in its data layer --
    `rows.filter(hasValue)` -- before a line series ever sees them, so a point
    with a `time` and no `value` contributes exactly one empty column and no
    gap. The stroke stays continuous, and a level then draws a DIAGONAL from
    one day's price to the next across the overnight gap. `LineType.WithSteps`
    is what actually separates the days.

    This is a source assertion because the repository has no JS runtime. It
    exists so a future edit that "restores the gap" with a whitespace point --
    the obvious-looking fix -- fails here instead of shipping the diagonal
    back to the chart.
    """

    body = _render_cpr_lines_source()

    assert "LightweightCharts.LineType.WithSteps" in body, (
        "renderCprLines no longer sets LineType.WithSteps; without it the CPR "
        "levels slope between days instead of stepping"
    )

    pushes = re.findall(r"points\.push\(\{(.*?)\}\)", body, re.DOTALL)
    assert pushes, "no points.push() found in renderCprLines; this test is checking nothing"
    valueless = [push.strip() for push in pushes if "value" not in push]
    assert not valueless, (
        "renderCprLines pushes a point with no value: " + repr(valueless) + ". "
        "The library drops it, so it breaks nothing and only adds a column."
    )


# ---------------------------------------------------------------------------
# History paging
# ---------------------------------------------------------------------------
HISTORY_PAGES = {
    "1:0": b'{"tf":"1","page":0,"pages":2,"bars":[{"time":300}]}',
    "1:1": b'{"tf":"1","page":1,"pages":2,"bars":[{"time":100}]}',
    "D:0": b'{"tf":"D","page":0,"pages":1,"bars":[{"time":1}]}',
}


def _serve(history_builder, *, refresh_seconds=30.0):
    """A dashboard whose history comes from `history_builder`."""

    return start_dashboard(
        port=0,
        refresh_seconds=refresh_seconds,
        builder=lambda: {"generated_at": "10:00:00", "chart": {"series_version": 1}},
        renderer=_render,
        chart_builder=lambda: {"series_version": 1, "bars": []},
        history_builder=history_builder,
        log=QUIET,
    )


def _await_history(url, *, expect=200, tries=50):
    """History loads on the builder thread just after the first payload."""

    for _ in range(tries):
        status, headers, body = get(url)
        if status == expect:
            return status, headers, body
        threading.Event().wait(0.05)
    return get(url)


@pytest.fixture()
def history_server():
    running = _serve(lambda: HISTORY_PAGES)
    assert running is not None
    try:
        yield running
    finally:
        running.stop(timeout=2.0)


def test_a_history_page_is_served(history_server):
    status, headers, body = _await_history(history_server.url + "api/history?tf=1&page=0")

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(body)["bars"] == [{"time": 300}]


def test_an_older_page_is_a_separate_request(history_server):
    _await_history(history_server.url + "api/history?tf=1&page=0")
    status, _, body = get(history_server.url + "api/history?tf=1&page=1")

    assert status == 200
    assert json.loads(body)["page"] == 1


def test_a_page_past_the_start_of_history_is_404(history_server):
    """How the browser learns to stop asking for older candles."""

    _await_history(history_server.url + "api/history?tf=1&page=0")
    status, _, _ = get(history_server.url + "api/history?tf=1&page=2")

    assert status == 404


def test_an_unknown_timeframe_is_refused(history_server):
    """The frozen set is the whole contract; nothing is inferred from the store."""

    _await_history(history_server.url + "api/history?tf=1&page=0")
    for timeframe in ("15", "", "../secrets", "1;DROP"):
        status, _, _ = get(history_server.url + f"api/history?tf={timeframe}&page=0")
        assert status == 404, timeframe


def test_a_page_that_is_not_a_number_is_a_bad_request(history_server):
    _await_history(history_server.url + "api/history?tf=1&page=0")
    for page in ("abc", "1.5", "0x1", ""):
        status, _, _ = get(history_server.url + f"api/history?tf=1&page={page}")
        assert status == 400, page


def test_a_negative_page_is_a_bad_request(history_server):
    _await_history(history_server.url + "api/history?tf=1&page=0")
    status, _, _ = get(history_server.url + "api/history?tf=1&page=-1")

    assert status == 400


def test_without_history_every_page_is_404_and_the_live_view_is_fine(server):
    """The operator may never have downloaded any of it."""

    status, _, _ = get(server.url + "api/history?tf=1&page=0")
    assert status == 404

    status, _, body = get(server.url + "api/state")
    assert status == 200
    assert json.loads(body)["generated_at"]


def test_a_history_builder_that_raises_costs_only_the_scroll_back():
    """History is a luxury; the session is not."""

    def explode():
        raise RuntimeError("no disk today")

    running = _serve(explode)
    assert running is not None
    try:
        status, _, _ = get(running.url + "api/history?tf=1&page=0")
        assert status == 404

        status, _, body = get(running.url + "api/state")
        assert status == 200, "the live view must be untouched by a history failure"
        assert json.loads(body)["generated_at"]
    finally:
        running.stop(timeout=2.0)


def test_a_query_string_still_does_not_reach_the_other_routes(server):
    """`do_GET` splits the query off before routing; that must stay true."""

    status, _, body = get(server.url + "api/state?cachebust=1")

    assert status == 200
    assert json.loads(body)["generated_at"]


def test_the_live_view_keeps_rebuilding_while_history_loads():
    """Half a minute of CSV must not freeze the page into its staleness banner.

    Building the real five-year history measured 29.5 seconds. On the builder
    thread that would stop the live document rebuilding for that whole time,
    and the page decides the runner has stopped after five quiet intervals.
    History therefore loads on a thread of its own, and this is what says so.
    """

    started = threading.Event()
    release = threading.Event()

    def slow_history():
        started.set()
        release.wait(timeout=10.0)
        return HISTORY_PAGES

    running = _serve(slow_history, refresh_seconds=0.05)
    assert running is not None
    try:
        assert started.wait(timeout=5.0), "history load never began"

        # History is still blocked here. The live view must keep advancing.
        first = get(running.url + "api/state")[1]["ETag"]
        for _ in range(60):
            threading.Event().wait(0.05)
            latest = get(running.url + "api/state")[1]["ETag"]
            if latest != first:
                break
        else:
            raise AssertionError("the live view froze while history loaded")

        # And history still arrives once it finishes.
        release.set()
        status, _, _ = _await_history(running.url + "api/history?tf=1&page=0")
        assert status == 200
    finally:
        release.set()
        running.stop(timeout=2.0)
