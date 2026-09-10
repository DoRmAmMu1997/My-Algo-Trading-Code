"""Loopback transport for the read-only monitoring dashboard.

This module owns everything between "the runner knows its state" and "a
browser tab on this machine shows it": a bounded in-memory mirror of the trade
events, one background thread that rebuilds the page's data on a fixed
cadence, and a tiny stdlib HTTP server that hands the result out.

**It is read-only by construction.** `do_GET` is the only request method
implemented; every other verb answers 405. No route reaches a broker, a
worker, or any mutable runner state. There is no code path from this module to
placing, cancelling or modifying an order.

**It binds 127.0.0.1 and nothing else.** `DASHBOARD_BIND_HOST` is a module
constant rather than a setting on purpose: there is no value anybody can put in
`.env` that makes this reachable from another machine. Exposing it on a network
should require a reviewed code change plus an access token, not an edit to a
config file at 09:10 on a trading morning.

**It must never slow trading down.** Every read it performs is either a
pure in-memory cache read or an attribute read; the assets are loaded once at
start so there is no per-request disk I/O; and the HTTP threads only ever hand
out an already-built immutable blob, so no request can reach a worker object.

Why stdlib rather than Flask/FastAPI: this runs inside a live-money process on
an old machine. `http.server` adds no dependency, no import-time surprises and
no framework thread pool, and one browser polling once a second does not need
anything more.

Configuration note: this module reads NO environment variables. Every knob is
read in the master through its `_env_*` helpers and passed in, because that is
the only place `check_env_config` audits against `env.example`.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

#: Loopback only. Deliberately NOT configurable -- see the module docstring.
DASHBOARD_BIND_HOST = "127.0.0.1"

#: Where the page's own files live. Read into memory once at start.
ASSETS_DIR = Path(__file__).resolve().parent / "dashboard_assets"

#: The complete set of servable URLs. A frozen whitelist rather than a path
#: join under a root directory: with no path arithmetic anywhere, directory
#: traversal is not merely blocked, it is unrepresentable.
ASSET_ROUTES: Mapping[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
    "/vendor/lightweight-charts.standalone.production.js": (
        "vendor/lightweight-charts.standalone.production.js",
        "text/javascript; charset=utf-8",
    ),
}

#: Sent on every response. `default-src 'none'` plus `script-src 'self'` means
#: the page cannot pull anything from the internet even if a future edit tries
#: to, which is the property that makes vendoring the chart library worthwhile.
SECURITY_HEADERS: Mapping[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    # `style-src` allows inline styles because the vendored charting library
    # injects its own <style> element for canvas layout; a hash would have to
    # be re-derived on every library upgrade. Everything that matters stays
    # strict -- `script-src 'self'` means no remote or inline JavaScript can
    # run, and `connect-src 'self'` means the page can only talk to this
    # server. There is no `form-action` target and no framing.
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'"
    ),
}

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"


# ---------------------------------------------------------------------------
# Trade-event mirror
# ---------------------------------------------------------------------------
class DashboardEventSink:
    """A bounded, in-memory copy of the trade events, for reporting only.

    Why a second sink at all, when `SessionStateStore` already records every
    event: reading that store back means `json.loads(json.dumps(state))` while
    holding the very lock `record_trade_event` needs, so a once-a-second poll
    would contend with a trading thread on the exit path. This sink is an
    `append` to a deque under a lock nobody else touches, and it keeps working
    when session state is switched off.

    `record` is called from `publish_trade_event`, i.e. from trading threads,
    so it does the least possible work: one shallow copy and one append.
    """

    def __init__(self, maxlen: int = 2000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(maxlen)))
        self._lock = threading.Lock()

    def record(self, event: Mapping[str, Any]) -> None:
        """Mirror one trade event. Called on a trading thread; keep it cheap."""

        with self._lock:
            self._events.append(dict(event))

    def seed(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Preload events carried over from an earlier run of the same day.

        Called exactly once, at startup, from the session-state document. That
        is the only time this process reads that store, so a same-day restart
        still shows the whole day's closed trades without ever putting a
        `snapshot()` call on a repeating timer.
        """

        with self._lock:
            for event in events:
                if isinstance(event, Mapping):
                    self._events.append(dict(event))

    def events(self) -> list[dict[str, Any]]:
        """A point-in-time copy, oldest first."""

        with self._lock:
            return list(self._events)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
class DashboardPublisher:
    """Holds the one rendered payload every HTTP thread serves.

    The whole concurrency story of this dashboard is here: exactly one thread
    ever builds a document, and request threads only swap out a reference to
    finished bytes. No handler thread touches a worker, a position or the
    market-data store, so no number of open browser tabs can add load to the
    trading path.
    """

    def __init__(self) -> None:
        self._payload: bytes = b""
        self._etag: str = ""
        self._generated_at: str = ""
        # The candle array is published separately and refreshed only when a
        # new minute closes, so the once-a-second poll carries the forming
        # candle alone (~120 bytes) instead of the whole session's history.
        self._chart_payload: bytes = b""
        self._chart_etag: str = ""
        self._lock = threading.Lock()

    def publish(self, payload: bytes, etag: str, generated_at: str = "") -> None:
        with self._lock:
            self._payload = payload
            self._etag = etag
            self._generated_at = generated_at

    def current(self) -> tuple[bytes, str]:
        with self._lock:
            return self._payload, self._etag

    def publish_chart(self, payload: bytes, etag: str) -> None:
        with self._lock:
            self._chart_payload = payload
            self._chart_etag = etag

    def current_chart(self) -> tuple[bytes, str]:
        with self._lock:
            return self._chart_payload, self._chart_etag

    @property
    def generated_at(self) -> str:
        with self._lock:
            return self._generated_at


class DashboardBuilderThread(threading.Thread):
    """Rebuilds the page's data on a fixed cadence, and never dies trying.

    A build that raises keeps the LAST GOOD payload on screen and logs once;
    the page's `generated_at` visibly stops advancing, which tells the operator
    the numbers are frozen without blanking the screen mid-session. Silently
    serving stale data would be worse than either.
    """

    #: After the first failure, log at DEBUG rather than repeating an
    #: exception every second into the file the EOD Sheet parse reads.
    def __init__(
        self,
        *,
        builder: Callable[[], Mapping[str, Any]],
        renderer: Callable[[Mapping[str, Any]], bytes],
        publisher: DashboardPublisher,
        refresh_seconds: float,
        log: logging.Logger,
        chart_builder: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(name="DashboardBuilder", daemon=True)
        self._builder = builder
        self._renderer = renderer
        self._publisher = publisher
        self._chart_builder = chart_builder
        self._refresh_seconds = max(0.05, float(refresh_seconds))
        self._log = log
        # NOT `_stop`: `threading.Thread` has a private `_stop()` METHOD that
        # CPython 3.12's `join()` calls through `_wait_for_tstate_lock`.
        # Shadowing it with an Event makes every `join()` raise
        # "'Event' object is not callable" -- so `stop()` would fail on 3.12
        # while passing on 3.13, which no longer takes that path.
        self._stop_event = threading.Event()
        self._failure_logged = False
        self._version = 0
        self._chart_version: object = None

    def stop(self) -> None:
        self._stop_event.set()

    def build_once(self) -> bool:
        """One build/render/publish cycle. Returns False if it failed."""

        try:
            self._version += 1
            document = dict(self._builder())
            document.setdefault("version", self._version)
            payload = self._renderer(document)
            self._publisher.publish(
                payload,
                etag=f'W/"{self._version}"',
                generated_at=str(document.get("generated_at", "")),
            )
            self._publish_chart_if_new(document)
            self._failure_logged = False
            return True
        except Exception:  # noqa: BLE001 - a monitor must never stop a session
            if not self._failure_logged:
                self._failure_logged = True
                self._log.exception(
                    "Monitoring dashboard could not rebuild its view; the last good "
                    "page stays on screen and trading is unaffected."
                )
            else:
                self._log.debug("Monitoring dashboard rebuild failed again.", exc_info=True)
            return False

    def _publish_chart_if_new(self, document: Mapping[str, Any]) -> None:
        """Re-render the candle array only when a new minute has closed.

        About 375 times a session rather than 22,500. The version in the state
        document is what tells the browser to refetch, so publishing here and
        stamping the same version keeps the two endpoints in step.
        """

        if self._chart_builder is None:
            return
        chart = document.get("chart")
        version = chart.get("series_version") if isinstance(chart, Mapping) else None
        if version is None or version == self._chart_version:
            return
        series = self._chart_builder()
        self._publisher.publish_chart(self._renderer(series), etag=f'W/"chart-{version}"')
        self._chart_version = version

    def run(self) -> None:
        # Build immediately so the first page load is never empty, then settle
        # into the cadence. `Event.wait` rather than `sleep` so a stop request
        # is honoured at once instead of after a full interval.
        self.build_once()
        while not self._stop_event.wait(self._refresh_seconds):
            self.build_once()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class _LoopbackHTTPServer(ThreadingHTTPServer):
    """A loopback-only threading server that cannot outlive the process.

    `allow_reuse_address = False` is the interesting one: if a previous runner
    is still holding the port, this bind FAILS and the caller logs and trades
    on without a dashboard. Silently sharing a port with an older process --
    and showing yesterday's numbers -- would be far worse than having no page.
    """

    daemon_threads = True
    allow_reuse_address = False
    address_family = socket.AF_INET

    def __init__(self, address, handler, *, publisher, log, allowed_hosts):
        self.publisher = publisher
        self.dashboard_log = log
        self.allowed_hosts = allowed_hosts
        self.assets: dict[str, tuple[bytes, str]] = {}
        super().__init__(address, handler)

    def handle_error(self, request, client_address) -> None:
        """Send handler tracebacks to the logger, never to stderr.

        `socketserver`'s default prints straight to `sys.stderr`. The runner's
        `setup_logging` attaches its handlers to the ROOT logger and installs
        the secret-redaction filter there, so anything bypassing logging is
        both unredacted and absent from the log file the end-of-day Google
        Sheet write parses.
        """

        self.dashboard_log.debug(
            "Monitoring dashboard request failed (client=%s).", client_address, exc_info=True
        )


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """GET-only handler over a fixed route whitelist."""

    # HTTP/1.0 keeps every response connection-closing, so no keep-alive
    # threads accumulate over a six-hour session. A fresh loopback handshake
    # once a second costs nothing.
    protocol_version = "HTTP/1.0"
    server_version = "AlgoDashboard"
    sys_version = ""

    # -- plumbing --------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:
        """Access logs go to DEBUG, not stderr (see `handle_error`)."""

        self.server.dashboard_log.debug(  # type: ignore[attr-defined]
            "dashboard %s - %s", self.address_string(), format % args
        )

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, **headers: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _plain(self, status: HTTPStatus, message: str, **headers: str) -> None:
        self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8", **headers)

    def _host_is_allowed(self) -> bool:
        """Reject a request whose Host header is not one of ours.

        Defence against DNS rebinding: a page on the open internet can resolve
        its own name to 127.0.0.1 and then read this dashboard from the
        operator's browser. Pinning the Host header closes that, and costs
        nothing for a real localhost visit.
        """

        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.server.allowed_hosts  # type: ignore[attr-defined]

    # -- the only verb ---------------------------------------------------
    def do_GET(self) -> None:
        if not self._host_is_allowed():
            self._plain(HTTPStatus.FORBIDDEN, "This dashboard serves localhost only.\n")
            return

        path = self.path.split("?", 1)[0]

        if path == "/api/state":
            payload, etag = self.server.publisher.current()  # type: ignore[attr-defined]
            if not payload:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    json.dumps({"error": "starting up"}).encode("utf-8"),
                    _JSON_CONTENT_TYPE,
                    Cache_Control="no-store",
                )
                return
            # A poll that finds nothing new should cost a header exchange, not
            # a re-render: the browser sends back the ETag it holds.
            if etag and self.headers.get("If-None-Match") == etag:
                self._send(
                    HTTPStatus.NOT_MODIFIED, b"", _JSON_CONTENT_TYPE,
                    ETag=etag, Cache_Control="no-store",
                )
                return
            self._send(
                HTTPStatus.OK, payload, _JSON_CONTENT_TYPE,
                ETag=etag, Cache_Control="no-store",
            )
            return

        if path == "/api/chart":
            payload, etag = self.server.publisher.current_chart()  # type: ignore[attr-defined]
            if not payload:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    json.dumps({"error": "no candles yet"}).encode("utf-8"),
                    _JSON_CONTENT_TYPE,
                    Cache_Control="no-store",
                )
                return
            if etag and self.headers.get("If-None-Match") == etag:
                self._send(
                    HTTPStatus.NOT_MODIFIED, b"", _JSON_CONTENT_TYPE,
                    ETag=etag, Cache_Control="no-store",
                )
                return
            self._send(
                HTTPStatus.OK, payload, _JSON_CONTENT_TYPE,
                ETag=etag, Cache_Control="no-store",
            )
            return

        asset = self.server.assets.get(path)  # type: ignore[attr-defined]
        if asset is None:
            self._plain(HTTPStatus.NOT_FOUND, "Not found.\n")
            return
        body, content_type = asset
        self._send(HTTPStatus.OK, body, content_type, Cache_Control="no-store")

    # -- everything else -------------------------------------------------
    def _method_not_allowed(self) -> None:
        """Structural read-only guarantee: nothing but GET is implemented."""

        self._plain(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "This dashboard is read-only; only GET is supported.\n",
            Allow="GET",
        )

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_HEAD = _method_not_allowed
    do_OPTIONS = _method_not_allowed


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def load_assets(assets_dir: Path = ASSETS_DIR) -> dict[str, tuple[bytes, str]]:
    """Read every whitelisted asset into memory once.

    Done at start rather than per request because the machine this runs on has
    a DRAM-less SSD where multi-second stalls are routine; a page that touched
    the disk on every poll would eventually block a request thread for seconds.
    A missing file is skipped with a warning rather than refusing to start --
    a dashboard with no chart library still shows the tables.
    """

    assets: dict[str, tuple[bytes, str]] = {}
    for route, (filename, content_type) in ASSET_ROUTES.items():
        path = assets_dir / filename
        try:
            assets[route] = (path.read_bytes(), content_type)
        except OSError:
            logging.getLogger(__name__).warning(
                "Monitoring dashboard asset missing: %s (route %s will 404).", path, route
            )
    return assets


class DashboardServer:
    """Owns the builder thread and the HTTP server as one startable unit."""

    def __init__(
        self,
        *,
        httpd: _LoopbackHTTPServer,
        builder_thread: DashboardBuilderThread,
        log: logging.Logger,
    ) -> None:
        self._httpd = httpd
        self._builder_thread = builder_thread
        self._log = log
        self._serve_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="DashboardHTTP",
            daemon=True,
            kwargs={"poll_interval": 0.2},
        )

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{DASHBOARD_BIND_HOST}:{self.port}/"

    def start(self) -> None:
        self._builder_thread.start()
        self._serve_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Release the port and stop rebuilding.

        Both threads are daemons and `daemon_threads` is set on the server, so
        the timeout is a courtesy for a tidy log line, never something process
        exit depends on.
        """

        self._builder_thread.stop()
        try:
            self._httpd.shutdown()
        finally:
            self._httpd.server_close()
        self._builder_thread.join(timeout=timeout)
        self._serve_thread.join(timeout=timeout)
        if self._serve_thread.is_alive() or self._builder_thread.is_alive():
            self._log.debug("Monitoring dashboard threads outlived their stop timeout.")


def start_dashboard(
    *,
    port: int,
    refresh_seconds: float,
    builder: Callable[[], Mapping[str, Any]],
    renderer: Callable[[Mapping[str, Any]], bytes],
    log: logging.Logger,
    chart_builder: Callable[[], Mapping[str, Any]] | None = None,
    assets_dir: Path = ASSETS_DIR,
) -> DashboardServer | None:
    """Bind, wire and start the dashboard, or return None and log why.

    Returning None rather than raising is the contract: the caller is `main()`
    in a live-money runner, and a monitoring page failing to start is never a
    reason not to trade. Pass `port=0` to let the OS pick a free one (the
    tests do; production passes the configured port so the URL is stable).
    """

    publisher = DashboardPublisher()
    try:
        httpd = _LoopbackHTTPServer(
            (DASHBOARD_BIND_HOST, int(port)),
            DashboardRequestHandler,
            publisher=publisher,
            log=log,
            allowed_hosts=frozenset(),
        )
    except OSError as exc:
        log.warning(
            "Monitoring dashboard could not bind %s:%s (%s); trading continues without it. "
            "An older runner may still be holding the port.",
            DASHBOARD_BIND_HOST,
            port,
            exc,
        )
        return None

    bound_port = int(httpd.server_address[1])
    # Built after binding because port 0 is only resolved by the bind itself.
    httpd.allowed_hosts = frozenset(
        {
            f"{DASHBOARD_BIND_HOST}:{bound_port}",
            f"localhost:{bound_port}",
        }
    )
    httpd.assets = load_assets(assets_dir)

    builder_thread = DashboardBuilderThread(
        builder=builder,
        renderer=renderer,
        publisher=publisher,
        refresh_seconds=refresh_seconds,
        log=log,
        chart_builder=chart_builder,
    )
    server = DashboardServer(httpd=httpd, builder_thread=builder_thread, log=log)
    server.start()
    return server
