"""
Thread-safe Zerodha Kite Connect v3 execution client for the multi-strategy runner.

The master runner deliberately knows almost nothing about broker APIs.  It asks
each broker helper to log in, preload a contract master, resolve one option and
place a confirmed market order.  This module implements that same surface for
Zerodha using only Kite Connect's documented REST endpoints and ``requests`` --
no SDK, so there is no extra dependency to install or audit.

It is used in two places, through the ONE singleton at the bottom of this file:

* as a ``LIVE_BROKER`` choice (``LIVE_BROKER=ZERODHA``) for the intraday
  strategies, exactly like the Kotak / Shoonya / Flattrade / Dhan helpers; and
* by the optional position-group monitor (``docs/adr/0022``), which reads the
  operator's carried positions here and squares a group off at its target.

Safety rules followed here (the same ones every broker helper follows):

* Importing this module never logs in, opens a browser, or sends an order.
* A supplied daily access token is verified through ``/user/profile`` before use.
* Without a valid token, the operator completes Kite's browser login and pastes
  the redirect URL; the API secret never leaves this process.  That interactive
  step runs ONLY on the main thread: a strategy or monitor thread that finds the
  session gone gets ``False`` and a log line, never a hidden ``input()`` prompt.
* Every HTTP call keeps a native timeout; one total ten-second deadline also
  covers rate-limit wait, session-lock wait, and slow-dribble response bodies.
* ``place_market_order`` returns a typed result only after the order's history
  proves an outcome.  An acknowledgement is not a fill.
* A refusal is called ``REJECTED`` only after the order book has been searched
  for the order's tag and found not to hold it.  A transport error, a timeout,
  a 5xx, a 429, or an ``error_type`` that admits the OMS may have acted
  (``NetworkException``, ``GeneralException``) is ``UNKNOWN``.
* A call that outlives its deadline poisons the session: no new order is
  accepted until ``recover_after_reconciliation`` clears it.

Beginner's mental model -- one live order travels through this file as follows:

1. ``ensure_logged_in()`` obtains or validates today's access token.
2. ``preload_scrip_master()`` downloads Kite's list of real F&O contracts.
3. ``resolve_option_symbol()`` converts strategy details such as
   NIFTY/27-NOV-2025/CE/24500 into Kite's exact ``tradingsymbol``.
4. ``place_market_order()`` translates friendly values such as INTRADAY into
   Kite's codes (MIS), then submits the order.
5. ``_confirm_fill()`` reads the order's history; only a terminal state with
   the full quantity filled is reported as ``FILLED``.

Small glossary:

* ``tradingsymbol`` is Kite's contract name, e.g. ``NIFTY25N2724500CE``.
* ``NFO`` / ``BFO`` are the NSE and BSE futures-and-options exchanges.
* ``MIS`` is intraday margin (auto-squared-off by the broker); ``NRML`` is the
  carry-forward product.  The friendly names INTRADAY / NORMAL map to them.
* ``tag`` is a short label Kite stores with an order.  Kite accepts only
  letters and digits, at most 20 of them, so the runner's hyphenated tag is
  squeezed to that form before it is sent AND before it is searched for.
* *Market protection* limits how far a MARKET order may chase a moving price;
  ``-1`` asks Kite to apply its own automatic band.

Kite's ``DELETE /session/token`` would invalidate the access token for the rest
of the day, and it puts the token in the URL.  ``logout()`` therefore only
closes the local HTTP session and erases the in-memory token.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
import math
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

# The broker helpers live in folders with spaces and are also executable as
# standalone diagnostics.  Add the shared ``Dependencies`` folder (for the bare
# ``broker_contract`` / ``broker_rate_limit`` names) and the repository root
# (for ``Dependencies.secret_redaction``) explicitly, so both import styles
# resolve the same way from the runner and from ``python <diagnostic>``.
_DEPENDENCIES_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _DEPENDENCIES_DIR.parent
for _import_root in (_REPO_ROOT, _DEPENDENCIES_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from broker_contract import (  # noqa: E402
    TERMINAL_BROKER_STATES,
    BrokerQueryResult,
    OpenOrder,
    OpenPosition,
    OrderResult,
    OrderStatus,
    PositionDetail,
    exact_int,
    normalize_order_result,
)
from broker_rate_limit import RollingWindowRateLimiter  # noqa: E402

from Dependencies.secret_redaction import redact_text  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_DEPENDENCIES_DIR / ".env", override=False)
except Exception:
    # ``python-dotenv`` is a core dependency, but keeping import-time behaviour
    # harmless means paper trading can still start and report a clear login error.
    pass


log = logging.getLogger(__name__)

_BASE_URL = "https://api.kite.trade"
_LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
_VERSION_HEADER = ("X-Kite-Version", "3")

_API_TIMEOUT_SECONDS = 10.0
_BROKER_CALL_DEADLINE_SECONDS = 10.0
# The instrument dump is several megabytes and is read once, before any order
# (at startup, or lazily by the resolver).  It gets a longer budget than an
# order call, as Kotak's scrip master does.  It never runs on an exit path.
_SCRIP_MASTER_TIMEOUT_SECONDS = 30.0
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 2.0

# Friendly runner vocabulary -> Kite product codes.  Kite's own codes are
# accepted too (as Dhan's helper accepts MARGIN): the position-group monitor
# closes a carried leg in the product the broker's position book reports.
_PRODUCT_MAP = {"INTRADAY": "MIS", "NORMAL": "NRML", "MIS": "MIS", "NRML": "NRML"}
_SIDES = frozenset({"BUY", "SELL"})
_OPTION_EXCHANGES = frozenset({"NFO", "BFO"})
_MAX_TAG_LENGTH = 20

# ``error_type`` values that prove Kite answered and refused, as opposed to
# "the request may have reached the OMS".  Kite documents NetworkException as
# "unable to communicate with the OMS" and GeneralException as "an unclassified
# error" -- neither proves nothing was created, so neither is listed here.
_REFUSAL_ERROR_TYPES = frozenset(
    {
        "INPUTEXCEPTION",
        "ORDEREXCEPTION",
        "TOKENEXCEPTION",
        "USEREXCEPTION",
        "PERMISSIONEXCEPTION",
        "MARGINEXCEPTION",
        "HOLDINGEXCEPTION",
    }
)


def _positive_price(value: Any) -> float:
    """Return a finite positive price, or 0.0 meaning "the broker reported none"."""

    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _env_str(name: str, default: str = "") -> str:
    """Read one environment value and remove accidental wrapping quotes."""

    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value or default


def _broker_tag(order_tag: Any) -> str:
    """Squeeze a runner tag into the form Kite stores: letters and digits, max 20.

    The execution ledger builds tags like ``M2-ABC-12345678-EN01`` (20 characters
    WITH hyphens).  Kite rejects anything that is not alphanumeric, so the
    hyphens are dropped.  The same function is applied to every tag read back
    from the order book, so a lookup always compares like with like.
    """

    cleaned = "".join(ch for ch in str(order_tag or "") if ch.isascii() and ch.isalnum())
    return cleaned[:_MAX_TAG_LENGTH]


def _as_date(value: Any) -> date | None:
    """Turn a date-like value (date, datetime/Timestamp, or text) into a date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    for candidate, fmt in ((text[:10], "%Y-%m-%d"), (text, "%d-%b-%Y"), (text, "%d-%m-%Y")):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def extract_request_token(pasted: str) -> str:
    """Accept either the full Kite redirect URL or the bare ``request_token``."""

    cleaned = str(pasted or "").strip()
    if "request_token=" in cleaned:
        values = parse_qs(urlparse(cleaned).query).get("request_token") or []
        if not values:
            values = parse_qs(cleaned.split("?", 1)[-1]).get("request_token") or []
        return values[0].strip() if values else ""
    return cleaned


def kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Kite's session checksum: SHA-256 of api_key + request_token + api_secret.

    This is the handshake Kite's ``/session/token`` endpoint mandates.  It proves
    we hold the secret without sending it, and binds the exchange to THIS
    request token -- it is not a stored password hash, so a slow password
    hashing function would simply be rejected by the API.
    """

    return hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InstrumentRow:
    """One listed option contract from Kite's instrument dump."""

    symbol: str
    exchange: str
    underlying: str
    expiry: date
    strike: float
    right: str
    lot_size: int


def parse_instrument_csv(text: str) -> list[InstrumentRow]:
    """Keep the F&O OPTION rows of an instrument CSV; skip everything else.

    Kite's columns are ``instrument_token, exchange_token, tradingsymbol, name,
    last_price, expiry, strike, tick_size, lot_size, instrument_type, segment,
    exchange``.  A row with an unparseable expiry or strike is dropped: a
    contract this module cannot identify exactly is simply absent, and an absent
    contract makes the caller refuse rather than guess.
    """

    rows: list[InstrumentRow] = []
    for raw in csv.DictReader(io.StringIO(text)):
        exchange = str(raw.get("exchange") or "").strip().upper()
        right = str(raw.get("instrument_type") or "").strip().upper()
        if exchange not in _OPTION_EXCHANGES or right not in {"CE", "PE"}:
            continue
        symbol = str(raw.get("tradingsymbol") or "").strip()
        underlying = str(raw.get("name") or "").strip().strip('"').upper()
        expiry = _as_date(raw.get("expiry"))
        try:
            strike = float(raw.get("strike") or "")
        except ValueError:
            continue
        lot_size = exact_int(raw.get("lot_size")) or 0
        if not symbol or not underlying or expiry is None or not math.isfinite(strike):
            continue
        rows.append(InstrumentRow(symbol, exchange, underlying, expiry, strike, right, lot_size))
    return rows


def _csv_text(response: Any) -> str:
    """Return a CSV response as text, un-gzipping it if the body is still gzip.

    Kite documents the dump as "a gzipped CSV".  ``requests`` normally undoes
    that transparently; the magic-byte check covers a server that sends the gzip
    file itself without the matching ``Content-Encoding`` header.
    """

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)) and bytes(content[:2]) == b"\x1f\x8b":
        return gzip.decompress(bytes(content)).decode("utf-8", errors="replace")
    return str(response.text)


class ZerodhaExecutionClient:
    """One lazily authenticated, lock-guarded Kite Connect execution session.

    The master creates many strategy threads but imports one singleton from the
    bottom of this module.  All those threads -- and the position-group monitor
    -- therefore share one HTTP session, one token, one contract cache, one
    order gate and one pair of rate-limit counters.  ``RLock`` makes nested
    helper calls safe without allowing simultaneous session writes.
    """

    broker_name = "ZERODHA"

    def __init__(self) -> None:
        self.client: requests.Session | None = None
        self.is_logged_in = False
        self._lock = threading.RLock()
        self._api_key = ""
        self._access_token = ""
        # exchange -> {tradingsymbol: row}; an exchange appears here only after
        # its dump downloaded and parsed, so a failure is retried, not cached.
        self._instruments: dict[str, dict[str, InstrumentRow]] = {}
        # (exchange, underlying, expiry, right, strike in paise) -> tradingsymbol
        self._resolver: dict[tuple[str, str, date, str, int], str] = {}
        # ``requests`` timeouts are inactivity limits, not a total wall-clock
        # budget: a slow-dribble response can otherwise run forever.  Execute one
        # native HTTP call at a time and impose a caller-visible deadline around
        # rate-limit wait, session-lock wait, and the complete response body.
        self._http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zerodha-http")
        self._http_poisoned = False
        self._timed_out_future: Future[Any] | None = None
        # Serialize acknowledgement plus order-history confirmation.  Without
        # this gate another thread can submit before a status timeout poisons
        # the session, leaving two simultaneously indeterminate live orders.
        self._order_submission_lock = threading.Lock()
        # Kite documents 10 requests/second for order placement and for "all
        # other endpoints", plus 400 orders/minute.  These local budgets stay
        # about a fifth under those figures, because a 429 from Kite arrives as
        # an indeterminate answer and an indeterminate order freezes trading.
        self._api_limiter = RollingWindowRateLimiter(
            8, 480, _MAX_RATE_LIMIT_WAIT_SECONDS, label="Zerodha API"
        )
        self._order_limiter = RollingWindowRateLimiter(
            8, 320, _MAX_RATE_LIMIT_WAIT_SECONDS, label="Zerodha order API"
        )

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _ensure_session_locked(self) -> requests.Session:
        """Create the shared HTTP session on first use; caller holds ``_lock``."""

        if self.client is None:
            self.client = requests.Session()
        return self.client

    def _secrets(self) -> tuple[str, ...]:
        """Values that must never appear in a log line or a returned reason."""

        return tuple(value for value in (self._api_key, self._access_token) if value)

    def _run_http_with_deadline(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        deadline: float,
        budget_seconds: float,
        allow_after_poison: bool = False,
    ) -> Any:
        """Run one native HTTP call without letting a slow response outlive its budget.

        The caller holds ``_lock``.  If the future times out, its socket may
        still be active inside ``requests``; poison the shared session so no
        new order can overlap that indeterminate in-flight operation.
        """

        if self._http_poisoned:
            abandoned_finished = (
                self._timed_out_future is not None and self._timed_out_future.done()
            )
            if not allow_after_poison or not abandoned_finished:
                raise RuntimeError(
                    "Zerodha session is indeterminate after a broker deadline; "
                    "new orders are blocked pending reconciliation."
                )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Zerodha {operation} deadline expired before HTTP submission.")
        future = self._http_executor.submit(call)
        try:
            return future.result(timeout=remaining)
        except FuturesTimeoutError as exc:
            # Python aliases concurrent.futures.TimeoutError to built-in
            # TimeoutError.  If the native function itself raised it, preserve
            # that evidence rather than falsely poisoning a completed future.
            if future.done():
                return future.result()
            self._http_poisoned = True
            self._timed_out_future = future
            future.cancel()
            raise TimeoutError(
                f"Zerodha {operation} exceeded the total {budget_seconds:.0f}s broker "
                "deadline; session state is indeterminate."
            ) from exc

    def recover_after_reconciliation(self) -> bool:
        """Explicitly clear deadline poison after the abandoned call has ended.

        "Poisoned" means an earlier HTTP call outlived its budget and was
        abandoned -- but Python cannot kill a running thread, so that call may
        STILL be dribbling a response (or completing an order) inside
        ``requests``.  Only after (a) reconciliation has proven the account
        state and (b) the abandoned call has actually finished is it safe to
        accept new orders; this method refuses (returns False) until both hold.
        """

        with self._lock:
            if self._timed_out_future is not None and not self._timed_out_future.done():
                return False
            self._http_poisoned = False
            self._timed_out_future = None
            return True

    def _call_api(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        is_order: bool = False,
        allow_after_poison: bool = False,
        authenticated: bool = True,
        budget_seconds: float | None = None,
    ) -> Any:
        """Send one Kite request under the limiters, the lock and a total deadline.

        Args:
            operation: Short label used only in error text.
            method / path: The documented HTTP method and path, e.g. ``GET /orders``.
            params / data: Query parameters / form fields.
            is_order: Also consume the stricter order budget (place, cancel).
            allow_after_poison: Permit reconciliation and risk-reducing calls
                once the abandoned timed-out call has actually ended.
            authenticated: False only for the login exchange, which has no token yet.
            budget_seconds: Total wall-clock budget; defaults to ten seconds,
                read at CALL time so a test can shorten it.

        Returns:
            The ``requests`` response; :meth:`_classify` decides what it means.

        Raises:
            RuntimeError: No token, no rate budget, or the session is poisoned.
            TimeoutError: The total budget expired (this poisons the session).
            requests.RequestException: A transport failure.
        """

        if budget_seconds is None:
            budget_seconds = _BROKER_CALL_DEADLINE_SECONDS
        if authenticated and not self._access_token:
            raise RuntimeError("Zerodha has no validated access token.")
        deadline = time.monotonic() + budget_seconds
        self._api_limiter.acquire(deadline)
        if is_order:
            self._order_limiter.acquire(deadline)

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(
                "Zerodha broker deadline expired while waiting for the shared session lock."
            )
        try:
            session = self._ensure_session_locked()
            headers = {_VERSION_HEADER[0]: _VERSION_HEADER[1]}
            if authenticated:
                headers["Authorization"] = f"token {self._api_key}:{self._access_token}"
            url = f"{_BASE_URL}{path}"
            # Keep the native inactivity timeout; the surrounding future enforces
            # the (possibly smaller) remaining total wall-clock budget.
            native_timeout = max(_API_TIMEOUT_SECONDS, budget_seconds)

            def send() -> Any:
                return session.request(
                    method, url, params=params, data=data, headers=headers, timeout=native_timeout
                )

            return self._run_http_with_deadline(
                operation,
                send,
                deadline=deadline,
                budget_seconds=budget_seconds,
                allow_after_poison=allow_after_poison,
            )
        finally:
            self._lock.release()

    def _classify(self, response: Any) -> tuple[str, Any, str]:
        """Split a response into ``("ok" | "refused" | "indeterminate", data, reason)``.

        Only a parsed ``status: error`` body with a refusal ``error_type`` AND a
        4xx status counts as "refused".  Everything else that is not a clean
        success -- a 5xx, a 429, a non-JSON gateway page -- is indeterminate.
        """

        status_code = int(getattr(response, "status_code", 0) or 0)
        try:
            body = response.json()
        except Exception:
            return ("indeterminate", None, f"HTTP {status_code} with a non-JSON body")
        if not isinstance(body, dict):
            return ("indeterminate", None, f"HTTP {status_code} with a malformed body")
        status = str(body.get("status") or "").strip().lower()
        if status == "success" and 200 <= status_code < 300:
            return ("ok", body.get("data"), "")
        message = redact_text(str(body.get("message") or "").strip(), self._secrets())
        error_type = str(body.get("error_type") or "").strip().upper()
        reason = f"{error_type or 'error'} (HTTP {status_code}): {message}".strip()
        # 429 is Kite's rate-limit answer: the request was not processed, but a
        # throttled ORDER call must still be proven through the order book.
        refused = (
            status == "error"
            and 400 <= status_code < 500
            and status_code != 429
            and error_type in _REFUSAL_ERROR_TYPES
        )
        return ("refused" if refused else "indeterminate", body.get("data"), reason)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def ensure_logged_in(self) -> bool:
        """Return True only when a validated Kite access token is available.

        Safe to call before every operation.  The fast path only checks
        in-memory state; the login work runs when no session is established.
        """

        with self._lock:
            if self.is_logged_in and self._access_token and self.client is not None:
                return True
            return self._login_locked()

    def _login_locked(self) -> bool:
        """Validate the ``.env`` token, or run Kite's browser request-token flow.

        Authentication order is deliberate:

        1. Require the API key used by every later ``Authorization`` header.
        2. Prefer today's stored token, but prove it with ``/user/profile``.
        3. If it is absent or expired, open Kite's login page, ask the operator
           to paste the redirect URL, and exchange its ``request_token``.

        Returns ``False`` instead of raising so startup can force every strategy
        back to paper mode.  Keys, secrets, request tokens and access tokens are
        never logged.
        """

        self.is_logged_in = False
        self._api_key = _env_str("ZERODHA_API_KEY")
        if not self._api_key:
            log.error("Zerodha login disabled: ZERODHA_API_KEY is blank.")
            return False
        self._ensure_session_locked()

        supplied_token = _env_str("ZERODHA_ACCESS_TOKEN")
        if supplied_token:
            self._access_token = supplied_token
            if self._validate_session_locked():
                self.is_logged_in = True
                log.info("Zerodha supplied access token validated successfully.")
                return True
            log.warning("ZERODHA_ACCESS_TOKEN was rejected or expired.")
            self._access_token = ""

        return self._interactive_login_locked()

    def _interactive_login_locked(self) -> bool:
        """Run Kite's browser login and exchange the request token; main thread only."""

        api_secret = _env_str("ZERODHA_API_SECRET")
        if not api_secret:
            log.error(
                "Zerodha login disabled: no valid ZERODHA_ACCESS_TOKEN, and "
                "ZERODHA_API_SECRET is blank so a new one cannot be requested."
            )
            return False
        if threading.current_thread() is not threading.main_thread():
            # A strategy or monitor thread must never sit on a hidden prompt
            # while holding the session lock.
            log.error(
                "Zerodha session is not valid and the browser login only runs on the "
                "main thread. Run 'python algo.py setup-broker-token --broker zerodha' "
                "and restart."
            )
            return False

        # The login URL carries the API key, so it goes straight to the browser
        # and is never logged: logs and console output are routinely shared.
        try:
            opened = bool(webbrowser.open(_LOGIN_URL.format(api_key=self._api_key)))
        except Exception:
            opened = False
        if opened:
            log.warning("Opening Kite login. Log in, then paste the URL you are redirected to.")
        else:
            log.warning(
                "Could not open a browser. Open %s yourself (with your ZERODHA_API_KEY in "
                "place of <key>), log in, then paste the URL you are redirected to.",
                _LOGIN_URL.format(api_key="<key>"),
            )
        try:
            pasted = input("Kite redirect URL (or request_token): ")
        except (EOFError, OSError):
            pasted = ""
        request_token = extract_request_token(pasted)
        if not request_token:
            log.error("Zerodha login aborted: no request_token supplied.")
            return False

        secrets = (self._api_key, api_secret, request_token)
        try:
            response = self._call_api(
                "token exchange",
                "POST",
                "/session/token",
                data={
                    "api_key": self._api_key,
                    "request_token": request_token,
                    "checksum": kite_checksum(self._api_key, request_token, api_secret),
                },
                authenticated=False,
            )
        except Exception as exc:
            log.error("Zerodha token exchange failed: %s", redact_text(exc, secrets))
            return False
        outcome, data, reason = self._classify(response)
        token = str(data.get("access_token") or "").strip() if isinstance(data, dict) else ""
        if outcome != "ok" or not token:
            # A silent refusal with a fresh request token usually means this
            # machine's public IP is not the static IP registered for the key.
            log.error(
                "Zerodha token exchange rejected: %s. Check that the request_token is "
                "fresh (it is single-use and short-lived) and that this machine's IP "
                "is registered for the API key.",
                redact_text(reason or "no access token returned", secrets),
            )
            return False

        self._access_token = token
        if not self._validate_session_locked():
            self._access_token = ""
            log.error("Zerodha issued a token, but /user/profile validation failed.")
            return False
        self.is_logged_in = True
        log.info("Zerodha execution client login successful.")
        return True

    def _validate_session_locked(self) -> bool:
        """Prove the current token with the cheapest authenticated read."""

        try:
            response = self._call_api("session validation", "GET", "/user/profile")
        except Exception as exc:
            log.error("Zerodha session validation failed: %s", redact_text(exc, self._secrets()))
            return False
        outcome, _data, reason = self._classify(response)
        if outcome != "ok":
            log.error("Zerodha session validation rejected (%s): %s", outcome, reason)
            return False
        return True

    def current_access_token(self) -> str:
        """Return the validated in-memory token, or "" when not logged in.

        Exists for ``setup-broker-token``, which stores a freshly minted token
        in ``.env`` so a restart later in the day does not need the browser.
        """

        with self._lock:
            return self._access_token if self.is_logged_in else ""

    def logout(self) -> dict[str, Any]:
        """Close the local session and erase sensitive in-memory state.

        Kite's remote logout would kill the token for the rest of the day (and
        the position-group monitor may share it), so this is local only: the
        connection pool is closed and the token, contract cache and rate
        histories are cleared.
        """

        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception as exc:
                    log.warning("Zerodha local session close failed: %s", exc)
            self.client = None
            self.is_logged_in = False
            self._access_token = ""
            self._instruments.clear()
            self._resolver.clear()
            self._api_limiter.reset()
            self._order_limiter.reset()
            return {"status": "success", "message": "Zerodha local session cleared"}

    # ------------------------------------------------------------------
    # Contract master
    # ------------------------------------------------------------------
    def preload_scrip_master(self) -> bool:
        """Log in and cache Kite's F&O contract lists.

        The master calls this once before worker threads start, so the first
        real order does not pay for a multi-megabyte download.  ``NFO`` (NIFTY,
        BANKNIFTY) is the list live strategies need and decides the result;
        ``BFO`` (SENSEX) only matters to the position-group monitor and is
        fetched best-effort.
        """

        if not self.ensure_logged_in():
            return False
        with self._lock:
            nfo_ready = self._ensure_exchange_master_locked("NFO")
            self._ensure_exchange_master_locked("BFO")
            return nfo_ready

    def _ensure_exchange_master_locked(self, exchange: str) -> bool:
        """Download and index one exchange's dump once; caller holds ``_lock``."""

        if exchange in self._instruments:
            return True
        try:
            response = self._call_api(
                "instrument master",
                "GET",
                f"/instruments/{exchange}",
                budget_seconds=_SCRIP_MASTER_TIMEOUT_SECONDS,
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code != 200:
                raise RuntimeError(f"HTTP {status_code}")
            rows = parse_instrument_csv(_csv_text(response))
        except Exception as exc:
            log.error(
                "Zerodha %s instrument master download/parse failed: %s",
                exchange,
                redact_text(exc, self._secrets()),
            )
            return False
        self._instruments[exchange] = {row.symbol: row for row in rows}
        for row in rows:
            key = (row.exchange, row.underlying, row.expiry, row.right, round(row.strike * 100))
            self._resolver[key] = row.symbol
        log.info("Zerodha %s instrument master loaded (%d option rows).", exchange, len(rows))
        return True

    def option_rows(self, exchange_segment: str = "NFO") -> list[InstrumentRow]:
        """Return the cached option contracts of one exchange (empty if not loaded).

        Used by the read-only diagnostic to pick a contract straight from the
        broker's own list and cross-check the resolver against it.
        """

        with self._lock:
            return list(self._instruments.get(str(exchange_segment).upper().strip(), {}).values())

    def lot_size_for_symbol(self, trading_symbol: str) -> int:
        """Return the cached lot size for a contract, or 0 when it is unknown."""

        with self._lock:
            for rows in self._instruments.values():
                row = rows.get(str(trading_symbol).strip())
                if row is not None:
                    return row.lot_size
        return 0

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry: Any,
        option_type: str,
        strike: Any,
        exchange_segment: str = "NFO",
    ) -> str:
        """Return the exact Kite ``tradingsymbol`` or ``""`` on a miss.

        Args:
            underlying: Index name, e.g. ``NIFTY`` or ``BANKNIFTY``.
            expiry: Date-like option expiry supplied by the Dhan contract lookup.
            option_type: ``CE`` for calls or ``PE`` for puts.
            strike: Human-scale strike such as 24500 (not a paise multiplier).
            exchange_segment: ``NFO`` (default) or ``BFO``.

        Resolution is exact rather than pattern-based: the requested expiry,
        strike and option type must all occur in Kite's downloaded list.
        Returning an empty string makes the caller skip the real order safely.
        """

        underlying_u = str(underlying).upper().strip()
        option_u = str(option_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NFO"
        if option_u not in {"CE", "PE"} or exchange_u not in _OPTION_EXCHANGES:
            log.warning(
                "Zerodha resolve skipped: unsupported option/exchange %r/%r.",
                option_type,
                exchange_segment,
            )
            return ""
        expiry_date = _as_date(expiry)
        try:
            strike_paise = round(float(strike) * 100)
        except (TypeError, ValueError):
            strike_paise = None
        if expiry_date is None or strike_paise is None:
            log.warning("Zerodha resolve skipped: invalid expiry/strike %r/%r.", expiry, strike)
            return ""
        if not self.ensure_logged_in():
            return ""
        with self._lock:
            if not self._ensure_exchange_master_locked(exchange_u):
                return ""
            symbol = self._resolver.get(
                (exchange_u, underlying_u, expiry_date, option_u, strike_paise), ""
            )
        if not symbol:
            log.warning(
                "Zerodha symbol NOT resolved: %s/%s/%s/strike %s on %s.",
                underlying_u,
                option_u,
                expiry_date,
                strike,
                exchange_u,
            )
        return symbol

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str = "NFO",
        product_type: str = "INTRADAY",
        *,
        order_tag: str = "",
    ) -> OrderResult:
        """Serialize one order from submission through typed fill confirmation.

        Raises:
            ValueError: A caller supplied an unsupported order value.  Nothing
                has been sent when this is raised.
        """

        side_u = str(side).upper().strip()
        product_u = str(product_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NFO"
        symbol_s = str(symbol).strip()
        quantity_i = exact_int(quantity)
        if side_u not in _SIDES:
            raise ValueError(f"Invalid side: {side!r}")
        if product_u not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if exchange_u not in _OPTION_EXCHANGES:
            raise ValueError(f"Invalid exchange_segment: {exchange_segment!r}")
        if quantity_i is None or quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not symbol_s:
            raise ValueError("Missing trading symbol for order.")

        if not self._order_submission_lock.acquire(timeout=_BROKER_CALL_DEADLINE_SECONDS):
            return self._unproven(
                quantity_i,
                "ORDER_GATE_TIMEOUT",
                "Zerodha order was not submitted because another order's outcome "
                "remained unresolved past the broker deadline.",
            )
        try:
            if self._http_poisoned:
                return self._unproven(
                    quantity_i,
                    "SESSION_POISONED",
                    "Zerodha session is indeterminate after an earlier deadline; "
                    "reconcile before submitting another order.",
                )
            if not self.ensure_logged_in():
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason="Order was not submitted because Zerodha login failed.",
                )
            return self._place_market_order_serialized(
                symbol_s, side_u, quantity_i, exchange_u, _PRODUCT_MAP[product_u], order_tag
            )
        finally:
            self._order_submission_lock.release()

    def _place_market_order_serialized(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange: str,
        product: str,
        order_tag: str,
    ) -> OrderResult:
        """Place one MARKET order and return an explicit normalized outcome.

        ``status: success`` only means Kite accepted the request.  The method
        still reads the order's history before reporting a fill.
        """

        form = {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": side,
            "order_type": "MARKET",
            "quantity": str(quantity),
            "product": product,
            "validity": "DAY",
            # Exchanges reject API MARKET orders without market protection;
            # -1 asks Kite to apply its automatic protection band.
            "market_protection": "-1",
        }
        tag = _broker_tag(order_tag)
        if tag:
            form["tag"] = tag
        try:
            response = self._call_api(
                "place order", "POST", "/orders/regular", data=form, is_order=True
            )
        except Exception as exc:
            return self._resolve_by_tag(
                quantity,
                tag,
                f"Zerodha place-order call is indeterminate: {redact_text(exc, self._secrets())}",
                refused=False,
            )
        outcome, data, reason = self._classify(response)
        order_id = str(data.get("order_id") or "").strip() if isinstance(data, dict) else ""
        if outcome == "ok" and order_id:
            return self._confirm_fill(order_id, quantity)
        return self._resolve_by_tag(
            quantity,
            tag,
            reason or "Zerodha acknowledged the order without an order id.",
            refused=outcome == "refused",
        )

    def _resolve_by_tag(self, quantity: int, tag: str, reason: str, *, refused: bool) -> OrderResult:
        """Search the order book for ``tag`` before concluding anything.

        A refusal becomes ``REJECTED`` only when the search succeeded and found
        no such order; every other path stays ``UNKNOWN``.
        """

        found_id, searched = self._find_order_id_by_tag(tag)
        if found_id:
            return self._confirm_fill(found_id, quantity)
        if refused and searched:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity,
                filled_quantity=0,
                broker_state="REJECTED",
                reason=f"Zerodha rejected the order: {reason}",
            )
        return self._unproven(quantity, "", reason)

    def _find_order_id_by_tag(self, tag: str) -> tuple[str, bool]:
        """Return ``(order_id, search_succeeded)`` for an order carrying ``tag``."""

        if not tag:
            return ("", False)
        try:
            response = self._call_api("order book", "GET", "/orders", allow_after_poison=True)
        except Exception as exc:
            log.warning(
                "Zerodha order-book lookup failed: %s", redact_text(exc, self._secrets())
            )
            return ("", False)
        outcome, data, _reason = self._classify(response)
        if outcome != "ok" or not isinstance(data, list):
            return ("", False)
        for row in data:
            if not isinstance(row, dict):
                return ("", False)
            # Kite reports ``tag`` (text) and, on some rows, ``tags`` (a list).
            raw_tags: list[Any] = []
            for value in (row.get("tag"), row.get("tags")):
                raw_tags.extend(value if isinstance(value, list) else [value])
            if tag in {_broker_tag(value) for value in raw_tags}:
                return (str(row.get("order_id") or "").strip(), True)
        return ("", True)

    @staticmethod
    def _unproven(quantity: int, broker_state: str, reason: str) -> OrderResult:
        """Build an ``UNKNOWN`` result: nothing about the order could be proven."""

        return normalize_order_result(
            order_id="",
            requested_quantity=quantity,
            filled_quantity=0,
            broker_state=broker_state,
            reason=reason,
        )

    def get_order_status(self, order_id: str, requested_quantity: int = 0) -> OrderResult:
        """Return one normalized order snapshot.

        Exceptions and malformed quantities are represented as ``UNKNOWN`` while
        retaining the caller's order id.  They never become zero-fill rejection.
        """

        requested = exact_int(requested_quantity)
        if requested is None or requested < 0:
            requested = 0
        order_id_s = str(order_id or "").strip()

        def failed(detail: str) -> OrderResult:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"Zerodha order-status query failed: {detail}",
            )

        if not order_id_s:
            return failed("no order id")
        try:
            response = self._call_api(
                "order status", "GET", f"/orders/{order_id_s}", allow_after_poison=True
            )
        except Exception as exc:
            return failed(redact_text(exc, self._secrets()))
        outcome, data, reason = self._classify(response)
        rows = data if isinstance(data, list) else [data]
        # Kite returns the order's whole state history, oldest first: the LAST
        # entry is the order as it stands now.
        latest = next((row for row in reversed(rows) if isinstance(row, dict)), None)
        if outcome != "ok" or latest is None:
            return failed(reason or "no order row")

        state = str(latest.get("status") or "").strip().upper()
        broker_qty = exact_int(latest.get("quantity"))
        status_message = redact_text(
            str(latest.get("status_message") or "").strip(), self._secrets()
        )
        if requested and broker_qty not in {None, requested}:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=None,
                broker_state=state,
                reason=(
                    f"Malformed Zerodha status: broker quantity {broker_qty} does not "
                    f"match requested quantity {requested}. {status_message}"
                ),
            )
        return normalize_order_result(
            order_id=order_id_s,
            requested_quantity=requested or broker_qty or 0,
            filled_quantity=latest.get("filled_quantity"),
            broker_state=state,
            reason=status_message,
            average_fill_price=_positive_price(latest.get("average_price")),
        )

    def _confirm_fill(self, order_id: str, want_qty: int) -> OrderResult:
        """Poll until the order reaches a truly terminal state, or time out.

        A market order is often observed mid-fill (state ``OPEN`` with some
        quantity already traded).  That is TRANSIENT -- the next poll usually
        shows ``COMPLETE`` -- so a partial/unknown snapshot becomes the real
        outcome only when the broker label is terminal or the timeout expires.
        """

        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_result = normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=0,
            broker_state="",
            reason="Zerodha acknowledgement received; fill status not read yet.",
        )
        while time.monotonic() < deadline:
            last_result = self.get_order_status(order_id, requested_quantity=want_qty)
            if last_result.status in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                return last_result
            if last_result.broker_state in TERMINAL_BROKER_STATES:
                # Terminal label with a partial/contradictory quantity snapshot:
                # no further fills are possible, so this IS the final outcome.
                return last_result
            # A failed status call is already indeterminate.  Repeating it inside
            # the same order interaction risks exceeding the deadline without
            # adding evidence, so return immediately with the known id.
            if "query failed" in last_result.reason.lower():
                return last_result
            time.sleep(_FILL_POLL_INTERVAL)
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=last_result.filled_quantity,
            broker_state=last_result.broker_state,
            reason=(
                f"Zerodha order {order_id} was not terminal within "
                f"{_FILL_TIMEOUT_SECONDS:.0f}s; exposure is indeterminate."
            ),
            average_fill_price=last_result.average_fill_price,
        )

    def cancel_order(self, order_id: str, requested_quantity: int = 0) -> OrderResult:
        """Request cancellation, then normalize the order's resulting status."""

        order_id_s = str(order_id).strip()
        requested = exact_int(requested_quantity)
        if not order_id_s:
            raise ValueError("Missing order id for cancellation.")
        if requested is None or requested < 0:
            raise ValueError(f"Invalid requested_quantity: {requested_quantity!r}")

        def unsent(detail: str) -> OrderResult:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=detail,
            )

        if not self.ensure_logged_in():
            return unsent("Zerodha cancellation was not submitted because login failed.")
        try:
            self._call_api(
                "cancel order",
                "DELETE",
                f"/orders/regular/{order_id_s}",
                is_order=True,
                allow_after_poison=True,
            )
        except Exception as exc:
            return unsent(
                "Zerodha cancellation response is indeterminate: "
                f"{redact_text(exc, self._secrets())}"
            )
        # Whatever the cancel call answered, the order's own history is the
        # only evidence of what actually happened to it.
        return self.get_order_status(order_id_s, requested_quantity=requested)

    # ------------------------------------------------------------------
    # Reconciliation queries
    # ------------------------------------------------------------------
    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        """Return orders that can still fill, or an explicit indeterminate result."""

        if not self.ensure_logged_in():
            return BrokerQueryResult.indeterminate("Zerodha login failed.")
        try:
            response = self._call_api("order book", "GET", "/orders", allow_after_poison=True)
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Zerodha open-order query failed: {redact_text(exc, self._secrets())}"
            )
        outcome, data, reason = self._classify(response)
        if outcome != "ok":
            return BrokerQueryResult.indeterminate(
                f"Zerodha open-order query returned {outcome}: {reason}"
            )
        if not isinstance(data, list):
            return BrokerQueryResult.indeterminate(
                "Zerodha open-order query returned malformed data."
            )

        orders: list[OpenOrder] = []
        for row in data:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Zerodha open-order query contained a malformed row."
                )
            raw_state = str(row.get("status") or "").strip().upper()
            snapshot = normalize_order_result(
                order_id=row.get("order_id"),
                requested_quantity=row.get("quantity"),
                filled_quantity=row.get("filled_quantity"),
                broker_state=raw_state,
                reason="Open-order snapshot",
            )
            symbol = str(row.get("tradingsymbol") or "").strip()
            if (
                not snapshot.order_id
                or not symbol
                or "malformed" in snapshot.reason.lower()
                or snapshot.requested_quantity <= 0
            ):
                return BrokerQueryResult.indeterminate(
                    "Zerodha open-order query contained incomplete order data."
                )
            if raw_state in TERMINAL_BROKER_STATES or snapshot.remaining_quantity <= 0:
                continue
            orders.append(
                OpenOrder(
                    order_id=snapshot.order_id,
                    symbol=symbol,
                    side=str(row.get("transaction_type") or "").strip().upper(),
                    requested_quantity=snapshot.requested_quantity,
                    filled_quantity=snapshot.filled_quantity,
                    remaining_quantity=snapshot.remaining_quantity,
                    broker_state=snapshot.broker_state,
                )
            )
        return BrokerQueryResult.success(orders)

    def _net_position_rows(self) -> tuple[list[dict[str, Any]] | None, str]:
        """Read ``/portfolio/positions`` and return ``(net rows, "")`` or ``(None, why)``."""

        if not self.ensure_logged_in():
            return (None, "Zerodha login failed.")
        try:
            response = self._call_api(
                "positions", "GET", "/portfolio/positions", allow_after_poison=True
            )
        except Exception as exc:
            return (None, f"Zerodha position query failed: {redact_text(exc, self._secrets())}")
        outcome, data, reason = self._classify(response)
        if outcome != "ok":
            return (None, f"Zerodha position query returned {outcome}: {reason}")
        rows = data.get("net") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return (None, "Zerodha position query returned no 'net' list.")
        if any(not isinstance(row, dict) for row in rows):
            return (None, "Zerodha position query contained a malformed row.")
        return (rows, "")

    def list_open_positions(self) -> BrokerQueryResult[OpenPosition]:
        """Return non-flat net positions, or an explicit indeterminate result."""

        rows, problem = self._net_position_rows()
        if rows is None:
            return BrokerQueryResult.indeterminate(problem)
        positions: list[OpenPosition] = []
        for row in rows:
            quantity = exact_int(row.get("quantity"))
            symbol = str(row.get("tradingsymbol") or "").strip()
            if quantity is None or not symbol:
                return BrokerQueryResult.indeterminate(
                    "Zerodha position query contained incomplete position data."
                )
            if quantity == 0:
                continue
            positions.append(
                OpenPosition(
                    symbol=symbol,
                    quantity=quantity,
                    product_type=str(row.get("product") or "").strip().upper(),
                    broker_state="OPEN",
                )
            )
        return BrokerQueryResult.success(positions)

    def list_position_details(self) -> BrokerQueryResult[PositionDetail]:
        """Return non-flat positions with average price and contract identity.

        The one method beyond the shared broker surface: the position-group
        monitor needs a price and the option's (underlying, expiry, strike,
        right), which ``OpenPosition`` deliberately does not carry.

        Identity comes from the ALREADY-CACHED contract master only.  This is
        called at the moment a group reaches its target, so it must never start
        a multi-megabyte download; call ``preload_scrip_master()`` beforehand.
        A row the cache does not know comes back with empty identity fields.
        """

        rows, problem = self._net_position_rows()
        if rows is None:
            return BrokerQueryResult.indeterminate(problem)
        positions: list[PositionDetail] = []
        for row in rows:
            quantity = exact_int(row.get("quantity"))
            symbol = str(row.get("tradingsymbol") or "").strip()
            exchange = str(row.get("exchange") or "").strip().upper()
            if quantity is None or not symbol or not exchange:
                return BrokerQueryResult.indeterminate(
                    "Zerodha position query contained incomplete position data."
                )
            if quantity == 0:
                continue
            with self._lock:
                known = self._instruments.get(exchange, {}).get(symbol)
            positions.append(
                PositionDetail(
                    symbol=symbol,
                    exchange=exchange,
                    product_type=str(row.get("product") or "").strip().upper(),
                    quantity=quantity,
                    average_price=_positive_price(row.get("average_price")),
                    underlying=known.underlying if known else "",
                    expiry=known.expiry if known else None,
                    strike=known.strike if known else None,
                    right=known.right if known else "",
                )
            )
        return BrokerQueryResult.success(positions)

    def extract_order_id(self, order_response: Any) -> str:
        """Recursively find Kite's ``order_id`` in a response payload.

        Only nested containers are searched: Kite wraps the id as
        ``{"status": "success", "data": {"order_id": ...}}``, so treating every
        text value as an id would return the word "success".  A bare string is
        accepted as an id only when it IS the whole response.
        """

        if isinstance(order_response, OrderResult):
            return order_response.order_id
        if order_response is None:
            return ""
        if isinstance(order_response, dict):
            for key in ("order_id", "orderId", "id"):
                value = order_response.get(key)
                if value:
                    return str(value).strip()
            nested = [v for v in order_response.values() if isinstance(v, (dict, list))]
            return next((found for v in nested if (found := self.extract_order_id(v))), "")
        if isinstance(order_response, list):
            nested = [v for v in order_response if isinstance(v, (dict, list))]
            return next((found for v in nested if (found := self.extract_order_id(v))), "")
        if isinstance(order_response, str):
            return order_response.strip()
        return ""


# The master imports and reuses this one object so all worker threads (and the
# position-group monitor) share the same authenticated session, contract cache,
# lock, order gate and rate-limit budgets.
zerodha_execution_client = ZerodhaExecutionClient()
