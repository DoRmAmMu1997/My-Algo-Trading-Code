"""
Thread-safe Flattrade Pi v2 execution client for the multi-strategy runner.

The master runner deliberately knows almost nothing about broker APIs.  It asks
each broker helper to log in, preload a contract master, resolve one NIFTY option
and place a confirmed market order.  This module implements that same small
surface for Flattrade using only its documented REST endpoints and ``requests``.

Safety rules followed here:

* Importing this module never logs in, opens a browser, or sends an order.
* A supplied daily access token is verified through ``UserDetails`` before use.
* Without a token, the operator completes Flattrade's browser authorization and
  pastes the short-lived request code; the API secret never leaves this process.
* Every HTTP call retains a native timeout; one total ten-second deadline also
  covers rate-limit wait, session-lock wait, and slow-dribble response bodies.
* ``place_market_order`` returns a typed result after ``SingleOrdHist`` checks
  the fill. Rejection, partial fill, and response loss remain distinct; ambiguity
  is ``UNKNOWN`` and poisons new submissions until explicit reconciliation.

Flattrade Pi v2 currently documents no logout endpoint.  ``logout()`` therefore
closes the local HTTP session and erases the in-memory token.

Beginner's mental model — one live order travels through this file as follows:

1. ``ensure_logged_in()`` obtains or validates today's access token.
2. ``preload_scrip_master()`` downloads Flattrade's list of real NFO contracts.
3. ``resolve_option_symbol()`` converts strategy details such as
   NIFTY/14-JUL-2026/CE/24150 into Flattrade's exact trading symbol.
4. ``place_market_order()`` translates friendly values such as BUY and INTRADAY
   into Flattrade's short API codes, then submits the order.
5. ``_confirm_fill()`` checks order history; an acknowledgement alone is never
   treated as proof that money changed hands.

Small glossary:

* ``jData`` is the JSON request object Flattrade expects inside the POST body.
* ``jKey`` is today's access token. It is sensitive and is never logged here.
* A *scrip master* is the broker's contract catalogue: symbol, expiry, strike,
  option type, token, and lot size for every listed instrument.
* ``norenordno`` is Flattrade's order number, used to query fill status later.
* *Market protection* limits how far a market order may chase a rapidly moving
  price; the default value comes from Flattrade's official Postman example.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

# The broker helpers live in folders with spaces and are also executable as
# standalone diagnostics.  Add their shared ``Dependencies`` parent explicitly
# so the broker-neutral contract imports the same way in both entry points.
_DEPENDENCIES_DIR = Path(__file__).resolve().parent.parent
if str(_DEPENDENCIES_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPENDENCIES_DIR))

from broker_contract import (  # noqa: E402
    BrokerQueryResult,
    OpenOrder,
    OpenPosition,
    OrderResult,
    OrderStatus,
    normalize_order_result,
)

try:
    from dotenv import load_dotenv

    load_dotenv(
        dotenv_path=Path(__file__).resolve().parent.parent / ".env",
        override=False,
    )
except Exception:
    # ``python-dotenv`` is a core dependency, but keeping import-time behaviour
    # harmless means paper trading can still start and report a clear login error.
    pass


log = logging.getLogger(__name__)

_BASE_URL = "https://piconnect.flattrade.in/PiConnectAPI"
_AUTH_URL = "https://auth.flattrade.in/?app_key={api_key}"
_TOKEN_URL = "https://authapi.flattrade.in/trade/apitoken"
_NFO_INDEX_MASTER_URL = (
    "https://flattrade.s3.ap-south-1.amazonaws.com/"
    "scripmaster/Nfo_Index_Derivatives.csv"
)

_API_TIMEOUT_SECONDS = 10.0
_SCRIP_MASTER_TIMEOUT_SECONDS = 10.0
_BROKER_CALL_DEADLINE_SECONDS = 10.0
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 2.0

_PRODUCT_MAP = {"INTRADAY": "I", "NORMAL": "M"}
_SIDE_MAP = {"BUY": "B", "SELL": "S"}
_TERMINAL_FAILURE_STATES = {"rejected", "cancelled", "canceled"}


def _exact_int(value: Any) -> int | None:
    """Parse a broker quantity only when it is a finite whole number."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _is_no_data_envelope(value: Any) -> bool:
    """Recognize only Flattrade's exact, determinate empty-book response."""

    if not isinstance(value, dict):
        return False
    state = str(value.get("stat") or "").strip().lower().replace("_", "")
    message = " ".join(
        str(value.get("emsg") or value.get("msg") or "").strip().lower().split()
    )
    return state in {"notok", "rejected"} and message == "no data"


def _env_str(name: str, default: str = "") -> str:
    """Read one environment value and remove accidental wrapping quotes."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value or default


def _env_non_negative_int(name: str, default: int) -> int:
    """Read a non-negative integer setting, falling back safely on bad input."""
    raw = _env_str(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r; using %s.", name, raw, default)
        return default
    if value < 0:
        log.warning("Invalid %s=%r; using %s.", name, raw, default)
        return default
    return value


class _RollingWindowRateLimiter:
    """Enforce per-second and per-minute request budgets across worker threads.

    Short bursts wait for a slot.  A wait longer than ``max_wait_seconds`` raises
    before the HTTP request is sent; a stale live order is more dangerous than a
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

        The deque stores only timestamps from the last minute. We separately
        count its last one-second slice because Flattrade publishes both limits.
        A short wait is acceptable; a long wait would make a trading decision
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
                f"Flattrade {self.label} deadline expired waiting for the limiter lock"
            )
        try:
            while True:
                now = self._clock()
                # Discard calls that are older than the longest (60-second)
                # window. Keeping the deque small also makes each check cheap.
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
                        f"Flattrade {self.label} rate limit exhausted; "
                        f"safe slot needs {wait_seconds:.2f}s"
                    )
                self._sleep(wait_seconds)
        finally:
            self._lock.release()

    def reset(self) -> None:
        """Forget local request history after a session is explicitly closed."""
        with self._lock:
            self._timestamps.clear()


class FlattradeExecutionClient:
    """One lazily authenticated, lock-guarded Flattrade execution session.

    The master creates many strategy threads but imports one singleton from the
    bottom of this module. All those threads therefore share one HTTP session,
    one token, one contract cache, and one pair of rate-limit counters. ``RLock``
    makes nested helper calls safe without allowing simultaneous session writes.
    """

    def __init__(self) -> None:
        self.client: requests.Session | None = None
        self.is_logged_in = False
        self._lock = threading.RLock()
        self._access_token = ""
        self._client_id = ""
        self._account_id = ""
        self._scrip_df: pd.DataFrame | None = None
        self._symbol_cache: dict[tuple, str] = {}
        # ``requests`` timeouts are inactivity limits, not a total wall-clock
        # budget: a slow-dribble response can otherwise run forever. Execute one
        # native HTTP call at a time and impose a caller-visible deadline around
        # rate-limit wait, session-lock wait, and the complete response body.
        self._http_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="flattrade-http",
        )
        self._http_poisoned = False
        self._timed_out_future: Future[Any] | None = None
        # Serialize acknowledgement plus SingleOrdHist confirmation. Without
        # this gate another strategy can submit before a status timeout poisons
        # the session, leaving two simultaneously indeterminate live orders.
        self._order_submission_lock = threading.Lock()
        # Flattrade documents 40/s + 200/min for general API calls and the
        # stricter 10/s + 40/min budget for order APIs.
        self._api_limiter = _RollingWindowRateLimiter(
            40,
            200,
            _MAX_RATE_LIMIT_WAIT_SECONDS,
            label="API",
        )
        self._order_limiter = _RollingWindowRateLimiter(
            10,
            40,
            _MAX_RATE_LIMIT_WAIT_SECONDS,
            label="order API",
        )

    def _ensure_session_locked(self) -> requests.Session:
        """Create the shared HTTP session on first use; caller holds ``_lock``."""
        if self.client is None:
            self.client = requests.Session()
        return self.client

    @staticmethod
    def _remaining_broker_budget(started: float) -> float:
        """Return seconds left in one complete broker-call wall-clock budget."""

        return _BROKER_CALL_DEADLINE_SECONDS - (time.monotonic() - started)

    def _run_http_with_deadline(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        started: float,
        allow_after_poison: bool = False,
    ) -> Any:
        """Run one native HTTP call without letting a slow response outlive 10s.

        The caller holds ``_lock``.  If the future times out, its socket may
        still be active inside ``requests``; poison the shared session so no
        second request can overlap that indeterminate in-flight operation.
        """

        if self._http_poisoned:
            abandoned_finished = (
                self._timed_out_future is not None
                and self._timed_out_future.done()
            )
            if not allow_after_poison or not abandoned_finished:
                raise RuntimeError(
                    "Flattrade session is indeterminate after a broker deadline; "
                    "new orders are blocked pending reconciliation."
                )
        remaining = self._remaining_broker_budget(started)
        if remaining <= 0:
            raise TimeoutError(f"Flattrade {operation} deadline expired before HTTP submission.")
        future = self._http_executor.submit(call)
        try:
            return future.result(timeout=remaining)
        except FuturesTimeoutError as exc:
            # Python aliases concurrent.futures.TimeoutError to built-in
            # TimeoutError. If the native function itself raised it, preserve
            # that evidence rather than falsely poisoning a completed future.
            if future.done():
                return future.result()
            self._http_poisoned = True
            self._timed_out_future = future
            future.cancel()
            raise TimeoutError(
                f"Flattrade {operation} exceeded the total "
                f"{_BROKER_CALL_DEADLINE_SECONDS:.0f}s broker deadline; "
                "session state is indeterminate."
            ) from exc

    def recover_after_reconciliation(self) -> bool:
        """Explicitly clear deadline poison after the abandoned call has ended."""

        with self._lock:
            if self._timed_out_future is not None and not self._timed_out_future.done():
                return False
            self._http_poisoned = False
            self._timed_out_future = None
            return True

    def ensure_logged_in(self) -> bool:
        """Return True only when a validated Flattrade daily token is available.

        This method is intentionally safe to call before every operation. The
        normal fast path only checks in-memory state; the interactive browser
        flow runs once when the session has not yet been established.
        """
        with self._lock:
            if self.is_logged_in and self._access_token and self.client is not None:
                return True
            return self._login_locked()

    def _login_locked(self) -> bool:
        """Validate an env token or perform Flattrade's browser request-code flow.

        Authentication order is deliberate:

        1. Require the client id used by all later ``jData`` payloads.
        2. Prefer a manually supplied daily token, but verify it with UserDetails.
        3. If absent/expired, open the official authorization page.
        4. Hash API key + request code + API secret exactly as Flattrade documents.
        5. Exchange that digest for a token and verify the resulting session.

        Returns ``False`` instead of raising so startup can force every strategy
        back to paper mode. Secrets, request codes, and tokens are never logged.
        """
        self.is_logged_in = False
        self._client_id = _env_str("FLATTRADE_CLIENT_ID")
        if not self._client_id:
            log.error("Flattrade login disabled: FLATTRADE_CLIENT_ID is blank.")
            return False

        self._ensure_session_locked()
        supplied_token = _env_str("FLATTRADE_ACCESS_TOKEN")
        if supplied_token:
            self._access_token = supplied_token
            if self._validate_session_locked():
                self.is_logged_in = True
                log.info("Flattrade supplied access token validated successfully.")
                return True
            log.warning(
                "FLATTRADE_ACCESS_TOKEN was rejected or expired; starting browser authorization."
            )
            self._access_token = ""

        api_key = _env_str("FLATTRADE_API_KEY")
        raw_secret = _env_str("FLATTRADE_API_SECRET")
        missing = [
            name
            for name, value in (
                ("FLATTRADE_API_KEY", api_key),
                ("FLATTRADE_API_SECRET", raw_secret),
            )
            if not value
        ]
        if missing:
            log.error(
                "Flattrade login disabled: missing %s.", ", ".join(missing)
            )
            return False

        authorization_url = _AUTH_URL.format(api_key=api_key)
        log.warning(
            "Opening Flattrade authorization. Complete login in the browser, then paste the request_code."
        )
        try:
            webbrowser.open(authorization_url)
        except Exception as exc:
            # A browser-launch problem is recoverable: the operator can open the
            # logged URL manually and still paste the request code.
            log.warning("Could not open the Flattrade browser automatically: %s", exc)
        try:
            request_code = input("Flattrade request_code: ").strip()
        except (EOFError, OSError):
            request_code = ""
        if not request_code:
            log.error("Flattrade login aborted: no request_code supplied.")
            return False

        digest = hashlib.sha256(
            f"{api_key}{request_code}{raw_secret}".encode()
        ).hexdigest()
        try:
            session = self._ensure_session_locked()
            started = time.monotonic()

            def exchange_token():
                response = session.post(
                    _TOKEN_URL,
                    json={
                        "api_key": api_key,
                        "request_code": request_code,
                        "api_secret": digest,
                    },
                    timeout=_API_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response.json()

            payload = self._run_http_with_deadline(
                "token exchange",
                exchange_token,
                started=started,
            )
        except Exception as exc:
            log.error("Flattrade token exchange failed: %s", exc)
            return False

        if not isinstance(payload, dict) or str(payload.get("status", "")).lower() != "ok":
            # Flattrade's apitoken endpoint often rejects with an EMPTY emsg, so
            # log every non-secret clue we have. Note this endpoint (uniquely)
            # uses "status", not the Noren-wide "stat". A silent rejection with
            # a valid request_code usually means the caller's public IP does not
            # match the STATIC IP registered with this API key (documented
            # requirement), or the API secret does not belong to this key.
            if isinstance(payload, dict):
                log.error(
                    "Flattrade token exchange rejected: emsg=%r status=%r "
                    "(token_present=%s, client=%r). If emsg is empty with a fresh "
                    "request_code, check that your CURRENT public IP matches the "
                    "static IP registered for this API key, and that "
                    "FLATTRADE_API_SECRET belongs to FLATTRADE_API_KEY.",
                    payload.get("emsg", ""),
                    payload.get("status", "<missing>"),
                    bool(str(payload.get("token") or "").strip()),
                    payload.get("client", ""),
                )
            else:
                log.error("Flattrade token exchange rejected: malformed response %r", type(payload).__name__)
            return False
        token = str(payload.get("token") or "").strip()
        returned_client = str(payload.get("client") or "").strip()
        if not token:
            log.error("Flattrade token exchange returned no access token.")
            return False
        if returned_client and returned_client.upper() != self._client_id.upper():
            log.error("Flattrade token belongs to a different client id; login rejected.")
            return False

        self._access_token = token
        if not self._validate_session_locked():
            self._access_token = ""
            log.error("Flattrade issued a token, but UserDetails validation failed.")
            return False
        self.is_logged_in = True
        log.info("Flattrade execution client login successful.")
        return True

    def _validate_session_locked(self) -> bool:
        """Validate the current token and remember Flattrade's account id."""
        try:
            payload = self._post_api("UserDetails", {"uid": self._client_id})
        except Exception as exc:
            log.error("Flattrade UserDetails validation failed: %s", exc)
            return False
        if not isinstance(payload, dict) or str(payload.get("stat", "")).lower() != "ok":
            message = payload.get("emsg", "invalid session") if isinstance(payload, dict) else "malformed response"
            log.error("Flattrade session validation rejected: %s", message)
            return False
        enabled_exchanges = {
            str(value).upper() for value in (payload.get("exarr") or [])
        }
        if enabled_exchanges and "NFO" not in enabled_exchanges:
            log.error("Flattrade account does not report NFO access; live trading disabled.")
            return False
        self._account_id = str(payload.get("actid") or self._client_id).strip()
        return bool(self._account_id)

    def _post_api(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        is_order: bool = False,
        allow_after_poison: bool = False,
    ) -> Any:
        """POST one documented ``jData``/``jKey`` request with limits and timeout.

        Args:
            endpoint: Final Pi endpoint name, such as ``UserDetails``.
            payload: Plain Python dictionary that becomes compact JSON in jData.
            is_order: Also consume the stricter order budget for write endpoints.
            allow_after_poison: Permit reconciliation/risk-reducing calls only
                after the abandoned timed-out HTTP future has actually ended.

        Returns:
            The decoded JSON object/list supplied by Flattrade.

        Raises:
            RuntimeError: No token, rate budget, or valid JSON is available.
            requests.RequestException: The network or HTTP status failed.

        ``urlencode`` matters even though jData itself is JSON: it protects symbols
        containing characters such as ``&`` from corrupting the outer POST body.
        """
        started = time.monotonic()
        if not self._access_token:
            raise RuntimeError("Flattrade has no validated access token.")
        deadline = started + _BROKER_CALL_DEADLINE_SECONDS
        self._api_limiter.acquire(deadline)
        if is_order:
            self._order_limiter.acquire(deadline)

        remaining = self._remaining_broker_budget(started)
        if remaining <= 0 or not self._lock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(
                "Flattrade broker deadline expired while waiting for the shared session lock."
            )
        try:
            session = self._ensure_session_locked()
            body = urlencode(
                {
                    "jData": json.dumps(payload, separators=(",", ":")),
                    "jKey": self._access_token,
                }
            )

            def send_and_decode() -> Any:
                response = session.post(
                    f"{_BASE_URL}/{endpoint}",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    # Retain the broker-native inactivity timeout unchanged;
                    # the surrounding future enforces the smaller remaining
                    # total wall-clock budget when lock/rate waits consumed time.
                    timeout=_API_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                try:
                    return response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"Flattrade {endpoint} returned malformed JSON"
                    ) from exc

            return self._run_http_with_deadline(
                endpoint,
                send_and_decode,
                started=started,
                allow_after_poison=allow_after_poison,
            )
        finally:
            self._lock.release()

    def preload_scrip_master(self) -> bool:
        """Log in and cache Flattrade's official NSE index-derivative CSV.

        The master calls this once before worker threads start. Returning ``False``
        makes startup disable live trading instead of discovering a missing or
        malformed contract catalogue during the first real order.
        """
        if not self.ensure_logged_in():
            return False
        with self._lock:
            return self._ensure_scrip_master_locked()

    def _ensure_scrip_master_locked(self) -> bool:
        """Download and normalize the contract master once; caller holds ``_lock``."""
        if self._scrip_df is not None:
            return not self._scrip_df.empty
        try:
            session = self._ensure_session_locked()
            started = time.monotonic()

            def download_scrip_master():
                response = session.get(
                    _NFO_INDEX_MASTER_URL,
                    timeout=_SCRIP_MASTER_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response.text

            csv_text = self._run_http_with_deadline(
                "scrip-master download",
                download_scrip_master,
                started=started,
            )
            raw = pd.read_csv(io.StringIO(csv_text))
            self._scrip_df = self._prepare_scrip_master(raw)
            log.info(
                "Flattrade NFO index scrip master loaded (%d rows).",
                len(self._scrip_df),
            )
            return True
        except Exception as exc:
            self._scrip_df = None
            log.error("Flattrade NFO scrip master download/parse failed: %s", exc)
            return False

    @staticmethod
    def _prepare_scrip_master(raw: pd.DataFrame) -> pd.DataFrame:
        """Validate official CSV columns and add normalized lookup fields.

        Human-facing CSV values vary in case and type (for example, strike may be
        text ``"24150.00"``). Columns prefixed with ``_`` are internal, normalized
        forms used for reliable comparisons; the original broker columns remain
        available to the diagnostic for display and lot-size inspection.

        Raises:
            ValueError: Required columns or usable NFO rows are missing.
        """
        required = {
            "Exchange",
            "Lotsize",
            "Symbol",
            "Tradingsymbol",
            "Expiry",
            "Strike",
            "Optiontype",
        }
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(
                "Flattrade scrip master missing columns: " + ", ".join(missing)
            )
        frame = raw.copy()
        frame["_exchange_u"] = frame["Exchange"].astype(str).str.strip().str.upper()
        frame["_symbol_u"] = frame["Symbol"].astype(str).str.strip().str.upper()
        frame["_trading_symbol"] = frame["Tradingsymbol"].astype(str).str.strip()
        frame["_option_u"] = frame["Optiontype"].astype(str).str.strip().str.upper()
        frame["_expiry_date"] = pd.to_datetime(
            frame["Expiry"], format="%d-%b-%Y", errors="coerce"
        ).dt.date
        frame["_strike_f"] = pd.to_numeric(frame["Strike"], errors="coerce")
        frame["_lotsize_i"] = pd.to_numeric(
            frame["Lotsize"], errors="coerce"
        ).fillna(0).astype(int)
        frame = frame[
            (frame["_exchange_u"] == "NFO")
            & frame["_expiry_date"].notna()
            & frame["_strike_f"].notna()
            & frame["_trading_symbol"].ne("")
        ].copy()
        if frame.empty:
            raise ValueError("Flattrade scrip master contains no usable NFO rows.")
        return frame

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry: Any,
        option_type: str,
        strike: Any,
        exchange_segment: str = "NFO",
    ) -> str:
        """Return the exact Flattrade ``Tradingsymbol`` or ``""`` on a miss.

        Args:
            underlying: Index name, currently ``NIFTY`` from the master runner.
            expiry: Date-like option expiry supplied by the Dhan contract lookup.
            option_type: ``CE`` for calls or ``PE`` for puts.
            strike: Human-scale strike such as 24150 (not a paise multiplier).
            exchange_segment: Must be ``NFO`` for this index-options helper.

        Resolution is exact rather than pattern-based: the requested expiry,
        strike, and option type must all occur in Flattrade's downloaded catalogue.
        Returning an empty string makes the caller skip the real order safely.
        """
        underlying_u = str(underlying).upper().strip()
        option_u = str(option_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NFO"
        if option_u not in {"CE", "PE"} or exchange_u != "NFO":
            log.warning(
                "Flattrade resolve skipped: unsupported option/exchange %r/%r.",
                option_type,
                exchange_segment,
            )
            return ""
        try:
            expiry_date = pd.Timestamp(expiry).date()
            strike_i = round(float(strike))
        except Exception:
            log.warning(
                "Flattrade resolve skipped: invalid expiry/strike %r/%r.",
                expiry,
                strike,
            )
            return ""
        cache_key = (underlying_u, expiry_date, option_u, strike_i)
        with self._lock:
            cached = self._symbol_cache.get(cache_key)
            if cached:
                return cached
        if not self.ensure_logged_in():
            return ""
        with self._lock:
            cached = self._symbol_cache.get(cache_key)
            if cached:
                return cached
            if not self._ensure_scrip_master_locked():
                return ""
            frame = self._scrip_df
            if frame is None:
                return ""
            matches = frame[
                (frame["_symbol_u"] == underlying_u)
                & (frame["_option_u"] == option_u)
                & (frame["_expiry_date"] == expiry_date)
                & (frame["_strike_f"].round().astype(int) == strike_i)
            ]
            if matches.empty:
                log.warning(
                    "Flattrade symbol NOT resolved: %s/%s/%s/strike %s.",
                    underlying_u,
                    option_u,
                    expiry_date,
                    strike_i,
                )
                return ""
            symbol = str(matches.iloc[0]["_trading_symbol"]).strip()
            self._symbol_cache[cache_key] = symbol
            return symbol

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
        """Serialize one order from submission through typed fill confirmation."""

        quantity_i = _exact_int(quantity)
        if quantity_i is None or quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not self._order_submission_lock.acquire(
            timeout=_BROKER_CALL_DEADLINE_SECONDS
        ):
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="ORDER_GATE_TIMEOUT",
                reason=(
                    "Flattrade order was not submitted because another order's "
                    "outcome remained unresolved past the broker deadline."
                ),
            )
        try:
            if self._http_poisoned:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="SESSION_POISONED",
                    reason=(
                        "Flattrade session is indeterminate after an earlier "
                        "deadline; reconcile before submitting another order."
                    ),
                )
            return self._place_market_order_serialized(
                symbol,
                side,
                quantity_i,
                exchange_segment,
                product_type,
                order_tag=order_tag,
            )
        finally:
            self._order_submission_lock.release()

    def _place_market_order_serialized(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str = "NFO",
        product_type: str = "INTRADAY",
        *,
        order_tag: str = "",
    ) -> OrderResult:
        """Place one MKT order and return an explicit normalized outcome.

        Args:
            symbol: Exact Flattrade trading symbol returned by the resolver.
            side: Friendly ``BUY`` or ``SELL`` value used by every broker helper.
            quantity: Number of option units, normally a whole-number lot multiple.
            exchange_segment: ``NFO``; other exchanges fail closed in this helper.
            product_type: ``INTRADAY`` (Flattrade ``I``) or ``NORMAL`` (``M``).

        Returns:
            An :class:`OrderResult`. An acknowledgement followed by response
            loss keeps its order id and becomes ``UNKNOWN`` rather than raising
            into the runner's former paper-fallback path.

        Raises:
            ValueError: A caller supplied an unsupported order value.
            RuntimeError: Local session setup cannot be attempted safely.

        Important: ``stat=Ok`` only means the broker accepted the request. The
        method still polls SingleOrdHist before reporting success to the strategy.
        """
        side_u = str(side).upper().strip()
        product_u = str(product_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NFO"
        symbol_s = str(symbol).strip()
        quantity_i = _exact_int(quantity)
        if side_u not in _SIDE_MAP:
            raise ValueError(f"Invalid side: {side!r}")
        if product_u not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if exchange_u != "NFO":
            raise ValueError(f"Invalid exchange_segment: {exchange_segment!r}")
        if quantity_i is None or quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not symbol_s:
            raise ValueError("Missing trading symbol for order.")
        if not self.ensure_logged_in():
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="REJECTED",
                reason="Order was not submitted because Flattrade login failed.",
            )

        protection = _env_non_negative_int("FLATTRADE_MARKET_PROTECTION", 5)
        payload = {
            "uid": self._client_id,
            "actid": self._account_id or self._client_id,
            "exch": exchange_u,
            "tsym": symbol_s,
            "qty": str(quantity_i),
            "prc": "0",
            "dscqty": "0",
            "prd": _PRODUCT_MAP[product_u],
            "trantype": _SIDE_MAP[side_u],
            "prctyp": "MKT",
            "ret": "DAY",
            "ordersource": "API",
            "mkt_protection": str(protection),
        }
        if order_tag:
            payload["remarks"] = str(order_tag)
        try:
            response = self._post_api("PlaceOrder", payload, is_order=True)
        except Exception as exc:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="",
                reason=f"Flattrade PlaceOrder response is indeterminate: {exc}",
            )
        if not self._is_order_ack(response):
            order_id = self.extract_order_id(response)
            explicit_rejection = (
                isinstance(response, dict)
                and str(response.get("stat", "")).strip().lower()
                in {"not_ok", "notok", "rejected"}
            )
            reason = (
                str(response.get("emsg") or response.get("rejreason") or response)
                if isinstance(response, dict)
                else f"Malformed acknowledgement: {response!r}"
            )
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="REJECTED" if explicit_rejection else "",
                reason=reason,
            )
        order_id = self.extract_order_id(response)
        return self._confirm_fill(order_id, quantity_i)

    def _is_order_ack(self, response: Any) -> bool:
        """True only for ``stat=Ok`` payloads carrying a Noren order number."""
        return (
            isinstance(response, dict)
            and str(response.get("stat", "")).lower() == "ok"
            and bool(str(response.get("norenordno") or "").strip())
        )

    def _order_status(
        self,
        order_id: str,
    ) -> tuple[str | None, int | None, int | None, str]:
        """Return normalized state, filled qty, order qty, and rejection reason."""
        response = self._post_api(
            "SingleOrdHist",
            {"uid": self._client_id, "norenordno": str(order_id)},
            allow_after_poison=True,
        )
        rows = response if isinstance(response, list) else [response]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("stat", "Ok")).lower() not in {"", "ok"}:
                continue
            state = str(row.get("status") or "").strip().lower() or None

            return (
                state,
                _exact_int(row.get("fillshares")),
                _exact_int(row.get("qty")),
                str(row.get("rejreason") or row.get("emsg") or "").strip(),
            )
        reason = (
            response.get("emsg", "unrecognised response")
            if isinstance(response, dict) else "unrecognised response"
        )
        return (None, None, None, str(reason))

    def get_order_status(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Return one normalized ``SingleOrdHist`` snapshot.

        Exceptions and malformed quantities are represented as ``UNKNOWN`` while
        retaining the caller's order id.  They never become zero-fill rejection.
        """
        requested = _exact_int(requested_quantity)
        if requested is None or requested < 0:
            requested = 0
        try:
            state, filled, broker_qty, reason = self._order_status(str(order_id))
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"Flattrade order-status query failed: {exc}",
            )

        effective_requested = requested or broker_qty or 0
        if requested and broker_qty not in {None, requested}:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=None,
                broker_state=state or "",
                reason=(
                    f"Malformed Flattrade status: broker quantity {broker_qty} "
                    f"does not match requested quantity {requested}. {reason}"
                ),
            )
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=effective_requested,
            filled_quantity=filled,
            broker_state=state or "",
            reason=reason,
        )

    def _confirm_fill(self, order_id: str, want_qty: int) -> OrderResult:
        """Poll briefly for a terminal result without hiding ambiguity."""
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_result = normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=0,
            broker_state="",
            reason="Flattrade acknowledgement received; fill status not read yet.",
        )
        while time.monotonic() < deadline:
            last_result = self.get_order_status(order_id, requested_quantity=want_qty)
            if last_result.status is not OrderStatus.UNKNOWN:
                return last_result
            # A failed/malformed status call is already indeterminate. Repeating
            # it inside the same order interaction risks exceeding the deadline
            # without adding evidence, so return immediately with the known id.
            if "query failed" in last_result.reason.lower():
                return last_result
            if last_result.broker_state not in {"", "OPEN", "PENDING", "TRIGGER_PENDING"}:
                return last_result
            time.sleep(_FILL_POLL_INTERVAL)
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=last_result.filled_quantity,
            broker_state=last_result.broker_state,
            reason=(
                f"Flattrade order {order_id} was not terminal within "
                f"{_FILL_TIMEOUT_SECONDS:.0f}s; exposure is indeterminate."
            ),
        )

    def cancel_order(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Request cancellation, then normalize the order's resulting status."""

        order_id_s = str(order_id).strip()
        requested = _exact_int(requested_quantity)
        if not order_id_s:
            raise ValueError("Missing order id for cancellation.")
        if requested is None or requested < 0:
            raise ValueError(f"Invalid requested_quantity: {requested_quantity!r}")
        if not self.ensure_logged_in():
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason="Flattrade cancellation was not submitted because login failed.",
            )
        try:
            self._post_api(
                "CancelOrder",
                {"uid": self._client_id, "norenordno": order_id_s},
                is_order=True,
                allow_after_poison=True,
            )
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"Flattrade cancellation response is indeterminate: {exc}",
            )
        return self.get_order_status(order_id_s, requested_quantity=requested)

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        """Return pending orders, or an explicit indeterminate query result."""

        try:
            response = self._post_api(
                "GetPendingOrders",
                {"uid": self._client_id},
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-order query failed: {exc}"
            )
        if isinstance(response, dict):
            if _is_no_data_envelope(response):
                return BrokerQueryResult.success([])
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-order query returned {response!r}"
            )
        if not isinstance(response, list):
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-order query returned malformed data: {response!r}"
            )

        orders: list[OpenOrder] = []
        for row in response:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Flattrade open-order query contained a malformed row."
                )
            raw_state = str(row.get("status") or "").strip().upper()
            snapshot = normalize_order_result(
                order_id=row.get("norenordno"),
                requested_quantity=row.get("qty"),
                filled_quantity=row.get("fillshares"),
                broker_state=row.get("status"),
                reason="Open-order snapshot",
            )
            symbol = str(row.get("tsym") or "").strip()
            if (
                not snapshot.order_id
                or not symbol
                or "malformed" in snapshot.reason.lower()
                or snapshot.requested_quantity <= 0
            ):
                return BrokerQueryResult.indeterminate(
                    "Flattrade open-order query contained incomplete order data."
                )
            if raw_state in {
                "COMPLETE", "COMPLETED", "FILLED", "TRADED", "EXECUTED",
                "REJECTED", "CANCELLED", "CANCELED", "CANCEL", "LAPSED",
            }:
                continue
            if snapshot.remaining_quantity <= 0:
                continue
            side = {"B": "BUY", "S": "SELL"}.get(
                str(row.get("trantype") or "").strip().upper(),
                str(row.get("trantype") or "").strip().upper(),
            )
            orders.append(
                OpenOrder(
                    order_id=snapshot.order_id,
                    symbol=symbol,
                    side=side,
                    requested_quantity=snapshot.requested_quantity,
                    filled_quantity=snapshot.filled_quantity,
                    remaining_quantity=snapshot.remaining_quantity,
                    broker_state=snapshot.broker_state,
                )
            )
        return BrokerQueryResult.success(orders)

    def list_open_positions(self) -> BrokerQueryResult[OpenPosition]:
        """Return non-flat positions, or an explicit indeterminate result."""

        try:
            response = self._post_api(
                "PositionBook",
                {"uid": self._client_id},
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-position query failed: {exc}"
            )
        if isinstance(response, dict):
            if _is_no_data_envelope(response):
                return BrokerQueryResult.success([])
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-position query returned {response!r}"
            )
        if not isinstance(response, list):
            return BrokerQueryResult.indeterminate(
                f"Flattrade open-position query returned malformed data: {response!r}"
            )

        positions: list[OpenPosition] = []
        for row in response:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Flattrade open-position query contained a malformed row."
                )
            quantity = _exact_int(row.get("netqty"))
            symbol = str(row.get("tsym") or "").strip()
            if quantity is None or not symbol:
                return BrokerQueryResult.indeterminate(
                    "Flattrade open-position query contained incomplete position data."
                )
            if quantity == 0:
                continue
            positions.append(
                OpenPosition(
                    symbol=symbol,
                    quantity=quantity,
                    product_type=str(row.get("prd") or "").strip(),
                    broker_state="OPEN",
                )
            )
        return BrokerQueryResult.success(positions)

    def extract_order_id(self, order_response: Any) -> str:
        """Recursively find Flattrade's ``norenordno`` in a response payload."""
        if isinstance(order_response, OrderResult):
            return order_response.order_id
        if order_response is None:
            return ""
        if isinstance(order_response, dict):
            for key in ("norenordno", "order_id", "orderId", "id"):
                value = order_response.get(key)
                if value:
                    return str(value).strip()
            for value in order_response.values():
                found = self.extract_order_id(value)
                if found:
                    return found
            return ""
        if isinstance(order_response, list):
            for value in order_response:
                found = self.extract_order_id(value)
                if found:
                    return found
            return ""
        if isinstance(order_response, str):
            return order_response.strip()
        return ""

    def logout(self) -> dict[str, Any]:
        """Close the local session and erase sensitive in-memory state.

        Flattrade Pi v2 exposes no documented remote logout call. Closing requests'
        connection pool, clearing the token/account fields, contract cache, and
        rate histories is therefore the complete local logout operation.
        """
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception as exc:
                    log.warning("Flattrade local session close failed: %s", exc)
            self.is_logged_in = False
            self._access_token = ""
            self._client_id = ""
            self._account_id = ""
            self._symbol_cache.clear()
            self._scrip_df = None
            self._api_limiter.reset()
            self._order_limiter.reset()
            return {"stat": "Ok", "message": "Flattrade local session cleared"}


# The master imports and reuses this one object so all worker threads share the
# same authenticated session, contract cache, lock, and rate-limit budgets.
flattrade_execution_client = FlattradeExecutionClient()
