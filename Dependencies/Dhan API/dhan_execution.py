"""
Thread-safe DhanHQ v2 execution client for the multi-strategy runner.

The master runner deliberately knows almost nothing about broker APIs.  It asks
each broker helper to log in, preload a contract master, resolve one NIFTY option
and place a confirmed market order.  This module implements that same small
surface for DhanHQ using the official ``dhanhq`` Python SDK.

Safety rules followed here:

* Importing this module never logs in or sends an order.
* The access token is read from the environment ONLY (``DHAN_ACCESS_TOKEN``,
  produced by ``Dependencies/dhan_token_setup.py``).  There is no browser flow
  and no TOTP prompt, so startup never blocks on operator input.
* Every SDK call carries a native timeout AND one total ten-second deadline that
  also covers rate-limit wait, session-lock wait, and slow-dribble response
  bodies.
* ``place_market_order`` returns a typed result only after the order book
  confirms the fill.  Rejection, partial fill, and response loss remain
  distinct; ambiguity is ``UNKNOWN`` and poisons new submissions until explicit
  reconciliation.

Beginner's mental model -- one live order travels through this file as follows:

1. ``ensure_logged_in()`` validates today's access token against the account.
2. ``preload_scrip_master()`` loads Dhan's contract catalogue from the local
   instrument CSV the runner already refreshes every evening.
3. ``resolve_option_symbol()`` converts strategy details such as
   NIFTY/14-JUL-2026/CE/24150 into Dhan's exact trading symbol.
4. ``place_market_order()`` translates that symbol back into the numeric
   ``securityId`` Dhan's order API actually wants, then submits the order.
5. ``_confirm_fill()`` polls the order book; an acknowledgement alone is never
   treated as proof that money changed hands.

TWO DHAN-SPECIFIC HAZARDS THIS MODULE EXISTS TO CONTAIN
-------------------------------------------------------

**1. The SDK turns every exception into a rejection-shaped envelope.**
``DhanHTTP._send_request`` wraps its request in a bare ``except Exception`` and
returns ``{'status': 'failure', 'remarks': str(e), 'data': ''}``.  A socket
timeout on an order that DID reach the exchange is therefore structurally
identical to a genuine broker rejection.  In this repo ``REJECTED`` means
"provably zero fill -- safe to fall back to paper", so believing that envelope
would re-enter on paper while real exposure sits at the broker.

This module NEVER derives ``REJECTED`` from the placement envelope.  It treats
the response as an acknowledgement only, and distinguishes the two failure
shapes the SDK can produce (see ``_classify_envelope``):

* ``remarks`` is a **dict** -- Dhan's server answered and refused with a
  structured ``errorCode``.  A candidate rejection, still confirmed before it is
  trusted.
* ``remarks`` is a **str** -- a transport-layer exception.  The order may have
  reached the exchange, so the outcome is ``UNKNOWN`` and freezes new entries.

Dhan gives us a recovery channel the other brokers lack: the ``order_tag`` is
sent as ``correlationId``, so ``get_order_by_correlationID`` can find an order
even when its POST response was lost entirely.  Both ambiguous paths try that
lookup before concluding anything.

**2. The SDK's native timeout is 60 seconds, not 10.**
``DhanHTTP.HTTP_DEFAULT_TIME_OUT`` is 60 -- six times this repo's deadline.  It
is a plain instance attribute, so ``_login_locked`` overrides it immediately
after building the context.  ``DhanContext.__init__`` also swallows its own
exceptions and can hand back a half-built object, so the attribute is verified
rather than assumed.

Small glossary:

* A *scrip master* is the broker's contract catalogue: symbol, expiry, strike,
  option type, security id, and lot size for every listed instrument.
* ``securityId`` is Dhan's numeric instrument key.  Dhan's order API takes it
  INSTEAD of a trading symbol, which is why this helper keeps a symbol -> id map.
* ``correlationId`` is Dhan's name for a caller-supplied order tag, echoed back
  in the order book and searchable through its own endpoint.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import pandas as pd
from dhanhq import DhanContext, dhanhq

# The broker helpers live in folders with spaces and are also executable as
# standalone diagnostics.  Add their shared ``Dependencies`` parent explicitly
# so the broker-neutral contract imports the same way in both entry points.
_DEPENDENCIES_DIR = Path(__file__).resolve().parent.parent
if str(_DEPENDENCIES_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPENDENCIES_DIR))

from broker_contract import (  # noqa: E402
    TERMINAL_BROKER_STATES,
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

# The runner refreshes this file every evening from Dhan's public detailed scrip
# master.  Reusing it (rather than re-downloading 35 MB inside a ten-second
# budget) also guarantees the adapter and the strategies resolve a strike to the
# very same contract.
_INSTRUMENT_MASTER_GLOB = "all_instrument *.csv"

_API_TIMEOUT_SECONDS = 10.0
_BROKER_CALL_DEADLINE_SECONDS = 10.0
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 2.0

# Dhan accepts these product names directly; ``NORMAL`` is this repo's neutral
# word for a carry-forward position, which Dhan calls ``MARGIN``.
_PRODUCT_MAP = {"INTRADAY": "INTRADAY", "NORMAL": "MARGIN", "MARGIN": "MARGIN"}
_SIDE_MAP = {"BUY": "BUY", "SELL": "SELL"}

# Dhan reports two order states the broker-neutral contract does not recognize.
# Translating them here keeps the shared vocabulary (and the other three broker
# adapters) untouched while still giving each state its correct meaning:
#   EXPIRED     -- the order died without trading; that is a terminal rejection,
#                  and leaving it unmapped would report a clean "never traded"
#                  as UNKNOWN and needlessly freeze every live strategy.
#   PART_TRADED -- a partial fill, which usually resolves through the quantity
#                  comparison anyway but must not read as an unknown label when
#                  the fill count is unavailable.
# TRANSIT and PENDING are deliberately NOT mapped: they are transient states a
# healthy order passes through, so they must stay UNKNOWN and keep the fill loop
# polling rather than being reported as a final outcome.
_DHAN_STATE_ALIASES = {"EXPIRED": "CANCELLED", "PART_TRADED": "PARTIAL"}

# Dhan's order-book field names, most-specific first.  The SDK documents these
# only in prose, so every read goes through ``_first_present`` and a casing
# change degrades to UNKNOWN instead of a silently wrong fill count.
_ORDER_ID_KEYS = ("orderId", "order_id", "orderid")
_ORDER_STATE_KEYS = ("orderStatus", "order_status", "status")
_FILLED_QTY_KEYS = ("filledQty", "filled_qty", "filledQuantity", "tradedQty")
_ORDER_QTY_KEYS = ("quantity", "orderQuantity", "qty")
_SYMBOL_KEYS = ("tradingSymbol", "trading_symbol", "symbol")
_SIDE_KEYS = ("transactionType", "transaction_type")
_PRODUCT_KEYS = ("productType", "product_type")
_NET_QTY_KEYS = ("netQty", "net_qty", "netQuantity")
_REASON_KEYS = ("omsErrorDescription", "oms_error_description", "errorMessage")


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


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value among ``keys``, else ``None``.

    Dhan's field casing is documented only in SDK docstrings, so every order-book
    read tolerates the plausible spellings instead of trusting one of them.
    """

    for key in keys:
        if key in row:
            value = row[key]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _alias_state(raw_state: Any) -> str:
    """Translate a Dhan order state into the broker-neutral vocabulary."""

    state = str(raw_state or "").strip().upper().replace("-", "_").replace(" ", "_")
    return _DHAN_STATE_ALIASES.get(state, state)


def _env_str(name: str, default: str = "") -> str:
    """Read one environment value and remove accidental wrapping quotes."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value or default


class _RollingWindowRateLimiter:
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
                f"Dhan {self.label} deadline expired waiting for the limiter lock"
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
                        f"Dhan {self.label} rate limit exhausted; "
                        f"safe slot needs {wait_seconds:.2f}s"
                    )
                self._sleep(wait_seconds)
        finally:
            self._lock.release()

    def reset(self) -> None:
        """Forget local request history after a session is explicitly closed."""
        with self._lock:
            self._timestamps.clear()


def _classify_envelope(envelope: Any) -> tuple[str, Any, str]:
    """Split a DhanHQ SDK reply into outcome, payload, and human reason.

    Every SDK method returns ``{'status', 'remarks', 'data'}``.  Both a genuine
    broker refusal and a local network exception arrive as ``status='failure'``,
    so the caller cannot use that field alone without risking the "timed-out
    order reported as rejected" bug this whole module guards against.  The
    ``remarks`` TYPE is the discriminator:

    * ``dict``  -- built by ``DhanHTTP._parse_response`` from a non-2xx reply,
      carrying ``error_code``/``error_type``/``error_message``.  Dhan's server
      answered and refused.
    * ``str``   -- built by ``DhanHTTP._send_request``'s bare exception handler
      (or a JSON decode failure).  We do not know whether the request reached
      the exchange.

    Returns:
        ``(outcome, data, reason)`` where outcome is one of ``"ok"``,
        ``"refused"``, or ``"indeterminate"``.
    """

    if not isinstance(envelope, dict):
        return ("indeterminate", None, f"Malformed Dhan response: {envelope!r}")

    status = str(envelope.get("status") or "").strip().lower()
    remarks = envelope.get("remarks")
    if status == "success":
        return ("ok", envelope.get("data"), "")

    if isinstance(remarks, dict):
        detail = " ".join(
            str(remarks.get(key) or "").strip()
            for key in ("error_code", "error_type", "error_message")
        ).strip()
        return ("refused", envelope.get("data"), detail or "Dhan refused the request.")

    return (
        "indeterminate",
        envelope.get("data"),
        str(remarks or "").strip() or "Dhan response did not complete.",
    )


class DhanExecutionClient:
    """One lazily authenticated, lock-guarded DhanHQ execution session.

    The master creates many strategy threads but imports one singleton from the
    bottom of this module.  All those threads therefore share one SDK client,
    one token, one contract cache, and one pair of rate-limit counters.
    ``RLock`` makes nested helper calls safe without allowing simultaneous
    session writes.
    """

    def __init__(self) -> None:
        self.client: Any = None
        self.is_logged_in = False
        self._lock = threading.RLock()
        self._context: Any = None
        self._client_code = ""
        self._scrip_df: pd.DataFrame | None = None
        self._symbol_cache: dict[tuple, str] = {}
        # Dhan's order API takes a numeric securityId, but the runner's generic
        # surface passes the human trading symbol returned by the resolver.
        # This map bridges the two so logs and Telegram alerts stay readable.
        self._security_id_by_symbol: dict[str, str] = {}
        # ``requests`` timeouts are inactivity limits, not a total wall-clock
        # budget: a slow-dribble response can otherwise run forever.  Execute one
        # SDK call at a time and impose a caller-visible deadline around
        # rate-limit wait, session-lock wait, and the complete response body.
        self._sdk_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dhan-sdk",
        )
        self._sdk_poisoned = False
        self._timed_out_future: Future[Any] | None = None
        # Serialize acknowledgement plus order-book confirmation.  Without this
        # gate another strategy can submit before a status timeout poisons the
        # session, leaving two simultaneously indeterminate live orders.
        self._order_submission_lock = threading.Lock()
        # Conservative floors that sit well under Dhan's published per-second
        # order ceiling.  With ~26 strategy threads real bursts are far smaller,
        # so these buy safety margin rather than costing throughput.
        self._api_limiter = _RollingWindowRateLimiter(
            20,
            800,
            _MAX_RATE_LIMIT_WAIT_SECONDS,
            label="API",
        )
        self._order_limiter = _RollingWindowRateLimiter(
            10,
            250,
            _MAX_RATE_LIMIT_WAIT_SECONDS,
            label="order API",
        )

    @staticmethod
    def _remaining_broker_budget(started: float) -> float:
        """Return seconds left in one complete broker-call wall-clock budget."""

        return _BROKER_CALL_DEADLINE_SECONDS - (time.monotonic() - started)

    def _run_sdk_with_deadline(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        started: float,
        allow_after_poison: bool = False,
    ) -> Any:
        """Run one SDK call without letting a slow response outlive 10s.

        The caller holds ``_lock``.  If the future times out, its socket may
        still be active inside ``requests``; poison the shared session so no
        second request can overlap that indeterminate in-flight operation.
        """

        if self._sdk_poisoned:
            abandoned_finished = (
                self._timed_out_future is not None
                and self._timed_out_future.done()
            )
            if not allow_after_poison or not abandoned_finished:
                raise RuntimeError(
                    "Dhan session is indeterminate after a broker deadline; "
                    "new orders are blocked pending reconciliation."
                )
        remaining = self._remaining_broker_budget(started)
        if remaining <= 0:
            raise TimeoutError(f"Dhan {operation} deadline expired before submission.")
        future = self._sdk_executor.submit(call)
        try:
            return future.result(timeout=remaining)
        except FuturesTimeoutError as exc:
            # Python aliases concurrent.futures.TimeoutError to built-in
            # TimeoutError.  If the SDK call itself raised it, preserve that
            # evidence rather than falsely poisoning a completed future.
            if future.done():
                return future.result()
            self._sdk_poisoned = True
            self._timed_out_future = future
            future.cancel()
            raise TimeoutError(
                f"Dhan {operation} exceeded the total "
                f"{_BROKER_CALL_DEADLINE_SECONDS:.0f}s broker deadline; "
                "session state is indeterminate."
            ) from exc

    def _call_api(
        self,
        operation: str,
        call: Callable[[Any], Any],
        *,
        is_order: bool = False,
        allow_after_poison: bool = False,
    ) -> Any:
        """Run one SDK method under the shared limits, lock, and deadline.

        Args:
            operation: Human label used in deadline and error messages.
            call: Receives the live ``dhanhq`` client and performs one request.
            is_order: Also consume the stricter order budget for write calls.
            allow_after_poison: Permit reconciliation/risk-reducing calls only
                after the abandoned timed-out future has actually ended.

        Returns:
            The raw ``{'status', 'remarks', 'data'}`` envelope from the SDK.

        Raises:
            RuntimeError: No client, or no rate budget, is available.
            TimeoutError: The ten-second total budget expired.
        """
        started = time.monotonic()
        deadline = started + _BROKER_CALL_DEADLINE_SECONDS
        self._api_limiter.acquire(deadline)
        if is_order:
            self._order_limiter.acquire(deadline)

        remaining = self._remaining_broker_budget(started)
        if remaining <= 0 or not self._lock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(
                "Dhan broker deadline expired while waiting for the shared session lock."
            )
        try:
            client = self.client
            if client is None:
                raise RuntimeError("Dhan has no authenticated SDK client.")
            return self._run_sdk_with_deadline(
                operation,
                lambda: call(client),
                started=started,
                allow_after_poison=allow_after_poison,
            )
        finally:
            self._lock.release()

    def recover_after_reconciliation(self) -> bool:
        """Explicitly clear deadline poison after the abandoned call has ended.

        "Poisoned" means an earlier SDK call outlived its 10-second budget and
        was abandoned -- but Python cannot kill a running thread, so that call
        may STILL be dribbling a response (or completing an order) inside
        ``requests``.  Only after (a) reconciliation has proven the account
        state and (b) the abandoned call has actually finished is it safe to
        accept new orders; this method refuses (returns False) until both hold.
        """

        with self._lock:
            if self._timed_out_future is not None and not self._timed_out_future.done():
                return False
            self._sdk_poisoned = False
            self._timed_out_future = None
            return True

    def ensure_logged_in(self) -> bool:
        """Return True only when a validated DhanHQ session is available.

        This method is intentionally safe to call before every operation.  The
        normal fast path only checks in-memory state; the validation request
        runs once when the session has not yet been established.
        """
        with self._lock:
            if self.is_logged_in and self.client is not None:
                return True
            return self._login_locked()

    def _login_locked(self) -> bool:
        """Build and validate a DhanHQ session from environment credentials.

        Unlike the other broker helpers there is no interactive step: the
        access token is produced offline by ``dhan_token_setup.py`` and read
        from the environment.  Returning ``False`` instead of raising lets
        startup force every strategy back to paper mode.  The token is never
        logged.

        Note the token is NOT long-lived: DhanHQ stamps a validity into it and
        the web console currently issues 24-hour tokens.  One that expires
        mid-session makes orders fail, and since a rejected live EXIT leaves a
        position open, refresh it outside market hours.
        """
        self.is_logged_in = False
        self.client = None
        self._context = None
        self._client_code = _env_str("DHAN_CLIENT_CODE")
        access_token = _env_str("DHAN_ACCESS_TOKEN")
        missing = [
            name
            for name, value in (
                ("DHAN_CLIENT_CODE", self._client_code),
                ("DHAN_ACCESS_TOKEN", access_token),
            )
            if not value
        ]
        if missing:
            log.error("Dhan login disabled: missing %s.", ", ".join(missing))
            return False

        try:
            context = DhanContext(self._client_code, access_token)
            # DhanContext.__init__ swallows its own exceptions and can return a
            # half-built object, so prove the HTTP layer exists before trusting
            # it -- otherwise the failure resurfaces as an AttributeError in the
            # middle of a live order.
            http = context.get_dhan_http()
            if http is None:
                raise RuntimeError("DhanContext produced no HTTP connection.")
            # The SDK ships a 60-second default, six times this repo's deadline.
            http.timeout = _API_TIMEOUT_SECONDS
            self._context = context
            self.client = dhanhq(context)
        except Exception as exc:
            self.client = None
            self._context = None
            log.error("Dhan session construction failed: %s", exc)
            return False

        if not self._validate_session_locked():
            self.client = None
            self._context = None
            return False
        self.is_logged_in = True
        log.info("Dhan execution client login successful.")
        return True

    def _validate_session_locked(self) -> bool:
        """Prove the token works with one cheap authenticated read."""
        try:
            envelope = self._call_api("fund limits", lambda client: client.get_fund_limits())
        except Exception as exc:
            log.error("Dhan session validation failed: %s", exc)
            return False
        outcome, _data, reason = _classify_envelope(envelope)
        if outcome != "ok":
            log.error(
                "Dhan session validation rejected (%s): %s. "
                "Re-run 'python algo.py setup-token' if the token has expired.",
                outcome,
                reason,
            )
            return False
        return True

    def preload_scrip_master(self) -> bool:
        """Log in and cache Dhan's option catalogue from the local instrument CSV.

        The master calls this once before worker threads start.  Returning
        ``False`` makes startup disable live trading instead of discovering a
        missing or malformed contract catalogue during the first real order.
        """
        if not self.ensure_logged_in():
            return False
        with self._lock:
            return self._ensure_scrip_master_locked()

    def _ensure_scrip_master_locked(self) -> bool:
        """Load and normalize the contract master once; caller holds ``_lock``."""
        if self._scrip_df is not None:
            return not self._scrip_df.empty
        try:
            path = self._latest_instrument_master_path()
            if path is None:
                raise FileNotFoundError(
                    f"No instrument master matching '{_INSTRUMENT_MASTER_GLOB}' in "
                    f"{_DEPENDENCIES_DIR}. The runner refreshes this file at end of day."
                )
            self._scrip_df = self._prepare_scrip_master(path)
            self._security_id_by_symbol = {
                str(row.trading_symbol): str(row.security_id)
                for row in self._scrip_df.itertuples(index=False)
            }
            log.info(
                "Dhan NSE index-option scrip master loaded (%d rows) from %s.",
                len(self._scrip_df),
                path.name,
            )
            return True
        except Exception as exc:
            self._scrip_df = None
            self._security_id_by_symbol = {}
            log.error("Dhan scrip master load/parse failed: %s", exc)
            return False

    @staticmethod
    def _latest_instrument_master_path() -> Path | None:
        """Return the newest ``all_instrument <date>.csv``, or ``None``.

        The filenames carry an ISO date, so a plain sort is already
        chronological; ``max`` therefore picks the freshest catalogue.
        """

        candidates = sorted(_DEPENDENCIES_DIR.glob(_INSTRUMENT_MASTER_GLOB))
        return candidates[-1] if candidates else None

    @staticmethod
    def _prepare_scrip_master(path: Path) -> pd.DataFrame:
        """Read the detailed scrip master and keep only usable index options.

        The filters mirror the runner's own option resolver so this adapter and
        the strategies can never disagree about which contract a strike means:
        NSE exchange, ``D`` (derivatives) segment, ``OPTIDX`` instrument, a real
        CE/PE type, and a parseable expiry, strike, and security id.

        Only the columns actually needed are read: the file is ~35 MB and this
        runs while the shared session lock is held.

        Raises:
            ValueError: Required columns or usable option rows are missing.
        """
        required = [
            "EXCH_ID",
            "SEGMENT",
            "INSTRUMENT",
            "SYMBOL_NAME",
            "SM_EXPIRY_DATE",
            "SECURITY_ID",
            "STRIKE_PRICE",
            "OPTION_TYPE",
            "UNDERLYING_SYMBOL",
        ]
        header = pd.read_csv(path, nrows=0)
        missing = [column for column in required if column not in header.columns]
        if missing:
            raise ValueError(
                f"Dhan scrip master {path.name} missing columns: " + ", ".join(missing)
            )
        optional = [
            column for column in ("DISPLAY_NAME", "LOT_SIZE") if column in header.columns
        ]

        raw = pd.read_csv(
            path,
            usecols=required + optional,
            dtype=str,
            low_memory=False,
        )
        frame = pd.DataFrame(
            {
                "_exchange_u": raw["EXCH_ID"].fillna("").astype(str).str.strip().str.upper(),
                "_segment_u": raw["SEGMENT"].fillna("").astype(str).str.strip().str.upper(),
                "_instrument_u": raw["INSTRUMENT"].fillna("").astype(str).str.strip().str.upper(),
                "trading_symbol": raw["SYMBOL_NAME"].fillna("").astype(str).str.strip(),
                "display_name": (
                    raw["DISPLAY_NAME"].fillna("").astype(str).str.strip()
                    if "DISPLAY_NAME" in optional
                    else ""
                ),
                "_underlying_u": (
                    raw["UNDERLYING_SYMBOL"].fillna("").astype(str).str.strip().str.upper()
                ),
                "_option_u": raw["OPTION_TYPE"].fillna("").astype(str).str.strip().str.upper(),
                "_expiry_date": pd.to_datetime(
                    raw["SM_EXPIRY_DATE"], errors="coerce"
                ).dt.date,
                "_strike_f": pd.to_numeric(raw["STRIKE_PRICE"], errors="coerce"),
                "security_id": pd.to_numeric(raw["SECURITY_ID"], errors="coerce"),
                "lot_size": (
                    pd.to_numeric(raw["LOT_SIZE"], errors="coerce")
                    if "LOT_SIZE" in optional
                    else 0
                ),
            }
        )
        frame = frame[
            (frame["_exchange_u"] == "NSE")
            & (frame["_segment_u"] == "D")
            & (frame["_instrument_u"] == "OPTIDX")
            & frame["_option_u"].isin(["CE", "PE"])
            & frame["_expiry_date"].notna()
            & frame["_strike_f"].notna()
            & frame["security_id"].notna()
            & frame["trading_symbol"].ne("")
        ].copy()
        if frame.empty:
            raise ValueError(
                f"Dhan scrip master {path.name} contains no usable index-option rows."
            )
        frame["security_id"] = frame["security_id"].astype(int)
        return frame

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry: Any,
        option_type: str,
        strike: Any,
        exchange_segment: str = "NSE_FNO",
    ) -> str:
        """Return the exact Dhan trading symbol or ``""`` on a miss.

        Args:
            underlying: Index name such as ``NIFTY`` or ``BANKNIFTY``.
            expiry: Date-like option expiry supplied by the contract lookup.
            option_type: ``CE`` for calls or ``PE`` for puts.
            strike: Human-scale strike such as 24150 (not a paise multiplier).
            exchange_segment: Must be ``NSE_FNO`` for this index-options helper.

        Resolution is exact rather than pattern-based: the requested expiry,
        strike, and option type must all occur in Dhan's catalogue.  Returning
        an empty string makes the caller skip the real order safely.

        The returned value is the human symbol so logs and Telegram alerts stay
        readable; ``place_market_order`` translates it back to the numeric
        ``securityId`` Dhan's order API requires.
        """
        underlying_u = str(underlying).upper().strip()
        option_u = str(option_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NSE_FNO"
        if option_u not in {"CE", "PE"} or exchange_u != "NSE_FNO":
            log.warning(
                "Dhan resolve skipped: unsupported option/exchange %r/%r.",
                option_type,
                exchange_segment,
            )
            return ""
        try:
            expiry_date = pd.Timestamp(expiry).date()
            strike_i = round(float(strike))
        except Exception:
            log.warning(
                "Dhan resolve skipped: invalid expiry/strike %r/%r.",
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
                (frame["_underlying_u"] == underlying_u)
                & (frame["_option_u"] == option_u)
                & (frame["_expiry_date"] == expiry_date)
                & (frame["_strike_f"].round().astype(int) == strike_i)
            ]
            if matches.empty:
                log.warning(
                    "Dhan symbol NOT resolved: %s/%s/%s/strike %s.",
                    underlying_u,
                    option_u,
                    expiry_date,
                    strike_i,
                )
                return ""
            row = matches.iloc[0]
            symbol = str(row["trading_symbol"]).strip()
            self._symbol_cache[cache_key] = symbol
            self._security_id_by_symbol[symbol] = str(int(row["security_id"]))
            return symbol

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str = "NSE_FNO",
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
                    "Dhan order was not submitted because another order's "
                    "outcome remained unresolved past the broker deadline."
                ),
            )
        try:
            if self._sdk_poisoned:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="SESSION_POISONED",
                    reason=(
                        "Dhan session is indeterminate after an earlier "
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
        exchange_segment: str = "NSE_FNO",
        product_type: str = "INTRADAY",
        *,
        order_tag: str = "",
    ) -> OrderResult:
        """Place one MARKET order and return an explicit normalized outcome.

        Args:
            symbol: Exact Dhan trading symbol returned by the resolver.
            side: Friendly ``BUY`` or ``SELL`` value used by every broker helper.
            quantity: Number of option units, normally a whole-number lot multiple.
            exchange_segment: ``NSE_FNO``; other segments fail closed here.
            product_type: ``INTRADAY`` or ``NORMAL`` (Dhan ``MARGIN``).
            order_tag: Sent as Dhan's ``correlationId`` so a lost response can
                still be traced back to its order.

        Returns:
            An :class:`OrderResult`.  An acknowledgement followed by response
            loss keeps its order id and becomes ``UNKNOWN`` rather than being
            mistaken for a rejection.

        Raises:
            ValueError: A caller supplied an unsupported order value.

        Important: a ``success`` envelope only means Dhan accepted the request.
        The method still polls the order book before reporting a fill.
        """
        side_u = str(side).upper().strip()
        product_u = str(product_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NSE_FNO"
        symbol_s = str(symbol).strip()
        quantity_i = _exact_int(quantity)
        if side_u not in _SIDE_MAP:
            raise ValueError(f"Invalid side: {side!r}")
        if product_u not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if exchange_u != "NSE_FNO":
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
                reason="Order was not submitted because Dhan login failed.",
            )

        security_id = self._security_id_for(symbol_s)
        if not security_id:
            # Nothing was sent, so this is a provable zero-fill.  Guessing an id
            # would be far worse than refusing: it could trade a different
            # contract entirely.
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="REJECTED",
                reason=(
                    f"Dhan order was not submitted: no securityId is known for "
                    f"symbol {symbol_s!r}. Resolve the contract first."
                ),
            )

        try:
            envelope = self._call_api(
                "place order",
                lambda client: client.place_order(
                    security_id=security_id,
                    exchange_segment=exchange_u,
                    transaction_type=_SIDE_MAP[side_u],
                    quantity=quantity_i,
                    order_type="MARKET",
                    product_type=_PRODUCT_MAP[product_u],
                    price=0,
                    tag=order_tag or None,
                ),
                is_order=True,
            )
        except Exception as exc:
            # The request may or may not have reached the exchange.  Try the
            # correlation lookup before giving up on identifying the order.
            return self._resolve_indeterminate_placement(
                quantity_i,
                order_tag,
                f"Dhan place-order call is indeterminate: {exc}",
            )

        outcome, data, reason = _classify_envelope(envelope)
        order_id = self.extract_order_id(data) if outcome == "ok" else ""
        if order_id:
            return self._confirm_fill(order_id, quantity_i)

        if outcome == "refused":
            # Dhan's server answered with a structured error.  Confirm through
            # the correlation id that no order actually exists before calling
            # this a provable zero fill.
            found_id, lookup_succeeded = self._find_order_id_by_tag(order_tag)
            if found_id:
                return self._confirm_fill(found_id, quantity_i)
            if lookup_succeeded or not order_tag:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason=f"Dhan rejected the order: {reason}",
                )
            return self._indeterminate_placement(
                quantity_i,
                f"Dhan refused the order ({reason}) but the correlation lookup "
                "could not confirm that nothing was created.",
            )

        # "ok" with no order id, or an indeterminate transport failure.
        return self._resolve_indeterminate_placement(
            quantity_i,
            order_tag,
            reason or "Dhan acknowledged the order without an order id.",
        )

    def _security_id_for(self, symbol: str) -> str:
        """Look up the numeric securityId cached for one trading symbol."""

        with self._lock:
            return self._security_id_by_symbol.get(symbol, "")

    def _find_order_id_by_tag(self, order_tag: str) -> tuple[str, bool]:
        """Search Dhan's order book for an order carrying ``order_tag``.

        Dhan echoes the tag back as ``correlationId`` and exposes a dedicated
        lookup, which is how a completely lost placement response can still be
        traced to a real order.

        Returns:
            ``(order_id, lookup_succeeded)``.  The flag distinguishes "looked
            and found nothing" (safe to call a rejection) from "could not look"
            (must stay indeterminate).
        """

        if not order_tag:
            return ("", False)
        try:
            envelope = self._call_api(
                "order by correlation id",
                lambda client: client.get_order_by_correlationID(str(order_tag)),
                allow_after_poison=True,
            )
        except Exception as exc:
            log.warning("Dhan correlation-id lookup failed: %s", exc)
            return ("", False)
        outcome, data, _reason = _classify_envelope(envelope)
        if outcome == "ok":
            return (self.extract_order_id(data), True)
        if outcome == "refused":
            # A structured refusal here means Dhan searched and found no order
            # for this correlation id.
            return ("", True)
        return ("", False)

    def _indeterminate_placement(self, quantity: int, reason: str) -> OrderResult:
        """Build the UNKNOWN result used whenever exposure cannot be proven."""

        return normalize_order_result(
            order_id="",
            requested_quantity=quantity,
            filled_quantity=0,
            broker_state="",
            reason=reason,
        )

    def _resolve_indeterminate_placement(
        self,
        quantity: int,
        order_tag: str,
        reason: str,
    ) -> OrderResult:
        """Try the correlation lookup before accepting an unknown placement."""

        found_id, _lookup_succeeded = self._find_order_id_by_tag(order_tag)
        if found_id:
            return self._confirm_fill(found_id, quantity)
        return self._indeterminate_placement(quantity, reason)

    def _order_status(
        self,
        order_id: str,
    ) -> tuple[str | None, int | None, int | None, str]:
        """Return normalized state, filled qty, order qty, and rejection reason."""
        envelope = self._call_api(
            "order status",
            lambda client: client.get_order_by_id(str(order_id)),
            allow_after_poison=True,
        )
        outcome, data, reason = _classify_envelope(envelope)
        if outcome != "ok":
            return (None, None, None, reason)
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if not isinstance(row, dict):
                continue
            return (
                _alias_state(_first_present(row, _ORDER_STATE_KEYS)) or None,
                _exact_int(_first_present(row, _FILLED_QTY_KEYS)),
                _exact_int(_first_present(row, _ORDER_QTY_KEYS)),
                str(_first_present(row, _REASON_KEYS) or "").strip(),
            )
        return (None, None, None, f"Unrecognised Dhan order-status payload: {data!r}")

    def get_order_status(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Return one normalized order-book snapshot.

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
                reason=f"Dhan order-status query failed: {exc}",
            )

        effective_requested = requested or broker_qty or 0
        if requested and broker_qty not in {None, requested}:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=None,
                broker_state=state or "",
                reason=(
                    f"Malformed Dhan status: broker quantity {broker_qty} "
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
        """Poll until the order reaches a truly terminal state, or time out.

        Why the loop keeps polling on PARTIAL/UNKNOWN snapshots: a market order
        is often observed mid-fill (Dhan state ``PART_TRADED`` with some
        quantity already on ``filledQty``).  That is TRANSIENT -- the next 0.5s
        poll usually shows ``TRADED`` -- so returning it as the final outcome
        would freeze every live strategy over a perfectly healthy order.  A
        partial/unknown snapshot becomes the real outcome only when the broker
        label is terminal (the order can never fill further) or the timeout
        below expires.
        """
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_result = normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=0,
            broker_state="",
            reason="Dhan acknowledgement received; fill status not read yet.",
        )
        while time.monotonic() < deadline:
            last_result = self.get_order_status(order_id, requested_quantity=want_qty)
            if last_result.status in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                return last_result
            if last_result.broker_state in TERMINAL_BROKER_STATES:
                # Terminal label with a partial/contradictory quantity snapshot:
                # no further fills are possible, so this IS the final outcome.
                return last_result
            # A failed/malformed status call is already indeterminate. Repeating
            # it inside the same order interaction risks exceeding the deadline
            # without adding evidence, so return immediately with the known id.
            if "query failed" in last_result.reason.lower():
                return last_result
            time.sleep(_FILL_POLL_INTERVAL)
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=last_result.filled_quantity,
            broker_state=last_result.broker_state,
            reason=(
                f"Dhan order {order_id} was not terminal within "
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
                reason="Dhan cancellation was not submitted because login failed.",
            )
        try:
            self._call_api(
                "cancel order",
                lambda client: client.cancel_order(order_id_s),
                is_order=True,
                allow_after_poison=True,
            )
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"Dhan cancellation response is indeterminate: {exc}",
            )
        return self.get_order_status(order_id_s, requested_quantity=requested)

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        """Return pending orders, or an explicit indeterminate query result."""

        try:
            envelope = self._call_api(
                "order list",
                lambda client: client.get_order_list(),
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Dhan open-order query failed: {exc}"
            )
        outcome, data, reason = _classify_envelope(envelope)
        if outcome != "ok":
            return BrokerQueryResult.indeterminate(
                f"Dhan open-order query returned {outcome}: {reason}"
            )
        if not isinstance(data, list):
            return BrokerQueryResult.indeterminate(
                f"Dhan open-order query returned malformed data: {data!r}"
            )

        orders: list[OpenOrder] = []
        for row in data:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Dhan open-order query contained a malformed row."
                )
            raw_state = _alias_state(_first_present(row, _ORDER_STATE_KEYS))
            snapshot = normalize_order_result(
                order_id=_first_present(row, _ORDER_ID_KEYS),
                requested_quantity=_first_present(row, _ORDER_QTY_KEYS),
                filled_quantity=_first_present(row, _FILLED_QTY_KEYS),
                broker_state=raw_state,
                reason="Open-order snapshot",
            )
            symbol = str(_first_present(row, _SYMBOL_KEYS) or "").strip()
            if (
                not snapshot.order_id
                or not symbol
                or "malformed" in snapshot.reason.lower()
                or snapshot.requested_quantity <= 0
            ):
                return BrokerQueryResult.indeterminate(
                    "Dhan open-order query contained incomplete order data."
                )
            if raw_state in {
                "COMPLETE", "COMPLETED", "FILLED", "TRADED", "EXECUTED",
                "REJECTED", "CANCELLED", "CANCELED", "CANCEL", "LAPSED",
            }:
                continue
            if snapshot.remaining_quantity <= 0:
                continue
            orders.append(
                OpenOrder(
                    order_id=snapshot.order_id,
                    symbol=symbol,
                    side=str(_first_present(row, _SIDE_KEYS) or "").strip().upper(),
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
            envelope = self._call_api(
                "positions",
                lambda client: client.get_positions(),
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Dhan open-position query failed: {exc}"
            )
        outcome, data, reason = _classify_envelope(envelope)
        if outcome != "ok":
            return BrokerQueryResult.indeterminate(
                f"Dhan open-position query returned {outcome}: {reason}"
            )
        if not isinstance(data, list):
            return BrokerQueryResult.indeterminate(
                f"Dhan open-position query returned malformed data: {data!r}"
            )

        positions: list[OpenPosition] = []
        for row in data:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Dhan open-position query contained a malformed row."
                )
            # A flat position legitimately reports netQty 0, which
            # ``_first_present`` skips as empty -- so read it directly.
            raw_quantity = next(
                (row[key] for key in _NET_QTY_KEYS if key in row),
                None,
            )
            quantity = _exact_int(raw_quantity)
            symbol = str(_first_present(row, _SYMBOL_KEYS) or "").strip()
            if quantity is None or not symbol:
                return BrokerQueryResult.indeterminate(
                    "Dhan open-position query contained incomplete position data."
                )
            if quantity == 0:
                continue
            positions.append(
                OpenPosition(
                    symbol=symbol,
                    quantity=quantity,
                    product_type=str(_first_present(row, _PRODUCT_KEYS) or "").strip(),
                    broker_state="OPEN",
                )
            )
        return BrokerQueryResult.success(positions)

    def extract_order_id(self, order_response: Any) -> str:
        """Recursively find Dhan's ``orderId`` in a response payload."""
        if isinstance(order_response, OrderResult):
            return order_response.order_id
        if order_response is None:
            return ""
        if isinstance(order_response, dict):
            for key in ("orderId", "order_id", "orderid", "id"):
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

        DhanHQ documents no remote logout call, so closing the SDK's connection
        pool and clearing the client, contract cache, and rate histories is the
        complete local logout operation.  The token itself stays valid until
        its own expiry regardless of what this process does, which is why it
        lives only in the environment and is never written back anywhere.
        """
        with self._lock:
            context = self._context
            if context is not None:
                try:
                    http = context.get_dhan_http()
                    session = getattr(http, "session", None)
                    if session is not None:
                        session.close()
                except Exception as exc:
                    log.warning("Dhan local session close failed: %s", exc)
            self.is_logged_in = False
            self.client = None
            self._context = None
            self._client_code = ""
            self._symbol_cache.clear()
            self._security_id_by_symbol.clear()
            self._scrip_df = None
            self._api_limiter.reset()
            self._order_limiter.reset()
            return {"status": "success", "message": "Dhan local session cleared"}


# The master imports and reuses this one object so all worker threads share the
# same authenticated session, contract cache, lock, and rate-limit budgets.
dhan_execution_client = DhanExecutionClient()
