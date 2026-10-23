"""
Thin, deadline-bounded broker clients for the position-group monitor.

What these are for
------------------
The operator carries option-selling positions for weeks in a broker account
that is NOT the runner's ``LIVE_BROKER`` (Zerodha today).  The position-group
monitor (``position_group_monitor.py``) needs exactly three things from that
account: read the open positions, place a closing MARKET order, and prove what
the order did.  This module provides just that surface
(:class:`broker_contract.PositionBrokerClient`) -- nothing that opens exposure,
no contract resolver, and these clients are NOT selectable as ``LIVE_BROKER``.

Why one module and no SDK
-------------------------
Zerodha's Kite Connect v3 and mStock's Type-A API are the same REST shape: the
same ``Authorization: token api_key:access_token`` header, the same
``/portfolio/positions`` and ``/orders/regular`` endpoints, the same
``{"status": "success" | "error", "data": ...}`` envelope and the same
instrument CSV columns.  The official SDKs are thin ``requests`` wrappers, so
this module talks ``requests`` (already a core dependency) directly and keeps
the per-broker differences -- base URL, version header, the order-status path
and one or two order fields -- in small subclasses.

Safety rules (the same ones the four ``LIVE_BROKER`` adapters follow)
--------------------------------------------------------------------
* Importing this module never logs in or sends anything.  Credentials come from
  the environment only, read at login time.
* Every HTTP call carries a native ``requests`` timeout AND one total ten-second
  wall-clock deadline that also covers rate-limit and session-lock waits.
* An acknowledgement is not a fill.  ``place_market_order`` returns only after
  the order book proves an outcome; anything it cannot prove is ``UNKNOWN``.
* A refusal is called ``REJECTED`` only after the order book is searched for the
  order's tag and found not to hold it.  A transport error, a timeout, a 5xx or
  an ``error_type`` that admits the OMS may have acted is ``UNKNOWN``.
* An order call that outlives its deadline poisons the client: no further order
  is accepted from it in this process.  The monitor then stops the group and
  alerts, and the operator reconciles by hand.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar

import requests

# The same dual-import arrangement as the other broker helpers: the repo root
# for ``Dependencies.secret_redaction`` and the Dependencies folder for the bare
# ``broker_contract`` / ``broker_rate_limit`` names.
_DEPENDENCIES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPENDENCIES_DIR.parent
for _import_root in (_REPO_ROOT, _DEPENDENCIES_DIR):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from broker_contract import (  # noqa: E402
    TERMINAL_BROKER_STATES,
    BrokerQueryResult,
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
    # python-dotenv is a core dependency; staying import-safe without it only
    # means login reports the missing credential clearly instead.
    pass


log = logging.getLogger(__name__)

_BROKER_CALL_DEADLINE_SECONDS = 10.0
# The instrument CSV is a few megabytes and is read once, before any order, so
# it gets a longer budget than an order call.  It never runs on an order path.
_INSTRUMENT_DOWNLOAD_DEADLINE_SECONDS = 30.0
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 2.0

_SIDES = frozenset({"BUY", "SELL"})
# Carried positions are NRML; MIS is accepted so an intraday leg the operator
# grouped by hand can still be closed in its own product.
_PRODUCTS = frozenset({"NRML", "MIS"})
_OPTION_EXCHANGES = frozenset({"NFO", "BFO"})

# ``error_type`` values that prove the broker answered and refused, as opposed
# to "the request may have reached the OMS".  Kite documents
# NetworkException as "unable to communicate with the OMS" and GeneralException
# as an unclassified failure -- neither proves nothing was created, so neither
# is here.  mStock reuses Kite's names and adds MiraeException for refusals.
_REFUSAL_ERROR_TYPES = frozenset(
    {
        "INPUTEXCEPTION",
        "ORDEREXCEPTION",
        "TOKENEXCEPTION",
        "PERMISSIONEXCEPTION",
        "MARGINEXCEPTION",
        "APIKEYEXCEPTION",
        "MIRAEEXCEPTION",
    }
)

# A transient order state that a healthy order passes through.  Anything else
# unrecognized is also transient (see ``TERMINAL_BROKER_STATES``); this alias
# table only translates the labels that DO mean something final.
_STATE_ALIASES = {"EXECUTED": "COMPLETE", "TRADED": "COMPLETE"}


def _env_str(name: str) -> str:
    """Read one environment value and remove accidental wrapping quotes."""

    value = str(os.getenv(name) or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _non_negative_price(value: Any) -> float:
    """Return a finite non-negative price, or 0.0 meaning "none reported"."""

    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _parse_expiry(value: Any) -> date | None:
    """Parse the instrument CSV's ``YYYY-MM-DD`` (or ``DD-MM-YYYY``) expiry."""

    text = str(value or "").strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_instrument_csv(text: str) -> dict[tuple[str, str], tuple[str, date, float, str]]:
    """Map ``(exchange, tradingsymbol)`` to ``(underlying, expiry, strike, right)``.

    Both brokers publish the Kite column set (``tradingsymbol``, ``name``,
    ``expiry``, ``strike``, ``instrument_type``, ``exchange``).  Only F&O option
    rows with a parseable expiry and strike are kept: everything the monitor
    cannot price through Dhan's feed is simply absent, and a missing row makes
    the monitor refuse the leg rather than guess.
    """

    index: dict[tuple[str, str], tuple[str, date, float, str]] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        exchange = str(row.get("exchange") or "").strip().upper()
        right = str(row.get("instrument_type") or "").strip().upper()
        if exchange not in _OPTION_EXCHANGES or right not in {"CE", "PE"}:
            continue
        symbol = str(row.get("tradingsymbol") or "").strip()
        underlying = str(row.get("name") or "").strip().upper().strip('"')
        expiry = _parse_expiry(row.get("expiry"))
        try:
            strike = float(row.get("strike") or "")
        except ValueError:
            continue
        if not symbol or not underlying or expiry is None or not math.isfinite(strike):
            continue
        index[(exchange, symbol)] = (underlying, expiry, strike, right)
    return index


class KiteStyleBrokerClient:
    """One lazily authenticated, lock-guarded Kite-shaped REST session.

    Subclasses fill in the broker-specific constants and the two methods whose
    paths differ (:meth:`_order_status_request`, :meth:`_instrument_paths`).
    """

    broker_name = ""
    base_url = ""
    version_header: tuple[str, str] = ("", "")
    api_key_env = ""
    access_token_env = ""
    # A cheap authenticated read used to prove a token works.
    validation_path = ""
    # Per-broker order-placement extras (e.g. Kite's market_protection).
    extra_order_fields: ClassVar[dict[str, str]] = {}

    def __init__(self, *, session: Any = None) -> None:
        self.is_logged_in = False
        self._session: Any = session
        self._owns_session = session is None
        self._api_key = ""
        self._access_token = ""
        self._lock = threading.RLock()
        self._order_submission_lock = threading.Lock()
        # requests timeouts are inactivity limits, not a wall-clock budget, so
        # each call runs on this single worker and the caller waits a bounded
        # time for it (the same arrangement as the Dhan adapter).
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{self.broker_name.lower()}-http"
        )
        self._order_poisoned = False
        self._instruments: dict[tuple[str, str], tuple[str, date, float, str]] | None = None
        # Both brokers publish ~10 orders/second; stay well under it.
        self._api_limiter = RollingWindowRateLimiter(
            5, 200, _MAX_RATE_LIMIT_WAIT_SECONDS, label=f"{self.broker_name} API"
        )

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        name, value = self.version_header
        return {
            name: value,
            "Authorization": f"token {self._api_key}:{self._access_token}",
        }

    def _request(
        self,
        operation: str,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        deadline_seconds: float | None = None,
        is_order: bool = False,
    ) -> requests.Response:
        """Run one HTTP call under the shared limiter, lock and total deadline.

        ``deadline_seconds`` defaults to the ten-second broker budget, read at
        call time rather than bound at definition time.

        Raises:
            TimeoutError: the total budget expired (an order call also poisons
                the client, because the order may still reach the exchange).
            RuntimeError: no authenticated session exists.
            requests.RequestException: a transport failure.
        """

        if deadline_seconds is None:
            deadline_seconds = _BROKER_CALL_DEADLINE_SECONDS
        started = time.monotonic()
        deadline = started + deadline_seconds
        self._api_limiter.acquire(deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(
                f"{self.broker_name} {operation} deadline expired waiting for the session lock."
            )
        try:
            session = self._session
            if session is None:
                raise RuntimeError(f"{self.broker_name} has no HTTP session.")
            url = f"{self.base_url}{path}"
            headers = self._headers()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.broker_name} {operation} deadline expired.")
            future: Future[requests.Response] = self._executor.submit(
                session.request,
                method,
                url,
                params=params,
                data=data,
                headers=headers,
                timeout=deadline_seconds,
            )
            try:
                return future.result(timeout=remaining)
            except FuturesTimeoutError as exc:
                if future.done():
                    return future.result()
                future.cancel()
                if is_order:
                    self._order_poisoned = True
                raise TimeoutError(
                    f"{self.broker_name} {operation} exceeded the total "
                    f"{deadline_seconds:.0f}s deadline; outcome is indeterminate."
                ) from exc
        finally:
            self._lock.release()

    def _classify(self, response: Any) -> tuple[str, Any, str]:
        """Split a response into ``("ok"|"refused"|"indeterminate", data, reason)``.

        Only a parsed ``status: error`` body with a refusal ``error_type`` and a
        4xx status counts as "refused"; everything else that is not a clean
        success is indeterminate.
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
        if status == "error" and 400 <= status_code < 500 and error_type in _REFUSAL_ERROR_TYPES:
            return ("refused", body.get("data"), reason)
        return ("indeterminate", body.get("data"), reason)

    def _secrets(self) -> tuple[str, ...]:
        return tuple(value for value in (self._api_key, self._access_token) if value)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def ensure_logged_in(self) -> bool:
        """Return True only when today's token has been proven against the API."""

        with self._lock:
            if self.is_logged_in:
                return True
            self._api_key = _env_str(self.api_key_env)
            self._access_token = _env_str(self.access_token_env)
            if not self._api_key or not self._access_token:
                # Only the broker's name is logged: nothing derived from a
                # credential, not even which of the two is the blank one.
                log.error(
                    "%s login disabled: its API key or access token is missing from .env.",
                    self.broker_name,
                )
                return False
            if self._session is None:
                self._session = requests.Session()
                self._owns_session = True
        try:
            response = self._request("session validation", "GET", self.validation_path)
        except Exception as exc:
            log.error(
                "%s session validation failed: %s",
                self.broker_name,
                redact_text(exc, self._secrets()),
            )
            return False
        outcome, _data, reason = self._classify(response)
        if outcome != "ok":
            log.error(
                "%s session validation rejected (%s): %s. Re-run "
                "'python algo.py setup-broker-token --broker %s' if the token expired.",
                self.broker_name,
                outcome,
                reason,
                self.broker_name.lower(),
            )
            return False
        self.is_logged_in = True
        log.info("%s position client login successful.", self.broker_name)
        return True

    def logout(self) -> dict[str, Any]:
        """Drop the local session.  The token itself stays valid until expiry."""

        with self._lock:
            if self._owns_session and self._session is not None:
                try:
                    self._session.close()
                except Exception as exc:
                    log.warning("%s session close failed: %s", self.broker_name, exc)
                self._session = None
            self.is_logged_in = False
            self._access_token = ""
            self._instruments = None
            return {"status": "success", "message": f"{self.broker_name} local session cleared"}

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def _instrument_paths(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _load_instruments(self) -> dict[tuple[str, str], tuple[str, date, float, str]]:
        """Download and cache the option rows of the broker's instrument master.

        A failure leaves an empty map, which makes every position come back
        without a contract identity -- unmonitorable, never mis-identified.
        """

        if self._instruments is not None:
            return self._instruments
        index: dict[tuple[str, str], tuple[str, date, float, str]] = {}
        for path in self._instrument_paths():
            try:
                response = self._request(
                    "instrument master",
                    "GET",
                    path,
                    deadline_seconds=_INSTRUMENT_DOWNLOAD_DEADLINE_SECONDS,
                )
                if int(getattr(response, "status_code", 0) or 0) != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                index.update(parse_instrument_csv(response.text))
            except Exception as exc:
                log.error(
                    "%s instrument master %s could not be read: %s",
                    self.broker_name,
                    path,
                    redact_text(exc, self._secrets()),
                )
                return {}
        self._instruments = index
        log.info("%s instrument master: %d option rows.", self.broker_name, len(index))
        return index

    def list_position_details(self) -> BrokerQueryResult[PositionDetail]:
        """Return every non-flat net position, or an explicit indeterminate result."""

        if not self.ensure_logged_in():
            return BrokerQueryResult.indeterminate(f"{self.broker_name} login failed.")
        try:
            response = self._request("positions", "GET", "/portfolio/positions")
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"{self.broker_name} position query failed: {redact_text(exc, self._secrets())}"
            )
        outcome, data, reason = self._classify(response)
        if outcome != "ok":
            return BrokerQueryResult.indeterminate(
                f"{self.broker_name} position query returned {outcome}: {reason}"
            )
        rows = data.get("net") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return BrokerQueryResult.indeterminate(
                f"{self.broker_name} position query returned no 'net' list."
            )
        instruments = self._load_instruments()
        positions: list[PositionDetail] = []
        for row in rows:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    f"{self.broker_name} position query contained a malformed row."
                )
            quantity = exact_int(row.get("quantity"))
            symbol = str(row.get("tradingsymbol") or "").strip()
            exchange = str(row.get("exchange") or "").strip().upper()
            if quantity is None or not symbol or not exchange:
                return BrokerQueryResult.indeterminate(
                    f"{self.broker_name} position query contained incomplete data."
                )
            if quantity == 0:
                continue
            identity = instruments.get((exchange, symbol))
            underlying, expiry, strike, right = identity if identity else ("", None, None, "")
            positions.append(
                PositionDetail(
                    symbol=symbol,
                    exchange=exchange,
                    product_type=str(row.get("product") or "").strip().upper(),
                    quantity=quantity,
                    average_price=_non_negative_price(row.get("average_price")),
                    underlying=underlying,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                )
            )
        return BrokerQueryResult.success(positions)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str,
        product_type: str,
        *,
        order_tag: str = "",
    ) -> OrderResult:
        """Place one MARKET order and return only a proven outcome.

        ``exchange_segment`` is the broker's own exchange code (``NFO``/``BFO``).
        Orders are serialized: a second order never starts while the first one's
        outcome is still being read.
        """

        side_u = str(side).strip().upper()
        product_u = str(product_type).strip().upper()
        exchange_u = str(exchange_segment).strip().upper()
        symbol_s = str(symbol).strip()
        quantity_i = exact_int(quantity)
        if side_u not in _SIDES:
            raise ValueError(f"Invalid side: {side!r}")
        if product_u not in _PRODUCTS:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if exchange_u not in _OPTION_EXCHANGES:
            raise ValueError(f"Invalid exchange_segment: {exchange_segment!r}")
        if quantity_i is None or quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not symbol_s:
            raise ValueError("Missing trading symbol for order.")
        if not self._order_submission_lock.acquire(timeout=_BROKER_CALL_DEADLINE_SECONDS):
            return self._unproven(quantity_i, "ORDER_GATE_TIMEOUT", "Another order is unresolved.")
        try:
            if self._order_poisoned:
                return self._unproven(
                    quantity_i,
                    "SESSION_POISONED",
                    f"{self.broker_name} session is indeterminate after an earlier "
                    "order deadline; reconcile by hand before another order.",
                )
            if not self.ensure_logged_in():
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason=f"Order was not submitted because {self.broker_name} login failed.",
                )
            return self._place_serialized(
                symbol_s, side_u, quantity_i, exchange_u, product_u, order_tag
            )
        finally:
            self._order_submission_lock.release()

    def _place_serialized(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange: str,
        product: str,
        order_tag: str,
    ) -> OrderResult:
        form = {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": side,
            "order_type": "MARKET",
            "quantity": str(quantity),
            "product": product,
            "validity": "DAY",
            **self.extra_order_fields,
        }
        if order_tag:
            form["tag"] = order_tag
        try:
            response = self._request(
                "place order", "POST", "/orders/regular", data=form, is_order=True
            )
        except Exception as exc:
            return self._resolve_by_tag(
                quantity,
                order_tag,
                f"{self.broker_name} place-order call is indeterminate: "
                f"{redact_text(exc, self._secrets())}",
                refused=False,
            )
        outcome, data, reason = self._classify(response)
        order_id = str(data.get("order_id") or "").strip() if isinstance(data, dict) else ""
        if outcome == "ok" and order_id:
            return self._confirm_fill(order_id, quantity)
        return self._resolve_by_tag(
            quantity,
            order_tag,
            reason or f"{self.broker_name} acknowledged the order without an order id.",
            refused=outcome == "refused",
        )

    def _resolve_by_tag(
        self, quantity: int, order_tag: str, reason: str, *, refused: bool
    ) -> OrderResult:
        """Search the order book for ``order_tag`` before concluding anything.

        A refusal becomes ``REJECTED`` only when the search succeeded and found
        no such order; every other path stays ``UNKNOWN``.
        """

        found_id, searched = self._find_order_id_by_tag(order_tag)
        if found_id:
            return self._confirm_fill(found_id, quantity)
        if refused and searched:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity,
                filled_quantity=0,
                broker_state="REJECTED",
                reason=f"{self.broker_name} rejected the order: {reason}",
            )
        return self._unproven(quantity, "", reason)

    def _find_order_id_by_tag(self, order_tag: str) -> tuple[str, bool]:
        """Return ``(order_id, search_succeeded)`` for an order carrying the tag."""

        if not order_tag:
            return ("", False)
        try:
            response = self._request("order book", "GET", "/orders")
        except Exception as exc:
            log.warning(
                "%s order-book lookup failed: %s",
                self.broker_name,
                redact_text(exc, self._secrets()),
            )
            return ("", False)
        outcome, data, _reason = self._classify(response)
        if outcome != "ok" or not isinstance(data, list):
            return ("", False)
        for row in data:
            if not isinstance(row, dict):
                return ("", False)
            tags = row.get("tag")
            tag_values = tags if isinstance(tags, list) else [tags]
            if order_tag in {str(tag or "").strip() for tag in tag_values}:
                return (str(row.get("order_id") or "").strip(), True)
        return ("", True)

    @staticmethod
    def _unproven(quantity: int, broker_state: str, reason: str) -> OrderResult:
        return normalize_order_result(
            order_id="",
            requested_quantity=quantity,
            filled_quantity=0,
            broker_state=broker_state,
            reason=reason,
        )

    def _order_status_request(self, order_id: str) -> tuple[str, dict[str, Any] | None]:
        """Return ``(path, params)`` for one order's history.  Per broker."""

        raise NotImplementedError

    def get_order_status(self, order_id: str, requested_quantity: int = 0) -> OrderResult:
        """Return one normalized order snapshot; failures are ``UNKNOWN``, never rejection."""

        requested = exact_int(requested_quantity)
        if requested is None or requested < 0:
            requested = 0
        path, params = self._order_status_request(str(order_id))
        try:
            response = self._request("order status", "GET", path, params=params)
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"{self.broker_name} order-status query failed: "
                f"{redact_text(exc, self._secrets())}",
            )
        outcome, data, reason = self._classify(response)
        rows = data if isinstance(data, list) else [data]
        # Kite returns the order's whole state history; the LAST entry is now.
        latest = next((row for row in reversed(rows) if isinstance(row, dict)), None)
        if outcome != "ok" or latest is None:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"{self.broker_name} order-status query failed: "
                f"{reason or 'no order row'}",
            )
        state = str(latest.get("status") or "").strip().upper()
        state = _STATE_ALIASES.get(state, state)
        broker_qty = exact_int(latest.get("quantity"))
        if requested and broker_qty not in {None, requested}:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=None,
                broker_state=state,
                reason=(
                    f"Malformed {self.broker_name} status: broker quantity {broker_qty} "
                    f"does not match requested quantity {requested}."
                ),
            )
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=requested or broker_qty or 0,
            filled_quantity=latest.get("filled_quantity"),
            broker_state=state,
            reason=redact_text(str(latest.get("status_message") or "").strip(), self._secrets()),
            average_fill_price=latest.get("average_price"),
        )

    def _confirm_fill(self, order_id: str, want_qty: int) -> OrderResult:
        """Poll until the order is terminal, or report ``UNKNOWN`` at the timeout.

        A mid-fill PARTIAL/OPEN snapshot is transient and keeps the loop polling;
        it becomes the outcome only with a terminal label or at the timeout.
        """

        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last = self._unproven(want_qty, "", "Acknowledged; fill not read yet.")
        while time.monotonic() < deadline:
            last = self.get_order_status(order_id, requested_quantity=want_qty)
            if last.status in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                return last
            if last.broker_state in TERMINAL_BROKER_STATES:
                return last
            if "query failed" in last.reason.lower():
                return last
            time.sleep(_FILL_POLL_INTERVAL)
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=last.filled_quantity,
            broker_state=last.broker_state,
            reason=(
                f"{self.broker_name} order {order_id} was not terminal within "
                f"{_FILL_TIMEOUT_SECONDS:.0f}s; exposure is indeterminate."
            ),
            average_fill_price=last.average_fill_price,
        )


class ZerodhaPositionClient(KiteStyleBrokerClient):
    """Zerodha Kite Connect v3 (the free personal plan covers everything here)."""

    broker_name = "ZERODHA"
    base_url = "https://api.kite.trade"
    version_header = ("X-Kite-Version", "3")
    api_key_env = "ZERODHA_API_KEY"
    access_token_env = "ZERODHA_ACCESS_TOKEN"
    validation_path = "/user/profile"
    # Exchanges now reject API MARKET orders without market protection; -1 asks
    # Kite to apply its automatic protection band.
    extra_order_fields: ClassVar[dict[str, str]] = {"market_protection": "-1"}

    def _instrument_paths(self) -> tuple[str, ...]:
        return ("/instruments/NFO", "/instruments/BFO")

    def _order_status_request(self, order_id: str) -> tuple[str, dict[str, Any] | None]:
        return (f"/orders/{order_id}", None)


class MStockPositionClient(KiteStyleBrokerClient):
    """Mirae Asset mStock Type-A API (a Kite-shaped clone with its own paths).

    Differences from Kite that matter here: the order-status endpoint is
    ``/order/details?order_no=...&segment=D`` rather than ``/orders/<id>``, the
    instrument master is one all-exchange CSV, there is no market-protection
    field, and MARKET orders still carry ``price=0``.  mStock does not publish
    its full order-status vocabulary; any label this module does not recognize
    stays transient and ends as ``UNKNOWN`` at the fill timeout -- never as a
    guessed fill or rejection.
    """

    broker_name = "MSTOCK"
    base_url = "https://api.mstock.trade/openapi/typea"
    version_header = ("X-Mirae-Version", "1")
    api_key_env = "MSTOCK_API_KEY"
    access_token_env = "MSTOCK_ACCESS_TOKEN"
    validation_path = "/portfolio/positions"
    extra_order_fields: ClassVar[dict[str, str]] = {"price": "0"}

    def _instrument_paths(self) -> tuple[str, ...]:
        return ("/instruments/scriptmaster",)

    def _order_status_request(self, order_id: str) -> tuple[str, dict[str, Any] | None]:
        return ("/order/details", {"order_no": order_id, "segment": "D"})


# Constructed lazily by the monitor; one client per broker per process.
POSITION_CLIENT_CLASSES: dict[str, Callable[[], KiteStyleBrokerClient]] = {
    "ZERODHA": ZerodhaPositionClient,
    "MSTOCK": MStockPositionClient,
}
