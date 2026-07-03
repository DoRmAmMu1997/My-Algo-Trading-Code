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
* Every HTTP call has an explicit timeout and every API/order call is rate-limited.
* ``place_market_order`` returns only after ``SingleOrdHist`` confirms the whole
  quantity filled.  Any ambiguity raises so the runner's existing fail-safe path
  can take over rather than pretending a real order succeeded.

Flattrade Pi v2 currently documents no logout endpoint.  ``logout()`` therefore
closes the local HTTP session and erases the in-memory token.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import pandas as pd
import requests

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

_API_TIMEOUT_SECONDS = 30
_SCRIP_MASTER_TIMEOUT_SECONDS = 120
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 2.0

_PRODUCT_MAP = {"INTRADAY": "I", "NORMAL": "M"}
_SIDE_MAP = {"BUY": "B", "SELL": "S"}
_TERMINAL_FAILURE_STATES = {"rejected", "cancelled", "canceled"}


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
    clear failure that activates the runner's paper fallback.
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

    def acquire(self) -> None:
        """Reserve one request slot or raise before a long/stale wait."""
        started = self._clock()
        with self._lock:
            while True:
                now = self._clock()
                minute_cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] <= minute_cutoff:
                    self._timestamps.popleft()

                recent_second = [
                    stamp for stamp in self._timestamps if stamp > now - 1.0
                ]
                second_full = len(recent_second) >= self.per_second
                minute_full = len(self._timestamps) >= self.per_minute

                if not second_full and not minute_full:
                    self._timestamps.append(now)
                    return

                waits = []
                if second_full:
                    waits.append(1.0 - (now - recent_second[0]))
                if minute_full:
                    waits.append(60.0 - (now - self._timestamps[0]))
                wait_seconds = max(0.0, max(waits))
                elapsed = now - started
                if elapsed + wait_seconds > self.max_wait_seconds:
                    raise RuntimeError(
                        f"Flattrade {self.label} rate limit exhausted; "
                        f"safe slot needs {wait_seconds:.2f}s"
                    )
                self._sleep(wait_seconds)

    def reset(self) -> None:
        """Forget local request history after a session is explicitly closed."""
        with self._lock:
            self._timestamps.clear()


class FlattradeExecutionClient:
    """One lazily authenticated, lock-guarded Flattrade execution session."""

    def __init__(self) -> None:
        self.client: requests.Session | None = None
        self.is_logged_in = False
        self._lock = threading.RLock()
        self._access_token = ""
        self._client_id = ""
        self._account_id = ""
        self._scrip_df: pd.DataFrame | None = None
        self._symbol_cache: dict[tuple, str] = {}
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

    def ensure_logged_in(self) -> bool:
        """Return True only when a validated Flattrade daily token is available."""
        with self._lock:
            if self.is_logged_in and self._access_token and self.client is not None:
                return True
            return self._login_locked()

    def _login_locked(self) -> bool:
        """Validate an env token or perform Flattrade's browser request-code flow."""
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
            f"{api_key}{request_code}{raw_secret}".encode("utf-8")
        ).hexdigest()
        try:
            response = self.client.post(
                _TOKEN_URL,
                json={
                    "api_key": api_key,
                    "request_code": request_code,
                    "api_secret": digest,
                },
                timeout=_API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            log.error("Flattrade token exchange failed: %s", exc)
            return False

        if not isinstance(payload, dict) or str(payload.get("status", "")).lower() != "ok":
            message = payload.get("emsg", "unknown error") if isinstance(payload, dict) else "malformed response"
            log.error("Flattrade token exchange rejected: %s", message)
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
    ) -> Any:
        """POST one documented ``jData``/``jKey`` request with limits and timeout."""
        if not self._access_token:
            raise RuntimeError("Flattrade has no validated access token.")
        self._api_limiter.acquire()
        if is_order:
            self._order_limiter.acquire()
        with self._lock:
            session = self._ensure_session_locked()
            body = urlencode(
                {
                    "jData": json.dumps(payload, separators=(",", ":")),
                    "jKey": self._access_token,
                }
            )
            response = session.post(
                f"{_BASE_URL}/{endpoint}",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=_API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            try:
                return response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Flattrade {endpoint} returned malformed JSON"
                ) from exc

    def preload_scrip_master(self) -> bool:
        """Log in and cache Flattrade's official NSE index-derivative CSV."""
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
            response = session.get(
                _NFO_INDEX_MASTER_URL,
                timeout=_SCRIP_MASTER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            raw = pd.read_csv(io.StringIO(response.text))
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
        """Validate official CSV columns and add normalized lookup fields."""
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
        """Return the exact Flattrade ``Tradingsymbol`` or ``""`` on a miss."""
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
    ) -> dict[str, Any]:
        """Place one MKT order and return only after the full fill is confirmed."""
        side_u = str(side).upper().strip()
        product_u = str(product_type).upper().strip()
        exchange_u = str(exchange_segment).upper().strip() or "NFO"
        symbol_s = str(symbol).strip()
        try:
            quantity_i = int(quantity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid quantity: {quantity!r}") from exc
        if side_u not in _SIDE_MAP:
            raise ValueError(f"Invalid side: {side!r}")
        if product_u not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if exchange_u != "NFO":
            raise ValueError(f"Invalid exchange_segment: {exchange_segment!r}")
        if quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not symbol_s:
            raise ValueError("Missing trading symbol for order.")
        if not self.ensure_logged_in():
            raise RuntimeError("Flattrade login failed; cannot place real order.")

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
        response = self._post_api("PlaceOrder", payload, is_order=True)
        if not self._is_order_ack(response):
            raise RuntimeError(f"Flattrade did not acknowledge the order: {response}")
        order_id = self.extract_order_id(response)
        self._confirm_fill(order_id, quantity_i)
        return response

    def _is_order_ack(self, response: Any) -> bool:
        """True only for ``stat=Ok`` payloads carrying a Noren order number."""
        return (
            isinstance(response, dict)
            and str(response.get("stat", "")).lower() == "ok"
            and bool(str(response.get("norenordno") or "").strip())
        )

    def _order_status(self, order_id: str) -> tuple[str | None, int, int, str]:
        """Return normalized state, filled qty, order qty, and rejection reason."""
        response = self._post_api(
            "SingleOrdHist",
            {"uid": self._client_id, "norenordno": str(order_id)},
        )
        rows = response if isinstance(response, list) else [response]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("stat", "Ok")).lower() not in {"", "ok"}:
                continue
            state = str(row.get("status") or "").strip().lower() or None

            def _as_int(value: Any) -> int:
                try:
                    return int(float(value or 0))
                except (TypeError, ValueError):
                    return 0

            return (
                state,
                _as_int(row.get("fillshares")),
                _as_int(row.get("qty")),
                str(row.get("rejreason") or row.get("emsg") or "").strip(),
            )
        reason = response.get("emsg", "unrecognised response") if isinstance(response, dict) else "unrecognised response"
        return (None, 0, 0, str(reason))

    def _confirm_fill(self, order_id: str, want_qty: int) -> None:
        """Poll SingleOrdHist until fully filled, rejected, or timed out."""
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_state = None
        while time.monotonic() < deadline:
            state, filled, order_qty, reason = self._order_status(order_id)
            last_state = state
            if state == "complete":
                if filled >= want_qty:
                    return
                raise RuntimeError(
                    f"Flattrade order {order_id} completed only {filled}/{want_qty}"
                )
            if state in _TERMINAL_FAILURE_STATES:
                raise RuntimeError(
                    f"Flattrade order {order_id} {state}: {reason or 'no reason given'}"
                )
            time.sleep(_FILL_POLL_INTERVAL)
        raise TimeoutError(
            f"Flattrade order {order_id} not confirmed filled within "
            f"{_FILL_TIMEOUT_SECONDS:.0f}s (last status: {last_state})"
        )

    def extract_order_id(self, order_response: Any) -> str:
        """Recursively find Flattrade's ``norenordno`` in a response payload."""
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
        """Close the local session and erase sensitive in-memory state."""
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
