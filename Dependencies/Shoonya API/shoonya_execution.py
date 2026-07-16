"""
Shared Shoonya (Finvasia NorenApi) order-execution layer for the Multithreading runner.

============================== Plain-English overview ==========================
This file is the ONE place that talks to the Shoonya (Finvasia) broker to place
real orders. The strategies decide *what* to trade; this module knows *how* to
send that order to Shoonya. When a strategy is in "paper" mode it never touches
this file at all; only "real"/live strategies do.

It is a drop-in replacement for the old Kotak Neo execution layer: it exposes the
SAME broker-neutral contract the master runner calls -- login/symbol resolution,
typed place/status/cancel outcomes, explicit open-order/open-position query
results, order-id extraction, logout, and the `is_logged_in` flag -- so the rest
of the runner stays broker-agnostic.

Glossary for newcomers (the broker world is full of jargon):
- TOTP   : the 6-digit code from an authenticator app that changes every 30s.
           Shoonya 2FA can be AUTOMATED: we store the TOTP *secret seed* and
           compute the current code with `pyotp`, so no typing at startup.
- 2FA    : two-factor authentication (here: password + TOTP factor2).
- vendor_code / api_secret / imei : the Shoonya API app credentials.
- lot    : options trade in fixed bundles ("lots"), e.g. 1 NIFTY lot = 75 units.
           quantity = lot_size * number_of_lots.
- MIS / NRML : product types. MIS = intraday (Shoonya code "I"); NRML =
           carry-forward (Shoonya code "M").
- tsym   : Shoonya's trading symbol for a contract, e.g. "NIFTY26JUN25C23950".
           Unlike Kotak's opaque pTrdSymbol it is DERIVABLE from the contract:
           <UNDERLYING><DDMMMYY><C|P><STRIKE> (proven by helper_shoonya.py).
- thread / lock : the runner uses many parallel "threads" (one per strategy). A
           "lock" makes sure only one thread talks to Shoonya at a time, because
           the session is not safe to use from two threads at once.
- lazy login : we only log in the first time it is actually needed, not at import.
===============================================================================

Design rules (mirrors the previous Dependencies/kotak_execution.py):
- This file owns ONLY order-side concerns. No strategy logic, no market-data
  logic. The master runner still resolves every contract from DhanHQ and passes
  the underlying/expiry/strike/option-type through; we turn that into a Shoonya
  tsym and place the order.

Why a dedicated module:
- The Shoonya NorenApi client lives in the top-level "Shoonya API/" folder
  (vendored from the existing repo). We add that folder to `sys.path` here so
  `from NorenApi import NorenApi` resolves without a global pip install.
- 24 strategy workers run as threads sharing ONE process. This wrapper holds a
  single authenticated session behind a lock, logs in lazily on the first real
  order, and is therefore safe to call from any worker thread.

How the master file uses this module:
1. `from Dependencies.shoonya_execution import shoonya_execution_client` (guarded
   by try/except so pure-paper runs work even without the SDK / deps installed).
2. Workers call `shoonya_execution_client.place_market_order(...)` on a real
   entry/exit leg. Login happens automatically on the first such call.
3. `logout()` is called during graceful shutdown.
"""

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
from getpass import getpass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from Dependencies.secret_redaction import redact_payload, redact_text

# Make the broker-neutral contract available both when the master dynamically
# loads this file and when the sibling diagnostic runs it as a standalone script.
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

# --- Make the vendored "Shoonya API" SDK importable ---------------------------
# The NorenApi client may live in a "Shoonya API" folder somewhere above this
# file, but the depth differs by repo layout. Rather than hardcode a parents[]
# index, walk UP the directory tree and use the first "Shoonya API" folder found.
# If none exists (the module was pip installed instead) we skip this and rely on
# the plain `from NorenApi import NorenApi`.
for _ancestor in Path(__file__).resolve().parents:
    _sdk_dir = _ancestor / "Shoonya API"
    if _sdk_dir.exists():
        if str(_sdk_dir) not in sys.path:
            sys.path.insert(0, str(_sdk_dir))
        break

# Defensively load the runner's .env so credentials are available even when this
# module is imported/used outside the master file (e.g. in unit tests). The
# master file also loads this same .env with override=False, so this is a no-op
# in the normal run.
try:
    from dotenv import load_dotenv

    # .env lives one level up in Dependencies/ (this file is in Dependencies/Shoonya API/).
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

from NorenApi import NorenApi  # noqa: E402  (resolved from "Shoonya API")

# Route status/errors through the logging system so they land in the master
# runner's log file/handlers (with timestamps + levels), not just stdout.
log = logging.getLogger(__name__)


# Shoonya's API expects short codes, not friendly words. These dicts translate
# the readable values our code uses (left) into the exact codes Shoonya wants
# (right). Example: we say "BUY", Shoonya wants "B"; "INTRADAY" -> "I".
_PRODUCT_MAP = {"INTRADAY": "I", "NORMAL": "M"}
_SIDE_MAP = {"BUY": "B", "SELL": "S"}

# Where Shoonya publishes the full NSE F&O contract list (used only to VALIDATE
# the trading symbols we construct -- order placement does not require it).
_NFO_SYMBOL_MASTER_URL = "https://api.shoonya.com/NFO_symbols.txt.zip"

# Fill confirmation: place_order only ACKNOWLEDGES that the request was accepted
# -- the broker can still reject it or leave it unfilled. After an ack we poll
# single_order_history until the order reaches a terminal state. Market orders
# normally fill in well under a second; we wait a few seconds then give up.
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
_BROKER_TIMEOUT_SECONDS = 10.0
_BROKER_CALL_DEADLINE_SECONDS = 10.0
# Shoonya order-status values, lower-cased.
_FILLED_ORDER_STATES = {"complete", "completed"}
_FAILED_ORDER_STATES = {"rejected", "cancelled", "canceled"}


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
    """Return True only for Shoonya's explicit, determinate empty-book reply."""

    if not isinstance(value, dict):
        return False
    state = str(value.get("stat") or "").strip().lower().replace("_", "")
    message = " ".join(
        str(value.get("emsg") or value.get("msg") or "").strip().lower().split()
    )
    return state in {"notok", "rejected"} and message == "no data"


class ShoonyaExecutionClient:
    """
    One shared "phone line" to Shoonya that every live strategy uses.

    Think of an instance of this class as a single logged-in Shoonya session that
    the whole program shares. It is:
    - "lazy-login": it only logs in the first time someone actually needs it.
    - "thread-safe": many strategy threads can call it at once; a lock makes sure
      only one of them is talking to Shoonya at any moment. Orders are infrequent,
      so this queuing has no real cost.

    The module creates ONE instance at the bottom (`shoonya_execution_client`) and
    everything imports and reuses that single object.
    """

    def __init__(self) -> None:
        # The live Shoonya SDK object. Stays None until we successfully log in.
        self.client: NorenApi | None = None
        # Simple flag so we don't re-login every call once we're authenticated.
        self.is_logged_in = False
        # The "one thread at a time" gate for login + orders + symbol lookups.
        self._lock = threading.RLock()
        # Noren's timeout is an inactivity limit, not a total wall-clock budget.
        # A single native worker plus a caller-side deadline covers lock wait and
        # slow-dribble responses without allowing overlapping session calls.
        self._native_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="shoonya-native",
        )
        self._native_poisoned = False
        self._timed_out_future: Future[Any] | None = None
        # Keep one order's acknowledgement and fill confirmation atomic at the
        # adapter boundary. Otherwise another strategy can submit in the small
        # gap before a timed-out status call poisons the shared session.
        self._order_submission_lock = threading.Lock()
        # Remember symbols we've already resolved, so we don't rebuild/re-validate
        # for the same contract. Key = (underlying, expiry "DDMMMYY", "CE"/"PE",
        # strike-as-int) -> Shoonya tsym string.
        self._symbol_cache: dict[tuple, str] = {}
        # The set of valid NFO trading symbols (uppercased), downloaded once and
        # reused for validation. Stays None until first loaded.
        self._symbol_set: set | None = None

    @staticmethod
    def _remaining_broker_budget(started: float) -> float:
        """Return seconds left in one complete native-call wall-clock budget."""

        return _BROKER_CALL_DEADLINE_SECONDS - (time.monotonic() - started)

    def _native_call(
        self,
        operation: str,
        call: Callable[[], Any],
        *,
        started: float | None = None,
        allow_after_poison: bool = False,
    ) -> Any:
        """Run one Noren/HTTP call within one budget including session-lock wait.

        A timed-out native function may still own a socket. The session is then
        poisoned so another strategy cannot queue a second order behind work
        whose broker outcome is unknown.
        """

        call_started = time.monotonic() if started is None else started
        remaining = self._remaining_broker_budget(call_started)
        if remaining <= 0 or not self._lock.acquire(timeout=max(0.0, remaining)):
            raise TimeoutError(
                "Shoonya broker deadline expired while waiting for the shared session lock."
            )
        try:
            if self._native_poisoned:
                abandoned_finished = (
                    self._timed_out_future is not None
                    and self._timed_out_future.done()
                )
                if not allow_after_poison or not abandoned_finished:
                    raise RuntimeError(
                        "Shoonya session is indeterminate after a broker deadline; "
                        "new orders are blocked pending reconciliation."
                    )
            remaining = self._remaining_broker_budget(call_started)
            if remaining <= 0:
                raise TimeoutError(
                    f"Shoonya {operation} deadline expired before native submission."
                )
            future = self._native_executor.submit(call)
            try:
                return future.result(timeout=remaining)
            except FuturesTimeoutError as exc:
                if future.done():
                    return future.result()
                self._native_poisoned = True
                self._timed_out_future = future
                future.cancel()
                raise TimeoutError(
                    f"Shoonya {operation} exceeded the total "
                    f"{_BROKER_CALL_DEADLINE_SECONDS:.0f}s broker deadline; "
                    "session state is indeterminate."
                ) from exc
        finally:
            self._lock.release()

    def recover_after_reconciliation(self) -> bool:
        """Explicitly clear deadline poison after the abandoned call has ended.

        "Poisoned" means an earlier call outlived its 10-second budget and was
        abandoned -- but Python cannot kill a running thread, so that call may
        STILL be executing inside the SDK and may still place/affect an order.
        Only after (a) reconciliation has proven the account state and (b) the
        abandoned call has actually finished is it safe to accept new orders;
        this method refuses (returns False) until both hold.
        """

        with self._lock:
            if self._timed_out_future is not None and not self._timed_out_future.done():
                return False
            self._native_poisoned = False
            self._timed_out_future = None
            return True

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_totp(secret: str) -> str:
        """
        Compute the current 6-digit TOTP from a stored secret seed via pyotp.

        Returns "" if the secret is blank, pyotp is unavailable, or generation
        fails (the caller then falls back to an interactive prompt / aborts).
        """
        secret = str(secret).strip().replace(" ", "")
        if not secret:
            return ""
        try:
            import pyotp
        except Exception as exc:
            log.warning(f"Shoonya TOTP secret is set but pyotp is not installed: {exc}")
            return ""
        try:
            return pyotp.TOTP(secret).now()
        except Exception as exc:
            log.error(f"Shoonya TOTP generation failed: {exc}")
            return ""

    @staticmethod
    def _prompt_totp() -> str:
        """
        Interactively read the current 6-digit TOTP from the authenticator app.

        Used only when no SHOONYA_TOTP_SECRET is configured. Returns "" if no
        input is available (e.g. a headless run), which the caller treats as an
        aborted login -> paper fallback.
        """
        try:
            return getpass("Enter Shoonya TOTP (input hidden): ").strip()
        except (EOFError, OSError):
            return ""

    def _login_locked(self) -> bool:
        """
        Do the actual Shoonya login and return True on success. "_locked" means
        the caller must already hold self._lock (so two threads never log in at
        the same time).

        Flow: read credentials from .env -> get the 2FA code (auto from the
        stored secret, else interactive) -> call NorenApi.login -> verify the
        response carries a real session token (stat == "Ok").
        """
        try:
            userid = (os.getenv("SHOONYA_USERID") or "").strip()
            password = (os.getenv("SHOONYA_PASSWORD") or "").strip()
            vendor_code = (os.getenv("SHOONYA_VENDOR_CODE") or "").strip()
            api_secret = (os.getenv("SHOONYA_API_SECRET") or "").strip()
            imei = (os.getenv("SHOONYA_IMEI") or "abc1234").strip() or "abc1234"
            totp_secret = (os.getenv("SHOONYA_TOTP_SECRET") or "").strip()

            # Fail early (with a clear message) if any required key is blank.
            missing = []
            if not userid:
                missing.append("SHOONYA_USERID")
            if not password:
                missing.append("SHOONYA_PASSWORD")
            if not vendor_code:
                missing.append("SHOONYA_VENDOR_CODE")
            if not api_secret:
                missing.append("SHOONYA_API_SECRET")
            if missing:
                raise ValueError("Missing Shoonya env keys: " + ", ".join(missing))

            # Auto-generate the 2FA code from the stored secret seed; if none is
            # configured, fall back to an interactive prompt (typed 6-digit code).
            two_fa = self._generate_totp(totp_secret) if totp_secret else self._prompt_totp()
            if not two_fa:
                self.is_logged_in = False
                self.client = None
                log.error("Shoonya login aborted: no TOTP available "
                      "(set SHOONYA_TOTP_SECRET, or enter a code when prompted).")
                return False

            # Initialise the client and perform the login handshake.
            self.client = NorenApi()
            native_client = self.client
            resp = self._native_call(
                "login",
                lambda: native_client.login(
                    userid=userid,
                    password=password,
                    twoFA=two_fa,
                    vendor_code=vendor_code,
                    api_secret=api_secret,
                    imei=imei,
                ),
            )

            # NorenApi.login returns the response dict (with susertoken) on
            # success and None on failure; treat anything without stat == "Ok" as
            # a failed login so the caller falls back to paper.
            if not isinstance(resp, dict) or str(resp.get("stat", "")).strip().lower() != "ok":
                self.is_logged_in = False
                self.client = None
                secrets = (userid, password, two_fa, vendor_code, api_secret, imei)
                log.error(
                    "Shoonya login failed (no valid session). Response: %s",
                    redact_payload(resp, secrets),
                )
                return False

            self.is_logged_in = True
            log.info(
                "Shoonya execution client login successful "
                f"(uid={userid}, name={resp.get('uname', '')!r})."
            )
            return True
        except Exception as exc:
            # Reset state on any failure so a stale half-session never leaks.
            self.is_logged_in = False
            self.client = None
            # NorenApi.login() does json.loads() on the raw reply with no status
            # check, so a Shoonya-side HTTP error (e.g. a 502 Bad Gateway or a
            # maintenance HTML page) surfaces here as a cryptic JSON decode error.
            # Add a clear hint so this transient, server-side condition is obvious.
            hint = ""
            if type(exc).__name__ == "JSONDecodeError" or "Expecting value" in str(exc):
                hint = (" -- Shoonya returned a non-JSON response (e.g. an HTTP 502 / "
                        "maintenance page), which usually means the broker API is "
                        "temporarily down. This is server-side; retry in a few minutes.")
            log.error("Shoonya execution client login failed: %s%s", redact_text(exc), hint)
            return False

    def ensure_logged_in(self) -> bool:
        """
        Make sure we have a live Shoonya session, logging in the first time only.

        Call this before any order/lookup. If we're already logged in it returns
        immediately; otherwise it performs the one-time login. Returns True when a
        usable session exists, False if login failed (caller falls back to paper).
        """
        with self._lock:  # only one thread may log in at a time
            if self.is_logged_in and self.client is not None:
                return True  # already authenticated - nothing to do
            return self._login_locked()

    # ------------------------------------------------------------------
    # Symbol resolution (contract -> Shoonya tsym)
    # ------------------------------------------------------------------
    def preload_scrip_master(self) -> bool:
        """
        Download + cache the NSE F&O symbol master NOW (e.g. at startup) so the
        first live order can be VALIDATED without paying a download cost mid-
        session. Unlike Kotak this is optional: symbols are derivable, so a failed
        download just disables validation (orders are still constructed + sent).
        Returns True if the master is loaded.
        """
        if not self.ensure_logged_in():
            return False
        with self._lock:
            return self._ensure_scrip_master_locked()

    def _ensure_scrip_master_locked(self) -> bool:
        """
        Download Shoonya's full NSE F&O symbol list ONCE and keep the set of valid
        trading symbols in memory. "_locked" => the caller already holds
        self._lock. Returns True if the set is ready. (Caller holds _lock.)
        """
        if self._symbol_set is not None:
            return True  # already downloaded earlier in this run
        try:
            # Download through requests so the symbol-master interaction has the
            # same ten-second native HTTP deadline as every Noren API call.
            response = self._native_call(
                "scrip-master download",
                lambda: requests.get(
                    _NFO_SYMBOL_MASTER_URL,
                    timeout=_BROKER_TIMEOUT_SECONDS,
                ),
            )
            response.raise_for_status()
            df = pd.read_csv(io.BytesIO(response.content), compression="zip")
            df = df.rename(columns=lambda c: str(c).strip())  # trim stray spaces
        except Exception as exc:
            log.warning(f"Shoonya NFO symbol master download/parse failed: {exc}")
            return False
        if "TradingSymbol" not in df.columns:
            log.warning(f"Shoonya NFO symbol master missing 'TradingSymbol' column: {list(df.columns)}")
            return False
        try:
            self._symbol_set = set(df["TradingSymbol"].astype(str).str.strip().str.upper())
            log.info(f"Shoonya NSE F&O symbol master loaded ({len(df)} rows).")
            return True
        except Exception as exc:
            log.warning(f"Shoonya NFO symbol master parse failed: {exc}")
            return False

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry,
        option_type: str,
        strike: float,
        exchange_segment: str = "NFO",
    ) -> str:
        """
        Resolve a Shoonya `tsym` for an option contract, cached.

        Shoonya's place_order expects `tradingsymbol` in the form
        <UNDERLYING><DDMMMYY><C|P><STRIKE>, e.g. "NIFTY26JUN25C23950" (proven by
        helper_shoonya.getOptionFormat). We BUILD that string here. If the symbol
        master has been preloaded we also VALIDATE the contract exists (returning
        "" on a miss, so the caller falls back to paper for that leg, mirroring
        the old Kotak safety). If the master is not loaded we trust the construction.

        Inputs: underlying e.g. "NIFTY"; expiry as a date object; option_type
        "CE"/"PE"; strike e.g. 23950.
        """
        # Guard against obviously bad inputs before doing any work.
        if expiry is None or strike is None or float(strike) <= 0:
            log.warning(f"Shoonya resolve skipped (bad inputs): expiry={expiry!r} strike={strike!r}")
            return ""
        underlying = str(underlying).strip().upper()
        option_type = str(option_type).strip().upper()
        if option_type not in ("CE", "PE"):
            log.warning(f"Shoonya resolve skipped (bad option_type): {option_type!r}")
            return ""
        try:
            # Shoonya uses a 2-digit year, zero-padded day: "26JUN25".
            expiry_str = expiry.strftime("%d%b%y").upper()
        except Exception:
            log.warning(f"Shoonya resolve skipped (expiry not a date): expiry={expiry!r} "
                  f"type={type(expiry).__name__}")
            return ""
        int_strike = round(float(strike))
        cache_key = (underlying, expiry_str, option_type, int_strike)

        # Fast path: have we already resolved this exact contract today?
        with self._lock:
            cached = self._symbol_cache.get(cache_key)
            if cached is not None:
                return cached

        # Build the trading symbol: e.g. "NIFTY" + "26JUN25" + "C" + "23950".
        tsym = f"{underlying}{expiry_str}{option_type[0]}{int_strike}"

        with self._lock:
            cached = self._symbol_cache.get(cache_key)  # re-check: another thread may have filled it
            if cached is not None:
                return cached
            # If we have the symbol master loaded, validate the contract exists.
            if self._symbol_set is not None and tsym.upper() not in self._symbol_set:
                self._log_resolution_miss(underlying, option_type, expiry_str, int_strike, tsym)
                return ""
            self._symbol_cache[cache_key] = tsym  # remember it for next time
            return tsym

    def _log_resolution_miss(
        self, underlying: str, option_type: str, expiry_str: str, int_strike: int, tsym: str
    ) -> None:
        """
        Print a helpful breakdown when a constructed symbol is not in the master,
        so we can tell WHICH part is wrong (expiry vs strike). Debug aid only.
        """
        log.warning(f"Shoonya symbol NOT in NFO master: {tsym} "
              f"({underlying}/{option_type}/{expiry_str}/strike {int_strike}).")
        if self._symbol_set:
            prefix = f"{underlying}{expiry_str}{option_type[0]}"
            sample = sorted(s for s in self._symbol_set if s.startswith(prefix))
            if sample:
                log.info(f"  sample {prefix}* symbols: {sample[:3]} ... {sample[-3:]}; wanted {tsym}")
            else:
                log.info(f"  no {prefix}* symbols found - check the expiry ({expiry_str}).")

    # ------------------------------------------------------------------
    # Order placement
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
        """Serialize one submission through its typed fill confirmation."""

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
                    "Shoonya order was not submitted because another order's "
                    "outcome remained unresolved past the broker deadline."
                ),
            )
        try:
            if self._native_poisoned:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="SESSION_POISONED",
                    reason=(
                        "Shoonya session is indeterminate after an earlier "
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
        """
        Place ONE market order (buy or sell at the current price) on Shoonya.

        Parameters:
          symbol   : Shoonya tsym for the contract (from resolve_option_symbol).
          side     : "BUY" or "SELL".
          quantity : total units (lot_size * lots), NOT number of lots.
          product_type : "INTRADAY" (MIS, same-day) or "NORMAL" (NRML, overnight).

        Local input errors still raise before submission. Every broker-side
        outcome is returned as an ``OrderResult`` so acknowledgement, partial
        fill, rejection, and response loss cannot collapse into one boolean.
        """
        # Normalise + validate inputs before sending anything to the broker.
        side_norm = str(side).upper().strip()
        product_norm = str(product_type).upper().strip()
        quantity_i = _exact_int(quantity)
        if side_norm not in _SIDE_MAP:
            raise ValueError(f"Invalid side: {side!r}")
        if product_norm not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if not symbol:
            raise ValueError("Missing trading symbol for order.")
        if quantity_i is None or quantity_i <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")
        if not self.ensure_logged_in():
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="REJECTED",
                reason="Order was not submitted because Shoonya login failed.",
            )

        exchange = str(exchange_segment).strip() or "NFO"

        def submit_order() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("Shoonya client is not initialised.")
            return native_client.place_order(
                    buy_or_sell=_SIDE_MAP[side_norm],
                    product_type=_PRODUCT_MAP[product_norm],
                    exchange=exchange,
                    tradingsymbol=str(symbol),
                    quantity=quantity_i,
                    discloseqty=0,
                    price_type="MKT",
                    price=0,
                    trigger_price=0,
                    retention="DAY",
                    remarks=order_tag or "multistrategy_master",
                )

        try:
            resp = self._native_call("PlaceOrder", submit_order)
        except Exception as exc:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="",
                reason=f"Shoonya PlaceOrder response is indeterminate: {exc}",
            )

        # NorenApi.place_order preserves both success and error dictionaries.
        # A genuine Not_Ok reply is a known rejection; malformed/response-lost
        # evidence stays UNKNOWN and therefore cannot trigger paper fallback.
        if not self._is_order_ack(resp):
            explicit_rejection = (
                isinstance(resp, dict)
                and str(resp.get("stat", "")).strip().lower()
                in {"not_ok", "notok", "rejected"}
            )
            reason = (
                str(resp.get("emsg") or resp.get("rejreason") or resp)
                if isinstance(resp, dict)
                else f"Malformed acknowledgement: {resp!r}"
            )
            return normalize_order_result(
                order_id=self.extract_order_id(resp),
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="REJECTED" if explicit_rejection else "",
                reason=reason,
            )

        # Acceptance != fill: the broker can still reject or leave it unfilled
        # after the ack. Confirm a REAL fill before returning success.
        order_id = self.extract_order_id(resp)
        if not order_id:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="",
                reason=f"Shoonya acknowledged the order without an order id: {resp}",
            )
        return self._confirm_fill(order_id, quantity_i)

    def _order_status(self, order_id: str):
        """
        Read the latest state of one order via single_order_history.

        Returns (state, filled_qty, order_qty, reason). `state` is None when the
        status cannot be read yet (a transient error, or the order not visible
        yet), in which case the caller should simply keep polling.
        """
        def read_history() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("client not initialised")
            return native_client.single_order_history(order_id)

        try:
            rows = self._native_call(
                "SingleOrdHist",
                read_history,
                allow_after_poison=True,
            )
        except Exception as exc:
            return (None, None, None, f"single_order_history error: {exc}")
        # Shoonya shape: a LIST of history rows (newest first). Prefer a terminal
        # row (complete/rejected/cancelled); otherwise fall back to the newest.
        if not isinstance(rows, list) or not rows:
            return (None, None, None, f"unrecognised single_order_history: {rows}")
        valid_rows = [row for row in rows if isinstance(row, dict)]
        if not valid_rows:
            return (None, None, None, "malformed single_order_history row")
        chosen = None
        for row in valid_rows:
            st = str(row.get("status", "")).strip().lower()
            if st in _FILLED_ORDER_STATES or st in _FAILED_ORDER_STATES:
                chosen = row
                break
        if chosen is None:
            chosen = valid_rows[0]

        return (
            str(chosen.get("status", "")).strip().lower(),
            _exact_int(chosen.get("fillshares")),
            _exact_int(chosen.get("qty")),
            str(chosen.get("rejreason", "")).strip(),
        )

    def get_order_status(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Return a normalized order-history snapshot without throwing ambiguity."""

        requested = _exact_int(requested_quantity)
        if requested is None or requested < 0:
            requested = 0
        state, filled, broker_qty, reason = self._order_status(str(order_id))
        effective_requested = requested or broker_qty or 0
        if requested and broker_qty not in {None, requested}:
            return normalize_order_result(
                order_id=order_id,
                requested_quantity=requested,
                filled_quantity=None,
                broker_state=state or "",
                reason=(
                    f"Malformed Shoonya status: broker quantity {broker_qty} "
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

        Polling happens outside the lock; each history request re-takes it only
        for the native, deadline-bound Noren call.

        Why the loop keeps polling on PARTIAL/UNKNOWN snapshots: a market order
        is often observed mid-fill (Noren state "open" with some quantity
        already traded on `fillshares`).  That is TRANSIENT -- the next 0.5s
        poll usually shows COMPLETE -- so returning it as the final outcome
        would freeze every live strategy over a perfectly healthy order.  A
        partial/unknown snapshot becomes the real outcome only when the broker
        label is terminal (the order can never fill further, e.g. a partial
        fill followed by a cancel) or the timeout below expires.
        """
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_result = normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=0,
            broker_state="",
            reason="Shoonya acknowledgement received; fill status not read yet.",
        )
        while time.monotonic() < deadline:
            last_result = self.get_order_status(order_id, requested_quantity=want_qty)
            if last_result.status in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                return last_result
            if last_result.broker_state in TERMINAL_BROKER_STATES:
                # Terminal label with a partial/contradictory quantity snapshot:
                # no further fills are possible, so this IS the final outcome.
                return last_result
            if "error:" in last_result.reason.lower():
                # A failed status read is already indeterminate; repeating it
                # inside the same order interaction adds no evidence.
                return last_result
            time.sleep(_FILL_POLL_INTERVAL)
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=last_result.filled_quantity,
            broker_state=last_result.broker_state,
            reason=(
                f"Shoonya order {order_id} was not terminal within "
                f"{_FILL_TIMEOUT_SECONDS:.0f}s; exposure is indeterminate."
            ),
        )

    def cancel_order(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Request cancellation, then return the order's normalized state."""

        order_id_s = str(order_id).strip()
        requested = _exact_int(requested_quantity)
        if not order_id_s:
            raise ValueError("Missing order id for cancellation.")
        if requested is None or requested < 0:
            raise ValueError(f"Invalid requested_quantity: {requested_quantity!r}")
        def submit_cancel() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("client not initialised")
            return native_client.cancel_order(order_id_s)

        try:
            self._native_call(
                "CancelOrder",
                submit_cancel,
                allow_after_poison=True,
            )
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason=f"Shoonya cancellation response is indeterminate: {exc}",
            )
        return self.get_order_status(order_id_s, requested_quantity=requested)

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        """Return non-terminal order-book rows or an indeterminate result."""

        def read_order_book() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("client not initialised")
            return native_client.get_order_book()

        try:
            response = self._native_call(
                "OrderBook",
                read_order_book,
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-order query failed: {exc}"
            )
        if response is None:
            return BrokerQueryResult.indeterminate(
                "Shoonya open-order query returned no response."
            )
        if _is_no_data_envelope(response):
            return BrokerQueryResult.success(())
        if isinstance(response, dict):
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-order query returned {response!r}"
            )
        if not isinstance(response, list):
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-order query returned malformed data: {response!r}"
            )

        orders: list[OpenOrder] = []
        for row in response:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Shoonya open-order query contained a malformed row."
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
                    "Shoonya open-order query contained incomplete order data."
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
        """Return non-flat positions or an explicit indeterminate result."""

        def read_positions() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("client not initialised")
            return native_client.get_positions()

        try:
            response = self._native_call(
                "PositionBook",
                read_positions,
                allow_after_poison=True,
            )
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-position query failed: {exc}"
            )
        if _is_no_data_envelope(response):
            return BrokerQueryResult.success(())
        if response is None or isinstance(response, dict):
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-position query returned {response!r}"
            )
        if not isinstance(response, list):
            return BrokerQueryResult.indeterminate(
                f"Shoonya open-position query returned malformed data: {response!r}"
            )
        positions: list[OpenPosition] = []
        for row in response:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Shoonya open-position query contained a malformed row."
                )
            quantity = _exact_int(row.get("netqty"))
            symbol = str(row.get("tsym") or "").strip()
            if quantity is None or not symbol:
                return BrokerQueryResult.indeterminate(
                    "Shoonya open-position query contained incomplete position data."
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

    def _is_order_ack(self, resp: Any) -> bool:
        """
        True only when `resp` is a genuine order acknowledgement.

        NorenApi.place_order returns a dict with stat == "Ok" and a norenordno on
        success (and None otherwise), so we require both.
        """
        if not isinstance(resp, dict):
            return False
        if str(resp.get("stat", "")).strip().lower() != "ok":
            return False
        return bool(resp.get("norenordno"))

    def extract_order_id(self, order_response: Any) -> str:
        """
        Dig the broker's order number ("norenordno") out of Shoonya's reply.

        The reply can be a dict, a list, or nested combinations, so this function
        recurses to search at any depth and returns the first order-id-looking
        value it finds (or "" if none). Used mainly for logging.
        """
        if isinstance(order_response, OrderResult):
            return order_response.order_id
        if order_response is None:
            return ""
        if isinstance(order_response, dict):
            for key in ("norenordno", "order_id", "orderId", "id"):
                if order_response.get(key):
                    return str(order_response.get(key))
            for value in order_response.values():
                found = self.extract_order_id(value)
                if found:
                    return found
            return ""
        if isinstance(order_response, list):
            for item in order_response:
                found = self.extract_order_id(item)
                if found:
                    return found
            return ""
        if isinstance(order_response, str):
            return order_response.strip()
        return ""

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def logout(self) -> dict[str, Any]:
        """Close the Shoonya session cleanly at shutdown (safe to call if never logged in)."""
        if self.client is None:
            return {"State": "NOT_OK", "message": "Client not initialised"}

        def submit_logout() -> Any:
            native_client = self.client
            if native_client is None:
                raise RuntimeError("Client not initialised")
            return native_client.logout()

        try:
            out = self._native_call(
                "Logout",
                submit_logout,
                allow_after_poison=True,
            )
            result = out if isinstance(out, dict) else {"State": "OK", "message": str(out)}
        except Exception as exc:
            result = {"State": "NOT_OK", "message": str(exc)}
        finally:
            with self._lock:
                self.is_logged_in = False
                self.client = None
        return result


# The ONE shared instance the whole program uses. Every file that needs to place
# a real order does `from Dependencies.shoonya_execution import shoonya_execution_client`
# and reuses this object (so there's a single login + single symbol list).
shoonya_execution_client = ShoonyaExecutionClient()
