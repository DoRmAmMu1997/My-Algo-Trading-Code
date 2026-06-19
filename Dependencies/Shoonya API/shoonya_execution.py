"""
Shared Shoonya (Finvasia NorenApi) order-execution layer for the Multithreading runner.

============================== Plain-English overview ==========================
This file is the ONE place that talks to the Shoonya (Finvasia) broker to place
real orders. The strategies decide *what* to trade; this module knows *how* to
send that order to Shoonya. When a strategy is in "paper" mode it never touches
this file at all; only "real"/live strategies do.

It is a drop-in replacement for the old Kotak Neo execution layer: it exposes the
SAME public methods the master runner already calls -- ensure_logged_in(),
preload_scrip_master(), resolve_option_symbol(), place_market_order(),
extract_order_id(), logout(), and the `is_logged_in` flag -- so the rest of the
runner did not have to change shape.

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

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

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
# Shoonya order-status values, lower-cased.
_FILLED_ORDER_STATES = {"complete", "completed"}
_FAILED_ORDER_STATES = {"rejected", "cancelled", "canceled"}


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
        self.client: Optional[NorenApi] = None
        # Simple flag so we don't re-login every call once we're authenticated.
        self.is_logged_in = False
        # The "one thread at a time" gate for login + orders + symbol lookups.
        self._lock = threading.Lock()
        # Remember symbols we've already resolved, so we don't rebuild/re-validate
        # for the same contract. Key = (underlying, expiry "DDMMMYY", "CE"/"PE",
        # strike-as-int) -> Shoonya tsym string.
        self._symbol_cache: Dict[tuple, str] = {}
        # The set of valid NFO trading symbols (uppercased), downloaded once and
        # reused for validation. Stays None until first loaded.
        self._symbol_set: Optional[set] = None

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
            print(f"Shoonya TOTP secret is set but pyotp is not installed: {exc}")
            return ""
        try:
            return pyotp.TOTP(secret).now()
        except Exception as exc:
            print(f"Shoonya TOTP generation failed: {exc}")
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
            return input("Enter Shoonya TOTP (6 digits from your authenticator app): ").strip()
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
                print("Shoonya login aborted: no TOTP available "
                      "(set SHOONYA_TOTP_SECRET, or enter a code when prompted).")
                return False

            # Initialise the client and perform the login handshake.
            self.client = NorenApi()
            resp = self.client.login(
                userid=userid,
                password=password,
                twoFA=two_fa,
                vendor_code=vendor_code,
                api_secret=api_secret,
                imei=imei,
            )

            # NorenApi.login returns the response dict (with susertoken) on
            # success and None on failure; treat anything without stat == "Ok" as
            # a failed login so the caller falls back to paper.
            if not isinstance(resp, dict) or str(resp.get("stat", "")).strip().lower() != "ok":
                self.is_logged_in = False
                self.client = None
                print(f"Shoonya login failed (no valid session). Raw response: {resp}")
                return False

            self.is_logged_in = True
            print(
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
            print(f"Shoonya execution client login failed: {exc}{hint}")
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
            # pandas infers compression="zip" from the .zip URL extension (the
            # same call helper_shoonya.py uses for the symbol master).
            df = pd.read_csv(_NFO_SYMBOL_MASTER_URL)
            df = df.rename(columns=lambda c: str(c).strip())  # trim stray spaces
        except Exception as exc:
            print(f"Shoonya NFO symbol master download/parse failed: {exc}")
            return False
        if "TradingSymbol" not in df.columns:
            print(f"Shoonya NFO symbol master missing 'TradingSymbol' column: {list(df.columns)}")
            return False
        try:
            self._symbol_set = set(df["TradingSymbol"].astype(str).str.strip().str.upper())
            print(f"Shoonya NSE F&O symbol master loaded ({len(df)} rows).")
            return True
        except Exception as exc:
            print(f"Shoonya NFO symbol master parse failed: {exc}")
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
            print(f"Shoonya resolve skipped (bad inputs): expiry={expiry!r} strike={strike!r}")
            return ""
        underlying = str(underlying).strip().upper()
        option_type = str(option_type).strip().upper()
        if option_type not in ("CE", "PE"):
            print(f"Shoonya resolve skipped (bad option_type): {option_type!r}")
            return ""
        try:
            # Shoonya uses a 2-digit year, zero-padded day: "26JUN25".
            expiry_str = expiry.strftime("%d%b%y").upper()
        except Exception:
            print(f"Shoonya resolve skipped (expiry not a date): expiry={expiry!r} "
                  f"type={type(expiry).__name__}")
            return ""
        int_strike = int(round(float(strike)))
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
        print(f"Shoonya symbol NOT in NFO master: {tsym} "
              f"({underlying}/{option_type}/{expiry_str}/strike {int_strike}).")
        if self._symbol_set:
            prefix = f"{underlying}{expiry_str}{option_type[0]}"
            sample = sorted(s for s in self._symbol_set if s.startswith(prefix))
            if sample:
                print(f"  sample {prefix}* symbols: {sample[:3]} ... {sample[-3:]}; wanted {tsym}")
            else:
                print(f"  no {prefix}* symbols found - check the expiry ({expiry_str}).")

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
    ) -> Dict[str, Any]:
        """
        Place ONE market order (buy or sell at the current price) on Shoonya.

        Parameters:
          symbol   : Shoonya tsym for the contract (from resolve_option_symbol).
          side     : "BUY" or "SELL".
          quantity : total units (lot_size * lots), NOT number of lots.
          product_type : "INTRADAY" (MIS, same-day) or "NORMAL" (NRML, overnight).

        This RAISES an exception on ANY problem (bad input, not logged in, or the
        broker rejecting the order). The caller catches that and falls back to
        paper, so we never silently record an order that didn't actually happen.
        """
        if not self.ensure_logged_in():
            raise RuntimeError("Shoonya login failed; cannot place real order.")

        # Normalise + validate inputs before sending anything to the broker.
        side_norm = str(side).upper().strip()
        product_norm = str(product_type).upper().strip()
        if side_norm not in _SIDE_MAP:
            raise ValueError(f"Invalid side: {side!r}")
        if product_norm not in _PRODUCT_MAP:
            raise ValueError(f"Invalid product_type: {product_type!r}")
        if not symbol:
            raise ValueError("Missing trading symbol for order.")
        if int(quantity) <= 0:
            raise ValueError(f"Invalid quantity: {quantity!r}")

        exchange = str(exchange_segment).strip() or "NFO"

        with self._lock:  # only one order goes over the shared session at a time
            if self.client is None:
                raise RuntimeError("Shoonya client is not initialised.")
            try:
                resp = self.client.place_order(
                    buy_or_sell=_SIDE_MAP[side_norm],
                    product_type=_PRODUCT_MAP[product_norm],
                    exchange=exchange,
                    tradingsymbol=str(symbol),
                    quantity=int(quantity),
                    discloseqty=0,
                    price_type="MKT",
                    price=0,
                    trigger_price=0,
                    retention="DAY",
                    remarks="multistrategy_master",
                )
            except Exception as exc:
                raise Exception(f"Order placement failed: {exc}") from exc

        # NorenApi.place_order returns the response dict (with norenordno) on
        # success and None when stat != "Ok". Treat anything without a genuine
        # order acknowledgement as a failure so the caller falls back to paper.
        if not self._is_order_ack(resp):
            raise Exception(f"Shoonya did not acknowledge the order: {resp}")

        # Acceptance != fill: the broker can still reject or leave it unfilled
        # after the ack. Confirm a REAL fill before returning success.
        order_id = self.extract_order_id(resp)
        if not order_id:
            raise Exception(f"Shoonya acknowledged the order but returned no order id: {resp}")
        self._confirm_fill(order_id, int(quantity))
        return resp

    def _order_status(self, order_id: str):
        """
        Read the latest state of one order via single_order_history.

        Returns (state, filled_qty, order_qty, reason). `state` is None when the
        status cannot be read yet (a transient error, or the order not visible
        yet), in which case the caller should simply keep polling.
        """
        with self._lock:  # single_order_history is a broker call -> needs the session
            if self.client is None:
                return (None, 0, 0, "client not initialised")
            try:
                rows = self.client.single_order_history(order_id)
            except Exception as exc:
                return (None, 0, 0, f"single_order_history error: {exc}")
        # Shoonya shape: a LIST of history rows (newest first). Prefer a terminal
        # row (complete/rejected/cancelled); otherwise fall back to the newest.
        if not isinstance(rows, list) or not rows:
            return (None, 0, 0, f"unrecognised single_order_history: {rows}")
        chosen = None
        for row in rows:
            st = str(row.get("status", "")).strip().lower()
            if st in _FILLED_ORDER_STATES or st in _FAILED_ORDER_STATES:
                chosen = row
                break
        if chosen is None:
            chosen = rows[0]

        def _as_int(value) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0

        return (
            str(chosen.get("status", "")).strip().lower(),
            _as_int(chosen.get("fillshares", 0)),
            _as_int(chosen.get("qty", 0)),
            str(chosen.get("rejreason", "")).strip(),
        )

    def _confirm_fill(self, order_id: str, want_qty: int) -> None:
        """
        Poll single_order_history until the order is fully filled, then return.

        Raises if the order is rejected/cancelled, or if it has not filled within
        `_FILL_TIMEOUT_SECONDS`. Polling happens OUTSIDE the lock (each status read
        re-takes it briefly), so a slow fill never blocks other workers' orders.
        """
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_state = "unknown"
        while time.monotonic() < deadline:
            state, filled, _order_qty, reason = self._order_status(order_id)
            if state is not None:
                last_state = state
                if state in _FAILED_ORDER_STATES:
                    raise RuntimeError(f"order {order_id} {state} ({reason or 'no reason given'})")
                if state in _FILLED_ORDER_STATES and (want_qty <= 0 or filled >= want_qty):
                    return  # fully filled - confirmed
            time.sleep(_FILL_POLL_INTERVAL)
        raise RuntimeError(
            f"order {order_id} not confirmed filled within "
            f"{_FILL_TIMEOUT_SECONDS:.0f}s (last status: {last_state})"
        )

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
    def logout(self) -> Dict[str, Any]:
        """Close the Shoonya session cleanly at shutdown (safe to call if never logged in)."""
        with self._lock:
            if self.client is None:
                return {"State": "NOT_OK", "message": "Client not initialised"}
            try:
                out = self.client.logout()
                self.is_logged_in = False
                self.client = None
                return out if isinstance(out, dict) else {"State": "OK", "message": str(out)}
            except Exception as exc:
                self.is_logged_in = False
                self.client = None
                return {"State": "NOT_OK", "message": str(exc)}


# The ONE shared instance the whole program uses. Every file that needs to place
# a real order does `from Dependencies.shoonya_execution import shoonya_execution_client`
# and reuses this object (so there's a single login + single symbol list).
shoonya_execution_client = ShoonyaExecutionClient()
