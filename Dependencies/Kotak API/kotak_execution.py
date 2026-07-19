"""
Shared Kotak Neo (v2 SDK) order-execution layer for the Multithreading master runner.

============================== Plain-English overview ==========================
This file is the ONE place that talks to the Kotak Neo broker to place real
orders. The strategies decide *what* to trade; this module knows *how* to send
that order to Kotak. When a strategy is in "paper" mode it never touches this
file at all; only "real"/live strategies do.

Glossary for newcomers (the broker world is full of jargon):
- TOTP   : the 6-digit code from an authenticator app (Google Authenticator,
           etc.) that changes every 30 seconds. Part of 2-factor login.
- 2FA    : two-factor authentication. Kotak needs your MPIN *and* a TOTP before
           it will let you trade. "edit_token"/"edit_sid" are the proof that 2FA
           finished; without them every order is rejected.
- MPIN   : the numeric PIN for your Kotak account.
- UCC    : "Unique Client Code" - your account id at the broker.
- lot    : options trade in fixed bundles ("lots"), e.g. 1 NIFTY lot = 75 units.
           quantity = lot_size * number_of_lots.
- MIS / NRML : product types. MIS = intraday (auto square-off same day),
           NRML = carry-forward/overnight.
- pTrdSymbol : Kotak's own name for a contract (e.g. "NIFTY26JUN23950CE"). It is
           DIFFERENT from DhanHQ's name for the same contract, so we must look it
           up in Kotak's "scrip master" file before ordering.
- scrip master : a big list (CSV) of every tradable contract and its details.
- thread / lock : the runner uses many parallel "threads" (one per strategy). A
           "lock" makes sure only one thread talks to Kotak at a time, because the
           Kotak session is not safe to use from two threads at once.
- lazy login : we only log in to Kotak the first time it is actually needed, not
           at import time.
===============================================================================

Design rules (mirrors `Renko Strategy/kotak_client.py`):
- This file owns ONLY order-side concerns. No strategy logic, no symbol
  resolution, no market-data logic. The master runner still resolves every
  contract from DhanHQ and passes the Dhan `trading_symbol` straight through
  as the Kotak order symbol (the pattern already used by the live Renko
  execution script).

Why a dedicated module (instead of importing the Renko v1 wrapper):
- The official SDK now used is Kotak-neo-api-v2, downloaded into the top-level
  "Kotak Neo API/" folder. We add that folder to `sys.path` here so
  `import neo_api_client` resolves without a global pip install.
- 24 strategy workers run as threads sharing ONE process. This wrapper holds a
  single authenticated session behind a lock, logs in lazily on the first real
  order, and is therefore safe to call from any worker thread.

How the master file uses this module:
1. `from Dependencies.kotak_execution import kotak_execution_client` (guarded by
   try/except ImportError so pure-paper runs work even without the SDK folder).
2. Workers call `kotak_execution_client.place_market_order(...)` on a real
   entry/exit leg. Login happens automatically on the first such call.
3. `logout()` is called during graceful shutdown.

NOTE: the v2 SDK kept the login + place_order surface backward-compatible with
v1, so the BUY/SELL->B/S, MARKET->MKT, INTRADAY->MIS maps below match the
existing proven wrapper.
"""

import io
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from getpass import getpass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Make the broker-neutral contract importable when this helper is loaded by the
# master and when the sibling diagnostic executes it as a standalone script.
#
# The REPO ROOT has to go on the path too, not just ``Dependencies``: the
# redaction helpers below are imported as ``Dependencies.secret_redaction``,
# which needs the package's PARENT directory.  The master already has it because
# it is launched from the repo root, but ``python <script>`` puts the SCRIPT's
# directory on ``sys.path[0]`` and never the working directory -- so the sibling
# diagnostic (directly or via ``algo.py diagnose``) used to die on import with
# "No module named 'Dependencies'".
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
    normalize_order_result,
)

from Dependencies.secret_redaction import redact_payload, redact_text  # noqa: E402

# --- Make the downloaded "Kotak Neo API" SDK importable (if vendored) ---------
# The SDK may live in a "Kotak Neo API" folder somewhere above this file, but the
# depth differs by repo layout (e.g. <repo>/Dependencies/ vs
# <repo>/Multithreading/Dependencies/). Rather than hardcode a parents[] index
# (which is wrong in one layout or the other), walk UP the directory tree and use
# the first "Kotak Neo API" folder found. If none exists - the SDK was pip
# installed instead - we skip this and rely on the plain `import neo_api_client`.
for _ancestor in Path(__file__).resolve().parents:
    _sdk_dir = _ancestor / "Kotak Neo API"
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

    # .env lives one level up in Dependencies/ (this file is in Dependencies/Kotak API/).
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass

from neo_api_client import NeoAPI  # noqa: E402  (resolved from "Kotak Neo API")

# Route status/errors through the logging system so they land in the master
# runner's log file/handlers (with timestamps + levels), not just stdout.
log = logging.getLogger(__name__)


# Kotak's API expects short codes, not friendly words. These three dicts
# translate the readable values our code uses (left) into the exact codes Kotak
# wants (right). Example: we say "BUY", Kotak wants "B".
_ORDER_TYPE_MAP = {"MARKET": "MKT", "LIMIT": "L", "SL": "SL", "SL-M": "SL-M"}
_PRODUCT_MAP = {"INTRADAY": "MIS", "NORMAL": "NRML"}
_SIDE_MAP = {"BUY": "B", "SELL": "S"}

# Fill confirmation: Kotak's place_order only ACKNOWLEDGES that the request was
# accepted - the broker can still reject it or leave it unfilled. After an ack we
# poll order_history until the order reaches a terminal state. Market orders
# normally fill in well under a second; we wait a few seconds then give up.
_FILL_TIMEOUT_SECONDS = 8.0
_FILL_POLL_INTERVAL = 0.5
# Kotak order-status (`ordSt`) values, lower-cased.
_FILLED_ORDER_STATES = {"complete", "traded", "executed", "filled"}
_FAILED_ORDER_STATES = {"rejected", "cancelled", "canceled", "cancel", "lapsed"}
_BROKER_DEADLINE_SECONDS = 10.0

# The scrip master is a multi-megabyte bulk CSV fetched once, not an order, so
# it gets its own (longer) budget.  Holding it to the ORDER deadline was a
# category error: ten seconds is chosen so a live order can never go stale in
# flight, and a catalogue download simply is not that kind of call -- on a
# slower link it was timing out and disabling live trading for the session.
#
# ``_BROKER_DEADLINE_SECONDS`` deliberately stays at exactly 10s: every order
# path still uses it, and a test pins that value.  The startup preload keeps
# this download off the order path in practice; the lazy fallback inside
# ``resolve_option_symbol`` runs before the order-submission gate, so a slow
# catalogue delays symbol resolution rather than an in-flight order.
_SCRIP_MASTER_TIMEOUT_SECONDS = 60.0
# Stream the CSV rather than reading it in one go, and report progress every few
# seconds.  Two fixed guesses (10s, then 20s) both expired with no way to tell
# whether the transfer was slow-but-moving or stalled outright; byte counts in
# the log answer that directly the next time it happens.
_SCRIP_MASTER_CHUNK_BYTES = 1 << 20
_SCRIP_MASTER_PROGRESS_SECONDS = 5.0


class _SdkDeadlineExceeded(TimeoutError):
    """Raised when a Kotak SDK call outlives the fixed broker deadline."""


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


def _response_rows(response: Any) -> list[Any] | None:
    """Extract the common Kotak list shapes without treating errors as empty."""

    if not isinstance(response, dict):
        return response if isinstance(response, list) else None

    def _is_error_envelope(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        state = str(
            value.get("stat")
            or value.get("State")
            or value.get("state")
            or value.get("status")
            or ""
        ).strip().lower()
        if state in {"not_ok", "notok", "rejected", "failed", "failure", "error"}:
            return True
        return bool(value.get("Error") or value.get("Error Message") or value.get("error"))

    if _is_error_envelope(response):
        return None
    data = response.get("data")
    if isinstance(data, list):
        return data
    if _is_error_envelope(data):
        return None
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    return None


class KotakExecutionClient:
    """
    One shared "phone line" to Kotak that every live strategy uses.

    Think of an instance of this class as a single logged-in Kotak session that
    the whole program shares. It is:
    - "lazy-login": it only logs in the first time someone actually needs it.
    - "thread-safe": many strategy threads can call it at once; a lock makes sure
      only one of them is talking to Kotak at any moment (the broker session
      cannot be used by two threads simultaneously). Orders are infrequent, so
      this queuing has no real cost.

    The module creates ONE instance at the bottom (`kotak_execution_client`) and
    everything imports and reuses that single object.
    """

    def __init__(self) -> None:
        # The live Kotak SDK object. Stays None until we successfully log in.
        self.client: NeoAPI | None = None
        # Simple flag so we don't re-login every call once we're authenticated.
        self.is_logged_in = False
        # The "one thread at a time" gate for login + orders + scrip lookups.
        self._lock = threading.RLock()
        # Kotak's SDK exposes no native timeout argument. Every SDK interaction
        # therefore runs through ONE single-worker executor. If a call times out,
        # its thread may still be inside the SDK, but the one-worker queue ensures
        # no later SDK call can overlap it.
        self._sdk_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="KotakBroker",
        )
        # Serialize callers before they reach the executor.  This prevents an
        # entry from sitting in the executor queue behind a call that later
        # times out: poison is recorded before the waiting entry can submit.
        self._sdk_call_lock = threading.Lock()
        self._session_poisoned = False
        self._timed_out_future: Future[Any] | None = None
        self._timed_out_futures: set[Future[Any]] = set()
        # Make the poison check and executor submission one atomic operation.
        # Without this gate, two strategy threads can both see a healthy session,
        # queue two orders, and let the second execute after the first times out.
        self._order_submission_lock = threading.Lock()
        # Cache of resolved Kotak symbols, keyed by
        # (underlying, expiry "DDMMMYYYY", option_type, int_strike). search_scrip
        # downloads the whole NFO scrip-master CSV per call, so caching per
        # contract keeps real trading cheap (one lookup per option per day).
        self._symbol_cache: dict[tuple, str] = {}
        # The full NSE F&O scrip master, downloaded once per process and reused
        # for every symbol lookup (the SDK's search_scrip re-downloads it on
        # EVERY call, which is far too slow for a live multi-strategy runner).
        self._scrip_df: pd.DataFrame | None = None

    @property
    def session_poisoned(self) -> bool:
        """Whether an SDK timeout requires explicit reconciliation/recovery."""

        with self._lock:
            return self._session_poisoned

    def _sdk_call(
        self,
        label: str,
        operation,
        *,
        block_when_poisoned: bool = False,
        deadline_seconds: float | None = None,
    ) -> Any:
        """Run one SDK operation serially under a total wall-clock deadline.

        Args:
            label: Operation name used in deadline messages.
            operation: Zero-argument callable performing the SDK request.
            block_when_poisoned: Refuse to start when the session is poisoned.
            deadline_seconds: Override the default budget.  Only the bulk
                scrip-master download uses this; every ORDER path keeps the
                fixed ten-second deadline, so a live order can never go stale
                in flight.
        """

        budget = (
            _BROKER_DEADLINE_SECONDS if deadline_seconds is None else deadline_seconds
        )
        started = time.monotonic()
        if not self._sdk_call_lock.acquire(timeout=budget):
            raise _SdkDeadlineExceeded(
                f"Kotak {label} exceeded {budget:g}s "
                "waiting for the SDK interaction gate"
            )
        try:
            with self._lock:
                if block_when_poisoned and self._session_poisoned:
                    raise RuntimeError(
                        "Kotak session is poisoned; a new order cannot be submitted "
                        "before explicit reconciliation and recovery."
                    )
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise _SdkDeadlineExceeded(
                    f"Kotak {label} exceeded {budget:g}s before SDK submission"
                )
            future = self._sdk_executor.submit(operation)
            try:
                return future.result(timeout=remaining)
            except FutureTimeoutError as exc:
                # ``concurrent.futures.TimeoutError`` aliases built-in TimeoutError.
                # If the SDK callable itself raised one, the future is already done;
                # propagate that broker error without falsely poisoning the executor.
                if future.done():
                    raise
                # A future queued behind another stuck SDK call may still be
                # cancellable. Cancel it so an entry whose caller already received
                # UNKNOWN can never execute later. Running futures cannot be killed;
                # track every one until reconciliation proves all have ended.
                cancelled_before_start = future.cancel()
                with self._lock:
                    self._session_poisoned = True
                    self._timed_out_future = future
                    if not cancelled_before_start:
                        self._timed_out_futures.add(future)
                raise _SdkDeadlineExceeded(
                    f"Kotak {label} exceeded {budget:g}s"
                ) from exc
        finally:
            self._sdk_call_lock.release()

    def recover_after_reconciliation(self) -> bool:
        """Clear timeout poison only after the abandoned SDK call has returned.

        "Poisoned" means an earlier SDK call outlived its 10-second budget and
        was abandoned -- but Python cannot kill a running thread, so that call
        may STILL be executing inside the SDK and may still place/affect an
        order.  MAT-102 can call this hook after it reconciles orders and
        positions.  The method refuses early recovery while any timed-out
        operation is still running, preserving the no-overlap guarantee.
        """

        with self._lock:
            self._timed_out_futures = {
                future for future in self._timed_out_futures if not future.done()
            }
            if self._timed_out_futures:
                return False
            self._session_poisoned = False
            self._timed_out_future = None
            return True

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mobile(raw: str) -> str:
        """
        Kotak Neo v2 requires the mobile number WITH the country code
        (e.g. "+919876543210"); a bare 10-digit number is rejected with
        "Invalid field 'MobileNumber'". Kotak accounts are India-only, so a
        plain 10-digit number is prefixed with +91. Already-prefixed numbers
        (or anything unexpected) are left untouched.
        """
        # Tidy up: drop spaces/dashes a user might have typed.
        m = str(raw).strip().replace(" ", "").replace("-", "")
        # Blank, or already in "+91..." form -> nothing to do.
        if not m or m.startswith("+"):
            return m
        # Keep only the digits so "(0)90136..." style input still works.
        digits = "".join(ch for ch in m if ch.isdigit())
        if len(digits) == 10:                      # plain local number -> add +91
            return "+91" + digits
        if len(digits) == 12 and digits.startswith("91"):  # "91" prefix, missing the +
            return "+" + digits
        return m  # anything unexpected: leave it as-is rather than guess wrong

    @staticmethod
    def _prompt_totp() -> str:
        """
        Interactively read the current 6-digit TOTP from the authenticator app.

        Returns "" if no input is available (e.g. a non-interactive / headless
        run), which the caller treats as an aborted login -> paper fallback.
        For the live runner this is called once at startup (see main()), never
        from a worker thread mid-session, so it cannot stall trading.
        """
        try:
            return getpass("Enter Kotak Neo TOTP (input hidden): ").strip()
        except (EOFError, OSError):
            return ""

    def _login_locked(self) -> bool:
        """
        Do the actual Kotak login (the multi-step 2FA handshake) and return True
        on success. "_locked" in the name means the caller must already hold
        self._lock (so two threads never try to log in at the same time).

        The flow is: read credentials from .env -> ask the user for a TOTP ->
        send mobile+UCC+TOTP -> send MPIN -> confirm 2FA actually completed.
        """
        try:
            # Credentials come from the .env file. (COSNSUMER_KEY is an old
            # misspelling kept for compatibility; CONSUMER_KEY also works.)
            consumer_key = (
                os.getenv("COSNSUMER_KEY") or os.getenv("CONSUMER_KEY") or ""
            ).strip()
            mobile = self._normalize_mobile(os.getenv("MOBILE") or "")
            mpin = (os.getenv("MPIN") or "").strip()
            ucc = (os.getenv("UCC") or "").strip()

            # Fail early (with a clear message) if any required key is blank.
            missing = []
            if not consumer_key:
                missing.append("COSNSUMER_KEY/CONSUMER_KEY")
            if not mobile:
                missing.append("MOBILE")
            if not mpin:
                missing.append("MPIN")
            if not ucc:
                missing.append("UCC")
            if missing:
                raise ValueError("Missing Kotak env keys: " + ", ".join(missing))

            # The TOTP is entered interactively (no secret stored in .env), so it
            # is read just before use to minimise the 30s expiry window.
            totp = self._prompt_totp()
            if not totp:
                self.is_logged_in = False
                self.client = None
                log.error("Kotak login aborted: no TOTP entered.")
                return False

            # Step 1: initialise API client (v2 constructor; same shape as v1).
            self.client = NeoAPI(
                environment="prod",
                access_token=None,
                neo_fin_key=None,
                consumer_key=consumer_key,
            )
            # Step 2: TOTP login handshake (generates the view token + session id).
            login_resp = self._sdk_call(
                "totp_login",
                lambda: self.client.totp_login(
                    mobile_number=mobile,
                    ucc=ucc,
                    totp=totp,
                ),
            )
            # Step 3: validate MPIN to generate the trade token (edit_token/edit_sid).
            validate_resp = self._sdk_call(
                "totp_validate",
                lambda: self.client.totp_validate(mpin=mpin),
            )

            # CRITICAL: totp_login / totp_validate RETURN error dicts instead of
            # raising, so a failed 2FA otherwise looks successful. Every order and
            # scrip-master call gates on configuration.edit_token + edit_sid
            # ("Complete the 2fa process..."), so verify those are actually set.
            cfg = getattr(self.client, "configuration", None)
            two_fa_complete = bool(
                getattr(cfg, "edit_token", None) and getattr(cfg, "edit_sid", None)
            )
            if not two_fa_complete:
                self.is_logged_in = False
                self.client = None
                secrets = (mobile, ucc, mpin, totp, consumer_key)
                log.error("Kotak login did NOT complete 2FA - orders & scrip lookups will be rejected.")
                log.error("  totp_login    -> %s", redact_payload(login_resp, secrets))
                log.error("  totp_validate -> %s", redact_payload(validate_resp, secrets))
                return False

            # Surface the ORDER-critical fields too, not just edit_token/edit_sid.
            # place_order authorizes with Auth(edit_token) + Sid(edit_sid) +
            # sId(serverId); login + scrip lookups (which authorize via
            # consumer_key) can succeed while serverId is empty, in which case
            # orders are rejected as "unauthorized".
            self.is_logged_in = True
            server_id = getattr(cfg, "serverId", None)
            log.info(
                "Kotak execution client login successful (2FA complete). "
                f"serverId={server_id!r}, "
                f"base_url={'set' if getattr(cfg, 'base_url', None) else 'MISSING'}, "
                f"data_center={getattr(cfg, 'data_center', None)!r}"
            )
            if not server_id:
                log.warning(
                    "Kotak login returned NO serverId (hsServerId). The order "
                    "endpoint needs Auth+Sid+serverId, so order placement will be rejected "
                    "as 'unauthorized' (data + scrip lookups still work via consumer_key). "
                    "This usually means the account/API key is not enabled for live order "
                    "placement - check Trade API order permission / F&O segment with Kotak."
                )
                # Show WHAT Kotak actually sent. Until now these two responses
                # were logged only when 2FA failed outright, so this exact case
                # -- 2FA fine, serverId missing -- discarded the only evidence
                # that could explain why. Everything goes through
                # ``redact_payload`` first: tokens, session ids and the supplied
                # credentials never reach the log.
                secrets = (mobile, ucc, mpin, totp, consumer_key)
                log.warning(
                    "  totp_login    -> %s", redact_payload(login_resp, secrets)
                )
                log.warning(
                    "  totp_validate -> %s", redact_payload(validate_resp, secrets)
                )
                log.warning(
                    "  configuration fields present: %s",
                    sorted(
                        name
                        for name in dir(cfg)
                        if not name.startswith("_")
                        and not callable(getattr(cfg, name, None))
                    ),
                )
                # Turn the "orders will be rejected" inference into evidence
                # WITHOUT placing an order.  ``limits`` is a read-only call that
                # authorizes with the very same ``sId`` query parameter as
                # place_order, so if an empty serverId is genuinely fatal this
                # fails the same way a real order would -- and if it succeeds,
                # the assumption was wrong and worth revisiting.
                try:
                    limits_probe = self._sdk_call(
                        "limits_sid_probe",
                        lambda: self.client.limits(
                            segment="ALL", exchange="ALL", product="ALL"
                        ),
                    )
                except Exception as exc:
                    log.warning(
                        "  sId probe (limits, read-only) raised: %s",
                        redact_text(str(exc), secrets),
                    )
                else:
                    log.warning(
                        "  sId probe (limits, read-only) -> %s",
                        redact_payload(limits_probe, secrets),
                    )
            return True
        except Exception as exc:
            # Reset state on any failure so a stale half-session never leaks.
            self.is_logged_in = False
            self.client = None
            log.error("Kotak execution client login failed: %s", redact_text(exc))
            return False

    def ensure_logged_in(self) -> bool:
        """
        Make sure we have a live Kotak session, logging in the first time only.

        Call this before any order/lookup. If we're already logged in it returns
        immediately; otherwise it performs the one-time login. Returns True when a
        usable session exists, False if login failed (caller falls back to paper).
        """
        with self._lock:  # only one thread may log in at a time
            if self.is_logged_in and self.client is not None:
                return True  # already authenticated - nothing to do
            return self._login_locked()

    # ------------------------------------------------------------------
    # Symbol resolution (Dhan contract -> Kotak pTrdSymbol)
    # ------------------------------------------------------------------
    def preload_scrip_master(self) -> bool:
        """
        Download + cache the NSE F&O scrip master NOW (e.g. at startup) so the
        first live order doesn't pay the multi-second download mid-session.
        Requires a logged-in session. Returns True if the master is loaded.
        """
        if not self.ensure_logged_in():
            return False
        with self._lock:
            return self._ensure_scrip_master_locked()

    def _ensure_scrip_master_locked(self) -> bool:
        """
        Download Kotak's full NSE F&O contract list ONCE and keep it in memory.

        "_locked" => the caller already holds self._lock. We do this once because
        the list is large; after this, looking up any contract is instant. Returns
        True if the table is ready. (Caller holds _lock.)
        """
        if self._scrip_df is not None:
            return True  # already downloaded earlier in this run
        if self.client is None:
            log.warning("Kotak scrip master unavailable: not logged in yet.")
            return False
        # Step 1: ask Kotak for the download URL of the F&O contract CSV.
        try:
            url = self._sdk_call(
                "scrip_master",
                lambda: self.client.scrip_master(exchange_segment="nse_fo"),
            )
        except Exception as exc:
            log.warning(f"Kotak scrip_master() call failed: {exc}")
            return False
        if not isinstance(url, str) or not url.lower().startswith("http"):
            # On failure the SDK returns an error dict instead of a URL string.
            log.warning(f"Kotak scrip_master() returned no CSV url: {url!r}")
            return False
        # Step 2: download the CSV and load it into a pandas table (DataFrame).
        # Log the URL so a stuck download can be reproduced with curl/a browser
        # outside this process.  It is a plain contract-catalogue path, but it
        # goes through redact_text in case Kotak ever signs it with a token.
        log.info("Kotak scrip master URL: %s", redact_text(url))
        try:
            def download_scrip_master() -> str:
                # Streamed, not response.text, so the log can show HOW FAR the
                # transfer got.  A timeout that reports "0.0 MB after 60s" is a
                # stall; one reporting steady megabytes is simply a slow link,
                # and only the second is fixed by a longer deadline.
                started_download = time.monotonic()
                received = 0
                next_report = _SCRIP_MASTER_PROGRESS_SECONDS
                parts: list[bytes] = []
                with requests.get(
                    url,
                    timeout=_SCRIP_MASTER_TIMEOUT_SECONDS,
                    stream=True,
                ) as response:  # native timeout stays the inactivity cap
                    response.raise_for_status()
                    encoding = response.encoding or "utf-8"
                    for chunk in response.iter_content(
                        chunk_size=_SCRIP_MASTER_CHUNK_BYTES
                    ):
                        parts.append(chunk)
                        received += len(chunk)
                        elapsed = time.monotonic() - started_download
                        if elapsed >= next_report:
                            log.info(
                                "Kotak scrip master: %.1f MB after %.0fs (%.2f MB/s).",
                                received / 1_000_000,
                                elapsed,
                                received / 1_000_000 / max(elapsed, 1e-9),
                            )
                            next_report = elapsed + _SCRIP_MASTER_PROGRESS_SECONDS
                log.info(
                    "Kotak scrip master downloaded %.1f MB in %.1fs.",
                    received / 1_000_000,
                    time.monotonic() - started_download,
                )
                return b"".join(parts).decode(encoding, errors="replace")

            # ``requests`` alone has no total wall-clock deadline for a body
            # that keeps dribbling bytes. Reuse the isolated serial executor so
            # the complete download is capped and poisons the session on timeout.
            csv_text = self._sdk_call(
                "scrip_master_download",
                download_scrip_master,
                deadline_seconds=_SCRIP_MASTER_TIMEOUT_SECONDS,
            )
            df = pd.read_csv(io.StringIO(csv_text))
            df = df.rename(columns=lambda c: c.strip())  # trim stray spaces in headers
            # Kotak stores the expiry as a quirky epoch number. Decode it exactly
            # like the SDK does (seconds since epoch + a ~10-year offset) and turn
            # it into a readable "DDMMMYYYY" string such as "26JUN2025".
            exp = pd.to_datetime(df["pExpiryDate"], unit="s", errors="coerce")
            exp = exp + pd.to_timedelta(315511200, unit="s")
            # Pre-compute a few helper columns so every later lookup is a fast,
            # consistent comparison (uppercased symbol/type, normalized expiry,
            # numeric strike). The leading "_" marks them as our own additions.
            df["_expiry_str"] = exp.dt.strftime("%d%b%Y").str.upper()
            df["_sym_u"] = df["pSymbolName"].astype(str).str.strip().str.upper()
            df["_opt_u"] = df["pOptionType"].astype(str).str.strip().str.upper()
            df["_strike_f"] = pd.to_numeric(df["dStrikePrice;"], errors="coerce")
            self._scrip_df = df
            log.info(f"Kotak NSE F&O scrip master loaded ({len(df)} rows).")
            return True
        except Exception as exc:
            log.warning(f"Kotak scrip master download/parse failed: {exc}")
            return False

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry,
        option_type: str,
        strike: float,
        exchange_segment: str = "nse_fo",
    ) -> str:
        """
        Resolve a Kotak Neo `pTrdSymbol` for an option contract, cached.

        Kotak's place_order expects `trading_symbol` = the `pTrdSymbol` from
        Kotak's own ScripMaster, which differs from DhanHQ's trading symbol. We
        resolve it from the locally-cached scrip master (downloaded once) rather
        than the SDK's search_scrip, which re-downloads the whole CSV per call.

        Returns the Kotak trading symbol, or "" if it could not be resolved
        (caller then falls back to paper for that leg).

        Inputs: underlying e.g. "NIFTY"; expiry as a date object; option_type
        "CE"/"PE"; strike e.g. 23950.
        """
        # Guard against obviously bad inputs before doing any work.
        if expiry is None or strike is None or float(strike) <= 0:
            log.warning(f"Kotak resolve skipped (bad inputs): expiry={expiry!r} strike={strike!r}")
            return ""
        underlying = str(underlying).strip().upper()
        option_type = str(option_type).strip().upper()
        try:
            # Convert the date into the same "DDMMMYYYY" text the scrip master uses.
            expiry_str = expiry.strftime("%d%b%Y").upper()  # e.g. "26JUN2025"
        except Exception:
            log.warning(f"Kotak resolve skipped (expiry not a date): expiry={expiry!r} type={type(expiry).__name__}")
            return ""
        int_strike = round(float(strike))
        cache_key = (underlying, expiry_str, option_type, int_strike)

        # Fast path: have we already resolved this exact contract today?
        with self._lock:
            cached = self._symbol_cache.get(cache_key)
            if cached is not None:
                return cached

        # Not cached -> we need a session (login may prompt for a TOTP).
        if not self.ensure_logged_in():
            return ""

        with self._lock:
            cached = self._symbol_cache.get(cache_key)  # re-check: another thread may have filled it
            if cached is not None:
                return cached
            if not self._ensure_scrip_master_locked():   # make sure the contract list is loaded
                return ""
            # Find the one row matching our exact contract and read its pTrdSymbol.
            resolved = self._match_in_df(self._scrip_df, underlying, option_type, expiry_str, int_strike)
            if not resolved:
                self._log_resolution_miss(underlying, option_type, expiry_str, int_strike)
            else:
                self._symbol_cache[cache_key] = resolved  # remember it for next time
            return resolved

    @staticmethod
    def _match_in_df(df, underlying: str, option_type: str, expiry_str: str, int_strike: int) -> str:
        """
        Look up one contract in the cached contract table and return its Kotak
        symbol (pTrdSymbol), or "" if there's no exact match.

        We require ALL FOUR to match: underlying, CE/PE, expiry, and strike.
        Note Kotak stores the strike multiplied by 100 (23950 -> 2395000), so we
        compare against strike * 100.
        """
        if df is None or len(df) == 0:
            return ""
        # `mask` is a True/False per row; rows where ALL conditions hold are kept.
        mask = (
            (df["_sym_u"] == underlying)
            & (df["_opt_u"] == option_type)
            & (df["_expiry_str"] == expiry_str)
            & (df["_strike_f"] == float(int_strike) * 100.0)  # Kotak strikes are x100
        )
        rows = df[mask]
        if len(rows):
            return str(rows.iloc[0]["pTrdSymbol"]).strip()  # take the first match
        return ""

    def _log_resolution_miss(self, underlying: str, option_type: str, expiry_str: str, int_strike: int) -> None:
        """
        Print a helpful breakdown when a lookup finds nothing, so we can tell
        WHICH part didn't match (wrong expiry vs wrong strike). Debug aid only.
        """
        df = self._scrip_df
        if df is None:
            log.warning("Kotak symbol NOT resolved (scrip master not loaded).")
            return
        base = df[(df["_sym_u"] == underlying) & (df["_opt_u"] == option_type)]
        exp_match = base[base["_expiry_str"] == expiry_str]
        log.warning(f"Kotak symbol NOT resolved: {underlying}/{option_type}/{expiry_str}/strike {int_strike} "
              f"-> {len(base)} {underlying} {option_type} row(s), {len(exp_match)} on that expiry.")
        if len(exp_match):
            strikes = sorted(exp_match["_strike_f"].dropna().unique().tolist())
            log.info(f"  strikes on expiry (dStrikePrice;): {strikes[:3]} ... {strikes[-3:]}; "
                  f"wanted {float(int_strike) * 100.0}")
        else:
            # Sort by real date (the "DDMMMYYYY" strings sort by day-of-month as text).
            exp = pd.to_datetime(base["_expiry_str"].dropna().unique(), format="%d%b%Y", errors="coerce")
            expiries = [d.strftime("%d%b%Y").upper() for d in sorted(exp.dropna())]
            log.info(f"  available expiries: {expiries[:12]}")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str = "nse_fo",
        product_type: str = "INTRADAY",
        *,
        order_tag: str = "",
    ) -> OrderResult:
        """
        Place ONE market order (buy or sell at the current price) on Kotak.

        Parameters:
          symbol   : Kotak's pTrdSymbol for the contract (from resolve_option_symbol).
          side     : "BUY" or "SELL".
          quantity : total units (lot_size * lots), NOT number of lots.
          product_type : "INTRADAY" (MIS, same-day) or "NORMAL" (NRML, overnight).

        Local input errors still raise before submission. Broker acknowledgements,
        fills, rejections, partials, and deadline failures return ``OrderResult``.
        Once any SDK call times out, new orders are blocked until explicit
        reconciliation calls :meth:`recover_after_reconciliation`.
        """
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
        if not self._order_submission_lock.acquire(
            timeout=_BROKER_DEADLINE_SECONDS
        ):
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="ORDER_GATE_TIMEOUT",
                reason=(
                    "Kotak order outcome is indeterminate because the submission "
                    f"gate did not become available within {_BROKER_DEADLINE_SECONDS:g}s."
                ),
            )
        try:
            if self.session_poisoned:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="SESSION_POISONED",
                    reason=(
                        "Kotak session is poisoned by an earlier timeout; reconcile "
                        "orders/positions and recover explicitly before another order."
                    ),
                )

            if not self.ensure_logged_in():
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason="Order was not submitted because Kotak login failed.",
                )

            with self._lock:
                client = self.client
            if client is None:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="REJECTED",
                    reason="Order was not submitted because Kotak client is not initialised.",
                )
            try:
                resp = self._sdk_call(
                    "place_order",
                    lambda: client.place_order(
                        exchange_segment=str(exchange_segment).strip(),
                        product=_PRODUCT_MAP[product_norm],
                        price="0",
                        order_type=_ORDER_TYPE_MAP["MARKET"],
                        quantity=str(quantity_i),
                        validity="DAY",
                        trading_symbol=str(symbol),
                        transaction_type=_SIDE_MAP[side_norm],
                        amo="NO",
                        disclosed_quantity="0",
                        market_protection="0",
                        pf="N",
                        trigger_price="0",
                        tag=order_tag or None,
                    ),
                    block_when_poisoned=True,
                )
            except Exception as exc:
                return normalize_order_result(
                    order_id="",
                    requested_quantity=quantity_i,
                    filled_quantity=0,
                    broker_state="SESSION_POISONED" if self.session_poisoned else "",
                    reason=f"Kotak PlaceOrder response is indeterminate: {exc}",
                )

            # Keep the atomic entry gate until the acknowledged order reaches a
            # typed outcome. If status polling times out, poison is visible
            # before another strategy can even consider submitting an order.
            return self._normalize_place_ack_and_confirm(resp, quantity_i)
        finally:
            self._order_submission_lock.release()

    def _normalize_place_ack_and_confirm(
        self,
        resp: Any,
        quantity_i: int,
    ) -> OrderResult:
        """Normalize one placement acknowledgement and confirm its final fill."""

        # The v2 SDK's place_order does NOT raise on failure -- it RETURNS an
        # error dict instead (e.g. {"Error": ...}, {"Error Message": "Complete
        # the 2fa ..."} when not logged in, or {"stat": "Not_Ok", ...} on a
        # broker rejection). Treat anything without a genuine order
        # acknowledgement as a failure so the caller falls back to paper rather
        # than recording an unconfirmed fill.
        if not self._is_order_ack(resp):
            explicit_rejection = (
                isinstance(resp, dict)
                and str(resp.get("stat", "")).strip().lower()
                in {"not_ok", "notok", "rejected"}
            )
            reason = (
                str(
                    resp.get("Error")
                    or resp.get("Error Message")
                    or resp.get("message")
                    or resp
                )
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

        # Acceptance != fill: Kotak can still reject the order or leave it
        # unfilled after the ack. Confirm a REAL fill before returning success,
        # so the caller never records an entry as LIVE (or flattens an exit's
        # position) on an order that did not actually execute.
        order_id = self.extract_order_id(resp)
        if not order_id:
            return normalize_order_result(
                order_id="",
                requested_quantity=quantity_i,
                filled_quantity=0,
                broker_state="",
                reason=f"Kotak acknowledged the order without an order id: {resp}",
            )
        return self._confirm_fill(order_id, quantity_i)

    def _order_status(self, order_id: str):
        """
        Read the latest state of one order via order_history.

        Returns (state, filled_qty, order_qty, reason). `state` is None when the
        status cannot be read yet (a transient error, or the order not visible
        yet), in which case the caller should simply keep polling.
        """
        with self._lock:
            client = self.client
        if client is None:
            return (None, None, None, "client not initialised")
        try:
            resp = self._sdk_call(
                "order_history",
                lambda: client.order_history(order_id=order_id),
            )
        except Exception as exc:
            return (None, None, None, f"order_history error: {exc}")
        # Kotak shape: {"data": {"stat": "Ok", "data": [ {rows, newest first} ]}}.
        rows = _response_rows(resp)
        if not isinstance(rows, list) or not rows:
            return (None, None, None, f"unrecognised order_history: {resp}")
        latest = rows[0]  # history is newest-first; row 0 is the current state
        if not isinstance(latest, dict):
            return (None, None, None, "malformed order_history row")

        return (
            str(latest.get("ordSt", "")).strip().lower(),
            _exact_int(latest.get("fldQty")),
            _exact_int(latest.get("qty")),
            str(latest.get("rejRsn", "")).strip(),
        )

    def get_order_status(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Return a normalized order-history snapshot without hiding timeouts."""

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
                    f"Malformed Kotak status: broker quantity {broker_qty} "
                    f"does not match requested quantity {requested}. {reason}"
                ),
            )
        return normalize_order_result(
            order_id=order_id,
            requested_quantity=effective_requested,
            filled_quantity=filled,
            broker_state=state or ("SESSION_POISONED" if self.session_poisoned else ""),
            reason=reason,
        )

    def _confirm_fill(self, order_id: str, want_qty: int) -> OrderResult:
        """Poll until the order reaches a truly terminal state, or time out.

        Every status read still flows through the same single-worker executor, so
        it cannot overlap the placement call that preceded it.

        Why the loop keeps polling on PARTIAL/UNKNOWN snapshots: a market order
        is often observed mid-fill (state "open" with some quantity already
        traded) or in one of Kotak's routine hand-off states ("validation
        pending", "put order req received").  Those are TRANSIENT -- the next
        0.5s poll usually shows COMPLETE -- so returning them as the final
        outcome would freeze every live strategy over a perfectly healthy
        order.  A partial/unknown snapshot becomes the real outcome only when
        the broker label is terminal (the order can never fill further, e.g. a
        partial fill followed by a cancel) or the timeout below expires.
        """
        deadline = time.monotonic() + _FILL_TIMEOUT_SECONDS
        last_result = normalize_order_result(
            order_id=order_id,
            requested_quantity=want_qty,
            filled_quantity=0,
            broker_state="",
            reason="Kotak acknowledgement received; fill status not read yet.",
        )
        while time.monotonic() < deadline:
            last_result = self.get_order_status(order_id, requested_quantity=want_qty)
            if last_result.status in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                return last_result
            if last_result.broker_state in TERMINAL_BROKER_STATES:
                # Terminal label with a partial/contradictory quantity snapshot:
                # no further fills are possible, so this IS the final outcome.
                return last_result
            if "error:" in last_result.reason.lower() or self.session_poisoned:
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
                f"Kotak order {order_id} was not terminal within "
                f"{_FILL_TIMEOUT_SECONDS:.0f}s; exposure is indeterminate."
            ),
        )

    def cancel_order(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult:
        """Request cancellation, then normalize the resulting order state."""

        order_id_s = str(order_id).strip()
        requested = _exact_int(requested_quantity)
        if not order_id_s:
            raise ValueError("Missing order id for cancellation.")
        if requested is None or requested < 0:
            raise ValueError(f"Invalid requested_quantity: {requested_quantity!r}")
        with self._lock:
            client = self.client
        if client is None:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="",
                reason="Kotak cancellation was not submitted: client not initialised.",
            )
        try:
            self._sdk_call(
                "cancel_order",
                lambda: client.cancel_order(order_id=order_id_s),
            )
        except Exception as exc:
            return normalize_order_result(
                order_id=order_id_s,
                requested_quantity=requested,
                filled_quantity=0,
                broker_state="SESSION_POISONED" if self.session_poisoned else "",
                reason=f"Kotak cancellation response is indeterminate: {exc}",
            )
        return self.get_order_status(order_id_s, requested_quantity=requested)

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]:
        """Return non-terminal Kotak orders or an indeterminate query result."""

        with self._lock:
            client = self.client
        if client is None:
            return BrokerQueryResult.indeterminate("Kotak client not initialised.")
        try:
            response = self._sdk_call("order_report", client.order_report)
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Kotak open-order query failed: {exc}",
                broker_state="SESSION_POISONED" if self.session_poisoned else "UNKNOWN",
            )
        rows = _response_rows(response)
        if rows is None:
            return BrokerQueryResult.indeterminate(
                f"Kotak open-order query returned malformed data: {response!r}"
            )
        orders: list[OpenOrder] = []
        for row in rows:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Kotak open-order query contained a malformed row."
                )
            raw_state = str(row.get("ordSt") or "").strip().upper()
            snapshot = normalize_order_result(
                order_id=row.get("nOrdNo") or row.get("orderId"),
                requested_quantity=row.get("qty"),
                filled_quantity=row.get("fldQty"),
                broker_state=row.get("ordSt"),
                reason="Open-order snapshot",
            )
            symbol = str(
                row.get("trdSym") or row.get("tradingSymbol") or row.get("pTrdSymbol") or ""
            ).strip()
            if (
                not snapshot.order_id
                or not symbol
                or "malformed" in snapshot.reason.lower()
                or snapshot.requested_quantity <= 0
            ):
                return BrokerQueryResult.indeterminate(
                    "Kotak open-order query contained incomplete order data."
                )
            if raw_state in {
                "COMPLETE", "COMPLETED", "FILLED", "TRADED", "EXECUTED",
                "REJECTED", "CANCELLED", "CANCELED", "CANCEL", "LAPSED",
            }:
                continue
            if snapshot.remaining_quantity <= 0:
                continue
            side = {"B": "BUY", "S": "SELL"}.get(
                str(row.get("trnsTp") or row.get("transactionType") or "").strip().upper(),
                str(row.get("trnsTp") or row.get("transactionType") or "").strip().upper(),
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
        """Return non-flat Kotak positions or an indeterminate query result."""

        with self._lock:
            client = self.client
        if client is None:
            return BrokerQueryResult.indeterminate("Kotak client not initialised.")
        try:
            response = self._sdk_call("positions", client.positions)
        except Exception as exc:
            return BrokerQueryResult.indeterminate(
                f"Kotak open-position query failed: {exc}",
                broker_state="SESSION_POISONED" if self.session_poisoned else "UNKNOWN",
            )
        rows = _response_rows(response)
        if rows is None:
            return BrokerQueryResult.indeterminate(
                f"Kotak open-position query returned malformed data: {response!r}"
            )
        positions: list[OpenPosition] = []
        for row in rows:
            if not isinstance(row, dict):
                return BrokerQueryResult.indeterminate(
                    "Kotak open-position query contained a malformed row."
                )
            raw_quantity = row.get("netQty") if "netQty" in row else row.get("netqty")
            quantity = _exact_int(raw_quantity)
            symbol = str(
                row.get("trdSym") or row.get("tradingSymbol") or row.get("pTrdSymbol") or ""
            ).strip()
            if quantity is None or not symbol:
                return BrokerQueryResult.indeterminate(
                    "Kotak open-position query contained incomplete position data."
                )
            if quantity == 0:
                continue
            positions.append(
                OpenPosition(
                    symbol=symbol,
                    quantity=quantity,
                    product_type=str(row.get("prod") or row.get("product") or "").strip(),
                    broker_state="OPEN",
                )
            )
        return BrokerQueryResult.success(positions)

    def _is_order_ack(self, resp: Any) -> bool:
        """
        True only when `resp` is a genuine order acknowledgement.

        The v2 place_order returns a dict either way, so we must distinguish a
        real ack from a returned error. A placed order carries a broker order
        number (nOrdNo); we also accept an explicit Ok / 200 status. Any
        Error / Error Message / Not_Ok marker is treated as failure.
        """
        if not isinstance(resp, dict):
            return False
        if "Error" in resp or "Error Message" in resp:
            return False
        stat = str(resp.get("stat", "")).strip().lower()
        if stat in ("not_ok", "notok", "rejected", "error"):
            return False
        if self.extract_order_id(resp):
            return True
        return stat == "ok" or str(resp.get("stCode", "")).strip() == "200"

    def extract_order_id(self, order_response: Any) -> str:
        """
        Dig the broker's order number ("nOrdNo") out of Kotak's reply.

        Kotak's reply can be a dict, a list, or nested combinations, so this
        function calls itself (recurses) to search at any depth and returns the
        first order-id-looking value it finds (or "" if none). Used only for
        logging, so we know which order we placed.
        """
        if isinstance(order_response, OrderResult):
            return order_response.order_id
        if order_response is None:
            return ""
        if isinstance(order_response, dict):
            for key in ("nOrdNo", "order_id", "orderId", "id"):
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
        """Close the Kotak session cleanly at shutdown (safe to call if never logged in)."""
        with self._lock:
            client = self.client
        if client is None:
            return {"State": "NOT_OK", "message": "Client not initialised"}
        try:
            out = self._sdk_call("logout", client.logout)
            with self._lock:
                self.is_logged_in = False
                self.client = None
            return out if isinstance(out, dict) else {"State": "OK", "message": str(out)}
        except Exception as exc:
            with self._lock:
                self.is_logged_in = False
                self.client = None
            return {"State": "NOT_OK", "message": str(exc)}


# The ONE shared instance the whole program uses. Every file that needs to place
# a real order does `from Dependencies.kotak_execution import kotak_execution_client`
# and reuses this object (so there's a single login + single contract list).
kotak_execution_client = KotakExecutionClient()
