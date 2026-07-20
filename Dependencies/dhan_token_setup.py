"""
================================================================================
One-time DhanHQ OAuth setup script.
================================================================================

WHAT THIS DOES (read this first if you have never run it)
---------------------------------------------------------
DhanHQ's "API Key + API Secret" auth flow produces an access token through a
3-step OAuth process. Two of those three steps happen here in code; the
middle step requires you to log in via a browser (DhanHQ does this for
security so they can show you the standard login page).

This script walks you through the whole thing and writes the resulting
access token back into the `.env` file that sits next to it (i.e.
`Multithreading/Dependencies/.env`). After that, the trading scripts read
the token from that `.env` automatically.

HOW LONG THE TOKEN LASTS -- CHECK, DO NOT ASSUME
------------------------------------------------
DhanHQ decides the validity when the token is minted, and it is stamped into
the token itself. Tokens generated from the web.dhan.co screen currently
default to **24 hours** (that screen says so). Do not assume a long-lived
token: this script prints the real expiry after it finishes, so read it.

This matters for live trading. If the token dies mid-session, order placement
starts failing -- and because a rejected live EXIT leaves the position open,
you can be left holding intraday positions you cannot square off through the
API. Mint the token OUTSIDE market hours (after the 15:30 close or before the
09:15 open) so its window covers the whole session, never at midday.

WHEN TO RUN THIS SCRIPT
-----------------------
- Once after you first generate the API Key / API Secret pair on
  web.dhan.co (My Profile -> DhanHQ Trading APIs -> Generate API).
- Again whenever the access token expires -- for a 24-hour token that means
  before each trading day, outside market hours.
- Again if you regenerate the API Key / Secret for any reason.

WHAT YOU NEED FIRST (before running this)
-----------------------------------------
Open `Multithreading/Dependencies/.env` and fill in:
    DHAN_CLIENT_CODE       <- your 10-digit dhan client id
    DHAN_API_KEY           <- the API Key  (also called "app_id"  in DhanHQ docs)
    DHAN_API_SECRET        <- the API Secret (also called "app_secret")
Leave DHAN_ACCESS_TOKEN blank; this script populates it for you.

HOW TO RUN THIS SCRIPT
----------------------
From the repo root, run:
    python Multithreading/Dependencies/dhan_token_setup.py

The script prints a login URL. Open it in any browser, log in with your
Dhan credentials, and after a successful login DhanHQ will redirect you
to a URL that looks like:

    https://your-redirect/?tokenId=<long_random_string>

Copy the value after `tokenId=` (or the whole URL, the script tolerates
both) and paste it into the prompt. The script then exchanges that
short-lived `tokenId` for the `accessToken` and writes it into the
local `.env`.

THE 3 OAUTH STEPS, EXPLAINED FOR BEGINNERS
------------------------------------------
1. We tell DhanHQ "this app (identified by its API Key + Secret) wants to
   start a login session." DhanHQ replies with a temporary `consentAppId`.
2. The user opens a browser, signs in to DhanHQ, and approves the consent.
   DhanHQ redirects the browser to a URL containing a `tokenId`.
3. The user pastes that `tokenId` back into this script. We send it +
   our API Key/Secret back to DhanHQ. If everything matches up, DhanHQ
   returns the `accessToken`.

This dance is essentially the same OAuth pattern you have probably seen
when granting a third-party app access to your Google or GitHub account.
The browser step exists so you can see what is being authorised and
explicitly approve it -- credentials never live inside this script.
"""

from __future__ import annotations

# --- Standard library imports ------------------------------------------------
# `os`     - used for reading environment variables once dotenv has loaded them.
# `re`     - small regex helpers for parsing the redirect URL and rewriting .env.
# `sys`    - used to exit cleanly with an exit code on errors.
# `Path`   - cross-platform path object for locating the sibling `.env` file.
import base64
import json
import os
import re
import sys
from datetime import UTC, datetime
from getpass import getpass
from pathlib import Path

# Keep the repo root importable when this file is launched directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dependencies.secret_redaction import redact_payload, redact_text  # noqa: E402

# --- Third-party imports (lazy, with friendly errors) ------------------------
# We import dhanhq lazily so the script can give a clear error message if the
# SDK is not installed yet, instead of a raw ImportError stack trace that
# tends to scare first-time users.
try:
    from dhanhq import DhanLogin
except ImportError:
    print(
        "ERROR: The dhanhq Python SDK is not installed (or is too old).\n"
        "Run:  pip install -U dhanhq\n"
        "Then re-run this script."
    )
    sys.exit(1)

# Same friendly-error pattern for python-dotenv. dotenv reads `.env` files and
# pushes the values into `os.environ`, which is how the rest of this script
# (and the trading scripts) sees them.
try:
    from dotenv import load_dotenv
except ImportError:
    print(
        "ERROR: python-dotenv is not installed.\n"
        "Run:  pip install python-dotenv\n"
        "Then re-run this script."
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Where the .env lives.
# -----------------------------------------------------------------------------
# This script lives in `Multithreading/Dependencies/`, and the `.env` file
# sits right next to it in the SAME folder. So we just take the directory
# of this script (Path(__file__).resolve().parent) and append `.env` to it.
# We do NOT use `.parent.parent` here because we deliberately want the
# script to read/write the `.env` that is its sibling, not anything higher
# up the tree.
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"


def _load_env() -> dict:
    """
    Load `.env` into `os.environ` AND return the raw key/value pairs as a dict.

    Why we need both views:
    - `os.environ`: easy reads via `os.getenv("KEY")` for our own credentials.
    - The dict: useful when we later rewrite the file in-place to update
      DHAN_ACCESS_TOKEN without losing comments or ordering.

    If the file is missing, we exit with a friendly hint instead of a
    confusing FileNotFoundError stack trace.
    """
    if not ENV_PATH.exists():
        print(
            f"ERROR: .env not found at {ENV_PATH}\n"
            "Open that file (it is a template at the same path) and fill in "
            "DHAN_CLIENT_CODE / DHAN_API_KEY / DHAN_API_SECRET first."
        )
        sys.exit(1)

    # `override=False` means: if a value is already set in your shell, keep
    # it; the .env only fills the gap. This is the standard dotenv idiom.
    load_dotenv(dotenv_path=ENV_PATH, override=False)

    # Build a {KEY: VALUE} dict by walking the file ourselves. We skip blank
    # lines, comments, and lines that have no `=` sign.
    pairs: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def _require(env_var: str) -> str:
    """
    Return a required env var, or exit with a clear error if it is missing.

    Some users habitually wrap values in quotes (`KEY="abc"`); we strip a
    matching pair of leading/trailing quotes so both `KEY=abc` and
    `KEY="abc"` work the same way.
    """
    value = (os.getenv(env_var) or "").strip()
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        value = value[1:-1]
    if not value:
        print(
            f"ERROR: {env_var} is empty in .env\n"
            f"Open {ENV_PATH} and set {env_var} before running this script."
        )
        sys.exit(1)
    return value


def _write_access_token_to_env(new_token: str) -> None:
    """
    Update DHAN_ACCESS_TOKEN inside the `.env` file in place.

    What "in place" means here:
    - We read the file line by line.
    - Any non-comment line whose key is DHAN_ACCESS_TOKEN gets replaced.
    - Every other line (comments, blank lines, other keys) is preserved
      exactly as the user wrote it.
    - If DHAN_ACCESS_TOKEN doesn't exist at all, we append it at the end.

    This way the user's hand-written comments and the file's section
    organization survive every refresh. Important for diffability and
    for the user's own peace of mind.
    """
    raw = ENV_PATH.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Pre-compile the regex once. `^\s*DHAN_ACCESS_TOKEN\s*=.*$` matches a
    # line that begins (after optional whitespace) with the literal key
    # `DHAN_ACCESS_TOKEN`, followed by `=` and anything else.
    pattern = re.compile(r"^\s*DHAN_ACCESS_TOKEN\s*=.*$")
    replaced = False
    new_lines = []
    for line in lines:
        # We deliberately exclude commented-out lines (those that start with
        # `#`) so a comment like `# DHAN_ACCESS_TOKEN=old_token_here` is
        # preserved untouched.
        if pattern.match(line) and not line.lstrip().startswith("#"):
            new_lines.append(f"DHAN_ACCESS_TOKEN={new_token}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # Key was missing entirely -> append it at the end of the file.
        new_lines.append(f"DHAN_ACCESS_TOKEN={new_token}")

    # Preserve trailing newline behaviour for clean diffs in version control.
    text = "\n".join(new_lines)
    if raw.endswith("\n") and not text.endswith("\n"):
        text += "\n"
    ENV_PATH.write_text(text, encoding="utf-8")


def _extract_token_id(user_input: str) -> str:
    """
    Pull the bare `tokenId` value out of whatever the user pasted.

    Why this is forgiving:
    - First-time users often paste the full redirect URL such as
      `https://localhost/?tokenId=abc123&foo=bar` rather than picking out
      the `abc123` substring by hand.
    - Some users just paste the bare `abc123` string.

    We accept both. If the input contains `tokenId=...`, we extract just the
    `...` part. Otherwise we treat the whole input as the tokenId. Empty
    input returns an empty string -- the caller decides whether to abort.
    """
    cleaned = user_input.strip()
    if not cleaned:
        return ""
    # `[^&\s]+` greedily matches anything that is not an ampersand or
    # whitespace, so the match stops at the next `&` (next query param) or
    # at the end of the URL.
    match = re.search(r"tokenId=([^&\s]+)", cleaned)
    if match:
        return match.group(1)
    return cleaned


def _describe_token_expiry(access_token: str) -> str:
    """Report when this token actually dies, read from the token itself.

    DhanHQ access tokens are JWTs whose middle segment carries an ``exp``
    claim.  Decoding it (no signature check -- we are only reading a
    timestamp we were just handed) beats assuming a validity, because the
    web console currently mints 24-hour tokens while this flow may issue
    something longer.

    The warning matters for live trading: if the token expires during market
    hours, order placement starts failing, and because a rejected live EXIT
    leaves a position open, intraday positions can be left un-squareable
    through the API.

    A weekday expiry lands in one of three situations, each with its own
    message (a weekend expiry needs none -- no session is at risk that day):

    - BETWEEN 09:15 and 15:30 -> the real danger: the token dies mid-session,
      so the loud DURING-market-hours warning fires.
    - BEFORE 09:15            -> nothing can fail mid-session, but that day's
      session is not covered at all; a calmer note says to re-mint before
      trading that day.
    - AFTER 15:30             -> the whole session of the expiry day is
      covered; say so, so the operator knows this is the GOOD outcome.

    Returns:
        A human-readable line for the operator; never raises, because a
        cosmetic decode problem must not fail an otherwise good setup run.
    """
    try:
        segments = access_token.split(".")
        if len(segments) != 3:
            return "Token validity: unknown (token is not a JWT); check web.dhan.co."
        padded = segments[1] + "=" * (-len(segments[1]) % 4)
        expires_at = datetime.fromtimestamp(
            json.loads(base64.urlsafe_b64decode(padded))["exp"], UTC
        ).astimezone()
    except Exception:
        return "Token validity: could not be read from the token; check web.dhan.co."

    hours_left = (expires_at - datetime.now().astimezone()).total_seconds() / 3600
    line = f"Token expires {expires_at:%Y-%m-%d %H:%M:%S} (about {hours_left:.0f}h from now)."
    if expires_at.weekday() >= 5:
        # Weekend expiry: no session is at risk that day, and main() already
        # prints the generic "re-run whenever it expires" advice.
        return line
    # 09:15-15:30 IST is the NSE equity-derivatives session. See the
    # docstring for the three weekday situations distinguished below.
    session_open = expires_at.replace(hour=9, minute=15, second=0, microsecond=0)
    session_close = expires_at.replace(hour=15, minute=30, second=0, microsecond=0)
    if session_open <= expires_at < session_close:
        line += (
            "\n  WARNING: that is DURING market hours. If it expires while positions"
            "\n  are open you may be unable to square off through the API. Re-run this"
            "\n  script outside market hours so the window covers a whole session."
        )
    elif expires_at < session_open:
        line += (
            "\n  NOTE: that is BEFORE that day's 09:15 open, so the token does NOT"
            "\n  cover that day's trading session at all. Nothing can fail"
            "\n  mid-session, but re-run this script before trading that day"
            "\n  (outside market hours)."
        )
    else:
        line += (
            "\n  Good: that is after the 15:30 close, so the token covers that"
            "\n  day's whole trading session."
        )
    return line


def main() -> None:
    """
    Driver: walks the user through Steps 1, 2, 3 of the OAuth flow.

    The function is intentionally linear and well-printed because it is
    operated by a human at the keyboard, not another script. Every step
    prints what it is about to do BEFORE it does it, so a failed step is
    easy to associate with the network call it came from.
    """
    print("=" * 72)
    print(" DhanHQ Access Token Setup (API Key / API Secret OAuth flow)")
    print("=" * 72)

    # Load .env and grab the three required pieces of identity. If any of
    # them is missing we abort here -- the rest of the flow can't proceed.
    _load_env()
    client_code = _require("DHAN_CLIENT_CODE")
    api_key = _require("DHAN_API_KEY")
    api_secret = _require("DHAN_API_SECRET")

    # `DhanLogin(client_code)` is a thin SDK helper that knows how to talk
    # to the auth.dhan.co endpoints. It does not call anything yet; the
    # actual network calls happen on its method invocations below.
    login = DhanLogin(client_code)

    # --- Step 1 of 3: ask DhanHQ for a consent ID -----------------------------
    # This call sends our API Key + Secret to /app/generate-consent and
    # gets back a one-shot `consentAppId` that ties the upcoming browser
    # login to *this* particular app.
    print("\nStep 1/3: requesting consent ID from DhanHQ...")
    try:
        consent = login.generate_login_session(api_key, api_secret)
    except Exception as exc:
        print(f"ERROR: generate_login_session failed: {redact_text(exc, (api_key, api_secret))}")
        sys.exit(2)

    # The SDK returns either a raw consent id string or a dict-like response
    # depending on version. We unwrap whichever shape we got, looking for the
    # `consentAppId` key in any nesting depth that DhanHQ might use.
    consent_id = ""
    if isinstance(consent, str):
        consent_id = consent.strip()
    elif isinstance(consent, dict):
        for key in ("consentAppId", "consent_app_id", "consentId", "data"):
            value = consent.get(key)
            if isinstance(value, str) and value.strip():
                consent_id = value.strip()
                break
            if isinstance(value, dict):
                inner = value.get("consentAppId") or value.get("consent_app_id")
                if isinstance(inner, str) and inner.strip():
                    consent_id = inner.strip()
                    break

    if not consent_id:
        # Hit this if DhanHQ changes their wire format. The raw response is
        # printed so the user (or a future version of this script) can
        # adapt the parsing.
        print(
            "ERROR: could not parse consentAppId from response shape: "
            f"{type(consent).__name__} {redact_payload(consent, (api_key, api_secret))!r}"
        )
        sys.exit(2)

    # The login URL is constructed by appending the consent ID. DhanHQ does
    # NOT generate this URL for us -- it is a documented format we build
    # ourselves on the client side.
    login_url = f"https://auth.dhan.co/login/consentApp-login?consentAppId={consent_id}"

    # --- Step 2 of 3: user logs in via browser --------------------------------
    # We can't automate this part: DhanHQ requires a real human at a
    # browser to log in and approve the consent. The script just prints
    # the URL and waits for the redirect-URL/tokenId to come back.
    print("\nStep 2/3: open the URL below in any browser, log in to your")
    print("Dhan account, and grant access. After a successful login the")
    print("page redirects to a URL that contains `tokenId=<long_string>`.")
    print()
    print(f"   {login_url}")
    print()
    print("Paste either the full redirect URL or just the tokenId value below.")

    # `input()` blocks the script until the user hits Enter. Paste-friendly
    # extraction happens in `_extract_token_id`.
    user_input = getpass("\ntokenId (or full redirect URL; input hidden): ").strip()
    token_id = _extract_token_id(user_input)
    if not token_id:
        print("ERROR: empty tokenId. Aborting.")
        sys.exit(2)

    # --- Step 3 of 3: exchange tokenId for the access token -------------------
    # This call sends the short-lived tokenId + our API Key/Secret to
    # /app/consumeApp-consent. DhanHQ verifies the consent ID matches the
    # one we created in Step 1 and, if all is well, returns the access token.
    # DhanHQ chooses the validity; `_describe_token_expiry` reads it back.
    print("\nStep 3/3: exchanging tokenId for the access token...")
    try:
        access_response = login.consume_token_id(token_id, api_key, api_secret)
    except Exception as exc:
        print(
            "ERROR: consume_token_id failed: "
            f"{redact_text(exc, (token_id, api_key, api_secret))}"
        )
        sys.exit(3)

    # Same defensive parsing as in Step 1 -- handle multiple possible
    # response shapes (string, flat dict, nested dict).
    access_token = ""
    if isinstance(access_response, str):
        access_token = access_response.strip()
    elif isinstance(access_response, dict):
        for key in ("accessToken", "access_token", "data"):
            value = access_response.get(key)
            if isinstance(value, str) and value.strip():
                access_token = value.strip()
                break
            if isinstance(value, dict):
                inner = value.get("accessToken") or value.get("access_token")
                if isinstance(inner, str) and inner.strip():
                    access_token = inner.strip()
                    break

    if not access_token:
        print(
            "ERROR: could not parse accessToken from response shape: "
            f"{type(access_response).__name__} "
            f"{redact_payload(access_response, (token_id, api_key, api_secret))!r}"
        )
        sys.exit(3)

    # --- Validation: hit /v2/profile to confirm the token actually works ------
    # We do this BEFORE writing to .env so a token that is somehow broken
    # gets caught immediately. If validation fails we still write the token
    # (so the user has something to debug from) but we print a warning.
    print("Validating token with /v2/profile...")
    try:
        profile = login.user_profile(access_token)
    except Exception as exc:
        print(
            "WARNING: token was generated but validation failed: "
            f"{redact_text(exc, (access_token, api_key, api_secret))}"
        )
        print("Writing it to .env anyway so you can debug from there.")
    else:
        # Surface a friendly identifier from the profile response so the
        # user knows their request landed on the right account.
        if isinstance(profile, dict):
            maybe_data = profile.get("data")
            data = maybe_data if isinstance(maybe_data, dict) else profile
            client_id_back = data.get("dhanClientId") or data.get("clientId") or "?"
            print(f"OK: token validated. Client ID returned by DhanHQ = {client_id_back}")

    # All good -> persist the token into .env so the trading scripts pick
    # it up on their next start.
    _write_access_token_to_env(access_token)
    print(f"\nDone. DHAN_ACCESS_TOKEN written to {ENV_PATH}")
    print(_describe_token_expiry(access_token))
    print("Re-run this script to refresh the token whenever it expires.")


# Standard "only run main() when executed directly" guard. Lets other code
# `import` this file (e.g. for testing) without triggering the OAuth flow.
if __name__ == "__main__":
    main()
