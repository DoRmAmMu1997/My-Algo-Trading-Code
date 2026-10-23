"""
Daily login for the position-group monitor's brokers (Zerodha, mStock).

Run it once each trading morning, before ``python algo.py run``:

    python algo.py setup-broker-token --broker zerodha

Why it is a separate, interactive step
--------------------------------------
Both brokers issue an access token that dies overnight, and both require a
human step to mint a new one (Zerodha: a browser login; mStock: an OTP sent to
your phone).  Automating that human step is exactly what the brokers forbid, so
this script only does the parts around it: it prints the login URL (or asks for
the OTP), exchanges the result for an access token, and writes that token into
``Dependencies/.env`` under ``<BROKER>_ACCESS_TOKEN``.  The runner reads it
from there at startup.  No token, secret or password is ever printed.

``.env`` keys used:

* ``ZERODHA_API_KEY`` / ``ZERODHA_API_SECRET`` (from developers.kite.trade)
  -> writes ``ZERODHA_ACCESS_TOKEN``
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dependencies.secret_redaction import environment_secrets, redact_text  # noqa: E402

ENV_PATH = Path(__file__).resolve().parent / ".env"
_HTTP_TIMEOUT_SECONDS = 10.0

KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
KITE_SESSION_URL = "https://api.kite.trade/session/token"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Return the ``KEY=value`` pairs of ``.env`` (comments and blanks skipped).

    ``path`` defaults to ``ENV_PATH`` looked up at CALL time, never bound at
    import, so a test (or caller) that repoints ``ENV_PATH`` cannot end up
    reading or writing the operator's real ``.env``.
    """

    path = ENV_PATH if path is None else path
    pairs: dict[str, str] = {}
    if not path.exists():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        pairs[key.strip()] = value.strip()
    return pairs


def write_env_value(key: str, value: str, path: Path | None = None) -> None:
    """Replace (or append) one ``KEY=value`` line, keeping every other line as-is.

    ``path`` defaults to ``ENV_PATH`` at call time (see ``read_env_file``).
    """

    path = ENV_PATH if path is None else path
    raw = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$")
    replaced = False
    lines: list[str] = []
    for line in raw.splitlines():
        if pattern.match(line) and not line.lstrip().startswith("#"):
            lines.append(f"{key}={value}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_request_token(pasted: str) -> str:
    """Accept either the full Kite redirect URL or the bare ``request_token``."""

    cleaned = pasted.strip()
    if "request_token=" in cleaned:
        values = parse_qs(urlparse(cleaned).query).get("request_token") or []
        if not values:
            values = parse_qs(cleaned.split("?", 1)[-1]).get("request_token") or []
        return values[0].strip() if values else ""
    return cleaned


def kite_checksum(api_key: str, request_token: str, api_secret: str) -> str:
    """Kite's session checksum: SHA-256 of api_key + request_token + api_secret."""

    return hashlib.sha256(f"{api_key}{request_token}{api_secret}".encode()).hexdigest()


def _access_token_from(response: Any, secrets: tuple[str, ...]) -> str:
    """Pull ``data.access_token`` out of a session response, or raise clearly."""

    try:
        body = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP {response.status_code} with a non-JSON body") from exc
    data = body.get("data") if isinstance(body, dict) else None
    token = str(data.get("access_token") or "").strip() if isinstance(data, dict) else ""
    if not token:
        message = body.get("message") if isinstance(body, dict) else body
        raise RuntimeError(redact_text(f"login refused: {message}", secrets))
    return token


def zerodha_login(
    env: dict[str, str],
    *,
    ask: Callable[[str], str] = input,
    post: Callable[..., Any] = requests.post,
) -> str:
    """Run the Kite browser-login exchange and return the new access token."""

    api_key = env.get("ZERODHA_API_KEY", "")
    api_secret = env.get("ZERODHA_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Set ZERODHA_API_KEY and ZERODHA_API_SECRET in Dependencies/.env first.")
    print("1. Open this URL in your browser and log in to Kite:")
    print("   " + KITE_LOGIN_URL.format(api_key=api_key))
    print("2. Paste the URL you were redirected to (or just its request_token).")
    request_token = extract_request_token(ask("Redirect URL: "))
    if not request_token:
        raise RuntimeError("No request_token found in what was pasted.")
    response = post(
        KITE_SESSION_URL,
        data={
            "api_key": api_key,
            "request_token": request_token,
            "checksum": kite_checksum(api_key, request_token, api_secret),
        },
        headers={"X-Kite-Version": "3"},
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    return _access_token_from(response, (api_secret, request_token))


BROKER_LOGINS: dict[str, tuple[str, Callable[..., str]]] = {
    "zerodha": ("ZERODHA_ACCESS_TOKEN", zerodha_login),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--broker", required=True, choices=sorted(BROKER_LOGINS))
    args = parser.parse_args(argv)
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found; copy Dependencies/env.example there first.")
        return 1
    env_key, login = BROKER_LOGINS[args.broker]
    env = read_env_file()
    try:
        token = login(env)
    except Exception as exc:
        print(f"ERROR: {redact_text(exc, environment_secrets(env))}")
        return 1
    write_env_value(env_key, token)
    print(f"OK: {env_key} updated in {ENV_PATH.name}. It expires overnight; rerun tomorrow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
