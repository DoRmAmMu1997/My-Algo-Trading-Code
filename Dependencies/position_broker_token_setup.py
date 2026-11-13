"""
Daily login for Zerodha or mStock, saved to ``.env`` for the rest of the day.

Run it once each trading morning, before ``python algo.py run``:

    python algo.py setup-broker-token --broker zerodha
    python algo.py setup-broker-token --broker mstock

Why this exists when the adapters can log in by themselves
----------------------------------------------------------
Both brokers issue an access token that dies overnight, and both need a human
step to mint a new one (Zerodha: a browser login; mStock: a password and an OTP
or authenticator code).  The login itself lives in each broker's adapter
(``Dependencies/Zerodha API/zerodha_execution.py`` and
``Dependencies/Mstock API/mstock_execution.py``), exactly as it does for the
other brokers, and the runner can run it at startup.

What the runner will NOT do is write the token down: a token minted at startup
lives in memory only.  This script runs that same adapter login ahead of time
and stores the result under ``<BROKER>_ACCESS_TOKEN`` in ``Dependencies/.env``,
so:

* the runner starts without stopping at a prompt,
* a restart later in the day does not ask for another OTP, and
* the position-group monitor (``docs/adr/0022``), whose threads are never
  allowed to prompt, always finds a token it can validate.

Nothing here knows how to log in.  It asks the adapter, checks the token is the
plain text it should be, and rewrites one line of ``.env``.  No token, secret
or password is ever printed.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Any

_DEPENDENCIES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEPENDENCIES_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Dependencies.env_file import replace_env_value  # noqa: E402

ENV_PATH = _DEPENDENCIES_DIR / ".env"

# Every access token either broker issues is plain ASCII: Kite's is
# alphanumeric, mStock's is a JWT (base64url segments joined by dots).  Anything
# else -- above all a line break or '=' -- is refused before it reaches .env.
_TOKEN_SHAPE = re.compile(r"[A-Za-z0-9._-]+")

# broker -> (adapter folder, module name, singleton name, .env key to write).
# The folders have spaces in their names, so the modules are found by putting
# the folder on ``sys.path`` -- the same way each adapter's diagnostic does.
BROKER_ADAPTERS: dict[str, tuple[str, str, str, str]] = {
    "zerodha": ("Zerodha API", "zerodha_execution", "zerodha_execution_client", "ZERODHA_ACCESS_TOKEN"),
    "mstock": ("Mstock API", "mstock_execution", "mstock_execution_client", "MSTOCK_ACCESS_TOKEN"),
}


def load_adapter_client(broker: str) -> Any:
    """Import one broker's adapter and return its shared client object.

    Also used by ``position_groups_cli.py``.  Raises ``KeyError`` for a broker
    this script does not know.
    """

    folder, module_name, attribute, _env_key = BROKER_ADAPTERS[broker.strip().lower()]
    adapter_dir = str(_DEPENDENCIES_DIR / folder)
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
    return getattr(importlib.import_module(module_name), attribute)


def main(argv: list[str] | None = None, *, client: Any = None) -> int:
    """Log in through the adapter and store the token.  Returns a process exit code.

    ``client`` is injectable so the tests can drive this without a broker.
    """

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--broker", required=True, choices=sorted(BROKER_ADAPTERS))
    args = parser.parse_args(argv)
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found; copy Dependencies/env.example there first.")
        return 1

    # The adapter explains every step through the log (which page it opened,
    # what to paste), so the log has to reach the console.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    env_key = BROKER_ADAPTERS[args.broker][3]
    if client is None:
        client = load_adapter_client(args.broker)
    try:
        logged_in = client.ensure_logged_in() is True
        token = client.current_access_token() if logged_in else ""
    except Exception as exc:  # noqa: BLE001 - never echo an SDK/transport error verbatim
        print(f"ERROR: {args.broker} login raised {type(exc).__name__}; nothing was written.")
        return 1
    if not logged_in:
        print(f"ERROR: {args.broker} login failed (see the messages above); nothing was written.")
        return 1
    if not _TOKEN_SHAPE.fullmatch(str(token)):
        print(
            f"ERROR: the broker returned a malformed access token; {ENV_PATH.name} was not changed."
        )
        return 1
    replace_env_value(ENV_PATH, env_key, token)
    print(f"OK: {env_key} is valid and saved in {ENV_PATH.name}. It expires overnight; rerun tomorrow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
