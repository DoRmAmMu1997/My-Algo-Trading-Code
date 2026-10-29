"""
Rewrite one ``KEY=value`` line of a ``.env`` file in place.

Shared by the two token-setup scripts (``dhan_token_setup.py`` and
``position_broker_token_setup.py``), which each write a freshly minted access
token back into ``Dependencies/.env``.  Every other line -- comments, blank
lines, other keys, their order -- is preserved exactly, so the operator's
hand-written file survives a daily refresh.
"""

from __future__ import annotations

import re
from pathlib import Path


def replace_env_value(path: Path, key: str, value: str) -> None:
    """Replace the live ``key=`` line (or append one) and keep everything else.

    A commented-out ``# key=old`` line is left alone.  The file keeps its
    trailing-newline style.

    Raises:
        ValueError: ``value`` contains a line break.  A value is written
            verbatim, so an embedded newline would smuggle extra settings (say
            ``LIVE_TRADING_ENABLED=true``) into the file.
    """

    if "\n" in value or "\r" in value:
        raise ValueError(f"refusing to write a multi-line value for {key}")
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
    text = "\n".join(lines)
    if raw.endswith("\n") or not raw:
        text += "\n"
    path.write_text(text, encoding="utf-8")
