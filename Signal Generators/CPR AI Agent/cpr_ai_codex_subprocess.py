"""Execute one optional Codex SDK turn in a temporary read-only process.

The parent gives this process a frozen public snapshot, not the trading
workspace or credentials.  SDK imports remain here so an absent package fails
only the optional CPR agent, while the parent can safely retain its normal
market-data and broker lifecycle.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REQUEST_KEYS = {"snapshot_path", "model", "reasoning_effort", "prompt", "output_schema"}
_EXPECTED_TOOLS = ("session_levels", "momentum_vwap", "market_structure", "position_state")


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    """Read a public SDK item's field without depending on a private SDK model."""

    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _tool_evidence(items: Any) -> list[dict[str, str]]:
    """Extract only MCP tool name/status evidence; unknown item kinds are rejected later."""

    evidence: list[dict[str, str]] = []
    for item in items if isinstance(items, (list, tuple)) else ():
        item_type = str(_item_value(item, "type", ""))
        if "mcp" not in item_type.lower() or "tool" not in item_type.lower():
            continue
        tool_name = _item_value(item, "name", _item_value(item, "tool_name", ""))
        status = _item_value(item, "status", "failed")
        evidence.append({"tool": str(tool_name), "status": str(status)})
    return evidence


def _usage_mapping(usage: Any) -> dict[str, int]:
    """Copy numeric token totals without serializing account or request metadata."""

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, Mapping):
        return {}
    return {str(key): int(value) for key, value in usage.items() if isinstance(value, int)}


def _run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Start one fresh ephemeral thread with only four frozen MCP reads enabled."""

    from openai_codex import Codex, Sandbox  # type: ignore[import-not-found]

    snapshot_path = str(request["snapshot_path"])
    runtime_directory = str(Path(snapshot_path).parent)
    server_command = [
        sys.executable,
        str(Path(__file__).with_name("cpr_ai_mcp_server.py")),
        "--snapshot",
        snapshot_path,
    ]
    config = {
        "model_reasoning_effort": request["reasoning_effort"],
        "output_schema": request["output_schema"],
        "features": {"shell_tool": False, "unified_exec": False, "multi_agent": False, "workspace_write": False},
        "web_search": "disabled",
        "mcp_servers": {
            "cpr_ai": {
                "command": server_command[0],
                "args": server_command[1:],
                "required": True,
                "enabled_tools": list(_EXPECTED_TOOLS),
            }
        },
    }
    with Codex() as codex:
        thread = codex.thread_start(
            model=str(request["model"]),
            config=config,
            cwd=runtime_directory,
            ephemeral=True,
            sandbox=Sandbox.read_only,
        )
        result = thread.run(str(request["prompt"]), sandbox=Sandbox.read_only)
    return {
        "ok": True,
        "final_response": str(_item_value(result, "final_response", "")),
        "tool_calls": _tool_evidence(_item_value(result, "items", ())),
        "token_usage": _usage_mapping(_item_value(result, "token_usage", {})),
        # Anything besides text/reasoning/MCP should be visible to the host's
        # strict evidence gate rather than silently treated as harmless.
        "unexpected_actions": [
            str(_item_value(item, "type", "unknown"))
            for item in _item_value(result, "items", ())
            if "mcp" not in str(_item_value(item, "type", "")).lower()
            and str(_item_value(item, "type", "")).lower() not in {"message", "reasoning"}
        ],
    }


def main() -> int:
    """Read one request and emit only a structured success or failure response."""

    try:
        request = json.load(sys.stdin)
        if not isinstance(request, Mapping) or set(request) != _REQUEST_KEYS:
            raise ValueError("Invalid isolated Codex request.")
        response = _run_request(request)
    except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as error:
        # No exception text is forwarded: SDK errors can contain local paths or
        # credentials, and the parent only needs a fail-closed error category.
        response = {"ok": False, "error": type(error).__name__}
    json.dump(response, sys.stdout)
    return 0 if response["ok"] else 2


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
