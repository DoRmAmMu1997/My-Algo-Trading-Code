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


def build_isolated_thread_config(snapshot_path: str, python_executable: str = sys.executable) -> dict[str, Any]:
    """Build the one authoritative, read-only SDK configuration for this child.

    Keeping this pure lets tests prove the exact object passed to the SDK.  The
    child has no shell, web, multi-agent, or workspace-write capability; its
    sole MCP server can read one frozen snapshot through four named tools.
    """

    return {
        "features": {"shell_tool": False, "unified_exec": False, "collab": False, "multi_agent": False},
        "web_search": "disabled",
        "shell_environment_policy": {"inherit": "none"},
        "mcp_servers": {
            "cpr_ai": {
                "command": python_executable,
                "args": [str(Path(__file__).with_name("cpr_ai_mcp_server.py")), "--snapshot", snapshot_path],
                "required": True,
                "enabled_tools": list(_EXPECTED_TOOLS),
                "default_tools_approval_mode": "approve",
            }
        },
    }


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    """Read a public SDK item's field without depending on a private SDK model."""

    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _root_item(item: Any) -> Any:
    """Unwrap the public ``ThreadItem.root`` discriminated-union container."""

    return _item_value(item, "root", item)


def _tool_evidence(items: Any) -> list[dict[str, str]]:
    """Extract only MCP tool name/status evidence; unknown item kinds are rejected later."""

    evidence: list[dict[str, str]] = []
    for wrapped in items if isinstance(items, (list, tuple)) else ():
        item = _root_item(wrapped)
        if _item_value(item, "type", "") != "mcpToolCall":
            continue
        tool_name = _item_value(item, "tool", "")
        status = _item_value(item, "status", "failed")
        status = getattr(status, "value", status)
        evidence.append({"tool": str(tool_name), "status": str(status)})
    return evidence


def _usage_mapping(usage: Any) -> dict[str, int]:
    """Copy numeric token totals without serializing account or request metadata."""

    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "__dict__"):
        usage = vars(usage)
    if not isinstance(usage, Mapping):
        return {}
    breakdown = usage.get("last") or usage.get("total") or {}
    if hasattr(breakdown, "model_dump"):
        breakdown = breakdown.model_dump()
    if not isinstance(breakdown, Mapping):
        breakdown = {}
    flattened = {
        str(key): int(value) for key, value in breakdown.items() if isinstance(value, int) and "token" in str(key)
    }
    context_window = usage.get("model_context_window")
    if isinstance(context_window, int):
        flattened["model_context_window"] = context_window
    return flattened


def _run_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Start one fresh ephemeral thread with only four frozen MCP reads enabled."""

    from openai_codex import ApprovalMode, Codex, Sandbox  # type: ignore[import-not-found]

    snapshot_path = str(request["snapshot_path"])
    runtime_directory = str(Path(snapshot_path).parent)
    config = build_isolated_thread_config(snapshot_path)
    with Codex() as codex:
        thread = codex.thread_start(
            model=str(request["model"]),
            config=config,
            cwd=runtime_directory,
            ephemeral=True,
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.deny_all,
        )
        result = thread.run(
            str(request["prompt"]),
            approval_mode=ApprovalMode.deny_all,
            output_schema=request["output_schema"],
            effort=request["reasoning_effort"],
        )
    return {
        "ok": True,
        "final_response": str(_item_value(result, "final_response", "")),
        "tool_calls": _tool_evidence(_item_value(result, "items", ())),
        "token_usage": _usage_mapping(_item_value(result, "usage", {})),
        # Anything besides text/reasoning/MCP should be visible to the host's
        # strict evidence gate rather than silently treated as harmless.
        "unexpected_actions": [
            str(_item_value(_root_item(item), "type", "unknown"))
            for item in _item_value(result, "items", ())
            if _item_value(_root_item(item), "type", "") not in {"agentMessage", "reasoning", "mcpToolCall"}
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
