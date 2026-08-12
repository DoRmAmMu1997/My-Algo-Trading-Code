"""Execute one optional Codex SDK turn inside the temporary child boundary.

The parent gives this process a frozen public snapshot, not the trading
workspace, mutable market-data store, broker session, or credentials.  The
child's only useful capability is a required MCP server with four no-argument
read tools.  It cannot execute an order; it can only return an advisory object
which the parent will independently validate.

SDK imports remain here so a missing or broken optional package disables only
the CPR AI worker.  Normal market-data, broker, and mechanical-exit lifecycles
remain in the long-running parent process.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REQUEST_KEYS = {"snapshot_path", "model", "reasoning_effort", "prompt", "output_schema"}
_EXPECTED_TOOLS = ("session_levels", "momentum_vwap", "market_structure", "position_state")
# ``build_system_prompt()`` remains the durable policy authority.  Repeating the
# four reads in the immediate turn request is intentional defense in depth: a
# production turn occasionally returned structured HOLD text without consulting
# MCP even though the developer prompt said those calls were mandatory.
_TURN_REQUEST = (
    "Before deciding, call each frozen MCP tool exactly once: session_levels, "
    "momentum_vwap, market_structure, and position_state. Wait for all four "
    "calls to complete, then evaluate the frozen CPR context and return one decision."
)
_ALLOWED_TURN_ITEM_TYPES = frozenset(
    {
        # The SDK records the prompt submitted through ``Thread.run`` as a
        # completed user-message item.  This is input bookkeeping, not a tool or
        # capability used by Codex, so it is safe to ignore during action checks.
        "userMessage",
        "agentMessage",
        "reasoning",
        "mcpToolCall",
    }
)


def build_isolated_thread_config(snapshot_path: str, python_executable: str = sys.executable) -> dict[str, Any]:
    """Build the one authoritative, read-only SDK configuration for this child.

    Keeping this pure lets tests prove the exact object passed to the SDK.  The
    child has no shell, web, multi-agent, external connector, or workspace-write
    capability; its sole required MCP server can read one frozen snapshot
    through four named tools.  These flags are defense in depth on top of the
    parent's auth-only ``CODEX_HOME`` and synthetic profile.
    """

    return {
        # Every key below is present in the pinned openai-codex 0.144.4 config
        # schema.  The auth-only CODEX_HOME is the primary boundary; these
        # supported flags are defense in depth against optional capability
        # discovery inside the isolated child.
        "features": {
            "apps": False,
            "browser_use": False,
            "browser_use_external": False,
            "collab": False,
            "collaboration_modes": False,
            "computer_use": False,
            "connectors": False,
            "enable_mcp_apps": False,
            "in_app_browser": False,
            "multi_agent": False,
            "plugin_sharing": False,
            "plugins": False,
            "remote_plugin": False,
            "shell_tool": False,
            "skill_mcp_dependency_install": False,
            "skill_search": False,
            "tool_search": False,
            "unified_exec": False,
        },
        "web_search": "disabled",
        "shell_environment_policy": {"inherit": "none"},
        "mcp_servers": {
            "cpr_ai": {
                "command": python_executable,
                "args": [str(Path(__file__).with_name("cpr_ai_mcp_server.py")), snapshot_path],
                "required": True,
                "enabled_tools": list(_EXPECTED_TOOLS),
                "default_tools_approval_mode": "approve",
            }
        },
    }


def _item_value(item: Any, name: str, default: Any = None) -> Any:
    """Read a public SDK field from either mapping or object representations.

    Staying on the documented public item surface keeps this adapter tolerant
    of harmless SDK representation differences without importing private types.
    """

    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _root_item(item: Any) -> Any:
    """Unwrap the public ``ThreadItem.root`` discriminated-union container."""

    return _item_value(item, "root", item)


def _tool_evidence(items: Any) -> list[dict[str, str]]:
    """Extract MCP tool names/statuses without trusting their returned content.

    Tool values already come from the same frozen snapshot the host owns.  What
    matters here is proving that Codex consulted every required section exactly
    once; the parent performs that exact-set check after this process exits.
    """

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
    """Copy numeric token totals without account or request metadata.

    Usage is useful for latency/cost observability, but SDK usage objects may
    grow extra fields.  An allowlist-by-shape (integer keys containing
    ``token`` plus context-window size) avoids logging identifiers by accident.
    """

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
    """Start a fresh ephemeral thread with only four frozen MCP reads enabled.

    ``read_only`` blocks workspace mutation and ``deny_all`` prevents an SDK
    approval round-trip from enabling something unexpected.  Ephemeral threads
    also prevent one bar's conversation state from influencing the next bar.
    """

    from openai_codex import ApprovalMode, Codex, Sandbox

    snapshot_path = str(request["snapshot_path"])
    runtime_directory = str(Path(snapshot_path).parent)
    config = build_isolated_thread_config(snapshot_path)
    with Codex() as codex:
        thread = codex.thread_start(
            model=str(request["model"]),
            config=config,
            cwd=runtime_directory,
            # The modular CPR prompt is policy for every turn, not merely the
            # user's one-off task.  Passing it through the SDK's dedicated
            # field makes the intended instruction hierarchy explicit.
            developer_instructions=str(request["prompt"]),
            ephemeral=True,
            sandbox=Sandbox.read_only,
            approval_mode=ApprovalMode.deny_all,
        )
        result = thread.run(
            _TURN_REQUEST,
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
            if _item_value(_root_item(item), "type", "") not in _ALLOWED_TURN_ITEM_TYPES
        ],
    }


def main() -> int:
    """Read one exact-shaped request and emit a minimal structured response.

    Failures expose only their exception category.  Exception messages can
    contain local paths, MCP diagnostics, or authentication material and are
    unnecessary for the parent's fail-closed HOLD decision.
    """

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
