"""Serve four frozen CPR context payloads through a local stdio MCP server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cpr_ai_schema import validate_position_state
from cpr_ai_tools import FrozenCPRContextRegistry


def load_snapshot_payload(path: str) -> dict[str, dict[str, Any]]:
    """Read a saved context and validate its exact four-tool public boundary."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("CPR MCP snapshot must be an object.")
    # Validate immediately after untrusted JSON parsing, before it is handed
    # to the registry.  The registry repeats this check at its freeze boundary.
    raw["position_state"] = validate_position_state(raw.get("position_state"))
    return FrozenCPRContextRegistry(raw).snapshot_payload()


def build_mcp_server(snapshot_path: str):
    """Build four no-argument tools that return fresh frozen payload copies."""

    from mcp.server.fastmcp import FastMCP

    payload = load_snapshot_payload(snapshot_path)
    server = FastMCP("cpr-srsi-vwap-context", log_level="ERROR")

    @server.tool(name="session_levels", description="Return frozen CPR and opening-session facts.")
    def session_levels() -> dict[str, Any]:
        """Return a deep copy of the session-level payload."""

        return json.loads(json.dumps(payload["session_levels"]))

    @server.tool(name="momentum_vwap", description="Return frozen SRSI, RSI, EMA, VWAP and candle facts.")
    def momentum_vwap() -> dict[str, Any]:
        """Return a deep copy of the momentum/VWAP payload."""

        return json.loads(json.dumps(payload["momentum_vwap"]))

    @server.tool(name="market_structure", description="Return frozen confirmed swings and R1 scale-in evidence.")
    def market_structure() -> dict[str, Any]:
        """Return a deep copy of the market-structure payload."""

        return json.loads(json.dumps(payload["market_structure"]))

    @server.tool(name="position_state", description="Return frozen host-supplied market/position facts.")
    def position_state() -> dict[str, Any]:
        """Return a deep copy of the host-supplied position payload."""

        return json.loads(json.dumps(payload["position_state"]))

    return server


def main(argv: list[str] | None = None) -> int:
    """Parse a frozen snapshot path and serve it over stdio only."""

    parser = argparse.ArgumentParser(description="Frozen read-only CPR context MCP tools.")
    parser.add_argument("snapshot_path")
    args = parser.parse_args(argv)
    build_mcp_server(args.snapshot_path).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
