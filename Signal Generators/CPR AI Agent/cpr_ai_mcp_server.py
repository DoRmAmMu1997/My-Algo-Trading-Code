"""Serve one frozen CPR snapshot through four local, read-only MCP tools.

Model Context Protocol (MCP) is used only as a structured transport between
the isolated Codex process and facts already calculated by the trading host.
The server has no market-data, filesystem-write, broker, or order capability.
Each no-argument tool returns a deep copy of one section from the same snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cpr_ai_schema import validate_position_state
from cpr_ai_tools import FrozenCPRContextRegistry


def load_snapshot_payload(path: str) -> dict[str, dict[str, Any]]:
    """Load untrusted JSON and return an exact, revalidated four-tool payload.

    Although the host created the file, this process treats every process
    boundary as untrusted.  Shape and position-state validation therefore run
    again before any MCP tool is registered.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("CPR MCP snapshot must be an object.")
    # Validate immediately after untrusted JSON parsing, before it is handed
    # to the registry.  The registry repeats this check at its freeze boundary.
    raw["position_state"] = validate_position_state(raw.get("position_state"))
    return FrozenCPRContextRegistry(raw).snapshot_payload()


def build_mcp_server(snapshot_path: str):
    """Create the local stdio server around one immutable snapshot.

    Importing MCP lazily keeps this agent optional: repositories that never
    enable CPR AI can still import and run the rest of the trading system
    without installing the Codex/MCP dependency set.
    """

    # This optional import belongs inside the factory so a missing MCP package
    # disables only this opt-in agent, not the master runner.
    from mcp.server.fastmcp import FastMCP

    # Load once before registering tools.  Tool calls below can only copy this
    # object; they cannot observe live feed updates during the Codex turn.
    payload = load_snapshot_payload(snapshot_path)
    server = FastMCP("cpr-srsi-vwap-context", log_level="ERROR")

    @server.tool(name="session_levels", description="Return frozen CPR and opening-session facts.")
    def session_levels() -> dict[str, Any]:
        """Return CPR, pivot, opening-range, distance, and prior-regime facts."""

        return json.loads(json.dumps(payload["session_levels"]))

    @server.tool(name="momentum_vwap", description="Return frozen SRSI, RSI, EMA, VWAP and candle facts.")
    def momentum_vwap() -> dict[str, Any]:
        """Return indicator and candle evidence calculated by the Python host."""

        return json.loads(json.dumps(payload["momentum_vwap"]))

    @server.tool(name="market_structure", description="Return frozen confirmed swings and R1 scale-in evidence.")
    def market_structure() -> dict[str, Any]:
        """Return confirmed swing comparisons and the long-only R1 candidate."""

        return json.loads(json.dumps(payload["market_structure"]))

    @server.tool(name="position_state", description="Return frozen host-supplied market/position facts.")
    def position_state() -> dict[str, Any]:
        """Return allowlisted premise and protection facts, never order details."""

        return json.loads(json.dumps(payload["position_state"]))

    return server


def main(argv: list[str] | None = None) -> int:
    """Run the snapshot server on standard input/output for one local client.

    The positional snapshot argument is deliberately the server's only input;
    there is no network listener, credential option, or execution subcommand.
    """

    parser = argparse.ArgumentParser(description="Frozen read-only CPR context MCP tools.")
    parser.add_argument("snapshot_path")
    args = parser.parse_args(argv)
    build_mcp_server(args.snapshot_path).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
