"""Freeze one completed-bar context behind four read-only MCP tool payloads.

The host builds all market evidence before inference begins.  This module then
serializes that evidence once and exposes four named views: session levels,
momentum/VWAP, market structure, and position state.  Tool reads never fetch
new data or recalculate an indicator, so every Codex tool call sees the same
authoritative facts even if the live feed changes during inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from cpr_ai_schema import validate_position_state

CPRContextToolName = Literal["session_levels", "momentum_vwap", "market_structure", "position_state"]
EXPECTED_TOOL_NAMES: tuple[CPRContextToolName, ...] = (
    "session_levels",
    "momentum_vwap",
    "market_structure",
    "position_state",
)


class FrozenCPRContextRegistry:
    """Own an immutable four-part snapshot for one completed five-minute bar.

    JSON is the freezing boundary: it removes references to mutable market data
    and gives every MCP call an independent object without a recalculation.
    Returning new objects also prevents one tool consumer from changing what a
    later tool call receives.
    """

    def __init__(self, context: Mapping[str, Any]) -> None:
        """Validate the exact public surface and serialize every section once.

        A missing tool would hide evidence from Codex; an extra tool would
        expand the capability boundary.  Both therefore fail immediately.
        """

        if set(context) != set(EXPECTED_TOOL_NAMES):
            raise ValueError("CPR context must contain exactly the four approved tool payloads.")
        validated_context = dict(context)
        # Revalidate at the freeze boundary as well as during context building.
        # This protects both direct registry callers and JSON loaded by MCP.
        validated_context["position_state"] = validate_position_state(context["position_state"])
        # Store canonical JSON rather than the caller's dictionaries.  This is
        # the moment the changing market-data world becomes an immutable turn.
        self._serialized = {
            name: json.dumps(validated_context[name], allow_nan=False, sort_keys=True, separators=(",", ":"))
            for name in EXPECTED_TOOL_NAMES
        }

    @property
    def tool_names(self) -> tuple[CPRContextToolName, ...]:
        """Return the stable public tool order used by prompt and MCP layers."""

        return EXPECTED_TOOL_NAMES

    def read(self, name: CPRContextToolName) -> dict[str, Any]:
        """Deserialize and return a fresh copy of one frozen tool payload.

        JSON decoding is intentionally repeated for each read.  The tiny cost
        buys isolation: mutations by one caller cannot leak into another call.
        """

        return json.loads(self._serialized[name])

    def snapshot_payload(self) -> dict[str, dict[str, Any]]:
        """Return all four independent copies for the isolated stdio server."""

        return {name: self.read(name) for name in EXPECTED_TOOL_NAMES}

    def write_snapshot_file(self, path: str) -> None:
        """Persist the snapshot used by one local MCP subprocess turn.

        The file is a transport artifact, not a cache or trading journal.  Its
        lifetime is controlled by the isolated runner that created it.
        """

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot_payload(), handle, allow_nan=False, sort_keys=True, separators=(",", ":"))


__all__ = ["EXPECTED_TOOL_NAMES", "CPRContextToolName", "FrozenCPRContextRegistry"]
