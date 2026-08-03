"""Freeze the independent CPR context behind four read-only MCP tool payloads."""

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
    """Serialize exactly one complete context and return fresh copies on demand.

    JSON is the freezing boundary: it removes references to mutable market data
    and gives every MCP call an independent object without a recalculation.
    """

    def __init__(self, context: Mapping[str, Any]) -> None:
        """Validate the four public sections and serialize them once."""

        if set(context) != set(EXPECTED_TOOL_NAMES):
            raise ValueError("CPR context must contain exactly the four approved tool payloads.")
        validated_context = dict(context)
        # Revalidate at the freeze boundary as well as during context building.
        # This protects both direct registry callers and JSON loaded by MCP.
        validated_context["position_state"] = validate_position_state(context["position_state"])
        self._serialized = {
            name: json.dumps(validated_context[name], allow_nan=False, sort_keys=True, separators=(",", ":"))
            for name in EXPECTED_TOOL_NAMES
        }

    @property
    def tool_names(self) -> tuple[CPRContextToolName, ...]:
        """Return the stable public tool order used by prompt and MCP layers."""

        return EXPECTED_TOOL_NAMES

    def read(self, name: CPRContextToolName) -> dict[str, Any]:
        """Return a fresh deep copy of the original frozen tool payload."""

        return json.loads(self._serialized[name])

    def snapshot_payload(self) -> dict[str, dict[str, Any]]:
        """Return all four fresh payloads for an isolated stdio MCP process."""

        return {name: self.read(name) for name in EXPECTED_TOOL_NAMES}

    def write_snapshot_file(self, path: str) -> None:
        """Persist the frozen context for the local MCP server handoff only."""

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot_payload(), handle, allow_nan=False, sort_keys=True, separators=(",", ":"))


__all__ = ["EXPECTED_TOOL_NAMES", "CPRContextToolName", "FrozenCPRContextRegistry"]
