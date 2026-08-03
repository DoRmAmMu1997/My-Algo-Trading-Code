"""Append sanitized CPR Codex decisions as machine-readable JSONL audit rows.

Logs are intended for later host execution analysis, so they retain frozen
evidence and validation outcomes while recursively removing values that look
like credentials.  Logging must never make a decision executable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SENSITIVE_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "AUTH", "BROKER", "ORDER", "VENUE")


def _sanitized(value: Any) -> Any:
    """Copy JSON-compatible evidence while dropping sensitive mapping fields."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitized(item)
            for key, item in value.items()
            if not any(marker in str(key).upper() for marker in _SENSITIVE_MARKERS)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitized(item) for item in value]
    if hasattr(value, "model_dump"):
        return _sanitized(value.model_dump())
    if hasattr(value, "__dict__"):
        return _sanitized(vars(value))
    return value


class CPRDecisionLogger:
    """Write one sanitized, append-only record per host decision when enabled."""

    def __init__(self, path: str, *, enabled: bool = True) -> None:
        """Store the configured audit destination without creating it eagerly."""

        self.path = Path(path)
        self.enabled = enabled

    def write(
        self,
        *,
        frozen_context: Mapping[str, Any],
        proposal: Any | None,
        outcome: Any,
        latency_ms: int,
        token_usage: Mapping[str, Any],
        tool_evidence: list[Mapping[str, Any]],
        execution: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a complete non-secret record; execution defaults to order-free."""

        if not self.enabled:
            return
        row = _sanitized(
            {
                "frozen_context": frozen_context,
                "proposal": proposal,
                "accepted_regime": outcome.accepted_regime,
                "validation": {
                    "accepted": outcome.accepted,
                    "code": outcome.validation_code,
                    "reason": outcome.validation_reason,
                },
                "authoritative_geometry": {
                    "action": outcome.action,
                    "entry_price": outcome.entry_price,
                    "stop_price": outcome.stop_price,
                    "milestone_price": outcome.milestone_price,
                    "final_target_price": outcome.final_target_price,
                    "risk_points": outcome.risk_points,
                    "scale_in_permitted": outcome.scale_in_permitted,
                },
                "execution": execution or {"mode": "ORDER_FREE", "submitted": False},
                "latency_ms": latency_ms,
                "token_usage": token_usage,
                "tool_evidence": tool_evidence,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


__all__ = ["CPRDecisionLogger"]
