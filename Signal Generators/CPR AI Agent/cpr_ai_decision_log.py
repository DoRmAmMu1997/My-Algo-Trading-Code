"""Append sanitized CPR Codex decisions as machine-readable JSONL audit rows.

Each line joins the exact frozen evidence, Codex proposal, host validation,
authoritative geometry, actual execution provenance, latency, token totals,
and MCP-call evidence for one decision.  That makes later paper/live review
possible without reconstructing mutable market state.

The logger is deliberately downstream of trading authority: it records what
happened but cannot make a decision executable.  Values under credential- or
execution-sensitive key tokens are removed recursively before anything is
written, and disabled logging performs no filesystem mutation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cpr_ai_tools import EXPECTED_TOOL_NAMES

# The isolated adapter reports terminal evidence only for these statuses. Keep
# the serializer closed so untrusted child text cannot become a JSONL disclosure.
_SAFE_TOOL_STATUSES = frozenset({"completed", "failed"})

_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "apikey",
        "apikeys",
        "broker",
        "brokers",
        "credential",
        "credentials",
        "path",
        "paths",
        "order",
        "orders",
        "password",
        "passwords",
        "secret",
        "secrets",
        "symbol",
        "symbols",
        "token",
        "quantity",
        "quantities",
        "reasoning",
        "response",
        "venue",
        "venues",
    }
)


def _sensitive_key(key: Any) -> bool:
    """Match credential/execution words as tokens, not innocent substrings.

    Substring matching discarded useful fields such as ``ordered`` and
    ``authoritative_geometry``.  Splitting snake_case, punctuation, and
    camelCase lets the filter remove ``api_keys``/``apiKeys`` without treating
    innocent word fragments as secrets.  Aggregate token-usage counters remain
    explicitly safe; authentication tokens and order/broker fields do not.
    """

    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", separated.lower()) if token)
    if not tokens:
        return False
    if tokens == ("token", "usage"):
        return False
    # CamelCase ``apiKeys`` and snake_case ``api_keys`` both become the two
    # tokens ``api`` and ``keys``.  Treat either singular or plural spelling
    # as credential data while leaving unrelated fields such as ``keys_seen``
    # available to the audit log.
    if "api" in tokens and ("key" in tokens or "keys" in tokens):
        return True
    if "tokens" in tokens and any(
        token in {
            "access",
            "auth",
            "authentication",
            "authorization",
            "bearer",
            "refresh",
            "session",
        }
        for token in tokens
    ):
        return True
    return any(
        token in _SENSITIVE_KEY_TOKENS
        for token in tokens
    )


def _sanitized(value: Any) -> Any:
    """Recursively copy audit evidence while dropping sensitive mapping fields.

    Lists, dataclasses, and Pydantic models may nest sensitive mappings several
    layers deep, so redaction happens before JSON serialization rather than at
    selected call sites.  Primitive market values pass through unchanged.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _sanitized(item)
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitized(item) for item in value]
    if hasattr(value, "model_dump"):
        return _sanitized(value.model_dump())
    if hasattr(value, "__dict__"):
        return _sanitized(vars(value))
    return value


class CPRDecisionLogger:
    """Write one sanitized, append-only host-decision record when enabled.

    JSONL keeps every decision independently parseable and avoids rewriting
    earlier audit history if the process stops unexpectedly.
    """

    def __init__(self, path: str, *, enabled: bool = True) -> None:
        """Store the destination without creating files while logging is idle."""

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
        audit_stage: str = "DIRECT",
        bar_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one complete sanitized record after a host decision.

        ``DIRECT`` is a safe default for older diagnostics that call this
        logger outside the worker.  The CPR worker labels its first record
        ``PRE_ACTION`` and its follow-up provenance record ``POST_ACTION``.
        Parent directories are created only after the enabled guard passes.
        """

        if not self.enabled:
            return
        bar = dict(bar_metadata or {})
        # The outcome is the authority for this value: it was sampled once at
        # validation and must not be rebuilt from mutable shared market data.
        bar["validation_current_signature"] = getattr(
            outcome, "validation_current_signature", None
        )
        row = _sanitized(
            {
                "recorded_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
                "audit_stage": audit_stage,
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
                # A value of two means the first isolated turn did not provide
                # complete frozen-tool evidence and the host used its one safe,
                # same-snapshot retry.  Recording it keeps latency and token
                # totals understandable during later operational review.
                "inference_attempts": int(getattr(outcome, "inference_attempts", 1)),
                "token_usage": token_usage,
                "tool_evidence": self._safe_tool_records(tool_evidence),
                "attempt_evidence": self._safe_attempt_evidence(
                    getattr(outcome, "attempt_evidence", ())
                ),
                "bar": bar,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    @staticmethod
    def _safe_tool_records(records: Any) -> list[dict[str, str]]:
        """Keep only canonical MCP tool names and terminal statuses.

        SDK error text can contain a local path, a response body, or credentials.
        The typed evidence code already says whether a tool failed, so retaining
        opaque error text is unnecessary and unsafe for an operational audit.
        """

        safe_records: list[dict[str, str]] = []
        for record in records:
            if isinstance(record, Mapping):
                tool = record.get("tool")
                status = record.get("status")
            else:
                tool = getattr(record, "tool", None)
                status = getattr(record, "status", None)
            if tool not in EXPECTED_TOOL_NAMES or status not in _SAFE_TOOL_STATUSES:
                # The typed evidence code still records why this attempt was
                # rejected, so retain no arbitrary tool/status payload at all.
                continue
            safe_records.append({"tool": tool, "status": status})
        return safe_records

    @classmethod
    def _safe_attempt_evidence(cls, attempts: Any) -> list[dict[str, Any]]:
        """Serialize Task 2 attempt facts without model text or SDK errors."""

        safe_attempts: list[dict[str, Any]] = []
        for attempt in attempts:
            if isinstance(attempt, Mapping):
                attempt_number = attempt.get("attempt_number", 0)
                request_kind = attempt.get("request_kind")
                evidence_code = attempt.get("evidence_code")
                tool_records = attempt.get("tool_records", ())
                token_usage = attempt.get("token_usage", {})
            else:
                attempt_number = getattr(attempt, "attempt_number", 0)
                request_kind = getattr(attempt, "request_kind", None)
                evidence_code = getattr(attempt, "evidence_code", None)
                tool_records = getattr(attempt, "tool_records", ())
                token_usage = getattr(attempt, "token_usage", {})
            safe_attempts.append(
                {
                    "attempt_number": int(attempt_number),
                    "request_kind": str(request_kind),
                    "evidence_code": evidence_code,
                    "tool_records": cls._safe_tool_records(tool_records),
                    "token_usage": dict(token_usage),
                }
            )
        return safe_attempts


__all__ = ["CPRDecisionLogger"]
