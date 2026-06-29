"""Lessons store for the SL Hunting AI Agent (v3 — learn from mistakes).

A small, HUMAN-GATED knowledge base the agent grows from its own trades. The coach
(`sl_hunting_coach.py`) writes PROPOSED lessons here; the operator reviews and
`--promote`s the good ones into the live store; the agent injects only APPROVED
lessons into its prompt (gated by `SL_HUNTING_LESSONS_ENABLED`). The store is bounded
and de-duplicated so the prompt never bloats and stale/contradictory lessons are pruned.

`ProposedLesson` is the coach's strict output schema; stored lessons are plain dicts
with an id/status/timestamps + the evidence behind them.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from pydantic import Field, field_validator

from sl_hunting_ai_validation import StrictAIModel

logger = logging.getLogger(__name__)

# Cap how many APPROVED lessons are injected into the prompt (keep it bounded).
MAX_LIVE_LESSONS = 12


class ProposedLesson(StrictAIModel):
    """One lesson the coach proposes — its strict output schema (validated like the
    agent's decision; integer ranges use `@field_validator` so the JSON schema stays
    free of min/max, which Claude rejects)."""

    scope: str = Field(description="Short tag for WHEN this applies (e.g. 'pivot_retest', 'gap_down', 'cross_index_wait').")
    lesson: str = Field(description="One-line tendency/directive — NOT a hard law.")
    rationale: str = Field(description="Why: the pattern across the trades behind it.")
    wins: int = Field(description="Winning trades supporting it.")
    losses: int = Field(description="Losing trades supporting it.")
    sample_size: int = Field(description="Total trades this lesson is drawn from.")
    confidence: int = Field(description="0-10 confidence given the evidence.")

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: int) -> int:
        if not 0 <= value <= 10:
            raise ValueError(f"confidence must be 0-10, got {value}")
        return value

    @field_validator("wins", "losses", "sample_size")
    @classmethod
    def _validate_nonneg(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trade counts must be >= 0")
        return value


class CoachOutput(StrictAIModel):
    """The coach's final message: a list of proposed lessons (possibly empty)."""

    lessons: list[ProposedLesson] = Field(default_factory=list)


def _slug(scope: str, lesson: str) -> str:
    base = f"{scope}-{lesson}".lower()
    cleaned = "".join(c if c.isalnum() else "-" for c in base)
    return "-".join(filter(None, cleaned.split("-")))[:48] or "lesson"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def proposed_to_record(p: ProposedLesson, *, status: str = "proposed") -> dict[str, Any]:
    """Wrap a coach `ProposedLesson` into a stored record (id/status/timestamps)."""
    now = _now()
    return {
        "id": _slug(p.scope, p.lesson),
        "scope": p.scope,
        "lesson": p.lesson,
        "rationale": p.rationale,
        "evidence": {"wins": p.wins, "losses": p.losses, "sample_size": p.sample_size},
        "confidence": p.confidence,
        "status": status,
        "created_at": now,
        "updated_at": now,
    }


def load_lessons(path: str) -> list[dict[str, Any]]:
    """Load a lessons JSON list (returns [] when missing/unreadable)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read lessons file %s", path, exc_info=True)
        return []


def save_lessons(path: str, lessons: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=2)


def _sample(lesson: dict[str, Any]) -> int:
    return int((lesson.get("evidence") or {}).get("sample_size", 0))


def consolidate(lessons: list[dict[str, Any]], max_lessons: int = MAX_LIVE_LESSONS) -> list[dict[str, Any]]:
    """De-duplicate by id (keep the better-evidenced copy), rank by sample size then
    recency, and cap at `max_lessons` — so the store stays bounded and non-contradictory."""
    by_id: dict[str, dict[str, Any]] = {}
    for lesson in lessons:
        lid = lesson.get("id")
        if not lid:
            continue
        prev = by_id.get(lid)
        if prev is None or _sample(lesson) > _sample(prev):
            by_id[lid] = lesson
    ranked = sorted(
        by_id.values(),
        key=lambda l: (_sample(l), str(l.get("updated_at", ""))),
        reverse=True,
    )
    return ranked[:max_lessons]


def format_lessons(lessons: list[dict[str, Any]]) -> str:
    """Render APPROVED lessons as a `LEARNED LESSONS` prompt block ('' if none)."""
    approved = [l for l in lessons if l.get("status") == "approved"]
    if not approved:
        return ""
    out = [
        "LEARNED LESSONS (from your own past trades — tendencies, not laws)",
        "-----------------------------------------------------------------",
        "Distilled from your trade journal and APPROVED by the operator. Weigh them as",
        "soft priors that refine the method; they never override a clear live read.",
        "",
    ]
    for l in consolidate(approved):
        ev = l.get("evidence") or {}
        out.append(
            f"- [{l.get('scope', '?')}] {str(l.get('lesson', '')).strip()} "
            f"(W{ev.get('wins', 0)}/L{ev.get('losses', 0)}, n={ev.get('sample_size', 0)})"
        )
    return "\n".join(out)


def add_proposed(proposed_path: str, proposed: list[ProposedLesson]) -> list[dict[str, Any]]:
    """Append coach proposals to the proposed store (deduped). Returns the new records."""
    records = [proposed_to_record(p) for p in proposed]
    merged = consolidate(load_lessons(proposed_path) + records, max_lessons=100)
    save_lessons(proposed_path, merged)
    return records


def promote(proposed_path: str, live_path: str, ids: list[str]) -> list[str]:
    """Move selected PROPOSED lessons into the live (APPROVED) store — the human gate."""
    proposed = {l.get("id"): l for l in load_lessons(proposed_path)}
    live = load_lessons(live_path)
    promoted: list[str] = []
    for lid in ids:
        src = proposed.get(lid)
        if src is None:
            logger.warning("promote: unknown proposed lesson id %r", lid)
            continue
        rec = dict(src)
        rec["status"] = "approved"
        rec["updated_at"] = _now()
        live.append(rec)
        promoted.append(lid)
    save_lessons(live_path, consolidate(live))
    return promoted
