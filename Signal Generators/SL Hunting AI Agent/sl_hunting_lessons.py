"""Lessons store for the SL Hunting AI Agent (v3 — learn from mistakes).

A small, HUMAN-GATED knowledge base the agent grows from its own trades. The coach
(`sl_hunting_coach.py`) writes PROPOSED lessons here; the operator reviews and
`--promote`s the good ones into the live store; the agent injects only APPROVED
lessons into its prompt (gated by `SL_HUNTING_LESSONS_ENABLED`). The store is bounded
and de-duplicated so the prompt never bloats and stale/contradictory lessons are pruned.

`ProposedLesson` is the coach's strict output schema. ``StoredLesson`` is the equally
strict on-disk schema: malformed records are ignored and every approved record carries
a SHA-256 digest of its reviewed content. That binding prevents an approved id from
silently pointing at changed text later.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from sl_hunting_ai_validation import StrictAIModel

logger = logging.getLogger(__name__)

# Cap how many APPROVED lessons are injected into the prompt (keep it bounded).
MAX_LIVE_LESSONS = 12
MAX_SCOPE_CHARS = 80
MAX_LESSON_CHARS = 280
MAX_RATIONALE_CHARS = 1_000


def _bounded_single_line(value: str, *, field: str, maximum: int) -> str:
    """Validate human/model text before it can enter either lessons store.

    Newlines and control characters are intentionally rejected. Lessons are rendered
    into a system prompt later, so keeping each field short and single-line makes the
    boundary unambiguous and prevents one record from reshaping the prompt block.
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be blank")
    if len(cleaned) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in cleaned):
        raise ValueError(f"{field} must be a single printable line")
    return cleaned


class ProposedLesson(StrictAIModel):
    """One lesson the coach proposes — its strict output schema (validated like the
    agent's decision; integer ranges use `@field_validator` so the JSON schema stays
    free of min/max, which Claude rejects)."""

    scope: str = Field(
        description="Short tag for WHEN this applies (e.g. 'pivot_retest', 'gap_down', 'cross_index_wait')."
    )
    lesson: str = Field(description="One-line tendency/directive — NOT a hard law.")
    rationale: str = Field(description="Why: the pattern across the trades behind it.")
    wins: int = Field(description="Winning trades supporting it.")
    losses: int = Field(description="Losing trades supporting it.")
    sample_size: int = Field(description="Total trades this lesson is drawn from.")
    confidence: int = Field(description="0-10 confidence given the evidence.")

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, value: str) -> str:
        return _bounded_single_line(value, field="scope", maximum=MAX_SCOPE_CHARS)

    @field_validator("lesson")
    @classmethod
    def _validate_lesson(cls, value: str) -> str:
        return _bounded_single_line(value, field="lesson", maximum=MAX_LESSON_CHARS)

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        return _bounded_single_line(value, field="rationale", maximum=MAX_RATIONALE_CHARS)

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

    @model_validator(mode="after")
    def _validate_sample_counts(self) -> ProposedLesson:
        if self.wins + self.losses > self.sample_size:
            raise ValueError("wins + losses cannot exceed sample_size")
        return self


class CoachOutput(StrictAIModel):
    """The coach's final message: a list of proposed lessons (possibly empty)."""

    lessons: list[ProposedLesson] = Field(default_factory=list)


class LessonEvidence(StrictAIModel):
    """Typed evidence stored alongside a proposed or approved lesson."""

    wins: int
    losses: int
    sample_size: int

    @field_validator("wins", "losses", "sample_size")
    @classmethod
    def _validate_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("lesson evidence counts must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_total(self) -> LessonEvidence:
        if self.wins + self.losses > self.sample_size:
            raise ValueError("evidence wins + losses cannot exceed sample_size")
        return self


class StoredLesson(StrictAIModel):
    """Strict, tamper-evident schema for one lessons JSON record."""

    id: str
    scope: str
    lesson: str
    rationale: str
    evidence: LessonEvidence
    confidence: int
    status: Literal["proposed", "approved"]
    created_at: str
    updated_at: str
    approval_digest: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = _bounded_single_line(value, field="id", maximum=48)
        if any(not (char.isalnum() or char == "-") for char in cleaned):
            raise ValueError("id may contain only letters, numbers, and hyphens")
        return cleaned

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, value: str) -> str:
        return _bounded_single_line(value, field="scope", maximum=MAX_SCOPE_CHARS)

    @field_validator("lesson")
    @classmethod
    def _validate_lesson(cls, value: str) -> str:
        return _bounded_single_line(value, field="lesson", maximum=MAX_LESSON_CHARS)

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        return _bounded_single_line(value, field="rationale", maximum=MAX_RATIONALE_CHARS)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: int) -> int:
        if not 0 <= value <= 10:
            raise ValueError("confidence must be 0-10")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        cleaned = _bounded_single_line(value, field="timestamp", maximum=40)
        try:
            datetime.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError("lesson timestamp must be ISO-8601") from exc
        return cleaned

    @model_validator(mode="after")
    def _validate_identity_and_approval(self) -> StoredLesson:
        if self.id != _slug(self.scope, self.lesson):
            raise ValueError("lesson id does not match its scope and content")
        if self.status == "approved":
            expected = lesson_content_digest(self)
            if not self.approval_digest or not hmac.compare_digest(self.approval_digest, expected):
                raise ValueError("approved lesson content digest is missing or invalid")
        elif self.approval_digest is not None:
            raise ValueError("proposed lessons must not carry an approval digest")
        return self


def _slug(scope: str, lesson: str) -> str:
    """Build a STABLE id from scope + lesson text (lowercase, non-alphanumerics → '-').

    Because the same lesson always slugs to the same id, ``consolidate`` can recognise
    duplicates across coach runs and keep just the best-evidenced copy. Capped to 48
    chars; falls back to ``"lesson"`` if the text had nothing alphanumeric.
    """
    base = f"{scope}-{lesson}".lower()
    cleaned = "".join(c if c.isalnum() else "-" for c in base)
    return "-".join(filter(None, cleaned.split("-")))[:48] or "lesson"


def _now() -> str:
    """Current UTC time as a stable ISO-8601 string (used for created/updated stamps)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def lesson_content_digest(record: StoredLesson | dict[str, Any]) -> str:
    """Return the canonical digest that binds operator approval to lesson content.

    What the digest is FOR: the operator approves a lesson by reading its exact
    text.  If the stored file were later edited (by a bug, a merge, or a
    prompt-injection-shaped payload), the approved id would silently point at
    words the operator never reviewed -- and those words are injected into the
    live agent's prompt.  ``StoredLesson`` therefore recomputes this digest on
    every load and rejects any approved record whose content no longer matches.

    Only the reviewed CONTENT participates (id, scope, lesson, rationale,
    evidence, confidence).  Timestamps and status are deliberately excluded so
    a harmless metadata touch cannot invalidate a genuine approval.  The
    canonical JSON form (sorted keys, fixed separators) makes the digest
    stable regardless of dict ordering or formatting.
    """
    source = record.model_dump(mode="json") if isinstance(record, StoredLesson) else record
    evidence = source.get("evidence") or {}
    content = {
        "id": source.get("id"),
        "scope": source.get("scope"),
        "lesson": source.get("lesson"),
        "rationale": source.get("rationale"),
        "evidence": {
            "wins": evidence.get("wins"),
            "losses": evidence.get("losses"),
            "sample_size": evidence.get("sample_size"),
        },
        "confidence": source.get("confidence"),
    }
    canonical = json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def proposed_to_record(
    p: ProposedLesson, *, status: Literal["proposed", "approved"] = "proposed"
) -> dict[str, Any]:
    """Wrap a coach `ProposedLesson` into a stored record (id/status/timestamps)."""
    now = _now()
    record: dict[str, Any] = {
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
    if status == "approved":
        record["approval_digest"] = lesson_content_digest(record)
    return StoredLesson.model_validate(record).model_dump(mode="json", exclude_none=True)


def _validated_records(lessons: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Validate external records and return safe JSON dictionaries plus reject count."""
    valid: list[dict[str, Any]] = []
    rejected = 0
    for record in lessons:
        try:
            parsed = StoredLesson.model_validate(record)
        except (TypeError, ValidationError, ValueError):
            rejected += 1
            continue
        valid.append(parsed.model_dump(mode="json", exclude_none=True))
    return valid, rejected


def load_lessons(path: str) -> list[dict[str, Any]]:
    """Load only schema-valid, digest-valid lesson records from a JSON list."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Lessons file %s must contain a JSON list; ignoring it.", path)
            return []
        records, rejected = _validated_records([item for item in data if isinstance(item, dict)])
        rejected += sum(1 for item in data if not isinstance(item, dict))
        if rejected:
            logger.warning("Ignored %d malformed or tampered lesson record(s) in %s.", rejected, path)
        return records
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Could not read lessons file %s", path, exc_info=True)
        return []


def save_lessons(path: str, lessons: list[dict[str, Any]]) -> None:
    """Validate then atomically replace ``path`` with the lessons JSON list.

    The temporary file is flushed and closed before ``os.replace``. Readers therefore
    see either the previous complete store or the new complete store, never a partially
    written JSON document after a crash.
    """
    # Validate BEFORE opening any file: a malformed record must fail loudly
    # here rather than half-replace the store.
    validated = [
        StoredLesson.model_validate(record).model_dump(mode="json", exclude_none=True)
        for record in lessons
    ]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # The temp file must live in the SAME directory as the target: os.replace
    # is atomic only within one filesystem, and a cross-device rename would
    # degrade to the copy-then-delete window this function exists to remove.
    directory = parent or "."
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(validated, file_handle, indent=2, ensure_ascii=False)
            file_handle.write("\n")
            # flush() pushes Python's buffer to the OS; fsync() pushes the OS
            # buffer to disk. Only after BOTH is the temp file guaranteed
            # complete on disk, making the replace below crash-safe.
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        # On success the temp path no longer exists (it BECAME the store); on
        # any failure this removes the orphaned partial file.
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _sample(lesson: dict[str, Any]) -> int:
    """Pull the supporting trade count out of a lesson's ``evidence`` (0 if absent)."""
    return int((lesson.get("evidence") or {}).get("sample_size", 0))


def consolidate(lessons: list[dict[str, Any]], max_lessons: int = MAX_LIVE_LESSONS) -> list[dict[str, Any]]:
    """De-duplicate by id (keep the better-evidenced copy), rank by sample size then
    recency, and cap at `max_lessons` — so the store stays bounded and non-contradictory."""
    by_id: dict[str, dict[str, Any]] = {}
    # Strict validation first: consolidation is a write-path chokepoint (both
    # add_proposed and promote funnel through it), so a malformed record dies
    # here instead of ever reaching the saved store.
    validated, _rejected = _validated_records(lessons)
    for lesson in validated:
        lid = lesson.get("id")
        if not lid:
            continue
        # Same slug = same lesson text seen across coach runs.  Keep the copy
        # backed by MORE trades: growing evidence should win, and a rerun on a
        # shrunken journal must not downgrade a well-supported lesson.
        prev = by_id.get(lid)
        if prev is None or _sample(lesson) > _sample(prev):
            by_id[lid] = lesson
    # Rank best-evidenced first (recency breaks ties), then cap: with a
    # bounded list, a new well-supported lesson naturally evicts the weakest
    # one, keeping the prompt block a stable size forever.
    ranked = sorted(
        by_id.values(),
        key=lambda rec: (_sample(rec), str(rec.get("updated_at", ""))),
        reverse=True,
    )
    return ranked[:max_lessons]


def format_lessons(lessons: list[dict[str, Any]]) -> str:
    """Render APPROVED lessons as a `LEARNED LESSONS` prompt block ('' if none)."""
    approved = consolidate([rec for rec in lessons if rec.get("status") == "approved"])
    if not approved:
        return ""
    out = [
        "LEARNED LESSONS (from your own past trades — tendencies, not laws)",
        "-----------------------------------------------------------------",
        "Distilled from your trade journal and APPROVED by the operator. Weigh them as",
        "soft priors that refine the method; they never override a clear live read.",
        "",
    ]
    for rec in approved:
        ev = rec.get("evidence") or {}
        out.append(
            f"- [{rec.get('scope', '?')}] {str(rec.get('lesson', '')).strip()} "
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
    proposed = {rec.get("id"): rec for rec in load_lessons(proposed_path)}
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
        rec["approval_digest"] = lesson_content_digest(rec)
        rec = StoredLesson.model_validate(rec).model_dump(mode="json", exclude_none=True)
        live.append(rec)
        promoted.append(lid)
    save_lessons(live_path, consolidate(live))
    return promoted
