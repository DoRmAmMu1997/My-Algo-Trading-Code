"""Bounded attempt budget around strict AI-output parsing.

Self-contained copy of the house helper used in
`../Streamlit Scanner App/backend/ai_validation.py`, kept local so this folder has
no cross-repo import. The agent asks Claude for a single strict-schema JSON object;
models occasionally return malformed output. `parse_with_retry` re-runs the agentic
loop a bounded number of times on parse/validation failures only — never on
infrastructure errors (SDK/CLI/usage-limit), which a retry cannot fix.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class StrictAIModel(BaseModel):
    """Strict base class for model-produced JSON (no coercion, no extra fields)."""

    model_config = ConfigDict(strict=True, extra="forbid")


class AIValidationError(RuntimeError):
    """Raised when AI output still fails strict parsing after every retry.

    Only safe structural metadata is exposed (attempt count + last error type) —
    never raw model text, which parser exceptions may contain.
    """

    def __init__(self, *, attempts: int, last_error_type: str) -> None:
        self.attempts = max(1, int(attempts))
        self.last_error_type = str(last_error_type)
        noun = "attempt" if self.attempts == 1 else "attempts"
        super().__init__(
            "AI output failed strict validation after "
            f"{self.attempts} {noun} (last error: {self.last_error_type})."
        )


def parse_with_retry[T](
    run_once: Callable[[], str],
    parse_once: Callable[[str], T],
    *,
    attempts: int,
    retry_on: tuple[type[BaseException], ...],
    label: str = "AI output",
) -> T:
    """Run an AI call and strictly parse it, retrying parse failures up to ``attempts``.

    ``run_once`` produces the model's raw final text; any exception it raises
    (SDK/CLI/usage-limit) propagates immediately and is NOT retried. ``parse_once``
    turns that text into the validated object, or raises one of ``retry_on`` when
    the output is malformed — those are retried. After every attempt fails, raises
    :class:`AIValidationError`.
    """
    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        text = run_once()  # infrastructure errors here propagate (outside the try)
        try:
            return parse_once(text)
        except retry_on as exc:
            if attempt >= total:
                raise AIValidationError(
                    attempts=total,
                    last_error_type=type(exc).__name__,
                ) from None
            logger.warning(
                "%s failed validation on attempt %d/%d (%s); retrying.",
                label, attempt, total, type(exc).__name__,
            )
    raise RuntimeError("parse_with_retry ran zero attempts")  # pragma: no cover
