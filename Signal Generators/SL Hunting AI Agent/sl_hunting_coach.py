"""Reflection COACH for the SL Hunting AI Agent (v3 — learn from mistakes).

A SEPARATE, read-only agent that reads the trade journal and PROPOSES a few concise
lessons to improve future decisions. It runs OFF the live trading loop (a manual CLI
you invoke end-of-day), so — unlike the trading agent — it has no side effects and a
retry on malformed output IS safe (reuses `parse_with_retry`).

Human-gated, paper-first: the coach only writes PROPOSED lessons; you review them and
`--promote` the good ones into the live store; only then does the agent inject them
(and only when `SL_HUNTING_LESSONS_ENABLED`).

Usage:
    python sl_hunting_coach.py --reflect [--since 2026-06-01] [--fake]
    python sl_hunting_coach.py --list
    python sl_hunting_coach.py --promote <lesson-id> [<lesson-id> ...]

Subscription billing: keep ANTHROPIC_API_KEY UNSET (same as the trading agent).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
from collections.abc import Awaitable
from typing import Any

from pydantic import ValidationError

from sl_hunting_ai_validation import parse_with_retry
from sl_hunting_lessons import (
    CoachOutput,
    ProposedLesson,
    add_proposed,
    consolidate,
    load_lessons,
    promote,
)

logger = logging.getLogger("sl_hunting_coach")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_DIR = os.path.join(_REPO_ROOT, "Backtest Outputs")
DEFAULT_JOURNAL = os.path.join(_OUT_DIR, "sl_hunting_journal.jsonl")
DEFAULT_PROPOSED = os.path.join(_OUT_DIR, "sl_hunting_lessons_proposed.json")
# The live (approved) store sits IN the agent folder so it ships with the strategy.
DEFAULT_LIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")


def _coach_system_prompt(min_sample: int, max_lessons: int) -> str:
    """Build the coach's system prompt — the skeptical "reviewer" persona + its guardrails.

    Bakes the overfitting defences into the instructions themselves: a minimum sample
    before any lesson, tendencies-not-laws phrasing, process-vs-outcome separation, and
    "return an empty list if the evidence is thin" (so it doesn't invent lessons).
    """
    return (
        "You are a trading COACH reviewing the SL Hunting agent's OWN trade journal to "
        "propose a few LESSONS that would improve its future decisions. You are rigorous "
        "and skeptical: small samples are noise and the market is non-stationary.\n\n"
        "Rules:\n"
        f"- Propose a lesson ONLY if at least {min_sample} journal trades support it.\n"
        "- Phrase each as a TENDENCY, not a hard law (\"prefer\", \"be cautious\", \"size "
        "down\"), scoped to a condition (a setup, a gap type, a cross_index state, etc.).\n"
        "- Separate PROCESS from OUTCOME: a trade with followed_method=true that lost is "
        "usually variance, NOT a mistake. Focus on repeated patterns where the PROCESS was "
        "weak (followed_method=false) or a specific condition consistently underperforms.\n"
        "- Prefer recent, repeated evidence. Be concise. Propose at most "
        f"{max_lessons} lessons. If the evidence is too thin, return an EMPTY list — do "
        "not invent lessons.\n\n"
        "Your FINAL message must be ONLY this JSON object and nothing else:\n"
        '{"lessons": [{"scope": str, "lesson": str, "rationale": str, "wins": int, '
        '"losses": int, "sample_size": int, "confidence": int (0-10)}, ...]}'
    )


def load_journal(path: str, since: str | None = None) -> list[dict[str, Any]]:
    """Read the journal JSONL (one row per closed trade); optional `since` date filter."""
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since and str(row.get("opened_at", "")) < since:
                    continue
                rows.append(row)
    except OSError:
        logger.warning("Could not read journal %s", path, exc_info=True)
    return rows


def summarize_journal(rows: list[dict[str, Any]], limit: int = 60) -> str:
    """Compact text rendering of the journal for the coach prompt (bounded)."""
    if not rows:
        return "No trades in the journal yet."
    n = len(rows)
    wins = sum(1 for r in rows if float((r.get("outcome") or {}).get("points", 0) or 0) > 0)
    lines = [f"Journal: {n} trades, {wins} winners ({100.0 * wins / n:.0f}%). Recent trades:", ""]
    for r in rows[-limit:]:
        out = r.get("outcome") or {}
        ctx = r.get("context") or {}
        xi = ctx.get("cross_index") or {}
        lines.append(
            f"- {r.get('direction')} setup={r.get('setup')} conf={r.get('confidence')} "
            f"followed_method={r.get('followed_method')} cross_bias={xi.get('bias')} "
            f"=> R={out.get('r_multiple')} pts={out.get('points')} exit={out.get('exit_reason')}"
        )
    return "\n".join(lines)


class CoachAgent:
    """Single-pass reflection agent over the journal (read-only; retry is safe)."""

    MAX_TURNS = 2

    def __init__(self, model: str, *, runner=None, fast_mode: bool = False, attempts: int = 2) -> None:
        self._model = model
        self._runner = runner
        self._fast_mode = bool(fast_mode)
        self._attempts = max(1, int(attempts))

    async def _default_run(self, prompt: str, *, system_prompt: str, model: str, max_turns: int) -> str:
        """Run one read-only reflection pass on the Claude Agent SDK; return final text.

        No tools and no order venue here (the coach only reads the journal text we pass
        in), so it's far simpler than the trading agent's loop. SDK imported lazily.
        """
        try:
            import claude_agent_sdk as claude_sdk  # type: ignore[import-not-found, unused-ignore]
            from claude_agent_sdk import (  # type: ignore[import-not-found, unused-ignore]
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                query,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "claude-agent-sdk is not installed. `pip install claude-agent-sdk` and "
                "sign in once with the Claude CLI (keep ANTHROPIC_API_KEY UNSET)."
            ) from exc

        ThinkingConfigDisabled = getattr(claude_sdk, "ThinkingConfigDisabled", None)
        options_kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "permission_mode": "dontAsk",
            "setting_sources": [],
        }
        if self._fast_mode and ThinkingConfigDisabled is not None:
            options_kwargs["thinking"] = ThinkingConfigDisabled()
        options = ClaudeAgentOptions(**options_kwargs)

        final_text = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                if getattr(message, "result", None):
                    final_text = message.result
            elif isinstance(message, AssistantMessage):
                for block in getattr(message, "content", None) or []:
                    text = getattr(block, "text", None)
                    if text:
                        final_text = text
        return final_text

    @staticmethod
    def _run_sync(coro: Awaitable[str]) -> str:
        """Drive the async run from a plain CLI (own thread + event loop; Windows-safe).

        Same bridge idea as the trading agent — a dedicated worker thread with its own
        loop (ProactorEventLoop on Windows, needed to spawn the Claude CLI subprocess).
        """
        def _runner() -> str:
            loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()

    def reflect(self, journal_rows: list[dict[str, Any]], *, min_sample: int = 5, max_lessons: int = 6) -> list[ProposedLesson]:
        """Run one reflection pass and return the proposed lessons (possibly empty)."""
        system_prompt = _coach_system_prompt(min_sample, max_lessons)
        prompt = (
            "Review this trade journal and propose lessons per your rules.\n\n"
            + summarize_journal(journal_rows)
        )
        runner = self._runner or self._default_run

        def _run_once() -> str:
            return self._run_sync(runner(prompt, system_prompt=system_prompt, model=self._model, max_turns=self.MAX_TURNS))

        def _parse(text: str) -> CoachOutput:
            payload = _extract_json_object(text)
            if payload is None:
                raise ValueError("Coach did not return a parseable CoachOutput JSON object.")
            return CoachOutput.model_validate(payload)

        try:
            output = parse_with_retry(
                _run_once, _parse, attempts=self._attempts,
                retry_on=(ValueError, ValidationError), label="CoachOutput",
            )
        except Exception as exc:  # noqa: BLE001 - reflection is best-effort, never fatal
            logger.warning("Coach reflection failed (%s); no lessons proposed.", type(exc).__name__)
            return []
        return output.lessons


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the coach's `{"lessons": [...]}` JSON out of its final message.

    Tolerant of a ```json fence or a leading sentence (fenced block first, else the
    outermost {...}); returns None if nothing parses. A local twin of the agent's
    extractor, kept here so the coach module stands alone.
    """
    import re

    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class _FakeEmptyCoach:
    """Built-in --fake runner: proposes nothing (pipeline smoke test, no SDK/cost)."""

    async def __call__(self, prompt, *, system_prompt, model, max_turns):
        return json.dumps({"lessons": []})


def _env(name: str, default: str) -> str:
    """Read an env var, trimming whitespace; fall back to ``default`` when blank/unset."""
    val = os.getenv(name, "")
    return val.strip() if val and val.strip() else default


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SL Hunting reflection coach (human-gated lessons).")
    parser.add_argument("--reflect", action="store_true", help="Read the journal and propose lessons (default).")
    parser.add_argument("--list", action="store_true", help="List proposed + live lessons and exit.")
    parser.add_argument("--promote", nargs="+", metavar="ID", help="Promote proposed lesson id(s) into the live store.")
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--proposed", default=DEFAULT_PROPOSED)
    parser.add_argument("--live", default=DEFAULT_LIVE)
    parser.add_argument("--since", help="Only consider trades opened on/after this date (YYYY-MM-DD).")
    parser.add_argument("--model", default=_env("SL_HUNTING_COACH_MODEL", _env("SL_HUNTING_MODEL", "claude-opus-4-8")))
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fake", action="store_true", help="Use the built-in no-op coach (no SDK/cost).")
    parser.add_argument("--min-sample", type=int, default=5)
    parser.add_argument("--max-lessons", type=int, default=6)
    args = parser.parse_args(argv)

    if args.promote:
        promoted = promote(args.proposed, args.live, args.promote)
        logger.info("Promoted %d lesson(s) into %s: %s", len(promoted), args.live, promoted)
        return 0

    if args.list:
        for label, path in (("PROPOSED", args.proposed), ("LIVE (approved)", args.live)):
            items = load_lessons(path)
            logger.info("=== %s (%d) ===", label, len(items))
            for l in consolidate(items, max_lessons=100):
                ev = l.get("evidence") or {}
                logger.info("  [%s] %s :: %s (W%s/L%s n%s, conf=%s)", l.get("id"), l.get("scope"),
                            l.get("lesson"), ev.get("wins"), ev.get("losses"), ev.get("sample_size"), l.get("confidence"))
        return 0

    # default: reflect
    rows = load_journal(args.journal, since=args.since)
    logger.info("Loaded %d journal trades from %s.", len(rows), args.journal)
    coach = CoachAgent(model=args.model, runner=_FakeEmptyCoach() if args.fake else None, fast_mode=args.fast)
    proposals = coach.reflect(rows, min_sample=args.min_sample, max_lessons=args.max_lessons)
    if not proposals:
        logger.info("No lessons proposed.")
        return 0
    records = add_proposed(args.proposed, proposals)
    logger.info("Proposed %d lesson(s) -> %s (review, then --promote the good ones):", len(records), args.proposed)
    for r in records:
        logger.info("  [%s] %s", r["id"], r["lesson"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
