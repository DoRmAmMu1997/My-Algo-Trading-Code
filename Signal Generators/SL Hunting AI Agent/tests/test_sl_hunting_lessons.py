"""Phase 2 tests: lessons store, coach reflection, and gated prompt injection."""

from __future__ import annotations

import json

import pytest

from sl_hunting_agent import SLHuntingAgent
from sl_hunting_coach import CoachAgent, summarize_journal
from sl_hunting_lessons import (
    CoachOutput,
    ProposedLesson,
    add_proposed,
    consolidate,
    format_lessons,
    load_lessons,
    promote,
    proposed_to_record,
)


# --------------------------------------------------------------------------
# Lesson schema
# --------------------------------------------------------------------------

def test_proposed_lesson_validates_and_omits_min_max():
    p = ProposedLesson(scope="gap_down", lesson="prefer up after panic", rationale="r",
                        wins=3, losses=1, sample_size=4, confidence=6)
    assert p.confidence == 6
    with pytest.raises(Exception):
        ProposedLesson(scope="x", lesson="y", rationale="r", wins=1, losses=0, sample_size=1, confidence=11)
    with pytest.raises(Exception):
        ProposedLesson(scope="x", lesson="y", rationale="r", wins=-1, losses=0, sample_size=1, confidence=5)
    schema = ProposedLesson.model_json_schema()["properties"]["confidence"]
    assert "minimum" not in schema and "maximum" not in schema


# --------------------------------------------------------------------------
# consolidate + format
# --------------------------------------------------------------------------

def test_consolidate_dedupes_by_id_keeping_bigger_sample_and_caps():
    base = proposed_to_record(ProposedLesson(scope="s", lesson="L", rationale="r", wins=1, losses=0, sample_size=2, confidence=5))
    bigger = dict(base)
    bigger["evidence"] = {"wins": 5, "losses": 1, "sample_size": 6}
    out = consolidate([base, bigger])
    assert len(out) == 1 and out[0]["evidence"]["sample_size"] == 6
    # cap
    many = [proposed_to_record(ProposedLesson(scope=f"s{i}", lesson=f"L{i}", rationale="r", wins=i, losses=0, sample_size=i, confidence=5)) for i in range(1, 20)]
    assert len(consolidate(many, max_lessons=5)) == 5


def test_format_lessons_only_renders_approved():
    proposed = proposed_to_record(ProposedLesson(scope="s", lesson="be cautious", rationale="r", wins=1, losses=3, sample_size=4, confidence=4))
    assert format_lessons([proposed]) == ""  # status=proposed -> nothing
    approved = dict(proposed, status="approved")
    block = format_lessons([approved])
    assert "LEARNED LESSONS" in block and "be cautious" in block and "n=4" in block


def test_add_proposed_then_promote_roundtrip(tmp_path):
    proposed_path = str(tmp_path / "proposed.json")
    live_path = str(tmp_path / "lessons.json")
    recs = add_proposed(proposed_path, [
        ProposedLesson(scope="cross_index_wait", lesson="skip when cross_index says wait",
                       rationale="losers", wins=0, losses=4, sample_size=4, confidence=7),
    ])
    lid = recs[0]["id"]
    assert load_lessons(proposed_path)  # written
    assert format_lessons(load_lessons(live_path)) == ""  # nothing live yet

    promoted = promote(proposed_path, live_path, [lid])
    assert promoted == [lid]
    live = load_lessons(live_path)
    assert live and live[0]["status"] == "approved"
    assert "skip when cross_index says wait" in format_lessons(live)


# --------------------------------------------------------------------------
# Coach (fake runner — no SDK)
# --------------------------------------------------------------------------

class _FakeCoach:
    def __init__(self, text):
        self.text = text

    async def __call__(self, prompt, *, system_prompt, model, max_turns):
        return self.text


def test_coach_reflect_parses_lessons():
    canned = json.dumps({"lessons": [
        {"scope": "gap_down", "lesson": "prefer up after panic", "rationale": "3 winners",
         "wins": 3, "losses": 1, "sample_size": 4, "confidence": 6},
    ]})
    coach = CoachAgent(model="test", runner=_FakeCoach(canned))
    out = coach.reflect([{"outcome": {"points": 10}}], min_sample=3)
    assert len(out) == 1 and out[0].scope == "gap_down" and out[0].confidence == 6


def test_coach_reflect_empty_on_malformed():
    coach = CoachAgent(model="test", runner=_FakeCoach("this is not json"))
    assert coach.reflect([{"outcome": {"points": -5}}]) == []


def test_summarize_journal_renders_trades():
    rows = [{"direction": "LONG", "setup": "pivot", "confidence": 7, "followed_method": True,
             "context": {"cross_index": {"bias": "up"}}, "outcome": {"r_multiple": 2.0, "points": 30, "exit_reason": "target_hit"}}]
    text = summarize_journal(rows)
    assert "1 trades" in text and "pivot" in text and "target_hit" in text
    assert summarize_journal([]).startswith("No trades")


# --------------------------------------------------------------------------
# Gated injection into the agent prompt
# --------------------------------------------------------------------------

def test_agent_injects_lessons_block_before_output_contract():
    block = "LEARNED LESSONS\n- [gap_down] prefer up (W3/L1, n=4)"
    agent = SLHuntingAgent(model="test-model", lessons_block=block)
    prompt = agent._system_prompt
    assert "LEARNED LESSONS" in prompt
    # Lessons must come BEFORE the strict output contract (which stays last).
    assert prompt.index("LEARNED LESSONS") < prompt.index("FINAL OUTPUT FORMAT")
    # No lessons by default.
    assert "LEARNED LESSONS" not in SLHuntingAgent(model="test-model")._system_prompt
