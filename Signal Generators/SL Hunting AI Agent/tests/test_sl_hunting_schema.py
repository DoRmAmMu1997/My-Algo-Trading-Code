"""Strict-schema tests for the SLHuntingDecision output contract."""

from __future__ import annotations

import pytest
from sl_hunting_agent import SLHuntingDecision
from sl_hunting_knowledge import FINAL_OUTPUT_INSTRUCTION, build_system_prompt


def _valid_payload(**overrides):
    payload = {
        "action": "ENTER_LONG",
        "stop": 24950.0,
        "target": 25100.0,
        "confidence": 7,
        "setup": "pivot_support_hammer",
        "reasoning": "Hammer at pivot with bullish confirmation; tight stop, clear target.",
        "model_used": "claude-opus-4-8",
    }
    payload.update(overrides)
    return payload


def test_valid_decision_parses():
    decision = SLHuntingDecision.model_validate(_valid_payload())
    assert decision.action == "ENTER_LONG"
    assert decision.confidence == 7


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence=11))
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence=-1))


def test_stop_and_target_bounds_are_enforced():
    """SLH-002: hallucinated stop/target garbage must fail schema validation.

    A negative or absurd stop silently disables the mechanical underlying
    stop for the trade (only max-loss/square-off remain), so the record of
    the decision must never carry such values.
    """
    for bad in (-1.0, -1e9, float("nan"), float("inf"), 10_000_001.0):
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(stop=bad))
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(target=bad))
    # 0.0 stays valid -- the documented placeholder for EXIT/HOLD decisions.
    decision = SLHuntingDecision.model_validate(
        _valid_payload(action="HOLD", stop=0.0, target=0.0)
    )
    assert decision.stop == 0.0 and decision.target == 0.0


def test_entry_actions_require_positive_stop_and_target():
    """SLH-002 / Codex (PR #43): an ENTER decision with a 0 (or omitted) stop or
    target must fail validation, so a hallucinated entry can't be recorded as a
    real trade with no levels (which would defeat the order-tool guard and
    corrupt the decision journal). EXIT/HOLD keep their 0.0 placeholders."""
    for action in ("ENTER_LONG", "ENTER_SHORT"):
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(action=action, stop=0.0))
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(action=action, target=0.0))
        # Omitted stop/target default to 0.0 -> also rejected for entries.
        with pytest.raises(Exception):
            payload = _valid_payload(action=action)
            payload.pop("stop")
            payload.pop("target")
            SLHuntingDecision.model_validate(payload)
        # Both positive -> valid.
        ok = SLHuntingDecision.model_validate(_valid_payload(action=action, stop=24950.0, target=25100.0))
        assert ok.action == action
    for action in ("EXIT", "HOLD"):
        ok = SLHuntingDecision.model_validate(_valid_payload(action=action, stop=0.0, target=0.0))
        assert ok.action == action


def test_json_schema_omits_min_max_on_stop_and_target():
    """Same Claude-schema constraint as confidence: bounds live in validators,
    never as minimum/maximum keys in the described JSON schema."""
    schema = SLHuntingDecision.model_json_schema()
    for field_name in ("stop", "target"):
        props = schema["properties"][field_name]
        assert "minimum" not in props and "maximum" not in props


def test_strict_rejects_unknown_fields_and_coercion():
    # extra field forbidden
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(unexpected="x"))
    # strict mode: a string is not coerced to int for confidence
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence="7"))


def test_invalid_action_rejected():
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(action="BUY"))


def test_exit_leg_defaults_to_both_and_validates():
    """Per-leg exit selector: default BOTH, accepts the three literals, rejects others."""
    assert SLHuntingDecision.model_validate(_valid_payload()).exit_leg == "BOTH"
    for leg in ("NIFTY", "BNF", "BOTH"):
        assert SLHuntingDecision.model_validate(_valid_payload(exit_leg=leg)).exit_leg == leg
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(exit_leg="SENSEX"))


def test_json_schema_omits_min_max_on_confidence():
    """Regression guard: Claude rejects minimum/maximum on integer types."""
    schema = SLHuntingDecision.model_json_schema()
    conf = schema["properties"]["confidence"]
    assert conf["type"] == "integer"
    assert "minimum" not in conf
    assert "maximum" not in conf


def test_system_prompt_has_final_output_marker():
    prompt = build_system_prompt() + FINAL_OUTPUT_INSTRUCTION
    assert "FINAL OUTPUT FORMAT" in prompt
    # The method's core rules should be present in the agent's "brain".
    assert "pivot" in prompt.lower()
    assert "confirmation" in prompt.lower()


def test_system_prompt_has_per_leg_exit_knowledge():
    """v5: the mirror is tied for hard risk but per-leg for premise exits (exit_leg)."""
    prompt = build_system_prompt() + FINAL_OUTPUT_INSTRUCTION
    assert "exit_leg" in prompt
    assert "PREMISE-INVALIDATION is PER-LEG" in prompt
    assert "HARD RISK stays TIED" in prompt


def test_system_prompt_has_v2_markers():
    """v2: BankNIFTY cross-confirmation section + the dynamic-sizing note are present."""
    prompt = build_system_prompt()
    assert "CROSS-INDEX CONFIRMATION" in prompt
    assert "bank_nifty" in prompt and "cross_index" in prompt
    # The agent is told sizing is automatic at ~Rs.2500 risk (it does not pick lots).
    assert "2500" in prompt


def test_system_prompt_has_v3_gap_knowledge():
    """v3: the gap/retail-positioning knowledge from the video is present."""
    prompt = build_system_prompt()
    assert "READING RETAIL POSITIONING" in prompt
    low = prompt.lower()
    assert "gap-up" in low and "gap-down" in low
    # The momentum-context nuance (don't fade every big candle).
    assert "momentum" in low


def test_system_prompt_has_v3a_bnf_knowledge():
    """v3a: the BankNIFTY live-trading methodology section + merged lessons are present."""
    prompt = build_system_prompt()
    # The new advisory BankNIFTY-specific section and its distinctive markers.
    assert "BANK NIFTY — SPECIFIC BEHAVIOUR" in prompt
    assert "Sensex" in prompt              # triple-index (BNF + NIFTY + Sensex) read
    assert "MAJOR index" in prompt         # BankNIFTY as the major/base index
    low = prompt.lower()
    assert "time-decay" in low             # G5 theta discipline merged into RISK
    assert "closing point" in low          # G2 closing-price invalidation level
    # It must sit AFTER the existing cross-index section (advisory context that extends it),
    # and must NOT weaken the mandatory candle+confirmation rule.
    assert prompt.index("BANK NIFTY — SPECIFIC BEHAVIOUR") > prompt.index("CROSS-INDEX CONFIRMATION")
    assert "execute NIFTY ATM options ONLY" in prompt


def test_system_prompt_has_v3d_conditional_gap_knowledge():
    """v3d: prior-days conditional gap read, reachability, and gap-size asymmetry are present."""
    prompt = build_system_prompt()
    assert "READ THE GAP AGAINST THE PRIOR DAYS" in prompt
    assert "SL-REACHABILITY TEST" in prompt
    assert "GAP-SIZE ASYMMETRY" in prompt
    # The flat-open seller-hunt long lives inside the OPENING DRIVE section as variant B.
    assert "Variant B" in prompt
    # v3c's opening-drive exception must still be present and scoped.
    assert "OPENING DRIVE" in prompt


def test_system_prompt_has_v3e_participation_knowledge():
    """v3e: both-sides participation, huge-gap nuance, third-index lag, setup staleness."""
    prompt = build_system_prompt()
    assert "BOTH-SIDES PARTICIPATION" in prompt
    assert "HUGE gap" in prompt
    assert "THIRD-INDEX LAG" in prompt
    assert "SETUP STALENESS" in prompt


def test_system_prompt_has_v3f_transcript_match_knowledge():
    """v3f: July 4-8 transcript + agent-match lessons are present."""
    prompt = build_system_prompt()
    assert "BUYER-INVENTORY FADE" in prompt
    assert "TARGET-BOOKED" in prompt
    assert "GAP-DOWN CONTINUATION SHORT" in prompt
    assert "NO INSTANT FLIP" in prompt
    assert "MASKED BNF LAG" in prompt


def test_system_prompt_has_v3g_full_2026_sweep_knowledge():
    """v3g: the Jan-Jul 2026 transcript sweep's carry-risk refinements are present."""
    prompt = build_system_prompt()
    assert "EVENT / HOLIDAY PARTICIPATION" in prompt
    assert "CONSTRUCTED-BASE CONTINUATION" in prompt
    assert "PREVIOUS-CHART LINKAGE" in prompt
    assert "WEEKEND / HOLIDAY CARRY-RISK" in prompt


def test_system_prompt_has_v3h_remaining_transcript_knowledge():
    """v3h: remaining-video fallback transcript lessons are present."""
    prompt = build_system_prompt()
    assert "UNIQUE-TRADE FILTER" in prompt
    assert "PROFIT-HOLD" in prompt
    assert "TIMEFRAME FIT" in prompt
    assert "PLAN-OF-EXECUTION" in prompt
    assert "NO DAILY-INCOME PRESSURE" in prompt
    assert "POST-LOSS SPEED LIMIT" in prompt


def test_system_prompt_has_v3i_premium_rr_knowledge():
    """v3i: 10 Jul live session — premium non-confirmation exit + R:R-bait read."""
    prompt = build_system_prompt()
    assert "PREMIUM NON-CONFIRMATION" in prompt
    assert "R:R-BAIT AT ROUND-NUMBER REJECTIONS" in prompt
    # The actionable exit rule: book the average target when premiums lag the spot move.
    assert "AVERAGE target" in prompt


def test_system_prompt_has_v3j_averaging_trap_knowledge():
    """v3j: 13-14 Jul gap-down sessions, cross-checked against the agent's own journal.

    Three lessons, each tied to a real 14 Jul decision:
    - AVERAGING TRAP fixes the trade-1 premise (the agent read "starved sellers" and
      went long where IH read yesterday's recovery-buyers as the trapped crowd).
    - MOVE-EXHAUSTION fixes trade 3 (re-shorting the same spent move into an expiry
      range, stopped out in 5 seconds) — the same-direction blind spot NO INSTANT FLIP
      does not cover.
    - The cross-index "stale verdict" escape hatch is scoped to the opening hour, since
      trade 3 used it at 10:04 to override an opposing verdict at confidence 6.
    """
    prompt = build_system_prompt()
    assert "AVERAGING TRAP" in prompt
    assert "MOVE-EXHAUSTION" in prompt
    # The entry-timing half of the averaging trap: never enter at the gap extreme.
    assert "do NOT enter at the gap extreme" in prompt
    # Expiry is fuel for an existing premise, never a premise of its own.
    assert "EXPIRY IS CONTEXT, NOT A PREMISE" in prompt
    # The "stale" escape hatch must be explicitly bounded to the opening hour.
    assert "SCOPE OF THIS \"STALE\" ESCAPE HATCH" in prompt


def test_system_prompt_has_v3k_flat_open_gate_knowledge():
    """v3k: 15 Jul sessions — the flat-open hunt needs a crowd that really participated.

    After a WEAK-momentum down day, a flat open puts nobody in pain (and leaves the
    closing-point support in the recovery's path) — the plan flips WITH the prior
    direction, while a gap in EITHER direction re-arms the seller-hunt. Scopes the
    blanket "FLAT or GAP-DOWN -> look UP" default.
    """
    prompt = build_system_prompt()
    assert "FLAT-OPEN PARTICIPATION GATE" in prompt
    # The asymmetry in one line: either-direction gap hunts, flat goes with-trend.
    assert "flat" in prompt and "go with the selling" in prompt
    # It must scope, not delete, the textbook flat/gap-down hunt above it.
    assert "PRIME TRAP zone" in prompt


def test_system_prompt_has_v3l_closing_point_and_shared_gap_knowledge():
    """v3l: 16 Jul split-gap session, cross-checked against journal rows 21-22.

    - CLOSING-POINT HOLD TEST answers whether an overnight crowd exists at all: a prior
      rejection that never BROKE the closing point means that crowd booked and left, so
      there is no inventory to hunt -> follow the move instead.
    - The OPENING DRIVE gap-up branch now requires the gap to be SHARED: the agent fired
      it on NIFTY's gap while BankNIFTY opened flat at its own closing point (IH read the
      same open as a short) and the basket lost Rs.1,333.
    """
    prompt = build_system_prompt()
    assert "CLOSING-POINT HOLD TEST" in prompt
    assert "SHARED-GAP REQUIREMENT" in prompt
    # The decisive tell: a flat major index beside a gapped NIFTY kills the long branch.
    assert "flat major index" in prompt.lower()
    # The hold test must state both arms (seated-and-huntable vs booked-and-gone).
    assert "BROKE it and held beyond" in prompt
    # The leader-fails-to-lead exit keeps its scope so it can't collide with RISK's
    # "SLOW-but-CONTINUOUS is the sustainable kind" rule.
    assert "SLOW-but-CONTINUOUS" in prompt


def test_system_prompt_has_v3m_gift_gap_and_loss_flip_knowledge():
    """v3m: 17 Jul flat-open loss day (IH's first loss in the series).

    - GIFT-GAP AFTER A NOBODY'S-CROWD DAY: after a small-momentum day with the
      closing point uncrossed, a gap in EITHER direction traps the side it appears
      to reward (fade it); flat means there is nobody to hunt.
    - NO INSTANT FLIP now also bans the mid-loss panic flip (booking a small loss to
      instantly reverse into the breakout), tying into POST-LOSS SPEED LIMIT.
    """
    prompt = build_system_prompt()
    assert "GIFT-GAP AFTER A NOBODY'S-CROWD DAY" in prompt
    # Each gap direction traps the side it appears to reward on a thin day.
    assert "traps its own recipient" in prompt
    # The losing-side flip ban lives inside the existing NO INSTANT FLIP bullet
    # (assert wrap-independent fragments, not exact line breaks).
    assert "LOSING side" in prompt and "whipsaw" in prompt
    assert "POST-LOSS SPEED LIMIT" in prompt


def test_system_prompt_has_v3n_closed_chart_knowledge():
    """v3n: 19 Jul closed-chart lecture (IH's week review + self-diagnosed loss).

    - RECRUITMENT HISTORY: two near-identical charts demand OPPOSITE plans, because a
      first reversal-type move recruits nobody while the SECOND consecutive
      same-direction day seats the crowd.
    - ONE BREAKDOWN, NOT TWO: the rule whose absence cost IH the 17 Jul trade — after
      one level break the next rarely breaks; sellers are likely seated and buyers are
      definitely evicted.
    - The CLOSING-POINT HOLD TEST's "held beyond" arm now requires real MOMENTUM: a
      break that idles for hours seats nobody (a correction to v3l).
    """
    prompt = build_system_prompt()
    assert "RECRUITMENT HISTORY, NOT CHART SHAPE" in prompt
    assert "ONE BREAKDOWN, NOT TWO" in prompt
    # The recruitment law, wrap-independent.
    assert "SECOND" in prompt and "consecutive same-direction day" in prompt
    # The asymmetric fallback: a breakdown always evicts the buyers.
    assert "buyers are never" in prompt
    # The v3l correction: break-and-held only seats a crowd if momentum followed.
    assert "produced actual MOMENTUM" in prompt


def test_system_prompt_has_v3o_flush_day_and_solo_leader_knowledge():
    """v3o: 20-21 Jul sessions (IH won the news gap-down, lost the flat-open long).

    - BOTH-WAYS FLUSH DAY: the second way a day ends with nobody seated — after a
      violent both-ways session there is nothing to fade; follow the opening type,
      and treat the flat-open first push as recruitment bait (it caught IH on 21 Jul).
    - SOLO-LEADER VETO: BankNIFTY-moving-first is void as an entry tell when the other
      two indices are capped below their closing points (IH: "I trusted BankNIFTY too
      much").
    """
    prompt = build_system_prompt()
    assert "BOTH-WAYS FLUSH DAY" in prompt
    # The plan collapse and the flat-branch bait, wrap-independent.
    assert "as the opening, so the plan" in prompt
    assert "recruitment bait" in prompt
    # The disambiguation question against GIFT-GAP.
    assert "WHY nobody is seated" in prompt
    assert "SOLO-LEADER VETO" in prompt
    # The veto's release condition.
    assert "reclaim its closing point" in prompt
