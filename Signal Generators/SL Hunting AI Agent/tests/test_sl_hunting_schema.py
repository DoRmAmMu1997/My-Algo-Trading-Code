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
