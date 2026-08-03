"""Cover detailed deterministic policy rejections for the CPR Codex boundary."""

from __future__ import annotations

import pytest
from cpr_ai_agent import CPRHostPolicy
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_schema import CPRAgentDecision


def _context() -> dict[str, dict[str, object]]:
    """Return a long-valid context with literal prices and indicator gates."""

    return {
        "session_levels": {"current_close": 100.0, "levels": {"r1": 110.0, "r2": 120.0, "s1": 90.0, "s2": 80.0}},
        "momentum_vwap": {
            "rsi14": 55.0,
            "stochastic_rsi": {"cross_up_in_oversold": True, "cross_down_in_overbought": True},
            "vwap": {"sequence_evidence": {"all_recent_above": True, "all_recent_below": False, "reclaimed": True, "lost": False}, "entry_candle": {"body_fraction_above": 0.5, "body_fraction_below": 0.5}},
            "ema": {"order": "EMA5_ABOVE_EMA20", "ema5_slope": 1.0, "ema20_slope": 0.5},
            "candle": {"low": 95.0, "high": 105.0},
        },
        "market_structure": {"swings": {"lows": [{"price": 94.0}], "highs": [{"price": 106.0}]}, "r1_scale_in_candidate": {"eligible": True, "direction": "LONG"}},
        "position_state": {"is_flat": True, "scale_in_count": 0},
    }


def _proposal(action: str, regime: str, setup: str) -> CPRAgentDecision:
    """Return a model classification that contains no execution geometry."""

    return CPRAgentDecision(action=action, regime=regime, setup=setup, confidence=7, reasoning="test", model_used="gpt-5.6-terra", prompt_version=CPR_AI_PROMPT_VERSION)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda c: c["momentum_vwap"]["vwap"]["sequence_evidence"].update({"all_recent_above": False}), "vwap_sequence_rejected"),
        (lambda c: c["momentum_vwap"].update({"rsi14": 45.0}), "rsi_rejected"),
        (lambda c: c["momentum_vwap"]["ema"].update({"ema5_slope": 0.0}), "ema_rejected"),
        (lambda c: c["momentum_vwap"]["candle"].update({"low": 69.0}), "risk_wider_than_30"),
    ],
)
def test_long_continuation_rejects_each_independent_hard_gate(change, code):
    """Each directional safety gate catches a realistic single-fact mutation."""

    context = _context()
    change(context)

    outcome = CPRHostPolicy().validate(context, _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION"))

    assert outcome.validation_code == code


@pytest.mark.parametrize(
    ("setup", "sequence", "code"),
    [
        ("TRENDING_VWAP_CONTINUATION", {"all_recent_above": True}, "accepted_entry"),
        ("TRENDING_VWAP_REVERSAL", {"reclaimed": True}, "accepted_entry"),
    ],
)
def test_short_continuation_and_reversal_use_mirrored_vwap_ema_and_cpr_geometry(setup, sequence, code):
    """Shorts use below/lost VWAP facts, falling EMAs, S1+2 and S2+2 prices."""

    context = _context()
    context["momentum_vwap"].update({"rsi14": 60.0, "ema": {"order": "EMA5_BELOW_EMA20", "ema5_slope": -1.0, "ema20_slope": -0.5}, "candle": {"low": 95.0, "high": 105.0}})
    context["momentum_vwap"]["vwap"] = {"sequence_evidence": {"all_recent_above": False, "all_recent_below": setup.endswith("CONTINUATION"), "reclaimed": False, "lost": setup.endswith("REVERSAL")}, "entry_candle": {"body_fraction_above": 0.0, "body_fraction_below": 0.5}}

    outcome = CPRHostPolicy().validate(context, _proposal("ENTER_SHORT", "TRENDING", setup))

    assert outcome.validation_code == code
    assert outcome.stop_price == 105.0
    assert outcome.milestone_price == 92.0
    assert outcome.final_target_price == 82.0


def test_geometry_rejects_sub_one_r_and_wrong_side_final_target():
    """Buffered levels must offer at least one R and a directionally valid target."""

    sub_one_r = _context()
    sub_one_r["session_levels"]["levels"]["r1"] = 104.0
    wrong_target = _context()
    wrong_target["session_levels"]["levels"]["r2"] = 101.0

    proposal = _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION")
    assert CPRHostPolicy().validate(sub_one_r, proposal).validation_code == "sub_one_r_milestone"
    assert CPRHostPolicy().validate(wrong_target, proposal).validation_code == "invalid_target_geometry"


def test_sideways_missing_swing_and_open_short_scale_in_are_rejected():
    """The host cannot invent a swing stop or permit the forbidden short scale-in."""

    missing_swing = _context()
    missing_swing["market_structure"]["swings"]["lows"] = []
    open_short = _context()
    open_short["position_state"] = {"is_flat": False, "direction": "SHORT", "scale_in_count": 0}

    assert CPRHostPolicy().validate(missing_swing, _proposal("ENTER_LONG", "SIDEWAYS", "SIDEWAYS_SRSI")).validation_code == "missing_swing_stop"
    assert CPRHostPolicy().validate(open_short, _proposal("SCALE_IN", "TRENDING", "R1_SCALE_IN")).validation_code == "scale_in_rejected"
