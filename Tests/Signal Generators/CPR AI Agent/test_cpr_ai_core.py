"""Exercise each deterministic host-policy gate with one-fact mutations.

The baseline context is deliberately valid for a long trend entry. Parameterized
tests change one frozen fact at a time, making a failure code attributable to a
specific gate rather than to model prose or a second accidental invalid value.
No broker, SDK, market feed, or order path is present in this module.
"""

from __future__ import annotations

import pytest
from cpr_ai_agent import CPRHostPolicy
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_schema import CPRAgentDecision


def _context() -> dict[str, dict[str, object]]:
    """Return a minimal long-valid snapshot with literal deterministic facts.

    Entry 100, candle stop 95, buffered R1 108, and buffered R2 118 make the
    accepted geometry easy to reason about. Every other field is the smallest
    true value needed for the policy path under test.
    """

    return {
        "session_levels": {
            "current_close": 100.0,
            "levels": {"r1": 110.0, "r2": 120.0, "s1": 90.0, "s2": 80.0},
            "next_levels": {"upside": {"name": "r1", "price": 110.0}, "downside": {"name": "s1", "price": 90.0}},
        },
        "momentum_vwap": {
            "rsi14": 55.0,
            "stochastic_rsi": {"cross_up_in_oversold": True, "cross_down_in_overbought": True},
            "vwap": {
                "sequence_evidence": {
                    "all_recent_above": True,
                    "all_recent_below": False,
                    "reclaimed": True,
                    "lost": False,
                },
                "entry_candle": {"body_fraction_above": 0.5, "body_fraction_below": 0.5},
            },
            "ema": {"order": "EMA5_ABOVE_EMA20", "ema5_slope": 1.0, "ema20_slope": 0.5},
            "candle": {"low": 95.0, "high": 105.0},
        },
        "market_structure": {
            "swings": {"lows": [{"price": 94.0}], "highs": [{"price": 106.0}]},
            "r1_scale_in_candidate": {"eligible": True, "direction": "LONG"},
        },
        "position_state": {
            "is_flat": True,
            "premise": "TRENDING_VWAP_CONTINUATION",
            "scale_in_eligible": True,
            "scale_in_count": 0,
        },
    }


def _proposal(action: str, regime: str, setup: str) -> CPRAgentDecision:
    """Build a strict advisory proposal containing no price, size, or order data."""

    return CPRAgentDecision(
        action=action,
        regime=regime,
        setup=setup,
        confidence=7,
        reasoning="test",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            lambda c: c["momentum_vwap"]["vwap"]["sequence_evidence"].update({"all_recent_above": False}),
            "vwap_sequence_rejected",
        ),
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
    ("action", "entry", "boundary_name", "boundary", "expected_code"),
    [
        ("ENTER_LONG", 121.0, "r2", 120.0, "continuation_outside_r2_s2"),
        ("ENTER_SHORT", 79.0, "s2", 80.0, "continuation_outside_r2_s2"),
    ],
)
def test_trend_continuation_rejects_closes_beyond_the_final_cpr_boundary(
    action,
    entry,
    boundary_name,
    boundary,
    expected_code,
):
    """A continuation cannot chase price above R2 or sell below S2.

    The literal entry and boundary pairs independently encode the two strict
    inequalities.  This test expects a dedicated policy rejection rather than
    relying on the later reward/target geometry to reject the trade by accident.
    """

    context = _context()
    context["session_levels"]["current_close"] = entry
    context["session_levels"]["levels"][boundary_name] = boundary
    if action == "ENTER_SHORT":
        # Keep every other short continuation fact valid so the new S2 boundary
        # is the first and only host-policy reason for rejecting this proposal.
        context["momentum_vwap"].update(
            {
                "rsi14": 60.0,
                "ema": {"order": "EMA5_BELOW_EMA20", "ema5_slope": -1.0, "ema20_slope": -0.5},
                "candle": {"low": 75.0, "high": 84.0},
            }
        )
        context["momentum_vwap"]["vwap"] = {
            "sequence_evidence": {"all_recent_below": True},
            "entry_candle": {"body_fraction_below": 0.5},
        }

    outcome = CPRHostPolicy().validate(
        context,
        _proposal(action, "TRENDING", "TRENDING_VWAP_CONTINUATION"),
    )

    assert outcome.validation_code == expected_code


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
    context["momentum_vwap"].update(
        {
            "rsi14": 60.0,
            "ema": {"order": "EMA5_BELOW_EMA20", "ema5_slope": -1.0, "ema20_slope": -0.5},
            "candle": {"low": 95.0, "high": 105.0},
        }
    )
    context["momentum_vwap"]["vwap"] = {
        "sequence_evidence": {
            "all_recent_above": False,
            "all_recent_below": setup.endswith("CONTINUATION"),
            "reclaimed": False,
            "lost": setup.endswith("REVERSAL"),
        },
        "entry_candle": {"body_fraction_above": 0.0, "body_fraction_below": 0.5},
    }

    outcome = CPRHostPolicy().validate(context, _proposal("ENTER_SHORT", "TRENDING", setup))

    assert outcome.validation_code == code
    assert outcome.stop_price == 105.0
    assert outcome.milestone_price == 92.0
    assert outcome.final_target_price == 82.0


def test_geometry_rejects_sub_one_r_and_wrong_side_final_target():
    """Buffered levels must offer at least one R and a directionally valid target."""

    sub_one_r = _context()
    sub_one_r["session_levels"]["levels"]["r1"] = 104.0
    sub_one_r["session_levels"]["next_levels"]["upside"] = {"name": "r1", "price": 104.0}
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

    assert (
        CPRHostPolicy().validate(missing_swing, _proposal("ENTER_LONG", "SIDEWAYS", "SIDEWAYS_SRSI")).validation_code
        == "missing_swing_stop"
    )
    assert (
        CPRHostPolicy().validate(open_short, _proposal("SCALE_IN", "TRENDING", "R1_SCALE_IN")).validation_code
        == "scale_in_rejected"
    )


@pytest.mark.parametrize(
    ("direction", "change", "code"),
    [
        (
            "ENTER_LONG",
            lambda c: c["momentum_vwap"]["stochastic_rsi"].update({"cross_up_in_oversold": False}),
            "srsi_cross_rejected",
        ),
        (
            "ENTER_SHORT",
            lambda c: c["momentum_vwap"]["stochastic_rsi"].update({"cross_down_in_overbought": False}),
            "srsi_cross_rejected",
        ),
        ("ENTER_SHORT", lambda c: c["momentum_vwap"].update({"rsi14": 65.0}), "rsi_rejected"),
        ("ENTER_SHORT", lambda c: c["momentum_vwap"]["ema"].update({"ema20_slope": 0.0}), "ema_rejected"),
    ],
)
def test_sideways_cross_and_directional_short_hard_gates_reject(direction, change, code):
    """Neither an incorrect SRSI zone nor one broken short gate may be inferred away."""

    context = _context()
    if direction == "ENTER_SHORT":
        context["momentum_vwap"].update(
            {
                "rsi14": 60.0,
                "ema": {"order": "EMA5_BELOW_EMA20", "ema5_slope": -1.0, "ema20_slope": -1.0},
                "candle": {"high": 105.0},
            }
        )
        context["momentum_vwap"]["vwap"] = {
            "sequence_evidence": {"all_recent_below": True},
            "entry_candle": {"body_fraction_below": 0.5},
        }
    change(context)
    setup = "SIDEWAYS_SRSI" if code == "srsi_cross_rejected" else "TRENDING_VWAP_CONTINUATION"
    regime = "SIDEWAYS" if setup == "SIDEWAYS_SRSI" else "TRENDING"

    assert CPRHostPolicy().validate(context, _proposal(direction, regime, setup)).validation_code == code


def test_valid_long_reversal_and_sideways_origin_scale_in_are_distinguished():
    """A reclaim is a valid long reversal, but it cannot rewrite a sideways premise."""

    reversal = _context()
    reversal["momentum_vwap"]["vwap"]["sequence_evidence"] = {"reclaimed": True}
    valid = CPRHostPolicy().validate(reversal, _proposal("ENTER_LONG", "TRENDING", "TRENDING_VWAP_REVERSAL"))
    scale = _context()
    scale["position_state"] = {
        "is_flat": False,
        "direction": "LONG",
        "premise": "SIDEWAYS_SRSI",
        "scale_in_eligible": True,
        "scale_in_count": 0,
    }
    rejected = CPRHostPolicy().validate(scale, _proposal("SCALE_IN", "TRENDING", "R1_SCALE_IN"))

    assert valid.validation_code == "accepted_entry"
    assert rejected.validation_code == "scale_in_rejected"
