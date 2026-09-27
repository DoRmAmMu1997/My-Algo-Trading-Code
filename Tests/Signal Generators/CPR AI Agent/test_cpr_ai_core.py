"""Exercise each deterministic host-policy gate with one-fact mutations.

The baseline context is deliberately valid for a bearish Trend-Day Rider
entry (the runtime suite covers the bullish mirror). Parameterized tests change
one frozen fact at a time, making a failure code attributable to a specific
gate rather than to model prose or a second accidental invalid value. No
broker, SDK, market feed, or order path is present in this module.
"""

from __future__ import annotations

import pytest
from cpr_ai_agent import CPRHostPolicy
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_schema import CPRAgentDecision
from pydantic import ValidationError


def _context() -> dict[str, dict[str, object]]:
    """Return a minimal short-valid snapshot: close 200, VWAP stop 212."""

    return {
        "session_levels": {"current_close": 200.0},
        "momentum_vwap": {},
        "market_structure": {
            "trend_day_candidate": {"eligible": True, "direction": "SHORT", "entry": 200.0, "stop": 212.0},
        },
        "position_state": {"is_flat": True, "entries_today": 0},
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


SHORT = ("ENTER_SHORT", "TRENDING", "TREND_DAY_CONTINUATION")


def test_bearish_candidate_is_accepted_with_host_geometry() -> None:
    outcome = CPRHostPolicy().validate(_context(), _proposal(*SHORT))

    assert outcome.accepted and outcome.validation_code == "accepted_entry"
    assert (outcome.entry_price, outcome.stop_price, outcome.risk_points) == (200.0, 212.0, 12.0)
    assert outcome.accepted_regime == "TRENDING"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda c: c["market_structure"]["trend_day_candidate"].update({"eligible": False}), "no_trend_day_candidate"),
        (lambda c: c["market_structure"]["trend_day_candidate"].update({"eligible": "true"}), "no_trend_day_candidate"),
        (lambda c: c["market_structure"]["trend_day_candidate"].update({"direction": "LONG"}),
         "candidate_direction_mismatch"),
        (lambda c: c["market_structure"]["trend_day_candidate"].update({"stop": 199.0}), "invalid_stop_geometry"),
        (lambda c: c["market_structure"]["trend_day_candidate"].update({"entry": float("nan")}),
         "missing_candidate_geometry"),
        (lambda c: c["session_levels"].update({"current_close": 199.5}), "candidate_close_mismatch"),
        (lambda c: c["position_state"].update({"entries_today": 1}), "session_entry_used"),
        (lambda c: c["market_structure"].pop("trend_day_candidate"), "invalid_frozen_context"),
        (lambda c: c["position_state"].update({"is_flat": None}), "invalid_position_state"),
    ],
)
def test_each_entry_gate_rejects_independently(change, code) -> None:
    context = _context()
    change(context)
    outcome = CPRHostPolicy().validate(context, _proposal(*SHORT))

    assert outcome.accepted is False
    assert outcome.validation_code == code
    assert outcome.entry_price is None and outcome.stop_price is None and outcome.risk_points is None


def test_policy_refuses_a_non_trending_entry_even_if_the_schema_is_bypassed() -> None:
    # ``model_construct`` skips validation, standing in for a future schema
    # regression. The host must still refuse on its own.
    rogue = CPRAgentDecision.model_construct(
        action="ENTER_SHORT", regime="SIDEWAYS", setup="TREND_DAY_CONTINUATION", confidence=5,
        reasoning="x", model_used="gpt-5.6-terra", prompt_version=CPR_AI_PROMPT_VERSION,
    )
    wrong_setup = CPRAgentDecision.model_construct(
        action="ENTER_SHORT", regime="TRENDING", setup="NONE", confidence=5,
        reasoning="x", model_used="gpt-5.6-terra", prompt_version=CPR_AI_PROMPT_VERSION,
    )

    assert CPRHostPolicy().validate(_context(), rogue).validation_code == "trending_regime_rejected"
    assert CPRHostPolicy().validate(_context(), wrong_setup).validation_code == "entry_setup_rejected"


@pytest.mark.parametrize(
    ("action", "regime", "setup"),
    [
        ("SCALE_IN", "TRENDING", "NONE"),
        ("ENTER_LONG", "TRENDING", "TRENDING_VWAP_CONTINUATION"),
        ("ENTER_LONG", "SIDEWAYS", "SIDEWAYS_SRSI"),
        ("ENTER_LONG", "SIDEWAYS", "TREND_DAY_CONTINUATION"),
        ("ENTER_SHORT", "TRENDING", "NONE"),
        ("EXIT", "TRENDING", "NONE"),
        ("HOLD", "TRENDING", "TREND_DAY_CONTINUATION"),
        ("ENTER_LONG", "UNDECIDED", "TREND_DAY_CONTINUATION"),
    ],
)
def test_schema_rejects_retired_and_contradictory_combinations(action, regime, setup) -> None:
    with pytest.raises(ValidationError):
        _proposal(action, regime, setup)


def test_open_position_may_only_hold_or_exit() -> None:
    context = _context()
    context["position_state"] = {"is_flat": False, "direction": "SHORT", "entries_today": 1}

    assert CPRHostPolicy().validate(context, _proposal(*SHORT)).validation_code == "open_action_rejected"
    exit_outcome = CPRHostPolicy().validate(context, _proposal("EXIT", "SIDEWAYS", "PREMISE_EXIT"))
    assert exit_outcome.accepted and exit_outcome.validation_code == "accepted_exit"
    hold = CPRHostPolicy().validate(context, _proposal("HOLD", "TRENDING", "NONE"))
    assert hold.accepted and hold.validation_code == "accepted_hold" and hold.accepted_regime == "TRENDING"
