"""Specify the independent deterministic context supplied to the CPR agent.

The fixtures use deliberately hand-derived one-minute prices and explicit IST
clocks. They never call an older CPR Strategy helper, which proves this package
owns completed-bar, CPR-level, ATR, VWAP, structure, and snapshot calculations.
The trend-day fixture checks that the frozen candidate is exactly what the
shared gate (and so the backtest) computes, while duplicate/missing/forming
minute cases protect live completed-bar cadence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from cpr_ai_agent import CPRHostPolicy
from cpr_ai_context import (
    build_completed_five_minute_bars,
    build_cpr_context,
)
from cpr_ai_mcp_server import build_mcp_server, load_snapshot_payload
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION, build_system_prompt
from cpr_ai_schema import CPRAgentDecision, validate_position_state
from cpr_ai_signals import freeze_cpr_context
from cpr_ai_tools import EXPECTED_TOOL_NAMES, FrozenCPRContextRegistry
from cpr_ai_trend_day import evaluate_trend_day_candidate
from pydantic import ValidationError


def _minute_rows(start: datetime, closes: list[float], *, volume: float | None = None) -> list[dict[str, object]]:
    """Create simple one-minute candles with predictable OHLC and timestamps.

    Each close gets a fixed half-point open and one-point high/low envelope, so
    resampled values can be checked without reproducing production calculations
    inside the test helper. Individual tests overwrite only the fact they need.
    """

    rows: list[dict[str, object]] = []
    for offset, close in enumerate(closes):
        rows.append(
            {
                "timestamp": start + timedelta(minutes=offset),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": volume,
            }
        )
    return rows


def _two_session_five_minute_frame(
    closes: list[float],
    *,
    current_session_bars: int = 8,
) -> pd.DataFrame:
    """Expand continuous five-minute closes across two trading sessions.

    The final eight bars represent 09:15 through 09:50 on the current day.
    Everything before them belongs to the previous session and should warm
    continuous indicators without entering session-reset calculations such as
    VWAP. Repeating each intended close five times keeps resampling literal.
    """

    if len(closes) <= current_session_bars:
        raise ValueError("The fixture needs prior-session and current-session bars.")
    split_at = len(closes) - current_session_bars
    previous_minutes = _minute_rows(
        datetime(2026, 8, 1, 9, 15),
        [close for close in closes[:split_at] for _ in range(5)],
    )
    current_minutes = _minute_rows(
        datetime(2026, 8, 2, 9, 15),
        [close for close in closes[split_at:] for _ in range(5)],
    )
    return pd.DataFrame(previous_minutes + current_minutes)


def _two_session_frame() -> pd.DataFrame:
    """Return hand-shaped prior-day levels plus a fully warmed current session.

    The previous session fixes H/L/C at 110/90/105 for literal CPR assertions.
    The current monotonic sequence provides enough completed bars for RSI,
    StochRSI, EMA, VWAP, opening ranges, and swing calculations.
    """

    previous = _minute_rows(datetime(2026, 8, 1, 9, 15), [100.0] * 30)
    # Hand-set the previous-day extremes and closing price used by CPR math.
    previous[0]["low"] = 90.0
    previous[1]["high"] = 110.0
    previous[-1]["close"] = 105.0
    current = _minute_rows(datetime(2026, 8, 2, 9, 15), [100.0 + index * 0.4 for index in range(180)])
    return pd.DataFrame(previous + current)


def test_completed_five_minute_bars_drop_a_partial_bucket_and_preserve_ohlc():
    """A partial 09:20 bucket must never reach a completed-bar decision."""

    frame = pd.DataFrame(_minute_rows(datetime(2026, 8, 2, 9, 15), [100, 101, 102, 103, 104, 105, 106]))

    bars = build_completed_five_minute_bars(frame)

    assert len(bars) == 1
    assert bars.iloc[0].to_dict() == {
        "timestamp": pd.Timestamp("2026-08-02 09:15:00"),
        "open": 99.5,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 0.0,
    }


def test_current_start_stamped_minute_cannot_complete_its_five_minute_bucket():
    """A changing 09:19 websocket candle is forming until the clock reaches 09:20."""

    frame = pd.DataFrame(
        _minute_rows(datetime(2026, 8, 3, 9, 15), [100.0, 101.0, 102.0, 103.0, 104.0])
    )
    before_close = datetime(2026, 8, 3, 9, 19, 45, tzinfo=ZoneInfo("Asia/Kolkata"))

    assert build_completed_five_minute_bars(frame, as_of=before_close).empty
    frame.loc[4, ["high", "close"]] = [110.0, 109.0]
    assert build_completed_five_minute_bars(frame, as_of=before_close).empty

    at_close = datetime(2026, 8, 3, 9, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    completed = build_completed_five_minute_bars(frame, as_of=at_close)
    assert completed["timestamp"].tolist() == [pd.Timestamp("2026-08-03 09:15:00")]
    assert completed.iloc[0]["close"] == 109.0


@pytest.mark.parametrize(
    "minute_offsets",
    [
        [0, 1, 2, 3],
        [0, 0, 2, 3, 4],
        [0, 1, 2, 3, 4, 4],
    ],
)
def test_completed_bucket_requires_each_exact_minute_once(minute_offsets):
    """A missing or duplicate minute must not masquerade as a completed bucket."""

    rows = _minute_rows(datetime(2026, 8, 3, 9, 15), [100.0] * len(minute_offsets))
    for row, offset in zip(rows, minute_offsets, strict=True):
        row["timestamp"] = datetime(2026, 8, 3, 9, 15) + timedelta(minutes=offset)

    completed = build_completed_five_minute_bars(
        pd.DataFrame(rows),
        as_of=datetime(2026, 8, 3, 9, 20, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert completed.empty


def test_context_rejects_a_newest_session_without_a_completed_five_minute_bar():
    """One to four newest-session minutes must not reuse yesterday's context."""

    newest_partial_session = _minute_rows(datetime(2026, 8, 3, 9, 15), [200.0, 201.0, 202.0, 203.0])

    with pytest.raises(ValueError, match="latest input session"):
        build_cpr_context(pd.concat([_two_session_frame(), pd.DataFrame(newest_partial_session)], ignore_index=True))


def test_context_and_freezer_share_one_explicit_completed_bar_cutoff():
    """The context and immutable registry must exclude the same current forming minute."""

    previous = _minute_rows(datetime(2026, 8, 2, 9, 15), [100.0] * 30)
    current = _minute_rows(datetime(2026, 8, 3, 9, 15), [101.0] * 9 + [109.0])
    frame = pd.DataFrame(previous + current)
    before_close = datetime(2026, 8, 3, 9, 24, 50, tzinfo=ZoneInfo("Asia/Kolkata"))

    with pytest.raises(ValueError, match="two complete"):
        build_cpr_context(frame, as_of=before_close)

    at_close = datetime(2026, 8, 3, 9, 25, tzinfo=ZoneInfo("Asia/Kolkata"))
    context = build_cpr_context(frame, as_of=at_close)
    frozen = freeze_cpr_context(frame, as_of=at_close).snapshot_payload()
    assert context["session_levels"]["current_close"] == 109.0
    assert frozen == context


def test_context_uses_hand_derived_previous_day_levels_and_opening_facts():
    """A wrong CPR formula or session boundary must change these literal facts."""

    context = build_cpr_context(_two_session_frame())
    levels = context["session_levels"]

    # Previous H/L/C are 110/90/105, so P=101.666..., BC=100, TC=103.333....
    assert levels["previous_day"] == {"high": 110.0, "low": 90.0, "close": 105.0}
    assert levels["levels"]["pivot"] == pytest.approx(101.6666666667)
    assert levels["levels"]["bc"] == 100.0
    assert levels["levels"]["tc"] == pytest.approx(103.3333333333)
    assert levels["levels"]["cpr_lower"] == 100.0
    assert levels["levels"]["cpr_upper"] == pytest.approx(103.3333333333)
    assert levels["levels"]["r1"] == pytest.approx(113.3333333333)
    assert levels["levels"]["s1"] == pytest.approx(93.3333333333)
    assert levels["opening"]["first_15_minutes"]["complete"] is True
    assert levels["opening"]["first_30_minutes"]["complete"] is True
    # Session open 99.5 is below the prior close 105: a 5.5-point gap down.
    assert levels["gap"] == {"session_open": 99.5, "points": -5.5, "direction": "DOWN"}
    # One prior session is fewer than the three the ATR needs: no ATR, no candidate.
    assert levels["atr"] == {"value": None, "sessions_used": 1, "prior_session_ranges": [20.0]}
    assert context["market_structure"]["trend_day_candidate"]["eligible"] is False


def test_context_carries_only_the_prior_host_accepted_regime_into_session_levels():
    """The next turn sees host memory without letting context infer a regime."""

    context = build_cpr_context(
        _two_session_frame(),
        prior_accepted_regime="TRENDING",
    )

    assert context["session_levels"]["prior_accepted_regime"] == "TRENDING"
    with pytest.raises(ValueError, match="prior_accepted_regime"):
        build_cpr_context(_two_session_frame(), prior_accepted_regime="BREAKOUT")


def test_early_session_vwap_excludes_prior_session_indicator_history():
    """Previous-day warm-up prices cannot leak into current-session VWAP."""

    previous_closes = [900.0 + index for index in range(35)]
    current_closes = [100.0 + index for index in range(8)]
    context = build_cpr_context(_two_session_five_minute_frame(previous_closes + current_closes))
    vwap = context["momentum_vwap"]["vwap"]

    # The synthetic OHLC envelope makes each completed bar's typical price
    # equal its intended close, so this is an independent session-only result.
    assert vwap["value"] == pytest.approx(sum(current_closes) / len(current_closes))
    assert vwap["method"] == "equal_weight_typical_price"
    assert vwap["recent_relations"] == ["ABOVE"] * 6
    assert vwap["fraction_of_bars_above"] == pytest.approx(7 / 8)


def test_prior_indicator_history_does_not_leak_into_session_structure_facts():
    """Only indicator math may consume prior bars; session facts remain today's."""

    previous_closes = [900.0 + index for index in range(35)]
    current_closes = [100.0 + index for index in range(8)]
    context = build_cpr_context(_two_session_five_minute_frame(previous_closes + current_closes))

    opening = context["session_levels"]["opening"]["opening_corridor"]
    recent = context["momentum_vwap"]["recent_candles"]
    structure = context["market_structure"]
    assert opening == {
        "complete": True,
        "minutes": 5,
        "open": 99.5,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "range": 2.0,
    }
    assert all(str(candle["timestamp"]).startswith("2026-08-02") for candle in recent)
    assert structure["swings"] == {"highs": [], "lows": []}
    assert structure["extremes"] == {"bars_since_session_high": 0, "bars_since_session_low": 7}
    # 09:50 is outside the 11:00-13:30 entry window.
    assert structure["trend_day_candidate"]["reason"] == "outside_window"


def test_market_structure_reports_confirmed_swings_and_objective_hh_hl_comparisons():
    """Two candles on each side are required before a swing can be advertised."""

    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 10
    frame = pd.DataFrame(
        _minute_rows(datetime(2026, 8, 1, 9, 15), [100.0] * 30) + _minute_rows(datetime(2026, 8, 2, 9, 15), closes)
    )
    # Shape two five-minute high/low swings: values below are placed inside exact buckets.
    for row_index, high, low in ((35, 120.0, 99.0), (45, 110.0, 98.0), (55, 125.0, 101.0), (65, 115.0, 100.0)):
        frame.loc[row_index, "high"] = high
        frame.loc[row_index, "low"] = low

    structure = build_cpr_context(frame)["market_structure"]

    assert structure["swing_window"] == 2
    assert structure["swings"]["highs"]
    assert structure["swings"]["lows"]
    assert structure["comparisons"]["highs"] in {"HH", "LH", "INSUFFICIENT"}
    assert structure["comparisons"]["lows"] in {"HL", "LL", "INSUFFICIENT"}
    assert "regime" not in structure


def test_decision_schema_only_allows_the_new_relationships_and_no_execution_fields():
    """A model cannot turn a context judgment into execution instructions."""

    valid = CPRAgentDecision(
        action="ENTER_LONG",
        regime="TRENDING",
        setup="TREND_DAY_CONTINUATION",
        confidence=7,
        reasoning="The session is a clean staircase above VWAP.",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )
    assert valid.action == "ENTER_LONG"

    with pytest.raises(ValidationError):
        CPRAgentDecision(
            action="ENTER_LONG",
            regime="TRENDING",
            setup="NONE",
            confidence=7,
            reasoning="invalid relationship",
            model_used="gpt-5.6-terra",
            prompt_version=CPR_AI_PROMPT_VERSION,
        )
    with pytest.raises(ValidationError):
        CPRAgentDecision.model_validate({**valid.model_dump(), "lots": 1})


def test_frozen_context_registry_has_exact_no_argument_tools_and_returns_deep_copies():
    """Every tool reads the same frozen bar, yet no caller can mutate another's view."""

    context = build_cpr_context(_two_session_frame(), position_state={"is_flat": True, "entry_price": None})
    registry = FrozenCPRContextRegistry(context)
    first = registry.read("session_levels")
    first["levels"]["pivot"] = -1
    second = registry.read("session_levels")

    assert (
        registry.tool_names
        == EXPECTED_TOOL_NAMES
        == (
            "session_levels",
            "momentum_vwap",
            "market_structure",
            "position_state",
        )
    )
    assert second["levels"]["pivot"] > 0
    assert first is not second
    assert first["levels"] is not second["levels"]
    assert registry.read("position_state") == {"is_flat": True, "entry_price": None}
    assert context["session_levels"]["levels"]["pivot"] > 0


def test_position_state_rejects_venue_credential_and_execution_fields_before_and_after_freezing(tmp_path):
    """Only validated market/position facts may cross either context boundary."""

    with pytest.raises(ValidationError):
        build_cpr_context(_two_session_frame(), position_state={"is_flat": True, "broker": "DHAN"})

    context = build_cpr_context(_two_session_frame(), position_state={"is_flat": True, "entry_price": None})
    context["position_state"] = {"is_flat": True, "api_key": "secret"}
    with pytest.raises(ValidationError):
        FrozenCPRContextRegistry(context)

    snapshot_path = tmp_path / "forbidden-position-state.json"
    snapshot_path.write_text(
        '{"session_levels":{},"momentum_vwap":{},"market_structure":{},"position_state":{"is_flat":true,"venue":"X"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_snapshot_payload(str(snapshot_path))


@pytest.mark.parametrize("falsey_payload", [[], "", 0, False])
def test_position_state_rejects_every_falsey_non_mapping_before_coercion(falsey_payload, tmp_path):
    """Falsey JSON values must not silently become an empty position payload."""

    with pytest.raises(TypeError, match="mapping or None"):
        validate_position_state(falsey_payload)

    snapshot_path = tmp_path / "falsey-position-state.json"
    snapshot_path.write_text(
        '{"session_levels":{},"momentum_vwap":{},"market_structure":{},"position_state":[]}',
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="mapping or None"):
        load_snapshot_payload(str(snapshot_path))


def test_position_state_allows_only_typed_host_judgment_facts():
    """Task 3 gets risk-management facts, never broker, contract, or size data."""

    validated = validate_position_state(
        {
            "is_flat": False,
            "direction": "LONG",
            "original_entry_price": 100.0,
            "original_risk_points": 5.0,
            "original_protective_stop": 95.0,
            "current_protective_stop": 95.0,
            "premise": "TREND_DAY_CONTINUATION",
            "setup": "TREND_DAY_CONTINUATION",
            "entries_today": 1,
        }
    )

    assert validated["original_risk_points"] == 5.0
    assert validated["entries_today"] == 1
    with pytest.raises(ValidationError):
        validate_position_state({"is_flat": True, "quantity": 1})
    # The retired trailing and add-on facts are no longer part of the contract.
    for retired in ({"trailing_stage": "BREAKEVEN"}, {"scale_in_count": 0}, {"premise": "SIDEWAYS_SRSI"}):
        with pytest.raises(ValidationError):
            validate_position_state({"is_flat": False, **retired})


def test_mcp_server_exposes_exactly_four_no_argument_frozen_context_tools(tmp_path):
    """The real MCPServer registration surface must match the prompt contract."""

    registry = FrozenCPRContextRegistry(
        build_cpr_context(_two_session_frame(), position_state={"is_flat": True, "entry_price": None})
    )
    snapshot_path = tmp_path / "cpr-context.json"
    registry.write_snapshot_file(str(snapshot_path))

    server = build_mcp_server(str(snapshot_path))

    assert tuple(server._tool_manager._tools) == EXPECTED_TOOL_NAMES
    for name in EXPECTED_TOOL_NAMES:
        tool = server._tool_manager.get_tool(name)
        assert tool is not None
        assert tool.parameters["properties"] == {}
        assert tool.fn() == registry.read(name)


def test_prompt_requires_tools_judgment_risk_boundary_and_future_knowledge_seam():
    """The prompt must guide judgment while reserving execution for the host."""

    prompt = build_system_prompt(
        model_used="configured-test-model",
        operator_approved_knowledge="Only after human approval.",
    )

    assert all(name in prompt for name in EXPECTED_TOOL_NAMES)
    assert "SIDEWAYS" in prompt and "TRENDING" in prompt and "UNDECIDED" in prompt
    assert "TREND_DAY_CONTINUATION" in prompt and "PREMISE_EXIT" in prompt
    assert "trend_day_candidate" in prompt and "VWAP" in prompt
    # The model must know it may only accept or veto, never reverse, a candidate,
    # and that the live expression is a SOLD option held for time decay.
    assert "never against the candidate's" in prompt and "veto" in prompt
    assert "SELLS the ATM put" in prompt and "SELLS\nthe ATM call" in prompt
    # The busted beliefs are model-facing knowledge too.
    assert "OI" in prompt and "BankNIFTY" in prompt and "trailing" in prompt
    assert "SCALE_IN" not in prompt and "SRSI" not in prompt
    assert "HOLD" in prompt and "NONE" in prompt
    assert "host-owned" in prompt.lower()
    assert "confidence" in prompt and "0 through 10" in prompt
    assert "model_used" in prompt and "configured-test-model" in prompt
    assert "FUTURE OPERATOR-APPROVED KNOWLEDGE" in prompt
    assert CPR_AI_PROMPT_VERSION in prompt


def _trend_day_frame() -> pd.DataFrame:
    """Six prior sessions plus a steadily rising current session up to 11:04.

    Prior-session ranges (oldest first) are 50, 40, 30, 20, 10, 60, so ATR is
    the mean of the newest five: 32. The last prior day (H 130, L 70, C 100)
    puts R1 at 130. Today rises 0.4 points a minute from 100, so the 11:00 bar
    closes near 143.6 -- above R1, far above VWAP, and at the session high.
    """

    rows: list[dict[str, object]] = []
    for day, session_range in zip(range(3, 9), (50.0, 40.0, 30.0, 20.0, 10.0, 60.0), strict=True):
        prior = _minute_rows(datetime(2026, 8, day, 9, 15), [100.0] * 30)
        prior[0]["low"] = 100.0 - session_range / 2
        prior[1]["high"] = 100.0 + session_range / 2
        rows += prior
    rows += _minute_rows(datetime(2026, 8, 10, 9, 15), [100.0 + 0.4 * index for index in range(110)])
    return pd.DataFrame(rows)


def test_context_candidate_is_exactly_the_shared_gate_verdict():
    """The frozen candidate must equal what the backtest's gate computes on the same bars."""

    frame = _trend_day_frame()
    as_of = datetime(2026, 8, 10, 11, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
    context = build_cpr_context(frame, as_of=as_of)
    bars = build_completed_five_minute_bars(frame, as_of=as_of)
    session_bars = bars.loc[bars["timestamp"].dt.date == datetime(2026, 8, 10).date()].reset_index(drop=True)
    expected = evaluate_trend_day_candidate(
        session_bars, prior_high=130.0, prior_low=70.0, prior_close=100.0, atr=32.0, atr_sessions_used=5
    ).to_dict()

    candidate = context["market_structure"]["trend_day_candidate"]
    assert candidate == expected
    assert candidate["eligible"] is True and candidate["direction"] == "LONG"
    assert candidate["entry"] == context["session_levels"]["current_close"]
    assert candidate["stop"] == pytest.approx(context["momentum_vwap"]["vwap"]["value"])
    assert (candidate["beyond_r1_s1"], candidate["gap_in_direction"], candidate["extended_from_vwap"]) == (
        True,
        False,
        True,
    )
    assert context["session_levels"]["atr"] == {
        "value": 32.0,
        "sessions_used": 5,
        "prior_session_ranges": [40.0, 30.0, 20.0, 10.0, 60.0],
    }
    assert context["momentum_vwap"]["vwap"]["distance_atr"] == pytest.approx(candidate["distance_from_vwap_atr"])
    assert context["momentum_vwap"]["candle"]["range_atr"] == pytest.approx(
        context["momentum_vwap"]["candle"]["range"] / 32.0
    )
    assert len(context["momentum_vwap"]["recent_candles"]) == 6
    # The frozen MCP snapshot is the same JSON the host validates against.
    assert freeze_cpr_context(frame, as_of=as_of).snapshot_payload() == context


def test_host_accepts_the_live_context_candidate_end_to_end():
    """Context plus policy: a real frozen candidate yields host-owned geometry."""

    context = build_cpr_context(
        _trend_day_frame(),
        position_state={"is_flat": True, "entries_today": 0},
        as_of=datetime(2026, 8, 10, 11, 5, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    proposal = CPRAgentDecision(
        action="ENTER_LONG",
        regime="TRENDING",
        setup="TREND_DAY_CONTINUATION",
        confidence=8,
        reasoning="Staircase above VWAP beyond R1.",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )

    outcome = CPRHostPolicy().validate(context, proposal)
    candidate = context["market_structure"]["trend_day_candidate"]

    assert outcome.accepted and outcome.validation_code == "accepted_entry"
    assert outcome.entry_price == candidate["entry"]
    assert outcome.stop_price == candidate["stop"]
    assert outcome.risk_points == pytest.approx(candidate["entry"] - candidate["stop"])

