"""Specify the independent five-minute CPR context supplied to the CPR agent.

The fixtures use deliberately hand-derived prices.  They do not call any CPR
Strategy helper, which proves this replacement layer owns its own calculations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from math import sin

import pandas as pd
import pytest
from cpr_ai_context import build_completed_five_minute_bars, build_cpr_context
from cpr_ai_mcp_server import build_mcp_server, load_snapshot_payload
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION, build_system_prompt
from cpr_ai_schema import CPRAgentDecision
from cpr_ai_tools import EXPECTED_TOOL_NAMES, FrozenCPRContextRegistry
from pydantic import ValidationError


def _minute_rows(start: datetime, closes: list[float], *, volume: float | None = None) -> list[dict[str, object]]:
    """Create one-minute candles whose OHLC values are easy to verify by hand."""

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


def _two_session_frame() -> pd.DataFrame:
    """Return a prior session plus enough current bars for all indicators."""

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


def test_context_rejects_a_newest_session_without_a_completed_five_minute_bar():
    """One to four newest-session minutes must not reuse yesterday's context."""

    newest_partial_session = _minute_rows(datetime(2026, 8, 3, 9, 15), [200.0, 201.0, 202.0, 203.0])

    with pytest.raises(ValueError, match="latest input session"):
        build_cpr_context(pd.concat([_two_session_frame(), pd.DataFrame(newest_partial_session)], ignore_index=True))


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
    assert levels["next_levels"]["buffer_points"] == 2.0
    # The current close is far above all calculated levels, so the buffered
    # view intentionally has no upside candidate and R2 is nearest below.
    assert levels["next_levels"]["upside"] is None
    assert levels["next_levels"]["downside"]["name"] == "r2"


def test_momentum_contains_tradingview_srsi_and_equal_weight_vwap_fallback():
    """Missing index volume uses typical-price averaging while SRSI exposes crosses."""

    rows = _minute_rows(datetime(2026, 8, 2, 9, 15), [100 + (index % 7) for index in range(180)], volume=None)
    # Supply a full earlier day so levels are available; all current volumes are absent.
    rows = _minute_rows(datetime(2026, 8, 1, 9, 15), [100.0] * 30) + rows
    context = build_cpr_context(pd.DataFrame(rows))
    momentum = context["momentum_vwap"]

    assert momentum["vwap"]["method"] == "equal_weight_typical_price"
    assert momentum["stochastic_rsi"]["rsi_length"] == 14
    assert momentum["stochastic_rsi"]["stochastic_length"] == 14
    assert momentum["stochastic_rsi"]["k_sma_length"] == 3
    assert momentum["stochastic_rsi"]["d_sma_length"] == 3
    assert momentum["stochastic_rsi"]["oversold"] == 20.0
    assert momentum["stochastic_rsi"]["overbought"] == 80.0
    assert {"current_k", "previous_k", "current_d", "previous_d", "cross_up", "cross_down"} <= set(
        momentum["stochastic_rsi"]
    )


def test_srsi_literal_crosses_report_their_direction_and_matching_zone_flags():
    """Known 14/14/3/3 outputs must preserve cross and zone semantics."""

    previous = _minute_rows(datetime(2026, 8, 1, 9, 15), [100.0] * 30)

    def srsi_for_completed_bars(count: int) -> dict[str, object]:
        """Build a repeatable five-minute close fixture without reusing production helpers."""

        five_minute_closes = [100.0 + 10.0 * sin(index * 0.65) for index in range(count)]
        current_minutes = _minute_rows(
            datetime(2026, 8, 2, 9, 15),
            [close for close in five_minute_closes for _ in range(5)],
        )
        return build_cpr_context(pd.DataFrame(previous + current_minutes))["momentum_vwap"]["stochastic_rsi"]

    # These values are literal results from the standard Wilder RSI(14),
    # Stoch(14), K SMA(3), D SMA(3) equations for the listed sine closes.
    oversold_cross = srsi_for_completed_bars(39)
    assert oversold_cross["current_k"] == pytest.approx(11.0362255097)
    assert oversold_cross["current_d"] == pytest.approx(5.5542834290)
    assert oversold_cross["cross_up"] is True
    assert oversold_cross["cross_down"] is False
    assert oversold_cross["cross_up_in_oversold"] is True
    assert oversold_cross["cross_down_in_overbought"] is False

    overbought_cross = srsi_for_completed_bars(73)
    assert overbought_cross["current_k"] == pytest.approx(82.6066025590)
    assert overbought_cross["current_d"] == pytest.approx(86.6324134948)
    assert overbought_cross["cross_up"] is False
    assert overbought_cross["cross_down"] is True
    assert overbought_cross["cross_up_in_oversold"] is False
    assert overbought_cross["cross_down_in_overbought"] is True


def test_momentum_exposes_rsi_ema_candle_and_deterministic_vwap_sequence_evidence():
    """Indicator consumers receive values and evidence, not an inferred regime."""

    context = build_cpr_context(_two_session_frame())
    momentum = context["momentum_vwap"]

    assert momentum["rsi14"] is not None
    assert momentum["ema"]["ema5"] is not None
    assert momentum["ema"]["ema20"] is not None
    assert momentum["ema"]["order"] in {"EMA5_ABOVE_EMA20", "EMA5_BELOW_EMA20", "EQUAL"}
    assert momentum["candle"]["colour"] == "BULLISH"
    assert momentum["candle"]["range"] > momentum["candle"]["body"] > 0
    assert len(momentum["recent_candles"]) == 5
    assert len(momentum["vwap"]["sequence_evidence"]["relations"]) == 3


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


def test_r1_candidate_is_long_only_after_bearish_touch_then_bullish_reclaim():
    """There is no symmetric S1-short setup in the deterministic context."""

    rows = _minute_rows(datetime(2026, 8, 1, 9, 15), [100.0] * 30)
    rows[0]["low"], rows[1]["high"], rows[-1]["close"] = 90.0, 110.0, 105.0
    current = _minute_rows(datetime(2026, 8, 2, 9, 15), [105.0] * 40)
    # R1 is 113.333...; shape the last two completed five-minute candles.
    for offset in range(5):
        current[30 + offset].update({"open": 114.0, "high": 114.5, "low": 112.8, "close": 113.0})
        current[35 + offset].update({"open": 113.1, "high": 114.2, "low": 113.0, "close": 114.0})

    candidate = build_cpr_context(pd.DataFrame(rows + current))["market_structure"]["r1_scale_in_candidate"]

    assert candidate["eligible"] is True
    assert candidate["direction"] == "LONG"
    assert "s1" not in candidate


def test_decision_schema_only_allows_the_new_relationships_and_no_execution_fields():
    """A model cannot turn a context judgment into execution instructions."""

    valid = CPRAgentDecision(
        action="SCALE_IN",
        regime="TRENDING",
        setup="R1_SCALE_IN",
        confidence=7,
        reasoning="The completed bar held R1.",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )
    assert valid.action == "SCALE_IN"

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


def test_mcp_server_exposes_exactly_four_no_argument_frozen_context_tools(tmp_path):
    """The real FastMCP registration surface must match the prompt contract."""

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

    prompt = build_system_prompt(operator_approved_knowledge="Only after human approval.")

    assert all(name in prompt for name in EXPECTED_TOOL_NAMES)
    assert "SIDEWAYS" in prompt and "TRENDING" in prompt and "UNDECIDED" in prompt
    assert "breakout" in prompt.lower() and "breakdown" in prompt.lower()
    assert "SRSI" in prompt and "VWAP" in prompt and "PREMISE_EXIT" in prompt
    assert "host-owned" in prompt.lower()
    assert "FUTURE OPERATOR-APPROVED KNOWLEDGE" in prompt
    assert CPR_AI_PROMPT_VERSION in prompt
