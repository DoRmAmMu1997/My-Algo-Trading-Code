"""Tests for the dashboard's chart indicators (`Dependencies/dashboard_indicators.py`).

Two properties carry most of the weight:

* **The maths is not forked.** The chart's CPR deliberately reads a truncated
  prior session, but nothing else about it may differ from what the CPR
  strategies compute. One test feeds an UNTRUNCATED session through both and
  demands the levels match, so "only the input window differs" is enforced
  rather than asserted in a comment.
* **Nothing reaches the renderer as NaN.** `render_document_bytes` uses
  `allow_nan=False`, so a single NaN raises inside the renderer and freezes the
  live dashboard on its last good document for the rest of the session. Every
  fixture here is run through it.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import pytest
from check_env_config import env_keys_read_by

# Bare import: this folder's conftest.py puts the SOURCE `Dependencies/` on
# sys.path, which is the same resolution the runtime performs.
from dashboard_indicators import (
    CPR_SESSION_END,
    CPR_SESSION_START,
    ChartCpr,
    bar_records,
    chart_cpr,
    chart_document,
    finite_or_none,
    forming_bucket,
    pivot_levels,
    prior_session_window,
    round_or_none,
    session_vwap_values,
    stochastic_values,
    timeframe_block,
)
from dashboard_snapshot import render_document_bytes

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_strategy_module(name: str, relative: str):
    """Load a real strategy module the way the master's `load_module` does.

    These live under the spaced `Signal Generators/` tree, so they cannot be
    imported normally. The equality tests below need the genuine article --
    a stand-in would prove nothing about drift.
    """

    if name in sys.modules:
        return sys.modules[name]
    # `cpr_strategy_logic` imports `Dependencies.market_data_health`, so the
    # repository root has to be importable as a package root.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    path = _REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _real_modules():
    """The three strategy modules, or a skip when TA-Lib is unavailable."""

    try:
        misc = _load_strategy_module(
            "misc_strategy_common", "Signal Generators/misc_strategy_common.py"
        )
        regime = _load_strategy_module(
            "regime_common", "Signal Generators/Regime Adaptive Strategy/regime_common.py"
        )
        cpr = _load_strategy_module(
            "cpr_strategy_logic", "Signal Generators/CPR Strategy/cpr_strategy_logic.py"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"strategy modules unavailable ({exc})")
    return misc, regime, cpr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def session_frame(
    sessions: tuple[str, ...] = ("2026-09-10", "2026-09-11"),
    bars: tuple[int, ...] = (375, 120),
    *,
    start: str = "09:15",
    seed: float = 24500.0,
) -> pd.DataFrame:
    """A realistic multi-session 1-minute frame.

    A deterministic zig-zag rather than a straight line, so highs and lows are
    not degenerate and a stochastic has something to range over.
    """

    rows = []
    price = seed
    for session, count in zip(sessions, bars, strict=True):
        opening = pd.Timestamp(f"{session} {start}")
        for index in range(count):
            # Two interfering periods: never flat, never monotonic.
            price = seed + 40.0 * math.sin(index / 17.0) + 12.0 * math.sin(index / 3.0)
            rows.append(
                {
                    "timestamp": opening + pd.Timedelta(minutes=index),
                    "open": round(price - 1.5, 2),
                    "high": round(price + 3.0, 2),
                    "low": round(price - 3.0, 2),
                    "close": round(price, 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value", [None, "24500", float("nan"), float("inf"), float("-inf"), True, [1], {}]
)
def test_non_numbers_and_non_finite_values_become_none(value):
    """The gate in front of `allow_nan=False`."""
    assert finite_or_none(value) is None
    assert round_or_none(value) is None


def test_real_numbers_survive_and_round():
    assert finite_or_none(3) == 3.0
    assert round_or_none(24500.12345) == 24500.12


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def test_bar_records_match_the_masters_row_at_a_time_version():
    """The whole reason the vectorized serializer is allowed to exist.

    pandas 3 defaults to microsecond resolution, so the obvious
    `astype("int64") // 10**9` yields timestamps a thousand times too small and
    draws the session in 1970. This test is what catches that.
    """

    master = _load_strategy_module("master_for_bar_record", "nifty_multi_strategy_master.py")
    frame = session_frame(("2026-09-11",), (40,))
    assert bar_records(frame) == [master._bar_record(row) for _, row in frame.iterrows()]


def test_bar_times_are_the_exchange_wall_clock():
    frame = session_frame(("2026-09-11",), (3,))
    first = bar_records(frame)[0]
    assert pd.Timestamp(first["time"], unit="s").strftime("%H:%M") == "09:15"


def test_an_empty_frame_serializes_to_nothing():
    assert bar_records(session_frame(("2026-09-11",), (0,))) == []


def test_the_forming_bucket_aggregates_only_the_finished_minutes():
    frame = session_frame(("2026-09-11",), (17,))  # 09:15..09:31 -> 09:30 bucket has 2 bars
    record, anchor = forming_bucket(frame, timeframe_minutes=5)
    assert anchor == pd.Timestamp("2026-09-11 09:30")
    tail = frame.tail(2)
    assert record["open"] == pytest.approx(tail["open"].iloc[0])
    assert record["high"] == pytest.approx(tail["high"].max())
    assert record["low"] == pytest.approx(tail["low"].min())
    assert record["close"] == pytest.approx(tail["close"].iloc[-1])


def test_a_complete_bucket_is_left_to_the_resampler():
    """15 bars ends exactly on a boundary, so nothing is forming."""
    assert forming_bucket(session_frame(("2026-09-11",), (15,)), timeframe_minutes=5) == (None, None)


def test_one_minute_has_no_forming_bucket():
    assert forming_bucket(session_frame(("2026-09-11",), (7,)), timeframe_minutes=1) == (None, None)


# ---------------------------------------------------------------------------
# CPR
# ---------------------------------------------------------------------------
def test_the_prior_session_is_truncated_at_1515():
    """A high printed after 15:15 must not reach the levels."""
    frame = session_frame(("2026-09-10", "2026-09-11"), (375, 30))
    # 09:15 + 374 minutes = 15:29, so this session genuinely runs past the cut.
    spike_at = frame.index[(frame["timestamp"] == pd.Timestamp("2026-09-10 15:22"))][0]
    frame.loc[spike_at, "high"] = 99999.0

    window = prior_session_window(frame)
    assert window.last_bar == "15:15"
    assert window.bars_used == 361  # 09:15..15:15 inclusive
    levels = chart_cpr(frame)
    assert levels.prev_high != 99999.0

    # ...and the untruncated view would have seen it.
    assert frame.loc[frame["timestamp"].dt.date == date(2026, 9, 10), "high"].max() == 99999.0


def test_only_the_input_window_differs_from_the_strategies_cpr():
    """Fed an UNTRUNCATED prior session, the chart reproduces `_add_daily_cpr`.

    This is the test that pins the algebra. If someone "improves" a formula on
    either side, this fails.
    """

    _misc, _regime, cpr_logic = _real_modules()
    frame = session_frame(("2026-09-10", "2026-09-11"), (375, 30))

    # Widen the chart's window so both see the whole prior session.
    mine = chart_cpr(frame, session_start=time(0, 0), session_end=time(23, 59))

    theirs = cpr_logic._add_daily_cpr(frame)
    today = theirs.loc[theirs["timestamp"].dt.date == date(2026, 9, 11)].iloc[-1]

    for name in ("pivot", "bc", "tc", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"):
        assert getattr(mine, name) == pytest.approx(float(today[name]), abs=5e-3), name
    assert mine.prev_high == pytest.approx(float(today["prev_high"]), abs=5e-3)
    assert mine.prev_low == pytest.approx(float(today["prev_low"]), abs=5e-3)
    assert mine.prev_close == pytest.approx(float(today["prev_close"]), abs=5e-3)


def test_levels_follow_the_standard_algebra():
    levels = pivot_levels(110.0, 90.0, 100.0)
    assert levels["pivot"] == pytest.approx(100.0)
    assert levels["bc"] == pytest.approx(100.0)
    assert levels["tc"] == pytest.approx(100.0)
    assert levels["r1"] == pytest.approx(110.0)
    assert levels["s1"] == pytest.approx(90.0)
    assert levels["r4"] == pytest.approx(levels["r3"] + 20.0)
    assert levels["s4"] == pytest.approx(levels["s3"] - 20.0)


def test_a_weekend_gap_uses_the_previous_available_session():
    frame = session_frame(("2026-09-11", "2026-09-14"), (375, 30))  # Friday then Monday
    assert prior_session_window(frame).session_date == date(2026, 9, 11)


def test_a_half_day_prior_session_still_yields_levels():
    frame = session_frame(("2026-09-10", "2026-09-11"), (180, 30))  # prior ends 12:14
    levels = chart_cpr(frame)
    assert levels.available
    assert levels.last_bar == "12:14"
    assert levels.partial is False  # it started at the open; it simply ended early


def test_no_prior_session_reports_unavailable_with_every_level_none():
    """A confident zero would be far worse than an empty chart."""
    levels = chart_cpr(session_frame(("2026-09-11",), (120,)))
    assert levels.available is False
    assert levels.unavailable_reason
    for name in ("pivot", "bc", "tc", "r1", "s1", "prev_high", "prev_close"):
        assert getattr(levels, name) is None


def test_a_prior_session_starting_late_is_flagged_but_still_drawn():
    frame = session_frame(("2026-09-10", "2026-09-11"), (375, 30))
    late = frame.loc[
        (frame["timestamp"].dt.date != date(2026, 9, 10))
        | (frame["timestamp"] >= pd.Timestamp("2026-09-10 11:00"))
    ].reset_index(drop=True)
    levels = chart_cpr(late)
    assert levels.available
    assert levels.partial is True


def test_the_strategies_pivot_is_carried_for_contrast():
    frame = session_frame(("2026-09-10", "2026-09-11"), (375, 30))
    levels = chart_cpr(frame)
    assert levels.strategies_pivot is not None
    # Different windows, so normally different numbers -- both must be real.
    assert math.isfinite(levels.pivot) and math.isfinite(levels.strategies_pivot)


def test_the_payload_always_says_it_is_chart_only():
    payload = chart_cpr(session_frame()).as_dict()
    assert payload["chart_only"] is True
    assert "chart cpr reads the prior session 09:15-15:15" in payload["caveat"].lower()
    assert "09:15" in payload["caveat"] and "15:15" in payload["caveat"]


def test_the_window_constants_are_the_documented_ones():
    assert (time(9, 15), time(15, 15)) == (CPR_SESSION_START, CPR_SESSION_END)


# ---------------------------------------------------------------------------
# VWAP and stochastic (against the real strategy helpers)
# ---------------------------------------------------------------------------
def test_vwap_equals_the_regime_strategys_own_vwap():
    _misc, regime, _cpr = _real_modules()
    frame = session_frame()
    values, is_proxy = session_vwap_values(
        frame,
        attach_session_date=regime.attach_session_date,
        attach_session_vwap=regime.attach_session_vwap,
    )
    expected = regime.attach_session_vwap(regime.attach_session_date(frame))["vwap"]
    assert values == [round(float(value), 2) for value in expected]
    assert is_proxy is True  # the index feed carries no volume


def test_vwap_resets_at_the_session_boundary():
    _misc, regime, _cpr = _real_modules()
    frame = session_frame()
    values, _ = session_vwap_values(
        frame,
        attach_session_date=regime.attach_session_date,
        attach_session_vwap=regime.attach_session_vwap,
    )
    first_of_today = len(frame.loc[frame["timestamp"].dt.date == date(2026, 9, 10)])
    typical = frame.iloc[first_of_today][["high", "low", "close"]].mean()
    assert values[first_of_today] == pytest.approx(round(float(typical), 2), abs=0.02)


def test_stochastic_equals_the_strategys_own_helper():
    misc, _regime, _cpr = _real_modules()
    frame = session_frame(("2026-09-11",), (200,))
    k_values, d_values = stochastic_values(
        frame, stochastic_fn=misc.stochastic, k_period=14, d_period=3, smooth_k=3
    )
    k_expected, d_expected = misc.stochastic(frame, k_period=14, d_period=3, smooth_k=3)
    assert k_values == [round_or_none(value) for value in k_expected]
    assert d_values == [round_or_none(value) for value in d_expected]
    assert all(value is None or 0.0 <= value <= 100.0 for value in k_values)


def test_a_frame_shorter_than_the_warmup_is_all_none_never_nan():
    misc, _regime, _cpr = _real_modules()
    k_values, d_values = stochastic_values(
        session_frame(("2026-09-11",), (5,)),
        stochastic_fn=misc.stochastic, k_period=14, d_period=3, smooth_k=3,
    )
    assert k_values == [None] * 5
    assert d_values == [None] * 5


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def test_indicator_columns_are_right_aligned_to_the_bars():
    """Indicators are computed over more history than is displayed."""
    block = timeframe_block(
        minutes=1,
        bars=[{"time": 1}, {"time": 2}, {"time": 3}],
        vwap=[10.0, 11.0, 12.0, 13.0, 14.0],  # longer: keep the newest 3
        stoch_k=[50.0],  # shorter: pad the front
        stoch_d=[],
    )
    assert block["vwap"] == [12.0, 13.0, 14.0]
    assert block["stoch_k"] == [None, None, 50.0]
    assert block["stoch_d"] == [None, None, None]
    assert "forming_from" not in block


def test_a_forming_bar_is_named_in_the_payload():
    block = timeframe_block(
        minutes=5, bars=[{"time": 1}], vwap=[], stoch_k=[], stoch_d=[], forming_from=1757600700
    )
    assert block["forming_from"] == 1757600700


def test_the_document_labels_the_vwap_as_a_proxy():
    document = chart_document(
        series_version=3,
        timeframes={"1": timeframe_block(minutes=1, bars=[], vwap=[], stoch_k=[], stoch_d=[])},
        cpr=ChartCpr(False),
        vwap_is_proxy=True,
        stochastic_settings={"k_period": 14, "d_period": 3, "smooth_k": 3},
    )
    assert document["indicators"]["vwap"]["label"] == "VWAP*"
    assert "no volume" in document["indicators"]["vwap"]["note"]
    assert document["indicators"]["stochastic"]["overbought"] == 80


# ---------------------------------------------------------------------------
# The renderer trap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("sessions", "bars"),
    [
        (("2026-09-10", "2026-09-11"), (375, 120)),  # ordinary
        (("2026-09-11",), (120,)),                   # no prior session
        (("2026-09-10", "2026-09-11"), (180, 5)),    # half day, tiny today
        (("2026-09-11",), (1,)),                     # a single bar
    ],
)
def test_every_shape_survives_the_json_renderer(sessions, bars):
    """`allow_nan=False`: one NaN would freeze the live dashboard for the day."""
    misc, regime, _cpr = _real_modules()
    frame = session_frame(sessions, bars)
    vwap, is_proxy = session_vwap_values(
        frame,
        attach_session_date=regime.attach_session_date,
        attach_session_vwap=regime.attach_session_vwap,
    )
    k_values, d_values = stochastic_values(
        frame, stochastic_fn=misc.stochastic, k_period=14, d_period=3, smooth_k=3
    )
    document = chart_document(
        series_version=1,
        timeframes={
            "1": timeframe_block(
                minutes=1, bars=bar_records(frame), vwap=vwap, stoch_k=k_values, stoch_d=d_values
            )
        },
        cpr=chart_cpr(frame),
        vwap_is_proxy=is_proxy,
        stochastic_settings={"k_period": 14, "d_period": 3, "smooth_k": 3},
    )
    assert render_document_bytes(document)  # must not raise


def test_the_module_reads_no_configuration():
    """All config is read in the master, the only place check-env audits."""


    path = _REPO_ROOT / "Dependencies" / "dashboard_indicators.py"
    # The repo's own AST auditor, so this and `algo.py check-env` can never
    # disagree about what counts as reading a setting.
    assert env_keys_read_by(path) == set()
    # `os` is the only other route to an unaudited read, and this module has no
    # use for it. Checked structurally: the docstring legitimately NAMES
    # `os.getenv` while explaining the rule.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "os" not in imported


# ---------------------------------------------------------------------------
# Degenerate inputs: every one of these has been seen or is one bad feed away
# ---------------------------------------------------------------------------
def test_an_empty_frame_has_no_prior_session():
    assert prior_session_window(pd.DataFrame()).session_date is None
    assert chart_cpr(pd.DataFrame()).available is False


def test_a_frame_without_a_timestamp_column_is_refused_not_crashed():
    assert prior_session_window(pd.DataFrame({"close": [1.0]})).session_date is None


def test_a_prior_session_entirely_outside_the_window_reports_why():
    """An afternoon-only prior session has nothing in 09:15-15:15."""
    frame = session_frame(("2026-09-10", "2026-09-11"), (375, 30))
    afternoon = frame.loc[
        (frame["timestamp"].dt.date != date(2026, 9, 10))
        | (frame["timestamp"] > pd.Timestamp("2026-09-10 15:15"))
    ].reset_index(drop=True)
    levels = chart_cpr(afternoon)
    assert levels.available is False
    assert levels.prior_session_date == "2026-09-10"
    assert "no bars between" in levels.unavailable_reason


def test_a_non_finite_prior_close_is_refused_rather_than_drawn():
    frame = session_frame(("2026-09-10", "2026-09-11"), (30, 10))
    frame.loc[frame["timestamp"] == pd.Timestamp("2026-09-10 09:44"), "close"] = float("nan")
    levels = chart_cpr(frame)
    assert levels.available is False
    assert "finite" in levels.unavailable_reason


def test_a_width_classifier_that_raises_costs_only_the_label():
    """A cosmetic tag must never take the levels down with it."""

    def explode(*_args):
        raise RuntimeError("classifier is broken")

    levels = chart_cpr(session_frame(), width_classifier=explode)
    assert levels.available is True
    assert levels.pivot is not None
    assert levels.width is None


def test_the_width_label_comes_from_the_injected_classifier():
    levels = chart_cpr(session_frame(), width_classifier=lambda *_: "narrow")
    assert levels.width == "narrow"


def test_the_strategies_pivot_is_omitted_when_the_full_day_cannot_be_read():
    frame = session_frame(("2026-09-10", "2026-09-11"), (30, 10))
    frame.loc[frame["timestamp"].dt.date == date(2026, 9, 10), "high"] = float("nan")
    # The truncated window fails first, so this simply must not raise.
    assert chart_cpr(frame).available is False


def test_empty_frames_yield_empty_indicator_columns():
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    assert session_vwap_values(
        empty, attach_session_date=lambda f: f, attach_session_vwap=lambda f: f
    ) == ([], True)
    assert stochastic_values(
        empty, stochastic_fn=lambda *a, **k: (None, None), k_period=14, d_period=3, smooth_k=3
    ) == ([], [])


def test_a_vwap_helper_without_the_proxy_flag_is_assumed_to_be_a_proxy():
    """Erring toward the caveat: an unlabelled VWAP is treated as the proxy."""
    frame = session_frame(("2026-09-11",), (5,))

    def attach(f):
        out = f.copy()
        out["vwap"] = f["close"]
        return out  # deliberately no `vwap_is_proxy` column

    values, is_proxy = session_vwap_values(
        frame, attach_session_date=lambda f: f, attach_session_vwap=attach
    )
    assert is_proxy is True
    assert values == [round(float(v), 2) for v in frame["close"]]
