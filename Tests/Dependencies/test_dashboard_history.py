"""The dashboard's history loader: paging, daily bars, and degrading politely.

The single most important behaviour here is the LAST one: the operator may not
have downloaded five years of candles, and the chart must then look exactly as
it did before rather than breaking. Everything else is arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from Dependencies import dashboard_history


def _renderer(document):
    return json.dumps(document, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _resample(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """A stand-in for the strategies' own resampler: complete buckets only."""

    if frame.empty:
        return frame
    grouped = (
        frame.set_index("timestamp")
        .resample(f"{minutes}min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return grouped


def _minutes(day: str, count: int, start: str = "09:15") -> pd.DataFrame:
    base = pd.Timestamp(f"{day} {start}")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(minutes=index) for index in range(count)],
            "open": [100.0 + index for index in range(count)],
            "high": [101.0 + index for index in range(count)],
            "low": [99.0 + index for index in range(count)],
            "close": [100.5 + index for index in range(count)],
        }
    )


def _csv(tmp_path: Path, frame: pd.DataFrame) -> Path:
    target = tmp_path / "history.csv"
    out = frame.copy()
    out["volume"] = 0.0
    out.to_csv(target, index=False)
    return target


# ------------------------------------------------------- degrading politely


def test_a_missing_csv_degrades_instead_of_raising(tmp_path):
    """The operator may simply not have downloaded the history yet."""

    history = dashboard_history.build_history(
        path=tmp_path / "nothing.csv", renderer=_renderer, resample=_resample
    )

    assert history.available is False
    assert history.series == {}
    assert "no history CSV" in (history.unavailable_reason or "")
    assert history.summary()["available"] is False


def test_an_unreadable_csv_degrades_too(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text("this is not a csv we can use\n", encoding="utf-8")

    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    )

    assert history.available is False
    assert history.unavailable_reason


def test_a_csv_with_no_in_session_bars_degrades(tmp_path):
    target = _csv(tmp_path, _minutes("2026-09-15", 5, start="18:40"))

    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    )

    assert history.available is False
    assert "no in-session bars" in (history.unavailable_reason or "")


# ------------------------------------------------------------------ loading


def test_bars_outside_the_session_never_reach_the_chart(tmp_path):
    """The synthetic wall-clock bar and the old era's 17:59 rows."""

    frame = pd.concat(
        [_minutes("2026-09-15", 3), _minutes("2026-09-15", 1, start="18:44")],
        ignore_index=True,
    )

    loaded = dashboard_history.load_history_frame(_csv(tmp_path, frame))

    assert len(loaded) == 3
    assert str(loaded["timestamp"].iloc[-1]) == "2026-09-15 09:17:00"


def test_weekend_rows_in_an_older_csv_are_dropped(tmp_path):
    """The loader does not trust the file's provenance.

    A resumed download can leave one CSV cleaned by two rule versions -- which
    is exactly what happened: 509 weekend rows survived in the chunks fetched
    before the extractor learned the weekday rule. They sit at times like 11:30,
    inside session HOURS, so the clock test alone passes them through.
    """

    frame = pd.concat(
        [
            _minutes("2026-09-11", 2),            # Friday
            _minutes("2026-09-12", 2, "11:30"),   # Saturday, mid-session by the clock
            _minutes("2026-09-14", 2),            # Monday
        ],
        ignore_index=True,
    )

    loaded = dashboard_history.load_history_frame(_csv(tmp_path, frame))

    assert len(loaded) == 4
    assert {value.date().isoformat() for value in loaded["timestamp"]} == {
        "2026-09-11",
        "2026-09-14",
    }


def test_duplicate_timestamps_are_collapsed(tmp_path):
    """lightweight-charts needs strictly ascending times or it drops bars."""

    frame = pd.concat([_minutes("2026-09-15", 3), _minutes("2026-09-15", 3)], ignore_index=True)

    loaded = dashboard_history.load_history_frame(_csv(tmp_path, frame))

    assert len(loaded) == 3
    assert loaded["timestamp"].is_monotonic_increasing
    assert not loaded["timestamp"].duplicated().any()


# -------------------------------------------------------------- daily bars


def test_daily_bars_collapse_one_session_into_one_bar():
    frame = pd.concat(
        [_minutes("2026-09-14", 5), _minutes("2026-09-15", 5)], ignore_index=True
    )

    days = dashboard_history.daily_bars(frame)

    assert len(days) == 2
    first = days.iloc[0]
    assert first["open"] == 100.0, "the session's first open"
    assert first["close"] == 104.5, "the session's last close"
    assert first["high"] == 105.0
    assert first["low"] == 99.0


def test_daily_bars_are_stamped_at_the_session_open():
    """Midnight is a time where nothing else on the chart lives."""

    days = dashboard_history.daily_bars(_minutes("2026-09-15", 5))

    assert str(days["timestamp"].iloc[0]) == "2026-09-15 09:15:00"


def test_a_weekend_gap_produces_no_empty_daily_bars():
    """Grouping by date, not resampling on a frequency, is what avoids this."""

    frame = pd.concat(
        [_minutes("2026-09-11", 3), _minutes("2026-09-14", 3)], ignore_index=True
    )

    days = dashboard_history.daily_bars(frame)

    assert len(days) == 2, "Saturday and Sunday must not appear at all"


# ------------------------------------------------------------------ paging


def test_pages_count_back_from_the_newest_bar():
    """Page 0 is the NEWEST slice: the browser scrolls left and asks 0, 1, 2."""

    assert dashboard_history.page_bounds(1000, 0, page_size=400) == (600, 1000)
    assert dashboard_history.page_bounds(1000, 1, page_size=400) == (200, 600)


def test_the_oldest_page_holds_the_remainder():
    assert dashboard_history.page_bounds(1000, 2, page_size=400) == (0, 200)


def test_a_page_past_the_start_of_history_is_refused():
    with pytest.raises(IndexError):
        dashboard_history.page_bounds(1000, 3, page_size=400)


def test_page_count_rounds_up():
    assert dashboard_history.page_count(1000, 400) == 3
    assert dashboard_history.page_count(800, 400) == 2
    assert dashboard_history.page_count(0, 400) == 0


def test_every_bar_appears_exactly_once_across_the_pages(tmp_path):
    """The arithmetic that matters: paging must not lose or repeat a candle."""

    target = _csv(tmp_path, _minutes("2026-09-15", 350))
    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample, page_size=100
    )

    series = history.series["1"]
    assert len(series.pages) == 4

    seen: list[int] = []
    for index in range(len(series.pages)):
        page = json.loads(series.page(index))
        assert page["page"] == index
        assert page["pages"] == 4
        seen = [bar["time"] for bar in page["bars"]] + seen

    assert len(seen) == 350
    assert len(set(seen)) == 350, "no candle repeated across pages"
    assert seen == sorted(seen), "the reassembled series is ascending"


def test_a_page_beyond_the_end_is_none_not_an_error(tmp_path):
    target = _csv(tmp_path, _minutes("2026-09-15", 50))
    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample, page_size=100
    )

    assert history.series["1"].page(99) is None
    assert history.series["1"].page(-1) is None


# ------------------------------------------------------------- the document


def test_every_timeframe_is_built(tmp_path):
    target = _csv(tmp_path, _minutes("2026-09-15", 60))

    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    )

    assert set(history.series) == {"1", "5", "D"}
    assert history.series["1"].bars == 60
    assert history.series["5"].bars == 12
    assert history.series["D"].bars == 1
    assert history.available is True


def test_the_summary_says_what_exists(tmp_path):
    target = _csv(tmp_path, _minutes("2026-09-15", 60))

    summary = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    ).summary()

    assert summary["available"] is True
    assert summary["timeframes"]["1"]["bars"] == 60
    assert summary["timeframes"]["1"]["pages"] == 1
    assert summary["timeframes"]["1"]["last_bar"] == "2026-09-15 10:14:00"


def test_pages_survive_the_strict_json_renderer(tmp_path):
    """`allow_nan=False` freezes the dashboard for the session if it ever trips."""

    target = _csv(tmp_path, _minutes("2026-09-15", 30))

    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    )

    for series in history.series.values():
        for page in series.pages:
            json.loads(page)


# --------------------------------------------------------- CPR segments


def _session(day: str, closes) -> pd.DataFrame:
    """One session whose closes are given minute by minute from 09:15."""

    base = pd.Timestamp(f"{day} 09:15")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(minutes=i) for i in range(len(closes))],
            "open": list(closes),
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": list(closes),
        }
    )


def test_each_day_takes_its_levels_from_the_day_before():
    """The whole point: one CPR per day, derived from that day's predecessor."""

    frame = pd.concat(
        [_session("2026-09-14", [100.0, 110.0, 105.0]), _session("2026-09-15", [200.0, 210.0])],
        ignore_index=True,
    )

    day = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["day"]

    assert len(day) == 1, "only the second day can have a band"
    assert day[0]["date"] == "2026-09-15"
    # 14th: high 111, low 99, close 105 -> pivot = (111 + 99 + 105) / 3
    assert day[0]["levels"]["pivot"] == pytest.approx((111.0 + 99.0 + 105.0) / 3, abs=0.01)
    assert day[0]["levels"]["prev_high"] == 111.0
    assert day[0]["levels"]["prev_low"] == 99.0


def test_the_first_day_has_no_band_at_all():
    """It has no predecessor, so zeros would be an invention."""

    frame = _session("2026-09-15", [100.0, 101.0])

    segments = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))

    assert segments["day"] == []


def test_the_levels_are_the_repositorys_own_algebra():
    """Computed through `pivot_levels`, not a second copy of the formulas."""

    from Dependencies.dashboard_indicators import pivot_levels

    frame = pd.concat(
        [_session("2026-09-14", [100.0, 120.0, 110.0]), _session("2026-09-15", [200.0])],
        ignore_index=True,
    )

    got = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["day"][0]
    expected = pivot_levels(121.0, 99.0, 110.0)

    for name in ("pivot", "bc", "tc", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"):
        assert got["levels"][name] == pytest.approx(expected[name], abs=0.01), name


def test_the_prior_session_is_truncated_at_1515():
    """The operator's own rule: the closing auction can distort the day's close.

    A bar after 15:15 must not become the close the levels are built on.
    """

    early = _session("2026-09-14", [100.0] * 3)
    late = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-09-14 15:20")],
            "open": [500.0], "high": [500.0], "low": [500.0], "close": [500.0],
        }
    )
    frame = pd.concat([early, late, _session("2026-09-15", [200.0])], ignore_index=True)

    day = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["day"]

    assert day[0]["levels"]["prev_high"] == 101.0, "the 15:20 spike must be excluded"
    assert day[0]["levels"]["pivot"] == pytest.approx((101.0 + 99.0 + 100.0) / 3, abs=0.01)


def test_a_band_spans_only_its_own_session():
    """A band must stop at the end of its day, not run across the chart."""

    frame = pd.concat(
        [_session("2026-09-14", [100.0] * 3), _session("2026-09-15", [200.0] * 4)],
        ignore_index=True,
    )

    day = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["day"][0]

    assert day["from"] == int(pd.Timestamp("2026-09-15 09:15").timestamp())
    assert day["to"] == int(pd.Timestamp("2026-09-15 09:18").timestamp())


def test_months_take_their_levels_from_the_month_before():
    """On the Daily timeframe the ladder is MONTHLY, not daily."""

    frame = pd.concat(
        [
            _session("2026-08-10", [100.0, 150.0]),
            _session("2026-08-11", [120.0]),
            _session("2026-09-14", [300.0]),
            _session("2026-09-15", [310.0]),
        ],
        ignore_index=True,
    )

    months = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["month"]

    assert len(months) == 1, "only the second month can have a band"
    assert months[0]["month"] == "2026-09"
    # August across both sessions: high 151, low 99, last close 120
    assert months[0]["levels"]["prev_high"] == 151.0
    assert months[0]["levels"]["prev_low"] == 99.0
    assert months[0]["levels"]["pivot"] == pytest.approx((151.0 + 99.0 + 120.0) / 3, abs=0.01)


def test_the_month_is_truncated_at_1515_like_the_day_is():
    """A month's close is its last session's 15:15 close, not the auction's.

    Reading the month off the DAILY bars looks equivalent and is not: every
    daily bar is stamped 09:15, so the 09:15-15:15 filter matches all of them
    and truncates nothing. That shipped, and the caption said "sessions
    truncated at 15:15" while the numbers were the full session's -- August 2026
    carried a pivot of 24282.77 against the truncated 24249.25, high 70.4 out.
    """

    august = pd.concat(
        [_session("2026-08-10", [100.0] * 3), _session("2026-08-11", [110.0] * 3)],
        ignore_index=True,
    )
    auction = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-08-11 15:25")],
            "open": [500.0], "high": [500.0], "low": [500.0], "close": [500.0],
        }
    )
    frame = pd.concat(
        [august, auction, _session("2026-09-14", [300.0])], ignore_index=True
    )

    months = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["month"]

    assert months[0]["month"] == "2026-09"
    assert months[0]["levels"]["prev_high"] == 111.0, "the 15:25 print must not set the high"
    assert months[0]["levels"]["pivot"] == pytest.approx((111.0 + 99.0 + 110.0) / 3, abs=0.01)


def test_a_month_band_spans_the_daily_bars_not_the_minutes():
    """The levels come from minutes; the span is the timeframe it is drawn on."""

    frame = pd.concat(
        [_session("2026-08-10", [100.0] * 3), _session("2026-09-14", [300.0] * 3),
         _session("2026-09-15", [310.0] * 3)],
        ignore_index=True,
    )

    months = dashboard_history.cpr_segments(frame, dashboard_history.daily_bars(frame))["month"]

    # Daily bars are stamped at the session open, so the band runs open-to-open.
    assert months[0]["from"] == int(pd.Timestamp("2026-09-14 09:15").timestamp())
    assert months[0]["to"] == int(pd.Timestamp("2026-09-15 09:15").timestamp())


def test_the_cpr_ladders_survive_the_strict_renderer(tmp_path):
    """One NaN freezes the dashboard for the rest of the session."""

    frame = pd.concat(
        [_session("2026-09-14", [100.0, 110.0]), _session("2026-09-15", [200.0])],
        ignore_index=True,
    )
    target = _csv(tmp_path, frame)

    history = dashboard_history.build_history(
        path=target, renderer=_renderer, resample=_resample
    )

    ladders = json.loads(history.cpr)
    assert ladders["day"] and isinstance(ladders["month"], list)
    assert "cpr:0" in dashboard_history.page_map(history)


# ------------------------------------------------- indicator columns


def _fake_stochastic(frame, k_period, d_period, smooth_k):
    """A stand-in with the shape the strategies' own helper returns."""

    close = frame["close"]
    return close.rolling(2, min_periods=1).mean(), close.rolling(3, min_periods=1).mean()


def test_the_daily_series_carries_its_own_stochastic(tmp_path):
    """A minute stochastic says nothing about a daily candle.

    The Daily timeframe exists for the slower read, so its oscillator has to be
    computed on daily bars rather than borrowed from the live minute payload.
    """

    frame = pd.concat(
        [_minutes(day, 4) for day in ("2026-09-08", "2026-09-09", "2026-09-10")],
        ignore_index=True,
    )
    history = dashboard_history.build_history(
        path=_csv(tmp_path, frame), renderer=_renderer, resample=_resample,
        stochastic_fn=_fake_stochastic,
        stochastic_settings={"k_period": 2, "d_period": 2, "smooth_k": 1},
    )

    page = json.loads(history.series["D"].page(0))
    assert len(page["bars"]) == 3
    assert len(page["stoch_k"]) == 3, "the column is sliced beside its bars"
    assert len(page["stoch_d"]) == 3


def test_the_minute_series_carry_no_indicator_columns(tmp_path):
    """Deliberate: VWAP and a minute stochastic are about the session traded.

    Scrolled-back candles show the CPR bands and nothing else, which keeps the
    pages small and the statement honest.
    """

    frame = pd.concat([_minutes(d, 4) for d in ("2026-09-08", "2026-09-09")], ignore_index=True)
    history = dashboard_history.build_history(
        path=_csv(tmp_path, frame), renderer=_renderer, resample=_resample,
        stochastic_fn=_fake_stochastic,
    )

    page = json.loads(history.series["1"].page(0))
    assert "stoch_k" not in page
    assert "vwap" not in page


def test_a_broken_stochastic_costs_the_column_not_the_chart(tmp_path):
    """An indicator failure must never take the Daily timeframe down with it."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("no talib today")

    frame = pd.concat([_minutes(d, 4) for d in ("2026-09-08", "2026-09-09")], ignore_index=True)
    history = dashboard_history.build_history(
        path=_csv(tmp_path, frame), renderer=_renderer, resample=_resample,
        stochastic_fn=explode,
    )

    page = json.loads(history.series["D"].page(0))
    assert len(page["bars"]) == 2, "the daily candles still arrive"
    assert "stoch_k" not in page
