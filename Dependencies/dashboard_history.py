"""Years of 1-minute index history for the dashboard chart.

Why this is a module of its own rather than part of `dashboard_indicators.py`:
that module is PURE, and `test_the_module_reads_no_configuration` AST-asserts it
never imports `os` and reads no environment. Anything that opens a file has to
live somewhere else, and this is that somewhere.

What this module deliberately does NOT do is talk to the runner. It never
touches `SharedMarketDataStore`, a worker, the broker or the session state -- it
reads one CSV that `Data Extractors/` wrote and shapes it. That is what lets the
chart gain years of scroll-back without putting a single new call on the trading
path, and it is why the dashboard's safety contract still holds.

The shape of the problem is size. Five years of one-minute candles is ~465,000
bars, which is roughly 36 MB of JSON -- far past what the dashboard's transport
should hand over in one response, since it has no compression and writes one
body in one call. So history is PAGED: each timeframe is cut into fixed pages,
newest first, and the browser asks for the next older one as the operator
scrolls back.

Pages are rendered to bytes ONCE, on the dashboard's own daemon thread at
startup, and the DataFrame is dropped afterwards. Request threads then hand out
finished bytes and do no work at all, which is the same contract the rest of
this dashboard keeps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import pandas as pd

# The package name, matching how the master imports every other module here.
# NOT the bare name `mypy_path` also resolves: one file under two module names
# makes mypy refuse to check either. This module is type-checked through
# `mypy nifty_multi_strategy_master.py`, which follows and checks its imports --
# the same way `execution_ledger` and `startup_exposure` are covered.
from Dependencies.dashboard_indicators import (
    CPR_SESSION_END,
    CPR_SESSION_START,
    bar_records,
    pivot_levels,
    round_or_none,
    stochastic_values,
)

#: The trading session. Bars outside it are not session data: Dhan's older index
#: windows run to 17:59, carry rows dated Saturday and Sunday, and the current
#: day carries a synthetic wall-clock bar.
#:
#: The extractor clips all of these at download time, but the loader repeats the
#: check rather than trusting the file. A CSV written before the extractor learned
#: a rule is still on disk and still loads -- a resumed download can even leave one
#: file cleaned by two different rule versions, which is exactly what happened
#: here: 509 weekend rows survived in the chunks fetched before the weekday rule
#: existed.
HISTORY_SESSION_START: time = time(9, 15)
HISTORY_SESSION_END: time = time(15, 30)

#: Bars per page. ~150 KB of JSON at one minute, which is a comfortable single
#: response and about five sessions -- roughly one scroll-back gesture.
HISTORY_PAGE_BARS: int = 2000

#: The daily bar is stamped at the session OPEN so it lands on a real time in
#: the chart's scale rather than at midnight, where nothing else lives.
DAILY_BAR_CLOCK: time = time(9, 15)

_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")

#: Epoch for bar times. Integer-dividing a Timedelta is independent of the
#: frame's RESOLUTION -- pandas 3 defaults to microseconds, so the obvious
#: `astype("int64") // 10**9` is silently a thousand times too small and draws
#: the whole chart in 1970.
_EPOCH = pd.Timestamp("1970-01-01")

#: The levels each CPR segment carries, in the order the chart draws them.
CPR_LEVEL_KEYS = (
    "pivot", "bc", "tc", "prev_high", "prev_low",
    "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4",
)


def _epoch_seconds(value: pd.Timestamp) -> int:
    return int((value - _EPOCH) // pd.Timedelta(seconds=1))


class HistoryUnavailable(RuntimeError):
    """The history CSV is missing or unusable.

    Its own type so the caller can degrade to a live-only chart on purpose
    rather than by catching everything.
    """


def clip_to_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars that fall on a weekday, inside session hours.

    The weekday half is not redundant with the clock half: the stray weekend
    rows sit at times like 11:30, squarely inside session HOURS, so a
    time-of-day test alone passes them straight through.
    """

    if frame.empty:
        return frame
    clock = frame["timestamp"].dt.time
    weekday = frame["timestamp"].dt.dayofweek < 5
    inside = weekday & (clock >= HISTORY_SESSION_START) & (clock <= HISTORY_SESSION_END)
    return frame.loc[inside].reset_index(drop=True)


def load_history_frame(path: Path) -> pd.DataFrame:
    """Read the extractor's CSV into a clean, session-clipped frame.

    The explicit `format=` is the one place this repo's CSV reading deliberately
    diverges from the six `load_ohlc_data` variants that let pandas infer: over
    465,000 rows inference is the difference between a few seconds and a great
    many, and on this machine's disk that matters.
    """

    if not path.exists():
        raise HistoryUnavailable(f"no history CSV at {path}")

    try:
        frame = pd.read_csv(
            path,
            usecols=list(_REQUIRED_COLUMNS),
            parse_dates=["timestamp"],
            date_format="%Y-%m-%d %H:%M:%S",
        )
    except (OSError, ValueError, KeyError) as error:
        raise HistoryUnavailable(f"could not read {path}: {error}") from error

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise HistoryUnavailable(f"{path} is missing columns: {', '.join(missing)}")

    frame = frame.dropna(subset=list(_REQUIRED_COLUMNS))
    # lightweight-charts needs STRICTLY ascending times; one repeat and it drops
    # bars silently, which reads as a chart that is mysteriously half empty.
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    frame = clip_to_session(frame)
    if frame.empty:
        raise HistoryUnavailable(f"{path} holds no in-session bars")
    return frame


def daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse minute bars into one bar per trading session.

    Grouped by calendar date rather than resampled on a fixed frequency, so
    weekends and holidays simply do not appear instead of arriving as empty
    bars. This is the `_add_daily_cpr` idiom the CPR strategy already uses.
    """

    if frame.empty:
        return frame

    grouped = (
        frame.assign(session_date=frame["timestamp"].dt.date)
        .groupby("session_date", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .reset_index()
    )
    stamps = pd.to_datetime(grouped["session_date"].astype(str)) + pd.Timedelta(
        hours=DAILY_BAR_CLOCK.hour, minutes=DAILY_BAR_CLOCK.minute
    )
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": grouped["open"],
            "high": grouped["high"],
            "low": grouped["low"],
            "close": grouped["close"],
        }
    )


def page_bounds(total: int, page: int, page_size: int = HISTORY_PAGE_BARS) -> tuple[int, int]:
    """Row range for one page, counting from the NEWEST bar backwards.

    Page 0 is the newest slice because that is the order a browser wants them:
    the operator scrolls left and asks for 0, then 1, then 2. Numbering from the
    oldest end would make the first request depend on how much history exists.
    """

    if page < 0 or page_size <= 0:
        raise ValueError("page must be >= 0 and page_size > 0")
    stop = total - page * page_size
    if stop <= 0:
        raise IndexError(f"page {page} is past the start of {total} bars")
    return max(0, stop - page_size), stop


def page_count(total: int, page_size: int = HISTORY_PAGE_BARS) -> int:
    """How many pages a series of `total` bars is cut into."""

    if total <= 0:
        return 0
    return -(-total // page_size)


@dataclass(frozen=True)
class HistorySeries:
    """One timeframe's history, already cut into rendered pages."""

    key: str
    minutes: int
    pages: tuple[bytes, ...]
    bars: int
    first_bar: str
    last_bar: str

    def page(self, index: int) -> bytes | None:
        """The rendered page, or None when it is past the start of history."""

        if index < 0 or index >= len(self.pages):
            return None
        return self.pages[index]


@dataclass(frozen=True)
class DashboardHistory:
    """Every timeframe's paged history, plus what it came from."""

    series: Mapping[str, HistorySeries]
    #: The per-day and per-month CPR ladders, already rendered. Static for the
    #: life of the process, like the pages.
    cpr: bytes = b""
    source: str = ""
    available: bool = False
    unavailable_reason: str | None = None

    def summary(self) -> dict[str, object]:
        """What the live payload advertises so the browser knows what exists."""

        return {
            "available": bool(self.available),
            "source": self.source,
            "unavailable_reason": self.unavailable_reason,
            "timeframes": {
                key: {
                    "pages": len(item.pages),
                    "bars": item.bars,
                    "first_bar": item.first_bar,
                    "last_bar": item.last_bar,
                }
                for key, item in self.series.items()
            },
        }


def build_series(
    *,
    key: str,
    minutes: int,
    frame: pd.DataFrame,
    renderer: Callable[[Mapping[str, object]], bytes],
    page_size: int = HISTORY_PAGE_BARS,
    columns: Mapping[str, Sequence[float | None]] | None = None,
) -> HistorySeries:
    """Cut one timeframe into rendered pages, newest first.

    `columns` are indicator values already aligned to `frame`, sliced with the
    bars so a page's arrays and its candles cannot drift apart.
    """

    records: Sequence[Mapping[str, object]] = bar_records(frame)
    total = len(records)
    pages: list[bytes] = []
    for index in range(page_count(total, page_size)):
        start, stop = page_bounds(total, index, page_size)
        payload: dict[str, object] = {
            "tf": key,
            "minutes": int(minutes),
            "page": index,
            "pages": page_count(total, page_size),
            "bars": list(records[start:stop]),
        }
        for name, values in (columns or {}).items():
            payload[name] = list(values[start:stop])
        pages.append(renderer(payload))
    return HistorySeries(
        key=key,
        minutes=int(minutes),
        pages=tuple(pages),
        bars=total,
        first_bar=str(frame["timestamp"].iloc[0]) if total else "",
        last_bar=str(frame["timestamp"].iloc[-1]) if total else "",
    )


def _cpr_inputs(frame: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """High, low and close of each group's 09:15-15:15 window.

    The truncated window is the CHART-ONLY divergence the dashboard already
    makes deliberately, and it is the operator's own rule: the day's closing
    price is settled by the call auction and can be distorted, so the level that
    matters is where the market actually was at 15:15. `CPR_SESSION_START/END`
    are `dashboard_indicators`' own constants so the two cannot drift.
    """

    clock = frame["timestamp"].dt.time
    inside = (clock >= CPR_SESSION_START) & (clock <= CPR_SESSION_END)
    window = frame.loc[inside]
    if window.empty:
        return pd.DataFrame(columns=["high", "low", "close"])
    return (
        window.assign(_key=key.loc[window.index])
        .groupby("_key", sort=True)
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    )


def _segments(frame: pd.DataFrame, key: pd.Series, label: str) -> list[dict[str, object]]:
    """One CPR band per group, derived from the group BEFORE it.

    This is the `_add_daily_cpr` shape the CPR strategy already uses -- group,
    aggregate, then shift by one -- rather than a second way of saying it. The
    first group has no predecessor and so has no band at all, which is correct
    and is why it is skipped rather than given zeros.
    """

    stats = _cpr_inputs(frame, key)
    # Read once into plain floats, keyed by group. `stats.loc[group]` would do
    # the same work per row and hands back something pandas-stubs cannot narrow.
    inputs = {
        group: (float(row["high"]), float(row["low"]), float(row["close"]))
        for group, row in stats.iterrows()
    }
    spans = (
        frame.assign(_key=key)
        .groupby("_key", sort=True)["timestamp"]
        .agg(["min", "max"])
    )

    segments: list[dict[str, object]] = []
    previous = None
    for group, span in spans.iterrows():
        if previous is not None and previous in inputs:
            high, low, close = inputs[previous]
            levels = pivot_levels(high, low, close)
            levels["prev_high"] = high
            levels["prev_low"] = low
            rounded = {name: round_or_none(levels.get(name)) for name in CPR_LEVEL_KEYS}
            # `allow_nan=False` turns one NaN into a frozen dashboard for the
            # rest of the session, so a level that cannot be computed is dropped
            # rather than carried as a number that is not one.
            segments.append(
                {
                    label: str(group),
                    "from": _epoch_seconds(span["min"]),
                    "to": _epoch_seconds(span["max"]),
                    "levels": {k: v for k, v in rounded.items() if v is not None},
                }
            )
        previous = group
    return segments


def cpr_segments(frame: pd.DataFrame, days: pd.DataFrame) -> dict[str, object]:
    """Per-day and per-month CPR bands for the whole history.

    Two ladders, drawn on different timeframes. The daily one spans each
    session and comes from the session before it; the monthly one spans each
    month of DAILY bars and comes from the month before it, which is what makes
    the Daily timeframe usable for the next session's macro read.
    """

    return {
        "day": _segments(frame, frame["timestamp"].dt.date, "date"),
        "month": _segments(
            days, days["timestamp"].dt.to_period("M").astype(str), "month"
        ),
    }


def page_map(history: DashboardHistory) -> dict[str, bytes]:
    """Flatten the history into the `"<tf>:<page>"` keys the transport serves.

    One flat dict of finished bytes is all the publisher needs to hold, and a
    missing key IS the "there is nothing older" answer -- so the endpoint needs
    no bounds arithmetic of its own, and cannot disagree with this one.
    """

    pages = {
        f"{series.key}:{index}": payload
        for series in history.series.values()
        for index, payload in enumerate(series.pages)
    }
    if history.cpr:
        # The CPR ladders ride the same endpoint because they are the same kind
        # of thing: static chart data, rendered once, handed out as bytes.
        pages["cpr:0"] = history.cpr
    return pages


def build_history(
    *,
    path: Path,
    renderer: Callable[[Mapping[str, object]], bytes],
    resample: Callable[[pd.DataFrame, int], pd.DataFrame],
    higher_timeframe_minutes: int = 5,
    page_size: int = HISTORY_PAGE_BARS,
    stochastic_fn: Callable[..., tuple[pd.Series, pd.Series]] | None = None,
    stochastic_settings: Mapping[str, int] | None = None,
) -> DashboardHistory:
    """Load the CSV and render every timeframe's pages.

    `resample` is injected rather than imported so this uses the STRATEGIES' own
    1m->5m resampler, the same object the live chart uses. Two resamplers would
    eventually disagree about a bucket boundary, and the chart would show a seam
    exactly where history meets the live session.

    A missing or unusable CSV is not an error here: it returns an unavailable
    history and the chart carries on showing the live session alone.
    """

    try:
        frame = load_history_frame(path)
    except HistoryUnavailable as error:
        return DashboardHistory(series={}, unavailable_reason=str(error))

    series: dict[str, HistorySeries] = {
        "1": build_series(
            key="1", minutes=1, frame=frame, renderer=renderer, page_size=page_size
        )
    }

    higher = resample(frame, higher_timeframe_minutes)
    if not higher.empty:
        series[str(higher_timeframe_minutes)] = build_series(
            key=str(higher_timeframe_minutes),
            minutes=higher_timeframe_minutes,
            frame=higher,
            renderer=renderer,
            page_size=page_size,
        )

    days = daily_bars(frame)
    if not days.empty:
        # `minutes` is the bucket width and a session is not a fixed number of
        # them, so 0 means "not a minute timeframe" rather than a bad value.
        # The DAILY series carries its own stochastic. The live payload's is
        # computed on one-minute bars, which says nothing about a daily candle,
        # and the Daily timeframe exists precisely for the slower read.
        #
        # The minute timeframes deliberately do NOT get indicator columns here:
        # scrolled-back candles show the CPR bands and nothing else, because
        # VWAP and the stochastic are about the session being traded.
        columns: dict[str, Sequence[float | None]] = {}
        if stochastic_fn is not None:
            settings = dict(stochastic_settings or {})
            try:
                k_values, d_values = stochastic_values(
                    days,
                    stochastic_fn=stochastic_fn,
                    k_period=int(settings.get("k_period", 14)),
                    d_period=int(settings.get("d_period", 3)),
                    smooth_k=int(settings.get("smooth_k", 3)),
                )
                columns = {"stoch_k": k_values, "stoch_d": d_values}
            except Exception:  # noqa: BLE001 - an indicator costs its own line, never the chart
                columns = {}

        series["D"] = build_series(
            key="D", minutes=0, frame=days, renderer=renderer,
            page_size=page_size, columns=columns,
        )

    return DashboardHistory(
        series=series,
        cpr=renderer(cpr_segments(frame, days)),
        source=str(path),
        available=True,
    )
