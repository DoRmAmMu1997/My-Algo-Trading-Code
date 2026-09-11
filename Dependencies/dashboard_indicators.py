"""Chart indicator shaping for the read-only monitoring dashboard.

The dashboard's chart shows candles; this module turns the same 1-minute frame
into the things an operator needs to read them against: CPR levels, a session
VWAP, a stochastic oscillator, and a 5-minute view of all three.

Same three rules as `dashboard_snapshot.py`:

1. **Pure.** pandas is allowed; threads, sockets, disk, the clock and `.env`
   are not. Every function is a transformation of its arguments. The module
   must never call `_env_*` or `os.getenv` -- configuration is read in the
   master, the only place `check_env_config` audits against `env.example`.
2. **Never invent a number.** Anything that cannot be derived honestly comes
   back as `None`, and the page draws nothing there. `render_document_bytes`
   serializes with `allow_nan=False`, so a stray NaN would raise inside the
   renderer and freeze the whole dashboard on its last good document -- every
   value crossing this boundary goes through `finite_or_none` first.
3. **Reporting only.** Nothing here feeds risk, sizing or execution.

**Indicators are computed on COMPLETED bars.** The forming candle is drawn as a
candle but carries no indicator point, which is both cheaper and truer: the
strategies themselves evaluate closed candles only.

**The maths is not reimplemented here.** `stochastic` and `attach_session_vwap`
are injected by the caller as the very objects the live strategies use, so a
line on the chart cannot drift from the figure a strategy traded on. The one
exception is CPR, and it is a deliberate one -- see `chart_cpr`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, time

import pandas as pd

#: The prior session is read from this window rather than the whole day. The
#: operator's reason: an index close is settled by the closing auction, so the
#: last prints of the day are the least trustworthy input to a pivot. The bar
#: labelled 15:15 is INCLUDED -- "the 3:15 candle close" means that candle's
#: close. Deliberately a constant, not a knob: it is a charting convention, and
#: a `.env` key here would escape the config audit this module is excluded from.
CPR_SESSION_START: time = time(9, 15)
CPR_SESSION_END: time = time(15, 15)

#: Epoch for the bar timestamps. Integer-dividing a Timedelta is independent of
#: the frame's datetime RESOLUTION -- pandas 3 defaults to microseconds, so the
#: obvious `astype("int64") // 10**9` silently yields values a thousand times
#: too small and draws the whole session in 1970.
_EPOCH = pd.Timestamp("1970-01-01")

_OHLC = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def finite_or_none(value: object) -> float | None:
    """A real number, or None. Rejects NaN and infinity; never raises.

    The gate in front of `allow_nan=False`. NaN arrives routinely and
    legitimately -- an indicator's warm-up window, a session with too few bars
    -- so this is a normal path, not an error path.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def round_or_none(value: object, digits: int = 2) -> float | None:
    """`finite_or_none`, rounded for the wire."""

    number = finite_or_none(value)
    return None if number is None else round(number, digits)


def _series_values(series: pd.Series[float], digits: int = 2) -> list[float | None]:
    """A JSON-ready column: rounded floats with None wherever the value is not real."""

    return [round_or_none(value, digits) for value in series.tolist()]


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def bar_records(frame: pd.DataFrame) -> list[dict]:
    """Serialize an OHLC frame into the shape the charting library expects.

    Produces exactly what the master's `_bar_record` produces for one row, and
    a test asserts that equivalence bar for bar. It exists because the
    row-at-a-time version costs ~39 ms for 375 bars against a 1000 ms build
    budget, where this costs ~1.6 ms.

    The timestamps are naive IST wall-clock and are labelled UTC on purpose:
    lightweight-charts renders every timestamp as UTC and has no timezone
    setting, so this makes the axis read 09:15, 09:16 ... as the exchange clock
    does. Localizing to IST instead would draw the session starting at 03:45.
    """

    if frame.empty:
        return []
    seconds = (frame["timestamp"] - _EPOCH) // pd.Timedelta(seconds=1)
    return [
        {
            "time": int(stamp),
            "open": round(float(open_), 2),
            "high": round(float(high), 2),
            "low": round(float(low), 2),
            "close": round(float(close), 2),
        }
        for stamp, open_, high, low, close in zip(
            seconds, frame["open"], frame["high"], frame["low"], frame["close"], strict=True
        )
    ]


def forming_bucket(
    frame: pd.DataFrame, *, timeframe_minutes: int
) -> tuple[dict | None, pd.Timestamp | None]:
    """The in-progress higher-timeframe bar, aggregated from its finished minutes.

    `resample_ohlc_from_1m` deliberately DISCARDS incomplete buckets, because
    that is the guarantee the 5-minute strategies stand on. The chart still
    wants to show the bar as it forms, so it is built here instead and
    concatenated afterwards -- the resampler is never weakened to produce it.

    Returns `(bar record, bucket start)`, or `(None, None)` when the newest
    minute happens to complete a bucket exactly and nothing is forming.
    """

    if frame.empty or int(timeframe_minutes) <= 1:
        return None, None
    newest = pd.Timestamp(frame["timestamp"].iloc[-1])
    anchor = newest.floor(f"{int(timeframe_minutes)}min")
    partial = frame.loc[frame["timestamp"] >= anchor]
    if partial.empty or len(partial) >= int(timeframe_minutes):
        # A full bucket is the resampler's business, not ours.
        return None, None
    record = {
        "time": int((anchor - _EPOCH) // pd.Timedelta(seconds=1)),
        "open": round(float(partial["open"].iloc[0]), 2),
        "high": round(float(partial["high"].max()), 2),
        "low": round(float(partial["low"].min()), 2),
        "close": round(float(partial["close"].iloc[-1]), 2),
    }
    return record, anchor


# ---------------------------------------------------------------------------
# CPR
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorSession:
    """The slice of the previous trading day the chart's CPR is built from."""

    session_date: date | None
    frame: pd.DataFrame
    first_bar: str | None
    last_bar: str | None
    bars_used: int
    #: True when the slice does not reach back to the session open, so the
    #: high and low may understate the day. Shown, not hidden.
    partial: bool


@dataclass(frozen=True)
class ChartCpr:
    """Daily CPR levels for the chart. NOT the levels the strategies trade on."""

    available: bool
    prior_session_date: str | None = None
    window: str | None = None
    last_bar: str | None = None
    bars_used: int = 0
    partial: bool = False
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None
    pivot: float | None = None
    bc: float | None = None
    tc: float | None = None
    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    r4: float | None = None
    s1: float | None = None
    s2: float | None = None
    s3: float | None = None
    s4: float | None = None
    width: str | None = None
    #: The pivot the live strategies see, from the FULL prior session. Carried
    #: purely so the page can print both numbers side by side; nothing computes
    #: with it.
    strategies_pivot: float | None = None
    unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form; the page reads these key names directly."""

        return {
            "available": self.available,
            "chart_only": True,
            "prior_session_date": self.prior_session_date,
            "window": self.window,
            "last_bar": self.last_bar,
            "bars_used": self.bars_used,
            "partial": self.partial,
            "prev_high": self.prev_high,
            "prev_low": self.prev_low,
            "prev_close": self.prev_close,
            "pivot": self.pivot,
            "bc": self.bc,
            "tc": self.tc,
            "r1": self.r1,
            "r2": self.r2,
            "r3": self.r3,
            "r4": self.r4,
            "s1": self.s1,
            "s2": self.s2,
            "s3": self.s3,
            "s4": self.s4,
            "width": self.width,
            "strategies_pivot": self.strategies_pivot,
            "unavailable_reason": self.unavailable_reason,
            "caveat": (
                "Chart CPR reads the prior session 09:15-15:15 -- high, low AND close. "
                "The live CPR, CPR Algo 3 and CPR AI strategies use the full session and "
                "its last intraday close, so their levels differ slightly. These lines are "
                "for the eye only and move nothing."
            ),
        }


def prior_session_window(
    frame: pd.DataFrame,
    *,
    session_start: time = CPR_SESSION_START,
    session_end: time = CPR_SESSION_END,
) -> PriorSession:
    """Slice the previous AVAILABLE session down to its `start..end` window.

    "Previous available" rather than "yesterday" so a weekend or a holiday
    resolves to whatever traded last -- the same day `_add_daily_cpr`'s
    `groupby(...).shift(1)` picks, which is what keeps the two implementations
    talking about the same session.
    """

    empty = PriorSession(None, frame.iloc[0:0], None, None, 0, False)
    if frame.empty or "timestamp" not in frame.columns:
        return empty

    dates = frame["timestamp"].dt.date
    today = dates.iloc[-1]
    earlier = [value for value in dates.unique() if value < today]
    if not earlier:
        return empty

    prior_date = max(earlier)
    day = frame.loc[dates == prior_date]
    clock = day["timestamp"].dt.time
    window = day.loc[(clock >= session_start) & (clock <= session_end)]
    if window.empty:
        return PriorSession(prior_date, window, None, None, 0, False)

    stamps = window["timestamp"]
    return PriorSession(
        session_date=prior_date,
        frame=window,
        first_bar=stamps.iloc[0].strftime("%H:%M"),
        last_bar=stamps.iloc[-1].strftime("%H:%M"),
        bars_used=len(window),
        # Measured against the session START, not against the day's own first
        # bar: comparing the slice to the day it was sliced from can never
        # detect anything, because a truncated frame truncates both. A prior
        # session whose history does not reach 09:15 may understate the high
        # and low, so the page says so rather than quietly drawing it.
        partial=stamps.iloc[0].time() > session_start,
    )


def pivot_levels(high: float, low: float, close: float) -> dict[str, float]:
    """Standard CPR / floor-pivot algebra.

    Transcribed from `Signal Generators/CPR Strategy/cpr_strategy_logic.py`
    rather than re-derived, so the chart and the strategies can only ever
    differ in their INPUT WINDOW. A test feeds an untruncated prior session
    through both and asserts the levels match.
    """

    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * pivot - bc
    r1 = 2.0 * pivot - low
    r2 = pivot + high - low
    r3 = high + 2.0 * (pivot - low)
    s1 = 2.0 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2.0 * (high - pivot)
    return {
        "pivot": pivot,
        "bc": bc,
        "tc": tc,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r3 + high - low,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s3 - (high - low),
    }


def chart_cpr(
    frame: pd.DataFrame,
    *,
    session_start: time = CPR_SESSION_START,
    session_end: time = CPR_SESSION_END,
    width_classifier: Callable[[float, float, float], str] | None = None,
) -> ChartCpr:
    """CPR levels for the chart, from a prior session truncated at 15:15.

    Every failure is `available=False` with a stated reason and every level
    `None` -- never a zero, and never today's own bars, either of which would
    draw a confident line in the wrong place.
    """

    prior = prior_session_window(frame, session_start=session_start, session_end=session_end)
    if prior.session_date is None:
        return ChartCpr(False, unavailable_reason="no prior session in the loaded history")
    if prior.frame.empty:
        return ChartCpr(
            False,
            prior_session_date=prior.session_date.isoformat(),
            unavailable_reason=f"prior session has no bars between {session_start:%H:%M} and {session_end:%H:%M}",
        )

    high = finite_or_none(prior.frame["high"].max())
    low = finite_or_none(prior.frame["low"].min())
    close = finite_or_none(prior.frame["close"].iloc[-1])
    if high is None or low is None or close is None:
        return ChartCpr(
            False,
            prior_session_date=prior.session_date.isoformat(),
            unavailable_reason="prior session high, low or close is not a finite number",
        )

    levels = pivot_levels(high, low, close)

    # The same algebra on the UNTRUNCATED session: what the live strategies
    # see. Carried so the page can show both figures instead of merely warning
    # that they differ.
    whole_day = frame.loc[frame["timestamp"].dt.date == prior.session_date]
    strategies_pivot = None
    if not whole_day.empty:
        full_high = finite_or_none(whole_day["high"].max())
        full_low = finite_or_none(whole_day["low"].min())
        full_close = finite_or_none(whole_day["close"].iloc[-1])
        if None not in (full_high, full_low, full_close):
            strategies_pivot = round_or_none(
                pivot_levels(full_high, full_low, full_close)["pivot"]  # type: ignore[arg-type]
            )

    width = None
    if width_classifier is not None:
        try:
            width = str(width_classifier(high, low, close))
        except Exception:  # noqa: BLE001 - a label must never break the chart
            width = None

    return ChartCpr(
        available=True,
        prior_session_date=prior.session_date.isoformat(),
        window=f"{prior.first_bar}-{prior.last_bar}",
        last_bar=prior.last_bar,
        bars_used=prior.bars_used,
        partial=prior.partial,
        prev_high=round_or_none(high),
        prev_low=round_or_none(low),
        prev_close=round_or_none(close),
        width=width,
        strategies_pivot=strategies_pivot,
        # Spelled out rather than splatted: a `**dict` hides which level is
        # which from both the reader and the type checker.
        pivot=round_or_none(levels["pivot"]),
        bc=round_or_none(levels["bc"]),
        tc=round_or_none(levels["tc"]),
        r1=round_or_none(levels["r1"]),
        r2=round_or_none(levels["r2"]),
        r3=round_or_none(levels["r3"]),
        r4=round_or_none(levels["r4"]),
        s1=round_or_none(levels["s1"]),
        s2=round_or_none(levels["s2"]),
        s3=round_or_none(levels["s3"]),
        s4=round_or_none(levels["s4"]),
    )


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------
def session_vwap_values(
    frame: pd.DataFrame,
    *,
    attach_session_date: Callable[[pd.DataFrame], pd.DataFrame],
    attach_session_vwap: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[list[float | None], bool]:
    """Session VWAP for every bar, plus whether it is the equal-weight proxy.

    Both callables are injected rather than imported: they are the SAME objects
    the Regime Adaptive strategy uses, so the drawn line cannot drift from the
    traded one, and no fourth copy of the VWAP formula enters the repository.

    The index feed carries no volume, so in live running this is always the
    proxy -- the expanding mean of `(H+L+C)/3` since the session open. The page
    must say so; `is_proxy` is how it knows.
    """

    if frame.empty:
        return [], True
    enriched = attach_session_vwap(attach_session_date(frame))
    values = _series_values(enriched["vwap"])
    proxy_column = enriched.get("vwap_is_proxy")
    is_proxy = True if proxy_column is None else bool(proxy_column.any())
    return values, is_proxy


def stochastic_values(
    frame: pd.DataFrame,
    *,
    stochastic_fn: Callable[..., tuple[pd.Series[float], pd.Series[float]]],
    k_period: int,
    d_period: int,
    smooth_k: int,
) -> tuple[list[float | None], list[float | None]]:
    """%K and %D for every bar, from the strategies' own stochastic helper.

    A frame shorter than the warm-up simply yields `None` throughout -- that is
    the honest answer, and it keeps NaN away from the renderer.
    """

    if frame.empty:
        return [], []
    k_series, d_series = stochastic_fn(
        frame, k_period=k_period, d_period=d_period, smooth_k=smooth_k
    )
    return _series_values(k_series), _series_values(d_series)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def timeframe_block(
    *,
    minutes: int,
    bars: Sequence[Mapping[str, object]],
    vwap: Sequence[float | None],
    stoch_k: Sequence[float | None],
    stoch_d: Sequence[float | None],
    forming_from: int | None = None,
) -> dict[str, object]:
    """One timeframe's candles and indicator columns.

    Indicator columns are parallel ARRAYS zipped against `bars[i].time`, not
    `{time, value}` objects: a bare number is ~9 bytes against ~36, and there
    are three of them per timeframe.

    The arrays are right-aligned to the bars and padded with `None` at the
    front, so a caller that computed indicators over more history than it
    displays still lines up.
    """

    count = len(bars)

    def aligned(values: Sequence[float | None]) -> list[float | None]:
        if len(values) >= count:
            return list(values[len(values) - count :])
        return [None] * (count - len(values)) + list(values)

    block: dict[str, object] = {
        "minutes": int(minutes),
        "bars": list(bars),
        "vwap": aligned(vwap),
        "stoch_k": aligned(stoch_k),
        "stoch_d": aligned(stoch_d),
    }
    if forming_from is not None:
        block["forming_from"] = int(forming_from)
    return block


def chart_document(
    *,
    series_version: int,
    timeframes: Mapping[str, Mapping[str, object]],
    cpr: ChartCpr,
    vwap_is_proxy: bool,
    stochastic_settings: Mapping[str, int],
) -> dict[str, object]:
    """The whole `/api/chart` payload."""

    return {
        "series_version": int(series_version),
        "timeframes": {key: dict(value) for key, value in timeframes.items()},
        "cpr": cpr.as_dict(),
        "indicators": {
            "vwap": {
                "label": "VWAP*" if vwap_is_proxy else "VWAP",
                "is_proxy": bool(vwap_is_proxy),
                "note": (
                    "Equal-weight proxy: the index feed carries no volume, so this is the "
                    "running mean of (H+L+C)/3 since the session open."
                )
                if vwap_is_proxy
                else "Volume-weighted.",
            },
            "stochastic": {
                **{key: int(value) for key, value in stochastic_settings.items()},
                "overbought": 80,
                "oversold": 20,
                "note": "Computed on completed bars; the forming candle carries no indicator point.",
            },
        },
    }
