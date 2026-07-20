"""Fail-closed validation and freshness state for live market data.

The trading runner has many strategy threads, but they all consume the same
one-minute candles and LTP cache.  This module keeps the safety rules in one
place so a malformed candle, a stale quote, or a briefly recovered connection
cannot be interpreted differently by different strategies.

Naive timestamps are treated as Asia/Kolkata wall-clock values because that is
the repository's longstanding DataFrame convention.  Aware timestamps are
converted to Asia/Kolkata and the returned DataFrame is made naive again for
backward compatibility with the signal generators.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
OHLC_COLUMNS = ("open", "high", "low", "close")


class MarketDataValidationError(ValueError):
    """Raised when a snapshot cannot safely be used for trading decisions."""


def _as_aware_ist(value: Any) -> datetime:
    """Parse one timestamp and return an aware Asia/Kolkata datetime."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise MarketDataValidationError("timestamp contains an invalid value")
    timestamp = (
        timestamp.tz_localize(IST)
        if timestamp.tzinfo is None
        else timestamp.tz_convert(IST)
    )
    return timestamp.to_pydatetime()


def validate_ohlc_frame(
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
    future_tolerance_seconds: float = 5.0,
) -> pd.DataFrame:
    """Return a validated copy or raise before unsafe OHLC reaches workers.

    Identical duplicate candles are harmless transport duplication and are
    collapsed.  Two rows for the same minute with different prices are
    ambiguous, so the entire snapshot is rejected rather than choosing one.
    Input order is also enforced: silently sorting a corrupted response can
    conceal broker replay or mixed-chunk data.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketDataValidationError("OHLC snapshot is empty")

    required = ("timestamp", *OHLC_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MarketDataValidationError(
            f"OHLC snapshot is missing required columns: {', '.join(missing)}"
        )

    normalized = frame.copy()
    aware_times = [_as_aware_ist(value) for value in normalized["timestamp"]]
    if any(timestamp.second != 0 or timestamp.microsecond != 0 for timestamp in aware_times):
        raise MarketDataValidationError("OHLC timestamps are not aligned to exact minutes")
    if any(later < earlier for earlier, later in pairwise(aware_times)):
        raise MarketDataValidationError("OHLC timestamps are not ordered")

    now_ist = _as_aware_ist(now or datetime.now(IST))
    future_limit = now_ist + timedelta(seconds=max(0.0, future_tolerance_seconds))
    if any(timestamp > future_limit for timestamp in aware_times):
        raise MarketDataValidationError("OHLC snapshot contains a future timestamp")

    normalized["timestamp"] = [
        pd.Timestamp(timestamp).tz_localize(None) for timestamp in aware_times
    ]
    for column in OHLC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        values = normalized[column]
        if values.isna().any() or not values.map(math.isfinite).all():
            raise MarketDataValidationError(f"OHLC column {column!r} contains a non-finite value")
        if (values <= 0).any():
            raise MarketDataValidationError(f"OHLC column {column!r} contains a non-positive value")

    duplicate_rows = normalized[normalized.duplicated("timestamp", keep=False)]
    for timestamp, group in duplicate_rows.groupby("timestamp", sort=False):
        first = group.iloc[0]
        if any(
            not math.isclose(
                float(value),
                float(first[column]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for column in OHLC_COLUMNS
            for value in group[column].iloc[1:]
        ):
            raise MarketDataValidationError(
                f"conflicting duplicate OHLC candle at {timestamp}"
            )
    normalized = normalized.drop_duplicates("timestamp", keep="first").reset_index(drop=True)

    open_ = normalized["open"]
    high = normalized["high"]
    low = normalized["low"]
    close = normalized["close"]
    invalid_geometry = (
        (low > open_)
        | (low > close)
        | (high < open_)
        | (high < close)
        | (low > high)
    )
    if invalid_geometry.any():
        raise MarketDataValidationError("OHLC snapshot contains impossible candle geometry")

    return normalized


def complete_minute_bucket_mask(
    timestamps: pd.DatetimeIndex,
    timeframe_minutes: int,
) -> pd.Series:
    """Mark rows whose resample bucket contains every exact minute once.

    A mere row count is insufficient: ``09:15, 09:15, 09:17, 09:18,
    09:19`` has five rows but is missing the 09:16 candle.  This helper makes
    that bucket incomplete and therefore ineligible for strategy evaluation.
    """

    minutes = int(timeframe_minutes)
    if minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")

    index = pd.DatetimeIndex(timestamps)
    valid = [False] * len(index)
    if index.empty:
        return pd.Series(valid, dtype=bool)

    bucket_keys = index.floor(f"{minutes}min")
    positions_by_bucket: dict[pd.Timestamp, list[int]] = {}
    for position, bucket in enumerate(bucket_keys):
        positions_by_bucket.setdefault(pd.Timestamp(bucket), []).append(position)

    for bucket, positions in positions_by_bucket.items():
        actual = [pd.Timestamp(index[position]) for position in positions]
        expected = list(pd.date_range(bucket, periods=minutes, freq="1min"))
        bucket_is_complete = len(actual) == minutes and actual == expected
        for position in positions:
            valid[position] = bucket_is_complete
    return pd.Series(valid, dtype=bool)


def newest_completed_minute_timestamp(
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Return the newest candle whose one-minute interval has completed.

    Candles are stamped with their START minute, so the row stamped with the
    CURRENT minute is still forming and is deliberately excluded -- freshness
    is judged only on candles the market has finished printing.
    """

    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return None
    now_ist = _as_aware_ist(now or datetime.now(IST))
    current_minute = now_ist.replace(second=0, microsecond=0)
    completed = [
        timestamp
        for value in frame["timestamp"]
        if (timestamp := _as_aware_ist(value)) < current_minute
    ]
    return max(completed, default=None)


@dataclass(frozen=True)
class MarketDataHealthSnapshot:
    """Immutable worker-facing view of the current feed safety state.

    - ``monitoring``            : False outside a live producer session; every
                                  gate then passes so offline tools/backtests
                                  are unaffected.
    - ``entry_allowed``         : True only when the feed is currently healthy
                                  AND has been healthy for the required streak
                                  of producer refreshes (recovery hysteresis).
    - ``liquidation_required``  : True once the feed has been continuously
                                  unhealthy past the liquidation deadline --
                                  live workers must start flattening.
    - ``healthy_streak``        : consecutive healthy producer refreshes.
    - ``unhealthy_seconds``     : how long the current unhealthy episode has
                                  lasted (0 while healthy).
    - ``reasons``               : operator-readable explanations, empty when
                                  healthy.
    """

    monitoring: bool
    entry_allowed: bool
    liquidation_required: bool
    healthy_streak: int
    unhealthy_seconds: float
    reasons: tuple[str, ...]


class MarketDataHealth:
    """Track freshness, recovery hysteresis, and the liquidation deadline.

    Beginner's map of the timing rules (all configurable via ``__init__``):

    - An LTP older than **10s** or a newest completed 1-minute bar older than
      **90s** marks the whole feed unhealthy.  90s (not 60s) for the bar
      because a candle stamped 09:20 only COMPLETES at 09:21, and the producer
      then needs a poll cycle to fetch it -- 90s is "one full candle plus
      slack", while 60s would false-alarm every minute boundary.
    - Unhealthy for **30s** continuously -> ``liquidation_required``: a blip
      should not dump positions, but half a minute without prices means live
      stops/targets are flying blind, so tracked exposure must come off.
    - Recovery needs **3 consecutive healthy refreshes** before entries
      resume.  A single good poll right after an outage is often the first
      gasp of a flapping connection; the streak requirement stops the gate
      from oscillating open/closed ("hysteresis").

    Two entry points feed the same state: the producer calls
    ``record_refresh`` once per fetch cycle (the only place the healthy streak
    can ADVANCE), while workers call ``snapshot`` on their own schedule (which
    re-evaluates ages so a silently WEDGED producer still goes stale on time
    -- see the ``snapshot`` docstring).
    """

    def __init__(
        self,
        *,
        ltp_max_age_seconds: float = 10.0,
        completed_bar_max_age_seconds: float = 90.0,
        liquidation_after_seconds: float = 30.0,
        recovery_refreshes: int = 3,
    ) -> None:
        self._lock = threading.Lock()
        self._ltp_max_age = float(ltp_max_age_seconds)
        self._bar_max_age = float(completed_bar_max_age_seconds)
        self._liquidation_after = float(liquidation_after_seconds)
        self._recovery_refreshes = max(1, int(recovery_refreshes))
        self._monitoring = False
        self._refresh_ok = False
        self._newest_completed_bar: datetime | None = None
        self._ltp_fetched_at: dict[tuple[str, int], datetime] = {}
        self._required_ltp_keys: set[tuple[str, int]] = set()
        self._healthy_streak = 0
        self._unhealthy_since: datetime | None = None

    def begin_monitoring(self, *, now: datetime | None = None) -> None:
        """Enable fail-closed live checks; offline consumers remain opt-out."""

        del now  # Kept in the API to make deterministic callers self-documenting.
        with self._lock:
            self._monitoring = True
            self._refresh_ok = False
            self._healthy_streak = 0
            self._unhealthy_since = None

    def stop_monitoring(self) -> None:
        """Disable live freshness gates after the producer has stopped."""

        with self._lock:
            self._monitoring = False

    def record_refresh(
        self,
        *,
        ohlc_ok: bool,
        newest_completed_bar: object | None,
        ltp_fetched_at: Mapping[tuple[str, int], object],
        required_ltp_keys: Iterable[tuple[str, int]],
        now: datetime | None = None,
    ) -> MarketDataHealthSnapshot:
        """Publish one producer-cycle result and advance recovery hysteresis."""

        now_ist = _as_aware_ist(now or datetime.now(IST))
        with self._lock:
            self._refresh_ok = bool(ohlc_ok)
            self._newest_completed_bar = (
                _as_aware_ist(newest_completed_bar)
                if newest_completed_bar is not None
                else None
            )
            self._ltp_fetched_at = {
                (str(segment), int(security_id)): _as_aware_ist(fetched_at)
                for (segment, security_id), fetched_at in ltp_fetched_at.items()
            }
            self._required_ltp_keys = {
                (str(segment), int(security_id))
                for segment, security_id in required_ltp_keys
            }
            reasons = self._reasons_locked(now_ist)
            if reasons:
                # Any problem resets the recovery streak to zero and pins the
                # START of the unhealthy episode (kept, not overwritten, so the
                # 30s liquidation clock measures the WHOLE outage).
                self._healthy_streak = 0
                if self._unhealthy_since is None:
                    self._unhealthy_since = now_ist
            else:
                # Only a producer refresh may advance the streak: three healthy
                # WORKER polls in the same second prove nothing new about the
                # feed, three healthy PRODUCER cycles do.
                self._healthy_streak += 1
                self._unhealthy_since = None
            return self._snapshot_locked(now_ist, reasons)

    def snapshot(self, *, now: datetime | None = None) -> MarketDataHealthSnapshot:
        """Evaluate current ages so a silent producer becomes stale on time.

        This deliberately re-runs the age checks against ``now`` instead of
        replaying the producer's last verdict: if the fetcher thread wedges and
        never calls ``record_refresh`` again, its final (healthy) result would
        otherwise stay "fresh" forever.  Recomputing here means the stored
        timestamps age naturally and the gates fail closed on schedule even
        when the producer has gone completely quiet.
        """

        now_ist = _as_aware_ist(now or datetime.now(IST))
        with self._lock:
            if not self._monitoring:
                # No live producer session: every gate passes so backtests and
                # offline diagnostics are unaffected by freshness rules.
                return MarketDataHealthSnapshot(
                    monitoring=False,
                    entry_allowed=True,
                    liquidation_required=False,
                    healthy_streak=self._healthy_streak,
                    unhealthy_seconds=0.0,
                    reasons=(),
                )
            reasons = self._reasons_locked(now_ist)
            if reasons:
                # Same episode bookkeeping as record_refresh -- but note the
                # healthy branch does NOT advance the streak here: only the
                # producer's own refreshes count as recovery evidence.
                self._healthy_streak = 0
                if self._unhealthy_since is None:
                    self._unhealthy_since = now_ist
            else:
                self._unhealthy_since = None
            return self._snapshot_locked(now_ist, reasons)

    def _reasons_locked(self, now: datetime) -> tuple[str, ...]:
        """Collect every current problem; an empty tuple means healthy."""
        reasons: list[str] = []
        if not self._refresh_ok:
            reasons.append("latest OHLC refresh failed")
        if self._newest_completed_bar is None:
            reasons.append("newest completed one-minute bar is unavailable")
        else:
            bar_age = (now - self._newest_completed_bar).total_seconds()
            # A slightly future timestamp is normal clock skew; more than 5s
            # ahead means the feed's clock (or an epoch-unit bug) cannot be
            # trusted, which is just as unsafe as stale data.
            if bar_age < -5.0:
                reasons.append("newest completed one-minute bar is in the future")
            elif bar_age > self._bar_max_age:
                reasons.append(f"newest completed one-minute bar is stale ({bar_age:.1f}s)")

        for key in sorted(self._required_ltp_keys):
            fetched_at = self._ltp_fetched_at.get(key)
            if fetched_at is None:
                reasons.append(f"LTP {key[0]}/{key[1]} is unavailable")
                continue
            age = (now - fetched_at).total_seconds()
            if age < -5.0:
                reasons.append(f"LTP {key[0]}/{key[1]} is future-dated")
            elif age > self._ltp_max_age:
                reasons.append(f"LTP {key[0]}/{key[1]} is stale ({age:.1f}s)")
        return tuple(reasons)

    def _snapshot_locked(
        self,
        now: datetime,
        reasons: tuple[str, ...],
    ) -> MarketDataHealthSnapshot:
        """Freeze the two gate decisions into one immutable, coherent view."""
        unhealthy_seconds = (
            max(0.0, (now - self._unhealthy_since).total_seconds())
            if self._unhealthy_since is not None
            else 0.0
        )
        healthy = not reasons
        return MarketDataHealthSnapshot(
            monitoring=self._monitoring,
            # Entries need BOTH currently-healthy and a full recovery streak;
            # liquidation needs BOTH currently-unhealthy and a full 30s episode.
            # The asymmetry is deliberate: reopening entries is the risky
            # direction, so it carries the extra hysteresis requirement.
            entry_allowed=healthy and self._healthy_streak >= self._recovery_refreshes,
            liquidation_required=bool(
                reasons and unhealthy_seconds >= self._liquidation_after
            ),
            healthy_streak=self._healthy_streak,
            unhealthy_seconds=unhealthy_seconds,
            reasons=reasons,
        )
