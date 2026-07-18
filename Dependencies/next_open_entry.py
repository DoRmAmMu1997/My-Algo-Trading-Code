"""One-bar lifetime and price rebasing for ``NEXT_OPEN`` strategy signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


def _finite_price(value: Any) -> float | None:
    """Return a finite positive price or ``None``."""

    if isinstance(value, bool):
        return None
    try:
        price = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


@dataclass(frozen=True, slots=True)
class RebasedNextOpenEntry:
    """Observed entry plus stop/target shifted by the setup distances."""

    direction: str
    observed_at: datetime
    entry: float
    stop: float
    target: float


@dataclass(frozen=True, slots=True)
class PendingNextOpenEntry:
    """A setup that may execute only at the immediately following bar open."""

    direction: str
    signal_at: datetime
    expected_open_at: datetime
    expires_at: datetime
    setup_entry: float
    stop_distance: float
    target_distance: float

    @classmethod
    def from_setup(
        cls,
        *,
        direction: str,
        signal_at: datetime,
        entry: float,
        stop: float,
        target: float,
        timeframe_minutes: int,
    ) -> PendingNextOpenEntry:
        """Validate a completed setup and schedule exactly its next bar."""

        normalized_direction = str(direction).upper().strip()
        if normalized_direction not in {"LONG", "SHORT"}:
            raise ValueError("direction must be LONG or SHORT")
        if not isinstance(signal_at, datetime):
            raise ValueError("signal_at must be a datetime")
        if type(timeframe_minutes) is not int or timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be a positive integer")

        setup_entry = _finite_price(entry)
        setup_stop = _finite_price(stop)
        setup_target = _finite_price(target)
        if None in {setup_entry, setup_stop, setup_target}:
            raise ValueError("entry, stop, and target must be finite positive prices")
        assert setup_entry is not None
        assert setup_stop is not None
        assert setup_target is not None

        if normalized_direction == "LONG":
            valid_geometry = setup_stop < setup_entry < setup_target
            stop_distance = setup_entry - setup_stop
            target_distance = setup_target - setup_entry
        else:
            valid_geometry = setup_target < setup_entry < setup_stop
            stop_distance = setup_stop - setup_entry
            target_distance = setup_entry - setup_target
        if not valid_geometry:
            raise ValueError("setup entry/stop/target geometry is invalid")

        bar = timedelta(minutes=timeframe_minutes)
        expected_open_at = signal_at + bar
        return cls(
            direction=normalized_direction,
            signal_at=signal_at,
            expected_open_at=expected_open_at,
            expires_at=expected_open_at + bar,
            setup_entry=setup_entry,
            stop_distance=stop_distance,
            target_distance=target_distance,
        )

    def expired_as_of(self, observed_at: datetime) -> bool:
        """Return whether the one permitted execution bar has ended."""

        if not isinstance(observed_at, datetime):
            raise ValueError("observed_at must be a datetime")
        return observed_at >= self.expires_at

    def rebase_at_open(
        self,
        *,
        observed_at: datetime,
        observed_entry: float,
    ) -> RebasedNextOpenEntry | None:
        """Shift stop/target distances onto the exact observed next-bar open."""

        if not isinstance(observed_at, datetime):
            return None
        if observed_at != self.expected_open_at:
            return None
        entry = _finite_price(observed_entry)
        if entry is None:
            return None

        if self.direction == "LONG":
            stop = entry - self.stop_distance
            target = entry + self.target_distance
            valid_geometry = stop < entry < target
        else:
            stop = entry + self.stop_distance
            target = entry - self.target_distance
            valid_geometry = target < entry < stop
        if not valid_geometry or not all(
            math.isfinite(price) and price > 0.0
            for price in (entry, stop, target)
        ):
            return None
        return RebasedNextOpenEntry(
            direction=self.direction,
            observed_at=observed_at,
            entry=entry,
            stop=stop,
            target=target,
        )
