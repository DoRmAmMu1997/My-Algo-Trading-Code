"""MAT-104 regressions for queued next-bar-open entry rebasing."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from Dependencies.next_open_entry import PendingNextOpenEntry


def test_long_gap_up_preserves_setup_distances_and_valid_geometry() -> None:
    pending = PendingNextOpenEntry.from_setup(
        direction="LONG",
        signal_at=datetime(2026, 7, 16, 10, 0),
        entry=22_500.0,
        stop=22_480.0,
        target=22_540.0,
        timeframe_minutes=5,
    )

    rebased = pending.rebase_at_open(
        observed_at=datetime(2026, 7, 16, 10, 5),
        observed_entry=22_560.0,
    )

    assert rebased is not None
    assert rebased.entry == pytest.approx(22_560.0)
    assert rebased.stop == pytest.approx(22_540.0)
    assert rebased.target == pytest.approx(22_600.0)
    assert rebased.stop < rebased.entry < rebased.target


def test_short_gap_down_preserves_setup_distances_and_valid_geometry() -> None:
    pending = PendingNextOpenEntry.from_setup(
        direction="SHORT",
        signal_at=datetime(2026, 7, 16, 11, 0),
        entry=22_500.0,
        stop=22_530.0,
        target=22_450.0,
        timeframe_minutes=5,
    )

    rebased = pending.rebase_at_open(
        observed_at=datetime(2026, 7, 16, 11, 5),
        observed_entry=22_420.0,
    )

    assert rebased is not None
    assert rebased.entry == pytest.approx(22_420.0)
    assert rebased.stop == pytest.approx(22_450.0)
    assert rebased.target == pytest.approx(22_370.0)
    assert rebased.target < rebased.entry < rebased.stop


def test_intent_only_accepts_the_exact_expected_open_slot() -> None:
    pending = PendingNextOpenEntry.from_setup(
        direction="LONG",
        signal_at=datetime(2026, 7, 16, 10, 0),
        entry=100.0,
        stop=95.0,
        target=110.0,
        timeframe_minutes=5,
    )

    assert (
        pending.rebase_at_open(
            observed_at=datetime(2026, 7, 16, 10, 6),
            observed_entry=102.0,
        )
        is None
    )


def test_intent_expires_after_one_expected_bar() -> None:
    pending = PendingNextOpenEntry.from_setup(
        direction="LONG",
        signal_at=datetime(2026, 7, 16, 10, 0),
        entry=100.0,
        stop=95.0,
        target=110.0,
        timeframe_minutes=5,
    )

    assert pending.expected_open_at == datetime(2026, 7, 16, 10, 5)
    assert pending.expires_at == datetime(2026, 7, 16, 10, 10)
    assert pending.expired_as_of(datetime(2026, 7, 16, 10, 9, 59)) is False
    assert pending.expired_as_of(datetime(2026, 7, 16, 10, 10)) is True


@pytest.mark.parametrize(
    ("direction", "entry", "stop", "target"),
    [
        ("LONG", 100.0, 101.0, 110.0),
        ("LONG", 100.0, 95.0, 99.0),
        ("SHORT", 100.0, 99.0, 90.0),
        ("SHORT", 100.0, 105.0, 101.0),
    ],
)
def test_invalid_setup_geometry_is_rejected(
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> None:
    with pytest.raises(ValueError, match="geometry"):
        PendingNextOpenEntry.from_setup(
            direction=direction,
            signal_at=datetime(2026, 7, 16, 10, 0),
            entry=entry,
            stop=stop,
            target=target,
            timeframe_minutes=5,
        )


def test_nonfinite_observed_open_is_not_rebased() -> None:
    pending = PendingNextOpenEntry.from_setup(
        direction="LONG",
        signal_at=datetime(2026, 7, 16, 10, 0),
        entry=100.0,
        stop=95.0,
        target=110.0,
        timeframe_minutes=5,
    )

    assert (
        pending.rebase_at_open(
            observed_at=pending.expected_open_at,
            observed_entry=float("inf"),
        )
        is None
    )
    assert pending.expires_at - pending.expected_open_at == timedelta(minutes=5)
