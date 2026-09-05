"""Tests for the derived NIFTY weekly expiry calendar.

The API this feeds returns no expiry date, so the whole positional use case
rests on this derivation being right. These tests pin the two things that can
realistically go wrong: the holiday roll-back, and the 01-Sep-2025 change of
expiry weekday.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

# Tests/Data Extractors/<this file> -> the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = _REPO_ROOT / "Data Extractors" / "expiry_calendar.py"
spec = importlib.util.spec_from_file_location("expiry_calendar", MODULE_PATH)
calendar = importlib.util.module_from_spec(spec)
sys.modules["expiry_calendar"] = calendar
spec.loader.exec_module(calendar)


def weekdays(start: date, days: int, holidays: set[date] | None = None) -> list[date]:
    """Mon-Fri sessions from `start`, minus any holidays given."""
    skip = holidays or set()
    out = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        if day.weekday() < 5 and day not in skip:
            out.append(day)
    return out


def test_nominal_weekday_is_thursday_before_the_rule_change():
    assert calendar.nominal_expiry_weekday(date(2023, 6, 14)) == 3
    assert calendar.nominal_expiry_weekday(date(2025, 8, 31)) == 3


def test_nominal_weekday_is_tuesday_from_01_sep_2025():
    # NSE moved NIFTY weeklies off Thursday effective 01-Sep-2025.
    assert calendar.nominal_expiry_weekday(date(2025, 9, 1)) == 1
    assert calendar.nominal_expiry_weekday(date(2026, 3, 2)) == 1


def test_nominal_weekday_rejects_a_date_before_every_rule():
    with pytest.raises(calendar.ExpiryCalendarError, match="no expiry weekday rule"):
        calendar.nominal_expiry_weekday(date(1990, 1, 1), rules=[(date(2000, 1, 1), 3)])


def test_nominal_weekday_rejects_an_empty_rule_table():
    with pytest.raises(calendar.ExpiryCalendarError, match="rule table is empty"):
        calendar.nominal_expiry_weekday(date(2025, 1, 1), rules=[])


def test_expiries_land_on_thursdays_in_the_thursday_era():
    days = weekdays(date(2025, 1, 1), 60)
    expiries = calendar.weekly_expiry_dates(days)

    assert expiries, "expected at least one expiry"
    assert all(e.weekday() == 3 for e in expiries), [str(e) for e in expiries]


def test_a_holiday_on_the_nominal_day_rolls_the_expiry_back_one_session():
    # 2025-01-16 is a Thursday; knock it out and the expiry must move to the
    # Wednesday, not forward to the Friday.
    days = weekdays(date(2025, 1, 1), 60, holidays={date(2025, 1, 16)})
    expiries = calendar.weekly_expiry_dates(days)

    assert date(2025, 1, 15) in expiries
    assert date(2025, 1, 16) not in expiries
    assert date(2025, 1, 17) not in expiries


def test_a_run_of_holidays_rolls_back_across_the_week_boundary():
    # Wipe out Mon-Thu of one week entirely. The roll-back must keep walking
    # backwards into the previous week rather than give up or skip forward.
    dead = {date(2025, 1, 13), date(2025, 1, 14), date(2025, 1, 15), date(2025, 1, 16)}
    days = weekdays(date(2025, 1, 1), 60, holidays=dead)
    expiries = calendar.weekly_expiry_dates(days)

    assert date(2025, 1, 10) in expiries  # the previous Friday
    assert not (dead & set(expiries))


def test_the_thursday_to_tuesday_transition_is_reproduced_exactly():
    days = weekdays(date(2025, 8, 11), 42)
    expiries = calendar.weekly_expiry_dates(days)

    # Last Thursday expiry, then straight to Tuesdays.
    assert date(2025, 8, 28) in expiries
    assert date(2025, 9, 2) in expiries
    assert date(2025, 9, 9) in expiries
    before = [e for e in expiries if e < date(2025, 9, 1)]
    after = [e for e in expiries if e >= date(2025, 9, 1)]
    assert all(e.weekday() == 3 for e in before), [str(e) for e in before]
    assert all(e.weekday() == 1 for e in after), [str(e) for e in after]


def test_an_unfinished_week_gets_no_invented_expiry():
    # Data stops on Monday 2025-01-06. That week's Thursday has not happened,
    # so labelling those bars with an expiry would be a fabrication.
    days = weekdays(date(2025, 1, 1), 6)
    expiries = calendar.weekly_expiry_dates(days)

    assert date(2025, 1, 2) in expiries
    assert max(expiries) == date(2025, 1, 2)


def test_no_trading_days_yields_no_expiries():
    assert calendar.weekly_expiry_dates([]) == []
    assert calendar.build_expiry_map([]) == {}


def test_expiry_map_points_expiry_day_at_itself():
    days = weekdays(date(2025, 1, 1), 30)
    mapping = calendar.build_expiry_map(days)

    assert mapping[date(2025, 1, 9)] == date(2025, 1, 9)
    assert mapping[date(2025, 1, 6)] == date(2025, 1, 9)
    assert mapping[date(2025, 1, 10)] == date(2025, 1, 16)


def test_expiry_map_omits_days_past_the_last_derived_expiry():
    days = weekdays(date(2025, 1, 1), 10)
    mapping = calendar.build_expiry_map(days)
    last_expiry = max(calendar.weekly_expiry_dates(days))

    assert all(day <= last_expiry for day in mapping)
    assert any(day > last_expiry for day in days), "test needs a day past the last expiry"


def test_trading_days_to_expiry_counts_sessions_not_calendar_days():
    days = weekdays(date(2025, 1, 1), 30)

    # Thu 09-Jan expiry, seen from Fri 03-Jan: Mon/Tue/Wed/Thu = 4 sessions,
    # while the calendar gap is 6 days.
    assert calendar.trading_days_to_expiry(days, date(2025, 1, 3), date(2025, 1, 9)) == 4
    assert (date(2025, 1, 9) - date(2025, 1, 3)).days == 6
    # Expiry day itself has nothing left.
    assert calendar.trading_days_to_expiry(days, date(2025, 1, 9), date(2025, 1, 9)) == 0


def test_trading_days_to_expiry_rejects_an_expiry_in_the_past():
    days = weekdays(date(2025, 1, 1), 30)
    with pytest.raises(calendar.ExpiryCalendarError, match="precedes trade date"):
        calendar.trading_days_to_expiry(days, date(2025, 1, 9), date(2025, 1, 2))


def test_a_custom_rule_table_overrides_the_shipped_one():
    days = weekdays(date(2025, 1, 1), 30)
    fridays = calendar.weekly_expiry_dates(days, rules=[(date(2000, 1, 1), 4)])

    assert all(e.weekday() == 4 for e in fridays), [str(e) for e in fridays]


# ---------------------------------------------------------------------------
# Special (Muhurat) sessions
# ---------------------------------------------------------------------------


def test_a_muhurat_session_cannot_be_an_expiry():
    """The real 2025 Diwali week, which the first full backfill got wrong.

    Tuesday 21-Oct-2025 was Diwali: a ~1 hour Muhurat session, so NSE moved that
    week's expiry back to Monday the 20th. Treating it as an ordinary trading
    day put the expiry on the 21st and mislabelled the whole week.
    """
    week = [date(2025, 10, 16), date(2025, 10, 17), date(2025, 10, 20), date(2025, 10, 21)]
    full = [d for d in week if d != date(2025, 10, 21)]

    naive = calendar.weekly_expiry_dates(week)
    fixed = calendar.weekly_expiry_dates(week, full_sessions=full)

    assert date(2025, 10, 21) in naive, "this is the bug being pinned"
    assert date(2025, 10, 21) not in fixed
    assert date(2025, 10, 20) in fixed


def test_the_2021_diwali_week_lands_on_the_wednesday():
    # Thursday 04-Nov-2021 was Diwali; the expiry was Wednesday the 3rd.
    week = [date(2021, 11, 1), date(2021, 11, 2), date(2021, 11, 3), date(2021, 11, 4)]
    full = [d for d in week if d != date(2021, 11, 4)]

    assert date(2021, 11, 3) in calendar.weekly_expiry_dates(week, full_sessions=full)
    assert date(2021, 11, 4) not in calendar.weekly_expiry_dates(week, full_sessions=full)


def test_a_muhurat_session_is_still_a_trading_day_with_an_expiry_ahead():
    """It cannot HOST an expiry, but its bars still deserve a label."""
    days = [date(2025, 10, 16), date(2025, 10, 17), date(2025, 10, 20), date(2025, 10, 21),
            date(2025, 10, 23), date(2025, 10, 24), date(2025, 10, 27), date(2025, 10, 28)]
    full = [d for d in days if d != date(2025, 10, 21)]

    mapping = calendar.build_expiry_map(days, full_sessions=full)

    assert mapping[date(2025, 10, 20)] == date(2025, 10, 20), "Monday hosts the expiry"
    # The Muhurat session belongs to the NEXT weekly expiry, not the past one.
    assert mapping[date(2025, 10, 21)] == date(2025, 10, 28)


def test_full_sessions_defaults_to_every_trading_day():
    days = [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8), date(2025, 1, 9)]
    assert calendar.weekly_expiry_dates(days) == calendar.weekly_expiry_dates(days, full_sessions=days)


def test_no_eligible_sessions_yields_no_expiries():
    days = [date(2025, 1, 6), date(2025, 1, 9)]
    assert calendar.weekly_expiry_dates(days, full_sessions=[]) == []
