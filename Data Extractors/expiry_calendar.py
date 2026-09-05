"""
Derive NIFTY weekly option expiry dates from a list of trading days.

Why this file exists
--------------------
DhanHQ's expired-options endpoint (`/charts/rollingoption`) answers in *relative*
terms: you ask for "the ATM call of the near weekly expiry" and it hands back
minute bars. What it never tells you is **which expiry** those bars belong to.

For an intraday strategy that does not matter. For a positional option seller it
matters enormously: "sell an option five days before expiry and hold it to
settlement" is not a rule you can even express without knowing the expiry date.

So we derive it. Two ingredients, and only two:

1. **Which weekday NIFTY weeklies expire on.** This is a published exchange rule
   that has changed once inside our five-year window (see the rule table below).
2. **Which days the market was actually open.** We do not hard-code a holiday
   list -- we read the trading days straight out of the downloaded data, so the
   calendar can never disagree with the data it describes.

The rule is then simply: a week's expiry is the last trading day on or before
that week's nominal expiry weekday. If the nominal day was a holiday, the expiry
rolls *backwards* to the previous trading day, which is exactly what NSE does.

This module is deliberately pure -- no network, no files, no clock. Everything it
knows arrives as arguments, which is what makes it testable and what makes the
`--verify-expiries` cross-check in the fetcher meaningful.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

# Nominal expiry weekday, in `date.weekday()` terms: Monday=0 ... Sunday=6.
#
# Read this as "from this date onwards, NIFTY weeklies expire on this weekday".
# Entries must stay sorted by date; `nominal_expiry_weekday` takes the last rule
# whose start date has already arrived.
#
# History inside our 5-year window:
#   - Thursday, as it had been for ~25 years.
#   - Tuesday from 01-Sep-2025, when NSE moved index and stock derivatives off
#     Thursday following SEBI's October 2024 circular. (BSE went the other way,
#     to Thursday, which is why this table is NIFTY-specific.)
#
# NSE announced a switch to Monday in March 2025 and then deferred it; that one
# never took effect and deliberately has no entry here.
WEEKLY_EXPIRY_WEEKDAY_RULES: tuple[tuple[date, int], ...] = (
    (date(2000, 1, 1), 3),  # Thursday
    (date(2025, 9, 1), 1),  # Tuesday
)


class ExpiryCalendarError(ValueError):
    """Raised when a calendar cannot be derived from the trading days given."""


def nominal_expiry_weekday(day: date, rules: Sequence[tuple[date, int]] | None = None) -> int:
    """
    Return the weekday NIFTY weeklies nominally expire on, for a given date.

    "Nominal" means *before* the holiday roll-back. On a normal week the nominal
    day IS the expiry; on a week where that day is a holiday the real expiry is
    earlier, and `weekly_expiry_dates` works that out.
    """
    table = WEEKLY_EXPIRY_WEEKDAY_RULES if rules is None else tuple(rules)
    if not table:
        raise ExpiryCalendarError("expiry weekday rule table is empty")

    weekday = None
    for start, candidate in table:
        if day >= start:
            weekday = candidate
    if weekday is None:
        raise ExpiryCalendarError(
            f"no expiry weekday rule covers {day.isoformat()}; "
            f"earliest rule starts {table[0][0].isoformat()}"
        )
    return weekday


def weekly_expiry_dates(
    trading_days: Iterable[date],
    rules: Sequence[tuple[date, int]] | None = None,
) -> list[date]:
    """
    Work out every weekly expiry date covered by ``trading_days``.

    The algorithm, per calendar week:

    1. Take that week's Monday and ask the rule table for the nominal weekday.
    2. Step forward to the nominal date (e.g. Monday + 3 = Thursday).
    3. Walk *backwards* to the most recent trading day on or before it.

    Step 3 is what handles holidays, and it searches the whole trading-day list
    rather than just that week -- so an expiry whose nominal day AND the days
    before it were all holidays still rolls back correctly into the prior week.

    A week whose nominal expiry date falls after the last trading day we were
    given is skipped: that week has not finished yet, and inventing an expiry
    for it would silently mislabel the most recent bars.
    """
    days = sorted(set(trading_days))
    if not days:
        return []

    first, last = days[0], days[-1]
    expiries: list[date] = []

    # Start from the Monday of the first trading day's week and step a week at a
    # time. Using Mondays (rather than the trading days themselves) means a week
    # that is entirely holiday still gets considered.
    monday = first - timedelta(days=first.weekday())
    while monday <= last:
        nominal = monday + timedelta(days=nominal_expiry_weekday(monday, rules))

        # An unfinished week: no expiry has happened yet, so do not invent one.
        if nominal > last:
            break

        # bisect_right gives the count of trading days <= nominal, so index-1 is
        # the latest such day -- the holiday roll-back, for free.
        position = bisect.bisect_right(days, nominal)
        if position > 0:
            expiries.append(days[position - 1])

        monday += timedelta(days=7)

    # Two adjacent weeks can roll back onto the same trading day when a run of
    # holidays spans an expiry. Real contracts cannot share an expiry date, so
    # collapse the duplicate rather than emit an impossible calendar.
    return sorted(set(expiries))


def build_expiry_map(
    trading_days: Iterable[date],
    rules: Sequence[tuple[date, int]] | None = None,
) -> dict[date, date]:
    """
    Map every trading day to the near weekly expiry that was current on it.

    "Current on it" includes the expiry day itself: on expiry Tuesday the near
    weekly contract is the one expiring that afternoon, which is exactly what
    the API returns for ``expiryCode=1``. That makes expiry day map to itself
    and gives it ``days_to_expiry == 0``.

    Trading days after the last derived expiry are left out entirely rather than
    guessed at -- the caller writes no expiry label for those bars.
    """
    days = sorted(set(trading_days))
    expiries = weekly_expiry_dates(days, rules)
    if not expiries:
        return {}

    mapping: dict[date, date] = {}
    for day in days:
        position = bisect.bisect_left(expiries, day)
        if position < len(expiries):
            mapping[day] = expiries[position]
    return mapping


def trading_days_to_expiry(trading_days: Sequence[date], day: date, expiry: date) -> int:
    """
    Count the trading sessions left after ``day``, up to and including ``expiry``.

    Expiry day itself returns 0. This is the number a theta-aware rule wants:
    calendar days overstate the decay left across a long weekend.

    ``trading_days`` must be sorted ascending; the fetcher passes the same list
    the calendar was derived from.
    """
    if expiry < day:
        raise ExpiryCalendarError(
            f"expiry {expiry.isoformat()} precedes trade date {day.isoformat()}"
        )
    left = bisect.bisect_right(trading_days, day)
    right = bisect.bisect_right(trading_days, expiry)
    return max(right - left, 0)
