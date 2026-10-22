"""SLH-027: the NIFTY stop fires on a candle CLOSE, with an intrabar backstop (operator, 22 Oct).

Whipsaw kept taking the agent's stop on a wick and then running to the target. On
22 Oct a LONG at 22,679.20 with its stop at 22,656.00 was stopped when the 10:19 low
reached 22,653.85; that candle closed at 22,661.00, and NIFTY stood 88 points above
the entry 16 minutes later. These tests pin the pure rule the worker applies on
every poll:

* the STOP is judged only on the close of a completed candle -- a wick, or the live
  price, does not fire it;
* an intrabar BACKSTOP at a multiple of the stop's distance still exits at once;
* the TARGET still fires on a touch;
* both sides mirror each other, and a missing entry falls back to a plain stop.

The worker-side wiring (which candle counts, the switch, the fail-safe) is pinned in
the master suite's ``test_slh027_*`` tests.
"""

from __future__ import annotations

import pytest
from sl_hunting_executor import (
    DEFAULT_STOP_BACKSTOP_MULTIPLE,
    backstop_level,
    closing_stop_or_target_hit,
)

# 22 Oct trade 3, verbatim.
ENTRY, STOP, TARGET = 22679.20, 22656.00, 22811.00


def test_default_backstop_is_one_and_a_half_times_the_stop_distance():
    assert DEFAULT_STOP_BACKSTOP_MULTIPLE == 1.5
    assert backstop_level("LONG", ENTRY, STOP, 1.5) == pytest.approx(22644.40)
    # SHORT mirrors it: entry 22,632.65, stop 22,657.00 (24.35 points) -> 22,669.175.
    assert backstop_level("SHORT", 22632.65, 22657.00, 1.5) == pytest.approx(22669.175)


def test_backstop_never_sits_inside_the_stop():
    # A multiple below 1 would put the emergency exit BEFORE the stop; it is floored at 1.
    assert backstop_level("LONG", ENTRY, STOP, 0.5) == STOP


def test_without_an_entry_the_backstop_is_the_stop_itself():
    assert backstop_level("LONG", 0.0, STOP, 1.5) == STOP
    assert closing_stop_or_target_hit("LONG", 0.0, STOP, TARGET, 22655.0, None) == ("AI_STOP", "backstop")


def test_22_oct_trade_3_survives_the_wick_on_a_closing_basis():
    # The live price at the 10:19 low, with the 10:18 candle (closed 22,661.05) as the last close.
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22653.85, 22661.05) is None
    # And once the 10:19 candle completes at 22,661.00 it is still open.
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22661.00, 22661.00) is None


def test_a_close_at_or_beyond_the_stop_exits():
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22662.0, 22656.00) == ("AI_STOP", "close")
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22662.0, 22650.00) == ("AI_STOP", "close")


def test_no_completed_candle_yet_leaves_only_the_backstop():
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22645.0, None) is None
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22644.40, None) == ("AI_STOP", "backstop")


def test_the_target_still_fires_on_a_touch():
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22811.0, 22700.0) == ("AI_TARGET", "target")
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22810.9, 22700.0) is None


def test_short_mirrors_long():
    entry, stop, target = 22632.65, 22657.00, 22555.00
    assert closing_stop_or_target_hit("SHORT", entry, stop, target, 22660.0, 22650.0) is None  # wick only
    assert closing_stop_or_target_hit("SHORT", entry, stop, target, 22650.0, 22657.0) == ("AI_STOP", "close")
    assert closing_stop_or_target_hit("SHORT", entry, stop, target, 22669.18, None) == ("AI_STOP", "backstop")
    assert closing_stop_or_target_hit("SHORT", entry, stop, target, 22555.0, 22600.0) == ("AI_TARGET", "target")


def test_a_stop_is_checked_before_a_target():
    # A close beyond the stop wins even if the live price is somehow at the target.
    assert closing_stop_or_target_hit("LONG", ENTRY, STOP, TARGET, 22811.0, 22650.0) == ("AI_STOP", "close")


def test_unset_levels_and_unknown_directions_never_fire():
    assert closing_stop_or_target_hit("LONG", ENTRY, 0.0, 0.0, 1.0, 1.0) is None
    assert closing_stop_or_target_hit("FLAT", ENTRY, STOP, TARGET, 1.0, 1.0) is None
