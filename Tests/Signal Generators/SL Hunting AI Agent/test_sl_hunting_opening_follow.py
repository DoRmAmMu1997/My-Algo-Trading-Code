"""v6c: the OPENING FOLLOW window is computed in code (operator, 29 Oct).

When yesterday closed in the outer third of its range but made that extreme by 13:30
and faded at least 0.15% from it, its crowd booked and left, so a FLAT open is
followed in yesterday's direction -- but only in the session's first two decisions.
The agent decides as each new minute appears, so its first two calls see a frame
ending at the 09:15 and then the 09:16 candle; from the 09:17 candle on the window
reads expired. These tests pin that contract:

* active only for a stalled prior session, a FLAT open, and a frame that started at
  09:15 and still ends by 09:16 -- in both directions;
* every reason it is not active, named;
* the settlement prints after 15:15 never count as yesterday's extreme;
* the kill switch, and BankNIFTY's levels leaving the block out.
"""

from __future__ import annotations

import pandas as pd
from sl_hunting_executor import StandaloneExecutor
from sl_hunting_indicators import SLHuntingIndicatorConfig, pivot_and_levels
from sl_hunting_tools import SLHuntingToolContext


def _session(day, path):
    """One session from (HH:MM, price) waypoints, filled minute by minute 09:15-15:29."""
    stamps = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min")
    marks = pd.Series({pd.Timestamp(f"{day} {hm}"): price for hm, price in path})
    prices = marks.reindex(stamps.union(marks.index)).interpolate().reindex(stamps).bfill().ffill()
    return pd.DataFrame({"timestamp": stamps, "open": prices.values, "high": prices.values + 1.0,
                         "low": prices.values - 1.0, "close": prices.values})


# 28 Oct's shape: sold to its low at 10:22, rallied to its high at 13:00, faded into
# a 3:15 level in the top quarter of its range (0.31% below the high).
UP_STALLED = [("09:15", 22678.0), ("10:22", 22498.0), ("13:00", 22777.0), ("15:14", 22707.0),
              ("15:15", 22707.0), ("15:29", 22707.0)]
# The mirror: rallied early, sold to its low at 12:30, bounced into the bottom quarter.
DOWN_STALLED = [("09:15", 22700.0), ("10:00", 22880.0), ("12:30", 22600.0), ("15:14", 22668.0),
                ("15:15", 22668.0), ("15:29", 22668.0)]


def _today(open_price, minutes):
    """Today's first `minutes` candles, opening at `open_price`, from 09:15."""
    stamps = pd.date_range("2026-10-29 09:15", periods=minutes, freq="1min")
    return pd.DataFrame({"timestamp": stamps, "open": open_price, "high": open_price + 3.0,
                         "low": open_price - 3.0, "close": open_price})


def _frame(prev_path, open_price, minutes):
    return pd.concat([_session("2026-10-28", prev_path), _today(open_price, minutes)], ignore_index=True)


def _follow(prev_path, open_price, minutes, cfg=None):
    return pivot_and_levels(_frame(prev_path, open_price, minutes), cfg)["opening_follow"]


def test_active_long_in_the_first_two_decisions_and_expired_from_the_third():
    for minutes in (1, 2):
        block = _follow(UP_STALLED, 22691.0, minutes)
        assert block["active"] is True
        assert block["direction"] == "LONG"
        assert block["reason"] is None
        assert block["prior_session"]["shape"] == "UP_STALLED"
        assert block["prior_session"]["extreme_time"] == "13:00"
        assert block["stop_points"] == 16.0
        assert block["target_points"] == 45.0
    expired = _follow(UP_STALLED, 22691.0, 3)
    assert expired["active"] is False
    assert expired["direction"] is None
    assert expired["reason"] == "expired: the window covers the session's first two decisions only"


def test_the_mirror_is_a_short():
    block = _follow(DOWN_STALLED, 22660.0, 1)
    assert block["active"] is True
    assert block["direction"] == "SHORT"
    assert block["prior_session"]["shape"] == "DOWN_STALLED"


def test_a_gapped_open_is_not_followed():
    block = _follow(UP_STALLED, 22707.0 * 1.006, 1)
    assert block["active"] is False
    assert block["reason"] == "today's open is GAP_UP, not FLAT"


def test_a_move_that_ran_into_the_close_is_not_a_stall():
    # Same day, but the high comes at 14:50 and the 3:15 level sits right under it.
    ran = [("09:15", 22678.0), ("10:22", 22498.0), ("14:50", 22777.0), ("15:14", 22770.0),
           ("15:15", 22770.0), ("15:29", 22770.0)]
    block = _follow(ran, 22760.0, 1)
    assert block["active"] is False
    assert block["prior_session"]["shape"] == "NONE"
    assert block["reason"].startswith("yesterday did not close in the outer third")


def test_an_extreme_made_after_1330_is_not_a_stall():
    # Same fade (0.31%) and the same top-quarter close, but the high comes at 14:00:
    # only the 13:30 rule tells this apart from 28 Oct.
    late_high = [("09:15", 22678.0), ("10:22", 22498.0), ("14:00", 22777.0), ("15:14", 22707.0),
                 ("15:15", 22707.0), ("15:29", 22707.0)]
    block = _follow(late_high, 22691.0, 1)
    assert block["active"] is False
    assert block["prior_session"]["shape"] == "NONE"


def test_a_fade_under_015_pct_is_not_a_stall():
    shallow = [("09:15", 22678.0), ("10:22", 22498.0), ("13:00", 22777.0), ("15:14", 22760.0),
               ("15:15", 22760.0), ("15:29", 22760.0)]
    assert _follow(shallow, 22755.0, 1)["active"] is False


def test_the_settlement_prints_never_count_as_yesterdays_extreme():
    # A closing-auction spike after 15:15 must not become "the high at 15:29".
    spiked = [*UP_STALLED[:-1], ("15:29", 22900.0)]
    block = _follow(spiked, 22691.0, 1)
    assert block["active"] is True
    assert block["prior_session"]["extreme_time"] == "13:00"


def test_a_frame_that_does_not_start_at_0915_is_not_the_window():
    frame = _frame(UP_STALLED, 22691.0, 2)
    late = frame[frame["timestamp"] != pd.Timestamp("2026-10-29 09:15")]
    block = pivot_and_levels(late)["opening_follow"]
    assert block["active"] is False


def test_the_kill_switch_reports_it_switched_off():
    block = _follow(UP_STALLED, 22691.0, 1, SLHuntingIndicatorConfig(opening_follow_enabled=False))
    assert block == {"available": False, "active": False, "direction": None,
                     "reason": "switched off (SL_HUNTING_OPENING_FOLLOW_ENABLED=false)"}


def test_banknifty_levels_leave_the_block_out_and_nifty_levels_carry_it():
    frame = _frame(UP_STALLED, 22691.0, 1)
    ctx = SLHuntingToolContext(candles=frame, cfg=SLHuntingIndicatorConfig(), executor=StandaloneExecutor(),
                               bnf_candles=frame)
    assert ctx.pivot_and_levels_payload()["opening_follow"]["active"] is True
    assert "opening_follow" not in ctx.bank_nifty_payload()["pivot_and_levels"]
