"""v6e: the up-run short veto is computed in code (operator, 14 Nov).

After four or more straight up days -- each session's 3:15 level above the one
before -- a FLAT open is not shorted until NIFTY trades below yesterday's low.
These tests pin that contract:

* the count walks back from yesterday on 3:15 levels, never on settlement prints;
* active only for four or more rises, a FLAT open, and yesterday's low intact;
* every reason it is not active, named, including a window too short to see four;
* BankNIFTY's levels leave the block out.
"""

from __future__ import annotations

import pandas as pd
from sl_hunting_executor import StandaloneExecutor
from sl_hunting_indicators import pivot_and_levels, session_315_levels
from sl_hunting_tools import SLHuntingToolContext

# Five sessions before "today" (2026-11-06), Fri 30 Oct to Thu 5 Nov.
PRIOR_DAYS = ["2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]


def _session(day, level_315, *, low=None, settlement=None):
    """A full session whose last pre-15:15 close is `level_315`.

    `settlement`, when given, is the flat value printed from 15:15 on, which must
    never be read as the session's level.
    """
    stamps = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min")
    close = pd.Series(float(level_315), index=stamps)
    if settlement is not None:
        close[close.index.time >= pd.Timestamp("15:15").time()] = float(settlement)
    lows = close - 1.0
    if low is not None:
        lows.iloc[60] = float(low)
    return pd.DataFrame({"timestamp": stamps, "open": close.values, "high": close.values + 1.0,
                         "low": lows.values, "close": close.values})


def _today(open_price, minutes=5, low=None):
    stamps = pd.date_range("2026-11-06 09:15", periods=minutes, freq="1min")
    frame = pd.DataFrame({"timestamp": stamps, "open": open_price, "high": open_price + 3.0,
                          "low": open_price - 3.0, "close": open_price})
    if low is not None:
        frame.loc[frame.index[-1], "low"] = float(low)
    return frame


def _run(levels, open_price, *, days=None, today_low=None, yesterday_low=None, settlement=None,
         history=None):
    days = days or PRIOR_DAYS[-len(levels):]
    sessions = [
        _session(day, lvl,
                 low=yesterday_low if i == len(levels) - 1 else None,
                 settlement=settlement if i == len(levels) - 1 else None)
        for i, (day, lvl) in enumerate(zip(days, levels, strict=True))
    ]
    frame = pd.concat([*sessions, _today(open_price, low=today_low)], ignore_index=True)
    return pivot_and_levels(frame, up_run_history=history)["up_run"]


RISING = [24000.0, 24100.0, 24200.0, 24300.0, 24400.0]   # four rises


def test_active_after_four_rises_on_a_flat_open():
    block = _run(RISING, 24420.0)                     # +0.08%: FLAT
    assert block["available"] is True
    assert block["straight_up_days"] == 4
    assert block["at_least"] is True                  # the window ran out while rising
    assert block["sessions_seen"] == 5
    assert block["open_verdict"] == "FLAT"
    assert block["short_veto_active"] is True
    assert block["reason"] is None
    assert "do NOT open a SHORT" in block["note"]
    assert "never creates a long" in block["note"]


def test_three_rises_are_not_enough():
    block = _run([24300.0, 24000.0, 24100.0, 24200.0, 24300.0], 24310.0)
    assert block["straight_up_days"] == 3
    assert block["at_least"] is False
    assert block["short_veto_active"] is False
    assert block["reason"].startswith("3 straight up day(s) visible before today; the veto needs 4")
    assert "window ran out" not in block["reason"]


def test_a_holiday_week_window_cannot_see_four_rises_and_says_so():
    """13 Nov: the 10 Nov holiday left four sessions in the 7-day window."""
    block = _run(RISING[1:], 24420.0, days=PRIOR_DAYS[1:])
    assert block["sessions_seen"] == 4
    assert block["straight_up_days"] == 3
    assert block["at_least"] is True
    assert block["short_veto_active"] is False
    assert "the true run may be longer" in block["reason"]


def test_a_down_day_inside_the_window_resets_the_count():
    block = _run([24000.0, 24100.0, 24050.0, 24300.0, 24400.0], 24410.0)
    assert block["straight_up_days"] == 2
    assert block["short_veto_active"] is False


def test_an_equal_level_is_not_a_rise():
    block = _run([24000.0, 24100.0, 24200.0, 24200.0, 24300.0], 24310.0)
    assert block["straight_up_days"] == 1


def test_not_active_on_a_gap_up_or_a_gap_down():
    assert _run(RISING, 24600.0)["reason"] == "today's open is GAP_UP, not FLAT"
    assert _run(RISING, 24200.0)["reason"] == "today's open is GAP_DOWN, not FLAT"
    assert _run(RISING, 24600.0)["short_veto_active"] is False


def test_ends_once_nifty_trades_below_yesterdays_low():
    held = _run(RISING, 24420.0, yesterday_low=24350.0, today_low=24360.0)
    assert held["yesterday_low"] == 24350.0
    assert held["short_veto_active"] is True
    broke = _run(RISING, 24420.0, yesterday_low=24350.0, today_low=24349.0)
    assert broke["short_veto_active"] is False
    assert broke["reason"] == "ended: NIFTY has traded below yesterday's low today"


def test_settlement_prints_never_count_as_yesterdays_level():
    """Yesterday's 3:15 level rose; its 15:15-on prints fell. The level counts."""
    block = _run(RISING, 24420.0, settlement=24250.0)
    assert block["straight_up_days"] == 4


def test_too_little_history_is_unavailable():
    block = _run([24400.0], 24420.0)
    assert block["available"] is False
    assert block["short_veto_active"] is False


def test_banknifty_levels_leave_the_block_out():
    frame = pd.concat([*(_session(d, lvl) for d, lvl in zip(PRIOR_DAYS, RISING, strict=True)), _today(24420.0)],
                      ignore_index=True)
    ctx = SLHuntingToolContext.build(frame, StandaloneExecutor(), bnf_candles=frame)
    assert ctx.pivot_and_levels_payload()["up_run"]["short_veto_active"] is True
    bnf = ctx.bank_nifty_payload()["pivot_and_levels"]
    assert "up_run" not in bnf
    assert "opening_follow" not in bnf


# --------------------------------------------------------------------------
# The longer history the worker fetches once per session
# --------------------------------------------------------------------------

def _d(text):
    return pd.Timestamp(text).date()


def test_history_lets_a_holiday_week_see_the_fourth_rise():
    """13 Nov's case: four sessions in the frame, the fifth only in the history."""
    history = [(_d("2026-10-30"), 24000.0)]
    block = _run(RISING[1:], 24420.0, days=PRIOR_DAYS[1:], history=history)
    assert block["sessions_seen"] == 5
    assert block["straight_up_days"] == 4
    assert block["short_veto_active"] is True


def test_history_extends_the_count_past_the_frame():
    history = [(_d("2026-10-28"), 23800.0), (_d("2026-10-29"), 23900.0)]
    block = _run(RISING, 24420.0, history=history)
    assert block["straight_up_days"] == 6
    assert block["sessions_seen"] == 7


def test_the_frame_wins_where_both_cover_a_day_and_today_is_never_history():
    # History claims 2 Nov fell; the frame's own 2 Nov rose, so the count holds.
    history = [(_d("2026-10-30"), 24000.0), (_d("2026-11-02"), 23000.0), (_d("2026-11-06"), 1.0)]
    block = _run(RISING, 24420.0, history=history)
    assert block["straight_up_days"] == 4
    assert block["sessions_seen"] == 5


def test_a_falling_day_in_the_history_stops_the_count():
    history = [(_d("2026-10-28"), 23700.0), (_d("2026-10-29"), 24500.0)]
    block = _run(RISING, 24420.0, history=history)
    assert block["straight_up_days"] == 4
    assert block["at_least"] is False


def test_session_315_levels_reads_every_session_before_settlement():
    frame = pd.concat([_session("2026-11-04", 24200.0, settlement=24100.0),
                       _session("2026-11-05", 24300.0)], ignore_index=True)
    assert session_315_levels(frame) == [(_d("2026-11-04"), 24200.0), (_d("2026-11-05"), 24300.0)]
    assert session_315_levels(frame.iloc[0:0]) == []


def test_the_tool_context_hands_the_history_to_the_levels_tool():
    frame = pd.concat([_session(d, lvl) for d, lvl in zip(PRIOR_DAYS[1:], RISING[1:], strict=True)]
                      + [_today(24420.0)], ignore_index=True)
    plain = SLHuntingToolContext.build(frame, StandaloneExecutor())
    assert plain.pivot_and_levels_payload()["up_run"]["short_veto_active"] is False
    ctx = SLHuntingToolContext.build(frame, StandaloneExecutor(),
                                     up_run_history=[(_d("2026-10-30"), 24000.0)])
    assert ctx.pivot_and_levels_payload()["up_run"]["short_veto_active"] is True
