"""Tests for the dashboard's pure data shaping (`Dependencies/dashboard_snapshot.py`).

The module has one hard job: turn a flat stream of trade events into "which
trades closed, when did each open, and what is still running" -- without ever
inventing a number. Most of what follows is the pairing logic, because that is
where a wrong answer would put a plausible-looking entry time beside a real
trade, and because three worker families publish several independent positions
under a single strategy name.
"""

from __future__ import annotations

import json
import math

import pytest

# Bare import: this folder's conftest.py puts the SOURCE `Dependencies/` on
# sys.path, which is the same resolution the runtime performs.
from dashboard_snapshot import (
    PAIR_DIRECTION,
    PAIR_EXACT,
    PAIR_OVERLAP,
    PAIR_SOLE_OPEN,
    PAIR_UNPAIRED,
    DocumentInputs,
    build_dashboard_document,
    closed_trades_by_strategy,
    entry_time_for,
    event_symbol_key,
    pair_trade_events,
    parse_event_timestamp,
    render_document_bytes,
    session_totals,
    strategy_rows,
)


def entry(strategy, ts, symbols, *, direction="BULLISH", quantity=75, price=100.0):
    return {
        "action": "ENTRY",
        "strategy": strategy,
        "ts": f"2026-09-10 {ts}",
        "mode": "PAPER",
        "direction": direction,
        "quantity": quantity,
        "legs": [{"symbol": s, "side": "BUY", "entry_price": price} for s in symbols],
    }


def exit_(strategy, ts, symbols, *, direction="BULLISH", pnl=100.0, action="EXIT"):
    return {
        "action": action,
        "strategy": strategy,
        "ts": f"2026-09-10 {ts}",
        "mode": "PAPER",
        "direction": direction,
        "quantity": 75,
        "pnl": pnl,
        "reason": "TARGET",
        "legs": [
            {"symbol": s, "side": "SELL", "entry_price": 100.0, "exit_price": 110.0}
            for s in symbols
        ],
    }


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------
def test_symbol_key_is_upper_cased_and_order_free():
    left = event_symbol_key({"legs": [{"symbol": "nifty-ce"}, {"symbol": "NIFTY-PE"}]})
    right = event_symbol_key({"legs": [{"symbol": "NIFTY-PE"}, {"symbol": " nifty-ce "}]})
    assert left == right == frozenset({"NIFTY-CE", "NIFTY-PE"})


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", 12345, "not a timestamp", "2026-09-10", "2026-09-10 25:00:00",
     "2026-09-10 10:70:00", {"legs": []}],
)
def test_unreadable_timestamps_yield_none_rather_than_raising(value):
    assert parse_event_timestamp(value) is None


def test_a_normal_timestamp_becomes_a_clock_time():
    assert parse_event_timestamp("2026-09-10 09:17:05") == "09:17:05"


def test_events_with_no_legs_are_simply_unkeyable():
    for event in ({}, {"legs": None}, {"legs": []}, {"legs": "NIFTY-CE"}, {"legs": [7]}):
        assert event_symbol_key(event) == frozenset()


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def test_a_simple_round_trip_carries_its_entry_time():
    ledger = pair_trade_events(
        [entry("Renko", "09:30:00", ["NIFTY-24500-CE"]),
         exit_("Renko", "09:52:10", ["NIFTY-24500-CE"])]
    )
    assert len(ledger.closed) == 1
    trade = ledger.closed[0]
    assert trade.entry_time == "09:30:00"
    assert trade.exit_time == "09:52:10"
    assert trade.pair_confidence == PAIR_EXACT
    assert ledger.open_entries == ()


def test_delta20_ce_and_pe_pair_by_symbol_not_by_arrival_order():
    """Both sides publish under ONE strategy name and interleave freely."""
    ledger = pair_trade_events(
        [
            entry("Delta20Hedged", "09:20:00", ["NIFTY-23000-CE", "NIFTY-23200-CE"],
                  direction="CE"),
            entry("Delta20Hedged", "09:21:00", ["NIFTY-22000-PE", "NIFTY-21800-PE"],
                  direction="PE"),
            # The PE side closes FIRST; naive FIFO would hand it the CE entry.
            exit_("Delta20Hedged", "14:00:00", ["NIFTY-22000-PE", "NIFTY-21800-PE"],
                  direction="PE"),
        ]
    )
    assert [trade.entry_time for trade in ledger.closed] == ["09:21:00"]
    assert ledger.closed[0].pair_confidence == PAIR_EXACT
    # The CE side is still open.
    assert [e.direction for e in ledger.open_entries] == ["CE"]


def test_the_sl_hunting_mirror_is_separated_by_symbol_not_direction():
    """The NIFTY leg and its BankNIFTY mirror share a strategy AND a direction.

    Only the contract symbol tells them apart, which is why OVERLAP must
    outrank DIRECTION in the resolution ladder.
    """
    ledger = pair_trade_events(
        [
            entry("SL Hunting AI", "09:45:00", ["NIFTY-24500-CE"], direction="LONG"),
            entry("SL Hunting AI", "09:45:01", ["BANKNIFTY-57900-CE"], direction="LONG"),
            exit_("SL Hunting AI", "10:05:00", ["BANKNIFTY-57900-CE"], direction="LONG"),
        ]
    )
    assert ledger.closed[0].entry_time == "09:45:01"
    assert ledger.closed[0].pair_confidence == PAIR_EXACT
    assert [e.symbols for e in ledger.open_entries] == [frozenset({"NIFTY-24500-CE"})]


def test_a_partial_leg_close_falls_back_to_the_largest_overlap():
    ledger = pair_trade_events(
        [
            entry("SupertrendBullish", "09:30:00", ["NIFTY-22000-PE", "NIFTY-21000-PE"]),
            exit_("SupertrendBullish", "11:00:00", ["NIFTY-21000-PE"]),
        ]
    )
    assert ledger.closed[0].pair_confidence == PAIR_OVERLAP
    assert ledger.closed[0].entry_time == "09:30:00"


def test_direction_is_the_backstop_when_no_symbol_matches():
    ledger = pair_trade_events(
        [
            entry("LongStrangle", "09:30:00", ["NIFTY-CE"], direction="CE"),
            entry("LongStrangle", "09:30:01", ["NIFTY-PE"], direction="PE"),
            # An exit whose legs name a contract neither entry mentions.
            exit_("LongStrangle", "12:00:00", ["NIFTY-OTHER-PE"], direction="PE"),
        ]
    )
    assert ledger.closed[0].pair_confidence == PAIR_DIRECTION
    assert ledger.closed[0].entry_time == "09:30:01"


def test_a_lone_open_entry_resolves_even_with_nothing_else_to_go_on():
    ledger = pair_trade_events(
        [entry("Renko", "09:30:00", ["NIFTY-CE"], direction=""),
         {"action": "EXIT", "strategy": "Renko", "ts": "2026-09-10 10:00:00", "pnl": 5.0}]
    )
    assert ledger.closed[0].pair_confidence == PAIR_SOLE_OPEN
    assert ledger.closed[0].entry_time == "09:30:00"


def test_a_re_entry_on_the_same_strike_pairs_oldest_first():
    ledger = pair_trade_events(
        [
            entry("LongStrangle", "09:30:00", ["NIFTY-CE"], direction="CE"),
            exit_("LongStrangle", "10:00:00", ["NIFTY-CE"], direction="CE"),
            entry("LongStrangle", "11:00:00", ["NIFTY-CE"], direction="CE"),
            exit_("LongStrangle", "12:00:00", ["NIFTY-CE"], direction="CE"),
        ]
    )
    assert [trade.entry_time for trade in ledger.closed] == ["09:30:00", "11:00:00"]
    assert ledger.open_entries == ()


def test_an_unpaired_exit_is_still_a_complete_row():
    """Only the entry TIME is unknown -- the exit event carries everything else.

    This is what a 5000-event truncation or a mid-day restart produces, and
    fabricating a time for it would be worse than a dash.
    """
    ledger = pair_trade_events([exit_("Renko", "10:00:00", ["NIFTY-CE"], pnl=-250.0)])
    trade = ledger.closed[0]
    assert trade.pair_confidence == PAIR_UNPAIRED
    assert trade.entry_time is None
    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0
    assert trade.pnl == -250.0
    assert ledger.unpaired_exits == 1


# ---------------------------------------------------------------------------
# Non-trade actions
# ---------------------------------------------------------------------------
def test_exit_failed_never_closes_anything():
    """The broker did NOT confirm the close, so the position is still open."""
    ledger = pair_trade_events(
        [
            entry("Renko", "09:30:00", ["NIFTY-CE"]),
            exit_("Renko", "10:00:00", ["NIFTY-CE"], action="EXIT_FAILED"),
        ]
    )
    assert ledger.closed == ()
    assert len(ledger.open_entries) == 1
    assert ledger.open_entries[0].exit_failed is True
    assert ledger.open_entries[0].last_exit_attempt == "10:00:00"
    assert [n.action for n in ledger.notices] == ["EXIT_FAILED"]
    assert ledger.notices[0].urgent is True


def test_a_retried_exit_after_a_failure_does_close_the_trade():
    ledger = pair_trade_events(
        [
            entry("Renko", "09:30:00", ["NIFTY-CE"]),
            exit_("Renko", "10:00:00", ["NIFTY-CE"], action="EXIT_FAILED"),
            exit_("Renko", "10:00:30", ["NIFTY-CE"]),
        ]
    )
    assert len(ledger.closed) == 1
    assert ledger.closed[0].entry_time == "09:30:00"
    assert ledger.open_entries == ()


@pytest.mark.parametrize(
    "action",
    ["INDETERMINATE_EXPOSURE", "UNHEDGED_LEG_OPEN", "UNHEDGED_LEG_CLOSED",
     "MARKET_DATA_AUTO_SQUARE_OFF", "SHUTDOWN_DEGRADED", "SOMETHING_ADDED_LATER"],
)
def test_non_trade_actions_become_notices_and_never_disturb_pairing(action):
    ledger = pair_trade_events(
        [
            entry("Renko", "09:30:00", ["NIFTY-CE"]),
            {"action": action, "strategy": "Renko", "ts": "2026-09-10 09:45:00",
             "reason": "why", "legs": [{"symbol": "NIFTY-CE"}]},
            exit_("Renko", "10:00:00", ["NIFTY-CE"]),
        ]
    )
    assert len(ledger.closed) == 1
    assert ledger.closed[0].entry_time == "09:30:00"
    assert [n.action for n in ledger.notices] == [action]


def test_malformed_events_are_skipped_rather_than_raising():
    """This runs on a monitoring thread while real money is trading."""
    ledger = pair_trade_events(
        [
            None,  # type: ignore[list-item]
            "not an event",  # type: ignore[list-item]
            {},
            {"action": "ENTRY"},
            {"action": "EXIT", "pnl": float("nan")},
            entry("Renko", "09:30:00", ["NIFTY-CE"]),
        ]
    )
    # The two headless events share the "(unknown)" strategy and pair with
    # each other, which is the right answer for events carrying nothing to
    # tell them apart. The real entry is untouched.
    assert [e.strategy for e in ledger.open_entries] == ["Renko"]
    assert ledger.closed[0].pnl is None  # NaN is not a number this page will show


# ---------------------------------------------------------------------------
# Open-position entry times
# ---------------------------------------------------------------------------
def test_entry_time_lookup_matches_by_symbol_then_falls_back():
    ledger = pair_trade_events(
        [entry("Renko", "09:30:00", ["NIFTY-CE"]),
         entry("Delta20Hedged", "09:31:00", ["NIFTY-PE"], direction="PE")]
    )
    assert entry_time_for(ledger, "Renko", frozenset({"NIFTY-CE"})) == ("09:30:00", False)
    # No symbols at all, but the strategy has exactly one open entry.
    assert entry_time_for(ledger, "Renko", frozenset()) == ("09:30:00", False)
    # A strategy with nothing open cannot say.
    assert entry_time_for(ledger, "Goldmine", frozenset({"X"})) == (None, False)


def test_entry_time_lookup_reports_a_failed_exit():
    ledger = pair_trade_events(
        [entry("Renko", "09:30:00", ["NIFTY-CE"]),
         exit_("Renko", "10:00:00", ["NIFTY-CE"], action="EXIT_FAILED")]
    )
    assert entry_time_for(ledger, "Renko", frozenset({"NIFTY-CE"})) == ("09:30:00", True)


# ---------------------------------------------------------------------------
# Rollups
# ---------------------------------------------------------------------------
def view(strategy, realized, marks, *, trades=1, mode="PAPER", live=False):
    return {
        "strategy": strategy,
        "mode": mode,
        "live_trading": live,
        "completed_trades": trades,
        "realized_pnl": realized,
        "positions": [{"unrealized_pnl": mark} for mark in marks],
    }


def test_strategy_rows_add_realized_and_open():
    rows = strategy_rows([view("Renko", -929.5, [250.0, -100.0])])
    assert rows[0]["realized"] == -929.5
    assert rows[0]["open"] == 150.0
    assert rows[0]["total"] == -779.5
    assert rows[0]["open_positions"] == 2


def test_one_unpriced_position_makes_open_and_total_unknown_but_not_realized():
    """A partial sum shown as a whole would understate an open loss."""
    rows = strategy_rows([view("Delta20Hedged", -400.0, [250.0, None])])
    assert rows[0]["realized"] == -400.0
    assert rows[0]["open"] is None
    assert rows[0]["total"] is None


def test_a_flat_strategy_reports_zero_open_not_unknown():
    rows = strategy_rows([view("Goldmine", 120.0, [])])
    assert rows[0]["open"] == 0.0
    assert rows[0]["total"] == 120.0


def test_rows_sort_by_total_with_unpriced_last():
    rows = strategy_rows(
        [view("Loser", -500.0, []), view("Winner", 900.0, []), view("Unknown", 10.0, [None])]
    )
    assert [row["strategy"] for row in rows] == ["Winner", "Loser", "Unknown"]


def test_session_totals_count_the_strategies_they_could_not_price():
    rows = strategy_rows(
        [view("A", 100.0, [50.0]), view("B", -40.0, [None]), view("C", 10.0, [])]
    )
    totals = session_totals(rows)
    assert totals["realized"] == 70.0
    assert totals["open"] == 50.0  # B's open is excluded, and counted instead
    assert totals["unpriced_strategies"] == 1
    assert totals["total"] == 120.0
    assert totals["strategies"] == 3


def test_closed_trades_group_by_strategy_with_their_own_subtotals():
    ledger = pair_trade_events(
        [
            entry("Renko", "09:30:00", ["A"]), exit_("Renko", "10:00:00", ["A"], pnl=100.0),
            entry("Renko", "10:30:00", ["B"]), exit_("Renko", "11:00:00", ["B"], pnl=-40.0),
            entry("Goldmine", "09:31:00", ["C"]),
            exit_("Goldmine", "10:01:00", ["C"], pnl=7.0),
        ]
    )
    groups = closed_trades_by_strategy(ledger)
    assert [group["strategy"] for group in groups] == ["Renko", "Goldmine"]
    assert groups[0]["trades"] == 2
    assert groups[0]["realized"] == 60.0
    assert groups[1]["realized"] == 7.0


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
def _document():
    ledger = pair_trade_events(
        [entry("Renko", "09:30:00", ["A"]), exit_("Renko", "10:00:00", ["A"], pnl=100.0)]
    )
    return build_dashboard_document(
        DocumentInputs(
            generated_at="10:00:01",
            version=3,
            poll_seconds=1.0,
            session={"date": "2026-09-10", "phase": "OPEN"},
            worker_views=[view("Renko", 100.0, [])],
            ledger=ledger,
            chart={"state": "OK", "series_version": 2},
            feed={"source": "REST"},
        )
    )


def test_the_document_carries_every_pane_the_page_reads():
    document = _document()
    assert set(document) == {
        "generated_at", "version", "poll_seconds", "session", "totals", "strategies",
        "open_positions", "closed_trades", "notices", "unpaired_exits", "chart", "feed",
    }
    assert document["totals"]["realized"] == 100.0
    assert document["closed_trades"][0]["strategy"] == "Renko"


def test_open_positions_are_flattened_and_tagged_with_their_strategy():
    document = build_dashboard_document(
        DocumentInputs(
            generated_at="10:00:01", version=1, poll_seconds=1.0, session={},
            worker_views=[
                {"strategy": "Delta20Hedged", "realized_pnl": 0.0,
                 "positions": [{"slot": "ce_pos"}, {"slot": "pe_pos"}]},
                {"strategy": "Broken", "realized_pnl": 0.0, "positions": None},
            ],
            ledger=pair_trade_events([]),
        )
    )
    assert [row["strategy"] for row in document["open_positions"]] == [
        "Delta20Hedged", "Delta20Hedged"
    ]


def test_rendering_produces_parseable_json():
    payload = render_document_bytes(_document())
    assert json.loads(payload)["version"] == 3


def test_rendering_refuses_nan_rather_than_emitting_invalid_json():
    """`JSON.parse` rejects bare NaN, so the whole page would go blank.

    Raising here instead means the builder logs it and keeps serving the last
    good document.
    """
    with pytest.raises(ValueError):
        render_document_bytes({"totals": {"open": math.nan}})
