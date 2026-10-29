"""Tests for the position-group target monitor.

Every collaborator is a fake: prices come from a dict, orders go to an
in-memory broker, alerts go to a list.  ``tick()`` is called directly, so no
thread, clock or network is involved.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import position_group_monitor as monitor_module
import pytest
from broker_contract import (
    BrokerQueryResult,
    OrderStatus,
    PositionDetail,
    normalize_order_result,
)
from position_group_monitor import (
    DhanContract,
    PositionGroupMonitor,
    build_dhan_contract_index,
    exit_plan,
    group_pnl,
    holdings_mismatch,
    within_action_window,
)
from position_groups import parse_config

MARKET_TIME = datetime(2026, 10, 23, 11, 0)  # a Friday
SHORT_CE = "NIFTY25N2724500CE"
LONG_CE = "NIFTY25N2725000CE"


def leg(symbol: str, quantity: int, entry: float, strike: int, right: str = "CE") -> dict:
    return {"symbol": symbol, "exchange": "NFO", "product": "NRML", "quantity": quantity,
            "entry_price": entry, "underlying": "NIFTY", "expiry": "2025-11-27",
            "strike": strike, "right": right}


def config_document(target: float = 1000, booked: float = 0, legs: list | None = None,
                    name: str = "Spread") -> dict:
    return {
        "schema_version": 1,
        "groups": [{
            "name": name, "broker": "ZERODHA", "target": target, "booked_pnl": booked,
            "legs": legs or [leg(SHORT_CE, -150, 100.0, 24500), leg(LONG_CE, 150, 40.0, 25000)],
        }],
        "unassigned": [],
    }


CONTRACTS = {
    ("NSE", "NIFTY", date(2025, 11, 27), 2450000, "CE"): DhanContract("NSE_FNO", 111, 75, 1801),
    ("NSE", "NIFTY", date(2025, 11, 27), 2500000, "CE"): DhanContract("NSE_FNO", 222, 75, 1801),
}


class FakeBroker:
    broker_name = "ZERODHA"
    is_logged_in = True

    def __init__(self) -> None:
        self.holdings: dict[str, int] = {SHORT_CE: -150, LONG_CE: 150}
        self.book_indeterminate = False
        self.outcomes: dict[str, OrderStatus] = {}
        self.orders: list[tuple[str, str, int, str, str, str]] = []

    def ensure_logged_in(self) -> bool:
        return True

    def list_position_details(self):
        if self.book_indeterminate:
            return BrokerQueryResult.indeterminate("down")
        return BrokerQueryResult.success(
            PositionDetail(symbol=s, exchange="NFO", product_type="NRML", quantity=q)
            for s, q in self.holdings.items()
        )

    def place_market_order(self, symbol, side, quantity, exchange_segment, product_type, *, order_tag=""):
        self.orders.append((symbol, side, quantity, exchange_segment, product_type, order_tag))
        status = self.outcomes.get(symbol, OrderStatus.FILLED)
        filled = {OrderStatus.FILLED: quantity, OrderStatus.PARTIAL: 1}.get(status, 0)
        state = {OrderStatus.FILLED: "COMPLETE", OrderStatus.REJECTED: "REJECTED",
                 OrderStatus.PARTIAL: "CANCELLED"}.get(status, "OPEN")
        return normalize_order_result(order_id="O", requested_quantity=quantity,
                                      filled_quantity=filled, broker_state=state,
                                      average_fill_price=10.0)

    def get_order_status(self, order_id, requested_quantity=0):
        raise AssertionError("not used")

    def logout(self):
        return {}


class Harness:
    def __init__(self, tmp_path: Path, document: dict | None = None) -> None:
        self.config_path = tmp_path / "groups.json"
        self.state_path = tmp_path / "groups.state.json"
        self.write(document or config_document())
        self.prices: dict[int, float] = {111: 100.0, 222: 40.0}
        self.alerts: list[dict[str, Any]] = []
        self.subscribed: list[int] = []
        self.unsubscribed: list[int] = []
        self.feed_log: list[tuple[str, int]] = []
        self.broker = FakeBroker()
        self.now = MARKET_TIME
        self.index_loads = 0
        self.monitor = self.build()

    def build(self) -> PositionGroupMonitor:
        def load_index():
            self.index_loads += 1
            return CONTRACTS

        return PositionGroupMonitor(
            config_path=self.config_path,
            state_path=self.state_path,
            clients={"ZERODHA": self.broker},
            contract_index_loader=load_index,
            subscribe=lambda _leg, contract: self._feed("sub", contract.security_id),
            unsubscribe=lambda contract: self._feed("unsub", contract.security_id),
            read_ltp=lambda contract: self.prices.get(contract.security_id, 0.0),
            alert=self.alerts.append,
            now=lambda: self.now,
            confirm_ticks=3,
        )

    def write(self, document: dict) -> None:
        self.config_path.write_text(json.dumps(document))
        # Force a visible mtime change even on coarse filesystem clocks.
        stamp = self.config_path.stat().st_mtime + len(json.dumps(document)) + 1
        os.utime(self.config_path, (stamp, stamp))

    def set_pnl(self, short_ltp: float, long_ltp: float) -> None:
        self.prices = {111: short_ltp, 222: long_ltp}

    def ticks(self, count: int) -> None:
        for _ in range(count):
            self.monitor.tick()

    def _feed(self, kind: str, security_id: int) -> None:
        (self.subscribed if kind == "sub" else self.unsubscribed).append(security_id)
        self.feed_log.append((kind, security_id))

    def live_feed(self) -> set[int]:
        """Replay the feed log with the store's semantics: ONE owner per contract."""
        live: set[int] = set()
        for kind, security_id in self.feed_log:
            (live.add if kind == "sub" else live.discard)(security_id)
        return live

    def actions(self) -> list[str]:
        return [alert["action"] for alert in self.alerts]

    def state(self) -> dict:
        return next(iter(json.loads(self.state_path.read_text())["groups"].values()))


# --- pure pieces --------------------------------------------------------------
def test_group_pnl_adds_booked_and_signed_mtm() -> None:
    (group,) = parse_config(config_document(booked=-500)).groups
    short, long_ = (leg.key for leg in group.legs)
    # Short decays 100 -> 90 (+1500), long 40 -> 35 (-750), booked -500.
    assert group_pnl(group, {short: 90.0, long_: 35.0}) == pytest.approx(250.0)
    assert group_pnl(group, {short: 90.0}) is None
    assert group_pnl(group, {short: 90.0, long_: 0.0}) is None


def test_exit_plan_buys_back_shorts_before_selling_longs_and_splits_at_freeze() -> None:
    legs = [leg(LONG_CE, 150, 40.0, 25000), leg(SHORT_CE, -1950, 100.0, 24500)]
    (group,) = parse_config(config_document(legs=legs)).groups
    contracts = {
        group.legs[0].key: CONTRACTS[("NSE", "NIFTY", date(2025, 11, 27), 2500000, "CE")],
        group.legs[1].key: CONTRACTS[("NSE", "NIFTY", date(2025, 11, 27), 2450000, "CE")],
    }
    plan = [(order.leg.symbol, order.side, order.quantity) for order in exit_plan(group, contracts)]
    # Freeze 1801 with lot 75 -> chunks of 1800.
    assert plan == [(SHORT_CE, "BUY", 1800), (SHORT_CE, "BUY", 150), (LONG_CE, "SELL", 150)]


def test_exit_plan_without_freeze_data_sends_one_order_per_leg() -> None:
    (group,) = parse_config(config_document()).groups
    plan = exit_plan(group, {})
    assert [(o.side, o.quantity) for o in plan] == [("BUY", 150), ("SELL", 150)]


def test_holdings_mismatch_catches_closed_resized_and_flipped_legs() -> None:
    (group,) = parse_config(config_document()).groups

    def held(**quantities):
        return [PositionDetail(symbol=s, exchange="NFO", product_type="NRML", quantity=q)
                for s, q in quantities.items()]

    assert holdings_mismatch(group, held(**{SHORT_CE: -150, LONG_CE: 300})) == []
    assert holdings_mismatch(group, held(**{LONG_CE: 150}))  # short closed by hand
    assert holdings_mismatch(group, held(**{SHORT_CE: -75, LONG_CE: 150}))  # resized
    assert holdings_mismatch(group, held(**{SHORT_CE: 150, LONG_CE: 150}))  # flipped


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (datetime(2026, 10, 23, 9, 14, 59), False),
        (datetime(2026, 10, 23, 9, 15), True),
        (datetime(2026, 10, 23, 15, 29, 30), True),
        (datetime(2026, 10, 23, 15, 30), False),
        (datetime(2026, 10, 24, 11, 0), False),  # Saturday
    ],
)
def test_action_window(moment: datetime, expected: bool) -> None:
    assert within_action_window(moment) is expected


def test_dhan_index_reads_options_with_lot_and_freeze(tmp_path: Path) -> None:
    path = tmp_path / "all_instrument.csv"
    path.write_text(
        "EXCH_ID,SEGMENT,SECURITY_ID,INSTRUMENT,UNDERLYING_SYMBOL,LOT_SIZE,SM_EXPIRY_DATE,"
        "STRIKE_PRICE,OPTION_TYPE,SM_FREEZE_QTY\n"
        "NSE,D,111,OPTIDX,NIFTY,75,2025-11-27 14:30:00,24500.00000,CE,1801\n"
        "BSE,D,333,OPTIDX,SENSEX,20,2025-11-27,80000,PE,1001\n"
        "NSE,D,444,FUTIDX,NIFTY,75,2025-11-27,0,XX,1801\n"
        "NSE,E,555,EQUITY,INFY,1,,,,\n"
    )
    index = build_dhan_contract_index(path)
    assert index[("NSE", "NIFTY", date(2025, 11, 27), 2450000, "CE")] == DhanContract("NSE_FNO", 111, 75, 1801)
    assert index[("BSE", "SENSEX", date(2025, 11, 27), 8000000, "PE")].segment == "BSE_FNO"
    assert len(index) == 2


# --- the thread's decisions -----------------------------------------------------
def test_target_needs_consecutive_confirmations_then_squares_off(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.set_pnl(90.0, 38.0)  # +1500 - 300 = +1200 >= 1000
    h.ticks(2)
    assert h.broker.orders == []
    h.set_pnl(95.0, 40.0)  # +750: streak resets
    h.ticks(1)
    h.set_pnl(90.0, 38.0)
    h.ticks(2)
    assert h.broker.orders == []
    h.ticks(1)
    assert [(o[0], o[1], o[2], o[4]) for o in h.broker.orders] == [
        (SHORT_CE, "BUY", 150, "NRML"), (LONG_CE, "SELL", 150, "NRML"),
    ]
    assert h.actions() == ["GROUP_TARGET_HIT", "GROUP_SQUARED_OFF"]
    state = h.state()
    assert state["status"] == "SQUARED_OFF"
    # Fills at 10.0 each: short (10-100)*-150 = 13500, long (10-40)*150 = -4500.
    assert state["realised_pnl"] == pytest.approx(9000.0)
    tags = [order[5] for order in h.broker.orders]
    assert len(set(tags)) == 2 and all(tag.isalnum() and len(tag) <= 20 for tag in tags)


def test_booked_pnl_counts_toward_the_target(tmp_path: Path) -> None:
    h = Harness(tmp_path, config_document(target=1000, booked=900))
    h.set_pnl(99.0, 40.0)  # +150 open, +900 booked
    h.ticks(3)
    assert h.actions()[0] == "GROUP_TARGET_HIT"


def test_a_missing_or_stale_price_never_triggers(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.prices = {111: 1.0}  # long leg unpriced
    h.ticks(5)
    assert h.broker.orders == []


def test_nothing_happens_outside_market_hours(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.now = datetime(2026, 10, 23, 15, 45)
    h.set_pnl(1.0, 40.0)
    h.ticks(5)
    assert h.broker.orders == []


def test_broker_book_mismatch_sends_no_orders(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.broker.holdings = {LONG_CE: 150}  # the short was closed by hand
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert h.broker.orders == []
    assert h.actions() == ["GROUP_EXIT_FAILED"]
    assert h.state()["status"] == "EXIT_FAILED"
    h.ticks(5)
    assert h.broker.orders == []  # terminal: never re-fired


def test_unreadable_book_sends_nothing_and_retries_later(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.broker.book_indeterminate = True
    h.set_pnl(1.0, 40.0)
    h.ticks(6)
    assert h.broker.orders == []
    assert h.actions() == ["GROUP_TARGET_HIT"]  # alerted once, not every retry
    h.broker.book_indeterminate = False
    h.ticks(3)
    assert h.actions()[-1] == "GROUP_SQUARED_OFF"


@pytest.mark.parametrize("outcome", [OrderStatus.REJECTED, OrderStatus.PARTIAL, OrderStatus.UNKNOWN])
def test_any_unproven_order_stops_the_group_without_retry(tmp_path: Path, outcome) -> None:
    h = Harness(tmp_path)
    h.broker.outcomes = {SHORT_CE: outcome}
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert [o[0] for o in h.broker.orders] == [SHORT_CE]  # the long was never sold
    assert h.actions() == ["GROUP_TARGET_HIT", "GROUP_EXIT_FAILED"]
    assert "STILL OPEN" in h.alerts[-1]["reason"]
    h.ticks(5)
    assert len(h.broker.orders) == 1


def test_order_exception_stops_the_group(tmp_path: Path) -> None:
    h = Harness(tmp_path)

    def boom(*args, **kwargs):
        raise TimeoutError("deadline")

    h.broker.place_market_order = boom
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert h.state()["status"] == "EXIT_FAILED"


def test_squared_off_group_stays_closed_after_restart(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    restarted = h.build()
    restarted.tick()
    restarted.tick()
    restarted.tick()
    assert len(h.broker.orders) == 2


def test_crash_mid_exit_comes_back_as_exit_failed(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    (group,) = parse_config(config_document()).groups
    h.state_path.write_text(json.dumps({"groups": {group.fingerprint: {"status": "EXITING"}}}))
    restarted = h.build()
    h.set_pnl(1.0, 40.0)
    for _ in range(3):
        restarted.tick()
    assert h.broker.orders == []
    assert restarted.state.status(group) == "EXIT_FAILED"


def test_corrupt_state_file_is_moved_aside(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.state_path.write_text("{corrupt")
    h.build()
    assert (tmp_path / "groups.state.json.unreadable").exists()


def test_invalid_edit_keeps_the_last_good_copy_and_alerts_once(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.ticks(1)
    h.write(config_document(target=-5))
    h.ticks(3)
    assert h.actions() == ["GROUP_FILE_INVALID"]
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert h.actions()[-1] == "GROUP_SQUARED_OFF"  # the last good group was still watched


def test_editing_the_target_keeps_watching_without_resubscribing(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.ticks(1)
    h.write(config_document(target=50000))
    h.set_pnl(1.0, 40.0)
    h.ticks(5)
    assert h.broker.orders == []
    assert h.subscribed == [111, 222]
    assert h.index_loads == 1


def test_unknown_contract_makes_the_group_unmonitorable(tmp_path: Path) -> None:
    legs = [leg(SHORT_CE, -150, 100.0, 24500), leg("NIFTY25N2799900CE", 150, 1.0, 99900)]
    h = Harness(tmp_path, config_document(legs=legs))
    h.set_pnl(1.0, 40.0)
    h.ticks(4)
    assert h.actions() == ["GROUP_UNMONITORABLE"]
    assert h.broker.orders == [] and h.subscribed == []


def test_group_for_an_unconfigured_broker_is_unmonitorable(tmp_path: Path) -> None:
    document = config_document()
    document["groups"][0]["broker"] = "MSTOCK"
    h = Harness(tmp_path, document)
    h.ticks(1)
    assert h.actions() == ["GROUP_UNMONITORABLE"]


def test_alert_failures_never_stop_the_watcher(tmp_path: Path) -> None:
    h = Harness(tmp_path)

    def broken(_event):
        raise RuntimeError("queue full")

    h.monitor._alert = broken
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert h.monitor.state.status(h.monitor._watches[next(iter(h.monitor._watches))].group) == "SQUARED_OFF"


def test_thread_runs_and_stops_on_request(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.monitor.poll_seconds = 0.2
    h.monitor.start()
    h.monitor.request_stop()
    h.monitor.join(3)
    assert not h.monitor.is_alive()


def test_tick_exceptions_are_contained(tmp_path: Path, monkeypatch) -> None:
    h = Harness(tmp_path)
    calls = threading.Event()

    def explode():
        calls.set()
        h.monitor.request_stop()
        raise RuntimeError("bad tick")

    monkeypatch.setattr(h.monitor, "tick", explode)
    h.monitor.poll_seconds = 0.2
    h.monitor.start()
    h.monitor.join(3)
    assert calls.is_set() and not h.monitor.is_alive()


def test_alert_events_carry_the_telegram_fields(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    event = h.alerts[0]
    assert event["strategy"] == monitor_module.STRATEGY_LABEL
    assert event["direction"] == "Spread" and event["mode"] == "LIVE" and "pnl" in event


# --- self-review fixes ------------------------------------------------------------
def _held(*rows: tuple[str, str, int]) -> list[PositionDetail]:
    return [PositionDetail(symbol=s, exchange="NFO", product_type=p, quantity=q) for s, p, q in rows]


def test_precheck_reads_only_the_legs_own_product_row() -> None:
    (group,) = parse_config(config_document()).groups
    # NRML short is down to -50; a larger intraday MIS row must not mask that.
    rows = _held((SHORT_CE, "NRML", -50), (SHORT_CE, "MIS", -200), (LONG_CE, "NRML", 150))
    assert any("broker holds -50" in p for p in holdings_mismatch(group, rows))
    rows = _held((SHORT_CE, "MIS", -200), (SHORT_CE, "NRML", -150), (LONG_CE, "NRML", 150))
    assert holdings_mismatch(group, rows) == []


def test_precheck_product_edge_cases() -> None:
    (group,) = parse_config(config_document()).groups
    long_ok = (LONG_CE, "NRML", 150)
    # One unlabelled row (mStock's sample shape) counts as the leg's own.
    assert holdings_mismatch(group, _held((SHORT_CE, "", -150), long_ok)) == []
    # Held only intraday: nothing NRML to close.
    assert any("no NRML" in p for p in holdings_mismatch(group, _held((SHORT_CE, "MIS", -150), long_ok)))
    # Two unlabelled rows: ambiguous, refuse.
    ambiguous = _held((SHORT_CE, "", -150), (SHORT_CE, "", -75), long_ok)
    assert any("several broker rows" in p for p in holdings_mismatch(group, ambiguous))


def test_unplannable_exit_fails_before_any_order(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    broken = dict(CONTRACTS)
    broken[("NSE", "NIFTY", date(2025, 11, 27), 2450000, "CE")] = DhanContract("NSE_FNO", 111, 75, 50)
    h.monitor._contract_index_loader = lambda: broken
    h.set_pnl(1.0, 40.0)
    h.ticks(3)
    assert h.broker.orders == []
    assert h.actions() == ["GROUP_EXIT_FAILED"]
    assert h.state()["status"] == "EXIT_FAILED"
    assert "no exit plan" in h.state()["reason"]


def test_periodic_pnl_log_formats_cleanly(tmp_path: Path, caplog, capsys) -> None:
    h = Harness(tmp_path)
    h.set_pnl(95.0, 40.0)
    with caplog.at_level("INFO", logger=monitor_module.log.name):
        h.ticks(1)
    assert "P&L 750 / target 1,000 [ACTIVE]" in caplog.text
    assert "Logging error" not in capsys.readouterr().err


def test_unmonitorable_alert_is_not_repeated_on_unrelated_edits(tmp_path: Path) -> None:
    legs = [leg(SHORT_CE, -150, 100.0, 24500), leg("NIFTY25N2799900CE", 150, 1.0, 99900)]
    h = Harness(tmp_path, config_document(legs=legs))
    h.ticks(1)
    for target in (2000, 3000):
        h.write(config_document(legs=legs, target=target))
        h.ticks(1)
    assert h.actions() == ["GROUP_UNMONITORABLE"]


def test_removed_group_releases_its_feed_subscriptions(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.ticks(1)
    # Same contracts, different quantities: a new group, but the feed must stay.
    h.write(config_document(legs=[leg(SHORT_CE, -75, 100.0, 24500), leg(LONG_CE, 75, 40.0, 25000)]))
    h.ticks(1)
    assert h.unsubscribed == []
    assert h.live_feed() == {111, 222}
    h.write({"schema_version": 1, "groups": [], "unassigned": []})
    h.ticks(1)
    assert sorted(h.unsubscribed) == [111, 222]
    assert h.live_feed() == set()
