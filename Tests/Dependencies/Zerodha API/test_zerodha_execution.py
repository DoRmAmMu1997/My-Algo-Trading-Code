"""Offline tests for the Zerodha Kite Connect execution adapter.

Everything runs against an in-memory fake HTTP session: no network, no real
credentials, and no order ever leaves the process.

Run from the repository root::

    python -m pytest "Tests/Dependencies/Zerodha API" -q
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest

# The adapter and its diagnostic import each other by bare name, so the SOURCE
# folder -- not this test folder -- goes on sys.path.
# Tests/Dependencies/Zerodha API/<this file> -> repository root is three up.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Dependencies" / "Zerodha API"))
import diagnose_zerodha_symbol as diag
import zerodha_execution as ze
from broker_contract import (
    ExecutionClient,
    OpenOrder,
    OpenPosition,
    OrderStatus,
    PositionDetailClient,
)

INSTRUMENT_CSV = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
    "tick_size,lot_size,instrument_type,segment,exchange\n"
    '1,2,NIFTY25N2724500CE,"NIFTY",0,2025-11-27,24500,0.05,75,CE,NFO-OPT,NFO\n'
    '3,4,NIFTY25N2723500PE,"NIFTY",0,2025-11-27,23500,0.05,75,PE,NFO-OPT,NFO\n'
    '5,6,NIFTY25NOVFUT,"NIFTY",0,2025-11-27,0,0.05,75,FUT,NFO-FUT,NFO\n'
)
HEADER_ONLY_CSV = INSTRUMENT_CSV.split("\n")[0] + "\n"


class FakeResponse:
    def __init__(self, status_code: int = 200, body: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json")
        return self._body


def ok(data: Any) -> FakeResponse:
    return FakeResponse(200, {"status": "success", "data": data})


def error(status_code: int, error_type: str, message: str = "nope") -> FakeResponse:
    return FakeResponse(
        status_code, {"status": "error", "error_type": error_type, "message": message}
    )


class FakeSession:
    """Routes ``(METHOD, path)`` to a response, a list of responses, or an exception."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict | None, dict | None, dict]] = []

    def request(self, method, url, *, params=None, data=None, headers=None, timeout=None):
        path = url.split("api.kite.trade", 1)[-1]
        self.calls.append((method, path, params, data, headers or {}))
        route = self.routes[(method, path)]
        if isinstance(route, list):
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, BaseException):
            raise route
        if callable(route):
            return route()
        return route

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("ZERODHA_API_KEY", "kite_key_123")
    monkeypatch.setenv("ZERODHA_ACCESS_TOKEN", "kite_token_secret_abc")
    monkeypatch.delenv("ZERODHA_API_SECRET", raising=False)
    monkeypatch.setattr(ze, "_FILL_POLL_INTERVAL", 0.0)


def zerodha(routes: dict[tuple[str, str], Any]) -> tuple[Any, FakeSession]:
    base = {
        ("GET", "/user/profile"): ok({"user_id": "AB1234"}),
        ("GET", "/instruments/NFO"): FakeResponse(200, None, INSTRUMENT_CSV),
        ("GET", "/instruments/BFO"): FakeResponse(200, None, HEADER_ONLY_CSV),
    }
    base.update(routes)
    session = FakeSession(base)
    client = ze.ZerodhaExecutionClient()
    client.client = session
    return client, session


def order_row(status: str, filled: int, quantity: int = 75, price: float = 12.5) -> dict:
    return {
        "order_id": "O1",
        "tradingsymbol": "NIFTY25N2724500CE",
        "transaction_type": "BUY",
        "status": status,
        "filled_quantity": filled,
        "quantity": quantity,
        "average_price": price,
        "status_message": "",
    }


# ---------------------------------------------------------------------------
# Shape: the same surface as every other LIVE_BROKER adapter
# ---------------------------------------------------------------------------
def test_client_satisfies_the_shared_execution_contract() -> None:
    client, _ = zerodha({})
    assert isinstance(client, ExecutionClient)
    assert isinstance(client, PositionDetailClient)


def test_module_exposes_one_shared_singleton() -> None:
    assert isinstance(ze.zerodha_execution_client, ze.ZerodhaExecutionClient)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def test_login_needs_an_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ZERODHA_API_KEY")
    client, session = zerodha({})
    assert client.ensure_logged_in() is False
    assert session.calls == []


def test_login_without_token_or_secret_fails_before_any_prompt(monkeypatch) -> None:
    monkeypatch.delenv("ZERODHA_ACCESS_TOKEN")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not prompt"))
    client, session = zerodha({})
    assert client.ensure_logged_in() is False
    assert session.calls == []


def test_login_sends_the_kite_auth_header_and_proves_the_token() -> None:
    client, session = zerodha({})
    assert client.ensure_logged_in() is True
    _method, path, _params, _data, headers = session.calls[0]
    assert path == "/user/profile"
    assert headers["Authorization"] == "token kite_key_123:kite_token_secret_abc"
    assert headers["X-Kite-Version"] == "3"


def test_expired_token_fails_login_without_leaking_it(caplog) -> None:
    client, _ = zerodha(
        {("GET", "/user/profile"): error(403, "TokenException", "bad kite_token_secret_abc")}
    )
    assert client.ensure_logged_in() is False
    assert "kite_token_secret_abc" not in caplog.text


def test_interactive_login_exchanges_the_request_token(monkeypatch, caplog, capsys) -> None:
    monkeypatch.delenv("ZERODHA_ACCESS_TOKEN")
    monkeypatch.setenv("ZERODHA_API_SECRET", "kite_secret_xyz_987")
    opened: list[str] = []
    monkeypatch.setattr(ze.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_: "https://127.0.0.1/?request_token=REQ123&action=login&status=success",
    )
    client, session = zerodha(
        {("POST", "/session/token"): ok({"access_token": "fresh_token_value_1"})}
    )
    assert client.ensure_logged_in() is True
    assert client.current_access_token() == "fresh_token_value_1"
    form = next(call[3] for call in session.calls if call[1] == "/session/token")
    assert form["request_token"] == "REQ123"
    assert form["checksum"] == ze.kite_checksum("kite_key_123", "REQ123", "kite_secret_xyz_987")
    # The login URL carries the API key: it goes to the browser, never to a log.
    assert "api_key=kite_key_123" in opened[0]
    shown = caplog.text + capsys.readouterr().out
    assert "kite_key_123" not in shown and "kite_secret_xyz_987" not in shown
    assert "fresh_token_value_1" not in shown


def test_interactive_login_never_runs_off_the_main_thread(monkeypatch) -> None:
    monkeypatch.delenv("ZERODHA_ACCESS_TOKEN")
    monkeypatch.setenv("ZERODHA_API_SECRET", "kite_secret_xyz_987")
    prompted: list[bool] = []
    monkeypatch.setattr("builtins.input", lambda *_: prompted.append(True) or "REQ123")
    client, session = zerodha({})
    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(client.ensure_logged_in()))
    worker.start()
    worker.join(5)
    assert result == [False]
    assert prompted == [] and session.calls == []


def test_logout_clears_the_token() -> None:
    client, _ = zerodha({})
    assert client.ensure_logged_in()
    client.logout()
    assert client.is_logged_in is False
    assert client._access_token == ""


# ---------------------------------------------------------------------------
# Contract master and resolver
# ---------------------------------------------------------------------------
def test_preload_then_resolve_finds_the_exact_contract() -> None:
    client, _ = zerodha({})
    assert client.preload_scrip_master() is True
    assert (
        client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "CE", 24500.0)
        == "NIFTY25N2724500CE"
    )
    assert client.resolve_option_symbol("nifty", "2025-11-27", "pe", 23500) == "NIFTY25N2723500PE"
    assert client.lot_size_for_symbol("NIFTY25N2724500CE") == 75


def test_resolve_returns_empty_on_a_miss_or_bad_input() -> None:
    client, _ = zerodha({})
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "CE", 99999) == ""
    assert client.resolve_option_symbol("NIFTY", "not-a-date", "CE", 24500) == ""
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "XX", 24500) == ""
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "CE", 24500, "NSE") == ""


def test_failed_master_download_is_not_cached_and_fails_preload() -> None:
    client, session = zerodha({("GET", "/instruments/NFO"): FakeResponse(500, None, "")})
    assert client.preload_scrip_master() is False
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "CE", 24500) == ""
    # Each attempt really retried the download rather than caching the failure.
    assert sum(1 for call in session.calls if call[1] == "/instruments/NFO") == 2


# ---------------------------------------------------------------------------
# Positions and open orders
# ---------------------------------------------------------------------------
POSITION_BOOK = {
    "net": [
        {"tradingsymbol": "NIFTY25N2724500CE", "exchange": "NFO",
         "product": "NRML", "quantity": -75, "average_price": 120.5},
        {"tradingsymbol": "NIFTY25N2723500PE", "exchange": "NFO",
         "product": "NRML", "quantity": 0, "average_price": 0},
        {"tradingsymbol": "INFY", "exchange": "NSE",
         "product": "CNC", "quantity": 10, "average_price": 0},
    ],
    "day": [],
}


def test_position_details_carry_price_and_identity_and_skip_flat_rows() -> None:
    client, _ = zerodha({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    assert client.preload_scrip_master()
    result = client.list_position_details()
    assert result.is_indeterminate is False
    short_call, equity = result.items
    assert short_call.quantity == -75 and short_call.average_price == 120.5
    assert (short_call.underlying, short_call.expiry, short_call.strike, short_call.right) == (
        "NIFTY", date(2025, 11, 27), 24500.0, "CE",
    )
    assert (short_call.exchange, short_call.product_type) == ("NFO", "NRML")
    # Not an option the master knows: present, but with no identity.
    assert equity.right == "" and equity.expiry is None


def test_position_details_never_download_the_master_on_their_own() -> None:
    """The monitor calls this at the moment a target is hit: no slow download there."""

    client, session = zerodha({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    (short_call, _equity) = client.list_position_details().items
    assert short_call.right == "" and short_call.strike is None
    assert not any(call[1].startswith("/instruments") for call in session.calls)


def test_open_positions_use_the_shared_reconciliation_row() -> None:
    client, _ = zerodha({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    result = client.list_open_positions()
    assert result.is_indeterminate is False
    assert all(isinstance(item, OpenPosition) for item in result.items)
    assert [(p.symbol, p.quantity, p.product_type) for p in result.items] == [
        ("NIFTY25N2724500CE", -75, "NRML"),
        ("INFY", 10, "CNC"),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"net": [{"tradingsymbol": "X", "exchange": "NFO", "quantity": 7.5}]},
        {"net": ["garbage"]},
        {"day": []},
        None,
    ],
)
def test_malformed_position_books_are_indeterminate_not_empty(payload) -> None:
    client, _ = zerodha({("GET", "/portfolio/positions"): ok(payload)})
    for query in (client.list_position_details, client.list_open_positions):
        result = query()
        assert result.is_indeterminate is True
        assert result.items == ()


def test_position_transport_failure_is_indeterminate() -> None:
    client, _ = zerodha({("GET", "/portfolio/positions"): ConnectionError("reset")})
    assert client.list_position_details().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_open_orders_keep_only_orders_that_can_still_fill() -> None:
    client, _ = zerodha(
        {
            ("GET", "/orders"): ok(
                [
                    order_row("COMPLETE", 75),
                    order_row("REJECTED", 0),
                    {**order_row("OPEN", 25), "order_id": "O2", "transaction_type": "SELL"},
                ]
            )
        }
    )
    result = client.list_open_orders()
    assert result.is_indeterminate is False
    (pending,) = result.items
    assert isinstance(pending, OpenOrder)
    assert (pending.order_id, pending.side, pending.remaining_quantity) == ("O2", "SELL", 50)


@pytest.mark.parametrize(
    "route",
    [ConnectionError("down"), error(503, "NetworkException"), ok({"not": "a list"}), ok(["x"])],
)
def test_unreadable_order_book_is_indeterminate_not_empty(route) -> None:
    client, _ = zerodha({("GET", "/orders"): route})
    assert client.list_open_orders().is_indeterminate is True


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
def test_market_order_sends_protection_and_a_kite_safe_tag_then_confirms_fill() -> None:
    client, session = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): [
                ok([order_row("OPEN", 0)]),
                ok([order_row("OPEN", 0), order_row("COMPLETE", 75)]),
            ],
        }
    )
    result = client.place_market_order(
        "NIFTY25N2724500CE", "BUY", 75, "NFO", "INTRADAY", order_tag="M2-ABC-12345678-EN01"
    )
    assert result.status is OrderStatus.FILLED
    assert result.average_fill_price == 12.5
    form = next(call[3] for call in session.calls if call[1] == "/orders/regular")
    assert form["market_protection"] == "-1"
    assert form["order_type"] == "MARKET"
    # Friendly INTRADAY becomes Kite's MIS; the hyphenated ledger tag becomes
    # the alphanumeric <= 20 characters Kite accepts.
    assert form["product"] == "MIS"
    assert form["tag"] == "M2ABC12345678EN01"


@pytest.mark.parametrize(("friendly", "kite"), [("NORMAL", "NRML"), ("NRML", "NRML"), ("MIS", "MIS")])
def test_product_types_map_to_kite_codes(friendly, kite) -> None:
    client, session = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ok([order_row("COMPLETE", 75)]),
        }
    )
    client.place_market_order("S", "BUY", 75, "BFO", friendly, order_tag="GM1")
    form = next(call[3] for call in session.calls if call[1] == "/orders/regular")
    assert form["product"] == kite and form["exchange"] == "BFO"


def test_partial_then_cancelled_is_terminal_partial() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ok([order_row("CANCELLED", 25)]),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.PARTIAL
    assert result.filled_quantity == 25


def test_refusal_is_rejected_only_after_the_order_book_proves_no_order() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): error(400, "InputException"),
            ("GET", "/orders"): ok([{"order_id": "OTHER", "tag": "SOMETHING"}]),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED


def test_refusal_with_unreadable_order_book_stays_unknown() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): error(400, "OrderException"),
            ("GET", "/orders"): ConnectionError("down"),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


def test_refusal_without_a_tag_cannot_be_proven_and_stays_unknown() -> None:
    client, _ = zerodha({("POST", "/orders/regular"): error(400, "InputException")})
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML")
    assert result.status is OrderStatus.UNKNOWN


@pytest.mark.parametrize(
    "response",
    [
        error(503, "NetworkException"),
        error(500, "GeneralException"),
        error(429, "NetworkException"),
        FakeResponse(502, None, "<html>bad gateway</html>"),
        ConnectionError("socket closed"),
    ],
)
def test_ambiguous_placement_is_recovered_through_the_tag(response) -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): response,
            # The book stores the tag in the alphanumeric form that was sent.
            ("GET", "/orders"): ok([{"order_id": "O1", "tag": "M2ABC12345678EN01"}]),
            ("GET", "/orders/O1"): ok([order_row("COMPLETE", 75)]),
        }
    )
    result = client.place_market_order(
        "S", "BUY", 75, "NFO", "NRML", order_tag="M2-ABC-12345678-EN01"
    )
    assert result.status is OrderStatus.FILLED


def test_tag_lookup_reads_the_tags_list_too() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): ConnectionError("reset"),
            ("GET", "/orders"): ok([{"order_id": "O1", "tag": None, "tags": ["GM1"]}]),
            ("GET", "/orders/O1"): ok([order_row("COMPLETE", 75)]),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.FILLED


def test_ambiguous_placement_not_found_is_unknown_never_rejected() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): error(503, "NetworkException"),
            ("GET", "/orders"): ok([]),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


def test_quantity_mismatch_in_status_is_unknown() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ok([order_row("COMPLETE", 150, quantity=150)]),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


def test_a_failed_status_read_is_unknown_and_keeps_the_order_id() -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ConnectionError("reset"),
        }
    )
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN
    assert result.order_id == "O1"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("S", "HOLD", 75, "NFO", "NRML"), "side"),
        (("S", "BUY", 0, "NFO", "NRML"), "quantity"),
        (("S", "BUY", 75, "NSE", "NRML"), "exchange"),
        (("S", "BUY", 75, "NFO", "CNC"), "product"),
        (("", "BUY", 75, "NFO", "NRML"), "symbol"),
    ],
)
def test_invalid_order_arguments_raise_before_anything_is_sent(args, message) -> None:
    client, session = zerodha({})
    with pytest.raises(ValueError, match=message):
        client.place_market_order(*args)
    assert session.calls == []


def test_login_failure_is_a_rejection_because_nothing_was_sent(monkeypatch) -> None:
    monkeypatch.delenv("ZERODHA_API_KEY")
    client, session = zerodha({})
    result = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED
    assert session.calls == []


def test_cancel_sends_delete_then_reports_the_resulting_status() -> None:
    client, session = zerodha(
        {
            ("DELETE", "/orders/regular/O1"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ok([order_row("CANCELLED", 0)]),
        }
    )
    result = client.cancel_order("O1", requested_quantity=75)
    assert result.status is OrderStatus.REJECTED and result.broker_state == "CANCELLED"
    assert ("DELETE", "/orders/regular/O1") in [(call[0], call[1]) for call in session.calls]


def test_cancel_transport_failure_is_unknown() -> None:
    client, _ = zerodha({("DELETE", "/orders/regular/O1"): ConnectionError("reset")})
    assert client.cancel_order("O1", requested_quantity=75).status is OrderStatus.UNKNOWN
    with pytest.raises(ValueError):
        client.cancel_order("", requested_quantity=75)


def test_extract_order_id_reads_results_and_payloads() -> None:
    client, _ = zerodha({})
    assert client.extract_order_id({"status": "success", "data": {"order_id": "O9"}}) == "O9"
    assert client.extract_order_id([{"x": 1}, {"order_id": 77}]) == "77"
    assert client.extract_order_id(None) == ""
    assert client.extract_order_id(client.get_order_status("", 0)) == ""


# ---------------------------------------------------------------------------
# Deadlines and poison
# ---------------------------------------------------------------------------
def test_order_deadline_poisons_the_session_until_reconciled(monkeypatch) -> None:
    monkeypatch.setattr(ze, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    release = threading.Event()

    def hang():
        release.wait(5)
        return ok({"order_id": "O1"})

    client, session = zerodha({("POST", "/orders/regular"): hang})
    assert client.ensure_logged_in()
    try:
        started = time.monotonic()
        first = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
        assert time.monotonic() - started < 3
        assert first.status is OrderStatus.UNKNOWN
        posts_before = sum(1 for call in session.calls if call[1] == "/orders/regular")
        second = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM2")
        assert second.status is OrderStatus.UNKNOWN
        assert second.broker_state == "SESSION_POISONED"
        assert sum(1 for call in session.calls if call[1] == "/orders/regular") == posts_before
        # The abandoned call is still running, so recovery must refuse.
        assert client.recover_after_reconciliation() is False
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while not client.recover_after_reconciliation() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.recover_after_reconciliation() is True
    assert client._http_poisoned is False


def test_deadline_includes_waiting_for_the_session_lock(monkeypatch) -> None:
    monkeypatch.setattr(ze, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    client, _ = zerodha({("GET", "/portfolio/positions"): ok({"net": []})})
    assert client.ensure_logged_in()
    outcome: list[BaseException] = []

    def call() -> None:
        # Runs on another thread so the RLock held below really is contended.
        try:
            client._call_api("positions", "GET", "/portfolio/positions")
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    client._lock.acquire()
    try:
        started = time.monotonic()
        worker = threading.Thread(target=call)
        worker.start()
        worker.join(3)
        assert time.monotonic() - started < 3
    finally:
        client._lock.release()
    assert isinstance(outcome[0], TimeoutError)
    assert "lock" in str(outcome[0])


# ---------------------------------------------------------------------------
# The read-only diagnostic (and its typed-YES real-order gate)
# ---------------------------------------------------------------------------
def test_diagnostic_selects_the_nearest_listed_expiry_and_matches_the_resolver() -> None:
    client, _ = zerodha({})
    assert client.preload_scrip_master()
    expiry, symbol, lot_size = diag.select_contract(
        client,
        underlying="NIFTY",
        option_type="CE",
        strike=24500,
        requested_expiry=None,
        today=date(2025, 11, 1),
    )
    assert (expiry, symbol, lot_size) == (date(2025, 11, 27), "NIFTY25N2724500CE", 75)
    assert client.resolve_option_symbol("NIFTY", expiry, "CE", 24500) == symbol
    with pytest.raises(ValueError, match="No "):
        diag.select_contract(
            client, underlying="NIFTY", option_type="CE", strike=1,
            requested_expiry=None, today=date(2025, 11, 1),
        )


def test_diagnostic_sends_no_order_without_a_typed_yes(monkeypatch) -> None:
    client, session = zerodha({})
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    assert diag.place_round_trip_test_order(client, "NIFTY25N2724500CE", 75) is False
    assert session.calls == []


def test_diagnostic_round_trip_buys_then_sells(monkeypatch) -> None:
    client, session = zerodha(
        {
            ("POST", "/orders/regular"): ok({"order_id": "O1"}),
            ("GET", "/orders/O1"): ok([order_row("COMPLETE", 75)]),
        }
    )
    monkeypatch.setattr("builtins.input", lambda *_: "YES")
    assert diag.place_round_trip_test_order(client, "NIFTY25N2724500CE", 75) is True
    sides = [call[3]["transaction_type"] for call in session.calls if call[1] == "/orders/regular"]
    assert sides == ["BUY", "SELL"]


def test_diagnostic_refuses_a_real_order_without_an_explicit_quantity() -> None:
    assert diag.main(["CE", "24500", "--place-order"]) == 2
    assert diag.main(["CE", "24500", "--place-order", "--qty", "0"]) == 2
    assert diag.main(["CE", "24500", "31XYZ99"]) == 2
