"""Tests for the Kite-shaped position clients used by the position-group monitor.

Everything runs against an in-memory fake HTTP session: no network, no real
credentials, no order ever leaves the process.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Any

import pytest

import position_brokers
from broker_contract import OrderStatus, PositionBrokerClient

INSTRUMENT_CSV = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
    "tick_size,lot_size,instrument_type,segment,exchange\n"
    '1,2,NIFTY25N2724500CE,"NIFTY",0,2025-11-27,24500,0.05,75,CE,NFO-OPT,NFO\n'
    '3,4,NIFTY25N2723500PE,"NIFTY",0,2025-11-27,23500,0.05,75,PE,NFO-OPT,NFO\n'
    '5,6,NIFTY25NOVFUT,"NIFTY",0,2025-11-27,0,0.05,75,FUT,NFO-FUT,NFO\n'
)


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
        path = url.split("api.kite.trade", 1)[-1].split("openapi/typea", 1)[-1]
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
    monkeypatch.setattr(position_brokers, "_FILL_POLL_INTERVAL", 0.0)


def zerodha(routes: dict[tuple[str, str], Any]) -> tuple[Any, FakeSession]:
    base = {
        ("GET", "/user/profile"): ok({"user_id": "AB1234"}),
        ("GET", "/instruments/NFO"): FakeResponse(200, None, INSTRUMENT_CSV),
        ("GET", "/instruments/BFO"): FakeResponse(200, None, INSTRUMENT_CSV.split("\n")[0]),
    }
    base.update(routes)
    session = FakeSession(base)
    return position_brokers.ZerodhaPositionClient(session=session), session


def order_row(status: str, filled: int, quantity: int = 75, price: float = 12.5) -> dict:
    return {
        "order_id": "O1",
        "status": status,
        "filled_quantity": filled,
        "quantity": quantity,
        "average_price": price,
        "status_message": "",
    }


def test_client_satisfies_the_position_broker_protocol() -> None:
    client, _ = zerodha({})
    assert isinstance(client, PositionBrokerClient)


def test_login_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ZERODHA_ACCESS_TOKEN")
    client, session = zerodha({})
    assert client.ensure_logged_in() is False
    assert session.calls == []


def test_login_sends_the_kite_auth_header_and_proves_the_token() -> None:
    client, session = zerodha({})
    assert client.ensure_logged_in() is True
    _method, _path, _params, _data, headers = session.calls[0]
    assert headers["Authorization"] == "token kite_key_123:kite_token_secret_abc"
    assert headers["X-Kite-Version"] == "3"


def test_expired_token_fails_login_without_leaking_it(caplog) -> None:
    client, _ = zerodha(
        {("GET", "/user/profile"): error(403, "TokenException", "bad kite_token_secret_abc")}
    )
    assert client.ensure_logged_in() is False
    assert "kite_token_secret_abc" not in caplog.text


def test_positions_carry_price_and_contract_identity_and_skip_flat_rows() -> None:
    client, _ = zerodha(
        {
            ("GET", "/portfolio/positions"): ok(
                {
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
            )
        }
    )
    result = client.list_position_details()
    assert result.is_indeterminate is False
    short_call, equity = result.items
    assert short_call.quantity == -75
    assert short_call.average_price == 120.5
    assert (short_call.underlying, short_call.expiry, short_call.strike, short_call.right) == (
        "NIFTY", date(2025, 11, 27), 24500.0, "CE",
    )
    # Not an option the master knows: present, but with no identity.
    assert equity.right == "" and equity.expiry is None


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
    result = client.list_position_details()
    assert result.is_indeterminate is True
    assert result.items == ()


def test_position_transport_failure_is_indeterminate() -> None:
    client, _ = zerodha({("GET", "/portfolio/positions"): ConnectionError("reset")})
    assert client.list_position_details().is_indeterminate is True


def test_instrument_failure_leaves_positions_unidentified_not_guessed() -> None:
    client, _ = zerodha(
        {
            ("GET", "/instruments/NFO"): FakeResponse(500, None, ""),
            ("GET", "/portfolio/positions"): ok(
                {"net": [{"tradingsymbol": "NIFTY25N2724500CE", "exchange": "NFO",
                          "product": "NRML", "quantity": -75, "average_price": 1}]}
            ),
        }
    )
    (position,) = client.list_position_details().items
    assert position.right == "" and position.strike is None


def test_market_order_sends_protection_and_tag_then_confirms_fill() -> None:
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
        "NIFTY25N2724500CE", "BUY", 75, "NFO", "NRML", order_tag="GM1"
    )
    assert result.status is OrderStatus.FILLED
    assert result.average_fill_price == 12.5
    form = next(call[3] for call in session.calls if call[1] == "/orders/regular")
    assert form["market_protection"] == "-1"
    assert form["order_type"] == "MARKET" and form["product"] == "NRML"
    assert form["tag"] == "GM1"


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


@pytest.mark.parametrize(
    "response",
    [
        error(503, "NetworkException"),
        error(500, "GeneralException"),
        FakeResponse(502, None, "<html>bad gateway</html>"),
        ConnectionError("socket closed"),
    ],
)
def test_ambiguous_placement_is_recovered_through_the_tag(response) -> None:
    client, _ = zerodha(
        {
            ("POST", "/orders/regular"): response,
            ("GET", "/orders"): ok([{"order_id": "O1", "tag": "GM1"}]),
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


def test_order_deadline_poisons_the_client_and_blocks_further_orders(monkeypatch) -> None:
    monkeypatch.setattr(position_brokers, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    release = threading.Event()

    def hang():
        release.wait(5)
        return ok({"order_id": "O1"})

    client, session = zerodha(
        {
            ("POST", "/orders/regular"): hang,
            ("GET", "/orders"): ConnectionError("down"),
        }
    )
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
    finally:
        release.set()


def test_deadline_includes_waiting_for_the_session_lock(monkeypatch) -> None:
    monkeypatch.setattr(position_brokers, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    client, _ = zerodha({("GET", "/portfolio/positions"): ok({"net": []})})
    assert client.ensure_logged_in()
    outcome: list[BaseException] = []

    def call() -> None:
        # Runs on another thread so the RLock held below really is contended.
        try:
            client._request("positions", "GET", "/portfolio/positions")
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


def test_logout_clears_the_token() -> None:
    client, _ = zerodha({})
    assert client.ensure_logged_in()
    client.logout()
    assert client.is_logged_in is False
    assert client._access_token == ""


# ---------------------------------------------------------------------------
# mStock: same shape, different paths
# ---------------------------------------------------------------------------
def mstock(routes: dict[tuple[str, str], Any], monkeypatch) -> tuple[Any, FakeSession]:
    monkeypatch.setenv("MSTOCK_API_KEY", "mstock_key_123")
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "mstock_token_secret")
    base = {
        ("GET", "/portfolio/positions"): ok({"net": []}),
        ("GET", "/instruments/scriptmaster"): FakeResponse(200, None, INSTRUMENT_CSV),
    }
    base.update(routes)
    session = FakeSession(base)
    return position_brokers.MStockPositionClient(session=session), session


def test_mstock_uses_its_own_headers_and_order_status_path(monkeypatch) -> None:
    client, session = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("COMPLETE", 75)]),
        },
        monkeypatch,
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.FILLED
    _method, _path, _params, form, headers = next(
        call for call in session.calls if call[1] == "/orders/regular"
    )
    assert headers["X-Mirae-Version"] == "1"
    assert headers["Authorization"] == "token mstock_key_123:mstock_token_secret"
    assert form["price"] == "0" and "market_protection" not in form
    status_call = next(call for call in session.calls if call[1] == "/order/details")
    assert status_call[2] == {"order_no": "M1", "segment": "D"}


def test_mstock_title_case_rejection_is_still_a_rejection(monkeypatch) -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Rejected", 0)]),
        },
        monkeypatch,
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED


def test_mstock_unrecognized_state_ends_unknown_not_guessed(monkeypatch) -> None:
    monkeypatch.setattr(position_brokers, "_FILL_TIMEOUT_SECONDS", 0.05)
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Pending", 0)]),
        },
        monkeypatch,
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


def test_mstock_tag_lookup_reads_list_shaped_tags(monkeypatch) -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ConnectionError("reset"),
            ("GET", "/orders"): ok([{"order_id": "M1", "tag": ["GM1"]}]),
            ("GET", "/order/details"): ok([order_row("Traded", 75)]),
        },
        monkeypatch,
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.FILLED


def test_mstock_positions_resolve_identity_from_the_scriptmaster(monkeypatch) -> None:
    client, _ = mstock(
        {
            ("GET", "/portfolio/positions"): ok(
                {"net": [{"tradingsymbol": "NIFTY25N2723500PE", "exchange": "NFO",
                          "product": "NRML", "quantity": 75, "average_price": 40}]}
            )
        },
        monkeypatch,
    )
    (position,) = client.list_position_details().items
    assert (position.underlying, position.strike, position.right) == ("NIFTY", 23500.0, "PE")
