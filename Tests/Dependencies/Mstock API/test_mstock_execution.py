"""Offline tests for the mStock (Mirae Asset, Type-A API) execution adapter.

Everything runs against an in-memory fake HTTP session: no network, no real
credentials, and no order ever leaves the process.

Run from the repository root::

    python -m pytest "Tests/Dependencies/Mstock API" -q
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
# Tests/Dependencies/Mstock API/<this file> -> repository root is three up.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Dependencies" / "Mstock API"))
import diagnose_mstock_symbol as diag
import mstock_execution as ms
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
    "1,2,NIFTY25N2724500CE,NIFTY,0,2025-11-27,24500,0.05,75,CE,NFO-OPT,NFO\n"
    "3,4,NIFTY25N2723500PE,NIFTY,0,2025-11-27,23500,0.05,75,PE,NFO-OPT,NFO\n"
    "5,6,SENSEX25N2785000CE,SENSEX,0,2025-11-27,85000,0.05,20,CE,BFO-OPT,BFO\n"
    "7,8,INFY-EQ,INFY,0,,0,0.05,1,EQ,NSE,NSE\n"
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
        status_code,
        {"status": "error", "error_type": error_type, "message": message, "data": None},
    )


class FakeSession:
    """Routes ``(METHOD, path)`` to a response, a list of responses, or an exception."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict | None, dict | None, dict]] = []

    def request(self, method, url, *, params=None, data=None, headers=None, timeout=None):
        path = url.split("openapi/typea", 1)[-1]
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
    monkeypatch.setenv("MSTOCK_API_KEY", "mstock_key_123")
    monkeypatch.setenv("MSTOCK_ACCESS_TOKEN", "mstock_token_secret")
    monkeypatch.delenv("MSTOCK_USERNAME", raising=False)
    monkeypatch.delenv("MSTOCK_TOTP_ENABLED", raising=False)
    monkeypatch.setattr(ms, "_FILL_POLL_INTERVAL", 0.0)


def mstock(routes: dict[tuple[str, str], Any]) -> tuple[Any, FakeSession]:
    base = {
        ("GET", "/portfolio/positions"): ok({"net": [], "day": None}),
        ("GET", "/instruments/scriptmaster"): FakeResponse(200, None, INSTRUMENT_CSV),
    }
    base.update(routes)
    session = FakeSession(base)
    client = ms.MstockExecutionClient()
    client.client = session
    return client, session


def order_row(status: str, filled: int, quantity: int = 75, price: float = 12.5) -> dict:
    return {
        "order_id": "M1",
        "tradingsymbol": "NIFTY25N2724500CE",
        "transaction_type": "SELL",
        "status": status,
        "filled_quantity": filled,
        "quantity": quantity,
        "average_price": price,
        "status_message": "",
        "tag": [],
    }


# ---------------------------------------------------------------------------
# Shape and login
# ---------------------------------------------------------------------------
def test_client_satisfies_the_shared_execution_contract() -> None:
    client, _ = mstock({})
    assert isinstance(client, ExecutionClient)
    assert isinstance(client, PositionDetailClient)
    assert isinstance(ms.mstock_execution_client, ms.MstockExecutionClient)


def test_login_sends_the_mirae_headers_and_proves_the_token() -> None:
    client, session = mstock({})
    assert client.ensure_logged_in() is True
    _method, path, _params, _data, headers = session.calls[0]
    assert path == "/portfolio/positions"
    assert headers["X-Mirae-Version"] == "1"
    assert headers["Authorization"] == "token mstock_key_123:mstock_token_secret"


def test_login_needs_an_api_key(monkeypatch) -> None:
    monkeypatch.delenv("MSTOCK_API_KEY")
    client, session = mstock({})
    assert client.ensure_logged_in() is False
    assert session.calls == []


def test_login_without_token_or_username_fails_before_any_prompt(monkeypatch) -> None:
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setattr(ms, "getpass", lambda *_: pytest.fail("must not prompt"))
    client, session = mstock({})
    assert client.ensure_logged_in() is False
    assert session.calls == []


@pytest.mark.parametrize(
    ("totp_enabled", "exchange_path", "expected_form"),
    [
        ("false", "/session/token",
         {"api_key": "mstock_key_123", "request_token": "654321", "checksum": "L"}),
        ("true", "/session/verifytotp", {"api_key": "mstock_key_123", "totp": "654321"}),
    ],
)
def test_interactive_login_uses_password_then_otp_or_totp(
    monkeypatch, caplog, capsys, totp_enabled, exchange_path, expected_form
) -> None:
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setenv("MSTOCK_USERNAME", "trader01")
    monkeypatch.setenv("MSTOCK_TOTP_ENABLED", totp_enabled)
    monkeypatch.setattr(ms, "getpass", lambda *_: "hunter2-password")
    monkeypatch.setattr("builtins.input", lambda *_: "654321")
    client, session = mstock(
        {
            ("POST", "/connect/login"): ok({"ugid": "x", "flag": 0}),
            ("POST", exchange_path): ok({"access_token": "fresh.jwt.token_1"}),
        }
    )
    assert client.ensure_logged_in() is True
    assert client.current_access_token() == "fresh.jwt.token_1"
    login_form = next(call[3] for call in session.calls if call[1] == "/connect/login")
    assert login_form == {"username": "trader01", "password": "hunter2-password"}
    assert next(call[3] for call in session.calls if call[1] == exchange_path) == expected_form
    shown = caplog.text + capsys.readouterr().out
    assert "hunter2-password" not in shown and "fresh.jwt.token_1" not in shown


def test_refused_password_login_never_echoes_the_password(monkeypatch, caplog) -> None:
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setenv("MSTOCK_USERNAME", "trader01")
    monkeypatch.setattr(ms, "getpass", lambda *_: "hunter2-password")
    client, _ = mstock(
        # mStock answers some refusals with HTTP 200 and status "error".
        {("POST", "/connect/login"): error(200, "MiraeException", "bad hunter2-password")}
    )
    assert client.ensure_logged_in() is False
    assert "hunter2-password" not in caplog.text


def test_interactive_login_never_runs_off_the_main_thread(monkeypatch) -> None:
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setenv("MSTOCK_USERNAME", "trader01")
    prompted: list[bool] = []
    monkeypatch.setattr(ms, "getpass", lambda *_: prompted.append(True) or "pw")
    client, session = mstock({})
    result: list[bool] = []
    worker = threading.Thread(target=lambda: result.append(client.ensure_logged_in()))
    worker.start()
    worker.join(5)
    assert result == [False]
    assert prompted == [] and session.calls == []


def test_logout_clears_the_token() -> None:
    client, _ = mstock({})
    assert client.ensure_logged_in()
    client.logout()
    assert client.is_logged_in is False and client._access_token == ""


# ---------------------------------------------------------------------------
# Contract master, positions, open orders
# ---------------------------------------------------------------------------
def test_one_scriptmaster_serves_both_option_exchanges() -> None:
    client, session = mstock({})
    assert client.preload_scrip_master() is True
    assert (
        client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "PE", 23500)
        == "NIFTY25N2723500PE"
    )
    assert (
        client.resolve_option_symbol("SENSEX", "2025-11-27", "CE", 85000, "BFO")
        == "SENSEX25N2785000CE"
    )
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "CE", 1) == ""
    assert client.lot_size_for_symbol("SENSEX25N2785000CE") == 20
    assert sum(1 for call in session.calls if call[1] == "/instruments/scriptmaster") == 1


def test_failed_master_download_is_retried_not_cached() -> None:
    client, session = mstock({("GET", "/instruments/scriptmaster"): FakeResponse(503, None, "")})
    assert client.preload_scrip_master() is False
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "PE", 23500) == ""
    assert sum(1 for call in session.calls if call[1] == "/instruments/scriptmaster") == 2


POSITION_BOOK = {
    "net": [
        # mStock's own documented sample leaves ``product`` blank.
        {"tradingsymbol": "NIFTY25N2723500PE", "exchange": "NFO",
         "product": "", "quantity": 75, "average_price": 40},
        {"tradingsymbol": "NIFTY25N2724500CE", "exchange": "NFO",
         "product": "NRML", "quantity": 0, "average_price": 0},
    ],
    "day": None,
}


def test_position_details_resolve_identity_from_the_cached_master() -> None:
    client, _ = mstock({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    assert client.preload_scrip_master()
    (position,) = client.list_position_details().items
    assert (position.underlying, position.strike, position.right) == ("NIFTY", 23500.0, "PE")
    assert position.product_type == "" and position.average_price == 40


def test_position_details_never_download_the_master_on_their_own() -> None:
    client, session = mstock({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    (position,) = client.list_position_details().items
    assert position.right == ""
    assert not any(call[1].startswith("/instruments") for call in session.calls)


def test_open_positions_and_their_failure_modes() -> None:
    client, _ = mstock({("GET", "/portfolio/positions"): ok(POSITION_BOOK)})
    (position,) = client.list_open_positions().items
    assert isinstance(position, OpenPosition)
    assert (position.symbol, position.quantity) == ("NIFTY25N2723500PE", 75)

    for payload in ({"net": ["garbage"]}, {"day": []}, None):
        broken, _ = mstock({("GET", "/portfolio/positions"): [ok({"net": []}), ok(payload)]})
        assert broken.list_open_positions().is_indeterminate is True
        assert broken.list_position_details().is_indeterminate is True


def test_open_orders_keep_only_orders_that_can_still_fill() -> None:
    client, _ = mstock(
        {
            ("GET", "/orders"): ok(
                [
                    order_row("Traded", 75),
                    order_row("Rejected", 0),
                    {**order_row("Pending", 25), "order_id": "M2"},
                ]
            )
        }
    )
    (pending,) = client.list_open_orders().items
    assert isinstance(pending, OpenOrder)
    assert (pending.order_id, pending.side, pending.remaining_quantity) == ("M2", "SELL", 50)
    down, _ = mstock({("GET", "/orders"): ConnectionError("down")})
    assert down.list_open_orders().is_indeterminate is True


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
def test_market_order_uses_price_zero_and_the_details_endpoint() -> None:
    client, session = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Traded", 75)]),
        }
    )
    result = client.place_market_order(
        "NIFTY25N2724500CE", "SELL", 75, "NFO", "NORMAL", order_tag="M2-ABC-12345678-XN01"
    )
    assert result.status is OrderStatus.FILLED
    form = next(call[3] for call in session.calls if call[1] == "/orders/regular")
    assert form["price"] == "0" and "market_protection" not in form
    assert form["product"] == "NRML" and form["order_type"] == "MARKET"
    assert form["tag"] == "M2ABC12345678XN01"
    status_call = next(call for call in session.calls if call[1] == "/order/details")
    # mStock's docs disagree on where these go (query vs. body), so both carry them.
    assert status_call[2] == {"order_no": "M1", "segment": "D"}
    assert status_call[3] == {"order_no": "M1", "segment": "D"}


def test_title_case_rejection_is_still_a_rejection() -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Rejected", 0)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED


def test_a_terminal_row_wins_wherever_it_sits_in_the_history() -> None:
    """The docs do not say whether the history is oldest-first or newest-first."""

    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Traded", 75), order_row("Pending", 0)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.FILLED


def test_unrecognized_state_ends_unknown_not_guessed(monkeypatch) -> None:
    monkeypatch.setattr(ms, "_FILL_TIMEOUT_SECONDS", 0.05)
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Pending", 0)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN and result.order_id == "M1"


def test_tag_lookup_reads_list_shaped_tags() -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ConnectionError("reset"),
            ("GET", "/orders"): ok([{"order_id": "M1", "tag": ["GM1"]}]),
            ("GET", "/order/details"): ok([order_row("Traded", 75)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.FILLED


def test_refusal_is_rejected_only_after_the_order_book_proves_no_order() -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): error(400, "InputException"),
            ("GET", "/orders"): ok([{"order_id": "OTHER", "tag": []}]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED


@pytest.mark.parametrize(
    "response",
    [
        # An "error" body on HTTP 200 does not prove what the server did.
        error(200, "InputException", "System is not connected"),
        error(500, "GeneralException"),
        FakeResponse(502, None, "<html>bad gateway</html>"),
    ],
)
def test_ambiguous_placement_not_found_is_unknown_never_rejected(response) -> None:
    client, _ = mstock({("POST", "/orders/regular"): response, ("GET", "/orders"): ok([])})
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
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
    client, session = mstock({})
    with pytest.raises(ValueError, match=message):
        client.place_market_order(*args)
    assert session.calls == []


def test_partial_then_cancelled_is_terminal_partial() -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Cancelled", 25)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.PARTIAL and result.filled_quantity == 25


def test_quantity_mismatch_in_status_is_unknown() -> None:
    client, _ = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Traded", 150, quantity=150)]),
        }
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


@pytest.mark.parametrize(
    "details", [ConnectionError("reset"), error(500, "GeneralException"), ok([]), ok("garbage")]
)
def test_a_failed_status_read_is_unknown_and_keeps_the_order_id(details) -> None:
    client, _ = mstock(
        {("POST", "/orders/regular"): ok({"order_id": "M1"}), ("GET", "/order/details"): details}
    )
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN and result.order_id == "M1"


def test_refusal_with_unreadable_order_book_stays_unknown() -> None:
    for book in (ConnectionError("down"), ok({"not": "a list"}), ok(["x"])):
        client, _ = mstock(
            {("POST", "/orders/regular"): error(400, "MiraeException"), ("GET", "/orders"): book}
        )
        result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
        assert result.status is OrderStatus.UNKNOWN


def test_an_acknowledgement_without_an_order_id_is_not_a_fill() -> None:
    client, _ = mstock({("POST", "/orders/regular"): ok({}), ("GET", "/orders"): ok([])})
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.UNKNOWN


def test_login_failure_is_a_rejection_because_nothing_was_sent(monkeypatch) -> None:
    monkeypatch.delenv("MSTOCK_API_KEY")
    client, session = mstock({})
    result = client.place_market_order("S", "SELL", 75, "NFO", "NRML", order_tag="GM1")
    assert result.status is OrderStatus.REJECTED and session.calls == []
    assert client.list_open_orders().is_indeterminate is True
    assert client.cancel_order("M1", requested_quantity=75).status is OrderStatus.UNKNOWN
    assert client.preload_scrip_master() is False
    assert client.resolve_option_symbol("NIFTY", date(2025, 11, 27), "PE", 23500) == ""


@pytest.mark.parametrize(("friendly", "code"), [("INTRADAY", "MIS"), ("MIS", "MIS"), ("NRML", "NRML")])
def test_product_types_map_to_broker_codes(friendly, code) -> None:
    client, session = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Traded", 75)]),
        }
    )
    client.place_market_order("S", "BUY", 75, "BFO", friendly, order_tag="GM1")
    form = next(call[3] for call in session.calls if call[1] == "/orders/regular")
    assert form["product"] == code and form["exchange"] == "BFO"


def test_expired_token_fails_login_without_leaking_it(caplog) -> None:
    client, _ = mstock(
        {("GET", "/portfolio/positions"): error(401, "TokenException", "bad mstock_token_secret")}
    )
    assert client.ensure_logged_in() is False
    assert "mstock_token_secret" not in caplog.text


@pytest.mark.parametrize("typed", [("", "654321"), ("hunter2-password", "")])
def test_an_empty_password_or_code_aborts_the_login(monkeypatch, typed) -> None:
    password, code = typed
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setenv("MSTOCK_USERNAME", "trader01")
    monkeypatch.setattr(ms, "getpass", lambda *_: password)
    monkeypatch.setattr("builtins.input", lambda *_: code)
    client, session = mstock({("POST", "/connect/login"): ok({"flag": 0})})
    assert client.ensure_logged_in() is False
    assert not any(call[1].startswith("/session") for call in session.calls)


def test_a_code_exchange_without_a_token_fails_the_login(monkeypatch) -> None:
    monkeypatch.delenv("MSTOCK_ACCESS_TOKEN")
    monkeypatch.setenv("MSTOCK_USERNAME", "trader01")
    monkeypatch.setattr(ms, "getpass", lambda *_: "hunter2-password")
    monkeypatch.setattr("builtins.input", lambda *_: "654321")
    client, _ = mstock(
        {("POST", "/connect/login"): ok({"flag": 0}), ("POST", "/session/token"): ok({"other": 1})}
    )
    assert client.ensure_logged_in() is False
    assert client.current_access_token() == ""


def test_cancel_transport_failure_is_unknown() -> None:
    client, _ = mstock({("DELETE", "/orders/regular/M1"): ConnectionError("reset")})
    assert client.cancel_order("M1", requested_quantity=75).status is OrderStatus.UNKNOWN


def test_cancel_sends_delete_then_reports_the_resulting_status() -> None:
    client, session = mstock(
        {
            ("DELETE", "/orders/regular/M1"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Cancelled", 0)]),
        }
    )
    result = client.cancel_order("M1", requested_quantity=75)
    assert result.status is OrderStatus.REJECTED and result.broker_state == "CANCELLED"
    assert ("DELETE", "/orders/regular/M1") in [(call[0], call[1]) for call in session.calls]
    with pytest.raises(ValueError):
        client.cancel_order("", requested_quantity=75)


def test_extract_order_id_reads_results_and_payloads() -> None:
    client, _ = mstock({})
    assert client.extract_order_id({"status": "success", "data": {"order_id": "M9"}}) == "M9"
    assert client.extract_order_id(None) == ""
    assert client.extract_order_id(client.get_order_status("", 0)) == ""


# ---------------------------------------------------------------------------
# Deadlines and poison
# ---------------------------------------------------------------------------
def test_order_deadline_poisons_the_session_until_reconciled(monkeypatch) -> None:
    monkeypatch.setattr(ms, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    release = threading.Event()

    def hang():
        release.wait(5)
        return ok({"order_id": "M1"})

    client, session = mstock({("POST", "/orders/regular"): hang})
    assert client.ensure_logged_in()
    try:
        first = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM1")
        assert first.status is OrderStatus.UNKNOWN
        posts_before = sum(1 for call in session.calls if call[1] == "/orders/regular")
        second = client.place_market_order("S", "BUY", 75, "NFO", "NRML", order_tag="GM2")
        assert second.broker_state == "SESSION_POISONED"
        assert sum(1 for call in session.calls if call[1] == "/orders/regular") == posts_before
        assert client.recover_after_reconciliation() is False
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while not client.recover_after_reconciliation() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.recover_after_reconciliation() is True


def test_deadline_includes_waiting_for_the_session_lock(monkeypatch) -> None:
    monkeypatch.setattr(ms, "_BROKER_CALL_DEADLINE_SECONDS", 0.2)
    client, _ = mstock({})
    assert client.ensure_logged_in()
    outcome: list[BaseException] = []

    def call() -> None:
        try:
            client._call_api("positions", "GET", "/portfolio/positions")
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    client._lock.acquire()
    try:
        worker = threading.Thread(target=call)
        worker.start()
        worker.join(3)
    finally:
        client._lock.release()
    assert isinstance(outcome[0], TimeoutError) and "lock" in str(outcome[0])


# ---------------------------------------------------------------------------
# The read-only diagnostic (and its typed-YES real-order gate)
# ---------------------------------------------------------------------------
def test_diagnostic_selects_the_nearest_listed_expiry_and_matches_the_resolver() -> None:
    client, _ = mstock({})
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
    client, session = mstock({})
    monkeypatch.setattr("builtins.input", lambda *_: "yes")
    assert diag.place_round_trip_test_order(client, "NIFTY25N2724500CE", 75) is False
    assert session.calls == []


def test_diagnostic_round_trip_buys_then_sells(monkeypatch) -> None:
    client, session = mstock(
        {
            ("POST", "/orders/regular"): ok({"order_id": "M1"}),
            ("GET", "/order/details"): ok([order_row("Traded", 75)]),
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
