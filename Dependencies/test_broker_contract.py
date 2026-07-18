"""Focused MAT-101 tests for the broker-neutral execution contract.

These tests stay entirely in memory.  They never construct a real broker
session, enable a live-trading flag, or make a network request.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import sys
import threading
import time
import zipfile
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _NeoApiTestDouble:
    """Import-only stand-in; behavioral tests inject their own SDK clients."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


# Kotak is an optional live dependency and is intentionally absent from the
# core CI environment.  Install only the import surface before any adapter or
# diagnostic module loads; every behavioral test replaces ``client`` with its
# own broker-shaped fake, so no SDK behavior is being simulated here.
_neo_api_test_module = ModuleType("neo_api_client")
_neo_api_test_module.NeoAPI = _NeoApiTestDouble
sys.modules["neo_api_client"] = _neo_api_test_module


def _load_file_module(name: str, relative_path: str) -> ModuleType:
    """Load a broker helper whose parent folder contains spaces."""

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _contract_module():
    """Import the wished-for MAT-101 contract with a clear RED failure."""

    try:
        return importlib.import_module("Dependencies.broker_contract")
    except ModuleNotFoundError:
        pytest.fail("MAT-101 shared broker contract module is missing")


def test_order_contract_is_exact_immutable_and_quantity_safe() -> None:
    """The shared result cannot hide new statuses or impossible quantities."""

    contract = _contract_module()

    assert {status.name for status in contract.OrderStatus} == {
        "REJECTED",
        "FILLED",
        "PARTIAL",
        "UNKNOWN",
    }

    result = contract.OrderResult(
        order_id="ORDER-1",
        requested_quantity=75,
        filled_quantity=25,
        remaining_quantity=50,
        status=contract.OrderStatus.PARTIAL,
        broker_state="OPEN",
        reason="25 units filled; 50 still open",
    )
    with pytest.raises(FrozenInstanceError):
        result.filled_quantity = 75

    with pytest.raises(ValueError, match="remaining_quantity"):
        contract.OrderResult(
            order_id="BROKEN",
            requested_quantity=75,
            filled_quantity=25,
            remaining_quantity=75,
            status=contract.OrderStatus.UNKNOWN,
            broker_state="malformed",
            reason="quantities disagree",
        )


@pytest.mark.parametrize(
    ("status_name", "filled"),
    [
        ("FILLED", 0),
        ("REJECTED", 25),
        ("PARTIAL", 0),
        ("PARTIAL", 75),
    ],
)
def test_order_result_rejects_semantically_contradictory_statuses(
    status_name: str,
    filled: int,
) -> None:
    contract = _contract_module()

    with pytest.raises(ValueError, match="status"):
        contract.OrderResult(
            order_id="CONTRADICTION",
            requested_quantity=75,
            filled_quantity=filled,
            remaining_quantity=75 - filled,
            status=contract.OrderStatus[status_name],
            broker_state=status_name,
            reason="contradictory direct construction",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_quantity", True, "requested_quantity"),
        ("requested_quantity", 75.5, "requested_quantity"),
        ("filled_quantity", -1, "filled_quantity"),
        ("remaining_quantity", "50", "remaining_quantity"),
        ("status", "FILLED", "OrderStatus"),
    ],
)
def test_order_result_rejects_malformed_direct_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    contract = _contract_module()
    kwargs: dict[str, object] = {
        "order_id": "BROKEN",
        "requested_quantity": 75,
        "filled_quantity": 25,
        "remaining_quantity": 50,
        "status": contract.OrderStatus.PARTIAL,
        "broker_state": "OPEN",
        "reason": "malformed direct construction",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        contract.OrderResult(**kwargs)


def test_order_result_rejects_fill_larger_than_request() -> None:
    contract = _contract_module()
    with pytest.raises(ValueError, match="cannot exceed"):
        contract.OrderResult(
            order_id="BROKEN",
            requested_quantity=75,
            filled_quantity=100,
            remaining_quantity=0,
            status=contract.OrderStatus.UNKNOWN,
            broker_state="OPEN",
            reason="impossible fill",
        )


@pytest.mark.parametrize(
    ("broker_state", "filled", "expected"),
    [
        ("REJECTED", 0, "REJECTED"),
        ("COMPLETE", 75, "FILLED"),
        ("COMPLETE", 25, "PARTIAL"),
        ("REJECTED", 25, "PARTIAL"),
        ("OPEN", 0, "UNKNOWN"),
        ("", 0, "UNKNOWN"),
    ],
)
def test_normalization_is_conservative_and_preserves_acknowledgement(
    broker_state: str,
    filled: int,
    expected: str,
) -> None:
    """Only explicit terminal evidence may become FILLED or REJECTED."""

    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ACK-123",
        requested_quantity=75,
        filled_quantity=filled,
        broker_state=broker_state,
        reason="broker detail",
    )

    assert result.status.name == expected
    assert result.order_id == "ACK-123"
    assert result.remaining_quantity == 75 - filled


def test_malformed_quantity_normalizes_to_unknown_instead_of_rejected() -> None:
    """Bad broker data must not manufacture a harmless zero-fill rejection."""

    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ACK-LOST",
        requested_quantity=75,
        filled_quantity="not-a-number",
        broker_state="REJECTED",
        reason="malformed reply",
    )

    assert result.status is contract.OrderStatus.UNKNOWN
    assert result.filled_quantity == 0
    assert result.remaining_quantity == 75
    assert "malformed" in result.reason.lower()


@pytest.mark.parametrize(
    ("requested", "filled"),
    [
        (True, 0),
        (-1, 0),
        ("75.5", 0),
        (75, True),
        (75, -1),
        (75, 76),
    ],
)
def test_quantity_normalization_rejects_nonexact_or_impossible_values(
    requested: object,
    filled: object,
) -> None:
    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id=None,
        requested_quantity=requested,
        filled_quantity=filled,
        broker_state=None,
    )
    assert result.status is contract.OrderStatus.UNKNOWN
    assert result.order_id == ""
    assert "malformed" in result.reason.lower()


@pytest.mark.parametrize(
    ("state", "filled", "expected", "reason_fragment"),
    [
        ("PARTIAL", 25, "PARTIAL", "partial fill"),
        ("PARTIAL", 0, "UNKNOWN", "terminal order outcome"),
        ("COMPLETE", 75, "FILLED", "requested quantity filled"),
        ("REJECTED", 0, "REJECTED", "terminal rejection"),
    ],
)
def test_normalization_supplies_status_specific_default_reasons(
    state: str,
    filled: int,
    expected: str,
    reason_fragment: str,
) -> None:
    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ORDER",
        requested_quantity=75,
        filled_quantity=filled,
        broker_state=state,
    )
    assert result.status.name == expected
    assert reason_fragment in result.reason.lower()


def test_open_snapshot_types_reuse_contract_invariants() -> None:
    contract = _contract_module()
    with pytest.raises(ValueError, match="remaining_quantity"):
        contract.OpenOrder(
            order_id="OPEN-1",
            symbol="NIFTY",
            side="BUY",
            requested_quantity=75,
            filled_quantity=25,
            remaining_quantity=75,
            broker_state="OPEN",
        )
    with pytest.raises(ValueError, match="integer"):
        contract.OpenPosition(
            symbol="NIFTY",
            quantity=True,
            product_type="MIS",
        )


def test_indeterminate_query_uses_safe_defaults() -> None:
    contract = _contract_module()
    result = contract.BrokerQueryResult.indeterminate(" ", broker_state=" timed-out ")
    assert result.items == ()
    assert result.is_indeterminate is True
    assert result.reason == "Broker query outcome is indeterminate."
    assert result.broker_state == "timed-out"


def test_query_failure_cannot_masquerade_as_a_successful_empty_list() -> None:
    """Callers can distinguish a truly empty book from a timed-out query."""

    contract = _contract_module()
    empty = contract.BrokerQueryResult.success([])
    timed_out = contract.BrokerQueryResult.indeterminate("broker timed out")

    assert empty.items == ()
    assert empty.is_indeterminate is False
    assert timed_out.items == ()
    assert timed_out.is_indeterminate is True
    assert timed_out.reason == "broker timed out"


def test_execution_client_protocol_covers_reconciliation_surface() -> None:
    """The shared protocol includes both old helpers and MAT-102 hooks."""

    contract = _contract_module()

    class CompleteFakeClient:
        is_logged_in = True

        def ensure_logged_in(self):
            return True

        def preload_scrip_master(self):
            return True

        def resolve_option_symbol(self, underlying, expiry, option_type, strike):
            return "NIFTY-OPTION"

        def place_market_order(
            self,
            symbol,
            side,
            quantity,
            exchange_segment,
            product_type,
            *,
            order_tag="",
        ):
            return None

        def get_order_status(self, order_id, requested_quantity=0):
            return None

        def cancel_order(self, order_id, requested_quantity=0):
            return None

        def list_open_orders(self):
            return None

        def list_open_positions(self):
            return None

        def recover_after_reconciliation(self):
            return True

        def extract_order_id(self, order_response):
            return ""

        def logout(self):
            return {}

    assert isinstance(CompleteFakeClient(), contract.ExecutionClient)


@pytest.fixture(scope="module")
def flattrade_module() -> ModuleType:
    """Return a private, network-free copy of the Flattrade adapter."""

    return _load_file_module(
        "mat101_flattrade_execution",
        "Dependencies/Flattrade API/flattrade_execution.py",
    )


def _ready_flattrade_client(flattrade_module: ModuleType, monkeypatch):
    """Build an authenticated-looking client without touching Flattrade."""

    client = flattrade_module.FlattradeExecutionClient()
    client._client_id = "TEST"
    client._account_id = "TEST"
    client._access_token = "not-a-real-token"
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


@pytest.mark.parametrize(
    ("status_row", "expected_status", "filled", "remaining"),
    [
        (
            {"stat": "Ok", "status": "REJECTED", "fillshares": "0", "qty": "75"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_flattrade_place_order_normalizes_terminal_outcomes(
    flattrade_module: ModuleType,
    monkeypatch,
    status_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    """An acknowledgement is followed by a typed fill snapshot."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter([{"stat": "Ok", "norenordno": "FT-1"}, [status_row]])
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "FT-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_flattrade_mid_fill_open_snapshot_polls_to_completion(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A market order caught mid-fill (state OPEN) is transient, not terminal.

    Returning that first snapshot as PARTIAL would freeze every live entry
    over a healthy order; the loop must poll again and report the full fill.
    """

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_FILL_POLL_INTERVAL", 0.001)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-MIDFILL"},
            [{"stat": "Ok", "status": "OPEN", "fillshares": "25", "qty": "75"}],
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.order_id == "FT-MIDFILL"
    assert result.filled_quantity == 75
    assert result.remaining_quantity == 0


def test_flattrade_partial_then_cancelled_is_terminal_partial(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A partial fill followed by a cancel can never fill further.

    The terminal broker label makes the very first snapshot the final
    outcome; no further polling is needed (or possible -- the reply iterator
    holds exactly one status response).
    """

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-PARTCXL"},
            [{"stat": "Ok", "status": "CANCELED", "fillshares": "25", "qty": "75"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "PARTIAL"
    assert result.filled_quantity == 25
    assert result.remaining_quantity == 50


def test_flattrade_transmits_execution_ledger_tag(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    captured_payloads = []
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-TAG"},
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )

    def post(_endpoint, payload, **_kwargs):
        captured_payloads.append(dict(payload))
        return next(replies)

    monkeypatch.setattr(client, "_post_api", post)
    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert captured_payloads[0]["remarks"] == "M2-A1B2-4F8D2Q7J-NE1"


def test_flattrade_zero_broker_quantity_cannot_confirm_a_fill(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-ZERO-QTY"},
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "0"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_flattrade_acknowledgement_then_response_loss_is_unknown(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The known order id survives when its status response is lost."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-LOST"},
            TimeoutError("SingleOrdHist response lost"),
        ]
    )

    def post(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(client, "_post_api", post)
    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "FT-LOST"
    assert "lost" in result.reason.lower()


def test_flattrade_cancel_and_reconciliation_queries_are_typed(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Cancel, open orders, and positions expose explicit query outcomes."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "result": "FT-1"},
            [{"stat": "Ok", "status": "CANCELLED", "fillshares": "0", "qty": "75"}],
            [
                {
                    "stat": "Ok",
                    "norenordno": "FT-OPEN",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "OPEN",
                    "fillshares": "25",
                    "qty": "75",
                },
                {
                    "stat": "Ok",
                    "norenordno": "FT-REJECTED",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "REJECTED",
                    "fillshares": "0",
                    "qty": "75",
                },
                {
                    "stat": "Ok",
                    "norenordno": "FT-CANCELLED-PARTIAL",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "CANCELLED",
                    "fillshares": "25",
                    "qty": "75",
                },
            ],
            [{"stat": "Ok", "tsym": "NIFTY", "netqty": "50", "prd": "I"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    cancelled = client.cancel_order("FT-1", requested_quantity=75)
    open_orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert open_orders.is_indeterminate is False
    assert len(open_orders.items) == 1
    assert open_orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == 50


def test_flattrade_query_timeout_is_indeterminate_not_empty(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A timeout never returns a successful empty order or position book."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("deadline")),
    )

    status = client.get_order_status("FT-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert status.status.name == "UNKNOWN"
    assert orders.is_indeterminate is True
    assert positions.is_indeterminate is True


def test_flattrade_successful_empty_queries_are_distinct_from_failure(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Two genuine empty books are successful, not indeterminate."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter([[], []])
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_flattrade_only_exact_no_data_envelope_proves_empty_books(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A session error containing the words 'no data' must fail closed."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Not_Ok", "emsg": "No Data"},
            {"stat": "Not_Ok", "emsg": "No Data"},
            {"stat": "Not_Ok", "emsg": "No data because session expired"},
            {"stat": "Not_Ok", "emsg": "No data because session expired"},
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    empty_orders = client.list_open_orders()
    empty_positions = client.list_open_positions()
    failed_orders = client.list_open_orders()
    failed_positions = client.list_open_positions()

    assert empty_orders.items == () and empty_orders.is_indeterminate is False
    assert empty_positions.items == () and empty_positions.is_indeterminate is False
    assert failed_orders.is_indeterminate is True
    assert failed_positions.is_indeterminate is True


def test_flattrade_http_constants_are_exactly_ten_seconds(
    flattrade_module: ModuleType,
) -> None:
    """Both authenticated API calls and the scrip download share the deadline."""

    assert flattrade_module._API_TIMEOUT_SECONDS == 10.0
    assert flattrade_module._SCRIP_MASTER_TIMEOUT_SECONDS == 10.0
    assert flattrade_module._BROKER_CALL_DEADLINE_SECONDS == 10.0


def test_flattrade_total_deadline_includes_lock_wait(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Waiting for the shared session lock cannot exceed the broker budget."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"stat": "Ok"}

    class _Session:
        def post(self, *args, **kwargs):
            return _Response()

    client.client = _Session()
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with client._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="lock"):
            client._post_api("UserDetails", {"uid": "TEST"})
    finally:
        release.set()
        holder.join(timeout=1)
    assert time.monotonic() - started < 0.25


def test_flattrade_total_deadline_includes_rate_limiter_lock_wait(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A stuck limiter mutex cannot postpone HTTP dispatch past the deadline."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("HTTP must not start after limiter deadline")

    session = _Session()
    client.client = session
    locked = threading.Event()
    release = threading.Event()

    def hold_limiter_lock() -> None:
        with client._api_limiter._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_limiter_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="limiter lock"):
            client._post_api("UserDetails", {"uid": "TEST"})
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert session.calls == 0


def test_flattrade_total_deadline_poison_blocks_calls_after_slow_response(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A slow-dribble request returns on time and cannot overlap a new call."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)
    release = threading.Event()

    class _Response:
        def __init__(self, payload) -> None:
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class _SlowSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                release.wait(0.4)
                return _Response({"stat": "Ok"})
            if "GetPendingOrders" in str(args[0]):
                return _Response([])
            return _Response({"stat": "Ok"})

    session = _SlowSession()
    client.client = session
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            client._post_api("PlaceOrder", {"uid": "TEST"}, is_order=False)
        assert time.monotonic() - started < 0.25
        with pytest.raises(RuntimeError, match="indeterminate"):
            client._post_api("PlaceOrder", {"uid": "TEST"}, is_order=False)
        assert session.calls == 1
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not client._timed_out_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.list_open_orders().is_indeterminate is False
    with pytest.raises(RuntimeError, match="indeterminate"):
        client._post_api("PlaceOrder", {"uid": "TEST"})
    assert client.recover_after_reconciliation() is True
    assert client._post_api("PlaceOrder", {"uid": "TEST"}) == {"stat": "Ok"}


def test_flattrade_status_timeout_cannot_release_a_queued_entry(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The placement gate stays held until the first fill status is known."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(client._order_limiter, "acquire", lambda *_args, **_kwargs: None)
    confirm_started = threading.Event()
    second_submitted = threading.Event()
    release_history = threading.Event()

    class _Response:
        def __init__(self, payload) -> None:
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class _RaceSession:
        def __init__(self) -> None:
            self.place_calls = 0
            self._lock = threading.Lock()

        def post(self, url, *args, **kwargs):
            if "PlaceOrder" in str(url):
                with self._lock:
                    self.place_calls += 1
                    call_number = self.place_calls
                if call_number == 2:
                    second_submitted.set()
                return _Response(
                    {"stat": "Ok", "norenordno": f"FT-{call_number}"}
                )
            if "SingleOrdHist" in str(url):
                release_history.wait(0.5)
                return _Response(
                    [
                        {
                            "stat": "Ok",
                            "status": "COMPLETE",
                            "fillshares": "75",
                            "qty": "75",
                        }
                    ]
                )
            return _Response({"stat": "Ok"})

    session = _RaceSession()
    client.client = session
    original_confirm = client._confirm_fill
    confirm_calls = 0
    confirm_lock = threading.Lock()

    def controlled_confirm(order_id, quantity):
        nonlocal confirm_calls
        with confirm_lock:
            confirm_calls += 1
            call_number = confirm_calls
        if call_number == 1:
            confirm_started.set()
            second_submitted.wait(0.15)
            return original_confirm(order_id, quantity)
        return flattrade_module.normalize_order_result(
            order_id=order_id,
            requested_quantity=quantity,
            filled_quantity=quantity,
            broker_state="COMPLETE",
        )

    monkeypatch.setattr(client, "_confirm_fill", controlled_confirm)
    results = {}

    def place(name: str) -> None:
        results[name] = client.place_market_order("NIFTY", "BUY", 75)

    first = threading.Thread(target=place, args=("first",))
    second = threading.Thread(target=place, args=("second",))
    try:
        first.start()
        assert confirm_started.wait(0.3)
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)
    finally:
        release_history.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert session.place_calls == 1
    assert results["first"].status.name == "UNKNOWN"
    assert results["second"].status.name == "UNKNOWN"


@pytest.fixture(scope="module")
def shoonya_module() -> ModuleType:
    """Return a private, network-free copy of the Shoonya adapter."""

    return _load_file_module(
        "mat101_shoonya_execution",
        "Dependencies/Shoonya API/shoonya_execution.py",
    )


class _FakeNorenClient:
    """Behavioral Shoonya boundary fake; no SDK method reaches the network."""

    def __init__(
        self,
        *,
        place_response=None,
        order_rows=None,
        order_book=None,
        positions=None,
    ) -> None:
        self.place_response = place_response or {"stat": "Ok", "norenordno": "SH-1"}
        self.order_rows = order_rows
        self.order_book = order_book
        self.position_rows = positions
        self.cancel_response = {"stat": "Ok", "result": "SH-1"}
        self.place_calls = 0
        self.last_place_kwargs = None

    def place_order(self, **kwargs):
        self.place_calls += 1
        self.last_place_kwargs = dict(kwargs)
        return self.place_response

    def single_order_history(self, order_id):
        if isinstance(self.order_rows, Exception):
            raise self.order_rows
        return self.order_rows

    def cancel_order(self, order_id):
        if isinstance(self.cancel_response, Exception):
            raise self.cancel_response
        return self.cancel_response

    def get_order_book(self):
        if isinstance(self.order_book, Exception):
            raise self.order_book
        return self.order_book

    def get_positions(self):
        if isinstance(self.position_rows, Exception):
            raise self.position_rows
        return self.position_rows


def _ready_shoonya_client(shoonya_module: ModuleType, monkeypatch, fake):
    client = shoonya_module.ShoonyaExecutionClient()
    client.client = fake
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


@pytest.mark.parametrize(
    ("status_row", "expected_status", "filled", "remaining"),
    [
        (
            {"stat": "Ok", "status": "REJECTED", "fillshares": "0", "qty": "75"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_shoonya_place_order_normalizes_terminal_outcomes(
    shoonya_module: ModuleType,
    monkeypatch,
    status_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    fake = _FakeNorenClient(order_rows=[status_row])
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "SH-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_shoonya_mid_fill_open_snapshot_polls_to_completion(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A mid-fill OPEN snapshot keeps polling until the fill completes."""

    class _SequencedNoren(_FakeNorenClient):
        """History fake that replays a sequence, then repeats the last reply."""

        def __init__(self, sequence) -> None:
            super().__init__()
            self._sequence = list(sequence)

        def single_order_history(self, order_id):
            if len(self._sequence) > 1:
                return self._sequence.pop(0)
            return self._sequence[0]

    fake = _SequencedNoren(
        [
            [{"stat": "Ok", "status": "OPEN", "fillshares": "25", "qty": "75"}],
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.filled_quantity == 75
    assert result.remaining_quantity == 0


def test_shoonya_transmits_execution_ledger_tag(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert fake.last_place_kwargs["remarks"] == "M2-A1B2-4F8D2Q7J-NE1"


def test_shoonya_zero_broker_quantity_cannot_confirm_a_fill(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "0"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_shoonya_acknowledgement_then_status_timeout_is_unknown(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(order_rows=TimeoutError("history deadline"))
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    # The pre-MAT-101 implementation polls ambiguous history until its fill
    # window expires. Keep this RED test deterministic and fast.
    monkeypatch.setattr(shoonya_module, "_FILL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(shoonya_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "SH-1"
    assert "deadline" in result.reason.lower()


def test_shoonya_explicit_place_rejection_preserves_broker_evidence(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        place_response={
            "stat": "Not_Ok",
            "norenordno": "SH-REJECTED",
            "emsg": "RMS limit exceeded",
        }
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "REJECTED"
    assert result.order_id == "SH-REJECTED"
    assert result.filled_quantity == 0
    assert "RMS limit exceeded" in result.reason


def test_shoonya_cancel_and_reconciliation_queries_are_typed(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "CANCELLED", "fillshares": "0", "qty": "75"}
        ],
        order_book=[
            {
                "stat": "Ok",
                "norenordno": "SH-OPEN",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "OPEN",
                "fillshares": "25",
                "qty": "75",
            },
            {
                "stat": "Ok",
                "norenordno": "SH-REJECTED",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "REJECTED",
                "fillshares": "0",
                "qty": "75",
            },
            {
                "stat": "Ok",
                "norenordno": "SH-CANCELLED-PARTIAL",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "CANCELLED",
                "fillshares": "25",
                "qty": "75",
            },
        ],
        positions=[{"stat": "Ok", "tsym": "NIFTY", "netqty": "-50", "prd": "I"}],
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    cancelled = client.cancel_order("SH-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert orders.is_indeterminate is False
    assert len(orders.items) == 1
    assert orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == -50


def test_shoonya_query_failures_are_indeterminate_not_empty(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=TimeoutError("status deadline"),
        order_book=TimeoutError("orders deadline"),
        positions=TimeoutError("positions deadline"),
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    assert client.get_order_status("SH-1", 75).status.name == "UNKNOWN"
    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_shoonya_malformed_history_row_normalizes_unknown(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_rows=["malformed-row"]),
    )

    result = client.get_order_status("SH-1", 75)

    assert result.status.name == "UNKNOWN"
    assert "malformed" in result.reason.lower()


def test_shoonya_successful_empty_queries_are_distinct_from_failure(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Native empty lists prove there are no open Shoonya rows."""

    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_book=[], positions=[]),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_shoonya_no_data_envelopes_are_successful_empty_queries(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Shoonya's explicit No Data response is determinate, not a transport loss."""

    no_data = {"stat": "Not_Ok", "emsg": "No Data"}
    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_book=no_data, positions=no_data),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_shoonya_nonempty_error_envelopes_remain_indeterminate(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    for message in ("Session expired", "No data because session expired"):
        error = {"stat": "Not_Ok", "emsg": message}
        client = _ready_shoonya_client(
            shoonya_module,
            monkeypatch,
            _FakeNorenClient(order_book=error, positions=error),
        )

        assert client.list_open_orders().is_indeterminate is True
        assert client.list_open_positions().is_indeterminate is True


def test_shoonya_native_http_timeout_has_a_ten_second_deadline(
    shoonya_module: ModuleType,
) -> None:
    """Both connect and read waits are capped by the MAT-101 deadline."""

    noren_module = sys.modules[shoonya_module.NorenApi.__module__]
    assert noren_module._HTTP_TIMEOUT == (10, 10)
    assert shoonya_module._BROKER_CALL_DEADLINE_SECONDS == 10.0


def test_shoonya_total_deadline_includes_lock_wait(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A queued strategy cannot wait forever for Shoonya's session lock."""

    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with client._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        result = client.place_market_order("NIFTY", "BUY", 75)
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert result.status.name == "UNKNOWN"
    assert "lock" in result.reason.lower()
    assert fake.place_calls == 0


def test_shoonya_total_deadline_poison_blocks_calls_after_slow_response(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Timed-out native work cannot be followed by a second queued order."""

    release = threading.Event()

    class _SlowNoren(_FakeNorenClient):
        def place_order(self, **kwargs):
            self.place_calls += 1
            release.wait(0.4)
            return self.place_response

    fake = _SlowNoren(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    started = time.monotonic()
    try:
        first = client.place_market_order("NIFTY", "BUY", 75)
        assert time.monotonic() - started < 0.25
        second = client.place_market_order("NIFTY", "BUY", 75)
        assert first.status.name == "UNKNOWN"
        assert second.status.name == "UNKNOWN"
        assert "indeterminate" in second.reason.lower()
        assert fake.place_calls == 1
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not client._timed_out_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.get_order_status("SH-1", 75).status.name == "FILLED"
    blocked = client.place_market_order("NIFTY", "BUY", 75)
    assert blocked.status.name == "UNKNOWN"
    assert fake.place_calls == 1
    assert client.recover_after_reconciliation() is True
    recovered = client.place_market_order("NIFTY", "BUY", 75)
    assert recovered.status.name == "FILLED"
    assert fake.place_calls == 2


def test_shoonya_status_timeout_cannot_release_a_queued_entry(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A second strategy cannot submit between acknowledgement and status poison."""

    confirm_started = threading.Event()
    second_submitted = threading.Event()
    release_history = threading.Event()

    class _RaceNoren(_FakeNorenClient):
        def __init__(self) -> None:
            super().__init__()
            self._place_lock = threading.Lock()

        def place_order(self, **kwargs):
            with self._place_lock:
                self.place_calls += 1
                call_number = self.place_calls
            if call_number == 2:
                second_submitted.set()
            return {"stat": "Ok", "norenordno": f"SH-{call_number}"}

        def single_order_history(self, order_id):
            release_history.wait(0.5)
            return [
                {
                    "stat": "Ok",
                    "status": "COMPLETE",
                    "fillshares": "75",
                    "qty": "75",
                }
            ]

    fake = _RaceNoren()
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    original_confirm = client._confirm_fill
    confirm_calls = 0
    confirm_lock = threading.Lock()

    def controlled_confirm(order_id, quantity):
        nonlocal confirm_calls
        with confirm_lock:
            confirm_calls += 1
            call_number = confirm_calls
        if call_number == 1:
            confirm_started.set()
            second_submitted.wait(0.15)
            return original_confirm(order_id, quantity)
        return shoonya_module.normalize_order_result(
            order_id=order_id,
            requested_quantity=quantity,
            filled_quantity=quantity,
            broker_state="COMPLETE",
        )

    monkeypatch.setattr(client, "_confirm_fill", controlled_confirm)
    results = {}

    def place(name: str) -> None:
        results[name] = client.place_market_order("NIFTY", "BUY", 75)

    first = threading.Thread(target=place, args=("first",))
    second = threading.Thread(target=place, args=("second",))
    try:
        first.start()
        assert confirm_started.wait(0.3)
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)
    finally:
        release_history.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert fake.place_calls == 1
    assert results["first"].status.name == "UNKNOWN"
    assert results["second"].status.name == "UNKNOWN"


def test_vendored_noren_preserves_non_ok_payloads(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """The adapter must receive broker rejection/no-data evidence intact."""

    noren_module = sys.modules[shoonya_module.NorenApi.__module__]
    payload = {"stat": "Not_Ok", "emsg": "No Data", "norenordno": "SH-REJ"}

    class _Response:
        text = json.dumps(payload)

    monkeypatch.setattr(noren_module.requests, "post", lambda *args, **kwargs: _Response())
    api = noren_module.NorenApi()
    api._NorenApi__username = "user"
    api._NorenApi__accountid = "account"
    api._NorenApi__susertoken = "token"

    placed = api.place_order("B", "I", "NFO", "NIFTY", 75, 0, "MKT")
    orders = api.get_order_book()
    positions = api.get_positions()

    assert placed == payload
    assert orders == payload
    assert positions == payload


@pytest.fixture(scope="module")
def kotak_module() -> ModuleType:
    """Return a private copy of the Kotak adapter; calls use behavioral fakes."""

    return _load_file_module(
        "mat101_kotak_execution",
        "Dependencies/Kotak API/kotak_execution.py",
    )


def test_kotak_adapter_fixture_does_not_require_optional_sdk(
    kotak_module: ModuleType,
) -> None:
    """Hosted tests use an import-only double instead of an installed SDK."""

    assert kotak_module.NeoAPI is _NeoApiTestDouble


class _FakeKotakSdk:
    """Behavioral Kotak SDK boundary fake with broker-shaped responses."""

    def __init__(self, *, history_row=None, orders=None, positions=None) -> None:
        self.history_row = history_row
        self.orders = orders
        self.position_rows = positions
        self.last_place_kwargs = None

    def place_order(self, **kwargs):
        self.last_place_kwargs = dict(kwargs)
        return {"stat": "Ok", "nOrdNo": "KT-1"}

    def order_history(self, order_id):
        if isinstance(self.history_row, Exception):
            raise self.history_row
        return {"data": {"stat": "Ok", "data": [self.history_row]}}

    def cancel_order(self, order_id):
        return {"stat": "Ok", "nOrdNo": order_id}

    def order_report(self):
        if isinstance(self.orders, Exception):
            raise self.orders
        return {"data": self.orders}

    def positions(self):
        if isinstance(self.position_rows, Exception):
            raise self.position_rows
        return {"data": self.position_rows}


def _ready_kotak_client(kotak_module: ModuleType, monkeypatch, fake):
    client = kotak_module.KotakExecutionClient()
    client.client = fake
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


def test_kotak_transmits_execution_ledger_tag(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert fake.last_place_kwargs["tag"] == "M2-A1B2-4F8D2Q7J-NE1"


@pytest.mark.parametrize(
    ("history_row", "expected_status", "filled", "remaining"),
    [
        (
            {"ordSt": "REJECTED", "fldQty": "0", "qty": "75", "rejRsn": "RMS"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"ordSt": "COMPLETE", "fldQty": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_kotak_place_order_normalizes_terminal_outcomes(
    kotak_module: ModuleType,
    monkeypatch,
    history_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    fake = _FakeKotakSdk(history_row=history_row)
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_FILL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(kotak_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "KT-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_kotak_transient_hand_off_states_poll_to_completion(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """Kotak's routine hand-off states must not end fill confirmation early.

    "put order req received" and "validation pending" are states every healthy
    Kotak order passes through.  Treating them as terminal would report a
    perfectly normal order as UNKNOWN and freeze all live entries.
    """

    class _SequencedKotakSdk(_FakeKotakSdk):
        """History fake that replays a sequence, then repeats the last row."""

        def __init__(self, rows) -> None:
            super().__init__()
            self._rows = list(rows)

        def order_history(self, order_id):
            row = self._rows.pop(0) if len(self._rows) > 1 else self._rows[0]
            return {"data": {"stat": "Ok", "data": [row]}}

    fake = _SequencedKotakSdk(
        [
            {"ordSt": "put order req received", "fldQty": "0", "qty": "75"},
            {"ordSt": "validation pending", "fldQty": "0", "qty": "75"},
            {"ordSt": "complete", "fldQty": "75", "qty": "75"},
        ]
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.filled_quantity == 75
    assert client.session_poisoned is False


def test_kotak_zero_broker_quantity_cannot_confirm_a_fill(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "0"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_kotak_acknowledgement_then_status_loss_preserves_order_id(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The accepted order remains discoverable when order_history is lost."""

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(history_row=TimeoutError("history response lost")),
    )

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "KT-1"
    assert "lost" in result.reason.lower()
    assert client.session_poisoned is False


def test_kotak_cancel_and_reconciliation_queries_are_typed(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "CANCELLED", "fldQty": "0", "qty": "75"},
        orders=[
            {
                "nOrdNo": "KT-OPEN",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "OPEN",
                "fldQty": "25",
                "qty": "75",
            },
            {
                "nOrdNo": "KT-REJECTED",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "REJECTED",
                "fldQty": "0",
                "qty": "75",
            },
            {
                "nOrdNo": "KT-CANCELLED-PARTIAL",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "CANCELLED",
                "fldQty": "25",
                "qty": "75",
            },
        ],
        positions=[{"trdSym": "NIFTY", "netQty": "50", "prod": "MIS"}],
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    cancelled = client.cancel_order("KT-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert orders.is_indeterminate is False
    assert len(orders.items) == 1
    assert orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == 50


def test_kotak_query_failures_are_indeterminate_not_empty(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row=TimeoutError("history deadline"),
        orders=TimeoutError("orders deadline"),
        positions=TimeoutError("positions deadline"),
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    assert client.get_order_status("KT-1", 75).status.name == "UNKNOWN"
    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_kotak_malformed_history_row_normalizes_unknown(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(history_row="malformed-row"),
    )

    result = client.get_order_status("KT-1", 75)

    assert result.status.name == "UNKNOWN"
    assert "malformed" in result.reason.lower()


def test_kotak_error_envelopes_cannot_masquerade_as_empty_books(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A Not_Ok wrapper with data=[] is still a failed reconciliation query."""

    class ErrorEnvelopeKotak(_FakeKotakSdk):
        def order_report(self):
            return {"stat": "Not_Ok", "emsg": "session expired", "data": []}

        def positions(self):
            return {
                "data": {
                    "stat": "Not_Ok",
                    "emsg": "session expired",
                    "data": [],
                }
            }

    client = _ready_kotak_client(kotak_module, monkeypatch, ErrorEnvelopeKotak())

    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_kotak_successful_empty_queries_are_distinct_from_failure(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """Empty Kotak lists are successful snapshots, not query failures."""

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(
            orders=[],
            positions=[{"trdSym": "NIFTY", "netQty": 0, "prod": "MIS"}],
        ),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_kotak_sdk_deadline_and_executor_are_fixed(
    kotak_module: ModuleType,
) -> None:
    """The SDK boundary is ten seconds and physically single-threaded."""

    client = kotak_module.KotakExecutionClient()
    assert kotak_module._BROKER_DEADLINE_SECONDS == 10.0
    assert client._sdk_executor._max_workers == 1


def test_kotak_scrip_master_download_has_total_deadline(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A slow-dribble CSV body cannot hold the broker lock indefinitely."""

    started = threading.Event()
    release = threading.Event()

    class ScripKotak(_FakeKotakSdk):
        def scrip_master(self, exchange_segment):
            return "https://example.invalid/nfo.csv"

    class _SlowResponse:
        text = (
            "pExpiryDate,pSymbolName,pOptionType,dStrikePrice;,pTrdSymbol\n"
            "0,NIFTY,CE,2250000,NIFTY-TEST\n"
        )

        def raise_for_status(self):
            return None

    def slow_get(*args, **kwargs):
        started.set()
        release.wait(0.5)
        return _SlowResponse()

    client = _ready_kotak_client(kotak_module, monkeypatch, ScripKotak())
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(kotak_module.requests, "get", slow_get)
    results = []

    worker = threading.Thread(target=lambda: results.append(client.preload_scrip_master()))
    try:
        worker.start()
        assert started.wait(0.3)
        worker.join(timeout=0.15)
        assert not worker.is_alive()
    finally:
        release.set()
        worker.join(timeout=1)

    assert results == [False]
    assert client.session_poisoned is True


def test_kotak_order_submission_gate_has_total_deadline(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A caller queued behind another order must receive a bounded UNKNOWN."""

    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.02, raising=False)
    locked = threading.Event()
    release = threading.Event()

    def hold_gate() -> None:
        with client._order_submission_lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_gate)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        result = client.place_market_order("NIFTY", "BUY", 75)
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert result.status.name == "UNKNOWN"
    assert result.broker_state == "ORDER_GATE_TIMEOUT"
    assert "submission gate" in result.reason.lower()


def test_kotak_timeout_poisoning_prevents_overlapping_later_orders(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A timed-out SDK thread remains isolated and blocks new submissions."""

    release = threading.Event()

    class BlockingKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__(
                history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
            )
            self.active = 0
            self.max_active = 0
            self.place_calls = 0
            self.guard = threading.Lock()

        def place_order(self, **kwargs):
            with self.guard:
                self.place_calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                release.wait(timeout=1.0)
                return {"stat": "Ok", "nOrdNo": "KT-SLOW"}
            finally:
                with self.guard:
                    self.active -= 1

    fake = BlockingKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.02, raising=False)

    try:
        first = client.place_market_order("NIFTY", "BUY", 75)
        second = client.place_market_order("NIFTY", "BUY", 75)

        assert first.status.name == "UNKNOWN"
        assert second.status.name == "UNKNOWN"
        assert client.session_poisoned is True
        assert fake.place_calls == 1
        assert fake.max_active == 1
    finally:
        release.set()
        time.sleep(0.03)


def test_kotak_concurrent_submission_cannot_queue_behind_a_timed_out_order(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The poison check and SDK submit are atomic across strategy threads."""

    started = threading.Event()
    release = threading.Event()

    class BlockingKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__()
            self.place_calls = 0
            self.guard = threading.Lock()

        def place_order(self, **kwargs):
            with self.guard:
                self.place_calls += 1
            started.set()
            release.wait(timeout=1.0)
            return {"stat": "Ok", "nOrdNo": f"KT-{self.place_calls}"}

    fake = BlockingKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.1)
    results = []

    def submit_order() -> None:
        results.append(client.place_market_order("NIFTY", "BUY", 75))

    first = threading.Thread(target=submit_order)
    second = threading.Thread(target=submit_order)
    try:
        first.start()
        assert started.wait(timeout=0.5)
        second.start()
        first.join(timeout=0.5)
        second.join(timeout=0.5)
        release.set()
        time.sleep(0.05)

        assert len(results) == 2
        assert all(result.status.name == "UNKNOWN" for result in results)
        assert client.session_poisoned is True
        assert fake.place_calls == 1
    finally:
        release.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)


def test_kotak_status_timeout_cannot_release_a_queued_entry(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The entry gate covers acknowledgement through final fill confirmation."""

    history_started = threading.Event()
    release = threading.Event()

    class BlockingHistoryKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__()
            self.place_calls = 0

        def place_order(self, **kwargs):
            self.place_calls += 1
            return {"stat": "Ok", "nOrdNo": f"KT-{self.place_calls}"}

        def order_history(self, order_id):
            history_started.set()
            release.wait(timeout=1.0)
            return {
                "data": {
                    "stat": "Ok",
                    "data": [
                        {"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
                    ],
                }
            }

    fake = BlockingHistoryKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.05)
    results = []

    def submit_order() -> None:
        results.append(client.place_market_order("NIFTY", "BUY", 75))

    first = threading.Thread(target=submit_order)
    second = threading.Thread(target=submit_order)
    try:
        first.start()
        assert history_started.wait(timeout=0.5)
        second.start()
        first.join(timeout=0.5)
        second.join(timeout=0.5)
        release.set()
        time.sleep(0.05)

        assert len(results) == 2
        assert all(result.status.name == "UNKNOWN" for result in results)
        assert client.session_poisoned is True
        assert fake.place_calls == 1
    finally:
        release.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)


def test_kotak_recovery_waits_for_every_abandoned_sdk_future(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A cancelled queued timeout cannot hide an earlier still-running call."""

    started = threading.Event()
    release = threading.Event()
    client = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.03)

    def blocking_call():
        started.set()
        release.wait(timeout=1.0)
        return "first"

    try:
        with pytest.raises(kotak_module._SdkDeadlineExceeded):
            client._sdk_call("first", blocking_call)
        assert started.is_set()
        first_future = client._timed_out_future
        assert first_future is not None and not first_future.done()

        with pytest.raises(kotak_module._SdkDeadlineExceeded):
            client._sdk_call("queued", lambda: "second")
        latest_future = client._timed_out_future
        assert latest_future is not None
        latest_future.cancel()

        assert client.recover_after_reconciliation() is False
        assert not first_future.done()
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not first_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.recover_after_reconciliation() is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, None),
        ("bad", None),
        (1.5, None),
        (float("inf"), None),
        ("75", 75),
    ],
)
def test_broker_quantity_helpers_require_exact_finite_integers(
    shoonya_module: ModuleType,
    kotak_module: ModuleType,
    flattrade_module: ModuleType,
    value: object,
    expected: int | None,
) -> None:
    """Each adapter rejects rounded or boolean quantities at its own boundary."""

    assert shoonya_module._exact_int(value) == expected
    assert kotak_module._exact_int(value) == expected
    assert flattrade_module._exact_int(value) == expected


def test_shoonya_login_success_and_missing_credentials_fail_closed(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Login accepts only a typed Ok response and never keeps a half-session."""

    captured: dict[str, object] = {}

    class LoginNoren:
        def login(self, **kwargs):
            captured.update(kwargs)
            return {"stat": "Ok", "uname": "Test Trader"}

    for name, value in {
        "SHOONYA_USERID": "USER",
        "SHOONYA_PASSWORD": "PASSWORD",
        "SHOONYA_VENDOR_CODE": "VENDOR",
        "SHOONYA_API_SECRET": "SECRET",
        "SHOONYA_TOTP_SECRET": "TOTP-SEED",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(shoonya_module, "NorenApi", LoginNoren)
    client = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(client, "_generate_totp", lambda _secret: "123456")

    assert client.ensure_logged_in() is True
    assert client.ensure_logged_in() is True
    assert client.is_logged_in is True
    assert captured["twoFA"] == "123456"

    for name in (
        "SHOONYA_USERID",
        "SHOONYA_PASSWORD",
        "SHOONYA_VENDOR_CODE",
        "SHOONYA_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    missing = shoonya_module.ShoonyaExecutionClient()
    assert missing.ensure_logged_in() is False
    assert missing.client is None


def test_shoonya_blank_totp_and_rejected_login_clear_session(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    class RejectedNoren:
        def login(self, **_kwargs):
            return {"stat": "Not_Ok", "emsg": "bad credentials"}

    for name, value in {
        "SHOONYA_USERID": "USER",
        "SHOONYA_PASSWORD": "PASSWORD",
        "SHOONYA_VENDOR_CODE": "VENDOR",
        "SHOONYA_API_SECRET": "SECRET",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SHOONYA_TOTP_SECRET", raising=False)

    blank = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(blank, "_prompt_totp", lambda: "")
    assert blank.ensure_logged_in() is False

    monkeypatch.setenv("SHOONYA_TOTP_SECRET", "SEED")
    monkeypatch.setattr(shoonya_module, "NorenApi", RejectedNoren)
    rejected = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(rejected, "_generate_totp", lambda _secret: "123456")
    assert rejected.ensure_logged_in() is False
    assert rejected.client is None


def test_shoonya_scrip_master_and_symbol_resolution_are_cached(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """The official master validates exact contracts without later downloads."""

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "NFO_symbols.txt",
            "Exchange,Token,LotSize,Symbol,TradingSymbol\n"
            "NFO,1,75,NIFTY,NIFTY16JUL26C22500\n",
        )

    class Response:
        content = archive.getvalue()

        @staticmethod
        def raise_for_status() -> None:
            return None

    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(),
    )
    monkeypatch.setattr(shoonya_module.requests, "get", lambda *args, **kwargs: Response())

    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    symbol = client.resolve_option_symbol("nifty", date(2026, 7, 16), "ce", 22500)
    assert symbol == "NIFTY16JUL26C22500"
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500) == symbol
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", None, "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "not-a-date", "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "XX", 22500) == ""


def test_shoonya_logout_and_recursive_order_id_extraction(
    shoonya_module: ModuleType,
) -> None:
    client = shoonya_module.ShoonyaExecutionClient()
    assert client.logout()["State"] == "NOT_OK"
    assert client.extract_order_id({"data": [{"norenordno": "SH-NESTED"}]}) == "SH-NESTED"
    assert client.extract_order_id([{"none": 0}, " SH-STRING "]) == "SH-STRING"
    assert client.extract_order_id(123) == ""

    class LogoutNoren:
        @staticmethod
        def logout():
            return "bye"

    client.client = LogoutNoren()
    client.is_logged_in = True
    assert client.logout() == {"State": "OK", "message": "bye"}
    assert client.client is None
    assert client.is_logged_in is False


def test_kotak_login_success_normalizes_mobile_and_validates_two_factor(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Configuration:
        edit_token = "trade-token"
        edit_sid = "trade-session"
        serverId = "server"
        base_url = "https://example.invalid"
        data_center = "dc"

    class LoginNeo:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs
            self.configuration = Configuration()

        def totp_login(self, **kwargs):
            captured["login"] = kwargs
            return {"stat": "Ok"}

        def totp_validate(self, **kwargs):
            captured["validate"] = kwargs
            return {"stat": "Ok"}

    for name, value in {
        "CONSUMER_KEY": "CONSUMER",
        "MOBILE": "98765-43210",
        "MPIN": "123456",
        "UCC": "UCC123",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("COSNSUMER_KEY", raising=False)
    monkeypatch.setattr(kotak_module, "NeoAPI", LoginNeo)
    client = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(client, "_prompt_totp", lambda: "654321")

    assert client.ensure_logged_in() is True
    assert client.ensure_logged_in() is True
    assert captured["login"]["mobile_number"] == "+919876543210"
    assert captured["validate"] == {"mpin": "123456"}
    assert kotak_module.KotakExecutionClient._normalize_mobile("919876543210") == "+919876543210"
    assert kotak_module.KotakExecutionClient._normalize_mobile("+919876543210") == "+919876543210"
    assert kotak_module.KotakExecutionClient._normalize_mobile("unexpected") == "unexpected"


def test_kotak_login_rejects_missing_credentials_and_incomplete_two_factor(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    for name in ("COSNSUMER_KEY", "CONSUMER_KEY", "MOBILE", "MPIN", "UCC"):
        monkeypatch.delenv(name, raising=False)
    assert kotak_module.KotakExecutionClient().ensure_logged_in() is False

    class Configuration:
        edit_token = ""
        edit_sid = ""

    class IncompleteNeo:
        def __init__(self, **_kwargs):
            self.configuration = Configuration()

        @staticmethod
        def totp_login(**_kwargs):
            return {"stat": "Ok"}

        @staticmethod
        def totp_validate(**_kwargs):
            return {"stat": "Not_Ok"}

    for name, value in {
        "CONSUMER_KEY": "CONSUMER",
        "MOBILE": "9876543210",
        "MPIN": "123456",
        "UCC": "UCC123",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(kotak_module, "NeoAPI", IncompleteNeo)
    incomplete = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(incomplete, "_prompt_totp", lambda: "654321")
    assert incomplete.ensure_logged_in() is False
    assert incomplete.client is None


def test_kotak_scrip_master_resolution_and_miss_diagnostics(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    expiry = date(2026, 7, 16)
    encoded_expiry = int(
        (pd.Timestamp(expiry) - pd.to_timedelta(315511200, unit="s")).timestamp()
    )

    class ScripKotak(_FakeKotakSdk):
        @staticmethod
        def scrip_master(exchange_segment):
            assert exchange_segment == "nse_fo"
            return "https://example.invalid/nfo.csv"

    class Response:
        text = (
            "pExpiryDate,pSymbolName,pOptionType,dStrikePrice;,pTrdSymbol\n"
            f"{encoded_expiry},NIFTY,CE,2250000,NIFTY26JUL22500CE\n"
        )

        @staticmethod
        def raise_for_status() -> None:
            return None

    client = _ready_kotak_client(kotak_module, monkeypatch, ScripKotak())
    monkeypatch.setattr(kotak_module.requests, "get", lambda *args, **kwargs: Response())

    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    symbol = client.resolve_option_symbol("nifty", expiry, "ce", 22500)
    assert symbol == "NIFTY26JUL22500CE"
    assert client.resolve_option_symbol("NIFTY", expiry, "CE", 22500) == symbol
    assert client.resolve_option_symbol("NIFTY", expiry, "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", None, "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "not-a-date", "CE", 22500) == ""
    assert kotak_module.KotakExecutionClient._match_in_df(None, "NIFTY", "CE", "", 1) == ""


def test_kotak_logout_and_recursive_order_id_extraction(
    kotak_module: ModuleType,
) -> None:
    client = kotak_module.KotakExecutionClient()
    assert client.logout()["State"] == "NOT_OK"
    assert client.extract_order_id({"data": [{"nOrdNo": "KT-NESTED"}]}) == "KT-NESTED"
    assert client.extract_order_id([{"none": 0}, " KT-STRING "]) == "KT-STRING"
    assert client.extract_order_id(123) == ""

    class LogoutKotak:
        @staticmethod
        def logout():
            return "bye"

    client.client = LogoutKotak()
    client.is_logged_in = True
    assert client.logout() == {"State": "OK", "message": "bye"}
    assert client.client is None


def _flattrade_master_fixture() -> pd.DataFrame:
    """Return one valid and one filtered-out Flattrade catalogue row."""

    return pd.DataFrame(
        {
            "Exchange": ["NFO", "NSE"],
            "Lotsize": ["75", "1"],
            "Symbol": ["NIFTY", "NIFTY"],
            "Tradingsymbol": ["NIFTY16JUL26C22500", "NIFTY-SPOT"],
            "Expiry": ["16-JUL-2026", "16-JUL-2026"],
            "Strike": ["22500.00", "0"],
            "Optiontype": ["CE", "XX"],
        }
    )


def test_flattrade_env_parsing_session_validation_and_login_fast_path(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAT_TEST_QUOTED", "' value '")
    monkeypatch.setenv("MAT_TEST_BAD_INT", "bad")
    monkeypatch.setenv("MAT_TEST_NEGATIVE", "-1")
    assert flattrade_module._env_str("MAT_TEST_QUOTED") == "value"
    assert flattrade_module._env_str("MAT_TEST_MISSING", "fallback") == "fallback"
    assert flattrade_module._env_non_negative_int("MAT_TEST_BAD_INT", 3) == 3
    assert flattrade_module._env_non_negative_int("MAT_TEST_NEGATIVE", 3) == 3

    client = flattrade_module.FlattradeExecutionClient()
    client._client_id = "CLIENT"
    client._access_token = "TOKEN"
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *_args, **_kwargs: {
            "stat": "Ok",
            "actid": "ACCOUNT",
            "exarr": ["NFO"],
        },
    )
    assert client._validate_session_locked() is True
    assert client._account_id == "ACCOUNT"

    monkeypatch.setenv("FLATTRADE_CLIENT_ID", "CLIENT")
    monkeypatch.setenv("FLATTRADE_ACCESS_TOKEN", "TOKEN")
    login = flattrade_module.FlattradeExecutionClient()
    monkeypatch.setattr(login, "_validate_session_locked", lambda: True)
    assert login.ensure_logged_in() is True
    assert login.ensure_logged_in() is True


def test_flattrade_browser_token_exchange_is_validated_before_login(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The interactive flow hashes the secret and accepts only a validated token."""

    captured: dict[str, object] = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, str]:
            return {"status": "Ok", "token": "DAILY-TOKEN", "client": "CLIENT"}

    class Session:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            captured["timeout"] = kwargs["timeout"]
            return Response()

    for name, value in {
        "FLATTRADE_CLIENT_ID": "CLIENT",
        "FLATTRADE_API_KEY": "API-KEY",
        "FLATTRADE_API_SECRET": "API-SECRET",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FLATTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "REQUEST-CODE")
    monkeypatch.setattr(flattrade_module.webbrowser, "open", lambda _url: True)

    client = flattrade_module.FlattradeExecutionClient()
    monkeypatch.setattr(client, "_ensure_session_locked", lambda: Session())
    monkeypatch.setattr(client, "_validate_session_locked", lambda: True)

    assert client.ensure_logged_in() is True
    assert client._access_token == "DAILY-TOKEN"
    expected_digest = flattrade_module.hashlib.sha256(
        b"API-KEYREQUEST-CODEAPI-SECRET"
    ).hexdigest()
    assert captured["json"] == {
        "api_key": "API-KEY",
        "request_code": "REQUEST-CODE",
        "api_secret": expected_digest,
    }


def test_flattrade_login_rejects_missing_identity_or_authorization_inputs(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    for name in (
        "FLATTRADE_CLIENT_ID",
        "FLATTRADE_ACCESS_TOKEN",
        "FLATTRADE_API_KEY",
        "FLATTRADE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False

    monkeypatch.setenv("FLATTRADE_CLIENT_ID", "CLIENT")
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False

    monkeypatch.setenv("FLATTRADE_API_KEY", "API-KEY")
    monkeypatch.setenv("FLATTRADE_API_SECRET", "API-SECRET")
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False


def test_flattrade_downloads_and_caches_the_official_scrip_master(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    csv_text = _flattrade_master_fixture().to_csv(index=False)

    class Response:
        text = csv_text

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Session:
        calls = 0

        def get(self, _url, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    client.client = session
    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    assert session.calls == 1
    assert client._scrip_df is not None
    assert len(client._scrip_df) == 1


def test_flattrade_scrip_preparation_resolution_and_logout(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        flattrade_module.FlattradeExecutionClient._prepare_scrip_master(pd.DataFrame())
    unusable = _flattrade_master_fixture()
    unusable["Exchange"] = "NSE"
    with pytest.raises(ValueError, match="no usable NFO"):
        flattrade_module.FlattradeExecutionClient._prepare_scrip_master(unusable)

    prepared = flattrade_module.FlattradeExecutionClient._prepare_scrip_master(
        _flattrade_master_fixture()
    )
    assert len(prepared) == 1
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    client._scrip_df = prepared
    assert (
        client.resolve_option_symbol("nifty", date(2026, 7, 16), "ce", 22500)
        == "NIFTY16JUL26C22500"
    )
    assert (
        client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500)
        == "NIFTY16JUL26C22500"
    )
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "bad-date", "CE", "bad") == ""
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500, "NSE") == ""

    class Session:
        closed = False

        def close(self):
            self.closed = True

    session = Session()
    client.client = session
    result = client.logout()
    assert result["stat"] == "Ok"
    assert session.closed is True
    assert client.is_logged_in is False
    assert client._access_token == ""


_DIAGNOSTIC_PATHS = (
    "Dependencies/Kotak API/diagnose_kotak_symbol.py",
    "Dependencies/Shoonya API/diagnose_shoonya_symbol.py",
    "Dependencies/Flattrade API/diagnose_flattrade_symbol.py",
)


@pytest.fixture(params=_DIAGNOSTIC_PATHS, ids=("kotak", "shoonya", "flattrade"))
def diagnostic_module(request, monkeypatch) -> ModuleType:
    """Load one CLI diagnostic without letting pytest flags become CLI input."""

    monkeypatch.setattr(sys, "argv", [str(ROOT / request.param)])
    module_name = "mat101_diagnostic_" + request.param.split("/")[-2].lower().replace(" ", "_")
    return _load_file_module(module_name, request.param)


def test_diagnostics_never_default_a_real_order_to_a_stale_lot_size(
    diagnostic_module: ModuleType,
) -> None:
    """Legacy Kotak/Shoonya diagnostics require an explicit current quantity."""

    if hasattr(diagnostic_module, "QTY"):
        assert diagnostic_module.QTY is None


class _DiagnosticClient:
    """Return a scripted sequence of typed results without contacting a broker."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, int]] = []

    def place_market_order(self, *, symbol, side, quantity, **kwargs):
        self.calls.append((symbol, side, quantity))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def extract_order_id(result) -> str:
        return str(getattr(result, "order_id", ""))


def _diagnostic_result(status_name: str, *, order_id: str = "DIAG-1"):
    """Build one internally consistent result for diagnostic behavior tests."""

    contract = _contract_module()
    status = contract.OrderStatus[status_name]
    filled = {
        "FILLED": 75,
        "PARTIAL": 25,
        "REJECTED": 0,
        "UNKNOWN": 0,
    }[status_name]
    return contract.OrderResult(
        order_id=order_id,
        requested_quantity=75,
        filled_quantity=filled,
        remaining_quantity=75 - filled,
        status=status,
        broker_state=status_name,
        reason=f"diagnostic {status_name.lower()} result",
    )


@pytest.mark.parametrize("status_name", ("PARTIAL", "UNKNOWN"))
def test_diagnostics_report_ambiguous_entry_as_indeterminate(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
    status_name: str,
) -> None:
    """An ambiguous BUY must never trigger a SELL or a no-position claim."""

    client = _DiagnosticClient([_diagnostic_result(status_name)])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert client.calls == [("NIFTY", "BUY", 75)]
    assert "indeterminate" in output.lower()
    assert "No position opened" not in output


def test_diagnostics_reserve_no_position_claim_for_explicit_rejection(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """Only a typed zero-fill rejection proves that the BUY opened nothing."""

    client = _DiagnosticClient([_diagnostic_result("REJECTED")])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert client.calls == [("NIFTY", "BUY", 75)]
    assert "No position opened" in output


def test_diagnostics_claim_flat_only_after_two_confirmed_fills(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """The happy path needs a full BUY and full SELL result."""

    client = _DiagnosticClient(
        [
            _diagnostic_result("FILLED", order_id="BUY-1"),
            _diagnostic_result("FILLED", order_id="SELL-1"),
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is True
    assert [call[1] for call in client.calls] == ["BUY", "SELL"]
    assert "-> flat" in output


@pytest.mark.parametrize("status_name", ("PARTIAL", "UNKNOWN"))
def test_diagnostics_do_not_claim_flat_for_ambiguous_exit(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
    status_name: str,
) -> None:
    """A full entry plus ambiguous exit still requires broker verification."""

    client = _DiagnosticClient(
        [_diagnostic_result("FILLED"), _diagnostic_result(status_name, order_id="EXIT-1")]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert len(client.calls) == 2
    assert "indeterminate" in output.lower()
    assert "-> flat" not in output


def test_diagnostics_treat_entry_exception_as_indeterminate(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """An exception after submission started cannot prove a harmless rejection."""

    client = _DiagnosticClient([TimeoutError("response lost")])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert "indeterminate" in output.lower()
    assert "No position opened" not in output
