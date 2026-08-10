"""MAT-102 tests for the read-only startup exposure gate.

Every fake in this module is in-memory.  The audit may read the broker's two
books, but it must never adopt, cancel, recover, or flatten anything.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from Dependencies.broker_contract import BrokerQueryResult, OpenOrder, OpenPosition
from Dependencies.startup_exposure import (
    StartupExposureAudit,
    audit_startup_exposure,
    is_relevant_index_symbol,
)


def _open_order(*, symbol: str = "NIFTY16JUL2622500CE") -> OpenOrder:
    return OpenOrder(
        order_id="ORDER-1",
        symbol=symbol,
        side="BUY",
        requested_quantity=50,
        filled_quantity=10,
        remaining_quantity=40,
        broker_state="OPEN",
    )


def _position(*, symbol: str, quantity: int = 50) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=quantity,
        product_type="NRML",
    )


class _BookClient:
    """Record the two allowed reads and fail if the audit mutates the account."""

    def __init__(self, *, orders: Any, positions: Any) -> None:
        self.orders = orders
        self.positions = positions
        self.calls: list[str] = []

    def list_open_orders(self) -> Any:
        self.calls.append("list_open_orders")
        if isinstance(self.orders, BaseException):
            raise self.orders
        return self.orders

    def list_open_positions(self) -> Any:
        self.calls.append("list_open_positions")
        if isinstance(self.positions, BaseException):
            raise self.positions
        return self.positions

    def cancel_order(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("startup audit must not cancel broker orders")

    def place_market_order(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("startup audit must not place broker orders")

    def recover(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("startup audit must not recover broker state")


@pytest.mark.parametrize(
    "symbol",
    [
        "NIFTY",
        "NIFTY 22500 CE",
        "NIFTY_22500_CE",
        "NIFTY-22500-CE",
        "NIFTY16JUL2622500CE",
        "BANKNIFTY",
        "BANKNIFTY 51000 PE",
        "BANKNIFTY_51000_PE",
        "BANKNIFTY-51000-PE",
        "BANKNIFTY16JUL2651000PE",
        " nifty16jul2622500ce ",
    ],
)
def test_relevant_index_symbol_accepts_only_supported_prefix_boundaries(
    symbol: str,
) -> None:
    assert is_relevant_index_symbol(symbol) is True


@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "NIFTYBEES",
        "NIFTYNXT50",
        "FINNIFTY16JUL2622500CE",
        "MIDCPNIFTY",
        "BANKNIFTYBEES",
        "XNIFTY16JUL2622500CE",
    ],
)
def test_relevant_index_symbol_rejects_lookalikes(symbol: str) -> None:
    assert is_relevant_index_symbol(symbol) is False


def test_empty_determinate_books_are_safe_and_result_is_immutable() -> None:
    client = _BookClient(
        orders=BrokerQueryResult.success([]),
        positions=BrokerQueryResult.success([]),
    )

    audit = audit_startup_exposure(client)

    assert audit == StartupExposureAudit(
        safe_to_enable_live=True,
        reasons=(),
        evidence=("open_orders=0", "relevant_positions=0"),
    )
    assert client.calls == ["list_open_orders", "list_open_positions"]
    with pytest.raises(FrozenInstanceError):
        audit.safe_to_enable_live = False


def test_any_open_order_blocks_live_regardless_of_symbol() -> None:
    client = _BookClient(
        orders=BrokerQueryResult.success([_open_order(symbol="RELIANCE-EQ")]),
        positions=BrokerQueryResult.success([]),
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is False
    assert audit.reasons == ("Broker reported 1 open order.",)
    assert audit.evidence == ("open_orders=1", "relevant_positions=0")


@pytest.mark.parametrize("quantity", [50, -50])
def test_nonzero_relevant_index_position_blocks_live(quantity: int) -> None:
    client = _BookClient(
        orders=BrokerQueryResult.success([]),
        positions=BrokerQueryResult.success(
            [
                _position(symbol="NIFTYBEES", quantity=100),
                _position(symbol="BANKNIFTY16JUL2651000PE", quantity=quantity),
                _position(symbol="NIFTY16JUL2622500CE", quantity=0),
            ]
        ),
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is False
    assert audit.reasons == (
        "Broker reported 1 non-flat NIFTY/BANKNIFTY position.",
    )
    assert audit.evidence == ("open_orders=0", "relevant_positions=1")


def test_irrelevant_and_zero_positions_do_not_block_live() -> None:
    client = _BookClient(
        orders=BrokerQueryResult.success([]),
        positions=BrokerQueryResult.success(
            [
                _position(symbol="NIFTYBEES", quantity=100),
                _position(symbol="FINNIFTY16JUL2622500CE", quantity=-40),
                _position(symbol="NIFTY16JUL2622500CE", quantity=0),
            ]
        ),
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is True
    assert audit.reasons == ()
    assert audit.evidence == ("open_orders=0", "relevant_positions=0")


def test_both_books_are_queried_when_first_query_raises() -> None:
    client = _BookClient(
        orders=RuntimeError("token=CANARY-SECRET\nforged-log-line"),
        positions=BrokerQueryResult.success(
            [_position(symbol="NIFTY16JUL2622500CE")]
        ),
    )

    audit = audit_startup_exposure(client)

    assert client.calls == ["list_open_orders", "list_open_positions"]
    assert audit.safe_to_enable_live is False
    assert audit.reasons == (
        "Open-order query was indeterminate.",
        "Broker reported 1 non-flat NIFTY/BANKNIFTY position.",
    )
    rendered = " ".join((*audit.reasons, *audit.evidence))
    assert "CANARY-SECRET" not in rendered
    assert "\n" not in rendered


def test_both_books_are_queried_when_first_query_is_indeterminate() -> None:
    client = _BookClient(
        orders=BrokerQueryResult.indeterminate(
            "password=CANARY-PASSWORD\nforged-log-line"
        ),
        positions=BrokerQueryResult.success([]),
    )

    audit = audit_startup_exposure(client)

    assert client.calls == ["list_open_orders", "list_open_positions"]
    assert audit.safe_to_enable_live is False
    assert audit.reasons == ("Open-order query was indeterminate.",)
    assert audit.evidence == (
        "open_orders=indeterminate",
        "relevant_positions=0",
    )
    assert "CANARY-PASSWORD" not in repr(audit)


@pytest.mark.parametrize(
    "bad_orders",
    [
        None,
        [],
        object(),
        BrokerQueryResult(items=(object(),), is_indeterminate=False),
        BrokerQueryResult(items=[_open_order()], is_indeterminate=False),
        BrokerQueryResult(items=(), is_indeterminate="false"),
    ],
)
def test_malformed_open_order_snapshot_is_indeterminate_and_blocks(
    bad_orders: Any,
) -> None:
    client = _BookClient(
        orders=bad_orders,
        positions=BrokerQueryResult.success([]),
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is False
    assert audit.reasons == ("Open-order query was indeterminate.",)


@pytest.mark.parametrize(
    "bad_positions",
    [
        None,
        [],
        object(),
        BrokerQueryResult(items=(object(),), is_indeterminate=False),
        BrokerQueryResult(items=[_position(symbol="NIFTY")], is_indeterminate=False),
        BrokerQueryResult(items=(), is_indeterminate=0),
    ],
)
def test_malformed_position_snapshot_is_indeterminate_and_blocks(
    bad_positions: Any,
) -> None:
    client = _BookClient(
        orders=BrokerQueryResult.success([]),
        positions=bad_positions,
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is False
    assert audit.reasons == ("Open-position query was indeterminate.",)


def test_audit_evidence_is_bounded_and_never_contains_raw_broker_fields() -> None:
    secret = "CANARY-TOKEN-" + "X" * 1_000
    client = _BookClient(
        orders=BrokerQueryResult.success(
            [_open_order(symbol=f"NIFTY {secret}\nforged") for _ in range(500)]
        ),
        positions=BrokerQueryResult.success(
            [_position(symbol=f"BANKNIFTY {secret}\tforged") for _ in range(500)]
        ),
    )

    audit = audit_startup_exposure(client)

    assert audit.safe_to_enable_live is False
    assert len(audit.reasons) <= 2
    assert len(audit.evidence) == 2
    assert all(len(item) <= 64 for item in (*audit.reasons, *audit.evidence))
    assert secret not in repr(audit)
    assert all("\n" not in item and "\t" not in item for item in audit.evidence)
