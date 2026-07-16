"""Broker-neutral execution results shared by every live-order adapter.

An acknowledgement is not a fill.  The small types in this module make that
distinction explicit so callers cannot accidentally treat a truthy broker reply
as proof that the requested quantity traded.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# Broker helpers can run both as repo modules (``Dependencies.broker_contract``)
# and as standalone scripts that import ``broker_contract`` after adding the
# Dependencies directory to ``sys.path``.  Point both names at this one module so
# ``isinstance(OrderResult)`` remains reliable at the live-order boundary.
sys.modules.setdefault("broker_contract", sys.modules[__name__])
sys.modules.setdefault("Dependencies.broker_contract", sys.modules[__name__])


class OrderStatus(Enum):
    """The only four normalized outcomes exposed to trading code."""

    REJECTED = "REJECTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class OrderResult:
    """Immutable normalized result for one broker order.

    Quantity fields are deliberately redundant.  Verifying the relationship at
    construction time catches malformed broker payloads before they can reach a
    live-money decision boundary.
    """

    order_id: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    status: OrderStatus
    broker_state: str
    reason: str

    def __post_init__(self) -> None:
        """Reject impossible or internally inconsistent quantity snapshots."""

        quantities = {
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
        }
        for name, value in quantities.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.filled_quantity > self.requested_quantity:
            raise ValueError("filled_quantity cannot exceed requested_quantity")
        expected_remaining = self.requested_quantity - self.filled_quantity
        if self.remaining_quantity != expected_remaining:
            raise ValueError(
                "remaining_quantity must equal requested_quantity - filled_quantity"
            )
        if not isinstance(self.status, OrderStatus):
            raise ValueError("status must be an OrderStatus value")
        if self.status is OrderStatus.FILLED and (
            self.requested_quantity <= 0
            or self.filled_quantity != self.requested_quantity
        ):
            raise ValueError("FILLED status requires the complete requested quantity")
        if self.status is OrderStatus.REJECTED and self.filled_quantity != 0:
            raise ValueError("REJECTED status requires a zero filled quantity")
        if self.status is OrderStatus.PARTIAL and not (
            0 < self.filled_quantity < self.requested_quantity
        ):
            raise ValueError("PARTIAL status requires a non-zero incomplete fill")


_FILLED_STATES = frozenset({"COMPLETE", "COMPLETED", "FILLED", "TRADED", "EXECUTED"})
_REJECTED_STATES = frozenset(
    {"REJECTED", "CANCELLED", "CANCELED", "CANCEL", "LAPSED"}
)
_PARTIAL_STATES = frozenset(
    {"PARTIAL", "PARTIALLY FILLED", "PARTIALLY_FILLED", "PARTIALLY-FILLED"}
)

# Every broker label that means "this order can never fill any further".  The
# adapters' fill-confirmation loops poll until they see one of these (or their
# timeout expires): anything else -- "OPEN", "PENDING", Kotak's "VALIDATION
# PENDING" / "PUT ORDER REQ RECEIVED", or an unrecognized label -- is a
# TRANSIENT state that a healthy order passes through on its way to COMPLETE,
# so returning early on it would report a half-finished snapshot as the final
# outcome.  This is deliberately the same terminality rule the execution
# ledger applies when it decides whether a PARTIAL fill is final.
TERMINAL_BROKER_STATES = _FILLED_STATES | _REJECTED_STATES


def _non_negative_int(value: Any) -> int | None:
    """Return an exact non-negative integer, never a rounded quantity."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def normalize_order_result(
    *,
    order_id: object,
    requested_quantity: object,
    filled_quantity: object,
    broker_state: object,
    reason: object = "",
) -> OrderResult:
    """Normalize broker evidence without guessing that ambiguity is rejection.

    A positive filled quantity always wins over a broker's rejection label because
    it proves exposure exists.  Conversely, a full requested quantity is called
    ``FILLED`` only when the broker also reports a recognized terminal fill state.
    """

    requested = _non_negative_int(requested_quantity)
    filled = _non_negative_int(filled_quantity)
    malformed_parts: list[str] = []
    if requested is None:
        malformed_parts.append("requested quantity is malformed")
        requested = 0
    if filled is None or filled > requested:
        malformed_parts.append("filled quantity is malformed or exceeds requested quantity")
        filled = 0

    state = str(broker_state or "").strip().upper()
    reason_text = str(reason or "").strip()
    if malformed_parts:
        status = OrderStatus.UNKNOWN
        detail = "; ".join(malformed_parts)
        reason_text = f"Malformed broker quantity: {detail}. {reason_text}".strip()
    elif 0 < filled < requested:
        status = OrderStatus.PARTIAL
    elif state in _PARTIAL_STATES:
        # A partial label with no usable fill count still proves the state is
        # ambiguous, not that zero units traded.
        status = OrderStatus.PARTIAL if filled > 0 else OrderStatus.UNKNOWN
    elif requested > 0 and filled == requested and state in _FILLED_STATES:
        status = OrderStatus.FILLED
    elif filled == 0 and state in _REJECTED_STATES:
        status = OrderStatus.REJECTED
    else:
        status = OrderStatus.UNKNOWN

    if not reason_text:
        reason_text = {
            OrderStatus.FILLED: "Broker confirmed the requested quantity filled.",
            OrderStatus.PARTIAL: "Broker reported a non-zero partial fill.",
            OrderStatus.REJECTED: "Broker confirmed zero fill and a terminal rejection.",
            OrderStatus.UNKNOWN: "Broker response did not prove a terminal order outcome.",
        }[status]

    return OrderResult(
        order_id=str(order_id or "").strip(),
        requested_quantity=requested,
        filled_quantity=filled,
        remaining_quantity=requested - filled,
        status=status,
        broker_state=state,
        reason=reason_text,
    )


@dataclass(frozen=True, slots=True)
class OpenOrder:
    """One broker order that may still change the account's exposure."""

    order_id: str
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    broker_state: str

    def __post_init__(self) -> None:
        """Apply the same quantity invariants used by :class:`OrderResult`."""

        OrderResult(
            order_id=self.order_id,
            requested_quantity=self.requested_quantity,
            filled_quantity=self.filled_quantity,
            remaining_quantity=self.remaining_quantity,
            status=OrderStatus.UNKNOWN,
            broker_state=self.broker_state,
            reason="Open-order snapshot",
        )


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """One non-flat broker position returned by a reconciliation query."""

    symbol: str
    quantity: int
    product_type: str
    broker_state: str = "OPEN"

    def __post_init__(self) -> None:
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("quantity must be an integer")


@dataclass(frozen=True, slots=True)
class BrokerQueryResult[QueryItem]:
    """Typed list-query result that distinguishes empty from indeterminate."""

    items: tuple[QueryItem, ...]
    is_indeterminate: bool
    reason: str = ""
    broker_state: str = ""

    @classmethod
    def success(cls, items: Iterable[QueryItem]) -> BrokerQueryResult[QueryItem]:
        """Build a successful result; an empty tuple now genuinely means flat."""

        return cls(items=tuple(items), is_indeterminate=False)

    @classmethod
    def indeterminate(
        cls,
        reason: str,
        *,
        broker_state: str = "UNKNOWN",
    ) -> BrokerQueryResult[QueryItem]:
        """Build a failed/ambiguous result that cannot look like an empty book."""

        return cls(
            items=(),
            is_indeterminate=True,
            reason=str(reason).strip() or "Broker query outcome is indeterminate.",
            broker_state=str(broker_state).strip(),
        )


@runtime_checkable
class ExecutionClient(Protocol):
    """Common broker surface used by the runner and future reconciliation."""

    is_logged_in: bool

    def ensure_logged_in(self) -> bool: ...

    def preload_scrip_master(self) -> bool: ...

    def resolve_option_symbol(
        self,
        underlying: str,
        expiry: Any,
        option_type: str,
        strike: float,
    ) -> str: ...

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchange_segment: str,
        product_type: str,
        *,
        order_tag: str = "",
    ) -> OrderResult: ...

    def get_order_status(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult: ...

    def cancel_order(
        self,
        order_id: str,
        requested_quantity: int = 0,
    ) -> OrderResult: ...

    def list_open_orders(self) -> BrokerQueryResult[OpenOrder]: ...

    def list_open_positions(self) -> BrokerQueryResult[OpenPosition]: ...

    def recover_after_reconciliation(self) -> bool: ...

    def extract_order_id(self, order_response: Any) -> str: ...

    def logout(self) -> dict[str, Any]: ...
