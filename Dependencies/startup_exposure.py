"""Read-only startup audit for broker orders and index-option positions.

Live workers must not start until both broker books prove there is no exposure
that this process could collide with.  This module deliberately performs only
two reads: it never adopts, cancels, recovers, or flattens broker state.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from Dependencies.broker_contract import (
    BrokerQueryResult,
    ExecutionClient,
    OpenOrder,
    OpenPosition,
)

_RELEVANT_INDEX_SYMBOL = re.compile(
    r"^(?:BANKNIFTY|NIFTY)(?=$|[\s_-]|\d)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StartupExposureAudit:
    """Immutable decision produced before any worker may trade live.

    ``reasons`` are operator-facing explanations.  ``evidence`` intentionally
    contains counts and normalized state only: raw broker errors, order IDs,
    symbols, and credentials can therefore never be copied into an alert.
    """

    safe_to_enable_live: bool
    reasons: tuple[str, ...]
    evidence: tuple[str, ...]


def is_relevant_index_symbol(symbol: object) -> bool:
    """Return whether a broker symbol belongs to NIFTY or BANKNIFTY.

    A boundary after the index name prevents lookalikes such as ``NIFTYBEES``
    and ``FINNIFTY`` from being treated as this runner's option exposure.  A
    digit is a valid boundary because compact broker symbols commonly place the
    expiry immediately after the underlying name.
    """

    if not isinstance(symbol, str):
        return False
    return _RELEVANT_INDEX_SYMBOL.match(symbol.strip()) is not None


def _read_typed_book[BookItem: (OpenOrder, OpenPosition)](
    query: Callable[[], object],
    item_type: type[BookItem],
) -> tuple[BookItem, ...] | None:
    """Return a validated book, or ``None`` for any ambiguous outcome."""

    try:
        result = query()
    except Exception:  # noqa: BLE001 - every broker/SDK failure must fail closed.
        return None

    if not isinstance(result, BrokerQueryResult):
        return None
    if type(result.is_indeterminate) is not bool:  # bool is intentional here.
        return None
    if not isinstance(result.items, tuple):
        return None
    if not isinstance(result.reason, str) or not isinstance(result.broker_state, str):
        return None
    if result.is_indeterminate:
        return None
    if any(not isinstance(item, item_type) for item in result.items):
        return None
    return result.items


def _pluralized_count(count: int, singular: str, plural: str) -> str:
    """Build a short fixed-vocabulary message without broker-provided text."""

    noun = singular if count == 1 else plural
    return f"Broker reported {count} {noun}."


def audit_startup_exposure(execution_client: ExecutionClient) -> StartupExposureAudit:
    """Read both broker books and fail closed unless startup is provably safe.

    Both queries are always attempted independently.  In particular, an order
    timeout must not hide a position snapshot that could provide additional
    exposure evidence for the operator.
    """

    open_orders = _read_typed_book(
        lambda: execution_client.list_open_orders(),
        OpenOrder,
    )
    open_positions = _read_typed_book(
        lambda: execution_client.list_open_positions(),
        OpenPosition,
    )

    reasons: list[str] = []
    evidence: list[str] = []

    if open_orders is None:
        reasons.append("Open-order query was indeterminate.")
        evidence.append("open_orders=indeterminate")
    else:
        order_count = len(open_orders)
        evidence.append(f"open_orders={order_count}")
        if order_count:
            reasons.append(
                _pluralized_count(order_count, "open order", "open orders")
            )

    if open_positions is None:
        reasons.append("Open-position query was indeterminate.")
        evidence.append("relevant_positions=indeterminate")
    else:
        relevant_count = sum(
            position.quantity != 0
            and is_relevant_index_symbol(position.symbol)
            for position in open_positions
        )
        evidence.append(f"relevant_positions={relevant_count}")
        if relevant_count:
            reasons.append(
                _pluralized_count(
                    relevant_count,
                    "non-flat NIFTY/BANKNIFTY position",
                    "non-flat NIFTY/BANKNIFTY positions",
                )
            )

    return StartupExposureAudit(
        safe_to_enable_live=not reasons,
        reasons=tuple(reasons),
        evidence=tuple(evidence),
    )
