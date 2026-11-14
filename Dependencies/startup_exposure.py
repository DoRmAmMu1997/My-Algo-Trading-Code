"""Read-only startup audit for broker orders and index-option positions.

Live workers must not start until both broker books prove there is no exposure
that this process could collide with.  This module deliberately performs only
two reads: it never adopts, cancels, recovers, or flattens broker state.

One opt-in exception (ADR-0023): the caller may pass the carried legs the
operator has DECLARED, and a position that matches one of them exactly no
longer counts as exposure.  This module does not know where that list comes
from and does not read any file; with no list it behaves exactly as before.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
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
    """Return a validated book, or ``None`` for any ambiguous outcome.

    ``None`` never means "empty" here -- an empty-but-successful book is the
    empty tuple.  ``None`` means "could not be proven", which the caller
    treats exactly like exposure.  The type checks below look paranoid on
    purpose: this function is the last line between an adapter bug (or a
    monkeypatched test double drifting from the contract) and the decision to
    let real orders start, so any shape that is not literally the contract's
    is refused rather than coerced.
    """

    try:
        result = query()
    except Exception:  # noqa: BLE001 - every broker/SDK failure must fail closed.
        return None

    if not isinstance(result, BrokerQueryResult):
        return None
    # ``type(...) is not bool`` rather than truthiness: a mock returning a
    # truthy non-bool (e.g. a MagicMock) must not read as "determinate".
    if type(result.is_indeterminate) is not bool:  # bool is intentional here.
        return None
    if not isinstance(result.items, tuple):
        return None
    if not isinstance(result.reason, str) or not isinstance(result.broker_state, str):
        return None
    if result.is_indeterminate:
        return None
    # Every element must be the exact typed item -- one foreign object means
    # the whole snapshot cannot be trusted.
    if any(not isinstance(item, item_type) for item in result.items):
        return None
    return result.items


def _pluralized_count(count: int, singular: str, plural: str) -> str:
    """Build a short fixed-vocabulary message without broker-provided text."""

    noun = singular if count == 1 else plural
    return f"Broker reported {count} {noun}."


def audit_startup_exposure(
    execution_client: ExecutionClient,
    *,
    tolerated_positions: Mapping[tuple[str, str], int] | None = None,
) -> StartupExposureAudit:
    """Read both broker books and fail closed unless startup is provably safe.

    Both queries are always attempted independently.  In particular, an order
    timeout must not hide a position snapshot that could provide additional
    exposure evidence for the operator.

    ``tolerated_positions`` maps ``(symbol, PRODUCT)`` to the signed quantity
    the operator has declared for a carried leg.  A broker position is excused
    only when its symbol, its product label and its exact signed quantity all
    match one entry, and each entry excuses ONE broker row.  A different
    quantity, the other sign, another product, a missing product label or a
    second identical row is ordinary exposure and blocks as before.  It never
    excuses an open order or an unreadable book.

    ``None`` (the default) keeps the historical behaviour and the historical
    evidence, byte for byte.  Passing a mapping -- even an empty one -- adds a
    ``declared_carried_positions=<count>`` evidence field, so a reader can tell
    that declared legs were consulted and how many were excused.
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
        # Each declared leg may excuse one row, so the mapping is consumed as
        # rows match: a second identical row means the account holds more
        # than the operator declared.
        unmatched_declared = dict(tolerated_positions or {})
        relevant_count = 0
        declared_count = 0
        for position in open_positions:
            if position.quantity == 0 or not is_relevant_index_symbol(position.symbol):
                continue
            key = (position.symbol.strip(), str(position.product_type).strip().upper())
            if unmatched_declared.get(key) == position.quantity:
                del unmatched_declared[key]
                declared_count += 1
            else:
                relevant_count += 1
        evidence.append(f"relevant_positions={relevant_count}")
        if tolerated_positions is not None:
            evidence.append(f"declared_carried_positions={declared_count}")
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
