"""Split an oversized options order into exchange-legal chunks.

NSE caps how much of a contract may be sent in ONE order -- the "freeze
quantity".  An order at or above it is rejected by the exchange, and in this
runner a rejected live ENTRY with zero fill is deliberately treated as a paper
fallback, so an oversized order would quietly trade on paper while the operator
believed it was live.  This module turns one oversized quantity into a sequence
of quantities the exchange will accept.

Worked example (the real NIFTY numbers from the instrument master):

    lot size        65 units per lot
    freeze quantity 1756  -> the largest LEGAL order is 1755 units
    largest chunk   floor(1755 / 65) = 27 lots = 1755 units

    A 30-lot order is 1950 units, over the limit.  ``split_order_quantity``
    returns ``(1755, 195)`` -- 27 lots then 3 lots, both legal, together
    exactly the 30 lots the strategy asked for.

Two properties the callers rely on:

* every chunk is a whole number of lots (the exchange rejects part-lots), and
* ``sum(chunks) == total_units`` exactly -- splitting never silently drops or
  invents quantity.

BankNIFTY has its own numbers (lot 30, freeze 601 -> 20 lots per chunk), which
is why the freeze quantity is passed in per contract rather than hard-coded:
the SL Hunting mirror splits against BankNIFTY's limit, not NIFTY's.

This module is deliberately pure -- no broker, no I/O, no logging -- so the
arithmetic can be tested exhaustively on its own.
"""

from __future__ import annotations


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive int or raise a named ``ValueError``.

    Booleans are rejected explicitly: ``True`` is an ``int`` in Python and
    would otherwise sail through as the quantity ``1``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not a bool")
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def max_legal_chunk_units(freeze_qty: int, lot_size: int) -> int:
    """Return the largest whole-lot quantity that stays under the freeze limit.

    ``freeze_qty`` is the exchange's THRESHOLD as published in the instrument
    master (NIFTY 1756), so the largest legal order is one unit below it.  That
    is then rounded DOWN to a whole number of lots.

    Raises ``ValueError`` when the limit cannot fit even a single lot, because
    that contract simply cannot be traded by this runner and silently sending
    nothing would be worse than a loud failure.
    """
    freeze = _positive_int(freeze_qty, "freeze_qty")
    lot = _positive_int(lot_size, "lot_size")
    chunk_lots = (freeze - 1) // lot
    if chunk_lots <= 0:
        raise ValueError(
            f"freeze_qty {freeze} cannot fit even one lot of {lot} units"
        )
    return chunk_lots * lot


def split_order_quantity(
    total_units: int,
    freeze_qty: int,
    lot_size: int,
) -> tuple[int, ...]:
    """Split ``total_units`` into chunks the exchange will accept.

    Returns a tuple of quantities in UNITS (not lots), largest-first, that sum
    to exactly ``total_units``.  A quantity already inside the limit comes back
    as a single-element tuple, so callers can always treat the result as "the
    orders to send" without special-casing the common path.

    ``total_units`` is expected to be a whole number of lots (every caller
    computes it as ``lots * lot_size``).  A non-multiple is not rejected --
    it is split with the odd remainder in the final chunk -- because refusing
    to place an order the strategy already sized would be a worse failure than
    forwarding an unusual quantity the broker can still validate.

    Raises ``ValueError`` on genuinely unusable inputs (non-positive or
    non-integer quantities, or a freeze limit too small for one lot).  Callers
    treat that as a refuse-to-trade condition rather than guessing a size.
    """
    total = _positive_int(total_units, "total_units")
    chunk_units = max_legal_chunk_units(freeze_qty, lot_size)

    if total <= chunk_units:
        return (total,)

    chunks: list[int] = []
    remaining = total
    while remaining > chunk_units:
        chunks.append(chunk_units)
        remaining -= chunk_units
    chunks.append(remaining)
    return tuple(chunks)
