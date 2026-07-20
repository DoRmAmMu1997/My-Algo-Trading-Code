"""Local pre-flight checks shared by every broker diagnostic script.

These run BEFORE a real order is submitted and BEFORE the typed-YES prompt, so
an order that is certain to be refused never reaches the broker and never
wastes the operator's confirmation.

Why this lives in one module rather than in each diagnostic: the same rule
applies to Kotak, Shoonya, Flattrade and Dhan, and four copies would drift.
The checks here are deliberately broker-neutral -- they take plain numbers, not
a client -- so nothing in this file can touch a live session.
"""

from __future__ import annotations


def validate_quantity_for_lot(quantity: int, lot_size: int) -> str:
    """Return an error message when ``quantity`` is not a whole-lot multiple.

    Index options trade only in exchange-defined lots, so a quantity such as 1
    against a lot of 65 can never be accepted.  Catching that locally matters
    because brokers report it obscurely: Dhan returns a generic
    ``DH-905 Input_Exception`` whose ``error_message`` has been observed to read
    "Invalid IP", which sends operators hunting a network fault that does not
    exist.

    This validates the operator's OWN number; it deliberately does not derive
    one.  ``--place-order`` still requires an explicit ``--qty`` on every
    diagnostic, because lot sizes change and guessing one would risk a
    wrong-sized live order.

    Args:
        quantity: The explicit unit quantity supplied on the command line.
        lot_size: Official lot size for the contract, or 0 when the broker's
            contract master does not expose one.

    Returns:
        An empty string when the quantity is placeable (or cannot be checked),
        else a human-readable reason the caller should print before refusing.
    """
    try:
        quantity_i = int(quantity)
        lot_size_i = int(lot_size)
    except (TypeError, ValueError):
        # Nothing trustworthy to compare; the broker remains the authority.
        return ""
    if lot_size_i <= 0:
        # The contract master carried no usable lot size, so there is nothing to
        # check against. Staying silent is right: refusing here would block a
        # perfectly good order over missing catalogue metadata.
        return ""
    if quantity_i % lot_size_i:
        return (
            f"--qty {quantity_i} is not a multiple of the official lot size "
            f"{lot_size_i}. Index options trade in whole lots only, so the "
            f"broker would reject this order. Use {lot_size_i} for one lot "
            f"(or {lot_size_i * 2} for two)."
        )
    return ""
