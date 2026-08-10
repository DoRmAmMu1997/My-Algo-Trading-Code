"""Exhaustive arithmetic tests for exchange-legal order splitting.

The two invariants every caller depends on -- chunks are whole lots, and they
sum to exactly the requested quantity -- are checked here against the REAL
instrument numbers so a lot-size or freeze-quantity revision shows up as a test
failure rather than a rejected live order.
"""

from __future__ import annotations

import pytest
from order_splitting import max_legal_chunk_units, split_order_quantity

# Straight from `Dependencies/all_instrument <date>.csv` (2026-07-22).
NIFTY_LOT, NIFTY_FREEZE = 65, 1756
BANKNIFTY_LOT, BANKNIFTY_FREEZE = 30, 601


class TestMaxLegalChunk:
    def test_real_instrument_limits(self):
        # 1755 legal units / 65 = 27 lots exactly.
        assert max_legal_chunk_units(NIFTY_FREEZE, NIFTY_LOT) == 27 * NIFTY_LOT == 1755
        # 600 legal units / 30 = 20 lots exactly.
        assert max_legal_chunk_units(BANKNIFTY_FREEZE, BANKNIFTY_LOT) == 20 * BANKNIFTY_LOT == 600

    def test_freeze_is_a_threshold_not_an_allowance(self):
        """The published number is the quantity that TRIPS the freeze, so the
        largest legal order is one unit below it."""
        # 200 units, lot 100 -> 199 legal -> only 1 whole lot fits.
        assert max_legal_chunk_units(200, 100) == 100
        # 201 units, lot 100 -> 200 legal -> 2 lots fit.
        assert max_legal_chunk_units(201, 100) == 200

    def test_limit_too_small_for_one_lot_is_an_error(self):
        with pytest.raises(ValueError, match="cannot fit even one lot"):
            max_legal_chunk_units(65, 65)  # 64 legal units < one 65-unit lot


class TestSplitting:
    def test_quantity_already_legal_returns_one_chunk_unchanged(self):
        """The common path: today's behaviour must be untouched."""
        for lots in (1, 5, 27):
            units = lots * NIFTY_LOT
            assert split_order_quantity(units, NIFTY_FREEZE, NIFTY_LOT) == (units,)

    def test_thirty_lot_nifty_order_splits_27_then_3(self):
        """The multiplier-6 case that motivated this module."""
        chunks = split_order_quantity(30 * NIFTY_LOT, NIFTY_FREEZE, NIFTY_LOT)
        assert chunks == (27 * NIFTY_LOT, 3 * NIFTY_LOT) == (1755, 195)

    def test_banknifty_mirror_splits_against_its_own_limit(self):
        """Multiplier 5 breaches BankNIFTY one step before NIFTY does."""
        chunks = split_order_quantity(25 * BANKNIFTY_LOT, BANKNIFTY_FREEZE, BANKNIFTY_LOT)
        assert chunks == (20 * BANKNIFTY_LOT, 5 * BANKNIFTY_LOT) == (600, 150)

    def test_exact_multiple_of_the_chunk_size_has_no_remainder(self):
        chunks = split_order_quantity(54 * NIFTY_LOT, NIFTY_FREEZE, NIFTY_LOT)
        assert chunks == (1755, 1755)

    def test_large_order_splits_into_many_chunks(self):
        """The 25x ceiling on a 5-lot cap: 125 lots.  No chunk limit applies."""
        chunks = split_order_quantity(125 * NIFTY_LOT, NIFTY_FREEZE, NIFTY_LOT)
        assert len(chunks) == 5
        assert chunks[:4] == (1755,) * 4
        assert chunks[4] == 125 * NIFTY_LOT - 4 * 1755

    @pytest.mark.parametrize("lots", range(1, 130))
    def test_invariants_hold_for_every_lot_count(self, lots):
        """Chunks are whole lots, each is legal, and they sum exactly."""
        units = lots * NIFTY_LOT
        chunks = split_order_quantity(units, NIFTY_FREEZE, NIFTY_LOT)

        assert sum(chunks) == units, "splitting must never drop or invent quantity"
        assert all(chunk % NIFTY_LOT == 0 for chunk in chunks), "part-lots are rejected by NSE"
        assert all(0 < chunk < NIFTY_FREEZE for chunk in chunks), "every chunk must be legal"

    def test_non_multiple_keeps_the_remainder_rather_than_refusing(self):
        """Callers always pass lots x lot_size, but an odd quantity is still
        forwarded intact -- dropping it would be the worse failure."""
        chunks = split_order_quantity(1800, NIFTY_FREEZE, NIFTY_LOT)
        assert sum(chunks) == 1800
        assert chunks == (1755, 45)


class TestInvalidInputs:
    @pytest.mark.parametrize("bad", [0, -1, -65])
    def test_non_positive_quantity_is_rejected(self, bad):
        with pytest.raises(ValueError, match="total_units"):
            split_order_quantity(bad, NIFTY_FREEZE, NIFTY_LOT)

    @pytest.mark.parametrize("bad", [None, "65", 65.0, [65]])
    def test_non_integer_quantity_is_rejected(self, bad):
        with pytest.raises(ValueError, match="total_units"):
            split_order_quantity(bad, NIFTY_FREEZE, NIFTY_LOT)  # type: ignore[arg-type]

    def test_bool_is_not_accepted_as_a_quantity(self):
        """True is an int in Python and would otherwise become quantity 1."""
        with pytest.raises(ValueError, match="not a bool"):
            split_order_quantity(True, NIFTY_FREEZE, NIFTY_LOT)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, None, 65.0, True])
    def test_invalid_freeze_or_lot_size_is_rejected(self, bad):
        with pytest.raises(ValueError):
            split_order_quantity(650, bad, NIFTY_LOT)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            split_order_quantity(650, NIFTY_FREEZE, bad)  # type: ignore[arg-type]
