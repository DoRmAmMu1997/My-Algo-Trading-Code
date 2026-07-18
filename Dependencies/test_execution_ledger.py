"""MAT-102 regressions for quantity-bearing live-exposure bookkeeping."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from Dependencies.broker_contract import OrderResult, OrderStatus
from Dependencies.execution_ledger import (
    ExecutionLedger,
    LegSpec,
    OrderIntent,
    build_order_tag,
)


def _spec(
    *,
    role: str = "N",
    correlation_id: str = "4F8D2Q7J",
    target_quantity: object = 50,
) -> LegSpec:
    return LegSpec(
        strategy="HeikinAshi",
        correlation_id=correlation_id,
        role=role,
        underlying="NIFTY",
        symbol="NIFTY-22500-CE",
        option_type="CE",
        strike=22500.0,
        expiry=date(2026, 7, 16),
        opening_side="BUY",
        target_quantity=target_quantity,  # type: ignore[arg-type]
    )


def _result(
    *,
    order_id: str,
    requested: int,
    filled: int,
    status: OrderStatus,
    broker_state: str,
) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        requested_quantity=requested,
        filled_quantity=filled,
        remaining_quantity=requested - filled,
        status=status,
        broker_state=broker_state,
        reason=f"test {broker_state.lower()}",
    )


def test_leg_is_registered_before_submission_and_cannot_look_flat() -> None:
    ledger = ExecutionLedger()

    leg = ledger.register(_spec())

    assert leg.requested_quantity == 50
    assert leg.filled_quantity == 0
    assert leg.remaining_quantity == 50
    assert leg.confirmed_live_quantity == 0
    assert leg.exposure_indeterminate is True
    assert leg.exposure_possible is True
    assert leg.broker_confirmed_flat is False
    assert leg.risk_quantity == 50


def test_partial_open_applies_only_fill_deltas_then_retries_unfinished_quantity() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    first_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)

    leg = ledger.apply_result(
        first_attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=20,
            status=OrderStatus.PARTIAL,
            broker_state="PARTIAL",
        ),
    )
    assert leg.confirmed_live_quantity == 20
    assert leg.safe_open_retry_quantity == 0
    assert leg.exposure_indeterminate is True

    # Broker status quantities are cumulative for this order. Seeing 30 later
    # adds only ten units, never the full 30 again.
    leg = ledger.apply_result(
        first_attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=30,
            status=OrderStatus.PARTIAL,
            broker_state="PARTIAL",
        ),
    )
    assert leg.confirmed_live_quantity == 30
    assert leg.requested_quantity == 50
    assert leg.filled_quantity == 30
    assert leg.remaining_quantity == 20

    # A terminal cancellation proves the unfinished 20 cannot still fill.
    leg = ledger.apply_result(
        first_attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=30,
            status=OrderStatus.PARTIAL,
            broker_state="CANCELLED",
        ),
    )
    assert leg.exposure_indeterminate is False
    assert leg.safe_open_retry_quantity == 20

    retry_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 20)
    leg = ledger.apply_result(
        retry_attempt,
        _result(
            order_id="OPEN-2",
            requested=20,
            filled=20,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )

    assert leg.confirmed_live_quantity == 50
    assert leg.requested_quantity == 50
    assert leg.filled_quantity == 50
    assert leg.remaining_quantity == 0
    assert leg.entry_complete is True
    assert leg.safe_open_retry_quantity == 0
    assert first_attempt.order_tag != retry_attempt.order_tag
    assert first_attempt.sequence == 1
    assert retry_attempt.sequence == 2


def test_partial_close_never_resubmits_original_quantity() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    open_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    leg = ledger.apply_result(
        open_attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=50,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )

    close_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.CLOSE, 50)
    leg = ledger.apply_result(
        close_attempt,
        _result(
            order_id="CLOSE-1",
            requested=50,
            filled=20,
            status=OrderStatus.PARTIAL,
            broker_state="PARTIAL",
        ),
    )
    assert leg.confirmed_live_quantity == 30
    assert leg.requested_quantity == 50
    assert leg.filled_quantity == 50
    assert leg.remaining_quantity == 0
    assert leg.risk_quantity == 30
    assert leg.safe_close_retry_quantity == 0
    assert leg.broker_confirmed_flat is False

    leg = ledger.apply_result(
        close_attempt,
        _result(
            order_id="CLOSE-1",
            requested=50,
            filled=20,
            status=OrderStatus.PARTIAL,
            broker_state="CANCELLED",
        ),
    )
    assert leg.safe_close_retry_quantity == 30

    retry_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.CLOSE, 30)
    leg = ledger.apply_result(
        retry_attempt,
        _result(
            order_id="CLOSE-2",
            requested=30,
            filled=30,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )
    assert leg.confirmed_live_quantity == 0
    assert leg.broker_confirmed_flat is True
    assert leg.exposure_possible is False


def test_unknown_open_without_order_id_remains_possible_and_not_retryable() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)

    leg = ledger.apply_result(
        attempt,
        _result(
            order_id="",
            requested=50,
            filled=0,
            status=OrderStatus.UNKNOWN,
            broker_state="RESPONSE_LOST",
        ),
    )

    assert leg.confirmed_live_quantity == 0
    assert leg.risk_quantity == 50
    assert leg.exposure_possible is True
    assert leg.safe_open_retry_quantity == 0
    assert leg.broker_confirmed_flat is False


def test_ledger_reuses_unfinished_leg_but_not_a_completed_flat_trade() -> None:
    ledger = ExecutionLedger()
    first = ledger.register(_spec())

    assert ledger.find_unfinished(_spec()) == first
    assert ledger.register(_spec()) == first

    attempt = ledger.start_attempt(first.exposure_id, OrderIntent.OPEN, 50)
    first = ledger.apply_result(
        attempt,
        _result(
            order_id="REJECT-1",
            requested=50,
            filled=0,
            status=OrderStatus.REJECTED,
            broker_state="REJECTED",
        ),
    )
    assert first.broker_confirmed_flat is True
    assert ledger.find_unfinished(_spec()) is None

    second = ledger.register(_spec(correlation_id="8A7B6C5D"))
    assert second.exposure_id != first.exposure_id


def test_order_tags_are_bounded_ascii_and_preserve_trade_correlation() -> None:
    entry_1 = build_order_tag("HeikinAshi", "4F8D2Q7J", "N", "E", 1)
    entry_2 = build_order_tag("HeikinAshi", "4F8D2Q7J", "N", "E", 2)
    mirror = build_order_tag("HeikinAshi", "4F8D2Q7J", "B", "E", 1)

    assert re.fullmatch(r"[A-Z0-9-]{1,20}", entry_1)
    assert len(entry_1) == 20
    assert entry_1[:-1] == entry_2[:-1]
    assert entry_1 != entry_2
    assert entry_1 != mirror


@pytest.mark.parametrize("broker_state", ["COMPLETE", "CANCELLED"])
def test_unknown_status_never_uses_raw_terminal_label_to_hide_exposure(
    broker_state: str,
) -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)

    leg = ledger.apply_result(
        attempt,
        _result(
            order_id="",
            requested=50,
            filled=0,
            status=OrderStatus.UNKNOWN,
            broker_state=broker_state,
        ),
    )

    assert leg.exposure_indeterminate is True
    assert leg.exposure_possible is True
    assert leg.broker_confirmed_flat is False
    assert leg.safe_open_retry_quantity == 0
    assert leg.risk_quantity == 50


def test_stale_attempt_result_is_rejected_before_it_can_mutate_exposure() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    first_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    leg = ledger.apply_result(
        first_attempt,
        _result(
            order_id="",
            requested=50,
            filled=0,
            status=OrderStatus.REJECTED,
            broker_state="REJECTED",
        ),
    )
    second_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    before_late_result = ledger.get(leg.exposure_id)

    with pytest.raises(RuntimeError, match="stale order-attempt handle"):
        ledger.apply_result(
            first_attempt,
            _result(
                order_id="LATE-OLD-ORDER",
                requested=50,
                filled=50,
                status=OrderStatus.FILLED,
                broker_state="COMPLETE",
            ),
        )

    assert ledger.get(leg.exposure_id) == before_late_result
    assert second_attempt.sequence == 2


def test_terminal_attempt_cannot_regress_on_equal_fill_nonterminal_snapshot() -> None:
    """A late OPEN snapshot must not undo stronger terminal evidence."""

    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    terminal = ledger.apply_result(
        attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=50,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )

    late = ledger.apply_result(
        attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=50,
            status=OrderStatus.UNKNOWN,
            broker_state="OPEN",
        ),
    )

    assert late == terminal
    assert late.entry_complete is True
    assert late.exposure_indeterminate is False
    assert late.latest_attempt is not None
    assert late.latest_attempt.terminal is True


def test_state_snapshot_exposes_the_exact_latest_attempt_handle() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)

    state = ledger.get(leg.exposure_id)

    assert state.latest_attempt_handle == attempt


def test_entry_cannot_reopen_a_leg_after_closing_has_started() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    open_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    leg = ledger.apply_result(
        open_attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=50,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )
    close_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.CLOSE, 50)
    leg = ledger.apply_result(
        close_attempt,
        _result(
            order_id="CLOSE-1",
            requested=50,
            filled=20,
            status=OrderStatus.PARTIAL,
            broker_state="CANCELLED",
        ),
    )

    with pytest.raises(RuntimeError, match="closing has started"):
        ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 20)

    retry_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.CLOSE, 30)
    leg = ledger.apply_result(
        retry_attempt,
        _result(
            order_id="CLOSE-2",
            requested=30,
            filled=30,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )
    assert leg.broker_confirmed_flat is True

    with pytest.raises(RuntimeError, match="closing has started"):
        ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)

    replacement = ledger.register(_spec(correlation_id="8A7B6C5D"))
    assert replacement.exposure_id != leg.exposure_id


def test_order_tags_remain_broker_safe_for_unbounded_attempt_sequences() -> None:
    attempt_36 = build_order_tag("HeikinAshi", "4F8D2Q7J", "N", "X", 36)
    attempt_10_000 = build_order_tag("HeikinAshi", "4F8D2Q7J", "N", "X", 10_000)

    for tag in (attempt_36, attempt_10_000):
        assert re.fullmatch(r"[A-Z0-9-]{1,20}", tag)
        assert len(tag) <= 20
    assert attempt_36 != attempt_10_000


def test_reducing_retries_continue_after_attempt_35() -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())
    open_attempt = ledger.start_attempt(leg.exposure_id, OrderIntent.OPEN, 50)
    leg = ledger.apply_result(
        open_attempt,
        _result(
            order_id="OPEN",
            requested=50,
            filled=50,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )

    latest_attempt = open_attempt
    for sequence in range(2, 42):
        latest_attempt = ledger.start_attempt(
            leg.exposure_id,
            OrderIntent.CLOSE,
            50,
        )
        leg = ledger.apply_result(
            latest_attempt,
            _result(
                order_id=f"REJECT-{sequence}",
                requested=50,
                filled=0,
                status=OrderStatus.REJECTED,
                broker_state="REJECTED",
            ),
        )

    assert latest_attempt.sequence == 41
    assert len(latest_attempt.order_tag) <= 20
    assert leg.confirmed_live_quantity == 50
    assert leg.safe_close_retry_quantity == 50


@pytest.mark.parametrize("invalid_quantity", [True, 50.0, 1.5, "50"])
def test_leg_target_quantity_requires_an_exact_integer(invalid_quantity: object) -> None:
    with pytest.raises(ValueError, match="target_quantity must be a positive integer"):
        _spec(target_quantity=invalid_quantity)


@pytest.mark.parametrize("invalid_quantity", [True, 50.0, 1.5, "50"])
def test_attempt_requested_quantity_requires_an_exact_integer(
    invalid_quantity: object,
) -> None:
    ledger = ExecutionLedger()
    leg = ledger.register(_spec())

    with pytest.raises(ValueError, match="requested_quantity must be a positive integer"):
        ledger.start_attempt(
            leg.exposure_id,
            OrderIntent.OPEN,
            invalid_quantity,  # type: ignore[arg-type]
        )


def test_public_states_are_immutable_coherent_snapshots() -> None:
    ledger = ExecutionLedger()
    registered = ledger.register(_spec())
    attempt = ledger.start_attempt(registered.exposure_id, OrderIntent.OPEN, 50)
    filled = ledger.apply_result(
        attempt,
        _result(
            order_id="OPEN-1",
            requested=50,
            filled=50,
            status=OrderStatus.FILLED,
            broker_state="COMPLETE",
        ),
    )

    with pytest.raises(FrozenInstanceError):
        registered.confirmed_live_quantity = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        filled.latest_attempt.terminal = False  # type: ignore[misc,union-attr]

    assert registered.confirmed_live_quantity == 0
    assert filled.confirmed_live_quantity == 50
    assert ledger.get(filled.exposure_id) == filled
    assert ledger.active_states() == (filled,)
