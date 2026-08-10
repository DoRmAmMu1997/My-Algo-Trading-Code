"""MAT-103 regressions for coordinated flatten-then-stop bookkeeping."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from Dependencies.trading_lifecycle import LifecycleState, TradingLifecycle


class FakeClock:
    """Small controllable monotonic clock used to avoid sleeping in tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _reach_reconciling(
    lifecycle: TradingLifecycle,
    *,
    reason: str = "operator interrupt",
) -> None:
    lifecycle.request_shutdown(reason)
    lifecycle.start_flattening()
    lifecycle.start_reconciling()


def test_shutdown_request_records_first_reason_and_blocks_entries() -> None:
    lifecycle = TradingLifecycle()

    assert lifecycle.snapshot().state is LifecycleState.RUNNING
    assert lifecycle.snapshot().entry_allowed is True

    blocked = lifecycle.request_shutdown("  Ctrl+C  ")
    repeated = lifecycle.request_shutdown("later max-loss request")

    assert blocked.state is LifecycleState.ENTRY_BLOCKED
    assert blocked.entry_allowed is False
    assert blocked.shutdown_reason == "Ctrl+C"
    assert repeated.shutdown_reason == "Ctrl+C"


def test_lifecycle_rejects_out_of_order_or_premature_stop_transitions() -> None:
    lifecycle = TradingLifecycle()

    with pytest.raises(RuntimeError, match="ENTRY_BLOCKED"):
        lifecycle.start_flattening()
    with pytest.raises(RuntimeError, match="FLAT"):
        lifecycle.mark_stopped()

    lifecycle.request_shutdown("15:15 square-off")
    with pytest.raises(RuntimeError, match="FLATTENING"):
        lifecycle.start_reconciling()

    lifecycle.start_flattening()
    with pytest.raises(RuntimeError, match="RECONCILING"):
        lifecycle.record_reconciliation(broker_flat=True)


def test_retry_backoff_is_exactly_one_two_then_capped_at_five_seconds() -> None:
    clock = FakeClock()
    lifecycle = TradingLifecycle(monotonic=clock)
    _reach_reconciling(lifecycle)

    for expected_delay in (1.0, 2.0, 5.0, 5.0, 5.0):
        waiting = lifecycle.record_reconciliation(broker_flat=False)

        assert waiting.state is LifecycleState.RECONCILING
        assert waiting.next_retry_at == pytest.approx(clock.now + expected_delay)
        assert lifecycle.retry_due() is False

        # A repeated negative status check must not postpone the retry.
        unchanged = lifecycle.record_reconciliation(broker_flat=False)
        assert unchanged.next_retry_at == waiting.next_retry_at

        clock.advance(expected_delay)
        assert lifecycle.retry_due() is True
        lifecycle.start_flattening()
        lifecycle.start_reconciling()


def test_transient_flatten_failure_can_recover_to_flat_then_stop() -> None:
    clock = FakeClock()
    lifecycle = TradingLifecycle(monotonic=clock)
    _reach_reconciling(lifecycle, reason="daily max loss")

    lifecycle.record_reconciliation(broker_flat=False)
    with pytest.raises(RuntimeError, match="not due"):
        lifecycle.start_flattening()

    clock.advance(1.0)
    lifecycle.start_flattening()
    lifecycle.start_reconciling()
    flat = lifecycle.record_reconciliation(broker_flat=True)

    assert flat.state is LifecycleState.FLAT
    assert flat.flatten_attempts == 2
    assert flat.next_retry_at is None
    assert lifecycle.mark_stopped().state is LifecycleState.STOPPED


def test_permanent_failure_stays_reconciling_with_unbounded_retries() -> None:
    clock = FakeClock()
    lifecycle = TradingLifecycle(monotonic=clock)
    _reach_reconciling(lifecycle)

    for expected_delay in (1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0):
        lifecycle.record_reconciliation(broker_flat=False)
        clock.advance(expected_delay)
        lifecycle.start_flattening()
        lifecycle.start_reconciling()

    snapshot = lifecycle.snapshot()
    assert snapshot.state is LifecycleState.RECONCILING
    assert snapshot.flatten_attempts == 9
    assert snapshot.entry_allowed is False
    with pytest.raises(RuntimeError, match="FLAT"):
        lifecycle.mark_stopped()


def test_concurrent_shutdown_requests_leave_one_coherent_blocked_snapshot() -> None:
    lifecycle = TradingLifecycle()

    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = tuple(
            executor.map(
                lifecycle.request_shutdown,
                (f"reason-{index}" for index in range(32)),
            )
        )

    final = lifecycle.snapshot()
    assert final.state is LifecycleState.ENTRY_BLOCKED
    assert final.entry_allowed is False
    assert final.shutdown_reason in {f"reason-{index}" for index in range(32)}
    assert {snapshot.shutdown_reason for snapshot in snapshots} == {
        final.shutdown_reason
    }
