"""Thread-safe, quantity-bearing state for every live broker leg.

The strategy runner must never infer exposure from a boolean.  A broker can
acknowledge an order, fill only part of it, lose the response, or fill more while
the process polls status.  This module records each leg *before* submission and
applies cumulative broker fill reports as deltas, so known quantity can never
silently disappear or be counted twice.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from datetime import date
from enum import Enum

from Dependencies.broker_contract import OrderResult, OrderStatus

_CORRELATION_RE = re.compile(r"^[A-Z0-9]{8}$")
_TAG_RE = re.compile(r"^[A-Z0-9-]{1,20}$")
_TERMINAL_BROKER_STATES = frozenset(
    {
        "CANCEL",
        "CANCELED",
        "CANCELLED",
        "COMPLETE",
        "COMPLETED",
        "EXECUTED",
        "FILLED",
        "LAPSED",
        "REJECTED",
        "TRADED",
    }
)
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TAG_ATTEMPT_MODULUS = len(_BASE36) ** 2


class OrderIntent(Enum):
    """Whether an order adds to or reduces one tracked leg."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"


@dataclass(frozen=True, slots=True)
class LegSpec:
    """Immutable identity and target quantity for one strategy basket leg."""

    strategy: str
    correlation_id: str
    role: str
    underlying: str
    symbol: str
    option_type: str
    strike: float
    expiry: date | None
    opening_side: str
    target_quantity: int
    owner_id: str = ""

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip()
        correlation = str(self.correlation_id).upper().strip()
        role = str(self.role).upper().strip()
        underlying = str(self.underlying).upper().strip()
        symbol = str(self.symbol).strip()
        option_type = str(self.option_type).upper().strip()
        opening_side = str(self.opening_side).upper().strip()
        owner_id = str(self.owner_id).upper().strip()
        if not strategy:
            raise ValueError("strategy is required")
        if not _CORRELATION_RE.fullmatch(correlation):
            raise ValueError("correlation_id must contain exactly 8 ASCII letters/digits")
        if len(role) != 1 or not role.isascii() or not role.isalnum():
            raise ValueError("role must be one ASCII letter/digit")
        if not underlying or not symbol:
            raise ValueError("underlying and symbol are required")
        if option_type not in {"CE", "PE"}:
            raise ValueError("option_type must be CE or PE")
        if opening_side not in {"BUY", "SELL"}:
            raise ValueError("opening_side must be BUY or SELL")
        if type(self.target_quantity) is not int or self.target_quantity <= 0:
            raise ValueError("target_quantity must be a positive integer")
        if owner_id and not _CORRELATION_RE.fullmatch(owner_id):
            raise ValueError("owner_id must be blank or exactly 8 ASCII letters/digits")
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "correlation_id", correlation)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "underlying", underlying)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "option_type", option_type)
        object.__setattr__(self, "opening_side", opening_side)
        object.__setattr__(self, "owner_id", owner_id)

    @property
    def active_signature(self) -> tuple[str, str, str, str, str, str]:
        """Fields that identify an unfinished leg across a signal retry."""

        return (
            self.strategy,
            self.role,
            self.underlying,
            self.symbol,
            self.opening_side,
            self.owner_id,
        )


@dataclass(frozen=True, slots=True)
class OrderAttempt:
    """Immutable snapshot of cumulative evidence for one broker submission."""

    intent: OrderIntent
    sequence: int
    order_tag: str
    requested_quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    order_id: str = ""
    status: OrderStatus = OrderStatus.UNKNOWN
    broker_state: str = "SUBMITTING"
    reason: str = "Order submission has started; final outcome is unknown."
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class OrderAttemptHandle:
    """Identity required when applying evidence to a specific submission.

    Broker timeouts and status probes can finish after a retry has already
    started.  Binding each result to this sequence prevents a late response
    from being mistaken for evidence about the newer order.
    """

    exposure_id: str
    sequence: int
    order_tag: str


@dataclass(frozen=True, slots=True)
class LiveLegState:
    """Coherent immutable snapshot of one live strategy leg.

    Beginner's map of the quantity fields (all in units, not lots):

    - ``spec.target_quantity``      : what the strategy WANTS open in total.
    - ``requested_quantity``        : entry quantity submitted so far.
    - ``filled_quantity``           : entry quantity the broker confirmed
                                      filled so far (cumulative).
    - ``remaining_quantity``        : entry quantity still to be filled
                                      (= target - filled).
    - ``confirmed_live_quantity``   : what is open at the broker RIGHT NOW --
                                      entry fills minus confirmed close fills.
    - ``exposure_indeterminate``    : True while the newest order attempt has
                                      not reached a terminal broker state, so
                                      MORE quantity than confirmed may exist.

    Brokers report fills CUMULATIVELY per order ("35 of 75 filled", later
    "75 of 75"), so the ledger applies each report as a DELTA against the
    previous snapshot of the same attempt.  Applying deltas -- instead of
    overwriting totals -- is what makes it impossible for a repeated or
    re-ordered status report to double-count a fill or silently erase one.
    """

    exposure_id: str
    spec: LegSpec
    requested_quantity: int = 0
    filled_quantity: int = 0
    remaining_quantity: int = 0
    confirmed_live_quantity: int = 0
    exposure_indeterminate: bool = True
    latest_attempt: OrderAttempt | None = None
    attempt_count: int = 0
    closing_started: bool = False

    @property
    def latest_attempt_handle(self) -> OrderAttemptHandle | None:
        """Return the exact identity required for a later status update."""

        attempt = self.latest_attempt
        if attempt is None:
            return None
        return OrderAttemptHandle(
            exposure_id=self.exposure_id,
            sequence=attempt.sequence,
            order_tag=attempt.order_tag,
        )

    @property
    def entry_complete(self) -> bool:
        """Whether the full target is confirmed open with no pending ambiguity."""

        return (
            not self.closing_started
            and self.filled_quantity == self.requested_quantity
            and not self.exposure_indeterminate
            and self.confirmed_live_quantity == self.spec.target_quantity
        )

    @property
    def broker_confirmed_flat(self) -> bool:
        """Whether broker evidence proves that this leg has zero exposure."""

        return (
            self.latest_attempt is not None
            and self.latest_attempt.terminal
            and not self.exposure_indeterminate
            and self.confirmed_live_quantity == 0
        )

    @property
    def exposure_possible(self) -> bool:
        """Whether known or unresolved broker quantity may still exist."""

        return not self.broker_confirmed_flat and (self.confirmed_live_quantity > 0 or self.exposure_indeterminate)

    @property
    def safe_open_retry_quantity(self) -> int:
        """Known unfinished entry quantity, or zero while the last order may fill.

        Zero means "do not submit anything yet": either the previous order is
        still non-terminal (it may fill more on its own -- adding quantity now
        could overshoot the target) or closing has already started.  A positive
        number is the exact remainder a retry may safely request.
        """

        if self.closing_started:
            return 0
        if self.latest_attempt is None:
            return self.remaining_quantity
        if self.exposure_indeterminate or not self.latest_attempt.terminal:
            return 0
        return max(0, self.remaining_quantity)

    @property
    def safe_close_retry_quantity(self) -> int:
        """Known remaining exposure safe to submit as a reducing order.

        Only broker-CONFIRMED open quantity may be closed; while the state is
        indeterminate this is zero, because selling units that may never have
        been bought would open a naked short instead of reducing risk.
        """

        if self.exposure_indeterminate:
            return 0
        return max(0, self.confirmed_live_quantity)

    @property
    def risk_quantity(self) -> int:
        """Conservative quantity to use for risk and shutdown decisions.

        The mirror-image of the two "safe retry" properties: where those round
        AMBIGUITY DOWN (never submit quantity that might not be needed), risk
        maths must round ambiguity UP -- an indeterminate open attempt is
        counted as if its whole remainder filled, so mark-to-market and
        max-loss checks can never treat possible exposure as flat.
        """

        attempt = self.latest_attempt
        if self.exposure_indeterminate and attempt is not None and attempt.intent is OrderIntent.OPEN:
            return min(
                self.spec.target_quantity,
                self.confirmed_live_quantity + attempt.remaining_quantity,
            )
        if self.exposure_indeterminate and attempt is None:
            return self.spec.target_quantity
        return self.confirmed_live_quantity


@dataclass(slots=True)
class _MutableOrderAttempt:
    """Lock-owned attempt record; never returned to callers."""

    intent: OrderIntent
    sequence: int
    order_tag: str
    requested_quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    order_id: str = ""
    status: OrderStatus = OrderStatus.UNKNOWN
    broker_state: str = "SUBMITTING"
    reason: str = "Order submission has started; final outcome is unknown."
    terminal: bool = False


@dataclass(slots=True)
class _MutableLiveLeg:
    """Lock-owned leg record used to create public immutable snapshots."""

    exposure_id: str
    spec: LegSpec
    requested_quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    confirmed_live_quantity: int = 0
    exposure_indeterminate: bool = True
    latest_attempt: _MutableOrderAttempt | None = None
    attempt_count: int = 0
    closing_started: bool = False


def build_order_tag(
    strategy: str,
    correlation_id: str,
    role: str,
    phase: str,
    attempt: int,
) -> str:
    """Build a bounded broker tag without limiting safety retries.

    The full sequence lives in :class:`OrderAttemptHandle`.  The final two tag
    characters are only supporting broker metadata and wrap every 1,296
    attempts, so a broker field limit can never stop a liquidation retry.
    """

    strategy_text = str(strategy).strip()
    correlation = str(correlation_id).upper().strip()
    role_text = str(role).upper().strip()
    phase_text = str(phase).upper().strip()
    if not strategy_text:
        raise ValueError("strategy is required for an order tag")
    if not _CORRELATION_RE.fullmatch(correlation):
        raise ValueError("correlation_id must contain exactly 8 ASCII letters/digits")
    for name, value in (("role", role_text), ("phase", phase_text)):
        if len(value) != 1 or not value.isascii() or not value.isalnum():
            raise ValueError(f"{name} must be one ASCII letter/digit")
    if type(attempt) is not int or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    strategy_code = (
        hashlib.blake2s(
            strategy_text.encode("utf-8"),
            digest_size=2,
        )
        .hexdigest()
        .upper()[:3]
    )
    bounded_attempt = attempt % _TAG_ATTEMPT_MODULUS
    attempt_code = _BASE36[bounded_attempt // len(_BASE36)] + _BASE36[bounded_attempt % len(_BASE36)]
    tag = f"M2-{strategy_code}-{correlation}-{role_text}{phase_text}{attempt_code}"
    if not _TAG_RE.fullmatch(tag):
        raise ValueError("generated order tag is not broker-safe")
    return tag


class ExecutionLedger:
    """Lock-guarded registry and cumulative fill applier for live legs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._legs: dict[str, _MutableLiveLeg] = {}

    @staticmethod
    def _exposure_id(spec: LegSpec) -> str:
        strategy_code = (
            hashlib.blake2s(
                spec.strategy.encode("utf-8"),
                digest_size=2,
            )
            .hexdigest()
            .upper()
        )
        return f"{strategy_code}-{spec.correlation_id}-{spec.role}"

    @staticmethod
    def _snapshot_attempt(attempt: _MutableOrderAttempt) -> OrderAttempt:
        """Copy one lock-owned attempt into an immutable public value."""

        return OrderAttempt(
            intent=attempt.intent,
            sequence=attempt.sequence,
            order_tag=attempt.order_tag,
            requested_quantity=attempt.requested_quantity,
            filled_quantity=attempt.filled_quantity,
            remaining_quantity=attempt.remaining_quantity,
            order_id=attempt.order_id,
            status=attempt.status,
            broker_state=attempt.broker_state,
            reason=attempt.reason,
            terminal=attempt.terminal,
        )

    @classmethod
    def _snapshot_leg(cls, leg: _MutableLiveLeg) -> LiveLegState:
        """Copy one lock-owned leg while its fields form a coherent snapshot."""

        attempt = leg.latest_attempt
        return LiveLegState(
            exposure_id=leg.exposure_id,
            spec=leg.spec,
            requested_quantity=leg.requested_quantity,
            filled_quantity=leg.filled_quantity,
            remaining_quantity=leg.remaining_quantity,
            confirmed_live_quantity=leg.confirmed_live_quantity,
            exposure_indeterminate=leg.exposure_indeterminate,
            latest_attempt=(cls._snapshot_attempt(attempt) if attempt is not None else None),
            attempt_count=leg.attempt_count,
            closing_started=leg.closing_started,
        )

    @classmethod
    def _exposure_possible(cls, leg: _MutableLiveLeg) -> bool:
        """Evaluate exposure using one lock-owned coherent state."""

        return cls._snapshot_leg(leg).exposure_possible

    def _find_unfinished_record(self, spec: LegSpec) -> _MutableLiveLeg | None:
        """Return a matching lock-owned record; caller must hold ``_lock``."""

        for leg in self._legs.values():
            if leg.spec.active_signature == spec.active_signature and self._exposure_possible(leg):
                return leg
        return None

    def find_unfinished(self, spec: LegSpec) -> LiveLegState | None:
        """Find possible exposure matching a signal retry, regardless of correlation."""

        with self._lock:
            leg = self._find_unfinished_record(spec)
            return self._snapshot_leg(leg) if leg is not None else None

    def register(self, spec: LegSpec) -> LiveLegState:
        """Register before submission, reusing any matching unfinished leg."""

        with self._lock:
            existing = self._find_unfinished_record(spec)
            if existing is not None:
                return self._snapshot_leg(existing)
            exposure_id = self._exposure_id(spec)
            if exposure_id in self._legs:
                raise ValueError("correlation_id/role already belongs to a completed leg")
            leg = _MutableLiveLeg(
                exposure_id=exposure_id,
                spec=spec,
                requested_quantity=spec.target_quantity,
                remaining_quantity=spec.target_quantity,
            )
            self._legs[exposure_id] = leg
            return self._snapshot_leg(leg)

    def get(self, exposure_id: str) -> LiveLegState:
        """Return a coherent immutable snapshot or raise for a programming error."""

        with self._lock:
            return self._snapshot_leg(self._legs[str(exposure_id)])

    def active_states(self) -> tuple[LiveLegState, ...]:
        """Return every leg whose broker exposure is known or possible."""

        with self._lock:
            return tuple(self._snapshot_leg(leg) for leg in self._legs.values() if self._exposure_possible(leg))

    def start_attempt(
        self,
        exposure_id: str,
        intent: OrderIntent,
        requested_quantity: int,
    ) -> OrderAttemptHandle:
        """Mark a submission in-flight and return its immutable identity."""

        if not isinstance(intent, OrderIntent):
            raise ValueError("intent must be OrderIntent.OPEN or OrderIntent.CLOSE")
        if type(requested_quantity) is not int or requested_quantity <= 0:
            raise ValueError("requested_quantity must be a positive integer")
        with self._lock:
            leg = self._legs[str(exposure_id)]
            # One attempt at a time per leg: while the previous order may still
            # fill, submitting more quantity could overshoot the target.
            if leg.latest_attempt is not None and not leg.latest_attempt.terminal:
                raise RuntimeError("the previous order attempt is not terminal")
            if intent is OrderIntent.OPEN:
                # A leg is one-directional: once closing has begun, "reopening"
                # would blur which fills belong to the entry and which to the
                # exit.  A fresh signal gets a fresh correlation instead.
                if leg.closing_started:
                    raise RuntimeError(
                        "cannot open a live leg after closing has started; "
                        "use a new correlation after broker-confirmed flat"
                    )
                safe_quantity = self._snapshot_leg(leg).safe_open_retry_quantity
                phase = "E"
            else:
                safe_quantity = self._snapshot_leg(leg).safe_close_retry_quantity
                phase = "X"
            # The caller must ask for EXACTLY the quantity the ledger deems
            # safe (see the safe-retry properties). Rejecting any other number
            # makes it impossible to bypass the quantity maths by accident.
            if requested_quantity != safe_quantity:
                raise ValueError(f"requested_quantity must equal safe retry quantity {safe_quantity}")
            sequence = leg.attempt_count + 1
            tag = build_order_tag(
                leg.spec.strategy,
                leg.spec.correlation_id,
                leg.spec.role,
                phase,
                sequence,
            )
            leg.attempt_count = sequence
            if intent is OrderIntent.CLOSE:
                leg.closing_started = True
            # From this instant until a terminal result arrives, the leg's
            # true exposure is unknown -- the order is (about to be) in flight.
            leg.exposure_indeterminate = True
            leg.latest_attempt = _MutableOrderAttempt(
                intent=intent,
                sequence=sequence,
                order_tag=tag,
                requested_quantity=requested_quantity,
                remaining_quantity=requested_quantity,
            )
            return OrderAttemptHandle(
                exposure_id=leg.exposure_id,
                sequence=sequence,
                order_tag=tag,
            )

    def apply_result(
        self,
        handle: OrderAttemptHandle,
        result: OrderResult,
    ) -> LiveLegState:
        """Apply cumulative evidence only to the attempt identified by ``handle``."""

        if not isinstance(handle, OrderAttemptHandle):
            raise ValueError("handle must be an OrderAttemptHandle")
        if not isinstance(result, OrderResult):
            raise ValueError("result must be an OrderResult")
        with self._lock:
            leg = self._legs[str(handle.exposure_id)]
            attempt = leg.latest_attempt
            if attempt is None:
                raise RuntimeError("cannot apply a result before start_attempt")
            # Identity checks: a status probe launched for attempt #1 can
            # return AFTER a retry already started attempt #2.  The handle's
            # sequence/tag pin the evidence to the attempt it describes, so a
            # late reply can never be mistaken for news about the newer order.
            if attempt.sequence != handle.sequence or attempt.order_tag != handle.order_tag:
                raise RuntimeError("stale order-attempt handle cannot mutate the current attempt")
            if result.requested_quantity != attempt.requested_quantity:
                raise ValueError("broker result requested quantity changed within an attempt")
            if attempt.order_id and result.order_id and result.order_id != attempt.order_id:
                raise ValueError("broker result order id changed within an attempt")
            # Brokers report fills CUMULATIVELY per order; a count that shrank
            # means one of the two snapshots is wrong, and guessing which
            # would corrupt the quantity books.
            if result.filled_quantity < attempt.filled_quantity:
                raise ValueError("broker cumulative filled quantity moved backwards")
            broker_state = str(result.broker_state).upper().strip()
            # PARTIAL alone is not final -- the order may still be filling.
            # It becomes terminal only alongside a terminal broker label
            # (e.g. partially filled, then cancelled): the same rule the
            # adapters' fill-confirmation loops poll by.
            terminal = result.status in {OrderStatus.FILLED, OrderStatus.REJECTED} or (
                result.status is OrderStatus.PARTIAL
                and broker_state in _TERMINAL_BROKER_STATES
            )
            if (
                attempt.terminal
                and not terminal
                and result.filled_quantity == attempt.filled_quantity
            ):
                # Broker/status probes can arrive out of order.  Equal-fill
                # non-terminal evidence is strictly weaker than a terminal
                # snapshot already applied to this attempt, so it cannot reopen
                # uncertainty or undo an entry/flat decision.
                return self._snapshot_leg(leg)
            # The DELTA is what this snapshot newly proves: cumulative report
            # minus what this attempt had already been credited.  Applying
            # deltas (never totals) is what makes a repeated or re-ordered
            # status report unable to double-count a fill.
            fill_delta = result.filled_quantity - attempt.filled_quantity
            new_entry_filled = leg.filled_quantity
            new_entry_remaining = leg.remaining_quantity
            new_live_quantity = leg.confirmed_live_quantity
            if attempt.intent is OrderIntent.OPEN:
                # Entry fills grow both the entry tally and the live position.
                if leg.filled_quantity + fill_delta > leg.spec.target_quantity:
                    raise ValueError("open fills exceed the leg target quantity")
                new_entry_filled += fill_delta
                new_entry_remaining = leg.spec.target_quantity - new_entry_filled
                new_live_quantity += fill_delta
            else:
                # Close fills only shrink the live position; the entry tally
                # keeps recording how much was ever opened.
                if fill_delta > leg.confirmed_live_quantity:
                    raise ValueError("close fills exceed confirmed live quantity")
                new_live_quantity -= fill_delta

            order_id = str(result.order_id or attempt.order_id)

            # All validation and derived calculations happen before this block,
            # so a rejected result cannot leave half-updated exposure fields.
            leg.filled_quantity = new_entry_filled
            leg.remaining_quantity = new_entry_remaining
            leg.confirmed_live_quantity = new_live_quantity
            attempt.order_id = order_id
            attempt.filled_quantity = result.filled_quantity
            attempt.remaining_quantity = result.remaining_quantity
            attempt.status = result.status
            attempt.broker_state = broker_state
            attempt.reason = str(result.reason)
            attempt.terminal = terminal
            leg.exposure_indeterminate = not terminal
            return self._snapshot_leg(leg)
