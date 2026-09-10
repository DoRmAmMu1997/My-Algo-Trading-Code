"""Pure data shaping for the read-only monitoring dashboard.

The runner already publishes a trade event for every entry and every exit
(`publish_trade_event` in the master). This module turns that flat, ordered
event stream plus a set of live worker readings into the document the browser
page renders: today's closed trades grouped by strategy, the entry TIME of
each still-open position, per-strategy running P&L, and a session total.

Three rules shape everything here:

1. **Pure.** No threads, no sockets, no disk, no clock, no `.env`. Every
   function is a plain transformation of its arguments, which is why the
   pairing logic below can be exhaustively unit-tested without a runner.
   The module must never call `_env_*` or `os.getenv`: all configuration is
   read in the master, which is the only place `check_env_config` audits
   against `env.example` for the whole runner.
2. **Never invent a number.** Where a value cannot be derived honestly the
   result is `None`, and the page renders a dash. A monitoring surface that
   guesses is worse than one that admits it does not know -- the operator may
   act on what it shows.
3. **Reporting only.** Nothing here feeds risk, sizing or execution. The
   broker remains the authority on what is actually open.

The counterpart transport module is `dashboard_server.py`; the collector that
reads live worker state lives in the master, beside the other functions that
reach into workers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------
# `publish_trade_event` is a single choke point for several kinds of event, not
# just trades. Only ENTRY and EXIT move the open book; everything else is an
# operator notice. Listing them explicitly (rather than treating "not ENTRY" as
# "close") is what keeps EXIT_FAILED from silently closing a position that is
# in fact still open.
ENTRY_ACTION = "ENTRY"
EXIT_ACTION = "EXIT"

#: An exit the broker did NOT confirm. The position stays OPEN and the runner
#: keeps retrying, so this must never pair as a close -- it only flags the
#: entry it refers to.
EXIT_FAILED_ACTION = "EXIT_FAILED"

#: Everything the runner publishes that is neither a trade nor an exit failure.
#: Rendered as notices, never allowed to disturb pairing.
NOTICE_ACTIONS = frozenset(
    {
        EXIT_FAILED_ACTION,
        "INDETERMINATE_EXPOSURE",
        "UNHEDGED_LEG_OPEN",
        "UNHEDGED_LEG_CLOSED",
        "MARKET_DATA_AUTO_SQUARE_OFF",
        "SHUTDOWN_DEGRADED",
    }
)

#: Notices an operator should not be able to miss.
URGENT_ACTIONS = frozenset({EXIT_FAILED_ACTION, "INDETERMINATE_EXPOSURE"})

#: How confident the ENTRY -> EXIT match is. Anything other than EXACT is shown
#: on the page, because an operator reading an entry time deserves to know how
#: it was arrived at.
PAIR_EXACT = "EXACT"
PAIR_OVERLAP = "OVERLAP"
PAIR_DIRECTION = "DIRECTION"
PAIR_SOLE_OPEN = "SOLE_OPEN"
PAIR_UNPAIRED = "UNPAIRED"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenEntry:
    """One ENTRY event that has not yet been matched by an EXIT."""

    strategy: str
    timestamp: str
    time_text: str | None
    direction: str
    symbols: frozenset[str]
    quantity: int
    mode: str
    legs: tuple[Mapping[str, object], ...]
    sequence: int
    #: Set when an EXIT_FAILED names this entry. The position is STILL OPEN --
    #: the runner keeps retrying the close -- and the page must say so.
    exit_failed: bool = False
    last_exit_attempt: str | None = None


@dataclass(frozen=True)
class ClosedTrade:
    """One completed round trip, ready to render."""

    strategy: str
    entry_time: str | None
    exit_time: str | None
    direction: str
    symbols: tuple[str, ...]
    quantity: int
    entry_price: float | None
    exit_price: float | None
    pnl: float | None
    reason: str
    mode: str
    pair_confidence: str

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form; the page reads these key names directly."""

        return {
            "strategy": self.strategy,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "direction": self.direction,
            "symbols": list(self.symbols),
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "reason": self.reason,
            "mode": self.mode,
            "pair_confidence": self.pair_confidence,
        }


@dataclass(frozen=True)
class Notice:
    """A non-trade event worth showing the operator verbatim."""

    timestamp: str
    time_text: str | None
    strategy: str
    action: str
    detail: str
    urgent: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "time": self.time_text,
            "strategy": self.strategy,
            "action": self.action,
            "detail": self.detail,
            "urgent": self.urgent,
        }


@dataclass(frozen=True)
class TradeLedger:
    """The whole event stream, resolved into closed trades and open entries.

    `open_entries` is NOT the open book. Its residue can contain ghosts -- an
    entry whose exit was published before the process restarted, say -- so the
    live worker reading is always the sole authority on what is open. This
    ledger only supplies each open position's entry TIME, which the live
    reading cannot provide for a position opened before a restart.
    """

    closed: tuple[ClosedTrade, ...] = ()
    open_entries: tuple[OpenEntry, ...] = ()
    notices: tuple[Notice, ...] = ()
    unpaired_exits: int = 0


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------
def event_symbol_key(event: Mapping[str, object]) -> frozenset[str]:
    """The set of contract symbols an event's legs name, upper-cased.

    This is the pairing key, and it has to be, because several workers publish
    more than one independent position under ONE strategy name:

    * Delta-0.2 and the long strangle run a CE side and a PE side at once;
    * SL Hunting mirrors every NIFTY entry with a BankNIFTY leg, and both
      carry the same LONG/SHORT direction, so only the symbol tells them apart.

    Returns an empty set for an event with no readable legs; callers treat that
    as "unkeyable" and fall back down the resolution ladder.
    """

    legs = event.get("legs")
    if not isinstance(legs, Iterable) or isinstance(legs, str | bytes):
        return frozenset()
    symbols = set()
    for leg in legs:
        if not isinstance(leg, Mapping):
            continue
        symbol = str(leg.get("symbol", "")).strip().upper()
        if symbol:
            symbols.add(symbol)
    return frozenset(symbols)


def parse_event_timestamp(value: object) -> str | None:
    """Extract "HH:MM:SS" from an event's `ts`, or None if unreadable.

    `publish_trade_event` stamps `"%Y-%m-%d %H:%M:%S"` naive local time. The
    page only ever shows the clock time -- the whole document is one session --
    so the date half is dropped here rather than in the browser. Anything that
    does not look like that stamp yields None, and the page shows a dash; a
    malformed timestamp must not raise on a monitoring path.
    """

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    tail = text.split(" ")[-1] if " " in text else text
    parts = tail.split(":")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    hours, minutes, seconds = (int(part) for part in parts)
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _finite_float(value: object) -> float | None:
    """A real number, or None. Rejects NaN and infinity, never raises."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _leg_price(event: Mapping[str, object], key: str) -> float | None:
    """The first readable per-leg price under `key`.

    Multi-leg events (the hedged pairs) have two prices and no single "the"
    price; the table shows the primary leg's and the symbol column makes the
    pairing obvious. Summing them would be meaningless and averaging them would
    be a lie, so neither is attempted.
    """

    legs = event.get("legs")
    if not isinstance(legs, Iterable) or isinstance(legs, str | bytes):
        return None
    for leg in legs:
        if isinstance(leg, Mapping):
            price = _finite_float(leg.get(key))
            if price is not None:
                return price
    return None


def _event_legs(event: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """The event's readable legs, skipping anything that is not a mapping."""

    legs = event.get("legs")
    if not isinstance(legs, Iterable) or isinstance(legs, str | bytes):
        return ()
    return tuple(leg for leg in legs if isinstance(leg, Mapping))


def _int_or_zero(value: object) -> int:
    """A whole number from an untyped event field, or zero. Never raises."""

    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _notice_detail(event: Mapping[str, object]) -> str:
    """A short, human line for the notices strip.

    Deliberately assembled from a fixed set of keys rather than dumping the
    event: the strip has to stay readable at a glance during a live session.
    """

    parts = []
    for key in ("reason", "side", "symbol", "status", "broker_state"):
        value = event.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def _resolve_open_entry(
    exit_event: Mapping[str, object],
    strategy: str,
    symbols: frozenset[str],
    open_by_key: dict[tuple[str, frozenset[str]], list[OpenEntry]],
) -> tuple[OpenEntry | None, str]:
    """Find the ENTRY an exit refers to, and say how confident the match is.

    A ladder, most specific first, because the cost of a wrong match is a wrong
    entry time shown beside a real trade:

    1. EXACT     -- same strategy AND same symbol set. Separates Delta-0.2's
                    two spreads and SL Hunting's NIFTY leg from its BankNIFTY
                    mirror, which is the only thing that can.
    2. OVERLAP   -- the open entry sharing the most symbols with this exit.
                    Covers a partial-leg close. Ranked ABOVE direction because
                    SL Hunting's two legs share a direction but never a symbol.
    3. DIRECTION -- the oldest open entry with the same LONG/SHORT (or CE/PE)
                    direction. A working backstop for Delta-0.2 and the strangle.
    4. SOLE_OPEN -- the strategy has exactly one open entry, so there is no
                    ambiguity left to resolve.
    5. UNPAIRED  -- nothing matched. The entry time renders as a dash.

    Within a bucket the OLDEST entry wins, which is correct for a leg that is
    stopped out and re-entered on the same strike: only one is ever open.
    """

    exact = open_by_key.get((strategy, symbols))
    if symbols and exact:
        return exact[0], PAIR_EXACT

    candidates = [
        (key, entries) for key, entries in open_by_key.items() if key[0] == strategy and entries
    ]
    if not candidates:
        return None, PAIR_UNPAIRED

    if symbols:
        best: tuple[int, int, OpenEntry] | None = None
        for key, entries in candidates:
            shared = len(key[1] & symbols)
            if shared == 0:
                continue
            entry = entries[0]
            if best is None or shared > best[0] or (shared == best[0] and entry.sequence < best[1]):
                best = (shared, entry.sequence, entry)
        if best is not None:
            return best[2], PAIR_OVERLAP

    direction = str(exit_event.get("direction", "")).strip().upper()
    if direction:
        matching = [
            entries[0] for _, entries in candidates if entries[0].direction == direction
        ]
        if matching:
            return min(matching, key=lambda entry: entry.sequence), PAIR_DIRECTION

    open_entries = [entry for _, entries in candidates for entry in entries]
    if len(open_entries) == 1:
        return open_entries[0], PAIR_SOLE_OPEN
    return None, PAIR_UNPAIRED


def _drop_open_entry(
    open_by_key: dict[tuple[str, frozenset[str]], list[OpenEntry]],
    entry: OpenEntry,
) -> None:
    """Remove a matched entry from the open book, pruning empty buckets."""

    key = (entry.strategy, entry.symbols)
    bucket = open_by_key.get(key)
    if bucket is None:
        return
    for index, candidate in enumerate(bucket):
        if candidate.sequence == entry.sequence:
            bucket.pop(index)
            break
    if not bucket:
        open_by_key.pop(key, None)


def _flag_exit_failed(
    open_by_key: dict[tuple[str, frozenset[str]], list[OpenEntry]],
    entry: OpenEntry,
    timestamp: str,
) -> None:
    """Mark an entry whose close was NOT confirmed. It stays open."""

    key = (entry.strategy, entry.symbols)
    bucket = open_by_key.get(key)
    if bucket is None:
        return
    for index, candidate in enumerate(bucket):
        if candidate.sequence == entry.sequence:
            bucket[index] = OpenEntry(
                **{
                    **candidate.__dict__,
                    "exit_failed": True,
                    "last_exit_attempt": parse_event_timestamp(timestamp),
                }
            )
            break


def pair_trade_events(events: Sequence[Mapping[str, object]]) -> TradeLedger:
    """Walk the event stream once and resolve it into a ledger.

    Events arrive in chronological order because every producer appends under a
    lock. The walk is deliberately forgiving: a malformed event is skipped or
    demoted to a notice rather than raising, because this runs on a monitoring
    thread while real money is trading.

    An unpaired EXIT is still a COMPLETE row -- the exit event itself carries
    the entry price, the exit price and the realized P&L. Only its entry time
    is unknown, and that renders as a dash rather than being fabricated.
    """

    open_by_key: dict[tuple[str, frozenset[str]], list[OpenEntry]] = {}
    closed: list[ClosedTrade] = []
    notices: list[Notice] = []
    unpaired_exits = 0

    for sequence, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        action = str(event.get("action", "")).strip().upper()
        strategy = str(event.get("strategy", "")).strip() or "(unknown)"
        timestamp = str(event.get("ts", ""))
        time_text = parse_event_timestamp(timestamp)
        symbols = event_symbol_key(event)

        if action == ENTRY_ACTION:
            entry = OpenEntry(
                strategy=strategy,
                timestamp=timestamp,
                time_text=time_text,
                direction=str(event.get("direction", "")).strip().upper(),
                symbols=symbols,
                quantity=_int_or_zero(event.get("quantity")),
                mode=str(event.get("mode", "")).strip().upper(),
                legs=_event_legs(event),
                sequence=sequence,
            )
            open_by_key.setdefault((strategy, symbols), []).append(entry)
            continue

        if action == EXIT_ACTION:
            matched, confidence = _resolve_open_entry(event, strategy, symbols, open_by_key)
            if matched is not None:
                _drop_open_entry(open_by_key, matched)
            else:
                unpaired_exits += 1
            closed.append(
                ClosedTrade(
                    strategy=strategy,
                    entry_time=matched.time_text if matched is not None else None,
                    exit_time=time_text,
                    direction=str(event.get("direction", "")).strip().upper(),
                    symbols=tuple(sorted(symbols)),
                    quantity=_int_or_zero(event.get("quantity")),
                    entry_price=_leg_price(event, "entry_price"),
                    exit_price=_leg_price(event, "exit_price"),
                    pnl=_finite_float(event.get("pnl")),
                    reason=str(event.get("reason", "")).strip(),
                    mode=str(event.get("mode", "")).strip().upper(),
                    pair_confidence=confidence,
                )
            )
            continue

        if action == EXIT_FAILED_ACTION:
            # An unconfirmed exit closes NOTHING. The runner keeps retrying and
            # the position is still live exposure, so the entry stays in the
            # open book and is merely flagged.
            matched, _ = _resolve_open_entry(event, strategy, symbols, open_by_key)
            if matched is not None:
                _flag_exit_failed(open_by_key, matched, timestamp)

        if action:
            notices.append(
                Notice(
                    timestamp=timestamp,
                    time_text=time_text,
                    strategy=strategy,
                    action=action,
                    detail=_notice_detail(event),
                    urgent=action in URGENT_ACTIONS,
                )
            )

    open_entries = tuple(
        sorted(
            (entry for bucket in open_by_key.values() for entry in bucket),
            key=lambda entry: entry.sequence,
        )
    )
    return TradeLedger(
        closed=tuple(closed),
        open_entries=open_entries,
        notices=tuple(notices),
        unpaired_exits=unpaired_exits,
    )


def closed_trades_by_strategy(ledger: TradeLedger) -> list[dict[str, object]]:
    """Group closed trades per strategy, newest strategy activity first.

    Each group carries its own subtotal so the page's collapsed summary line is
    useful without expanding it. Groups are ordered by trade count then name,
    which keeps the busiest strategies at the top and the order stable.
    """

    grouped: dict[str, list[ClosedTrade]] = {}
    for trade in ledger.closed:
        grouped.setdefault(trade.strategy, []).append(trade)

    groups = []
    for strategy, trades in grouped.items():
        priced = [trade.pnl for trade in trades if trade.pnl is not None]
        groups.append(
            {
                "strategy": strategy,
                "trades": len(trades),
                "realized": round(sum(priced), 2),
                "unpriced": len(trades) - len(priced),
                "rows": [trade.as_dict() for trade in trades],
            }
        )
    groups.sort(key=lambda group: (-_int_or_zero(group["trades"]), str(group["strategy"])))
    return groups


def entry_time_for(
    ledger: TradeLedger,
    strategy: str,
    symbols: frozenset[str],
) -> tuple[str | None, bool]:
    """Best-known entry time for a LIVE position, and whether its exit failed.

    Matched against the ledger's open entries by the same symbol-set rule the
    pairing uses. Returns `(None, False)` when the ledger cannot say -- for a
    position resumed from a previous process, for instance -- and the caller
    then falls back to the position's own `entry_timestamp`.
    """

    candidates = [entry for entry in ledger.open_entries if entry.strategy == strategy]
    if not candidates:
        return None, False
    if symbols:
        exact = [entry for entry in candidates if entry.symbols == symbols]
        if exact:
            return exact[0].time_text, exact[0].exit_failed
        overlapping = [entry for entry in candidates if entry.symbols & symbols]
        if overlapping:
            best = max(overlapping, key=lambda entry: len(entry.symbols & symbols))
            return best.time_text, best.exit_failed
    if len(candidates) == 1:
        return candidates[0].time_text, candidates[0].exit_failed
    return None, False


# ---------------------------------------------------------------------------
# Rollups
# ---------------------------------------------------------------------------
def strategy_rows(worker_views: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Per-strategy realized / open / total, sorted by total P&L.

    `open` and `total` are `None` -- not zero -- whenever ANY of that
    strategy's open positions could not be marked. Showing a partial sum as
    though it were the whole would understate an open loss, which is the one
    direction a monitoring page must never be wrong in. `realized` is always
    shown, because it is banked and never in doubt.
    """

    rows = []
    for view in worker_views:
        positions = view.get("positions")
        positions = positions if isinstance(positions, Sequence) else ()
        realized = _finite_float(view.get("realized_pnl")) or 0.0

        marks = [
            _finite_float(position.get("unrealized_pnl"))
            for position in positions
            if isinstance(position, Mapping)
        ]
        unpriced = any(mark is None for mark in marks)
        open_pnl = None if unpriced else round(sum(mark or 0.0 for mark in marks), 2)

        rows.append(
            {
                "strategy": str(view.get("strategy", "")),
                "mode": str(view.get("mode", "PAPER")),
                "live_trading": bool(view.get("live_trading", False)),
                "snapshot_valid": bool(view.get("snapshot_valid", True)),
                "trades": _int_or_zero(view.get("completed_trades")),
                "realized": round(realized, 2),
                "open": open_pnl,
                "total": None if open_pnl is None else round(realized + open_pnl, 2),
                "open_positions": len(marks),
            }
        )

    # Losers first would bury the day's worst strategy at the bottom of a long
    # table, so sort by total ascending is tempting -- but the operator reads
    # this to find what is working, and an unpriced row must not float to an
    # arbitrary end. Unpriced rows sort last, then by total descending.
    rows.sort(
        key=lambda row: (
            row["total"] is None,
            -(row["total"] if isinstance(row["total"], int | float) else 0.0),
            row["strategy"],
        )
    )
    return rows


def session_totals(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Whole-session rollup, honest about what it could not price."""

    realized = round(sum(_finite_float(row.get("realized")) or 0.0 for row in rows), 2)
    priced = [row for row in rows if row.get("open") is not None]
    unpriced = len(rows) - len(priced)
    open_pnl = round(sum(_finite_float(row.get("open")) or 0.0 for row in priced), 2)
    return {
        "realized": realized,
        # A partial sum, and labelled as one: `unpriced_strategies` is what the
        # page prints beside it so the number is never read as complete.
        "open": open_pnl,
        "total": round(realized + open_pnl, 2),
        "unpriced_strategies": unpriced,
        "trades": sum(_int_or_zero(row.get("trades")) for row in rows),
        "open_positions": sum(_int_or_zero(row.get("open_positions")) for row in rows),
        "strategies": len(rows),
    }


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentInputs:
    """Everything the page needs, gathered by the caller.

    Bundled into one object so `build_dashboard_document` keeps a single
    parameter and adding a pane later does not ripple through every call site.
    """

    generated_at: str
    version: int
    poll_seconds: float
    session: Mapping[str, object]
    worker_views: Sequence[Mapping[str, object]]
    ledger: TradeLedger
    chart: Mapping[str, object] = field(default_factory=dict)
    feed: Mapping[str, object] = field(default_factory=dict)


def _open_position_rows(
    worker_views: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Flatten every worker's positions into one table, tagged by strategy."""

    rows: list[dict[str, object]] = []
    for view in worker_views:
        positions = view.get("positions")
        if not isinstance(positions, Sequence) or isinstance(positions, str | bytes):
            continue
        for position in positions:
            if isinstance(position, Mapping):
                rows.append(dict(position, strategy=view.get("strategy", "")))
    return rows


def build_dashboard_document(inputs: DocumentInputs) -> dict[str, object]:
    """Assemble the single JSON document `/api/state` serves."""

    rows = strategy_rows(inputs.worker_views)
    return {
        "generated_at": inputs.generated_at,
        "version": inputs.version,
        "poll_seconds": inputs.poll_seconds,
        "session": dict(inputs.session),
        "totals": session_totals(rows),
        "strategies": rows,
        "open_positions": _open_position_rows(inputs.worker_views),
        "closed_trades": closed_trades_by_strategy(inputs.ledger),
        "notices": [notice.as_dict() for notice in inputs.ledger.notices],
        "unpaired_exits": inputs.ledger.unpaired_exits,
        "chart": dict(inputs.chart),
        "feed": dict(inputs.feed),
    }


def render_document_bytes(document: Mapping[str, object]) -> bytes:
    """Serialize a document to compact UTF-8 JSON.

    `allow_nan=False` on purpose: JSON has no NaN, so the default would emit
    bare `NaN` and every browser's `JSON.parse` would reject the whole
    response. Failing here instead makes a bad value a caught build error that
    keeps the last good page on screen, rather than a blank dashboard.
    """

    return json.dumps(
        document,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
