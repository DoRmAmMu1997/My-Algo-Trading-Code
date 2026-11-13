"""
Position-group target monitor: square off a group of carried legs at its target.

The problem it solves
---------------------
The operator carries option-selling positions for a week to three months in a
separate broker account (Zerodha / mStock), adjusts them by hand, and wants the
whole strategy closed the moment its P&L reaches a target -- without watching
the screen all day.  This module is that watcher.  It runs as one optional
daemon thread inside the master runner (``GROUP_MONITOR_ENABLED``).

How one tick works
------------------
1. Re-read ``position_groups.json`` if it changed on disk (the operator edits it
   while the runner is up).  An invalid file keeps the last good copy watched
   and sends one alert.
2. For every ACTIVE group, price each leg from the runner's own Dhan feed (each
   leg is subscribed through the shared store, so websocket and REST both
   work), and compute ``booked_pnl + sum((ltp - entry_price) * quantity)``.
   A group with ANY leg's price missing or stale is skipped for that tick.
3. If the P&L sits at or above ``target`` for ``confirm_ticks`` CONSECUTIVE
   fresh ticks, the group is re-priced ONCE at prices it could really be closed
   at -- shorts at the ask, longs at the bid, from Dhan's option chain.  Only
   if THAT P&L is also at or above ``target`` is the group squared off.  A last
   traded price can sit far from the live market on a quiet strike, and the
   consecutive ticks all re-read that same print, so it may raise the question
   but never answers it.  An unreadable quote sends no orders.

Square-off, and why it is shaped this way
-----------------------------------------
* Before any order, the broker's position book must still show every leg with
  the same sign and at least the file's quantity.  If the operator has already
  closed or resized something by hand, sending the file's orders could open a
  brand-new position, so a mismatch stops the group with NO orders.
* Shorts are bought back FIRST, longs sold after.  Selling the hedges first
  would leave naked shorts for a few seconds -- a margin spike and, in a fast
  market, a real risk.
* Each order is split under the exchange freeze quantity (lot size and freeze
  limit come from the Dhan instrument CSV, through ``order_splitting``).
* Every order must come back ``FILLED``.  Anything else (REJECTED, PARTIAL,
  UNKNOWN, an exception) stops the group immediately and alerts with what is
  done and what is still open.  There is NO automatic retry: an unproven
  order retried blindly can close twice and flip into a fresh position.
* The group is written ``EXITING`` to the state file before the first order,
  and no order is sent if that write fails.  A crash mid-exit therefore comes
  back as ``EXIT_FAILED`` -- never re-fired.

Groups that reach ``SQUARED_OFF`` or ``EXIT_FAILED`` stay that way across
restarts (keyed by a fingerprint of name + legs) until the operator removes or
changes them in the file.

The thread knows nothing about the master: every dependency (price reads,
subscriptions, alerts, broker clients, the clock) is injected, so the whole
module is testable without a broker or a feed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Sibling modules are imported by bare name, like the broker adapters do, so
# this file resolves the same way whether the master imports it as
# ``Dependencies.position_group_monitor`` or a test imports it directly.
# ``broker_contract`` registers both of its names, so ``OrderStatus`` here is
# the very object the broker clients return.
_DEPENDENCIES_DIR = Path(__file__).resolve().parent
if str(_DEPENDENCIES_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPENDENCIES_DIR))

from broker_contract import OrderStatus, PositionDetailClient  # noqa: E402
from order_splitting import split_order_quantity  # noqa: E402
from position_groups import (  # noqa: E402
    Group,
    GroupConfig,
    GroupConfigError,
    Leg,
    load_config,
    write_document_atomic,
)

log = logging.getLogger(__name__)

STRATEGY_LABEL = "Position Groups"
MARKET_OPEN = time(9, 15)
# The last moment a square-off may START.  A group is several orders, each of
# which can take seconds to prove; starting later risks buying the shorts back
# and then having the hedge sales refused at the 15:30 close.
MARKET_LAST_ACTION = time(15, 25)
# How long to wait before re-trying a groups file that could not be loaded.
APPLY_RETRY_SECONDS = 30.0
# How long to wait before re-reading the bid/ask after it did not support an
# exit.  Dhan allows one option-chain request per underlying every 3 seconds.
QUOTE_RECHECK_SECONDS = 15.0

ACTIVE = "ACTIVE"
EXITING = "EXITING"
SQUARED_OFF = "SQUARED_OFF"
EXIT_FAILED = "EXIT_FAILED"

# Broker exchange code -> (Dhan EXCH_ID in the instrument CSV, Dhan feed segment).
_EXCHANGE_TO_DHAN = {"NFO": ("NSE", "NSE_FNO"), "BFO": ("BSE", "BSE_FNO")}


# ---------------------------------------------------------------------------
# Dhan contract lookup
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DhanContract:
    """Where to read a leg's price, and how big one exchange order may be."""

    segment: str
    security_id: int
    lot_size: int
    freeze_qty: int
    # The UNDERLYING's own Dhan id and segment (NIFTY is 13 on IDX_I).  Dhan's
    # option chain -- where a live bid/ask comes from -- is asked for by
    # underlying and expiry, not by the option's own id.
    underlying_security_id: int = 0
    underlying_segment: str = ""


ContractKey = tuple[str, str, date, int, str]


def _contract_key(exchange_id: str, underlying: str, expiry: date, strike: float, right: str) -> ContractKey:
    # Strikes are keyed in paise so a stock option's 1312.5 is exact.
    return (exchange_id, underlying.upper(), expiry, round(float(strike) * 100), right.upper())


def build_dhan_contract_index(
    csv_path: Path, underlyings: Iterable[str] | None = None
) -> dict[ContractKey, DhanContract]:
    """Index the NSE/BSE options in the Dhan instrument CSV by their identity.

    ``underlyings`` narrows the index to the names given (``{"NIFTY"}``), which
    is what the monitor passes: the file lists every option on two exchanges,
    and a groups file names a handful of legs on one or two underlyings.

    Read on the monitor's own thread, when a group is new -- never on a trading
    path.  Rows with an unparseable field are dropped; a leg whose contract is
    missing is then reported unmonitorable, never guessed.
    """

    columns = [
        "EXCH_ID", "SEGMENT", "INSTRUMENT", "UNDERLYING_SYMBOL", "SM_EXPIRY_DATE",
        "STRIKE_PRICE", "OPTION_TYPE", "SECURITY_ID", "LOT_SIZE", "SM_FREEZE_QTY",
        "UNDERLYING_SECURITY_ID",
    ]
    header = pd.read_csv(csv_path, nrows=0)
    usecols = [column for column in columns if column in header.columns]
    raw = pd.read_csv(csv_path, usecols=usecols, dtype=str, low_memory=False)
    frame = raw[
        raw["EXCH_ID"].str.upper().isin(["NSE", "BSE"])
        & (raw["SEGMENT"].str.upper() == "D")
        & raw["INSTRUMENT"].str.upper().isin(["OPTIDX", "OPTSTK"])
        & raw["OPTION_TYPE"].str.upper().isin(["CE", "PE"])
    ]
    if underlyings is not None:
        wanted = {str(name).strip().upper() for name in underlyings}
        frame = frame[frame["UNDERLYING_SYMBOL"].str.strip().str.upper().isin(wanted)]
    index: dict[ContractKey, DhanContract] = {}
    for row in frame.itertuples(index=False):
        try:
            # Every column was read as text (dtype=str); str() just tells the
            # type checker so.
            expiry = pd.Timestamp(str(row.SM_EXPIRY_DATE)).date()
            strike = float(str(row.STRIKE_PRICE))
            security_id = int(float(str(row.SECURITY_ID)))
        except (TypeError, ValueError):
            continue
        lot = _positive_int_or_zero(getattr(row, "LOT_SIZE", ""))
        freeze = _positive_int_or_zero(getattr(row, "SM_FREEZE_QTY", ""))
        exchange_id = str(row.EXCH_ID).upper()
        segment = "NSE_FNO" if exchange_id == "NSE" else "BSE_FNO"
        # An index option's underlying is an index (IDX_I); a stock option's is
        # the share on the same exchange.  Only the index case has been seen
        # working; a wrong guess for a stock just means "no quote", which
        # never sends an order.
        underlying_segment = (
            "IDX_I" if str(row.INSTRUMENT).upper() == "OPTIDX" else f"{exchange_id}_EQ"
        )
        key = _contract_key(
            exchange_id, str(row.UNDERLYING_SYMBOL).strip(), expiry, strike, str(row.OPTION_TYPE)
        )
        index[key] = DhanContract(
            segment,
            security_id,
            lot,
            freeze,
            _positive_int_or_zero(getattr(row, "UNDERLYING_SECURITY_ID", "")),
            underlying_segment,
        )
    return index


def _positive_int_or_zero(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if math.isfinite(number) and number > 0 and number.is_integer() else 0


def contract_for(leg: Leg, index: Mapping[ContractKey, DhanContract]) -> DhanContract | None:
    exchange = _EXCHANGE_TO_DHAN.get(leg.exchange)
    if exchange is None:
        return None
    return index.get(_contract_key(exchange[0], leg.underlying, leg.expiry, leg.strike, leg.right))


# ---------------------------------------------------------------------------
# Pure decisions
# ---------------------------------------------------------------------------
def group_pnl(group: Group, ltps: Mapping[tuple[str, str, str], float]) -> float | None:
    """``booked_pnl + sum((ltp - entry) * quantity)``, or None if any leg is unpriced."""

    total = group.booked_pnl
    for leg in group.legs:
        ltp = ltps.get(leg.key)
        if ltp is None or not math.isfinite(ltp) or ltp <= 0:
            return None
        total += (ltp - leg.entry_price) * leg.quantity
    return total


def executable_pnl(
    group: Group, quotes: Mapping[tuple[str, str, str], tuple[float, float]]
) -> float | None:
    """The group's P&L if it were closed NOW at the live bid/ask, or None.

    ``quotes`` maps each leg to ``(bid, ask)``.  Closing a short means BUYING
    it back, which costs the ask; closing a long means SELLING it, which earns
    the bid.  That is the P&L a market exit can actually achieve, so it -- not
    the last traded price -- decides whether the target was really reached.

    None when any leg has no usable two-sided quote (missing, zero, or crossed):
    "unknown" must never be mistaken for "at target".
    """

    total = group.booked_pnl
    for leg in group.legs:
        quote = quotes.get(leg.key)
        if quote is None:
            return None
        bid, ask = quote
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask < bid:
            return None
        exit_price = ask if leg.quantity < 0 else bid
        total += (exit_price - leg.entry_price) * leg.quantity
    return total


@dataclass(frozen=True, slots=True)
class ExitOrder:
    leg: Leg
    side: str
    quantity: int


def exit_plan(group: Group, contracts: Mapping[tuple[str, str, str], DhanContract]) -> list[ExitOrder]:
    """Closing orders: every short bought back before any long is sold.

    Each leg is split under its freeze quantity when the Dhan CSV gave both a
    lot size and a freeze limit; otherwise it goes as one order and the broker
    validates it.
    """

    ordered = sorted(group.legs, key=lambda leg: 0 if leg.quantity < 0 else 1)
    orders: list[ExitOrder] = []
    for leg in ordered:
        side = "BUY" if leg.quantity < 0 else "SELL"
        units = abs(leg.quantity)
        contract = contracts.get(leg.key)
        if contract is not None and contract.lot_size and contract.freeze_qty:
            chunks = split_order_quantity(units, contract.freeze_qty, contract.lot_size)
        else:
            chunks = (units,)
        orders.extend(ExitOrder(leg, side, chunk) for chunk in chunks)
    return orders


def _held_quantity(leg: Leg, positions: list[Any]) -> int | None:
    """The broker quantity for this leg IN ITS OWN PRODUCT, or None if ambiguous.

    A broker lists one row per (symbol, product), so a carried NRML leg and an
    intraday MIS trade on the same strike are two rows.  The closing order is
    sent in the leg's product, so it must be checked against that product's row
    alone -- reading the MIS row could pass the check and then flip the NRML
    position.  A single row with no product label (mStock's documented sample
    leaves ``product`` blank) is accepted as the leg's; several rows with no
    matching label are ambiguous and refuse the exit.
    """

    rows = [p for p in positions if p.exchange == leg.exchange and p.symbol == leg.symbol]
    if not rows:
        return 0
    matching = [p for p in rows if str(p.product_type or "").strip().upper() == leg.product]
    if len(matching) == 1:
        return int(matching[0].quantity)
    if len(rows) == 1 and not str(rows[0].product_type or "").strip():
        return int(rows[0].quantity)
    if not matching and all(str(p.product_type or "").strip() for p in rows):
        # Held, but only in other products: nothing of THIS product to close.
        return 0
    return None


def holdings_mismatch(group: Group, positions: list[Any]) -> list[str]:
    """Return why the broker book no longer matches the group (empty = safe to close)."""

    problems: list[str] = []
    for leg in group.legs:
        quantity = _held_quantity(leg, positions)
        if quantity is None:
            problems.append(f"{leg.symbol}: several broker rows, none clearly {leg.product}")
        elif quantity == 0:
            problems.append(f"{leg.symbol}: no {leg.product} position held at {group.broker}")
        elif (quantity < 0) != (leg.quantity < 0) or abs(quantity) < abs(leg.quantity):
            problems.append(f"{leg.symbol}: broker holds {quantity}, file says {leg.quantity}")
    return problems


def within_action_window(now: datetime) -> bool:
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_LAST_ACTION


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
class GroupStateFile:
    """The runner-owned status file: one record per group fingerprint.

    Only the runner writes it.  It exists so a square-off decision survives a
    restart: a group that was closed (or half-closed) is never fired again.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        try:
            document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            records = document.get("groups", {}) if isinstance(document, dict) else {}
            self.records = {str(k): dict(v) for k, v in records.items() if isinstance(v, dict)}
        except (OSError, ValueError) as exc:
            # Keep the unreadable copy for the operator and start empty, loudly.
            # Starting empty cannot re-fire a group that was already closed: the
            # broker pre-check before any order finds its legs gone and stops it.
            log.error("Position-group state file %s unreadable (%s); starting empty.", path, exc)
            with contextlib.suppress(OSError):
                path.replace(path.with_name(path.name + ".unreadable"))
        for record in self.records.values():
            if record.get("status") == EXITING:
                record["status"] = EXIT_FAILED
                record["reason"] = "runner stopped mid-exit; check the broker book by hand"

    def status(self, group: Group) -> str:
        return str(self.records.get(group.fingerprint, {}).get("status") or ACTIVE)

    def update(self, group: Group, **fields: Any) -> bool:
        """Record ``fields`` for the group; return whether they reached the disk.

        The in-memory record is updated either way, so a terminal status keeps
        its hold on this process even while the disk is unwritable.
        """

        record = self.records.setdefault(group.fingerprint, {"name": group.name})
        record.update(fields, updated_at=datetime.now().isoformat(timespec="seconds"))
        try:
            write_document_atomic(self.path, {"schema_version": 1, "groups": self.records})
        except OSError:
            log.exception("Could not write position-group state file %s.", self.path)
            return False
        return True

    def forget(self, group: Group) -> None:
        """Drop an in-memory record that never reached the disk.

        Used only when ``EXITING`` could not be written and therefore no order
        was sent: the group goes back to ACTIVE so a later tick can try again.
        """

        self.records.pop(group.fingerprint, None)


# ---------------------------------------------------------------------------
# The thread
# ---------------------------------------------------------------------------
@dataclass
class _Watch:
    """In-memory view of one group under watch."""

    group: Group
    contracts: dict[tuple[str, str, str], DhanContract]
    streak: int = 0
    last_pnl: float | None = None
    precheck_alerted: bool = False
    quote_alerted: bool = False
    state_alerted: bool = False
    # No bid/ask re-check before this moment (set after one that did not pass).
    recheck_after: datetime | None = None


class PositionGroupMonitor(threading.Thread):
    """Watches every group's P&L and squares a group off at its target."""

    def __init__(
        self,
        *,
        config_path: Path,
        state_path: Path,
        clients: Mapping[str, PositionDetailClient],
        contract_index_loader: Callable[[frozenset[str]], Mapping[ContractKey, DhanContract]],
        subscribe: Callable[[Leg, DhanContract], None],
        unsubscribe: Callable[[DhanContract], None],
        read_ltp: Callable[[DhanContract], float],
        read_quote: Callable[[Leg, DhanContract], tuple[float, float]] | None = None,
        alert: Callable[[dict[str, Any]], None],
        now: Callable[[], datetime],
        stop_event: threading.Event | None = None,
        poll_seconds: float = 1.0,
        confirm_ticks: int = 3,
        log_every_seconds: float = 60.0,
        quote_recheck_seconds: float = QUOTE_RECHECK_SECONDS,
    ) -> None:
        super().__init__(name="PositionGroupMonitor", daemon=True)
        self.config_path = config_path
        self.state = GroupStateFile(state_path)
        self.clients = clients
        self._contract_index_loader = contract_index_loader
        self._subscribe = subscribe
        self._unsubscribe = unsubscribe
        self._read_ltp = read_ltp
        # ``(bid, ask)`` for one leg, or None when no quote source is wired (the
        # last traded price is then all there is).  The runner always wires one.
        self._read_quote = read_quote
        self.quote_recheck_seconds = max(0.0, float(quote_recheck_seconds))
        self._alert = alert
        self._now = now
        self.stop_event = stop_event or threading.Event()
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.confirm_ticks = max(1, int(confirm_ticks))
        self.log_every_seconds = float(log_every_seconds)
        self._config_mtime: float | None = None
        self._bad_mtime_alerted: float | None = None
        # A file that parsed but could not be APPLIED: alerted once per version,
        # retried on a back-off, and never remembered as loaded.
        self._apply_failed_mtime: float | None = None
        self._apply_retry_at: datetime | None = None
        self._watches: dict[str, _Watch] = {}
        # (fingerprint, reason) pairs already alerted as unmonitorable, so a
        # later edit to some OTHER group does not repeat the same alert.
        self._unmonitorable_alerted: set[tuple[str, str]] = set()
        self._last_log: datetime | None = None

    # -- lifecycle --------------------------------------------------------
    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        log.info("Position-group monitor started (file %s).", self.config_path)
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the watcher must keep watching
                log.exception("Position-group monitor tick failed; continuing.")
            self.stop_event.wait(self.poll_seconds)
        log.info("Position-group monitor stopped.")

    # -- one tick -----------------------------------------------------------
    def tick(self) -> None:
        self._reload_if_changed()
        for watch in list(self._watches.values()):
            if self.state.status(watch.group) != ACTIVE:
                continue
            pnl = group_pnl(watch.group, self._prices(watch))
            watch.last_pnl = pnl
            # The clock is read per group: squaring off an earlier group in this
            # same tick can take long enough to pass the last start time.
            if pnl is None or not within_action_window(self._now()):
                watch.streak = 0
                continue
            watch.streak = watch.streak + 1 if pnl >= watch.group.target else 0
            if watch.streak >= self.confirm_ticks:
                executable = self._executable_pnl_at_target(watch, pnl)
                if executable is not None:
                    self._square_off(watch, executable)
        self._maybe_log(self._now())

    def _executable_pnl_at_target(self, watch: _Watch, ltp_pnl: float) -> float | None:
        """Return the P&L the group can really be closed at, if THAT is at target.

        Called once the last-traded-price P&L has held at the target.  That
        price is a memory of the last trade, not an offer: on a quiet far
        strike one odd print stays "the price" for minutes.  So the group is
        re-priced at the live bid/ask before any order.  Returns None -- send
        nothing -- when the quote does not support the exit or cannot be read;
        the next re-check then waits ``quote_recheck_seconds``.
        """

        if self._read_quote is None:
            return ltp_pnl
        group = watch.group
        now = self._now()
        if watch.recheck_after is not None and now < watch.recheck_after:
            return None
        quotes: dict[tuple[str, str, str], tuple[float, float]] = {}
        try:
            for leg in group.legs:
                bid, ask = self._read_quote(leg, watch.contracts[leg.key])
                quotes[leg.key] = (float(bid), float(ask))
        except Exception:  # noqa: BLE001 - an unreadable quote is "no quote"
            log.exception("Position-group quote read failed for %r.", group.name)
            quotes = {}
        executable = executable_pnl(group, quotes)
        if executable is not None and executable >= group.target:
            watch.recheck_after = None
            return executable
        watch.recheck_after = now + timedelta(seconds=self.quote_recheck_seconds)
        if executable is None:
            if not watch.quote_alerted:
                watch.quote_alerted = True
                self._emit(
                    "GROUP_QUOTE_UNREADABLE", group.name,
                    "the last traded prices put the group at its target, but a live "
                    "bid/ask could not be read for every leg, so NO orders were sent; "
                    "it will be re-checked",
                    pnl=ltp_pnl,
                )
        else:
            log.info(
                "Group %r: last-traded P&L %s is at target %s, but closing at the live "
                "bid/ask would give %s; not squaring off.",
                group.name, f"{ltp_pnl:,.0f}", f"{group.target:,.0f}", f"{executable:,.0f}",
            )
        return None

    def _prices(self, watch: _Watch) -> dict[tuple[str, str, str], float]:
        prices: dict[tuple[str, str, str], float] = {}
        for leg in watch.group.legs:
            contract = watch.contracts.get(leg.key)
            if contract is not None:
                prices[leg.key] = float(self._read_ltp(contract) or 0.0)
        return prices

    def _reload_if_changed(self) -> None:
        try:
            mtime = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
        except OSError:
            return
        if mtime == self._config_mtime:
            return
        if (
            self._apply_failed_mtime == mtime
            and self._apply_retry_at is not None
            and self._now() < self._apply_retry_at
        ):
            return
        try:
            config = load_config(self.config_path)
        except (GroupConfigError, OSError) as exc:
            if self._bad_mtime_alerted != mtime:
                self._bad_mtime_alerted = mtime
                problems = exc.problems if isinstance(exc, GroupConfigError) else [str(exc)]
                log.error("Position-groups file invalid; keeping the last good copy: %s", problems)
                self._emit("GROUP_FILE_INVALID", "file", "; ".join(problems)[:500])
            return
        # The file counts as loaded only once it has actually been applied.  If
        # applying fails (say the Dhan contract list is mid-rewrite), remembering
        # it as loaded would leave this thread running all day watching nothing.
        try:
            self._apply(config)
        except Exception as exc:  # noqa: BLE001 - reported and retried
            log.exception("Position-groups file could not be applied; will retry.")
            self._apply_retry_at = self._now() + timedelta(seconds=APPLY_RETRY_SECONDS)
            if self._apply_failed_mtime != mtime:
                self._apply_failed_mtime = mtime
                self._emit(
                    "GROUP_MONITOR_ERROR", "file",
                    f"the groups file is valid but could not be loaded ({type(exc).__name__}); "
                    "new or changed groups in it are NOT watched yet. Retrying every "
                    f"{APPLY_RETRY_SECONDS:.0f}s.",
                )
            return
        self._apply_failed_mtime = None
        self._apply_retry_at = None
        self._config_mtime = mtime

    def _apply(self, config: GroupConfig) -> None:
        # The Dhan contract list is read only when a group is new to us, only for
        # the underlyings the file names, and is not kept afterwards: a groups
        # file is a handful of legs, the list is every option on two exchanges.
        index: Mapping[ContractKey, DhanContract] | None = None
        watches: dict[str, _Watch] = {}
        for group in config.groups:
            existing = self._watches.get(group.fingerprint)
            if existing is not None:
                # Same legs: keep the streak, take the edited target/booked/prices.
                existing.group = group
                watches[group.fingerprint] = existing
                continue
            if group.broker not in self.clients:
                self._unmonitorable(
                    group, f"no logged-in {group.broker} client (no API key, or its token expired)"
                )
                continue
            if index is None:
                index = self._contract_index_loader(
                    frozenset(leg.underlying for each in config.groups for leg in each.legs)
                )
            contracts: dict[tuple[str, str, str], DhanContract] = {}
            missing: list[str] = []
            for leg in group.legs:
                contract = contract_for(leg, index)
                if contract is None:
                    missing.append(leg.symbol)
                else:
                    contracts[leg.key] = contract
            if missing:
                self._unmonitorable(
                    group, "no Dhan contract for " + ", ".join(missing) + "; this group is NOT watched"
                )
                continue
            for leg in group.legs:
                self._subscribe(leg, contracts[leg.key])
            watches[group.fingerprint] = _Watch(group, contracts)
            log.info(
                "Watching group %r [%s]: %d leg(s), target %.0f, booked %.0f, status %s.",
                group.name, group.broker, len(group.legs), group.target, group.booked_pnl,
                self.state.status(group),
            )
        # Release the feed for groups that left the file or changed their legs --
        # but never a contract a watched group still uses. Every leg is held
        # under ONE subscription owner, so releasing a contract the new set
        # just re-registered would silently unsubscribe it.
        still_used = {contract for watch in watches.values() for contract in watch.contracts.values()}
        released: set[DhanContract] = set()
        for fingerprint, old in self._watches.items():
            if fingerprint in watches:
                continue
            for contract in old.contracts.values():
                if contract not in still_used and contract not in released:
                    released.add(contract)
                    self._unsubscribe(contract)
        self._watches = watches

    def _unmonitorable(self, group: Group, reason: str) -> None:
        key = (group.fingerprint, reason)
        if key in self._unmonitorable_alerted:
            return
        self._unmonitorable_alerted.add(key)
        self._emit("GROUP_UNMONITORABLE", group.name, reason)

    def _maybe_log(self, now: datetime) -> None:
        if self._last_log and (now - self._last_log).total_seconds() < self.log_every_seconds:
            return
        self._last_log = now
        for watch in self._watches.values():
            pnl = "n/a (a leg has no fresh price)" if watch.last_pnl is None else f"{watch.last_pnl:,.0f}"
            log.info(
                "Group %r: P&L %s / target %s [%s]",
                watch.group.name, pnl, f"{watch.group.target:,.0f}", self.state.status(watch.group),
            )

    # -- square-off ----------------------------------------------------------
    def _square_off(self, watch: _Watch, pnl: float) -> None:
        group = watch.group
        client = self.clients[group.broker]
        watch.streak = 0
        book = client.list_position_details()
        if book.is_indeterminate:
            # Nothing has been sent, so retrying on a later confirmed tick is safe.
            if not watch.precheck_alerted:
                watch.precheck_alerted = True
                self._emit(
                    "GROUP_TARGET_HIT", group.name,
                    f"target reached but the {group.broker} position book is unreadable "
                    f"({book.reason}); no orders sent, will retry",
                    pnl=pnl,
                )
            return
        mismatch = holdings_mismatch(group, list(book.items))
        if mismatch:
            self.state.update(group, status=EXIT_FAILED, reason="; ".join(mismatch), last_pnl=pnl)
            self._emit(
                "GROUP_EXIT_FAILED", group.name,
                "target reached but the broker book differs from the file, so NO orders "
                "were sent: " + "; ".join(mismatch),
                pnl=pnl,
            )
            return

        try:
            plan = exit_plan(group, watch.contracts)
        except ValueError as exc:
            # E.g. a freeze limit that cannot fit one lot: refuse before any
            # order rather than leave the group half-planned.
            self.state.update(group, status=EXIT_FAILED, reason=f"no exit plan: {exc}", last_pnl=pnl)
            self._emit(
                "GROUP_EXIT_FAILED", group.name,
                f"target reached but no exit plan could be built ({exc}); NO orders were sent",
                pnl=pnl,
            )
            return

        if not self.state.update(group, status=EXITING, trigger_pnl=pnl):
            # Nothing has been sent.  Without EXITING on disk a crash mid-exit
            # would reload this group as ACTIVE, so the exit does not start.
            self.state.forget(group)
            if not watch.state_alerted:
                watch.state_alerted = True
                self._emit(
                    "GROUP_STATE_UNWRITABLE", group.name,
                    f"target reached but {self.state.path.name} could not be written, "
                    "so NO orders were sent; will retry",
                    pnl=pnl,
                )
            return
        self._emit(
            "GROUP_TARGET_HIT", group.name,
            f"P&L {pnl:,.0f} >= target {group.target:,.0f}; squaring off", pnl=pnl,
        )
        # One stamp per ATTEMPT.  A tag is how a lost acknowledgement is found
        # again in the broker's order book, so two attempts on the same group
        # must never share one.  Letters and digits only, 19 characters.
        attempt = self._now().strftime("%H%M%S")
        realised = group.booked_pnl
        done: list[str] = []
        for number, order in enumerate(plan, start=1):
            tag = f"GM{group.fingerprint[:8]}{attempt}{number:03d}"
            try:
                result = client.place_market_order(
                    order.leg.symbol, order.side, order.quantity,
                    order.leg.exchange, order.leg.product, order_tag=tag,
                )
            except Exception as exc:  # noqa: BLE001 - reported, never retried
                # The full text goes to the log, which is redacted; the alert and
                # the state file carry only the exception's type.
                log.exception("Position-group order raised for %r.", group.name)
                self._fail(group, done, order, f"order raised {type(exc).__name__}", pnl)
                return
            if result.status is not OrderStatus.FILLED:
                self._fail(
                    group, done, order,
                    f"{result.status.value} ({result.broker_state}): {result.reason} "
                    f"[order {result.order_id or 'unknown'}, filled {result.filled_quantity}]",
                    pnl,
                )
                return
            fill = result.average_fill_price or self._read_ltp(watch.contracts[order.leg.key])
            # Back to the leg's own sign: a BUY closes a short (negative) leg.
            leg_units = -order.quantity if order.side == "BUY" else order.quantity
            realised += (fill - order.leg.entry_price) * leg_units
            done.append(f"{order.side} {order.quantity} {order.leg.symbol} @ {fill:g}")
        self.state.update(group, status=SQUARED_OFF, realised_pnl=round(realised, 2), orders=done)
        self._emit(
            "GROUP_SQUARED_OFF", group.name,
            f"all {len(done)} order(s) filled; realised about {realised:,.0f}", pnl=realised,
        )

    def _fail(self, group: Group, done: list[str], order: ExitOrder, why: str, pnl: float) -> None:
        reason = (
            f"stopped at {order.side} {order.quantity} {order.leg.symbol}: {why}. "
            f"Already filled: {done or 'nothing'}. Everything else is STILL OPEN -- close it by hand."
        )
        self.state.update(group, status=EXIT_FAILED, reason=reason, orders=done)
        self._emit("GROUP_EXIT_FAILED", group.name, reason, pnl=pnl)

    def _emit(self, action: str, group_name: str, reason: str, *, pnl: float | None = None) -> None:
        failures = {
            "GROUP_EXIT_FAILED", "GROUP_UNMONITORABLE", "GROUP_FILE_INVALID",
            "GROUP_MONITOR_ERROR", "GROUP_QUOTE_UNREADABLE", "GROUP_STATE_UNWRITABLE",
        }
        log_fn = log.error if action in failures else log.warning
        log_fn("%s | %s | %s", action, group_name, reason)
        event: dict[str, Any] = {
            "action": action,
            "strategy": STRATEGY_LABEL,
            "direction": group_name,
            "reason": reason,
            "mode": "LIVE",
            "ts": self._now().isoformat(timespec="seconds"),
        }
        if pnl is not None:
            event["pnl"] = round(pnl, 2)
        try:
            self._alert(event)
        except Exception:  # noqa: BLE001 - an alert must never stop the watcher
            log.exception("Position-group alert could not be queued.")
