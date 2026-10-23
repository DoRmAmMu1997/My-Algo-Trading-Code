"""
The position-groups file: which carried legs belong together, and each group's target.

The file is the OPERATOR's.  ``python algo.py group-legs`` adds newly found
broker legs to its ``unassigned`` list; the operator moves legs into groups and
sets each group's ``target`` and ``booked_pnl`` by editing it; the runner's
position-group monitor only ever READS it.  Nothing in the runner writes it,
so a hand edit can never be clobbered mid-session.

Shape (``Dependencies/position_groups.example.json`` is a filled-in copy)::

    {
      "schema_version": 1,
      "groups": [
        {"name": "Nov IC", "broker": "ZERODHA", "target": 15000, "booked_pnl": -2300,
         "legs": [{"symbol": "NIFTY25N2724500CE", "exchange": "NFO", "product": "NRML",
                   "quantity": -75, "entry_price": 120.5, "underlying": "NIFTY",
                   "expiry": "2025-11-27", "strike": 24500, "right": "CE"}]}
      ],
      "unassigned": [ {"broker": "ZERODHA", ...same leg fields...} ]
    }

Meaning of the numbers:

* ``quantity`` is signed units (not lots): negative = sold, positive = bought.
* ``entry_price`` is the operator's average price for that leg.  It is
  pre-filled from the broker but the operator owns it -- Zerodha reports 0 or a
  re-based price for some carried-forward rows.
* ``booked_pnl`` is the rupee P&L already realised on legs of this strategy
  that are now closed (adjustments).  The group's running P&L is
  ``booked_pnl + sum((ltp - entry_price) * quantity)`` over its open legs, and
  the whole group is squared off once that reaches ``target``.

Validation is all-or-nothing: one bad group makes the whole file invalid, and
the monitor keeps the last good copy it loaded rather than watching half of it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KNOWN_BROKERS = frozenset({"ZERODHA", "MSTOCK"})
OPTION_EXCHANGES = frozenset({"NFO", "BFO"})
PRODUCTS = frozenset({"NRML", "MIS"})


class GroupConfigError(ValueError):
    """The file is unusable; ``problems`` lists every reason, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class Leg:
    """One option position held at a broker."""

    broker: str
    symbol: str
    exchange: str
    product: str
    quantity: int
    entry_price: float
    underlying: str
    expiry: date
    strike: float
    right: str

    @property
    def key(self) -> tuple[str, str, str]:
        """The identity a leg may have only once across the whole file."""

        return (self.broker, self.exchange, self.symbol)

    def to_json(self, *, include_broker: bool) -> dict[str, Any]:
        document: dict[str, Any] = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "product": self.product,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "underlying": self.underlying,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "right": self.right,
        }
        if include_broker:
            document = {"broker": self.broker, **document}
        return document


@dataclass(frozen=True, slots=True)
class Group:
    """Legs that are squared off together once their combined P&L hits ``target``."""

    name: str
    broker: str
    target: float
    booked_pnl: float
    legs: tuple[Leg, ...]

    @property
    def fingerprint(self) -> str:
        """Name plus exact leg set, so a re-used name with new legs is a new group.

        Entry prices and the target are deliberately excluded: correcting a
        typo in either must not resurrect a group that was already squared off.
        """

        parts = [self.name, self.broker] + sorted(
            f"{leg.exchange}:{leg.symbol}:{leg.quantity}" for leg in self.legs
        )
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class GroupConfig:
    groups: tuple[Group, ...]
    unassigned: tuple[Leg, ...]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_leg(
    raw: Any,
    where: str,
    problems: list[str],
    *,
    broker: str | None,
    require_entry_price: bool,
) -> Leg | None:
    """Validate one leg; append every problem found and return None if any."""

    if not isinstance(raw, Mapping):
        problems.append(f"{where}: a leg must be a JSON object")
        return None
    start = len(problems)
    leg_broker = str(raw.get("broker") or broker or "").strip().upper()
    if broker is not None and raw.get("broker") and leg_broker != broker:
        problems.append(f"{where}: leg broker {leg_broker} differs from its group's {broker}")
    if leg_broker not in KNOWN_BROKERS:
        problems.append(f"{where}: broker must be one of {sorted(KNOWN_BROKERS)}")
    symbol = str(raw.get("symbol") or "").strip()
    if not symbol:
        problems.append(f"{where}: symbol is missing")
    exchange = str(raw.get("exchange") or "").strip().upper()
    if exchange not in OPTION_EXCHANGES:
        problems.append(f"{where}: exchange must be NFO or BFO")
    product = str(raw.get("product") or "").strip().upper()
    if product not in PRODUCTS:
        problems.append(f"{where}: product must be NRML or MIS")
    quantity = raw.get("quantity")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity == 0:
        problems.append(f"{where}: quantity must be a non-zero whole number of units")
    entry_price = _finite(raw.get("entry_price"))
    if entry_price is None or entry_price < 0 or (require_entry_price and entry_price == 0):
        problems.append(f"{where}: entry_price must be a positive number")
    underlying = str(raw.get("underlying") or "").strip().upper()
    if not underlying:
        problems.append(f"{where}: underlying is missing")
    try:
        expiry = date.fromisoformat(str(raw.get("expiry") or ""))
    except ValueError:
        expiry = None
        problems.append(f"{where}: expiry must be YYYY-MM-DD")
    strike = _finite(raw.get("strike"))
    if strike is None or strike <= 0:
        problems.append(f"{where}: strike must be a positive number")
    right = str(raw.get("right") or "").strip().upper()
    if right not in {"CE", "PE"}:
        problems.append(f"{where}: right must be CE or PE")
    if len(problems) > start:
        return None
    assert expiry is not None and strike is not None and entry_price is not None
    return Leg(
        broker=leg_broker,
        symbol=symbol,
        exchange=exchange,
        product=product,
        quantity=int(quantity),  # type: ignore[arg-type]
        entry_price=entry_price,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        right=right,
    )


def parse_config(document: Any) -> GroupConfig:
    """Validate a parsed JSON document.

    Raises:
        GroupConfigError: listing every problem found.
    """

    problems: list[str] = []
    if not isinstance(document, Mapping):
        raise GroupConfigError(["the file must hold one JSON object"])
    if document.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    raw_groups = document.get("groups", [])
    raw_unassigned = document.get("unassigned", [])
    if not isinstance(raw_groups, list):
        problems.append("groups must be a list")
        raw_groups = []
    if not isinstance(raw_unassigned, list):
        problems.append("unassigned must be a list")
        raw_unassigned = []

    groups: list[Group] = []
    names: set[str] = set()
    seen_legs: dict[tuple[str, str, str], str] = {}
    for index, raw in enumerate(raw_groups):
        where = f"groups[{index}]"
        if not isinstance(raw, Mapping):
            problems.append(f"{where}: a group must be a JSON object")
            continue
        name = str(raw.get("name") or "").strip()
        where = f"group {name!r}" if name else where
        if not name:
            problems.append(f"{where}: name is missing")
        elif name in names:
            problems.append(f"{where}: name is used twice")
        names.add(name)
        broker = str(raw.get("broker") or "").strip().upper()
        if broker not in KNOWN_BROKERS:
            problems.append(f"{where}: broker must be one of {sorted(KNOWN_BROKERS)}")
        target = _finite(raw.get("target"))
        if target is None or target <= 0:
            problems.append(f"{where}: target must be a positive rupee amount")
        booked = _finite(raw.get("booked_pnl", 0))
        if booked is None:
            problems.append(f"{where}: booked_pnl must be a number")
        raw_legs = raw.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            problems.append(f"{where}: legs must be a non-empty list")
            raw_legs = []
        legs: list[Leg] = []
        for leg_index, raw_leg in enumerate(raw_legs):
            leg = _parse_leg(
                raw_leg,
                f"{where} leg {leg_index}",
                problems,
                broker=broker,
                require_entry_price=True,
            )
            if leg is None:
                continue
            if leg.key in seen_legs:
                problems.append(
                    f"{where}: {leg.symbol} is already in {seen_legs[leg.key]}"
                )
            seen_legs[leg.key] = where
            legs.append(leg)
        if target is not None and booked is not None and broker in KNOWN_BROKERS:
            groups.append(Group(name, broker, target, booked, tuple(legs)))

    unassigned: list[Leg] = []
    for index, raw_leg in enumerate(raw_unassigned):
        leg = _parse_leg(
            raw_leg,
            f"unassigned[{index}]",
            problems,
            broker=None,
            require_entry_price=False,
        )
        if leg is None:
            continue
        if leg.key in seen_legs:
            problems.append(f"unassigned[{index}]: {leg.symbol} is already in {seen_legs[leg.key]}")
        seen_legs[leg.key] = f"unassigned[{index}]"
        unassigned.append(leg)

    if problems:
        raise GroupConfigError(problems)
    return GroupConfig(groups=tuple(groups), unassigned=tuple(unassigned))


def read_document(path: Path) -> dict[str, Any]:
    """Read the raw JSON object; a missing file is an empty, valid document."""

    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "groups": [], "unassigned": []}
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise GroupConfigError(["the file must hold one JSON object"])
    return document


def load_config(path: Path) -> GroupConfig:
    """Read and validate the file.  Raises ``GroupConfigError`` or ``OSError``."""

    try:
        return parse_config(read_document(path))
    except json.JSONDecodeError as exc:
        raise GroupConfigError([f"not valid JSON: {exc}"]) from exc


def write_document_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write via a sibling temp file and ``os.replace``, so a crash leaves old or new, never half."""

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What ``merge_broker_legs`` did, for the CLI to print."""

    added: tuple[Leg, ...]
    unidentified: tuple[str, ...]
    quantity_changed: tuple[str, ...]
    no_longer_held: tuple[str, ...]


def merge_broker_legs(
    document: dict[str, Any],
    broker: str,
    positions: Iterable[Any],
) -> MergeReport:
    """Append broker positions not yet anywhere in the file to ``unassigned``.

    ``positions`` are ``broker_contract.PositionDetail`` rows.  Nothing already
    in the file is modified: a quantity that differs from the broker's, or a
    leg the broker no longer holds, is only REPORTED, because the operator owns
    the file and may be mid-adjustment.  Rows without an option identity (the
    broker's instrument master did not list them) are reported, not added.

    Raises:
        GroupConfigError: the existing document is invalid; it is not touched.
    """

    broker = broker.strip().upper()
    config = parse_config(document)
    existing = {leg.key: leg for group in config.groups for leg in group.legs}
    existing.update({leg.key: leg for leg in config.unassigned})
    held: set[tuple[str, str, str]] = set()
    added: list[Leg] = []
    unidentified: list[str] = []
    changed: list[str] = []
    for position in positions:
        key = (broker, position.exchange, position.symbol)
        held.add(key)
        product = position.product_type or "NRML"
        if (
            not position.right
            or position.expiry is None
            or not position.strike
            or position.exchange not in OPTION_EXCHANGES
            or product not in PRODUCTS
        ):
            unidentified.append(position.symbol)
            continue
        if key in existing:
            if existing[key].quantity != position.quantity:
                changed.append(
                    f"{position.symbol}: file {existing[key].quantity}, broker {position.quantity}"
                )
            continue
        leg = Leg(
            broker=broker,
            symbol=position.symbol,
            exchange=position.exchange,
            product=product,
            quantity=position.quantity,
            entry_price=round(float(position.average_price), 2),
            underlying=position.underlying,
            expiry=position.expiry,
            strike=float(position.strike),
            right=position.right,
        )
        added.append(leg)
    document.setdefault("schema_version", SCHEMA_VERSION)
    document.setdefault("groups", [])
    document.setdefault("unassigned", [])
    document["unassigned"].extend(leg.to_json(include_broker=True) for leg in added)
    gone = sorted(key[2] for key in existing if key[0] == broker and key not in held)
    return MergeReport(tuple(added), tuple(unidentified), tuple(changed), tuple(gone))
