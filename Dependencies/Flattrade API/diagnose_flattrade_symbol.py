"""Read-only Flattrade NIFTY option diagnostic with an opt-in REAL test order.

Examples from the repository root::

    python "Dependencies/Flattrade API/diagnose_flattrade_symbol.py" CE 24150
    python "Dependencies/Flattrade API/diagnose_flattrade_symbol.py" PE 24000 14JUL26

The script logs in, downloads the same official index-derivative scrip master
used by live execution, prints the exact contract row, and checks the shared
resolver.  It does not place an order unless ``--place-order`` is supplied and
    the operator supplies an explicit ``--qty`` and then types exactly ``YES``.
    The real-money path attempts the matching SELL only after the BUY has a
    confirmed full fill; a partial or unknown entry stops for reconciliation.

The normal read-only flow has four visible checkpoints: authenticate, download
the official contract list, select one exact row, and ask the production resolver
for the same symbol. A mismatch is reported before the optional order prompt, so
the diagnostic tests the same boundary that live strategies depend upon.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flattrade_execution as fe
from broker_contract import OrderResult, OrderStatus
from diagnostic_preflight import validate_quantity_for_lot

UNDERLYING = "NIFTY"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used directly and through ``algo.py diagnose``.

    ``algo.py`` consumes only ``--broker flattrade`` and forwards every remaining
    argument unchanged, so this parser remains the single source of truth for the
    broker-specific option type, strike, expiry, quantity, and real-order switch.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Flattrade NIFTY option symbol. Read-only unless "
            "--place-order is supplied and YES is typed."
        )
    )
    parser.add_argument(
        "option_type",
        nargs="?",
        default="CE",
        choices=("CE", "PE"),
        type=str.upper,
        help="Option type (default: CE).",
    )
    parser.add_argument(
        "strike",
        nargs="?",
        default=23950,
        type=int,
        help="NIFTY strike (default: 23950).",
    )
    parser.add_argument(
        "expiry",
        nargs="?",
        help="Optional expiry as DDMMMYY, for example 14JUL26.",
    )
    parser.add_argument(
        "--place-order",
        action="store_true",
        help="After another YES prompt, place a REAL BUY then SELL round trip.",
    )
    parser.add_argument(
        "--qty",
        type=int,
        help="Required explicit unit quantity for the REAL test order.",
    )
    return parser


def _parse_expiry(raw: str | None) -> dt.date | None:
    """Parse the optional DDMMMYY CLI expiry with a friendly error."""
    if not raw:
        return None
    try:
        return dt.datetime.strptime(raw.upper().strip(), "%d%b%y").date()
    except ValueError as exc:
        raise ValueError(
            f"Could not parse expiry {raw!r}; expected DDMMMYY like 14JUL26."
        ) from exc


def select_contract(
    client: Any,
    *,
    underlying: str,
    option_type: str,
    strike: int,
    requested_expiry: dt.date | None,
    today: dt.date | None = None,
) -> tuple[dt.date, str, int]:
    """Select one exact CSV row and return expiry, trading symbol, and lot size.

    When no expiry is supplied, the nearest non-expired contract containing the
    requested strike is chosen.  Filtering the strike before choosing the date
    avoids selecting an expiry that does not list that particular contract.

    Args:
        client: Logged-in FlattradeExecutionClient with ``_scrip_df`` preloaded.
        underlying: Index name (the CLI currently fixes this to NIFTY).
        option_type: ``CE`` or ``PE``.
        strike: Whole-number option strike.
        requested_expiry: Exact date, or ``None`` to choose the nearest future row.
        today: Injectable clock used by deterministic unit tests.

    Returns:
        ``(expiry_date, trading_symbol, lot_size)`` from one official CSV row.

    Raises:
        ValueError: The master is unavailable, no exact row exists, or lot size is
            invalid. It never guesses a symbol or quantity.
    """
    frame = getattr(client, "_scrip_df", None)
    if frame is None or frame.empty:
        raise ValueError("Flattrade NFO scrip master is not loaded.")
    underlying_u = str(underlying).upper().strip()
    option_u = str(option_type).upper().strip()
    strike_i = round(float(strike))
    matches = frame[
        (frame["_symbol_u"] == underlying_u)
        & (frame["_option_u"] == option_u)
        & (frame["_strike_f"].round().astype(int) == strike_i)
    ].copy()
    if requested_expiry is not None:
        matches = matches[matches["_expiry_date"] == requested_expiry]
    else:
        today = today or dt.date.today()
        matches = matches[matches["_expiry_date"] >= today]
        if not matches.empty:
            nearest = min(matches["_expiry_date"])
            matches = matches[matches["_expiry_date"] == nearest]
    if matches.empty:
        expiry_text = requested_expiry.isoformat() if requested_expiry else "nearest future"
        raise ValueError(
            f"No Flattrade {underlying_u} {option_u} strike {strike_i} "
            f"contract found for {expiry_text} expiry."
        )
    row = matches.sort_values(["_expiry_date", "_trading_symbol"]).iloc[0]
    lot_size = int(row["_lotsize_i"])
    if lot_size <= 0:
        raise ValueError("Resolved Flattrade contract has no valid Lotsize.")
    return (
        row["_expiry_date"],
        str(row["_trading_symbol"]).strip(),
        lot_size,
    )


def place_round_trip_test_order(client: Any, symbol: str, quantity: int) -> bool:
    """Place a confirmation-gated REAL BUY then SELL; return True only if flat.

    This is deliberately separate from ``main`` so tests can prove that anything
    except uppercase ``YES`` sends zero orders. If entry or exit confirmation is
    ambiguous, the message assumes a live position *may* exist and directs the
    operator to Flattrade; silence would be unsafe in a real-money diagnostic.
    """
    print("\n" + "=" * 72)
    print("REAL ROUND-TRIP TEST ORDER on Flattrade")
    print(
        f"  BUY {quantity} {symbol} (product=INTRADAY); after a confirmed "
        f"full fill, attempt SELL {quantity} to flatten."
    )
    print("  THIS PLACES REAL ORDERS WITH REAL MONEY.")
    try:
        confirmation = input(
            "  Type exactly YES to proceed (anything else aborts): "
        ).strip()
    except (EOFError, OSError):
        confirmation = ""
    if confirmation != "YES":
        print("  Aborted (no confirmation).")
        return False

    print(f"  Placing BUY {quantity} {symbol} ...")
    try:
        entry = client.place_market_order(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            product_type="INTRADAY",
        )
    except Exception as exc:
        print(f"  ENTRY EXPOSURE IS INDETERMINATE: client raised {exc}")
        print(
            f"  !! A LIVE long position in {symbol} (qty={quantity}) MAY BE OPEN. "
            "CHECK FLATTRADE BEFORE RETRYING OR PLACING ANOTHER ORDER."
        )
        return False
    if _is_zero_fill_rejection(entry, quantity):
        print(f"  ENTRY rejected with zero fill: {entry.reason}\n  No position opened.")
        return False
    if not _is_full_fill(entry, quantity):
        _print_indeterminate("ENTRY", entry, symbol, quantity)
        return False
    print(f"  ENTRY filled. OrderId={entry.order_id}")

    print(f"  Squaring off: SELL {quantity} {symbol} ...")
    try:
        exit_order = client.place_market_order(
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            product_type="INTRADAY",
        )
    except Exception as exc:
        print(f"  !! EXIT EXPOSURE IS INDETERMINATE: client raised {exc}")
        print(
            f"  !! A LIVE long position in {symbol} (qty={quantity}) MAY BE OPEN. "
            "CHECK FLATTRADE AND SQUARE IT OFF MANUALLY NOW."
        )
        return False
    if _is_zero_fill_rejection(exit_order, quantity):
        print(f"  !! EXIT rejected with zero fill: {exit_order.reason}")
        print(
            f"  !! The LIVE long {symbol} qty={quantity} remains OPEN. "
            "Square it off manually now."
        )
        return False
    if not _is_full_fill(exit_order, quantity):
        _print_indeterminate("EXIT", exit_order, symbol, quantity)
        return False
    print(
        f"  EXIT filled. OrderId={exit_order.order_id} "
        "-> flat. Round-trip OK."
    )
    return True


def _is_full_fill(result: Any, quantity: int) -> bool:
    """Return True only for an exact, typed full-quantity fill."""
    return (
        isinstance(result, OrderResult)
        and result.status is OrderStatus.FILLED
        and result.requested_quantity == quantity
        and result.filled_quantity == quantity
        and result.remaining_quantity == 0
    )


def _is_zero_fill_rejection(result: Any, quantity: int) -> bool:
    """Return True only when the broker explicitly proves no units traded."""
    return (
        isinstance(result, OrderResult)
        and result.status is OrderStatus.REJECTED
        and result.requested_quantity == quantity
        and result.filled_quantity == 0
        and result.remaining_quantity == quantity
    )


def _print_indeterminate(
    stage: str,
    result: Any,
    symbol: str,
    quantity: int,
) -> None:
    """Print typed broker evidence without claiming that exposure is flat."""
    if isinstance(result, OrderResult):
        evidence = (
            f"status={result.status.value} order_id={result.order_id or 'UNKNOWN'} "
            f"filled={result.filled_quantity}/{result.requested_quantity} "
            f"remaining={result.remaining_quantity} reason={result.reason}"
        )
    else:
        evidence = f"untyped_result={result!r}"
    print(f"  !! {stage} EXPOSURE IS INDETERMINATE: {evidence}")
    print(
        f"  !! Check {symbol} qty={quantity} in Flattrade before retrying or "
        "placing another order."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the safe symbol diagnostic and optional confirmation-gated order.

    Return codes follow normal CLI convention: 0 success, 1 operational failure
    (login/resolution/order), and 2 invalid command-line input.
    """
    # Parse and validate all local input before login. Typos should never open a
    # browser or touch the broker API.
    args = build_parser().parse_args(argv)
    if args.qty is not None and args.qty <= 0:
        print("--qty must be a positive integer.", file=sys.stderr)
        return 2
    if args.place_order and args.qty is None:
        print(
            "--place-order requires an explicit positive --qty; the diagnostic "
            "will not guess a live quantity from a changing exchange lot size.",
            file=sys.stderr,
        )
        return 2
    try:
        requested_expiry = _parse_expiry(args.expiry)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    client = fe.flattrade_execution_client
    try:
        # ``preload_scrip_master`` authenticates first, then downloads once. The
        # production resolver below reuses the already-normalized in-memory table.
        print("Logging in to Flattrade and loading the official NFO index master...")
        if not client.preload_scrip_master():
            print("Could not log in / load the Flattrade scrip master.")
            return 1

        expiry, csv_symbol, lot_size = select_contract(
            client,
            underlying=UNDERLYING,
            option_type=args.option_type,
            strike=args.strike,
            requested_expiry=requested_expiry,
        )
        resolved = client.resolve_option_symbol(
            UNDERLYING,
            expiry,
            args.option_type,
            args.strike,
        )
        master_rows = 0 if client._scrip_df is None else len(client._scrip_df)
        print(f"\nContract master rows: {master_rows}")
        print(f"Selected expiry : {expiry:%d%b%y}")
        print(f"Trading symbol  : {csv_symbol}")
        print(f"Official lot    : {lot_size}")
        print(
            "Resolver result : "
            f"resolve_option_symbol({UNDERLYING}, {expiry}, "
            f"{args.option_type}, {args.strike}) -> {resolved!r}"
        )
        if not resolved or resolved != csv_symbol:
            print("Resolver did not reproduce the exact official CSV symbol.")
            return 1

        # Read-only is the default. Reaching the order helper requires both the
        # explicit flag here and its second, typed-YES confirmation inside.
        if not args.place_order:
            print("\nRead-only diagnostic complete. No order was placed.")
            return 0

        # Pre-flight before the typed-YES prompt, so an order the exchange can
        # never accept does not reach the broker or burn a confirmation.
        lot_error = validate_quantity_for_lot(args.qty, lot_size)
        if lot_error:
            print(f"\n{lot_error}", file=sys.stderr)
            return 2

        return 0 if place_round_trip_test_order(client, resolved, args.qty) else 1
    except Exception as exc:
        print(f"Flattrade diagnostic failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # Also runs on parse/resolution/order exceptions. Flattrade has no remote
        # logout endpoint, so this clears the local token/session/cache state.
        client.logout()


if __name__ == "__main__":
    raise SystemExit(main())
