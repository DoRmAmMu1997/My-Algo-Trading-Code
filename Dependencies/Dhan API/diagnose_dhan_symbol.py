"""Read-only Dhan index-option diagnostic with an opt-in REAL test order.

Examples from the repository root::

    python "Dependencies/Dhan API/diagnose_dhan_symbol.py" CE 24150
    python "Dependencies/Dhan API/diagnose_dhan_symbol.py" PE 24000 14JUL26
    python "Dependencies/Dhan API/diagnose_dhan_symbol.py" --underlying BANKNIFTY CE 59500

The script logs in, loads the same local instrument master used by live
execution, prints the exact contract row, and checks the shared resolver.  It
does not place an order unless ``--place-order`` is supplied and the operator
supplies an explicit ``--qty`` and then types exactly ``YES``.  The real-money
path attempts the matching SELL only after the BUY has a confirmed full fill;
a partial or unknown entry stops for reconciliation.

The normal read-only flow has five visible checkpoints: authenticate, load the
contract list, select one exact row, ask the production resolver for the same
symbol, and confirm that symbol maps back to the same numeric ``securityId``.
That last step is Dhan-specific and matters: Dhan's order API is driven by the
security id, not the symbol, so a broken mapping would silently trade the wrong
contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dhan_execution as de
from broker_contract import OrderResult, OrderStatus


def build_parser() -> argparse.ArgumentParser:
    """Build the parser used directly and through ``algo.py diagnose``.

    ``algo.py`` consumes only ``--broker dhan`` and forwards every remaining
    argument unchanged, so this parser remains the single source of truth for the
    broker-specific option type, strike, expiry, quantity, and real-order switch.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Dhan index-option symbol. Read-only unless "
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
        help="Option strike (default: 23950).",
    )
    parser.add_argument(
        "expiry",
        nargs="?",
        help="Optional expiry as DDMMMYY, for example 14JUL26.",
    )
    parser.add_argument(
        "--underlying",
        default="NIFTY",
        type=str.upper,
        help="Index name (default: NIFTY). BANKNIFTY diagnoses the mirror leg.",
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
) -> tuple[dt.date, str, str, int]:
    """Select one exact catalogue row; return expiry, symbol, security id, lot.

    When no expiry is supplied, the nearest non-expired contract containing the
    requested strike is chosen.  Filtering the strike before choosing the date
    avoids selecting an expiry that does not list that particular contract.

    Args:
        client: Logged-in DhanExecutionClient with ``_scrip_df`` preloaded.
        underlying: Index name such as NIFTY or BANKNIFTY.
        option_type: ``CE`` or ``PE``.
        strike: Whole-number option strike.
        requested_expiry: Exact date, or ``None`` to choose the nearest future row.
        today: Injectable clock used by deterministic unit tests.

    Returns:
        ``(expiry_date, trading_symbol, security_id, lot_size)`` from one row.

    Raises:
        ValueError: The master is unavailable or no exact row exists. It never
            guesses a symbol, a security id, or a quantity.
    """
    frame = getattr(client, "_scrip_df", None)
    if frame is None or frame.empty:
        raise ValueError("Dhan scrip master is not loaded.")
    underlying_u = str(underlying).upper().strip()
    option_u = str(option_type).upper().strip()
    strike_i = round(float(strike))
    matches = frame[
        (frame["_underlying_u"] == underlying_u)
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
            f"No Dhan {underlying_u} {option_u} strike {strike_i} "
            f"contract found for {expiry_text} expiry."
        )
    row = matches.sort_values(["_expiry_date", "trading_symbol"]).iloc[0]
    # LOT_SIZE is optional in the catalogue, so report 0 rather than crashing;
    # --place-order requires an explicit --qty regardless.
    try:
        lot_size = int(row["lot_size"])
    except (TypeError, ValueError):
        lot_size = 0
    return (
        row["_expiry_date"],
        str(row["trading_symbol"]).strip(),
        str(int(row["security_id"])),
        lot_size,
    )


def validate_quantity_for_lot(quantity: int, lot_size: int) -> str:
    """Return an error message when ``quantity`` is not a whole-lot multiple.

    Index options trade only in exchange-defined lots, so a quantity such as 1
    against a lot of 65 can never be accepted.  Catching that locally matters
    because Dhan reports it as a generic ``DH-905 Input_Exception`` whose
    ``error_message`` has been observed to read "Invalid IP" -- an error that
    sends operators hunting a network fault that does not exist.

    This validates the operator's OWN number; it deliberately does not derive
    one.  ``--place-order`` still requires an explicit ``--qty`` because lot
    sizes change and guessing one would risk a wrong-sized live order.

    Args:
        quantity: The explicit unit quantity supplied on the command line.
        lot_size: Official lot size from the contract catalogue, 0 if absent.

    Returns:
        An empty string when the quantity is placeable, else the reason.
    """
    if lot_size <= 0:
        # The catalogue row carried no lot size, so there is nothing to check
        # against. The broker remains the authority.
        return ""
    if quantity % lot_size:
        return (
            f"--qty {quantity} is not a multiple of the official lot size "
            f"{lot_size}. Index options trade in whole lots only, so Dhan "
            f"would reject this order. Use {lot_size} for one lot "
            f"(or {lot_size * 2} for two)."
        )
    return ""


def check_entry_margin(
    client: Any,
    *,
    security_id: str,
    quantity: int,
    product_type: str = "INTRADAY",
) -> tuple[bool, str]:
    """Ask Dhan whether the BUY entry is affordable, without placing anything.

    ``margin_calculator`` validates the same contract and quantity fields as
    ``place_order`` but creates no order, so it is a safe pre-flight.

    Only the ENTRY is checked.  An exit must never be gated on a pre-flight
    query: if a position is open, squaring it off has to stay possible even
    when a margin lookup fails.

    Returns:
        ``(placeable, message)``.  ``placeable`` is ``False`` only when the
        broker positively reports a shortfall.  An unavailable or malformed
        margin reply returns ``True`` with a warning -- an advisory check that
        cannot run must not block the operator, since the broker itself is
        still the real gate.
    """
    try:
        envelope = client._call_api(
            "margin calculator",
            lambda sdk: sdk.margin_calculator(
                security_id=str(security_id),
                exchange_segment="NSE_FNO",
                transaction_type="BUY",
                quantity=int(quantity),
                product_type=product_type,
                price=0,
            ),
        )
    except Exception as exc:
        return (True, f"Margin pre-check could not run ({exc}); continuing.")

    outcome, data, reason = de._classify_envelope(envelope)
    if outcome != "ok" or not isinstance(data, dict):
        return (True, f"Margin pre-check was inconclusive ({outcome}: {reason}); continuing.")

    # Affordability is decided by comparing the two balances DIRECTLY.
    #
    # Dhan's `insufficientBalance` field is deliberately ignored: despite the
    # name it is the UNSIGNED difference, abs(totalMargin - availableBalance),
    # so it is positive whether you are short OR have money to spare. Measured
    # against this account (available 10000.39):
    #
    #   required   2808.00 -> insufficientBalance  7192.39   (surplus!)
    #   required  11232.00 -> insufficientBalance  1231.61   (short)
    #   required  21895.25 -> insufficientBalance 11894.86   (short)
    #   required  87581.00 -> insufficientBalance 77580.61   (short)
    #
    # Reading it as a shortfall rejects every affordable order. The response
    # carries no boolean "sufficient" flag, so the comparison below is the
    # only reliable test.
    required = data.get("totalMargin")
    available = data.get("availableBalance")
    if required is None or available is None:
        return (True, "Margin pre-check omitted the margin/balance figures; continuing.")
    try:
        required_f = float(required)
        available_f = float(available)
    except (TypeError, ValueError):
        return (True, "Margin pre-check returned unusable figures; continuing.")

    summary = f"required {required_f:.2f}, available {available_f:.2f}"
    if available_f < required_f:
        return (
            False,
            f"Insufficient funds for this entry: {summary}, "
            f"short by {required_f - available_f:.2f}. Fund the account or test "
            "with a cheaper (further out-of-the-money) strike.",
        )
    return (True, f"Margin pre-check OK: {summary}.")


def place_round_trip_test_order(client: Any, symbol: str, quantity: int) -> bool:
    """Place a confirmation-gated REAL BUY then SELL; return True only if flat.

    This is deliberately separate from ``main`` so tests can prove that anything
    except uppercase ``YES`` sends zero orders. If entry or exit confirmation is
    ambiguous, the message assumes a live position *may* exist and directs the
    operator to Dhan; silence would be unsafe in a real-money diagnostic.
    """
    print("\n" + "=" * 72)
    print("REAL ROUND-TRIP TEST ORDER on Dhan")
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
            "CHECK DHAN BEFORE RETRYING OR PLACING ANOTHER ORDER."
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
            "CHECK DHAN AND SQUARE IT OFF MANUALLY NOW."
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
        f"  !! Check {symbol} qty={quantity} in Dhan before retrying or "
        "placing another order."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the safe symbol diagnostic and optional confirmation-gated order.

    Return codes follow normal CLI convention: 0 success, 1 operational failure
    (login/resolution/order), and 2 invalid command-line input.
    """
    # Parse and validate all local input before login. Typos should never touch
    # the broker API.
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
    client = de.dhan_execution_client
    try:
        # ``preload_scrip_master`` authenticates first, then loads the local
        # instrument CSV once. The production resolver below reuses the same
        # already-normalized in-memory table.
        print("Logging in to Dhan and loading the local instrument master...")
        if not client.preload_scrip_master():
            print("Could not log in / load the Dhan scrip master.")
            return 1

        expiry, csv_symbol, csv_security_id, lot_size = select_contract(
            client,
            underlying=args.underlying,
            option_type=args.option_type,
            strike=args.strike,
            requested_expiry=requested_expiry,
        )
        resolved = client.resolve_option_symbol(
            args.underlying,
            expiry,
            args.option_type,
            args.strike,
        )
        master_rows = 0 if client._scrip_df is None else len(client._scrip_df)
        print(f"\nContract master rows: {master_rows}")
        print(f"Selected expiry : {expiry:%d%b%y}")
        print(f"Trading symbol  : {csv_symbol}")
        print(f"Security id     : {csv_security_id}")
        print(f"Official lot    : {lot_size}")
        print(
            "Resolver result : "
            f"resolve_option_symbol({args.underlying}, {expiry}, "
            f"{args.option_type}, {args.strike}) -> {resolved!r}"
        )
        if not resolved or resolved != csv_symbol:
            print("Resolver did not reproduce the exact official catalogue symbol.")
            return 1

        # Dhan-specific: the order API takes the security id, so prove the
        # symbol the resolver returned maps back to the very same contract.
        mapped_security_id = client._security_id_for(resolved)
        print(f"Order-API mapping: {resolved} -> securityId {mapped_security_id!r}")
        if mapped_security_id != csv_security_id:
            print(
                "Symbol does not map back to the catalogue securityId; a live "
                "order could reach the WRONG contract. Refusing to continue."
            )
            return 1

        # Read-only is the default. Reaching the order helper requires both the
        # explicit flag here and its second, typed-YES confirmation inside.
        if not args.place_order:
            print("\nRead-only diagnostic complete. No order was placed.")
            return 0

        # Pre-flight. Both checks run BEFORE the typed-YES prompt so an order
        # that is certain to fail never reaches the broker (and never wastes
        # the operator's confirmation on it).
        lot_error = validate_quantity_for_lot(args.qty, lot_size)
        if lot_error:
            print(f"\n{lot_error}", file=sys.stderr)
            return 2

        placeable, margin_message = check_entry_margin(
            client,
            security_id=csv_security_id,
            quantity=args.qty,
        )
        print(f"\n{margin_message}")
        if not placeable:
            return 1

        return 0 if place_round_trip_test_order(client, resolved, args.qty) else 1
    except Exception as exc:
        print(f"Dhan diagnostic failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # Also runs on parse/resolution/order exceptions. Dhan documents no
        # remote logout, so this only clears local session state.
        client.logout()


if __name__ == "__main__":
    raise SystemExit(main())
