"""
Diagnostic for Shoonya option symbol resolution (and an optional REAL test order).

Run during a LIVE Shoonya session (TOTP auto-generated from SHOONYA_TOTP_SECRET,
or it prompts):

    python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" [OPT] [STRIKE] [EXPIRY]
    # EXPIRY uses the broker's DDMMMYY form.

It logs in, downloads the NSE F&O symbol master ONCE (the same cache the live
runner uses), and shows whether our resolver builds a valid tsym for the given
option type / strike / expiry. Read-only by default.

Optional REAL round-trip test order (BUY then auto square-off) - REAL MONEY:
    python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" CE 23950 DDMMMYY --place-order --qty <current-lot-size>
It places the explicitly requested quantity on the resolved contract. Only a
confirmed full BUY fill permits the automatic SELL-to-flatten. Requires typing
YES to confirm. (Needs an EXPIRY.)
"""
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shoonya_execution as se  # handles sys.path to the SDK + loads .env
from broker_contract import OrderResult, OrderStatus
from diagnostic_preflight import validate_quantity_for_lot


def _arg_val(name, default):
    """Read `--name VALUE` or `--name=VALUE` from argv; else `default`."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# Positional args (OPT, STRIKE, EXPIRY) ignore any --flags so they can be combined.
_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
UNDERLYING = "NIFTY"
OPT = _POS[0].upper() if len(_POS) > 0 else "CE"
STRIKE = int(_POS[1]) if len(_POS) > 1 else 23950
# Expiry as DDMMMYY. Optional; without it we just sample the
# live symbol format rather than resolving a specific contract.
EXPIRY = _POS[2].upper() if len(_POS) > 2 else None
PLACE_ORDER = "--place-order" in sys.argv
_QTY_RAW = _arg_val("--qty", None)
try:
    QTY = int(_QTY_RAW) if _QTY_RAW is not None else None
except (TypeError, ValueError):
    QTY = None


def _is_full_fill(result, quantity):
    """Return True only for an exact, typed full-quantity fill."""
    return (
        isinstance(result, OrderResult)
        and result.status is OrderStatus.FILLED
        and result.requested_quantity == quantity
        and result.filled_quantity == quantity
        and result.remaining_quantity == 0
    )


def _is_zero_fill_rejection(result, quantity):
    """Return True only when the broker explicitly proves no units traded."""
    return (
        isinstance(result, OrderResult)
        and result.status is OrderStatus.REJECTED
        and result.requested_quantity == quantity
        and result.filled_quantity == 0
        and result.remaining_quantity == quantity
    )


def _print_indeterminate(stage, result, symbol, quantity, detail=""):
    """Print enough evidence for an operator to reconcile an ambiguous order."""
    if isinstance(result, OrderResult):
        evidence = (
            f"status={result.status.value} order_id={result.order_id or 'UNKNOWN'} "
            f"filled={result.filled_quantity}/{result.requested_quantity} "
            f"remaining={result.remaining_quantity} reason={result.reason}"
        )
    else:
        evidence = f"untyped_result={result!r}"
    if detail:
        evidence = f"{detail}; {evidence}"
    print(f"  !! {stage} EXPOSURE IS INDETERMINATE: {evidence}")
    print(
        f"  !! Check {symbol} qty={quantity} in Shoonya before retrying or "
        "placing another order."
    )


def place_round_trip_test_order(client, symbol, qty, broker="Shoonya"):
    """Place a REAL market BUY, then attempt SELL only after a confirmed fill.
    Guarded by a typed YES confirmation. REAL MONEY."""
    print("\n" + "=" * 72)
    print(f"REAL ROUND-TRIP TEST ORDER on {broker}")
    print(
        f"  BUY {qty} {symbol} (product=INTRADAY); after a confirmed full fill, "
        f"attempt SELL {qty} to flatten."
    )
    print("  THIS PLACES REAL ORDERS WITH REAL MONEY.")
    try:
        confirm = input("  Type exactly YES to proceed (anything else aborts): ").strip()
    except (EOFError, OSError):
        confirm = ""
    if confirm != "YES":
        print("  Aborted (no confirmation).")
        return False
    print(f"  Placing BUY {qty} {symbol} ...")
    try:
        entry = client.place_market_order(symbol=symbol, side="BUY", quantity=qty, product_type="INTRADAY")
    except Exception as exc:
        _print_indeterminate("ENTRY", None, symbol, qty, detail=f"client raised {exc}")
        return False
    if _is_zero_fill_rejection(entry, qty):
        print(f"  ENTRY rejected with zero fill: {entry.reason}\n  No position opened.")
        return False
    if not _is_full_fill(entry, qty):
        _print_indeterminate("ENTRY", entry, symbol, qty)
        return False
    print(f"  ENTRY filled. OrderId={entry.order_id}")
    print(f"  Squaring off: SELL {qty} {symbol} ...")
    try:
        ex = client.place_market_order(symbol=symbol, side="SELL", quantity=qty, product_type="INTRADAY")
    except Exception as exc:
        _print_indeterminate("EXIT", None, symbol, qty, detail=f"client raised {exc}")
        return False
    if _is_zero_fill_rejection(ex, qty):
        print(f"  !! EXIT rejected with zero fill: {ex.reason}")
        print(f"  !! The LIVE long {symbol} qty={qty} remains OPEN. Square it off manually now.")
        return False
    if not _is_full_fill(ex, qty):
        _print_indeterminate("EXIT", ex, symbol, qty)
        return False
    print(f"  EXIT filled. OrderId={ex.order_id} -> flat. Round-trip OK.")
    return True


def main():
    # Without this the root logger sits at WARNING, so shoonya_execution's INFO
    # diagnostics -- symbol-master load counts and lot-size availability -- are
    # silently dropped, which is exactly what someone runs this script to see.
    # The Kotak, Flattrade and Dhan diagnostics configure logging the same way.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    client = se.shoonya_execution_client
    print("Logging in to Shoonya...")
    if not client.preload_scrip_master():
        print("Could not log in / load the symbol master. Check the messages above.")
        return

    symbols = client._symbol_set or set()
    print(f"\nNFO symbol master: {len(symbols)} trading symbols.")
    nifty_syms = sorted(s for s in symbols if s.startswith(UNDERLYING))
    print(f"{UNDERLYING}* symbols: {len(nifty_syms)}")

    if EXPIRY is None:
        print(f"\nNo expiry given (3rd arg, format DDMMMYY). Showing a few sample "
              f"{UNDERLYING} {OPT[0]} symbols so you can read the live format:")
        for s in sorted(s for s in nifty_syms if OPT[0] in s)[:10]:
            print("   ", s)
        if PLACE_ORDER:
            print("\n--place-order needs an EXPIRY (3rd arg) to resolve the contract; no order placed.")
        return

    try:
        expiry = dt.datetime.strptime(EXPIRY, "%d%b%y").date()
    except ValueError:
        print(f"Could not parse expiry {EXPIRY!r}; expected DDMMMYY.")
        return

    sym = client.resolve_option_symbol(UNDERLYING, expiry, OPT, STRIKE)
    print(f"\nresolve_option_symbol({UNDERLYING}, {expiry}, {OPT}, {STRIKE}) -> {sym!r}")
    if not sym:
        near_prefix = f"{UNDERLYING}{EXPIRY}{OPT[0]}"
        near = sorted(s for s in symbols if s.startswith(near_prefix))
        print(f"  sample {near_prefix}* symbols: {near[:5]} ... {near[-5:] if near else ''}")

    # Optional REAL round-trip test order (only when --place-order is passed).
    if PLACE_ORDER:
        if QTY is None or QTY <= 0:
            print(
                "\n--place-order requires an explicit positive --qty using the "
                "current contract lot size; no order placed."
            )
        elif not sym:
            print("\n--place-order requested but the symbol did not resolve; no order placed.")
        else:
            # Pre-flight before the typed-YES prompt, so an order the exchange
            # can never accept never reaches the broker.
            lot_size = client.lot_size_for_symbol(sym)
            print(f"official lot    : {lot_size or 'unknown (not in this master)'}")
            lot_error = validate_quantity_for_lot(QTY, lot_size)
            if lot_error:
                print(f"\n{lot_error}\nNo order placed.")
            else:
                place_round_trip_test_order(client, sym, QTY, broker="Shoonya")


if __name__ == "__main__":
    main()
