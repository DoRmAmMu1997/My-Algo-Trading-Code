"""
Diagnostic for Kotak option symbol resolution (and an optional REAL test order).

Run during a LIVE Kotak session (it prompts for a TOTP at login):

    python "Dependencies/Kotak API/diagnose_kotak_symbol.py" [OPT] [STRIKE]
    # e.g. python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950

It downloads the NSE F&O scrip master ONCE (the same cache the live runner uses),
then shows how Kotak stores NIFTY option rows near your strike, the available
expiries, and whether our resolver finds the exact pTrdSymbol. Read-only by default.

Optional REAL round-trip test order (BUY then auto square-off) - REAL MONEY:
    python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950 --place-order --qty <current-lot-size>
It places the explicitly requested quantity on the resolved nearest-expiry
contract. Only a confirmed full BUY fill permits the automatic SELL-to-flatten.
Requires typing YES to confirm.
"""
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kotak_execution as ke  # handles sys.path to the SDK + loads .env
from broker_contract import OrderResult, OrderStatus


def _arg_val(name, default):
    """Read `--name VALUE` or `--name=VALUE` from argv; else `default`."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


# Positional args (OPT, STRIKE) ignore any --flags so the two can be combined.
_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
UNDERLYING = "NIFTY"
OPT = _POS[0].upper() if len(_POS) > 0 else "CE"
STRIKE = int(_POS[1]) if len(_POS) > 1 else 23950
PLACE_ORDER = "--place-order" in sys.argv
_QTY_RAW = _arg_val("--qty", None)
try:
    QTY = int(_QTY_RAW) if _QTY_RAW is not None else None
except (TypeError, ValueError):
    QTY = None
FIELDS = ["pSymbolName", "pOptionType", "_expiry_str", "dStrikePrice;", "pTrdSymbol"]


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
        f"  !! Check {symbol} qty={quantity} in Kotak before retrying or "
        "placing another order."
    )


def place_round_trip_test_order(client, symbol, qty, broker="Kotak"):
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
    # Without this the root logger sits at WARNING, so kotak_execution's INFO
    # diagnostics -- the scrip-master URL, download progress and byte counts --
    # are silently dropped. They are the whole point of running this script when
    # a download stalls, so surface them. The Flattrade and Dhan diagnostics
    # already configure logging the same way.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    client = ke.kotak_execution_client
    print("Logging in (you'll be prompted for a TOTP)...")
    if not client.preload_scrip_master():
        print("Could not log in / load the scrip master. Check the messages above.")
        return

    df = client._scrip_df
    if df is None:
        print("Scrip master is unexpectedly empty after preload.")
        return
    print(f"\nScrip master rows: {len(df)}")
    base = df[(df["_sym_u"] == UNDERLYING) & (df["_opt_u"] == OPT)]
    print(f"{UNDERLYING} {OPT} rows: {len(base)}")
    if base.empty:
        print("No NIFTY rows for that option type - check pSymbolName values:",
              sorted(df["_sym_u"].dropna().unique().tolist())[:20])
        return

    # Sort by actual date (the _expiry_str "DDMMMYYYY" strings sort by day-of-month
    # if compared as text, which looks scrambled).
    expiries = sorted(base["_expiry_str"].dropna().unique().tolist(),
                      key=lambda s: dt.datetime.strptime(s, "%d%b%Y"))
    print(f"available expiries ({len(expiries)}):", expiries[:15])

    # Show rows near our strike on the x100 scale, across the nearest expiries.
    near = base[(base["_strike_f"] >= STRIKE * 100 - 5000) & (base["_strike_f"] <= STRIKE * 100 + 5000)]
    print(f"\nrows near strike {STRIKE} (x100 = {STRIKE * 100}): {len(near)}")
    for rec in near.sort_values(["_strike_f"])[FIELDS].head(12).to_dict("records"):
        print("   ", rec)

    # Try the actual resolver against the nearest available expiry as a demo.
    sym = ""
    if expiries:
        demo_exp = dt.datetime.strptime(expiries[0], "%d%b%Y").date()
        sym = client.resolve_option_symbol(UNDERLYING, demo_exp, OPT, STRIKE)
        print(f"\nresolve_option_symbol({UNDERLYING}, {demo_exp}, {OPT}, {STRIKE}) -> {sym!r}")
    print("\nIf the rows above show your strike but resolve returns '', compare the")
    print("'_expiry_str' values to the expiry your strategy actually trades.")

    # Optional REAL round-trip test order (only when --place-order is passed).
    if PLACE_ORDER:
        if QTY is None or QTY <= 0:
            print(
                "\n--place-order requires an explicit positive --qty using the "
                "current contract lot size; no order placed."
            )
        elif sym:
            place_round_trip_test_order(client, sym, QTY, broker="Kotak")
        else:
            print("\n--place-order requested but the symbol did not resolve; no order placed.")


if __name__ == "__main__":
    main()
