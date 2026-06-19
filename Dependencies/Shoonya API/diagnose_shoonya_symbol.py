"""
Diagnostic for Shoonya option symbol resolution (and an optional REAL test order).

Run during a LIVE Shoonya session (TOTP auto-generated from SHOONYA_TOTP_SECRET,
or it prompts):

    python "Multithreading/Dependencies/diagnose_shoonya_symbol.py" [OPT] [STRIKE] [EXPIRY]
    # e.g. python ".../diagnose_shoonya_symbol.py" CE 23950 26JUN25

It logs in, downloads the NSE F&O symbol master ONCE (the same cache the live
runner uses), and shows whether our resolver builds a valid tsym for the given
option type / strike / expiry. Read-only by default.

Optional REAL round-trip test order (BUY then auto square-off) - REAL MONEY:
    python ".../diagnose_shoonya_symbol.py" CE 23950 26JUN25 --place-order [--qty 75]
It places a 1-lot market BUY on the resolved contract, confirms the fill, then
immediately SELLs to flatten. Requires typing YES to confirm. (Needs an EXPIRY.)
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shoonya_execution as se  # handles sys.path to the SDK + loads .env


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
# Expiry as DDMMMYY (e.g. 26JUN25). Optional; without it we just sample the
# live symbol format rather than resolving a specific contract.
EXPIRY = _POS[2].upper() if len(_POS) > 2 else None
PLACE_ORDER = "--place-order" in sys.argv
QTY = int(_arg_val("--qty", "75"))  # NIFTY lot = 75


def place_round_trip_test_order(client, symbol, qty, broker="Shoonya"):
    """Place a REAL 1-lot market BUY, confirm the fill, then auto SELL to flatten.
    Guarded by a typed YES confirmation. REAL MONEY."""
    print("\n" + "=" * 72)
    print(f"REAL ROUND-TRIP TEST ORDER on {broker}")
    print(f"  BUY {qty} {symbol} (product=INTRADAY), then auto SELL {qty} to flatten.")
    print("  THIS PLACES REAL ORDERS WITH REAL MONEY.")
    try:
        confirm = input("  Type exactly YES to proceed (anything else aborts): ").strip()
    except (EOFError, OSError):
        confirm = ""
    if confirm != "YES":
        print("  Aborted (no confirmation).")
        return
    print(f"  Placing BUY {qty} {symbol} ...")
    try:
        entry = client.place_market_order(symbol=symbol, side="BUY", quantity=qty, product_type="INTRADAY")
    except Exception as exc:
        print(f"  ENTRY FAILED: {exc}\n  No position opened.")
        return
    print(f"  ENTRY filled. OrderId={client.extract_order_id(entry)}")
    print(f"  Squaring off: SELL {qty} {symbol} ...")
    try:
        ex = client.place_market_order(symbol=symbol, side="SELL", quantity=qty, product_type="INTRADAY")
        print(f"  EXIT filled. OrderId={client.extract_order_id(ex)} -> flat. Round-trip OK.")
    except Exception as exc:
        print(f"  !! EXIT FAILED: {exc}")
        print(f"  !! A LIVE long position in {symbol} (qty={qty}) IS OPEN. SQUARE IT OFF MANUALLY NOW.")


def main():
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
        print(f"\nNo expiry given (3rd arg, e.g. 26JUN25). Showing a few sample "
              f"{UNDERLYING} {OPT[0]} symbols so you can read the live format:")
        for s in sorted(s for s in nifty_syms if OPT[0] in s)[:10]:
            print("   ", s)
        if PLACE_ORDER:
            print("\n--place-order needs an EXPIRY (3rd arg) to resolve the contract; no order placed.")
        return

    try:
        expiry = dt.datetime.strptime(EXPIRY, "%d%b%y").date()
    except ValueError:
        print(f"Could not parse expiry {EXPIRY!r}; expected DDMMMYY like 26JUN25.")
        return

    sym = client.resolve_option_symbol(UNDERLYING, expiry, OPT, STRIKE)
    print(f"\nresolve_option_symbol({UNDERLYING}, {expiry}, {OPT}, {STRIKE}) -> {sym!r}")
    if not sym:
        near_prefix = f"{UNDERLYING}{EXPIRY}{OPT[0]}"
        near = sorted(s for s in symbols if s.startswith(near_prefix))
        print(f"  sample {near_prefix}* symbols: {near[:5]} ... {near[-5:] if near else ''}")

    # Optional REAL round-trip test order (only when --place-order is passed).
    if PLACE_ORDER:
        if sym:
            place_round_trip_test_order(client, sym, QTY, broker="Shoonya")
        else:
            print("\n--place-order requested but the symbol did not resolve; no order placed.")


if __name__ == "__main__":
    main()
