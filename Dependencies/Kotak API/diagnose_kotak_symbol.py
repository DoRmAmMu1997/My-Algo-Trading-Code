"""
Diagnostic for Kotak option symbol resolution (and an optional REAL test order).

Run during a LIVE Kotak session (it prompts for a TOTP at login):

    python "Multithreading/Dependencies/diagnose_kotak_symbol.py" [OPT] [STRIKE]
    # e.g. python ".../diagnose_kotak_symbol.py" CE 23950

It downloads the NSE F&O scrip master ONCE (the same cache the live runner uses),
then shows how Kotak stores NIFTY option rows near your strike, the available
expiries, and whether our resolver finds the exact pTrdSymbol. Read-only by default.

Optional REAL round-trip test order (BUY then auto square-off) - REAL MONEY:
    python ".../diagnose_kotak_symbol.py" CE 23950 --place-order [--qty 75]
It places a 1-lot market BUY on the resolved nearest-expiry contract, confirms the
fill, then immediately SELLs to flatten. Requires typing YES to confirm.
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kotak_execution as ke  # handles sys.path to the SDK + loads .env


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
QTY = int(_arg_val("--qty", "75"))  # NIFTY lot = 75
FIELDS = ["pSymbolName", "pOptionType", "_expiry_str", "dStrikePrice;", "pTrdSymbol"]


def place_round_trip_test_order(client, symbol, qty, broker="Kotak"):
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
    client = ke.kotak_execution_client
    print("Logging in (you'll be prompted for a TOTP)...")
    if not client.preload_scrip_master():
        print("Could not log in / load the scrip master. Check the messages above.")
        return

    df = client._scrip_df
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
        if sym:
            place_round_trip_test_order(client, sym, QTY, broker="Kotak")
        else:
            print("\n--place-order requested but the symbol did not resolve; no order placed.")


if __name__ == "__main__":
    main()
