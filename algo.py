"""
algo.py - one friendly command-line entry point for the whole project.

============================== What is this? ==================================
This repo has several separate scripts you normally run one at a time, each with
its own long path and its own options:

    python "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py" --lookback 5y
    python "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py" --data ...
    python "Nifty Multi Strategy Front Test - Master File.py"

Remembering those exact paths is fiddly. This file is a small "front desk": you
tell it WHAT you want to do (a short command like `fetch-data` or `backtest`) and
it runs the right underlying script for you. It does not re-implement anything -
it just launches the existing scripts, so each one still works on its own exactly
as before. (This is the unified CLI requested in issue #16.)

============================== How do I use it? ===============================
From the repo root, the shape is always:

    python algo.py <command> [options...]

The commands:

  fetch-data   Download 1-minute historical OHLC data for an index.
      python algo.py fetch-data --index nifty
      python algo.py fetch-data --index banknifty --interval 5 --lookback 1y
      python algo.py fetch-data --index nifty --start-date 2024-01-01 --end-date 2024-03-31 --output mydata.csv

  backtest     Run one strategy's backtest against a CSV of historical data.
      python algo.py backtest --strategy renko --data "Backtest Outputs/nifty_renko_futures_5y_1min_data.csv"
      python algo.py backtest --strategy goldmine --data five_min.csv --dataset nifty

  run          Start the multithreaded front-test master (all strategies at once).
               Paper by default; goes live only if you configured a broker in
               Dependencies/.env (see the README). Reads ALL its settings from
               that .env, so this command takes no options of its own.
      python algo.py run

  setup-token  One-time DhanHQ login that writes a fresh access token into .env.
               Interactive (opens a browser / asks you to paste a code).
      python algo.py setup-token

  diagnose     Read-only broker connectivity + option-symbol check (can optionally
               place a confirmation-gated round-trip TEST order with --place-order).
      python algo.py diagnose --broker kotak CE 23950
      python algo.py diagnose --broker shoonya CE 23950 26JUN25 --place-order
      python algo.py diagnose --broker flattrade CE 24150 14JUL26

Tips:
  - `python algo.py --help` lists the commands; `python algo.py <command> --help`
    shows that command's selector choices.
  - Any option this CLI does not recognise is passed straight through to the
    underlying script, so every flag those scripts already accept still works.

============================== How it works ===================================
Each command maps to one existing script (see the dictionaries below). The CLI
parses only its own "selector" (e.g. --index / --strategy / --broker) and forwards
every other argument, untouched, to that script via a normal subprocess. The
script's own exit code is returned, so this CLI is transparent to automation.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# The repo root is simply the folder this file lives in. We resolve script paths
# relative to it so the CLI works no matter which directory you launch it from.
REPO_ROOT = Path(__file__).resolve().parent

# ---- command -> script mappings ----------------------------------------------
# Each key is the friendly value you pass on the command line; each value is the
# real script (relative to the repo root) that gets launched.

# `fetch-data --index <key>`
INDEX_SCRIPTS = {
    "nifty": "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py",
    "banknifty": "Data Extractors/Banknifty 1m 5Y Data Fetch Dhan.py",
    "finnifty": "Data Extractors/Finnifty 1m 5Y Data Fetch Dhan.py",
}

# `backtest --strategy <key>`
BACKTEST_SCRIPTS = {
    "renko": "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py",
    "ema": "My Backtest Files (For Reference)/Nifty EMA Trend Strategy Backtest.py",
    "heikin": "My Backtest Files (For Reference)/Nifty Heiken Ashi Futures 5Y Backtest.py",
    "cpr": "My Backtest Files (For Reference)/Nifty CPR Strategy Backtest.py",
    "profit-shooter": "My Backtest Files (For Reference)/profit_shooter_backtest.py",
    "goldmine": "My Backtest Files (For Reference)/Subhamoy Strategies/Nifty Goldmine Strategy Backtest.py",
    "money-machine": "My Backtest Files (For Reference)/Subhamoy Strategies/Nifty Money Machine Strategy Backtest.py",
}

# `diagnose --broker <key>`
BROKER_DIAGNOSTICS = {
    "flattrade": "Dependencies/Flattrade API/diagnose_flattrade_symbol.py",
    "kotak": "Dependencies/Kotak API/diagnose_kotak_symbol.py",
    "shoonya": "Dependencies/Shoonya API/diagnose_shoonya_symbol.py",
}

# Commands that always map to exactly one script (no selector needed).
MASTER_SCRIPT = "Nifty Multi Strategy Front Test - Master File.py"
TOKEN_SETUP_SCRIPT = "Dependencies/dhan_token_setup.py"


def _run(relative_path: str, forwarded_args: list) -> int:
    """
    Launch one underlying script and return its exit code.

    `relative_path` is the script's location relative to the repo root, and
    `forwarded_args` are the leftover command-line options we hand straight to it.
    We run it with the repo root as the working directory so the script's own
    relative paths (e.g. the "Backtest Outputs/" folder) resolve correctly.
    """
    script = REPO_ROOT / relative_path
    if not script.is_file():
        # A clear, beginner-friendly error beats a confusing Python traceback.
        print(f"[algo] ERROR: expected script not found:\n  {script}", file=sys.stderr)
        return 2
    command = [sys.executable, str(script), *forwarded_args]
    # Echo what we're about to do so it's obvious which script is running.
    print(f"[algo] running: {script.name} {' '.join(forwarded_args)}".rstrip())
    return subprocess.run(command, cwd=str(REPO_ROOT)).returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-command per task."""
    parser = argparse.ArgumentParser(
        prog="algo",
        description="Unified command-line entry point for data fetching, backtesting, "
                    "live/paper execution, and setup. Run `python algo.py <command> --help` "
                    "for a command's options.",
    )
    # Intentionally NOT `required=True`: a bare `python algo.py` (no command) should
    # print the full help (handled in main()) rather than raise a terse argparse error.
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    fetch = sub.add_parser(
        "fetch-data",
        help="Download 1-minute OHLC data for an index (NIFTY/BANKNIFTY/FINNIFTY).",
    )
    fetch.add_argument(
        "--index", required=True, choices=sorted(INDEX_SCRIPTS),
        help="Which index to download. All other flags (--interval, --lookback, "
             "--start-date, --end-date, --output, ...) pass through to the fetcher.",
    )

    backtest = sub.add_parser(
        "backtest",
        help="Run one strategy's backtest against a historical-data CSV.",
    )
    backtest.add_argument(
        "--strategy", required=True, choices=sorted(BACKTEST_SCRIPTS),
        help="Which strategy to backtest. Flags like --data and --dataset pass "
             "through to that strategy's backtest script.",
    )

    sub.add_parser(
        "run",
        help="Start the multithreaded front-test master (paper by default; live per Dependencies/.env).",
    )

    sub.add_parser(
        "setup-token",
        help="One-time interactive DhanHQ login that writes a fresh token into .env.",
    )

    diagnose = sub.add_parser(
        "diagnose",
        help="Read-only broker connectivity / option-symbol check (supports --place-order).",
    )
    diagnose.add_argument(
        "--broker", required=True, choices=sorted(BROKER_DIAGNOSTICS),
        help="Which broker to check. Extra args (e.g. CE 23950 --place-order) pass "
             "through to the broker's diagnostic script.",
    )

    return parser


def main(argv=None) -> int:
    """
    Parse the command line and launch the matching underlying script.

    Only the selector for each command (--index / --strategy / --broker) is parsed
    here; `parse_known_args` collects everything else into `forwarded` so it can be
    passed straight through. Returns the launched script's exit code.
    """
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)

    # A bare `python algo.py` (no command) prints the full help and exits cleanly —
    # friendlier than argparse's terse "a command is required" error. (Explicit
    # `--help` on the program or any sub-command is handled by argparse already.)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "fetch-data":
        return _run(INDEX_SCRIPTS[args.index], forwarded)
    if args.command == "backtest":
        return _run(BACKTEST_SCRIPTS[args.strategy], forwarded)
    if args.command == "run":
        return _run(MASTER_SCRIPT, forwarded)
    if args.command == "setup-token":
        return _run(TOKEN_SETUP_SCRIPT, forwarded)
    if args.command == "diagnose":
        return _run(BROKER_DIAGNOSTICS[args.broker], forwarded)

    # Unreachable: argparse rejects unknown commands before we get here.
    parser.error(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
