"""Standalone PAPER runner for the SL Hunting AI Agent.

What this is (beginner note)
----------------------------
A self-contained harness to validate the agent on real (or synthetic) NIFTY data
WITHOUT touching the live multithreaded master. It loads 1-minute OHLC, resamples
to the agent's decision timeframe, then replays the session bar by bar: on each
completed bar it lets the agent decide (paper execution via `StandaloneExecutor`),
auto-fills paper stop/target hits, and prints a session summary.

Modes
-----
- ``--synthetic``  : generate a random-walk session (no data file, no SDK needed).
- ``--fake``       : drive the agent with a built-in always-HOLD runner (zero cost,
                     no Claude CLI) — use this to smoke-test the whole pipeline.
- default          : use the real Claude Agent SDK (needs ``pip install
                     claude-agent-sdk`` and a one-time ``claude`` CLI login;
                     keep ``ANTHROPIC_API_KEY`` UNSET for subscription billing).

Examples
--------
    # zero-cost pipeline smoke test (no data, no SDK):
    python sl_hunting_runner.py --synthetic --fake

    # replay a real 1-min CSV with the live agent, capped to 30 decisions:
    python sl_hunting_runner.py --csv "../../Backtest Outputs/nifty_1m.csv" --max-bars 30

This runner is PAPER-ONLY by design. Real-money trading goes through the master
front-test worker (one shared, lock-guarded broker session) — see the folder
README.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Hashable

import numpy as np
import pandas as pd

# Running this file directly puts its folder on sys.path[0], so these bare imports
# resolve. (The folder name has spaces, so it can't be a normal package.)
from sl_hunting_agent import AgentRunResult, SLHuntingAgent
from sl_hunting_executor import StandaloneExecutor
from sl_hunting_indicators import prepare_candles
from sl_hunting_journal import TradeJournal, make_entry_record
from sl_hunting_lessons import format_lessons, load_lessons

logger = logging.getLogger("sl_hunting_runner")

# Default journal path: <repo root>/Backtest Outputs/ (gitignored).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_JOURNAL = os.path.join(_REPO_ROOT, "Backtest Outputs", "sl_hunting_journal.jsonl")
_DEFAULT_LESSONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")


def _open_journal_row(journal: TradeJournal, decision, window, bnf_window, ex: StandaloneExecutor) -> str:
    """Open a journal row for a freshly-entered trade (paper).

    The EXECUTED trade details (direction / stop / target / entry / lots) are read
    from the EXECUTOR position -- the source of truth -- not from `decision`. The
    agent acts via the order tool DURING decide(), so the position is already live;
    if the model's final JSON then comes back malformed or disagrees with the tool
    call (e.g. a safe HOLD), trusting `decision.action` here would mislabel the trade
    (any non-ENTER_LONG -> "SHORT") and store the wrong stop/target, corrupting the
    rows the coach later learns from. Only the rationale (setup / confidence /
    reasoning) is taken from the decision. Mirrors the master worker's _journal_open_row.
    """
    entry = make_entry_record(
        direction=ex.pos.direction, setup=decision.setup, confidence=decision.confidence,
        stop=ex.pos.stop, target=ex.pos.target, reasoning=decision.reasoning,
        entry_underlying=ex.pos.entry, lots=ex.pos.lots, nifty_df=window, bnf_df=bnf_window,
    )
    return journal.open_trade(entry)


def _close_journal_row(journal: TradeJournal, open_id: str, ex: StandaloneExecutor) -> None:
    """Close the open journal row from the executor's most recent closed trade."""
    if not ex.closed_trades:
        return
    t = ex.closed_trades[-1]
    journal.close_trade(open_id, {
        "exit_underlying": t["exit"], "exit_reason": t["exit_reason"],
        "points": t["points"], "option_pnl": t["pnl_proxy"], "lots": t["lots"],
    })


# ---------------------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    """Load a 1-minute OHLC CSV into a clean candle frame (tolerant of the time column name)."""
    df = pd.read_csv(path)
    # Be forgiving about the timestamp column name.
    rename = {}
    for cand in ("datetime", "date", "Datetime", "Date", "time", "Time"):
        if cand in df.columns and "timestamp" not in df.columns:
            rename[cand] = "timestamp"
    if rename:
        df = df.rename(columns=rename)
    return prepare_candles(df)


def _synthetic_session(bars_1m: int = 375, seed: int = 7, start_price: float = 25000.0) -> pd.DataFrame:
    """Generate one trading day of synthetic 1-minute candles (random walk)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 4.0, size=bars_1m).cumsum()
    closes = start_price + steps
    ts = pd.date_range(start="2026-06-26 09:15", periods=bars_1m, freq="1min")
    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0, 6, size=bars_1m)
    lows = np.minimum(opens, closes) - rng.uniform(0, 6, size=bars_1m)
    return pd.DataFrame(
        {"timestamp": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": 0}
    )


def _synthetic_bnf(nifty_1m: pd.DataFrame, scale: float = 2.2, seed: int = 11) -> pd.DataFrame:
    """Derive a correlated synthetic BankNIFTY 1-min series from the NIFTY one.

    BankNIFTY trades at roughly `scale`x NIFTY and moves together with it, so we
    scale the NIFTY closes and add small idiosyncratic noise. Same timestamps, so
    the two align bar-for-bar. Used only by --synthetic to exercise cross-index.
    """
    nifty_1m = prepare_candles(nifty_1m)
    rng = np.random.default_rng(seed)
    n = len(nifty_1m)
    closes = nifty_1m["close"].to_numpy() * scale + rng.normal(0, 12, size=n)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0, 15, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 15, size=n)
    return pd.DataFrame({
        "timestamp": nifty_1m["timestamp"].to_numpy(),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 0,
    })


def resample_1m_to_n(candles: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-minute OHLC up to `minutes`-minute candles, keeping ONLY complete bars.

    Uses left-closed, left-labelled bins — the NSE intraday convention (the 09:15 bar
    covers 09:15-09:19) — and drops any bucket that did not receive all `minutes`
    source 1-minute rows (e.g. a 09:15-09:17 tail in a CSV), exactly like the master's
    `resample_ohlc_from_1m`. This stops the agent from ever deciding on a partial bar.
    """
    if minutes <= 1:
        return prepare_candles(candles)
    prepared = prepare_candles(candles)
    timestamps = pd.DatetimeIndex(prepared["timestamp"])
    bucket_keys = timestamps.floor(f"{minutes}min")
    complete = np.zeros(len(prepared), dtype=bool)
    for bucket in bucket_keys.unique():
        positions = np.flatnonzero(bucket_keys == bucket)
        actual = list(timestamps[positions])
        expected = list(pd.date_range(bucket, periods=minutes, freq="1min"))
        if len(actual) == minutes and actual == expected:
            complete[positions] = True
    prepared = prepared.loc[complete].reset_index(drop=True)
    if prepared.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = prepared.set_index("timestamp")
    rule = f"{minutes}min"
    # The Hashable key annotation matches the Mapping type that pandas-stubs
    # expects for .agg(); a plain dict[str, str] is rejected (invariant keys).
    agg: dict[Hashable, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    counts = df["close"].resample(rule, label="left", closed="left").count()
    out = out[counts.reindex(out.index) == minutes]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Paper position management between agent decisions
# ---------------------------------------------------------------------------

def _manage_open_position(ex: StandaloneExecutor, bar: pd.Series) -> None:
    """Auto-fill a paper stop/target hit using the current bar's high/low."""
    if not ex.pos.active:
        return
    high, low = float(bar.high), float(bar.low)
    if ex.pos.direction == "LONG":
        if ex.pos.stop and low <= ex.pos.stop:
            ex.exit("stop_hit", price=ex.pos.stop)
        elif ex.pos.target and high >= ex.pos.target:
            ex.exit("target_hit", price=ex.pos.target)
    else:  # SHORT
        if ex.pos.stop and high >= ex.pos.stop:
            ex.exit("stop_hit", price=ex.pos.stop)
        elif ex.pos.target and low <= ex.pos.target:
            ex.exit("target_hit", price=ex.pos.target)


class _AlwaysHoldRunner:
    """A built-in fake runner (for --fake): no SDK, no cost, always HOLD."""

    async def __call__(self, prompt, *, system_prompt, model, max_turns, tool_context=None):
        import json
        text = json.dumps(
            {"action": "HOLD", "stop": 0, "target": 0, "confidence": 0,
             "setup": "none", "reasoning": "Fake runner: pipeline smoke test.", "model_used": model}
        )
        return AgentRunResult(text=text, cost_usd=0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    """Read an env var (trimmed); fall back to ``default`` when blank/unset.

    Lets every CLI flag below default to the matching ``SL_HUNTING_*`` env value, so the
    standalone runner honours the same `.env` knobs the master front-test uses.
    """
    val = os.getenv(name, "")
    return val.strip() if val and val.strip() else default


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Standalone PAPER runner for the SL Hunting AI Agent.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", help="Path to a 1-minute NIFTY OHLC CSV (timestamp,open,high,low,close[,volume]).")
    src.add_argument("--synthetic", action="store_true", help="Use a generated random-walk session.")
    parser.add_argument("--bnf-csv",
                        help="Optional 1-minute BankNIFTY OHLC CSV for cross-confirmation (paired with --csv).")
    parser.add_argument("--timeframe", type=int, default=int(_env("SL_HUNTING_DERIVED_TIMEFRAME_MINUTES", "1")),
                        help="Decision timeframe in minutes (resampled from 1-min; SL Hunting's default is 1).")
    parser.add_argument("--model", default=_env("SL_HUNTING_MODEL", "claude-opus-4-8"), help="Claude model id.")
    parser.add_argument("--fast", action="store_true", help="Disable extended thinking (lower latency).")
    parser.add_argument("--fake", action="store_true", help="Use the built-in always-HOLD runner (no SDK/cost).")
    parser.add_argument("--warmup", type=int, default=30, help="Min completed bars before the first decision.")
    parser.add_argument("--max-bars", type=int, default=0, help="Cap the number of decisions (0 = all).")
    parser.add_argument(
        "--lots",
        type=int,
        default=int(_env("SL_HUNTING_LOTS", "1")),
        help="Legacy recovery fallback only; accepted entries use hard-budget sizing.",
    )
    parser.add_argument("--lot-size", type=int, default=int(_env("NIFTY_LOT_SIZE", "75")))
    parser.add_argument(
        "--risk-budget",
        type=float,
        default=float(_env("SL_HUNTING_RISK_BUDGET", "2500")),
        help="Hard NIFTY-leg risk budget used to floor affordable whole lots.",
    )
    parser.add_argument(
        "--max-lots",
        type=int,
        default=int(_env("SL_HUNTING_MAX_LOTS", "5")),
        help="Hard ceiling for dynamically sized NIFTY lots.",
    )
    parser.add_argument("--journal", default=_DEFAULT_JOURNAL,
                        help="Path to the trade-journal JSONL (the coach reads this).")
    parser.add_argument("--no-journal", action="store_true", help="Disable trade journaling.")
    parser.add_argument("--lessons", choices=["on", "off"], default="off",
                        help="Inject the approved LEARNED LESSONS into the agent (A/B against 'off').")
    parser.add_argument("--lessons-path", default=_DEFAULT_LESSONS)
    args = parser.parse_args(argv)

    # 1. Load data.
    if args.csv:
        candles_1m = _load_csv(args.csv)
        logger.info("Loaded %d 1-min candles from %s", len(candles_1m), args.csv)
    else:
        candles_1m = _synthetic_session()
        logger.info("Using synthetic session (%d 1-min candles).", len(candles_1m))
        if not args.synthetic and not args.fake:
            logger.warning("No --csv given; defaulting to a synthetic session. Pass --csv for real data.")

    bars = resample_1m_to_n(candles_1m, args.timeframe)
    if len(bars) <= args.warmup + 1:
        logger.error("Not enough %d-min bars (%d) for warmup=%d.", args.timeframe, len(bars), args.warmup)
        return 2
    logger.info("Resampled to %d %d-min bars.", len(bars), args.timeframe)

    # 1b. Optional BankNIFTY (cross-confirmation). A real --bnf-csv when given;
    # a correlated synthetic series in synthetic mode; otherwise None (NIFTY-only).
    bnf_bars = None
    if args.bnf_csv:
        bnf_bars = resample_1m_to_n(_load_csv(args.bnf_csv), args.timeframe)
        logger.info("Loaded BankNIFTY for cross-confirmation from %s.", args.bnf_csv)
    elif not args.csv:
        bnf_bars = resample_1m_to_n(_synthetic_bnf(candles_1m), args.timeframe)
        logger.info("Using synthetic BankNIFTY for cross-confirmation.")
    else:
        logger.info("No --bnf-csv: running NIFTY-only (cross-index will report unavailable).")

    # 2. Build the agent (fake runner for --fake; real Claude Agent SDK otherwise).
    lessons_block = ""
    if args.lessons == "on":
        lessons_block = format_lessons(load_lessons(args.lessons_path))
        logger.info("Lessons ON: injected %d learned-lesson chars from %s.", len(lessons_block), args.lessons_path)
    runner = _AlwaysHoldRunner() if args.fake else None
    agent = SLHuntingAgent(model=args.model, runner=runner, fast_mode=args.fast, lessons_block=lessons_block)
    ex = StandaloneExecutor(
        lots=args.lots,
        lot_size=args.lot_size,
        risk_budget=args.risk_budget,
        max_lots=args.max_lots,
    )

    journal = None if args.no_journal else TradeJournal(args.journal)
    if journal is not None:
        logger.info("Journaling trades to %s", args.journal)
    open_id = None

    # 3. Replay bar by bar.
    decisions = 0
    n = len(bars)
    for i in range(args.warmup, n):
        bar = bars.iloc[i]
        # Two position snapshots are taken per bar so we can tell WHY a trade closed:
        #  • was_active  — were we in a trade BEFORE filling this bar's stop/target? If we
        #                  were and now we're flat, the bar's high/low closed it.
        #  • pre_active  — (below) were we in a trade BEFORE the agent decided? Lets us tell
        #                  an agent EXIT apart from a brand-new entry, to journal correctly.
        was_active = ex.pos.active
        _manage_open_position(ex, bar)  # fill paper stop/target from this bar first
        if journal is not None and open_id is not None and was_active and not ex.pos.active:
            _close_journal_row(journal, open_id, ex)  # closed by a stop/target hit
            open_id = None

        window = bars.iloc[: i + 1]
        bnf_window = None
        if bnf_bars is not None:
            bnf_window = bnf_bars[bnf_bars["timestamp"] <= bar.timestamp]
            if bnf_window.empty:
                bnf_window = None
        pre_active = ex.pos.active
        decision = agent.decide(window, ex, bnf_candles=bnf_window, live_active=False, broker=None)
        decisions += 1
        if journal is not None:
            if pre_active and not ex.pos.active and open_id is not None:
                _close_journal_row(journal, open_id, ex)  # agent EXIT
                open_id = None
            elif not pre_active and ex.pos.active:
                open_id = _open_journal_row(journal, decision, window, bnf_window, ex)  # new entry
        if decision.action != "HOLD":
            logger.info(
                "[%s] %s (conf=%d, setup=%s) stop=%s target=%s :: %s",
                str(bar.timestamp)[:19], decision.action, decision.confidence,
                decision.setup, decision.stop, decision.target, decision.reasoning,
            )
        if args.max_bars and decisions >= args.max_bars:
            logger.info("Reached --max-bars=%d; stopping.", args.max_bars)
            break

    # 4. Close any open position at the last price and summarise.
    if ex.pos.active:
        ex.exit("session_end", price=float(bars.iloc[min(i, n - 1)].close))
        if journal is not None and open_id is not None:
            _close_journal_row(journal, open_id, ex)
            open_id = None

    trades = ex.closed_trades
    total = round(sum(t["pnl_proxy"] for t in trades), 2)
    wins = sum(1 for t in trades if t["pnl_proxy"] > 0)
    logger.info("=" * 60)
    logger.info("Session complete: %d decisions, %d paper trades.", decisions, len(trades))
    if trades:
        logger.info("Total P&L proxy: %.2f | wins %d/%d (%.0f%%)", total, wins, len(trades), 100.0 * wins / len(trades))
        for t in trades:
            logger.info("  %s %s->%s (%+.2f pts, %+.2f) [%s]", t["direction"], t["entry"],
                        t["exit"], t["points"], t["pnl_proxy"], t["exit_reason"])
    else:
        logger.info("No trades taken (agent held throughout).")
    logger.info("NOTE: P&L proxy is on the NIFTY underlying, not option premium - validates decisions, not pricing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
