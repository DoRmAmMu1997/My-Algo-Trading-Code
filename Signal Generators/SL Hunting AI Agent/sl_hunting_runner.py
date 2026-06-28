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
import sys

import numpy as np
import pandas as pd

# Running this file directly puts its folder on sys.path[0], so these bare imports
# resolve. (The folder name has spaces, so it can't be a normal package.)
from sl_hunting_agent import AgentRunResult, SLHuntingAgent
from sl_hunting_executor import StandaloneExecutor
from sl_hunting_indicators import prepare_candles

logger = logging.getLogger("sl_hunting_runner")


# ---------------------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
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


def resample_1m_to_n(candles: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-minute OHLC up to `minutes`-minute candles.

    Uses left-closed, left-labelled bins — the NSE intraday convention (the 09:15
    bar covers 09:15-09:19), matching the master's `resample_ohlc_from_1m`.
    """
    if minutes <= 1:
        return prepare_candles(candles)
    df = prepare_candles(candles).set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna(subset=["open"])
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
    val = os.getenv(name, "")
    return val.strip() if val and val.strip() else default


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Standalone PAPER runner for the SL Hunting AI Agent.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", help="Path to a 1-minute OHLC CSV (timestamp,open,high,low,close[,volume]).")
    src.add_argument("--synthetic", action="store_true", help="Use a generated random-walk session.")
    parser.add_argument("--timeframe", type=int, default=int(_env("SL_HUNTING_DERIVED_TIMEFRAME_MINUTES", "5")),
                        help="Decision timeframe in minutes (resampled from 1-min).")
    parser.add_argument("--model", default=_env("SL_HUNTING_MODEL", "claude-opus-4-8"), help="Claude model id.")
    parser.add_argument("--fast", action="store_true", help="Disable extended thinking (lower latency).")
    parser.add_argument("--fake", action="store_true", help="Use the built-in always-HOLD runner (no SDK/cost).")
    parser.add_argument("--warmup", type=int, default=30, help="Min completed bars before the first decision.")
    parser.add_argument("--max-bars", type=int, default=0, help="Cap the number of decisions (0 = all).")
    parser.add_argument("--lots", type=int, default=int(_env("SL_HUNTING_LOTS", "1")))
    parser.add_argument("--lot-size", type=int, default=int(_env("NIFTY_LOT_SIZE", "75")))
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

    # 2. Build the agent (fake runner for --fake; real Claude Agent SDK otherwise).
    runner = _AlwaysHoldRunner() if args.fake else None
    agent = SLHuntingAgent(model=args.model, runner=runner, fast_mode=args.fast)
    ex = StandaloneExecutor(lots=args.lots, lot_size=args.lot_size)

    # 3. Replay bar by bar.
    decisions = 0
    n = len(bars)
    for i in range(args.warmup, n):
        bar = bars.iloc[i]
        _manage_open_position(ex, bar)  # fill paper stop/target from this bar first
        window = bars.iloc[: i + 1]
        decision = agent.decide(window, ex, live_active=False, broker=None)
        decisions += 1
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

    trades = ex.closed_trades
    total = round(sum(t["pnl_proxy"] for t in trades), 2)
    wins = sum(1 for t in trades if t["pnl_proxy"] > 0)
    logger.info("=" * 60)
    logger.info("Session complete: %d decisions, %d paper trades.", decisions, len(trades))
    if trades:
        logger.info("Total P&L proxy: %.2f | wins %d/%d (%.0f%%)", total, wins, len(trades), 100.0 * wins / len(trades))
        for t in trades:
            logger.info("  %s %s->%s (%+.2f pts, %+.2f) [%s]", t["direction"], t["entry"], t["exit"], t["points"], t["pnl_proxy"], t["exit_reason"])
    else:
        logger.info("No trades taken (agent held throughout).")
    logger.info("NOTE: P&L proxy is on the NIFTY underlying, not option premium - validates decisions, not pricing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
