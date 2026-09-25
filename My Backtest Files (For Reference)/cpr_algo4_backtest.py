"""
Backtest CPR Algo 4 (the deterministic "Intraday SRSI VWAP" playbook).

Beginner flow:
1. Read a raw 1-minute (or 5-minute) OHLC CSV.
2. Build the SAME completed 5-minute frame the live worker builds
   (`build_cpr_algo4_frame`: CPR levels, VWAP, RSI, EMAs, Stochastic RSI).
3. Walk the bars in order and drive the SAME `CPRAlgo4Engine` the live worker
   drives, with the SAME `check_intrabar_exit` stop/target check.
4. Save the trade list, a daily P&L table and a summary.

Why not backtesting.py like the older CPR backtest? That library fills stops and
targets with its own rules. Algo 4's live worker checks the stop FIRST on every
poll and trails on completed closes; a plain replay loop that calls the shared
engine functions keeps this backtest and the live worker making identical
decisions.

WHAT THE NUMBERS MEAN: P&L is measured in NIFTY SPOT POINTS (a LONG earns
exit - entry, a SHORT entry - exit). Live, every signal buys an ATM option, so
rupee results also depend on premium, delta and decay. Use this to compare
signal quality -- e.g. TARGET vs TRAIL, or the first-30-minute target on/off --
not to predict option P&L.

Fill assumptions (deliberately conservative):
- entries, adds and completed-bar exits fill at that bar's close;
- a candle that touches both the stop and a target is treated as a stop;
- a candle that OPENS beyond the stop fills at its open (a gap is a gap);
- any open trade is closed at the close of the bar ending at 15:15.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

# The backtest lives in `My Backtest Files (For Reference)`; the Algo 4 engine
# lives in `Signal Generators/CPR Strategy` and imports `Dependencies.*` from the
# repository root. Put both on sys.path so the script runs from anywhere.
ROOT_DIR = Path(__file__).resolve().parent.parent
CPR_SIGNAL_GENERATOR_DIR = ROOT_DIR / "Signal Generators" / "CPR Strategy"
for _path in (ROOT_DIR, CPR_SIGNAL_GENERATOR_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# (Imported after the sys.path setup above.)
from cpr_algo4_signal_generator import (
    EXIT_MODES,
    CPRAlgo4Config,
    CPRAlgo4Engine,
    CPRAlgo4TradePlan,
    build_cpr_algo4_frame,
    check_intrabar_exit,
)

OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"
DEFAULT_DATA_PATHS = {
    "nifty": OUTPUT_DIR / "nifty_renko_futures_5y_1min_data.csv",
    "banknifty": OUTPUT_DIR / "banknifty_renko_futures_5y_1min_data.csv",
    "finnifty": OUTPUT_DIR / "finnifty_renko_futures_5y_1min_data.csv",
}
SQUARE_OFF_TIME = dt_time(15, 15)
# Rows kept BEFORE --start so the previous session's CPR levels and the
# Stochastic RSI are already warmed up on the first requested day.
WARMUP_DAYS = 10


@dataclass
class ClosedTrade:
    """One finished backtest trade, in spot points."""

    session: date
    regime: str
    direction: str
    premise: str
    entry_time: datetime
    entry: float
    stop: float
    target: float
    final_target: float
    exit_time: datetime
    exit: float
    reason: str
    added: bool
    add_price: float
    points: float


@dataclass
class _OpenTrade:
    plan: CPRAlgo4TradePlan
    regime: str
    entry_time: datetime
    add_price: float = math.nan


def _gap_aware_fill(plan: CPRAlgo4TradePlan, reason: str, level: float, bar_open: float) -> float:
    """Fill price for an intrabar exit, honouring a gap through the level.

    A LONG stop that the candle OPENS below fills at the open (worse); a target
    it opens above fills at the open (better). Shorts mirror this.
    """

    if not math.isfinite(bar_open):
        return level
    is_stop = reason.endswith("_STOP")
    if plan.is_long:
        gapped = bar_open <= level if is_stop else bar_open >= level
    else:
        gapped = bar_open >= level if is_stop else bar_open <= level
    return bar_open if gapped else level


def _points(trade: _OpenTrade, exit_price: float) -> float:
    sign = 1.0 if trade.plan.is_long else -1.0
    points = sign * (exit_price - trade.plan.entry)
    if math.isfinite(trade.add_price):
        points += sign * (exit_price - trade.add_price)
    return points


def replay(frame: pd.DataFrame, config: CPRAlgo4Config) -> tuple[list[ClosedTrade], dict[date, str]]:
    """Walk completed 5-minute rows through the shared engine; return trades and day types.

    Per bar, in the same order the live worker experiences it:
    1. the open trade's stop / target / final level against this bar's range;
    2. the engine's completed-bar decision (entries, premise exits, trailing, add);
    3. the 15:15 square-off at the close of the bar that ends then.
    """

    engine = CPRAlgo4Engine(config)
    trades: list[ClosedTrade] = []
    regimes: dict[date, str] = {}
    open_trade: _OpenTrade | None = None
    last_close = math.nan
    last_time: datetime | None = None

    def close(trade: _OpenTrade, when: datetime, price: float, reason: str) -> None:
        plan = trade.plan
        trades.append(
            ClosedTrade(
                session=trade.entry_time.date(),
                regime=trade.regime,
                direction=plan.direction,
                premise=plan.premise,
                entry_time=trade.entry_time,
                entry=plan.entry,
                stop=plan.original_stop,
                target=plan.target,
                final_target=plan.final_target,
                exit_time=when,
                exit=float(price),
                reason=reason,
                added=math.isfinite(trade.add_price),
                add_price=trade.add_price,
                points=_points(trade, float(price)),
            )
        )
        engine.on_exit(when)

    for row in frame.to_dict("records"):
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        # A trade can never survive into a new session (the square-off below
        # normally closes it; this only guards a data hole at 15:10).
        if open_trade is not None and last_time is not None and ts.date() != last_time.date():
            close(open_trade, last_time, last_close, "SESSION_END")
            open_trade = None

        high, low, bar_open = float(row["high"]), float(row["low"]), float(row["open"])
        if open_trade is not None:
            hit = check_intrabar_exit(open_trade.plan, high=high, low=low)
            if hit is not None:
                reason, level = hit
                close(open_trade, ts, _gap_aware_fill(open_trade.plan, reason, level, bar_open), reason)
                open_trade = None

        decision = engine.on_bar(row, open_trade.plan if open_trade is not None else None)
        if engine.regime is not None:
            regimes.setdefault(ts.date(), engine.regime)
        bar_close = float(row["close"])
        if open_trade is not None and decision.action == "EXIT":
            close(open_trade, ts, bar_close, decision.reason)
            open_trade = None
        elif open_trade is not None and decision.action == "SCALE_IN":
            open_trade.plan.scale_in_used = True
            open_trade.add_price = bar_close
        elif open_trade is None and decision.action in ("ENTER_LONG", "ENTER_SHORT"):
            if decision.plan is not None:
                open_trade = _OpenTrade(plan=decision.plan, regime=engine.regime or "", entry_time=ts)
                engine.on_entry_filled(decision.plan)

        bar_end = (pd.Timestamp(ts) + pd.Timedelta(minutes=config.bar_minutes)).time()
        if open_trade is not None and bar_end >= SQUARE_OFF_TIME:
            close(open_trade, ts, bar_close, "SQUARE_OFF")
            open_trade = None
        last_close, last_time = bar_close, ts

    if open_trade is not None and last_time is not None:
        close(open_trade, last_time, last_close, "DATA_END")
    return trades, regimes


def summarize(trades: list[ClosedTrade], regimes: dict[date, str]) -> str:
    """Plain-text summary: totals, drawdown, and breakdowns by premise and exit reason."""

    lines = [f"Sessions: {len(regimes)}"]
    regime_counts = pd.Series(list(regimes.values()), dtype="object").value_counts()
    lines += [f"  {name}: {count}" for name, count in regime_counts.items()]
    if not trades:
        return "\n".join([*lines, "Trades: 0"])
    table = pd.DataFrame([asdict(trade) for trade in trades])
    points = table["points"]
    wins, losses = points[points > 0].sum(), -points[points < 0].sum()
    equity = points.cumsum()
    drawdown = (equity.cummax().clip(lower=0.0) - equity).max()
    lines += [
        f"Trades: {len(table)}  (with R1 add: {int(table['added'].sum())})",
        f"Win rate: {(points > 0).mean() * 100:.1f}%",
        f"Total points: {points.sum():.1f}   Avg/trade: {points.mean():.2f}",
        f"Profit factor: {wins / losses:.2f}" if losses > 0 else "Profit factor: n/a (no losing trade)",
        f"Max drawdown (points): {drawdown:.1f}",
        "",
        "By premise (trades / total points / win rate %):",
    ]
    for key, group in table.groupby("premise"):
        win_rate = (group["points"] > 0).mean() * 100
        lines.append(f"  {key}: {len(group)} / {group['points'].sum():.1f} / {win_rate:.1f}")
    lines.append("By exit reason (trades / total points):")
    for key, group in table.groupby("reason"):
        lines.append(f"  {key}: {len(group)} / {group['points'].sum():.1f}")
    return "\n".join(lines)


def load_frame(data_path: Path, config: CPRAlgo4Config, start: str = "", end: str = "") -> pd.DataFrame:
    """Read the CSV, trim to the requested window (plus warm-up) and build the 5-minute frame."""

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    raw = pd.read_csv(data_path)
    time_column = next((c for c in raw.columns if str(c).strip().lower() in ("timestamp", "datetime", "date")), None)
    if time_column is not None and (start or end):
        stamps = pd.to_datetime(raw[time_column], errors="coerce")
        keep = stamps.notna()
        if start:
            keep &= stamps >= pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)
        if end:
            keep &= stamps < pd.Timestamp(end) + pd.Timedelta(days=1)
        raw = raw.loc[keep.to_numpy()]
    frame = build_cpr_algo4_frame(raw, config)
    if start and not frame.empty:
        frame = frame.loc[(frame["timestamp"] >= pd.Timestamp(start)).to_numpy()].reset_index(drop=True)
    return frame


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (also reachable as `python algo.py backtest --strategy cpr-algo4`)."""

    parser = argparse.ArgumentParser(description="CPR Algo 4 (Intraday SRSI VWAP) spot-points backtest")
    parser.add_argument("--dataset", default="nifty", choices=sorted(DEFAULT_DATA_PATHS))
    parser.add_argument("--data", default="", help="Optional explicit OHLC CSV path")
    parser.add_argument("--exit-mode", default="TARGET", type=str.upper, choices=EXIT_MODES)
    parser.add_argument(
        "--first30-target", action="store_true", help="Sideways trades may book at the first-30-min extreme"
    )
    parser.add_argument("--no-scale-in", action="store_true", help="Disable the one R1 add")
    parser.add_argument("--start", default="", help="First session to trade, YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Last session to trade, YYYY-MM-DD")
    args = parser.parse_args(argv)

    config = CPRAlgo4Config(
        exit_mode=args.exit_mode,
        first30_target=args.first30_target,
        scale_in_enabled=not args.no_scale_in,
    )
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATHS[args.dataset]
    variant = args.exit_mode.lower()
    variant += "_first30" if args.first30_target else ""
    variant += "_noadd" if args.no_scale_in else ""
    prefix = OUTPUT_DIR / f"{args.dataset}_cpr_algo4_{variant}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(f"{prefix}_backtest.log"), logging.StreamHandler()],
    )
    logging.info("CPR Algo 4 backtest | data=%s | config=%s", data_path, config)

    frame = load_frame(data_path, config, args.start, args.end)
    logging.info("Built %s completed 5-minute bars", len(frame))
    trades, regimes = replay(frame, config)

    table = pd.DataFrame([asdict(trade) for trade in trades])
    table.to_csv(f"{prefix}_trades.csv", index=False)
    if not table.empty:
        daily = table.groupby("session")["points"].agg(["count", "sum"]).rename(
            columns={"count": "trades", "sum": "points"}
        )
        daily["cumulative_points"] = daily["points"].cumsum()
        daily.to_csv(f"{prefix}_daily.csv")
    summary = summarize(trades, regimes)
    Path(f"{prefix}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    logging.info("Summary:\n%s", summary)
    logging.info("Outputs written with prefix %s", prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
