"""
Backtest the deterministic core of CPR AI's Trend-Day Rider.

Beginner flow:
1. Read the NIFTY 1-minute CSV and group it into sessions.
2. For each session, build the completed 5-minute bars the live worker builds
   (only buckets with all five minutes) and ask the SAME
   `evaluate_trend_day_candidate` the live host uses whether the newest bar is
   a candidate. The first eligible bar of the day is taken -- this is the
   baseline in which Codex accepts every candidate; live, Codex may veto.
3. Enter a couple of minutes after the bar closes (the model needs time),
   hold until the spot touches the VWAP stop or 15:15, and never re-enter.
4. Price the trade on REAL option premiums when the expired-options folder
   (`python algo.py fetch-expired-options`) is present: SELL the opposite ATM
   option (the live expression) and, for comparison, BUY the directional one.
   Without that folder the script reports NIFTY spot points only.

Fill assumptions:
- entry spot and premium are the close of the minute `--entry-delay` minutes
  after the signal bar's last minute; if the spot is already through the stop
  there, the candidate is skipped (live blocks it as `stop_already_breached`)
  and later bars may still qualify;
- a stop is detected on the 1-minute high/low; spot P&L books the stop level,
  premium P&L books the contract's close `--exit-delay` minutes later;
- `--cost` premium points are charged per round trip (spread, slippage, fees);
- `--max-loss-rupees` > 0 mimics the worker's max-loss kill switch on the sold
  leg's minute-by-minute mark-to-market at `--lot-size`.
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT_DIR / "Signal Generators" / "CPR AI Agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# (Imported after the sys.path setup above.)
from cpr_ai_trend_day import (
    DEFAULT_TREND_DAY_CONFIG,
    LONG,
    TrendDayConfig,
    evaluate_trend_day_candidate,
    session_atr,
)

OUTPUT_DIR = ROOT_DIR / "Backtest Outputs"
DEFAULT_DATA_PATH = OUTPUT_DIR / "nifty_renko_futures_5y_1min_data.csv"
DEFAULT_OPTIONS_DIR = OUTPUT_DIR / "expired_options" / "nifty"
SESSION_OPEN = 9 * 60 + 15
LAST_MINUTE = 15 * 60 + 29
SQUARE_OFF_MINUTE = 15 * 60 + 14  # the close of this minute is the 15:15 mark
STRIKE_STEP = 50
# Keys pack (strike step index, session index, minute of day) into one int64.
_STRIKE_FACTOR = 10_000_000
_SESSION_FACTOR = 1_440


@dataclass
class Trade:
    """One finished trade: spot points plus (when priced) option premium points."""

    session: date
    direction: str
    bar_start: str
    entry_time: str
    entry_spot: float
    stop: float
    exit_time: str
    exit_spot: float
    reason: str
    spot_points: float
    strike: float
    sold_right: str
    sold_entry: float
    sold_exit: float
    sell_points: float
    buy_points: float
    confluence_score: int
    beyond_r1_s1: bool
    gap_in_direction: bool
    extended_from_vwap: bool
    range_atr: float
    risk_points: float
    days_to_expiry: float


class OptionBook:
    """Strike-keyed lookup of 1-minute premiums across the ATM+-n files.

    The expired-options files label strikes RELATIVE to spot (ATM+1 ...), so a
    fixed contract drifts between files as spot moves. Merging every file and
    keying by the absolute strike lets a trade follow its own contract.
    """

    def __init__(self, options_dir: Path, sessions: list[date]) -> None:
        self._session_index = {session: index for index, session in enumerate(sessions)}
        self._books: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for right, suffix in (("CE", "CALL"), ("PE", "PUT")):
            parts = [
                _read_csv(Path(path), ["timestamp", "close", "strike_price"])
                for path in sorted(glob.glob(str(options_dir / f"nifty_1m_WEEK_ATM*_{suffix}.csv")))
            ]
            if not parts:
                raise FileNotFoundError(f"No {suffix} files in {options_dir}")
            frame = pd.concat(parts, ignore_index=True)
            stamps = pd.to_datetime(frame["timestamp"])
            session_ids = stamps.dt.date.map(self._session_index)
            keep = session_ids.notna().to_numpy()
            minutes = (stamps.dt.hour * 60 + stamps.dt.minute).to_numpy()[keep]
            keys = (
                np.round(frame["strike_price"].to_numpy()[keep] / STRIKE_STEP).astype(np.int64) * _STRIKE_FACTOR
                + session_ids.to_numpy()[keep].astype(np.int64) * _SESSION_FACTOR
                + minutes.astype(np.int64)
            )
            order = np.argsort(keys, kind="stable")
            unique_keys, first = np.unique(keys[order], return_index=True)
            self._books[right] = (unique_keys, frame["close"].to_numpy(dtype=float)[keep][order][first])

    def path(self, right: str, strike: float, session: date, start_minute: int, end_minute: int) -> np.ndarray:
        """Premium closes for one contract, minute by minute (last value carried up to 5 minutes)."""

        keys, values = self._books[right]
        base = round(strike / STRIKE_STEP) * _STRIKE_FACTOR + self._session_index[session] * _SESSION_FACTOR
        wanted = base + np.arange(start_minute, end_minute + 1)
        found = np.searchsorted(keys, wanted, side="right") - 1
        safe = np.maximum(found, 0)
        ok = (found >= 0) & (keys[safe] >= base + SESSION_OPEN) & (wanted - keys[safe] <= 5)
        return np.where(ok, values[safe], np.nan)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns, using the faster pyarrow engine when available."""

    try:
        return pd.read_csv(path, usecols=columns, engine="pyarrow")
    except (ImportError, ValueError):
        return pd.read_csv(path, usecols=columns)


def load_sessions(data_path: Path, start: str = "", end: str = "") -> dict[date, pd.DataFrame]:
    """Return regular-session 1-minute bars per day, sorted, with a minute-of-day column."""

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    raw = _read_csv(data_path, ["timestamp", "open", "high", "low", "close"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    minute = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
    raw = raw.loc[((minute >= SESSION_OPEN) & (minute <= LAST_MINUTE)).to_numpy()].copy()
    raw["minute"] = minute
    if start:
        # Keep a fortnight before --start so the first requested day has ATR5.
        raw = raw.loc[(raw["timestamp"] >= pd.Timestamp(start) - pd.Timedelta(days=14)).to_numpy()]
    if end:
        raw = raw.loc[(raw["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1)).to_numpy()]
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return {
        session: group.reset_index(drop=True)
        for session, group in raw.groupby(raw["timestamp"].dt.date, sort=True)
    }


def completed_five_minute_bars(minutes: pd.DataFrame) -> pd.DataFrame:
    """Resample one session into 5-minute bars, keeping only buckets with all five minutes."""

    indexed = minutes.set_index("timestamp")
    bars = indexed.resample("5min", label="left", closed="left", origin="start_day").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        count=("close", "count"),
    )
    return bars.loc[bars["count"] == 5].drop(columns="count").reset_index()


def replay_session(
    session: date,
    minutes: pd.DataFrame,
    *,
    prior: pd.DataFrame,
    prior_ranges: list[float],
    config: TrendDayConfig,
    entry_delay: int,
) -> tuple[dict, dict] | None:
    """Find the day's first tradable candidate and walk it to the stop or 15:15.

    Returns ``(trade_fields, assessment_dict)`` or ``None`` when no candidate
    was tradable. Option pricing happens later, outside this pure spot replay.
    """

    atr, used = session_atr(prior_ranges, config)
    bars = completed_five_minute_bars(minutes)
    by_minute = minutes.set_index("minute")
    for index in range(len(bars)):
        # Bars outside the window can never be eligible; skipping them here
        # only saves time (the evaluator would reject them identically).
        if not config.window_start <= pd.Timestamp(bars.iloc[index]["timestamp"]).time() <= config.window_end:
            continue
        assessment = evaluate_trend_day_candidate(
            bars.iloc[: index + 1],
            prior_high=float(prior["high"].max()),
            prior_low=float(prior["low"].min()),
            prior_close=float(prior["close"].iloc[-1]),
            atr=atr,
            atr_sessions_used=used,
            config=config,
        )
        if not assessment.eligible:
            continue
        bar_start = pd.Timestamp(bars.iloc[index]["timestamp"])
        entry_minute = bar_start.hour * 60 + bar_start.minute + 4 + entry_delay
        if entry_minute > SQUARE_OFF_MINUTE or entry_minute not in by_minute.index:
            continue
        long = assessment.direction == LONG
        stop = float(assessment.stop or 0.0)
        entry_spot = float(by_minute.loc[entry_minute, "close"])
        if (long and entry_spot <= stop) or (not long and entry_spot >= stop):
            continue  # already through the stop: live blocks this entry
        exit_minute, exit_spot, reason = SQUARE_OFF_MINUTE, math.nan, "SQUARE_OFF"
        after = by_minute.loc[(by_minute.index > entry_minute) & (by_minute.index <= SQUARE_OFF_MINUTE)]
        for minute, row in after.iterrows():
            touched = row["low"] <= stop if long else row["high"] >= stop
            if touched:
                gapped = row["open"] <= stop if long else row["open"] >= stop
                exit_minute, exit_spot, reason = int(minute), float(row["open"] if gapped else stop), "VWAP_STOP"
                break
        if reason == "SQUARE_OFF":
            last = after.loc[after.index <= SQUARE_OFF_MINUTE]
            exit_spot = float(last["close"].iloc[-1]) if not last.empty else entry_spot
            exit_minute = int(last.index[-1]) if not last.empty else entry_minute
        spot_points = (exit_spot - entry_spot) if long else (entry_spot - exit_spot)
        fields = {
            "session": session,
            "direction": assessment.direction,
            "bar_start": assessment.bar_start,
            "entry_minute": entry_minute,
            "entry_spot": entry_spot,
            "stop": stop,
            "exit_minute": exit_minute,
            "exit_spot": exit_spot,
            "reason": reason,
            "spot_points": spot_points,
        }
        return fields, assessment.to_dict()
    return None


def _clock(session: date, minute: int) -> str:
    """Format a minute-of-day as an ISO timestamp for the trade table."""

    return datetime.combine(session, dt_time(minute // 60, minute % 60)).isoformat()


def price_trade(
    book: OptionBook | None,
    fields: dict,
    *,
    cost: float,
    exit_delay: int,
    lot_size: int,
    max_loss_rupees: float,
) -> dict:
    """Attach SELL-opposite and BUY-directional premium P&L to a spot trade."""

    long = fields["direction"] == LONG
    strike = round(fields["entry_spot"] / STRIKE_STEP) * STRIKE_STEP
    sold_right, bought_right = ("PE", "CE") if long else ("CE", "PE")
    priced = {"strike": float(strike), "sold_right": sold_right, "sold_entry": math.nan, "sold_exit": math.nan,
              "sell_points": math.nan, "buy_points": math.nan}
    if book is None:
        return priced
    start = fields["entry_minute"]
    end = min(fields["exit_minute"] + (exit_delay if fields["reason"] == "VWAP_STOP" else 0), SQUARE_OFF_MINUTE)
    sold = pd.Series(book.path(sold_right, strike, fields["session"], start, end)).ffill().to_numpy()
    bought = pd.Series(book.path(bought_right, strike, fields["session"], start, end)).ffill().to_numpy()
    if not (np.isfinite(sold[0]) and np.isfinite(sold[-1]) and np.isfinite(bought[0]) and np.isfinite(bought[-1])):
        return priced
    exit_at = len(sold) - 1
    if max_loss_rupees > 0:
        # The kill switch watches the open leg's rupee MTM; the first breach
        # closes the trade at that minute's mark.
        breach = np.nonzero((sold - sold[0]) * lot_size >= max_loss_rupees)[0]
        if breach.size:
            exit_at = int(breach[0])
            fields["reason"] = "MAX_LOSS"
            fields["exit_minute"] = start + exit_at
    priced.update(
        sold_entry=float(sold[0]),
        sold_exit=float(sold[exit_at]),
        sell_points=float(sold[0] - sold[exit_at] - cost),
        buy_points=float(bought[exit_at] - bought[0] - cost),
    )
    return priced


def replay(
    sessions: dict[date, pd.DataFrame],
    *,
    book: OptionBook | None,
    config: TrendDayConfig,
    cost: float,
    entry_delay: int,
    exit_delay: int,
    lot_size: int,
    max_loss_rupees: float,
    start: str,
    days_to_expiry: dict[date, float],
) -> list[Trade]:
    """Walk every session in order; at most one trade per session."""

    trades: list[Trade] = []
    ordered = list(sessions)
    ranges: list[float] = []
    for position, session in enumerate(ordered):
        minutes = sessions[session]
        if position > 0 and (not start or session >= pd.Timestamp(start).date()):
            found = replay_session(
                session, minutes, prior=sessions[ordered[position - 1]], prior_ranges=ranges,
                config=config, entry_delay=entry_delay,
            )
            if found is not None:
                fields, assessment = found
                priced = price_trade(book, fields, cost=cost, exit_delay=exit_delay, lot_size=lot_size,
                                     max_loss_rupees=max_loss_rupees)
                trades.append(
                    Trade(
                        session=session,
                        direction=fields["direction"],
                        bar_start=fields["bar_start"],
                        entry_time=_clock(session, fields["entry_minute"]),
                        entry_spot=fields["entry_spot"],
                        stop=fields["stop"],
                        exit_time=_clock(session, fields["exit_minute"]),
                        exit_spot=fields["exit_spot"],
                        reason=fields["reason"],
                        spot_points=fields["spot_points"],
                        confluence_score=int(assessment["confluence_score"]),
                        beyond_r1_s1=bool(assessment["beyond_r1_s1"]),
                        gap_in_direction=bool(assessment["gap_in_direction"]),
                        extended_from_vwap=bool(assessment["extended_from_vwap"]),
                        range_atr=float(assessment["range_atr"] or math.nan),
                        risk_points=float(assessment["risk_points"] or math.nan),
                        days_to_expiry=days_to_expiry.get(session, math.nan),
                        **priced,
                    )
                )
        ranges.append(float(minutes["high"].max() - minutes["low"].min()))
    return trades


def _stats(values: pd.Series) -> str:
    """One-line trades / avg / win rate / PF / max drawdown summary."""

    values = values.dropna()
    if values.empty:
        return "no trades"
    wins, losses = values[values > 0].sum(), -values[values < 0].sum()
    equity = values.cumsum()
    drawdown = (equity.cummax().clip(lower=0.0) - equity).max()
    factor = f"{wins / losses:.2f}" if losses > 0 else "n/a"
    return (
        f"n={len(values)} total={values.sum():.1f} avg={values.mean():.2f} "
        f"win={(values > 0).mean() * 100:.1f}% PF={factor} maxDD={drawdown:.1f}"
    )


def summarize(trades: list[Trade], *, priced: bool, lot_size: int) -> str:
    """Plain-text summary with the breakdowns the research used."""

    if not trades:
        return "Trades: 0"
    table = pd.DataFrame([asdict(trade) for trade in trades])
    metric = "sell_points" if priced else "spot_points"
    lines = [
        f"Spot points      : {_stats(table['spot_points'])}",
    ]
    if priced:
        unpriced = int(table["sell_points"].isna().sum())
        lines += [
            f"SELL opposite ATM: {_stats(table['sell_points'])}   (live expression)",
            f"BUY directional  : {_stats(table['buy_points'])}   (comparison only)",
            f"SELL per lot     : Rs {table['sell_points'].sum() * lot_size:,.0f} total at lot size {lot_size}",
            f"Unpriced trades  : {unpriced} (contract premium missing in the options files)",
        ]
    lines.append(f"\nBreakdowns use {metric}.")
    table["year"] = pd.to_datetime(table["session"]).dt.year
    half = table["session"].iloc[len(table) // 2]
    table["half"] = np.where(table["session"] < half, "first half", "second half")
    for label, key in (
        ("By year", "year"),
        ("By half (out-of-sample style split)", "half"),
        ("By direction", "direction"),
        ("By confluence score", "confluence_score"),
        ("By exit reason", "reason"),
        ("By days to expiry", "days_to_expiry"),
    ):
        lines.append(f"{label}:")
        for value, group in table.groupby(key, dropna=False):
            lines.append(f"  {value}: {_stats(group[metric])}")
    return "\n".join(lines)


def load_days_to_expiry(options_dir: Path) -> dict[date, float]:
    """Read trading days to the weekly expiry from the fetcher's calendar, if present."""

    calendar = options_dir / "_weekly_expiry_calendar.csv"
    if not calendar.exists():
        return {}
    frame = pd.read_csv(calendar)
    sessions = pd.to_datetime(frame["trade_date"]).dt.date
    return dict(zip(sessions, frame["trading_days_to_expiry"].astype(float), strict=True))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (also reachable as `python algo.py backtest --strategy cpr-ai-trend-day`)."""

    parser = argparse.ArgumentParser(description="CPR AI Trend-Day Rider deterministic-core backtest")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="NIFTY 1-minute OHLC CSV")
    parser.add_argument("--options-dir", default=str(DEFAULT_OPTIONS_DIR), help="Expired weekly options folder")
    parser.add_argument("--no-options", action="store_true", help="Report spot points only")
    parser.add_argument("--cost", type=float, default=2.0, help="Premium points per round trip")
    parser.add_argument("--entry-delay", type=int, default=2, help="Minutes after the bar's last minute")
    parser.add_argument("--exit-delay", type=int, default=1, help="Minutes between a stop touch and the fill")
    parser.add_argument("--lot-size", type=int, default=65, help="NIFTY lot size (65 since 2026)")
    parser.add_argument("--max-loss-rupees", type=float, default=0.0, help="Kill-switch level per lot; 0 = off")
    parser.add_argument("--start", default="", help="First session to trade, YYYY-MM-DD")
    parser.add_argument("--end", default="", help="Last session to trade, YYYY-MM-DD")
    args = parser.parse_args(argv)
    if args.entry_delay < 0 or args.exit_delay < 0 or args.lot_size <= 0 or args.cost < 0:
        parser.error("delays and cost must be non-negative and lot size positive")

    # The file prefix names the variant so runs never overwrite each other.
    variant = "_spot" if args.no_options else ""
    variant += f"_maxloss{int(args.max_loss_rupees)}" if args.max_loss_rupees > 0 else ""
    prefix = OUTPUT_DIR / f"nifty_cpr_ai_trend_day{variant}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(f"{prefix}_backtest.log"), logging.StreamHandler()],
    )
    logging.info("Trend-Day Rider backtest | args=%s | config=%s", vars(args), DEFAULT_TREND_DAY_CONFIG)
    sessions = load_sessions(Path(args.data), args.start, args.end)
    logging.info("Loaded %s sessions", len(sessions))
    options_dir = Path(args.options_dir)
    book = None
    if not args.no_options and options_dir.exists():
        logging.info("Loading option premiums from %s (this takes about a minute)", options_dir)
        book = OptionBook(options_dir, list(sessions))
    elif not args.no_options:
        logging.warning("Options folder %s not found; reporting spot points only.", options_dir)
    trades = replay(
        sessions,
        book=book,
        config=DEFAULT_TREND_DAY_CONFIG,
        cost=args.cost,
        entry_delay=args.entry_delay,
        exit_delay=args.exit_delay,
        lot_size=args.lot_size,
        max_loss_rupees=args.max_loss_rupees,
        start=args.start,
        days_to_expiry=load_days_to_expiry(options_dir),
    )
    table = pd.DataFrame([asdict(trade) for trade in trades])
    table.to_csv(f"{prefix}_trades.csv", index=False)
    if not table.empty:
        metric = "sell_points" if book is not None else "spot_points"
        daily = table.groupby("session")[metric].sum().to_frame("points")
        daily["cumulative_points"] = daily["points"].cumsum()
        daily.to_csv(f"{prefix}_daily.csv")
    summary = summarize(trades, priced=book is not None, lot_size=args.lot_size)
    Path(f"{prefix}_summary.txt").write_text(summary + "\n", encoding="utf-8")
    logging.info("Summary:\n%s", summary)
    logging.info("Outputs written with prefix %s", prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
