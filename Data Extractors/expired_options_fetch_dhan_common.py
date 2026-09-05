"""
Shared engine for downloading EXPIRED index-option history from DhanHQ.

What this gets you that the index fetchers do not
-------------------------------------------------
`index_1m_5y_data_fetch_dhan_common.py` downloads the NIFTY *index* itself, so
every options backtest built on it has to model option premium rather than read
it. This module downloads the real thing: minute bars of actual expired NIFTY
option contracts, with open interest, implied volatility and the spot level that
was printing at the time.

The endpoint is `POST /v2/charts/rollingoption`, reached through the pinned
`dhanhq` SDK's `expired_options_data()`. Five years of history are available.

The one thing to understand before reading further
--------------------------------------------------
This endpoint is **strike-relative, not contract-based**. You do not ask for
"the 24000 CE expiring on 12 March". There is no security ID for an expired
contract, and none exists in `Dependencies/all_instrument *.csv` either -- that
file only ever holds live contracts. Instead you ask for "the ATM call", or
"three strikes above ATM", and Dhan resolves that against the spot of the day.

Two consequences, and they drive most of the design here:

1. A rolling label is not a tradeable instrument, and the churn is far faster
   than "day to day" -- it is **minute to minute**. Measured against the live
   API on 2025-01-06: the `ATM` call series switched strike **69 times in that
   single session**, and one four-day `ATM` file held 13 distinct contracts.
   The rule is simply the nearest 50-point strike to spot, re-evaluated every
   bar (verified: `strike_price` equalled the nearest 50 to `spot` on 1500 of
   1500 rows, never more than 25 points away).

   This is why we always request the `strike` field and write it out as
   `strike_price`. Without it the data would be unusable: re-keying on
   `(expiry_date, strike_price, option_type)` is the only way to recover a
   series that corresponds to something you could actually have held.

2. The response carries no expiry date at all. We derive one -- see
   `expiry_calendar.py` -- from a published exchange rule plus the trading days
   the data itself reveals, and `--verify-expiries` then checks that derivation
   against the option prices.

And one hard limit worth stating loudly: the API offers ATM +/- 10 strikes,
which at NIFTY's 50-point spacing is a **+/-500 point band**. A contract whose
strike drifts further than that from spot simply stops appearing. A backtest
must not read that silence as "the option expired worthless".

Flow
----
1. Parse arguments and resolve a concrete date range.
2. Download the ATM call at a coarse interval to learn which days the market was
   open, and build the weekly expiry calendar from that.
3. For each (strike label, option type) series in turn, walk the date range in
   chunks, label each bar with its expiry, and append to that series' CSV.

Step 3 is deliberately series-outer rather than chunk-outer: it keeps exactly one
file open and writes it straight through, which matters a great deal more than it
looks like it should on a slow disk.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv

# Run as a script (which is how `algo.py` launches us), Python puts only THIS
# folder on sys.path. Both roots are needed: the repo root for `Dependencies.`
# imports, and this folder for the sibling extractor modules.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from expiry_calendar import (  # noqa: E402
    build_expiry_map,
    trading_days_to_expiry,
)
from index_1m_5y_data_fetch_dhan_common import (  # noqa: E402
    infer_epoch_unit,
    resolve_date_range,
    validate_single_epoch_unit,
)

if TYPE_CHECKING:
    # mypy_path lists Dependencies/, so these modules are known there by their
    # bare names. Importing them as `Dependencies.x` at type-check time would
    # give one file two module names and mypy refuses that outright.
    #
    # At RUNTIME the dotted form is the correct one, and not merely a style
    # choice: `index_1m_5y_data_fetch_dhan_common` (imported just above) already
    # loaded market_data_health as `Dependencies.market_data_health`. A bare
    # import here would build a SECOND module object with a second, unrelated
    # MarketDataValidationError -- and `except MarketDataValidationError` would
    # then quietly fail to catch the one that was raised.
    from broker_rate_limit import RollingWindowRateLimiter
    from market_data_health import MarketDataValidationError
else:
    from Dependencies.broker_rate_limit import RollingWindowRateLimiter
    from Dependencies.market_data_health import MarketDataValidationError

# Credentials come from Dependencies/.env, exactly as everywhere else.
load_dotenv(dotenv_path=_REPO_ROOT / "Dependencies" / ".env", override=False)

log = logging.getLogger(__name__)

# Everything the endpoint can return. We always ask for all of it: the marginal
# cost is bytes on an already-large response, and `strike`/`spot` in particular
# are what make the rolling labels usable at all.
REQUIRED_DATA_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "iv",
    "volume",
    "strike",
    "oi",
    "spot",
)

# Column order of every per-series CSV. `strike_label` and `option_type` are
# constant within a file and therefore redundant per row -- they are written
# anyway so that concatenating all the files and re-keying on the real strike is
# a one-liner, which is the whole point for a positional backtest.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
    "iv",
    "strike_price",
    "spot",
    "strike_label",
    "option_type",
    "expiry_date",
    "days_to_expiry",
)

CALENDAR_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "weekly_expiry_date",
    "days_to_expiry",
    "trading_days_to_expiry",
)

# Dhan documents a 30-day ceiling per call, and `toDate` is inclusive (measured
# -- see `chunk_ranges`), so a 29-day window spans 29 days and keeps a day of
# headroom under the cap.
MAX_CHUNK_DAYS = 30
DEFAULT_CHUNK_DAYS = 29

# The API's own strike ceiling for index options.
MAX_STRIKE_RANGE = 10

CALENDAR_FILENAME = "_weekly_expiry_calendar.csv"
MANIFEST_FILENAME = "_manifest.json"


@dataclass(frozen=True)
class ExpiredOptionsDefaults:
    """Per-underlying settings; everything else is identical between indices."""

    display_name: str
    security_id: int
    default_output_dir: str
    exchange_segment: str = "NSE_FNO"
    instrument_type: str = "OPTIDX"
    interval: int = 1
    expiry_flag: str = "WEEK"
    # 1, not 0. Dhan's annexure documents 0 = current expiry, but that table
    # describes /charts/historical. This endpoint rejects 0 outright with
    # "DH-905 expiryCode is required" -- it reads a zero as a missing field --
    # so the near expiry is 1, the next is 2, and so on. Confirmed against the
    # live API on 2026-09-05; see docs/adr/0015.
    expiry_code: int = 1
    strike_range: int = MAX_STRIKE_RANGE
    lookback: str = "5y"
    # Coarse interval for the trading-day pre-pass. Hourly bars are ~1/60th the
    # payload of minute bars and answer the only question being asked: was the
    # market open on this date?
    calendar_interval: int = 60


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(defaults: ExpiredOptionsDefaults, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build and parse the CLI. The access token is attached afterwards."""

    parser = argparse.ArgumentParser(
        description=f"Download expired {defaults.display_name} option history from DhanHQ.",
    )
    # As in the index fetchers: --client-id may be overridden, the ACCESS TOKEN
    # never can. A token typed on a command line lands in shell history and in
    # every process listing on the machine.
    parser.add_argument("--client-id", default=os.getenv("DHAN_CLIENT_CODE", ""))
    parser.add_argument(
        "--security-id",
        type=int,
        default=int(defaults.security_id),
        help=f"Underlying security ID for {defaults.display_name}.",
    )
    parser.add_argument("--exchange-segment", default=defaults.exchange_segment)
    parser.add_argument("--instrument-type", default=defaults.instrument_type)
    parser.add_argument("--interval", type=int, default=int(defaults.interval), choices=[1, 5, 15, 25, 60])
    parser.add_argument("--expiry-flag", default=defaults.expiry_flag, choices=["WEEK", "MONTH"])
    parser.add_argument(
        "--expiry-code",
        type=int,
        default=int(defaults.expiry_code),
        help="1 = near expiry, 2 = next, 3 = far. This endpoint rejects 0.",
    )
    parser.add_argument(
        "--strike-range",
        type=int,
        default=int(defaults.strike_range),
        help=f"Strikes each side of ATM (0-{MAX_STRIKE_RANGE}); 10 gives 21 labels.",
    )
    parser.add_argument(
        "--option-types",
        default="CALL,PUT",
        help="Comma-separated subset of CALL,PUT.",
    )
    parser.add_argument(
        "--lookback",
        default=defaults.lookback,
        choices=["1d", "7d", "15d", "1m", "3m", "6m", "1y", "5y"],
        help="Period to fetch when explicit start/end dates are not given.",
    )
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help=f"Days per API call (1-{MAX_CHUNK_DAYS}).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=2.0,
        help="Request pacing. Dhan allows ~5/s on data APIs; 2 leaves headroom.",
    )
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per chunk on transient failures.")
    parser.add_argument("--output-dir", default=defaults.default_output_dir)
    parser.add_argument(
        "--calendar-interval",
        type=int,
        default=int(defaults.calendar_interval),
        choices=[1, 5, 15, 25, 60],
        help="Interval for the cheap trading-day pre-pass.",
    )
    parser.add_argument(
        "--calendar-csv",
        default="",
        help="Skip calendar derivation and read trade_date,weekly_expiry_date from this CSV.",
    )
    parser.add_argument(
        "--verify-expiries",
        action="store_true",
        help="After downloading, check derived expiries against ATM straddle collapse.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing manifest and start over.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full call plan and exit without touching the network.",
    )
    args = parser.parse_args(argv)

    # Attached after parsing so it can never arrive from the command line.
    args.access_token = os.getenv("DHAN_ACCESS_TOKEN") or os.getenv("DHAN_TOKEN_ID") or ""
    return args


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return every problem with the parsed arguments, rather than just the first."""

    problems: list[str] = []
    if not 1 <= int(args.chunk_days) <= MAX_CHUNK_DAYS:
        problems.append(f"--chunk-days must be between 1 and {MAX_CHUNK_DAYS}, got {args.chunk_days}")
    if not 0 <= int(args.strike_range) <= MAX_STRIKE_RANGE:
        problems.append(f"--strike-range must be between 0 and {MAX_STRIKE_RANGE}, got {args.strike_range}")
    if int(args.expiry_code) < 1:
        problems.append(f"--expiry-code must be >= 1 (this endpoint reads 0 as absent), got {args.expiry_code}")
    if float(args.requests_per_second) <= 0:
        problems.append(f"--requests-per-second must be positive, got {args.requests_per_second}")
    for option_type in parse_option_types(args.option_types):
        if option_type not in {"CALL", "PUT"}:
            problems.append(f"--option-types accepts CALL and PUT, got {option_type!r}")
    if not parse_option_types(args.option_types):
        problems.append("--option-types must name at least one of CALL,PUT")
    return problems


def parse_option_types(raw: str) -> list[str]:
    """Split and normalise the --option-types value."""

    return [part.strip().upper() for part in str(raw).split(",") if part.strip()]


# ---------------------------------------------------------------------------
# The download plan
# ---------------------------------------------------------------------------


def strike_labels(strike_range: int) -> list[str]:
    """Return the relative strike labels, deepest put side first.

    ``strike_range=2`` gives ``['ATM-2', 'ATM-1', 'ATM', 'ATM+1', 'ATM+2']``.
    """
    width = int(strike_range)
    labels = []
    for offset in range(-width, width + 1):
        if offset == 0:
            labels.append("ATM")
        else:
            labels.append(f"ATM{offset:+d}")
    return labels


def series_plan(strike_range: int, option_types: Sequence[str]) -> list[tuple[str, str]]:
    """Every (strike label, option type) pair that will get its own CSV."""

    return [(label, option_type) for label in strike_labels(strike_range) for option_type in option_types]


def chunk_ranges(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    """
    Split ``[start, end]`` into API-sized windows, both ends INCLUSIVE.

    Dhan's documentation calls ``toDate`` non-inclusive. It is not. Measured
    against the live API on 2026-09-05, asking for 06-Jan to 08-Jan returns
    three sessions ending on the 8th, and 06-Jan to 10-Jan returns five ending
    on the 10th -- the end date is served.

    Getting this wrong is not cosmetic. Half-open windows made adjacent chunks
    overlap by a day, and the last chunk then carried bars dated ``to_date``
    that `validate_options_frame` rejected as out-of-window. It only survived a
    five-day smoke run because that window happened to end on a Saturday.

    So windows tile the range exactly: no overlap, no gap. The writer still
    de-duplicates on timestamp, which now costs nothing and would absorb the
    behaviour changing back.
    """
    if start > end:
        raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
    if chunk_days < 1:
        raise ValueError(f"chunk_days must be >= 1, got {chunk_days}")

    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=chunk_days - 1), end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def series_csv_path(output_dir: Path, defaults: ExpiredOptionsDefaults, args: argparse.Namespace,
                    strike_label: str, option_type: str) -> Path:
    """Where one series lives, e.g. ``nifty_1m_WEEK_ATM+3_CALL.csv``."""

    stem = (
        f"{defaults.display_name.lower()}_{args.interval}m_{args.expiry_flag}_"
        f"{strike_label}_{option_type}"
    )
    return output_dir / f"{stem}.csv"


# ---------------------------------------------------------------------------
# Talking to the API
# ---------------------------------------------------------------------------


def classify_failure(remarks: Any) -> str:
    """
    Decide whether a failed envelope is worth retrying.

    The Dhan SDK collapses two very different things into one shape. A *dict*
    `remarks` is a structured refusal that reached us from the server -- it
    means the request was seen and rejected. A *str* `remarks` is the SDK's
    bare `except Exception` handler stringifying a transport error, so we do
    not actually know whether the server saw anything.

    This is the same distinction `Dependencies/Dhan API/dhan_execution.py`
    exists to draw, applied here to reads instead of orders. Reads are safe to
    repeat, so the indeterminate case retries; a genuine refusal does not,
    unless it is a rate-limit complaint.
    """
    if isinstance(remarks, dict):
        blob = " ".join(str(value) for value in remarks.values()).lower()
        if "rate" in blob and "limit" in blob:
            return "retryable"
        if "too many" in blob:
            return "retryable"
        return "fatal"
    return "retryable"


def is_empty_window(remarks: Any) -> bool:
    """True when the server is saying 'nothing here', not 'you asked wrongly'."""

    text = " ".join(str(value) for value in remarks.values()) if isinstance(remarks, dict) else str(remarks)
    lowered = text.lower()
    return any(token in lowered for token in ("no data", "no records", "not found", "does not exist"))


def fetch_rolling_chunk(
    dhan: Any,
    *,
    args: argparse.Namespace,
    strike_label: str,
    option_type: str,
    interval: int,
    from_date: date,
    to_date: date,
    limiter: RollingWindowRateLimiter | None = None,
    sleeper: Any = time.sleep,
) -> dict[str, Any] | None:
    """
    Fetch one window for one series, retrying only what is safe to retry.

    Returns the inner ``{'ce': ..., 'pe': ...}`` mapping, or ``None`` when the
    window genuinely holds no data (a long market closure, or a strike that was
    never that far from spot).
    """
    attempts = max(1, int(args.max_retries) + 1)
    last_error: Any = None

    for attempt in range(attempts):
        if limiter is not None:
            limiter.acquire()
        response = dhan.expired_options_data(
            security_id=int(args.security_id),
            exchange_segment=str(args.exchange_segment),
            instrument_type=str(args.instrument_type),
            expiry_flag=str(args.expiry_flag),
            expiry_code=int(args.expiry_code),
            strike=strike_label,
            drv_option_type=option_type,
            required_data=list(REQUIRED_DATA_FIELDS),
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval=int(interval),
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected API response type: {type(response).__name__}")

        status = str(response.get("status", "")).strip().lower()
        if status == "success":
            return extract_payload(response.get("data"))

        remarks = response.get("remarks")
        last_error = remarks
        if is_empty_window(remarks):
            return None
        if classify_failure(remarks) == "fatal":
            raise RuntimeError(
                f"API refused {strike_label} {option_type} "
                f"{from_date.isoformat()} -> {to_date.isoformat()}: {remarks}"
            )
        if attempt < attempts - 1:
            # Exponential backoff. Reads are idempotent, so repeating one is
            # free apart from the rate-limit budget.
            delay = min(2.0**attempt, 30.0)
            log.warning(
                "transient failure on %s %s %s (attempt %d/%d), retrying in %.1fs: %s",
                strike_label, option_type, from_date.isoformat(), attempt + 1, attempts, delay, remarks,
            )
            sleeper(delay)

    raise RuntimeError(
        f"API failed {attempts}x for {strike_label} {option_type} "
        f"{from_date.isoformat()} -> {to_date.isoformat()}: {last_error}"
    )


def extract_payload(data: Any) -> dict[str, Any] | None:
    """
    Peel the response envelope down to the ``{'ce': ..., 'pe': ...}`` mapping.

    The SDK assigns the entire JSON body to its own ``data`` key, and the body
    itself has a ``data`` key, so the real payload usually sits one level deeper
    than it looks. Both shapes are accepted so a change on either side does not
    break the parse silently.
    """
    if not isinstance(data, dict):
        return None
    inner = data.get("data")
    if isinstance(inner, dict):
        data = inner
    if "ce" not in data and "pe" not in data:
        return None
    return data


# ---------------------------------------------------------------------------
# Turning parallel arrays into rows
# ---------------------------------------------------------------------------


def normalize_rolling_payload(payload: dict[str, Any] | None, option_type: str, strike_label: str) -> pd.DataFrame:
    """
    Convert one side of the response into a tidy frame.

    The endpoint answers in *column* form -- a separate equal-length array per
    field, aligned by position -- rather than a list of bars. Any length
    mismatch means the alignment assumption is broken, so it raises instead of
    silently zipping mismatched data together.
    """
    empty = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    if payload is None:
        return empty

    side = "ce" if option_type.upper() == "CALL" else "pe"
    block = payload.get(side)
    if not isinstance(block, dict):
        return empty

    stamps = block.get("timestamp")
    if not isinstance(stamps, list) or not stamps:
        return empty

    columns: dict[str, Any] = {"timestamp_raw": stamps}
    for field in REQUIRED_DATA_FIELDS:
        values = block.get(field)
        if not isinstance(values, list) or not values:
            columns[field] = [None] * len(stamps)
            continue
        if len(values) != len(stamps):
            raise MarketDataValidationError(
                f"{strike_label} {option_type}: field {field!r} has {len(values)} values "
                f"for {len(stamps)} timestamps"
            )
        columns[field] = values

    frame = pd.DataFrame(columns)
    for field in REQUIRED_DATA_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")

    validate_single_epoch_unit(frame["timestamp_raw"])
    unit = infer_epoch_unit(frame["timestamp_raw"])
    stamped = pd.to_datetime(frame["timestamp_raw"], unit=unit, errors="coerce", utc=True)
    # India market time, tz-naive -- the convention every other CSV in this repo
    # and every backtest already uses.
    frame["timestamp"] = stamped.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    frame = frame.rename(columns={"strike": "strike_price"})
    frame["strike_label"] = strike_label
    frame["option_type"] = option_type.upper()
    frame["expiry_date"] = pd.NA
    frame["days_to_expiry"] = pd.NA

    frame = frame.dropna(subset=["timestamp", "close"])
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    return frame[list(OUTPUT_COLUMNS)].reset_index(drop=True)


def validate_options_frame(frame: pd.DataFrame, from_date: date, to_date: date, label: str) -> None:
    """
    Sanity-check one chunk before it is written.

    Deliberately NOT `Dependencies/market_data_health.validate_ohlc_frame`: that
    one rejects any non-positive price, which is right for an index and wrong
    for a deep out-of-the-money option that can legitimately print at the 0.05
    tick floor. It also has no concept of open interest, IV, strike or spot.
    """
    if frame.empty:
        return

    stamps = frame["timestamp"]
    if stamps.isna().any():
        raise MarketDataValidationError(f"{label}: chunk contains unparseable timestamps")
    if not stamps.is_monotonic_increasing:
        raise MarketDataValidationError(f"{label}: chunk timestamps are not ascending")
    if stamps.duplicated().any():
        raise MarketDataValidationError(f"{label}: chunk contains duplicate timestamps")

    # Both ends inclusive, because that is what the API actually serves for
    # `toDate` regardless of what the documentation says. See `chunk_ranges`.
    outside = stamps[(stamps.dt.date < from_date) | (stamps.dt.date > to_date)]
    if not outside.empty:
        raise MarketDataValidationError(
            f"{label}: {len(outside)} candle(s) outside the requested window "
            f"{from_date.isoformat()} -> {to_date.isoformat()}, first {outside.iloc[0]}"
        )

    highs, lows = frame["high"], frame["low"]
    opens, closes = frame["open"], frame["close"]
    if (highs < lows).any():
        raise MarketDataValidationError(f"{label}: chunk has a candle with high < low")
    if (highs < opens.combine(closes, max)).any() or (lows > opens.combine(closes, min)).any():
        raise MarketDataValidationError(f"{label}: chunk has a candle whose body escapes its range")
    for column in ("open", "high", "low", "close"):
        if (frame[column] < 0).any():
            raise MarketDataValidationError(f"{label}: chunk has a negative {column}")
    for column in ("volume", "oi"):
        series = frame[column].dropna()
        if not series.empty and (series < 0).any():
            raise MarketDataValidationError(f"{label}: chunk has a negative {column}")


def label_with_expiry(frame: pd.DataFrame, expiry_map: dict[date, date]) -> pd.DataFrame:
    """Attach expiry_date and days_to_expiry, dropping bars we cannot label."""

    if frame.empty:
        return frame

    trade_dates = frame["timestamp"].dt.date
    expiries = trade_dates.map(expiry_map.get)
    labelled = frame.copy()
    labelled["expiry_date"] = [e.isoformat() if isinstance(e, date) else pd.NA for e in expiries]
    labelled["days_to_expiry"] = [
        (e - d).days if isinstance(e, date) else pd.NA for d, e in zip(trade_dates, expiries, strict=True)
    ]
    # A bar past the last derived expiry cannot be labelled honestly, and an
    # unlabelled bar is worse than no bar for a positional backtest.
    kept = labelled[labelled["expiry_date"].notna()].reset_index(drop=True)
    dropped = len(labelled) - len(kept)
    if dropped:
        # Debug, not warning: the systematic cause is the tail of the requested
        # range, and `warn_about_unlabelled_tail` says that once up front rather
        # than once per series per chunk.
        lost = sorted({d.isoformat() for d, e in zip(trade_dates, expiries, strict=True) if not isinstance(e, date)})
        log.debug("dropped %d bar(s) with no derivable expiry, on %s", dropped, ", ".join(lost))
    return kept


# ---------------------------------------------------------------------------
# Resumable writing
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    """Read the resume manifest, treating any damage as 'start over'."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically so a crash cannot leave it half-written."""

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_chunk(csv_path: Path, frame: pd.DataFrame, *, last_timestamp: str | None) -> tuple[int, int, str | None]:
    """
    Append one chunk to a series CSV.

    Returns ``(rows_written, file_size, newest_timestamp)``. Rows at or before
    ``last_timestamp`` are dropped, which is what makes a re-run idempotent and
    what absorbs a duplicated boundary bar between adjacent chunks.
    """
    if last_timestamp is not None and not frame.empty:
        frame = frame[frame["timestamp"] > pd.Timestamp(last_timestamp)]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        size = csv_path.stat().st_size if csv_path.exists() else 0
        return 0, size, last_timestamp

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, header=write_header, index=False)
        handle.flush()
        os.fsync(handle.fileno())

    newest = str(frame["timestamp"].iloc[-1])
    return len(frame), csv_path.stat().st_size, newest


def truncate_to(csv_path: Path, size: int) -> None:
    """
    Roll a series file back to its last manifest-recorded byte length.

    The manifest is written after the data, so a crash between the two leaves a
    file that is longer than the manifest believes. Those trailing bytes are a
    partially written chunk; discarding them is what lets the resume start from
    a known-good boundary.
    """
    if not csv_path.exists():
        return
    actual = csv_path.stat().st_size
    if actual > size:
        log.warning("trimming %s from %d to %d bytes (interrupted chunk)", csv_path.name, actual, size)
        with csv_path.open("r+b") as handle:
            handle.truncate(size)


# ---------------------------------------------------------------------------
# The trading-day / expiry calendar
# ---------------------------------------------------------------------------


def trading_days_from_frames(frames: Sequence[pd.DataFrame]) -> list[date]:
    """Every distinct session date present in the given frames."""

    days: set[date] = set()
    for frame in frames:
        if not frame.empty:
            days.update(frame["timestamp"].dt.date.tolist())
    return sorted(days)


def write_calendar_csv(path: Path, trading_days: Sequence[date], expiry_map: dict[date, date]) -> pd.DataFrame:
    """Write the derived calendar out so the labelling can be audited by eye."""

    rows = []
    for day in trading_days:
        expiry = expiry_map.get(day)
        if expiry is None:
            continue
        rows.append(
            {
                "trade_date": day.isoformat(),
                "weekly_expiry_date": expiry.isoformat(),
                "days_to_expiry": (expiry - day).days,
                "trading_days_to_expiry": trading_days_to_expiry(trading_days, day, expiry),
            }
        )
    frame = pd.DataFrame(rows, columns=list(CALENDAR_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def warn_about_unlabelled_tail(trading_days: Sequence[date], expiry_map: dict[date, date], end: date) -> str | None:
    """
    Say up front which sessions will be dropped for want of an expiry.

    Every bar is labelled with the expiry that was current on it, and the last
    derived expiry is the last one that actually happened inside the requested
    range. Sessions after it therefore cannot be labelled -- their expiry lies
    in the future, beyond the data. Those bars are discarded.

    That is the honest thing to do with them, but it is silent data loss unless
    somebody says so: a five-year backfill ending today quietly loses the whole
    current part-week. Returns the message (also logged), or None when the range
    ends neatly on an expiry.
    """
    unlabelled = [day for day in trading_days if day not in expiry_map]
    if not unlabelled:
        return None

    message = (
        f"{len(unlabelled)} session(s) at the end of the range have no expiry yet "
        f"({unlabelled[0].isoformat()} to {unlabelled[-1].isoformat()}) and will be SKIPPED. "
        f"Their weekly expiry falls after --end-date {end.isoformat()}. "
        f"To keep them, extend --end-date past that expiry and re-run."
    )
    log.warning("%s", message)
    return message


def read_calendar_csv(path: Path) -> dict[date, date]:
    """Load an operator-supplied calendar, bypassing derivation entirely."""

    frame = pd.read_csv(path)
    missing = {"trade_date", "weekly_expiry_date"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    # Parse column-wise rather than per row: `itertuples` erases the column
    # types, and pandas' own vectorised parse is both faster and the shape the
    # type checker can actually verify.
    trade_dates = pd.to_datetime(frame["trade_date"]).dt.date
    expiry_dates = pd.to_datetime(frame["weekly_expiry_date"]).dt.date
    return dict(zip(trade_dates, expiry_dates, strict=True))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# On expiry afternoon an at-the-money straddle is nearly all gone: both legs sit
# within half a strike of spot with minutes of life left. One session earlier it
# still carries a day of time value. The gap between those two states is wide
# enough to check a derived calendar against, without pretending to be exact.
EXPIRY_STRADDLE_CEILING = 40.0


def verify_expiries(
    call_frame: pd.DataFrame,
    put_frame: pd.DataFrame,
    expiry_dates: Sequence[date],
    ceiling: float = EXPIRY_STRADDLE_CEILING,
) -> list[str]:
    """
    Cross-check derived expiry dates against what the option prices actually did.

    Returns a list of human-readable complaints; empty means every derived
    expiry behaved like an expiry. This is a smell test, not a proof, and it is
    aimed squarely at the failure mode worth catching: a calendar that is off by
    a day, or that missed an exchange rule change.
    """
    problems: list[str] = []
    if call_frame.empty or put_frame.empty:
        return ["no ATM data available to verify expiries against"]

    def last_close_by_day(frame: pd.DataFrame) -> dict[date, float]:
        ordered = frame.sort_values("timestamp")
        days = ordered["timestamp"].dt.date
        closes = ordered["close"]
        return {day: float(close) for day, close in zip(days, closes, strict=True)}

    calls = last_close_by_day(call_frame)
    puts = last_close_by_day(put_frame)

    for expiry in expiry_dates:
        call_close, put_close = calls.get(expiry), puts.get(expiry)
        if call_close is None or put_close is None:
            problems.append(f"{expiry.isoformat()}: derived as an expiry but has no ATM close")
            continue
        straddle = call_close + put_close
        if straddle > ceiling:
            problems.append(
                f"{expiry.isoformat()}: ATM straddle closed at {straddle:.2f}, above the "
                f"{ceiling:.0f} ceiling -- this may not be an expiry day"
            )
    return problems


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def describe_plan(args: argparse.Namespace, start: date, end: date) -> str:
    """Render the whole call plan as text, for --dry-run and for the log."""

    option_types = parse_option_types(args.option_types)
    series = series_plan(args.strike_range, option_types)
    windows = chunk_ranges(start, end, args.chunk_days)
    calendar_calls = 0 if args.calendar_csv else len(windows) * 2
    lines = [
        f"range          : {start.isoformat()} -> {end.isoformat()}",
        f"interval       : {args.interval} minute",
        f"expiry         : {args.expiry_flag}, expiryCode={args.expiry_code}",
        f"strikes        : {', '.join(strike_labels(args.strike_range))}",
        f"option types   : {', '.join(option_types)}",
        f"series         : {len(series)} files",
        f"chunks/series  : {len(windows)} windows of up to {args.chunk_days} days",
        f"calendar pass  : {calendar_calls} calls at {args.calendar_interval} minute",
        f"TOTAL API CALLS: {len(series) * len(windows) + calendar_calls}",
        f"pacing         : {args.requests_per_second}/s",
        f"output         : {args.output_dir}",
    ]
    return "\n".join(f"  {line}" for line in lines)


def build_client(args: argparse.Namespace) -> Any:
    """Construct the DhanHQ client, failing clearly when credentials are absent."""

    if not args.client_id or not args.access_token:
        raise ValueError(
            "Missing credentials. Set DHAN_CLIENT_CODE and DHAN_ACCESS_TOKEN in "
            "Dependencies/.env (run `python algo.py setup-token` to refresh the "
            "token). Only --client-id may be overridden on the command line; "
            "the token is environment-only."
        )
    return dhanhq(DhanContext(args.client_id, args.access_token))


def build_expiry_calendar(
    dhan: Any,
    args: argparse.Namespace,
    start: date,
    end: date,
    limiter: RollingWindowRateLimiter | None,
) -> tuple[list[date], dict[date, date], pd.DataFrame, pd.DataFrame]:
    """
    Learn the trading days from a cheap pre-pass, then derive the expiry map.

    Hourly ATM bars are a fraction of the payload of minute bars and answer the
    only question being asked here: was the market open that day? Deriving the
    calendar from the data itself means it can never disagree with the bars it
    is about to label.
    """
    log.info("calendar pre-pass: ATM CALL/PUT at %d-minute resolution", args.calendar_interval)
    collected: dict[str, pd.DataFrame] = {}
    for option_type in ("CALL", "PUT"):
        parts = []
        for from_date, to_date in chunk_ranges(start, end, args.chunk_days):
            payload = fetch_rolling_chunk(
                dhan,
                args=args,
                strike_label="ATM",
                option_type=option_type,
                interval=args.calendar_interval,
                from_date=from_date,
                to_date=to_date,
                limiter=limiter,
            )
            frame = normalize_rolling_payload(payload, option_type, "ATM")
            if not frame.empty:
                parts.append(frame)
        collected[option_type] = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        )

    call_frame, put_frame = collected["CALL"], collected["PUT"]
    trading_days = trading_days_from_frames([call_frame, put_frame])
    if not trading_days:
        raise RuntimeError(
            "calendar pre-pass returned no bars at all, so trading days cannot be derived. "
            "Check the date range, and that the Data API subscription is active."
        )
    expiry_map = build_expiry_map(trading_days)
    log.info(
        "derived %d trading days and %d weekly expiries (%s -> %s)",
        len(trading_days),
        len(set(expiry_map.values())),
        trading_days[0].isoformat(),
        trading_days[-1].isoformat(),
    )
    return trading_days, expiry_map, call_frame, put_frame


def download_series(
    dhan: Any,
    args: argparse.Namespace,
    defaults: ExpiredOptionsDefaults,
    output_dir: Path,
    strike_label: str,
    option_type: str,
    windows: Sequence[tuple[date, date]],
    expiry_map: dict[date, date],
    manifest: dict[str, Any],
    manifest_path: Path,
    limiter: RollingWindowRateLimiter | None,
) -> int:
    """Download one series end to end, resuming from the manifest where it can."""

    key = f"{strike_label}_{option_type}"
    csv_path = series_csv_path(output_dir, defaults, args, strike_label, option_type)
    state: dict[str, Any] = {} if args.no_resume else manifest.get(key, {})

    if state:
        truncate_to(csv_path, int(state.get("bytes", 0)))
    elif args.no_resume and csv_path.exists():
        csv_path.unlink()

    resume_after = state.get("last_to_date")
    last_timestamp = state.get("last_timestamp")
    rows_total = int(state.get("rows", 0))

    for from_date, to_date in windows:
        if resume_after and to_date <= date.fromisoformat(resume_after):
            continue

        payload = fetch_rolling_chunk(
            dhan,
            args=args,
            strike_label=strike_label,
            option_type=option_type,
            interval=args.interval,
            from_date=from_date,
            to_date=to_date,
            limiter=limiter,
        )
        frame = normalize_rolling_payload(payload, option_type, strike_label)
        validate_options_frame(frame, from_date, to_date, key)
        frame = label_with_expiry(frame, expiry_map)

        written, size, newest = append_chunk(csv_path, frame, last_timestamp=last_timestamp)
        rows_total += written
        if newest is not None:
            last_timestamp = newest

        manifest[key] = {
            "last_to_date": to_date.isoformat(),
            "last_timestamp": last_timestamp,
            "bytes": size,
            "rows": rows_total,
        }
        save_manifest(manifest_path, manifest)

    log.info("%-14s -> %-48s %8d rows", key, csv_path.name, rows_total)
    return rows_total


def run_expired_options_fetcher(defaults: ExpiredOptionsDefaults, argv: Sequence[str] | None = None) -> None:
    """Entry point that every per-underlying wrapper calls."""

    args = parse_args(defaults, argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    install_log_redaction()

    problems = validate_args(args)
    if problems:
        raise ValueError("Invalid arguments:\n  - " + "\n  - ".join(problems))

    start, end = resolve_date_range(args)
    plan = describe_plan(args, start, end)
    if args.dry_run:
        print(f"[dry-run] {defaults.display_name} expired options plan:\n{plan}")
        return
    log.info("%s expired options download:\n%s", defaults.display_name, plan)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest: dict[str, Any] = {} if args.no_resume else load_manifest(manifest_path)

    dhan = build_client(args)
    limiter = RollingWindowRateLimiter(
        per_second=max(1, int(args.requests_per_second)),
        per_minute=max(1, int(args.requests_per_second * 60)),
        # Generous on purpose. A downloader wants to WAIT for a slot; a live
        # order would rather fail fast than send something stale.
        max_wait_seconds=120.0,
        label="Dhan expired-options",
    )

    if args.calendar_csv:
        expiry_map = read_calendar_csv(Path(args.calendar_csv))
        trading_days = sorted(expiry_map)
        call_frame = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        put_frame = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        log.info("using supplied calendar %s (%d trading days)", args.calendar_csv, len(trading_days))
    else:
        trading_days, expiry_map, call_frame, put_frame = build_expiry_calendar(dhan, args, start, end, limiter)
        write_calendar_csv(output_dir / CALENDAR_FILENAME, trading_days, expiry_map)
        log.info("wrote %s", output_dir / CALENDAR_FILENAME)
        warn_about_unlabelled_tail(trading_days, expiry_map, end)

    windows = chunk_ranges(start, end, args.chunk_days)
    option_types = parse_option_types(args.option_types)
    grand_total = 0
    for strike_label, option_type in series_plan(args.strike_range, option_types):
        grand_total += download_series(
            dhan,
            args,
            defaults,
            output_dir,
            strike_label,
            option_type,
            windows,
            expiry_map,
            manifest,
            manifest_path,
            limiter,
        )

    log.info("done: %d rows across %d series in %s", grand_total, len(manifest), output_dir)

    if args.verify_expiries:
        complaints = verify_expiries(call_frame, put_frame, sorted(set(expiry_map.values())))
        if complaints:
            log.warning("expiry verification raised %d concern(s):", len(complaints))
            for complaint in complaints:
                log.warning("  %s", complaint)
        else:
            log.info("expiry verification: every derived expiry behaved like one")


def install_log_redaction() -> None:
    """
    Scrub secrets from logs, best-effort.

    The repo-wide guard lives in Dependencies/. If it cannot be imported (a bare
    checkout, say) the download is still safe to run -- this module never logs a
    token -- so a missing helper must not stop the job.

    Imported dynamically rather than with a normal statement for the same
    module-identity reason as the block at the top of this file: naming
    `Dependencies.secret_redaction` in an import statement would give mypy a
    second name for a file it already knows as `secret_redaction`.
    """
    try:
        redaction = importlib.import_module("Dependencies.secret_redaction")
    except Exception:  # noqa: BLE001 - redaction is a hardening layer, not a dependency
        log.debug("secret redaction helper unavailable; continuing without it")
        return
    redaction.install_redaction_filter(logging.getLogger(), redaction.environment_secrets(os.environ))
