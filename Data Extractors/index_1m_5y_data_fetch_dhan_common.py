"""
Shared helper for index data download scripts.

Why this file exists:
- Your project already had one NIFTY-only script.
- You now wanted BankNifty and FINNIFTY versions with the same behavior.
- Instead of copy-pasting the full fetch logic many times, this file keeps the
  common flow in one place and lets the thin wrapper scripts only define
  index-specific defaults such as security ID and output CSV name.

High-level flow used by the wrapper scripts:
1. Read command-line arguments.
2. Resolve a concrete date range from either explicit dates or a lookback.
3. Break the full range into smaller chunks because Dhan minute API does not
   allow a very large range in a single request.
4. Download each chunk, normalize the broker response into a clean OHLC table,
   and append it to the CSV as it arrives.
5. Record the progress in a manifest beside the CSV, so an interrupted run
   resumes from the last completed chunk instead of downloading five years
   again.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from dhanhq import DhanContext, dhanhq
from dotenv import load_dotenv

# `python algo.py fetch-data ...` launches this file as a SCRIPT, so Python puts
# the script's own folder on sys.path -- not the repository root, and not the
# working directory algo.py sets. Without this insert the `Dependencies.` import
# below raises ModuleNotFoundError. The pytest suite never saw it because pytest
# puts the rootdir on sys.path itself.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    # mypy_path includes Dependencies/, where this module is known by its bare
    # name. Runtime entry points execute from the repository root instead.
    from market_data_health import (
        MARKET_SESSION_END,
        MARKET_SESSION_START,
        MarketDataValidationError,
        validate_ohlc_frame,
    )
else:
    from Dependencies.market_data_health import (
        MARKET_SESSION_END,
        MARKET_SESSION_START,
        MarketDataValidationError,
        validate_ohlc_frame,
    )

# Credentials live in Dependencies/.env like everywhere else in this repo.
# Without this the extractors could only see variables already exported in the
# shell, which is why they silently refused to run. `override=False` keeps a
# value the operator exported deliberately ahead of the file's.
load_dotenv(dotenv_path=_REPO_ROOT / "Dependencies" / ".env", override=False)


@dataclass(frozen=True)
class IndexFetchDefaults:
    """
    Container for the settings that change from one index script to another.

    Example:
    - NIFTY uses one security ID and one default output path
    - BANKNIFTY uses a different security ID and a different CSV path

    Keeping those values inside one small dataclass makes the shared fetch logic
    reusable without making the wrapper scripts complicated.
    """

    display_name: str
    security_id: str
    default_output: str
    exchange_segment: str = "IDX_I"
    instrument_type: str = "INDEX"
    interval: int = 1
    lookback: str = "5y"
    # SECURITY: never hardcode credentials here. The client id resolves CLI
    # flag -> environment variable (DHAN_CLIENT_CODE) -> this blank. The access
    # token is a SECRET and resolves environment variable (DHAN_TOKEN_ID, e.g.
    # from Dependencies/.env) -> this blank ONLY -- there is deliberately no
    # CLI flag for it (MAT-108): a token typed on the command line lands in
    # shell history and process listings. A real client id + access token used
    # to live in these defaults; they were removed (and remain in old git
    # history, so treat that token as burned).
    default_client_id: str = ""
    default_access_token: str = ""


def parse_args(defaults: IndexFetchDefaults):
    """
    Read user inputs from the command line.

    The wrapper script passes its own defaults into this function, so the same
    parser can behave like a BANKNIFTY fetcher or a FINNIFTY fetcher depending
    on which wrapper called it.
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Fetch 1-minute {defaults.display_name} OHLC data for a selectable "
            "recent period (1d/7d/15d/1m/3m/6m/1y/5y) using Dhan API and save "
            "it to CSV."
        )
    )

    # Credentials:
    # - the CLIENT ID is an account identifier (not a secret), so it may come
    #   from the CLI flag, then the environment, then the wrapper default.
    # - the ACCESS TOKEN is a secret and is read from the environment ONLY
    #   (DHAN_ACCESS_TOKEN, loaded from Dependencies/.env above). There is
    #   deliberately no --access-token flag: a token typed on the command
    #   line would land in shell history and process listings.
    parser.add_argument(
        "--client-id",
        default=os.getenv("DHAN_CLIENT_CODE", defaults.default_client_id),
    )

    parser.add_argument(
        "--security-id",
        default=str(defaults.security_id),
        help=f"{defaults.display_name} index security ID",
    )
    parser.add_argument("--exchange-segment", default=defaults.exchange_segment)
    parser.add_argument("--instrument-type", default=defaults.instrument_type)
    parser.add_argument(
        "--interval",
        type=int,
        default=int(defaults.interval),
        choices=[1, 5, 15, 25, 60],
    )
    parser.add_argument(
        "--lookback",
        default=defaults.lookback,
        choices=["1d", "7d", "15d", "1m", "3m", "6m", "1y", "5y"],
        help="Historical period to fetch when start/end dates are not provided.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=90,
        help="Maximum days per API call (keep <= 90).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.35,
        help="Pause between chunk requests to avoid aggressive request bursts.",
    )
    parser.add_argument(
        "--output",
        default=defaults.default_output,
        help="CSV path where the final OHLC data should be saved.",
    )
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing manifest and download the whole range again.",
    )
    args = parser.parse_args()
    # Attach the token AFTER parsing so it can never be supplied (or leaked)
    # through the command line; downstream code keeps reading args.access_token.
    # DHAN_ACCESS_TOKEN is the key the rest of the repo uses and the one
    # `algo.py setup-token` writes. DHAN_TOKEN_ID is accepted only so an
    # operator who had exported the old name keeps working.
    args.access_token = (
        os.getenv("DHAN_ACCESS_TOKEN")
        or os.getenv("DHAN_TOKEN_ID")
        or defaults.default_access_token
    )
    return args


def resolve_date_range(args):
    """
    Turn the user's date inputs into actual start and end dates.

    There are two supported styles:
    1. Exact dates:
       `--start-date 2025-01-01 --end-date 2025-12-31`
    2. Relative lookback:
       `--lookback 1y`

    If explicit dates are given, they win.
    Otherwise we count backward from today using the selected lookback period.
    """
    if args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        end_dt = datetime.now().date()
        lookback_days_map = {
            "1d": 1,
            "7d": 7,
            "15d": 15,
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "5y": 365 * 5,
        }
        lookback_days = lookback_days_map[args.lookback]
        start_dt = end_dt - timedelta(days=lookback_days)

    if start_dt > end_dt:
        raise ValueError("start-date must be <= end-date")

    return start_dt, end_dt


def infer_epoch_unit(values: pd.Series) -> Literal["s", "ms", "us"]:
    """
    Guess the timestamp unit from the size of the numbers.

    Different APIs sometimes send timestamps as:
    - seconds
    - milliseconds
    - microseconds

    We inspect the magnitude and choose the most likely unit so the timestamp
    conversion step can work correctly.
    """
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return "ms"

    max_value = float(nums.max())
    if max_value > 1e14:
        return "us"
    if max_value > 1e11:
        return "ms"
    return "s"


def validate_single_epoch_unit(values: pd.Series) -> None:
    """Reject a chunk whose numeric timestamps mix epoch units."""

    numbers = pd.to_numeric(values, errors="coerce").dropna().abs()
    if numbers.empty:
        return
    units = {
        "us" if value > 1e14 else "ms" if value > 1e11 else "s"
        for value in numbers
    }
    if len(units) != 1:
        raise MarketDataValidationError(
            f"Dhan chunk mixes epoch units: {', '.join(sorted(units))}"
        )


def normalize_response_data(data, *, instrument_type: str = "") -> pd.DataFrame:
    """
    Convert the raw broker payload into one clean OHLC DataFrame.

    Why this function is useful:
    - Broker responses can arrive in slightly different shapes.
    - Column names can differ in capitalization.
    - Timestamps may be strings or numeric epoch values.

    This function standardizes all of that into:
    timestamp, open, high, low, close, volume
    """
    if data is None:
        return pd.DataFrame()

    try:
        if isinstance(data, (list, dict)):
            df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    normalized = {str(col).strip().lower(): col for col in df.columns}

    ts_col = None
    for candidate in ["start_time", "starttime", "timestamp", "time", "datetime", "date"]:
        if candidate in normalized:
            ts_col = normalized[candidate]
            break

    o_col = normalized.get("open")
    h_col = normalized.get("high")
    l_col = normalized.get("low")
    c_col = normalized.get("close")
    v_col = normalized.get("volume")

    required = [ts_col, o_col, h_col, l_col, c_col]
    if any(col is None for col in required):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "timestamp_raw": df[ts_col],
            "open": pd.to_numeric(df[o_col], errors="coerce"),
            "high": pd.to_numeric(df[h_col], errors="coerce"),
            "low": pd.to_numeric(df[l_col], errors="coerce"),
            "close": pd.to_numeric(df[c_col], errors="coerce"),
            "volume": pd.to_numeric(df[v_col], errors="coerce") if v_col else 0,
        }
    )

    if pd.api.types.is_numeric_dtype(out["timestamp_raw"]):
        validate_single_epoch_unit(out["timestamp_raw"])
        unit = infer_epoch_unit(out["timestamp_raw"])
        ts = pd.to_datetime(out["timestamp_raw"], unit=unit, errors="coerce", utc=True)
    else:
        maybe_num = pd.to_numeric(out["timestamp_raw"], errors="coerce")
        if maybe_num.notna().sum() >= max(1, len(out) // 2):
            validate_single_epoch_unit(maybe_num)
            unit = infer_epoch_unit(maybe_num)
            ts = pd.to_datetime(maybe_num, unit=unit, errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(out["timestamp_raw"], errors="coerce", utc=True)

    # Dhan timestamps are normalized into India market time because that is the
    # timezone your backtest data uses across the project.
    out["timestamp"] = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    out = out.drop(columns=["timestamp_raw"])
    # A timestamp that would not parse is refused HERE, before anything filters
    # it away. `errors="coerce"` turns it into NaT, and NaT compares False
    # against a `datetime.time`, so the session clip below would silently drop
    # the row instead: the chunk would come back quietly short, be appended, and
    # advance the resume point past a gap nobody was told about.
    if out["timestamp"].isna().any():
        raise MarketDataValidationError(
            f"Dhan chunk has {int(out['timestamp'].isna().sum())} unparseable timestamp(s)"
        )
    # Drop the non-session rows BEFORE validating: they are the bulk of what an
    # older window returns, and validating them first would either pass junk
    # through (they are minute-aligned) or fail the chunk on rows nobody wants.
    out = clip_to_session(out)
    out = drop_impossible_candles(out)
    if out.empty:
        return out
    out = validate_ohlc_frame(out)

    # NaN belongs in `invalid`, not filled away before the test. An absent
    # volume is exactly as wrong as a negative one for an instrument where
    # volume MEANS something, and filling it first quietly exempted EVERY
    # instrument from a guard that is only meant to be relaxed for indices.
    volume = pd.to_numeric(out["volume"], errors="coerce")
    invalid = ~volume.map(math.isfinite) | (volume < 0)
    if invalid.any():
        if str(instrument_type).strip().upper() == "INDEX":
            # An index has no traded volume. Dhan returns zeros for the whole of
            # 2021, small negative counters (-1, -2, -3) on some 2022 bars --
            # 221 of them in one 90-day window -- and sometimes nothing at all,
            # and every backtest loader in this repo forces Volume to 0
            # regardless. All three are the same non-answer about a field the
            # instrument does not have.
            #
            # The PRICES on those bars are sound: once weekends and out-of-hours
            # rows are clipped, every session in that window holds exactly 375
            # bars. Dropping 221 real price bars to protect a field nobody reads
            # would lose information for nothing, so the field is zeroed and the
            # count is reported.
            print(f"Zeroing {int(invalid.sum())} invalid index volume value(s)")
            volume = volume.mask(invalid, 0.0)
        else:
            # Anywhere volume is a real quantity, a negative one is corruption.
            raise MarketDataValidationError("Dhan chunk contains invalid volume")
    out["volume"] = volume

    return out[["timestamp", "open", "high", "low", "close", "volume"]]


#: A chunk with a larger share of self-contradicting candles than this is not
#: noisy, it is the wrong data -- those keep failing instead of being trimmed.
MAX_DROPPED_CANDLE_FRACTION = 0.001


def drop_impossible_candles(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop candles that contradict themselves; refuse a chunk that is mostly bad.

    Dhan's older index history carries the occasional print whose high sits
    below its own open. Measured while backfilling: exactly ONE row in 23,380
    across 2021-12-26..2022-03-25, at 2022-03-25 09:15, open 17289.00 against a
    high of 17287.10.

    Such a row is provably wrong -- no reading of a candle makes its high lower
    than the open it contains -- and failing a five-year backfill on one of them
    is the wrong trade. Isolated ones are therefore dropped and NAMED, never
    silently. Many of them is a different thing: it says the response is not the
    series we asked for, and that still fails.

    This is the historical backfill path only. `validate_ohlc_frame` is untouched
    and still refuses, on the live feed, everything it refused before -- including
    the rows dropped here, which is why they are removed before it runs rather
    than by relaxing it.
    """

    if frame.empty:
        return frame

    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    impossible = (
        (frame["high"] < body_high)
        | (frame["low"] > body_low)
        | (frame["high"] < frame["low"])
    )

    dropped = int(impossible.sum())
    if not dropped:
        return frame

    share = dropped / len(frame)
    if share > MAX_DROPPED_CANDLE_FRACTION:
        raise MarketDataValidationError(
            f"Dhan chunk has {dropped} self-contradicting candles of {len(frame)} "
            f"({share:.2%}) -- too many to be stray prints, refusing the chunk"
        )

    names = ", ".join(str(value) for value in frame.loc[impossible, "timestamp"].head(5))
    print(f"Dropping {dropped} self-contradicting candle(s): {names}")
    return frame.loc[~impossible].reset_index(drop=True)


def clip_to_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the bars that belong to a trading session.

    Dhan's index history carries rows that are not session data, and they differ
    by era:

    - windows before roughly mid-2022 come back with ~650 bars a day running out
      to 17:59, against the 375 a real session has;
    - the same era carries rows on WEEKENDS, whose prices are visibly not the
      index: measured on Saturday 2022-04-09, the close jumps 18396 -> 18584 ->
      18371 -> 18842 inside one hour;
    - the CURRENT day carries a synthetic "now" bar stamped at the wall clock
      with flat OHLC and no volume.

    Both are minute-ALIGNED, so `validate_ohlc_frame` accepts them -- which makes
    them more dangerous than the malformed kind, not less: they would land in the
    CSV and drag a day's high, low and close with them. Clipping to the session
    is what makes a 2021 chunk and a 2026 chunk mean the same thing.

    The window is `market_data_health`'s own 09:15-15:30, so the extractor and the
    runner agree on what a session is.
    """

    if frame.empty:
        return frame
    clock = frame["timestamp"].dt.time
    weekday = frame["timestamp"].dt.dayofweek < 5
    inside = weekday & (clock >= MARKET_SESSION_START) & (clock <= MARKET_SESSION_END)
    return frame.loc[inside].reset_index(drop=True)


def fetch_chunk(
    dhan: dhanhq,
    security_id: str,
    exchange_segment: str,
    instrument_type: str,
    interval: int,
    chunk_start: date,
    chunk_end: date,
) -> pd.DataFrame:
    """
    Download exactly one chunk of candles from the API.

    Keeping this as a separate function helps in two ways:
    - the main loop stays easy to read
    - error handling for one request is kept in one place
    """
    resp = dhan.intraday_minute_data(
        security_id=str(security_id),
        exchange_segment=exchange_segment,
        instrument_type=instrument_type,
        from_date=chunk_start.strftime("%Y-%m-%d"),
        to_date=chunk_end.strftime("%Y-%m-%d"),
        interval=interval,
    )

    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected API response type: {type(resp).__name__}")

    status = str(resp.get("status", "")).strip().lower()
    if status and status != "success":
        remarks = resp.get("remarks") or resp.get("message") or resp.get("data")
        remarks_text = str(remarks).strip().lower()

        # Empty windows can happen on holidays or on date ranges where the API
        # simply has no candles. Those cases should not crash the whole script.
        if any(
            token in remarks_text
            for token in ["no data", "no records", "not found", "does not exist"]
        ):
            return pd.DataFrame()

        raise RuntimeError(
            f"API failed for {chunk_start} -> {chunk_end}: status={status}, details={remarks}"
        )

    normalized = normalize_response_data(resp.get("data"), instrument_type=instrument_type)
    if normalized.empty:
        return normalized

    dates = pd.DatetimeIndex(normalized["timestamp"]).date
    if any(timestamp_date < chunk_start or timestamp_date > chunk_end for timestamp_date in dates):
        raise MarketDataValidationError(
            f"Dhan returned a candle outside requested chunk {chunk_start} -> {chunk_end}"
        )
    return normalized


def normalize_exchange_segment(segment: str) -> str:
    """
    Convert friendly segment labels into the exact wire value Dhan expects.

    This makes the script more forgiving if you later pass a friendlier alias
    such as `NSE_IDX` instead of `IDX_I`.
    """
    value = str(segment or "").strip().upper()
    mapping = {
        "NSE_IDX": "IDX_I",
        "IDX_I": "IDX_I",
        "NSE_EQ": "NSE_EQ",
        "NSE_FNO": "NSE_FNO",
        "BSE_EQ": "BSE_EQ",
        "BSE_FNO": "BSE_FNO",
        "MCX_COMM": "MCX_COMM",
    }
    return mapping.get(value, segment)


def fetch_1m_history(
    args,
    defaults: IndexFetchDefaults,
    *,
    on_chunk: Callable[[pd.DataFrame, date], None] | None = None,
    resume_after: date | None = None,
) -> pd.DataFrame:
    """
    Download the full requested date range in many smaller pieces.

    Why chunking matters:
    - Dhan minute API has a practical limit on how much history can be fetched
      in one request.
    - So we walk from start date to end date chunk by chunk, save each chunk,
      and then merge them into one final DataFrame.

    `on_chunk` is how the resumable path takes delivery. It is called with every
    chunk AND that chunk's end date -- EMPTY ones included, so a run of holidays
    still moves the resume point forward instead of being re-requested next
    time. When it is supplied, chunks are handed over as they arrive and never
    accumulated, so memory stays flat across five years and the returned frame
    is empty. With no `on_chunk` the original behaviour is unchanged: collect
    everything and return it in one frame.

    `resume_after` skips any chunk ending on or before it, because those rows
    are already on disk.
    """
    start_dt, end_dt = resolve_date_range(args)
    # dhanhq >= 2.1 (we pin 2.2.0) takes a DhanContext(client_id, access_token),
    # not two positional args -- the old `dhanhq(client_id, access_token)` form
    # raises TypeError under the pinned SDK. Build the context explicitly, the
    # same way the master runner's DhanBrokerClient does.
    dhan_context = DhanContext(args.client_id, args.access_token)
    dhan = dhanhq(dhan_context)
    exchange_segment = normalize_exchange_segment(args.exchange_segment)

    all_chunks = []
    total_rows = 0
    cursor = start_dt

    print(
        f"Fetching {defaults.display_name} data from {start_dt} to {end_dt} "
        f"(interval={args.interval}m, chunk_days={args.chunk_days})"
    )

    while cursor <= end_dt:
        chunk_end = min(cursor + timedelta(days=args.chunk_days - 1), end_dt)

        if resume_after is not None and chunk_end <= resume_after:
            print(f"Already fetched, skipping: {cursor} -> {chunk_end}")
            cursor = chunk_end + timedelta(days=1)
            continue

        print(f"Requesting chunk: {cursor} -> {chunk_end}")

        chunk_df = fetch_chunk(
            dhan=dhan,
            security_id=args.security_id,
            exchange_segment=exchange_segment,
            instrument_type=args.instrument_type,
            interval=args.interval,
            chunk_start=cursor,
            chunk_end=chunk_end,
        )

        row_count = len(chunk_df)
        total_rows += row_count
        print(f"Chunk rows: {row_count} | Running total: {total_rows}")

        if on_chunk is not None:
            on_chunk(chunk_df, chunk_end)
        elif not chunk_df.empty:
            all_chunks.append(chunk_df)

        cursor = chunk_end + timedelta(days=1)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if not all_chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Final cleanup after all chunks are fetched:
    # - drop bad rows
    # - sort by time
    # - remove duplicate timestamps
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.dropna()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def atomic_write_csv(frame: pd.DataFrame, output: str | os.PathLike[str]) -> None:
    """Replace a CSV only after writing its complete sibling temporary file."""

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Resumable writing
# ---------------------------------------------------------------------------
#
# A five-year pull is ~21 requests over ~10 minutes. Accumulating every chunk in
# memory and writing once at the end means a failure in the last request throws
# away every earlier one, which is a bad trade for a download you repeat each
# time you refresh the data. Chunks are therefore appended as they arrive and a
# manifest records how far the run got.
#
# This mirrors the expired-options engine's manifest
# (`expired_options_fetch_dhan_common.py`), deliberately: the same failure modes
# apply, and one shape to learn is better than two.


#: Suffix of the progress file, alongside the CSV it describes.
MANIFEST_SUFFIX = ".manifest.json"

#: Reserved key describing the run the progress belongs to. Progress lives under
#: "progress", so a dunder name cannot collide with it.
RUN_SIGNATURE_KEY = "__run__"


def manifest_path_for(output: str | os.PathLike[str]) -> Path:
    """The progress file that belongs to one output CSV."""

    return Path(str(output) + MANIFEST_SUFFIX)


def load_manifest(path: Path) -> dict[str, object]:
    """Read the resume manifest, treating any damage as 'start over'."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def manifest_int(value: object, default: int = 0) -> int:
    """Read one integer out of manifest JSON without trusting its type.

    The manifest is a file on disk that anything could have written, so every
    field is `object` until proved otherwise. A wrong type means "no progress",
    which costs a re-download and never a corrupt file.
    """

    return value if isinstance(value, int) and not isinstance(value, bool) else default


def manifest_text(value: object) -> str | None:
    """Read one string out of manifest JSON, or None if it is anything else."""

    return value if isinstance(value, str) else None


def save_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write the manifest atomically so a crash cannot leave it half-written."""

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_signature(args, start: date) -> dict[str, object]:
    """Identify the run that a manifest's progress belongs to.

    Every field here changes what a row MEANS or where the chunk boundaries
    fall, so a change to any of them makes stored progress meaningless. The END
    date is deliberately absent: extending it is the normal case and is exactly
    what resuming is for.
    """

    return {
        "start": start.isoformat(),
        "interval": int(args.interval),
        "chunk_days": int(args.chunk_days),
        "security_id": str(args.security_id),
        "exchange_segment": normalize_exchange_segment(args.exchange_segment),
        "instrument_type": str(args.instrument_type),
    }


def resumable_manifest(
    manifest: dict[str, object], signature: dict[str, object]
) -> tuple[dict[str, object], str | None]:
    """Decide whether stored progress may be resumed, and say why not.

    The dangerous case is a manifest from a NARROWER or EARLIER run -- a one
    month smoke test, say, followed by the five-year backfill. Every chunk
    ending on or before the stored `last_to_date` would be skipped, so the
    earlier years would never be fetched and the command would report success
    over a file missing most of its history.

    Progress is therefore resumable only when it came from a run with the same
    identity AND the same start date. Anything else starts over.

    Returns ``(manifest_to_use, reason_it_was_discarded)``.
    """

    stored = manifest.get(RUN_SIGNATURE_KEY)
    if not manifest:
        return {}, None
    if not isinstance(stored, dict):
        return {}, "the existing manifest predates run signatures"

    differing = [
        f"{field}: {stored.get(field)!r} -> {value!r}"
        for field, value in signature.items()
        if stored.get(field) != value
    ]
    if differing:
        return {}, "this run does not continue the stored one (" + "; ".join(differing) + ")"
    return manifest, None


def append_chunk(
    csv_path: Path, frame: pd.DataFrame, *, last_timestamp: str | None
) -> tuple[int, int, str | None]:
    """Append one chunk to the output CSV.

    Returns ``(rows_written, file_size, newest_timestamp)``. Rows at or before
    ``last_timestamp`` are dropped, which is what makes a re-run idempotent and
    what absorbs the duplicated boundary bar between adjacent chunks. Each chunk
    has already been through `validate_ohlc_frame`, which REJECTS unordered
    timestamps rather than sorting them, so appending preserves the file's
    ascending order without re-sorting five years of rows.
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


def csv_first_row(csv_path: Path) -> str | None:
    """The first DATA line of the CSV, or None if there is not one yet.

    The manifest's byte count says how much of a file belongs to it; this says
    WHICH file. Without it a rewritten, restored or hand-edited CSV of the same
    name is trusted on length alone, and `truncate_to` cuts it at a boundary
    that means nothing -- leaving a hole the resume then skips straight over,
    because it appends from the stored `last_timestamp`. The result still looks
    ascending and de-duplicated, which is what makes it worth checking.
    """

    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            handle.readline()  # the header
            row = handle.readline().strip()
    except OSError:
        return None
    return row or None


def truncate_to(csv_path: Path, size: int) -> None:
    """Roll the CSV back to its last manifest-recorded byte length.

    The manifest is written after the data, so a crash between the two leaves a
    file LONGER than the manifest believes. Those trailing bytes are a partly
    written chunk; discarding them is what lets the resume start from a
    known-good boundary.
    """

    if not csv_path.exists():
        return
    actual = csv_path.stat().st_size
    if actual > size:
        print(f"Trimming {csv_path.name} from {actual} to {size} bytes (interrupted chunk)")
        with csv_path.open("r+b") as handle:
            handle.truncate(size)


def run_index_fetcher(defaults: IndexFetchDefaults) -> None:
    """
    Main script flow used by each wrapper.

    This is the function that turns the helper into a real CLI script:
    1. Parse user inputs
    2. Validate required settings
    3. Download all chunks
    4. Save the final CSV
    """
    args = parse_args(defaults)

    if not args.client_id or not args.access_token:
        raise ValueError(
            "Missing credentials. Set DHAN_CLIENT_CODE and DHAN_ACCESS_TOKEN in "
            "Dependencies/.env (run `python algo.py setup-token` to refresh the "
            "token). Only --client-id may be overridden on the command line; "
            "the token is environment-only."
        )

    if args.chunk_days <= 0 or args.chunk_days > 90:
        raise ValueError("--chunk-days must be between 1 and 90.")

    output_path = Path(args.output)
    manifest_path = manifest_path_for(output_path)

    if args.no_resume:
        # The original all-at-once behaviour, kept as the escape hatch: collect
        # every chunk, replace the file in one atomic write, and leave no
        # manifest behind for the next run to trust.
        frame = fetch_1m_history(args, defaults)
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        atomic_write_csv(frame, args.output)
        manifest_path.unlink(missing_ok=True)
        print(f"Saved {len(frame)} rows to: {args.output}")
        return

    # `fetch_1m_history` resolves the range again for itself. Under --lookback
    # that reads the clock, so a run straddling midnight signs the manifest with
    # one start date and fetches from another; the signature check then discards
    # the progress and downloads again. Wasteful on one day of the year, never
    # wrong -- which is the direction this has to fail in.
    start_dt, _ = resolve_date_range(args)
    signature = run_signature(args, start_dt)

    manifest, discarded = resumable_manifest(load_manifest(manifest_path), signature)
    if discarded:
        print(f"Starting over: {discarded}")

    stored = manifest.get("progress")
    progress: dict[str, object] = stored if isinstance(stored, dict) else {}

    stored_first_row = manifest_text(progress.get("first_row"))
    claims_rows = bool(manifest_int(progress.get("rows")))

    if progress and claims_rows and not output_path.exists():
        # Progress claiming rows that are no longer on disk. Honouring it would
        # skip every chunk it covers and write a CSV missing its own history.
        #
        # Progress claiming ZERO rows is a different thing and perfectly normal:
        # a range that opens on holidays has a resume point and no file yet, and
        # treating that as damage would re-request those empty windows forever.
        print(f"Ignoring the manifest: {output_path.name} is gone, starting over")
        manifest, progress = {}, {}
    elif progress and claims_rows and stored_first_row != csv_first_row(output_path):
        # Same name, different file. The byte count would still "fit", so
        # truncating on it would cut at a boundary that means nothing and the
        # resume would then append past whatever was lost.
        print(
            f"Ignoring the manifest: {output_path.name} is not the file it describes, "
            "starting over"
        )
        manifest, progress = {}, {}
        output_path.unlink()
    elif progress:
        truncate_to(output_path, manifest_int(progress.get("bytes")))
        print(
            f"Resuming after {progress.get('last_to_date')}: "
            f"{progress.get('rows', 0)} rows already on disk"
        )
    elif output_path.exists():
        # A CSV with no usable progress -- a deleted manifest, or a file written
        # by --no-resume or by this script before it kept one. There is no last
        # timestamp to de-duplicate against, so the only safe move is to start
        # the file over rather than append a second copy of the history.
        print(f"Ignoring existing {output_path.name}: no usable resume state, starting over")
        output_path.unlink()

    stored_to = manifest_text(progress.get("last_to_date"))
    resume_after = date.fromisoformat(stored_to) if stored_to else None

    rows_on_disk = manifest_int(progress.get("rows"))
    last_timestamp = manifest_text(progress.get("last_timestamp"))
    first_row = stored_first_row if progress else None

    def record(frame: pd.DataFrame, chunk_end: date) -> None:
        """Append one chunk and checkpoint, so a crash costs one chunk."""

        nonlocal rows_on_disk, last_timestamp, first_row

        written, size, newest = append_chunk(
            output_path, frame, last_timestamp=last_timestamp
        )
        rows_on_disk += written
        last_timestamp = newest
        if first_row is None and written:
            # Captured once, on the first append: this is what identifies the
            # file the byte count belongs to.
            first_row = csv_first_row(output_path)
        # The manifest is written AFTER the rows on purpose: it may lag the file
        # (which `truncate_to` repairs) but must never claim rows that are not
        # there, which nothing could repair.
        manifest[RUN_SIGNATURE_KEY] = signature
        manifest["progress"] = {
            "rows": rows_on_disk,
            "bytes": size,
            "first_row": first_row,
            "last_timestamp": last_timestamp,
            "last_to_date": chunk_end.isoformat(),
        }
        save_manifest(manifest_path, manifest)

    fetch_1m_history(args, defaults, on_chunk=record, resume_after=resume_after)

    print(f"Saved {rows_on_disk} rows to: {args.output}")
