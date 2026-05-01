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
   and combine the chunks.
5. Save the final result as a CSV that your backtests can read later.
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from dhanhq import dhanhq


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
    default_client_id: str = "1102601655"
    default_access_token: str = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc2NzAwOTY5LCJpYXQiOjE3NzY2MTQ1NjksInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAyNjAxNjU1In0.Ivic07sLx-lbLd1LYtcwNkNGJk5XOtN7LBfGeeEH5jNrLevrRADofSI3DL4OABo0svY68k0mFeX8GWDQk6Ofcw"


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
    # - first preference: explicit CLI values
    # - second preference: environment variables
    # - last fallback: wrapper-provided defaults (blank unless you choose
    #   to hardcode them in the wrapper)
    parser.add_argument(
        "--client-id",
        default=os.getenv("DHAN_CLIENT_CODE", defaults.default_client_id),
    )
    parser.add_argument(
        "--access-token",
        default=os.getenv("DHAN_TOKEN_ID", defaults.default_access_token),
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
    return parser.parse_args()


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


def infer_epoch_unit(values: pd.Series) -> str:
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


def normalize_response_data(data) -> pd.DataFrame:
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
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
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
        unit = infer_epoch_unit(out["timestamp_raw"])
        ts = pd.to_datetime(out["timestamp_raw"], unit=unit, errors="coerce", utc=True)
    else:
        maybe_num = pd.to_numeric(out["timestamp_raw"], errors="coerce")
        if maybe_num.notna().sum() >= max(1, len(out) // 2):
            unit = infer_epoch_unit(maybe_num)
            ts = pd.to_datetime(maybe_num, unit=unit, errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(out["timestamp_raw"], errors="coerce", utc=True)

    # Dhan timestamps are normalized into India market time because that is the
    # timezone your backtest data uses across the project.
    out["timestamp"] = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    out = out.drop(columns=["timestamp_raw"]).dropna()

    return out[["timestamp", "open", "high", "low", "close", "volume"]]


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

    return normalize_response_data(resp.get("data"))


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


def fetch_1m_history(args, defaults: IndexFetchDefaults) -> pd.DataFrame:
    """
    Download the full requested date range in many smaller pieces.

    Why chunking matters:
    - Dhan minute API has a practical limit on how much history can be fetched
      in one request.
    - So we walk from start date to end date chunk by chunk, save each chunk,
      and then merge them into one final DataFrame.
    """
    start_dt, end_dt = resolve_date_range(args)
    dhan = dhanhq(args.client_id, args.access_token)
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

        if not chunk_df.empty:
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
            "Missing credentials. Set DHAN_CLIENT_CODE and DHAN_TOKEN_ID, "
            "or pass --client-id and --access-token."
        )

    if args.chunk_days <= 0 or args.chunk_days > 90:
        raise ValueError("--chunk-days must be between 1 and 90.")

    df = fetch_1m_history(args, defaults)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} rows to: {args.output}")
