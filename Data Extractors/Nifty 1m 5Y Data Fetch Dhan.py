"""
Beginner-friendly NIFTY wrapper script.

This file now matches the same pattern as the BANKNIFTY and FINNIFTY scripts.
That means:
- the shared download logic lives in one common helper file
- this wrapper only provides NIFTY-specific defaults
- the overall flow is easier to maintain because the three scripts now stay
  consistent with each other

What this wrapper does:
1. Define the default NIFTY security ID and output path.
2. Reuse the shared chunked Dhan download engine.
3. Save the final OHLC CSV into the Backtest Outputs folder.

Default NIFTY values used here:
- security_id = 13
- exchange segment = IDX_I
- instrument type = INDEX
- output CSV = Backtest Outputs/nifty_renko_futures_5y_1min_data.csv
"""

import os

from index_1m_5y_data_fetch_dhan_common import IndexFetchDefaults, run_index_fetcher


# These defaults make the shared helper behave like a NIFTY-specific downloader.
NIFTY_DEFAULTS = IndexFetchDefaults(
    display_name="NIFTY",
    security_id="13",
    default_output=os.path.join(
        "Backtest Outputs", "nifty_renko_futures_5y_1min_data.csv"
    ),
)


if __name__ == "__main__":
    run_index_fetcher(NIFTY_DEFAULTS)
