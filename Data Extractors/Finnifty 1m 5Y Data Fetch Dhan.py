"""
Beginner-friendly FINNIFTY wrapper script.

This file stays lightweight on purpose. The shared fetching logic lives in:
`index_1m_5y_data_fetch_dhan_common.py`

What this wrapper does:
1. Supply FINNIFTY-specific defaults.
2. Reuse the same chunked OHLC download flow as the other index scripts.
3. Save the final CSV in the Backtest Outputs folder.

Default FINNIFTY values used here:
- security_id = 27
- exchange segment = IDX_I
- instrument type = INDEX
- output CSV = Backtest Outputs/finnifty_renko_futures_5y_1min_data.csv
"""

import os

from index_1m_5y_data_fetch_dhan_common import IndexFetchDefaults, run_index_fetcher


# These defaults make the shared helper behave like a FINNIFTY fetcher.
FINNIFTY_DEFAULTS = IndexFetchDefaults(
    display_name="FINNIFTY",
    security_id="27",
    default_output=os.path.join(
        "Backtest Outputs", "finnifty_renko_futures_5y_1min_data.csv"
    ),
)


if __name__ == "__main__":
    run_index_fetcher(FINNIFTY_DEFAULTS)
