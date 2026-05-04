"""
Beginner-friendly BANKNIFTY wrapper script.

This file is intentionally small because the heavy lifting is shared in:
`index_1m_5y_data_fetch_dhan_common.py`

What this wrapper is responsible for:
1. Define BANKNIFTY-specific defaults.
2. Point the shared fetch engine to those defaults.
3. Run the download flow.

Default BANKNIFTY values used here:
- security_id = 25
- exchange segment = IDX_I
- instrument type = INDEX
- output CSV = Backtest Outputs/banknifty_renko_futures_5y_1min_data.csv
"""

import os

from index_1m_5y_data_fetch_dhan_common import IndexFetchDefaults, run_index_fetcher


# Anchor the default output to <repo_root>/Backtest Outputs/ so the CSV
# always lands in the repo's shared output folder regardless of the cwd
# the script is launched from. `__file__` is at <repo>/Data Extractors/...
# so dirname(dirname(__file__)) == <repo_root>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# These defaults tell the shared fetch engine which index it should download
# when this specific wrapper script is executed.
BANKNIFTY_DEFAULTS = IndexFetchDefaults(
    display_name="BANKNIFTY",
    security_id="25",
    default_output=os.path.join(
        _REPO_ROOT, "Backtest Outputs", "banknifty_renko_futures_5y_1min_data.csv"
    ),
)


if __name__ == "__main__":
    run_index_fetcher(BANKNIFTY_DEFAULTS)
