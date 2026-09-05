"""
Download expired NIFTY weekly option history from DhanHQ.

This is a thin wrapper. Everything that actually does the work -- paging, retry,
expiry labelling, resumable writing -- lives in
`expired_options_fetch_dhan_common.py`, exactly as the index fetchers keep their
logic in `index_1m_5y_data_fetch_dhan_common.py`. All this file decides is which
underlying to ask about and where the CSVs land.

Typical use:

    python algo.py fetch-expired-options --index nifty --dry-run
    python algo.py fetch-expired-options --index nifty --lookback 5y --verify-expiries

Credentials come from `Dependencies/.env` (`DHAN_CLIENT_CODE` and
`DHAN_ACCESS_TOKEN`); the token is never a command-line flag. The endpoint needs
an active DhanHQ Data API subscription.

A word on what you get back, because it is easy to misread: the strikes are
RELATIVE. `ATM+3` is whichever strike sat three above spot at the time, so it is
not one contract over time. Re-key on the `strike_price` and `expiry_date`
columns to rebuild fixed contracts. See the engine module's docstring and
docs/adr/0015 before building a backtest on this.
"""

import os

from expired_options_fetch_dhan_common import ExpiredOptionsDefaults, run_expired_options_fetcher

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NIFTY_EXPIRED_OPTIONS_DEFAULTS = ExpiredOptionsDefaults(
    display_name="NIFTY",
    # 13 on the index side, the same ID the index fetchers and the live runner
    # use (NIFTY_INDEX_SECURITY_ID). Note this is NOT the 26000 that appears as
    # UNDERLYING_SECURITY_ID in the instrument master.
    security_id=13,
    default_output_dir=os.path.join(_REPO_ROOT, "Backtest Outputs", "expired_options", "nifty"),
)

if __name__ == "__main__":
    run_expired_options_fetcher(NIFTY_EXPIRED_OPTIONS_DEFAULTS)
