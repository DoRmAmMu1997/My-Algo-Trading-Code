# What is a data extractor?
Data extractor extracts 1 minute historical OHLC data of your preferred index(NIFTY/BANKNIFTY/FINNIFTY) for your preferred time - like 3 months or 6 months or 1 year or even 5 years

# The use?
If you want to implement your backtesting logics(like or unlike how I implememted my own), that data would be helpful

# Files in this folder
- `Nifty 1m 5Y Data Fetch Dhan.py` — NIFTY (security_id 13) wrapper.
- `Banknifty 1m 5Y Data Fetch Dhan.py` — BANKNIFTY (security_id 25) wrapper.
- `Finnifty 1m 5Y Data Fetch Dhan.py` — FINNIFTY (security_id 27) wrapper.
- `index_1m_5y_data_fetch_dhan_common.py` — shared chunked-download engine. Don't run this directly; the three wrappers above call it.

# How to run
Each wrapper has the index-specific defaults baked in, so this is enough:
```
python "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py"
```
Override anything via CLI — `--from-date`, `--to-date`, `--output`, `--client-id`, `--access-token`, `--chunk-days`. Run with `--help` for the full list.

# Where the CSV lands
By default, in `<repo_root>/Backtest Outputs/<index>_renko_futures_5y_1min_data.csv`. The folder is auto-created. Override with `--output`.

# Credentials
Either set `DHAN_CLIENT_CODE` and `DHAN_TOKEN_ID` as environment variables, or pass them via `--client-id` and `--access-token`.
