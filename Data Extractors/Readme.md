# What is a data extractor?
Data extractor extracts 1 minute historical OHLC data of your preferred index(NIFTY/BANKNIFTY/FINNIFTY) for your preferred time - like 3 months or 6 months or 1 year or even 5 years

# The use?
If you want to implement your backtesting logics(like or unlike how I implememted my own), that data would be helpful

# Files in this folder
- `nifty_1m_5y_data_fetch_dhan.py` — NIFTY (security_id 13) wrapper.
- `banknifty_1m_5y_data_fetch_dhan.py` — BANKNIFTY (security_id 25) wrapper.
- `finnifty_1m_5y_data_fetch_dhan.py` — FINNIFTY (security_id 27) wrapper.
- `index_1m_5y_data_fetch_dhan_common.py` — shared chunked-download engine. Don't run this directly; the three wrappers above call it.

# How to run
Each wrapper has the index-specific defaults baked in, so this is enough:
```
python "Data Extractors/nifty_1m_5y_data_fetch_dhan.py"
```
Override anything via CLI — `--start-date`, `--end-date`, `--output`, `--client-id`, `--chunk-days`. Run with `--help` for the full list. (The access token has no CLI flag on purpose — see Credentials.)

# Resuming an interrupted download
A five-year pull is around 21 requests over about ten minutes. Each chunk is appended to the CSV as it arrives and the progress is recorded in `<output>.manifest.json`, so a run that dies partway through picks up from the last completed chunk instead of downloading everything again. Just run the same command a second time.

The manifest is only trusted when it describes the *same* run — same start date, interval, chunk size, security id and segment. Change any of those and the download starts over, because progress from a different run would skip windows that were never actually fetched. Pass `--no-resume` to ignore the manifest deliberately and write the whole file in one atomic replace.

# Where the CSV lands
By default, in `<repo_root>/Backtest Outputs/<index>_renko_futures_5y_1min_data.csv`. The folder is auto-created. Override with `--output`.

# Credentials
Set `DHAN_CLIENT_CODE` and `DHAN_ACCESS_TOKEN` as environment variables (e.g. in `Dependencies/.env`) — `DHAN_ACCESS_TOKEN` is the key the rest of the repo uses and the one `python algo.py setup-token` writes. The older `DHAN_TOKEN_ID` is still accepted as a fallback. The client id may also be passed via `--client-id`; the access token is deliberately environment-only — a token typed on the command line would land in shell history and process listings.
