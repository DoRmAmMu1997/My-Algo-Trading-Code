# Dependencies

Shared configuration, the safety/data primitives the front-test master imports on
every run (paper or live), and the live-execution layer. The broker subfolders only
come into play when live trading is switched on.

## Contents
```
Dependencies/
├── env.example              # template; copy to .env (git-ignored) and fill in
├── dhan_token_setup.py      # one-time DhanHQ OAuth login; writes DHAN_ACCESS_TOKEN into .env
│
│   # Shared runtime primitives (imported by the master on every run)
├── broker_contract.py       # OrderResult/OrderStatus types + the contract every broker adapter implements
├── market_data_health.py    # OHLC validation + live feed-freshness gates (entry freeze / liquidation)
├── tick_bar_builder.py      # pure websocket tick -> 1-min bar + official-candle true-up helpers
├── execution_ledger.py      # per-leg live order ledger; exposure is never inferred from a boolean
├── startup_exposure.py      # startup broker-book audit; live workers blocked until proven flat
├── trading_lifecycle.py     # safe live-session stop/square-off state machine
├── next_open_entry.py       # "enter at next candle open" pending-entry primitive (Goldmine/Money Machine)
├── risk_sizing.py           # shared risk-budget position sizing (whole lots, skip-over-budget)
├── secret_redaction.py      # strips credentials/tokens from anything headed to logs
├── diagnostic_preflight.py  # safety checks before a diagnostic test order's typed-YES prompt
├── test_*.py                # focused pytest suites for the modules above + repository policy
│
│   # Broker live-execution adapters (one folder each, same surface)
├── Kotak API/           # Kotak Neo live execution
│   ├── kotak_execution.py        # KotakExecutionClient (shared singleton)
│   └── diagnose_kotak_symbol.py  # read-only symbol check + optional test order
├── Shoonya API/         # Shoonya (Finvasia) live execution
│   ├── NorenApi.py               # vendored Finvasia NorenApi client
│   ├── shoonya_execution.py      # ShoonyaExecutionClient (shared singleton)
│   └── diagnose_shoonya_symbol.py# read-only symbol check + optional test order
├── Flattrade API/       # Flattrade Pi v2 REST live execution
│   ├── flattrade_execution.py      # FlattradeExecutionClient (shared singleton)
│   ├── diagnose_flattrade_symbol.py# read-only symbol check + optional test order
│   └── test_flattrade_execution.py # offline unit tests for the adapter
└── Dhan API/            # Dhan live execution (same SDK as market data, separate session)
    ├── dhan_execution.py           # DhanBrokerClient execution counterpart (shared singleton)
    ├── diagnose_dhan_symbol.py     # read-only symbol check + optional test order
    └── test_dhan_execution.py      # offline unit tests for the adapter
```

`.env` itself is git-ignored (it holds secrets); `env.example` is the committed template.
All execution modules load `.env` from this `Dependencies/` folder. A few files appear
here only at runtime and stay untracked: `.env`, `log_files/` (the master's append-mode
log), the nightly-refreshed `all_instrument <date>.csv` Dhan instrument master (used
for option-contract resolution by the runner and the Dhan adapter), and the Google
Sheets OAuth token cache when the EOD sheet writer is enabled.

(`test_market_data_health.py` lives at the repo root next to the master's own suite;
everything else in this folder is tested right here.)

## How live execution works
The master file guarded-loads all four broker clients and then picks one with
`LIVE_BROKER` (`KOTAK`, `SHOONYA`, `FLATTRADE`, or `DHAN`). The rest of the runner
only touches a single generic `execution_client`, so the brokers are interchangeable.
All clients expose the **same surface**:
`ensure_logged_in()`, `preload_scrip_master()`, `resolve_option_symbol()`,
`place_market_order()` (with fill confirmation), `get_order_status()`,
`cancel_order()`, `list_open_orders()`, `list_open_positions()`,
`recover_after_reconciliation()`, `extract_order_id()`, `logout()`, and an
`is_logged_in` flag.

Safety gates (all in `.env`):
- `LIVE_TRADING_ENABLED` — global kill-switch (default `false` = everything paper).
- `LIVE_BROKER` — `KOTAK`, `SHOONYA`, `FLATTRADE`, or `DHAN`. An unrecognised value
  **fails closed** (live disabled, paper only) rather than guessing a broker.
- `<PREFIX>_LIVE_TRADING` — per strategy. A strategy goes live only when the global
  switch **and** its own flag are both true.
- A live entry falls back to paper only after an explicit typed `REJECTED` result
  with zero fill. `PARTIAL` or `UNKNOWN` means broker exposure may exist: entries
  freeze, exits remain available, and reconciliation continues. A rejected or
  ambiguous exit keeps the leg tracked and open.
- Startup queries open orders and relevant option positions before any live
  worker starts. Exposure or an indeterminate broker reply blocks all live
  workers; it is never auto-adopted or auto-flattened.

Product/exchange per broker is derived from `LIVE_BROKER`: Kotak uses `nse_fo` +
`KOTAK_PRODUCT_TYPE`; Shoonya uses `NFO` + `SHOONYA_PRODUCT_TYPE`; Flattrade uses
`NFO` + `FLATTRADE_PRODUCT_TYPE` (INTRADAY → MIS/I, NORMAL → NRML/M); Dhan uses
`NSE_FNO` + `DHAN_PRODUCT_TYPE`.

## Credentials (in `.env`)
- **Kotak Neo:** `CONSUMER_KEY`, `MOBILE`, `MPIN`, `UCC`. TOTP is typed at startup
  (not stored).
- **Shoonya:** `SHOONYA_USERID`, `SHOONYA_PASSWORD`, `SHOONYA_VENDOR_CODE`,
  `SHOONYA_API_SECRET`, `SHOONYA_IMEI`, `SHOONYA_TOTP_SECRET` (base32 seed; the TOTP
  is auto-generated via `pyotp`, or you're prompted if blank). The client is vendored
  here.
- **Flattrade:** `FLATTRADE_CLIENT_ID`, `FLATTRADE_API_KEY`, and
  `FLATTRADE_API_SECRET`. `FLATTRADE_ACCESS_TOKEN` is optional: when blank, startup
  opens Flattrade authorization and asks for the returned `request_code`. Tokens
  remain in memory unless you manually paste one into `.env`. Uses core `requests`
  and `pandas`; no extra SDK is required. Token exchange must originate from the
  static IP registered against your Flattrade API key.
- **Dhan:** reuses the market-data credentials (`DHAN_CLIENT_CODE` +
  `DHAN_ACCESS_TOKEN` from `dhan_token_setup.py`) — no extra keys. Execution runs
  on its own SDK session, separate from the market-data session, and resolves
  option contracts from the local `all_instrument <date>.csv` instrument master.

The exact upstream broker compatibility environment is recorded in:
```
pip install -r requirements-brokers.txt
```
Run that command only in a clean broker-validation environment: Kotak v2.0.1
declares older exact pandas/requests constraints that conflict with the app's
audited core runtime. In the live app environment install Shoonya with
`pip install pyotp==2.9.0 websocket-client==1.8.0`, and install Kotak with the
official tagged command plus `--no-deps` shown in the root README. Flattrade
uses the pinned core `requests`/`pandas` set.

> Note: Finvasia is decommissioning the legacy Shoonya QuickAuth endpoint (it returns
> HTTP 502), so Shoonya live login needs the new OAuth API before it works again.

## Diagnostics (read-only, with an optional REAL test order)
Each broker has a diagnostic that logs in and shows whether a contract resolves to a
valid trading symbol. Passing `--place-order` and an explicit `--qty` additionally
places a **real**, confirmation-gated BUY. It attempts the matching SELL only after
the BUY is confirmed fully filled; an ambiguous result requires broker reconciliation.
The diagnostic asks you to type `YES` before submitting anything:
```
python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950 --place-order --qty <current-lot-size>
python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" CE 23950 DDMMMYY --place-order --qty <current-lot-size>
python "Dependencies/Flattrade API/diagnose_flattrade_symbol.py" CE 24150 DDMMMYY --place-order --qty <current-lot-size>
python "Dependencies/Dhan API/diagnose_dhan_symbol.py" CE 24150 DDMMMYY --place-order --qty <current-lot-size>
```
