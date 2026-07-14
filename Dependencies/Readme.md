# Dependencies

Shared configuration and the live-execution layer used by the front-test master file.
Nothing here is needed for paper trading except `.env` (for tunable parameters); the
broker subfolders only come into play when live trading is switched on.

## Contents
```
Dependencies/
├── env.example          # template; copy to .env (git-ignored) and fill in
├── dhan_token_setup.py  # one-time DhanHQ OAuth login; writes DHAN_ACCESS_TOKEN into .env
├── Kotak API/           # Kotak Neo live execution
│   ├── kotak_execution.py        # KotakExecutionClient (shared singleton)
│   └── diagnose_kotak_symbol.py  # read-only symbol check + optional test order
├── Shoonya API/         # Shoonya (Finvasia) live execution
│   ├── NorenApi.py               # vendored Finvasia NorenApi client
│   ├── shoonya_execution.py      # ShoonyaExecutionClient (shared singleton)
│   └── diagnose_shoonya_symbol.py# read-only symbol check + optional test order
└── Flattrade API/       # Flattrade Pi v2 REST live execution
    ├── flattrade_execution.py      # FlattradeExecutionClient (shared singleton)
    └── diagnose_flattrade_symbol.py# read-only symbol check + optional test order
```

`.env` itself is git-ignored (it holds secrets); `env.example` is the committed template.
All execution modules load `.env` from this `Dependencies/` folder.

## How live execution works
The master file guarded-loads all three broker clients and then picks one with
`LIVE_BROKER` (`KOTAK`, `SHOONYA`, or `FLATTRADE`). The rest of the runner only
touches a single generic `execution_client`, so the brokers are interchangeable.
All clients expose the **same surface**:
`ensure_logged_in()`, `preload_scrip_master()`, `resolve_option_symbol()`,
`place_market_order()` (with fill confirmation), `extract_order_id()`, `logout()`,
and an `is_logged_in` flag.

Safety gates (all in `.env`):
- `LIVE_TRADING_ENABLED` — global kill-switch (default `false` = everything paper).
- `LIVE_BROKER` — `KOTAK`, `SHOONYA`, or `FLATTRADE`. An unrecognised value **fails closed**
  (live disabled, paper only) rather than guessing a broker.
- `<PREFIX>_LIVE_TRADING` — per strategy. A strategy goes live only when the global
  switch **and** its own flag are both true.
- On any order failure (login, unresolved symbol, reject, unconfirmed fill) the
  strategy falls back to paper for that trade.

Product/exchange per broker is derived from `LIVE_BROKER`: Kotak uses `nse_fo` +
`KOTAK_PRODUCT_TYPE`; Shoonya uses `NFO` + `SHOONYA_PRODUCT_TYPE`; Flattrade uses
`NFO` + `FLATTRADE_PRODUCT_TYPE` (INTRADAY → MIS/I, NORMAL → NRML/M).

## Credentials (in `.env`)
- **Kotak Neo:** `CONSUMER_KEY`, `MOBILE`, `MPIN`, `UCC`. TOTP is typed at startup
  (not stored). Needs `pip install neo_api_client`.
- **Shoonya:** `SHOONYA_USERID`, `SHOONYA_PASSWORD`, `SHOONYA_VENDOR_CODE`,
  `SHOONYA_API_SECRET`, `SHOONYA_IMEI`, `SHOONYA_TOTP_SECRET` (base32 seed; the TOTP
  is auto-generated via `pyotp`, or you're prompted if blank). The client is vendored
  here; needs `pip install pyotp websocket-client`.
- **Flattrade:** `FLATTRADE_CLIENT_ID`, `FLATTRADE_API_KEY`, and
  `FLATTRADE_API_SECRET`. `FLATTRADE_ACCESS_TOKEN` is optional: when blank, startup
  opens Flattrade authorization and asks for the returned `request_code`. Tokens
  remain in memory unless you manually paste one into `.env`. Uses core `requests`
  and `pandas`; no extra SDK is required. Token exchange must originate from the
  static IP registered against your Flattrade API key.

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
```
