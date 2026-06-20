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
└── Shoonya API/         # Shoonya (Finvasia) live execution
    ├── NorenApi.py               # vendored Finvasia NorenApi client
    ├── shoonya_execution.py      # ShoonyaExecutionClient (shared singleton)
    └── diagnose_shoonya_symbol.py# read-only symbol check + optional test order
```

`.env` itself is git-ignored (it holds secrets); `env.example` is the committed template.
Both execution modules load `.env` from this `Dependencies/` folder.

## How live execution works
The master file loads **both** broker clients (each guarded — a missing SDK just
disables that broker) and then picks one with `LIVE_BROKER` (`KOTAK` or `SHOONYA`).
The rest of the runner only touches a single generic `execution_client`, so the two
brokers are interchangeable. Both clients expose the **same surface**:
`ensure_logged_in()`, `preload_scrip_master()`, `resolve_option_symbol()`,
`place_market_order()` (with fill confirmation), `extract_order_id()`, `logout()`,
and an `is_logged_in` flag.

Safety gates (all in `.env`):
- `LIVE_TRADING_ENABLED` — global kill-switch (default `false` = everything paper).
- `LIVE_BROKER` — `KOTAK` or `SHOONYA`. An unrecognised value **fails closed**
  (live disabled, paper only) rather than guessing a broker.
- `<PREFIX>_LIVE_TRADING` — per strategy. A strategy goes live only when the global
  switch **and** its own flag are both true.
- On any order failure (login, unresolved symbol, reject, unconfirmed fill) the
  strategy falls back to paper for that trade.

Product/exchange per broker is derived from `LIVE_BROKER`: Kotak uses `nse_fo` +
`KOTAK_PRODUCT_TYPE`; Shoonya uses `NFO` + `SHOONYA_PRODUCT_TYPE` (INTRADAY → MIS/I,
NORMAL → NRML/M).

## Credentials (in `.env`)
- **Kotak Neo:** `CONSUMER_KEY`, `MOBILE`, `MPIN`, `UCC`. TOTP is typed at startup
  (not stored). Needs `pip install neo_api_client`.
- **Shoonya:** `SHOONYA_USERID`, `SHOONYA_PASSWORD`, `SHOONYA_VENDOR_CODE`,
  `SHOONYA_API_SECRET`, `SHOONYA_IMEI`, `SHOONYA_TOTP_SECRET` (base32 seed; the TOTP
  is auto-generated via `pyotp`, or you're prompted if blank). The client is vendored
  here; needs `pip install pyotp websocket-client`.

> Note: Finvasia is decommissioning the legacy Shoonya QuickAuth endpoint (it returns
> HTTP 502), so Shoonya live login needs the new OAuth API before it works again.
> Kotak Neo is the working live path today.

## Diagnostics (read-only, with an optional REAL test order)
Each broker has a diagnostic that logs in and shows whether a contract resolves to a
valid trading symbol. Passing `--place-order` additionally places a **real**,
confirmation-gated **round-trip** order (1-lot BUY, confirm fill, then auto SELL to
flatten) — it asks you to type `YES` first:
```
python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950 --place-order
python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" CE 23950 26JUN25 --place-order
```
