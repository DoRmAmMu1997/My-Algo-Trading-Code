# LLD — Data extraction and backtesting

**Owns:** `Data Extractors/` · `My Backtest Files (For Reference)/` · the
`fetch-data` and `backtest` commands in `algo.py`

---

## 1. Responsibility

Produce the historical 1-minute OHLC that strategies are validated against, and
run those strategies over it before they are given capital.

Neither of these touches money or a broker. That is why they live outside the
runner's safety machinery entirely.

---

## 2. Data extraction

```
 Data Extractors/
   index_1m_5y_data_fetch_dhan_common.py     ← the shared engine
   nifty_1m_5y_data_fetch_dhan.py            ┐
   banknifty_1m_5y_data_fetch_dhan.py        ├ thin per-index wrappers
   finnifty_1m_5y_data_fetch_dhan.py         ┘
```

One engine, three wrappers. The wrappers differ only in the instrument they
name; everything about paging, retry, epoch-unit handling and CSV writing lives
in the common module. All four are under mypy: the wrappers were renamed to
identifier names under [ADR-0014](../adr/0014-tiered-rename-of-spaced-filenames.md)
and joined the type gate in the same commit.

Output lands in `Backtest Outputs/`, which is gitignored.

```bash
python algo.py fetch-data --index nifty --interval 5 --lookback 5y
```

Any flag beyond the selector passes straight through to the underlying script,
and each script still runs standalone.

### 2.1 Epoch-unit inference

The Dhan intraday response has been observed with different epoch units.
`_infer_epoch_unit` / `_validate_single_epoch_unit` (in the master, mirrored in
the extractor engine) infer the unit and then **assert it is consistent across
the response** — a frame with mixed units would otherwise produce candles
decades apart with no obvious symptom.

### 2.2 Expired options

```
 Data Extractors/
   expired_options_fetch_dhan_common.py     ← the engine
   expiry_calendar.py                       ← pure expiry derivation
   nifty_expired_options_fetch_dhan.py      ← thin wrapper
```

```bash
python algo.py fetch-expired-options --index nifty --dry-run
python algo.py fetch-expired-options --index nifty --lookback 5y --verify-expiries
```

Real expired NIFTY option bars — OHLC **plus volume, open interest, implied
volatility, the actual strike and the spot** — via `POST /v2/charts/rollingoption`
(`dhanhq.expired_options_data`). Five years, minute resolution. Output is one CSV
per `(strike label, option type)` under
`Backtest Outputs/expired_options/nifty/`, alongside a
`_weekly_expiry_calendar.csv` and a `_manifest.json`.

Three properties worth knowing before using it, all covered by
[ADR-0015](../adr/0015-rolling-relative-strike-expired-options.md):

- **Strikes are relative, not contracts, and re-pick every bar.** Measured: the
  `ATM` call switched strike 69 times in one session, and a four-session `ATM`
  file held 13 distinct contracts. Re-key on `strike_price` and `expiry_date` to
  rebuild a fixed contract; never treat `strike_label` as an instrument.
- **The tail of a range is dropped.** Sessions after the last expiry inside the
  window cannot be labelled, so a backfill ending today loses the current
  part-week. The run warns about it; extend `--end-date` past the next expiry.
- **A session is usually 375 bars, but not always** — 20-Jan-2025 has a 15:30
  print as well, giving 376. Do not assume the last bar is 15:29.
- **±10 strikes is a ±500-point band.** A contract that drifts further from spot
  stops appearing. Silence is missing data, not a worthless option.
- **`expiryCode` is 1-based here** (1 = near), unlike the annexure's table for
  `/charts/historical`. A zero is rejected as a missing field.

The download is resumable per chunk: `_manifest.json` records each series' byte
length, and a restart trims any chunk a crash left half-written before carrying
on. Pacing goes through the shared `RollingWindowRateLimiter`.

---

## 3. Backtesting

```
 My Backtest Files (For Reference)/
   renko_strategy_backtest.py
   ema_trend_strategy_backtest.py
   heikin_ashi_futures_5y_backtest.py
   cpr_strategy_backtest.py
   profit_shooter_backtest.py
   Subhamoy Strategies/
     goldmine_strategy_backtest.py
     money_machine_strategy_backtest.py
     subhamoy_backtest_common.py
```

Built on [`backtesting.py`](https://pypi.org/project/backtesting/), run against
the CSVs from §2:

```bash
python algo.py backtest --strategy renko --data "Backtest Outputs/nifty_renko_futures_5y_1min_data.csv"
```

These are explicitly **reference material**. They are excluded from Ruff's
default treatment (`E402` is allowed — they keep the deliberate
`sys.path`-before-import pattern), excluded from Bandit, and excluded from
coverage. They are kept because they record how a strategy was evaluated, not
because they are part of the runtime.

---

## 4. The shared-candle invariant

The single most important property linking these three phases:

> **Backtests, the REST producer, and the websocket producer must all be looking
> at the same candles.**

That is why the websocket producer trues its tick-built bars up against official
REST candles once per minute, with official always winning (see
[`market-data.md`](market-data.md) §3.2). Without it, live bars would slowly
drift away from the bars every backtest result was computed on, and no
individual test would fail.

Both phases also share the "candles are labelled by their START time" convention
and the completed-candle rule.

---

## 5. Typical workflow

```
1. python algo.py fetch-data --index nifty          → CSV in Backtest Outputs/
2. python algo.py backtest --strategy renko --data "…csv"
3. python algo.py run                                → paper by default
4. <PREFIX>_LIVE_TRADING=true + LIVE_TRADING_ENABLED=true → live, one strategy at a time
```

Step 3 should run for at least one full session before step 4 for any strategy
whose size or logic changed.

---

## 6. Testing

`Tests/Data Extractors/test_index_fetch_construction.py` covers the shared
engine's construction and request-building. Its module path anchor points back
at the **source** `Data Extractors/` folder.

`test_expired_options_fetch.py` and `test_expiry_calendar.py` cover the
expired-options engine and its calendar. Neither touches the network — the
DhanHQ client is a `SimpleNamespace` duck. The cases that earn their keep encode
what the API taught us: `expiryCode` being 1-based, a `str` vs `dict` `remarks`
meaning two completely different things, the parallel-array response shape, the
holiday roll-back, and the 01-Sep-2025 Thursday→Tuesday change.

The backtests themselves are not tested — they are reference scripts, not
runtime code. `compileall` is their only gate.

---

## 7. Limitations

- **No survivorship or corporate-action handling.** Index data, so not usually a
  concern, but it is not modelled.
- **Backtests do not model the spread gate, sizing rejections, or partial fills.**
  A backtest is an idea filter, not a P&L forecast. The paper-trading phase is
  what exercises the execution path.
- **No volume** in the *index* source data, so volume-derived indicators are
  proxies there — the same limitation the live feed has. The expired-options
  data of §2.2 does carry real volume, OI and IV; it is the one dataset here
  that does not need the proxy.
- **The expired-options set has no bid/ask** either, so spread and slippage
  still cannot be modelled from it, and its ±500-point strike band means a
  drifted contract goes missing rather than reading as worthless (§2.2).
