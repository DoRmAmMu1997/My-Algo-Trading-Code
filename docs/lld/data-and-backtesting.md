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

The backtests themselves are not tested — they are reference scripts, not
runtime code. `compileall` is their only gate.

---

## 7. Limitations

- **No survivorship or corporate-action handling.** Index data, so not usually a
  concern, but it is not modelled.
- **Backtests do not model the spread gate, sizing rejections, or partial fills.**
  A backtest is an idea filter, not a P&L forecast. The paper-trading phase is
  what exercises the execution path.
- **No volume** in the source data, so volume-derived indicators are proxies here
  too — the same limitation the live feed has.
