# ADR-0020: A second index on the monitoring chart

**Status:** Accepted
**Date:** 2026-10-22
**Deciders:** repository owner

## Context

The monitoring dashboard charts NIFTY and only NIFTY, because nothing in the
pipeline had an instrument dimension:

| Layer | Shape before this change |
|---|---|
| `SharedMarketDataStore._snapshots` | `dict[str, MarketSnapshot]` keyed by **timeframe**, and `REQUIRED_TIMEFRAMES` is `("1",)` — so one series, ever |
| `CentralMarketDataFetcher.fetch_ohlc` | hard-coded `NIFTY_INDEX_*`, and raises for any timeframe but `"1"` |
| `WebSocketMarketDataFetcher` | one `TickBarAggregator`, and the tick→bar path gated on the NIFTY security id |
| `DASHBOARD_HISTORY_CSV` | one NIFTY filename |
| `/api/history` page keys | `"<tf>:<page>"` |
| `dashboard.js` | one `candleSeries`; `historyBars`/`historyCols`/`historyNextPage`/`historyExhausted` all three-key literals |

BankNIFTY is not an abstract want. The SL Hunting agent reads it for
cross-confirmation, and its mirror leg is a **real BankNIFTY position** — so
"where do I stand right now" already had a BankNIFTY half the chart could not
show. `docs/lld/market-data.md` had named the work in advance: *"Making a second
index first-class means parameterizing this component."*

The constraint is the same one ADR-0016 was written under. This code runs inside
a live-money process, and the store is **the** rendezvous between the producer
thread and ~29 worker threads. A monitoring feature that disturbs it is worse
than no monitoring feature.

## Decision

### 1. An instrument dimension in the store, with the primary as the default

`_snapshots` is keyed `(instrument, timeframe)`. `update`, `get` and
`peek_snapshot_meta` take `instrument` as a **keyword-only argument defaulting
to `PRIMARY_INSTRUMENT`**, so all ~29 `store.get("1")` call sites are byte for
byte what they were. The change is four lines of the store's internals.

The alternative considered first was a second, parallel pool for secondary
instruments. Rejected: two pools for one kind of thing, and the next instrument
would need a third.

### 2. The feed-health gate stays pinned to the primary, by name

`record_market_data_refresh` reads `self._snapshots[(PRIMARY_INSTRUMENT, "1")]`,
hard-coded, with a comment saying it must stay that way.

**This is the load-bearing safety property of the whole change.** That result
drives the 30-second liquidation clock. A secondary instrument published on a
60-second cadence is *routinely* staler than the primary; if the gate could see
it, an ordinary BankNIFTY REST hiccup could pull a real auto-square-off forward.

`test_the_health_gate_reads_the_primary_frame_alone` builds two independent
stores — one with a two-year-old secondary frame, one with none — and asserts the
two health snapshots are identical **in full**. Two stores rather than two
refreshes of one, because the healthy streak counts refreshes and would differ
for a reason that has nothing to do with the instrument.

### 3. BankNIFTY comes from REST in both feed modes, never from ticks

`SecondaryIndexPublisher` owns the fetch. Both producers hold one and call
`publish_if_due()` once per cycle, **after** the health gate has been recorded
and logged. Four properties, each with a test:

- it never calls the health gate, and never touches the producer's `ohlc_ok`;
- it never raises into the caller, so a cycle that cannot price BankNIFTY still
  publishes NIFTY — and in the websocket producer still syncs subscriptions;
- it has its own cadence, re-armed **before** the call so a refusing broker is
  not retried every cycle;
- disabled, it is one boolean read and no call at all.

It is deliberately **not** wired into the tick path. That would mean a second
aggregator, a second rollover clock and a second true-up loop inside the live
bar producer, for an index nothing trades off. A minute-latency REST bar is the
right answer, and it keeps the tick path untouched in both modes.

`validate_ohlc_frame` is applied unchanged: a monitoring instrument is not a
reason to accept a frame the primary one would reject.

### 4. Off by default, and the page says so

`DASHBOARD_BANKNIFTY_ENABLED=false`. On, it costs one REST call per
`DASHBOARD_BANKNIFTY_REFRESH_SECONDS` (default 60) plus a second back-history
load — another ~29 seconds and ~63 MB of RSS on the five-year file.

The state document's `chart` block is keyed by instrument, so **its keys are the
set of instruments being published**. The page reads them and dims the button
for anything absent, naming the `.env` key in the button's title. The page is
never told separately and so cannot disagree.

### 5. `/api/chart` becomes the second endpoint that reads the request

It takes `inst`, validated against a frozen `CHART_INSTRUMENTS` exactly as
`/api/history` validates `tf`: absent is the default instrument, so the URLs
that worked before still work; anything not an exact match is a 404; an
explicitly empty value is refused, which is what `keep_blank_values` is for.
Matching is case-sensitive, because the store's keys are the names as written.

History page keys gain the instrument as the **leading** component,
`"<instrument>:<tf>:<page>"`, so every instrument's pages share one publisher
dict without colliding.

## Options considered

### Where BankNIFTY's live bars come from

| Option | Verdict |
|---|---|
| **An instrument dimension in the store, fed by REST on its own cadence** | **Chosen** by the operator. The honest shape, and the LLD had already named it. Cost: a change to the live-money rendezvous, confined to four lines and defended by §2's test. |
| A dashboard-only cache the fetcher parks a frame in | What I had recommended, as the smaller change: it keeps the store untouched. Rejected in favour of the real dimension, which the next instrument also gets for free. |
| History only, no live session | Zero trading-path change, but the BankNIFTY chart would end at the last `fetch-data` run. Too weak for a "where do I stand now" surface. |
| Reuse the SL Hunting agent's per-bar fetch | No new broker calls at all, but availability would depend on an unrelated strategy being enabled — and that worker returns *before* its BankNIFTY fetch when flat past 10:30, so the chart would silently freeze mid-session. |
| Subscribe BankNIFTY on the websocket feed | `IDX_I` maps to feed code 0 already, so it is subscribable. Rejected: see §3. |

### Where the chart's state block lives

| Option | Verdict |
|---|---|
| **`document["chart"]` keyed by instrument** | **Chosen.** One key, generalized. Its keys double as the "what is being published" answer §4 needs. |
| Keep `chart` as the primary's and add `charts` | Rejected: the primary's block would appear twice in one document. |

## Trade-off analysis

**What this costs.** A change to the store every worker reads, one more REST
call a minute when enabled, a second history load, and a page whose per-instrument
state is now five maps rather than five scalars.

**Why the store change is acceptable.** The dimension is additive: a
keyword-only argument with a default, and the four internal touch points are the
whole surface — `_snapshots` is private and nothing outside the class reads it.
The 647-test master suite passed unchanged before any new test was written,
which is the evidence that call sites were untouched rather than merely
believed to be.

**What is deliberately NOT parameterized.** The feed-health gate (§2). The tick
path (§3). The LTP cache, which is already keyed per instrument but gains no
BankNIFTY index entry — so the header's spot chip stays NIFTY. Every strategy.
`DASHBOARD_BIND_HOST`. The chart is the only consumer of the second instrument.

## Consequences

- **A BankNIFTY REST failure is invisible to trading** and costs one log line at
  WARNING. The chart keeps its last bars.
- **The secondary's history lands later than the primary's**, so
  `historyExhausted` had to become per-instrument. A shared flag would let the
  primary reaching the start of its history switch off the secondary's
  scroll-back before it served a page — the same 404-has-two-meanings trap, one
  level up.
- **`cprSignature` carries the instrument.** Two indices' ladders can agree on
  band count, extent and even a pivot while describing entirely different
  prices, and an unchanged signature skips the redraw — which would leave one
  index's levels drawn over the other's candles.
- **A stored `inst` preference for an instrument this run is not publishing**
  falls back to the primary rather than leaving a permanently empty chart.
- **Switching to an instrument with no payload yet clears every series.**
  Leaving the outgoing candles under the new instrument's name would be the
  chart stating something untrue.
- **`FINNIFTY` is not offered**, though `algo.py fetch-data --index finnifty` and
  `Data Extractors/finnifty_1m_5y_data_fetch_dhan.py` exist. Adding it is now a
  constant, a CSV and a button; it is left out because no CSV has been
  downloaded and nothing in the system reads that index.
