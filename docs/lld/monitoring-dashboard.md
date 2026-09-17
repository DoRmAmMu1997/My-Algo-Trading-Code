# Low-level design: the read-only monitoring dashboard

**Status:** implemented, opt-in, off by default (`DASHBOARD_ENABLED=false`).
**Decision record:** [`ADR-0016`](../adr/0016-read-only-loopback-monitoring-dashboard.md).

A browser page the runner serves itself, answering the one question the log
file, Telegram and the end-of-day Google Sheet do not: **where do I stand right
now?** Open positions with live marks, today's closed trades grouped by
strategy, per-strategy running P&L, and a NIFTY candle chart.

## The safety contract

This runs inside a process that places real orders. Four properties hold, and
each has a test that fails if it stops holding.

| Property | How it is enforced |
|---|---|
| It cannot place, cancel or modify an order | `do_GET` is the only method implemented; every other verb answers `405 Allow: GET`. No route reaches a broker or a worker. |
| It cannot be reached from another machine | Binds `127.0.0.1`, and `DASHBOARD_BIND_HOST` is a **module constant with no `.env` knob**. A `Host` allowlist answers `403` to anything else (DNS-rebinding defence). |
| It cannot slow a trading decision | All work happens on one dedicated daemon thread; HTTP handler threads only hand out already-rendered bytes. Every read is an in-memory cache read or an attribute read. Assets are loaded once at start, so there is no per-request disk I/O. |
| It cannot move a risk gate | `market_data_health.snapshot()` is never called — it **mutates** the healthy-streak and unhealthy-since fields driving the 30 s liquidation clock. Neither are `_get_open_position_pnl()`, `_get_option_ltp()` or `_get_underlying_spot()`, which fall back to the broker on a cold cache. |

`Tests/test_nifty_multi_strategy_master.py::TestDashboardCollector::test_building_a_document_never_reaches_the_broker_or_a_mutating_gate`
asserts the last two directly, with mocks.

## Components

```
trading threads ──publish_trade_event──► DashboardEventSink   (deque + own lock)
                                                  │
SharedMarketDataStore  ──cache-only reads─────────┤
worker objects         ──attribute reads──────────┤
                                                  ▼
                          DashboardBuilderThread  (1 daemon thread, 1 Hz)
                                                  │ one immutable JSON blob + ETag
                                                  ▼
                     ThreadingHTTPServer @127.0.0.1  ──►  browser polls /api/state
```

| Unit | File | Responsibility |
|---|---|---|
| Chart indicators | `Dependencies/dashboard_indicators.py` | CPR from a truncated prior session, session VWAP, stochastic %K/%D, the forming higher-timeframe bucket, and the `/api/chart` document. Pure: pandas allowed, nothing else. The VWAP and stochastic helpers are INJECTED so the chart uses the strategies' own objects. |
| Pure shaping | `Dependencies/dashboard_snapshot.py` | Pairs the event stream into closed trades and open-entry times; rolls up per-strategy and session totals; renders the document. No threads, no I/O, no clock, no `.env`. |
| Transport | `Dependencies/dashboard_server.py` | Event sink, publisher, builder thread, HTTP server, asset whitelist. Reads **no** configuration of its own. |
| Collector | `nifty_multi_strategy_master.py` (after `_session_state_snapshots`) | The only code that reaches into live workers and the store. It lives in the master because a `Dependencies/` module doing that would have to import the master back. |
| Page | `Dependencies/dashboard_assets/` | `index.html`, `dashboard.css`, `dashboard.js`, and the vendored chart library under `vendor/`. |

A test asserts that neither `Dependencies/` module imports `os` or reads an env
key, using the repo's own AST auditor (`check_env_config.env_keys_read_by`) so
it can never disagree with `algo.py check-env`.

## Where the data comes from

### Closed trades and open-position entry times

From the runner's own trade events, mirrored into `DashboardEventSink` by three
lines in `publish_trade_event`. Those lines sit **between** the session-state
write and the Telegram hand-off, with their own `try`/`except`, so a sink defect
can neither displace the durable write nor be skipped by the
`event_queue is None` early return. The sink is `None` unless the dashboard is
enabled, so with it off the choke point costs one attribute read.

Not read from `SessionStateStore` on a tick, for three reasons: `snapshot()`
serializes the whole document under the lock `record_trade_event` needs, it
disappears entirely when `SESSION_STATE_ENABLED=false`, and its
`trades_recorded` gate would freeze at the 5000-event cap. It **is** read once,
at startup, to seed the sink so a same-day restart still shows the carried-forward book.

### ENTRY → EXIT pairing

Three families publish several independent positions under **one** strategy
name, so pairing cannot be FIFO. The key is `(strategy, frozenset(leg symbols))`,
resolved down a ladder that records its own confidence:

| Rank | Rule | Why it is needed |
|---|---|---|
| `EXACT` | same strategy and same symbol set | Separates Delta-0.2's CE and PE spreads, and SL Hunting's NIFTY leg from its BankNIFTY mirror |
| `OVERLAP` | largest non-empty symbol intersection | A partial-leg close. Ranked **above** direction because SL Hunting's two legs share a direction but never a symbol |
| `DIRECTION` | oldest open entry with the same direction | Backstop for Delta-0.2 and the strangle, whose direction is `CE`/`PE` |
| `SOLE_OPEN` | the strategy has exactly one open entry | No ambiguity left |
| `UNPAIRED` | nothing matched | Entry time renders `—`, never a guess |

`EXIT_FAILED` **closes nothing.** The broker did not confirm, the runner keeps
retrying, and the position is still exposure — so it flags the matching entry
`EXIT FAILED — STILL OPEN` and emits a notice. The other five non-trade actions
(`INDETERMINATE_EXPOSURE`, `UNHEDGED_LEG_OPEN`, `UNHEDGED_LEG_CLOSED`,
`MARKET_DATA_AUTO_SQUARE_OFF`, `SHUTDOWN_DEGRADED`) become notices and never
touch pairing.

An unpaired EXIT is still a complete row: the exit event itself carries entry
price, exit price and realized P&L. Only the entry *time* is unknown.

The residue of unmatched entries is **not** the open book — it can hold ghosts
across a restart. The live worker read is always the sole authority on what is
open; the ledger only supplies times.

### Open positions and marks

Through `_owned_open_positions()`, so the Delta-0.2 spreads, the strangle legs
and the SL-Hunting mirror all appear (see
[`reporting-and-observability.md`](reporting-and-observability.md)). Marks come
from `store.get_ltp_by_secid`, which never calls the broker; a mark is "stale"
when the same read with `max_age_seconds=DASHBOARD_STALE_MARK_SECONDS` returns
nothing while the unbounded read returns a price.

### Chart indicators and the 1m/5m/D toggle

Three overlays, and a toggle because several strategies derive 5-minute bars --
plus a Daily timeframe for the next-session macro read.

| Indicator | Source | Note |
|---|---|---|
| CPR | `dashboard_indicators.chart_cpr` (today) + `dashboard_history.cpr_segments` (every earlier day) | **Chart-only.** Prior session 09:15-15:15 inclusive -- high, low AND close -- against the strategies' full session. One band **per day**, spanning that day alone, from that day's predecessor. On the Daily timeframe the ladder is **monthly** instead. Thirteen levels in four toggle groups: `core` (pivot/BC/TC) and `pd` (PDH/PDL, the prior session's traded extremes) on by default, the two R/S ladders opt-in. All solid, matching VWAP. See [`ADR-0017`](../adr/0017-chart-only-cpr-on-a-truncated-prior-session.md). |
| VWAP | `regime_common.attach_session_vwap`, injected | The strategies' own object. Always an equal-weight proxy in live running -- the index feed carries no volume -- so it is labelled `VWAP*` with a footnote. |
| Stochastic %K/%D | `misc_strategy_common.stochastic`, injected | The same TA-Lib `STOCH` and the same `STOCHASTIC_*` periods the Stochastic Oscillator strategy trades on. Own pane, 80/20 guides. The **Daily** series carries its own, computed on daily candles -- the live payload's is a statement about one-minute bars. |

The helpers are reached with `load_module` under **bare** names, which returns
`sys.modules[name]` when already loaded -- so these are the very objects the
strategies use, not copies. A prefixed alias would create a second instance
that type-checks, runs, and diverges the moment anyone retunes a helper; a
test asserts identity against `sys.modules`.

Indicators are computed on **completed bars**; the forming candle carries no
indicator point. Cheaper, and truer to what the strategies evaluate.

Both timeframes ride in ONE `/api/chart` payload, so switching is instant from
data the browser already holds -- measured at zero network requests -- and the
transport keeps its single publisher slot and zero-argument builder.
`resample_ohlc_from_1m` is used unchanged for the completed 5-minute bars; the
forming bucket is aggregated separately and concatenated, so the resampler's
"completed buckets only" guarantee stays intact for the strategies that depend
on it.

Two things that look like details and are not:

- The 5-minute window is a **row** tail, not a time window. Anchored to
  "now minus 575 minutes" it reaches past midnight at 09:15 and excludes the
  entire prior session, leaving the pane near-empty every morning while the
  1-minute chart beside it shows yesterday's close.
- The 5-minute series is **de-duplicated** on append. The bulk rebuild can
  already hold the bucket the incremental path re-derives, and
  lightweight-charts requires strictly ascending times -- one repeat makes it
  silently drop bars, with nothing in the console to say why.

### Back-history

`Dependencies/dashboard_history.py` reads the CSV `algo.py fetch-data` writes
and serves it as the chart's scroll-back. It is a module of its own because
`dashboard_indicators` is pure -- a test AST-asserts it never imports `os` --
and nothing that opens a file can live there.

It never touches the market-data store, a worker, the broker or the session
state. It reads one file. That is what lets the chart gain five years without
adding a single call to the trading path.

Measured on the real five-year download (459,152 bars, 28.8 MB, 1,226
sessions):

- **~45 MB of JSON**, so it is **paged**: 2,000 bars a page (~160 KB), page 0
  the NEWEST, because that is the order a browser wants them -- scroll left,
  ask for 0, then 1. A missing key IS the "nothing older" answer, so the
  endpoint needs no bounds arithmetic that could disagree with the store.
- **29.5 seconds** to read and render 277 pages, which is why it loads on a
  thread of its OWN. On the builder thread the live view would stop rebuilding
  for half a minute and the page would show its staleness banner.
- **~63 MB of RSS.** Real; the dashboard stays opt-in and off by default.

A missing or unreadable CSV is not an error: the chart shows the live session
alone, exactly as it did before history existed, and the log says how to
populate it.

The loader repeats the extractor's session and weekday checks rather than
trusting the file. A resumed download can leave one CSV cleaned by two rule
versions -- which is exactly what happened: 509 weekend rows survived in the
chunks fetched before the extractor learned the weekday rule.

History and the live window are spliced in the BROWSER, at draw time. History
is cut at the first live timestamp and the live copy wins, because it is the
one that keeps updating. Prepending never refits: `fitContent` exists for
timeframe switches, and firing it when a page arrives would yank the viewport
back to the whole series. Prepending N bars shifts every logical index by N,
so the range is moved by the same N.

### The chart

`store.get("1")` copies a ~2,200-row DataFrame, which is the wrong price for a
once-a-second "anything new?" question. `SharedMarketDataStore.peek_snapshot_meta`
answers it from three already-materialized attributes under the same lock,
mutating nothing:

- signature unchanged → **no copy at all**;
- signature changed, same `source_candle_ts` → copy, keep the last row only
  (the forming candle, ~120 bytes on the wire);
- `source_candle_ts` advanced → copy, re-serialize the tail, bump
  `series_version`, and the browser refetches `/api/chart` — about 375 times a
  session rather than 22,500.

Candle timestamps are naive IST wall-clock labelled **UTC** on purpose:
lightweight-charts renders every timestamp as UTC and has no timezone setting,
so this makes the axis read 09:15, 09:16 … Localizing to IST instead would draw
the session starting at 03:45. There is no volume histogram; the frame carries
no volume column and synthesising one would be a fabrication.

## Endpoints

| Route | Notes |
|---|---|
| `GET /` `/dashboard.css` `/dashboard.js` `/vendor/…js` | Frozen whitelist, loaded into memory once at start |
| `GET /api/state` | The whole document; `ETag` + `If-None-Match` → `304`; `Cache-Control: no-store` |
| `GET /api/chart` | The full candle array; fetched only when `series_version` moves |
| `GET /api/history?tf=<1\|5\|D\|cpr>&page=<n>` | One page of back-history, or the CPR ladders. The only endpoint that reads the request, so it validates rather than infers: `tf` from a frozen set, `page` a plain non-negative integer, blanks included (`keep_blank_values`, so `?page=` is refused rather than read as absent). Past the start of history → `404`, which is how the browser learns to stop asking. No `ETag`: every response here is `no-store`, so a `304` would leave the browser with no body to draw |
| anything else | `404` |
| any non-GET | `405` + `Allow: GET` |

Every response carries `nosniff`, `Referrer-Policy: no-referrer`,
`X-Frame-Options: DENY` and a CSP of `default-src 'none'; script-src 'self';
style-src 'self' 'unsafe-inline'; connect-src 'self'`. There is never a CORS
header. Inline **styles** are allowed only because the vendored chart library
injects its own `<style>` element and a hash would have to be re-derived on
every upgrade; scripts and connections stay strict, which is where the risk is.

## Honesty rules

A number that cannot be derived truthfully renders as an em dash, never a zero.
The CSS styles a dash distinctly from a value for exactly that reason.

| Situation | Rendered |
|---|---|
| No cached LTP for a leg | LTP `—`, Unrealised `—` |
| A hedged pair with only one leg priced | Unrealised `—` — the unpriced leg is the one that offsets the other |
| `live_leg.exposure_indeterminate` | Unrealised `—` plus an `INDETERMINATE` flag; the *quantity* is unknown |
| A strategy with any unpriced position | its Open **and** Total `—`; Realized still shown |
| Session Open total | the sum of what could be priced, annotated `(N unpriced)` |
| Unpaired closed trade | entry time `—` plus the pairing marker |

## Degraded states

| State | Behaviour |
|---|---|
| `SESSION_STATE_ENABLED=false` | Everything still works through the sink; the closed-trade pane notes it covers this process only |
| Pre-open | Empty tables that say so, "waiting for the first candle" |
| Cold LTP cache | Open rows show `—` for LTP and Unrealised; Realized still shown |
| Builder raised | The **last good** payload keeps serving; five identical polls in a row surface a "frozen" banner |
| Port already bound | `start_dashboard` returns `None`, the runner logs a warning and trades on |
| Runner exited | The page keeps the last data on screen behind a dimmed banner and retries every 5 s |

## Lifecycle

Started in `main()` between the session-state block and
`_start_and_supervise_runtime_threads` — after the virtual-trading filter and
live-mode decisions, and **before any worker thread starts**, so the sink is
attached before the first event can be published. Any exception detaches the
sink from every worker and continues.

Stopped **last**, after `_wait_for_shutdown_account_flat` → `_finalize_flat_session`
→ the Google Sheet write → `mark_clean_shutdown`, and immediately before
`stop_event.set()`. It never reads `stop_event`, so it stays visible through the
whole reconciliation — which is when an operator most wants to watch. Its
threads are daemons, so `DASHBOARD_SHUTDOWN_TIMEOUT_SECONDS` is a courtesy, not
something process exit depends on. A source-order test pins that sequence.

## Layout

The chart pane fills the viewport below the sticky header, TradingView-style;
the tables sit below the fold. `#chart` is `flex: 1` inside a flex-column pane,
so the chart absorbs whatever the head and the provenance caption leave -- the
head wraps freely at narrow widths and nothing needs recalculating.

The one number CSS cannot know is the header's height, because `#header` wraps
(62px open, 94px on a narrow window). `dashboard.js` measures it and publishes
`--header-h`. It is re-measured **from the poll loop**, not left to a
`ResizeObserver` alone -- see the traps below.

## Traps worth remembering

**A `ResizeObserver` that never fires leaves no trace.** In at least one
embedded browser the observer does not run at all -- not even the initial
callback the spec guarantees on `observe()`. Anything layout-critical hanging
off one needs a second, unconditional path: the chart's range fit and the
header-height measurement both retry from the once-a-second poll, which costs a
`getBoundingClientRect` and writes only on a change.

**A new CPR group must be added to the `cprSignature` memo array** in
`renderCprLines`. That memo skips the redraw when its signature is unchanged,
so a group left out of it sets its preference and then draws nothing -- the
checkbox appears to do nothing at all.

**A wired element id missing from `index.html` blanks the page.** The control
loop does `const box = el(id)` with no null guard, so one absent id throws
inside the IIFE and nothing renders.
`test_every_element_the_page_script_looks_up_exists_in_the_markup` covers both
lookup shapes, including the `["element-id", "prefKey"]` pairs where the id
never appears beside `el(`.


**`fitContent()` against a zero-width container silently does nothing** and
leaves a nonsense bar spacing behind, drawing every candle squeezed into a few
pixels. On a first paint, or in a tab that is still hidden, the container can
genuinely have no width. A `ResizeObserver` fits the range when layout gives
the element a size, with a per-render retry behind it.

**pandas 3 defaults to MICROSECOND datetime resolution.** The obvious
`astype("int64") // 10**9` for epoch seconds therefore yields values a
thousand times too small and draws the whole session in 1970. `bar_records`
integer-divides a `Timedelta` instead, which is resolution-independent, and a
test asserts it matches the master's `_bar_record` bar for bar.


`DashboardBuilderThread` keeps its stop flag in `_stop_event`, **not** `_stop`.
`threading.Thread` has a private `_stop()` *method* that CPython 3.12's
`join()` calls through `_wait_for_tstate_lock`, so shadowing it with an
`Event` makes every join raise `'Event' object is not callable`. Python 3.13
removed that call path, so the bug passed the local run *and* the 3.13 CI leg
while failing only on 3.12 — and `DashboardServer.stop()` would have failed
the same way in production on 3.12.

`test_the_builder_thread_shadows_nothing_that_threading_owns` guards it, and
deliberately unions the running interpreter's `dir(threading.Thread)` with the
names newer versions deleted: reflection alone is blind to this on exactly the
interpreter that is not failing.

**A scroll-back trigger that depends on a fit.** Loading the first history
page from a visible-range event looks natural and silently never fires: the
event only arrives once a FIT has landed, and `applyPendingFit` refuses a
zero-width container -- a state the chart can sit in indefinitely, because the
`ResizeObserver` meant to catch the layout has been measured NOT firing in some
panes. The first page is fetched outright instead.

**A 404 has two meanings.** Past the start of history it means "nothing older";
before the server has finished building -- 29.5 seconds on the five-year file --
it means "not yet". Treating the second as the first disables scroll-back for
the whole session. It only counts as exhausted once a page has actually landed.

**A history page can add nothing.** If every bar in it falls inside the live
window the merge drops all of it: the chart does not grow, the viewport does
not move, and no range event arrives to ask for the next page -- scroll-back
just stops. Not reachable at the shipped sizes (2,000 against 375), but
`DASHBOARD_CHART_BARS` goes to 2,200. A page that adds nothing pulls the next.

**CPR bands must be clipped to the bars loaded.** Every series shares one time
scale built from the UNION of their times, so handing over five years of bands
while a week of candles is loaded stretches the scale across five years of
empty chart.

**The memo cannot key on one pivot.** With hundreds of bands it has to key on
the set's extent and its newest pivot, or a new session, a newly loaded page
and a timeframe switch all look identical to it.

**Assets are read into memory at startup.** Editing `dashboard.js` on disk does
nothing to a running server; restart it, or spend a while debugging code that
is not the code being served.

## Vendored code

`Dependencies/dashboard_assets/vendor/lightweight-charts.standalone.production.js`
(TradingView, Apache-2.0), pinned and recorded with its SHA-256 in
`NOTICE-lightweight-charts.md` beside the licence text. `.gitattributes` marks
the folder `linguist-vendored`. The library's attribution logo is left enabled
and the page carries a footer credit.

## Configuration

Seven keys, all read in the master and clamped at read time so a typo in a
monitoring knob cannot stop a session starting. See
[`configuration.md`](configuration.md) and `Dependencies/env.example`. There is
deliberately no host key.

## Tests

| Suite | Covers |
|---|---|
| `Tests/Dependencies/test_dashboard_snapshot.py` | Every pairing confidence, the Delta-0.2 / SL-Hunting-mirror / re-entry shapes, `EXIT_FAILED` not closing, malformed events, the honesty rules, NaN rejection |
| `Tests/Dependencies/test_dashboard_server.py` | Real loopback socket: 200/304/403/404/405, security headers, no CORS, empty stderr, root logger untouched, bind-in-use, no config read |
| `Tests/Dependencies/test_dashboard_indicators.py` | CPR truncation and every degenerate session shape; **the equality test that pins the CPR algebra against `_add_daily_cpr`**; VWAP and stochastic equality with the strategies' helpers; the forming bucket; and every fixture rendered through `render_document_bytes` |
| `Tests/test_nifty_multi_strategy_master.py` | The collector, the chart cache, the sink wiring in `publish_trade_event`, the "never reaches the broker or a mutating gate" assertion, the recompute-cadence guards, and fail-soft when an indicator raises |

`Dependencies/dashboard_snapshot.py` and `Dependencies/dashboard_indicators.py`
each carry a 90 % coverage budget in `scripts/check_coverage_thresholds.py`
(the latter measures 98.9 %). `dashboard_server.py` deliberately does
not: socket-bound code has branches that cannot honestly be covered, and a
budget you cannot meet is worse than none.
