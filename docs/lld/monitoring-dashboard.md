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

## A trap worth remembering

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
| `Tests/test_nifty_multi_strategy_master.py` | The collector, the chart cache, the sink wiring in `publish_trade_event`, and the "never reaches the broker or a mutating gate" assertion |

`Dependencies/dashboard_snapshot.py` carries a 90 % coverage budget in
`scripts/check_coverage_thresholds.py`. `dashboard_server.py` deliberately does
not: socket-bound code has branches that cannot honestly be covered, and a
budget you cannot meet is worse than none.
