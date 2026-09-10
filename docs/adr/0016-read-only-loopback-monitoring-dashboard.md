# ADR-0016: A read-only loopback dashboard, served by the runner itself

**Status:** Accepted
**Date:** 2026-09-10
**Deciders:** repository owner

## Context

The runner has three observability surfaces and none of them answers the
question an operator actually asks mid-session.

| Surface | Answers | Does not answer |
|---|---|---|
| Append-mode log file | "what happened, in order" | anything, at a glance, across ~29 strategies |
| Telegram alerts | "something just happened" | the current book |
| Google Sheet | "how did the day end" | anything before a clean end of day |

The Sheet is written **once**, at a clean end of day, from a re-parse of the log
file ([`0012`](0012-crash-durable-session-state.md) exists because a mid-session
crash otherwise loses the day's books entirely). So between 09:15 and 15:30 the
only way to know which positions are open, at what mark, and what each strategy
is running at is to read a scrolling log.

Building the answer surfaced a second problem. `_worker_session_state_snapshot`
read only `worker.pos`, so three families that keep exposure elsewhere — the
Delta-0.2 CE/PE spreads, the two long-strangle legs, the SL-Hunting BankNIFTY
mirror — were invisible not just to a UI but to `session_state.marks.json`. That
was fixed first and separately; this ADR covers the dashboard built on top.

The binding constraint is that this code runs **inside a live-money process**.
A monitoring surface that can slow a trading decision, hold a lock a trading
thread needs, or move a risk gate is worse than no monitoring at all.

## Decision

### 1. An in-process stdlib HTTP server on 127.0.0.1, rendering in a browser

`http.server.ThreadingHTTPServer` on a daemon thread, serving one page and two
JSON endpoints. No Flask, no FastAPI, no Streamlit, no new dependency.

### 2. Read-only by construction, not by convention

`do_GET` is the only method implemented; every other verb answers `405` with
`Allow: GET`. Routes are a **frozen whitelist** with no path arithmetic, so
traversal is not blocked so much as unrepresentable. No route reaches a broker,
a worker, or any mutable runner state.

### 3. Loopback only, with no host setting

`DASHBOARD_BIND_HOST = "127.0.0.1"` is a module constant. **The absence of an
`.env` knob is the safety property**: exposing live position data off-box should
require a reviewed code change plus an access token, not a one-line edit at
09:10 on a trading morning. A `Host`-header allowlist closes DNS rebinding.

### 4. One builder thread publishes an immutable blob; HTTP threads never touch a worker

Exactly one thread reads runner state, on a fixed cadence
(`DASHBOARD_REFRESH_SECONDS`, default 1.0s). It renders one JSON payload and
publishes it behind a lock. Request threads hand out those finished bytes, so
the number of open browser tabs cannot add load to the trading path.

Three calls are permanently forbidden from that thread, and a test asserts each:

- **`store.market_data_health.snapshot()`** — it *mutates* `_healthy_streak` and
  `_unhealthy_since`, which drive the 30-second liquidation clock. A monitor
  polling it once a second could pull a real risk decision earlier. Feed
  freshness is derived from `peek_snapshot_meta` timestamps instead.
- **`worker._get_open_position_pnl()` / `_get_option_ltp()` / `_get_underlying_spot()`**
  — all three fall back to `broker.fetch_ltp_map` on a cold cache.
- **`SessionStateStore.snapshot()` on a tick** — it serializes the whole document
  under the lock `record_trade_event` needs, i.e. the lock a trading thread takes
  on the exit path. It is called exactly **once**, at startup, to seed the
  event mirror after a same-day restart.

### 5. Its own bounded event mirror, fed from `publish_trade_event`

Three lines in that choke point append a shallow copy to a `deque` under a lock
nothing else contends for. They sit **between** the session-state write and the
Telegram hand-off, with their own `try`/`except`, so a sink defect can neither
displace the durable write nor be skipped by the `event_queue is None` early
return. The sink is `None` unless the dashboard is enabled.

### 6. `lightweight-charts` vendored, not loaded from a CDN

TradingView's Apache-2.0 standalone build is committed under
`Dependencies/dashboard_assets/vendor/` with its licence, a NOTICE recording the
pinned version and the file's SHA-256, and a `.gitattributes`
`linguist-vendored` marker.

### 7. Honesty over completeness

Any number that cannot be derived truthfully renders as an em dash, never a
zero: no cached mark, indeterminate live exposure, an unpaired exit's entry
time. A strategy with one unpriced position shows `—` for Open *and* Total while
still showing Realized, and the session's Open total is annotated with how many
strategies it could not price.

## Options considered

### Where the page renders

| Option | Verdict |
|---|---|
| **Browser, stdlib HTTP server** | **Chosen.** No dependency, no fight with the console logger, a real chart is easy, and the read-only property is structural. |
| Terminal TUI (Rich / Textual) | Rejected. Adds a dependency, and would fight the existing stderr logging *and* the five raw `print()` entry/exit banners on stdout — the operator would have to give up the live console tail. A candle chart in a terminal is poor. |
| Tkinter window | Rejected. Tkinter must own the main thread on Windows, which is exactly where `main()` runs the shutdown supervisor. Highest risk to the path that must never break. |
| Tail the log in a second window | Rejected. It is what exists today; it does not aggregate across ~29 strategies. |

### Where the work happens

| Option | Verdict |
|---|---|
| **One builder thread, immutable published blob** | **Chosen.** One reader of worker state at a bounded cadence, independent of how many tabs are open. |
| HTTP handler threads read workers per request | Rejected. Load would scale with browser tabs, and every handler thread would be a fresh reader of dataclasses that trading threads mutate. |
| Build on the existing supervisor tick | Rejected. That loop calls `worker.join(timeout=1.0)` per worker, so with ~29 alive workers its period is ~29 seconds, not one. |

### Chart library

| Option | Verdict |
|---|---|
| **Vendored `lightweight-charts` (Apache-2.0)** | **Chosen.** Crosshair, zoom and auto-scaling, with no runtime network dependency. Cost: a minified blob in git plus attribution obligations. |
| Hand-rolled inline SVG candles | Rejected, though close. Zero dependencies, but no zoom or pan, and ~100 lines of chart code to own forever. |
| CDN | Rejected outright. Remote JavaScript in a page displaying live positions, on a box that must keep trading when the internet does not. |

### Reachability

| Option | Verdict |
|---|---|
| **127.0.0.1 only, no knob** | **Chosen.** |
| `0.0.0.0` behind a token, opt-in | Deferred, deliberately. Watching P&L from a phone is genuinely useful, and it can be added later — as a reviewed change, which is the point. |

## Trade-off analysis

**What this costs.** Three lines in a live-money choke point (§5); one
in-process HTTP server; ~200 KB of vendored third-party JavaScript; and a page
whose numbers can be up to one refresh interval stale.

**Why the staleness is acceptable.** The dashboard is informational — the
footer says so, and the broker remains the authority. Nothing trades off this
page.

**Why the torn-read risk is acceptable.** The collector reads position
dataclasses that trading threads mutate, without a lock. The load-bearing
invariant is that `active` is never flipped in place: entry and exit both
*replace* the whole position object, so a reader that binds the object to a
local once and re-checks `.active` on that local sees a coherent snapshot. The
worst case is one extra or one missing row for one refresh interval. Taking a
lock instead would put a monitoring surface on the trading path, which is the
thing this design refuses to do.

**Why not simply read `session_state.json` from a separate process.** It holds
no OHLC, so the chart would need its own broker feed; its `open_position` lags
30 seconds; and the operator's machine has a DRAM-less SSD where multi-second
disk stalls are routine, so a once-a-second read of a growing JSON file is
exactly the I/O this system should not add.

## Consequences

- **Off by default.** `DASHBOARD_ENABLED=false`. A bind failure, a bad port, or
  any exception during startup logs and leaves the session trading without it.
- **A second runner cannot silently share the port.** `allow_reuse_address` is
  `False`, so the second bind fails rather than showing the older process's
  numbers.
- **Any local process on the box can read the page.** Accepted for a
  single-user machine; it is the reason there is no LAN bind.
- **The console log stays clean.** Both `log_message` and `handle_error` are
  routed to a logger. Stray stderr writes would bypass the secret-redaction
  filter and pollute the log the end-of-day Sheet write parses.
- **Shutdown ordering is fixed.** The dashboard is stopped after flatten →
  broker-confirmed flat → finalize → Sheet → `mark_clean_shutdown`, so it can
  delay none of them, and the operator can watch the reconciliation happen.
- **CSP allows inline styles.** The vendored chart library injects its own
  `<style>` element and a hash would have to be re-derived on every upgrade.
  `script-src 'self'` and `connect-src 'self'` stay strict, which is where the
  risk actually lives.
- **A LAN bind is now a one-file change** (`DASHBOARD_BIND_HOST`) plus the token
  and review this ADR defers. That is intentional friction, not an oversight.

## Action items

- [x] `_owned_open_positions()` hook, so the three multi-position families are
      visible to both the marks file and the dashboard.
- [x] `owned_positions` wired all the way through `session_state.py` — its
      `update_worker_snapshot` copies a fixed key list, so a new key that is not
      named there is silently dropped.
- [x] `PaperPosition.entry_timestamp`, matching `HedgedPaperPosition`.
- [x] Vendored library recorded with version and SHA-256, `linguist-vendored`.
- [ ] Run at least one full paper session with `DASHBOARD_ENABLED=true` before
      enabling it during live trading.
