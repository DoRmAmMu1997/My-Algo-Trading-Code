# ADR-0012: Persist per-trade P&L and open positions during the session

**Status:** Accepted
**Date:** 2026-08-11
**Deciders:** repository owner

## Context

Per-strategy results reach the Google Sheet **only at the end of a clean
session**, from `_finalize_flat_session`. That is a single write at a single
moment, and it assumes the process survives to reach it.

On **2026-08-10** it did not. The runner started at 09:13, traded normally, and
the machine hung around 10:30. The operator rebooted and restarted at 10:37. The
09:13 instance died without ever writing a `Result summary`, so:

- the morning's realized P&L existed nowhere but the log file, and had to be
  reconstructed by hand — nine strategies, **−₹10,661.75**; and
- **thirteen positions were open** at the moment of the hang. Their P&L had to
  be reconstructed separately by pulling real Dhan candles for the 10:31 bar
  and marking each leg out manually — a further **−₹4,735.25**.

The existing trades-count guard in the Sheet writer cannot help here. It stops a
`Trades=0` summary from *overwriting* real figures, but a crash writes **no
summary at all** — there is nothing for the guard to reject.

Two distinct things were lost, and they have different requirements:

1. **Realized P&L** — a fact about trades that already closed. Recoverable in
   principle from the log, but only by hand, and only if the log survives.
2. **Open positions** — not merely a number. Resuming one needs its entry fill
   price, its stop and target, its quantity, and its contract identifiers.

## Decision

Add `Dependencies/session_state.py`: a small, atomically-written JSON file kept
current **during** the session.

- **Every trade event is written immediately**, hooked at the runner's existing
  `publish_trade_event` choke point.
- **Open positions are snapshotted on a slow cadence** (default 30s) from the
  supervisor thread, each with `entry_trade_price`, stop, target, quantity,
  contract identifiers and a `last_mark_ltp` read from the shared LTP cache.
- **Resume is opt-in, paper-only, and single-leg-only**
  (`SESSION_STATE_RESUME_ENABLED`, default false).
- **Restart rollover is lossless.** The exact old file is first moved to a
  timestamped recovery archive. Compatible same-day event/P&L bookkeeping seeds
  the replacement session, while old exposure remains archive-only unless it
  passes the explicit resume gate.

## Options considered

### Where to hook the P&L recording

**Option A: one hook in `publish_trade_event` (chosen).** Every worker family
already publishes an event on each entry and exit — 25 call sites funnelling
into one method that is already `try/except`-wrapped so it can never disturb
trading.
**Pros:** one insertion point in live-money code; new worker families are
covered automatically; failures remain exception-isolated. The durability write
is intentionally synchronous through `fsync`, so unlike the Telegram queue it
may briefly block the caller; writes over 250ms produce an operational warning.
**Cons:** the record is the notification event, so its shape is driven by what
Telegram wanted rather than by what a recovery wants.

**Option B: hook each `realized_pnl +=` site.** There are seven, one per worker
family.
**Pros:** records the exact accounting values rather than the event payload.
**Cons:** seven separate edits to live-money exit paths, and an eighth family
added later would silently not persist. Rejected on both counts.

### File format

**Option A: single JSON document, fully rewritten atomically (chosen).** Write a
sibling `.tmp`, `flush` + `fsync`, then `os.replace`.
**Pros:** a reader or a crash can only ever see a complete document; the
per-strategy rollup and the raw event list stay consistent with each other.
**Cons:** O(n) bytes rewritten per event. Irrelevant at this scale — a busy day
is a few dozen events and tens of KB.

**Option B: append-only JSONL.**
**Pros:** O(1) writes.
**Cons:** a torn append leaves a partial final line, so every reader needs
recovery logic; and the per-strategy rollup would have to be recomputed on every
read. Rejected: the write volume never justified it.

#### Amendment (2026-08-11): split into a durable document and a marks sibling

The first live session measured what the single-document design actually costs.
223 writes crossed the 250 ms warning threshold — **210 of them on the supervisor
thread** (median 0.830 s, max **8.732 s**, 85 over one second) against 13 on
trading threads (median 0.414 s). The file was only 79 KB, so this is disk
contention, not payload size; no amount of shrinking the document would fix it.

The obvious fix — stop calling `fsync` on the 30-second snapshot — is **not safe
on its own**. `os.replace` is atomic for the *name*, not for the *data*: a hard
kill during an un-fsynced rewrite can publish a present-but-garbage file. Since
that one document also held `trades[]` and the P&L rollup, a single torn
*snapshot* could destroy the record this ADR exists to protect, turning a
performance fix into a reintroduction of the original incident.

So the state is now **two files**:

| File | Contents | Written by | fsync |
|---|---|---|---|
| `session_state.json` | schema, session date, shutdown flags, `recorded_pnl` / `recorded_trades`, `trades[]` | trade events, clean shutdown, and once at construction | **yes** |
| `session_state.marks.json` | per-strategy live counters and `open_position` with `last_mark_ltp` | the 30 s supervisor snapshot | no |

The durable file is now touched *only* by durable writes, so the snapshot loop
cannot corrupt it however it fails. Losing the marks file costs at most one
snapshot interval of mark data, which this ADR already documented as acceptable.

`load_session_state` merges the pair back into the single-document shape every
existing reader expects, so `resumable_open_positions`, `recorded_realized_pnl`
and the runner's resume path were unchanged. The merge is deliberately
asymmetric: a corrupt **durable** file means no recovery and returns `None`; a
corrupt or missing **marks** file returns the P&L anyway and simply offers no
positions for resume.

Rejected alongside it: moving the trade-event write to a queue and a writer
thread. It would remove the remaining 13 stalls, but only by weakening
durability from "guaranteed before the call returns" to a sub-second window —
which is the guarantee this ADR was written to provide. At a 0.414 s median,
13 times a session, that trade is not worth making.

#### Amendment (2026-08-13): off-thread durable writes, offered but not imposed

The split amendment above **rejected** moving trade-event writes to a writer
thread, on the grounds that a 0.414 s median stall did not justify weakening
durability. That reasoning was incomplete, and the correction is worth stating
plainly: the stalls do not merely add latency, **they drop the market feed**.

On 2026-08-11, **86% of the websocket disconnects (25 of 29) landed within 20
seconds of a session-state write stall**, and `keepalive ping timeout` is the
dominant feed error across the whole log — the signature of a blocked event
loop. Pre-feature days show 1–18 reconnects with zero write stalls. After the
durable/marks split cut stalls from 281 to 36, correlated errors fell from 25 to
3 and total reconnects halved. The remaining stalls are the ~15 trade-event
writes still on trading threads.

Two things about the trade that were previously mis-stated:

- **The data does not land later.** The write takes the same time either way, so
  it reaches the platter at the same wall-clock moment. What changes is that the
  publishing thread is not frozen meanwhile — and it was frozen for precisely
  that interval before, unable to act on anything.
- **A lost queued event still has its log line.** `publish_trade_event` is called
  *after* the `EXIT` log record is emitted, and that log is what
  `_parse_eod_pnl_by_day` reads for the Sheet. The fallback is the same one the
  system used before this module existed.

What genuinely changes: a hard kill can lose events queued in the last write
cycle rather than only the one in flight. Coalescing bounds that — the document
is a full rewrite, so a burst becomes ONE write, which also reduces total fsyncs.

**Decision: implement it, default OFF** (`SESSION_STATE_ASYNC_WRITES`). This is
a genuine reduction of the guarantee this ADR was written to provide, so it is
the operator's call rather than a silent default change — the same treatment
`SESSION_STATE_RESUME_ENABLED` got. `stop_durable_writer()` drains before the
clean-shutdown flag is written, so an orderly end of day is exactly as durable
as the synchronous path.

#### Amendment (2026-08-13): the snapshot loop must be observable

The split's first session exposed a second, independent hole. The supervisor
stopped writing marks at **12:30** while workers traded on to **15:10** — no
exception, no partial file, no log line. It was found hours later by comparing
file mtimes, and even then the cause could not be determined: MainThread emits
nothing during a healthy session, so its silence carried no information.

Worse, the marks left on disk described **11 open positions where 23 were
genuinely open**. Had resume been enabled, it would have restored that
2h40m-stale set as a current book — inventing exposure that had been closed and
omitting exposure that had been opened, which is the exact failure this ADR
exists to prevent, arriving through the file it created.

Two additions:

- **A supervisor heartbeat** (`SESSION_STATE_HEARTBEAT_SECONDS`, default 300s)
  logging workers alive, completed marks writes, marks age, open positions and
  trades recorded. Its presence proves the loop is turning; the gap between the
  last heartbeat and a crash localises where it stopped. `health()` supplies the
  counters and `warn_if_marks_stalled()` raises one ERROR per stall episode.
- **A stale-marks guard.** `load_session_state` records `marks_age_seconds` from
  the two documents' `updated_at` stamps, and `resumable_open_positions` refuses
  every position when that lag exceeds `MAX_RESUMABLE_MARKS_AGE_SECONDS` (300s)
  or cannot be determined. Fail-closed on an unknown lag is deliberate: a book
  of unknown age is not a book.

Note what is NOT guarded: the durable half needs none of this. Trade events are
written by the trading threads themselves, so a frozen supervisor cannot affect
realized P&L — which is why the 12 Aug reconciliation was still exact.

##### Measured after the change (2026-08-12, first session on the split)

| | warnings | median | max | >1s |
|---|---|---|---|---|
| **11 Aug** supervisor (durable, 250 ms threshold) | 254 | 0.909 s | 8.732 s | 114 |
| **11 Aug** trading (durable) | 27 | 0.441 s | 5.111 s | 5 |
| **12 Aug** supervisor (marks, 2 s threshold) | **7** | 2.284 s | 10.065 s | 7 |
| **12 Aug** trading (durable) | **15** | 0.652 s | 2.348 s | 3 |

Total warnings fell from **281 to 22**, and the supervisor path from 254 to 7
against a threshold eight times looser. Trading-thread stalls behaved as
predicted — still present, slightly fewer and with a lower maximum, because the
durable document no longer carries the position blobs.

Two honest caveats. The 12 Aug figures are a partial session (measured at 14:15).
And the worst *marks* write was 10.065 s **without any fsync at all**, which says
the underlying disk contention is real and not purely fsync-driven; what the
split bought is that those seconds now delay supervision instead of a trading
decision, and are labelled as such in the log.

### Whether resume may restore a LIVE position

**Rejected.** In live trading the **broker account** is the authority on what is
open, and the runner already reconciles against it at startup
(`_configure_startup_live_trading`, the startup exposure audit, and
`recover_after_reconciliation`). A JSON file that disagrees with the account is
worse than no file: it would invent exposure the broker does not hold, or mask
exposure it does.

Restoring realized-P&L **bookkeeping** carries no such hazard and is therefore
done for every matching same-day worker in both modes. The resume switch controls
open paper exposure only.

## Trade-off analysis

The snapshot cadence is the real trade-off. Marks are refreshed every 30s, so a
crash loses at most 30 seconds of *mark* movement on an open position. Trade
events are not subject to this — their durable write starts the instant they
happen. Every prior file is archived before replacement (both halves, under one
timestamp), so a restart cannot erase the realized-P&L journal that was
expensive to rebuild by hand.

That cadence trade-off is also what licenses the durable/marks split above: the
marks file may skip `fsync` precisely because its contents were already declared
losable, while the trades and P&L never were.

Writing from the supervisor thread rather than from a new thread is deliberate:
that loop is already awake once a second, never trades, and holds the canonical
worker list. `_position_leg_marks` therefore reads the **LTP cache only** — a
broker round-trip there would stall shutdown supervision for every worker.

Resume is restricted to single-leg `PaperPosition` because hedged and
agent-specific shapes carry per-leg broker state this file deliberately does not
persist. A half-restored hedge is worse than a reported one: the runner logs it
and starts flat so the operator squares it off knowingly.

## Consequences

**Easier:** reconstructing a crashed session (read the file instead of the log);
answering "what had actually banked at the moment it died"; carrying the day's
realized P&L across a mid-session restart so the max-loss kill-switch is not
reset to a fresh full-size budget.

**Harder:** one more file to reason about, and an operator who enables resume
must understand its four preconditions (today's date, unclean shutdown, paper,
single-leg). The file holds live position and P&L detail, so it is gitignored as
operational data.

**To revisit when:** hedged or agent position shapes need resume (they need
their per-leg broker state modelled first), or if the file is ever wanted as an
input to the Sheet writer rather than as an operator-facing recovery record.

## Action items

- [x] `Dependencies/session_state.py`, atomic writes, thread-safe, date-scoped.
- [x] Hook `publish_trade_event`; snapshot from the supervisor loop.
- [x] Record local `clean_shutdown` separately from `results_published`, because
      a flat Ctrl+C or a Google outage is orderly but not exported.
- [x] Archive the prior file before replacement and always carry same-day
      realized-P&L bookkeeping forward independently of exposure resume.
- [x] Paper-only, opt-in resume with an explicit refusal log per skipped record.
- [x] 90% data-safety coverage budget in `scripts/check_coverage_thresholds.py`
      (measured 94.6%).
- [x] Four `SESSION_STATE_*` keys in `env.example`; `check-env` reports zero
      undocumented keys.
