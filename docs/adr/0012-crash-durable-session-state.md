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
happen. Every prior file is archived before replacement, so a restart cannot
erase the realized-P&L journal that was expensive to rebuild by hand.

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
