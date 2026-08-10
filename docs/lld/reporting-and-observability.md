# LLD — Reporting and observability

**Owns:** `TelegramMessageWorker`, `format_trade_message`, `_publish_eod_summary`,
`_parse_eod_pnl_by_day`, `_compute_pnl_sheet_updates`, `_update_pnl_google_sheet`,
`setup_logging` (master file)

---

## 1. Responsibility

Tell the operator what happened — during the session (Telegram), after it
(Google Sheet), and forensically (the log) — **without ever being able to affect
trading**.

Every design choice here follows from that last clause. Reporting is
best-effort, off the trading path, and a safe no-op when unconfigured.

---

## 2. Telegram alerts

```
 worker (any)                     queue.Queue                TelegramMessageWorker
 ────────────                     ───────────                ─────────────────────
 entry / exit  ──► event dict ──► put_nowait()  ──────────►  get() → format → POST
                                  (unbounded, non-blocking)      │
                                                                 └─ failure: log and continue
```

The queue is the whole point. Telegram latency or downtime can never block a
trading decision, because the worker thread never waits on the network — it
hands off a dict and returns to its loop.

Alerts carry the strategy, the exact option instrument(s), lot size, entry and
exit price, and P&L. Hedged spreads show both legs. `_format_inr` and
`_execution_mode_parts` / `_combined_execution_mode` keep the formatting
consistent, including the PAPER / LIVE / MIXED labelling.

Disabled (`TELEGRAM_ENABLED=false`, the default) it is a cheap no-op — the
worker is not started and events are dropped.

---

## 3. End-of-day P&L to Google Sheets

On a clean end of day, the runner writes each strategy's realised P&L into a
tracker sheet: one row per strategy, one column per calendar day.

```
 rotating log file (append mode)
        │
        ▼
 _parse_eod_pnl_by_day(log_path, today)     ← parses the RUN'S OWN LOG
        │                                     _asctime_in_pnl_window filters to
        │                                     the session; _normalize_pnl_strategy_name
        │                                     maps log labels to sheet rows
        ▼
 _compute_pnl_sheet_updates(values, pnl_by_day, today)
        │   ├─ overwrite today's cell
        │   └─ backfill BLANK earlier-this-month cells from the same log
        ▼
 gspread (OAuth user token) ──► the sheet
```

### 3.1 Why it parses a log instead of reading memory

Reading in-memory totals would be simpler and would also be wrong in the case
that matters: a session that ended messily. The append-mode log is the durable
record of what actually happened, so parsing it means a partial session still
reports honestly, and blank earlier-in-the-month cells can be backfilled without
re-running anything.

### 3.2 Row labelling

PAPER results use the existing row labels in column A (e.g. `Renko Strategy`).
LIVE and MIXED results use **separate** rows — `Renko Strategy [LIVE]`,
`Renko Strategy [MIXED]` — so real-money outcomes can never contaminate paper
history. Unmatched strategies are skipped with a warning rather than written to
a guessed row.

Auth is OAuth user-token via `gspread` (`GSHEET_OAUTH_CLIENT_FILE`,
`GSHEET_OAUTH_TOKEN_FILE`, both gitignored). Leave `GSHEET_ID` blank to disable —
a safe no-op that never disturbs shutdown.

---

## 4. Logging

`setup_logging()` returns the root logger with two things attached:

1. Console + append-mode file handlers.
2. **`install_redaction_filter(environment_secrets(os.environ))`** — see
   [`risk-and-safety.md`](risk-and-safety.md) §8. Every record is scrubbed,
   including lazy `%s` args and exception tracebacks.

Conventions:

- Library code uses a module-level `logging.getLogger(__name__)`, **never
  `print()`**.
- Do not hand-redact new call sites; the root filter covers them.
- `*.log` is gitignored.

The log is load-bearing, not decorative: the EOD sheet is derived from it (§3),
so a change to a trade log line's format is a change to the P&L pipeline.

---

## 4a. Crash-durable session state

The Sheet write in §3 happens **once**, at the end of a clean session. A process
that dies before it writes nothing at all — and the trades-count guard cannot
help, because there is no summary to reject. On 2026-08-10 a machine hang cost a
whole morning's books plus thirteen open positions, all rebuilt by hand.

`Dependencies/session_state.py` keeps a JSON file current *during* the session:

| What | When it is written | Why |
|---|---|---|
| Every trade event | Immediately, from `publish_trade_event` | Realized P&L is what a crash makes expensive to rebuild |
| Per-strategy rollup (`recorded_pnl` / `recorded_trades`) | With each `EXIT` event | Answers "what had banked" without re-summing the events |
| Open positions | Every `SESSION_STATE_SNAPSHOT_SECONDS` (30s) from the supervisor loop | Entry fill price, stop, target, quantity, contract ids and `last_mark_ltp` — everything a resume needs |
| `clean_shutdown: true` | After runner exposure is proven flat and local shutdown completes | Its **absence** is the crash signal for open-position recovery |
| `results_published: true` | Only after a real Google Sheet cell batch succeeds | Distinguishes local cleanup from external reporting success |

Properties that make it trustworthy:

- **Atomic** — `.tmp` + `flush` + `fsync` + `os.replace`. A reader never sees a
  partial document.
- **Restart-safe** — an existing file is moved to a timestamped recovery archive
  before the replacement run writes anything. Compatible same-day trades and
  P&L totals seed the new session; old exposure never carries implicitly.
- **Never raises** — every entry point swallows and logs. Persistence failing is
  a reporting problem; the first failure logs loudly, then stays quiet.
- **Cache-only marks** — `_position_leg_marks` reads the shared LTP cache and
  never the broker, because it runs on the supervisor thread.
- **Resume is opt-in and narrow** — `SESSION_STATE_RESUME_ENABLED` (default
  false), and a record is offered back only if it is from **today**, from an
  **unclean** shutdown, **paper**, and a single-leg `PaperPosition`. Live
  positions are never restored: the broker account is the authority there and
  the runner already reconciles against it. Everything refused is logged by name.

Same-day `recorded_pnl` / `recorded_trades` bookkeeping is always carried into
matching workers, in paper and live modes, so a restart cannot grant a fresh
daily max-loss budget. `SESSION_STATE_RESUME_ENABLED` controls only whether an
eligible open paper position is reconstructed.

See [ADR-0012](../adr/0012-crash-durable-session-state.md). The file is
gitignored — it holds live position and P&L detail.

---

## 5. Failure behaviour

| Failure | Effect on trading |
|---|---|
| Telegram down / token wrong | None. Logged, event dropped. |
| Google Sheet unreachable | None. Shutdown completes; the day can be backfilled on a later run. |
| Sheet row label unmatched | That strategy is skipped with a warning; nothing is written to a guessed row. |
| Log file unwritable | Console logging continues; the EOD sheet loses its source for that session. |
| Session state file unwritable | None. First failure logs loudly; trading continues without crash recovery for that session. |
| Session state durable write exceeds 250ms | Trading still waits for that `fsync` so the event is genuinely durable; a warning identifies local-disk latency for operator action. |
| Session state file corrupt on read | Ignored, logged; the run starts with no recovery rather than refusing to start. |

---

## 6. Testing

- `Tests/test_nifty_multi_strategy_master.py` covers `format_trade_message`,
  the queue worker, `_parse_eod_pnl_by_day`, `_compute_pnl_sheet_updates` and
  the PAPER/LIVE/MIXED labelling.
- `Tests/Dependencies/test_secret_redaction.py` covers the redaction filter.
- `Tests/Dependencies/test_session_state.py` covers the state module's own
  rules (atomicity, concurrency, resume eligibility); the runner's half — the
  `publish_trade_event` hook, snapshots and resume — is in
  `TestSessionStatePersistence` in the master suite.

Neither Telegram nor Google Sheets is contacted in tests.

---

## 7. Gaps

- **No metrics or health endpoint.** Observability is a log file plus Telegram.
  For a single-operator, single-session system that is proportionate; it would
  not be if the runner ever ran unattended.
- **P&L is derived from log parsing**, so log-line format is an implicit
  contract. If the runner ever grows a structured event store, this is the first
  thing that should move to it.
