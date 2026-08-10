# LLD — Master runner: process lifecycle and thread supervision

**Owns:** `Nifty Multi Strategy Front Test - Master File.py` (`main()` and the
startup/shutdown helpers around it)
**Depends on:** every other component
**Read first:** [`../hld/system-overview.md`](../hld/system-overview.md) §6 (concurrency), §7 (failure model)

---

## 1. Responsibility

`main()` is the only place that knows the *order* in which the system must come
up and go down. Its contract:

1. Nothing that can place an order starts before configuration is validated and
   the broker book is proven clean.
2. Nothing reports a clean shutdown until every tracked leg is closed and the
   runner's execution ledger is broker-reconciled flat. A separate account-wide
   audit then warns about manual or otherwise untracked exposure.

Everything between those two statements is supervision.

---

## 2. Startup sequence

```
main()
  │
  1. setup_logging()
  │     └─ install_redaction_filter(environment_secrets(os.environ)) on the ROOT logger
  │        ── so every record from here on is scrubbed, including tracebacks
  │
  2. read + validate configuration
  │     ├─ _env_str / _env_bool / _env_int / _env_float      (plain knobs)
  │     ├─ _scaled_int / _scaled_float                       (size-bearing knobs)
  │     └─ _live_config_errors(...)  ── malformed size knobs BLOCK that
  │                                     strategy from live; paper falls back to 1
  │
  3. _configure_startup_live_trading(...)
  │     ├─ LIVE_TRADING_ENABLED false      ──► everything paper, skip the rest
  │     ├─ _select_execution_client(LIVE_BROKER)
  │     │     └─ unknown name ──► FAIL CLOSED: live disabled, paper only
  │     └─ _cpr_ai_startup_errors() etc. for optional agents
  │
  4. startup exposure audit          (Dependencies/startup_exposure.py)
  │     ├─ read open ORDERS  ─┐
  │     └─ read open POSITIONS┴─► both must be clean for live workers to start
  │        (read-only: never adopts, cancels, recovers or flattens)
  │     └─ _enqueue_startup_exposure_alert(...) tells the operator either way
  │
  5. build the worker list
  │     └─ filter by _strategy_virtual_trading_enabled(name)
  │        ── <PREFIX>_VIRTUAL_TRADING=false means the thread never starts,
  │           so the strategy does neither paper nor live
  │
  6. _start_and_supervise_runtime_threads(...)
        ├─ market data producer thread   (exactly one, see market-data.md)
        ├─ TelegramMessageWorker         (if enabled)
        └─ N strategy worker threads
```

### 2.1 Why the exposure audit is read-only

The temptation is to have the runner "clean up" a stale position it finds at
startup. It deliberately does not. A position in the account may belong to the
operator's own manual trading in the same account — the runner has no way to
tell whose it is, and flattening someone else's position is worse than refusing
to start. The audit therefore reports and refuses; a human decides.

This is also why end-of-day gates are measured against **the runner's own
ledger**, not the account books.

---

## 3. Supervision

`_start_and_supervise_runtime_threads` keeps the process alive while threads
run, and watches for the conditions that should end the session. Worker threads
are cooperative: they check a shutdown flag on their own poll cadence rather
than being interrupted, because interrupting a thread mid-order is exactly the
state the whole safety model exists to avoid.

`_request_worker_shutdown` sets the flag **per worker**. The scope is deliberate:
one strategy hitting its daily max-loss stops that strategy, not the session.

---

## 4. Shutdown sequence

```
shutdown requested (end of day, max loss, operator, or fatal supervision event)
  │
  1. TradingLifecycle → SHUTTING_DOWN
  │     └─ new entries blocked process-wide from this instant
  │
  2. each worker closes its own tracked legs
  │     └─ a REJECTED exit does NOT count as closed
  │
  3. _wait_for_shutdown_account_flat(...)
  │     └─ despite the legacy function name, reconcile the RUNNER'S tracked ledger
  │        ├─ flat      ──► continue
  │        └─ not flat  ──► stay alive in RECONCILIATION; do NOT report clean
  │
  4. _warn_if_account_not_flat / _advisory_account_audit
  │     └─ advisory only: the operator may hold manual positions in the same
  │        account, so a non-flat ACCOUNT is a warning, not an error, as long as
  │        the runner's own ledger is flat
  │
  5. _finalize_flat_session(...)
        ├─ _publish_eod_summary(...)          → Telegram
        ├─ _update_pnl_google_sheet()         → per-strategy P&L row/column
        └─ _refresh_instrument_master_for_next_day()
```

Step 3 is the one that matters. "Stopping the threads" and "safely stopping a
live trading process" are different things, and `Dependencies/trading_lifecycle.py`
exists to keep them separate: the process refuses to claim a clean exit while a
leg it opened is still open. Step 4 is deliberately advisory because an
account-wide position may belong to the operator rather than this runner.

---

## 5. Module loading

Most strategy files have spaces in their names and cannot be imported normally.
`load_module(module_name, file_path)` wraps `importlib.util.spec_from_file_location`
so the runner can load them by path. Consequences worth knowing:

- The master itself is loaded the same way by its test suite.
- mypy cannot type-check spaced-name files; they are covered by `compileall`
  plus the unittest suite instead (see [`testing-and-ci.md`](testing-and-ci.md)).
- Anything imported this way is **not** on `sys.path` for its siblings, which is
  why the agent folders carry their own path bootstraps.

See [ADR-0009](../adr/0009-importlib-loading-for-spaced-filenames.md).

---

## 6. Failure handling in this component

| Condition | Behaviour |
|---|---|
| Unknown `LIVE_BROKER` | Live disabled entirely; the session runs paper. Never guesses a broker. |
| Malformed `<PREFIX>_SIZE_MULTIPLIER` | Paper uses 1; that strategy is blocked from live by `_live_config_errors`. |
| Exposure found at startup | Live workers do not start. Operator is alerted. |
| A worker thread dies | Supervision notices; the session does not silently continue believing that strategy is running. |
| Broker not flat at shutdown | Process stays in reconciliation rather than exiting clean. |
| Google Sheet / Telegram unconfigured | Safe no-op. Reporting never blocks or fails shutdown. |

---

## 7. Testing

`Tests/test_nifty_multi_strategy_master.py` loads this file via `importlib` with
`dhanhq` mocked, then drives the startup and shutdown paths directly. The suite
is the primary gate for this component because mypy cannot see the file.

Areas it specifically covers: env toggles, the fail-closed `LIVE_BROKER` switch,
paper/live routing, order fill-confirmation, symbol resolution, and the
shutdown-flatten sequence.

---

## 8. Known debt

- **One 17k-line file.** Cohesive but slow to change. A split is viable but must
  be behaviour-preserving and would need the test suite's `importlib` loading
  reworked in the same change.
- **In-process ledger.** A mid-session crash leaves recovery to operator-driven
  reconciliation. Persisting the ledger is the natural next step if unattended
  running is ever wanted.
