# LLD — Execution and the broker layer

**Owns:** `ExecutionSafetyCoordinator`, `OptionsContractResolver`,
`_select_execution_client` (master file) · `Dependencies/broker_contract.py` ·
`Dependencies/execution_ledger.py` · `Dependencies/startup_exposure.py` ·
`Dependencies/order_splitting.py` · the four adapters under `Dependencies/*/`
**Related ADRs:** [0002](../adr/0002-broker-agnostic-execution-contract.md), [0003](../adr/0003-acknowledgement-is-not-a-fill.md)

> This is the most safety-critical component in the repository. Read
> [`risk-and-safety.md`](risk-and-safety.md) alongside it.

---

## 1. Responsibility

Turn "buy 3 lots of the NIFTY 24150 CE" into a real order at whichever broker is
configured, and return an answer the caller **cannot misread**.

The second half is the hard part. Broker APIs answer with acknowledgements, order
IDs, truthy dicts, and — in at least one case — a transport error that is
byte-identical to a rejection. None of those are fills.

---

## 2. Layering

```
 worker.enter_position(...)
   │
   ▼
 ExecutionSafetyCoordinator          ── ONE shared, lock-guarded broker session
   ├─ acquire the broker lock
   ├─ rate limit
   ├─ 10-second deadline  ◄── INCLUDES the lock + rate-limit wait, not just the HTTP call
   ├─ execution_ledger.record_attempt(...)   ── BEFORE submission
   ▼
 execution_client                    ── broker-agnostic surface (ADR-0002)
   ├─ ensure_logged_in / is_logged_in / logout
   ├─ preload_scrip_master / resolve_option_symbol
   ├─ place_market_order / get_order_status / cancel_order
   ├─ list_open_orders / list_open_positions
   ├─ recover_after_reconciliation
   └─ extract_order_id
   │
   ├─ Dependencies/Kotak API/kotak_execution.py
   ├─ Dependencies/Shoonya API/shoonya_execution.py   (+ vendored NorenApi.py)
   ├─ Dependencies/Flattrade API/flattrade_execution.py
   └─ Dependencies/Dhan API/dhan_execution.py
   │
   ▼
 typed OrderResult  ──► execution_ledger applies cumulative filled qty as a DELTA
```

The runner only ever touches the generic `execution_client`. Adding a broker
means implementing that surface and adding a coverage row in
`scripts/check_coverage_thresholds.py` — the latter is part of adding a broker,
not an afterthought, because a broker without a row silently escapes the 80%
adapter budget.

---

## 3. The contract — `Dependencies/broker_contract.py`

Four normalized outcomes, and only four:

| `OrderStatus` | Meaning | What the caller must do |
|---|---|---|
| `FILLED` | The requested quantity traded | Proceed |
| `PARTIAL` | Some traded, some did not | **Exposure exists.** Freeze new live entries; reconcile |
| `REJECTED` | The broker refused | Entry: fall back to paper **only if filled qty is zero**. Exit: the position stays open |
| `UNKNOWN` | The outcome could not be determined | Treat exactly like `PARTIAL` — exposure may exist |

`OrderResult` carries the filled quantity, not just a status, because a status
without a quantity cannot answer "how much am I holding?".

The module also pins itself into `sys.modules` under **both**
`broker_contract` and `Dependencies.broker_contract`. Adapters run both as repo
modules and as standalone diagnostic scripts that add `Dependencies/` to
`sys.path`; without this, the same class would be imported twice under two names
and `isinstance(result, OrderResult)` would be false at the live-order boundary.

---

## 4. The ledger — `Dependencies/execution_ledger.py`

State per live leg, thread-safe, quantity-bearing:

- A leg is recorded **before** submission, so a lost response still leaves a
  trace of what was attempted.
- Broker fill reports are **cumulative totals**, applied as deltas. Reports can
  arrive late, out of order, or repeat; applying them as deltas means known
  quantity can never silently disappear or double-count.
- The runner never infers exposure from a boolean.

---

## 5. Startup and shutdown boundaries

**Startup** — `Dependencies/startup_exposure.py` performs exactly two reads:
open orders and open index-option positions. It never adopts, cancels, recovers
or flattens. If either read shows exposure, live workers do not start.

**Shutdown** — `Dependencies/trading_lifecycle.py` blocks new entries the moment
shutdown begins, then requires every tracked leg to be closed and the broker to
reconcile the runner's ledger flat before the session may be called clean. A
failed close keeps the process alive in reconciliation.

The two boundaries intentionally use different scopes:

- Startup is **account-wide** because a fresh process has no trustworthy local
  ledger yet. Any open order or relevant index-option position blocks live
  startup and requires an operator decision.
- Shutdown is **runner-owned** because the in-process ledger identifies the legs
  this run opened. Those legs must be broker-reconciled flat. Once they are, a
  separate account-wide audit is advisory because remaining exposure may belong
  to the operator's manual trading.

---

## 6. Contract resolution — `OptionsContractResolver`

Maps (underlying, spot, offset, expiry rule) → a tradable option symbol.

- Strike selection: ATM by default; some strategies take an ITM/OTM offset
  (`CPR_ALGO3_ITM_OFFSET`, the SL Hunting mirror's near-expiry ITM steps).
- Expiry: the ATM family buys the **next-next** expiry.
- Dhan resolves contracts from the local `Dependencies/all_instrument <date>.csv`
  (refreshed by `_refresh_instrument_master_for_next_day()` at shutdown), not a
  live download.
- Every adapter also exposes `resolve_option_symbol` for its own wire format.

### 6.1 The BankNIFTY monthly exception

BankNIFTY lists only monthly series, so the ATM family's "next-next" rule would
put an intraday mirror leg two months out. The SL Hunting mirror therefore
always trades the **nearest** monthly expiry and never rolls forward — Kotak
rejects MIS (intraday) orders on next-month contracts, which repeatedly killed
the live mirror leg (operator finding, 2026-07-23). Expiry week is handled on
the *strike* axis instead: inside the final week the mirror buys a deep-ITM
strike, which is mostly intrinsic value and tracks the index rather than
bleeding near-expiry time premium.

---

## 7. Order splitting — `Dependencies/order_splitting.py`

NSE caps how much of a contract may be sent in one order (the freeze quantity).
An order at or above it is rejected by the exchange.

That is dangerous here specifically because a rejected live **entry** with zero
fill is deliberately treated as a paper fallback — so without splitting, an
oversized order would quietly trade on paper while the operator believed it was
live. `max_legal_chunk_units` / `split_order_quantity` keep every submission
legal. 90% branch-coverage budget.

---

## 8. Broker-specific quirks the adapters exist to contain

### Dhan

- Its SDK returns `{'status': 'failure', 'remarks': str(exc)}` for **transport**
  errors — shape-identical to a genuine rejection. So `REJECTED` is never
  derived from the placement envelope: a `dict` `remarks` means the server
  refused; a `str` means the outcome is indeterminate (`UNKNOWN`).
- `order_tag` is sent as Dhan's `correlationId`, so `get_order_by_correlationID`
  can recover an order whose response was lost.
- Non-contract states are aliased adapter-locally: `EXPIRED`→`CANCELLED`,
  `PART_TRADED`→`PARTIAL`. `TRANSIT` and `PENDING` stay unmapped so they remain
  transient rather than being forced into a terminal state.
- The SDK ships a 60s default timeout; `_login_locked` overrides it down to 10s.

### Shoonya

- The `NorenApi` client is **vendored** under `Dependencies/Shoonya API/`. It is
  never linted, never rewritten locally, and excluded from mypy and Bandit.
- Every HTTP call has an explicit timeout — a hung call would otherwise stall a
  worker thread *and* the shared broker lock.

### Kotak

- The official v2 client comes from the `v2.0.1` Git tag and pins older
  pandas/requests, so it lives in `requirements-brokers.txt` and is validated in
  its own CI job rather than combined with the core set.

### Flattrade

- Pi v2 browser-token flow, exact NFO index scrip master, documented request
  limits, market-order protection, and `SingleOrdHist` fill confirmation.

### Dhan as both data and execution

`DhanBrokerClient` (market data) and `dhan_execution_client` (orders) are
separate sessions on purpose, so a data-side token problem cannot silently
affect order placement.

---

## 9. Timeouts

Every broker network/SDK call has a **ten-second deadline that includes its
shared lock and rate-limit wait**. Native HTTP timeouts remain enabled for
Shoonya, Flattrade and Dhan on top of that.

The "includes the wait" part is the design: a deadline that starts after the
lock is acquired does not protect the other 29 workers queued behind a hung
call.

---

## 10. Testing

| Suite | Covers |
|---|---|
| `Tests/Dependencies/test_broker_contract.py` | The four outcomes, the Protocol, the dual `sys.modules` registration |
| `Tests/Dependencies/test_execution_ledger.py` | Cumulative-delta application, thread safety |
| `Tests/Dependencies/test_startup_exposure.py` | Read-only audit semantics |
| `Tests/Dependencies/test_order_splitting.py` | Freeze-quantity arithmetic |
| `Tests/Dependencies/Dhan API/test_dhan_execution.py` | The transport-vs-rejection distinction, state aliasing |
| `Tests/Dependencies/Flattrade API/test_flattrade_execution.py` | Wire format, fill confirmation |
| `Tests/test_nifty_multi_strategy_master.py` | Paper/live routing, the fail-closed broker switch, coordinator behaviour |

Coverage budgets: 90% for the safety modules, **80% for every broker adapter**,
enforced by `scripts/check_coverage_thresholds.py`. A module missing from the
coverage report is a **failure**, not a pass — otherwise renaming a file would
silently retire its budget.

CI runs the broker contract and Flattrade adapter suites a second time in the
isolated `requirements-brokers.txt` environment.

---

## 11. Adding a broker — checklist

1. Implement the full `execution_client` surface (§2). No partial adapters.
2. Contain the broker's quirks inside the adapter; never leak a broker-specific
   status into the runner.
3. Never derive `REJECTED` from anything that could also be a transport error.
4. Give every call an explicit HTTP timeout.
5. Add a diagnostic script (`diagnose_<broker>_symbol.py`) using
   `Dependencies/diagnostic_preflight.py`.
6. Add a row to `BROKER_THRESHOLDS` in `scripts/check_coverage_thresholds.py`.
7. Add the adapter to `mypy.files` in `pyproject.toml`.
8. Tests under `Tests/Dependencies/<Broker> API/`.
