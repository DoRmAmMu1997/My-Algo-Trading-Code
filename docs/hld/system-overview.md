# HLD — NIFTY multi-strategy trading system

**Status:** Current
**Scope:** the whole repository
**Audience:** anyone about to change the runtime, add a strategy, or add a broker

---

## 1. What the system is

A single-machine, single-process trading system for NIFTY index options. It runs
an approximately 27-strategy core roster plus two independently opt-in AI agents
concurrently against one shared market-data feed, decides entries and exits per
strategy, and executes those decisions either on paper (default) or through a
real broker (explicitly enabled, per strategy).

It has been running live since May 2026. Every design decision in this document
is weighted by that: **the system is allowed to miss a trade; it is not allowed
to lose track of a position.**

### The three phases

```
   fetch                        backtest                      run
   -----                        --------                      ---
 DhanHQ REST      ──►     backtesting.py over          multithreaded front test
 1-min OHLC CSV           the same 1-min CSV           (paper by default, live
 (Data Extractors/)       (My Backtest Files/)          when explicitly enabled)
```

Only the third phase touches money. The first two exist so that a strategy's
behaviour is known before it is given capital, and both consume the *same*
1-minute candles the live runner consumes — which is why the websocket producer
trues its bars up against official REST candles (see §5.2).

---

## 2. Requirements

### 2.1 Functional

| # | Requirement |
|---|---|
| F1 | Run many independent strategies concurrently against one market-data feed. |
| F2 | Each strategy independently decides entry/exit on NIFTY ATM options (some on multi-leg baskets). |
| F3 | Execute on paper by default; execute live only when explicitly enabled for both the process and the strategy. |
| F4 | Support more than one broker behind one interface, switchable by configuration. |
| F5 | Never lose track of live exposure — including across partial fills, lost responses, and restarts. |
| F6 | Report every entry/exit to Telegram, and per-strategy end-of-day P&L to a Google Sheet. |
| F7 | Be entirely configuration-driven from one file; nothing hard-coded per run. |

### 2.2 Non-functional

| Dimension | Target and why |
|---|---|
| **Latency** | Seconds, not milliseconds. Strategies act on *completed* 1-minute (or resampled 5-minute) candles, so a 2–5 second data lag is inside the decision granularity. This is the single most important non-functional fact about the system: **it is not a low-latency system**, and no design should be justified by latency. |
| **Throughput** | One index, ~30 workers, a handful of legs each. Trivial. Never the constraint. |
| **Availability** | One trading session per day, ~09:15–15:30 IST. A crash mid-session is a *safety* event (exposure may exist), not an availability event. |
| **Correctness** | Fail-closed everywhere. An unknown broker name disables live trading. An unreadable quote refuses a live entry. Ambiguous fill state freezes new live entries rather than guessing. |
| **Cost** | One DhanHQ subscription, one broker account, optionally one Claude and one Codex subscription. No cloud infrastructure. |
| **Operability** | One operator (the author). Everything must be diagnosable from one log file and one `.env` audit command. |

### 2.3 Constraints that shaped the design

- **One person maintains this.** Complexity that needs a team to keep alive is a
  liability, not a feature. This rules out an event-bus, a database, or a
  service mesh for a workload that fits in one process (see [ADR-0001](../adr/0001-single-process-thread-per-strategy.md)).
- **Broker APIs are the unreliable part.** Not the strategies, not the data
  volume. The safety machinery is concentrated at that boundary.
- **Many source files have spaces in their names** (`Nifty Multi Strategy Front
  Test - Master File.py`). They cannot be imported normally, which shapes module
  loading, mypy scope and test layout (see [ADR-0009](../adr/0009-importlib-loading-for-spaced-filenames.md)).
- **The feed carries no volume.** Anything volume-derived (true VWAP, breadth)
  is either a documented proxy or deliberately unimplemented.

---

## 3. Context

```
                       ┌───────────────────────────────┐
   DhanHQ Data API ───►│                               │
   (REST + optional    │   Front-test master process   │──► Telegram (alerts)
    marketfeed WS)     │   (one Python process)        │
                       │                               │──► Google Sheets (EOD P&L)
   Broker API      ◄──►│                               │
   (Kotak | Shoonya |  │                               │──► rotating log file
    Flattrade | Dhan)  └───────────────────────────────┘
                                     ▲
   Claude / Codex  ◄─────────────────┘  (optional, only if the AI agents are enabled)
   subscriptions
```

Note that DhanHQ appears on both sides of the boundary: it is the market-data
provider for every run, and it is *also* one of the four selectable execution
brokers. The two sessions are deliberately kept separate (`DhanBrokerClient` for
data, `dhan_execution_client` for orders) so a data-side token problem cannot
silently affect order placement, or vice versa.

---

## 4. Component view

```
                       Nifty Multi Strategy Front Test - Master File.py
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                                                                              │
 │  main()                                                                      │
 │   ├─ config load + validation      (_env_* / _scaled_* / _live_config_errors)│
 │   ├─ startup exposure audit        (Dependencies/startup_exposure.py)        │
 │   ├─ thread start + supervision    (_start_and_supervise_runtime_threads)    │
 │   └─ shutdown: flatten → confirm   (Dependencies/trading_lifecycle.py)       │
 │                                                                              │
 │  ┌─────────────────────┐      writes      ┌──────────────────────────────┐   │
 │  │ Market data producer│ ───────────────► │   SharedMarketDataStore      │   │
 │  │ (exactly one of):   │                  │   ─ lock-guarded             │   │
 │  │  CentralMarketData  │                  │   ─ 1-min OHLC frames        │   │
 │  │  Fetcher   (REST)   │                  │   ─ LTP cache per leg        │   │
 │  │  WebSocketMarketData│                  │   ─ MarketDataHealth state   │   │
 │  │  Fetcher   (ticks)  │                  └──────────────┬───────────────┘   │
 │  └─────────────────────┘                        reads    │                   │
 │                                                          ▼                   │
 │  ┌────────────────────────────────────────────────────────────────────────┐  │
 │  │  Strategy worker threads (one per enabled strategy)                     │  │
 │  │                                                                        │  │
 │  │   BasePaperStrategyWorker                                              │  │
 │  │    └─ AtmSingleLegStrategyWorker ── Renko, EMA, HeikinAshi,            │  │
 │  │        │                            ProfitShooter, OpeningStrike,      │  │
 │  │        │                            CPR, CPRAlgo3, CPR AI,             │  │
 │  │        │                            + 14 factory-built ports           │  │
 │  │        └─ NextOpenAtmStrategyWorker ── Goldmine, MoneyMachine          │  │
 │  │    ├─ SupertrendBullishWorker / DonchianBearishWorker (hedged puts)    │  │
 │  │    ├─ Delta20HedgedSpreadWorker  (4-leg spread)                        │  │
 │  │    ├─ LongStrangleWorker         (dual-leg OTM basket)                 │  │
 │  │    └─ SLHuntingAIWorker          (optional, Claude)                    │  │
 │  └───────────────┬─────────────────────────────────┬──────────────────────┘  │
 │                  │ enter_position/exit_position    │ event dicts             │
 │                  ▼                                 ▼                         │
 │  ┌────────────────────────────────┐   ┌────────────────────────────────┐     │
 │  │ ExecutionSafetyCoordinator     │   │ queue.Queue                    │     │
 │  │  ─ one shared broker lock      │   │   └─► TelegramMessageWorker    │     │
 │  │  ─ rate limiting               │   │        (best-effort, one thread)│     │
 │  │  ─ 10s deadline per call       │   └────────────────────────────────┘     │
 │  │  ─ execution_ledger (qty)      │                                          │
 │  └───────────────┬────────────────┘                                          │
 └──────────────────┼───────────────────────────────────────────────────────────┘
                    ▼
        execution_client  (broker-agnostic surface, ADR-0002)
        ├─ Kotak API/kotak_execution.py
        ├─ Shoonya API/shoonya_execution.py   (+ vendored NorenApi.py)
        ├─ Flattrade API/flattrade_execution.py
        └─ Dhan API/dhan_execution.py
```

Shared, broker-neutral primitives live in `Dependencies/` and are deliberately
*small, pure and heavily tested* — they carry the highest coverage budgets in the
repository (90%; see [`../lld/testing-and-ci.md`](../lld/testing-and-ci.md)):

| Module | Responsibility |
|---|---|
| `broker_contract.py` | The four normalized order outcomes and the adapter Protocol |
| `execution_ledger.py` | Quantity-bearing state per live leg; cumulative fills as deltas |
| `startup_exposure.py` | Read-only pre-flight audit of broker orders and positions |
| `trading_lifecycle.py` | Flatten-then-stop shutdown state machine |
| `market_data_health.py` | Candle validation and feed-freshness state |
| `tick_bar_builder.py` | Pure tick→bar helpers for the websocket producer |
| `risk_sizing.py` | Fail-closed lot sizing against a rupee budget |
| `next_open_entry.py` | One-bar lifetime and price rebasing for `NEXT_OPEN` signals |
| `order_splitting.py` | Split oversized orders into exchange-legal chunks |
| `secret_redaction.py` | Scrub credentials from every log record |
| `diagnostic_preflight.py` | Local checks shared by the four broker diagnostics |

---

## 5. Data flow

### 5.1 The decision loop (one strategy, one bar)

```
 producer thread                worker thread
 ───────────────                ─────────────
 poll/tick ─► validate ─► store
                            │
                            ├─► worker wakes on its poll interval
                            │     read 1-min frame + LTP snapshot
                            │     health gate: is the feed fresh?      ── stale ──► hold / liquidate
                            │     resample to the strategy's timeframe
                            │     signal logic (Signal Generators/…)   ── none ───► sleep
                            │     resolve ATM strike + expiry
                            │     spread gate (<PREFIX>_MAX_SPREAD_PCT) ─ too wide ► skip
                            │     size: risk_sizing.SizingDecision      ─ over budget ► skip
                            │     double gate: LIVE_TRADING_ENABLED
                            │                  && <PREFIX>_LIVE_TRADING
                            ├─────► paper path: record fill at LTP
                            └─────► live path: ExecutionSafetyCoordinator
                                                 └─► execution_client.place_market_order
                                                       └─► poll status → typed OrderResult
                                                             └─► execution_ledger applies filled qty
```

The critical property of this loop is that **every gate fails closed**. A
missing quote, a malformed candle, a malformed size multiplier, an unknown
broker name — each one refuses the live action rather than proceeding on a
guess. Several of them (the spread gate, the size multiplier) deliberately
behave *differently* in paper and live: paper is allowed to proceed when the
failure is an API problem rather than a market fact, so an infrastructure blip
does not cost a paper data point.

### 5.2 Market data, two producers, one contract

Exactly one producer thread runs, selected by `MARKET_DATA_SOURCE` and failing
closed to REST on any unrecognised value.

```
 REST (default)                        WEBSOCKET (opt-in, paid Data API)
 ──────────────                        ─────────────────────────────────
 poll full window every 2-5s           ticks build the forming minute live
        │                              REST warmup at start
        │                              REST gap-backfill on reconnect
        │                              once per minute: true-up completed
        │                                candles against official REST candles
        ▼                                (official wins; divergence logged)
        └──────────────► SharedMarketDataStore ◄──────────────┘
```

Both write the *same* shape into the same store, so no strategy knows or cares
which producer is running. The true-up is what makes that claim safe: without
it, tick-built candles could drift from the candles the backtests used. See
[ADR-0005](../adr/0005-rest-vs-websocket-market-data.md).

### 5.3 Reporting

Trade events are published to a `queue.Queue` and drained by a single
`TelegramMessageWorker`. The queue exists so Telegram latency or downtime can
never block a trading decision — alerting is best-effort by design.

End-of-day P&L takes a deliberately indirect route: on a clean shutdown, the
runner **parses its own append-mode log** for each strategy's realised P&L and
writes it to the Google Sheet. Parsing a log rather than reading in-memory
totals means a partially crashed session still reports what it actually did, and
the month can be backfilled from the same log.

---

## 6. Concurrency model

| Thread | Count | Role |
|---|---|---|
| Market data producer | 1 | Poll or stream; write the shared store |
| Strategy workers | ~27 core + up to 2 optional agents, minus disabled ones | Read the store, decide, execute |
| Telegram worker | 1 (if enabled) | Drain the event queue |
| Main | 1 | Start, supervise, shut down |

**Shared mutable state is exactly two objects**, and both are lock-guarded:

1. `SharedMarketDataStore` — many readers, one writer.
2. The broker session behind `ExecutionSafetyCoordinator` — many writers, serialized.

Everything else a worker touches is its own. This is what makes a
thread-per-strategy model tractable for one maintainer: the concurrency review
surface is two objects, not thirty (see [ADR-0001](../adr/0001-single-process-thread-per-strategy.md)).

The GIL is not a problem here because the workload is overwhelmingly I/O-bound
(HTTP calls and sleeps) and the CPU work per bar is small pandas operations on
frames of a few hundred rows.

---

## 7. Failure model

The system is designed around the failure modes that actually happen, in
descending order of how much they can cost:

| Failure | Detection | Response |
|---|---|---|
| **Order acknowledged but fill unknown** | `OrderStatus.UNKNOWN` / `PARTIAL` from the adapter | Exposure may exist. Freeze new live entries, keep exits available, reconcile. **Never** treat an ack or an order ID as proof of fill. ([ADR-0003](../adr/0003-acknowledgement-is-not-a-fill.md)) |
| **Live exit rejected** | typed `REJECTED` on an exit | The position stays open. The runner does not pretend it closed. |
| **Live entry rejected, zero fill** | typed `REJECTED` **and** zero filled quantity | Only then does the entry fall back to paper. |
| **Broker call hangs** | 10-second deadline that *includes* the shared-lock and rate-limit wait | Abort the call. One hung HTTP request must not stall the shared lock and with it every other worker. |
| **Feed goes stale** | `MarketDataHealth` (10s LTP / 150s bar / 30s liquidation thresholds) | Refuse new entries; liquidate open positions past the liquidation threshold. |
| **Malformed candle** | `validate_ohlc_frame` | Reject the snapshot; do not publish it. Strategies see the last good data, never a bad one. |
| **Exposure present at startup** | `startup_exposure` read-only audit of orders *and* positions | Do not start live workers into a book this process did not create. |
| **Credentials in a log** | `install_redaction_filter` on the root logger | Scrubbed before the record reaches the console or file, including exception tracebacks and lazy `%s` args. |
| **AI agent errors** | any SDK/agent exception | Becomes a safe HOLD. The separate mechanical risk loop keeps checking stop, target, max-loss, stale data and square-off regardless. |
| **Config typo** | `_live_config_errors` at startup, `algo.py check-env` on demand | Malformed size knobs block that strategy from live (paper falls back to 1). |

The recurring shape: **an ambiguous state is treated as the dangerous state.**

---

## 8. Key trade-offs

| Decision | What we gain | What we pay | ADR |
|---|---|---|---|
| One process, thread per strategy | Two lock-guarded objects to reason about; trivial deployment; one log | No horizontal scale; one crash stops everything; GIL caps CPU-bound work | [0001](../adr/0001-single-process-thread-per-strategy.md) |
| Broker-agnostic contract | Swap brokers by config; one execution path to test | Every adapter must contain its broker's quirks rather than leak them | [0002](../adr/0002-broker-agnostic-execution-contract.md) |
| Typed outcomes + quantity ledger | Cannot mistake an ack for a fill | More states for callers to handle; `UNKNOWN` needs an operator | [0003](../adr/0003-acknowledgement-is-not-a-fill.md) |
| Paper by default, two flags for live | A single flag can never be enough to risk money | Two places to change; easy to think you are live when you are not | [0004](../adr/0004-paper-by-default-double-gate.md) |
| REST default, websocket opt-in | Lower API load without betting the session on a socket | Two producers to keep behaviourally identical | [0005](../adr/0005-rest-vs-websocket-market-data.md) |
| Per-strategy size multiplier, no global | One typo cannot enlarge every strategy | Scaling the whole book means editing every strategy | [0006](../adr/0006-per-strategy-size-multiplier.md) |
| LLM agents opt-in, host owns gates | A model can be wrong without being dangerous | Two runtimes and two subscriptions to maintain | [0007](../adr/0007-llm-agents-as-opt-in-workers.md) |
| One `.env` | One place to look; one place to audit | A large flat file; discovery depends on `env.example` staying current | [0008](../adr/0008-single-env-as-config-source.md) |
| Spaced filenames + `importlib` | No churn renaming files the author navigates by name | mypy cannot see the master; tests need explicit path bootstraps | [0009](../adr/0009-importlib-loading-for-spaced-filenames.md) |

---

## 9. Scale and what to revisit

The system is nowhere near a resource limit, so "scale" here means *scope*, not
load. The things that would actually force a redesign:

| If this changes | What breaks first | Likely response |
|---|---|---|
| **A second index traded as a first-class product** (not just BankNIFTY confirmation/mirror) | `SharedMarketDataStore` and the ATM resolution path both assume NIFTY is the primary underlying | Parameterize the underlying; one producer per index |
| **A strategy needs sub-second reaction** | The whole completed-candle model, and the 2–5s REST cadence | Not an increment — a different system. Do not stretch this one into it. |
| **More than ~50 workers** | Thread count is still fine; the shared broker lock becomes the queue | Per-broker connection pool, or batch order submission |
| **Multi-account or multi-operator** | `execution_ledger` and the startup audit assume one account, one process | Account-scoped ledger keys; a real store instead of in-process state |
| **The runner must survive a mid-session restart** | Ledger state is in-process; recovery today is operator-driven reconciliation | Persist the ledger (SQLite) and rebuild from the broker book on start |
| **Someone else joins the project** | Nothing technical — but the "small enough to hold in one head" premise behind ADR-0001 | Revisit ADR-0001 honestly rather than growing the master file |

The one piece of debt worth naming explicitly: the master runner is a single
17k-line file. It is coherent and heavily tested, but it is the main reason a
change here is slower than it should be. Splitting it is a real option; it has
not been done because the test suite loads it as one module via `importlib` and
the split would have to be done without a single behavioural change to
live-money code.

---

## 10. Where to go next

- Component internals: [`../lld/`](../lld/)
- Why a decision was made: [`../adr/`](../adr/)
- How to run and configure it: root [`README.md`](../../README.md)
- Live-safety rules in condensed form: [`CLAUDE.md`](../../CLAUDE.md) / [`AGENTS.md`](../../AGENTS.md)
