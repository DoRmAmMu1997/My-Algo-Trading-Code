# ADR-0001: One process, one thread per strategy

**Status:** Accepted
**Date:** 2026-08-10 (retrospective — records a decision already in force)
**Deciders:** repository owner

## Context

The system runs roughly 27 core strategies plus up to two optional AI agents
concurrently against one market-data feed, executing through one broker account.

The workload's actual shape:

- Strategies act on **completed 1-minute (or resampled 5-minute) candles**. A
  2–5 second data lag sits inside the decision granularity.
- Per-bar CPU work is small pandas operations on frames of a few hundred rows.
- Almost all wall-clock time is spent in HTTP calls and sleeps.
- One person maintains the whole thing.
- One trading session per day, on one machine.

## Decision

Run everything in **one Python process** with **one `threading.Thread` per
strategy**, one producer thread for market data, and one worker thread for
Telegram. Share exactly two mutable objects, both lock-guarded:
`SharedMarketDataStore` and the broker session behind `ExecutionSafetyCoordinator`.

## Options considered

### Option A: One process, thread per strategy (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | Nil |
| Scalability | Bounded by one machine — irrelevant at this size |
| Team familiarity | Total |

**Pros:** the concurrency review surface is two objects; one broker session and
one lock make order serialization trivial; one log file; deployment is
`python algo.py run`; a shared in-memory feed needs no serialization.
**Cons:** no horizontal scale; one crash stops every strategy; the GIL caps
CPU-bound work; a leaked exception in one thread can affect the process.

### Option B: Process per strategy

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | Nil in money, high in code |
| Scalability | Real, and unneeded |
| Team familiarity | Moderate |

**Pros:** true isolation and parallelism; one strategy cannot take down another.
**Cons:** the broker session must be shared across processes, which turns the
simplest safety property in the system (one lock, one session, serialized
orders) into a distributed-locking problem. The market-data store would need IPC
or a broker. **The exposure ledger would have to become cross-process state** —
and that ledger is the thing that must never be wrong.

### Option C: Single-threaded asyncio

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Cost | Nil |
| Scalability | Fine |
| Team familiarity | Lower |

**Pros:** no locks; natural fit for an I/O-bound workload; cheap concurrency.
**Cons:** every broker SDK in use is synchronous and would need thread-pool
wrapping anyway, reintroducing the thread boundary at the least convenient
place. One blocking call in one strategy stalls **all** of them — the exact
failure the 10-second deadline exists to contain. The strategy code would have
to be rewritten in colour.

## Trade-off analysis

The deciding factor is not performance — no option is anywhere near a resource
limit. It is **how many places can be wrong about exposure.**

Option A has one: an in-process ledger guarded by one lock. Option B has as many
as there are processes, plus the coordination between them. Option C has one,
but reintroduces threads at the broker boundary while making a single blocking
SDK call a process-wide stall.

For live-money code maintained by one person, minimising the number of places
that can disagree about "how much am I holding?" beats every other consideration.

## Consequences

**Easier:** reasoning about concurrency (two objects); serializing orders;
deploying; debugging from one log; adding a strategy (add a thread).

**Harder:** surviving a crash (everything stops together); CPU-bound work
(GIL); running strategies on separate machines (not possible without redesign).

**To revisit when:** the runner must survive a mid-session restart unattended
(persist the ledger first — see the HLD §9), a second operator joins, or a
strategy genuinely needs sub-second reaction (that is a different system, not an
increment of this one).

## Action items

- [x] One lock-guarded `SharedMarketDataStore`.
- [x] One lock-guarded broker session behind `ExecutionSafetyCoordinator`.
- [x] Ten-second deadline **including** lock wait, so one hung call cannot stall
      the other workers ([ADR-0002](0002-broker-agnostic-execution-contract.md)).
- [ ] Persist the execution ledger if unattended running is ever wanted.
