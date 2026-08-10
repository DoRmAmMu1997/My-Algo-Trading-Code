# ADR-0005: REST polling as the default, websocket as an opt-in producer

**Status:** Accepted
**Date:** 2026-08-10 (retrospective — records PR #79)
**Deciders:** repository owner

## Context

The REST producer pulls the full OHLC window every 2–5 seconds, for NIFTY spot
plus every subscribed option leg. With multi-leg baskets (hedged pairs, the
Delta-0.2 four-leg spread, strangle legs, the SL Hunting BankNIFTY mirror) that
is a lot of repeated requests for data that mostly has not changed.

Dhan offers a marketfeed websocket under its **paid Data API** subscription.

Constraints:

- Strategies act on completed candles, so this is **not** a latency problem. The
  motivation is API load and quota, not speed.
- Every backtest was run on official exchange candles. Live bars must remain the
  same bars, or results stop being comparable — and nothing would fail loudly if
  they drifted.
- A trading session cannot be gambled on a socket that may drop.

## Decision

Support both producers behind one contract, selected by `MARKET_DATA_SOURCE`:

- **`REST` (default)** — `CentralMarketDataFetcher`. Any unrecognised value also
  yields REST (fail closed).
- **`WEBSOCKET` (opt-in)** — `WebSocketMarketDataFetcher`. Ticks build the
  forming minute live; REST is retained for warmup history and reconnect
  gap-backfill; and **once per minute the completed candles are trued up against
  Dhan's official REST candles, with official always winning** and any divergence
  logged.

Both write the same shape into `SharedMarketDataStore`, so no strategy knows
which producer is running. Pure tick→bar logic lives in
`Dependencies/tick_bar_builder.py`.

Rollback is `MARKET_DATA_SOURCE=REST` plus a restart. No state migration.

## Options considered

### Option A: REST only

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | No paid subscription |
| Correctness | Perfect — every bar is the official candle |

**Pros:** no reconnect logic, no partial-bar state, nothing to true up.
**Cons:** one full-window pull every few seconds; LTPs only as fresh as the poll.

### Option B: Websocket only

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Cost | Paid Data API |
| Correctness | Depends entirely on tick handling |

**Pros:** lowest API load; real-time LTPs; the forming minute visible live.
**Cons:** a dropped socket has no fallback within the session; tick-built candles
can silently diverge from official ones; requires a paid subscription to run at
all — including in a paper session.

### Option C: Both, REST default, websocket opt-in with a true-up (chosen)

**Pros:** the load win is available without betting a session on the socket;
the true-up bounds divergence to at most one minute and always resolves in
favour of the official candle; rollback is a flag; unrecognised values fail
closed to the safe producer; the pure helpers are unit-testable without a feed.
**Cons:** two producers must be kept behaviourally identical — real, ongoing
maintenance; the websocket path has more moving parts (pump thread, supervisor,
warmup, backfill, true-up).

## Trade-off analysis

The API-load saving (from one full-window pull every 2–5 s to roughly one per
minute) is worth having, but not at the cost of the shared-candle invariant that
links backtests, paper and live.

The true-up is what makes Option C safe rather than merely convenient: without
it, tick-built candles would slowly stop being the candles the strategies were
validated on, and **no individual test would fail**. Making "official always
wins" a blunt, unconditional rule removes any judgement from the reconciliation.

Keeping REST as the default means the paid subscription is an optimisation, not
a dependency — the system runs without it.

## Consequences

**Easier:** staying inside Dhan's rate limits with many subscribed legs;
real-time LTPs for multi-leg baskets; adding a producer later (the contract
exists).

**Harder:** two producers to maintain; the health gates needed a tick-aware
twist — a quiet-but-subscribed leg is fresh **only while the socket is
demonstrably alive**, because otherwise a dead socket looks exactly like a quiet
strike.

**To revisit when:** the true-up starts logging routine divergence (that is a
signal the tick path is wrong, not that the true-up should be relaxed), or if
mid-session failover from websocket to REST becomes worth building.

## Action items

- [x] `_select_market_data_fetcher_class()` fails closed to REST.
- [x] Pure tick logic in `Dependencies/tick_bar_builder.py`, 90% coverage budget.
- [x] Per-minute true-up against official candles; divergence logged.
- [x] Tick-aware freshness for quiet subscribed legs.
- [x] Operator gate: ≥2 clean paper sessions before enabling live on the
      websocket producer — CI never opens a real socket, so a green build proves
      nothing about the transport.
