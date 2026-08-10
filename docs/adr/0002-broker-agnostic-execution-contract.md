# ADR-0002: Broker-agnostic execution contract, fail-closed broker selection

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

Four brokers are usable for live execution: Kotak Neo, Shoonya (Finvasia),
Flattrade Pi v2, and Dhan. They differ in authentication (OAuth, browser token,
TOTP), symbol format, order-status vocabulary, error shapes, and rate limits.

Two forces:

1. Brokers are swapped for real reasons — Shoonya's legacy QuickAuth endpoint is
   being decommissioned; Kotak rejects MIS orders on next-month contracts. The
   system must not require a rewrite when that happens.
2. The runner's order path is the most safety-critical code in the repository.
   It must be tested **once**, not once per broker.

## Decision

Define one broker-agnostic surface that every adapter implements, and route all
real orders through a single generic `execution_client`. The runner never
imports a broker module.

```
ensure_logged_in · is_logged_in · logout
preload_scrip_master · resolve_option_symbol
place_market_order · get_order_status · cancel_order
list_open_orders · list_open_positions
recover_after_reconciliation · extract_order_id
```

`LIVE_BROKER` selects the adapter. **An unrecognised value fails closed:** live
trading is disabled and the session runs on paper. It never falls back to a
default broker.

Shared result types live in `Dependencies/broker_contract.py`
([ADR-0003](0003-acknowledgement-is-not-a-fill.md)).

## Options considered

### Option A: One contract, adapters contain the quirks (chosen)

**Pros:** one tested order path; swapping brokers is a config change; each
broker's weirdness is isolated where it can be unit-tested; a new broker is
additive.
**Cons:** the contract must be the *intersection* of what four brokers can do;
an adapter that cannot honour a method must fail explicitly rather than
half-implement it; quirks that leak are hard to spot in review.

### Option B: Broker-specific call sites in the runner

**Pros:** no abstraction to design; each broker uses its most natural API.
**Cons:** the safety logic (fill confirmation, ledger updates, deadlines) would
be duplicated four times and would drift. This is disqualifying: four copies of
"is this a fill?" is four chances to get it wrong.

### Option C: A third-party unified broker library

**Pros:** no adapter code to maintain.
**Cons:** the failure semantics this system depends on — the difference between
a transport error and a rejection, cumulative-vs-absolute fill quantities — are
exactly what such libraries normalise away. Adding a dependency at the
live-order boundary also adds an unaudited party to the most sensitive path.

## Trade-off analysis

The contract's cost is real: it is the intersection of four brokers, so it
cannot expose anything one of them lacks. The benefit is that the roughly
30 workers share **one** execution path, and every safety rule is written and
tested exactly once.

The fail-closed selection is the deciding detail. A typo in `LIVE_BROKER`
silently routing to a default broker would place real orders through an account
the operator did not intend. Refusing to trade live is always the cheaper error.

## Consequences

**Easier:** swapping brokers; testing execution once; adding a broker; reasoning
about the order path.

**Harder:** using a broker-specific feature (it must be contained in the adapter
or added to the contract for all four); a broker whose model genuinely differs
would strain the intersection.

**To revisit when:** a broker requires a capability the contract cannot express
(streaming order updates, bracket orders, basket submission) — extend the
contract deliberately rather than leaking the capability upward.

## Consequences already realised

- Dhan's transport errors are shape-identical to rejections; the adapter
  contains that (see [ADR-0003](0003-acknowledgement-is-not-a-fill.md)).
- Kotak's pandas/requests pins conflict with the audited core set, so its
  environment is validated in a separate CI job.
- Shoonya's `NorenApi` is vendored, excluded from lint/mypy/Bandit, and every
  call has an explicit timeout.

## Action items

- [x] `Dependencies/broker_contract.py` holds the shared types and Protocol.
- [x] `_select_execution_client` fails closed on an unknown name.
- [x] Ten-second deadline per call, **including** shared lock and rate-limit wait.
- [x] 80% branch-coverage budget per adapter, enforced by
      `scripts/check_coverage_thresholds.py`; a missing module is a failure.
- [x] Adding a broker requires a `BROKER_THRESHOLDS` row — treated as part of
      the work, after the Dhan adapter first landed without one.
