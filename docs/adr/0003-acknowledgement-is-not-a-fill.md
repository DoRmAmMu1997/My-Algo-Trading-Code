# ADR-0003: An acknowledgement is not a fill — typed outcomes and a quantity ledger

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

Broker APIs answer an order with an acknowledgement, an order ID, a truthy dict,
or an error. None of those is a fill. The states that actually occur:

- The order fills completely.
- It fills partially, and the rest is pending or cancelled.
- It is rejected outright.
- **The response is lost** — the order may or may not exist at the broker.
- The status is reported cumulatively, late, out of order, or repeated.
- **Dhan's SDK returns `{'status': 'failure', 'remarks': str(exc)}` for
  transport errors** — byte-identical in shape to a genuine rejection.

The consequence of getting this wrong is not a bad trade. It is an untracked
live position.

There is a second, subtler trap specific to this system: a rejected live **entry
with zero fill** is deliberately treated as a paper fallback. So anything that
wrongly looks like a zero-fill rejection causes the runner to trade on paper
while the operator believes it is live.

## Decision

1. Normalize every broker reply into exactly **four** outcomes —
   `FILLED`, `PARTIAL`, `REJECTED`, `UNKNOWN` — carried by an `OrderResult`
   that also holds the **filled quantity**.
2. Record every leg in `Dependencies/execution_ledger.py` **before** submission,
   and apply broker fill reports as **deltas against cumulative totals**.
3. Never derive `REJECTED` from anything that could also be a transport error.
   In the Dhan adapter: a `dict` `remarks` means the server refused; a `str`
   means the outcome is indeterminate → `UNKNOWN`.
4. Treat `PARTIAL` and `UNKNOWN` identically at the call site: exposure may
   exist — freeze new live entries, keep exits available, reconcile.

## Options considered

### Option A: Typed outcomes + quantity-bearing ledger (chosen)

**Pros:** callers cannot mistake an ack for a fill — there is no boolean to
misread; `UNKNOWN` is representable, so ambiguity is handled instead of
collapsed; cumulative-delta application survives late, repeated and out-of-order
reports; the ledger's pre-submission record survives a lost response.
**Cons:** four states for every caller to handle; `UNKNOWN` ultimately needs an
operator; more code than `if result: ...`.

### Option B: Boolean success + separate quantity lookup

**Pros:** simplest call sites.
**Cons:** `False` conflates "refused" with "unknown", which are opposite
instructions — one means fall back to paper, the other means assume exposure.
Collapsing them is precisely the bug this ADR exists to prevent.

### Option C: Trust order status polling alone, no local ledger

**Pros:** the broker is the source of truth, so no duplicate state.
**Cons:** a lost placement response leaves nothing to poll *for*. Without a
pre-submission record there is no order ID and no evidence the attempt happened.
Dhan's `correlationId` recovery exists exactly because this case is real.

## Trade-off analysis

Option B is materially simpler and is wrong in the one scenario that costs the
most. The extra states in Option A are not incidental complexity — they are the
problem's actual shape. Every simplification here works by discarding the
distinction between "refused" and "unknown", and those demand opposite actions.

The ledger (Option A over C) costs duplicated state, and duplicated state can
drift. That is accepted because the alternative — having no record of an
attempt whose response was lost — has no recovery path at all.

## Consequences

**Easier:** answering "how much am I holding?" at any moment; recovering an
order whose response was lost (Dhan `correlationId`); reasoning about partial
fills; auditing the live path.

**Harder:** every call site must handle four states; `UNKNOWN` blocks new live
entries until reconciled, which will occasionally stop a session that was
actually fine; local state must be kept consistent with the broker.

**To revisit when:** a broker offers a reliable streaming order-update channel
that removes the lost-response case — the ledger would still be wanted, but
`UNKNOWN` could become rarer.

## Action items

- [x] `Dependencies/broker_contract.py` — four outcomes, quantity carried.
- [x] `Dependencies/execution_ledger.py` — pre-submission record, cumulative deltas.
- [x] Dhan adapter never derives `REJECTED` from the placement envelope.
- [x] Dhan `order_tag` sent as `correlationId` for lost-response recovery.
- [x] 90% branch-coverage budget on both modules.
- [x] Paper fallback only on `REJECTED` **with zero filled quantity**.
