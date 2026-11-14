# ADR-0023: The account audit may excuse declared carried legs

**Status:** Accepted
**Date:** 2026-11-14
**Deciders:** repository owner

## Context

ADR-0022 made Zerodha and mStock ordinary broker adapters, and so valid
`LIVE_BROKER` choices. The operator's hand-placed option-selling positions,
carried for a week to three months, sit in that same account.

Live trading arms only after `Dependencies/startup_exposure.py` proves the
account holds no open order and no NIFTY/BANKNIFTY position. The rule is
account-wide on purpose: a fresh process has no ledger, so a position it did
not open may be a leftover from a crashed session, and arming on top of it
could double real exposure. With carried positions in the account that rule
can never pass, so the strategies could not trade live in the very account
ADR-0022 opened to them. ADR-0022 named this and left it as a separate
decision. This is that decision.

Two facts shape it:

* **The operator already declares those legs.** ADR-0022's groups file
  (`Dependencies/position_groups.json`) lists every carried leg with its
  symbol, product and signed quantity, inside a named group.
* **Four decisions rest on the same account check**, through three call
  sites: arming live at startup, a dashboard paper-to-live promotion
  (ADR-0021), resuming entries after an ambiguous order, and the shutdown
  warning. A rule applied to only some of them would leave the others blocked
  -- or, worse, would let them drift apart.

## Decision

### 1. One opt-in switch, off by default

`LIVE_TOLERATE_DECLARED_CARRIED_LEGS` (default `false`). With it off the
runner makes the historical call, `audit_startup_exposure(client)`, argument
for argument, and the audit's evidence is byte-for-byte what it was.

The switch only *reads* the groups file. It does not need
`GROUP_MONITOR_ENABLED`, and it never writes anything.

### 2. A broker position is excused only on an exact match

`audit_startup_exposure` gains one keyword argument,
`tolerated_positions: {(symbol, product): signed quantity}`. A non-flat
NIFTY/BANKNIFTY row is left out of the count only when its symbol, its product
label (compared without regard to case or padding) and its **exact signed
quantity** all match an entry. One entry excuses one row. Everything else
still counts and still blocks live.

The runner builds that mapping from the groups file, and deliberately narrowly:

| A leg is declared only if ... | Why |
|---|---|
| it sits **inside a group** | `algo.py group-legs` appends whatever the broker holds to `unassigned`. If `unassigned` counted, a stray position from a crashed session could excuse itself after one command. Moving a leg into a group is the operator saying "this one is mine". |
| its group's broker is `LIVE_BROKER` | another account's legs say nothing about this one |
| its product is `NRML` | an intraday (`MIS`) position is not a carried leg |

Nothing here weakens the rest of the audit: **any open order still blocks
live**, and an unreadable book is still indeterminate.

### 3. The strategies must trade INTRADAY

A broker nets a position per symbol *and product*. If the strategies also
traded the carry product, a strategy order on a carried strike would merge
into the operator's row. So when `LIVE_PRODUCT_TYPE` is not `INTRADAY`, the
mapping is empty and nothing is tolerated (evidence
`declared_legs=product_not_isolated`).

### 4. Every failure tolerates nothing

A groups file that cannot be read or does not validate yields an empty mapping
(`declared_legs=unreadable`). A missing file yields an empty mapping. An empty
mapping excuses no row, so each of these leaves the account check exactly as
strict as it is with the switch off. The notes are fixed vocabulary and the
tolerance adds only a count (`declared_carried_positions=N`) to the evidence,
so no symbol or quantity reaches a log line or a Telegram alert.

### 5. One helper for all four decisions

`_audit_live_account(client, store)` in the master is now the only caller of
`audit_startup_exposure`. Startup and promotion reach it through
`_preload_and_audit_live_account`; `_try_unfreeze_after_flat_reconciliation`
and `_advisory_account_audit` call it directly. A test parses the master and
fails if any other function calls the audit, or if a call site stops handing
over the store.

### 6. The runner never opens a trade on a declared contract

Each audit records the declared symbols on the store. In
`_place_real_leg_with_entry_lease`, straight after the broker symbol is
resolved, an order that would *open* exposure on one of those symbols is not
sent. It returns the same synthetic `REJECTED` / `LOCAL_NOT_SUBMITTED` result
as an unresolvable symbol, so that one trade runs on paper and nothing is
frozen. Exits are never refused.

The product split of section 3 already keeps the two positions in separate
broker rows. This guard is defence in depth: one contract has one owner, so
the position book stays readable and nothing that later matches by symbol can
confuse the runner's leg with the operator's.

## Consequences and known limits

* **The declared quantity must be kept current.** After adjusting a carried
  position by hand, the groups file must be updated, or the next account check
  blocks live. That is the intended failure direction. The file is re-read on
  every check, so a correction takes effect at the next one without a restart.
* **The check is a point in time, as it always was.** It runs at startup, on a
  promotion while nothing is live, on a resume after a freeze and at shutdown.
  A position opened by hand in between is not seen until the next one, and a
  leg declared mid-session is not known to the order-path guard until then.
* **A resting order still blocks.** A stop or limit order left working on a
  carried leg is an open order, and any open order blocks live. Excusing
  declared orders would be a further decision; this ADR does not make it.
* **A group the monitor squared off** leaves declared entries with no broker
  row. They excuse nothing, which is harmless. A group left half closed
  (`EXIT_FAILED`) no longer matches its declared quantities and blocks a
  resume until the operator corrects the file.
* **mStock's product label is unverified** against a real account. A row
  whose label is blank or unexpected is simply not excused.
* The groups file knows only Zerodha and mStock, so for any other
  `LIVE_BROKER` the switch declares nothing and changes nothing.
