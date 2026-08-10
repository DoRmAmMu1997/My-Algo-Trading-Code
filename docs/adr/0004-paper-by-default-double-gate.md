# ADR-0004: Paper by default, behind a two-flag live gate

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

The same process runs strategies that are being evaluated on paper and
strategies that are trading real money, at the same time, in the same session.
The operator routinely edits `.env` between sessions.

The failure to prevent: **a strategy trading live that the operator did not
intend to be live.** Causes include a leftover flag from yesterday, a typo, a
copy-pasted block, or a new strategy inheriting a global setting.

## Decision

A strategy places a real order only when **both** flags are true:

```
LIVE_TRADING_ENABLED   = true     # process-wide kill-switch, default FALSE
<PREFIX>_LIVE_TRADING  = true     # per strategy,             default FALSE
```

Plus a recognised `LIVE_BROKER`; an unknown value disables live entirely
([ADR-0002](0002-broker-agnostic-execution-contract.md)).

The paper gate is deliberately **asymmetric**:

| | Global switch | Per-strategy switch | Default |
|---|---|---|---|
| **Live** | required | required | off |
| **Virtual (paper)** | *none, by design* | `<PREFIX>_VIRTUAL_TRADING` | on |

`<PREFIX>_VIRTUAL_TRADING=false` stops the worker thread from starting at all,
so the strategy does neither paper nor live.

## Options considered

### Option A: Two flags for live, one for paper, no global paper switch (chosen)

**Pros:** no single edit can put a strategy live; the global switch is a genuine
kill-switch that stops everything in one line; a new strategy is paper by
default because both flags default false; the safe default differs correctly per
mode — "run everything" for paper, "run nothing" for live.
**Cons:** two places to change to go live; it is possible to believe you are
live when only one flag is set.

### Option B: One global `LIVE_TRADING_ENABLED` only

**Pros:** one switch; nothing to forget.
**Cons:** flipping it puts **every** strategy live simultaneously, including
ones added since the last review. Unacceptable — going live must be per
strategy, deliberately.

### Option C: Per-strategy flag only, no global switch

**Pros:** precise; no redundancy.
**Cons:** no single action stops all live trading. During an incident the
operator would edit ~27 lines. A kill-switch that takes 27 edits is not a
kill-switch.

### Option D: A `--live` CLI flag instead of config

**Pros:** explicit at launch; cannot be left over from yesterday.
**Cons:** conflicts with [ADR-0008](0008-single-env-as-config-source.md) (one
config source) and makes live-ness invisible to `check-env`. It also moves the
decision to the moment of *starting*, when the operator is least likely to be
reviewing per-strategy risk.

## Trade-off analysis

Option A's redundancy is the point. Options B and C each optimise away one flag
and lose a distinct property — B loses per-strategy control, C loses the
kill-switch. Both properties are needed, so both flags stay.

The asymmetry with paper is worth stating plainly because it looks inconsistent
until you name the safe default: for paper, running is safe and the risk is
*forgetting to enable* a strategy you meant to evaluate; for live, not running
is safe and the risk is *enabling* one you did not mean to. Defaults follow the
safe direction in each case, so they point in opposite directions.

The residual risk Option A accepts — thinking you are live when only one flag is
set — is a *missed trade*, not a *lost trade*. That is the correct direction for
the error to fall.

## Consequences

**Easier:** going live one strategy at a time; stopping all live trading in one
edit; adding a strategy safely (paper by default); paper-validating a change
before it risks money.

**Harder:** going live (two edits, deliberately); noticing that a strategy is
*not* live when you meant it to be.

**To revisit when:** the operator wants live trading on by default for a proven
core set — that would be a real change of posture and deserves its own ADR, not
a quiet default flip.

## Action items

- [x] Both flags default false; `_configure_startup_live_trading` enforces the
      conjunction.
- [x] `_live_config_errors` blocks a strategy from live on malformed size config
      (paper falls back to 1) — see [ADR-0006](0006-per-strategy-size-multiplier.md).
- [x] Startup exposure audit must be clean before live workers start.
- [x] Telegram and the EOD sheet label rows PAPER / LIVE / MIXED so a
      mislabelled run is visible after the fact.
