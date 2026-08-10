# ADR-0006: A per-strategy size multiplier, deliberately not global

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

Each strategy's size is described by four knobs that must stay consistent:
`<PREFIX>_LOTS`, `<PREFIX>_MAX_LOTS`, `<PREFIX>_RISK_BUDGET`, and the absolute
`<PREFIX>_MAX_LOSS`.

As the account grows, size should grow. Editing four numbers per strategy by
hand is error-prone in a specific, expensive way: **raising the lot cap while
forgetting the budget** produces a position larger than the risk model allows,
and nothing complains.

## Decision

Add `<PREFIX>_SIZE_MULTIPLIER` — one whole number per strategy, default `1`,
range 1–25 (capped by `MAX_SIZE_MULTIPLIER`) — that scales that strategy's whole
size/risk set together.

Applied at **env-read time** via `_scaled_int` / `_scaled_float` /
`_strategy_size_multiplier`, so `Dependencies/risk_sizing.py` is untouched and
scaled values flow through sizing, the kill-switch, Telegram and the Sheet
unchanged.

Three properties are deliberate:

1. **Per strategy only. No global switch.**
2. **Applies to paper and live alike.**
3. **Malformed values (`0`, `2.5`, `30`, `"two"`) fall back to 1 for paper and
   BLOCK that strategy from live** via `_live_config_errors`.

Two knobs are **not** scaled, because their totals already inherit the
multiplier and scaling them would square it:

- `<PREFIX>_MAX_LOSS_PER_LOT` (Delta20) — multiplied by scaled lots downstream.
- `<PREFIX>_STARTING_CAPITAL` / `_DAILY_MAX_LOSS_PCT` — their *product* carries it.

## Options considered

### Option A: Per-strategy multiplier applied at read time (chosen)

**Pros:** one number per strategy instead of four kept consistent by hand; the
sizing module never learns about it; a typo can only affect one strategy;
scaling is visible in every downstream report because the scaled values *are*
the values.
**Cons:** the double-scaling trap is real and needs the two documented
exceptions; a reader of `.env` must remember the multiplier when interpreting
the other four numbers.

### Option B: A global `SIZE_MULTIPLIER`

**Pros:** one number scales the whole book; matches "the account grew".
**Cons:** **one typo enlarges every enabled strategy at once**, including ones
the operator has not looked at in weeks. The blast radius is the entire roster.
Rejected on that alone.

### Option C: Apply the multiplier inside `risk_sizing.py`

**Pros:** one place; sizing owns sizing.
**Cons:** the kill-switch, Telegram messages and the Sheet would keep showing
*unscaled* numbers, so reports would disagree with reality. It would also
require passing the multiplier through every call site.

### Option D: Just edit the four numbers

**Pros:** nothing to build; total transparency.
**Cons:** the exact failure described in Context — an inconsistent set that
allows a position larger than the risk model, with no error.

## Trade-off analysis

Option B is what most systems do and is the one clearly wrong answer here: a
global multiplier turns a single-character mistake into a book-wide risk change.
The cost of Option A over B — editing N numbers to scale the whole book — is
paid rarely and deliberately, which is exactly when the operator should be
thinking about risk.

Applying it at read time (A over C) makes the scaled values the *only* values
anything downstream ever sees, which is what keeps the kill-switch, the alerts
and the P&L sheet honest.

Paper-and-live parity matters because it lets an enlarged size be validated on
paper first — the same reason the whole system is paper-by-default.

## Consequences

**Easier:** growing position size with the account; paper-validating a size
change; keeping the four size knobs consistent.

**Harder:** reading `.env` (the effective size is a product, not a literal);
reasoning about the SL Hunting basket, where the BankNIFTY mirror already
roughly doubles risk — with multiplier `M` the basket sits near **2 × M** times
the single-leg budget.

Two known second-order effects, both documented rather than "fixed":

- The scaled budget also loosens the "one lot exceeds the budget" skip.
- Because lots are floored, a 2× can land slightly above a pure doubling while
  staying strictly inside the scaled budget.

**To revisit when:** a global scale is genuinely wanted — it should then be an
explicit, separately-named control with its own confirmation, not a default.

## Action items

- [x] `_scaled_int` / `_scaled_float` / `_strategy_size_multiplier` at read time.
- [x] Malformed → 1 for paper, blocked from live.
- [x] The two deliberately unscaled knobs documented in code and in
      [`../lld/risk-and-safety.md`](../lld/risk-and-safety.md) §4.
- [x] Drift-guard test: a new strategy reading a size knob with the raw `_env_*`
      helpers fails the build.
