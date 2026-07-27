# MAT-111 SL Hunting Post-Exit Guard Design

## Problem

SL Hunting trades a basket that can contain a NIFTY leg and a mechanically
mirrored BankNIFTY leg. The agent may close either leg independently when that
leg's premise is invalidated.

The current cooldown starts in `after_exit()`, which runs when the NIFTY leg
closes. If the BankNIFTY leg remains open, the timer starts before the trade is
flat and can expire while exposure still exists. Closing the BankNIFTY leg later
can then be followed by an immediate re-entry, defeating the cooldown's purpose.
The same premature start can occur when a tied BankNIFTY close is partial or
indeterminate.

Two adjacent live-entry controls also fail closed incompletely:

- the code fallback for the documented noon cutoff is 10:30;
- malformed cooldown and no-new-entry values can fall back to defaults without
  disabling live mode.

The exchange freeze-limit splitter and its metadata are intentional future
groundwork for size multipliers up to 25. MAT-111 does not modify or wire that
code.

## Required Behavior

The cooldown applies only after an SL Hunting trade closes because of target,
stop-loss, or premise invalidation. Operational shutdown closes such as max-loss
and square-off may reach the same flat transition, but their lifecycle already
blocks re-entry.

A trade is closed only when all locally tracked SL Hunting exposure is flat:

- the NIFTY position is inactive; and
- the BankNIFTY mirror is inactive, including no partial or indeterminate live
  quantity retained for reconciliation.

The first successful entry marks a trade as open. Closing one leg while the
other remains open does not start the timer. The successful close of the final
leg starts the full configured interval. Repeated close/finalization calls for
the same trade do not restart or extend the timer.

The timer blocks entry only. It is never consulted by any NIFTY or BankNIFTY
exit path.

## State and Clock

`SLHuntingAIWorker` will keep two private fields:

- `_cooldown_trade_open: bool` records whether a successfully opened trade has
  not yet produced its one basket-flat cooldown transition.
- `_post_exit_cooldown_deadline_monotonic: float | None` stores the enforcement
  deadline.

`enter_position()` sets `_cooldown_trade_open` only after the NIFTY entry
succeeds. `_arm_post_exit_cooldown_if_flat()` checks both legs, consumes the
open-trade marker exactly once, and records `time.monotonic() + minutes * 60`.
It is called after the NIFTY exit hook and after a confirmed BankNIFTY mirror
close so either leg can be the final leg.

`post_exit_cooldown_remaining_seconds()` returns zero before the first completed
trade, when disabled, or after expiry. Otherwise it returns the non-negative
difference between the deadline and `time.monotonic()`. Wall-clock changes
therefore cannot shorten or extend an active cooldown.

## Guard Failure Policy

The master executor accepts a finite, non-negative remaining-seconds value.

- In live mode, an exception, non-numeric value, NaN, infinity, or negative
  value rejects the new entry with a stable safety reason.
- In paper mode and in the standalone paper runner, the same guard failure stays
  fail-open so experiments are not turned into a trading outage.
- A worker without the optional cooldown hook continues to work as before.

This policy affects entries only; exits do not call the cooldown hook.

## Configuration

The in-code defaults for `SL_HUNTING_NO_NEW_ENTRY_HOUR` and
`SL_HUNTING_NO_NEW_ENTRY_MINUTE` will be corrected to `12` and `0`, matching
`env.example`, README, AGENTS.md, and CLAUDE.md.

For a live-enabled SL Hunting worker, `_live_config_errors()` will validate:

- `SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES` as a non-negative integer;
- `SL_HUNTING_NO_NEW_ENTRY_HOUR` as an integer from 0 through 23;
- `SL_HUNTING_NO_NEW_ENTRY_MINUTE` as an integer from 0 through 59;
- the resolved cooldown and resolved cutoff fields against the same bounds.

Paper mode retains the forgiving environment helpers.

## Verification

Focused regressions will prove:

- no cooldown exists before a completed trade;
- a NIFTY-only close does not start the timer while BankNIFTY remains open;
- a BankNIFTY-only close does not start the timer while NIFTY remains open;
- closing the final leg starts the full interval;
- a partial or indeterminate final close does not start the timer;
- repeated finalization does not extend the interval;
- the timer uses monotonic time;
- a broken or non-finite guard blocks live entries but not paper entries;
- exits remain available while the guard is active or broken;
- missing no-new-entry configuration resolves to noon;
- malformed or negative cooldown/cutoff configuration disables live mode.

The complete repository test, coverage, dependency-audit, lint, type, compile,
Bandit, and pre-commit gates will run before publication. A diff-scoped Codex
Security scan will cover the final MAT-111 change.
