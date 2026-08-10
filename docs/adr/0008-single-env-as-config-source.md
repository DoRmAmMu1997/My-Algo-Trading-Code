# ADR-0008: A single `.env` as the only configuration source

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

The system has on the order of 300+ settings: per-strategy sizes, risk budgets,
time windows, poll intervals, broker credentials, feed selection, notifier
tokens, agent toggles. It runs on one machine, for one operator, one session a
day.

The question is where those settings live, and how the operator knows what is in
force.

## Decision

**One gitignored `Dependencies/.env`, copied from the committed
`Dependencies/env.example` template, is the single source of truth.** Nothing is
hard-coded per run, and there are no CLI overrides, profiles, or
environment-specific layers.

Values are read through `_env_str` / `_env_bool` / `_env_int` / `_env_float`, or
`_scaled_int` / `_scaled_float` for size-bearing knobs. **Never ad-hoc
`os.getenv`.**

Two mechanisms keep the three places a setting can live from drifting apart:

- `python algo.py check-env` — a read-only audit reporting keys missing from
  `.env`, mistyped/stale keys, and knobs missing from the template. It prints
  key **names only**, never values, so its output is safe to share.
- A CI gate (`test_every_env_setting_the_code_reads_is_documented_in_env_example`)
  that fails the build when a new `_env_*` key lands without an `env.example`
  entry. It imports the *same* helpers the operator command uses, so the two can
  never disagree.

## Options considered

### Option A: One `.env` + template + drift audit (chosen)

**Pros:** one place to look and one to audit; secrets stay out of git by
construction; trivially diffable; no framework; the same file describes paper
and live runs, so nothing changes shape when going live.
**Cons:** a large flat file; discovery depends entirely on `env.example` staying
current (hence the CI gate); no typing beyond the helpers; no per-environment
layering.

### Option B: Layered config (YAML base + environment overrides)

**Pros:** structure, typing, per-environment defaults.
**Cons:** "what is actually in force?" becomes a merge question. For a
single-machine system that is pure cost. It also fits secrets badly — they end
up in a separate mechanism anyway, so the count of configuration sources goes
*up*, not down.

### Option C: CLI flags for the volatile settings

**Pros:** explicit at launch; nothing left over from yesterday.
**Cons:** live-ness would become invisible to `check-env`, and the launch
command would become the real config — undocumented and unreviewable. It also
moves risk decisions to the moment of starting, when the operator is least
likely to be reviewing them.

### Option D: A settings module in Python

**Pros:** typed, IDE-navigable, no parsing.
**Cons:** secrets in a tracked file, or a second mechanism for secrets. Changing
a lot size would become a code change.

## Trade-off analysis

The decisive question is: **when something goes wrong mid-session, how many
places must be checked to know what the runner is doing?**

Option A answers "one file, and one command that audits it". Every alternative
answers "two or more, plus a merge rule". For live-money code debugged under
time pressure by one person, that is worth more than typing or structure.

The genuine weakness Option A accepts is the silent-default problem: a key
present in the code and the template but **missing from `.env` is not an error**
— the runner just uses the in-code default. Nothing complains, and an unseen
default ends up governing a live-money run. `check-env` exists for exactly that
failure mode, and it is the reason this ADR would be unsafe without it.

## Consequences

**Easier:** knowing what a run will do; diffing a config change; keeping secrets
out of git; auditing before a live session.

**Harder:** discovering settings (mitigated by `env.example` + the CI gate);
running two different configurations simultaneously (not supported, and not
wanted); type errors are runtime, not authoring, errors.

**To revisit when:** the system runs in more than one environment at once, or a
second operator needs a different configuration on the same machine.

## Action items

- [x] `env.example` is the only discovery surface; holds **blank placeholders**.
- [x] All reads through the `_env_*` / `_scaled_*` helpers.
- [x] `python algo.py check-env` — read-only, names-only output, non-zero exit on
      findings so it can gate a pre-flight script.
- [x] CI gate: code → template, one direction only. The reverse would flag the
      ~200 `<PREFIX>_*` knobs built from f-strings, which the AST cannot see.
- [x] The gate asserts the AST walk finds >300 keys, so a renamed helper cannot
      make it silently pass while checking nothing.
