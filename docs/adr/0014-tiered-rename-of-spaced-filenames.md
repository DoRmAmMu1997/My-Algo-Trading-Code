# ADR-0014: Rename spaced-name files in reviewable tiers, master last

**Status:** Accepted
**Date:** 2026-09-02
**Deciders:** repository owner
**Supersedes:** [ADR-0009](0009-importlib-loading-for-spaced-filenames.md)

## Context

[ADR-0009](0009-importlib-loading-for-spaced-filenames.md) chose to keep spaced
filenames and load them with `importlib`, and it considered and rejected the
rename as "Option B". Its reasoning is accepted in full and is worth quoting,
because this ADR does not disagree with it:

> Option B is the technically correct end state and is the one that would be
> chosen for a new repository. It is not chosen here because the *timing* is
> wrong: the cost is a repository-wide diff against running live code... The
> risk is not the rename itself — it is that a rename this broad makes any
> *behavioural* change hidden inside it invisible in review.

That objection is to **one repository-wide diff**, not to renaming. ADR-0009 set
its own revisit trigger — "when the master runner is split into modules" — on
the assumption that the split is what would make the diff reviewable.

A tech-debt audit on 2026-09-02 established two things ADR-0009 did not have:

1. **The master split is not being done.** The HLD names it "the one piece of
   debt worth naming explicitly", and it remains unattempted because it would be
   a multi-week rewrite of live execution code. Waiting for it means waiting
   indefinitely.
2. **The rename decomposes.** Only **33** files have a space in the *basename*,
   not the 79 with a space anywhere in the path — spaced *folders* do not block
   mypy, and 19 files inside them are already in its scope. Those 33 fall into
   tiers with sharply different risk, from five files nothing references at all
   to the 18,568-line master runner.

Tiers make the diff reviewable without the split. That answers ADR-0009's actual
objection rather than overriding it.

## Decision

Rename spaced **basenames** to snake_case, in tiers, a few files per PR, each
independently reviewable. Keep spaced **folder** names — they cost nothing that
matters (see Consequences).

**Tier 4 — no code references** (this PR). Five files mentioned only in
`Signal Generators/Readme.md` and one folder README.

**Tier 3 — standalone scripts.** `Data Extractors/` fetchers and
`My Backtest Files (For Reference)/` backtests: `subprocess`-launched from
`algo.py`, never imported. Each also appears in Ruff, coverage and Bandit
exclude lists, which must move in the same commit.

**Tier 1 — the 18 signal generators loaded into the live master.** Each has one
path literal in the master plus one to three in tests. `Nifty CPR Algo 3 Signal
Generator.py` goes last in this tier: it depends on the
`sys.modules.setdefault("cpr_strategy_logic", ...)` alias the master registers.

**Tier 2 — the master runner, last.** Renamed on its own, *without* being added
to mypy in the same PR. Measured cost of putting it in scope: **288 errors**
(193 `attr-defined`, 25 `arg-type`, 17 `assignment`). Those are worked down in
later batches by running mypy on the file by path; only then does it join
`[tool.mypy] files`.

**Naming.** Drop the redundant `nifty_` prefix and use
`<descriptor>_signal_generator.py`, matching the identifier-named siblings that
already exist in the same folders (`cpr_strategy_logic.py`,
`goldmine_strategy_logic.py`, `money_machine_strategy_logic.py`).

**`load_module()` stays.** It takes `(module_name, file_path)` as independent
arguments and caches on the *name*, never the path, so a rename changes only a
path literal. The loader needs no modification at any tier, and mixed spaced and
renamed files coexist indefinitely. That is what makes the tiers safe to stop
between.

## Consequences

**Easier:** each renamed file can enter mypy's scope; imports look like ordinary
Python; the diff at every step is small enough that a behavioural change hidden
inside it would be visible — the thing ADR-0009 was protecting.

**Harder / accepted:**

- Documented paths change. `algo.py` invocations, README examples and the
  operator's muscle memory all move. This is the cost ADR-0009 declined to pay
  and this ADR accepts, in instalments.
- Every rename must move its non-import references in the same commit:
  `pyproject.toml` (`[tool.mypy] files`, Ruff `per-file-ignores`,
  `[tool.coverage.run] omit`), `scripts/check_coverage_thresholds.py`, the CI
  workflow's Bandit excludes, `algo.py`, and the docs tables.
- `Tests/Dependencies/test_repository_policy.py` asserts the master is
  deliberately *outside* mypy. At Tier 2 that assertion must be **inverted**,
  not edited — it encodes ADR-0009's position.
- Folder names keep their spaces, so `conftest.py` bootstraps, quoted shell
  paths and quoted tooling config all remain. Renaming folders was considered
  and rejected: adapters import `broker_contract` by **bare** name, relying on
  `load_module`'s temporary `sys.path` insert, so converting them to dotted
  imports would change behaviour on the live-order path for a cosmetic gain.
- ADR-0009's constraint survives unchanged: **test file basenames must stay
  unique repository-wide**, because pytest's `prepend` import mode with no
  `__init__.py` files keys modules by basename.

**Unchanged:** `compileall` still covers every Python file, spaced or not, and
the unittest suites remain the behavioural gate for anything outside mypy.

**To revisit when:** nothing. This supersedes ADR-0009 outright. If the master
runner is later split, Tier 2 simply becomes easier.

## Action items

- [x] Tier 4: rename the five files with no code references.
- [x] Delete the dead `BACKTEST_PATH` constant in
      `Tests/Signal Generators/CPR Strategy/test_cpr_strategy_signal_generators.py`,
      which pointed at `Signal Generators/CPR Strategy/Nifty CPR Strategy
      Backtest.py` — a path that has never existed, since that backtest lives in
      `My Backtest Files (For Reference)/`.
- [ ] Tier 3: `Data Extractors/` and `My Backtest Files (For Reference)/`.
- [ ] Tier 1: the 18 loaded signal generators, CPR Algo 3 last.
- [ ] Tier 2a: rename the master runner, without adding it to mypy.
- [ ] Tier 2b: work the 288 mypy errors down, then add it to `[tool.mypy] files`
      and invert the policy assertion.
