# ADR-0010: Consolidate tests into a mirrored top-level `Tests/` tree

**Status:** Accepted
**Date:** 2026-08-10
**Deciders:** repository owner

## Context

Test files were scattered across the repository, co-located with the code they
exercise:

```
test_nifty_multi_strategy_master.py                     (repo root, 504 KB)
test_market_data_health.py                              (repo root)
Dependencies/test_*.py                                  (11 files)
Dependencies/Dhan API/test_dhan_execution.py
Dependencies/Flattrade API/test_flattrade_execution.py
Data Extractors/test_index_fetch_construction.py
Signal Generators/test_*.py                             (3 files)
Signal Generators/*/test_*.py  and  */tests/*.py        (14 files)
Signal Generators/*/conftest.py                         (3 files)
```

Consequences of that layout: `ls Dependencies/` showed 11 test files against 15
runtime modules; the two largest files in the repository root were test suites;
and "run the tests" required naming four different paths.

Every one of these files is location-dependent. They anchor on `__file__` to
find the repo root or the module under test, or they rely on pytest's `prepend`
import mode inserting the test's own directory into `sys.path` so a bare
`from check_env_config import audit` resolves.

## Decision

Move every test file and every test `conftest.py` into a top-level `Tests/`
directory whose internal structure **mirrors the source tree**:

```
Tests/
  test_nifty_multi_strategy_master.py
  test_market_data_health.py
  Dependencies/
    test_broker_contract.py …
    Dhan API/test_dhan_execution.py
    Flattrade API/test_flattrade_execution.py
  Data Extractors/test_index_fetch_construction.py
  Signal Generators/
    test_renko_bounds.py …
    CPR AI Agent/           conftest.py + tests
    CPR Strategy/
    Regime Adaptive Strategy/
    SL Hunting AI Agent/    conftest.py + tests
    Subhamoy Strategies/
```

Each folder that mirrors a spaced-name source folder carries a `conftest.py`
that puts the corresponding **source** folder — not the test folder — on
`sys.path`, preserving the bare-name sibling imports the strategy modules use at
runtime.

Path anchors are rewritten to point back at the source tree
(`Path(__file__).resolve().parents[N]`), and every runner path is updated: the CI
workflow (both jobs), `pyproject.toml` (Ruff per-file-ignores, coverage `omit`),
and the documented commands in `README.md`, `CLAUDE.md` and `AGENTS.md`.

## Options considered

### Option A: Mirrored `Tests/` tree (chosen)

**Pros:** one obvious home for tests; a 1:1 mapping from a source path to its
test path; component folders show only runtime code; each conftest keeps the
narrow, deliberate `sys.path` scope the co-located ones had; test basenames stay
unique, which pytest's `prepend` import mode requires without `__init__.py`.
**Cons:** a test is no longer adjacent to its subject; the path anchors are one
level further from what they point at; the move itself touches every test file.

### Option B: Flat `Tests/` folder

**Pros:** simplest possible layout; one conftest.
**Cons:** ~28 files in one directory with no grouping, and a single conftest
would have to insert **every** source directory on `sys.path`. That is exactly
what the existing conftests deliberately avoid: a repository-wide path lets a
test pass through an import production never uses, hiding a missing dependency
or an accidental cross-strategy coupling.

### Option C: Leave tests co-located

**Pros:** zero risk; a test sits next to its subject.
**Cons:** the status quo being changed — no single place to look, component
folders dominated by test files, and the repository root led by two large test
suites.

## Trade-off analysis

Adjacency (Option C) is a real benefit and is what is being given up. It is
traded for discoverability and for a clean separation between runtime code and
test code in a repository where the runtime is the product.

Option B was rejected on a safety argument rather than an aesthetic one: the
per-folder conftests are load-bearing. `Signal Generators/CPR AI Agent/conftest.py`
inserts *only* that agent's directory precisely so a test cannot resolve an
import that production would fail on. A single flat conftest would erase that
property for every strategy at once.

The main risk of this change is silent test loss — a file that stops being
collected still looks like a green build. That is mitigated by comparing exact
counts before and after (see Action items), not by inspection.

## Consequences

**Easier:** finding tests; running everything (`python -m pytest Tests -q`);
reading a component folder; keeping runtime and test code visually separate.

**Harder:** a test is no longer adjacent to its subject; path anchors are
indirect and must be right; **test file basenames must remain unique
repository-wide**, since pytest keys modules by basename with no `__init__.py`
present; and anyone adding a test must mirror the source path rather than drop
the file next to the code.

**To revisit when:** `__init__.py` files are introduced (that would change
pytest's import behaviour and relax the basename constraint), or if the
mirroring proves to drift from the source tree in practice.

## Action items

- [x] Move all test files and test conftests with `git mv` so history follows.
- [x] Rewrite every `__file__`-relative anchor to point at the source tree.
- [x] One `conftest.py` per mirrored spaced-name folder, each inserting only the
      source folder it needs.
- [x] Update the CI workflow (both jobs), `pyproject.toml`, `README.md`,
      `CLAUDE.md`, `AGENTS.md` and the per-folder Readmes.
- [x] **Verify by exact count, not by "it passed":** 487 master + 26
      market-data-health + 1089 pytest = 1602 before; the same after.
