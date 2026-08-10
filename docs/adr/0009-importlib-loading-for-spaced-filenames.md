# ADR-0009: Load spaced-name files with `importlib` instead of renaming them

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

Many files in this repository have spaces in their names:

```
Nifty Multi Strategy Front Test - Master File.py
Signal Generators/Nifty Supertrend Signal Generator.py
Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py
Signal Generators/SL Hunting AI Agent/          ← the folder, too
```

Python cannot import these with a normal dotted import: the filename is not a
valid module name, and neither is a directory with spaces.

The names are not accidental — the author navigates the repository by them, and
they read as descriptions rather than identifiers. The live track record,
external references, and the operator's own muscle memory all point at these
paths.

## Decision

Keep the names. Load spaced-name files through
`load_module(module_name, file_path)`, a wrapper around
`importlib.util.spec_from_file_location`, and add explicit `sys.path` bootstraps
where sibling modules import each other by bare name.

Accept the consequences deliberately:

- **mypy scope is limited to identifier-named modules.** Files with spaces
  cannot be mypy modules. They are covered by `compileall` (syntax) plus the
  unittest suite (behaviour) instead.
- **Tests need explicit path bootstraps.** Each test folder for a spaced-name
  source folder carries a `conftest.py` that puts the corresponding **source**
  folder on `sys.path`.
- **The master's own test suite loads it via `importlib`** with `dhanhq` mocked.

## Options considered

### Option A: Keep the names, load via `importlib` (chosen)

**Pros:** no rename churn across a live-money repository; paths in the README,
the CLI, external references and the operator's habits all keep working; the
loader is ~10 lines.
**Cons:** mypy cannot see the largest file in the repository; every test folder
needs a path bootstrap; imports do not look like normal Python; tooling
configuration (`pyproject.toml`, CI, Bandit excludes) has to name paths with
spaces, which means quoting everywhere.

### Option B: Rename everything to snake_case

**Pros:** ordinary imports; full mypy coverage including the master runner; no
conftest bootstraps; no quoting.
**Cons:** a very large, purely mechanical diff across live-money code, touching
every doc, every CLI example and every import site at once. The risk is not the
rename itself — it is that a rename this broad makes any *behavioural* change
hidden inside it invisible in review. Against a system running real money, that
is the wrong trade for a cosmetic gain.

### Option C: Add a packaging layer (`__init__.py`, a `[project]` section)

**Pros:** proper package semantics; imports resolve normally.
**Cons:** does not solve the problem — a package still cannot contain a module
whose *filename* has spaces. It would add packaging machinery on top of the
`importlib` loading that would still be required. Also, `__init__.py` files
would change pytest's `rootdir` insertion behaviour and break the bare-name
sibling imports the strategy folders rely on.

## Trade-off analysis

Option B is the technically correct end state and is the one that would be
chosen for a new repository. It is not chosen here because the *timing* is
wrong: the cost is a repository-wide diff against running live code, and the
benefit — mypy coverage of the master file — is partially available another way
(`compileall` plus a 487-case unittest suite).

The honest summary: this is a decision to defer a cleanup, not a claim that
spaces are good. The cost is paid continuously in small amounts (quoting,
conftests, mypy scope) rather than once in a large amount.

If the master runner is ever split into modules — the other significant piece of
debt named in the HLD — that is the natural moment to rename, because the files
are being rewritten anyway and the diff is already under review.

## Consequences

**Easier:** navigating by descriptive names; keeping every documented path,
README example and CLI invocation stable.

**Harder:**
- mypy is scoped to identifier-named modules only (`pyproject.toml` lists them
  explicitly, and a policy test asserts that every `cpr_ai_*.py` module is inside
  that scope so a new one cannot silently escape).
- Every shell command and tooling exclude needs quoting.
- Test folders need `conftest.py` path bootstraps, and each one must insert
  **only** the folder it needs — a repository-wide path would let tests pass
  through imports production never uses and hide a missing dependency.
- Test file basenames must stay unique repository-wide, because pytest's
  `prepend` import mode with no `__init__.py` files keys modules by basename.

**To revisit when:** the master runner is split into modules. Do the rename
then, in the same review, not before.

## Action items

- [x] `load_module()` in the master runner (~L1141).
- [x] mypy `files` + `mypy_path` list every identifier-named module and root.
- [x] `compileall` covers every Python file, spaced names included.
- [x] Per-folder `conftest.py` bootstraps under `Tests/`, each inserting only
      the source folder it needs ([ADR-0010](0010-tests-in-a-mirrored-tests-tree.md)).
- [x] Bandit and Ruff excludes name the vendored and reference-only paths.
