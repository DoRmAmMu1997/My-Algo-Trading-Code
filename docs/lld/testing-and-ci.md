# LLD — Test architecture, coverage budgets, and CI

**Owns:** `Tests/` · `.github/workflows/quality-and-security.yml` ·
`scripts/check_coverage_thresholds.py` · the `[tool.*]` sections of `pyproject.toml` ·
`.pre-commit-config.yaml`
**Related ADRs:** [0009](../adr/0009-importlib-loading-for-spaced-filenames.md), [0010](../adr/0010-tests-in-a-mirrored-tests-tree.md)

---

## 1. Layout

**Every test lives under `Tests/`, mirroring the source tree.** Runtime folders
contain only runtime code.

```
Tests/
├── test_nifty_multi_strategy_master.py        487 tests — the master runner
├── test_market_data_health.py                  26 tests — feed validation/freshness
├── Data Extractors/
│   └── test_index_fetch_construction.py
├── Dependencies/
│   ├── conftest.py                            puts SOURCE Dependencies/ on sys.path
│   ├── test_broker_contract.py                test_check_env_config.py
│   ├── test_dhan_token_setup.py               test_diagnostic_preflight.py
│   ├── test_execution_ledger.py               test_next_open_entry.py
│   ├── test_order_splitting.py                test_repository_policy.py
│   ├── test_risk_sizing.py                    test_secret_redaction.py
│   ├── test_startup_exposure.py               test_tick_bar_builder.py
│   ├── test_trading_lifecycle.py
│   ├── Dhan API/test_dhan_execution.py
│   └── Flattrade API/test_flattrade_execution.py
└── Signal Generators/
    ├── test_deterministic_strategy_safety.py  test_renko_bounds.py
    ├── test_trading_bot_ports.py
    ├── CPR AI Agent/          conftest.py + 4 suites
    ├── CPR Strategy/          1 suite
    ├── Regime Adaptive Strategy/ conftest.py + 1 suite
    ├── SL Hunting AI Agent/   conftest.py + 8 suites
    └── Subhamoy Strategies/   1 suite
```

**36 test files, 4 conftests, 1602 tests.**

### 1.1 Two rules when adding a test

1. **Put it at the mirrored path.** A test for `Dependencies/foo.py` goes to
   `Tests/Dependencies/test_foo.py`.
2. **Give it a repository-unique filename.** There are no `__init__.py` files, so
   pytest's `prepend` import mode keys modules by *basename*. Two files named
   `test_utils.py` anywhere in the tree would collide.

### 1.2 The conftests

Four folders carry a `conftest.py`. Each one puts the **source** folder on
`sys.path` — never the test folder:

| Conftest | Inserts |
|---|---|
| `Tests/Dependencies/` | `Dependencies/` |
| `Tests/Signal Generators/CPR AI Agent/` | `Signal Generators/CPR AI Agent/` |
| `Tests/Signal Generators/SL Hunting AI Agent/` | `Signal Generators/SL Hunting AI Agent/` |
| `Tests/Signal Generators/Regime Adaptive Strategy/` | that folder **and** `Signal Generators/` |

They exist because those source folders have spaces in their names and their
modules import each other by bare name (`import sl_hunting_tools`,
`from check_env_config import audit`). Pointing at the source folder means the
tests exercise **the same import resolution production uses**.

Each conftest inserts **only what it needs**. A repository-wide `sys.path` entry
would let a test resolve an import production never performs, hiding a missing
dependency or an accidental cross-strategy coupling — which is exactly what the
CPR AI conftest's comment warns about.

The Regime Adaptive one adds the parent too, because `regime_common` re-exports
shared indicators from `misc_strategy_common` one level up.

---

## 2. Running the suites

```bash
python -m unittest Tests.test_nifty_multi_strategy_master
```

```bash
python -m unittest Tests.test_market_data_health
```

```bash
python -m pytest "Tests/Signal Generators" "Tests/Dependencies" "Tests/Data Extractors" -q
```

The two `unittest` suites are invoked as dotted module paths (`Tests` resolves as
a PEP 420 namespace package). They are kept separate from the pytest run so the
three counts stay independently verifiable — running `pytest Tests` would collect
the unittest `TestCase` classes as well and merge the numbers.

---

## 3. Why the master suite is the master's only real gate

mypy cannot see `nifty_multi_strategy_master.py` — its filename
is not a valid module name ([ADR-0009](../adr/0009-importlib-loading-for-spaced-filenames.md)).
So the 17k-line runner is covered by:

- `compileall` — syntax and consistency,
- `Tests/test_nifty_multi_strategy_master.py` — 487 cases loading it via
  `importlib` with `dhanhq` mocked.

That suite is therefore not optional detail; it is the type checker's stand-in.

---

## 4. Coverage budgets

Branch coverage is enabled globally (`[tool.coverage.run] branch = true`,
`concurrency = ["thread"]` because safety paths execute in worker threads,
`relative_files = true` so the JSON report is identical on Windows and Linux).

Coverage.py has exactly **one** global `fail_under`, so the stricter per-module
budgets are enforced from `coverage.json` by
`scripts/check_coverage_thresholds.py`:

| Tier | Floor | Modules |
|---|---|---|
| Repository floor | **70%** | everything (CI measures **70.2%**; original MAT-110 baseline was 54.7%) |
| Safety / data-safety | **90%** | `broker_contract`, `execution_ledger`, `startup_exposure`, `trading_lifecycle`, `market_data_health`, `tick_bar_builder`, `next_open_entry`, `risk_sizing`, `order_splitting`, `secret_redaction` |
| Broker adapters | **80%** | Kotak, Shoonya, Flattrade, Dhan |

Two deliberate properties of the checker:

- **A module missing from the coverage report is a FAILURE**, not a pass.
  Otherwise renaming or deleting a file would silently retire its budget.
- **Every live adapter must have a `BROKER_THRESHOLDS` row.** A broker added
  without one escapes the 80% policy — which is exactly what happened when the
  Dhan adapter first landed. Adding the row is part of adding a broker.

`omit` excludes `Tests/`, the reference backtests, and the vendored `NorenApi.py`.

---

## 5. CI

`.github/workflows/quality-and-security.yml`, on every push and pull request,
across **Python 3.12 and 3.13** (3.13 is the operator's live runtime; 3.12 catches
newer-only syntax). `permissions: contents: read` — the workflow only reads.

**Job `verify`:**

| Step | Purpose |
|---|---|
| install core + dev + ai + codex-ai, `pip check` | one resolvable environment |
| `pip_audit --local` | audits the clean resolved tree, not a developer's system Python |
| `pre_commit validate-config` | the hook config itself is valid |
| branch-enabled coverage over all three suites + threshold script | the budgets above |
| `compileall` | the syntax gate for every file, including those outside mypy |
| `ruff check .` | lint |
| `mypy` | scoped in `pyproject.toml` to identifier-named modules |
| `bandit -r .` | security; B101/B105/B110 skipped, vendored + reference code excluded |

**Job `broker-dependencies`:** installs `requirements-brokers.txt` in its own
clean environment and re-runs the broker contract and Flattrade adapter suites.
It is separate because Kotak's official `v2.0.1` tag pins older pandas/requests
that conflict with the audited core set — asking pip to build that combined graph
is impossible, so the upstream environment is validated in isolation instead.

CI never runs the authenticated CPR AI smoke command, and never opens a real
socket — **a green build proves nothing about the websocket transport**, which is
why enabling it live requires clean paper sessions
([ADR-0005](../adr/0005-rest-vs-websocket-market-data.md)).

---

## 6. Policy tests

`Tests/Dependencies/test_repository_policy.py` tests the *repository*, not the
runtime. It asserts, without contacting any network:

- dependency sets are exact (`==` pins) and Kotak uses its official Git tag;
- CI runs the audit, branch coverage, and every dependency set;
- Dependabot updates pip and GitHub Actions weekly;
- coverage config stays branch-enabled at the 70% floor;
- every `cpr_ai_*.py` module is inside mypy's scope;
- `env.example` documents every `_env_*` key the code reads (>300 found, as a
  sanity check that the AST walk still works);
- `CLAUDE.md` and `AGENTS.md` share one identical runtime section;
- architecture docs make both optional agents visible and carry no stale
  worker-roster claims;
- every committed ADR and LLD is linked from `docs/README.md`, and the index
  links nothing that does not exist;
- every relative link inside `docs/` resolves.

The last three are the **documentation staleness gate**, and they answer three
different ways docs rot:

| Failure | Caught by |
|---|---|
| A doc describes a roster or agent set the code no longer has | `test_current_architecture_docs_distinguish_core_from_optional_agents` — covers `README.md`, `Signal Generators/Readme.md`, `AGENTS.md`, `CLAUDE.md`, the master file, **and `docs/hld/system-overview.md`** |
| A new ADR/LLD lands unlinked, or the index points at a deleted file | `test_every_committed_design_document_is_linked_from_the_docs_index` (checked in both directions) |
| A rename breaks a cross-reference between documents | `test_relative_links_inside_the_committed_docs_resolve` |

`docs/superpowers/` is excluded from all three: it is gitignored session working
material, not product documentation ([ADR-0011](../adr/0011-committed-docs-untracked-superpowers.md)).

Each of these was verified to **fail** on a deliberate mutation before being
committed — an unlinked ADR, a dangling index link, a renamed cross-reference,
and an agent dropped from the HLD. A policy test that cannot fail is worse than
no test, because it reads like coverage.

---

## 7. Local gates

```bash
pip install -r requirements.txt
```

Then the same commands CI runs (§2 and §5). `.pre-commit-config.yaml` wires
check-only hooks — ruff, merge-conflict, YAML, large-file, debug-statement.
**Policy: hooks check, never rewrite**, so a commit never changes content the
author did not review. Install once with `pre-commit install`.

---

## 8. Testing conventions

- **No network, ever.** Broker HTTP, browser flows, Telegram, Google Sheets and
  both LLM SDKs are mocked or injected. `test_broker_contract.py` spawns
  short-lived subprocesses to verify `sys.path` semantics, which is the one place
  a subprocess is legitimate.
- **Optional dependencies skip, not fail.** Broker/SDK-specific cases skip when
  the dependency is absent.
- **Fixtures are small and literal.** Safety tests assert on specific candles and
  quantities rather than generated data, so a failure names the case.
- **Add the test before the change** for anything in the safety tier.
