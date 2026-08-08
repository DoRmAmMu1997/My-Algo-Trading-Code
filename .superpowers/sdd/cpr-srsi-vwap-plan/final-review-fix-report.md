# Final whole-branch fix report

## Scope and safety boundary

- Started from clean handoff `deb6118844fa36c3cc17f56671fe25bb8aaa7578`
  on `codex/cpr-codex-ai-groundwork` and implemented the consolidated final
  review wave only.
- Read the repository instructions, approved plan, final-review brief, TDD
  instructions, and the pinned `openai-codex==0.144.4` implementation before
  changing behavior.
- No authenticated Codex/model call, broker connection, or order call was
  made. The only runner command used the synthetic fake path and printed
  `NO ORDER`. Nothing was pushed.

## RED evidence recorded before production edits

The first production edit for each behavior followed a failing focused test.

| Area | Focused RED result | Failure proved |
| --- | --- | --- |
| Completed bars, deterministic cutoff, and indicators | `python -m pytest "Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py" -q` -> 7 failed, 20 passed | The context had no explicit `as_of`; a start-stamped forming minute, duplicate slot, and missing slot could be mishandled; the freezer did not share the cutoff; EWM produced the wrong Wilder seed (`78.2739977` instead of literal `70`); and the old oscillating Stoch RSI K expectation was `11.0362` rather than the independently derived `13.2986`. |
| Codex auth/profile isolation, timeout, and sanitizer | Focused runtime selection -> 7 failed | There was no auth-only `CODEX_HOME`, no missing-auth pre-launch failure, the subprocess still used a fixed 90-second timeout instead of configured `0.25`/`135`, invalid timeouts were accepted, and sanitizer substring matching removed the legitimate `ordered` field. |
| Worker cadence, P&L rows, and actual execution provenance | Focused `test_nifty_multi_strategy_master` selection -> 6 failed and 2 errors in 6 selected methods/subtests | The master lacked deterministic `as_of`/completed-bucket identity, had no CPR AI sheet row mapping, could call again after a true-up, and labeled entry/exit rows from configured live mode instead of actual broker/position provenance. |
| Plural camel-case credential key found during final re-review | Focused decision-log test -> 1 failed | `accessTokens` survived recursive sanitization and appeared in JSONL. |
| Host-blocked entry found during final re-review | Focused master provenance test -> 1 failed | A host-blocked, never-submitted entry was mislabeled `LIVE_REJECTED` instead of `NOT_SUBMITTED`. |

The last two REDs were discovered after the initial GREEN pass, added as
regressions, observed failing, and fixed before the final full reruns.

## Implemented corrections

### Completed-bar and indicator correctness

- One aware IST `as_of` is threaded through one-minute completion filtering,
  five-minute resampling, context freezing, worker bucket identity, and content
  signature calculation.
- A five-minute candle is accepted only after the forming minute is excluded
  and each exact constituent minute appears once. Duplicates remain visible
  until this completeness check instead of being silently collapsed.
- Immutable session/bucket identity enforces one inference per completed bar;
  the separate OHLC signature still rejects an output made stale during an
  in-flight true-up. A later true-up cannot consume the bucket again.
- RSI now uses the literal first-14-change SMA seed and Wilder recursion.
  TradingView Stoch RSI defaults remain RSI 14, stochastic 14, K 3, D 3 with
  20/80 boundaries, including explicit flat and monotonic handling.

### Codex isolation and fixed capability boundary

- Pinned official source inspected:
  - `codex-rs/core/src/config/mod.rs`: `CODEX_HOME` owns Codex state and an
    explicit home must exist/canonicalize.
  - `codex-rs/login/src/auth/storage.rs`: file authentication is stored and may
    be refreshed at `$CODEX_HOME/auth.json`.
  - `codex-rs/core/config.schema.json`: every defense-in-depth feature key used
    below is supported by the pinned tag.
  - the Python SDK launches its bundled absolute Codex binary with the supplied
    child environment.
- The parent creates one process-lifetime temporary auth-only Codex home, copies
  the operator `auth.json` exactly once, and reuses it under the existing
  one-inference lock. A refresh written by the isolated child therefore
  survives subsequent turns without overwriting the operator profile. The
  temporary home is cleaned at process exit; no symlink or reverse sync exists.
- Every turn still uses its own temporary cwd and synthetic HOME, USERPROFILE,
  AppData, and TEMP surfaces. Operator config, MCP servers, plugins, skills,
  apps, and rules are absent. Missing source authentication fails before child
  launch.
- Defense-in-depth disables only pinned-schema-supported feature keys:
  `apps`, `plugins`, `connectors`, `enable_mcp_apps`, `plugin_sharing`,
  `remote_plugin`, `browser_use`, `browser_use_external`, `in_app_browser`,
  `computer_use`, `skill_search`, `skill_mcp_dependency_install`,
  `tool_search`, and the existing shell/collaboration features. The MCP
  inventory remains exactly the four zero-argument CPR read tools.
- The validated positive configured timeout now reaches the subprocess; no
  fixed 90-second child deadline remains.

### Operations and audit accuracy

- `_PNL_SHEET_ROW_LABELS` now maps `CPR AI` to `CPR AI Agent Strategy`, allowing
  the shared updater to address PAPER, `[LIVE]`, and `[MIXED]` rows. The focused
  operator README names all three labels.
- Final entry and exit JSONL rows derive mode from actual execution and retained
  position provenance: `PAPER`, `LIVE`, `PAPER_FALLBACK`, or conservative
  `LIVE_INDETERMINATE`. Exit provenance is captured before confirmed exit state
  is cleared. A host block with no submission is recorded as `NOT_SUBMITTED`.
- Token-aware recursive sanitization preserves `next_levels.ordered`,
  `authoritative_geometry`, and `token_usage` while removing access-token,
  authentication, broker, order, venue, and API-key families, including plural
  camel-case variants.

## Final GREEN verification

### Tests, smoke, and coverage

| Command | Exit | Fresh result |
| --- | ---: | --- |
| `python -m pytest "Signal Generators/CPR AI Agent/tests" -q` | 0 | 67 passed, 1 third-party warning |
| `python -m pytest "Signal Generators/CPR Strategy" -q` | 0 | 17 passed, 1 third-party warning |
| `python -m unittest test_nifty_multi_strategy_master` | 0 | 487 passed, 52 skipped |
| `python -m unittest test_market_data_health` | 0 | 26 passed |
| `python -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q` | 0 | 1085 passed, 1 third-party warning |
| `python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake` | 0 | `HOLD validation=accepted_hold NO ORDER` |

The exact branch-enabled coverage chain (`coverage erase`, master unittest,
market-health unittest, full Signal Generators/Dependencies/Data Extractors
pytest, JSON export, report, and policy check) passed. Final totals were 18,768
statements, 4,980 missed statements, 6,012 branches, 1,088 partial branches,
and **69.6% total branch coverage**. `scripts/check_coverage_thresholds.py`
reported: `Coverage policy passed for all safety and broker modules.`

### Static, security, dependency, and repository gates

| Command | Exit | Fresh result |
| --- | ---: | --- |
| `python -m compileall -q . -x "(__pycache__|Backtest Outputs|\\.git)"` | 0 | Clean |
| `python -m ruff check .` | 0 | All checks passed |
| `python -m mypy` | 0 | Success: no issues in 53 source files |
| `python -m bandit -r . -q -x "./Backtest Outputs,./My Backtest Files (For Reference),./Dependencies/Shoonya API/NorenApi.py" --skip B101,B105,B110` | 0 | No findings; Bandit comment-parser warnings only |
| `python -m pre_commit run --all-files` | 0 | All five configured hooks passed |
| `python -m pip_audit -r requirements.txt --no-deps --progress-spinner off` | 0 | No known vulnerabilities |
| `python -m pip_audit -r requirements-ai.txt --no-deps --progress-spinner off` | 0 | No known vulnerabilities |
| `python -m pip_audit -r requirements-codex-ai.txt --no-deps --progress-spinner off` | 0 | No known vulnerabilities |
| `python -m pip_audit -r requirements-dev.txt --no-deps --progress-spinner off` | 0 | No known vulnerabilities |
| `git diff --check` | 0 | Clean |

## Final re-review and residual operational notes

The final whole-diff re-review found the two extra edge cases recorded above;
both were fixed through fresh RED/GREEN cycles. The post-fix re-review found no
remaining correctness, live-safety, isolation, audit, or documentation finding
in this wave.

The authenticated smoke was deliberately not run because it would make a real
model call. The PowerShell host prints its existing execution-policy warning
when it cannot load the user's profile; this does not affect command exits.
The only test warning is the existing third-party `dateutil` warning, and the
only Bandit output is its comment-parser warning. Operators must create the
three documented sheet rows and retain paper-first enablement before any live
use.
