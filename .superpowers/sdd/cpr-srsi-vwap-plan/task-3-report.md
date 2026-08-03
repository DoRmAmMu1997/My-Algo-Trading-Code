# Task 3 report: master worker, mechanical management, and one-time scale-in

## Result

Task 3 is implemented in two reviewed TDD commits:

- `537e79cde738da0c249391c9aed7e730752fddaa` — independent CPR AI context/cadence foundation.
- `ccebf1a681a15c63f36e19153c263057f3afd48c` — mechanical management, live-safe one-time scale-in, combined bookkeeping, and focused integration tests.

The pre-existing Task 4 documentation/configuration changes remain dirty and
uncommitted. No real Codex SDK, broker, order, or network call was made.

## RED evidence

The focused tests were written and observed failing before their production
behavior was added. Representative failures were:

- Prior regime: `TypeError: build_cpr_context() got an unexpected keyword argument 'prior_accepted_regime'`.
- Direct inheritance: expected `AtmSingleLegStrategyWorker`, received `CPRAlgo3StrategyWorker`.
- Completed-bar cadence: expected one `agent.decide` call, received zero.
- CPR sidecar/mechanics: `AttributeError: module 'master_file' has no attribute 'CPRAITradeState'`.
- Startup coexistence: paper-only and CPR/CPR Algo 3 coexistence errors were still returned.
- Mechanical ordering: the agent was called once on a bar already consumed by an SRSI exit.
- Every-poll protection: `_run_prebar_safety` did not exist.
- Weighted exit: realized P&L was `250.0`, expected `300.0` after a paper add.
- Audit result: only one logger write existed, while the final execution outcome required a second record.
- Paper add retry semantics: failed pricing incorrectly consumed `scale_in_used`.
- Reversal stage two: `_initialize_trade_state` did not accept frozen CPR levels for the following milestone.

## GREEN evidence

Final verification from the Task 3 worktree:

- `python -m pytest "Signal Generators/CPR AI Agent/tests" -q` — **53 passed**.
- `python -m pytest "Signal Generators/CPR Strategy" -q` — **17 passed**.
- `python -m unittest test_nifty_multi_strategy_master` — **430 passed, 51 skipped**.
- Targeted Ruff for the five changed Python files — **All checks passed**.
- `python -m py_compile` for the five changed Python files — **passed**.
- `git diff --check` — **passed**.

The only pytest output outside pass counts was the repository's existing
`dateutil` deprecation warning.

## Files changed

- `Nifty Multi Strategy Front Test - Master File.py`
  - Replaced the old Algo arbiter with a direct ATM worker.
  - Added isolated optional-module loading, independent five-minute cadence,
    accepted-regime memory, allowlisted position context, and audit handling.
  - Added CPR-AI sidecar state, every-poll hard protection, completed-bar SRSI
    and trailing rules, staged stop ratchets, and same-bar reversal prevention.
  - Added one-time paper/live role-`A` scale-in using the locked primary
    contract, ledger-aware two-leg exit, and aggregate MTM/P&L bookkeeping.
  - Restored ordinary CPR/CPR Algo 3 coexistence and the normal live double gate.
- `Signal Generators/CPR AI Agent/cpr_ai_context.py`
  - Added validated `prior_accepted_regime` to `session_levels`.
- `Signal Generators/CPR AI Agent/cpr_ai_signals.py`
  - Passed prior accepted regime through the freeze boundary.
- `Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py`
  - Added prior-regime context coverage.
- `test_nifty_multi_strategy_master.py`
  - Added focused architecture, cadence, cutoff, mechanical-exit, ratchet,
    audit, paper/live scale-in, weighted P&L, and startup coexistence coverage.

## Safety notes

- New exposure remains fail-closed on stale context, audit failure, malformed
  optional runtime, and ambiguous broker outcomes.
- A live add is marked used immediately before its first broker submission;
  rejected, partial, and unknown results never create a paper add.
- Both live ledger legs must be broker-confirmed flat before local state clears
  or the locked option subscription is removed.
- Risk-reducing exits continue even when decision logging fails.

## Fix round 1

Commit `167023b44bcd815e9af99730583e47532475e163` closes the three
review findings without touching the Task 4 documentation/configuration work.

### Findings fixed

- Scale-in mode now follows the primary position's execution provenance. A
  live-enabled worker whose rejected primary fell back to paper books a paper
  add and never calls the broker; a broker-backed primary continues through
  role `A` with no paper fallback for rejected, partial, or unknown outcomes.
- Flat-entry host gates are rechecked after the potentially slow agent turn:
  stop event, lifecycle/entry permission, market-data health, square-off time,
  and the 15:00 entry cutoff all block a late submission and are audited.
- Partial and unknown role-`A` exposure is included conservatively in open MTM
  and max-loss using ledger `risk_quantity`. Realized P&L uses the broker's
  entry average when present and a nonzero same-contract fallback otherwise.
  Local state still clears only after both live legs are broker-confirmed flat.

### RED evidence

The new focused tests failed before the production patch with these observed
reproductions:

- A live-enabled paper-fallback primary returned `False` and entered the live
  scale-in path instead of booking its paper add.
- A cutoff or shutdown/lifecycle transition during inference still called
  `enter_position` after the turn completed.
- Primary `50@10` plus partial add `25@12` marked at `8` reported `-100`
  instead of `-200`; unknown add exposure was likewise omitted.

### GREEN evidence

- Focused CPR AI worker class: **20 passed**.
- `python -m pytest "Signal Generators/CPR AI Agent/tests" -q`: **53 passed**.
- `python -m pytest "Signal Generators/CPR Strategy" -q`: **17 passed**.
- `python -m unittest test_nifty_multi_strategy_master`: **435 passed, 51 skipped**.
- Targeted Ruff for the master and master-test files: **All checks passed**.
- `python -m py_compile` for the master and master-test files: **passed**.
- `git diff --check`: **passed**.

No real Codex SDK, broker, order, or network call was made; the apparent live
order text in the master suite comes from fully mocked diagnostic tests.

## Fix round 2

Commit `872da09ad71e924aa4896ab8ad05d2b505d67d2a` corrects the
time-dependent test defect and adds the missing safety characterizations. No
production file changed in this round, and the Task 4 files remained untouched.

### RED evidence

At the actual after-15:00 IST wall clock, the unpinned 20-test CPR AI worker
class reproduced **8 failures and 2 errors**. The real cutoff short-circuited
ordinary cadence, accepted-entry, scale-in, and shutdown-race tests before
their intended behavior could run.

### Test corrections and added coverage

- `TestCPRAIWorkerFoundation.setUp` now pins `_ist_now` to 10:00 IST and
  restores it with `addCleanup`, so every ordinary test is deterministic.
- The 15:15 test locally advances that pinned clock during the agent turn;
  existing 15:00 tests retain their explicit cutoff overrides.
- New post-inference tests prove market-data-unhealthy and 15:15 square-off
  transitions reject the late entry and audit their exact blocked reasons.
- A two-live-ledger retry test proves one unconfirmed role-`A` leg retains the
  primary position, CPR state, and option subscription. A later retry with
  both legs confirmed flat performs one cleanup and books the hand-checked
  primary `50@10` plus add `25@12` exit at `8` as `-200`.

The three new tests characterized already-correct production behavior and did
not require a production edit.

### GREEN evidence

- At actual wall time `2026-08-03T15:12:26+05:30`, the deterministic CPR AI
  worker class completed **23 passed**.
- `python -m pytest "Signal Generators/CPR AI Agent/tests" -q`: **53 passed**.
- `python -m pytest "Signal Generators/CPR Strategy" -q`: **17 passed**.
- `python -m unittest test_nifty_multi_strategy_master`: **438 passed, 51 skipped**.
- Targeted Ruff for `test_nifty_multi_strategy_master.py`: **All checks passed**.
- `python -m py_compile test_nifty_multi_strategy_master.py`: **passed**.
- `git diff --check -- test_nifty_multi_strategy_master.py`: **passed**.

No real Codex SDK, broker, order, or network call was made. Live-order-looking
master-suite output came only from the repository's mocked diagnostic tests.

## Fix round 3

Commit `26e8cf82a727010fdaaa0666a3065b6865a1330c` closes the
post-inference scale-in race without changing EXIT's fail-open path.

### RED evidence

The focused live-scale-in test transitioned the worker during the agent turn.
Before the production patch, stop-event, lifecycle-shutdown, and unhealthy
market-data cases all reached real scale-in bookkeeping, set
`scale_in_used=True`, and reached the mocked broker boundary. Each subtest
failed on the hand-checked expectation that add state remain untouched.

The EXIT control passed during RED, proving the test isolated only the
exposure-increasing branch.

### Fix and coverage

- The post-inference entry helper is now a shared exposure-increase gate used
  by both flat entry and scale-in.
- A blocked accepted scale-in records `SCALE_IN_BLOCKED`, `submitted=False`,
  and the exact host reason. It is never reported submitted or confirmed.
- Tests cover `stop_event`, `worker_not_running`, `market_data_unhealthy`, and
  the preserved 15:00 `entry_cutoff`, with no add state or broker call.
- Accepted EXIT remains outside that gate and still executes after stop/feed
  transitions because reducing risk must remain fail-open.

### GREEN evidence

- At actual wall time `2026-08-03T15:23:29+05:30`, the deterministic CPR AI
  worker class completed **25 passed**.
- `python -m pytest "Signal Generators/CPR AI Agent/tests" -q`: **53 passed**.
- `python -m pytest "Signal Generators/CPR Strategy" -q`: **17 passed**.
- `python -m unittest test_nifty_multi_strategy_master`: **440 passed, 51 skipped**.
- Targeted Ruff for the master and master-test files: **All checks passed**.
- `python -m py_compile` for the master and master-test files: **passed**.
- `git diff --check`: **passed**.

The Task 4 files remained untouched. No real Codex SDK, broker, order, or
network call was made; live-order-looking suite output came from mocks.
