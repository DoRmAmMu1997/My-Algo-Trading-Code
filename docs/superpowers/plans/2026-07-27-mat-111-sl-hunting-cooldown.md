# MAT-111 SL Hunting Post-Exit Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SL Hunting cooldown begin exactly once when the complete NIFTY/BankNIFTY trade becomes flat, while keeping first entries and every exit path unblocked.

**Architecture:** Keep the behavior inside the existing `SLHuntingAIWorker` and `MasterWorkerExecutor` seams. Replace wall-clock elapsed-time enforcement with one monotonic deadline, arm it only on a tracked trade's basket-flat transition, validate its output at the entry choke point, and extend the existing live-config validator rather than adding a new configuration system.

**Tech Stack:** Python 3.12/3.13, `unittest`, `pytest`, existing environment helpers, Ruff, mypy, Coverage.py, Bandit, pre-commit, pip-audit.

## Global Constraints

- The exchange freeze-limit splitter and its metadata are intentional future groundwork and must not be modified.
- The cooldown starts only after a successfully opened SL Hunting trade becomes fully flat.
- A partial, unknown, or failed close that leaves either leg tracked must not start the cooldown.
- First entries and all exit paths remain unaffected.
- Live entry fails closed on corrupt cooldown state; paper and standalone entry remain fail-open.
- `SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES=0` remains the supported disable value.
- The documented no-new-entry default remains 12:00 IST.
- Live flags remain off during development and verification.

---

### Task 1: Arm the cooldown on the basket-flat transition

**Files:**
- Modify: `Nifty Multi Strategy Front Test - Master File.py`
- Test: `test_nifty_multi_strategy_master.py`

**Interfaces:**
- Consumes: `SLHuntingAIWorker.enter_position(...) -> bool`, `PaperPosition.active`, and `time.monotonic()`.
- Produces: `SLHuntingAIWorker._arm_post_exit_cooldown_if_flat() -> None` and `SLHuntingAIWorker.post_exit_cooldown_remaining_seconds() -> float`.

- [ ] **Step 1: Write failing basket-transition tests**

Add focused tests to `TestSLHuntingAIWorker` that use literal monotonic values:

```python
def test_cooldown_waits_for_final_mirror_close(self):
    worker, _ = self._make_worker()
    with patch.object(master_file.time, "monotonic", return_value=100.0):
        self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
        worker.exit_nifty_leg_only("NIFTY_PREMISE_INVALID")
    self.assertIsNone(worker._post_exit_cooldown_deadline_monotonic)

    with patch.object(master_file.time, "monotonic", return_value=200.0):
        worker.exit_bnf_mirror_only("BNF_PREMISE_INVALID")
    self.assertEqual(
        worker._post_exit_cooldown_deadline_monotonic,
        200.0 + master_file.SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES * 60.0,
    )


def test_cooldown_waits_for_final_nifty_close(self):
    worker, _ = self._make_worker()
    self.assertTrue(worker.enter_position("LONG", 24300.0, 24290.0, 24400.0))
    worker.exit_bnf_mirror_only("BNF_PREMISE_INVALID")
    self.assertIsNone(worker._post_exit_cooldown_deadline_monotonic)

    with patch.object(master_file.time, "monotonic", return_value=300.0):
        worker.exit_position("NIFTY_PREMISE_INVALID")
    self.assertEqual(
        worker._post_exit_cooldown_deadline_monotonic,
        300.0 + master_file.SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES * 60.0,
    )
```

Add a live partial-close regression using the existing scripted broker fake. It
must assert that the deadline remains `None` after the retained mirror position
and becomes a full new interval only after the retry confirms flat. Add an
idempotence assertion that a second no-op close does not change the deadline.

- [ ] **Step 2: Run the new tests and verify the old implementation fails**

Run:

```powershell
python -m unittest `
  test_nifty_multi_strategy_master.TestSLHuntingAIWorker.test_cooldown_waits_for_final_mirror_close `
  test_nifty_multi_strategy_master.TestSLHuntingAIWorker.test_cooldown_waits_for_final_nifty_close -v
```

Expected: failure because `_post_exit_cooldown_deadline_monotonic` does not
exist and the old `_last_exit_at` is armed on the first NIFTY-leg close.

- [ ] **Step 3: Implement the smallest basket-flat state machine**

Initialize:

```python
self._cooldown_trade_open = False
self._post_exit_cooldown_deadline_monotonic: float | None = None
```

After a successful NIFTY entry:

```python
if ok:
    self._cooldown_trade_open = True
```

Add:

```python
def _arm_post_exit_cooldown_if_flat(self) -> None:
    if not self._cooldown_trade_open:
        return
    if self.pos.active or self._mirror_pos.active:
        return
    self._cooldown_trade_open = False
    if SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES <= 0:
        self._post_exit_cooldown_deadline_monotonic = None
        return
    self._post_exit_cooldown_deadline_monotonic = (
        time.monotonic() + SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES * 60.0
    )
```

Call the helper after a confirmed mirror close and at the end of
`after_exit()`. Replace the wall-clock calculation with:

```python
deadline = self._post_exit_cooldown_deadline_monotonic
if SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES <= 0 or deadline is None:
    return 0.0
return max(deadline - time.monotonic(), 0.0)
```

- [ ] **Step 4: Run focused and surrounding worker tests**

Run:

```powershell
python -m unittest test_nifty_multi_strategy_master.TestSLHuntingAIWorker -v
```

Expected: all SL Hunting worker tests pass.

- [ ] **Step 5: Commit the basket transition**

```powershell
git add -- "Nifty Multi Strategy Front Test - Master File.py" test_nifty_multi_strategy_master.py
git commit -m "fix: arm SL Hunting cooldown when basket is flat" `
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 2: Validate the cooldown result at the entry boundary

**Files:**
- Modify: `Signal Generators/SL Hunting AI Agent/sl_hunting_executor.py`
- Test: `Signal Generators/SL Hunting AI Agent/tests/test_sl_hunting_agent.py`

**Interfaces:**
- Consumes: optional `worker.post_exit_cooldown_remaining_seconds()` and `worker.live_trading`.
- Produces: an entry rejection dictionary for corrupt live cooldown state.

- [ ] **Step 1: Replace the fail-open-only test with explicit live and paper cases**

Add a `live_trading` flag to the raising test worker and prove the two policies:

```python
def test_broken_cooldown_hook_blocks_live_entry_but_not_paper():
    class _Boom(_FakeWorker):
        def __init__(self, live_trading):
            super().__init__()
            self.live_trading = live_trading

        def post_exit_cooldown_remaining_seconds(self):
            raise RuntimeError("clock unavailable")

    live = MasterWorkerExecutor(_Boom(True))
    rejected = live.enter("LONG", stop=1, target=2, reason="x", price=25000)
    assert rejected["accepted"] is False
    assert "cooldown safety check unavailable" in rejected["reason"].lower()
    assert live._w.entries == []

    paper = MasterWorkerExecutor(_Boom(False))
    assert paper.enter("LONG", stop=1, target=2, reason="x", price=25000)["accepted"] is True
```

Parametrize `float("nan")`, `float("inf")`, `float("-inf")`, and `-1.0` as
corrupt live values. Keep the existing test proving a worker without the hook
still enters and the existing test proving exits do not consult the hook.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m pytest `
  "Signal Generators/SL Hunting AI Agent/tests/test_sl_hunting_agent.py" `
  -k "cooldown" -q
```

Expected: the live raising-hook and non-finite cases fail because the current
executor converts failures to zero and enters.

- [ ] **Step 3: Implement finite, non-negative validation**

Import `math`. In `MasterWorkerExecutor.enter()`, treat exceptions and invalid
values as a guard failure. Return the following only when
`bool(getattr(self._w, "live_trading", False))` is true:

```python
{
    "accepted": False,
    "reason": (
        "cooldown safety check unavailable; live entry rejected. "
        "Exits remain available."
    ),
}
```

Paper mode continues to zero the remaining interval on a guard failure. Valid
positive values keep the existing human-readable cooldown rejection.

- [ ] **Step 4: Run the complete executor test module**

Run:

```powershell
python -m pytest `
  "Signal Generators/SL Hunting AI Agent/tests/test_sl_hunting_agent.py" -q
```

Expected: all tests in the module pass.

- [ ] **Step 5: Commit the entry-boundary hardening**

```powershell
git add -- `
  "Signal Generators/SL Hunting AI Agent/sl_hunting_executor.py" `
  "Signal Generators/SL Hunting AI Agent/tests/test_sl_hunting_agent.py"
git commit -m "fix: fail closed on corrupt live cooldown state" `
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 3: Align and validate SL Hunting entry-control configuration

**Files:**
- Modify: `Nifty Multi Strategy Front Test - Master File.py`
- Test: `test_nifty_multi_strategy_master.py`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `Signal Generators/SL Hunting AI Agent/README.md`
- Modify: `Signal Generators/SL Hunting AI Agent/sl_hunting_doc.md`

**Interfaces:**
- Consumes: `_live_config_errors(worker, "SL_HUNTING")`.
- Produces: noon defaults and live-mode errors for malformed or invalid
  `SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES` and
  `SL_HUNTING_NO_NEW_ENTRY_{HOUR,MINUTE}`.

- [ ] **Step 1: Write failing configuration tests**

In `TestStrictLiveConfiguration`, add:

```python
def test_sl_hunting_entry_controls_are_strictly_validated_for_live(self):
    worker = self._Worker("SL Hunting AI")
    worker.no_new_entry_hour = 12
    worker.no_new_entry_minute = 0
    cases = {
        "SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES": "-1",
        "SL_HUNTING_NO_NEW_ENTRY_HOUR": "24",
        "SL_HUNTING_NO_NEW_ENTRY_MINUTE": "60",
    }
    for name, raw in cases.items():
        with self.subTest(name=name), patch.dict(os.environ, {name: raw}):
            errors = master_file._live_config_errors(worker, "SL_HUNTING")
        self.assertTrue(any(name in error for error in errors), errors)
```

Add resolved-value cases for a negative cooldown and invalid resolved cutoff.
Add a worker-level behavior test that deletes the two environment variables,
loads a fresh master module through the test's existing loader, and proves a
flat worker starts skipping new inference at 12:00 rather than 10:30.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m unittest `
  test_nifty_multi_strategy_master.TestStrictLiveConfiguration.test_sl_hunting_entry_controls_are_strictly_validated_for_live -v
```

Expected: failure because the three raw settings are not currently part of the
live-config rule set.

- [ ] **Step 3: Add the narrow validation rules and correct defaults**

Change the worker defaults to:

```python
no_new_entry_hour = _env_int("SL_HUNTING_NO_NEW_ENTRY_HOUR", 12)
no_new_entry_minute = _env_int("SL_HUNTING_NO_NEW_ENTRY_MINUTE", 0)
```

For `normalized_prefix == "SL_HUNTING"`, extend `raw_rules` with a
`nonnegative_integer` rule for the cooldown and existing `integer_range` rules
for the cutoff. Add the `nonnegative_integer` branch to the shared loop. Validate
the resolved module cooldown plus the worker's resolved no-new-entry fields.

- [ ] **Step 4: Update operator documentation without changing splitter docs**

Document that the cooldown starts when the whole basket is flat, uses the final
leg's close time, and rejects corrupt live entry state. Keep `AGENTS.md` and
`CLAUDE.md` byte-identical from “What this project is” downward.

- [ ] **Step 5: Run focused configuration and documentation checks**

Run:

```powershell
python -m unittest `
  test_nifty_multi_strategy_master.TestStrictLiveConfiguration `
  test_nifty_multi_strategy_master.TestSLHuntingAIWorker -v
python algo.py check-env
python -c "from pathlib import Path; a=Path('AGENTS.md').read_text(encoding='utf-8').split('## What this project is',1)[1]; c=Path('CLAUDE.md').read_text(encoding='utf-8').split('## What this project is',1)[1]; assert a == c"
```

Expected: tests pass, `check-env` reports no repository/template drift, and the
documentation parity assertion exits zero.

- [ ] **Step 6: Commit configuration and docs**

```powershell
git add -- `
  "Nifty Multi Strategy Front Test - Master File.py" `
  test_nifty_multi_strategy_master.py `
  AGENTS.md CLAUDE.md `
  "Signal Generators/SL Hunting AI Agent/README.md" `
  "Signal Generators/SL Hunting AI Agent/sl_hunting_doc.md"
git commit -m "fix: validate SL Hunting entry controls for live mode" `
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 4: Verify, security-scan, and publish MAT-111

**Files:**
- Modify only if a verification failure identifies a MAT-111 regression.
- Create ignored scan artifacts outside committed secret-bearing paths.

**Interfaces:**
- Consumes: the complete `origin/main..HEAD` diff.
- Produces: a green local gate, a diff-scoped security conclusion, and one PR
  from `codex/mat-111-sl-hunting-cooldown` to `main`.

- [ ] **Step 1: Run all repository gates**

Run the repository commands documented in `AGENTS.md`: both unittest suites,
all repository pytest suites, branch-enabled coverage plus
`scripts/check_coverage_thresholds.py`, pip-audit, Ruff, mypy, compileall,
Bandit, and pre-commit.

- [ ] **Step 2: Review the exact diff**

Run:

```powershell
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git status --short
```

Confirm no freeze-limit splitter file or code path changed and no generated
test/coverage artifacts are tracked.

- [ ] **Step 3: Run the Codex Security diff workflow**

Use the `codex-security:security-diff-scan` workflow on `origin/main...HEAD`.
Complete its threat-model, discovery, validation, and attack-path phases for
every candidate it produces, and keep generated scan artifacts ignored.

- [ ] **Step 4: Push and open the PR**

```powershell
git push -u origin codex/mat-111-sl-hunting-cooldown
gh pr create --base main --head codex/mat-111-sl-hunting-cooldown `
  --title "MAT-111: harden SL Hunting post-exit cooldown" `
  --body-file .github-mat-111-pr-body.md
```

The PR body must include the basket-flat invariant, the first-entry/exit
non-regression, strict live-config behavior, full local verification evidence,
security conclusion, and Codex co-authorship.

- [ ] **Step 5: Verify hosted checks and mergeability**

Watch the PR checks to a terminal result. Fix only MAT-111 regressions, rerun the
relevant local gate, push normally, and confirm all required Python 3.12/3.13
jobs pass and GitHub reports the PR mergeable.
