# CPR AI Beginner Documentation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain the CPR SRSI/VWAP Codex agent in beginner-friendly module, class, function, and inline documentation without changing executable behavior.

**Architecture:** Treat the package as three teaching layers: the agent-facing contract, the deterministic/runtime safety core, and the master-worker integration. Edit only comments and docstrings, then prove semantic equivalence to commit `9d40d010ba9ea19ef33ae2f6481b5f729c28913d` by comparing abstract syntax trees after docstring nodes are removed.

**Tech Stack:** Python 3.12/3.13, pandas, Pydantic, MCP, `openai-codex`, unittest, pytest, Coverage.py, Ruff, mypy, Bandit, pre-commit, Git, and GitHub CLI.

## Global Constraints

- Only comments and docstrings may change in implementation or test Python files.
- Do not change statements, expressions, constants, annotations, imports, schemas, prompt text, environment defaults, public interfaces, test expectations, or trading behavior.
- Explain purpose, inputs, outputs, authority, failure behavior, and safety rationale in plain English; do not narrate obvious syntax.
- Preserve deterministic host authority over levels, geometry, risk, timing, orders, and exits.
- Preserve the exact four no-argument MCP tools and the isolated authentication-only Codex runtime.
- Preserve all existing CPR, CPR Algo 3, and other strategy behavior.
- Every commit must end with `Co-authored-by: Codex <codex@openai.com>`.
- No authenticated Codex call, broker call, or order may occur during verification.

## File Responsibility Map

- `Signal Generators/CPR AI Agent/cpr_ai_schema.py`: strict model-decision and position-state contracts.
- `Signal Generators/CPR AI Agent/cpr_ai_tools.py`: immutable four-tool context registry.
- `Signal Generators/CPR AI Agent/cpr_ai_prompt.py`: modular agent knowledge and authority boundary.
- `Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py`: read-only MCP exposure of one frozen snapshot.
- `Signal Generators/CPR AI Agent/cpr_ai_signals.py`: small host-facing context-freezing facade.
- `Signal Generators/CPR AI Agent/cpr_ai_runner.py`: order-free synthetic smoke entry point.
- `Signal Generators/CPR AI Agent/conftest.py`: test import-path setup only.
- `Signal Generators/CPR AI Agent/cpr_ai_context.py`: deterministic completed-bar, indicator, level, and structure calculations.
- `Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py`: host-side isolated subprocess and authentication-home lifecycle.
- `Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py`: SDK-side thread configuration, tool evidence, and structured output extraction.
- `Signal Generators/CPR AI Agent/cpr_ai_agent.py`: inference cadence, required-tool validation, and deterministic host policy.
- `Signal Generators/CPR AI Agent/cpr_ai_decision_log.py`: sanitized append-only audit records.
- `Nifty Multi Strategy Front Test - Master File.py`: worker timing, mechanical exits, execution, scale-in, reconciliation, configuration, and P&L wiring.
- CPR AI test modules and `test_nifty_multi_strategy_master.py`: behavioral examples and safety-regression fixtures.

---

### Task 1: Explain the Agent-Facing Contract and Thin Adapters

**Files:**
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_schema.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_tools.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_prompt.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_signals.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_runner.py`
- Modify: `Signal Generators/CPR AI Agent/conftest.py`
- Test: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py`
- Test: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py`

**Interfaces:**
- Consumes: the existing `CPRAgentDecision`, `CPRPositionState`, `FrozenCPRContextRegistry`, `build_system_prompt`, and four frozen tool names.
- Produces: documentation that teaches how model output, host state, MCP snapshots, and the order-free smoke path relate; no interface changes.

- [ ] **Step 1: Expand module and public-interface docstrings**

Use docstrings with this content pattern, adapted to each real module/function:

```python
"""Expose one frozen host snapshot through four read-only MCP tools.

The worker calculates every market fact before Codex starts.  This module only
publishes deep-copied evidence; it cannot fetch prices, place orders, or change
the snapshot while the model is reasoning.
"""
```

Cover the schema's forbidden execution fields, position-state validation, deep-copy behavior, modular prompt seams, parser entry point, and why the synthetic runner can never create an order.

- [ ] **Step 2: Add inline comments at non-obvious adapter boundaries**

Add plain-English comments immediately before:

- strict Pydantic cross-field validation;
- position-state allowlisting and falsey-payload rejection;
- deep-copy returns from the frozen registry;
- MCP closure creation over one immutable payload;
- prompt-section assembly that keeps future discretionary knowledge separate;
- the fake/authenticated smoke split and explicit `NO ORDER` output.

Do not comment enum declarations, straightforward dictionary literals, or obvious return statements.

- [ ] **Step 3: Prove executable Python is unchanged**

Run this AST comparison for the edited files:

```powershell
$script = @'
import ast
import subprocess
from pathlib import Path

BASE = "9d40d010ba9ea19ef33ae2f6481b5f729c28913d"
PATHS = [
    "Signal Generators/CPR AI Agent/cpr_ai_schema.py",
    "Signal Generators/CPR AI Agent/cpr_ai_tools.py",
    "Signal Generators/CPR AI Agent/cpr_ai_prompt.py",
    "Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py",
    "Signal Generators/CPR AI Agent/cpr_ai_signals.py",
    "Signal Generators/CPR AI Agent/cpr_ai_runner.py",
    "Signal Generators/CPR AI Agent/conftest.py",
]

class WithoutDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return self.generic_visit(node)

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body

def normalized(source: str) -> str:
    tree = WithoutDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)

for path in PATHS:
    before = subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    after = Path(path).read_text(encoding="utf-8")
    if normalized(before) != normalized(after):
        raise SystemExit(f"Executable AST changed: {path}")
print("Documentation-only AST check passed for agent-facing modules.")
'@
$script | python -
```

Expected: `Documentation-only AST check passed for agent-facing modules.`

- [ ] **Step 4: Run focused contract tests and static checks**

Run:

```powershell
python -m pytest "Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py" -q
python -m ruff check "Signal Generators/CPR AI Agent"
python -m compileall -q "Signal Generators/CPR AI Agent"
```

Expected: all tests pass and both static commands exit `0`.

- [ ] **Step 5: Commit the agent-facing documentation**

```powershell
git add -- "Signal Generators/CPR AI Agent/cpr_ai_schema.py" "Signal Generators/CPR AI Agent/cpr_ai_tools.py" "Signal Generators/CPR AI Agent/cpr_ai_prompt.py" "Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py" "Signal Generators/CPR AI Agent/cpr_ai_signals.py" "Signal Generators/CPR AI Agent/cpr_ai_runner.py" "Signal Generators/CPR AI Agent/conftest.py"
git diff --cached --check
git commit -m "docs(cpr-ai): explain agent-facing contracts" -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 2: Explain Deterministic Context, Isolation, and Host Arbitration

**Files:**
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_context.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_agent.py`
- Modify: `Signal Generators/CPR AI Agent/cpr_ai_decision_log.py`
- Test: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py`
- Test: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_core.py`
- Test: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py`

**Interfaces:**
- Consumes: completed one-minute candles, prior host-accepted regime, the strict decision schema, and a frozen four-tool snapshot.
- Produces: deterministic context, isolated structured inference, fail-closed host outcomes, and sanitized JSONL audit evidence; behavior remains unchanged.

- [ ] **Step 1: Expand deterministic-context explanations**

Document the sequence `prepare minutes -> exclude forming minute -> require exact five slots -> resample -> compute previous-session CPR -> compute Wilder RSI/StochRSI/VWAP/EMA -> confirm swings -> freeze context`.

Add block comments explaining:

- why a row count alone cannot prove a five-minute bar is complete;
- why Wilder RSI needs an initial simple-average seed;
- why equal-weight VWAP is a fallback when volume is unavailable;
- why swing points require future bars for confirmation;
- why prior regime is advisory memory while all levels remain host facts.

- [ ] **Step 2: Expand Codex isolation and subprocess explanations**

Document the process-lifetime authentication-only `CODEX_HOME`, copy-once `auth.json`, removal of operator profile surfaces, per-turn temporary working directory, fixed four-tool configuration, supported capability disables, configured deadline, and structured SDK item extraction.

Make clear that authentication refreshes survive only inside the temporary process home and are never synchronized back to the operator's real profile.

- [ ] **Step 3: Expand host-policy and audit explanations**

Document the order `schema -> tool evidence -> stale/cadence checks -> deterministic host policy -> execution boundary`. Explain sideways/trending gates, continuation versus reversal geometry, stop-width and one-R rejection, scale-in eligibility, late-result suppression, one-inference lock, and why model proposals can only reduce or select host-approved actions.

In the audit logger, explain token-aware sensitive-key matching, preservation of legitimate near-matches, recursive sanitization, append-only JSONL behavior, and best-effort logging that cannot stop mechanical risk handling.

- [ ] **Step 4: Prove executable Python is unchanged**

Run:

```powershell
$script = @'
import ast
import subprocess
from pathlib import Path

BASE = "9d40d010ba9ea19ef33ae2f6481b5f729c28913d"
PATHS = [
    "Signal Generators/CPR AI Agent/cpr_ai_context.py",
    "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py",
    "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py",
    "Signal Generators/CPR AI Agent/cpr_ai_agent.py",
    "Signal Generators/CPR AI Agent/cpr_ai_decision_log.py",
]

class WithoutDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return self.generic_visit(node)

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body

def normalized(source: str) -> str:
    tree = WithoutDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)

for path in PATHS:
    before = subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    after = Path(path).read_text(encoding="utf-8")
    if normalized(before) != normalized(after):
        raise SystemExit(f"Executable AST changed: {path}")
print("Documentation-only AST check passed for deterministic/runtime modules.")
'@
$script | python -
```

Expected: `Documentation-only AST check passed for deterministic/runtime modules.` Any `Executable AST changed` message blocks the task.

- [ ] **Step 5: Run focused deterministic/runtime tests**

```powershell
python -m pytest "Signal Generators/CPR AI Agent/tests" -q
python -m ruff check "Signal Generators/CPR AI Agent"
python -m mypy
```

Expected: the CPR AI suite passes, Ruff is clean, and mypy reports no issues in its configured source set.

- [ ] **Step 6: Commit the deterministic/runtime documentation**

```powershell
git add -- "Signal Generators/CPR AI Agent/cpr_ai_context.py" "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py" "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py" "Signal Generators/CPR AI Agent/cpr_ai_agent.py" "Signal Generators/CPR AI Agent/cpr_ai_decision_log.py"
git diff --cached --check
git commit -m "docs(cpr-ai): explain deterministic safety boundaries" -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 3: Explain Master-Worker Mechanics and Behavioral Tests

**Files:**
- Modify: `Nifty Multi Strategy Front Test - Master File.py` (CPR AI sections only)
- Modify: `test_nifty_multi_strategy_master.py` (CPR AI helpers/tests only)
- Modify: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py`
- Modify: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_core.py`
- Modify: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_master_integration.py`
- Modify: `Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py`

**Interfaces:**
- Consumes: `CPRAgent`, frozen deterministic context, shared market data, shared execution ledger, and normal live-trading double gates.
- Produces: beginner-readable worker lifecycle and test fixtures; no timing, execution, reconciliation, or assertion changes.

- [ ] **Step 1: Expand master CPR AI class and helper docstrings**

Explain configuration loading, prior-regime memory, completed-bucket identity versus content signature, flat/open decision matrices, mechanical-precheck ordering, post-inference rechecks, entry geometry copying, actual execution provenance, and why exits remain available when entries are frozen.

- [ ] **Step 2: Add inline comments at live-money safety transitions**

Add comments before the non-obvious blocks that:

- run hard stop/max-loss/final-target checks before inference;
- suppress a response for a closed/replaced position;
- recheck market state after a slow inference;
- ratchet breakeven/one-R/prior-candle trailing without loosening stops;
- reuse the locked option contract and original filled quantity for R1 scale-in;
- distinguish full, rejected, partial, and unknown add-on outcomes;
- retain both ledger legs until broker-confirmed flat;
- classify PAPER, LIVE, PAPER_FALLBACK, LIVE_INDETERMINATE, and NOT_SUBMITTED audit modes;
- write PAPER/LIVE/MIXED Google Sheet rows.

- [ ] **Step 3: Explain reusable test fixtures and scenario intent**

Add or expand docstrings for CPR AI helper builders, fake runners, worker setup, websocket forming-bar fixtures, live-ledger fixtures, and parameterized safety matrices. Add comments only where fixture state is intentionally unrealistic to isolate one safety boundary.

Do not comment individual assertions whose test name already explains the behavior.

- [ ] **Step 4: Prove executable Python is unchanged**

Run:

```powershell
$script = @'
import ast
import subprocess
from pathlib import Path

BASE = "9d40d010ba9ea19ef33ae2f6481b5f729c28913d"
PATHS = [
    "Nifty Multi Strategy Front Test - Master File.py",
    "test_nifty_multi_strategy_master.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_core.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_master_integration.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py",
]

class WithoutDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return self.generic_visit(node)

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body

def normalized(source: str) -> str:
    tree = WithoutDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)

for path in PATHS:
    before = subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    after = Path(path).read_text(encoding="utf-8")
    if normalized(before) != normalized(after):
        raise SystemExit(f"Executable AST changed: {path}")
print("Documentation-only AST check passed for worker and test modules.")
'@
$script | python -
```

Expected: `Documentation-only AST check passed for worker and test modules.`

- [ ] **Step 5: Run worker and integration tests**

```powershell
python -m unittest test_nifty_multi_strategy_master
python -m unittest test_market_data_health
python -m pytest "Signal Generators/CPR AI Agent/tests" -q
```

Expected: all three commands exit `0` with no failure.

- [ ] **Step 6: Commit the worker/test documentation**

```powershell
git add -- "Nifty Multi Strategy Front Test - Master File.py" "test_nifty_multi_strategy_master.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_core.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_master_integration.py" "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py"
git diff --cached --check
git commit -m "docs(cpr-ai): explain worker safety mechanics" -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 4: Verify, Review, and Publish the Documentation Pass

**Files:**
- Verify: `Signal Generators/CPR AI Agent/conftest.py`
- Verify: all eleven `Signal Generators/CPR AI Agent/cpr_ai_*.py` modules
- Verify: all four `Signal Generators/CPR AI Agent/tests/test_cpr_ai_*.py` modules
- Verify: `Nifty Multi Strategy Front Test - Master File.py`
- Verify: `test_nifty_multi_strategy_master.py`
- Verify: `.github/workflows/quality-and-security.yml`
- Publish: branch `codex/cpr-codex-ai-groundwork` to `origin`
- Create: draft pull request into `main`

**Interfaces:**
- Consumes: the complete documentation-only branch and existing quality/security configuration.
- Produces: a verified remote branch and draft PR; no code or broker/model side effects.

- [ ] **Step 1: Run the complete documentation-only AST comparison**

Run:

```powershell
$script = @'
import ast
import subprocess
from pathlib import Path

BASE = "9d40d010ba9ea19ef33ae2f6481b5f729c28913d"
PATHS = [
    "Signal Generators/CPR AI Agent/conftest.py",
    "Signal Generators/CPR AI Agent/cpr_ai_agent.py",
    "Signal Generators/CPR AI Agent/cpr_ai_codex_runner.py",
    "Signal Generators/CPR AI Agent/cpr_ai_codex_subprocess.py",
    "Signal Generators/CPR AI Agent/cpr_ai_context.py",
    "Signal Generators/CPR AI Agent/cpr_ai_decision_log.py",
    "Signal Generators/CPR AI Agent/cpr_ai_mcp_server.py",
    "Signal Generators/CPR AI Agent/cpr_ai_prompt.py",
    "Signal Generators/CPR AI Agent/cpr_ai_runner.py",
    "Signal Generators/CPR AI Agent/cpr_ai_schema.py",
    "Signal Generators/CPR AI Agent/cpr_ai_signals.py",
    "Signal Generators/CPR AI Agent/cpr_ai_tools.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_context.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_core.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_master_integration.py",
    "Signal Generators/CPR AI Agent/tests/test_cpr_ai_runtime.py",
    "Nifty Multi Strategy Front Test - Master File.py",
    "test_nifty_multi_strategy_master.py",
]

class WithoutDocstrings(ast.NodeTransformer):
    def _visit_body(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return self.generic_visit(node)

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body

def normalized(source: str) -> str:
    tree = WithoutDocstrings().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)

for path in PATHS:
    before = subprocess.run(
        ["git", "show", f"{BASE}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    after = Path(path).read_text(encoding="utf-8")
    if normalized(before) != normalized(after):
        raise SystemExit(f"Executable AST changed: {path}")
print("Documentation-only AST check passed for the complete editorial pass.")
'@
$script | python -
```

Expected: `Documentation-only AST check passed for the complete editorial pass.`

- [ ] **Step 2: Run the complete automated test and coverage gates**

```powershell
python -m unittest test_nifty_multi_strategy_master
python -m unittest test_market_data_health
python -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q
python -m coverage erase
python -m coverage run -m unittest test_nifty_multi_strategy_master
python -m coverage run --append -m unittest test_market_data_health
python -m coverage run --append -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q
python -m coverage json -o coverage.json
python scripts/check_coverage_thresholds.py coverage.json
```

Expected: all suites and the repository coverage policy pass.

- [ ] **Step 3: Run complete static, security, and smoke gates**

```powershell
python -m compileall -q . -x "(__pycache__|Backtest Outputs|\.git)"
python -m ruff check .
python -m mypy
python -m bandit -r . -q -x "./Backtest Outputs,./My Backtest Files (For Reference),./Dependencies/Shoonya API/NorenApi.py" --skip B101,B105,B110
python -m pre_commit run --all-files
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake
git diff --check main..HEAD
```

Expected: every command exits `0`; the smoke output includes `HOLD validation=accepted_hold NO ORDER`.

- [ ] **Step 4: Request a fresh read-only review**

Give the reviewer the design, this plan, base `9d40d010ba9ea19ef33ae2f6481b5f729c28913d`, current head, and the full documentation diff. Require explicit confirmation that comments are accurate, beginner-friendly, non-redundant, and that no executable AST changed. Resolve any Critical or Important finding before publication.

- [ ] **Step 5: Confirm branch and GitHub prerequisites**

```powershell
git status -sb
git merge-base --is-ancestor main HEAD
gh auth status
gh repo view --json nameWithOwner,defaultBranchRef,url
```

Expected: clean named branch, `main` is an ancestor, GitHub authentication is active, and the default branch is `main`.

- [ ] **Step 6: Push and create the draft pull request**

```powershell
git push -u origin codex/cpr-codex-ai-groundwork
```

Create a draft PR titled `Add independent CPR SRSI/VWAP Codex agent` into `main`. The body must describe the independent deterministic/nondeterministic boundary, host-side live safety, isolated Codex runtime, comments/docstrings pass, verification results, order-free automated validation, paper-first operational recommendation, required Google Sheet row setup, and `Co-authored-by: Codex <codex@openai.com>`.

- [ ] **Step 7: Verify the remote PR state**

```powershell
gh pr view --json number,title,url,isDraft,baseRefName,headRefName,headRefOid,mergeStateStatus
gh pr checks --watch
```

Expected: the draft PR targets `main`, uses `codex/cpr-codex-ai-groundwork`, points at the local head commit, and hosted checks reach a terminal state. Preserve the linked worktree for PR follow-up.
