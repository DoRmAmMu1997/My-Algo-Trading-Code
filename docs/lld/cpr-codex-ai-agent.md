# LLD — CPR Codex AI Agent (optional, Codex)

**Owns:** `Signal Generators/CPR AI Agent/` · `CPRAIWorker`, `CPRAITradeState` (master file)
**Status:** optional, **disabled by default** (`CPR_AI_ENABLED=false`) and
**live-disabled by default** (`CPR_AI_LIVE_TRADING=false`)
**Related ADR:** [0007 — LLM agents as opt-in workers](../adr/0007-llm-agents-as-opt-in-workers.md)
**Operator detail:** the folder's own [`README.md`](../../Signal%20Generators/CPR%20AI%20Agent/README.md)

---

## 1. Responsibility

An independent five-minute SRSI/VWAP strategy in which Codex judges regime,
setup, and premise exits, while the host owns every deterministic entry and risk
gate plus all execution.

It is *not* an arbiter over the other CPR strategies. Ordinary CPR, CPR Algo 3,
Regime Adaptive and CPR AI are **independent strategies that may run together
with independent positions and independent P&L**.

---

## 2. Modules

| File | Role |
|---|---|
| `cpr_ai_agent.py` | `CPRAgent`, `CPRHostPolicy`, `CPRAgentRunResult`, `CPRToolCallRecord` |
| `cpr_ai_context.py` | Builds the frozen per-bar context |
| `cpr_ai_signals.py` | `freeze_cpr_context` — the snapshot boundary |
| `cpr_ai_tools.py` | `FrozenCPRContextRegistry`, `EXPECTED_TOOL_NAMES` |
| `cpr_ai_mcp_server.py` | In-process MCP server exposing the four tools |
| `cpr_ai_schema.py` | `CPRAgentDecision`, `validate_position_state` (strict pydantic) |
| `cpr_ai_prompt.py` | Versioned system prompt (`CPR_AI_PROMPT_VERSION`) |
| `cpr_ai_codex_runner.py` | Thread config, `safe_subprocess_environment` |
| `cpr_ai_codex_subprocess.py` | The child process boundary |
| `cpr_ai_decision_log.py` | JSONL decision log |
| `cpr_ai_runner.py` | Standalone smoke runner (`--synthetic --fake` / `--authenticated`) |

---

## 3. The frozen-context boundary

This is the component's defining design choice.

```
 completed 5-min bar
        │
        ▼
 freeze_cpr_context(...)      ← ONE snapshot, taken once, immutable
        │
        ▼
 FrozenCPRContextRegistry
        │
        ├── session_levels()      ┐
        ├── momentum_vwap()       │  FOUR no-argument MCP tools
        ├── market_structure()    │  read-only, no parameters
        └── position_state()      ┘
        │
        ▼
     Codex  ──►  CPRAgentDecision (strict pydantic)
```

Why **no-argument** tools:

- The model cannot ask about a different instrument, strike, or timeframe than
  the one the host froze. There is no parameter to smuggle a request through.
- Every tool answers from the *same* snapshot, so the model cannot see the
  market move mid-reasoning and produce a decision based on two different states.
- The tool surface is a fixed set (`EXPECTED_TOOL_NAMES`), verified by tests, so
  a new capability cannot appear without a code change and a review.

---

## 4. Division of labour

| Codex decides | The host decides |
|---|---|
| Regime | Entry geometry |
| Setup validity | Stop distance |
| Premise exits | Level validation |
| | Sizing (`CPR_AI_LOTS`, `CPR_AI_MAX_LOSS`, `CPR_AI_SIZE_MULTIPLIER`) |
| | Time cutoffs (start 09:30, entry cutoff 15:00, square-off 15:15) |
| | Lifecycle state (`CPRAITradeState`) |
| | All execution |

Indicator settings are fixed and documented: RSI 14 / Stochastic 14 / K 3 / D 3,
zones 20 and 80; one equal-size add; 30-NIFTY-point and 2-NIFTY-point geometry
constants; 0.40 threshold. These live in `env.example` and are asserted by
`Tests/Dependencies/test_repository_policy.py` so the operator-facing
explanation cannot drift from the code.

---

## 5. Process isolation

Codex runs in a **subprocess** with `safe_subprocess_environment` — a strict
allowlist, so trading and API secrets are not inherited by the child. The child
boundary is `cpr_ai_codex_subprocess.py`; the parent side is
`cpr_ai_codex_runner.py`.

`CPR_AI_SDK_TIMEOUT_SECONDS` (default 90) bounds the call. A timeout is a HOLD.

---

## 6. Safety posture

- Disabled by default; live-disabled by default. Real orders require **both**
  `LIVE_TRADING_ENABLED=true` and `CPR_AI_LIVE_TRADING=true`, plus the normal
  startup exposure audit and config validation.
- `_cpr_ai_startup_errors()` refuses to start a misconfigured agent.
- Decisions are strict-pydantic validated; a malformed decision is rejected, not
  coerced.
- `CPR_AI_DECISION_LOGGING_ENABLED` (default true) writes every decision to
  `Backtest Outputs/cpr_ai_decisions.jsonl` for after-the-fact review.
- Any SDK/agent failure is a HOLD; the mechanical risk loop is unaffected.

---

## 7. Configuration

Defaults live in `Dependencies/env.example` and are pinned by the policy test:

| Key | Default |
|---|---|
| `CPR_AI_ENABLED` | false |
| `CPR_AI_VIRTUAL_TRADING` | true |
| `CPR_AI_LIVE_TRADING` | false |
| `CPR_AI_MODEL` | `gpt-5.6-terra` |
| `CPR_AI_REASONING_EFFORT` | medium |
| `CPR_AI_SDK_TIMEOUT_SECONDS` | 90 |
| `CPR_AI_LOTS` / `CPR_AI_MAX_LOSS` / `CPR_AI_SIZE_MULTIPLIER` | 1 / 5500 / 1 |
| `CPR_AI_POLL_SECONDS` | 5 |
| `CPR_AI_TRADING_START_HOUR` / `_MINUTE` | 09:30 |
| `CPR_AI_ENTRY_CUTOFF_HOUR` / `_MINUTE` | 15:00 |
| `CPR_AI_SQUARE_OFF_HOUR` / `_MINUTE` | 15:15 |
| `CPR_AI_DECISION_LOGGING_ENABLED` / `_LOG_PATH` | true / `Backtest Outputs/cpr_ai_decisions.jsonl` |

Install the exact optional set from `requirements-codex-ai.txt`. Both AI agents
run inside the same process, so they must agree on one MCP package version —
`mcp==1.29.0` appears in both `requirements-ai.txt` and `requirements-codex-ai.txt`,
and the policy test asserts it.

---

## 8. Verification without spending money

Two zero-order smoke commands:

```bash
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake
```

```bash
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --authenticated
```

`--fake` makes **no billed/model/broker call** at all. CI runs only the
unauthenticated path — the authenticated smoke is an operator action.

---

## 9. Testing

`Tests/Signal Generators/CPR AI Agent/` — context freezing, tool registry,
schema validation, runtime/subprocess behaviour, and master integration. Its
`conftest.py` puts the **source** agent folder on `sys.path`, and deliberately
only that folder: adding a repository-wide path would let tests pass through
imports production never uses and could hide a missing dependency or an
accidental legacy-CPR coupling.

`Tests/Dependencies/test_repository_policy.py` additionally asserts that every
`cpr_ai_*.py` module is inside mypy's scope, so a new module cannot silently
escape type checking.

---

## 10. Contrast with SL Hunting

| | CPR Codex AI | SL Hunting |
|---|---|---|
| Provider | Codex (subprocess + MCP) | Claude (`claude-agent-sdk`, in-process) |
| Timeframe | completed 5-min bars | completed 1-min bars |
| Tool surface | four frozen no-argument MCP tools | prompt context + order tool |
| Instruments | NIFTY only | NIFTY + mechanical BankNIFTY mirror |
| Learning loop | none — decision log only | journal → coach → human-gated `lessons.json` |
| Shared | opt-in, off by default, host-owned gates, fail-soft to HOLD, same double gate | ← identical |
