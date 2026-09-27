# LLD — CPR Codex AI Agent (optional, Codex)

**Owns:** `Signal Generators/CPR AI Agent/` · `CPRAIWorker`, `CPRAITradeState` (master file)
**Status:** optional, **disabled by default** (`CPR_AI_ENABLED=false`) and
**live-disabled by default** (`CPR_AI_LIVE_TRADING=false`)
**Related ADRs:** [0007 — LLM agents as opt-in workers](../adr/0007-llm-agents-as-opt-in-workers.md),
[0019 — the Trend-Day Rider](../adr/0019-cpr-ai-trend-day-rider.md)
**Operator detail:** the folder's own [`README.md`](../../Signal%20Generators/CPR%20AI%20Agent/README.md)

---

## 1. Responsibility

A five-minute **Trend-Day Rider**. A deterministic host gate flags trend-day candidates. Codex may
accept or veto each candidate and may make premise exits on an open position. The host owns every
price, gate and order. Every entry sells the opposite ATM option on the current weekly expiry, with a
fixed VWAP spot stop.

The strategy was chosen from a five-year study on real weekly option premiums; the evidence,
including the ideas that failed, is in ADR-0019. It is *not* an arbiter over the other CPR strategies.
Ordinary CPR, CPR Algo 3, CPR Algo 4, Regime Adaptive and CPR AI are **independent strategies that may
run together with independent positions and independent P&L**. CPR Algo 4 now trades the SRSI/VWAP
playbook CPR AI used to trade — see [`cpr-algo4.md`](cpr-algo4.md).

---

## 2. Modules

| File | Role |
|---|---|
| `cpr_ai_trend_day.py` | **The deterministic gate** — `TrendDayConfig`, `session_atr`, `session_vwap`, `evaluate_trend_day_candidate`. Pandas/numpy only; shared with the backtest |
| `cpr_ai_agent.py` | `CPRAgent`, `CPRHostPolicy`, `CPRAgentRunResult`, `CPRToolCallRecord` |
| `cpr_ai_context.py` | Builds the frozen per-bar context (levels, gap, ATR5, VWAP facts, swings, candidate) |
| `cpr_ai_signals.py` | `freeze_cpr_context` — the snapshot boundary |
| `cpr_ai_tools.py` | `FrozenCPRContextRegistry`, `EXPECTED_TOOL_NAMES` |
| `cpr_ai_mcp_server.py` | Isolated MCP server exposing the four tools |
| `cpr_ai_schema.py` | `CPRAgentDecision`, `validate_position_state` (strict pydantic) |
| `cpr_ai_prompt.py` | Versioned system prompt (`CPR_AI_PROMPT_VERSION = "cpr-trend-day-rider-v1"`) |
| `cpr_ai_codex_runner.py` | Thread config, `safe_subprocess_environment` |
| `cpr_ai_codex_subprocess.py` | The child process boundary |
| `cpr_ai_decision_log.py` | JSONL decision log |
| `cpr_ai_runner.py` | Standalone smoke runner (`--synthetic --fake` / `--authenticated`) |

The backtest is `My Backtest Files (For Reference)/cpr_ai_trend_day_backtest.py`
(`python algo.py backtest --strategy cpr-ai-trend-day`).

---

## 3. The candidate gate

On the newest completed five-minute bar, `evaluate_trend_day_candidate` checks, in this order, and
names the first failure as `reason`:

| Gate | Rule | Rejection reason |
|---|---|---|
| Window | bar **start** 11:00–13:30 IST, inclusive | `outside_window` |
| ATR | mean high−low of up to the 5 prior sessions; fewer than 3 → none | `atr_unavailable` |
| Expansion | session high−low **>** 1.0 × ATR5 | `range_not_expanded` |
| Trend extreme | close location ≥ 0.85 **and** close > VWAP (LONG), or ≤ 0.15 **and** close < VWAP (SHORT) | `not_at_trend_extreme` |
| Confluence | LONG needs ≥ 2 of {close > R1, open > prior close, \|close − VWAP\| > 0.35 × ATR5}; SHORT needs 0 | `confluence_too_low` |

Entry is the bar close; the stop is that bar's session VWAP (equal-weight typical price, because
index candles carry no volume). Every measurable fact is filled in even for a rejected bar, so the
model sees the context and the decision log keeps it.

The ATR uses five sessions because the live store holds about seven calendar days
(`INTRADAY_LOOKBACK_DAYS=7`). The backtest found the same edge with a 3-, 5- or 14-session ATR.

---

## 4. The frozen-context boundary

```
 completed 5-min bar
        │
        ▼
 freeze_cpr_context(...)      ← ONE snapshot, taken once, immutable
        │
        ▼
 FrozenCPRContextRegistry
        │
        ├── session_levels()      ┐  CPR levels, prior day, gap, ATR5, opening corridors
        ├── momentum_vwap()       │  VWAP value/distance/side fractions, newest candle, recent candles
        ├── market_structure()    │  swings, HH/HL, extreme recency, trend_day_candidate
        └── position_state()      ┘  allowlisted premise/risk facts, entries_today
        │
        ▼
     Codex  ──►  CPRAgentDecision (strict pydantic)
```

Why **no-argument** tools:

- The model cannot ask about a different instrument, strike, or timeframe than the one the host
  froze. There is no parameter to smuggle a request through.
- Every tool answers from the *same* snapshot, so the model cannot see the market move mid-reasoning
  and produce a decision based on two different states.
- The tool surface is a fixed set (`EXPECTED_TOOL_NAMES`), verified by tests, so a new capability
  cannot appear without a code change and a review.

---

## 5. Division of labour

| Codex decides | The host decides |
|---|---|
| Accept or veto a candidate | Whether a candidate exists at all (§3) |
| Premise exits on an open position | Entry price (bar close) and stop (bar VWAP) |
| Advisory regime label | Contract expression: SELL the opposite ATM, current weekly expiry |
| | Sizing (`CPR_AI_LOTS`, `CPR_AI_MAX_LOSS`, `CPR_AI_SIZE_MULTIPLIER`) |
| | One entry per session; time cutoffs (start 09:30, entry cutoff 15:00, square-off 15:15) |
| | The VWAP spot stop, checked every poll |
| | Lifecycle state (`CPRAITradeState`) and all execution |

The decision contract (`CPRAgentDecision`) allows actions `HOLD`, `ENTER_LONG`, `ENTER_SHORT`,
`EXIT`; setups `NONE`, `TREND_DAY_CONTINUATION`, `PREMISE_EXIT`; regimes `TRENDING`, `SIDEWAYS`
("not a trend day") and `UNDECIDED`. Entries require `TRENDING`. Extra fields are rejected, so a model
cannot supply a price, size, strike or expiry.

`CPRHostPolicy` accepts an entry only when:
- the frozen candidate is eligible;
- the candidate points the same way as the proposal;
- the candidate's entry equals the frozen completed close;
- the stop is on the protective side;
- `position_state.entries_today` is exactly 0.

Otherwise the result is a typed HOLD, for example `candidate_direction_mismatch` or `session_entry_used`.

---

## 6. Worker lifecycle

1. **Every poll** (`CPR_AI_POLL_SECONDS`): shutdown, max-loss, the 15:15 square-off, feed health, then
   the VWAP spot stop. None of this waits for a model turn.
2. **Each completed bar**, once all five official source minutes are present (websocket mode):
   - **Flat, entry already used today** → nothing.
   - **Flat, no eligible candidate** → nothing. No Codex call is made, which is most bars of most days.
   - **Flat with a candidate** → one Codex turn, then the audit hook. If accepted, a fresh recheck of
     stop/lifecycle/feed/time runs, plus `stop_already_breached` if the fresh spot has crossed the
     VWAP stop during the turn. Then `enter_position(..., option_opening_side="SELL",
     option_contract_direction=<opposite>, use_current_expiry=True)`.
   - **Open** → one Codex turn for HOLD or `PREMISE_EXIT`, then the normal safety pass again.
3. A submitted entry, or possible live exposure (`LIVE_INDETERMINATE`), uses up the session's entry.
   A clean refusal (spread gate, contract lookup) leaves it available for a later bar.
4. Exits use the base single-leg `exit_position`: a sold leg closes with a BUY, and a live close that
   is not broker-confirmed flat keeps the position open for reconciliation.

---

## 7. Process isolation

Codex runs in a **subprocess** with `safe_subprocess_environment` — a strict allowlist, so trading
and API secrets are not inherited by the child. The child boundary is `cpr_ai_codex_subprocess.py`;
the parent side is `cpr_ai_codex_runner.py`.

`CPR_AI_SDK_TIMEOUT_SECONDS` (default 90) bounds the call. A timeout is a HOLD.

---

## 8. Safety posture

- Disabled by default; live-disabled by default. Real orders require **both**
  `LIVE_TRADING_ENABLED=true` and `CPR_AI_LIVE_TRADING=true`, plus the normal startup exposure
  audit and config validation. The same double gate authorizes the sold options; there is no third
  short-premium switch.
- `_cpr_ai_startup_errors()` refuses to start a misconfigured agent.
- Decisions are strict-pydantic validated; a malformed decision is rejected, not coerced.
- `CPR_AI_DECISION_LOGGING_ENABLED` (default true) writes every decision, with its full frozen context
  (candidate included), to `Backtest Outputs/cpr_ai_decisions.jsonl`. That makes vetoes auditable
  against the deterministic baseline.
- Any SDK/agent failure is a HOLD; the mechanical risk loop is unaffected.
- A spot stop triggers an exit but cannot guarantee a fill: a gap through the stop on a sold
  option can lose more than planned.

---

## 9. Configuration

Defaults live in `Dependencies/env.example` and are pinned by the policy test. The strategy's
thresholds (window, ATR multiple, location, confluence) are code-owned in `TrendDayConfig`.

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

`CPR_AI_MAX_LOSS`: at a 75-unit lot, ₹5,500 is about 73 premium points, and in the backtest it cut
PF from 1.69 to 1.52 by closing sold legs that later recovered. ₹10,000 kept PF at 1.63. The default
is unchanged; see ADR-0019.

Install the exact optional set from `requirements-ai.txt`, which carries BOTH AI agents. They run
inside the same process, so Python can only ever install one version of what they share —
`mcp` and `pydantic` (pinned in `requirements-ai.txt`). That constraint is why the two files were merged (PR #125):
keeping them apart meant pinning the shared packages twice and keeping them equal by hand.

---

## 10. Verification without spending money

Two zero-order smoke commands:

```bash
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake
```

```bash
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --authenticated
```

`--fake` makes **no billed/model/broker call** at all. CI runs only the unauthenticated path — the
authenticated smoke is an operator action.

The deterministic baseline:

```bash
python algo.py backtest --strategy cpr-ai-trend-day
```

It needs `Backtest Outputs/nifty_renko_futures_5y_1min_data.csv`. With
`Backtest Outputs/expired_options/nifty` (from `python algo.py fetch-expired-options`) it prices the
sold leg on real premiums; without it, it reports spot points.

---

## 11. Testing

`Tests/Signal Generators/CPR AI Agent/` — the candidate gate (`test_cpr_ai_trend_day.py`), context
freezing, tool registry, schema validation, host policy, runtime/subprocess behaviour, and master
integration. A context test proves the frozen candidate equals what the backtest's gate computes on
the same bars. Its `conftest.py` puts the **source** agent folder on `sys.path`, and deliberately only
that folder: adding a repository-wide path would let tests pass through imports production never uses
and could hide a missing dependency or an accidental legacy-CPR coupling.

`TestCPRAIWorkerFoundation` in the master suite covers the worker:
- the flat no-candidate skip and the one-entry-per-session rule;
- the sold expression and `stop_already_breached`;
- audit provenance and the post-inference rechecks;
- a rejected live exit keeping the position open.

`Tests/Dependencies/test_repository_policy.py` additionally asserts that every `cpr_ai_*.py` module
is inside mypy's scope, so a new module cannot silently escape type checking.

---

## 12. Contrast with SL Hunting

| | CPR Codex AI | SL Hunting |
|---|---|---|
| Provider | Codex (subprocess + MCP) | Claude (`claude-agent-sdk`, in-process) |
| Timeframe | completed 5-min bars | completed 1-min bars |
| Entry authority | host gate; Codex may only veto | the agent decides |
| Tool surface | four frozen no-argument MCP tools | prompt context + order tool |
| Instruments | NIFTY only, sold ATM options | NIFTY + mechanical BankNIFTY mirror |
| Learning loop | none — decision log only | journal → coach → human-gated `lessons.json` |
| Shared | opt-in, off by default, host-owned gates, fail-soft to HOLD, same double gate | ← identical |
