# ADR-0007: LLM agents as opt-in workers with host-owned deterministic gates

**Status:** Accepted
**Date:** 2026-08-10 (retrospective)
**Deciders:** repository owner

## Context

Two strategies are worth trading but resist being written as rules: the
discretionary "SL Hunting" price-action method, and a five-minute SRSI/VWAP
strategy whose regime read is a judgement call. Both were candidates for an LLM.

The obvious risks:

- A model is **non-deterministic** — the same bar can produce different answers.
- A model can be **confidently wrong**, and will explain itself persuasively.
- A model can **reason its way around an instruction** in its own prompt.
- An SDK can time out, error, or return something unparseable.
- Model calls cost money and time.

None of that is acceptable near a live order path — unless the model is
prevented from touching the parts where being wrong is expensive.

## Decision

Run each agent as an ordinary worker thread, **off by default**, with a strict
division of labour:

| The model decides | The host decides |
|---|---|
| Is this a real setup? | Position size |
| Is the premise still alive? | Stop and target |
| Direction (or HOLD) | Daily max loss |
| Regime (CPR AI) | Entry cutoff and square-off |
| | Post-exit cooldown |
| | Stale-data liquidation |
| | Paper vs live (the standard double gate) |
| | All execution |

Plus four invariants:

1. **Fail-soft.** Any agent or SDK error becomes a **HOLD**, and the separate
   mechanical risk loop keeps checking stop, target, max-loss, stale data and
   square-off regardless of whether the model ever answers.
2. **Same execution door.** Agents act through the same tested
   `enter_position` / `exit_position` path as every other worker, so broker
   selection, paper/live routing, max-loss, square-off and Telegram behave
   identically.
3. **Lazy imports.** A missing optional dependency disables the strategy; it
   never breaks the run.
4. **Schema-validated decisions.** A malformed decision is rejected, not coerced.

The two agents differ in isolation mechanism: SL Hunting runs the
`claude-agent-sdk` in-process; CPR AI runs Codex in a **subprocess** with a
strict environment allowlist and exposes context through **four frozen
no-argument MCP tools**, so the model cannot request a different instrument,
strike or timeframe than the one the host froze.

## Options considered

### Option A: Opt-in worker, host owns every mechanical gate (chosen)

**Pros:** the model can be wrong without being dangerous; it composes with the
existing safety machinery instead of bypassing it; disabling it is one flag;
non-determinism is confined to entry/exit *judgement*, never to size or risk.
**Cons:** two extra runtimes and subscriptions; prompt/knowledge maintenance;
decisions are not reproducible, which complicates post-hoc analysis (mitigated
by the journal and the decision log).

### Option B: LLM proposes, human approves each trade

**Pros:** maximum safety.
**Cons:** defeats the purpose — the method trades on 1-minute bars. A human in
the loop per bar is not a system.

### Option C: LLM as an advisory overlay on existing strategies

**Pros:** no new execution path.
**Cons:** an "advisor" that can veto or amplify existing strategies couples
everything to it, and a bad call would degrade strategies that were working.
Independent workers with independent P&L make the agent's contribution
measurable in isolation.

### Option D: Give the model the risk controls too

**Pros:** simpler code; the model could size to its own conviction.
**Cons:** disqualifying. Sizing and stops are exactly where a confident wrong
answer is most expensive, and where deterministic code is strictly better.

## Trade-off analysis

The design rests on one observation: the model's *comparative advantage* is
pattern judgement on a chart, and its *comparative disadvantage* is arithmetic
discipline under pressure. Deterministic code is the reverse. So the split
follows the strengths rather than the convenience of the API.

The live journal supplied the strongest evidence for keeping mechanical rules in
code: SL Hunting's prompt already contained a judgement-based post-exit re-entry
gate, and the agent **talked past it twice** (23 and 27 Jul 2026) by relabelling
the same price structure as a fresh setup. The rule now lives in the host, where
it cannot be reasoned away. That is the general principle — *a rule the model
must not be able to argue with does not belong in the prompt.*

## Consequences

**Easier:** trading a discretionary method without hand-coding it; disabling an
agent instantly; comparing an agent's P&L against deterministic strategies.

**Harder:** reproducing a decision; testing (the SDK is never called in tests —
decisions are injected); cost and latency management (mitigated: SL Hunting
skips the call entirely when flat past its 10:30 cutoff).

**To revisit when:** a rule keeps needing enforcement in code after being stated
in the prompt — that is a signal the boundary should move further toward the
host, not that the prompt needs rewording.

## Action items

- [x] Both agents default off; both live-disabled by default; both use the
      standard double gate ([ADR-0004](0004-paper-by-default-double-gate.md)).
- [x] Fail-soft to HOLD; the mechanical risk loop is independent.
- [x] CPR AI: subprocess isolation, `safe_subprocess_environment` allowlist,
      four frozen no-argument MCP tools, strict pydantic decisions.
- [x] SL Hunting: post-exit cooldown enforced in code, not the prompt; the
      BankNIFTY mirror is mechanical, not a model choice.
- [x] Learning (SL Hunting) is **human-gated**: journal → off-loop coach →
      operator approval → `lessons.json`, injected only when
      `SL_HUNTING_LESSONS_ENABLED`. Automating promotion would let the agent
      rewrite its own instructions from its own losses.
