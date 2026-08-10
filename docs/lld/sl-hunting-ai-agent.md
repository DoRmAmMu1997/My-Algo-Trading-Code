# LLD — SL Hunting AI Agent (optional, Claude)

**Owns:** `Signal Generators/SL Hunting AI Agent/` · `SLHuntingAIWorker` (master file)
**Status:** optional, **off by default** (`SL_HUNTING_ENABLED`), paper unless
explicitly enabled
**Related ADR:** [0007 — LLM agents as opt-in workers](../adr/0007-llm-agents-as-opt-in-workers.md)
**Operator detail:** the folder's own [`README.md`](../../Signal%20Generators/SL%20Hunting%20AI%20Agent/README.md)

---

## 1. Responsibility

Trade the discretionary "SL Hunting" price-action method on NIFTY ATM options by
asking a Claude agent (via `claude-agent-sdk`, on a Claude subscription — **no
API key**) for one decision per completed 1-minute bar.

The design question this component answers is: *how do you let a language model
trade real money without letting it be dangerous?* The answer is the division of
labour in §3 — the model judges **premise**, the host owns **everything
mechanical**.

---

## 2. Modules

| File | Role |
|---|---|
| `sl_hunting_agent.py` | The agent itself: prompt assembly, SDK call, `SLHuntingDecision` |
| `sl_hunting_knowledge.py` | The curated method knowledge injected into the prompt |
| `sl_hunting_tools.py` | The frozen tool surface the agent may call |
| `sl_hunting_executor.py` | `MasterWorkerExecutor` / `StandaloneExecutor` — routes agent actions into the host's `enter_position` / `exit_position` |
| `sl_hunting_indicators.py` | Indicator helpers incl. `cross_index_signal` (BankNIFTY confirmation) |
| `sl_hunting_journal.py` | Per-trade journal — the input to learning |
| `sl_hunting_coach.py` | Off-loop reflection coach that proposes lessons |
| `sl_hunting_lessons.py` | `lessons.json` load/validate; human-gated promotion |
| `sl_hunting_premarket.py` | Pre-open note (`premarket_note.json`) |
| `sl_hunting_ai_validation.py` | Schema/decision validation |
| `sl_hunting_runner.py` | Standalone runner for offline replay |

Dependencies are **lazily imported**: a missing `claude-agent-sdk` simply
disables the strategy (`SLHuntingAIWorker` stays `None` and `main()` skips it).
It never breaks the run.

---

## 3. Division of labour

```
 ┌──────────────────────── the model decides ────────────────────────┐
 │  Is there a real level here?                                      │
 │  Is this a genuine stop-hunt setup or noise?                      │
 │  Does BankNIFTY confirm the NIFTY read?                           │
 │  Is the premise for this position still alive?  (per leg)         │
 │  Direction: LONG / SHORT / HOLD                                   │
 └───────────────────────────────────────────────────────────────────┘
 ┌──────────────────────── the HOST decides ─────────────────────────┐
 │  Position size          risk_sizing.SizingDecision, floored lots,  │
 │                         never over SL_HUNTING_RISK_BUDGET,         │
 │                         capped at SL_HUNTING_MAX_LOTS (default 5)  │
 │  Stop / target          mechanical, checked every loop             │
 │  Daily max loss         scaled kill-switch                         │
 │  Entry cutoff           10:30 by default — no NEW positions        │
 │  Square-off             15:15 — closes both legs                   │
 │  Post-exit cooldown     time-based, enforced in code               │
 │  Stale-data liquidation shared health gates                        │
 │  Paper vs live          the standard double gate                   │
 │  The BankNIFTY mirror   mechanical; the agent does not choose it   │
 └───────────────────────────────────────────────────────────────────┘
```

The agent **does not choose lots**. It supplies an underlying stop distance; the
host floors affordable whole lots from it.

---

## 4. The decision loop

```
every completed 1-min bar:
   if flat and past the 10:30 cutoff:  skip the LLM call entirely   ← saves tokens
   build context: NIFTY 1-min frame + BankNIFTY (fetched per bar,
                  like CPR Algo 3) + open-position state
                  + knowledge + optionally lessons.json
   ask the agent  ──► SLHuntingDecision (schema-validated)
   │
   ├─ ENTER (NIFTY only)  ──► host gates ──► enter_position(NIFTY ATM)
   │                                     └─► mechanical mirror: equal-lot
   │                                         BankNIFTY ATM leg
   ├─ EXIT  ──► exit_leg selector: NIFTY | BNF | BOTH
   └─ HOLD  ──► nothing
   │
   └─ ANY exception ──► safe HOLD
                        (the mechanical risk loop keeps running regardless)
```

**Fail-soft is the whole safety posture for the model half.** Any agent or SDK
error becomes a HOLD. The separate mechanical loop keeps checking stop, target,
max-loss, stale data and square-off whether or not the model ever answers.

---

## 5. The BankNIFTY mirror

`SL_HUNTING_BNF_MIRROR` (default **true**). Every NIFTY entry is mirrored with an
equal-lot BankNIFTY ATM leg.

| Aspect | Behaviour |
|---|---|
| Entry | **NIFTY only.** The mirror copies it; the agent never enters BankNIFTY directly. |
| Hard risk | **Tied.** Stop, target, max-loss and the 15:15 square-off close both legs. |
| Premise | **Independent.** The agent evaluates each leg separately and can cut one alone via the EXIT `exit_leg` selector (`NIFTY` \| `BNF` \| `BOTH`). |
| Expiry | Always the **nearest monthly** — never rolls forward (see [`execution-and-brokers.md`](execution-and-brokers.md) §6.1). |
| Expiry week | Switches to a deep-ITM strike (`..._NEAR_EXPIRY_ITM_STEPS`) once fewer than `..._ROLLOVER_DAYS` remain. |
| Failure | Fail-soft: any mirror problem skips the mirror, never the NIFTY leg. |

> ⚠️ **The mirror roughly DOUBLES the basket's rupee risk beyond
> `SL_HUNTING_RISK_BUDGET`.** This is operator-accepted, and the daily max-loss
> kill-switch still caps the day. With `SL_HUNTING_SIZE_MULTIPLIER=M`, the basket
> sits near **2 × M** times the single-leg budget. Anyone changing sizing here
> must account for that.

---

## 6. Post-exit cooldown

`SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES` blocks a new entry after a target, stop,
or premise-invalidating exit. Three properties, each of which was a bug first:

1. The timer starts only from the moment **the whole NIFTY/BankNIFTY basket is
   confirmed flat**. A lone or partly closed leg does not run it down.
2. **Exits never consult it.** A cooldown must never prevent closing a position.
3. **Corrupt guard state rejects new LIVE entries** rather than defaulting to
   "allowed".

It exists in code because the prompt's judgement-based version was talked past
twice in the live journal (23 and 27 Jul 2026) by relabelling the same price
structure as a fresh setup. Rules the model must not reason its way around
belong in the host.

---

## 7. Knowledge and learning

**Knowledge** (`sl_hunting_knowledge.py`) is curated, versioned method content
injected into the prompt — including a `BNF_SPECIFIC` section (triple-index
BNF + NIFTY + Sensex read, BankNIFTY as the "major index", expiry-day priority,
round-number magnets). That section is **advisory context for the cross-index
read only — execution stays NIFTY-only.** Provenance is recorded in
`sl_hunting_doc.md`.

**Learning** is a deliberately slow, human-gated loop:

```
 live trade ──► sl_hunting_journal.py (per-trade record)
                       │  OFF the trading loop
                       ▼
                sl_hunting_coach.py   tool-free, schema-validated reflection
                       │              proposes candidate lessons
                       ▼
                OPERATOR REVIEW       digest-bound approval
                       │
                       ▼
                lessons.json ──► injected into the prompt ONLY when
                                 SL_HUNTING_LESSONS_ENABLED (default false)
```

No lesson reaches the prompt without a human promoting it. The coach runs off
the trading loop so reflection can never delay a decision.

---

## 8. Configuration

| Key | Default | Meaning |
|---|---|---|
| `SL_HUNTING_ENABLED` | false | Master switch for the whole agent |
| `SL_HUNTING_LIVE_TRADING` | false | Second half of the live double gate |
| `SL_HUNTING_RISK_BUDGET` | 2500 | Rupee risk for the **NIFTY leg** (mirror doubles the basket) |
| `SL_HUNTING_MAX_LOTS` | 5 | Hard cap |
| `SL_HUNTING_SIZE_MULTIPLIER` | 1 | Scales budget, lots, max-loss together |
| `SL_HUNTING_BNF_MIRROR` | true | Mechanical BankNIFTY mirror |
| `SL_HUNTING_BNF_MIRROR_ROLLOVER_DAYS` | 7 | Days-to-expiry at which the mirror switches to ITM strikes (the name is legacy — it no longer rolls) |
| `SL_HUNTING_BNF_MIRROR_NEAR_EXPIRY_ITM_STEPS` | 4 | How deep ITM in that window |
| `SL_HUNTING_NO_NEW_ENTRY_HOUR` / `_MINUTE` | 10:30 | Entry cutoff, **not** a square-off |
| `SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES` | — | Re-entry block after a flat basket |
| `SL_HUNTING_LESSONS_ENABLED` | false | Inject `lessons.json` into the prompt |

Setup: `pip install -r requirements-ai.txt`, then a one-time `claude setup-token`.
Keep `ANTHROPIC_API_KEY` **unset** so it bills the Claude plan rather than
per-token API usage.

---

## 9. Testing

`Tests/Signal Generators/SL Hunting AI Agent/` — agent, indicators, journal,
lessons, premarket, runner, schema, and the v2 behavioural suite. Its
`conftest.py` puts the **source** agent folder on `sys.path` (the folder name has
spaces and the modules import each other by bare name).

The SDK is never called in tests; decisions are injected. Master-side wiring
(mirror, cooldown, cutoff, executor routing) is covered in
`Tests/test_nifty_multi_strategy_master.py`.

---

## 10. Known limitations

- **Non-deterministic.** The same bar can produce different decisions. The
  mechanical half is what makes that acceptable; do not move a rule from the host
  to the prompt.
- **Cost and latency** are bounded by skipping the call when flat past the cutoff.
- **Basket risk is ~2× the named budget** (§5). This surprises people; it is
  documented in four places for that reason.
- **Learning is human-gated on purpose.** Automating lesson promotion would let
  the agent rewrite its own instructions from its own losses.
