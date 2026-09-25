# ADR-0018: CPR Algo 4 is its own worker, not a fourth algo inside the CPR worker

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** repository owner

## Context

The operator's "Intraday SRSI VWAP" document sets out a CPR day-type playbook:
1. The 09:25 five-minute close against [MIN(S1, PDL), MAX(R1, PDH)] decides SIDEWAYS or TRENDING.
2. SIDEWAYS days trade Stochastic RSI reversals.
3. TRENDING days trade VWAP pullbacks that flip on swing structure.
4. Every trade obeys a 30-point stop cap, a 1:1 next-level gate and a 15:00 entry cutoff.

The optional CPR Codex AI Agent already trades these ideas, but a model makes its regime judgment.
The operator asked for a deterministic version, **CPR Algo 4**. They also asked whether it could live
inside the existing `CPRStrategyWorker`, which already runs Algo 1 + Algo 2 (+ RSI divergence), or
whether it must run on its own like CPR Algo 3.

## Decision

### 1. A separate `CPRAlgo4StrategyWorker`, integrated at the library level only

CPR Algo 4 is its own `AtmSingleLegStrategyWorker` subclass. It has its own position, its own
`CPR_ALGO4_*` prefix (and therefore its own `CPR_ALGO4_LIVE_TRADING` gate) and its own Sheet row.
It shares code with the CPR family where the maths is genuinely the same:
- It lives in `Signal Generators/CPR Strategy/`.
- Its frame comes from `cpr_strategy_logic.build_cpr_with_indicators`: CPR levels, PDH/PDL, R/S
  ladder, session VWAP, RSI and EMAs, exactly as CPR and CPR Algo 3 compute them.
- It adds only what that builder lacks: Stochastic RSI.

### 2. A broker-free engine that the worker and the backtest both drive

All decisions live in `CPRAlgo4Engine`: day type, trend direction, swing structure, the reversal
sequence, entries, premise exits, staged trailing and the R1 add. The engine is fed every completed
five-minute bar in order. `check_intrabar_exit` is a pure stop/target check. The live worker executes
the engine's answers, and `cpr_algo4_backtest.py` replays the same functions. A difference between
paper and backtest therefore cannot come from two implementations of one rule.

### 3. The R1 add and two-leg exit are a standalone copy of CPR AI's mechanics

CPR AI already has a hardened add leg: a separate role-A ledger leg, never retried after
PARTIAL/UNKNOWN, max-loss at the conservative risk quantity, and state kept until both legs are
broker-confirmed flat. Extracting it into a shared base class would have meant refactoring the CPR AI
worker. The operator chose to leave CPR AI untouched and copy the mechanics into Algo 4. Without an
add, Algo 4 exits through the shared single-leg path unchanged.

## Options considered

| Option | Verdict |
|---|---|
| Add "ALGO4" to `CPRSignalEngine._entry_candidates` and run it in `CPRStrategyWorker` | Rejected. See trade-off analysis. |
| Build Algo 4 on the CPR AI worker with a rule-based decider in place of Codex | Rejected. Algo 4 would inherit the optional AI dependencies (pydantic) and be unavailable on a core install, and it would couple a deterministic strategy to a 1,300-line agent worker. |
| Separate worker on the shared CPR frame builder (chosen) | Independent risk, gate and P&L; shared indicator maths. |
| Extract a shared base class for the add-leg mechanics | Deferred by the operator. It would remove the copy, but it would touch the CPR AI worker. |

## Trade-off analysis

Folding Algo 4 into `CPRStrategyWorker` looks like less code, but it breaks four things:
- **One position slot.** Algo 1/2 and Algo 4 would block each other. The engine's rule that holds
  when a long and a short candidate appear on the same candle would also suppress Algo 4 signals
  that disagree with Algo 1/2.
- **One live gate.** If CPR is live, Algo 4 would go live with it, with no paper phase of its own.
- **One Sheet row.** Algo 4's P&L could never be judged apart from Algo 1/2's.
- **One exit model.** `CPRSignalEngine._evaluate_exit` only knows "a five-minute high/low touched a
  static stop or target". Algo 4 needs:
  - Stochastic RSI and structure-flip premise exits;
  - staged trailing;
  - a per-poll spot stop;
  - an add leg;
  - day-level state: the regime, the trend direction and the reversal sequence.

  Encoding those would rewrite the engine the live CPR worker depends on.

The cost of the chosen option is a ~350-line copy of the add-leg and two-leg-exit mechanics. It is
contained by comments and by CLAUDE.md, which say a fix to one copy must be mirrored in the other.

## Consequences

- The core roster grows from approximately 27 to approximately 28 strategies (about 30 with both
  optional agents). The documentation drift guard now treats "27" as stale.
- CPR Algo 4 is paper by default. Live needs `LIVE_TRADING_ENABLED=true` **and**
  `CPR_ALGO4_LIVE_TRADING=true`. An unknown `CPR_ALGO4_EXIT_MODE` or an impossible entry cutoff runs
  paper on the defaults and is refused for live.
- CPR Algo 4 and CPR AI make a clean A/B pair: the same playbook, one judged by rules and one by a
  model, on independent ledgers.
- Because of the doc's own "do not enter if the next level is under 1:1" rule, TARGET mode
  effectively books at 1R. The next-level rule works as an entry filter. See
  [`../lld/cpr-algo4.md`](../lld/cpr-algo4.md).
- Manual operator step: add the Sheet rows `CPR Algo 4 Strategy`, `... [LIVE]` and `... [MIXED]`.
