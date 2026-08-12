# CPR Codex AI Agent

This optional worker is an independent five-minute SRSI/VWAP strategy. It does
not import or arbitrate CPR Algo 1, Algo 2, or Algo 3. Those strategies may run
at the same time as CPR AI, and all three keep independent positions and
independent P&L.

CPR AI separates judgment from authority. A fresh ephemeral Codex turn makes a
nondeterministic, dynamic regime/setup judgment and may declare an existing
premise invalid. The deterministic Python host owns the completed-bar cadence,
entry and risk gates, quantity, contract, lifecycle checks, audit, and every
broker submission. Missing, stale, contradictory, timed-out, or malformed model
evidence becomes `HOLD` rather than an executable action.

## Four frozen tools

Each turn must call all four no-argument MCP tools exactly once. They return
deep-copy views of the same frozen completed-bar context:

- `session_levels`: previous-session CPR, R1/R2/S1/S2, opening corridors, the
  current completed close, buffered next levels, and prior accepted regime.
- `momentum_vwap`: TradingView-style Stoch RSI (RSI 14, stochastic 14, K 3,
  D 3, zones 20/80), RSI14, EMA5/EMA20 order and slopes, candle facts, and
  deterministic VWAP sequence/body evidence.
- `market_structure`: confirmed swing highs/lows, HH/LH/HL/LL comparisons, and
  the host-computed long R1 add candidate.
- `position_state`: an allowlist of market-position facts needed to judge
  `HOLD`, premise `EXIT`, or the one permitted `SCALE_IN` request.

The snapshot contains no order surface, account, credential, broker, venue, or
execution object. The immediate turn request repeats the four exact tool names
in addition to the developer prompt. If the first isolated turn still omits a
tool, or one tool reports failure, the host permits one retry against the same
immutable snapshot using only the time left in the original SDK deadline. A
second incomplete result, a duplicate/unapproved tool, or any unexpected agent
action invalidates the turn and produces `HOLD`.

## Decision contract and host gates

Codex must return the strict `CPRAgentDecision` schema with exactly these fields:

- `action`: `HOLD`, `ENTER_LONG`, `ENTER_SHORT`, `EXIT`, or `SCALE_IN`
- `regime`: `SIDEWAYS`, `TRENDING`, or `UNDECIDED`
- `setup`: `NONE`, `SIDEWAYS_SRSI`, `TRENDING_VWAP_CONTINUATION`,
  `TRENDING_VWAP_REVERSAL`, `PREMISE_EXIT`, or `R1_SCALE_IN`
- `confidence`, `reasoning`, `model_used`, and `prompt_version`

Codex cannot supply entry, stop, target, trail, lots, quantity, symbol, expiry,
broker, venue, order, or any other execution field. Extra fields fail schema
validation. For a proposed entry, the host independently derives the completed
five-minute close, protective stop, risk, next buffered milestone, and buffered
R2/S2 final target. It rejects risk above 30 NIFTY points, reward below one R,
invalid geometry, or a missing deterministic setup gate.

The agent may dynamically label a completed bar `SIDEWAYS`, `TRENDING`, or
`UNDECIDED`; a convincing completed-bar break can change that judgment. The
host then enforces the selected framework:

- Sideways entries need a Stoch RSI K/D cross in the 20 or 80 zone and a
  confirmed swing for the protective stop.
- Trend continuation needs the directional three-bar VWAP sequence. Trend
  reversal needs a completed reclaim/loss of VWAP. Both require at least 0.40
  of the entry candle body on the trade side, directional RSI, EMA order and
  slopes, and the completed candle extreme as stop.
- Open positions can receive a model-requested premise exit, while mechanical
  hard stops, SRSI reversals, max loss, stale-data handling, and square-off do
  not wait for model permission.
- Trailing is staged: the host ratchets to breakeven at the first milestone;
  reversal trades add a second risk/CPR milestone before prior-close trailing.
  Buffered R2 for longs or S2 for shorts is the final booking level.
- Only a trending long may request one equal-size R1 add, after the host's
  bearish-touch/bullish-reclaim pattern. The cap is fixed at one and cannot be
  raised by configuration. Shorts and sideways-origin positions cannot add.

## Session cadence

- Before 09:30 IST the worker waits.
- It polls mechanical safety every five seconds and calls Codex at most once for
  each newly completed five-minute candle. A start-stamped one-minute candle is
  not complete until the next minute begins, and all five exact minute slots
  must exist once.
- In websocket mode, clock completeness alone is not enough. Every shared OHLC
  snapshot carries an atomic `official_candle_ts` watermark. A 09:55 five-minute
  bucket waits until the REST source covers its final 09:59 one-minute candle,
  even when Dhan's true-up takes longer than its normal five-second delay. This
  is a condition check rather than a hard-coded sleep. A later official revision
  can still invalidate an in-flight result, but it cannot create a second model
  call for an already-consumed bucket.
- At 15:00 IST new entries and adds stop; management and exits continue.
- At 15:15 IST the host square-off closes exposure and stops the worker.

## Live safety

Defaults are `CPR_AI_ENABLED=false`, `CPR_AI_VIRTUAL_TRADING=true`, and
`CPR_AI_LIVE_TRADING=false`. A real CPR AI order is possible only through the
standard double gate: both global `LIVE_TRADING_ENABLED=true` and strategy-level
`CPR_AI_LIVE_TRADING=true`, after normal startup exposure audit and configuration
validation. CPR, CPR Algo 3, and CPR AI do not need to be disabled for one
another; they may coexist as separate ledgers.

The one R1 add reuses the primary leg's exact option contract and initial filled
quantity. A live add has a separate execution-ledger leg. Once a live add is
submitted, an ambiguous, partial, unknown, or rejected response cannot create a
paper fill or retry the same add. Exit handling retains local state until both
the primary and add legs are broker-confirmed flat.

Every exposure-increasing action requires a successful pre-action audit and a
fresh post-inference recheck of lifecycle, market-data health, entry cutoff, and
square-off state. Risk-reducing exits remain available if decision logging fails.

## Isolated Codex runtime

Install the exact optional stack separately from broker dependencies:

```powershell
python -m pip install -r requirements-codex-ai.txt
```

Use the operator's existing subscription-backed Codex/ChatGPT authentication.
Do not put an OpenAI API key, broker credential, or trading credential in the
CPR subprocess. The parent passes only a frozen public snapshot into a temporary
directory. It copies the operator's `auth.json` once into a process-lifetime,
auth-only temporary `CODEX_HOME`, then reuses that isolated copy so a child token
refresh survives later serialized turns. The copy is never synchronized back or
symlinked; config, global MCP servers, plugins, skills, apps, and rules are not
copied. HOME, USERPROFILE, AppData, and temp paths point to a separate synthetic
profile under a strict allowlist. It runs read-only with deny-all approvals; shell, unified exec, web
search, collaboration, multi-agent actions, browser/computer use, plugins, apps,
and workspace writes are disabled. Its only enabled tools are the four local
read-only MCP tools above. Missing isolatable authentication fails before child
launch. Trading and API secrets remain excluded from the child environment.

## Operator P&L sheet rows

Add these exact labels to column A of the configured monthly result sheet:

- `CPR AI Agent Strategy` for PAPER results
- `CPR AI Agent Strategy [LIVE]` for LIVE results
- `CPR AI Agent Strategy [MIXED]` when one session contains both modes

The normal end-of-day updater writes the matching row automatically and skips a
missing row with a warning; the labels remain separate from legacy CPR workers.

## Decision audit

With `CPR_AI_DECISION_LOGGING_ENABLED=true`, the host appends sanitized JSONL to
`Backtest Outputs/cpr_ai_decisions.jsonl` by default. Each row records the frozen
context, proposal, accepted regime, validation code/reason, authoritative host
geometry, execution outcome, latency, inference-attempt count, aggregate token
usage across a retry, and final tool-call evidence.
Credential-like mapping fields are removed recursively before serialization.
Logging never makes a proposal executable, and an enabled log must succeed
before an entry or add may be submitted.

## Zero-order smoke commands

Automated verification uses fake mode. It makes no billed/model/broker call,
does not authenticate, and does not write an actual decision log:

```powershell
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --fake
```

The authenticated smoke exercises the isolated Codex/MCP path after the optional
stack is installed and Codex login already exists. It still has no broker or
order object, but it does make a real model call, so run it manually only:

```powershell
python "Signal Generators/CPR AI Agent/cpr_ai_runner.py" --synthetic --authenticated
```

Future discretionary prompt knowledge belongs in the modular
`operator_approved_knowledge`/`discretionary_context` extension seam. Keep every
addition advisory, operator-approved, and host-validated; never move levels,
sizing, risk, execution, or exit authority into the model or an MCP tool.
