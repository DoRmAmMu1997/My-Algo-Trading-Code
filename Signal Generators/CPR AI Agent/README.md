# CPR Codex AI Agent — the Trend-Day Rider

This optional worker trades one five-minute strategy, the **Trend-Day Rider**. It does not import or
arbitrate CPR Algo 1, 2, 3 or 4. Those strategies may run at the same time as CPR AI, and each keeps
its own position and P&L.

CPR AI separates judgment from authority:
- **The host proposes.** A deterministic gate (`cpr_ai_trend_day.py`) decides whether the newest
  completed bar is a trend-day candidate.
- **Codex disposes.** A fresh ephemeral Codex turn may accept or veto that candidate, and may declare
  an open position's premise broken.
- **The host executes.** The Python host owns the cadence, every price and gate, quantity, contract,
  lifecycle, audit, and every broker submission.

Missing, stale, contradictory, timed-out or malformed model evidence becomes `HOLD`, never an
executable action.

## The strategy

**Thesis.** By late morning, a NIFTY session that has already out-ranged its recent average and is
pinned at one extreme, on the trend side of VWAP, is a trend day, and trend days tend to close near
their extremes. The strategy sells the ATM option on the side the market is leaving, so time decay
works with the drift.

**Candidate** (all must hold on the newest completed five-minute bar):
- the bar starts between 11:00 and 13:30 IST;
- today's high−low is greater than 1.0 × ATR5, the mean high−low of up to the five prior sessions
  (fewer than three sessions: no candidate);
- the close is in the top 15% of today's range and above session VWAP (bullish), or in the bottom
  15% and below VWAP (bearish);
- **bullish only:** at least two of: close above R1, a gap up from the prior close, close more than
  0.35 × ATR5 from VWAP. Bearish candidates need none.

**Trade:**
- bullish → **sell the ATM PE**; bearish → **sell the ATM CE**, both on the current weekly expiry;
- stop: that bar's VWAP on NIFTY spot, fixed, checked every poll;
- no target, no trailing, no add: hold until the stop, 15:15, or a premise exit;
- one entry per session: no re-entry and no flip after an exit.

**Evidence** (5 years, real weekly option premiums, 2-point costs, entry two minutes after the bar):
264 trades, +9.6 premium points per trade, profit factor 1.69, every year positive (2025 and 2026 the
weakest). Buying the directional option instead managed PF 1.27. ADR-0019 lists the ideas that
did **not** work, including OI walls, narrow-CPR trend days and BankNIFTY confirmation. Reproduce
the numbers with:

```powershell
python algo.py backtest --strategy cpr-ai-trend-day
```

## Four frozen tools

Each turn must call all four no-argument MCP tools exactly once. They return deep-copy views of the
same frozen completed-bar context:

- `session_levels`: previous-day H/L/C, CPR and R1/R2/S1/S2, the opening gap, ATR5 with the prior
  session ranges it used, opening corridors, the current close, and the prior accepted regime.
- `momentum_vwap`: session VWAP (its method, distance, distance in ATR, and the share of bars above
  and below it), the newest candle, and the last six candles with their VWAP.
- `market_structure`: confirmed swing highs/lows with HH/LH/HL/LL comparisons, bars since the
  session high and low, and the host's `trend_day_candidate` verdict with every measured fact and the
  reason it did or did not qualify.
- `position_state`: an allowlist of position facts (direction, original entry/risk/stop, premise,
  `entries_today`), never symbols, quantities or order details.

VWAP and candle facts use only the current session and reset every trading day. The snapshot
contains no order surface, account, credential, broker, venue or execution object. A repair turn is
considered only when a missing or failed required tool is the sole remaining defect, reuses the same
immutable snapshot, and gets only the time left in the original deadline. A second incomplete result,
a duplicate or unapproved tool, or any unexpected agent action produces `HOLD`.

## Decision contract and host gates

Codex must return the strict `CPRAgentDecision` schema with exactly these fields:

- `action`: `HOLD`, `ENTER_LONG`, `ENTER_SHORT`, or `EXIT`
- `regime`: `TRENDING`, `SIDEWAYS` (not a trend day), or `UNDECIDED`
- `setup`: `NONE`, `TREND_DAY_CONTINUATION`, or `PREMISE_EXIT`
- `confidence`, `reasoning`, `model_used`, and `prompt_version`

Codex cannot supply entry, stop, target, lots, quantity, symbol, expiry, broker, venue, order, or any
other execution field; extra fields fail validation. The host accepts an entry only if it names the
candidate's own direction, regime `TRENDING` and setup `TREND_DAY_CONTINUATION`, the candidate
describes the frozen close, and no entry has been taken this session. The entry price is that close
and the stop is the candidate's VWAP, both host-derived. An open position may only hold or exit.

## Session cadence

- Before 09:30 IST the worker waits. Mechanical safety (max-loss, stale feed, square-off, VWAP stop)
  runs on every poll regardless.
- Each newly completed five-minute bucket is evaluated once; a start-stamped one-minute candle is not
  complete until the next minute begins, and all five exact minute slots must exist once. In websocket
  mode the host also waits until all five **official** REST source minutes are present, so an
  intermediate REST hole blocks inference instead of letting invented OHLC through.
- **Flat:** Codex is called only when the bar is an eligible candidate, and never after today's
  entry. On most bars of most days there is no model call at all.
- **Open:** Codex is called once per completed bar for HOLD or a premise exit.
- At 15:00 new entries stop; exits continue. At 15:15 the host squares off and stops the worker.

## Live safety

Defaults are `CPR_AI_ENABLED=false`, `CPR_AI_VIRTUAL_TRADING=true` and `CPR_AI_LIVE_TRADING=false`.
A real order is possible only through the standard double gate: both global
`LIVE_TRADING_ENABLED=true` and `CPR_AI_LIVE_TRADING=true`, after the normal startup exposure audit
and configuration validation.

**Every CPR AI entry is a naked short option.** The double gate authorizes it; there is no third
short-premium switch. Budget naked-option margin (roughly ₹1.5–2 lakh per lot). The spot stop is a
strict host trigger, not a guaranteed fill: gaps, illiquidity, broker latency, or a rejected
buy-to-close can exceed the planned loss. The worst backtest trade lost 268 premium points on an
expiry-day crash.

**Max-loss.** `CPR_AI_MAX_LOSS` (default ₹5,500) is checked on the sold leg's mark-to-market. At a
75-unit lot that is about 73 premium points. In the backtest it closed sold legs that dipped and
later recovered, cutting PF from 1.69 to 1.52; ₹10,000 kept PF at 1.63. The default is unchanged;
choosing it is an operator decision.

Every exposure-increasing action requires a successful pre-action audit and a fresh post-inference
recheck of lifecycle, market-data health, entry cutoff, square-off, and that the fresh spot has not
already crossed the VWAP stop (`stop_already_breached`). A submitted entry, or possible live
exposure, uses up the session's single entry; a clean refusal leaves it available for a later bar. A
live close that is not broker-confirmed flat keeps the position open for reconciliation.
Risk-reducing exits remain available if decision logging fails.

## Isolated Codex runtime

Install the exact optional stack separately from broker dependencies:

```powershell
python -m pip install -r requirements-ai.txt
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
`Backtest Outputs/cpr_ai_decisions.jsonl` by default. Each row has an IST
`recorded_at` timestamp and an `audit_stage`: `PRE_ACTION` is the host record
before an entry may increase exposure, and `POST_ACTION` records the actual
submission/confirmation result afterward. Direct diagnostic callers retain the
safe `DIRECT` stage.

Each row keeps the full frozen context, including the `trend_day_candidate`
verdict, so a vetoed candidate can later be scored against the deterministic
baseline. The `bar` object records the start timestamp, frozen signature, the
one current signature captured when the host finalizes validation or a terminal
fail-closed outcome, all five required official minute stamps, the required
stamps present in the inference snapshot, and the resulting exact coverage
boolean.

Each `attempt_evidence` item records only its request kind (`normal` or the fixed
`tool_repair`), typed evidence result, safe tool name/status records, and token
usage. A provisional timeout marker can appear when a turn was selected but no
child evidence returned before the shared deadline; empty tool/usage fields do
not claim that a launched child consumed zero tokens. Any terminal failure
remains a fail-closed HOLD.

Credential-like mapping fields are removed recursively before serialization, and
the logger deliberately omits model reasoning/final responses, auth data, local
paths, broker/order/venue details, symbols, quantities, and SDK error text.
Logging never makes a proposal executable, and an enabled log must succeed before
an entry may be submitted.

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
sizing, risk, execution, or exit authority into the model or an MCP tool. Any
change to the prompt's evidence numbers should come from a fresh run of
`cpr_ai_trend_day_backtest.py`, with a prompt-version bump.
