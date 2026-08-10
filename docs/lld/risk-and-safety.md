# LLD — Risk, sizing, and the live-trading safety model

**Owns:** `Dependencies/risk_sizing.py`, `Dependencies/next_open_entry.py`,
`Dependencies/secret_redaction.py`, `_strategy_size_multiplier` / `_scaled_*` /
`_live_config_errors` / `_spread_gate_allows_entry` (master file)
**Related ADRs:** [0003](../adr/0003-acknowledgement-is-not-a-fill.md), [0004](../adr/0004-paper-by-default-double-gate.md), [0006](../adr/0006-per-strategy-size-multiplier.md)

> **This is live-money code.** Every rule below exists because the alternative
> costs real money. When in doubt, the safe direction is to refuse the trade.

---

## 1. The one principle

**Ambiguity is treated as the dangerous case.** Not the neutral case, not the
optimistic case. Concretely:

- An unknown broker name does not mean "pick the default" — it means live
  trading is off for the session.
- An unreadable bid/ask quote does not mean "the spread is probably fine" — it
  means the live entry is refused.
- An acknowledged order does not mean "it filled" — it means the fill is unknown
  until proven, and unknown means exposure may exist.
- A malformed size multiplier does not mean "assume 1 and carry on live" — paper
  gets 1, live gets blocked.

---

## 2. The live-trading double gate

A strategy places a real order only when **both** are true:

```
LIVE_TRADING_ENABLED = true          # global kill-switch, default FALSE
<PREFIX>_LIVE_TRADING = true         # per strategy,       default FALSE
```

Plus `LIVE_BROKER` ∈ {`KOTAK`, `SHOONYA`, `FLATTRADE`, `DHAN`}. **An unknown
value fails closed** — live disabled, paper only.

The mirror-image gate for paper is deliberately *asymmetric*:

| | Global switch | Per-strategy | Default |
|---|---|---|---|
| **Live** | `LIVE_TRADING_ENABLED` (required) | `<PREFIX>_LIVE_TRADING` (required) | off |
| **Virtual/paper** | *none by design* | `<PREFIX>_VIRTUAL_TRADING` | on |

There is no global paper switch because "run everything" is the safe default for
paper and "run nothing live" is the safe default for live. One flag can never be
enough to risk money; one flag is enough to silence a strategy.

See [ADR-0004](../adr/0004-paper-by-default-double-gate.md).

---

## 3. Position sizing — `Dependencies/risk_sizing.py`

`SizingDecision` is the single authority for both the master runner and the
standalone SL Hunting executor.

```
lots = floor(risk_budget / (stop_distance × lot_size))
       capped at <PREFIX>_MAX_LOTS
```

Three deliberate behaviours:

1. **Floor, never ceil, and never force a minimum of one lot.** The older
   per-strategy helpers used `ceil` and forced at least one lot; both can exceed
   the configured budget.
2. **One-lot-over-budget is an explicit rejection**, not a rounded-down trade.
   If the smallest legal size costs more than the budget allows, the setup is
   skipped.
3. **Invalid inputs are rejections**, not defaults.

---

## 4. Per-strategy size multiplier

`<PREFIX>_SIZE_MULTIPLIER` (default 1, whole numbers 1–25, capped by
`MAX_SIZE_MULTIPLIER`) scales that strategy's whole size/risk set *together*:
`_LOTS`, `_MAX_LOTS`, `_RISK_BUDGET` and the absolute `_MAX_LOSS`.

Applied at **env-read time** through `_scaled_int` / `_scaled_float` /
`_strategy_size_multiplier`, so `risk_sizing.py` is untouched and scaled values
flow through sizing, the kill-switch, Telegram and the Sheet unchanged.

| Property | Reason |
|---|---|
| Per-strategy only, **no global switch** | One typo must not enlarge every enabled strategy. |
| Applies to **paper and live alike** | An enlarged size can be paper-validated before it risks money. |
| Malformed → 1 for paper, **blocked from live** | Guessing a size is worse than refusing. |

**Two knobs are deliberately NOT scaled**, because their totals already inherit
the multiplier and scaling them would square it:

- `<PREFIX>_MAX_LOSS_PER_LOT` (Delta20) — multiplied by scaled lots downstream.
- `<PREFIX>_STARTING_CAPITAL` / `_DAILY_MAX_LOSS_PCT` — their *product* carries it.

A drift-guard test fails if a new strategy reads a size-bearing knob with the
raw `_env_*` helpers instead of `_scaled_*`.

Two consequences worth knowing: the scaled budget also loosens the
"one lot exceeds the budget" skip; and because lots are floored, a 2× can land
slightly above a pure doubling while staying strictly inside the scaled budget.

---

## 5. The bid/ask spread gate

Every strategy here **buys** options, so a 2%-of-mid spread is a 2% loss booked
at the instant of entry, before the idea has done anything.

`_spread_gate_allows_entry` reads `top_bid_price` / `top_ask_price` off the
`/optionchain` response for the exact strike and expiry being bought and refuses
an entry quoted wider than `<PREFIX>_MAX_SPREAD_PCT`.

| Situation | Paper | Live | Why |
|---|---|---|---|
| Spread too wide | refuse | refuse | It is a market fact, so paper rows stay predictive. |
| Quote unreadable | allow (warn) | **refuse** | An API failure should not cost a paper data point — but it also should not spend real money on a check that did not run. |

Workers share one 3-second cache (`_fetch_option_chain_cached`) because Dhan
allows a single option-chain request per 3 s per underlying/expiry.

Default is `0` (off) for every strategy **except Regime Adaptive** (2.0), so
introducing the gate changed no existing strategy's behaviour.

---

## 6. Kill-switches and time gates

| Control | Scope | Effect |
|---|---|---|
| `<PREFIX>_MAX_LOSS` (scaled) | one strategy, one day | That worker stops; the session continues. |
| `LIVE_TRADING_ENABLED=false` | process | No real orders at all. |
| Square-off time (15:15 by default) | one strategy | Close everything, including both legs of a basket. |
| Entry cutoff | one strategy | No **new** positions; exits and square-off still run. |
| Stale-data liquidation (~30 s) | one strategy | Close rather than hold blind. |
| Startup exposure audit | process | Live workers do not start into a dirty book. |
| Post-exit cooldown (SL Hunting) | one strategy | Blocks re-entry from the moment the **whole** basket is confirmed flat. |

Note the difference between an entry cutoff and a square-off. SL Hunting's
10:30 `NO_NEW_ENTRY` is a cutoff: open positions, their stops and targets, and
the 15:15 square-off are all unaffected. When flat past the cutoff it skips the
LLM call entirely.

---

## 7. `NEXT_OPEN` signal safety — `Dependencies/next_open_entry.py`

Goldmine and Money Machine generate a setup on a *completed* candle but may only
enter at the **open of the following candle**. Two rules make that safe:

1. **Exactly one bar of life.** Candles are labelled by their START time, so a
   pending signal that is not taken on the very next bar expires. It cannot be
   silently carried into a different market.
2. **Price rebasing.** The entry reference is the next bar's open, not the
   signal bar's close.

---

## 8. Credential-safe logging — `Dependencies/secret_redaction.py`

`setup_logging()` installs `install_redaction_filter` on the **root** logger with
`environment_secrets(os.environ)` — every `.env` value whose KEY looks sensitive
and is ≥8 characters. Every record is scrubbed before it reaches the console or
the append-mode log, including lazy `%s` args and exception tracebacks.

This is not theoretical: `dhanhq`'s marketfeed puts the live access token **in
its websocket URL**, so a connect error would otherwise write it verbatim into a
log operators routinely share.

Short values (a 4-digit MPIN) are deliberately excluded from exact-match
replacement — they would blank strike prices and quantities — and are caught by
`redact_text`'s `name=value` pass instead.

**Do not hand-redact new call sites.** The root-logger filter covers them; a
local redaction is one more thing to forget.

---

## 9. Order-outcome rules (restated, because they are the ones that cost money)

| Result | Entry | Exit |
|---|---|---|
| `FILLED` | Position open | Position closed |
| `REJECTED` **with zero fill** | Fall back to paper | Position **stays open** |
| `REJECTED` with non-zero fill | Exposure exists — treat as `PARTIAL` | Partially closed |
| `PARTIAL` | Freeze new live entries, keep exits available, reconcile | Partially closed |
| `UNKNOWN` | Same as `PARTIAL` | Same as `PARTIAL` |

**Never** treat an acknowledgement, a truthy value, or an order ID as proof of
fill.

---

## 10. Testing

Every module named here carries a **90% branch-coverage budget** enforced by
`scripts/check_coverage_thresholds.py`:

`risk_sizing.py`, `next_open_entry.py`, `order_splitting.py`,
`secret_redaction.py`, `broker_contract.py`, `execution_ledger.py`,
`startup_exposure.py`, `trading_lifecycle.py`, `market_data_health.py`,
`tick_bar_builder.py`.

Suites: `Tests/Dependencies/test_risk_sizing.py`,
`test_next_open_entry.py`, `test_secret_redaction.py`, plus the double-gate,
spread-gate and multiplier cases in `Tests/test_nifty_multi_strategy_master.py`.

---

## 11. When changing anything in this document

1. Ask which direction the failure falls. If the answer is "it trades", stop.
2. Paper and live may differ **only** when the failure is an infrastructure
   problem rather than a market fact.
3. Add the test before the change.
4. Paper-validate for at least one session before enabling live.
