# ADR-0019: CPR AI trades the Trend-Day Rider, a backtested gate that Codex may only veto

**Status:** Accepted
**Date:** 2026-09-27
**Deciders:** repository owner

## Context

CPR AI used to trade the operator's SRSI/VWAP playbook with Codex judging the regime. CPR Algo 4
(ADR-0018) now trades that playbook with fixed rules, so the agent was free for a new strategy. The
operator asked for a **custom** five-minute strategy that may be nondeterministic (its judgment
living in the system prompt), designed from backtests on the five-year NIFTY and BankNIFTY one-minute
data.

The local expired-options set (`Backtest Outputs/expired_options/nifty`, weekly ATM±10 strikes,
one-minute premiums, 2021-09 to 2026-09; see ADR-0015) made it possible to score every idea on
**real option premiums** rather than spot points. Each trade was priced on its exact contract, with
a 2-point round-trip cost and the agent's decision delay (entry two minutes after the bar).

### What the research found

| Idea | Result (option premium points unless noted) | Verdict |
|---|---|---|
| Intraday momentum, NIFTY and BankNIFTY (first 30/60 min → rest of day) | correlation ≈ 0 | none |
| Narrow CPR ⇒ trend day | range/ATR flat across CPR-width quintiles | refuted |
| Option OI "walls" as support/resistance (distance-matched) | held no more often than other strikes | refuted |
| Expiry-day pin to the max-OI strike | price drifted away (slope −0.09) | refuted |
| Opening-range breakout (15/30 min) | spot +7 pts/trade, but BUY −1.4 and SELL +1.6 (PF 1.08) | too thin |
| VWAP-stretch fade; reversal after a failed trend day | PF 0.69; PF 0.88 | loses |
| PDH/PDL sweeps; NIFTY-BankNIFTY SMT divergence | PF ≈ 1.1 in spot, below 1 as options | none |
| Selling the ATM straddle every day 09:35→15:15 | ≈0 after costs, worst day −373 | no free premium |
| IV/RV, "richness", gap or opening range as volatility timers | no monotonic effect | none |
| Squeeze breakouts; expiry-afternoon breakouts | PF 0.8–1.16 | none |
| BankNIFTY confirmation of any of the above | no improvement | none |
| **Trend-day continuation** | see below | **edge** |

Trend-day continuation was the only robust effect. By late morning, a session that has out-ranged
its recent ATR and is pinned at one extreme on the trend side of VWAP tends to close near that
extreme. All 135 parameter variants tested were profitable when the trade sold the opposite ATM
option (72% with PF > 1.2). The effect grew monotonically with expansion and earliness, survived
costs up to 3 points, and did not depend on the ATR look-back (3, 5 or 14 sessions all gave PF ≈ 1.47).

## Decision

### 1. A deterministic host gate, shared by the worker and the backtest

`Signal Generators/CPR AI Agent/cpr_ai_trend_day.py` (pandas/numpy only) flags a candidate on the
newest completed five-minute bar when all of these hold:
- the bar starts 11:00–13:30 IST;
- the session range is greater than 1.0 × ATR5 (mean high−low of up to the five prior sessions in
  the store; fewer than three means no candidate);
- the close is in the top 15% of the session range and above VWAP (bullish), or in the bottom 15%
  and below VWAP (bearish);
- **bullish only:** at least 2 of 3 confluence factors — close beyond R1, a gap up from the prior
  close, close more than 0.35 × ATR5 from VWAP.

The stop is that bar's VWAP. The constants are code-owned, not `.env` knobs, because the backtest
numbers describe exactly this definition. The same module builds the frozen `trend_day_candidate`
fact and drives `cpr_ai_trend_day_backtest.py`.

### 2. The expression is always a sold ATM option

Every entry SELLS the ATM option on the side the market is leaving, on the current weekly expiry:
bullish sells the PE, bearish sells the CE. Time decay then works with the drift. There is no target,
no trailing and no add: the trade is held until the VWAP spot stop, the 15:15 square-off, or a
premise exit. One entry per session; no re-entry and no flip.

### 3. Codex may only veto

Codex is consulted only on an eligible candidate while flat, and on every bar while a position is
open. When flat it may accept the candidate (in its own direction, setup `TREND_DAY_CONTINUATION`)
or veto it with `HOLD`. When open it may hold or make a premise exit. The host policy accepts an
entry only if it restates the frozen candidate on the frozen close with the candidate's stop, and
only while `entries_today` is 0. The prompt (`cpr-trend-day-rider-v1`) carries the evidence above,
including the refuted beliefs, and tells the model to default to accepting and to veto only for a
concrete red flag.

### 4. What CPR AI no longer does

The SRSI/RSI/EMA context, the SIDEWAYS/TRENDING setups, the 30-point cap and 1R geometry, SRSI
reversal exits, staged trailing, the R2/S2 target, the TRENDING next-next BUY path and the R1 add
with its two-leg books are gone. The four frozen no-argument tools keep their names; their contents
changed. CPR Algo 4 keeps its own copy of the add-leg mechanics, now the only copy.

## Options considered

| Option | Why not chosen |
|---|---|
| Keep the SRSI/VWAP playbook in CPR AI | Algo 4 already trades it; its backtest was marginal (PF 1.05–1.14 in spot points). |
| BUY the directional ATM option on the same gate | PF 1.27 and a drawdown more than twice as deep in the final backtest. |
| A looser host gate with Codex selecting among ~2× the candidates | Makes the result depend on unmeasured model judgment; operator chose veto-only. |
| Codex-led entries with only risk gates | Least testable; the backtest could no longer serve as a baseline. |
| BankNIFTY cross-confirmation (per-bar REST fetch) | Measured no gain, so not worth the extra broker call. |

## Trade-off analysis

Selling options earns the drift plus theta and turned a 22-point spot edge into +9.6 premium points
per trade (264 trades, PF 1.69, max drawdown 300 points, 2021-09 to 2026-09). The cost is naked
short-option risk and margin: roughly ₹1.5–2 lakh per lot, and a gap through the stop can lose more
than the plan. The worst sold trade lost 268 points on an expiry-day crash.

Veto-only keeps the validated gate in charge. The backtest is a fair baseline, since it assumes
Codex accepts everything, and every veto is auditable. The model can still hurt returns by vetoing
winners or exiting early. The decision log keeps the full frozen candidate on every turn, so vetoes
can be scored counterfactually against the baseline.

The recent years are the weakest: PF 1.29 in 2025 and 1.07 in 2026 to date. The edge is real but
not large, and it is concentrated in a minority of strong trend days.

## Consequences

- `CPR_AI_*` names, the Sheet rows (`CPR AI Agent Strategy` and its `[LIVE]`/`[MIXED]` variants) and
  the double gate are unchanged. No new `.env` knobs.
- Codex runs far less often: flat sessions without a candidate make no model call at all.
- **Max-loss.** The default `CPR_AI_MAX_LOSS` of ₹5,500 (about 73 premium points on a 75-unit lot)
  cuts PF from 1.69 to 1.52 in the backtest, because it closes sold legs that dip and then recover.
  At ₹10,000 it fires twice in five years (PF 1.63, drawdown 273). The default was left unchanged;
  raising a risk limit is the operator's decision.
- Run `python algo.py backtest --strategy cpr-ai-trend-day` to reproduce the numbers (about two
  minutes with the options folder; spot points without it). `--max-loss-rupees` simulates the kill
  switch.
- Paper first (`CPR_AI_LIVE_TRADING=false`), with at least two clean paper sessions before any live use.
- See [`../lld/cpr-codex-ai-agent.md`](../lld/cpr-codex-ai-agent.md) and ADR-0018.
