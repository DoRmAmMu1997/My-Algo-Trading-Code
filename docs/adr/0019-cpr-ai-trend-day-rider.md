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
only while `entries_today` is 0. The prompt (`cpr-trend-day-rider-v1`, now v2 — see the update
below) carries the evidence above,
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
- **Max-loss.** The default `CPR_AI_MAX_LOSS` of ₹5,500 (about 85 premium points on the 65-unit
  NIFTY lot) cuts PF from 1.69 to 1.56 in the backtest, because it closes sold legs that dip and
  then recover. At ₹10,000 (about 154 points) it fires once in five years and PF is 1.73. The
  default was left unchanged; raising a risk limit is the operator's decision. *(Corrected
  2026-09-28: the first version of this ADR assumed a 75-unit lot.)*
- Run `python algo.py backtest --strategy cpr-ai-trend-day` to reproduce the numbers (about two
  minutes with the options folder; spot points without it). `--max-loss-rupees` simulates the kill
  switch.
- Paper first (`CPR_AI_LIVE_TRADING=false`), with at least two clean paper sessions before any live use.
- See [`../lld/cpr-codex-ai-agent.md`](../lld/cpr-codex-ai-agent.md) and ADR-0018.

## Update 2026-09-28: the first paper day, and prompt v2

The first paper session took one bearish candidate (the 11:00 bar: gap down 76 points, range 1.5 ×
ATR5, below S1). Codex accepted it and the host sold the Sep-29 22800 CE at 132.80. After an
hour and a half in which price drifted sideways toward the VWAP stop without touching it, Codex
made a premise exit at 141.45 with regime `SIDEWAYS` (−₹562). Price then fell; another worker
bought the same contract back at 90.25 at 14:34. Holding to the stop or 15:15, as the backtest
does, would have been worth roughly +40 premium points. The exit also did not meet the prompt's
own example of a failed trend day (it had retraced 13% of the move, not half).

A five-year test of rule-based "stall" exits on top of the VWAP stop confirmed the lesson. Exiting
after 60–90 minutes without a new session extreme, or when price drifted near the stop, fired on
73–137 of 264 trades, helped and hurt about equally, and cut the total by 150–450 points.

**Decision (operator):** keep Codex's premise exits, but tighten the guidance. Prompt
`cpr-trend-day-rider-v2` states that a stall, a sideways drift or a pullback toward the stop is not
a failure for a sold option, that the session must not be relabelled `SIDEWAYS` just because price
paused, that an exit needs a completed bar retracing at least half of the session's trend move, and
that doubt resolves to `HOLD`. It also quotes the stall-exit evidence. Removing premise exits, or a
host gate on them, were considered and not chosen.

The day also exposed an infrastructure risk that is not specific to CPR AI. The trading laptop was
put to sleep twice during the session (Windows logged "Sleep Reason: Application API", 9 and 17
minutes), and Windows automatic maintenance (a defrag/re-trim of C: and a licensing-service
migration) froze the machine six more times in the afternoon. That is about 74 minutes with no stop
monitoring for any worker. Nothing in the runner can act while the machine is asleep, so this is an
operator-side fix: no sleep on AC power during market hours, and automatic maintenance scheduled
outside them.

## Update 2026-10-01: prompt v3, veto examples checked against the data

The next two paper sessions had no candidate, which is correct behaviour. By 13:30 the session
range had reached only 0.90 × ATR5 on 29 Sep (a V-shaped expiry day) and 0.75 × ATR5 on 30 Sep, so
the host never consulted Codex. On 1 Oct the first candidate under v2 came on the 12:45 bar. That
was a single 60-point bar that lifted the range from 0.81 to 1.08 × ATR5, and Codex accepted it.

Bars like that one raised a question: v2 listed four example red flags, but the backtest takes
every candidate, so none of them had been tested. Each was scored on the 264 five-year candidates
(SELL-opposite premium points after costs; the baseline is PF 1.69):

| v2 veto example (as measured) | Trades | Result |
|---|---|---|
| Whipsaw: four or more closes switching side of the running VWAP before the candidate | 96 | PF 1.71 vs 1.67 for the rest; vetoing would give up 908 of 2,542 points |
| A push into R2/S2 or a prior-day extreme (close within 0.1 × ATR5) after three same-direction bars covering more than 0.5 × ATR5 | 6 | Five won; average +103 points; the best group |
| One bar at least half the session range, and the candidate bar's wick against the trend at least 40% of its range | 7 | PF 0.81 (−34 points) |
| Bullish, confluence exactly 2, expiry day | 11 | PF 1.01 (+4 points) |

The size of the candidate bar points the same way. Candidates whose own bar spanned at least a
quarter of ATR5 had PF 2.98 (36 trades). Candidates whose bar alone lifted the range past 1.0 × ATR5
had PF 1.86, against 1.44 for those that were already expanded.

**Decision (operator):** prompt `cpr-trend-day-rider-v3` drops the whipsaw and the R2/S2 examples
and keeps the two rare, roughly break-even ones. The climactic-bar example now says that the wick is
the warning, not the size of the bar. The prompt also tells Codex not to veto for a big candidate
bar, a choppy morning or a push into R2/S2 or a prior-day extreme, and quotes the evidence above.
The host gate, the stop and Codex's veto-only role are unchanged.

## Update 2026-10-01 (after the close): prompt v4, a 70% retrace for premise exits

The 1 Oct trade (sold the Oct-06 22400 CE at 139.45 on the 12:45 bar, NIFTY 22,378) ran 161 points in
its favour to 22,217 at 14:05. NIFTY then rallied to 22,445 by 15:14. On the 15:05 bar the close had
retraced 50.8% of the session range, which just cleared v2's "at least half of the session's trend
move" line. Codex exited at 15:10 at 152.35 (−₹838.50, confidence 10). Holding to the 15:15 square-off
would have been somewhat worse; by a delta estimate, about −₹1,300.

v2's half-retrace rule had never been tested, so it was scored on the 264 five-year trades. It
exits at the first post-entry bar whose close is at least halfway back across the session range, and
the exit fills six minutes after the bar starts (one minute after it completes), as on 1 Oct. A rule
only counts if it fires before the trade's own stop or square-off.

| Exit cut-off (share of the session move retraced) | Trades it fired on | Helped / hurt | Change in the five-year total |
|---|---|---|---|
| 50% (v2) | 13 | 6 / 7 | −342 points (2,542 → 2,200) |
| 50%, but not after 14:30 | 6 | 3 / 3 | −239 |
| 60% | 3 | 2 / 1 | +76 |
| 70% | 2 | 1 / 1 | +29 |

Together with the stall exits tested on 28 Sep, every rule-based premise exit so far has lost against
holding to the stop or 15:15. The 60% and 70% rows are too small to show an edge either way.

**Decision (operator):** keep premise exits but raise the cut-off to 70%. Prompt
`cpr-trend-day-rider-v4` requires a completed bar retracing at least 70% of the session's trend move,
read off the frozen `market_structure.trend_day_candidate.location` (0.70 or more for a short, 0.30 or
less for a long). It says that a half retrace is not enough, and quotes the evidence above. Removing
premise exits altogether (the backtest's own assumption) was offered and not chosen. At 70% a premise
exit should be rare, about once in two years of trades.
