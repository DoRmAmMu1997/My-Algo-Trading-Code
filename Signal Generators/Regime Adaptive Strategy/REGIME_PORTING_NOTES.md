# Regime-Adaptive Port — What Was Ported, What Was Not, and Why

**Source:** [`workratananmol-hub/nifty-options-paper-trading-bot`](https://github.com/workratananmol-hub/nifty-options-paper-trading-bot),
`src/strategies/` — **MIT licence**, reuse permitted. The rules below were
reimplemented in this repo's idiom rather than copied file-for-file; the origin is
attributed in each new module's docstring.

**Files added by this port** — all of them in this folder,
`Signal Generators/Regime Adaptive Strategy/`:

| File | Role |
|---|---|
| `regime_common.py` | Session date, session VWAP, session opening range |
| `regime_candidates.py` | The two candidate rules, as pure column-producing functions |
| `Nifty Regime Adaptive Signal Generator.py` | The router — the only new worker |
| `conftest.py` | pytest `sys.path` bootstrap for this folder |
| `test_regime_adaptive.py` | Behaviour tests for the above |

**One note on imports.** The master's `load_module()` puts only the loaded file's
own directory on `sys.path`, so a module in here cannot see
`misc_strategy_common.py` one level up. Exactly one module — `regime_common.py` —
does the path bootstrap and re-exports the shared indicators; everything else here
imports only from its own siblings, and `conftest.py` does the same for pytest. If
you add a file to this folder, import through `regime_common` rather than adding a
second bootstrap.

---

## Read this first: the fidelity gap

**VWAP here is a proxy, and both candidate rules are VWAP-centric.**

`normalize_dhan_intraday_response` emits only `timestamp/open/high/low/close` —
the live runner never receives volume. So `attach_session_vwap` computes a true
volume-weighted VWAP **only when a `volume` column is present** (backtests on the
Data Extractors' CSVs), and otherwise falls back to an **equal-weight expanding
mean of the typical price** per session.

Every bar carries a `vwap_is_proxy` boolean recording which one was used, and the
router puts it in each decision's `debug` dict, so the journal shows it per trade.
But no test can catch the consequence: **in live and paper sessions this strategy
is not fading the same line the source fades.** An equal-weight mean lags a
volume-weighted one differently on days with a lopsided volume profile — exactly
the trending-open days the breakout branch cares about.

This is the single biggest reason the strategy ships paper-only.

---

## Fade distance: diverged, then corrected

Found on a later line-by-line re-read of the source, not at porting time, and
recorded here because the first paper sessions ran against neither value.

| | Source (`vwap_mean_reversion.py`) | Port, as first written | Port now |
|---|---|---|---|
| Fade trigger | `max(atr * 0.6, 15.0)` | `atr * 1.5`, **no floor** | `max(atr * 0.6, 15.0)` |

The original port demanded the price stretch **2.5x further** from VWAP before
fading, and had no absolute floor, so the mean-reversion branch would have fired
far less often here than in the project it came from — on a quiet day, possibly
never. Corrected to match: `meanrev_atr_mult=0.6` plus a new
`meanrev_min_points=15.0` (`REGIME_ADAPTIVE_MEANREV_MIN_POINTS`), mirroring the
multiplier-plus-floor shape the breakout buffer already had.

The floor earns its place independently of fidelity: on a very quiet session
`atr * 0.6` shrinks to a couple of points, at which point ordinary noise around
VWAP reads as an "extension" worth fading. Two tests pin both halves — one where
the multiple dominates, one where the floor does.

**Expect this branch to trade noticeably more than it did.** That is the intended
effect, but it means the fade side is effectively untested in paper until it has
run at these values.

Everything else numeric matches the source: the breakout buffer
(`max(atr * 0.05, 2.0)`), the fade's ADX stand-down (`adx >= 25`), and the
router's own trend threshold (`adx >= 20`).

---

## Dropped: gates this port does not implement

The operator's decision was **drop and document**, not proxy and not fail-closed —
a fabricated proxy for a risk gate is worse than a documented absence, because it
reads as protection that is not there.

> **Correction (2026-08-03).** An earlier revision of this file called these gates
> "structurally unavailable" and said the data had "no source in this repo". That
> was WRONG, and wrong in the direction that discourages work: it would tell you
> not to bother trying. The source project runs on **Dhan — the same broker this
> runner uses** — and gets nearly all of it from ordinary REST calls we already
> make or easily could. What follows is what it actually does.

| Source gate | Where the SOURCE gets it | Status here |
|---|---|---|
| Max 2% bid/ask spread | The **option-chain response**, not the tick feed. Dhan's `/optionchain` returns `top_bid_price` / `top_ask_price` / `top_bid_quantity` / `top_ask_quantity` per CE/PE node; their `normalize_quote_depth()` reads those with a `depth.buy[0]`/`depth.sell[0]` fallback. | **IMPLEMENTED** — see "The spread gate" below. `REGIME_ADAPTIVE_MAX_SPREAD_PCT=2.0`. |
| Chain liquidity score | Same chain response — top-of-book quantities plus OI/volume. | Buildable, same call, not built. The quantities are in the payload the spread gate already fetches. |
| India VIX "filter" | Dhan quote on **security id 21, segment `IDX_I`**. | **Not a filter in the source either.** `build_market_context` fetches it onto `ctx.india_vix` and **no strategy ever reads it** — it is telemetry. The router's only volatility veto is on `global_context["US_VIX"] >= 30`, a *different* instrument from *yfinance*. |
| Market-breadth "filter" | Dhan `quote_data` batched over the **NIFTY-50 constituents on `NSE_EQ`**, counting advances by `net_change`, off a shipped `fixtures/nifty50_symbols.json`. | **Never gates a trade.** In `opening_range.py` breadth only adds `+0.1` to `conf`, the uncalibrated heuristic score. The entry condition is purely OR-break + VWAP. `vwap_mean_reversion.py` ignores breadth entirely. Our contract has no score field, so this would cost ~50 quotes/refresh to compute a number with nowhere to go. |
| Liquidity score (`< 30` veto) | `compute_chain_metrics`: median `spread_pct` → `100 - med*8`, median OI → `oi/100`, blended `0.6/0.4`. | **IMPLEMENTED** — see "The liquidity gate" below. `REGIME_ADAPTIVE_MIN_LIQUIDITY_SCORE=30.0`. |
| Futures-basis filter | Spot vs NIFTY future LTP via `NSE_FNO` batch quote. | Recorded on the context; not read by either ported branch. |
| Event-risk / news blackout | **Does not exist in the source either.** Its `global_context.py` pulls *global market proxies* (USDINR, crude, gold, SPX, Nasdaq, Dow, Nikkei, Hang Seng, VIX) from **yfinance**, 5-day historical closes, self-described as "research proxies only". No news feed, no economic calendar. | Not ported. Would add a new unpinned dependency with Yahoo ToS caveats for last-close proxies. |

**What is genuinely true about the tick feed:** we subscribe `MarketFeed.Ticker`
only and `Dependencies/tick_bar_builder.py` drops depth packets, so there is no
bid/ask *in the tick path*. That part stands. The error was concluding the runner
therefore has no access to bid/ask at all — the source never used ticks for it.

**What this means operationally:** less than the earlier wording implied. Reading
the source's two candidate strategies line by line, **neither VIX nor breadth
gates an entry** — breadth only moves a heuristic score, and India VIX is fetched
but never read. Porting them would not change a single trade decision here. The
genuinely missing veto is `liquidity_score < 30`, which does gate the router.
The existing runner-level protections apply (per-strategy daily max-loss cap,
15:15 square-off, the `_get_dealable_option_ltp` freshness/refusal path on live),
and as of the spread gate below, entries into a wide book are now refused too.

---

## The spread gate (implemented)

Every strategy in this runner BUYS options, so a 2%-of-mid spread is a 2% loss
booked at entry, before the thesis has done anything. This was the cheapest gap
to close because the data was already arriving and being thrown away.

- **Quote source:** `top_bid_price` / `top_ask_price` out of the `/optionchain`
  response for the strike and expiry actually being bought, parsed by
  `_parse_option_chain_quote`. Falls back through the alternate key spellings and
  then the depth ladder, because the SDK has shifted casing between releases.
- **Metric:** `(ask - bid) / mid * 100`. Mid is the reference because that is what
  a marketable order is measured against, and it keeps the number symmetric.
  A crossed or half-empty book returns `None` — **unknown, never 0.0** — so a
  broken quote can never read as a tight one.
- **Where it runs:** last check in `enter_position`, *after* sizing. Sizing is
  local and free; the chain call is rate-limited, so a trade sizing would have
  rejected never spends one.
- **Rate limit:** Dhan allows one `/optionchain` per 3s per (underlying, expiry),
  so all workers share one short TTL cache (`OPTION_CHAIN_QUOTE_CACHE_SECONDS`).

**The paper/live split, which mirrors `_get_dealable_option_ltp`:**

| Situation | Paper | Live |
|---|---|---|
| Spread known, wider than the cap | **Refuse** | **Refuse** |
| Spread unknown (chain failed, or no quote for that strike) | Proceed, warn | **Refuse** |

A too-wide spread is a deterministic property of the market, so refusing in paper
too keeps the Sheet's paper rows predictive of live. An *unreadable* quote is an
infrastructure failure, not a market fact: real money is not spent on a check that
did not run, but paper keeps the observation rather than losing a data point to a
transient API error.

**Off by default everywhere else.** `<PREFIX>_MAX_SPREAD_PCT` defaults to 0 for
every strategy except Regime Adaptive (`_DEFAULT_MAX_SPREAD_PCT`), so no existing
worker's behaviour changed. The default lives in code, not only in `env.example`,
so deleting the `.env` line cannot silently disarm it.

---

## The liquidity gate (implemented)

The one veto the source applies that this port was genuinely missing — and unlike
VIX and breadth, this one really does gate the router:

```python
if ctx.liquidity_score is not None and ctx.liquidity_score < 30:
    return NO_TRADE  # liquidity_veto
```

It asks a **different question from the spread gate**: that one asks "is the
contract I am buying quoted tightly", this asks "is this chain tradeable at all".

Reproduces `compute_chain_metrics` exactly, including its upper-median convention
(`sorted(x)[len(x)//2]`, not the mean of the two middles) and its habit of
substituting 50.0 for a component whose input list is empty:

```
spread_score = clamp(100 - median(spread_pct) * 8, 0, 100)
oi_score     = clamp(median(oi) / 100,             0, 100)
score        = spread_score * 0.6 + oi_score * 0.4      ->  veto below 30
```

Costs **no extra API call** — it reads the same cached `/optionchain` response the
spread gate just fetched for that expiry. One fetch, two gates.

**One deliberate deviation.** Upstream, a chain with *no strikes at all* scores
`50*0.6 + 50*0.4 = 50` and sails through the veto. Here that returns `None`
instead: receiving no strikes is not evidence of a liquid market. The usual
paper/live split then applies — live refuses, paper proceeds with a warning.

### Watch this on the first paper day

The medians run over **every listed strike in the expiry**, both CE and PE. NSE
lists a long tail of far-OTM strikes that barely trade, so the median spread can
be dominated by contracts nobody would ever buy, and the score can sit low even
when the ATM options are perfectly liquid. In that case the gate is measuring
NSE's strike listing rather than the market, and it would quietly stop the
strategy trading at all.

That is why a veto logs at WARNING with **every component** — strike count, quoted
count, median spread, median OI, and both sub-scores. One session's logs should be
enough to tell "genuinely illiquid" from "arithmetic dominated by dead strikes".
`REGIME_ADAPTIVE_MIN_LIQUIDITY_SCORE=0` disables it if the latter turns out to be
the case.

---

## Dropped: the third branch

The source router dispatches between **three** candidates. Only two are ported.

`oi_liquidity_momentum` needs the option chain per evaluation. The factory
contract (`build_*_with_indicators(frame, config)` → `evaluate_candle(frame, position)`)
can pass a frame and nothing else, so this branch cannot be expressed as a
signal generator at all — it would need a hand-written chain-consuming worker plus
a shared chain cache that does not exist. The chain endpoint is also already
rate-budgeted to Opening Strike (30s refresh) and Delta-0.2 (once daily).

Consequence: **when ADX is below the trend threshold, this port always takes the
mean-reversion branch**, where the source would sometimes have taken the OI
branch instead. The dispatch is a two-way split, not a three-way one.

---

## Dropped: the contract selector

The source picks its option by nearest expiry, delta 0.35–0.60, and a 2%
spread cap. Not ported and not needed — this repo already has
`OptionsContractResolver` and the SLH-008 current-week expiry convention, and has
no bid/ask to gate on. The port buys the ATM CE/PE like every other member of the
`AtmSingleLegStrategyWorker` family.

---

## Deliberate deviations in the ported code

### Router-only topology

In the source, the two candidates are standalone strategies **and** router inputs.
Here `regime_adaptive` is the **only** worker; `Regime Adaptive Strategy/regime_candidates.py` exposes no
`Config`, no `Engine`, no `PositionContext`, has no env prefix and no P&L row, and
is absent from `test_trading_bot_ports.py`'s `PORTS` table.

The reason is exposure, not tidiness: if a candidate were also a worker, it and
the router could take the **same signal in the same session** — genuine double
size that the Google Sheet would not reveal, because each row would look like an
ordinary independent strategy.

### Exits are branch-agnostic

`RegimeAdaptivePositionContext` is fixed by the worker factory to
`direction / entry_underlying / stop_underlying / target_underlying`. There is
**nowhere to record which branch opened the trade.**

So `_evaluate_exit` checks, in order: stop → target → an opposing setup on the
**currently active** branch. If the regime flips mid-trade, the stop and target
still govern, so a flip can never strand a position — but the "opposite signal"
exit may be evaluated by a different rule than the one that entered.

### The single-builder collapse

The factory calls exactly one builder and one `evaluate_candle`. The builder
therefore computes **both** branches every bar (`br_*` and `mr_*` prefixes), sets
a `regime` column, and collapses the active branch into the repo's canonical
column contract (`long_setup` / `short_setup` / `*_entry_price` /
`*_stop_from_setup` / `*_target_from_setup`).

The routing decision thus lives **in the data**, which is why
`test_regime_adaptive.py` can assert the branch selection without running the
engine at all.

### Fail-closed rules kept from the source

- **ADX missing → `HOLD`.** Not a default, not a guess. `regime` becomes `NONE`.
- **ATR missing → no setup**, both branches. The breakout buffer is undefined
  without it, and the fade distance is undefined without it.
- **VWAP missing → no setup**, both branches.
- **Mean reversion stands down at `adx >= adx_meanrev_ceiling`** (default 25),
  above the router's own trend threshold (default 20). The bands overlap on
  purpose: between 20 and 25 the breakout branch is already active, so the
  ceiling only matters if the thresholds are retuned apart.
- **Opening range is `NaN` until its window closes.** A half-formed range is
  never publishable — you cannot break a level that is still being formed — and
  it never leaks across a session boundary.
- **Unusable entry levels → `HOLD`** (this repo's convention rather than the
  source's). A setup whose entry/stop/target is non-finite or mis-ordered is
  refused with reason `<REGIME>_<side>_invalid_levels` and `signal_triggered`
  still `True`, so the refusal is visible in the log instead of looking like a
  bar with no setup. This matches the other 13 ported strategies. It matters
  because the master sizes the position off the entry-to-stop distance: at
  `stop == entry` it would size off nothing, and a fade whose VWAP has converged
  onto the close produces exactly that (`target == entry`).

---

## Before this goes live

1. Several clean **paper** sessions. It is unproven code on a proxy VWAP.
2. Confirm in the log that the `regime` column actually flips branch as ADX
   crosses the threshold — a strategy stuck on one branch all day is a bug, not a
   regime read.
3. Decide what to do about the absent spread/liquidity veto (see the table above).

`REGIME_ADAPTIVE_LIVE_TRADING=false` and the global `LIVE_TRADING_ENABLED` gate
both have to be flipped for any of this to reach a broker; the default ships with
the strategy's own gate off.
