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

## Dropped: gates with no data behind them

The operator's decision was **drop and document**, not proxy and not fail-closed —
a fabricated proxy for a risk gate is worse than a documented absence, because it
reads as protection that is not there.

| Source gate | Why it is not here |
|---|---|
| Max 2% bid/ask spread | **Structurally unavailable.** The feed subscribes `MarketFeed.Ticker` only, and `Dependencies/tick_bar_builder.py` explicitly drops depth packets. There is no bid and no ask anywhere in this runner. |
| Chain liquidity score | Same: needs depth, plus a chain snapshot per evaluation. |
| India VIX filter | No data source in this repo. |
| Market-breadth filter | No data source in this repo. |
| Event-risk / news blackout | No data source in this repo. |

**What this means operationally:** nothing stops this strategy from entering into
a wide or thin option. The existing runner-level protections still apply (the
per-strategy daily max-loss cap, the 15:15 square-off, the `_get_dealable_option_ltp`
freshness/refusal path on live), but there is no spread or liquidity veto at the
signal layer. Treat that as a live-trading precondition to solve, not a footnote.

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
