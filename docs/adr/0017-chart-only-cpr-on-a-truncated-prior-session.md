# ADR-0017: The dashboard's CPR reads a truncated prior session, and the strategies' does not

**Status:** Accepted
**Date:** 2026-09-11
**Deciders:** repository owner

## Context

[`ADR-0016`](0016-read-only-loopback-monitoring-dashboard.md) added a read-only chart. Reading it
against what the strategies are doing needs the levels they watch, so the chart gained CPR, a
session VWAP and a stochastic oscillator.

VWAP and the stochastic were easy: the chart is handed the *same callables* the live strategies use
(`regime_common.attach_session_vwap`, `misc_strategy_common.stochastic`), so a drawn line cannot
drift from a traded figure.

CPR was not, because the operator wanted a different input window from the one the strategies use.

**What the strategies do today.** `cpr_strategy_logic.py::_add_daily_cpr` aggregates the previous
session as `prev_high=("high","max")`, `prev_low=("low","min")`, `prev_close=("close","last")` over
the whole day. Note it already takes the last *intraday* bar (~15:29), **not** the official
CAS-settled close — so the runner is not, and never was, using the auction print.

**What the operator asked for.** The prior session read from **09:15–15:15 inclusive** — high, low
**and** close — on the reasoning that an index close is settled by an auction and the last prints of
the day are the least trustworthy input to a pivot.

These cannot both be one number. The choice was between changing what two live strategies trade on,
and drawing a line on the chart that those strategies do not use.

## Decision

### 1. The chart's CPR is chart-only. The strategies are untouched.

`cpr_strategy_logic.py` and the CPR / CPR Algo 3 / CPR AI workers are not modified by one line.
A charting preference is not a reason to change live-money behaviour.

### 2. Only the input WINDOW differs — and a test enforces that

The pivot/BC/TC/R1–R4/S1–S4 algebra in `Dependencies/dashboard_indicators.py::pivot_levels` is
transcribed from `cpr_strategy_logic.py` unchanged. `test_only_the_input_window_differs_from_the_strategies_cpr`
feeds an **untruncated** prior session through both implementations and asserts the levels match
level-for-level. If anyone "improves" a formula on either side, that test fails.

### 3. Say it, in four places, one of them derived from the data

A caveat nobody reads is not a mitigation. So:

- the pane heading names the window and says *(chart only)*;
- an amber `CHART-ONLY` chip sits beside the CPR checkbox, its tooltip carrying the full caveat;
- every price line is titled `(chart)`, so a screenshot of the chart alone still carries it;
- a provenance caption prints the **actual inputs** — `Prior 2026-09-10 09:15-15:15 · 361 bars ·
  H 24630.08 L 24376.35 C 24485.94 · pivot 24497.46 (chart) · strategies 24491.21 (full session)`.

That last line is the real mitigation: it shows **both** figures. An operator can see the gap rather
than be warned one exists.

### 4. High and low are truncated too

The operator chose one consistent window over textbook CPR.

## Options considered

| Option | Verdict |
|---|---|
| **Chart only, strategies untouched** | **Chosen.** Reversible, zero live-money risk. Cost: the drawn line is not the traded line, which is why §3 exists. |
| Change `_add_daily_cpr` so both agree | Rejected *for now*. It is the only way to make chart and strategy provably identical, but it shifts the pivot and all eight R/S levels for CPR and CPR Algo 3, and CPR AI keeps a second copy that would silently diverge. Not a change to make as a side effect of a charting request. |
| Draw both sets of lines | Rejected. It answers the question honestly but puts up to 22 horizontal lines on one chart. Superseded by printing both *numbers* in the caption, which costs one aggregation per session and no clutter. |
| Truncate the close only, keep full-session H/L | Rejected by the operator, recorded here because it is the better-argued position. The auction argument is strong for the close and weak for the extremes: a genuine high printed at 15:22 is real price action, and discarding it moves R1–R4/S1–S4 more than the close change moves the pivot. The window is a parameter, so splitting it later is a two-line change. |

## Trade-off analysis

**What this costs.** Two definitions of CPR in one repository. That is a real cost and the reason
for the enforcement test and the four labels.

**What bounds it.** The chart is read-only and moves nothing; the divergence is small (a 15:15 close
differing by ~5 points moves the pivot by ~1.7); and the page shows both numbers.

**Why not compute the strategies' CPR by calling their code.** `_add_daily_cpr` is private, runs
inside a pipeline that resamples to 5-minute bars and needs TA-Lib, and returns a full enriched
frame. The chart needs three numbers from a different window. Re-using the *algebra* while
supplying a different window is the smaller coupling, and the equality test keeps it honest.

## Consequences

- **`Dependencies/dashboard_indicators.py` is pure** — pandas, no threads, I/O, clock or `.env`. It
  carries a 90 % branch-coverage budget and measures 98.9 %.
- **VWAP and the stochastic carry no such divergence**: they are the strategies' own objects,
  resolved through `load_module` under bare names, which returns the already-loaded instance. A test
  asserts identity with `sys.modules` — a prefixed alias would create a second copy that type-checks,
  runs, and diverges silently.
- **VWAP must always be labelled `VWAP*`.** It is an equal-weight proxy on every live row because
  the index feed carries no volume. Calling it plain "VWAP" would be a correctness error.
- **The 15:15 boundary is a module constant**, not an `.env` key, so it cannot drift per machine and
  cannot escape the config audit the module is deliberately outside.
- **If the operator later wants one definition everywhere**, the path is: change `_add_daily_cpr`,
  delete the truncation from `chart_cpr`, and the equality test starts passing on the *default*
  window rather than a widened one.
