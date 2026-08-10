# LLD — Regime Adaptive router

**Owns:** `Signal Generators/Regime Adaptive Strategy/`
**Ported from:** [`workratananmol-hub/nifty-options-paper-trading-bot`](https://github.com/workratananmol-hub/nifty-options-paper-trading-bot) (MIT)
**Read before enabling live:** [`REGIME_PORTING_NOTES.md`](../../Signal%20Generators/Regime%20Adaptive%20Strategy/REGIME_PORTING_NOTES.md)

---

## 1. Responsibility

One worker that switches its trading **rule** based on the measured regime,
instead of running a single rule and hoping the market suits it.

```
 read ADX on each completed bar
   │
   ├─ trending  ──► opening-range breakout, confirmed by VWAP
   ├─ ranging   ──► fade back to VWAP
   └─ ADX missing ──► NO TRADE
```

The third branch is the interesting one: it **never guesses the regime**. A
missing ADX is not treated as "probably ranging" — it is treated as "I don't
know", and not knowing means not trading.

---

## 2. Structure

| File | Role |
|---|---|
| `Nifty Regime Adaptive Signal Generator.py` | The router: reads ADX, selects a candidate, returns the signal |
| `regime_candidates.py` | The two candidate rules, as **library code** |
| `regime_common.py` | Shared helpers; re-exports indicators from `misc_strategy_common` |
| `REGIME_PORTING_NOTES.md` | What was and was not ported, and why |

### 2.1 Why the candidates have no worker of their own

`regime_candidates.py` is deliberately library code with **no worker**. If the
breakout rule also ran as its own strategy, the router and that strategy could
take the *same* signal at the same moment — doubling size on one idea while the
roster appeared diversified. Keeping the candidates worker-less makes that
impossible by construction rather than by convention.

If a candidate is ever wanted as a standalone strategy, the router must gain an
explicit exclusion; do not simply add a worker.

---

## 3. Wiring

Regime Adaptive is the fourteenth strategy built through
`_build_signal_gen_worker_class` (the same factory as the 13 TradingBot ports),
so it is an ordinary `AtmSingleLegStrategyWorker` from the runner's point of
view. Knobs are `REGIME_ADAPTIVE_*`.

It is also the **first and only user of the shared bid/ask spread gate** at a
non-zero default: `<PREFIX>_MAX_SPREAD_PCT` is `2.0` here and `0` (off) for
every other strategy, so introducing the gate changed nothing else. See
[`risk-and-safety.md`](risk-and-safety.md) §5.

---

## 4. Porting gaps — read before enabling live

These are the honest differences from the source project. They are recorded here
and in `REGIME_PORTING_NOTES.md` because a ported strategy that quietly differs
from its origin is a trap.

| Source behaviour | Here | Why |
|---|---|---|
| Volume-weighted VWAP | **Equal-weight proxy** | This runner's feed carries no volume. |
| India VIX veto | **Not implemented** | Absent *by choice*, not for want of data — the source project also runs on Dhan. |
| Market-breadth veto | **Not implemented** | Same. |

The VWAP proxy matters most to the ranging branch, which fades *to* VWAP. A
proxy VWAP is a different line from a true one, so the fade target is not
identical to the source's.

---

## 5. Testing

`Tests/Signal Generators/Regime Adaptive Strategy/test_regime_adaptive.py`, with
a `conftest.py` that puts both the **source** strategy folder and its parent
`Signal Generators/` on `sys.path` — the parent because `regime_common`
re-exports shared indicators from `misc_strategy_common` one level up. At runtime
`regime_common` bootstraps that itself; under pytest the import can arrive
through a different entry point, so it is done in the conftest as well.

Coverage: the router's three branches, including the ADX-missing no-trade path,
are the cases that matter most.

---

## 6. Known limitations

- Equal-weight VWAP (§4) — the single biggest divergence from the source.
- Two vetoes unimplemented (§4). Adding them is a real improvement, not a port
  fix; they would need a VIX feed and a breadth source wired in.
- The regime read is ADX-only. A second regime input would change the router's
  contract and should be an ADR, not a quiet edit.
