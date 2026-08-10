# LLD — Strategy workers

**Owns:** `BasePaperStrategyWorker`, `AtmSingleLegStrategyWorker`,
`NextOpenAtmStrategyWorker`, the hedged/spread/strangle workers, and
`_build_signal_gen_worker_class` (master file) · `Signal Generators/`
**Depends on:** [`market-data.md`](market-data.md), [`risk-and-safety.md`](risk-and-safety.md), [`execution-and-brokers.md`](execution-and-brokers.md)

---

## 1. Responsibility

One worker = one strategy = one thread. A worker owns its own positions, its own
P&L, its own risk knobs and its own poll cadence. Workers never call one another.
They share the market-data and execution-safety aggregate, lifecycle/shutdown
signals, and—when enabled—the Telegram event queue; each boundary is explicitly
synchronized.

That separation is why a strategy can be added or disabled without coupling its
position and decision state to another worker.

---

## 2. Class hierarchy

```
threading.Thread
 └─ BasePaperStrategyWorker            paper bookkeeping, P&L, max-loss,
     │                                 square-off, Telegram events, the
     │                                 enter_position / exit_position contract
     │
     ├─ AtmSingleLegStrategyWorker     buy ONE ATM option (CE or PE)
     │   │                             ── strike + expiry resolution
     │   │                             ── spread gate, sizing, stop/target
     │   │
     │   ├─ RenkoStrategyWorker
     │   ├─ EMATrendStrategyWorker
     │   ├─ HeikinAshiStrategyWorker
     │   ├─ ProfitShooterStrategyWorker
     │   ├─ OpeningStrikePCRVWAPATRWorker
     │   ├─ CPRStrategyWorker
     │   ├─ CPRAlgo3StrategyWorker     (multi-instrument: spot + ITM CE + ITM PE)
     │   ├─ CPRAIWorker                (optional — see cpr-codex-ai-agent.md)
     │   ├─ 14 factory-built ports     (_build_signal_gen_worker_class)
     │   │                              13 TradingBot ports + Regime Adaptive
     │   └─ NextOpenAtmStrategyWorker  signal on a completed bar, ENTER at the
     │       ├─ GoldmineStrategyWorker      next bar's OPEN
     │       └─ MoneyMachineStrategyWorker
     │
     ├─ SupertrendBullishWorker        hedged puts (main + protective leg)
     ├─ DonchianBearishWorker          hedged puts
     ├─ Delta20HedgedSpreadWorker      4-leg hedged spread
     ├─ LongStrangleWorker             time-based dual-leg OTM1 CE+PE buy
     └─ SLHuntingAIWorker              optional — see sl-hunting-ai-agent.md
```

`SLHuntingAIWorker` is defined only when its optional dependencies imported
successfully; otherwise the name stays `None` and `main()` skips it. A missing
optional dependency disables a strategy — it never breaks the run.

---

## 3. The worker loop

```
run():
  while not shutdown:
      sleep(<PREFIX>_POLL_SECONDS)

      # ---- always, position or not -------------------------------
      mechanical risk checks     stop · target · daily max-loss
                                 · stale-data liquidation · square-off time

      # ---- only when flat and inside the trading window -----------
      if flat and within window and not past entry cutoff:
          frame = store.read()                    # immutable copy
          if not health.entries_allowed(): continue
          candles = resample_ohlc_from_1m(frame, timeframe)
          signal  = <strategy logic>(candles)     # Signal Generators/…
          if signal:
              strike, expiry = resolve_atm(...)
              if not _spread_gate_allows_entry(...): continue
              lots = SizingDecision(...)          # fail-closed
              enter_position(...)                 # paper or live
```

Two properties are load-bearing:

1. **The risk checks run first and unconditionally.** They do not depend on the
   signal logic succeeding, on the LLM agents answering, or on the feed being
   fresh enough to enter. A worker that cannot decide anything can still stop
   itself out.
2. **`enter_position` / `exit_position` are the only execution doors.** Every
   worker — including the AI agents, which reach them through an executor
   shim — goes through the same tested path, so paper/live routing, broker
   selection, max-loss, square-off and Telegram behave identically everywhere.

---

## 4. The signal-generator factory

14 of the ATM strategies are not hand-written worker classes. `_signal_gen_ops(prefix)`
builds the env-knob accessors for a prefix, and `_build_signal_gen_worker_class(...)`
produces a worker class from a spec tuple. The spec list `_SIGNAL_GEN_WORKER_SPECS`
also feeds `STRATEGY_ENV_PREFIX`, so a ported strategy gets its name→prefix
mapping automatically.

Adding a ported strategy is therefore: write the signal module in
`Signal Generators/`, add one spec row, add its `<PREFIX>_*` keys to
`Dependencies/env.example`. The CI drift gate fails the build if the last step
is skipped.

`STRATEGY_ENV_PREFIX` maps display name → env prefix for the hand-written
workers (`"Renko" → "RENKO"`, `"CPRAlgo3" → "CPR_ALGO3"`, …) and merges the
factory specs in.

---

## 5. Per-strategy knobs

Every strategy reads the same shape of configuration under its own prefix:

| Knob | Meaning |
|---|---|
| `<PREFIX>_VIRTUAL_TRADING` | Default **true**. False = the thread never starts. |
| `<PREFIX>_LIVE_TRADING` | Default false. Live needs this **and** the global switch. |
| `<PREFIX>_SIZE_MULTIPLIER` | Default 1. Scales `_LOTS`, `_MAX_LOTS`, `_RISK_BUDGET`, `_MAX_LOSS` together. |
| `<PREFIX>_LOTS` / `_MAX_LOTS` | Size and hard cap. |
| `<PREFIX>_RISK_BUDGET` | Rupee risk per trade, used by `SizingDecision`. |
| `<PREFIX>_MAX_LOSS` | Daily kill-switch for this strategy. |
| `<PREFIX>_MAX_SPREAD_PCT` | Bid/ask spread cap. Default 0 = off, except Regime Adaptive. |
| `<PREFIX>_POLL_SECONDS` | Loop cadence. |
| `<PREFIX>_*_HOUR` / `_MINUTE` | Trading window, entry cutoff, square-off. |

There is **no global virtual-trading switch**: the default is that everything
runs and you silence strategies individually. Live trading is the opposite —
default off, and it needs two flags. The asymmetry is intentional (see
[ADR-0004](../adr/0004-paper-by-default-double-gate.md)).

---

## 6. Multi-leg workers

Three workers manage baskets rather than a single option:

| Worker | Legs | Notes |
|---|---|---|
| `SupertrendBullishWorker`, `DonchianBearishWorker` | main + protective put | Hedge sized and closed with the main leg. |
| `Delta20HedgedSpreadWorker` | 4 | `<PREFIX>_MAX_LOSS_PER_LOT` is deliberately **not** scaled by the size multiplier — the total already inherits it. |
| `LongStrangleWorker` | OTM1 CE + OTM1 PE | Time-based dual-leg BUY with momentum re-entry. |

For a basket, "flat" means **every** leg is closed. This matters most for the SL
Hunting BankNIFTY mirror, whose post-exit cooldown only starts once the whole
basket is confirmed flat — a lone surviving leg must not run the timer down.

---

## 7. Coexistence

`CPR`, `CPRAlgo3`, `Regime Adaptive` and `CPR AI` are independent strategies
that may run together with independent positions and independent P&L. They are
not variants of one another and none disables another.

Regime Adaptive's two candidate rules live in
`Signal Generators/Regime Adaptive Strategy/regime_candidates.py` as library
code with **no worker of their own**, specifically so the router and a candidate
can never take the same signal twice. See [`regime-adaptive.md`](regime-adaptive.md).

---

## 8. Testing

| Suite | Covers |
|---|---|
| `Tests/test_nifty_multi_strategy_master.py` | Worker construction, the loop, entry/exit routing, risk checks, factory output |
| `Tests/Signal Generators/test_deterministic_strategy_safety.py` | Shared safety properties across the deterministic strategies |
| `Tests/Signal Generators/test_trading_bot_ports.py` | The 13 ported signal modules |
| `Tests/Signal Generators/test_renko_bounds.py` | Renko brick bounds |
| `Tests/Signal Generators/CPR Strategy/`, `Subhamoy Strategies/` | Per-family signal logic |

---

## 9. Adding a strategy — checklist

1. Signal logic in `Signal Generators/` (pure function over a candle frame).
2. Either a spec row in `_SIGNAL_GEN_WORKER_SPECS`, or a worker subclass plus a
   `STRATEGY_ENV_PREFIX` entry.
3. `<PREFIX>_*` keys in `Dependencies/env.example` — CI enforces this.
4. Size knobs read through `_scaled_int` / `_scaled_float`, never the raw
   `_env_*` helpers. A drift-guard test fails otherwise.
5. Tests under `Tests/`, mirroring where the code lives.
6. Update this document if the worker introduces a new *shape* of strategy.
