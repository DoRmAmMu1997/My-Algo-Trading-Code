# LLD — CPR Algo 4 (deterministic "Intraday SRSI VWAP")

**Owns:** `Signal Generators/CPR Strategy/cpr_algo4_signal_generator.py`, `CPRAlgo4StrategyWorker`
in the master, `My Backtest Files (For Reference)/cpr_algo4_backtest.py`
**Why a separate worker:** [ADR-0018](../adr/0018-cpr-algo4-as-separate-worker.md)
**Sibling:** [`cpr-codex-ai-agent.md`](cpr-codex-ai-agent.md), which trades the same playbook with a
model making the regime judgment

---

## 1. Responsibility

This worker trades the operator's "Intraday SRSI VWAP" document with **fixed rules only**:

```
 09:25 5-min close vs zone [MIN(S1,PDL), MAX(R1,PDH)]   (previous-session levels)
   │
   ├─ inside (edges count as inside) ──► SIDEWAYS: Stochastic RSI reversals
   ├─ above ──► TRENDING UP   (calls)  ─┐ direction flips on
   ├─ below ──► TRENDING DOWN (puts)   ─┘ LH+LL / HH+HL structure
   └─ 09:25 bar missing / levels missing ──► NO TRADE today
```

The day type is fixed at 09:30. Every signal **buys** the ATM CE (bullish) or ATM PE (bearish) of the
next-next expiry through the shared `enter_position` path.

---

## 2. Structure

| Piece | Role |
|---|---|
| `build_cpr_algo4_frame` | `cpr_strategy_logic.build_cpr_with_indicators` plus TradingView Stochastic RSI (14/14/3/3), computed over the whole multi-day history so it is warm at the open. |
| `CPRAlgo4Engine` | Per-session state machine: day type, trend direction, confirmed swings, the reversal sequence, the first-30-minute range. It is fed **every** completed bar in order and resets on a new date. |
| `build_trade_plan` | The doc's risk rules; derives every price of a trade. |
| `check_intrabar_exit` | A pure stop → target → final-level check. The worker passes spot; the backtest passes the bar's range. |
| `CPRAlgo4StrategyWorker` | Executes the engine's answers and owns contracts, fills, the add leg and the ledger. |
| `cpr_algo4_backtest.py` | A bar-by-bar replay of the same engine, in spot points. |

---

## 3. Rules

### 3.1 SIDEWAYS
- **Entry (LONG):** %K crosses above %D with `max(prev_k, k) <= 20`. The SHORT entry is the mirror:
  a cross down with `min >= 80`.
- **Stop:** the latest *confirmed* swing low (high) of **today's** session. A swing is a strict
  fractal, 2 bars each side, confirmed 2 bars later. If no swing exists yet, there is no trade.
- **Premise exit:** a Stochastic RSI cross back inside the *opposite* zone (`CPR_ALGO4_SRSI_EXIT`).
- **Target 4 (opt-in, `CPR_ALGO4_FIRST30_TARGET`):** the first-30-minute high (low) becomes a
  booking level when it lies ahead of entry: in TARGET mode alongside 1:1, in TRAIL mode as the
  only fixed target. The doc places this under the sideways section only.
- The RSI/EMA filters do **not** apply: an oversold SRSI buy almost never has RSI > 45 and rising
  EMAs. CPR AI makes the same call.

### 3.2 TRENDING
- **Continuation**, in the day's direction:
  - UP: the previous bar closed below VWAP, and this bar closes above it with ≥ 40% of its **body**
    above VWAP. A doji never qualifies.
  - The doc's alternative also counts: a red candle supported on VWAP (`low <= VWAP < close`)
    followed by a green one supported on VWAP.
  - DOWN mirrors both patterns.
- **Filters:** RSI > 45 (calls) / < 65 (puts), and EMA5 above (below) EMA20 with both slopes
  pointing the trade's way.
- **Stop:** the entry candle's low (high).
- **Structure flip:** needs BOTH the last two confirmed highs AND lows to agree. LH + LL turns UP
  into DOWN; HH + HL turns DOWN into UP. An open trade is cut (`CPR_ALGO4_STRUCTURE_FLIP`), and
  `reversal_pending` is set.
- **Reversal sequence** (required while `reversal_pending`, shown for a new DOWN; steps are ordered,
  not necessarily consecutive, and never on the flip bar):
  1. a close below VWAP;
  2. a green close above VWAP;
  3. a red close below VWAP with ≥ 40% of the body below → buy PE.

  A reversal *fill* clears `reversal_pending`, and continuation setups resume.

### 3.3 Every trade
- **Risk:** skip if the stop is 0 or more than 30 points from entry.
- **Next level:** the ladder is S2, S1, PDL, CPR-low, pivot, CPR-high, PDH, R1, R2. Each level is
  cut 2 points short, and a level already inside that buffer counts as reached. Skip if the next one
  is nearer than 1:1.
- **Final level:** R2 − 2 (S2 + 2) always books. Skip the trade if it is not ahead of entry.
- **Entry cutoff:** no entry or add on a candle that ends at or after 15:00. The 15:15 square-off
  closes everything.
- **One position at a time,** and no new entry on the bar during which an exit happened.

### 3.4 Exit modes (`CPR_ALGO4_EXIT_MODE`)

| Mode | Behaviour |
|---|---|
| `TARGET` (default) | Book at the earliest of 1R, the buffered next level, and target 4 if enabled. |
| `TRAIL` | Normal trades: at the first milestone (the earlier of 1R and the next level) the stop moves to breakeven and the trail arms. Reversal trades: breakeven first, then a stop locked at 1R once the close reaches the earlier of 2R and the following level; the trail arms there. Once armed, a close below the previous candle's **low** (above its high for puts) exits. |

**Worth knowing.** The 1:1 entry rule guarantees the next level is at least 1R away, so "1:1 OR the
next level, whichever comes first" always books at **1R**. The next-level rule acts as an entry
filter. The TRAIL first milestone is likewise effectively 1R. Stops only ever ratchet toward profit.

### 3.5 The R1 add (`CPR_ALGO4_SCALE_IN_ENABLED`)
- **Pattern:** a red candle touching R1 (±2), then a green candle closing at or above R1 with its
  low ≥ R1 − 2.
- **Conditions:** trending **longs** only, once per trade, before the cutoff.
- **Execution:** equal to the initial filled quantity in the locked contract, and it must pass the
  spread and liquidity gates.

---

## 4. Worker execution model

- **Completed bars only.** The forming minute is dropped before resampling, because candles carry
  their start minute.
- **Catch-up replay.** Every unseen bar of today's session is fed to the engine in order. Exposure is
  opened or added only on the **newest** bar; EXIT decisions are honoured on any bar.
- **Per-poll safety.** The worker uses the shared run loop and overrides its `poll_safety_checks()`
  hook (a no-op for other workers), so `_check_spot_boundaries()` checks the spot stop, the fixed
  target and the final level on every poll. It also retries a one-shot premise exit whose live close
  did not confirm flat.
- **Frame cost.** The CPR builder takes ~0.7 s on a 7-day snapshot, so the built frame is cached on
  (row count, newest completed minute) and rebuilt at most once a minute, not on every poll.
- **Reporting.** The R1 add is reported as its own `add_pos` slot through `_owned_open_positions()`,
  so the dashboard and the crash-durable snapshot see the full open quantity.
- **Add-leg ledger.** This is a standalone copy of the CPR AI worker's mechanics, so a fix to one copy
  must be mirrored in the other:
  - the add is a separate role-A ledger leg;
  - live marks the add used *before* submission, so it is never retried after PARTIAL/UNKNOWN;
  - max-loss counts the add at the ledger's conservative `risk_quantity`;
  - the two-leg exit keeps local state until both legs are broker-confirmed flat.

  Without an add, exits use the shared single-leg path.
- **Restart.** The engine rebuilds today's state by replaying the frame. Positions are not resumed.

---

## 5. Configuration

| Key | Default | Notes |
|---|---|---|
| `CPR_ALGO4_LOTS` / `_MAX_LOSS` / `_SIZE_MULTIPLIER` | 1 / 5500 / 1 | These are size knobs, scaled together. |
| `CPR_ALGO4_TRADING_START_HOUR/_MINUTE` | 09:30 | Processing starts here; earlier bars still count for the day type and swings. |
| `CPR_ALGO4_ENTRY_CUTOFF_HOUR/_MINUTE` | 15:00 | Live requires start < cutoff ≤ square-off. An impossible value runs paper on 15:00 and blocks live. |
| `CPR_ALGO4_SQUARE_OFF_HOUR/_MINUTE` | 15:15 | |
| `CPR_ALGO4_EXIT_MODE` | TARGET | Any value other than TARGET/TRAIL runs paper as TARGET and blocks live. |
| `CPR_ALGO4_FIRST30_TARGET` | false | Target 4, sideways only; a booking level in either exit mode. |
| `CPR_ALGO4_SCALE_IN_ENABLED` | true | The one R1 add. |
| `CPR_ALGO4_MAX_SPREAD_PCT` / `_MIN_LIQUIDITY_SCORE` | 0 / 0 | Shared market-quality gates; they also gate the add. |
| `CPR_ALGO4_VIRTUAL_TRADING` / `_LIVE_TRADING` | true / false | The standard double gate. |

The playbook's fixed numbers (30-point stop, 2-point buffer, 0.40 body fraction, RSI 45/65,
Stochastic RSI 14/14/3/3 and 20/80, 2-bar fractals) are `CPRAlgo4Config` fields. The backtest can
vary them; they are not `.env` knobs.

---

## 6. Backtest

```
python algo.py backtest --strategy cpr-algo4 --exit-mode TARGET|TRAIL [--first30-target] [--no-scale-in] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
```

- **P&L unit:** spot points, because it measures signal quality, not option premium.
- **Fills:**
  - Entries, adds and completed-bar exits fill at the bar close.
  - On a candle touching both the stop and a target, the stop wins.
  - A gap through a level fills at the open.
  - Square-off happens at the close of the bar ending 15:15.
- **Outputs:** `Backtest Outputs/<dataset>_cpr_algo4_<variant>_{trades.csv,daily.csv,summary.txt,backtest.log}`.

---

## 7. Testing

| Suite | Covers |
|---|---|
| `Tests/Signal Generators/CPR Strategy/test_cpr_algo4_signal_generator.py` | Stochastic RSI maths, no look-ahead in the frame, config validation, regime edges, trade-plan geometry, every entry/exit rule, trailing stages, the add, intrabar ordering |
| `Tests/test_nifty_multi_strategy_master.py::TestCPRAlgo4StrategyWorker` | Real-engine entry through the ATM path, catch-up replay, per-poll exits, forming-minute exclusion, the paper/live add, two-leg exit retention, max-loss with an add, live-config refusals |
