# Signal Generator folder
This folder contains the signal generators which will be imported into the main front test file

# What is a signal generator? 
Signal generator expects the OHLC data DataFrame as an argument(which will be provided by the main front test file) and works on the data to generate a bullish or a bearish signal

# The coding itself?
- Claude Opus 4.7 Max: Generated Donchian Signal Generator Bearish.py and Supertrend Signal Generator Bullish.py
- GPT-5.4-xhigh: Generated ema_trend_strategy_logic.py, heikin_ashi_strategy_logic.py, profit_shooter_strategy_logic.py, renko_strategy_logic.py and renko_strategy_logic_9_21.py
- GPT-5.5-xhigh: Generated the CPR Strategy folder with shared CPR logic, Algo 1, Algo 2, and combined signal-generator wrappers
- GPT-5.5-xhigh: Generated the Subhamoy Strategies folder with Goldmine and Money Machine shared engines and NIFTY wrappers
- Claude Opus 4.8 Max: Ported 13 strategies from the public TradingBot project (the `Nifty * Signal Generator.py` files listed below) plus the shared `misc_strategy_common.py`, and wired them into the front-test master
- Claude Opus 4.8 Max: Built the **SL Hunting AI Agent** (`SL Hunting AI Agent/`) — an LLM-driven strategy (a Claude agent), unlike the deterministic generators above (see its own README)
- Codex: Built the independent, opt-in **CPR Codex AI Agent** (`CPR AI Agent/`) — a five-minute SRSI/VWAP worker whose host owns every mechanical risk and execution gate

# Where each generator is used
| File | Shape | Used by |
|---|---|---|
| `CPR Strategy/cpr_strategy_logic.py` | Stateful CPR engine with CPR levels, Algo 1, Algo 2, and RSI divergence | CPR backtest + future front-test integration |
| `CPR Strategy/cpr_algo1_signal_generator.py` | Algo 1 trend-only CPR wrapper | CPR trend-only callers |
| `CPR Strategy/cpr_algo2_signal_generator.py` | Algo 2 sideways/reversal CPR wrapper | CPR sideways/reversal callers |
| `CPR Strategy/cpr_combined_signal_generator.py` | Full CPR PDF strategy wrapper (Algo 1 + Algo 2, single-chart) | CPR backtest + future front-test integration |
| `CPR Strategy/Nifty CPR Algo 3 Signal Generator.py` | Multi-instrument CPR Algo 3 (spot + ITM CE + ITM PE); takes three frames, returns a `CPRDecision` | front-test master — the `CPRAlgo3StrategyWorker` fetches the ITM CE/PE feeds on demand |
| `CPR AI Agent/` | Frozen five-minute context, four no-argument tools, Codex judgment, and host-owned risk/execution policy | independently opt-in `CPRAIWorker` in the front-test master |
| `Subhamoy Strategies/goldmine_strategy_logic.py` | Stateful Goldmine pullback/engulfing engine | Goldmine backtest + future front-test integration |
| `Subhamoy Strategies/money_machine_strategy_logic.py` | Stateful Money Machine compression/Hulk engine | Money Machine backtest + future front-test integration |
| `Subhamoy Strategies/goldmine_signal_generator.py` | Thin NIFTY Goldmine wrapper | Goldmine callers that prefer wrapper functions |
| `Subhamoy Strategies/money_machine_signal_generator.py` | Thin NIFTY Money Machine wrapper | Money Machine callers that prefer wrapper functions |
| `Donchian Signal Generator Bearish.py` | DataFrame in -> DataFrame with signal columns out (stateless) | front-test master |
| `Supertrend Signal Generator Bullish.py` | DataFrame in -> DataFrame with signal columns out (stateless) | front-test master |
| `ema_trend_strategy_logic.py` | Stateful signal engine (class) | EMA backtest + front-test master |
| `heikin_ashi_strategy_logic.py` | Stateful signal engine (class) | front-test master |
| `Subhamoy Strategies/profit_shooter_strategy_logic.py` | Stateful signal engine (class) | Profit Shooter backtest + front-test master |
| `renko_strategy_logic.py` | Stateful Renko engine — 5/21/44 EMA variant | original Renko logic (kept for reference) |
| `renko_strategy_logic_9_21.py` | Stateful Renko engine — 9/21 EMA variant | Renko backtest + front-test master |

# TradingBot strategy ports (13, ATM single-leg)
Thirteen strategies ported from the public TradingBot project, kept flat in this
folder. Each is self-contained (frozen `Config` + `PositionContext` + `Decision`
dataclasses, a `build_*_with_indicators()`, a stateful `*SignalEngine`, and a
`*SignalGenerator`) and shares `misc_strategy_common.py` for its indicators
(the mandatory TA-Lib 0.6.8 backend). All are wired into the front-test master via the
shared `_build_signal_gen_worker_class` factory as ATM single-leg workers, each
independently tunable from `.env` by its own prefix (e.g. `SMA_CROSSOVER_*`).

| File | Strategy idea |
|---|---|
| `sma_crossover_signal_generator.py` | fast/slow SMA crossover |
| `bollinger_bands_signal_generator.py` | bounce off a band (mean reversion) |
| `keltner_squeeze_signal_generator.py` | BB-inside-KC squeeze release + MACD sign |
| `mean_reversion_zscore_signal_generator.py` | fade z-score extremes back to the mean |
| `ml_ensemble_signal_generator.py` | RandomForest P(up) — **requires scikit-learn** |
| `multi_timeframe_signal_generator.py` | trend SMA + EMA crossover + RSI band |
| `opening_range_breakout_signal_generator.py` | close breaks open +/- ATR |
| `parabolic_sar_signal_generator.py` | SAR flip filtered by ADX |
| `rsi_divergence_signal_generator.py` | price vs RSI swing divergence |
| `rsi_reversal_signal_generator.py` | oversold/overbought reversal |
| `stochastic_oscillator_signal_generator.py` | %K/%D cross in zone, trend-filtered |
| `supertrend_signal_generator.py` | ATR-band Supertrend flip |
| `volatility_breakout_signal_generator.py` | Larry Williams prev-range breakout |
| `misc_strategy_common.py` | shared indicators used by all 13 (SMA, EMA, RSI, MACD, Bollinger, Keltner, Stochastic, ADX, Parabolic SAR, Supertrend, z-score, swing detection) |

# Regime Adaptive port (`Regime Adaptive Strategy/`) — different source project
One more ATM single-leg worker, adapted from the MIT-licensed
[`workratananmol-hub/nifty-options-paper-trading-bot`](https://github.com/workratananmol-hub/nifty-options-paper-trading-bot).
Same factory, same `.env` prefix convention (`REGIME_ADAPTIVE_*`), same execution
family — but it is a **router**, not a single rule. It reads ADX each bar and
switches which rule applies:

- ADX missing → **no trade** (it refuses to guess the regime)
- ADX ≥ `REGIME_ADAPTIVE_ADX_TREND_THRESHOLD` → opening-range breakout, confirmed by VWAP
- ADX below it → fade an extension away from VWAP

Everything for it lives in its own folder, `Regime Adaptive Strategy/`:

| File | Role |
|---|---|
| `Nifty Regime Adaptive Signal Generator.py` | the router — the only worker of the three |
| `regime_candidates.py` | the two candidate rules, as pure column-producing functions |
| `regime_common.py` | session date, session VWAP, session opening range — and this folder's **only** `sys.path` bootstrap, which is why it re-exports the shared indicators from `misc_strategy_common` one level up |
| `REGIME_PORTING_NOTES.md` | **read before enabling live** — what was dropped and why |
| _(tests)_ | at the mirrored path `Tests/Signal Generators/Regime Adaptive Strategy/`, whose `conftest.py` is the pytest equivalent of that bootstrap (same pattern as `SL Hunting AI Agent/`) |

Two things to know before touching it:

1. **The candidates are library code, deliberately.** They expose no
   `Config`/`Engine`/`PositionContext`, have no env prefix and no P&L row, and are
   absent from `test_trading_bot_ports.py`'s `PORTS` table. If either were also a
   worker, it and the router could take the *same signal in the same session* —
   real double size that the Google Sheet would not reveal.
2. **VWAP here is a proxy.** The live feed carries no volume, so `vwap` is an
   equal-weight session mean unless a `volume` column is present (backtests). Both
   rules are VWAP-centric, so this is a genuine fidelity gap that no test catches;
   every bar carries `vwap_is_proxy` to record which was used. Paper only until it
   has several clean sessions.

# CPR Codex AI Agent (`CPR AI Agent/`) — independent, opt-in worker
This is not another deterministic CPR wrapper. `CPRAIWorker` freezes completed
five-minute SRSI/VWAP context behind four no-argument tools, asks Codex for a
regime/setup or premise-exit judgment, and then applies host-owned entry, sizing,
time, lifecycle, and execution gates. It is disabled by default and live-disabled
by default. Ordinary CPR, CPR Algo 3, Regime Adaptive, SL Hunting, and CPR AI have
independent prefixes, workers, positions, and P&L and may coexist when their own
enable and virtual-trading gates permit it. See `CPR AI Agent/README.md` for the
isolation boundary and order-free synthetic smoke command.

# SL Hunting AI Agent (`SL Hunting AI Agent/`) — LLM-driven, a different kind
Unlike everything else in this folder (deterministic "DataFrame in → signal out" transforms,
or stateful engines that compute a signal from a formula), the **SL Hunting AI Agent** is an
**LLM trader**. A Claude agent — via [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/)
on your Claude subscription (no API key) — reasons over the recent NIFTY chart each completed
bar and acts through **tool calls**, rather than returning a computed signal. It trades the
discretionary *SL Hunting* price-action method on NIFTY ATM options, with **BankNIFTY
cross-confirmation** and dynamic **~₹2,500 risk-per-trade** sizing. Every NIFTY entry is also
mirrored with an **equal-lot BankNIFTY ATM** leg (`SL_HUNTING_BNF_MIRROR`) — tied for hard risk,
but cut per-leg on premise-invalidation (see its README). It is wired into the
front-test master as an **independently opt-in agent** (`SL_HUNTING_ENABLED`, off by default;
paper unless explicitly enabled; **fail-soft** — a safe HOLD on any error). It also **learns
from its own trades** (a per-trade journal → an off-loop reflection coach → human-gated lessons
injected into its prompt) and writes a **per-bar decision log**. The agent has its own subfolder
— deterministic detectors, an in-process MCP tool server, strict-Pydantic output validation, and
a pytest suite — fully documented (design, setup via `claude setup-token`, safety model, the
learning loop) in **`SL Hunting AI Agent/README.md`**.

# Two flavors of "signal generator" in this folder
- The **Donchian / Supertrend** files are pure transformations: pass a DataFrame in, get one back with extra signal columns. Stateless.
- The **`*_strategy_logic.py`** files are stateful engines: create one engine object, then call `evaluate_candle(...)` per new bar. They track entries, exits, and re-entries internally. The backtests in `My Backtest Files (For Reference)/` each use one of these.

# `renko_strategy_logic.py` vs `renko_strategy_logic_9_21.py`
- `renko_strategy_logic.py` — original 5/21/44 EMA variant.
- `renko_strategy_logic_9_21.py` — 9/21 EMA variant. Same public class/function names as the original so callers can swap imports without other code changes. This is the one the Renko backtest and front-test master currently use.
