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

# Where each generator is used
| File | Shape | Used by |
|---|---|---|
| `CPR Strategy/cpr_strategy_logic.py` | Stateful CPR engine with CPR levels, Algo 1, Algo 2, and RSI divergence | CPR backtest + future front-test integration |
| `CPR Strategy/Nifty CPR Algo 1 Signal Generator.py` | Algo 1 trend-only CPR wrapper | CPR trend-only callers |
| `CPR Strategy/Nifty CPR Algo 2 Signal Generator.py` | Algo 2 sideways/reversal CPR wrapper | CPR sideways/reversal callers |
| `CPR Strategy/Nifty CPR Combined Signal Generator.py` | Full CPR PDF strategy wrapper (Algo 1 + Algo 2, single-chart) | CPR backtest + future front-test integration |
| `CPR Strategy/Nifty CPR Algo 3 Signal Generator.py` | Multi-instrument CPR Algo 3 (spot + ITM CE + ITM PE); takes three frames, returns a `CPRDecision` | front-test master — the `CPRAlgo3StrategyWorker` fetches the ITM CE/PE feeds on demand |
| `Subhamoy Strategies/goldmine_strategy_logic.py` | Stateful Goldmine pullback/engulfing engine | Goldmine backtest + future front-test integration |
| `Subhamoy Strategies/money_machine_strategy_logic.py` | Stateful Money Machine compression/Hulk engine | Money Machine backtest + future front-test integration |
| `Subhamoy Strategies/Nifty Goldmine Signal Generator.py` | Thin NIFTY Goldmine wrapper | Goldmine callers that prefer wrapper functions |
| `Subhamoy Strategies/Nifty Money Machine Signal Generator.py` | Thin NIFTY Money Machine wrapper | Money Machine callers that prefer wrapper functions |
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
(TA-Lib first, pandas fallback). All are wired into the front-test master via the
shared `_build_signal_gen_worker_class` factory as ATM single-leg workers, each
independently tunable from `.env` by its own prefix (e.g. `SMA_CROSSOVER_*`).

| File | Strategy idea |
|---|---|
| `Nifty SMA Crossover Signal Generator.py` | fast/slow SMA crossover |
| `Nifty Bollinger Bands Signal Generator.py` | bounce off a band (mean reversion) |
| `Nifty Keltner Squeeze Signal Generator.py` | BB-inside-KC squeeze release + MACD sign |
| `Nifty Mean Reversion Zscore Signal Generator.py` | fade z-score extremes back to the mean |
| `Nifty ML Ensemble Signal Generator.py` | RandomForest P(up) — **requires scikit-learn** |
| `Nifty Multi Timeframe Signal Generator.py` | trend SMA + EMA crossover + RSI band |
| `Nifty Opening Range Breakout Signal Generator.py` | close breaks open +/- ATR |
| `Nifty Parabolic SAR Signal Generator.py` | SAR flip filtered by ADX |
| `Nifty RSI Divergence Signal Generator.py` | price vs RSI swing divergence |
| `Nifty RSI Reversal Signal Generator.py` | oversold/overbought reversal |
| `Nifty Stochastic Oscillator Signal Generator.py` | %K/%D cross in zone, trend-filtered |
| `Nifty Supertrend Signal Generator.py` | ATR-band Supertrend flip |
| `Nifty Volatility Breakout Signal Generator.py` | Larry Williams prev-range breakout |
| `misc_strategy_common.py` | shared indicators used by all 13 (SMA, EMA, RSI, MACD, Bollinger, Keltner, Stochastic, ADX, Parabolic SAR, Supertrend, z-score, swing detection) |

# SL Hunting AI Agent (`SL Hunting AI Agent/`) — LLM-driven, a different kind
Unlike everything else in this folder (deterministic "DataFrame in → signal out" transforms,
or stateful engines that compute a signal from a formula), the **SL Hunting AI Agent** is an
**LLM trader**. A Claude agent — via [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/)
on your Claude subscription (no API key) — reasons over the recent NIFTY chart each completed
bar and acts through **tool calls**, rather than returning a computed signal. It trades the
discretionary *SL Hunting* price-action method on NIFTY ATM options, with **BankNIFTY
cross-confirmation** and dynamic **~₹2,500 risk-per-trade** sizing, and is wired into the
front-test master as the **optional, opt-in 27th worker** (`SL_HUNTING_ENABLED`, off by default;
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
