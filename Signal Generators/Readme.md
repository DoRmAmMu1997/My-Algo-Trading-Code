# Signal Generator folder
This folder contains the signal generators which will be imported into the main front test file

# What is a signal generator? 
Signal generator expects the OHLC data DataFrame as an argument(which will be provided by the main front test file) and works on the data to generate a bullish or a bearish signal

# The coding itself?
- Claude Opus 4.7 Max: Generated Donchian Signal Generator Bearish.py and Supertrend Signal Generator Bullish.py
- GPT-5.4-xhigh: Generated ema_trend_strategy_logic.py, heikin_ashi_strategy_logic.py, profit_shooter_strategy_logic.py, renko_strategy_logic.py and renko_strategy_logic_9_21.py

# Where each generator is used
| File | Shape | Used by |
|---|---|---|
| `Donchian Signal Generator Bearish.py` | DataFrame in -> DataFrame with signal columns out (stateless) | front-test master |
| `Supertrend Signal Generator Bullish.py` | DataFrame in -> DataFrame with signal columns out (stateless) | front-test master |
| `ema_trend_strategy_logic.py` | Stateful signal engine (class) | EMA backtest + front-test master |
| `heikin_ashi_strategy_logic.py` | Stateful signal engine (class) | front-test master |
| `profit_shooter_strategy_logic.py` | Stateful signal engine (class) | Profit Shooter backtest + front-test master |
| `renko_strategy_logic.py` | Stateful Renko engine — 5/21/44 EMA variant | original Renko logic (kept for reference) |
| `renko_strategy_logic_9_21.py` | Stateful Renko engine — 9/21 EMA variant | Renko backtest + front-test master |

# Two flavors of "signal generator" in this folder
- The **Donchian / Supertrend** files are pure transformations: pass a DataFrame in, get one back with extra signal columns. Stateless.
- The **`*_strategy_logic.py`** files are stateful engines: create one engine object, then call `evaluate_candle(...)` per new bar. They track entries, exits, and re-entries internally. The backtests in `My Backtest Files (For Reference)/` each use one of these.

# `renko_strategy_logic.py` vs `renko_strategy_logic_9_21.py`
- `renko_strategy_logic.py` — original 5/21/44 EMA variant.
- `renko_strategy_logic_9_21.py` — 9/21 EMA variant. Same public class/function names as the original so callers can swap imports without other code changes. This is the one the Renko backtest and front-test master currently use.
