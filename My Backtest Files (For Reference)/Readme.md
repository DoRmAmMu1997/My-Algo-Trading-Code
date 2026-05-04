# How do backtest files work
You first extract data by using the data extractors. Then, this file uses the backtesting.py framework which accepts the extracted data as an argument and runs the backtests according to your imported signal generator file

# What is backtesting.py?
Backtesting.py is a lightweight, fast, and user-friendly Python framework for backtesting trading strategies on historical data. Built on top of Pandas, NumPy, and Bokeh, it offers a clean high-level API, vectorized and event-driven backtesting, a built-in SAMBO optimizer for testing many strategy variants, and interactive visualizations. It is library-agnostic for technical indicators (works with TA-Lib, Tulip, pandas-ta, etc.) and supports any financial instrument with historical candlestick data (forex, crypto, stocks, futures).
Link: https://kernc.github.io/backtesting.py/

# Files in this folder
| File | Strategy | Imports from `Signal Generators/` |
|---|---|---|
| `Nifty EMA Trend Strategy Backtest.py` | 4/11/18 EMA trend, ADX-filtered | `ema_trend_strategy_logic.py` |
| `Nifty Heiken Ashi Futures 5Y Backtest.py` | Heikin Ashi + Bollinger Bands | (self-contained) |
| `Nifty Renko Strategy Backtest.py` | Renko + 9/21 EMA | `renko_strategy_logic_9_21.py` |
| `profit_shooter_backtest.py` | Profit Shooter — supports NIFTY/BANKNIFTY/FINNIFTY via `--dataset` | `profit_shooter_strategy_logic.py` |

# How to run
```
python "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py"
```
By default each backtest reads `<repo_root>/Backtest Outputs/nifty_renko_futures_5y_1min_data.csv`. Override with `--data <path>`. The profit-shooter file additionally supports `--dataset nifty|banknifty|finnifty`.

# Where outputs land
In `<repo_root>/Backtest Outputs/`:
- `<strategy>_5y_trades.csv` — every closed trade
- `<strategy>_5y_daily_equity.csv` — equity curve sampled daily
- `<strategy>_5y_stats.txt` — summary statistics
- `<strategy>_5y_backtest.log` — run log

# Strategy parameter tuning
EMA Trend, Renko, and Profit Shooter each load a `.env` from this folder for tunable thresholds (`EMA_TREND_FAST_PERIOD`, `BROKERAGE`, etc.). Each backtest documents the env var names it reads near the top.
