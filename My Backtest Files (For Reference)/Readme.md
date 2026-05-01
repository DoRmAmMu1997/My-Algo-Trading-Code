# How do backtest files work
You first extract data by using the data extractors. Then, this file uses the backtesting.py framework which accepts the extracted data as an argument and runs the backtests according to your imported signal generator file

# What is backtesting.py?
Backtesting.py is a lightweight, fast, and user-friendly Python framework for backtesting trading strategies on historical data. Built on top of Pandas, NumPy, and Bokeh, it offers a clean high-level API, vectorized and event-driven backtesting, a built-in SAMBO optimizer for testing many strategy variants, and interactive visualizations. It is library-agnostic for technical indicators (works with TA-Lib, Tulip, pandas-ta, etc.) and supports any financial instrument with historical candlestick data (forex, crypto, stocks, futures).
Link: https://kernc.github.io/backtesting.py/
