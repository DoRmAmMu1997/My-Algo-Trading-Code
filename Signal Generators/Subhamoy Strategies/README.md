# Subhamoy Strategies

This folder groups strategy files from Subhamoy-style setups.

Included strategies in this repo folder:
- Goldmine
- Money Machine
- Profit Shooter

Profit Shooter's shared engine (`profit_shooter_strategy_logic.py`) now lives in
this folder alongside Goldmine and Money Machine. Its backtest remains at
`My Backtest Files (For Reference)/profit_shooter_backtest.py`.

## Signal Generators

- `goldmine_strategy_logic.py`
- `money_machine_strategy_logic.py`
- `profit_shooter_strategy_logic.py`
- `goldmine_signal_generator.py`
- `money_machine_signal_generator.py`

The Goldmine and Money Machine modules expect already-prepared 5-minute OHLC
data. They do not resample 1-minute data because that belongs in the front-test
or data-preparation file.

## Backtests

- `goldmine_strategy_backtest.py`
- `money_machine_strategy_backtest.py`

Run the new backtests with an explicit 5-minute CSV:

```powershell
python "My Backtest Files (For Reference)\Subhamoy Strategies\goldmine_strategy_backtest.py" --data "path\to\five_minute_data.csv" --dataset nifty
python "My Backtest Files (For Reference)\Subhamoy Strategies\money_machine_strategy_backtest.py" --data "path\to\five_minute_data.csv" --dataset nifty
```

Both backtests write logs, trades, daily equity, stats, and daily max-loss files
under `Backtest Outputs`.
