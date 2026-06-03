# Subhamoy Strategies

This folder groups strategy files from Subhamoy-style setups.

Included strategies in this repo folder:
- Goldmine
- Money Machine

Profit Shooter already exists at `Signal Generators/profit_shooter_strategy_logic.py`
and `My Backtest Files (For Reference)/profit_shooter_backtest.py`, so this folder
contains only the new Subhamoy signal-generator files added by this change.

## Signal Generators

- `goldmine_strategy_logic.py`
- `money_machine_strategy_logic.py`
- `Nifty Goldmine Signal Generator.py`
- `Nifty Money Machine Signal Generator.py`

The Goldmine and Money Machine modules expect already-prepared 5-minute OHLC
data. They do not resample 1-minute data because that belongs in the front-test
or data-preparation file.

## Backtests

- `Nifty Goldmine Strategy Backtest.py`
- `Nifty Money Machine Strategy Backtest.py`

Run the new backtests with an explicit 5-minute CSV:

```powershell
python "My Backtest Files (For Reference)\Subhamoy Strategies\Nifty Goldmine Strategy Backtest.py" --data "path\to\five_minute_data.csv" --dataset nifty
python "My Backtest Files (For Reference)\Subhamoy Strategies\Nifty Money Machine Strategy Backtest.py" --data "path\to\five_minute_data.csv" --dataset nifty
```

Both backtests write logs, trades, daily equity, stats, and daily max-loss files
under `Backtest Outputs`.
