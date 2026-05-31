# My-Algo-Trading-Code
This contains all the code I have written for the signal generation and the front test where I fetch data using Dhan API

# The code
Although I own the code, the coding itself was done entirely using GPT-5.4-xhigh, GPT-5.5-xhigh and Claude Opus 4.7, Claude Opus 4.8 on Max effort. GPT wrote majority of the signal generators and the data fetch files. Claude wrote the big one - the multithreaded Front Test worker. I just did the reviews and the testing

# What is included?
- Data extractors which extract historical data for NIFTY/BANKNIFTY/FINNIFTY indices
- The backtest files I used to backtest
- The signal generators I created to generate signals
- The main front test file which uses miltithreading to execute all strategies together
- Live Telegram alerts: the front-test master file can post every entry/exit (option instrument, lot size, entry/exit price, and P&L) to a Telegram group/channel

# Recent additions
- **CPR (Central Pivot Range) strategy is now live in the front test.** It runs as an ATM single-leg worker (`CPRStrategyWorker`) alongside the other strategies: the master file feeds it 1-min OHLC, the CPR logic resamples to complete 5-min candles internally, and a LONG/SHORT signal buys the ATM CE/PE of the next-next expiry. Tunable via `CPR_*` knobs in the `.env` (lots, max-loss, poll, 09:25-15:15 window). This brings the master file to 9 workers running together.
- **Telegram trade notifications.** A queue-based `TelegramMessageWorker` posts a message to a Telegram group/channel on every entry and exit from *any* of the 9 workers. Each alert shows the strategy, the exact option instrument(s), lot size, entry and exit price, and P&L (hedged spreads show both legs). It runs on its own thread so Telegram latency or downtime never blocks the trading loop, and it's a cheap no-op when disabled. See Setup below to switch it on.

# Pro Tip
You might have to adjust the import addresses from which the files are to be imported because the files are in different directories in my local machine(fixed in the latest Claude commit)

# Repository structure
```
.
├── Nifty Multi Strategy Front Test - Master File.py   # multithreaded paper-trading runner
├── Data Extractors/                                   # 1m OHLC downloaders + shared helper
├── My Backtest Files (For Reference)/                 # backtesting.py-based backtests
└── Signal Generators/                                 # strategy / signal logic modules
```
Each subfolder has its own `Readme.md` with the details.

# Setup
1. Python 3.10+ (I'm running 3.13).
2. Install dependencies:
   ```
   pip install dhanhq pandas numpy backtesting python-dotenv
   ```
3. Set Dhan credentials as environment variables:
   ```
   DHAN_CLIENT_CODE=your_client_code
   DHAN_TOKEN_ID=your_access_token
   ```
   The backtests and the front-test master file additionally each load a local `.env` from their own folder for tunable strategy parameters.
4. (Optional) Turn on Telegram trade alerts by adding these to the master file's `.env`:
   ```
   TELEGRAM_ENABLED=true
   TELEGRAM_BOT_TOKEN=your_botfather_token
   TELEGRAM_CHAT_ID=@your_channel_or_-100xxxxxxxxxx
   ```
   Create the bot via @BotFather and add it to your group/channel as an admin. Leave `TELEGRAM_ENABLED=false` (the default) to run without alerts. The token stays in `.env`, which is git-ignored.

# Typical workflow
1. Pull historical data — e.g. `python "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py"`. The CSV lands in `Backtest Outputs/`.
2. Run a backtest against that CSV — e.g. `python "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py"`.
3. Once a strategy looks good, run `Nifty Multi Strategy Front Test - Master File.py` for paper-traded multi-strategy execution.

The `Backtest Outputs/` folder is `.gitignore`-d, so generated CSVs/logs stay local.
