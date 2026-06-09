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
- **End-of-day P&L is now written to a Google Sheet.** When all workers exit on a clean end of day, the master parses the run's log for each strategy's realised P&L and writes it into a tracker sheet — one row per strategy, one column per calendar day — overwriting today's cell and backfilling any blank earlier-this-month cells from the (append-mode) log. Auth is OAuth user-token via `gspread`; configure `GSHEET_ID` + an OAuth client in `.env` (see Setup). It's a safe no-op when unconfigured, so it never disturbs shutdown.
- **13 TradingBot signal-generator ports added — the front test now runs 24 workers.** Thirteen more ATM single-leg strategies were ported into `Signal Generators/` (SMA Crossover, Bollinger Bands, Keltner Squeeze, Mean Reversion Z-Score, ML Ensemble, Multi-Timeframe, Opening Range Breakout, Parabolic SAR, RSI Divergence, RSI Reversal, Stochastic, Supertrend, Volatility Breakout), all sharing `misc_strategy_common.py` (TA-Lib-first indicator helpers). They're wired into the master via one shared factory as `AtmSingleLegStrategyWorker`s — the same family as Renko/Goldmine/CPR — and each is fully tunable from `.env` by its own prefix (e.g. `SMA_CROSSOVER_*`, `KELTNER_SQUEEZE_*`). This brings the runner to **24 workers** (21 ATM single-leg + 2 Hedged Puts + 1 Delta-0.2). ML Ensemble needs `scikit-learn`.
- **CPR (Central Pivot Range) strategy is now live in the front test.** It runs as an ATM single-leg worker (`CPRStrategyWorker`) alongside the other strategies: the master file feeds it 1-min OHLC, the CPR logic resamples to complete 5-min candles internally, and a LONG/SHORT signal buys the ATM CE/PE of the next-next expiry. Tunable via `CPR_*` knobs in the `.env` (lots, max-loss, poll, 09:25-15:15 window). (This brought the master file to nine workers at the time; see the latest addition at the top of this list for the current total.)
- **Telegram trade notifications.** A queue-based `TelegramMessageWorker` posts a message to a Telegram group/channel on every entry and exit from *any* worker. Each alert shows the strategy, the exact option instrument(s), lot size, entry and exit price, and P&L (hedged spreads show both legs). It runs on its own thread so Telegram latency or downtime never blocks the trading loop, and it's a cheap no-op when disabled. See Setup below to switch it on.

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
   pip install dhanhq pandas numpy backtesting python-dotenv gspread
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

5. (Optional) End-of-day P&L to Google Sheets. After all workers exit, the master writes each strategy's day-end P&L into a tracker sheet (one row per strategy, one column per day, with month backfill). Enable it by adding to the master's `.env`:
   ```
   GSHEET_ID=your_spreadsheet_id
   GSHEET_OAUTH_CLIENT_FILE=Dependencies/gsheet_oauth_client.json
   GSHEET_OAUTH_TOKEN_FILE=Dependencies/gsheet_oauth_token.json
   ```
   Auth is OAuth user-token via `gspread`: in Google Cloud enable the Sheets API, create an OAuth client of type **Desktop app**, download its JSON to `GSHEET_OAUTH_CLIENT_FILE`, and share the sheet with your Google account. The first run opens a browser once for consent and caches a token at `GSHEET_OAUTH_TOKEN_FILE`. The sheet needs one row per strategy in column A with exact labels (e.g. `Renko Strategy`, `SMA Crossover Strategy`, `Stochastic Oscillator Strategy`, `Supertrend Strategy`, …); unmatched strategies are skipped with a warning. Leave `GSHEET_ID` blank to disable (safe no-op).

# Typical workflow
1. Pull historical data — e.g. `python "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py"`. The CSV lands in `Backtest Outputs/`.
2. Run a backtest against that CSV — e.g. `python "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py"`.
3. Once a strategy looks good, run `Nifty Multi Strategy Front Test - Master File.py` for paper-traded multi-strategy execution.

The `Backtest Outputs/` folder is `.gitignore`-d, so generated CSVs/logs stay local.
