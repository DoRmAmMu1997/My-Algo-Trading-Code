# My-Algo-Trading-Code
This contains all the code I have written for the signal generation and the front test where I fetch data using Dhan API

# Live track record
I have been running these strategies **live** (real broker orders) since **May 2026**. The day-by-day results are recorded here:

📈 **[Live results spreadsheet](https://docs.google.com/spreadsheets/d/1y4VgThcLywZbOibKC_pyKbh0A5u1xtgL_cZvyHp3FYg/edit?gid=1960070915#gid=1960070915)**

# The code
Although I own the code, the coding itself was done entirely using GPT-5.4-xhigh, GPT-5.5-xhigh and Claude Opus 4.7, Claude Opus 4.8 on Max effort. GPT wrote majority of the signal generators and the data fetch files. Claude wrote the big one - the multithreaded Front Test worker. I just did the reviews and the testing

# What is included?
- Data extractors which extract historical data for NIFTY/BANKNIFTY/FINNIFTY indices
- The backtest files I used to backtest
- The signal generators I created to generate signals
- The main front test file which uses miltithreading to execute all strategies together
- Live order execution to a real broker — selectable among **Kotak Neo**, **Shoonya (Finvasia)**, **Flattrade Pi v2**, and **Dhan** — gated by a global kill-switch and per-strategy paper/live toggles (everything defaults to paper)
- Live Telegram alerts: the front-test master file can post every entry/exit (option instrument, lot size, entry/exit price, and P&L) to a Telegram group/channel
- An **optional, opt-in LLM trading agent** — the "SL Hunting AI Agent" — a Claude agent that trades a discretionary price-action method on NIFTY options; off by default, paper unless explicitly enabled, and fail-soft (see Recent additions)

# Recent additions
- **Per-strategy size multiplier — `<PREFIX>_SIZE_MULTIPLIER`.** One knob per strategy (default **1**, whole numbers up to **25**) that scales that strategy's entire size/risk set together: its lot count, its per-trade risk budget, its hard lot cap, and its daily max-loss kill-switch. At `2` a 5-lot cap becomes 10, a Rs.2,500 budget becomes Rs.5,000, a Rs.5,500 daily cap becomes Rs.11,000, and a setup that would have taken 4 lots takes 8 — so position size can grow with the **account** by editing one number instead of four that must be kept consistent by hand. Deliberately **per-strategy only** (no global switch, so one typo cannot enlarge all 27 workers) and it applies to **paper and live alike**, so an enlarged size can be paper-validated first. Anything malformed (`0`, `2.5`, `30`, `"two"`) falls back to 1 for paper and **blocks that strategy from live trading** rather than guessing a size. Leave every multiplier unset to trade exactly as before. Two consequences worth knowing: the scaled budget also loosens the "one lot exceeds the budget" skip, and because lots are floored a 2x can land slightly above a pure doubling (still strictly inside the scaled budget). For SL Hunting, note its BankNIFTY mirror already roughly doubles basket risk, so a multiplier of M leaves the basket near 2xM times the single-leg budget.
- **Optional websocket market data producer — `MARKET_DATA_SOURCE=WEBSOCKET`.** The runner can now source its market data from Dhan's marketfeed websocket instead of REST polling (requires Dhan's paid Data API subscription; any other value of the flag fails closed to the default REST poller). Ticks build the 1-minute candles live — the forming minute updates in real time — and once per minute the completed candles are trued-up against Dhan's official REST candles (official wins; divergence is logged), so strategy bars converge to exactly what the backtests used. All LTPs (NIFTY spot plus every subscribed option leg, including multi-leg baskets: hedged pairs, Delta-0.2's four legs, strangle legs, the SL Hunting BankNIFTY mirror) stream from ticks in real time, with legs subscribed/unsubscribed dynamically as workers enter and exit. REST stays for warmup history and reconnect gap-backfill; API load drops from one full-window pull every 2-5s to ~1/minute. The market-data health gates (10s LTP / 150s bar / 30s liquidation) behave identically, with one tick-feed-aware twist: quiet-but-subscribed legs stay fresh only while the socket is demonstrably alive. Rollback is `MARKET_DATA_SOURCE=REST` + restart; the tick logic lives in `Dependencies/tick_bar_builder.py`.
- **Per-strategy "off" switch — `<PREFIX>_VIRTUAL_TRADING`.** Every strategy now has a virtual (paper) toggle that **defaults to true**. Set it false to stop that strategy's worker thread from starting at all — so it does no paper trading (and, since the thread never runs, no live trading either). Unlike live trading there is **no** global master switch: the default is that everything runs, and you silence individual strategies. Lets you run just the strategies you want on a given day instead of the whole roster.
- **Quality gates & CI.** A GitHub Actions workflow (`.github/workflows/quality-and-security.yml`) runs the full gate on every push/PR across Python 3.12 + 3.13: all repository suites, branch-coverage budgets, `pip-audit`, `compileall`, **Ruff**, **mypy** (scoped in `pyproject.toml`), **Bandit**, and pre-commit validation. Exact tooling lives in `requirements-dev.txt`.
- **SL Hunting AI Agent — BankNIFTY mirror basket + newer knowledge (v3c–v3e).** The agent now trades Intraday Hunter's multi-index style: every NIFTY entry is mirrored with an **equal-lot BankNIFTY ATM** leg (`SL_HUNTING_BNF_MIRROR`, default true). The two legs are **tied for hard risk** (stop/target, max-loss, 15:15 square-off close both) but the agent evaluates each leg's **premise independently** and can cut one alone via the EXIT `exit_leg` selector (`NIFTY` | `BNF` | `BOTH`). Entry stays NIFTY-only (the mirror copies it). Its knowledge also grew several distilled-from-video layers — a scoped **gap-up opening-drive**, a **2-week verbatim transcript sweep**, and a **live-day match** against the agent's own journal (details in `Signal Generators/SL Hunting AI Agent/README.md`).
- **Optional LLM trading agent — the "SL Hunting AI Agent" (opt-in 27th worker).** A Claude agent (via the [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) on your Claude subscription — **no API key**) trades the discretionary *SL Hunting* price-action method on NIFTY ATM options. Once per completed 1-min bar (the method's native timeframe) it reads the NIFTY chart (with **BankNIFTY cross-confirmation**) and — only on a confirmed setup at a real level — acts through the SAME tested `enter_position`/`exit_position` path as every other worker. Position sizing floors affordable whole lots, never exceeds `SL_HUNTING_RISK_BUDGET`, skips one-lot-over-budget setups, and caps at `SL_HUNTING_MAX_LOTS` (default 5); the equal-lot BankNIFTY mirror can roughly double basket risk. It **stops opening new positions after noon** (`SL_HUNTING_NO_NEW_ENTRY_HOUR`, default 12:00) — *not* a square-off: open positions, their stops/targets, and the 15:15 square-off are unaffected. It is **off by default** (`SL_HUNTING_ENABLED`), trades **paper** unless both `LIVE_TRADING_ENABLED` and `SL_HUNTING_LIVE_TRADING` are set, and is **fail-soft** — any agent/SDK error becomes a safe HOLD while its separate mechanical risk loop keeps checking stop, target, max-loss, stale data, and square-off. It can also **learn from its own trades** through a tool-free, schema-validated reflection coach with digest-bound human approval (paper-first, off by default). Install the exact optional stack with `pip install -r requirements-ai.txt` and run one-time `claude setup-token` (keep `ANTHROPIC_API_KEY` **UNSET** so it bills your Claude plan, not per-token API). Full details — knowledge, tools, safety model, the learning loop — are in `Signal Generators/SL Hunting AI Agent/README.md`. This is the optional **27th** worker.
- **CPR Algo 3 (multi-instrument) is now wired into the front test.** A new `CPRAlgo3StrategyWorker` runs the "CPR basic setup" strategy, which watches THREE charts at once — the NIFTY spot plus a ~ITM CE and a ~ITM PE of the current-week expiry — and only fires when VWAP and the CPR band align across all three (RSI/ARSI on spot). The two ITM options are **observation only**: a signal still BUYS the ATM CE/PE of the next-next expiry through the same tested path as the other directional workers, so it shares CPR's risk knobs (tunable via `CPR_ALGO3_*` in `.env`, including `CPR_ALGO3_ITM_OFFSET`). It fetches the two option 1-min OHLC feeds on demand and drives its own spot target/stop exit. This brings the runner to **26 workers**. (The standalone Algo 3 signal generator + its unit tests live under `Signal Generators/CPR Strategy/`.)
- **Code-quality pass.** Added a `requirements.txt`; gave every Shoonya broker HTTP call a timeout (a hung call could otherwise stall a worker thread and the shared broker lock); removed hardcoded credentials from the vendored Shoonya client; routed the execution layer's status/errors through `logging` instead of `print()`; and ported the master test suite into the repo (`test_nifty_multi_strategy_master.py` — see Tests below).
- **Live broker execution is broker-selectable (Kotak Neo, Shoonya, or Flattrade).** `LIVE_BROKER` picks `KOTAK`, `SHOONYA`, or `FLATTRADE`, and every real order goes through one generic `execution_client`. The global `LIVE_TRADING_ENABLED` kill-switch and each strategy's `<PREFIX>_LIVE_TRADING` flag must both be true; unknown broker names fail closed to paper. Each broker folder contains an execution client and a read-only diagnostic with an optional, typed-`YES`, round-trip test order. Flattrade uses its official Pi v2 browser-token flow, exact NFO index scrip master, documented request limits, market-order protection, and `SingleOrdHist` fill confirmation. Everything still defaults to paper. (Shoonya's legacy QuickAuth endpoint is being decommissioned by Finvasia.)
- **End-of-day P&L is now written to a Google Sheet.** When all workers exit on a clean end of day, the master parses the run's log for each strategy's realised P&L and writes it into a tracker sheet — one row per strategy, one column per calendar day — overwriting today's cell and backfilling any blank earlier-this-month cells from the (append-mode) log. Auth is OAuth user-token via `gspread`; configure `GSHEET_ID` + an OAuth client in `.env` (see Setup). It's a safe no-op when unconfigured, so it never disturbs shutdown.
- **13 TradingBot signal-generator ports.** Thirteen ATM single-leg strategies were ported into `Signal Generators/` (SMA Crossover, Bollinger Bands, Keltner Squeeze, Mean Reversion Z-Score, ML Ensemble, Multi-Timeframe, Opening Range Breakout, Parabolic SAR, RSI Divergence, RSI Reversal, Stochastic, Supertrend, Volatility Breakout), all sharing `misc_strategy_common.py` and the mandatory TA-Lib 0.6.8 indicator backend. They're wired through the shared `AtmSingleLegStrategyWorker` factory and each is tunable from `.env` by its own prefix. ML Ensemble needs `scikit-learn`.
- **CPR (Central Pivot Range) strategy is now live in the front test.** It runs as an ATM single-leg worker (`CPRStrategyWorker`) alongside the other strategies: the master file feeds it 1-min OHLC, the CPR logic resamples to complete 5-min candles internally, and a LONG/SHORT signal buys the ATM CE/PE of the next-next expiry. Tunable via `CPR_*` knobs in the `.env` (lots, max-loss, poll, 09:25-15:15 window). (This brought the master file to nine workers at the time; see the latest addition at the top of this list for the current total.)
- **Telegram trade notifications.** A queue-based `TelegramMessageWorker` posts a message to a Telegram group/channel on every entry and exit from *any* worker. Each alert shows the strategy, the exact option instrument(s), lot size, entry and exit price, and P&L (hedged spreads show both legs). It runs on its own thread so Telegram latency or downtime never blocks the trading loop, and it's a cheap no-op when disabled. See Setup below to switch it on.

# Pro Tip
You might have to adjust the import addresses from which the files are to be imported because the files are in different directories in my local machine(fixed in the latest Claude commit)

# Repository structure
```
.
├── Nifty Multi Strategy Front Test - Master File.py   # multithreaded paper/live runner (~26 + optional LLM worker)
├── Data Extractors/                                   # 1m OHLC downloaders + shared helper
├── My Backtest Files (For Reference)/                 # backtesting.py-based backtests
├── Signal Generators/                                 # strategy / signal logic modules
└── Dependencies/                                      # shared config + live-execution layer
    ├── env.example                                    # copy to Dependencies/.env and fill in
    ├── dhan_token_setup.py                            # one-time DhanHQ OAuth token setup
    ├── Kotak API/                                     # kotak_execution.py + diagnose_kotak_symbol.py
    ├── Shoonya API/                                   # NorenApi.py + shoonya_execution.py + diagnose_shoonya_symbol.py
    ├── Flattrade API/                                 # flattrade_execution.py + diagnose_flattrade_symbol.py
    └── Dhan API/                                      # dhan_execution.py + diagnose_dhan_symbol.py
```
Each subfolder has its own `Readme.md` with the details.

# Setup
1. Python 3.10+ (I'm running 3.13).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   That covers the core (data fetch, backtests, runner). Install exact optional sets only when needed:
   ```
   pip install -r requirements-ai.txt        # SL Hunting Claude SDK stack
   pip install -r requirements-dev.txt       # local quality/security gates
   pip install pyotp==2.9.0 websocket-client==1.8.0  # Shoonya runtime
   pip install --no-deps "git+https://github.com/Kotak-Neo/Kotak-neo-api-v2.git@v2.0.1#egg=neo_api_client"
   ```
   Flattrade uses the core `requests` and `pandas` dependencies. Shoonya's NorenApi
   client is vendored. Kotak's official tag declares older exact pandas/requests
   versions, so `--no-deps` prevents it from silently downgrading the audited core
   runtime. `requirements-brokers.txt` records and tests the upstream broker
   dependency environment separately in CI; do not combine it with `requirements.txt`.
3. Configure credentials. Copy `Dependencies/env.example` to `Dependencies/.env` and fill it in (`.env` is git-ignored). Set your Dhan credentials there, then run the one-time token setup:
   ```
   python "Dependencies/dhan_token_setup.py"
   ```
   It walks you through the DhanHQ OAuth login and writes a fresh `DHAN_ACCESS_TOKEN` back into `.env`. All tunable strategy parameters live in this same `.env`.
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
   Auth is OAuth user-token via `gspread`: in Google Cloud enable the Sheets API, create an OAuth client of type **Desktop app**, download its JSON to `GSHEET_OAUTH_CLIENT_FILE`, and share the sheet with your Google account. The first run opens a browser once for consent and caches a token at `GSHEET_OAUTH_TOKEN_FILE`. PAPER results use the existing row labels in column A (e.g. `Renko Strategy`); LIVE and MIXED results use separate `Renko Strategy [LIVE]` and `Renko Strategy [MIXED]` rows so real-money outcomes cannot contaminate paper history. Unmatched strategies are skipped with a warning. Leave `GSHEET_ID` blank to disable (safe no-op).

6. (Optional) Live broker execution. Everything is paper by default. To place REAL orders, set in `Dependencies/.env`:
   ```
   LIVE_TRADING_ENABLED=true        # global kill-switch (default false)
   LIVE_BROKER=KOTAK                # KOTAK, SHOONYA, FLATTRADE, or DHAN
   RENKO_LIVE_TRADING=true          # flip the specific strategies you want live
   ```
   Then fill the selected broker's credential block. Flattrade needs `FLATTRADE_CLIENT_ID`, `FLATTRADE_API_KEY`, and `FLATTRADE_API_SECRET`; its optional `FLATTRADE_ACCESS_TOKEN` is validated when supplied, otherwise startup opens browser authorization and asks for the returned `request_code`. A strategy trades live only when `LIVE_TRADING_ENABLED` **and** its own `<PREFIX>_LIVE_TRADING` are both true. An entry falls back to paper only after a typed zero-fill `REJECTED`; `PARTIAL` or `UNKNOWN` means exposure may exist, freezes new live entries, and starts reconciliation. Check connectivity first with the read-only diagnostics — they can place a confirmation-gated round-trip (buy + auto square-off) test order via `--place-order`:
   ```
   python "Dependencies/Kotak API/diagnose_kotak_symbol.py" CE 23950 --place-order
   python "Dependencies/Shoonya API/diagnose_shoonya_symbol.py" CE 23950 26JUN25 --place-order
   python "Dependencies/Flattrade API/diagnose_flattrade_symbol.py" CE 24150 14JUL26 --place-order
   python "Dependencies/Dhan API/diagnose_dhan_symbol.py" CE 24150 14JUL26 --place-order
   ```

# Command-line interface
`algo.py` is a single entry point for every script in this repo via short commands. It just launches the underlying scripts (so each one still works on its own), and any flag beyond the selector passes straight through. From the repo root:

| Command | What it does | Example |
|---|---|---|
| `fetch-data --index {nifty,banknifty,finnifty}` | Download 1-min OHLC for an index | `python algo.py fetch-data --index nifty --interval 5 --lookback 5y` |
| `backtest --strategy {renko,ema,heikin,cpr,profit-shooter,goldmine,money-machine}` | Backtest one strategy against a CSV | `python algo.py backtest --strategy renko --data "Backtest Outputs/nifty_renko_futures_5y_1min_data.csv"` |
| `run` | Start the front-test master (paper by default; live per `.env`) | `python algo.py run` |
| `setup-token` | One-time DhanHQ token setup (writes `.env`) | `python algo.py setup-token` |
| `diagnose --broker {kotak,shoonya,flattrade,dhan}` | Read-only broker/symbol check (add `--place-order` for a test order) | `python algo.py diagnose --broker flattrade CE 24150 14JUL26` |

Run `python algo.py --help`, or `python algo.py <command> --help`, for the details.

# Typical workflow
1. Pull historical data — e.g. `python "Data Extractors/Nifty 1m 5Y Data Fetch Dhan.py"`. The CSV lands in `Backtest Outputs/`.
2. Run a backtest against that CSV — e.g. `python "My Backtest Files (For Reference)/Nifty Renko Strategy Backtest.py"`.
3. Once a strategy looks good, run `Nifty Multi Strategy Front Test - Master File.py` for multi-strategy execution — paper by default, or live once you've configured a broker (Setup step 6).

(Or do all three with the unified CLI above: `python algo.py fetch-data --index nifty` → `python algo.py backtest --strategy renko --data ...` → `python algo.py run`.)

The `Backtest Outputs/` folder is `.gitignore`-d, so generated CSVs/logs stay local.

# Tests
The front-test master has a unittest suite — env toggles, broker paper/live routing and the fail-closed `LIVE_BROKER` switch, order fill-confirmation, and symbol resolution. Run it from the repo root:
```
python -m unittest test_nifty_multi_strategy_master
```
Broker/SDK-specific cases skip automatically when optional dependencies are absent, and all broker HTTP/browser/order behaviour is mocked. Signal generators, execution/reconciliation primitives, data extractors, and repository-policy checks have focused suites under their respective folders. CI runs the whole quality gate on every push/PR — see "Quality gates & CI" below.

# Quality gates & CI
A GitHub Actions workflow (`.github/workflows/quality-and-security.yml`) runs on every push and pull request across Python 3.12 and 3.13. Locally, the same gate is:
```
pip install -r requirements-dev.txt
pip install -r requirements-ai.txt
python -m unittest test_nifty_multi_strategy_master
python -m unittest test_market_data_health
python -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q
python -m coverage erase
python -m coverage run -m unittest test_nifty_multi_strategy_master
python -m coverage run --append -m unittest test_market_data_health
python -m coverage run --append -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q
python -m coverage json -o coverage.json
python scripts/check_coverage_thresholds.py coverage.json
python -m pip_audit -r requirements.txt --no-deps --progress-spinner off
python -m pip_audit -r requirements-ai.txt --no-deps --progress-spinner off
python -m pip_audit -r requirements-dev.txt --no-deps --progress-spinner off
python -m compileall -q .
python -m ruff check .
python -m mypy
```
Coverage is branch-enabled: overall runtime coverage may not fall below 54.7%, new execution/reconciliation/data-safety modules require 90%, and every broker adapter requires 80%. The three local audit commands check committed direct pins; CI additionally audits the complete resolved dependency tree in a clean hosted environment. `pyproject.toml` holds the coverage, Ruff, and mypy config (mypy is scoped to the identifier-named modules — the spaced-name master file is covered by `compileall` + the unittest suite instead). `.pre-commit-config.yaml` wires the check-only hooks; install them once with `pre-commit install`.

# License
Released under the MIT License — see [LICENSE](LICENSE).
