# AGENTS.md — My-Algo-Trading-Code

> Working principles for ANY coding agent in this repo: simplicity first, surgical changes,
> surface tradeoffs, and verify before claiming done. This is live-money trading code — bias
> toward caution. (Claude Code additionally loads its skills per `CLAUDE.md`; both files must
> be kept factually in sync.)

## What this project is
A NIFTY index-options, multi-strategy trading system. The flow is: **fetch** 1-minute OHLC history from
the DhanHQ API → **backtest** strategies on it → **run** a multithreaded "front test" that executes ~26
strategies together — on paper by default, and live through a real broker when explicitly enabled.
Running live since May 2026; daily per-strategy results are tracked in a Google Sheet.

## Architecture (runtime)
One process, cooperating threads:
- `CentralMarketDataFetcher` (one thread) polls DhanHQ and writes into a **lock-guarded
  `SharedMarketDataStore`** (1-min OHLC + LTPs).
- **~26 strategy worker threads** read that store and decide trades: the `AtmSingleLegStrategyWorker`
  family (Renko / EMA / Heikin-Ashi / Profit-Shooter / Goldmine / Money-Machine / CPR / CPR Algo 3
  (multi-instrument: spot + ITM CE + ITM PE) / Opening-Strike + 13 ported TradingBot strategies), two
  **hedged-puts** workers, one **Delta-0.2** hedged-spread worker,
  and one **long-strangle** worker (time-based dual-leg BUY of OTM1 CE+PE, with momentum re-entry).
  One **optional, opt-in** 27th worker is LLM-driven: the **SL Hunting AI Agent** (a Claude agent via
  `claude-agent-sdk`) — off by default (`SL_HUNTING_ENABLED`), it decides once per completed 1-min bar
  (with BankNIFTY cross-confirmation, fetched per bar like CPR Algo 3, and dynamic ~₹2500 risk-based
  sizing) and acts through the same ATM `enter_position`/`exit_position`; its deps are lazily imported
  so a missing dep just disables it. Every NIFTY entry is mechanically MIRRORED with an equal-lot
  BankNIFTY ATM leg (`SL_HUNTING_BNF_MIRROR`, default true) — NOTE: the mirror roughly DOUBLES the
  basket's rupee risk beyond `SL_HUNTING_RISK_BUDGET` (operator-accepted; the daily max-loss
  kill-switch still caps the day): the legs are TIED for hard risk
  (stop/target, max-loss, 15:15 square-off close both) but the agent evaluates each leg's
  premise INDEPENDENTLY and can cut one alone via the EXIT `exit_leg` selector (NIFTY|BNF|BOTH).
  Entry stays NIFTY-only (the mirror copies it). It stops opening NEW positions after noon
  (`SL_HUNTING_NO_NEW_ENTRY_HOUR`, default 12:00) — not a square-off (exits + the 15:15 square-off
  still run; when flat past the cutoff it skips the LLM call entirely). It can also **learn from its own trades** (v3): a per-trade journal
  feeds an off-loop reflection coach (`sl_hunting_coach.py`) that proposes lessons; the operator promotes
  approved ones into `lessons.json`, injected into the prompt only when `SL_HUNTING_LESSONS_ENABLED`
  (human-gated, paper-first, off by default). Its knowledge also carries a curated BankNIFTY
  live-trading layer (v3a, knowledge-only): a `BNF_SPECIFIC` section (triple-index BNF+NIFTY+Sensex
  read, BankNIFTY as the "major index", expiry-day priority, round-number magnets) that is **advisory
  context for the cross-index read — execution stays NIFTY-only** — plus general lessons merged into
  the existing sections (distilled from Intraday Hunter videos; provenance in `sl_hunting_doc.md`).
- Each entry/exit is published to a `queue.Queue` consumed by a single `TelegramMessageWorker`
  (best-effort alerts; never blocks trading).
- Real orders go through ONE shared, lock-guarded broker session via a broker-agnostic
  **`execution_client`** (see Broker layer). On a clean end-of-day, per-strategy P&L is written to a
  Google Sheet. All behaviour is driven by a single `.env` — nothing is hard-coded per run.

## Repository layout
```
Nifty Multi Strategy Front Test - Master File.py   # the multithreaded paper/live runner (the "big one")
algo.py                                             # unified CLI: fetch-data / backtest / run / setup-token / diagnose
test_nifty_multi_strategy_master.py                # unittest suite for the master (127 tests)
requirements.txt                                   # core deps (+ commented optional broker/ML deps)
Data Extractors/                                   # DhanHQ 1-min OHLC downloaders (shared engine + per-index wrappers)
My Backtest Files (For Reference)/                 # backtesting.py backtests (+ Subhamoy Strategies/)
Signal Generators/                                 # strategy signal logic (+ CPR Strategy/, Subhamoy Strategies/,
                                                   #   SL Hunting AI Agent/ — optional Claude-agent strategy)
Dependencies/
  env.example                                      # template; copy to Dependencies/.env (gitignored)
  dhan_token_setup.py                              # one-time DhanHQ OAuth token setup
  Kotak API/     -> kotak_execution.py, diagnose_kotak_symbol.py
  Shoonya API/   -> NorenApi.py (vendored client), shoonya_execution.py, diagnose_shoonya_symbol.py
  Flattrade API/ -> flattrade_execution.py, diagnose_flattrade_symbol.py,
                    test_flattrade_execution.py
pyproject.toml                                     # ruff + mypy quality-gate configuration
.github/workflows/quality-and-security.yml         # CI: tests + compileall + ruff + mypy + bandit
Backtest Outputs/                                  # generated CSVs/logs (gitignored)
```

## Conventions
- **Config:** one `.env` (gitignored) is the single source of truth, copied from `Dependencies/env.example`.
  Read values through the `_env_str` / `_env_bool` / `_env_int` / `_env_float` helpers (master ~L253-292),
  not ad-hoc `os.getenv`. Per-strategy knobs are namespaced `<PREFIX>_*` (e.g. `RENKO_*`, `CPR_*`); the
  name→prefix map is `STRATEGY_ENV_PREFIX`. **Never commit secrets** — `env.example` holds blank placeholders.
- **Live-trading safety (critical):** paper by default. A strategy trades live ONLY when the global
  `LIVE_TRADING_ENABLED` **and** that strategy's `<PREFIX>_LIVE_TRADING` are both true. `LIVE_BROKER`
  (`KOTAK` | `SHOONYA` | `FLATTRADE`) selects the broker; an unknown value **fails closed** (live
  disabled, paper only).
  Any order failure falls back to paper for that trade. Every broker HTTP call must have a timeout.
- **Per-strategy on/off:** each strategy also has a `<PREFIX>_VIRTUAL_TRADING` gate (default true).
  Set it false to stop that strategy's worker thread from starting at all (neither paper nor live).
  Unlike live trading there is **no** global master switch — default is everything runs. `main()`
  filters the `workers` list via `_strategy_virtual_trading_enabled` before starting threads.
- **Broker layer:** the Kotak and Shoonya clients expose the SAME surface — `ensure_logged_in`,
  `preload_scrip_master`, `resolve_option_symbol`, `place_market_order`, `extract_order_id`, `logout`,
  `is_logged_in` — so the runner only ever touches the generic `execution_client`. The Shoonya `NorenApi`
  client is vendored under `Dependencies/Shoonya API/`.
- **Code style:** detailed, beginner-friendly module + function docstrings and plain-English inline
  comments — match the existing density. Type hints where practical. `snake_case` functions/modules,
  `PascalCase` classes, `UPPER_SNAKE` constants and env keys. In library code use a module
  `logging.getLogger(__name__)` logger, **not `print()`**. Strategy files have spaces in their names and
  are imported via `load_module()` (master ~L742), not regular imports.
- **CLI:** prefer `python algo.py <command>` (`fetch-data` / `backtest` / `run` / `setup-token` /
  `diagnose`); each underlying script still runs standalone, and any flag beyond the selector passes
  straight through. A bare `python algo.py` prints help.
- **Tests:** `python -m unittest test_nifty_multi_strategy_master` (loads the master via `importlib`,
  mocks `dhanhq`; broker/SDK-specific cases skip when those deps are absent). Signal-generator tests live
  under `Signal Generators/`.
- **Dependencies:** `pip install -r requirements.txt`; optional broker/ML extras are documented (commented) inside it.
- **Git / PRs:** branch off `main`; open PRs into `main` with `gh`; end commit messages with a
  `Co-Authored-By:` trailer identifying the agent that produced the change. `.env`,
  `Backtest Outputs/`, and `*.log` stay gitignored.
- **Quality gates (run locally before pushing):** `python -m unittest test_nifty_multi_strategy_master`,
  `python -m pytest "Signal Generators" -q`, `python -m ruff check .`, `python -m mypy`,
  `python -m compileall -q .` — CI enforces the same set on Python 3.12 + 3.13.
