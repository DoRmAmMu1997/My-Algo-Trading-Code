# AGENTS.md — My-Algo-Trading-Code

> Working principles for ANY coding agent in this repo: simplicity first, surgical changes,
> surface tradeoffs, and verify before claiming done. This is live-money trading code — bias
> toward caution. (Claude Code additionally loads its skills per `CLAUDE.md`; everything from
> "What this project is" down is kept identical in both files — edit them together.)

## What this project is
A NIFTY index-options, multi-strategy trading system. The flow is: **fetch** 1-minute OHLC history from
the DhanHQ API → **backtest** strategies on it → **run** a multithreaded "front test" that executes ~26
strategies together — on paper by default, and live through a real broker when explicitly enabled.
Running live since May 2026; daily per-strategy results are tracked in a Google Sheet.

## Architecture (runtime)
One process, cooperating threads:
- `CentralMarketDataFetcher` (one thread) polls DhanHQ and writes into a **lock-guarded
  `SharedMarketDataStore`** (1-min OHLC + LTPs). Setting `MARKET_DATA_SOURCE=WEBSOCKET`
  (fails closed to REST on any other value; needs the paid Dhan Data API subscription)
  swaps in `WebSocketMarketDataFetcher`: Dhan marketfeed ticks build the bars/LTPs
  (pure helpers in `Dependencies/tick_bar_builder.py`), with REST kept for warmup and a
  once-per-minute true-up against official candles.
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
  Google Sheet with separate PAPER/LIVE/MIXED row labels. All behaviour is driven by a single `.env`
  — nothing is hard-coded per run.

## Repository layout
```
Nifty Multi Strategy Front Test - Master File.py   # the multithreaded paper/live runner (the "big one")
algo.py                                             # unified CLI: fetch-data / backtest / run / setup-token / diagnose / check-env
test_nifty_multi_strategy_master.py                # unittest suite for the master
test_market_data_health.py                         # unittest suite for the shared feed-health gates
requirements.txt                                   # exact core runtime dependencies
requirements-brokers.txt                           # exact Kotak/Shoonya optional live set
requirements-ai.txt                                # exact optional Claude Agent SDK stack
requirements-dev.txt                               # exact local/CI quality tools
Data Extractors/                                   # DhanHQ 1-min OHLC downloaders (shared engine + per-index wrappers)
My Backtest Files (For Reference)/                 # backtesting.py backtests (+ Subhamoy Strategies/)
Signal Generators/                                 # strategy signal logic (+ CPR Strategy/, Subhamoy Strategies/,
                                                   #   SL Hunting AI Agent/ — optional Claude-agent strategy)
Dependencies/
  env.example                                      # template; copy to Dependencies/.env (gitignored)
  dhan_token_setup.py                              # one-time DhanHQ OAuth token setup
  check_env_config.py                              # `algo.py check-env` config-drift audit (read-only)
  Kotak API/     -> kotak_execution.py, diagnose_kotak_symbol.py
  Shoonya API/   -> NorenApi.py (vendored client), shoonya_execution.py, diagnose_shoonya_symbol.py
  Flattrade API/ -> flattrade_execution.py, diagnose_flattrade_symbol.py,
                    test_flattrade_execution.py
  Dhan API/      -> dhan_execution.py, diagnose_dhan_symbol.py, test_dhan_execution.py
pyproject.toml                                     # ruff + mypy quality-gate configuration
.github/workflows/quality-and-security.yml         # CI: tests + compileall + ruff + mypy + bandit
scripts/check_coverage_thresholds.py               # branch-coverage policy gate
Backtest Outputs/                                  # generated CSVs/logs (gitignored)
```

## Conventions
- **Config:** one `.env` (gitignored) is the single source of truth, copied from `Dependencies/env.example`.
  Read values through the `_env_str` / `_env_bool` / `_env_int` / `_env_float` helpers (master ~L352-406),
  not ad-hoc `os.getenv`; size-bearing knobs go through `_scaled_int` / `_scaled_float` instead (see
  size multiplier below). Per-strategy knobs are namespaced `<PREFIX>_*` (e.g. `RENKO_*`, `CPR_*`); the
  name→prefix map is `STRATEGY_ENV_PREFIX`. **Never commit secrets** — `env.example` holds blank placeholders.
- **Live-trading safety (critical):** paper by default. A strategy trades live ONLY when the global
  `LIVE_TRADING_ENABLED` **and** that strategy's `<PREFIX>_LIVE_TRADING` are both true. `LIVE_BROKER`
  (`KOTAK` | `SHOONYA` | `FLATTRADE` | `DHAN`) selects the broker; an unknown value **fails closed**
  (live disabled, paper only).
  An entry falls back to paper only after an explicit `REJECTED` result with zero fill. `PARTIAL` or
  `UNKNOWN` means exposure may exist: freeze new live entries, keep exits available, and reconcile;
  never treat an acknowledgement, truthy value, or order ID as proof of fill. A rejected live exit
  keeps the position open. Every broker network/SDK call has a ten-second deadline that includes its
  shared lock/rate-limit wait; native HTTP timeouts remain enabled for Shoonya, Flattrade and Dhan
  (the Dhan SDK ships a 60s default that `_login_locked` overrides down to 10s).
- **Per-strategy on/off:** each strategy also has a `<PREFIX>_VIRTUAL_TRADING` gate (default true).
  Set it false to stop that strategy's worker thread from starting at all (so it does neither paper
  nor live). Unlike live trading there is **no** global master switch — default is everything runs.
  `main()` filters the `workers` list via `_strategy_virtual_trading_enabled` before starting threads.
- **Per-strategy size multiplier:** `<PREFIX>_SIZE_MULTIPLIER` (default 1, whole numbers 1-25,
  ceiling `MAX_SIZE_MULTIPLIER`) scales that strategy's whole size/risk set together — `_LOTS`,
  `_MAX_LOTS`, `_RISK_BUDGET` and the absolute `_MAX_LOSS` — so size can grow with the account by
  editing one number. Applied at env-read time via `_scaled_int` / `_scaled_float` /
  `_strategy_size_multiplier` (master ~L409-467), so `Dependencies/risk_sizing.py` is untouched and
  scaled values flow through sizing, the kill-switch, Telegram and the Sheet unchanged. Deliberately
  **per-strategy only** (no global switch) and it applies to **paper and live alike**. Malformed
  values fall back to 1 for paper but are **blocked from live** by `_live_config_errors`. Two things
  are deliberately NOT scaled, because their totals already inherit the multiplier and scaling them
  would square it: `<PREFIX>_MAX_LOSS_PER_LOT` (Delta20) and `<PREFIX>_STARTING_CAPITAL` /
  `_DAILY_MAX_LOSS_PCT` (their product carries it). A drift-guard test fails if a new strategy reads
  a size knob with the raw `_env_*` helpers.
- **Broker layer:** the Kotak, Shoonya, Flattrade and Dhan clients expose the SAME surface —
  `ensure_logged_in`, `preload_scrip_master`, `resolve_option_symbol`, `place_market_order`,
  `get_order_status`, `cancel_order`, `list_open_orders`, `list_open_positions`,
  `recover_after_reconciliation`, `extract_order_id`, `logout`, `is_logged_in` — so the runner only
  touches the generic `execution_client`. The shared result types live in
  `Dependencies/broker_contract.py`. The Shoonya `NorenApi` client is vendored under
  `Dependencies/Shoonya API/`. Dhan is the only broker whose SDK serves both market data and
  execution, but the two sessions stay separate (`DhanBrokerClient` vs `dhan_execution_client`).
  Two Dhan quirks the adapter exists to contain: its SDK returns `{'status':'failure',
  'remarks': str(exc)}` for *transport* errors, which is shape-identical to a real rejection — so
  `REJECTED` is never derived from the placement envelope (a `dict` `remarks` means the server
  refused, a `str` means it is indeterminate); and `order_tag` is sent as Dhan's `correlationId`
  so `get_order_by_correlationID` can recover an order whose response was lost. Dhan's
  non-contract states are aliased adapter-locally (`EXPIRED`→`CANCELLED`,
  `PART_TRADED`→`PARTIAL`); `TRANSIT`/`PENDING` stay unmapped so they remain transient.
  Dhan resolves contracts from the local `Dependencies/all_instrument <date>.csv`, not a download.
- **Credential-safe logging:** `setup_logging()` installs `install_redaction_filter` on the root
  logger with `environment_secrets(os.environ)` (every `.env` value whose KEY looks sensitive, ≥8
  chars), so **every** record — lazy `%s` args and exception tracebacks included — is scrubbed before
  it reaches the console or the append-mode log. This matters concretely: dhanhq's marketfeed puts
  the live access token in its websocket URL, so a connect error would otherwise write it verbatim
  into a log operators routinely share. Do not hand-redact new call sites; the guard covers them.
  Short values (a 4-digit MPIN) are deliberately excluded from exact-match replacement — they would
  blank strike prices and quantities — and are caught by `redact_text`'s `name=value` pass instead.
- **Code style:** detailed, beginner-friendly module + function docstrings and plain-English inline
  comments — match the existing density. Type hints where practical. `snake_case` functions/modules,
  `PascalCase` classes, `UPPER_SNAKE` constants and env keys. In library code use a module
  `logging.getLogger(__name__)` logger, **not `print()`**. Strategy files have spaces in their names and
  are imported via `load_module()` (master ~L1024), not regular imports.
- **CLI:** prefer `python algo.py <command>` (`fetch-data` / `backtest` / `run` / `setup-token` /
  `diagnose` / `check-env`); each underlying script still runs standalone, and any flag beyond the
  selector passes straight through. A bare `python algo.py` prints help.
- **Config drift:** `python algo.py check-env` (`Dependencies/check_env_config.py`) audits
  `Dependencies/.env` against `env.example` and against the keys the code's `_env_*` calls actually
  read, reporting settings missing from `.env` (an unseen in-code default is in force), mistyped or
  stale keys, and knobs missing from the template. Read-only, and it prints key NAMES only — never a
  value out of `.env` — so its output is safe to share. `test_repository_policy.py` imports the same
  helpers so CI fails when a new `_env_*` key lands without an `env.example` entry.
- **Tests:** `python -m unittest test_nifty_multi_strategy_master` (loads the master via `importlib`,
  mocks `dhanhq`; broker/SDK-specific cases skip when those deps are absent). Signal-generator tests live
  under `Signal Generators/`.
- **Quality gates (run before pushing; CI enforces on Python 3.12 + 3.13):**
  `python -m unittest test_nifty_multi_strategy_master`,
  `python -m unittest test_market_data_health`,
  `python -m pytest "Signal Generators" "Dependencies" "Data Extractors" -q`,
  the branch-enabled Coverage.py run plus `scripts/check_coverage_thresholds.py`,
  pip-audit of committed pins locally plus the clean resolved CI environment,
  Ruff, mypy, compileall,
  Bandit, and pre-commit. Coverage floors are 54.7% overall, 90% for new
  execution/reconciliation/data-safety modules, and 80% per broker adapter.
- **Dependencies:** install core with `pip install -r requirements.txt`; add
  `requirements-ai.txt` for SL Hunting and `requirements-dev.txt` for local
  gates. `requirements-brokers.txt` is the isolated upstream compatibility
  environment and must not be combined with core because Kotak pins older
  pandas/requests; use the safe per-broker commands in README. Kotak v2 comes
  from its official `v2.0.1` Git tag.
- **Git / PRs:** branch off `main`; open PRs into `main` with `gh`; end commit messages with a
  `Co-Authored-By:` trailer identifying the agent that produced the change. `.env`,
  `Backtest Outputs/`, and `*.log` stay gitignored.
