# Architecture documentation

This folder is the committed design record for the NIFTY multi-strategy trading
system. It answers "how is this thing built, and why is it built that way"
without requiring a reader to open the 17k-line master runner first.

It is **not** operator documentation. For setup, credentials, `.env` keys and
day-to-day commands, read the root [`README.md`](../README.md) and each
component folder's own `Readme.md`.

## Layout

| Folder | What lives there | When to read it |
|---|---|---|
| [`hld/`](hld/) | One high-level design covering the whole repository | First. Start with [`hld/system-overview.md`](hld/system-overview.md). |
| [`lld/`](lld/) | One low-level design per component | When changing that component. |
| [`adr/`](adr/) | Architecture Decision Records — the *why* behind decisions already baked in | Before proposing to reverse one. |
| `superpowers/` | Per-session agent plans and specs. **Untracked** (`.gitignore`). | Never — it is a working scratchpad, not product documentation. |

## Reading order for a new contributor

1. [`hld/system-overview.md`](hld/system-overview.md) — the process, the threads, the data flow.
2. [`lld/risk-and-safety.md`](lld/risk-and-safety.md) — this is live-money code; read the safety model before anything else.
3. [`lld/master-runner.md`](lld/master-runner.md) — startup gates, supervision, shutdown.
4. Then the LLD for whichever component you are touching.

## Low-level designs

| Component | Document |
|---|---|
| Process lifecycle, thread supervision, startup/shutdown | [`lld/master-runner.md`](lld/master-runner.md) |
| Market data: REST poller, websocket producer, shared store, health gates | [`lld/market-data.md`](lld/market-data.md) |
| Strategy worker family and the signal-generator factory | [`lld/strategy-workers.md`](lld/strategy-workers.md) |
| Option contract resolution and order placement | [`lld/execution-and-brokers.md`](lld/execution-and-brokers.md) |
| Risk, sizing, and the live-trading safety model | [`lld/risk-and-safety.md`](lld/risk-and-safety.md) |
| SL Hunting AI Agent (optional, Claude) | [`lld/sl-hunting-ai-agent.md`](lld/sl-hunting-ai-agent.md) |
| CPR Codex AI Agent (optional, Codex) | [`lld/cpr-codex-ai-agent.md`](lld/cpr-codex-ai-agent.md) |
| Regime Adaptive router | [`lld/regime-adaptive.md`](lld/regime-adaptive.md) |
| Configuration and drift detection | [`lld/configuration.md`](lld/configuration.md) |
| Telegram alerts, EOD P&L sheet, logging | [`lld/reporting-and-observability.md`](lld/reporting-and-observability.md) |
| Data extraction and backtesting | [`lld/data-and-backtesting.md`](lld/data-and-backtesting.md) |
| Test architecture, coverage budgets, CI | [`lld/testing-and-ci.md`](lld/testing-and-ci.md) |

## Decision records

| ADR | Decision |
|---|---|
| [0001](adr/0001-single-process-thread-per-strategy.md) | One process, one thread per strategy |
| [0002](adr/0002-broker-agnostic-execution-contract.md) | Broker-agnostic execution contract, fail-closed broker selection |
| [0003](adr/0003-acknowledgement-is-not-a-fill.md) | Typed order outcomes and a quantity-bearing ledger |
| [0004](adr/0004-paper-by-default-double-gate.md) | Paper by default behind a two-flag live gate |
| [0005](adr/0005-rest-vs-websocket-market-data.md) | REST polling as default, websocket as opt-in producer |
| [0006](adr/0006-per-strategy-size-multiplier.md) | Per-strategy size multiplier, deliberately not global |
| [0007](adr/0007-llm-agents-as-opt-in-workers.md) | LLM agents as opt-in workers with host-owned gates |
| [0008](adr/0008-single-env-as-config-source.md) | A single `.env` as the only configuration source |
| [0009](adr/0009-importlib-loading-for-spaced-filenames.md) | `importlib` loading instead of renaming spaced files |
| [0010](adr/0010-tests-in-a-mirrored-tests-tree.md) | Tests consolidated into a mirrored top-level `Tests/` tree |
| [0011](adr/0011-committed-docs-untracked-superpowers.md) | Committed `docs/` set; Superpowers workspace untracked |
| [0012](adr/0012-crash-durable-session-state.md) | Per-trade P&L and open positions persisted during the session |
| [0013](adr/0013-codeql-false-positive-triage.md) | Five CodeQL alerts dismissed as false positives, not patched |
| [0014](adr/0014-tiered-rename-of-spaced-filenames.md) | Spaced filenames renamed in reviewable tiers, master last (supersedes 0009) |

## Keeping these documents honest

Architecture documents that drift are worse than no documents. Two habits keep
this set usable:

- **Change the LLD in the same commit as the code.** If a pull request changes
  how a component behaves, its LLD is part of that pull request.
- **Never restate a number the code owns.** Prefer "the default is defined in
  `Dependencies/env.example`" to copying the value here. Where a specific value
  genuinely matters to the design (the 10-second broker deadline, the 90%
  coverage budget), it is stated once with the file that owns it named next to it.

ADRs are append-only history. When a decision changes, add a new ADR that
supersedes the old one and mark the old one `Superseded`; do not edit the
original's decision.
