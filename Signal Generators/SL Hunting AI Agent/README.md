# SL Hunting AI Agent

An **LLM-driven** NIFTY index-options strategy. Instead of a deterministic signal
engine, a **Claude agent** (via the [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/))
trades the discretionary *"SL Hunting"* price-action method: pivot retests,
previous-day OHLC, Fibonacci 50/61/78 reversals, the mandatory *candlestick
pattern + confirmation candle* rule, trendline "3rd-point / break" logic, W/M
patterns, the gap-up/down playbook, and the core psychology of trading **opposite
to where retail stop-losses sit**.

It mirrors the in-house agent pattern from the sibling `Streamlit Scanner App`
repo: `claude-agent-sdk` on your Claude subscription (no API key), in-process MCP
tool servers, an injectable `runner=` testing seam, a Windows-safe async→sync
bridge, and strict Pydantic output validation.

## How it works
Once per **completed N-minute bar** (default 5), the agent is handed the recent
NIFTY candles and runs one agentic pass:
1. It calls **read-only context tools** for deterministic facts —
   `pivot_and_levels`, `candle_patterns` (with confirmation status), `fibo_levels`,
   `market_structure`, `position_state`, plus `bank_nifty` and `cross_index` for
   the **BankNIFTY cross-confirmation** (the NF/BNF rules; advisory).
2. If — and only if — a confirmed setup at a real level with a tight stop and a
   worthwhile target is present, it calls its **single order tool** to act.
3. Its final message is a strict `SLHuntingDecision` JSON object (logged as a
   record of what it did).

`ENTER_LONG` buys an ATM **CALL**; `ENTER_SHORT` buys an ATM **PUT**. Stops and
targets are levels on the NIFTY **underlying** (spot). The agent does **not** choose
the lot count — position size is computed automatically so the worst-case risk is
**~₹2500 per trade** (`SL_HUNTING_RISK_BUDGET`), from the agent's stop distance.

## Files
| File | Purpose |
|---|---|
| `sl_hunting_doc.md` | Verbatim source methodology (the agent's ground truth). |
| `sl_hunting_knowledge.py` | `build_system_prompt()` + strict-JSON `FINAL_OUTPUT_INSTRUCTION` — the agent's "brain". |
| `sl_hunting_indicators.py` | Deterministic detectors (pivot/levels, fibo, candlestick patterns + confirmation, structure, NF/BNF cross-index). |
| `sl_hunting_tools.py` | In-process MCP server: 7 read-only tools (incl. `bank_nifty`/`cross_index`) + 1 env-selected order tool. |
| `sl_hunting_executor.py` | `StandaloneExecutor` (paper) and `MasterWorkerExecutor` (delegates to the master worker). |
| `sl_hunting_ai_validation.py` | `StrictAIModel` + `parse_with_retry` (bounded retry on malformed JSON). |
| `sl_hunting_agent.py` | `SLHuntingAgent` + the strict `SLHuntingDecision` schema. |
| `sl_hunting_runner.py` | Standalone PAPER harness (synthetic/CSV replay). |
| `tests/` | pytest suite — runs with a fake runner, no SDK/CLI/network. |

## Setup (one-time)
```bash
pip install claude-agent-sdk pydantic
claude login            # sign in once with your Claude subscription
# Ensure ANTHROPIC_API_KEY is UNSET so the agent bills your plan, not per-token API.
```

## Run standalone (paper-first)
```bash
cd "Signal Generators/SL Hunting AI Agent"

# Zero-cost pipeline smoke test (no data file, no SDK):
python sl_hunting_runner.py --synthetic --fake

# Replay a real 1-min NIFTY CSV with the live agent, capped to 30 decisions:
python sl_hunting_runner.py --csv path/to/nifty_1m.csv --max-bars 30

# Add BankNIFTY for cross-confirmation (pair a 1-min BankNIFTY CSV with the NIFTY one);
# --synthetic generates a correlated BankNIFTY automatically:
python sl_hunting_runner.py --csv nifty_1m.csv --bnf-csv banknifty_1m.csv --max-bars 30
```
The standalone runner is **paper-only** by design (its P&L is a proxy on the
underlying, to validate decisions — not option pricing).

## Run inside the master front-test (the live path)
Set in `Dependencies/.env`:
```
SL_HUNTING_ENABLED=true          # include the worker in the master roster
SL_HUNTING_MODEL=claude-opus-4-8 # or claude-sonnet-4-6 to cut cost
```
Then run the master as usual (`python algo.py run`). It trades on **paper** unless
both `LIVE_TRADING_ENABLED` and `SL_HUNTING_LIVE_TRADING` are `true`; the live
broker is the existing `LIVE_BROKER` (`KOTAK`/`SHOONYA`). Live orders go through the
master's one shared, lock-guarded broker session and its `enter_position` /
`exit_position` (so max-loss, square-off and Telegram all apply). See the
`SL_HUNTING_*` block in `Dependencies/env.example` for all knobs.

## Safety
- **Paper by default.** The agent is given exactly **one** order tool, chosen by
  the env (`place_paper_order` / `place_kotak_order` / `place_shoonya_order`) — it
  can never pick paper-vs-real or the broker.
- The agent **never raises** into the trading loop: any failure (SDK missing,
  malformed output, usage limit) returns a safe `HOLD`.
- Both extra deps are **lazily imported**, so a missing dep just disables this one
  worker — the rest of the master and its test suite are unaffected.

## Tests
```bash
pytest "Signal Generators/SL Hunting AI Agent/tests"
```

## BankNIFTY cross-confirmation (v2)
The agent applies the method's NF/BNF rules via the `bank_nifty` + `cross_index`
tools. BankNIFTY 1-min OHLC is **not** in the master's shared store, so the worker
fetches it on demand each bar through the shared broker (`fetch_index_1m_ohlc`, the
same path CPR Algo 3 uses) — set `SL_HUNTING_USE_BNF=false` to disable. It is
**advisory**: it strengthens/weakens a NIFTY setup but never hard-gates, and any
BankNIFTY fetch failure auto-degrades to NIFTY-only for that bar.

## Source images (v2)
The source doc's ~108 pages of annotated chart screenshots were exported and
reviewed; they are illustrative of the prose rules and contained **no net-new
knowledge**, so nothing was added to the agent's knowledge from them (see the note
at the top of `sl_hunting_doc.md`).
