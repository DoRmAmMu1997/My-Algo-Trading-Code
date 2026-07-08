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
Once per **completed N-minute bar** (default **1** — the method's native timeframe;
`SL_HUNTING_DERIVED_TIMEFRAME_MINUTES`), the agent is handed the recent NIFTY candles
and runs one agentic pass:
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
| `sl_hunting_journal.py` | v3 trade journal — entry context + exit outcome per trade (JSONL). |
| `sl_hunting_coach.py` | v3 reflection coach — proposes lessons from the journal (`--reflect`/`--promote`/`--list`). |
| `sl_hunting_lessons.py` | v3 lessons store (schema, consolidate, `format_lessons` for prompt injection). |
| `lessons.json` | The APPROVED (live) lessons the agent injects — starts empty; you promote into it. |
| `tests/` | pytest suite — runs with a fake runner, no SDK/CLI/network. |

## Setup (one-time)
```bash
pip install claude-agent-sdk pydantic
# Authenticate to your Claude SUBSCRIPTION. For an UNATTENDED live runner prefer a
# long-lived token; interactive `claude login` also works but its OAuth login expires.
claude setup-token
# Keep ANTHROPIC_API_KEY UNSET so the agent bills your Claude plan, not per-token API.
```
Run this in the **same terminal/profile that launches the runner**: a headless process
has nothing to refresh an expired login, so a token created elsewhere (e.g. an IDE
session) will eventually return **HTTP 401**. Verify with `claude -p "say hi"` there.

**Troubleshooting auth:** if the worker logs `authentication failed (HTTP 401) ... run claude setup-token`
(older builds surfaced this opaquely as *"Claude Code returned an error result: success"*), the
spawned `claude` CLI has no valid subscription token in the runner's environment — re-run
`claude setup-token` as above and restart. A `429` likewise means the plan's usage/rate limit was
hit. Either way the agent just **HOLDs** (no trade).

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

By default the agent **stops opening new positions at 12:00** (`SL_HUNTING_NO_NEW_ENTRY_HOUR`
/ `_MINUTE`) — mirroring the "no fresh trades after noon" rule. This is **not** a square-off:
open positions keep running and are only force-closed by the existing `SQUARE_OFF_*` gate
(15:15); stop/target, AI exits and max-loss all keep working. As a bonus, once flat past the
cutoff the agent isn't called at all, so it makes **no LLM calls for the rest of the day**.

## Safety
- **Paper by default.** The agent is given exactly **one** order tool, chosen by
  the env (`place_paper_order` / `place_kotak_order` / `place_shoonya_order`) — it
  can never pick paper-vs-real or the broker.
- The agent **never raises** into the trading loop: any failure (SDK missing,
  malformed output, **auth 401 / usage-limit 429**, …) returns a safe `HOLD`, and the
  warning log names the cause (e.g. "authentication failed (HTTP 401) — run
  `claude setup-token`") so it's actionable.
- Each SDK call is **time-bounded** (`SL_HUNTING_SDK_TIMEOUT_SECONDS`, default 90s).
  The per-bar decision blocks the worker thread that also enforces stop/target,
  max-loss and the 15:15 square-off, so a hung CLI call is abandoned at the budget:
  that bar's order tool is disarmed (a late-waking loop cannot fire a zombie order)
  and the agent records a fail-soft `HOLD`. If the CLI stays hung, subsequent bars
  are **gated** until the abandoned call finishes — so at most one hung agent
  call/subprocess exists at a time instead of one accumulating per bar.
- Entry **stop/target are sanity-checked at the order tool** against the live price
  (correct side; stop within ~3%, target within ~10%) and bounded in the schema —
  a hallucinated level cannot silently disable the mechanical stop; the rejected
  order returns to the agent mid-loop so it can correct its levels.
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

## Learning from mistakes (v3)
The agent grows a **lessons memory** from its own trades — not by fine-tuning, but by a
human-reviewed loop:

1. **Journal** — every trade's entry context (decision + a pivot/fibo/pattern/cross-index
   snapshot) and exit outcome (reason, points, R-multiple, P&L, `followed_method`) is
   appended to a gitignored JSONL (`Backtest Outputs/sl_hunting_journal.jsonl`).
   On by default (`SL_HUNTING_JOURNAL_ENABLED`); a no-op when disabled.
2. **Reflect** — a separate read-only coach turns the journal into *proposed* lessons:
   ```bash
   python sl_hunting_coach.py --reflect            # proposes lessons -> *_proposed.json
   python sl_hunting_coach.py --list               # review proposed + live
   python sl_hunting_coach.py --promote <lesson-id> # human gate: approve into lessons.json
   ```
3. **Inject** — `lessons.json` (approved only) is injected into the agent's prompt **only
   when `SL_HUNTING_LESSONS_ENABLED=true`** (default off), loaded once per session so
   prompt caching holds. Validate first on paper: `sl_hunting_runner.py --lessons on|off`.

**Safety/ML guardrails:** lessons are **human-gated, paper-first, and off by default**;
the coach runs off the live loop; lessons are phrased as tendencies (not laws), require a
minimum sample, separate process from outcome (a sound setup that lost ≠ a mistake), and
the store is bounded/de-duplicated. Fine-tuning/RL and auto-promotion are out of scope.

## Bank Nifty methodology (v3a)
Knowledge-only drop distilled from 9 "Intraday Hunter" live-trading videos. General lessons
(gap-driven bias, "closing price" as the invalidation level, behavioural confirmation,
premise-invalidation stops, option **time-decay** discipline, book-on-weakness) were merged
into the existing curated sections; **BankNIFTY-specific** behaviour lives in a new
`BNF_SPECIFIC` section — a **triple-index** (BankNIFTY + NIFTY + Sensex) read with BankNIFTY
as the "major index", expiry-day index priority, and round-number magnets. It is **advisory
context for the cross-index read only — the agent still trades NIFTY ATM options**; nothing
about execution changes. The video audio is Hindi and raw transcripts weren't retrievable,
so the rules were distilled via YouTube's built-in "Ask"/Gemini summaries (a secondary AI
summary, recorded with provenance in `sl_hunting_doc.md` and operator-reviewable).

## Gap-up opening drive (v3c)
Knowledge-only drop from the 2026-07-02 live session (`WhfVxV0h5bo`) reviewed the same day
against the agent's decision log and journal — this time from the **verbatim Hindi
auto-transcript** (YouTube transcript panel), not an AI summary. The trader went LONG with a
small gap-up ~1 minute after open ("nobody is trapped on a mild gap-up, so there is no
SL-hunt — go WITH the market"); the agent, awake only from 09:25 and pattern-gated, took
three shorts instead. Changes: a **TRAP-DENSITY TEST** before every fade and a
**first-trade-WITH-the-gap** rule in `RETAIL_POSITIONING`; a new **`OPENING_DRIVE`** section —
the one scoped exception to the pattern+confirmation rule (with-gap LONG only, first ~15 min,
strict no-major-rejection conditions, honest wide stop with risk-budget sizing); cross-index
cautions (stale early verdicts; an opposing verdict means HOLD); and an expiry-day-as-fuel
note. Full analysis + provenance in the v3c addendum of `sl_hunting_doc.md`. The operator
separately moved the worker's start to 09:15 via `SL_HUNTING_TRADING_START_*` in `.env`
(config, not code).

## Two-week verbatim sweep (v3c → v3d)
All 18 in-window Intraday Hunter videos (18 Jun – 2 Jul sessions: 8 live trades, 9 nightly
plan clips, 1 weekly) were re-extracted from **verbatim transcripts** (15 captured; 2 have no
transcript; 1 was v3c). The wins AND losses together yielded the v3d layer: **read the gap
against the prior days** (a big counter-gap is a lure, not a signal; small counter-gaps read
as flat), the **SL-reachability test** (no hunt when the crowd's stops sit beyond an intact
level), an **OPENING_DRIVE variant B** (flat-open seller-hunt long after multi-day selling),
**gap-size asymmetry** across the three indices, the trap-construction leg, direction-first
hierarchy over divergence setups, and several risk heuristics (two-rejections rule,
early-adverse = wrong direction on expiry days, loss-streak bias). Per-video provenance and
the win/loss evidence table live in the v3d addendum of `sl_hunting_doc.md`.

## Live-day match + weekly sweep (v3e)
First session with v3c+v3d live (3 Jul): the agent and IH independently refused the opening
drive on a first-candle rejection, and the agent's winning double-top short shared IH's exact
premise — strong convergence (agent +31.7 pts across 5 trades). v3e adds what IH had that the
agent didn't: **both-sides participation** (exact-touch bounces that attract only one side are
fades), the **huge-gap mindset trap** (no nearby SLs exist — fade the first post-gap push),
**third-index lag** (two indices breaking a level doesn't commit the third), **setup staleness**
(late breaks reverse on their followers → normal target only), and loss-recovery discipline
from the revenge-trading lecture. Provenance and the per-trade match table live in the v3e
addendum of `sl_hunting_doc.md`.

## BankNIFTY mirror basket (v4)
Intraday Hunter trades a multi-index basket; the worker now mirrors him: every NIFTY entry
also BUYS the **same lot count** on the **BankNIFTY ATM** option of the **current BNF monthly
expiry** (BNF has no weekly series), rolling to the next month once fewer than
`SL_HUNTING_BNF_MIRROR_ROLLOVER_DAYS` (default 7) days remain to it — never the illiquid
second month out. Entry stays NIFTY-only (the mirror copies it). The mirror
is mechanical; same paper/live gates as the NIFTY leg; fail-soft (a mirror problem only skips
the mirror); **basket risk ≈ 2× the ~Rs.2500 budget** (operator-accepted — the daily max-loss
kill-switch still caps the day). Toggle: `SL_HUNTING_BNF_MIRROR` (default true). Journal rows'
`option_pnl` includes both legs; MIRROR ENTRY/EXIT lines appear in the log and Telegram.

### Exit coupling (v5): tied for hard risk, independent for premise
The two legs are coupled **differently on the way out**:
- **Hard risk stays TIED** — the NIFTY leg's stop/target, the daily max-loss, and the 15:15
  square-off each close **both** legs. The mirror carries no stop/target of its own.
- **Premise-invalidation is PER-LEG** — the agent judges each leg on its own read (NIFTY on
  NIFTY structure, the mirror on BankNIFTY's own structure via `bank_nifty`/`cross_index`) and
  can cut just one. It acts through a new `exit_leg` on the EXIT decision: `NIFTY` (cut NIFTY,
  keep the mirror), `BNF` (cut the mirror, keep NIFTY), or `BOTH` (default). `position_state`
  now shows the mirror as its own leg with `nifty_leg_pnl` + a `mirror` block alongside the
  basket `unrealized_pnl`. If the agent cuts one leg, the lone survivor is still swept by
  max-loss + the 15:15 square-off (no orphan can leak open). A **lone mirror still reads as
  "in position"**, and a fresh NIFTY entry is refused until it is closed (one basket at a
  time — no stale-mirror pairing). When the NIFTY leg is cut first, its **journal row is held
  open** until the mirror closes too, so `option_pnl` always reflects the whole basket.

## Decision log
The agent decides once per completed bar, but the worker only *logs* the bars where it
acts — so a HOLD leaves just a `decision cost` line with no record of **what** it decided or
**why**. The **decision log** fills that gap: every decision (HOLD included) is appended to a
gitignored JSONL with the action, confidence, setup, stop/target, reasoning, and the
deterministic context the agent saw (pivot/fibo/patterns/cross-index).

```bash
tail -f "Backtest Outputs/sl_hunting_decisions.jsonl"
```

On by default (`SL_HUNTING_DECISIONS_ENABLED`, path `SL_HUNTING_DECISIONS_PATH`); a no-op
when disabled. It is **separate from the trade journal** above on purpose: that journal is
the reflection coach's learning input and must stay trade-only (a HOLD has no win/loss
outcome, and ~70 no-trade bars a day would skew the coach's stats). So: trade journal =
*completed trades* the coach learns from; decision log = *every decision* for you to review.
