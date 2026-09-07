# LLD — Market data: producers, shared store, health gates

**Owns:** `CentralMarketDataFetcher`, `WebSocketMarketDataFetcher`,
`SharedMarketDataStore`, `MarketSnapshot`, `LTPSnapshot`, `OptionSubscription`
(master file) · `Dependencies/market_data_health.py` ·
`Dependencies/tick_bar_builder.py`
**Consumed by:** every strategy worker
**Related ADR:** [0005 — REST vs websocket](../adr/0005-rest-vs-websocket-market-data.md)

---

## 1. Responsibility

Produce one authoritative view of the market that ~30 worker threads can read
concurrently, and make "is this data safe to trade on?" a question with exactly
one answer for all of them.

The second half is the important half. If each strategy decided independently
whether the feed was stale, they would disagree, and some would trade on data
others had rejected.

---

## 2. The shared store

```python
SharedMarketDataStore            # lock-guarded; one writer, many readers
  ├─ 1-minute OHLC frames        # per instrument (NIFTY spot, BankNIFTY, option legs)
  ├─ LTP cache                   # per subscribed leg, with a timestamp
  ├─ MarketDataHealth            # freshness state, shared by all readers
  └─ OptionSubscription set      # which legs are currently subscribed
```

Readers take the lock, copy what they need, and release. Workers never hold the
lock across a decision, let alone across a broker call.

`MarketSnapshot` and `LTPSnapshot` are the immutable value objects handed out —
a worker cannot accidentally mutate shared state by holding onto what it read.

---

## 3. Two producers, one contract

Exactly one producer thread runs per process. Selection is by
`MARKET_DATA_SOURCE`, and **any value other than `WEBSOCKET` yields REST** —
including typos. `_select_market_data_fetcher_class()` owns that decision.

### 3.1 `CentralMarketDataFetcher` (REST, default)

```
loop:
  sleep(poll interval, 2-5s)
  GET intraday OHLC for the full window
  normalize_dhan_intraday_response(resp)      # epoch-unit inference, column mapping
  validate_ohlc_frame(frame)                  # fail closed on bad geometry
  build_last_row_signature(frame)             # cheap change detection
  store.publish(frame, ltps)
```

Simple, no state to recover, and every bar is by definition the official
exchange candle. Its cost is API load: one full-window pull every few seconds.

### 3.2 `WebSocketMarketDataFetcher` (opt-in, needs the paid Data API)

Two cooperating pieces:

```
  pump thread                  supervisor
  ───────────                  ──────────
  dhanhq.marketfeed packets ─► tick_bar_builder (pure helpers)
                                 ├─ update the FORMING minute in real time
                                 ├─ close the minute at the boundary
                                 └─ update LTP per leg
                               once per minute:
                                 REST fetch official candles
                                 true-up completed bars  ← official wins,
                                                            divergence logged
                               on connect / reconnect:
                                 REST warmup + gap backfill
```

Legs are subscribed and unsubscribed dynamically as workers enter and exit
positions — including multi-leg baskets (hedged pairs, the Delta-0.2 four-leg
spread, strangle legs, the SL Hunting BankNIFTY mirror).

**Why the true-up exists.** Tick-built candles and official candles can disagree
(missed packets, boundary handling, exchange corrections). Backtests were run on
official candles. Without the true-up, live bars would slowly stop being the
thing the strategies were validated against. The rule is deliberately blunt:
official always wins.

**Why the tick logic is a separate pure module.** `Dependencies/tick_bar_builder.py`
holds no sockets, no threads and no clock of its own, so bar-boundary and
out-of-order-tick behaviour is unit-testable without a live feed. It carries a
90% branch-coverage budget for the same reason the REST validators do: it feeds
the same frames.

**Rollback** is `MARKET_DATA_SOURCE=REST` plus a restart. No state migration.

---

## 4. Validation and health gates

`Dependencies/market_data_health.py` is the single authority. It is pure and
has no knowledge of threads or brokers.

### 4.1 Candle validation — `validate_ohlc_frame`

Rejects, rather than repairs:

- non-finite or non-positive prices
- broken geometry (`high < low`, close outside `[low, high]`, …)
- duplicate or non-monotonic timestamps
- incomplete trailing minutes (`complete_minute_bucket_mask`,
  `newest_completed_minute_timestamp`)

A rejected frame is **not published**. Strategies keep seeing the last good
snapshot; they never see a bad one. Naive timestamps are treated as Asia/Kolkata.

### 4.2 Freshness — `MarketDataHealth`

Three independent thresholds, each answering a different question:

| Gate | Question | Effect when breached |
|---|---|---|
| LTP staleness (~10s) | Is the price I would trade at current? | Refuse new entries |
| Bar staleness (~150s) | Is the candle stream alive? | Refuse new entries |
| Liquidation (~30s) | Has this gone on long enough that holding is worse than exiting blind? | Liquidate open positions |

The websocket producer adds one twist: a quiet-but-subscribed leg (a real,
untraded option) is legitimately silent. Its LTP is treated as fresh **only
while the socket is demonstrably alive** — otherwise a dead socket would look
identical to a quiet strike.

---

## 5. Resampling

`resample_ohlc_from_1m(ohlc, timeframe_minutes)` turns the shared 1-minute
frames into whatever timeframe a strategy wants (5-minute for CPR, and so on).

The invariant that matters: **candles are labelled by their START time**, and a
strategy acts only on *completed* candles. `Dependencies/next_open_entry.py`
encodes the consequence for `NEXT_OPEN` strategies — a signal born on a
completed candle gets exactly one bar of life and is rebased to the next
candle's open.

---

## 6. Interfaces

| Direction | Contract |
|---|---|
| Producer → store | validated frames + LTPs + health timestamps, under the lock |
| Store → worker | immutable `MarketSnapshot` / `LTPSnapshot` copies |
| Worker → store | subscribe/unsubscribe an `OptionSubscription` on entry/exit |
| Store → health | freshness timestamps; the health object answers the gates |

---

## 7. Testing

- `Tests/test_market_data_health.py` — validation and freshness rules (unittest).
- `Tests/Dependencies/test_tick_bar_builder.py` — pure tick→bar behaviour.
- `Tests/test_nifty_multi_strategy_master.py` — producer threads, store locking,
  subscription lifecycle, and the fail-closed source selection.

Both `market_data_health.py` and `tick_bar_builder.py` sit in the 90%
branch-coverage tier enforced by `scripts/check_coverage_thresholds.py`.

---

## 8. Known limitations

- **No volume in the feed.** Anything volume-weighted is a documented proxy
  (Regime Adaptive's VWAP is equal-weight — see
  [`regime-adaptive.md`](regime-adaptive.md)) or unimplemented.
- **NIFTY is assumed primary.** BankNIFTY is fetched per bar for confirmation
  and mirroring, not as a co-equal underlying. Making a second index
  first-class means parameterizing this component.
- **One producer per process.** There is no failover from websocket to REST
  mid-session; recovery is restart with the flag flipped.

---

## 9. Known failure modes

### 9.1 `did not receive a valid HTTP response`

The websocket pump reports this when `websockets` raises `InvalidMessage` — the
client opened TLS to `api-feed.dhan.co`, waited for an HTTP status line, and the
server closed the connection without sending one.

**It cannot be a credential or subscription problem**, and that is the whole
point of this entry. Dhan's feed edge completes the websocket upgrade *without
validating the token*: a deliberately invalid token connects successfully
(verified 2026-09-04). Authentication happens after the upgrade, inside the feed
protocol — so nothing about the token, the Data API subscription, or IP
whitelisting can make the handshake itself fail. Silence at this stage means the
edge is not serving upgrades, whoever is asking.

The pump logs only `str(exc)` (`nifty_multi_strategy_master.py:4456`), which
renders the bare sentence above. The useful detail is on `exc.__cause__` —
typically `EOFError('connection closed while reading HTTP status line')`.

To check the edge directly, without touching `.env` or sending real credentials:

```bash
python -c "
import asyncio, websockets
async def main():
    try:
        async with websockets.connect('wss://api-feed.dhan.co?version=2&token=DUMMY&clientId=0&authType=2', open_timeout=20):
            print('connected')
    except Exception as e:
        print(type(e).__name__, '|', e, '| cause:', repr(e.__cause__))
asyncio.run(main())
"
```

`connected` means the edge is healthy and the fault is elsewhere — the local
network path, or post-handshake authentication. `InvalidMessage` means the edge
is down and no local change will help.

**Observed 2026-09-04.** Five launches between 07:20 and 08:06 produced 26
consecutive handshake failures. REST was healthy throughout on the same
credentials — warmup history, the per-minute true-up, and the 33 MB
instrument-master download all succeeded. The failure then reproduced outside the
runner using the probe above, with no real credential in play at all, which ruled
out the token, the subscription, the `websockets` and `dhanhq` versions, and this
component's own code in one step. It cleared on Dhan's side with no local change:
the 08:53:44 launch connected five seconds later on its first attempt.

Historically this error clusters hard in the early morning: all 33 occurrences in
five months of logs fall in hours 07 and 08, and none between 09:00 and 16:00
across 180 successful connects. Before 2026-09-04 every one of them self-healed
on the very next retry.

### 9.2 Operator response

Fall back to the REST producer — `MARKET_DATA_SOURCE=REST` plus a restart, no
state migration ([ADR-0005](../adr/0005-rest-vs-websocket-market-data.md)).
This is always available: the paid subscription is an optimisation, not a
dependency, so the system runs without the feed at all.

Two things to know before deciding to leave a WEBSOCKET-mode session running
through an outage:

- **A dead feed never gets louder.** The pump retries forever with no give-up
  (`nifty_multi_strategy_master.py:4436`) and stays at WARNING for every attempt
  (`:4456`). Backoff caps at 30s and then repeats indefinitely.
- **Market data freezes rather than degrading.** The per-minute true-up is gated
  on the tick aggregator's version counter (`:4283-4285`), which cannot advance
  while no ticks arrive, so OHLC stays pinned to the warmup frame. Paper workers
  are exempt from the health gate by the 2026-07-17 decision at `:6379-6380`, so
  they keep trading on it. This is the concrete shape of the "no failover from
  websocket to REST mid-session" limitation in §8 — the practical consequence is
  that an unattended WEBSOCKET session can paper-trade a whole day against the
  previous close.

### 9.3 Reconnect churn

A feed can be reachable and still degraded. On 2026-09-04, after the morning
outage cleared, the socket dropped and recovered roughly 36 times between 09:08
and 12:00 — mostly `keepalive ping timeout` — each recovering within 1–3 seconds.
That day logged 37 connects against a normal range of 1–7.

Integrity survives this by design: every reconnect requests an immediate true-up
(`nifty_multi_strategy_master.py:4451`), and that full-window merge is the gap
backfill for the ticks missed while the socket was down. The cost is REST load
well above what ADR-0005 assumed when it traded polling for a socket, which
erodes the API-load saving that motivated the websocket producer. Sustained churn
at this level is a reason to re-read that trade-off, not a data-safety concern.
