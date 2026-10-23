# ADR-0021: A target monitor for carried option-selling positions

**Status:** Accepted
**Date:** 2026-10-23
**Deciders:** repository owner

## Context

Beside the intraday option-buying strategies, the operator runs option-selling
strategies by hand in a separate broker account (Zerodha today, mStock
possibly). They are carried for one week to three months and adjusted along the
way. The exit rule is simple: close the whole strategy once its P&L reaches a
target set in advance. Doing that by hand means sitting at the terminal until
the number prints.

Three facts shape the design:

* **The positions are not the runner's.** They were opened by hand, in an
  account that is not `LIVE_BROKER`, so the runner has no `ExecutionLedger`
  record for them. Its exit path rejects a reducing order with no local ledger
  state (`LOCAL_LIVE_STATE_MISSING`), and that is correct for strategy exits.
* **Brokers do not remember strategies.** An adjusted strategy has already
  booked P&L on closed legs, and a broker's position book forgets that once the
  day ends. Zerodha also reports `average_price` 0 or a re-based price on some
  carried-forward rows (its own docs show 0 on a net row with
  `overnight_quantity` 3).
* **The runner already has a live Dhan price feed**, websocket or REST, with
  dynamic per-leg subscription through `SharedMarketDataStore`.

## Decision

### 1. One opt-in thread, its own broker clients, its own order path

`PositionGroupMonitor` (`Dependencies/position_group_monitor.py`) is a daemon
thread started by `main()` when `GROUP_MONITOR_ENABLED` is true (default
false). It holds its own broker clients and never touches `execution_client`,
the `ExecutionLedger`, the startup flat-account audit or any worker. Each
dependency (price reads, subscriptions, alerts, clients, the clock) is
injected, so the thread is tested without a broker, a feed or a clock.

### 2. The operator owns the groups file; the runner only reads it

`Dependencies/position_groups.json` (gitignored; the template is
`position_groups.example.json`) lists groups. Each group has a broker, a
`target`, a `booked_pnl` and legs. Each leg has a symbol, exchange, product,
signed quantity, `entry_price` and the option's identity. `python algo.py
group-legs --broker zerodha` adds newly found broker legs to `unassigned`, with
the broker's average price pre-filled for the operator to correct. It reports
changed quantities and vanished legs and never edits them. The runner
re-reads the file when it changes and never writes it. An invalid edit keeps
the last good copy watched and raises one alert. Validation is all-or-nothing.

Group P&L is `booked_pnl + Σ (ltp − entry_price) × quantity`. The operator
asked for this shape, because only they know what an adjustment booked.

### 3. Prices from the existing Dhan feed, exempt from feed health

Each leg is mapped to its Dhan contract through the Dhan instrument CSV, by
(exchange, underlying, expiry, strike, right). It is then subscribed with
`OptionSubscription(affects_feed_health=False)`. This is the one change to
shared runtime code. Before it, every subscribed leg was a required key for
`MarketDataHealth`, and that gate drives the **30-second liquidation of every
live strategy**. A watched strike with no print yet would have
force-flattened the intraday book. Health ownership is tracked per owner, so
a contract the monitor shares with a trading worker stays gated for as long
as that worker holds it. Both producers still fetch and re-stamp watched legs.
They are just not required.

A leg the CSV cannot map makes its whole group unmonitorable, with an alert.
It is never watched partially and never guessed.

### 4. Square-off rules

* The target must hold for `GROUP_MONITOR_CONFIRM_TICKS` (default 3)
  consecutive fresh ticks between 09:15 and 15:29 IST on a weekday. Any leg
  without a price no older than `GROUP_MONITOR_MAX_LTP_AGE_SECONDS` skips the
  tick.
* **Broker pre-check before any order.** Every leg must still be held with the
  same sign and at least the file's quantity. If not, the group goes to
  `EXIT_FAILED` with **no orders**, because sending the file's orders against a
  book the operator already changed could open a fresh position.
* **Shorts first.** Buy-backs go before the hedges are sold, so naked shorts
  never exist mid-exit.
* Each order is sliced under the freeze quantity from the Dhan CSV
  (`order_splitting.split_order_quantity`) and must come back `FILLED`.
  Anything else (REJECTED, PARTIAL, UNKNOWN or an exception) stops the group at
  once, alerts with what filled and what is still open, and is **never
  retried**.
* `EXITING` is persisted to `position_groups.state.json` before the first
  order. A crash mid-exit therefore reloads as `EXIT_FAILED`, never as a
  fresh trigger. `SQUARED_OFF` and `EXIT_FAILED` are terminal across restarts,
  keyed by a fingerprint of the group's name and exact leg set. Edits to price
  or target do not re-arm a group.

### 5. A narrow broker surface, one Kite-shaped client

`PositionBrokerClient` (in `broker_contract.py`) has only `ensure_logged_in`,
`list_position_details`, `place_market_order`, `get_order_status` and
`logout`, plus a new `PositionDetail` row with an average price and the
option's identity. Zerodha's Kite Connect v3 and mStock's Type-A API are the
same REST shape, so `Dependencies/position_brokers.py` implements both over
`requests` with no new dependency. Two small subclasses carry the differences:
base URL, version header, the order-status path, Kite's `market_protection=-1`
and mStock's `price=0`. The adapters' rules are kept:

* a ten-second total deadline that includes lock and rate-limit waits;
* `REJECTED` only after the order book proves no order carries the tag;
* transport, 5xx, `NetworkException` and `GeneralException` count as
  `UNKNOWN`;
* an order deadline poisons the client.

These clients are not `LIVE_BROKER` choices. `python algo.py
setup-broker-token --broker zerodha|mstock` runs the daily human login step
and writes `<BROKER>_ACCESS_TOKEN` into `.env`. The runner logs each configured
broker in at startup and alerts on failure, so an expired token is found at
09:00, not at the target.

### 6. Lifetime

The monitor starts before the fetcher and keeps watching after the
strategies' 15:15 square-off, until 15:30. It is waited for after the
unchanged end-of-day sequence (flat proof, Sheet write, clean-shutdown mark)
and before the fetcher and notifier are released. An interrupted session stops
it at once. A square-off in progress may take up to 120 s to finish.

## Deviation from the repository's live-trading convention

Everywhere else a real order needs `LIVE_TRADING_ENABLED` **and** a
per-strategy `<PREFIX>_LIVE_TRADING`. Here `GROUP_MONITOR_ENABLED` is the
**only** switch, and it ignores `LIVE_TRADING_ENABLED`. The operator asked for
this so the monitor can act on real positions while the intraday strategies
stay on paper. The remaining guards are the default-off switch, the broker
pre-check, the confirmation ticks and the per-order proof of fill.

## Consequences and known limits

* The monitor only watches while the runner is up.
* It prices by **last traded price**. On an illiquid far strike the LTP can
  sit away from the bid/ask, so a group can reach its target on paper and then
  fill worse. The confirmation ticks reduce false triggers but cannot fix
  slippage.
* MARKET orders on illiquid strikes can slip. Kite's automatic market
  protection may reject an order instead, and that stops the group with an
  alert.
* mStock does not publish its order-status vocabulary. An unrecognized state
  stays transient and ends `UNKNOWN`, which stops the group. The mStock path
  has not traded against a real account yet.
* Zerodha now rejects API orders from an unregistered static IP, and SEBI's
  retail-algo framework points the other brokers the same way. Register the
  machine's IP with each broker before relying on the monitor.
