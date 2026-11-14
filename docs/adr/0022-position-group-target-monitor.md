# ADR-0022: A target monitor for carried option-selling positions

**Status:** Accepted
**Date:** 2026-10-23 (revised 2026-11-13: section 5 rewritten, the bid/ask
re-check and the review fixes in section 4, the end-of-day logout in section 6)
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

### 1. One opt-in thread with its own order path

`PositionGroupMonitor` (`Dependencies/position_group_monitor.py`) is a daemon
thread started by `main()` when `GROUP_MONITOR_ENABLED` is true (default
false). It closes positions through the Zerodha / mStock adapters (section 5)
and never touches the `ExecutionLedger`, the startup flat-account audit or any
worker: the positions were opened by hand, so the runner's books have no
record of them. Each
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
  consecutive fresh ticks on a weekday. Any leg without a price no older than
  `GROUP_MONITOR_MAX_LTP_AGE_SECONDS` skips the tick.
* **Then the group is re-priced at the live bid/ask, once, before any order.**
  A last traded price is a memory of the last trade: on a quiet far strike one
  odd print stays "the price" until the next trade, the websocket producer
  keeps re-stamping it as fresh, and the REST producer refreshes every two
  seconds against a one-second poll -- so three "consecutive fresh ticks" can
  be one single observation. The ticks only filter a momentary glitch. The
  decision is made on Dhan's `/optionchain` quote (the same fields the spread
  gate reads): shorts are valued at the ask, longs at the bid, and the group is
  squared off only if THAT P&L is at or above the target. A missing, zero or
  crossed quote on any leg sends no orders, alerts once
  (`GROUP_QUOTE_UNREADABLE`) and is re-checked after 15 seconds. Verified for
  index options; a stock option's underlying segment is a best guess, and a
  wrong guess means "no quote", never an order.
* **An exit may start only between 09:15 and 15:25 IST.** A group is several
  orders, each of which can take seconds to prove; starting at 15:29 could buy
  the shorts back and have the hedge sales refused at the close. The clock is
  read again after the quote re-check and the position-book query, right
  before `EXITING` is written, so slow pre-checks cannot start an exit late.
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
  order, and **no order is sent if that write fails** (the group stays
  `ACTIVE`, alerts `GROUP_STATE_UNWRITABLE` and retries). A crash mid-exit
  therefore reloads as `EXIT_FAILED`, never as a fresh trigger. `SQUARED_OFF`
  and `EXIT_FAILED` are terminal across restarts, keyed by a fingerprint of
  the group's name and exact leg set. Edits to price or target do not re-arm
  a group. Correcting a leg's underlying, expiry, strike or right keeps the
  status too, but looks its Dhan contract up again and moves the feed over.
* **A state file that exists but cannot be read fails closed.** Starting empty
  would re-arm groups already closed, and the pre-check above accepts a
  holding larger than the group's share, so it cannot always stop a second
  exit. The file is left in place, prices are still watched, one
  `GROUP_STATE_UNREADABLE` alert is sent and **no group is squared off that
  session**. The operator repairs the file, or deletes it to reset, and
  restarts.
* Each order's tag carries the attempt's time (`GM` + fingerprint + `HHMMSS` +
  order number, 19 letters and digits). The tag is how a lost acknowledgement
  is found again in the broker's order book, so a second attempt on the same
  group must never share one with the first.
* A groups file that parses but cannot be applied (the Dhan contract list
  unreadable, say) is **not** remembered as loaded: it alerts once
  (`GROUP_MONITOR_ERROR`) and is retried every 30 seconds. The contract list is
  read only for the underlyings the file names and is not kept in memory.

### 5. Ordinary broker adapters, also selectable as `LIVE_BROKER`

The first version of this ADR gave the monitor a narrow five-method protocol
and one shared module with a Kite-shaped base class. That was a second way of
doing brokers in a repository that already had one, and it left real gaps: no
`cancel_order`, no open-order list, and no way to clear a poisoned session.

Zerodha and mStock are now adapters like the other four
(ADR-0002): `Dependencies/Zerodha API/zerodha_execution.py` and
`Dependencies/Mstock API/mstock_execution.py`, each a self-contained client
modelled on the Flattrade adapter, with a module singleton, the whole
`ExecutionClient` surface, a `diagnose_<broker>_symbol.py`, mirrored tests and an
80% coverage floor. The Kite-shaped plumbing is duplicated between the two
files, exactly as the Noren-shaped plumbing is between Shoonya and Flattrade:
a broker's quirks stay in that broker's file.

They add one method, `list_position_details` (`PositionDetailClient` in
`broker_contract.py`), returning `PositionDetail` rows with an average price and
the option's identity. It reads the already-cached contract list and never
downloads: the monitor calls it at the moment a target is hit. Startup logs each
configured broker in and calls `preload_scrip_master()`.

Rules the two adapters keep or add:

* a ten-second total deadline that includes lock and rate-limit waits; any call
  that outlives it poisons the session until `recover_after_reconciliation`;
* `REJECTED` only after a 4xx refusal AND an order-book search that finds no
  order with the tag; transport errors, 5xx, 429, `NetworkException` and
  `GeneralException` are `UNKNOWN`;
* Kite's `tag` is alphanumeric, at most 20 characters, so the execution
  ledger's hyphenated tag is squeezed before it is sent and before it is
  searched for;
* login lives in the adapter (validate the `.env` token, else the interactive
  flow) and the interactive flow runs **on the main thread only** -- a worker or
  the monitor thread gets `False`, never a hidden prompt;
* `logout()` is local, because the remote logout ends the token for the day.

`python algo.py setup-broker-token --broker zerodha|mstock` is now a thin wrapper:
it runs the adapter's login and writes `<BROKER>_ACCESS_TOKEN` into `.env`, so
the runner starts without a prompt and a mid-day restart needs no second OTP.

**`LIVE_BROKER=ZERODHA` and `LIVE_BROKER=MSTOCK` are valid.** The monitor's
clients are the same singletons, so a shared broker is one session, one order
gate and one poison flag -- as it should be for one account. One consequence
is deliberate and unchanged: the startup flat-account audit blocks live
trading while the selected account holds any NIFTY/BANKNIFTY position, so
`LIVE_BROKER=ZERODHA` will not arm while carried positions sit in that same
account. Teaching the audit to ignore legs declared in the groups file would
be a separate decision.

The mStock adapter was written from documentation that contradicts itself and
has not traded against a real account. Each doubt is resolved toward
`UNKNOWN`: the order-details fields are sent as both query and body, a
terminal row in an order's history wins wherever it sits, an error body on
HTTP 200 is not a refusal, and an unrecognized status word stays transient.

### 6. Lifetime

The monitor starts before the fetcher and keeps watching after the
strategies' 15:15 square-off, until 15:30. It is waited for after the
unchanged end-of-day sequence (flat proof, Sheet write, clean-shutdown mark)
and before the fetcher and notifier are released -- and before the dashboard
stops, so the page stays up while anything can still place an order. An
interrupted session stops it at once. A square-off in progress may take up to
120 s to finish. Exits may start only until 15:25 (section 4); the last five
minutes are watch-only.

The end-of-day sequence logs the live broker out at about 15:16. When the
monitor is still running on that same session the logout is skipped, and the
monitor's sessions are logged out once its thread has stopped. A monitor that
will not stop keeps its session: it may be mid-order.

## Deviation from the repository's live-trading convention

Everywhere else a real order needs `LIVE_TRADING_ENABLED` **and** a
per-strategy `<PREFIX>_LIVE_TRADING`. Here `GROUP_MONITOR_ENABLED` is the
**only** switch, and it ignores `LIVE_TRADING_ENABLED`. The operator asked for
this so the monitor can act on real positions while the intraday strategies
stay on paper. The remaining guards are the default-off switch, the broker
pre-check, the bid/ask re-check and the per-order proof of fill.

## Consequences and known limits

* The monitor only watches while the runner is up.
* It WATCHES by **last traded price** but DECIDES on the bid/ask. A group whose
  LTP P&L is at target while its executable P&L is not simply waits; nothing
  alerts, because nothing is wrong. The re-check costs one option-chain call
  per underlying and expiry, at most every 15 seconds while it is failing.
  Between that quote and the fills the market can still move.
* MARKET orders on illiquid strikes can slip. Kite's automatic market
  protection may reject an order instead, and that stops the group with an
  alert.
* mStock does not publish its order-status vocabulary. An unrecognized state
  stays transient and ends `UNKNOWN`, which stops the group. The mStock path
  has not traded against a real account yet, and neither adapter has made a
  real broker call: run `python algo.py diagnose --broker <name>` and a one-lot
  round trip first.
* Zerodha now rejects API orders from an unregistered static IP, and SEBI's
  retail-algo framework points the other brokers the same way. Register the
  machine's IP with each broker before relying on the monitor.
