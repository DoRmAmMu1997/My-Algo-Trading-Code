# ADR-0021: A per-strategy LIVE/PAPER toggle on the dashboard

**Status:** Accepted
**Date:** 2026-11-02
**Deciders:** repository owner
**Supersedes:** the "GET and nothing but GET" clauses of ADR-0016 §2 and ADR-0020 §8,
for this one route only

## Context

Switching a strategy between paper and live needed an `.env` edit and a restart.
During a session, that means flattening everything to change one strategy. The
operator asked for a button on the dashboard.

Three facts of the existing code shaped the design:

1. **A strategy's mode is a single attribute, `worker.live_trading`, set once at
   startup.** `_configure_startup_live_trading` only sets it after a strict
   live-config check, a broker login, a symbol preload and a flat-account audit.
   If every strategy starts on paper, the broker is never logged in at all.
2. **The order path reads `live_trading` more than once per order.**
   `_place_real_leg` reads it to decide whether to take the shared entry lease,
   and `_place_real_leg_with_entry_lease` reads it again for the paper
   short-circuit. If another thread flipped the flag between those two reads, a
   live order could go out without the lease.
3. **The dashboard was GET-only as a safety property** (ADR-0016 §2, ADR-0020
   §8), and it binds loopback with no authentication. A page on any other
   website open in the same browser can make the browser POST to
   `127.0.0.1`. The existing Host-header check would not stop that, because the
   Host header of such a request is genuinely `127.0.0.1:<port>`.

## Decision

### 1. Only the worker writes `live_trading`, only when it is flat

Each worker gets a `LiveModeSlot` (`Dependencies/live_mode_control.py`), a small
lock-guarded mailbox. The controller only ever writes a *request* into it.

`_apply_pending_live_mode_if_safe` runs from `_run_shutdown_cycle_if_requested`.
Every worker loop already calls that once per poll, on the worker's own thread
and never in the middle of an order. The request is applied there only when the
worker owns no position, has no orphan live leg and no unresolved exit. A
promotion is also re-checked against both entry freezes at that point.

While a position is open, the request stays **PENDING**, in both directions:

- **Live → paper with a live leg open** would switch off the live-only stale-feed
  liquidation for real exposure, and a later add-on would be paper on top of a
  live position.
- **Paper → live with a paper position open** would make a live add-on to a paper
  trade.

"Apply from the next entry" is exactly that rule.

### 2. Paper → live repeats the startup gate at runtime

`_live_promotion_refusal` runs on one `LiveModeController` daemon thread, never
on an HTTP thread, because a login can take ten seconds. It refuses, in fixed
wording, unless every check passes:

- **The global switch.** `LIVE_TRADING_ENABLED` must have been true **at
  startup**. The global switch stays `.env`-only. The page can arm a strategy
  only inside a session that was started with live trading allowed.
- **Live config.** `_live_config_errors` must report nothing for that strategy.
- **Freezes.** There must be no store-wide or per-worker entry freeze.
- **The broker.**
  - If nothing has gone live yet this session, it runs the startup sequence:
    login, symbol preload and the flat-account audit. That sequence is extracted
    into `_login_and_audit_live_account`, so startup and runtime cannot drift
    apart.
  - If a strategy is already live, other strategies may legitimately hold
    positions, so the flat-account audit no longer applies and only a fresh login
    check runs.

A refusal **never freezes anything**: nothing is armed, so there is nothing to
protect, and the next click runs the whole gate again.

Live → paper needs no broker call. An open live position still exits at the
broker, because exits follow the position's `live_leg`, not the worker's flag.

### 3. Exactly one POST, guarded against cross-site forgery

`DASHBOARD_LIVE_TOGGLE_ENABLED` is off by default. When it is off, nothing
changes: POST answers 405 and no button is drawn.

When it is on, `/api/live-mode` accepts a POST, but only after these checks:

- the existing Host allowlist;
- an `Origin` exactly equal to the page's own origin, and a `Sec-Fetch-Site`, if
  sent, of `same-origin`;
- `Content-Type: application/json`, which a plain form cannot send and a
  cross-site script cannot send without a preflight (OPTIONS still answers 405);
- a body of at most 1 KB, checked before it is read;
- a per-process token from `secrets.token_urlsafe(32)`, compared in constant
  time. It is readable only from `GET /api/live-mode`, which a page on another
  origin cannot read;
- the strategy name typed again (`confirm == strategy`), in both directions;
- a strategy from the frozen set this runner is running, and a target of exactly
  `LIVE` or `PAPER`.

The HTTP thread then only *queues* the request (202, or 503 if the bounded queue
is full). PUT, PATCH, DELETE, HEAD and OPTIONS still answer 405, and the bind
host is still a constant with no `.env` knob.

The page's only write lives in its own file, `dashboard_live_toggle.js`.
`dashboard.js` stays free of any write, and the existing test that pins that is
unchanged.

### 4. Session-only, never written back

A toggle lasts until the process exits. `.env` stays the single source of truth
(ADR-0008). Session-state resume already never restores a live worker.

Alerts (`LIVE_MODE_REQUESTED` / `_APPLIED` / `_REFUSED`) go straight onto the
Telegram queue, the way the startup-block alert does, and **not** through
`publish_trade_event`. A toggle is not a trade, so it must not enter the
session-state trade list, the dashboard ledger, or the PAPER/LIVE/MIXED
provenance of a Sheet row. A strategy that actually trades in both modes still
reports MIXED, from its fills.

## Consequences

- **The operator can arm a strategy mid-session without a restart**, and demote
  one without touching the broker.
- **An urgent demotion is not instant.** It waits for the strategy to be flat.
  Flattening a live position now is still the job of the strategy's own exits,
  the max-loss kill-switch, or Ctrl+C.
- **The dashboard is no longer strictly read-only when the toggle is on.** That
  is the cost of this feature, and the reason it is opt-in and documented here.
  The page still cannot place, cancel or modify an order.
- **A first promotion logs in mid-session.** If login or the audit fails, the
  strategy simply stays on paper.
