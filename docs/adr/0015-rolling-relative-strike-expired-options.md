# ADR-0015: Take expired-options history in rolling relative-strike form, and derive the expiry date

**Status:** Accepted
**Date:** 2026-09-05
**Deciders:** repository owner

## Context

Every backtest in this repository runs on NIFTY **index** 1-minute bars. Option
P&L is therefore simulated rather than observed, and
[`data-and-backtesting.md`](../lld/data-and-backtesting.md) §7 has long recorded
the consequences: no volume, no open interest, no implied volatility, and no way
to check a fill against a premium that actually printed.

DhanHQ closed that gap in API v2.3 (08 Sep 2025) with
`POST /v2/charts/rollingoption`: five years of minute-level expired-options data
carrying OHLC, volume, OI, IV, strike and spot. The pinned `dhanhq==2.2.0`
already exposes it as `expired_options_data()`, so no dependency moved.

The immediate motivation is a rule-based **positional option-selling** strategy,
which is precisely the use case the endpoint's shape complicates.

### What a live probe established

Documentation and third-party wrappers disagreed on `expiryCode`, so it was
resolved against the live API on 2026-09-05. Input validation runs before
authentication, which made the answer legible even with an expired token:

| Request | Response |
|---|---|
| `expiryCode: 0` | `DH-905 Input_Exception: "expiryCode is required"` |
| `expiryCode: 1` | passed input validation, reached the auth check |

So **`expiryCode` on this endpoint is 1-based**: 1 is the near expiry, 2 the
next. Dhan's annexure documents `0 = current expiry`, but that table describes
`/charts/historical`; this endpoint reads a zero as a missing field. Following
the annexure would have produced a run that failed on every single call.

### What the smoke run established

A live 8-call run over 06–10 Jan 2025 settled the rest, and corrected the
documentation on one point that had already produced a bug:

| Question | Answer |
|---|---|
| `securityId` | **13** with `NSE_FNO` + `OPTIDX` — the index-side ID, not the `26000` in the instrument master. |
| Response envelope | Doubled: `resp["data"]["data"]["ce"]`. |
| `toDate` | **Inclusive**, *not* "non-inclusive" as documented. 06→08 Jan returns three sessions ending on the 8th. |
| ATM cadence | Re-picked **every bar**, not daily (see below). |
| Field coverage | All nine fields fully populated — 1500/1500 rows for volume, OI and IV. |

The `toDate` correction was not cosmetic. Half-open chunk windows made adjacent
chunks overlap by a day, and the trailing bars then tripped the out-of-window
guard: `375 candle(s) outside the requested window 2025-01-06 -> 2025-01-17`.
The five-day smoke run had passed only because its boundary landed on a
Saturday. Windows now tile the range inclusively, with no overlap and no gap.

Two further observations, both upstream characteristics rather than defects:

- **Deep-OTM options really do print at the 0.05 tick floor** — the ATM call
  closed at exactly 0.05 on expiry day. This is the concrete reason
  `validate_ohlc_frame`, which rejects non-positive prices, is unsuitable.
- **Session length is not always 375 bars.** 20-Jan-2025 carries a 15:30 print
  on top of the usual 09:15–15:29, giving 376. A rule that assumes "the last bar
  is 15:29" will be wrong on such days.

The expiry derivation checks out against the prices: the ATM straddle's closing
value across that week ran 377.75 → 233.15 → 136.75 → **23.50** on the derived
expiry (Thursday 09-Jan), a wide margin either side of the 40-point ceiling
`--verify-expiries` uses.

## Decision

### 1. Store the data in the rolling, relative-strike form the API returns

The endpoint is addressed by `ATM`, `ATM±1` … `ATM±10` and `CALL`/`PUT`, not by
a contract. There is no security ID for an expired contract, and
`Dependencies/all_instrument *.csv` holds only live ones, so there is nothing to
map to. We keep the API's own shape: one CSV per `(strike label, option type)`.

The cost is that a rolling label **is not a tradeable instrument**, and the
smoke run showed the churn is much faster than "day to day" — it is **minute to
minute**. On 06-Jan-2025 the `ATM` call series switched strike **69 times in one
session**, and a single four-session `ATM` file held **13 distinct contracts**.
The rule is the nearest 50-point strike to spot, re-evaluated every bar:
`strike_price` matched the nearest 50 to `spot` on 1500 of 1500 rows, never more
than 25 points away.

We therefore always request the `strike` field and write it as `strike_price`,
alongside `spot`. Re-keying on `(expiry_date, strike_price, option_type)`
reconstructs fixed contracts; `strike_label` and `option_type` are written on
every row, redundantly, so that reconstruction is a concat plus a groupby.
Without `strike_price` the dataset would be worthless for anything positional.

### 2. Derive the expiry date, because the API does not return one

Without it, "sell five days out and hold to settlement" cannot even be
expressed. `Data Extractors/expiry_calendar.py` derives it from two inputs:

- a **rule table** of nominal expiry weekdays — Thursday, then Tuesday from
  01-Sep-2025, when NSE moved index derivatives off Thursday following SEBI's
  October 2024 circular; and
- the **trading days present in the downloaded data itself**, from a cheap
  hourly ATM pre-pass, rather than a hard-coded holiday list.

A week's expiry is the last trading day on or before its nominal weekday, which
gives NSE's holiday roll-back for free. Deriving the calendar from the same data
it labels means the two can never disagree.

### 3. Verify that derivation against the prices

`--verify-expiries` checks each derived expiry against the ATM straddle's last
close: on a real expiry afternoon it collapses, one session earlier it still
carries a day of time value. This is a smell test aimed at the failure mode that
matters — a calendar off by a day, or one that missed a rule change.
`--calendar-csv` overrides derivation entirely when an authoritative calendar is
available.

## Consequences

**Good**

- Backtests can read real premium, OI and IV instead of modelling them.
- The relative-strike shape is a natural fit for the existing ATM/OTM1 workers.
- No new dependency; a full five-year NIFTY backfill is ~2,772 calls.
- The download is resumable at chunk granularity, which matters on a machine
  that has hung mid-session before ([ADR-0012](0012-crash-durable-session-state.md)).

**Bad, and permanent**

- **±10 strikes is a ±500-point band** at NIFTY's 50-point spacing. A contract
  whose strike drifts further than that from spot stops appearing. A backtest
  must not read that silence as "the option expired worthless" — for a
  positional seller this is the single most likely source of a falsely
  optimistic result.
- **Expiry dates are inferred, not sourced.** Verified, but not authoritative.
- **The tail of a range is dropped.** Sessions after the last expiry inside the
  requested window cannot be labelled, so they are discarded — a backfill ending
  today loses the current part-week. `warn_about_unlabelled_tail` says so
  explicitly at the start of a run; extend `--end-date` past the next expiry to
  keep them.
- **No bid/ask.** Slippage and spread still cannot be modelled from this data,
  exactly as with the index feed.

**Deliberately not done**

- No conversion to fixed-strike contract files. The re-key is a few lines
  against data whose real behaviour we can now observe, and building it on an
  unverified expiry calendar would bake in any error the calendar has.
- `validate_ohlc_frame` is **not** reused. It rejects any non-positive price,
  which is correct for an index and wrong for a deep-OTM option printing at the
  0.05 tick floor, and it knows nothing of OI, IV, strike or spot. The engine
  carries its own validator instead.
