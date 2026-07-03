"""Unit tests for the pure logic in the Flattrade broker helpers.

Everything here runs offline: no login, no HTTP, no orders. The tests cover the
pieces a live session depends on but that do not need a live session to verify —
the rate limiter's budget math, scrip-master normalization, exact contract
selection, order-status parsing, and order-id extraction.

Run from the repository root::

    python -m pytest "Dependencies/Flattrade API" -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_flattrade_symbol as diag
import flattrade_execution as fe

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _FakeClock:
    """Deterministic clock + sleeper so the limiter tests never really wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, 0.0)


def _limiter(per_second: int, per_minute: int, max_wait: float, fake: _FakeClock):
    return fe._RollingWindowRateLimiter(
        per_second, per_minute, max_wait, clock=fake.clock, sleeper=fake.sleep
    )


def test_rate_limiter_grants_within_budget_without_sleeping():
    fake = _FakeClock()
    limiter = _limiter(2, 10, max_wait=1.0, fake=fake)
    limiter.acquire()
    fake.now += 0.1
    limiter.acquire()
    assert fake.slept == []


def test_rate_limiter_sleeps_for_the_per_second_window():
    fake = _FakeClock()
    limiter = _limiter(1, 10, max_wait=2.0, fake=fake)
    limiter.acquire()
    limiter.acquire()  # second call must wait out the 1-second window
    assert fake.slept and fake.slept[0] == pytest.approx(1.0)


def test_rate_limiter_raises_instead_of_waiting_past_max():
    fake = _FakeClock()
    limiter = _limiter(1, 1, max_wait=0.5, fake=fake)
    limiter.acquire()
    with pytest.raises(RuntimeError):
        limiter.acquire()  # a 60s minute-window wait exceeds max_wait=0.5s


# ---------------------------------------------------------------------------
# Scrip-master normalization + contract selection
# ---------------------------------------------------------------------------

def _raw_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Exchange": ["NFO", "NFO", "NSE", "NFO"],
            "Lotsize": [75, 75, 1, "75"],
            "Symbol": ["NIFTY", "NIFTY", "NIFTY", "nifty"],
            "Tradingsymbol": ["NIFTY14JUL26C24150", "NIFTY14JUL26P24150", "NIFTY-EQ", "NIFTY21JUL26C24150"],
            "Expiry": ["14-Jul-2026", "14-Jul-2026", "", "21-Jul-2026"],
            "Strike": ["24150.00", 24150, "", 24150],
            "Optiontype": ["CE", "PE", "", "ce"],
        }
    )


def test_prepare_scrip_master_normalizes_and_filters_to_nfo():
    frame = fe.FlattradeExecutionClient._prepare_scrip_master(_raw_master())
    assert len(frame) == 3  # the NSE equity row is dropped
    assert set(frame["_option_u"]) == {"CE", "PE"}
    assert all(frame["_lotsize_i"] == 75)


def test_prepare_scrip_master_raises_on_missing_columns():
    with pytest.raises(ValueError):
        fe.FlattradeExecutionClient._prepare_scrip_master(pd.DataFrame({"Exchange": ["NFO"]}))


def test_select_contract_picks_nearest_future_expiry_for_strike():
    client = fe.FlattradeExecutionClient()
    client._scrip_df = fe.FlattradeExecutionClient._prepare_scrip_master(_raw_master())
    expiry, symbol, lot = diag.select_contract(
        client,
        underlying="NIFTY",
        option_type="CE",
        strike=24150,
        requested_expiry=None,
        today=pd.Timestamp("2026-07-10").date(),
    )
    assert (str(expiry), symbol, lot) == ("2026-07-14", "NIFTY14JUL26C24150", 75)


def test_select_contract_raises_when_no_exact_row_exists():
    client = fe.FlattradeExecutionClient()
    client._scrip_df = fe.FlattradeExecutionClient._prepare_scrip_master(_raw_master())
    with pytest.raises(ValueError):
        diag.select_contract(
            client,
            underlying="NIFTY",
            option_type="CE",
            strike=99999,
            requested_expiry=None,
            today=pd.Timestamp("2026-07-10").date(),
        )


# ---------------------------------------------------------------------------
# Order acknowledgement / status parsing / order-id extraction
# ---------------------------------------------------------------------------

def test_is_order_ack_requires_ok_and_order_number():
    client = fe.FlattradeExecutionClient()
    assert client._is_order_ack({"stat": "Ok", "norenordno": "123"})
    assert not client._is_order_ack({"stat": "Ok"})
    assert not client._is_order_ack({"stat": "Not_Ok", "norenordno": "123"})
    assert not client._is_order_ack("Ok")


def test_order_status_parses_the_latest_ok_row(monkeypatch):
    client = fe.FlattradeExecutionClient()
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *a, **k: [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
    )
    state, filled, qty, reason = client._order_status("ORD1")
    assert (state, filled, qty, reason) == ("complete", 75, 75, "")


def test_order_status_surfaces_rejection_reason(monkeypatch):
    client = fe.FlattradeExecutionClient()
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *a, **k: [{"stat": "Ok", "status": "REJECTED", "rejreason": "RMS limit"}],
    )
    state, _filled, _qty, reason = client._order_status("ORD1")
    assert state == "rejected" and reason == "RMS limit"


def test_extract_order_id_searches_nested_payloads():
    client = fe.FlattradeExecutionClient()
    assert client.extract_order_id({"norenordno": " 42 "}) == "42"
    assert client.extract_order_id({"data": [{"order_id": "77"}]}) == "77"
    assert client.extract_order_id(None) == ""
