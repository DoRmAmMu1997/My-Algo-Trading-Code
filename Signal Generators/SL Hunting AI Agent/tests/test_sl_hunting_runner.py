"""Tests for the standalone runner's non-trivial helpers."""

from __future__ import annotations

import pandas as pd

from sl_hunting_executor import StandaloneExecutor
from sl_hunting_runner import _manage_open_position, resample_1m_to_n


def _one_min(n=20, start_price=25000.0):
    ts = pd.date_range(start="2026-06-26 09:15", periods=n, freq="1min")
    closes = [start_price + i for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [10] * n,
        }
    )


def test_resample_1m_to_5m_aggregates_ohlc():
    df = _one_min(n=20)
    out = resample_1m_to_n(df, 5)
    # 20 one-minute bars → 4 five-minute bars.
    assert len(out) == 4
    # First 5-min bar's high should be the max of its constituent 1-min highs.
    assert out.iloc[0]["high"] == df.iloc[:5]["high"].max()
    assert set(["timestamp", "open", "high", "low", "close"]).issubset(out.columns)


def test_manage_open_position_fills_long_target_and_stop():
    ex = StandaloneExecutor()
    ex.enter("LONG", stop=24950, target=25100, reason="t", price=25000)
    # A bar whose high reaches the target fills the target.
    bar = pd.Series({"high": 25120.0, "low": 25010.0})
    _manage_open_position(ex, bar)
    assert ex.snapshot()["in_position"] is False
    assert ex.closed_trades[-1]["exit_reason"] == "target_hit"

    ex2 = StandaloneExecutor()
    ex2.enter("LONG", stop=24950, target=25100, reason="t", price=25000)
    bar2 = pd.Series({"high": 25010.0, "low": 24940.0})  # low breaches stop
    _manage_open_position(ex2, bar2)
    assert ex2.snapshot()["in_position"] is False
    assert ex2.closed_trades[-1]["exit_reason"] == "stop_hit"


def test_manage_open_position_fills_short():
    ex = StandaloneExecutor()
    ex.enter("SHORT", stop=25050, target=24900, reason="t", price=25000)
    bar = pd.Series({"high": 25010.0, "low": 24880.0})  # low reaches target
    _manage_open_position(ex, bar)
    assert ex.closed_trades[-1]["exit_reason"] == "target_hit"
