"""Contract tests for the 13 TradingBot-port signal generators (TEST-PORTS).

Until now these ports were exercised only indirectly, through the master
runner's factory (`_build_signal_gen_worker_class`). These tests pin the
uniform contract that factory relies on, per port and in isolation:

- the module exposes `<Name>SignalEngine`, `build_<name>_with_indicators`,
  `<Name>PositionContext` and `<Name>Config` (defaults constructible);
- `minimum_history_bars()` is a sane positive int;
- `evaluate_candle(frame)` on a warm frame returns a decision from the closed
  action set, and an ENTER decision carries usable underlying levels;
- two fresh engines are deterministic on the same frame;
- with an open position the engine still answers from the closed action set
  (the master only honours EXIT while in a trade).

The synthetic data is generic trending-with-wobble 5-minute sessions (built
like the CPR strategy tests): full 09:15 sessions so session-aware ports
(Opening Range Breakout, Multi Timeframe) see real day boundaries.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

GEN_DIR = Path(__file__).resolve().parent

# (file name, attr prefix, build-function name, optional import the port needs)
PORTS = [
    ("Nifty SMA Crossover Signal Generator.py", "SMACrossover", "sma_crossover", None),
    ("Nifty Bollinger Bands Signal Generator.py", "BollingerBands", "bollinger_bands", None),
    ("Nifty Keltner Squeeze Signal Generator.py", "KeltnerSqueeze", "keltner_squeeze", None),
    ("Nifty Mean Reversion Zscore Signal Generator.py", "MeanReversionZscore", "mean_reversion_zscore", None),
    ("Nifty ML Ensemble Signal Generator.py", "MLEnsemble", "ml_ensemble", "sklearn"),
    ("Nifty Multi Timeframe Signal Generator.py", "MultiTimeframe", "multi_timeframe", None),
    ("Nifty Opening Range Breakout Signal Generator.py", "OpeningRangeBreakout", "opening_range_breakout", None),
    ("Nifty Parabolic SAR Signal Generator.py", "ParabolicSAR", "parabolic_sar", None),
    ("Nifty RSI Divergence Signal Generator.py", "RSIDivergence", "rsi_divergence", None),
    ("Nifty RSI Reversal Signal Generator.py", "RSIReversal", "rsi_reversal", None),
    ("Nifty Stochastic Oscillator Signal Generator.py", "StochasticOscillator", "stochastic_oscillator", None),
    ("Nifty Supertrend Signal Generator.py", "Supertrend", "supertrend", None),
    ("Nifty Volatility Breakout Signal Generator.py", "VolatilityBreakout", "volatility_breakout", None),
]
PORT_IDS = [prefix for (_f, prefix, _b, _d) in PORTS]

VALID_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"}


@cache
def _load_port(filename: str):
    """Load one spaced-name generator module (same mechanism as the master)."""
    path = GEN_DIR / filename
    assert path.exists(), f"Expected generator at {path}"
    name = "test_port_" + filename.replace(" ", "_").removesuffix(".py").lower()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_sessions(min_rows: int) -> pd.DataFrame:
    """Full 5-minute trading sessions (09:15, 75 bars/day), gently trending with
    a sine wobble so no indicator sees degenerate flat data."""
    rows: list[dict] = []
    day_start = pd.Timestamp("2026-06-01 09:15")
    price = 25000.0
    day = 0
    while len(rows) < min_rows:
        session_open = day_start + pd.Timedelta(days=day)
        day += 1
        if session_open.dayofweek >= 5:  # skip weekends like a real calendar
            continue
        for i in range(75):
            base = price + math.sin(i / 5.0) * 20.0
            close = base + 4.0
            rows.append({
                "timestamp": session_open + pd.Timedelta(minutes=5 * i),
                "open": base,
                "high": max(base, close) + 6.0,
                "low": min(base, close) - 6.0,
                "close": close,
                "volume": 1000 + i,
            })
            price += 1.5
    return pd.DataFrame(rows)


def _port_under_test(filename: str, prefix: str, build_name: str, needs: str | None):
    """Load the module and return (module, config, engine, warm strategy frame)."""
    if needs:
        pytest.importorskip(needs)
    module = _load_port(filename)
    config = getattr(module, f"{prefix}Config")()
    engine = getattr(module, f"{prefix}SignalEngine")(config)
    min_bars = engine.minimum_history_bars()
    assert isinstance(min_bars, int) and 0 < min_bars < 5000
    ohlc = _make_sessions(min_bars + 80)
    frame = getattr(module, f"build_{build_name}_with_indicators")(ohlc, config)
    assert len(frame) >= min_bars
    return module, config, engine, frame


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_port_exposes_the_factory_contract(filename, prefix, build_name, needs):
    """The master's worker factory looks these attributes up by name."""
    if needs:
        pytest.importorskip(needs)
    module = _load_port(filename)
    for attr in (f"{prefix}SignalEngine", f"build_{build_name}_with_indicators",
                 f"{prefix}PositionContext", f"{prefix}Config"):
        assert hasattr(module, attr), f"{filename} is missing {attr}"


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_flat_evaluation_returns_a_valid_decision(filename, prefix, build_name, needs):
    _module, _config, engine, frame = _port_under_test(filename, prefix, build_name, needs)
    decision = engine.evaluate_candle(frame, position=None)
    assert decision.action in VALID_ACTIONS
    if decision.action in ("ENTER_LONG", "ENTER_SHORT"):
        # The master feeds these straight into enter_position -- they must be
        # real underlying levels, with the stop on a different level to entry.
        assert float(decision.entry_underlying) > 0
        assert float(decision.stop_underlying) > 0
        assert decision.stop_underlying != decision.entry_underlying


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_two_fresh_engines_agree_on_the_same_frame(filename, prefix, build_name, needs):
    module, config, engine_a, frame = _port_under_test(filename, prefix, build_name, needs)
    engine_b = getattr(module, f"{prefix}SignalEngine")(config)
    first = engine_a.evaluate_candle(frame, position=None)
    second = engine_b.evaluate_candle(frame, position=None)
    assert first.action == second.action


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_in_position_evaluation_stays_in_the_closed_action_set(filename, prefix, build_name, needs):
    module, _config, engine, frame = _port_under_test(filename, prefix, build_name, needs)
    last_close = float(frame["close"].iloc[-1])
    position = getattr(module, f"{prefix}PositionContext")(
        direction="LONG",
        entry_underlying=last_close - 50.0,
        stop_underlying=last_close - 120.0,
        target_underlying=last_close + 150.0,
    )
    decision = engine.evaluate_candle(frame, position=position)
    # The master only honours EXIT while in a trade; anything else is a hold.
    assert decision.action in VALID_ACTIONS
