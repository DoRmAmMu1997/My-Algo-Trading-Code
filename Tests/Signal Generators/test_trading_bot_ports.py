"""Contract tests for the 14 TradingBot-port signal generators (TEST-PORTS).

Until now these ports were exercised only indirectly, through the master
runner's factory (`_build_signal_gen_worker_class`). These tests pin the
uniform contract that factory relies on, per port and in isolation:

- the module exposes `<Name>SignalEngine`, `build_<name>_with_indicators`,
  `<Name>PositionContext` and `<Name>Config` (defaults constructible);
- `minimum_history_bars()` is a sane positive int;
- `evaluate_candle(frame)` on a warm frame returns a decision from the closed
  action set, and an ENTER decision carries usable underlying levels;
- the built frame types those columns correctly — flags boolean, levels float;
- forcing the port's own entry trigger produces an ENTER carrying exactly the
  levels the frame held;
- two fresh engines are deterministic on the same frame;
- with an open position the engine still answers from the closed action set
  (the master only honours EXIT while in a trade).

The synthetic data is generic trending-with-wobble 5-minute sessions (built
like the CPR strategy tests): full 09:15 sessions so session-aware ports
(Opening Range Breakout, Multi Timeframe) see real day boundaries.

WHY THE ENTRY CONTRACT IS FORCED RATHER THAN OBSERVED
-----------------------------------------------------
Those generic sessions only provoke a live ENTER out of 2 of the 14 ports
(Parabolic SAR and Regime Adaptive); Opening Range Breakout and Supertrend
raise no setup flag at all across the whole frame, so no amount of extra
synthetic volatility would reach their entry branch either. Leaving the entry
assertions conditional on `decision.action` therefore left 12 ports' levels
unchecked — and that is exactly how the Regime Adaptive collapse bug (stop and
target cast to `bool`, so every entry sized off a stop of 0.0 or 1.0) survived
review. The two tests below close that gap without depending on data luck:
`test_built_frame_types_the_entry_columns_correctly` catches a builder that
types a level wrongly, and `test_forced_entry_carries_the_frames_own_levels`
catches an engine that reads the wrong column on the way out.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import fields, replace
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

# Tests/Signal Generators/<this file> -> the repository root is two levels up.
GEN_DIR = Path(__file__).resolve().parents[2] / "Signal Generators"

# Ports that live in a subfolder import their siblings by bare name, so that
# folder has to be importable before `_load_port` executes them. (The master's
# `load_module` does the same thing at runtime.)
for _sub in ("Regime Adaptive Strategy",):
    _sub_dir = str(GEN_DIR / _sub)
    if _sub_dir not in sys.path:
        sys.path.insert(0, _sub_dir)
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

# (path relative to this folder, attr prefix, build-function name, optional import the port needs)
PORTS = [
    ("sma_crossover_signal_generator.py", "SMACrossover", "sma_crossover", None),
    ("bollinger_bands_signal_generator.py", "BollingerBands", "bollinger_bands", None),
    ("keltner_squeeze_signal_generator.py", "KeltnerSqueeze", "keltner_squeeze", None),
    ("mean_reversion_zscore_signal_generator.py", "MeanReversionZscore", "mean_reversion_zscore", None),
    ("ml_ensemble_signal_generator.py", "MLEnsemble", "ml_ensemble", "sklearn"),
    ("multi_timeframe_signal_generator.py", "MultiTimeframe", "multi_timeframe", None),
    ("opening_range_breakout_signal_generator.py", "OpeningRangeBreakout", "opening_range_breakout", None),
    ("parabolic_sar_signal_generator.py", "ParabolicSAR", "parabolic_sar", None),
    ("rsi_divergence_signal_generator.py", "RSIDivergence", "rsi_divergence", None),
    ("rsi_reversal_signal_generator.py", "RSIReversal", "rsi_reversal", None),
    ("stochastic_oscillator_signal_generator.py", "StochasticOscillator", "stochastic_oscillator", None),
    ("supertrend_signal_generator.py", "Supertrend", "supertrend", None),
    ("volatility_breakout_signal_generator.py", "VolatilityBreakout", "volatility_breakout", None),
    ("Regime Adaptive Strategy/regime_adaptive_signal_generator.py",
     "RegimeAdaptive", "regime_adaptive", None),
]
PORT_IDS = [prefix for (_f, prefix, _b, _d) in PORTS]

VALID_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"}

# The column contract every builder publishes and every engine reads back. The
# flags gate the entry; the levels are handed straight to the master's
# `enter_position`, which sizes the trade off the entry-to-stop distance.
SETUP_FLAG_COLUMNS = ("long_setup", "short_setup")
LEVEL_COLUMNS = (
    "long_entry_price",
    "short_entry_price",
    "long_stop_from_setup",
    "short_stop_from_setup",
    "long_target_from_setup",
    "short_target_from_setup",
)

# The one port that publishes no setup flags: ML Ensemble gates its entry on the
# model's up-probability instead. Named rather than tolerated, so a port that
# silently LOSES its flag columns still fails.
PORTS_WITHOUT_SETUP_FLAGS = {"MLEnsemble"}


@cache
def _load_port(filename: str):
    """Load one spaced-name generator module (same mechanism as the master).

    `filename` may include a subfolder ("Regime Adaptive Strategy/..."), so the
    derived module name flattens both spaces and separators.
    """
    path = GEN_DIR / filename
    assert path.exists(), f"Expected generator at {path}"
    stem = filename.replace("\\", "/").replace("/", "__").replace(" ", "_")
    name = "test_port_" + stem.removesuffix(".py").lower()
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


def _raise_setup_flag(module, engine, frame, index, side: str) -> None:
    """Default trigger: raise this side's setup flag and lower the other one.

    Every port but ML Ensemble gates its entry on `<side>_setup`, and they all
    hold rather than guess when BOTH flags are true, so the opposite side has to
    come down too.
    """
    del module, engine  # the flag lives in the frame, not on the engine
    other = "short" if side == "long" else "long"
    frame.loc[index, f"{side}_setup"] = True
    frame.loc[index, f"{other}_setup"] = False


def _raise_setup_flag_and_regime(module, engine, frame, index, side: str) -> None:
    """Regime Adaptive also refuses to trade a regime it could not measure.

    Pinning the branch keeps this test independent of whatever ADX happens to be
    on the synthetic frame's last bar.
    """
    frame.loc[index, "regime"] = module.MEAN_REVERSION_BRANCH
    _raise_setup_flag(module, engine, frame, index, side)


def _pin_ml_probability(module, engine, frame, index, side: str) -> None:
    """ML Ensemble has no setup flag -- its entry gate is the model itself.

    Stubbing the two model steps on this throwaway engine is the only seam into
    the entry branch, and it keeps the test about the LEVEL plumbing rather than
    about what scikit-learn happens to learn from synthetic bars.
    """
    del module, frame, index
    probability = 0.99 if side == "long" else 0.01
    engine._maybe_train = lambda _frame: None
    engine._predict_up_probability = lambda _row: probability


# Only the two exceptions need an entry; everything else uses the flag default.
ENTRY_TRIGGERS = {
    "MLEnsemble": _pin_ml_probability,
    "RegimeAdaptive": _raise_setup_flag_and_regime,
}

# Levels ordered the way every port validates them before entering: stop below
# entry below target for a long, mirrored for a short.
FORCED_LEVELS = {
    "long": (25000.0, 24800.0, 25300.0),
    "short": (25000.0, 25200.0, 24700.0),
}


def test_ml_training_discards_infinite_feature_rows():
    """scikit-learn must never receive infinity from malformed market data."""
    pytest.importorskip("sklearn")
    module = _load_port("ml_ensemble_signal_generator.py")
    config = module.MLEnsembleConfig(
        training_window=8,
        min_training_rows=4,
        forward_bars=1,
        retrain_every=1,
    )
    engine = module.MLEnsembleSignalEngine(config)
    rows = 8
    frame = pd.DataFrame({column: [float(index + 1) for index in range(rows)]
                          for column in module.FEATURE_COLUMNS})
    frame["ml_target"] = [0, 1, 0, 1, 0, 1, 0, 1]
    frame.loc[2, module.FEATURE_COLUMNS[0]] = float("inf")

    engine._maybe_train(frame)

    assert engine.model is not None


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_port_exposes_the_factory_contract(filename, prefix, build_name, needs):
    """The master's worker factory looks these attributes up by name.

    Deliberately does NOT skip on `needs`: the ML Ensemble module imports
    scikit-learn lazily (only when it trains), so its class/function NAMES are
    present without the optional dep. Skipping here would leave the ML entry in
    the master's factory table unguarded against name regressions in CI, which
    has no scikit-learn (Codex PR #46). Only the tests that actually
    construct/evaluate the engine (`_port_under_test`) keep the dependency skip.
    """
    module = _load_port(filename)
    for attr in (f"{prefix}SignalEngine", f"build_{build_name}_with_indicators",
                 f"{prefix}PositionContext", f"{prefix}Config"):
        assert hasattr(module, attr), f"{filename} is missing {attr}"


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_config_rejects_non_finite_numeric_values(filename, prefix, build_name, needs):
    """NaN must never slip through a comparison and become live configuration."""

    module = _load_port(filename)
    config = getattr(module, f"{prefix}Config")()
    float_field = next(
        field.name
        for field in fields(config)
        if isinstance(getattr(config, field.name), float)
    )

    with pytest.raises(ValueError, match="finite"):
        replace(config, **{float_field: float("nan")})


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_config_rejects_percentage_at_or_above_one(filename, prefix, build_name, needs):
    """Decimal percentage settings must remain below 100% (1.0)."""

    module = _load_port(filename)
    config = getattr(module, f"{prefix}Config")()
    percentage_field = next(field.name for field in fields(config) if field.name.endswith("_pct"))

    with pytest.raises(ValueError, match="percentage"):
        replace(config, **{percentage_field: 1.0})


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_flat_evaluation_returns_a_valid_decision(filename, prefix, build_name, needs):
    _module, _config, engine, frame = _port_under_test(filename, prefix, build_name, needs)
    decision = engine.evaluate_candle(frame, position=None)
    assert decision.action in VALID_ACTIONS
    if decision.action in ("ENTER_LONG", "ENTER_SHORT"):
        # The master feeds these straight into enter_position -- they must be
        # real underlying levels, with the stop on a different level to entry.
        # Only 2 of the 14 ports actually reach here on this frame, which is why
        # the two tests below force the same contract for every port.
        assert float(decision.entry_underlying) > 0
        assert float(decision.stop_underlying) > 0
        assert decision.stop_underlying != decision.entry_underlying


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_built_frame_types_the_entry_columns_correctly(filename, prefix, build_name, needs):
    """Flags must stay boolean and levels must stay float, in every port.

    A level that arrives as a bool still reads as a number downstream: it just
    becomes 0.0 or 1.0, so `enter_position` sizes the trade off a stop that is a
    rupee away from an index at 25000. That is the Regime Adaptive collapse bug,
    and it is invisible to any test that only inspects decisions.
    """
    _module, _config, _engine, frame = _port_under_test(filename, prefix, build_name, needs)

    expect_flags = prefix not in PORTS_WITHOUT_SETUP_FLAGS
    for column in SETUP_FLAG_COLUMNS:
        assert (column in frame.columns) is expect_flags, f"{prefix}: unexpected {column}"
        if expect_flags:
            assert frame[column].dtype == bool, f"{prefix}: {column} must stay a flag"

    for column in LEVEL_COLUMNS:
        assert column in frame.columns, f"{prefix} is missing {column}"
        assert frame[column].dtype == float, f"{prefix}: {column} must stay a price level"


@pytest.mark.parametrize(("filename", "prefix", "build_name", "needs"), PORTS, ids=PORT_IDS)
def test_forced_entry_carries_the_frames_own_levels(filename, prefix, build_name, needs):
    """Force each port's entry trigger and check the levels survive the trip.

    The values are deliberately distinct from anything the synthetic frame could
    produce, so reading the wrong column -- the opposite side's, or the stop
    where the target belongs -- cannot coincidentally pass.
    """
    module, _config, engine, frame = _port_under_test(filename, prefix, build_name, needs)
    index = frame.index[-1]
    trigger = ENTRY_TRIGGERS.get(prefix, _raise_setup_flag)

    for side, action in (("long", "ENTER_LONG"), ("short", "ENTER_SHORT")):
        entry, stop, target = FORCED_LEVELS[side]
        trigger(module, engine, frame, index, side)
        frame.loc[index, f"{side}_entry_price"] = entry
        frame.loc[index, f"{side}_stop_from_setup"] = stop
        frame.loc[index, f"{side}_target_from_setup"] = target

        decision = engine.evaluate_candle(frame, position=None)

        assert decision.action == action, f"{prefix}: {side} trigger did not enter"
        assert decision.signal_triggered
        assert decision.entry_underlying == pytest.approx(entry)
        assert decision.stop_underlying == pytest.approx(stop)
        assert decision.target_underlying == pytest.approx(target)


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
