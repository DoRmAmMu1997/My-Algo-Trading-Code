"""Regression tests for MAT-106 deterministic strategy behavior.

These checks pin the fail-closed behavior that matters before a signal reaches
the live-order layer.  They intentionally use tiny fixed fixtures so the same
assertions run unchanged in the hosted Python 3.12 and 3.13 jobs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GENERATOR_DIR = Path(__file__).resolve().parent


def _load_module(relative_path: str, module_name: str):
    """Load a spaced-name strategy module the same way the master runner does."""

    path = GENERATOR_DIR / relative_path
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_talib_is_an_exact_mandatory_runtime_dependency():
    requirements = (GENERATOR_DIR.parent / "requirements.txt").read_text(encoding="utf-8")
    active_lines = [line.strip() for line in requirements.splitlines() if not line.lstrip().startswith("#")]

    assert "TA-Lib==0.6.8" in active_lines


def test_heikin_ashi_drops_each_invalid_timestamp():
    module = _load_module("heikin_ashi_strategy_logic.py", "mat106_heikin")
    candles = pd.DataFrame(
        {
            "timestamp": ["2026-07-16 09:15", "not-a-time", "2026-07-16 09:17"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        }
    )

    result = module.build_heikin_ashi(candles)

    assert len(result) == 2
    assert result["timestamp"].notna().all()


def test_profit_shooter_open_stop_precedes_indicator_warmup():
    module = _load_module(
        "Subhamoy Strategies/profit_shooter_strategy_logic.py",
        "mat106_profit_shooter",
    )
    config = module.ProfitShooterConfig()
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-16 09:15", periods=2, freq="min"),
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 94.0],
            "close": [100.0, 95.0],
        }
    )
    frame = module.build_profit_shooter_with_indicators(raw, config)
    position = module.ProfitShooterPositionContext(
        direction="LONG",
        entry_underlying=100.0,
        stop_underlying=95.0,
        target_underlying=107.5,
    )

    decision = module.ProfitShooterSignalEngine(config).evaluate_candle(frame, position)

    assert decision.action == "EXIT"
    assert decision.exit_reason == "STOP"
    assert decision.exit_underlying == 95.0


def test_supertrend_warmup_is_neutral_not_artificial_long():
    module = _load_module(
        "Supertrend Signal Generator Bullish.py",
        "mat106_supertrend_bullish",
    )
    high = pd.Series([101.0, 102.0, 103.0])
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([100.0, 101.0, 102.0])

    line, direction = module._compute_supertrend(high, low, close, atr_length=14, factor=3.0)

    assert np.isnan(line).all()
    assert np.equal(direction, int(module.Direction.NEUTRAL)).all()


def test_opening_strike_is_consumed_only_after_entry_acknowledgement():
    module = _load_module(
        "Nifty Opening Strike PCR VWAP ATR Signal Generator.py",
        "mat106_opening_strike",
    )
    engine = module.NiftyOpeningStrikePCRVWAPATRSignalGenerator(
        module.NiftyOpeningStrikePCRVWAPATRConfig(allow_multiple_entries=False)
    )

    first = engine._decision_for_entry(
        action="BUY_CALL",
        option_type="CE",
        selected_strike=25000,
        entry_underlying=25010.0,
        debug={},
    )
    retry_before_ack = engine._decision_for_entry(
        action="BUY_CALL",
        option_type="CE",
        selected_strike=25000,
        entry_underlying=25010.0,
        debug={},
    )

    assert first.action == "BUY_CALL"
    assert retry_before_ack.action == "BUY_CALL"
    assert engine._entry_signal_sent is False

    engine.acknowledge_entry()
    after_ack = engine._decision_for_entry(
        action="BUY_CALL",
        option_type="CE",
        selected_strike=25000,
        entry_underlying=25010.0,
        debug={},
    )
    assert after_ack.action == "NO_SIGNAL"
    assert engine._entry_signal_sent is True


@pytest.mark.parametrize(
    ("relative_path", "module_name", "config_name", "invalid_kwargs"),
    [
        ("ema_trend_strategy_logic.py", "mat106_ema_config", "EMATrendConfig", {"adx_threshold": float("nan")}),
        (
            "Subhamoy Strategies/profit_shooter_strategy_logic.py",
            "mat106_profit_config",
            "ProfitShooterConfig",
            {"tick_size": float("inf")},
        ),
        (
            "Subhamoy Strategies/goldmine_strategy_logic.py",
            "mat106_goldmine_config",
            "GoldmineStrategyConfig",
            {"engulf_tolerance": float("nan")},
        ),
        (
            "Subhamoy Strategies/money_machine_strategy_logic.py",
            "mat106_money_config",
            "MoneyMachineStrategyConfig",
            {"target_atr_multiple": float("inf")},
        ),
        (
            "CPR Strategy/cpr_strategy_logic.py",
            "mat106_cpr_config",
            "CPRStrategyConfig",
            {"trend_move_pct": float("nan")},
        ),
        (
            "CPR Strategy/Nifty CPR Algo 3 Signal Generator.py",
            "mat106_cpr3_config",
            "CPRAlgo3Config",
            {"call_arsi_min": float("nan")},
        ),
        (
            "Nifty Opening Strike PCR VWAP ATR Signal Generator.py",
            "mat106_opening_config",
            "NiftyOpeningStrikePCRVWAPATRConfig",
            {"pcr_bullish_threshold": float("nan")},
        ),
        (
            "Supertrend Signal Generator Bullish.py",
            "mat106_supertrend_config",
            "SupertrendSettings",
            {"factor": float("nan")},
        ),
        (
            "SL Hunting AI Agent/sl_hunting_indicators.py",
            "mat106_slh_config",
            "SLHuntingIndicatorConfig",
            {"cross_index_band_pct": float("inf")},
        ),
    ],
)
def test_core_strategy_configs_reject_non_finite_values(
    relative_path: str,
    module_name: str,
    config_name: str,
    invalid_kwargs: dict[str, float],
):
    module = _load_module(relative_path, module_name)

    with pytest.raises(ValueError, match="finite"):
        getattr(module, config_name)(**invalid_kwargs)


def test_core_strategy_configs_enforce_cross_field_and_percentage_invariants():
    ema = _load_module("ema_trend_strategy_logic.py", "mat106_ema_invariants")
    opening = _load_module(
        "Nifty Opening Strike PCR VWAP ATR Signal Generator.py",
        "mat106_opening_invariants",
    )
    slh = _load_module(
        "SL Hunting AI Agent/sl_hunting_indicators.py",
        "mat106_slh_invariants",
    )
    donchian = _load_module(
        "Donchian Signal Generator Bearish.py",
        "mat106_donchian_invariants",
    )

    with pytest.raises(ValueError):
        ema.EMATrendConfig(ema_fast_period=18, ema_mid_period=11, ema_slow_period=4)
    with pytest.raises(ValueError):
        opening.NiftyOpeningStrikePCRVWAPATRConfig(
            pcr_bearish_threshold=1.5,
            pcr_bullish_threshold=1.2,
        )
    with pytest.raises(ValueError):
        slh.SLHuntingIndicatorConfig(doji_body_max_ratio=1.1)
    with pytest.raises(ValueError):
        donchian.DonchianSettings(length=0)


def _fixed_ohlc_fixture(rows: int = 260) -> pd.DataFrame:
    """Return a version-independent, non-random candle fixture."""

    index = np.arange(rows, dtype=float)
    center = 24000.0 + index * 1.25 + np.sin(index / 7.0) * 12.0
    open_price = center + np.cos(index / 5.0) * 1.5
    close_price = center + np.sin(index / 3.0) * 2.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-13 09:15", periods=rows, freq="min"),
            "open": open_price,
            "high": np.maximum(open_price, close_price) + 4.0 + (index % 3) * 0.1,
            "low": np.minimum(open_price, close_price) - 4.0 - (index % 5) * 0.1,
            "close": close_price,
        }
    )


def test_fixed_fixture_indicator_and_signal_snapshot():
    """Pin exact rounded indicator values and decisions on Python 3.12/3.13."""

    candles = _fixed_ohlc_fixture()
    ema_module = _load_module("ema_trend_strategy_logic.py", "mat106_snapshot_ema")
    profit_module = _load_module(
        "Subhamoy Strategies/profit_shooter_strategy_logic.py",
        "mat106_snapshot_profit",
    )
    supertrend_module = _load_module(
        "Nifty Supertrend Signal Generator.py",
        "mat106_snapshot_supertrend",
    )
    common = _load_module("misc_strategy_common.py", "mat106_snapshot_common")
    renko_module = _load_module("renko_strategy_logic.py", "mat106_snapshot_renko")

    ema_frame = ema_module.build_ema_trend_with_indicators(candles)
    profit_frame = profit_module.build_profit_shooter_with_indicators(candles)
    supertrend_frame = supertrend_module.build_supertrend_with_indicators(candles)
    macd_line, macd_signal, macd_hist = common.macd(candles["close"])
    renko_frame = renko_module.build_renko_with_indicators(_fixed_ohlc_fixture(800))

    snapshot = {
        "ema": {
            "action": ema_module.EMATrendSignalEngine().evaluate_candle(ema_frame).action,
            "tail": [
                round(float(ema_frame[name].iloc[-1]), 8)
                for name in ("ema4", "ema11", "ema18", "atr", "adx")
            ],
        },
        "profit": {
            "action": profit_module.ProfitShooterSignalEngine().evaluate_candle(profit_frame).action,
            "tail": [
                round(float(profit_frame[name].iloc[-1]), 8)
                for name in ("sma20", "sma200", "ema9", "atr")
            ],
        },
        "supertrend": {
            "action": supertrend_module.SupertrendSignalEngine().evaluate_candle(supertrend_frame).action,
            "tail": [
                round(float(supertrend_frame["supertrend"].iloc[-1]), 8),
                int(supertrend_frame["supertrend_dir"].iloc[-1]),
            ],
        },
        "macd": [
            round(float(series.iloc[-1]), 8)
            for series in (macd_line, macd_signal, macd_hist)
        ],
        "renko": {
            "action": renko_module.RenkoSignalEngine().evaluate_candle(renko_frame).action,
            "tail": [
                round(float(renko_frame[name].iloc[-1]), 8)
                for name in ("ema5", "ema21", "ema44")
            ],
        },
    }

    assert snapshot == {
        "ema": {
            "action": "HOLD",
            "tail": [24311.28104476, 24307.85909114, 24305.48829532, 9.56402904, 84.35841828],
        },
        "profit": {
            "action": "HOLD",
            "tail": [24304.39983172, 24198.79520551, 24308.61186229, 9.56402904],
        },
        "supertrend": {"action": "HOLD", "tail": [24286.14877429, 1]},
        "macd": [5.17905935, 5.20168339, -0.02262404],
        "renko": {"action": "ENTER_LONG", "tail": [24975.0, 24875.0, 24731.25]},
    }
