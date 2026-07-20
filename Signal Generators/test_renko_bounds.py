"""Regression tests for deterministic Renko memory/CPU expansion limits."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

GEN_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "filename",
    ["renko_strategy_logic.py", "renko_strategy_logic_9_21.py"],
)
def test_extreme_single_candle_move_fails_closed_with_bounded_expansion(filename: str) -> None:
    path = GEN_DIR / filename
    module_name = f"test_bound_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-16 09:15", "2026-07-16 09:16"]),
            "close": [100.0, 1_000_000.0],
        }
    )

    result = module.build_renko_from_close(frame, box_size=1.0)

    assert result.empty


@pytest.mark.parametrize(
    "filename",
    ["renko_strategy_logic.py", "renko_strategy_logic_9_21.py"],
)
def test_total_build_expansion_is_also_bounded(filename: str) -> None:
    path = GEN_DIR / filename
    module_name = f"test_total_bound_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.MAX_RENKO_BRICKS_PER_BUILD = 5
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-16 09:15", periods=5, freq="1min"),
            "close": [100.0, 102.0, 104.0, 106.0, 108.0],
        }
    )

    result = module.build_renko_from_close(frame, box_size=1.0)

    assert result.empty
