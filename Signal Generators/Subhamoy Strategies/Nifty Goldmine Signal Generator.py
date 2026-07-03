"""
NIFTY Goldmine Signal Generator.

This wrapper keeps the public file naming style used by the repo while the
actual strategy math lives in `goldmine_strategy_logic.py`.
"""

from __future__ import annotations

import pandas as pd
from goldmine_strategy_logic import (
    GoldmineDecision,
    GoldminePositionContext,
    GoldmineSignalGenerator,
    GoldmineStrategyConfig,
    generate_goldmine_signals,
    get_latest_goldmine_signal,
)


class NiftyGoldmineSignalGenerator(GoldmineSignalGenerator):
    """NIFTY-flavored wrapper around the shared Goldmine signal generator."""


def generate_nifty_goldmine_signals(
    data: pd.DataFrame,
    config: GoldmineStrategyConfig | None = None,
) -> pd.DataFrame:
    """Return full-history NIFTY Goldmine signals."""
    return generate_goldmine_signals(data, config=config)


def get_latest_nifty_goldmine_signal(
    data: pd.DataFrame,
    config: GoldmineStrategyConfig | None = None,
    position: GoldminePositionContext | None = None,
) -> GoldmineDecision:
    """Return only the newest NIFTY Goldmine decision."""
    return get_latest_goldmine_signal(data, config=config, position=position)


__all__ = [
    "NiftyGoldmineSignalGenerator",
    "generate_nifty_goldmine_signals",
    "get_latest_nifty_goldmine_signal",
]
