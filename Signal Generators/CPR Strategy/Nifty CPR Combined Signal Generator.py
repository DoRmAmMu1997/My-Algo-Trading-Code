"""
NIFTY CPR Combined Signal Generator.

Use this file when you want the full PDF strategy in one decision stream:
1. Algo 1 condition 1
2. Algo 1 condition 2
3. Algo 2 sideways zone trades
4. RSI divergence reversals

If opposite-direction setups appear on the same candle, the shared engine
returns HOLD with a conflict reason instead of guessing.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from cpr_strategy_logic import (
    CPRDecision,
    CPRPositionContext,
    CPRSignalGenerator,
    CPRStrategyConfig,
    generate_cpr_signals,
    get_latest_cpr_signal,
)


class NiftyCPRCombinedSignalGenerator(CPRSignalGenerator):
    """Full CPR PDF strategy wrapper."""

    def __init__(self, config: Optional[CPRStrategyConfig] = None) -> None:
        super().__init__(
            config=config,
            enabled_algorithms=("ALGO1", "ALGO2_ZONE", "RSI_DIVERGENCE"),
        )


def generate_nifty_cpr_combined_signals(
    data: pd.DataFrame,
    config: Optional[CPRStrategyConfig] = None,
) -> pd.DataFrame:
    """Return full-history combined CPR signals."""
    return generate_cpr_signals(
        data,
        config=config,
        enabled_algorithms=("ALGO1", "ALGO2_ZONE", "RSI_DIVERGENCE"),
    )


def get_latest_nifty_cpr_combined_signal(
    data: pd.DataFrame,
    config: Optional[CPRStrategyConfig] = None,
    position: Optional[CPRPositionContext] = None,
) -> CPRDecision:
    """Return only the newest combined CPR decision."""
    return get_latest_cpr_signal(
        data,
        config=config,
        position=position,
        enabled_algorithms=("ALGO1", "ALGO2_ZONE", "RSI_DIVERGENCE"),
    )


__all__ = [
    "NiftyCPRCombinedSignalGenerator",
    "generate_nifty_cpr_combined_signals",
    "get_latest_nifty_cpr_combined_signal",
]
