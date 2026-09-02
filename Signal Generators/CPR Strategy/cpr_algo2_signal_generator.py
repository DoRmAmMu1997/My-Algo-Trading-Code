"""
NIFTY CPR Algo 2 Signal Generator.

Algo 2 is the sideways/reversal part of the CPR PDF:
- trade from support/resistance zones that have been touched at least twice
- trade RSI divergence reversals between confirmed swing points

The rules are intentionally coded as explicit defaults in `CPRStrategyConfig`
so the behavior is repeatable in both backtests and future live/front tests.
"""

from __future__ import annotations

import pandas as pd
from cpr_strategy_logic import (
    CPRDecision,
    CPRPositionContext,
    CPRSignalGenerator,
    CPRStrategyConfig,
)


class NiftyCPRAlgo2SignalGenerator(CPRSignalGenerator):
    """CPR Algo 2 only: sideways zones plus RSI divergence reversals."""

    def __init__(self, config: CPRStrategyConfig | None = None) -> None:
        super().__init__(config=config, enabled_algorithms=("ALGO2_ZONE", "RSI_DIVERGENCE"))


def generate_nifty_cpr_algo2_signals(
    data: pd.DataFrame,
    config: CPRStrategyConfig | None = None,
) -> pd.DataFrame:
    """Return full-history Algo 2 signals."""
    return NiftyCPRAlgo2SignalGenerator(config=config).generate(data)


def get_latest_nifty_cpr_algo2_signal(
    data: pd.DataFrame,
    config: CPRStrategyConfig | None = None,
    position: CPRPositionContext | None = None,
) -> CPRDecision:
    """Return only the newest Algo 2 decision."""
    return NiftyCPRAlgo2SignalGenerator(config=config).latest_signal(data, position=position)


__all__ = [
    "NiftyCPRAlgo2SignalGenerator",
    "generate_nifty_cpr_algo2_signals",
    "get_latest_nifty_cpr_algo2_signal",
]
