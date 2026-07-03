"""
NIFTY CPR Algo 1 Signal Generator.

Algo 1 is the trending-market part of the CPR PDF:
- Condition 1: break out of the CPR neutral zone with EMA/VWAP/RSI support.
- Condition 2: after a 0.5% move or swing break, retrace to EMA20 and continue.

This wrapper keeps Algo 1 separate for front-tests/backtests that only want the
trend logic. The math lives in `cpr_strategy_logic.py` so all wrappers stay in
sync.
"""

from __future__ import annotations

import pandas as pd
from cpr_strategy_logic import (
    CPRDecision,
    CPRPositionContext,
    CPRSignalGenerator,
    CPRStrategyConfig,
)


class NiftyCPRAlgo1SignalGenerator(CPRSignalGenerator):
    """CPR Algo 1 only: narrow/medium CPR trend entries."""

    def __init__(self, config: CPRStrategyConfig | None = None) -> None:
        # `enabled_algorithms=("ALGO1",)` means the shared engine ignores
        # sideways zones and RSI-divergence reversal entries.
        super().__init__(config=config, enabled_algorithms=("ALGO1",))


def generate_nifty_cpr_algo1_signals(
    data: pd.DataFrame,
    config: CPRStrategyConfig | None = None,
) -> pd.DataFrame:
    """Return full-history Algo 1 signals."""
    return NiftyCPRAlgo1SignalGenerator(config=config).generate(data)


def get_latest_nifty_cpr_algo1_signal(
    data: pd.DataFrame,
    config: CPRStrategyConfig | None = None,
    position: CPRPositionContext | None = None,
) -> CPRDecision:
    """Return only the newest Algo 1 decision."""
    return NiftyCPRAlgo1SignalGenerator(config=config).latest_signal(data, position=position)


__all__ = [
    "NiftyCPRAlgo1SignalGenerator",
    "generate_nifty_cpr_algo1_signals",
    "get_latest_nifty_cpr_algo1_signal",
]
