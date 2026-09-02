"""
NIFTY Money Machine Signal Generator.

This wrapper keeps the public file naming style used by the repo while the
actual strategy math lives in `money_machine_strategy_logic.py`.
"""

from __future__ import annotations

import pandas as pd
from money_machine_strategy_logic import (
    MoneyMachineDecision,
    MoneyMachinePositionContext,
    MoneyMachineSignalGenerator,
    MoneyMachineStrategyConfig,
    generate_money_machine_signals,
    get_latest_money_machine_signal,
)


class NiftyMoneyMachineSignalGenerator(MoneyMachineSignalGenerator):
    """NIFTY-flavored wrapper around the shared Money Machine signal generator."""


def generate_nifty_money_machine_signals(
    data: pd.DataFrame,
    config: MoneyMachineStrategyConfig | None = None,
) -> pd.DataFrame:
    """Return full-history NIFTY Money Machine signals."""
    return generate_money_machine_signals(data, config=config)


def get_latest_nifty_money_machine_signal(
    data: pd.DataFrame,
    config: MoneyMachineStrategyConfig | None = None,
    position: MoneyMachinePositionContext | None = None,
) -> MoneyMachineDecision:
    """Return only the newest NIFTY Money Machine decision."""
    return get_latest_money_machine_signal(data, config=config, position=position)


__all__ = [
    "NiftyMoneyMachineSignalGenerator",
    "generate_nifty_money_machine_signals",
    "get_latest_nifty_money_machine_signal",
]
