"""Small host-facing adapters for the independently calculated CPR context."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from cpr_ai_context import build_cpr_context
from cpr_ai_tools import FrozenCPRContextRegistry


def freeze_cpr_context(
    one_minute_candles: pd.DataFrame,
    *,
    position_state: dict[str, Any] | None = None,
    prior_accepted_regime: str | None = None,
    as_of: datetime | None = None,
) -> FrozenCPRContextRegistry:
    """Build and freeze one CPR context without invoking a strategy generator."""

    return FrozenCPRContextRegistry(
        build_cpr_context(
            one_minute_candles,
            position_state=position_state,
            prior_accepted_regime=prior_accepted_regime,
            as_of=as_of,
        )
    )


__all__ = ["freeze_cpr_context"]
