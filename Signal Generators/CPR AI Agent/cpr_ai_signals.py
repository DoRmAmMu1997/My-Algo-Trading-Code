"""Provide the master worker with one small context-building entry point.

This facade keeps the master runner unaware of indicator implementation and
snapshot serialization details.  It calls only this independent CPR AI package;
the three legacy CPR strategy generators are not imported or consulted.
"""

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
    """Calculate and freeze all four tool payloads for one decision turn.

    ``as_of`` gives the host one authoritative completed-minute cutoff.  The
    same cutoff is used for calculations and for the immutable snapshot, so a
    forming candle cannot slip into one layer but not the other.
    """

    # Build first, then serialize immediately.  Codex never receives live
    # references to the shared DataFrame or position dictionary.
    return FrozenCPRContextRegistry(
        build_cpr_context(
            one_minute_candles,
            position_state=position_state,
            prior_accepted_regime=prior_accepted_regime,
            as_of=as_of,
        )
    )


__all__ = ["freeze_cpr_context"]
