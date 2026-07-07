"""Regression test for the DhanHQ client construction in the shared fetcher.

dhanhq >= 2.1 (pinned 2.2.0) constructs from a `DhanContext(client_id,
access_token)`, not two positional args. The old `dhanhq(client_id,
access_token)` form raises `TypeError` on a fresh install before any OHLC is
downloaded (Codex P1 on the DEPS-001 pin). This locks the DhanContext path
without touching the network.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parent / "index_1m_5y_data_fetch_dhan_common.py"
spec = importlib.util.spec_from_file_location("index_1m_5y_data_fetch_dhan_common", MODULE_PATH)
fetcher = importlib.util.module_from_spec(spec)
sys.modules["index_1m_5y_data_fetch_dhan_common"] = fetcher
spec.loader.exec_module(fetcher)


def _args():
    return SimpleNamespace(
        client_id="CLIENT123", access_token="TOKEN456",
        exchange_segment="IDX_I", security_id=13, instrument_type="INDEX",
        interval=1, chunk_days=5, sleep_seconds=0,
    )


def test_fetch_builds_dhan_context_not_two_positional_args():
    defaults = SimpleNamespace(display_name="NIFTY")
    fake_context = object()
    with (
        patch.object(fetcher, "DhanContext", return_value=fake_context) as ctx,
        patch.object(fetcher, "dhanhq") as dhan_ctor,
        patch.object(fetcher, "resolve_date_range", return_value=(date(2026, 1, 1), date(2026, 1, 1))),
        patch.object(fetcher, "fetch_chunk", return_value=pd.DataFrame()) as chunk,
    ):
        result = fetcher.fetch_1m_history(_args(), defaults)

    # The SDK is built from a DhanContext(client_id, access_token) ...
    ctx.assert_called_once_with("CLIENT123", "TOKEN456")
    # ... and dhanhq() receives that context object, never two positional args.
    dhan_ctor.assert_called_once_with(fake_context)
    chunk.assert_called_once()
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
