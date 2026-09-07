"""Regression test for the DhanHQ client construction in the shared fetcher.

dhanhq >= 2.1 (pinned 2.2.0) constructs from a `DhanContext(client_id,
access_token)`, not two positional args. The old `dhanhq(client_id,
access_token)` form raises `TypeError` on a fresh install before any OHLC is
downloaded (Codex P1 on the DEPS-001 pin). This locks the DhanContext path
without touching the network.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess  # nosec B404 - launching the wrapper as a script IS what this test asserts
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

# Tests/Data Extractors/<this file> -> the repository root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = _REPO_ROOT / "Data Extractors" / "index_1m_5y_data_fetch_dhan_common.py"
spec = importlib.util.spec_from_file_location("index_1m_5y_data_fetch_dhan_common", MODULE_PATH)
fetcher = importlib.util.module_from_spec(spec)
sys.modules["index_1m_5y_data_fetch_dhan_common"] = fetcher
spec.loader.exec_module(fetcher)


def _args():
    # Built from a dict (not keyword args) so Bandit's B106 doesn't read the
    # dummy `access_token` literal as a hardcoded password.
    fields = {
        "client_id": "CLIENT123", "access_token": "dummy-token",
        "exchange_segment": "IDX_I", "security_id": 13, "instrument_type": "INDEX",
        "interval": 1, "chunk_days": 5, "sleep_seconds": 0,
    }
    return SimpleNamespace(**fields)


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
    ctx.assert_called_once_with("CLIENT123", "dummy-token")
    # ... and dhanhq() receives that context object, never two positional args.
    dhan_ctor.assert_called_once_with(fake_context)
    chunk.assert_called_once()
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_access_token_is_environment_only_never_a_cli_flag(monkeypatch):
    """MAT-108: a secret typed on the command line lands in shell history."""

    defaults = fetcher.IndexFetchDefaults(
        display_name="NIFTY",
        security_id="13",
        default_output="out.csv",
    )
    # Hermetic: the engine now load_dotenv()s Dependencies/.env at import, so
    # every key this assertion depends on is pinned here. Leaving one unset
    # would let a real token decide the result -- and print it on failure.
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "env-only-token")
    monkeypatch.delenv("DHAN_TOKEN_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["fetcher"])

    args = fetcher.parse_args(defaults)
    assert args.access_token == "env-only-token"

    # The old --access-token flag must be gone: argparse rejects it (exit 2).
    monkeypatch.setattr(sys, "argv", ["fetcher", "--access-token", "cli-token"])
    with pytest.raises(SystemExit):
        fetcher.parse_args(defaults)


def _payload(timestamps, *, close=None):
    count = len(timestamps)
    return {
        "timestamp": timestamps,
        "open": [100.0] * count,
        "high": [102.0] * count,
        "low": [99.0] * count,
        "close": close or [101.0] * count,
        "volume": [1000.0] * count,
    }


def test_normalize_rejects_mixed_epoch_units_and_non_finite_candles():
    second = int(datetime(2026, 1, 1, 9, 15).timestamp())
    with pytest.raises(fetcher.MarketDataValidationError):
        fetcher.normalize_response_data(_payload([second, (second + 60) * 1000]))

    with pytest.raises(fetcher.MarketDataValidationError):
        fetcher.normalize_response_data(
            _payload([second, second + 60], close=[101.0, float("inf")])
        )


def test_fetch_chunk_rejects_timestamp_outside_requested_window():
    outside = int(datetime(2026, 1, 3, 9, 15).timestamp())
    dhan = SimpleNamespace(
        intraday_minute_data=lambda **_kwargs: {
            "status": "success",
            "data": _payload([outside]),
        }
    )
    with pytest.raises(fetcher.MarketDataValidationError, match="outside requested chunk"):
        fetcher.fetch_chunk(
            dhan=dhan,
            security_id="13",
            exchange_segment="IDX_I",
            instrument_type="INDEX",
            interval=1,
            chunk_start=date(2026, 1, 1),
            chunk_end=date(2026, 1, 2),
        )


def test_atomic_csv_replace_preserves_existing_file_on_write_failure(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text("old-safe-data\n", encoding="utf-8")
    frame = pd.DataFrame({"timestamp": ["2026-01-01"], "close": [100.0]})

    with (
        patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        fetcher.atomic_write_csv(frame, target)

    assert target.read_text(encoding="utf-8") == "old-safe-data\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_csv_replace_commits_complete_file(tmp_path):
    target = tmp_path / "history.csv"
    target.write_text("old-safe-data\n", encoding="utf-8")

    fetcher.atomic_write_csv(pd.DataFrame({"close": [101.0]}), target)

    assert "101.0" in target.read_text(encoding="utf-8")


def test_wrapper_runs_as_a_script_the_way_algo_py_launches_it():
    """`algo.py fetch-data` runs the wrapper as a script, not as an import.

    Python then puts the SCRIPT's folder on sys.path -- not the repository root,
    and not the cwd algo.py sets -- so the engine's `from Dependencies....`
    import used to die with ModuleNotFoundError before argparse ever ran. This
    test reproduces that exact launch and asserts the module imports.

    `--help` exits 0 without credentials or network, which is enough: the old
    failure happened at import time, well before argument parsing.
    """
    wrapper = _REPO_ROOT / "Data Extractors" / "nifty_1m_5y_data_fetch_dhan.py"

    # Strip PYTHONPATH so a developer's own path setup cannot mask the bug, and
    # clear the credential keys so nothing here depends on a populated .env.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.pop("DHAN_CLIENT_CODE", None)
    env.pop("DHAN_ACCESS_TOKEN", None)
    env.pop("DHAN_TOKEN_ID", None)

    # nosec B603 - argv is [sys.executable, <a path built from __file__>, "--help"];
    # nothing here comes from user input, and no shell is involved.
    completed = subprocess.run(  # nosec B603
        [sys.executable, str(wrapper), "--help"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert "ModuleNotFoundError" not in completed.stderr, completed.stderr
    assert completed.returncode == 0, completed.stderr
    assert "--lookback" in completed.stdout


def test_token_is_read_from_dhan_access_token_with_legacy_fallback(monkeypatch):
    """DHAN_ACCESS_TOKEN is the repo-wide key and must win.

    The engine used to read DHAN_TOKEN_ID only -- a name documented nowhere and
    present in no .env -- so credentials could never load. DHAN_TOKEN_ID stays
    accepted so an operator who exported the old name is not broken.
    """
    defaults = fetcher.IndexFetchDefaults(
        display_name="NIFTY", security_id="13", default_output="out.csv"
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setenv("DHAN_CLIENT_CODE", "CLIENT123")

    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "primary-token")
    monkeypatch.setenv("DHAN_TOKEN_ID", "legacy-token")
    assert fetcher.parse_args(defaults).access_token == "primary-token"

    monkeypatch.delenv("DHAN_ACCESS_TOKEN")
    assert fetcher.parse_args(defaults).access_token == "legacy-token"

    monkeypatch.delenv("DHAN_TOKEN_ID")
    assert fetcher.parse_args(defaults).access_token == ""
