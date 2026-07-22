"""Focused MAT-101 tests for the broker-neutral execution contract.

These tests stay entirely in memory.  They never construct a real broker
session, enable a live-trading flag, or make a network request.

One deliberate exception: ``test_adapter_imports_under_its_diagnostic_search_path``
spawns a short-lived subprocess per adapter.  ``sys.path`` semantics cannot be
tested faithfully in-process without polluting the running interpreter, and that
search path is precisely what the test exists to pin down.  Those subprocesses
still only import a module -- no session, no flag, no network.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import re
import subprocess  # nosec B404 - used only to import this repo's own adapters
import sys
import threading
import time
import zipfile
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _NeoApiTestDouble:
    """Import-only stand-in; behavioral tests inject their own SDK clients."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


# Kotak is an optional live dependency and is intentionally absent from the
# core CI environment.  Install only the import surface before any adapter or
# diagnostic module loads; every behavioral test replaces ``client`` with its
# own broker-shaped fake, so no SDK behavior is being simulated here.
_neo_api_test_module = ModuleType("neo_api_client")
_neo_api_test_module.NeoAPI = _NeoApiTestDouble
sys.modules["neo_api_client"] = _neo_api_test_module


class _DhanhqTestDouble:
    """Import-only stand-in; behavioral tests inject their own SDK clients."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


# ``dhanhq`` differs from Kotak's SDK: it is a CORE dependency (it also serves
# market data), so the main quality job HAS the real package and other suites in
# the same pytest process bind names from it.  The isolated broker-dependency
# job installs requirements-brokers.txt alone, where it is deliberately absent.
#
# Hence the conditional: stand in only when the real package is genuinely
# missing.  Clobbering an installed core dependency for every run would be a
# latent trap for whichever suite happens to import it next.
try:  # pragma: no cover - which branch runs depends on the CI environment
    import dhanhq as _installed_dhanhq  # noqa: F401
except ModuleNotFoundError:
    _dhanhq_test_module = ModuleType("dhanhq")
    _dhanhq_test_module.DhanContext = _DhanhqTestDouble
    _dhanhq_test_module.dhanhq = _DhanhqTestDouble
    sys.modules["dhanhq"] = _dhanhq_test_module


def _load_file_module(name: str, relative_path: str) -> ModuleType:
    """Load a broker helper whose parent folder contains spaces."""

    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _contract_module():
    """Import the wished-for MAT-101 contract with a clear RED failure."""

    try:
        return importlib.import_module("Dependencies.broker_contract")
    except ModuleNotFoundError:
        pytest.fail("MAT-101 shared broker contract module is missing")


def test_order_contract_is_exact_immutable_and_quantity_safe() -> None:
    """The shared result cannot hide new statuses or impossible quantities."""

    contract = _contract_module()

    assert {status.name for status in contract.OrderStatus} == {
        "REJECTED",
        "FILLED",
        "PARTIAL",
        "UNKNOWN",
    }

    result = contract.OrderResult(
        order_id="ORDER-1",
        requested_quantity=75,
        filled_quantity=25,
        remaining_quantity=50,
        status=contract.OrderStatus.PARTIAL,
        broker_state="OPEN",
        reason="25 units filled; 50 still open",
    )
    with pytest.raises(FrozenInstanceError):
        result.filled_quantity = 75

    with pytest.raises(ValueError, match="remaining_quantity"):
        contract.OrderResult(
            order_id="BROKEN",
            requested_quantity=75,
            filled_quantity=25,
            remaining_quantity=75,
            status=contract.OrderStatus.UNKNOWN,
            broker_state="malformed",
            reason="quantities disagree",
        )


@pytest.mark.parametrize(
    ("status_name", "filled"),
    [
        ("FILLED", 0),
        ("REJECTED", 25),
        ("PARTIAL", 0),
        ("PARTIAL", 75),
    ],
)
def test_order_result_rejects_semantically_contradictory_statuses(
    status_name: str,
    filled: int,
) -> None:
    contract = _contract_module()

    with pytest.raises(ValueError, match="status"):
        contract.OrderResult(
            order_id="CONTRADICTION",
            requested_quantity=75,
            filled_quantity=filled,
            remaining_quantity=75 - filled,
            status=contract.OrderStatus[status_name],
            broker_state=status_name,
            reason="contradictory direct construction",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_quantity", True, "requested_quantity"),
        ("requested_quantity", 75.5, "requested_quantity"),
        ("filled_quantity", -1, "filled_quantity"),
        ("remaining_quantity", "50", "remaining_quantity"),
        ("status", "FILLED", "OrderStatus"),
    ],
)
def test_order_result_rejects_malformed_direct_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    contract = _contract_module()
    kwargs: dict[str, object] = {
        "order_id": "BROKEN",
        "requested_quantity": 75,
        "filled_quantity": 25,
        "remaining_quantity": 50,
        "status": contract.OrderStatus.PARTIAL,
        "broker_state": "OPEN",
        "reason": "malformed direct construction",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        contract.OrderResult(**kwargs)


def test_order_result_rejects_fill_larger_than_request() -> None:
    contract = _contract_module()
    with pytest.raises(ValueError, match="cannot exceed"):
        contract.OrderResult(
            order_id="BROKEN",
            requested_quantity=75,
            filled_quantity=100,
            remaining_quantity=0,
            status=contract.OrderStatus.UNKNOWN,
            broker_state="OPEN",
            reason="impossible fill",
        )


@pytest.mark.parametrize(
    ("broker_state", "filled", "expected"),
    [
        ("REJECTED", 0, "REJECTED"),
        ("COMPLETE", 75, "FILLED"),
        ("COMPLETE", 25, "PARTIAL"),
        ("REJECTED", 25, "PARTIAL"),
        ("OPEN", 0, "UNKNOWN"),
        ("", 0, "UNKNOWN"),
    ],
)
def test_normalization_is_conservative_and_preserves_acknowledgement(
    broker_state: str,
    filled: int,
    expected: str,
) -> None:
    """Only explicit terminal evidence may become FILLED or REJECTED."""

    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ACK-123",
        requested_quantity=75,
        filled_quantity=filled,
        broker_state=broker_state,
        reason="broker detail",
    )

    assert result.status.name == expected
    assert result.order_id == "ACK-123"
    assert result.remaining_quantity == 75 - filled


def test_malformed_quantity_normalizes_to_unknown_instead_of_rejected() -> None:
    """Bad broker data must not manufacture a harmless zero-fill rejection."""

    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ACK-LOST",
        requested_quantity=75,
        filled_quantity="not-a-number",
        broker_state="REJECTED",
        reason="malformed reply",
    )

    assert result.status is contract.OrderStatus.UNKNOWN
    assert result.filled_quantity == 0
    assert result.remaining_quantity == 75
    assert "malformed" in result.reason.lower()


@pytest.mark.parametrize(
    ("requested", "filled"),
    [
        (True, 0),
        (-1, 0),
        ("75.5", 0),
        (75, True),
        (75, -1),
        (75, 76),
    ],
)
def test_quantity_normalization_rejects_nonexact_or_impossible_values(
    requested: object,
    filled: object,
) -> None:
    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id=None,
        requested_quantity=requested,
        filled_quantity=filled,
        broker_state=None,
    )
    assert result.status is contract.OrderStatus.UNKNOWN
    assert result.order_id == ""
    assert "malformed" in result.reason.lower()


@pytest.mark.parametrize(
    ("state", "filled", "expected", "reason_fragment"),
    [
        ("PARTIAL", 25, "PARTIAL", "partial fill"),
        ("PARTIAL", 0, "UNKNOWN", "terminal order outcome"),
        ("COMPLETE", 75, "FILLED", "requested quantity filled"),
        ("REJECTED", 0, "REJECTED", "terminal rejection"),
    ],
)
def test_normalization_supplies_status_specific_default_reasons(
    state: str,
    filled: int,
    expected: str,
    reason_fragment: str,
) -> None:
    contract = _contract_module()
    result = contract.normalize_order_result(
        order_id="ORDER",
        requested_quantity=75,
        filled_quantity=filled,
        broker_state=state,
    )
    assert result.status.name == expected
    assert reason_fragment in result.reason.lower()


def test_open_snapshot_types_reuse_contract_invariants() -> None:
    contract = _contract_module()
    with pytest.raises(ValueError, match="remaining_quantity"):
        contract.OpenOrder(
            order_id="OPEN-1",
            symbol="NIFTY",
            side="BUY",
            requested_quantity=75,
            filled_quantity=25,
            remaining_quantity=75,
            broker_state="OPEN",
        )
    with pytest.raises(ValueError, match="integer"):
        contract.OpenPosition(
            symbol="NIFTY",
            quantity=True,
            product_type="MIS",
        )


def test_indeterminate_query_uses_safe_defaults() -> None:
    contract = _contract_module()
    result = contract.BrokerQueryResult.indeterminate(" ", broker_state=" timed-out ")
    assert result.items == ()
    assert result.is_indeterminate is True
    assert result.reason == "Broker query outcome is indeterminate."
    assert result.broker_state == "timed-out"


def test_query_failure_cannot_masquerade_as_a_successful_empty_list() -> None:
    """Callers can distinguish a truly empty book from a timed-out query."""

    contract = _contract_module()
    empty = contract.BrokerQueryResult.success([])
    timed_out = contract.BrokerQueryResult.indeterminate("broker timed out")

    assert empty.items == ()
    assert empty.is_indeterminate is False
    assert timed_out.items == ()
    assert timed_out.is_indeterminate is True
    assert timed_out.reason == "broker timed out"


def test_execution_client_protocol_covers_reconciliation_surface() -> None:
    """The shared protocol includes both old helpers and MAT-102 hooks."""

    contract = _contract_module()

    class CompleteFakeClient:
        is_logged_in = True

        def ensure_logged_in(self):
            return True

        def preload_scrip_master(self):
            return True

        def resolve_option_symbol(self, underlying, expiry, option_type, strike):
            return "NIFTY-OPTION"

        def place_market_order(
            self,
            symbol,
            side,
            quantity,
            exchange_segment,
            product_type,
            *,
            order_tag="",
        ):
            return None

        def get_order_status(self, order_id, requested_quantity=0):
            return None

        def cancel_order(self, order_id, requested_quantity=0):
            return None

        def list_open_orders(self):
            return None

        def list_open_positions(self):
            return None

        def recover_after_reconciliation(self):
            return True

        def extract_order_id(self, order_response):
            return ""

        def logout(self):
            return {}

    assert isinstance(CompleteFakeClient(), contract.ExecutionClient)


@pytest.mark.parametrize(
    ("relative_dir", "module_name"),
    [
        ("Dependencies/Kotak API", "kotak_execution"),
        ("Dependencies/Shoonya API", "shoonya_execution"),
        ("Dependencies/Flattrade API", "flattrade_execution"),
        ("Dependencies/Dhan API", "dhan_execution"),
    ],
)
def test_adapter_imports_under_its_diagnostic_search_path(
    relative_dir: str,
    module_name: str,
) -> None:
    """Every adapter must import the way its sibling diagnostic launches it.

    ``python <script>`` puts the SCRIPT's directory on ``sys.path[0]`` and never
    the working directory.  So an adapter that reaches for ``Dependencies.<x>``
    has to put the repo ROOT on the path itself -- adding only ``Dependencies``
    is not enough, because that import needs the package's PARENT.

    This is a regression guard: MAT-108 added
    ``from Dependencies.secret_redaction import ...`` to the Kotak and Shoonya
    adapters ABOVE their ``sys.path`` setup, which broke both diagnostics -- run
    directly and through ``algo.py diagnose`` -- with "No module named
    'Dependencies'".  The runner never noticed because it is launched from the
    repo root, and no test imported an adapter under a diagnostic's search path.

    A genuinely absent optional broker SDK is a different thing and is
    tolerated: CI deliberately runs one job without ``dhanhq`` and another
    without ``neo_api_client``.
    """

    # Rebuild the interpreter's search path exactly as `python <script>` would:
    # the script's own directory first, with '' and the CWD removed so the repo
    # root cannot leak in and mask the bug.
    probe = (
        "import os, sys\n"
        "cwd = os.getcwd()\n"
        f"sys.path[:] = [{str(ROOT / relative_dir)!r}]"
        " + [p for p in sys.path if p not in ('', cwd)]\n"
        f"import {module_name}\n"
        "print('IMPORT OK')\n"
    )
    # nosec B603 - argv is [sys.executable, "-c", <literal built from ROOT and a
    # hard-coded parametrize entry>]; no shell, and no external input reaches it.
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert "No module named 'Dependencies'" not in output, (
        f"{module_name} cannot import under its diagnostic's search path; the "
        f"repo root is missing from sys.path.\n{output}"
    )
    if "IMPORT OK" in output:
        return
    absent_module = re.search(r"No module named '([\w.]+)'", output)
    if absent_module is None:
        pytest.fail(f"{module_name} failed to import:\n{output}")
    pytest.skip(f"optional broker SDK {absent_module.group(1)!r} is not installed")


@pytest.fixture(scope="module")
def flattrade_module() -> ModuleType:
    """Return a private, network-free copy of the Flattrade adapter."""

    return _load_file_module(
        "mat101_flattrade_execution",
        "Dependencies/Flattrade API/flattrade_execution.py",
    )


def _ready_flattrade_client(flattrade_module: ModuleType, monkeypatch):
    """Build an authenticated-looking client without touching Flattrade."""

    client = flattrade_module.FlattradeExecutionClient()
    client._client_id = "TEST"
    client._account_id = "TEST"
    client._access_token = "not-a-real-token"
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


@pytest.mark.parametrize(
    ("status_row", "expected_status", "filled", "remaining"),
    [
        (
            {"stat": "Ok", "status": "REJECTED", "fillshares": "0", "qty": "75"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_flattrade_place_order_normalizes_terminal_outcomes(
    flattrade_module: ModuleType,
    monkeypatch,
    status_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    """An acknowledgement is followed by a typed fill snapshot."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter([{"stat": "Ok", "norenordno": "FT-1"}, [status_row]])
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "FT-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_flattrade_mid_fill_open_snapshot_polls_to_completion(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A market order caught mid-fill (state OPEN) is transient, not terminal.

    Returning that first snapshot as PARTIAL would freeze every live entry
    over a healthy order; the loop must poll again and report the full fill.
    """

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_FILL_POLL_INTERVAL", 0.001)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-MIDFILL"},
            [{"stat": "Ok", "status": "OPEN", "fillshares": "25", "qty": "75"}],
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.order_id == "FT-MIDFILL"
    assert result.filled_quantity == 75
    assert result.remaining_quantity == 0


def test_flattrade_partial_then_cancelled_is_terminal_partial(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A partial fill followed by a cancel can never fill further.

    The terminal broker label makes the very first snapshot the final
    outcome; no further polling is needed (or possible -- the reply iterator
    holds exactly one status response).
    """

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-PARTCXL"},
            [{"stat": "Ok", "status": "CANCELED", "fillshares": "25", "qty": "75"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "PARTIAL"
    assert result.filled_quantity == 25
    assert result.remaining_quantity == 50


def test_flattrade_transmits_execution_ledger_tag(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    captured_payloads = []
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-TAG"},
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )

    def post(_endpoint, payload, **_kwargs):
        captured_payloads.append(dict(payload))
        return next(replies)

    monkeypatch.setattr(client, "_post_api", post)
    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert captured_payloads[0]["remarks"] == "M2-A1B2-4F8D2Q7J-NE1"


def test_flattrade_zero_broker_quantity_cannot_confirm_a_fill(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-ZERO-QTY"},
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "0"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_flattrade_acknowledgement_then_response_loss_is_unknown(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The known order id survives when its status response is lost."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "norenordno": "FT-LOST"},
            TimeoutError("SingleOrdHist response lost"),
        ]
    )

    def post(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(client, "_post_api", post)
    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "FT-LOST"
    assert "lost" in result.reason.lower()


def test_flattrade_cancel_and_reconciliation_queries_are_typed(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Cancel, open orders, and positions expose explicit query outcomes."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Ok", "result": "FT-1"},
            [{"stat": "Ok", "status": "CANCELLED", "fillshares": "0", "qty": "75"}],
            [
                {
                    "stat": "Ok",
                    "norenordno": "FT-OPEN",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "OPEN",
                    "fillshares": "25",
                    "qty": "75",
                },
                {
                    "stat": "Ok",
                    "norenordno": "FT-REJECTED",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "REJECTED",
                    "fillshares": "0",
                    "qty": "75",
                },
                {
                    "stat": "Ok",
                    "norenordno": "FT-CANCELLED-PARTIAL",
                    "tsym": "NIFTY",
                    "trantype": "B",
                    "status": "CANCELLED",
                    "fillshares": "25",
                    "qty": "75",
                },
            ],
            [{"stat": "Ok", "tsym": "NIFTY", "netqty": "50", "prd": "I"}],
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    cancelled = client.cancel_order("FT-1", requested_quantity=75)
    open_orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert open_orders.is_indeterminate is False
    assert len(open_orders.items) == 1
    assert open_orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == 50


def test_flattrade_query_timeout_is_indeterminate_not_empty(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A timeout never returns a successful empty order or position book."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("deadline")),
    )

    status = client.get_order_status("FT-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert status.status.name == "UNKNOWN"
    assert orders.is_indeterminate is True
    assert positions.is_indeterminate is True


def test_flattrade_successful_empty_queries_are_distinct_from_failure(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Two genuine empty books are successful, not indeterminate."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter([[], []])
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_flattrade_only_exact_no_data_envelope_proves_empty_books(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A session error containing the words 'no data' must fail closed."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    replies = iter(
        [
            {"stat": "Not_Ok", "emsg": "No Data"},
            {"stat": "Not_Ok", "emsg": "No Data"},
            {"stat": "Not_Ok", "emsg": "No data because session expired"},
            {"stat": "Not_Ok", "emsg": "No data because session expired"},
        ]
    )
    monkeypatch.setattr(client, "_post_api", lambda *args, **kwargs: next(replies))

    empty_orders = client.list_open_orders()
    empty_positions = client.list_open_positions()
    failed_orders = client.list_open_orders()
    failed_positions = client.list_open_positions()

    assert empty_orders.items == () and empty_orders.is_indeterminate is False
    assert empty_positions.items == () and empty_positions.is_indeterminate is False
    assert failed_orders.is_indeterminate is True
    assert failed_positions.is_indeterminate is True


def test_flattrade_http_constants_are_exactly_ten_seconds(
    flattrade_module: ModuleType,
) -> None:
    """Both authenticated API calls and the scrip download share the deadline."""

    assert flattrade_module._API_TIMEOUT_SECONDS == 10.0
    assert flattrade_module._SCRIP_MASTER_TIMEOUT_SECONDS == 10.0
    assert flattrade_module._BROKER_CALL_DEADLINE_SECONDS == 10.0


def test_flattrade_total_deadline_includes_lock_wait(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """Waiting for the shared session lock cannot exceed the broker budget."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"stat": "Ok"}

    class _Session:
        def post(self, *args, **kwargs):
            return _Response()

    client.client = _Session()
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with client._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="lock"):
            client._post_api("UserDetails", {"uid": "TEST"})
    finally:
        release.set()
        holder.join(timeout=1)
    assert time.monotonic() - started < 0.25


def test_flattrade_total_deadline_includes_rate_limiter_lock_wait(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A stuck limiter mutex cannot postpone HTTP dispatch past the deadline."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("HTTP must not start after limiter deadline")

    session = _Session()
    client.client = session
    locked = threading.Event()
    release = threading.Event()

    def hold_limiter_lock() -> None:
        with client._api_limiter._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_limiter_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="limiter lock"):
            client._post_api("UserDetails", {"uid": "TEST"})
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert session.calls == 0


def test_flattrade_total_deadline_poison_blocks_calls_after_slow_response(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """A slow-dribble request returns on time and cannot overlap a new call."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)
    release = threading.Event()

    class _Response:
        def __init__(self, payload) -> None:
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class _SlowSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                release.wait(0.4)
                return _Response({"stat": "Ok"})
            if "GetPendingOrders" in str(args[0]):
                return _Response([])
            return _Response({"stat": "Ok"})

    session = _SlowSession()
    client.client = session
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            client._post_api("PlaceOrder", {"uid": "TEST"}, is_order=False)
        assert time.monotonic() - started < 0.25
        with pytest.raises(RuntimeError, match="indeterminate"):
            client._post_api("PlaceOrder", {"uid": "TEST"}, is_order=False)
        assert session.calls == 1
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not client._timed_out_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.list_open_orders().is_indeterminate is False
    with pytest.raises(RuntimeError, match="indeterminate"):
        client._post_api("PlaceOrder", {"uid": "TEST"})
    assert client.recover_after_reconciliation() is True
    assert client._post_api("PlaceOrder", {"uid": "TEST"}) == {"stat": "Ok"}


def test_flattrade_status_timeout_cannot_release_a_queued_entry(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The placement gate stays held until the first fill status is known."""

    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    monkeypatch.setattr(flattrade_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(client._order_limiter, "acquire", lambda *_args, **_kwargs: None)
    confirm_started = threading.Event()
    second_submitted = threading.Event()
    release_history = threading.Event()

    class _Response:
        def __init__(self, payload) -> None:
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class _RaceSession:
        def __init__(self) -> None:
            self.place_calls = 0
            self._lock = threading.Lock()

        def post(self, url, *args, **kwargs):
            if "PlaceOrder" in str(url):
                with self._lock:
                    self.place_calls += 1
                    call_number = self.place_calls
                if call_number == 2:
                    second_submitted.set()
                return _Response(
                    {"stat": "Ok", "norenordno": f"FT-{call_number}"}
                )
            if "SingleOrdHist" in str(url):
                release_history.wait(0.5)
                return _Response(
                    [
                        {
                            "stat": "Ok",
                            "status": "COMPLETE",
                            "fillshares": "75",
                            "qty": "75",
                        }
                    ]
                )
            return _Response({"stat": "Ok"})

    session = _RaceSession()
    client.client = session
    original_confirm = client._confirm_fill
    confirm_calls = 0
    confirm_lock = threading.Lock()

    def controlled_confirm(order_id, quantity):
        nonlocal confirm_calls
        with confirm_lock:
            confirm_calls += 1
            call_number = confirm_calls
        if call_number == 1:
            confirm_started.set()
            second_submitted.wait(0.15)
            return original_confirm(order_id, quantity)
        return flattrade_module.normalize_order_result(
            order_id=order_id,
            requested_quantity=quantity,
            filled_quantity=quantity,
            broker_state="COMPLETE",
        )

    monkeypatch.setattr(client, "_confirm_fill", controlled_confirm)
    results = {}

    def place(name: str) -> None:
        results[name] = client.place_market_order("NIFTY", "BUY", 75)

    first = threading.Thread(target=place, args=("first",))
    second = threading.Thread(target=place, args=("second",))
    try:
        first.start()
        assert confirm_started.wait(0.3)
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)
    finally:
        release_history.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert session.place_calls == 1
    assert results["first"].status.name == "UNKNOWN"
    assert results["second"].status.name == "UNKNOWN"


@pytest.fixture(scope="module")
def shoonya_module() -> ModuleType:
    """Return a private, network-free copy of the Shoonya adapter."""

    return _load_file_module(
        "mat101_shoonya_execution",
        "Dependencies/Shoonya API/shoonya_execution.py",
    )


class _FakeNorenClient:
    """Behavioral Shoonya boundary fake; no SDK method reaches the network."""

    def __init__(
        self,
        *,
        place_response=None,
        order_rows=None,
        order_book=None,
        positions=None,
    ) -> None:
        self.place_response = place_response or {"stat": "Ok", "norenordno": "SH-1"}
        self.order_rows = order_rows
        self.order_book = order_book
        self.position_rows = positions
        self.cancel_response = {"stat": "Ok", "result": "SH-1"}
        self.place_calls = 0
        self.last_place_kwargs = None

    def place_order(self, **kwargs):
        self.place_calls += 1
        self.last_place_kwargs = dict(kwargs)
        return self.place_response

    def single_order_history(self, order_id):
        if isinstance(self.order_rows, Exception):
            raise self.order_rows
        return self.order_rows

    def cancel_order(self, order_id):
        if isinstance(self.cancel_response, Exception):
            raise self.cancel_response
        return self.cancel_response

    def get_order_book(self):
        if isinstance(self.order_book, Exception):
            raise self.order_book
        return self.order_book

    def get_positions(self):
        if isinstance(self.position_rows, Exception):
            raise self.position_rows
        return self.position_rows


def _ready_shoonya_client(shoonya_module: ModuleType, monkeypatch, fake):
    client = shoonya_module.ShoonyaExecutionClient()
    client.client = fake
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


@pytest.mark.parametrize(
    ("status_row", "expected_status", "filled", "remaining"),
    [
        (
            {"stat": "Ok", "status": "REJECTED", "fillshares": "0", "qty": "75"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_shoonya_place_order_normalizes_terminal_outcomes(
    shoonya_module: ModuleType,
    monkeypatch,
    status_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    fake = _FakeNorenClient(order_rows=[status_row])
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "SH-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_shoonya_mid_fill_open_snapshot_polls_to_completion(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A mid-fill OPEN snapshot keeps polling until the fill completes."""

    class _SequencedNoren(_FakeNorenClient):
        """History fake that replays a sequence, then repeats the last reply."""

        def __init__(self, sequence) -> None:
            super().__init__()
            self._sequence = list(sequence)

        def single_order_history(self, order_id):
            if len(self._sequence) > 1:
                return self._sequence.pop(0)
            return self._sequence[0]

    fake = _SequencedNoren(
        [
            [{"stat": "Ok", "status": "OPEN", "fillshares": "25", "qty": "75"}],
            [{"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}],
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.filled_quantity == 75
    assert result.remaining_quantity == 0


def test_shoonya_transmits_execution_ledger_tag(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert fake.last_place_kwargs["remarks"] == "M2-A1B2-4F8D2Q7J-NE1"


def test_shoonya_zero_broker_quantity_cannot_confirm_a_fill(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "0"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_shoonya_acknowledgement_then_status_timeout_is_unknown(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(order_rows=TimeoutError("history deadline"))
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    # The pre-MAT-101 implementation polls ambiguous history until its fill
    # window expires. Keep this RED test deterministic and fast.
    monkeypatch.setattr(shoonya_module, "_FILL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(shoonya_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "SH-1"
    assert "deadline" in result.reason.lower()


def test_shoonya_explicit_place_rejection_preserves_broker_evidence(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        place_response={
            "stat": "Not_Ok",
            "norenordno": "SH-REJECTED",
            "emsg": "RMS limit exceeded",
        }
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "REJECTED"
    assert result.order_id == "SH-REJECTED"
    assert result.filled_quantity == 0
    assert "RMS limit exceeded" in result.reason


def test_shoonya_cancel_and_reconciliation_queries_are_typed(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "CANCELLED", "fillshares": "0", "qty": "75"}
        ],
        order_book=[
            {
                "stat": "Ok",
                "norenordno": "SH-OPEN",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "OPEN",
                "fillshares": "25",
                "qty": "75",
            },
            {
                "stat": "Ok",
                "norenordno": "SH-REJECTED",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "REJECTED",
                "fillshares": "0",
                "qty": "75",
            },
            {
                "stat": "Ok",
                "norenordno": "SH-CANCELLED-PARTIAL",
                "tsym": "NIFTY",
                "trantype": "B",
                "status": "CANCELLED",
                "fillshares": "25",
                "qty": "75",
            },
        ],
        positions=[{"stat": "Ok", "tsym": "NIFTY", "netqty": "-50", "prd": "I"}],
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    cancelled = client.cancel_order("SH-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert orders.is_indeterminate is False
    assert len(orders.items) == 1
    assert orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == -50


def test_shoonya_query_failures_are_indeterminate_not_empty(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeNorenClient(
        order_rows=TimeoutError("status deadline"),
        order_book=TimeoutError("orders deadline"),
        positions=TimeoutError("positions deadline"),
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)

    assert client.get_order_status("SH-1", 75).status.name == "UNKNOWN"
    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_shoonya_malformed_history_row_normalizes_unknown(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_rows=["malformed-row"]),
    )

    result = client.get_order_status("SH-1", 75)

    assert result.status.name == "UNKNOWN"
    assert "malformed" in result.reason.lower()


def test_shoonya_successful_empty_queries_are_distinct_from_failure(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Native empty lists prove there are no open Shoonya rows."""

    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_book=[], positions=[]),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_shoonya_no_data_envelopes_are_successful_empty_queries(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Shoonya's explicit No Data response is determinate, not a transport loss."""

    no_data = {"stat": "Not_Ok", "emsg": "No Data"}
    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(order_book=no_data, positions=no_data),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_shoonya_nonempty_error_envelopes_remain_indeterminate(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    for message in ("Session expired", "No data because session expired"):
        error = {"stat": "Not_Ok", "emsg": message}
        client = _ready_shoonya_client(
            shoonya_module,
            monkeypatch,
            _FakeNorenClient(order_book=error, positions=error),
        )

        assert client.list_open_orders().is_indeterminate is True
        assert client.list_open_positions().is_indeterminate is True


def test_shoonya_native_http_timeout_has_a_ten_second_deadline(
    shoonya_module: ModuleType,
) -> None:
    """Both connect and read waits are capped by the MAT-101 deadline."""

    noren_module = sys.modules[shoonya_module.NorenApi.__module__]
    assert noren_module._HTTP_TIMEOUT == (10, 10)
    assert shoonya_module._BROKER_CALL_DEADLINE_SECONDS == 10.0


def test_shoonya_total_deadline_includes_lock_wait(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A queued strategy cannot wait forever for Shoonya's session lock."""

    fake = _FakeNorenClient(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with client._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        result = client.place_market_order("NIFTY", "BUY", 75)
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert result.status.name == "UNKNOWN"
    assert "lock" in result.reason.lower()
    assert fake.place_calls == 0


def test_shoonya_total_deadline_poison_blocks_calls_after_slow_response(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Timed-out native work cannot be followed by a second queued order."""

    release = threading.Event()

    class _SlowNoren(_FakeNorenClient):
        def place_order(self, **kwargs):
            self.place_calls += 1
            release.wait(0.4)
            return self.place_response

    fake = _SlowNoren(
        order_rows=[
            {"stat": "Ok", "status": "COMPLETE", "fillshares": "75", "qty": "75"}
        ]
    )
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    started = time.monotonic()
    try:
        first = client.place_market_order("NIFTY", "BUY", 75)
        assert time.monotonic() - started < 0.25
        second = client.place_market_order("NIFTY", "BUY", 75)
        assert first.status.name == "UNKNOWN"
        assert second.status.name == "UNKNOWN"
        assert "indeterminate" in second.reason.lower()
        assert fake.place_calls == 1
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not client._timed_out_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.get_order_status("SH-1", 75).status.name == "FILLED"
    blocked = client.place_market_order("NIFTY", "BUY", 75)
    assert blocked.status.name == "UNKNOWN"
    assert fake.place_calls == 1
    assert client.recover_after_reconciliation() is True
    recovered = client.place_market_order("NIFTY", "BUY", 75)
    assert recovered.status.name == "FILLED"
    assert fake.place_calls == 2


def test_shoonya_status_timeout_cannot_release_a_queued_entry(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """A second strategy cannot submit between acknowledgement and status poison."""

    confirm_started = threading.Event()
    second_submitted = threading.Event()
    release_history = threading.Event()

    class _RaceNoren(_FakeNorenClient):
        def __init__(self) -> None:
            super().__init__()
            self._place_lock = threading.Lock()

        def place_order(self, **kwargs):
            with self._place_lock:
                self.place_calls += 1
                call_number = self.place_calls
            if call_number == 2:
                second_submitted.set()
            return {"stat": "Ok", "norenordno": f"SH-{call_number}"}

        def single_order_history(self, order_id):
            release_history.wait(0.5)
            return [
                {
                    "stat": "Ok",
                    "status": "COMPLETE",
                    "fillshares": "75",
                    "qty": "75",
                }
            ]

    fake = _RaceNoren()
    client = _ready_shoonya_client(shoonya_module, monkeypatch, fake)
    monkeypatch.setattr(shoonya_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    original_confirm = client._confirm_fill
    confirm_calls = 0
    confirm_lock = threading.Lock()

    def controlled_confirm(order_id, quantity):
        nonlocal confirm_calls
        with confirm_lock:
            confirm_calls += 1
            call_number = confirm_calls
        if call_number == 1:
            confirm_started.set()
            second_submitted.wait(0.15)
            return original_confirm(order_id, quantity)
        return shoonya_module.normalize_order_result(
            order_id=order_id,
            requested_quantity=quantity,
            filled_quantity=quantity,
            broker_state="COMPLETE",
        )

    monkeypatch.setattr(client, "_confirm_fill", controlled_confirm)
    results = {}

    def place(name: str) -> None:
        results[name] = client.place_market_order("NIFTY", "BUY", 75)

    first = threading.Thread(target=place, args=("first",))
    second = threading.Thread(target=place, args=("second",))
    try:
        first.start()
        assert confirm_started.wait(0.3)
        second.start()
        first.join(timeout=1)
        second.join(timeout=1)
    finally:
        release_history.set()
        first.join(timeout=1)
        second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert fake.place_calls == 1
    assert results["first"].status.name == "UNKNOWN"
    assert results["second"].status.name == "UNKNOWN"


def test_vendored_noren_preserves_non_ok_payloads(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """The adapter must receive broker rejection/no-data evidence intact."""

    noren_module = sys.modules[shoonya_module.NorenApi.__module__]
    payload = {"stat": "Not_Ok", "emsg": "No Data", "norenordno": "SH-REJ"}

    class _Response:
        text = json.dumps(payload)

    monkeypatch.setattr(noren_module.requests, "post", lambda *args, **kwargs: _Response())
    api = noren_module.NorenApi()
    api._NorenApi__username = "user"
    api._NorenApi__accountid = "account"
    api._NorenApi__susertoken = "token"

    placed = api.place_order("B", "I", "NFO", "NIFTY", 75, 0, "MKT")
    orders = api.get_order_book()
    positions = api.get_positions()

    assert placed == payload
    assert orders == payload
    assert positions == payload


class _FakeStreamingResponse:
    """Model a streamed ``requests`` response, not the old ``.text`` shape.

    The Kotak scrip-master download streams so it can log how far a transfer
    got before a deadline expired, which means the doubles have to support the
    context-manager + ``iter_content`` protocol the real response provides.
    """

    encoding = "utf-8"

    def __init__(self, body: str, chunk_size: int = 64) -> None:
        self._payload = body.encode("utf-8")
        self._chunk_size = chunk_size

    def __enter__(self) -> _FakeStreamingResponse:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 0):
        step = chunk_size or self._chunk_size
        for start in range(0, len(self._payload), step):
            yield self._payload[start : start + step]


@pytest.fixture(scope="module")
def kotak_module() -> ModuleType:
    """Return a private copy of the Kotak adapter; calls use behavioral fakes."""

    return _load_file_module(
        "mat101_kotak_execution",
        "Dependencies/Kotak API/kotak_execution.py",
    )


def test_kotak_adapter_fixture_does_not_require_optional_sdk(
    kotak_module: ModuleType,
) -> None:
    """Hosted tests use an import-only double instead of an installed SDK."""

    assert kotak_module.NeoAPI is _NeoApiTestDouble


class _FakeKotakSdk:
    """Behavioral Kotak SDK boundary fake with broker-shaped responses."""

    def __init__(self, *, history_row=None, orders=None, positions=None) -> None:
        self.history_row = history_row
        self.orders = orders
        self.position_rows = positions
        self.last_place_kwargs = None

    def place_order(self, **kwargs):
        self.last_place_kwargs = dict(kwargs)
        return {"stat": "Ok", "nOrdNo": "KT-1"}

    def order_history(self, order_id):
        if isinstance(self.history_row, Exception):
            raise self.history_row
        return {"data": {"stat": "Ok", "data": [self.history_row]}}

    def cancel_order(self, order_id):
        return {"stat": "Ok", "nOrdNo": order_id}

    def order_report(self):
        if isinstance(self.orders, Exception):
            raise self.orders
        return {"data": self.orders}

    def positions(self):
        if isinstance(self.position_rows, Exception):
            raise self.position_rows
        return {"data": self.position_rows}


def _ready_kotak_client(kotak_module: ModuleType, monkeypatch, fake):
    client = kotak_module.KotakExecutionClient()
    client.client = fake
    client.is_logged_in = True
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    return client


def test_kotak_transmits_execution_ledger_tag(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    result = client.place_market_order(
        "NIFTY",
        "BUY",
        75,
        order_tag="M2-A1B2-4F8D2Q7J-NE1",
    )

    assert result.status.name == "FILLED"
    assert fake.last_place_kwargs["tag"] == "M2-A1B2-4F8D2Q7J-NE1"


@pytest.mark.parametrize(
    ("history_row", "expected_status", "filled", "remaining"),
    [
        (
            {"ordSt": "REJECTED", "fldQty": "0", "qty": "75", "rejRsn": "RMS"},
            "REJECTED",
            0,
            75,
        ),
        (
            {"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"},
            "FILLED",
            75,
            0,
        ),
        (
            {"ordSt": "COMPLETE", "fldQty": "25", "qty": "75"},
            "PARTIAL",
            25,
            50,
        ),
    ],
)
def test_kotak_place_order_normalizes_terminal_outcomes(
    kotak_module: ModuleType,
    monkeypatch,
    history_row: dict[str, str],
    expected_status: str,
    filled: int,
    remaining: int,
) -> None:
    fake = _FakeKotakSdk(history_row=history_row)
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_FILL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(kotak_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "KT-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_kotak_transient_hand_off_states_poll_to_completion(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """Kotak's routine hand-off states must not end fill confirmation early.

    "put order req received" and "validation pending" are states every healthy
    Kotak order passes through.  Treating them as terminal would report a
    perfectly normal order as UNKNOWN and freeze all live entries.
    """

    class _SequencedKotakSdk(_FakeKotakSdk):
        """History fake that replays a sequence, then repeats the last row."""

        def __init__(self, rows) -> None:
            super().__init__()
            self._rows = list(rows)

        def order_history(self, order_id):
            row = self._rows.pop(0) if len(self._rows) > 1 else self._rows[0]
            return {"data": {"stat": "Ok", "data": [row]}}

    fake = _SequencedKotakSdk(
        [
            {"ordSt": "put order req received", "fldQty": "0", "qty": "75"},
            {"ordSt": "validation pending", "fldQty": "0", "qty": "75"},
            {"ordSt": "complete", "fldQty": "75", "qty": "75"},
        ]
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_FILL_POLL_INTERVAL", 0.001)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.filled_quantity == 75
    assert client.session_poisoned is False


def test_kotak_zero_broker_quantity_cannot_confirm_a_fill(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "0"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert "does not match" in result.reason.lower()


def test_kotak_acknowledgement_then_status_loss_preserves_order_id(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The accepted order remains discoverable when order_history is lost."""

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(history_row=TimeoutError("history response lost")),
    )

    result = client.place_market_order("NIFTY", "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.order_id == "KT-1"
    assert "lost" in result.reason.lower()
    assert client.session_poisoned is False


def test_kotak_cancel_and_reconciliation_queries_are_typed(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row={"ordSt": "CANCELLED", "fldQty": "0", "qty": "75"},
        orders=[
            {
                "nOrdNo": "KT-OPEN",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "OPEN",
                "fldQty": "25",
                "qty": "75",
            },
            {
                "nOrdNo": "KT-REJECTED",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "REJECTED",
                "fldQty": "0",
                "qty": "75",
            },
            {
                "nOrdNo": "KT-CANCELLED-PARTIAL",
                "trdSym": "NIFTY",
                "trnsTp": "B",
                "ordSt": "CANCELLED",
                "fldQty": "25",
                "qty": "75",
            },
        ],
        positions=[{"trdSym": "NIFTY", "netQty": "50", "prod": "MIS"}],
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    cancelled = client.cancel_order("KT-1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert orders.is_indeterminate is False
    assert len(orders.items) == 1
    assert orders.items[0].remaining_quantity == 50
    assert positions.is_indeterminate is False
    assert positions.items[0].quantity == 50


def test_kotak_query_failures_are_indeterminate_not_empty(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    fake = _FakeKotakSdk(
        history_row=TimeoutError("history deadline"),
        orders=TimeoutError("orders deadline"),
        positions=TimeoutError("positions deadline"),
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)

    assert client.get_order_status("KT-1", 75).status.name == "UNKNOWN"
    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_kotak_malformed_history_row_normalizes_unknown(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(history_row="malformed-row"),
    )

    result = client.get_order_status("KT-1", 75)

    assert result.status.name == "UNKNOWN"
    assert "malformed" in result.reason.lower()


# One real row captured from Kotak's live position book, immediately after a
# BUY 65 / SELL 65 round trip on NIFTY26JUL24500CE. The point of keeping it
# verbatim: it carries NO netQty/netqty key at all. Kotak reports the four
# component quantities and expects the caller to compute the net, so an adapter
# reading row["netQty"] sees None and calls a flat account "indeterminate" --
# which blocked live trading at startup on 2026-07-22.
_KOTAK_FLAT_ROUND_TRIP_POSITION = {
    "actId": "YHRT0",
    "brdLtQty": 65,
    "cfBuyQty": "0",
    "cfSellQty": "0",
    "flBuyQty": "65",
    "flSellQty": "65",
    "buyAmt": "1972.75",
    "sellAmt": "1982.50",
    "exSeg": "nse_fo",
    "prod": "MIS",
    "trdSym": "NIFTY26JUL24500CE",
    "optTp": "CE",
    "sym": "NIFTY",
    "lotSz": "65",
}


def _kotak_position(**overrides):
    """Return the captured live row with specific quantities overridden."""

    row = dict(_KOTAK_FLAT_ROUND_TRIP_POSITION)
    row.update(overrides)
    return row


def test_kotak_completed_round_trip_position_reads_as_flat(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A bought-then-sold position is flat, not an unreadable row.

    Kotak's book has no net-quantity field, so the net must be computed as
    (cfBuy + flBuy) - (cfSell + flSell). Reading a missing netQty as "cannot
    parse" made every startup audit fail once any position row existed.
    """

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(positions=[_kotak_position()]),
    )

    result = client.list_open_positions()

    assert result.is_indeterminate is False, result.reason
    # Flat legs are dropped, so a closed round trip leaves nothing open.
    assert result.items == ()


@pytest.mark.parametrize(
    ("overrides", "expected_quantity"),
    [
        # Open long: bought today, nothing sold back.
        ({"flBuyQty": "65", "flSellQty": "0"}, 65),
        # Open short.
        ({"flBuyQty": "0", "flSellQty": "65"}, -65),
        # Carry-forward legs must count toward the net too.
        ({"cfBuyQty": "130", "flBuyQty": "0", "flSellQty": "65"}, 65),
        # Partially closed: 130 bought, 65 sold back.
        ({"flBuyQty": "130", "flSellQty": "65"}, 65),
    ],
)
def test_kotak_net_position_is_computed_from_its_components(
    kotak_module: ModuleType,
    monkeypatch,
    overrides: dict,
    expected_quantity: int,
) -> None:
    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(positions=[_kotak_position(**overrides)]),
    )

    result = client.list_open_positions()

    assert result.is_indeterminate is False, result.reason
    assert len(result.items) == 1
    assert result.items[0].quantity == expected_quantity
    assert result.items[0].symbol == "NIFTY26JUL24500CE"


def test_kotak_explicit_net_quantity_still_wins_when_present(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """If a future master revision supplies netQty, trust it over the parts."""

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(positions=[_kotak_position(netQty="-65")]),
    )

    result = client.list_open_positions()

    assert result.is_indeterminate is False, result.reason
    assert result.items[0].quantity == -65


@pytest.mark.parametrize(
    "overrides",
    [
        # No quantity information of any kind.
        {"cfBuyQty": None, "cfSellQty": None, "flBuyQty": None, "flSellQty": None},
        # Present but unparseable: never silently treat this as zero.
        {"flBuyQty": "abc"},
        {"flSellQty": "6.5"},
        {"netQty": "not-a-number"},
    ],
)
def test_kotak_unreadable_position_quantity_fails_closed(
    kotak_module: ModuleType,
    monkeypatch,
    overrides: dict,
) -> None:
    """An unreadable row must never be reported as flat.

    "Flat" is what lets live trading start, so a quantity we cannot parse has
    to stay indeterminate rather than default to zero.
    """

    row = _kotak_position()
    for key, value in overrides.items():
        if value is None:
            row.pop(key, None)
        else:
            row[key] = value
    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(positions=[row]),
    )

    assert client.list_open_positions().is_indeterminate is True


def test_kotak_indeterminate_book_queries_are_logged(
    kotak_module: ModuleType,
    monkeypatch,
    caplog,
) -> None:
    """The reason must reach the log, not just the returned object.

    ``audit_startup_exposure`` deliberately strips broker text from its alert,
    so an unlogged reason is invisible: the 2026-07-22 startup block reported
    only "indeterminate" and the cause had to be reproduced by hand.
    """

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(positions=[{"trdSym": "NIFTY26JUL24500CE"}]),
    )

    with caplog.at_level(logging.WARNING):
        result = client.list_open_positions()

    assert result.is_indeterminate is True
    assert any(
        "indeterminate" in record.message.lower() for record in caplog.records
    ), f"no warning logged; records={[r.message for r in caplog.records]}"


def test_kotak_error_envelopes_cannot_masquerade_as_empty_books(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A Not_Ok wrapper with data=[] is still a failed reconciliation query."""

    class ErrorEnvelopeKotak(_FakeKotakSdk):
        def order_report(self):
            return {"stat": "Not_Ok", "emsg": "session expired", "data": []}

        def positions(self):
            return {
                "data": {
                    "stat": "Not_Ok",
                    "emsg": "session expired",
                    "data": [],
                }
            }

    client = _ready_kotak_client(kotak_module, monkeypatch, ErrorEnvelopeKotak())

    assert client.list_open_orders().is_indeterminate is True
    assert client.list_open_positions().is_indeterminate is True


def test_kotak_successful_empty_queries_are_distinct_from_failure(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """Empty Kotak lists are successful snapshots, not query failures."""

    client = _ready_kotak_client(
        kotak_module,
        monkeypatch,
        _FakeKotakSdk(
            orders=[],
            positions=[{"trdSym": "NIFTY", "netQty": 0, "prod": "MIS"}],
        ),
    )

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_kotak_sdk_deadline_and_executor_are_fixed(
    kotak_module: ModuleType,
) -> None:
    """The SDK boundary is ten seconds and physically single-threaded."""

    client = kotak_module.KotakExecutionClient()
    assert kotak_module._BROKER_DEADLINE_SECONDS == 10.0
    assert client._sdk_executor._max_workers == 1


def test_kotak_scrip_master_budget_is_separate_from_the_order_deadline(
    kotak_module: ModuleType,
) -> None:
    """A bulk catalogue download must not be capped by the ORDER deadline.

    Ten seconds exists so a live order can never go stale in flight.  The scrip
    master is a multi-megabyte CSV fetched once, which is not that kind of call
    -- holding it to the same budget made it time out on slower links and
    disabled live trading for the whole session.  The two budgets are therefore
    separate, and the order one must stay exactly ten seconds.
    """

    assert kotak_module._BROKER_DEADLINE_SECONDS == 10.0
    assert kotak_module._SCRIP_MASTER_TIMEOUT_SECONDS > (
        kotak_module._BROKER_DEADLINE_SECONDS
    )


def test_kotak_scrip_master_download_has_total_deadline(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A slow-dribble CSV body cannot hold the broker lock indefinitely."""

    started = threading.Event()
    release = threading.Event()

    class ScripKotak(_FakeKotakSdk):
        def scrip_master(self, exchange_segment):
            return "https://example.invalid/nfo.csv"

    def slow_get(*args, **kwargs):
        started.set()
        release.wait(0.5)
        return _FakeStreamingResponse(
            "pExpiryDate,pSymbolName,pOptionType,dStrikePrice;,pTrdSymbol\n"
            "0,NIFTY,CE,2250000,NIFTY-TEST\n"
        )

    client = _ready_kotak_client(kotak_module, monkeypatch, ScripKotak())
    # The download runs on its own budget now, so shrink THAT one; the order
    # deadline is shrunk too so a regression that reverts to it still trips
    # this test rather than silently waiting 20s.
    monkeypatch.setattr(kotak_module, "_SCRIP_MASTER_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(kotak_module.requests, "get", slow_get)
    results = []

    worker = threading.Thread(target=lambda: results.append(client.preload_scrip_master()))
    try:
        worker.start()
        assert started.wait(0.3)
        worker.join(timeout=0.15)
        assert not worker.is_alive()
    finally:
        release.set()
        worker.join(timeout=1)

    assert results == [False]
    assert client.session_poisoned is True


def test_kotak_order_submission_gate_has_total_deadline(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A caller queued behind another order must receive a bounded UNKNOWN."""

    fake = _FakeKotakSdk(
        history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
    )
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.02, raising=False)
    locked = threading.Event()
    release = threading.Event()

    def hold_gate() -> None:
        with client._order_submission_lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_gate)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        result = client.place_market_order("NIFTY", "BUY", 75)
    finally:
        release.set()
        holder.join(timeout=1)

    assert time.monotonic() - started < 0.25
    assert result.status.name == "UNKNOWN"
    assert result.broker_state == "ORDER_GATE_TIMEOUT"
    assert "submission gate" in result.reason.lower()


def test_kotak_timeout_poisoning_prevents_overlapping_later_orders(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A timed-out SDK thread remains isolated and blocks new submissions."""

    release = threading.Event()

    class BlockingKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__(
                history_row={"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
            )
            self.active = 0
            self.max_active = 0
            self.place_calls = 0
            self.guard = threading.Lock()

        def place_order(self, **kwargs):
            with self.guard:
                self.place_calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                release.wait(timeout=1.0)
                return {"stat": "Ok", "nOrdNo": "KT-SLOW"}
            finally:
                with self.guard:
                    self.active -= 1

    fake = BlockingKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.02, raising=False)

    try:
        first = client.place_market_order("NIFTY", "BUY", 75)
        second = client.place_market_order("NIFTY", "BUY", 75)

        assert first.status.name == "UNKNOWN"
        assert second.status.name == "UNKNOWN"
        assert client.session_poisoned is True
        assert fake.place_calls == 1
        assert fake.max_active == 1
    finally:
        release.set()
        time.sleep(0.03)


def test_kotak_concurrent_submission_cannot_queue_behind_a_timed_out_order(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The poison check and SDK submit are atomic across strategy threads."""

    started = threading.Event()
    release = threading.Event()

    class BlockingKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__()
            self.place_calls = 0
            self.guard = threading.Lock()

        def place_order(self, **kwargs):
            with self.guard:
                self.place_calls += 1
            started.set()
            release.wait(timeout=1.0)
            return {"stat": "Ok", "nOrdNo": f"KT-{self.place_calls}"}

    fake = BlockingKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.1)
    results = []

    def submit_order() -> None:
        results.append(client.place_market_order("NIFTY", "BUY", 75))

    first = threading.Thread(target=submit_order)
    second = threading.Thread(target=submit_order)
    try:
        first.start()
        assert started.wait(timeout=0.5)
        second.start()
        first.join(timeout=0.5)
        second.join(timeout=0.5)
        release.set()
        time.sleep(0.05)

        assert len(results) == 2
        assert all(result.status.name == "UNKNOWN" for result in results)
        assert client.session_poisoned is True
        assert fake.place_calls == 1
    finally:
        release.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)


def test_kotak_status_timeout_cannot_release_a_queued_entry(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """The entry gate covers acknowledgement through final fill confirmation."""

    history_started = threading.Event()
    release = threading.Event()

    class BlockingHistoryKotak(_FakeKotakSdk):
        def __init__(self) -> None:
            super().__init__()
            self.place_calls = 0

        def place_order(self, **kwargs):
            self.place_calls += 1
            return {"stat": "Ok", "nOrdNo": f"KT-{self.place_calls}"}

        def order_history(self, order_id):
            history_started.set()
            release.wait(timeout=1.0)
            return {
                "data": {
                    "stat": "Ok",
                    "data": [
                        {"ordSt": "COMPLETE", "fldQty": "75", "qty": "75"}
                    ],
                }
            }

    fake = BlockingHistoryKotak()
    client = _ready_kotak_client(kotak_module, monkeypatch, fake)
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.05)
    results = []

    def submit_order() -> None:
        results.append(client.place_market_order("NIFTY", "BUY", 75))

    first = threading.Thread(target=submit_order)
    second = threading.Thread(target=submit_order)
    try:
        first.start()
        assert history_started.wait(timeout=0.5)
        second.start()
        first.join(timeout=0.5)
        second.join(timeout=0.5)
        release.set()
        time.sleep(0.05)

        assert len(results) == 2
        assert all(result.status.name == "UNKNOWN" for result in results)
        assert client.session_poisoned is True
        assert fake.place_calls == 1
    finally:
        release.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)


def test_kotak_recovery_waits_for_every_abandoned_sdk_future(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    """A cancelled queued timeout cannot hide an earlier still-running call."""

    started = threading.Event()
    release = threading.Event()
    client = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(kotak_module, "_BROKER_DEADLINE_SECONDS", 0.03)

    def blocking_call():
        started.set()
        release.wait(timeout=1.0)
        return "first"

    try:
        with pytest.raises(kotak_module._SdkDeadlineExceeded):
            client._sdk_call("first", blocking_call)
        assert started.is_set()
        first_future = client._timed_out_future
        assert first_future is not None and not first_future.done()

        with pytest.raises(kotak_module._SdkDeadlineExceeded):
            client._sdk_call("queued", lambda: "second")
        latest_future = client._timed_out_future
        assert latest_future is not None
        latest_future.cancel()

        assert client.recover_after_reconciliation() is False
        assert not first_future.done()
    finally:
        release.set()

    deadline = time.monotonic() + 0.5
    while not first_future.done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert client.recover_after_reconciliation() is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, None),
        ("bad", None),
        (1.5, None),
        (float("inf"), None),
        ("75", 75),
    ],
)
def test_broker_quantity_helpers_require_exact_finite_integers(
    shoonya_module: ModuleType,
    kotak_module: ModuleType,
    flattrade_module: ModuleType,
    value: object,
    expected: int | None,
) -> None:
    """Each adapter rejects rounded or boolean quantities at its own boundary."""

    assert shoonya_module._exact_int(value) == expected
    assert kotak_module._exact_int(value) == expected
    assert flattrade_module._exact_int(value) == expected


def test_shoonya_login_success_and_missing_credentials_fail_closed(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """Login accepts only a typed Ok response and never keeps a half-session."""

    captured: dict[str, object] = {}

    class LoginNoren:
        def login(self, **kwargs):
            captured.update(kwargs)
            return {"stat": "Ok", "uname": "Test Trader"}

    for name, value in {
        "SHOONYA_USERID": "USER",
        "SHOONYA_PASSWORD": "PASSWORD",
        "SHOONYA_VENDOR_CODE": "VENDOR",
        "SHOONYA_API_SECRET": "SECRET",
        "SHOONYA_TOTP_SECRET": "TOTP-SEED",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(shoonya_module, "NorenApi", LoginNoren)
    client = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(client, "_generate_totp", lambda _secret: "123456")

    assert client.ensure_logged_in() is True
    assert client.ensure_logged_in() is True
    assert client.is_logged_in is True
    assert captured["twoFA"] == "123456"

    for name in (
        "SHOONYA_USERID",
        "SHOONYA_PASSWORD",
        "SHOONYA_VENDOR_CODE",
        "SHOONYA_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    missing = shoonya_module.ShoonyaExecutionClient()
    assert missing.ensure_logged_in() is False
    assert missing.client is None


def test_shoonya_blank_totp_and_rejected_login_clear_session(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    class RejectedNoren:
        def login(self, **_kwargs):
            return {"stat": "Not_Ok", "emsg": "bad credentials"}

    for name, value in {
        "SHOONYA_USERID": "USER",
        "SHOONYA_PASSWORD": "PASSWORD",
        "SHOONYA_VENDOR_CODE": "VENDOR",
        "SHOONYA_API_SECRET": "SECRET",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SHOONYA_TOTP_SECRET", raising=False)

    blank = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(blank, "_prompt_totp", lambda: "")
    assert blank.ensure_logged_in() is False

    monkeypatch.setenv("SHOONYA_TOTP_SECRET", "SEED")
    monkeypatch.setattr(shoonya_module, "NorenApi", RejectedNoren)
    rejected = shoonya_module.ShoonyaExecutionClient()
    monkeypatch.setattr(rejected, "_generate_totp", lambda _secret: "123456")
    assert rejected.ensure_logged_in() is False
    assert rejected.client is None


def test_shoonya_scrip_master_and_symbol_resolution_are_cached(
    shoonya_module: ModuleType,
    monkeypatch,
) -> None:
    """The official master validates exact contracts without later downloads."""

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "NFO_symbols.txt",
            "Exchange,Token,LotSize,Symbol,TradingSymbol\n"
            "NFO,1,75,NIFTY,NIFTY16JUL26C22500\n",
        )

    class Response:
        content = archive.getvalue()

        @staticmethod
        def raise_for_status() -> None:
            return None

    client = _ready_shoonya_client(
        shoonya_module,
        monkeypatch,
        _FakeNorenClient(),
    )
    monkeypatch.setattr(shoonya_module.requests, "get", lambda *args, **kwargs: Response())

    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    symbol = client.resolve_option_symbol("nifty", date(2026, 7, 16), "ce", 22500)
    assert symbol == "NIFTY16JUL26C22500"
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500) == symbol
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", None, "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "not-a-date", "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "XX", 22500) == ""


def test_shoonya_logout_and_recursive_order_id_extraction(
    shoonya_module: ModuleType,
) -> None:
    client = shoonya_module.ShoonyaExecutionClient()
    assert client.logout()["State"] == "NOT_OK"
    assert client.extract_order_id({"data": [{"norenordno": "SH-NESTED"}]}) == "SH-NESTED"
    assert client.extract_order_id([{"none": 0}, " SH-STRING "]) == "SH-STRING"
    assert client.extract_order_id(123) == ""

    class LogoutNoren:
        @staticmethod
        def logout():
            return "bye"

    client.client = LogoutNoren()
    client.is_logged_in = True
    assert client.logout() == {"State": "OK", "message": "bye"}
    assert client.client is None
    assert client.is_logged_in is False


def test_kotak_login_success_normalizes_mobile_and_validates_two_factor(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Configuration:
        edit_token = "trade-token"
        edit_sid = "trade-session"
        serverId = "server"
        base_url = "https://example.invalid"
        data_center = "dc"

    class LoginNeo:
        def __init__(self, **kwargs):
            captured["constructor"] = kwargs
            self.configuration = Configuration()

        def totp_login(self, **kwargs):
            captured["login"] = kwargs
            return {"stat": "Ok"}

        def totp_validate(self, **kwargs):
            captured["validate"] = kwargs
            return {"stat": "Ok"}

    for name, value in {
        "CONSUMER_KEY": "CONSUMER",
        "MOBILE": "98765-43210",
        "MPIN": "123456",
        "UCC": "UCC123",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("COSNSUMER_KEY", raising=False)
    monkeypatch.setattr(kotak_module, "NeoAPI", LoginNeo)
    client = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(client, "_prompt_totp", lambda: "654321")

    assert client.ensure_logged_in() is True
    assert client.ensure_logged_in() is True
    assert captured["login"]["mobile_number"] == "+919876543210"
    assert captured["validate"] == {"mpin": "123456"}
    assert kotak_module.KotakExecutionClient._normalize_mobile("919876543210") == "+919876543210"
    assert kotak_module.KotakExecutionClient._normalize_mobile("+919876543210") == "+919876543210"
    assert kotak_module.KotakExecutionClient._normalize_mobile("unexpected") == "unexpected"


def test_kotak_login_rejects_missing_credentials_and_incomplete_two_factor(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    for name in ("COSNSUMER_KEY", "CONSUMER_KEY", "MOBILE", "MPIN", "UCC"):
        monkeypatch.delenv(name, raising=False)
    assert kotak_module.KotakExecutionClient().ensure_logged_in() is False

    class Configuration:
        edit_token = ""
        edit_sid = ""

    class IncompleteNeo:
        def __init__(self, **_kwargs):
            self.configuration = Configuration()

        @staticmethod
        def totp_login(**_kwargs):
            return {"stat": "Ok"}

        @staticmethod
        def totp_validate(**_kwargs):
            return {"stat": "Not_Ok"}

    for name, value in {
        "CONSUMER_KEY": "CONSUMER",
        "MOBILE": "9876543210",
        "MPIN": "123456",
        "UCC": "UCC123",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(kotak_module, "NeoAPI", IncompleteNeo)
    incomplete = kotak_module.KotakExecutionClient()
    monkeypatch.setattr(incomplete, "_prompt_totp", lambda: "654321")
    assert incomplete.ensure_logged_in() is False
    assert incomplete.client is None


def test_kotak_scrip_master_resolution_and_miss_diagnostics(
    kotak_module: ModuleType,
    monkeypatch,
) -> None:
    expiry = date(2026, 7, 16)
    encoded_expiry = int(
        (pd.Timestamp(expiry) - pd.to_timedelta(315511200, unit="s")).timestamp()
    )

    class ScripKotak(_FakeKotakSdk):
        @staticmethod
        def scrip_master(exchange_segment):
            assert exchange_segment == "nse_fo"
            return "https://example.invalid/nfo.csv"

    csv_body = (
        "pExpiryDate,pSymbolName,pOptionType,dStrikePrice;,pTrdSymbol\n"
        f"{encoded_expiry},NIFTY,CE,2250000,NIFTY26JUL22500CE\n"
    )

    client = _ready_kotak_client(kotak_module, monkeypatch, ScripKotak())
    monkeypatch.setattr(
        kotak_module.requests,
        "get",
        lambda *args, **kwargs: _FakeStreamingResponse(csv_body),
    )

    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    symbol = client.resolve_option_symbol("nifty", expiry, "ce", 22500)
    assert symbol == "NIFTY26JUL22500CE"
    assert client.resolve_option_symbol("NIFTY", expiry, "CE", 22500) == symbol
    assert client.resolve_option_symbol("NIFTY", expiry, "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", None, "CE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "not-a-date", "CE", 22500) == ""
    assert kotak_module.KotakExecutionClient._match_in_df(None, "NIFTY", "CE", "", 1) == ""


def test_kotak_logout_and_recursive_order_id_extraction(
    kotak_module: ModuleType,
) -> None:
    client = kotak_module.KotakExecutionClient()
    assert client.logout()["State"] == "NOT_OK"
    assert client.extract_order_id({"data": [{"nOrdNo": "KT-NESTED"}]}) == "KT-NESTED"
    assert client.extract_order_id([{"none": 0}, " KT-STRING "]) == "KT-STRING"
    assert client.extract_order_id(123) == ""

    class LogoutKotak:
        @staticmethod
        def logout():
            return "bye"

    client.client = LogoutKotak()
    client.is_logged_in = True
    assert client.logout() == {"State": "OK", "message": "bye"}
    assert client.client is None


def _flattrade_master_fixture() -> pd.DataFrame:
    """Return one valid and one filtered-out Flattrade catalogue row."""

    return pd.DataFrame(
        {
            "Exchange": ["NFO", "NSE"],
            "Lotsize": ["75", "1"],
            "Symbol": ["NIFTY", "NIFTY"],
            "Tradingsymbol": ["NIFTY16JUL26C22500", "NIFTY-SPOT"],
            "Expiry": ["16-JUL-2026", "16-JUL-2026"],
            "Strike": ["22500.00", "0"],
            "Optiontype": ["CE", "XX"],
        }
    )


def test_flattrade_env_parsing_session_validation_and_login_fast_path(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAT_TEST_QUOTED", "' value '")
    monkeypatch.setenv("MAT_TEST_BAD_INT", "bad")
    monkeypatch.setenv("MAT_TEST_NEGATIVE", "-1")
    assert flattrade_module._env_str("MAT_TEST_QUOTED") == "value"
    assert flattrade_module._env_str("MAT_TEST_MISSING", "fallback") == "fallback"
    assert flattrade_module._env_non_negative_int("MAT_TEST_BAD_INT", 3) == 3
    assert flattrade_module._env_non_negative_int("MAT_TEST_NEGATIVE", 3) == 3

    client = flattrade_module.FlattradeExecutionClient()
    client._client_id = "CLIENT"
    client._access_token = "TOKEN"
    monkeypatch.setattr(
        client,
        "_post_api",
        lambda *_args, **_kwargs: {
            "stat": "Ok",
            "actid": "ACCOUNT",
            "exarr": ["NFO"],
        },
    )
    assert client._validate_session_locked() is True
    assert client._account_id == "ACCOUNT"

    monkeypatch.setenv("FLATTRADE_CLIENT_ID", "CLIENT")
    monkeypatch.setenv("FLATTRADE_ACCESS_TOKEN", "TOKEN")
    login = flattrade_module.FlattradeExecutionClient()
    monkeypatch.setattr(login, "_validate_session_locked", lambda: True)
    assert login.ensure_logged_in() is True
    assert login.ensure_logged_in() is True


def test_flattrade_browser_token_exchange_is_validated_before_login(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    """The interactive flow hashes the secret and accepts only a validated token."""

    captured: dict[str, object] = {}

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, str]:
            return {"status": "Ok", "token": "DAILY-TOKEN", "client": "CLIENT"}

    class Session:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            captured["timeout"] = kwargs["timeout"]
            return Response()

    for name, value in {
        "FLATTRADE_CLIENT_ID": "CLIENT",
        "FLATTRADE_API_KEY": "API-KEY",
        "FLATTRADE_API_SECRET": "API-SECRET",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FLATTRADE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt: "REQUEST-CODE")
    monkeypatch.setattr(flattrade_module.webbrowser, "open", lambda _url: True)

    client = flattrade_module.FlattradeExecutionClient()
    monkeypatch.setattr(client, "_ensure_session_locked", lambda: Session())
    monkeypatch.setattr(client, "_validate_session_locked", lambda: True)

    assert client.ensure_logged_in() is True
    assert client._access_token == "DAILY-TOKEN"
    expected_digest = flattrade_module.hashlib.sha256(
        b"API-KEYREQUEST-CODEAPI-SECRET"
    ).hexdigest()
    assert captured["json"] == {
        "api_key": "API-KEY",
        "request_code": "REQUEST-CODE",
        "api_secret": expected_digest,
    }


def test_flattrade_login_rejects_missing_identity_or_authorization_inputs(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    for name in (
        "FLATTRADE_CLIENT_ID",
        "FLATTRADE_ACCESS_TOKEN",
        "FLATTRADE_API_KEY",
        "FLATTRADE_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False

    monkeypatch.setenv("FLATTRADE_CLIENT_ID", "CLIENT")
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False

    monkeypatch.setenv("FLATTRADE_API_KEY", "API-KEY")
    monkeypatch.setenv("FLATTRADE_API_SECRET", "API-SECRET")
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert flattrade_module.FlattradeExecutionClient().ensure_logged_in() is False


def test_flattrade_downloads_and_caches_the_official_scrip_master(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    csv_text = _flattrade_master_fixture().to_csv(index=False)

    class Response:
        text = csv_text

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Session:
        calls = 0

        def get(self, _url, **_kwargs):
            self.calls += 1
            return Response()

    session = Session()
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    client.client = session
    assert client.preload_scrip_master() is True
    assert client.preload_scrip_master() is True
    assert session.calls == 1
    assert client._scrip_df is not None
    assert len(client._scrip_df) == 1


def test_flattrade_scrip_preparation_resolution_and_logout(
    flattrade_module: ModuleType,
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        flattrade_module.FlattradeExecutionClient._prepare_scrip_master(pd.DataFrame())
    unusable = _flattrade_master_fixture()
    unusable["Exchange"] = "NSE"
    with pytest.raises(ValueError, match="no usable NFO"):
        flattrade_module.FlattradeExecutionClient._prepare_scrip_master(unusable)

    prepared = flattrade_module.FlattradeExecutionClient._prepare_scrip_master(
        _flattrade_master_fixture()
    )
    assert len(prepared) == 1
    client = _ready_flattrade_client(flattrade_module, monkeypatch)
    client._scrip_df = prepared
    assert (
        client.resolve_option_symbol("nifty", date(2026, 7, 16), "ce", 22500)
        == "NIFTY16JUL26C22500"
    )
    assert (
        client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500)
        == "NIFTY16JUL26C22500"
    )
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "PE", 22500) == ""
    assert client.resolve_option_symbol("NIFTY", "bad-date", "CE", "bad") == ""
    assert client.resolve_option_symbol("NIFTY", date(2026, 7, 16), "CE", 22500, "NSE") == ""

    class Session:
        closed = False

        def close(self):
            self.closed = True

    session = Session()
    client.client = session
    result = client.logout()
    assert result["stat"] == "Ok"
    assert session.closed is True
    assert client.is_logged_in is False
    assert client._access_token == ""


_DIAGNOSTIC_PATHS = (
    "Dependencies/Kotak API/diagnose_kotak_symbol.py",
    "Dependencies/Shoonya API/diagnose_shoonya_symbol.py",
    "Dependencies/Flattrade API/diagnose_flattrade_symbol.py",
)


@pytest.fixture(params=_DIAGNOSTIC_PATHS, ids=("kotak", "shoonya", "flattrade"))
def diagnostic_module(request, monkeypatch) -> ModuleType:
    """Load one CLI diagnostic without letting pytest flags become CLI input."""

    monkeypatch.setattr(sys, "argv", [str(ROOT / request.param)])
    module_name = "mat101_diagnostic_" + request.param.split("/")[-2].lower().replace(" ", "_")
    return _load_file_module(module_name, request.param)


def test_diagnostics_never_default_a_real_order_to_a_stale_lot_size(
    diagnostic_module: ModuleType,
) -> None:
    """Legacy Kotak/Shoonya diagnostics require an explicit current quantity."""

    if hasattr(diagnostic_module, "QTY"):
        assert diagnostic_module.QTY is None


class _DiagnosticClient:
    """Return a scripted sequence of typed results without contacting a broker."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, int]] = []

    def place_market_order(self, *, symbol, side, quantity, **kwargs):
        self.calls.append((symbol, side, quantity))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def extract_order_id(result) -> str:
        return str(getattr(result, "order_id", ""))


def _diagnostic_result(status_name: str, *, order_id: str = "DIAG-1"):
    """Build one internally consistent result for diagnostic behavior tests."""

    contract = _contract_module()
    status = contract.OrderStatus[status_name]
    filled = {
        "FILLED": 75,
        "PARTIAL": 25,
        "REJECTED": 0,
        "UNKNOWN": 0,
    }[status_name]
    return contract.OrderResult(
        order_id=order_id,
        requested_quantity=75,
        filled_quantity=filled,
        remaining_quantity=75 - filled,
        status=status,
        broker_state=status_name,
        reason=f"diagnostic {status_name.lower()} result",
    )


@pytest.mark.parametrize("status_name", ("PARTIAL", "UNKNOWN"))
def test_diagnostics_report_ambiguous_entry_as_indeterminate(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
    status_name: str,
) -> None:
    """An ambiguous BUY must never trigger a SELL or a no-position claim."""

    client = _DiagnosticClient([_diagnostic_result(status_name)])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert client.calls == [("NIFTY", "BUY", 75)]
    assert "indeterminate" in output.lower()
    assert "No position opened" not in output


def test_diagnostics_reserve_no_position_claim_for_explicit_rejection(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """Only a typed zero-fill rejection proves that the BUY opened nothing."""

    client = _DiagnosticClient([_diagnostic_result("REJECTED")])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert client.calls == [("NIFTY", "BUY", 75)]
    assert "No position opened" in output


def test_diagnostics_claim_flat_only_after_two_confirmed_fills(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """The happy path needs a full BUY and full SELL result."""

    client = _DiagnosticClient(
        [
            _diagnostic_result("FILLED", order_id="BUY-1"),
            _diagnostic_result("FILLED", order_id="SELL-1"),
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is True
    assert [call[1] for call in client.calls] == ["BUY", "SELL"]
    assert "-> flat" in output


@pytest.mark.parametrize("status_name", ("PARTIAL", "UNKNOWN"))
def test_diagnostics_do_not_claim_flat_for_ambiguous_exit(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
    status_name: str,
) -> None:
    """A full entry plus ambiguous exit still requires broker verification."""

    client = _DiagnosticClient(
        [_diagnostic_result("FILLED"), _diagnostic_result(status_name, order_id="EXIT-1")]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert len(client.calls) == 2
    assert "indeterminate" in output.lower()
    assert "-> flat" not in output


def test_diagnostics_treat_entry_exception_as_indeterminate(
    diagnostic_module: ModuleType,
    monkeypatch,
    capsys,
) -> None:
    """An exception after submission started cannot prove a harmless rejection."""

    client = _DiagnosticClient([TimeoutError("response lost")])
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    flat = diagnostic_module.place_round_trip_test_order(client, "NIFTY", 75)
    output = capsys.readouterr().out

    assert flat is False
    assert "indeterminate" in output.lower()
    assert "No position opened" not in output


# ---------------------------------------------------------------------------
# Dhan adapter
# ---------------------------------------------------------------------------
# Dhan's SDK collapses BOTH a genuine broker refusal and a local network
# exception into ``{'status': 'failure', ...}``.  Most of the tests below exist
# to prove the adapter keeps those two apart, because mistaking a timed-out live
# order for a rejection would send the runner back to paper while real exposure
# sits at the broker.

DHAN_SYMBOL = "NIFTY-Jul2026-24000-CE"
DHAN_SECURITY_ID = "45022"


@pytest.fixture(scope="module")
def dhan_module() -> ModuleType:
    """Return a private, network-free copy of the Dhan adapter."""

    return _load_file_module(
        "mat101_dhan_execution",
        "Dependencies/Dhan API/dhan_execution.py",
    )


class _FakeDhanSdk:
    """Minimal stand-in for the ``dhanhq`` client used by the adapter."""

    def __init__(self, replies=None, correlation_reply=None) -> None:
        self._replies = iter(replies or [])
        self._correlation_reply = correlation_reply
        self.place_calls: list[dict] = []
        self.correlation_calls: list[str] = []
        self.cancelled: list[str] = []

    def _next(self):
        return next(self._replies)

    def place_order(self, **kwargs):
        self.place_calls.append(dict(kwargs))
        return self._next()

    def get_order_by_id(self, order_id):
        return self._next()

    def get_order_by_correlationID(self, correlation_id):
        self.correlation_calls.append(str(correlation_id))
        if self._correlation_reply is None:
            raise AssertionError("correlation lookup was not expected here")
        if isinstance(self._correlation_reply, Exception):
            raise self._correlation_reply
        return self._correlation_reply

    def cancel_order(self, order_id):
        self.cancelled.append(str(order_id))
        return self._next()

    def get_order_list(self):
        return self._next()

    def get_positions(self):
        return self._next()

    def get_fund_limits(self):
        return self._next()


def _ok(data):
    """Build the SDK's success envelope."""

    return {"status": "success", "remarks": "", "data": data}


def _refused(message="DH-905 insufficient margin"):
    """Build the envelope Dhan returns when its SERVER answers and refuses.

    ``remarks`` is a dict here -- that is the only signal distinguishing this
    from a local transport failure.
    """

    return {
        "status": "failure",
        "remarks": {
            "error_code": "DH-905",
            "error_type": "Order_Error",
            "error_message": message,
        },
        "data": "",
    }


def _transport_failure(message="ConnectionError(read timed out)"):
    """Build the envelope the SDK returns when its own request RAISED.

    ``remarks`` is a plain string.  The order may well have reached the
    exchange, so this must never normalize to REJECTED.
    """

    return {"status": "failure", "remarks": message, "data": ""}


def _status_row(state, filled, quantity=75):
    return {
        "orderId": "DH-1",
        "orderStatus": state,
        "filledQty": filled,
        "quantity": quantity,
        "tradingSymbol": DHAN_SYMBOL,
        "transactionType": "BUY",
    }


def _ready_dhan_client(dhan_module: ModuleType, monkeypatch, fake):
    """Build an authenticated-looking client without touching Dhan."""

    client = dhan_module.DhanExecutionClient()
    client._client_code = "TEST"
    client.client = fake
    client.is_logged_in = True
    # The order API is driven by securityId, so the resolver's map must already
    # know this symbol -- exactly as it would after preload_scrip_master().
    client._security_id_by_symbol[DHAN_SYMBOL] = DHAN_SECURITY_ID
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)
    monkeypatch.setattr(dhan_module, "_FILL_POLL_INTERVAL", 0.001)
    return client


def test_dhan_adapter_loads_without_an_installed_sdk(
    dhan_module: ModuleType,
) -> None:
    """The adapter must import in the SDK-free broker-dependency job.

    That job installs requirements-brokers.txt alone, so ``dhanhq`` is absent
    and the import-only double above stands in; the core quality job has the
    real package and uses it as-is.  Either way the module must load, because
    every behavioral test replaces ``client`` with its own broker-shaped fake
    and none of them exercise the real SDK.
    """

    assert dhan_module.DhanContext is sys.modules["dhanhq"].DhanContext
    assert callable(dhan_module.dhanhq)


@pytest.mark.parametrize(
    ("state", "filled", "expected_status", "remaining"),
    [
        ("REJECTED", 0, "REJECTED", 75),
        ("TRADED", 75, "FILLED", 0),
        ("TRADED", 25, "PARTIAL", 50),
        # EXPIRED is not in the shared contract vocabulary; the adapter aliases
        # it to CANCELLED so a clean "never traded" is a terminal rejection
        # rather than an UNKNOWN that would freeze every live strategy.
        ("EXPIRED", 0, "REJECTED", 75),
    ],
)
def test_dhan_place_order_normalizes_terminal_outcomes(
    dhan_module: ModuleType,
    monkeypatch,
    state: str,
    filled: int,
    expected_status: str,
    remaining: int,
) -> None:
    """An acknowledgement is followed by a typed fill snapshot."""

    fake = _FakeDhanSdk(
        [
            _ok({"orderId": "DH-1", "orderStatus": "TRANSIT"}),
            _ok(_status_row(state, filled)),
        ]
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75)

    assert result.status.name == expected_status
    assert result.order_id == "DH-1"
    assert result.filled_quantity == filled
    assert result.remaining_quantity == remaining


def test_dhan_mid_fill_part_traded_snapshot_polls_to_completion(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A market order caught mid-fill is transient, not terminal.

    Returning that first PART_TRADED snapshot as the outcome would freeze
    every live entry over a healthy order; the loop must poll again.
    """

    fake = _FakeDhanSdk(
        [
            _ok({"orderId": "DH-MIDFILL", "orderStatus": "PENDING"}),
            _ok(_status_row("PART_TRADED", 25)),
            _ok(_status_row("TRADED", 75)),
        ]
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75)

    assert result.status.name == "FILLED"
    assert result.order_id == "DH-MIDFILL"
    assert result.filled_quantity == 75


def test_dhan_transport_failure_is_unknown_never_rejected(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """THE critical Dhan case: a lost response is not proof of a zero fill.

    ``DhanHTTP._send_request`` turns a post-submission socket timeout into
    ``{'status': 'failure', 'remarks': '<exception text>'}`` -- shape-identical
    to a real rejection.  Believing it would re-enter on paper while a real
    position sits at the broker.
    """

    fake = _FakeDhanSdk([_transport_failure()])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.filled_quantity == 0
    assert result.remaining_quantity == 75
    # No correlation tag was supplied, so no lookup should have been attempted.
    assert fake.correlation_calls == []


def test_dhan_transport_failure_recovers_the_order_via_correlation_id(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A completely lost placement response is still traceable by tag.

    This is the payoff of sending ``order_tag`` as Dhan's ``correlationId``:
    the order is found and its real fill reported, instead of the runner being
    frozen on an avoidable unknown.
    """

    fake = _FakeDhanSdk(
        [
            _transport_failure(),
            _ok(_status_row("TRADED", 75)),
        ],
        correlation_reply=_ok({"orderId": "DH-1", "orderStatus": "TRADED"}),
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="TAG-1")

    assert fake.correlation_calls == ["TAG-1"]
    assert result.status.name == "FILLED"
    assert result.order_id == "DH-1"
    assert result.filled_quantity == 75


def test_dhan_transport_failure_stays_unknown_when_lookup_also_fails(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Two failed observations still prove nothing about exposure."""

    fake = _FakeDhanSdk(
        [_transport_failure()],
        correlation_reply=_transport_failure(),
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="TAG-2")

    assert fake.correlation_calls == ["TAG-2"]
    assert result.status.name == "UNKNOWN"


def test_dhan_server_refusal_confirmed_by_lookup_is_rejected(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A structured refusal plus "no such order" is a provable zero fill.

    Only this combination -- Dhan's server answering with an errorCode AND the
    correlation lookup finding nothing -- earns REJECTED, which is what lets
    the runner safely fall back to paper for that trade.
    """

    fake = _FakeDhanSdk([_refused()], correlation_reply=_refused("not found"))
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="TAG-3")

    assert result.status.name == "REJECTED"
    assert result.filled_quantity == 0
    assert "insufficient margin" in result.reason


def test_dhan_server_refusal_that_still_created_an_order_is_not_rejected(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """If the lookup finds an order, the refusal envelope was misleading."""

    fake = _FakeDhanSdk(
        [_refused(), _ok(_status_row("TRADED", 75))],
        correlation_reply=_ok({"orderId": "DH-1", "orderStatus": "TRADED"}),
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="TAG-4")

    assert result.status.name == "FILLED"
    assert result.order_id == "DH-1"


def test_dhan_server_refusal_with_unverifiable_lookup_is_unknown(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A refusal we could not corroborate must not become a rejection."""

    fake = _FakeDhanSdk([_refused()], correlation_reply=_transport_failure())
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="TAG-5")

    assert result.status.name == "UNKNOWN"


def test_dhan_unknown_symbol_is_rejected_without_submitting(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Refusing beats guessing: a wrong securityId would trade another contract."""

    fake = _FakeDhanSdk([])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order("NIFTY-NEVER-RESOLVED-CE", "BUY", 75)

    assert result.status.name == "REJECTED"
    assert fake.place_calls == []
    assert "securityId" in result.reason


def test_dhan_transmits_execution_ledger_tag_and_order_fields(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """The tag reaches Dhan as correlationId, and the order is a MARKET order."""

    fake = _FakeDhanSdk(
        [
            _ok({"orderId": "DH-TAG", "orderStatus": "TRANSIT"}),
            _ok(_status_row("TRADED", 75)),
        ]
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    client.place_market_order(DHAN_SYMBOL, "BUY", 75, order_tag="STRAT-A1-E1")

    assert len(fake.place_calls) == 1
    sent = fake.place_calls[0]
    assert sent["tag"] == "STRAT-A1-E1"
    assert sent["security_id"] == DHAN_SECURITY_ID
    assert sent["transaction_type"] == "BUY"
    assert sent["order_type"] == "MARKET"
    assert sent["product_type"] == "INTRADAY"
    assert sent["exchange_segment"] == "NSE_FNO"
    assert sent["quantity"] == 75


def test_dhan_zero_broker_quantity_cannot_confirm_a_fill(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A status row disagreeing about size is malformed, not a fill."""

    fake = _FakeDhanSdk(
        [
            _ok({"orderId": "DH-QTY", "orderStatus": "TRANSIT"}),
            _ok(_status_row("TRADED", 75, quantity=10)),
        ]
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    result = client.place_market_order(DHAN_SYMBOL, "BUY", 75)

    assert result.status.name == "UNKNOWN"
    assert result.filled_quantity == 0


def test_dhan_cancel_and_reconciliation_queries_are_typed(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Cancel returns a typed snapshot; book queries return typed collections."""

    fake = _FakeDhanSdk(
        [
            _ok({"orderId": "DH-C1"}),
            _ok(_status_row("CANCELLED", 0)),
            _ok(
                [
                    {
                        "orderId": "DH-OPEN",
                        "orderStatus": "PENDING",
                        "quantity": 75,
                        "filledQty": 25,
                        "tradingSymbol": DHAN_SYMBOL,
                        "transactionType": "BUY",
                    },
                    {
                        "orderId": "DH-DONE",
                        "orderStatus": "TRADED",
                        "quantity": 75,
                        "filledQty": 75,
                        "tradingSymbol": DHAN_SYMBOL,
                        "transactionType": "BUY",
                    },
                ]
            ),
            _ok(
                [
                    {"tradingSymbol": DHAN_SYMBOL, "netQty": 75, "productType": "INTRADAY"},
                    {"tradingSymbol": "NIFTY-FLAT-PE", "netQty": 0, "productType": "INTRADAY"},
                ]
            ),
        ]
    )
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    cancelled = client.cancel_order("DH-C1", requested_quantity=75)
    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert cancelled.status.name == "REJECTED"
    assert fake.cancelled == ["DH-C1"]
    # Only the still-working order survives; the completed one is filtered out.
    assert orders.is_indeterminate is False
    assert [order.order_id for order in orders.items] == ["DH-OPEN"]
    assert orders.items[0].remaining_quantity == 50
    # A flat (netQty 0) leg is not an open position.
    assert positions.is_indeterminate is False
    assert [position.symbol for position in positions.items] == [DHAN_SYMBOL]
    assert positions.items[0].quantity == 75


def test_dhan_query_failures_are_indeterminate_not_empty(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A failed book query must never look like a flat account."""

    fake = _FakeDhanSdk([_transport_failure(), _refused()])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.is_indeterminate is True
    assert positions.is_indeterminate is True


def test_dhan_successful_empty_queries_are_distinct_from_failure(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A genuinely empty book is a success, not an indeterminate result."""

    fake = _FakeDhanSdk([_ok([]), _ok([])])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)

    orders = client.list_open_orders()
    positions = client.list_open_positions()

    assert orders.items == () and orders.is_indeterminate is False
    assert positions.items == () and positions.is_indeterminate is False


def test_dhan_deadline_constants_are_exactly_ten_seconds(
    dhan_module: ModuleType,
) -> None:
    """The SDK boundary is ten seconds and physically single-threaded."""

    client = dhan_module.DhanExecutionClient()
    assert dhan_module._API_TIMEOUT_SECONDS == 10.0
    assert dhan_module._BROKER_CALL_DEADLINE_SECONDS == 10.0
    assert client._sdk_executor._max_workers == 1


def test_dhan_login_overrides_the_sdk_sixty_second_timeout(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """The SDK ships a 60s default; leaving it would blow the 10s budget.

    ``DhanContext.__init__`` also swallows its own exceptions, so the adapter
    must verify the HTTP layer exists rather than assume it.
    """

    class _Http:
        timeout = 60

    class _Context:
        def __init__(self, client_id, access_token) -> None:
            self.http = _Http()

        def get_dhan_http(self):
            return self.http

    contexts: list[_Context] = []

    def make_context(client_id, access_token):
        context = _Context(client_id, access_token)
        contexts.append(context)
        return context

    monkeypatch.setenv("DHAN_CLIENT_CODE", "1000000001")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "not-a-real-token")
    monkeypatch.setattr(dhan_module, "DhanContext", make_context)
    monkeypatch.setattr(dhan_module, "dhanhq", lambda context: _FakeDhanSdk([_ok({})]))

    client = dhan_module.DhanExecutionClient()

    assert client.ensure_logged_in() is True
    assert contexts[0].http.timeout == 10.0


def test_dhan_login_fails_closed_on_half_built_context(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A DhanContext that swallowed its own error must not reach an order."""

    class _BrokenContext:
        def __init__(self, client_id, access_token) -> None:
            pass

        def get_dhan_http(self):
            return None

    monkeypatch.setenv("DHAN_CLIENT_CODE", "1000000001")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "not-a-real-token")
    monkeypatch.setattr(dhan_module, "DhanContext", _BrokenContext)

    client = dhan_module.DhanExecutionClient()

    assert client.ensure_logged_in() is False
    assert client.client is None


def test_dhan_login_fails_closed_without_credentials(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Missing credentials disable live trading instead of raising."""

    monkeypatch.delenv("DHAN_CLIENT_CODE", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)

    client = dhan_module.DhanExecutionClient()

    assert client.ensure_logged_in() is False


def test_dhan_login_failure_never_logs_the_access_token(
    dhan_module: ModuleType,
    monkeypatch,
    caplog,
) -> None:
    """MAT-108 parity: a token-bearing construction error must be redacted.

    The exception message here deliberately does NOT look like a
    ``token=value`` assignment, so only the known-secret replacement can
    scrub it -- proving the adapter passes the real token to ``redact_text``
    rather than relying on the pattern pass getting lucky.
    """

    canary = "CANARY-DHAN-ACCESS-TOKEN"

    class _LeakyContext:
        def __init__(self, client_id, access_token) -> None:
            raise RuntimeError(f"handshake rejected for {access_token} by gateway")

    monkeypatch.setenv("DHAN_CLIENT_CODE", "1000000001")
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", canary)
    monkeypatch.setattr(dhan_module, "DhanContext", _LeakyContext)

    client = dhan_module.DhanExecutionClient()
    with caplog.at_level(logging.ERROR):
        assert client.ensure_logged_in() is False

    assert canary not in caplog.text
    assert "<redacted>" in caplog.text


def test_dhan_logout_clears_local_session_state(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Logout is local-only: close the pool and erase every session cache."""

    class _Session:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    class _Http:
        def __init__(self) -> None:
            self.session = _Session()

    class _Context:
        def __init__(self) -> None:
            self.http = _Http()

        def get_dhan_http(self):
            return self.http

    client = _ready_dhan_client(dhan_module, monkeypatch, _FakeDhanSdk([]))
    context = _Context()
    client._context = context
    client._symbol_cache[("NIFTY",)] = DHAN_SYMBOL

    result = client.logout()

    assert result["status"] == "success"
    assert context.http.session.closed == 1
    assert client.is_logged_in is False
    assert client.client is None
    assert client._context is None
    assert client._symbol_cache == {}
    assert client._security_id_by_symbol == {}
    assert client._scrip_df is None


def test_dhan_logout_survives_a_failing_session_close(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A pool that will not close still ends with a cleared local session."""

    class _BrokenContext:
        def get_dhan_http(self):
            raise RuntimeError("connection pool already torn down")

    client = _ready_dhan_client(dhan_module, monkeypatch, _FakeDhanSdk([]))
    client._context = _BrokenContext()

    result = client.logout()

    assert result["status"] == "success"
    assert client.client is None
    assert client.is_logged_in is False


def test_dhan_total_deadline_includes_lock_wait(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """Waiting for the shared session lock cannot exceed the broker budget."""

    fake = _FakeDhanSdk([_ok({})])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)
    monkeypatch.setattr(dhan_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_a, **_k: None)

    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with client._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="lock"):
            client._call_api("fund limits", lambda sdk: sdk.get_fund_limits())
    finally:
        release.set()
        holder.join(timeout=1)
    assert time.monotonic() - started < 0.25


def test_dhan_total_deadline_includes_rate_limiter_lock_wait(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """A stuck limiter mutex cannot postpone dispatch past the deadline."""

    class _NeverCalled(_FakeDhanSdk):
        def get_fund_limits(self):
            raise AssertionError("SDK must not start after limiter deadline")

    fake = _NeverCalled([])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)
    monkeypatch.setattr(dhan_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)

    locked = threading.Event()
    release = threading.Event()

    def hold_limiter_lock() -> None:
        with client._api_limiter._lock:
            locked.set()
            release.wait(0.4)

    holder = threading.Thread(target=hold_limiter_lock)
    holder.start()
    assert locked.wait(0.2)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="limiter lock"):
            client._call_api("fund limits", lambda sdk: sdk.get_fund_limits())
    finally:
        release.set()
        holder.join(timeout=1)
    assert time.monotonic() - started < 0.25


def test_dhan_slow_response_poisons_the_session_until_reconciliation(
    dhan_module: ModuleType,
    monkeypatch,
) -> None:
    """An abandoned in-flight call blocks new orders until it has ended."""

    release = threading.Event()

    class _SlowSdk(_FakeDhanSdk):
        def get_fund_limits(self):
            release.wait(0.5)
            return _ok({})

    fake = _SlowSdk([])
    client = _ready_dhan_client(dhan_module, monkeypatch, fake)
    monkeypatch.setattr(dhan_module, "_BROKER_CALL_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(client._api_limiter, "acquire", lambda *_a, **_k: None)

    try:
        with pytest.raises(TimeoutError, match="broker deadline"):
            client._call_api("fund limits", lambda sdk: sdk.get_fund_limits())
        assert client._sdk_poisoned is True
        # A poisoned session refuses new orders outright.
        blocked = client.place_market_order(DHAN_SYMBOL, "BUY", 75)
        assert blocked.status.name == "UNKNOWN"
        assert blocked.broker_state == "SESSION_POISONED"
        # Recovery is refused while the abandoned call is still running.
        assert client.recover_after_reconciliation() is False
    finally:
        release.set()
    client._timed_out_future.result(timeout=1)
    assert client.recover_after_reconciliation() is True
