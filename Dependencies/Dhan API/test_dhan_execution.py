"""Offline unit tests for the Dhan execution helper's local logic.

These cover the parts that never reach the network: rate limiting, instrument
CSV parsing, envelope classification, order-state aliasing, and order-id
extraction.  The broker-contract conformance tests (place/status/cancel and the
deadline behaviour) live in ``Dependencies/test_broker_contract.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module() -> ModuleType:
    """Load the adapter from its space-containing folder."""

    path = ROOT / "Dependencies" / "Dhan API" / "dhan_execution.py"
    spec = importlib.util.spec_from_file_location("dhan_execution_unit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dhan_execution_unit_test"] = module
    spec.loader.exec_module(module)
    return module


de = _load_module()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic monotonic clock whose sleeps just advance the time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _limiter(clock: _FakeClock, per_second=2, per_minute=5, max_wait=2.0):
    return de._RollingWindowRateLimiter(
        per_second,
        per_minute,
        max_wait,
        clock=clock,
        sleeper=clock.sleep,
        label="test",
    )


def test_rate_limiter_grants_calls_inside_the_budget() -> None:
    clock = _FakeClock()
    limiter = _limiter(clock)

    limiter.acquire()
    limiter.acquire()

    assert clock.slept == []


def test_rate_limiter_waits_for_the_next_one_second_window() -> None:
    clock = _FakeClock()
    limiter = _limiter(clock)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert clock.slept and clock.slept[0] == pytest.approx(1.0)


def test_rate_limiter_raises_rather_than_waiting_past_the_cap() -> None:
    """A stale live order is worse than a clear, immediate refusal."""

    clock = _FakeClock()
    limiter = _limiter(clock, per_second=1, per_minute=1, max_wait=0.5)
    limiter.acquire()

    with pytest.raises(RuntimeError, match="rate limit exhausted"):
        limiter.acquire()


# ---------------------------------------------------------------------------
# Envelope classification -- the safety-critical discrimination
# ---------------------------------------------------------------------------


def test_success_envelope_is_ok_and_carries_data() -> None:
    outcome, data, _reason = de._classify_envelope(
        {"status": "success", "remarks": "", "data": {"orderId": "DH-1"}}
    )

    assert outcome == "ok"
    assert data == {"orderId": "DH-1"}


def test_dict_remarks_means_the_server_answered_and_refused() -> None:
    outcome, _data, reason = de._classify_envelope(
        {
            "status": "failure",
            "remarks": {
                "error_code": "DH-905",
                "error_type": "Order_Error",
                "error_message": "insufficient margin",
            },
            "data": "",
        }
    )

    assert outcome == "refused"
    assert "insufficient margin" in reason


def test_str_remarks_means_indeterminate_not_refused() -> None:
    """The SDK's bare ``except Exception`` produces this shape.

    Treating it as a refusal is exactly the bug that would report a timed-out
    but live order as a harmless rejection.
    """

    outcome, _data, reason = de._classify_envelope(
        {"status": "failure", "remarks": "ConnectionError(read timed out)", "data": ""}
    )

    assert outcome == "indeterminate"
    assert "timed out" in reason


@pytest.mark.parametrize("envelope", [None, "boom", 42, [], {"unexpected": True}])
def test_malformed_envelopes_are_indeterminate(envelope) -> None:
    outcome, _data, _reason = de._classify_envelope(envelope)

    assert outcome == "indeterminate"


# ---------------------------------------------------------------------------
# Order-state aliasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EXPIRED", "CANCELLED"),
        ("PART_TRADED", "PARTIAL"),
        ("part-traded", "PARTIAL"),
        ("TRADED", "TRADED"),
        ("REJECTED", "REJECTED"),
        # Transient states stay themselves so the fill loop keeps polling.
        ("PENDING", "PENDING"),
        ("TRANSIT", "TRANSIT"),
        (None, ""),
    ],
)
def test_state_aliasing(raw, expected: str) -> None:
    assert de._alias_state(raw) == expected


# ---------------------------------------------------------------------------
# Quantity and field parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(75, 75), ("75", 75), (75.0, 75), ("75.5", None), (True, None), ("abc", None), (None, None)],
)
def test_exact_int_rejects_anything_but_whole_numbers(value, expected) -> None:
    assert de._exact_int(value) == expected


def test_first_present_tolerates_field_casing_differences() -> None:
    """Dhan's field casing is documented only in prose, so reads are lenient."""

    assert de._first_present({"filledQty": 25}, de._FILLED_QTY_KEYS) == 25
    assert de._first_present({"filled_qty": 25}, de._FILLED_QTY_KEYS) == 25
    assert de._first_present({"orderStatus": "TRADED"}, de._ORDER_STATE_KEYS) == "TRADED"
    # Empty strings count as absent so a blank field cannot masquerade as data.
    assert de._first_present({"filledQty": ""}, de._FILLED_QTY_KEYS) is None
    assert de._first_present({}, de._FILLED_QTY_KEYS) is None


def test_extract_order_id_searches_nested_payloads() -> None:
    assert de.dhan_execution_client.extract_order_id({"orderId": " DH-1 "}) == "DH-1"
    assert de.dhan_execution_client.extract_order_id({"data": {"orderId": "DH-2"}}) == "DH-2"
    assert de.dhan_execution_client.extract_order_id([{"orderId": "DH-3"}]) == "DH-3"
    assert de.dhan_execution_client.extract_order_id({}) == ""
    assert de.dhan_execution_client.extract_order_id(None) == ""


# ---------------------------------------------------------------------------
# Instrument-master parsing
# ---------------------------------------------------------------------------

_HEADER = (
    "EXCH_ID,SEGMENT,INSTRUMENT,SYMBOL_NAME,DISPLAY_NAME,SM_EXPIRY_DATE,"
    "LOT_SIZE,SECURITY_ID,STRIKE_PRICE,OPTION_TYPE,UNDERLYING_SYMBOL\n"
)
_NIFTY_CE = (
    "NSE,D,OPTIDX,NIFTY-Jul2026-24000-CE,NIFTY 24000 CALL,2026-07-21,"
    "65,45001,24000,CE,NIFTY\n"
)
_NIFTY_PE = (
    "NSE,D,OPTIDX,NIFTY-Jul2026-24000-PE,NIFTY 24000 PUT,2026-07-21,"
    "65,45002,24000,PE,NIFTY\n"
)
_BNF_CE = (
    "NSE,D,OPTIDX,BANKNIFTY-Jul2026-59500-CE,BANKNIFTY 59500 CALL,2026-07-28,"
    "35,45003,59500,CE,BANKNIFTY\n"
)
# Rows that must be filtered out: a stock option, an equity row, and a future.
_NOISE = (
    "NSE,D,OPTSTK,RELIANCE-Jul2026-3000-CE,RELIANCE 3000 CALL,2026-07-21,"
    "250,45004,3000,CE,RELIANCE\n"
    "NSE,E,EQUITY,RELIANCE,Reliance Industries,,1,2885,0,,RELIANCE\n"
    "NSE,D,FUTIDX,NIFTY-Jul2026-FUT,NIFTY FUT,2026-07-21,65,45005,0,,NIFTY\n"
)


def _write_master(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "all_instrument 2026-07-18.csv"
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def test_prepare_scrip_master_keeps_only_index_options(tmp_path: Path) -> None:
    frame = de.DhanExecutionClient._prepare_scrip_master(
        _write_master(tmp_path, _NIFTY_CE + _NIFTY_PE + _BNF_CE + _NOISE)
    )

    assert len(frame) == 3
    assert set(frame["_underlying_u"]) == {"NIFTY", "BANKNIFTY"}
    assert set(frame["_option_u"]) == {"CE", "PE"}
    # Security ids must survive as real integers for the order API.
    assert sorted(frame["security_id"]) == [45001, 45002, 45003]
    assert frame["security_id"].dtype.kind == "i"


def test_prepare_scrip_master_raises_on_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "all_instrument 2026-07-18.csv"
    path.write_text("EXCH_ID,SEGMENT\nNSE,D\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        de.DhanExecutionClient._prepare_scrip_master(path)


def test_prepare_scrip_master_raises_when_no_option_rows_survive(tmp_path: Path) -> None:
    """An all-noise catalogue must fail loudly, not resolve to nothing later."""

    with pytest.raises(ValueError, match="no usable index-option rows"):
        de.DhanExecutionClient._prepare_scrip_master(_write_master(tmp_path, _NOISE))


def test_latest_instrument_master_path_picks_the_newest(tmp_path: Path, monkeypatch) -> None:
    for name in ("all_instrument 2026-07-16.csv", "all_instrument 2026-07-18.csv"):
        (tmp_path / name).write_text(_HEADER, encoding="utf-8")
    monkeypatch.setattr(de, "_DEPENDENCIES_DIR", tmp_path)

    picked = de.DhanExecutionClient._latest_instrument_master_path()

    assert picked is not None and picked.name == "all_instrument 2026-07-18.csv"


def test_missing_instrument_master_disables_live_trading(tmp_path: Path, monkeypatch) -> None:
    """No catalogue means preload fails closed rather than resolving blindly."""

    monkeypatch.setattr(de, "_DEPENDENCIES_DIR", tmp_path)
    client = de.DhanExecutionClient()

    assert client._ensure_scrip_master_locked() is False
    assert client._scrip_df is None


# ---------------------------------------------------------------------------
# Symbol -> securityId bridge
# ---------------------------------------------------------------------------


def test_resolver_populates_the_security_id_map(tmp_path: Path, monkeypatch) -> None:
    """The resolver returns a readable symbol that still maps to the right id."""

    monkeypatch.setattr(de, "_DEPENDENCIES_DIR", tmp_path)
    _write_master(tmp_path, _NIFTY_CE + _NIFTY_PE + _BNF_CE)
    client = de.DhanExecutionClient()
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)

    symbol = client.resolve_option_symbol("NIFTY", "2026-07-21", "CE", 24000)

    assert symbol == "NIFTY-Jul2026-24000-CE"
    assert client._security_id_for(symbol) == "45001"
    # The BankNIFTY mirror leg must resolve from the same catalogue.
    bnf = client.resolve_option_symbol("BANKNIFTY", "2026-07-28", "CE", 59500)
    assert bnf == "BANKNIFTY-Jul2026-59500-CE"
    assert client._security_id_for(bnf) == "45003"


def test_resolver_returns_empty_string_on_a_miss(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(de, "_DEPENDENCIES_DIR", tmp_path)
    _write_master(tmp_path, _NIFTY_CE)
    client = de.DhanExecutionClient()
    monkeypatch.setattr(client, "ensure_logged_in", lambda: True)

    assert client.resolve_option_symbol("NIFTY", "2026-07-21", "CE", 99999) == ""
    assert client.resolve_option_symbol("NIFTY", "2026-12-31", "CE", 24000) == ""
    assert client.resolve_option_symbol("NIFTY", "2026-07-21", "XX", 24000) == ""
    assert client._security_id_for("NIFTY-NOT-A-SYMBOL") == ""


# ---------------------------------------------------------------------------
# Diagnostic pre-flight guards
# ---------------------------------------------------------------------------
# These exist because Dhan reports a bad quantity as a generic DH-905
# Input_Exception whose message has been observed to read "Invalid IP",
# which points the operator at a network fault that does not exist.


def _diagnostic() -> ModuleType:
    path = ROOT / "Dependencies" / "Dhan API" / "diagnose_dhan_symbol.py"
    spec = importlib.util.spec_from_file_location("diagnose_dhan_unit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_dhan_unit_test"] = module
    # The diagnostic imports the adapter by bare name from its own folder.
    sys.modules.setdefault("dhan_execution", de)
    spec.loader.exec_module(module)
    return module


diag = _diagnostic()


def test_quantity_check_is_the_shared_one_not_a_local_copy() -> None:
    """Behaviour lives in Dependencies/test_diagnostic_preflight.py.

    All four diagnostics import one implementation; this only pins that the
    Dhan one is wired to it rather than carrying a copy that could drift.
    """

    # Imported here, not at module scope: it resolves through the sys.path the
    # adapter sets up when this test module loads the diagnostic.
    import diagnostic_preflight

    assert diag.validate_quantity_for_lot is diagnostic_preflight.validate_quantity_for_lot


class _MarginClient:
    """Stands in for the adapter's _call_api during margin pre-checks."""

    def __init__(self, envelope) -> None:
        self.envelope = envelope
        self.calls: list[dict] = []

    def _call_api(self, _operation, call, **_kwargs):
        if isinstance(self.envelope, Exception):
            raise self.envelope

        class _Sdk:
            def margin_calculator(_self, **kwargs):
                self.calls.append(dict(kwargs))
                return self.envelope

        return call(_Sdk())


def _margin_payload(required: float, available: float) -> dict:
    """Build a margin reply with Dhan's real (unsigned) insufficientBalance.

    Reproducing the quirk faithfully is the point: `insufficientBalance` is
    abs(required - available), so it is positive even when the order is
    comfortably affordable.
    """

    return {
        "status": "success",
        "remarks": "",
        "data": {
            "totalMargin": required,
            "availableBalance": available,
            "insufficientBalance": round(abs(required - available), 2),
            "leverage": "1X",
        },
    }


def test_margin_precheck_blocks_an_unaffordable_entry() -> None:
    client = _MarginClient(_margin_payload(required=21895.25, available=0.39))

    placeable, message = diag.check_entry_margin(client, security_id="57340", quantity=65)

    assert placeable is False
    assert "Insufficient funds" in message
    assert "short by 21894.86" in message
    # It must ask about the real contract and quantity, and create nothing.
    assert client.calls[0]["security_id"] == "57340"
    assert client.calls[0]["quantity"] == 65
    assert client.calls[0]["transaction_type"] == "BUY"


def test_margin_precheck_allows_an_affordable_entry() -> None:
    """Regression: a surplus must not be misread as a shortfall.

    These are the exact figures from a real Dhan reply where the order was
    affordable (2808 needed, 10000.39 on hand) yet insufficientBalance came
    back as 7192.39. Trusting that field blocked a perfectly valid order.
    """

    client = _MarginClient(_margin_payload(required=2808.0, available=10000.39))

    placeable, message = diag.check_entry_margin(client, security_id="57360", quantity=65)

    assert placeable is True
    assert "OK" in message
    assert "required 2808.00" in message and "available 10000.39" in message


@pytest.mark.parametrize(
    ("required", "available", "placeable"),
    [
        (2808.0, 10000.39, True),      # comfortable surplus
        (11232.0, 10000.39, False),    # just short
        (10000.39, 10000.39, True),    # exactly affordable
        (10000.40, 10000.39, False),   # one paisa short
        (0.0, 0.0, True),              # nothing required
    ],
)
def test_margin_precheck_compares_balances_not_the_unsigned_field(
    required: float,
    available: float,
    placeable: bool,
) -> None:
    client = _MarginClient(_margin_payload(required=required, available=available))

    got, _message = diag.check_entry_margin(client, security_id="57360", quantity=65)

    assert got is placeable


@pytest.mark.parametrize(
    "envelope",
    [
        RuntimeError("network down"),
        {"status": "failure", "remarks": "ConnectionError(...)", "data": ""},
        {"status": "success", "remarks": "", "data": "not-a-dict"},
        # Missing either figure means the comparison cannot be made.
        {"status": "success", "remarks": "", "data": {"totalMargin": 100}},
        {"status": "success", "remarks": "", "data": {"availableBalance": 100}},
        {"status": "success", "remarks": "", "data": {"totalMargin": "n/a", "availableBalance": 100}},
    ],
)
def test_margin_precheck_that_cannot_run_does_not_block(envelope) -> None:
    """An advisory check that fails must not stop the operator.

    The broker is still the real gate, and the typed-YES confirmation is
    still ahead -- so failing closed here would add friction without safety.
    """

    client = _MarginClient(envelope)

    placeable, message = diag.check_entry_margin(client, security_id="57340", quantity=65)

    assert placeable is True
    assert "continuing" in message
