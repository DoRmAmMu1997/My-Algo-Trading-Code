"""Tests for the position-groups file model and the ``group-legs`` CLI."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest

import position_groups
import position_groups_cli
from broker_contract import BrokerQueryResult, PositionDetail

REPO_ROOT = Path(__file__).resolve().parents[2]


def leg(symbol: str = "NIFTY25N2724500CE", quantity: int = -75, **overrides) -> dict:
    document = {
        "symbol": symbol,
        "exchange": "NFO",
        "product": "NRML",
        "quantity": quantity,
        "entry_price": 120.5,
        "underlying": "NIFTY",
        "expiry": "2025-11-27",
        "strike": 24500,
        "right": "CE",
    }
    document.update(overrides)
    return document


def document(*groups: dict, unassigned: list | None = None) -> dict:
    return {"schema_version": 1, "groups": list(groups), "unassigned": unassigned or []}


def group(name: str = "IC", legs: list | None = None, **overrides) -> dict:
    body = {"name": name, "broker": "ZERODHA", "target": 15000, "booked_pnl": -2300,
            "legs": legs if legs is not None else [leg()]}
    body.update(overrides)
    return body


def detail(symbol: str = "NIFTY25N2724500CE", quantity: int = -75, **overrides) -> PositionDetail:
    fields = dict(symbol=symbol, exchange="NFO", product_type="NRML", quantity=quantity,
                  average_price=120.456, underlying="NIFTY", expiry=date(2025, 11, 27),
                  strike=24500.0, right="CE")
    fields.update(overrides)
    return PositionDetail(**fields)


def test_committed_example_file_is_valid() -> None:
    config = position_groups.load_config(REPO_ROOT / "Dependencies/position_groups.example.json")
    (only,) = config.groups
    assert only.target == 15000 and only.booked_pnl == -2300 and len(only.legs) == 4


def test_valid_document_parses_into_typed_groups() -> None:
    config = position_groups.parse_config(document(group()))
    (parsed,) = config.groups
    assert parsed.broker == "ZERODHA"
    assert parsed.legs[0].broker == "ZERODHA"  # inherited from the group
    assert parsed.legs[0].expiry == date(2025, 11, 27)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["groups"][0].update(target=0), "target"),
        (lambda d: d["groups"][0].update(target="lots"), "target"),
        (lambda d: d["groups"][0].update(broker="DHAN"), "broker"),
        (lambda d: d["groups"][0].update(booked_pnl="x"), "booked_pnl"),
        (lambda d: d["groups"][0].update(legs=[]), "non-empty"),
        (lambda d: d["groups"][0].update(name=""), "name"),
        (lambda d: d["groups"].append(group()), "used twice"),
        (lambda d: d["groups"].append(group("Other")), "already in"),
        (lambda d: d["groups"][0]["legs"][0].update(quantity=0), "quantity"),
        (lambda d: d["groups"][0]["legs"][0].update(quantity=1.5), "quantity"),
        (lambda d: d["groups"][0]["legs"][0].update(entry_price=0), "entry_price"),
        (lambda d: d["groups"][0]["legs"][0].update(exchange="NSE"), "exchange"),
        (lambda d: d["groups"][0]["legs"][0].update(product="CNC"), "product"),
        (lambda d: d["groups"][0]["legs"][0].update(expiry="27-11-2025"), "expiry"),
        (lambda d: d["groups"][0]["legs"][0].update(right="FUT"), "right"),
        (lambda d: d["groups"][0]["legs"][0].update(broker="MSTOCK"), "differs"),
        (lambda d: d["unassigned"].append(dict(leg(), broker="ZERODHA")), "already in"),
    ],
)
def test_every_invalid_document_is_rejected_whole(mutate, message) -> None:
    raw = document(group())
    mutate(raw)
    with pytest.raises(position_groups.GroupConfigError, match=message):
        position_groups.parse_config(raw)


def test_errors_list_every_problem_not_just_the_first() -> None:
    raw = document(group(target=0, legs=[leg(quantity=0)]))
    with pytest.raises(position_groups.GroupConfigError) as raised:
        position_groups.parse_config(raw)
    assert len(raised.value.problems) == 2


def test_unassigned_legs_may_carry_a_zero_broker_price() -> None:
    raw = document(unassigned=[dict(leg(entry_price=0), broker="ZERODHA")])
    (unassigned,) = position_groups.parse_config(raw).unassigned
    assert unassigned.entry_price == 0


def test_fingerprint_ignores_price_and_target_but_not_legs() -> None:
    base = position_groups.parse_config(document(group())).groups[0]
    retargeted = position_groups.parse_config(
        document(group(target=1, legs=[leg(entry_price=99)]))
    ).groups[0]
    resized = position_groups.parse_config(document(group(legs=[leg(quantity=-150)]))).groups[0]
    assert base.fingerprint == retargeted.fingerprint
    assert base.fingerprint != resized.fingerprint


def test_load_config_reports_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text("{not json")
    with pytest.raises(position_groups.GroupConfigError, match="JSON"):
        position_groups.load_config(path)


def test_missing_file_is_an_empty_valid_document(tmp_path: Path) -> None:
    config = position_groups.load_config(tmp_path / "absent.json")
    assert config.groups == () and config.unassigned == ()


def test_atomic_write_round_trips_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    position_groups.write_document_atomic(path, document(group()))
    assert json.loads(path.read_text())["groups"][0]["name"] == "IC"
    assert not (tmp_path / "g.json.tmp").exists()


def test_merge_adds_only_new_identified_legs_and_reports_the_rest() -> None:
    raw = document(group(legs=[leg(), leg("NIFTY25N2723500PE", 75, right="PE", strike=23500)]))
    before = copy.deepcopy(raw)
    report = position_groups.merge_broker_legs(
        raw,
        "zerodha",
        [
            detail(quantity=-150),  # in a group with -75: reported, not changed
            detail("NIFTY25N2725000CE", 75, strike=25000.0),  # new
            detail("INFY", 10, exchange="NSE", right="", expiry=None, strike=None),
        ],
    )
    assert [added.symbol for added in report.added] == ["NIFTY25N2725000CE"]
    assert report.unidentified == ("INFY",)
    assert report.quantity_changed == ("NIFTY25N2724500CE: file -75, broker -150",)
    assert report.no_longer_held == ("NIFTY25N2723500PE",)
    assert raw["groups"] == before["groups"]
    (new,) = raw["unassigned"]
    assert new["broker"] == "ZERODHA" and new["entry_price"] == 120.46
    position_groups.parse_config(raw)  # still valid after the merge


def test_merge_refuses_an_invalid_document() -> None:
    raw = document(group(target=-1))
    with pytest.raises(position_groups.GroupConfigError):
        position_groups.merge_broker_legs(raw, "ZERODHA", [detail()])


class FakeClient:
    def __init__(self, result) -> None:
        self.result = result

    def list_position_details(self):
        return self.result


def test_cli_fetch_writes_new_legs(tmp_path: Path, capsys) -> None:
    path = tmp_path / "g.json"
    client = FakeClient(BrokerQueryResult.success([detail()]))
    assert position_groups_cli._fetch(path, "ZERODHA", client) == 0
    assert json.loads(path.read_text())["unassigned"][0]["symbol"] == "NIFTY25N2724500CE"
    assert "1 new leg(s)" in capsys.readouterr().out


def test_cli_fetch_leaves_the_file_alone_when_the_broker_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps(document(group())))
    before = path.read_text()
    client = FakeClient(BrokerQueryResult.indeterminate("down"))
    assert position_groups_cli._fetch(path, "ZERODHA", client) == 1
    assert path.read_text() == before


def test_cli_fetch_refuses_to_touch_an_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text("{broken")
    client = FakeClient(BrokerQueryResult.success([detail()]))
    assert position_groups_cli._fetch(path, "ZERODHA", client) == 1
    assert path.read_text() == "{broken"


def test_cli_check_reports_problems(tmp_path: Path, capsys) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps(document(group(target=0))))
    assert position_groups_cli.main(["--check", "--file", str(path)]) == 1
    assert "target" in capsys.readouterr().out
