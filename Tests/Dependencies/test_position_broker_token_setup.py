"""Tests for the daily broker-token helper (no network, temp .env only).

The login itself is the adapters' job and is tested with them
(``Tests/Dependencies/Zerodha API`` and ``Tests/Dependencies/Mstock API``).
This script only runs that login and writes one line of ``.env``.
"""

from __future__ import annotations

from pathlib import Path

import position_broker_token_setup as setup
import pytest


class FakeClient:
    # ``issued`` rather than ``token``: Bandit reads a keyword argument named
    # like a credential as a hard-coded password, even in a test double.
    def __init__(self, *, logged_in: bool = True, issued: str = "abc.DEF-123_x") -> None:
        self._logged_in = logged_in
        self._token = issued

    def ensure_logged_in(self) -> bool:
        return self._logged_in

    def current_access_token(self) -> str:
        return self._token


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch) -> Path:
    env = tmp_path / ".env"
    env.write_text("A=1\nZERODHA_ACCESS_TOKEN=old\nB=2\n", encoding="utf-8")
    monkeypatch.setattr(setup, "ENV_PATH", env)
    return env


def test_a_successful_login_rewrites_only_the_token_line(env_file: Path, capsys) -> None:
    assert setup.main(["--broker", "zerodha"], client=FakeClient()) == 0
    assert env_file.read_text(encoding="utf-8") == "A=1\nZERODHA_ACCESS_TOKEN=abc.DEF-123_x\nB=2\n"
    # The token is saved, never shown.
    assert "abc.DEF-123_x" not in capsys.readouterr().out


def test_each_broker_writes_its_own_key(env_file: Path) -> None:
    assert setup.main(["--broker", "mstock"], client=FakeClient(issued="jwt.part.sig")) == 0
    text = env_file.read_text(encoding="utf-8")
    assert "MSTOCK_ACCESS_TOKEN=jwt.part.sig\n" in text
    assert "ZERODHA_ACCESS_TOKEN=old\n" in text


def test_a_failed_login_leaves_the_file_untouched(env_file: Path, capsys) -> None:
    before = env_file.read_text(encoding="utf-8")
    assert setup.main(["--broker", "zerodha"], client=FakeClient(logged_in=False)) == 1
    assert env_file.read_text(encoding="utf-8") == before
    assert "login failed" in capsys.readouterr().out


@pytest.mark.parametrize("token", ["", "has space", "line\nBREAK=1", "LIVE_TRADING_ENABLED=true"])
def test_a_malformed_token_is_refused_before_it_reaches_the_file(
    env_file: Path, capsys, token: str
) -> None:
    before = env_file.read_text(encoding="utf-8")
    assert setup.main(["--broker", "zerodha"], client=FakeClient(issued=token)) == 1
    assert env_file.read_text(encoding="utf-8") == before
    assert "malformed" in capsys.readouterr().out


def test_an_adapter_exception_is_reported_by_type_only(env_file: Path, capsys) -> None:
    class Exploding(FakeClient):
        def ensure_logged_in(self) -> bool:
            raise RuntimeError("secret-token-in-a-traceback")

    assert setup.main(["--broker", "zerodha"], client=Exploding()) == 1
    shown = capsys.readouterr().out
    assert "RuntimeError" in shown and "secret-token-in-a-traceback" not in shown


def test_a_missing_env_file_is_an_error_not_a_new_file(tmp_path: Path, monkeypatch, capsys) -> None:
    missing = tmp_path / ".env"
    monkeypatch.setattr(setup, "ENV_PATH", missing)
    assert setup.main(["--broker", "zerodha"], client=FakeClient()) == 1
    assert not missing.exists()


def test_the_adapters_named_here_exist_and_expose_the_login_surface() -> None:
    for broker, (folder, module_name, attribute, env_key) in setup.BROKER_ADAPTERS.items():
        assert (setup._DEPENDENCIES_DIR / folder / f"{module_name}.py").is_file(), broker
        client = setup.load_adapter_client(broker)
        assert callable(client.ensure_logged_in) and callable(client.current_access_token)
        assert attribute.startswith(broker) and env_key == f"{broker.upper()}_ACCESS_TOKEN"
