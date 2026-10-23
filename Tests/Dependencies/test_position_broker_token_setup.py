"""Tests for the daily position-broker login helper (no network, temp .env only)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import position_broker_token_setup as setup
import pytest


class FakeResponse:
    def __init__(self, body: Any, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> Any:
        return self._body


def test_write_env_value_replaces_in_place_and_keeps_comments(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# ZERODHA_ACCESS_TOKEN=commented\nA=1\nZERODHA_ACCESS_TOKEN=old\nB=2\n")
    setup.write_env_value("ZERODHA_ACCESS_TOKEN", "new", env)
    assert env.read_text() == "# ZERODHA_ACCESS_TOKEN=commented\nA=1\nZERODHA_ACCESS_TOKEN=new\nB=2\n"


def test_write_env_value_appends_a_missing_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    setup.write_env_value("MSTOCK_ACCESS_TOKEN", "t", env)
    assert env.read_text() == "A=1\nMSTOCK_ACCESS_TOKEN=t\n"


def test_read_env_file_strips_quotes_and_skips_comments(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# X=1\nZERODHA_API_KEY="abc"\n\nJUNK\n')
    assert setup.read_env_file(env) == {"ZERODHA_API_KEY": "abc"}


@pytest.mark.parametrize(
    ("pasted", "expected"),
    [
        ("https://127.0.0.1/?action=login&type=login&status=success&request_token=RT1", "RT1"),
        ("  RT2  ", "RT2"),
        ("status=success&request_token=RT3", "RT3"),
        ("", ""),
    ],
)
def test_extract_request_token(pasted: str, expected: str) -> None:
    assert setup.extract_request_token(pasted) == expected


def test_zerodha_login_exchanges_the_request_token_with_the_documented_checksum() -> None:
    sent: dict[str, Any] = {}

    def post(url, *, data, headers, timeout):
        sent.update(url=url, data=data, headers=headers, timeout=timeout)
        return FakeResponse({"status": "success", "data": {"access_token": "ACCESS"}})

    opened: list[str] = []
    token = setup.zerodha_login(
        {"ZERODHA_API_KEY": "key", "ZERODHA_API_SECRET": "secret"},
        ask=lambda _prompt: "https://x/?request_token=RT",
        post=post,
        open_url=lambda url: opened.append(url) or True,
    )
    assert token == "ACCESS"
    assert opened == ["https://kite.zerodha.com/connect/login?v=3&api_key=key"]
    assert sent["data"]["checksum"] == hashlib.sha256(b"keyRTsecret").hexdigest()
    assert sent["headers"] == {"X-Kite-Version": "3"}
    assert sent["timeout"] == 10.0


def test_login_url_with_the_api_key_is_never_printed(capsys) -> None:
    def post(url, *, data, headers, timeout):
        return FakeResponse({"status": "success", "data": {"access_token": "A"}})

    for browser_opened in (True, False):
        setup.zerodha_login(
            {"ZERODHA_API_KEY": "kite_key_value", "ZERODHA_API_SECRET": "s"},
            ask=lambda _p: "RT",
            post=post,
            open_url=lambda _url, result=browser_opened: result,
        )
    assert "kite_key_value" not in capsys.readouterr().out


def test_zerodha_login_needs_key_and_secret() -> None:
    with pytest.raises(RuntimeError, match="ZERODHA_API_SECRET"):
        setup.zerodha_login({"ZERODHA_API_KEY": "key"}, ask=lambda _p: "RT", open_url=lambda _u: True)


def test_refused_login_raises_without_echoing_the_secret() -> None:
    def post(url, *, data, headers, timeout):
        return FakeResponse({"status": "error", "message": "bad checksum for secret123"}, 403)

    with pytest.raises(RuntimeError) as raised:
        setup.zerodha_login(
            {"ZERODHA_API_KEY": "key", "ZERODHA_API_SECRET": "secret123"},
            ask=lambda _p: "RT",
            post=post,
            open_url=lambda _u: True,
        )
    assert "secret123" not in str(raised.value)


def test_main_writes_the_token_into_env(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("ZERODHA_API_KEY=key\nZERODHA_API_SECRET=secret\n")
    monkeypatch.setattr(setup, "ENV_PATH", env)
    monkeypatch.setitem(
        setup.BROKER_LOGINS, "zerodha", ("ZERODHA_ACCESS_TOKEN", lambda _env: "FRESH")
    )
    assert setup.main(["--broker", "zerodha"]) == 0
    assert "ZERODHA_ACCESS_TOKEN=FRESH" in env.read_text()


def test_main_failure_leaves_env_untouched(tmp_path: Path, monkeypatch, capsys) -> None:
    env = tmp_path / ".env"
    env.write_text("ZERODHA_API_KEY=key\n")
    monkeypatch.setattr(setup, "ENV_PATH", env)
    assert setup.main(["--broker", "zerodha"]) == 1
    assert env.read_text() == "ZERODHA_API_KEY=key\n"
    assert "ERROR" in capsys.readouterr().out


def _mstock_post(calls: list[tuple[str, dict]]):
    def post(url, *, data, headers, timeout):
        calls.append((url.rsplit("/typea", 1)[-1], data))
        assert headers == {"X-Mirae-Version": "1"}
        if url.endswith("/connect/login"):
            return FakeResponse({"status": "success", "data": {"cid": "X"}})
        return FakeResponse({"status": "success", "data": {"access_token": "MTOKEN"}})

    return post


def test_mstock_login_uses_otp_by_default() -> None:
    calls: list[tuple[str, dict]] = []
    token = setup.mstock_login(
        {"MSTOCK_API_KEY": "k", "MSTOCK_USERNAME": "u"},
        ask=lambda _p: "123456",
        ask_secret=lambda _p: "pw",
        post=_mstock_post(calls),
    )
    assert token == "MTOKEN"
    assert calls == [
        ("/connect/login", {"username": "u", "password": "pw"}),
        ("/session/token", {"api_key": "k", "request_token": "123456", "checksum": "L"}),
    ]


def test_mstock_login_uses_totp_when_enabled() -> None:
    calls: list[tuple[str, dict]] = []
    setup.mstock_login(
        {"MSTOCK_API_KEY": "k", "MSTOCK_USERNAME": "u", "MSTOCK_TOTP_ENABLED": "true"},
        ask=lambda _p: "654321",
        ask_secret=lambda _p: "pw",
        post=_mstock_post(calls),
    )
    assert calls[1] == ("/session/verifytotp", {"api_key": "k", "totp": "654321"})


def test_mstock_bad_password_raises_without_echoing_it() -> None:
    def post(url, *, data, headers, timeout):
        return FakeResponse({"status": "error", "message": "Invalid password hunter2"})

    with pytest.raises(RuntimeError) as raised:
        setup.mstock_login(
            {"MSTOCK_API_KEY": "k", "MSTOCK_USERNAME": "u"},
            ask=lambda _p: "1",
            ask_secret=lambda _p: "hunter2",
            post=post,
        )
    assert "hunter2" not in str(raised.value)


@pytest.mark.parametrize("token", ["abc\nLIVE_TRADING_ENABLED=true", "a=b", "has space", ""])
def test_malformed_token_never_reaches_env(tmp_path: Path, monkeypatch, token: str) -> None:
    env = tmp_path / ".env"
    env.write_text("ZERODHA_API_KEY=key\n")
    monkeypatch.setattr(setup, "ENV_PATH", env)
    monkeypatch.setitem(setup.BROKER_LOGINS, "zerodha", ("ZERODHA_ACCESS_TOKEN", lambda _env: token))
    assert setup.main(["--broker", "zerodha"]) == 1
    assert env.read_text() == "ZERODHA_API_KEY=key\n"


def test_jwt_shaped_token_is_accepted(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("")
    monkeypatch.setattr(setup, "ENV_PATH", env)
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig-_x"
    monkeypatch.setitem(setup.BROKER_LOGINS, "mstock", ("MSTOCK_ACCESS_TOKEN", lambda _env: jwt))
    assert setup.main(["--broker", "mstock"]) == 0
    assert env.read_text() == f"MSTOCK_ACCESS_TOKEN={jwt}\n"
