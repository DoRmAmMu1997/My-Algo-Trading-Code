"""Tests for the shared single-line ``.env`` rewrite (temp files only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from env_file import replace_env_value


def test_replaces_the_live_line_and_keeps_comments_and_order(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# TOKEN=commented\nA=1\nTOKEN=old\nB=2\n")
    replace_env_value(env, "TOKEN", "new")
    assert env.read_text() == "# TOKEN=commented\nA=1\nTOKEN=new\nB=2\n"


def test_appends_a_missing_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    replace_env_value(env, "TOKEN", "t")
    assert env.read_text() == "A=1\nTOKEN=t\n"


def test_keeps_a_file_without_a_trailing_newline_that_way(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("TOKEN=old")
    replace_env_value(env, "TOKEN", "new")
    assert env.read_text() == "TOKEN=new"


def test_creates_a_missing_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    replace_env_value(env, "TOKEN", "t")
    assert env.read_text() == "TOKEN=t\n"


def test_does_not_match_a_key_that_merely_shares_a_prefix(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("TOKEN_2=keep\n")
    replace_env_value(env, "TOKEN", "t")
    assert env.read_text() == "TOKEN_2=keep\nTOKEN=t\n"


@pytest.mark.parametrize("value", ["a\nLIVE_TRADING_ENABLED=true", "a\rb"])
def test_refuses_a_value_with_a_line_break(tmp_path: Path, value: str) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\n")
    with pytest.raises(ValueError, match="multi-line"):
        replace_env_value(env, "TOKEN", value)
    assert env.read_text() == "A=1\n"
