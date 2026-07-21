"""Tests for the read-only `.env` audit behind `python algo.py check-env`.

The audit is a diagnostic, so the properties that matter are: it finds real
drift, it never leaks a value out of the operator's `.env`, and it changes
nothing on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from check_env_config import (
    audit,
    env_keys_read_by,
    main,
    parse_env_file,
    render,
    source_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_repo(tmp_path: Path, *, template: str, live: str | None, code: str = "") -> Path:
    """Build a throwaway repo skeleton the audit can run against."""
    (tmp_path / "Dependencies").mkdir()
    (tmp_path / "Dependencies" / "env.example").write_text(template, encoding="utf-8")
    if live is not None:
        (tmp_path / "Dependencies" / ".env").write_text(live, encoding="utf-8")
    (tmp_path / "Nifty Multi Strategy Front Test - Master File.py").write_text(
        code, encoding="utf-8"
    )
    return tmp_path


class TestEnvFileParsing:
    def test_reads_pairs_and_ignores_comments_and_blanks(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            "# a comment\n\nALPHA=1\n  BETA = two  \nGAMMA=\nnot_a_pair\n",
            encoding="utf-8",
        )
        assert parse_env_file(path) == {"ALPHA": "1", "BETA": "two", "GAMMA": ""}

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert parse_env_file(tmp_path / "nope.env") == {}


class TestEnvKeyExtraction:
    def test_extracts_literal_keys_from_every_reader(self, tmp_path):
        path = tmp_path / "mod.py"
        path.write_text(
            "A = _env_str('S_KEY', 'x')\n"
            "B = _env_int('I_KEY', 1)\n"
            "C = _env_float('F_KEY', 1.0)\n"
            "D = _env_bool('B_KEY', True)\n"
            "E = _scaled_int('PFX', 'SCALED_KEY', 1)\n"
            "F = _scaled_float('PFX', 'SCALED_F_KEY', 1.0)\n",
            encoding="utf-8",
        )
        assert env_keys_read_by(path) == {
            "S_KEY", "I_KEY", "F_KEY", "B_KEY", "SCALED_KEY", "SCALED_F_KEY",
        }

    def test_skips_fstring_keys_and_unrelated_calls(self, tmp_path):
        path = tmp_path / "mod.py"
        path.write_text(
            "A = _env_int(f'{prefix}_LOTS', 1)\n"  # per-strategy, not a literal
            "B = os.getenv('NOT_A_READER')\n"
            "C = _env_str('REAL_KEY', '')\n",
            encoding="utf-8",
        )
        assert env_keys_read_by(path) == {"REAL_KEY"}

    def test_unparseable_file_is_skipped_rather_than_raising(self, tmp_path):
        path = tmp_path / "broken.py"
        path.write_text("def (((\n", encoding="utf-8")
        assert env_keys_read_by(path) == set()


class TestAudit:
    def test_reports_missing_unknown_and_undocumented(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            template="SHARED=1\nONLY_IN_TEMPLATE=7\n",
            live="SHARED=1\nONLY_IN_LIVE=2\n",
            code="X = _env_int('READ_BUT_UNDOCUMENTED', 3)\n",
        )
        findings = audit(repo)
        assert findings["missing"] == ["ONLY_IN_TEMPLATE"]
        assert findings["unknown"] == ["ONLY_IN_LIVE"]
        assert findings["undocumented"] == ["READ_BUT_UNDOCUMENTED"]

    def test_clean_repo_reports_nothing(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            template="SHARED=1\n",
            live="SHARED=99\n",  # differing VALUES are the operator's business
            code="X = _env_int('SHARED', 3)\n",
        )
        assert audit(repo) == {"missing": [], "unknown": [], "undocumented": []}

    def test_this_repository_is_currently_clean(self):
        """The real checkout must stay drift-free (CI gates the same thing)."""
        assert audit(REPO_ROOT)["undocumented"] == []

    def test_source_files_include_the_master_and_dependencies(self):
        names = {path.name for path in source_files(REPO_ROOT)}
        assert "Nifty Multi Strategy Front Test - Master File.py" in names
        assert "check_env_config.py" in names


class TestSecretSafety:
    def test_never_prints_a_value_from_the_live_env(self, tmp_path, capsys):
        """The whole report must be safe to paste into an issue."""
        secret = "super-secret-token-value"
        repo = _make_repo(
            tmp_path,
            template="DHAN_ACCESS_TOKEN=\nMISSING_ONE=template-value\n",
            live=f"DHAN_ACCESS_TOKEN={secret}\nSTRAY_KEY={secret}\n",
        )
        exit_code = main(["--repo-root", str(repo)])
        output = capsys.readouterr().out

        assert exit_code == 1
        # Key names are reported ...
        assert "STRAY_KEY" in output
        assert "MISSING_ONE" in output
        # ... but no value out of .env ever is.
        assert secret not in output
        # Template values ARE shown, so a missing line can be copied across.
        assert "template-value" in output


class TestCommandLine:
    def test_clean_repo_exits_zero(self, tmp_path, capsys):
        repo = _make_repo(tmp_path, template="SHARED=1\n", live="SHARED=1\n")
        assert main(["--repo-root", str(repo)]) == 0
        assert "Clean" in capsys.readouterr().out

    def test_drift_exits_one(self, tmp_path, capsys):
        repo = _make_repo(tmp_path, template="A=1\nB=2\n", live="A=1\n")
        assert main(["--repo-root", str(repo)]) == 1
        assert "finding(s)" in capsys.readouterr().out

    def test_absent_live_env_explains_how_to_create_it(self, tmp_path, capsys):
        repo = _make_repo(tmp_path, template="A=1\n", live=None)
        assert main(["--repo-root", str(repo)]) == 1
        assert "copy" in capsys.readouterr().out.lower()

    def test_absent_template_is_a_hard_error(self, tmp_path, capsys):
        (tmp_path / "Dependencies").mkdir()
        assert main(["--repo-root", str(tmp_path)]) == 2
        assert "template not found" in capsys.readouterr().out

    def test_audit_never_writes_to_disk(self, tmp_path):
        repo = _make_repo(tmp_path, template="A=1\nB=2\n", live="A=1\nC=3\n")
        before = {
            path: path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        }
        main(["--repo-root", str(repo)])
        after = {
            path: path.read_bytes()
            for path in sorted(repo.rglob("*"))
            if path.is_file()
        }
        assert before == after


class TestRendering:
    @pytest.mark.parametrize(
        "bucket,expected",
        [
            ("missing", "MISSING"),
            ("unknown", "UNKNOWN"),
            ("undocumented", "UNDOCUMENTED"),
        ],
    )
    def test_each_bucket_gets_a_labelled_section(self, bucket, expected):
        findings = {"missing": [], "unknown": [], "undocumented": []}
        findings[bucket] = ["SOME_KEY"]
        text = "\n".join(render(findings, {"SOME_KEY": "v"}))
        assert expected in text
        assert "SOME_KEY" in text

    def test_clean_findings_say_none(self):
        text = "\n".join(render({"missing": [], "unknown": [], "undocumented": []}, {}))
        assert text.count("none") == 3
