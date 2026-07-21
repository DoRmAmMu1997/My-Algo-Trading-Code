"""Regression tests for MAT-110 dependency and CI policy.

These checks keep the safety controls reviewable in ordinary pytest runs. They
do not contact package indexes or GitHub; they only validate committed policy.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import yaml
from check_env_config import audit, env_keys_read_by, source_files

ROOT = Path(__file__).resolve().parent.parent


def _requirement_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_optional_dependency_sets_are_exact_and_kotak_uses_official_tag():
    core = _requirement_lines("requirements.txt")
    ai = _requirement_lines("requirements-ai.txt")
    brokers = _requirement_lines("requirements-brokers.txt")

    assert "requests==2.34.2" in core
    assert "python-dotenv==1.2.2" in core
    # The full quality job imports the vendored Shoonya client while measuring
    # broker-adapter coverage, so its import-time WebSocket dependency belongs
    # in the core test/runtime environment as well as the isolated broker set.
    assert "websocket-client==1.8.0" in core
    # dhanhq.marketfeed (the websocket market data producer) hard-imports the
    # async `websockets` library at package import time, so the exact version
    # must be pinned in core rather than left to transitive resolution.
    assert "websockets==16.0" in core
    assert "claude-agent-sdk==0.2.123" in ai
    assert "pydantic==2.13.4" in ai
    assert all("==" in line for line in ai)
    assert "pyotp==2.9.0" in brokers
    assert "websocket-client==1.8.0" in brokers
    assert any(
        line == (
            "neo_api_client @ git+https://github.com/Kotak-Neo/"
            "Kotak-neo-api-v2.git@v2.0.1"
        )
        for line in brokers
    )
    assert all("==" in line or " @ git+" in line for line in brokers)


def test_ci_runs_audit_branch_coverage_and_every_exact_dependency_set():
    workflow = (ROOT / ".github/workflows/quality-and-security.yml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    core_job = workflow.split("\n  broker-dependencies:", maxsplit=1)[0]

    assert set(parsed["jobs"]) == {"verify", "broker-dependencies"}
    assert "requirements-ai.txt" in workflow
    assert "requirements-brokers.txt" in workflow
    assert "broker-dependencies:" in workflow
    assert "requirements-brokers.txt" not in core_job
    assert "python -m pip_audit" in workflow
    assert "python -m coverage run" in workflow
    assert "scripts/check_coverage_thresholds.py" in workflow


def test_dependabot_updates_python_and_github_actions_weekly():
    config = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
    ecosystems = {
        item["package-ecosystem"]: item["schedule"]["interval"]
        for item in config["updates"]
    }

    assert ecosystems == {"pip": "weekly", "github-actions": "weekly"}


def test_coverage_config_is_branch_enabled_and_preserves_overall_baseline():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["coverage"]["run"]["branch"] is True
    assert config["tool"]["coverage"]["report"]["fail_under"] == 54.7


def test_every_env_setting_the_code_reads_is_documented_in_env_example():
    """A new `.env` knob must ship with its `env.example` entry.

    `env.example` is the ONLY discovery surface for configuration -- the real
    `.env` is gitignored, so a key that never reaches the template is invisible
    to the operator and silently runs on whatever in-code default it was born
    with. This gate closes that gap at the point it opens: twelve keys had
    already drifted out of the operator's file before it was added.

    One direction only (code -> template). The reverse would flag the ~200
    per-strategy `<PREFIX>_*` knobs that `_signal_gen_ops` builds from
    f-strings, which are real settings the AST cannot see.
    """
    # Same helpers the `python algo.py check-env` diagnostic uses, so the gate
    # and the operator-facing tool can never disagree about what "documented"
    # means.
    read: set[str] = set()
    for path in source_files(ROOT):
        read |= env_keys_read_by(path)

    # Sanity check: if the AST walk silently stopped matching (a helper was
    # renamed, say), this test would "pass" while checking nothing at all.
    assert len(read) > 300, f"env-key extraction looks broken: found only {len(read)}"

    undocumented = audit(ROOT)["undocumented"]
    assert not undocumented, (
        "these env settings are read by the code but missing from "
        "Dependencies/env.example: " + ", ".join(undocumented)
    )


def test_coverage_threshold_checker_enforces_safety_and_broker_budgets():
    path = ROOT / "scripts/check_coverage_thresholds.py"
    spec = importlib.util.spec_from_file_location("check_coverage_thresholds", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    safety_path = next(iter(module.SAFETY_THRESHOLDS))
    broker_path = next(iter(module.BROKER_THRESHOLDS))
    report = {
        "files": {
            safety_path: {
                "summary": {"percent_covered": 89.99, "num_branches": 2},
            },
            broker_path.replace("/", "\\"): {
                "summary": {"percent_covered": 80.0, "num_branches": 2},
            },
        }
    }

    failures = module.evaluate_coverage(
        report,
        safety_thresholds={safety_path: 90.0},
        broker_thresholds={broker_path: 80.0},
    )

    assert len(failures) == 1
    assert safety_path in failures[0]
