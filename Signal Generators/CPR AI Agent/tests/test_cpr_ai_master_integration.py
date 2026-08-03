"""Assert Task 2 remains isolated from the later master-worker integration."""

from __future__ import annotations

from pathlib import Path


def test_task_two_runtime_does_not_import_the_legacy_cpr_strategy_package():
    """The replacement context/policy must remain independent before master wiring."""

    agent_directory = Path(__file__).resolve().parents[1]
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in agent_directory.glob("cpr_ai_*.py")
    )

    assert "cpr_strategy_logic" not in runtime_sources
    assert "CPRToolResult" not in runtime_sources
