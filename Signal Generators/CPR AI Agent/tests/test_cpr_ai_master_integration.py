"""Guard the package boundary between the new agent and legacy CPR strategies.

This intentionally simple source-level regression scans only CPR AI runtime
modules. It catches accidental reintroduction of the old Algo 1/2/3 arbiter or
its ``CPRToolResult`` contract before master-worker behavior is considered.
"""

from __future__ import annotations

from pathlib import Path


def test_task_two_runtime_does_not_import_the_legacy_cpr_strategy_package():
    """Runtime modules must stay independent before the master wires execution."""

    agent_directory = Path(__file__).resolve().parents[1]
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in agent_directory.glob("cpr_ai_*.py")
    )

    assert "cpr_strategy_logic" not in runtime_sources
    assert "CPRToolResult" not in runtime_sources
