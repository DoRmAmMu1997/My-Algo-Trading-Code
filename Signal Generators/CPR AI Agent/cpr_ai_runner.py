"""Order-free smoke runner for the isolated CPR Codex host boundary.

Both modes use fakes in this repository: ``--fake`` proves the full host-policy
path without an SDK, while ``--authenticated-fake`` exercises the same strict
tool evidence shape a future authenticated subprocess must provide.  Neither
mode imports a broker or exposes an order operation.
"""

from __future__ import annotations

import argparse

from cpr_ai_agent import CPRAgent, CPRAgentRunResult, CPRToolCallRecord
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_schema import CPRAgentDecision
from cpr_ai_tools import EXPECTED_TOOL_NAMES


def _context() -> dict[str, dict[str, object]]:
    """Return a minimal HOLD-only frozen context for a no-order smoke run."""

    return {
        "session_levels": {"current_close": 100.0, "levels": {"r1": 110.0, "r2": 120.0, "s1": 90.0, "s2": 80.0}},
        "momentum_vwap": {}, "market_structure": {}, "position_state": {"is_flat": True},
    }


def _fake_runner(**_kwargs: object) -> CPRAgentRunResult:
    """Emit valid four-tool evidence and a safe HOLD without contacting Codex."""

    decision = CPRAgentDecision(
        action="HOLD", regime="UNDECIDED", setup="NONE", confidence=0,
        reasoning="Synthetic order-free smoke.", model_used="gpt-5.6-terra", prompt_version=CPR_AI_PROMPT_VERSION,
    )
    return CPRAgentRunResult(
        final_response=decision.model_dump_json(),
        tool_calls=tuple(CPRToolCallRecord(tool=name, status="completed") for name in EXPECTED_TOOL_NAMES),
        token_usage={"total_tokens": 0},
    )


def main(argv: list[str] | None = None) -> int:
    """Run a synthetic host decision and state explicitly that no order exists."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="required guard for local smoke use")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--authenticated-fake", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic:
        parser.error("Only --synthetic order-free smoke modes are available.")
    outcome = CPRAgent(runner=_fake_runner).decide(_context(), bar_signature="synthetic-bar")
    print(f"{outcome.action} validation={outcome.validation_code} NO ORDER")
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
