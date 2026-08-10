"""Order-free smoke runner for the isolated CPR Codex host boundary.

Fake mode proves the full host-policy path without an SDK.  The optional
authenticated mode uses the normal isolated SDK/MCP boundary, but both modes
remain structurally unable to create an order because this module imports no
broker or execution surface.  This is an operational diagnostic, not a market
simulation: it uses a deliberately minimal context whose only valid outcome is
HOLD and prints ``NO ORDER`` so an operator cannot mistake success for a trade.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from cpr_ai_agent import CPRAgent, CPRAgentRunResult, CPRToolCallRecord
from cpr_ai_prompt import CPR_AI_PROMPT_VERSION
from cpr_ai_schema import CPRAgentDecision
from cpr_ai_tools import EXPECTED_TOOL_NAMES


def _context() -> dict[str, dict[str, object]]:
    """Return enough synthetic facts to exercise validation but never entry."""

    return {
        "session_levels": {"current_close": 100.0, "levels": {"r1": 110.0, "r2": 120.0, "s1": 90.0, "s2": 80.0}},
        "momentum_vwap": {},
        "market_structure": {},
        "position_state": {"is_flat": True},
    }


def _fake_runner(**_kwargs: object) -> CPRAgentRunResult:
    """Imitate a successful Codex turn without authentication or model usage.

    The fake still reports all four required tool calls, so the same host-side
    evidence validation runs in fake and authenticated smoke modes.
    """

    decision = CPRAgentDecision(
        action="HOLD",
        regime="UNDECIDED",
        setup="NONE",
        confidence=0,
        reasoning="Synthetic order-free smoke.",
        model_used="gpt-5.6-terra",
        prompt_version=CPR_AI_PROMPT_VERSION,
    )
    return CPRAgentRunResult(
        final_response=decision.model_dump_json(),
        tool_calls=tuple(CPRToolCallRecord(tool=name, status="completed") for name in EXPECTED_TOOL_NAMES),
        token_usage={"total_tokens": 0},
    )


def main(argv: list[str] | None = None, *, authenticated_runner: Callable[..., CPRAgentRunResult] | None = None) -> int:
    """Run one synthetic decision through the real host-policy boundary.

    ``--synthetic`` is a mandatory operator acknowledgement.  ``--fake`` stays
    fully local, while ``--authenticated`` may verify subscription login and
    MCP calls; neither path imports or receives an order function.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="required guard for local smoke use")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--authenticated", action="store_true")
    args = parser.parse_args(argv)
    # Refuse an invocation that does not explicitly acknowledge synthetic,
    # order-free operation; there is no implicit "normal trading" mode here.
    if not args.synthetic:
        parser.error("Only --synthetic order-free smoke modes are available.")
    # Dependency injection makes authenticated smoke testable without letting
    # this command know anything about broker or worker construction.
    agent = CPRAgent(runner=_fake_runner if args.fake else authenticated_runner)
    outcome = agent.decide(_context(), bar_signature="synthetic-bar")
    # The marker is intentionally unconditional so copied console output always
    # communicates that validation success did not submit an order.
    print(f"{outcome.action} validation={outcome.validation_code} NO ORDER")
    return 0 if args.fake or outcome.validation_code == "accepted_hold" else 1


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
