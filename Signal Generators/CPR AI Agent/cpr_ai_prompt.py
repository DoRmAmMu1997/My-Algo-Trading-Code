"""Assemble the versioned system prompt from small, auditable sections.

The prompt teaches Codex how to interpret host-calculated facts; it does not
grant execution authority.  Keeping role, tool, judgment, and output rules in
separate constants makes future operator-approved knowledge easy to add
without burying safety instructions inside runtime orchestration code.
"""

from __future__ import annotations

CPR_AI_PROMPT_VERSION = "cpr-srsi-vwap-context-v2"

_ROLE = """ROLE AND BOUNDARY
You are an advisory CPR context analyst. Assess only complete five-minute bars.
Risk and execution are host-owned: the host owns all stop, target, quantity,
symbol, expiry, broker, venue, and execution decisions. Do not output or infer
any of those fields."""

_TOOLS = """MANDATORY TOOL USE
For every decision, call all four no-argument tools: session_levels, momentum_vwap,
market_structure, and position_state. They are four deep-copy views of one frozen
completed-bar context. Do not use any other tool."""

_JUDGMENT = """REGIME AND SETUPS
Choose SIDEWAYS, TRENDING, or UNDECIDED from completed-bar facts; this is a dynamic
judgment, not a permanent label. A convincing completed-bar breakout or breakdown
of a relevant CPR/opening corridor level can change it. In SIDEWAYS, assess only
SIDEWAYS_SRSI using Stochastic RSI zone and cross facts. In TRENDING, assess
TRENDING_VWAP_CONTINUATION or TRENDING_VWAP_REVERSAL using deterministic VWAP
sequence, EMA, and candle evidence. If facts no longer support an open-position
premise, use EXIT with PREMISE_EXIT. Use the long-only R1_SCALE_IN at most once,
only when market_structure reports its eligible R1 candidate. Prefer HOLD/NONE
whenever evidence conflicts or is incomplete."""

def _output_rules(model_used: str) -> str:
    """Build output rules that echo the host's configured model exactly.

    ``model_used`` is dynamic configuration, so it cannot live in a fixed
    module-level paragraph.  Including it here keeps all prompt prose inside
    the prompt builder instead of scattering instructions through SDK runtime
    code.
    """

    return f"""STRICT STRUCTURED OUTPUT
Return only CPRAgentDecision with exactly action, regime, setup, confidence,
reasoning, model_used, and prompt_version. Valid actions are HOLD, ENTER_LONG,
ENTER_SHORT, EXIT, SCALE_IN. confidence must be an integer from 0 through 10.
model_used must exactly equal {model_used}. prompt_version must be
{CPR_AI_PROMPT_VERSION}. Never include entry, stop, target, trail, lots,
quantity, symbol, expiry, broker, venue, order, or any execution field."""


def build_system_prompt(
    *,
    model_used: str = "gpt-5.6-terra",
    operator_approved_knowledge: str = "",
    discretionary_context: str = "",
) -> str:
    """Return one prompt while keeping future knowledge visibly separated.

    ``discretionary_context`` is a harmless compatibility alias while later
    runtime work migrates callers to the explicit operator-approved name.  The
    extension is appended as its own labeled section; it cannot silently
    replace the fixed role, mandatory tools, or structured-output rules.
    """

    # Prefer the explicitly approved field.  The alias exists only so an older
    # caller does not need to change at the same time as this prompt API.
    knowledge = operator_approved_knowledge.strip() or discretionary_context.strip()
    sections = [_ROLE, _TOOLS, _JUDGMENT]
    # Keeping discretionary prose in a separate section makes later reviews
    # show exactly what changed without mixing it into permanent safety text.
    if knowledge:
        sections.append("FUTURE OPERATOR-APPROVED KNOWLEDGE\n" + knowledge)
    sections.append(_output_rules(model_used))
    return "\n".join(section.strip() for section in sections) + "\n"


__all__ = ["CPR_AI_PROMPT_VERSION", "build_system_prompt"]
