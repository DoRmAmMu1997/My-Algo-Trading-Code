"""Define the narrow data contract between Codex and the CPR host worker.

The model receives read-only market facts and returns only an advisory action,
regime, setup, confidence score, and explanation.  It cannot choose an option
contract, quantity, entry price, stop, target, broker, venue, or order field;
the deterministic Python host calculates and validates those values later.

Pydantic is the trust boundary here.  Strict types, forbidden extra fields,
and cross-field validators turn an imaginative or malformed model response
into a safe validation failure instead of letting it reach trading code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CPRAction = Literal["HOLD", "ENTER_LONG", "ENTER_SHORT", "EXIT", "SCALE_IN"]
CPRRegime = Literal["SIDEWAYS", "TRENDING", "UNDECIDED"]
CPRSetup = Literal[
    "NONE",
    "SIDEWAYS_SRSI",
    "TRENDING_VWAP_CONTINUATION",
    "TRENDING_VWAP_REVERSAL",
    "PREMISE_EXIT",
    "R1_SCALE_IN",
]


class CPRAgentDecision(BaseModel):
    """Represent one advisory judgment about the current completed bar.

    ``extra='forbid'`` is important here.  It makes all execution-like fields
    invalid even if a model tries to include a plausible looking ``lots`` or
    ``stop`` value in an otherwise valid response.  The action, regime, and
    setup must also describe one coherent idea; a separately valid value in
    each field is not enough if their combination is unsafe or contradictory.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    action: CPRAction
    regime: CPRRegime
    setup: CPRSetup
    # Field bounds become JSON-Schema minimum/maximum values.  Codex therefore
    # sees the same 0-10 contract that Pydantic enforces after the turn returns.
    confidence: int = Field(ge=0, le=10)
    reasoning: str
    model_used: str
    prompt_version: str

    @field_validator("confidence")
    @classmethod
    def _confidence_is_a_simple_score(cls, value: int) -> int:
        """Keep confidence as an operator-readable score, never a sizing input.

        The host records this value for audit and future analysis.  It does not
        use model confidence to increase lots, widen a stop, or bypass a gate.
        """

        if not 0 <= value <= 10:
            raise ValueError("confidence must be between 0 and 10 inclusive.")
        return value

    @model_validator(mode="after")
    def _action_matches_the_declared_setup(self) -> CPRAgentDecision:
        """Reject combinations whose individual fields disagree with each other.

        Checking the relationship in one place keeps callers from having to
        remember a matrix of valid combinations.  Any mismatch causes schema
        validation to fail before host-policy or execution code can run.
        """

        # HOLD intentionally carries no setup.  This prevents the audit log
        # from looking as though a trade setup was accepted but not executed.
        if self.action == "HOLD" and self.setup != "NONE":
            raise ValueError("HOLD must use the NONE setup.")
        # Entry actions may name only the three entry setups documented in the
        # strategy.  Premise exits and scale-ins have separate host checks.
        if self.action in {"ENTER_LONG", "ENTER_SHORT"} and self.setup not in {
            "SIDEWAYS_SRSI",
            "TRENDING_VWAP_CONTINUATION",
            "TRENDING_VWAP_REVERSAL",
        }:
            raise ValueError("Entries require a SRSI or VWAP setup.")
        if self.action == "EXIT" and self.setup != "PREMISE_EXIT":
            raise ValueError("EXIT must use the PREMISE_EXIT setup.")
        if self.action == "SCALE_IN" and self.setup != "R1_SCALE_IN":
            raise ValueError("SCALE_IN must use the R1_SCALE_IN setup.")
        if self.setup == "SIDEWAYS_SRSI" and self.regime != "SIDEWAYS":
            raise ValueError("SIDEWAYS_SRSI requires the SIDEWAYS regime.")
        if (
            self.setup
            in {
                "TRENDING_VWAP_CONTINUATION",
                "TRENDING_VWAP_REVERSAL",
                "R1_SCALE_IN",
            }
            and self.regime != "TRENDING"
        ):
            raise ValueError("VWAP and R1 setups require the TRENDING regime.")
        if self.regime == "UNDECIDED" and self.action not in {"HOLD", "EXIT"}:
            raise ValueError("UNDECIDED may only HOLD or exit an existing premise.")
        return self


class CPRPositionState(BaseModel):
    """Allowlist the market facts Codex may know about an existing position.

    This deliberately has no broker, venue, credential, order, quantity, or
    execution field.  ``extra='forbid'`` makes an accidental host hand-off of
    any such data fail before the frozen Model Context Protocol (MCP) snapshot
    is created or served.  The model can reason about premise and protection,
    but it never receives the information needed to submit an order itself.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    is_flat: bool | None = None
    direction: Literal["LONG", "SHORT"] | None = None
    entry_price: float | None = None
    entry_timestamp: str | None = None
    unrealized_pnl: float | None = None
    bars_held: int | None = None
    original_entry_price: float | None = None
    original_risk_points: float | None = None
    original_protective_stop: float | None = None
    current_protective_stop: float | None = None
    trailing_stage: Literal["NONE", "BREAKEVEN", "R1_LOCKED", "TRAILING"] | None = None
    milestone_price: float | None = None
    final_target_price: float | None = None
    premise: CPRSetup | None = None
    setup: CPRSetup | None = None
    scale_in_eligible: bool | None = None
    scale_in_count: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> CPRPositionState:
        """Validate a host mapping while treating only ``None`` as empty state.

        A falsey list, string, number, or boolean is still malformed input.  It
        must not be silently coerced into an empty payload at a trust boundary.
        This distinction catches upstream programming mistakes that might
        otherwise make an open position appear flat to the model.
        """

        if payload is None:
            return cls()
        # Do not use ``if not payload`` here: values such as ``[]`` and ``0``
        # are falsey, but they are not valid position-state containers.
        if not isinstance(payload, Mapping):
            raise TypeError("CPR position state payload must be a mapping or None.")
        return cls.model_validate(dict(payload))


def validate_position_state(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a plain dictionary containing only validated, allowlisted facts.

    The JSON snapshot layer consumes ordinary dictionaries, so this helper
    converts the strict Pydantic model back to a minimal serializable payload.
    """

    # ``exclude_unset`` preserves the host's concise payload while still
    # validating a supplied ``None`` such as ``entry_price=None``.
    return CPRPositionState.from_payload(payload).model_dump(exclude_unset=True)


__all__ = [
    "CPRAction",
    "CPRAgentDecision",
    "CPRPositionState",
    "CPRRegime",
    "CPRSetup",
    "validate_position_state",
]
