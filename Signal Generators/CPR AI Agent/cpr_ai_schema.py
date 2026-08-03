"""Strict advisory schema for the independent CPR context agent.

The agent judges a frozen market context.  It never provides an order, a
quantity, a price, or risk geometry: those decisions remain with the host.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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
    """One narrow, structured judgment about the current completed bar.

    ``extra='forbid'`` is important here.  It makes all execution-like fields
    invalid even if a model tries to include a plausible looking ``lots`` or
    ``stop`` value in an otherwise valid response.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    action: CPRAction
    regime: CPRRegime
    setup: CPRSetup
    confidence: int
    reasoning: str
    model_used: str
    prompt_version: str

    @field_validator("confidence")
    @classmethod
    def _confidence_is_a_simple_score(cls, value: int) -> int:
        """Keep the explanation score on an operator-readable zero-to-ten scale."""

        if not 0 <= value <= 10:
            raise ValueError("confidence must be between 0 and 10 inclusive.")
        return value

    @model_validator(mode="after")
    def _action_matches_the_declared_setup(self) -> CPRAgentDecision:
        """Reject action/setup/regime combinations that would be ambiguous."""

        if self.action == "HOLD" and self.setup != "NONE":
            raise ValueError("HOLD must use the NONE setup.")
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
    """Allowlisted host facts about an existing market position.

    This deliberately has no broker, venue, credential, order, quantity, or
    execution field.  ``extra='forbid'`` makes an accidental host hand-off of
    any such data fail before the frozen MCP snapshot is created or served.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    is_flat: bool | None = None
    direction: Literal["LONG", "SHORT"] | None = None
    entry_price: float | None = None
    entry_timestamp: str | None = None
    unrealized_pnl: float | None = None
    bars_held: int | None = None
    premise: str | None = None
    scale_in_count: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> CPRPositionState:
        """Build position state from a mapping, accepting only ``None`` as empty.

        A falsey list, string, number, or boolean is still malformed input.  It
        must not be silently coerced into an empty payload at a trust boundary.
        """

        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise TypeError("CPR position state payload must be a mapping or None.")
        return cls.model_validate(dict(payload))


def validate_position_state(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and copy only explicit host-provided market/position facts."""

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
