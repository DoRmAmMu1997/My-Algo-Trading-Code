"""Strict advisory schema for the independent CPR context agent.

The agent judges a frozen market context.  It never provides an order, a
quantity, a price, or risk geometry: those decisions remain with the host.
"""

from __future__ import annotations

from typing import Literal

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


__all__ = ["CPRAction", "CPRAgentDecision", "CPRRegime", "CPRSetup"]
