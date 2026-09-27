"""Convert an advisory Codex proposal into fail-closed host permission.

Codex may classify a completed five-minute bar and explain whether a documented
setup is present.  It never receives an order surface and never decides entry,
stop, target, quantity, contract, broker, or venue.  This module is the trust
boundary: it first proves the model read the exact frozen evidence, then parses
the strict schema, verifies model/prompt identity and freshness, and finally
recalculates every executable field from deterministic facts.

Tool evidence that remains incomplete after one same-snapshot retry, a stale
bar, malformed response, optional SDK problem, or contradictory market fact
becomes ``HOLD``.  Open-position mechanical safety and order execution remain
outside this module in the master worker.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any

from cpr_ai_tools import EXPECTED_TOOL_NAMES

# Missing/failed reads are safe to retry because all four tools are read-only
# views of the exact same immutable snapshot.  Capability violations are never
# retried: they indicate that the turn left the allowlisted contract.
_RETRIABLE_TOOL_EVIDENCE_CODES = frozenset({"missing_tool_call", "failed_tool_call"})


class CPRTurnRequestKind(StrEnum):
    """Name the only two fixed turn requests allowed across the child boundary.

    The parent controls this enum.  In particular, no model response or caller
    string can become a follow-up instruction, which keeps the corrective turn
    limited to re-reading the four already-frozen facts.
    """

    NORMAL = "normal"
    TOOL_REPAIR = "tool_repair"


@dataclass(frozen=True)
class CPRToolCallRecord:
    """One SDK-observed MCP read, retained only to prove tool coverage.

    The record does not carry or authorize an order.  The host checks that all
    four allowlisted tools completed exactly once before trusting model text.
    """

    tool: str
    status: str
    error: str | None = None


@dataclass(frozen=True)
class CPRAgentRunResult:
    """Minimal SDK-neutral result returned by the isolated child process.

    Keeping this type free of SDK classes prevents optional dependencies from
    leaking into the master runtime and makes malformed evidence easy to test.
    """

    final_response: str
    tool_calls: tuple[CPRToolCallRecord, ...]
    token_usage: dict[str, int] = field(default_factory=dict)
    unexpected_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CPRAttemptEvidence:
    """Auditable record for one isolated turn slot within a bar deadline.

    Most records describe a child turn that returned. A record can instead be
    a conservative pre-launch/timeout marker when the shared deadline expires
    before usable child evidence comes back. Empty tool records and token usage
    then mean "nothing was returned and proved," not "the child proved it used
    zero tokens." The audit never keeps chain-of-thought, returned MCP payloads,
    authentication data, or any execution capability.
    """

    attempt_number: int
    request_kind: CPRTurnRequestKind
    evidence_code: str | None
    tool_records: tuple[CPRToolCallRecord, ...]
    token_usage: dict[str, int]


@dataclass
class CPRAgentOutcome:
    """Host-owned outcome separating advice from executable geometry.

    ``proposal`` preserves what Codex asked for; ``accepted`` and the validation
    fields record what the host allowed; price/risk fields exist only when the
    deterministic policy derived them.  Confidence is intentionally absent
    because model confidence never changes position size in this groundwork.
    """

    action: str = "HOLD"
    proposal: Any | None = None
    accepted: bool = False
    accepted_regime: str | None = None
    validation_code: str = "not_run"
    validation_reason: str = "No decision was evaluated."
    entry_price: float | None = None
    stop_price: float | None = None
    risk_points: float | None = None
    latency_ms: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    tool_evidence: tuple[CPRToolCallRecord, ...] = ()
    inference_attempts: int = 1
    attempt_evidence: tuple[CPRAttemptEvidence, ...] = ()
    # This is the single current-signature sample captured when the host
    # finalizes either normal validation or a terminal fail-closed outcome.
    # Keeping it on the outcome prevents a later audit writer from accidentally
    # observing a newer shared-market generation.
    validation_current_signature: str | None = None


def _hold(code: str, reason: str, proposal: Any | None = None, *, regime: str | None = None) -> CPRAgentOutcome:
    """Create the universal non-executing fallback with an audit reason.

    Centralizing HOLD construction ensures error paths cannot accidentally
    retain stale entry geometry or an executable action.
    """

    return CPRAgentOutcome(
        proposal=proposal,
        accepted_regime=regime,
        validation_code=code,
        validation_reason=reason,
    )


class CPRHostPolicy:
    """Derive permission and geometry from one immutable completed-bar snapshot.

    The Trend-Day Rider's entry authority is the frozen
    ``market_structure.trend_day_candidate`` verdict computed by
    ``cpr_ai_trend_day`` -- the same code the backtest replays. Codex may only
    accept that candidate (in its direction) or veto it; the policy owns every
    Boolean gate and every price, so fabricated supporting prose changes
    nothing.
    """

    def validate(self, context: Mapping[str, Any], proposal: Any) -> CPRAgentOutcome:
        """Accept only a proposal whose deterministic evidence proves its safety.

        The model's action is merely a request. The flat/open action matrix is
        checked first: a flat worker may enter, while an open worker may only
        hold or exit. Missing, malformed, or contradictory data returns HOLD so
        a transient issue cannot increase exposure by accident.
        """

        try:
            position = self._mapping(context, "position_state")
            is_flat = position.get("is_flat")
            if not isinstance(is_flat, bool):
                return _hold("invalid_position_state", "Frozen position state must declare is_flat.", proposal)
            if proposal.action == "HOLD":
                return CPRAgentOutcome(
                    action="HOLD",
                    proposal=proposal,
                    accepted=True,
                    accepted_regime=proposal.regime,
                    validation_code="accepted_hold",
                    validation_reason="A valid HOLD remains non-executing but may persist its regime.",
                )
            if is_flat and proposal.action not in {"ENTER_LONG", "ENTER_SHORT"}:
                return _hold("flat_action_rejected", "A flat position may only hold or enter.", proposal)
            if not is_flat and proposal.action != "EXIT":
                return _hold("open_action_rejected", "An open position may only hold or exit.", proposal)
            if proposal.action == "EXIT":
                if proposal.setup != "PREMISE_EXIT":
                    return _hold("exit_setup_rejected", "EXIT requires PREMISE_EXIT.", proposal)
                return CPRAgentOutcome(
                    action="EXIT",
                    proposal=proposal,
                    accepted=True,
                    accepted_regime=proposal.regime,
                    validation_code="accepted_exit",
                    validation_reason="Open-position premise exit is host permitted.",
                )
            return self._entry(context, proposal, position)
        except (KeyError, TypeError, ValueError) as error:
            return _hold("invalid_frozen_context", f"Frozen context is incomplete: {error}", proposal)

    @staticmethod
    def _mapping(context: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        """Get a required mapping without silently accepting a wrong shape."""

        value = context[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        return value

    def _entry(self, context: Mapping[str, Any], proposal: Any, position: Mapping[str, Any]) -> CPRAgentOutcome:
        """Accept a flat entry only when it restates the host's own candidate.

        The candidate must be eligible, point the same way as the proposal,
        and describe the same completed close the snapshot froze. Entry is that
        close and the stop is the candidate's VWAP, so a long's stop is below
        entry and a short's above. At most one entry is allowed per session.
        """

        if proposal.setup != "TREND_DAY_CONTINUATION":
            return _hold("entry_setup_rejected", "Entries need the TREND_DAY_CONTINUATION setup.", proposal)
        if proposal.regime != "TRENDING":
            return _hold("trending_regime_rejected", "Trend-day entries require the TRENDING regime.", proposal)
        entries_today = position.get("entries_today")
        if type(entries_today) is not int or entries_today != 0:
            # ``type(...) is int`` also refuses a bool, which is an int subclass.
            return _hold("session_entry_used", "The Trend-Day Rider allows one entry per session.", proposal)
        candidate = self._mapping(self._mapping(context, "market_structure"), "trend_day_candidate")
        if candidate.get("eligible") is not True:
            return _hold("no_trend_day_candidate", "The host found no trend-day candidate on this bar.", proposal)
        direction = "LONG" if proposal.action == "ENTER_LONG" else "SHORT"
        if candidate.get("direction") != direction:
            return _hold(
                "candidate_direction_mismatch",
                "The proposal points against the host's trend-day candidate.",
                proposal,
            )
        entry = self._finite_number(candidate.get("entry"))
        stop = self._finite_number(candidate.get("stop"))
        current_close = self._finite_number(self._mapping(context, "session_levels").get("current_close"))
        if entry is None or stop is None or current_close is None:
            return _hold("missing_candidate_geometry", "The candidate entry, stop, or close is unavailable.", proposal)
        if entry != current_close:
            return _hold(
                "candidate_close_mismatch",
                "The candidate does not describe the frozen completed close.",
                proposal,
            )
        risk = entry - stop if direction == "LONG" else stop - entry
        if risk <= 0:
            return _hold("invalid_stop_geometry", "Protective stop is on the wrong side of entry.", proposal)
        return CPRAgentOutcome(
            action=proposal.action,
            proposal=proposal,
            accepted=True,
            accepted_regime=proposal.regime,
            validation_code="accepted_entry",
            validation_reason="The proposal matches the host's eligible trend-day candidate.",
            entry_price=entry,
            stop_price=stop,
            risk_points=risk,
        )

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        """Return a finite float, refusing bools, strings, NaN, and infinity."""

        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return float(value)


class CPRAgent:
    """Coordinate at most one isolated Codex inference per completed bar.

    One lock protects the completed-bar cadence set; a second protects the
    longer-lived inference boundary.  Their separation matters after a timeout:
    Python cannot kill the SDK thread safely, so later bars must HOLD until the
    old thread actually finishes, and its late result must never be consumed.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., CPRAgentRunResult] | None = None,
        model: str = "gpt-5.6-terra",
        reasoning_effort: str = "medium",
        prompt_version: str | None = None,
        timeout_seconds: float = 90.0,
        policy: CPRHostPolicy | None = None,
    ) -> None:
        """Configure the optional agent without importing its SDK eagerly.

        Lazy loading means environments that install only core trading
        dependencies can still run every deterministic strategy.  A non-finite
        or non-positive deadline is rejected now rather than becoming an
        effectively unbounded live worker call.
        """

        try:
            validated_timeout = float(timeout_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError("timeout_seconds must be a positive finite number.") from error
        if not math.isfinite(validated_timeout) or validated_timeout <= 0.0:
            raise ValueError("timeout_seconds must be a positive finite number.")
        self.runner = runner or self._default_runner
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt_version = prompt_version
        self.timeout_seconds = validated_timeout
        self.policy = policy or CPRHostPolicy()
        self._lock = __import__("threading").Lock()
        # A timed-out Python thread cannot be killed safely.  This separate
        # gate prevents a later bar from starting another SDK turn until that
        # late thread exits, preserving the one-inference-at-a-time contract.
        self._inference_lock = __import__("threading").Lock()
        self._seen_bars: set[str] = set()

    @staticmethod
    def _default_runner(**kwargs: Any) -> CPRAgentRunResult:
        """Load the optional Codex adapter only when the agent is actually used."""

        from cpr_ai_codex_runner import run_codex_turn

        return run_codex_turn(**kwargs)

    def decide(
        self, context: Mapping[str, Any], *, bar_signature: str, current_signature: Callable[[], str] | None = None
    ) -> CPRAgentOutcome:
        """Run one turn and return only contemporaneous host-owned permission.

        A bar signature is consumed before inference starts, so a failed host
        decision is never started again by the next worker poll.  Inside this
        single decision only, incomplete read-only tool evidence may receive one
        retry against the same frozen snapshot and total deadline.
        ``current_signature`` lets the host suppress a result when fresher
        completed evidence arrived while Codex was thinking.
        """

        with self._lock:
            if bar_signature in self._seen_bars:
                return _hold("duplicate_bar", "This completed bar already received one inference.")
            self._seen_bars.add(bar_signature)
        if not self._inference_lock.acquire(blocking=False):
            return _hold("inference_in_progress", "A prior Codex turn is still finishing after its deadline.")
        started = monotonic()
        executor = ThreadPoolExecutor(max_workers=1)
        release_when_finished = False
        # The parent may reach its wall-clock deadline while the child thread is
        # still unwinding.  This shared, append-only audit lets that fail-closed
        # path retain every completed attempt rather than returning a blank HOLD.
        attempt_evidence: list[CPRAttemptEvidence] = []
        try:
            future = executor.submit(
                self._run_turn_with_tool_retry,
                context,
                bar_signature,
                started + self.timeout_seconds,
                current_signature,
                attempt_evidence,
            )
            try:
                result, recorded_attempts, combined_usage, terminal_code = future.result(
                    timeout=self.timeout_seconds
                )
            except TimeoutError:
                # Python cannot safely kill an already running SDK call.  Do
                # not wait for it during shutdown and never retain its future,
                # so the late result cannot affect this or a later market bar.
                future.cancel()
                release_when_finished = True
                future.add_done_callback(lambda _future: self._inference_lock.release())
                return self._terminal_hold_with_attempt_evidence(
                    "timeout", attempt_evidence, current_signature
                )
        except Exception as error:  # optional SDK failure must disable this agent only
            # Timeout-class errors retain their conservative pre-launch audit.
            # Other first-turn runtime errors keep the established generic
            # runtime outcome and never expose an exception message.
            if self._is_timeout_error(error):
                return self._terminal_hold_with_attempt_evidence(
                    "timeout", attempt_evidence, current_signature
                )
            return self._terminal_hold_with_attempt_evidence(
                "runtime_error", attempt_evidence, current_signature
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if not release_when_finished:
                self._inference_lock.release()
        latency_ms = int((monotonic() - started) * 1000)
        outcome = (
            self._terminal_hold_with_attempt_evidence(
                terminal_code, list(recorded_attempts), current_signature
            )
            if terminal_code is not None
            else self._validate_run(result, context, bar_signature, current_signature)
        )
        outcome.latency_ms = latency_ms
        outcome.token_usage = combined_usage
        outcome.tool_evidence = result.tool_calls
        outcome.inference_attempts = len(recorded_attempts)
        outcome.attempt_evidence = recorded_attempts
        return outcome

    def _run_turn_with_tool_retry(
        self,
        context: Mapping[str, Any],
        bar_signature: str,
        deadline: float,
        current_signature: Callable[[], str] | None,
        attempt_evidence: list[CPRAttemptEvidence],
    ) -> tuple[CPRAgentRunResult, tuple[CPRAttemptEvidence, ...], dict[str, int], str | None]:
        """Retry incomplete frozen-tool evidence once inside one total deadline.

        Both attempts receive the same in-memory context and bar signature.  A
        retry therefore cannot silently move to newer market facts.  The second
        attempt receives only the wall-clock budget left after the first one.
        That shared budget includes parent/child process setup as well as the
        SDK turn itself, so this recovery path never doubles the configured
        CPR inference timeout.
        """

        results: list[CPRAgentRunResult] = []
        for attempt_index in range(2):
            if attempt_index == 0:
                # Preserve the configured value exactly on the normal path.  It
                # is useful operator evidence and keeps existing runner behavior
                # unchanged when no retry is required.
                attempt_timeout = self.timeout_seconds
            else:
                attempt_timeout = deadline - monotonic()
                if attempt_timeout <= 0:
                    # The repair was selected but no wall-clock budget remains.
                    # Record that fact explicitly without inventing a tool call
                    # or token count for a child process that never launched.
                    attempt_evidence.append(
                        CPRAttemptEvidence(
                            attempt_number=2,
                            request_kind=CPRTurnRequestKind.TOOL_REPAIR,
                            evidence_code="timeout",
                            tool_records=(),
                            token_usage={},
                        )
                    )
                    return results[-1], tuple(attempt_evidence), self._combined_token_usage(results), "timeout"
            request_kind = (
                CPRTurnRequestKind.NORMAL
                if attempt_index == 0
                else CPRTurnRequestKind.TOOL_REPAIR
            )
            # Install a conservative no-evidence marker before the optional
            # runtime starts. If the parent deadline wins the race, this still
            # records which turn slot was selected. Empty tool/usage fields mean
            # the child returned no provable evidence; they are not an assertion
            # that a launched child consumed exactly zero tokens.
            diagnostic_index = len(attempt_evidence)
            attempt_evidence.append(
                CPRAttemptEvidence(
                    attempt_number=attempt_index + 1,
                    request_kind=request_kind,
                    evidence_code="timeout",
                    tool_records=(),
                    token_usage={},
                )
            )
            try:
                result = self._run_turn(
                    context,
                    bar_signature,
                    request_kind=request_kind,
                    timeout_seconds=attempt_timeout,
                )
            except Exception as error:
                # Preserve the usual first-turn exception behavior.  Once the
                # normal result proved a repair was needed, however, a failed
                # repair must retain that completed first audit trail.
                if attempt_index == 0:
                    if not self._is_timeout_error(error):
                        # A normal child failed before returning any evidence.
                        # Keep one typed record without inventing tools, usage,
                        # or an exception string for the operator audit.
                        attempt_evidence[diagnostic_index] = CPRAttemptEvidence(
                            attempt_number=1,
                            request_kind=CPRTurnRequestKind.NORMAL,
                            evidence_code="runtime_error",
                            tool_records=(),
                            token_usage={},
                        )
                    raise
                terminal_code = "timeout" if self._is_timeout_error(error) else "runtime_error"
                attempt_evidence[diagnostic_index] = (
                    CPRAttemptEvidence(
                        attempt_number=2,
                        request_kind=CPRTurnRequestKind.TOOL_REPAIR,
                        evidence_code=terminal_code,
                        tool_records=(),
                        token_usage={},
                    )
                )
                return results[-1], tuple(attempt_evidence), self._combined_token_usage(results), terminal_code
            results.append(result)
            evidence_error = self._tool_evidence_error(result)
            completed_evidence = CPRAttemptEvidence(
                attempt_number=attempt_index + 1,
                request_kind=request_kind,
                evidence_code=None if evidence_error is None else evidence_error[0],
                tool_records=result.tool_calls,
                token_usage=dict(result.token_usage),
            )
            attempt_evidence[diagnostic_index] = completed_evidence
            if not self._retry_is_allowed(
                result,
                context,
                bar_signature,
                current_signature,
                evidence_error,
            ):
                break

        # The first call either produced a result or raised to ``decide()``, so
        # this list cannot be empty.  Keeping the assertion documents that local
        # invariant without converting an SDK exception into trusted evidence.
        assert results
        return results[-1], tuple(attempt_evidence), self._combined_token_usage(results), None

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        """Recognize local deadline exceptions without retaining their details.

        The optional adapter may surface either the standard library's
        ``TimeoutExpired`` or the executor's ``TimeoutError``.  The host stores
        only the safe classification, never a potentially sensitive message.
        """

        return isinstance(error, TimeoutError) or type(error).__name__ == "TimeoutExpired"

    @staticmethod
    def _terminal_failure_reason(code: str) -> str:
        """Return a credential-safe reason for a terminal Codex-turn outcome.

        Both the initial turn and an optional repair use this helper. The
        historical human-readable text uses the word ``corrective`` for either
        path, so ``attempt_evidence.request_kind`` is the authoritative field
        when an operator needs to distinguish ``normal`` from ``tool_repair``.
        The reason itself never includes exception text, child output, command
        arguments, or local paths.
        """

        if code == "timeout":
            return "The corrective Codex turn exhausted the original deadline."
        return "The optional corrective Codex runtime failed."

    def _terminal_hold_with_attempt_evidence(
        self,
        code: str,
        attempt_evidence: list[CPRAttemptEvidence],
        current_signature: Callable[[], str] | None,
    ) -> CPRAgentOutcome:
        """Build a terminal HOLD with its one retained host signature sample.

        These paths never reach ``_validate_run`` because the initial or repair
        child timed out or failed. Capture one signature here rather than
        leaving a JSONL row ambiguous or calling the mutable shared-data getter
        later. The call happens once during finalization, so the audit describes
        this decision even if the market store advances immediately afterward.
        """

        recorded_attempts = tuple(attempt_evidence)
        outcome = _hold(code, self._terminal_failure_reason(code))
        outcome.inference_attempts = len(recorded_attempts)
        outcome.attempt_evidence = recorded_attempts
        outcome.token_usage = self._combined_attempt_token_usage(recorded_attempts)
        outcome.tool_evidence = next(
            (record.tool_records for record in reversed(recorded_attempts) if record.tool_records),
            (),
        )
        return self._with_validation_current_signature(
            outcome,
            current_signature() if current_signature is not None else None,
        )

    @staticmethod
    def _combined_attempt_token_usage(
        attempts: tuple[CPRAttemptEvidence, ...],
    ) -> dict[str, int]:
        """Combine retained attempt counters with the same context-window rule."""

        combined: dict[str, int] = {}
        for attempt in attempts:
            for key, value in attempt.token_usage.items():
                if key == "model_context_window":
                    combined[key] = max(combined.get(key, 0), int(value))
                else:
                    combined[key] = combined.get(key, 0) + int(value)
        return combined

    def _retry_is_allowed(
        self,
        result: CPRAgentRunResult,
        context: Mapping[str, Any],
        bar_signature: str,
        current_signature: Callable[[], str] | None,
        evidence_error: tuple[str, str] | None,
    ) -> bool:
        """Allow the one repair only for otherwise-valid missing/failed reads.

        A repair exists to recover an incomplete observation of immutable MCP
        facts.  It must not hide a stale bar, bad schema/model/prompt echo, or
        deterministic host-policy rejection behind a second model attempt.
        """

        if evidence_error is None or evidence_error[0] not in _RETRIABLE_TOOL_EVIDENCE_CODES:
            return False
        if current_signature is not None and current_signature() != bar_signature:
            return False
        try:
            from cpr_ai_schema import CPRAgentDecision

            proposal = CPRAgentDecision.model_validate_json(result.final_response)
        except Exception:
            return False
        if proposal.model_used != self.model:
            return False
        expected_prompt = self.prompt_version
        if expected_prompt is None:
            from cpr_ai_prompt import CPR_AI_PROMPT_VERSION

            expected_prompt = CPR_AI_PROMPT_VERSION
        if proposal.prompt_version != expected_prompt:
            return False
        return self.policy.validate(context, proposal).accepted

    @staticmethod
    def _combined_token_usage(
        results: list[CPRAgentRunResult],
    ) -> dict[str, int]:
        """Aggregate billed counters while retaining one context-window size.

        A retry is a second model turn and its tokens must remain visible in the
        JSONL audit.  ``model_context_window`` describes capacity rather than
        consumption, so it uses the largest reported value instead of a sum.
        """

        combined: dict[str, int] = {}
        for result in results:
            for key, value in result.token_usage.items():
                if key == "model_context_window":
                    combined[key] = max(combined.get(key, 0), int(value))
                else:
                    combined[key] = combined.get(key, 0) + int(value)
        return combined

    def _run_turn(
        self,
        context: Mapping[str, Any],
        bar_signature: str,
        *,
        request_kind: CPRTurnRequestKind = CPRTurnRequestKind.NORMAL,
        timeout_seconds: float | None = None,
    ) -> CPRAgentRunResult:
        """Build prompt/schema lazily and pass only advisory inputs to the child.

        The bar signature is cadence metadata; the frozen context is the only
        market evidence.  No broker client, order callback, lot count, symbol,
        or mutable position handle crosses this boundary.
        """

        from cpr_ai_prompt import CPR_AI_PROMPT_VERSION, build_system_prompt
        from cpr_ai_schema import CPRAgentDecision

        return self.runner(
            # Tell the prompt builder which configured model identifier must be
            # echoed in the strict response.  This remains advisory metadata;
            # the host independently verifies it before accepting a decision.
            prompt=build_system_prompt(model_used=self.model),
            context=context,
            bar_signature=bar_signature,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=self.prompt_version or CPR_AI_PROMPT_VERSION,
            output_schema=CPRAgentDecision.model_json_schema(),
            request_kind=request_kind,
            # Direct diagnostic callers historically used this helper without
            # an explicit deadline.  Normal and retry paths pass their exact
            # per-attempt budget; the fallback preserves that diagnostic API.
            timeout_seconds=(
                self.timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
        )

    def _validate_run(
        self,
        result: CPRAgentRunResult,
        context: Mapping[str, Any],
        bar_signature: str,
        current_signature: Callable[[], str] | None,
    ) -> CPRAgentOutcome:
        """Validate a child result in strict least-trust order.

        Tool completeness is checked before model text; freshness before schema;
        schema before model/prompt echoes; and all of those before trading
        policy.  A valid new regime may persist even when deterministic entry
        geometry is rejected, because regime memory is advisory, not authority
        to place a trade.
        """

        # Sample mutable market identity exactly once for this validation.  The
        # audit row must describe this host decision, not a later poll that may
        # have received an official-candle correction in the meantime.
        validation_current_signature = (
            current_signature() if current_signature is not None else None
        )
        evidence_error = self._tool_evidence_error(result)
        if evidence_error is not None:
            return self._with_validation_current_signature(
                _hold(*evidence_error), validation_current_signature
            )
        if (
            current_signature is not None
            and validation_current_signature != bar_signature
        ):
            return self._with_validation_current_signature(
                _hold("stale_bar_signature", "The frozen completed bar is no longer current."),
                validation_current_signature,
            )
        try:
            from cpr_ai_schema import CPRAgentDecision

            proposal = CPRAgentDecision.model_validate_json(result.final_response)
        except Exception:
            return self._with_validation_current_signature(
                _hold("malformed_output", "Codex output did not match the strict decision schema."),
                validation_current_signature,
            )
        if proposal.model_used != self.model:
            return self._with_validation_current_signature(
                _hold("model_mismatch", "Model echo does not match the configured model.", proposal),
                validation_current_signature,
            )
        expected_prompt = self.prompt_version
        if expected_prompt is None:
            from cpr_ai_prompt import CPR_AI_PROMPT_VERSION

            expected_prompt = CPR_AI_PROMPT_VERSION
        if proposal.prompt_version != expected_prompt:
            return self._with_validation_current_signature(
                _hold("prompt_version_mismatch", "Prompt-version echo does not match the host prompt.", proposal),
                validation_current_signature,
            )
        outcome = self.policy.validate(context, proposal)
        # The SDK boundary has proved that this was a contemporaneous, pinned
        # regime classification.  Preserve it even when hard execution gates
        # reject the proposed entry or scale-in.
        if outcome.accepted_regime is None and outcome.validation_code not in {
            "invalid_position_state",
            "invalid_frozen_context",
        }:
            outcome.accepted_regime = proposal.regime
        return self._with_validation_current_signature(
            outcome, validation_current_signature
        )

    @staticmethod
    def _with_validation_current_signature(
        outcome: CPRAgentOutcome,
        validation_current_signature: str | None,
    ) -> CPRAgentOutcome:
        """Retain the one validation-time signature on every audited outcome."""

        outcome.validation_current_signature = validation_current_signature
        return outcome

    @staticmethod
    def _tool_evidence_error(result: CPRAgentRunResult) -> tuple[str, str] | None:
        """Require exactly four allowlisted, unique, successful read operations.

        Extra capability use is as unsafe as a missing fact: either means the
        turn did not follow the narrow contract and must be discarded in full.
        """

        if result.unexpected_actions:
            return "unexpected_agent_action", "The SDK reported a disabled capability."
        names = [record.tool for record in result.tool_calls]
        if any(name not in EXPECTED_TOOL_NAMES for name in names):
            return "unexpected_agent_action", "An unallowlisted tool was attempted."
        if len(names) != len(set(names)):
            return "unexpected_agent_action", "A tool was called more than once."
        expected_names: set[str] = set(EXPECTED_TOOL_NAMES)
        missing = expected_names - set(names)
        if missing:
            return "missing_tool_call", "One or more required frozen tools were not called."
        if any(record.status != "completed" for record in result.tool_calls):
            return "failed_tool_call", "One or more required frozen tools failed."
        return None


__all__ = [
    "CPRAgent",
    "CPRAgentOutcome",
    "CPRAgentRunResult",
    "CPRAttemptEvidence",
    "CPRHostPolicy",
    "CPRToolCallRecord",
    "CPRTurnRequestKind",
]
