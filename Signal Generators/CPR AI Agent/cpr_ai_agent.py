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
    """Auditable result of one isolated model attempt within a bar deadline.

    This records only request mode, tool coverage, and numeric usage.  It does
    not keep chain-of-thought, returned MCP payloads, authentication data, or
    any execution capability.
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
    milestone_price: float | None = None
    final_target_price: float | None = None
    risk_points: float | None = None
    scale_in_permitted: bool = False
    latency_ms: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    tool_evidence: tuple[CPRToolCallRecord, ...] = ()
    inference_attempts: int = 1
    attempt_evidence: tuple[CPRAttemptEvidence, ...] = ()


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

    The model selects among documented setup names, but this policy owns every
    Boolean hard gate and every price.  It therefore remains authoritative even
    if the model fabricates supporting prose.
    """

    max_risk_points = 30.0
    milestone_buffer = 2.0

    def validate(self, context: Mapping[str, Any], proposal: Any) -> CPRAgentOutcome:
        """Accept only a proposal whose deterministic evidence proves its safety.

        The model's action is merely a request.  The flat/open action matrix is
        checked first: a flat worker may enter, while an open worker may only
        exit or request the single documented add.  Missing, malformed, or
        contradictory data returns HOLD so a transient issue cannot increase
        exposure by accident.
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
            if not is_flat and proposal.action not in {"EXIT", "SCALE_IN"}:
                return _hold("open_action_rejected", "An open position may only hold, exit, or scale in.", proposal)
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
            if proposal.action == "SCALE_IN":
                return self._scale_in(context, proposal, position)
            return self._entry(context, proposal)
        except (KeyError, TypeError, ValueError) as error:
            return _hold("invalid_frozen_context", f"Frozen context is incomplete: {error}", proposal)

    @staticmethod
    def _mapping(context: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        """Get a required mapping without silently accepting a wrong shape."""

        value = context[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        return value

    def _entry(self, context: Mapping[str, Any], proposal: Any) -> CPRAgentOutcome:
        """Route a flat entry request to its deterministic setup validator.

        Unknown setup names are rejected rather than treated as discretionary
        variants.  This is the seam where future approved setups can be added.
        """

        direction = "LONG" if proposal.action == "ENTER_LONG" else "SHORT"
        if proposal.setup == "SIDEWAYS_SRSI":
            return self._sideways_entry(context, proposal, direction)
        if proposal.setup in {"TRENDING_VWAP_CONTINUATION", "TRENDING_VWAP_REVERSAL"}:
            return self._trending_entry(context, proposal, direction)
        return _hold("entry_setup_rejected", "Entries need a documented SRSI or VWAP setup.", proposal)

    def _sideways_entry(self, context: Mapping[str, Any], proposal: Any, direction: str) -> CPRAgentOutcome:
        """Validate a sideways SRSI entry and select its confirmed-swing stop.

        A long needs a bullish cross wholly in the oversold zone; a short needs
        the bearish overbought mirror.  The stop comes from the latest already
        confirmed swing, never from a model-supplied price or forming extreme.
        """

        if proposal.regime != "SIDEWAYS":
            return _hold("sideways_regime_rejected", "SRSI entries require the SIDEWAYS regime.", proposal)
        momentum = self._mapping(context, "momentum_vwap")
        srsi = self._mapping(momentum, "stochastic_rsi")
        expected = "cross_up_in_oversold" if direction == "LONG" else "cross_down_in_overbought"
        if srsi.get(expected) is not True:
            return _hold("srsi_cross_rejected", "Frozen SRSI cross and zone do not support this entry.", proposal)
        structure = self._mapping(context, "market_structure")
        swings = self._mapping(structure, "swings")
        swing_name = "lows" if direction == "LONG" else "highs"
        points = swings.get(swing_name)
        if not isinstance(points, list) or not points or not isinstance(points[-1], Mapping):
            return _hold("missing_swing_stop", "Sideways entries need a confirmed latest swing.", proposal)
        return self._geometry(context, proposal, direction, float(points[-1]["price"]))

    def _trending_entry(self, context: Mapping[str, Any], proposal: Any, direction: str) -> CPRAgentOutcome:
        """Apply every documented VWAP, candle-body, RSI, and EMA trend gate.

        Continuation and reversal use different frozen VWAP sequences, but
        both require at least 40 percent of the entry body on the trade side,
        directional RSI, ordered/sloping EMAs, and the completed candle extreme
        as stop.  A continuation also cannot chase a long above R2 or a short
        below S2.  Model reasoning cannot waive any of these conditions.
        """

        if proposal.regime != "TRENDING":
            return _hold("trending_regime_rejected", "VWAP entries require the TRENDING regime.", proposal)
        long = direction == "LONG"
        if proposal.setup == "TRENDING_VWAP_CONTINUATION":
            levels = self._mapping(context, "session_levels")
            level_map = self._mapping(levels, "levels")
            entry = levels.get("current_close")
            boundary_name = "r2" if long else "s2"
            boundary = level_map.get(boundary_name)

            # The comparison is deliberately strict: the new knowledge says
            # "above R2" and "below S2." Existing reward/target geometry still
            # decides whether an entry exactly at or near a level is practical.
            outside_boundary = (
                isinstance(entry, (int, float))
                and isinstance(boundary, (int, float))
                and ((long and entry > boundary) or (not long and entry < boundary))
            )
            if outside_boundary:
                return _hold(
                    "continuation_outside_r2_s2",
                    "Trend continuation cannot enter long above R2 or short below S2.",
                    proposal,
                )
        momentum = self._mapping(context, "momentum_vwap")
        vwap = self._mapping(momentum, "vwap")
        sequence = self._mapping(vwap, "sequence_evidence")
        body = self._mapping(vwap, "entry_candle")
        sequence_key = (
            "all_recent_above"
            if proposal.setup == "TRENDING_VWAP_CONTINUATION" and long
            else "all_recent_below"
            if proposal.setup == "TRENDING_VWAP_CONTINUATION"
            else "reclaimed"
            if long
            else "lost"
        )
        if sequence.get(sequence_key) is not True:
            return _hold("vwap_sequence_rejected", "The documented directional VWAP sequence is absent.", proposal)
        fraction = body.get("body_fraction_above" if long else "body_fraction_below")
        if not isinstance(fraction, (int, float)) or float(fraction) < 0.4:
            return _hold(
                "vwap_body_fraction_rejected",
                "At least 40% of the entry body must be on the VWAP side.",
                proposal,
            )
        rsi = momentum.get("rsi14")
        if not isinstance(rsi, (int, float)) or (long and rsi <= 45) or (not long and rsi >= 65):
            return _hold("rsi_rejected", "RSI does not meet the directional hard gate.", proposal)
        ema = self._mapping(momentum, "ema")
        correct_order = "EMA5_ABOVE_EMA20" if long else "EMA5_BELOW_EMA20"
        if (
            ema.get("order") != correct_order
            or not isinstance(ema.get("ema5_slope"), (int, float))
            or not isinstance(ema.get("ema20_slope"), (int, float))
            or (long and (ema["ema5_slope"] <= 0 or ema["ema20_slope"] <= 0))
            or (not long and (ema["ema5_slope"] >= 0 or ema["ema20_slope"] >= 0))
        ):
            return _hold("ema_rejected", "EMA order and both directional slopes are required.", proposal)
        candle = self._mapping(momentum, "candle")
        stop = candle.get("low" if long else "high")
        if not isinstance(stop, (int, float)):
            return _hold("missing_candle_stop", "Trending entries need the completed candle extreme.", proposal)
        return self._geometry(context, proposal, direction, float(stop))

    def _geometry(self, context: Mapping[str, Any], proposal: Any, direction: str, stop: float) -> CPRAgentOutcome:
        """Calculate all executable entry geometry from frozen price facts.

        Entry is the completed-bar close.  Risk must be positive and at most 30
        NIFTY points.  The next CPR level, adjusted by the two-point buffer,
        must offer at least 1R; the similarly buffered R2/S2 is the definite
        final target.  Codex never supplies or alters any of these values.
        """

        levels = self._mapping(context, "session_levels")
        entry = levels.get("current_close")
        level_map = self._mapping(levels, "levels")
        if not isinstance(entry, (int, float)):
            return _hold("missing_entry", "The completed five-minute close is unavailable.", proposal)
        entry = float(entry)
        risk = entry - stop if direction == "LONG" else stop - entry
        if risk <= 0:
            return _hold("invalid_stop_geometry", "Protective stop is on the wrong side of entry.", proposal)
        if risk > self.max_risk_points:
            return _hold("risk_wider_than_30", "Risk exceeds the 30 NIFTY-point hard limit.", proposal)
        next_levels = self._mapping(levels, "next_levels")
        next_level = next_levels.get("upside" if direction == "LONG" else "downside")
        raw_milestone = next_level.get("price") if isinstance(next_level, Mapping) else None
        final_raw = level_map.get("r2" if direction == "LONG" else "s2")
        if not isinstance(raw_milestone, (int, float)) or not isinstance(final_raw, (int, float)):
            return _hold("missing_cpr_levels", "Required CPR milestone or final target is missing.", proposal)
        milestone = (
            float(raw_milestone) - self.milestone_buffer
            if direction == "LONG"
            else float(raw_milestone) + self.milestone_buffer
        )
        final_target = (
            float(final_raw) - self.milestone_buffer
            if direction == "LONG"
            else float(final_raw) + self.milestone_buffer
        )
        reward = milestone - entry if direction == "LONG" else entry - milestone
        if reward < risk:
            return _hold("sub_one_r_milestone", "The next buffered CPR milestone offers under one R.", proposal)
        if (direction == "LONG" and final_target <= entry) or (direction == "SHORT" and final_target >= entry):
            return _hold("invalid_target_geometry", "The final CPR target is on the wrong side of entry.", proposal)
        return CPRAgentOutcome(
            action=proposal.action,
            proposal=proposal,
            accepted=True,
            accepted_regime=proposal.regime,
            validation_code="accepted_entry",
            validation_reason="All deterministic entry and risk gates passed.",
            entry_price=entry,
            stop_price=stop,
            milestone_price=milestone,
            final_target_price=final_target,
            risk_points=risk,
        )

    def _scale_in(self, context: Mapping[str, Any], proposal: Any, position: Mapping[str, Any]) -> CPRAgentOutcome:
        """Permit only the documented one-time R1 add to a trending long.

        This outcome is permission, not execution.  The master rechecks market
        health and lifecycle state, reuses the locked contract, applies spread
        and liquidity gates, and owns quantity/fill/reconciliation accounting.
        """

        candidate = self._mapping(self._mapping(context, "market_structure"), "r1_scale_in_candidate")
        if (
            proposal.regime != "TRENDING"
            or position.get("direction") != "LONG"
            or position.get("premise") not in {"TRENDING_VWAP_CONTINUATION", "TRENDING_VWAP_REVERSAL"}
            or position.get("scale_in_eligible") is not True
            or position.get("scale_in_count") not in {None, 0}
            or candidate.get("eligible") is not True
            or candidate.get("direction") != "LONG"
        ):
            return _hold(
                "scale_in_rejected",
                "Scale-in requires one open trending long R1 candidate and no prior use.",
                proposal,
            )
        return CPRAgentOutcome(
            action="SCALE_IN",
            proposal=proposal,
            accepted=True,
            accepted_regime=proposal.regime,
            validation_code="accepted_scale_in",
            validation_reason="Host permits only the documented one-time R1 scale-in.",
            scale_in_permitted=True,
        )


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
                result, attempt_evidence, combined_usage, terminal_code = future.result(
                    timeout=self.timeout_seconds
                )
            except TimeoutError:
                # Python cannot safely kill an already running SDK call.  Do
                # not wait for it during shutdown and never retain its future,
                # so the late result cannot affect this or a later market bar.
                future.cancel()
                release_when_finished = True
                future.add_done_callback(lambda _future: self._inference_lock.release())
                return self._hold_with_attempt_evidence("timeout", attempt_evidence)
        except Exception as error:  # optional SDK failure must disable this agent only
            # Timeout-class errors retain their conservative pre-launch audit.
            # Other first-turn runtime errors keep the established generic
            # runtime outcome and never expose an exception message.
            if self._is_timeout_error(error):
                return self._hold_with_attempt_evidence("timeout", attempt_evidence)
            return _hold("runtime_error", f"Optional Codex runtime failed: {type(error).__name__}.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if not release_when_finished:
                self._inference_lock.release()
        latency_ms = int((monotonic() - started) * 1000)
        outcome = (
            _hold(terminal_code, self._terminal_failure_reason(terminal_code))
            if terminal_code is not None
            else self._validate_run(result, context, bar_signature, current_signature)
        )
        outcome.latency_ms = latency_ms
        outcome.token_usage = combined_usage
        outcome.tool_evidence = result.tool_calls
        outcome.inference_attempts = len(attempt_evidence)
        outcome.attempt_evidence = attempt_evidence
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
        attempt receives only the wall-clock budget left after the first one,
        so this recovery path never doubles the configured SDK timeout.
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
            # Install a conservative no-evidence timeout diagnostic before the
            # optional runtime starts.  If the parent deadline wins the race,
            # this proves which turn began while never inventing tool calls or
            # usage that the child did not return.
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
        """Return a safe generic reason for a repair failure audit outcome."""

        if code == "timeout":
            return "The corrective Codex turn exhausted the original deadline."
        return "The optional corrective Codex runtime failed."

    def _hold_with_attempt_evidence(
        self,
        code: str,
        attempt_evidence: list[CPRAttemptEvidence],
    ) -> CPRAgentOutcome:
        """Build a terminal HOLD while retaining only safe completed-attempt data."""

        recorded_attempts = tuple(attempt_evidence)
        outcome = _hold(code, self._terminal_failure_reason(code))
        outcome.inference_attempts = len(recorded_attempts)
        outcome.attempt_evidence = recorded_attempts
        outcome.token_usage = self._combined_attempt_token_usage(recorded_attempts)
        outcome.tool_evidence = next(
            (record.tool_records for record in reversed(recorded_attempts) if record.tool_records),
            (),
        )
        return outcome

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

        evidence_error = self._tool_evidence_error(result)
        if evidence_error is not None:
            return _hold(*evidence_error)
        if current_signature is not None and current_signature() != bar_signature:
            return _hold("stale_bar_signature", "The frozen completed bar is no longer current.")
        try:
            from cpr_ai_schema import CPRAgentDecision

            proposal = CPRAgentDecision.model_validate_json(result.final_response)
        except Exception:
            return _hold("malformed_output", "Codex output did not match the strict decision schema.")
        if proposal.model_used != self.model:
            return _hold("model_mismatch", "Model echo does not match the configured model.", proposal)
        expected_prompt = self.prompt_version
        if expected_prompt is None:
            from cpr_ai_prompt import CPR_AI_PROMPT_VERSION

            expected_prompt = CPR_AI_PROMPT_VERSION
        if proposal.prompt_version != expected_prompt:
            return _hold("prompt_version_mismatch", "Prompt-version echo does not match the host prompt.", proposal)
        outcome = self.policy.validate(context, proposal)
        # The SDK boundary has proved that this was a contemporaneous, pinned
        # regime classification.  Preserve it even when hard execution gates
        # reject the proposed entry or scale-in.
        if outcome.accepted_regime is None and outcome.validation_code not in {
            "invalid_position_state",
            "invalid_frozen_context",
        }:
            outcome.accepted_regime = proposal.regime
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
