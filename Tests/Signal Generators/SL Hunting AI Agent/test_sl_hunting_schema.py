"""Strict-schema tests for the SLHuntingDecision output contract."""

from __future__ import annotations

import pytest
from sl_hunting_agent import SLHuntingDecision
from sl_hunting_knowledge import (
    FINAL_OUTPUT_INSTRUCTION,
    MAX_SYSTEM_PROMPT_CHARS,
    build_system_prompt,
)


def test_system_prompt_stays_inside_regression_budget():
    prompt = build_system_prompt() + FINAL_OUTPUT_INSTRUCTION
    assert len(prompt) <= MAX_SYSTEM_PROMPT_CHARS


def _valid_payload(**overrides):
    payload = {
        "action": "ENTER_LONG",
        "stop": 24950.0,
        "target": 25100.0,
        "confidence": 7,
        "setup": "pivot_support_hammer",
        "reasoning": "Hammer at pivot with bullish confirmation; tight stop, clear target.",
        "model_used": "claude-opus-4-8",
    }
    payload.update(overrides)
    return payload


def test_valid_decision_parses():
    decision = SLHuntingDecision.model_validate(_valid_payload())
    assert decision.action == "ENTER_LONG"
    assert decision.confidence == 7


def test_confidence_out_of_range_is_rejected():
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence=11))
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence=-1))


def test_stop_and_target_bounds_are_enforced():
    """SLH-002: hallucinated stop/target garbage must fail schema validation.

    A negative or absurd stop silently disables the mechanical underlying
    stop for the trade (only max-loss/square-off remain), so the record of
    the decision must never carry such values.
    """
    for bad in (-1.0, -1e9, float("nan"), float("inf"), 10_000_001.0):
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(stop=bad))
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(target=bad))
    # 0.0 stays valid -- the documented placeholder for EXIT/HOLD decisions.
    decision = SLHuntingDecision.model_validate(
        _valid_payload(action="HOLD", stop=0.0, target=0.0)
    )
    assert decision.stop == 0.0 and decision.target == 0.0


def test_entry_actions_require_positive_stop_and_target():
    """SLH-002 / Codex (PR #43): an ENTER decision with a 0 (or omitted) stop or
    target must fail validation, so a hallucinated entry can't be recorded as a
    real trade with no levels (which would defeat the order-tool guard and
    corrupt the decision journal). EXIT/HOLD keep their 0.0 placeholders."""
    for action in ("ENTER_LONG", "ENTER_SHORT"):
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(action=action, stop=0.0))
        with pytest.raises(Exception):
            SLHuntingDecision.model_validate(_valid_payload(action=action, target=0.0))
        # Omitted stop/target default to 0.0 -> also rejected for entries.
        with pytest.raises(Exception):
            payload = _valid_payload(action=action)
            payload.pop("stop")
            payload.pop("target")
            SLHuntingDecision.model_validate(payload)
        # Both positive -> valid.
        ok = SLHuntingDecision.model_validate(_valid_payload(action=action, stop=24950.0, target=25100.0))
        assert ok.action == action
    for action in ("EXIT", "HOLD"):
        ok = SLHuntingDecision.model_validate(_valid_payload(action=action, stop=0.0, target=0.0))
        assert ok.action == action


def test_json_schema_omits_min_max_on_stop_and_target():
    """Same Claude-schema constraint as confidence: bounds live in validators,
    never as minimum/maximum keys in the described JSON schema."""
    schema = SLHuntingDecision.model_json_schema()
    for field_name in ("stop", "target"):
        props = schema["properties"][field_name]
        assert "minimum" not in props and "maximum" not in props


def test_strict_rejects_unknown_fields_and_coercion():
    # extra field forbidden
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(unexpected="x"))
    # strict mode: a string is not coerced to int for confidence
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(confidence="7"))


def test_invalid_action_rejected():
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(action="BUY"))


def test_exit_leg_defaults_to_both_and_validates():
    """Per-leg exit selector: default BOTH, accepts the three literals, rejects others."""
    assert SLHuntingDecision.model_validate(_valid_payload()).exit_leg == "BOTH"
    for leg in ("NIFTY", "BNF", "BOTH"):
        assert SLHuntingDecision.model_validate(_valid_payload(exit_leg=leg)).exit_leg == leg
    with pytest.raises(Exception):
        SLHuntingDecision.model_validate(_valid_payload(exit_leg="SENSEX"))


def test_json_schema_omits_min_max_on_confidence():
    """Regression guard: Claude rejects minimum/maximum on integer types."""
    schema = SLHuntingDecision.model_json_schema()
    conf = schema["properties"]["confidence"]
    assert conf["type"] == "integer"
    assert "minimum" not in conf
    assert "maximum" not in conf


def test_system_prompt_has_final_output_marker():
    prompt = build_system_prompt() + FINAL_OUTPUT_INSTRUCTION
    assert "FINAL OUTPUT FORMAT" in prompt
    # The method's core rules should be present in the agent's "brain".
    assert "pivot" in prompt.lower()
    assert "confirmation" in prompt.lower()


def test_system_prompt_has_per_leg_exit_knowledge():
    """v5: the mirror is tied for hard risk but per-leg for premise exits (exit_leg)."""
    prompt = build_system_prompt() + FINAL_OUTPUT_INSTRUCTION
    assert "exit_leg" in prompt
    assert "PREMISE-INVALIDATION is PER-LEG" in prompt
    assert "HARD RISK stays TIED" in prompt


def test_system_prompt_has_v2_markers():
    """v2: BankNIFTY cross-confirmation section + the dynamic-sizing note are present."""
    prompt = build_system_prompt()
    assert "CROSS-INDEX CONFIRMATION" in prompt
    assert "bank_nifty" in prompt and "cross_index" in prompt
    # The agent is told sizing is automatic at ~Rs.2500 risk (it does not pick lots).
    assert "2500" in prompt


def test_system_prompt_has_v3_gap_knowledge():
    """v3: the gap/retail-positioning knowledge from the video is present."""
    prompt = build_system_prompt()
    assert "READING RETAIL POSITIONING" in prompt
    low = prompt.lower()
    assert "gap-up" in low and "gap-down" in low
    # The momentum-context nuance (don't fade every big candle).
    assert "momentum" in low


def test_system_prompt_has_v3a_bnf_knowledge():
    """v3a: the BankNIFTY live-trading methodology section + merged lessons are present."""
    prompt = build_system_prompt()
    # The new advisory BankNIFTY-specific section and its distinctive markers.
    assert "BANK NIFTY — SPECIFIC BEHAVIOUR" in prompt
    assert "Sensex" in prompt              # triple-index (BNF + NIFTY + Sensex) read
    assert "MAJOR index" in prompt         # BankNIFTY as the major/base index
    low = prompt.lower()
    assert "time-decay" in low             # G5 theta discipline merged into RISK
    assert "closing point" in low          # G2 closing-price invalidation level
    # It must sit AFTER the existing cross-index section (advisory context that extends it),
    # and must NOT weaken the mandatory candle+confirmation rule.
    assert prompt.index("BANK NIFTY — SPECIFIC BEHAVIOUR") > prompt.index("CROSS-INDEX CONFIRMATION")
    assert "execute NIFTY ATM options ONLY" in prompt


def test_system_prompt_has_v3d_conditional_gap_knowledge():
    """v3d: prior-days conditional gap read, reachability, and gap-size asymmetry are present."""
    prompt = build_system_prompt()
    assert "READ THE GAP AGAINST THE PRIOR DAYS" in prompt
    assert "SL-REACHABILITY TEST" in prompt
    assert "GAP-SIZE ASYMMETRY" in prompt
    # The flat-open seller-hunt long lives inside the OPENING DRIVE section as variant B.
    assert "Variant B" in prompt
    # v3c's opening-drive exception must still be present and scoped.
    assert "OPENING DRIVE" in prompt


def test_system_prompt_has_v3e_participation_knowledge():
    """v3e: both-sides participation, huge-gap nuance, third-index lag, setup staleness."""
    prompt = build_system_prompt()
    assert "BOTH-SIDES PARTICIPATION" in prompt
    assert "HUGE gap" in prompt
    assert "THIRD-INDEX LAG" in prompt
    assert "SETUP STALENESS" in prompt


def test_system_prompt_has_v3f_transcript_match_knowledge():
    """v3f: July 4-8 transcript + agent-match lessons are present."""
    prompt = build_system_prompt()
    assert "BUYER-INVENTORY FADE" in prompt
    assert "TARGET-BOOKED" in prompt
    assert "GAP-DOWN CONTINUATION SHORT" in prompt
    assert "NO INSTANT FLIP" in prompt
    assert "MASKED BNF LAG" in prompt


def test_system_prompt_has_v3g_full_2026_sweep_knowledge():
    """v3g: the Jan-Jul 2026 transcript sweep's carry-risk refinements are present."""
    prompt = build_system_prompt()
    assert "EVENT / HOLIDAY PARTICIPATION" in prompt
    assert "CONSTRUCTED-BASE CONTINUATION" in prompt
    assert "PREVIOUS-CHART LINKAGE" in prompt
    assert "WEEKEND / HOLIDAY CARRY-RISK" in prompt


def test_system_prompt_has_v3h_remaining_transcript_knowledge():
    """v3h: remaining-video fallback transcript lessons are present."""
    prompt = build_system_prompt()
    assert "UNIQUE-TRADE FILTER" in prompt
    assert "PROFIT-HOLD" in prompt
    assert "TIMEFRAME FIT" in prompt
    assert "PLAN-OF-EXECUTION" in prompt
    assert "NO DAILY-INCOME PRESSURE" in prompt
    assert "POST-LOSS SPEED LIMIT" in prompt


def test_system_prompt_has_v3i_premium_rr_knowledge():
    """v3i: 10 Jul live session — premium non-confirmation exit + R:R-bait read."""
    prompt = build_system_prompt()
    assert "PREMIUM NON-CONFIRMATION" in prompt
    assert "R:R-BAIT AT ROUND-NUMBER REJECTIONS" in prompt
    # The actionable exit rule: book the average target when premiums lag the spot move.
    assert "AVERAGE target" in prompt


def test_system_prompt_has_v3j_averaging_trap_knowledge():
    """v3j: 13-14 Jul gap-down sessions, cross-checked against the agent's own journal.

    Three lessons, each tied to a real 14 Jul decision:
    - AVERAGING TRAP fixes the trade-1 premise (the agent read "starved sellers" and
      went long where IH read yesterday's recovery-buyers as the trapped crowd).
    - MOVE-EXHAUSTION fixes trade 3 (re-shorting the same spent move into an expiry
      range, stopped out in 5 seconds) — the same-direction blind spot NO INSTANT FLIP
      does not cover.
    - The cross-index "stale verdict" escape hatch is scoped to the opening hour, since
      trade 3 used it at 10:04 to override an opposing verdict at confidence 6.
    """
    prompt = build_system_prompt()
    assert "AVERAGING TRAP" in prompt
    assert "MOVE-EXHAUSTION" in prompt
    # The entry-timing half of the averaging trap: never enter at the gap extreme.
    assert "do NOT enter at the gap extreme" in prompt
    # Expiry is fuel for an existing premise, never a premise of its own.
    assert "EXPIRY IS CONTEXT, NOT A PREMISE" in prompt
    # The "stale" escape hatch must be explicitly bounded to the opening hour.
    assert "SCOPE OF THIS \"STALE\" ESCAPE HATCH" in prompt


def test_system_prompt_has_v3k_flat_open_gate_knowledge():
    """v3k: 15 Jul sessions — the flat-open hunt needs a crowd that really participated.

    After a WEAK-momentum down day, a flat open puts nobody in pain (and leaves the
    closing-point support in the recovery's path) — the plan flips WITH the prior
    direction, while a gap in EITHER direction re-arms the seller-hunt. Scopes the
    blanket "FLAT or GAP-DOWN -> look UP" default.
    """
    prompt = build_system_prompt()
    assert "FLAT-OPEN PARTICIPATION GATE" in prompt
    # The asymmetry in one line: either-direction gap hunts, flat goes with-trend.
    assert "flat" in prompt and "go with the selling" in prompt
    # It must scope, not delete, the textbook flat/gap-down hunt above it.
    assert "PRIME TRAP zone" in prompt


def test_system_prompt_has_v3l_closing_point_and_shared_gap_knowledge():
    """v3l: 16 Jul split-gap session, cross-checked against journal rows 21-22.

    - CLOSING-POINT HOLD TEST answers whether an overnight crowd exists at all: a prior
      rejection that never BROKE the closing point means that crowd booked and left, so
      there is no inventory to hunt -> follow the move instead.
    - The OPENING DRIVE gap-up branch now requires the gap to be SHARED: the agent fired
      it on NIFTY's gap while BankNIFTY opened flat at its own closing point (IH read the
      same open as a short) and the basket lost Rs.1,333.
    """
    prompt = build_system_prompt()
    assert "CLOSING-POINT HOLD TEST" in prompt
    assert "SHARED-GAP REQUIREMENT" in prompt
    # The decisive tell: a flat major index beside a gapped NIFTY kills the long branch.
    assert "flat major index" in prompt.lower()
    # The hold test must state both arms (seated-and-huntable vs booked-and-gone).
    assert "BROKE it and held beyond" in prompt
    # The leader-fails-to-lead exit keeps its scope so it can't collide with RISK's
    # "SLOW-but-CONTINUOUS is the sustainable kind" rule.
    assert "SLOW-but-CONTINUOUS" in prompt


def test_system_prompt_has_v3m_gift_gap_and_loss_flip_knowledge():
    """v3m: 17 Jul flat-open loss day (IH's first loss in the series).

    - GIFT-GAP AFTER A NOBODY'S-CROWD DAY: after a small-momentum day with the
      closing point uncrossed, a gap in EITHER direction traps the side it appears
      to reward (fade it); flat means there is nobody to hunt.
    - NO INSTANT FLIP now also bans the mid-loss panic flip (booking a small loss to
      instantly reverse into the breakout), tying into POST-LOSS SPEED LIMIT.
    """
    prompt = build_system_prompt()
    assert "GIFT-GAP AFTER A NOBODY'S-CROWD DAY" in prompt
    # Each gap direction traps the side it appears to reward on a thin day.
    assert "traps its own recipient" in prompt
    # The losing-side flip ban lives inside the existing NO INSTANT FLIP bullet
    # (assert wrap-independent fragments, not exact line breaks).
    assert "LOSING side" in prompt and "whipsaw" in prompt
    assert "POST-LOSS SPEED LIMIT" in prompt


def test_system_prompt_has_v3n_closed_chart_knowledge():
    """v3n: 19 Jul closed-chart lecture (IH's week review + self-diagnosed loss).

    - RECRUITMENT HISTORY: two near-identical charts demand OPPOSITE plans, because a
      first reversal-type move recruits nobody while the SECOND consecutive
      same-direction day seats the crowd.
    - ONE BREAKDOWN, NOT TWO: the rule whose absence cost IH the 17 Jul trade — after
      one level break the next rarely breaks; sellers are likely seated and buyers are
      definitely evicted.
    - The CLOSING-POINT HOLD TEST's "held beyond" arm now requires real MOMENTUM: a
      break that idles for hours seats nobody (a correction to v3l).
    """
    prompt = build_system_prompt()
    assert "RECRUITMENT HISTORY, NOT CHART SHAPE" in prompt
    assert "ONE BREAKDOWN, NOT TWO" in prompt
    # The recruitment law, wrap-independent.
    assert "SECOND" in prompt and "consecutive same-direction day" in prompt
    # The asymmetric fallback: a breakdown always evicts the buyers.
    assert "buyers are never" in prompt
    # The v3l correction: break-and-held only seats a crowd if momentum followed.
    assert "produced actual MOMENTUM" in prompt


def test_system_prompt_has_v3o_flush_day_and_solo_leader_knowledge():
    """v3o: 20-21 Jul sessions (IH won the news gap-down, lost the flat-open long).

    - BOTH-WAYS FLUSH DAY: the second way a day ends with nobody seated — after a
      violent both-ways session there is nothing to fade; follow the opening type,
      and treat the flat-open first push as recruitment bait (it caught IH on 21 Jul).
    - SOLO-LEADER VETO: BankNIFTY-moving-first is void as an entry tell when the other
      two indices are capped below their closing points (IH: "I trusted BankNIFTY too
      much").
    """
    prompt = build_system_prompt()
    assert "BOTH-WAYS FLUSH DAY" in prompt
    # The plan collapse and the flat-branch bait, wrap-independent.
    assert "as the opening, so the plan" in prompt
    assert "recruitment bait" in prompt
    # The disambiguation question against GIFT-GAP.
    assert "WHY nobody is seated" in prompt
    assert "SOLO-LEADER VETO" in prompt
    # The veto's release condition.
    assert "reclaim its closing point" in prompt


def test_system_prompt_has_v3p_runaway_trend_knowledge():
    """v3p: 22 Jul — the agent HELD 59/59 bars on a one-way breakdown IH traded well.

    Every HOLD ended "no confirmed reversal pattern at a level", because the prompt
    had no with-trend entry path outside OPENING_DRIVE's 15-minute window. RUNAWAY
    TREND is the third (and last) exception to pattern+confirmation: the ABSENCE of a
    retracement is the signal, and the first real retracement is the invalidation.
    """
    prompt = build_system_prompt()
    assert "RUNAWAY TREND" in prompt
    assert "THE ABSENCE OF A RETRACEMENT IS ITSELF THE SIGNAL" in prompt
    # The invalidation must be explicit -- this branch has no reversal pattern to lean on.
    assert "INVALIDATION IS THE FIRST REAL RETRACEMENT" in prompt
    # It must be gated on all three indices agreeing, and never be a fade.
    assert "ALL THREE indices agree" in prompt
    assert "NEVER as a" in prompt and "counter-trend fade" in prompt
    # The three entry gates must all advertise the new exception, or it is unreachable.
    assert "RUNAWAY TREND no-retracement continuation" in prompt   # ROLE + DECISION_RULES
    # PSYCHOLOGY's "wait in a fast trend" line must carry its limiting clause.
    assert "IMPORTANT LIMIT ON THAT" in prompt


def test_system_prompt_has_v3q_reentry_gate_and_expiry_pin_knowledge():
    """v3q: 23 Jul — the agent's 3 re-entries all lost (net -Rs.7,055 on a +Rs.13,688 day).

    MOVE-EXHAUSTION / NO INSTANT FLIP already banned those re-entries, but both are
    judgement rules the agent satisfied rhetorically by naming a fresh setup each time.
    The POST-EXIT RE-ENTRY GATE makes the same ban mechanically checkable. Plus IH's
    expiry-pinning read: take the level-break trigger from a NON-expiring index.
    """
    prompt = build_system_prompt()
    assert "POST-EXIT RE-ENTRY GATE" in prompt
    # The gate must be checkable, not another judgement call. v3s moved the TIME arm
    # out of prose and into the order tool, so the wording here changed with it --
    # what must survive is that a time floor exists and runs from the last close.
    assert "a hard cooldown runs from your last close" in prompt
    assert "NEW STRUCTURAL EVENT" in prompt
    # The exact loophole that cost money today must be named.
    assert "A DIFFERENT PATTERN NAME ON THE SAME STRUCTURE IS NOT A NEW PREMISE" in prompt
    # Entries only -- exits must never be delayed by the gate.
    assert "This gate governs ENTRIES ONLY" in prompt
    # Expiry pinning: the expiring index is the wrong place to look for a clean break.
    assert "EXPIRING INDEX RESISTS THE BREAK" in prompt
    assert "Fuel yes, trigger no" in prompt
    # Crowd-behaviour nuance: aligned crowds don't cascade.
    assert "A CONFIDENT CROWD DOES NOT STAMPEDE" in prompt


def test_system_prompt_has_v3s_laggards_and_enforced_cooldown_knowledge():
    """v3s: 27 Jul — IH booked without his breakdown because BankNIFTY delivered
    alone while Sensex/NIFTY never broke; and the re-entry gate's TIME arm moved
    into code after the prompt version was talked past twice."""
    prompt = build_system_prompt()
    assert "LAGGARDS NEVER JOINED" in prompt
    # The booking trigger: leader spent while the followers are still unbroken.
    # (Wrap-independent fragments only -- the phrase spans a line break.)
    assert "leader spent" in prompt and "laggards absent" in prompt
    assert "is the booking signal" in prompt
    # The urgency: your own position becomes the next hunted inventory.
    assert "liquidity for someone else's trade" in prompt
    # The agent must know the time arm is now refused by the tool, not self-policed.
    assert "ENFORCED IN CODE" in prompt
    # ...and that clearing the clock is not by itself permission to trade.
    assert "does NOT authorise a trade" in prompt


def test_system_prompt_has_v3t_expiry_asymmetry_and_morning_speed_knowledge():
    """v3t: 28 Jul — IH booked into strength rather than waiting for confirmation,
    and warned that a fast morning stop-out is normal variance, not a reason to retry.

    CORRECTED 2026-07-29. The original rule was scoped to "expiry day" and quoted a
    ~3.5x asymmetry. Both were wrong. The 3.5x came from BASKET option_pnl (a 7-DTE
    NIFTY leg plus a 0-DTE BankNIFTY mirror) divided by NIFTY-ONLY spot points — two
    underlyings and two expiries in one ratio. On the NIFTY leg alone the figures are
    139.45->131.00 on 650 qty for 4.55 adverse points (1.86 per point) against
    131.00->150.90 on 390 qty for 24.35 favourable points (0.82 per point): ~2.3x.
    And IH was trading the EXPIRING series while the agent's NIFTY leg was 7 days
    out, so "expiry-day time-value collapse" was never the mechanism for our leg.
    The rule now keys off the held option's own days-to-expiry.
    """
    prompt = build_system_prompt()
    # The holding rule -- distinct from PREMIUM NON-CONFIRMATION above it.
    assert "PREMIUM ASYMMETRY" in prompt
    assert "BOOK INTO STRENGTH" in prompt
    # The corrected, leg-level measurement must be what is quoted.
    assert "2.3x asymmetry" in prompt
    assert "3.5x" not in prompt          # the bad basket-derived figure is gone
    # It must key off OUR contract, not the calendar -- the original scoping error.
    assert "the days-to-expiry of the option you actually hold" in prompt
    assert "NOT whether some index" in prompt
    # Magnitude is situational, not a constant.
    assert "situational, not a constant" in prompt
    # ...and it still must not be read as licence to cut winners early.
    assert "PROFIT-HOLD still governs" in prompt
    # A fast morning stop-out must not be read as evidence about the next trade.
    assert "MORNING SPEED IS NOT INFORMATION" in prompt
    assert "is a FLOOR, not the standard" in prompt
    # ...but the rule must NOT harden into a one-trade-per-morning ban: both of the
    # agent's recorded morning winners were second trades after a stop-out.
    assert "NOT a ban on a" in prompt and "second trade of the morning" in prompt
    # Entry precheck: quantify the loss before entering, not after.
    assert "PRE-COMPUTE BOTH NUMBERS" in prompt
    assert "a loss accepted BEFORE entry" in prompt


def test_system_prompt_has_v3u_gap_size_and_no_fuel_knowledge():
    """v3u: 29 Jul — a large gap-up with nobody trapped.

    IH bought WITH the gap but said plainly that the oversized gap made it riskier,
    that buyers'/sellers' stops were not available nearby, and that he would take a
    normal profit rather than a runner. The agent's own book supplied the sharp
    edge: a LONG held 105 seconds gained 4.65 spot points and still lost Rs.5,300.
    """
    prompt = build_system_prompt()
    # A bigger gap is a worse trade, not a better one.
    assert "GAP SIZE IS A RISK DIAL, NOT A CONFIDENCE DIAL" in prompt
    # Must not be confused with the existing cross-index gap rule.
    assert "GAP-SIZE ASYMMETRY, which compares the" in prompt
    # Following the market (no trapped crowd) means a normal target, decided up front.
    assert "NO NEARBY STOPS" in prompt
    assert "NORMAL / average-target" in prompt
    # ...but it must not become a licence to take trades that are too small.
    assert "the answer is still HOLD" in prompt
    # Premium can go NEGATIVE on a favourable spot move, not merely lag.
    assert "IT CAN GO NEGATIVE, NOT MERELY WEAK" in prompt
    assert "never read" in prompt and "as \"I am in" in prompt
    # The round-trip cost of abandoning a trade immediately.
    assert "pays the round-trip cost for no exposure" in prompt


def test_system_prompt_has_v3v_small_gap_and_carryover_knowledge():
    """v3v: 30 Jul — after 2-3 positive days IH sold puts minutes after a flat /
    slightly-gap-down open, and said he would have targeted the SAME seated buyers
    even on a slight gap-up. He then booked early, citing yesterday's session going
    sideways after its opening move.

    Distilled from the VIDEO only: the agent's own 30 Jul book is unusable (an
    unjournalled trade, a manual intervention, repeated market-data outages, and a
    stale entry LTP that overstated one trade by ~Rs.4,855).
    """
    prompt = build_system_prompt()
    # The gap-up escape hatch needs a gap proportional to the run that seated them.
    assert "A SMALL GAP DOES NOT RESCUE A SEATED CROWD" in prompt
    assert "against the SIZE OF THE RUN" in prompt
    # The plain consequence: the crowd picks the side, not the open.
    assert "the OPEN direction does not" in prompt
    assert "the trapped crowd does" in prompt
    # Session character carries over and TIGHTENS the target.
    assert "YESTERDAY'S MOMENTUM CHARACTER CALIBRATES TODAY'S PATIENCE" in prompt
    assert "it is chop" in prompt
    # It must not be confused with the two existing previous-session rules.
    assert "which asks WHO was" in prompt and "which asks WHICH WAY" in prompt


def test_v3v_small_gap_rule_sits_inside_recruitment_history():
    """The refinement must stay attached to the rule it qualifies.

    Read alone it would contradict the gap-up branch ("already in profit and cannot
    be targeted"); it only makes sense as a size qualifier on that same branch.
    """
    prompt = build_system_prompt()
    start = prompt.index("RECRUITMENT HISTORY")
    small_gap = prompt.index("A SMALL GAP DOES NOT RESCUE A SEATED CROWD")
    assert small_gap > start
    # ...and before the NEXT top-level bullet after the block it qualifies.
    nxt = prompt.index("WEEKEND / HOLIDAY CARRY-RISK")
    assert small_gap < nxt


def test_system_prompt_has_v3w_entry_point_and_counter_move_knowledge():
    """v3w: 31 Jul — IH's LOSING session, which is rarer material than the wins.

    He went with a flat open on the buy side, then cut: "the trade still looks
    okay, but because of the ENTRY POINT a problem is being created", and "in a
    trade that is going wrong you cannot apply your mind". Before entering he had
    also named the range test: a sudden BIG adverse move means the market wants to
    stay in the range, while small selling alongside a breakout is fine.
    """
    prompt = build_system_prompt()
    # A right read taken from the wrong place is a wrong trade.
    assert "THE ENTRY POINT IS PART OF THE PREMISE" in prompt
    assert "being eventually right does not" in prompt
    # The self-deception this rule exists to name.
    assert "You cannot think your way out of a" in prompt
    assert "hope wearing the clothes of" in prompt
    # The size of the move AGAINST you is a premise test, not a pullback.
    assert "COUNTER-MOVE SIZE SAYS RANGE OR BREAKOUT" in prompt
    assert "intends to STAY in the range" in prompt
    # ...and must not be confused with the with-trend momentum-quality rule.
    assert "which reads the WITH-trend move" in prompt


def test_system_prompt_has_v3x_aggregate_inventory_and_option_rr_knowledge():
    """v3x: the 2 Aug weekly lecture scopes the crowd and the achievable target.

    The agent must reason about the dominant aggregate inventory rather than one
    hypothetical trader, reset a stale seller read after repeated failed breaks,
    and permit 1:1 only for an unusually clear, time-constrained option trade.
    """
    prompt = build_system_prompt()
    assert "AGGREGATE-INVENTORY TEST" in prompt
    assert "greatest aggregate quantity" in prompt
    assert "REPEATED-FAILURE INVENTORY RESET" in prompt
    assert "repeated breakdown-and-recovery" in prompt
    assert "OPTION-TIME-ADJUSTED REWARD/RISK" in prompt
    assert "approximately 1:1" in prompt
    assert "Less than 1:1" in prompt and "HOLD" in prompt
    # The refinement must not erase the guardrails it relies on.
    assert "UNIQUE-TRADE FILTER" in prompt
    assert "TARGET-BOOKED crowd test" in prompt
    assert "TIMEFRAME FIT" in prompt
    assert "POST-EXIT RE-ENTRY GATE" in prompt


def _worst_case_runtime_blocks() -> tuple[str, str]:
    """Render the LARGEST lessons block and pre-open note the runtime can inject.

    Built through the REAL formatters rather than by arithmetic, because the
    arithmetic is what went wrong before: the previous version of this test
    assumed `12 * 280 + 2500`, which ignored each lesson's 80-character scope
    and its per-lesson formatting, and understated the note by ~1,100. It read
    2,659 characters light in total.

    Two fixture traps, both discovered the hard way:

    * `StoredLesson` is tamper-evident. An approved record needs `id` equal to
      `_slug(scope, lesson)` AND an `approval_digest` equal to
      `lesson_content_digest(record)`; anything else is silently rejected by
      `_validated_records` and renders as an empty block.
    * `_slug` truncates to 48 characters, so records whose scope shares a long
      prefix collide on id and `consolidate` keeps only one. The varying part
      must come FIRST.
    """
    import datetime as _dt
    import json
    import os
    import tempfile

    import sl_hunting_lessons as L
    import sl_hunting_premarket as P

    def _lesson(index: int) -> dict:
        scope = f"s{index:02d}" + "S" * (L.MAX_SCOPE_CHARS - 3)
        lesson = f"l{index:02d}" + "X" * (L.MAX_LESSON_CHARS - 3)
        record = {
            "id": L._slug(scope, lesson),
            "scope": scope,
            "lesson": lesson,
            "rationale": "r" * L.MAX_RATIONALE_CHARS,
            "evidence": {"wins": 999, "losses": 999, "sample_size": 9999},
            "confidence": 5,
            "status": "approved",
            "created_at": "2026-08-27T00:00:00Z",
            "updated_at": "2026-08-27T00:00:00Z",
        }
        record["approval_digest"] = L.lesson_content_digest(record)
        return record

    lessons = [_lesson(i) for i in range(L.MAX_LIVE_LESSONS)]
    validated, rejected = L._validated_records(lessons)
    assert not rejected and len(validated) == L.MAX_LIVE_LESSONS, (
        "the worst-case lessons fixture is being rejected, so this test would "
        f"measure an empty block and prove nothing (rejected={len(rejected)})"
    )
    lessons_block = L.format_lessons(lessons)

    note = {
        "for_date": "2026-08-28",
        "source": "s" * P.MAX_SOURCE_CHARS,
        "context": "c" * P.MAX_LINE_CHARS,
        "plan": ["p" * P.MAX_LINE_CHARS for _ in range(P.MAX_PLAN_ITEMS)],
        "levels": [
            {
                "index": "I" * P.MAX_INDEX_CHARS,
                "resistance": [99999.5] * P.MAX_LEVELS_PER_SIDE,
                "support": [99999.5] * P.MAX_LEVELS_PER_SIDE,
            }
            for _ in range(P.MAX_INDEX_BLOCKS)
        ],
    }
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(note, stream)
        loaded = P.load_premarket_note(path)
        assert loaded is not None, "the worst-case note fixture failed to validate"
        note_block = P.format_premarket_note(loaded, _dt.date(2026, 8, 28))
    finally:
        os.unlink(path)
    assert note_block, "the worst-case note rendered empty"
    return lessons_block, note_block


# The ceiling the runtime-injected material must stay under. Measured at 8,519
# on 2026-08-27 (lessons 4,918 + note 3,601); 12,000 leaves ~40% of room for an
# ordinary widening while still failing on anything that changes the order of
# magnitude. This number is deliberately NOT derived from the per-field caps --
# a derived bound would move silently when those caps move, which is precisely
# the change this test exists to catch.
MAX_RUNTIME_INJECTED_CHARS = 12_000


def test_runtime_injected_blocks_stay_small():
    """The real guard on runaway lessons / notes, now that the cap cannot be.

    `MAX_SYSTEM_PROMPT_CHARS` was raised to 350,000 on 2026-08-27, which is far
    too loose to catch a runaway lessons file or a malformed note -- the failure
    its own comment block says it exists to catch. The detection was moved here
    on purpose.

    This fails if anyone widens `MAX_LESSON_CHARS`, `MAX_LIVE_LESSONS`,
    `MAX_PLAN_ITEMS`, `MAX_LINE_CHARS`, `MAX_LEVELS_PER_SIDE` or
    `MAX_INDEX_BLOCKS` enough to matter, regardless of what the cap says.
    """
    lessons_block, note_block = _worst_case_runtime_blocks()
    total = len(lessons_block) + len(note_block)

    assert total <= MAX_RUNTIME_INJECTED_CHARS, (
        f"worst-case runtime injection is {total:,} chars "
        f"(lessons {len(lessons_block):,} + note {len(note_block):,}), over the "
        f"{MAX_RUNTIME_INJECTED_CHARS:,} ceiling. The prompt cap will NOT catch "
        "this -- either narrow the per-field caps or raise this ceiling "
        "deliberately, with the new number measured and recorded."
    )


def test_assembled_prompt_plus_worst_case_runtime_fits_the_cap():
    """Knowledge + the worst the runtime can add must still fit, with room spare.

    The cap is a sanity bound, not a budget knowledge must squeeze into, so this
    asserts real headroom rather than a bare fit. It uses the MEASURED worst case
    from `_worst_case_runtime_blocks`, not arithmetic -- the arithmetic version
    of this test read 2,659 characters light and reported the budget as 154,140
    when it was really 151,481.
    """
    from sl_hunting_knowledge import MAX_SYSTEM_PROMPT_CHARS

    lessons_block, note_block = _worst_case_runtime_blocks()
    assembled = len(build_system_prompt() + FINAL_OUTPUT_INSTRUCTION)
    worst_case = assembled + len(lessons_block) + len(note_block)

    assert worst_case < MAX_SYSTEM_PROMPT_CHARS, (
        f"assembled prompt {assembled:,} plus worst-case runtime injection "
        f"{len(lessons_block) + len(note_block):,} exceeds the "
        f"{MAX_SYSTEM_PROMPT_CHARS:,} cap"
    )


def test_system_prompt_has_v3x_profit_depth_and_known_road_knowledge():
    """v3x (3 Aug live session): IH held a gap-up long and booked it while it was
    still working, because the move had narrowed to BankNIFTY alone.

    "More momentum could come, but this is not one of the setups that work for us...
    we waited as far as we knew the road. Now we do not know the road." He also
    split "the buyers" by profit depth: the Friday cohort was shaken out by the
    gap-up, while traders positioned from far below never moved -- and the tell was
    that no big, quick selling appeared.
    """
    prompt = build_system_prompt()
    # One side is two cohorts, and only the marginal one is huntable.
    assert "PROFIT DEPTH SPLITS ONE SIDE INTO TWO COHORTS" in prompt
    # Wrap-independent: this phrase spans a line break in the source.
    assert "NOT weak" in prompt and "riding the move" in prompt
    # Character of the counter-move identifies who is leaving.
    assert "THE COUNTER-MOVE'S SIZE AND SPEED SAY WHICH COHORT IS LEAVING" in prompt
    assert "BIG, QUICK selling" in prompt
    # Exit when the read runs out, not only when the thesis breaks.
    assert "ONLY RIDE AS FAR AS YOU KNOW THE ROAD" in prompt
    # Wrap-independent fragment: the sentence spans a line break in the source.
    assert "paying to find out" in prompt
    # It must be distinguished from the two exits it is NOT.
    assert "NOT the same as premise-invalidation" in prompt
    assert "no read, no position" in prompt


def test_system_prompt_has_v3y_seated_buyer_and_index_hierarchy_knowledge():
    """v3y (4 Aug live session): IH's LOSING trade, and the day the same opening
    type produced the opposite plan two days running.

    Both days gapped up. Day one he bought WITH the gap; day two he sold puts
    AGAINST the buyers -- because day one had a mid-week holiday ahead (thin crowd)
    and a retracement inside the rally, while day two had no holiday and all three
    indices sat on exact round-number support (seated crowd). He then cut the trade
    for a loss the moment BankNIFTY turned up, saying he could have handled NIFTY
    and Sensex ticking against him but not the major index.
    """
    prompt = build_system_prompt()
    # The gap-up long branch must first prove the buyers are actually absent.
    assert "SEATED-BUYER TEST" in prompt
    # Wrap-independent fragments: these sentences span line breaks in the source.
    assert "EXACT round-number support" in prompt
    assert "takes LESS risk" in prompt
    assert "identical-looking gap-up reads the OPPOSITE way" in prompt
    # The hunt needs the break, not merely the approach to the level.
    assert "CLOSING-PRICE BREAKDOWN IS THE TRIGGER" in prompt
    assert "Sitting on that level is not" in prompt
    # The indices are not equal once a position is losing.
    assert "INDEX HIERARCHY ON THE WAY OUT" in prompt
    assert "DISQUALIFYING" in prompt
    # ...and the three discipline lessons the loss paid for.
    assert "A TRIGGER THAT NEVER FIRED IS AN EXIT REASON" in prompt
    assert "BEING DIRECTIONALLY RIGHT DOES NOT EARN THE HOLD" in prompt
    assert "A SLOW GRIND AT THE LEVEL RECRUITS THE WRONG CROWD" in prompt
    assert "VOLATILE-DAY SIZING WIDENS BOTH ENDS" in prompt


def test_v3y_gap_conflict_does_not_contradict_the_opening_drive_branch():
    """The seated-buyer test must READ AS a precondition of the gap-up long, not as
    a second, competing gap-up rule. If both are stated flatly the agent can pick
    whichever suits the bar it is looking at."""
    prompt = build_system_prompt()
    seated = prompt.index("SEATED-BUYER TEST")
    gap_size = prompt.index("GAP SIZE IS A RISK DIAL")
    drive = prompt.index("OPENING DRIVE — early-session continuation exceptions")
    # It lives inside the OPENING DRIVE conditions, ahead of the risk-dial rule.
    assert drive < seated < gap_size
    # And it is explicitly ordered before the branch may fire.
    assert "BEFORE the long branch fires" in prompt


def test_system_prompt_has_v3z_missing_rip_and_rule_discipline_knowledge():
    """v3z (5 Aug live session): a WIN, and the session where he re-examines the
    v3y loss and keeps the rule anyway.

    Big gap-up, then rejection. He sold -- not fading the gap, but reading that
    retail never got short: "if retail HAD sold, the market would have started
    rising directly, leaving no time". No sellers above means a further push up
    attracts only buyers, so down is the path. He then booked early because the
    PREVIOUS day was a loss, and reflected that yesterday's INDEX HIERARCHY cut
    was wrong in outcome -- the market fell from almost exactly where he exited --
    and that the rule stands regardless.
    """
    prompt = build_system_prompt()
    # Absence of the hunt is evidence about who is absent.
    assert "THE MISSING RIP IS THE TELL" in prompt
    assert "leaving no time" in prompt
    # A big gap has nowhere to set the lure.
    assert "BAIT ROOM" in prompt
    # The meta-discipline lesson, which exists to protect v3y's exit rule.
    assert "A RULE THAT COST YOU MONEY YESTERDAY IS STILL THE RULE" in prompt
    assert "invisible by construction" in prompt
    assert "sample of one" in prompt
    # In-trade twin of BOTH-SIDES PARTICIPATION.
    assert "TWO-SIDED FLOW PROTECTS AN OPEN PROFIT" in prompt
    # Post-loss target discipline, distinct from the re-entry speed limit.
    assert "AFTER A LOSING DAY, TAKE THE GOOD PROFIT RATHER THAN THE BIG ONE" in prompt
    assert "POST-LOSS SPEED LIMIT, which governs" in prompt
    assert "NAME THE ONE WAY THIS TRADE FAILS" in prompt


def test_v3z_rule_discipline_cannot_be_read_as_licence_to_hold():
    """The dangerous misreading of "the rule cost me money" is "so hold longer".

    v3z must reinforce the v3y exit, never soften it, so the prompt has to keep
    both the hierarchy exit and the never-hold-a-loser rule intact alongside it.
    """
    prompt = build_system_prompt()
    assert "INDEX HIERARCHY ON THE WAY OUT" in prompt
    assert "NEVER hold a loser hoping for a reversal" in prompt
    # The lesson is explicitly about NOT relaxing an exit rule.
    # Wrap-independent: this sentence spans a line break in the source.
    assert "Never widen, delay, or suspend an exit" in prompt


def test_system_prompt_has_v4a_second_day_recruitment_knowledge():
    """v4a (6 Aug live session): IH bought a gap-up to hunt SELLERS, and the
    reasoning dates the inventory.

    One down day after a positive stretch recruits almost nobody -- traders cannot
    believe the turn. The SECOND consecutive down day is when confidence arrives
    and shorts actually get seated. So two down days plus a gap-up is a long
    against them, and the move should be sharp but small because a freshly
    recruited crowd holds tight stops.
    """
    prompt = build_system_prompt()
    assert "SECOND-DAY RECRUITMENT" in prompt
    # Wrap-independent fragments: these sentences span line breaks in the source.
    assert "confidence arrives" in prompt
    assert "a single session's move is not a crowd" in prompt
    # Tight stops -> sharp but small, and slow means the cluster is not there.
    assert "A FRESHLY RECRUITED CROWD HAS TIGHT STOPS" in prompt
    assert "SIGNATURE" in prompt
    assert "reduce the target, do not extend the hold" in prompt
    # The two-phase handling of a wobble, and the other two lessons.
    assert "A REJECTION BEFORE THE FLUSH IS NOISE" in prompt
    assert "ERRORS IN PROFIT ARE CHEAP" in prompt
    assert "PREFER A DIP TO A CHASE" in prompt


def test_v4a_rejection_rule_cannot_be_read_as_licence_to_hold_a_loser():
    """The dangerous misreading of "a rejection is noise" is "so sit through it".

    This is the same failure mode v3z's rule-discipline lesson had, and it matters
    more here because this one is explicitly about NOT closing. The prompt must
    keep every exit rule intact beside it and scope the narrowing precisely.
    """
    prompt = build_system_prompt()
    # It must state its own scope.
    assert "THIS IS NOT LICENCE TO HOLD A LOSER" in prompt
    assert "premise-invalidation" in prompt
    # ...and the exits it must not weaken must still be present.
    assert "NEVER hold a loser hoping for a reversal" in prompt
    assert "INDEX HIERARCHY ON THE WAY OUT" in prompt
    assert "A TRIGGER THAT NEVER FIRED IS AN EXIT REASON" in prompt
    # The discriminator has to be an observable, not a feeling.
    assert "THE DISCRIMINATOR IS FACTUAL, NOT A FEELING" in prompt


def test_system_prompt_has_v4b_post_gap_bounce_and_averaging_target_knowledge():
    """v4b (7 Aug live session): a WIN on the put side where he ENLARGED the
    target mid-trade.

    The central idea inverts the naive read of a post-gap bounce: a gap that fell
    straight down would let the trapped crowd out in two or three minutes, so the
    bounce exists to give them hope, make them hold or average, and deepen the
    loss. A crowd that has averaged then justifies a BIGGER target, and a stall
    mid-flush predicts one more leg rather than the end.
    """
    prompt = build_system_prompt()
    assert "THE POST-GAP BOUNCE IS THE TRAP DEEPENING" in prompt
    # Wrap-independent fragments: these sentences span line breaks in the source.
    assert "three minutes" in prompt
    assert "REGIME MEMORY DECIDES WHO SHOWS UP AT A LEVEL" in prompt
    assert "who has been PAID and who has been PUNISHED" in prompt
    # Target sizing from crowd behaviour, and the pair it completes.
    assert "A CROWD THAT HAS AVERAGED DOWN EARNS A BIGGER TARGET" in prompt
    assert "A FRESHLY RECRUITED CROWD HAS TIGHT STOPS" in prompt
    assert "EXPECT A SECOND LEG AFTER THE PAUSE" in prompt
    # The entry/exit asymmetry of the major index.
    assert "THE HIERARCHY IS ASYMMETRIC" in prompt
    assert "Be slow to enter on BankNIFTY alone" in prompt


def test_v4b_bounce_rule_names_what_would_actually_invalidate():
    """The bounce rule tells the agent NOT to exit on a bounce, so it must also
    say what a real invalidation looks like -- otherwise it reads as "ignore
    adverse movement", which is the failure mode every one of these lessons has.
    """
    prompt = build_system_prompt()
    # It must point at a concrete, checkable invalidation.
    assert "RECLAIMING the level" in prompt
    # ...and the second-leg rule must scope itself off an offside position.
    assert "does not extend to a position that is offside" in prompt
    # The exits it must not weaken are still present.
    assert "NEVER hold a loser hoping for a reversal" in prompt
    assert "INDEX HIERARCHY ON THE WAY OUT" in prompt


def test_system_prompt_has_v4c_manufactured_inventory_knowledge():
    """v4c (weekend lecture, not a session): where the market CREATES stops.

    Every other part of the method finds inventory that is already trapped. This
    adds the phase after that supply runs out: the market must manufacture a new
    crowd, and it does so wherever demand and supply can be made highest --
    which is what a breakout is FOR.
    """
    prompt = build_system_prompt()
    assert "WHEN THE TRAPPED INVENTORY IS SPENT, THE MARKET MANUFACTURES MORE" in prompt
    # Wrap-independent fragments: these sentences span line breaks in the source.
    assert "WHERE THE NEXT CROWD WILL BE BUILT" in prompt
    # The mechanism: doubt keeps size small, a break resolves it.
    assert "AMBIGUITY SUPPRESSES SIZE" in prompt
    assert "RECRUITMENT DEVICE" in prompt
    # ...and its direct corollary.
    assert "A FAILED BREAKOUT IS THE NORMAL OUTCOME" in prompt
    assert "ROUND NUMBERS AMPLIFY RECRUITMENT" in prompt
    assert "CROWD SIZE IS THE THIRD TARGET INPUT" in prompt


def test_v4c_breakout_rule_does_not_turn_into_a_fade_everything_rule():
    """"A failed breakout is normal" must not become "always fade breakouts".

    The runner has a live BREAKOUT branch (Regime Adaptive) and the method has a
    with-the-gap opening drive, so a blanket anti-breakout reading would
    contradict working strategy. The lesson is about asking WHO committed size,
    not about a default direction -- and the existing continuation branches must
    still stand.
    """
    prompt = build_system_prompt()
    # It frames the question, rather than prescribing a side.
    assert "who just committed size because of" in prompt
    # The continuation knowledge it must not override is still present.
    assert "OPENING DRIVE" in prompt
    assert "RUNAWAY" in prompt or "runaway" in prompt


def test_system_prompt_has_v4d_flat_open_and_round_number_booking_knowledge():
    """v4d (10 Aug live session): a large win on puts from a FLAT open.

    The opening type is reframed as a PARTICIPATION reading rather than a
    strength reading -- a gap runs because it denied everyone entry, a flat open
    cannot because it granted it. He then booked a big profit deliberately BEFORE
    the round number, because a three-index move is what recruits the late crowd
    and their targets all sit at the round figure.
    """
    prompt = build_system_prompt()
    assert "A FLAT OPEN CANNOT RUN THE WAY A GAP CAN" in prompt
    # Wrap-independent fragments: these span line breaks in the source.
    assert "PARTICIPATION reading" in prompt
    assert "BOOK BEFORE THE ROUND NUMBER" in prompt
    assert "everyone else's target IS" in prompt
    # The two new target inputs, and the tolerated-adverse-move band.
    assert "YOUR ENTRY PRICE IS THE FOURTH TARGET INPUT" in prompt
    assert "PRE-COMMIT THE ADVERSE MOVE YOUR THESIS TOLERATES" in prompt


def test_target_sizing_inputs_are_all_present_and_distinct():
    """The four target-sizing inputs accumulated across v4a-v4d must coexist.

    Each was added in a different version and they pull in different directions,
    so a later edit that dropped one would quietly change how every target is
    sized without failing any other test.
    """
    prompt = build_system_prompt()
    assert "A FRESHLY RECRUITED CROWD HAS TIGHT STOPS" in prompt        # v4a: recency
    assert "A CROWD THAT HAS AVERAGED DOWN EARNS A BIGGER TARGET" in prompt  # v4b
    assert "CROWD SIZE IS THE THIRD TARGET INPUT" in prompt             # v4c
    assert "YOUR ENTRY PRICE IS THE FOURTH TARGET INPUT" in prompt      # v4d
    # v4d's is the only one about the trader rather than the crowd.
    assert "a property of YOU" in prompt


def test_system_prompt_has_v4e_recruitment_and_losing_session_knowledge():
    """v4e (11 Aug live session): IH's LOSS, which is why it is worth encoding.

    He named the disqualifying fact himself -- no stops seated on either side --
    then traded a FORECAST of who would arrive, and the market simply kept
    selling. The session also refines v4d: a gap-down recruits POSITIONAL
    sellers, a flat open only INTRADAY ones, so the same-shaped trap is smaller
    and more perishable after a flat open.
    """
    prompt = build_system_prompt()
    assert "WHICH CROWD THE OPEN RECRUITS DECIDES HOW BIG THE TRAP IS" in prompt
    assert "A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE OF WHO IS SEATED" in prompt
    # v4e's A SHARP FIRST SLIDE BAITS was PRUNED in v4h: it shipped as a weak
    # prior with its own counter-example attached, and v4h's warning-move rule
    # covers the same ground with an observable mechanism. See the v4h addendum.
    assert "A SHARP FIRST SLIDE BAITS" not in prompt
    assert "NAME THE LAST POINT, NOT ONLY THE STOP" in prompt
    assert "DISCIPLINE IS ASYMMETRIC BETWEEN WINNERS AND LOSERS" in prompt
    # The recruitment distinction is the point; both halves must be present.
    assert "recruits POSITIONAL sellers" in prompt
    assert "recruits INTRADAY sellers only" in prompt


def test_v4e_empty_book_is_a_no_trade_not_a_forecasting_licence():
    """The v4e lesson must not be readable as "predict the crowd instead".

    The whole method rests on hunting inventory that already exists. If this
    rule ever drifted into permitting a trade built on who is LIKELY to arrive,
    it would license exactly the loss it was distilled from.
    """
    prompt = build_system_prompt()
    section = prompt[prompt.index("A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE"):]
    section = section[: section.index("\n- ")] if "\n- " in section else section
    assert "the correct output is HOLD" in section
    assert "not an invitation to forecast one into existence" in section
    # It must also reconcile with v4c rather than silently contradicting it.
    assert "MANUFACTURES MORE" in section


def test_v4h_entry_and_direction_and_the_loss_limit_pair_with_their_neighbours():
    """v4h (14 Aug live session): a win held through a full drawdown.

    Two of its rules sit directly on top of earlier ones and would be dangerous
    read alone, so both reconciliations are asserted here.
    """
    prompt = build_system_prompt()
    # The prose is hard-wrapped, so a phrase this test cares about can straddle a
    # line break. Collapse whitespace before asserting on wording -- otherwise the
    # test breaks on a re-wrap that changed nothing about what the agent reads.
    flat = " ".join(prompt.split())
    assert "A SEATED CROWD IS WARNED BEFORE IT IS HUNTED" in flat
    assert "ENTRY QUALITY AND DIRECTION ARE SEPARATE JUDGEMENTS" in flat
    assert "THE LOSS LIMIT IS A PERMISSION TO WAIT" in flat

    # The loss-limit rule must not read as "hold losers": it is bounded on BOTH
    # sides, and it has to name the v4e rule it refines rather than contradict.
    limit = prompt[prompt.index("THE LOSS LIMIT IS A PERMISSION TO WAIT"):]
    limit = " ".join((limit[: limit.index("\n- ")] if "\n- " in limit else limit).split())
    assert "DISCIPLINE IS ASYMMETRIC" in limit
    assert "not against it" in limit
    assert "forbids holding past it" in limit
    assert "forbids" in limit and "cutting inside it" in limit

    # Elapsed time is the one thing allowed to override that patience, and the
    # time rule has to say so or the two become contradictory advice.
    assert "EXPIRES THE PREMISE" in flat
    assert "retire the premise before price ever reaches the limit" in flat


def _flat_rule(prompt: str, heading: str) -> str:
    """Return one '- HEADING ...' bullet, whitespace-collapsed.

    Rule prose is hard-wrapped, so asserting on wording against the raw prompt
    breaks whenever a paragraph is re-flowed -- a change the agent never sees.

    Anchors on the BULLET ("\n- HEADING") in preference to a bare match, because
    rules cite each other by name: a plain `.index(heading)` can land on another
    rule's cross-reference and silently assert against the wrong prose. Falls
    back to the bare heading so a non-bullet heading still resolves.
    """
    bullet = "\n- " + heading
    start = prompt.find(bullet)
    body = prompt[start + 1:] if start != -1 else prompt[prompt.index(heading):]
    if "\n- " in body:
        body = body[: body.index("\n- ")]
    return " ".join(body.split())


def test_v4i_lagging_index_rule_governs_the_basket_and_keeps_its_per_leg_escape():
    """v4i (17 Aug live session): a WORKING three-index basket booked on divergence.

    This is the first rule that acts on the BankNIFTY mirror from the HOLDING
    side, so two things have to survive edits: it must not collapse into "exit
    whenever the indices disagree", and it must keep pointing at `exit_leg`
    rather than always taking the whole basket down.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE LAGGING INDEX DECIDES THE BASKET'S EXIT")

    # The mechanism: the laggard caps the basket, so a healthy NIFTY leg alone is
    # not proof the trade is still working.
    assert "nifty_leg_pnl" in rule
    assert "BOOK signal" in rule

    # The equal-lot mapping is the reason this transfers to our basket at all --
    # IH was deliberately size-weighted into BankNIFTY and we are not.
    assert "EQUAL-LOT" in rule
    assert "NOT equal-rupee" in rule

    # The per-leg escape hatch must stay, or the rule becomes strictly worse than
    # the capability the host already exposes.
    assert "exit_leg" in rule
    assert "exit BOTH" in rule


def test_v4i_double_bottom_rule_stays_scoped_to_countertrend_patterns():
    """The shake-out reading must not generalise into "ignore reversal patterns".

    Read carelessly this rule would cancel PATTERNS_AND_CONFIRMATION wholesale,
    so the scope qualifier and the falsifier are asserted explicitly.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A DOUBLE BOTTOM INSIDE AN ESTABLISHED DOWNTREND")

    assert "does NOT license ignoring reversal patterns generally" in rule
    assert "AGAINST an established, already-moving trend" in rule
    # The falsifier: a real reversal produces its own momentum.
    assert "actually recruits buyers goes UP" in rule
    # And it must not become a licence to enter counter-trend.
    assert "it is not an entry against the trend" in rule


def test_v4i_crowd_fear_rule_pairs_with_the_agent_fear_rule_and_the_time_rule():
    """THEIR fear vs YOUR fear -- the two must not read as one contradictory rule."""
    prompt = build_system_prompt()
    flat = " ".join(prompt.split())
    rule = _flat_rule(prompt, "THE CROWD'S FEAR IS YOUR WINDOW")

    assert "FEAR IS NOT A SIGNAL" in flat
    assert "FEAR IS NOT A SIGNAL above" in rule  # names the rule it sits beside
    assert "governs YOUR fear" in rule and "reads THEIRS" in rule

    # It explains v4h's time rule rather than competing with it.
    assert "TIME EXPIRES THE PREMISE" in rule
    # And it is not an entry licence.
    assert "does NOT license entering merely because a move is fast" in rule


def test_system_prompt_has_v4f_repeat_chart_and_confirmation_knowledge():
    """v4f (12 Aug live session): a WIN taken from the same empty book that lost.

    The session repeated the previous day's shape exactly -- flat open, immediate
    drop -- and IH read that SAMENESS as the tell that a trap was forming rather
    than a continuation. He then waited for the recovery to actually start before
    entering, which is the whole difference from the 11 Aug loss.
    """
    prompt = build_system_prompt()
    assert "THE CHART DOES NOT REPEAT TWO DAYS RUNNING" in prompt
    assert "A MOVE THAT DENIED YOU ENTRY WAS NOT YOUR MOVE" in prompt
    assert "AN EMPTY BOOK MEANS A TRAP IS COMING" in prompt
    assert "THE SHARPEST RECOVERY NAMES THE LEADING INDEX" in prompt
    assert "BOOK WHEN THE PROFIT STOPS GROWING" in prompt


def test_v4f_confirmation_rule_does_not_reopen_the_v4e_forecasting_hole():
    """The two rules are a matched pair and must stay one.

    v4e says an empty book is a no-trade; v4f says an empty book means a trap is
    coming. Read alone, v4f would license exactly the forecast v4e forbids. The
    reconciliation is CONFIRMATION IN PRICE, and both halves have to survive
    together or the pair becomes permission to guess.
    """
    prompt = build_system_prompt()
    section = prompt[prompt.index("AN EMPTY BOOK MEANS A TRAP IS COMING"):]
    section = section[: section.index("\n- ")] if "\n- " in section else section

    # It must demand confirmation, not merely a hypothesis...
    assert "CONFIRMATION IN PRICE" in section
    # ...and it must name what it does NOT tell you.
    assert "which side the trap is aimed at" in section
    # The v4e rule it reconciles with must still be present and still say HOLD.
    assert "A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE OF WHO IS SEATED" in prompt
    forecast = prompt[prompt.index("A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE"):]
    forecast = forecast[: forecast.index("\n- ")] if "\n- " in forecast else forecast
    assert "the correct output is HOLD" in forecast


def test_v4f_exit_rule_stays_on_the_winning_side_only():
    """"Book when profit stops growing" must never become "cut a loser early".

    It is a profit-taking rule. If it drifted into the loss branch it would
    contradict DISCIPLINE IS ASYMMETRIC, which puts the patience on winners and
    the mechanical exit on losers.
    """
    prompt = build_system_prompt()
    section = prompt[prompt.index("BOOK WHEN THE PROFIT STOPS GROWING"):]
    section = section[: section.index("\n- ")] if "\n- " in section else section
    assert "not a price level and not a" in section  # "...not a loss."
    assert "already captured one momentum" in section.lower() or "captured one momentum" in section
    # The asymmetry rule it builds on must still be there.
    assert "DISCIPLINE IS ASYMMETRIC BETWEEN WINNERS AND LOSERS" in prompt


def test_system_prompt_has_v4g_stop_hunt_completion_and_discipline_knowledge():
    """v4g (13 Aug live session): a reduced win, and the richest discipline block.

    Two structural ideas -- a completed stop-hunt marks the END of that
    direction, and the round number is the declared invalidation rather than
    only a target -- plus the psychology that guards the v4f exit rule.
    """
    prompt = build_system_prompt()
    assert "A COMPLETED STOP-HUNT ENDS THAT DIRECTION" in prompt
    assert "THE ROUND NUMBER IS WHERE THE THESIS DIES" in prompt
    assert "FEAR IS NOT A SIGNAL" in prompt
    assert "TIME SPENT IN THE TRADE SHRINKS THE ACHIEVABLE TARGET" in prompt
    assert "NEVER EXIT AT ZERO AFTER A GOOD PROFIT HAS PRINTED" in prompt


def test_v4g_fear_rule_and_v4f_book_rule_do_not_cancel_each_other():
    """The most dangerous pair in the whole prompt, and the reason for this test.

    v4f says BOOK WHEN THE PROFIT STOPS GROWING. v4g says never exit because you
    are afraid it might turn. Read carelessly the second reads as "hold through
    everything" and would disable the first; read carelessly the first licenses
    exactly the fear-driven early exit the second forbids. The distinction is
    MEASUREMENT versus EMOTION, and both halves have to say so explicitly.
    """
    prompt = build_system_prompt()

    fear = prompt[prompt.index("FEAR IS NOT A SIGNAL"):]
    fear = fear[: fear.index("\n- ")] if "\n- " in fear else fear
    # It must name the measurement rule it is guarding, not contradict it...
    assert "BOOK WHEN THE PROFIT STOPS GROWING" in fear
    assert "MEASUREMENT" in fear
    assert "EMOTION" in fear
    # ...and it must require a checkable reason rather than banning exits.
    assert "NAMED, checkable reason" in fear
    for allowed in ("stop", "target", "premise invalidated", "time cutoff"):
        assert allowed in fear, allowed

    # The zero-zero floor must not become a second fear-driven exit: it triggers
    # on an observed fact, and only then sets the floor.
    floor = prompt[prompt.index("NEVER EXIT AT ZERO AFTER A GOOD PROFIT HAS PRINTED"):]
    floor = floor[: floor.index("\n- ")] if "\n- " in floor else floor
    assert "NOT the fear rule above in disguise" in floor
    assert "the move has stopped working" in floor


def test_v4g_stop_hunt_rule_does_not_become_always_fade_the_bounce():
    """The completed-stop-hunt rule needs its escape hatch intact.

    "Follow the original direction after a clearing retracement" is powerful and
    would be dangerous as an unconditional rule, so the gap exception that voids
    it must survive any later edit.
    """
    prompt = build_system_prompt()
    section = prompt[prompt.index("A COMPLETED STOP-HUNT ENDS THAT DIRECTION"):]
    section = section[: section.index("\n- ")] if "\n- " in section else section
    assert "SPENT its fuel" in section
    assert "do NOT" in section and "chase the retracement" in section
    # The voiding condition.
    assert "fresh large gap" in section
    assert "recruits a new crowd" in section


def test_reentry_gate_does_not_contradict_the_exit_rules():
    """The re-entry gate must never be readable as a reason to delay an EXIT.

    Regression guard: the gate sits inside RISK next to the exit rules, so it has to
    state its entries-only scope in the same breath as the mechanical exit paths.
    """
    prompt = build_system_prompt()
    gate = prompt[prompt.index("POST-EXIT RE-ENTRY GATE"):]
    gate = gate[: gate.index("\n- ")] if "\n- " in gate else gate
    assert "Exits are never delayed by it" in gate
    assert "square-off" in gate


def test_system_prompt_has_v3r_profit_booking_recovery_and_lagging_index_knowledge():
    """v3r: 24 Jul - classify the recovery first, then time entry from the laggard.

    A first bounce after a paid multi-day selloff is not automatically a seller-hunt
    long. Once the short day-direction premise is established, a lagging index may
    locate the entry, but it may not invent the direction.
    """
    prompt = build_system_prompt()
    assert "PROFIT-BOOKING RECOVERY TEST" in prompt
    assert "first bounce alone" in prompt and "not a LONG" in prompt
    assert "LAGGING-INDEX ENTRY LOCATOR" in prompt
    assert "entry-timing cue only" in prompt
    # v3r scopes the existing rules; it does not replace or weaken them.
    assert "TARGET-BOOKED crowd test" in prompt
    assert "GAP-DOWN CONTINUATION SHORT" in prompt
    assert "MASKED BNF LAG" in prompt


def test_runaway_trend_section_is_composed_into_the_prompt():
    """The section constant must actually be wired into build_system_prompt()."""
    from sl_hunting_knowledge import RUNAWAY_TREND

    prompt = build_system_prompt()
    assert RUNAWAY_TREND.strip() in prompt
    # It belongs with the other continuation exception, before the levels rules.
    assert prompt.index("RUNAWAY TREND —") > prompt.index("OPENING DRIVE —")


def test_v4j_call_writer_rule_stays_an_exit_read_and_not_a_reversal_trade():
    """v4j (18 Aug live session): IH's losing expiry-day trade, diagnosed.

    The dangerous misreading is obvious and expensive: "writers are seated
    above" sounds like a short signal. It is not -- their interest is a RANGE,
    so flipping short expects a collapse the mechanism argues against. The
    stand-down and the both-directions signature are what make it usable.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A SHARP SPIKE THAT IMMEDIATELY STALLS")

    # The signature is dead momentum BOTH ways, not a turn against you.
    assert "Momentum dies in BOTH directions" in rule
    assert "NOT a reversal" in rule
    # The trap this rule must never become.
    assert "do not flip short" in rule
    assert "RANGE, not a collapse" in rule
    # Why an option BUYER specifically must leave rather than sit.
    assert "theta runs" in rule
    assert "EXIT read" in rule


def test_v4j_per_leg_cut_rule_does_not_cancel_the_v4i_per_leg_escape():
    """A BankNIFTY reversal is a basket verdict; a BankNIFTY-only problem is not.

    Read carelessly this rule deletes `exit_leg` entirely, which would undo
    v4i and throw away a real host capability. The idiosyncratic-vs-turned
    distinction is the whole rule, so it is asserted directly -- along with the
    measured evidence, since the numbers are what make it more than an opinion.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CUTTING THE MIRROR ON A BANKNIFTY REVERSAL")

    # The distinction that keeps v4i's escape hatch alive.
    assert "IDIOSYNCRATIC" in rule
    assert "is this BankNIFTY-only" in rule
    assert "EXIT BOTH" in rule
    # v4i's rule must still be present and still offer the per-leg cut.
    lagging = _flat_rule(prompt, "THE LAGGING INDEX DECIDES THE BASKET'S EXIT")
    assert "exit_leg" in lagging

    # The measured 2026-08-18 sequence: +303.00 saved on the mirror, -435.50
    # lost on the leg held six minutes longer on information already in hand.
    assert "+303.00" in rule and "-435.50" in rule
    assert "only the conclusion was late" in rule

    # An opposing cross-index verdict on an OPEN position removes benefit of the
    # doubt without becoming an automatic exit -- both halves must survive.
    assert "It does not by itself force an exit while the premise" in rule
    assert "REMOVES the benefit of the doubt" in rule


def test_v4j_time_to_profit_rule_bounds_v4h_without_becoming_cut_everything():
    """The third bound on THE LOSS LIMIT IS A PERMISSION TO WAIT.

    Without it v4h reads as "sit until the stop"; with it read carelessly it
    reads as "cut anything slow". The rule survives only if it keeps naming
    what the test is NOT (the stop, and fear) alongside what it is.
    """
    prompt = build_system_prompt()
    flat = " ".join(prompt.split())
    rule = _flat_rule(prompt, "THE PERMISSION TO WAIT ENDS WHEN THE TIME TO PROFIT")

    # It must sit with, and name, the rule it bounds.
    assert "THE LOSS LIMIT IS A PERMISSION TO WAIT" in flat
    assert "loss-limit rule below" in rule

    # The counter-intuitive half: being right about direction is not enough.
    assert '"not falling" is not "rising"' in rule
    # It is neither the stop nor fear -- both exclusions must stay.
    assert "not the stop and not fear" in rule
    # And the concrete test that replaces them.
    assert "in the time you have" in rule


def test_post_loss_speed_limit_keeps_both_halves_and_the_name_others_cite():
    """The post-loss protocol was two overlapping rules; it is now one.

    Two things could regress silently. NO INSTANT FLIP defers to this rule BY
    NAME, so a rename leaves a dangling pointer that no import or type check
    would catch. And the merge folded in the recovery half -- spread a big loss
    over several trades, distrust the day's "one last trade" -- which is the
    part a future tightening pass would drop first, because it reads like a
    restatement of the cooldown when it is actually about position sizing of
    the recovery.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "POST-LOSS SPEED LIMIT")

    # The name other rules cite.
    assert "POST-LOSS SPEED LIMIT then governs" in " ".join(prompt.split())

    # Half one: no fast re-entry.
    assert "quick-decision mode is disabled" in rule
    assert "never use the next candle as a recovery attempt" in rule
    # Half two: how a big loss is recovered, and the end-of-day trap.
    assert "MULTIPLE ordinary trades, never in one" in rule
    assert "one last trade" in rule


def test_printed_profit_obligation_is_owned_by_exactly_one_rule():
    """v4f states the exit TRIGGER; v4g owns what a printed profit obliges.

    Both rules stay -- they are a trigger and a floor -- but the "a target you
    saw counts as reached" idea was written into both, so a reader met it twice
    and neither rule owned it. v4f now points at v4g instead.
    """
    prompt = build_system_prompt()
    book = _flat_rule(prompt, "BOOK WHEN THE PROFIT STOPS GROWING")
    floor = _flat_rule(prompt, "NEVER EXIT AT ZERO AFTER A GOOD PROFIT HAS PRINTED")

    # v4f keeps the trigger and defers the floor.
    assert "the RATE at which the position is still gaining" in book
    assert "NEVER EXIT AT ZERO's job, not this rule's" in book
    assert "ALREADY SEEN a good target" not in book

    # v4g still owns it.
    assert "the floor for that trade stops being breakeven" in floor


def test_v4k_fast_move_rule_extends_momentum_quality_without_contradicting_v3y():
    """v4k (19 Aug live session): speed in your favour is a clock, not a signal.

    Two ways this decays. It can lose the MECHANISM and collapse back into the
    retracement-risk line it extends, which is a weaker and different claim. And
    it can be read against v3y's A SLOW GRIND AT THE LEVEL, which says slowness
    is BAD -- that rule is about price stalling before entry, this one about pace
    after entry, and the prose has to keep saying so.
    """
    prompt = build_system_prompt()
    flat = " ".join(prompt.split())

    # The mechanism: the trapped crowd is what pays, so speed decides duration.
    assert "WHY SLOW IS BETTER, AND IT IS NOT ONLY RETRACEMENT RISK" in flat
    assert "flushes the whole crowd in one burst" in flat
    assert "is a shorter clock" in flat
    # It must not read as "fast means it is working".
    assert "NOT extra confirmation" in flat

    # The reconciliation with v3y must survive verbatim enough to be findable.
    assert "A SLOW GRIND AT THE LEVEL RECRUITS THE WRONG CROWD" in flat
    assert "BEFORE you are in" in flat and "AFTER you are in" in flat


def test_v4k_only_one_index_paying_rule_is_direction_agnostic():
    """v4i named the mirror lagging; v4k says it works whichever index lags.

    The asymmetric reading is the failure mode: an agent that only checks
    "is the mirror lagging?" misses the mirror carrying a basket whose NIFTY leg
    has stalled, which is the same message.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE LAGGING INDEX DECIDES THE BASKET'S EXIT")

    assert "IT IS DIRECTION-AGNOSTIC" in rule
    assert "whichever index that is" in rule
    # Both readings spelled out, so neither can be dropped as redundant.
    assert "the mirror lagging while NIFTY runs" in rule
    assert "NIFTY lagging while" in rule

    # The practical instruction, and the measurement that earned it.
    assert "quote the BASKET number" in rule
    assert "-519" in rule and "-467" in rule


def test_v4k_visible_support_rule_keeps_both_of_its_opposite_consequences():
    """A held support in a three-index sell-off: bad EVIDENCE, good PLACE.

    The two consequences point opposite ways, which is exactly why a tightening
    pass would drop one. Losing the first turns it into a seated-buyer read
    (the error it exists to prevent); losing the second turns a usable entry
    location into a no-go zone.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A SUPPORT EVERYONE CAN SEE RECRUITS NOBODY")

    # Worthless as evidence...
    assert "As EVIDENCE it is worthless" in rule
    assert "NOT a seated-buyer read" in rule
    # ...but clean as a place.
    assert "As a PLACE it is unusually clean" in rule
    assert "nobody is queued there competing with you" in rule
    # And the caveat that stops it becoming "the level is safe".
    assert "safe FROM competition, not where" in rule


def test_v4l_cut_every_leg_raises_the_bar_on_exit_leg_without_removing_it():
    """v4l (20 Aug live session): EXIT BOTH is the default, not a coin flip.

    The danger runs both ways. Read strongly the rule deletes `exit_leg` and
    throws away a host capability v4i added deliberately; read weakly it leaves
    the per-leg cut as an equal option, which is the failure IH describes. The
    prose has to keep BOTH the default and the narrow exception.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CUTTING THE MIRROR ON A BANKNIFTY REVERSAL")

    assert "WHEN YOU CUT, CUT EVERY LEG" in rule
    # The mechanism -- why half-closing is punished, not merely suboptimal.
    assert "a loss on BOTH sides" in rule
    # The default/exception asymmetry, in both halves.
    assert "EXIT BOTH is the default" in rule
    assert "`exit_leg` is the exception" in rule
    assert "independently intact" in rule
    assert "never merely because the other leg is the one currently hurting" in rule

    # The structural honesty: his legs hedge, ours do not.
    assert "his legs are a HEDGE" in rule
    assert "The arithmetic does not transfer, but the failure does" in rule


def test_v4l_divergence_is_judged_on_movement_not_on_rupee_share():
    """Corrects v4k the day after it shipped, using this book's own numbers.

    The equal-lot mirror makes BankNIFTY carry the larger rupee share by
    construction, so "the mirror is making most of the money" is the NORMAL
    state. An agent applying v4k to a rupee split books working trades early.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE LAGGING INDEX DECIDES THE BASKET'S EXIT")

    assert "JUDGE IT ON MOVEMENT, NOT ON RUPEE SHARE" in rule
    # Both measured cases must stay: the ordinary split and the real divergence.
    assert "+461.50 NIFTY / +1,092.00 mirror" in rule
    assert "arithmetic, not" in rule
    assert "0.15 premium points" in rule and "20.15" in rule
    # The test itself, and the cost of getting it wrong.
    assert "stopped MOVING" in rule
    assert "books working trades early" in rule


def test_v4l_entry_time_rule_is_a_tradeoff_not_a_preference_for_waiting():
    """It must not collapse into "trade later, it is safer".

    IH's claim is that the two move TOGETHER -- a later entry buys a slower
    tape, not a free lunch -- and that the open is not automatically the best
    place to be. Both halves are needed or the rule becomes a bias.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "ENTRY TIME IS A RISK DIAL")

    assert "in proportion to the risk you take" in rule
    # Neither end is presented as simply better.
    assert "not simply a better entry" in rule
    assert "not simply a safer one" in rule
    # The opening window is not privileged...
    assert "as the only place a trade exists" in rule
    assert "a trade made right at the open will be better" in rule
    # ...and an opening entry changes the TARGET, not just the stop.
    assert "size the EXPECTATION as well as the" in rule
    # Kept distinct from the rule it sits beside.
    assert "MORNING SPEED IS NOT INFORMATION" in rule


def test_v4m_closing_price_rule_gains_the_all_or_nothing_cross_index_property():
    """v4m (21 Aug live session): IH lost on the same side our agent did.

    The existing rule already said the breakdown is the trigger. What today
    adds is that it is BINARY ACROSS INDICES -- so the check is cheap -- and
    that the failure mode while it holds is CHOP rather than a clean loss. The
    early-entry trap is asserted too, because it is the error IH made and the
    one an opening rejection most invites.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CLOSING-PRICE BREAKDOWN IS THE TRIGGER")

    assert "ALL-OR-NOTHING ACROSS THE INDICES" in rule
    assert "it will break in all three" in rule
    assert "all three will sit and hold" in rule
    # The regime it leaves you in, which is the part that costs an option buyer.
    assert "it is CHOP" in rule
    # And the trap: a rejection at the open is not the breakdown.
    assert "not a substitute for it breaking" in rule


def test_v4m_loss_limit_permission_is_regime_conditional_without_licensing_feel():
    """The third bound on v4h, alongside the clock.

    Read carelessly this becomes "cut early when it feels slow", which is the
    exact error v4h exists to prevent. The prose has to keep naming an
    OBSERVABLE, so the assertion checks the reconciliation survives.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE LOSS LIMIT IS A PERMISSION TO WAIT")

    assert "THE PERMISSION IS REGIME-CONDITIONAL" in rule
    assert "a market that still MOVES" in rule
    # Both regimes must stay, or it collapses into a bias one way.
    assert "the patience above stands as" in rule
    assert "stalling into sideways" in rule
    # It must not license cutting on feel -- the whole point of v4h.
    assert "none of them licenses cutting on" in rule
    assert "names an observable that has changed" in rule


def test_v4m_disqualifying_fact_carries_into_the_next_entry():
    """Four shorts, each killed by the leading index, each re-entered fresh.

    The rule must stay about the LEADING index specifically -- a generic "do
    not re-enter" duplicates the post-exit gate below it -- and must keep the
    measured evidence, which is what makes it more than an opinion.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE FACT THAT DISQUALIFIED YOUR LAST TRADE")

    assert "FOUR shorts" in rule and "51 minutes" in rule
    assert "-2,300.75" in rule
    # The specific bar it raises, and on what.
    assert "it needs BankNIFTY itself to stop opposing" in rule
    assert "the honest output is HOLD" in rule


def test_v4m_stale_hatch_may_not_be_used_selectively():
    """The companion failure: re-rating a signal to fit the chosen action.

    This must NOT read as "the stale hatch is closed" -- v3 scoped it
    deliberately and it is still valid inside the opening hour. What is banned
    is choosing staleness by whether the verdict agrees.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "YOU MAY NOT RE-RATE A SIGNAL")

    # The hatch survives...
    assert "that hatch is real" in rule
    # ...but the basis for using it is fixed.
    assert "never from whether it agrees with you" in rule
    # Both measured occurrences, so a later edit cannot reduce it to one anecdote.
    assert "at 10:05" in rule and "at 10:12" in rule
    assert "09:18 and 09:27" in rule
    # The check the agent can actually run.
    assert "would I still call it stale if it" in rule
    assert "it is inconvenient" in rule


def test_v4n_do_not_think_through_a_loss_does_not_become_stop_observing():
    """v4n (24 Aug live session): the thinking is what enlarges the loss.

    This is the rule most likely to be mis-drafted into something dangerous.
    Read too strongly it says ignore the chart while losing, which would
    disable stops and named invalidations. It must keep BOTH the ban on
    searching for a rescuing reading AND the licence to keep observing.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "DO NOT THINK YOUR WAY THROUGH A LOSS")

    assert "APPLYING THOUGHT WHILE IN A LOSS IS THE THING THAT MAKES THE LOSS BIGGER" in rule
    # The mechanism, aimed at what an LLM does by construction.
    assert "generating another plausible reading of the chart is" in rule
    assert "evidence about your discomfort" in rule
    # Only pre-committed questions stay live.
    assert "you had ALREADY committed to" in rule
    # ...and the guard against over-reading it.
    assert "It does NOT mean stop observing" in rule
    assert "stop SEARCHING for a reading that rescues the position" in rule


def test_v4n_distant_stop_crowd_rule_keeps_both_removal_mechanisms():
    """Not every seated crowd is huntable; some are faded out instead.

    The rule is only useful if it changes the ENTRY TRIGGER, so the
    break-versus-fade distinction is asserted rather than the colour.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A CROWD WHOSE STOPS ARE TOO FAR AWAY")

    assert "first let them SEE some profit, then reduce that profit" in rule
    # A move their way is not evidence they were right.
    assert "not evidence they were right" in rule
    # The operational half: which trigger applies.
    assert "waits for the BREAK" in rule and "waits for the FADE" in rule
    assert "a break that this crowd was never going to produce" in rule


def test_v4n_re_rating_rule_gains_a_procedure_after_being_broken():
    """v4m's rule was violated 48 seconds apart on its first live session.

    The addition is mechanical on purpose -- a disposition ("be consistent")
    had already failed, so what is asserted here is the CHECK, and the
    measured evidence that motivated it.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "YOU MAY NOT RE-RATE A SIGNAL")

    # The original test survives.
    assert "would I still call it stale if it" in rule
    # The new procedure, and the fact that it is a procedure.
    assert "STATE WHAT IT READ ON YOUR PREVIOUS DECISION" in rule
    assert '"flipped" is not available to you' in rule
    # The measured case, including the detail that makes it damning.
    assert "It had not flipped" in rule
    assert "forty-eight seconds later" in rule
    # Overruling then citing the same verdict is the banned combination.
    assert "citing it as support now is not permitted" in rule


def test_v4o_method_is_not_always_a_fade_reconciles_with_v4e_and_v4f():
    """v4o (25 Aug live session): hunting is the best trade, not the only one.

    This is the most dangerous addition in the series so far, because read
    loosely it licenses entering with no crowd read at all -- which is the
    exact loss v4e was written from. It survives only while it keeps BOTH
    reconciliations and the honesty requirement about which trade you are in.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE METHOD IS NOT ALWAYS A FADE")

    assert "That is NOT so" in rule
    assert "If the sellers' stop-losses are not" in rule
    # It must name BOTH rules that appear to forbid it, and answer each.
    assert "A FORECAST OF WHO WILL ARRIVE" in rule
    assert "it claims no crowd at all" in rule
    assert "AN EMPTY BOOK MEANS A TRAP IS COMING" in rule
    assert "already ESTABLISHED" in rule
    # The management difference is the operational half.
    assert "no huntable inventory, following the move" in rule
    assert "a hunt is over when the crowd is flushed" in rule


def test_v4o_level_not_direction_bound_does_not_gut_the_index_hierarchy():
    """A leading index moving against you is not your invalidation breaking.

    Read too strongly this cancels the index-hierarchy disqualification, which
    has genuinely closed losing baskets. It is admissible ONLY when a level was
    named in advance, so the no-level fallback is asserted explicitly.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CUTTING THE MIRROR ON A BANKNIFTY REVERSAL")

    # The original disqualification survives intact.
    assert "the honest action is EXIT BOTH" in rule
    # The bound, and what actually distinguishes the two readings.
    assert "MOVING AGAINST YOU IS NOT THE SAME AS YOUR LEVEL BREAKING" in rule
    assert "not the direction of the last few candles" in rule
    # Both same-session observations are kept, and neither is claimed as proof.
    assert "-1,176.25" in rule
    assert "it is proof the two readings are DIFFERENT" in rule
    # The fallback that stops this becoming a general excuse.
    assert "If you cannot name the level, you do not have this" in rule


def test_v4o_double_expiry_rule_answers_with_timing_not_direction():
    """Two expiries on one day is a TIMING instruction, not a bias."""
    prompt = build_system_prompt()
    flat = " ".join(prompt.split())

    assert "TWO EXPIRIES AT ONCE MULTIPLY THE TRAPS" in flat
    assert "one selling momentum, then one buying momentum" in flat
    assert "The response is timing, not direction" in flat
    # The specific misreading it prevents.
    assert "expect the small counter-trap before the move rather than reading it as invalidation" in flat


def test_v4p_enter_quickly_rule_never_relaxes_the_confirmation_requirement():
    """v4p (26 Aug live session): "the more you wait, the more you lose".

    This is the rule most likely to be misread into skipping confirmation,
    which would gut the method. It must keep BOTH the scope limit (it governs
    WHERE you enter, not WHETHER a setup was needed) and the conditional -- he
    waited deliberately two sessions earlier and the prose has to say so, or
    "enter quickly" becomes a standing bias.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "WHEN THE DIRECTION HAS ALREADY DECLARED ITSELF")

    assert "THE MORE YOU WAIT, THE MORE YOU LOSE" in rule
    assert "we bought into a RUNNING market" in rule

    # The scope limit -- the half that keeps the method intact.
    assert "WHAT THIS DOES NOT RELAX" in rule
    assert "pattern-plus-confirmation requirement is" in rule
    assert "it never means enter without one" in rule

    # Both branches of the conditional, so it cannot collapse into a bias.
    assert "Direction already established at the open" in rule
    assert "waiting is right" in rule

    # And the reconciliation with the rule it sits beside.
    assert "ENTRY TIME IS A RISK DIAL" in rule


def test_v4p_stop_is_a_nifty_trigger_note_does_not_argue_for_wider_stops():
    """The stop watches one index but closes two legs.

    Stated as a FACT about the basket, with the tempting wrong conclusion
    ruled out explicitly -- a rule that quietly licensed wider stops on a
    live-money system would be the worst possible reading of it.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "BASKET NOTE (BankNIFTY mirror)")

    assert "YOUR STOP IS A NIFTY-SPOT TRIGGER ON A TWO-INDEX BASKET" in rule
    # The measured case, both legs.
    assert "-399.75" in rule and "+1,242.00" in rule
    assert "the basket exited +842.25" in rule
    # The conclusion it must NOT support.
    assert "not a reason to widen the stop" in rule
    assert "protects the NIFTY premise" in rule


def test_v4q_completed_stop_hunt_test_also_gates_the_entry():
    """v4q (27 Aug live session): the rule reads as an exit, but it disqualifies entries.

    Measured the same morning: the agent cited "the completed stop-hunt has
    spent its fuel" to EXIT trade 2 at 09:54 -- two minutes after opening it
    with a target (24188.6) the session had already tagged at 09:32. It had
    the rule and ran it in only one of the two places that matter.

    The already-printed-target form is what makes it mechanical rather than a
    judgement call, so it must survive verbatim.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A COMPLETED STOP-HUNT ENDS THAT DIRECTION")

    assert "RUN THIS TEST BEFORE THE ENTRY, NOT ONLY AS AN EXIT CHECK" in rule

    # The mechanical form. Without this the extension is just an exhortation.
    assert (
        "the level you are about to name as the target has ALREADY printed in "
        "this session" in rule
    )
    assert "the move you are entering has already happened" in rule

    # Why a bounce toward the entry is not a fresh trap.
    assert "the completed hunt TURNING" in rule
    assert "require a NEW trap rather than the exhausted one" in rule


def test_v4q_obviousness_exit_never_becomes_a_licence_to_sit():
    """v4q: an exit trigger keyed on who else can now see the trade.

    Two ways this rule could go wrong, both asserted against:

    1. It could be read as permission to HOLD (it arrives from a session where
       IH held through a pullback), which on live money is the dangerous
       direction. The prose must keep v4f, the stop, the max loss and
       premise-invalidation ranked above it.
    2. It could lose the reason -- that the edge was being positioned before
       the crowd -- and decay into "exit when the move is big", which is a
       price rule, not a participation one.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE HOLD IS LICENSED BY THE EARLY IMPULSE")

    # It is about participation, not size or rate.
    assert "WHO ELSE CAN NOW SEE THE TRADE" in rule
    assert (
        "before other people apply their minds, apply yours first, book the "
        "profit and get out" in rule
    )
    assert "buyers of your position, not fuel for it" in rule

    # Usable as a test at the moment of decision.
    assert "a newcomer would now enter here" in rule

    # And it must never outrank the rules that protect the account.
    assert "It does NOT license sitting through a stall" in rule
    assert "v4f still books when the rate of gain dies" in rule
    assert "the stop, the max loss and premise-invalidation all outrank it" in rule


def test_stall_or_reversal_discriminator_keeps_the_two_basket_rules_consistent():
    """SLH-015: the two basket rules used to give OPPOSITE defaults for `exit_leg`.

    `THE LAGGING INDEX DECIDES THE BASKET'S EXIT` said "Prefer that to a
    whole-basket exit WHEN the NIFTY premise is genuinely intact", while
    `CUTTING THE MIRROR ON A BANKNIFTY REVERSAL` said "EXIT BOTH is the default
    and `exit_leg` is the exception, not a pair of equal options" -- and the
    tool itself defaults to BOTH in code. The agent read both and could justify
    either, on live-money exits.

    They were always reconcilable: a mirror that has STALLED is a question about
    how much the basket can still collect (per-leg available), a mirror that has
    REVERSED is a question about whether the move is still on (EXIT BOTH). That
    discriminator was simply never stated. Neither rule was changed in substance
    -- all fifteen v4i/v4j/v4k/v4l/v4o assertions still pass untouched -- the
    per-leg preference was SCOPED to the case it was always about.
    """
    prompt = build_system_prompt()
    lagging = _flat_rule(prompt, "THE LAGGING INDEX DECIDES THE BASKET'S EXIT")
    cutting = _flat_rule(prompt, "CUTTING THE MIRROR ON A BANKNIFTY REVERSAL")

    # Stated exactly ONCE, so the two rules cannot drift apart again.
    assert prompt.count("THE STALL-OR-REVERSAL TEST") == 1
    assert "THE STALL-OR-REVERSAL TEST" in cutting
    assert "the discriminator both basket rules turn on, stated here once" in cutting

    # Both halves of the test, and why it cannot be read off a P&L number.
    assert "a mirror that has STALLED" in cutting
    assert "a question about whether the MOVE IS STILL ON" in cutting
    assert "identical on a P&L screen and completely different on a chart" in cutting

    # The lagging rule owns the STALL case, says so, and defers rather than
    # restating a competing default.
    assert "available HERE, in the STALL case, and only here" in lagging
    assert "NOT the licence for a mirror that has TURNED" in lagging
    assert "see the stall-or-reversal test in the rule below" in lagging
    assert "EXIT BOTH is the default and per-leg is the narrow exception" in lagging

    # The regression itself: the unscoped preference must never come back.
    assert "Prefer that to a whole-basket exit" not in prompt

    # And the reversal case still carries the stricter default it always had.
    assert "EXIT BOTH is the default" in cutting
    assert "the honest action is EXIT BOTH" in cutting


def test_v4r_laggards_never_joined_is_scoped_to_the_majority_refusing():
    """v4r (28 Aug live session): the rule is about the MAJORITY, not any one index.

    `LAGGARDS NEVER JOINED` is written for "the other TWO indices never break
    their own levels". On a two-index basket that phrasing is easy to invert:
    one index trailing reads as "the laggards never joined" when it is nothing
    of the sort.

    Measured the same morning: a basket up +968.75 was closed 100 seconds after
    entry, citing a mirror that was "flat (-22.5), a textbook laggard-never-joined
    signal". Flat is the not-yet-arrived case, which is what this scope limit
    exists to separate from the refusing case.

    Three things a summarising edit would flatten, each asserted:

    1. The COUNT is the gate -- majority absent, not any single index behind.
    2. The classification test is the trailing index's SIGN, not its distance.
       Flat or positive-but-slower has not refused; negative or a confirmed
       reversal is the disqualifying case.
    3. It hands the disqualifying case to THE STALL-OR-REVERSAL TEST rather than
       ruling on it here, so the two cannot drift into competing verdicts the way
       the basket pair did before SLH-015.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "LAGGARDS NEVER JOINED")

    # 1. The majority gate.
    assert "COUNT WHO IS ABSENT BEFORE YOU APPLY THIS" in rule
    assert "written for the MAJORITY refusing" in rule
    assert "NOT a rule about any single index being behind" in rule

    # 2. Sign, not distance -- the part that decides the action.
    assert "the test is its SIGN, not its distance" in rule
    assert "FLAT or positive-but-slower has not refused, it has not arrived yet" in rule

    # 3. It defers the disqualifying case rather than ruling on it.
    assert "THE STALL-OR-REVERSAL TEST in the basket rules, not this one" in rule

    # The mechanism, which is what makes a trailing index readable at all.
    assert "just to CHANGE" in rule and "PSYCHOLOGY" in rule
    assert "BankNIFTY just needs to not stay negative" in rule

    # The measured case, including the honest half that does NOT support the rule.
    assert "+968.75" in rule and "flat (-22.5)" in rule
    assert "-579.75" in rule
    assert "That is not proof holding is" in rule
    assert "which is a fair v4f book on its own" in rule


def test_v4s_accuracy_method_rule_limits_trade_COUNT_not_holding_time():
    """v4s (30 Aug knowledge video): the two setup families need opposite tuning.

    IH: pattern setups are inherently low-accuracy and are paid for by RATIO, so
    they need a high R:R and more attempts; price-action setups are paid for by
    being RIGHT, so they need accuracy and FEW trades. SL hunting is the second
    family, and "the biggest mistake a price-action trader makes is not
    controlling the number of trades".

    Two ways this rule could go wrong, both asserted against:

    1. It could be read as "hold longer", which is the dangerous direction on a
       live book and would contradict v4f and v4q. The prose must keep saying it
       governs how many times you ENTER, not when you leave.
    2. It could decay into "take fewer trades" with the reason removed, at which
       point it is a slogan rather than a test. The measured evidence -- two
       consecutive sessions where the second trade gave back about a third of
       the first -- is what makes it checkable.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "YOURS IS AN ACCURACY METHOD")

    # The two families, and which metric each is paid by.
    assert "PATTERN setups" in rule and "PRICE-ACTION setups" in rule
    assert "SL HUNTING IS THE SECOND FAMILY" in rule
    assert "what was the point of working on price" in rule

    # Trade count as a risk control, with IH's arithmetic for ignoring it.
    assert "not controlling the number of trades" in rule
    assert "you will hand back as an equal loss" in rule

    # The measured evidence, both sessions.
    assert "+857.00" in rule and "-308.00" in rule
    assert "+1,569.50" in rule and "-579.75" in rule
    assert "worse than the exit that preceded them" in rule

    # The re-entry bar, and its separation from the existing R:R-bait rule.
    assert "genuinely NEW trapped crowd" in rule
    assert "a crowd you have ALREADY hunted" in rule

    # It must never read as permission to hold.
    assert "not a licence to hold longer" in rule
    assert "It governs HOW MANY times you enter, not when you leave" in rule


def test_v4t_open_classification_rule_is_behavioural_and_carries_its_calibration():
    """v4t (31 Aug live session): the flat/gap call inverts the whole day.

    Every open-shape rule and every pre-open note branches on flat vs gap, and
    nothing said where the line is. Measured that morning: NIFTY opened 58
    points (~0.24%) below the prior close. IH called it "almost flat" all
    session and traded the SELL side; the agent called it a gap-down and bought
    it three times, finishing +368.75 with the last two trades giving back 73%
    of the first.

    Two things this rule must keep or it stops working:

    1. The test is PARTICIPATION, not arithmetic -- did the open recruit the
       other side directly? The percentage is calibration, not the rule. If it
       decays into a bare threshold it will misfire the first time a 0.4% open
       behaves like a gap.
    2. The tie-break must survive: hesitating between the two words means flat.
       Without it the rule adds a judgement call without resolving one.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CLASSIFY THE OPEN BEFORE YOU BRANCH ON IT")

    # Participation is the test; arithmetic is only calibration.
    assert "THE TEST IS PARTICIPATION, NOT ARITHMETIC" in rule
    assert "did the open itself RECRUIT the other side" in rule
    assert "then our thing does not work" in rule

    # The measured calibration point, with both halves of the number.
    assert "58 points below the previous close" in rule
    assert "0.24%" in rule
    assert "a quarter of a percent is not a gap" in rule

    # The tie-break.
    assert "that hesitation IS the answer: it is flat" in rule

    # The consequence, stated so the rule cannot be read as bookkeeping.
    assert "+368.75" in rule
    assert "on the wrong side of the day" in rule
    assert "State the classification and the reason for it" in rule

    # SLH-017 wiring: the arithmetic is supplied, so the rule must say so and
    # must forbid re-deriving it. An unwired fact is one the model will simply
    # recompute its own way, which is the failure this whole rule is about.
    assert "SLH-017 NOW SUPPLIES THE ARITHMETIC" in rule
    assert "open_classification" in rule
    assert "READ THAT VERDICT AND STATE IT" in rule
    assert "Do not recompute it" in rule
    assert "never whether the number clears the line" in rule


def test_v4t_early_retracement_rule_points_at_follow_not_fade():
    """v4t: the first pull-back keeps the crowd OUT, so nobody gets seated.

    The dangerous misreading is to treat the retracement as a trapped crowd to
    squeeze -- it is the opposite, an absence of one -- so the rule has to send
    the reader to THE METHOD IS NOT ALWAYS A FADE rather than to a hunt.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE EARLY RETRACEMENT IS THE TRAP")

    assert "keep the crowd OUT of the move, not to end it" in rule
    assert "cannot sell directly" in rule
    assert "JUST A TRAP" in rule

    # Entry timing: it is where you join.
    assert "the retracement is where YOU" in rule

    # And who is NOT there -- the half that prevents an inverted read.
    assert "never got seated, so there is nobody to hunt" in rule
    assert "makes the trade a FOLLOW" in rule
    assert "THE METHOD IS NOT ALWAYS A FADE" in rule


def test_v4u_open_is_compared_to_the_315_level_not_the_official_close():
    """v4u (1 Sep live session): which price the open is measured against.

    v4t said to classify the open by participation. It did not say what to
    compare it TO, and that gap cost a day: NIFTY opened flat on any reference,
    but BankNIFTY read as "gapped down hard (~0.8%)" against its OFFICIAL close,
    which flipped the pre-open note's branch from BUY to SELL. IH, judging the
    same session off the 3:15 level, called it flat and booked his target.

    The rule must keep the REASON (the last fifteen minutes carry prints no
    crowd traded around) or it decays into an arbitrary preference for one
    timestamp, and it must keep the disagreement case, which is what actually
    fired here.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "CLASSIFY THE OPEN BEFORE YOU BRANCH ON IT")

    assert "COMPARE THE OPEN TO THE 3:15 LEVEL, NOT THE OFFICIAL CLOSE" in rule
    assert "more than the closing, we go by where the market" in rule
    # The reason, not just the instruction.
    assert "an artefact of the close rather than a fact about positioning" in rule

    # The measured case, both sides of it.
    assert "gapped down hard (~0.8%)" in rule
    assert "-1,488.25" in rule

    # The disagreement rule, and its link to the existing one it mirrors.
    assert "the flat one is the honest read" in rule
    assert "SHARED-GAP REQUIREMENT" in rule
    assert "A gap in ONE index is not a gapped market" in rule


def test_v4u_hierarchy_needs_time_and_never_overrides_the_stop():
    """v4u: the leading index cannot disqualify a trade seconds after entry.

    Measured the same session: a long was cut 43 seconds after entry because
    BankNIFTY was falling, and a short was cut 51 seconds after entry because
    BankNIFTY was rising. The hierarchy disqualified opposite directions inside
    twenty minutes.

    The dangerous misreading is "so hold through reversals", which on a live
    book is how a small loss becomes a large one. The rule must keep the stop,
    the max loss and premise-invalidation ranked above it, and must keep the
    counter-evidence: the same session's second trade was stopped mechanically
    for -1,689.00 and that stop was right.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "ONE CANDLE IS NOT A REVERSAL")

    # The mechanism: the first minute is the one the setup predicted.
    assert "the market have to do? It has to CREATE" in rule
    assert "that minute" in rule or "the first minute after entry" in rule

    # Both measured cuts, and the point that they contradicted each other.
    assert "FORTY-THREE SECONDS later" in rule
    assert "fifty-one seconds" in rule
    assert "disqualified a" in rule and "long for falling and a short for rising" in rule

    # The bar it sets, tied to the entry standard rather than a new one.
    assert "the same bar you would demand" in rule
    assert "If the only thing that has changed since entry is price" in rule

    # And the ranking that stops it being read as permission to hold. Named in
    # full: asserting only "never overrides the" passes while the list of what
    # it does not override is deleted, which is the half that does the work.
    assert "never overrides the" in rule
    assert "the max loss or premise-invalidation" in rule
    assert "does NOT license sitting through a real reversal" in rule
    assert "-1,689.00" in rule
    assert "that stop did its job" in rule


def test_v4v_dead_premise_rule_bars_the_NEXT_entry_without_relaxing_any_exit():
    """v4v (2 Sep live session): six shorts on one dead premise, -3,877.25.

    The rule sits in a genuine gap. Every other re-entry rule keys on a booked
    WINNER (v4s, MOVE-EXHAUSTION), and POST-LOSS SPEED LIMIT only asks the next
    setup be fresh and high-quality -- a bar the agent cleared six times over by
    naming a new PATTERN each time and the same crowd every time.

    Four things a summarising edit would flatten, each of which inverts the rule:

    1. The scope is the SESSION, not the entry. "Dead for this trade" is what the
       agent already believed, and it re-entered five times on it.
    2. It is NOT about direction. IH took the same side on the same premise and
       was also wrong. An edit that reads this as "do not short a gap-down" would
       learn precisely the wrong lesson from a day where the read was defensible.
    3. It is not "never re-enter" -- the sixth entry WON. The bar is a new crowd
       trapped AFTER the break, not a ban on trading the direction again.
    4. It governs OPENING, never closing. Read as permission to sit through a
       loss, it would turn a -3,877 day into an uncapped one.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A BREAKOUT THROUGH THE LEVEL YOUR PREMISE FORBADE")

    # It must name the gap it fills, or a reader will assume v4s already covers it.
    assert "LOSS-side twin" in rule
    assert "MOVE-EXHAUSTION also keys on a" in rule
    assert "POST-LOSS SPEED LIMIT only asks" in rule

    # IH's mechanism AND his prediction of the exact failure.
    assert "WHOLE DAY'S TRAP" in rule
    assert "YOU WILL BE WRONG IN ALL TWO-THREE PLACES" in rule
    # Why a gap in your favour is not safety.
    assert "OTHER PEOPLE ALSO START SELLING" in rule
    assert "trap FOR THAT DAY" in rule

    # 1. Scope is the session, and a pattern may not stand in for a crowd.
    assert "dead for the SESSION, not merely for that entry" in rule
    assert "the same dead premise" in rule and "wearing a new candle" in rule
    assert "trapped by price action that happened AFTER the" in rule
    assert "Another clean bearish engulfing" in rule

    # The clock, as the earlier detector.
    assert "5-10 minutes" in rule
    assert "the time it is taking is EXTRA for us" in rule

    # The measured evidence, including WHICH entry killed the premise.
    assert "SIX shorts in 65 minutes" in rule
    assert "-3,877.25" in rule
    assert "23837" in rule and "-1,086.50" in rule

    # 2. Not a direction rule -- the counter-evidence must survive.
    assert "NOT a rule about direction" in rule
    assert "IH took the SAME side" in rule and "was ALSO wrong" in rule

    # 3. Not an absolute bar -- the winning re-entry must survive too.
    assert 'Nor is it "never re-enter"' in rule
    assert "+672.00" in rule

    # 4. Opening only. This must never read as permission to hold a loser.
    assert "not a licence to hold a loser longer" in rule
    assert "INDEX HIERARCHY ON THE WAY OUT still cuts the basket" in rule
    assert "whether you may OPEN the next trade, never whether you may close" in rule


def test_v4w_post_break_slowness_is_a_book_signal_without_reversing_v4k():
    """v4w (3 Sep live session): slow candles AFTER the break invite your side.

    This rule sits between two that already exist and contradict each other if
    read carelessly, so the guard pins the seam rather than the sentiment:

    * v3y governs a slow grind AT the level, BEFORE entry -- it recruits
      opponents whose stops become fuel against you.
    * v4k governs pace AFTER entry and says slow is the SUSTAINABLE kind,
      because it keeps a trapped crowd seated.

    Today is a third case neither covers: the level has broken, the position is
    in profit, and the candles go small. The discriminator is WHO the slowness
    lets in -- an opposite-side crowd still being squeezed is fuel, fresh
    same-side traders are the retracement. Four ways an edit could break it:

    1. Reading it as a reversal of v4k rather than a scope limit on it. The
       rule must keep saying slow is fuel while a trapped crowd is squeezed.
    2. Losing the who-does-it-let-in test and leaving a bare "book when slow",
       which would fire before the break, where v3y already rules.
    3. Dropping IH's counterfactual, which is the only thing that shows the
       level and target were unchanged and only the invitation differed.
    4. Dropping the re-entry consequence -- the measured 67.5% give-back came
       from re-entering the retracement, not from booking too early.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "AFTER THE BREAK, SLOW CANDLES RECRUIT YOUR OWN SIDE")

    # 1. A scope limit, explicitly, and v4k's mechanism kept intact.
    assert "scope limit on the rule above, not a reversal" in rule
    assert "keeping a crowd trapped on the OTHER side seated" in rule
    assert "they are what pays you" in rule

    # IH's mechanism, in his words.
    assert "SMALL candles, so this will INVITE buyers" in rule
    assert "NEEDLESSLY REDUCE our profit" in rule

    # 2. The test is the crowd, not the pace.
    assert "THE TEST IS WHO THE SLOWNESS LETS IN, never the pace on its own" in rule
    assert "still squeezing a crowd trapped on the OTHER side" in rule
    assert "letting fresh traders onto YOUR side" in rule

    # 3. The counterfactual that isolates the invitation as the deciding factor.
    assert "was NOT giving others a chance to buy" in rule
    assert "only the invitation decided" in rule

    # 4. The re-entry consequence, and why it is not a second setup.
    assert "DO NOT RE-ENTER THE RETRACEMENT YOU JUST BOOKED AHEAD OF" in rule
    assert "joining them at the worst price" in rule

    # The measured evidence, including that the BOOK itself was correct.
    assert "+3,254.00" in rule
    assert "-1,726.50" in rule
    assert "handed back 67.5% of itself" in rule

    # It must not read as permission to book early anywhere, nor relax an exit.
    assert "not licence to book on any slow bar" in rule
    assert "A SLOW GRIND AT THE LEVEL RECRUITS THE WRONG CROWD already governs" in rule
    assert "the stop, the max loss and premise-invalidation are unchanged" in rule


def test_v4x_stop_is_measured_from_the_fill_not_from_the_decision_price():
    """v4x (04 Sep, measured on this book): the stop's basis is the FILL.

    This is the only rule in the corpus about WHERE the stop sits relative to
    the price that is actually in front of you, and it exists because being
    right about direction did not save the trade. IH bought the same BankNIFTY
    57500 call in the same minute and it paid him the day; ours was stopped
    three seconds after it opened. Five ways an edit could quietly break it:

    1. Losing the decision-versus-fill distinction and leaving a generic "give
       the trade room". v4d already names the tolerated move in advance; the
       whole content here is that the named distance has to be re-measured
       when the position opens, because the reference price can be gone.
    2. Dropping the measured numbers. 25.3 points sized, 4.4 points left at the
       fill, stopped 3 seconds later, recovered within 9 -- without those the
       rule is an opinion rather than an arithmetic check.
    3. Keeping only the "re-derive the stop" branch and losing "skip it". A
       re-derived stop that cannot be sized must end as NO TRADE.
    4. Leaving the third option -- keeping the old stop to keep the old size --
       merely discouraged rather than forbidden. That is the exact move the
       sizing arithmetic rewards, so it has to be named and refused.
    5. Losing the risk-invariance clause, which is what stops this reading as
       permission to widen a stop or exceed the budget.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE STOP IS A DISTANCE FROM THE FILL")

    # 1. The distinction this rule exists for, and its link back to v4d.
    assert "NOT FROM THE PRICE YOU REASONED ON" in rule
    assert "RE-MEASURED at the instant the position actually opens" in rule
    assert "Deciding and filling are not the same moment" in rule

    # 2. The measured case, including that the READ was right.
    assert "MEASURED ON THIS BOOK (2026-09-04)" in rule
    assert "matched IH's to the strike" in rule
    assert "same BankNIFTY 57500 call in the same minute" in rule
    assert "entry=23935.30 against a stop at 23910.00" in rule
    assert "25.3-point allowance" in rule
    assert "spot at 23914.40" in rule
    assert "4.4 points of room" in rule
    assert "83% of the allowance was already spent" in rule
    assert "stopped 3 seconds later at 23908.40" in rule
    assert "back above 23940 within 9 seconds" in rule

    # The test is run against the live price, not the remembered one.
    assert "never against the price you reasoned on" in rule
    assert "not the trade on offer any more" in rule

    # 3. Both honest branches survive, skip included.
    assert "accept the SMALLER size" in rule
    assert "skip it" in rule
    assert "IF THE RE-DERIVED STOP IS TOO WIDE TO SIZE, THE ANSWER IS NO TRADE" in rule
    assert "declining a trade it could not have held" in rule

    # 4. The tempting third option, named and forbidden outright.
    assert "NEVER keep the original stop in order to keep the original size" in rule
    assert "a loss that has already been arranged" in rule
    assert "never an invitation to walk the stop closer until the size fits" in rule

    # 5. It can only shrink a position, never enlarge the risk.
    assert "NOTHING HERE WIDENS RISK" in rule
    assert "make it SMALLER or make it not exist" in rule

    # And where it applies hardest -- the moment the agent most wants to act.
    assert "bites hardest in the opening minutes" in rule
def test_v4y_retracement_size_decides_turn_versus_continuation():
    """v4y (07 Sep live session + this book): measure the counter-move.

    v4u already forbids cutting on "merely a move against you" and demands a
    CONFIRMED reversal on the leading index. Monday's exit SATISFIED that -- a
    confirmed BankNIFTY hammer -- and still gave back the day, because a
    confirmed one-candle pattern off a fresh low can be true and trivial at the
    same time. v4y is the size test that v4u lacks. Six ways an edit breaks it:

    1. Losing the link to v4u and leaving a free-standing "hold longer". The
       rule's whole claim is that it supplies a MEASUREMENT v4u does not have.
    2. Dropping IH's mechanism -- a market that means to continue cannot afford
       a big retracement, because that is everyone else's good entry. Without it
       the rule is an assertion instead of a reason.
    3. Losing "measure it in index points, never in premium". Option buying is
       what makes an 11-point retracement look like a reversal, and the P&L is
       precisely where the agent was looking.
    4. Dropping the measured numbers. 11 points against a 90-point session move,
       and 557.50 -> 538.25 on the same 11 points, are what make it checkable.
    5. Losing the booking corollary. The same measurement says an ABSENT
       retracement is overdue, not a promise -- without it the rule reads as
       one-directional "never exit".
    6. Losing the override clause, which keeps the stop and max loss supreme.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "SIZE THE RETRACEMENT")

    # 1. It is explicitly the measurement half of v4u, not a replacement.
    assert "A BIG ONE MEANS A TURN, A SMALL ONE MEANS CONTINUATION" in rule
    assert "quantitative half of the rule above" in rule
    assert "satisfy v4u's letter while being nothing at all" in rule

    # 2. IH's mechanism, in his words, and the tie back to v4w.
    assert "HOW MUCH will the retracement be" in rule
    assert "CAN this market turn" in rule
    assert "IF IT DID A BIG RETRACEMENT AND THEN FELL, THEN ANYONE WOULD START SELLING" in rule
    assert "CANNOT hand out a big" in rule
    assert "A BIG RETRACEMENT HAPPENS WHEN THE MARKET HAS TO TURN" in rule

    # 3. Where to measure it -- and the one place that must not be used.
    assert "MEASURE IT IN INDEX POINTS, AGAINST THE MOVE YOU ARE ALREADY IN" in rule
    assert "Never in premium" in rule
    assert "A SMALL GREEN CANDLE REDUCES OUR PROFIT A LOT" in rule
    assert "the P&L is the one place you must not look" in rule

    # 4. The measured case, including that v4u had been satisfied.
    assert "MEASURED ON THIS BOOK (2026-09-07)" in rule
    assert "confirmed BankNIFTY" in rule
    assert "ELEVEN POINTS" in rule
    assert "90 points down from its open" in rule
    assert "557.50 to 538.25" in rule
    assert "basket lost 2,169.00" in rule
    assert "closed near the low" in rule
    assert "v4u's confirmed-pattern requirement had been satisfied" in rule

    # 5. The same measurement pointed the other way -- when to BOOK.
    assert "THE BOOKING COROLLARY" in rule
    assert "no retracement in continuous selling across all three indices" in rule
    assert "An absent retracement is not a promise of more trend, it is an" in rule

    # 6. It can never outrank the hard risk controls.
    assert "never overrides the stop, the daily max loss, or premise-invalidation" in rule
    assert "does not license sitting through a genuine turn" in rule
    assert "too small to be the turn it is being called" in rule
def test_v4z_a_counter_move_needs_fuel_to_be_dangerous():
    """v4z (08 Sep live session + this book): whose stops would it eat?

    v4y measures a counter-move after it prints. This one asks whether it can
    grow at all. Six ways an edit could break it:

    1. Losing the direction of the mechanism. The fuel is YOUR OWN side's stops
       above you -- other shorts, whose stop-outs are buy orders. Flip that and
       the test points at the wrong crowd.
    2. Turning it into "ignore reversal patterns". It adds a question before
       acting on a pattern; it does not delete v4u.
    3. Dropping IH's compounding description -- creates SLs, rises, creates more
       -- which is the only thing that explains why a fuelled move is different
       in kind rather than just bigger.
    4. Losing the two readable reasons the seats were empty that day (the
       retracement already given at round-number support, and premiums too high
       for size). Without them the test is unanswerable in the moment.
    5. Dropping the measured pair. The SAME session got it wrong at 09:50 and
       right at 10:18, which is what makes it teachable rather than a scolding.
    6. Losing the override clause.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A COUNTER-MOVE IS ONLY DANGEROUS WHEN IT HAS FUEL")

    # 1. Direction of the mechanism, and how it compounds.
    assert "THE FUEL IS YOUR OWN SIDE'S STOPS SITTING ABOVE IT" in rule
    assert "other SHORTS above you" in rule
    assert "stopping them out puts buy orders into the book" in rule
    # The compounding half of the sentence, not just the first link.
    assert "which lifts price, which stops out more of them" in rule

    # It composes with v4y rather than replacing it.
    assert "measures a counter-move AFTER it prints" in rule
    assert "a counter-move with no fuel cannot become the big one" in rule

    # 3. IH's words, including the compounding loop.
    assert "THE SELLERS' SLs ARE NOT AVAILABLE HERE" in rule
    assert "creates more SLs then rises" in rule
    assert "the market will not go higher than this" in rule

    # The question the agent must actually ask.
    assert "WHOSE STOPS WOULD IT EAT ON THE WAY UP?" in rule
    assert 'the question is never "how convincing is this reversal pattern?"' in rule
    assert "aimed at the traders arriving NOW" in rule

    # 4. Why the seats were empty -- both readable from the session itself.
    assert "already given its retracement earlier and was holding round-number support" in rule
    assert "but NOT in big quantity" in rule

    # 5. The measured pair, wrong then right, in one session.
    assert "MEASURED ON THIS BOOK (2026-09-08)" in rule
    assert "09:45:13 from 23670.60" in rule
    assert "premise-invalidation bullish reversal cluster" in rule
    assert "By 10:03 it was" in rule
    assert "23660.55, BELOW the entry" in rule
    assert "cluster died in ten minutes" in rule
    assert "-786.25" in rule
    assert "IH sat through the very same move" in rule
    assert "booking +716.00" in rule

    # 2 and 6. Scope: it does not delete v4u, and cannot outrank hard risk.
    assert "It is not \"ignore reversal patterns\"" in rule
    assert "does not weaken v4u" in rule
    assert "a shakeout aimed at you, not a turn" in rule
    assert "never overrides the stop, the daily max loss, or premise-invalidation" in rule
def test_v5a_a_tighter_stop_buys_size_rather_than_safety():
    """v5a (09 Sep, measured on this book): the stop is a size dial.

    The agent cannot derive this from inside a decision -- the sizing maths runs
    in the host, and the number the budget watches does not move when the stop
    narrows. Six ways an edit could break it:

    1. Losing the arithmetic. Without one_lot_risk and the floor(), the claim
       that rupee risk is held constant is an assertion rather than a mechanism.
    2. Dropping the 25.30 -> 1 lot and 7.65 -> 5 lots endpoints, which are what
       make the size effect concrete rather than directional.
    3. Losing the MIRROR consequence. That leg is equal-lot and outside the
       budget, so it is where tightening actually costs money; an edit that
       keeps only "more quantity" loses the part that bites.
    4. Dropping either of the other two quantity-scaled costs (per-unit
       slippage, and a constant loss suffered more OFTEN).
    5. Losing the measured trade, including that the DIRECTION was right and a
       premise-width stop would have survived. Without it the rule reads as
       "use wider stops", which is not what it says.
    6. Losing the practical form -- stop first, size second, never the reverse.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A TIGHTER STOP IS NOT LESS RISK")

    # 1. The mechanism, and its relationship to v4x.
    assert "IT IS MORE SIZE" in rule
    assert "v4x handles a stop too WIDE to size" in rule
    # It must claim to cover the OTHER direction, not restate v4x.
    assert "the direction it does not cover" in rule
    assert "one_lot_risk = |entry - stop| * lot_size" in rule
    assert "lots = min(max_lots, floor(budget / one_lot_risk))" in rule
    assert "held CONSTANT by construction" in rule
    assert "What the stop actually sets is the QUANTITY" in rule

    # 2. The measured endpoints, same budget.
    assert "MEASURED ACROSS 19 SIZING DECISIONS" in rule
    assert "25.30-point stop bought 1 lot" in rule
    assert "7.65-point stop bought 5" in rule

    # 3. The mirror -- equal-lot, outside the budget, where it actually costs.
    assert "THE BANKNIFTY MIRROR IS EQUAL-LOT AND SITS OUTSIDE THE BUDGET" in rule
    assert "about 15,350 of premium" in rule
    assert "about 46,845" in rule
    assert "TRIPLED the exposure the budget does not measure" in rule

    # 4. The other two costs that scale with quantity.
    assert "charged per unit" in rule
    # The probability claim is kept as MECHANICS but explicitly corrected by the
    # book, so a future reader cannot re-derive the stop-floor gate the journal
    # already refused. Both halves are pinned: the intuition AND its refutation.
    assert "MECHANICALLY the probability the stop is hit rises as it tightens" in rule
    assert "constant rupee loss suffered more often is not a smaller loss" in rule
    assert "BUT THIS BOOK DOES NOT SHOW IT" in rule
    assert "stopped out 32% of the time and averaged +688" in rule
    assert "stopped out 45% of the time and averaged +301" in rule
    assert "NEGATIVE at all of them, costing between 8,417 and 65,413" in rule
    assert "a SYMPTOM of entry quality rather than an independent risk dial" in rule
    assert "never on its own as a reason to refuse it or to size down" in rule
    assert "leverage wearing the costume of caution" in rule

    # 5. The measured trade, including that the read was right.
    assert "MEASURED ON THIS BOOK (2026-09-09)" in rule
    assert "15.15 points away, which bought 2 lots" in rule
    assert "NINETY-ONE SECONDS later at 23519.30" in rule
    assert "-1,202.00" in rule
    assert "The direction was right" in rule
    assert "NIFTY reached 23474.85 by 10:09" in rule
    assert "would have bought ONE lot -- would have survived" in rule

    # 6. Stop first, size second -- and the tie back to v4x for the other end.
    assert "choose the stop THE PREMISE REQUIRES" in rule
    assert "Never choose a stop in order to obtain a size" in rule
    assert "shrinking stop across consecutive trades as a warning" in rule
    assert "v4x already governs and the answer is no trade" in rule
def test_v5b_hold_duration_decides_evicted_versus_trapped():
    """v5b (10 Sep live session): time the hold before naming the crowd.

    v3z reads a crowd's ABSENCE from a missing rip. This rule explains how a
    crowd that WAS there becomes absent, and supplies the measurement -- the
    duration of the hold, calibrated to the participant. Six ways an edit
    could break it:

    1. Losing the counterfactual. Up-then-drop and up-then-hold are the SAME
       price shape; without both halves the rule cannot discriminate.
    2. Dropping the conversion mechanism -- sellers do not merely fail to be
       trapped, they exit and flip. That is why the inventory does not exist
       afterwards.
    3. Losing the per-participant time scale. "Minutes vs hours" is the whole
       measurement; without the intraday/positional split it is a vibe.
    4. Dropping the no-quick-return clause, which is what makes the eviction
       persist into following sessions rather than one bar.
    5. Losing the tie to v4z (fuel answered in advance) or the loss-limit
       caveat, which is what stops this becoming licence to sit.
    6. Dropping the both-scales observation, which shows the mechanism is not
       only a multi-day phenomenon.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A LONG ENOUGH HOLD EVICTS A CROWD")

    # 1. Both halves of the counterfactual, in IH's words.
    assert "A QUICK REVERSAL TRAPS IT" in rule
    # It must claim to be v4e's axis pointed at removal, not a new topic.
    assert "the same axis pointed at REMOVAL" in rule
    # And the core claim: one price shape, two meanings, decided by duration.
    assert "THE SAME UP-MOVE MEANS OPPOSITE THINGS DEPENDING ON HOW LONG IT HOLDS" in rule
    assert "fallen IMMEDIATELY, or fallen after stopping only a LITTLE" in rule
    assert "you would find sellers SEATED" in rule
    assert "Up-then-drop traps a crowd" in rule

    # 2. Conversion, not merely non-trapping.
    assert "start thinking of BUYING" in rule
    assert "they all EXIT their selling trade" in rule
    assert "the market COMPLETELY REMOVED the sellers" in rule

    # 3. The measurement, and the per-participant calibration.
    assert "THE TIME SIGNATURE IS THE MEASUREMENT" in rule
    assert "INTRADAY traders exit on a small retracement" in rule
    assert "roughly 12:30 to 2:30, that is TWO HOURS" in rule
    assert "minutes clear only the intraday book, hours clear the positional one too" in rule

    # 4. Why the eviction persists.
    assert "does not quickly make a selling trade again" in rule
    assert "neither restocks the inventory you would have hunted" in rule

    # 5. What it licenses -- and the boundary that stops it becoming licence to sit.
    assert "the trend is a FOLLOW, not a hunt" in rule
    assert "answers v4z's fuel question in advance" in rule
    assert "THIRD day of it" in rule
    assert "does NOT license is holding past the loss limit" in rule

    # 6. Same mechanism, two scales, one session.
    assert "RUNS AT BOTH SCALES IN ONE SESSION" in rule
    assert "their SLs are around 500" in rule
    assert "cleared inside the hour" in rule

    # The instruction the agent must actually follow.
    assert "TIME IT before naming it" in rule
    assert "Minutes and the opposing crowd is still seated" in rule
    assert "Hours and they are gone" in rule
def test_v5c_a_big_retracement_recruits_the_next_crowd():
    """v5c (11 Sep live session + this book): measure the counter-move's SIZE.

    Paired with v5b deliberately -- same counter-move, two axes. v5b times it
    and finds an eviction; v5c sizes it and finds a recruitment. Six ways an
    edit could break it:

    1. Losing the pairing, or the small-versus-big contrast. "Let it fall" after
       a small one and "difficult to fall" after a big one are the whole rule.
    2. Dropping WHY size matters -- after a small retracement nobody can tell it
       happened, so nobody is recruited. Without that the rule is superstition.
    3. Losing the recruitment loop, which is what makes the dip predictable
       rather than merely survivable.
    4. Dropping either precondition. Negative sentiment and an unbroken lower
       point are both required; an edit keeping only one turns a conditional
       setup into a standing bias.
    5. Losing the measured trade, including that the exit was AT the session low
       and that the move afterwards was 62 points.
    6. Losing the breakdown/loss-limit exits, which is what stops this being
       read as permission to sit.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A BIG RETRACEMENT RECRUITS THE CROWD THAT GETS HUNTED NEXT")

    # 1. The pairing with v5b, and the two-sided contrast.
    assert "a long HOLD evicts a crowd, a big RETRACEMENT manufactures one" in rule
    assert "v5b measures its duration, this one measures its SIZE" in rule
    assert "falls after a SMALL retracement, let it fall" in rule
    assert "falling after a BIG retracement becomes DIFFICULT for the market" in rule

    # 2. Why size is the thing that matters.
    assert "cannot even tell a retracement happened" in rule
    assert "EVERYONE will sell there" in rule
    assert "goes UP through their stops" in rule

    # 3. The loop, narrated while it ran.
    assert "this candle formed only to give GREED" in rule
    assert "As premiums rise he will make a put trade" in rule
    assert "when the market eats all their SLs, see how fast the momentum comes" in rule

    # 4. Both preconditions -- and that they are REQUIRED, not decoration.
    assert "TWO CONDITIONS, AND BOTH MUST HOLD" in rule
    assert "THE SENTIMENT MUST ALREADY BE NEGATIVE" in rule
    assert "On a fresh or two-sided chart a big retracement recruits nobody" in rule
    assert "THE SESSION'S LOWER POINT MUST NOT BE CROSSED" in rule
    assert "only sellers are going to come and nobody will buy" in rule
    assert "after a breakdown the market takes TIME making a trap" in rule
    assert "a level test in the sense of v4v" in rule

    # 5. The measured trade -- cut at the low, and what followed.
    assert "MEASURED ON THIS BOOK (2026-09-11)" in rule
    assert "210-point gap down" in rule
    assert "confirmed bearish reversal cluster" in rule
    assert "23245.35 -- THE SESSION LOW" in rule
    assert "23307.75 by 10:22, sixty-two points" in rule
    assert "closed at the exact moment its premise was being manufactured" in rule

    # 6. The two exits that still bind.
    assert "not permission to sit through a breakdown" in rule
    assert "the loss goes far outside the LOSS LIMIT" in rule
    assert "The lower point and the loss limit both still bind" in rule
def test_v5d_an_unloseable_setup_is_a_broken_setup():
    """v5d (13 Sep Edge lecture): the contrapositive of v3z.

    v3z asks you to name the way the trade fails. This says the inability to
    name one is itself the disqualifying signal. Five ways an edit breaks it:

    1. Losing the tie to v3z, or inverting the direction of the claim -- the
       whole rule is that NOT seeing a loss is bad news, not confidence.
    2. Dropping "re-check it". The instruction is to revisit the read, not to
       trade it smaller, and a summariser will reach for the latter.
    3. Losing the checkable symptom -- a rationale where up, down, sideways and
       momentum all profit. Without it the rule cannot be applied to anything.
    4. Dropping the passage that aims it at this agent specifically: a long
       rationale that answers every objection FEELS thorough and is the failure.
    5. Losing the capital clause, which forecloses the obvious wrong fix.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "IF YOU CANNOT SEE HOW THIS LOSES")

    # 1. Explicitly the contrapositive of v3z, in the stated direction.
    assert "THE SETUP IS BROKEN" in rule
    assert "contrapositive of the rule above" in rule
    assert "the inability to name it is not confidence, it is a defect" in rule

    # IH's own words, including that he has chased the other kind.
    assert "will only work IF it has a CHANCE OF LOSS" in rule
    assert "the setup that makes profit is the one that HAS a margin of loss" in rule
    # The instruction inside the quote, not just the claim around it.
    assert "you should RE-CHECK it" in rule
    # Anchored on the setup NOT lasting, which is the whole point of the anecdote.
    assert "but it runs for a few days, then its number comes up" in rule

    # 2 and 3. The instruction, and the symptom that triggers it.
    assert "RE-CHECKED rather than taken" in rule
    assert "sideways and you profit, momentum and you profit" in rule
    assert "stopped being a read and become an argument" in rule

    # 4. Why it is pointed at this agent.
    assert "WHY THIS RULE IS AIMED AT YOU" in rule
    assert "answers every objection FEELS like thoroughness" in rule
    assert "state the branch that costs you money" in rule
    assert "do not size the trade down -- re-check the read" in rule

    # 5. Capital is not the repair.
    assert "Neither big nor small capital repairs this" in rule
    assert "until it goes outside the range" in rule


def test_v5d_the_read_carries_more_edge_than_the_trade():
    """v5d (13 Sep Edge lecture): analysis and execution are scored apart.

    Five ways an edit could break it:

    1. Reversing which side carries more edge. The claim is that the READ is the
       stronger one, and it is the reason to fix the read first.
    2. Losing WHY -- no entry, no target, no clock in the read, so emotion stays
       out. Without the mechanism it is an unsupported ranking.
    3. Dropping "a right analysis does not mean a right trade", which he repeats
       three times and which is the operative consequence.
    4. Losing the mapping onto this agent -- note plus direction read is the
       analysis, entry/stop/exit is the trade -- which is what makes it usable.
    5. Dropping the scoring rule, so a losing session is allowed to falsify a
       correct read (or a correct read to excuse a bad trade).
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "YOUR EDGE IN THE READ IS HIGHER THAN YOUR EDGE IN THE TRADE")

    # 1 and 2. The ranking, and the reason for it.
    assert "the gap is structural rather than a run of bad luck" in rule
    assert "in analysis there is no matter of entry/exit timing" in rule
    assert "the edge there is LOWER" in rule
    assert "no entry, no target and no clock" in rule
    assert "so emotion stays out of it" in rule

    # 3. The consequence, and that he grades himself by it.
    assert "THE CONSEQUENCE HE REPEATS THREE TIMES" in rule
    assert "does NOT mean your trade will also be right" in rule
    assert "in analysis you could say we were 100% right" in rule
    assert "here we had a loss" in rule

    # 4. The mapping onto this agent's own parts.
    assert "The pre-open note plus your direction read is the ANALYSIS" in rule
    assert "the entry, the stop and the exit are the TRADE" in rule

    # 5. Scored apart, and which one to fix first.
    assert "a losing session does NOT falsify the read" in rule
    assert "a correct read does not entitle the trade" in rule
    assert "the way v4h scores entry apart from direction" in rule
    assert "then your trading time will come" in rule


def test_v5e_the_stop_must_clear_the_level_the_premise_named():
    """v5e (15 Sep): the stop sat 0.55 of a point inside the level it cited.

    The agent shorted a reversed gap-up, naming the break of the previous
    close as its premise, then put the stop BELOW that same previous close.
    Price came back, stopped 0.15 short of the close -- not even a completed
    retest -- and that was enough. The move then paid the full target and 197
    points beyond. Six ways an edit breaks the rule:

    1. Losing the tie to the pattern-stop bullet it corrects. Without
       "necessary and not sufficient" this reads as a competing rule, and the
       older bare "stop just beyond the pattern" keeps winning.
    2. Losing the mechanism -- that price PROBES a level rather than halting
       at it, and the most-watched reference is the one probed. Without it the
       clearance requirement is an arbitrary preference.
    3. Losing the numbers that carry the authority: 0.55 inside, a 23397.95
       return, 0.15 short of the close. The incompleteness of the retest is
       the whole proof.
    4. Losing "the read was right". A summariser files this under bad trades
       and the lesson inverts into "be less confident", which is backwards.
    5. Losing the portable form -- a stop AT a level is a stop INSIDE it.
    6. Losing the NO TRADE clause, which forecloses the obvious wrong fix of
       tucking the stop back in to keep the size.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "THE LEVEL YOU BROKE TO GET IN IS THE LEVEL PRICE COMES BACK TO")

    # 1. A refinement of the bullet above it, not a rival to it.
    assert "The rule above puts the stop just beyond the PATTERN" in rule
    assert "That is necessary and it is not sufficient" in rule

    # 2. The mechanism, and what it does to a stop placed short of the level.
    assert "Price does not halt at a number, it probes through it" in rule
    assert "the reference the whole market is watching is the one that gets probed" in rule
    assert "it is standing in the queue ahead of it" in rule

    # 3. The measured case, including the margin that decided it.
    assert "and 0.55 of a point BELOW that same previous close" in rule
    assert (
        "Price returned to 23397.95: 0.40 past the stop, and still 0.15 SHORT "
        "of the previous close"
    ) in rule
    assert "The retest of the named level never even completed" in rule

    # 4. The read was RIGHT -- this is an execution loss, not a confidence one.
    assert "on a day the READ WAS RIGHT" in rule
    assert "crossing the 23300 target around 12:59" in rule
    assert "The thesis paid in full and the trade collected none of it" in rule

    # 5. The test, and the portable one-line form of it.
    assert "name the nearest reference BEYOND your stop" in rule
    assert "A stop AT a level is a stop INSIDE it" in rule
    assert "the probe that makes a level a level overshoots it by definition" in rule

    # 6. Size is what gives -- never the stop. Both cross-references intact.
    assert "THE CLEARANCE IS NEVER BOUGHT BACK FROM THE STOP" in rule
    assert "A TIGHTER STOP IS NOT LESS RISK" in rule
    assert "THE STOP IS A DISTANCE FROM THE FILL" in rule
    assert "the answer is NO TRADE, never the nearer stop that happens to fit" in rule
    assert "in both the thing that gives is the SIZE, never the stop" in rule


def test_v5f_a_one_way_day_seats_nobody_and_the_open_is_the_receipt():
    """v5f (16 Sep): the premise UNDER the branch, not the branch itself.

    Four entries assumed a 457-point one-way session had seated sellers, then
    read a 0.36% open as the gap-up that would trap them. IH read the same open
    as flat, followed the selling, and won. Seven ways an edit breaks it:

    1. Losing the tie to v4t -- that rule picks the WORD, this one says what the
       SIZE measures. Apart, they read as two opinions about gaps.
    2. Losing the mechanism: a crowd survives only through a RETRACEMENT, and
       continuous momentum makes everyone book and leave. Without it, "a big
       move seats a crowd" is the intuitive default and wins again.
    3. Losing the link to v5c, which supplies the same mechanism one scale down.
    4. Losing the falsification test, which is the only part that is actionable
       BEFORE the trade: say what the open would look like if the crowd existed.
    5. Softening "a small gap is evidence AGAINST" into "weaker evidence for".
       That inversion is the whole rule; a summariser will flatten it.
    6. Losing the three-strike history. One loss reads as bad luck; three
       sessions decided by one word is what makes it the morning's key call.
    7. Losing the ordered practical form, where the retracement question comes
       FIRST and can stop the enquiry on its own.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A ONE-WAY DAY LEAVES NOBODY SEATED")

    # 1. Explicitly the size-half of the classification v4t makes.
    assert "AND THE NEXT OPEN IS THE RECEIPT" in rule
    assert "The rule above decides which WORD describes the open" in rule
    assert "what the open's SIZE is actually measuring" in rule

    # 2. The mechanism, in IH's words, both directions.
    assert "when do sellers REMAIN seated? When a retracement happens" in rule
    assert "every trader keeps fearing that a retracement might come" in rule
    assert "traders do come, but they also book their target and leave" in rule
    assert "the chances of HOLDING are low here" in rule
    assert "ends with almost NOBODY carrying inventory, however large its range" in rule

    # 3. The scale relationship to v5c.
    assert "That is v5c one scale up" in rule
    assert "a retracement is what lets a crowd survive into the NEXT session" in rule

    # 4. The test that can run before the entry, in its stated order.
    assert "if sellers had been seated in good quantity, what would the market have opened as" in rule
    assert "state what the open WOULD look like if your trapped crowd existed" in rule

    # 5. Evidence AGAINST, not weaker evidence for.
    assert "A small gap is not a weak version of that open" in rule
    assert "it is the evidence AGAINST it" in rule
    assert "the crowd that would have forced a big gap is not there to force one" in rule

    # 6. The measured case and the repeat history that gives it weight.
    assert "ran 457 points in one continuous direction" in rule
    assert "+0.36% on NIFTY and +0.38% on BankNIFTY" in rule
    assert "IH read the SAME open as FLAT" in rule
    assert "its only winner was closed by an erroneous exit" in rule
    assert "never once tested against the calibration above" in rule
    assert "THIS IS THE THIRD SESSION DECIDED BY THAT ONE WORD" in rule
    assert "Nothing downstream was wrong on any of the three" in rule

    # 7. The ordered form, including the early stop.
    assert "did yesterday RETRACE?" in rule
    assert "If it ran one way, assume no seated crowd and stop there" in rule
    assert "the plan is to FOLLOW the move rather than to hunt it" in rule


def test_v5g_an_eviction_leaves_you_in_the_same_sideways_stretch():
    """v5g (17 Sep): the reflexive half of v5b, measured on both sides at once.

    IH and this book reached the identical read, took the identical side, and
    finished the day in opposite columns -- he at his loss limit, the book
    +11,355. The difference was the box, not the judgement. Seven ways an edit
    breaks it:

    1. Losing the tie to v5b. Alone this reads as generic "cut quickly" advice
       rather than the specific thing a FOLLOW inherits.
    2. Losing why the follow is exposed: nobody trapped means nothing forced to
       move. That is the mechanism; without it the rule is a mood.
    3. Losing that the eviction argument is about the PAST. It is the single
       sentence that stops "they are gone" being read as "so it will run".
    4. Losing the reflexive read-back -- the decay clock that emptied them is
       the clock you are about to sit on. This is the whole idea.
    5. Losing the both-sides measurement. One side alone reads as luck; the
       pair is what makes it evidence.
    6. Losing the honesty clause. The winning exit was the cutoff firing while
       the model was down, and an edit that drops it turns the day into the
       agent out-trading him, which is false and flattering.
    7. Losing the distinction at the end -- the premise can still be intact
       while the trade is already dead.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "A CROWD EVICTED BY A SIDEWAYS STRETCH")

    # 1 and 2. What the follow inherits, and why it is exposed.
    assert "LEAVES YOU IN THE SAME SIDEWAYS STRETCH" in rule
    assert "v5b reads an eviction and licenses a FOLLOW rather than a hunt" in rule
    assert "once nobody is trapped, nothing is forced to move" in rule
    assert "the position can idle exactly as theirs did" in rule

    # 3. Evidence about the past, not a promise about the next hour.
    assert "The eviction argument is evidence about the PAST" in rule
    assert "it promises nothing about the next hour" in rule

    # 4. The reflexive read-back, in IH's own reasoning.
    assert "READ THE PREMISE BACK AT YOURSELF" in rule
    assert "his premium decays and he will not hold" in rule
    assert "on the same clock, paying the same decay" in rule
    assert "a trade entered INTO a sideways stretch" in rule

    # 5. Both sides of the same idea, with the numbers that make it evidence.
    assert "uniquely on BOTH sides of the same idea" in rule
    assert "took the identical side" in rule
    assert "23,358.70 by 12:38" in rule
    assert "there is no need to apply the brain here, cut the trade and get out" in rule
    assert "in two BOUNDED windows" in rule
    assert "finished +11,355 on a day the same read cost him his limit" in rule
    assert "right for about ninety minutes and wrong afterwards" in rule

    # 6. The honesty clause -- this was not superior judgement.
    assert "DO NOT READ THAT AS OUT-TRADING HIM" in rule
    assert "the model had been failing every bar since 10:42" in rule
    assert "What won was the BOX, not the judgement" in rule

    # 7. The practical form, including the closing distinction.
    assert "name the window before entering" in rule
    assert "carried by momentum ALREADY present" in rule
    assert "a stall is the exit rather than something to wait through" in rule
    assert "the premise has not failed -- but the trade has" in rule


def test_v5j_the_exit_is_judged_on_price_not_on_the_rupee_figure():
    """v5j (23 Sep): IH's "look at the chart, not the premium", measured here.

    The book agreed with him independently, in both outcome classes. Eight
    ways an edit breaks it:

    1. Losing that it qualifies v4f rather than replacing it. It supplies the
       INPUT to v4f's trigger; read as a new trigger it fights the rule above.
    2. Losing IH's words, and that his own invalidation was a PRICE.
    3. Losing why the figure is LATE for this agent specifically: the ~30s
       turn, and the one-directional sign flips it produces.
    4. Losing the measurement, especially that it survives the outcome split.
       Unsplit, it could be read as "winners vs losers" in disguise.
    5. Losing the reconciliation with v4f, which is what keeps the two rules
       from being read as a contradiction.
    6. Losing "not a licence to exit less". The same exits beat the bracket
       overall, and 23 Sep's trade 1 would have been STOPPED if held -- without
       that, the rule re-derives a hold-longer gate the book refuses.
    7. Losing the check. Without a concrete test it is a mood, and moods are
       what get talked past.
    8. Losing the limits, which are what stop it being quoted as precise.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "JUDGE THE EXIT ON THE CHART, NOT ON THE PREMIUM")

    # 1. It reads v4f's trigger; it does not replace it.
    assert "The rule above says WHEN to leave" in rule
    assert "the answer is price, never the rupee figure" in rule

    # 2. His words, and his line was a price.
    assert "IH SAID IT MID-TRADE" in rule
    assert "So look at the trade LESS" in rule
    assert "the chart looks better than the premiums right now" in rule
    assert "He had named what would prove him wrong as a PRICE" in rule

    # 3. Late, not just noisy -- and which way it fails.
    assert "IT IS LATE" in rule
    assert "a turn that takes about half a minute" in rule
    assert "median 37.0s on 23 Sep" in rule
    assert "differed by a median of 496.50" in rule
    assert "FOUR booked what the agent believed was a profit and realised a loss" in rule
    assert "never the reverse" in rule
    assert "filled at -1,173.75" in rule

    # 4. The measurement, split by outcome.
    assert "beat holding by +241.25 points" in rule
    assert "WORSE than holding, by -49.70" in rule
    assert "It survives splitting by outcome" in rule
    assert "among losers +2.08 against -7.38" in rule
    assert "did worse than letting the stop fire" in rule

    # 5. Reconciled with v4f.
    assert "THIS DOES NOT CONTRADICT v4f, IT TELLS YOU HOW TO READ IT" in rule

    # 6. Not a hold-longer rule.
    assert "IT IS NOT A LICENCE TO EXIT LESS" in rule
    assert "+191.55 points overall" in rule
    assert "would have been stopped at -11.75" in rule
    assert "The exit was right; the rupee figure it gave as a reason was false" in rule

    # 7. The check.
    assert "strike the rupee figure out of your exit reason" in rule
    assert "It must still name a PRICE" in rule
    assert "you do not yet have an exit reason" in rule

    # 8. The limits.
    assert "only ten rupee-citing losers" in rule
    assert "It is direction, not a precise figure" in rule


def test_v5i_the_follow_trade_seats_the_crowd_it_was_entered_on():
    """v5i (22 Sep): the only rule in the family where YOU are the crowd.

    IH bought a flat open because "the buyers' SLs are not available", then
    narrated his own position becoming the inventory that got hunted. Seven
    ways an edit breaks it:

    1. Losing that this one puts you INSIDE the crowd. Without that contrast
       it reads as a restatement of v5c or v4e, which keep you outside.
    2. Losing that the read is about the INSTANT of entry. That is the whole
       claim -- a premise that is true when you act and false while you hold.
    3. Losing IH's first-person narration. He is the evidence; paraphrased
       into the third person it becomes a generic warning about crowds.
    4. Losing the link to v5h's standing-premise clause, which is what this
       rule supplies a mechanism for rather than merely repeating.
    5. Losing ANY of the three refusals. Each is a plausible repair that this
       book priced and rejected, and an unrefused repair gets re-derived --
       the time stop especially, because IH states it himself.
    6. Losing the inversion inside the third refusal. "Do not tighten it" is
       advice; "the blocked trades made +1,219 and the kept ones lost 341" is
       a finding, and it points the opposite way from the intuition.
    7. Losing the admission that it would not have changed today. The rule is
       carried on IH's book, and hiding that would make it look self-proving.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "WHEN NOBODY IS SEATED, YOUR OWN ENTRY IS WHAT SEATS THEM")

    # 1 and 2. What is new: you are inside it, and the read expires on contact.
    assert "This is the one where you are INSIDE it" in rule
    assert "a statement about the INSTANT you act on it" in rule
    assert "not a property of the trade you go on holding" in rule

    # 3. His own words, in the first person.
    assert "IH NARRATES HIMSELF BECOMING THE CROWD" in rule
    assert "AND WE HAD ALREADY BOUGHT HERE TOO" in rule
    assert "the crowd his own read said was not there" in rule

    # 4. It explains v5h rather than repeating it.
    assert "the seat you found empty is the seat you took" in rule
    assert "after it has already been spent" in rule

    # 5. All three refusals survive, with the numbers that make them refusals.
    assert "NOT A TIME STOP" in rule
    assert "correlates POSITIVELY with P&L" in rule
    assert "costing between 8,535 and 53,184" in rule
    assert "NOT A BAN ON RE-ENTERING AFTER A STOP" in rule
    assert "+358 a trade across nine of them" in rule
    assert "NOT A TIGHTER READING" in rule
    assert "forgoes 36,580 of a 55,337 book" in rule

    # 6. The inversion, which is the finding rather than the advice.
    assert "THE INVERSION IS THE FINDING" in rule
    assert "NOT beyond your last entry is the BETTER trade" in rule

    # 7. The honest limit.
    assert "IT WOULD NOT HAVE CHANGED EITHER TRADE" in rule
    assert "not on a claim that it would have saved" in rule


def test_v5h_a_range_makes_the_crowd_read_unpayable():
    """v5h (18 Sep): learned on a day the book was RIGHT and still gave it back.

    Four shorts, every one correctly on the note's flat branch off the computed
    FLAT verdict, inside a 44-point session. The first made +4,466.25; the next
    three netted MINUS 158 between them. Seven ways an edit breaks it:

    1. Losing that this is v5g's general case -- apart they look like two
       moods about quiet markets rather than one rule and its limit.
    2. Losing that it is a limit on the METHOD, stated by its own source. That
       provenance is what stops it being read as generic caution.
    3. Losing "whether you find out the SLs or NOT" -- the claim is that
       correctness is not the variable, which is the surprising part.
    4. Losing IH's own retrospective, which is what makes it a pattern rather
       than one bad day.
    5. Losing the measured asymmetry. "Four trades, small profit" is a
       different and much weaker fact than "the first was the whole day".
    6. Losing the checkable tell. Without the new-extreme test the rule needs
       the model to first agree it is in a range, which is the judgement that
       fails in real time.
    7. Losing the standing-premise clause, which explains why the note could be
       cited four times and why v3q's gate did not catch any of them.
    """
    prompt = build_system_prompt()
    rule = _flat_rule(prompt, "IN A RANGE THE CROWD READ CANNOT PAY")

    # 1 and 2. Its relationship to v5g, and whose limit it is.
    assert "HOWEVER RIGHT IT IS" in rule
    assert "The general case of the rule above" in rule
    assert "the only limit on this whole method stated by the person it came from" in rule
    assert "hunt or follow, correct or not" in rule

    # 3. The surprising claim, in his words.
    assert "IH PUTS THE LIMIT ON HIS OWN METHOD" in rule
    assert "whether you find out the SLs or NOT, the market can give you a loss" in rule
    assert "Correctness is not the variable" in rule

    # 4. His own retrospective on the prior session.
    assert "sellers' SLs were not available there either" in rule
    assert "When the market works inside a RANGE, that is where loss happens" in rule

    # 5. The measured asymmetry, with the numbers that carry it.
    assert "on a day it was RIGHT and still learned this" in rule
    assert "about 44 points (23,292.55 to 23,336)" in rule
    assert "+4,466.25 in fourteen minutes" in rule
    assert "MINUS 158 between them" in rule
    assert "the session simply had nothing left to pay" in rule

    # 6. The checkable tell, and why it is phrased that way.
    assert "is a judgement and your own re-entry is not" in rule
    assert "REQUIRE A NEW EXTREME BEYOND YOUR PREVIOUS EXIT" in rule
    assert "the next trade is a tax rather than a trade" in rule
    assert "the market never made a new low after the first exit" in rule

    # 7. Why a standing premise supplies no freshness, and why v3q missed it.
    assert "A STANDING PREMISE IS NOT A SECOND REASON" in rule
    assert "it selects a SIDE, it does not refresh" in rule
    assert "asks for a nameable NEW trapped crowd, which is vacuous on a follow" in rule
