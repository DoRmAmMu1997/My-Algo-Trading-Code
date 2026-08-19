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


def test_prompt_cap_leaves_room_for_lessons_and_a_note():
    """The cap is a sanity bound, not a budget knowledge must squeeze into.

    It was raised on 2026-07-31 because knowledge alone had reached ~68k against a
    75k cap, leaving too little for lessons plus a pre-open note. Keep provable
    headroom so an ordinary addendum cannot silently disable the agent.
    """
    from sl_hunting_knowledge import MAX_SYSTEM_PROMPT_CHARS

    assembled = len(build_system_prompt() + FINAL_OUTPUT_INSTRUCTION)
    # Worst case the runtime can add: 12 lessons at their own cap, plus a note.
    worst_case_runtime_additions = 12 * 280 + 2_500
    assert assembled + worst_case_runtime_additions < MAX_SYSTEM_PROMPT_CHARS


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
