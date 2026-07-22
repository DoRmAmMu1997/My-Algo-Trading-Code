# SL Hunting — Source Methodology (verbatim reference)

> This is the verbatim text of the strategy author's Google Doc, kept here as the
> agent's **ground-truth reference**. The agent's actual system prompt is the
> *curated* version in `knowledge.py` (`build_system_prompt()`); this file is the
> source it was distilled from, preserved so the curation can always be reviewed
> against the original.
>
> NOTE: the original doc interleaves many chart screenshots (shown below as
> `[image]`). The images are **not** reproduced here — only the prose rules are.
> The agent therefore reasons from the written rules plus deterministic
> tool-computed facts (pivot, fibo, candlestick patterns, structure), not from the
> original chart pictures. (Source doc:
> https://docs.google.com/document/d/1lzUwGFlILcELHsv1X6nMY3PNU5WqkQ6lhaoyfJdYCBs )
>
> **v2 image review (verified):** the full doc (108 pages, ~122 MB) was exported to
> PDF and reviewed — a visual sample of pages spanning every section, plus the
> complete extracted text layer of all 108 pages. The embedded images are
> **illustrative** (annotated TradingView screenshots and whiteboard sketches of the
> setups described in the prose) and contain **no net-new rules**: every page's
> caption is prose already captured below. So nothing was added to the agent's
> knowledge from the images — the text rules here, and the curation in
> `sl_hunting_knowledge.py`, already encode the method.

---

## Daily Trades (tf 1 min)

1. Mark pivot, OHLC of previous day.
2. Don't trade on the first candle.
3. If market comes below pivot, wait for candlestick pattern formation with a confirmation candle (e.g. for an inside-bar / bearish-harami, wait for the confirmation candle to close below the low of the mother candle) for a sell trade.
4. If small-small candles are forming then traders are not interested in that direction and the market may reverse.
5. Wait for the level to come. If a doji forms with a wick above a resistance level followed by a strong confirmation red candle, take a sell trade. Market falls and gives a pullback. During the pullback it doesn't give any bullish pattern followed by confirmation, so no need to fear the slow up-move — continue the trade.

- After that, it breaks pivot and support; people will sell, we'll buy after seeing bullish price action. Here we find an inside-bar candle and then a strong confirmation candle.
- The next entry comes when it breaks the previous high (wicked high above resistance). If the market takes the exact resistance of the previous wick then we would not have taken that. Targets are all the hammer points after which the market went up.
- If the market touches the trendline a third time, we trade in the direction of the trend; here bullish — but if there is no bullish pattern followed by confirmation the trade is avoided.
- After a trendline break, everyone will think to sell; we look to buy after confirmation of a bullish pattern. Where will the candle turn back from? We get it from fibo (50/61/78) levels of the last low–high swing.
- A breakout of a triangle pattern can't be sold if it has taken the exact resistance of a level. After taking the exact resistance, an upward trade activates and cancels the earlier sell activation of the triangle breakout. After it takes the resistance, check the bullish pattern at the fibo level because everyone has sold at the resistance.

General rules:
1. Any long-wicked candle (instead of a doji) can be used for entry after proper confirmation.
2. If a candle breaks a closing, think about an upside trade, especially after a gap up.
3. If a candle breaks a resistance (so it "has to" come down) and a psych level like 25K is within range, it will not come down heavily since a major chunk of money sits on the psych level.
4. If a trendline break occurs to the downside but a psych level is upside, it is more confirmed to buy; the trade can be taken on the first engulfing candle itself without confirmation.
5. Entry off the first candle's high or low, if cut by a wick, can be taken.
6. If the market crosses a resistance, that resistance becomes support; if the market comes back to the support and creates a wick there, it will go up.
7. Simultaneously observe BNF too: if a similar setup forms in BNF we can trade Nifty. E.g. if BNF breaks 55500 we can trade Nifty for a downward setup. If Nifty breaks a triangle upward and BNF is just about to break 55500, wait for it to break that psych level then look for a downward setup in Nifty. While falling the market may pull back from a resistance-turned-support or FVG area, but we watch the candles and wait.
8. Most money is made in a sideways-to-trending market, not in a trending market, because we can't know where a trend will go. A purely sideways market won't make money either.
9. If BNF breaks today's low and Nifty doesn't, the market will go up.
10. The first candle's low and high are used for trapping. Target is the opposite side of the first candle.
11. If a bullish candlestick forms and then the next 2-3 candles form inside it, the pattern remains valid; if confirmation comes later we can take the trade — otherwise the confirmation candle has to form immediately.
12. If the SL is coming out large, take a pullback entry or avoid it.
13. If the market goes fast in one direction and slow in the other, the slow-side reversal is used for SL hunting.
14. Trade fibo levels 50/61/78 — look for a pattern to form there.
15. The market will hunt the SLs of major turning points.
16. A reversal bar consists of two candles; after that we need a confirmation candle.
17. If open and low are together, treat it as a zone, not a single line.
18. If price opens gap-down it tends toward the previous closing.
19. If the market keeps going one way, we don't get an entry. After that small-small candles from retailers form; their SLs get hunted from above coming down.
20. If a reverse ("ulta") fibo is plotted on the first candle, then 161 / 261 levels are reversing levels.
21. After a long move it won't come down the first time; it will take the SL and then come.
22. If price goes up it will fall from a fibo level to book profit.
23. If the market takes support from the pivot, we can take a trade.
24. If the market goes up, then after a correction it will go up again.
25. If at the previous day's low the market forms a bullish pattern, we can enter to buy.
26. An engulfing candle should cover the wicks too.
27. If a candle breaks the pivot and then takes its resistance with a pin bar, we can take a sell entry — pivot-point retest.
28. The next target is the immediate swing; the max target is the day's low.
29. If Nifty is within ~50 points of a psych level and BNF is within ~100 points, the psych level attracts the market.
30. If Nifty takes a resistance and BNF breaks out of some level, the market will fall.
31. If Nifty takes support of pivot and BNF breaks some support (like the day's low), the support break will fail. If the same support line is broken again and the target is not met we can exit.
32. If, after a trend, the market retraces up to a retracement level and then follows the trend, in following sessions don't enter near that retracement level — the market may reverse from there again.
33. There are buyers above a level and sellers below a level.

More:
- Along with OHLC and psych level, also mark yesterday's trendline (may be relevant today) and the fibo levels between yesterday's close and today's open, so the market may reverse at the 50% level — but look for a candlestick pattern there.
- If the market reaches support-turned-resistance and forms a bearish candlestick pattern, trade down; if it starts hovering below that level, look for a bullish pattern.
- If the market breaks a level (lower SL / money is gone) and then forms a hammer followed by a full-body green candle, that confirms a bullish trade — but check target vs SL.
- If the market makes a slow move it may reverse; the market only trends with fast movement so no one can enter.
- After a gap up, wait for it to reach the closing point.
- If the market starts falling after a gap up, it may form an N/M pattern at the 50% fibo and then fall again to reach the closing point. Otherwise if it breaks today's high everyone thinks to buy; the market will go slow there and wait for a bearish setup.
- If the market is going up and there is a double top, that is the target; after breaching the double top the market falls.
- Psychology means: the direction the market is going — do we have retail SLs there? The market only goes to eat retailers' SLs.
- Trendline break after the 4th point → trade upward.
- M-pattern breakdown, support breakdown, trendline breakdown and then activation below the line → the market can't go down now, look for a bullish trade.
- Where a setup forms and then fails, expect a small target there.
- Double-bottom breakdown, SL over → the market will go up.
- First candle's high & low marked: its low gets broken but BNF takes support at that time. Everyone shorts but we buy (psychology confirmation) — but we need candlestick confirmation (technical confirmation). Here we have a reversal pattern plus a confirmation candle.
- Inside-bar candle then a very long confirmation candle making SL 20 pts — at max our SL should be 15 points in spot, so here we compulsorily wait for a pullback. Nobody knows whether it will come, but only if it comes do we take the trade.

---

## Logic

1. Retailer price actions: act opposite to them.
2. Price action starts and expires at S/R levels; after expiring, a new price action may start.
3. The market takes resistance → it goes up; it takes support → it goes down (i.e., a clean break of the level). If it traps by a wick or an immediately-returning candle at S/R, it reverses.
4. Before entering, look for a candlestick pattern AND confirmation. The pattern may consist of multiple candles, but a confirmation candle is still needed after it.
5. The market is a fierce battle — die-or-kill. Retailers come unprepared (money, entry, SL, risk management) and lose; they are greedy and impatient, entering without waiting for the setup and exiting quickly. Insiders come fully prepared, patient, with confidence in their setup; they wait for the setup, the right entry, and exit after the target.
6. Market moves fast in a trend, then moves slow the opposite way to create SLs.
7. At a turning point all SLs are gone, so the market takes a pullback after a turning point.
8. When the market has gone down twice and sellers' SLs are available, the market moves up to take those SLs; how far up it can go is found by fibo on the first candle low–high (the low and high so far).

### Entry and exit candles
1. A long-wicked candle (hammer, etc.) tells us a target and hence exit/entry — money is parked there. The longer the wick, the more money/SL is there.
2. A full-body candle is the confirmation candle for entry confirmation at a reversal point.
3. After a hammer, direction depends on where the full-body confirmation candle closes — above the high or below the low of the hammer. Color does not matter for wicked candles.
4. For engulfing patterns, we need a later confirmation candle after the two engulfing candles. The market goes in the last engulfing candle's color direction. **Color matters.**
5. Inside bar is the opposite of engulfing — again direction is of the second candle, but the confirmation candle should confirm at the end of the mother candle. **Color matters.**
6. Reversal-bar setup: the second candle should be the same length as the first; the market goes in the second candle's direction. Formed at S/R levels. Still need a strong confirmation candle (length may be smaller). **Color matters.**
7. Activation: buyers accumulating, SLs below — so the market can sweep them.

### S/R behavior
- A falling market that stays above a level → buyers' SLs accumulate below the level.
- A rising market slowing below a resistance is inviting sellers; their SLs are above the level. Works at all levels (OHLC, psych level).
- If equal SLs are on both sides, it may trap below and then go up.
- After a breakout, if the market stops, it will come down.
- When taking a level trade, the candlestick pattern must form at the very top (of the level), not in between.
- 4-candle rule: at any level, 4 candles should stop at that level.
- If a level is broken both up and down, it's of no value (not tradeable).
- A falling market taking support at a level means it will go further down; if it makes a bullish pattern above the level it may go up. Exception: the pivot line, and only the first time — there the market may take resistance/support directly.
- Support can be taken at resistance-turned-support. If the move is long, the market may not retrace fully to support; for a small move it will.
- If a candle forms a wick at support, it must be a bullish trade, not bearish.

### NF/BNF comparison
- If both indices take exact support → trade down.
- If both indices break down → trade up.
- If one breaks down/up and the other does not, the break will fail and the one that took support/resistance wins.
- If one supports and the other takes resistance, wait until both align.
- If the market makes a wick at a level, the activation there is considered failed.
- If after a fall the market retraces to a fibo level (50/61/78), it must stop *above* a level, not below (we expect it to keep falling). If it stops below a certain level (maybe a pivot), ignore that logic.

---

## Trendline Logic

- For a trendline's 2nd point to be enabled, the market must retrace by at least 50% of the previous swing; only then can we take the trade at the 3rd point.
- The market just turns back after crossing a high because SLs are over there.
- Take the trendline trade only on the 3rd point, not the 4th/5th/later. After that, think of a break and then trade.
- At the 3rd point we can take the trade if it takes support and also if it breaks support. From the 3rd point onward we trade only on a trendline break.
- At the 3rd point, if it breaks and then goes below 78% of the swing previous to the broken level, even then we can take a bullish trade (it may do activation below the level).
- On a trendline, the up direction has fast movement and the down direction (pullback) has slow movement with wicks.
- If a psych level or closing point lies in our trade direction, the success rate increases. A psych point attracts the market within ~50 points (Nifty) and ~100 points (BNF), but there's no such distance for closing-point attraction.
- In a sideways market, first wait for the market to break one direction (to collect SLs from that side); then trade the *opposite* direction from where it first breaks.
- A successful trendline first breaks the double top (to take all SLs and show bullishness), then falls. It may make a big red candle to entice us, but wait for a proper bearish candlestick setup.
- After the fall, be alert at all SL levels (trendline points): activation above → falls further; activation below → may go up (slower move = better) before eventually falling, then we may sell again at the 50% fibo retracement.
- Targets below the trendline are doji/wicked patterns at the points on the trendline; be alert at fibo levels from the lowest to the highest point of the trendline.
- A trendline breakdown fails if it hasn't trapped the double-top buyers.
- After a trendline breakdown, don't trade down (everyone is trading there) — wait for an upside opportunity.
- If the first two points are close in time, treat them as one point and count the next as the next point.
- If two points are close in time they count as one; later we get the "4th" point.

---

## Trade Setups (summary)
1. After a gap down, let it come to the closing and find a bearish pattern.
2. After a gap up, let it come to the closing and find a bullish pattern.
3. At a trendline: support or break from the 3rd point in trend direction; from the 4th point on, only after a trendline break.
4. Double top / double bottom break: take a reversal trade after the break.
5. OHLC, pivot and psych-level trades.
6. W and M pattern trades.
7. Fibo level (50/61/78) trades.

Pivot can be used as S/R directly or after retracement. Pivot changes if the timeframe changes.

---

## Gap up / Gap down (normal case ≈ 50 pt Nifty, 200 pt BNF)
1. The closing point holds both buyers' and sellers' SLs, so the closing attracts the market.
2. After a gap up, look for a downward setup; if you don't get a trade, let the market reach the closing point then look for an up trade. Mirror for gap down.
- Why does the market gap? If there's no SL on the down side, it has to open gap-up or go up after opening.
- After a gap down, sellers are in profit (a ₹10 option becomes ₹200); as soon as the market moves up they exit, so the operator can't get their SLs — so the operator shows them more profit (greed), and slowly moves up to create and take SLs. People think the market falls after a gap down; we think the opposite.
- As soon as it gets a level it moves fast upward.
- In a gap down, while moving to the closing, if the market gets a level it may retrace ~50% then rise again to reach the closing.
- If it forms a big candle after opening and then stays a while, it may come down, cut everyone's SL, then move up. Mark the first candle's high & low for the trade.
- If it gets a level and activates below it, take an up trade.
- If the market creates a range at opening, it may give a false breakdown then go up to the closing.
- Down trade after breaking up is risky because the lower target is small — especially in morning time.
- Logical: if the first move is fast and the second move is too slow to recover, the first move will continue.
- If it takes support at any OHLC level, it will fall again.

## Big Gap up / Gap down (≈ 150 pt Nifty, 400-500 pt BNF)
- The retailer has huge profit; if the market moves down the operator gets nothing (retail just exits). So the market keeps going up to give the retailer confidence, creates SLs, traps, and falls a little, again creates SLs, traps and falls — that's how it falls.
- Better to trade upside from the 50% level — a big fall again is difficult. The operator may take down SLs the next day or later.
- As soon as the market breaks the trendline it falls sharply.

---

## W and M Patterns
- Breakout trading is wrong (it may turn back any time). The correct trade is when it *activates below the neckline*. If a psych level is upside it goes there, otherwise it goes to the swing where it started falling from. If a closing point is below, the market won't fall directly — it traps first, then falls.
- In a lower W pattern we can enter at the low too because the W's first leg is already broken. Be alert at the trendline break; don't fear a resistance at the trendline because it will activate there and break out.
- When Nifty is breaking the W's neckline and BNF is taking pivot resistance, both fall and the Nifty breakout fails.
- If Nifty activation is above a level and BNF activation is below a level, wait until both give the same side signal.
- M-pattern works the same way as W. There are three kinds of M patterns.

---

## Fibo
1. Whatever logic applies at normal S/R levels applies to fibo levels too (activation, taking support/resistance, etc.).
2. The market should fall ~20% before a turning point forms and we can apply fibo.
3. Levels are only 50/61/78. If it takes support at a level it will fall and break it; if it moves up from in between levels it may go up. While going up, resistance at a level → it keeps moving up; if it breaks a level and sustains it may fall. A wick at a level → it goes up. Find the candlestick pattern first. If Nifty is taking support below 50 and BNF has broken down a level, we can take a trade.
4. Taking support at the 78% level is valid — the market may move up from there. Take confirmation from the second index. The 78% level forms in an FVG area; even without an FVG we trust the 78% level more.
5. If it breaks the 100% level, all SLs are over, so the chance is to reverse.
6. For 3-4 day charts, the 38% level may also work.
7. Fibo is applied to the recent breakout swing.
8. Once price does a 100% retracement, shift the lower "1" (100% point) to the further-down turning point.
9. If two OHLC levels appear at the same price it becomes a strong level (lots of SLs). Likewise if a first swing's 100% and a later swing's 78% are at the same point.
10. 1/2/3 are 50% fibo levels of three swings; level 3 is strongest because most SLs are exhausted by then.
11. If the bullish move is fast and the retracement on fibo is slow, trade upward only (probability rises). But if the second move is also fast, still trade upward.
12. For 161 / 261 fibo targets, place fibo on the first correction swing in a series of swings; then 161 and 261 give the targets.
13. Usually the market reverses from the 100% level; once it leaves into a domain with no levels, 161 and 261 guide us. After breaking 161, to go up after taking 161 support it must make a trap at 161.
14. Nifty average momentum ≈ 200-250 a day. If Nifty has already made that, look for a level where it can fall.
15. On a big move (e.g. 261), trade a bearish candle cautiously — V-shape recovery isn't usual; it will trap first then fall.

---

## Pivot Point
1. Pivot = (H + L + C) / 3. Major SLs are here. Above pivot = buyers' market; below = sellers' market.
2. Pivot is a neutral/balance point — it can go either way (like a car on a hill crest). Support → up; resistance → down. Activation works here, and direct S/R too — but a candlestick must form.
3. Pivot can be exact support/resistance and is the strongest S/R.
4. Below pivot → sell; above pivot → buy (don't overthink today's direction). But if the two indices are on opposite sides of their pivots, treat pivot as a normal S/R.

### Opening setups
- **Setup 1 (Opening, tf 5 min):** market opens and forms support at the pivot with a hammer (or any bullish candle — doji, reversal bar) and the next candle is a confirmation candle that closes above the hammer. Trade; target is above where the market fell from yesterday. Valid even if both indices are taking support. Usable on 1 min too, but 5 min has higher accuracy (also forex/crypto use 5/15 min).
  - Better not to trade until the candle touches the pivot; even a full-body candle after the first red candle has its SL below the pivot. After the candle touches the pivot, the range is the low of that candle to the high of the first; the next candles build the range; a candle closing above the range is the bullish trade (SL below that breakout candle). Target = turning points before that (slow movement). Watch for activation before/after the target. A lower wick on the breakout candle means it trapped the in-between candles; otherwise (1 min) it must be a full-body candle. If the breakout candle's low is below earlier candles, it becomes the base pattern and needs another confirmation candle.
- **Setup 2 (Red candle at opening):** the confirmation candle closes below the first wicked red candle (any wicked candle — body up/down/middle, color irrelevant for wicked candles) or another bearish pattern. SL above the first candle or the pivot.
- **Setup 2 (Gap up):** wait till the market reaches the pivot. It reaches via smaller successive candles, makes a doji at the pivot, then a confirmation candle. If there's a level at the pivot and it traps that level (or the pivot), even better. Even if gap-down with closing upside, the market will go up to the closing after bouncing from the pivot.
- **Setup 4 (Opens below pivot):** wait till it reaches the pivot (we know it's below, so it will fall). 2nd candle touches the pivot as a doji; if the 3rd candle doesn't trap the 2nd we'd trade when the 4th closes below the 2nd — but if the 3rd traps the 2nd, the 4th must close below the low of the 3rd. If many candles form the hi/lo range and the market reaches much lower then gives the confirmation candle below the range, don't trade (SL is large and above the pivot). Smaller candles = sellers losing interest; the market may reverse from any level (psych level). A closing below the range is valid for any level. Two hammers with opposite wicks → the market traverses one direction, reverses at that hammer's wick, then hunts the other hammer's SL; keep SL a little beyond to escape SL hunting.
- **Setup 5 (>50 pt gap up):** can trade other levels between pivot and opening too; but to trade at the pivot, wait for the market to reach it. The pivot can also be a target (market becomes neutral for another trade). Support-taking at the pivot is one-time; after that treat the pivot as a normal S/R. If a closing aligns with the pivot, the market goes up after pivot support (strength enhanced). If a psych level is below the pivot within ~50/100 pts, let the market decide (SLs are down too — it goes down to hunt those first).
- **Setup 6 (Large gap-down below pivot):** market opens below yesterday's low; trade upside with target just before the pivot (other targets just before levels). After reaching the pivot a new price action starts. If price breaches the pivot, sellers push it down and a hammer forms; if the next candle closes below it, the sell trade is on — but if another red candle forms and price goes back above the pivot, we need a closing below that candle's wick. Target = 50% fibo or where the price action started.
- **Setup 7 (Reversal sell trade):** price opens above pivot, comes to the pivot, a candle touches the pivot, another closes above it → enter buy with SL below the pivot. If SL hits, immediately reverse to sell. Don't exit till SL hits (unless SL is in the system) — a green hammer at the pivot shows the below-pivot SLs are gone; a green confirmation above the hammer = a buy entry too. At a pivot break, both-side entries are possible. Once a red candle closes below the green hammer, enter the sell trade. Risk exists (not a full body, some wick), but it's mitigated since price did activation above the pivot then came down.
- **Setup 8 (Gap-down reversal):** if the 3rd candle didn't close below the first candle's wick (it was green), don't sell — the setup is cancelled. After another red candle, need a fresh closing below it (which the next candles didn't do). A 7th hammer candle traps earlier candles → buyers/sellers trapped → goes buy side. Then a green candle closes above the 2nd candle's wick = the entry candle. Main SL below the low (if no level there); now SL is just below the breakout candle (it broke the pivot). If instead a long-upper-wick hammer closing below the pivot forms and the next red candle closes below it, we'd trade down (≈ activation below the pivot). If the market formed a W pattern yesterday, assume activation below the neckline → buy trade.
- **Setup 9 (Predicts tomorrow's gap):** if the market has been opening below the pivot for ≥2-3 days and then opens above it, the market is bullish; reverse for above→below. Count only bodies vs the pivot. The more days on one side, the more bullish/bearish when it opens on the other side. Same logic for weeks (invest that week). Pivots differ by timeframe: day (15 min), week (4 H), month (D), year. We don't use the standard pivot S/R that everyone uses. Above the Day-level pivot we can start investing (it goes till it finishes the previous top's SLs). Market is bullish above pivot, bearish below.

---

## Misc
- Niftybees (Nippon Nifty 50 ETF) is for long-term investment — no time decay; can hold if we don't get the target.
- If the market keeps going one way, at some point it turns back and finds support at a fibo or psych level.
- To get upside targets, plot fibo on the down-swing; 161 and 261 give the targets.
- When putting fibo in reverse, look at the first reverse swing; look for reversals at 161 and 261; no trade in between even with confirmation.

---

## Video addendum — reading retail positioning from the opening gap (v3)

> Source: a separate SL-hunting video (timestamps in parentheses). Distilled into the
> curated `RETAIL_POSITIONING` section of `sl_hunting_knowledge.py`. Verbatim notes:

Identifying retail positioning is not about traditional indicators like support/resistance.
Instead, gauge retail participation by:

- **Analysing the opening gap** — the primary indicator of sentiment. A gap-up often
  suggests retail traders do NOT have significant active positions (caught off guard),
  which lets you follow the existing momentum (2:45–3:00, 4:56–5:06).
- **Evaluating price action and traps** — instead of chart patterns, look for retail
  traps. In a flat-to-gap-down scenario the market is more likely trapping retail who
  entered on previous panic selling; identifying these traps lets you target their stop
  losses (6:38–7:15).
- **Observing market psychology** — traders develop a bias (urge to sell the top after a
  long rally, or buy when the market looks sideways). By observing where retail likely
  placed stops after these moves, you can see where smart money will move the market
  (8:09–8:21, 11:50–12:15).
- **Contextualising momentum** — don't automatically trade against every big momentum
  candle. Read the overall context: whether the market has moved sharply (retail likely
  trapped) or has been stagnant (retail not yet participated) (1:28–2:02).

On opening gaps specifically (02:45–03:12):
- **Gap-up openings** — a gap-up often indicates retail has no significant active
  positions; because they aren't heavily positioned the market is less likely to be
  trapped, so you can follow the prevailing momentum (02:48–03:01).
- **Flat-to-gap-down openings** — prime environments for traps. A preceding period of
  negative sentiment / panic selling lets the market trap retail who are positioned the
  wrong way; this is an opportunity to target their stop losses and trade opposite the
  initial panic (03:02–03:10, 06:38–07:15).

---

## Video addendum — Bank Nifty live-trading methodology (v3a)

> Source: 9 "Intraday Hunter" YouTube videos (8 live BankNIFTY option-trading sessions +
> 1 weekly-analysis lesson). The audio is Hindi and raw transcripts were not retrievable
> in our environment, so the methodology below was **distilled via YouTube's built-in
> "Ask" (Gemini) summaries** of each video (prompted for concrete rules with timestamps),
> not from a verbatim transcript. Treat it as a **secondary AI summary, operator-reviewable**.
> 4 videos were captured in full (both market regimes); the rest restated the same method
> with no net-new rule. General lessons were merged into the curated knowledge sections;
> BankNIFTY-specific behaviour went into the new `BNF_SPECIFIC` section (advisory context
> for the cross-index read — NIFTY-only execution is unchanged).
>
> Videos: `s41N7OS17Wk` (Weekly Charts Analysis — general), and the live BankNIFTY sessions
> `gMu0DU4HJ00`, `1e14YWvOtzs`, `LmO-Y1XzqgY`, `O_PHs9q1QqA`, `G9HR80PLK8E`, `a3jih441RZo`,
> `XHIlEHikp6k`, `o0a5gq5i_Mo`.

**General method (merged into the existing sections):**
- **Bias is read from the open.** Gap-up / immediate positive momentum ⇒ buyers not
  threatened, no trapped shorts to hunt ⇒ follow the trend, don't fade. A flat open that
  then struggles to push up ⇒ had it meant to rise it would have gapped up; the hesitant
  flat open that lures buyers to "support" expecting a breakout is a trap for them ⇒ short
  and hunt their SLs. Gap-down after panic ⇒ trapped shorts ⇒ hunt upward.
- **"Closing price" (previous-day close) is the pivotal intraday level** — both S/R and the
  trade's invalidation: a long dies if price falls back to it; indices stalling at
  closing-price resistance without a decisive breakout is the short tell.
- **Confirmation is also behavioural** (complements, never replaces, the candle rule): enter
  when price holds without aggressive selling (long) or fails to break out and stalls
  (short); accept a slightly worse price rather than miss the anticipated move.
- **Stops = premise-invalidation + a hard intraday loss cap;** never hold a loser hoping for
  a reversal — you cannot wait indefinitely intraday.
- **Time-decay discipline (option BUYER):** a bought option bleeds premium when the market
  goes sideways — most sharply near/at expiry; if the move doesn't come quickly, exit.
- **Targets = the hunted-SL liquidity zone; book on weakness** (momentum failure / leading
  index stalls / opposing reversal), not a fixed number.
- **Don't over-focus on accuracy** — the edge is the positioning read + discipline.

**Bank Nifty-specific (new `BNF_SPECIFIC` section — advisory; the agent still trades NIFTY only):**
- **Triple-index read:** watch BankNIFTY + NIFTY + Sensex together; a thesis needs momentum
  confirmed across them, and concurrent rejection across all three invalidates it.
- **BankNIFTY is the "major index"** that sets the base bias (NIFTY/Sensex confirm); exit when
  BankNIFTY weakens vs the others, especially if the weakest one starts reversing.
- **Prioritise the index whose expiry is that day** (Sensex/NIFTY) — it concentrates the
  action and theta.
- **Round-number levels weigh more on BankNIFTY** (its larger point range): the round
  "…500"/"…000" levels are prime trap/breakout magnets where breakout-buyers get trapped.
- (Context only, not an agent rule:) the trader executes a **basket across BankNIFTY + NIFTY +
  Sensex legs concurrently** — noted for realism; our agent trades NIFTY only.

---

## Video addendum — daily "Analysis" videos review (v3b)

> Diligence record (no knowledge change). I reviewed every "Analysis"-titled Intraday Hunter
> video from the previous ~2 weeks (15–29 Jun 2026) for **net-new durable method**, via YouTube's
> built-in "Ask"/Gemini panel (same path as v3a). **Outcome: confirmatory only — nothing net-new
> was added.** The daily clips are short (~2 min) **pre-market prediction calls**: their content is
> *ephemeral* (that day's specific support/resistance levels and bias), and the durable themes they
> touch — gap-up/gap-down read, "avoid flat markets", expiry handling, round-number/S-R levels,
> SL-hunting, volatility — are **already** in the knowledge base (see `RETAIL_POSITIONING`,
> `LEVELS_AND_PIVOT`, `RISK`, `BNF_SPECIFIC`, `BNF_CROSS_CONFIRMATION`). No rule is added that a
> video did not actually state (no fabrication), so the knowledge sections are unchanged for v3b.

Videos reviewed (id — date — Ask/Gemini signal):
- `I2BGDZIEc4c` — 29 Jun — gap-down caution, "closing price" importance, breakout-trap skepticism (known)
- `st8p4CkP8mo` — 25 Jun — Sensex-expiry day; Ask panel not offered on this clip; on-screen chart is day-specific S/R only
- `ZXQZy735-Fo` — 24 Jun — "trading plan", "SENSEX outlook", "SL hunting for intraday" (known)
- `GCpBLoj3DSw` — 23 Jun — "why avoid trading when flat", "prepare for Nifty expiry" (known)
- `0Pq2Arc7gRo` — 22 Jun — "outlook for Nifty", "trade Bank Nifty & Sensex", "SL hunting strategy" (known)
- `Z2cVRE3sa6s` — 19 Jun — "trend analysis", "today's plan", "key S/R levels" (ephemeral)
- `2aDHbVBT6gM` — 18 Jun — "outlook", "key levels", "why avoid trading on a gap down" (known)
- `aBvETyWqKSQ` — 17 Jun — "strategy", "handle sudden gaps", "key S/R levels" (known/ephemeral)
- `OKpfb0Nky2I` — 16 Jun — same daily-prediction format (ephemeral)
- `lN4qRl5VQgs` — 15 Jun — "strategy for the day", "key S/R levels", "handle volatility" (ephemeral)

Also: the **"Analysis Q&A | Most Asked Questions"** (`7a3dAL7mBJY`) is **members-only / inaccessible**;
its preview lists method topics (e.g. "how to predict Monday's direction when retail exits on Friday",
"regain confidence after a big loss", "60% profit booking") — a potential net-new "weekend / Friday-exit
→ Monday-open" refinement — but the answers are paywalled, so nothing is added from it. (If the operator
shares those notes, the Friday→Monday concept can be folded into `RETAIL_POSITIONING` the same way.)

---

## Video addendum — live gap-up session + same-day journal review (v3c)

> Source: "Live Bank Nifty Option Trading" (`WhfVxV0h5bo`), the 2026-07-02 live session,
> reviewed the same day against the agent's own decision log and trade journal. Unlike v3a
> (Ask/Gemini summaries), this one was distilled from the **verbatim Hindi auto-transcript**
> captured via YouTube's transcript panel — primary-source provenance. Timestamps in
> parentheses are video time (session opens at 0:06 ≈ 09:15 IST).

**The trade (what the agent was benchmarked against):**
- Market opens gap-up; "no big rejection, only small green-to-red candles" (0:06–0:17).
- LONG (call-side) basket built ~1 minute after open: BankNifty 1170 qty (0:24), Sensex
  900 qty (0:43), NIFTY 1365 qty (0:53) — triple-index, with-gap.
- Reasoning (1:08–2:27): the market moved up but with NO BIG momentum, so few traders
  could have bought; whatever few longs exist have SLs below the closing price —
  unreachable without a major rejection. "We can't target those buyers directly. Had it
  opened flat or gap-down we would hunt them; but on a gap-up we should go WITH the
  market" → call-side trade. NIFTY's 24,000 round level was the same read (3:18–3:25).
- Risk framing (3:03–3:18): the danger is a rejection that drags price back to the round
  number — that's where the loss limit would be crossed. No big rejection = stay.
- Catalyst (3:52): "today Sensex has expiry — overall positive momentum is possible."
- Booked ~9 minutes in when momentum arrived across Sensex + NIFTY and BankNifty printed
  2-3 strong candles (5:59–6:49).
- Discipline riffs: don't make 2-3 trades in 5-10 minutes; give the market time after a
  trade (4:09–4:49); put effort into the trade that's working (7:06–7:21); "greed has a
  limit" — book, don't sit (8:32–8:40).

**Same-day journal/decisions review (2026-07-02; 149 decisions, 3 trades, all SHORT):**
- **T1** 09:28 `psych_round_number_bearish_engulfing_fade`, -19.0 pts (-1.11R): faded a
  ~60-pt gap-grind 13 minutes in by declaring "late breakout longs trapped" — exactly the
  read the video refutes (small momentum = nobody trapped). The agent's own 09:26 HOLD had
  said "classic gap-up momentum, no reason to fade blindly"; two bars later it faded on
  the first confirmed bearish pattern. → the TRAP-DENSITY TEST now in `RETAIL_POSITIONING`.
- **T2** 09:46 `double_top_doji_confirmation_reversal`, +42.35 pts (+3.04R): the CORRECT
  fade — a ~170-pt run to 24159 trapped real breakout buyers at a double top. Validates
  the same test from the winning side (extended run → real trapped SLs → hunt works).
- **T3** 10:33 `shooting_star_evening_star_fib_rejection`, -16.55 pts (-1.15R): entered
  SHORT while `cross_index` read "both_at_resistance → bias UP" — the knowledge already
  said "disagrees → prefer HOLD" and was overridden. → the two cautions appended to
  `BNF_CROSS_CONFIRMATION`. (T1's cross read, "both_at_support → bias down" during a
  gap-up rally, was the opposite failure: a stale yesterday-levels verdict — same fix.)
- A valid with-gap LONG trigger existed inside the agent's own rules — its 09:33 exit
  reasoning cites a bullish doji (09:26) confirmed above 24117.85 at 09:31 — and was
  never taken. → the GAP-UP FIRST-TRADE bullet in `RETAIL_POSITIONING`.

**Why the agent could not capture the video's trade (root causes):**
1. It was asleep: the worker's `trading_start` used the shared 09:25 default
   (`_signal_gen_ops`), so its first decision landed 09:26:16 vs the video's ~09:16 entry.
   The operator has since set the start to 09:15 via the existing
   `SL_HUNTING_TRADING_START_HOUR/MINUTE` knobs in `.env` — a config change, not code.
2. Even awake, the method had no opening-drive entry: every entry required a reversal
   pattern + confirmation at a level, and the first candle was untradeable — the video's
   trade is a positioning/context trade, structurally outside those rules.
3. No trap-density notion — see T1 vs T2 above.
4. Cross-index verdicts nudged/allowed the wrong side — see T3/T1 above.

**Knowledge changes made for v3c (all prose, no logic):**
- `RETAIL_POSITIONING`: TRAP-DENSITY TEST + GAP-UP MORNING → FIRST TRADE WITH THE GAP
  (incl. "a stopped-out fade on a gap-up morning is evidence of gap-and-go — don't re-fade").
- New `OPENING_DRIVE` section: the ONE scoped exception to pattern+confirmation — with-gap
  LONG only, first ~15 minutes, clear gap-up above prev close + round number, entry only
  after the first 1-min candle closes, no full-body green-to-red rejection, behavioural
  confirmation substitutes, stop below first-candle/opening-range low (size auto-shrinks
  via the risk budget), exit immediately on a major rejection, book on weakness. No
  gap-down mirror (flat/gap-down stays a hunt-UP trap per the existing read).
- `ROLE`, `LEVELS_AND_PIVOT`, `DECISION_RULES`: the blanket "pattern only / never the
  first candle" statements now carry the scoped OPENING DRIVE exception (never DURING the
  forming first candle; from its close onward, only under ALL its conditions).
- `BNF_CROSS_CONFIRMATION`: two cautions (stale early verdicts vs the opening-gap context;
  an opposing verdict is a real vote → HOLD unless textbook).
- `BNF_SPECIFIC`: expiry-day index = extra fuel for with-gap momentum on a gap morning.

---

## Video addendum — 2-week verbatim sweep (v3d)

> Source: every Intraday Hunter video for the 18 Jun – 2 Jul 2026 sessions, re-extracted from
> the **verbatim Hindi auto-transcripts** (YouTube transcript panel → page-text capture, the
> v3c method) — including videos v3a/v3b had only covered via lossy Ask/Gemini summaries.
> 15 of 18 in-window videos captured; `st8p4CkP8mo` has no transcript, the weekly
> `s41N7OS17Wk` never loads its panel (v3a's Gemini coverage stands), `WhfVxV0h5bo` was v3c.

Videos (id — session — type — outcome/signal):
- `G9HR80PLK8E` — 18 Jun — live — WIN: post-flush flat-open long; "flushed buyers don't
  return"; BNF-moves-first entry; slow-continuous momentum read.
- `O_PHs9q1QqA` — 19 Jun — live — small LOSS: bought a BIG gap-down against an up-streak
  ("only the gap itself tells you to sell"); cut when the smaller-gap index (BNF) refused to
  join the recovery → gap-size asymmetry lesson.
- `Z2cVRE3sa6s` — for 19 Jun — plan — crowd-QUANTITY read (drip-buyers not huntable).
- `0Pq2Arc7gRo` — for 22 Jun — plan — gap-size gradation (small counter-gap ≈ flat).
- `LmO-Y1XzqgY` — 23 Jun — live — LOSS: faded a flat-open recovery after a SIDEWAYS period
  (positioning unclear = no crowd); held to limit, cut on time + expiry decay.
- `ZXQZy735-Fo` — for 24 Jun — plan — the conditional plan (gap-up → sell-side; flat/gap-down
  → buy-side) after a big down day.
- `J64qDUp2M88` — 24 Jun — live — WIN: executed that plan; flat open + first positive momentum
  → quick triple-index long; "had it gapped up we'd have SOLD it".
- `st8p4CkP8mo` — for 25 Jun — plan — no transcript.
- `1e14YWvOtzs` — 25 Jun — live — WIN: gap-up continuing strength on Sensex expiry → follow;
  sized up the expiring index; exited when the weakest index stalled.
- `s41N7OS17Wk` — 28 Jun — weekly — transcript unavailable.
- `I2BGDZIEc4c` — for 29 Jun — plan — breakout-failure day → trap read; produced the losing
  29 Jun long (plans can be wrong; direction-first still governs).
- `gMu0DU4HJ00` — 29 Jun — live — LOSS: trap-CONSTRUCTION leg premise (rally to re-add buyers
  after a flush); cut when all three rejected together.
- `FXugPeqs2HQ` — for 30 Jun — plan — KEY CONTRAST: close freshly below the round number →
  sellers' SLs unreachable → flat/gap-down = go WITH the selling; only a gap-up activates the
  hunt (→ SL-REACHABILITY TEST).
- `SKgchmcArt0` — 30 Jun — live — LOSS: textbook cross-index divergence entry failed because
  the day's DIRECTION was down (→ direction-first hierarchy); open exactly ON the round number
  = ambiguous; admits loss-streak bias.
- `yVFhGqGCjMc` — for 1 Jul — plan — mindset-based plan after a choppy day (lower conviction).
- `Jj9yec-QDvI` — 1 Jul — live — WIN: flat open after 2-3 down days → hunt the comfortable
  sellers up; gap-down would make old sellers SAFE (fresh herd traps itself); gap-up after a
  selling streak = "no trust".
- `kW5phlWuMKM` — for 2 Jul — plan — reads 1 Jul as an over-the-day seller trap; "gap-up →
  go with, buy-side" = the v3c live trade the agent missed.
- `2vO3onLbhPc` — for 3 Jul — plan — day-specific (ephemeral), captured for the record.

**Knowledge changes (v3d, all prose):**
- `PSYCHOLOGY`: trap-CONSTRUCTION leg (post-flush single momentum leg — capture and leave).
- `RETAIL_POSITIONING`: READ THE GAP AGAINST THE PRIOR DAYS (continuation vs big-counter-gap
  lure vs small-gap gradation); MULTI-DAY ACCUMULATION (flat-open hunt vs sideways = no crowd;
  drip-crowds not huntable); SL-REACHABILITY TEST.
- `OPENING_DRIVE`: Variant B — flat-open seller-hunt long after an extended multi-day down
  move (same first-candle/no-major-rejection discipline; still no short / no gap-down variant).
- `BNF_CROSS_CONFIRMATION`: direction-first hierarchy (divergence setups are entry-timing
  tools; 30 Jun live loss).
- `BNF_SPECIFIC`: gap-size asymmetry across the three indices + BNF-moves-first entry tell.
- `RISK`: loss caps in trade units; two-rejections/third-momentum heuristic; early-adverse =
  wrong direction on one-directional (expiry) days; slow-continuous vs fast momentum quality;
  loss-streak "recovery trade" bias.
- `DECISION_RULES`: rule 7 — no-plan zones (abstaining is a valid plan).
- Test marker: `test_system_prompt_has_v3d_conditional_gap_knowledge`.

---

## Video addendum — 3 Jul live match + weekly/lecture sweep (v3e)

> Sources: the 3 Jul live session `BvkCsOgkigI` (**verbatim transcript**, 11:25) matched against
> the agent's same-day journal; the "Weekly Recap Step 2 Step" `yRITNBXsAXY` (3 May session,
> 16:13) and the "STOP Revenge Trading" lecture `wBHAjFxfXJE` (14 Jun, 15:39) via YouTube's
> Ask/Gemini panel (secondary AI summary — the transcript panel never populates on >12-minute
> videos in our environment, re-verified, and the timedtext endpoint stays gated). The 28 Jun
> weekly `s41N7OS17Wk` remains covered by v3a's Gemini pass; the other three long lectures
> (`YRTuOxYDKhw`, `dVGgbkCtCGM`, `QXMuGzdu0CE`) are deferred to a future Ask-panel pass.

**3 Jul: IH vs the agent (the first session with v3c+v3d + the 09:15 start live):**
- IH: HUGE gap-up (Sensex/NIFTY large, BNF small). Waited out the first momentum; went SHORT on
  two stacked reads: after a huge gap NO SLs exist nearby — the premise is the MINDSET trap on
  fresh buyers who add into the first post-gap push; and BNF's bounce off an EXACT closing-price
  touch would attract only buyers ("the market only works where BOTH sides want to engage") →
  unsustainable → fade. Booked a NORMAL target after the market held too long ("holding candles
  INVITE sellers; a late breakdown attracts followers and reverses").
- Agent (5 trades, +31.7 pts, +Rs.1,053): declined the opening drive on the first-candle
  full-body rejection (v3c behaving exactly as designed — IH waited too); LONG the flush-to-24300
  reclaim +27.05; theta stall-exit +1.65; SHORT the rebuilt double-top +19 (the SAME premise as
  IH's short); double-bottom long -7.7 (stall exit); re-fade of a buyers'-market dip -8.3
  (stopped — v3d's don't-re-fade counsel plus the new staleness/participation reads all argue
  that skip).
- Verdict: strong convergence; the deltas feeding v3e are the participation principle, the
  huge-gap mindset-trap, and setup staleness.

**Knowledge changes (v3e, all prose):**
- `PSYCHOLOGY`: BOTH-SIDES PARTICIPATION (+ exact-touch support fragility vs small-rejection
  go-with tell).
- `RETAIL_POSITIONING`: HUGE-gap nuance appended to the conditional-gap block (no nearby SLs
  exist; fade the first post-gap push as a mindset trap, strict loss limit).
- `BNF_SPECIFIC`: THIRD-INDEX LAG (two indices breaking a shared level does not commit the
  third; its refusal is a divergence signal) — from the weekly recap (09:48-10:40).
- `RISK`: SETUP STALENESS (late breaks attract followers then reverse → normal target only) +
  loss-recovery discipline (no immediate re-entry after a loss; recover big losses across
  multiple trades; the "one last trade" trap).
- NOT encoded: the lecture's "observe-only first 1-1.5 hours" rule — it is beginner-discipline
  framing that contradicts the operator's opening-drive edge (IH himself trades the open).
- Test marker: `test_system_prompt_has_v3e_participation_knowledge`.

---

## Video addendum - July 4-8 transcript sweep + agent match (v3f)

> Sources: Intraday Hunter channel uploads published 4 Jul 2026 through 8 Jul 2026,
> extracted via YouTube's transcript panel where available, then matched against
> `Backtest Outputs/sl_hunting_decisions.jsonl` and `Backtest Outputs/sl_hunting_journal.jsonl`.
> The 4 Jul lecture `lxY9snUinyg` advertised Hindi ASR captions but yielded no transcript
> segments in the UI/timedtext path, so no knowledge is added from it.

Videos captured for the trade-method sweep:
- `F9APQ4MnAcA` - prediction for 6 Jul.
- `ohxweLy3H2Q` - live BankNIFTY session on 6 Jul.
- `P3dFob-ZHtw` - prediction for 7 Jul.
- `pEXtxlA1u-k` - live BankNIFTY session on 7 Jul.
- `DTd4Mtz1ppg` - prediction for 8 Jul.
- `4oV5tP8nzv4` - live BankNIFTY session on 8 Jul.
- `_y-hk-sl-aQ` - prediction for 9 Jul; captured for provenance, but no 2026-07-09
  agent decision rows existed in the log, so there is no same-day agent match yet.

**Transcript + agent match ledger:**
- **6 Jul:** IH took CALL/long after huge first-minute positive momentum, then waited
  for a pullback/rejection read: prior Friday was negative and a holiday followed, so
  buyers had not built a reachable SL base; early rejection lured sellers, making the
  long the operator-side trade. The agent matched the first trade (`opening_drive_gapup_continuation`
  at 09:17, booked profit), but then immediately took four short fades. Those shorts
  contradicted the still-live opening thesis: after profit booking, a small pullback
  was more likely a seller lure unless price spent enough time/space recruiting fresh
  buyers first.
- **7 Jul:** IH took PUT/short because several positive sessions with only shallow
  retracement meant buyers could still be holding; a modest gap-up/flat-to-gap-down
  open could hunt that buyer inventory. The agent did not cleanly take this trade:
  the decision log mostly held around the open, and the journal contains timeout /
  `agent_error` noise including an opposite long. Only the repeated method mismatch
  is encoded as knowledge: buyer inventory after a shallow up-streak can be the short
  target, and the agent must not auto-read every modest gap-up as continuation.
- **8 Jul:** IH took PUT/short after prior breakdown + retracement + continuation:
  the prior put buyers had likely booked profit and left, so they were not the day's
  target; the flat/gap-down open plus incomplete recovery failed to reclaim the close
  / round area, allowing put-side continuation. The agent stayed flat, first waiting
  for the old bullish gap-down reversal pattern, then becoming usage-limited. The
  durable knowledge change is the target-booked crowd test plus a narrow gap-down
  continuation-short exception; the usage-limit behaviour is not a prompt rule.

**Knowledge changes (v3f, all prose):**
- `RETAIL_POSITIONING`: BUYER-INVENTORY FADE; TARGET-BOOKED crowd test; direct-momentum
  / current-session trap reset after the old crowd has been paid/flushed.
- `OPENING_DRIVE`: replaces the absolute no-short/no-gap-down wording with the strict
  GAP-DOWN CONTINUATION SHORT exception: only narrow/moderate gap-down, sellers not
  huntable, early recovery fails below close/round/opening range, no bullish reclaim.
- `RISK`: NO INSTANT FLIP after a correct opening/day-direction trade has been booked;
  plus the 2-3 hour open-thesis timeout for stalled option-buyer premises.
- `BNF_SPECIFIC`: MASKED BNF LAG - temporary BankNIFTY weakness can keep NIFTY/Sensex
  breakout buyers away; it invalidates only when it actually breaks the trade premise.
- Test marker: `test_system_prompt_has_v3f_transcript_match_knowledge`.

---

## Video addendum - 2026 transcript sweep through 9 Jul (v3g)

> Sources: Intraday Hunter public channel uploads from 1 Jan 2026 through 9 Jul 2026.
> The extraction ledger found 274 in-window channel entries: 250 public metadata rows,
> 185 public transcripts extracted successfully, 65 public transcript payloads that
> stayed blocked/empty, and 24 inaccessible/member-only rows. The blocked public rows
> returned YouTube transcript `429` / attestation / empty-panel failures even after a
> signed-in Chrome attempt; they are deferred to a possible third commit. No knowledge
> below is derived from blocked or member-only videos.

Sweep result:
- The successfully extracted daily/live clips overwhelmingly confirm v3a-v3f: gap
  context, target-booked crowds, current-session trap reset, opening-drive nuance,
  both-sides participation, BNF lag caution, and no-instant-flip discipline.
- The main net-new durable source was the public long-form lesson `ywHZfvKsy5Q`
  (8 Mar 2026), "90% of Traders Ignore This Previous Day Chart Strategy".
- The July 6 live transcript `ohxweLy3H2Q` independently supports the same
  holiday/carry-risk theme: after a negative Friday and holiday gap, assume the
  obvious old buyer crowd may not be holding unless the current chart proves it.

**Net-new method distilled from the 185 extracted transcripts:**
- Previous-chart linkage: connect today's read to yesterday's chart, but ask what
  the prior chart already paid, flushed, or made unreachable. After a big gap or a
  completed target, prioritize the new chart's fresh trap over stale assumptions.
- Event / holiday participation: known news shocks, Fridays, weekends, and
  multi-day holidays can remove one side from the risk pool. Do not hunt a crowd
  that likely exited or avoided large overnight/news risk.
- Constructed-base continuation: after a large event-driven move, direct
  continuation that would attract only one obvious side is weaker. For continuation,
  expect the market to build supports, resistances, bases, or retests that bring
  both buyers and sellers back in before the next SL hunt.
- Weekend / holiday carry-risk: non-trading gaps reduce the reliability of assumed
  large retail inventory; use current-session price action to prove the crowd exists
  before targeting its stops.

**Knowledge changes (v3g, all prose):**
- `PSYCHOLOGY`: EVENT / HOLIDAY PARTICIPATION and CONSTRUCTED-BASE CONTINUATION.
- `RETAIL_POSITIONING`: PREVIOUS-CHART LINKAGE and WEEKEND / HOLIDAY CARRY-RISK.
- Not encoded: any rule from the 65 blocked public videos or 24 inaccessible rows.
- Test marker: `test_system_prompt_has_v3g_full_2026_sweep_knowledge`.

---

## Video addendum - remaining blocked public transcripts via NoteGPT fallback (v3h)

> Sources: the 65 public 2026 Intraday Hunter videos that v3g could not extract
> from YouTube's transcript panel. The operator approved NoteGPT
> (`https://notegpt.io/youtube-transcript-generator`) as a third-party fallback
> source for this pass. The temporary extraction ledger is
> `%TEMP%\intradayhunter-2026-transcripts\notegpt_remaining_ui_2026.jsonl`.

Fallback extraction result:
- 64 of 65 previously blocked public videos recovered transcript text through
  the NoteGPT UI fallback.
- `st8p4CkP8mo` remains unresolved: the YouTube UI had no transcript button, and
  NoteGPT returned `message: no transcript` with no usable segments.
- Combined 2026 public coverage is now 249 of 250 public videos: 185 direct
  YouTube-panel transcripts + 64 NoteGPT fallback transcripts.
- The 24 inaccessible/member-only rows remain excluded. No knowledge below is
  derived from those rows or from the unresolved `st8p4CkP8mo` video.

High-signal recovered sources:
- `lxY9snUinyg` (4 Jul hidden psychology): choose only direct, high-clarity
  unique trades; do not convert uncertain reads into trades.
- `YRTuOxYDKhw` (7 Jun position holding): hold a valid winning trade while the
  premise remains intact instead of cutting it to chase a weaker second trade.
- `wBHAjFxfXJE` (14 Jun revenge trading): no daily-income pressure, no immediate
  recovery trade after a loss, and no revenge loop after one failed setup.
- `ZLpWNw34zGQ` (22 Mar candlestick/timeframe): match timeframe to the question;
  broader context guides the read, entry timeframe controls execution.
- `0lWj6kaDpFU` (4 Jan quick decisions): fast execution is acceptable only when
  the plan is already defined: trapped crowd, reason to move, invalidation, target.
- Confirmatory recovered weekly/live sessions: `s41N7OS17Wk`, `dVGgbkCtCGM`,
  `QXMuGzdu0CE`, `yRITNBXsAXY`, and `7NDj21y5K60`.

**Net-new method distilled from the 64 recovered fallback transcripts:**
- Unique-trade filter: the market is not fixed. The agent should trade only
  obvious, direct setups where the target crowd, level, direction, invalidation,
  and target can be named before entry; guess trades are HOLD.
- Profit-hold: once a valid trade is working, do not exit merely to hunt a
  second-best or third-best setup. Hold until target, stall/theta, or premise
  invalidation.
- Timeframe fit: use higher/multi-day context for broader strength, weakness,
  and inventory; use the 1-minute/opening chart for execution. A noisy small
  candle should not override the broader read by itself.
- Plan-of-execution: quick decisions are allowed only when the trade was already
  pre-defined. If the agent cannot state who is trapped, why price can move,
  where invalidation is, and where profit is expected, it must HOLD.
- No daily-income pressure: a quiet/no-trade day is valid. Forcing a trade
  because "today must pay" is a revenge/over-trading seed.
- Post-loss speed limit: after a loss, disable quick-decision mode and wait for
  a fresh, deliberate, high-quality setup instead of trying to recover immediately.

**Knowledge changes (v3h, all prose):**
- `PSYCHOLOGY`: UNIQUE-TRADE FILTER.
- `LEVELS_AND_PIVOT`: TIMEFRAME FIT.
- `RISK`: PROFIT-HOLD, NO DAILY-INCOME PRESSURE, and POST-LOSS SPEED LIMIT.
- `DECISION_RULES`: PLAN-OF-EXECUTION.
- Test marker: `test_system_prompt_has_v3h_remaining_transcript_knowledge`.

---

## Video addendum - 10 Jul live gap-up seller-hunt session (v3i)

> Source: `sImrqns7fBo` (10 Jul 2026, "Live Bank Nifty Option Trading", 10:35).
> Full Hindi auto-transcript captured the same morning from YouTube's transcript
> panel (the ≤12-min recipe worked first try). The 1:51 daily prediction clip
> (`LoT91UMHeVo`) was not mined, per the v3b finding that the daily prediction
> clips are ephemeral day-calls whose durable themes are already captured.

Session summary and match:
- After the prior day's big selling with only a weak recovery, IH read the crowd
  as sellers built on the retracement with no buyer inventory. On the
  flat-to-gap-up open he went LONG calls on all three indices right at open
  (BankNIFTY 1170 qty, Sensex 900, NIFTY 1365), sat through early drawdown while
  no major rejection printed, and booked an "average target" in profit.
- Textbook execution of EXISTING knowledge — OPENING DRIVE gap-up branch /
  variant B seller-hunt, MULTI-DAY ACCUMULATION, TRAP-DENSITY and
  SL-REACHABILITY tests, round-number magnets (BNF_SPECIFIC). Strong
  confirmation; nothing to change in the day-direction read.

**Net-new method distilled:**
- Premium non-confirmation exit: he booked the AVERAGE target instead of
  stretching for the breakout specifically because option premiums were not
  rising with the spot move on a NON-expiry day (Sensex legs lagging), with a
  BankNIFTY round number approaching — "after seeing this profit, watching it
  become a loss is not right."
- R:R-bait at round-number rejections: small rejections at a round number during
  a with-trend grind are the market MANUFACTURING put trades whose SL/target
  ratio "looks right" (SL just past the round number, target at the prior low)
  but has no premise; those freshly built stops fuel the next leg up. Round-number
  "resistance" during momentum is an invitation, not an inability to cross.
- Confirmed but deliberately NOT re-encoded: the averaging-destroys-capital
  lecture (the agent structurally cannot average — one position at a time and
  the order tool rejects entries while positioned) and the seller-hunt long
  itself (already OPENING DRIVE variant B / gap-up branch).

**Knowledge changes (v3i, all prose):**
- `RETAIL_POSITIONING`: R:R-BAIT AT ROUND-NUMBER REJECTIONS.
- `RISK`: PREMIUM NON-CONFIRMATION.
- Test marker: `test_system_prompt_has_v3i_premium_rr_knowledge`.

## Video addendum - 13-14 Jul gap-down sessions + averaging trap (v3j)

> Sources (full Hindi auto-transcripts from YouTube's transcript panel):
> - `qjz6uAM81Jg` (12 Jul 2026, "Prediction For 13 JULY 2026", 1:42)
> - `OvqxvtVbZFU` (13 Jul 2026, "Live Bank Nifty Option Trading", 9:20)
> - `xssPyxt65Mc` (13 Jul 2026, "Prediction For 14 JULY 2026", 1:53)
> - `DuaQYSrYK2U` (14 Jul 2026, "Live Bank Nifty Option Trading", 11:43 — NIFTY expiry)
>
> Unlike previous addenda, this one is cross-checked against the agent's OWN journal
> (`Backtest Outputs/sl_hunting_journal.jsonl`, rows 17-19, all 2026-07-14). That
> match is what makes the net-new below evidence-backed rather than speculative.

Session summary:
- **13 Jul** — big gap-down. IH refused to sell it: a direct fall is unlikely because
  everyone seeing the gap-down will sell, so the market must first take THEIR stops.
  He bought CALLs, rode the recovery, booked. Textbook EXISTING knowledge (gap-down
  seller-hunt long).
- **14 Jul (expiry)** — gap-down AGAIN, but yesterday's recovery had recruited BUYERS.
  IH bought PUTs, and crucially **waited for the market to push up first** before
  entering ("if we sell directly here the market will trap us"). One trade, booked,
  stopped for the day.

**Agent vs IH on 14 Jul** (IH: 1 trade; agent: 3 trades, net +Rs.3,793):

| # | Agent trade | Time | Result |
|---|---|---|---|
| 1 | LONG `gap_down_trap_flush_reversal` | 09:20-09:24 | +Rs.1,488 |
| 2 | SHORT `pivot_double_top_evening_star` | 09:40-09:55 | +Rs.3,176 (1.97R) |
| 3 | SHORT `shooting_star_doubletop_fibo50_reversal` | 10:04-10:04 | **-Rs.871 (AI_STOP in 5s)** |

- **Trade 2 IS IH's trade — full match.** The agent's own reasoning ("a confirmed
  evening-star ... trapped buyers who chased that reclaim") names the same crowd and
  direction as IH, and it booked the average target on stall. Strong confirmation;
  nothing changed for this case.
- **Trade 1 diverged from IH's read.** The agent called the gap-down "a trap for
  starved sellers" and went LONG; IH read the same open as a trap for the BUYERS that
  yesterday's recovery recruited, and was short from the pre-open. `BUYER-INVENTORY
  FADE` was already in the prompt but never fired because nothing triggered on the
  TWO-DAY sequence (gap-down -> strong recovery -> next-day gap-down). The agent
  profited on the opposite premise, then had to flip into trade 2.
- **Trade 3 is the exact mistake IH's outro warns against, and it is the only loser.**
  Nine minutes after booking the move, the agent re-shorted the SAME exhausted move on
  a smaller/later pattern into a stalling expiry tape and was stopped out in 5 seconds.
  `NO INSTANT FLIP` did not catch it (same direction, not a flip). It also overrode an
  opposing `cross_index` verdict by calling it "a stale mechanical ... label anchored to
  yesterday's resistance levels" — but that escape hatch was written for the OPENING
  HOUR, and this was 10:04 at confidence 6 (not "textbook").

**Net-new method distilled:**
- Averaging trap: the counter-bounce after a gap traps a crowd is BAIT to make them
  AVERAGE DOWN; the real move comes only AFTER they add ("if it fell directly they'd
  run away quickly — by making them average first, the market extracts far more").
  Carries the two-day trigger the agent lacked, and the entry-timing rule: never enter
  at the gap extreme, wait for the bounce — the bounce is not a threat to the fade, it
  IS the setup. Corroborated by his 13 Jul self-critique ("the trade isn't wrong, the
  entry was early").
- Move-exhaustion: once the thesis's move is booked and momentum has stalled, the
  thesis is SPENT — do not re-enter the same direction chasing its tail. On expiry days
  the market then builds a wide RANGE and chops both sides; take what you got and stop
  ("we won't make 10 days' profit in one day"). Expiry is context, never a premise on
  its own — a deliberate counterweight to the existing "expiry = extra FUEL" note.
- Cross-index "stale verdict" escape hatch scoped to the opening hour: outside it, an
  opposing `cross_index` verdict is a veto, not a footnote.

**Confirmed but deliberately NOT re-encoded (already present, executed correctly):**
the gap-down seller-hunt long, the closing-price gate on targeting a crowd, the
trap-density read, "don't capture both directions in one day" (NO INSTANT FLIP),
loss-limit discipline, and `DECISION_RULES` #8 (a gap-down that falls directly is
unreadable -> no trade). Dropped on purpose: "the counter-gap recovery must arrive
FAST" (it would collide with the existing with-trend "SLOW-but-CONTINUOUS is the
sustainable kind" rule, and the journal shows no need — trade 1's recovery was fast
and booked in 4 minutes); the small-capital/psychology advice (agent sizing is
automatic at ~Rs.2500 risk); and the prediction clips' level tables (ephemeral day
calls, per the v3b finding).

**Knowledge changes (v3j, all prose):**
- `RETAIL_POSITIONING`: AVERAGING TRAP (mechanism + two-day trigger + entry timing).
- `RISK`: MOVE-EXHAUSTION — ONE MOVE PER THESIS (incl. EXPIRY-DAY RANGE).
- `BNF_CROSS_CONFIRMATION`: SCOPE OF THIS "STALE" ESCAPE HATCH + opposing-verdict veto.
- Test marker: `test_system_prompt_has_v3j_averaging_trap_knowledge`.

## Video addendum - 15 Jul gap-up seller-hunt + flat-open participation gate (v3k)

> Sources (full Hindi auto-transcripts from YouTube's transcript panel):
> - `40j_l5DtwS4` (14 Jul 2026 evening, "Prediction For 15 JULY 2026", 1:48)
> - `ciQ19XPXoXk` (15 Jul 2026, "Live Bank Nifty Option Trading", 8:35)
>
> Cross-checked against the agent's journal (`Backtest Outputs/sl_hunting_journal.jsonl`,
> row 20, 2026-07-15).

Session summary:
- Pre-open plan (prediction clip): after 14 Jul's gap-down + persistent selling that
  closed BankNIFTY below the round number, sellers may be seated — **gap-up OR
  gap-down -> hunt those sellers (buy-side); FLAT -> "we cannot follow that structure",
  go WITH the market (sell-side)**, because the fearful sellers may never have really
  sized in ("the market made no big momentum; if no scared traders are seated, benefit
  by going with the market") and a flat open parks price on the closing-point support.
- Live session: mild gap-up + immediate positive momentum -> instant with-gap CALL
  basket (BankNIFTY 1170 / Sensex 900 / NIFTY 1430 qty), sharp continuation with no
  retracement, booked ~140 pts at the round-number test ("greed has a limit — we
  tested the round number directly, cut and go"). Textbook EXISTING knowledge:
  OPENING DRIVE gap-up branch, prior-day seller inventory, round-number booking, and
  the momentum-quality read (his "fast momentum invites greedy buyers -> retracement
  risk; small candles would be safer" is the RISK momentum-quality rule verbatim).

**Agent vs IH on 15 Jul** (IH: 1 trade, ~+140 pts; agent: 1 trade):
- The agent SHORTED at 09:32 (`shooting_star`/`evening_star`/`inside_bar` at the 24200
  psych level, entry 24195) on a gap-up-and-go morning with price +123 pts above pivot
  and `cross_index` reading "up_context" — the exact fade that GAP-UP MORNING /
  TRAP-DENSITY / R:R-BAIT already prohibit ("a bearish pattern at a psych level is
  NOT, by itself, a short on a gap-up morning"). IH rode the same morning long.
- The exit redeemed it: 5 minutes later the agent named "gap-up-and-go continuation
  confirmed", cut both legs ~10 pts before the stop, and the basket closed POSITIVE
  (+Rs.832, the BankNIFTY mirror leg outran the NIFTY leg's -7.9 pts).
- **Journal-fidelity finding (operational, not knowledge):** row 20 carries
  `setup: "agent_error"`, `confidence: 0`, `reasoning: "Agent call timed out;
  holding."` for a REAL trade. The LLM call that placed the short TIMED OUT after the
  order tool had already fired, so the worker's timeout placeholder — not the model's
  reasoning — became the journal row. The entry cannot be audited, and the reflection
  coach would learn from placeholder text. Worth a separate fix (journal the order
  tool's `reason` argument as a fallback); no knowledge change can address it.

**Net-new method distilled:**
- FLAT-OPEN PARTICIPATION GATE: the flat-open SL-hunt requires the prior crowd to have
  really participated. After a WEAK-momentum down day (hesitant selling, no big move),
  a flat open puts nobody in pain and leaves the closing-point support in the way —
  plan WITH the prior direction there, while a gap in EITHER direction re-arms the
  hunt (gap-up pressures the sellers; gap-down pays them into complacency and the
  recovery hunts their stops). Scopes the blanket "FLAT or GAP-DOWN -> look UP" rule
  and complements MULTI-DAY ACCUMULATION ("a crowd that only TRICKLED in is not
  huntable").

**Confirmed but deliberately NOT re-encoded (already present):** the with-gap opening
drive and its behavioural confirmation, prior-day seller-inventory read, round-number
booking without greed, momentum-quality (fast spike -> retracement risk), and the
round-number-magnet notes. The live session contained no averaging/psychology segment.

**Knowledge changes (v3k, all prose):**
- `RETAIL_POSITIONING`: FLAT-OPEN PARTICIPATION GATE.
- Test marker: `test_system_prompt_has_v3k_flat_open_gate_knowledge`.

## Video addendum - 16 Jul split-gap session + closing-point hold test (v3l)

> Sources (full Hindi auto-transcripts from YouTube's transcript panel):
> - `1uB29qR9V0A` (15 Jul 2026 evening, "Prediction For 16 JULY 2026", 2:07)
> - `ojc_NGulszU` (16 Jul 2026, "Live Bank Nifty Option Trading", 9:51)
>
> Cross-checked against the agent's journal (`Backtest Outputs/sl_hunting_journal.jsonl`,
> rows 21-22, 2026-07-16).
>
> EXTRACTION NOTE: both 16 Jul watch pages loaded as SKELETON placeholders — zero
> `ytd-engagement-panel-section-list-renderer` nodes, no "Show transcript" button, no
> recommendations sidebar — across reloads AND a fresh tab. Fix that worked:
> `resize_window` (e.g. 1400x900) THEN `location.reload()`; the forced re-layout makes
> the panels hydrate (9 of them), after which the normal recipe applies. Captions
> existed the whole time (`hi/asr` in `ytInitialPlayerResponse`), so an absent button
> is a hydration failure, NOT a missing transcript.

Session summary:
- Pre-open plan (prediction clip): the prior day rejected but **held above the closing
  price**, so "not many are sitting short" — plan was flat/gap-down -> sell-side; a
  direct gap-up -> buy-side (a gap-up would put any seated sellers in trouble).
- Live open: **a SPLIT gap** — Sensex and NIFTY opened with a mild GAP-UP while
  **BankNIFTY opened FLAT, right at its own closing price**. IH read the flat major
  index as the honest tell: if BNF took support there and cleared 58,000, "only buyers
  would come" and the down-move would be dead; instead he expected the small trap and a
  fall. He bought PUTs across all three (BNF 57800+57700 PE, Sensex 900 qty — Sensex
  expiry that day, NIFTY 1365 qty).
- Core reasoning (the day's whole thesis): the prior rejection **never broke the
  closing point**, so whoever sold it booked the momentum and left rather than holding
  overnight -> **no seller inventory** -> there are no seller SLs to hunt upward ->
  therefore FOLLOW the selling down instead of hunting sellers up. He stated the
  converse explicitly: had the market broken down and then HELD below, sellers WOULD be
  seated, and then the market would reject and run them UP.
- Exit: not a stop or a target. BankNIFTY — the index he expected to LEAD — failed to
  lead, and all three indices began drifting down together in small steps. He cut,
  because an evenly shared, visible move "invites sellers", and a freshly recruited
  seller crowd is exactly what gets hunted next ("then the market can suddenly turn").

**Agent vs IH on 16 Jul** (IH: 1 trade, booked; agent: 2 trades, net -Rs.709):

| # | Agent trade | Time | Result |
|---|---|---|---|
| 1 | LONG `opening_drive_gapup_continuation` | 09:17-09:22 | +2.3 pts but **-Rs.1,333** |
| 2 | SHORT `gap_up_after_selldown_buyer_trap_short` | 09:28-09:42 | +2.45 pts, **+Rs.624** |

- **Trade 1 is the divergence, and it is the loser.** The agent fired the OPENING DRIVE
  gap-up branch on NIFTY's own mild gap (24142 vs 24073.45 prev close) — "retail is
  largely un-positioned so there's no SL-hunt available; the with-gap continuation is
  the trade" — while IH was reading the SAME open as a short because **BankNIFTY had no
  gap at all**. Nothing in OPENING_DRIVE required the gap to be SHARED, and
  `cross_index` returned "neutral"/"none", so the split gap never registered. Note the
  basket cost: +2.3 NIFTY points still lost Rs.1,333 because the BankNIFTY mirror leg —
  the very index that was flat — went against it.
- **Trade 2 converged with IH's direction** (short) and made money, though via a
  different premise (agent: untrustworthy gap-up recruiting fresh buyers; IH: no seller
  inventory -> follow the selling). The agent then exited on a stall at 09:42 while IH
  held the same direction longer for a better target.
- Exit discipline was sound in both rows (row 1 cut in ~5 min on a confirmed rejection,
  well before the stop) — the loss came from the ENTRY premise, not the management.

**Net-new method distilled:**
- CLOSING-POINT HOLD TEST: whether an overnight crowd exists at all is answered by one
  question — did the prior rejection/selling BREAK the closing point and HOLD beyond it?
  Broke-and-held -> the crowd is seated with live SLs -> huntable (look the other way).
  Never broke it -> they booked and left -> no inventory -> FOLLOW the prevailing move.
  Sharper and more mechanical than the existing TARGET-BOOKED test (which keys off
  breakdown+retracement+continuation); this keys off a single named level.
- OPENING DRIVE — SHARED-GAP REQUIREMENT: the gap-up branch's premise ("nobody is
  positioned, so there is no hunt available") is FALSE when the major index (BankNIFTY)
  opened FLAT at its own closing point while NIFTY gapped. A flat major index beside a
  gapped NIFTY is a SHORT tell — GAP-SIZE ASYMMETRY ("the smaller-gap index is the
  tell") at its strongest, a zero-gap index. Directly prevents the agent's -Rs.1,333.
- Leader-fails-to-lead exit MECHANISM (folded into the existing BNF_SPECIFIC bullet,
  which already prescribed the exit): an evenly shared, small-step move across all three
  indices RECRUITS the crowd onto your own side, and a freshly recruited crowd is the
  next hunt target — once your side IS the crowd, the edge is gone. Scoped explicitly to
  the leader failing to lead, so it does not collide with the RISK rule that
  leader-led "SLOW-but-CONTINUOUS" momentum is the sustainable kind.

**Confirmed but deliberately NOT re-encoded (already present):** the one-directional-day
read ("either profit or loss, the market will pick a side — don't dream it comes back"),
premise-invalidation exits, average-target booking without greed, expiry-day context, and
round-number magnets.

**Knowledge changes (v3l, all prose):**
- `RETAIL_POSITIONING`: CLOSING-POINT HOLD TEST.
- `OPENING_DRIVE`: SHARED-GAP REQUIREMENT condition.
- `BNF_SPECIFIC`: the "why" clause on the leader-fails-to-lead exit.
- Test marker: `test_system_prompt_has_v3l_closing_point_and_shared_gap_knowledge`.

## Video addendum - 17 Jul flat-open loss day + gift-gap read (v3m)

> Sources (full Hindi auto-transcripts from YouTube's transcript panel):
> - `hGWenJz7Us4` (16 Jul 2026 evening, "Prediction For 17 JULY 2026", 2:43)
> - `xTwmjkvkrQQ` (17 Jul 2026, "Live Bank Nifty Option Trading", 8:07)
>
> Cross-checked against the agent's journal (no 2026-07-17 rows) and the runner log
> (`Dependencies/log_files/nifty_multi_strategy_master_front_test_dhanhq.log`).

Session summary — **IH's first LOSING day in this series**:
- Pre-open plan (prediction clip): the prior day again had small momentum with the
  closing point uncrossed, and this time BOTH sides could be thinly present. The
  conditional was two-sided: **gap-up -> the (thin) buyers feel "it's all mine" and
  sit -> trap forms for THEM -> sell-side setups; gap-down -> same trap for sellers
  -> buy-side setups; FLAT -> "whom do we target?" -> nobody -> go WITH the market
  (sell-side)**.
- Live session: all three indices opened FLAT. Per the plan he sold the first
  positive push (BNF 57500 PE 1170 qty, Sensex 900, NIFTY 24100 PE 1365), naming the
  invalidation BEFORE entry ("this resistance must not cross; BankNIFTY must not go
  up"). The market broke out upward instead — Sensex/NIFTY first, and when BankNIFTY
  joined, he CUT at his loss limit. The discipline segment is the day's real
  content: "if we're wrong, the market does what we did NOT plan for"; "can we
  control the market? Not at all — what's in our control is the limit"; "don't book
  a small loss and hurriedly build a CALL trade — you may flip and the market turns
  back down"; "no averaging"; "the brain only works right when the trade is going
  right — when it's wrong, look at the limit and leave."

**Agent vs IH on 17 Jul — not comparable on premise:** IH took 1 trade (a loss); the
agent took ZERO trades because the market-data health gate blocked ALL entries (paper
included) through the opening window. Log timeline: runner start 08:09 pre-open ->
newest bar is yesterday's ("stale 60,032s") -> every worker logs "Blocking new
entries" (192 lines that day) and fires the empty 30-s flatten; a worker restart
~09:22 triggered a second flatten at 09:24 ("stale 120.4s"); entry gates reopened
only 09:24-09:29 — after IH's ~09:16 entry. The agent's LLM ran (decision-cost lines
from 09:16) but no order could land. The block accidentally "saved" the agent from a
likely losing day — luck, not design. Fixed alongside this addendum: the stale-feed
entry block and 30-s auto square-off are now scoped to LIVE workers only.

**Net-new method distilled:**
- GIFT-GAP AFTER A NOBODY'S-CROWD DAY: after a small-momentum day whose closing point
  was never crossed, both sides are thin; a gap in EITHER direction is a gift that
  traps its recipient (fade the gap side on confirmation), and a flat open means
  there is nobody to hunt — go with the drift. Generalises the v3k FLAT-OPEN
  PARTICIPATION GATE (seller-crowd case) to the two-sided thin-inventory case.
- NO INSTANT FLIP extended to the LOSING side: booking a small loss to immediately
  reverse into the breakout that is hurting you is the classic whipsaw; exit at the
  limit / invalidation and let POST-LOSS SPEED LIMIT govern the next entry.

**Confirmed but deliberately NOT re-encoded (already present):** limit-based loss
exits, no averaging (structurally impossible for the agent), "control only the
loss", the binary one-directional-day read, and the thin-crowd/closing-point entry
premise itself (v3k/v3l — IH's entry premise WAS the encoded rule; the trade still
lost, which is the "sound process, losing trade" case the knowledge already
accepts). Note: IH's sell-the-first-push entry WITHOUT pattern+confirmation is what
lost — evidence FOR keeping the agent's stricter mandatory-confirmation entry rule;
nothing was loosened.

**Knowledge changes (v3m, all prose):**
- `RETAIL_POSITIONING`: GIFT-GAP AFTER A NOBODY'S-CROWD DAY.
- `RISK`: NO INSTANT FLIP extended with the losing-side panic-flip ban.
- Test marker: `test_system_prompt_has_v3m_gift_gap_and_loss_flip_knowledge`.

## Video addendum - 19 Jul closed-chart lecture (v3n)

> Source: `OVs8-y2HTl8` (19 Jul 2026, "The Secret of the Closed Chart | Every Trader
> Must Know", 22:22). Full Hindi auto-transcript captured from the transcript panel.
>
> NO agent-vs-IH journal comparison for this addendum: 19 Jul was a Sunday, markets
> shut, so there are no 2026-07-19 journal rows and no trade to compare. The last
> trading day (Fri 17 Jul) is covered by v3m.

What this video is: not a trade session but a **week-in-review teaching session**. IH
walks the CLOSED chart of each day of 13-17 Jul and explains how the next day's
conditional plan was built from it — **including a self-diagnosis of the 17 Jul losing
trade**. It therefore exposes the reasoning BEHIND the conditional gap plans distilled
piecemeal since v3d, and it patches a gap that cost him real money.

**Net-new method distilled:**
- RECRUITMENT HISTORY, NOT CHART SHAPE: he shows two consecutive days whose charts are
  near-identical ("even point-wise both are almost the same") and asks why the plans
  were OPPOSITE. The discriminator is what the prior move RECRUITED — a chart that was
  negative and then reversed up recruits NO buyers (the move ran against the mood and
  turned too suddenly for them to join), whereas an already-positive chart that goes
  positive AGAIN does seat buyers ("traders slowly start taking risk"). The law: a
  FIRST, reversal-type move recruits nobody; the SECOND consecutive same-direction day
  seats the crowd. This turns the existing PREVIOUS-CHART LINKAGE instruction into an
  actual test, and is the general principle underneath v3j's two-day AVERAGING TRAP
  trigger (its gap-down-specific case).
- ONE BREAKDOWN, NOT TWO — the rule whose absence cost him 17 Jul. Diagnosing that
  loss: "when the market breaks down one level, normally the market does NOT break the
  second level". After the 500-level breakdown, sellers had joined progressively and
  were SEATED, so the correct plan was the seller-seated template (gap-up -> buy,
  gap-down -> buy, flat -> sell); he instead read buyers as available, planned
  flat -> sell, and lost. Stated corollary: even if the breakdown did NOT seat
  sellers, it AT MINIMUM evicted the buyers — so after a level breakdown, buyers are
  never the target. That asymmetry alone rules out the buyer-hunt when the seated side
  is uncertain.
- BREAK-WITHOUT-MOMENTUM — **a correction to v3l**. He shows a 58,000 breakout that
  then produced no momentum for ~2 hours and concludes "even if someone bought, they
  would not have held; we don't need to target the buyers". As originally written, the
  v3l CLOSING-POINT HOLD TEST said break-AND-held-beyond => crowd seated, which would
  misread exactly this case. The "held beyond" arm now additionally requires that the
  break produced real MOMENTUM; a break that idles beyond the level for hours seats
  nobody and falls into the no-inventory arm instead.

**Confirmed but deliberately NOT re-encoded (already present):** crowd-opposite
psychology ("if the crowd is buying, the one who makes money sells"); the
gap-up/gap-down/flat conditional framework (v3d/v3k/v3l/v3m); a big gap-down changing
the whole structure; counter-trend risk-takers booking their momentum and leaving
rather than holding (already TARGET-BOOKED); and the loss discipline ("accept the
mistake, take the loss, don't sit insisting the market must fall") from v3m.

Also NOT encoded, deliberately: the lecture's meta-thesis that learning happens on the
CLOSED chart and that tomorrow's plan is built from it overnight. True for a human
studying after hours, but the agent decides per completed 1-minute bar and performs no
overnight study — it is not actionable for it. (The reflection coach in
`sl_hunting_coach.py` is the closest analogue and already operates off-loop.)

**Knowledge changes (v3n, all prose):**
- `RETAIL_POSITIONING`: RECRUITMENT HISTORY, NOT CHART SHAPE (after PREVIOUS-CHART
  LINKAGE, which it sharpens).
- `RETAIL_POSITIONING`: ONE BREAKDOWN, NOT TWO (beside the other inventory-existence
  tests).
- `RETAIL_POSITIONING`: momentum requirement added to the CLOSING-POINT HOLD TEST's
  "BROKE it and held beyond" arm (self-correction to v3l).
- Test marker: `test_system_prompt_has_v3n_closed_chart_knowledge`.

## Video addendum - 20-21 Jul flush-day follow + solo-leader veto (v3o)

> Sources (Hindi auto-transcripts from the transcript panel):
> - `IrOy9cExWd8` (19 Jul 2026 evening, "Prediction For 20 JULY 2026", full transcript)
> - `9_eSvyc2VFE` (20 Jul 2026, "Live Bank Nifty Option Trading", full transcript)
> - `0HnoI5CMaFE` (21 Jul 2026, "Live Bank Nifty Option Trading", full transcript)
> - `xrficoYHDSE` (20 Jul 2026 evening, "Prediction For 21 JULY 2026") — the transcript
>   panel NEVER populated on this one clip (hidden across a reload, a resize+reload,
>   and a fresh tab, while the same recipe worked on the other three the same hour).
>   Its plan is reconstructed from IH's own verbatim recap inside the 21 Jul live
>   session ("we had kept a simple plan: gap-up → positive side, gap-down → negative
>   side, flat → the market first adds buyers, then may fall") plus a viewer comment
>   corroborating the flat/sell lean. Treated as secondary, clearly-flagged sourcing.
>
> Cross-checked against journal rows 23-24 (2026-07-20) and 25-27 (2026-07-21).

Session summaries:
- **20 Jul (WIN):** news-driven HUGE gap-down against the prior positive week. His
  pre-open plan explicitly did NOT apply ("if it falls directly, no plan can be made")
  — so he waited for the open, watched the first momentum, read the gap-down + negative
  news as GREED recruiting fresh current-session sellers, and bought CALLs for the
  seller-flush retracement near the closing-price/round-number confluence. Booked the
  recovery. He named the alternative regime honestly: "when the market has to make a
  1000-2000 point move it keeps falling without retracement — if that's this market,
  we take the loss." Textbook EXISTING knowledge (current-session trap reset, AVERAGING
  TRAP mechanism, huge-gap rule with strict loss limit).
- **21 Jul (LOSS):** flat open after the both-ways 20 Jul. BankNIFTY started moving
  first and he entered CALLs QUICKLY on the first push ("we made the trade a bit early
  ... because BankNIFTY is where our position is biggest"). NIFTY/Sensex never crossed
  their closing points; BNF retraced through 58,000; he cut at his limit. Self-diagnosis:
  "the greed trap is exactly what caught US — I trusted BankNIFTY too much; I thought
  NIFTY and Sensex would stay mildly negative while BNF gave the momentum, but BNF came
  down too."

**Agent vs IH (the first days the comparison flips):**

| Day | IH | Agent |
|---|---|---|
| 20 Jul | WIN (call basket on the seller flush) | 2 trades, net **-Rs.946** |
| 21 Jul | LOSS (early flat-open long) | 3 SHORTS, net **+Rs.5,090** |

- 20 Jul row 23: the agent had the RIGHT direction (hammer-reversal LONG at 09:26) but
  exited after 56 seconds with `exit_reason: "placeholder"` — un-auditable — then
  2.5 minutes later FLIPPED short (row 24, basket -Rs.711). That flip is precisely the
  v3m losing-side flip ban; the rule merged to main on 17 Jul but the running build
  may not have carried it yet. Evidence FOR the rule, nothing new to encode.
- 21 Jul: the agent was short all day and its row-25 premise — "the operator's
  constructed trap catching the breakout-chasing longs recruited by the sharp recovery
  leg" — is LITERALLY the trap IH fell into as one of those early longs. The BankNIFTY
  mirror legs printed most of the basket profit as BNF broke 58,000. First clear
  agent-beats-IH day; credit to existing trap-construction knowledge, not luck of
  direction alone (all three entries had confirmed patterns + aligned cross-index).
- **Journal-fidelity note:** rows 23 and 27 carry `exit_reason: "placeholder"` on real
  closed trades — a SECOND variant of the journaling gap (the first was the timeout
  placeholder on 2026-07-15 row 20). Already tracked as a spawned fix task; recorded
  here for provenance.

**Net-new method distilled:**
- BOTH-WAYS FLUSH DAY → FOLLOW THE OPENING: a second, distinct way a day ends with
  nobody seated. After a VIOLENT both-ways session (big gap + real momentum in both
  directions) nobody holds overnight — but unlike the thin small-momentum day, there
  is nothing to fade: no side is being "rewarded" against a held position. The plan
  collapses to "as the opening, so the plan" (gap-up → buy-side, gap-down → sell-side,
  flat → the market must RECRUIT a crowd before it can move against them, so the
  flat-open first push is the recruitment bait — do not chase it). Distinguisher vs
  GIFT-GAP: ask WHY nobody is seated — thin → fade the gap; flushed → follow the
  opening type. (20 Jul evening prediction + IH's own 21 Jul recap; the flat-branch
  bait is exactly what caught him on 21 Jul.)
- SOLO-LEADER VETO (clause on the GAP-SIZE ASYMMETRY entry tell): "BankNIFTY moving
  FIRST" is void as an entry tell when the other TWO indices sit capped below their
  own closing points — a lone leader against a capped majority is suspect; the capped
  indices are the honest read (divergence-fails, two holders vs one breaker). Wait for
  at least one other index to reclaim its closing point. (IH's 21 Jul loss diagnosis.)

**Confirmed but deliberately NOT re-encoded (already present):** the pre-open
no-plan branch ("if it falls directly, no plan can be made" = DECISION_RULES #8);
small-quantity/trickle buyers + holiday booking → not huntable (MULTI-DAY
ACCUMULATION + WEEKEND/HOLIDAY CARRY-RISK); the current-session seller-flush call
trade (CURRENT-SESSION TRAP RESET + AVERAGING TRAP + huge-gap loss-limit rule); and
the 1000-2000-point one-way regime caveat (already inside the HUGE-gap rule).

**Knowledge changes (v3o, all prose):**
- `RETAIL_POSITIONING`: BOTH-WAYS FLUSH DAY → FOLLOW THE OPENING (after GIFT-GAP,
  which it disambiguates).
- `BNF_SPECIFIC`: SOLO-LEADER VETO clause on the leader-moves-first entry tell.
- Test marker: `test_system_prompt_has_v3o_flush_day_and_solo_leader_knowledge`.

## Video addendum - 22 Jul runaway trend + the all-HOLD day (v3p)

> Source: `d-B4_cGK-ng` (22 Jul 2026, "Live Bank Nifty Option Trading", 7:26).
> Full Hindi auto-transcript. (The transcript panel again refused on the FIRST tab
> across reload and resize+reload; a FRESH TAB worked — same fix as 21 Jul.)
>
> Cross-checked against BOTH agent artefacts for the first time:
> `Backtest Outputs/sl_hunting_decisions.jsonl` (every per-bar decision, incl. HOLDs)
> and `Backtest Outputs/sl_hunting_journal.jsonl` (completed trades only).

**The tally for 2026-07-22:**

| Source | Result |
|---|---|
| decisions | **59 decisions, ALL HOLD** — zero entries (09:17:01 → 10:31:47) |
| journal | **no rows at all** (file's last write is 21 Jul) |
| breakdown | 51 genuine HOLDs (confidence 3/2) + **8 `agent_error`** |
| IH | **WIN** — puts on the no-retracement breakdown, over-achieved target |

The 8 error rows are operational, not analytical: 1 invalid output (10:03), **6
consecutive "Agent usage-limited"** (10:12-10:16), 1 timeout (10:19). Worth
tracking — a usage-limit burst silently costs six consecutive decision bars.

Session summary (IH): the market opened straight into selling across all three
indices. He deliberately did NOT rush ("a retracement here would be large, so no
hurry"), then entered PUTs once continuous selling with no major retracement had
proved itself, reasoning: **"if a big move is going to happen the market will NOT
retrace — it just keeps falling; so follow that momentum"**, and its converse, "if a
large retracement happens the big move probably won't come — others add and it goes
sideways". He also noted the already-seated sellers were in good profit so
"targeting them makes no sense at all" (TARGET-BOOKED, already encoded). He booked an
over-achieved target on the first stall, and closed by contrasting with 21 Jul:
"sit and watch in a CORRECT trade; there is no benefit sitting in a WRONG trade".

**Agent vs IH — the agent's ANALYSIS was right and its ACTION was absent.** Its
reasoning repeatedly and correctly applied the encoded knowledge — it named the
averaging-trap setup and refused to enter at the gap extreme (v3j), called
move-exhaustion on a spent bounce (v3j), identified "a genuine breakdown (evicts
buyers, not a buyer-hunt)" (v3n's ONE BREAKDOWN, NOT TWO), and repeatedly discounted
the stale mechanical cross-index verdict within the opening hour (v3l). But every
one of the 51 genuine HOLDs terminates in the same clause: *"no confirmed reversal
pattern at a level right now"*. The agent only ever evaluated REVERSAL entries. On a
one-way day the reversal setup never prints, so it waited out the whole move.

Root cause in the prompt (verified by grep before writing): there was **no with-trend
entry path outside OPENING_DRIVE's first-15-minutes window**, and `PSYCHOLOGY`
actively said *"In a pure fast trend you rarely get a clean entry — wait."* Nothing
covered "no retracement" / "runaway" as a signal.

**Net-new method distilled:**
- RUNAWAY TREND — the no-retracement continuation. The ABSENCE of a retracement is
  itself the signal of a large one-way move; on such a day the reversal pattern will
  never print, so the with-trend continuation IS the trade. Its converse is equally
  actionable: once a LARGE retracement appears, the big move is less likely (others
  add, price goes sideways) — stand aside. Because this branch has no reversal
  pattern to lean on, its invalidation is explicit: **the first real retracement**.

**Scope decision (operator-approved).** This is the THIRD and final exception to the
mandatory pattern+confirmation rule, and the only one valid outside the opening
window — so it is hedged hard: a sustained one-way move that has broken a real level,
NO meaningful retracement since (a pullback through the 50% fibo of the leg kills the
branch), ALL THREE indices agreeing, entry only on a shallow pause and never at a
fresh spike or as a counter-trend fade, an honest stop, exit on the first real
retracement, and book the average-to-over-achieved target at the first stall. The
operator explicitly chose the scoped entry exception over a knowledge-only guard,
because the guard would not have changed this day at all.

**Confirmed but deliberately NOT re-encoded (already present):** don't hunt the
already-profitable with-trend crowd (TARGET-BOOKED); don't rush the opening minutes;
book on the first stall rather than the perfect target (PREMIUM NON-CONFIRMATION /
MOVE-EXHAUSTION); and "sit in a correct trade, not a wrong one" (PROFIT-HOLD + the
v3m losing-side flip ban).

**Knowledge changes (v3p, all prose):**
- NEW section `RUNAWAY_TREND`, composed after `OPENING_DRIVE`.
- `ROLE`: the exception list now names the runaway-trend continuation.
- `PSYCHOLOGY`: limiting clause on "in a pure fast trend ... wait" (it means don't
  FADE and don't chase a spike — not sit out a one-way day).
- `DECISION_RULES` #3: names the new exception and adds a third-consecutive-HOLD
  self-check on a strongly one-way day.
- Test markers: `test_system_prompt_has_v3p_runaway_trend_knowledge`,
  `test_runaway_trend_section_is_composed_into_the_prompt`.
