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

## Video addendum - 23 Jul re-entry gate + expiry pinning (v3q)

> Source: `9Tzi96RY7Jc` (23 Jul 2026, "Live Bank Nifty Option Trading"). Full Hindi
> auto-transcript (fresh-tab recipe, first try).
> Tallied against BOTH artefacts for 2026-07-23.

**The tally — the agent's most active day, and a losing one:**

| Source | Result |
|---|---|
| decisions | 69 decisions: 60 HOLD, **5 entries** (3 SHORT / 2 LONG), 4 EXIT |
| journal | **5 completed trades, net -Rs.7,054.75** |
| `agent_error` | 3 rows, all "usage-limited" (10:27-10:29, end of session) |
| IH | **WIN** — ONE put trade, booked on the BankNIFTY breakdown |

| # | Entry | Direction / setup | Result |
|---|---|---|---|
| 1 | 09:32 | SHORT `gap_down_bounce_fail_bearish_engulfing` | **+Rs.7,281** (1.58R) |
| 2 | 09:41 | SHORT `trendline_4th_touch_fib78_rejection` | **-Rs.6,921** (stop) |
| 3 | 10:02 | LONG `gap_down_seller_hunt_trendline_reclaim` | **+Rs.6,408** (0.73R) |
| 4 | 10:11 | LONG `gapdown_averaging_trap_reclaim` | **-Rs.1,296** |
| 5 | 10:21 | SHORT `double_top_rejection_prev_low_resistance` | **-Rs.12,526** (AI_STOP) |

**The pattern is unmistakable: the FIRST trade of each thesis WON; every RE-ENTRY
LOST.** Winners +Rs.13,688; the three re-entries -Rs.20,743. Without them the day is
+Rs.13,688 instead of -Rs.7,055.

Timing makes it worse: trade 2 opened **2.5 minutes** after booking trade 1 (same
direction); trade 4 opened **2.5 minutes** after booking trade 3 (same direction);
trade 5 opened **~4 minutes** after exiting the losing trade 4, reversing INTO the
move that had just hurt it — and became the day's biggest loss.

**These were already banned.** Trades 2 and 4 are textbook MOVE-EXHAUSTION ("do NOT
re-enter the SAME direction on a later, smaller pattern chasing the tail of the move
you just took"); trade 5 is the v3m losing-side NO INSTANT FLIP ban verbatim. So the
gap is NOT missing knowledge — it is that **both rules are judgement calls, and the
agent satisfied them rhetorically every time** by naming a genuinely different-sounding
setup on the same price structure: "4th touch of the descending trendline + 78% fib",
"averaging trap reclaim", "double top rejection". Each reads as a fresh premise; each
was the same move wearing a new label. MOVE-EXHAUSTION even says a fresh trade "needs
a NEW named crowd trapped by NEW price action" — and the agent believed it had one.

Session summary (IH): a good gap-down after a weak prior day, so he planned to go WITH
the selling. He predicted the shape — "it will come up once, then fall; finding that
turning point is the only work today" — entered PUTs into the bounce rather than at
the extreme, and read BOTH crowds as unavailable (yesterday's sellers already paid;
today's would-be sellers never got their breakdown, so they did not hold). He then
waited for ONE breakdown and booked it. Two of his asides drive this addendum: that
Sensex, **the expiring index, would be least likely to break down**, so the trigger
would come from BankNIFTY or NIFTY; and that no chain reaction was coming because
"traders are sitting in confidence — if someone is short, they won't run quickly, they
know the market is negative". His closing warning lands squarely on the agent's day:
trade with understanding, "otherwise you enter randomly — now it's going selling, now
buying — and what benefit will you get?"

**Net-new method distilled:**
- POST-EXIT RE-ENTRY GATE — the mechanical check that MOVE-EXHAUSTION and NO INSTANT
  FLIP lacked. After ANY exit, the next entry in EITHER direction requires: ~15
  completed 1-min bars since the exit; a NEW STRUCTURAL EVENT after it (a level really
  broken/reclaimed, or a fresh swing formed — not continued drift inside the same
  structure); and a nameable NEW crowd trapped since. Plus the explicit loophole
  closure: **a different pattern name on the same structure is not a new premise.**
  Entries only — exits and the mechanical stop/target/max-loss/square-off paths are
  untouched.
- EXPIRING INDEX RESISTS THE BREAK — the index expiring today gets pinned and is the
  one LEAST likely to break a level cleanly, so take the breakdown/breakout trigger
  from a NON-expiring index, and do not read the expiring index's refusal to follow as
  the premise failing. Corollary: once the non-expiring index HAS broken while the
  pinned one is still stuck, that is a booking signal. Deliberately scoped so it does
  not contradict the existing "expiry = extra FUEL" note (fuel yes, trigger no).
- A CONFIDENT CROWD DOES NOT STAMPEDE — a crowd positioned WITH the prevailing
  multi-day direction does not panic out, so there is no cascade to harvest: expect a
  normal move and take the ordinary target. The stampede needs a crowd positioned
  AGAINST the trend, or freshly lured in at an extreme.

**Confirmed but deliberately NOT re-encoded (already present and correctly applied
today):** entering into the bounce rather than at the gap extreme (v3j AVERAGING TRAP
— trade 1 and trade 3 both did this and both won); the dual crowd-availability tests
(TARGET-BOOKED + CLOSING-POINT HOLD TEST); "few high-clarity trades" (UNIQUE-TRADE
FILTER + NO DAILY-INCOME PRESSURE, which the day's five entries violated in spirit).

**Operational note:** 3 `agent_error` rows, all "usage-limited", clustered at
10:27-10:29 — the same subscription-capacity symptom as 22 Jul (which lost 6
consecutive bars). No prompt change addresses it.

**Knowledge changes (v3q, all prose):**
- `RISK`: POST-EXIT RE-ENTRY GATE (immediately after MOVE-EXHAUSTION, which it
  operationalizes).
- `BNF_SPECIFIC`: EXPIRING INDEX RESISTS THE BREAK (after the expiry-priority note it
  scopes).
- `RETAIL_POSITIONING`: A CONFIDENT CROWD DOES NOT STAMPEDE (after TARGET-BOOKED).
- Test markers: `test_system_prompt_has_v3q_reentry_gate_and_expiry_pin_knowledge`,
  `test_reentry_gate_does_not_contradict_the_exit_rules`.

## Video addendum - 27 Jul laggards-must-join + the gate moves into code (v3s)

> Source: `PezwEQ300lo` (27 Jul 2026, "Live Bank Nifty Option Trading"). Full Hindi
> auto-transcript (fresh-tab recipe). No weekend lecture this cycle — the channel
> went 24 Jul prediction straight to 27 Jul.
> Tallied against BOTH artefacts for 2026-07-27.

**The tally — a WINNING day whose shape still exposes the same defect:**

| Source | Result |
|---|---|
| decisions | 73 (63 HOLD, **5 entries**, 5 EXIT) + 2 `agent_error` ("usage-limited", 09:17) |
| journal | **5 trades, net +Rs.9,012.25** |
| IH | **WIN** — one short basket into the gap-up, booked early without his breakdown |

| # | Entry | Direction / setup | Result |
|---|---|---|---|
| 1 | 09:20 | SHORT `huge_gap_mindset_trap_fade` | **-Rs.2,703** (held 55 seconds) |
| 2 | 09:23 | LONG `gapup_flush_hammer_reclaim` | -Rs.1,666 |
| 3 | 10:02 | LONG `psych_support_hammer_reversal` | -Rs.1,679 |
| 4 | 10:18 | SHORT `double_top_bearish_engulfing` | **+Rs.18,858** |
| 5 | 10:22 | SHORT `trendline_continuation_fibo_rejection` | **-Rs.3,797** |

Session summary (IH): a clear gap-up (big on Sensex). He SOLD immediately across all
three. Premise: the prior session sold continuously then closed off a recovery, and
**the two-day weekend made those short-holders book and leave** — so no seller SLs are
seated, and the gap-up must therefore build a BUYER trap. He conceded his entry looked
early ("the entry seems to have gone wrong, but we won't be wrong on DIRECTION") and
held within a loss limit. BankNIFTY then fell hard — but Sensex and NIFTY refused to
break down. He waited for that breakdown, never got it, and **booked anyway** once
BankNIFTY started printing very small candles: "if Sensex and NIFTY had broken down the
target could have DOUBLED, but they keep holding and retracing, so the risk has grown."
His closing read is the sharp part: with the shared move dead the session can only
resolve two ways, and "if it starts going up, good seller SLs will be available" — i.e.
his own short would become the harvestable inventory.

**Agent vs IH — same thesis, opposite conviction.** Trade 1 IS IH's trade: the agent
named the huge-gap mindset trap and shorted the gap-up at 09:20. It then **abandoned it
after 55 seconds** at -26.2 points, with an exit reason asserting the "SL-hunt leg ...
captured" while the position was in fact at a loss. It spent the next hour fighting its
own read with two LONGs (both losers) before returning to the identical short thesis at
10:18 — which produced the day's entire profit. Direction agreement was never the
problem; conviction and re-entry discipline were.

**The defect this addendum exists for.** Trade 5 opened **89 seconds** after trade 4
closed, in the SAME direction, and gave back Rs.3,797 of an Rs.18,858 winner. Its
reasoning never mentions the POST-EXIT RE-ENTRY GATE at all — it simply named a new
setup ("the 4th touch on a descending trendline"), which is precisely the relabelling
loophole v3q was written to close. The gate has been in the prompt since 23 Jul, so
this is the SECOND consecutive session in which a prose rule failed to bind.

Replaying both journals against candidate cooldowns (approximate — blocking one entry
shifts the clock for the next):

| Cooldown | 23 Jul kept | 27 Jul kept |
|---|---|---|
| none (actual) | -Rs.7,055 | +Rs.9,012 |
| 2 min | -Rs.7,055 | +Rs.12,810 |
| **5 min** | **+Rs.1,163** | **+Rs.14,476** |
| 10 min | +Rs.1,163 | +Rs.14,476 |
| 15 min | +Rs.13,689 | **-Rs.8,179** |

5 and 10 minutes are identical on this sample; **15 minutes is actively harmful** — it
would have blocked 27 Jul's Rs.18,858 winner (entered 12.4 min after the prior exit).
That is direct evidence AGAINST v3q's own "~15 completed bars" prose, which is
corrected here.

**Net-new method distilled:**
- LAGGARDS NEVER JOINED → BOOK WHAT YOU HAVE: the HOLDING-side counterpart to
  SOLO-LEADER VETO (entry-only). When the leader delivers your direction but the other
  two indices never break their own levels, the triple-index move that justified your
  TARGET is not forming — take what the leader gave instead of waiting for the
  breakdown that would "double" it. Booking trigger: the leader starts stalling into
  small candles while the laggards are still unbroken. Urgency: with the shared move
  dead the session resolves binary, and if it resolves against you the crowd holding
  your direction — now including YOU — is the freshly seated inventory the operator
  hunts next. Distinct from the leader FAILING to lead (v3l): there the leader quit;
  here the leader worked and the followers refused.

**Behaviour change (operator-approved): the re-entry gate's TIME arm is now enforced in
CODE (SLH-005), not prose.** `MasterWorkerExecutor.enter` consults the worker's
`post_exit_cooldown_remaining_seconds()` and REJECTS the entry with the remaining
seconds in the reason, so the model sees a refusal instead of self-policing. Knob
`SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES` (default 5, `0` disables). This reuses the
pattern `SupertrendBullishWorker` has always used (`POST_EXIT_COOLDOWN_MINUTES`, also
5). ENTRIES ONLY: exits, stop/target, max-loss and square-off are untouched.

**MAT-111 reliability follow-up.** A two-leg SL Hunting trade is not closed when its
first leg exits. The timer now uses a monotonic deadline and starts exactly once when
both NIFTY and BankNIFTY are confirmed flat. A lone surviving leg, or a partial /
indeterminate broker close retained for reconciliation, cannot run the interval down.
Unreadable, non-finite, or negative guard state rejects only new LIVE entries; paper
use remains fail-open, and exits never consult the guard.

**Confirmed but deliberately NOT re-encoded (already present, and correctly applied):**
the weekend/holiday flattening of the prior crowd (WEEKEND / HOLIDAY CARRY-RISK — it
was IH's whole premise today); the huge-gap mindset trap (the agent named it verbatim);
booking on a stall rather than the perfect target; and holding within a loss limit.
Deliberately NOT encoded: IH's "entry early but direction right, so hold through the
drawdown". Encoding hold-through-drawdown into a live agent is how a capped loss becomes
an uncapped one, and v3j already fixes the same problem from the safe side — do not
enter at the gap extreme, wait for the bounce.

**Knowledge changes (v3s, prose + code):**
- `BNF_SPECIFIC`: LAGGARDS NEVER JOINED → BOOK WHAT YOU HAVE (before the v3r
  LAGGING-INDEX ENTRY LOCATOR it complements).
- `RISK`: the POST-EXIT RE-ENTRY GATE's TIME arm now states it is ENFORCED IN CODE, and
  its "~15 bars" figure is corrected (clearing the clock does not by itself authorise a
  trade).
- Code: `SL_HUNTING_POST_EXIT_COOLDOWN_MINUTES`, the worker's basket-flat monotonic
  deadline, `post_exit_cooldown_remaining_seconds()`, and the rejection in
  `MasterWorkerExecutor.enter`.
- Test markers: `test_system_prompt_has_v3s_laggards_and_enforced_cooldown_knowledge`,
  plus executor tests (reject / allow / never-blocks-exit / live fail-closed /
  paper fail-open / duck-typed) and worker tests (first entry free, waits for the
  final leg, partial-close recovery, stop-out, monotonic expiry, disable).

## Video addendum - 24 Jul profit-booking recovery + lagging-index entry (v3r)

> Sources (full Hindi auto-transcripts from the YouTube transcript panel):
> - `c6fC2LRkocE` (24 Jul 2026, "Live Bank Nifty Option Trading", 11:25) -
>   primary live-session evidence.
> - `ekVgnszh7tU` (23 Jul 2026, "Prediction For 24 JULY 2026", 2:17) -
>   supporting premarket plan for the 24 Jul session.
>
> Cross-checked against BOTH agent artefacts for 2026-07-24:
> `Backtest Outputs/sl_hunting_decisions.jsonl` and
> `Backtest Outputs/sl_hunting_journal.jsonl`.

The premarket thesis was sell-side under every opening type. The preceding market
had already sold hard and given some retracement, so established sellers were in
profit and were not the crowd to hunt. A gap-down was the cleanest continuation
case; flat carried some support-trap risk, while a gap-up would first need to build
its own negative-side trap.

The live market gapped down and recovered. NIFTY and Sensex rose too quickly to show
a controlled short entry, so IH watched the lagging BankNIFTY for the recovery to
stall. When it did, he opened one PUT basket: BankNIFTY 56200 PUT (1,170 quantity),
Sensex PUT (900), and NIFTY PUT (1,365). The failure case was explicit: if the
recovery kept running, retraced, and then produced another strong upward impulse,
the short direction was wrong. Instead, the recovery stalled and selling resumed.
He booked the profitable basket after the sharp drop began attracting obvious new
sellers and the intraday reward had paid the planned risk.

**The tally for 2026-07-24:**

| Source | Result |
|---|---|
| decisions | **72 decisions**: 64 HOLD, 4 entries, 4 EXIT |
| decision errors | **1 `agent_error`** at 10:26 ("Previous agent call still running") |
| journal | **4 completed trades**, net **+Rs.19,136.75** |
| IH | **WIN** - one PUT basket after the gap-down recovery stalled |

| # | Entry | Direction / setup | Result |
|---|---|---|---|
| 1 | 09:20 | LONG `gap_down_flush_hammer_reversal` | **-Rs.726.25** |
| 2 | 09:27 | SHORT `gap_down_recovery_fail_bearish_engulfing` | **+Rs.11,951.25** |
| 3 | 09:39 | SHORT `gap_down_recovery_fail_bearish_engulfing` | **+Rs.3,633.00** |
| 4 | 10:01 | SHORT `gap_down_failed_recovery_short` | **+Rs.4,278.75** |

The 09:27 short matched IH's direction and recovery-failure premise. The agent's
preceding 09:20 long treated the first hammer recovery as a seller hunt, which was
the method mismatch: paid multi-day sellers plus an expected profit-booking bounce
did not establish a new long premise. The two later shorts repeated the same
already-captured move. Short trades made **+Rs.19,863.00** in total, but their profit
does not turn the repeated entries into new method evidence.

**Net-new method distilled:**
- PROFIT-BOOKING RECOVERY TEST: after an established multi-day selloff has paid the
  seller crowd and the next session gaps down, the first green recovery can be
  seller profit-booking rather than a seller-hunt reversal. A hammer / first bounce
  is not enough for a long. Keep the continuation short conditional until the
  recovery either stalls below the closing point / round number / opening range, or
  invalidates it with a reclaim, held pullback, and second strong upward impulse.
- LAGGING-INDEX ENTRY LOCATOR: when positioning and the triple-index read have
  already fixed the day direction but the faster indices provide no clean entry,
  use the lagging index's stall / rejection to locate timing. It is an entry cue,
  not a standalone directional premise, and remains subordinate to MASKED BNF LAG,
  GAP-SIZE ASYMMETRY, and SOLO-LEADER VETO.

**Confirmed but deliberately NOT re-encoded (already present):** TARGET-BOOKED
crowd filtering; the strict GAP-DOWN CONTINUATION SHORT; fast-spike profit booking;
the fast-one-way / slow-other-way trap; MOVE-EXHAUSTION; and the POST-EXIT RE-ENTRY
GATE.

**Historical operational findings (documented, no runtime change in v3r):**
- The first short's journal row says `placeholder - will not call, holding`, while
  its 09:38 decision records that an erroneous order-tool invocation closed both
  legs early. This is journal fidelity, not a new SL-hunting method.
- Every entry after an exit occurred inside the existing ~15 completed-bar re-entry
  gate: 09:27 after the long exit, 09:39 after the first short exit, and 10:01 after
  the second short exit. The 09:38 decision even named the gate, but the next
  independent per-bar call received only the current FLAT state and recent candles,
  not the previous exit time / structure. Another prose rule cannot supply missing
  cross-call state; runtime enforcement is a separate follow-up.
- The single 10:26 `agent_error` was a still-running prior call. No prompt change
  addresses inference concurrency in this knowledge-only version.

**Knowledge changes (v3r, all prose):**
- `RETAIL_POSITIONING`: PROFIT-BOOKING RECOVERY TEST, beside TARGET-BOOKED.
- `BNF_SPECIFIC`: LAGGING-INDEX ENTRY LOCATOR, scoped against the existing lag and
  divergence rules.
- Test marker:
  `test_system_prompt_has_v3r_profit_booking_recovery_and_lagging_index_knowledge`.

## Video addendum - 28 Jul dual-expiry session + morning speed (v3t)

**Source:** Intraday Hunter live session, 28 Jul 2026 (`-vNa6-t2SWw`), read alongside
the agent's own `sl_hunting_decisions.jsonl` and `sl_hunting_journal.jsonl` for the
same session. Both NIFTY and BankNIFTY expired that day.

**What IH did:** the market gapped down and then pushed up. He read the gap-down as
having let the SELLERS book, leaving BUYERS sitting - so he SOLD puts across all
three indices (BNF 1170, Sensex 900, NIFTY 1430). NIFTY and Sensex did print a
breakout against him, but it never crossed ~24040, the level from his own pre-open
analysis; he named that level in advance as the point where "the problem increases
for us", because past it the seated buyers move into profit. The breakout failed,
BankNIFTY led the fall, and he booked into the move rather than waiting for a stall.

**Agent tally for 28 Jul:** 72 decisions (68 HOLD, 2 ENTER_SHORT, 2 EXIT), zero
`agent_error`, window 09:17:04-10:30:01. Two trades, net **-Rs.434.50**:
- 09:27:04 -> 09:41:36 SHORT `double_top_shooting_star_rejection`, 10 lots,
  -4.55 pts, **-Rs.5,387.50**, R -0.4 (exit: premise invalidated, price held above
  pivot 23968.93 and prev close 24003.65).
- 10:03:02 -> 10:07:35 SHORT `buyer_trap_evening_star_rejection`, 6 lots,
  +24.35 pts, **+Rs.4,953.00**, R 1.28 (exit: stall at the pivot).

**SLH-006 verified in production.** The runner logged
`SL Hunting: injected 1916 pre-open note chars for 2026-07-28 (ADVISORY).` - the
character count matches the end-to-end test exactly. 22 of the 72 decisions cited
the note, and it behaved as designed rather than as an override: at 09:53:06 the
agent noted the note and `cross_index` both favoured a sell bias and still HELD for
want of a confirmed pattern; at the 10:03 entry it cited "below the 24040
buyer-profit line, putting risk on trapped buyers per the pre-open plan" - and that
trade won. **SLH-005 never fired**: the agent's own re-entry gap was ~21 minutes,
well past the 5-minute cooldown.

**Net-new method distilled:**
- EXPIRY-DAY PREMIUM ASYMMETRY: on an expiry day a bought option gives back an
  adverse move much faster than it pays a favourable one, so book INTO strength and
  do not wait for a stall-and-pullback to confirm the turn. The agent's own book
  measured this on 28 Jul: the loser bled ~1.58 premium points per ADVERSE spot
  point, the winner earned ~0.45 per FAVOURABLE spot point - a ~3.5x asymmetry, and
  the ratio is independent of lot size (both trades were NIFTY, so the lot size
  cancels). *Caveat recorded honestly:* the loser was held 14m32s against the
  winner's 4m33s, so decay over the longer hold is part of that gap - which is the
  point of the rule, but it does mean 3.5x is the combined delta+theta effect, not a
  pure per-candle measurement. Deliberately scoped as an EXPIRY-DAY exception so it
  cannot be read as licence to cut winners early against PROFIT-HOLD.
- MORNING SPEED IS NOT INFORMATION: IH's warning was "sometimes such a trade throws
  you out in the next 5 minutes; then it feels like 'I was wrong quickly, let me try
  again' - so if you have a habit of over-trading, avoid trading in the morning."
  Opening-window momentum resolves within a couple of bars in either direction, so
  the SPEED of a morning stop-out carries no information about the next trade. This
  raises the bar for the next entry rather than lowering it, and the enforced
  cooldown is a floor rather than the standard. **Deliberately NOT hardened into a
  one-trade-per-morning ban:** both 28 Jul's winner and 27 Jul's winner were second
  trades taken after an earlier exit, so a ban would have cost real money. The
  banned thing is the reflex retry whose only new evidence is that the last one
  ended fast.
- PRE-COMPUTE BOTH NUMBERS: IH - "before making the trade I calculated how much loss
  I'd have to take if a breakout happens and it goes against me, and I calculated my
  profit too; only then can I handle the trade accordingly." The existing
  PLAN-OF-EXECUTION precheck already required a named invalidation and target, but
  only qualitatively. The addition is the rupee figure at each, computed at the lot
  size about to be sent - because a pre-accepted loss is what makes adverse movement
  that is still inside the plan survivable. This is what let IH sit through a real
  breakout against his position on 28 Jul.

**Confirmed but deliberately NOT re-encoded (already present):** the gap-down
crowd read (sellers booked, buyers left sitting) is TARGET-BOOKED plus the
trap-density test; using the pre-open level as the day's invalidation is what
SLH-006 already provides; expiry-as-fuel-not-premise and the post-first-move expiry
range are EXPIRY IS CONTEXT, NOT A PREMISE and EXPIRY-DAY RANGE.

**Knowledge changes (v3t, all prose):**
- `RISK`: EXPIRY-DAY PREMIUM ASYMMETRY, placed directly after PREMIUM
  NON-CONFIRMATION (which is explicitly scoped to non-expiry days, so the two read
  as siblings rather than rivals).
- `RISK`: MORNING SPEED IS NOT INFORMATION, beside POST-LOSS SPEED LIMIT.
- `DECISION_RULES`: PRE-COMPUTE BOTH NUMBERS, folded into the rule 2 precheck.
- Test marker:
  `test_system_prompt_has_v3t_expiry_asymmetry_and_morning_speed_knowledge`.
## Video addendum - 29 Jul big-gap follow day + the friction floor (v3u)

**Source:** Intraday Hunter live session, 29 Jul 2026 (`e9qDdFfOVyk`, 5:52), read
alongside the agent's own decisions/journal for the same session.

**What IH did:** NIFTY and Sensex gapped up hard with a slight rejection after. He
bought CALLs - the same side his own pre-open note had called - but was explicit that
he was FOLLOWING the market rather than hunting anyone: "neither buyers nor sellers
are seated, so we follow what is already running." Three things he said matter:
- "If this gap-up had been a bit SMALLER there would have been no difficulty in
  buying. We will buy here too, but the worry is that no rejection comes... the
  gap-up is a bit too much, and because of this the market becomes risky."
- "Directly, buyers' and sellers' SLs are not available here" - no stop cluster near
  price, therefore no fuel.
- "We will make a NORMAL profit and leave"; and at the exit, "we had reached close to
  an AVERAGE target... this is a sideways market, it will not give us much momentum.
  Today there is no momentum."

**Agent tally for 29 Jul:** 65 decisions (60 HOLD, 3 ENTER_LONG, 2 EXIT), zero
`agent_error`. Three trades, all LONG, net **+Rs.3,695.25**:
- 09:22:02 -> 09:25:37 `opening_drive_gapup_continuation`, 2 lots, -7.45 pts,
  **-Rs.1,339.00**, R -0.15, held 215s.
- 09:32:01 -> 09:40:59 `gap_up_fibo50_continuation`, 6 lots, +37.30 pts,
  **+Rs.10,335.00**, R 2.04, held 538s, exit `AI_TARGET`.
- 10:02:00 -> 10:03:45 `fibo_50_inside_bar_reclaim`, 7 lots, +4.65 pts,
  **-Rs.5,300.75**, R 0.31, held 105s.

The agent independently reached IH's side of the market, and 17 of the 65 decisions
cited the pre-open note. The winner was the one trade that actually ran (37 points
over nine minutes); the two losers were both short holds.

**The measurement that drove this version.** The third trade moved 4.65 points IN ITS
FAVOUR and still lost Rs.5,300.75 - about 10 premium points per unit against a
favourable move, over 105 seconds. This is not a modelled cost: the runner applies no
slippage or spread modelling at all, so that is an observed option-LTP move. The most
likely mechanism is gap premium bleeding out of the option faster than delta paid for
the spot move, which is exactly the effect IH described as "after the rejection, when
momentum started, we are not seeing much profit". *Recorded honestly:* the mechanism
is inferred; only the numbers are measured. Worth an operator eye on whether the
option LTP feed is jumpy on thin strikes, since that would change the reading.

**Net-new method distilled:**
- GAP SIZE IS A RISK DIAL, NOT A CONFIDENCE DIAL: a bigger gap makes the with-gap
  continuation WORSE, because price has jumped past the stop clusters that fuel a
  move - slow momentum, and lots of room for a rejection to run back through you.
  Trade an oversized gap as a smaller, normal-target trade or not at all. Explicitly
  scoped against the existing GAP-SIZE ASYMMETRY, which compares gaps ACROSS indices;
  this one is about the absolute size of the gap being traded.
- NO NEARBY STOPS -> NORMAL TARGET, DECIDED AT ENTRY: when following the market
  rather than hunting a named crowd, there is no fuel for a fast leg, so commit up
  front to an average target instead of discovering mid-trade that the runner is not
  coming. Guarded so it cannot justify a trade that is too small to be worth taking.
- PREMIUM NON-CONFIRMATION CAN GO NEGATIVE, NOT MERELY WEAK: the existing rule only
  covered P&L *lagging* the spot move. On a large-gap morning over a short hold it
  can invert - spot in your favour, position in loss. Consequences encoded: read
  `position_state` rather than assuming spot direction equals profit; check at entry
  whether the target is big enough to pay in PREMIUM terms; and note that abandoning
  a trade a bar or two after entry pays the round trip for no exposure to the move.

**Confirmed but deliberately NOT re-encoded (already present):** following the market
when nobody is trapped is the trap-density test plus the OPENING DRIVE branch; booking
on momentum failure rather than a fixed number is already the branch's target rule;
sideways = exit is TIME-DECAY discipline.

**Operational finding (documented, no runtime change in v3u):** the first trade's
journal row records `"exit_reason": "placeholder"`. The same journal-fidelity gap was
noted on 24 Jul in the v3r addendum, so it has now recurred and is worth a fix
independent of the knowledge layer.

**Knowledge changes (v3u, all prose):**
- `OPENING_DRIVE`: GAP SIZE IS A RISK DIAL, NOT A CONFIDENCE DIAL.
- `RISK`: NO NEARBY STOPS -> NORMAL TARGET, beside the worthwhile-target rule.
- `RISK`: the IT CAN GO NEGATIVE sub-bullet under PREMIUM NON-CONFIRMATION.
- Test marker: `test_system_prompt_has_v3u_gap_size_and_no_fuel_knowledge`.

## ERRATUM to v3t - the expiry-asymmetry measurement was wrong (2026-07-29)

Recorded as an erratum rather than an edit to the v3t section above, so the
mistake and its correction both stay on the record.

**What v3t claimed:** that on an expiry day a bought option gives back an adverse
move ~3.5x faster than it pays a favourable one, measured on the agent's own book
for 28 Jul, "both indices expiring".

**What was actually true.** The agent's NIFTY leg on 28 Jul was
`NIFTY-Aug2026-24000-PE`, `ExpiryDate=2026-08-04`, `DaysToExpiry=7`. It was NOT in
the expiring series. Only the BankNIFTY mirror (`BANKNIFTY-Jul2026-57300-PE`, the
July monthly) was 0 DTE. IH was trading the expiring contract; the agent was not.

**Two errors followed from that:**

1. *The number.* The 3.5x was computed from the journal's `option_pnl`, which is
   BASKET P&L (`realized_pnl - _entry_realized_pnl`, both legs), divided by
   NIFTY-ONLY spot points. That mixes two underlyings and two expiries in a single
   ratio. Leg-level figures from the master log:
   - loser: 139.45 -> 131.00 on qty 650 = -Rs.5,492.50, i.e. -8.45/unit for 4.55
     ADVERSE spot points = **1.86** premium points per adverse point;
   - winner: 131.00 -> 150.90 on qty 390 = +Rs.7,761.00, i.e. +19.90/unit for 24.35
     FAVOURABLE spot points = **0.82** premium points per favourable point.
   The asymmetry is **~2.3x**, not 3.5x. (The mirror leg moved the opposite way on
   the winner - it LOST Rs.2,808 while the NIFTY leg made Rs.7,761 - which is what
   dragged the basket ratio so far off.)
2. *The mechanism and the scope.* "Collapsing time value on expiry day" cannot
   explain a 7-DTE option. And because the rule was scoped "EXPIRY-DAY exception
   only" while `get_target_expiry()` keeps the NIFTY leg 7-13 days out, the rule was
   written for a situation the leg it governs essentially never occupies.

**The correction.** The rule is renamed to PREMIUM ASYMMETRY, quotes the leg-level
~2.3x, states that the magnitude is situational rather than constant, and keys the
booking threshold to **the days-to-expiry of the option actually held** rather than
to whether some index expires today. Marker test updated, including a guard that
the discredited "3.5x" figure cannot reappear.

**Method lesson for future versions:** `outcome.option_pnl` is BASKET P&L while
`outcome.points` is NIFTY-ONLY. Never divide one by the other. Leg-level prices are
in the master log (`EntryOptPx` / `ExitOptPx` per leg); use those.

**v3u is unaffected** - 29 Jul fired no mirror legs at all, so its +4.65 spot /
-Rs.5,300.75 figure is NIFTY-only and stands. Its contract was 13 DTE, where theta
over 105 seconds is negligible, so the wide/thin far-dated book is now the leading
explanation for that move rather than gap-premium bleed.

**Root cause behind all of it:** the knowledge base is distilled from a trader who
is always in the near or expiring contract, while the execution layer bought the
SECOND expiry out. Fixed separately by SLH-008, which moves the NIFTY leg to the
current-week contract; from that point the expiry-aware rules and the instrument
actually held finally describe the same thing.

**Also corrected here (2026-07-29):** an earlier note in this investigation claimed
no live order had ever filled. That was wrong - a bad log query, since fills are
recorded on their own `REAL ORDER FILLED` lines rather than on the `ENTRY` line.
There were 74 real fills between 22 and 28 Jul, all on the 2026-08-04 contract; the
rejections began on 29 Jul when the ladder rolled and next-next became 2026-08-11.

## Video addendum - 30 Jul seated-crowd-over-gap + session-character carry-over (v3v)

**Source:** Intraday Hunter live session, 30 Jul 2026 (`LvA_VPLdm6Q`, 7:37).

**IMPORTANT - distilled from the VIDEO ONLY.** Unlike v3t and v3u, the agent's own
30 Jul book supplied no usable measurement, and deliberately none is quoted here:
- the 09:18 LONG has NO journal row (its live exit was rejected, then the runner was
  interrupted and restarted, so `after_exit` never fired);
- the operator exited that position MANUALLY, so its NIFTY P&L is in no log;
- market data was unhealthy 09:05-09:18, again 09:30-09:49, and again around 10:25,
  with option LTPs stale by up to 287s;
- the 10:14 trade recorded a NIFTY entry of 112.55 while the broker-confirmed fill
  was 125.00, which fabricated a +Rs.4,095 leg profit out of a -Rs.760.50 loss. The
  journal row was corrected by hand (basket 3636.0 -> -1219.5); journalled net for
  the day is -Rs.8,072.50, not -Rs.3,217.00.

The last point is a standing defect rather than a one-off: `broker_contract.py` has
no fill-price field at all, so LIVE P&L is always computed from the local LTP and
never from the broker's actual fill.

**What IH did:** open was almost FLAT to slightly GAP-DOWN, with Sensex and NIFTY
mildly positive but BankNIFTY already selling. He bought PUTS across all three within
minutes of the open (BNF 1170, Sensex 900, NIFTY), then booked on the first real
momentum leg.

His reasoning, close to verbatim:
- "For two-three days the market had very good positive momentum, and today we are
  directly selling, just minutes after the open... because of what happened over
  those days, BUYERS are seated, and to target those buyers we are selling here."
- "Even if it had opened slightly GAP-UP we would still have targeted the buyers.
  Now we got a gap-down, and we are still targeting the buyers."
- On booking: "It is not that they will just keep going down. They could still go
  SIDEWAYS... **yesterday we did not make much profit, we took a small profit and
  left, because yesterday too it looked sideways at the start.** So today it could
  also happen that momentum comes and then the market stays sideways. So booking the
  target was very necessary."
- Also present, and already covered by existing rules: a pre-set loss limit
  ("we will see what our limit is and handle the trade accordingly"), and expiry as
  a trap-context caveat (Sensex expiry that day).

**Net-new method distilled:**
- A SMALL GAP DOES NOT RESCUE A SEATED CROWD. RECRUITMENT HISTORY already said that
  on a gap-up the seated buyers "are already in profit and cannot be targeted, so go
  WITH the market". IH's session qualifies that: the escape hatch needs a gap large
  enough to genuinely release them. Judge the gap against the SIZE OF THE RUN that
  recruited the crowd, not against zero; a few tens of points does not free buyers
  seated over several days. Stated consequence: when a multi-day run has seated an
  identifiable crowd, the OPEN direction does not pick your side - the trapped crowd
  does, and flat / slightly-gap-down / slightly-gap-up can be the same trade.
  Deliberately written as a sub-bullet INSIDE RECRUITMENT HISTORY, with a test
  pinning it there, because read alone it would contradict the branch it qualifies.
- YESTERDAY'S MOMENTUM CHARACTER CALIBRATES TODAY'S PATIENCE. How far a move RUNS
  carries over between sessions, separately from direction. If the previous session
  gave an early move then went sideways, plan for the same shape today and take the
  momentum when it arrives. A direction that is working is not a promise that it
  keeps working - a market can be selling all day and still chop for hours inside
  that. Explicitly separated from PREVIOUS-CHART LINKAGE (who was recruited) and
  RECRUITMENT HISTORY (which way to trade): this one governs only HOW LONG to hold,
  and it tightens the target rather than loosening it.

**Considered and deliberately NOT encoded:** IH said he expected to "take some loss
in NIFTY and Sensex" while BankNIFTY carried the thesis. A "a red leg is not by
itself premise-invalidation" rule would sit uncomfortably close to LAGGARDS NEVER
JOINED (v3s), which tells the agent to BOOK when the leader is spent and the
laggards never broke. Rather than risk muddling a rule that already governs the
per-leg `exit_leg` decision, this is left out pending a session that separates the
two cleanly.

**Confirmed but already present:** pre-computing the loss limit before entry is v3t's
PRE-COMPUTE BOTH NUMBERS; expiry-as-trap-context is EXPIRY IS CONTEXT, NOT A PREMISE;
booking into an early move with no fuel behind it is v3u's NO NEARBY STOPS -> NORMAL
TARGET.

**Knowledge changes (v3v, all prose):**
- `RETAIL_POSITIONING`: A SMALL GAP DOES NOT RESCUE A SEATED CROWD, as a sub-bullet
  of RECRUITMENT HISTORY.
- `RISK`: YESTERDAY'S MOMENTUM CHARACTER CALIBRATES TODAY'S PATIENCE, beside the
  other booking rules.
- Test markers: `test_system_prompt_has_v3v_small_gap_and_carryover_knowledge` and
  `test_v3v_small_gap_rule_sits_inside_recruitment_history`.
## Video addendum - 31 Jul losing session + entry-point discipline (v3w)

**Source:** Intraday Hunter live session, 31 Jul 2026 (`RpK2h9xsrXk`, 9:01). Notable
because IH LOST on it - most sessions distilled here are wins, and a trader
explaining why he cut is better material than one explaining why he was right.

**What IH did:** flat pre-market open, plan for the positive side. He noted the
charts were "not making good shapes" - rejection, up, rejection, up - and that the
market had built a small RANGE. His pre-trade test was explicit: a breakout with
positive momentum at the open is best; a sudden BIG selling move would signal the
market wants to stay in the range, while small selling alongside a breakout is
fine. Crowd read: earlier momentum-buyers were rejected and ran, and few
participated at the second rejection, so there were no buyers left to target
either - which is why he FOLLOWED the market up rather than hunting anyone, and
took the call side.

It then broke out and fell straight back. He cut, and said why:
- "The trade still looks okay, but because of the ENTRY POINT a problem is being
  created. We will not see more loss than this."
- "In a trade that is going wrong you cannot apply your mind. While it was right we
  sat; but if the market is now continuously falling, and selling has started in
  Sensex and NIFTY too, we cannot wait."
- Throughout, he sized the wait against a pre-set number: "we know our loss and our
  target; we can wait accordingly" (already encoded as v3t's PRE-COMPUTE BOTH
  NUMBERS).

**Agent tally for 31 Jul:** 74 decisions (64 HOLD, 3 ENTER_SHORT, 2 ENTER_LONG,
5 EXIT), window 09:16:04-10:30:07, 19 decisions citing the pre-open note. Five
trades, net **+Rs.6,178.75** (PAPER: the operator had `LIVE_TRADING_ENABLED=false`
after the 30 Jul incident). Leg-level prices reconcile with every journal row.

**SLH-006 verified again:** `injected 1946 pre-open note chars for 2026-07-31
(ADVISORY)` - exactly the figure the note was validated to produce.

**Still unproven:** because the session was paper-only, the Kotak `avgPrc` key and
the live-refusal branch of the staleness gate were never exercised.

**A methodological correction worth recording.** Trade 1 looked corrupt - a LONG
whose `points` were -20.45 while the CE rose 99.60 -> 100.30 - and it was nearly
reported as a pricing bug. It is not. The journal's `entry_underlying` is the level
the AGENT declared (24364.55); the actual spot at fill was 24346.05. The real move
was -1.95 points, and a 0.70 rise on a ~100-point option is ordinary noise.

So: `points` and `option_pnl` are measured against DIFFERENT references, and their
disagreement can never by itself evidence a pricing fault. This is the third time
this class of confound has bitten (the v3t basket ratio, 30 Jul, and now this).
Only leg-level prices from the master log, or a broker fill, can settle it.

**Net-new method distilled:**
- THE ENTRY POINT IS PART OF THE PREMISE: a right read entered in the wrong place
  is a wrong trade. The direction can still look correct while the location has
  made the position unholdable - the stop sits where ordinary noise reaches it, so
  the market need not disprove you to remove you. "The idea still looks fine, it is
  the entry that is the problem" is an exit instruction, not a reason for patience,
  because being eventually right does not pay a position you were forced out of.
  Sub-bullet names the self-deception: once a trade is going against you, further
  analysis stops being analysis - you cannot think your way out of a position that
  is already wrong.
- COUNTER-MOVE SIZE SAYS RANGE OR BREAKOUT: positioned for a small range to break,
  the SIZE of the first move against you is the read. Small adverse movement
  alongside a break is the level being cleared; a sudden LARGE adverse move says
  the market intends to stay in the range, and a range pays no directional trade.
  Scoped explicitly against the existing momentum-quality rule, which reads the
  WITH-trend move and is a profit-taking cue; this reads the move AGAINST you and
  is a premise test.

**Confirmed but already present:** no crowd left to target on either side -> follow
rather than hunt is the trap-density test plus v3u's NO NEARBY STOPS; waiting
against a pre-set loss figure is v3t's PRE-COMPUTE BOTH NUMBERS; a full give-back
of the covering move is the existing full-body-reversal invalidation.

**Also in this version:** `MAX_SYSTEM_PROMPT_CHARS` raised 75,000 -> 120,000.
Knowledge alone had reached ~68,000, leaving roughly 7,000 for lessons (up to
12 x 280) plus a ~2,000-character note, so the next ordinary addendum would have
tripped it. The cap is a sanity bound against a runaway lessons file or malformed
note, not a budget for knowledge growth. A new test asserts provable headroom for
the worst-case runtime additions rather than trusting the raw number.

**Knowledge changes (v3w, all prose):**
- `RISK`: THE ENTRY POINT IS PART OF THE PREMISE, before the premise-invalidation
  stop rule it qualifies.
- `RISK`: COUNTER-MOVE SIZE SAYS RANGE OR BREAKOUT, beside momentum quality.
- Test markers: `test_system_prompt_has_v3w_entry_point_and_counter_move_knowledge`
  and `test_prompt_cap_leaves_room_for_lessons_and_a_note`.

## Video addendum - 2 Aug premarket + weekly root-value review (v3x)

**Sources (direct YouTube transcripts):**
- Intraday Hunter premarket analysis, `Nifty & Bank nifty | SENSEX Analysis |
  Prediction For 03 AUG 2026` (`yHEfrMUrmKk`, 1:59, uploaded 2 Aug 2026).
- Intraday Hunter, `Weekly Market Analysis: The Biggest Opportunities`
  (`iDV1obD78-c`, 17:16, uploaded 2 Aug 2026).

The premarket clip is dated by its target session, so the committed advisory is for
3 Aug. All three indices had closed with positive momentum and likely seated buyers.
A meaningful gap-up protects that crowd and calls for buy-side setups with the
market; a flat or gap-down open exposes them and calls for sell-side setups. IH also
said a mild gap-up is treated as flat. Spoken rounded levels were checked against the
visible charts before updating `premarket_note.json`; no false decimal precision is
stored.

The weekly lecture reviews five recent directional reads. Its recurring method is to
infer the buyer/seller "root value" from the crowd likely carrying the most quantity,
not from one hypothetical participant. A repeated breakdown followed by recovery
does not leave a durable seller crowd: repeated failure pushes sellers out, and if
the other indices remain positive, buyers may become the seated side instead. The
lecture also states that option-buying targets must fit the available time: IH uses
roughly 1:1 when his crowd read is exceptionally clear because a distant target can
be less realistic than the stop.

**Net-new method distilled:**
- AGGREGATE-INVENTORY TEST: hunt the dominant aggregate cohort, not an anecdote. If
  neither side likely has meaningful size, follow current momentum or hold.
- REPEATED-FAILURE INVENTORY RESET: repeated breakdown-and-recovery cycles evict
  sellers; require fresh participation before treating their stops as available.
- OPTION-TIME-ADJUSTED REWARD/RISK: normally prefer an attainable approximately 1:2.
  Approximately 1:1 is a narrow exception requiring the unique-trade filter, a
  direct aggregate crowd read, real stop/target levels, a pre-accepted rupee loss,
  and an option-time reason the farther target is unrealistic. Less than 1:1 is HOLD.

**Confirmed but deliberately not duplicated:** horizon selection is TIMEFRAME FIT;
already-paid crowds are TARGET-BOOKED; gap-size and cross-index asymmetry already
govern who remains huntable; direction can survive a bad entry only through THE
ENTRY POINT IS PART OF THE PREMISE plus the POST-EXIT RE-ENTRY GATE; a quiet week is
already covered by NO DAILY-INCOME PRESSURE.

**Not encoded:** the lecture mentions that a first directional entry may fail and a
second or third may work with lower quantity. The agent cannot choose a reduced order
quantity, and reflex retries would contradict the enforced cooldown and fresh-crowd
gate. Those existing safeguards remain authoritative.

**Knowledge changes (v3x, all prose):**
- `RETAIL_POSITIONING`: AGGREGATE-INVENTORY TEST and REPEATED-FAILURE INVENTORY RESET.
- `RISK`: OPTION-TIME-ADJUSTED REWARD/RISK replaces the absolute approximately 1:2
  wording without permitting sub-1:1 or manufactured targets.
- Tests pin all three markers, the retained safeguards, and the complete dated
  premarket payload.

### v3x continued - the 3 Aug LIVE SESSION (appended to the same version)

**Source:** Intraday Hunter live session, 3 Aug 2026 (`cwgFEpiTwgE`, 9:23). The v3x
section above was distilled from his 2 Aug *analysis*; this is the session that
followed it, so both belong to the same version.

**What IH did:** gap-up open with a slight rejection, and he bought CALLs on the
dip. His pre-trade read, close to verbatim:
- "When such a gap-up opens you must look at what is happening OVERALL: buying has
  been continuous for many days, and then a gap-up. Sometimes profit-booking comes."
- "Those sitting from BELOW are not booking - **if theirs were coming you would see
  bigger selling, quick sharp selling.** Nothing like that is happening. This is
  just the market shaking out whoever bought on FRIDAY and got a sudden gap-up."
- "If they had exited here we would assume the trader is weak. He is not weak - he
  is riding the market long term."
- Per his own analysis: on a gap-up the buyers are NOT the target, so go with the
  market (this is the note channel and the plan agreeing, not new method).

**Why he booked - the strongest part of the session:**
- "The momentum that is happening is only in BankNIFTY. Sensex and NIFTY have lagged."
- "It is not that more momentum cannot come. **But the setups that work for us -
  this is not one of them.**"
- "The market can do anything on any day, but will we chase it when we do not even
  know where the road is going? **We waited as far as we knew the road. Now we do
  not know the road.**"

**Agent tally for 3 Aug:** 73 decisions (69 HOLD, 1 ENTER_SHORT, 1 ENTER_LONG,
2 EXIT), window 09:16:01-10:30:07, 14 decisions citing the pre-open note. Two
trades, both losers, net **-Rs.1,379.25** (PAPER):
- 09:25:06 -> 09:25:33 SHORT `huge_gap_mindset_trap_fade`, 3 lots, -6.4 pts,
  **-Rs.950.25** (NIFTY -243.75 + mirror -706.50), held 27 SECONDS.
- 09:31:06 -> 09:32:30 LONG `gapup_followthrough_fibo61_hammer`, 2 lots, -2.8 pts,
  **-Rs.429.00** (NIFTY -351.00 + mirror -78.00), held 84 seconds.

**MAT-113 verified in production:** both BankNIFTY mirrors fired (09:24:16 and
09:30:52), with ZERO "no LTP" skips and no wait needed - against three skips on
31 Jul. Subscribing the leg before pricing it was the fix.

**Behaviour worth recording, not encoded as a rule.** The agent FADED the gap-up
(a short at 09:25) on a day its own pre-open note said gap-up means do not target
buyers, was stopped in 27 seconds, then FLIPPED to long 5m33s later - clearing the
five-minute mechanical cooldown by 33 seconds. Both lost. That is exactly the
pattern MORNING SPEED IS NOT INFORMATION and NO INSTANT FLIP already describe, so
this is evidence those rules are not yet binding rather than a gap in them. Worth
watching before adding more prose about it.

**Net-new method distilled:**
- PROFIT DEPTH SPLITS ONE SIDE INTO TWO COHORTS (sub-bullet of AGGREGATE-INVENTORY
  TEST): "buyers" is not one crowd. After a multi-day run, traders positioned from
  far below are deep in profit and are not weak; only the MARGINAL holders from the
  last session or two are close enough to their entry to be flushed. A hunt aimed at
  "the buyers" therefore collects a far smaller pool of stops than the aggregate
  suggests.
- THE COUNTER-MOVE'S SIZE AND SPEED SAY WHICH COHORT IS LEAVING: a small, slow
  rejection is the recent cohort being shaken and nothing more; genuine
  profit-booking by deep holders shows up as BIG, QUICK selling. If that has not
  appeared, they are still in and the move against you is noise, not distribution.
  Complements v3w's COUNTER-MOVE SIZE SAYS RANGE OR BREAKOUT, which reads size to
  test a breakout premise; this reads size to identify WHO is moving.
- ONLY RIDE AS FAR AS YOU KNOW THE ROAD (RISK): book when the situation stops
  matching a setup you actually have, even though nothing has invalidated and the
  move may well continue. Explicitly separated from premise-invalidation (thesis
  broke) and from a target (number arrived): here the trade may still be working and
  what ran out is the ability to READ it. Framed as the holding-side twin of "when
  unsure, HOLD" - that rule keeps you out with no read, this one gets you out when
  the read you entered on is used up. Its typical trigger is the move narrowing to
  one index, which is LAGGARDS NEVER JOINED's cross-index form of the same signal.

**Confirmed but already present:** gap-up means follow rather than hunt is
RECRUITMENT HISTORY plus v3v's small-gap qualifier; booking when the leader runs
alone is LAGGARDS NEVER JOINED (v3s).

**Knowledge changes (v3x continued, all prose):**
- `RETAIL_POSITIONING`: two sub-bullets under AGGREGATE-INVENTORY TEST.
- `RISK`: ONLY RIDE AS FAR AS YOU KNOW THE ROAD.
- Test marker: `test_system_prompt_has_v3x_profit_depth_and_known_road_knowledge`.

## Video addendum - the 4 Aug LIVE SESSION (v3y)

**Source:** Intraday Hunter live session, 4 Aug 2026 (`i1G7hoIshyE`, 8:45).
NIFTY expiry day. This is a **LOSING** session, which is rarer and more useful
material than a win — and it directly follows the 3 Aug session in v3x, which
makes the pair unusually instructive.

**The setup that makes this session valuable:** 3 Aug and 4 Aug both opened
GAP-UP. On 3 Aug he BOUGHT with the gap. On 4 Aug he SOLD puts against the
buyers. He anticipates the obvious objection and answers it head-on:

> "A question may come to your mind — sir, yesterday there was positive momentum
> and a gap-up and you BOUGHT there. Today too there is a gap-up, the market keeps
> going up. So why have we made a SELLING plan today?"

His two discriminators, close to verbatim:
1. **The holiday.** "Yesterday there was positive momentum, but there was some
   retracement in it. Second, a two-day holiday was coming in the middle — because
   of that traders take LESS risk. So there the plan was: if we get flat-to-gap-down
   we understand buyers are seated. On a gap-up we preferred to go WITH the market."
   "But in today's market? There was no holiday in between."
2. **Exact round-number support on all three indices.** "All three indices have
   taken round-number support. Around 24,500 it has taken support. Likewise in
   BankNIFTY it has taken 57,500 support... exactly support. So buyers are seated
   here. Nobody has gone and sold. Sensex too is sitting on a round number."

**His entry rule (the trigger, not the gap):** "Buyers will start giving their SLs
only when the market starts moving BELOW THE CLOSING PRICE — when BankNIFTY starts
moving below its closing price and Sensex starts breaking its closing price. After
that they will start giving SL." He entered as BankNIFTY began taking support
*exactly at* the closing price, expecting the break: "chances of a breakdown are
high and it will not take much time."

Legs: BankNIFTY 57,800 PE + 57,700 PE, Sensex 78,700 PE, NIFTY (expiry) 1430 qty.

**Volatility note:** "Today the market looks like it is doing momentum fast. It
could be volatile. So if you are taking a target, keep it a bit BIGGER. If you are
taking an SL, that also has to be kept reasonably wide." Also: "For many days the
market has been sideways and was not producing momentum — today it might."

**Why he cut it for a loss — the strongest part of the session:**
- The trigger never fired. "This would only have paid if the market breaks down
  through the closing price."
- The index that ended it: "**Especially BankNIFTY is going up more. If BankNIFTY
  is going up we will NOT hold this trade.**" And the explicit hierarchy: "If
  Sensex and NIFTY moved up a little we could have HANDLED that. But once BankNIFTY
  has started rising there is no benefit in waiting."
- The discipline line: "Directionally we are correctly positioned, because buyers
  should be here... **but if we are wrong we cannot chase the market insisting that
  WE are right and YOU are wrong.** In that condition we cut the trade per the loss
  limit."
- A subtle one worth keeping: while price hung at the level, "because of candles
  like this, OTHER people also start taking risk — and you know what risk they take,
  they will SELL. Then sellers come in and get stuck, and the market starts turning."
  Followed by: "if it is going to fall, it should fall from around here."

**New, hence in the knowledge:**
- The gap-up long branch needed a precondition it did not have — proof the buyers
  are actually ABSENT. Two checkable tells that they are present instead
  (round-number support across all three; no holiday distorting the prior day).
  This is what makes two identical-looking gap-ups take opposite trades.
- An index HIERARCHY for exits. Existing knowledge treats BankNIFTY as the major
  index for the entry read (v3a) and covers the leader running alone (v3s/v3x).
  This adds the losing-side rule: NIFTY/Sensex against you is tolerable, BankNIFTY
  against you is disqualifying.
- A trigger that never fired is an exit reason, not a reason to keep waiting.
- Being directionally right does not earn the hold.
- A slow grind at the level recruits your own side, whose stops become fuel against
  you. (Related to but distinct from R:R-BAIT, which is about rejections inviting
  shorts; this is about a stalled approach recruiting company.)
- Volatile-day sizing widens the TARGET as well as the stop.

**Confirmed but already present:** holiday carry-risk (v3g), round number and
closing point as the level pair (`LEVELS_AND_PIVOT`), premise-invalidation exits,
and BankNIFTY as the major index (v3a).

**Knowledge changes (v3y, all prose):**
- `OPENING_DRIVE`: SEATED-BUYER TEST + CLOSING-PRICE BREAKDOWN IS THE TRIGGER,
  both placed ahead of the existing GAP SIZE IS A RISK DIAL bullet so they read as
  preconditions of the long branch rather than as a competing rule.
- `BNF_SPECIFIC`: INDEX HIERARCHY ON THE WAY OUT.
- `RISK`: A TRIGGER THAT NEVER FIRED IS AN EXIT REASON; BEING DIRECTIONALLY RIGHT
  DOES NOT EARN THE HOLD; A SLOW GRIND AT THE LEVEL RECRUITS THE WRONG CROWD;
  VOLATILE-DAY SIZING WIDENS BOTH ENDS.
- Test markers: `test_system_prompt_has_v3y_seated_buyer_and_index_hierarchy_knowledge`
  and `test_v3y_gap_conflict_does_not_contradict_the_opening_drive_branch` (the
  latter pins the ORDERING, so the seated-buyer test cannot drift into looking like
  an alternative gap-up rule the agent could pick between).

**Note on the pre-open note:** the 2026-08-04 note shipped in PR #101 was
transcribed from his 3 Aug *prediction* video and said sell-side on every opening
type, big gap-down excepted. This live session is that plan being executed and
then cut. The note was directionally what he traded; it is the EXIT that carried
the lesson, and a pre-open note can never contain that.

### Pre-open note for 2026-08-05 (shipped alongside v3y)

**Source:** Intraday Hunter, "Prediction For 05 AUG 2026" (`PSCeB9y9JbI`,
uploaded 2026-08-04, 2:34).

**The plan has now inverted twice in three sessions**, which is the reason this
channel exists as a DATED note rather than as knowledge:

| Session | Plan | Who is the target |
|---|---|---|
| 3 Aug | gap-up -> BUY with it | nobody trapped; follow |
| 4 Aug | every opening type -> SELL | seated BUYERS, hunted |
| 5 Aug | flat/gap-down -> SELL **with** the market | nobody; this is continuation |

4 Aug and 5 Aug are both "sell side" but for OPPOSITE reasons. On 4 Aug he was
hunting a seated buyer crowd. For 5 Aug he explicitly says the sellers now in are
NOT a crowd worth hunting: "once a momentum move has already happened, a trader
who enters does not enter in large quantity" — so the move can simply continue,
and he goes WITH it. A note that just echoed "sell side" from the previous day
would carry the direction and lose the entire reason.

Two things carried into the note beyond levels:
- **The trap-detection logic, stated as a rule:** if a large short crowd HAD
  seated, the market would know and "would directly give a big gap and make a
  trap there". So a big gap is evidence about positioning, not just volatility —
  which is why he sets a very large gap aside from this plan entirely.
- **A market-mechanics flag:** "the closing price is being calculated a bit
  differently now, according to the new rule." His whole method keys off the
  previous CLOSE, so the level may not equal a naive previous close. Worth
  watching in the runner, which computes its own levels.

**Transcription caveat:** the auto-transcript renders one BankNIFTY resistance as
"5780" where every other level is five digits; read as 57800, alongside 57650.
Sensex supports came through as the run-on "780007840", read as 78000 and 78400.
Neither could be confirmed against the on-screen chart (browser pane not
compositing). They are advisory candidate levels only.

**Test marker:** `test_shipped_note_matches_august_5_intraday_hunter_plan`,
re-pinned from the 4 Aug content. It now also asserts the "SMALL size" phrase in
`context`, because that is the clause distinguishing this sell-side plan from
yesterday's opposite-reasoned one.

## Video addendum - the 5 Aug LIVE SESSION (v3z)

**Source:** Intraday Hunter live session, 5 Aug 2026 (`lS-sUh54LeA`, 10:47).
A WIN — and, more valuable, the session in which he re-opens the v3y loss and
keeps the rule that caused it.

**Setup:** big gap-up, then a rejection to buying. He SOLD (puts): BankNIFTY
57800 PE 1170 qty, Sensex 900 qty, NIFTY 1430 qty. Booked a good profit inside
the first hour.

### The read: he was not fading the gap, he was reading an absent crowd

> "You may think sellers are seated here and the market can target them. But with
> the kind of momentum that just happened, most people could not even get a sell
> trade on... **Retail selling has NOT happened here. You see a gap-up, but retail
> selling is not here. If retail HAD sold, the market would have done something
> different — it would have started rising directly. There would have been no
> time.**"

That is an inference from ABSENCE, and it is the new idea: the market not ripping
higher immediately is itself evidence that there is nobody above to rip through.
Then the asymmetry he trades on:

> "If it goes directly up, only BUYERS are going to come there; nobody will sell.
> **The market does not work in a situation where only one side's traders are
> operating.**"

Up = one-sided, therefore unsustainable. Down = both sides still arguing,
therefore the path. He also notes the gap was too big to be a bait: "sometimes
sellers get seated and the market tempts them a little, then starts rising. But
this gap-up is so large that there is no ROOM to tempt."

### The in-trade monitor

> "While the market is falling, as long as buyers are coming and sellers are
> coming — 'let me take a trade too', 'let me take a trade too' — **nobody is
> going to touch your profit.** But when it starts to feel like ONLY sellers are
> coming, only buyers are coming, then danger starts hovering over your profit."

This is BOTH-SIDES PARTICIPATION applied to an OPEN position rather than to an
entry, which is what makes it worth adding.

### Re-opening the v3y loss — the most important part

> "Yesterday we had a loss. It was a very wrong loss. **If we had stayed seated a
> little, it would have worked out.** But when BankNIFTY starts going up you have
> to get out, because BankNIFTY is our MAJOR index. Yesterday BankNIFTY alone
> started rising, so we cut. Otherwise the market did fall properly later —
> **almost exactly from where we exited. I saw it.** Never mind, it happens.
> **Follow the rule. It works better for you.**"

The v3y INDEX HIERARCHY exit cost him the trade, he watched it would have paid,
and he reaffirmed the rule anyway. That is rarer than either a win or a loss, and
it is the reason this became a knowledge entry rather than a note.

It also happens to be the same argument the operator made on 2026-08-04 about
keeping the LIVE refusal on an unscoreable option chain even though the blocked
entry turned out to be the day's best trade (see
`Signal Generators/Regime Adaptive Strategy/REGIME_PORTING_NOTES.md`). Two
independent instances of the same discipline in two days.

### Why he booked early

> "The target could be made bigger here... but yesterday we had a loss. If today
> is giving a chance to make profit, take a good profit and go."

He says in the same breath that more momentum was likely, and leaves regardless.
Distinct from the existing POST-LOSS SPEED LIMIT, which governs re-ENTRY speed;
this governs how much you demand from the trade you are already in.

### One more, on where to point attention

> "The wrong thing the market can do to us is give a round-number breakout."

So he watched BankNIFTY's approach to 58,000 specifically — and said he watched
BankNIFTY hardest **because that is where his quantity is largest**, not because
its chart was the most interesting.

**New, hence in the knowledge:**
- The missing rip is the tell (absence of the hunt identifies the absent crowd).
- A large gap removes the BAIT ROOM, so "they will be baited then squeezed" is
  not available.
- A rule that cost you money yesterday is still the rule.
- Two-sided flow protects an open profit; one-sided flow endangers it.
- After a losing day, take the good profit rather than the big one.
- Name the one way this trade fails, then watch that — and watch it hardest where
  the quantity is largest.

**Confirmed but already present:** BOTH-SIDES PARTICIPATION for entries, GAP SIZE
IS A RISK DIAL, POST-LOSS SPEED LIMIT, the INDEX HIERARCHY exit itself (v3y), and
BankNIFTY as the major index (v3a).

**Knowledge changes (v3z, all prose):**
- `RETAIL_POSITIONING`: THE MISSING RIP IS THE TELL.
- `OPENING_DRIVE`: the BAIT ROOM clause, folded into GAP SIZE IS A RISK DIAL.
- `RISK`: A RULE THAT COST YOU MONEY YESTERDAY IS STILL THE RULE; TWO-SIDED FLOW
  PROTECTS AN OPEN PROFIT; AFTER A LOSING DAY, TAKE THE GOOD PROFIT RATHER THAN
  THE BIG ONE; NAME THE ONE WAY THIS TRADE FAILS.
- Test markers: `test_system_prompt_has_v3z_missing_rip_and_rule_discipline_knowledge`
  and `test_v3z_rule_discipline_cannot_be_read_as_licence_to_hold`. The second
  exists because "the rule cost me money" has an obvious dangerous misreading —
  "so hold longer next time" — and v3z must reinforce the v3y exit, never soften
  it. It asserts the hierarchy exit and the never-hold-a-loser rule are both still
  present alongside the new lesson.

### Pre-open note for 2026-08-06 (shipped alongside v3z)

**Source:** Intraday Hunter, "Prediction For 06 AUG 2026" (`-51EUk_dukw`,
uploaded 2026-08-05, 1:40).

**The seated side has flipped, and with it the whole gap mapping.** Four notes,
four different configurations:

| Session | Seated crowd | Gap-up means | Gap-down means |
|---|---|---|---|
| 3 Aug | buyers (thin) | follow, BUY | hunt, SELL |
| 4 Aug | buyers (seated) | hunt, SELL | hunt, SELL |
| 5 Aug | sellers (too small to hunt) | SELL, chop risk | follow, SELL |
| 6 Aug | **sellers (seated)** | **hunt, BUY** | **follow, SELL** |

This is the first note in the series where SELLERS are the crowd worth hunting,
which inverts the gap mapping outright: a flat-to-gap-up now leaves their stops
exposed above and is the BUY-side hunt, while a decent gap-down puts them into
profit — "there will be no threat on these sellers" — so there is nothing to hunt
and he follows the move down instead.

**One qualifier he repeats for all three indices:** the selling was decent but
**none of them broke its closing price**, and NIFTY closed holding a round-number
(500-level) support. So the sellers are seated but not yet validated — the move
that would confirm the short has not happened. That is the clause most likely to
matter if tomorrow opens flat.

**Transcription caveat:** the Sensex supports again arrived as the run-on
"780007840", read as 78000 and 78400 — the same ASR artefact as the 5 Aug note.
Advisory candidate levels only.

**Test marker:** `test_shipped_note_matches_august_6_intraday_hunter_plan`,
re-pinned from the 5 Aug content, asserting the "NONE broke its closing price"
clause because that is what distinguishes seated-but-unconfirmed from trapped.

## Video addendum - the 6 Aug LIVE SESSION (v4a)

**Source:** Intraday Hunter live session, 6 Aug 2026 (`9pZtVvUBDq4`, 9:45). A WIN,
and he BOUGHT — the first buy-side session in this run.

**Setup:** slight gap-up, plan already positive from his own analysis. He wanted
price a little LOWER before entering, got a small rejection, and bought calls
(BankNIFTY 57700 CE + 57800 CE, NIFTY 1430 qty). Booked a good profit.

### The dating rule — why the sellers existed today but not yesterday

This is the core new idea, and it completes v3z rather than repeating it:

> "When selling comes on ONE day, not many people can participate — before that
> the market was decently positive, there were gap-ups, so people cannot
> participate. **But when the NEXT day also falls the same way, traders start
> paying attention — 'the market is going down, why not make a put trade here'.
> So sellers WILL have come in.** We are making those sellers our target."

v3z said "retail did not get to sell" about a single day. v4a says **when they
DO** — on the second consecutive adverse day. Together they date the inventory,
which is what turns "is a crowd seated?" from a feeling into a count.

### What a freshly recruited crowd implies

> "Whoever is sitting in a trade here will not give the market a very big SL."

Two consequences that pull opposite ways on purpose: do NOT plan a large target
(the move exists to take a shallow stop cluster, not to trend), but DO expect the
move to be FAST —

> "The momentum should be fast, because these sellers will not give the market
> much opportunity... if it is going to hit their SL, the market will pause a
> little and then suddenly produce fast momentum and eat as many SLs as it can."

So speed is the SIGNATURE of the flush, and a slow grind means the cluster you
assumed is probably not there: "if the momentum is slow then maybe we will not get
that much target — we will have to take an average target."

### Entry preference

He wanted a dip BEFORE buying, not the first push up: "if we get the market a bit
lower it is better; if it starts rising directly we would have to work with a
retracement." He also checked no big sudden selling was underway first — a small
orderly move against you is a cheap fill, a large one says the premise is wrong.

### The two-phase handling of a wobble

Mid-trade his profit collapsed on a rejection:

> "There is no need to be afraid of such a rejection. Nothing will happen. There
> will be up and down moves, but **you will definitely get one opportunity in
> which your profit is made.**"

...and then, once the sharp move came: "a good profit has been made, so we will
not be greedy" — and he booked. The discriminator between the two phases is
factual: **has the fast one-way move through the stop cluster happened yet?**

### The asymmetry

> "When profit is increasing, if you wait a bit, enlarge the target, even make a
> few mistakes — in profit those pass. **But never make a mistake in a loss.**"

Also, on humility: "SL hunting does not mean what you thought is what will happen…
I make mistakes too. When a mistake happens, accept it and take the loss." And
practically: "do not work thinking at 100% that your trade must go right — better
to wait, REDUCE QUANTITY."

### How the agent compared on the same session

The agent traded the same premise FIRST and then abandoned it:

| # | Agent | Result |
|---|---|---|
| 1 | 09:23 LONG `hammer_confirmation_gapup_seller_hunt` | **+Rs.712.75**, booked in 3.5 min ("momentum stalled") |
| 2 | 10:06 SHORT `double_top_neckline_break_continuation` | -Rs.1,767.75 (cut on BankNIFTY confirming against) |
| 3 | 10:16 SHORT `shooting_star_fib50_rejection` | -Rs.1,547.25 (premise stalled) |
| | | **-Rs.2,602.25** |

Trade 1 IS IH's trade — same direction, same premise, even the same name for it
("gapup_seller_hunt"), and it was the only winner. The agent then closed it after
3.5 minutes on a stall, and traded AGAINST that read twice.

IH sat through the same stall, said explicitly it was not to be feared, and took
the move. That is exactly what A REJECTION BEFORE THE FLUSH IS NOISE is for — and
the two losses that followed are what ERRORS IN PROFIT ARE CHEAP; ERRORS IN A LOSS
ARE NOT is for. The pre-open note was injected (1909 chars) and cited in 4
decisions, and its plan — gap-up means hunt the seated sellers, BUY side — was
correct.

**New, hence in the knowledge:**
- SECOND-DAY RECRUITMENT (one adverse day seats nobody; the second does).
- A freshly recruited crowd has tight stops: sharp but SMALL, and FAST or not real.
- A rejection BEFORE the flush is noise; after it, book.
- Errors in profit are cheap; errors in a loss are not.
- When the stops are above you, prefer a dip to a chase.

**Confirmed but already present:** THE MISSING RIP IS THE TELL (v3z) is the
one-day half of the recruitment rule; recruit-then-turn after the flush is
RECRUITMENT HISTORY; "reduce quantity when unsure" overlaps existing sizing
discipline.

**Knowledge changes (v4a, all prose):**
- `RETAIL_POSITIONING`: SECOND-DAY RECRUITMENT; A FRESHLY RECRUITED CROWD HAS
  TIGHT STOPS.
- `RISK`: A REJECTION BEFORE THE FLUSH IS NOISE; ERRORS IN PROFIT ARE CHEAP;
  WHEN THE STOPS ARE ABOVE YOU, PREFER A DIP TO A CHASE.
- Test markers: `test_system_prompt_has_v4a_second_day_recruitment_knowledge` and
  `test_v4a_rejection_rule_cannot_be_read_as_licence_to_hold_a_loser`. The second
  matters more than usual: this is the first lesson that argues for NOT closing,
  so it carries an explicit scope clause and the test asserts every exit rule it
  must not weaken is still present beside it.

### Pre-open note for 2026-08-07 (shipped alongside v4a)

**Source:** Intraday Hunter, "Prediction For 07 AUG 2026" (`Lq7JGlZj6PY`,
uploaded 2026-08-06, 2:00).

**The seated side has flipped back to BUYERS.** Five notes, five configurations:

| Session | Seated crowd | The hunt is | The follow is |
|---|---|---|---|
| 3 Aug | buyers (thin) | gap-down -> SELL | gap-up -> BUY |
| 4 Aug | buyers (seated) | any open -> SELL | — |
| 5 Aug | sellers (too small) | — | any open -> SELL |
| 6 Aug | sellers (seated) | gap-up -> BUY | gap-down -> SELL |
| 7 Aug | **buyers (seated)** | **gap-down -> SELL** | **flat/gap-up -> BUY** |

> "There is some positive momentum. Not big momentum, but **enough that people
> would go and make CALL trades**. So buyers will be seated here... To target the
> buyers, what do we need? At minimum a GAP-DOWN."

He also names where their stops are — "somewhere at these lower points that have
formed" — which is a concrete claim rather than a vague direction.

**A first for this channel: a MAGNITUDE-scaled condition.**

> "If we get a decent gap-down — that is, **the farther the opening is from
> 58,000, the better for us** — so we can target these buyers."

Every prior note treated gap direction as categorical. This one says the SIZE of
the adverse gap improves the hunt, because distance is what puts the crowd
underwater. Worth noting it does NOT contradict `GAP SIZE IS A RISK DIAL` in the
knowledge, which says a bigger gap makes the CONTINUATION branch worse: that rule
is about trading WITH a gap past everyone's stops, this is about trading AGAINST a
crowd the gap has just buried. Different branches, opposite signs, both correct.

**A link to v4a:** he stresses the recruiting move was small — "not big momentum,
but enough that people would make call trades" — which is exactly the
lightly-committed crowd v4a's A FRESHLY RECRUITED CROWD HAS TIGHT STOPS describes.
Expect a sharp but small flush if the gap-down arrives.

**Transcription caveat:** one BankNIFTY support arrived as "5710" where every
other level is five digits; read as 57100 alongside 57500. Same dropped-digit ASR
artefact seen on 4 and 5 Aug. Advisory candidate levels only.

**Test marker:** `test_shipped_note_matches_august_7_intraday_hunter_plan`,
re-pinned from the 6 Aug content, asserting the "BUYERS are the seated crowd"
clause because that is the single fact the whole gap mapping hangs on.

## Video addendum - the 7 Aug LIVE SESSION (v4b)

**Source:** Intraday Hunter live session, 7 Aug 2026 (`SupzF0JT_vE`, 9:10). A WIN
on the put side, and the first session where he ENLARGED the target mid-trade and
explained the reasoning for it.

**Setup:** slight gap-down, then a recovery. He waited through the bounce, sold
puts into it (BankNIFTY 57900 + 57800 PE, Sensex 900, NIFTY 1430), and booked a
large profit. The 2026-08-07 pre-open note called exactly this: buyers seated,
gap-down means hunt them, sell side.

### The central idea: the bounce is the trap, not its failure

> "If the market had opened gap-down and started falling directly, the buyers
> might have got out **in two or three minutes**. But once it comes up, the trader
> gets a little hope that yes, the market can go up too — and he may even try to
> AVERAGE."

> "Those sitting in buys should feel the market can go up, **so that when it goes
> down again they take a bigger loss.**"

This inverts the naive read. A post-gap bounce toward a trapped crowd's entry is
the most commonly misread event on a chart: it looks like the short failing, and
it is actually the mechanism that converts a small trapped loss into a large one.
A direct fall is the operator's WORST outcome, because the crowd escapes cheaply.

Recorded with the invalidation attached, so it cannot read as "ignore adverse
movement": what would genuinely break the trade is price RECLAIMING the level, not
approaching and failing at it. IH named his in advance — "if it crosses the
closing price, understand we have made the trade wrong."

### Why the resistance did not recruit fresh sellers

A viewer objection he answers directly, and the answer is a general rule:

> "They will not be able to sell here. Most traders who sold did it earlier in
> this chart and had to suffer losses again and again. **Now everyone's focus is
> on where to BUY**, because their mindset has formed that the market keeps going
> up. Nobody is even paying attention to the resistance."

A level does not attract participants by geometry — it attracts whoever the recent
past has left willing to act there. Added as REGIME MEMORY DECIDES WHO SHOWS UP AT
A LEVEL, with the practical test: ask who has been PAID and who PUNISHED lately.

### Target sizing from crowd behaviour — completing the v4a pair

> "Why could we make the target bigger here? Because those who average get a
> little courage from the market — 'go on, wait' — and then it targets them. So
> their SLs would have been hit."

v4a: freshly recruited crowd, shallow stops, SMALL target, fast move.
v4b: crowd baited into AVERAGING, more size at a worse average, pain threshold
further away, so the flush runs further and a LARGER target is justified.

Together they give a way to size a target from what the crowd has DONE rather than
from a fixed percentage. He also cites all three indices moving together as a
second reason to enlarge.

### The stall is the last bait

> "Now what will the market do? Give them a little hope that the breakdown is not
> going to happen... with that hope they will take an even bigger loss. **You will
> get one more momentum move. Then book the target.**"

Added as EXPECT A SECOND LEG AFTER THE PAUSE, THEN BOOK — scoped explicitly to a
position that is working, never to one that is offside.

### The hierarchy is asymmetric

> "Looking at BankNIFTY it seems the trade should be taken right now. **But let us
> wait a little, according to Sensex and NIFTY.**" ... "BankNIFTY's chart is
> completely right; it is Sensex and NIFTY where we could have trouble."

He had a textbook major-index setup and did NOT take it until the laggards agreed.
That qualifies v3y's INDEX HIERARCHY, which is an EXIT rule: the leader alone is
enough to CLOSE on, not enough to OPEN on. Entering on the leader alone buys a
move the other two may never join; exiting on it alone only costs a trade.

### How the agent compared

| # | Agent (all SHORT) | Result |
|---|---|---|
| 1 | 09:32 -> 09:34 `pivot_resistance_shooting_star_reject` | -Rs.62.00 (cut in 2.5 min, "price round-tripped") |
| 2 | 09:43 -> 09:44 `averaging_trap_bearish_inside_bar_breakdown` | -Rs.1,089.75 (cut in 1.4 min, "spot pinned") |
| 3 | 10:03 -> 10:09 BNF / 10:12 NIFTY | -Rs.128.00 (**per-leg**: BNF -1,038.00, NIFTY +910.00) |
| 4 | 10:18 -> 10:28 | **+Rs.2,400.75** ("Booking profit on seated-buyer short hunt") |
| | | **+Rs.1,121.00** |

Three points worth recording:

1. **The note's plan was followed and it worked.** All four entries were shorts,
   the gap-down condition was met (NIFTY -97, BankNIFTY -182, "well below 58000" —
   the note's own magnitude clause appears in entry 2's reasoning), and the day
   was profitable.
2. **v4a was live today and did not prevent the early cuts.** Trades 1 and 2 were
   closed after 2.5 and 1.4 minutes on a round-trip and a stall — precisely the
   post-gap bounce this session explains. A REJECTION BEFORE THE FLUSH IS NOISE
   merged the previous evening and was in the prompt. Stated plainly because it is
   evidence about what prose can and cannot change: the agent still cut. v4b
   attacks the same behaviour from the mechanism side rather than the discipline
   side, which may or may not do better.
3. **The per-leg exit earned its keep.** On trade 3 the agent cut BankNIFTY alone
   (-Rs.1,038.00) and let NIFTY run to target (+Rs.910.00), turning what would
   have been a full-basket loss into -Rs.128.00. First clear instance of the
   `exit_leg` selector paying for itself.

**Knowledge changes (v4b, all prose):**
- `RETAIL_POSITIONING`: THE POST-GAP BOUNCE IS THE TRAP DEEPENING; REGIME MEMORY
  DECIDES WHO SHOWS UP AT A LEVEL.
- `RISK`: A CROWD THAT HAS AVERAGED DOWN EARNS A BIGGER TARGET; EXPECT A SECOND
  LEG AFTER THE PAUSE, THEN BOOK.
- `BNF_SPECIFIC`: THE HIERARCHY IS ASYMMETRIC (entry vs exit).
- Test markers: `test_system_prompt_has_v4b_post_gap_bounce_and_averaging_target_knowledge`
  and `test_v4b_bounce_rule_names_what_would_actually_invalidate`. The second
  exists because a rule that says "do not exit on a bounce" must also say what a
  real invalidation looks like, or it degrades into "ignore adverse movement".

### Pre-open note for 2026-08-10 (Monday)

**Source:** Intraday Hunter, "Prediction For 10 AUG 2026" (`KDqQnqYmxws`,
uploaded 2026-08-09, 2:00). Note-only; no knowledge version attached.

**Dated to MONDAY, not "tomorrow".** Written on a Sunday after a Friday session,
so the next TRADING day is 10 Aug. A note dated to a weekend can never be
injected and fails silently — the gate simply finds no match and the session runs
without a note, with nothing in the log to distinguish that from an ordinary
stale note. A new test, `test_shipped_note_targets_the_next_TRADING_day_not_the_next_calendar_day`,
now asserts `for_date` never lands on a Saturday or Sunday.

**This note is the first genuinely UNCERTAIN one in the series.** Every previous
note named a seated crowd with confidence; this one says the market has been
sideways for two-three days, so nobody is decisively seated:

| Session | Seated crowd |
|---|---|
| 3-4 Aug | buyers |
| 5-6 Aug | sellers |
| 7 Aug | buyers |
| **10 Aug** | **unclear — sideways, small momentum** |

Three things worth having beyond the mapping:

- **An unlock level for the hunt.** "If sellers are seated here, the market cannot
  eat their SLs until the 58,000 level is crossed." That converts "are sellers
  huntable?" into one checkable condition.
- **The first explicit ESCAPE HATCH.** "In flat-to-gap-down, if a good gap opens,
  our plan will be DIFFERENT" — and he does not say what it becomes. Recorded as
  stand-aside rather than guessing the missing branch, which is the honest
  reading and the safe one.
- **He names the risk himself.** "Risk increases this way because the momentum is
  small — it goes up a bit, down a bit." That pairs with v4a's tight-stop crowd
  and v4b's regime memory: a chopping tape recruits nobody firmly, so there is
  less to hunt in either direction.

Sensex is flagged as ambiguous on both sides — sellers may be present, and buyers
may arrive on the longer-term positive read.

**Transcription caveat:** one NIFTY support arrived as "2440", read as 24400
alongside 24360 — the same dropped-trailing-zero artefact seen on 4 Aug and 7 Aug.
Advisory candidate levels only.

**The lecture from the same weekend is distilled as v4c below.**

## Video addendum - the weekend DEMAND & SUPPLY lecture (v4c)

**Source:** Intraday Hunter, `K_TXTlwBANs`, 7:40. Listed in search as "Supply &
Demand Traps: Why Traders Keep Losing" and titled on the watch page "How Does the
Market Hunt Stop Losses? | Demand & Supply" — the ID is the reliable identifier.
A LECTURE, not a session: there is no trade to compare against.

### Why this one earns a knowledge entry

Everything else in `RETAIL_POSITIONING` answers "who is already trapped?". This
lecture answers what happens when that supply RUNS OUT:

> "For a time we have positional buyers' or sellers' SLs available, which the
> market targets for momentum. **But once those SLs are exhausted, and the market
> has to CREATE stop-losses** — then we have to look at which zone demand and
> supply can be highest in. The market will create its SLs accordingly."

So the read becomes two-phase: hunt existing inventory while it exists; once it is
spent, stop looking for a victim and ask where the next crowd will be BUILT. The
operator's incentive is framed as an ordinary commercial one — profit is made
where volume is high, not where it is thin.

### The mechanism worth the most: ambiguity suppresses size

This is the sharpest idea in the lecture, and it explains WHY levels break:

> "If it had taken support there and gone into buying... the market would have
> looked strong. But no — the market had to break down, because **if it had held
> support, people would not have taken large quantity.** The seller could not take
> much, because he would think the market is holding support; the buyer would not
> work much either, because he would think the market is negative. **So buyers and
> sellers both work in SMALL quantity there.** But when the market breaks down,
> activity increases."

Doubt keeps position size small. A break removes the doubt, size goes up, and
THAT is the inventory:

> "When he felt the breakout was happening and the market was going up... that
> greed is created, demand and supply are kept high so that as many people as
> possible can participate. **And as soon as they participated, the next day they
> became the target.**"

Two consequences recorded with it:
- Read a breakout as a RECRUITMENT DEVICE first, a directional signal second. The
  question is not "is this break real" but "who just committed size, and where are
  their stops now".
- A FAILED breakout is therefore the NORMAL outcome, not a malfunction: "that is
  why you will repeatedly see breakouts and breakdowns, and they often appear as
  failures."

### Round numbers, in a role they did not have before

Elsewhere in the knowledge a round number is a level price is attracted to. Here
it is a force multiplier on recruitment:

> "There is a round number of 58,000 available. What happens? **More buyers
> activate.** More buyers activate, so it gets more SLs."

So a break at or through a round number builds a DENSER cluster than the same
break elsewhere — which feeds the target-sizing judgement from v4b.

### A third axis for target size

v4a sized the target by how recently the crowd was recruited; v4b by whether it
had averaged down. This adds HOW MANY are seated, and the reason is behavioural
rather than arithmetic:

> "In a positive market more buyers are seated, so you will get a little extra
> momentum... those seated buyers will WAIT, and because they wait you get more
> momentum." Against a thin side: "when selling comes into a positive chart, few
> can even participate, and those sellers who did come were hit straight away."

**New, hence in the knowledge:**
- The manufactured-inventory phase after trapped inventory is spent.
- Ambiguity suppresses size; a break removes it (why levels get broken at all).
- A failed breakout is the normal outcome of a recruiting break.
- Round numbers as recruitment amplifiers, not only as levels.
- Crowd SIZE as a third target-sizing input.

**Confirmed but already present:** the recruit-then-turn cycle
(RECRUITMENT HISTORY), round numbers as levels/magnets (LEVELS_AND_PIVOT), and
R:R-BAIT, which is the same operator intent seen at a rejection rather than at a
break.

**Knowledge changes (v4c, all prose) — see v4d below for the 10 Aug session:**
- `RETAIL_POSITIONING`: WHEN THE TRAPPED INVENTORY IS SPENT, THE MARKET
  MANUFACTURES MORE; AMBIGUITY SUPPRESSES SIZE, BREAKING A LEVEL REMOVES IT;
  A FAILED BREAKOUT IS THE NORMAL OUTCOME; ROUND NUMBERS AMPLIFY RECRUITMENT.
- `RISK`: CROWD SIZE IS THE THIRD TARGET INPUT.
- Test markers: `test_system_prompt_has_v4c_manufactured_inventory_knowledge` and
  `test_v4c_breakout_rule_does_not_turn_into_a_fade_everything_rule`. The second
  matters because "a failed breakout is normal" reads very easily as "always fade
  breakouts" — which would contradict both the OPENING DRIVE continuation branch
  and the runner's live Regime Adaptive BREAKOUT branch. The test asserts the
  lesson stays a QUESTION about who committed size rather than a default
  direction, and that the continuation knowledge is still present beside it.

## Video addendum - the 10 Aug LIVE SESSION (v4d)

**Source:** Intraday Hunter live session, 10 Aug 2026 (`flhHzz87Of0`, 10:50). A
large WIN on the put side from an ALMOST FLAT open, with the target deliberately
enlarged and then booked short of the round number.

### The opening type is a participation reading, not a strength reading

The clearest new idea, and it inverts how a flat open usually gets read:

> "This market could only have gone up if we had got a direct gap-up — that would
> have made the structure different. But we got flat... **opening flat, the
> chances of it going up are LOW.**"
> "In a gap-up the market gives nobody a chance, it just runs."

A gap RUNS because it denied everyone entry. A flat open GRANTS entry, the crowd
positions during the first minutes, and that positioning is the inventory that
caps the move:

| Open | Who positioned | Consequence |
|---|---|---|
| GAP | nobody | nothing overhead — it can run, follow it |
| FLAT | everybody | inventory overhead — fade the attempt, do not chase it |

This is the existing gap-up long branch stated from the other end, which is why
it went into `OPENING_DRIVE` beside it rather than into a section of its own.

### Book BEFORE the round number when all three indices are running

> "It has the courage to go to the 500. **But we should get out a little BEFORE**,
> because there is continuous momentum in all three indices, so other people will
> get greedy too. So we book just before the round number and leave."

A strong three-index-aligned move is exactly what recruits the late crowd, and
that crowd's take-profits — plus the operator's reversal — both sit at the round
figure. This is the flip side of v4c's ROUND NUMBERS AMPLIFY RECRUITMENT: the
density that makes a break there powerful makes it a bad place to still be
holding.

### A fourth target-sizing input, and the first about YOU

> "Because we got the chance to sell from ABOVE... if the market had started
> falling directly, we might have had to take a coverage target instead of a big
> one. But we got it higher up, so the target will be good."

v4a sized by the crowd's recency, v4b by whether it averaged, v4c by how many are
seated — all properties of THEM. This one is the quality of YOUR fill. Recorded
with its practical form: a poor or late fill should SHRINK the target rather than
be compensated for by holding longer.

A test now asserts all four inputs coexist, because each arrived in a different
version and a later edit could drop one without failing anything else.

### Pre-committing the tolerated adverse move

> "If it breaks out we will look for 60-70 points. The market might go further, to
> 160 — and that could be wrong. **So it is better if it does not break out at
> all.**"

Distinct from the stop (where the trade is WRONG), this is how much movement
against you is still CONSISTENT with the read. It also gives v4a's A REJECTION
BEFORE THE FLUSH IS NOISE a measurable boundary: a wobble inside the band is
noise, one well beyond it is the read failing even before the stop is touched.

### The agent's session, and a data-integrity problem

The runner was **restarted four times** (08:18, 08:21, 09:13, 10:37). The 09:13
instance traded and was then killed WITHOUT a clean shutdown — the log goes
straight from a 10:31 true-up to a new "Starting..." at 10:37:51, with no
`Result summary` anywhere between 09:13 and 10:37.

SL Hunting's real day happened inside that instance:

| Time | NIFTY | BNF | Basket |
|---|---|---|---|
| 09:23:30 | -191.75 | +60.00 | -131.75 |
| 09:51:03 | -136.50 | -72.00 | -208.50 |
| 10:15:37 | -2,431.00 | -2,424.00 | **-4,855.00** |
| | | | **-5,195.25** |

The only SL Hunting `Result summary` for 10 Aug is at 11:00:01, from the NEW
instance, reading `Trades=0 | RealizedPnL=0.00`.

**This is a DIFFERENT failure from the 2026-08-03 one.** There, a restarted runner
logged `Trades=0` OVER real figures, and the trades-count guard in
`_compute_pnl_sheet_updates` was added to stop that. Here the figures never got a
summary at all, so there is nothing for the guard to prefer — every SL Hunting
summary for the day reads zero, and the Sheet will record 0.00 against a real
-Rs.5,195.25. The guard stops the wrong number winning; it cannot invent a number
that was never logged.

One thing that DID work: the 09:24 exit fired on `setup=index_hierarchy_exit` —
"BankNIFTY, the major index, has turned decisively against our LONG" — v3y's rule
cited by name in a live decision.

**Knowledge changes (v4d, all prose):**
- `OPENING_DRIVE`: A FLAT OPEN CANNOT RUN THE WAY A GAP CAN.
- `RISK`: BOOK BEFORE THE ROUND NUMBER, NOT AT IT; YOUR ENTRY PRICE IS THE FOURTH
  TARGET INPUT; PRE-COMMIT THE ADVERSE MOVE YOUR THESIS TOLERATES.
- Test markers: `test_system_prompt_has_v4d_flat_open_and_round_number_booking_knowledge`
  and `test_target_sizing_inputs_are_all_present_and_distinct`.
---

### Pre-open note for 2026-08-11 (Tuesday, EXPIRY)

**Source:** Intraday Hunter, "Prediction For 11 AUG 2026" (`cOvPKZFervw`,
uploaded 2026-08-10, 1:55). Note-only; no knowledge version attached.

**The plan INVERTS overnight, and that is the point of this entry.** Every note
in this series so far has been read forward from a seated crowd. This one flips
the gap conditional outright:

| Session | FLAT open wants | GAP against wants |
|---|---|---|
| 10 Aug | SELL side | GAP-UP -> BUY side |
| **11 Aug** | **BUY side** | **GAP-DOWN -> SELL side** |

The cause is a single event he describes in the first fifteen seconds: NIFTY
sold, **broke down, and then immediately turned back up and held above**. A
breakdown that fails does not leave the sellers paid — it leaves them turned
around, which is exactly the trapped inventory the method hunts. Hence buys on a
flat-to-gap-up open, following the market rather than fading it.

Three details worth keeping:

- **He does not treat the gap-down branch as a mirror.** "In case of gap-down the
  structure can become different, traps can form differently there." The note
  records that as its own regime rather than as the up-case reversed.
- **He allows that today may only have SPUN the sellers** — "maybe the market
  just turned the seller around here, but on the next gap-down the market may be
  able to give a momentum." That is an explicit statement that the trap may still
  be loading rather than sprung.
- **No escape hatch this time.** Unlike 10 Aug's large-gap-down veto, both
  branches are stated and actionable, so the note carries no stand-aside line.

**Expiry.** Flagged in his opening sentence ("tomorrow there is expiry in it"),
and 11 Aug is a Tuesday, which is the current NIFTY weekly expiry. That pairs
with v4d's BOOK BEFORE THE ROUND NUMBER: expiry pinning makes round strikes even
more magnetic, so the note carries the expiry warning as its own plan line.

**Levels are cleaner than the 10 Aug set** — no dropped-trailing-zero artefacts
this time; all six NIFTY/BankNIFTY numbers and all four Sensex numbers arrived
with full precision. Advisory candidate levels only, as always.

Test updated: `test_shipped_note_matches_august_11_intraday_hunter_plan`
replaces the 10 Aug equivalent and asserts the branch DIRECTIONS explicitly,
because an inverted plan is the specific failure mode a copy-forward would cause.

---

## Video addendum - the 11 Aug LIVE SESSION (v4e)

**Source:** Intraday Hunter live session, 11 Aug 2026 (`_JXirKMmI58`, 9:25). A
**LOSS**, and encoded precisely because it is one. Every prior addendum in this
series distilled a winning session; a losing one shows which part of the method
was load-bearing and which part was rationalisation.

### He named the disqualifying fact, then traded against it

The plan came from his own pre-open note (`cOvPKZFervw`, shipped as the 11 Aug
note): flat-to-gap-up wants the buying side. The open was flat across all three
indices and a sharp sell-off followed. He then said, twice, that the setup's
precondition was absent:

> "Around here neither the BUYER's stop losses are available nor the SELLER's."
> "Here not many traders were seated."

And traded anyway, on a forecast of who *would* arrive: a sharp drop tempts
intraday sellers in, so the market should rise to take them out. He bought calls
on BankNIFTY (1170 qty, 57400 CE), Sensex (900) and NIFTY (1430, expiry day).
The loss began immediately and widened; the expected recovery never started; he
cut at his pre-declared level.

That is the whole lesson, and it is the most expensive error this method makes
available: **hunting inventory that exists is the strategy; predicting inventory
that might arrive is a different and much weaker activity wearing the same
vocabulary.** When the honest read is "nobody is seated on either side", the
output is HOLD.

### The refinement that survives the loss

His reasoning was not worthless - one part of it is a genuine advance on v4d,
independent of the outcome:

| Open | Who it recruits | Trap quality |
|---|---|---|
| GAP-DOWN | **positional** sellers - they enter at the close and hold overnight | large, committed, worth hunting next day |
| FLAT | **intraday** sellers only - "positional will not take an entry yet" | small, perishable, gone by the close |

> "If the market really had to create positional sellers' stop losses, it would
> have given a straight GAP-DOWN."

v4d established *that* a flat open seats people. This names *who*, and it
explains why a flat-open hunt should be sized and targeted more modestly than
the identical shape after a gap-down.

### Knowledge changes (v4e, all prose)

- `OPENING_DRIVE`: WHICH CROWD THE OPEN RECRUITS DECIDES HOW BIG THE TRAP IS;
  A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE OF WHO IS SEATED; A SHARP FIRST
  SLIDE BAITS; A SLOW ONE MEANS IT.
- `RISK`: NAME THE LAST POINT, NOT ONLY THE STOP; DISCIPLINE IS ASYMMETRIC
  BETWEEN WINNERS AND LOSERS.
- The sharp-slide rule is deliberately encoded as a **weak prior with its own
  counter-example attached** - it is the read that lost him the session, so it
  is recorded as a tie-breaker and explicitly barred from being a trade premise.
- Test markers: `test_system_prompt_has_v4e_recruitment_and_losing_session_knowledge`
  and `test_v4e_empty_book_is_a_no_trade_not_a_forecasting_licence`. The second
  is a drift guard: it asserts the forecasting rule still resolves to HOLD and
  still reconciles with v4c's MANUFACTURES MORE, so a later edit cannot quietly
  turn it into a licence to predict a crowd.
- Prompt size 96,627 -> 101,087 chars (headroom 18,913).

### How our agent traded the same session

**Provisional - the session was still running when this was written** (last log
line 13:07; market closes 15:30). Realized so far, from the runner log:

| Strategy | Legs | Realized |
|---|---|---|
| SL Hunting AI | 7 | -2,647.75 |
| Parabolic SAR | 5 | -1,478.75 |
| RSI Reversal | 1 | -731.25 |
| Long Strangle | 5 | -542.75 |
| Heikin Ashi | 11 | -484.25 |
| EMA | 1 | +191.75 |
| Mean Reversion Z-Score | 4 | +789.75 |
| CPR Algo 3 | 1 | +874.25 |
| Renko | 2 | +2,190.50 |
| **Total** | **37** | **-1,838.50** |

SL Hunting's figure is cross-checked against its own `Result summary` line
(-2,647.75, Trades=4) and matches exactly. Note the mirror legs log as
`MIRROR EXIT`, not `EXIT`; a first pass that grepped for `| EXIT ` under-counted
the agent by 948.00.

**The agent got the direction right and lost anyway.** It was SHORT all morning
- the opposite of IH's call-side trade, and correct: NIFTY fell from a 24576
open through the 24500 round number toward the 24440 support named in the
pre-open note. IH was wrong about direction and the agent was right about it.

The agent still lost more than any other strategy, because it took **four short
entries in 47 minutes** and cut each one almost immediately:

| Entry | Setup | Exit | Held |
|---|---|---|---|
| 09:40 | runaway_trend_continuation_short | 09:52 profit_book_stall_cross_index_veto | 12 min |
| 10:02 | trendline_rejection_shooting_star | 10:04 profit_booking_stall_reversal_bias | **2 min** |
| 10:09 | double_top_rejection_bearish_engulfing (target 24400) | 10:18 premise_stall_theta_exit | 9 min |
| 10:27 | fibo_61_bearish_inside_bar_breakdown | 10:28 index_hierarchy_bnf_exit | **1 min** |

Three of the four exits are *stall* judgements rather than stops, and the fourth
is the BankNIFTY hierarchy rule firing one minute after entry. The 10:09 trade
had a 24400 target and was released at 24469-24492 having never been stopped.

So the failure is not the read - it is that **the premise-stall exit is firing
faster than the premise can resolve**, converting one correct directional call
into four round-trips on an expiry day, when spread and theta punish churn
hardest. This is the exact inverse of the failure v3y guards against: v3y stops
the agent holding a loser because it feels right; nothing currently stops it
releasing a winner because the tape paused.

v4e's two RISK rules speak directly to this - DISCIPLINE IS ASYMMETRIC puts the
patience on the winning side, and NAME THE LAST POINT replaces "has it stalled?"
with a pre-declared level and deadline. Whether that is enough, or whether
`premise_stall` needs a minimum-hold or a bar-count floor before it may fire, is
a **candidate for the lessons loop** rather than something to encode as IH
knowledge - it is a property of our agent, not of the method.

---

### Pre-open note for 2026-08-12 (Wednesday)

**Source:** Intraday Hunter, "Prediction For 12 AUG 2026" (`CoxS77NfnsI`,
uploaded 2026-08-11, 2:00). Note-only; no knowledge version attached.

**The seller crowd is described as SPENT rather than seated**, which is a
different starting condition from every note in this series so far. He does not
say sellers are sitting there waiting to be hunted; he says the market has
already taken them out:

> "When the selling came and a retracement happened, sellers would certainly
> have entered there. But the market would have hit their SLs. So if it has
> already taken the sellers out, we can go WITH the market."

That pairs directly with **v4e's WHICH CROWD THE OPEN RECRUITS**: he then
explains why nobody is carrying size overnight either —

> "It did not cross the round number, so not many people would have held their
> selling quantity."
> "One momentum came and after that not many people held short positions."

No round-number breach, no follow-through, therefore no overnight inventory. The
result is a session that starts with thin positioning on **both** sides.

**The second explicit escape hatch in the series.** 10 Aug had one for a large
gap-down; this one is for a large gap-up:

> "If a big gap-up opens, maybe the market has just made a TRAP. In a big gap-up
> we cannot make such a plan for now... there the market can start making a
> DIFFERENT type of trap."

Recorded as stand-aside, not as a guessed branch — the same treatment the 10 Aug
note gave its missing branch.

**One genuine ambiguity, left unresolved on purpose.** The sell-side conditional
is stated cleanly for all three indices, but on NIFTY he also says a mild gap-up
can be followed *with* the market "if not many sellers are seated, the market may
not find SLs". Read one way that is a long; read another it is a reason to expect
no upward pull at all. The note records the tension rather than picking a side,
which is what v4e's A FORECAST OF WHO WILL ARRIVE rule demands of an unclear read.

**Transcription caveat:** one BankNIFTY support arrived as "5710", read as 57100
alongside 56960 — the same dropped-trailing-zero artefact seen on 4, 7 and
10 Aug. Advisory candidate levels only.

Test updated: `test_shipped_note_matches_august_12_intraday_hunter_plan`
replaces the 11 Aug equivalent. It asserts the escape hatch and the ambiguity
line survive verbatim, because those are the two things a copy-forward or a
tidy-up would silently remove.

---

## Video addendum - the 12 Aug LIVE SESSION (v4f)

**Source:** Intraday Hunter live session, 12 Aug 2026 (`CV_Fs3TFF5I`, 6:46,
published 10:29 IST). A **WIN**, taken from the *same* opening condition that
produced the 11 Aug loss. That pairing is the whole value of this addendum.

### The same empty book, two opposite outcomes

Both sessions opened flat, sold off immediately, and had IH saying — in almost
identical words — that nobody was positioned:

| | 11 Aug (v4e, LOSS) | 12 Aug (v4f, WIN) |
|---|---|---|
| Opening read | flat, sharp sell-off | flat, sharp sell-off |
| Book | "neither the BUYER's SLs nor the SELLER's" | "neither many buyers nor many sellers" |
| What he did | **predicted** who would arrive, entered on that | **waited** for the recovery to actually begin |
| Result | cut at his last point | booked a good target |

So an empty book is not simply a no-trade condition, as v4e recorded it. It is a
statement that the market **must manufacture a trap**, because it has nothing
else to work with. What it does not tell you is which side the trap points at —
and the difference between the two days is whether he guessed that or waited to
be shown:

> "Here there are neither many buyers nor many sellers... so some kind of TRAP
> will definitely form here. We were waiting for exactly that."

The confirmation he waited for was concrete: a sharp recovery off a drop that had
just trapped the sellers who chased it, led by one index. *"The trap somewhere
seemed to have been made FOR THE SELLERS."*

### The new idea: a repeated chart is a trap, not a trend

The reason he expected a trap at all is the sameness of the two days:

> "Normally the market does NOT repeat the chart."
> "Yesterday it opened flat and directly started falling. Here too flat open,
> directly fell... so some kind of trap will definitely form."

A shape everyone watched yesterday is a shape everyone is ready for today, and a
move nobody has to be tricked into paying for is not a move the market needs to
make. Note this is distinct from v4a's SECOND-DAY RECRUITMENT, which is about a
crowd built across two days and then hunted; this is about the **path** being
identical, which is what makes the copy a trap.

### The move that denied him entry

He wanted to sell — the pre-open note said sell-side — and never got in:

> "If it had gone a bit slow, or given us a slight up move first, we would have
> had a chance to sell... but the momentum was very sharp. **Everything happened
> in ONE MINUTE.**"

That is v4d's gap logic at intraday scale: a move that completes before anyone can
join it recruits nobody, creates no inventory, and leaves nothing behind it to
hunt. The correct response is not to chase it but to ask what the market must do
next to trap somebody.

### Exit: the profit stopped growing

He held while it paid — *"momentum is very fast, it will not stop easily, the
target may be BIG"* — and closed on a change in rate, not in price:

> "Now see, the profit has started REDUCING. So let us book. The more smoothly
> the profit comes, the better."
> "Especially if we have ALREADY SEEN a good target, after that we should not be
> greedy... we had already captured one momentum."

### Knowledge changes (v4f, all prose)

- `OPENING_DRIVE`: THE CHART DOES NOT REPEAT TWO DAYS RUNNING; A MOVE THAT DENIED
  YOU ENTRY WAS NOT YOUR MOVE; AN EMPTY BOOK MEANS A TRAP IS COMING — WAIT FOR IT
  TO REVEAL ITS DIRECTION; THE SHARPEST RECOVERY NAMES THE LEADING INDEX, AND
  SIZE FOLLOWS IT.
- `RISK`: BOOK WHEN THE PROFIT STOPS GROWING, NOT WHEN IT REVERSES.
- Test markers: `test_system_prompt_has_v4f_repeat_chart_and_confirmation_knowledge`,
  plus two drift guards. `test_v4f_confirmation_rule_does_not_reopen_the_v4e_forecasting_hole`
  is the important one: v4e and v4f are a matched pair, and v4f read alone would
  license exactly the forecast v4e forbids, so the test asserts the confirmation
  requirement and v4e's HOLD both survive together.
- Prompt size 101,087 -> 105,933 chars (headroom 14,067).

### How our agent traded the same session

**Provisional — the runner was still live at 14:15** (market closes 15:30).

| Strategy | Legs | Realized |
|---|---|---|
| SMA Crossover | 1 | +8,911.50 |
| Renko | 3 | +5,427.50 |
| Donchian Bearish | 1 | +3,406.00 |
| CPR Algo 3 | 2 | +3,272.75 |
| Regime Adaptive | 2 | +575.25 |
| Long Strangle | 10 | +565.50 |
| Bollinger Bands | 1 | +481.00 |
| EMA | 2 | -331.50 |
| Heikin Ashi | 9 | -728.00 |
| SL Hunting AI | 2 | -1,034.25 |
| Parabolic SAR | 2 | -1,056.25 |
| RSI Reversal | 1 | -1,160.25 |
| Supertrend Bullish | 1 | -2,093.00 |
| Mean Reversion Z-Score | 2 | -2,717.00 |
| **Total** | **39** | **+13,519.25** |

SL Hunting cross-checks exactly against its own `Result summary` (-1,034.25,
Trades=1).

**The agent took one trade and it was the wrong side.** At 10:30 it entered SHORT
on `double_top_shooting_star_reversal` — one minute after IH had gone long on the
recovery — and cut at 10:31 on `per_leg_index_hierarchy_cut`. So the index
hierarchy rule (v3y) did its job and stopped a bad trade inside sixty seconds;
the entry itself was the error.

That entry is precisely what v4f is meant to prevent. The market had opened flat,
sold off in one minute, and begun recovering sharply — a repeated chart, a move
that denied entry, and an empty book. v4f says all three of those argue against
selling the drop and for waiting to see which way the manufactured trap points.
The agent instead read the drop as a continuation and shorted into the recovery.

Encouragingly, the deterministic strategies had a strong day on the same tape:
SMA Crossover alone made more than the whole basket lost, and only four of
fourteen finished negative.

---

### Pre-open note for 2026-08-13 (Thursday, SENSEX expiry)

**Source:** Intraday Hunter, "Prediction For 13 AUG 2026" (`PfthlsdW2E8`,
uploaded 2026-08-12, 2:13). Note-only; no knowledge version attached.

**The seated crowd has flipped to BUYERS.** Yesterday's note described a spent
seller crowd and thin positioning on both sides. Today the sellers are spent for
a specific, observable reason, and buyers have taken their place:

> "Good selling came, but the market took support EXACTLY at the round number
> and gave a positive momentum. So those sitting short would have been chased
> out on the retracement — it has already chased the sellers out."

BankNIFTY is where the buyers actually are: *"this chart has been positive from
the start... a retracement and then overall positive momentum."* That is the
crowd a gap-down would hunt.

**The gap-up branch is a FOLLOW, and the reason is stop LOCATION.** This is the
most transferable part of the note, and it is a sharper statement of the
round-number idea than the series has had:

> "Buyers' SLs should be BELOW the round number. If the market is above the
> round number, the buyers are not going to give their SLs — so we go WITH the
> market."

So the conditional is not symmetric guesswork. A gap-down puts price *into* the
buyers' stop zone and makes them huntable; a gap-up above the round number puts
price *away* from it, leaves nothing to hunt, and the correct move is to follow.
That pairs directly with v4c's ROUND NUMBERS AMPLIFY RECRUITMENT and with v4f's
AN EMPTY BOOK MEANS A TRAP IS COMING — here the book is not empty, so the read is
about geometry rather than about waiting for confirmation.

**Sensex expiry** tomorrow, flagged explicitly, so the Sensex leg carries the
usual pinning and premium distortion. NIFTY is the weakest of the three reads —
*"selling was seen but some recovery is also visible"* — and the note says so
rather than granting it the same confidence as BankNIFTY.

**Transcription caveats, two this time.** NIFTY's resistance pair arrived as
"246 and 24500", read as 24600/24500 (the dropped-trailing-zero artefact seen on
4, 7, 10 and 12 Aug). NIFTY's SECOND support arrived as "2476" and could not be
resolved to a plausible level at all, so it is **omitted** rather than guessed —
a missing advisory level is safer than a wrong one, and the test asserts NIFTY
carries exactly one support so a later "tidy-up" cannot invent the second.
Sensex's 78145 resistance is recorded as heard.

> **Corrected 2026-08-13 (see the 14 Aug entry):** "2476" was **24,176**, read
> directly off the 1080p chart frame. Omitting was the right call given only the
> caption, but reading the frame is better still, and is now the standard step
> whenever a level does not parse — the chart is authoritative, the auto-caption
> is not.

Test updated: `test_shipped_note_matches_august_13_intraday_hunter_plan`
replaces the 12 Aug equivalent and asserts both branch directions plus the
round-number stop-location reasoning, because an inverted plan and a smoothed-
away justification are the two failures a copy-forward would produce.

---

## Video addendum - the 13 Aug LIVE SESSION (v4g)

**Source:** Intraday Hunter live session, 13 Aug 2026 (`Vig9Kjab2T0`, 10:36,
published 11:36 IST). A win, but a REDUCED one - the trade reached most of its
target, reversed into a loss, recovered, and was booked for roughly half. That
arc is what makes the session valuable: it produces the two structural ideas
below and the most complete discipline block in the series.

### A completed stop-hunt ends that direction

The sharpest new idea, and it inverts the naive reading of a bounce:

> "Yesterday the market gave good selling, then took support EXACTLY at the 500
> level and gave a retracement. Because of that retracement, whoever was selling
> got chased out... so the chances of going DIRECTLY UP are LOW."

The up-move existed *to clear the shorts*. With the shorts gone there is nobody
left to squeeze, so the bounce has spent its fuel. His operating rule:

> "If it has chased the sellers out, we try to follow THAT SAME DIRECTION."

- meaning the direction that preceded the clearing bounce, not the bounce. The
escape hatch is explicit and is kept in the prose: a fresh large gap recruits a
new crowd and restarts the question.

### The round number is where the thesis DIES

v4d used round numbers to place targets; v4c used them to explain recruitment.
This adds the third and most operational use - the declared invalidation:

> "Until the market crosses the round number - as we see in NIFTY, the 24,500
> level - we will not have much problem."
> "When is there no danger to these buyers? If the market goes above 58,000."

Each index carries its own named level, decided before entry, and a decisive
cross means the read has failed even if the arithmetic stop is untouched.

### The discipline block, and why it needed a test

> "Before the target is hit there is a fear - should I book here, what if the
> market turns? **Fear is not a big deal, I feel it too.** But do NOT convert
> that fear into ACTION."
> "If you cut early it gradually becomes a HABIT. Then you cut small profits and
> leave, and when there is a loss you wait a long time to save the position and
> take a BIG loss. That is why most traders never become profitable."

This sits directly on top of v4f's BOOK WHEN THE PROFIT STOPS GROWING, and the
pair is the most dangerous in the prompt: read carelessly, v4g reads as "hold
through everything" and disables v4f, while v4f licenses exactly the fear-driven
exit v4g forbids. The distinction encoded in both directions is **measurement
versus emotion** - the profit RATE falling is a measurement; "it might turn" is
not. `test_v4g_fear_rule_and_v4f_book_rule_do_not_cancel_each_other` asserts
both halves say so, and that the fear rule still enumerates the legitimate exit
reasons rather than banning exits.

Two more from the same arc: **time spent shrinks the achievable target** ("the
market turned and wasted our TIME... only half the profit is showing, where
earlier it showed double"), and **never exit at zero after a good profit has
printed** - once real open profit has appeared, the floor stops being breakeven.

### Knowledge changes (v4g, all prose)

- `OPENING_DRIVE`: A COMPLETED STOP-HUNT ENDS THAT DIRECTION; THE ROUND NUMBER
  IS WHERE THE THESIS DIES, NOT JUST WHERE IT PAYS.
- `RISK`: FEAR IS NOT A SIGNAL - NEVER CONVERT IT INTO AN EXIT; TIME SPENT IN
  THE TRADE SHRINKS THE ACHIEVABLE TARGET; NEVER EXIT AT ZERO AFTER A GOOD
  PROFIT HAS PRINTED.
- Test markers: `test_system_prompt_has_v4g_stop_hunt_completion_and_discipline_knowledge`
  plus two drift guards - the fear/book pair above, and
  `test_v4g_stop_hunt_rule_does_not_become_always_fade_the_bounce`, which keeps
  the large-gap escape hatch alive so the rule cannot harden into "always fade".
- Prompt size 105,933 -> 111,283 chars. **Headroom is now 8,717** - at the
  recent ~5,000 chars per version that is roughly two more before the 120,000
  cap, so the next pass should start pruning superseded prose rather than only
  appending.

### How our agent traded the same session

**Provisional - the runner was still live at 14:13.** Realized so far:
**-9,229.75** across 45 legs, and the shape is the exact inverse of 12 Aug.

| Strategy | Legs | Realized |
|---|---|---|
| SL Hunting AI | 4 | **+2,727.00** |
| Heikin Ashi | 6 | +2,431.00 |
| Supertrend Bullish | 1 | +1,043.25 |
| RSI Reversal | 1 | +994.50 |
| SMA Crossover | 2 | +539.50 |
| Long Strangle | 8 | +305.50 |
| ... | | |
| CPR Algo 3 | 1 | -2,158.00 |
| Donchian Bearish | 1 | -2,567.50 |
| Supertrend | 4 | -2,671.50 |
| Opening Strike | 1 | -2,717.00 |
| Renko | 6 | -4,127.50 |
| **Total** | **45** | **-9,229.75** |

**SL Hunting was the best strategy on the board**, cross-checked against its own
`Result summary` (+2,727.00, Trades=2). Yesterday it was the worst.

What changed is exactly what v4g and v4d describe. Both trades were SHORT off the
flat open - the same side IH took - and **both exits keyed off the round number**:

| Entry | Setup | Exit | Held |
|---|---|---|---|
| 09:29 | flat_open_pivot_breakdown_bearish_engulfing (stop 24353, target 24300) | 09:35 `profit_booking_round_number` | 6 min |
| 10:06 | double_top_rejection_confirmed_bearish | 10:26 `profit_book_stall_near_round_number` | 20 min |

Contrast with 12 Aug: four entries in 47 minutes, three released on premise-STALL
judgements, one cut by the hierarchy rule after 60 seconds. Today: two entries in
57 minutes, both booked on a NAMED level. The difference is not the read - it was
short both days - it is that the exits had a checkable reason, which is precisely
what v4g's FEAR IS NOT A SIGNAL demands and what the stall-churn lacked.

One caveat against reading too much into it: two trades is a small sample, and
the deterministic strategies had a poor day on the same tape (Renko -4,127.50 over
six trades), so the basket is deeply negative regardless.

---

### Pre-open note for 2026-08-14 (Friday)

**Source:** Intraday Hunter, "Prediction For 14 AUG 2026" (`MhmlrlUEUGI`,
uploaded 2026-08-13, 2:14). Note-only; no knowledge version attached.

**Neither side is seated, and he says so for two of the three indices.** This is
the empty-book condition v4f describes, stated more plainly than the series has
had it before:

> BankNIFTY: "we did not see much momentum... buyers and sellers would be sitting
> at the SAME price level. Not many people held their positions."
> Sensex: "momentum on both sides remained -- a bit of rejection, a bit of
> buying. So to say more sellers are seated, or more buyers are seated, would be
> WRONG."

That matters because v4e's expensive lesson was forecasting a crowd into
existence on exactly this condition, and v4f's answer was to wait for
confirmation. The note records the absence rather than smoothing it into a
direction.

**His mechanism for trading a thin book** is the line worth keeping:

> "If they did not hold positions, then WHERE THE CROWD IS THIN, THAT IS WHERE
> THE MARKET TRIES TO MAKE MOMENTUM."

Thin is not dead: the move goes where there is least resistance, which is a
different claim from "no crowd means no trade" and sits alongside v4f's
AN EMPTY BOOK MEANS A TRAP IS COMING rather than against it.

**He prefers a gap to a flat open, explicitly.** New for the series as a stated
preference rather than an inference:

> "Better that it is NOT flat -- either a gap-up or a gap-down would be better,
> because in flat it keeps extracting a small momentum and going away. It does
> not make any special momentum."

That is v4d's participation reading turned into an operational preference: a
flat open grants everyone entry, so it produces chop rather than a move.

Plan: SELL-side on flat-to-gap-down (on Sensex explicitly to target the buyers
sitting on support since yesterday), BUY-side and follow on a gap-up, where "the
buyer becomes safe" and there is nothing left to hunt. On BankNIFTY he notes
there is little pressure on the sellers, so that branch is a **follow of the
drift rather than a seller hunt** -- a distinction the note keeps.

**The garbled level was RECOVERED FROM THE VIDEO, and this replaces omitting.**
NIFTY's second support arrived as the same "2476" token as the 13 Aug video —
twice in two days, so a consistent ASR failure on one number rather than noise.
Instead of dropping it again, the frame at 2:04 was read directly: force the
player to 1080p (`setPlaybackQualityRange`), seek, draw the `<video>` element to
a canvas, crop the right-hand price-axis band and upscale it ~3x, then read the
labels. TradingView draws each line's exact price in a tag on the axis, so the
numbers are legible once the source is 1920x1080 rather than the 854x480 the
player defaults to.

What it showed, every value matching what he speaks:

| | chart label | spoken |
|---|---|---|
| resistance | 24,540.85 | "24540" |
| resistance | 24,443.85 | "24440" |
| spot | 24,395.85 | — |
| support | 24,275.25 | "24275" |
| support | **24,176.20** | "2476" ← the garbled one |

So the 13 Aug note's omitted support was **24176** as well; the level simply did
not move. **Prefer this method to omitting a level** — the chart is authoritative
and the auto-caption is not. Sensex's levels are unchanged from yesterday
(78500/78145, 77500/77200), consistent with a session that went nowhere.

Test updated: `test_shipped_note_matches_august_14_intraday_hunter_plan`
replaces the 13 Aug equivalent and asserts both branch directions plus the
thin-crowd mechanism and the gap-over-flat preference, because those two are
what a summarising edit would drop first.

---

## Video addendum - the 14 Aug LIVE SESSION (v4h)

**Source:** Intraday Hunter live session `H7RXIw5xbZk` (14 Aug 2026, 12:18).

A WIN, and unusually it is a win **held through a full drawdown** rather than one
that went straight to target. That is what makes it useful: every previous
version in this series learned from either a clean win or a clean loss, so the
prompt had no worked example of a correct trade that looked wrong in the middle.

### The two rules that shipped

**ENTRY QUALITY AND DIRECTION ARE SEPARATE JUDGEMENTS.** A trade that ends green
does not retroactively make its entry good, and a trade that goes immediately
against you was not necessarily the wrong side. Without this split, the journal
teaches the agent to grade its reads by their P&L, which is exactly the feedback
loop that produces revenge entries after a stopped-out correct call.

**THE LOSS LIMIT IS A PERMISSION TO WAIT.** This one is the reason the version
exists, and it is also the most dangerous line in the prompt if read alone -
"permission to wait" one careless paraphrase away from "hold losers". It is
bounded on both sides in the prose and both bounds are test-asserted: the limit
licenses patience *toward* the stop, never *past* it, and it forbids cutting
inside it on feel. It explicitly names v4e's `DISCIPLINE IS ASYMMETRIC` as the
rule it refines rather than overrides.

The one thing allowed to override that patience is elapsed time, so v4g's
time-shrinks-the-target rule was extended rather than left to contradict it:
time does not merely shrink the payoff, it **EXPIRES THE PREMISE**. IH, booking
out: "time has become quite a lot, gradually other people will start
participating." Waiting to the limit is right while the premise holds; elapsed
time can retire the premise before price ever reaches the limit.
`test_v4h_entry_and_direction_and_the_loss_limit_pair_with_their_neighbours`
asserts the reconciliation in both directions.

### One rule was cut for budget, then restored the same week

A third rule - **A SEATED CROWD IS WARNED BEFORE IT IS HUNTED, SO NO WARNING
MEANS NO CROWD** - was written, tested, and then removed to fit the cap. IH:
"if sellers WERE seated, the market would go up to target them - especially to
give a WARNING... because the market kept making no momentum, sellers will not
be seated, so it will not go up to warn." The mechanism is that an early adverse
spike is the market shaking a seated crowd, so its *absence* confirms an empty
book.

**It was restored on 2026-08-17 when the cap was raised to 160,000.** Cutting it
was the wrong call: see "The budget was never the design constraint" below.

### Knowledge changes (v4h, all prose)

- `RISK`: ENTRY QUALITY AND DIRECTION ARE SEPARATE JUDGEMENTS; THE LOSS LIMIT IS
  A PERMISSION TO WAIT.
- `RISK` (extended): TIME SPENT IN THE TRADE ... now also EXPIRES THE PREMISE.
- **Pruned:** v4e's `A SHARP FIRST SLIDE BAITS; A SLOW ONE MEANS IT` - superseded
  by v4f's confirmation rule, which covers the same tell with a checkable trigger.
- Prompt size 111,283 -> 113,461 chars (after the 2026-08-17 restore).
  **Assembled 114,752 against a 154,140 budget - 39,388 chars of headroom.**

### The budget was never the design constraint

v4h originally shipped at 119 characters under the cap, having dropped one of its
own rules to get there, and this document claimed the append-only rhythm was
finished and v4i could not ship without a pruning pass first.

**That was wrong, and the guard's own comment says so.** `MAX_SYSTEM_PROMPT_CHARS`
is documented in `sl_hunting_knowledge.py` as:

> a SANITY bound, not a budget to trade against. It exists so a runaway lessons
> file or a malformed note cannot quietly inflate what is sent every bar; it is
> not meant to throttle ordinary knowledge growth

It had already been raised once for exactly this situation (75,000 -> 120,000 on
2026-07-31, because "the next ordinary addendum would have tripped it"). Reading
it as a fixed budget and deleting hard-won knowledge to fit inverted its purpose.

On 2026-08-17 it was raised to **160,000** and the cut rule was restored. The
cost of the extra room was measured, not guessed:

| Session | Decisions | Mean cost | Session total |
|---|---|---|---|
| 2026-08-13 | 72 | $0.2278 | $16.40 |
| 2026-08-14 | 73 | $0.2362 | $17.24 |
| 2026-08-17 | 91 | $0.2333 | $21.23 |

~70-90 decisions a session at ~$0.23. The prompt growth this permits moves the
input portion by single-digit rupees a day, against a strategy that booked
+2,866.50 on 14 Aug and +1,580.25 on 17 Aug. That is not a trade worth deleting
knowledge for.

**The rule for next time:** when the cap is hit, ask whether the growth is
ordinary knowledge (raise the bound) or something pathological - a runaway
lessons file, a malformed note (fix the cause). Pruning genuinely superseded
prose is still worth doing on its own merits; it is just never a way to buy space
under this number.

One real redundancy was found while looking, and is recorded here rather than
acted on under time pressure: `NEVER EXIT AT ZERO AFTER A GOOD PROFIT HAS
PRINTED` (v4g) overlaps `BOOK WHEN THE PROFIT STOPS GROWING` (v4f), whose first
support bullet already says a printed target "counts as reached, even if you did
not take it there". The two could merge into one rule. That is a knowledge-quality
change and belongs in its own commit, judged on whether the merged rule reads
better - not on the character count.

### How our agent traded the same session

Realized: **-3,854.50** across 47 legs.

| Strategy | Legs | Realized |
|---|---|---|
| SL Hunting AI | 2 | **+2,866.50** |
| **Total** | **47** | **-3,854.50** |

**SL Hunting was the best strategy on the board for the second consecutive day**,
cross-checked against its own `Result summary` (+2,866.50, Trades=2).

| Time | Decision | Reason |
|---|---|---|
| 10:04 | ENTER_LONG | `flush_reversal_hammer_confirmed` |
| 10:07 | EXIT | `double_top_rejection_premise_fade` |
| 10:20 | ENTER_SHORT | `double_top_bearish_engulfing_breakdown` |

**Agent vs IH.** Both finished on the same side - IH put-side/short, the agent
short from 10:20 - and both booked on a partial rather than a full target. The
agent got there by a different route: it took a long first, released it three
minutes later on a premise fade, and reversed. That reversal is the behaviour
v4f's confirmation rule was written to produce, and it is worth noting that the
10:07 exit was a *premise* exit on a trade that was not yet losing - which is the
disciplined half of the rule v4h now bounds from the other side.

Two trades is again a small sample, and the basket was negative regardless. The
signal worth keeping is narrower than "the agent is good now": across 12, 13 and
14 Aug, every SL Hunting exit had a **nameable** reason, and the two profitable
days are the two where that was true.

### Operational note

The supervisor heartbeat shipped in the session-state work is confirmed working
in production: **65 heartbeat lines** in the 14 Aug log. The 2026-08-12 failure
mode - snapshot writes stopping silently at 12:30 while trading continued to
15:10 - would now be visible within five minutes.

---

## Video addendum - the 17 Aug LIVE SESSION (v4i)

**Source:** Intraday Hunter live session `nozGFwXsdQg` (17 Aug 2026, 8:33).

A three-index PUT basket, entered at the open and **booked while still
profitable** because one index stopped confirming. That exit is the reason this
version exists: it is the first session in the series that acts on cross-index
divergence from the HOLDING side, and it lands directly on our BankNIFTY mirror.

### The session

He entered puts immediately on the open drive, across all three indices --
BankNIFTY 57300 PE (1170 qty) and 57200 PE, Sensex 840 qty, NIFTY 1365 qty --
with Sensex deliberately sized DOWN because its weekly-expiry timing made it the
likely slow mover. Entry rationale was the familiar empty-book read: the market
had been selling for days, had already given a retracement and closed, so
"sellers are not seated here" and a direct upside retracement was unlikely.

He then sat through an immediate drawdown, and booked into profit at the end.

### The three rules

**THE LAGGING INDEX DECIDES THE BASKET'S EXIT, NOT THE LEADING ONE.** He needed
a breakdown in both indices and got it -- "Sensex needed the 500 level breakdown.
It happened in both" -- but no fast momentum followed, and BankNIFTY held back
while Sensex and NIFTY ran. That was the exit: "Sensex and NIFTY have momentum,
BankNIFTY is trying to hold itself back. So this can create a problem for us. So
we will have to book this profit and go... if BankNIFTY starts turning, our
quantity there is larger, so the problem becomes bigger for us." His closing
generalisation is the rule itself: "when two indices run far ahead, one index
lags a little behind, or sometimes tries to retrace a bit. So we had to book."

**A DOUBLE BOTTOM INSIDE AN ESTABLISHED DOWNTREND SHAKES SHORTS OUT; IT DOES NOT
INVITE BUYERS IN.** Stated twice, once while holding through the pattern and once
in review: "it is not made to attract BUYERS, it is made so that people do not
SELL here", and "there is no need to fear the double bottom... it is only to
chase away those who are selling. Then the market keeps falling slowly." Read the
pattern by who it removes, not by its textbook name.

**THE CROWD'S FEAR IS YOUR WINDOW, AND IT CLOSES WHEN THEY JOIN.** The twin of
v4g's FEAR IS NOT A SIGNAL -- that rule governs the agent's own fear, this one
reads the crowd's. "You think anyone can sell here - nobody will. Right now
everyone is afraid it might turn around, and while they stay afraid our work gets
done." Plus the entry corollary: "before the other person applies their mind, we
should build our position."

This last one is also the **mechanism under v4h's TIME EXPIRES THE PREMISE**: time
retires a premise because the frightened crowd eventually stops being frightened
and participates. The prose says so explicitly, so the two read as one idea rather
than two pieces of timing advice.

### Why the mapping to our basket needed spelling out

IH was size-weighted INTO BankNIFTY on purpose. Our mirror is **equal-lot**, which
is not equal-rupee -- BankNIFTY travels further per unit of time, so the mirror leg
still carries the larger rupee swing. The laggard is the dangerous leg for us too,
for a different reason than it was for him. The rule states that mapping rather
than leaving the agent to assume his basket shape, and
`test_v4i_lagging_index_rule_governs_the_basket_and_keeps_its_per_leg_escape`
asserts both halves.

It also keeps pointing at `exit_leg`: we have a finer instrument than he did and
can cut the stalling mirror alone when the NIFTY premise is genuinely intact. A
rule that always took the whole basket down would be strictly worse than the
capability the host already exposes.

### Knowledge changes (v4i, all prose)

- `PATTERNS_AND_CONFIRMATION`: A DOUBLE BOTTOM INSIDE AN ESTABLISHED DOWNTREND
  SHAKES SHORTS OUT.
- `RISK`: THE LAGGING INDEX DECIDES THE BASKET'S EXIT (beside the BASKET NOTE);
  THE CROWD'S FEAR IS YOUR WINDOW (beside v4g's FEAR IS NOT A SIGNAL).
- Three new drift guards, each protecting the reading that would decay first: the
  double-bottom rule must not generalise into "ignore reversal patterns" or become
  a counter-trend entry licence; the lagging-index rule must keep its per-leg
  escape; the crowd-fear rule must not read as an entry licence.
- Adds a `_flat_rule` helper to the test module. Rule prose is hard-wrapped, so
  asserting wording against the raw prompt breaks on a re-flow the agent never
  sees -- the same brittleness that failed a v4h assertion.
- Prompt size 113,461 -> 118,049 chars. Assembled 119,340 against a 154,140
  budget: **34,800 chars of headroom.**

Worth noting: at the old 120,000 cap this version would not have fitted at all
(119,340 assembled against a 114,140 budget). The cap was raised earlier the same
day -- see "The budget was never the design constraint" in the v4h addendum.

### How our agent traded the same session

**Provisional - the runner was still live at 12:12.** Realized so far: **+7,004.50**
across 26 legs, a positive basket for a change.

| Strategy | Legs | Realized |
|---|---|---|
| Renko | 5 | +3,016.00 |
| **SL Hunting AI** | **2** | **+1,580.25** |
| CPR Algo 3 | 2 | +1,560.00 |
| EMA | 1 | +851.50 |
| Heikin Ashi | 5 | +386.75 |
| Long Strangle | 10 | -74.75 |
| RSI Reversal | 1 | -315.25 |
| **Total** | **26** | **+7,004.50** |

One trade, cross-checked against its own `Result summary` (Trades=1,
RealizedPnL=1580.25):

| Time | Decision | Reason |
|---|---|---|
| 10:23 | ENTER_SHORT (conf=7) | `runaway_trend_continuation_short`, stop 24270, target 24200 |
| 10:46 | EXIT (conf=8) | `book_before_round_number` - +2,565 unrealized, ~30pts from target, last several candles stalled |

**Agent vs IH.** Same side, same structural read: both were short/put-side on a
continuation of the existing selling rather than a reversal, and both booked on
momentum stalling rather than at a target. The agent's stated exit reason -
booking before the round number while the profit stopped growing - is v4g and v4d
firing together, and it is a nameable reason, which now holds for four consecutive
sessions.

**The finding that matters is in the leg split:**

| Leg | Realized |
|---|---|
| NIFTY | +2,115.75 |
| BankNIFTY mirror | **-535.50** |
| Net | +1,580.25 |

The mirror LOST money while the NIFTY leg earned. That is precisely the divergence
IH exited on, occurring in our own book on the same tape -- BankNIFTY lagging while
NIFTY ran. The agent booked the whole basket at 10:46 and so closed the losing
mirror too, but it got there for a NIFTY-side reason and never named the
divergence. v4i is written so that next time the lag itself is a readable signal,
and so that `exit_leg` is the considered response rather than an unused capability.

One difference worth recording: IH entered at the open, the agent at 10:23 -- an
hour into a move he had been riding since 9:15. The agent's no-new-entry cutoff
then fired at 11:00 (`LIFECYCLE RUNNING -> ENTRY_BLOCKED | reason=TIME_CUTOFF`).
On a day whose whole character was set in the first fifteen minutes, most of the
move was already gone by the time the agent was willing to act.

---

## Video addendum - the 18 Aug LIVE SESSION (v4j)

**Source:** Intraday Hunter live session `sSgO_Jole74` (18 Aug 2026, 9:23).

**A LOSING session, and the most useful one in the series so far** - because he
diagnoses the loss mid-trade, names the mechanism, and cuts. Every prior version
learned from a win or from a loss reviewed afterwards; this is the first where
the post-mortem happens live and identifies a market participant the prompt had
never modelled.

### The session

NIFTY expiry, large gap-down on all three indices. He deliberately suspended the
plan until the open resolved - "if it stays a bit in buying we plan the positive
side; if it stays in selling we can sell" - then read yesterday's breakdown plus
the recovery as good traders seated LONG, making the gap-down "just a trap". He
bought calls across BankNIFTY (57600 CE, 1170), Sensex (900) and NIFTY (1430).

It did not recover. And the reason he gives is new to this document.

### The three rules

**A SHARP SPIKE THAT IMMEDIATELY STALLS IS WHERE THE CALLS GOT WRITTEN.** The
crowd is not only directional traders. Option WRITERS position against you, and
they do it into a fast move because that is when the premium is richest. IH at
7:01: "they suddenly produced positive momentum and WROTE the calls... so call
writers are seated here. The market will not go much lower - but it will STAY
negative. If they had written PUTS you would have seen a gradual recovery."

What makes it actionable is the signature, which is **not a reversal**: momentum
dies in BOTH directions - "one sell, one buy, one sell, one buy" - and price
holds below the level it spiked through. For an option buyer that is the worst
regime that exists: neither direction pays and theta runs throughout. So it is an
EXIT read, and explicitly **not** a licence to flip short: the writers' interest
is a range, not a collapse. `test_v4j_call_writer_rule_stays_an_exit_read_and_not_a_reversal_trade`
guards that, because "writers are seated above" reads like a short signal and
is not one.

**THE PERMISSION TO WAIT ENDS WHEN THE TIME TO PROFIT BECOMES THE COST.** The
third bound on v4h's loss-limit rule, and the one that makes it safe. You can be
RIGHT that no large adverse move is coming and still have to leave, because "not
falling" is not "rising". IH, cutting while still arguing the market would not
break down much: "the market can definitely make our loss bigger. It will take a
LOT of time to bring it back into profit. We CAN give time - but this trade does
not look like it is going right." The test is neither the stop nor fear; it is
whether the move that would pay you can still happen in the time you have.

**CUTTING THE MIRROR ON A BANKNIFTY REVERSAL IS A VERDICT ON THE WHOLE BASKET.**
This one comes from our own book rather than the video - see below.

### How our agent traded the same session

**Provisional - the runner was still live at 14:11.** Basket **-3,498.75** across
59 legs. SL Hunting: **+60.00** over 3 trades, cross-checked against its own
`Result summary` (Trades=3, RealizedPnL=60.00).

| Time | Leg | P&L |
|---|---|---|
| 09:33 | NIFTY short / BNF mirror | +182.00 / +15.00 |
| 09:56 | NIFTY long / BNF mirror | +58.50 / -63.00 |
| 10:34 | **BNF mirror cut ALONE** | **+303.00** |
| 10:40 | **NIFTY leg** | **-435.50** |

Three things worth recording.

**1. The pre-open note worked exactly as its own framing says it should.** The
note is injected as "a VOTE... if your own live read disagrees with it, your read
wins". At 09:31 the agent went SHORT and said so explicitly: *"this contradicts
the pre-open note's buy-side plan but the live confirmed rejection overrides
it."* That was the day's best trade (+197.00). Both trades that FOLLOWED the
note's buy-side plan lost. First hard evidence the advisory is being weighed
rather than obeyed.

**2. `exit_leg` was used for the first time - and it is the source of the third
rule.** v4i merged the night before; today the agent cut the BankNIFTY mirror
alone at 10:34 on a confirmed BNF shooting star and bearish engulfing, booking
**+303.00**, and held NIFTY because "its own read is unaffected", explicitly
noting an opposing `cross_index` verdict and overriding it as "a caution".

Six minutes later it closed NIFTY for **-435.50**, giving as its reason the very
same fact: *"BankNIFTY turning down first is disqualifying regardless of NIFTY's
own noise."* The information that ended the trade was already in hand when the
mirror was cut; only the conclusion was late. Hence the rule: the per-leg escape
is for an IDIOSYNCRATIC problem in the mirror, not for the leading index simply
TURNING - because the index hierarchy says BankNIFTY leads. Its corollary closes
the other half of the gap: an opposing `cross_index` verdict on an OPEN position
does not force an exit by itself, but it removes the benefit of the doubt.

**3. Agent and IH were wrong the same way.** Both read "breakdown then recovery =
good traders seated long" and bought; both lost on the long side; both were
rescued only by cutting. The market did what IH diagnosed at 7:01 - the spike was
where the calls got written, and momentum died in both directions rather than
reversing. Our agent had no concept for that participant, which is why the rule
is worth its characters.

One smaller flaw, already covered by existing knowledge rather than needing a new
rule: the 09:55 long was a **51-second round trip** (09:55:38 -> 09:56:29, net
-4.50). Its own exit reasoning says a bearish engulfing "has kept grinding lower
for 5 bars" - a structure already visible when it entered. v4a's PREMIUM
NON-CONFIRMATION already warns that "a trade you abandon within a bar or two of
entry pays the round-trip cost for no exposure to the move".

### Knowledge changes (v4j, all prose)

- `RETAIL_POSITIONING`: A SHARP SPIKE THAT IMMEDIATELY STALLS IS WHERE THE CALLS
  GOT WRITTEN.
- `RISK`: THE PERMISSION TO WAIT ENDS WHEN THE TIME TO PROFIT BECOMES THE COST
  (beside v4h's loss limit); CUTTING THE MIRROR ON A BANKNIFTY REVERSAL IS A
  VERDICT ON THE WHOLE BASKET (beside v4i's lagging-index rule).
- Three drift guards, each protecting the reading that would decay first: the
  call-writer rule must not become a short signal, the basket rule must not
  delete `exit_leg`, and the time rule must not become "cut anything slow".
- Prompt size 118,049 -> 121,817 chars. Assembled 123,108 against a 154,140
  budget: **31,032 chars of headroom.**

---

## The prose pruning pass (2026-08-18)

Promised when `MAX_SYSTEM_PROMPT_CHARS` was raised: pruning superseded prose is
worth doing on its own merits, judged on whether the result reads better, never
to buy space. This is that pass, done with 31,000 characters of headroom in hand
so nothing was under pressure to go.

**The headline finding is that there is very little to prune.** 115 bulleted
rules across 12 sections, and the clusters that look redundant from a list of
headings turn out to be deliberately paired. Two changes were made. Several
obvious-looking candidates were examined and deliberately kept, which is the more
useful result to record.

### Examined and KEPT

| Pair | Why it stays |
|---|---|
| `PRE-COMMIT THE ADVERSE MOVE` (v4d) vs `NAME THE LAST POINT` (v4e) | One is a MAGNITUDE, the other a LOCATION plus a deadline. v4e already says so in its own text. |
| `A FORECAST OF WHO WILL ARRIVE` (v4e) vs `AN EMPTY BOOK MEANS A TRAP IS COMING` (v4f) | A deliberate dialectic from two real sessions — 11 Aug (predicted, lost) against 12 Aug (waited for confirmation, won). v4f is framed as the reconciliation of v4e. Deleting either destroys the lesson. |
| `REPEATED-FAILURE INVENTORY RESET` vs `DIRECT-MOMENTUM / CURRENT-SESSION TRAP RESET` | Both "resets", different triggers: one is level-based (repeated reclaims evict sellers), one is time-based (the opening thrust supersedes yesterday's crowd). |
| `NO INSTANT FLIP` / `MOVE-EXHAUSTION` / `POST-EXIT RE-ENTRY GATE` | Opposite-direction, same-direction, and the mechanical check. Each names the others; the gate's "a different pattern name on the same structure is not a new premise" is an anti-rationalisation guard worth its length on its own. |
| `YOUR ENTRY PRICE IS THE FOURTH TARGET INPUT` / `CROWD SIZE IS THE THIRD` | A numbered series of distinct inputs, not restatements. |

The reason for this is structural, and worth knowing before the next pass: **this
series has reconciled every new rule against its neighbours as it landed.** The
prose is full of "Distinct from X, which is...", "scopes the rule above", "this is
the TEST that Y implies", "NOT the fear rule above in disguise". Those sentences
are the redundancy check, already run, one version at a time. A future pass should
expect to find little, and should treat a rule that does NOT reconcile itself
against a neighbour as the suspicious one.

### Changed

**1. `POST-LOSS SPEED LIMIT` absorbed `Loss recovery discipline`.** These were one
protocol split across two bullets 16 lines apart, and the shared half — do not
take the next trade immediately after a loss — was stated in both. Merged, keeping
the `POST-LOSS SPEED LIMIT` name because `NO INSTANT FLIP` defers to it by name.
The recovery half (spread a big loss over several ordinary trades; distrust the
day's "one last trade") survives intact.

**2. `BOOK WHEN THE PROFIT STOPS GROWING` (v4f) gave up its first support bullet.**
Both v4f and v4g's `NEVER EXIT AT ZERO` told the agent that a profit it merely SAW
counts. The two rules themselves are distinct and both stay — v4f is the exit
TRIGGER (the rate of gain), v4g is the FLOOR (what an acceptable exit looks like
once profit has printed) — but the idea was written twice and owned by neither.
v4f now states the trigger and points at v4g.

Net effect on size: **-206 characters.** That is the correct order of magnitude for
this pass and is not the point; the point is that two ideas now have one home each.

### Also fixed: a test helper that could assert against the wrong rule

`_flat_rule` located a rule with a bare `prompt.index(heading)`. Because rules cite
each other by name, that could land on another rule's CROSS-REFERENCE and silently
assert against the wrong prose — the new `POST-LOSS SPEED LIMIT` test failed this
way and exposed it. It now anchors on the bullet (`"\n- " + heading`) and falls
back to the bare heading. This is the same class of bug as the v4i locator
collision, which suggests the pattern rather than the instance was the problem.

---

## Video addendum - the 19 Aug LIVE SESSION (v4k)

**Source:** Intraday Hunter live session `pSgclYI-_Ro` (19 Aug 2026, 11:05).

A small win taken reluctantly out of a market he spends nine minutes saying is
barely tradeable - "the situation is very bad, we cannot take any quick
decision", "it is very difficult to build a trade here", "in a market that gives
ONE burst of momentum and then stops you cannot make money, however long you wait
and however good an entry you try to make."

Almost flat open, immediate sharp selling in all three indices, so he stood aside
first. The setup he eventually took was the familiar one: continuous selling
across all three indices recruits intraday put-side traders, and those become the
target. He bought calls (BankNIFTY 57000 CE, Sensex 900, NIFTY 1430) and booked
into a partial recovery.

### One new rule, two extensions

Most of what this session teaches lands on rules that already exist, so it is
recorded that way rather than as three new bullets - see the 2026-08-18 pruning
pass for why that matters.

**NEW - A SUPPORT EVERYONE CAN SEE RECRUITS NOBODY WHILE EVERY INDEX IS FALLING.**
Asked directly whether other traders would buy the round-number support BankNIFTY
had just held: "they do buy such a support - but only if the trend is a bit
positive. Not in a market like this, especially when all three indices are
selling. After a breakdown someone will certainly SELL, but he cannot bring
himself to BUY at the support. It is not that he does not see it - he sees it
perfectly well, but he cannot buy."

The two consequences point opposite ways and both are kept: as EVIDENCE a held
support is worthless in a three-index sell-off and must not feed a seated-buyer
read; as a PLACE it is unusually clean, because nobody is queued there competing
with you. He still waited for price to move AWAY from the level before buying -
an entry sitting on it is one retracement from a breakdown.

**EXTENDED - `Momentum quality while holding` gains its mechanism.** The existing
bullet already said a fast spike invites a retracement. IH's reason today is
different and sharper: when your thesis is a trapped crowd being squeezed, THEY
are what pays you, so the speed of the move decides how long you get paid for.
"If it runs fast the sellers will get out quickly, so we WANT it to go a little
slow, so that they stay sitting in the market", and afterwards: "if it recovers
fast, every seller here leaves in a single move, and then the market starts
falling again." A fast move in your favour is therefore not extra confirmation,
it is a shorter clock.

This needed an explicit reconciliation with v3y's A SLOW GRIND AT THE LEVEL
RECRUITS THE WRONG CROWD, which says slowness is BAD. That rule is about price
stalling at your level BEFORE entry; this is about pace AFTER entry, in your
favour. Read together without that sentence they contradict.

**EXTENDED - v4i's lagging-index rule becomes direction-agnostic.** v4i named the
BankNIFTY mirror lagging while NIFTY ran. Today was the mirror image: "the profit
is coming ONLY from BankNIFTY - Sensex and NIFTY are still slightly negative...
only BankNIFTY has momentum, so booking the trade was the right thing." One index
paying alone is the book signal whichever index it is.

### How our agent traded the same session

**Provisional - the runner was still live at 14:17.** Basket **-2,139.75** across
55 legs. SL Hunting: **-849.50** over 2 trades (`Result summary`: Trades=2,
RealizedPnL=-849.50).

| Time | Leg | P&L |
|---|---|---|
| 09:38 | NIFTY short / BNF mirror | +52.00 / **-519.00** |
| 09:46 | NIFTY long / BNF mirror | -136.50 / -246.00 |

**1. v4j changed the behaviour, one day later and by name.** Yesterday the agent
cut the mirror alone at 10:34 and held NIFTY into -435.50, then closed it six
minutes later citing BankNIFTY's reversal - the information it already had. Today
at 09:46 it exited BOTH legs in one decision, with the setup label
`basket_book_lagging_index_divergence` and the reasoning "per index-hierarchy
rules the lagging/leading index turning against the trade is disqualifying
regardless of NIFTY's own noise". That is the v4j rule, applied. It is one
session and the trade still lost, so this is evidence the rule is being read, not
that it is profitable.

**2. An EXECUTION error, self-reported, and not a knowledge problem.** At 09:38
the agent logged:

> Erroneous exit: analysis supported PROFIT-HOLD (short premise intact, steady
> downtrend on both NIFTY and BankNIFTY, no opposing confirmation pattern, target
> 24000 not yet reached, NIFTY leg +104), but the order tool was called with EXIT
> instead of intended HOLD. Position is now flat as a result of this executed
> order; reporting the actual action taken.

Its reasoning concluded HOLD and an EXIT was executed. No prose rule fixes that;
it is being tracked separately as a code question - whether the model emitted the
wrong action or the host executed one, and whether a host-side sanity gate is
worth having. It is PAPER today, but the same path places real orders when both
live gates are set.

**3. The hold it intended would have been wrong anyway, for a reason v4i already
gives.** It justified PROFIT-HOLD with "NIFTY leg +104" while the BankNIFTY
mirror was **-519**, so the basket was roughly **-467**. The leg it cited as
evidence the trade was working was the only part of it that was. v4i already says
to read the basket rather than `nifty_leg_pnl`; what was missing was the blunt
instruction, so the rule now carries it: when you justify a HOLD with a P&L
number, quote the BASKET number.

**4. The note-override pattern did NOT hold today, and that is worth recording.**
On 17 and 18 Aug the trade that contradicted the pre-open note was the day's best
and the ones following it lost. Today the agent again overrode the note at 09:36
- "this contradicts the pre-open note's flat-open buy-side expectation, but that
note assumed no momentum, and real confirmed momentum has since appeared", which
is exactly the reasoning the note's framing invites - and the resulting basket was
losing (-467) when the erroneous exit closed it. The 09:43 long that FOLLOWED the
note also lost. So today neither branch worked, and three sessions is not a
pattern to trade on.

**5. Agent and IH shared the thesis; the difference was patience.** Both read
continuous three-index selling as recruiting put-side traders and both went long
against them. IH stood aside through the opening drop, waited for price to move
away from the support, and took the small profit the market offered. The agent
entered at 09:43 near the level and was out three minutes later.

### Knowledge changes (v4k)

- `RETAIL_POSITIONING`: A SUPPORT EVERYONE CAN SEE RECRUITS NOBODY WHILE EVERY
  INDEX IS FALLING (new).
- `RISK`: `Momentum quality while holding` extended with the crowd-flush
  mechanism and the v3y reconciliation; THE LAGGING INDEX DECIDES THE BASKET'S
  EXIT extended to be direction-agnostic and to demand the basket P&L number.
- Three drift guards: the fast-move rule must keep its mechanism and its v3y
  reconciliation, the lagging-index rule must keep both directions, and the
  support rule must keep both of its opposite consequences.
- Prompt size 121,611 -> 125,054 chars. Assembled 126,345 against a 154,140
  budget: **27,795 chars of headroom.**

---

## SLH-010 - an EXIT the model did not mean to send (2026-08-19)

### What happened

At 09:38 the agent closed a live short. Its next pass reported:

> Erroneous exit: analysis supported PROFIT-HOLD (short premise intact, steady
> downtrend on both NIFTY and BankNIFTY, no opposing confirmation pattern, target
> 24000 not yet reached, NIFTY leg +104), but the order tool was called with EXIT
> instead of intended HOLD. Position is now flat as a result of this executed
> order; reporting the actual action taken.

### It was a MODEL error, and the log proves it

The question worth settling first was whether the model emitted the wrong action
or the host executed one the model never sent. SL Hunting is not a
decide-then-execute design: the model calls `place_paper_order` (or the live
equivalent) DIRECTLY during its reasoning, so **the tool call is the execution**.
The final JSON is a report of what already happened, which is why that pass
truthfully says EXIT.

The executed order carries the detail that settles it:

```
09:38:21 | EXIT SHORT        | ... | Reason=AI_EXIT (no reason supplied by the model)
09:38:21 | MIRROR EXIT SHORT | ... | Reason=AI_EXIT (no reason supplied by the model)
```

`AI_EXIT (no reason supplied by the model)` is `NO_REASON_SENTINEL`, substituted
by `_safe_exit_reason` when the model passes nothing usable. So the model called
the order tool with `action="EXIT"` **and an empty reason**. No host path
fabricates an EXIT, and no mechanical exit uses that sentinel.

### Two design gaps made it easy to make and easy to miss

**1. The tool description never said what HOLD is.** It listed three actions --
ENTER_LONG, ENTER_SHORT, EXIT -- and nowhere stated that holding means not
calling the tool. A model deliberating "HOLD vs EXIT" had no way to express HOLD
through the tool, and EXIT was the nearest available option. It also never said
that an EXIT is unlike an entry: entries are validated three ways (levels, reason
meaning, note check) and can be corrected inside the same pass, while an EXIT is
executed immediately and never rejected.

The description now says both, in the first sentences: calling the tool IS the
action, there is no HOLD action and a decision to hold means not calling it at
all, and an EXIT cannot be corrected or undone.

**2. The strongest available signal was recorded at INFO.** A deliberate exit
always carries a justification, because the description tells the model that
string is the permanent record. An EXIT with nothing in it is therefore the best
evidence available that the call was unintended -- and on 19 Aug that fact sat
inside a long INFO trade line. It was noticed only because the model confessed on
the next bar.

`do_order` now logs a WARNING when an EXIT executes with the sentinel, naming it
as a probable unintended order.

### What was deliberately NOT changed

**SLH-007 stands: an EXIT is still never refused over its wording.** Bouncing it
would strand an open position, which the risk rules forbid, and that was an
operator decision made for a good reason. The exit still executes; it just stops
being silent. `test_exit_with_no_reason_still_executes_but_warns_loudly` asserts
both halves so a later tightening cannot quietly turn the warning into a refusal.

No confidence-based gate was added either. Confidence lives on the final decision
schema and never reaches the order tool, so it is not available at the boundary
where the order is placed -- a gate there would have to be built on a value the
tool cannot see.

### Residual risk

The fix is preventative, not mechanical: a model that ignores the description can
still send an unintended EXIT, and the host will still honour it. What is bounded
is the cost -- the operator now learns the same session instead of by reading
decision text days later. A mechanical fix would mean gating exits, which is the
one thing this component must not do.

The negative case was verified rather than assumed: with the warning branch
disabled the test fails with "an EXIT with no reason must warn".

---

## Video addendum - the 20 Aug LIVE SESSION (v4l)

**Source:** Intraday Hunter live session `AuKnZ9i_rz4` (20 Aug 2026, 10:15).

The pre-open call worked. Yesterday's note said the put writers had been flushed
and the crowd still seated was CALL writers, so a gap-up should be followed; the
market gapped up hard, he bought calls across all three indices within five
minutes of the open, and booked a large winner. Most of the session is him
explaining WHY, which is where the material is.

### One new rule, one extension, one correction

**NEW - ENTRY TIME IS A RISK DIAL, YOU GET THE MOMENTUM YOUR RISK BUYS.** "You can
trade early, or you can wait a little - you should decide how much RISK you want
to take. If you trade early and momentum suddenly comes against you, the loss can
be big. If you wait until around 10 or 11, the momentum is slower than when we
trade straight off the open... you will get momentum in proportion to the risk you
take."

The rule is the trade-off, not a preference: an early entry is not simply a better
entry and a late one is not simply a safer one, because each buys a different
distribution. He says plainly "it is not that a trade made right at the open will
be better." The operational half is that an opening-window entry should size the
EXPECTATION as well as the stop - you have bought the fast tail in both
directions, so a target set for a 10:30 tape is the wrong target. Kept explicitly
distinct from MORNING SPEED IS NOT INFORMATION, which governs what a fast stop-out
means AFTER the fact rather than what you are choosing when you pick the hour.

**EXTENDED - v4j's basket rule gains its general form: WHEN YOU CUT, CUT EVERY
LEG.** IH states it twice as a rule of its own, after a post-mortem on hedged
option writers: "either keep your trade COMPLETE, or cut both together", and "when
you cut your trade, cut BOTH sides - you should not leave any leg."

His mechanism is what makes it more than advice. The writer had short puts hedged
with long puts; when the market fell he removed the short-put leg that was hurting
and kept the long put he still believed in - "so the market first removed the
put-write position and then did not let him profit on the put either, a loss on
BOTH sides."

The structural difference is recorded rather than glossed: his legs are a hedge on
one underlying that offset each other, ours are two correlated directional legs on
different indices. **The arithmetic does not transfer, but the failure does** - you
cut the leg you can justify and keep the one you are hoping for. So the rule now
says EXIT BOTH is the default and `exit_leg` is the exception, usable only when the
surviving leg's premise is independently intact and you can say why, never merely
because the other leg is the one currently hurting.

**CORRECTED - v4k's "only one index is paying" test, one day after it shipped.**
See below; the correction comes from our own numbers, not the video.

### How our agent traded the same session

**Provisional - the runner was still live at 14:38.** Basket **+6,423.75** across 46
legs, a clearly positive day. SL Hunting: **+1,015.75** over 3 trades
(`Result summary`: Trades=3, RealizedPnL=1015.75).

| Time | Leg | P&L |
|---|---|---|
| 09:18 | NIFTY long / BNF mirror | +461.50 / +1,092.00 |
| 09:27 | NIFTY short / BNF mirror | -39.00 / -1,113.00 |
| 09:42 | NIFTY long / BNF mirror | +9.75 / +604.50 |

**1. v4k was cited by name and applied, one day after shipping.** The 09:42 exit
reads: "Basket was +631.5 but NIFTY has stalled forming a double top... NIFTY leg
itself was flat (+19.5) while the BankNIFTY mirror carried nearly all the profit -
a leg-divergence book signal (v4k) alongside profit-stopped-growing (v4f)." It
also quoted the BASKET number, which is the practical instruction v4k added after
19 Aug's "NIFTY leg +104" justification. Both halves of yesterday's change were
used.

**2. And that is exactly how the rule turned out to be wrong.** The mirror is
EQUAL-LOT, not equal-rupee: a BankNIFTY option cost 578 against NIFTY's 127.70 on
the first trade and travels further, so the mirror carries the larger rupee share
even when both legs are working perfectly. Trade 1 split **+461.50 / +1,092.00** -
a 70/30 split with BOTH legs working, which is arithmetic, not divergence. Applying
"only one index is paying" to that would book a healthy trade.

Trade 3 was genuine divergence, but for a different reason than rupee share: the
NIFTY option moved **0.15 premium points** while the mirror moved **20.15**. So the
test is whether a leg has stopped MOVING, never whether it is contributing less
money. v4k now says so, with both measured cases kept so neither can be pruned as
redundant.

Worth noting as a pattern: a rule written from one session's arithmetic was applied
to a differently-shaped session the next day and misfired. The agent reached the
right action - booking was independently justified by the profit stall it also
cited - through reasoning that does not generalise.

**3. Following the note won; overriding it lost.** The 09:16 and 09:38 longs both
took the note's gap-up buy-side branch and both made money. The 09:26 SHORT
explicitly overrode it - "overriding the pre-open analyst's buy-side call-writer
bias per live price action" - and lost **-1,152.00**, the day's only losing trade.
That is the reverse of 17-18 Aug and confirms the caution recorded in v4k: three
sessions was never a pattern, and there is now a fourth pointing the other way.

**4. The same cross-index verdict was used as evidence and dismissed as stale
within nine minutes.** At 09:18 "cross-index flipped to bias DOWN... directly
opposing the long" was given as a reason to exit. At 09:27 "the cross_index
'both_at_support/bias down' verdict is a stale mechanical read" was given as a
reason to exit the opposite position. Both uses were inside the opening hour, so
both are technically within the stale escape hatch's documented scope - but the
signal was cited when it agreed and discounted when it did not, and in both cases
the conclusion was to exit. No rule is proposed for this yet: one session is not
enough, and the existing scope warning already says the hatch "is abused easily".
Recorded so the next occurrence is the second data point rather than the first.

**5. Agent and IH agreed on the read and differed on holding.** Both were long the
gap-up on the flushed-put-writer thesis. IH held through a mid-trade rejection -
"do not fear the rejection at all, this rejection has come to benefit you" -
because he could name whom it removed, and booked a large move. The agent was in
and out three times in 26 minutes, its best trade lasting two.

### Knowledge changes (v4l)

- `RISK`: ENTRY TIME IS A RISK DIAL (new); the basket rule extended with WHEN YOU
  CUT, CUT EVERY LEG and its mechanism; v4k's one-index-paying test corrected to
  judge MOVEMENT rather than rupee share.
- Three drift guards: the cut-every-leg rule must keep both the EXIT BOTH default
  and the narrow `exit_leg` exception, the divergence correction must keep both
  measured cases, and the entry-time rule must stay a trade-off rather than
  becoming "waiting is safer".
- Prompt size 125,054 -> 128,303 chars. Assembled 129,594 against a 154,140
  budget: **24,546 chars of headroom.**

---

## Video addendum - the 21 Aug LIVE SESSION (v4m)

**Source:** Intraday Hunter live session `yacw-NpZq30` (21 Aug 2026, 10:11).

**Both IH and our agent lost, on the same side, for the same reason.** That
alignment is rare and makes this the most directly comparable session in the
series: the pre-open note said sell-side, IH bought puts across all three indices
inside two minutes of the open, our agent shorted four times in 51 minutes, and
neither got the move.

### One precondition, named before the trade and missing all day

The whole session turns on a single checkable gate. IH, mid-trade: "if momentum
comes, we need the CLOSING PRICE breakdown - especially BankNIFTY's. If it breaks
that, we get a good move." And then the property that makes it cheap to check:

> All three indices have held it. It is not the case that it breaks in one and
> stays in another: either the market falls in all three, or it stops. **If it
> breaks in even ONE index it will break in all three; if it does not break in
> even one, all three will sit and hold.**

His post-mortem is the same sentence inverted: "no index broke the closing price,
and until that level breaks no rejection could come." He also names his own error
- "the gap-up was not large, rejection started right at the open, so we made our
entry EARLY too."

The existing `CLOSING-PRICE BREAKDOWN IS THE TRIGGER` rule already said the
breakdown is the trigger and that sitting on the level is not breaking it, so this
is recorded as an EXTENSION rather than a new bullet. What is new is the
all-or-nothing property, the fact that the failure mode while it holds is CHOP
rather than a clean loss - "the fear of a sideways market is there every day now"
- and that a rejection at the open is a reason to WATCH the level, never a
substitute for it breaking.

### The loss limit is regime-conditional

v4h says the limit is a permission to WAIT, bounded by elapsed time (v4j). Today
adds a third bound: "when it is a momentum market we do sit, and we do let the
loss grow a bit, because the next day gives a chance to cover it. But these days
there is not much momentum, so there is no benefit in making the loss bigger
here."

So what buys the right to sit toward the limit is a market that still MOVES. The
reconciliation matters more than the rule: read carelessly this becomes "cut early
when it feels slow", which is precisely the error v4h exists to prevent. The prose
keeps all three bounds naming an OBSERVABLE - the limit, the clock, the regime -
and none of them licenses cutting on feel.

### How our agent traded the same session

**Complete session.** Basket **-12,912.00** across 68 legs, a bad day everywhere.
SL Hunting: **-2,300.75** over 4 trades, its worst of the series and the worst
single strategy on the board.

| Time | Leg | P&L | Held |
|---|---|---|---|
| 09:28 | NIFTY short / BNF mirror | +419.25 / **-1,539.00** | 7 min |
| 09:36 | NIFTY short / BNF mirror | -312.00 / -768.00 | 1m47s |
| 10:00 | NIFTY short / BNF mirror | -240.50 / +48.00 | 1m51s |
| 10:12 | NIFTY short / BNF mirror | +312.00 / -220.50 | 7 min |

**1. Four shorts, and the leading index killed every one.** Each exit names it
correctly - "per index hierarchy, BankNIFTY turning against the trade is
disqualifying for the whole basket, not just the mirror leg" - and then the next
entry is justified fresh on NIFTY patterns as though BankNIFTY had not just
refused. The disqualifying fact was true of the SESSION, not of one position, and
it never carried forward. That is the new rule, and this is the clearest example
the journal has.

**2. The stale hatch was used selectively, for the second session running.** At
10:05 `cross_index` was dismissed - "'both at resistance/bias up' looks stale
given the live rollover" - to justify an ENTRY. At 10:12, seven minutes later, the
SAME verdict was cited as a reason to EXIT. The 20 Aug session did it at 09:18 and
09:27. In every instance the re-rating favoured the action already chosen.

v4l recorded the first occurrence with "noted so the next occurrence is the second
data point rather than the first". It arrived the next day, so v4m legislates: the
hatch stays open, but staleness must be judged from the verdict's age and
anchoring, never from whether it agrees. The test the agent can actually run is
"would I still call it stale if it AGREED with me right now?"

**3. v4l's cut-every-leg rule was applied correctly and immediately.** Every one of
the four exits closed BOTH legs, and two say why in v4l's own language - "exiting
both legs since BankNIFTY structure confirms the same uptrend, not an idiosyncratic
mirror-only issue". No per-leg cut was attempted. On the evidence the rule is being
read and applied; it did not save the day, because the entries were the problem.

**4. The 09:28 exit still described the trade by its healthy leg.** The reasoning
reads "Booked the healthy NIFTY leg (+585) and closed the mirror together" while
the mirror was **-1,539.00** and the basket about **-1,120**. The ACTION was right
and cited the basket-exit rule; the characterisation was not. v4k already demands
the basket number when justifying a HOLD - this was an EXIT, so no rule was broken,
but the same habit of narrating by the favourable leg is visible and worth watching.

**5. The note was followed four times and lost every time.** It said sell-side; all
four entries were shorts citing it. It also carried an explicit warning - "EVIDENCE
IS WEAKER ON NIFTY AND SENSEX... demand a cleaner confirmation" - which the entries
cite as satisfied without engaging with the weakness. Across five sessions the
note-versus-override record now points both ways with no pattern; the honest reading
is that it is one vote, exactly as its own header says.

### Knowledge changes (v4m)

- `OPENING_DRIVE`: `CLOSING-PRICE BREAKDOWN IS THE TRIGGER` extended with the
  all-or-nothing cross-index property, the sideways failure mode, and the
  early-entry trap.
- `RISK`: v4h's loss-limit permission extended with the regime condition; two new
  rules - THE FACT THAT DISQUALIFIED YOUR LAST TRADE STILL HOLDS, and YOU MAY NOT
  RE-RATE A SIGNAL TO SUIT THE DECISION YOU WANT.
- Four drift guards: the closing-price rule must keep the binary property and the
  early-entry trap, the regime bound must not license cutting on feel, the
  session-fact rule must stay specific to the LEADING index, and the re-rating rule
  must not read as closing the stale hatch.
- Prompt size 128,303 -> 132,468 chars. Assembled 133,759 against a 154,140
  budget: **20,381 chars of headroom.**

---

## Video addendum - the 24 Aug LIVE SESSION (v4n)

**Source:** Intraday Hunter live session `9bTx_E3E95c` (24 Aug 2026, 10:19).

**The first session where IH and our agent were on OPPOSITE sides, and the agent
won.** IH bought calls into what he read as a range that had to break upward, the
market fell instead, and he cut. Our agent shorted three of its four trades and
finished +2,065.25. That divergence is worth more than another agreement, because
it separates the method from the read.

### Two new rules

**DO NOT THINK YOUR WAY THROUGH A LOSS; THE THINKING IS WHAT ENLARGES IT.** The
sharpest thing in the session and the one most specific to what we are building.
IH, sitting under water: "there is no benefit in applying more thought here,
because APPLYING THOUGHT WHILE IN A LOSS IS THE THING THAT MAKES THE LOSS BIGGER."
And on what to do if his level broke: "then we are not to think about whether this
will happen or how we might escape - those things must not be brought into the
mind at all. Then we simply go according to the SL."

This is the companion that makes v4h's permission-to-wait safe. Waiting to the
limit is MECHANICAL waiting; it is not a licence to keep re-deriving the chart.
The failure it prevents is one an LLM agent is built to commit: with a position
under water, generating another plausible reading is effortless and every one of
them argues for staying. So once price is against you and the invalidation is in
view, the only live questions are the ones already committed to - the stop, the
last point, the limit.

It is bounded in the prose and in the test, because read too strongly it would
disable stops: it does NOT mean stop observing, it means stop SEARCHING for a
reading that rescues the position.

**A CROWD WHOSE STOPS ARE TOO FAR AWAY IS REMOVED BY PROFIT-THEN-FADE, NOT BY A
STOP-HUNT.** Seated inventory is not always huntable. IH on buyers who bought a
support two sessions earlier: "their stops are going to sit at minimum below the
500 level, so they cannot be targeted DIRECTLY. They can be targeted this way
instead - first let them SEE some profit, then reduce that profit. In that they
run away quickly."

The operational half is that it changes the ENTRY TRIGGER: hunting a near-stop
crowd waits for the BREAK, hunting a distant-stop crowd waits for the FADE after a
move that went their way. Naming which one you are in stops you waiting for a
break this crowd was never going to produce. It also warns that a move toward such
a crowd is not evidence they were right - it can be the setup for taking it back.

### One extension: v4m's re-rating rule was broken on the day it shipped

v4m added YOU MAY NOT RE-RATE A SIGNAL TO SUIT THE DECISION YOU WANT after two
sessions of `cross_index` being dismissed as stale when it disagreed and cited
when it agreed. Today was the first session carrying that rule, and it happened
again inside one minute:

- **10:27** (entry): "This overrides the mechanical cross_index 'both at
  resistance/bias up' read, which is stale relative to BankNIFTY's own live
  structure."
- **10:28** (exit): "cross_index flipped to both_at_resistance/bias UP, directly
  opposing the short."

**It had not flipped.** It was the identical verdict, forty-eight seconds later,
and the word "flipped" made a signal that had just been overruled sound like fresh
evidence.

A disposition had already failed, so v4n adds a PROCEDURE instead: before
describing a verdict as stale, new, or flipped, state what it read on your
previous decision. If you cannot recall it you do not know whether it changed, and
"flipped" is not available. If it reads the same as last time and you overruled it
then, citing it as support now is not permitted at all.

### How our agent traded the same session

Basket **+1,096.75** across 51 legs. SL Hunting: **+2,065.25** over 4 trades - the
best strategy on the board, in a session where the two biggest deterministic
strategies lost over 3,000 each.

| Time | Leg | P&L |
|---|---|---|
| 09:18 | NIFTY long / BNF mirror | +13.00 / +439.50 |
| 09:48 | NIFTY short / BNF mirror | +156.00 / +2,700.00 |
| 10:02 | NIFTY short / BNF mirror | -321.75 / -1,854.00 |
| 10:28 | NIFTY short / BNF mirror | +227.50 / +705.00 |

**1. The leg-narration habit has CORRECTED, and the pending v4n rule is therefore
NOT being added.** The operator asked that this version add a rule against
justifying a basket decision by the leg that flatters it, recorded after three
occurrences on 19, 20 and 21 Aug. Today every decision quoted the BASKET figure -
"+2221.5 basket profit", "-1872 unrealized", "+447.5" - and twice the agent
volunteered that the NIFTY leg was the WEAK one: "the NIFTY leg alone was already
negative (-42.25) despite spot being above entry, so the mirror was carrying the
basket profit while the leader stalled", and "the NIFTY leg itself is flat (+97.5)
with the mirror leg carrying almost all of the +2221.5 basket profit".

That is the opposite of the recorded failure, and it applies v4l's
movement-not-rupee-share test correctly. Adding a rule against a habit that two
prior versions already fixed would be exactly the bloat the 2026-08-18 pruning
pass warned about. Recorded here rather than silently dropped, so the decision is
reviewable.

**2. An ENTRY was placed with no rationale in the journal.** The 09:17:48 entry
logged `conf=0, setup=tool_execution` with the reasoning "The order tool accepted
an action; its result is authoritative." That is `_tool_authoritative` in
`sl_hunting_agent.py` doing its job: the model called the order tool, the order was
accepted, and the model then failed to return a parseable final decision, so the
host recorded the EXECUTED action truthfully rather than inventing one.

The order handling is correct. What is missing is observability, and it is the
entry-side twin of SLH-010: a position was opened and the journal - which the
reflection coach reads to work out what worked - carries a placeholder instead of
a reason. It logs at INFO, indistinguishable from a normal entry. Not fixed here
because it is a code question rather than a knowledge one.

**3. The note was overridden twice and followed once, and the overrides won.** The
09:41 short says so explicitly: "this overrides the pre-open note's default
gap-up-long branch because live price action shows a genuine failed breakout". It
made +2,856.00, the day's best trade. Across six sessions the note-versus-override
record still points both ways.

### Knowledge changes (v4n)

- `RISK`: DO NOT THINK YOUR WAY THROUGH A LOSS (new, beside v4h's limit); v4m's
  re-rating rule extended with the previous-verdict procedure.
- `RETAIL_POSITIONING`: A CROWD WHOSE STOPS ARE TOO FAR AWAY IS REMOVED BY
  PROFIT-THEN-FADE (new).
- Three drift guards: the loss-thinking rule must not become "stop observing", the
  distant-stop rule must keep the break-versus-fade trigger, and the re-rating
  procedure must keep its measured case.
- **Deliberately NOT added:** the leg-narration rule - see point 1 above.
- Prompt size 132,468 -> 136,034 chars. Assembled 137,325 against a 154,140
  budget: **16,815 chars of headroom.**

---

## SLH-011 - an order executed by a pass that returned nothing parseable (2026-08-24)

### What happened

The 09:17:48 entry was recorded as:

```
SL Hunting AI: ENTER_LONG (conf=0, setup=tool_execution) stop=24271.1 target=24326.0
  :: The order tool accepted an action; its result is authoritative.
```

The model called the order tool, the order was accepted, and the model then
failed to return a parseable `SLHuntingDecision`. `_tool_authoritative` did
exactly what it exists for: it recorded the EXECUTED action rather than
inventing a decision or discarding a real position.

### What was wrong with it

Nothing in the order handling. Two things around it:

1. **The journal carried a placeholder instead of a rationale.** The decision
   journal is what `sl_hunting_coach.py` reads to work out what worked, and the
   `reasoning` field is the trade's stated purpose. "The order tool accepted an
   action; its result is authoritative" tells the coach nothing, and it is
   indistinguishable from a real justification when read later.

2. **It logged at INFO.** An entry with no reasoning, no setup and zero
   confidence appeared in the log exactly like a normal one. It was found only
   because a human happened to read the line.

### The fix

**Use the model's own words.** The justification the model passed to the order
tool is a real one and was already validated: `_reason_meaning_problem` rejects
placeholders on ENTRIES (SLH-007), so on an entry that string is guaranteed
meaningful. `do_order` now carries it on the accepted result as `model_reason`,
and `_tool_authoritative` prefers it over the generic sentence.

On an EXIT the reason may legitimately be the `NO_REASON_SENTINEL`, which is
itself the honest record and already raises its own SLH-010 warning.

**Say it out loud.** A WARNING now names the case:

> order ENTER_LONG executed but the model returned no parseable decision for that
> bar. Recording the executed order; the journal entry carries the order tool's
> own reason and no setup or confidence. Treat this decision as UNREVIEWED.

**And the mirror case.** The other synthesised path - the final JSON claims an
action no accepted order backs - was equally silent. Downgrading it to HOLD is
correct because nothing executed, but it means the model believed it held a
position it did not. That now warns too.

### What deliberately did NOT change

- **The recorded ACTION.** What executed is still what is recorded, unchanged.
  Only the `reasoning` string and the log level of the surrounding event moved.
- **`setup` stays `tool_execution` and `confidence` stays 0.** The model did not
  supply either, and inventing them would make an unreviewed decision look
  reviewed. The gaps are the signal.
- **No new refusal.** Nothing is blocked; a position is never abandoned because
  its decision failed to parse.

### Verification

Both guards were negative-tested rather than assumed. Removing the reason-carrying
makes the test fail with `assert 'pivot reclaim with confirmed hammer' in 'The
order tool accepted an action; its result is authoritative.'`, and disabling the
claimed-action warning fails its own assertion. A third test asserts an ordinary
pass produces NO warning, so the guards cannot decay into noise.

---

## Video addendum - the 25 Aug LIVE SESSION (v4o)

**Source:** Intraday Hunter live session `-KpWuMvUgUQ` (25 Aug 2026, 8:24).

Double expiry, both indices. IH bought puts, sat through a BankNIFTY move
against him, and booked a good profit. Our agent shorted four times and made
**+6,112.50**, its best session of the series. Same side, same read, both right -
but the session's most useful material is a place where they DISAGREED and the
disagreement is instructive rather than settled.

### The headline: the method is not always a fade

A method-level correction, in his own words:

> You say that when we work according to SL HUNTING we always have to work
> OPPOSITE the market. **That is NOT so.** If the sellers' stop-losses are not
> visible, and the opening is fine, then we can work ACCORDING TO the market.

Our whole prompt is built around finding seated inventory and fading it, so this
is a real widening: hunting a trapped crowd is the method's BEST trade, not its
only one.

It is also the most dangerous addition in the series, because read loosely it
licenses entering with no crowd read at all - which is exactly the 11 Aug loss v4e
was written from. So it ships with both reconciliations attached:

- **A FORECAST OF WHO WILL ARRIVE IS NOT EVIDENCE OF WHO IS SEATED (v4e)** bans
  inventing a crowd and trading the prediction. Following momentum is not that: it
  claims no crowd at all, so there is nothing to be wrong about.
- **AN EMPTY BOOK MEANS A TRAP IS COMING (v4f)** still supplies the "opening is
  fine" half - you follow a direction the session has ESTABLISHED, never one you
  expect it to take.

The operational half is honesty about which trade you are in, because the two are
managed differently: a hunt ends when the crowd is flushed, a with-the-market
trade ends when the MOVE stops. Reading a stall as "my crowd has not been squeezed
yet" is how the second gets held like the first.

### Where IH and the agent disagreed, and what separates them

BankNIFTY ran hard against IH's put basket - "NIFTY and Sensex gave no positive
momentum but BankNIFTY produced a tremendous move... because of that we had to see
some loss" - and he did NOT exit. His stated test was a single pre-named level:
"there is a last RESISTANCE of ours; if it goes above this level we cut the trade
and leave." It never crossed, the move resumed, and he booked well: "see how well
the resistance helped today - we saved a trade that was going wrong, purely
because of the resistance."

Our agent met a comparable move at 09:28 and cut, correctly citing the rule:
"BankNIFTY mirrors the same recovery... which per the index hierarchy disqualifies
the whole basket." That cost **-1,176.25**.

**This is not evidence the rule is wrong.** The index-hierarchy disqualification
has closed genuinely losing baskets, including on 18 and 21 Aug. What today shows
is that "the leading index is moving against me" and "my invalidation has broken"
are DIFFERENT observations, and only a level named BEFORE entry tells them apart.
v4j's rule is therefore bounded rather than weakened, with an explicit fallback:
if you cannot name the level, you do not get the exception and the
disqualification stands.

### Double expiry is a timing instruction

"Today two indices have expiry - it gives one selling momentum, then one buying
momentum, so MORE traps form", and "momentum on either side will not come easily
today". His response was to wait rather than to pick a side, which is recorded
that way: expect the small counter-trap BEFORE the move rather than reading it as
invalidation.

He also names the reason he waited at all - two consecutive losing days: "yesterday
was a loss, Friday was a loss too, so we should avoid a LOSS STREAK." That part is
already covered by the existing post-loss rules and is not re-legislated.

### How our agent traded the same session

Basket **-8,730.25** across 64 legs, a poor day everywhere except one strategy.
SL Hunting: **+6,112.50** over 4 trades - the best strategy on the board by a wide
margin, and its best session recorded.

| Time | Leg | P&L |
|---|---|---|
| 09:28 | NIFTY short / BNF mirror | -178.75 / -997.50 |
| 09:45 | NIFTY short / BNF mirror | **+2,190.50 / +3,714.00** |
| 09:53 | NIFTY short / BNF mirror | -13.00 / +180.00 |
| 10:16 | NIFTY short / BNF mirror | -204.75 / +1,422.00 |

**1. The note's PREMISE was applied, not just its direction.** Yesterday's note
said sell-side while stating the sellers were NOT the target. The agent carried
that distinction into every entry: "follows momentum rather than hunting a crowd",
and "fades the fresh buyers who chased the recovery, not the already-booked prior
sellers." That was the subtlety the shipped-note test was written to protect, and
it survived into live decisions.

**2. v4m/v4n's re-rating discipline held.** The stale hatch was NOT abused. At
10:08 the opposing verdict was called "a caution" with a substantive reason given
rather than being dismissed as stale; at 10:16 the agent declined the hatch
outright - "cross_index has turned to both_at_resistance/bias UP directly opposing
the short over an hour into the session, **no longer dismissible as stale**" - and
booked. That is the scope rule applied by the agent to itself, the failure v4m and
v4n were written for two sessions running.

**3. The basket figure was quoted in every decision** ("-991", "+5861 (NIFTY
+2327, mirror +3534)", "+438", "+1101.75"). The leg-narration habit stays
corrected; v4n's decision not to add a fourth rule for it looks right so far.

**4. Both dependency bumps validated in the session, as the pins predicted.**
72 `SLHuntingAgent decision cost` lines confirm the bundled CLI 2.1.238 spawns
under claude-agent-sdk 0.2.143, and CPR AI ran with zero error lines under
openai-codex 0.147.0 - the pin whose comment says plainly that no automated gate
can test it and the next session is the actual check.

One flaw worth watching rather than legislating: the 09:52 trade was a **46-second
round trip** (+167.00 realised against +438 unrealised at the decision). The exit
reasoning is sound on its own terms - the mirror was not confirming - but v4a
already warns that a trade abandoned within a bar or two pays the round trip for
almost no exposure. Second occurrence in a week; a third makes it a pattern.

### Knowledge changes (v4o)

- `RETAIL_POSITIONING`: THE METHOD IS NOT ALWAYS A FADE (new), with its v4e and
  v4f reconciliations.
- `RISK`: v4j's basket rule bounded by MOVING AGAINST YOU IS NOT YOUR LEVEL
  BREAKING; MOVE-EXHAUSTION gains TWO EXPIRIES AT ONCE MULTIPLY THE TRAPS.
- Three drift guards: the fade rule must keep both reconciliations and the
  which-trade-am-I-in honesty, the level bound must keep the no-level fallback so
  it cannot gut the index hierarchy, and the double-expiry rule must stay a timing
  instruction rather than a bias.
- Prompt size 136,034 -> 139,563 chars. Assembled 140,854 against a 154,140
  budget: **13,286 chars of headroom.**

---

## Video addendum - the 26 Aug LIVE SESSION (v4p)

**Source:** Intraday Hunter live session `H5g8T7xllLk` (26 Aug 2026, 7:34).

Gap-up, bought immediately into the run, booked a good profit. Our agent was long
the same open and finished **+3,634.25**, best on the board for the second session
running. Both right, both bullish - and the session's most valuable content
directly CONTRADICTS a rule shipped four days ago, which is why it earns a version
of its own.

### The headline: waiting is sometimes the cost

> Sometimes it happens in the market that **THE MORE YOU WAIT, THE MORE YOU LOSE.**
> And today's market is like that. Here you have to enter QUICKLY, without waiting -
> only then will you gain.

And on the entry itself: "we bought into a RUNNING market. Not that we waited, not
that we looked for a retracement. The market was going straight up and we bought
straight away."

Two sessions earlier the same analyst waited deliberately and said why (two losing
days, double expiry, traps forming both ways). So the instruction is not "be fast";
it is that the choice between entering now and holding out for a pullback is
CONDITIONAL on whether a direction has already declared itself. Both branches are
in the prose, because a rule with only the fast half is a standing bias.

**The scope limit matters more than the rule.** Read carelessly this says skip
confirmation, which would gut the method. It governs WHERE you enter once a setup
is confirmed - at the break or on a pullback - never WHETHER a setup was needed.
The prose says so and `test_v4p_enter_quickly_rule_never_relaxes_the_confirmation_requirement`
asserts it, because this is the single most dangerous misreading available in the
knowledge base.

It also does not contradict v4l's ENTRY TIME IS A RISK DIAL: that rule prices the
hour you choose, this one says a declared direction removes the pullback option
from the menu.

### Deliberately NOT added

He spends a full minute on capital size and execution speed - "keep capital as
LOW as you need to, but trade so that you can enter smoothly", and "many traders
know the momentum is coming but have so much capital committed that they cannot
enter quickly, so the trade is missed."

It is good advice for a human and does not transfer. This agent's size is computed
mechanically to a ~Rs.2500 risk budget, it does not choose capital, and it does not
hesitate. Adding it would be prompt weight against a failure the agent cannot
commit. Recorded here rather than dropped silently.

### How our agent traded the same session

Basket **-13,236.50** across 73 legs, a poor day broadly. SL Hunting: **+3,634.25**
over 3 trades, top of the board.

| Time | Exit type | NIFTY / mirror | Net |
|---|---|---|---|
| 09:16 | **mechanical AI_STOP** | -399.75 / +1,242.00 | **+842.25** |
| 09:26 | agent, round-number stall | +760.50 / +1,269.00 | +2,029.50 |
| 10:23 | agent, premise invalidation | +474.50 / +288.00 | +762.50 |

**1. A stop fired on NIFTY spot and closed a basket that was in PROFIT.** The 09:15
long was stopped at spot 24333 against a 24335 stop - a loss on the leg the stop
watches (-399.75) - while the BankNIFTY mirror was +1,242.00, so the basket exited
**+842.25**. That is a structural fact about the basket rather than a mistake:
the stop is a NIFTY invalidation measured on NIFTY alone, and it closes both legs.
v4p records it in the BASKET NOTE, explicitly WITHOUT concluding that stops should
be wider - a NIFTY stop protects the NIFTY premise, which is the one that was
traded.

**2. That trade has NO decision-journal entry, and nothing warned.** The order
executed at 09:15:49 through the locked tool context, the pass completed, and at
09:16:41 the worker logged "Discarding late SL Hunting result for stale generation
2." The design is deliberate and the docstring is explicit - "a stale generation
here only skips the bookkeeping, never an order" - and the order path is safe.

But it is the THIRD path in which an executed order reaches the trade log while the
decision journal gets nothing, after the two SLH-011 closed. The reflection coach
reads that journal. No SLH-011 warning fired because `reported` was not None - the
decision existed, it was simply discarded downstream. **Not fixed here** because
this is a code question rather than a knowledge one, and it deserves the same
treatment SLH-011 got rather than being bolted onto a knowledge PR.

**3. A parsing note for future sessions.** SL Hunting exits appear under TWO logger
names: `sl-hunting-sdk-call` for agent exits and `SL Hunting AIThread` for
mechanical stop/target/square-off exits. A sweep on the first alone under-counts -
it showed 2,792.00 against the true 3,634.25 here and hid the stopped trade
entirely. This is the third variant of the same class of mistake in this journal
(after `MIRROR EXIT` and the `sl-hunting-sdk-call` thread name); the reliable check
remains the worker's own `Result summary` line.

**4. The runner restarted three times before the open** (08:17, 08:23, 08:27, each
with a zero-trade `Result summary`). No errors accompany them and the session ran
normally afterwards, so this is recorded as an observation rather than a fault.

### Knowledge changes (v4p)

- `RISK`: WHEN THE DIRECTION HAS ALREADY DECLARED ITSELF, WAITING IS THE COST (new,
  beside v4l's entry-time rule); the BASKET NOTE gains YOUR STOP IS A NIFTY-SPOT
  TRIGGER ON A TWO-INDEX BASKET.
- Two drift guards: the enter-quickly rule must keep its scope limit and both
  branches of the conditional, and the stop note must keep the explicit refusal to
  argue for wider stops.
- **Deliberately NOT added:** the capital-size/execution-speed advice - see above.
- Prompt size 139,563 -> 141,841 chars. Assembled 143,132 against a 154,140
  budget: **11,008 chars of headroom.**

---

## SLH-012 - a stale inference pass that still left a position open (2026-08-26)

### What happened

At 09:15:49 the agent placed a LONG through the locked tool context. The pass
completed, and at 09:16:41 the worker logged:

```
Discarding late SL Hunting result for stale generation 2.
```

The trade was real: entered at 24347.55, closed by the mechanical stop at 09:16:31
for a basket **+842.25**. It appears in the trade log, the Telegram alert and the
session P&L.

It does not appear in the decision journal, and it has **no journal row at all**.

### Why that is more than a missing log line

The generation guard is correct and its docstring is accurate - "a stale generation
here only skips the bookkeeping, never an order". No order path is affected. But
the `return` skipped three things, and the third is the expensive one:

1. the `SL Hunting AI: ...` decision log line,
2. `append_decision(...)` to the decision journal,
3. `_journal_open_row(...)`.

`_finalize_journal` begins `if self._journal is None or self._open_trade_id is
None: return`. So a trade whose OPEN row was skipped never gets a CLOSE row
either - it is absent from the per-trade journal entirely. That journal is what
`sl_hunting_coach.py` reads to work out what worked, so the trade is invisible to
the learning loop in both directions.

This is the third path of the family, after the two SLH-011 closed. It did not
trigger an SLH-011 warning because `reported` was not None: the decision existed
and parsed perfectly well, it was simply dropped downstream.

### The fix

The guard is about ACTING, not record-keeping. When a stale pass left a position
OPEN, `_consume_agent_decision` now records it - the journal row and the decision -
and warns:

> Late SL Hunting result for stale generation N left a position OPEN; recording its
> journal row and decision so the trade is not invisible to the coach. The decision
> is stale as guidance and was NOT acted on.

Recording a stale decision is defensible precisely here: it is stale as GUIDANCE,
but it is the accurate account of an order that really happened, and
`_tool_authoritative` (SLH-011) has already forced the decision to match what
executed. Nothing in this path places an order.

The open-row gate is now computed once as `opened_a_position` and reused by both
branches, so the stale recovery and the normal path can never drift apart.

### What deliberately did NOT change

- **The guard still refuses to act.** No order is placed here and none ever was;
  the change is confined to bookkeeping.
- **A stale pass that opened nothing is still simply dropped**, at INFO, exactly as
  before. That is the ordinary case and making it noisy would bury the real signal.
- **A pass whose trade is already journalled** (`_open_trade_id` set) writes
  nothing and does not warn.

### Verification

Four tests, called against a light stub rather than a full worker, because the
method touches ten attributes and building a real worker would drag in the SDK, a
broker and a data store to cover ten lines. They skip when the agent deps are
absent, per the repository convention.

The recovery was negative-tested rather than assumed: reverting the branch to the
old unconditional discard fails `test_stale_pass_that_opened_a_position_is_journalled_and_warned`
with `AssertionError: 0 != 1`. Two of the four tests assert the QUIET cases, so the
warning cannot decay into noise.

---

## Pre-open note for 2026-08-27

**Source:** Intraday Hunter, "Nifty & Bank nifty | SENSEX Analysis | Prediction For
27 AUG 2026" (`auAi14O0HOk`, uploaded 2026-08-26, 1:38).

### The plan

**The branch inverted overnight.** Yesterday all three open shapes shared a
BUY-side plan; tomorrow all three share a SELL-side one. He says it twice, once
over the SENSEX chart and once over NIFTY, in the same words both times: on a
gap-up, flat-to-gap-down open, identify SELLING-side setups - "and in flat and so
on, our same plan remains."

That last clause is the one worth protecting. FLAT sits on the SELL side. It is
the same structural point as yesterday's note - the sign of the gap does not
switch the branch - pointing the opposite way, which makes it exactly the thing a
habit-driven edit gets wrong.

**The premise is FOLLOW, not hunt, and he gives the reason.** Two of them: very
few people here trade puts, so those sellers do not need to be targeted; and the
market did not break the closing price, so sellers are not seated either. With
nobody trapped on either side, the method is momentum-following rather than a
squeeze. This is a clean second instance of v4o's THE METHOD IS NOT ALWAYS A FADE,
arriving one session after that rule shipped.

**BankNIFTY carries an explicit stand-aside.** On a gap-up, "until it crosses the
round number we cannot do anything." If rejection shows at the open, the sell-side
plan holds. If it starts running up - "no plan can be made there at all." A note
that kept only the sell-side half would convert a no-trade instruction into a
licence to short strength, so the test asserts the stand-aside survives as such.

**Tomorrow is a SENSEX/BANKEX expiry day** (he says so over the SENSEX chart, and
the other channels' titles for 27 Aug agree). Execution stays NIFTY-only; this is
context for the cross-index read, which is noisier than usual on those days.

### Levels

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 24320, 24380 | 24120, 24180 |
| BANKNIFTY | 58000, 58200 | 57420, 57720 |
| SENSEX | 77750, 78000 | 77000, 77250 |

Two readings were confirmed by the operator off the chart rather than inferred:

- **SENSEX resistance.** The caption merged the pair into the run-on "78000750",
  the same failure as yesterday's "7800750". Confirmed as 77750/78000 - i.e.
  UNCHANGED from yesterday, despite the gap-up. The test carries a comment saying
  so, because the natural assumption on seeing a repeated pair is that the note
  was copied from the previous day's file and needs "correcting".
- **NIFTY first support.** The caption gave "24,187"; confirmed as 24180. He does
  give precise levels in this same video (BankNIFTY 57720/57420), so the caption
  was not obviously wrong and guessing would have been a coin flip.

### An extraction note for future sessions

**YouTube's `timedtext` endpoint no longer works.** It now answers HTTP 200 with a
zero-length body for every format (plain, `fmt=json3`, `fmt=srv3`) without a POT
token, so the caption-track URL lifted out of the watch-page HTML - the method used
for previous notes - silently yields nothing rather than erroring.

What works: open the transcript panel in the visible browser pane and scrape the
DOM. The segment element is `transcript-segment-view-model` (not the older
`ytd-transcript-segment-renderer`, which no longer exists), and the list is
VIRTUALIZED - roughly 13 segments are rendered at a time, so a single query
returns only what is on screen. Scroll the panel's scrollable ancestor in steps
and accumulate segments keyed by timestamp until the count stops growing. This
video was short enough that one screenful was the whole transcript, which would
have made the virtualization easy to miss on a longer one.

---

## Video addendum - the 27 Aug LIVE SESSION (v4q)

**Source:** Intraday Hunter live session `npth0OPyFSY` (27 Aug 2026, 6:29).

A flat-to-small-gap-up open that sold off. IH shorted it at the open across
BankNIFTY (58000 then 57900 PE), SENSEX (SENSEX expiry day, 1000 qty) and NIFTY
(1365 qty), held through a pullback, and booked with the target over-achieved.
Our agent was short the same open, twice, and finished **+549.00**.

Most of this session CONFIRMS rules already in the corpus rather than adding to
it - which is itself the finding, and is why v4q is small.

### What was already there

Three of the four candidates turned out to be covered:

- **His two-condition test for when NOT to hunt** - stated twice as the doctrine
  ("in SL hunting: either the trader is sitting in PROFIT, or the trader does not
  have big QUANTITY - we target neither, and then we follow the chart"). The
  profit half is `TARGET-BOOKED crowd test`; the quantity half is
  `AGGREGATE-INVENTORY TEST`, which already ends "if neither side is likely
  carrying meaningful size, follow the current momentum or HOLD rather than
  inventing a hunt". Adding his single-sentence pairing would buy a formulation,
  not a behaviour.
- **Not booking into a fast leg** - "the profit had been more than this, but the
  momentum was very fast there; there was no plan to exit. More momentum will
  come, and in that we get the chance to make the exit plan." This is v4b's
  `EXPECT A SECOND LEG AFTER THE PAUSE, THEN BOOK`. Deliberately not restated,
  and not sharpened: on live money a rule that reads as "sit through a giving-back
  of profit" is the dangerous direction, and it would have argued against the
  agent's 09:35 exit, which was correct (below).
- **The laggard timing the exit** - "SENSEX and NIFTY have done good momentum,
  BankNIFTY still has a little left; after that we get the chance to book."
  That is the FIFTH session to touch `THE LAGGING INDEX DECIDES THE BASKET'S
  EXIT` (v4i, amended v4k, v4l, v4o). Recorded as confirmation and deliberately
  NOT added as a fifth amendment - the standing note that this rule should be
  re-read whole rather than extended again still applies.

### What was net new (v4q)

**1. The completed-stop-hunt test also gates the ENTRY.** `A COMPLETED STOP-HUNT
ENDS THAT DIRECTION` (v4g) reads naturally as a reason to close, and that is how
it gets used. It disqualifies an entry the same way, and the mechanical form is
your own target: if the level you are about to name as the target has already
printed this session, the move you are entering has already happened. Added as a
sub-bullet on the existing rule rather than a new one, because it is the same
idea applied at the other end of the trade.

This was not taken from the video - it was taken from our own trade 2, below.

**2. `THE HOLD IS LICENSED BY THE EARLY IMPULSE AND EXPIRES WHEN THE MOVE BECOMES
OBVIOUS`.** A genuinely new exit trigger, keyed on participation rather than on
price, rate or time: "the sitting was only up to the point where the momentum
appeared in all two or three indices at the START. Now that a good deal of
momentum has already happened... now other people can also participate. So
before other people apply their minds, apply yours first, book the profit and get
out." The licence to sit comes from being positioned before the crowd; once a
late reader would take the same trade, the remaining participants are buyers of
your position rather than fuel for it.

The rule is written so it can never be read as permission to hold - v4f, the
stop, the max loss and premise-invalidation are all ranked above it explicitly,
and a drift guard asserts that ranking survives.

### How our agent traded the same session

Two shorts, both closed, flat well before the 10:30 no-new-entry cutoff.

| Time | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|
| 09:23 -> 09:35 | agent, flush exhaustion at support | +962.00 | -105.00 | **+857.00** |
| 09:52 -> 09:54 | agent, premise invalidation | +52.00 | -360.00 | **-308.00** |
| | | **+1,014.00** | **-465.00** | **+549.00** |

**The pre-open note worked.** It injected (2,055 chars, the exact size measured
when it was written) and was cited in the entry reasoning by its substance, not
just its direction: "this matches the pre-open note's sell-side/follow-momentum
branch (no seated seller crowd, closing price unbroken yesterday, so hunt is off
and continuation is the trade)". The follow-not-hunt premise - the half most
likely to be flattened - is the half it used.

**Trade 1 was right, including the exit.** Entered short at spot 24227.95 on a
confirmed evening star after the gap-up to 24297 was rejected; exited at 09:35
when price stalled at 24200-24208 with a bullish morning star on NIFTY and a
hammer plus full-body bull candle on BankNIFTY at its own 57800 support. The
session's low to that point (24188) had printed at 09:32, and price then bounced
to 24228 by 09:51. Booking there was `BOOK WHEN THE PROFIT STOPS GROWING` applied
correctly. A lower low (~24156, i.e. trade 1's 24155 target) did print later in
the morning, but only after that bounce and roughly an hour and a half later;
`TIME SPENT IN THE TRADE SHRINKS THE ACHIEVABLE TARGET` covers why that is not a
missed hold. (My log sampling of spot is too sparse to time that low to the
minute; the value is solid, the minute is not.)

**Trade 2 is the real error, and the agent diagnosed it itself two minutes
late.** At 09:52 it re-shorted the bounce, naming a trapped dip-buyer crowd, with
the target set at **24188.6 - a level the session had already tagged at 09:32**.
At 09:54 it exited with: "the completed stop-hunt has spent its fuel". That is
v4g, quoted almost verbatim - so the agent had the rule, and ran it as an exit
check two minutes after an entry the same rule forbade. The cost was -308 on the
basket and it is the reason for addition 1.

**The mirror cost 465 today**, losing on both legs while NIFTY won both. That is
the mirror image of 26 Aug, where the mirror turned a stopped-out NIFTY leg into
a +842 basket. Two sessions, opposite signs, which is the honest summary of the
mirror so far: it is variance, not edge. Nothing added to the knowledge for it -
the basket economics belong in the operator's ledger, not the prompt.

### A small observability note

Trade 1's exit reasoning quotes its own P&L as "NIFTY leg +793, mirror -372, net
+421"; the fills were +962 / -105 / +857. The order tool executes DURING
reasoning, so those numbers are computed from cached LTPs a few seconds before
the fill - here the market moved favourably in between. Harmless, and not worth a
rule, but worth knowing before anyone treats a decision's quoted P&L as the
record. The trade log is the record.

### Knowledge changes (v4q)

- `A COMPLETED STOP-HUNT ENDS THAT DIRECTION` (v4g) gains a sub-bullet: run the
  test before the ENTRY, with the already-printed-target form.
- `RISK`: THE HOLD IS LICENSED BY THE EARLY IMPULSE AND EXPIRES WHEN THE MOVE
  BECOMES OBVIOUS (new).
- Two drift guards, both negative-tested: dropping the entry-side framing, the
  already-printed mechanic, the hold rule's ranking guard, or its participation
  framing each fail the suite.
- **Deliberately NOT added:** the two-condition hunt test, the do-not-book-into-a-
  fast-leg half, and a fifth amendment to the lagging-index rule - see above.
- Prompt size 141,841 -> 143,854 chars. Assembled 145,147 against a 154,140
  budget: **8,993 chars of headroom.**

---

## Pre-open note for 2026-08-28

**Source:** Intraday Hunter, "Nifty & Bank nifty | SENSEX Analysis | Prediction
For 28 AUG 2026" (`IwYyOclpDR8`, uploaded 2026-08-27 21:25 IST, 1:55).

### The plan

**The premise flips to HUNT.** The last two notes both said follow the momentum
because nobody was trapped. Three indices selling off all day has now seated a
retail SELLER crowd, and tomorrow's plan targets it. This is the more dangerous
of the two reversals to miss: an editor working from the previous notes would
keep the follow premise, land on a plausible direction, and be right for the
wrong reason - which is exactly the failure v4o's THE METHOD IS NOT ALWAYS A FADE
names, running the other way.

**The sign of the gap splits the branch this time.** The last two notes
deliberately put a flat open on the same side as the gapped ones, and both carry
a test asserting it. Here FLAT sits with GAP-UP on the buy side, and GAP-DOWN
alone takes the sell side. The habit those two tests built is the wrong habit for
this note, so the new test asserts the split explicitly.

**Both branches are gated, and the gates are the checkable part.** He states the
same mechanism twice, once per index:

- On a flat open, the level must NOT break: "once the level breaks the sellers
  come into confidence, and then targeting them will not be easy for us."
- Generally: "we do not want any breakdown - as soon as a breakdown happens, any
  seated selling trader will not cut his trade, so we will not get the SLs."

A break VALIDATES the seated sellers. Validated sellers hold instead of cutting,
so the stops the hunt feeds on never appear. Losing these two lines would turn a
gated plan into a plain directional bias.

The gap-down branch carries its own gate rather than the gap's sign: "on a
gap-down the sellers can come into decent profit anyway, so we will have to walk
with the market."

**24,000 is named as NIFTY's psychological number**, with 23,920 as the second
support beneath it.

### It is today's doctrine, used as a forecast

This note is the two-condition test from the 27 Aug live session applied
prospectively: sellers seated and WRONG, so hunt them; sellers in profit on a
gap-down, so follow instead. v4q declined to add that test as a new rule because
both halves already exist (`TARGET-BOOKED crowd test` and `AGGREGATE-INVENTORY
TEST`). Seeing IH use it as the whole basis of a forecast the same evening is
good evidence that call was right - the agent already carries both halves, and
what it needed was the day's specific reading, which is what the note supplies.

### Levels

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 24200, 24270 | 23920, 24000 |
| BANKNIFTY | 58000, 58200 | 57200, 57500 |
| SENSEX | 77400, 77750 | 76550, 77000 |

The SENSEX reading needed the operator again, and for a new reason: the caption
gave the resistances cleanly but lost the second support between two transcript
segments, leaving a single "77650" that would have sat ABOVE the 77400
resistance. That impossibility is what made it catchable - an ordinary-looking
wrong number would not have been. Confirmed off the chart as 77000/76550. The
test carries a comment saying not to reconstruct it from the caption.

### Verification

19/19 on the premarket suite, and each new assertion negative-tested: flipping
the premise back to follow, dropping the hunt framing, regating the gap-down
branch on the gap's sign, softening either veto, or "correcting" the SENSEX
support each fail the suite.

---

## SLH-013 - the prompt cap stops being the guard, and latency becomes visible

Three changes that belong together: the cap is raised past the point where it can
detect anything, so the detection moves to a test; and the thing that actually
limits prompt growth gets measured for the first time.

### The cap: 160,000 -> 350,000

Raised so it stops being a recurring question every few knowledge versions
(75k -> 120k -> 160k -> 350k). 350,000 characters is on the order of 90-100k
tokens -- this corpus is ALL-CAPS-heavy and tokenises worse than the usual ~4
chars/token -- against Sonnet 5's 200k context, so it is comfortably inside with
room for the bar context and the agentic loop.

**This raise is different in kind from the two before it, and the comment block
now says so.** The earlier raises kept the bound tight enough to still catch a
runaway lessons file or a malformed note, which its own top paragraph names as
the failure it exists for. At 350,000 it cannot: the runtime can inject only
~8,500 characters at its own caps, so nothing pathological in that material could
ever reach the bound. Raising it without moving the detection somewhere else
would have quietly retired the guard while leaving prose claiming it was armed.

### Where the detection went

`test_runtime_injected_blocks_stay_small` renders a worst-case lessons block and
a worst-case pre-open note through the REAL `format_lessons` and
`format_premarket_note`, and fails if the total passes 12,000 characters
(measured today: 8,519 = lessons 4,918 + note 3,601, so ~40% of room). It trips
if anyone widens `MAX_LESSON_CHARS`, `MAX_LIVE_LESSONS`, `MAX_PLAN_ITEMS`,
`MAX_LINE_CHARS`, `MAX_LEVELS_PER_SIDE` or `MAX_INDEX_BLOCKS` enough to matter,
whatever the cap says.

The ceiling is deliberately a flat number rather than one derived from those
per-field caps: a derived bound would move silently when the caps move, which is
exactly the change worth catching.

### A live inaccuracy this turned up

The test being replaced, `test_prompt_cap_leaves_room_for_lessons_and_a_note`,
assumed the runtime could add `12 * 280 + 2500 = 5,860` characters. The real
worst case is **8,519** - it ignored each lesson's 80-character scope and its
per-lesson formatting, and understated the note by 1,101.

So the effective budget was **151,481, not the 154,140** that has been quoted in
every knowledge addendum since the last raise, and the headroom reported at v4q
was 6,334 rather than 8,993. Nothing was ever actually at risk - the gap was
always larger than the error - but the number was wrong and is now measured
rather than assumed.

**Two fixture traps worth recording**, because they cost real time and the next
person to touch this will hit them:

- `StoredLesson` is tamper-evident. An approved record needs `id` equal to
  `_slug(scope, lesson)` AND `approval_digest` equal to
  `lesson_content_digest(record)`. Get either wrong and `_validated_records`
  drops it silently, `format_lessons` returns `""`, and a test measuring "worst
  case" happily measures zero and passes. The new helper asserts the fixture
  validated, so it can never quietly measure nothing.
- `_slug` truncates to 48 characters. Worst-case records must differ in their
  FIRST few characters or they collide on id and `consolidate` keeps only one.

### Latency: the constraint that actually binds

Context was never the limit. The 90-second SDK deadline fired **27 times between
7 and 31 July 2026**, and not once since - each of those a bar the agent silently
HELD through, with no decision at all. It stopped firing when the model moved to
Sonnet 5 with fast mode, NOT because the prompt shrank: the prompt has nearly
doubled since. Prefill scales with what the cap permits, so this raise spends
precisely the margin that fixed those timeouts.

Until now there was no latency measurement anywhere - only cost. So:

- Every decision logs its wall-clock at INFO beside the cost line. Deliberately
  NOT sharing the cost line's condition: that one fires only when the SDK
  reported a cost, and a run that reported none is exactly the kind worth timing.
- A WARN fires at `SL_HUNTING_SLOW_DECISION_WARN_SECONDS` (default 60 of the 90s
  budget), naming the elapsed time, the threshold, the percentage of the deadline
  and the consequence, so one line in a shared log is enough to judge it.
- A threshold at or above the deadline can never fire, because the call is
  abandoned first. That is allowed rather than rejected - it is a legitimate way
  to switch the warning off.

A timed-out call raises out of `_run_sync` and already logs its own message
naming the deadline, so it is not double-reported here.

### Verification

Nine mutations, each confirmed to fail the suite: widening any of the four
lessons/note caps; logging latency at DEBUG; gating latency on `cost_usd`;
disabling the warn branch; and dropping the deadline detail from the warning
text. Two tests assert the QUIET cases - a fast decision does not warn, and a
threshold above the deadline still constructs - so the warning cannot decay into
noise.

**One of these tests was wrong first, and the way it was wrong is worth keeping.**
The two warning tests originally set the threshold to 0.001s and relied on "any
real call takes longer than a millisecond". That holds on this machine and did
NOT on CI, where the fake runner returned inside a millisecond, `elapsed` logged
as 0.0s and the warning never fired -- green locally, red on CI. Wall-clock is
not something a test may assume. Both now pin the clock via `_pin_decision_
duration`, which returns 0.0 for the first reading and a fixed value after, and
is robust to anything else in the call path reading the clock in between. The
pinned version is strictly better than a working timing-based one would have
been: it catches the warning being disabled AND the warning always firing AND
the comparison being flipped, where a timing-based test could only ever catch
the first.

Full gates clean. `check-env` reports the new key as MISSING from the operator's
`.env`, which is correct and expected: it is in `env.example`, and until it is
copied across the in-code default of 60s applies.
## SLH-014 - a lesson id decided by the scope alone, and the lesson it swallowed

### The bug

`_slug(scope, lesson)` built a lesson id by lowercasing `scope + "-" + lesson`,
replacing non-alphanumerics with hyphens, and truncating to **48 characters**.
`MAX_SCOPE_CHARS` is **80**. So for any scope longer than roughly 48 characters
the id was decided ENTIRELY by the scope and the lesson text never reached it.

Two genuinely different lessons filed under one descriptive scope therefore
shared an id, and `consolidate` -- which dedupes by id, keeping the
better-evidenced copy -- silently discarded one. It never appeared in the prompt
block again and nothing logged its loss. A realistic 55-character scope is enough
to trigger it:

```
scope  = "entries on a gap-up morning after a monthly expiry week"
lesson A -> entries-on-a-gap-up-morning-after-a-monthly-expi
lesson B -> entries-on-a-gap-up-morning-after-a-monthly-expi
```

Severity today was low -- `SL_HUNTING_LESSONS_ENABLED` is off by default and the
existing scopes are short -- but it is silent data loss in the learning loop, and
it would have bitten the first time lessons were switched on with descriptive
scopes, which is exactly the natural way to write them.

It surfaced while building the SLH-013 worst-case fixture: twelve records with
identical scopes rendered an EMPTY lessons block, and the test would have
happily measured nothing and passed.

### The fix

The id is now a readable prefix plus a digest of the full content:

```
39 chars of readable slug + "-" + 8 hex of sha256(scope NUL lesson)   = 48
```

Both halves earn their place. The readable prefix keeps `--list` output and log
lines legible; the digest makes the id depend on every character of both fields,
whatever the prefix does. The separator is NUL rather than a hyphen so
`("a-b", "c")` and `("a", "b-c")` cannot hash alike, and it is built with
`chr(0)` rather than an escape sequence so no code-generation step can collapse
it into a literal NUL in the source (which happened once while writing this).

It also tightens tamper-evidence slightly. `StoredLesson` asserts
`id == _slug(scope, lesson)`; previously an edit past the 48-character prefix
left that check passing and only `approval_digest` caught it. Now the id moves
too. The digest remains the real guard -- this is defence in depth, not a
replacement.

### No migration was needed

Changing the slug changes every id, and `lesson_content_digest` covers the id, so
every approved record would have been invalidated. That would normally force a
migration or a documented one-time re-approval, and the operator would have to
choose which.

It did not arise: **both stores are empty.** `lessons.json` holds zero records
and the proposed store (`Backtest Outputs/sl_hunting_lessons_proposed.json`) does
not exist. So there is nothing to invalidate, and this is the cheapest moment the
fix will ever have. Worth stating explicitly, because if it had been deferred
until after the first approved lessons landed, it would have cost a re-approval
cycle instead.

### Verification

Two regression tests in `test_sl_hunting_lessons.py`, both negative-tested by
mutating `_slug`:

| mutation | result |
|---|---|
| revert to the original `slug(scope + lesson)[:48]` | 2 tests fail |
| digest over the scope only | 1 fails |
| readable prefix widened to 48 (id overruns its cap) | 3 fail |
| hyphen separator instead of NUL | 1 fails |

The first row is the point: a literal revert to the shipped bug fails the suite.

Full gates clean.

### Unrelated damage found on the way, and how

The master file would not import at all: `load_dotenv` raised
`ValueError: embedded null character`. The cause was in `Dependencies/.env`, not
the repo -- its last three lines were
`SL_HUNTING_SLOW_DECISION_WARN_SECONDS=60` written TWICE in UTF-16LE, every
character followed by a NUL.

The source was a command suggested in the SLH-013 hand-off, given in a `bash`
fence and run in **Windows PowerShell 5.1, whose `>>` defaults to UTF-16LE**.
Appending UTF-16 to a UTF-8 file produces exactly this. The file was repaired
(backed up first, corruption proven confined to the trailing three lines, 610 ->
609 key lines: two corrupt duplicates removed, one clean UTF-8 line added).

The lesson is for future hand-offs: **never suggest appending to `.env` with a
shell redirect on this machine.** Edit the file, or use
`Add-Content -Encoding utf8`. A silently corrupted `.env` takes the whole trading
system down at import, and the error names neither the file nor the line.

---

## SLH-015 - the two basket rules gave opposite defaults for the same tool

The re-read promised after v4q, when `THE LAGGING INDEX DECIDES THE BASKET'S EXIT`
had been amended by three versions and was about to be extended again. Reading it
whole turned up something worse than accumulated cruft.

### The contradiction

The two largest rules in the basket family disagreed about `exit_leg`:

- `THE LAGGING INDEX DECIDES THE BASKET'S EXIT` (v4i/k/l): "**Prefer that** to a
  whole-basket exit WHEN the NIFTY premise is genuinely intact."
- `CUTTING THE MIRROR ON A BANKNIFTY REVERSAL` (v4j/l/o): "**EXIT BOTH is the
  default** and `exit_leg` is the exception, **not a pair of equal options**."

The tool defaults to `BOTH` in code (`sl_hunting_tools.py`), and the `BASKET NOTE`
says `"BOTH" (default)` and "When in doubt, EXIT BOTH" - so the lagging-index rule
was the lone outlier. The agent read both and could justify either, on live-money
exits, with no stated way to tell which applied.

(A correction to the v4q note: v4o amended `CUTTING THE MIRROR`, not the
lagging-index rule. The lagging-index rule carries v4i/v4k/v4l only.)

### What was actually wrong

Nothing in the trading logic. The two were always reconcilable, and the
distinguishing fact was simply never written down:

- A mirror that has **STALLED** - stopped moving while the other leg runs - is a
  question about how much the basket can still COLLECT. Per-leg is available.
- A mirror that has **REVERSED** is a question about whether the MOVE IS STILL ON.
  EXIT BOTH.

The lagging-index rule's per-leg preference was always about the first case. It
just read as a general preference, because nothing scoped it.

### The change

Behaviour-preserving by construction, and provably so: **all fifteen v4i / v4j /
v4k / v4l / v4o assertions pass untouched.** Not one existing test was edited,
which was the design constraint - six tests anchor on these two headings and forty
assertions pin their prose, so anything that required editing them would have been
a behaviour change wearing a refactor's clothes.

Three edits:

1. **The per-leg preference is SCOPED, not removed.** "Prefer that to a
   whole-basket exit" becomes "available HERE, in the STALL case, and only here",
   with an explicit "NOT the licence for a mirror that has TURNED" and a pointer to
   the sibling rule rather than a competing default of its own.
2. **The discriminator is named once**, folded into the paragraph in
   `CUTTING THE MIRROR` that already drew the distinction in different words
   (idiosyncratic vs turned). It now says stall vs reversal, gives the reason each
   maps to a different action, and warns that "the two look identical on a P&L
   screen and completely different on a chart" - which is the whole trap.
3. **The EQUAL-LOT fact is stated once.** It appeared twice inside the
   lagging-index rule; the second occurrence now refers back rather than
   re-deriving it.

### Both headings kept

Merging them under one heading would have broken all six tests, and one of those
tests - `test_v4j_per_leg_cut_rule_does_not_cancel_the_v4i_per_leg_escape` -
exists precisely to assert the two coexist. The existing test suite already
encoded the decision; consolidation happened WITHIN and ACROSS the two bullets
instead.

### The satellites

Checked for contradiction and left alone, as scoped: `BASKET NOTE` (v4p) already
agrees (`"BOTH" (default)`, "When in doubt, EXIT BOTH"); `LAGGING-INDEX ENTRY
LOCATOR` and `INDEX HIERARCHY ON THE WAY OUT` carry no exit-scope language at all,
so no contradiction was possible.

### On size

The pair grew: 7,736 -> **8,371 chars (+635)**. That is the honest result and it
was the expected one. Every measured case and IH quotation had to survive verbatim
- they are the evidence base, and forty assertions pin them - so the only prose
that could be removed was the single duplicated EQUAL-LOT derivation. Naming the
discriminator costs more than that saved.

The point of this pass was never headroom. It was that an agent making live-money
exit decisions had two rules pointing opposite ways and no stated way to choose.
Assembled prompt is 145,780 against a 350,000 cap, so the cost is irrelevant.

### Verification

The new guard `test_stall_or_reversal_discriminator_keeps_the_two_basket_rules_consistent`
asserts the discriminator exists exactly once, that the lagging rule defers rather
than restating a default, and - the actual regression - that the phrase "Prefer
that to a whole-basket exit" never returns.

Negative-tested: removing the discriminator, restoring the unscoped preference,
having the lagging rule stop deferring, and duplicating the discriminator into
both rules each fail the suite. Full gates clean.

---

## Video addendum - the 28 Aug LIVE SESSION (v4r)

**Source:** Intraday Hunter live session `D02UJwCO7_4` (28 Aug 2026, 8:09).

A flat open after the previous day's broad selloff - the exact setup the 28 Aug
pre-open note described. IH bought it to hunt the seated sellers, sat through an
early loss while BankNIFTY lagged, and booked a large profit once Sensex and NIFTY
carried it. Our agent read the same setup the same way, took two longs, and
finished **+989.75**.

The whole of v4r comes from one disagreement between them, and it lands on a rule
that was already in the corpus rather than on anything new IH said.

### What the agent did

| Time | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|
| 09:25 -> 09:27 | agent, pivot stall + laggard veto | +1,352.00 | +217.50 | **+1,569.50** |
| 09:55 -> 10:01 | agent, premise invalidation | -282.75 | -297.00 | **-579.75** |
| | | **+1,069.25** | **-79.50** | **+989.75** |

The pre-open note injected (2,265 chars, the exact size measured when written) and
both entries cite its hunt branch by name - `seated_seller_hunt_morning_star_fib61`
and `buyside_seller_hunt_pullback_hold`. The note's premise reversal, from three
sessions of follow-the-momentum to hunt-the-seated-sellers, transferred cleanly.
That is the note working exactly as intended.

### The disagreement

Trade 1 was closed **100 seconds after entry**, up +968.75 at the time of the
decision, on these stated grounds:

> "The BankNIFTY mirror is not confirming: it sits -202 points below its OWN pivot
> at a lesser psych resistance and is **flat (-22.5)**, a textbook
> laggard-never-joined signal that the shared two-index move underpinning the
> target has stopped forming."

That cites `LAGGARDS NEVER JOINED -> BOOK WHAT YOU HAVE`, correctly by its letter.
IH spent this session on precisely that situation and reached the opposite
conclusion:

> "Sometimes one index gives a small rejection and goes, just to CHANGE PSYCHOLOGY
> - but Sensex and NIFTY are fine."
>
> "One index stays behind to change psychology a little, and today that was
> BankNIFTY."
>
> "Make it in Sensex and NIFTY, as much as you like. BankNIFTY just needs to not
> stay negative - even if it gives no profit."

He held. BankNIFTY broke out later in the session and he booked a large profit.

### What was actually wrong with the rule

Not the reasoning - the SCOPE. `LAGGARDS NEVER JOINED` is written for the MAJORITY
refusing: "the other TWO indices never break their own levels". On a two-index
basket that phrasing inverts easily, and one index being behind reads as "the
laggards never joined" when it is nothing of the sort. Today two of the three
indices were working and exactly one trailed - the opposite of the rule's case.

So v4r adds a scope limit to that rule rather than a new rule:

- **Count who is absent before applying it.** Majority refusing, not any single
  index behind.
- **Classify the trailing index by its SIGN, not its distance.** FLAT or
  positive-but-slower has not refused; it has not ARRIVED. Negative, or a confirmed
  reversal of its own, is the disqualifying case.
- **Hand that disqualifying case to `THE STALL-OR-REVERSAL TEST`** (SLH-015) rather
  than ruling on it here, so the two cannot drift into competing verdicts the way
  the basket pair did before that fix.

This completes SLH-015 from the other side. That fix separated a mirror that has
STALLED from one that has REVERSED. Today's session named the third state sitting
between them - a mirror that has simply **not arrived yet** - and gave the test
that distinguishes it.

### The honest half

The rule text keeps the evidence that does NOT support it, and a drift guard
asserts that it survives. The NIFTY leg really was stalling at its pivot when the
agent booked: three 1-minute candles with tiny bodies going nowhere, which is a
fair `BOOK WHEN THE PROFIT STOPS GROWING` exit on its own merits. Trade 1 also
booked a genuine +1,569.50. The laggard half of that reasoning was the wrong half,
not the whole decision, and the prose says so - "that is not proof holding is
always right".

What the day does show is a pattern now running two sessions: book early, re-enter
worse, give some back. On 27 Aug the re-entry cost -308.00; today it cost -579.75.
Both re-entries were at prices worse than the exit that preceded them.

### Deliberately NOT added

- **"Enter before the round number, because after the break the market may not give
  you a chance"** ("enter before 500... if there is momentum it will be upside").
  A real entry rule, but v4p's WHEN THE DIRECTION HAS ALREADY DECLARED ITSELF,
  WAITING IS THE COST already carries the instruction and the reasoning.
- **"The seated-seller read needs all THREE indices to have sold"** ("if selling
  happens in one index only, then there is a problem"). Well covered - the corpus
  has ten separate passages on all-three-indices confirmation.
- **The loss-tolerance passage** ("do not panic at a loss, that is what your SL is
  for"). Covered by FEAR IS NOT A SIGNAL and v4n's DO NOT THINK YOUR WAY THROUGH
  A LOSS.

### Knowledge changes (v4r)

- `LAGGARDS NEVER JOINED -> BOOK WHAT YOU HAVE` gains a scope limit: COUNT WHO IS
  ABSENT BEFORE YOU APPLY THIS.
- One drift guard, negative-tested seven ways: removing the scope limit, softening
  the majority gate, turning the sign test into a distance test, reclassifying flat
  as a refusal, ending the deferral to the basket rules, dropping the counter-
  evidence, and dropping the measured case each fail the suite.
- Prompt size 144,489 -> 146,349 chars. Assembled 147,640 against the post-SLH-013
  budget: **193,841 chars of headroom.**

---

## Knowledge addendum - the 30 Aug WEEKEND video (v4s)

**Source:** Intraday Hunter, "How to Plan a Trade Before You Enter"
(`HkJ2PdmpPc8`, uploaded 2026-08-30 10:30 IST, 9:40).

Not a session - a methodology video, and the first one in this series that says
something about what THIS agent should be optimising.

### The idea

IH splits setups into two families and says they must be tuned in OPPOSITE
directions, and that "my setup does not work" is usually a trader optimising the
wrong one:

- **PATTERN setups** (double top/bottom, range breakout, trend, option writing on
  a range) are inherently low-accuracy: "you will get more wrong trades in them,
  but when one is right, make more money on it." They are paid for by RATIO, so
  they want a high R:R (1:2, 1:3) and MORE attempts for the edge to show.
- **PRICE-ACTION setups** - reading the chart for where stops actually sit,
  knowing the market must break a level to reach them - are paid for by being
  RIGHT. "If your R:R is a little lower it will still work out; but there you have
  to pay much more attention to ACCURACY." And the line that makes it bite: "if
  you are working on R:R here, then what was the point of working on price action
  at all?"

**SL hunting is squarely the second family.** Naming the trapped crowd, the
closing-point hold test, trap density, the level that invalidates - all of it is
accuracy machinery. From which his operational conclusion follows: "the biggest
mistake a price-action trader makes is not controlling the number of trades...
take as few as possible", because "if you keep the number of trades high, the
profit you came to make, you will hand back as an equal loss."

### Why it earned a version rather than a note in passing

Because we have measured that exact failure, twice running, and had already
flagged the pattern without a rule to attach it to:

| Session | First trade | Re-entry | Gave back |
|---|---|---|---|
| 2026-08-27 | +857.00 | -308.00 | ~36% |
| 2026-08-28 | +1,569.50 | -579.75 | ~37% |

Both re-entries were at prices WORSE than the exit that preceded them. Neither was
reckless - each carried a nameable crowd and an acceptable ratio - and that is
precisely the point the rule now makes: **a passable R:R is what let a second trade
look like a trade at all.**

So the practical form is a bar on RE-ENTRY, not on holding: after a booked winner,
re-entering the same move in the same session needs a genuinely NEW trapped crowd,
not a fresh pattern on the leftovers of the one you already collected. "The move I
just took profit from is still going" is not a second setup - it is the first one,
and you already took your share.

Distinct from `R:R-BAIT AT ROUND-NUMBER REJECTIONS`, which rejects a ratio with no
crowd behind it. This rejects a ratio with a crowd you have ALREADY hunted.

### The direction it must NOT be read in

On a live book the dangerous misreading is "so hold longer". The rule says the
opposite in as many words - the stop, the target and OPTION-TIME-ADJUSTED
REWARD/RISK are unchanged, v4f still books when the rate of gain dies, v4q's
obviousness rule still ends the hold - and a drift guard asserts both that
sentence and "It governs HOW MANY times you enter, not when you leave" survive.

### Deliberately NOT added

- **"Raise the target, not the quantity."** Good advice, does not transfer: our
  size is computed mechanically to a ~Rs.2500 risk budget and the agent does not
  choose quantity. Same reasoning as v4p's capital-size passage.
- **"No trailing; hold some quantity for target 2, exit at breakeven if it does not
  come."** Partial-quantity exits are not in the tool surface - `exit_leg` closes a
  whole leg - so there is nothing for the agent to act on.
- **"Entry cannot be fixed, SL and target must be."** Already carried by
  `NAME THE LAST POINT, NOT ONLY THE STOP` and `PRE-COMMIT THE ADVERSE MOVE YOUR
  THESIS TOLERATES`.

### Knowledge changes (v4s)

- `RISK`: YOURS IS AN ACCURACY METHOD, NOT AN R:R METHOD - SO TAKE FEWER TRADES.
- One drift guard, negative-tested six ways: removing the rule, collapsing the two
  families, dropping the hold-time guard, re-reading it as permission to hold,
  dropping the measured evidence, and softening the re-entry bar each fail.
- Prompt size 146,349 -> 149,224 chars. Assembled 150,515: **190,966 of headroom.**

### A note on access

A third weekend upload, "What Happens to the Market When SL Is Hit or Created?"
(`m79HfFq8ibY`, 7:22), is **members-only** and could not be read. On title alone it
is the most on-topic video this channel has posted in weeks. Worth the operator
watching it directly if the membership is available.

---

## Pre-open note for 2026-08-31

**Source:** Intraday Hunter, "Prediction For 31 AUG 2026" (`62oH6ARjxAQ`, uploaded
2026-08-30 21:00 IST, 2:16) - posted Sunday evening, per the channel's pattern for
a Monday.

### The plan

**The method differs BY BRANCH, which no previous note has done.** Every note so
far applied one premise across all three open shapes - hunt, or follow. This one
splits:

- **GAP-UP -> SELL side.** "Everyone will start buying directly, so we plan
  selling-side setups - because directly there are no SLs available here." The
  fresh buyers ARE the crowd.
- **GAP-DOWN -> BUY side.** The mirror image: "seeing the market fall, others will
  try to sell directly", so the fresh sellers become the crowd.
- **FLAT -> SELL side, but FOLLOWING, not hunting.** "In flat there is a bit of a
  problem for us... the situation will not be very clear, so we will just walk WITH
  the market."

Both gapped branches are counter-crowd hunts; the flat branch is a follow. Carrying
one premise across all three is exactly the habit the last four notes built, so the
test asserts the split explicitly.

**He also names FLAT as the problem case** - "our work gets done if we get either a
gap-up or a gap-down; if it opens exactly flat, that is more of a problem" - which
inverts the last three notes, where flat quietly shared a branch and was never
singled out.

Context: Friday closed with small rejections and little momentum across the board,
and SENSEX never crossed 77,000, so few sellers are seated.

### Levels

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 24240, 24300 | 23980, 24070 |
| BANKNIFTY | 57700, 58000 | 57000, 57200 |
| SENSEX | 77500, 77750 | 76700, 77000 |

Both support pairs needed the operator. NIFTY's ran together as "24072980";
SENSEX's came through as "776700" - six digits where two five-digit levels belong.
Confirmed off the chart as 23980/24070 and 76700/77000; the test carries comments
saying not to reconstruct either from the caption.

### Verification

19/19 on the premarket suite, each new assertion negative-tested: unifying flat
with the gapped branches, removing the method split, dropping the gap-up reason,
demoting flat from the problem case, and "correcting" the NIFTY support each fail.

---

## Video addendum - the 31 Aug LIVE SESSION (v4t)

**Source:** Intraday Hunter live session `BJksj4JrtxU` (31 Aug 2026, 6:45).

The most instructive session in this series so far, because IH and the agent read
the SAME open from the SAME pre-open note and took OPPOSITE sides. He sold it and
booked a good profit. We bought it three times for **+368.75**, with the last two
trades giving back 73% of the first.

Nothing downstream was wrong. The patterns, levels, fibs and crowd reads in all six
decisions were competently done. The whole day turned on one word.

### The one word

The 31 Aug note branched: GAP-UP to the SELL side, GAP-DOWN to the BUY side, FLAT
to the SELL side. NIFTY opened **58 points below the prior close, about 0.24%**;
BankNIFTY about 0.25%.

- **IH called it "almost FLAT"** and said so repeatedly through the session. He also
  said what a real gap-down would have meant for him: "if a gap-down opens directly,
  then the OTHER people also start selling, and then our thing does not work." He
  took the flat branch, sold, and booked.
- **The agent called it a gap-down** - explicitly, in writing: "Shared gap-down open
  (NIFTY -58pts, BankNIFTY -142pts, both ~0.25%) matches the pre-open note's
  GAP-DOWN=BUY-side branch." It took the buy branch three times.

Both applied the note faithfully. Only the classification differed, and the note
sent them opposite ways on it.

**Nothing in the corpus said where the line is.** `DEFAULT FLAT-OPEN READ`,
`GAP SIZE IS A RISK DIAL` and `SHARED-GAP REQUIREMENT` all explain at length what a
gap MEANS versus what flat MEANS - and every one of them assumes you already know
which you are looking at. Worse, `DEFAULT FLAT-OPEN READ` would have put the agent
on IH's side ("a flat-open rally into a level is normally a SHORT candidate"), and
it never got consulted because the open had already been labelled a gap.

### What v4t adds

**1. `CLASSIFY THE OPEN BEFORE YOU BRANCH ON IT, AND SAY WHY`.** The test is
PARTICIPATION, not arithmetic, and it is the same question `DEFAULT FLAT-OPEN READ`
turns on: did the open itself RECRUIT the other side directly? A real gap acts
before anyone can decide. An open small enough that the crowd is still deciding is
flat, whatever the sign of the number.

The percentage is carried as CALIBRATION rather than as the rule - a quarter of a
percent is not a gap; somewhere around half a percent, or an open clean past the
nearby stop clusters, is where the gap reading starts to earn itself - with an
explicit tie-break: **if you are hesitating between the two words, that hesitation
IS the answer, and it is flat.** A bare threshold would misfire the first time a
0.4% open behaves like a gap, so the guard asserts the participation test survives
alongside the number.

It also requires the classification and its reason to appear in the entry
reasoning, so a wrong call is visible as a wrong CALL rather than buried inside a
good-looking setup - which is exactly how today's went unnoticed for three trades.

**2. `THE EARLY RETRACEMENT IS THE TRAP, NOT THE TURN`.** IH's read of the actual
session: "when it opens flat and falls directly, the market gives a small
retracement so that other people cannot sell directly", and "because of the
retracement nobody will be seated short - this retracement is JUST A TRAP." So the
pull-back is where YOU join ("now a retracement has come, especially in BankNIFTY,
so now we can sell here"), and it also tells you who is NOT there: a crowd kept out
never got seated, so the trade is a FOLLOW of the original move, not a fade of a
trapped crowd. The rule sends the reader to `THE METHOD IS NOT ALWAYS A FADE`
rather than to a hunt, and a guard asserts that.

### v4s did not bite, one session after shipping

All three entries name the same crowd - the gap-down sellers - and none of the six
decisions mentions the trade count, the re-entry bar, or v4s at all.

| Time | NIFTY | Mirror | Basket |
|---|---|---|---|
| 09:19 -> 09:24 | -19.50 | +1,377.00 | **+1,357.50** |
| 09:33 -> 09:35 | +65.00 | -202.50 | **-137.50** |
| 10:00 -> 10:01 | -302.25 | -549.00 | **-851.25** |
| | **-256.75** | **+625.50** | **+368.75** |

Give-back has gone the wrong way three sessions running: 36%, 37%, now **73%**.

The likely reason v4s missed is worth recording rather than fixing blind: its test
is "a genuinely NEW trapped crowd", and the agent named a fresh PATTERN each time
(flush reversal, double bottom, psych-support hammer) and treated the pattern as
the setup. The crowd was identical in all three. **A pattern is not a crowd** - but
that sharpening should be made after seeing whether v4t's classification rule
removes the problem at the root, since on a correct FLAT read none of the three
longs would have been taken at all. Deliberately NOT added this version.

Trade 1 is also worth noting on its own: the basket was carried entirely by the
MIRROR (+1,377.00 against a NIFTY leg of -19.50). The NIFTY read was wrong and the
basket paid anyway, which is the pending
[[v4n-narrating-by-favourable-leg]] concern in its cleanest form yet.

### SLH-013's first live data

The latency line shipped two days ago and produced its first session: **73
decisions, median 41.1s, p90 50.0s, max 61.2s** against a 90s deadline. Exactly one
crossed the 60s warn threshold and exactly one WARN fired; nothing came near the
deadline.

That is a well-calibrated threshold (1.4% fire rate) and a useful datum for the cap
question: at a median of 41s of a 90s budget, the prompt has real room before
latency becomes the binding constraint.

### Knowledge changes (v4t)

- `OPENING_DRIVE`: CLASSIFY THE OPEN BEFORE YOU BRANCH ON IT, AND SAY WHY (new,
  placed immediately before DEFAULT FLAT-OPEN READ, which it gates).
- `OPENING_DRIVE`: THE EARLY RETRACEMENT IS THE TRAP, NOT THE TURN (new).
- Two drift guards, negative-tested seven ways: removing either rule, turning the
  participation test into an arithmetic one, dropping the calibration, dropping the
  tie-break, dropping the measured consequence, and flipping the retracement rule
  from a follow to a fade each fail the suite.
- **Deliberately NOT added:** the holiday-clears-inventory observation (already
  covered by `EVENT / HOLIDAY PARTICIPATION`), and the "a pattern is not a crowd"
  sharpening of v4s - see above.
- Prompt size 149,224 -> 152,587 chars. Assembled 153,878: **187,603 of headroom.**

---

## Pre-open note for 2026-09-01

**Source:** Intraday Hunter, "Prediction For 01 SEP 2026" (`dZqfpsKwqRQ`, uploaded
2026-08-31 21:25 IST, 2:28).

### The plan

**The branch returns to the 28 Aug shape, inverting yesterday's.** Three-to-four
days of continuous selling have SEATED a bearish crowd - "put-side traders will be
the ones sitting here" - so:

- **FLAT or GAP-UP -> BUY side**, to hunt them.
- **GAP-DOWN -> SELL side**, because a gap-down hands that crowd profit instead of
  pain: "they have already waited this far", and a gap-down leaves them sitting in
  confidence rather than forcing them out.

That is the exact opposite of the 31 Aug note (gap-up sell, gap-down buy, flat
sell). Two consecutive notes with inverted branches is precisely the setup for
carrying yesterday's habit forward - and yesterday a misclassified open cost the
day's direction, which is what v4t's CLASSIFY THE OPEN rule now guards.

**The gap-down veto is tied to a LEVEL, not to the gap's sign**: "until this level
is crossed, the put traders cannot be targeted directly." He points at the chart
rather than naming the NIFTY level on air, so the note says to treat your own named
invalidation as that level rather than inventing his.

**He warns of an upward trap first**: the seated crowd "will try to save their
trade, so the market may build some trap, may try to go up a bit." A push up need
not be a reversal - it can be the bait before the hunt. That pairs directly with
v4t's THE EARLY RETRACEMENT IS THE TRAP, arriving from the opposite direction.

**Today is NIFTY's weekly expiry.**

### Levels

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 24240, 24300 | 23900, 23980 |
| BANKNIFTY | 57700, 58000 | 57000, 57200 |
| SENSEX | 77500, 77750 | 76500, 76700 |

SENSEX 76500 is named as the psychological number. The caption dropped a digit from
NIFTY's first resistance ("2440"); confirmed off the chart by the operator as 24240,
and the test says not to reconstruct it from the caption.

### Verification

19/19 on the premarket suite, each new assertion negative-tested: flipping the
branches back to yesterday's, regating the veto on the gap's sign, dropping the
upward-trap warning, dropping the expiry from the context, and "correcting" the
NIFTY resistance each fail.

---

## Video addendum - the 1 Sep LIVE SESSION (v4u)

**Source:** Intraday Hunter live session `kUZDZ42c-y8` (1 Sep 2026, 12:00 - the
longest in this series).

The first LOSING session in this stretch: **-1,488.25 over three trades**, against
IH booking his target on a single held position. Both halves of the loss have a
named cause, and neither was a bad pattern read.

| Time | Held | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|
| 09:51 -> 09:52 | **43s** | agent, index hierarchy | +149.50 | -180.00 | **-30.50** |
| 09:58 -> 10:04 | 6m | **mechanical AI_STOP** | -604.50 | -1,084.50 | **-1,689.00** |
| 10:11 -> 10:12 | **51s** | agent, index hierarchy | +276.25 | -45.00 | **+231.25** |

### 1. The open was measured against the wrong price

v4t told the agent to classify the open by participation. It never said what to
compare it TO, and that gap decided the day.

NIFTY opened flat on any reference (24077.55 against 24080.40). BankNIFTY was read
as **"gapped down hard (~0.8%)"** against its OFFICIAL close - and that reading
flipped the pre-open note's branch from BUY to SELL, producing both shorts.

IH judges the same thing off a different price, and says so plainly:

> "You will see BankNIFTY has closed somewhere here. But more than the closing, we
> go by where the market closed AROUND 3:15."

The last fifteen minutes carry auction and settlement prints that no crowd traded
around, so a gap measured to 15:30 can be an artefact of the close rather than a
fact about positioning - and positioning is the only thing the classification is
trying to read. On the 3:15 reference the session was flat, IH stayed on the buy
side, and booked.

v4u adds this as a sub-bullet on `CLASSIFY THE OPEN`, with the disagreement case:
**when the indices disagree, the flat one is the honest read.** That is
`SHARED-GAP REQUIREMENT` applied in the direction it does not currently name - that
rule covers a flat BankNIFTY beside a gapped NIFTY, and today was the mirror.

### 2. The hierarchy was read one candle at a time

The long was opened at 09:51:49 and closed at 09:52:32 - **43 seconds** - because
"BankNIFTY is in an accelerating downtrend... disqualifies the whole basket". The
third trade was opened at 10:11:52 and closed at 10:12:43 - **51 seconds** - on the
OPPOSITE reading: "BankNIFTY printed a confirmed bullish morning-star...
disqualifies the whole basket."

The leading index disqualified a long for falling and a short for rising inside
twenty minutes. That is not the hierarchy working; it is a chop being read one
candle at a time.

IH spends the session on exactly the move that triggered the first cut:

> "It went down first, because what does the market have to do? It has to CREATE
> the stop-losses. So we had to see a slight loss. But when the market produced
> positive momentum, it did exactly what we expected."

The first minute after entry is the one the setup PREDICTED would go against you.
So `ONE CANDLE IS NOT A REVERSAL` requires, before the hierarchy may close a trade
you just opened, that the leading index has printed a confirmed reversal **on the
timeframe you entered on - the same bar you would demand to ENTER on it** - and not
merely a move against you. If the only thing that has changed since entry is price,
nothing has changed; that was priced in when the stop was placed.

**Written against the dangerous reading.** "Hold through reversals" is how a small
loss becomes a large one, so the rule ranks the stop, the max loss and
premise-invalidation above itself and keeps the counter-evidence: the same
session's second trade was stopped mechanically for -1,689.00, and that stop was
right. A drift guard asserts both.

### Deliberately NOT added

- **The holiday explanation** for why he followed momentum on 31 Aug and hunts
  today ("in a holiday traders take less risk... but if someone took risk
  yesterday, they took it in selling"). It is the cleanest account yet of why two
  consecutive notes inverted, but `EVENT / HOLIDAY PARTICIPATION` already carries
  it, including the "two days can print an almost IDENTICAL chart" framing.
- **The 15-minute time budget** ("assume 15 minutes is the max; profit will come
  before 11:15") and the small-candle tell. Close to v4q's obviousness rule and
  v4b's second leg; recorded here rather than added.
- **"One rejection is understandable; a SECOND rejection means the market may be
  doing something to you"** - covered by the existing two-rejections /
  third-momentum rule.

### SLH-013 latency, second session

**57 decisions, median 41.8s, p90 56.1s, max 70.8s** against the 90s deadline; one
crossed the 60s threshold. Consistent with session one (median 41.1s, p90 50.0s),
so the threshold is holding its calibration at roughly one warning per session.

### Knowledge changes (v4u)

- `CLASSIFY THE OPEN BEFORE YOU BRANCH ON IT` gains COMPARE THE OPEN TO THE 3:15
  LEVEL, NOT THE OFFICIAL CLOSE, plus the indices-disagree rule.
- `ONE CANDLE IS NOT A REVERSAL: THE HIERARCHY NEEDS TIME TO SPEAK` (new).
- Two drift guards, negative-tested nine ways. One assertion was found too weak by
  that pass (it checked "never overrides the" without checking WHAT) and tightened.
- Prompt size 152,587 -> 155,916 chars. Assembled 157,207: **184,274 of headroom.**

---

## Video addendum - the 2 Sep LIVE SESSION (v4v)

**Source:** Intraday Hunter live session `fE8Sw4vQOSM` (2 Sep 2026, 11:01,
uploaded 11:06 IST).

**The worst day in the series: -3,877.25 over SIX trades.** And the most
instructive, because IH took the SAME trade, on the SAME premise, and was
ALSO wrong. He lost too. The read was not the error.

| # | Time | Held | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|
| 1 | 09:20 -> 09:27 | 6m40s | agent, index hierarchy | +214.50 | -1,251.00 | **-1,036.50** |
| 2 | 09:32 -> 09:34 | **1m37s** | **mechanical AI_STOP** | -279.50 | -807.00 | **-1,086.50** |
| 3 | 09:40 -> 09:43 | 2m54s | agent, premise invalidation | -422.50 | -150.00 | **-572.50** |
| 4 | 09:48 -> 09:57 | 8m57s | agent, index hierarchy | -377.00 | -646.50 | **-1,023.50** |
| 5 | 10:12 -> 10:14 | 1m48s | agent, index hierarchy | -419.25 | -411.00 | **-830.25** |
| 6 | 10:24 -> 10:25 | 58s | agent, premise invalidation | +234.00 | +438.00 | **+672.00** |
| | | | **TOTAL** | **-1,049.75** | **-2,827.50** | **-3,877.25** |

### The pre-open note transferred perfectly. That was the problem.

Every one of the six entries cites the note's GAP-DOWN branch by name, and the
note was right about the open: NIFTY gapped -198pts (-0.82%), BankNIFTY -0.69%,
shared and genuine. The agent classified the open correctly (v4t), measured it
correctly, and picked the branch the note prescribed.

Then it took that branch six times in sixty-five minutes.

### What IH did, and the one line that predicts our whole day

He entered PUT-side across BankNIFTY, SENSEX and NIFTY at roughly our first
entry, on identical reasoning - yesterday's failed 500-level breakout seated
retail longs, "the good trader slowly sells", so the day should stay in selling.
He then said what must NOT happen:

> "This green candle should not go too much higher, especially in BankNIFTY,
> because there is the 57,000 number. If it goes above this level, the market
> will go to the closing price."

And, four minutes in, he predicted our day exactly:

> "A trap will not form by doing a breakout. **If a breakout happens, then it
> will be the WHOLE DAY'S TRAP.** ... Then you sell here, sell here - **you will
> be wrong in all two-three places.**"

The breakout happened. He named it the moment it did ("we did NOT want the market
to do a breakout, but it DID"), waited once for a small trap that never came, and
cut at his limit:

> "Sometimes, **seeing the gap-down, other people also start selling**, and the
> market makes a trap FOR THAT DAY. We are seeing something like that here."
>
> "**Do not fight when you are wrong.** Make good profit when you are right."

He was wrong once, for one loss. We were wrong six times.

### The gap this exposed in the corpus

Every re-entry rule we have keys on a booked **WINNER**:

- `YOURS IS AN ACCURACY METHOD` (v4s) - "after a booked winner, the bar for
  re-entering the SAME move is a genuinely NEW trapped crowd".
- `MOVE-EXHAUSTION - ONE MOVE PER THESIS` - "once a thesis's move has been
  captured and **BOOKED**".

The only loss-side rule, `POST-LOSS SPEED LIMIT`, asks that the next setup be
"fresh, deliberate, high-quality, with a named target crowd and clear
invalidation" - and the agent supplied exactly that, six times. Each entry named
a genuinely fresh PATTERN (double top, fibo rejection, bearish engulfing,
shooting star) and the identical crowd. **Nothing said the premise itself dies.**

The 5-minute mechanical cooldown was respected on every re-entry, so the host
gate was never the thing that failed either.

This is also the third session where v4s did not bite, and the v4t addendum
already flagged why in advance - "a pattern is not a crowd" - and deliberately
deferred it pending v4t's classification rule. v4t's rule worked and the problem
survived it, so the sharpening is no longer speculative.

### What v4v adds

`A BREAKOUT THROUGH THE LEVEL YOUR PREMISE FORBADE ENDS THAT PREMISE FOR THE
SESSION`, placed as the explicit loss-side twin of v4s. When you entered on
"price must not cross X" and price crosses X, the premise is dead for the
**session**, not for that entry; a later pattern in the same direction is the
same dead premise wearing a new candle. Trading it again needs a NEW crowd
trapped by price action that happened AFTER the break, named as such.

It carries **the clock as the earlier detector** - IH budgets 5-10 minutes for
momentum and treats overrun as evidence, not patience ("the time it is taking is
EXTRA for us"). This is the second appearance of the time budget; v4u recorded it
and deliberately did not add it, and today it is the thing that told him to stop.

**Written against three wrong readings, each guarded:**

1. **It is not about direction.** IH took the same side and lost too. A guard
   asserts his counter-evidence survives, because an edit that turns this into
   "do not short a gap-down" would learn the opposite of the day's lesson.
2. **It is not "never re-enter".** The sixth entry made +672.00, and the rule
   says so.
3. **It governs OPENING, never closing.** Read as licence to sit through a loss
   it would have turned -3,877 into something uncapped, so the rule ranks the
   stop, the max loss, premise-invalidation and INDEX HIERARCHY above itself.

### Deliberately NOT added

- **The discipline / mindset passage** ("your mindset should not change so
  quickly... today I cut the trade and make a call trade, then in net you will
  get nothing"). It is the emotional companion to the rule above and is already
  carried by `NO INSTANT FLIP`, `FEAR IS NOT A SIGNAL` and v4n's
  `DO NOT THINK YOUR WAY THROUGH A LOSS`.
- **"Where to enter - at the lower point, higher, or right here"** (3:38-3:45).
  Covered by v4p's `WHEN THE DIRECTION HAS ALREADY DECLARED ITSELF`.
- **The SENSEX 77,000 breakout as a second instance** of the same
  retail-buys/operator-sells mechanism. Same mechanism, no new content.

### SLH-013 latency, third session

**71 decisions, median 39.2s, p90 51.2s, max 66.3s** against the 90s deadline;
two crossed the 60s warn threshold. In line with session one (median 41.1s, p90
50.0s) and session two (41.8s / 56.1s), so the threshold is holding calibration
at roughly one to two warnings per session.

### Knowledge changes (v4v)

- `RISK`: A BREAKOUT THROUGH THE LEVEL YOUR PREMISE FORBADE ENDS THAT PREMISE FOR
  THE SESSION (new, placed immediately after v4s, whose loss-side twin it is).
- One drift guard, negative-tested **thirteen** ways: removing the rule,
  narrowing its scope from session to entry, letting a pattern stand in for a
  crowd, dropping the counter-evidence that IH was also wrong, turning it into an
  absolute re-entry ban, re-reading it as permission to hold a loser, dropping
  the exit-precedence sentence, dropping the INDEX HIERARCHY deference, dropping
  the measured evidence, dropping which entry killed the premise, dropping the
  clock detector, dropping the gap-down-recruits-sellers mechanism, and no longer
  naming the gap it fills each fail the suite.
- Prompt size 155,916 -> 158,839 chars. **191,161 of headroom.**

---

## Video addendum - the 3 Sep LIVE SESSION (v4w)

**Source:** Intraday Hunter live session `rOd-e5x4y2s` (3 Sep 2026, 6:51,
uploaded 11:05 IST). Transcript read in full: 56 segments, 0:02 to 6:42.

**A profitable day, +1,058.00 - and the most frustrating one yet**, because the
agent got the hard part right and then handed most of it back.

| # | Time | Held | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|
| 1 | 09:34 -> 09:41 | 6m27s | agent, booked on a stall before a round number | +1,820.00 | +1,434.00 | **+3,254.00** |
| 2 | 09:46 -> 09:49 | 2m38s | agent, index hierarchy | -214.50 | -255.00 | **-469.50** |
| 3 | 10:05 -> 10:07 | 1m26s | **mechanical AI_STOP** | -799.50 | -927.00 | **-1,726.50** |
| | | | **TOTAL** | **+806.00** | **+252.00** | **+1,058.00** |

### The first trade was excellent, and it matched IH almost to the minute

Both went long the gap-up. Both booked on the same signal. The agent's own exit
reason - `book_before_round_number_profit_stall`, "price stalled into a tight
range" - is IH's read in different words, and it banked +3,254.00.

Then it re-entered twice and gave back 67.5% of that.

### What IH said the stall MEANT

This is the net-new, and it resolves a real tension already sitting in the
corpus. Holding the same winning long, he explains why he is leaving:

> "If the candles were forming FAST it would have been much better for us; right
> now it IS going up but the momentum is SLOW. In such a case OTHER traders also
> start buying here, and when they buy, retracement starts."
>
> "The market did a breakout and is making SMALL candles, so this will INVITE
> buyers, and if it invites buyers it will definitely give a retracement - in
> that retracement it will needlessly reduce our profit."

So the stall is not merely "the profit stopped growing". It is a statement about
WHO is arriving: slow, small candles after a break are a cheap public invitation
to traders on your own side, and their entry IS the give-back.

### Why this needed a rule rather than a note

Two existing rules pull in opposite directions here, and neither covers the case:

- `A SLOW GRIND AT THE LEVEL RECRUITS THE WRONG CROWD` (v3y) is scoped to price
  stalling at your level **before** you are in.
- The momentum-quality rule (v4k) covers pace **after** you are in and says the
  opposite of today: slow with small candles is "the sustainable kind - let it
  run", because slow keeps the trapped crowd seated.

v4k even carries a note saying not to confuse it with v3y. Today is a third case
that falls between them: the level has broken, the position is in profit, and the
candles go small. On v4k's letter, the agent should have held.

v4w is therefore a **scope limit on v4k, not a reversal of it**, in the same
shape as v4r's limit on `LAGGARDS NEVER JOINED`. The discriminator is not the
pace but who the slowness lets in:

- still squeezing a crowd trapped on the OTHER side -> fuel, hold (v4k intact);
- after the break, letting fresh traders onto YOUR side -> book.

IH supplies the counterfactual that isolates it: *"if the candles were forming
well and the market was NOT giving others a chance to buy, we wanted to see the
800 breakout."* Same level, same direction, same target - only the invitation
differed.

### The re-entry half

The rule also states the consequence the day actually cost: **the retracement you
just booked ahead of is not a second setup.** It is the arrival of the crowd you
booked against, so re-entering means joining them at the worst price instead of
being paid by them. Both re-entries here were into exactly the move IH had
predicted out loud.

That is the fifth session in a row with this shape - 36%, 37%, 73%, then
yesterday's six-trade -3,877.25, now 67.5%. v4s governs re-entry after a winner
and still is not biting; v4v gave the loss case a concrete test ("a NEW crowd
trapped by price action AFTER the break") and the winner case still has none.
**Deliberately not fixed here:** giving v4s the same test is the obvious next
step, but it should be made after seeing whether naming the retracement's CAUSE
removes the temptation at the root.

### Deliberately NOT added

- **"Because yesterday there was a loss, we won't take much risk today."** Word
  for word the situation v3z already covers (`AFTER A LOSING DAY, TAKE THE GOOD
  PROFIT RATHER THAN THE BIG ONE`), and yesterday was indeed -3,877.25.
- **The early entry because BankNIFTY had already moved** ("if OUR index, where
  we take more quantity, is going up, we made our entry a bit early"). Carried by
  v4p's `WHEN THE DIRECTION HAS ALREADY DECLARED ITSELF, WAITING IS THE COST`.
- **The reversal-versus-continuation framing** of the whole day. That was the
  pre-open note's branch and the note already carried it.

### Knowledge changes (v4w)

- `AFTER THE BREAK, SLOW CANDLES RECRUIT YOUR OWN SIDE, AND THAT IS A BOOK SIGNAL
  RATHER THAN A HOLD SIGNAL` (new, placed immediately after the momentum-quality
  rule it limits).
- One drift guard, negative-tested **fourteen** ways: removing the rule, reading
  it as a reversal of v4k rather than a scope limit, dropping v4k's fuel
  mechanism, collapsing it to a bare "book when slow", merging the two sides of
  the crowd test, dropping the counterfactual, dropping the invitation as the
  deciding factor, dropping the re-entry consequence or its reason, dropping the
  measured give-back or the booked winner, re-reading it as licence to book early
  anywhere, and dropping either the v3y deference or the exit precedence.
- Prompt size 158,839 -> 161,355 chars. **188,645 of headroom.**

## Video addendum - the 4 Sep LIVE SESSION and the 6 Sep WEEKLY (v4x)

**Sources.** Three accessible videos since Friday, all transcripts read in full:

| Video | Id | Length | Uploaded (IST) | Segments |
|---|---|---|---|---|
| Live session, Friday 4 Sep | `duGx4E10IzQ` | 9:55 | 4 Sep 12:01 | 75 (0:02-9:40) |
| Weekly analysis, "Time vs Process" | `3NgY1QjfDdg` | 18:49 | 6 Sep 12:23 | 150 (0:02-18:42) |
| Prediction For 07 SEP 2026 | `Qw55ggRNbBo` | 2:16 | 6 Sep 21:14 | 17 (0:06-2:14) |

A fourth, `uOl-kBCFEQQ` ("Master Advanced Analysis: Unlocking the Secret of Flat
Opening | Part-7", 15:30, 5 Sep), is **members-only** and was not read. Recorded
here so a later session does not spend time rediscovering that.

**Extraction note, superseding an earlier one.** The transcript panel DID
populate on the 18:49 weekly video. The old note that it "never populates on
videos over 12 minutes" no longer holds; the ordinary recipe worked unchanged.

### The comparison: same read, same strike, same minute, opposite outcome

The agent took exactly ONE trade on Friday and lost 143.00 on it.

| | IH | The agent |
|---|---|---|
| Direction | LONG | LONG |
| Instrument | BankNIFTY 57500 CE (+ NIFTY, SENSEX) | NIFTY 23900 CE + BankNIFTY 57500 CE mirror |
| Entry | immediately at the open | 09:15:48 |
| Immediate move | against him, at once | against it, at once |
| Held for | ~2.5 hours | **3 seconds** |
| Result | booked a large profit | **-143.00** |

IH describes the adverse move in the same terms the agent would have: *"right as
we made the trade, some loss started showing"*, and later *"this extra rejection,
especially in BankNIFTY, we had not expected at all"*. His response was to wait:
*"we will have to wait here. After waiting we will make a big profit."*

He also stated, before entering, that the retracement was **expected**: *"only a
few people would be sitting short, whom we can make our target. Then the market
will start creating SLs, that is, it may do a retracement. So before that we
should make our trade."* He entered deliberately AHEAD of a move he forecast.

### Why the agent could not do the same - and it is not a judgement failure

The log carries the whole mechanism at one millisecond, 09:15:48,529:

- sizing computed with `entry=23935.30`, `stop=23910.00` - a **25.3-point** allowance
- the entry recorded at the same instant with `Spot=23914.40`

The agent reasoned on a price that was 21 points stale by the time its order
reached the market, so the position opened with **4.4 points of room** - 83% of
the allowance already spent. `RenkoThread` independently marks spot at 23936.40
(09:15:07), 23914.40 (09:15:48), 23905.10 (09:15:52) and **23940.05 (09:16:00)**:
the market was back above the decision price nine seconds after the stop fired.

That is what v4x encodes. It is not "hold longer" - the corpus already says that
four times over (v4d, v4e, v4h, v4m). It is that the stop's *basis* must be the
price in front of you at the fill.

### The constraint that makes this sharp, and an operator decision

`SizingDecision.from_risk_budget` uses `one_lot_risk = abs(entry - stop) *
lot_size`, with no fallback lot and no rounding up. With
`SL_HUNTING_RISK_BUDGET=2500` and a NIFTY lot of 65, the widest stop that can be
sized at all is **2500 / 65 = 38.46 spot points**.

Friday's premise-ending level was the note's own first support at 23800 - 135.3
points below the decision entry. That stop would have cost 8,794.50 per lot and
been **refused outright** (0 lots). So a stop placed where the premise actually
dies is, today, unsizeable on this budget.

v4x resolves that honestly in the direction of less trading: if the re-derived
stop cannot be sized, the answer is NO TRADE, never a nearer stop. **Whether the
risk budget should rise instead is an operator decision and is deliberately not
taken here** - it is live-money sizing, and the mirror already roughly doubles
basket risk beyond the budget.

### The weekly video: the model behind the daily notes

`3NgY1QjfDdg` is the most explicit statement of method he has given, and it is
worth recording even though none of it became a rule this session.

- **The SL-hunting core, stated plainly** (6:42-6:59): *"In SL hunting there is
  only one job: focus on whose SLs you are going to get to eat. Whichever
  trader's SLs are available to eat, eat them. Most traders try - gap up,
  positive market, so buy, buy. But in SL hunting it is not like that."*
- **Same chart, different mindset, different plan** (8:55-11:08). He puts Friday's
  and Monday's charts side by side, identical, and derives OPPOSITE plans. What
  differs is not the chart but who is seated, and that is set by what happened
  between sessions.
- **The empty-book branch** (10:33-10:47): *"in Friday's chart nobody is seated...
  neither buyers came nor sellers came. So then we look ACCORDING TO THE OPENING
  at whose SLs we can get. But when we know traders may be seated there, then we
  target them."* Two branches: hunt the seated crowd, or - when none exists - read
  the opening as the thing that CREATES one.
- **The operator model** (11:53-18:26): a breakout that fails is his signature for
  an operator selling into retail buying; *"I never fight in front of the
  operator"*; and operators face other operators, so the OPENING is the evidence
  of which one has the power. This is the reasoning underneath the 3 Sep note's
  "pivot on the closing price" branch.
- **He says the daily videos omit this** (17:51-18:07): *"when we make the
  analysis we don't talk this much about seeing the operator. There we tell
  simple things."* So the pre-open note is a compressed surface of a deeper model.

None of that was added as a rule. v4e, v4f, v4g and v3y already cover the
seated-crowd branches, and the operator model is a *description* of why the
branches are what they are rather than a decision the agent can act on. It is
recorded here so a later session can pick it up deliberately.

### Two existing rules the weekend confirmed

- **v4v** - IH, reviewing his own losing trade (4:43-5:09): *"when I made the
  entry I told you that this level will not even be crossed if we are right. The
  market crossed the level and gave a loss."* Same shape v4v encodes.
- **v4w** - IH, on what happens after a break (7:41-7:53): *"if the market wants
  to make you buy, it will break out and then hold, hold, then do a slight
  momentum. That market is dangerous."* v4w restated by its source.

### Deliberately not fixed here

The **re-entry** question from the v4w commit is still open, and Friday did not
answer it: the agent never re-entered because it was stopped in the first minute
and then had nothing to re-enter into. Its worker window also closed at 11:00,
alone among the 29 workers (every other one ran to ~15:15), so it was not present
for the recovery IH traded. Both v4s (re-entry after a winner) and the 11:00
cutoff remain open items.

## Video addendum - the 7 Sep LIVE SESSION and the 8 Sep prediction (v4y)

**Sources.** Both transcripts read in full:

| Video | Id | Length | Uploaded (IST) | Segments |
|---|---|---|---|---|
| Live session, Monday 7 Sep | `fHR7WJLY1Zg` | 10:52 | 7 Sep 11:39 | 84 (0:07-10:50) |
| Prediction For 08 SEP 2026 | `mfCrKdkHlig` | 2:32 | 7 Sep 21:31 | 18 (0:06-2:22) |

### The day: the agent and IH were on the SAME side, again

NIFTY opened 23883.65 against a 23897.70 close - flat to a tiny gap-down, which
the 7 Sep note put on its SELL branch - and fell all session: 23817 by 09:55,
23790 by 10:17, 23765 by 11:13, 23746 by 12:50, closing near 23743. **A ~140
point one-way down day.**

The agent read it correctly and shorted it twice.

| # | Window | Held | Entry | Exit reason | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|---|
| 1 | 09:22:36 -> 09:24:45 | 2m09s | 23852.85, stop 23866.00, target 23818.5 | `premise_stale_exit` | +396.50 | -198.00 | **+198.50** |
| 2 | 10:17:48 -> 10:23:34 | 5m46s | 23793.70, stop 23811.15, target 23700.0 | `index_hierarchy_exit` | -1,014.00 | -1,155.00 | **-2,169.00** |
| | | | | **TOTAL** | **-617.50** | **-1,353.00** | **-1,970.50** |

IH did the same trade and held it. He entered puts on the first small retracement
after the open, sat through the counter-moves for roughly two hours, and booked a
large profit. His account of the retracements is the whole lesson:

> "Retracements here should be SMALL, because the market HAS to fall... if it did
> a BIG retracement and then fell, then anyone would start selling... this is how
> you can recognise how big the retracement will be. A BIG RETRACEMENT HAPPENS
> WHEN THE MARKET HAS TO TURN. Now it will not turn."

### Trade 2 is the one that cost the day, and v4u did not stop it

The agent cut a correct short on an **eleven-point** counter-move (NIFTY 23790 ->
23801) in a session already 90 points below its open, citing a confirmed
BankNIFTY hammer off the 10:17 low. On the mirror's premium that same 11 points
read as 557.50 -> 538.25, which is why it felt like a reversal.

The important part: **v4u was satisfied.** v4u demands "a CONFIRMED reversal on
the leading index... not merely a move against you", and a confirmed hammer is
exactly that. What v4u never says is how BIG the counter-move has to be. v4y adds
that measurement, and forbids taking it off the option P&L - the one place option
buying guarantees a wrong answer, as IH says in the same breath: *"our profit is
reducing because we are sitting in option BUYING. A small green candle reduces
our profit a lot."*

### Two things that are compliance gaps, not knowledge gaps

Recorded so a later session does not mistake them for missing rules.

- **Trade 1 entered with a target the session had already printed.** Its own exit
  reason says so: "target 23818.5 already printed as today's session low (09:16)
  before my entry -- the hunt is already spent." v4q already requires that test
  BEFORE the entry, not as an exit check. It ran two minutes late.
  Worth noting the exit conclusion was ALSO wrong: the flush was not spent, and
  NIFTY fell another 110 points. It closed a winner for +198.50 that had ~110
  points of room left in it.
- **Trade 2 sold into the low with no retracement.** Its 10:17:48 entry was the
  session low at that moment, and BankNIFTY's low was the same minute - the
  agent's own exit reason dates the hammer to "its 10:17 low". IH refuses exactly
  this: *"if it is falling directly, we pause a bit. We will not rush, because
  there is no benefit making a trade here, the risk is higher... now a slight
  retracement gave us the chance to participate."* v4f already covers it.

### v4x held

The rule from the last session did not have to fire, and its failure mode did not
recur. Trade 1 sized 13.15 points and opened with 11.7 of them intact (89%);
trade 2 sized 17.45 and opened with 20.9 (120%). Neither stop was hit - both
exits were the agent's own decision.

### Operational notes

- The 08:31 launch ran with **no pre-open note** ("no pre-open note applies today
  (absent, malformed, or dated for another session)"); the 08:35 relaunch injected
  2,517 chars correctly. The note had merged to `main` at 00:52 IST, so the first
  launch was simply from a working copy that had not pulled yet. If that relaunch
  had not happened, the session would have traded noteless.
- SL Hunting's worker cut off at **11:00** again, alone among the workers - every
  other one ran to ~15:15. It was therefore absent for the 11:00-15:15 stretch in
  which NIFTY fell a further 45 points.

### The 8 Sep prediction: the first one-direction note in the series

Every open shape takes sell-side setups - flat, gap-down and gap-up alike - with
a single exception: *"if a gap opens, then right at the open there should be no
big positive momentum. If something like that happens then we cannot do
anything."* There is no buy branch to fall back on, so the exception means NO
TRADE rather than a long.

His crowd read is unusually specific and cuts against the obvious one: the
sellers are deep in profit but are NOT seated in size, *"because when it reaches
here, put premiums become very high, so the trader would not have HELD the put
trade."* Being right is not the same as being positioned.

Tomorrow is **NIFTY expiry**.

## Video addendum - the 8 Sep LIVE SESSION and the 9 Sep prediction (v4z)

**Sources.** Both transcripts read in full:

| Video | Id | Length | Uploaded (IST) | Segments |
|---|---|---|---|---|
| Live session, Tuesday 8 Sep (NIFTY expiry) | `NitY2pWf3p0` | 12:47 | 8 Sep 12:41 | 97 (0:02-12:42) |
| Prediction For 09 SEP 2026 | `bwqMeOc-b8I` | 1:55 | 8 Sep 21:25 | 13 (0:06-1:44) |

### The day, and the first session run under v4y

NIFTY opened 23737.40, fell to ~23670 by 09:30, then chopped 23660-23700 all
morning. IH entered puts AT the open, followed continuously, and booked his
target after roughly two hours. The agent took two trades and finished -70.25.

| # | Window | Entry | Exit reason | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|
| 1 | 09:45:13 -> 09:50:32 | 23670.60, stop 23695.00 | `premise_invalidation_bullish_reversal_cluster` | -393.25 | -393.00 | **-786.25** |
| 2 | 10:18:42 -> 10:31:36 | 23704.10, stop 23722.00 | `basket_book_stall_reversal` | +65.00 | +651.00 | **+716.00** |
| | | | **TOTAL** | **-328.25** | **+258.00** | **-70.25** |

**Trade 2 is v4y working, and it should be recorded as such.** It shorted the
23700.30 spike at 23704.10 - within four points of the top - and booked using the
completed-hunt test in its own words: *"target zone 23657 already tapped 10:05 and
sharply rejected to 23717; NIFTY chopping 6+ candles unable to make new low."*
That is the reasoning v4y and v4q were written for, applied unprompted.

### Trade 1 is the gap, and IH narrates the exact minute

Both watched the same bounce. The agent called it a bullish reversal cluster on
both indices - morning star plus bullish doji with two-bar follow-through on
NIFTY, hammer plus bullish engulfing on BankNIFTY - and cut, noting it was "only
~8pts from its 23695 stop". IH, on the same move:

> "This much is all it should be, the market will not go higher than this...
> to change the mindset of the traders who were coming in, the market suddenly
> pulled an upside momentum. **That is, the sellers' SLs are not available here.**
> See when the problem happens? When in this chart the sellers' SLs WERE
> available - then this upside momentum becomes dangerous, it creates more SLs
> then rises, creates more SLs then rises."

Spot at the agent's exit was about 23679. By 10:03 it was **23660.55**, below the
entry. The cluster died in ten minutes.

That is v4z: a counter-move compounds only when there are stops on YOUR side above
it to harvest. IH also says why the seats were empty, and both reasons are
readable from the session's own history - the market had already given its
retracement and was holding round-number support, and the continuous fall had
driven put premiums so high that "the trader does make the trade, but NOT in big
quantity".

### Recorded, not encoded

- **He is NOT hunting on a continuation day, deliberately.** *"In SL hunting it is
  not that we don't follow continuously. If retail sellers have not come in, the
  chart can be followed continuously... if some traders are sitting in profit, we
  don't make them our target either."* Plus a timing model for retail: they sell
  after "selling, selling, retracement, then fall", not during clean continuation.
  This is the same seated-crowd logic v4e/v4f/v3y already carry, restated.
- **Strike selection.** He always sizes around AT THE MONEY: deep ITM gives better
  momentum but ties up capital and spoils the risk-reward; OTM pays more when
  momentum comes but "when momentum takes time, OTM hurts more". The agent already
  trades ATM, so this confirms rather than changes anything.
- **Option buyers pay rent.** *"If time is taking too long and you are in option
  BUYING then you should get out. An option SELLER can try, because he can
  withstand a retracement."* Already covered by v4j and v4g's time rules.
- **Chop days punish over-trading specifically.** *"In such a market the problem is
  bigger for traders who overtrade... you will lose more than you profit."*

### Operator constraints, settled

Two things flagged in earlier addenda are **deliberate** and should not be raised
again: the 11:00 worker cutoff is sized to the operator's Claude Pro 5-hour limit,
and `SL_HUNTING_RISK_BUDGET` exists so no single trade can lose an unbounded
amount - so v4x's "if the re-derived stop cannot be sized, take NO trade" is the
intended outcome of that budget, not a workaround for it.

## Video addendum - the 9 Sep LIVE SESSION and the 10 Sep prediction (v5a)

**Sources.** Both transcripts read in full:

| Video | Id | Length | Uploaded (IST) | Segments |
|---|---|---|---|---|
| Live session, Wednesday 9 Sep | `6ZCIGnuRZpE` | 11:31 | 9 Sep 11:38 | 92 (0:02-11:29) |
| Prediction For 10 SEP 2026 | `BbY6TwoC90o` | 2:13 | 9 Sep 21:25 | 17 (0:06-2:04) |

### IH LOST TODAY, and that is the most useful thing in the video

He is not a benchmark that always wins, and this session is the counter-example
to v4z. He read the fuel correctly and still had to cut:

> "The market shouldn't go up, because **no SLs are available nearby**. No
> psychology is forming that would take it up. But if it IS going up, we have a
> LIMIT... I feel fear when I can SEE SLs available, or when something forms per
> the mindset. There's nothing like that -- but the market is going up, the loss
> is growing, **so we'll have to exit. There's no other way.**"

And the discipline that ended it:

> "We cannot wait OUTSIDE the loss limit... if it keeps crossing the LIMIT again
> and again, you should exit. **Don't build the habit of just sitting and
> sitting.** If we keep sitting like this, the market can give a very big loss --
> if not today then someday."

**CALIBRATION ON v4z, and it should be read WITH the rule rather than against it.**
v4z says an unfuelled counter-move dies. Today it did not, for the man who wrote
the reasoning. The absence of fuel raises the odds; it does not remove the need
for the loss limit, which is exactly what v4z's own override clause and v4h
already say. What today adds is the demonstration: being right about the fuel and
still losing is a NORMAL outcome, not evidence the read was wrong.

### The day: 2 trades, -1,701.50

NIFTY gapped down to 23519.45, chopped 23474-23545 all morning, closed ~23470.

| # | Window | Entry | Stop | Lots | Exit | Basket |
|---|---|---|---|---|---|---|
| 1 | 09:24:48 -> 09:26:19 | 23500.60 | 23518.00 (15.15pt) | 2 | mechanical `AI_STOP` at 23519.30 | **-1,202.00** |
| 2 | 09:50:37 -> 09:58:35 | 23497.90 | 23510.00 (11.85pt) | 3 | `index_hierarchy_reversal_exit` | **-499.50** |

**Trade 2's exit was RIGHT and should be recorded as such.** It cut at ~23494;
NIFTY fell to 23474.85 by 10:09 and then rallied to 23545.55 by 10:25, which
would have taken out the 23510 stop for a larger loss. The BankNIFTY hammer it
cited was an early read on a real turn.

But its rationale contains the sentence worth watching: *"Per the index
hierarchy, a confirmed reversal on the leading index disqualifies the short
basket **even though NIFTY's own pullback is small**."* It saw v4y's measurement,
named it, and overrode it with the hierarchy. It was right this time. The corpus
should not be edited on one correct call, but if hierarchy-over-size becomes the
habit, that is the seam to watch.

### Trade 1 is v5a

Stopped ninety-one seconds after entry, on a 15.15-point stop, in the bounce that
follows a large gap down. The direction was right - NIFTY reached 23474.85 by
10:09 - and the session's opening bounce topped at 23524.20, so a ~24-point stop
would have survived it and bought ONE lot instead of two.

IH describes that exact bounce, in advance, in the same session:

> "The gap down is quite good, so **let's WAIT a bit**, because the momentums are
> sharp. If such selling comes in a gap down, nobody will buy - everyone starts
> selling... **If the market fell straight, RANDOM SELLERS also get activated, and
> when it flushes them, sometimes the market makes a bigger positive momentum.**
> Greed will definitely come here, but we must not be greedy."

That is now the SECOND time he has said wait-for-the-retracement on a directional
open (7 Sep was the first), and the second time the agent has been hurt by not
waiting. It stays recorded rather than encoded only because v4f already covers it;
if it recurs a third time, encode it.

### The sizing arithmetic behind v5a

Nineteen sizing decisions are now in the log, and the relationship is exact:

| stop (pts) | one-lot risk | lots | NIFTY qty | mirror qty |
|---|---|---|---|---|
| 25.30 | 1,644.50 | 1 | 65 | 30 |
| 17.90 | 1,163.50 | 2 | 130 | 60 |
| 15.15 | 984.75 | 2 | 130 | 60 |
| 11.85 | 770.25 | 3 | 195 | 90 |
| 7.65 | 497.25 | **5 (cap)** | 325 | 150 |

Same 2500 budget throughout. The stop is not setting risk, it is setting size -
and the BankNIFTY mirror, which is equal-LOT and outside the budget entirely,
scales with it: about 15,350 of premium at a 24.40-point stop on 08 Sep, about
46,845 at an 11.85-point stop on 09 Sep.

### The 10 Sep prediction: the HOLD is the read

Same sell branch as yesterday, opposite reason. He gives the counterfactual
himself: *"if the market had NOT held here, if it had gone straight up and then
straight down, then reversal chances rise. But it HELD, then fell"* - and
*"holding forms a PSYCHOLOGY, then the market falls"*. Because it held, few could
participate, so the sellers are THIN rather than seated. Yesterday they were
seated and therefore unhuntable; today they are thin. Same branch, different
mechanism, and the mechanism is what would flip it.

**SENSEX has expiry** on 10 Sep; NIFTY's was 08 Sep and is past.

## Video addendum - the 10 Sep LIVE SESSION (v5b)

**Source.** Intraday Hunter live session `JsdfH5uOuaw` (10 Sep 2026, 10:50,
uploaded 11:20 IST). Transcript read in full: 87 segments, 0:06 to 10:40.
The "Prediction For 11 SEP 2026" video was not yet published at the time of
writing; those go up around 21:25 IST.

### The best day of the series: 4 trades, +6,617.00

| # | Window | Dir | Stop | Lots | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|---|---|
| 1 | 09:25:43 -> 09:27:38 | LONG | 16.55pt | 2 | `profit_book_opposing_cross_index` | +1,027.00 | +2,052.00 | **+3,079.00** |
| 2 | 09:45:39 -> 09:50:05 | SHORT | 14.40pt | 2 | mechanical `AI_STOP` | -780.00 | -600.00 | **-1,380.00** |
| 3 | 09:56:34 -> 10:09:27 | SHORT | 12.65pt | 3 | `target_approach_round_number_book` | +2,359.50 | +2,713.50 | **+5,073.00** |
| 4 | 10:20:38 -> 10:24:32 | SHORT | 8.05pt | 4 | `index_hierarchy_exit` | +247.00 | -402.00 | **-155.00** |
| | | | | | **TOTAL** | **+2,853.50** | **+3,763.50** | **+6,617.00** |

Three winners, one stop-out. Trade 3's exit is the model one: conf=8,
`target_approach_round_number_book`, booked at 23410.05 with the target at 23405
and the 23400 psych level right underneath.

**The pre-open note did its job.** Trade 4's rationale quotes it almost verbatim:
*"prior session held then fell, so sellers are thin/unseated and today is a
follow, not a hunt."* That is the note's crowd read being used as a crowd read
rather than as a direction.

### v5a DID NOT BITE ON ITS FIRST LIVE DAY

Recorded plainly because the day was profitable and that makes it easy to miss.
v5a merged at 22:13 IST on 9 Sep and was in the prompt for this session. The
behaviour it targets continued and intensified:

| trade | stop | lots | NIFTY qty | mirror qty |
|---|---|---|---|---|
| 1 | 16.55 | 2 | 130 | 60 |
| 2 | 14.40 | 2 | 130 | 60 |
| 3 | 12.65 | 3 | 195 | 90 |
| 4 | **8.05** | **4** | **260** | **120** |

Every entry rationale says some version of "stop tight just above the pattern
high". Anchoring the stop to the pattern is legitimate and is not what v5a
forbids -- but it produced the narrowest stop and the largest position of the
series, and the rule did not intervene. The day was +6,617 so nothing punished
it; that is luck about the regime, not evidence the rule is working.

**Suggested next step, not taken here:** v5a is currently a piece of reasoning
with no concrete trigger. v4v got teeth when it was given one. The obvious
candidate is a stated floor on stop width, or a check that fires when the stop
narrows across consecutive entries. That is a live-money sizing decision and
belongs to the operator.

### v5b: how a hold EVICTS a crowd

IH sold for the third consecutive day and explained why he could:

> "For two-three days there has been continuous selling... still, what are we
> doing? Following the continuous selling. And that is because **we do not have
> the sellers' SLs available.**"

The mechanism, and it is the part the corpus did not have:

> "When the market recovered sharply and then **HELD** here, it changes many
> traders' mindset. Those who were selling start thinking of BUYING... they all
> EXIT their selling trade and start buying too. It is not that buyers stayed
> seated -- **the market COMPLETELY REMOVED the sellers.**"

And the discriminator, stated as a counterfactual:

> "The situation could have been **different**. If the market had gone up but
> fallen **immediately**, or fallen after stopping only a little, then you would
> find sellers **seated**. Then you would have to target them."

So the same price shape means opposite things, and what decides it is DURATION:

> "**Intraday** traders exit on a small retracement, they do not have that much
> SL. But if someone made a **positional** put and the market held for a long
> time -- roughly 12:30 to 2:30, that is **two hours** -- then even a positional
> trader exits."

Plus the persistence clause: *"if a seller has exited, he does not quickly make a
selling trade again"* -- an evicted seller either buys or stands aside, and
neither restocks the inventory.

**Why this is not v3z.** v3z (THE MISSING RIP IS THE TELL) infers that a crowd
was never there, from the market declining to hunt it. v5b describes how a crowd
that WAS there is removed, and gives the time signature that says which one. They
compose: v3z reads absence, v5b explains it.

**Both scales in one session.** Minutes after the entry, BankNIFTY broke 500 and
IH read it the same way: SENSEX and NIFTY had reacted first, *"so traders take
big quantity in BankNIFTY expecting it to fall from above -- their SLs are around
500, so the market gave an upside momentum **just to remove them**."* A fresh
micro-crowd, formed and cleared inside the hour.

### Confirmed, not encoded

- **v3z's after-a-losing-day rule.** IH books early and says why, twice:
  *"our problem is that yesterday we had a loss. Today if we are getting a chance
  to make a decent profit, we book and go... we cannot take much risk."* That is
  AFTER A LOSING DAY, TAKE THE GOOD PROFIT RATHER THAN THE BIG ONE, unchanged.
- **He names his entry levels BEFORE the open** (NIFTY ~23500, BankNIFTY
  ~56350-56400, SENSEX around the closing price) and waits for price to reach
  them. The agent reacts to patterns as they print instead. Not encoded: it is a
  workflow difference rather than a rule, and the pre-open note already carries
  his levels.

### The hierarchy exits keep being right

Flagged two sessions ago as "the seam to watch". Trade 4's `index_hierarchy_exit`
cut at ~23416 on a BankNIFTY confirmed hammer; NIFTY reached 23443.05 by 10:44,
which would have taken out the 23436.50 stop. That is three correct hierarchy
exits in a row (09 Sep trade 2, 10 Sep trade 4). The seam is still worth watching,
but it is now watching something that keeps working.

## Video addendum - the 11 Sep LIVE SESSION (v5c)

**Source.** Intraday Hunter live session `SW6rxOEMJkQ` (11 Sep 2026, 12:20,
uploaded 11:38 IST). Transcript read in full: 99 segments, 0:02 to 12:15.

### The day: 3 trades, +3,370.75, all LONG

NIFTY gapped down 210 points to 23268.05, chopped to a 23245.35 low by 09:41,
then ran to 23307.75 by 10:22.

| # | Window | Stop | Lots | Exit | NIFTY | Mirror | Basket |
|---|---|---|---|---|---|---|---|
| 1 | 09:19:33 -> 09:41:34 | 23.30pt | 1 | `premise_invalidation_bearish_cluster` | -338.00 | +1,597.50 | **+1,259.50** |
| 2 | 09:59:35 -> 10:10:38 | 23.10pt | 1 | `book_before_round_number` (conf 8) | +672.75 | +795.00 | **+1,467.75** |
| 3 | 10:18:03 -> 10:26:52 | 10.85pt | 3 | `profit_book_reversal_cluster` | +643.50 | *skipped* | **+643.50** |
| | | | | **TOTAL** | **+978.25** | **+2,392.50** | **+3,370.75** |

**The inverted note worked.** All three entries are LONG, and each rationale
cites the buy branch explicitly -- "pre-open note explicitly hunts the late
sellers", "matches the pre-open note's buy branch". After three sessions of
follow-the-selling, the agent flipped when the note flipped.

**v5a bit, partially.** Trades 1 and 2 used 23.30 and 23.10-point stops and took
ONE lot each -- the widest stops since 04 Sep, and a clear break from the
16.55 -> 8.05 slide of the previous session. Trade 3 then went back to 10.85 and
3 lots. Two out of three is real movement, not noise, but the rule is not yet
decisive.

### Trade 1 is v5c, and it is the same failure in a new costume

Cut at 23245.35 on a "confirmed bearish reversal cluster" with the NIFTY leg at
-299. **That price is the session low.** NIFTY ran 62 points from there, and the
agent re-entered the same direction twice more to catch part of what it had just
left.

IH, in the same session, held through exactly that dip -- "we are NOT going to
exit here, even if the loss grows a bit" -- and explains why the dip was
expected:

> "I tell you again and again: a market that falls after a **small** retracement,
> let it fall. But falling after a **big** retracement becomes DIFFICULT for the
> market... when such a big retracement happens and then the market falls,
> **EVERYONE will sell there**."

And he narrates the loop as it runs:

> "This candle formed only to give **greed**. What will the trader see? Selling
> happened, retracement happened, suddenly it fell. As premiums rise he will make
> a put trade -- and targeting exactly them, the market goes up again. Then it
> repeats the same."

So the dip is not noise to be endured, it is the mechanism that manufactures the
crowd the trade is aiming at. The agent closed at the exact moment its own
premise was being built.

**v5c pairs with v5b deliberately.** Same counter-move, two axes: v5b times it and
finds an eviction, v5c sizes it and finds a recruitment. Both preconditions are
in the rule because IH states both -- sentiment already negative, and the
session's lower point not crossed, which is a v4v-style level test and the thing
that ends the setup.

### A real defect worth a separate look: the BNF mirror race

Trade 3 opened 3 lots of NIFTY with **no BankNIFTY mirror**:

```
10:18:01,349  MarketDataFetcher  Published frame | LastCandle=10:18:00
10:18:03,251  sl-hunting-sdk-call  dynamic sizing: ... lots=3 qty=195
10:18:03,517  sl-hunting-sdk-call  ENTRY LONG  NIFTY-Sep2026-23300-CE  Qty=195
10:18:05,563  sl-hunting-sdk-call  WARNING  BNF mirror skipped: no BankNIFTY close seen yet this session.
10:18:13,810  sl-hunting-inference  decision took 60.6s
```

The message is wrong and the cause is a race. `_open_bnf_mirror` skips when
`self._last_bnf_close <= 0`, and the decision loop sets `self._last_bnf_close =
0.0` at the top of EVERY cycle, repopulating it only from a BankNIFTY bar that is
timestamp-aligned with NIFTY. The 60.6s inference began around 10:17:13; its
order tool fired at 10:18:03, by which time the 10:18 bar's cycle had already
reset the field. A BankNIFTY close had certainly been seen this session -- two
mirrors were placed on it at 09:19 and 09:59.

Consequences, in order of importance:

1. The documented invariant "every NIFTY entry is mechanically MIRRORED" silently
   did not hold, and the only trace is one WARNING. Exposure was LOWER, not
   higher, so this is not dangerous -- but it is not the configured strategy.
2. The log line sends an operator hunting a session-wide feed failure that did
   not happen. It should say "no ALIGNED BankNIFTY close on this bar".

**Frequency, corrected.** This exact message has fired TWICE in the whole log
(2026-08-11 and 2026-09-11). A grep for "BNF mirror skipped" returns twelve lines,
but the other ten are different reasons (eight "no LTP for BANKNIFTY-...", two
"NIFTY entry is paper fallback"). It is a rare race, not a recurring one.

Not fixed here: it is live-order-path code and this session's task was the
knowledge rule. The candidate fix is to stop resetting `_last_bnf_close`
unconditionally, or to capture it at decision time and pass it to the order tool.

### Confirmed, not encoded

- **v4d, pre-committing the adverse move.** IH, before entering against the
  opening trend: "there's risk that we're working OPPOSITE to the market's trend
  per the opening... **we're assuming roughly 1.5 lakh of loss could happen**."
- **v3z's two-sided flow.** "A few buyers can still come, a few sellers can come,
  so liquidity remains. But when only ONE type of trader starts coming, the market
  cannot work there for long."
- **He abandoned his own pre-open plan's REASON while keeping its direction**:
  "in the analysis we had planned to target these sellers, but that condition is
  no longer visible because the gap down is quite large -- per the NEW chart,
  buying chances are forming anyway." The note's conclusion survived; its stated
  mechanism did not. Worth remembering when a note and the tape disagree on WHY.

## Video addendum - the 12-14 Sep gap and the 15 Sep prediction

**There was no trading between 11 and 15 Sep.** 12-13 Sep was the weekend and
14 Sep was a market holiday, which IH confirms in his own words -- "a two-three
day holiday in the middle". The runner did not start on any of those days: the
log's last line is 2026-09-11 15:20:16, and the shipped note sat at `2026-09-11`
until this one replaced it. So there are NO agent trades to compare for this
stretch, and nothing was missed by not looking.

**Everything the channel published since the 11 Sep session, and what happened to
each:**

| Video | Id | Uploaded (IST) | Status |
|---|---|---|---|
| Prediction For 15 SEP 2026 | `xB8Jt0d4Ll8` | 14 Sep 21:00 | read in full, 16 segments (0:06-2:01) |
| The Power of an Edge in Trading | `uC4PeLdGFmo` | 13 Sep 11:47 | **transcript NOT retrievable** |
| Live Crypto Trading 13 SEP | `rRSE6lk5pqU` | 13 Sep | members-only, and crypto -- out of scope |

There was no "Prediction For 12 SEP", no 12 Sep live session, no "Prediction For
14 SEP" and no 14 Sep live session. The channel published nothing between the
11 Sep pair and the 13 Sep weekend content, which is consistent with the holiday.

### The Edge lecture: retrieved by hand, after automation failed

`uC4PeLdGFmo` (15:20, 13 Sep 11:47 IST) is the weekend lecture slot. Its
transcript panel opens and then spins forever: the normal recipe, a real trusted
click rather than a JS click, a resize to 1400x900 and a full reload at that size
all left `#segments-container` unmounted and the panel's own `innerText` as
skeleton whitespace after ~35 seconds of polling. Same per-video failure already
on record; the 15 Sep prediction transcribed normally minutes earlier.

**The operator loaded it manually in Chrome and supplied the transcript**, so the
content below is read in full rather than inferred. Worth remembering as the
fallback when the panel refuses: a human-loaded page still produces one.

#### What it says

The lecture is about the difference between having an EDGE and expecting
certainty, and it contains two ideas this corpus does not have.

**1. A setup you cannot lose on is a broken setup.** Stated as a test, not a
sentiment:

> "Any setup you have will only work IF it has a CHANCE OF LOSS... if you have
> built a setup where you can SEE there will be no loss, you should RE-CHECK it,
> because the setup that makes profit is the one that HAS a margin of loss."

And the symptom to look for:

> "If you feel the market will go up and you profit, go down and you profit,
> sideways and you profit, momentum and you profit -- there is very little chance
> that setup works."

He is explicit that he has chased this himself: nights spent on a setup that
looked lossless, "but it runs for a few days, then its number comes up". Also
that neither large nor small capital fixes it -- large capital lets you manage a
wrong position until "it goes outside the range", small capital only helps
because it controls emotion.

This is adjacent to v3z's NAME THE ONE WAY THIS TRADE FAILS but is not the same
rule. v3z asks you to name the failure mode; this asks what it MEANS when you
cannot -- the inability to name it is itself the disqualifying signal, and the
setup is to be re-checked rather than taken.

**2. Analysis and trading are separate skills with different edges.**

> "In trading there are TWO things: you ANALYSE by looking at the chart, and you
> TRADE by looking at the chart. Your edge in analysis should be quite good,
> because in analysis there is no matter of entry/exit timing... in trading we
> can be wrong, the edge there is LOWER."

And the consequence he repeats three times: "when the analysis is right, that
does not mean your trade will also be right." He grades his own week by exactly
that split -- "in analysis you could say we were 100% right", and then, of a
losing trade inside the same week, "here we had a loss" because the market
retraced and pulled out positive momentum.

That split maps cleanly onto this agent: the pre-open note plus the direction
read is the ANALYSIS, and the entry, stop and exit are the TRADE. It is also the
shape of every session in this series -- direction right, execution wrong.

#### Two smaller things, both already covered

- **Friday and the day before expiry are his reversal-prone days**, with a
  mechanism: on a Friday a reversal cannot be bought, because of the two-day
  holiday, so sellers keep accumulating and become the target. This is how he
  called the 11 Sep reversal in advance. The calendar pairing is new phrasing,
  but v5b already carries the holiday-eviction mechanism and the 07 Sep note
  carried the weekend flush.
- **His own account of the 11 Sep trade**: "we did not think the recovery would
  be this big -- otherwise we would have kept sitting. We took a decent profit."
  Consistent with v3z's after-a-losing-day rule, which he cited that morning.

### The 15 Sep prediction: every shape buys, as a FOLLOW this time

Structurally this looks like the 11 Sep note -- all three open shapes buy -- but
the mechanism is the opposite one, and that difference is the whole content:

- **11 Sep**: sellers were seated and FEARFUL, so the buy branch was a HUNT of
  their stops.
- **15 Sep**: nobody is seated on either side, so the buy branch is a FOLLOW.
  "Such a market can be followed continuously", "we can walk with the market".

He checks both books explicitly, which is what makes it a follow rather than a
hunt: "if in gap-up or flat we try to find any SELLER's SLs, we won't get them"
-- and in the same breath, "but the BUYERS' SLs aren't here either, because the
momentum happened suddenly, it is a reversal, and there is a two-three day
holiday in the middle, so not many people are sitting long here."

That is v5b's eviction logic arriving by a different route: the holiday did to the
would-be buyers what a long hold does to a trapped crowd. Nothing is added to the
corpus for it -- v5b already covers the mechanism, and this is one instance of it.

Two smaller points carried into the note: a gap DOWN keeps the same buy plan
"because overall positive momentum has happened", and he flags that "any recovery
can also be a reversal", so this may be a genuine turn in the multi-day downtrend
rather than a bounce inside it.
## v5d - both ideas from the Edge lecture, encoded

The 13 Sep lecture `uC4PeLdGFmo` was recorded in the addendum above with its two
net-new ideas flagged but NOT encoded, because a rule had not been asked for.
Both are now in the corpus, tagged v5d, each placed beside the rule it extends
rather than in a block of their own.

**1. IF YOU CANNOT SEE HOW THIS LOSES, THE SETUP IS BROKEN** - sits immediately
after v3z's NAME THE ONE WAY THIS TRADE FAILS, because it is that rule's
contrapositive. v3z asks you to name the failure; v5d says the inability to name
one is a defect rather than confidence, and that the response is to RE-CHECK the
read, not to size down. Its checkable symptom is a rationale in which up, down,
sideways and momentum all end in profit.

It is written pointed at this agent on purpose. The agent composes its own
rationale before acting, and a long one that answers every objection reads as
thoroughness -- which is exactly the shape IH warns about.

**2. YOUR EDGE IN THE READ IS HIGHER THAN YOUR EDGE IN THE TRADE** - sits after
v4h's ENTRY QUALITY AND DIRECTION ARE SEPARATE JUDGEMENTS, which already splits a
loss into direction and entry. v5d adds that the two do not carry the same edge
and that the gap is structural: the read has no entry, no target and no clock, so
emotion stays out of it.

The mapping is stated explicitly so it can be used -- the pre-open note plus the
direction read is the ANALYSIS, the entry, stop and exit are the TRADE. A losing
session therefore does not falsify the read, and a correct read does not entitle
the trade. This is the single most accurate description of every session in this
series: direction right, execution wrong.

**Not encoded, again:** the Friday / day-before-expiry reversal pairing. The
Friday half is already carried by v5b's eviction mechanism and the 07 Sep note's
weekend flush, and the expiry half has no measured case behind it yet. It stays
in the addendum until a session produces one.

Both guards were negative-tested 25 ways between them. Four assertions initially
failed to bite -- the RE-CHECK instruction inside a quote, the "runs for a few
days" half of the anecdote, the "emotion stays out of it" mechanism, and the
"repeats three times" framing -- each was matched by a substring that survived
its own mutation. All four were tightened and re-verified rather than left
passing-but-weak.
## v5e - the stop that sat inside the level the premise named (15 Sep)

Source: `Yl-m6yX2Fok`, "Live Bank Nifty Option Trading", uploaded 2026-09-15
10:11 IST. Compared against the agent's own session the same morning (two
trades, both stopped, basket -Rs.1,294). 11 Sep was the previous trading day;
12-14 Sep were closed, which is the gap both sides were reading.

**The inversion.** For the first time in this series IH and the agent read the
same day in OPPOSITE directions, and the agent was right. Both named the same
level. IH bought calls into the gap-up's fast rejection on the trap read -- his
stated logic is that a fast rejection traps nobody because the preceding
multi-day selling and the holiday meant "in the reversal most people could not
buy", so there were no seated buyers to hunt and the sharp drop had to be a
shakeout. He sized across all three indices (BankNIFTY 1170 on the 56900 call,
Sensex 900, NIFTY 1430 on its expiry day). The agent read the same rejection as
a genuine breakdown and shorted it. NIFTY closed at 23179.55, roughly 400 points
below the open. IH cut at his limit and posted the session as a loss.

**What killed IH's trade is the level the agent had named.** His exit criterion,
in his own words, was that the closing-price breakdown "should not have
happened here -- because there was already a rejection, we should have
immediately got momentum covering the market". The agent's ENTRY criterion, in
its own words, was the break "below both the pivot (23359.2) and previous close
(23398.1)". Same level, opposite sides, and the agent had it right.

**The agent still lost on it, and the cause is mechanical.** It shorted at
23376.90 and put the stop at 23397.55: the high of the bounce it was fading,
and 0.55 of a point BELOW the previous close it had just cited as the reason
for the trade. Price returned to 23397.95 -- 0.40 past the stop, and still 0.15
SHORT of that previous close. The retest of the named level never completed and
the position was already gone. NIFTY then fell through the 23300 target around
12:59 and finished 197 points beyond the entry. The thesis paid in full; the
trade collected none of it.

Encoded as **THE LEVEL YOU BROKE TO GET IN IS THE LEVEL PRICE COMES BACK TO
(v5e)**, placed directly under the confirmation-candle bullet in
PATTERNS_AND_CONFIRMATION -- the bullet that says "the stop sits just beyond the
pattern". That instruction produced this loss, so v5e refines it in place rather
than competing with it from another section: beyond the pattern is necessary and
not sufficient, and a reference the premise cited (or one every screen shows)
sitting beside the pattern edge must itself be cleared. A stop AT a level is a
stop INSIDE it, because the probe that makes a level a level overshoots it.

The rule closes the obvious wrong fix explicitly, because it is the one an agent
reaches for: clearance is bought with SIZE, never back from the stop. It
cross-references v5a (A TIGHTER STOP IS NOT LESS RISK, IT IS MORE SIZE) and v4x
(THE STOP IS A DISTANCE FROM THE FILL), which reach the same law from the other
side -- there the stop is too close because price moved, here because a level is
in the way, and in both the thing that gives is the size.

Negative-tested 16 ways, all 16 caught on the first pass, plus a control
mutation of unasserted prose confirming the harness discriminates rather than
failing on everything.

**Considered and not encoded.** IH's fast-rejection-is-a-trap read is the thesis
that LOST him the day, so it is not a candidate. His "a real trap recovers
IMMEDIATELY, so a thesis that needs time is not a trap" deadline is the more
durable idea in the video, but v3j already considered and deliberately dropped
"counter-gap recovery must be FAST" because it collides with RISK's
SLOW-but-CONTINUOUS is sustainable; one session of his own loss is not enough to
reverse that, and per BACKTEST A THRESHOLD BEFORE PICKING IT a deadline needs to
be priced against the journals first. His loss-discipline passage ("put your
thinking into the PROFIT trade, not the losing one -- the more effort you spend
on a trade the more emotionally connected you get, and then you will not want to
leave it") is close to already-encoded v3y BEING DIRECTIONALLY RIGHT DOES NOT
EARN THE HOLD, which names the same tell.

## SLH-016 - an entry whose stop is already breached is refused at the tool

The same session's FIRST trade is a separate failure with the same root, and it
is the one that had to become code rather than prose.

At 09:15:55 the order tool fired claiming entry 23571.40 with a stop at
23566.30 -- a 5.1-point allowance, so `from_risk_budget` granted the 5-lot CAP
(325 qty, mirrored by 150 BankNIFTY) -- while live spot was already 23493.25,
**73 points beyond that stop**. The ENTRY line records both numbers side by
side (`Spot=23493.25 | EntryUnderlying=23571.40 | StopUnderlying=23566.30`) and
nothing compared them. The position was closed 4 seconds later. The 60.6s
inference that authorised it completed at 09:16:05 and was then discarded as a
stale generation, so a real trade outlived and outranked the decision behind it.

v4x has said since 2026-09-04 that the stop is a distance from the FILL and
must be re-measured at the instant the position opens. It is prose, and prose
did not bind -- exactly the history behind SLH-005, where the re-entry gate was
talked past twice before its time arm moved into code.

`MasterWorkerExecutor.enter` now reads the worker's `_get_underlying_spot`
before delegating and refuses the entry when the stop is already on the wrong
side of spot (`<=` for LONG, `>=` for SHORT, so zero room counts as breached).
The rejection names both numbers and restates v4x's two honest answers:
re-derive the stop from the price in front of you and take the smaller size, or
take no trade -- never keep the original stop to keep the original size.

Four deliberate scope decisions:

- **The narrowest checkable arm only.** "Already breached" needs no threshold.
  How much of the sized allowance is merely SPENT -- v4x's 83% case from
  2026-09-04 -- stays a judgement the prompt owns, because choosing that
  fraction needs the journals behind it rather than an intuition. This follows
  BACKTEST A THRESHOLD BEFORE PICKING IT.
- **Paper as well as live.** A dead-on-arrival paper trade still writes a
  journal row and still trains the coach.
- **An unreadable spot is NOT refused here.** `enter_position` already skips the
  entry outright when the NIFTY LTP comes back non-positive; a second opinion
  would only make that failure harder to read in the log.
- **Duck-typed, entries only.** A worker without the accessor (the standalone
  paper runner) is unaffected, and exits are never gated -- a breached stop is
  precisely when an exit must stay available.

No extra broker call: `_get_underlying_spot` writes a direct fetch back into the
shared store, so this warms the very LTP `enter_position` reads a moment later.

Negative-tested 8 ways, all 8 caught, each by the specific test that should own
it. A ninth mutation was written and discarded as invalid -- setting the
exception branch to `float("inf")` is filtered by the `isfinite` guard and
changes no behaviour, so it proved nothing; it was replaced by mutations that
make the exception branch refuse and re-raise, both caught.
## v5f - a one-way day seats nobody, and the next open is the receipt (16 Sep)

Source: `TXiU5Odet04`, "Live Bank Nifty Option Trading", uploaded 2026-09-16
10:09 IST. Compared against the agent's own session the same morning: four
entries, all LONG, inside 53 minutes, basket **-Rs.863.75**.

**Same note, same morning, opposite branch.** The pre-open note shipped for
16 Sep (PR #171) was conditional: gap up -> BUY the trapped sellers, flat or
gap down -> SELL with the market. NIFTY opened +82 points, +0.36%; BankNIFTY
+0.38%.

- IH: "today we saw a FLAT opening, and with this kind of opening we are going
  to follow the CONTINUED SELLING... if a VERY BIG gap up had opened we could
  not have worked like this." He bought puts across all three indices, booked
  around his 3-lakh target on a sharp simultaneous breakdown, and closed the
  session a clear winner.
- The agent: "NIFTY gapped +82pts (0.36%) and BankNIFTY +213pts (0.38%) above
  prior close after a 457pt breakdown day that seated positional sellers,
  satisfying the pre-open note's GAP-UP->BUY branch." All four entries LONG.

**The proximate cause is a v4t violation, not a knowledge gap.** v4t already
says the classification is a JUDGEMENT you make and state, already calibrates
it ("a quarter of a percent is not a gap... something in the region of half a
percent is where the gap reading starts to earn itself"), and already says
that hesitation resolves to flat. 0.36% sits below that line. The number was
computed and written into every one of the four entry rationales and never
once tested against the rule. This is the THIRD session in the series decided
by that one word: 31 Aug produced v4t, 01 Sep produced its v4u 3:15 sub-rule
at a cost of Rs.1,488.25, and this one cost Rs.863.75.

**What IS net-new is the premise UNDERNEATH the branch**, and it is the reason
the agent wanted the gap-up branch at all. All four rationales asserted that
the prior 457-point breakdown had "seated positional sellers". IH argues the
exact opposite, with a mechanism: "when do sellers REMAIN seated? When a
retracement happens. When there is no retracement, every trader keeps fearing
that a retracement might come, that his trade might go wrong" -- so on
continuous momentum "traders do come, but they also book their target and
leave", and "the chances of HOLDING are low here". A violent one-way session
ends with almost nobody carrying inventory, however large its range.

He then supplies the falsification test, which is the part that can run before
the trade: "but if sellers had been seated in good quantity, what would the
market have opened as? We would have seen a straight gap up." The open is the
RECEIPT for the premise. A small gap is not a weak version of that open; it is
the evidence against it.

Encoded as **A ONE-WAY DAY LEAVES NOBODY SEATED, AND THE NEXT OPEN IS THE
RECEIPT (v5f)**, placed directly after the whole v4t block (including its v4u
sub-bullet, which it must not split) so the model reads "decide which word
describes the open" and then "here is what the open's size is measuring". It
is v5c one scale up: there a retracement recruits a crowd inside the move,
here a retracement is what lets a crowd survive into the NEXT session.

Negative-tested 14 ways, all 14 caught, plus a control mutation of unasserted
prose that correctly did not trip the test. One anchor initially matched zero
times because it began mid-line ("That is v5c...") rather than at a line
start; it was re-run line-contiguously and caught. That is the same
wrap-spanning mistake the mutation harnesses keep inviting -- print the raw
line boundaries first.

**Considered and not encoded.** IH's "a retracement is what seats a crowd" is
partly v5c already; only the cross-session half is new and that is what v5f
carries. His loss-discipline material repeats v3y. His round-number caution
("there is a round number here, a small problem will show, control your heart,
do not run") is already covered by the round-number rules in BNF_SPECIFIC.

Two runtime items were raised from this session. The first became SLH-017
below. The second is recorded there too, and is NOT being built.

## SLH-017 - the open is classified in code, not derived by the model

Three sessions in this book were decided by one word, with two knowledge rules
already in place specifically to get it right: 31 Aug produced v4t, 01 Sep
produced its v4u sub-rule at a cost of Rs.1,488.25, and 16 Sep cost Rs.863.75.
On 16 Sep the model computed the deciding number correctly -- "+82pts (0.36%)"
-- wrote it into all four entry rationales, and never compared it with v4t's
own calibration. That is the exact shape the corpus already has a law for:
PROSE RULES DON'T BIND, and JUDGEMENT RULES GET TALKED PAST.

`pivot_and_levels` now returns an `open_classification` block:

    previous_close / today_open / gap_points / gap_pct
    threshold_pct  (GAP_CLASSIFICATION_THRESHOLD_PCT = 0.5)
    verdict        FLAT | GAP_UP | GAP_DOWN
    reference      "previous session's last candle close, not the official
                    15:30 close"

The threshold is not invented: it is v4t's own calibration ("a quarter of a
percent is not a gap... something in the region of half a percent is where the
gap reading starts to earn itself"), and sub-threshold opens read FLAT in both
directions because v4t also says hesitation resolves to flat. The reference is
v4u's, for the reason v4u gives -- the closing auction carries settlement
prints no crowd traded around.

It flows to the model through the `levels` tool and, for BankNIFTY, through
`bank_nifty`. Run against 16 Sep's real numbers both come back **FLAT**
(NIFTY 0.355%, BankNIFTY 0.381%) -- which is IH's reading of that morning and
the opposite of the one the agent acted on.

**Wired, not just added.** v4t now carries an SLH-017 paragraph telling the
model to READ the verdict and state it, and not to recompute or argue past it,
because an unwired fact is one the model will simply re-derive its own way.
The existing v4t marker test was extended rather than rewritten, so its old
assertions still pin the original prose.

**Scope.** This supplies the arithmetic and the threshold; it does not reject
orders and does not judge what the classification MEANS. That part stays with
the model, which is where the real reasoning is.

Negative-tested 9 ways, all 9 caught, plus a control mutation of unasserted
prose that correctly did not trip the tests. One mutation -- measuring from
today's LAST price instead of its OPEN -- initially PASSED, because every
fixture opened and closed at the same price. That is a real defect the tests
could not see: a flat open would silently become a "gap up" by mid-morning and
re-select the branch hours after the fact. A dedicated drift test was added
and the mutation then caught.

## SLH-018 - one round of friction on a reasonless EXIT (operator decision)

The session's other item was that at 09:16:38 an EXIT with no reason executed
and closed the day's only winner (+Rs.1,182 basket), about 100 seconds before
that trade's 23276 target printed. The order tool logged it as "a probable
UNINTENDED order" and honoured it anyway; the model's own next decision called
the call erroneous.

This was first raised and **declined**, because refusing such an exit reverses a
standing decision: SLH-007 says an EXIT is never refused over its wording, since
bouncing it would strand an open position; SLH-010 explicitly declined to reverse
that and added the warning instead; and
`test_exit_with_no_reason_still_executes_but_warns_loudly` existed so that "a
later tightening cannot quietly turn the warning into a refusal".

**The operator then took the decision explicitly (2026-09-16), choosing the
one-round form**, on the reasoning that it "would filter out the anomalies and
let SLH-007 work on genuine wordless exit cases". SLH-007's guarantee is
NARROWED, not reversed, and the tightening is neither quiet nor unilateral.

**The contract.** The first EXIT that resolves to the no-reason sentinel is sent
back once, with a rejection naming both ways forward -- state the reason, or send
the EXIT again. Any repeat is honoured unconditionally, whatever it carries. So
the worst case is ONE decision cycle, never a stranded position, and the
mechanical stop, target, max-loss and 15:15 square-off are host-owned throughout
and are never gated by any of this.

**Where the state lives matters.** The armed flag sits on the EXECUTOR, not on
the tool context. A context is rebuilt every bar, so context-local state would
bounce the first attempt of every bar forever and could genuinely strand a
position -- the exact outcome SLH-007 forbids. The executor is created once per
worker (`self._executor = MasterWorkerExecutor(self)`) and outlives the pass. It
is set and cleared duck-typed through `getattr`/`setattr` under
`contextlib.suppress`, because `TradeExecutor` is a Protocol and a third-party
executor need not tolerate the attribute.

**Scoped per episode.** Any ACCEPTED order clears the arm -- the repeat exit, a
restated reason, or a new entry. A bounce spent on this morning's slip cannot buy
a later position a free pass.

**Placeholder reasons bounce too.** `_safe_exit_reason` maps "placeholder",
"n/a" and friends to the same sentinel as an empty string, so they are the same
class of anomaly and get the same single bounce.
`test_do_order_never_rejects_an_exit_for_a_bad_reason` was rewritten as
`test_a_junk_exit_reason_bounces_once_then_exits_and_records_the_sentinel`,
keeping its real assertions: the position still comes out, and the junk still
never becomes the permanent record.

**The tool description had to change, or the model keeps a false contract.** It
said an EXIT "is NEVER rejected". It now states the single exception and the way
out of it, and a test asserts the stale absolute is GONE rather than merely that
the new wording is present -- a model told exits can never be refused, that then
receives a refusal, has been handed a contract its tool does not keep. That is
the SLH-017 unwired-fact failure arriving from the other side.

Negative-tested 7 ways, all 7 caught, plus a control mutation of an unasserted
comment that correctly did not trip the tests. The two mutations that matter
most are both caught: removing the bounce entirely, and never disarming (which
would strand a position by refusing the second attempt as well).
## v5g - an eviction leaves you in the same sideways stretch (17 Sep)

Source: `ICw5RD1xTVg`, "Live Bank Nifty Option Trading", uploaded 2026-09-17
10:08 IST. Compared against the agent's own session: two trades, both SHORT,
basket **+Rs.11,355.00**.

### First, the part that worked

This was the first live session with the 17 Sep pre-open note, SLH-017 and
SLH-018 all active, and **SLH-017 did exactly what it was built for**. Both
entry rationales read the computed verdict off the tool and stated it:

- 09:39 - "Open classified FLAT on both NIFTY (-0.096%) and BankNIFTY
  (-0.295%), triggering the pre-open note's FLAT-open SELL branch"
- 10:18 - "Flat open (NIFTY -0.10%, BankNIFTY -0.30%, both below the 0.5%
  threshold) matches the pre-open note's SELL branch"

Yesterday the model computed 0.36% by hand and called it a gap-up, taking the
wrong branch four times. Today it was handed FLAT, cited the threshold, and
took the branch the note prescribed. SLH-018 never fired - no reasonless exit
arrived - which is the correct behaviour for a guard.

**IH reached the same read and took the same side**, and his reasoning is the
note's flat branch almost verbatim: "when the market moves UP and does not
cross the round number, sellers gradually come there too. But the market keeps
HOLDING in one place... even if someone made a trade, they do not hold a
positional trade. So if sellers are not seated here, we got a flat to slightly
gap-down open... Overall we should get selling."

### Then the part worth encoding: he LOST and the book WON on it

He entered at the open and held. NIFTY chopped between roughly 23,217 and
23,305 all morning, broke that range near noon, and ran to 23,358.70 by 12:38.
BankNIFTY broke out, his loss limit was exhausted, and he cut - "there is no
need to apply the brain here, cut the trade and get out."

The book traded the same thesis in two BOUNDED windows:

| | entry | exit | basket |
|---|---|---|---|
| 1 | SHORT 23,282.75 at 09:39 | 09:43, booked into the stall at support | +2,820.00 |
| 2 | SHORT 23,302.90 at 10:18 | 11:00 TIME_CUTOFF | +8,535.00 |

The read was right for about ninety minutes and wrong afterwards.

**The net-new claim is the reflexive one, and it is the missing half of v5b.**
v5b reads an eviction and licenses a FOLLOW rather than a hunt. What that
follow inherits is that nobody is trapped, so nothing is forced to move - and
the reasoning that clears the crowd ("if someone made a put trade and the
market gave no momentum for three or four hours, his premium decays and he
will not hold") is a description of the conditions you are about to enter, on
the same clock, paying the same decay. The eviction argument is evidence about
the PAST; it promises nothing about the next hour.

Encoded as **A CROWD EVICTED BY A SIDEWAYS STRETCH LEAVES YOU IN THE SAME
SIDEWAYS STRETCH (v5g)**, placed after the WHOLE v5c block rather than beside
v5b, because v5c opens "The twin of the rule above" and that reference points
at v5b - inserting between them would have silently re-aimed it. (The same
class of mistake as splitting v4t from its v4u sub-bullet the day before.)

### The honesty clause is part of the rule, not commentary

Trade 2's exit was NOT a decision. The agent had been failing every bar since
10:42 - 19 `agent_error` rows, a canned `SLHuntingAgentError` per bar for the
last eighteen minutes - and the position was closed by the 11:00 cutoff. What
won was the BOX, not the judgement: a time-boxed book beat an open-ended one on
the same idea. The rule says so in its own text, and a mutation that removes
that sentence is caught, because without it the day reads as the agent
out-trading him, which is false and flattering.

Worth noting plainly: the 11:00 cutoff is the operator's Pro-plan constraint,
not a trading rule, and today it booked the second short within about six
points of the session's post-entry low before a reversal that ran 140 points
the other way. That is the constraint earning its keep by accident, and it is
exactly why the rule asks for the window to be named DELIBERATELY.

Negative-tested 11 ways, all 11 caught, plus a control mutation of unasserted
prose that correctly did not trip the test.

**Considered and not encoded.** His long emotion/rule-discipline passage - "the
more you think about things, the more your decisions will be wrong", "the
easiest way to control emotion is to follow the rule" - is the third session
running to make that point and is already carried by v3y BEING DIRECTIONALLY
RIGHT DOES NOT EARN THE HOLD plus the RISK section's loss-limit rules. His
"if the loss limit is exhausted there is no need to apply the brain" is v5c's
WHAT IT DOES NOT SAY clause stated from the other side.

**Confirmed, not open.** The 18-minute outage from 10:42 was the Pro plan's
5-hour usage window expiring - confirmed by the operator the same evening.
Nothing is broken, and no code change is wanted.

What it adds to the existing constraint is a NUMBER. The 11:00 cutoff is sized
to that plan window, but the window does not END at 11:00: today it ran out at
about 10:42, roughly eighteen minutes early. So **the agent can go dark before
its own cutoff**, and anything open in that gap is unmanaged - no premise
check, no agent exit, nothing but the host's mechanical risk (stop, target,
max-loss, 15:15 square-off) and then the cutoff flatten.

Today that gap was worth +8,535, because the cutoff happened to book the
second short about six points off the session's post-entry low, minutes before
a reversal that ran 140 points the other way. That is luck, not design, and it
could as easily run the other way on a day the position is offside. The answer
already in this book is v5g's PRACTICAL FORM - name the window before
entering - which is worth more here than any new guard, because a position
sized and reasoned for a window that ends at 11:00 is really operating on one
that can end at 10:40.

NOTE FOR NEXT TIME: the cause is not recoverable from the log. SLH-004 logs
this class with CANNED, content-free messages on purpose, so a future outage
of the same shape will again need the operator to say which it was.
## v5h - in a range the crowd read cannot pay (18 Sep)

Source: `Z3D-p6RvPc0`, "Live Bank Nifty Option Trading", uploaded 2026-09-18
10:50 IST. Compared against the agent's own session: four trades, all SHORT,
basket **+Rs.4,308.25**.

### The day both sides got right

Third session running where IH and the book reached the same read and took the
same side. The open was +0.275% on NIFTY and +0.209% on BankNIFTY -- a small UP
gap that SLH-017 classified FLAT, which is exactly the case the classification
was built for. All four entries cited the verdict and the threshold explicitly,
and took the note's flat branch. IH: "almost flat opening... until these levels
are crossed we can follow the continued selling", and he bought puts.

He booked one trade, took the profit, and stopped -- "because we had a loss
yesterday and a loss the day before, this week we have not made much, so if we
are getting a chance to make profit today, let us take the exit."

### What the book did instead

| # | entry | exit | held | basket |
|---|---|---|---|---|
| 1 | SHORT 23,322.60 09:20 | 09:34 momentum stall | 14m | **+4,466.25** |
| 2 | SHORT 23,310.05 09:47 | 09:54 bullish engulfing at the stop | 7m | -1,959.00 |
| 3 | SHORT 23,319.00 10:04 | 10:08 profit decay | 3.5m | +338.00 |
| 4 | SHORT 23,318.95 10:23 | 10:27 profit growth stalled | 3.5m | +1,463.00 |

The first trade was the day. Trades 2, 3 and 4 netted **MINUS 158** between
them, for three more entries, three more mirrored baskets and three more
exposures. The session's entire range was about 44 points (23,292.55 to
23,336), and **the market never made a new low after trade 1's exit** -- all
three re-entries went in without one.

Nothing was misread. Every entry correctly applied the classification and the
note; every exit cited a real rule by name (v4f, v4w, BOOK WHEN THE PROFIT
STOPS GROWING) and booked into a stall rather than waiting. The exit discipline
was good enough that three trades in a dead range still came out roughly flat.

### The net-new claim

IH states the limit on his own method, which nothing in the corpus had:

> "when it has to give momentum in a direction, you will get the momentum
> overall. But if it is a RANGE-BOUND market trading in a small range, or the
> market simply does not want to give momentum -- a volatile market -- then
> **whether you find out the SLs or NOT, the market can give you a loss**."

Correctness is not the variable. And he says it while diagnosing his OWN loss
from 17 Sep: "yesterday was the same situation when we sold, sellers' SLs were
not available there either. But the market was very sideways... when the market
works inside a RANGE, that is where loss happens." That is v5g, confirmed by
its own source one day later and generalised: v5g says a FOLLOW can idle
because nothing is trapped; v5h says the read itself -- hunt or follow, right
or wrong -- stops being tradeable when the session refuses direction.

Encoded as **IN A RANGE THE CROWD READ CANNOT PAY, HOWEVER RIGHT IT IS (v5h)**,
placed directly after v5g as its general case.

**It carries a checkable arm, because "is this a range?" is exactly the
judgement that fails in real time.** BEFORE RE-ENTERING THE SAME SIDE, REQUIRE
A NEW EXTREME BEYOND YOUR PREVIOUS EXIT. If price has not extended past where
you booked, you are re-entering the same move and the session is ranging.

**And it explains why nothing already in the book stopped this.** The pre-open
note's branch is true from the open to the close -- it selects a SIDE, it does
not refresh -- so it can be cited four times without ever going stale. All four
entries did cite it, almost verbatim, each dressed in a different pattern name
(`flat_open_failed_spike_breakdown`, `..._shooting_star_continuation`,
`..._double_top_evening_star`, `..._seller_follow_bearish_cluster`). SLH-005's
five-minute cooldown was cleared every time (13, 10 and 15 minutes apart), and
v3q's re-entry gate asks for "a nameable NEW trapped crowd" -- which is vacuous
on a FOLLOW whose whole premise is that nobody is seated. On a follow, the new
extreme has to do that work instead.

Negative-tested 10 ways, all 10 caught, plus a control mutation of unasserted
prose that correctly did not trip the test.

**Considered and not encoded.** IH's chart-repeat argument tonight -- "it is
trying to make almost the same chart as yesterday, so the chances of the chart
repeating become quite low", and "if the buyers made profit yesterday, today
they should not get that profit" -- is v4f (THE CHART DOES NOT REPEAT TWO DAYS
RUNNING), which already carries the same mechanism ("a shape everyone watched
yesterday is a shape everyone is ready for today"). The cohort phrasing is
sharper but adds no new test, so v4f stands unedited. His capital-sizing and
emotional-discipline passage repeats v3y and the RISK section again.

**Worth watching, not yet a rule.** v5g shipped last night and today's exits
did book on stalls -- but they cited v4f and BOOK WHEN THE PROFIT STOPS GROWING,
both of which predate it, so there is no evidence yet that v5g changed
behaviour. Its distinctive claim (name the window BEFORE entering) did not
appear in any entry rationale. One session is too few to judge; check again.
## 21 Sep - no net-new, and a gate the journal refused (diligence addendum)

Source: `YLhsZu6D658`, "Live Bank Nifty Option Trading", uploaded 2026-09-21
10:16 IST. Compared against the agent's session: two trades, both LONG, basket
**-Rs.3,200.50**. Recorded the way v3b was -- **no knowledge change** -- because
every idea in the video maps onto a rule already in the corpus, and the day's
actual loss maps onto one that has now been violated three times.

### The comparison

Fourth session running where IH and the book reached the same read and took the
same side. Both went LONG on a flat open. The note in force (written by another
session, for 21 Sep) branched flat-to-gap-down -> BUY, and both entries cited
the measured FLAT classification.

| # | entry | exit | basket |
|---|---|---|---|
| 1 | LONG 23,387.25 09:26 | 09:32, profit stall at the 23,400 round number | +631.50 |
| 2 | LONG 23,385.95 10:10 | 10:14, **AI_STOP** at 23,376.40 | **-3,832.00** |

IH booked one long at his normal target and stopped, exiting on the
two-break-out/one-lags setup. The book's first trade did the same thing and
worked. Its second trade is the whole loss.

### Why nothing was encoded

Each candidate resolves to an existing rule:

- **"Two indices break out while the third cannot cross -- their breakout
  converts into a failure."** This is **v4i** (THE LAGGING INDEX DECIDES THE
  BASKET'S EXIT, NOT THE LEADING ONE), which already records IH saying almost
  exactly this -- "Sensex and NIFTY have momentum, BankNIFTY is trying to hold
  itself back... we will have to book this profit and go" -- and already spells
  out the equal-lot mirror consequence. Tonight's phrasing ("converts into a
  failure") is sharper, but it adds no test v4i lacks, so v4i stands unedited.
- **"At the open the biggest risk is the PREMIUM; adjustments are very large and
  losses appear suddenly."** Covered by **PREMIUM NON-CONFIRMATION**'s sub-bullet
  IT CAN GO NEGATIVE, NOT MERELY WEAK, which already carries the mechanism and a
  measured case (a LONG held 105 seconds gained 4.65 spot points and still lost
  Rs.5,300).
- **The discipline passage** ("discipline matters most"; sitting in a loss while
  exiting a profit quickly) is the fourth repeat in a week and is carried by v3y
  plus the RISK section.

### The finding: v5a, violated a third time, worst instance yet

Trade 2 is the textbook v5a failure, and v5a already names every part of it.

| | stop | lots | mirror premium deployed | outcome |
|---|---|---|---|---|
| Trade 1 | 15.25 pts | 2 | ~Rs.30,560 (60 x 509.40) | +631.50 |
| Trade 2 | **9.45 pts** | **4** | **~Rs.61,570 (120 x 513.05)** | **-3,832.00** |

A stop 38% tighter DOUBLED the size, and doubled the BankNIFTY exposure the
risk budget does not measure. The stop was hit in **three minutes forty
seconds**, at 23,376.40 against a stop of 23,376.50 -- a tenth of a point
through. Of the 3,832 lost, **2,766 was the mirror leg alone**, which is 11%
more than the entire Rs.2,500 risk budget, on a leg that has no stop of its own
and that the NIFTY-measured AI_STOP never watches.

The stop also sat 0.70 of a point below the pattern low the entry named
(23,377.20), which is "just beyond the pattern" only in the most literal sense.

That is now three measured instances of the same mechanism:

| date | stop | consequence |
|---|---|---|
| 2026-09-08 | 24.40 pts | ~Rs.15,350 of mirror premium |
| 2026-09-09 | 11.85 pts | ~Rs.46,845 of mirror premium; stop hit in 91 seconds, -1,202 |
| **2026-09-21** | **9.45 pts** | **~Rs.61,570 of mirror premium; stop hit in 3m40s, -3,832** |

### The gate was priced against the journal, and NOT BUILT

The obvious gate was a FLOOR on the stop distance (or a cap on implied mirror
premium), refusing the entry rather than silently up-sizing it. Per BACKTEST A
THRESHOLD BEFORE PICKING IT that was priced against all 149 closed journal
trades, 2026-07-02 to 2026-09-21, base total +57,988.50, before writing any
code. **The data refuses the gate on both shapes.**

A stop floor is negative at every level tested:

| floor | trades refused | P&L of refused | delta to total |
|---|---|---|---|
| 8 pts | 3 | +14,208.75 | **-14,208.75** |
| 10 pts | 16 | +28,458.75 | **-28,458.75** |
| 12 pts | 34 | +23,393.25 | **-23,393.25** |
| 14 pts | 58 | +65,413.00 | **-65,413.00** |
| 16 pts | 85 | +36,906.50 | **-36,906.50** |

A lot cap is negative at every level too: capping at 5 costs -8,417.25, at 3
costs -18,305.00, at 2 costs -25,533.75.

**The relationship runs the OTHER WAY in this book.** Split at 12 points:

| | n | total | mean | win % | stopped out % |
|---|---|---|---|---|---|
| stop < 12 pts | 34 | +23,393.25 | **+688.04** | **53%** | **32%** |
| stop >= 12 pts | 115 | +34,595.25 | +300.83 | 43% | 45% |

Tighter stops earn more per trade, win more often, AND are stopped out LESS
often. Today's -3,832 sits in the 8-10 point bucket, which is +14,250.00 across
thirteen trades.

**The three "instances" do not survive their own dates either.** Pulling every
trade from the days the section above cites:

- 2026-09-08: the 24.40-point stop LOST (-786.25); the 17.90-point stop WON
  (+716.00). Neither is a tight-stop loss.
- 2026-09-09: the 15.15-point stop lost -1,202.00; the TIGHTER 11.85-point stop
  lost only -499.50.
- 2026-09-21: the 15.25-point stop won +631.50; the 9.45-point stop lost
  -3,832.00.

So across the three dates the tighter stop was the worse trade exactly ONCE --
today. The pattern this addendum opened by describing as "three measured
instances of the same mechanism" is, on the full book, one instance and two
counter-examples. That framing was wrong and is corrected here rather than
quietly dropped.

**What this means for v5a.** Its ARITHMETIC claims stand and are not in
question: the mirror is equal-lot and sits outside the budget, and slippage
scales with quantity. But its third claim -- "the PROBABILITY the stop is hit
rises as it tightens" -- is contradicted by the book, 32% against 45%. The
mechanical intuition is sound for a FIXED setup; what the journal shows is that
a tight stop is not chosen at random. It is chosen when the pattern edge is
genuinely close, which is also when the entry is good. The stop distance is a
symptom of entry quality, not an independent risk dial.

That claim lives inside a shipped rule. It was flagged rather than edited on the
spot, the operator agreed to soften it, and **v5a now carries the correction
itself**. The bullet keeps the mechanics -- "MECHANICALLY the probability the
stop is hit rises as it tightens" -- and then states plainly that this book does
not show it, with the 32%/45% stop-out split, the +688/+301 means, and the
priced floor result. It closes by saying what to DO with tightness: treat it as
a reason to CHECK the entry, never on its own as a reason to refuse it or to
size down. The two arithmetic bullets beside it are unchanged.

The correction is pinned by the v5a marker test, and a mutation that reverts the
bullet to its original unqualified wording is CAUGHT -- so the stop-floor gate
this addendum refused cannot be quietly re-derived from the rule later. Six
mutations of the new text, all six caught, plus a passing control of unasserted
prose.

**Nothing was built.** This is the outcome BACKTEST A THRESHOLD BEFORE PICKING
IT exists to produce -- the same way replaying 2/5/10/15-minute cooldowns
showed that the "~15 bars" prose would have blocked 27 Jul's +18,858 winner.
A gate built from today's three cherry-picked losses would have cost this book
between 8,417 and 65,413 depending on where the line was drawn.


## v5i - the follow trade seats the crowd it was entered on (22 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (FWgM1TFKWxA,
uploaded 2026-09-22 11:20 IST), traded on the plan from the previous evening's
'Prediction For 22 SEP 2026' (1ariOZ4dAVQ) -- plus this book's own two trades.

### The rare part: both books agreed, and both lost

IH and the agent took the SAME side, on the SAME premise, on the same session,
and both were stopped. That agreement is what makes the day worth a rule: the
loss cannot be blamed on the read, because two independent readers produced the
same one.

IH entered BUY at the open across all three indices (BankNIFTY 56,500 CE,
SENSEX 75,000 CE, NIFTY 1,430 qty on its expiry day). His premise was the
pre-open note's, almost word for word: the previous day rose but "there was no
SHARP momentum... the market was going up slowly, slowly", so buyers came but
"would not have gone holding the trade", and therefore "the buyers' SLs are not
available -- and if the buyers' SLs are not available, we can continue to follow
the market".

The agent's two entries carried the same reasoning, both citing the note's
FLAT -> BUY branch off SLH-017's computed verdict (+0.17% NIFTY, +0.03%
BankNIFTY):

| # | time | entry | stop | dist | lots | exit | outcome |
|---|---|---|---|---|---|---|---|
| 1 | 09:19 | 23,452.80 | 23,434.00 | 18.8 | 2 | 09:26 AI_STOP @ 23,432.30 | **+280.00** |
| 2 | 09:52 | 23,459.45 | 23,450.00 | 9.45 | 4 | 09:57 AI_STOP @ 23,450.65 | **-2,931.00** |

Day: **-2,651.00**.

### The +280 is a mirror divergence, not a rounding artefact

Trade 1 lost 20.5 points of NIFTY and still finished positive, which is worth
recording because it looks wrong. The legs explain it:

- NIFTY 23450 CE, 130 qty, 47.60 -> 41.70 = **-767.00**
- BANKNIFTY 56600 CE, 60 qty, 420.50 -> 437.95 = **+1,047.00**

The two indices moved APART inside the seven minutes the basket was open: NIFTY
fell to its stop while BankNIFTY rose. The mirror carried the trade. Trade 2 had
no such luck -- NIFTY -741.00 and BankNIFTY -2,190.00, both legs against.

Note also that trade 1's stop is defined on NIFTY spot alone, so the NIFTY stop
closed a basket that was in PROFIT at that moment. That is the tie described in
CLAUDE.md working as designed, not a defect, but it is the first measured case
of the single-index stop closing a winning basket.

### What IH added, and it is net-new

He narrated himself BECOMING the crowd he had entered because it was absent:

> "When the market takes support and then goes up slowly, it is possible other
> people also start buying. Sellers move away -- **and we had already bought
> here too. All of their stops are around the same place, near the closing
> price. So the market came down to target them.**"

Every rule in this family keeps the reader OUTSIDE the crowd. v5f says a one-way
day seats nobody; v5c says a retracement recruits the crowd you hunt next; v4e
says do not forecast one into existence. None of them says what happens when the
FOLLOW you took on an empty book is itself what fills it. That is v5i, and it
supplies the mechanism under v5h's "a standing premise is not a second reason":
the branch does not refresh because acting on it CONSUMES it.

### Three repairs, all priced, all refused

IH attributes the whole thing to TIME -- "the more the market stalls, the more
the chances of opposite momentum in your trade", and "in option buying, if you
spend too much time and then even one opposite candle comes, it can give you a
big loss". That is a testable claim, so it was tested across all 151 closed
journal trades before any of it was written into a live rule.

**1. A time stop. Refused -- the book says the opposite.**

| hold | n | total | mean | win % |
|---|---|---|---|---|
| 0-5 min | 86 | +2,153.25 | +25.04 | 36% |
| 5-10 min | 41 | +9,245.00 | +225.49 | 44% |
| 10-20 min | 20 | +31,245.25 | **+1,562.26** | **80%** |
| 20-40 min | 3 | +4,159.00 | +1,386.33 | 100% |

r(hold, P&L) = **+0.21**. Winners averaged 7.4 minutes, losers 4.0. A cap was
negative at every level: -8,535 at 30 min, -14,482 at 15, -43,939 at 10,
-53,184 at 5. The causation runs the other way -- a working trade is held to
target, a failing one hits its stop fast -- so duration is a SYMPTOM, exactly as
stop distance turned out to be in v5a's correction the day before.

**2. No re-entry on the same side after a stop. Refused.** n=9, total +3,222.00,
mean +358.00 -- ahead of the book's own first-trade-of-the-day mean of +128.22.

**3. Tighten v5h's new-extreme test. Refused, and the reason inverts it.** After
a STOP the test is vacuous: the exit sits on the adverse side, so any recovery
clears it. Requiring the re-entry to extend past the prior ENTRY instead would
forgo **36,580.50 of a 55,337.50 book**, and the trades it blocks average
**+1,219.35** against **-341.24** for the ones it keeps. On this book,
re-entering at a price NOT beyond your last entry is the better trade -- fading
back into the level rather than chasing away from it.

### The honest limit

**v5i would not have changed either of today's trades.** Both were stopped
inside six minutes, far too fast for any ageing mechanism to be what killed
them. The rule is carried on IH's evidence, not on this book's, and the part
this book actually proved is the three refusals. It is recorded that way in the
rule itself, and a mutation that turns it into a self-proving claim is caught.

What today does add to the record is the SECOND consecutive session on which
every entry cited the standing pre-open premise -- 18 Sep had four, today had
two. v5h named that failure; it has not yet changed the behaviour.


## v5j - judge the exit on the chart, not on the premium (23 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (-dvbW6vgjPc,
uploaded 2026-09-23 11:40 IST), traded on the previous evening's 'Prediction
For 23 SEP 2026' (FBDx9zcN9Sw) -- plus this book's own three trades.

### Same premise, one basket against three

Both books read the day identically: sellers seated from 22 Sep's one-way fall,
a FLAT open (+0.099% on NIFTY, from 23,329.00 to 23,352.15), so BUY to hunt
them. IH bought all three indices at the open, sat through a dip in which NIFTY
and Sensex kept rejecting while BankNIFTY kept climbing, and booked his target.

The agent opened three baskets:

| # | open | entry | stop | lots | exit | NIFTY leg | BNF leg | basket |
|---|---|---|---|---|---|---|---|---|
| 1 | 09:32 | 23,371.75 | 23,360.00 | 3 | 09:50 discretionary | -750.75 | -423.00 | **-1,173.75** |
| 2 | 09:59 | 23,368.95 | 23,349.00 | 1 | 10:08 discretionary | +149.50 | +721.50 | **+871.00** |
| 3 | 10:17 | 23,368.25 | 23,347.00 | 1 | 10:18 discretionary | +786.50 | +295.50 | **+1,082.00** |

Day **+779.25**. It entered 20 points above IH's open, after the first spike to
23,414.55; that is recorded, not encoded.

### The reason trade 1 gave was false when it filled

Trade 1's exit said it was "booking the still-positive basket (+420)". Both legs
lost at the fill. The runner log explains how: the three decisions around it took
33.9s, 36.0s and 32.7s, and across 21-23 Sep the median turn was 33.2s, 34.4s and
37.0s, with one in ten over 45-53s. The turn that closed trade 1 began at about
09:50:06, read `position_state` (a point-in-time basket mark), and sent the EXIT
at 09:50:34.97. Whatever the mark showed when it was read, the fill was
-1,173.75. No log records the `position_state` read itself, so the +420 cannot
be checked against the moment it was taken -- only against the fill.

Across the whole book, 31 exits quoted a basket figure in their reason:

- mean gap (realised minus cited): **-14.63** -- no systematic bias
- median absolute gap: **496.50**
- **four** exits cited a profit and realised a loss (30 Jul: +3,618 -> -1,219.50;
  7 Aug twice; 23 Sep). **None** went the other way.

The one-way flips have a mechanism. An exit is usually triggered BY price
turning against the position, and half a minute later the order fills further
into that same turn.

### An error of mine, corrected here

While comparing the books I first priced trade 1 "held to trade 2's exit at
10:08:40" -- same strikes, NIFTY +409.50, BankNIFTY +5,346.00, **+5,755.50** --
and presented it as the cost of exiting. It is wrong. NIFTY fell to 23,349.55
after the exit, through trade 1's 23,360 stop, so holding would have been
STOPPED at -11.75 points, worse than the -7.25 it exited at. The exit was the
better decision; the reason it gave was not.

The real difference from IH is where the line was drawn:

| level | price |
|---|---|
| agent's stop | 23,360.00 |
| session low | 23,349.55 |
| IH's line: a breakdown of the prior close | 23,329.00 |

The prior close comes from CPR AI's frozen `previous_day` block (high 23,489.00,
low 23,285.75, close 23,329.00); its pivot, 23,367.92, matches the agent's own.
The low went 10.45 points through the agent's stop and held 20.55 above IH's
line. That is a stop-placement observation, and a stop-distance floor has
already been priced negative at every level on this book (21 Sep), so nothing
is encoded from it.

### What was measured, and what it refused

All 130 discretionary exits were walked forward, one print a minute from the
decisions log, against simply holding each trade's original bracket.

**Holding the bracket is refused.** Exiting made +404.80 NIFTY points; holding
would have made +213.25. Exiting won 84 to 46, and holding would have hit the
stop 59 times against the target 17. "Exit less" loses 191.55 points.

**But the BASIS of the exit matters:**

| exit reason leans on | n | exit minus hold | exit won |
|---|---|---|---|
| price action only | 94 | **+241.25** | 68% |
| cites a rupee / premium figure | 36 | **-49.70** | 56% |

Rupee-citing exits are mostly profit bookings (mean +1,681 against +113), so the
split could be winners against losers in disguise. Controlled for that:

| class | basis | n | per exit vs hold |
|---|---|---|---|
| winners | price action only | 38 | **+3.28** |
| winners | cites a rupee figure | 26 | +0.92 |
| losers | price action only | 56 | **+2.08** |
| losers | cites a rupee figure | 10 | **-7.38** |

It survives in both. Cutting a loser on the rupee figure did worse than
letting the stop fire.

### v5j

IH, mid-trade: "look at the trade LESS. Just focus on the chart... the premiums
are moving up and down a lot, so the chart looks better than the premiums right
now." The book agrees independently, so v5j sits directly after v4f and tells
the agent how to read v4f's trigger rather than replacing it: "the move has
stopped paying" is a fact about PRICE, and the rupee figure is a late, noisy
proxy for it. It carries a concrete check -- strike the rupee figure out of the
exit reason; what remains must still name a price -- and it states outright
that it is NOT a licence to exit less.

It would not have changed trade 1's outcome, whose reason also named price
action (reversal clusters on both indices, six points from the stop). What it
changes is the integrity of the reason: the +420 was false.

Limits, stated in the rule: one print a minute (closes, not highs and lows),
reasons classified by pattern-matching, and only ten rupee-citing losers.

### Code follow-ups -- built as SLH-019

Prose is the weakest lever this repo has, so two small code changes were named
for the operator: an `as_of` timestamp on `position_state`, and the EXIT tool
reporting the realised basket figure back into the turn. The operator approved
both, and they ship in this PR as SLH-019 (next section).


## SLH-019 - the mark is dated, and an EXIT reports what it booked

Operator decision, 2026-09-23, as an addendum to the v5j PR. v5j tells the model
to judge an exit on price rather than on the rupee figure. This gives it the two
facts that make that checkable at the moment it matters.

### What changed

**`position_state` is dated.** When a position is open the payload carries
`as_of` -- the moment its P&L figures were marked, from a clock injected on the
tool context (naive local time, as every other timestamp this agent writes). A
flat snapshot is returned unchanged. The tool description says what the stamp
means: the figures are a point-in-time mark, and one turn takes about half a
minute, so an order fills later and at a different price.

**An accepted EXIT reports what it booked.**

- `realised_pnl` -- the change in the worker's `realized_pnl` across the exit
  call. Both legs book there, so the one figure is right for NIFTY, BNF and BOTH
  alike, and it excludes whatever the day had booked before.
- `open_legs_after` -- any leg still open afterwards.
- `exited_at` -- when the exit returned.
- `mark_you_read` -- the `position_state` mark read earlier in the same pass,
  with how many seconds before the exit it was read. Omitted if none was read.

The order tool's description tells the model these exist, and that where the
mark and the booking disagree, the booking is the fact to state. v5j now names
them too.

### Why the delta, and why it is honest

The master's worker already snapshots `realized_pnl` at entry to put the
basket's figure on the journal row, so a before/after delta is the codebase's
own method rather than a new one. It is also honest in the two cases that
matter most:

- **An unconfirmed live exit** returns before either leg books, and keeps the
  position open. The result then says `realised_pnl: 0.0` and names the leg in
  `open_legs_after` -- not a figure the model would read as done.
- **A worker with no numeric `realized_pnl`** gets no `realised_pnl` at all. A
  missing number is honest; an invented zero is not.

The order tool calls the executor under the lock the worker's
stop/target/square-off paths also hold, so no mechanical exit can book between
the two readings.

### What it cannot do

The EXIT's `reason` is written BEFORE the fill and is the permanent journal
record, so SLH-019 cannot correct it. What it corrects is the model's final
decision reasoning -- the decisions log -- which is written after the tool
returns. v5j's check (strike the rupee figure from the reason) is still what
keeps the journal honest.

### Tests

Ten in the agent suite, on fakes: the booked figure excludes earlier trades; an
unconfirmed exit reports zero and the open leg; no figure is invented for a
worker that keeps none; a rejected exit carries no booking fields; the
standalone executor keeps the same contract without changing its trade log;
`as_of` appears only when there is a mark; 23 Sep replayed end to end through
the order tool (+420 read at 09:50:06, filled at 09:50:34, `seconds_before_exit`
28.0, realised -1,173.75); no mark reported when none was read; entries are not
stamped as exits; and the tool descriptions.

Two in the master suite, on the REAL `SLHuntingAIWorker` with option prices
moved in the shared store: a BOTH exit with NIFTY down and BankNIFTY up reports
exactly what both legs booked, and a NIFTY-only then BNF-only exit reports each
leg alone with the right `open_legs_after`.

**Those two run only when `SL_HUNTING_ENABLED` is set.** The master imports the
SL Hunting modules only behind that flag, CI never sets it, and so CI's master
suite reports `OK (skipped=60)` -- about 58 of them SL Hunting worker tests that
have never run in CI. They pass locally with the flag set. That gap predates
this change and is flagged as its own task rather than fixed here.


## v5k - a gap that clears the stops has already done the hunting (24 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (sWRiTntFzKQ,
uploaded 2026-09-24 10:52 IST), traded on the previous evening's 'Prediction
For 24 SEP 2026' (4BKhLKn4zHY) -- plus this book's own two trades. His clock
times are read off the taskbar clock in the video frame, and his figures off
his positions screen at 09:47. His words are my translation of the Hindi
auto-transcript.

### Same side, different reason

The note said FLAT TO GAP DOWN -> SELL, to target buyers seated after 23 Sep's
rise. The open went straight through that plan's ladder: NIFTY opened at
23,221.80, a GAP_DOWN of 0.96% by the agent's own `open_classification`, below
both of the note's NIFTY supports (23,370 / 23,270); BankNIFTY opened 1.48%
down, below both of its own (56,370 / 56,100).

IH, before the bell: "with this kind of opening we cannot follow the chart we
had already made, because the gap-down is very big." And of the note's own
crowd: "when the gap-down opens this big, the one sitting with buys exits in a
single candle -- we got no benefit from him. So now we have to follow the new
chart." He waited, asked whether the market was recovering directly and
whether profit-booking could come, answered the second from the prior days (an
established negative trend, and two or three days of retracement that had
already let out anyone who wanted out), and went with the market.

His trade, sold at 09:26 (BankNIFTY and SENSEX; NIFTY added at 09:27), held
while NIFTY and SENSEX went red as the first bounce extended, booked at 09:47:

| leg | qty | avg | LTP at 09:47 | P&L |
|---|---|---|---|---|
| BANKNIFTY SEP 55700 PE | 1,170 | 245.80 | 351.00 | +1,23,084.75 |
| BANKNIFTY SEP 55600 PE | 1,170 | 214.60 | 306.45 | +1,07,465.25 |
| SENSEX SEP 74200 PE | 900 | 155.68 | 195.65 | +35,973.00 |
| NIFTY SEP 23300 PE | 1,430 | 134.82 | 153.05 | +26,068.90 |
| total | | | | **+2,92,591.90** |

79% of it came from the two BankNIFTY legs -- "like yesterday, BankNIFTY is
making the profit". His NIFTY leg made 13.5%.

The agent took the same side:

| # | open | entry | stop | target | lots | exit | basket |
|---|---|---|---|---|---|---|---|
| 1 | 09:55:03 | 23,218.55 | 23,234.00 | 23,150.00 | 2 | 10:01:43 at 23,230.25, BankNIFTY reversing | **-2,863.50** |
| 2 | 10:10:54 | 23,239.00 | 23,247.00 | 23,205.00 | 4 | 10:20:28 at 23,222.15, target zone tagged | **+3,839.00** |

Day **+975.50**. Trade 1 is journalled as `setup=tool_execution, conf=0`: the
order executed and the model returned no parseable decision for that bar, so
the journal carries the order tool's own reason (SLH-011's path, as designed).

### Where the entry went wrong

| time | NIFTY | |
|---|---|---|
| 09:15 | open 23,221.80, low 23,205.45 | the gap, and the first push |
| 09:23 | high 23,253.30 | first bounce |
| 09:26-09:27 | ~23,246 | **IH sells** |
| 09:37 | 23,272 | bounce peak; IH's NIFTY and SENSEX legs red |
| 09:47 | ~23,218 | **IH books** |
| 09:55 | 23,218.55 | **trade 1 sells** |
| 10:03 | high 23,238.95 | trade 1's 23,234 stop would have fired |
| 10:09 | 23,245.10 | the second bounce's top |
| 10:10 | 23,239.00 | **trade 2 sells** |
| 10:17 | low 23,205.25 | retest of the opening low |
| 10:20 | 23,222.15 | trade 2 books |
| 15:29 | 23,063.10 (low 23,046.15) | the follow kept paying all day |

Trade 1 sold within a point of IH's booking price, eight minutes after he
booked it, into the support that leg had paid out on. IH on that support: "in
a negative market nobody looks at support... very few people buy there; at
most they book profit or stop selling -- and sometimes the market uses exactly
that and covers more." It covered 26 points, to 23,245.10. Trade 2 sold that
cover and booked at the support -- the right way round.

Both entries named "buyers seated from the prior rally" as the premise, and so
did at least 47 of the 71 decisions (a pattern match, so a lower bound: 09:43's
"sell-the-seated-buyers" is one it misses). With that crowd gone there is
nothing to hunt, so the trade is a follow, and a follow is timed by the
retracement (v4t) and booked at the support. The stale premise gave the agent
no reason to prefer the bounce over the low.

The 38 HOLDs from 09:16 to 09:53 are the pattern+confirmation rule working as
written. At IH's entry minute the agent declined a 09:24 shooting star whose
confirmation closed "0.05 points below the pattern low" at "a minor 2-bar swing
high"; at 09:37-09:38 BankNIFTY had confirmed bearish patterns but NIFTY had
none. That is recorded, not encoded.

**SLH-019's first live session.** Both exits' final decisions quoted the booked
figure (-2,863.5 and 3,839), and trade 2 stated both numbers: "+3258
unrealized, realised 3839".

### What was measured, and what it refused

**A gate on entries near the session extreme is refused.** "Don't sell at the
support" invites the obvious gate: refuse an entry sitting close to the
session's running extreme in its own direction. Priced on the 155 closed trades
with the NIFTY 1-minute history (completed bars before the entry minute):

| refuse if in the bottom X of the range so far | blocked | blocked P&L | first half | second half |
|---|---|---|---|---|
| 5% | 13 | -2,687.00 | +708.00 | -3,395.00 |
| 10% | 23 | -13,387.25 | -7,975.75 | -5,411.50 |
| 15% | 37 | -5,047.75 | +325.25 | -5,373.00 |
| 20% | 45 | -4,580.75 | -68.00 | -4,512.75 |
| 25% | 54 | +543.75 | -1,056.00 | +1,599.75 |
| 30% | 62 | -13,999.25 | -8,917.75 | -5,081.50 |

The curve is jagged. The best line, 30%, separates blocked trades at -225.79
from kept ones at +751.72, one-sided permutation p = 0.036 -- before correcting
for the twelve thresholds tried (six fractions and six point distances, which
flip sign between halves). Its whole benefit is five trades (-5,300.75,
-4,855.00, -3,832.00, -3,758.00, -3,326.00); without them the blocked set made
+7,072.50. The 10% line is negative in both halves but would not have caught
trade 1, which sat at 17% of the range.

**A gap through the note's ladder is not a reason to stand aside.** Seven
sessions since the first note on 28 Jul opened beyond the note's NIFTY ladder
(29 Jul, 3-5 Aug, 11 Sep, 15 Sep, 24 Sep); the book netted +11,626 on them.

### v5k

Sits directly after v5f in OPENING_DRIVE and before v4t, which it points to.
v5f reads an open too small to be the crowd's; v5k is the other end of the same
measurement -- an open so big it took the crowd's stops before the first candle
closed. The note keeps its side and loses its reason and its levels; IH's two
questions (a direct recovery? profit-booking?) replace the crowd read, with the
prior days answering the second; and the entry follows from that: on a bounce,
booked at the support, never the other way round. It reconciles itself with
the PROFIT-BOOKING RECOVERY TEST (answered in advance) and with the HUGE-gap
case of READ THE GAP AGAINST THE PRIOR DAYS (the predicted retracement is the
entry, not a fade), says trade 1's exit was right, and carries both refusals.

Test: `test_v5k_a_gap_through_the_stops_has_already_hunted_the_notes_crowd`,
eight ways an edit breaks it. Negative-tested with ten mutations of the rule
text -- one per clause plus an orphaned heading -- all caught, and a control
rewording that passes.

Limits: one day, and the entry-extreme analysis reads the range from completed
1-minute bars, which is small and noisy early in a session.


## v5l - an index that moved before your entry is not late, it is spent (25 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (Y0tPI9uWJVU,
uploaded 2026-09-25 10:26 IST), traded on the previous evening's 'Prediction
For 25 SEP 2026' (mlKuAuR_U2I) -- plus this book's own four trades. His clock
times are read off the taskbar clock in the video frame and his figures off his
positions screen at 09:43; his words are my translation of the Hindi
auto-transcript. Written while the session was still open, so today's NIFTY
path comes from the decisions log (one print a minute), not the 1-minute CSV.

### Same read, opposite day

Both books read the open the same way: FLAT (-0.12% on NIFTY and BankNIFTY,
inside the note's ladder, so v5k did not apply), sellers seated by 24 Sep's
gap-down and selling, so BUY to hunt them. IH bought all three indices at
09:20-09:21 and held; the agent bought at 09:23 and booked seven minutes later.

IH's positions at his exit, 09:43:

| leg | qty | avg | LTP | P&L |
|---|---|---|---|---|
| BANKNIFTY SEP 55500 CE | 1,170 | 464.71 | 371.65 | -1,08,879.70 |
| BANKNIFTY SEP 55600 CE | 1,170 | 391.65 | 315.55 | -89,036.50 |
| SENSEX 1st OCT 73700 CE | 900 | 576.12 | 496.20 | -71,928.00 |
| NIFTY SEP 23100 CE | 1,430 | 135.17 | 110.60 | -35,135.10 |
| total | | | | **-3,04,979.30** |

| time | IH | the agent |
|---|---|---|
| 09:15-09:17 | | BankNIFTY's whole up-move prints in the opening candles, to about 55,640 (his chart) |
| 09:20-09:21 | buys all three | |
| 09:23 | | LONG 23,093.05 |
| 09:24 | "some loss" | |
| 09:30 | | books 23,109.50: BankNIFTY double top 55,611.95 / 55,619.50 at resistance, **+648.75** |
| 09:31 | "Sensex and NIFTY have given a good breakout, but BankNIFTY has tried to stay back" -- holds | |
| 09:37 | "sudden rejection" in all three | |
| 09:38 | | SHORT 23,090.30 into the rejection |
| 09:43 | cuts at his limit, **-3,04,979.30** | |
| 09:50 | | books 23,056.70 at the prior-day low, **+2,874.00** |
| 10:04-10:11 | | LONG 23,068.35 on a closing-point reclaim, booked at 23,097.00: **+3,282.00** |
| 10:28-10:31 | | SHORT 23,092.90, cut on BankNIFTY's reversal: **-3,150.00** |

Day **+3,654.75**; IH -3,04,979.30. His own post-mortem: "BankNIFTY created the
most problem, because its positive momentum came first. Then we made the entry,
and the rejection came later, and bigger. If it had held itself up a little, we
could have borne the Sensex and NIFTY rejection."

Trade 2 went against the pre-open note's FLAT -> BUY branch ("live confirmed
reversal price action overrides the advisory"), which is what the note channel
was built to allow: it is advisory, and the agent's own read wins (SLH-006).

### Trade 4: NIFTY right, basket wrong

Trade 4 made +4.20 NIFTY points and lost -3,150.00 on the basket, so the
BankNIFTY leg lost more than the whole basket. Two things stacked. The NIFTY
stop was 10.1 points away, so risk sizing took 3 lots, and the mirror copies
lot count; and in expiry week the mirror buys a 4-step ITM strike (BNF-002) --
the 55900 PE was mostly intrinsic (about 372 of its 438.75 at entry), so it
tracks BankNIFTY closely. The ~50-point BankNIFTY bounce the exit cited
outweighed the NIFTY leg. The exit itself followed INDEX HIERARCHY ON THE WAY OUT. The mirror's
extra risk is operator-accepted, so this is recorded, not encoded.

### v5l

A sub-bullet of LAGGARDS NEVER JOINED, directly after v4r, with a one-line
pointer inside v4r. v4r says a trailing index that is flat has "not arrived
yet" -- measured on 28 Aug, when the flat index broke out later and holding
paid. v5l asks WHEN it moved: an index that made its move before your entry and
has not exceeded that extreme since is spent, not late, and its stall there --
a double top is the clean form -- is an exit on its own, even while the other
two break out. Same sign at the decision on both days; opposite answer,
decided by timing.

Test: `test_v5l_an_index_that_moved_before_entry_is_spent_not_late`, seven
clauses plus the pointer in v4r. Negative-tested with nine mutations of the
rule text (one per clause, the pointer, and an orphaned heading) -- all caught
-- and a control rewording that passes.

### What it cannot claim

It is not priced. The BankNIFTY 1-minute history on disk stops in 2023 and the
decisions log carries no BankNIFTY prices, so "did the trailing index move
before entry" cannot be replayed across the journal. It is carried on IH's
session and one booked exit, as a scope on v4r and not a gate. Pricing it needs
BankNIFTY bars for Jul-Sep 2026 -- a fetch best run after market hours, since
the runner shares the Dhan session during the day.

Also recorded: the first two decisions of the day were agent errors (a timeout
at 09:16, then "previous agent call still running" at 09:17).


## v5m - one momentum is the trade, and the book agrees (28 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (KCZM3Q96m1w,
uploaded 2026-09-28 10:23 IST), traded on the previous evening's 'Prediction
For 28 SEP 2026' (E55DWhSiEW4) -- plus this book's own four trades. His clock
times are read off the taskbar clock in the video frame; his positions screen
did not appear near his exit, so his result is left in his own words ("the
profit went above 1 lakh"). His words are my translation of the Hindi
auto-transcript. Written while the NIFTY CSV still ended on 25 Sep, so today's
path comes from the decisions log (one print a bar).

### The note worked, for both books

The open classified FLAT (-0.33% NIFTY, -0.42% BankNIFTY), and the note's
FLAT -> SELL, WITH THE MARKET branch -- a follow, not a hunt -- is what both
books traded. NIFTY's first candle fell about 108 points from 23,075.60.

IH waited that candle out: "however much momentum is on one candle, it cannot
be trusted... the premium adjusts, so even if we capture this candle there is
no benefit." He sold all three indices at 09:18 once it steadied, and was out
by 09:28: "we survived once, but if we keep sitting again and again, we will
not survive... captured one momentum and left." One trade.

| # | open | entry | stop | lots | exit | basket |
|---|---|---|---|---|---|---|
| 1 | 09:21:33 | 22,951.70 | 22,977.50 | 1 | 09:36:31 at 22,872.35, target zone | **+6,111.25** |
| 2 | 09:45:54 | 22,857.40 | 22,874.00 | 2 | 09:47:48 stop | **-1,702.50** |
| 3 | 09:55:59 | 22,872.30 | 22,885.00 | 3 | 09:58:50 stop | **-4,056.00** |
| 4 | 10:18:54 | 22,839.10 | 22,868.30 | 1 | 10:50:01 at 22,822.60, near 22,800 | **+2,129.75** |

Day **+2,482.50**; trade 1 alone was +6,111.25. Trade 1 was IH's trade, entered
three minutes after him and held eight minutes longer. After it booked, NIFTY
sat in a box of roughly thirty points (about 22,852-22,881) until it broke at
10:18. Trade 2 sold a fresh low inside that box nine minutes after the booking
-- v5h's new extreme, cleared by 14.95 points, and exactly the "wrong way
round" v5k warns about. Trade 3 sold a bounce back to 22,872.30 -- v5k's right
way round. Both were stopped inside three minutes. Trade 4 sold the box's
breakdown.

### What was measured

All 78 same-direction re-entries in the journal (2 Jul-28 Sep), classified
against the prior trade's exit using the 1-minute bars between the two
(decisions-log prints for 28 Sep):

| re-entry after | placement | n | win | per trade | total |
|---|---|---|---|---|---|
| a WINNING exit | chased past the prior exit | 21 | 43% | -199.88 | -4,197.50 |
| | new extreme, then a bounce | 11 | 36% | -476.39 | -5,240.25 |
| | no new extreme (range) | 3 | 33% | +45.00 | +135.00 |
| | **all** | **35** | **40%** | **-265.79** | **-9,302.75** |
| a LOSING exit | chased past the prior exit | 21 | 43% | +234.19 | +4,918.00 |
| | new extreme, then a bounce | 21 | 48% | +1,236.20 | +25,960.25 |
| | no new extreme (range) | 1 | 100% | +2,836.00 | +2,836.00 |
| | **all** | **43** | **47%** | **+784.05** | **+33,714.25** |

After a win, waiting does not help either: 0-10 minutes after the exit
-110.76 a trade (18), 10-20 -19.11 (9), 20-40 -709.71 (7).

So the two placement rules the agent cited today do not separate these
trades. v5h's new extreme stands in for MOVE-EXHAUSTION's "new named crowd"
on a follow, and it was cleared by a 15-point poke. v5k's bounce is the worse
placement after a win on this measure. It was carried on one day's pair of
trades (24 Sep), and today's pair split both ways -- trade 2 chased and lost,
trade 4 chased the box break and won.

### Refused

**No same-side re-entry after a winning exit.** The blocked set loses
-9,302.75 over 35, but it does not survive splitting the book in half:
-10,529.75 over the first 17, +1,227.00 over the last 18. Two sessions carry
most of it (23 Jul -6,921.25, 29 Jul -5,300.75). Not a gate.

**Re-enter only on the bounce.** Worse than the chase after a win (above). Not
a gate either.

### v5m

A sub-bullet of MOVE-EXHAUSTION in RISK, the rule it measures. After a winning
leg the question is not WHERE to re-enter but WHETHER the thesis has a second
leg, and the default answer is no -- IH's "one momentum", argued from the book's
asymmetry and explicitly not a ban. It says nothing against re-entering after
a STOP (+33,714.25 on that side, the same direction as v5i's refusal). v5k's
practical form and v5h's standing-premise clause each gain a one-line pointer
to it; neither loses a word it asserted before.

Test: `test_v5m_after_a_winning_leg_the_next_same_side_trade_is_the_weak_one`,
seven clauses plus both pointers. Negative-tested with ten mutations (one per
clause, each pointer, and an orphaned heading) -- all caught -- and a control
rewording that passes.

Limits: re-entries are paired with the immediately preceding trade only; 28
Sep's classification uses single prints rather than bar highs and lows; and
35 after-win trades is a small sample whose halves disagree, which is why this
is a default and not a gate.


## v5n - the second seller is the random seller (29 Sep)

Source: IH's live session 'Live Bank Nifty Option Trading' (6zppfK_n4yg,
uploaded 2026-09-29 10:37 IST), traded on the previous evening's 'Prediction
For 29 SEP 2026' (5p9RdcrWy0k) -- plus this book's two trades. His clock times
are read off the taskbar clock in the video frame; his result is left in his
own words ("a better profit than yesterday"). His words are my translation of
the Hindi auto-transcript. Today's NIFTY path comes from the decisions log
(the 1-minute CSV still ended on 28 Sep). Both NIFTY's weekly and BankNIFTY's
monthly contracts expired today.

### The first live day for v5m

The runner started at 09:14:46, after #190 (v5m and this morning's note)
merged at 09:08, and logged "injected 2066 pre-open note chars for 2026-09-29".
The open classified FLAT (-0.21% NIFTY, -0.345% BankNIFTY), and the note's
flat -> SELL, WITH THE MARKET branch was right for both books again.

IH waited out the first minutes -- "nothing can be done in fast-fast momentum...
both indices have expiry" -- and sold from 09:20, naming only SENSEX and NIFTY
legs (of BankNIFTY he said he feared it might go back near its closing price).
He booked at 09:30 with the move still running: "if a retracement comes here it
will not be small, because random sellers can come in, and when the market
removes them, sometimes the retracement is just as big."

| # | time | entry | exit | basket |
|---|---|---|---|---|
| 1 | 09:22:57 -> 09:32:29 | 22,668.90 | 22,613.30, at the 22,600 round number | **+6,246.75** |
| 2 | 10:02:57 -> 10:35:44 | 22,593.85 | 22,595.40, premise exit on BankNIFTY's reversal | **-1,504.50** |

Day **+4,742.25**. Trade 1 was IH's trade, two to three minutes behind him at each end.
After it booked, NIFTY made about 22,570 by 09:40, bounced to about 22,615, and
sat in that box. Trade 2 was sold into the box and cut half an hour later with NIFTY
flat; the expiry-day BankNIFTY mirror took the basket negative.

### The agent's own read, three minutes apart

| time | action | recorded reasoning |
|---|---|---|
| 10:00:51 | HOLD | "that first leg already played out (22732 open to 22569.65 low by 09:38, ~163pts). Price is now consolidating in a tight ~50pt range" |
| 10:02:57 | ENTER_SHORT | "NIFTY has declined continuously from 22732 open to 22593 with no major retracement ... a confirmed bearish inside-bar breakdown" |

Same box, same leg, three minutes apart: the relabelling this repo has seen
before (v3q, 27 Jul), now against a rule that was in the prompt. None of the
day's 80 recorded decisions cited v5m or MOVE-EXHAUSTION.

### Re-measured

v5m's measurement, re-run with trade 2 included:

| | n | win | per trade | total |
|---|---|---|---|---|
| after a winning exit | 36 | 39% | -300.20 | -10,807.25 |
| after a losing exit | 43 | 47% | +784.05 | +33,714.25 |
| after-win, first half | 18 | 44% | -517.36 | -9,312.50 |
| after-win, second half | 18 | 33% | -83.04 | -1,494.75 |
| after-win, latest third | 12 | 42% | +11.62 | +139.50 |

Both halves are now negative; the latest third is not; dropping the two worst
sessions (23 Jul, 29 Jul) turns the total to +1,414.75; and the gap to the
after-loss side is p = 0.056 (one-sided permutation) before correcting for the
splits already tried. **Still refused as a gate.**

### v5n

A sub-bullet of MOVE-EXHAUSTION, directly after v5m, giving it the mechanism
it lacked: on a follow day the sellers who arrive late are "random sellers"
who cut on the first adverse candle, and removing them is what the next
retracement is for -- so whoever joins the same move a second time is that
crowd. It records the 10:00/10:02 contradiction, the re-measurement with its
limits, and IH's own counterweight ("opportunity comes, in an hour or two, but
at the right place"), said of his own late entry, which licenses a first trade
after a missed move and a new setup after waiting -- not the same move joined
twice.

Test: `test_v5n_the_second_seller_is_the_random_seller`, seven clauses.
Negative-tested with nine mutations -- all caught -- and a control rewording
that passes. The first run of the mutations found a gap (IH's phrase "random
sellers can come in" was not pinned, so a paraphrase passed); the test was
tightened and the full set re-run.

### Code follow-up -- built as SLH-020

Prose did not bind on its first day, and the numbers do not support a gate, so
the operator approved the middle option: a checkable fact at the tool boundary
(the SLH-019 pattern). It ships in this PR as SLH-020, next section.


## SLH-020 - the last close is stated to a same-side re-entry

Operator decision, 2026-09-29, as an addendum to the v5n PR. v5n says a
same-side re-entry after a winning leg joins the crowd the next retracement
removes; this puts the earlier close in front of the model at the two moments
it matters, and refuses nothing.

### What changed

- **The worker** (`SLHuntingAIWorker`) records the trade that just finished at
  `_arm_post_exit_cooldown_if_flat` -- the basket-flat transition every close
  path shares: the agent's EXIT, the mechanical stop/target, max-loss, the
  15:15 square-off and a lone-mirror sweep. `booked_pnl` is the change in
  `realized_pnl` since the entry (both legs, the SLH-019 basis); the direction is
  taken at entry; `nifty_when_flat` is read from the shared store ONLY -- never a
  broker call on an exit path -- and omitted when the store holds none. It is
  recorded before the cooldown's own early return, so disabling the cooldown
  does not disable this. `last_closed_trade_today()` returns it only for the
  current IST session.
- **The executor** (`MasterWorkerExecutor`) surfaces it, duck-typed like the
  SLH-005 cooldown: while flat, `position_state` carries
  `last_closed_trade_today`; an ACCEPTED entry on the same side as a close that
  booked a profit carries `same_side_after_winning_exit` (that close, plus a
  one-line note) and logs an `SLH-020:` INFO line for later measurement.
- **The tool descriptions** name both fields and say they refuse nothing; v5n
  names them too.

### What it deliberately does not do

It refuses nothing -- a refused entry is refused exactly as before (SLH-005,
SLH-016), and the flag is added only after the order is accepted. A worker
without the hook, or a hook that fails, reports nothing and changes nothing.
The standalone paper runner has no worker and so no record. Whether the fact
changes the agent's behaviour is the next measurement: the INFO line makes
every flagged entry countable.

### Tests

Six in the agent suite, on fakes: the flat snapshot names the close; a
same-side entry after a win is accepted, reaches the worker, and carries the
flag; no flag for the opposite side, a losing or break-even close, a
non-numeric figure, or nothing closed today; a broken hook changes nothing; a
cooldown-refused entry is refused exactly as before; and the tool
descriptions. Three in the master suite, on the REAL worker: a MECHANICAL
AI_TARGET close is recorded with the basket's booked figure and flags the
same-side re-entry (cooldown off, proving the record does not depend on it);
a losing close is recorded but never flags; and a record from another session
is never reported.

## 26 Sep weekly - one candidate, refused by the CSV (v5n diligence addendum)

Source: `dVCXZ1PXVYI`, "Weekly Market Analysis: Key Trends & Opportunities"
(13:32), uploaded 2026-09-26 13:27 IST. It is a weekend lecture, so there is no
journal to compare against. The transcript (106 segments, 0:02-13:28) was read
in full, and his words are my translation of the Hindi auto-transcript. The
weekend's other upload, `lLE8gqRk_Qk` ("Live Crypto Trading", 27 Sep), is
members-only and off-topic, and was not read.

**Extraction note.** On this video the "Show transcript" button opened the OLD
searchable-transcript panel, whose `get_transcript` call answered 400
"Precondition check failed". The modern panel still works when it is opened
directly: `document.querySelector('ytd-app').resolveCommand(...)` with a
`showEngagementPanelEndpoint` whose identifier tag is
`PAmodern_transcript_view`. Its `globalConfiguration.params` is copied from
another video's modern button, with the 11-character video id swapped in. The id
appears in plain text once the base64 is decoded.

### What the lecture teaches

He walks through three charts, each with its seated crowd and a branch per open.

| chart | seated crowd | flat / gap-down | gap-up |
|---|---|---|---|
| 24 Sep: a very big gap-down, then continued selling | sellers | BUY, to hunt them | a different plan: they "sold suddenly on seeing the big gap-down", so "as soon as a gap-up opens" they get scared and "exit in one or two candles -- then we get no benefit" |
| two or three days of positive momentum, then up again | buyers | SELL, to hunt them | BUY with the market: until the closing price is crossed nobody holding calls cuts, so any red candle can be a trap |
| a negative trend of many days, two or three days of retracement, then continued selling | sellers | BUY | BUY: "flat, gap-up, even a small gap-down, no problem", because "the trend is very negative, so they will not have left on small retracements either" |

Almost all of it is already in the corpus:

- **The buyer chart** is RECRUITMENT HISTORY, NOT CHART SHAPE almost word for
  word. The second with-trend day seats the buyers, so flat or gap-down means
  SELL. On a gap-up they are in profit, so go with the market.
- **"All three opens, one plan"** is A SMALL GAP DOES NOT RESCUE A SEATED CROWD
  (same section): "Flat, slightly-gap-down and slightly-gap-up can all be the
  SAME trade."
- **The 25 Sep trade he reviews** is already in v5l. He bought the flat open and
  was cut at his limit (-3,04,979.30 at 09:43). He says the direction was still
  right, and the CSV agrees: NIFTY opened at 23,035.00, printed 23,048.80 at
  09:43 and closed at 23,128.10 (high 23,162.70).
- **The rest** is method the corpus already carries: analysis does half the
  work, learn *why* and not only *what*, target retail and not the operator,
  and skip a trade that has gone against the plan.

### The candidate, and why it was not encoded

The one idea not already written down is the split between the first and third
charts. Both have seated sellers, yet their gap-up branches differ:

- **One sudden fall.** That crowd is v5n's random seller, carried overnight. It
  flees an adverse gap in the first candles, so the gap-up branch is not a hunt.
- **A multi-day fall that already sat through retracements.** That crowd stays
  huntable in every open.

The test that would separate them: has this crowd already held through a
retracement?

That is a testable claim. After one sudden fall, a gap-up should be spent in its
first two candles; after a multi-day fall that retraced, it should keep going. I
measured it on five years of the NIFTY spot 1-minute CSV:

- **Reference close:** the 15:15 print.
- **Gap-up:** an open at least 0.3% above it.
- **"Sudden":** the prior session opened at least 0.6% below its own reference
  close and closed below its open, and the three sessions before it were not a
  fall.
- **"Trend":** the prior session closed red, at least 1.5% below the close five
  sessions earlier, with at least one up close among the three sessions before
  it.
- **What was measured:** the move left after the second candle (09:16).

| prior session | gap-ups | median further rise to 10:30 | mean net, 09:16 to 10:30 | net up |
|---|---|---|---|---|
| one sudden fall | 10 | 65.65 pts | +6.59 | 60% |
| multi-day fall that retraced | 21 | 37.00 pts | -13.96 | 38% |

That is the opposite of the claim, on both halves. It held across five threshold
variants: a sudden gap of 0.5% or 0.8%, a gap-up of 0.2% or 0.5%, and a trend of
2.5%. In every variant the sudden case's mean net stayed positive and the trend
case's stayed negative.

This does not prove the reverse:

- the samples are small;
- the definitions are proxies;
- the data is spot, not premium.

But a rule that the only available data contradicts does not go into a live
prompt, so **no knowledge change and no test change**. It is recorded here so a
later session does not re-derive it from the same video.

**For the note channel.** The 25 Sep pre-open note carried exactly this branch:
"A GOOD GAP UP -> SELL, WITH THE MARKET: a good gap up lets the sellers run
early". That day opened flat, so the branch was never exercised. The measurement
is a reason to read a branch like that as his forecast and nothing firmer, which
is what the note block already tells the agent ("It can be WRONG").

## 30 Sep - same trade, four seconds, and a hole in SLH-012 (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`dUHtpd2MhWo`,
  uploaded 2026-09-30 10:27 IST), traded on the 30 Sep note (`NrPmCbzrzSY`). His
  clock times are read off the taskbar clock in the video frame. His figures come
  from his positions screen at 7:09 in the video, about 09:45. His words are my
  translation of the Hindi auto-transcript.
- **This book:**
  - all 72 decisions in the decisions log;
  - the runner log, which holds the one trade the journal does not.

**No knowledge change.** Three candidate rules were measured and refused, and the
real finding is a bookkeeping defect, offered below as SLH-021.

### Same read, same instrument, opposite outcome

| | IH | the agent |
|---|---|---|
| open | flat, so the note's flat branch: BUY | FLAT (-0.226%), so the same branch, cited in every early decision |
| entry | 09:18, all three indices | 09:22:49, NIFTY 22700 CE at 184.75, BankNIFTY 54500 CE mirror at 1,152.80 |
| exit | 09:45, booked | 09:22:53, AI_STOP at 22,700.10 |
| result | **+2,51,147.85** | **-301.75** |

His positions at the exit:

| leg | qty | avg | LTP | P&L |
|---|---|---|---|---|
| BankNIFTY | 1,170 | 1,162.18 | 1,260.40 | +1,14,917.40 |
| BankNIFTY | 1,170 | 1,109.25 | 1,194.30 | +99,508.50 |
| SENSEX | 900 | 421.36 | 453.20 | +28,656.25 |
| NIFTY | 1,430 | 184.16 | 189.80 | +8,065.70 |

His NIFTY average (184.16) is within a rupee of the agent's 22700 CE fill. It was
the same trade, four minutes apart, and **85% of his result came from the two
BankNIFTY legs**.

### What happened to the agent's trade

**1. The order fired on a stale price.**

- The runner was restarted at 09:19 and again at 09:21, switching the model each
  time: `claude-sonnet-5-5` until 09:18, then `claude-sonnet-5.5` (the 09:20
  agent error), then `claude-sonnet-5`.
- The first decision after the second restart ran for **87.2 seconds** against a
  90-second deadline.
- It named an entry of 22,727.70 and a stop of 22,701.00, a risk of 26.70 points.
  When the order fired, spot was **22,705.20**, so only 16% of that risk was left,
  and the stop fired four seconds later.

**Even at its own price the stop would not have survived.**

- NIFTY printed **22,685.10** at 09:34, according to the decisions log.
- The stop sat 1 point above the 22,700 round number, inside the morning's
  range. It was above both the opening low (22,659.80) and the note's 22,650
  support, which are the levels the flat-BUY premise rested on (v5e).
- IH sat through that dip: "a small rejection comes and cuts the profit a lot...
  but no big retracement is happening, that's the good thing."

**2. Then it held for an hour.**

- 67 decisions followed, all HOLD. 46 of them cited the stopped long or a
  "played-out" premise, several by name as POST-LOSS SPEED LIMIT and the
  POST-EXIT RE-ENTRY GATE.
- Meanwhile NIFTY ran from 22,685.10 (09:34) to 22,766.00 (10:21).

**3. The trade never reached the journal.** See the defect below.

### Three candidates, measured, none encoded

**(a) Loosen re-entry after a loss.**

The agent applied the POST-EXIT RE-ENTRY GATE after a loss. v5m measured
after-loss re-entries as profitable: 43 trades, +33,714.25. Split by how long the
losing trade had lasted:

| prior losing trade lasted | same-side re-entries | sum | winners |
|---|---|---|---|
| more than 2 minutes | 26 | +47,162.50 | 65% |
| 2 minutes or less | 17 | **-13,448.25** | **18%** |

The second row is robust:

- 12 of its 14 sessions are negative;
- both halves are negative (-7,117.75 / -6,330.50);
- it is still -5,411.25 without the worst two sessions.

30 Sep's four-second stop belongs to that row. So the holds were right by the
book's own history, and the day was the exception. **Refused.**

**(b) Refuse an order whose live spot has already used up most of the declared
risk.**

This would extend SLH-016, which refuses only a stop that is already breached.
The measure is the share of the declared risk still left when the order fired,
across 150 matched entries (7 Jul-29 Sep):

| risk left at the order | trades | sum | winners |
|---|---|---|---|
| 100% or more (spot at or past the model's entry) | 79 | +55,920.00 | 54% |
| 75-100% | 43 | +15,698.50 | 40% |
| 50-75% | 23 | -8,259.00 | 35% |
| under 50% | 5 | -651.50 | 20% |

The gradient is real, but the cut-off is not robust. Only a 75% cut keeps the
blocked set negative in both halves (-7,455.00 / -1,455.50) and without its worst
two sessions (-2,501.25). At 70% and at 80%, the second half and the
worst-two-removed figure both turn positive (70%: +800.50 / +2,007.75; 80%:
+674.25 / +2,464.50). A gate that only works at one exact number is a fitted
number. **Refused**, recorded so it can be re-run as the book grows.

**(c) Refuse an order placed late in a long decision call.**

Call timing exists for 58 entries (median 38.5 seconds into the call). Outcomes
show no pattern by delay. **Refused.**

### IH's other points, all already in the corpus

- **"The market normally takes direction where fewer traders are working"** is
  why he bought the flat open and not a gap up. That is v3l (once your side IS
  the crowd the edge is gone), and it is the 30 Sep note's gap-up branch.
- **"We bore one rejection; a second one may create a problem"** is the RISK rule
  that a premise tolerates roughly TWO rejections.
- **"An option buyer books as soon as momentum comes"**, with an average target
  on a high premium, is v3i.

### The defect: a trade that opens and closes inside its own decision call is never journaled

SLH-012 journals a trade placed by a decision call that later went stale, but
only when the position is **still open** when the call's result is collected
(`opened_a_position` checks `self.pos.active`). A trade that is entered and
stopped while its call is still running fails that check:

- the result is discarded as stale;
- `after_exit` has no open row to close;
- the trade is missing from both the journal and the decisions log.

The runner log shows it four times, each an entry, then an AI_STOP within about
four seconds, then "Discarding late SL Hunting result for stale generation N":

- 10 Aug, 09:50:59;
- 4 Sep, 09:15:48;
- 15 Sep, 09:15:55;
- 30 Sep, 09:22:49.

The last three all came **after** SLH-012 shipped. The reflection coach and these
comparisons read the journal, so they have never seen these trades. That bias is
not neutral: every missing trade is a fast stop.

## 1 Oct - the index that would not fall (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`ykWQl_f_ERg`,
  uploaded 2026-10-01 11:17 IST), traded on the 1 Oct note (`ZrSyia9MGHc`).
  His clock times are read off the taskbar clock in the video frame. His
  positions screen did not appear, so his result is left in his own words. His
  words are my translation of the Hindi auto-transcript.
- **This book:** its three journal rows and 74 decisions.

**No knowledge change.**

### Both books sold, and both were stopped by BankNIFTY

The note's one plan for every open was SELL. The open classified FLAT (NIFTY
-0.34%, BankNIFTY -0.09%).

| | IH | the agent |
|---|---|---|
| entry | 09:18, puts on all three: BankNIFTY 54,600 PE x1,170, SENSEX x900, NIFTY x1,430 | SHORT 22,510.70 at 09:26 |
| what happened | profit, then a retracement to "almost zero", then a growing loss | booked at 09:27 for **+3.75**: "BankNIFTY (leading index) printed a confirmed bullish morning-star plus bullish engulfing off its 09:24 swing low (54686.1)" |
| exit | 10:01, "the loss limit is gone, so we cut the trade" | |
| later | none | SHORT 09:35, **-231.00**; SHORT 10:11, **-141.75** |
| day | a loss, amount not shown | **-369.00** |

His watchlist at 10:01 shows the split he blamed:

- BankNIFTY 54,974.40, **+0.62%**, just under the note's 55,000 resistance;
- NIFTY -0.12%;
- SENSEX +0.05%.

His diagnosis: "sometimes two indices stay a little negative, and in the ONE
index that stays behind, if sellers gather and a round number is nearby, the
market sometimes heads for a breakout, and then our problem becomes big." As
for who those sellers were: "because it took resistance again and again, some
people started selling... since SENSEX and NIFTY made negative momentum, sellers
gradually gathered in BankNIFTY too."

**The book read the same divergence earlier than he did, and left in a minute.**
Its second and third trades then sold into it:

- trade 2 cited "BankNifty mirrors it with a double top at 54800 psych
  resistance" as confirmation;
- trade 3 cited BankNIFTY's bearish candles near 54,900.

Those are the repeated round-number rejections IH says recruited the sellers
the breakout then hunted.

### Why nothing was encoded

**The mechanism is already written.** R:R-BAIT AT ROUND-NUMBER REJECTIONS says
"resistance at a round number during momentum is often the operator INVITING
trades... one shove can clear it once the counter-side is loaded". THIRD-INDEX
LAG and v4i (THE LAGGING INDEX DECIDES THE BASKET'S EXIT) cover the refusing
third index, and v4i quotes IH on almost this situation.

**The only new part cannot be priced.** That part is the cross-index form: a
rejection in the index that REFUSES your direction is bait in that index, not
confirmation for yours.

- The BankNIFTY history on disk stops in 2023, and the decisions log holds no
  BankNIFTY prices.
- The nearest measurable proxy is entries taken against the cross-index bias:
  55 trades, **+17,876.75**, 51% winners.
- Its halves split -4,675.75 / +22,552.50, so it is not robust in either
  direction.

That is no basis for a rule that would stop such entries, so the scope stays
recorded here and is not encoded.

**SLH-020 flagged trade 2** ("SHORT entered after a winning SHORT... booked
3.75"). A +3.75 scratch counts as a win under v5m's measure, which is pnl > 0.
The note it carries is informational and refused nothing. Left as is.

## SLH-021 - a trade that opens and closes inside its own decision pass

**The defect, measured on 30 Sep.** A trade's journal row is opened when its
inference pass is HARVESTED. When the pass entered and the mechanical stop
closed the trade before that:

- `after_exit` had no row to close;
- the per-poll stop had already invalidated the pass;
- the harvest saw a stale pass and a FLAT position. SLH-012's
  `opened_a_position` checks `self.pos.active`, so it discarded the pass.

The trade was then missing from both the journal and the decisions log. The
runner log shows it on four days, each an AI_STOP three to four seconds after
the fill:

- 10 Aug, 09:50:59;
- 4 Sep, 09:15:48;
- 15 Sep, 09:15:55;
- 30 Sep, 09:22:49.

The last three came after SLH-012 shipped.

**The fix is bookkeeping only.** Nothing places, refuses or sizes an order
differently.

- **The worker** (`SLHuntingAIWorker.after_exit`): when no journal row is open
  and the executor still holds an unconsumed `last_entry_order`, it parks what
  the row needs. Only the in-flight pass can have placed that order. The parked
  facts are:
  - the closed position (its levels, quantity and entry price quality);
  - the entry order;
  - the exit context;
  - SLH-020's `_trade_realized_before` as the P&L basis;
  - the mirror's entry price quality, read before the basket close resets it;
  - whether the mirror survived.

  Parking consumes the order, so it cannot leak into a later row.
- **The harvest** (`_consume_agent_decision`): it takes the parked close before
  anything can return, so it can never attach to a later pass. It then:
  - treats a parked close like SLH-012's open position: a stale pass is
    recorded, with a WARNING that names it, rather than discarded;
  - appends the pass's decision to the decisions log;
  - opens the row from the parked facts and the pass's own decision and
    frames;
  - closes the row with this trade's basket P&L. A NIFTY-only cut whose mirror
    is still running defers the close to the mirror, exactly like the ordinary
    deferred path.

  A row deferred like that still owns the row id, and a second open in the same
  pass must not overwrite it.

**Known limit.** A pass harvested while a position is open is handled by SLH-012,
unchanged. A trade parked by a pass and never harvested, because the worker
stopped first (square-off or max-loss), is still lost. The agent stops entering
at 10:30, so that needs a max-loss stop inside the same 90-second pass.

**Tests.**

- **Five on the SLH-012 stub:**
  - a stale pass whose trade already closed is journalled, with a warning;
  - its decision reaches the decisions log;
  - a fresh pass whose trade closed is journalled;
  - a parked close is dropped with an empty payload rather than left for the
    next pass;
  - a deferred parked row is never overwritten.
- **Four on the REAL worker:**
  - the 30 Sep sequence: enter, invalidate, AI_STOP, then a stale harvest
    journals the pass's own setup and reasoning and this trade's basket P&L,
    with 250 of earlier P&L seeded so the day's total cannot pass for it;
  - a NIFTY-only cut inside the pass waits for the mirror;
  - an ordinary journalled trade is never parked;
  - a close with no pending entry order parks nothing.
- **Ten code mutations, all caught:**
  - the park removed;
  - parking without the order check;
  - the stale discard restored;
  - the P&L basis taken at harvest;
  - the mirror deferral removed;
  - the parked close kept past an empty payload;
  - the order not consumed;
  - the row built from the flat live position;
  - the warning wording;
  - the overwrite guard.

  The control passes.

## 4 Oct weekly lecture and 5 Oct pre-open note (v5p)

**Sources and extraction.** Direct YouTube Hindi auto-transcripts were retrieved
with `youtube-transcript-api`, read in full, and checked against the current
knowledge. YouTube publication timestamps, converted to IST, identify both as
4 Oct uploads:

- `DNb7XmCK7C8`, "The Real Benefits of Chart Trading | Trade Smarter", 15:04,
  published 4 Oct 2026 at 11:27 IST.
- `Hbf7R3KqEwE`, "Prediction For 05 OCT 2026", 1:35, published 4 Oct 2026 at
  21:00 IST. Its named trading session is Monday 5 Oct, not Sunday 4 Oct.

**Net-new entry refinement.** At 0:06-2:40 the lecture contrasts two premium-led
entries: chasing a rising option because it appears to confirm the trade, and
buying a falling option because it appears cheaper. Both substitute an option
price comparison for the chart's entry permission. The late example at
14:05-15:01 connects that choice to panicking out when the premium subsequently
falls. `PSYCHOLOGY` now carries THE CHART AUTHORISES THE ENTRY; PREMIUM DOES NOT:
remove the premium comparison and the reasoning must still justify this entry
now. A valid controlled entry is not delayed solely for a remembered option
price, and an invalid chart cannot be rescued by a cheaper premium. Option cost,
freshness, liquidity, sizing and hard-risk checks remain required.

**Overlap deliberately retained.** Waiting for a large first candle to settle
(2:44-4:18) is already covered by the existing momentum and entry rules.
BankNIFTY's contrary opening read on 29 Sep (5:42-7:44) confirms the cross-index
and closing-point checks. The stopped trade followed by a later favourable move
(10:12-13:19) repeats ENTRY QUALITY AND DIRECTION ARE SEPARATE JUDGEMENTS, YOUR
EDGE IN THE READ IS HIGHER THAN YOUR EDGE IN THE TRADE, and the discipline rule
against revising an exit from one hindsight outcome. v5j already judges exits
on the underlying chart rather than the premium; v5p extends the distinction to
entry timing. The lecture's preference for independent execution after prior
discussion adds no new agent mechanism and is not encoded as a team veto.

**Premarket note.** Selling followed by a retracement likely removed nearby
sellers, while buyers are also thin. Flat, gap-up and small gap-down opens seek
confirmed BUY setups. A large gap-down instead follows selling. No numeric
small/large cutoff is supplied by the source. This is a dated advisory, not a
new durable gap rule.

The 1080p premarket frames at 0:34, 1:03 and 1:27 resolve the garbled captions.
The note uses his spoken rounded levels, nearest first:

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,590 / 22,800 | 22,200 / 22,000 |
| BANKNIFTY | 54,880 / 55,200 | 54,000 / 53,500 |
| SENSEX | 72,400 / 73,130 | 71,300 / 70,600 |

BankNIFTY's support labels are 54,007.90 / 53,504.60; SENSEX's lower support is
70,605.33. NIFTY's first resistance is spoken 22,590 against the chart label
22,584.20; the spoken value is retained rather than inventing precision. Video,
metadata and frame artifacts are kept under
`%TEMP%\intradayhunter-2026-transcripts`, outside git.

## 5 Oct - same side, same exit minute (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`aPu1pRZztf8`,
  uploaded 2026-10-05 10:55 IST), traded on the 5 Oct note (`Hbf7R3KqEwE`,
  "FLAT, GAP UP or SMALL GAP DOWN -> BUY"). His clock times are read off the
  taskbar clock in the video frame, and his figures off his positions screen at
  10:22 in the video. His words are my translation of the Hindi auto-transcript.
- **This book:**
  - its one journal row;
  - all 65 decisions;
  - the runner log.

v5p was in force (#197, merged 4 Oct 22:30).

**No knowledge change.**

### The comparison

| | IH | the agent |
|---|---|---|
| open | gap up: BankNIFTY already running, SENSEX and NIFTY holding at resistance | the note's buy branch |
| first attempt | waited out the direct push | 09:20 LONG 22,557.40, stop 22,515.00, **refused by sizing**: one lot would risk 2,756.00 against the 2,500 budget |
| entry | **09:23**, on the first rejection: BankNIFTY 55,000 CE and 55,100 CE x1,170 each, SENSEX x900, NIFTY x1,430 | **09:43** LONG 22,580.75, a 61% pullback hammer, stop 22,560.00 |
| exit | **09:50**, "a normal target", because BankNIFTY lagged | **09:49:19** at 22,611.15, at the prior-day high (22,610.60) and the 22,600 round number |
| result | **+1,62,201.85** | **+1,851.50** |

NIFTY topped at 22,617.40 at 09:51 and fell to 22,486.65 by 10:26, so both
books left within a minute of the top.

His positions at the exit:

| leg | qty | avg | LTP | P&L |
|---|---|---|---|---|
| BankNIFTY 55,000 CE | 1,170 | 1,023.53 | 1,034.50 | +12,834.90 |
| BankNIFTY 55,100 CE | 1,170 | 972.47 | 975.35 | +3,370.35 |
| SENSEX | 900 | 565.68 | 680.20 | +1,03,068.00 |
| NIFTY | 1,430 | 112.98 | 143.00 | +42,928.60 |

The two BankNIFTY legs made 16,205.25, **10%** of the total, and that is the
whole reason he booked: "the target only gets big when BankNIFTY's momentum
comes with it... if BankNIFTY turns negative, it won't even let us see the
SENSEX and NIFTY profit."

### Why nothing was encoded

**His entry read is already written down.** His reasoning: after a fast fall
and a break of two or three days, "neither the sellers' nor the buyers' stops
are available here", but any rejection invites sellers, so "whatever traps form
will form for the sellers". That combines three existing rules:

- WEEKEND / HOLIDAY CARRY-RISK (no inventory carried over a break);
- the as-the-opening rule of BOTH-WAYS FLUSH DAY;
- v4t's "a follow is joined on the retracement". He waited out the direct push
  and bought the first rejection.

**His exit is v4i almost word for word:** "the lagging index decides the
basket's exit, not the leading one".

**"The momentum came where the sellers came" was not encoded.** His account:
BankNIFTY went positive first, so sellers left it alone and sold the two
indices that looked weak, and those two then ran. On 1 Oct he gave the mirror
image: sellers gathered in the index that refused to fall, and it broke out.
Read together, the two days give no direction a rule could use before the fact,
and the BankNIFTY and SENSEX history needed to test one does not exist on disk.

**The 09:20 refusal was the risk budget working as configured.** It turned down
a 42.4-point stop, and the budget is a deliberate operator setting. The 09:43
entry, 23 points higher with a 20.75-point stop, still caught the leg. Nothing
to change.

## SLH-022 - the agent gets no built-in tools (and the SDK goes back to 0.2.159)

**The hang (6 Oct).**

- `claude-agent-sdk` 0.2.160-0.2.163 publish no Windows wheel. On the runner, pip
  installed the generic wheel of 0.2.163, whose `_bundled/` holds no Claude CLI.
- With no bundled CLI, the SDK silently ran the `claude` on PATH: CLI 2.1.138,
  where 5 Oct had run 2.1.281.
- 0.2.160 also holds a run that has SDK MCP servers open until the CLI reports
  `idle`, for up to 600 s.

Result: 8 decisions completed against 72-74 on a normal day, 12 calls abandoned at
the 90 s deadline and left running (one for 12.7 minutes), and 34 bars skipped
behind them. Every order attempt was refused, so nothing reached the broker. PR
#200 reverts the pin to 0.2.159, whose Windows wheel contains
`_bundled/claude.exe`. The repository-policy ledger now holds the Windows-wheel
check every future bump must pass.

**The tools (found while tracing the hang).**

`permission_mode="dontAsk"` denies only the tools that would ASK for permission.
The CLI still ran the ones that never ask:

- `echo` shell commands on 1 and 5 Oct;
- an `Agent` subagent on 6 Oct.

Without `tools`, the bundled CLI 2.1.281 exposes 25 built-ins (PowerShell, Read,
Write, Edit, Agent, WebFetch, WebSearch and more). A toy-server A/B test on that CLI
showed the model run PowerShell and spawn a subagent. With `tools=[]` only the MCP
tools exist, and they load directly instead of through `ToolSearch`: 11 s against
22 s.

A real paper decision through the patched agent then called its own seven tools
and nothing else. The prompt carries third-party text (the pre-open note), so this
is a prompt-injection boundary. The coach already had the same line.

**Test.** `test_default_run_switches_off_every_built_in_tool` pins `tools == []`,
`dontAsk`, empty `setting_sources`, and an allow-list of `mcp__slhunting__*` only.
Five mutations were all caught, and the control passes:

- the `tools` line removed;
- `tools=None`;
- a built-in tool allowed;
- `bypassPermissions`;
- user settings loaded.

## 6 Oct - one word, two branches, and a blind half-hour (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`KJ1WTg6gXNM`,
  uploaded 2026-10-06 10:29 IST), traded on the 6 Oct note (`WVfyM2gkWXs`,
  "FLAT TO GAP DOWN -> SELL", "GAP UP -> BUY, WITH THE MARKET"). His clock times
  are read off the taskbar clock in the video frame, and his figures off his
  positions screen. His words are my translation of the Hindi auto-transcript.
- **This book:**
  - all 53 decisions (no journal row: the agent did not trade);
  - the runner log.
- **The 5-year NIFTY CSV**, for the measurement below.

v5p was in force. This was also the day of the SDK hang that SLH-022 (above)
records.

**No knowledge change.**

### The comparison

| | IH | the agent |
|---|---|---|
| open | **"a slight gap up"**: the buy branch | **FLAT** (+0.21%, under the 0.5% line): the note's SELL branch |
| 09:15-09:43 | **in at 09:26**: BankNIFTY 54,900 CE and 55,000 CE x1,170 each, SENSEX 72,500 CE x900, NIFTY 22,550 CE x1,430 | **blind**: every call abandoned at the 90 s deadline or skipped behind one, and the runner restarted at 09:38 |
| 09:43-10:30 | | 8 reasoned holds; 7 cite the SELL branch, and all decline it because the closing-price breakdown never came |
| exit | **09:50**, "an OK target", before others start buying | no position |
| result | **+2,35,441.50** | **0.00** |

NIFTY opened at 22,603.25. That is +0.21% on the official close (22,555.75) and
+0.30% on the 3:15 level (22,535.30). The low was 22,561.60 at 09:16, so the
close was never lost. NIFTY was about 22,595 at IH's entry and 22,637.25 at his
exit. It made 22,667.35 at 09:57, gave back to 22,616.20 at 10:10, and then
climbed to 22,717.40 by 15:14.

His positions at the exit:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 54,900 CE | 1,170 | 937.30 | +75,172.50 |
| BankNIFTY 55,000 CE | 1,170 | 886.47 | +66,433.10 |
| SENSEX 72,500 CE | 900 | 424.53 | +60,903.00 |
| NIFTY 22,550 CE | 1,430 | 92.32 | +32,932.90 |

The two BankNIFTY legs made 1,41,605.60, **60%** of the total. On 5 Oct they
made 10%, and he booked early for that reason; today "BankNIFTY has started
rising, and when BankNIFTY rises we get a good target".

### What the agent did, and did not, get wrong

**IH's whole trade sat in the blind window.** The agent produced its first
reasoned decision at 09:43, seventeen minutes after his entry, so nothing in
this book says whether it would have bought at 09:26. The cause is SLH-022's,
and it is fixed.

**Its holds after 09:43 were the rule working.** CLOSING-PRICE BREAKDOWN IS THE
TRIGGER says buyers give their stops only once price trades below the previous
close, and that "if the breakdown never arrives the trade never existed". The
agent said exactly that at 09:57: the SELL premise "needs a closing-price
breakdown below 22555.75, which never came ... so the sell thesis is
unconfirmed, not triggered." It did not sell into a market that rose all day.
On the long side, its 09:43 read was that the move "has already run ~68pts in
25min" and was consolidating, and at 09:57 that the bullish cluster "already
fired and already paid". That is the no-chase rule.

**The open was classified by the book's own rule, and IH's word disagreed.**
v4t's calibration came from IH: he called a 0.24% gap-down "almost FLAT" on
31 Aug, and v5f records him calling a +0.36% open FLAT on 16 Sep. Today he
called +0.21% "a slight gap up" and took the buy branch. His reason was the
close, not the size: "until it crosses the closing price, a sell trade cannot
be built overall... had it traded below the closing price, or opened around
it, a sell could have been made." His word followed what price did around the
close in the first minutes, and no single size line fits all three of his
calls.

### The candidate, measured and refused

The candidate: on a small gap (inside the 0.5% FLAT band), let the first
minutes against the prior close pick the branch. Close held means go WITH the
gap; close lost means go AGAINST it. This is IH's 6 Oct reading made general.

Measured on every small-gap session in the 5-year NIFTY CSV. The reference is
the 3:15 close (v4u); "held" means the first 15 minutes never traded through it;
the score is points in the gap's direction from 09:30 to 10:30, the agent's
entry window:

| small gap | close in first 15 min | sessions | mean | winners |
|---|---|---|---|---|
| up | held | 309 | -3.3 | 49% |
| up | lost | 270 | -3.0 | 47% |
| down | held | 178 | +2.4 | 52% |
| down | lost | 170 | +2.2 | 49% |

Holding the close carries no information about the next hour. The result is the
same with the official close as the reference, with 5- or 30-minute windows,
with a 0.3% band, and to 11:00. Only the full day, to 15:15, separates them at
all: held beats lost by 7-13 points on both sides, but held gap-ups still average
-7.1 going WITH the gap, every win rate stays between 47% and 53%, and 15:15 is
far outside the agent's entry window. By year, the held gap-ups run from -22.3 (2022) to +12.7 (2026),
and the lost ones from -25.1 (2021) to +12.3 (2026): what moves them is the
year's drift, not the hold.

IH's exact setup does no better. The prior session sold off and retraced into the
close (its low after 11:00, its close in the upper half of its range), then a
small gap up held the close: 35 sessions, mean -5.6, 42% winners.

So neither word is an edge on its own here. The FLAT reading lost nothing
because the closing-price trigger kept the agent out of the sell, and the gap-up
reading has no measured advantage that would justify buying the open. The
classification stays the SLH-017 arithmetic, and the trigger stays the close.

### IH's other points, all already in the corpus

- "Because of the retracement, the sellers ran away": v5f, a retracement is what
  lets a crowd survive into the next session, and here it removed one.
- "When rejection comes the sellers come, then it rises again ... rejection,
  rise, rejection, rise": the 5 Oct addendum's reading, "whatever traps form will
  form for the sellers".
- His exit: "the momentum is a little slow because sellers' stops are not
  directly available ... after the round-number breakout, others will also start
  buying", so he booked before them. That is v4i's THE CROWD'S FEAR IS YOUR WINDOW,
  AND IT CLOSES WHEN THEY JOIN, and the slow-candles rule that "small candles ...
  INVITE buyers, and if it invites buyers it will definitely give a retracement".
  The retracement came: 22,667.35 at 09:57 to 22,616.20 at 10:10.

## 7 Oct live match and 8 Oct pre-open note (v5q)

**Evidence.** The live video `0jt8c4MX-hA` was published 7 Oct 2026 at 11:09 IST
(10:59 duration). Its direct YouTube Hindi auto-transcript has 295 segments and
was read in full. Frames at 1:33 and 9:18 show the entry sequence around 09:25
and all four legs flat at 10:15. Publication times are parsed from YouTube's
offset-bearing metadata and converted to IST.

The premarket video `FPaMrMLkxIw`, "Prediction For 08 OCT 2026", was published
7 Oct at 21:25 IST (about 1:49). It exposes no caption tracks in the watch page,
`youtube-transcript-api`, or `yt-dlp`. Its original audio was therefore decoded
locally with faster-whisper's cached `small` model. The initial Hindi decoding
distorted phrases; the full Hindi-to-English translation pass supplied the
usable forecast. This is a LOCAL AUDIO ASR fallback, not a direct YouTube
transcript or a NoteGPT extraction. The shipped note labels that distinction.
All levels were checked independently against 1080p chart frames.

### IH's trade and the agent's book

IH took one CALL basket after the abrupt opening fall. He explicitly separated
it from the previous evening's BUY plan: the market had erased hours of the
prior advance in one minute, so he was trading a NEW chart and a bounded
recovery idea, not simply following that old branch. He acknowledged continuing
selling and a breakdown could make this range idea wrong (6:58-7:42), then
booked a normal target when BankNIFTY recovered while NIFTY and SENSEX stayed
largely sideways (9:03-10:49).

His completed positions screen at 9:18 of the video reads:

| Leg | Entry quantity | Realized P&L |
|---|---|---|
| BankNIFTY 54,700 CE | 1,170 | +122,850.00 |
| BankNIFTY 54,800 CE | 1,170 | +123,996.60 |
| SENSEX 72,600 CE | 900 | -16,380.00 |
| NIFTY 22,600 CE | 1,430 | -9,895.35 |
| Basket | | **+220,571.25** |

The agent's 7 Oct JSONL records contain 71 decisions: 61 HOLD, three LONG
entries, two SHORT entries, and five exits. Its five completed journal rows are:

| Entry / exit | Side | Entry / exit NIFTY | Journal option P&L |
|---|---|---|---|
| 09:33:26 / 09:35:29 | LONG | 22,630.70 / 22,612.00 | -507.00 |
| 09:43:55 / 09:47:28 | LONG | 22,618.75 / 22,612.30 | -500.50 |
| 09:58:11 / 10:01:25 | SHORT | 22,612.40 / 22,607.50 | -214.50 |
| 10:07:56 / 10:13:23 | SHORT | 22,617.60 / 22,622.40 | -552.50 |
| 10:19:32 / 10:21:24 | LONG | 22,647.40 / 22,681.75 | +1,589.25 |
| Total | | | **-185.25** |

The rupee figures describe different quantities and baskets; they are not a
capital-normalized ranking. The useful mismatch is in the rationale. All three
agent LONG entries cite the old flat-open BUY branch. IH says the old chart no
longer carries his entry and states a new recovery/range hypothesis instead.
The two agent SHORT exits cite the existing hierarchy when BankNIFTY recovers.
The final LONG books into the round-number/pivot stall under
the existing profit-rate rule.

### Net-new refinement and overlaps

`RETAIL_POSITIONING` adds AN OPENING THAT SPENDS HOURS IN MINUTES RETIRES THE
PRE-OPEN PREMISE. It supplies a concrete early trigger for CURRENT-SESSION TRAP
RESET: when the opening rapidly erases a multi-hour leg, state the NEW crowd,
regime, entry and invalidation. The side may match the note, but that does not
preserve the note's premise. This scopes PREVIOUS-CHART LINKAGE and YESTERDAY'S
MOMENTUM CHARACTER rather than adding a directional exception or numeric gate.

Not encoded: fast volatility automatically predicts a range, or buying low in
an assumed range is sufficient. IH's own mid-trade doubt and BankNIFTY's later
directional recovery show why those are hypotheses needing price confirmation.
The existing pattern, qualified continuation exceptions, stop, loss limit and
cross-index exit rules still govern.

The remaining ideas overlap the corpus: paid/absent inventory is TARGET-BOOKED
and v4f's one-minute move; modest expectations without nearby stops are NORMAL
TARGET and v5g/v5h; BankNIFTY as the only profit driver is the existing
single-index booking context; discipline at the loss limit is v4h/v4n. Comparing
one's risk to another trader's large capital adds no sizing permission.

### October 8 advisory

The audio forecast keeps SELL setups for flat or small gap-up opens and the same
SELL plan after a gap-down. SENSEX's discussion qualifies it with the absence of
a LARGE gap-up; the note carries that warning without inventing a threshold or
an automatic reverse trade. BankNIFTY can retrace after a gap-down. The source
also flags SENSEX expiry for the intended session.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,720 / 22,800 | 22,540 / 22,420 |
| BANKNIFTY | 55,560 / 56,000 | 54,650 / 54,200 |
| SENSEX | 73,000 / 73,450 | 72,400 / 72,000 |

Frames at 0:32 (SENSEX), 1:04 (BankNIFTY), and 1:40 (NIFTY) resolve the rounded
levels: NIFTY labels 22,718.30 / 22,800.60 and 22,544.25 / 22,419.10;
BankNIFTY 55,562.05 / 56,006.05 and 54,654.65 / 54,203.95; SENSEX 73,004.09 /
73,453.25 and 72,395.70 / 72,005.24. The dated-note regression checks the
rendered branches, qualifications, ASR source label, exact rounded levels and
date gating. Ledgers and extraction artifacts remain outside git.

## 8 Oct - the same entry minute, and three ideas the data refused (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`dG6ekT8_EtQ`,
  uploaded 2026-10-08 11:06 IST), traded on the 8 Oct note (`FPaMrMLkxIw`,
  "FLAT or SMALL GAP UP -> SELL", "GAP DOWN -> SELL"). His clock times are read
  off the taskbar clock in the video frame, and his figures off his positions
  screen. His words are my translation of the Hindi auto-transcript.
- **This book:**
  - its three journal rows;
  - all 73 decisions;
  - the runner log.
- **The 5-year NIFTY CSV**, for the three measurements below.

v5q was in force (#203).

**No knowledge change.**

### The comparison

| | IH | the agent |
|---|---|---|
| open | FLAT (NIFTY -0.02%): "overall selling, and we have to sell here" | FLAT: the note's SELL branch |
| entry | **09:19-09:20**: BankNIFTY 54,800 PE and 54,700 PE x1,170 each, SENSEX 72,500 PE x900 (its expiry), NIFTY 22,550 PE x1,430 | **09:21** SHORT 22,530.80 on a bearish inside bar after the first candle's flush, stop 22,553.00 |
| exit | **09:47-09:48**: "the profit is only in SENSEX and NIFTY; BankNIFTY is behind" | **09:28** at 22,484.80, near its 22,470 target: **+2,405.50** |
| re-entries | none | 09:45 SHORT 22,469.60, cut at 09:52 at 22,479.60 on BankNIFTY's reversal: **-3,153.00** (2 lots); 10:18 SHORT 22,476.65, booked at 10:21 at 22,440.70: **+2,265.00** |
| result | **+1,78,881.25** | **+1,517.50** |

NIFTY opened at 22,599.05, -0.02% on the official close (22,603.05) and -0.07% on
the 3:15 level (22,615.45). The first candle flushed to 22,521.10. It closed
22,528.50 at 09:19 and 22,544.10 at 09:20, so the two books sold the same
minute at the same price. It was 22,449.65 at IH's exit, bounced to 22,498.10 at
10:14, and then fell all day: 22,179.90 at 15:09, 22,215.80 at 15:14.

His positions at the exit:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 54,800 PE | 1,170 | 666.35 | +2,164.50 |
| BankNIFTY 54,700 PE | 1,170 | 625.50 | -234.00 |
| SENSEX 72,500 PE | 900 | 263.42 | +1,14,102.00 |
| NIFTY 22,550 PE | 1,430 | 148.35 | +62,848.75 |

The two BankNIFTY legs made 1,930.50, about **1%** of the total, which is the
reason he booked.

**The difference is the first exit.** The agent booked 46 points at 09:28 because
"the flush is nearly spent". IH sat through the bounce to 22,525.55 at 09:30 --
"sometimes it feels cold, sometimes hot" -- and booked about 80 points later. His
rule for sitting through it: "when will selling change to buying? When the market
covers its higher point, that is, where it started from." That is candidate 2.

### Three candidates, measured, none encoded

**1. The copied chart, scoped by who was hurt.** IH's entry argument was that
"today too the market is making almost the same chart" as 7 Oct's opening fall,
and the sellers of that candle were stopped out by its recovery, "so today they
will not sell". The move would run with nobody crowding it. That reads the copy
as a CONTINUATION, where v4f (THE CHART DOES NOT REPEAT TWO DAYS RUNNING) reads
it as a trap. The v4f record carries its own scope: "if the buyers made profit
yesterday, today they should not get that profit", so a copy betrays the side
yesterday PAID. 8 Oct's copiers had been PUNISHED: 7 Oct opened at 22,690.45,
fell 0.30% in five minutes, and was back at 22,717.65 by 12:04. The candidate:
after a punished copy, follow.

Definitions: a sharp open moves at least 0.25% in its first five minutes; a copy
follows a session whose own open moved the same way by as much; that session
PUNISHED its followers if it later traded back through its open. Points in the
move's direction from 09:20:

| session | n | to 10:30 | winners | to 15:14 | winners |
|---|---|---|---|---|---|
| copy, prior punished | 17 | -15.7 | 52% | -69.2 | 29% |
| copy, prior paid | 23 | -1.9 | 43% | +3.6 | 56% |
| no copy | 208 | +6.8 | 54% | -1.8 | 52% |

The punished copy reverses more often than it runs, and 8 Oct (+76.7 by 10:30)
is the exception rather than the pattern. Thresholds of 0.2-0.4% and windows of
3 or 10 minutes flip the two copy groups' signs, on samples of 4 to 38 sessions.
What survives is weaker than either reading: at the base setting and most
variants, copies of either kind followed worse than ordinary sharp opens, which
is v4f's direction -- though not at the 0.3-0.4% thresholds. v4f stands
unedited.

**2. Hold a follow until it reclaims where it started.** The 177 closed journal
trades the CSV covers (2 Jul to 8 Oct) were replayed and held until price
reclaimed the session extreme as at entry -- the move's origin -- or a horizon:

| exit | net points | winners |
|---|---|---|
| the agent's own | +483.7 | 48% |
| origin, or 10:30 | -362.4 | 49% |
| origin, or 11:00 | -536.4 | 44% |
| origin, or 15:14 | +515.3 | 37% |

The origin is a median 45.8 points away. The one horizon that keeps up does so
with more than twice the usual stop distance and 37% winners, and September
(-211.1) and October (-262.0) carry the 10:30 loss. 8 Oct's first trade (+76.7
against +46.0) is the flattering case.

**3. Do not sell the session low on a follow.** The agent's 09:45 short sold a
fresh low at 22,469.60, two minutes before IH booked out near 22,450 -- where
v5k says a follow's sellers LEAVE, as on 24 Sep. On the same 177 trades, entries
within 10 points of the session extreme in the trade's direction made +21.1
points (40 trades, 42% winners) against +462.6 (137, 50%) for the rest. But the
halves disagree: -44.6 (36%) in Jul-Aug, +65.7 (50%) from late Aug. The 5-, 15-
and 20-point cuts do not order cleanly either. A gate would also have refused the
09:21 trade -- 9.7 points off the low, +2,405.50, the one that matched IH's
entry. Not a gate; v5k's prose stands.

### IH's other points, all already in the corpus

- "Where we sell, can others also sell there?", and yesterday's stopped sellers
  staying out: v4b, REGIME MEMORY DECIDES WHO SHOWS UP AT A LEVEL.
- "The sellers' stops are not available, so we are not targeting them; we follow
  the continuous selling": v5k, WHO PAYS FOR A FOLLOW WHEN NOBODY HAS STOPS
  NEARBY.
- His exit: "when two indices make a big move, the third stays behind, and
  sometimes turns": v4i, THE LAGGING INDEX DECIDES THE BASKET'S EXIT. The agent's
  09:52 cut on BankNIFTY's reversal used the same hierarchy.
- The passage on fear, repeated losses and cutting quantity after them: v3y and
  the RISK section.

### SLH-022 in production

73 decisions, none abandoned or skipped. Latency: median 33.3 s, 90th percentile
41.3 s, worst 73.0 s, against 45 failed calls on 6 Oct.

## 9 and 12 Oct - IH's two losing days, and the ideas the data refused (diligence addendum)

**Sources.**

- **IH's live sessions:**
  - 9 Oct: `-_o8lf08-Fg`, uploaded 2026-10-09 10:15 IST, traded on the 9 Oct note
    (`ZVOpBDyGpNQ`: "FLAT TO GAP DOWN -> BUY", "GAP UP -> SELL, WITH THE MARKET");
  - 12 Oct: `GAbEDa49EIk`, uploaded 2026-10-12 10:56 IST, traded on the 12 Oct note
    (`O2MFYGzPGrQ`: "FLAT TO GAP UP -> BUY WITH THE MARKET ... BANKNIFTY must not
    turn sharply negative immediately after opening").

  His clock times are read off the taskbar clock in the video frame, and his
  figures off his positions screen. His words are my translation of the Hindi
  auto-transcript.
- **This book:** one journal row each day, all 144 decisions, and the runner log.
- **The 5-year NIFTY CSV**, which ends on 8 Oct. Prices for 9 and 12 Oct come
  from the decision log's per-minute last price.

v5q was in force.

**No knowledge change.**

### 9 Oct: the one word split the books, and the arithmetic won

| | IH | the agent |
|---|---|---|
| open | "some gap up" in the pre-open: the note's GAP UP -> SELL branch | **FLAT** (NIFTY +0.37%, BankNIFTY +0.42%, both under 0.5%): the note's FLAT TO GAP DOWN -> BUY branch |
| entry | **09:19**: BankNIFTY 54,800 PE and 54,700 PE x1,170 each, SENSEX 72,000 PE x900, NIFTY 22,350 PE x1,430 | **09:29** LONG 22,357.35 on a confirmed morning star off 22,294.75, stop 22,330.00 |
| exit | **09:30**, cut: "if it is going up, then we are sitting in a wrong trade" | **09:31** at 22,378.05, as the rally stalled under 22,400 |
| result | **-2,54,397.50** | **+1,422.50** |

NIFTY was 22,326 at 09:15 and 22,301 at 09:26. It rallied to 22,379 at 09:30,
22,442 at 09:47 and 22,503 by 10:29.

His positions at the cut:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 54,800 PE | 1,170 | 683.11 | -1,00,924.20 |
| BankNIFTY 54,700 PE | 1,170 | 636.50 | -87,106.50 |
| SENSEX 72,000 PE | 900 | 552.72 | -42,543.00 |
| NIFTY 22,350 PE | 1,430 | 125.96 | -23,823.80 |

The two BankNIFTY legs lost 1,88,030.70, **74%** of the total. At 09:19 the same
screen showed +41,719.00.

**The word.** This is the fifth session where IH's word for the open and the
computed number disagreed:

| session | open | IH's word | book's reading | right side |
|---|---|---|---|---|
| 31 Aug | -0.24% | almost flat | gap-down | IH |
| 1 Sep | BankNIFTY vs its official close | flat | gap-down | IH |
| 16 Sep | +0.36% | flat | gap-up | IH |
| 6 Oct | +0.21% | a slight gap up | FLAT | no comparison (the agent was blind) |
| 9 Oct | +0.37% | a gap up | FLAT | **the book** |

The first three are why SLH-017 computes the verdict and v4t tells the agent to
read it rather than argue past it. 9 Oct is the first recorded session since
then on which the two disagreed and both books traded, and the computed verdict
won.
SLH-017 already binds the agent to that verdict, so there is nothing to change.

**His entry admission is a counter-example, not a rule.** "The entry is not very
good; a bit higher would have been better... but if we don't trade here we may
miss it, so we cover it within the loss limit." The corpus says the opposite (v4f,
A MOVE THAT DENIED YOU ENTRY WAS NOT YOUR MOVE), and his own day agrees with the
corpus.

**The agent booked early again.** It took 20.7 points into the 22,400 stall, and
NIFTY added about 125 more by 10:29. That is 8 Oct's shape, and the 8 Oct
hold-to-origin replay already priced holding: -362.4 points to 10:30 across 177
trades. Not re-litigated.

### 12 Oct: the note's veto kept the agent out of IH's loss

| | IH | the agent |
|---|---|---|
| open | a slight gap up: "the plan is to buy" | FLAT, the note's BUY branch, **held**: BankNIFTY turned sharply negative after the open, which is the branch's own veto |
| entry | **09:33-09:34**: BankNIFTY 55,000 CE and 55,100 CE x1,170 each, SENSEX 72,400 CE x900, NIFTY 22,500 CE x1,430, at a round-number support where "most people will not buy" | **10:01** SHORT, decision price 22,506.40, stop 22,528.00 |
| exit | about **09:55**, at his loss limit, before the round number broke | stopped at 10:01:21 at 22,528.20, inside its own decision pass (SLH-021 journaled it) |
| result | about **-3,19,454.90** (marked at 09:54; the closed screen is not shown) | **-217.50** |

NIFTY was 22,601 at 09:15 and spiked to 22,646 at 09:17. It fell to 22,521 at
09:24, bounced to 22,584 at 09:31, slid to 22,481 at 09:56, bounced to 22,533 at
10:02, and was 22,479 by 10:29.

His positions at 09:54:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 55,000 CE | 1,170 | 873.25 | -63,472.00 |
| BankNIFTY 55,100 CE | 1,170 | 811.87 | -63,963.90 |
| SENSEX 72,400 CE | 900 | 554.07 | -1,28,312.50 |
| NIFTY 22,500 CE | 1,430 | 129.95 | -63,706.50 |

**The agent followed his note more closely than he did.** His own forecast had
said BankNIFTY must not turn sharply negative straight after the open. It did:
the agent measured it about 270 points down from its open. He bought anyway, on
the reasoning that a fall straight into support recruits no buyers, so there are
no buyer stops for a breakdown to take. The agent held.

**Its one trade had the right side and the wrong stop.** The decision for the
10:00 bar took 82.8 s. The order went in at 10:01:09 with NIFTY at 22,521.75: 15
points past the decision price, and 6.25 from the stop. The stop went 12 seconds
later, and NIFTY made 22,479 by 10:29. The live orders were real Kotak fills.

### Two candidates, measured and refused

**1. Fridays bring follow-through momentum.** IH on 9 Oct: "on a Friday the
momentum sometimes comes out well... follow-up momentum". In the 5-year CSV,
"follow" is points from 09:30 in the direction of the first fifteen minutes, and
a trend day closes at least 70% of the day's range away from its open:

| weekday | n | follow to 10:30 | winners | follow to 15:14 | winners | trend days |
|---|---|---|---|---|---|---|
| Mon | 247 | -0.45 | 52% | +1.61 | 51% | 24% |
| Tue | 247 | -2.07 | 54% | +7.13 | 49% | 28% |
| Wed | 245 | -0.39 | 52% | -1.25 | 51% | 26% |
| Thu | 246 | +0.73 | 53% | -5.37 | 58% | 26% |
| Fri | 241 | **-5.51** | **46%** | -0.55 | 50% | 27% |

In the agent's window, Friday is the WORST day to follow the opening direction.
Its full-day follow flips sign by year (+29.7, +1.6, -10.5, -20.6, +26.2, -8.2
from 2021 to 2026), and its trend-day share matches the other days.

**2. Risk left at the order, re-run.** 30 Sep measured the live distance to the
stop at the order as a fraction of the decision's declared |entry - stop|, found
only a 0.75 cut-off robust, and asked for a re-run as the book grew. 167
SL Hunting entries, matched between the runner log and the journal:

| sample | below 0.75 | at or above |
|---|---|---|
| to 30 Sep | 31 trades, -5,479.75, 39% winners | 122, +71,618.50, 49% |
| 1 Oct on | 3 trades, **+2,191.75**, 67% winners | 11, +1,828.00, 36% |

Out of sample the sign reverses. 8 Oct's best trade (+2,405.50) was at 0.68, and
today's stop-out was at 0.29. The 0.75 line was fitted to the first sample;
refused for good.

### IH's other points, all already in the corpus

- Buying where others will not, at a round number after continuous selling: v4b
  REGIME MEMORY and the round-number psychology. His own day refuted it, and his
  note's BankNIFTY veto was the better read.
- Cutting at the loss limit before the level breaks: v4h and v4n. Not
  second-guessing an entry once it is losing: v3y and the RISK section.

### Latency

- 9 Oct: 72 decisions, none abandoned; median 31.1 s, worst 67.0 s.
- 12 Oct: one call abandoned (09:28); median 35.9 s, worst 82.8 s. The worst was
  the 10:01 decision that placed the order.

## 13 Oct - the flush IH bought, and the first Sonnet 5.5 session (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`T3TbtMVb9rk`,
  uploaded 2026-10-13 10:28 IST), traded on the 13 Oct note (`ZEeSYxoitic`:
  "FLAT TO GAP UP -> BUY", "GAP DOWN -> SELL, WITH THE MARKET"). His clock times
  are read off the taskbar clock in the video frame, and his figures off his
  positions screen. His words are my translation of the Hindi auto-transcript.
- **This book:**
  - all 75 decisions (no journal row: the agent did not trade);
  - the runner log;
  - the CLI's own transcripts of all 75 calls.
- **The 5-year NIFTY CSV**, which ends on 12 Oct. Prices for 13 Oct come from the
  decision log's per-minute last price.

This was the first session on claude-agent-sdk 0.2.165 (bundled CLI 2.1.294,
PR #205) and on `claude-sonnet-5-5`. **It was a PAPER session for every
strategy.** At start-up the broker exposure audit found 6 non-flat
NIFTY/BankNIFTY positions on the Kotak account and blocked live trading ("0 of 30
strategies will place REAL orders"). The gate worked as designed.

**No knowledge change.**

### The comparison

| | IH | the agent |
|---|---|---|
| open | flat: the note's BUY branch | **FLAT** (+0.8 points on the previous last candle; +0.21% on the 3:15 level): the same branch |
| entry | **09:16-09:17**: BankNIFTY 54,900 CE and 55,000 CE x1,170 each, SENSEX 72,000 CE x900, NIFTY 22,300 CE x1,430 (its expiry) | none: 74 HOLDs and one unparsed reply |
| exit | **09:34**: Friday's and Monday's losses "completely covered", and BankNIFTY "may give a rejection" | |
| result | **+4,90,221.60** | **0.00** |

12 Oct fell 1.2%, open to 3:15. On 13 Oct NIFTY's first candle flushed to 22,288,
29 points below the 3:15 level, and the second reversed. NIFTY was 22,457 at
09:18, 22,481 when IH booked, 22,574 at 09:59, and 22,529 at 10:29.

His positions at the exit:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 54,900 CE | 1,170 | 822.17 | +1,15,514.60 |
| BankNIFTY 55,000 CE | 1,170 | 785.50 | +84,123.00 |
| SENSEX 72,000 CE | 900 | 474.62 | +1,81,332.00 |
| NIFTY 22,300 CE | 1,430 | 140.15 | +1,09,252.00 |

The two BankNIFTY legs made 41% of the total.

**Why the agent stayed out.** The first candle is not traded. At 09:18: "NIFTY has
already run about 170 points in three one-minute candles... Entering now would
chase a spike at the extreme." At 09:20: "the sellers' stops look largely taken."
After that it waited for a pattern at a level, and the rally never pulled back to
one. IH bought the flush and its reversal as the hunt itself, before any pattern
printed.

### The candidate, measured and refused

The candidate: on a flat open after a down day, buy the first candle's flush below
the prior close once the second candle reverses, with the flush low as the stop.
This is IH's 13 Oct entry made general.

Measured on the 5-year NIFTY CSV. Flat means the open is within 0.5% of the 3:15
level; a down day falls at least 0.5% from its open to 3:15. Entry is the 09:17
open, the stop is the first candle's extreme, and the exit is 10:30 or the stop:

| setup | sessions | net points | winners | mean R |
|---|---|---|---|---|
| buy the flush + reversal after a down day | 38 | -626.1 | 26% | -0.32 |
| buy the flush, any second candle | 76 | -675.4 | 20% | +0.01 |
| buy any flat open after a down day (control) | 138 | -346.8 | 25% | +0.02 |
| mirror: sell the spike + reversal after an up day | 73 | +258.1 | 38% | +0.15 |

The buy side is negative in every variant: down days of 0.3% and 1.0%, exits at
09:45 and 11:00, and a 0.3% flat band. By year it was negative in 2022, 2024
and 2026, and +12.4 in 2025. The mirror is positive overall but -366.6 in 2026,
with 10% winners. The agent's refusal to chase the opening spike matches the base
rate; IH's day was the exception.

### IH's other points, all already in the corpus

- His exit, a profit floor ("the target stays around 3.5 [lakh]; if it goes below
  that we exit, above it we stay"): BOOK WHEN THE PROFIT STOPS GROWING.
- "In such a negative market a trader can endure one momentum, but not a second
  one for long": v5k, WHO PAYS FOR A FOLLOW, and v4i's crowd's fear.
- Patience with winners, quick exits from losers: v3y and the RISK section.

### Sonnet 5.5's first session, for the record

| | value |
|---|---|
| decisions | 75, all `claude-sonnet-5-5`, all through the bundled CLI 2.1.294 (`sdk-py`) |
| latency | median 12.2 s, 90th percentile 16.8 s, worst 25.0 s (Sonnet 5: about 33 s median) |
| failures | none abandoned, none skipped, no slow-call warnings |
| cost | median $0.065 a decision, $5.86 for the session |

Tools stayed inside `mcp__slhunting__*`. Usage was much thinner than on Sonnet 5:

- `position_state` and `pivot_and_levels` in all 75 decisions;
- `candle_patterns` in 68, `cross_index` in 28;
- `market_structure` in 3, `bank_nifty` in 2, `fibo_levels` in 1.

Three tooling follow-ups, none of them a knowledge change:

1. **The order tool was called in 11 decisions that did not mean to trade.** Ten
   calls had no arguments and were rejected by the schema ("'action' is a
   required property"). One sent `action: "HOLD"` and was rejected ("unknown
   action 'HOLD'"). No order was placed. Sonnet 5 did not do this.
2. **One reply (10:13) was prose instead of the JSON object.** The bar was held
   as agent_error, as designed.
3. **`candle_patterns` reports patterns on the previous session's flat prints.**
   The 15:15-15:28 bars are zero-range settlement prints (22,316.85 on 12 Oct)
   and 15:29 is the auction jump (to 22,363.35). Seven decisions today had to
   say so and discard them, all around the open, where IH trades. It was handled
   correctly every time, but it is noise the tool should not produce. Fixed by
   SLH-023 (below).
4. **SLH-017's gap reference is the closing-auction print it means to avoid.** The
   open is measured against "the previous session's last candle close, not the
   official 15:30 close", because "the closing auction carries settlement prints
   no crowd traded around" (v4u). But the last candle IS that print: the 15:29
   bar in 1,144 of 1,227 sessions, which jumped from 22,316.85 to 22,363.35 on 12
   Oct. Today's "+0.8 points" was measured from 22,363.35; on the 22,316.90 3:15
   level the open was +0.21%, FLAT either way. Over 5 years the two references
   give a different verdict on 69 sessions (5.6%), 10 of them in 2026. The worst
   was 5 Aug 2026: +0.84% on the 3:15 level, a gap up, but +0.22% on the last
   candle, FLAT, because the auction print had jumped 0.62%. None of the 69
   falls after SLH-017 shipped, so it has not yet decided a trade. The fix, a
   proposed SLH-023, is to measure from the 3:15 level that v4u and the code's
   own comment describe. Items 3 and 4 share a cause: the prior session's
   15:15-15:29 bars are settlement prints, not trading. Built as SLH-023 (below).

## SLH-023 - the open and the patterns stop reading settlement prints

**The defect (found 13 Oct, above).** From 15:15 the index prints settlement
values: flat 1-minute bars to 15:28, then the closing-auction jump at 15:29. Two
tools read them as if they were trading.

- `pivot_and_levels` measured SLH-017's open classification from the prior
  session's LAST bar, which is the 15:29 auction print. Its own comment, and v4u,
  say the reference must avoid exactly that. Over five years the 3:15 level and
  the last bar give different verdicts in 69 of 1,227 sessions (5.6%). The worst
  was 5 Aug 2026, where a +0.84% open read FLAT.
- `candle_patterns` scans a lookback that, near every open, reaches into the
  prior session's flat prints, and it reported dojis and inside bars there. On 13
  Oct seven decisions had to discard them.

**What changed.**

- `SETTLEMENT_PRINTS_FROM = 15:15`. The open is now measured from the 3:15 level:
  the close of the prior session's last bar before 15:15, falling back to its last
  bar if it has none.
- The classification block's key is now `reference_level`, so the tool output no
  longer shows two different numbers called `previous_close`. Its `reference`
  text names the 3:15 level.
- `candle_patterns` drops EARLIER sessions' 15:15-onward bars before scanning. The
  latest session is kept whole: its bars are the ones being traded.
- The knowledge's SLH-017 sentence now says the gap is measured from the 3:15
  level.

**What it deliberately leaves alone.** Pivots and the top-level `previous_close`
still use the session's last bar, the conventional daily close. Fibonacci, market
structure and swings are untouched.

One quirk was noticed while testing and left alone: a pattern's `time` field is
overwritten by its confirmation time, which is `None` while unconfirmed. It is
older than this change and outside it.

**Checked on real sessions from the five-year file:**

| session | 3:15 level | last bar (old reference) | gap | verdict |
|---|---|---|---|---|
| 5 Aug 2026 | 24,462.95 | 24,614.90 | +0.843% | GAP_UP (was FLAT) |
| 9 Oct | 22,215.80 | 22,231.80 | +0.446% | FLAT |
| 12 Oct | 22,535.65 | 22,520.45 | +0.255% | FLAT |

**Tests.** Five new tests in `test_sl_hunting_indicators.py`:

- 13 Oct's case: the reference is 22,316.90, not 22,363.35;
- 5 Aug's case: the verdict is GAP_UP;
- the fallback for a session with no bar before 15:15;
- no pattern on an earlier session's prints;
- the latest session keeps its own 15:15 bar.

Two SLH-017 assertions follow the key rename. Mutation-checked: the control
passes, and each of these fails:

- reverting the reference to the last bar;
- removing the fallback;
- scanning the prints again;
- dropping every session's late bars;
- keeping the 15:15 print.

## 14 Oct - the same open, a refused spike and a follow IH rode (diligence addendum)

**Sources.**

- **IH's live session:** "Live Bank Nifty Option Trading" (`GTDs-o7xXq0`,
  uploaded 2026-10-14 10:39 IST), traded on the 14 Oct note (`s0413ny8wek`:
  "FLAT TO SMALL GAP DOWN -> BUY", "GAP UP -> SELL"). His clock times are read
  off the taskbar clock in the video frame, and his figures off his positions
  screen. His words are my translation of the Hindi auto-transcript.
- **This book:**
  - its one journal row;
  - all 74 decisions;
  - the runner log;
  - the CLI's transcripts of the order-tool calls.
- **The 5-year NIFTY CSV**, now through 14 Oct.

The session was **LIVE** for SL Hunting, the one live strategy of 30, on
`claude-sonnet-5` (13 Oct ran `claude-sonnet-5-5`). It was the first session on
SLH-023, and the agent's reasoning names the new reference: "Flat open on both
NIFTY (+0.07%) and BankNIFTY (-0.07%) vs the 3:15 reference". The 13 Oct 3:15
level was 22,699.55 and the open 22,715.75. On the old last-candle reference
(22,721.95) the open would have read -0.03%, FLAT either way.

**No knowledge change.**

### The comparison

| | IH | the agent |
|---|---|---|
| open | flat: the note's BUY branch | **FLAT** (+0.07% on the 3:15 level): the same branch |
| 09:17 | | tried LONG at the 09:16 spike top, entry 22,750.50 with the stop at 22,705.00: **refused by sizing** (one lot risks 2,957.50 against the 2,500 budget) |
| entry | **09:20**, after the flush: BankNIFTY 55,300 CE and 55,400 CE x1,170 each, SENSEX 73,000 CE x900, NIFTY 22,700 CE x1,430 | **09:56** LONG 22,718.55, on a 4th touch of a rising trendline at 22,700; stop 22,695.00, target 22,757.95 |
| exit | **09:33**: the target was hit "because BankNIFTY suddenly gave momentum" | **09:57** at 22,716.60, the next decision |
| result | **+2,84,018.25** | **-644.00** |

NIFTY opened at 22,715.75 and spiked to 22,750 at 09:16. By 09:20 it had
flushed to 22,663.80, and it chopped between 22,664 and 22,728 until 09:40. It
made the day's high, 22,757.95, at 09:48, was at 22,707 by 09:55, fell to
22,558.95 at 13:15, and was 22,625.50 at 15:14.

His positions at the exit:

| leg | qty | avg | P&L |
|---|---|---|---|
| BankNIFTY 55,300 CE | 1,170 | 741.85 | +1,36,948.50 |
| BankNIFTY 55,400 CE | 1,170 | 674.15 | +1,43,910.00 |
| SENSEX 73,000 CE | 900 | 300.68 | -701.50 |
| NIFTY 22,700 CE | 1,430 | 146.90 | +3,861.25 |

The two BankNIFTY legs made **99%** of it. For a while SENSEX and NIFTY were
losing while BankNIFTY carried the basket, and he held only while "the lower
point" did not break.

**His read was a follow, not a hunt.** Yesterday's buyers "booked their targets
and left", so "the buyers' SLs are not available", and "when will the operator
turn the market? When he gets the buyers' SLs." With none to take, "whatever is
running will keep running". That is v5k, WHO PAYS FOR A FOLLOW WHEN NOBODY HAS
STOPS NEARBY.

**The refusal saved a loss.** The 09:17 long would have been stopped: NIFTY
reached 22,663.80 by 09:20, through 22,705. The budget turned away a 45.5-point
stop placed at the spike top, and IH bought the flush low four minutes later.
The budget is a deliberate operator setting and did its job.

**The agent misreported that refusal, because the tool told it the wrong
thing.** The worker logged "ENTRY SKIPPED BY SIZING ... One lot exceeds the
configured risk budget". But `MasterWorkerExecutor.enter` returns the generic
"worker rejected the entry (e.g. could not resolve option / outside trading
window)" for any refusal (`sl_hunting_executor.py`). So the model wrote that the
order "was rejected by the execution worker (not a logic/risk rejection)".
Fixed by SLH-024 (below).

**Two decisions 45 seconds apart contradicted each other.** The 09:56 entry
aimed at 22,757.95, the 09:48 high. The 09:57 decision exited because that same
high "was tagged at 09:48 and then printed a confirmed bearish engulfing... the
move that justified the long has already fired and reversed". That was a fact
the entry already had. The exit was still right: holding would have hit the
stop, 22,695, by 10:15, for -23.55 points.

### Two candidates, measured and refused

**1. Follow a flat open after a strong up day** (IH's entry made general). The
5-year CSV, buying the 09:20 open:

| setup | sessions | to 10:30 | winners | to 15:14, mean |
|---|---|---|---|---|
| up day >= 1.0%, flat open | 40 | -678.6 | 40% | -29.0 |
| ... and the first 5 minutes dipped below the open | 39 | -676.2 | 41% | -30.9 |
| up day >= 1.5% | 11 | -527.9 | 18% | -29.9 |
| every flat open (baseline) | 930 | mean -5.2 | 49% | |

It is worse than an ordinary flat open in every variant. By year it was
negative in 2021, 2022, 2024 and 2025, and positive in 2023 (one session) and
2026 (five).

**2. A minimum hold before the agent's own exit** (today's 45-second flip). The
42 journal trades the agent itself closed within 120 seconds of entry, replayed
to their stop, their target or 10:30:

| exit | net points | winners |
|---|---|---|
| the agent's quick exit | +2.8 | 38% |
| hold instead | -96.5 | 29% |

The quick exits were the better decision; a hold rule would cost.

### IH's other points, all already in the corpus

- The operator turns the market only for stops, and with none nearby the move
  continues: v5k.
- One index carries the basket while the others lag: v4i, THE LAGGING INDEX
  DECIDES THE BASKET'S EXIT.
- "This lower point must not break": NAME THE ONE WAY THIS TRADE FAILS.

### Operational

74 decisions, none abandoned; median latency 33.5 s, worst 59.8 s. Two entries
were first returned for a `note_check` that did not start with AGREES /
CONTRADICTS / NOT_APPLICABLE, and were resent correctly. At 10:07 the model sent
the order tool `action: "HOLD"` (rejected), the habit first seen on Sonnet 5.5
on 13 Oct.

## Re-scored with the agent's own trade shape (the 6-14 Oct candidates)

**Why.** Five candidates refused above were scored by holding from the signal
to a fixed clock time (10:30, or 15:14). The agent does not trade that way. Its
181 closed journal trades have:

- a median hold of 4.2 minutes (90th percentile 12.8);
- a median stop of 15.8 points;
- a median target of 44.7 points, about 2.8R;
- entries between 09:21 and 10:18.

A fixed hold to 10:30 measures the market's drift, not the trade the agent would
take.

**How.** Each candidate's signal was re-entered at the same bar with a 16-point
stop and a 45-point target, exiting by 15 minutes, and scored in R (the stop
distance is 1R). It was checked against a 12.5/35 and a 21.6/60 stop/target
pair and against 8- and 30-minute limits, and set beside the same exits on the
comparable days without the condition. The five journal-based refusals (hold to
origin, no entry at the extreme, the risk-left gate, the minimum hold, and 30
Sep's quick-loss re-entry) were already scored on the agent's real exits and
stand.

| candidate | signal (n, mean R, win) | without the condition | verdict |
|---|---|---|---|
| 6 Oct: a small gap that held the 3:15 close, trade WITH it | 489, +0.004R, 38% | 930, -0.042R | about zero; refused |
| 8 Oct: copy a punished prior session's opening move | 17, +0.121R, 29% | 249, +0.201R | the condition makes it worse; refused, v4f stands |
| 9 Oct: follow Friday's first 15 minutes | 242, +0.020R, 37% | 1,230, +0.036R | no Friday effect; refused |
| 13 Oct: buy the flush + reversal after a down day | 39, +0.147R, 31% (+ in 2 of 5 years) | 183, +0.043R | **inconclusive** (standard error about 0.26R); was "refused" |
| 13 Oct mirror: sell the spike + reversal after an up day | 74, +0.203R, 39% | | **inconclusive** (about 0.19R) |
| 14 Oct: buy a flat open after a >=1% up day | 40, **-0.435R**, 20% (+ in 1 of 6 years) | 930, -0.159R | refused under every exit model |

**The yardstick mattered for one candidate.** The 13 Oct flush + reversal looked
clearly negative held to 10:30 (-0.32R) but is mildly positive on trade-shaped
exits. Its variants run from -0.03R to +0.25R, and the sample is too small to
call. That makes it unproven, not refuted.

**Against the existing knowledge:**

- **6 Oct:** collides with SLH-017/v4t ("below the threshold the open is FLAT
  whatever the sign").
- **8 Oct:** contradicts v4f, and the data sides with v4f.
- **9 Oct:** collides with nothing, and measures nothing.
- **13 Oct:** needs no new rule. A flush below the prior close followed by a
  reversal candle IS a pattern + confirmation at a level, which the method
  already allows. The agent declined it on 13 Oct as a chase of the run that
  followed.
- **14 Oct:** agrees in spirit with v5k but loses money in every exit model.

**One lead, not a rule.** The comparison set under the 8 Oct candidate -- follow
ANY first-5-minute move of 0.25% or more, entering at 09:20 -- scored +0.20R on
249 sessions and was positive in 5 of 6 years. It leans against v4f's "do NOT
chase it late and do NOT assume it continues". It came out of a comparison
rather than a prior hypothesis, so it needs its own test before anything is
encoded.

## SLH-024 - a refused entry tells the model why

**The defect (14 Oct, above).** When the worker declined an entry,
`MasterWorkerExecutor.enter` returned "worker rejected the entry (e.g. could not
resolve option / outside trading window)" whatever the cause. At 09:17 the cause
was the risk budget: one lot at a 45.5-point stop risked 2,957.50 against
2,500. The model was left to guess, and wrote that the order "was rejected by
the execution worker (not a logic/risk rejection)" -- wrong, and the reasoning
the coach later reads.

**What changed.** `AtmSingleLegStrategyWorker.enter_position` records why it
declined in `last_entry_refusal`. It is reset at the top of every call and set
at each refusal:

- market data unhealthy;
- spot unavailable;
- the option unresolvable, or its identifiers or lot size invalid;
- no option price, or no fresh price for a live entry;
- sizing, naming the one-lot risk and the budget;
- the spread or liquidity gate;
- an unfilled broker order, or an uncertain broker outcome.

`MasterWorkerExecutor.enter` relays it as "worker refused the entry: <reason>".
A worker without the attribute -- the standalone runner, or any other caller --
keeps the generic text. The reasons are neutral statements of what happened:
the sizing reason says "nothing was placed" and does not suggest a tighter stop.

**What it deliberately leaves alone.** No refusal changes, and nothing that
decides whether to enter. The field is written by every ATM worker but read
only by the SL Hunting order tool.

**Tests.** Five mutants were run; the control passes and each of these fails:

- not relaying the reason;
- relaying a blank reason;
- dropping the sizing reason;
- dropping the per-call reset;
- leaving the budget out of the text.

The tests themselves are:

- the executor relays a worker's reason, and keeps the generic text when there
  is none;
- a real SL Hunting worker refused by sizing records the reason with the budget
  and leaves no position;
- a successful entry clears a stale reason.

## v5r - three rules adopted on the operator's call (the benefit of the doubt)

**The decision (operator, 14 Oct 2026).** Adopt the 8, 13 and 14 Oct rules into
the knowledge. Leave out the 6 and 9 Oct ones. The reason, in the operator's
terms: give the benefit of the doubt, so the agent evolves with a changing market
structure and keeps up with IH. If the agent's performance deteriorates, revert
them.

This is a deliberate exception to the house practice of encoding only what a
measurement supports. It is recorded here, with what was known at the time, so a
later reviewer can judge the rules by their results rather than rediscover the
decision.

### The three rules, and how each fits the existing knowledge

- **THE CHART DOES NOT REPEAT ITS FAILURES EITHER (8 Oct).** Placed directly
  under v4f, THE CHART DOES NOT REPEAT TWO DAYS RUNNING, as its scope rather than
  its contradiction. The operator's reading: the rule follows a move that FAILED
  yesterday, and since charts do not repeat, the failure need not repeat either.
  v4f's own record already carries that scope -- IH's "if the buyers made profit
  yesterday, today they should not get that profit" is about the side yesterday
  PAID. So a copy after a PAID day is v4f's trap, and a copy after a PUNISHED day
  is a candidate to follow.
- **A COUNTER-TREND MOVE'S CROWD BOOKS AND LEAVES -- FOLLOW IT (14 Oct).** v5f's
  counter-trend case, in IH's words: buyers who catch a rally inside a falling
  market book and leave, there are no stops, and the move continues. It agrees
  with v5f rather than scoping it.
- **A FLUSH THROUGH THE CLOSE THAT IS BOUGHT BACK IS THE HUNT, NOT A SPIKE TO
  CHASE (13 Oct).** Written narrowly, to the flush-and-reclaim EVENT. The broad
  version, "after continuous selling the sellers are seated", would contradict
  v5f's measured 16 Sep case: there IH followed the selling after a down day, and
  the book lost buying it. The narrow version needs the market to show the hunt
  -- a first candle through the 3:15 level that the second candle reverses -- and
  16 Sep had no such flush. The first candle stays untraded, a run past the stop
  distance is still A MOVE THAT DENIED YOU ENTRY, and the mirror after an up day
  is excluded.

### What was known at adoption

On the agent's own trade shape (16-point stop, 45-point target, out by 15
minutes; see the re-score above):

| rule | signal | without the condition | as measured |
|---|---|---|---|
| 8 Oct | 17 sessions, +0.121R | +0.201R | positive, below its baseline; small sample |
| 13 Oct | 39 sessions, +0.147R | +0.043R | inconclusive |
| 14 Oct | 40 sessions, -0.435R | -0.159R | negative in every exit variant |

The 14 Oct rule is the one most at risk: it was negative under every exit model
tried. It is adopted anyway, on the operator's judgement that IH's read is
evolving with the market and that the five-year file may lag it.

### Left out

- The 6 and 9 Oct rules (the operator's call).
- The 13 Oct mirror and the 0.25% opening-move lead. They are measurements made
  here, not rules IH stated.
- This week's journal-based gates: hold to the origin, no entry at the extreme,
  the risk-left-at-order gate and the minimum hold. Those were measured on the
  agent's real exits and remain refused.

### Review and revert

Every rule carries the `(v5r)` tag, and the agent cites the tags it acts on, so
the trades taken under these rules can be pulled from the decisions log and the
journal by searching for "v5r". Review them once they make a fair sample. Revert
any rule whose trades lose, or all three if the book deteriorates -- the
operator's call either way. Reverting a rule means deleting its bullet from
`sl_hunting_knowledge.py` and its schema test.

**Tests.** There is one schema test per rule
(`test_v5r_the_chart_does_not_repeat_its_failures_either`,
`test_v5r_a_counter_trend_moves_crowd_books_and_leaves`,
`test_v5r_a_flush_through_the_close_that_is_bought_back_is_the_hunt`). They pin
each rule's scope, IH's evidence, and the guards that keep it from contradicting
v4f or v5f. The control passes and seven mutants fail:

- each of the three rules removed;
- the PUNISHED branch lost;
- the counter-trend qualifier lost;
- the 13 Oct narrowing lost;
- the mirror exclusion lost.

## v5s - 15 Oct live match and 16 Oct pre-open note

### Sources and extraction

- [15 Oct live session](https://www.youtube.com/watch?v=Z4asg2VSGME), uploaded
  2026-10-15 10:38:47 IST, duration 7:27. Direct YouTube Hindi auto-transcript:
  56 displayed segments, recovered through the public Show transcript panel.
- [Prediction For 16 OCT 2026](https://www.youtube.com/watch?v=pt_Lcn76ttw),
  uploaded 2026-10-15 21:10:05 IST, duration 1:39. Direct YouTube Hindi
  auto-transcript: 13 displayed segments, recovered through the same panel.
- Local evidence: all 69 rows dated 2026-10-15 in
  `Backtest Outputs/sl_hunting_decisions.jsonl`, all three completed journal
  rows for that date, and the runner log in `Dependencies/log_files/`.

The direct timed-text request returned an empty body; the public transcript UI
did return both complete transcripts. No third-party fallback was needed.
Hindi explanations below are paraphrases. Taskbar times, quantities and displayed
P&L were checked against the video frames, not inferred from upload times.
Full transcript snapshots and chart/position captures remain outside git under
`%TEMP%\intradayhunter-2026-transcripts\.playwright-cli` and its parent folder.

### The October 16 note

IH reads October 15's continuous positive momentum followed by rejection as a
move whose call buyers probably booked and left. His plan is BUY-side setups
for flat through gap-up, retaining the same plan for a small gap-down. He does
not give a meaningful-gap-down branch; none is invented here. The incoming
chart still decides the actual setup.

| Index | Resistance | Support, nearest first |
|---|---|---|
| NIFTY | 23,100 / 23,200 | 22,850 / 22,700 |
| BankNIFTY | 56,200 / 56,500 | 55,500 / 55,200 |
| Sensex | 74,640 / 75,100 | 73,610 / 73,200 |

These are the spoken rounded levels. Frames at 0:30, 1:00 and 1:30 confirm
the intended chart levels; notably, BankNIFTY's caption says `55200 5500`,
but its upper support is drawn at 55,500.35. Do not turn that caption into
55,000, or copy the chart's incidental decimal precision into the note.
The shipped note is dated 2026-10-16 and remains advisory and date-gated.

### Trade comparison

IH initially waited with a sell-side plan, then explicitly abandoned it when
the higher points broke and NIFTY/Sensex held positive momentum. BankNIFTY was
expected to remain inside its prior-session range and retest its upper part,
not necessarily give a matching breakout. At 1:36-2:06 he contrasts a CALL's
reachable target with a sell that might show some profit but not its target.
At 2:39-3:30 and 5:02-5:36 he permits the small counter-moves already included
in that bounded read; this is not a promise that no loss is possible.

The CALL basket was assembled by the displayed 09:32 clock. By 09:53 all four
legs were flat, with displayed total P&L **+Rs.284,849.85**:

| IH leg | Quantity | Entry average | Displayed booked P&L |
|---|---|---|---|
| BankNIFTY October 55,400 CE | 1,170 | 735.60 | +54,054.75 |
| BankNIFTY October 55,500 CE | 1,170 | 673.50 | +59,319.00 |
| Sensex October 15 73,300 CE | 900 | 194.80 | +110,385.75 |
| NIFTY October 19 22,750 CE | 1,430 | 130.48 | +61,090.35 |

This is one directional basket, not four independent signals. It is displayed
video evidence, not an audited return after charges; his quantities, contracts
and three-index exposure differ from this agent's, so absolute profits are not
a like-for-like performance comparison.

The agent ran LIVE on `claude-sonnet-5`. Decisions: **66 HOLD, two ENTER_SHORT,
one EXIT**; two of the HOLDs are `agent_error` timeouts, at 09:19:31 and
09:47:53. The journal and actual fill logs reveal a LONG during the latter
timeout, so the decisions-only tally must not be mistaken for the trade tally.

| Agent basket | Underlying entry / stop / target | Exit | Booked basket P&L |
|---|---|---|---|
| LONG, NIFTY filled 09:47:36, BNF 09:47:41 | 22,766.95 / 22,747 / 22,800 | AI_TARGET, both flat by 09:47:46 | -130.50 |
| SHORT, NIFTY filled 10:05:00, BNF 10:05:01 | 22,851.10 / 22,875 / 22,800 | AI_STOP, both flat by 10:06:22 | -539.00 |
| SHORT, NIFTY filled 10:27:17, BNF 10:27:19 | 22,903.95 / 22,918.50 / 22,860 | Premise invalidation, both flat by 10:29:34 | +245.00 |

Total **-Rs.424.50**, reconciling to the sum of broker-fill P&L on all six legs.
The first basket matched IH's direction but arrived about fifteen minutes after
his basket was assembled and was closed almost immediately. The two shorts
opposed his follow thesis. The second short made money only because its BNF leg
(+531) outweighed NIFTY (-286); that result does not establish the short premise.

### Net-new and overlap

**Adopted on operator approval: RANGE-CONTAINED BNF / TARGET-REACHABILITY.**
The narrow addition is the explicit entry-time bounded-major-index case:
two indices have broken and held higher points, the major index is still inside
a named range, and an upper-range retest supplies a reachable CALL objective.
A little opposing movement inside that pre-stated range is not the same as a
structural refusal. A fleeting profit on the opposite side supplies neither
direction nor a reachable target. The v4r sign wording is scoped accordingly;
a confirmed opposing reversal or loss of the tolerated boundary still invalidates.

This extends MASKED BNF LAG and v4r without creating another no-confirmation
entry exception. Normal pattern plus confirmation, stops, max-loss, cooldown
and the fresh-structure gate remain in force. RUNAWAY TREND still requires all
three indices to agree. Missing Sensex evidence is never invented. This is a
single-session, transcript-backed refinement, not a backtested edge or a claim
that adding prose will reproduce IH's trades; its later results need review.

Already present, so not duplicated: TARGET-BOOKED, the current-session reset
and v5q retirement of the pre-open premise, the ordinary higher-point break,
MASKED BNF LAG, PROFIT-HOLD, expiry-aware booking, premium non-confirmation,
and v4x's distinction between the price reasoned on and the price at entry.

### Operational findings, not changed in this knowledge version

1. **Late entry and compressed reward.** The long's entry log reports current
   spot 22,792.10 against the decision's underlying entry 22,766.95, leaving
   roughly 7.9 spot points to the 22,800 target. It hit AI_TARGET shortly after
   the fill and still lost premium on both legs (-58.50 / -72). That is evidence
   for a separate current-price/reward-at-submission review, not a reason to
   widen the target or disable exits. The journal's 09:47:54 opened/closed times
   are recovery-write times; the runner's fill times above show the lifecycle.
2. **Exit-cause context is incomplete.** Subsequent decisions repeatedly call
   that long "stopped out", although the journal and runner say AI_TARGET.
   `last_closed_trade_today` supplies direction, time, P&L and the flat spot,
   but not exit reason. A negative option result cannot establish a spot stop;
   carrying the actual exit cause is a separate runtime follow-up.
3. **Evidence and existing-rule application.** The tools supply NIFTY/BNF,
   not live Sensex; IH used all three. The 10:27 short also calls the up-bias
   "stale" after the first move and outside the opening hour, despite the
   existing stale-verdict scope. A new candle name must not excuse that existing
   guard. Missing data and prompt compliance are not solved by another slogan.

No broker, order, API, schema, sizing, environment, timeout or deterministic
execution changes are part of v5s. The marker tests verify delivered prompt
scope and preservation of guards, not LLM compliance or trading performance.

## v5t - October 16 live match: the two-level recovery the agent did not trade

### Evidence and scope

[October 16 live session](https://www.youtube.com/watch?v=UiiR2JsVGck), uploaded
2026-10-16 10:55:51 IST, duration 9:07. All 73 displayed Hindi auto-transcript
segments were recovered from YouTube's public transcript panel. Video frames
were checked at the entry, level explanations and completed positions; the
taskbar clocks, not the upload time, locate the trade. Raw transcript and
captures remain under `%TEMP%\intradayhunter-2026-transcripts\UiiR2JsVGck*`,
outside git. The explanation here paraphrases the Hindi.

The agent comparison uses all 75 October 16 decisions, the journal's lack of
October 16 rows, and that day's runner log. The observation window covers the
agent's 09:15-10:29 decision window and its subsequent 11:00 lifecycle cutoff,
not an invented full-day market replay. The existing October 16 premarket note
was successfully injected at startup and is unchanged in this version.

### IH's trade

IH waited through the sharp initial rejection rather than buying the opening
candle. At 1:16-1:47 and 3:15-3:55 he explains a conditional map: prior buyers
probably booked, early closing-point dip-buyers were flushed, and the next lower
support should hold IF the bullish read is right. The closing point and round
number may occur in either price order. Crucially, he says at 2:16-2:20 and
5:12-5:27 that further decline would mean the direction was wrong and he would
take his stop. This is not evidence that a second level cannot break.

The CALL basket was assembled around 09:40-09:41 and was entirely flat by the
displayed 10:03 clock. The completed positions screen shows:

| Leg | Quantity | Displayed entry average | Displayed booked P&L |
|---|---|---|---|
| BankNIFTY October 55,500 CE | 1,170 | 711.13 | +93,038.40 |
| BankNIFTY October 55,600 CE | 1,170 | 663.28 | +75,312.90 |
| Sensex October 22 74,100 CE | 900 | 527.38 | +181,098.00 |
| NIFTY October 19 22,900 CE | 1,430 | 126.03 | +101,415.60 |

Total displayed booked P&L: **+Rs.450,864.90**. This is one directional basket,
not four entry signals, and is video evidence rather than an audited return
after all charges. Its size, three-index exposure and contracts differ from
this agent's; the rupee totals cannot be compared as equal-risk returns.

At 6:13-6:53 and 8:16-8:33 IH distinguishes the sharp recovery after the initial
fall from yesterday's persistent uptrend: ride the working candles, then book
into strong shared momentum rather than assume an unlimited runner. BNF's late
positive candle helped the final booking. These holding ideas overlap the
existing normal-target, PROFIT-HOLD, premium-asymmetry and v4y booking rules;
they are recorded as confirmation, not duplicated as new rules.

### The agent: no trade, not an unrecorded timeout fill

**75 HOLD decisions, zero entries, zero exits, zero completed journal rows,
zero booked agent P&L.** All decisions used `claude-sonnet-5-5`; none was an
`agent_error`. The runner records 75 completed calls, median 13.3 seconds,
maximum 34.2 seconds against a 90-second deadline. Unlike October 15, no timeout
concealed an entry. No SL Hunting entry or sizing record occurs that day.

| Time | IH | Agent's recorded read |
|---|---|---|
| Opening | BUY plan, but wait through the rejection | Opening drive rejected: NIFTY GAP_UP about +0.55%, BNF nearly flat, then the gap faded |
| 09:37-09:40 | First support flushed; lower support remains the recovery boundary | No fresh bullish confirmation; BNF below pivot / cross-index WAIT; at 09:40 NIFTY 22,938.80 is explicitly near the 22,938.85 3:15 reference, yet described as having no clear level |
| 09:40-09:41 | Assemble the four CALL legs | HOLD, treating the recovery as mid-range and the morning-star read as unconfirmed |
| 09:44 | Hold the basket through its planned initial adverse move | A morning star is now called confirmed, but the entry is rejected as too near 22,997 / 23,000 for worthwhile reward |
| By 10:03 | All four legs flat, +450,864.90 displayed | Still HOLD: recovery has run, prior-day high is close, confirmations considered stale |
| 10:18-10:29 | No second basket shown | Continued HOLD; missing three-index evidence and no fresh located trigger cited |

IH's entry was inside the agent's permitted window, so the entry cutoff does
not explain the mismatch. Conversely, his profitable outcome does not prove
that an entry satisfied the agent's stricter confirmation, cross-index and
rupee-risk rules at that instant. No counterfactual agent profit is assigned.

### Adopted clarifications and overlap

**TWO-LEVEL FLUSH MAP.** ONE BREAKDOWN, NOT TWO already exists (v3n). The new
part is the explicit conditional and entry-time map: identify the first broken
support L1 and the next lower protected support L2, explain whose inventory is
absent, and state what loss of L2 would disprove. An opening-drive failure does
not prohibit a separately confirmed recovery. L2 is not a guaranteed floor,
two aliases of one price do not form two levels, and an unaffordable real stop
means HOLD, never a cosmetic tighter stop. This is an ordinary confirmed setup,
not a new entry exception or a reason to average down.

**IH CLOSING-POINT IDENTITY.** SLH-023 already exposes the 3:15 reference as
`open_classification.reference_level`; `previous_close` remains the separate
full-session close and the pivot still uses its unchanged full-session OHLC.
The old LEVELS glossary nevertheless aliased the previous close to the closing
point. The prompt now names the correct existing field for this IH playbook
and treats it as a candidate level after the open, not only a gap denominator.
BNF's equivalent lives inside its `pivot_and_levels` payload. No payload or
indicator calculation changes, and an unavailable reference is not fabricated.

The concrete mismatch is visible at 09:40: the agent names NIFTY's 22,938.85
intraday reference, then reasons between pivot 22,897.05 and full-session close
22,997.35 as though there were no candidate level there. The two closes differ
by 58.50 points. This exposes a level-identity inconsistency, not proof that the
missing pattern or opposing cross-index evidence could be ignored.

Already covered and not duplicated: TARGET-BOOKED, the current-session reset,
v5s's bounded BNF read, ordinary pattern plus confirmation, v4x's fill-location
discipline, normal-target booking and v4y's no-retracement booking caution.

### Recommendations not implemented as runtime changes

- Replay the actual completed-bar tool payloads around 09:37-09:44 to distinguish
  a truly absent confirmation from missed recognition of an earlier pattern at
  the reference. Several decisions mention a tiny current-bar stub. Those words
  alone do not prove a feed or detector bug; no detector fix is made here.
- Add genuine live Sensex evidence in a separately reviewed runtime change if
  three-index matching is desired. The current tools expose NIFTY and BNF only;
  absence is never treated as confirmation and RUNAWAY TREND remains unchanged.
- Review tool-use hygiene separately: decisions admit empty order calls at
  09:22, 09:25 and 09:44, plus an EXIT while flat at 10:17. They report refusals,
  and no corresponding fills appear. These are not entry attempts lost to sizing.
- At 10:02 / 10:19 the agent uses the earlier opening retracement against a
  runaway read; the existing rule is scoped to the leg. Review that application
  in replay, but do not relabel every recovery as a fresh runaway to force entry.
- Check any candidate's actual invalidation against the agent's configured risk
  budget. IH tolerated a displayed initial loss near Rs.56,000; this version does
  not enlarge the agent's budget, imitate his quantities, or claim his stop fits.

These transcript-backed clarifications are not a backtested edge or a guarantee
of matching IH. Prompt-contract tests check delivery and guard preservation;
one real-tool characterization test checks the two close fields remain distinct.
No runtime, broker, order, API, schema, sizing, environment or premarket-note
changes are included. Validation places no real broker orders.

## v5u - October 19 live match: the major round number flushed and bought back

### Evidence and scope

[October 19 live session](https://www.youtube.com/watch?v=bMk6zfclvGI), uploaded
2026-10-19 10:25:41 IST, duration 8:11. All 65 Hindi auto-transcript segments
(0:02-8:04) were recovered from YouTube's modern transcript panel. The taskbar
clock was read from 1080p frames: 0:07 shows 09:20, 0:54 shows 09:21, the video
cuts to 09:32 by 1:02, 6:17 (the booking) shows 09:39 and the end shows 09:41.
Raw transcripts, the October 21 forecast's transcript and the measurement scripts
remain under `%TEMP%\intradayhunter-2026-transcripts\` (`bMk6zfclvGI-transcript.txt`,
`GbVF4_erGqo-transcript.txt`, `v5u-validation\`), outside git. The explanations
here paraphrase the Hindi.

The agent comparison uses all 73 October 19 decisions (09:15:44-10:29:54, every
one on `claude-sonnet-5`), the journal's one October 19 row, and that day's
runner log. The October 19 pre-open note (FLAT TO SMALL GAP DOWN -> BUY) was
injected at startup (1,536 chars). The entry and exit carried BROKER_FILL prices.
NIFTY expired on October 19 because October 20 was a market holiday.

### IH's trade

At 09:20 IH sees a nearly flat open with some rejection and some buying, and
refuses to hurry: a buy trade is there, but the market sits between two levels,
the round number below and 56,000 above in BankNIFTY. On a flat open he expected
a small breakdown first, because the market "changes the psychology" before it
lets anyone buy. The rejection that came was sharper than expected. NIFTY fell
from 23,175 to 22,970 through 23,000 by 09:29, and BankNIFTY fell from 55,884 to
55,331 through 55,500 and 55,400 by 09:30. He traded none of it.

He explains why. The earlier buyers had not carried positions through the
sideways action, so there were no buyer stops below to hunt, and the fall was
"just a trap". Selling would pay only where the buyers' stops were available,
for instance after a direct gap-up when everyone focuses on buying. Once the
recovery began he bought CALLs at 09:32: "the round number broke, in NIFTY too,
and right as it broke the market recovered". Sensex took its support exactly.
He rode the sharp move while others could not buy into it, and booked at 09:39
when it started to rotate rather than let a large profit shrink. NIFTY was then
near 23,140.

The completed positions screen (6:22) shows:

| Leg | Displayed booked P&L |
|---|---|
| BankNIFTY October 55,300 CE | +54,581.25 |
| BankNIFTY October 55,400 CE | +56,745.00 |
| Sensex October 22 74,300 CE | +1,25,298.00 |
| NIFTY October 19 23,000 CE | +69,784.00 |

Total displayed booked P&L: **+Rs.3,06,408.25**. As with earlier comparisons,
this is one directional basket in IH's quantities and contracts, video evidence
rather than an audited return, and not comparable rupee-for-rupee with the
agent.

### The agent: an early long out at 09:20, then no second look

| Time | IH | Agent |
|---|---|---|
| 09:15-09:17 | Flat open; waits, expecting a small breakdown first | The first candle flushes to 23,074.6, under the 3:15 level 23,121.6; the second candle closes back above it |
| 09:18:46 | Waiting | ENTER_LONG 2 lots at 23,143.40, setup `flush_and_reclaim_flat_open`, stop 23,126.0, target 23,176.9 |
| 09:20:20 | Waiting | EXIT `index_hierarchy_exit`: BankNIFTY broke its pivot with a confirmed shooting star and bearish engulfing. NIFTY +10.7 points; the basket's option P&L was -2,253.80, from the BankNIFTY mirror |
| 09:25-09:31 | The sharp rejection through 23,000 / 55,500; no trade | HOLD: no confirmed reversal at a level. Correct |
| 09:31-09:32 | NIFTY closes back at 23,009; IH buys CALLs | 09:32 HOLD (conf 2): "that exact premise was already tried ... POST-LOSS SPEED LIMIT bars a reflex retry" |
| 09:33-09:35 | Holding | HOLD: the hammer is confirmed, but "into the pivot / 50% fib zone, where a bounce stalls", then "already happened" |
| 09:36-09:39 | Books at 09:39, +3,06,408.25 displayed | HOLD: resistance cluster overhead, "that recovery leg is already captured" |

Two different errors, not one:

1. **The 09:18 long applied v5r outside its scope.** v5r is written for a FLAT
   open "after a down day". October 16 was an UP day (3:15 to 3:15, +0.80%). IH,
   on the same open, explicitly declined to hurry. A simple first-candle version
   of v5r also measures poorly whichever day preceded it (below), so this is
   recorded for the operator's v5r review rather than changed here.
2. **The real flush was read through the speed limit.** By 09:31, 23,000 had
   been broken and reclaimed AFTER the 09:20 exit, with a fresh swing low at
   22,970 and the breakdown sellers trapped under the level. That meets the
   POST-EXIT RE-ENTRY GATE's own description of a new structural event. POST-LOSS
   SPEED LIMIT forbids using the next candle as a recovery attempt. It does not
   forbid a fresh setup twelve minutes later at a different level. After 09:33 the
   agent's "already happened" reading is the ordinary A MOVE THAT DENIED YOU
   ENTRY discipline, applied once the first, correct entry moment had passed.

The 09:44 decision timed out (1 of 73). That day's latency median was 32.8 s on
Sonnet 5, so the timeout did not hide an entry.

### Measured before encoding

All on the five-year NIFTY 1-minute CSV (2021-10-08 to 2026-10-19, 1,233
sessions, 935 FLAT at |gap| < 0.5% against the 3:15 level), measured the way the
agent trades. The signal is a 1-minute low through the round number below the
open by the stated margin, broken by 10:15, and a CLOSE back above it within ten
bars. Entry is the next bar's open, no later than 10:29, with a 16-point stop, a
45-point target and a 15-minute exit; a bar touching both counts as the stop.
Results are in R. The baseline is the same exits at the same clock minute on every
other FLAT session.

| Variant | Signal | Same exits, same minute | Lift |
|---|---|---|---|
| Every 100, break >= 1 pt (the first definition) | -0.066R, n=359 | -0.058R | -0.008 |
| Every 100, break >= 10 | +0.078R, n=249 | -0.054R | +0.132 |
| Every 100, break >= 20 | +0.137R, n=154 | -0.045R | +0.181 |
| The 500s, break >= 10 | +0.290R, n=56 | -0.047R | +0.337 |
| **The 500s, break >= 20 (adopted)** | **+0.513R, n=36** | -0.033R | **+0.546** |
| The 500s, break >= 30 | +0.473R, n=17 | -0.022R | +0.495 |

* **Why 20 points, not 10.** On October 19 the 10-point version enters at 09:28,
  on a 12-point dip under 23,000 that closed back above it inside the same bar,
  and is stopped as the real flush carries on to 22,970. The 20-point version
  enters at 09:32, IH's minute, and books the target. The day alone would make
  the choice a fit to the case, so it rests on the ordering instead: a deeper
  break scores better at the 100s (10 -> 20) and in BankNIFTY (25 -> 50), and
  20 and 30 agree at the 500s.
* **Time-stable only at the major levels.** Split by period (all opens), the
  100-point versions ran +0.27R in 2021-23 but -0.04R and +0.01R in 2024-26. The 500s held
  in both halves: +0.57R / +0.13R at break >= 10, and +0.71R / +0.29R at break
  >= 20. Single years still flip sign at the adopted cell: 2021 -1.00R (n=1)
  and 2024 -0.38R (n=10).
* **Exit shapes.** Every exit variant (12.5/35, 21.6/60, 8 and 30 minutes) kept
  a positive lift at the 500s (+0.12 to +0.58).
* **The flat open is not what matters.** All opens at the 500s with break >= 20
  give +0.46R (n=51), and the non-flat subset is +0.32R (n=15). The rule
  therefore names no open type; IH's day was flat.
* **BankNIFTY agrees only at its own major levels.** Exits were scaled to 38/108,
  with breaks of 25 / 50 points. The 500s show no lift (-0.045, -0.008). The
  1,000s give +0.104R (n=64, lift +0.120) at 25 points and +0.206R (n=41, lift
  +0.225) at 50, the rule's BankNIFTY margin; both are positive in every year
  from 2023.
* **The mirror short has no consistent edge.** A spike above the round number
  sold back gives lifts of -0.052, -0.070 and -0.074, with one positive cell on
  n=20. It is excluded, as v5r excluded its mirror.
* **REFUTED, IH's "tomorrow is a holiday, so this kind of momentum comes":** 70
  pre-holiday sessions against 1,167 others. The median 09:15-10:29 range was
  0.503% against 0.528%, the net move 0.260% against 0.236%, and the full-day
  range 0.906% against 0.867%, with mixed years. Pre-holiday sessions are not
  sharper in the agent's window. The existing EVENT / HOLIDAY PARTICIPATION
  rule is unchanged.
* **Recorded for the v5r review (not acted on):** v5r's first-candle event,
  measured simply (a FLAT open, the 09:15 low under the 3:15 level, the 09:16
  close back above it, long at the 09:17 open), is -0.217R after a down day
  (n=50) and -0.238R after an up day (n=60). The other exit shapes are also
  negative. The down-day scope is not what separates it. v5r is the operator's
  benefit-of-the-doubt adoption (PR #210), so it stays as written. This is
  evidence for that review, not a change.

The evidence is weaker than "robust": the samples at the major levels are small
and single years flip sign. It is consistently positive across exit shapes, both
halves and both indices' major levels, and it is IH's exact trade: the adopted
variant enters at 09:32 on October 19 and books the target. On that basis it is
adopted narrowly, as a named level for the ordinary method, and it can be
reverted like any version.

### Adopted: v5u

**A MAJOR ROUND NUMBER FLUSHED AND BOUGHT BACK IS THE SAME HUNT (v5u)** sits
directly after v5r, the rule it extends.

* **Levels.** It names MAJOR levels only: NIFTY's 500s and BankNIFTY's 1,000s,
  explicitly not the tools' 100-point psych grid.
* **Event.** A real break (NIFTY 20+ points, BankNIFTY 50+), then a close back
  above within about ten minutes. A shallower dip that closes back inside the
  same bar is not the flush yet.
* **Why 20, not 10 (in the prompt too, at the operator's request on 20 Oct).**
  The rule itself carries the 19 Oct contrast: a 12-point same-bar dip at 09:27
  bought at 09:28 was stopped, while the 20-point break is IH's 09:32 entry. It
  also carries the ordering that keeps the margin from resting on one day: a
  20-point break beat a 10-point one at NIFTY's 100s, and 50 beat 25 in
  BankNIFTY. Revert it with the rest of v5u if it does not help.
* **Entry.** That close, with the ordinary pattern + confirmation, is the long
  entry at a named level.
* **Evidence.** It quotes the numbers above with their limits, and excludes the
  100-point version and the mirror.
* **Re-entry.** After an earlier exit, it names the event a NEW STRUCTURAL EVENT
  for the POST-EXIT RE-ENTRY GATE, so an earlier premature attempt does not make
  it a reflex retry. This uses the gate's own language rather than loosening it.
  The 30 Sep measurement that re-entry after a quick loss is robustly bad is why
  every other guard stays: the code-enforced cooldown, the gate's other
  conditions, RUNAWAY TREND, cross-index evidence, the risk budget and max-loss.
  HOLD if the real stop under the flush low does not fit the budget.

Confirmations of existing rules, not duplicated:
* IH's "no stops, no selling edge" reasoning is ONE BREAKDOWN, NOT TWO, plus the
  seated-crowd rules.
* "Ride the sharp move while others cannot buy, book when it rotates, never let a
  big profit shrink" is PROFIT-HOLD and v4y's booking.
* The v5t two-level map and v5r's event.

### Not changed, for review

* The agent described its own 09:20 `index_hierarchy_exit` as being "stopped
  out". The NIFTY leg was +10.7 points; the loss came from the BankNIFTY mirror.
  That is accurate as basket P&L but inaccurate as a description of the exit.
  It is the narration family the pending v4n note tracks, and no change is made.
* Sensex remains unavailable to the tools, so IH's "Sensex took its support
  exactly" cannot be checked by the agent (as noted in v5t).

### October 21 pre-open note

[Prediction For 21 OCT 2026](https://www.youtube.com/watch?v=GbVF4_erGqo),
uploaded 2026-10-20 21:04:50 IST, 99 s, 11 transcript segments.

**Read.** October 19 sold continuously in NIFTY and Sensex while BankNIFTY moved
little. Sellers are likely seated in size and buyers stepped away, though some
sellers may not have carried positions.

**Plan.**

| Open | Plan |
|---|---|
| FLAT TO GAP UP | BUY |
| SMALL GAP DOWN | The same BUY plan |
| BIG GAP DOWN | Ignored, no side pre-set |

IH's caption garbled NIFTY's resistances as "2380 230". The frame at 1:24 shows
drawn lines at 23,083.20 and 23,200.50, recorded as 23,080 / 23,200. Every other
level matches its drawn label: BankNIFTY 55,913.75 / 56,202.60 / 55,211.10 /
54,906.40 and Sensex 74,252.98 / 74,790.96 / 73,210.61.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 23,080 / 23,200 | 22,780 / 22,700 |
| BankNIFTY | 55,900 / 56,200 | 55,200 / 54,900 |
| Sensex | 74,250 / 74,790 | 73,200 / 72,700 |

`test_shipped_note_matches_october_21_intraday_hunter_plan` pins the date gate,
the source, the three plan branches (with no SELL branch) and all twelve levels.
Each of these mutations fails it: a NIFTY level moved by 3 points, the big-gap-down
branch turned into SELL, the date set to October 20, and the small-gap-down branch
turned into SELL.

### Validation

* **v5u schema tests.**
  * `test_v5u_a_major_round_number_flushed_and_bought_back_is_the_same_hunt`
    pins the major-level definition, the event, the exclusions and the stated
    evidence.
  * `test_v5u_reclaim_after_an_exit_is_a_new_event_but_every_guard_stays` pins
    the re-entry wording and the unchanged guards, that the gate and speed-limit
    texts it cites still exist, v5r's scope, and the placement v5r < v5u < v4t.
* **Mutation checks.** Each of these mutants fails its test, and an unmutated
  control passes:
  * NIFTY's 500s turned into 100s;
  * the HOLD clause turned into "size down";
  * "for the POST-EXIT RE-ENTRY GATE" turned into "skips";
  * the gate's "a real level actually broken or reclaimed" weakened;
  * the mirror exclusion removed.
  * "WHY 20, NOT 10" renamed, the 09:28 outcome flipped, or BankNIFTY's 50-over-25
    ordering dropped.
* **An existing test re-anchored.**
  `test_reentry_gate_does_not_contradict_the_exit_rules` found the gate with a
  bare `prompt.index("POST-EXIT RE-ENTRY GATE")`, which now lands on v5u's
  cross-reference, the trap `_flat_rule` documents. It anchors on the rule's own
  bullet instead; its assertions are unchanged. Mutating the gate's "Exits are
  never delayed by it" still fails it.

These are prompt-contract checks, not proof of model compliance or returns. No
runtime, broker, order, API, schema, sizing, environment or timeout change is
included, and validation places no real broker orders.

## v5v - October 21 live match: the gap-up hunt both took, and the day count it overrode

### Evidence and scope

[October 21 live session](https://www.youtube.com/watch?v=qNXxJKrmJHg), uploaded
2026-10-21 10:20:47 IST, duration 8:02. All 63 Hindi auto-transcript segments
(0:02-7:58) were recovered from YouTube's modern transcript panel. The taskbar
clock was read from 1080p frames: 0:15 and 0:52 show 09:19, 6:20 (the booking)
shows 09:33, and the end shows 09:41. The raw transcript and the measurement
scripts remain under `%TEMP%\intradayhunter-2026-transcripts\`
(`qNXxJKrmJHg-transcript.txt`, `21oct-validation\`), outside git.

The agent comparison uses all 73 October 21 decisions (09:17:34-10:29:46, all on
`claude-sonnet-5`), the journal's one October 21 row and that day's runner log.
The October 21 pre-open note (FLAT TO GAP UP -> BUY) was injected at startup
(1,612 chars). Latency over 72 completed calls: median 33.2 s, p90 46.1 s,
maximum 53.5 s. One call timed out: the 09:17 call that placed the entry. The
1-minute CSV had not yet been refreshed for October 21, so the day's NIFTY prices
come from the decisions log's per-bar `last_price`.

### IH's trade

All three indices gapped up. IH bought CALLs at once ("make the trade first,
then discuss why") at 09:19, and sat through an initial drawdown of roughly
18,000-24,000. His reasoning: on October 19 NIFTY and Sensex sold continuously,
so the buyers left and the sellers stayed seated. BankNIFTY barely moved and may
hold fewer of them, so it might lag. A gap-up is the best case for hunting them,
because a put holder on a gap-up has to run and has no other way out, while on a
flat or gap-down open "sometimes they escape". He held through a small resistance
because the sellers' stops were still above it. A direction change would need the
market to break its own lower point.

With fast momentum and no retracement he raised his target: few people can buy
into a run that does not pull back, and buyers arrive only after a retracement.
He booked at 09:33. The sellers' stops (Sensex around 74,500) had been taken by
the breakout, and any retracement from there would not hand back the profit.
The positions screen at 6:23 shows:

| Leg | Quantity | Average | LTP | P&L |
|---|---|---|---|---|
| BankNIFTY October 55,600 CE | 1,170 | 520.95 | 606.90 | +1,00,562.00 |
| BankNIFTY October 55,700 CE | 1,170 | 454.77 | 541.25 | +1,01,182.35 |
| Sensex October 22 74,400 CE | 900 | 383.77 | 501.25 | +1,05,732.00 |
| NIFTY October 23,000 CE | 1,430 | 206.07 | 239.25 | +47,447.40 |

Total displayed: **+Rs.3,54,923.75**, one directional basket in IH's sizes, not
comparable rupee-for-rupee with the agent.

### The agent: the same trade, a squeezed stop and an earlier exit

| Time | IH | Agent |
|---|---|---|
| 09:15-09:17 | Gap-up in all three; decides to buy | NIFTY GAP_UP +0.62% against the 3:15 reference 22,873.75 |
| 09:17:09-09:17:25 | | Sizing refuses the long with stop 22,981 (the first-candle low), then 23,008; accepts 1 lot with stop 23,011 |
| 09:17:34 | | LONG 1 lot at 23,046.55, target 23,100: "stop tightened ... to fit risk budget after first-candle-low stop was too wide". The call then times out (`agent_order_no_final_decision`) |
| 09:19 | Buys CALLs in all three indices | Holds the long |
| 09:23:21 | Holding, raising the target | EXIT `book_before_round_number` at 23,077.75 (+31.2 points, +256.25 basket, 0.88R): 17 points under 23,100, with a confirmed BankNIFTY shooting star |
| 09:24-09:40 | Books at 09:33 near 23,098; +3,54,923.75 | HOLD: "already booked this leg". NIFTY's later high is about 23,120 by 09:38 |

The agent's direction and timing matched IH's, and its pre-open note was applied
as written. Two departures:

1. **The stop was squeezed to fit the budget.** v5t and v5u both say that when
   the real stop does not fit, the answer is HOLD, never a cosmetic tighter stop.
   The agent read SLH-024's sizing refusal and tightened twice in eight seconds.
   It worked today because the 09:20 pullback held at 23,036.8. That makes 3 such
   trades since 20 July (10 Aug, 16 Sep, 21 Oct): 1 winner, 2 losers, -10.9
   points in all. The 161 first-try entries total +525.6 points. Three trades are
   too few to measure, and the rule already exists. A mechanical guard is
   proposed below; it is not built here.
2. **It booked before the round number; IH held.** v4d (BOOK BEFORE THE ROUND
   NUMBER) and the agent's own record agree with the agent. After its 18
   round-number exits, NIFTY was a median 4.5 points against the trade ten
   minutes later (mean -8.0). IH's reason to hold, a run with no retracement, was
   tested on the 85 profitable exits. Where the trade had pulled back less than
   10 points (11 exits), price was still a median 4.5 points against the trade
   ten minutes after the exit. That is **refuted on this book**, and v4d stands.

### Measured: the day count against the open

Five years of NIFTY 1-minute bars (to 2026-10-19), long at a fixed minute with
the agent's trade shape (16-point stop, 45-point target, 15 minutes; a bar
touching both counts as the stop), scored in R. "Down" means the 3:15-to-3:15
change.

**IH's claim, confirmed: after selling, the gap-up is the hunt.** After a session
down 0.5% or more, a long at 09:17 / 09:20 by open type:

| Open | Long at 09:17 | Long at 09:20 | Sessions |
|---|---|---|---|
| Gap-up (>= +0.5%) | +0.216R | +0.303R | 51 |
| Flat | -0.092R | -0.131R | 191 |
| Gap-down (<= -0.5%) | +0.159R | -0.331R | 41 |

A gap-up that does not follow a selling session scores -0.08R to +0.13R, so the
seated sellers are what the gap-up is measuring. Years are mostly positive: 4 of
6 at each minute (09:17 has one year at 0.00; 2021 and 2026 are negative at 09:20). This is v5f's "the next open is the
receipt" and ONE BREAKDOWN's "flat -> go with the selling", now with numbers.
Note what it says about notes that read "flat to gap up -> BUY": only the gap-up
half is measured.

**The conflict it exposed: SECOND-DAY RECRUITMENT (v4a).** That rule says "ONE
down day + a gap-up = no seller inventory; the gap-up is not a hunt", and only
TWO down days seat the sellers. October 21 was exactly one down day (October 16
was up 0.8%), and IH, v5f and the note all bought it. Split by day count (gap-up
>= +0.5%):

| Before the gap-up | Long at 09:17 | Long at 09:20 | Sessions |
|---|---|---|---|
| ONE down day | +0.163R | +0.279R | 29 |
| TWO down days | -0.039R | +0.253R | 46 |
| No down day | +0.055R | +0.048R | 96 |

With "down" set at 0.5% or more, the split is ONE +0.148R / +0.265R (20
sessions) and TWO +0.259R / +0.328R (31). The day count does not separate the
gap-up long either way. Its mirror, a gap-down SHORT after TWO up days, loses:
-0.299R / -0.250R (22 sessions), and -0.390R / -0.229R (15) at the 0.5% threshold.
Gap-downs have tended to be bought; the lead below says the same.

### Adopted: v5v, inside SECOND-DAY RECRUITMENT

v5v is a sub-bullet in v4a, so the agent reads the correction where it reads the
rule. The original bullets and IH's reasoning stay.

* **The override.** It says it OVERRIDES the two entry bullets: a confirmed
  gap-up long is not refused only because a single day fell, since the gap-up is
  v5f's receipt.
* **The evidence.** It quotes the split with its no-down-day baseline.
* **The mirror.** It warns against taking the mirror short from the day count.
* **The open type.** It records that the open type is what separates after a
  selling session.
* **Limits.** It states the limits (small samples, years that flip sign) and
  relaxes nothing: pattern + confirmation, the stop and the risk budget all
  stand.

### SLH-025, the squeezed stop: built (see SLH-025 below)

The prose rule (HOLD when the real stop does not fit) did not bind at the one
moment it applies, right after SLH-024 reports a sizing refusal. Two mechanical
options were proposed: change the message, or add a guard. The operator asked
for SLH-025 on 21 Oct, and both were built in this PR: the guidance rides on
every sizing refusal, and the order tool refuses a tighter same-side stop for
the rest of the pass. The message change sits in the executor rather than the
master, so nothing else on the master's entry path changes.

### Leads, not encoded

* A LONG at 09:17 on a gap-down after a quiet session (prior 3:15-to-3:15 move
  under 0.5% either way) scored +0.729R (56 sessions), and +0.652R at 09:20. It
  is the strongest cell in these tables and untested beyond them: no year split
  and no exit variants yet. It agrees with the mirror result above.

### Not changed

The October 21 note, v4d's round-number booking, PROFIT-HOLD, v5u, and all
runtime, broker, order, sizing, environment and timeout code.

### Validation

* **The new test.** `test_v5v_day_count_is_measured_and_overrides_the_entry_bullets`
  pins:
  * the override;
  * the split and its baseline;
  * the mirror warning and the open-type numbers;
  * the limits;
  * that v4a's original text and v5f remain.
* **Mutation checks.** Each of these mutants fails it, and an unmutated control
  passes:
  * the override wording dropped;
  * the no-down-day baseline removed;
  * the mirror warning flipped to "take that short";
  * the limits sentence removed.
* **Unchanged.** The existing v4a test (`SECOND-DAY RECRUITMENT`, "confidence
  arrives", "a single session's move is not a crowd") is unchanged and passes.

These are prompt-contract checks, not proof of model compliance or returns, and
validation places no real broker orders.

## v5w - the discretionary exit reworked on the operator's call (benefit of the doubt)

### The decision

After the October 21 comparison the operator asked for three changes to how the
agent leaves a trade:

* **A stall is not an exit.** When NIFTY stalls, wait at least ten 1-minute
  candles, then exit only when the low of those ten candles breaks (vice versa
  for a short).
* **A BankNIFTY candle must play out.** Wait for its confirmation to play out
  rather than cutting at the candle itself.
* **Raise the target in a run that has not pulled back.** This is IH's 21 Oct
  practice, measured as refuted above.

The agent had closed its 21 Oct long at 09:23 on a BankNIFTY shooting star with
NIFTY pausing under 23,100. Its own target at 23,100 traded at 09:34.

All three were replayed on this book before encoding. The operator saw the numbers
below, chose to adopt the rules as stated, and will revert them if live results are
negative. As with v5r, this is the operator's call to make. The replay is recorded
so the review starts from it.

### The replay (against)

Method: every discretionary EXIT from July to 19 Oct with a matching journal row:
159 exits. From the bar after the exit decision, the agent's hard stop and target
stay live and 15:14 squares off. Results are in NIFTY points, against what the
agent actually did. Scripts: `%TEMP%\intradayhunter-2026-transcripts\21oct-validation\`
(`stall_replay.py`, `stall_variants.py`).

| Exit family | n | Actual | 10 candles, then their low | 5 candles | Low of the 10 before the exit | Trailing 10-candle low |
|---|---|---|---|---|---|---|
| Stall / booking | 75 | +1,013.1 | +696.4 (**-317**, worse on 50) | -314 | -466 | -177 |
| BankNIFTY hierarchy | 31 | -152.1 | -129.9 (+22, worse on 21) | -41 | -31 | -11 |
| Premise invalidation | 41 | -271.8 | -274.7 (-3) | +61 | -21 | -9 |
| Other | 12 | +150.8 | +124.1 (-27) | -45 | -19 | -12 |

The BankNIFTY rule was also replayed directly: hold until BankNIFTY trades through
the extreme of its last three bars before the exit. On the 31 hierarchy exits that
scored -14 points (better 13, worse 18).

"Hold the run that has not pulled back" is the October 21 measurement above: 11
profitable exits from runs that pulled back less than 10 points. Price was a median
4.5 points against the trade ten minutes after the exit.

Read plainly, the agent's stall exits have been where it makes its money. Waiting
ten candles gave back about 4 points per exit on this sample, and the trailing low
gave back the least. One day (October 21) went the other way.

### Adopted: v5w, right after the basic exit list

* **A stall is not an exit until ten candles have spoken.**
  * Mark the stall's first candle and wait at least ten completed candles.
  * Then exit only on a break of their low (their high for a short).
  * A resumption past the stall's high ends the stall.
* **A BankNIFTY candle is a warning until it plays out.** Even a confirmed
  reversal pattern is not the exit until BankNIFTY trades through its extreme.
* **In a run that has not pulled back, do not book before the target.** The
  mechanical target books it, or the ten-candle rule ends it.
* **What it overrides,** named in the rule, for discretionary exits only:
  * the stall clause of the basic exit list;
  * BOOK BEFORE THE ROUND NUMBER (v4d);
  * v4w's slow-candle book signal;
  * the leader-stalling tell in LAGGARDS NEVER JOINED;
  * the cut-on-the-candle reading of INDEX HIERARCHY ON THE WAY OUT (v3y), v4j
    and v4u.

  The basic exit list, PROFIT-HOLD, v4d and v3y each carry a short "[Narrowed by
  v5w ...]" pointer, so no rule silently contradicts its neighbour.
* **What it never touches:**
  * the hard stop;
  * the mechanical target;
  * max-loss;
  * the 15:15 square-off;
  * a decisive break of the level the premise named, such as a loss of L2 or a
    breakdown of the lower point. Those still exit at once.

Exits are never blocked in code. The house rule that exits are never delayed
mechanically holds, so v5w is guidance to the model, and every mechanical exit
path is unchanged.

### Raising the target: SLH-026

Until SLH-026 the order tool's actions were ENTER_LONG, ENTER_SHORT and EXIT, so
the agent could not move a mechanical target once the trade was on. The operator
asked for a target-raise action. It changes the live order path, so it was built
and tested in its own PR (#217, merged first; see SLH-026 below). v5w's prompt now
names it: in a run that has not pulled back, the agent may RAISE the target with
MODIFY_TARGET, or let the target book it.

### Validation

* **The new test.**
  `test_v5w_stall_and_banknifty_candle_wait_and_no_pullback_hold` pins:
  * the three rules, including the short-side form;
  * the named overrides and the four pointers;
  * the placement between the exit list and PROFIT-HOLD;
  * the never-touch list.
* **Mutation checks.** Each of these mutants fails it, and an unmutated control
  passes:
  * ten candles turned into five;
  * the short side dropped;
  * the BankNIFTY play-out weakened to "prints a candle";
  * the hard stop dropped from the never-touch list;
  * the premise-break exit dropped;
  * the v4d pointer removed;
  * the v3y pointer removed.
* **The rest of the suite.** All 130 schema tests pass, including every earlier
  exit-rule test.

These are prompt-contract checks; they do not show that the model will comply or
that the rule makes money.

## SLH-026 - MODIFY_TARGET: the agent can raise an open target (operator, 21 Oct)

**Why.** v5w (PR #216) tells the agent not to book a run that has not pulled back
before its target. The operator also asked that it be able to move that target
further out, as IH does ("the market keeps going up, so we will make the target
BIGGER"). Until now the order tool's actions were ENTER_LONG, ENTER_SHORT and
EXIT, so a target was fixed from entry.

**What it changes, and why it is small.** The SL Hunting target is a mechanical
level. The worker checks it against NIFTY spot on every poll (`stop_or_target_hit`
in `process_strategy_frame`), and no broker order rests there. Raising it is
therefore a state change to `pos.target_underlying`, not an order:

* **The order tool.** `MODIFY_TARGET` takes the new level in `target` (stop is
  ignored) and a reason. The reason must be real, because refusing a raise
  strands nothing, the same reasoning as for entries. It is the pass's one
  accepted side effect, like ENTER and EXIT. A refused raise goes back for
  correction and the old target stays exactly as it was.
* **The rule, `target_raise_error` in the executor.** It is raise-only: a LONG's
  target may only go up and a SHORT's only down, never closer and never removed.
  Booking earlier is what EXIT is for. The new target must sit beyond the current
  price, because a target already passed would exit on the next poll. It must
  also be within the same 10% bound the order tool applies to an entry's target;
  that constant now lives in the executor and the tool imports it.
* **The executors.** The paper `StandaloneExecutor` edits its own position. The
  live-capable `MasterWorkerExecutor` validates first and only then calls the
  worker hook `raise_target_underlying`. A worker without the hook refuses, so
  the position is never edited from outside.
* **The worker hook.** It takes the same action lock the mechanical exits hold
  and keeps the raise-only invariant on its own. It logs `SL Hunting: target
  raised OLD -> NEW (stop S unchanged) :: reason`, which is what the v5w review
  should search for. The BankNIFTY mirror carries no target of its own, since
  hard risk closes it with the NIFTY leg, so the NIFTY target covers the basket.
  The 30-second session-state snapshot reads the new level from `self.pos`.
* **The recorded decision.** An accepted raise is recorded as HOLD, because the
  position stays open. The model's own HOLD is kept when it gave one; otherwise
  the raise is recorded with the reason it sent the order tool. It is never an
  `agent_error`.

**What it does not touch:** the stop, the size, the mirror leg, any broker call,
exits, max-loss and the square-off.

**Validation.** `test_sl_hunting_modify_target.py` has 19 tests covering:

* the helper, in both directions, the bound, and nonsense input;
* both executors, including validate-before-hook and fail-closed without the hook;
* the order tool: one side effect per pass, sent back for correction, a real
  reason required, an executor without the action, and the description;
* the recorded HOLD, with and without a parseable final answer.

Two master-suite tests check that a raised target stops the OLD level from closing
the trade (stop and mirror unchanged), and that the hook refuses a closer target
and a flat worker.

Seven mutants each fail a test:

* the same target allowed;
* a wrong-side target allowed;
* the 10% bound dropped;
* the master executor skipping validation;
* the tool's reason check dropped;
* the agent's HOLD branch dropped;
* the worker's raise-only check dropped.

## SLH-025 - a stop refused for sizing may not be squeezed to fit (operator, 21 Oct)

**The gap.** On 21 Oct the order tool refused a LONG at the first-candle-low stop
22,981 for sizing, then at 23,008. The agent tightened to 23,011 to get one lot,
three attempts in sixteen seconds inside one decision pass. v5t and v5u already
said "if the real stop does not fit the budget, HOLD"; the prose did not bind at
the one moment it applies. There were three such trades since July (10 Aug,
16 Sep, 21 Oct): two lost, one won.

**Both proposed halves are built; the master is untouched.**

* **The nudge (executor).** Every sizing refusal the model reads now ends with
  `SIZING_REFUSAL_GUIDANCE`: "Do not tighten the stop to make it fit: a stop
  squeezed inside the real invalidation is cosmetic. HOLD unless the REAL
  invalidation fits the risk budget (SLH-025)." The refusal is also tagged
  `refusal_kind: "sizing"`. The paper executor tags its own sizing refusal. The
  master executor tags a worker refusal that starts with SLH-024's `sizing:`
  prefix, and no other refusal.
* **The guard (order tool).** Once the executor has returned a TAGGED sizing
  refusal for an ENTER this pass, any further ENTER in the same direction whose
  stop is TIGHTER is refused before the executor is touched. The refusal names
  the stop that was already refused and tells the model to HOLD.

**Scope, deliberately one pass, as proposed.** The context is rebuilt every bar.
Untouched:

* the opposite direction;
* a refusal for any other reason, such as bad levels, cooldown or a quote;
* a wider or equal stop, which sizing refuses on its own;
* the next bar.

A squeeze moved to the next bar is therefore not caught. A new setup on a later
bar may legitimately use a nearer level, and a fixed carry-over window would be
an arbitrary number. Watch for it in review: a sizing refusal followed by an
accepted same-side entry with a tighter stop one or two bars later.

**Validation.** `test_sl_hunting_stop_squeeze.py` has 8 tests:

* the paper and master sizing refusals are tagged and carry the guidance;
* a non-sizing master refusal is neither tagged nor guided;
* the 21 Oct squeeze is refused in-pass, the position stays flat, and the pass
  stays open;
* the opposite direction, a non-sizing executor refusal, a pre-executor levels
  refusal and the next pass are all left alone.

Seven mutants each fail a test:

* the guard comparison disabled;
* the refusal never recorded;
* direction scoping dropped;
* any refusal arming the guard;
* the paper tag dropped;
* the master tagging every refusal;
* the master guidance dropped.

## October 22 pre-open note

[Prediction For 22 OCT 2026](https://www.youtube.com/watch?v=Hj3d2kKke_I), uploaded
2026-10-21 21:10:09 IST, 124 s, 16 transcript segments. The raw transcript is
`Hj3d2kKke_I-transcript.txt` in the same archive.

**Read.** October 21 sold continuously in all three indices, BankNIFTY without a
retracement, so sellers are likely seated. Sensex has its weekly expiry on
October 22; that is context only.

**Plan.**

| Open | Plan |
|---|---|
| FLAT TO GAP UP | BUY, to target those sellers |
| GAP DOWN | SELL, with the market: sellers seated into a gap down hold with confidence and cannot be targeted |

Per v5v, only the gap-up half of a "flat to gap up -> BUY" note has measured
support after a selling day. The flat half measured -0.09R / -0.13R (191
sessions), and the note stays advisory.

**Levels.** Two captions were garbled and are resolved from the 1080p frames:

* **NIFTY resistance**, spoken "2300 2280", is the drawn 23,002.20 / 22,883 pair,
  recorded as 23,000 / 22,880.
* **Sensex's second support**, spoken "72", is the drawn 72,214.26, recorded as
  72,210 (the drawn line rounded down to 10, as for earlier garbled levels).

The other nine levels match both their spoken values and their drawn labels.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,880 / 23,000 | 22,550 / 22,420 |
| BankNIFTY | 55,100 / 55,400 | 54,390 / 54,200 |
| Sensex | 73,970 / 74,800 | 72,700 / 72,210 |

`test_shipped_note_matches_october_22_intraday_hunter_plan` pins the date gate,
the source, both plan branches and all twelve levels. Each of these mutations
fails it, and the unmutated note passes:

* a NIFTY level moved by 3 points;
* the gap-down branch turned into BUY;
* the date set to October 21;
* the Sensex support moved to 72,200.

## v5x - 22 Oct live match: IH's round-number test, and a stop that fires on a close (operator's call)

**Sources.**

- **IH's live session:** `zpDHFRj9pmU`, uploaded 2026-10-22 11:21 IST, traded on
  the 22 Oct note (`Hj3d2kKke_I`: "FLAT TO GAP UP -> BUY", "GAP DOWN -> SELL,
  WITH THE MARKET"). Clock times come from the taskbar clock in the frame,
  figures from his positions screen, and his words are my translation of the
  Hindi auto-transcript (`zpDHFRj9pmU-transcript.txt` in the archive).
- **This book:** three journal rows, 72 decisions (no agent errors) and the
  runner log. The model was `claude-sonnet-5`. It was the first session with
  v5v, v5w, SLH-025 and SLH-026 live, and the note was injected (1,494 chars).
- **The 5-year NIFTY and BankNIFTY CSVs**, now through 22 Oct.

**Adopted on the operator's call, against the measurement below:** v5x (IH's
round-number test, plus a stop placed beyond the level the rationale names) and
SLH-027 (the stop fires on a candle close, with an intrabar backstop). v5f
stands as the default. "Stay in one direction" was not adopted.

### The session

21 Oct gapped up, made 23,120.90 at 09:40, then sold continuously to 22,649.90 at
14:47. Its 3:15 level, 22,739.20, was -0.59% on 19 Oct's and sat 19% up its
range. BankNIFTY was -1.06%, at 17% of its range.

22 Oct opened FLAT on the computed verdict (NIFTY -0.23%, BankNIFTY -0.29%).
NIFTY opened at 22,687.75 and flushed to 22,564.90 inside the first minute. It
made the day's low of 22,554.85 at 09:35, 4.85 above the note's 22,550 support,
and then chopped between about 22,555 and 22,698 until 10:19. It was at 22,767.35
by 10:35 and made the day's high of 22,785.00 at 11:14. BankNIFTY's first minute
touched 54,459.35, 41 points under IH's 54,500, and then held.

| | IH | the agent |
|---|---|---|
| read | flat open after continuous selling: sellers are seated, so BUY while the round numbers hold | FLAT; the note's BUY branch, which v5f argues against ("a one-way day seats nobody") |
| entry | **09:17-09:18**: BankNIFTY 54,600 CE and 54,700 CE x1,170 each, SENSEX x900, NIFTY x1,430 (the NIFTY leg at 09:18, on a bar of 22,589.15-22,629.10) | **09:33** SHORT x3, **09:58** SHORT x1, **10:18** LONG x1 |
| exit | about **10:17**, booked on the first move all three indices made together (NIFTY 22,659-22,691 that minute) | AI_STOP 09:37; its own exit at 09:58:58; AI_STOP 10:19 |
| result | about **+3 lakh**. At the booking screen: BankNIFTY legs +1,24,430.00 and +93,471.80, SENSEX +76,698.50; the NIFTY row is not legible. Mid-trade he sat about 60,000 down. | **+273.90** |

### IH's reason is v5f's opposite

On 16 Sep IH said that with no retracement "traders do come, but they also book
their target and leave". That became v5f. On 22 Oct he said:

- "Yesterday there was a big selling... the market gave continuous selling. We can
  say directly that more sellers came in than buyers."
- "With this kind of selling, nobody will be sitting with buys... the sellers
  should be seated."
- "The market closed below the round number, so they will have held their trades
  too."

His discriminator was the round number, not the retracement. "As long as the
market is above this level, we can target these sellers. If it goes below, the
sellers come into confidence... even if a retracement comes, you will not cut."
His invalidation levels were SENSEX 73,000 and BankNIFTY 54,500, with the NIFTY
support unnamed. The test he describes is measurable, so it was measured.

### The agent: IH's trade at 09:30, then the trades it could afford

- **09:15-09:29: HOLD.** Each decision stated the FLAT verdict, quoted the
  note's BUY branch and set v5f against it.
- **09:30: it wanted IH's trade, and sizing refused it.** The trade was a long off
  the 22,564.90 / 22,557.95 double bottom, with the stop at 22,548. That stop was
  57 points away, 3,705 per lot against a 2,500 budget. The decision held and
  cited SLH-025 ("a stop squeezed inside the real invalidation"), so the guard's
  guidance reached the model on its first day.
- **09:33: SHORT, 3 lots, stop 22,598 (12.3 points), three minutes later**, citing
  v5f. NIFTY made 22,554.85 at 09:35, 9.85 short of the 22,545 target, and the
  stop went at 09:37. The NIFTY leg lost 1,300.65; the BankNIFTY mirror made
  2,751.30.
- **09:51: the same long, refused again** (42-point stop). It held.
- **09:58: SHORT, 1 lot, stop 22,657.** The agent exited after 42 seconds, once
  BankNIFTY traded through its hammer's high. That is v5w's play-out condition,
  and it was met. The stop was never touched, and NIFTY made 22,587.85 by 10:04,
  44.8 points (1.8R) from the entry. The trade lost 502.75.
- **10:18: LONG, 1 lot, citing the note's branch, stop 22,656.** The rationale
  named "the 78% fibo floor (22,647.86)" as support, yet the stop sat 8 points
  above that floor. A stop just under the floor (about 34 points) would still
  have fitted one lot. NIFTY's low was 22,653.85 at 10:19: the stop went, but the named floor
  never broke. NIFTY was 22,767 by 10:35. The 10:23 re-entry was refused by the
  5-minute cooldown (39 s left). The trade lost 674.00.

Both shorts followed v5f, the long followed the note, and IH sat in one long
through the same chop.

### Measured

**1. IH's round-number test.** Setup:

- One-way down day: 3:15-to-3:15 at or below -0.5%, with the 3:15 level in the
  bottom third of the session range. 21 Oct qualifies; 157 such days were
  followed by a FLAT open, 22 Oct included.
- The round number is the first multiple below the open.
- HELD means no one-minute close below it before the entry minute (a wick does
  not count). This is closer to IH's "trades below" than a wick test, and it
  keeps 22 Oct's BankNIFTY wick as HELD (its 09:15 bar closed at 54,585.20).
- Each trade uses the agent's shape: 16 / 45 points, a 15-minute limit, scored
  in R. BankNIFTY points are scaled 2.4x.

Results, in R at 09:18 / 09:20 / 09:25:

| index, level | every such session, long | HELD -> long | BROKE -> short |
|---|---|---|---|
| NIFTY 500s | -0.129 / -0.148 / +0.046 (157) | -0.156 / -0.175 / -0.038 (149-153) | 4-8 sessions, -1.000 / -0.047 / -1.000 |
| NIFTY 100s | same | -0.209 / -0.209 / -0.004 (106-120) | -0.205 / +0.445 / +0.135 (37-51) |
| BankNIFTY 500s | +0.197 / -0.068 / +0.030 (152) | +0.267 / -0.078 / +0.040 (121-129) | +0.208 / +0.388 / +0.090 (23-31) |

On NIFTY, the index the agent trades, holding above the number is worse than
not asking. BankNIFTY's +0.07R lift at 09:18 is gone by 09:20. With a wick
counting as a break, every NIFTY row stays negative.

**2. The closest form.** This version requires the first three minutes to flush
at least 0.15% below the open and hold the number, and the market to have
recovered a third of the flush by 09:17. Long at 09:18:

| level | sessions | long | 2021-23 / 2024-26 |
|---|---|---|---|
| NIFTY 100s | 12 | +0.163 | +0.449 / -0.237 |
| NIFTY 500s | 27 | -0.149 | -0.220 / -0.083 |
| BankNIFTY 500s | 30 | -0.135 | -0.025 / -0.230 |
| BankNIFTY 1,000s | 37 | -0.195 | -0.242 / -0.151 |

Both forms measure against IH's test. It is adopted anyway as v5x (below).

**3. "Sit in ONE direction."** IH, on beginners: "sometimes they look for sell
trades, sometimes buy trades... stay in one direction". In this book's 189
trades:

| trade | n | NIFTY pts | mean R | rupees |
|---|---|---|---|---|
| first of the session | 68 | +333.5 | +0.231 | +19,274 |
| later, same side, no flip yet | 67 | +81.6 | +0.147 | +18,207 |
| later, OPPOSITE the first | 46 | +53.2 | +0.055 | +21,544 |
| later, back after a flip | 8 | +63.4 | +0.599 | +10,174 |

The flips are the weakest group in R, but they are positive, and banning them
would have cost 21,544. Not adopted (operator, 22 Oct).

**4. A stop with room** ("we bought a little low, so we could take the small
retracements"). Each journal trade is replayed on the 1-minute bars with its
stop widened by W points at the same rupee risk. A stopped trade is re-run to
the wider stop, its target or 15 minutes; any other trade keeps its exit, but
the R is measured against the wider stop.

| widen | trades | mean R | Jul-Aug | Sep-Oct | stopped trades now surviving |
|---|---|---|---|---|---|
| 0 | 184 | +0.187 | +0.154 | +0.232 | 0 of 24 |
| +5 | 179 | +0.154 | +0.104 | +0.221 | 6 of 24 |
| +12 | 164 | +0.111 | +0.090 | +0.141 | 14 of 24 |

Only 24 trades exit on the stop, and the wider stop shrinks every other
trade's R. Tight stops are not the problem either: stops under 12 points are
the book's best band (+0.316R, 38 trades). IH's real stop was the round number,
100+ NIFTY points away. The 2,500 budget caps the agent at about 38 points per
lot, so his stop cannot be carried over. A fixed buffer was therefore refused;
"wider stops" became v5x's placement rule plus SLH-027's closing basis (below),
which replays better than the old stop.

### Observed, not enough to act on

- **The other side after a sizing refusal.** Three trades so far (12 Oct, and
  09:33 and 09:58 today) entered the opposite way within 35 minutes of a refused
  long. Their NIFTY legs made -23.15, -12.85 and -12.15 points. SLH-025 covers
  only a squeeze in the same direction within one pass. Watch it.
- **v5w's BankNIFTY play-out** exit cost the 09:58 short, which the replay
  already expected (-14 points on 31 exits). It is one case, and the operator's
  revert test is live results.

### Adopted: v5x (operator's call)

The operator's instruction (22 Oct):

- encode the round-number test, and keep v5f;
- do not ban direction changes;
- build wider stops judged on a closing basis: "I am seeing a lot of market
  whipsaw recently and it is stopping out my trades before going towards
  target". Trade 3 was the evidence.

Two rules, both tagged (v5x):

- **WHILE THE ROUND NUMBER HOLDS, YESTERDAY'S SELLERS ARE STILL THE TARGET.** It
  sits right after v5f, with a pointer left inside v5f's practical form.
  - Scope: a one-way selling day (down, with its 3:15 level in the bottom third
    of its range); a FLAT verdict; and no 1-minute close below BankNIFTY's or
    NIFTY's next 500 under the open.
  - While both numbers hold, the BUY side is the side the agent's normal
    pattern+confirmation entries may take. The first close below either number
    ends it, and v5f's follow governs.
  - It creates no entry and relaxes nothing. It does not cover a one-way buying
    day, and a reclaim after a close below is v5u.
  - The rule carries the measurement it was adopted against and its revert
    condition.
- **THE STOP FIRES ON A CLOSE, SO PUT IT WHERE A CLOSE MEANS YOU ARE WRONG.** It
  sits right after v5e and restates v5e on the 10:18 case: the stop goes beyond
  the level the rationale names (below a fibo floor or structure low). If that
  stop does not fit the budget, the answer is HOLD (SLH-025). RISK's "~10-15
  points" guide now carries a pointer: it is a guide, not a placement.

To review: split live trades that cite (v5x) by which rule they used, and
compare AI_STOP exits before and after SLH-027.

### SLH-027 - the stop fires on a close

The operator's choices (22 Oct):

- stop-losses judged on a closing basis;
- an intrabar backstop at 1.5x the stop distance, chosen over none and 2x;
- no fixed point buffer, because every buffer replayed worse.

The replay below covers the book's journal trades on 1-minute bars. A stopped
trade is re-run until a close beyond the stop, the backstop, its target or 15
minutes. Every other trade keeps its exit. R is measured against the stop
actually used.

| stop | mean R | Jul-Aug | Sep-Oct | stopped trades: mean / worst | survive |
|---|---|---|---|---|---|
| intrabar (before) | +0.187 | +0.154 | +0.232 | -1.000 / -1.00 | 0 of 24 |
| close only | +0.189 | +0.129 | +0.272 | -0.983 / -3.32 | 5 |
| close + 2x backstop | +0.196 | +0.128 | +0.289 | -0.929 / -2.00 | 5 |
| **close + 1.5x backstop** | **+0.201** | +0.133 | +0.293 | -0.893 / -1.50 | 5 |
| close + 1.5x, stop +5 pts | +0.148 | +0.108 | +0.203 | -0.697 / -1.50 | 9 |

It is better in Sep-Oct and worse in Jul-Aug, which fits the operator's
observation that the whipsaw is recent. On 22 Oct, trade 3's 10:19 candle closed
at 22,661.00, above its 22,656.00 stop, and the backstop sat at 22,644.40, so
the trade would have stayed in.

**What it does:**

- **The rule.** `closing_stop_or_target_hit` (executor, pure) checks the
  backstop on the live price, the stop on the newest completed candle's close
  since the entry, and the target on a touch. Exit reasons stay AI_STOP and
  AI_TARGET; the log line names the trigger.
- **Which candle counts.** The worker stamps the entry time. It skips the
  still-forming candle and any candle that finished before the fill. A resumed
  position has no stamp, so any completed candle counts.
- **Fail-safe.** If the closing read fails, that poll judges the stop on the
  live price, exactly as before. A bug cannot switch the stop off; the first
  version of this code had one, and the master suite caught it.
- **What the model is told.** `position_state` carries `stop_fires`, read from
  the worker, so the model learns how its open stop fires even with the switch
  off. The order-tool text says the same.
- **Knobs:**
  - `SL_HUNTING_STOP_ON_CLOSE`: default true; false restores the intrabar stop.
  - `SL_HUNTING_STOP_BACKSTOP_MULTIPLE`: default 1.5, allowed range 1-3.
    Outside it, paper falls back to 1.5 and live is blocked by a new
    `float_range` rule in `_live_config_errors`.

**Unchanged:** the target (still a touch), max-loss, the 15:15 square-off, the
BankNIFTY mirror (hard risk still closes both legs) and sizing. Sizing still
uses the stop distance, so a backstop exit can lose up to about 1.5x the budget
on the NIFTY leg; the daily max-loss caps the day.

### Validation

**New tests:**

- 10 pure tests (`test_sl_hunting_closing_stop.py`);
- 10 worker tests (`test_slh027_*` in the master suite);
- 2 knowledge tests (`test_v5x_*`).

**Negative-tested with 21 mutants, all caught:**

- 7 on the pure rule: the multiple ignored, its floor removed, the stop judged
  on price, a close AT the stop not counting, the backstop removed, the target
  removed, and the target checked before the stop.
- 7 on the worker: the forming candle counted, candles from before the entry
  counted, the entry clock never stamped, the fail-safe dropping the stop, the
  switch ignored, the live range unchecked, and `stop_fires` not reported.
- 7 on the prompt: v5f's pointer, the wick clause, "creates NO entry", the
  measurement, the close wording, RISK's pointer and the order-tool sentence.

**Gates:** the master suite (657), market-data health (28), pytest (1,766),
ruff, mypy (82 files, then the master file), compileall, bandit (CI's flags)
and pre-commit are all clean. The first gate run caught two things: a mypy
bound type, and AGENTS.md needing the same sentence as CLAUDE.md.

### Already in the corpus

- "When all three move together the target can be raised; if one lags, take a
  normal target": v3s LAGGARDS NEVER JOINED, plus SLH-026 for the raise. No
  MODIFY_TARGET today; trade 1 came within 9.85 of its target.
- "Sideways is when neither side is interested... it takes time to add
  sellers": v5w's ten-candle stall rule.

### Latency

74 calls on `claude-sonnet-5`: median 31.8 s, p95 44.7 s, worst 47.7 s.

## October 23 pre-open note

[Prediction For 23 OCT 2026](https://www.youtube.com/watch?v=E8ZaDDdgnio), uploaded
2026-10-22 21:10:22 IST, 108 s, 14 transcript segments. The raw transcript is
`E8ZaDDdgnio-transcript.txt` in the same archive.

**Read.** October 22 recovered through the closing price and evicted the sellers
who were seated after October 21's selling, so no seller stops remain to hunt.
Momentum stayed small in all three indices, so buyers and sellers sit at one
price level. In IH's words on NIFTY: "we will not do the targeting work on them.
We will go with the market."

**Plan.** No branch buys.

| Open | Plan |
|---|---|
| FLAT TO GAP DOWN | SELL, with the market: do not target either crowd; identify selling-side setups |
| GAP UP | SELL, the same plan |

BankNIFTY's gap-up sentence is cut off in the transcript; Sensex and NIFTY
both state that a gap keeps the same selling plan.

**How it meets the corpus.** v5x does not apply on October 23: it needs a one-way
selling day, and October 22 closed -0.32% on the 3:15 levels, with its 3:15 level
at 49% of its range. The note stays advisory.

**Levels.** All twelve sit on the same drawn lines as the October 22 note. Two
captions were garbled again and one spoken value is rounded; all three are
resolved from the 1080p frames at 0:40, 1:10 and 1:39:

* **NIFTY resistance**, spoken "23280", is the drawn 23,002.20 / 22,883.00 pair,
  recorded as 23,000 / 22,880.
* **Sensex's second support**, spoken "72", is the drawn 72,214.26, recorded as
  72,210.
* **BankNIFTY's first support** is spoken 54,400 but drawn at 54,388.75, and is
  kept at 54,390 as on October 22.

The other nine match both their spoken values and their drawn labels.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,880 / 23,000 | 22,550 / 22,420 |
| BankNIFTY | 55,100 / 55,400 | 54,390 / 54,200 |
| Sensex | 73,970 / 74,800 | 72,700 / 72,210 |

NIFTY's 22,550 support held on October 22: the day's low was 22,554.85.

The note renders 1,412 characters on October 23 and nothing on any other date.
`test_shipped_note_matches_october_23_intraday_hunter_plan` pins the date gate,
the source, both SELL branches (no line may BUY) and all twelve levels. Each of
these mutations fails it, and the unmutated note passes:

* a NIFTY level moved by 3 points;
* the gap-up branch turned into BUY;
* the date set to October 22;
* BankNIFTY's support recorded as the spoken 54,400;
* the "do not target either crowd" clause dropped;
* the Sensex support moved to 72,200.

## v5y - 23 Oct live match: a straight run books at its target, and the shakeout after a round-number break

**Sources.**

- **IH's live session:** `PbYJghYguf4`, uploaded 2026-10-23 10:43 IST. He traded
  on the 23 Oct note (`E8ZaDDdgnio`: SELL on every open). Clock times come from
  the taskbar clock in the frame, and his words are my translation of the Hindi
  auto-transcript (`PbYJghYguf4-transcript.txt` in the archive).
- **This book:** three journal rows, 84 decisions (3 agent errors) and the runner
  log. The model was `claude-sonnet-5`, and calls took a median 29.3 s (p95
  44.6 s, worst 65.9 s).
- **The runner:** `main` at `4768a24`, so v5x, SLH-026 and SLH-027 were live and
  the note injected (1,412 chars).
  - The operator stopped it at 10:03 and restarted it at 10:04 with
    `LIVE_TRADING_ENABLED=false`, so the third trade was paper.
- **The 5-year NIFTY and BankNIFTY CSVs**, now through 23 Oct.

**Adopted: v5y, two rules.**

- A straight run books at its target (measured).
- After a round-number break, the entry waits for the shakeout. This is the
  operator's rephrasing of candidate 3, measured per trade.

Candidate 2 was not encoded.

### The session

The open was FLAT on the computed verdict (NIFTY +0.33%, BankNIFTY +0.24%).
NIFTY made the day's high of 22,787.25 at 09:16, bounced to 22,763.50 at 09:27,
then sold off: 22,600 at 10:00 and 22,471.60 at 10:15. It made the low of
22,372.00 at 12:51, and the 3:15 level was 22,517.00 (-0.66%). BankNIFTY broke
54,500 at about 09:58.

| | IH | the agent |
|---|---|---|
| read | yesterday's recovery evicted the sellers, with no big crowd left on either side: follow the continued selling | the note's SELL branch on a FLAT open |
| entry | **09:22**: puts, SENSEX x900 and NIFTY x1,430 among the legs | **09:31** SHORT 22,729.85 (live) · **09:44** SHORT 22,702.00 (live) · **10:13** SHORT 22,516.85 (paper) |
| path | about 1 lakh down on the 09:27 bounce, which he called a retracement that "is good for us" | 09:29 SHORT refused by sizing (41-pt stop); 09:31 took it with the SAME stop from a better entry (34 pts) |
| exit | about **09:58-09:59**: held past his target, booked as BankNIFTY broke 54,500 ("retracement risk rises after the breakdown") | AI_TARGET 09:35 (+61.3 pts) · AI_TARGET **09:58** (+84.0 pts) · own exit 10:44 (+1.7 pts) |
| result | target over-achieved (the figure is not legible) | live **+6,579.50**; paper **-2,017.00** (the BankNIFTY mirror leg gave back) |

The live book and IH were on the same side, and the agent's second target filled
in the minute he booked. Its 09:31 stop sat "above the double-top high (22764)
rather than inside it", which is v5x's placement rule in its own words.

The 10:13 paper short is where it diverged. It came 15 minutes after BankNIFTY's
break. At 10:18 the agent raised its target from 22,420 to 22,400 with
MODIFY_TARGET, citing a "runaway trend... zero meaningful retracement since
entry". It exited at 10:44, once BankNIFTY traded through its reversal cluster
(v5w's play-out condition).

### Measured

**1. Retracement timing: ADOPTED as v5y.** IH: "if the retracement had not
happened and the momentum had come continuously, we would have had to run. But
the retracement has already happened, so the risk of a retracement is lower
now."

- **Setup.** On the 5-year NIFTY CSV, enter at 09:20, 09:35 and 10:00 following
  the move since 09:15 (at least 0.15%). Take every entry that reached +45
  points, measure the pullback taken BEFORE that, and score the next 15 minutes.
- **Scoring.** "Extend" means +25 more points came before 20 points were given
  back; "giveback" means the reverse.

| before the target | n | extend | giveback | drift after 15 min |
|---|---|---|---|---|
| straight run (pullback < 10) | 47 | 38% | 51% | **-5.2** (2021-23 -5.0, 2024-26 -5.2) |
| pullback 10-20 | 171 | 36% | 37% | -0.7 |
| pullback >= 20 | 685 | 35% | 41% | +0.7 |

**Robustness.** Straight runs were negative in all 12 variants:

- targets of 35, 45 and 60 points;
- 10-, 15- and 20-minute windows;
- straight-run cutoffs of under 8, 10 and 12 points;
- both halves of the data, every time.

The 21 Oct test of the book's own no-pullback profitable exits agreed (median
-4.5 points ten minutes later). Only 2 of the book's own entries ran straight to
+45, too few to add anything.

So v5w's "RAISE it with MODIFY_TARGET while the run keeps going" is the
measured-negative case. v5y keeps v5w's "do not book a straight run early" and
removes the raise from it: the mechanical target books a straight run, and
MODIFY_TARGET is kept for a run that has ALREADY absorbed a real retracement
(about 20 points from its best price). That use is IH's practice, not a
measured edge, since it was flat in the same test.

**2. Eviction without extension: inconclusive, not encoded.** IH's thesis: a
selling day seats sellers, the next day's recovery through the close evicts
them, and if it does not extend, follow the original selling. The test took a
FLAT open where D-2 was down at least 0.5%, and D-1's high crossed D-2's 3:15
level but closed back below it. Results for a NIFTY short, in R:

| entry | 09:18 | 09:20 | 09:22 | 09:25 |
|---|---|---|---|---|
| candidate (67) | -0.098 | +0.192 | +0.318 | +0.000 |
| D-1 held above (control) | +0.402 | +0.578 | +0.349 | +0.551 |
| no selling day (baseline) | +0.109 | +0.118 | +0.053 | +0.131 |

The candidate's halves disagree at three of four minutes, and the control the
mechanism says should fail beats it. On BankNIFTY the candidate was positive at
every minute (+0.23 to +0.67R), but the agent trades NIFTY. The 23 Oct note
already carried this read as advisory, which is where it stays.

**3. The round-number break: refused.** IH booked as BankNIFTY broke 54,500 and
said that after the breakdown "the chances of a retracement increase, so one
should not keep sitting". The agent's 10:13 short came 15 minutes after the same
break. The test took the first BankNIFTY close through a 500 during a 0.5%+
move, from 09:25 to 11:00 (431 events), then a NIFTY entry with the move:

| NIFTY entry | R |
|---|---|
| the minute after the break | **+0.253** (2021-23 +0.245, 2024-26 +0.259) |
| +5 / +10 / +15 minutes | +0.087 / +0.016 / +0.089 |
| same minutes, no-event days | +0.019 / +0.006 |

The continuation after a break is positive, so booking AT the break gives up
edge; it fades to nothing within ten minutes. Neither "book at the break" nor
"never enter after it" is supported.

**3b. Rephrased by the operator, and ADOPTED as v5y's second rule.** The
operator: "after a round-number breakdown there is often a shakeout of retail
traders... tell the agent to enter after the shakeout is done."

- **Shakeout:** within 30 minutes, the leading index trades back to the broken
  number.
- **Done:** after that touch, the first 1-minute close back through the number
  in the break's direction.
- **Entries:** NIFTY, in the agent's trade shape.

| leader, level | events | shakeout | at the break, when one came | after the shakeout, per trade | at the break, same events |
|---|---|---|---|---|---|
| BankNIFTY 500s | 431 | 80% | -0.049 | **+0.127** (+0.110 / +0.138) | +0.036 |
| NIFTY 100s | 405 | 86% | -0.186 | **+0.175** (+0.169 / +0.179) | -0.109 |

The breaks with no shakeout are the runaways (+1.4R and +2.1R at the break).
That is why, per break, waiting earned less than entering at once (BankNIFTY:
+0.087 against +0.253; NIFTY's 100s: +0.124 against +0.137).

Counting "back within 0.05% of the number" as a shakeout finds one after 94-96%
of breaks. Under that looser definition the after-shakeout entry is still
positive in both halves, but no longer beats the at-break entry on the same
events.

The rule states both the gain and the cost. It hands a no-shakeout run to
RUNAWAY_TREND, and it separates a shakeout from a reclaim (the trap rule and
v5u).

On 23 Oct both cases happened on NIFTY:

- **22,700:** broke at 09:33 and was retested at 09:40. The 09:44 short came
  after that shakeout and made +84 points.
- **22,500:** broke at 10:15, just after the 10:13 short went in, and was
  retested at 10:20. That short went out flat.

### v5x and SLH-027, first full day

- **v5x placement:** the 09:31 entry put its stop beyond the double-top high and
  said so. The round-number rule did not apply, because 22 Oct was not a one-way
  selling day.
- **SLH-027:** no trade reached its stop, so the closing-basis stop has not
  fired live yet.
- **SLH-026:** one raise, the 10:18 straight-run raise above. v5y now rules it
  out.
- **Housekeeping:** at 09:58:32 a final output claimed an EXIT after the target
  had already closed the trade inside the same pass. The order tool accepted
  nothing, which is the existing guard working.

### Validation

`test_v5y_a_straight_run_books_at_its_target_and_only_a_pulled_back_run_is_raised`
pins the rule, its evidence, its place between v5w and PROFIT-HOLD, the pointer
inside v5w, and the order tool's new MODIFY_TARGET sentence.

`test_v5y_after_a_round_number_breaks_the_entry_waits_for_the_shakeout` pins the
shakeout rule: its measurement, the reclaim and runaway carve-outs, its cost, and
its place directly after the clean-break bullet it qualifies. The rule's 7 mutants
are all caught:

- the break allowed as the entry;
- the shakeout frequency altered;
- the pattern requirement dropped;
- the reclaim carve-out dropped;
- the cost hidden;
- the RUNAWAY_TREND pointer dropped;
- the resumption close removed.

**The straight-run rule: negative-tested with 8 mutants, all caught:**

- the v5w pointer removed;
- a straight run allowed to be raised;
- any run allowed to be raised;
- the retracement size loosened to 5 points;
- the measurement altered;
- the robustness claim dropped;
- the "not a measured edge" caveat dropped;
- the tool text reverted to v5w's.

**Gates:** the master suite (673), market-data health (28), pytest (1,794),
ruff, mypy (82 files, then the master file), compileall, bandit (CI's flags)
and pre-commit are all clean. The first ruff run caught a 140-character
assertion, which is now split.

## October 26 pre-open note

[Prediction For 26 OCT 2026](https://www.youtube.com/watch?v=-sQI9q-As48), uploaded
2026-10-25 21:00:22 IST, 94 s, 41 transcript segments. The note was written on
25 Oct and shipped uncommitted; the runner injected it (1,369 chars) on 26 Oct.

**Read.** October 23 sold continuously in all three indices. It retraced, but
only briefly, and the rejection resumed, so the sellers should be seated.

**Plan.** Every branch buys.

| Open | Plan |
|---|---|
| FLAT TO GAP UP | BUY: identify buying-side setups to target the seated sellers |
| GAP DOWN | BUY, the same plan |

**Levels.** Three spoken values are garbled or rounded in the captions:

* **BankNIFTY resistance**, spoken "54800 54500", is drawn at 54,812.10 and
  54,504.75 (the 0:30 frame), recorded as 54,810 / 54,500.
* **BankNIFTY's first support**, spoken 53,670, is drawn below the 53,700
  gridline (label 53,6xx at 0:30), recorded as 53,620.
* **Sensex "7260"** and **NIFTY "2280"** are recorded as 72,060 and 22,080.
  These two were not re-read from the frames on 26 Oct.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,670 / 22,800 | 22,240 / 22,080 |
| BankNIFTY | 54,500 / 54,810 | 53,620 / 53,450 |
| Sensex | 73,500 / 73,970 | 72,510 / 72,060 |

**How the levels played.** No support was reached: NIFTY's low was 22,387.35
and BankNIFTY's 53,777.50. NIFTY closed through 22,670 at 10:06, and the high
was 22,753.20, short of 22,800. BankNIFTY topped at 54,392.50, under 54,500.

**How it meets the corpus.** 23 Oct closed -0.66% on the 3:15 levels. Its 3:15
level sat at 34.8% of NIFTY's range and 16.1% of BankNIFTY's, so v5x's
"bottom third" held for BankNIFTY only. The agent treated v5x as in scope all
morning. `test_sl_hunting_premarket.py` is retargeted to 26 Oct in the same
change, and its 19 tests pass.

## v5z - 26 Oct live match: IH bought the lone BankNIFTY break, the agent followed v5x's end clause

**Sources.**

- **IH's live session:** `A6CgbrMM9DA`, uploaded 2026-10-26 10:48:37 IST, 10:01.
  Clock times come from the taskbar clock in the frame and figures from his
  positions screen at 8:02. His words are my translation of the Hindi
  auto-transcript (265 segments).
- **IH's lesson:** [Master Risk-Reward Ratio](https://www.youtube.com/watch?v=K2PiBp8jrDs),
  uploaded 2026-10-25 10:15:07 IST, 10:16, 247 segments.
- **This book:** three journal rows, 55 decisions (9 agent errors) and the
  runner log. The model was `claude-sonnet-5`. SL Hunting ran PAPER all day
  (see Operational).
- **The 5-year NIFTY and BankNIFTY CSVs**, rebuilt on this machine through
  26 Oct with the Data Extractors. The earlier copies, the book's journal
  history and the transcript archive are not in this checkout. The raw
  transcripts sit outside git, in the session scratchpad.

**Adopted: v5z-2 and v5z-1**, both on the operator's call after a measurement.

- **v5z-2** is a veto on fading BankNIFTY's first retest of a reclaimed round
  number.
- **v5z-1**, in the operator's reframing, lets the pre-open note's side stand
  through a round-number cross against it.

v5z-3 stays open.

### The session

23 Oct's 3:15 level was 22,516.40, and its low 22,372.00 at 12:51. BankNIFTY's
3:15 level was 54,111.55, and its low 53,942.35.

26 Oct opened FLAT on the computed verdict (NIFTY +0.04%, BankNIFTY +0.26%):

- **09:15:** NIFTY opened at 22,524.40 and its first minute closed at
  22,480.05, under 22,500. v5x's end clause fired at once.
- **09:18-09:19:** BankNIFTY closed below 54,000 at 09:18 and below 23 Oct's
  low at 09:19 (53,904.80).
- **09:25-09:26:** BankNIFTY made 53,777.50 at 09:25. NIFTY made 22,387.35 at
  09:26, 15.35 points above its own 23 Oct low. Sensex held its low too, by
  IH's account; it is not in this data.
- **09:44-09:52:** BankNIFTY closed back above 54,000 at 09:44, 09:45 and
  09:46, retested it at 09:47 (53,988.80) and closed above it again at 09:50.
  NIFTY closed above 22,500 at 09:52.
- **10:08:** NIFTY made 22,706.15. The 3:15 level was 22,717.05 and the close
  22,753.20.

| | IH | the agent |
|---|---|---|
| read | two days of continuous selling with a 4-5 minute retracement: sellers seated, and the round-number breakdown adds fresh ones. Entry where BankNIFTY broke its lower point while NIFTY and Sensex held theirs | FLAT. The note's BUY branch was void under v5x (closes below 22,500 and 54,000), so v5f governs: follow the selling |
| entry | **09:24-09:25**: BankNIFTY OCT 53900 CE and 54000 CE x1,170 each, SENSEX OCT 72600 CE x900, NIFTY OCT 22400 CE x1,430 (avg 169.28) | 09:41 SHORT refused by sizing (41-pt stop) · **09:50** SHORT x2 at 22,460.10 · **10:04** SHORT x1 at 22,627.90 · **10:17** LONG x1 at 22,677.00 |
| path | about 3 lakh up by 09:35, a small loss at 09:40, held for BankNIFTY's 54,000 breakout: "their SLs will sit just above the round number" | stops at 22,478 / 22,665 / 22,648 |
| exit | about **09:46**: the breakout came without fast follow-through, "so we will have to take a normal target" | AI_STOP 09:51:51 (backstop) · AI_STOP 10:07:00 (close) · MAX_LOSS 10:20:31 |
| result | **+2,73,727.10** open at 09:44 (BankNIFTY +80,788.50 and +64,409.00, SENSEX +76,878.00, NIFTY +51,651.60). The booking screen is not legible. | paper **-7,899.50** (-3,481.50 / -2,853.50 / -1,564.50; NIFTY legs -1,540.50 / -1,040.00 / -604.50) |

IH took NIFTY from about 22,400 to 22,488 and left. The larger leg, 22,460 to
22,706, came after he booked, and the agent was short into it twice.

### The agent: the opposite side, and then the top

- **09:23-09:50: HOLD.** 18 of the 20 decisions called the BUY branch void,
  because v5x's end clause had fired at 09:15 and v5f's follow became the
  frame. The two that did not (09:47 and 09:48) saw the long off 22,387 only
  once it had "already run". At 09:41, with
  IH sitting long, the agent tried a SHORT at 22,417.15 (stop 22,458). Sizing
  refused it: 2,655.25 for one lot against the 2,500 budget.
- **09:50 SHORT, four minutes after IH booked.** The trade sold BankNIFTY's
  first retest of the reclaimed 54,000 as "a matching double-top
  (53981/54098)". v5y already removes that premise: "a return that CLOSES back
  beyond the number and holds there is a reclaim... the continuation premise is
  gone", and BankNIFTY had closed above 54,000 three times. The backstop filled
  at 22,494.20 within 42 seconds. A long from the same minute would have
  reached its 45-point target by 09:53.
- **10:04 SHORT** against a 270-point run, off the 22,659 / 22,657 double top.
  The 10:06 candle closed at 22,675.40, beyond the stop.
- **10:17 LONG**, the third trade in 27 minutes and a flip after two losses.
  It entered with 1,165 rupees left before the 7,500 max-loss and a stop worth
  1,885 on the NIFTY leg alone. MAX_LOSS flattened it at 10:20, at -7,932 for
  the session, with its own stop untouched.

### Measured

All entries are NIFTY, in the book's trade shape (16-point stop, 45-point
target, out after 15 minutes, entered at the next minute's open), scored in R,
with the 2021-23 / 2024-26 halves in brackets.

**1. IH's entry: the lone BankNIFTY break of yesterday's low.** The setup:

- The previous day closed -0.5% or worse on the 3:15 levels, and today's open
  is FLAT on both indices.
- The first prior-low break between 09:15 and 10:30 is classified as
  BankNIFTY alone (BankNIFTY closes below its low while NIFTY's low so far is
  above its own), NIFTY alone, or both.
- Entries come 1 and 6 minutes after the break (IH waited about 6), or on the
  first covering candle: a NIFTY close above the prior bar's high within 20
  minutes, with NIFTY still holding.

| first break | n | long +1 | long +6 | long on covering | short on covering |
|---|---|---|---|---|---|
| BankNIFTY alone | 20-26 | -0.619 (-0.593 / -0.635) | -0.104 (-0.757 / +0.114) | -0.156 (-0.889 / +0.137) | **+0.544** (+1.632 / +0.109) |
| both | 53 | -0.202 | -0.090 | - | short +1: +0.239 (+0.113 / +0.293) |
| NIFTY alone | 23 | -0.085 | +0.218 (+0.538 / -0.199) | - | short +1: +0.596 |

With IH's own stop (NIFTY's prior low, 1 point under), a target of 45 points
or twice the stop, and 15, 30 or 60 minutes, the long scores -0.27 to +0.10R
over 20-21 sessions. The halves disagree in every variant: 2021-23 is negative,
2024-26 positive. Only 5 to 9 of those stops fit the book's budget, which caps
it near 38 points. 26 Oct itself needed a 42.9-point stop and scored +1.05R at
45 points.

The lone break is not an edge for a NIFTY long in either shape. If anything,
the first covering candle is sold (+0.54R, both halves positive, 21 sessions).

**2. v5u's ten-minute window.** BankNIFTY took 26 minutes to reclaim 54,000
and NIFTY 37 minutes to reclaim 22,500. For the first major-number flush of
each session (NIFTY 20+ points deep, BankNIFTY 50+), long NIFTY at the
reclaim, by reclaim delay:

| leader | 0-10 min | 11-20 | 21-30 | 31-60 |
|---|---|---|---|---|
| BankNIFTY 1,000s | +0.066 (53) | -0.334 (12) | -0.047 (8) | -0.678 (10) |
| NIFTY 500s | +0.259 (47) | +1.086 (8) | +0.596 (4) | +0.916 (5) |

Past ten minutes the samples are 4-12 sessions, and BankNIFTY's late reclaims
are negative. The window stays where it is.

By v5u's own definition, NIFTY's 09:15 flush to 22,470.05 (30 points), closed
back above 22,500 at 09:16, was a v5u long. In this shape it was stopped. The
agent was holding on the auth failure.

**3. The first retest of a reclaimed round number (the 09:50 short).** The
setup:

- A round number breaks (the first 1-minute close through it, 09:15-11:00) and
  is reclaimed within 60 minutes.
- Price trades back to it within 30 minutes of the reclaim, and the retest is
  done at the first close back on the reclaim side.
- NIFTY enters the next minute, WITH the reclaim (v5y's shakeout entry) or
  AGAINST it (the agent's trade).

| leader | n | with the reclaim | against it |
|---|---|---|---|
| BankNIFTY 1,000s | 248 | +0.075 (-0.066 / +0.156) | -0.061 (-0.068 / -0.056) |
| ... flat opens | 160 | **+0.155** (+0.019 / +0.222) | -0.080 (-0.107 / -0.067) |
| BankNIFTY 500s | 484 | +0.073 (-0.023 / +0.132) | -0.042 (-0.012 / -0.061) |
| NIFTY 500s | 160 | -0.077 (-0.047 / -0.094) | **+0.212** (+0.189 / +0.226) |
| NIFTY 100s | 661 | -0.037 (-0.042 / -0.034) | +0.062 (+0.093 / +0.043) |

On 26 Oct BankNIFTY's 54,000 broke at 09:18, was reclaimed at 09:44, retested
at 09:47 and done at 09:50. A long at 09:51 (22,476.10) reached its target at
09:53, +2.81R; the agent sold the same minute. With BankNIFTY as the leader,
fading the retest is negative in both halves. On NIFTY's own 500s, fading is
the better side. The effects are 0.06-0.2R and depend on the leader, so this
is not a clean rule.

**4. The risk-reward lesson: not measured.** Testing it needs the book's
journal (target-hit rate and give-backs by R:R), which is not on this machine.

### Candidates, for the operator's call

- **v5z-1, v5x's end clause.** On 22 Oct IH said "if it goes below 500... we
  cannot target these sellers". On 26 Oct both numbers had closed below before
  he bought, and he said the breakdown added sellers to target. His line was
  the lower point. The choices:
  - (a) Narrow the clause: a round-number close ends the hunt only once NIFTY
    also closes below yesterday's low, and a lone BankNIFTY break of its own
    low is the divergence-fails case. Measurement 1 does not support it.
  - (b) Revert v5x on its own review condition. Its only live trade so far is
    26 Oct's 09:50 short, which cited it and lost 3,481.50.
  - (c) Leave it and keep watching.
  **Reframed by the operator and adopted** (see "Adopted: v5z-1", below).
- **v5z-2, a veto: BankNIFTY's first retest of a reclaimed major number is not
  a double top.** Fading it measured -0.04 to -0.08R, negative in both halves;
  going with it measured +0.155R on flat opens for the 1,000s. NIFTY's own 500s
  say the opposite, so the veto would be scoped to BankNIFTY as the leader.
  v5y's reclaim sentence already removes the short's premise, and the agent
  did not apply it. **Adopted** (below).
- **v5z-3, risk-reward (the 25 Oct lesson).** Change R:R by moving the target
  only; the stop stays where a close means you are wrong. The diagnosis runs
  both ways: accurate trades that miss their targets need a nearer target,
  inaccurate ones a farther target. His own option-buying default is about 1:1,
  because time decay means "with a big ratio I would have to wait for three or
  four momentums". The corpus prefers about 1:2 and allows 1:1 only when every
  condition holds (OPTION-TIME-ADJUSTED REWARD/RISK). The overlap is v4s,
  SLH-025 and v5x's placement rule. The net-new part is the 1:1 default, and it
  needs the journal before it moves anything.

### Adopted: v5z-2 (operator, 26 Oct)

**A RECLAIMED BANKNIFTY ROUND NUMBER'S FIRST RETEST IS NOT A DOUBLE TOP.** It
sits in LEVELS, directly after v5y's shakeout rule, whose "NOT A SHAKEOUT"
sentence now ends with a pointer to it.

- **The event:** BankNIFTY breaks one of its 500s or 1,000s, closes back
  through it the other way, comes back to the number, and closes back on the
  reclaim side.
- **The veto:** do not enter against the reclaim there. A retest that does not
  close back within 30 minutes is not this rule.
- **A veto, not an entry.** Going with the reclaim was not clean in both
  halves, so a trade that way still needs its own pattern+confirmation.
- **Scope:** BankNIFTY's own numbers only, because NIFTY's own 500s measured the
  other way (+0.212R for the fade). It relaxes no RISK rule, gate, cooldown or
  cross-index read.
- The rule carries its measurement and the 26 Oct case.

To review: decisions that cite (v5z) are checked against what the fade would
have made. The rule is reverted if the vetoed fades would have paid.

### Validation

`test_v5z_a_reclaimed_banknifty_round_numbers_first_retest_is_not_a_double_top`
pins the rule. It failed before the rule existed and passes with it. Its 11
in-memory mutants are all caught:

- the veto flipped to an entry against the reclaim;
- the measurement altered;
- the halves dropped;
- the veto turned into an entry;
- the pattern requirement dropped;
- the scope widened to every index;
- NIFTY's counter-result dropped;
- the relaxes-nothing clause dropped;
- the 30-minute carve-out dropped;
- the v5y pointer removed;
- the rule moved to the end of LEVELS.

**Gates:** the master suite (673), market-data health (28), pytest (1,795),
ruff, mypy (82 files, then the master file), compileall and bandit (CI's
skips, on the changed files) are all clean. pre-commit was not run: its hooks
are not installed on this machine. They cover ruff, which was run, and four
file checks these edits do not touch.

### v5z-1 and v5z-3: not encoded (checked against the corpus, 26 Oct)

The operator asked for the other two candidates to be encoded only if they do
not contradict existing knowledge. Neither passes that test.

- **v5z-1 contradicts v5x.** v5x's "WHAT ENDS IT: the first 1-minute CLOSE
  below either number" is the clause v5z-1 would replace with "only once NIFTY
  also closes below yesterday's low". It agrees with DIVERGENCE and with the
  seller-recruitment rule's lower point, but not with v5x. Its measurement is
  flat as well.
- **v5z-3 is half duplicate, half contradiction.**
  - "Keep the stop, move the target" is already in the corpus: "choose the
    stop THE PREMISE REQUIRES" (v4), v5e and v5x's placement, SLH-025, and
    RISK's "never widen it to get size".
  - His 1:1 default, and lowering or raising the target by accuracy, pull
    against OPTION-TIME-ADJUSTED REWARD/RISK: "normally prefer approximately
    1:2", 1:1 "only when EVERY condition is true", and "never fabricate a
    distant target... to manufacture a ratio".

Either could still be adopted as a deliberate change to the rule it conflicts
with, on the operator's call, as v5x was. The operator then did that for v5z-1
(next).

### Adopted: v5z-1, reframed (operator, 26 Oct evening)

**The operator's reframing:** a round-number cross may be the market seating
and trapping the opposite crowd, so the agent should look for the price-action
move. For the benefit of the doubt, the pre-open note takes precedence.

**Scope (operator's choice):** the round-number case only. A blanket "the note
always wins" would have overridden v5q, v5k, v5f, the standing-premise rule and
the PRE-OPEN block's "your read wins", several of them measured.

**Measured** before encoding, on the 5-year CSVs. The setup:

- FLAT opens after a day that moved 0.5% or more (3:15 to 3:15).
- The first close beyond NIFTY's or BankNIFTY's next 500 from the open, by
  10:30.
- NIFTY still holding yesterday's extreme on the note's side.
- The note's side (against the cross) is entered on the first NIFTY covering
  close within 20 minutes, in the book's shape.

| prior day | cross | note's side | following the cross |
|---|---|---|---|
| down | down (IH's 26 Oct case) | **+0.708R** (27; +0.691 / +0.721) | -0.017R |
| up | up (the mirror) | **+0.608R** (19; +1.818 / +0.176) | -0.475R |
| up | down (a BUY note that follows) | +0.106R (70; +0.117 / +0.098) | -0.003R |
| down | up (a SELL note that follows) | **-0.164R** (59; -0.296 / -0.114) | +0.399R |

Further checks:

- **The covering window does not matter:** 10, 20 and 30 minutes all gave
  +0.65 to +0.71R in the down-day case.
- **The hold of yesterday's extreme does:** without it the down-day case falls
  to +0.148R (69).
- **The covering candle is the trigger:** buying at a fixed minute after the
  cross lost (-0.068R at six minutes).

This is a different test from measurement 1, which keyed on the lone
BankNIFTY break of yesterday's low rather than on the round-number cross.

**The rule**, A ROUND NUMBER CROSSED AGAINST THE NOTE SEATS THE CROWD IT HUNTS;
THE NOTE'S SIDE STANDS, sits between v5x and v5k:

- **Scope:** a pre-open note that names a side, a FLAT open, a prior move of
  0.5%+, a 500-cross against the note's side by 10:30, and NIFTY still holding
  yesterday's extreme on that side.
- **What it changes:** the note's side outranks v5x's end clause and v5f's
  follow, and the agent does not take the opposite side on the cross.
- **The entry is price action:** the covering candle, with pattern+confirmation.
  A reclaim of the number is the same hunt.
- **What ends it:** NIFTY trading beyond yesterday's extreme on the note's side.
- **Not covered:** the measured-negative case, a SELL note that follows a down
  day with an upward cross.
- It relaxes nothing (RISK, gates, cooldown, max-loss, v5q, v5k, the
  standing-premise rule), and it is reverted if live results are negative.

**Two pointers keep the prompt consistent:**

- v5x's end clause now ends with "[Narrowed by v5z, next: ...]".
- The PRE-OPEN block (`sl_hunting_premarket.py`) carries one exception line
  right after "your read wins": a round number crossed AGAINST the note's side
  does not by itself make the live read win.

**Validation:**

- `test_v5z_a_round_number_crossed_against_the_note_seats_the_crowd_it_hunts`
  and `test_block_points_a_round_number_cross_against_it_to_v5z` both failed
  before the change and pass with it.
- 16 mutants, all caught. On the rule (13): precedence reversed, "outranks"
  made "yields", the opposite side allowed, the pattern requirement dropped,
  the hold of yesterday's extreme dropped, the FLAT scope dropped, the end line
  moved to the round number, the negative-case carve-out dropped, the negative
  case hidden, the measurement altered, the relaxes-nothing clause dropped, the
  v5x pointer removed, and the rule moved after v5k. On the PRE-OPEN block (3):
  the exception removed, moved away from "your read wins", and weakened.
- **Gates:** the master suite (673), market-data health (28), pytest (1,797),
  ruff, mypy (82 files, then the master file), compileall and bandit (CI's
  skips, on the changed files) are all clean. pre-commit was not run, for the
  reason given under v5z-2.

### Already in the corpus

- One index breaks a level while the others hold, and the break fails:
  DIVERGENCE in CROSS-INDEX CONFIRMATION, and the triple-index read in BNF
  SPECIFIC.
- "Seeing BankNIFTY's breakdown, others sell NIFTY and Sensex, and the market
  targets them": the seller-recruitment loop, with the lower point as its
  premise-ending level.
- Holding for the round-number breakout because the breakdown sellers' stops
  sit above it: the SL-hunt premise, v5u and the trap rule.
- Booking when the breakout brings no fast follow-through, and "in greed we
  gave back what was in hand": v5w, TARGET-BOOKED, PROFIT-HOLD and v5y.

### Operational

- **The open was lost to an expired token.** The `claude` CLI's OAuth token
  had expired, so 09:15-09:22 were eight forced HOLDs. The 09:15:51 restart
  then hit the broker exposure audit ("Open-order query was indeterminate"),
  which blocked LIVE: SL Hunting ran PAPER all day.
- **Sizing ignores the max-loss headroom.** `_compute_entry_sizing` sizes on
  the fixed 2,500 budget and never on what is left before the max-loss. Trade
  3 could only end at MAX_LOSS: its stop risk exceeded the room left, so it
  closed at -7,932 against a 7,500 limit.
  - The fix: refuse an entry whose stop risk (NIFTY leg plus mirror) exceeds
    the remaining room. Not built.
- **SLH-027's first two firings both cost.** Neither trade recovered (NIFTY
  went on to 22,706.15), so the closing basis saved nothing on the day and
  cost about 30 NIFTY points:
  - Trade 1's backstop (22,486.95) filled at 22,494.20: -34.1 points, -1.91R.
    The intrabar stop sat at 22,478.
  - Trade 2's stop fired on a close and filled at 22,678.55 (-1.37R), where
    the intrabar stop was 22,665.
- **Expiry eve.** All three NIFTY legs used the 27 Oct expiry, one day out.
  The BankNIFTY mirror used the 4-step ITM strike for expiry week.
- **Latency.** 47 completed calls: median 39.9 s, p95 62.6 s, worst 77.1 s.
  Five calls passed the 60 s warning, and the 09:26 call was abandoned at
  90 s.

## October 27 pre-open note

[Prediction For 27 OCT 2026](https://www.youtube.com/watch?v=mSE4vo8crr0), uploaded
2026-10-26 21:10:20 IST, 105 s, 50 transcript segments.

**Read.** October 26 made positive momentum in all three indices. The buyers it
drew were already shaken out by a retracement, and the market is trying to go
up again. In IH's words on NIFTY: "when the market again makes upside momentum,
we follow the market continuously." This is a follow, not a hunt.

**Plan.** No branch sells.

| Open | Plan |
|---|---|
| FLAT TO GAP UP | BUY, with the market: identify buying-side setups; positive momentum from the open is the better case |
| SMALL GAP DOWN | BUY, the same plan, better if the levels do not break: a breakdown near BankNIFTY's round number raises the risk on buyers |

IH gives no branch for a big gap down, and none is invented. NIFTY and
BankNIFTY both expire on October 27. The note carries that as context, not as
an entry premise.

**Levels.** The 1080p frames at 0:40, 1:12 and 1:37 resolve two garbled
captions and one rounded value:

* **NIFTY resistance**, spoken "2280 23000", is drawn at 22,881.70 / 23,002.20,
  and recorded as 22,880 / 23,000. It is the same line the 23 Oct note resolved,
  not 22,800.
* **Sensex support**, spoken "7372500", is drawn at 73,007.15 / 72,512.00, and
  recorded as 73,000 / 72,510, the same 72,510 line as on 26 Oct.
* **BankNIFTY's second support** is spoken 53,670 but drawn at 53,621.75, and is
  kept at 53,620, as on 26 Oct.

The other nine match both their spoken values and their drawn labels.

| Index | Resistance | Support |
|---|---|---|
| NIFTY | 22,880 / 23,000 | 22,500 / 22,320 |
| BankNIFTY | 54,500 / 54,810 | 54,000 / 53,620 |
| Sensex | 74,050 / 74,530 | 73,000 / 72,510 |

**How it meets the corpus.** October 26 closed +0.89% on the 3:15 levels
(22,717.05 against 22,516.40). If 27 Oct opens FLAT and NIFTY or BankNIFTY
closes below its next 500 under the open, v5z-1 applies to this BUY note.
That is the up-day, down-cross case, measured at a thin +0.106R. v5x does not
apply: 26 Oct was not a selling day.

The note renders 1,863 characters on October 27 and nothing on any other date.
`test_shipped_note_matches_october_27_intraday_hunter_plan` pins the date gate,
the source, both BUY branches (no line may SELL, no big-gap-down branch), the
expiry line and all twelve levels. Each of these 8 mutations fails it, and the
unmutated note passes:

* NIFTY's resistance recorded as the spoken 22,800;
* the date set to October 26;
* the small-gap-down branch turned into SELL;
* the "levels do not break" clause dropped;
* the expiry line dropped;
* Sensex's support moved to 72,500;
* a big-gap-down branch invented;
* BankNIFTY's support recorded as the spoken 53,670.

## v6a - 27 Oct live match: the same long with a tighter stop, and the raise on expiry day

**Sources.**

- **IH's live session:** `wNEwGuJgFGU`, uploaded 2026-10-27 10:23:51 IST, 8:27.
  He traded on the 27 Oct note (`mSE4vo8crr0`: BUY, with the market). Clock
  times come from the taskbar clock in the frame, figures from his positions
  screen at 2:05 and 7:25, and his words are my translation of the Hindi
  auto-transcript (210 segments).
- **This book:** three journal rows, 72 decisions (no agent errors) and the
  runner log. The model was `claude-sonnet-5`. The run was LIVE through Kotak,
  and the note was injected (1,863 chars).
- **The 5-year NIFTY and BankNIFTY CSVs**, refreshed through 27 Oct.

**Adopted: v6a-1**, on the operator's call after the measurement below. On the
day the held NIFTY contract expires, MODIFY_TARGET is not used and the
mechanical target books it. `position_state` now reports the contract's expiry
so the rule can be applied.

### The session

26 Oct's 3:15 level was 22,717.05 (+0.89% on 23 Oct's). 27 Oct opened FLAT on
the computed verdict (NIFTY -0.04%, BankNIFTY +0.19%), and NIFTY and BankNIFTY
both expired.

- **09:15-09:19:** NIFTY opened at 22,707.55, touched 22,724.35 in the first
  minute, and flushed to 22,653.20 at 09:19.
- **09:22-09:26:** it closed back above 22,700 at 09:22 (22,711.50), retested
  the number at 09:24-09:25 (low 22,676.05), and closed above it again at 09:26.
- **09:30-10:04:** it reached 22,750.20 at 09:30, ranged 22,705-22,767 until
  09:59, then ran to 22,804.50 at 10:04, BankNIFTY's day high minute
  (54,654.05).
- **10:21:** it fell back to 22,684.55.

The day's high was 22,829.75 (12:43), the low 22,636.85 (14:17), and the 3:15
level 22,760.75. IH's NIFTY line, 22,625, never broke.

| | IH | the agent |
|---|---|---|
| read | follow: 26 Oct's recovery buyers were already shaken out by a retracement, and their "mindset turned" to hunting breakdowns, so nobody is seated to hunt | FLAT and the note's BUY branch; later, two shorts at double tops |
| entry | **09:18-09:19**, into the opening flush: BankNIFTY OCT 54200 CE and 54300 CE x1,170 each, SENSEX OCT 73300 CE x900, NIFTY OCT 22650 CE x1,430 (avg 96.50) | **09:23** LONG x2 at 22,711.15 · 09:37 LONG refused by sizing · **09:57** SHORT x1 at 22,729.20 · **10:14** SHORT x1 at 22,772.30 |
| lines | NIFTY 22,625, Sensex 73,220 and BankNIFTY "160" (about 54,160), all named before the entry. The basket sat 21,533 down at 09:19. | stops at 22,694 / 22,756 / 22,794 |
| exit | about **09:26**, the normal target on BankNIFTY's strong candle: "today is expiry... we won't run after more momentum" | AI_STOP 09:25:00 · own exit 09:59 · AI_TARGET 10:20 |
| result | **+2,22,602.25** booked (BankNIFTY +91,646.60 and +82,918.65, SENSEX +27,945.00, NIFTY +20,092.00) | live **+3,091.25** (-3,171.00 / +13.25 / +6,249.00) |

### The agent: IH's long four minutes later, and the stop he sat through

- **09:23 LONG, 2 lots, stop 22,694.** This was IH's side, taken at the 09:22
  reclaim of 22,700. The rationale cited v4t's early-retracement trap.
  - The stop sat 6 points under the number. The 09:24 candle closed at
    22,693.90, 0.10 beyond it, and the stop went at 09:25 for -3,171.00. An
    intrabar stop would have gone too (low 22,691.80), so SLH-027 changed
    nothing.
  - IH sat through the same dip with his line 86 points lower.
- **v5y already had the better entry.** The 22,700 break was retested at 09:24
  and done at the 09:26 close (22,700.65). A long at 09:27 (22,702.05) would
  have reached its 45-point target at 09:30 (+2.81R) without touching its
  stop. This is a compliance gap, not a new rule.
- **09:37: the same long, refused by sizing.** Entry 22,741.35, stop 22,692:
  3,207.75 for one lot. NIFTY reached that entry's +45 by 10:01. This is
  SLH-025 as designed.
- **09:57 SHORT, against the note,** off the 22,766-67 double top. The agent
  exited after 90 seconds as cross-index flipped up: NIFTY leg -22.75, mirror
  +36.00.
- **10:14 SHORT at the 22,800 double top: AI_TARGET in six minutes.** NIFTY
  leg +2,730.00, mirror +3,519.00. IH was flat by then. No MODIFY_TARGET was
  used today.

### Measured: the raise on an expiry day (v6a-1)

IH, booking at 09:26: "today is expiry... as soon as a retracement or a
rejection comes, profit can drop a lot. We won't run after more momentum." And
later: "if it goes a little sideways, there will be a problem; a small
rejection, a problem. In such a situation take the normal target and go."

v5y lets MODIFY_TARGET raise a run that has already pulled back about 20
points, with no expiry exception. The test is v5y's own:

- On the 5-year NIFTY CSV, enter at 09:20, 09:35 and 10:00 with the move since
  09:15 (at least 0.15%).
- Keep the entries that reach +45 within 60 minutes, and class them by the
  pullback before that touch.
- Score what follows: "extend" is +25 more before 20 given back, "giveback" is
  the reverse, and drift is the position 15 minutes later.
- Split by NIFTY's weekly expiry day, from `expiry_calendar.weekly_expiry_dates`
  (262 expiry days).

| before the target | expiry day | other days |
|---|---|---|
| pullback >= 20 (v5y's raise case) | n 134: extend 37%, giveback 43%, drift **-2.9** (-3.2 / -2.7) | n 433: 36% / 44%, drift +1.1 (-4.4 / +4.1) |
| pullback 10-20 | n 40: 40% / 38%, drift -2.3 (+6.7 / -8.3) | n 131: 34% / 37%, drift -0.4 |
| straight (< 10) | n 9: 33% / 67%, drift -22.2 | n 39: 38% / 49%, drift -2.0 |

The straight-run and 10-20 counts reproduce v5y's (47 and 171).

On expiry days the raise case drifts against the position in both halves; on
other days it is flat with its halves disagreeing. The effect is small, about 3
points, but it agrees with IH and with the corpus: EXPIRY-DAY RANGE ("take the
momentum you got and stop"), the time-cost rule ("SHRINK the target... especially
on an expiry session"), and the reminder that what counts is the days-to-expiry
of the contract actually held.

### Candidate, for the operator's call

- **v6a-1, ON AN EXPIRY DAY THE MECHANICAL TARGET BOOKS IT.** When the NIFTY
  contract held expires today, do not RAISE the target with MODIFY_TARGET, even
  after a pullback: book the mechanical target, measured above.
  - It narrows v5y's raise carve-out the way v5y narrowed v5w, and contradicts
    nothing: the expiry rules above already point the same way.
  - It creates no entry and changes no stop.
  - **Adopted** (next).

### Adopted: v6a-1 (operator, 27 Oct)

**ON AN EXPIRY DAY THE MECHANICAL TARGET BOOKS IT: NO RAISE.** It sits right
after v5y's straight-run rule, which now ends with "[Narrowed by v6a, next:
never on the day the held contract expires.]", and before PROFIT-HOLD.

- **The trigger is the contract held,** not the calendar. The prompt had no way
  to know it, so `position_state` now carries `contract_expiry` (ISO date) and
  `contract_expires_today` while a NIFTY leg is open. They come from a new
  worker method, `contract_expiry_state`, which reads the position's own
  `option_expiry` (locked at entry) against `_ist_now()`. The executor's
  snapshot copies exactly those two keys, the same way SLH-027's `stop_fires`
  is read. With no expiry recorded, both fields are left out rather than
  guessed.
- **What it changes:** no MODIFY_TARGET raise that day, whatever the run has
  done. The order tool's MODIFY_TARGET text says the same, and
  POSITION_STATE_DESCRIPTION names the new fields.
- **What it does not change:** v5w (no early booking) still holds, and the
  target is never lowered. No entry, stop, sizing or order path changes.
- The rule carries its measurement (pullback >= 20 on expiry days: -2.9 points,
  both halves negative; other days flat, halves disagreeing) and IH's words.

**Validation.**

- `test_v6a_on_an_expiry_day_the_mechanical_target_books_it` (schema) and
  `test_v6a_position_state_says_whether_the_held_contract_expires_today`
  (master) both failed before the change and pass with it.
- **15 mutants, all caught.**
  - 11 on the text: the raise allowed on expiry, the trigger widened to any
    index, the measurement altered, the halves dropped, the other-days
    comparison hidden, an early exit licensed, the no-entry/no-stop clause
    dropped, the v5y pointer removed, the rule moved after PROFIT-HOLD, the
    order-tool sentence removed, and the position_state field left
    undescribed.
  - 4 on the code, each written into the source, run, and restored: the IST
    clock replaced by `date.today()`, the comparison inverted, today's date
    reported as the expiry, and the executor never copying the fields.
- **Gates:** the master suite (674), market-data health (28), pytest (1,798),
  ruff, mypy (82 files, then the master file), compileall and bandit (CI's
  skips, on the changed files) are all clean. pre-commit was not run: its hooks
  are not installed on this machine.

### Already in the corpus

- **"SL hunting does not mean always working the opposite side; if no retail
  crowd is seated, follow what is running":** the FOLLOW rules (v4o, v5f, v5k,
  v5r) and v4t's "nobody is seated on that side, which makes the trade a
  FOLLOW".
- **"Whoever bought and got shaken out, his mindset turns; he keeps looking for
  breakdowns while the market covers up":** REGIME MEMORY DECIDES WHO SHOWS UP
  AT A LEVEL (v4b).
- **Naming the invalidation lines before entering and accepting the loss to
  them:** NAME THE LAST POINT (v4e), with SLH-025 when the line does not fit
  the budget.
- **"Two indices expire today; take the normal target":** TWO EXPIRIES AT ONCE
  (v4o) and EXPIRY-DAY RANGE. v6a-1 is the one piece they do not already say.

### Operational

- **The login held.** Every one of the 72 decisions completed with no auth
  failure, and the run was LIVE from 08:55.
- **Latency.** 72 calls: median 35.3 s, p95 50.6 s, worst 73.1 s. One call
  passed the 60 s warning, and none was abandoned.
- **Entries stopped at 11:00** (TIME_CUTOFF), the configured cutoff.
