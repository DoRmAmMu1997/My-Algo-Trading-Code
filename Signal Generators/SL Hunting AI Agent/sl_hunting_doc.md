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
