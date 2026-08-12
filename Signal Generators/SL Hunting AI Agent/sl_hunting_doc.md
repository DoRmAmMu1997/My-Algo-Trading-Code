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

Test updated: `test_shipped_note_matches_august_13_intraday_hunter_plan`
replaces the 12 Aug equivalent and asserts both branch directions plus the
round-number stop-location reasoning, because an inverted plan and a smoothed-
away justification are the two failures a copy-forward would produce.
