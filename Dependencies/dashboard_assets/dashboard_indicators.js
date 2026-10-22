/* Chart-only indicators, computed in the BROWSER.
 *
 * Why here and not on the server, where VWAP and the stochastic are computed:
 * the history pages are rendered ONCE at startup into immutable bytes -- 233
 * pages for the one-minute series on the operator's five-year file -- and
 * handed out as finished bytes forever after. There is no per-request render,
 * so a server-side indicator could only ever cover the live window and would
 * vanish the moment the operator scrolled back. Computing over whatever bars
 * the browser holds is what makes an indicator work across five years of
 * scroll-back, and it costs the live-money process nothing at all.
 *
 * The price of that is a SECOND implementation of algebra this repository
 * already has in `Signal Generators/misc_strategy_common.py`. It is bounded
 * two ways. First, these values are labelled CHART-ONLY, the same honesty
 * convention ADR-0017 established for CPR: they are for looking at, and no
 * strategy reads them. Second, and more usefully,
 * `Tests/Dependencies/test_dashboard_chart_indicators_js.py` runs every
 * `compute` below under node against a fixture and asserts agreement with the
 * strategies' own TA-Lib helpers element by element, leading NaN run included.
 *
 * So these functions must reproduce TA-LIB's semantics, not a textbook's. The
 * differences are not rounding: TA-Lib seeds an EMA with the SMA of its first
 * `period` values rather than the first value, uses Wilder smoothing for RSI,
 * ATR and ADX, and populates its outputs only from the index where each is
 * fully determined. A naive implementation is wrong from the first bar and
 * stays wrong. When the drift test fails, the fix is to match TA-Lib -- never
 * to loosen the tolerance.
 *
 * The one exception is Supertrend, which is not a TA-Lib function: it is
 * `misc_strategy_common.supertrend`'s own band carry-forward, ported here rule
 * for rule, over a TA-Lib ATR.
 *
 * Everything in this file is PURE: numbers in, numbers out, no DOM and no
 * chart. `dashboard.js` owns the series, the panes and the menu.
 */
"use strict";

var DashboardIndicators = (function () {
  /* A run of `count` nulls, then nothing. Every series below is right-aligned
   * to the bars it was computed from, so a warm-up is leading nulls rather
   * than a shorter array -- the same contract the server's indicator columns
   * use, so `pointsFrom` treats both identically. */
  function filled(count) {
    return new Array(Math.max(0, count)).fill(null);
  }

  function closes(bars) {
    return bars.map((bar) => bar.close);
  }

  /* ------------------------------------------------------------- averages */

  /* TA-Lib SMA: the first output sits at index `period - 1`. */
  function sma(values, period) {
    const out = filled(values.length);
    if (period < 1 || values.length < period) return out;
    let total = 0;
    for (let i = 0; i < values.length; i += 1) {
      total += values[i];
      if (i >= period) total -= values[i - period];
      if (i >= period - 1) out[i] = total / period;
    }
    return out;
  }

  /* TA-Lib EMA, CLASSIC compatibility: the first output sits at index
   * `period - 1` and is the SMA of the first `period` values, NOT the first
   * value. Seeding with values[0] instead shifts the whole line and takes
   * hundreds of bars to converge -- the single most common way a hand-rolled
   * EMA disagrees with the one a strategy trades on.
   *
   * `firstIndex` moves where the seed window ENDS, and so where the output
   * begins. It exists for MACD, which needs both of its EMAs to start at the
   * same bar; everything else takes the default and gets the plain TA-Lib EMA. */
  function ema(values, period, firstIndex) {
    const out = filled(values.length);
    const first = firstIndex === undefined ? period - 1 : firstIndex;
    if (period < 1 || first < period - 1 || values.length <= first) return out;
    let seed = 0;
    for (let i = first - period + 1; i <= first; i += 1) seed += values[i];
    let previous = seed / period;
    out[first] = previous;
    const k = 2 / (period + 1);
    for (let i = first + 1; i < values.length; i += 1) {
      previous = (values[i] - previous) * k + previous;
      out[i] = previous;
    }
    return out;
  }

  /* --------------------------------------------------------------- ranges */

  /* True range, indexed with the bars. TR[0] is undefined without a previous
   * close, so it is null and every Wilder average below starts from index 1. */
  function trueRange(bars) {
    const out = filled(bars.length);
    for (let i = 1; i < bars.length; i += 1) {
      const previousClose = bars[i - 1].close;
      out[i] = Math.max(
        bars[i].high - bars[i].low,
        Math.abs(bars[i].high - previousClose),
        Math.abs(bars[i].low - previousClose),
      );
    }
    return out;
  }

  /* Wilder's smoothing over a series whose first real value is at `from`:
   * seed with the SIMPLE mean of `period` values, then
   * `next = (previous * (period - 1) + current) / period`.
   *
   * This is a slower average than an EMA of the same length -- Wilder's
   * effective k is 1/period, not 2/(period+1) -- which is why RSI and ATR are
   * visibly different lines if an EMA is substituted. */
  function wilder(values, period, from) {
    const out = filled(values.length);
    const first = from + period - 1;
    if (period < 1 || values.length <= first) return out;
    let total = 0;
    for (let i = from; i <= first; i += 1) total += values[i];
    let previous = total / period;
    out[first] = previous;
    for (let i = first + 1; i < values.length; i += 1) {
      previous = (previous * (period - 1) + values[i]) / period;
      out[i] = previous;
    }
    return out;
  }

  /* TA-Lib ATR: Wilder over the true range, first output at index `period`. */
  function atr(bars, period) {
    return wilder(trueRange(bars), period, 1);
  }

  /* ----------------------------------------------------------- oscillators */

  /* TA-Lib RSI: Wilder averages of gains and losses, first output at index
   * `period`. An all-up window pins it at 100 and an all-down one at 0, which
   * is the behaviour, not a defect. */
  function rsi(values, period) {
    const out = filled(values.length);
    if (period < 1 || values.length <= period) return out;
    let gain = 0;
    let loss = 0;
    for (let i = 1; i <= period; i += 1) {
      const change = values[i] - values[i - 1];
      if (change > 0) gain += change;
      else loss -= change;
    }
    let averageGain = gain / period;
    let averageLoss = loss / period;
    out[period] = averageLoss === 0 ? 100 : 100 - 100 / (1 + averageGain / averageLoss);
    for (let i = period + 1; i < values.length; i += 1) {
      const change = values[i] - values[i - 1];
      averageGain = (averageGain * (period - 1) + (change > 0 ? change : 0)) / period;
      averageLoss = (averageLoss * (period - 1) + (change < 0 ? -change : 0)) / period;
      out[i] = averageLoss === 0 ? 100 : 100 - 100 / (1 + averageGain / averageLoss);
    }
    return out;
  }

  /* TA-Lib MACD.
   *
   * The arithmetic is the obvious one -- fast EMA minus slow EMA, then an EMA
   * of that -- but the OUTPUT WINDOW is not. TA-Lib emits every output of a
   * multi-output function from the same index, and here that is
   * `slow - 1 + signal - 1`: the macd line exists from `slow - 1`, yet those
   * first `signal - 1` values are consumed seeding the signal line and are
   * never published. Emitting them anyway draws a stub of macd line with no
   * signal beside it, at values no strategy reading TA-Lib would ever see.
   *
   * The second subtlety is the FAST EMA. TA-Lib starts both EMAs at the same
   * bar, `slow - 1`, which means the fast one is seeded on the `fast` values
   * ENDING there rather than on the first `fast` values of the series. Seeding
   * it the ordinary way leaves a difference that decays but never vanishes --
   * about 1.6 index points at the first published bar of a 400-bar fixture. */
  function macd(values, fastPeriod, slowPeriod, signalPeriod) {
    const alignAt = slowPeriod - 1;
    const fast = ema(values, fastPeriod, alignAt);
    const slow = ema(values, slowPeriod, alignAt);
    const raw = filled(values.length);
    for (let i = 0; i < values.length; i += 1) {
      if (fast[i] !== null && slow[i] !== null) raw[i] = fast[i] - slow[i];
    }

    const start = slowPeriod - 1;
    const line = filled(values.length);
    const signal = filled(values.length);
    const histogram = filled(values.length);
    if (start < values.length) {
      /* An EMA over the macd line's valid TAIL, written back at the offset it
       * came from -- rather than an EMA of an array padded with nulls, which
       * would seed on the padding. */
      const tail = raw.slice(start).map((value) => (value === null ? 0 : value));
      const smoothed = ema(tail, signalPeriod);
      for (let i = 0; i < smoothed.length; i += 1) {
        if (smoothed[i] === null) continue;
        line[start + i] = raw[start + i];
        signal[start + i] = smoothed[i];
        histogram[start + i] = raw[start + i] - smoothed[i];
      }
    }
    return { line, signal, histogram };
  }

  /* TA-Lib ADX. Trend STRENGTH, not direction, and the longest warm-up of
   * anything here: the first output sits at index `2 * period - 1`.
   *
   * It does NOT reuse `wilder` above, and that is the whole subtlety. TA-Lib's
   * ATR seeds on the MEAN of the first `period` true ranges; its ADX
   * accumulates a running SUM over the first `period - 1` deltas and then
   * starts smoothing, so its seed is
   * `sum(1..period-1) * (period-1)/period + value[period]` rather than
   * `mean(1..period)`. The two differ by `sum/period^2`, which is small, never
   * zero, and never decays -- it was worth about 0.09 of an ADX point on a
   * 400-bar fixture. Sharing `wilder` here looks like the tidier code and is
   * quietly a different indicator from the one the Regime Adaptive router
   * reads its ADX from. */
  function adx(bars, period) {
    const out = filled(bars.length);
    if (period < 1 || bars.length <= 2 * period) return out;

    const ranges = trueRange(bars);
    const plusMoves = filled(bars.length);
    const minusMoves = filled(bars.length);
    for (let i = 1; i < bars.length; i += 1) {
      const up = bars[i].high - bars[i - 1].high;
      const down = bars[i - 1].low - bars[i].low;
      plusMoves[i] = up > down && up > 0 ? up : 0;
      minusMoves[i] = down > up && down > 0 ? down : 0;
    }

    /* Running SUMS, not averages, carried in this order throughout. */
    let sumRange = 0;
    let sumPlus = 0;
    let sumMinus = 0;
    for (let i = 1; i < period; i += 1) {
      sumRange += ranges[i];
      sumPlus += plusMoves[i];
      sumMinus += minusMoves[i];
    }

    function step(i) {
      sumRange = sumRange - sumRange / period + ranges[i];
      sumPlus = sumPlus - sumPlus / period + plusMoves[i];
      sumMinus = sumMinus - sumMinus / period + minusMoves[i];
      if (sumRange === 0) return 0;
      const plus = (100 * sumPlus) / sumRange;
      const minus = (100 * sumMinus) / sumRange;
      const span = plus + minus;
      return span === 0 ? 0 : (100 * Math.abs(plus - minus)) / span;
    }

    /* The seed is a SIMPLE mean of the first `period` DX values... */
    let total = 0;
    for (let i = period; i <= 2 * period - 1; i += 1) total += step(i);
    let previous = total / period;
    out[2 * period - 1] = previous;

    /* ...and everything after it is Wilder. */
    for (let i = 2 * period; i < bars.length; i += 1) {
      previous = (previous * (period - 1) + step(i)) / period;
      out[i] = previous;
    }
    return out;
  }

  /* `misc_strategy_common.supertrend`, rule for rule. Not a TA-Lib function:
   * the bands carry forward, and the direction only flips when a close breaks
   * the opposite band. `direction` stays null through the ATR warm-up. */
  function supertrend(bars, atrPeriod, multiplier) {
    const n = bars.length;
    const atrValues = atr(bars, atrPeriod);
    const line = filled(n);
    const direction = filled(n);
    const finalUpper = filled(n);
    const finalLower = filled(n);

    for (let i = 0; i < n; i += 1) {
      if (atrValues[i] === null) continue;
      const mid = (bars[i].high + bars[i].low) / 2;
      const basicUpper = mid + multiplier * atrValues[i];
      const basicLower = mid - multiplier * atrValues[i];

      if (i === 0 || finalUpper[i - 1] === null) {
        finalUpper[i] = basicUpper;
        finalLower[i] = basicLower;
        direction[i] = bars[i].close >= mid ? 1 : -1;
        line[i] = direction[i] > 0 ? finalLower[i] : finalUpper[i];
        continue;
      }

      finalUpper[i] =
        basicUpper < finalUpper[i - 1] || bars[i - 1].close > finalUpper[i - 1]
          ? basicUpper
          : finalUpper[i - 1];
      finalLower[i] =
        basicLower > finalLower[i - 1] || bars[i - 1].close < finalLower[i - 1]
          ? basicLower
          : finalLower[i - 1];

      direction[i] =
        direction[i - 1] >= 0
          ? bars[i].close < finalLower[i] ? -1 : 1
          : bars[i].close > finalUpper[i] ? 1 : -1;
      line[i] = direction[i] > 0 ? finalLower[i] : finalUpper[i];
    }
    return { line, direction };
  }

  /* ------------------------------------------------------------- registry */

  /* `kind` decides where it draws: "overlay" shares the candles' pane, "pane"
   * gets one of its own. `params` drive the menu's number inputs and are
   * clamped there, so `compute` can trust them. `compute` returns one entry
   * per LINE, each right-aligned to `bars`. */
  const INDICATORS = [
    {
      id: "ema",
      label: "EMA",
      kind: "overlay",
      params: [{ name: "period", label: "Period", value: 21, min: 2, max: 400 }],
      compute(bars, p) {
        return [
          { title: `EMA ${p.period}`, color: "#e0803a", values: ema(closes(bars), p.period) },
        ];
      },
    },
    {
      id: "sma",
      label: "SMA",
      kind: "overlay",
      params: [{ name: "period", label: "Period", value: 50, min: 2, max: 400 }],
      compute(bars, p) {
        return [
          { title: `SMA ${p.period}`, color: "#6f9ad3", values: sma(closes(bars), p.period) },
        ];
      },
    },
    {
      id: "bbands",
      label: "Bollinger Bands",
      kind: "overlay",
      params: [
        { name: "period", label: "Period", value: 20, min: 2, max: 400 },
        { name: "stddev", label: "Std dev", value: 2, min: 0.1, max: 10, step: 0.1 },
      ],
      compute(bars, p) {
        /* TA-Lib BBANDS with matype 0: an SMA middle and a POPULATION standard
         * deviation (divide by n, not n-1) over the same window. The sample
         * form would widen every band by a factor of sqrt(n/(n-1)). */
        const values = closes(bars);
        const middle = sma(values, p.period);
        const upper = filled(values.length);
        const lower = filled(values.length);
        for (let i = p.period - 1; i < values.length; i += 1) {
          if (middle[i] === null) continue;
          let squares = 0;
          for (let j = i - p.period + 1; j <= i; j += 1) {
            squares += (values[j] - middle[i]) ** 2;
          }
          const deviation = Math.sqrt(squares / p.period) * p.stddev;
          upper[i] = middle[i] + deviation;
          lower[i] = middle[i] - deviation;
        }
        return [
          { title: `BB upper ${p.period}`, color: "#8b94a3", values: upper },
          { title: `BB mid ${p.period}`, color: "#5f6672", values: middle },
          { title: `BB lower ${p.period}`, color: "#8b94a3", values: lower },
        ];
      },
    },
    {
      id: "supertrend",
      label: "Supertrend",
      kind: "overlay",
      params: [
        { name: "period", label: "ATR period", value: 10, min: 2, max: 200 },
        { name: "multiplier", label: "Multiplier", value: 3, min: 0.1, max: 20, step: 0.1 },
      ],
      compute(bars, p) {
        const result = supertrend(bars, p.period, p.multiplier);
        return [
          {
            title: `Supertrend ${p.period}/${p.multiplier}`,
            color: "#9b7fd4",
            values: result.line,
          },
        ];
      },
    },
    {
      id: "rsi",
      label: "RSI",
      kind: "pane",
      params: [{ name: "period", label: "Period", value: 14, min: 2, max: 200 }],
      guides: [70, 30],
      compute(bars, p) {
        return [
          { title: `RSI ${p.period}`, color: "#5b9dd9", values: rsi(closes(bars), p.period) },
        ];
      },
    },
    {
      id: "macd",
      label: "MACD",
      kind: "pane",
      params: [
        { name: "fast", label: "Fast", value: 12, min: 2, max: 200 },
        { name: "slow", label: "Slow", value: 26, min: 2, max: 400 },
        { name: "signal", label: "Signal", value: 9, min: 1, max: 200 },
      ],
      guides: [0],
      compute(bars, p) {
        const result = macd(closes(bars), p.fast, p.slow, p.signal);
        return [
          { title: `MACD ${p.fast}/${p.slow}`, color: "#5b9dd9", values: result.line },
          { title: `Signal ${p.signal}`, color: "#e8b13a", values: result.signal },
        ];
      },
    },
    {
      id: "atr",
      label: "ATR",
      kind: "pane",
      params: [{ name: "period", label: "Period", value: 14, min: 2, max: 200 }],
      compute(bars, p) {
        return [{ title: `ATR ${p.period}`, color: "#c98b5e", values: atr(bars, p.period) }];
      },
    },
    {
      id: "adx",
      label: "ADX",
      kind: "pane",
      params: [{ name: "period", label: "Period", value: 14, min: 2, max: 200 }],
      guides: [25],
      compute(bars, p) {
        return [{ title: `ADX ${p.period}`, color: "#3fc9d9", values: adx(bars, p.period) }];
      },
    },
  ];

  function byId(id) {
    return INDICATORS.find((entry) => entry.id === id) || null;
  }

  /* The menu writes whatever the operator typed; this is what makes it safe to
   * hand to `compute`. A missing or unparseable value falls back to the
   * parameter's default rather than producing NaN down the whole series. */
  function normalizeParams(entry, raw) {
    const params = {};
    for (const spec of entry.params) {
      let value = Number((raw || {})[spec.name]);
      if (!Number.isFinite(value)) value = spec.value;
      value = Math.min(Math.max(value, spec.min), spec.max);
      /* Periods are bar COUNTS: a fractional one would silently floor inside
       * the loops and disagree with the label beside the line. */
      if (!spec.step) value = Math.round(value);
      params[spec.name] = value;
    }
    return params;
  }

  return {
    INDICATORS,
    byId,
    normalizeParams,
    /* Exported for the drift test, which lifts them out of this file and runs
     * them against the strategies' own TA-Lib helpers. */
    sma,
    ema,
    wilder,
    trueRange,
    atr,
    rsi,
    macd,
    adx,
    supertrend,
  };
})();

/* The page loads this as a classic script before `dashboard.js`; node needs
 * the export to lift it for the drift test. */
if (typeof module !== "undefined" && module.exports) module.exports = DashboardIndicators;
