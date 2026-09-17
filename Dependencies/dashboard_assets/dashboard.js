/* Read-only monitoring dashboard: poll /api/state, render, repeat.
 *
 * Three rules mirror the Python side:
 *
 * 1. `null` is rendered as an em dash, never as 0. The server sends null
 *    whenever a number could not be derived honestly, and turning that into a
 *    zero would understate an open loss.
 * 2. Nothing here is ever sent back. The page issues GET requests and no
 *    other verb; the server answers 405 to anything else anyway.
 * 3. `setTimeout` chaining, never `setInterval`: on a slow machine a stacked
 *    queue of in-flight polls is the last thing this process needs.
 */
"use strict";

(function () {
  const REFRESH_FALLBACK_SECONDS = 1.0;
  const HIDDEN_TAB_SECONDS = 5.0;
  const RETRY_SECONDS = 5.0;
  /* How long the document may go unchanged before the page says so. Measured
   * in TIME, not in polls: a poll count silently assumes this tab polls no
   * faster than the runner publishes, and a count of consecutive 304s says
   * nothing about how long they took. Five refresh intervals of real silence
   * is a stopped builder; anything shorter is jitter. */
  const STALE_AFTER_INTERVALS = 5;
  const STALE_FLOOR_SECONDS = 5.0;

  const el = (id) => document.getElementById(id);
  const nodes = {
    dot: el("liveness-dot"),
    banner: el("banner"),
    sessionDate: el("session-date"),
    modeBadge: el("mode-badge"),
    phaseBadge: el("phase-badge"),
    feedSource: el("feed-source"),
    feedSpot: el("feed-spot"),
    feedBar: el("feed-bar"),
    staleChip: el("stale-chip"),
    totalRealized: el("total-realized"),
    totalOpen: el("total-open"),
    totalTotal: el("total-total"),
    generatedAt: el("generated-at"),
    openBody: document.querySelector("#open-table tbody"),
    openEmpty: el("open-empty"),
    openCount: el("open-count"),
    strategyBody: document.querySelector("#strategy-table tbody"),
    strategyEmpty: el("strategy-empty"),
    closedGroups: el("closed-groups"),
    closedEmpty: el("closed-empty"),
    closedCount: el("closed-count"),
    noticesPane: el("notices-pane"),
    notices: el("notices"),
    sessionStateNote: el("session-state-note"),
    chart: el("chart"),
    chartEmpty: el("chart-empty"),
    chartTitle: el("chart-title"),
    cprProvenance: el("cpr-provenance"),
    vwapLabel: el("vwap-label"),
    vwapFootnote: el("vwap-footnote"),
    tf1: el("tf-1"),
    tf5: el("tf-5"),
    tfD: el("tf-d"),
  };

  let etag = null;
  let pollSeconds = REFRESH_FALLBACK_SECONDS;
  let lastFreshAt = Date.now();
  /* Exactly one pending timer and one in-flight request at any moment. Without
   * these, every return to the tab started an ADDITIONAL poll chain: the old
   * timer was never cancelled, so N tab-switches meant N concurrent loops all
   * polling a server that publishes once per interval. Most of those requests
   * then got a 304, which the page read as "the runner has gone quiet" and
   * flashed a staleness banner that cleared on the next publish. */
  let pollTimer = null;
  let pollInFlight = false;
  let chart = null;
  let candleSeries = null;
  let vwapSeries = null;
  let kSeries = null;
  let dSeries = null;
  let seriesVersion = -1;
  /* The last `/api/chart` response, holding BOTH timeframes. Keeping it means
   * the timeframe toggle redraws from memory instead of issuing a request
   * that could fail and leave a blank chart. */
  let chartPayload = null;
  /* The per-day and per-month CPR ladders, fetched once. Each entry is one
   * band: a time span and the levels that belong to it. */
  let cprLadders = null;
  let cprLaddersInFlight = false;
  /* Series handles, NOT price-line handles. A price line in this library is
   * full chart width by definition -- that is the API, not a style -- so a band
   * that stops at the end of its own day has to be a LineSeries. */
  let cprSeries = [];
  let cprSignature = null;
  /* Which timeframe the series currently hold. Switching changes both the bar
   * spacing and the span, so the visible range has to be refit -- but ONLY
   * then, never on the once-a-minute refresh, or the operator's zoom would be
   * yanked back every minute. */
  let renderedTimeframe = null;
  /* Set whenever the visible range needs refitting, cleared once a fit has
   * actually landed. See `requestFit`. */
  let pendingFit = false;

  /* ------------------------------------------------------------- history */
  /* Years of candles, fetched a page at a time as the operator scrolls back
   * and kept for the session. Page 0 is the NEWEST slice, so scrolling left
   * just asks for the next number.
   *
   * These are held SEPARATELY from `chartPayload`, which is replaced wholesale
   * every time a minute closes. Merging them at draw time is what stops that
   * once-a-minute refresh from throwing away everything scrolled back to. */
  const historyBars = { "1": [], "5": [], "D": [] };
  /* Indicator columns that came WITH the history pages, kept the same length as
   * `historyBars` so a slice of one lines up with a slice of the other. Only
   * the Daily series carries any: its stochastic is computed on daily candles,
   * which is a different statement from the live payload's minute one. */
  const historyCols = { "1": {}, "5": {}, "D": {} };
  const historyNextPage = { "1": 0, "5": 0, "D": 0 };
  const historyExhausted = { "1": false, "5": false, "D": false };
  let historyInFlight = false;
  /* How many bars the series currently hold, so a prepend can work out how far
   * the logical indices moved and put the viewport back where it was. */
  let renderedBarCount = 0;
  /* Start fetching while this many bars are still to the left of the viewport,
   * so the next page is usually there before the operator reaches it. */
  const HISTORY_PREFETCH_BARS = 200;
  /* One `/api/chart` request at a time. It is fired without `await` from the
   * poll loop, so back-to-back polls could otherwise put two in the air. */
  let chartFetchInFlight = false;

  /* ------------------------------------------------------------ formatting */
  const DASH = '<span class="dash">—</span>';

  function money(value) {
    if (value === null || value === undefined) return DASH;
    const sign = value > 0 ? "+" : "";
    const cls = value > 0 ? "pos" : value < 0 ? "neg" : "";
    const text = sign + value.toLocaleString("en-IN", {
      minimumFractionDigits: 2, maximumFractionDigits: 2,
    });
    return `<span class="${cls}">${text}</span>`;
  }

  function price(value) {
    if (value === null || value === undefined || value === 0) return DASH;
    return value.toFixed(2);
  }

  function text(value) {
    if (value === null || value === undefined || value === "") return DASH;
    return String(value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function flagClass(flag) {
    if (flag === "LIVE") return "flag live";
    if (flag === "INDETERMINATE" || flag.startsWith("EXIT FAILED")) return "flag bad";
    if (flag === "STALE MARK") return "flag warn";
    return "flag";
  }

  /* ---------------------------------------------------------------- header */
  function renderHeader(doc) {
    const session = doc.session || {};
    const feed = doc.feed || {};
    const totals = doc.totals || {};

    nodes.sessionDate.textContent = session.date || "—";
    nodes.modeBadge.textContent = session.any_live ? "LIVE" : "PAPER";
    nodes.modeBadge.className = session.any_live ? "badge live" : "badge";
    nodes.phaseBadge.textContent = (session.phase || "").replace("_", " ");

    nodes.feedSource.textContent = feed.source || "—";
    nodes.feedSpot.textContent = "NIFTY " + (feed.spot === null ? "—" : feed.spot);
    nodes.feedBar.textContent = "bar " + (feed.newest_bar || "—");

    const age = feed.fetch_age_seconds;
    const feedStale = age !== null && age !== undefined && age > 30;
    nodes.staleChip.hidden = !feedStale;
    if (feedStale) nodes.staleChip.textContent = `FEED ${age.toFixed(0)}s OLD`;

    nodes.totalRealized.innerHTML = money(totals.realized);
    nodes.totalTotal.innerHTML = money(totals.total);
    /* An open figure that omits unpriced strategies is a partial sum, and is
     * labelled as one rather than presented as the whole. */
    const unpriced = totals.unpriced_strategies || 0;
    nodes.totalOpen.innerHTML =
      money(totals.open) +
      (unpriced ? ` <span class="dash">(${unpriced} unpriced)</span>` : "");
    nodes.generatedAt.textContent = doc.generated_at || "—";

    nodes.sessionStateNote.hidden = session.session_state_enabled !== false;
  }

  /* ---------------------------------------------------------- open positions */
  function renderOpenPositions(doc) {
    const positions = doc.open_positions || [];
    nodes.openCount.textContent = positions.length;
    nodes.openEmpty.hidden = positions.length > 0;
    if (positions.length === 0) {
      nodes.openEmpty.textContent =
        (doc.session || {}).phase === "PRE_OPEN"
          ? "Market has not opened yet."
          : "No open positions.";
    }

    const rows = [];
    for (const position of positions) {
      const legs = position.legs || [];
      const primary = legs[0] || {};
      const flags = (position.flags || [])
        .map((flag) => `<span class="${flagClass(flag)}">${text(flag)}</span>`)
        .join("");

      rows.push(`<tr>
        <td>${text(position.strategy)}</td>
        <td class="muted">${text(position.slot)}</td>
        <td>${text(position.direction)}</td>
        <td class="mono">${text(primary.symbol)}</td>
        <td>${text(primary.side)}</td>
        <td class="num">${position.quantity || DASH}</td>
        <td class="mono">${text(position.entry_time)}</td>
        <td class="num">${price(primary.entry_price)}</td>
        <td class="num">${price(primary.ltp)}</td>
        <td class="num">${money(position.unrealized_pnl)}</td>
        <td>${flags}</td>
      </tr>`);

      /* A hedged pair renders as a parent row plus one indented row per leg,
       * so the offsetting leg is never invisible. */
      if (legs.length > 1) {
        for (const leg of legs) {
          rows.push(`<tr class="leg-row">
            <td colspan="3">${text(leg.label)}</td>
            <td class="mono">${text(leg.symbol)}</td>
            <td>${text(leg.side)}</td>
            <td class="num">${leg.quantity || DASH}</td>
            <td></td>
            <td class="num">${price(leg.entry_price)}</td>
            <td class="num">${price(leg.ltp)}</td>
            <td colspan="2">${leg.stale_mark ? '<span class="flag warn">STALE</span>' : ""}</td>
          </tr>`);
        }
      }
    }
    nodes.openBody.innerHTML = rows.join("");
  }

  /* -------------------------------------------------------- strategy rollup */
  function renderStrategies(doc) {
    const rows = doc.strategies || [];
    nodes.strategyEmpty.hidden = rows.length > 0;
    nodes.strategyBody.innerHTML = rows
      .map(
        (row) => `<tr>
          <td>${text(row.strategy)}${row.snapshot_valid === false
            ? ' <span class="flag bad">READ FAILED</span>' : ""}</td>
          <td><span class="badge ${row.live_trading ? "live" : "subtle"}">${text(row.mode)}</span></td>
          <td class="num">${row.trades}</td>
          <td class="num">${money(row.realized)}</td>
          <td class="num">${money(row.open)}</td>
          <td class="num">${money(row.total)}</td>
        </tr>`
      )
      .join("");
  }

  /* ---------------------------------------------------------- closed trades */
  function renderClosed(doc) {
    const groups = doc.closed_trades || [];
    const total = groups.reduce((sum, group) => sum + group.trades, 0);
    nodes.closedCount.textContent = total;
    nodes.closedEmpty.hidden = total > 0;

    /* `<details>` keeps its own open/closed state, so groups are only rebuilt
     * when the set of strategies or their trade counts change -- otherwise a
     * once-a-second re-render would snap every expanded group shut. */
    const signature = groups.map((group) => `${group.strategy}:${group.trades}`).join("|");
    if (signature === nodes.closedGroups.dataset.signature) return;
    nodes.closedGroups.dataset.signature = signature;

    nodes.closedGroups.innerHTML = groups
      .map((group) => {
        const rows = group.rows
          .map(
            (trade) => `<tr>
              <td class="mono">${text(trade.entry_time)}</td>
              <td class="mono">${text(trade.exit_time)}</td>
              <td>${text(trade.direction)}</td>
              <td class="mono">${text((trade.symbols || []).join(" + "))}</td>
              <td class="num">${trade.quantity || DASH}</td>
              <td class="num">${price(trade.entry_price)}</td>
              <td class="num">${price(trade.exit_price)}</td>
              <td class="num">${money(trade.pnl)}</td>
              <td>${text(trade.reason)}</td>
              <td class="muted">${text(trade.mode)}</td>
              <td>${trade.pair_confidence === "EXACT" ? ""
                : `<span class="flag warn">${text(trade.pair_confidence)}</span>`}</td>
            </tr>`
          )
          .join("");
        return `<details>
          <summary>
            <span class="name">${text(group.strategy)}</span>
            <span class="muted">${group.trades} trade(s)</span>
            <span class="mono">${money(group.realized)}</span>
          </summary>
          <div class="scroll"><table>
            <thead><tr>
              <th>Entry</th><th>Exit</th><th>Dir</th><th>Symbol</th><th class="num">Qty</th>
              <th class="num">Entry px</th><th class="num">Exit px</th><th class="num">P&amp;L</th>
              <th>Reason</th><th>Mode</th><th>Pairing</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
        </details>`;
      })
      .join("");
  }

  function renderNotices(doc) {
    const notices = doc.notices || [];
    nodes.noticesPane.hidden = notices.length === 0;
    nodes.notices.innerHTML = notices
      .slice(-40)
      .reverse()
      .map(
        (notice) => `<li class="${notice.urgent ? "urgent" : ""}">
          <span class="mono">${text(notice.time)}</span>
          ${text(notice.strategy)} — <strong>${text(notice.action)}</strong>
          ${notice.detail ? text(notice.detail) : ""}
        </li>`
      )
      .join("");
  }

  /* ----------------------------------------------------------------- chart */
  /* ----------------------------------------------------------------- chart */
  /* Indicator visibility and the timeframe are per-browser preferences, held
   * in a VERSIONED key so a future rename falls back to defaults instead of
   * silently hiding a line. Every access is wrapped: a private window, cleared
   * site data or a corrupt value must not take the page down with it. */
  const PREFS_KEY = "algoDashboard.chart.v1";
  const PREF_DEFAULTS = {
    tf: "1", cpr: true, cprPD: true, cprRS1: false, cprRS3: false, vwap: true, stoch: true,
  };

  function loadPrefs() {
    try {
      const stored = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
      return { ...PREF_DEFAULTS, ...(stored && typeof stored === "object" ? stored : {}) };
    } catch (error) {
      return { ...PREF_DEFAULTS };
    }
  }

  function savePrefs() {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    } catch (error) {
      /* A preference that cannot be remembered is not worth an error. */
    }
  }

  const prefs = loadPrefs();

  /* CPR groups, each with its own checkbox. `core` and `pd` are on by
   * default; the R/S ladders are opt-in because thirteen horizontal lines over
   * candles is not a chart, it is a net.
   *
   * `pd` is the prior day's traded HIGH and LOW rather than arithmetic derived
   * from them, which is why it toggles separately from the pivot family. Both
   * share one colour: they are a range, and the axis labels name each end.
   *
   * The pivot is cyan rather than the amber it started as, because VWAP is
   * amber too and both draw on pane 0 -- once every CPR level became solid
   * the two were only a glance apart. Cyan is the one hue this pane was not
   * already spending: the ladders own red and green, `pd` owns violet, BC
   * and TC are grey. (%D in the oscillator pane stays amber; it never
   * shares a pane with VWAP.) */
  const CPR_LEVELS = [
    { key: "pivot",     group: "core", title: "P",   color: "#3fc9d9" },
    { key: "bc",        group: "core", title: "BC",  color: "#8b94a3" },
    { key: "tc",        group: "core", title: "TC",  color: "#8b94a3" },
    { key: "prev_high", group: "pd",   title: "PDH", color: "#a78bda" },
    { key: "prev_low",  group: "pd",   title: "PDL", color: "#a78bda" },
    { key: "r1",        group: "rs1",  title: "R1",  color: "#ef5f5f" },
    { key: "r2",        group: "rs1",  title: "R2",  color: "#ef5f5f" },
    { key: "s1",        group: "rs1",  title: "S1",  color: "#35c46b" },
    { key: "s2",        group: "rs1",  title: "S2",  color: "#35c46b" },
    { key: "r3",        group: "rs3",  title: "R3",  color: "#8a3b3b" },
    { key: "r4",        group: "rs3",  title: "R4",  color: "#8a3b3b" },
    { key: "s3",        group: "rs3",  title: "S3",  color: "#2c6b45" },
    { key: "s4",        group: "rs3",  title: "S4",  color: "#2c6b45" },
  ];

  function ensureChart() {
    if (chart || typeof LightweightCharts === "undefined") return;
    chart = LightweightCharts.createChart(nodes.chart, {
      layout: {
        background: { color: "#1b1f26" },
        textColor: "#8b94a3",
        /* Left enabled deliberately: the vendored library is Apache-2.0 and
         * this is its attribution. See vendor/NOTICE-lightweight-charts.md. */
        attributionLogo: true,
        /* The library's own separator defaults are light-theme greys
         * (#E0E3EB) and would draw a bright bar across this panel. */
        panes: {
          enableResize: true,
          separatorColor: "#2c333d",
          separatorHoverColor: "rgba(91, 157, 217, 0.35)",
        },
      },
      grid: {
        vertLines: { color: "#232932" },
        horzLines: { color: "#232932" },
      },
      rightPriceScale: { borderColor: "#2c333d" },
      timeScale: { borderColor: "#2c333d", timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      autoSize: true,
    });

    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#35c46b", downColor: "#ef5f5f",
      borderUpColor: "#35c46b", borderDownColor: "#ef5f5f",
      wickUpColor: "#35c46b", wickDownColor: "#ef5f5f",
    }, 0);

    vwapSeries = chart.addSeries(LightweightCharts.LineSeries, {
      color: "#e8b13a", lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, title: "VWAP*", visible: prefs.vwap,
    }, 0);

    /* Pane 1: the oscillator. v5 takes the pane index as addSeries' third
     * argument and creates the pane on demand. */
    kSeries = chart.addSeries(LightweightCharts.LineSeries, {
      color: "#5b9dd9", lineWidth: 1, priceLineVisible: false, title: "%K",
    }, 1);
    dSeries = chart.addSeries(LightweightCharts.LineSeries, {
      color: "#e8b13a", lineWidth: 1, priceLineVisible: false, title: "%D",
    }, 1);
    for (const level of [80, 20]) {
      kSeries.createPriceLine({
        price: level, color: "#39414d", lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: true,
      });
    }
    applyStochVisibility();

    /* Fitting the visible range requires the container to HAVE a width. On a
     * first paint -- or in a tab that is still hidden -- it can be zero, and
     * fitting against a zero-width box silently does nothing while leaving a
     * nonsense bar spacing behind, which draws every candle squeezed into a
     * few pixels at the right edge. Observing the element means the fit
     * happens exactly when layout gives it a size, however late that is. */
    try {
      new ResizeObserver(() => applyPendingFit()).observe(nodes.chart);
    } catch (error) {
      /* No ResizeObserver: the per-render retry below still covers it. */
    }

    /* Scrolling left past the start of what is loaded fetches the next older
     * page. `from` is a LOGICAL index and goes negative once the operator
     * scrolls past the first bar, so the test is "close to, or past, the left
     * edge" rather than a positive threshold. */
    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (!range) return;
      if (range.from <= HISTORY_PREFETCH_BARS) loadNextHistoryPage();
    });
  }

  function requestFit() {
    pendingFit = true;
    applyPendingFit();
  }

  function applyPendingFit() {
    if (!pendingFit || !chart) return;
    if (!nodes.chart.clientWidth) return;  // not laid out yet; try again later
    chart.timeScale().fitContent();
    pendingFit = false;
  }

  const STOCH_PANE_HEIGHT = 110;

  function applyStochVisibility() {
    if (!kSeries) return;
    kSeries.applyOptions({ visible: prefs.stoch });
    dSeries.applyOptions({ visible: prefs.stoch });
    /* Hiding the lines is not enough: the pane itself keeps its height and
     * the price chart stays short, which is the opposite of what turning the
     * oscillator off is for. Collapse it so the candles get the room back. */
    try {
      chart.panes()[1].setHeight(prefs.stoch ? STOCH_PANE_HEIGHT : 1);
    } catch (error) {
      /* Pane sizing is cosmetic; never let it stop the chart drawing. */
    }
  }

  /* Indicator columns arrive as bare numbers zipped against bars[i].time --
   * ~9 bytes a point instead of ~36 for {time, value} objects. */
  function pointsFrom(bars, values) {
    const points = [];
    for (let index = 0; index < bars.length; index += 1) {
      const value = values ? values[index] : null;
      if (value !== null && value !== undefined) {
        points.push({ time: bars[index].time, value });
      }
    }
    return points;
  }

  /* History and the live window, spliced into one strictly ascending array.
   *
   * lightweight-charts DROPS bars silently when times are not strictly
   * ascending, so the seam matters: the live block and the newest history page
   * overlap, and the live copy wins because it is the one that keeps updating.
   */
  /* The CPR ladders, fetched once and kept. They are static: the server
   * renders them when it loads history and they never change after that. */
  async function ensureCprLadders() {
    if (cprLadders || cprLaddersInFlight) return;
    cprLaddersInFlight = true;
    try {
      const response = await fetch("/api/history?tf=cpr&page=0", { cache: "no-store" });
      if (!response.ok) return;          /* no history: today's band still draws */
      cprLadders = await response.json();
      if (chartPayload) applyTimeframe();
    } catch (error) {
      /* Without them the chart shows today's band alone, which is what it did
       * before per-day CPR existed. The next redraw tries again. */
    } finally {
      cprLaddersInFlight = false;
    }
  }

  function historyCut(liveBars) {
    const history = historyBars[prefs.tf] || [];
    if (!liveBars.length) return history.length;
    const firstLive = liveBars[0].time;
    let cut = history.length;
    while (cut > 0 && history[cut - 1].time >= firstLive) cut -= 1;
    return cut;
  }

  function mergedBars(liveBars, cut) {
    const history = historyBars[prefs.tf] || [];
    if (!history.length) return liveBars;
    return history.slice(0, cut).concat(liveBars);
  }

  /* One indicator column over the merged series: the stored part where history
   * carried it, then the live part. */
  function mergedColumn(name, liveValues, cut) {
    const history = historyBars[prefs.tf] || [];
    const stored = (historyCols[prefs.tf] || {})[name] || new Array(history.length).fill(null);
    return stored.slice(0, cut).concat(liveValues || []);
  }

  /* Today, as ONE daily candle, folded out of the live minute bars.
   *
   * The runner's store holds minutes, so the Daily timeframe would otherwise
   * stop at yesterday -- the history CSV is only as fresh as the last download.
   * Stamped at 09:15 of the session exactly as the server stamps its own daily
   * bars, so today lands on the same grid instead of a slot of its own. */
  function dailyLiveBars() {
    const minute = chartPayload && chartPayload.timeframes && chartPayload.timeframes["1"];
    const bars = (minute && minute.bars) || [];
    if (!bars.length) return [];
    const lastDay = Math.floor(bars[bars.length - 1].time / 86400);
    let start = bars.length - 1;
    while (start > 0 && Math.floor(bars[start - 1].time / 86400) === lastDay) start -= 1;
    const session = bars.slice(start);
    let high = session[0].high;
    let low = session[0].low;
    for (const bar of session) {
      if (bar.high > high) high = bar.high;
      if (bar.low < low) low = bar.low;
    }
    return [{
      time: lastDay * 86400 + 9 * 3600 + 15 * 60,
      open: session[0].open,
      high,
      low,
      close: session[session.length - 1].close,
    }];
  }

  /* Which bands belong on the chart right now.
   *
   * Two rules, and both matter. The ladder follows the timeframe: minutes get
   * the DAILY bands, the Daily timeframe gets the MONTHLY ones, because a daily
   * band on a daily candle is one bar wide and says nothing.
   *
   * And bands are clipped to the bars actually loaded. Every series shares one
   * time scale built from the UNION of their times, so handing over five years
   * of bands while only a week of candles is loaded would stretch the scale
   * across five years of empty chart. */
  function cprBandsForView(bars) {
    if (!bars.length) return [];
    const monthly = prefs.tf === "D";
    const ladder = (cprLadders && (monthly ? cprLadders.month : cprLadders.day)) || [];
    const leftEdge = bars[0].time;
    const rightEdge = bars[bars.length - 1].time;

    /* Where today starts, so the live band covers exactly today and the stored
     * ones stop before it. Bar times are IST wall-clock labelled UTC, so a
     * plain division by a day lands on the session date. */
    const lastDay = Math.floor(rightEdge / 86400);
    let todayFrom = bars.length - 1;
    while (todayFrom > 0 && Math.floor(bars[todayFrom - 1].time / 86400) === lastDay) {
      todayFrom -= 1;
    }
    const todayStart = bars[todayFrom].time;

    /* `band.to < todayStart` exists ONLY to clear the way for the live band
     * pushed below, and there is no live band on the monthly ladder. Applying it
     * there dropped the CURRENT month every time: each daily bar is its own day,
     * so `todayStart` is simply the last bar, which is exactly where the newest
     * month band ends. The chart then showed last month's levels while the
     * caption named this month's -- 24167.97 drawn against 24282.77 captioned. */
    const bands = ladder.filter(
      (band) => band.to >= leftEdge
        && band.from <= rightEdge
        && (monthly || band.to < todayStart),
    );

    /* Today's band comes from the LIVE payload, never the stored ladder: the
     * history CSV can be days old, and today is the one the operator is
     * trading. On the Daily timeframe the monthly ladder already covers the
     * current month, so there is nothing to add. */
    const live = chartPayload && chartPayload.cpr;
    if (!monthly && live && live.available) {
      bands.push({ from: todayStart, to: rightEdge, levels: live });
    }
    return bands;
  }

  /* The month band the Daily caption describes, plus the month its numbers
   * came FROM -- which is the month before it, and the thing worth naming. */
  function newestMonthBand() {
    const ladder = (cprLadders && cprLadders.month) || [];
    if (!ladder.length) return { levels: null };
    const band = ladder[ladder.length - 1];
    const previous = ladder.length > 1 ? ladder[ladder.length - 2].month : null;
    return { levels: band.levels, previous: previous || "" };
  }

  function renderCprLines(bars) {
    if (!chart || !candleSeries) return;
    const bands = cprBandsForView(bars || []);

    /* EVERY group pref belongs in this signature. One left out means ticking
     * its box updates the pref and then this function early-returns on an
     * unchanged signature, so the band never appears and the bug looks like the
     * box doing nothing.
     *
     * Keying on a single pivot is no longer enough either, now that there are
     * hundreds of bands: the set is identified by its extent and its newest
     * pivot, so a new session, a newly loaded page and a timeframe switch each
     * change it. */
    const newest = bands.length ? bands[bands.length - 1] : null;
    const signature = JSON.stringify([
      prefs.tf === "D" ? "month" : "day",
      bands.length,
      bands.length ? bands[0].from : null,
      newest ? newest.to : null,
      newest ? newest.levels.pivot : null,
      prefs.cpr, prefs.cprPD, prefs.cprRS1, prefs.cprRS3,
    ]);
    if (signature === cprSignature) return;
    cprSignature = signature;

    for (const handle of cprSeries) chart.removeSeries(handle);
    cprSeries = [];
    if (!bands.length || !prefs.cpr) return;

    /* A whitespace point -- a time with no value -- is what BREAKS the line
     * between one day's band and the next. Without it the series would draw a
     * diagonal from yesterday's pivot to today's, straight across the gap.
     * One second past the band's end, so it cannot collide with a real bar. */
    const gap = 1;

    for (const level of CPR_LEVELS) {
      const on = level.group === "core"
        || (level.group === "pd" && prefs.cprPD)
        || (level.group === "rs1" && prefs.cprRS1)
        || (level.group === "rs3" && prefs.cprRS3);
      if (!on) continue;

      const points = [];
      for (const band of bands) {
        const price = band.levels[level.key];
        if (price === null || price === undefined) continue;
        points.push({ time: band.from, value: price });
        if (band.to > band.from) points.push({ time: band.to, value: price });
        points.push({ time: band.to + gap });
      }
      if (!points.length) continue;

      const series = chart.addSeries(LightweightCharts.LineSeries, {
        color: level.color,
        lineWidth: 1,
        priceLineVisible: false,
        /* The last value still gets an axis label, and `title` puts the level's
         * NAME beside it -- so the "(chart)" caveat still travels with a
         * screenshot of just the chart, as it did when these were price lines
         * and none of the page's other captions would. */
        lastValueVisible: true,
        title: `${level.title} (chart)`,
      }, 0);
      series.setData(points);
      cprSeries.push(series);
    }
  }

  function renderCprProvenance(cpr, monthly) {
    /* On the Daily timeframe the bands are MONTHLY, so the daily payload's
     * caption would describe a session nobody is looking at. Describe the month
     * band instead, from its own numbers. */
    if (monthly) {
      if (!monthly.levels) {
        nodes.cprProvenance.hidden = true;
        return;
      }
      const levels = monthly.levels;
      nodes.cprProvenance.textContent = [
        `Prior month ${monthly.previous || ""}`.trim(),
        `H ${levels.prev_high} L ${levels.prev_low}`,
        `pivot ${levels.pivot} (chart)`,
        "sessions truncated at 15:15",
      ].join(" · ");
      nodes.cprProvenance.hidden = false;
      return;
    }
    if (!cpr || !cpr.available) {
      nodes.cprProvenance.hidden = false;
      nodes.cprProvenance.textContent = cpr && cpr.unavailable_reason
        ? `CPR unavailable — ${cpr.unavailable_reason}.`
        : "";
      nodes.cprProvenance.hidden = !nodes.cprProvenance.textContent;
      return;
    }
    /* Built from the payload's OWN inputs, not a fixed string: the operator
     * can see the 15:15 close rather than take it on trust, and can see how
     * far it sits from the figure the strategies actually trade. */
    const parts = [
      `Prior ${cpr.prior_session_date} ${cpr.window}`,
      `${cpr.bars_used} bars`,
      `H ${cpr.prev_high} L ${cpr.prev_low} C ${cpr.prev_close}`,
      `pivot ${cpr.pivot} (chart)`,
    ];
    if (cpr.strategies_pivot !== null && cpr.strategies_pivot !== undefined) {
      parts.push(`strategies ${cpr.strategies_pivot} (full session)`);
    }
    if (cpr.partial) parts.push("partial prior session");
    if (cpr.width) parts.push(cpr.width);
    nodes.cprProvenance.textContent = parts.join(" · ");
    nodes.cprProvenance.hidden = false;
  }

  function applyTimeframe() {
    if (!chartPayload || !candleSeries) return;
    /* The runner publishes minute timeframes only. The Daily series is
     * history plus today folded out of the live minutes, so it has a block of
     * its own rather than one from the payload. */
    const daily = prefs.tf === "D";
    const block = daily
      ? { minutes: 0, bars: [], vwap: [], stoch_k: [], stoch_d: [] }
      : (chartPayload.timeframes[prefs.tf] || chartPayload.timeframes["1"]);
    if (!block) return;

    const live = daily ? dailyLiveBars() : (block.bars || []);
    const cut = historyCut(live);
    const bars = mergedBars(live, cut);
    candleSeries.setData(bars);
    renderedBarCount = bars.length;

    /* On the minute timeframes the indicator columns cover the LIVE window
     * alone -- history pages carry candles and nothing else, because VWAP and a
     * minute stochastic are statements about the session being traded. On Daily
     * the stochastic comes the other way round, out of the history pages, since
     * it is computed on daily candles. Either way the arrays are index-aligned
     * with the bars they were sliced beside. */
    vwapSeries.setData(pointsFrom(bars, mergedColumn("vwap", block.vwap, cut)));
    kSeries.setData(pointsFrom(bars, mergedColumn("stoch_k", block.stoch_k, cut)));
    dSeries.setData(pointsFrom(bars, mergedColumn("stoch_d", block.stoch_d, cut)));

    /* A session VWAP has no meaning on a daily candle, so it is hidden rather
     * than drawn as a line that looks like it means something. */
    vwapSeries.applyOptions({ visible: prefs.vwap && !daily });
    nodes.vwapLabel.classList.toggle("disabled", daily);

    if (renderedTimeframe !== prefs.tf) {
      renderedTimeframe = prefs.tf;
      /* 75 five-minute bars cover a different span from 375 one-minute ones,
       * so without a refit the new candles sit outside the old visible range
       * and the pane looks empty apart from the price lines.
       *
       * Deferred to the next PAINT, not called inline: on a first load the
       * container can still have zero width, and fitting against a zero-width
       * box silently does nothing and leaves the stale range behind. A hidden
       * tab does not run animation frames at all, so this also waits, by
       * itself, until the tab is actually shown. */
      requestFit();
    }

    const minutes = block.minutes || 1;
    const window = (chartPayload.cpr && chartPayload.cpr.window) || "09:15-15:15";
    nodes.chartTitle.textContent = daily
      ? "NIFTY · daily · CPR from the prior MONTH (chart only)"
      : `NIFTY · ${minutes} minute${minutes === 1 ? "" : "s"} · CPR from prior session `
        + `${window} (chart only)`;
    renderCprLines(bars);
    renderCprProvenance(chartPayload.cpr, daily ? newestMonthBand() : null);
    ensureCprLadders();

    const vwapMeta = (chartPayload.indicators || {}).vwap || {};
    nodes.vwapLabel.title = vwapMeta.note || "";
    nodes.vwapFootnote.hidden = vwapMeta.is_proxy === false;

    /* Fetch the first page outright rather than waiting for a range event to
     * say the viewport is near the left edge.
     *
     * That event only lands if a FIT has landed, and `applyPendingFit` refuses
     * to fit a zero-width container -- which is the state a chart can sit in
     * indefinitely, because the ResizeObserver meant to catch the layout has
     * been measured NOT firing in some panes. Waiting on it meant a chart that
     * silently never loaded any history at all. The re-entrant call is stopped
     * by `historyInFlight`, and by history existing once a page has landed. */
    if (!(historyBars[prefs.tf] || []).length && !historyExhausted[prefs.tf]) {
      loadNextHistoryPage();
    }
  }

  /* Redraw after prepending history, WITHOUT moving what the operator is
   * looking at.
   *
   * `fitContent` is deliberately not used here. It is the only fit call site
   * and it exists for timeframe switches; firing it on a prepend would yank
   * the viewport back to the whole series every time a page arrived, which is
   * the opposite of what scrolling back is for. Prepending N bars shifts every
   * logical index by N, so the range is simply moved by the same N. */
  function redrawKeepingViewport() {
    const scale = chart ? chart.timeScale() : null;
    const before = scale ? scale.getVisibleLogicalRange() : null;
    const previousCount = renderedBarCount;
    applyTimeframe();
    const added = renderedBarCount - previousCount;
    if (scale && before && added > 0) {
      scale.setVisibleLogicalRange({ from: before.from + added, to: before.to + added });
    }
    return added;
  }

  /* Fetch the next older page for the CURRENT timeframe.
   *
   * A 404 means there is nothing older -- the server answers that way whether
   * history was never loaded or the operator has reached the start of it -- so
   * the timeframe is marked exhausted and stops asking. */
  async function loadNextHistoryPage() {
    const timeframe = prefs.tf;
    if (historyInFlight || historyExhausted[timeframe]) return;
    historyInFlight = true;
    try {
      const page = historyNextPage[timeframe];
      const response = await fetch(
        `/api/history?tf=${encodeURIComponent(timeframe)}&page=${page}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        /* A 404 has two meanings and they must not be confused. Once a page has
         * loaded it means the operator has reached the start of history, and
         * asking again is pointless. BEFORE that it means the server is still
         * building -- measured at 29.5 seconds on the five-year file -- and
         * treating it as exhausted would disable scroll-back for the whole
         * session over a few seconds of startup. */
        if ((historyBars[timeframe] || []).length) historyExhausted[timeframe] = true;
        return;
      }
      const doc = await response.json();
      const bars = doc.bars || [];
      if (!bars.length) {
        historyExhausted[timeframe] = true;
        return;
      }
      historyBars[timeframe] = bars.concat(historyBars[timeframe] || []);
      for (const name of ["stoch_k", "stoch_d"]) {
        /* A page without the column still contributes its LENGTH, as nulls, so
         * the arrays stay index-aligned with the bars either way. */
        const incoming = doc[name] || new Array(bars.length).fill(null);
        const columns = historyCols[timeframe];
        columns[name] = incoming.concat(columns[name] || []);
      }
      historyNextPage[timeframe] = page + 1;
      if (doc.pages && historyNextPage[timeframe] >= doc.pages) {
        historyExhausted[timeframe] = true;
      }
      /* Only redraw if the operator has not switched timeframe while this was
       * in the air; otherwise the bars belong to a chart nobody is looking at. */
      const added = prefs.tf === timeframe ? redrawKeepingViewport() : 0;

      if (!added && !historyExhausted[timeframe]) {
        /* Every bar in that page was already inside the live window, so the
         * merge dropped all of it: the chart gained nothing and the viewport
         * did not move, which means NO range event will arrive to ask for the
         * next page. Without this the scroll-back would simply stop.
         *
         * It happens whenever a page is shorter than the live window -- not at
         * the shipped sizes (2,000 against 375) but `DASHBOARD_CHART_BARS` goes
         * up to 2,200, which would silently break it. Deferred rather than
         * recursive so `historyInFlight` has been cleared first; it ends as
         * soon as one page reaches back past the live window, or at the 404. */
        setTimeout(loadNextHistoryPage, 0);
      }
    } catch (error) {
      /* A failed page costs the scroll-back, never the chart. The next scroll
       * tries again; the timeframe is deliberately NOT marked exhausted. */
    } finally {
      historyInFlight = false;
    }
  }

  function setTimeframe(value) {
    prefs.tf = value;
    savePrefs();
    for (const [button, key] of [[nodes.tf1, "1"], [nodes.tf5, "5"], [nodes.tfD, "D"]]) {
      button.classList.toggle("active", value === key);
      button.setAttribute("aria-pressed", String(value === key));
    }
    /* Switching redraws from the payload the browser ALREADY holds: no fetch,
     * so the toggle has no way to fail. */
    applyTimeframe();
  }

  async function renderChart(doc) {
    const info = doc.chart || {};
    const hasData = info.state === "OK" && info.last_bar;
    nodes.chartEmpty.hidden = hasData;
    if (!hasData) return;

    ensureChart();
    if (!candleSeries) return;

    /* The arrays are refetched only when a new MINUTE closed -- roughly 375
     * times a session rather than once per poll. In between, the forming
     * candle is updated in place, which is what makes the chart move
     * tick-by-tick under MARKET_DATA_SOURCE=WEBSOCKET for a few hundred bytes. */
    if (info.series_version !== seriesVersion && !chartFetchInFlight) {
      /* Guarded: this runs from `render`, which the poll loop calls WITHOUT
       * awaiting, so two polls landing close together could otherwise put two
       * requests in the air for the same version. */
      chartFetchInFlight = true;
      try {
        const response = await fetch("/api/chart", { cache: "no-store" });
        if (response.ok) {
          chartPayload = await response.json();
          seriesVersion = info.series_version;
          /* Keeping the viewport, not refitting: by now the operator may have
           * scrolled back through years of history, and a minute closing must
           * not drag them back to today. */
          redrawKeepingViewport();
        }
      } catch (error) {
        /* Leave the previous series on screen; the next poll retries. */
        return;
      } finally {
        chartFetchInFlight = false;
      }
    }

    /* Only the forming candle moves between minutes. Indicators are computed
     * on COMPLETED bars, so they deliberately have no point here. */
    const forming = prefs.tf === "5" ? info.last_bar_5m : info.last_bar;
    if (forming) candleSeries.update(forming);
    applyPendingFit();
  }

  /* ------------------------------------------------------------------ poll */
  function setLiveness(state, message) {
    nodes.dot.className = "dot " + state;
    if (message) {
      nodes.banner.textContent = message;
      nodes.banner.className = "banner" + (state === "dead" ? " dead" : "");
      nodes.banner.hidden = false;
    } else {
      nodes.banner.hidden = true;
    }
  }

  function render(doc) {
    renderHeader(doc);
    renderOpenPositions(doc);
    renderStrategies(doc);
    renderClosed(doc);
    renderNotices(doc);
    renderChart(doc);
  }

  function schedulePoll(delaySeconds) {
    /* Cancelling first is what keeps this to ONE loop. */
    if (pollTimer !== null) clearTimeout(pollTimer);
    pollTimer = setTimeout(poll, delaySeconds * 1000);
  }

  async function poll() {
    /* A request is already out; it will schedule the next one when it lands. */
    if (pollInFlight) return;
    pollInFlight = true;
    let delay = pollSeconds;
    try {
      const headers = etag ? { "If-None-Match": etag } : {};
      const response = await fetch("/api/state", { headers, cache: "no-store" });

      if (response.status === 304) {
        /* Byte-identical, including the document's own clock -- so nothing has
         * been published since THIS tab's last 200. (Other tabs cannot cause
         * this: each keeps its own ETag, so every tab gets a 200 on every new
         * version.) One 304 just means we polled inside a publish gap; only
         * sustained silence means the builder has actually stopped. */
        const quietFor = (Date.now() - lastFreshAt) / 1000;
        if (quietFor >= Math.max(STALE_FLOOR_SECONDS, pollSeconds * STALE_AFTER_INTERVALS)) {
          setLiveness("stale", "The runner has not published an update recently; " +
            "these numbers are frozen. Trading is unaffected.");
        }
      } else if (response.ok) {
        etag = response.headers.get("ETag");
        lastFreshAt = Date.now();
        const doc = await response.json();
        if (doc.poll_seconds) pollSeconds = doc.poll_seconds;
        render(doc);
        setLiveness("ok", null);
      } else if (response.status === 503) {
        setLiveness("stale", "The runner is still starting up…");
      } else {
        setLiveness("dead", `Unexpected response from the runner (${response.status}).`);
      }
    } catch (error) {
      /* The runner exited, or the port closed. Keep the last data on screen --
       * an operator reading a closing session still wants the figures. */
      setLiveness("dead", "Runner not reachable — showing the last data received.");
      delay = RETRY_SECONDS;
    }

    pollInFlight = false;
    /* Cheap and idempotent: it writes only when the header's height actually
     * changed, so a stable window costs one getBoundingClientRect a second. */
    publishHeaderHeight();

    if (document.visibilityState === "hidden") delay = Math.max(delay, HIDDEN_TAB_SECONDS);
    schedulePoll(delay);
  }

  /* The chart pane sizes itself against the viewport minus the sticky header,
   * and that header WRAPS to two or three rows on a narrow window -- 62px wide
   * open, 94px wrapped. Publishing its measured height as a CSS variable keeps
   * the chart flush at any width, where a hardcoded offset leaves a gap or
   * pushes the page into an unnecessary scroll.
   *
   * Re-measured from the poll loop rather than trusted to a ResizeObserver
   * alone: an observer that never fires leaves the variable stale and the gap
   * visible, and that failure is invisible in code review. Writing only on a
   * CHANGE makes the once-a-second check free. */
  let headerHeight = 0;

  function publishHeaderHeight() {
    const header = el("header");
    if (!header) return;
    const height = Math.ceil(header.getBoundingClientRect().height);
    if (!height || height === headerHeight) return;
    headerHeight = height;
    document.documentElement.style.setProperty("--header-h", `${height}px`);
  }

  publishHeaderHeight();
  try {
    /* Instant response when it does work; the poll is the guarantee. */
    new ResizeObserver(publishHeaderHeight).observe(el("header"));
  } catch (error) {
    window.addEventListener("resize", publishHeaderHeight);
  }

  /* Controls. Each checkbox flips visibility rather than re-setting data, so
   * toggling never costs a redraw of the series it keeps. */
  nodes.tf1.addEventListener("click", () => setTimeframe("1"));
  nodes.tf5.addEventListener("click", () => setTimeframe("5"));
  nodes.tfD.addEventListener("click", () => setTimeframe("D"));

  for (const [id, key] of [
    ["ind-cpr", "cpr"], ["ind-cpr-pd", "cprPD"],
    ["ind-cpr-rs1", "cprRS1"], ["ind-cpr-rs3", "cprRS3"],
    ["ind-vwap", "vwap"], ["ind-stoch", "stoch"],
  ]) {
    const box = el(id);
    box.checked = Boolean(prefs[key]);
    box.addEventListener("change", () => {
      prefs[key] = box.checked;
      savePrefs();
      if (key === "vwap" && vwapSeries) vwapSeries.applyOptions({ visible: prefs.vwap });
      else if (key === "stoch") applyStochVisibility();
      else if (chartPayload) applyTimeframe();
    });
  }
  setTimeframe(["1", "5", "D"].includes(prefs.tf) ? prefs.tf : "1");

  document.addEventListener("visibilitychange", () => {
    /* Bring the NEXT poll forward rather than starting a second loop. */
    if (document.visibilityState === "visible") schedulePoll(0);
  });

  schedulePoll(0);
})();
