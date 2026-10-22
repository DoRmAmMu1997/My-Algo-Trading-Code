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
    indMenuButton: el("ind-menu-button"),
    indMenuPanel: el("ind-menu-panel"),
    indPicker: el("ind-picker"),
    indParams: el("ind-params"),
    indAdd: el("ind-add"),
    indActive: el("ind-active"),
    indCount: el("ind-count"),
    instNifty: el("inst-nifty"),
    instBankNifty: el("inst-banknifty"),
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
  /* The instruments the chart can show, primary first. Mirrors the server's
   * frozen `CHART_INSTRUMENTS`; the secondary simply reports NO_DATA and
   * serves no history pages when the runner is not publishing it, which the
   * page already knows how to draw. */
  const INSTRUMENTS = ["NIFTY", "BANKNIFTY"];

  /* Everything the chart holds that belongs to ONE instrument. Each gets its
   * own copy, because a NIFTY payload and a BankNIFTY payload arrive
   * independently and share nothing: not a price scale, not a CPR ladder, not
   * a scroll-back position. Switching instrument therefore costs no refetch of
   * anything already loaded, which is the same reason both timeframes ride in
   * one payload. */
  function freshInstrumentView() {
    return {
      /* The last `/api/chart` response for this instrument, holding BOTH
       * timeframes. Keeping it means the timeframe toggle redraws from memory
       * instead of issuing a request that could fail and leave a blank chart. */
      chartPayload: null,
      seriesVersion: -1,
      /* The per-day and per-month CPR ladders, fetched once per instrument.
       * Each entry is one band: a time span and the levels that belong to it. */
      cprLadders: null,
      cprLaddersInFlight: false,
      /* Years of candles, fetched a page at a time as the operator scrolls
       * back and kept for the session. Page 0 is the NEWEST slice, so
       * scrolling left just asks for the next number.
       *
       * Held SEPARATELY from `chartPayload`, which is replaced wholesale every
       * time a minute closes. Merging them at draw time is what stops that
       * once-a-minute refresh from throwing away everything scrolled back to. */
      historyBars: { "1": [], "5": [], "D": [] },
      /* Indicator columns that came WITH the history pages, kept the same
       * length as `historyBars` so a slice of one lines up with a slice of the
       * other. Only the Daily series carries any: its stochastic is computed on
       * daily candles, which is a different statement from the live payload's
       * minute one. */
      historyCols: { "1": {}, "5": {}, "D": {} },
      historyNextPage: { "1": 0, "5": 0, "D": 0 },
      /* PER INSTRUMENT, deliberately. A 404 means "nothing older" for the
       * instrument that was asked, and the secondary's pages land later than
       * the primary's -- a shared flag would let one instrument finishing its
       * history switch off the other's scroll-back entirely. */
      historyExhausted: { "1": false, "5": false, "D": false },
      /* When history was first asked for and refused. A 404 before any page
       * has loaded means "still building" -- but only for so long; after that
       * the answer is simply that there is none. */
      historyFirstRefusalAt: 0,
    };
  }

  const instrumentViews = {};
  for (const name of INSTRUMENTS) instrumentViews[name] = freshInstrumentView();

  /* The state belonging to the instrument on screen right now. */
  function view() {
    return instrumentViews[prefs.inst] || instrumentViews[INSTRUMENTS[0]];
  }
  /* Series handles, NOT price-line handles. A price line in this library is
   * full chart width by definition -- that is the API, not a style -- so a band
   * that stops at the end of its own day has to be a LineSeries. */
  let cprSeries = [];
  let cprSignature = null;
  /* Which timeframe and instrument the series currently hold. Switching either
   * changes the span -- and an instrument switch changes the price scale
   * outright -- so the visible range has to be refit. But ONLY then, never on
   * the once-a-minute refresh, or the operator's zoom would be yanked back
   * every minute. */
  let renderedTimeframe = null;
  let renderedInstrument = null;
  /* Set whenever the visible range needs refitting, cleared once a fit has
   * actually landed. See `requestFit`. */
  let pendingFit = false;

  /* ------------------------------------------------------------- history */
  /* One request at a time across the whole page -- a resource guard, not
   * per-instrument state. The instrument each reply belongs to is captured
   * before its `await`, so a switch mid-flight cannot merge one index's bars
   * into the other's series. */
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
  /* The bars the chart currently holds. Kept so a CPR checkbox can redraw the
   * levels alone instead of re-uploading every candle. */
  let lastMergedBars = [];
  const HISTORY_GIVE_UP_MS = 5 * 60 * 1000;

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
    /* The first non-scalar preference, which is why `loadPrefs` repairs it
     * rather than trusting it: this comes back from localStorage and a
     * previous version could have written anything into it. */
    indicators: [],
    inst: "NIFTY",
    tf: "1", cpr: true, cprPD: true, cprRS1: false, cprRS3: false, vwap: true, stoch: true,
  };

  function loadPrefs() {
    try {
      const stored = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
      const merged = {
        ...PREF_DEFAULTS,
        ...(stored && typeof stored === "object" ? stored : {}),
      };
      /* A non-array here would throw on the first draw and blank the page --
       * the IIFE has no outer guard. Entries are validated individually later,
       * by `indicatorSpecs`, because an unknown id can arrive the same way. */
      if (!Array.isArray(merged.indicators)) merged.indicators = [];
      return merged;
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

  /* ------------------------------------------------------------ indicators */
  /* The operator's own indicator stack, from the menu. Chart-only: the maths
   * lives in `dashboard_indicators.js` and runs here, over whatever bars the
   * browser holds, so an indicator follows the operator back through years of
   * scroll-back. Nothing the strategies read comes from any of it.
   *
   * Structure is rebuilt only when the SET changes -- adding, removing, or
   * editing a parameter. A new minute or a loaded history page only re-sets
   * data on series that already exist, exactly as the candles do. */
  let indicatorSeries = [];
  /* What `indicatorSeries` was built FROM. The structure is rebuilt only
   * when this changes, so the once-a-minute redraw costs a string compare. */
  let indicatorSignature = null;

  /* Pane 0 is price, pane 1 is the stochastic. Oscillators from the menu take
   * 2 upward, in the order they were added. */
  const FIRST_MENU_PANE = 2;

  function indicatorSpecs() {
    const stored = Array.isArray(prefs.indicators) ? prefs.indicators : [];
    /* Every entry re-validated on the way out, not just on the way in: this
     * comes from localStorage, which can hold anything a previous version
     * wrote, and a bad id here would throw inside the draw path. */
    return stored
      .map((item) => {
        const entry = item && DashboardIndicators.byId(item.id);
        if (!entry) return null;
        return {
          uid: String(item.uid || item.id),
          id: entry.id,
          entry,
          params: DashboardIndicators.normalizeParams(entry, item.params),
        };
      })
      .filter(Boolean);
  }

  /* Tear every menu indicator down, panes included. Panes are removed from the
   * highest index DOWN: the library re-indexes on removal, so going upward
   * would delete the wrong ones. */
  function clearIndicatorSeries() {
    if (!chart) return;
    for (const item of indicatorSeries) {
      for (const handle of item.handles) {
        try {
          chart.removeSeries(handle);
        } catch (error) {
          /* Already gone with its pane; nothing to undo. */
        }
      }
    }
    indicatorSeries = [];
    try {
      const panes = chart.panes();
      for (let index = panes.length - 1; index >= FIRST_MENU_PANE; index -= 1) {
        chart.removePane(index);
      }
    } catch (error) {
      /* An older build without pane removal still draws; it just keeps the
       * empty pane until the next full redraw. */
    }
  }

  /* Build the series the active set needs. Pane indices are assigned here and
   * nowhere else, so removing one from the middle cannot leave a hole. */
  /* Rebuild only when the set has actually changed -- or when the chart did
   * not exist last time it was asked, which is the state the page boots in:
   * `ensureChart` runs inside the first draw, so at startup there is nothing
   * to hang a series on and the attempt has to be repeated from the draw path.
   * Driving it from there rather than from boot alone is also what makes an
   * add, a removal and an instrument switch all converge on one code path. */
  function syncIndicatorSeries() {
    if (!chart) return;
    const signature = JSON.stringify(
      indicatorSpecs().map((spec) => [spec.uid, spec.id, spec.params]),
    );
    if (signature === indicatorSignature) return;
    rebuildIndicatorSeries();
    indicatorSignature = signature;
  }

  function rebuildIndicatorSeries() {
    if (!chart) return;
    clearIndicatorSeries();
    let pane = FIRST_MENU_PANE;
    for (const spec of indicatorSpecs()) {
      const own = spec.entry.kind === "pane";
      const paneIndex = own ? pane : 0;
      if (own) pane += 1;
      const lines = spec.entry.compute([], spec.params);
      const handles = lines.map((line) =>
        chart.addSeries(
          LightweightCharts.LineSeries,
          {
            color: line.color,
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: true,
            title: line.title,
          },
          paneIndex,
        ),
      );
      /* Reference levels -- RSI's 70/30, MACD's zero, ADX's 25 -- hung off the
       * first line so they come and go with it. */
      if (own && spec.entry.guides && handles.length) {
        for (const level of spec.entry.guides) {
          handles[0].createPriceLine({
            price: level,
            color: "#3a4150",
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
          });
        }
      }
      indicatorSeries.push({ uid: spec.uid, spec, handles, paneIndex: own ? paneIndex : null });
    }
    sizeIndicatorPanes();
  }

  function sizeIndicatorPanes() {
    try {
      const panes = chart.panes();
      for (let index = FIRST_MENU_PANE; index < panes.length; index += 1) {
        panes[index].setHeight(STOCH_PANE_HEIGHT);
      }
    } catch (error) {
      /* Heights are cosmetic; the library's own default still draws. */
    }
  }

  /* Data only. Called from the single draw path with the merged bars, so an
   * indicator covers history and the live window alike. */
  function refreshIndicators(bars) {
    for (const item of indicatorSeries) {
      let lines;
      try {
        lines = item.spec.entry.compute(bars, item.spec.params);
      } catch (error) {
        /* One broken indicator costs its own line and nothing else -- the same
         * rule the server-side indicators follow. */
        lines = [];
      }
      item.handles.forEach((handle, index) => {
        const line = lines[index];
        handle.setData(line ? pointsFrom(bars, line.values) : []);
      });
    }
  }

  /* ------------------------------------------------------- the menu itself */
  function renderIndicatorParams() {
    const entry = DashboardIndicators.byId(nodes.indPicker.value);
    if (!entry) {
      nodes.indParams.innerHTML = "";
      return;
    }
    nodes.indParams.innerHTML = entry.params
      .map(
        (spec) => `<label>${text(spec.label)}
          <input type="number" data-param="${text(spec.name)}"
                 value="${spec.value}" min="${spec.min}" max="${spec.max}"
                 step="${spec.step || 1}"></label>`,
      )
      .join("");
  }

  function renderIndicatorList() {
    const specs = indicatorSpecs();
    nodes.indCount.textContent = specs.length ? String(specs.length) : "";
    nodes.indActive.innerHTML = specs
      .map((spec) => {
        const lines = spec.entry.compute([], spec.params);
        const colour = (lines[0] || {}).color || "var(--muted)";
        const title = (lines[0] || {}).title || spec.entry.label;
        return `<li>
          <span class="swatch" style="background:${text(colour)}"></span>
          <span class="name">${text(title)}</span>
          <button type="button" class="drop" data-uid="${text(spec.uid)}"
                  title="Remove">&times;</button>
        </li>`;
      })
      .join("");
  }

  /* One place that writes the preference and redraws, so the three entry
   * points -- add, remove, and a corrupt value repaired on load -- cannot
   * drift apart. */
  function commitIndicators(specs) {
    prefs.indicators = specs.map((spec) => ({
      uid: spec.uid,
      id: spec.id,
      params: spec.params,
    }));
    savePrefs();
    renderIndicatorList();
    syncIndicatorSeries();
    if (lastMergedBars.length) refreshIndicators(lastMergedBars);
  }

  function addIndicator() {
    const entry = DashboardIndicators.byId(nodes.indPicker.value);
    if (!entry) return;
    const raw = {};
    for (const input of nodes.indParams.querySelectorAll("input[data-param]")) {
      raw[input.dataset.param] = input.value;
    }
    const specs = indicatorSpecs();
    specs.push({
      /* Unique per instance so two EMAs of different lengths are two rows,
       * not one overwriting the other. */
      uid: `${entry.id}-${Date.now()}-${specs.length}`,
      id: entry.id,
      entry,
      params: DashboardIndicators.normalizeParams(entry, raw),
    });
    commitIndicators(specs);
  }

  function removeIndicator(uid) {
    commitIndicators(indicatorSpecs().filter((spec) => spec.uid !== uid));
  }

  function setIndicatorMenu(open) {
    nodes.indMenuPanel.hidden = !open;
    nodes.indMenuButton.setAttribute("aria-expanded", String(open));
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
    /* Captured BEFORE the await, and every write below goes to this view --
     * not `view()`. The operator can switch instrument while the request is in
     * the air, and a reply filed under whichever instrument happens to be on
     * screen when it lands would draw one index's pivots over the other's
     * candles. */
    const instrument = prefs.inst;
    const self = instrumentViews[instrument];
    if (!self || self.cprLadders || self.cprLaddersInFlight) return;
    self.cprLaddersInFlight = true;
    try {
      const response = await fetch(
        `/api/history?inst=${encodeURIComponent(instrument)}&tf=cpr&page=0`,
        { cache: "no-store" },
      );
      if (!response.ok) return;          /* no history: today's band still draws */
      self.cprLadders = await response.json();
      if (instrument === prefs.inst && self.chartPayload) applyTimeframe();
    } catch (error) {
      /* Without them the chart shows today's band alone, which is what it did
       * before per-day CPR existed. The next redraw tries again. */
    } finally {
      self.cprLaddersInFlight = false;
    }
  }

  function historyCut(liveBars) {
    const history = view().historyBars[prefs.tf] || [];
    if (!liveBars.length) return history.length;
    const firstLive = liveBars[0].time;
    let cut = history.length;
    while (cut > 0 && history[cut - 1].time >= firstLive) cut -= 1;
    return cut;
  }

  function mergedBars(liveBars, cut) {
    const history = view().historyBars[prefs.tf] || [];
    if (!history.length) return liveBars;
    return history.slice(0, cut).concat(liveBars);
  }

  /* One indicator column over the merged series: the stored part where history
   * carried it, then the live part. */
  function mergedColumn(name, liveValues, cut) {
    const history = view().historyBars[prefs.tf] || [];
    const stored =
      (view().historyCols[prefs.tf] || {})[name]
      || new Array(history.length).fill(null);
    return stored.slice(0, cut).concat(liveValues || []);
  }

  /* Today, as ONE daily candle, folded out of the live minute bars.
   *
   * The runner's store holds minutes, so the Daily timeframe would otherwise
   * stop at yesterday -- the history CSV is only as fresh as the last download.
   * Stamped at 09:15 of the session exactly as the server stamps its own daily
   * bars, so today lands on the same grid instead of a slot of its own. */
  function dailyLiveBars() {
    const payload = view().chartPayload;
    const minute = payload && payload.timeframes && payload.timeframes["1"];
    const bars = (minute && minute.bars) || [];
    if (!bars.length) return [];
    const lastDay = Math.floor(bars[bars.length - 1].time / 86400);
    let start = bars.length - 1;
    while (start > 0 && Math.floor(bars[start - 1].time / 86400) === lastDay) start -= 1;
    const session = bars.slice(start);
    const sessionOpen = lastDay * 86400 + 9 * 3600 + 15 * 60;

    if (session[0].time > sessionOpen) {
      /* The live window does not reach the open -- the runner was started
       * mid-session, which this system explicitly supports. Its open, high and
       * low then describe PART of the day while being stamped as the whole of
       * it, and because the stamp collides with history's own bar for today the
       * partial one would REPLACE the complete one rather than lose to it.
       * History's bar wins when there is one; when there is not, a partial bar
       * still beats today missing from the timeframe entirely. */
      const stored = view().historyBars["D"] || [];
      if (stored.length && stored[stored.length - 1].time === sessionOpen) return [];
    }

    let high = session[0].high;
    let low = session[0].low;
    for (const bar of session) {
      if (bar.high > high) high = bar.high;
      if (bar.low < low) low = bar.low;
    }
    return [{
      time: sessionOpen,
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
  /* The day ladder, with the sessions the CSV has not caught up with filled in.
   *
   * The stored ladder is built once from the history CSV; the candles beside it
   * come from the runner's store, which is seeded with several days of REST
   * history. So by the end of a week the chart draws sessions the ladder has
   * never heard of -- and a level with no successor used to run flat straight
   * across them, showing one day's pivot over another day's session.
   *
   * The stored band WINS on a date both hold: the CSV's sessions are complete
   * downloads, while the store's oldest is whatever its window happens to
   * reach back to. */
  function dayLadder() {
    const self = view();
    const stored = (self.cprLadders && self.cprLadders.day) || [];
    const live = (self.chartPayload && self.chartPayload.cpr_days) || [];
    if (!live.length) return stored;
    const known = new Set(stored.map((band) => band.date));
    const filled = stored.concat(live.filter((band) => !known.has(band.date)));
    return filled.sort((a, b) => a.from - b.from);
  }

  function cprBandsForView(bars) {
    if (!bars.length) return [];
    const monthly = prefs.tf === "D";
    const ladders = view().cprLadders;
    const ladder = monthly ? ((ladders && ladders.month) || []) : dayLadder();
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
    const live = view().chartPayload && view().chartPayload.cpr;
    if (!monthly && live && live.available) {
      bands.push({ from: todayStart, to: rightEdge, levels: live });
    }
    return bands;
  }

  /* The month band the Daily caption describes, plus the month its numbers
   * came FROM -- which is the month before it, and the thing worth naming. */
  function newestMonthBand() {
    const ladder = (view().cprLadders && view().cprLadders.month) || [];
    if (!ladder.length) return { levels: null };
    const band = ladder[ladder.length - 1];
    const previous = ladder.length > 1 ? ladder[ladder.length - 2].month : null;
    return { levels: band.levels, previous: previous || "" };
  }

  /* The index of the first bar at or after `time`, or -1 if there is none.
   *
   * Every CPR point is placed on a bar time that already exists, which is what
   * keeps the levels from adding COLUMNS to the shared time scale. Two things
   * used to add them: a band's `to` of 15:29 is not a 5-minute bar (buckets end
   * 15:25), and a band covering days the loaded candles do not have -- which
   * happens whenever the history CSV is older than the live window -- sat in
   * empty space entirely. */
  function firstBarAtOrAfter(bars, time) {
    let low = 0;
    let high = bars.length - 1;
    let found = -1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (bars[mid].time >= time) {
        found = mid;
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return found;
  }

  /* Which session a bar time belongs to, and which month. The scale is IST
   * wall-clock labelled UTC, so a plain division by a day lands on the date. */
  function dayKey(time) {
    return Math.floor(time / 86400);
  }

  function monthKey(time) {
    const when = new Date(time * 1000);
    return when.getUTCFullYear() * 12 + when.getUTCMonth();
  }

  /* Split the bands into RUNS of periods that sit next to each other.
   *
   * A band's `from`/`to` covers its own session and nothing else -- the server
   * guarantees it, and a test there says so in as many words. The renderer has
   * to honour that, and a stepped line on its own cannot: it runs flat from one
   * point to the next, so a band with no successor stretches across every day
   * between them. That is how 2026-09-16's levels came to be drawn over the
   * 17th, 18th and 21st, which the stored ladder had no bands for.
   *
   * A run therefore ends where a whole PERIOD -- a session on the minute
   * timeframes, a month on Daily -- lies between one band and the next, and
   * each run is drawn as a series of its own, which simply stops.
   *
   * By period, and deliberately not "any bar in between": the store's copy of
   * a day can hold a bar the CSV's copy does not (a 15:30 print, or anything
   * before the open), and the two ladders meet exactly there. Splitting on
   * that would put a second set of price labels down the axis for a seam
   * nobody can see.
   *
   * Pure: no chart, no prefs, no levels. `bars` need only carry `time`,
   * ascending. Returns `[{ anchors: [{ index, band }], endIndex }]`, where
   * `endIndex` is the last loaded bar the run's final band owns. */
  function cprRuns(bands, bars, keyOf) {
    const runs = [];
    if (!bands.length || !bars.length) return runs;

    /* No loaded candle inside a band -- a stretch of the ladder the chart has
     * nothing to draw against. It anchors no point, so it neither draws nor
     * splits a run. */
    const anchors = [];
    for (const band of bands) {
      const index = firstBarAtOrAfter(bars, band.from);
      if (index === -1 || bars[index].time > band.to) continue;
      anchors.push({ index, band });
    }

    let current = null;
    for (let position = 0; position < anchors.length; position += 1) {
      const anchor = anchors[position];
      const next = anchors[position + 1];
      const key = keyOf(bars[anchor.index].time);

      /* Everything up to the next band, or to the right edge -- but never
       * past this band's own period. */
      let end = next ? next.index - 1 : bars.length - 1;
      const broken = !next || keyOf(bars[end].time) !== key;
      if (broken) {
        while (end > anchor.index && keyOf(bars[end].time) !== key) end -= 1;
      }

      if (current) {
        current.anchors.push(anchor);
        current.endIndex = end;
      } else {
        current = { anchors: [anchor], endIndex: end };
        runs.push(current);
      }
      if (broken) current = null;
    }
    return runs;
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
     * change it.
     *
     * The BARS belong in it too, now that the runs are cut against them: a
     * scroll that loads candles into a stretch no band covers splits a run in
     * two without touching the band set at all. */
    const newest = bands.length ? bands[bands.length - 1] : null;
    const signature = JSON.stringify([
      /* The INSTRUMENT belongs here as much as any group pref. Two indices'
       * ladders can agree on band count, extent and even a pivot while
       * describing completely different prices -- and an unchanged signature
       * skips the redraw, leaving one index's levels drawn over the other's
       * candles. */
      prefs.inst,
      prefs.tf === "D" ? "month" : "day",
      bands.length,
      bands.length ? bands[0].from : null,
      newest ? newest.to : null,
      newest ? newest.levels.pivot : null,
      bars.length,
      bars.length ? bars[0].time : null,
      bars.length ? bars[bars.length - 1].time : null,
      prefs.cpr, prefs.cprPD, prefs.cprRS1, prefs.cprRS3,
    ]);
    if (signature === cprSignature) return;
    cprSignature = signature;

    for (const handle of cprSeries) chart.removeSeries(handle);
    cprSeries = [];
    if (!bands.length || !prefs.cpr) return;

    /* ONE point per band, at the bar its day opens on, drawn as a STEP, and
     * one series per RUN of adjacent days.
     *
     * There is deliberately no whitespace point here. A whitespace item cannot
     * break a line in lightweight-charts 5.2.1: the data layer runs
     * `rows.filter(hasValue)` before the series ever sees them, so the stroke
     * is one continuous path and the break never happened -- which is what drew
     * a diagonal from yesterday's level to today's. `LineType.WithSteps` below
     * is what separates one day from the NEXT: the renderer moves horizontally
     * at the previous value and only then vertically, so each level runs flat
     * across its session and steps at the next open. A separate series per run
     * is what stops a level reaching across days that have no band at all. */
    const runs = cprRuns(bands, bars, prefs.tf === "D" ? monthKey : dayKey);
    if (!runs.length) return;

    for (const level of CPR_LEVELS) {
      const on = level.group === "core"
        || (level.group === "pd" && prefs.cprPD)
        || (level.group === "rs1" && prefs.cprRS1)
        || (level.group === "rs3" && prefs.cprRS3);
      if (!on) continue;

      runs.forEach((run, runIndex) => {
        const points = [];
        for (const anchor of run.anchors) {
          const price = anchor.band.levels[level.key];
          if (price === null || price === undefined) continue;
          points.push({ time: bars[anchor.index].time, value: price });
        }
        if (!points.length) return;
        const newestRun = runIndex === runs.length - 1;

        /* Close the run on the last bar its final band owns. A step ends at
         * its last point, so without this the freshest level would stop at
         * its own open -- and the run would not stop anywhere at all. */
        const end = bars[run.endIndex];
        const last = points[points.length - 1];
        if (end.time > last.time) points.push({ time: end.time, value: last.value });

        const series = chart.addSeries(LightweightCharts.LineSeries, {
          color: level.color,
          lineWidth: 1,
          /* `case 1` of the library's line renderer is `lineTo(x, previousY)`
           * then `lineTo(x, y)` -- horizontal, then vertical -- so a diagonal
           * is not reachable in this branch. */
          lineType: LightweightCharts.LineType.WithSteps,
          priceLineVisible: false,
          /* The last value still gets an axis label, and `title` puts the
           * level's NAME beside it -- so the "(chart)" caveat still travels
           * with a screenshot of just the chart, as it did when these were
           * price lines and none of the page's other captions would.
           *
           * Only on the NEWEST run: a label per run would stack one column of
           * the price axis per gap in the ladder, which is thirteen more
           * labels each time and none of them current.
           *
           * BOTH have to go, not just the value. The library draws two axis
           * views per series, and the second is gated on
           * `"" !== title || seriesLastValueMode === 0` -- `lastValueVisible`
           * is nowhere in that test. An older run with a title kept therefore
           * still printed "R1 (chart)" on the axis, with no number beside it. */
          lastValueVisible: newestRun,
          title: newestRun ? `${level.title} (chart)` : "",
        }, 0);
        series.setData(points);
        cprSeries.push(series);
      });
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
    const self = view();
    const chartPayload = self.chartPayload;
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
    lastMergedBars = bars;

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

    if (renderedTimeframe !== prefs.tf || renderedInstrument !== prefs.inst) {
      renderedTimeframe = prefs.tf;
      /* An instrument switch refits too, and must: BankNIFTY trades near
       * 55,000 while NIFTY is near 24,000, so the incoming candles sit
       * entirely outside the outgoing price range and the pane would look
       * empty but for the axis. */
      renderedInstrument = prefs.inst;
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
    /* The payload names its own instrument rather than the page assuming the
     * reply matches the request it made. */
    const name = chartPayload.instrument || prefs.inst;
    nodes.chartTitle.textContent = daily
      ? `${name} · daily · CPR from the prior MONTH (chart only)`
      : `${name} · ${minutes} minute${minutes === 1 ? "" : "s"} · CPR from prior session `
        + `${window} (chart only)`;
    renderCprLines(bars);
    syncIndicatorSeries();
    refreshIndicators(bars);
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
    if (!(self.historyBars[prefs.tf] || []).length && !self.historyExhausted[prefs.tf]) {
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
    /* Both captured before the await, and every write below goes through
     * `self`. A switch while the page is in the air must not splice one
     * index's bars into the other's series. */
    const instrument = prefs.inst;
    const self = instrumentViews[instrument];
    if (!self || historyInFlight || self.historyExhausted[timeframe]) return;
    historyInFlight = true;
    try {
      const page = self.historyNextPage[timeframe];
      const response = await fetch(
        `/api/history?inst=${encodeURIComponent(instrument)}`
        + `&tf=${encodeURIComponent(timeframe)}&page=${page}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        /* A 404 has two meanings and they must not be confused. Once a page has
         * loaded it means the operator has reached the start of history, and
         * asking again is pointless. BEFORE that it means the server is still
         * building -- measured at 29.5 seconds on the five-year file -- and
         * treating it as exhausted would disable scroll-back for the whole
         * session over a few seconds of startup. */
        if (!self.historyFirstRefusalAt) self.historyFirstRefusalAt = Date.now();
        const loaded = (self.historyBars[timeframe] || []).length;
        const waitedOutTheBuild =
          Date.now() - self.historyFirstRefusalAt > HISTORY_GIVE_UP_MS;
        /* Bounded by TIME, not by a retry count: `applyTimeframe` can fire
         * several times in the first second -- boot, a checkbox, the first
         * chart render -- and a count would be spent before the server has
         * finished building. Five minutes comfortably covers the 29.5s measured
         * on the five-year file, and then stops asking every minute forever for
         * history that is simply not there. */
        if (loaded || waitedOutTheBuild) self.historyExhausted[timeframe] = true;
        return;
      }
      const doc = await response.json();
      const bars = doc.bars || [];
      if (!bars.length) {
        self.historyExhausted[timeframe] = true;
        return;
      }
      self.historyBars[timeframe] = bars.concat(self.historyBars[timeframe] || []);
      for (const name of ["stoch_k", "stoch_d"]) {
        /* A page without the column still contributes its LENGTH, as nulls, so
         * the arrays stay index-aligned with the bars either way. */
        const incoming = doc[name] || new Array(bars.length).fill(null);
        const columns = self.historyCols[timeframe];
        columns[name] = incoming.concat(columns[name] || []);
      }
      self.historyNextPage[timeframe] = page + 1;
      if (doc.pages && self.historyNextPage[timeframe] >= doc.pages) {
        self.historyExhausted[timeframe] = true;
      }
      /* Only redraw if the operator has not switched timeframe OR instrument
       * while this was in the air; otherwise the bars belong to a chart nobody
       * is looking at. */
      const onScreen = prefs.tf === timeframe && prefs.inst === instrument;
      const added = onScreen ? redrawKeepingViewport() : 0;

      if (!added && !self.historyExhausted[timeframe]) {
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

  /* Empty every series. Used when switching to an instrument the runner is
   * not publishing yet: leaving the outgoing candles on screen under the new
   * instrument's name would be the chart stating something untrue, which is
   * the one thing this dashboard does not do. */
  function clearChartSeries() {
    if (!candleSeries) return;
    candleSeries.setData([]);
    vwapSeries.setData([]);
    kSeries.setData([]);
    dSeries.setData([]);
    for (const handle of cprSeries) chart.removeSeries(handle);
    cprSeries = [];
    for (const item of indicatorSeries) {
      for (const handle of item.handles) handle.setData([]);
    }
    /* So the next real draw rebuilds rather than matching a stale memo. */
    cprSignature = null;
    renderedBarCount = 0;
    lastMergedBars = [];
    nodes.cprProvenance.hidden = true;
  }

  /* Which instruments the runner is actually publishing. The state document's
   * chart block is keyed by instrument, so its KEYS are the answer -- the page
   * never has to be told separately, and cannot disagree. */
  function applyInstrumentAvailability(offered) {
    for (const [button, key] of [
      [nodes.instNifty, "NIFTY"], [nodes.instBankNifty, "BANKNIFTY"],
    ]) {
      const on = offered.includes(key);
      button.disabled = !on;
      button.classList.toggle("disabled", !on);
      if (!on) {
        button.title = `Not being published. Set DASHBOARD_BANKNIFTY_ENABLED=true `
          + `in Dependencies/.env and restart to chart ${key}.`;
      }
    }
    /* A stored preference for an instrument this run is not publishing would
     * otherwise leave the operator on a permanently empty chart. */
    if (!offered.length || offered.includes(prefs.inst)) return;
    setInstrument(offered[0]);
  }

  function setInstrument(value) {
    if (!INSTRUMENTS.includes(value)) return;
    prefs.inst = value;
    savePrefs();
    for (const [button, key] of [
      [nodes.instNifty, "NIFTY"], [nodes.instBankNifty, "BANKNIFTY"],
    ]) {
      button.classList.toggle("active", value === key);
      button.setAttribute("aria-pressed", String(value === key));
    }
    /* Whatever this instrument already holds is redrawn from memory. When it
     * holds nothing yet, the chart keeps the outgoing series on screen until
     * the first payload lands rather than blanking -- and `renderChart` asks
     * for it on the very next poll, because this view's `seriesVersion` is
     * still -1 and cannot match anything the server reports. */
    if (view().chartPayload) applyTimeframe();
    else {
      clearChartSeries();
      nodes.chartEmpty.hidden = false;
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
    /* `doc.chart` is keyed by instrument. Only the one on screen is drawn, so
     * an instrument the runner is not publishing costs nothing at all. */
    const charts = doc.chart || {};
    applyInstrumentAvailability(Object.keys(charts));
    const instrument = prefs.inst;
    const self = view();
    const info = charts[instrument] || {};
    const hasData = info.state === "OK" && info.last_bar;
    nodes.chartEmpty.hidden = hasData;
    if (!hasData) return;

    ensureChart();
    if (!candleSeries) return;

    /* The arrays are refetched only when a new MINUTE closed -- roughly 375
     * times a session rather than once per poll. In between, the forming
     * candle is updated in place, which is what makes the chart move
     * tick-by-tick under MARKET_DATA_SOURCE=WEBSOCKET for a few hundred bytes. */
    if (info.series_version !== self.seriesVersion && !chartFetchInFlight) {
      /* Guarded: this runs from `render`, which the poll loop calls WITHOUT
       * awaiting, so two polls landing close together could otherwise put two
       * requests in the air for the same version. */
      chartFetchInFlight = true;
      try {
        const response = await fetch(
          `/api/chart?inst=${encodeURIComponent(instrument)}`,
          { cache: "no-store" },
        );
        if (response.ok) {
          /* Filed under the instrument CAPTURED above, not whichever is on
           * screen when the reply lands. */
          self.chartPayload = await response.json();
          self.seriesVersion = info.series_version;
          /* Keeping the viewport, not refitting: by now the operator may have
           * scrolled back through years of history, and a minute closing must
           * not drag them back to today. */
          if (instrument === prefs.inst) redrawKeepingViewport();
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
  /* The indicator menu. Its rows are built from the preference rather than
   * wired one by one, so the handlers are DELEGATED -- a row that does not
   * exist yet still works, and nothing here needs an element id of its own. */
  nodes.indPicker.innerHTML = DashboardIndicators.INDICATORS.map(
    (entry) => `<option value="${text(entry.id)}">${text(entry.label)}`
      + `${entry.kind === "pane" ? " (own pane)" : ""}</option>`,
  ).join("");
  nodes.indPicker.addEventListener("change", renderIndicatorParams);
  nodes.indAdd.addEventListener("click", addIndicator);
  nodes.indActive.addEventListener("click", (event) => {
    const button = event.target.closest("button.drop");
    if (button) removeIndicator(button.dataset.uid);
  });
  nodes.indMenuButton.addEventListener("click", () => {
    setIndicatorMenu(nodes.indMenuPanel.hidden);
  });
  /* Click-away and Escape, so the panel never strands itself over the chart.
   * Bound on the document because the panel is absolutely positioned and the
   * chart beneath it swallows its own pointer events. */
  document.addEventListener("click", (event) => {
    if (nodes.indMenuPanel.hidden) return;
    if (!event.target.closest(".menu")) setIndicatorMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setIndicatorMenu(false);
  });
  renderIndicatorParams();
  renderIndicatorList();

  nodes.instNifty.addEventListener("click", () => setInstrument("NIFTY"));
  nodes.instBankNifty.addEventListener("click", () => setInstrument("BANKNIFTY"));
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
      /* Only the CPR series change, so only those are redrawn. Going through
       * `applyTimeframe` here re-uploaded every candle, VWAP point and
       * stochastic point -- hundreds of thousands of bars once the operator has
       * scrolled back -- for a toggle that touches none of them. */
      else if (view().chartPayload) renderCprLines(lastMergedBars);
    });
  }
  /* Instrument FIRST: `setTimeframe` draws, and drawing reads `view()`. */
  setInstrument(INSTRUMENTS.includes(prefs.inst) ? prefs.inst : INSTRUMENTS[0]);
  setTimeframe(["1", "5", "D"].includes(prefs.tf) ? prefs.tf : "1");

  document.addEventListener("visibilitychange", () => {
    /* Bring the NEXT poll forward rather than starting a second loop. */
    if (document.visibilityState === "visible") schedulePoll(0);
  });

  schedulePoll(0);
})();
