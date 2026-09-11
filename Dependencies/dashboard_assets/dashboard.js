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
  let cprLines = [];
  let cprSignature = null;
  /* Which timeframe the series currently hold. Switching changes both the bar
   * spacing and the span, so the visible range has to be refit -- but ONLY
   * then, never on the once-a-minute refresh, or the operator's zoom would be
   * yanked back every minute. */
  let renderedTimeframe = null;
  /* Set whenever the visible range needs refitting, cleared once a fit has
   * actually landed. See `requestFit`. */
  let pendingFit = false;

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
  const PREF_DEFAULTS = { tf: "1", cpr: true, cprRS1: false, cprRS3: false, vwap: true, stoch: true };

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

  /* CPR groups. `core` is on by default; the ladders are opt-in because
   * eleven horizontal lines over candles is not a chart, it is a net. */
  const CPR_LEVELS = [
    { key: "pivot", group: "core", title: "P",  color: "#e8b13a", dashed: false },
    { key: "bc",    group: "core", title: "BC", color: "#8b94a3", dashed: true },
    { key: "tc",    group: "core", title: "TC", color: "#8b94a3", dashed: true },
    { key: "r1",    group: "rs1",  title: "R1", color: "#ef5f5f", dashed: true },
    { key: "r2",    group: "rs1",  title: "R2", color: "#ef5f5f", dashed: true },
    { key: "s1",    group: "rs1",  title: "S1", color: "#35c46b", dashed: true },
    { key: "s2",    group: "rs1",  title: "S2", color: "#35c46b", dashed: true },
    { key: "r3",    group: "rs3",  title: "R3", color: "#8a3b3b", dashed: true },
    { key: "r4",    group: "rs3",  title: "R4", color: "#8a3b3b", dashed: true },
    { key: "s3",    group: "rs3",  title: "S3", color: "#2c6b45", dashed: true },
    { key: "s4",    group: "rs3",  title: "S4", color: "#2c6b45", dashed: true },
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

  function renderCprLines(cpr) {
    if (!candleSeries) return;
    const signature = JSON.stringify([
      cpr && cpr.available ? cpr.pivot : null, prefs.cpr, prefs.cprRS1, prefs.cprRS3,
    ]);
    if (signature === cprSignature) return;
    cprSignature = signature;

    for (const handle of cprLines) candleSeries.removePriceLine(handle);
    cprLines = [];
    if (!cpr || !cpr.available || !prefs.cpr) return;

    for (const level of CPR_LEVELS) {
      const on = level.group === "core"
        || (level.group === "rs1" && prefs.cprRS1)
        || (level.group === "rs3" && prefs.cprRS3);
      const price = cpr[level.key];
      if (!on || price === null || price === undefined) continue;
      cprLines.push(candleSeries.createPriceLine({
        price,
        color: level.color,
        lineWidth: 1,
        lineStyle: level.dashed ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        /* The "(chart)" suffix travels with a screenshot of just the chart,
         * where none of the page's other captions would. */
        title: `${level.title} (chart)`,
      }));
    }
  }

  function renderCprProvenance(cpr) {
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
    const block = chartPayload.timeframes[prefs.tf] || chartPayload.timeframes["1"];
    if (!block) return;

    candleSeries.setData(block.bars || []);
    vwapSeries.setData(pointsFrom(block.bars || [], block.vwap));
    kSeries.setData(pointsFrom(block.bars || [], block.stoch_k));
    dSeries.setData(pointsFrom(block.bars || [], block.stoch_d));

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
    nodes.chartTitle.textContent =
      `NIFTY · ${minutes} minute${minutes === 1 ? "" : "s"} · CPR from prior session `
      + `${(chartPayload.cpr && chartPayload.cpr.window) || "09:15-15:15"} (chart only)`;
    renderCprLines(chartPayload.cpr);
    renderCprProvenance(chartPayload.cpr);

    const vwapMeta = (chartPayload.indicators || {}).vwap || {};
    nodes.vwapLabel.title = vwapMeta.note || "";
    nodes.vwapFootnote.hidden = vwapMeta.is_proxy === false;
  }

  function setTimeframe(value) {
    prefs.tf = value;
    savePrefs();
    nodes.tf1.classList.toggle("active", value === "1");
    nodes.tf5.classList.toggle("active", value === "5");
    nodes.tf1.setAttribute("aria-pressed", String(value === "1"));
    nodes.tf5.setAttribute("aria-pressed", String(value === "5"));
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
    if (info.series_version !== seriesVersion) {
      try {
        const response = await fetch("/api/chart", { cache: "no-store" });
        if (response.ok) {
          chartPayload = await response.json();
          seriesVersion = info.series_version;
          applyTimeframe();
        }
      } catch (error) {
        /* Leave the previous series on screen; the next poll retries. */
        return;
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
    if (document.visibilityState === "hidden") delay = Math.max(delay, HIDDEN_TAB_SECONDS);
    schedulePoll(delay);
  }

  /* Controls. Each checkbox flips visibility rather than re-setting data, so
   * toggling never costs a redraw of the series it keeps. */
  nodes.tf1.addEventListener("click", () => setTimeframe("1"));
  nodes.tf5.addEventListener("click", () => setTimeframe("5"));

  for (const [id, key] of [
    ["ind-cpr", "cpr"], ["ind-cpr-rs1", "cprRS1"], ["ind-cpr-rs3", "cprRS3"],
    ["ind-vwap", "vwap"], ["ind-stoch", "stoch"],
  ]) {
    const box = el(id);
    box.checked = Boolean(prefs[key]);
    box.addEventListener("change", () => {
      prefs[key] = box.checked;
      savePrefs();
      if (key === "vwap" && vwapSeries) vwapSeries.applyOptions({ visible: prefs.vwap });
      else if (key === "stoch") applyStochVisibility();
      else if (chartPayload) renderCprLines(chartPayload.cpr);
    });
  }
  setTimeframe(prefs.tf === "5" ? "5" : "1");

  document.addEventListener("visibilitychange", () => {
    /* Bring the NEXT poll forward rather than starting a second loop. */
    if (document.visibilityState === "visible") schedulePoll(0);
  });

  schedulePoll(0);
})();
