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
  let seriesVersion = -1;

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
  function ensureChart() {
    if (chart || typeof LightweightCharts === "undefined") return;
    chart = LightweightCharts.createChart(nodes.chart, {
      layout: {
        background: { color: "#1b1f26" },
        textColor: "#8b94a3",
        /* Left enabled deliberately: the vendored library is Apache-2.0 and
         * this is its attribution. See vendor/NOTICE-lightweight-charts.md. */
        attributionLogo: true,
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
    });
  }

  async function renderChart(doc) {
    const info = doc.chart || {};
    const hasData = info.state === "OK" && info.last_bar;
    nodes.chartEmpty.hidden = hasData;
    if (!hasData) return;

    ensureChart();
    if (!candleSeries) return;

    /* The full candle array is refetched only when a new MINUTE closed --
     * roughly 375 times a session rather than once per poll. In between, the
     * forming candle is updated in place, which is what makes the chart move
     * tick-by-tick under MARKET_DATA_SOURCE=WEBSOCKET for ~120 bytes a poll. */
    if (info.series_version !== seriesVersion) {
      try {
        const response = await fetch("/api/chart", { cache: "no-store" });
        if (response.ok) {
          const payload = await response.json();
          candleSeries.setData(payload.bars || []);
          seriesVersion = info.series_version;
        }
      } catch (error) {
        /* Leave the previous series on screen; the next poll retries. */
        return;
      }
    }
    candleSeries.update(info.last_bar);
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

  document.addEventListener("visibilitychange", () => {
    /* Bring the NEXT poll forward rather than starting a second loop. */
    if (document.visibilityState === "visible") schedulePoll(0);
  });

  schedulePoll(0);
})();
