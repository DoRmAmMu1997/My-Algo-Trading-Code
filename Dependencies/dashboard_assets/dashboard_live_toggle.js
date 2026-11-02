/* The dashboard's per-strategy LIVE/PAPER toggle (ADR-0021).
 *
 * The ONLY file on this page that sends anything to the runner. It is kept
 * apart from dashboard.js on purpose: that file is read-only and a test keeps
 * it that way, and this one is small enough to review in a minute.
 *
 * Flow: on load, ask GET /api/live-mode whether the runner switched the toggle
 * on. If it did, remember the per-process token and mark the body so
 * dashboard.js draws a button in each strategy row. A click asks the operator
 * to TYPE the strategy's name, in both directions, then sends one POST. The
 * runner only queues it; whether it was applied, is still pending (the
 * strategy holds a position) or was refused shows up in the strategy table on
 * the next poll.
 */
"use strict";

(function () {
  const PATH = "/api/live-mode";
  let token = "";
  let sending = false;

  async function init() {
    try {
      const response = await fetch(PATH, { cache: "no-store" });
      if (!response.ok) return;
      const status = await response.json();
      if (status.enabled === true && typeof status.token === "string" && status.token) {
        token = status.token;
        document.body.dataset.liveToggle = "on";
      }
    } catch (error) {
      /* Runner not reachable: no buttons, which is the safe default. */
    }
  }

  function confirmText(strategy, target) {
    const lines = [`Switch ${strategy} to ${target}?`, ""];
    if (target === "LIVE") {
      lines.push("LIVE places REAL broker orders with real money.");
    }
    lines.push(
      "It takes effect from the strategy's next entry, once it is flat.",
      "",
      "Type the strategy name exactly to confirm:",
    );
    return lines.join("\n");
  }

  async function send(strategy, target) {
    if (sending || !token) return;
    const typed = window.prompt(confirmText(strategy, target));
    if (typed === null) return;
    if (typed !== strategy) {
      window.alert("The name did not match. Nothing was changed.");
      return;
    }
    sending = true;
    try {
      const response = await fetch(PATH, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy, target, confirm: typed, token }),
      });
      if (response.status !== 202) {
        const message = (await response.text()).trim();
        window.alert(`Not accepted (${response.status}): ${message}`);
      }
    } catch (error) {
      window.alert("Runner not reachable. Nothing was changed.");
    } finally {
      sending = false;
    }
  }

  /* Delegated: the strategy table is redrawn every poll, so a listener on a
   * row's button would be thrown away a second later. */
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button.live-toggle");
    if (!button) return;
    send(button.dataset.strategy, button.dataset.target);
  });

  init();
})();
