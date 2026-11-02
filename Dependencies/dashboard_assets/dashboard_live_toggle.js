/* The dashboard's per-strategy LIVE/PAPER toggle (ADR-0021).
 *
 * The ONLY file on this page that sends anything to the runner. It is kept
 * apart from dashboard.js on purpose: that file is read-only and a test keeps
 * it that way, and this one is small enough to review in a minute.
 *
 * Flow: on load, ask GET /api/live-mode whether the runner switched the toggle
 * on. If it did, remember the per-process token and mark the body so
 * dashboard.js draws a button in each strategy row. A click opens an in-page
 * dialog asking the operator to TYPE the strategy's name, in both directions,
 * then sends one POST. The runner only queues it; whether it was applied, is
 * still pending (the strategy holds a position) or was refused shows up in the
 * strategy table on the next poll.
 *
 * Why a <dialog> and not the browser's built-in prompt/alert boxes: a browser
 * can silently suppress those ("prevent this page from creating additional
 * dialogs"), and then a click would do nothing with no explanation.
 */
"use strict";

(function () {
  const PATH = "/api/live-mode";
  const el = (id) => document.getElementById(id);
  const dialog = el("live-toggle-dialog");
  const message = el("live-toggle-message");
  const input = el("live-toggle-input");
  const cancelButton = el("live-toggle-cancel");
  const confirmButton = el("live-toggle-confirm");
  let token = "";
  let sending = false;
  /* What the open dialog is asking about; null when it only shows a notice. */
  let pending = null;

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

  function ask(strategy, target) {
    pending = { strategy, target };
    message.textContent = confirmText(strategy, target);
    input.hidden = false;
    input.value = "";
    cancelButton.hidden = false;
    confirmButton.textContent = "Confirm";
    dialog.showModal();
    input.focus();
  }

  function notify(text) {
    pending = null;
    message.textContent = text;
    input.hidden = true;
    cancelButton.hidden = true;
    confirmButton.textContent = "OK";
    if (!dialog.open) dialog.showModal();
  }

  async function send(strategy, target, typed) {
    sending = true;
    try {
      const response = await fetch(PATH, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy, target, confirm: typed, token }),
      });
      if (response.status === 202) {
        dialog.close();
        return;
      }
      const reply = (await response.text()).trim();
      notify(`Not accepted (${response.status}): ${reply}`);
    } catch (error) {
      notify("Runner not reachable. Nothing was changed.");
    } finally {
      sending = false;
    }
  }

  function onConfirm() {
    if (pending === null) {
      dialog.close();
      return;
    }
    if (sending) return;
    const { strategy, target } = pending;
    const typed = input.value;
    if (typed !== strategy) {
      notify("The name did not match. Nothing was changed.");
      return;
    }
    send(strategy, target, typed);
  }

  cancelButton.addEventListener("click", () => dialog.close());
  confirmButton.addEventListener("click", onConfirm);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") onConfirm();
  });

  /* Delegated: the strategy table is redrawn every poll, so a listener on a
   * row's button would be thrown away a second later. */
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button.live-toggle");
    if (!button || !token || sending) return;
    ask(button.dataset.strategy, button.dataset.target);
  });

  init();
})();
