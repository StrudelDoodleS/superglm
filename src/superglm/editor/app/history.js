import { escapeHTML } from "./format.js";

export function renderHistory(history, node) {
  if (!node) return;
  const active = history && Array.isArray(history.active) ? history.active : [];
  const redo = history && Array.isArray(history.redo) ? history.redo : [];
  if (!active.length && !redo.length) {
    node.innerHTML = `<div class="history-empty">No edits yet.</div>`;
    return;
  }
  node.innerHTML = [
    historyList("Active edits", active),
    redo.length ? historyList("Redo stack", redo, true) : ""
  ].join("");
}

function historyList(title, records, muted = false) {
  const items = records.map((record) => historyItem(record, muted)).join("");
  return `<section class="history-section${muted ? " muted" : ""}">
    <h3>${escapeHTML(title)}</h3>
    <ol class="history-list">${items}</ol>
  </section>`;
}

function historyItem(record, muted) {
  const params = paramsLabel(record.params);
  return `<li class="history-item${record.is_head ? " head" : ""}${muted ? " muted" : ""}">
    <code class="history-hash">${escapeHTML(record.hash || "-------")}</code>
    <div class="history-body">
      <div><strong>${escapeHTML(record.operation || "edit")}</strong> · ${escapeHTML(record.term || "")}</div>
      <div class="history-meta">${Number(record.n_points || 0)} points${params ? ` · ${escapeHTML(params)}` : ""}</div>
    </div>
  </li>`;
}

function paramsLabel(params) {
  if (!params || typeof params !== "object") return "";
  return Object.entries(params)
    .slice(0, 3)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
}
