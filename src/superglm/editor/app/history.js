import { escapeHTML } from "./format.js";

/** @typedef {import('./api/contracts.js').TimelineEntry} TimelineEntry */

/**
 * Render the session's timeline: every edit and structural step in order, a
 * marker at the current position, and what Redo would put back, muted, after it.
 * @param {TimelineEntry[]|undefined} timeline
 * @param {HTMLElement|null} node
 */
export function renderHistory(timeline, node) {
  if (!node) return;
  const entries = Array.isArray(timeline) ? timeline : [];
  if (entries.length <= 1) {
    node.innerHTML = `<div class="history-empty">Nothing yet.</div>`;
    return;
  }
  node.innerHTML = `<ol class="history-list">${entries.map(historyItem).join("")}</ol>`;
  node.querySelector(".history-now")?.scrollIntoView({ block: "nearest" });
}

/** @param {TimelineEntry} entry */
function historyItem(entry) {
  if (entry.kind === "marker") {
    return `<li class="history-now" role="separator" aria-label="Current position">now</li>`;
  }
  const meta = entry.kind === "edit" ? `<div class="history-meta">${editMeta(entry)}</div>` : "";
  return `<li class="history-item ${entry.kind}${entry.redo ? " redo" : ""}">
    ${entryMark(entry)}
    <div class="history-body">
      <div class="history-label">${escapeHTML(entry.label || "")}</div>
      ${meta}
    </div>
  </li>`;
}

/** @param {TimelineEntry} entry */
function entryMark(entry) {
  return entry.kind === "edit"
    ? `<code class="history-hash">${escapeHTML(entry.hash || "-------")}</code>`
    : `<span class="history-chip">step</span>`;
}

/** @param {TimelineEntry} entry */
function editMeta(entry) {
  const params = paramsLabel(entry.params);
  return `${Number(entry.n_points || 0)} points${params ? ` · ${escapeHTML(params)}` : ""}`;
}

/** @param {Record<string, unknown>|undefined} params */
function paramsLabel(params) {
  if (!params || typeof params !== "object") return "";
  return Object.entries(params)
    .slice(0, 3)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
}
