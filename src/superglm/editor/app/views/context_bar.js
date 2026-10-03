// @ts-check

import { fmt, fmtPercent } from "../format.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../api/contracts.js').TermReference} TermReference */

/** @type {Readonly<Record<TermReference['policy'], string>>} */
const REFERENCE_POLICY = Object.freeze({
  most_exposed: "most exposed",
  first: "first",
  pinned: "pinned",
  kept: "kept",
});

/** The status line's note while changes wait and nothing is selected. */
const FROM_LAST_REFIT = "the curve and metrics are from the last refit";

/** @param {number} count */
export function waitingLabel(count) {
  return `${count} ${count === 1 ? "change" : "changes"} waiting for refit`;
}

/**
 * @param {{nameNode?:HTMLElement|null, kindNode:HTMLElement, edfNode:HTMLElement, referenceNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string, pendingCount?:number}} context
 */
export function renderContextBar(
  { nameNode = null, kindNode, edfNode, referenceNode, statusNode },
  { name, term, selectionSize, note = "", pendingCount = 0 },
) {
  const kind = term.term_type || term.kind || "term";
  if (nameNode) nameNode.textContent = name;
  kindNode.textContent = kind;
  edfNode.textContent = term.effective_df === null || term.effective_df === undefined
    ? "EDF unavailable"
    : `EDF ${fmt(term.effective_df)}`;
  const reference = term.reference;
  referenceNode.hidden = !reference;
  referenceNode.textContent = reference
    ? `reference ${reference.level} · ${REFERENCE_POLICY[reference.policy]}`
    : "";
  const impact = term.impact || {};
  const suffix = note ? ` · ${note}` : "";
  const selected = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  if (pendingCount > 0) {
    // While changes wait, the line leads with them: what is drawn is the last refit.
    const waiting = statusNode.ownerDocument.createElement("strong");
    waiting.className = "status-waiting";
    waiting.textContent = waitingLabel(pendingCount);
    statusNode.replaceChildren(waiting, ` · ${selectionSize ? selected : FROM_LAST_REFIT}`);
  } else {
    statusNode.textContent = selected;
  }
  statusNode.dataset.term = name;
}
