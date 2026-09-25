// @ts-check

import { fmt, fmtPercent } from "../format.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../api/contracts.js').TermReference} TermReference */

/** @type {Readonly<Record<TermReference['policy'], string>>} */
const REFERENCE_POLICY = Object.freeze({
  most_exposed: "most exposed",
  first: "first",
  pinned: "pinned",
});

/**
 * @param {{nameNode?:HTMLElement|null, kindNode:HTMLElement, edfNode:HTMLElement, referenceNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string}} context
 */
export function renderContextBar(
  { nameNode = null, kindNode, edfNode, referenceNode, statusNode },
  { name, term, selectionSize, note = "" },
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
  statusNode.textContent = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  statusNode.dataset.term = name;
}
