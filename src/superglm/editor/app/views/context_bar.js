// @ts-check

import { fmt, fmtPercent } from "../format.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/**
 * @param {{kindNode:HTMLElement, edfNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string}} context
 */
export function renderContextBar(
  { kindNode, edfNode, statusNode },
  { name, term, selectionSize, note = "" },
) {
  const kind = term.term_type || term.kind || "term";
  kindNode.textContent = kind;
  edfNode.textContent = term.effective_df === null || term.effective_df === undefined
    ? "EDF unavailable"
    : `EDF ${fmt(term.effective_df)}`;
  const impact = term.impact || {};
  const suffix = note ? ` · ${note}` : "";
  statusNode.textContent = `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  statusNode.dataset.term = name;
}
