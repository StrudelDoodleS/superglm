// @ts-check

import { fmt, fmtEdf, fmtPercent } from "../format.js";
import { STRUCTURE_HELP } from "./help_content.js";

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

/**
 * Whether the Chart / Table switch steps to the end of the toolbar row:
 * Contrib and Build are shown and, with the switch in its place, are not on
 * its row. Rows are told apart by vertical overlap.
 * @param {{top:number, bottom:number}} toggle the switch in its place
 * @param {{top:number, bottom:number}|null} tools Contrib and Build, null when not shown
 */
export function viewToggleGoesLast(toggle, tools) {
  return tools !== null && !(tools.top < toggle.bottom && toggle.top < tools.bottom);
}

/**
 * Contrib and Build never part; when the switch's row cannot hold them too,
 * the switch gives way to the end of the row. It is measured in its place
 * each time, so the outcome does not depend on where it stood before.
 * @param {HTMLElement} bar the context bar @param {HTMLElement} toggle the Chart / Table switch
 * @param {HTMLElement} tools the group holding Contrib and Build
 */
export function placeTermViewToggle(bar, toggle, tools) {
  bar.dataset.viewToggle = "inline";
  const shown = tools.getClientRects().length > 0;
  const last = viewToggleGoesLast(
    toggle.getBoundingClientRect(),
    shown ? tools.getBoundingClientRect() : null,
  );
  if (last) bar.dataset.viewToggle = "last";
}

/** @param {number} count */
export function waitingLabel(count) {
  return `${count} ${count === 1 ? "change" : "changes"} waiting for refit`;
}

/**
 * @param {{nameNode?:HTMLElement|null, kindNode:HTMLElement, edfNode:HTMLElement, referenceNode:HTMLElement, statusNode:HTMLElement}} nodes
 * @param {{name:string, term:TermPayload, selectionSize:number, note?:string, pendingCount?:number,
 *   range?:{lo:string, hi:string}|null}} context `range` names the ends of a Shift-click
 *   span, as the axis reads them, while the selection is that span
 */
export function renderContextBar(
  { nameNode = null, kindNode, edfNode, referenceNode, statusNode },
  { name, term, selectionSize, note = "", pendingCount = 0, range = null },
) {
  const kind = term.term_type || term.kind || "term";
  if (nameNode) nameNode.textContent = name;
  kindNode.textContent = kind;
  edfNode.textContent = term.effective_df === null || term.effective_df === undefined
    ? "EDF unavailable"
    : fmtEdf(term.effective_df);
  const reference = term.reference;
  const waitingReference = term.pending ? term.pending.reference : null;
  referenceNode.hidden = !reference && !waitingReference;
  referenceNode.textContent = waitingReference
    ? `reference ${waitingReference} · waiting`
    : reference
      ? `reference ${reference.level} · ${REFERENCE_POLICY[reference.policy]}`
      : "";
  referenceNode.dataset.waiting = waitingReference ? "true" : "false";
  const impact = term.impact || {};
  const suffix = note ? ` · ${note}` : "";
  const exposure = `selected exposure ${fmtPercent(impact.selected_weight_share || 0)}${suffix}`;
  const selected = range
    ? `Range ${range.lo} – ${range.hi} · ${selectionSize} of ${term.n_points} points · ${exposure}`
    : `${selectionSize} of ${term.n_points} selected · average edit relativity ${fmt(impact.weighted_mean_relativity || 1)}x · ${exposure}`;
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

/**
 * The "New levels →" select: where levels the fit never saw go when the model
 * predicts. Only a plain categorical has one (`term.unseen` is null on every
 * other term). Its options are rebuilt only when the choices change, so a
 * render while the list is open leaves it open. While the choice cannot be
 * made the select is disabled and its popover gives the reason.
 * @param {{wrap:HTMLElement, select:HTMLSelectElement}} nodes
 * @param {TermPayload} term
 */
export function renderNewLevelsControl({ wrap, select }, term) {
  const unseen = term.unseen ?? null;
  wrap.hidden = unseen === null;
  if (unseen === null) return;
  const choices = JSON.stringify(unseen.choices);
  if (select.dataset.choices !== choices) {
    select.replaceChildren(...unseen.choices.map(({ value, label }) => {
      const option = select.ownerDocument.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }));
    select.dataset.choices = choices;
  }
  select.value = unseen.policy;
  select.disabled = unseen.reason !== null;
  wrap.dataset.popoverTitle = STRUCTURE_HELP.new_levels.title;
  wrap.dataset.popoverBody = unseen.reason ?? STRUCTURE_HELP.new_levels.body;
}
