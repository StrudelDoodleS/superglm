// @ts-check

import { isCheckReport } from "../api/contracts.js";
import { escapeHTML } from "../format.js";
import { ASSESS_GROUPS, OVERVIEW_ENTRY, assessEntry } from "../state/assess.js";

/** @typedef {import('../api/contracts.js').AppView} AppView */
/** @typedef {import('../state/assess.js').AssessEntry} AssessEntry */
/** @typedef {import('../state/assess.js').OverviewEntry} OverviewEntry */
/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/**
 * The checks the model offers, and those of them flagged; a check it does not offer is hidden.
 * @typedef {{offered: readonly CheckReport[], flagged: readonly CheckReport[]}} ListChecks
 */

/** @type {ListChecks} */
const NO_CHECKS = Object.freeze({ offered: Object.freeze([]), flagged: Object.freeze([]) });

const STEP_KEYS = new Set(["ArrowUp", "ArrowDown", "Home", "End"]);

/**
 * The entry that holds the list's tab stop: the open one, else the Overview.
 * @param {AppView} view
 */
function currentEntry(view) {
  return assessEntry(view) ? view : OVERVIEW_ENTRY.id;
}

/**
 * Whether the row of entry `id` is hidden: a check the model does not offer.
 * @param {string} id @param {ListChecks} checks
 */
function rowHidden(id, checks) {
  return isCheckReport(id) && !checks.offered.includes(id);
}

/**
 * Whether the row of entry `id` is flagged.
 * @param {string} id @param {ListChecks} checks
 */
function rowFlagged(id, checks) {
  return isCheckReport(id) && checks.flagged.includes(id);
}

/** A row's accessible name, which says when its check is flagged. @param {string} label @param {boolean} flagged */
const rowName = (label, flagged) => (flagged ? `${label}, flagged` : label);

/**
 * @param {Readonly<AssessEntry>|Readonly<OverviewEntry>} entry
 * @param {string} current
 * @param {ListChecks} checks
 */
function rowMarkup(entry, current, checks) {
  const open = entry.id === current;
  const flagged = rowFlagged(entry.id, checks);
  const label = escapeHTML(entry.label);
  return `<button class="assess-row" type="button" data-assess-entry="${entry.id}" data-flagged="${flagged}"` +
    `${rowHidden(entry.id, checks) ? " hidden" : ""}` +
    `${open ? ' aria-current="true"' : ""} tabindex="${open ? 0 : -1}" aria-label="${escapeHTML(rowName(entry.label, flagged))}"` +
    ` data-popover-title="${label}" data-popover-body="${escapeHTML(entry.hint)}">` +
    '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">' +
    `<path d="${entry.icon}"></path></svg>` +
    `<span class="assess-row-label">${label}</span><span class="assess-flag" aria-hidden="true"></span></button>`;
}

/**
 * The Assess list: the Overview, then each group under its heading. A check the model does
 * not offer is hidden, and a group with nothing shown with it.
 * @param {AppView} view the view on screen; its entry is current and the tab stop
 * @param {ListChecks} [checks]
 * @returns {string}
 */
export function assessListMarkup(view, checks = NO_CHECKS) {
  const current = currentEntry(view);
  const groups = ASSESS_GROUPS.map((group) =>
    `<div class="assess-group" data-assess-group="${group.key}"` +
    `${group.entries.every((entry) => rowHidden(entry.id, checks)) ? " hidden" : ""}>` +
    `<h3 class="assess-group-title">${escapeHTML(group.name)}</h3>` +
    group.entries.map((entry) => rowMarkup(entry, current, checks)).join("") +
    "</div>");
  return rowMarkup(OVERVIEW_ENTRY, current, checks) + groups.join("");
}

/** The entries shown, in list order. @param {HTMLElement} root @returns {HTMLElement[]} */
function rowsOf(root) {
  const rows = Array.from(root.querySelectorAll("[data-assess-entry]")).filter(
    /** @returns {row is HTMLElement} */ (row) => row instanceof HTMLElement,
  );
  return rows.filter((row) => !row.hidden);
}

/**
 * Draw the list once, then show the checks the model offers, flag those flagged, and mark the
 * entry of `view` current with the single tab stop. The rows stay in place, so focus on one
 * survives.
 * @param {HTMLElement} root
 * @param {AppView} view
 * @param {ListChecks} [checks]
 */
export function renderAssessList(root, view, checks = NO_CHECKS) {
  if (!root.querySelector("[data-assess-entry]")) root.innerHTML = assessListMarkup(view, checks);
  const current = currentEntry(view);
  // An open check the model does not offer has no row to hold the tab stop; the Overview holds it.
  const stop = rowHidden(current, checks) ? OVERVIEW_ENTRY.id : current;
  for (const row of root.querySelectorAll("[data-assess-entry]")) {
    if (!(row instanceof HTMLElement)) continue;
    const id = row.dataset.assessEntry ?? "";
    const entry = assessEntry(id);
    const flagged = rowFlagged(id, checks);
    row.hidden = rowHidden(id, checks);
    row.dataset.flagged = String(flagged);
    if (entry) row.setAttribute("aria-label", rowName(entry.label, flagged));
    if (id === current) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
    row.tabIndex = id === stop ? 0 : -1;
  }
  for (const group of root.querySelectorAll("[data-assess-group]")) {
    if (group instanceof HTMLElement) group.hidden = !group.querySelector("[data-assess-entry]:not([hidden])");
  }
}

/**
 * The entry a key moves to from `from`: ArrowUp and ArrowDown step and stop at
 * the ends, Home and End jump to them. Any other key, or an unknown entry, is null.
 * @param {readonly string[]} ids the entries in list order
 * @param {string} from
 * @param {string} key
 * @returns {string|null}
 */
export function stepEntry(ids, from, key) {
  const index = ids.indexOf(from);
  if (index < 0 || !STEP_KEYS.has(key)) return null;
  if (key === "Home") return ids[0];
  if (key === "End") return ids[ids.length - 1];
  const next = index + (key === "ArrowDown" ? 1 : -1);
  return ids[Math.min(Math.max(next, 0), ids.length - 1)];
}

/**
 * Wire the list once. An entry is a button, so a click, Enter or Space opens
 * it; the arrows, Home and End move focus and the tab stop with it.
 * @param {HTMLElement} root
 * @param {{onOpen: (view: AppView) => unknown}} handlers
 * @returns {{destroy(): void}}
 */
export function bindAssessList(root, { onOpen }) {
  /** @param {EventTarget|null} target */
  const rowOf = (target) => {
    const row = target instanceof Element ? target.closest("[data-assess-entry]") : null;
    return row instanceof HTMLElement && root.contains(row) ? row : null;
  };

  /** @param {MouseEvent} event */
  function onClick(event) {
    const row = rowOf(event.target);
    const view = row?.dataset.assessEntry;
    if (view && assessEntry(view)) onOpen(/** @type {AppView} */ (view));
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const row = rowOf(event.target);
    if (!row) return;
    const rows = rowsOf(root);
    const next = stepEntry(rows.map((each) => each.dataset.assessEntry ?? ""), row.dataset.assessEntry ?? "", event.key);
    if (next === null) return;
    event.preventDefault();
    for (const each of rows) each.tabIndex = each.dataset.assessEntry === next ? 0 : -1;
    rows.find((each) => each.dataset.assessEntry === next)?.focus();
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}
