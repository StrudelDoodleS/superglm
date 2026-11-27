// @ts-check

import { escapeHTML } from "../format.js";
import { ASSESS_GROUPS, OVERVIEW_ENTRY, assessEntry } from "../state/assess.js";

/** @typedef {import('../api/contracts.js').AppView} AppView */
/** @typedef {import('../state/assess.js').AssessEntry} AssessEntry */
/** @typedef {import('../state/assess.js').OverviewEntry} OverviewEntry */

const STEP_KEYS = new Set(["ArrowUp", "ArrowDown", "Home", "End"]);

/**
 * The entry that holds the list's tab stop: the open one, else the Overview.
 * @param {AppView} view
 */
function currentEntry(view) {
  return assessEntry(view) ? view : OVERVIEW_ENTRY.id;
}

/**
 * @param {Readonly<AssessEntry>|Readonly<OverviewEntry>} entry
 * @param {string} current
 */
function rowMarkup(entry, current) {
  const open = entry.id === current;
  const label = escapeHTML(entry.label);
  return `<button class="assess-row" type="button" data-assess-entry="${entry.id}"` +
    `${open ? ' aria-current="true"' : ""} tabindex="${open ? 0 : -1}" aria-label="${label}"` +
    ` data-popover-title="${label}" data-popover-body="${escapeHTML(entry.hint)}">` +
    '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">' +
    `<path d="${entry.icon}"></path></svg>` +
    `<span class="assess-row-label">${label}</span></button>`;
}

/**
 * The Assess list: the Overview, then each group under its heading.
 * @param {AppView} view the view on screen; its entry is current and the tab stop
 * @returns {string}
 */
export function assessListMarkup(view) {
  const current = currentEntry(view);
  const groups = ASSESS_GROUPS.map((group) =>
    `<div class="assess-group" data-assess-group="${group.key}">` +
    `<h3 class="assess-group-title">${escapeHTML(group.name)}</h3>` +
    group.entries.map((entry) => rowMarkup(entry, current)).join("") +
    "</div>");
  return rowMarkup(OVERVIEW_ENTRY, current) + groups.join("");
}

/** @param {HTMLElement} root @returns {HTMLElement[]} */
function rowsOf(root) {
  return Array.from(root.querySelectorAll("[data-assess-entry]")).filter(
    (row) => row instanceof HTMLElement && !row.hidden,
  );
}

/**
 * Draw the list once, then mark the entry of `view` current and give it the
 * single tab stop. The rows stay in place, so focus on one survives.
 * @param {HTMLElement} root
 * @param {AppView} view
 */
export function renderAssessList(root, view) {
  if (!root.querySelector("[data-assess-entry]")) root.innerHTML = assessListMarkup(view);
  const current = currentEntry(view);
  for (const row of rowsOf(root)) {
    const open = row.dataset.assessEntry === current;
    if (open) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
    row.tabIndex = open ? 0 : -1;
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
