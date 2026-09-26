// @ts-check

import { fmt } from "../format.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {[string, string[]]} FeatureGroup */
/** @typedef {{terms:Record<string, TermPayload>, activeTerm:string, tabStop:string|undefined}} RowState */

const STORAGE_KEY = "superglm.editor.featureList";
const STEP_KEYS = new Set(["ArrowUp", "ArrowDown", "Home", "End"]);

/**
 * Keep the features whose name contains the query, ignoring case; a group left
 * empty drops out. An empty query returns the groups untouched.
 *
 * @param {FeatureGroup[]} groups
 * @param {string} query
 * @returns {FeatureGroup[]}
 */
export function filterFeatures(groups, query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return groups;
  /** @type {FeatureGroup[]} */
  const kept = [];
  for (const [group, names] of groups) {
    const matches = names.filter((name) => name.toLowerCase().includes(needle));
    if (matches.length) kept.push([group, matches]);
  }
  return kept;
}

/**
 * Wire the feature list once. A row is a button, so Enter and Space select it
 * through its click; ArrowUp/ArrowDown, Home and End step focus through the
 * visible rows, ArrowUp on the first row returns to the search box, ArrowDown
 * in the search box enters the list, and Enter there opens the first match.
 *
 * @param {{search:HTMLInputElement, rows:HTMLElement, toggle:HTMLButtonElement}} nodes
 * @param {{onSelect:(name:string)=>unknown, onQuery:(query:string)=>unknown, onToggle:()=>unknown}} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindFeatureList({ search, rows, toggle }, { onSelect, onQuery, onToggle }) {
  /** @param {Event} event */
  function onRowsClick(event) {
    const row = rowOf(event.target);
    if (row) onSelect(termOf(row));
  }

  /** @param {KeyboardEvent} event */
  function onRowsKeyDown(event) {
    const row = rowOf(event.target);
    if (!row || !STEP_KEYS.has(event.key)) return;
    event.preventDefault();
    const visible = visibleRows(rows);
    const index = visible.indexOf(row);
    if (event.key === "ArrowUp" && index === 0) {
      search.focus();
      return;
    }
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? visible.length - 1
        : index + (event.key === "ArrowDown" ? 1 : -1);
    visible[Math.min(next, visible.length - 1)].focus();
  }

  /** @param {KeyboardEvent} event */
  function onSearchKeyDown(event) {
    if (event.key === "ArrowDown") {
      const stop = rows.querySelector('[data-term][tabindex="0"]');
      if (!(stop instanceof HTMLElement)) return;
      event.preventDefault();
      stop.focus();
    } else if (event.key === "Enter") {
      const first = visibleRows(rows)[0];
      if (!first || !search.value.trim()) return;
      event.preventDefault();
      onSelect(termOf(first));
    } else if (event.key === "Escape" && search.value) {
      event.preventDefault();
      search.value = "";
      onQuery("");
    }
  }

  function onSearchInput() {
    onQuery(search.value);
  }

  rows.addEventListener("click", onRowsClick);
  rows.addEventListener("keydown", onRowsKeyDown);
  search.addEventListener("input", onSearchInput);
  search.addEventListener("keydown", onSearchKeyDown);
  toggle.addEventListener("click", onToggle);

  return Object.freeze({
    destroy() {
      rows.removeEventListener("click", onRowsClick);
      rows.removeEventListener("keydown", onRowsKeyDown);
      search.removeEventListener("input", onSearchInput);
      search.removeEventListener("keydown", onSearchKeyDown);
      toggle.removeEventListener("click", onToggle);
    },
  });
}

/**
 * Render the list from editor state: rows grouped as the chart groups terms,
 * the active row marked current, made the list's single tab stop and scrolled
 * into view, and the collapsed strip naming it. A row that had focus keeps it.
 *
 * @param {{root:HTMLElement, rows:HTMLElement, toggle:HTMLButtonElement, strip:HTMLElement}} nodes
 * @param {{groups:FeatureGroup[], terms:Record<string, TermPayload>, activeTerm:string, query:string, open:boolean}} state
 */
export function renderFeatureList(
  { root, rows, toggle, strip },
  { groups, terms, activeTerm, query, open },
) {
  root.dataset.open = String(open);
  toggle.setAttribute("aria-expanded", String(open));
  strip.textContent = activeTerm;

  const doc = root.ownerDocument;
  const visible = filterFeatures(groups, query);
  const names = visible.flatMap(([, groupNames]) => groupNames);
  const tabStop = names.includes(activeTerm) ? activeTerm : names[0];
  const focused = doc.activeElement;
  const focusedTerm = focused instanceof HTMLElement && rows.contains(focused)
    ? focused.dataset.term
    : undefined;

  /** @type {RowState} */
  const rowState = { terms, activeTerm, tabStop };
  rows.replaceChildren(...visible.map(([group, groupNames]) => featureGroup(doc, group, groupNames, rowState)));
  if (query.trim() && names.length === 0) {
    const notice = doc.createElement("p");
    notice.className = "feature-empty";
    notice.textContent = "No features match.";
    rows.appendChild(notice);
  }

  const active = rows.querySelector('[aria-current="true"]');
  if (active instanceof HTMLElement) active.scrollIntoView({ block: "nearest" });
  const refocus = visibleRows(rows).find((row) => row.dataset.term === focusedTerm);
  if (refocus) refocus.focus();
}

/**
 * Whether the list was left open or collapsed; with no stored choice, or
 * unusable storage (private mode, a blocked origin), `fallback` decides.
 * @param {boolean} fallback
 * @param {Pick<Storage, 'getItem'>} [storage] defaults to localStorage, whose access may itself throw
 */
export function readFeatureListOpen(fallback, storage) {
  try {
    const stored = (storage ?? localStorage).getItem(STORAGE_KEY);
    return stored === null ? fallback : stored === "open";
  } catch {
    return fallback;
  }
}

/** @param {boolean} open @param {Pick<Storage, 'setItem'>} [storage] */
export function storeFeatureListOpen(open, storage) {
  try {
    (storage ?? localStorage).setItem(STORAGE_KEY, open ? "open" : "collapsed");
  } catch {
    // Unusable storage: the preference lasts this page only.
  }
}

/** @param {Document} doc @param {string} group @param {string[]} names @param {RowState} rowState */
function featureGroup(doc, group, names, rowState) {
  const section = doc.createElement("section");
  section.className = "feature-group";
  const title = doc.createElement("h3");
  title.className = "feature-group-title";
  title.textContent = group;
  section.append(title, ...names.map((name) => featureRow(doc, name, rowState)));
  return section;
}

/** @param {Document} doc @param {string} name @param {RowState} rowState */
function featureRow(doc, name, { terms, activeTerm, tabStop }) {
  const term = terms[name];
  const row = doc.createElement("button");
  row.type = "button";
  row.className = "feature-row";
  row.dataset.term = name;
  row.tabIndex = name === tabStop ? 0 : -1;
  if (name === activeTerm) row.setAttribute("aria-current", "true");
  row.append(
    span(doc, "feature-row-name", name),
    span(doc, "feature-row-kind", term.term_type || term.kind || "term"),
    span(doc, "feature-row-edf", edfLabel(term.effective_df)),
  );
  return row;
}

/** @param {Document} doc @param {string} className @param {string} text */
function span(doc, className, text) {
  const node = doc.createElement("span");
  node.className = className;
  node.textContent = text;
  return node;
}

/** @param {number|null|undefined} edf */
function edfLabel(edf) {
  return edf === null || edf === undefined ? "EDF —" : `EDF ${fmt(edf)}`;
}

/** @param {HTMLElement} rows @returns {HTMLElement[]} */
function visibleRows(rows) {
  return /** @type {HTMLElement[]} */ ([...rows.querySelectorAll("[data-term]")]);
}

/** @param {EventTarget|null} target @returns {HTMLElement|null} */
function rowOf(target) {
  const row = target instanceof Element ? target.closest("[data-term]") : null;
  return row instanceof HTMLElement ? row : null;
}

/** @param {HTMLElement} row */
function termOf(row) {
  return /** @type {string} */ (row.dataset.term);
}
