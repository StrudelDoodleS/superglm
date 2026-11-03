// @ts-check

import { sparkline } from "../chart/sparkline.js";
import { fmtEdf } from "../format.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/**
 * A heading and its terms; a distributional model's group also names the
 * parameter's role, which colours the heading and the rows' mini curves.
 * @typedef {[string, string[]]|[string, string[], "average"|"spread"]} FeatureGroup
 */
/** @typedef {{terms:Record<string, TermPayload>, activeTerm:string, tabStop:string|undefined, waiting:ReadonlySet<string>}} RowState */

const STORAGE_KEY = "superglm.editor.featureList";
const STEP_KEYS = new Set(["ArrowUp", "ArrowDown", "Home", "End"]);
const SVG_NS = "http://www.w3.org/2000/svg";
const MINI_CURVE = { width: 96, height: 16 };

/**
 * Keep the features whose displayed name contains the query, ignoring case; a
 * group left empty drops out. An empty query returns the groups untouched.
 *
 * @param {FeatureGroup[]} groups
 * @param {string} query
 * @param {(name:string)=>string} [label] the name a feature is shown and searched by
 * @returns {FeatureGroup[]}
 */
export function filterFeatures(groups, query, label = (name) => name) {
  const needle = query.trim().toLowerCase();
  if (!needle) return groups;
  /** @type {FeatureGroup[]} */
  const kept = [];
  for (const [group, names, ...role] of groups) {
    const matches = names.filter((name) => label(name).toLowerCase().includes(needle));
    if (matches.length) kept.push(/** @type {FeatureGroup} */ ([group, matches, ...role]));
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
 * @param {{groups:FeatureGroup[], terms:Record<string, TermPayload>, activeTerm:string, query:string, open:boolean, waiting?:ReadonlySet<string>}} state
 */
export function renderFeatureList(
  { root, rows, toggle, strip },
  { groups, terms, activeTerm, query, open, waiting = new Set() },
) {
  root.dataset.open = String(open);
  toggle.setAttribute("aria-expanded", String(open));
  strip.textContent = stripLabel(terms[activeTerm], activeTerm);

  const doc = root.ownerDocument;
  const visible = filterFeatures(groups, query, (name) => terms[name]?.term_label ?? name);
  const names = visible.flatMap(([, groupNames]) => groupNames);
  const tabStop = names.includes(activeTerm) ? activeTerm : names[0];
  const focused = doc.activeElement;
  const focusedTerm = focused instanceof HTMLElement && rows.contains(focused)
    ? focused.dataset.term
    : undefined;

  /** @type {RowState} */
  const rowState = { terms, activeTerm, tabStop, waiting };
  rows.replaceChildren(...visible.map(([group, groupNames, role]) => featureGroup(doc, group, groupNames, role, rowState)));
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

/**
 * @param {Document} doc @param {string} group @param {string[]} names
 * @param {"average"|"spread"|undefined} role @param {RowState} rowState
 */
function featureGroup(doc, group, names, role, rowState) {
  const section = doc.createElement("section");
  section.className = "feature-group";
  const title = doc.createElement("h3");
  title.className = "feature-group-title";
  title.textContent = group;
  if (role) section.dataset.role = title.dataset.role = role;
  section.append(title, ...names.map((name) => featureRow(doc, name, rowState)));
  return section;
}

/** @param {Document} doc @param {string} name @param {RowState} rowState */
function featureRow(doc, name, { terms, activeTerm, tabStop, waiting }) {
  const term = terms[name];
  const row = doc.createElement("button");
  row.type = "button";
  row.className = "feature-row";
  row.dataset.term = name;
  row.tabIndex = name === tabStop ? 0 : -1;
  if (name === activeTerm) row.setAttribute("aria-current", "true");
  row.append(span(doc, "feature-row-name", term.term_label ?? name));
  if (waiting.has(name)) row.append(waitingDot(doc));
  if (term.parameter_role !== undefined) row.append(miniCurve(doc, term));
  row.append(
    span(doc, "feature-row-kind", term.term_type || term.kind || "term"),
    span(doc, "feature-row-edf", edfLabel(term.effective_df)),
  );
  return row;
}

/**
 * The active feature as the collapsed strip names it: a distributional
 * model's feature carries its parameter, since one name is in two groups.
 * @param {TermPayload|undefined} term @param {string} name
 */
function stripLabel(term, name) {
  if (!term?.term_label) return name;
  return term.parameter_label ? `${term.term_label} · ${term.parameter_label}` : term.term_label;
}

/**
 * The feature's effect drawn small, in its parameter's colour. It repeats what
 * the row's name and kind already say, so it is hidden from assistive tech.
 * @param {Document} doc @param {TermPayload} term
 */
function miniCurve(doc, term) {
  const { width, height } = MINI_CURVE;
  const svg = doc.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "mini-curve");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("aria-hidden", "true");
  const curve = sparkline(term, MINI_CURVE);
  if (curve.kind === "line") {
    const path = doc.createElementNS(SVG_NS, "path");
    path.setAttribute("d", curve.d);
    svg.append(path);
    return svg;
  }
  for (const bar of curve.bars) {
    const rect = doc.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(bar.x));
    rect.setAttribute("y", String(bar.y));
    rect.setAttribute("width", String(bar.w));
    rect.setAttribute("height", String(bar.h));
    svg.append(rect);
  }
  return svg;
}

/** An amber dot right after the name of a term with a change waiting for Refit. @param {Document} doc */
function waitingDot(doc) {
  const dot = doc.createElement("span");
  dot.className = "feature-row-waiting";
  dot.setAttribute("role", "img");
  dot.setAttribute("aria-label", "Waiting for refit");
  return dot;
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
  return edf === null || edf === undefined ? "EDF —" : fmtEdf(edf);
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
