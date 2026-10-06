// @ts-check
// The inspector summary's view of the compact coefficient rows: the term each
// row belongs to, which rows a search and a filter keep, which term sections
// are open as the summary follows the chart, and the marks a search draws.
// Pure but for three small bindings; summary.js turns the result into markup.

import { escapeHTML } from "../format.js";

/**
 * The fields of one compact summary row (Python's `_compact_summary_row`)
 * read here.
 * @typedef {Object} SummaryRowLike
 * @property {string} [name]
 * @property {string} [group]
 * @property {string} [level_group]
 * @property {string} [kind]
 * @property {number|null} [edf]
 * @property {number|null} [p_value]
 * @property {string} [sig_class]
 * @property {string} [sig_code]
 */
/** @typedef {'all'|'edited'|'waiting'} SummaryFilter */
/**
 * What the inspector shows of the summary. `termNames` are the editor's terms,
 * which a decorated summary group such as "x_poly P(2)" is mapped back to.
 * `currentTerm` is the chart's; its section opens and the others fold, unless
 * `toggled` says the analyst opened or closed one. With no current term every
 * section is open. `waiting` counts the changes waiting for refit per term.
 * @typedef {Object} SummaryView
 * @property {string} query
 * @property {readonly string[]} termNames
 * @property {SummaryFilter} [filter]
 * @property {string} [currentTerm]
 * @property {ReadonlyMap<string, boolean>} [toggled]
 * @property {readonly string[]} [edited]
 * @property {Readonly<Record<string, number>>} [waiting]
 * @property {Readonly<Record<string, string>>} [kinds]
 */
/**
 * The p-value a section's line shows: its whole-term test's. A categorical
 * has no whole-term test, so its line shows none.
 * @typedef {Object} SummaryChip
 * @property {number} p
 * @property {string} sigClass
 * @property {string} sigCode
 */
/**
 * @typedef {Object} SummaryViewRow
 * @property {number} index position in the payload's rows
 * @property {boolean} hidden
 */
/**
 * One term's rows, in payload order. `label` is the group the summary prints
 * ("x_poly P(2)"); `term` is the editor term it belongs to ("x_poly"). The
 * intercept is a section without a header, always open.
 * @typedef {Object} SummaryViewSection
 * @property {string} term
 * @property {string} label
 * @property {boolean} header
 * @property {boolean} hidden
 * @property {boolean} open
 * @property {boolean} current
 * @property {string} kind
 * @property {number} waiting
 * @property {number|null} edf
 * @property {SummaryChip|null} chip
 * @property {SummaryViewRow[]} rows
 */
/**
 * @typedef {Object} SummaryViewModel
 * @property {SummaryViewSection[]} sections
 * @property {number} termCount sections with a row shown, the intercept aside
 * @property {number} rowCount rows shown in those sections
 * @property {boolean} empty a search or filter left no term to show
 */

/** @type {Readonly<SummaryView>} */
export const DEFAULT_SUMMARY_VIEW = Object.freeze({ query: "", termNames: Object.freeze([]) });

const INTERCEPT = "Intercept";
const REGEXP_SYNTAX = /[.*+?^${}()|[\]\\]/g;
// Row kinds that carry a whole-term test (Python's group rows).
const GROUP_TEST_KINDS = new Set(["spline", "piecewise"]);

/**
 * The group a row prints under: its group, or the text before its "[".
 * @param {SummaryRowLike} row
 */
function rowGroupLabel(row) {
  const group = row.group ? String(row.group) : "";
  if (group) return group;
  const name = row.name ? String(row.name) : "";
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(0, bracket) : name;
}

/**
 * The term a summary row belongs to: its group when that is an editor term,
 * else the longest editor term the group decorates ("x_poly P(2)" belongs to
 * "x_poly"), else the group itself.
 * @param {SummaryRowLike} row @param {readonly string[]} termNames
 */
export function summaryRowTerm(row, termNames) {
  const label = rowGroupLabel(row);
  if (termNames.includes(label)) return label;
  let owner = "";
  for (const term of termNames) {
    if (label.startsWith(`${term} `) && term.length > owner.length) owner = term;
  }
  return owner || label;
}

/**
 * The level a row names: the text inside its brackets, "10" in "bonus[10]";
 * empty for a whole-term row.
 * @param {SummaryRowLike} row @param {string} term
 */
export function summaryLevelLabel(row, term) {
  const name = row.name ? String(row.name) : "";
  if (!name.endsWith("]")) return "";
  const prefix = `${term}[`;
  if (name.startsWith(prefix)) return name.slice(prefix.length, -1);
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(bracket + 1, -1) : "";
}

/**
 * A case-insensitive pattern for the search text taken literally, or null when
 * the box is empty. Labels are compared as text: "1" finds levels 1 and 10.
 * @param {string} query @returns {RegExp|null}
 */
function queryPattern(query) {
  const needle = query.trim();
  return needle ? new RegExp(needle.replace(REGEXP_SYNTAX, "\\$&"), "giu") : null;
}

/** @param {string} text @param {RegExp} pattern */
function hasMatch(text, pattern) {
  return text.search(pattern) >= 0;
}

/**
 * `text` escaped for HTML, each match of the search wrapped in <mark>.
 * @param {string} text @param {string} query
 */
export function highlightMatches(text, query) {
  const pattern = queryPattern(query);
  if (!pattern) return escapeHTML(text);
  let html = "";
  let at = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index ?? 0;
    html += `${escapeHTML(text.slice(at, start))}<mark>${escapeHTML(match[0])}</mark>`;
    at = start + match[0].length;
  }
  return html + escapeHTML(text.slice(at));
}

/**
 * A row's name with the search marked in the two parts it reads: the term,
 * and the level inside the brackets.
 * @param {string} name @param {string} term @param {string} query
 */
export function highlightRowName(name, term, query) {
  const prefix = `${term}[`;
  if (term && name.startsWith(prefix) && name.endsWith("]")) {
    const level = name.slice(prefix.length, -1);
    return `${highlightMatches(term, query)}[${highlightMatches(level, query)}]`;
  }
  return highlightMatches(name, query);
}

/**
 * Group the rows into term sections, in payload order, and apply the view.
 * The search: a section whose term matches keeps every row; otherwise a row
 * stays when its level or level group matches; the intercept, which is no
 * term, leaves while searching. The filter keeps every term, the terms with
 * hand edits, or the terms waiting for refit. A section is open when the
 * analyst opened it, else when a search is active, else when it is the chart's
 * term; a folded section shows its line only.
 * @param {readonly SummaryRowLike[]} rows @param {SummaryView} view
 * @returns {SummaryViewModel}
 */
export function summaryViewModel(rows, view) {
  /** @type {SummaryViewSection[]} */
  const sections = [];
  rows.forEach((row, index) => {
    const term = summaryRowTerm(row, view.termNames);
    let section = sections.at(-1);
    if (!section || section.term !== term) {
      section = newSection(term, rowGroupLabel(row), view);
      sections.push(section);
    }
    section.rows.push({ index, hidden: false });
  });

  const pattern = queryPattern(view.query);
  const filter = view.filter ?? "all";
  const edited = new Set(view.edited ?? []);
  let termCount = 0;
  let rowCount = 0;
  for (const section of sections) {
    summarizeSection(section, rows);
    const matched = section.rows.map((entry) => rowMatches(rows[entry.index], section, pattern));
    section.hidden = !passesFilter(section, filter, edited) || !matched.some(Boolean);
    section.open = !section.header || (
      view.toggled?.get(section.term) ?? (pattern !== null || !view.currentTerm || section.current)
    );
    section.rows.forEach((entry, k) => {
      entry.hidden = section.hidden || !matched[k] || !section.open;
    });
    if (section.header && !section.hidden) {
      termCount += 1;
      rowCount += matched.filter(Boolean).length;
    }
  }
  const empty = termCount === 0 && (pattern !== null || filter !== "all");
  return { sections, termCount, rowCount, empty };
}

/**
 * @param {string} term @param {string} label @param {SummaryView} view
 * @returns {SummaryViewSection}
 */
function newSection(term, label, view) {
  return {
    term,
    label,
    header: term !== INTERCEPT,
    hidden: false,
    open: true,
    current: term === view.currentTerm,
    kind: view.kinds?.[term] ?? "",
    waiting: view.waiting?.[term] ?? 0,
    edf: null,
    chip: null,
    rows: []
  };
}

/**
 * Whether a row is in the search; every row is while the box is empty.
 * @param {SummaryRowLike} row @param {SummaryViewSection} section @param {RegExp|null} pattern
 */
function rowMatches(row, section, pattern) {
  if (pattern === null) return true;
  if (!section.header) return false;
  return hasMatch(section.term, pattern) ||
    hasMatch(summaryLevelLabel(row, section.term), pattern) ||
    hasMatch(row.level_group ? String(row.level_group) : "", pattern);
}

/**
 * @param {SummaryViewSection} section @param {SummaryFilter} filter
 * @param {ReadonlySet<string>} edited
 */
function passesFilter(section, filter, edited) {
  if (filter === "edited") return section.header && edited.has(section.term);
  if (filter === "waiting") return section.waiting > 0;
  return true;
}

/** @param {unknown} value @returns {value is number} */
function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The figures on a section's line: the EDF its rows carry, and the p-value of
 * its whole-term test. A categorical has no whole-term test, so its line has
 * no p chip: the smallest of its level p-values would overstate the term's
 * significance. Significance comes from Python's class and code for the test
 * row; nothing is re-derived here.
 * @param {SummaryViewSection} section @param {readonly SummaryRowLike[]} rows
 */
function summarizeSection(section, rows) {
  const sectionRows = section.rows.map((entry) => rows[entry.index]);
  const edfs = sectionRows.map((row) => row.edf).filter(isFiniteNumber);
  section.edf = edfs.length ? edfs.reduce((sum, value) => sum + value, 0) : null;
  const test = sectionRows.find(
    (row) => GROUP_TEST_KINDS.has(String(row.kind)) && isFiniteNumber(row.p_value)
  );
  if (!test) {
    section.chip = null;
    return;
  }
  section.kind ||= String(test.kind);
  section.chip = {
    p: Number(test.p_value),
    sigClass: String(test.sig_class ?? ""),
    sigCode: String(test.sig_code ?? "")
  };
}

/**
 * The changes waiting for refit per term, from the state's top-level `pending`.
 * @param {readonly {term:string}[]} pending @returns {Record<string, number>}
 */
export function waitingCounts(pending) {
  /** @type {Record<string, number>} */
  const counts = {};
  for (const step of pending) counts[step.term] = (counts[step.term] ?? 0) + 1;
  return counts;
}

/** @param {number} count @param {string} noun */
function counted(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * What the search box reports: nothing while it is empty, else the terms and
 * rows found, or that none match.
 * @param {SummaryViewModel} model @param {string} query
 */
export function summaryCountText(model, query) {
  if (!query.trim()) return "";
  if (model.termCount === 0) return "No terms match.";
  return `${counted(model.termCount, "term")} · ${counted(model.rowCount, "row")}`;
}

/** @param {string|undefined} value @returns {SummaryFilter|null} */
function summaryFilter(value) {
  return value === "all" || value === "edited" || value === "waiting" ? value : null;
}

/**
 * Wire the All / Edited / Waiting buttons inside `root`.
 * @param {HTMLElement} root @param {(filter:SummaryFilter)=>unknown} onFilter
 * @returns {{destroy:()=>void}}
 */
export function bindSummaryFilter(root, onFilter) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("[data-summary-filter]");
    if (!(button instanceof HTMLElement) || !root.contains(button)) return;
    const filter = summaryFilter(button.dataset.summaryFilter);
    if (filter) onFilter(filter);
  }

  root.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
    },
  });
}

/** @param {HTMLElement} root @param {SummaryFilter} filter */
export function renderSummaryFilter(root, filter) {
  for (const button of root.querySelectorAll("[data-summary-filter]")) {
    if (button instanceof HTMLElement) {
      button.setAttribute("aria-pressed", String(button.dataset.summaryFilter === filter));
    }
  }
}

/**
 * Wire the section lines in the summary frame: a click reports the term and
 * whether its section was open.
 * @param {HTMLElement} frame @param {(term:string, open:boolean)=>unknown} onToggle
 * @returns {{destroy:()=>void}}
 */
export function bindSummarySections(frame, onToggle) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("[data-summary-section]");
    if (!(button instanceof HTMLElement) || !frame.contains(button)) return;
    onToggle(button.dataset.summarySection ?? "", button.getAttribute("aria-expanded") === "true");
  }

  frame.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      frame.removeEventListener("click", onClick);
    },
  });
}

/**
 * Wire the search box: each input reports the query; Escape clears a box that
 * holds text, and keeps the key from also closing a narrow inspector.
 * @param {HTMLInputElement} search @param {(query:string)=>unknown} onQuery
 * @returns {{destroy:()=>void}}
 */
export function bindSummarySearch(search, onQuery) {
  function onInput() {
    onQuery(search.value);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key !== "Escape" || !search.value) return;
    event.preventDefault();
    event.stopPropagation();
    search.value = "";
    onQuery("");
  }

  search.addEventListener("input", onInput);
  search.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      search.removeEventListener("input", onInput);
      search.removeEventListener("keydown", onKeyDown);
    },
  });
}
