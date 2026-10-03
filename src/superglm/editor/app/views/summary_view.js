// @ts-check
// The inspector summary's view of the compact coefficient rows: the term each
// row belongs to, which rows a search keeps, and the marks it draws. Pure but
// for the search box binding; summary.js turns the result into markup.

import { escapeHTML } from "../format.js";

/**
 * The fields of one compact summary row (Python's `_compact_summary_row`)
 * read here.
 * @typedef {Object} SummaryRowLike
 * @property {string} [name]
 * @property {string} [group]
 * @property {string} [level_group]
 */
/**
 * What the inspector shows of the summary. `termNames` are the editor's terms,
 * which a decorated summary group such as "x_poly P(2)" is mapped back to.
 * @typedef {Object} SummaryView
 * @property {string} query
 * @property {readonly string[]} termNames
 */
/**
 * @typedef {Object} SummaryViewRow
 * @property {number} index position in the payload's rows
 * @property {boolean} hidden
 */
/**
 * One term's rows, in payload order. `label` is the group the summary prints
 * ("x_poly P(2)"); `term` is the editor term it belongs to ("x_poly"). The
 * intercept is a section without a header.
 * @typedef {Object} SummaryViewSection
 * @property {string} term
 * @property {string} label
 * @property {boolean} header
 * @property {boolean} hidden
 * @property {SummaryViewRow[]} rows
 */
/**
 * @typedef {Object} SummaryViewModel
 * @property {SummaryViewSection[]} sections
 * @property {number} termCount sections with a row shown, the intercept aside
 * @property {number} rowCount rows shown in those sections
 */

/** @type {Readonly<SummaryView>} */
export const DEFAULT_SUMMARY_VIEW = Object.freeze({ query: "", termNames: Object.freeze([]) });

const INTERCEPT = "Intercept";
const REGEXP_SYNTAX = /[.*+?^${}()|[\]\\]/g;

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
 * Group the rows into term sections, in payload order, and apply the search:
 * a section whose term matches keeps every row; otherwise a row stays when its
 * level or level group matches. While a search is active the intercept, which
 * is no term, is hidden.
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
      section = { term, label: rowGroupLabel(row), header: term !== INTERCEPT, hidden: false, rows: [] };
      sections.push(section);
    }
    section.rows.push({ index, hidden: false });
  });

  const pattern = queryPattern(view.query);
  let termCount = 0;
  let rowCount = 0;
  for (const section of sections) {
    if (pattern !== null) {
      const keepAll = section.header && hasMatch(section.term, pattern);
      for (const entry of section.rows) {
        const row = rows[entry.index];
        entry.hidden = !section.header || !(
          keepAll ||
          hasMatch(summaryLevelLabel(row, section.term), pattern) ||
          hasMatch(row.level_group ? String(row.level_group) : "", pattern)
        );
      }
    }
    section.hidden = section.rows.every((entry) => entry.hidden);
    if (section.header && !section.hidden) {
      termCount += 1;
      rowCount += section.rows.filter((entry) => !entry.hidden).length;
    }
  }
  return { sections, termCount, rowCount };
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
