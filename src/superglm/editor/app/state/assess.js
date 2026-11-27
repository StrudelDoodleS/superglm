// @ts-check

import { isReportName } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').AppView} AppView */
/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/**
 * One report in the Assess list: its view, its name, the sentence its popover
 * gives, and its icon's path in the 24-unit toolbar box.
 * @typedef {{id: ReportName, label: string, hint: string, icon: string}} AssessEntry
 */
/** @typedef {{id: "assess", label: string, hint: string, icon: string}} OverviewEntry */
/** @typedef {{key: string, name: string, entries: readonly AssessEntry[]}} AssessGroup */

/** @type {Readonly<OverviewEntry>} */
export const OVERVIEW_ENTRY = Object.freeze({
  id: /** @type {"assess"} */ ("assess"),
  label: "Overview",
  hint: "One card per report, each with its headline number.",
  icon: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
});

/**
 * The Assess list below its Overview: one group per heading, its entries in
 * list order. Selecting a view is one tagged value, `activeView`, written by
 * main.js's `showView` alone; every view reaches every other, so this table of
 * entries is what there is to walk.
 * @type {readonly Readonly<AssessGroup>[]}
 */
export const ASSESS_GROUPS = Object.freeze([
  Object.freeze({
    key: "reports",
    name: "Reports",
    entries: Object.freeze([
      Object.freeze({
        id: /** @type {ReportName} */ ("validation"),
        label: "Validation",
        hint: "The model as edited, scored on each split, beside the model as fitted.",
        icon: "M12 3a9 9 0 1 0 0.01 0M8 12.5l2.7 2.7L16.5 9",
      }),
      Object.freeze({
        id: /** @type {ReportName} */ ("cv"),
        label: "Cross-validation",
        hint: "The same scores across the folds of a cross_validate() result. Run CV and Final fit start here.",
        icon: "M4 6h16M4 12h16M4 18h16M9 4v4M15 10v4M7 16v4",
      }),
      Object.freeze({
        id: /** @type {ReportName} */ ("final"),
        label: "Final fit",
        hint: "The model refitted on train and validation rows, with your hand edits put back.",
        icon: "M5 21V4M5 4h11l-2 4 2 4H5",
      }),
      Object.freeze({
        id: /** @type {ReportName} */ ("diagnose"),
        label: "Smoothing search",
        hint: "Whether each fit's smoothing search converged, and what to try if not. Opening it refits nothing.",
        icon: "M3 15c3-6 6-6 9-2s6 4 9-2M3 20h18",
      }),
    ]),
  }),
]);

/**
 * The top-bar tab a view sits under: the overview and every report are Assess.
 * @param {AppView} view
 * @returns {'editor'|'assess'|'risk'}
 */
export function tabOf(view) {
  if (view === "assess" || isReportName(view)) return "assess";
  return view === "risk" ? "risk" : "editor";
}

/**
 * The list entry of `view`, or null for a view outside Assess.
 * @param {string} view
 * @returns {Readonly<AssessEntry>|Readonly<OverviewEntry>|null}
 */
export function assessEntry(view) {
  if (view === OVERVIEW_ENTRY.id) return OVERVIEW_ENTRY;
  for (const group of ASSESS_GROUPS) {
    const entry = group.entries.find((candidate) => candidate.id === view);
    if (entry) return entry;
  }
  return null;
}

/**
 * The view to open for a requested one: an unknown view is the Editor, and a
 * window that holds Assess alone opens its overview for the Editor or Explain.
 * @param {string} view
 * @param {{assessWindow: boolean}} options
 * @returns {AppView}
 */
export function openableView(view, { assessWindow }) {
  /** @type {AppView} */
  const known = view === "risk" || assessEntry(view) !== null
    ? /** @type {AppView} */ (view)
    : "editor";
  return assessWindow && tabOf(known) !== "assess" ? "assess" : known;
}
