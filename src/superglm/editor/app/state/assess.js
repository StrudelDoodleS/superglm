// @ts-check

import { CHECK_REPORTS, isReportName } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').AppView} AppView */
/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/** @typedef {import('../api/contracts.js').ReportView} ReportView */
/**
 * One report or check in the Assess list: its view, its name (which also heads
 * its large view), the sentence its popover gives, and its icon's path in the
 * 24-unit toolbar box.
 * @typedef {{id: ReportName, label: string, hint: string, icon: string}} AssessEntry
 */
/** @typedef {{id: "assess", label: string, hint: string, icon: string}} OverviewEntry */
/** @typedef {{key: string, name: string, entries: readonly AssessEntry[]}} AssessGroup */

/**
 * The reports the overview's cards read: the Final fit card reads the cv report's `final_fit`.
 * @type {readonly ReportView[]}
 */
export const OVERVIEW_REPORTS = Object.freeze(/** @type {ReportView[]} */ (["validation", "cv", "diagnose"]));

/**
 * What the overview asks of Python, in turn: its reports, then each check the model offers,
 * cheapest first.
 * @param {readonly CheckReport[]} offered
 * @returns {ReportName[]}
 */
export function overviewReports(offered) {
  return [...OVERVIEW_REPORTS, ...CHECK_REPORTS.filter((name) => offered.includes(name))];
}

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
  Object.freeze({
    key: "distribution",
    name: "Distribution",
    entries: Object.freeze([
      Object.freeze({
        id: /** @type {ReportName} */ ("check_qq"),
        label: "Q-Q plot",
        hint: "Sorted quantile residuals against their expected values, with a 95% band that holds for every point at once.",
        icon: "M4 20 20 4M6 17l2-1M10 14l2-1M14 10l2-2",
      }),
      Object.freeze({
        id: /** @type {ReportName} */ ("check_pit"),
        label: "PIT histogram",
        hint: "Where each observed value falls in its own predicted distribution, in 20 bins, with a 95% band that holds for every bin at once.",
        icon: "M4 20V9h3v11M10.5 20V7h3v13M17 20V10h3v10M3 20h18",
      }),
    ]),
  }),
  Object.freeze({
    key: "calibration",
    name: "Calibration",
    entries: Object.freeze([
      Object.freeze({
        id: /** @type {ReportName} */ ("check_ae"),
        label: "Actual over expected",
        hint: "Observed total over predicted total, by band of predicted Mean and by each input, with 95% intervals that hold for every group at once.",
        icon: "M3 12h18M6 8v8M12 5v8M18 10v6",
      }),
      Object.freeze({
        id: /** @type {ReportName} */ ("check_calibration"),
        label: "Quantile calibration",
        hint: "How often each predicted quantile is exceeded against how often it should be, with a 95% band that holds for every quantile at once.",
        icon: "M4 20h16M4 20V4M7 17l3-4 3 1 3-5 2-3",
      }),
    ]),
  }),
]);

/**
 * The top-bar tab a view sits under: the overview, every report and every check are Assess.
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
