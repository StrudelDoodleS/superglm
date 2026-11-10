// @ts-check
import assert from "node:assert/strict";
import test from "node:test";

import {
  FOLDED,
  SECTION,
  chosenFit,
  diagnoseTabMarkup,
  findingsMarkup,
  fitsTableMarkup,
  moveSection,
  phasesMarkup,
  searchPathMarkup,
  sharedConvergedSentence,
  slowFits,
  summaryMarkup,
  termMatrixMarkup,
  worstRowsMarkup,
} from "../../src/superglm/editor/app/views/diagnose_tab.js";

import { foldColour } from "../../src/superglm/editor/app/views/cv_tab.js";

/** @typedef {import('../../src/superglm/editor/app/views/diagnose_tab.js').DiagnoseFit} DiagnoseFit */
/** @typedef {import('../../src/superglm/editor/app/views/diagnose_tab.js').DiagnosePayload} DiagnosePayload */

/** @param {string} id @param {string} label @param {Partial<DiagnoseFit>} [extra] @returns {DiagnoseFit} */
function fit(id, label, extra = {}) {
  return {
    id,
    label,
    rows: 1000,
    rounds: 12,
    stop: "practical_plateau",
    stop_text: "the REML score stopped improving",
    status: "converged",
    verdict: "Converged. The smoothing search reached an optimum and passed its own check.",
    explanation: null,
    search: "Quick smoothing updates, then Newton steps to an optimum",
    converged: true,
    seconds: 18.04,
    path: [10.5, 10.1, 10.0],
    late_gain: 0.1,
    settled_round: 2,
    terms: {
      "theta:age": { edf: 3.48, lambda: 1.05, shape: "curve", at_limit: false },
      "mean:area": { edf: 5, lambda: null, shape: "unsmoothed", at_limit: false },
    },
    phases: [],
    findings: [],
    ...extra,
  };
}

/** @param {DiagnoseFit[]} fits @returns {DiagnosePayload} */
function payload(fits) {
  return {
    report: "diagnose",
    title: "Diagnose",
    note: "Folds from Run CV on the current model.",
    fits,
    terms: [
      { name: "mean:area", label: "area · Average", short: "area", group: "Average", role: "average" },
      { name: "theta:age", label: "age · Spread", short: "age", group: "Spread", role: "spread" },
    ],
    cost_label: "Log score",
  };
}

const flat = fit("fold-2", "Fold 2", {
  rounds: 42,
  terms: {
    "theta:age": { edf: 1, lambda: 3.6e6, shape: "straight", at_limit: true },
    "mean:area": { edf: 5, lambda: null, shape: "unsmoothed", at_limit: false },
  },
});
const book = () => payload([fit("model", "All rows"), fit("fold-1", "Fold 1"), flat, fit("fold-3", "Fold 3", { rounds: 13 })]);

test("a fold with at least twice the middle fold's rounds is slow", () => {
  assert.deepEqual([...slowFits(book().fits)], ["fold-2"]);
});

test("rounds below ten are not slow, however they compare", () => {
  const quick = [fit("model", "All rows"), fit("fold-1", "Fold 1", { rounds: 2 }), fit("fold-2", "Fold 2", { rounds: 9 }), fit("fold-3", "Fold 3", { rounds: 3 })];
  assert.equal(slowFits(quick).size, 0);
});

test("two folds are too few to call one slow", () => {
  const pair = [fit("model", "All rows"), fit("fold-1", "Fold 1", { rounds: 10 }), fit("fold-2", "Fold 2", { rounds: 80 })];
  assert.equal(slowFits(pair).size, 0);
});

test("this model's rounds do not make a fold slow, nor is it one", () => {
  const fits = [fit("model", "All rows", { rounds: 500 }), fit("fold-1", "Fold 1"), fit("fold-2", "Fold 2"), fit("fold-3", "Fold 3")];
  assert.equal(slowFits(fits).size, 0);
});

test("the tab opens on the slowest fold unless one is asked for", () => {
  assert.equal(chosenFit(book(), "").id, "fold-2");
  assert.equal(chosenFit(book(), "fold-1").id, "fold-1");
  assert.equal(chosenFit(book(), "fold-9").id, "fold-2");
  assert.equal(chosenFit(payload([fit("model", "All rows")]), "").id, "model");
});

test("the tab opens on the fold with the most rounds, marked slow or not", () => {
  const rounds = (/** @type {number[]} */ counts) => payload([
    fit("model", "All rows", { rounds: 500 }),
    ...counts.map((count, index) => fit(`fold-${index + 1}`, `Fold ${index + 1}`, { rounds: count })),
  ]);
  assert.equal(chosenFit(rounds([12, 25, 40, 12, 12]), "").id, "fold-3");
  assert.equal(chosenFit(rounds([12, 14, 15]), "").id, "fold-3");
  assert.equal(chosenFit(rounds([0, 0, 0]), "").id, "model");
});

test("a fold keeps its own colour when an earlier fold failed", () => {
  const html = fitsTableMarkup(payload([fit("model", "All rows"), fit("fold-2", "Fold 2")]), "model");
  assert.match(html, new RegExp(`background: ${foldColour(1).replace(/[()]/g, "\\$&")}`));
  assert.doesNotMatch(html, new RegExp(`background: ${foldColour(0).replace(/[()]/g, "\\$&")}`));
});

/** A book of four fits where folds 1 and 2 did not converge and fold 3 stopped on a looser rule. */
const failing = () => payload([
  fit("model", "All rows"),
  fit("fold-1", "Fold 1", {
    status: "not_converged",
    converged: false,
    verdict: "Did not converge. The numbers are from the last step reached, so do not rely on standard errors or effective degrees of freedom.",
    explanation: "SuperLSS fit did not converge: refit with a larger max_reml_iter.",
    stop_text: "it reached its limit of 2 smoothing rounds (max_reml_iter) before settling",
    path: [10.5, 10.0, 9.9],
  }),
  fit("fold-2", "Fold 2", { status: "not_converged", converged: false, verdict: "Did not converge. The numbers are from the last step reached." }),
  fit("fold-3", "Fold 3", { status: "loose", verdict: "Converged, with a looser stop. The numbers are usable.", stop_text: "the REML score stopped improving by more than the quick stopping rule's threshold" }),
]);

test("each fit's verdict leads its row, with the word that says it, and a fit that did not converge gives its reason", () => {
  const html = fitsTableMarkup(failing(), "fold-1");
  const rows = Object.fromEntries([...html.matchAll(/<tr data-selected="\w+" data-status="(\w+)">([\s\S]*?)<\/tr>/g)].map((match) => [match[2].match(/data-diagnose-fit="([^"]+)"/)?.[1], match]));
  assert.match(rows["model"][2], /<strong>Converged\.<\/strong> The smoothing search reached an optimum/);
  assert.match(rows["fold-1"][2], /✕<\/span>\s*<strong>Did not converge\.<\/strong> The numbers are from the last step reached/);
  assert.match(rows["fold-1"][2], /Stopped because it reached its limit of 2 smoothing rounds \(max_reml_iter\) before settling\./);
  assert.equal(rows["fold-1"][1], "not_converged");
  assert.match(rows["fold-3"][2], /<strong>Converged, with a looser stop\.<\/strong>/);
  assert.match(rows["fold-3"][2], /Stopped because the REML score stopped improving/);
  assert.doesNotMatch(rows["model"][2], /Stopped because/);
  // The verdict is the first cell after the fit's name, not the last column.
  assert.ok(html.indexOf("<th class=\"diagnose-text diagnose-result\">Result</th>") < html.indexOf("<th>Smoothing rounds</th>"));
});

test("the fits table marks the slow fold and says rows and time", () => {
  const html = fitsTableMarkup(book(), "fold-2");
  assert.match(html, /18\.0 s/);
  assert.equal((html.match(/diagnose-slow/g) ?? []).length, 1);
  assert.match(html, /data-diagnose-fit="fold-2" aria-pressed="true"/);
  assert.match(html, /data-diagnose-fit="fold-1" aria-pressed="false"/);
});

test("a fit with no rows reads as a dash and a table with no timings has no time column", () => {
  const html = fitsTableMarkup(payload([fit("model", "All rows", { rows: null, seconds: null })]), "model");
  assert.equal((html.match(/<td>--<\/td>/g) ?? []).length, 1);
  assert.doesNotMatch(html, /Fit time/);
});

test("a search that differs between fits gets a column; one shared by every fit is said once, on the chart", () => {
  const same = book();
  assert.doesNotMatch(fitsTableMarkup(same, "fold-2"), /<th class="diagnose-text diagnose-search">Search<\/th>/);
  assert.match(searchPathMarkup(same, "fold-2"), /Quick smoothing updates, then Newton steps to an optimum\./);
  const mixed = payload([fit("model", "All rows"), fit("fold-1", "Fold 1", { search: "Quick smoothing updates only" })]);
  const html = fitsTableMarkup(mixed, "model");
  assert.match(html, /<th class="diagnose-text diagnose-search">Search<\/th>/);
  assert.match(html, /Quick smoothing updates only/);
  assert.doesNotMatch(searchPathMarkup(mixed, "model"), /Quick smoothing/);
});

test("a fit whose search is not recorded says nothing about one", () => {
  const glm = payload([fit("model", "All rows", { search: null })]);
  assert.doesNotMatch(fitsTableMarkup(glm, "model"), /EFS|Newton|Search<\/th>/);
  assert.doesNotMatch(searchPathMarkup(glm, "model"), /EFS|Newton/);
});

test("one line across the fits says how many did not converge and each name selects its fit", () => {
  const html = summaryMarkup(failing().fits);
  assert.match(html, /<strong>2 of 4 fits did not converge:<\/strong>/);
  assert.match(html, /data-diagnose-fit="fold-1">Fold 1<\/button>, <button[^>]*data-diagnose-fit="fold-2">Fold 2<\/button>/);
  assert.match(html, /1 converged with a looser stop: <button[^>]*data-diagnose-fit="fold-3">/);
  assert.match(summaryMarkup(book().fits), /^All 4 fits converged\. The smoothing search reached an optimum and passed its own check\.$/);
  assert.equal(summaryMarkup([fit("model", "All rows")].slice(0, 1)), "");
});

test("when every fit had its smoothing set, the page says so once and drops the search columns and chart", () => {
  const fixedFit = (/** @type {string} */ id, /** @type {string} */ label) => fit(id, label, {
    status: "fixed", verdict: "No smoothing search. You set the smoothing.", rounds: 0, path: [], stop: null, stop_text: "no smoothing search ran",
    late_gain: null, settled_round: null, search: null,
  });
  const data = payload([fixedFit("model", "All rows"), fixedFit("fold-1", "Fold 1")]);
  const html = diagnoseTabMarkup(data, "");
  assert.equal((html.match(/No smoothing search\./g) ?? []).length, 1);
  assert.doesNotMatch(html, /Smoothing rounds|Result<|Search path|<svg/);
  assert.match(html, /<h3>Smoothing by term<\/h3>/);
});

test("the term matrix tints the fold that went straight and says why on hover", () => {
  const html = termMatrixMarkup(book(), "fold-2");
  assert.match(html, /data-shape="straight" title="smoothed to a straight line, smoothing 3\.60e\+6, at the largest smoothing allowed">1\.00<span class="diagnose-limit"/);
  assert.match(html, /data-shape="curve" title="shaped by the data, smoothing 1\.05">3\.48/);
  assert.match(html, /data-shape="unsmoothed" title="no smoothing">5\.00/);
  assert.ok(html.indexOf(">Average<") < html.indexOf(">Spread<"));
});

test("a term whose penalty leaves more than a line says simplest shape, tinted", () => {
  const simplest = fit("fold-1", "Fold 1", { terms: { "theta:age": { edf: 2, lambda: 1e10, shape: "simplest", at_limit: false } } });
  assert.match(termMatrixMarkup(payload([simplest]), "fold-1"), /data-shape="simplest" title="flattened to the simplest shape its penalty allows, smoothing 1\.00e\+10">2\.00/);
});

test("a fit that released its design says its smoothing was not kept", () => {
  const released = fit("model", "All rows", { terms: { "mean:area": { edf: 2.5, lambda: null, shape: "unrecorded", at_limit: false } } });
  assert.match(termMatrixMarkup(payload([released]), "model"), /title="smoothing not kept \(this fit used retain_fit_state=False\)">2\.50/);
});

test("the term table keys its numbers, its shading and its dots in one line, neutral not warning", () => {
  const html = termMatrixMarkup(book(), "fold-2");
  assert.match(html, /EDF \(effective degrees of freedom, how flexible a term is\)\. Shaded: a straight line or smoothed away\. Dot: at the largest smoothing allowed\./);
  assert.equal((html.match(/EDF \(effective degrees of freedom/g) ?? []).length, 1);
});

test("a term one fit lacks reads as a dash", () => {
  const partial = fit("fold-1", "Fold 1", { terms: { "mean:area": { edf: 5, lambda: null, shape: "unsmoothed", at_limit: false } } });
  assert.match(termMatrixMarkup(payload([partial]), "fold-1"), /<td class="diagnose-cell">--<\/td>/);
});

test("the search path draws one line per fit and the chosen one last", () => {
  const html = searchPathMarkup(book(), "fold-1");
  assert.equal((html.match(/class="diagnose-line"/g) ?? []).length, 4);
  const lines = [...html.matchAll(/data-selected="(true|false)"/g)].map((match) => match[1]);
  assert.equal(lines.at(-1), "true");
  assert.equal(lines.filter((value) => value === "true").length, 1);
});

test("the search path puts a settled round on the floor, not below the chart", () => {
  const html = searchPathMarkup(payload([fit("fold-1", "Fold 1", { path: [5, 5, 5] })]), "fold-1");
  assert.doesNotMatch(html, /NaN|Infinity/);
});

/** The y of a line's last point, from its path data. @param {string} html @param {string} label */
function endY(html, label) {
  const path = html.match(new RegExp(`<path class="diagnose-line"[^>]* d="([^"]+)"[^>]*><title>${label}`));
  const points = /** @type {RegExpMatchArray} */ (path)[1].split(/[ML]/).filter(Boolean);
  return Number(points[points.length - 1].split(",")[1]);
}

test("a search that settled ends on the floor; one that stopped while still gaining ends high and is crossed", () => {
  // Both lose REML slowly then both stop. The first one's last rounds gain nothing; the second's last round still gains 0.5.
  const settled = fit("fold-1", "Fold 1", { path: [20, 10, 5, 4.9999, 4.9999, 4.9999] });
  const unfinished = fit("fold-2", "Fold 2", { status: "not_converged", converged: false, path: [20, 10, 5, 4.5] });
  const html = searchPathMarkup(payload([settled, unfinished]), "fold-2");
  const floor = endY(html, "Fold 1");
  assert.ok(endY(html, "Fold 2") < floor - 20, "the unfinished search's last point sits well above the floor");
  assert.equal((html.match(/class="diagnose-end"/g) ?? []).length, 1, "only the unfinished search is crossed");
  assert.match(html, /Fold 2: stopped before it converged \(its last round gained 0\.5\)/);
  assert.match(html, /Fold 1: 12 smoothing rounds/);
  assert.match(html, />REML score gained per round</);
  assert.match(html, />Smoothing round</);
});

test("a one-round search is a visible point, and the chart takes the width it is given", () => {
  const html = searchPathMarkup(payload([fit("model", "All rows", { rounds: 1, path: [10, 9] })]), "model", 1700);
  assert.match(html, /<circle class="diagnose-end"/);
  assert.match(html, /width="1700" height="300" viewBox="0 0 1700 300"/);
  assert.match(searchPathMarkup(payload([fit("model", "All rows")]), "model", 500), /height="180"/);
});

test("a fit with no search draws no search path", () => {
  assert.equal(searchPathMarkup(payload([fit("model", "All rows", { rounds: 0, path: [] })]), "model"), "");
});

test("a fit that searched but kept no path says so, beside any that kept one", () => {
  const alone = searchPathMarkup(payload([fit("model", "All rows", { path: [] })]), "model");
  assert.match(alone, /Not recorded by this fit's search: All rows\./);
  assert.doesNotMatch(alone, /<svg/);
  const beside = searchPathMarkup(payload([fit("model", "All rows", { path: [] }), fit("fold-1", "Fold 1")]), "model");
  assert.match(beside, /Not recorded by this fit's search: All rows\./);
  assert.equal((beside.match(/class="diagnose-line"/g) ?? []).length, 1);
});

const OPEN = { search: "open", phases: "open" };

test("phases give seconds and share, widest bar full, once the section is opened", () => {
  const timed = fit("model", "All rows", {
    phases: [{ name: "Likelihood evaluation", seconds: 1.5, share: 0.5 }, { name: "Other", seconds: 75, share: 0.25 }],
  });
  assert.doesNotMatch(phasesMarkup(timed), /Likelihood evaluation/);
  assert.match(phasesMarkup(timed), /data-diagnose-fold="phases" aria-expanded="false">Where the time went/);
  const html = phasesMarkup(timed, /** @type {import('../../src/superglm/editor/app/views/diagnose_tab.js').Sections} */ (OPEN));
  assert.match(html, /Likelihood evaluation/);
  assert.match(html, /width: 100\.0%/);
  assert.match(html, /width: 50\.0%/);
  assert.match(html, /1\.50 s/);
  assert.match(html, /1:15/);
  assert.match(html, /50%/);
});

test("a phase under five milliseconds has no row, and no row at all draws no section", () => {
  const quick = fit("model", "All rows", { phases: [{ name: "Tiny", seconds: 0.004, share: 0.01 }, { name: "Setting up the terms", seconds: 0.02, share: 0.5 }] });
  const html = phasesMarkup(quick, /** @type {any} */ (OPEN));
  assert.match(html, /Setting up the terms/);
  assert.doesNotMatch(html, /Tiny/);
  assert.equal(phasesMarkup(fit("model", "All rows", { phases: [{ name: "Tiny", seconds: 0.004, share: 1 }] }), /** @type {any} */ (OPEN)), "");
});

test("no phases or findings draw nothing, as a SuperGLM has neither", () => {
  assert.equal(phasesMarkup(fit("model", "All rows")), "");
  assert.equal(findingsMarkup(fit("model", "All rows")), "");
});

const finding = (/** @type {string} */ headline, /** @type {boolean} */ affects, /** @type {string} */ severity = "warning") =>
  ({ severity, headline, observed: "It moved.", question: "Is it real?", affects_numbers: affects });

test("a finding says its severity in words and escapes its text", () => {
  const html = findingsMarkup(fit("model", "All rows", { findings: [{ ...finding("Drift <up>", true), observed: "It moved <a lot>." }] }));
  assert.match(html, /diagnose-severity">warning/);
  assert.match(html, /Drift &lt;up&gt;/);
  assert.match(html, /It moved &lt;a lot&gt;\./);
  assert.match(html, /Is it real\?/);
});

test("findings that affect the numbers show first; those about the search are folded until opened", () => {
  const mixed = fit("fold-1", "Fold 1", {
    findings: [finding("The smoothing search did not converge.", true, "error"), finding("Power's curve moved most rounds.", false, "info"), finding("Smoothing steps were shortened.", false, "info")],
  });
  const folded = findingsMarkup(mixed);
  assert.match(folded, /Affects your numbers/);
  assert.match(folded, /did not converge/);
  assert.match(folded, /data-diagnose-fold="search" aria-expanded="false">About the search \(2\)/);
  assert.doesNotMatch(folded, /moved most rounds|shortened/);
  const open = findingsMarkup(mixed, /** @type {any} */ (OPEN));
  assert.match(open, /aria-expanded="true">About the search \(2\)/);
  assert.match(open, /moved most rounds[\s\S]*shortened/);
  assert.ok(open.indexOf("did not converge") < open.indexOf("moved most rounds"));
  // Only the findings that bear on the numbers carry the question to check.
  assert.equal((open.match(/Is it real\?/g) ?? []).length, 1);
  const quiet = findingsMarkup(fit("fold-1", "Fold 1", { findings: [finding("Power's curve moved most rounds.", false, "info")] }));
  assert.doesNotMatch(quiet, /Affects your numbers/);
  assert.match(quiet, /About the search \(1\)/);
});

test("a section folds and opens through one table, and only that table moves it", () => {
  assert.deepEqual([...SECTION.states].sort(), ["folded", "open"]);
  assert.deepEqual([...SECTION.events].sort(), ["fold", "open"]);
  assert.equal(SECTION.next("folded", "open"), "open");
  assert.equal(SECTION.next("open", "fold"), "folded");
  assert.equal(SECTION.next("folded", "fold"), null);
  assert.equal(SECTION.initial, "folded");
  const once = moveSection(FOLDED, "phases");
  assert.deepEqual(once, { search: "folded", phases: "open" });
  assert.deepEqual(moveSection(once, "phases"), FOLDED);
  assert.deepEqual(FOLDED, { search: "folded", phases: "folded" }, "a move returns a new value");
});

test("worst rows list a fold's held-out rows, and the training rows of the fit on all rows", () => {
  const data = book();
  assert.equal(worstRowsMarkup(data, data.fits[0]), "");
  const whole = fit("model", "All rows", {
    worst_rows: { columns: ["age"], rows: [{ row: 7, cost: 3.25, response: 4, predicted: 0.5, weight: null, values: [41] }] },
  });
  const wholeHtml = worstRowsMarkup(data, whole);
  assert.match(wholeHtml, /<td>7<\/td><td>3\.250<\/td>/);
  // The fit saw its training rows, and says so; a fold's rows were held out.
  assert.match(wholeHtml, /The training rows this fit predicted worst, by log score\. It was fitted on them/);
  const withRows = fit("fold-1", "Fold 1", {
    worst_rows: { columns: ["age", "area"], rows: [{ row: 19471, cost: 12.5, response: 11, predicted: 0.05, weight: 1, values: [65, "A"] }] },
  });
  const html = worstRowsMarkup(data, withRows);
  assert.match(html, /<td>19471<\/td><td>12\.500<\/td>/);
  assert.match(html, /<th>Log score<\/th>/);
  assert.match(html, /<td>A<\/td>/);
  assert.match(html, /<th>age<\/th><th>area<\/th>/);
  assert.match(worstRowsMarkup(data, fit("fold-1", "Fold 1")), /not available here/);
});

test("the tab opens on the first fit that did not converge, before the slowest fold", () => {
  const data = failing();
  assert.equal(chosenFit(data, "").id, "fold-1");
  assert.equal(chosenFit(data, "fold-3").id, "fold-3");
  // Without a failure it is still the slowest fold; and this model, when it is the one that failed.
  assert.equal(chosenFit(book(), "").id, "fold-2");
  const modelFailed = payload([fit("model", "All rows", { status: "not_converged" }), ...failing().fits.slice(1)]);
  assert.equal(chosenFit(modelFailed, "").id, "model");
});

test("the whole tab draws from one payload, the verdicts and the search path first", () => {
  const html = diagnoseTabMarkup(failing(), "");
  for (const title of ["Fits", "Search path", "What the fit reports", "Smoothing by term", "Rows that cost the most"]) {
    if (title === "What the fit reports") continue;
    assert.ok(html.includes(`<h3>${title}</h3>`), title);
  }
  assert.ok(html.indexOf("<h3>Fits</h3>") < html.indexOf("<h3>Search path</h3>"));
  assert.ok(html.indexOf("<h3>Search path</h3>") < html.indexOf("<h3>Smoothing by term</h3>"));
  // The chosen fit's way out, in Python's words, follows the chart.
  assert.match(html, /<p class="diagnose-explain"><strong>Fold 1:<\/strong> SuperLSS fit did not converge: refit with a larger max_reml_iter\./);
  assert.ok(html.indexOf("<h3>Search path</h3>") < html.indexOf("<h3>What to try</h3>"));
  assert.doesNotMatch(html, /undefined|NaN/);
  assert.doesNotMatch(diagnoseTabMarkup(book(), ""), /diagnose-explain/);
});

test("the sentence every converged fit shares is said once above the table, not on each row", () => {
  const html = fitsTableMarkup(book(), "fold-2");
  assert.equal((html.match(/The smoothing search reached an optimum and passed its own check\./g) ?? []).length, 1);
  assert.equal((html.match(/<strong>Converged\.<\/strong>/g) ?? []).length, 4);
  // Every row still leads with the word that says it.
  assert.match(html, /data-status="converged"/);
});

test("a fit whose verdict differs keeps its own text, and the shared sentence is said for the converged ones", () => {
  const html = fitsTableMarkup(failing(), "fold-1");
  assert.match(html, /<strong>Did not converge\.<\/strong> The numbers are from the last step reached/);
  assert.match(html, /<strong>Converged, with a looser stop\.<\/strong> The numbers are usable\./);
  // Only "All rows" converged outright here, so nothing is shared and its sentence stays on its row.
  assert.equal((html.match(/The smoothing search reached an optimum/g) ?? []).length, 1);
  const more = payload([...failing().fits, fit("fold-4", "Fold 4")]);
  const shared = fitsTableMarkup(more, "fold-1");
  // Two fits converged outright, so the sentence moves from their rows to the line above.
  assert.equal((shared.match(/The smoothing search reached an optimum/g) ?? []).length, 0);
  assert.equal((shared.match(/the smoothing search reached an optimum/g) ?? []).length, 1);
  assert.match(summaryMarkup(more.fits), /For the fits that converged: the smoothing search reached an optimum and passed its own check\./);
});

test("the shared sentence is a converged fit's, never a fixed or failed one's", () => {
  const fixed = (/** @type {string} */ id) => fit(id, id, { status: "fixed", verdict: "No smoothing search. You set the smoothing." });
  assert.equal(sharedConvergedSentence([fixed("a"), fixed("b")]), "");
  assert.equal(sharedConvergedSentence([fit("a", "A")]), "");
  assert.equal(sharedConvergedSentence(book().fits), "The smoothing search reached an optimum and passed its own check.");
});
