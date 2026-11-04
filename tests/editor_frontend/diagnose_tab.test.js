// @ts-check
import assert from "node:assert/strict";
import test from "node:test";

import {
  chosenFit,
  diagnoseTabMarkup,
  findingsMarkup,
  fitsTableMarkup,
  phasesMarkup,
  searchPathMarkup,
  slowFits,
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
    stop_text: "REML stopped improving",
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
const book = () => payload([fit("model", "This model"), fit("fold-1", "Fold 1"), flat, fit("fold-3", "Fold 3", { rounds: 13 })]);

test("a fold with at least twice the middle fold's rounds is slow", () => {
  assert.deepEqual([...slowFits(book().fits)], ["fold-2"]);
});

test("rounds below ten are not slow, however they compare", () => {
  const quick = [fit("model", "This model"), fit("fold-1", "Fold 1", { rounds: 2 }), fit("fold-2", "Fold 2", { rounds: 9 }), fit("fold-3", "Fold 3", { rounds: 3 })];
  assert.equal(slowFits(quick).size, 0);
});

test("two folds are too few to call one slow", () => {
  const pair = [fit("model", "This model"), fit("fold-1", "Fold 1", { rounds: 10 }), fit("fold-2", "Fold 2", { rounds: 80 })];
  assert.equal(slowFits(pair).size, 0);
});

test("this model's rounds do not make a fold slow, nor is it one", () => {
  const fits = [fit("model", "This model", { rounds: 500 }), fit("fold-1", "Fold 1"), fit("fold-2", "Fold 2"), fit("fold-3", "Fold 3")];
  assert.equal(slowFits(fits).size, 0);
});

test("the tab opens on the slowest fold unless one is asked for", () => {
  assert.equal(chosenFit(book(), "").id, "fold-2");
  assert.equal(chosenFit(book(), "fold-1").id, "fold-1");
  assert.equal(chosenFit(book(), "fold-9").id, "fold-2");
  assert.equal(chosenFit(payload([fit("model", "This model")]), "").id, "model");
});

test("the tab opens on the fold with the most rounds, marked slow or not", () => {
  const rounds = (/** @type {number[]} */ counts) => payload([
    fit("model", "This model", { rounds: 500 }),
    ...counts.map((count, index) => fit(`fold-${index + 1}`, `Fold ${index + 1}`, { rounds: count })),
  ]);
  assert.equal(chosenFit(rounds([12, 25, 40, 12, 12]), "").id, "fold-3");
  assert.equal(chosenFit(rounds([12, 14, 15]), "").id, "fold-3");
  assert.equal(chosenFit(rounds([0, 0, 0]), "").id, "model");
});

test("a fold keeps its own colour when an earlier fold failed", () => {
  const html = fitsTableMarkup(payload([fit("model", "This model"), fit("fold-2", "Fold 2")]), "model");
  assert.match(html, new RegExp(`background: ${foldColour(1).replace(/[()]/g, "\\$&")}`));
  assert.doesNotMatch(html, new RegExp(`background: ${foldColour(0).replace(/[()]/g, "\\$&")}`));
});

test("the fits table names why each search stopped and marks the slow fold", () => {
  const html = fitsTableMarkup(book(), "fold-2");
  assert.match(html, /REML stopped improving/);
  assert.match(html, /18\.0 s/);
  assert.equal((html.match(/diagnose-slow/g) ?? []).length, 1);
  assert.match(html, /data-diagnose-fit="fold-2" aria-pressed="true"/);
  assert.match(html, /data-diagnose-fit="fold-1" aria-pressed="false"/);
});

test("a fit with no timing or rows reads as dashes", () => {
  const html = fitsTableMarkup(payload([fit("model", "This model", { rows: null, seconds: null, late_gain: null, settled_round: null })]), "model");
  assert.equal((html.match(/<td>--<\/td>/g) ?? []).length, 4);
});

test("the term matrix tints the fold that went straight and says why on hover", () => {
  const html = termMatrixMarkup(book(), "fold-2");
  assert.match(html, /data-shape="straight" title="smoothed to a straight line, smoothing 3\.60e\+6, at the smoothing limit">1\.00<span class="diagnose-limit"/);
  assert.match(html, /data-shape="curve" title="shaped by the data, smoothing 1\.05">3\.48/);
  assert.match(html, /data-shape="unsmoothed" title="no smoothing">5\.00/);
  assert.ok(html.indexOf(">Average<") < html.indexOf(">Spread<"));
});

test("a term whose penalty leaves more than a line says simplest shape, tinted", () => {
  const simplest = fit("fold-1", "Fold 1", { terms: { "theta:age": { edf: 2, lambda: 1e10, shape: "simplest", at_limit: false } } });
  assert.match(termMatrixMarkup(payload([simplest]), "fold-1"), /data-shape="simplest" title="smoothed to the simplest shape its penalty allows, smoothing 1\.00e\+10">2\.00/);
});

test("a fit that released its design says its smoothing was not kept", () => {
  const released = fit("model", "This model", { terms: { "mean:area": { edf: 2.5, lambda: null, shape: "unrecorded", at_limit: false } } });
  assert.match(termMatrixMarkup(payload([released]), "model"), /title="smoothing not kept: this fit used retain_fit_state=False">2\.50/);
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

test("a fit with no search draws no search path", () => {
  assert.equal(searchPathMarkup(payload([fit("model", "This model", { rounds: 0, path: [] })]), "model"), "");
});

test("a fit that searched but kept no path says so, beside any that kept one", () => {
  const alone = searchPathMarkup(payload([fit("model", "This model", { path: [] })]), "model");
  assert.match(alone, /Not recorded by this fit's search: This model\./);
  assert.doesNotMatch(alone, /<svg/);
  const beside = searchPathMarkup(payload([fit("model", "This model", { path: [] }), fit("fold-1", "Fold 1")]), "model");
  assert.match(beside, /Not recorded by this fit's search: This model\./);
  assert.equal((beside.match(/class="diagnose-line"/g) ?? []).length, 1);
});

test("phases give seconds and share, widest bar full", () => {
  const html = phasesMarkup(fit("model", "This model", {
    phases: [{ name: "Likelihood evaluation", seconds: 1.5, share: 0.5 }, { name: "Other", seconds: 75, share: 0.25 }],
  }));
  assert.match(html, /Likelihood evaluation/);
  assert.match(html, /width: 100\.0%/);
  assert.match(html, /width: 50\.0%/);
  assert.match(html, /1\.50 s/);
  assert.match(html, /1:15/);
  assert.match(html, /50%/);
});

test("no phases or findings draw nothing, as a SuperGLM has neither", () => {
  assert.equal(phasesMarkup(fit("model", "This model")), "");
  assert.equal(findingsMarkup(fit("model", "This model")), "");
});

test("a finding says its severity in words and escapes its text", () => {
  const html = findingsMarkup(fit("model", "This model", {
    findings: [{ severity: "warning", headline: "Drift <up>", observed: "It moved.", question: "Is it real?" }],
  }));
  assert.match(html, /diagnose-severity">warning/);
  assert.match(html, /Drift &lt;up&gt;/);
  assert.match(html, /Is it real\?/);
});

test("worst rows list a fold's rows; this model points at the folds", () => {
  const data = book();
  assert.match(worstRowsMarkup(data, data.fits[0]), /pick a fold/);
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

test("the whole tab draws from one payload", () => {
  const html = diagnoseTabMarkup(book(), "");
  for (const title of ["Fits", "Smoothing by term", "Search path", "Rows that cost the most"]) {
    assert.ok(html.includes(`<h3>${title}</h3>`), title);
  }
  assert.doesNotMatch(html, /undefined|NaN/);
});
