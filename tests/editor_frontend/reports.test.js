// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { renderReport } from "../../src/superglm/editor/app/reports.js";

function render(payload) {
  const nodes = { reportTitle: {}, reportStatus: {}, reportFrame: { innerHTML: "" } };
  renderReport(payload, nodes);
  return nodes.reportFrame.innerHTML;
}

const split = (edited, delta) => ({
  name: "validation", label: "Validation", n_obs: 200, metrics: { edited, delta },
});

const LSS_LABELS = { log_score: "Log score", crps: "CRPS", actual_expected: "Actual / expected" };

// A report of an edited model: the changes are shown. CRPS is scored here.
const LSS_REPORT = {
  report: "validation",
  available: true,
  compares_edit: true,
  metric_labels: LSS_LABELS,
  splits: [
    split(
      { log_score: 1.2345, crps: 1009.8, actual_expected: 1.0012 },
      { log_score: -0.05, crps: 0.4, actual_expected: 0.0012 },
    ),
  ],
  cv_report: null,
};

// A count model has no CRPS: every split carries null for it.
const NO_CRPS = {
  ...LSS_REPORT,
  splits: [
    split(
      { log_score: 1.2345, crps: null, actual_expected: 1.01 },
      { log_score: -0.05, crps: null, actual_expected: 0.01 },
    ),
  ],
};

test("a SuperLSS report has a column per metric it names", () => {
  const html = render(LSS_REPORT);
  const headers = [...html.matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.deepEqual(headers, ["Split", "Log score", "CRPS", "Actual / expected"]);
  assert.doesNotMatch(html, /NaN|null|undefined/);
});

test("a family with no CRPS has no CRPS column, as on the Cross-validation tab", () => {
  const html = render(NO_CRPS);
  const headers = [...html.matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.deepEqual(headers, ["Split", "Log score", "Actual / expected"]);
  assert.doesNotMatch(html, /--|NaN|null|undefined/);
  // One split that can score it keeps the column, and the one that cannot shows a dash there.
  const mixed = render({ ...LSS_REPORT, splits: [...LSS_REPORT.splits, ...NO_CRPS.splits.map((row) => ({ ...row, name: "test", label: "Test" }))] });
  assert.match(mixed, /<th>CRPS<\/th>/);
  assert.match(mixed.split("<td>")[7], /^\s*--/);
});

test("actual over expected is shown to four decimals, and so is its change", () => {
  const html = render(LSS_REPORT);
  const cell = html.split("<td>")[4];
  assert.match(cell, /^\s*1\.0012\s*<span class="report-delta" data-direction="neutral">Δ \+0\.0012<\/span>/);
  assert.match(render({ ...LSS_REPORT, splits: [split({ log_score: 1, crps: 2, actual_expected: 1 }, { log_score: 0, crps: 0, actual_expected: 0 })] }), /1\.0000/);
});

test("the changes are shown only where an edited model is compared with the opened one", () => {
  assert.match(render(LSS_REPORT), /Δ -0\.05/);
  const unedited = render({ ...LSS_REPORT, compares_edit: false });
  assert.doesNotMatch(unedited, /Δ|report-delta" data-direction/);
  assert.match(unedited, /1\.2345|1\.23/);
  assert.match(unedited, /1\.0012/);
});

test("row counts carry thousands separators, as elsewhere in the editor", () => {
  const big = { ...LSS_REPORT, splits: [{ ...LSS_REPORT.splits[0], n_obs: 678007 }] };
  assert.match(render(big), /678,007 rows/);
  assert.doesNotMatch(render(big), /678007/);
});

test("a lower log score reads as better", () => {
  assert.match(render(LSS_REPORT), /data-direction="better">Δ -0\.05</);
});

test("a SuperGLM report keeps its own columns", () => {
  const html = render({
    report: "validation",
    available: true,
    compares_edit: true,
    metric_labels: { deviance: "Deviance", pearson_chi2: "Pearson chi2", effective_df: "EDF" },
    splits: [split({ deviance: 10, effective_df: 4 }, { deviance: -1, effective_df: 0 })],
    cv_report: null,
  });
  const headers = [...html.matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.equal(headers.length, 7);
  assert.deepEqual(headers.slice(0, 2), ["Split", "Deviance"]);
  assert.equal(headers.includes("Pearson chi2"), false);
});

test("a Final fit of a SuperLSS model with no summary lists its rows and not empty summary lines", () => {
  const html = render({
    report: "final",
    available: true,
    metric_labels: LSS_REPORT.metric_labels,
    splits: LSS_REPORT.splits,
    final_fit: { available: true, stale: false, n_rows: 900, splits: ["train", "validation"], carried: [], pending: 0, summary: null },
  });
  assert.match(html, /<th>Rows<\/th><td>900<\/td>/);
  assert.doesNotMatch(html, /Deviance<\/th>|AIC<\/th>|BIC<\/th>|NaN|null/);
});

const FINAL = { available: true, stale: false, n_rows: 678007, splits: ["train"], carried: [], pending: 0, summary: null };

test("a Final fit says the test split stays held out only when there is one", () => {
  const section = (finalFit) => render({ report: "final", available: true, metric_labels: LSS_LABELS, splits: LSS_REPORT.splits, final_fit: finalFit });
  const none = section({ ...FINAL, has_test_split: false });
  assert.match(none, /Refitted on 678,007 train rows\./);
  assert.doesNotMatch(none, /test split/);
  assert.match(none, /<th>Rows<\/th><td>678,007<\/td>/);
  assert.match(section({ ...FINAL, has_test_split: true }), /Refitted on 678,007 train rows; the test split stays held out\./);
});

test("a Final fit reports how the refit ended, in the sentence the Diagnose tab uses", () => {
  const verdict = "Converged. The smoothing search reached an optimum and passed its own check.";
  const html = render({
    report: "final", available: true, metric_labels: LSS_LABELS, splits: LSS_REPORT.splits,
    final_fit: { ...FINAL, has_test_split: false, fit_status: "converged", fit_verdict: verdict },
  });
  assert.match(html, /<tr><th>Fit<\/th><td data-fit-status="converged">Converged\. The smoothing search reached an optimum and passed its own check\.<\/td><\/tr>/);
  // A refit with no record of its search says nothing about one.
  assert.doesNotMatch(render({ report: "final", available: true, metric_labels: LSS_LABELS, splits: LSS_REPORT.splits, final_fit: { ...FINAL, fit_status: null, fit_verdict: null } }), /<th>Fit<\/th>/);
});

test("a split the model cannot score shows its reason across the metric columns", () => {
  const reason = "The Validation split cannot be scored with its weights. <Reason>.";
  const html = render({
    ...LSS_REPORT,
    splits: [
      { name: "validation", label: "Validation", n_obs: 100, available: false, error: reason },
      ...LSS_REPORT.splits.map((row) => ({ ...row, name: "test", label: "Test" })),
    ],
  });
  assert.match(
    html,
    /<td class="report-note" colspan="3">The Validation split cannot be scored with its weights\. &lt;Reason&gt;\.<\/td>/,
  );
  assert.doesNotMatch(html, /NaN|null|undefined/);
  // The other split keeps its metrics.
  assert.match(html, /data-direction="better">Δ -0\.05</);
});

test("a SuperGLM metric the report has no value for shows a dash and no change", () => {
  const html = render({
    report: "validation",
    available: true,
    compares_edit: true,
    metric_labels: { deviance: "Deviance", effective_df: "EDF" },
    splits: [split({ deviance: null, effective_df: 4 }, { deviance: null, effective_df: 0 })],
    cv_report: null,
  });
  const deviance = html.split("<td>")[2];
  assert.match(deviance, /^\s*--\s*<span class="report-delta" data-direction="neutral">Δ --<\/span>/);
  assert.doesNotMatch(html, /NaN|null|undefined/);
});

test("a Final fit lists its scores on the held-out test split, skipping one the model cannot compute", () => {
  const html = render({
    report: "final",
    available: true,
    metric_labels: LSS_REPORT.metric_labels,
    splits: LSS_REPORT.splits,
    final_fit: {
      available: true, stale: false, n_rows: 900, splits: ["train", "validation"], carried: [],
      test_scores: { log_score: 0.205, crps: null, actual_expected: 0.981 },
    },
  });
  assert.match(html, /<th>Test Log score<\/th><td>0\.205<\/td>/);
  assert.match(html, /<th>Test Actual \/ expected<\/th><td>0\.9810<\/td>/);
  assert.doesNotMatch(html, /Test CRPS|null|NaN/);
});

test("a validation report with no supplied CV report has no empty section for one", () => {
  assert.doesNotMatch(render(LSS_REPORT), /CV Report|No CV report/);
});
