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

const LSS_REPORT = {
  report: "validation",
  available: true,
  metric_labels: { log_score: "Log score", crps: "CRPS", actual_expected: "Actual / expected" },
  splits: [
    split(
      { log_score: 1.2345, crps: null, actual_expected: 1.01 },
      { log_score: -0.05, crps: null, actual_expected: 0.01 },
    ),
  ],
  cv_report: null,
};

test("a SuperLSS report has a column per metric it names, and a null score shows a dash", () => {
  const html = render(LSS_REPORT);
  const headers = [...html.matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.deepEqual(headers, ["Split", "Log score", "CRPS", "Actual / expected"]);
  assert.doesNotMatch(html, /NaN|null|undefined/);
  const crps = html.split("<td>")[3];
  assert.match(crps, /^\s*--\s*<span class="report-delta" data-direction="neutral">Δ --<\/span>/);
});

test("a lower log score reads as better", () => {
  assert.match(render(LSS_REPORT), /data-direction="better">Δ -0\.05</);
});

test("a SuperGLM report keeps its own columns", () => {
  const html = render({
    report: "validation",
    available: true,
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
    metric_labels: { deviance: "Deviance", effective_df: "EDF" },
    splits: [split({ deviance: null, effective_df: 4 }, { deviance: null, effective_df: 0 })],
    cv_report: null,
  });
  const deviance = html.split("<td>")[2];
  assert.match(deviance, /^\s*--\s*<span class="report-delta" data-direction="neutral">Δ --<\/span>/);
  assert.doesNotMatch(html, /NaN|null|undefined/);
});
