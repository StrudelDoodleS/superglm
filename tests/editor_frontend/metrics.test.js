// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { renderMetricGrid } from "../../src/superglm/editor/app/metrics.js";

const KEYS = ["deviance", "aic", "bic", "log_likelihood", "explained_deviance", "pearson_chi2", "effective_df"];

function render(payload) {
  const nodes = { metricGrid: { innerHTML: "", textContent: "" }, metricSelect: { options: [] } };
  renderMetricGrid(payload, nodes);
  return nodes.metricGrid.innerHTML;
}

const scores = (value) => Object.fromEntries(KEYS.map((key) => [key, value]));

test("an edited model is compared with the original's scores", () => {
  const html = render({ available: true, dataset_label: "Validation", metrics: { original: scores(2), edited: scores(3) } });
  assert.match(html, /change against the original model/);
  assert.match(html, /<span>\+1<\/span>/);
  assert.match(html, /was 2</);
});

// An edit that restores the quotes the original model lacks: its scores show, with no change.
test("an original with no scores shows the edited ones and no change", () => {
  const html = render({
    available: true,
    dataset_label: "Validation",
    note: "The original model has no quote for some of these rows, so the change against it is not shown.",
    metrics: { original: scores(null), edited: scores(3) },
  });
  assert.match(html, /Validation data\. The original model has no quote/);
  assert.match(html, /<span class="metric-item-value">3<\/span>/);
  assert.match(html, /was --/);
  assert.doesNotMatch(html, /\+3|NaN|null|undefined|metric-arrow/);
  assert.doesNotMatch(html, /data-direction="(better|worse)"/);
});
