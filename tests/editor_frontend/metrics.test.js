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

/** Each metric's rendered text, as tokens, and the direction it is coloured by. */
function metricItems(html) {
  return html.split(/<div class="metric-item"(?=[\s>])/).slice(1).map((item) => ({
    direction: /data-direction="([^"]*)"/.exec(item)?.[1],
    drawsArrow: /<svg/.test(item),
    tokens: item.replace(/<[^>]*>/g, " ").replace(/^[^>]*>/, " ").split(/\s+/).filter(Boolean),
  }));
}

const NUMBER = /^[+\u2212-]?\d/;

test("an edited model is compared with the original's scores", () => {
  const html = render({ available: true, dataset_label: "Validation", metrics: { original: scores(2), edited: scores(3) } });
  assert.match(html, /change against the original/i);
  const items = metricItems(html);
  assert.equal(items.length, KEYS.length);
  for (const { tokens } of items) {
    // the edited value, the original it was, and the signed change between them
    assert.deepEqual(tokens.filter((token) => NUMBER.test(token)).sort(), ["+1", "2", "3"]);
  }
});

// An edit that restores the quotes the original model lacks: its scores show, with no change.
test("an original with no scores shows the edited ones and no change", () => {
  const html = render({
    available: true,
    dataset_label: "Validation",
    note: "The original model has no quote for some of these rows, so the change against it is not shown.",
    metrics: { original: scores(null), edited: scores(3) },
  });
  assert.match(html, /The original model has no quote/);
  const items = metricItems(html);
  assert.equal(items.length, KEYS.length);
  for (const { direction, drawsArrow, tokens } of items) {
    // the edited value alone: no original, no change, and nothing coloured better or worse
    assert.deepEqual(tokens.filter((token) => NUMBER.test(token)), ["3"]);
    assert.ok(!tokens.some((token) => /NaN|null|undefined/.test(token)), tokens.join(" "));
    assert.ok(!["better", "worse"].includes(direction) && !drawsArrow, direction);
  }
});
