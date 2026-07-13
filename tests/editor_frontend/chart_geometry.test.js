import assert from "node:assert/strict";
import test from "node:test";

import {
  evenlySpacedIndices,
  fitMeasuredLabel,
  planCategoricalAxis,
  rotatedExtent,
  splitLabelGraphemes,
} from "../../src/superglm/editor/app/chart/geometry.js";

/**
 * @param {string} label
 * @param {number} [widthPerGrapheme]
 * @param {number} [height]
 * @returns {import('../../src/superglm/editor/app/chart/geometry.js').LabelMeasurement}
 */
function measurement(label, widthPerGrapheme = 7, height = 11) {
  const graphemes = splitLabelGraphemes(label);
  return {
    fullWidth: graphemes.length * widthPerGrapheme,
    prefixWidths: graphemes.map((_, index) => (index + 1) * widthPerGrapheme),
    ellipsisWidth: widthPerGrapheme,
    height,
  };
}

test("tick reduction retains first, last, and evenly spaced interior categories", () => {
  assert.deepEqual(evenlySpacedIndices(10, 5), [0, 2, 5, 7, 9]);
  assert.deepEqual(evenlySpacedIndices(3, 5), [0, 1, 2]);
});

test("tick reduction handles empty, single, exact-limit, and thirty-cap inputs", () => {
  assert.deepEqual(evenlySpacedIndices(0, 5), []);
  assert.deepEqual(evenlySpacedIndices(5, 0), []);
  assert.deepEqual(evenlySpacedIndices(1, 30), [0]);
  assert.deepEqual(evenlySpacedIndices(8, 1), [0]);
  assert.deepEqual(evenlySpacedIndices(30, 30), Array.from({ length: 30 }, (_, i) => i));

  const capped = evenlySpacedIndices(100, 30);
  assert.equal(capped.length, 30);
  assert.equal(capped[0], 0);
  assert.equal(capped.at(-1), 99);
  assert.equal(new Set(capped).size, capped.length);
});

test("measured truncation uses a Unicode end ellipsis without changing the source", () => {
  const source = "MyReallyLongCategoryNameThatWouldNeverFit";
  const fitted = fitMeasuredLabel(source, measurement(source), 112);
  assert.equal(fitted, "MyReallyLongCat…");
  assert.equal(source, "MyReallyLongCategoryNameThatWouldNeverFit");
});

test("measured truncation respects exact and sub-ellipsis budgets", () => {
  const source = "ABCDE";
  const measured = measurement(source);
  assert.equal(fitMeasuredLabel(source, measured, 35), source);
  assert.equal(fitMeasuredLabel(source, measured, 21), "AB…");
  assert.equal(fitMeasuredLabel(source, measured, 7), "…");
  assert.equal(fitMeasuredLabel(source, measured, 6), "");
});

test("truncation follows measured variable-width prefixes", () => {
  assert.equal(
    fitMeasuredLabel(
      "Wide",
      {
        fullWidth: 21,
        prefixWidths: [9, 12, 20, 21],
        ellipsisWidth: 4,
        height: 11,
      },
      16,
    ),
    "Wi…",
  );
});

test("grapheme segmentation keeps combining and joined emoji intact", () => {
  const family = "👨‍👩‍👧‍👦";
  const source = `Ae\u0301${family}ZY`;
  assert.deepEqual(splitLabelGraphemes(source), ["A", "e\u0301", family, "Z", "Y"]);
  assert.equal(fitMeasuredLabel(source, measurement(source, 10), 40), `Ae\u0301${family}…`);
});

test("rotated extent projects width into the bottom gutter", () => {
  const extent = rotatedExtent(100, 11, -45);
  assert.ok(extent.width > 70 && extent.width < 80);
  assert.ok(extent.height > 70 && extent.height < 80);
  assert.deepEqual(rotatedExtent(100, 11, 0), { width: 100, height: 11 });
  const rightAngle = rotatedExtent(100, 11, 90);
  assert.ok(Math.abs(rightAngle.width - 11) < 1e-9);
  assert.ok(Math.abs(rightAngle.height - 100) < 1e-9);
});

test("categorical layout reserves title space and preserves full labels", () => {
  const labels = Array.from({ length: 10 }, (_, index) =>
    `TerritoryCategoryNumber${index + 1}`
  );
  const layout = planCategoricalAxis({
    values: labels.map((_, index) => index),
    labels,
    measurements: labels.map((label) => measurement(label)),
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.equal(layout.ticks[0].fullLabel, labels[0]);
  assert.equal(layout.ticks.at(-1)?.fullLabel, labels.at(-1));
  assert.ok(layout.ticks.some((tick) => tick.displayLabel.endsWith("…")));
  assert.ok(layout.bottom > 72);
  assert.ok(layout.titleY > layout.axisY + layout.maxLabelHeight);
  assert.ok(layout.titleY + layout.titleHeight <= 520 - 12);
});

test("empty and single-category layouts remain finite and bounded", () => {
  const empty = planCategoricalAxis({
    values: [],
    labels: [],
    measurements: [],
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.deepEqual(empty.ticks, []);
  assert.ok(Number.isFinite(empty.axisY));
  assert.ok(empty.titleY + empty.titleHeight <= 520 - 12);

  const single = planCategoricalAxis({
    values: ["only"],
    labels: ["Only category"],
    measurements: [measurement("Only category")],
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.equal(single.ticks.length, 1);
  assert.equal(single.ticks[0].fullLabel, "Only category");
  assert.ok(single.titleY + single.titleHeight <= 520 - 12);
});

test("categorical layout caps one hundred categories at thirty unique ticks", () => {
  const labels = Array.from({ length: 100 }, (_, index) => `Category ${index + 1}`);
  const layout = planCategoricalAxis({
    values: labels.map((_, index) => index),
    labels,
    measurements: labels.map((label) => measurement(label)),
    availableWidth: 4000,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.equal(layout.ticks.length, 30);
  assert.equal(layout.ticks[0].index, 0);
  assert.equal(layout.ticks.at(-1)?.index, 99);
  assert.equal(new Set(layout.ticks.map((tick) => tick.index)).size, 30);
});

test("horizontal centered edge labels respect the viewport-side budget", () => {
  const labels = ["FourteenCharsAB", "FourteenCharsCD"];
  const layout = planCategoricalAxis({
    values: [0, 1],
    labels,
    measurements: labels.map((label) => measurement(label, 10)),
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  });
  assert.equal(layout.ticks[0].angle, 0);
  assert.ok(layout.ticks.every((tick) => tick.displayLabel.endsWith("…")));
  assert.ok(layout.ticks.every((tick) => tick.width <= 128));
});

test("angled budgeting uses the tallest selected measurement", () => {
  const labels = ["First long category", "Second long category"];
  const base = {
    values: [0, 1],
    labels,
    availableWidth: 100,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
  };
  const even = planCategoricalAxis({
    ...base,
    measurements: labels.map((label) => measurement(label, 7, 8)),
  });
  const varied = planCategoricalAxis({
    ...base,
    measurements: [measurement(labels[0], 7, 8), measurement(labels[1], 7, 30)],
  });
  assert.equal(varied.ticks[0].angle, -45);
  assert.ok(varied.labelBudget < even.labelBudget);
  assert.ok(varied.maxLabelHeight > even.maxLabelHeight);
  assert.ok(varied.titleY + varied.titleHeight <= 520 - 12);
});

test("geometry rejects mismatched arrays and malformed measurements", () => {
  assert.throws(
    () => planCategoricalAxis({
      values: [0],
      labels: ["one", "two"],
      measurements: [measurement("one")],
      availableWidth: 788,
      svgHeight: 520,
      baseLeft: 76,
      baseBottom: 72,
    }),
    /same length/,
  );
  assert.throws(
    () => fitMeasuredLabel(
      "two",
      { fullWidth: 14, prefixWidths: [7], ellipsisWidth: 7, height: 11 },
      10,
    ),
    /prefixWidths/,
  );
  assert.throws(
    () => fitMeasuredLabel(
      "A",
      { fullWidth: -1, prefixWidths: [7], ellipsisWidth: 7, height: 11 },
      10,
    ),
    /fullWidth/,
  );
});

test("geometry rejects nonfinite and negative dimensions", () => {
  assert.throws(() => evenlySpacedIndices(Number.NaN, 2), /count/);
  assert.throws(() => evenlySpacedIndices(2, Number.POSITIVE_INFINITY), /maximum/);
  assert.throws(() => evenlySpacedIndices(-1, 2), /count/);
  assert.throws(() => fitMeasuredLabel("A", measurement("A"), Number.NaN), /budget/);
  assert.throws(() => fitMeasuredLabel("A", measurement("A"), -1), /budget/);
  assert.throws(() => rotatedExtent(Number.POSITIVE_INFINITY, 10, 0), /width/);
  assert.throws(() => rotatedExtent(10, -1, 0), /height/);
  assert.throws(() => rotatedExtent(10, 10, Number.NaN), /degrees/);
  assert.throws(
    () => planCategoricalAxis({
      values: [0],
      labels: ["one"],
      measurements: [measurement("one")],
      availableWidth: -1,
      svgHeight: 520,
      baseLeft: 76,
      baseBottom: 72,
    }),
    /availableWidth/,
  );
  assert.throws(
    () => planCategoricalAxis({
      values: [0],
      labels: ["one"],
      measurements: [measurement("one")],
      availableWidth: 788,
      svgHeight: Number.POSITIVE_INFINITY,
      baseLeft: 76,
      baseBottom: 72,
    }),
    /svgHeight/,
  );
});
