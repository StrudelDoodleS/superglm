import assert from "node:assert/strict";
import test from "node:test";

import {
  FALLBACK_CHART_SIZE,
  categoricalTickIndices,
  chartSize,
  fitMeasuredLabel,
  planCategoricalAxis,
  rotatedExtent,
  splitLabelGraphemes,
  strideIndices,
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

test("tick strides handle empty, single, exact-limit, and thirty-cap inputs", () => {
  assert.deepEqual(strideIndices(5, 0), []);
  assert.deepEqual(strideIndices(1, 30), [0]);
  assert.deepEqual(strideIndices(30, 30), Array.from({ length: 30 }, (_, i) => i));
  assert.equal(strideIndices(100, 30).length, 25);
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

test("categorical layout caps one hundred categories at thirty ticks, one step apart", () => {
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
  const indices = layout.ticks.map((tick) => tick.index);
  assert.ok(indices.length <= 30);
  assert.equal(indices[0], 0);
  assert.equal(indices.at(-1), 99);
  const gaps = indices.slice(1).map((index, k) => index - indices[k]);
  const [step, last] = [gaps[0], gaps[gaps.length - 1]];
  assert.ok(gaps.slice(0, -1).every((gap) => gap === step) && last >= step, String(gaps));
});

test("labels never fall on neighbouring levels while others skip, so none collide", () => {
  // 24 ordered levels in room for 14 labels: rounding a 1.77 step labelled
  // levels 5 and 6, 12 and 13, 19 and 20 side by side, and they overlapped.
  assert.deepEqual(strideIndices(24, 14), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]);
  assert.deepEqual(strideIndices(10, 5), [0, 3, 6, 9]);
  assert.deepEqual(strideIndices(11, 5), [0, 3, 6, 10]);
  assert.deepEqual(strideIndices(3, 5), [0, 1, 2]);
  assert.deepEqual(strideIndices(8, 1), [0]);
  assert.deepEqual(strideIndices(0, 5), []);

  // Short labels stay level, centred one whole step apart.
  const labels = Array.from({ length: 24 }, (_, index) => `Mi${String(6 * (index + 1)).padStart(3, "0")}`);
  const ticks = planCategoricalAxis({
    values: labels.map((_, index) => index),
    labels,
    measurements: labels.map((label) => measurement(label)),
    availableWidth: 760,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
    domain: [-0.5, 23.5],
  }).ticks;
  assert.ok(ticks.every((tick) => tick.angle === 0));
  assert.deepEqual(ticks.map((tick) => tick.index), strideIndices(24, 14));
  assertNoLevelLabelOverlaps(ticks, 760, [-0.5, 23.5]);
});

test("label room is measured on the padded axis the chart draws", () => {
  // Five levels on a 400 px plot padded half a level each side sit 80 px
  // apart, not the 100 px the ticks' own span gives: an 88 px label kept
  // level overlapped each neighbour by 8 px.
  const labels = ["Level one A", "Level two B", "Level thr C", "Level fou D", "Level fiv E"];
  const ticks = planCategoricalAxis({
    values: labels.map((_, index) => index),
    labels,
    measurements: labels.map((label) => measurement(label, 8)),
    availableWidth: 400,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
    domain: [-0.5, 4.5],
  }).ticks;
  assertNoLevelLabelOverlaps(ticks, 400, [-0.5, 4.5]);
});

test("the chart labels more than thirty levels one whole step apart", () => {
  // The chart measures only the levels categoricalTickIndices picks, so the
  // stride must hold over all the levels, not over a rounded preselection.
  const wide = categoricalTickIndices(40, 1000);
  assert.deepEqual(wide, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]);
  assert.deepEqual(categoricalTickIndices(40, 760), wide);
  assert.deepEqual(categoricalTickIndices(100, 4000), strideIndices(100, 30));

  const labels = Array.from({ length: 40 }, (_, index) => `L${index}`);
  const picked = wide.map((index) => labels[index]);
  const ticks = planCategoricalAxis({
    values: wide,
    labels: picked,
    measurements: picked.map((label) => measurement(label)),
    availableWidth: 1000,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 72,
    domain: [-0.5, 39.5],
  }).ticks;
  // planCategoricalAxis labels every level it is given.
  assert.deepEqual(ticks.map((tick) => tick.value), wide);
});

/**
 * Each level label fits the room to its neighbours on the padded axis.
 * @param {readonly {value:unknown, angle:number, width:number, fullLabel:string}[]} ticks
 * @param {number} width @param {[number, number]} domain
 */
function assertNoLevelLabelOverlaps(ticks, width, [lo, hi]) {
  const pixel = (/** @type {number} */ value) => (width * (value - lo)) / (hi - lo);
  for (let k = 1; k < ticks.length; k += 1) {
    const room = pixel(Number(ticks[k].value)) - pixel(Number(ticks[k - 1].value));
    const halves = (ticks[k].width + ticks[k - 1].width) / 2;
    assert.ok(
      ticks[k].angle !== 0 || halves <= room,
      `${ticks[k - 1].fullLabel} and ${ticks[k].fullLabel}: ${halves} > ${room}`
    );
  }
}

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
  assert.throws(() => strideIndices(Number.NaN, 2), /count/);
  assert.throws(() => strideIndices(2, Number.POSITIVE_INFINITY), /maximum/);
  assert.throws(() => strideIndices(-1, 2), /count/);
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

test("chart size is the laid-out viewport in whole pixels, or the fallback without layout", () => {
  assert.deepEqual(chartSize(1241.6, 829.4), { width: 1242, height: 829 });
  assert.deepEqual(chartSize(694, 452), { width: 694, height: 452 });
  assert.deepEqual(FALLBACK_CHART_SIZE, { width: 940, height: 520 });
  for (const [width, height] of [[0, 0], [0, 452], [694, 0], [NaN, 452], [-1, 452]]) {
    assert.equal(chartSize(width, height), FALLBACK_CHART_SIZE);
  }
  assert.ok(Object.isFrozen(FALLBACK_CHART_SIZE));
});

test("a taller title row grows the bottom gutter and leaves the labels where they were", () => {
  const labels = ["T01", "T02", "T03"];
  /** @param {number} titleHeight */
  const plan = (titleHeight) => planCategoricalAxis({
    values: [0, 1, 2],
    labels,
    measurements: labels.map((label) => measurement(label)),
    availableWidth: 788,
    svgHeight: 520,
    baseLeft: 76,
    baseBottom: 0,
    titleHeight,
  });
  const plain = plan(14);
  const roomy = plan(34);
  assert.ok(plain.axisY < plain.labelsBottom && plain.labelsBottom < plain.titleY);
  assert.equal(roomy.bottom - plain.bottom, 20);
  assert.equal(roomy.labelsBottom - roomy.axisY, plain.labelsBottom - plain.axisY);
  assert.equal(roomy.titleY - roomy.labelsBottom, plain.titleY - plain.labelsBottom);
});
