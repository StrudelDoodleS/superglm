// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

import {
  GROUPED_EDGE,
  NOT_CONTIGUOUS,
  TOO_FEW_POINTS,
  shapeButtonState,
  shapeRangeDescription,
  shapeRangeExtent,
  shapeRangeForSelection
} from "../../src/superglm/editor/app/shapes.js";
import {
  OPERATION_HELP,
  helpForElement
} from "../../src/superglm/editor/app/views/help_content.js";

const AVAILABLE = Object.freeze({ available: true, reason: null, ranges: [] });
const numeric = Object.freeze({
  term_type: "spline",
  x: [18, 20, 25, 30, 40, 50],
  levels: null,
  shape: AVAILABLE
});
const BANDS = ["B1", "B2", "B3", "B4", "B5", "B6"];

function ordered(levelGroups = []) {
  return {
    term_type: "ordered categorical",
    x: BANDS.map((_, i) => i),
    levels: BANDS.slice(),
    level_groups: levelGroups,
    shape: AVAILABLE
  };
}

const expanded = { x: BANDS.map((_, i) => i), displayToSourceIndices: BANDS.map((_, i) => [i]) };

test("contiguous numeric selection gives its x extent", () => {
  assert.deepEqual(shapeRangeForSelection(numeric, new Set([1, 2, 3])), { lo: 20, hi: 30 });
});

test("a gap in the selection gives no range, nor does an empty one", () => {
  assert.equal(shapeRangeForSelection(numeric, new Set([1, 3])), null);
  assert.equal(shapeRangeForSelection(numeric, new Set()), null);
});

test("an ordered selection names its first and last bands whatever the insertion order", () => {
  assert.deepEqual(shapeRangeForSelection(ordered(), new Set([3, 1, 2])), { lo: "B2", hi: "B4" });
});

test("unavailable terms show disabled icons with the backend reason", () => {
  const term = { ...numeric, shape: { available: false, reason: "Shapes need a spline term.", ranges: [] } };
  assert.deepEqual(shapeButtonState(term, new Set([1, 2])), {
    visible: true,
    enabled: false,
    reason: "Shapes need a spline term."
  });
});

test("categorical terms hide the icons", () => {
  const term = {
    term_type: "categorical",
    x: [0, 1],
    levels: ["N", "S"],
    shape: { available: false, reason: "Shapes need a spline term.", ranges: [] }
  };
  assert.equal(shapeButtonState(term, new Set([0])).visible, false);
});

test("a broken run or a single point disables the icons and says so", () => {
  assert.deepEqual(shapeButtonState(numeric, new Set([1, 3])), {
    visible: true,
    enabled: false,
    reason: NOT_CONTIGUOUS
  });
  assert.equal(shapeButtonState(numeric, new Set([2])).reason, TOO_FEW_POINTS);
  assert.deepEqual(shapeButtonState(numeric, new Set([1, 2, 3])), {
    visible: true,
    enabled: true,
    reason: null
  });
});

test("a run ending inside a collapsed group is refused before it is sent; a group inside is fine", () => {
  const term = ordered([{ label: "B2+B3", indices: [1, 2] }]);
  assert.equal(shapeButtonState(term, new Set([1, 2, 3])).reason, GROUPED_EDGE);
  assert.equal(shapeButtonState(term, new Set([0, 1, 2])).reason, GROUPED_EDGE);
  assert.equal(shapeButtonState(term, new Set([0, 1, 2, 3])).enabled, true);
});

test("bands bound the degree an ordered range can carry", () => {
  const term = ordered();
  assert.equal(shapeButtonState(term, new Set([1, 2]), 1).enabled, true);
  assert.equal(
    shapeButtonState(term, new Set([1, 2]), 2).reason,
    "Select at least 3 bands for a Quadratic."
  );
  assert.equal(shapeButtonState(term, new Set([1, 2, 3]), 2).enabled, true);
  assert.equal(
    shapeButtonState(term, new Set([1, 2, 3]), 3).reason,
    "Select at least 4 bands for a Cubic."
  );
});

test("a numeric run is gated on the values Python counts between its snapped edges", () => {
  // Grid 18..50; the run 1..2 holds through[2] - below[1] = 5 - 3 = 2 values.
  const support = { below: [0, 3, 4, 7, 10, 12], through: [1, 4, 5, 8, 12, 13] };
  const term = { ...numeric, shape: { ...AVAILABLE, support } };
  assert.equal(shapeButtonState(term, new Set([1, 2]), 1).enabled, true);
  assert.deepEqual(shapeButtonState(term, new Set([1, 2]), 2), {
    visible: true,
    enabled: false,
    reason: "Select at least 3 distinct values for a Quadratic."
  });
  assert.equal(shapeButtonState(term, new Set([1, 2, 3]), 3).enabled, true);
  assert.equal(
    shapeButtonState(term, new Set([4, 5]), 3).reason,
    "Select at least 4 distinct values for a Cubic."
  );
  // Without counts (no retained data) Python decides when the range is sent.
  assert.equal(shapeButtonState(numeric, new Set([1, 2]), 3).enabled, true);
});

test("the overlay names the pinned shape and its edges", () => {
  assert.equal(
    shapeRangeDescription({ lo: 30, hi: 45, degree: 1, label: "Line" }),
    "Pinned to a straight line from 30 to 45."
  );
  assert.equal(
    shapeRangeDescription({ lo: "B2", hi: "B4", degree: 0, label: "Flat" }),
    "Pinned to a flat level from B2 to B4."
  );
});

test("a numeric range spans its edges; bands span from the first left gap to the last right gap", () => {
  const range = { lo: 30, hi: 45, degree: 1, label: "Line" };
  assert.deepEqual(shapeRangeExtent(numeric, expanded, range), [30, 45]);
  const bands = { lo: "B2", hi: "B4", degree: 1, label: "Line" };
  assert.deepEqual(shapeRangeExtent(ordered(), expanded, bands), [0.5, 3.5]);
});

test("a collapsed display places the edge bands through their source mapping", () => {
  const collapsed = { x: [0, 1, 2, 3, 4], displayToSourceIndices: [[0], [1, 2], [3], [4], [5]] };
  const range = { lo: "B1", hi: "B6", degree: 2, label: "Quadratic" };
  assert.deepEqual(shapeRangeExtent(ordered(), collapsed, range), [-0.5, 4.5]);
});

test("an edge that is not a band on the axis draws nothing", () => {
  assert.equal(shapeRangeExtent(ordered(), expanded, { lo: 1, hi: 3, degree: 1, label: "Line" }), null);
  assert.equal(
    shapeRangeExtent(ordered(), expanded, { lo: "B2", hi: "B9", degree: 1, label: "Line" }),
    null
  );
});

class FakeElement {
  constructor(dataset) {
    this.dataset = dataset;
  }
}
globalThis.HTMLElement = FakeElement;

test("a disabled shape icon's reason outranks its operation help", () => {
  assert.equal(OPERATION_HELP.shape_line.title, "Line and refit");
  assert.deepEqual(Object.keys(OPERATION_HELP).filter((key) => key.startsWith("shape_")), [
    "shape_flat", "shape_line", "shape_quadratic", "shape_cubic"
  ]);
  const enabled = new FakeElement({ helpOperation: "shape_line" });
  assert.strictEqual(helpForElement(enabled), OPERATION_HELP.shape_line);
  const disabled = new FakeElement({
    helpOperation: "shape_line",
    popoverTitle: "Line and refit",
    popoverBody: GROUPED_EDGE
  });
  assert.deepEqual(helpForElement(disabled), { title: "Line and refit", body: GROUPED_EDGE });
});
