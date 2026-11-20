// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  bandPath,
  collapsedKinds,
  hasInterval,
  intervalRuns,
  intervalValues,
  nonEstimableKind,
  nonEstimableLines
} from "../../src/superglm/editor/app/chart/intervals.js";

const sx = (value) => 10 * value;
const sy = (value) => 100 - 10 * value;

/** The closed areas of a band path, each as its "x y" points. */
function areas(d) {
  return d.split("Z").map((part) => part.trim()).filter(Boolean).map((part) =>
    [...part.matchAll(/(-?\d+\.\d+) (-?\d+\.\d+)/g)].map((m) => [Number(m[1]), Number(m[2])])
  );
}

test("a band with every interval is the one closed area it always was", () => {
  const x = [0, 1, 2];
  const d = bandPath(x, [1, 2, 3], [2, 3, 4], sx, sy);
  assert.equal(
    d,
    "M 0.00 80.00 L 10.00 70.00 L 20.00 60.00 L 20.00 70.00 L 10.00 80.00 L 0.00 90.00 Z"
  );
});

test("a point without an interval breaks the band, and no area reaches it", () => {
  const x = [0, 1, 2, 3, 4, 5];
  const lower = [1, 1, null, 1, 1, Number.NaN];
  const upper = [2, 2, null, 2, 2, 2];
  assert.deepEqual(intervalRuns(lower, upper), [[0, 1], [3, 4]]);
  const shapes = areas(bandPath(x, lower, upper, sx, sy));
  assert.equal(shapes.length, 2);
  const xs = shapes.flat().map(([px]) => px);
  assert.deepEqual([...new Set(xs)].sort((a, b) => a - b), [0, 10, 30, 40]);
  for (const [, py] of shapes.flat()) assert.ok(Number.isFinite(py) && py >= 80 && py <= 90);
});

test("a lone point keeps its interval as a thin bar, and no point leaves no band", () => {
  const shapes = areas(bandPath([0, 1, 2], [null, 1, null], [null, 3, null], sx, sy));
  assert.deepEqual(shapes, [[[8.5, 70], [11.5, 70], [11.5, 90], [8.5, 90]]]);
  assert.equal(bandPath([0, 1], [null, null], [null, null], sx, sy), "");
});

test("the y range takes the interval ends that are numbers", () => {
  assert.deepEqual(intervalValues([1, null, Number.NaN], [2, null, 5]), [1, 2, 5]);
  assert.equal(hasInterval([1, null], [2, 3], 1), false);
  assert.equal(hasInterval([1, null], [2, 3], 0), true);
});

test("a point the fit does not determine names its kind and why", () => {
  const view = { non_estimable: ["", "alias", "unresolved"] };
  assert.equal(nonEstimableKind(view, 0), "");
  assert.equal(nonEstimableKind(view, 1), "alias");
  assert.equal(nonEstimableKind({ non_estimable: null }, 1), "");
  assert.deepEqual(nonEstimableLines(""), []);
  assert.deepEqual(nonEstimableLines("alias"), ["Not determined: no interval", "aliased: held by the constraint"]);
  assert.deepEqual(nonEstimableLines("unresolved"), ["Not determined: no interval", "not resolved at float64"]);
  // each line fits the tooltip's widest box (240 px at 7 px a character, less its padding)
  for (const kind of ["alias", "separated", "unresolved", "unfitted"]) {
    for (const line of nonEstimableLines(kind)) assert.ok(line.length * 7 + 18 <= 240, line);
  }
});

test("a collapsed group takes the kind of the first of its levels the fit does not determine", () => {
  assert.deepEqual(
    collapsedKinds(["", "alias", "", "separated"], [[0, 2], [1, 3], [3]]),
    ["", "alias", "separated"]
  );
  assert.equal(collapsedKinds(null, [[0]]), null);
});
