// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  contributionX,
  levelPolyline,
  shiftedCurve,
  splineCurves,
} from "../../src/superglm/editor/app/chart/ordered_spline.js";

function orderedTerm(overrides = {}) {
  return {
    x: [0, 1, 2, 3],
    y: [0.8, 1, 1.2, 1.5],
    levels: ["a", "b", "c", "MISSING"],
    controls: { grid_x: [0, 0.5, 1, 1.5, 2], build_basis: [[1, 0.5, 0, 0, 0]] },
    spline_view: {
      available: true,
      reason: null,
      x: [0, 0.5, 1, 1.5, 2],
      y: [0.8, 0.9, 1, 1.1, 1.2],
      original_y: [0.8, 0.9, 1, 1.1, 1.2],
      level_indices: [0, 1, 2],
      fits_levels: true,
    },
    ...overrides,
  };
}

test("an ordered spline draws its spline on the grid", () => {
  const curves = splineCurves(orderedTerm());

  assert.deepEqual(curves.x, [0, 0.5, 1, 1.5, 2]);
  assert.deepEqual(curves.y, [0.8, 0.9, 1, 1.1, 1.2]);
  assert.deepEqual(curves.levelIndices, [0, 1, 2]);
});

test("levels off the spline are joined instead, and other terms get no spline", () => {
  const term = orderedTerm();
  term.spline_view.fits_levels = false;

  assert.equal(splineCurves(term).y, null);
  assert.equal(splineCurves({ ...term, spline_view: { ...term.spline_view, available: false } }), null);
  assert.equal(splineCurves({ x: [0, 1], y: [1, 1], spline_view: null }), null);
});

test("the level polyline leaves special levels out", () => {
  const term = orderedTerm();

  assert.deepEqual(levelPolyline(term.x, term.y, [0, 1, 2]), { x: [0, 1, 2], y: [0.8, 1, 1.2] });
});

test("basis rows are sampled on the grid for an ordered spline and on x otherwise", () => {
  assert.deepEqual(contributionX(orderedTerm()), [0, 0.5, 1, 1.5, 2]);
  assert.deepEqual(contributionX({ x: [3, 4], controls: { build_basis: [] } }), [3, 4]);
  assert.deepEqual(contributionX({ x: [3, 4], controls: null }), [3, 4]);
});

test("a moved coefficient scales the curve by its basis function", () => {
  const moved = shiftedCurve([1, 2, 4], [0, 0.5, 1], Math.log(2));

  assert.deepEqual(moved.map((value) => Number(value.toFixed(12))), [1, 2 * Math.SQRT2, 8].map(
    (value) => Number(value.toFixed(12))
  ));
});
