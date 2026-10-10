// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  axisMap,
  basisAt,
  basisCount,
  basisCurves,
  changedFunctions,
  functionsHoldingKnot,
  knotVector,
} from "../../src/superglm/editor/app/chart/knot_basis.js";
import { basisChange, basisOverlay } from "../../src/superglm/editor/app/chart/basis_overlay.js";
import { knotFrame } from "../../src/superglm/editor/app/chart/knot_marks.js";

const OPEN = { degree: 3, ends: "open", boundary: [0, 10], level_values: null };
const CLAMPED = { degree: 3, ends: "clamped", boundary: [0, 10], level_values: null };
// Uneven on purpose: crowded on the left, as a hand placement leaves them.
const UNEVEN = [0.8, 1.5, 2.1, 4, 7.5];
const NO_UI = { selected: null, drag: null, hover: null, pending: null };
const PLOT = { xMin: 0, xMax: 10, left: 50, right: 450, top: 20, axisY: 300, bottom: 360 };
const grid = (lo, hi, n) => Array.from({ length: n + 1 }, (_, g) => lo + ((hi - lo) * g) / n);

test("the knot vectors match superglm's open and clamped constructions", () => {
  // Open: the boundary widened by 0.001 of its range, carried on at the end spacings.
  const open = knotVector(OPEN, [2, 5]);
  // The first spacing is 2 - (-0.01) = 2.01, the last 10.01 - 5 = 5.01.
  const expected = [-6.04, -4.03, -2.02, -0.01, 2, 5, 10.01, 15.02, 20.03, 25.04];
  open.forEach((t, k) => assert.ok(Math.abs(t - expected[k]) < 1e-12, `${k}: ${t}`));
  assert.deepEqual(knotVector(CLAMPED, [5, 2]), [0, 0, 0, 0, 2, 5, 10, 10, 10, 10]);
  assert.equal(basisCount(open, 3), 6);
  // An ordered term's knots and boundary map through its level values.
  const ordered = { ...CLAMPED, boundary: [0, 3], level_values: [0, 1, 4, 5] };
  assert.deepEqual(knotVector(ordered, [1.5, 2.5]), [0, 0, 0, 0, 2.5, 4.5, 5, 5, 5, 5]);
  assert.equal(axisMap([0, 1, 4, 5])(2.25), 4.25);
});

// The fixture tests/test_editor_knots.py writes: for each case, the payload's basis and chart
// positions, and the fitted spec._knots that knotVector must reproduce from the first two.
const KNOT_BASIS_CASES = JSON.parse(
  readFileSync(new URL("./fixtures/knot_basis.json", import.meta.url), "utf8"),
);

// The rounding count c of a case, the bound being c u M for u = 2**-53 and M the knots' scale.
// Each count is of the roundings on one construction's longest chain, and the two constructions
// (superglm's and the browser's) differ by at most the sum of their errors:
// - Clamped on numbers (cubic regression): the knots are copies of the boundary and the positions,
//   with no arithmetic, so c = 0 and the rebuilt vector is the fitted one exactly.
// - Open on numbers (P-spline, B-spline), degree 3: the padded end lo - pad, pad = 0.001 (hi - lo),
//   is within 1.01 u M of exact (0.01 u M from pad's own rounding, one more from the subtraction).
//   The first spacing is within 3.01 u M (its subtraction rounds at most 2 u M more). The carried
//   knot is the end plus the spacing times the degree: 1.01 + 3 (3.01 + 2) + 1 = 17.04 u M, the 2
//   being the product's rounding per unit of degree and the last 1 the sum's. Each side is within
//   18 u M of exact, so the two differ by at most c = 36.
// - Ordered (clamped here), L levels with largest gap G: numpy places the positions by inverse
//   interpolation, which rounds the chart index to within (L + 3) u, so a knot is off by at most
//   G (L + 3) u. The browser's map back rounds the gap twice (2 G u, its offset being at most 1)
//   and the sum once (u M). So c = (L + 5) G / M + 1.
function roundingCount(basis, scale) {
  if (!basis.level_values) return basis.ends === "open" ? 36 : 0;
  assert.equal(basis.ends, "clamped", "an ordered open spline needs its open count added to this one");
  const values = basis.level_values;
  const gap = Math.max(...values.slice(1).map((v, i) => v - values[i]));
  return ((values.length + 5) * gap) / scale + 1;
}

test("the knots the browser builds reproduce the fitted knots, within their rounding bound", () => {
  // M = max(|lo|, |hi|, R), with lo and hi on the spline's own axis and R the largest |knot|.
  const u = Number.EPSILON / 2;
  for (const [name, { basis, positions, knots }] of Object.entries(KNOT_BASIS_CASES)) {
    const toAxis = axisMap(basis.level_values);
    const lo = toAxis(basis.boundary[0]);
    const hi = toAxis(basis.boundary[1]);
    const scale = Math.max(Math.abs(lo), Math.abs(hi), ...knots.map(Math.abs));
    const bound = roundingCount(basis, scale) * u * scale;
    const rebuilt = knotVector(basis, positions);
    assert.equal(rebuilt.length, knots.length, name);
    rebuilt.forEach((t, k) => {
      assert.ok(Math.abs(t - knots[k]) <= bound, `${name}, knot ${k}: ${t} against ${knots[k]}, bound ${bound}`);
    });
  }
});

test("inside the boundary the basis sums to one and never goes below zero", () => {
  for (const basis of [OPEN, CLAMPED, { ...CLAMPED, degree: 2 }]) {
    const curves = basisCurves(basis, UNEVEN, grid(0, 10, 400));
    for (let g = 0; g <= 400; g++) {
      const column = curves.map((curve) => curve[g]);
      const sum = column.reduce((total, value) => total + value, 0);
      // A sum of degree + 1 terms, each a few roundings: well inside 64 u.
      assert.ok(Math.abs(sum - 1) <= 64 * 2 ** -53, `${basis.ends} at ${g}: ${sum}`);
      assert.ok(column.every((value) => value >= 0), `${basis.ends} at ${g}`);
    }
  }
  const ordered = { ...OPEN, boundary: [0, 5], level_values: [0, 1, 4, 5, 9, 10] };
  const curves = basisCurves(ordered, [1.2, 3.7], grid(0, 5, 100));
  for (let g = 0; g <= 100; g++) {
    assert.ok(Math.abs(curves.reduce((total, curve) => total + curve[g], 0) - 1) < 1e-14);
  }
});

test("each basis function lives on its own knots and nowhere else", () => {
  const knots = knotVector(OPEN, UNEVEN);
  const xs = grid(0, 10, 500);
  for (let i = 0; i < basisCount(knots, 3); i++) {
    const lo = knots[i];
    const hi = knots[i + 4];
    for (const x of xs) {
      const { first, values } = basisAt(knots, 3, x);
      const value = i >= first && i <= first + 3 ? values[i - first] : 0;
      if (x <= lo || x >= hi) assert.equal(value, 0, `function ${i} at ${x}`);
      else assert.ok(value > 0, `function ${i} at ${x}`);
    }
  }
});

test("a dragged knot reshapes the degree + 2 functions holding it, and no other", () => {
  assert.deepEqual(functionsHoldingKnot(2, 3, 9), [2, 3, 4, 5, 6]);
  assert.deepEqual(functionsHoldingKnot(0, 3, 9), [0, 1, 2, 3, 4]);
  // Moving interior knot 2 from 2.1 to 3 changes exactly the functions built on it.
  for (const basis of [OPEN, CLAMPED]) {
    const before = knotVector(basis, UNEVEN);
    const after = knotVector(basis, [0.8, 1.5, 3, 4, 7.5]);
    assert.deepEqual(changedFunctions(after, before, 3, 1e-12), functionsHoldingKnot(2, 3, 9));
  }
});

test("a waiting change reaches the functions the knots in force do not have", () => {
  const before = knotVector(CLAMPED, [2, 5, 8]);
  // Adding a knot between 5 and 8 leaves the functions clear of it as they were.
  assert.deepEqual(changedFunctions(knotVector(CLAMPED, [2, 5, 6, 8]), before, 3, 1e-12), [2, 3, 4, 5, 6]);
  assert.deepEqual(changedFunctions(knotVector(CLAMPED, [2, 5, 8]), before, 3, 1e-12), []);
});

function term({ positions = UNEVEN, pending = null, basis = OPEN } = {}) {
  return {
    kind: "spline", term_type: "spline", x: [0, 10], levels: null,
    shape: { available: true, reason: null, ranges: [], support: null, specials: [] },
    knots: {
      available: true, reason: null, positions, count: positions.length, strategy: "explicit",
      alpha: 0.2, from_editor: true, lo: 0, hi: 10, min_gap: 0.1, max_count: null,
      resettable: true, even_only: null, basis,
    },
    pending: pending ? { knots: pending } : null,
  };
}

test("the overlay shows only while a knot is dragged or a change waits, in Knots mode", () => {
  const frame = knotFrame(term(), PLOT, true);
  assert.equal(basisOverlay(frame, NO_UI), null);
  assert.equal(basisOverlay(knotFrame(term(), PLOT, false), NO_UI), null);
  assert.equal(basisOverlay(knotFrame(term({ basis: null }), PLOT, true), NO_UI), null);

  const drag = { from: 2.1, x: 3, moved: true, remove: false };
  const dragging = basisOverlay(frame, { ...NO_UI, drag });
  assert.deepEqual(
    dragging.reached.flatMap((reached, index) => (reached ? [index] : [])),
    functionsHoldingKnot(2, 3, 9),
  );
  assert.equal(dragging.curves.length, 9);
  // A press that has not moved yet changes nothing.
  assert.equal(basisOverlay(frame, { ...NO_UI, drag: { ...drag, moved: false } }), null);

  const waiting = term({
    positions: [2, 5, 8],
    pending: { positions: [2, 5, 6, 8], count: 4, strategy: "explicit", alpha: 0.2 },
    basis: CLAMPED,
  });
  const overlay = basisOverlay(knotFrame(waiting, PLOT, true), NO_UI);
  assert.deepEqual(overlay.reached.flatMap((reached, index) => (reached ? [index] : [])), [2, 3, 4, 5, 6]);
});

test("a knot dragged below the axis, or a change being staged, reaches what it changes", () => {
  const frame = knotFrame(term({ positions: [2, 5, 8], basis: CLAMPED }), PLOT, true);
  const removing = basisChange(frame, { ...NO_UI, drag: { from: 5, x: 5, moved: true, remove: true } });
  assert.deepEqual(removing.positions, [2, 8]);
  assert.deepEqual(
    removing.reached,
    changedFunctions(knotVector(CLAMPED, [2, 8]), knotVector(CLAMPED, [2, 5, 8]), 3, 1e-12),
  );
  const staged = basisChange(frame, { ...NO_UI, pending: [2, 5, 6, 8] });
  assert.deepEqual(staged.reached, [2, 3, 4, 5, 6]);
});
