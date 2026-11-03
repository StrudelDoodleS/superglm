// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { sparkline } from "../../src/superglm/editor/app/chart/sparkline.js";

const SIZE = { width: 40, height: 16 };

/** The points of a "M x y L x y ..." path. */
function points(d) {
  return [...d.matchAll(/(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)/g)].map((m) => [Number(m[1]), Number(m[2])]);
}

test("a spline gives a line from the left edge to the right, higher values higher up", () => {
  const curve = sparkline(
    { kind: "spline", term_type: "spline", x: [0, 1, 2, 3], y: [1, 2, 3, 2], levels: null },
    SIZE,
  );
  assert.equal(curve.kind, "line");
  const p = points(curve.d);
  assert.equal(p.length, 4);
  assert.ok(p[0][0] <= p[1][0] && p[2][0] <= p[3][0]);
  assert.ok(p[0][0] >= 0 && p[3][0] <= SIZE.width);
  assert.ok(p[2][1] < p[1][1] && p[1][1] < p[0][1], "SVG y grows downward");
  for (const [, y] of p) assert.ok(y >= 0 && y <= SIZE.height);
});

test("a flat spline is a level line inside the box", () => {
  const curve = sparkline(
    { kind: "spline", term_type: "spline", x: [0, 1, 2], y: [1.5, 1.5, 1.5], levels: null },
    SIZE,
  );
  const ys = points(curve.d).map(([, y]) => y);
  assert.equal(new Set(ys).size, 1);
  assert.ok(ys[0] > 0 && ys[0] < SIZE.height);
});

test("categorical terms give bars measured from the baseline, signed by their side of it", () => {
  const curve = sparkline(
    { kind: "categorical", term_type: "categorical", x: [0, 1, 2], y: [1.5, 1, 0.5], levels: ["a", "b", "c"], baseline: 1 },
    SIZE,
  );
  assert.equal(curve.kind, "bars");
  const [up, level, down] = curve.bars;
  const zero = up.y + up.h; // above the baseline: the bar's bottom edge sits on it
  assert.ok(Math.abs(down.y - zero) < 1e-9); // below: its top edge does
  assert.ok(up.h > 0 && down.h > 0 && Math.abs(up.h - down.h) < 1e-9);
  assert.equal(level.h, 0);
  assert.ok(up.x < level.x && level.x < down.x);
  for (const bar of curve.bars) {
    assert.ok(bar.x >= 0 && bar.x + bar.w <= SIZE.width);
    assert.ok(bar.y >= 0 && bar.y + bar.h <= SIZE.height);
  }
});

test("bars default to a baseline of 1, and a zero baseline measures coefficient-like values from 0", () => {
  const terms = { kind: "categorical", term_type: "categorical", x: [0, 1], levels: ["a", "b"] };
  const relativity = sparkline({ ...terms, y: [2, 0.5] }, SIZE).bars;
  assert.ok(Math.abs(relativity[0].y + relativity[0].h - relativity[1].y) < 1e-9);
  const shifted = sparkline({ ...terms, y: [2, -1], baseline: 0 }, SIZE).bars;
  assert.ok(Math.abs(shifted[0].y + shifted[0].h - shifted[1].y) < 1e-9);
  assert.ok(shifted[0].h > shifted[1].h);
});

test("an ordered categorical drawn as a curve stays a line", () => {
  const curve = sparkline(
    { kind: "categorical", term_type: "ordered categorical", x: [0, 1, 2], y: [1, 1.2, 1.4], levels: ["a", "b", "c"] },
    SIZE,
  );
  assert.equal(curve.kind, "line");
});

test("a term with no points gives an empty line rather than throwing", () => {
  assert.deepEqual(sparkline({ kind: "spline", term_type: "spline", x: [], y: [], levels: null }, SIZE), { kind: "line", d: "" });
});

test("non-finite values are skipped", () => {
  const curve = sparkline(
    { kind: "spline", term_type: "spline", x: [0, 1, 2], y: [1, Number.NaN, 2], levels: null },
    SIZE,
  );
  assert.equal(points(curve.d).length, 2);
});
