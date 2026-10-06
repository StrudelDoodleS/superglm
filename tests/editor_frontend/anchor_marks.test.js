// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  anchorMarks,
  bindDragWatch,
  spanRange,
} from "../../src/superglm/editor/app/chart/anchor_marks.js";

// A numeric axis as chart.js resolves it: one display point per grid point.
const X = [18, 22.5, 29.25, 40.3, 62.68, 85];
const numeric = { x: X, levels: null, displayToSourceIndices: X.map((_, i) => [i]) };

test("the anchor is ringed and tagged with its x as the axis prints it", () => {
  const marks = anchorMarks(numeric, "age", { term: "age", index: 4 }, null, new Set([4]));
  assert.deepEqual(marks, [{ role: "anchor", display: 4, value: "62.7", tag: "click · 62.7" }]);
  // Another term's anchor marks nothing here.
  assert.deepEqual(anchorMarks(numeric, "age", { term: "area", index: 4 }, null, new Set()), []);
  assert.deepEqual(anchorMarks(numeric, "age", null, null, new Set()), []);
});

test("after a Shift-click its end is tagged too, and the status range runs low to high", () => {
  const anchor = { term: "age", index: 4 };
  const span = { term: "age", from: 4, to: 1, indices: [1, 2, 3, 4] };
  const selection = new Set([1, 2, 3, 4]);
  assert.deepEqual(anchorMarks(numeric, "age", anchor, span, selection), [
    { role: "anchor", display: 4, value: "62.7", tag: "click · 62.7" },
    { role: "end", display: 1, value: "22.5", tag: "Shift-click · 22.5" },
  ]);
  assert.deepEqual(spanRange(numeric, "age", anchor, span, selection), { lo: "22.5", hi: "62.7" });
});

test("a selection changed since the Shift-click, or a moved anchor, is no span", () => {
  const span = { term: "age", from: 4, to: 1, indices: [1, 2, 3, 4] };
  const anchor = { term: "age", index: 4 };
  // A Ctrl-click took one point out: the anchor stays tagged, the end does not.
  assert.deepEqual(
    anchorMarks(numeric, "age", anchor, span, new Set([1, 2, 4])).map((mark) => mark.role),
    ["anchor"],
  );
  assert.equal(spanRange(numeric, "age", anchor, span, new Set([1, 2, 4])), null);
  // A click moved the anchor: the span it started from is gone.
  assert.equal(spanRange(numeric, "age", { term: "age", index: 2 }, span, new Set([1, 2, 3, 4])), null);
  // A Shift-click on the anchor itself spans one point, with nothing at a far end.
  const onAnchor = { term: "age", from: 4, to: 4, indices: [4] };
  assert.equal(anchorMarks(numeric, "age", anchor, onAnchor, new Set([4])).length, 1);
  assert.equal(spanRange(numeric, "age", anchor, onAnchor, new Set([4])), null);
});

test("on a categorical axis the marks name the level, through a collapsed display too", () => {
  const levels = ["B1", "B2", "B10", "B11", "B12"];
  const expanded = { x: [0, 1, 2, 3, 4], levels, displayToSourceIndices: levels.map((_, i) => [i]) };
  const anchor = { term: "brand", index: 3 };
  const span = { term: "brand", from: 3, to: 0, indices: [0, 1, 2, 3] };
  const selection = new Set([0, 1, 2, 3]);
  assert.deepEqual(spanRange(expanded, "brand", anchor, span, selection), { lo: "B1", hi: "B11" });

  // Drawn collapsed, B2 and B10 show as one group point; the anchor is source level B11.
  const collapsed = {
    x: [0, 1, 2, 3],
    levels: ["B1", "B2+B10", "B11", "B12"],
    displayToSourceIndices: [[0], [1, 2], [3], [4]],
  };
  assert.deepEqual(anchorMarks(collapsed, "brand", anchor, span, selection), [
    { role: "anchor", display: 2, value: "B11", tag: "click · B11" },
    { role: "end", display: 0, value: "B1", tag: "Shift-click · B1" },
  ]);
});

test("the tags step aside while the pointer drags past the click slop, and come back after", () => {
  const listeners = new Map();
  const svg = { addEventListener: (name, listener) => listeners.set(name, listener) };
  const shell = { dataset: {} };
  bindDragWatch(svg, shell, 3);

  listeners.get("pointerdown")({ button: 0, clientX: 100, clientY: 100 });
  listeners.get("pointermove")({ clientX: 102, clientY: 101 });
  assert.equal(shell.dataset.dragging, undefined, "a click's wobble is not a drag");
  listeners.get("pointermove")({ clientX: 108, clientY: 101 });
  assert.equal(shell.dataset.dragging, "true");
  listeners.get("pointerup")({});
  assert.equal(shell.dataset.dragging, undefined);

  // A move with no button down is a hover, not a drag.
  listeners.get("pointermove")({ clientX: 150, clientY: 150 });
  assert.equal(shell.dataset.dragging, undefined);
  listeners.get("pointerdown")({ button: 0, clientX: 10, clientY: 10 });
  listeners.get("pointermove")({ clientX: 10, clientY: 30 });
  listeners.get("pointercancel")({});
  assert.equal(shell.dataset.dragging, undefined);
});
