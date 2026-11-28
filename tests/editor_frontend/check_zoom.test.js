// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHECK_VIEWPORT,
  ZOOM_IN,
  ZOOM_OUT,
  fitDomain,
  isWhole,
  moveViewport,
  panBy,
  splitOf,
  stepZoom,
  viewportDomain,
  zoomAbout,
  zoomToBox,
} from "../../src/superglm/editor/app/chart/check_zoom.js";

const RECT = { left: 50, top: 30, right: 850, bottom: 430 };
/** A Q-Q's frame: one plot, both axes on [0, 10]. */
const square = (domain = { x: [0, 10], y: [0, 10] }) => ({
  base: { x: [0, 10], y: [0, 10] }, domain,
  plots: [{ rect: RECT, x: [RECT.left, RECT.right], y: [RECT.bottom, RECT.top] }],
});
/** An A/E's frame: two panels sharing y on [0.5, 2], x not zoomed. */
const panels = (domain = { x: null, y: [0.5, 2] }) => {
  const upper = { left: 50, top: 30, right: 400, bottom: 200 };
  const lower = { left: 50, top: 260, right: 400, bottom: 430 };
  return {
    base: { x: null, y: [0.5, 2] }, domain,
    plots: [upper, lower].map((rect) => ({ rect, x: null, y: [rect.bottom, rect.top] })),
  };
};
/** The value at px along an axis of the first plot of `frame`, drawn at `domain`. */
const at = (frame, domain, axis, px, plot = frame.plots[0]) => {
  const [p0, p1] = plot[axis];
  const [d0, d1] = domain[axis];
  return d0 + ((px - p0) * (d1 - d0)) / (p1 - p0);
};
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test("a wheel zoom keeps the value under the pointer and narrows both Q-Q axes alike", () => {
  const frame = square();
  const point = { x: 650, y: 130 };
  const next = zoomAbout(frame, point, ZOOM_IN);
  for (const axis of ["x", "y"]) {
    close(at(frame, next, axis, point[axis]), at(frame, frame.domain, axis, point[axis]));
    close(next[axis][1] - next[axis][0], 10 * ZOOM_IN);
  }
  // Zoomed out past the whole plot, the zoom is cleared.
  assert.equal(zoomAbout(square(next), point, ZOOM_OUT * ZOOM_OUT), null);
  assert.equal(zoomAbout(frame, point, ZOOM_OUT), null);
});

test("a zoom stops at a 200th of the whole and never leaves it", () => {
  let frame = square();
  for (let i = 0; i < 60; i++) {
    const next = zoomAbout(frame, { x: 849, y: 31 }, ZOOM_IN);
    frame = square(next);
    for (const axis of ["x", "y"]) assert.ok(next[axis][0] >= 0 && next[axis][1] <= 10, JSON.stringify(next));
  }
  close(frame.domain.x[1] - frame.domain.x[0], 10 / 200, 1e-12);
  // One frame's ticks arrive as one factor, which stops at the narrowest rather than refusing.
  const once = zoomAbout(square(), { x: 450, y: 230 }, ZOOM_IN ** 40);
  close(once.x[1] - once.x[0], 10 / 200, 1e-12);
});

test("a box zooms to exactly its corners' values on each axis, and a tiny box to nothing", () => {
  const frame = square();
  // 200 px by 50 px: x spans 2.5 to 5 (800 px is 10), y 2.5 to 3.75 (400 px is 10). The Q-Q's
  // two axes take the box's own ranges, not one square range.
  const next = zoomToBox(frame, { x: 250, y: 330 }, { x: 450, y: 280 });
  close(next.x[0], 2.5); close(next.x[1], 5);
  close(next.y[0], 2.5); close(next.y[1], 3.75);
  assert.deepEqual(zoomToBox(frame, { x: 250, y: 330 }, { x: 255, y: 230 }), null);
  // A box past the plot is cut to it.
  const cut = zoomToBox(frame, { x: 650, y: 130 }, { x: 2000, y: -500 });
  close(cut.x[1], 10); close(cut.y[1], 10);
});

test("A/E zooms its one shared y, about the panel under the pointer, and leaves x alone", () => {
  const frame = panels();
  const lowerPoint = { x: 200, y: 300 };
  const next = zoomAbout(frame, lowerPoint, ZOOM_IN);
  assert.equal(next.x, null);
  const lower = frame.plots[1];
  close(at(frame, next, "y", lowerPoint.y, lower), at(frame, frame.domain, "y", lowerPoint.y, lower));
  const box = zoomToBox(frame, { x: 60, y: 300 }, { x: 62, y: 400 });
  assert.equal(box.x, null);
  close(box.y[1], at(frame, frame.domain, "y", 300, lower));
  close(box.y[0], at(frame, frame.domain, "y", 400, lower));
});

test("a drag pans the values with the pointer and stops at the whole plot's edge", () => {
  const frame = square({ x: [4, 6], y: [4, 6] });
  // 80 px right and 40 px down is 0.2 left in x and 0.2 up in y, at 2 per 800 and per 400 px.
  const next = panBy(frame, { x: 400, y: 200 }, { x: 480, y: 240 });
  close(next.x[0], 3.8); close(next.x[1], 5.8);
  close(next.y[0], 4.2); close(next.y[1], 6.2);
  const far = panBy(frame, { x: 400, y: 200 }, { x: 40000, y: 200 });
  assert.deepEqual(far.x, [0, 2]);
  const ae = panBy(panels({ x: null, y: [1, 1.5] }), { x: 100, y: 100 }, { x: 300, y: 134 });
  assert.equal(ae.x, null);
  close(ae.y[0], 1.1); close(ae.y[1], 1.6);
});

test("the keys zoom about the plot's middle", () => {
  const next = stepZoom(square(), ZOOM_IN);
  close((next.x[0] + next.x[1]) / 2, 5); close((next.y[0] + next.y[1]) / 2, 5);
  assert.equal(stepZoom(square(next), ZOOM_OUT * ZOOM_OUT), null);
});

test("a held domain is fitted inside a scene's whole one, moved rather than cut", () => {
  const base = { x: [0, 10], y: [-1, 1] };
  assert.deepEqual(fitDomain(null, base), base);
  assert.deepEqual(fitDomain({ x: [8, 12], y: [-0.5, 0.5] }, base), { x: [6, 10], y: [-0.5, 0.5] });
  assert.deepEqual(fitDomain({ x: [-50, 50], y: [Number.NaN, 1] }, base), base);
  assert.ok(isWhole(fitDomain({ x: [-50, 50], y: [-3, 3] }, base), base));
  assert.ok(!isWhole({ x: [0, 9], y: [-1, 1] }, base));
});

// The viewport's events, each made on the Q-Q check.
const ON_TRAIN = { x: [0, 1], y: [0, 1] };
const VIEWPORT_EVENTS = {
  zoom: { type: "zoom", split: "train#1", domain: ON_TRAIN },
  home: { type: "home" },
  replace: { type: "replace", split: "train#2" },
};

test("the viewport table: every state and event, each move landing where it says", () => {
  assert.deepEqual([...CHECK_VIEWPORT.states], ["whole", "zoomed"]);
  assert.deepEqual([...CHECK_VIEWPORT.events].sort(), Object.keys(VIEWPORT_EVENTS).sort());
  const held = { whole: {}, zoomed: { check_qq: { state: "zoomed", split: "train#1", domain: { x: [0, 2], y: [0, 2] } } } };
  for (const state of CHECK_VIEWPORT.states) {
    for (const type of CHECK_VIEWPORT.events) {
      const before = held[state];
      const after = moveViewport(before, "check_qq", VIEWPORT_EVENTS[type]);
      const target = CHECK_VIEWPORT.next(state, type);
      if (target === null) assert.strictEqual(after, before, `${state} takes no ${type}`);
      else assert.equal(after.check_qq?.state ?? "whole", target, `${state} --${type}--> ${target}`);
    }
  }
});

test("a zoom, Home and the rows replaced move one check's viewport, and a no-op keeps the map", () => {
  const one = moveViewport({}, "check_qq", VIEWPORT_EVENTS.zoom);
  assert.deepEqual(one, { check_qq: { state: "zoomed", split: "train#1", domain: ON_TRAIN } });
  assert.equal(moveViewport(one, "check_qq", VIEWPORT_EVENTS.zoom), one);
  // The rows the zoom was made on, read again, replace nothing.
  assert.equal(moveViewport(one, "check_qq", { type: "replace", split: "train#1" }), one);
  assert.deepEqual(moveViewport(one, "check_qq", VIEWPORT_EVENTS.replace), {});
  assert.deepEqual(moveViewport(one, "check_qq", VIEWPORT_EVENTS.home), {});
  const two = moveViewport(one, "check_ae", { type: "zoom", split: "train#2", domain: { x: null, y: [1, 2] } });
  // Another check's rows are its own.
  assert.deepEqual(Object.keys(moveViewport(two, "check_ae", { type: "replace", split: "train#3" })), ["check_qq"]);
  assert.equal(moveViewport(two, "check_qq", { type: "replace", split: "train#1" }), two);
});

test("a zoom is drawn only on the rows it was made on, other rows of the same name and size included", () => {
  const split = (rows_id) => ({ split: { name: "train", label: "Train", n_obs: 200, rows_id } });
  const before = split(3);
  const after = split(4);
  assert.notEqual(splitOf(before), splitOf(after));
  assert.equal(splitOf(split(3)), splitOf(before));
  const viewport = { state: "zoomed", split: splitOf(before), domain: ON_TRAIN };
  assert.equal(viewportDomain(viewport, splitOf(before)), ON_TRAIN);
  assert.equal(viewportDomain(viewport, splitOf(after)), null);
  assert.equal(viewportDomain(undefined, splitOf(before)), null);
  assert.equal(splitOf(null), "");
  assert.equal(splitOf({ split: null }), "");
});
