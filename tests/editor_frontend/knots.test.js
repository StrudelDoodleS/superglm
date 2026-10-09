// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  AT_LEAST_ONE,
  SHOWN_GROUPED,
  addOutcome,
  addSpot,
  dropOutcome,
  freeSpot,
  knotAxis,
  knotChip,
  knotTagText,
  knotToolState,
  nudgeKnot,
  removeOutcome,
  ruleParams,
  snapKnot,
  stepRule,
  stepperState,
  shownKnots,
  tooManyKnots,
} from "../../src/superglm/editor/app/knots.js";
import {
  REMOVE_LABEL,
  inRemoveZone,
  knotAt,
  knotFrame,
  knotLayout,
  knotPx,
  onKnotBand,
  waitingMarks,
} from "../../src/superglm/editor/app/chart/knot_marks.js";
import { bindKnotGestures } from "../../src/superglm/editor/app/knot_gestures.js";

const NO_UI = { selected: null, drag: null, hover: null };
const AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"];

/** A numeric spline on 0..10: the grid, and Python's least gap, is 0.1. */
function numericTerm({ knots = {}, pending = null } = {}) {
  return {
    kind: "spline",
    term_type: "spline",
    x: [0, 10],
    levels: null,
    shape: { available: true, reason: null, ranges: [], support: null, specials: [] },
    knots: {
      available: true, reason: null, positions: [2, 4, 6, 8], count: 4, strategy: "uniform",
      alpha: 0.2, from_editor: false, lo: 0, hi: 10, min_gap: 0.1, max_count: null,
      resettable: false, ...knots,
    },
    pending,
  };
}

/** An ordered term with six levels on its curve, at 0..5, and so at most five knots. */
function orderedTerm({ knots = {}, specials = [], levels = AGE_BANDS } = {}) {
  return {
    kind: "ordered categorical",
    term_type: "ordered categorical",
    x: levels.map((_, i) => i),
    levels,
    shape: { available: true, reason: null, ranges: [], support: null, specials },
    knots: {
      available: true, reason: null, positions: [2.5], count: 1, strategy: "uniform",
      alpha: 0.2, from_editor: false, lo: 0, hi: 5, min_gap: 0.1, max_count: 5,
      resettable: false, ...knots,
    },
    pending: null,
  };
}

/** A numeric axis with ends and least gap of its own. */
function axisOf(lo, hi, gap) {
  return knotAxis(numericTerm({ knots: { lo, hi, min_gap: gap } }));
}

const PLOT = { xMin: 0, xMax: 10, left: 50, right: 450, top: 20, axisY: 300, bottom: 360 };

test("a numeric term snaps to three significant figures of its span, an ordered one to a tenth", () => {
  const ten = knotAxis(numericTerm());
  assert.equal(snapKnot(3.14159, ten), 3.1);
  assert.equal(snapKnot(2.71, ten, 1), 2.8);
  assert.equal(snapKnot(2.79, ten, -1), 2.7);
  assert.equal(snapKnot(2.7, ten, 1), 2.7);
  // Rounding to the grid's places drops the binary residue of k * step.
  assert.equal(snapKnot(0.30000000000000004, ten), 0.3);
  assert.equal(snapKnot(0.7, ten), 0.7);
  assert.equal(snapKnot(24.36, axisOf(18, 100, 0.1)), 24.4);
  assert.equal(snapKnot(345.6, axisOf(0, 999, 1)), 346);
  assert.equal(snapKnot(345.6, axisOf(0, 1000, 10)), 350);
  assert.equal(snapKnot(0.12345, axisOf(0, 0.5, 0.001)), 0.123);
  const ordered = knotAxis(orderedTerm());
  assert.equal(ordered.step, 0.1);
  assert.equal(snapKnot(2.46, ordered), 2.5);
  assert.equal(ordered.maxCount, 5);
});

test("a term without knots to adjust has no axis, and its tool says why", () => {
  const off = numericTerm({
    knots: { available: false, reason: "A term used by an interaction keeps its knots.",
      positions: null, lo: null, hi: null, min_gap: null },
  });
  assert.equal(knotAxis(off), null);
  assert.equal(shownKnots(off), null);
  assert.deepEqual(knotToolState(off, false), {
    available: false, reason: "A term used by an interaction keeps its knots.",
  });
  assert.deepEqual(knotToolState({ kind: "numeric" }, false), { available: false, reason: null });
  assert.deepEqual(knotToolState(orderedTerm(), false), { available: true, reason: null });
  assert.deepEqual(knotToolState(orderedTerm(), true), { available: false, reason: SHOWN_GROUPED });
});

test("a knot dropped too close to another settles on the nearest free spot, or stays put", () => {
  const axis = axisOf(0, 10, 0.5);
  assert.equal(freeSpot([3, 5], 4, axis), 4);
  assert.equal(freeSpot([3, 5], 3.2, axis), 3.5);
  assert.equal(freeSpot([3, 5], 4.7, axis), 4.5);
  // Too near an end is too near: the nearest free spot keeps the gap from it.
  assert.equal(freeSpot([5], 0.2, axis), 0.5);
  // On a span with no room left, it goes back where it was.
  assert.equal(freeSpot([0.5], 0.52, axisOf(0, 1, 0.4)), null);
});

test("a dragged knot may pass its neighbours; below the axis it is removed, never the last", () => {
  const axis = knotAxis(numericTerm());
  const positions = [2, 4, 6, 8];
  assert.deepEqual(dropOutcome(positions, 1, { x: 5.1, moved: true, remove: false }, axis), {
    params: { positions: [2, 5.1, 6, 8] }, select: 5.1,
  });
  assert.deepEqual(dropOutcome(positions, 0, { x: 7, moved: true, remove: false }, axis), {
    params: { positions: [4, 6, 7, 8] }, select: 7,
  });
  // A press that did not move, or a drop where it started, asks for nothing.
  assert.equal(dropOutcome(positions, 1, { x: 4, moved: false, remove: false }, axis), null);
  assert.equal(dropOutcome(positions, 1, { x: 4, moved: true, remove: false }, axis), null);
  assert.deepEqual(dropOutcome(positions, 1, { x: 4, moved: true, remove: true }, axis), {
    params: { positions: [2, 6, 8] }, select: null,
  });
  assert.deepEqual(removeOutcome([5], 0), { refusal: AT_LEAST_ONE });
});

test("an arrow key nudges a knot one step, ten with Shift, and hops a neighbour it would crowd", () => {
  const axis = axisOf(0, 10, 0.5);
  assert.equal(nudgeKnot([3, 5], 0, 1, 1, axis), 3.1);
  assert.equal(nudgeKnot([3, 5], 0, 1, 10, axis), 4);
  assert.equal(nudgeKnot([3, 3.5], 0, 1, 1, axis), 4);
  assert.equal(nudgeKnot([3, 3.5], 1, -1, 1, axis), 2.5);
  // At the end of the axis there is nowhere further to go.
  assert.equal(nudgeKnot([0.5, 5], 0, -1, 1, axis), null);
});

test("a click on the axis adds a knot on the grid, except where it crowds one or the term is full", () => {
  const axis = knotAxis(numericTerm());
  const positions = [2, 4, 6, 8];
  const spot = addSpot(positions, 5.03, axis);
  assert.equal(spot, 5);
  assert.deepEqual(addOutcome(positions, spot, axis), {
    params: { positions: [2, 4, 5, 6, 8] }, select: 5,
  });
  assert.equal(addSpot(positions, 4.02, axis), null);
  assert.equal(addOutcome(positions, null, axis), null);
  const full = knotAxis(orderedTerm());
  assert.deepEqual(addOutcome([0.5, 1.5, 2.5, 3.5, 4.5], 2, full), { refusal: tooManyKnots(5) });
  assert.equal(tooManyKnots(5), "This term has 6 levels on its curve, so it takes at most 5 knots.");
});

test("the tag reads a numeric knot as the axis does, an ordered one by its levels", () => {
  assert.equal(knotTagText(24.3, axisOf(18, 100, 0.1)), "24.3");
  const ordered = knotAxis(orderedTerm());
  assert.equal(knotTagText(2, ordered), "at 35-44");
  assert.equal(knotTagText(2.5, ordered), "35-44 to 45-54");
  // A special level has no place on the curve, so no knot sits by it.
  const special = knotAxis(orderedTerm({ levels: ["A", "B", "C", "S"], specials: ["S"] }));
  assert.equal(knotTagText(2, special), "at C");
  assert.equal(knotTagText(2.5, special), "2.5");
});

test("the stepper stops at one knot and at an ordered term's most, saying why", () => {
  const one = shownKnots(numericTerm({ knots: { positions: [5], count: 1 } }));
  const numeric = stepperState(one, knotAxis(numericTerm()), "uniform");
  assert.deepEqual(numeric.fewer, { enabled: false, body: AT_LEAST_ONE });
  assert.deepEqual(numeric.more, { enabled: true, body: "2 knots, all re-placed by even spacing." });
  const term = orderedTerm({ knots: { positions: [0.5, 1.5, 2.5, 3.5, 4.5], count: 5 } });
  const ordered = stepperState(shownKnots(term), knotAxis(term), "quantile");
  assert.deepEqual(ordered.more, { enabled: false, body: tooManyKnots(5) });
  assert.deepEqual(ordered.fewer, { enabled: true, body: "4 knots, all re-placed by quantiles of values." });
});

test("a new count re-places by the shown rule, else the one in force, else even spacing", () => {
  const byHand = numericTerm({
    knots: { strategy: "quantile_tempered", alpha: 0.5 },
    pending: { knots: { positions: [2, 5, 6, 8], count: 4, strategy: "explicit", alpha: 0.5 } },
  });
  assert.deepEqual(stepRule(byHand), { strategy: "quantile_tempered", alpha: 0.5 });
  assert.deepEqual(stepRule(numericTerm({ knots: { strategy: "explicit" } })), {
    strategy: "uniform", alpha: 0.2,
  });
  assert.deepEqual(ruleParams(5, "quantile_tempered", 0.5), {
    count: 5, strategy: "quantile_tempered", alpha: 0.5,
  });
  assert.deepEqual(ruleParams(5, "quantile_rows", 0.5), { count: 5, strategy: "quantile_rows" });
});

test("the chip names the shown knots and their rule, and tints while a knot change waits", () => {
  assert.deepEqual(knotChip(numericTerm()), { text: "4 knots · even spacing", waiting: false });
  assert.deepEqual(knotChip(numericTerm({ knots: { strategy: "explicit" } })), {
    text: "4 knots · listed in code", waiting: false,
  });
  assert.deepEqual(knotChip(numericTerm({ knots: { strategy: "explicit", from_editor: true } })), {
    text: "4 knots · placed by hand", waiting: false,
  });
  const waiting = numericTerm({
    pending: { knots: { positions: [5], count: 1, strategy: "quantile_rows", alpha: 0.2 } },
  });
  assert.deepEqual(knotChip(waiting), { text: "1 knot · quantiles of rows", waiting: true });
  assert.equal(knotChip(numericTerm({ knots: { positions: null } })), null);
});

test("a waiting change's moved knots are ghosts, its removed ones crossed, its new ones placed", () => {
  const inForce = [2, 4, 6, 8];
  assert.deepEqual(waitingMarks(inForce, [2, 5, 6, 8], 1e-7), {
    placed: [false, true, false, false], ghosts: [{ x: 4, removed: false }],
  });
  assert.deepEqual(waitingMarks(inForce, [2, 6, 8], 1e-7), {
    placed: [false, false, false], ghosts: [{ x: 4, removed: true }],
  });
  assert.deepEqual(waitingMarks(inForce, [2, 4, 5, 6, 8], 1e-7).ghosts, []);
  // Re-placed four to three: every knot moves, and the one paired with none is removed.
  const replaced = waitingMarks(inForce, [2.5, 5, 7.5], 1e-7);
  assert.deepEqual(replaced.placed, [true, true, true]);
  assert.deepEqual(replaced.ghosts.filter((ghost) => ghost.removed), [{ x: 4, removed: true }]);
  assert.equal(replaced.ghosts.length, 4);
});

test("outside Knots mode the knots are ticks under the axis, the waiting ones placed", () => {
  const term = numericTerm({
    pending: { knots: { positions: [2, 5, 6, 8], count: 4, strategy: "explicit", alpha: 0.2 } },
  });
  const zoomed = knotFrame(term, { ...PLOT, xMin: 3 }, false);
  const layout = knotLayout(zoomed, NO_UI);
  // Zoomed past it, the knot at 2 is not drawn.
  assert.deepEqual(layout.ticks.map(({ x, placed }) => [x, placed]), [[5, true], [6, false], [8, false]]);
  assert.deepEqual(layout.ghosts.map((ghost) => ghost.removed), [false]);
  assert.deepEqual([layout.handles, layout.guides, layout.band, layout.tag], [[], [], false, null]);
});

test("in Knots mode the selected or dragged knot is tagged, and a drop below the axis removes it", () => {
  const frame = knotFrame(numericTerm(), PLOT, true);
  const selected = knotLayout(frame, { ...NO_UI, selected: 4 });
  assert.equal(selected.band, true);
  assert.equal(selected.guides.length, 4);
  assert.deepEqual(selected.handles.map((handle) => handle.selected), [false, true, false, false]);
  assert.deepEqual(selected.tag, { px: knotPx(frame, 4), text: "4" });
  assert.equal(selected.removeZone, null);

  const dragging = knotLayout(frame, { ...NO_UI, drag: { from: 4, x: 5.1, moved: true, remove: false } });
  assert.equal(dragging.handles[1].px, knotPx(frame, 5.1));
  assert.equal(dragging.handles[1].placed, true);
  assert.equal(dragging.tag.text, "5.1");
  assert.equal(dragging.removeZone.label, REMOVE_LABEL);

  const removing = knotLayout(frame, { ...NO_UI, drag: { from: 4, x: 4, moved: true, remove: true } });
  const zone = removing.removeZone;
  assert.equal(removing.handles[1].cy, zone.top + zone.height / 2);
  assert.equal(removing.handles[1].removing, true);
  assert.equal(removing.guides.length, 3);
  const last = knotFrame(numericTerm({ knots: { positions: [5], count: 1 } }), PLOT, true);
  assert.equal(knotLayout(last, { ...NO_UI, drag: { from: 5, x: 5, moved: true, remove: true } })
    .removeZone.label, AT_LEAST_ONE);
});

test("the dashed ghost of a new knot shows where a click adds one, but not on a full term", () => {
  const frame = knotFrame(numericTerm(), PLOT, true);
  assert.equal(knotLayout(frame, { ...NO_UI, hover: 5 }).adding, knotPx(frame, 5));
  const full = orderedTerm({ knots: { positions: [0.5, 1.5, 2.5, 3.5, 4.5], count: 5 } });
  const fullFrame = knotFrame(full, { ...PLOT, xMax: 5 }, true);
  assert.equal(knotLayout(fullFrame, { ...NO_UI, hover: 2 }).adding, null);
});

test("a handle answers within its reach, the band along the axis, and the zone below it", () => {
  const frame = knotFrame(numericTerm(), PLOT, true);
  const at = knotPx(frame, 2);
  assert.equal(knotAt(frame, { x: at + 5, y: 305 }), 0);
  assert.equal(knotAt(frame, { x: at + 15, y: 305 }), null);
  assert.equal(knotAt(frame, { x: at, y: 320 }), null);
  assert.equal(onKnotBand(frame, { x: 300, y: 290 }), true);
  assert.equal(onKnotBand(frame, { x: 300, y: 280 }), false);
  assert.equal(onKnotBand(frame, { x: 30, y: 300 }), false);
  assert.equal(inRemoveZone(frame, { y: 327 }), true);
  assert.equal(inRemoveZone(frame, { y: 320 }), false);
});

/**
 * The gestures on a fake chart whose client and svg coordinates agree, with
 * the knot frame chart.js would leave on it.
 */
function gestureHarness(term, { editing = true, staged = true, plot = PLOT } = {}) {
  const listeners = new Map();
  const changes = [];
  let statusRenders = 0;
  const svg = {
    _knotFrame: knotFrame(term, plot, editing),
    viewBox: { baseVal: { x: 0, y: 0, width: 500, height: 400 } },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener(name) { listeners.delete(name); },
    setPointerCapture() {},
    focus() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 500, height: 400 }; },
  };
  const gestures = bindKnotGestures({
    svg,
    active: () => editing,
    onChange: async (params) => { changes.push(params); return staged; },
    onStatus: () => { statusRenders += 1; },
    redraw: () => {},
  });
  const pointer = (name, x, y) => listeners.get(name)?.({
    button: 0, pointerId: 1, clientX: x, clientY: y,
    shiftKey: false, ctrlKey: false, metaKey: false, preventDefault() {},
  });
  const key = (name, shiftKey = false) => listeners.get("keydown")?.({
    key: name, shiftKey, altKey: false, ctrlKey: false, metaKey: false, preventDefault() {},
  });
  const px = (x) => knotPx(svg._knotFrame, x);
  return {
    svg, gestures, changes, pointer, key, px,
    get statusRenders() { return statusRenders; },
  };
}

test("dragging a knot along the axis stages its new place, and below the axis removes it", async () => {
  const harness = gestureHarness(numericTerm());
  const { pointer, px, changes, gestures } = harness;
  pointer("pointerdown", px(4), 300);
  pointer("pointermove", px(5.1), 302);
  assert.equal(gestures.ui().drag.x, 5.1);
  pointer("pointerup", px(5.1), 302);
  assert.deepEqual(changes, [{ positions: [2, 5.1, 6, 8] }]);
  assert.equal(gestures.ui().selected, 5.1);
  assert.equal(gestures.ui().drag, null);

  pointer("pointerdown", px(6), 300);
  pointer("pointermove", px(6), 330);
  assert.equal(gestures.ui().drag.remove, true);
  pointer("pointerup", px(6), 330);
  assert.deepEqual(changes.at(-1), { positions: [2, 4, 8] });
  assert.equal(gestures.ui().selected, null);
});

test("a click on the band adds a knot, refused on the status line when the term is full", () => {
  const harness = gestureHarness(numericTerm());
  harness.pointer("pointerdown", harness.px(5), 296);
  harness.pointer("pointerup", harness.px(5), 296);
  assert.deepEqual(harness.changes, [{ positions: [2, 4, 5, 6, 8] }]);

  const full = gestureHarness(
    orderedTerm({ knots: { positions: [0.5, 1.5, 2.5, 3.5, 4.5], count: 5 } }),
    { plot: { ...PLOT, xMax: 5 } },
  );
  full.pointer("pointerdown", full.px(2), 300);
  full.pointer("pointerup", full.px(2), 300);
  assert.deepEqual(full.changes, []);
  assert.equal(full.gestures.message(), tooManyKnots(5));
  assert.equal(full.statusRenders, 1);
});

test("arrow keys nudge the selected knot, Delete removes it and Escape lets it go", () => {
  const { pointer, key, px, changes, gestures } = gestureHarness(numericTerm());
  // A click selects without staging anything.
  pointer("pointerdown", px(4), 300);
  pointer("pointerup", px(4), 300);
  assert.deepEqual(changes, []);
  assert.equal(gestures.ui().selected, 4);
  key("ArrowRight");
  assert.deepEqual(changes, [{ positions: [2, 4.1, 6, 8] }]);
  assert.equal(gestures.ui().selected, 4.1);
  // The payload has not moved the knot yet: the frame still shows it at 4.
  gestures.reset();
  pointer("pointerdown", px(6), 300);
  pointer("pointerup", px(6), 300);
  key("ArrowLeft", true);
  assert.deepEqual(changes.at(-1), { positions: [2, 4, 5, 8] });
  gestures.reset();
  key("ArrowRight");
  assert.equal(gestures.ui().selected, 2);
  key("Delete");
  assert.deepEqual(changes.at(-1), { positions: [4, 6, 8] });
  key("Escape");
  assert.equal(gestures.ui().selected, null);
});

test("a change that is not staged gives the selection back, and outside Knots mode nothing acts", async () => {
  const harness = gestureHarness(numericTerm(), { staged: false });
  harness.pointer("pointerdown", harness.px(4), 300);
  harness.pointer("pointerup", harness.px(4), 300);
  harness.key("ArrowRight");
  assert.equal(harness.gestures.ui().selected, 4.1);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(harness.gestures.ui().selected, 4);

  const idle = gestureHarness(numericTerm(), { editing: false });
  idle.pointer("pointerdown", idle.px(4), 300);
  idle.pointer("pointerup", idle.px(4), 300);
  idle.key("Delete");
  assert.deepEqual(idle.changes, []);
  assert.equal(idle.gestures.ui().selected, null);
});
