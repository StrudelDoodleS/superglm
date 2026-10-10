// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  AT_LEAST_ONE,
  SHOWN_GROUPED,
  addOutcome,
  addSpot,
  decadeGrid,
  dropOutcome,
  freeSpot,
  knotAxis,
  knotChip,
  knotFits,
  knotPenaltyTitle,
  knotGrid,
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

/** A numeric spline on 0..10, knots 2 apart: its grid there, and Python's least gap, is 0.1. */
function numericTerm({ knots = {}, pending = null } = {}) {
  return {
    kind: "spline",
    term_type: "spline",
    x: [0, 10],
    levels: null,
    shape: { available: true, reason: null, ranges: [], support: null, specials: [] },
    knots: {
      available: true, reason: null, positions: [2, 4, 6, 8], count: 4, strategy: "uniform",
      alpha: 0.2, from_editor: false, lo: 0, hi: 10, min_gap: null, max_count: null,
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

/** A numeric axis with ends of its own. */
function axisOf(lo, hi) {
  return knotAxis(numericTerm({ knots: { lo, hi, min_gap: null } }));
}

const PLOT = { xMin: 0, xMax: 10, left: 50, right: 450, top: 20, axisY: 300, bottom: 360 };

test("a numeric knot snaps to two significant figures of the space it sits in, an ordered one to a tenth", () => {
  assert.deepEqual(decadeGrid(4), { step: 0.1, places: 1 });
  assert.deepEqual(decadeGrid(10), { step: 1, places: 0 });
  assert.deepEqual(decadeGrid(999.9), { step: 10, places: 0 });
  assert.deepEqual(decadeGrid(1000), { step: 100, places: 0 });
  assert.deepEqual(decadeGrid(0.05), { step: 0.001, places: 3 });
  const ten = knotAxis(numericTerm());
  assert.equal(ten.gap, null);
  const grid = knotGrid(3.14159, [2, 4, 6, 8], ten);
  assert.deepEqual(grid, { step: 0.1, places: 1 });
  assert.equal(snapKnot(3.14159, grid), 3.1);
  assert.equal(snapKnot(2.71, grid, 1), 2.8);
  assert.equal(snapKnot(2.79, grid, -1), 2.7);
  assert.equal(snapKnot(2.7, grid, 1), 2.7);
  // Rounding to the grid's places drops the binary residue of k * step.
  assert.equal(snapKnot(0.30000000000000004, grid), 0.3);
  assert.equal(snapKnot(0.7, grid), 0.7);
  // Knots a rule put close together where the data is dense take a finer grid
  // than knots far apart on the same axis.
  const wide = axisOf(0, 27000);
  const crowded = [29, 61, 95, 140, 9000, 20000];
  assert.equal(knotGrid(45, crowded, wide).step, 1);
  assert.equal(knotGrid(15000, crowded, wide).step, 1000);
  const ordered = knotAxis(orderedTerm());
  assert.equal(ordered.gap, 0.1);
  assert.equal(snapKnot(2.46, knotGrid(2.46, [], ordered)), 2.5);
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
  const axis = knotAxis(numericTerm());
  assert.equal(freeSpot([3, 5], 4, axis), 4);
  assert.equal(freeSpot([3, 5], 3.05, axis), 3.1);
  assert.equal(freeSpot([3, 5], 4.97, axis), 4.9);
  // Too near an end is too near: the nearest free spot keeps the step from it.
  assert.equal(freeSpot([5], 0.05, axis), 0.1);
  // Among knots crowded where the data is dense, a drop keeps their own step.
  assert.equal(freeSpot([29, 61, 95, 9000], 45, axisOf(0, 27000)), 45);
  assert.equal(freeSpot([29, 61, 95, 9000], 61.4, axisOf(0, 27000)), 62);
  // On an ordered span with no room left, it goes back where it was.
  const tight = knotAxis(orderedTerm({ knots: { lo: 0, hi: 0.25 } }));
  assert.equal(freeSpot([0.1], 0.12, tight), null);
});

test("a knot's room is held to the round-off of the values, at any magnitude", () => {
  // On 1e7 to 1e7 + 5 the step is 0.1, and 1e7 + 0.1 reads 0.1 - 3.7e-10 from
  // 1e7, which a 1e-9 relative slack refused.
  const far = axisOf(1e7, 1e7 + 5);
  assert.equal(knotFits(1e7 + 0.1, [], far), true);
  assert.equal(knotFits(1e7 + 2.1, [1e7 + 2], far), true);
  assert.equal(knotFits(1e7 + 0.1 - 1e-8, [], far), false);
  // Near zero, 5e-11 short of the step is outside the round-off.
  const near = axisOf(0, 5);
  assert.equal(knotFits(0.1, [], near), true);
  assert.equal(knotFits(0.1 - 5e-11, [], near), false);
  // Where five roundings reach the step, float64 cannot tell a knot from its end.
  assert.equal(knotFits(1e16 + 2, [], axisOf(1e16, 1e16 + 10)), false);
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
  const axis = knotAxis(numericTerm());
  assert.equal(nudgeKnot([3, 5], 0, 1, 1, axis), 3.1);
  assert.equal(nudgeKnot([3, 5], 0, 1, 10, axis), 4);
  assert.equal(nudgeKnot([3, 3.1], 0, 1, 1, axis), 3.2);
  assert.equal(nudgeKnot([3, 3.1], 1, -1, 1, axis), 2.9);
  // At the end of the axis there is nowhere further to go.
  assert.equal(nudgeKnot([0.1, 5], 0, -1, 1, axis), null);
  // A knot among others crowded where the data is dense moves by their step.
  assert.equal(nudgeKnot([1000, 1010, 1030, 9000], 1, 1, 1, axisOf(0, 10000)), 1011);
  assert.equal(nudgeKnot([1000, 1010, 1030, 9000], 3, -1, 1, axisOf(0, 10000)), 8900);
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

test("the chip's hover text names a penalty other than the standard one, for the knots in force", () => {
  assert.equal(knotPenaltyTitle(numericTerm()), null);
  assert.equal(knotPenaltyTitle(numericTerm({ knots: { difference_penalty: "standard" } })), null);
  assert.equal(
    knotPenaltyTitle(numericTerm({ knots: { difference_penalty: "general" } })),
    "Smoothing penalty: general, for unevenly spaced knots.",
  );
  assert.equal(
    knotPenaltyTitle(numericTerm({ knots: { difference_penalty: "projected" } })),
    "Smoothing penalty: standard, adjusted for knots too uneven for the general one.",
  );
  const waiting = numericTerm({
    knots: { difference_penalty: "projected" },
    pending: { knots: { positions: [5], count: 1, strategy: "quantile_rows", alpha: 0.2 } },
  });
  assert.equal(knotPenaltyTitle(waiting), null);
});

test("the chip's hover text hides the penalty while a kind, shape or level change waits too", () => {
  const general = { difference_penalty: "general" };
  const shown = "Smoothing penalty: general, for unevenly spaced knots.";
  // The payload's pending record when nothing waits: every field null or empty.
  const nothing = {
    groups: null, ranges: [], reference: null, specials: null, knots: null, basis: null,
  };
  assert.equal(knotPenaltyTitle(numericTerm({ knots: general, pending: nothing })), shown);
  const kind = { ...nothing, basis: { kind: "cr", select: false } };
  assert.equal(knotPenaltyTitle(numericTerm({ knots: general, pending: kind })), null);
  const shaped = {
    ...nothing,
    ranges: [{ lo: 2, hi: 6, degree: 1, join: "tangent", label: "Line" }],
    basis: { kind: "bs", select: false },
  };
  assert.equal(knotPenaltyTitle(numericTerm({ knots: general, pending: shaped })), null);
  const collapsed = { ...nothing, groups: { "18-34": ["18-24", "25-34"] } };
  assert.equal(knotPenaltyTitle({ ...orderedTerm({ knots: general }), pending: collapsed }), null);
  const special = { ...nothing, specials: ["65+"] };
  assert.equal(knotPenaltyTitle({ ...orderedTerm({ knots: general }), pending: special }), null);
  assert.equal(knotPenaltyTitle({ ...orderedTerm({ knots: general }), pending: nothing }), shown);
});

test("a waiting reset reads as listed in code, a waiting hand move as placed by hand", () => {
  const reset = numericTerm({
    knots: { positions: [2, 5, 6, 8], strategy: "explicit", from_editor: true },
    pending: {
      knots: {
        positions: [2, 4, 6, 8], count: 4, strategy: "explicit", alpha: 0.2, from_editor: false,
      },
    },
  });
  assert.deepEqual(knotChip(reset), { text: "4 knots · listed in code", waiting: true });
  const moved = numericTerm({
    knots: { strategy: "explicit" },
    pending: {
      knots: {
        positions: [2, 5, 6, 8], count: 4, strategy: "explicit", alpha: 0.2, from_editor: true,
      },
    },
  });
  assert.deepEqual(knotChip(moved), { text: "4 knots · placed by hand", waiting: true });
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
  assert.equal(dragging.removeZone.labelX, (PLOT.left + PLOT.right) / 2);

  const removing = knotLayout(frame, { ...NO_UI, drag: { from: 4, x: 4, moved: true, remove: true } });
  const zone = removing.removeZone;
  assert.equal(removing.handles[1].cy, zone.top + zone.height / 2);
  assert.equal(removing.handles[1].removing, true);
  assert.equal(removing.tag, null);
  assert.equal(removing.guides.length, 3);
  // The knot drops into the zone's left half, so its label moves to the right half.
  assert.equal(removing.removeZone.labelX, (PLOT.left + 3 * PLOT.right) / 4);
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
function gestureHarness(term, { editing = true, staged = true, plot = PLOT, onChange = null } = {}) {
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
    onChange: onChange
      ? (params) => { changes.push(params); return onChange(params); }
      : async (params) => { changes.push(params); return staged; },
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
  // The change is answered before the next gesture.
  await new Promise((resolve) => setImmediate(resolve));

  pointer("pointerdown", px(6), 300);
  pointer("pointermove", px(6), 330);
  assert.equal(gestures.ui().drag.remove, true);
  pointer("pointerup", px(6), 330);
  assert.deepEqual(changes.at(-1), { positions: [2, 4, 8] });
  assert.equal(gestures.ui().selected, null);
});

test("a dropped knot stays where it was dropped until its change is answered", async () => {
  const { pointer, px, gestures, svg } = gestureHarness(numericTerm());
  pointer("pointerdown", px(4), 300);
  pointer("pointermove", px(5.1), 302);
  pointer("pointerup", px(5.1), 302);
  const shown = knotLayout(svg._knotFrame, gestures.ui());
  assert.deepEqual(shown.handles.map((handle) => handle.x), [2, 5.1, 6, 8]);
  assert.equal(shown.handles[1].placed, true);
  assert.deepEqual(shown.ghosts.map((ghost) => ghost.removed), [false]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(gestures.ui().pending, null);
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

const EVEN_ONLY = "The penalty order of 'curve' is above its degree, which needs evenly spaced knots; "
  + "change its count here, or lower m in code to place its knots freely.";

test("on a term that takes evenly spaced knots only, no knot moves or arrives by hand", () => {
  const term = numericTerm({ knots: { strategy: "quantile", even_only: EVEN_ONLY } });
  assert.equal(knotAxis(term).evenOnly, EVEN_ONLY);
  assert.deepEqual(stepRule(term), { strategy: "uniform", alpha: 0.2 });
  // The handles still show, but the band offers no new knot.
  const layout = knotLayout(knotFrame(term, PLOT, true), { ...NO_UI, hover: 5 });
  assert.equal(layout.handles.length, 4);
  assert.equal(layout.adding, null);

  const { pointer, key, px, changes, gestures } = gestureHarness(term);
  pointer("pointerdown", px(4), 300);
  pointer("pointermove", px(5.1), 300);
  pointer("pointerup", px(5.1), 300);
  assert.equal(gestures.ui().drag, null);
  assert.equal(gestures.ui().selected, null);
  assert.equal(gestures.message(), EVEN_ONLY);
  pointer("pointerdown", px(5), 296);
  pointer("pointerup", px(5), 296);
  pointer("pointermove", px(5), 296);
  assert.equal(gestures.ui().hover, null);
  key("ArrowRight");
  key("Delete");
  assert.deepEqual(changes, []);
  assert.equal(gestures.message(), EVEN_ONLY);
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

/**
 * A /stage that answers only when told to, as the server does some time
 * after a change is sent; ``answer`` also redraws the frame from the payload.
 */
function slowStage() {
  const waiting = [];
  return {
    onChange: () => new Promise((resolve) => waiting.push(resolve)),
    get sent() { return waiting.length; },
    async answer(harness, staged, positions) {
      if (staged) {
        harness.svg._knotFrame = knotFrame(numericTerm({
          pending: { knots: { positions, count: positions.length, strategy: "explicit", alpha: 0.2 } },
        }), PLOT, true);
      }
      waiting.shift()(staged);
      await new Promise((resolve) => setImmediate(resolve));
    },
  };
}

const drawnKnots = (harness) =>
  knotLayout(harness.svg._knotFrame, harness.gestures.ui()).handles.map((handle) => handle.x);

test("arrow keys pressed while a knot change is being staged move the same knot on", async () => {
  const stage = slowStage();
  const harness = gestureHarness(numericTerm(), { onChange: stage.onChange });
  const { pointer, key, px, changes, gestures } = harness;
  pointer("pointerdown", px(4), 300);
  pointer("pointerup", px(4), 300);
  key("ArrowRight");
  key("ArrowRight");
  key("ArrowRight");
  // One change is sent; the knot moves on from where the arrow keys left it.
  assert.deepEqual(changes, [{ positions: [2, 4.1, 6, 8] }]);
  assert.equal(gestures.ui().selected, 4.3);
  assert.deepEqual(drawnKnots(harness), [2, 4.3, 6, 8]);
  // Once the first is staged, the latest of the presses follows it, and the
  // knot never flashes back on the way.
  await stage.answer(harness, true, [2, 4.1, 6, 8]);
  assert.deepEqual(changes.at(-1), { positions: [2, 4.3, 6, 8] });
  assert.equal(changes.length, 2);
  assert.deepEqual(drawnKnots(harness), [2, 4.3, 6, 8]);
  await stage.answer(harness, true, [2, 4.3, 6, 8]);
  assert.equal(gestures.ui().pending, null);
  key("ArrowRight");
  assert.deepEqual(changes.at(-1), { positions: [2, 4.4, 6, 8] });
});

test("a click while a dropped knot is being staged adds to the knots as dropped", async () => {
  const stage = slowStage();
  const harness = gestureHarness(numericTerm(), { onChange: stage.onChange });
  const { pointer, px, changes } = harness;
  pointer("pointerdown", px(4), 300);
  pointer("pointermove", px(5.1), 302);
  pointer("pointerup", px(5.1), 302);
  pointer("pointerdown", px(9), 296);
  pointer("pointerup", px(9), 296);
  assert.deepEqual(drawnKnots(harness), [2, 5.1, 6, 8, 9]);
  await stage.answer(harness, true, [2, 5.1, 6, 8]);
  assert.deepEqual(changes, [{ positions: [2, 5.1, 6, 8] }, { positions: [2, 5.1, 6, 8, 9] }]);
});

test("a change that is not staged drops the one waiting to follow it and gives the selection back", async () => {
  const stage = slowStage();
  const harness = gestureHarness(numericTerm(), { onChange: stage.onChange });
  const { pointer, key, px, changes, gestures } = harness;
  pointer("pointerdown", px(4), 300);
  pointer("pointerup", px(4), 300);
  key("ArrowRight");
  key("ArrowRight");
  await stage.answer(harness, false);
  assert.equal(changes.length, 1);
  assert.equal(gestures.ui().selected, 4);
  assert.deepEqual(drawnKnots(harness), [2, 4, 6, 8]);
});

test("while a kind change waits, the frame carries the basis it puts in force", () => {
  const inForce = { degree: 2, ends: "open", boundary: [0, 10], level_values: null };
  const waiting = { degree: 3, ends: "clamped", boundary: [0, 10], level_values: null };
  assert.equal(knotFrame(numericTerm({ knots: { basis: inForce } }), PLOT, true).basis, inForce);
  const frame = knotFrame(
    numericTerm({ knots: { basis: inForce, waiting_basis: waiting } }), PLOT, true,
  );
  assert.equal(frame.basis, waiting);
});
