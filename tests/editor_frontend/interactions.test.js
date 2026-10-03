// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

const interactionsModulePath = "../../src/superglm/editor/app/interactions.js";
const { bindInteractions } = await import(interactionsModulePath);

// A press draws a brush rectangle; the gestures only need it to exist.
globalThis.document = {
  createElementNS: () => ({ setAttribute() {}, remove() {} })
};

function selectionHarness({ displayIsCollapsed, displayToSourceIndices = undefined }) {
  const listeners = new Map();
  const mutations = [];
  const svg = {
    _scale: { displayIsCollapsed, displayToSourceIndices },
    viewBox: { baseVal: { x: 0, y: 0, width: 100, height: 100 } },
    addEventListener(name, listener) {
      listeners.set(name, listener);
    },
    removeEventListener() {},
    setPointerCapture() {},
    appendChild() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
  const term = {
    term_type: "categorical",
    levels: ["a", "b", "c", "d"],
    level_groups: [{ indices: [0, 1, 2, 3] }],
  };
  const context = {
    svg,
    currentTerm: () => term,
    currentSelection: () => new Set(),
    selectionAnchor: () => null,
    setSelectionAnchor() {},
    mode: () => "select",
    selectedTerm: () => "feature",
    actions: {
      async executeSelectionMutation(payload) {
        mutations.push(payload);
      },
    },
  };

  bindInteractions(context);

  return {
    mutations,
    async ctrlClick(displayIndex) {
      const event = {
        button: 0,
        pointerId: 1,
        clientX: 10,
        clientY: 10,
        ctrlKey: true,
        metaKey: false,
        shiftKey: false,
        target: { dataset: { index: String(displayIndex) } },
        preventDefault() {},
      };
      await listeners.get("pointerdown")(event);
      await listeners.get("pointerup")(event);
    },
  };
}

function moveHarness({ mutationResult, levelGroups = [], mode = "move" }) {
  const listeners = new Map();
  const previews = [];
  const mutations = [];
  let clears = 0;
  const svg = {
    _scale: {
      displayIsCollapsed: false,
      yMin: 0,
      yMax: 10,
      margin: { top: 0 },
      innerH: 100,
    },
    viewBox: { baseVal: { x: 0, y: 0, width: 100, height: 100 } },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener() {},
    setPointerCapture() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
  };
  const term = {
    term_type: "categorical",
    levels: ["a", "b", "c", "d"],
    level_groups: levelGroups,
    y: [2, 2, 4, 5],
    controls: mode === "handles" ? { y: [2], count: 1, basis: null } : null,
  };
  const context = {
    svg,
    currentTerm: () => term,
    currentSelection: () => new Set(),
    mode: () => mode,
    selectedTerm: () => "feature",
    setPreviewTerm(_term, preview, selection) {
      previews.push({ preview, selection });
    },
    clearPreviewTerm() { clears += 1; },
    actions: {
      async executeStateMutation(descriptor) {
        mutations.push(descriptor);
        return mutationResult;
      },
    },
  };

  bindInteractions(context);

  return {
    term,
    previews,
    mutations,
    get clears() { return clears; },
    async drag(displayIndex, clientY = 40) {
      const event = {
        button: 0,
        shiftKey: false,
        target: {
          dataset: mode === "handles"
            ? { controlIndex: String(displayIndex) }
            : { index: String(displayIndex) }
        },
        pointerId: 1,
        clientX: 25,
        clientY,
      };
      await listeners.get("pointerdown")(event);
      listeners.get("pointermove")(event);
      await listeners.get("pointerup")(event);
    },
  };
}

test("expanded grouped levels remain individually selectable for regrouping", async () => {
  const harness = selectionHarness({ displayIsCollapsed: false });

  await harness.ctrlClick(0);

  assert.deepEqual(harness.mutations, [{ term: "feature", indices: [0] }]);
});

test("collapsed group points select all represented source levels", async () => {
  const harness = selectionHarness({
    displayIsCollapsed: true,
    displayToSourceIndices: [[0, 1, 2, 3]],
  });

  await harness.ctrlClick(0);

  assert.deepEqual(harness.mutations, [{ term: "feature", indices: [0, 1, 2, 3] }]);
});

test("a skipped point drag clears its private curve preview", async () => {
  const harness = moveHarness({ mutationResult: { ok: false, skipped: true } });

  await harness.drag(0);

  assert.equal(harness.previews.length > 0, true);
  assert.equal(harness.clears, 1);
});

test("a skipped handle drag clears its private curve preview", async () => {
  const harness = moveHarness({
    mutationResult: { ok: false, skipped: true },
    mode: "handles",
  });

  await harness.drag(0);

  assert.equal(harness.previews.length > 0, true);
  assert.equal(harness.clears, 1);
});

test("expanded structural groups move together without widening individual selection", async () => {
  const harness = moveHarness({
    mutationResult: { ok: true, snapshot: {} },
    levelGroups: [{ label: "a + b", indices: [0, 1] }],
  });

  await harness.drag(0);

  const latest = harness.previews.at(-1);
  assert.deepEqual(latest.selection, [0]);
  assert.equal(latest.preview.y[0], latest.preview.y[1]);
  assert.deepEqual(harness.mutations[0].payload.indices, [0]);
});

// SVG units are client pixels here: data x maps to 100 + 10x and relativity
// to 300 - 100y, on a plot 200 wide and 300 tall.
const SPLINE_X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const SPLINE_Y = [1, 2.5, 0.5, 2, 1, 2.8, 0.2, 1.5, 1, 2];

function gestureHarness({
  x = SPLINE_X,
  y = SPLINE_Y,
  levels = null,
  selection = [],
  anchor = null,
  displayToSourceIndices = null,
}) {
  const listeners = new Map();
  const mutations = [];
  const zooms = [];
  const state = { selection: new Set(selection), anchor, span: null };
  const scale = {
    sx: (value) => 100 + 10 * value,
    sy: (value) => 300 - 100 * value,
    x,
    y,
    xMin: 0, xMax: 20, yMin: 0, yMax: 3,
    baseXMin: 0, baseXMax: 20, baseYMin: 0, baseYMax: 3,
    margin: { left: 100, top: 0 },
    innerW: 200,
    innerH: 300,
    displayIsCollapsed: displayToSourceIndices !== null,
    displayToSourceIndices: displayToSourceIndices ?? x.map((_, i) => [i]),
  };
  const svg = {
    _scale: scale,
    viewBox: { baseVal: { x: 0, y: 0, width: 400, height: 400 } },
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener() {},
    setPointerCapture() {},
    appendChild() {},
    getScreenCTM() { return null; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 400, height: 400 }; },
  };
  const term = { term_type: levels ? "categorical" : "spline", x, y, levels };
  const context = {
    svg,
    currentTerm: () => term,
    currentSelection: () => new Set(state.selection),
    selectionAnchor: () => state.anchor,
    setSelectionAnchor(next) { state.anchor = next; },
    setSelectionSpan(next) { state.span = next; },
    mode: () => "select",
    selectedTerm: () => "age",
    setZoom(_term, range) { zooms.push(range); },
    clearZoom() {},
    setPreviewTerm() {},
    clearPreviewTerm() {},
    actions: {
      async executeSelectionMutation(payload) {
        mutations.push(payload);
        state.selection = new Set(payload.indices);
        return { ok: true };
      },
    },
  };
  bindInteractions(context);

  const pointAt = (i) => ({ x: scale.sx(x[i]), y: scale.sy(y[i]) });
  // Press, move once, release: on a point when `index` is given, else on
  // whatever lies under `at` (the curve, or empty plot).
  async function gesture({ index = null, at, to = at, keys = {} }) {
    const base = {
      button: 0,
      pointerId: 1,
      shiftKey: false,
      ctrlKey: false,
      metaKey: false,
      ...keys,
      target: { dataset: index === null ? {} : { index: String(index) } },
      preventDefault() {},
    };
    await listeners.get("pointerdown")({ ...base, clientX: at.x, clientY: at.y });
    listeners.get("pointermove")({ ...base, clientX: to.x, clientY: to.y });
    await listeners.get("pointerup")({ ...base, clientX: to.x, clientY: to.y });
  }

  return {
    mutations,
    zooms,
    state,
    pointAt,
    click: (i, keys = {}) => gesture({ index: i, at: pointAt(i), keys }),
    clickAt: (at, keys = {}) => gesture({ at, keys }),
    drag: (i, to, keys = {}) => gesture({ index: i, at: pointAt(i), to, keys }),
  };
}

test("a click on a point selects it alone and makes it the anchor", async () => {
  const chart = gestureHarness({ selection: [7, 8] });

  await chart.click(3);

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [3] }]);
  assert.deepEqual(chart.state.anchor, { term: "age", index: 3 });
});

test("Shift-click selects every point between the anchor and the point by x, whatever their heights", async () => {
  const chart = gestureHarness({});

  await chart.click(2);
  await chart.click(6, { shiftKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [2, 3, 4, 5, 6] });

  // The anchor stays put, so another Shift-click spans from it again.
  await chart.click(0, { shiftKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [0, 1, 2] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 2 });
  assert.deepEqual(chart.zooms, []);
});

test("a Shift-click keeps the span it made, from the anchor to the point clicked", async () => {
  const chart = gestureHarness({});

  await chart.click(6);
  await chart.click(2, { shiftKey: true });
  assert.deepEqual(chart.state.span, { term: "age", from: 6, to: 2, indices: [2, 3, 4, 5, 6] });

  // Drawn collapsed, each end is the source level its display point shows first.
  const collapsed = gestureHarness({
    x: [0, 1, 2, 3],
    y: [1, 1.2, 0.8, 1.1],
    levels: ["a", "b", "c", "d", "e"],
    anchor: { term: "age", index: 4 },
    displayToSourceIndices: [[0], [1, 2], [3], [4]],
  });
  await collapsed.click(1, { shiftKey: true });
  assert.deepEqual(collapsed.state.span, { term: "age", from: 4, to: 1, indices: [1, 2, 3, 4] });
});

test("without an anchor on this term a Shift-click selects the point and anchors it", async () => {
  const chart = gestureHarness({ anchor: { term: "other", index: 1 } });

  await chart.click(5, { shiftKey: true });

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [5] }]);
  assert.deepEqual(chart.state.anchor, { term: "age", index: 5 });
});

test("a Shift press pans only once it moves past the click slop", async () => {
  const chart = gestureHarness({});
  const from = chart.pointAt(4);

  await chart.drag(4, { x: from.x + 30, y: from.y }, { shiftKey: true });
  assert.deepEqual(chart.mutations, []);
  assert.ok(chart.zooms.length > 0);

  const panned = chart.zooms.length;
  await chart.drag(4, { x: from.x + 2, y: from.y - 2 }, { shiftKey: true });
  assert.equal(chart.zooms.length, panned);
  assert.deepEqual(chart.mutations, [{ term: "age", indices: [4] }]);
});

test("Ctrl/Cmd-click toggles one point on a spline and moves the anchor", async () => {
  const chart = gestureHarness({ selection: [1, 2] });

  await chart.click(5, { ctrlKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [1, 2, 5] });

  await chart.click(2, { metaKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [1, 5] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 2 });
});

test("a click on the curve between points snaps to the nearest point by x; off it nothing changes", async () => {
  // Points 3, 4 and 5 sit at (130, 190), (140, 180) and (150, 180).
  const chart = gestureHarness({ y: [1, 1, 1, 1.1, 1.2, 1.2, 1, 1, 1, 1] });

  // On the line from 3 to 4, nearer 4 by x.
  await chart.clickAt({ x: 137, y: 183 });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [4] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 4 });

  // 8 px above the line from 4 to 5, nearer 5 by x.
  await chart.clickAt({ x: 147, y: 172 });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [5] });

  // 16 px above that line, then empty plot: neither changes the selection.
  await chart.clickAt({ x: 145, y: 164 });
  await chart.clickAt({ x: 250, y: 20 });
  assert.equal(chart.mutations.length, 2);
});

test("on a collapsed display the anchor is a source level and the span selects every source level in it", async () => {
  // Display point 1 is the group b + c; the anchor, source level c, lies in it.
  const chart = gestureHarness({
    x: [0, 1, 2, 3],
    y: [1, 1.2, 0.8, 1.1],
    levels: ["a", "b", "c", "d", "e"],
    anchor: { term: "age", index: 2 },
    displayToSourceIndices: [[0], [1, 2], [3], [4]],
  });

  await chart.click(3, { shiftKey: true });

  assert.deepEqual(chart.mutations, [{ term: "age", indices: [1, 2, 3, 4] }]);
});

test("a click on a collapsed display anchors the source level it shows, not the display point", async () => {
  // Display point 3 shows source level e, index 4: the group b + c sits before it.
  const chart = gestureHarness({
    x: [0, 1, 2, 3],
    y: [1, 1.2, 0.8, 1.1],
    levels: ["a", "b", "c", "d", "e"],
    displayToSourceIndices: [[0], [1, 2], [3], [4]],
  });

  await chart.click(3);
  assert.deepEqual(chart.mutations, [{ term: "age", indices: [4] }]);
  assert.deepEqual(chart.state.anchor, { term: "age", index: 4 });

  // A Ctrl/Cmd-click anchors the source level too.
  await chart.click(2, { ctrlKey: true });
  assert.deepEqual(chart.mutations.at(-1), { term: "age", indices: [3, 4] });
  assert.deepEqual(chart.state.anchor, { term: "age", index: 3 });
});

test("a handle drag on an ordered spline moves its drawn curve with the level dots", async () => {
  const harness = moveHarness({ mutationResult: { ok: true }, mode: "handles" });
  const term = harness.term;
  term.y = [1, 1, 1, 1];
  term.controls = {
    y: [1],
    log_effect: [0],
    count: 1,
    basis_index: [0],
    basis: [[0.5, 1, 0.5, 0]],
    build_basis: [[0.25, 0.5, 1, 0.5, 0]],
    build_log_effect: [0],
    grid_x: [0, 0.5, 1, 1.5, 2],
  };
  term.spline_view = {
    available: true,
    reason: null,
    x: [0, 0.5, 1, 1.5, 2],
    y: [1, 1, 1, 1, 1],
    original_y: [1, 1, 1, 1, 1],
    level_indices: [0, 1, 2],
    fits_levels: true,
  };

  await harness.drag(0);

  const { preview } = harness.previews.at(-1);
  const deltaLog = Math.log(preview.controls.y[0]);
  assert.notEqual(deltaLog, 0);
  const expected = [0.25, 0.5, 1, 0.5, 0].map((row) => Math.exp(row * deltaLog));
  preview.spline_view.y.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= 1e-12 * expected[index]);
  });
  assert.ok(Math.abs(preview.y[1] - Math.exp(deltaLog)) <= 1e-12 * Math.exp(deltaLog));
  assert.equal(preview.y[3], 1);
});

test("a dragged level of an ordered spline is joined, not drawn on the stale spline", async () => {
  const harness = moveHarness({ mutationResult: { ok: true } });
  harness.term.spline_view = {
    available: true, reason: null, x: [0, 1], y: [2, 2], original_y: [2, 2],
    level_indices: [0, 1], fits_levels: true,
  };

  await harness.drag(0);

  assert.equal(harness.previews[0].preview.spline_view.fits_levels, false);
  assert.equal(harness.term.spline_view.fits_levels, true);
});
