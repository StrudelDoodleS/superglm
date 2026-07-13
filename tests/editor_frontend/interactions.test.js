// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

const interactionsModulePath = "../../src/superglm/editor/app/interactions.js";
const { bindInteractions } = await import(interactionsModulePath);

function selectionHarness({ displayIsCollapsed, displayToSourceIndices = undefined }) {
  const listeners = new Map();
  const mutations = [];
  const svg = {
    _scale: { displayIsCollapsed, displayToSourceIndices },
    addEventListener(name, listener) {
      listeners.set(name, listener);
    },
    removeEventListener() {},
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
      await listeners.get("pointerdown")({
        button: 0,
        ctrlKey: true,
        metaKey: false,
        shiftKey: false,
        target: { dataset: { index: String(displayIndex) } },
        preventDefault() {},
      });
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
