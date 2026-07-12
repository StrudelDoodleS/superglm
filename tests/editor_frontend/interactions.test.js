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
