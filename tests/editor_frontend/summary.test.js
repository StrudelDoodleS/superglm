import assert from "node:assert/strict";
import test from "node:test";

const summaryModulePath = "../../src/superglm/editor/app/summary.js";
const {
  collapseTransition,
  uncollapseTransition,
  ungroupTransition
} = await import(summaryModulePath);

test("structural transition descriptors are pure route descriptions", () => {
  assert.deepEqual(collapseTransition("region"), {
    name: "Refitting collapsed levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(ungroupTransition("region"), {
    name: "Refitting ungrouped levels",
    path: "/ungroup_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(uncollapseTransition(), {
    name: "Restoring previous collapsed-level model",
    path: "/uncollapse_levels",
    payload: {}
  });
});

test("transition descriptor payloads are independent caller-owned values", () => {
  const first = collapseTransition("region");
  first.payload.term = "mutated";

  assert.deepEqual(collapseTransition("region").payload, {
    term: "region",
    method: "auto"
  });
});
