import assert from "node:assert/strict";
import test from "node:test";

const summaryModulePath = "../../src/superglm/editor/app/summary.js";
const {
  collapseTransition,
  renderStaleSummary,
  uncollapseTransition,
  ungroupTransition
} = await import(summaryModulePath);

test("structural transition descriptors are pure route descriptions", () => {
  assert.deepEqual(collapseTransition("region"), {
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(ungroupTransition("region"), {
    name: "ungroup levels",
    path: "/ungroup_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(uncollapseTransition(), {
    name: "restore collapsed levels",
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

test("reconciled state replaces stale summary content with an unavailable message", () => {
  const nodes = {
    summaryStatus: { textContent: "Old summary" },
    summaryNote: { textContent: "old note" },
    summaryFrame: { innerHTML: "<p>stale model summary</p>" }
  };

  renderStaleSummary(nodes);

  assert.equal(nodes.summaryStatus.textContent, "Summary unavailable");
  assert.equal(nodes.summaryNote.textContent, "");
  assert.match(nodes.summaryFrame.innerHTML, /reconciled/i);
  assert.match(nodes.summaryFrame.innerHTML, /stale/i);
  assert.match(nodes.summaryFrame.innerHTML, /refresh/i);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /stale model summary/);
});
