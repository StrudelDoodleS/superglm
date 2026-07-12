import assert from "node:assert/strict";
import test from "node:test";

import { createEditorActions } from "../../src/superglm/editor/app/state/actions.js";
import {
  createEditorStore,
  createInitialEditorState
} from "../../src/superglm/editor/app/state/store.js";

const summaryModulePath = "../../src/superglm/editor/app/summary.js";
const {
  collapseTransition,
  renderSummary,
  uncollapseTransition,
  ungroupTransition
} = await import(summaryModulePath);

/** @param {number} revision */
function snapshot(revision) {
  return {
    model_revision: revision,
    selected_term: "age",
    terms: {
      age: {
        kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1],
        previous_y: null, levels: null, n_points: 1, controls: null,
        group_display: null, impact: {}
      }
    },
    selection: { age: [0] },
    can_uncollapse_levels: false,
    last_collapse: null,
    history: { active: [], redo: [] }
  };
}

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

test("rendering unchanged summary markup preserves the existing table DOM", () => {
  let writes = 0;
  let markup = "";
  const summaryFrame = {
    get innerHTML() { return markup; },
    set innerHTML(value) {
      writes += 1;
      markup = value;
    }
  };
  const nodes = {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame
  };
  const payload = { available: false, label: "Summary", error: "Unavailable" };

  renderSummary(payload, nodes);
  renderSummary(payload, nodes);

  assert.equal(writes, 1);
});

test("state-only recovery publishes a stale summary payload when remote summary is null", async () => {
  const nodes = {
    summaryStatus: { textContent: "Old manual summary" },
    summaryNote: { textContent: "old note" },
    summaryFrame: { innerHTML: "<p>old manual summary html</p>" }
  };
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  let summaryRenders = 0;
  store.subscribe((state) => state.remote.summary, (summary) => {
    summaryRenders += 1;
    assert.ok(summary);
    renderSummary(summary, nodes);
  });
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw new Error("response lost"); },
      getState: async () => snapshot(3)
    }
  });

  const result = await actions.executeStateMutation({ name: "drag", path: "/drag", payload: {} });

  assert.equal(result.ok, false);
  assert.equal(store.getState().remote.snapshot?.model_revision, 3);
  assert.equal(store.getState().remote.summary?.available, false);
  assert.doesNotThrow(() => JSON.stringify(store.getState().remote.summary));
  assert.equal(summaryRenders, 1);
  assert.equal(nodes.summaryStatus.textContent, "Summary unavailable");
  assert.match(nodes.summaryFrame.innerHTML, /reconciled/i);
  assert.match(nodes.summaryFrame.innerHTML, /refresh/i);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /old manual summary html/);
});
