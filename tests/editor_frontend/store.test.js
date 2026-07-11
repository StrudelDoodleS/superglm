import assert from "node:assert/strict";
import test from "node:test";

import * as selectors from "../../src/superglm/editor/app/state/selectors.js";
import * as storeModule from "../../src/superglm/editor/app/state/store.js";

/** @typedef {import('../../src/superglm/editor/app/api/contracts.js').EditorState} EditorState */

const {
  beginEvidence,
  commitRemote,
  commitStructuralTransition,
  completeEvidence,
  createEditorStore,
  createInitialEditorState,
  failEvidence,
  patchView,
  setPreviewTerm
} = storeModule;
const {
  selectActiveTermName,
  selectCurrentSelection,
  selectCurrentTerm,
  selectEvidence,
  selectGroupDisplayMode,
  selectModelRevision,
  selectMutation,
  selectRenderableTerm,
  selectSnapshot
} = selectors;

/** @returns {import('../../src/superglm/editor/app/api/contracts.js').EditorSnapshot} */
function snapshot(revision = 0) {
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

function transitionEnvelope() {
  return {
    state: snapshot(7),
    summary: { available: true, source: "in_force" },
    timing: {
      operation: "collapse_levels",
      fit_ms: 6,
      summary_ms: 2,
      state_ms: 1,
      server_total_ms: 12
    }
  };
}

test("initial remote state has no summary and ordinary commits preserve the confirmed summary", () => {
  const initial = createInitialEditorState(snapshot(1));
  assert.deepEqual(initial.remote, { snapshot: initial.remote.snapshot, summary: null });

  const envelope = transitionEnvelope();
  const structural = commitStructuralTransition(initial, envelope);
  const store = createEditorStore(structural);
  let summaryNotifications = 0;
  store.subscribe((state) => state.remote.summary, () => { summaryNotifications += 1; });
  store.update((state) => commitRemote(state, snapshot(8)));
  const ordinary = store.getState();

  assert.strictEqual(ordinary.remote.summary, envelope.summary);
  assert.equal(summaryNotifications, 0);
});

test("structural transition commits snapshot and summary atomically without ending the mutation", () => {
  const envelope = transitionEnvelope();
  let initial = patchView(createInitialEditorState(snapshot(1)), { activeTerm: "removed" });
  initial = {
    ...initial,
    request: {
      ...initial.request,
      mutation: { status: "running", operation: "collapse", error: null }
    }
  };
  const store = createEditorStore(initial);
  /** @type {EditorState['remote'][]} */
  const commits = [];
  /** @type {Array<EditorState['remote']['snapshot']>} */
  const snapshots = [];
  /** @type {Array<EditorState['remote']['summary']>} */
  const summaries = [];
  store.subscribe((state) => state.remote, (remote) => commits.push(remote));
  store.subscribe((state) => state.remote.snapshot, (snapshot) => snapshots.push(snapshot));
  store.subscribe((state) => state.remote.summary, (summary) => summaries.push(summary));

  store.update((state) => commitStructuralTransition(state, envelope));

  const committed = store.getState();
  assert.equal(commits.length, 1);
  assert.equal(snapshots.length, 1);
  assert.equal(summaries.length, 1);
  assert.deepEqual(commits[0], { snapshot: envelope.state, summary: envelope.summary });
  assert.strictEqual(snapshots[0], envelope.state);
  assert.strictEqual(summaries[0], envelope.summary);
  assert.strictEqual(committed.remote.snapshot, envelope.state);
  assert.strictEqual(committed.remote.summary, envelope.summary);
  assert.equal(committed.view.activeTerm, "age");
  assert.equal(committed.request.mutation.status, "running");
  assert.equal(committed.request.mutation.operation, "collapse");
});

test("structural summary becomes the confirmed payload retained by a later failed refresh", () => {
  const oldSummary = { available: true, label: "Before structural refit" };
  const oldRetry = { path: "/summary", payload: { source: "in_force" } };
  let state = createInitialEditorState(snapshot(0));
  state = beginEvidence(state, "summary", 0, 1, oldRetry);
  state = completeEvidence(state, "summary", 0, 1, oldSummary);

  const envelope = transitionEnvelope();
  state = commitStructuralTransition(state, envelope);

  assert.deepEqual(state.request.evidence.summary, {
    status: "current",
    revision: 7,
    sequence: 1,
    payload: envelope.summary,
    error: null,
    retry: null
  });

  state = commitRemote(state, snapshot(8));
  state = beginEvidence(
    state,
    "summary",
    8,
    2,
    { path: "/summary", payload: { source: "in_force" } }
  );
  assert.strictEqual(state.request.evidence.summary.payload, envelope.summary);

  state = failEvidence(state, "summary", 8, 2, "offline");
  assert.equal(state.request.evidence.summary.status, "stale");
  assert.strictEqual(state.request.evidence.summary.payload, envelope.summary);
  assert.equal(state.request.evidence.summary.error, "offline");
});

test("store keeps confirmed remote data separate from a chart preview", () => {
  const initial = createInitialEditorState(snapshot());
  const preview = { ...snapshot().terms.age, y: [1.4] };
  const next = setPreviewTerm(initial, "age", preview, [4, 2]);
  assert.deepEqual(next.remote.snapshot?.terms.age.y, [1]);
  assert.deepEqual(selectRenderableTerm(next)?.y, [1.4]);
  assert.deepEqual(next.view.preview, {
    term: "age",
    payload: preview,
    selection: [4, 2]
  });
});

test("remote commit clears preview and preserves valid view state", () => {
  let state = createInitialEditorState(snapshot());
  state = patchView(state, { mode: "move", showCi: true });
  state = setPreviewTerm(state, "age", { ...snapshot().terms.age, y: [1.4] });
  state = commitRemote(state, snapshot(1));
  assert.equal(state.view.mode, "move");
  assert.equal(state.view.showCi, true);
  assert.equal(state.view.preview, null);
  assert.equal(selectActiveTermName(state), "age");
  assert.deepEqual(selectCurrentSelection(state), [0]);
});

test("selector subscriptions ignore unrelated state changes", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  let calls = 0;
  store.subscribe(selectActiveTermName, () => { calls += 1; });
  store.update((state) => patchView(state, { showCi: true }));
  assert.equal(calls, 0);
  store.update((state) => patchView(state, { activeTerm: "missing" }));
  assert.equal(calls, 0);
});

test("missing selections use one stable immutable empty value", () => {
  const missingSelection = snapshot();
  missingSelection.selection = {};
  const store = createEditorStore(createInitialEditorState(missingSelection));
  const first = selectCurrentSelection(store.getState());
  let calls = 0;
  store.subscribe(selectCurrentSelection, () => { calls += 1; });

  assert.strictEqual(selectCurrentSelection(store.getState()), first);
  assert.equal(Object.isFrozen(first), true);

  store.update((state) => patchView(state, { showCi: true }));

  assert.equal(calls, 0);
  assert.strictEqual(selectCurrentSelection(store.getState()), first);
});

test("initial evidence panel states are independent objects", () => {
  const state = createInitialEditorState(snapshot());
  const { metrics, summary, report } = state.request.evidence;
  assert.notStrictEqual(metrics, summary);
  assert.notStrictEqual(metrics, report);
  assert.notStrictEqual(summary, report);
});

test("view patches are immutable and leave prior state untouched", () => {
  const prior = createInitialEditorState(snapshot());
  const next = patchView(prior, { showContrib: true, inspectorPane: "help" });

  assert.notStrictEqual(next, prior);
  assert.notStrictEqual(next.view, prior.view);
  assert.strictEqual(next.remote, prior.remote);
  assert.strictEqual(next.request, prior.request);
  assert.equal(prior.view.showContrib, false);
  assert.equal(prior.view.inspectorPane, "summary");
  assert.equal(next.view.showContrib, true);
  assert.equal(next.view.inspectorPane, "help");
});

test("a no-op store update does not run selectors or notify listeners", () => {
  const initial = createInitialEditorState(snapshot());
  const store = createEditorStore(initial);
  let selectorCalls = 0;
  let calls = 0;
  store.subscribe((state) => {
    selectorCalls += 1;
    return state;
  }, () => { calls += 1; });

  assert.equal(selectorCalls, 1);

  store.update((state) => state);

  assert.strictEqual(store.getState(), initial);
  assert.equal(selectorCalls, 1);
  assert.equal(calls, 0);
});

test("subscriptions notify only for selected changes and stop after unsubscribe", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  /** @type {Array<[boolean, boolean]>} */
  const notifications = [];
  const unsubscribe = store.subscribe(
    (state) => state.view.showCi,
    (value, previous) => notifications.push([value, previous])
  );

  store.update((state) => patchView(state, { showContrib: true }));
  store.update((state) => patchView(state, { showCi: true }));
  store.update((state) => patchView(state, { inspectorOpen: false }));
  unsubscribe();
  store.update((state) => patchView(state, { showCi: false }));

  assert.deepEqual(notifications, [[true, false]]);
});

test("nested updates refresh selector caches before notifying remaining subscribers", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  /** @type {string[]} */
  const notifications = [];
  store.subscribe(
    (state) => state.view.showCi,
    (value, previous) => {
      notifications.push(`trigger:${previous}->${value}`);
      if (value) {
        store.update((state) => patchView(state, { showContrib: true }));
      }
    }
  );
  store.subscribe(
    (state) => `${state.view.showCi}:${state.view.showContrib}`,
    (value, previous) => notifications.push(`peer:${previous}->${value}`)
  );

  store.update((state) => patchView(state, { showCi: true }));

  assert.equal(store.getState().view.showContrib, true);
  assert.deepEqual(notifications, [
    "trigger:false->true",
    "peer:false:false->true:true"
  ]);
});

test("subscription changes during notification take effect immediately", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  let controllerArmed = true;
  let removedCalls = 0;
  let addedCalls = 0;
  let unsubscribeRemoved = () => {};

  store.subscribe(
    (state) => state.view.showCi,
    () => {
      if (!controllerArmed) return;
      controllerArmed = false;
      unsubscribeRemoved();
      store.subscribe(
        (state) => state.view.showCi,
        () => { addedCalls += 1; }
      );
    }
  );
  unsubscribeRemoved = store.subscribe(
    (state) => state.view.showCi,
    () => { removedCalls += 1; }
  );

  store.update((state) => patchView(state, { showCi: true }));

  assert.equal(removedCalls, 0);
  assert.equal(addedCalls, 0);

  store.update((state) => patchView(state, { showCi: false }));
  assert.equal(removedCalls, 0);
  assert.equal(addedCalls, 1);
});

test("listener errors do not starve later subscribers and rethrow after notification", () => {
  const store = createEditorStore(createInitialEditorState(snapshot()));
  /** @type {string[]} */
  const notifications = [];
  let throwOnce = true;
  store.subscribe(
    (state) => state.view.showCi,
    (value, previous) => {
      notifications.push(`first:${previous}->${value}`);
      if (throwOnce) {
        throwOnce = false;
        throw new Error("listener exploded");
      }
    }
  );
  store.subscribe(
    (state) => state.view.showCi,
    (value, previous) => notifications.push(`second:${previous}->${value}`)
  );

  assert.throws(
    () => store.update((state) => patchView(state, { showCi: true })),
    /listener exploded/
  );

  assert.equal(store.getState().view.showCi, true);
  assert.deepEqual(notifications, ["first:false->true", "second:false->true"]);

  store.update((state) => patchView(state, { showCi: false }));
  assert.deepEqual(notifications, [
    "first:false->true",
    "second:false->true",
    "first:true->false",
    "second:true->false"
  ]);
});

test("remote commit falls back to the selected term when the active term disappears", () => {
  const initial = patchView(createInitialEditorState(snapshot()), { activeTerm: "missing" });
  const nextSnapshot = {
    ...snapshot(2),
    selected_term: "weight",
    terms: {
      weight: { ...snapshot().terms.age, y: [2] },
      height: { ...snapshot().terms.age, y: [3] }
    },
    selection: { weight: [2], height: [3] }
  };

  const next = commitRemote(initial, nextSnapshot);

  assert.equal(next.view.activeTerm, "weight");
  assert.equal(selectCurrentTerm(next)?.y[0], 2);
});

test("remote commit falls back to the first term when selected term is empty", () => {
  const initial = patchView(createInitialEditorState(snapshot()), { activeTerm: "missing" });
  const nextSnapshot = {
    ...snapshot(3),
    selected_term: "",
    terms: {
      height: { ...snapshot().terms.age, y: [3] },
      weight: { ...snapshot().terms.age, y: [2] }
    },
    selection: { height: [3], weight: [2] }
  };

  assert.equal(commitRemote(initial, nextSnapshot).view.activeTerm, "height");
});

test("term resolution ignores a stale selected term", () => {
  const staleSelection = snapshot(4);
  staleSelection.selected_term = "removed";
  const initial = createInitialEditorState(staleSelection);

  assert.equal(initial.view.activeTerm, "removed");
  assert.equal(selectActiveTermName(initial), "age");
});

test("remote commit ignores a stale selected term when the active term disappeared", () => {
  const initial = patchView(createInitialEditorState(snapshot()), { activeTerm: "missing" });
  const nextSnapshot = {
    ...snapshot(5),
    selected_term: "removed",
    terms: {
      height: { ...snapshot().terms.age, y: [3] },
      weight: { ...snapshot().terms.age, y: [2] }
    },
    selection: { height: [3], weight: [2] }
  };

  assert.equal(commitRemote(initial, nextSnapshot).view.activeTerm, "height");
});

test("selectors expose confirmed state defaults and per-term display overrides", () => {
  const confirmed = snapshot(7);
  confirmed.terms.age.group_display = {
    available: true,
    default_mode: "collapsed",
    collapsed: null
  };
  let state = createInitialEditorState(confirmed);

  assert.strictEqual(selectSnapshot(state), confirmed);
  assert.equal(selectModelRevision(state), 7);
  assert.equal(selectActiveTermName(state), "age");
  assert.strictEqual(selectCurrentTerm(state), confirmed.terms.age);
  assert.deepEqual(selectCurrentSelection(state), [0]);
  assert.equal(selectGroupDisplayMode(state), "collapsed");
  assert.strictEqual(selectMutation(state), state.request.mutation);
  assert.strictEqual(selectEvidence("metrics")(state), state.request.evidence.metrics);

  state = patchView(state, { groupModeByTerm: { age: "expanded" } });
  assert.equal(selectGroupDisplayMode(state), "expanded");

  const empty = createInitialEditorState();
  assert.equal(selectModelRevision(empty), -1);
  assert.equal(selectActiveTermName(empty), "");
  assert.equal(selectCurrentTerm(empty), null);
  assert.deepEqual(selectCurrentSelection(empty), []);
  assert.equal(selectRenderableTerm(empty), null);
  assert.equal(selectGroupDisplayMode(empty), "expanded");
});

test("state modules expose only their requested public symbols", () => {
  assert.deepEqual(Object.keys(storeModule).sort(), [
    "beginEvidence",
    "commitRemote",
    "commitStructuralTransition",
    "completeEvidence",
    "createEditorStore",
    "createInitialEditorState",
    "failEvidence",
    "patchView",
    "setPreviewTerm"
  ]);
  assert.deepEqual(Object.keys(selectors).sort(), [
    "selectActiveTermName",
    "selectCurrentSelection",
    "selectCurrentTerm",
    "selectEvidence",
    "selectGroupDisplayMode",
    "selectModelRevision",
    "selectMutation",
    "selectRenderableTerm",
    "selectSnapshot"
  ]);
});
