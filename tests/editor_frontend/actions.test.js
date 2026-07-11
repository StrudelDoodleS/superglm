import assert from "node:assert/strict";
import test from "node:test";

import * as actionsModule from "../../src/superglm/editor/app/state/actions.js";
import {
  commitRemote,
  createEditorStore,
  createInitialEditorState,
  patchView as patchViewState,
  setPreviewTerm
} from "../../src/superglm/editor/app/state/store.js";

const { createEditorActions } = actionsModule;

/** @returns {import('../../src/superglm/editor/app/api/contracts.js').TermPayload} */
function termPayload() {
  return {
    kind: "spline",
    term_type: "spline",
    x: [1],
    y: [1],
    original_y: [1],
    previous_y: null,
    levels: null,
    n_points: 1,
    controls: null,
    group_display: null,
    impact: {}
  };
}

/** @param {number} revision @returns {import('../../src/superglm/editor/app/api/contracts.js').EditorSnapshot} */
function snapshot(revision) {
  return {
    model_revision: revision,
    selected_term: "age",
    terms: { age: termPayload() },
    selection: { age: [0] },
    can_uncollapse_levels: false,
    last_collapse: null,
    history: { active: [], redo: [] }
  };
}

/**
 * @template T
 * @returns {{promise:Promise<T>, resolve:(value:T)=>void, reject:(reason:unknown)=>void}}
 */
function deferred() {
  /** @type {(value:T|PromiseLike<T>)=>void} */
  let resolvePromise = () => {};
  /** @type {(reason:unknown)=>void} */
  let rejectPromise = () => {};
  const promise = new Promise((resolve, reject) => {
    resolvePromise = resolve;
    rejectPromise = reject;
  });
  return {
    promise,
    resolve: (value) => resolvePromise(value),
    reject: (reason) => rejectPromise(reason)
  };
}

test("initialize commits the authoritative snapshot and view patches stay local", async () => {
  const store = createEditorStore(createInitialEditorState());
  const authoritative = snapshot(4);
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => ({}), getState: async () => authoritative }
  });

  assert.strictEqual(await actions.initialize(), authoritative);
  assert.strictEqual(store.getState().remote.snapshot, authoritative);

  const remote = store.getState().remote;
  actions.patchView({ showCi: true });
  assert.equal(store.getState().view.showCi, true);
  assert.strictEqual(store.getState().remote, remote);
});

test("successful mutation commits once and schedules only a new revision", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(0)));
  /** @type {number[]} */
  const scheduled = [];
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => snapshot(1), getState: async () => snapshot(1) },
    scheduleEvidence: (revision) => { scheduled.push(revision); }
  });

  const result = await actions.executeStateMutation({
    name: "shift",
    path: "/op",
    payload: { operation: "shift_up" }
  });

  assert.equal(result.ok, true);
  assert.equal(store.getState().remote.snapshot?.model_revision, 1);
  assert.deepEqual(store.getState().request.mutation, {
    status: "idle", operation: null, error: null
  });
  assert.equal(store.getState().request.recovery, null);
  assert.deepEqual(scheduled, [1]);
});

test("an asynchronous scheduling error cannot turn a confirmed mutation into a failure", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(0)));
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => snapshot(1), getState: async () => snapshot(1) },
    scheduleEvidence: async () => { throw new Error("scheduler bug"); }
  });

  const result = await actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });

  assert.equal(result.ok, true);
  assert.equal(store.getState().remote.snapshot?.model_revision, 1);
  assert.equal(store.getState().request.mutation.status, "idle");
  assert.equal(store.getState().request.recovery, null);
  await Promise.resolve();
});

test("same-revision mutation clears preview without scheduling evidence", async () => {
  const confirmed = snapshot(2);
  let state = createInitialEditorState(confirmed);
  state = setPreviewTerm(state, "age", { ...termPayload(), y: [1.5] });
  const store = createEditorStore(state);
  /** @type {number[]} */
  const scheduled = [];
  const returned = snapshot(2);
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => returned, getState: async () => returned },
    scheduleEvidence: (revision) => { scheduled.push(revision); }
  });

  const result = await actions.executeStateMutation({
    name: "select",
    path: "/select",
    payload: { term: "age", indices: [0] }
  });

  assert.equal(result.ok, true);
  assert.strictEqual(store.getState().remote.snapshot, returned);
  assert.equal(store.getState().view.preview, null);
  assert.deepEqual(scheduled, []);
});

test("a running mutation suppresses duplicate submissions", async () => {
  const pendingResponse = deferred();
  const store = createEditorStore(createInitialEditorState(snapshot(0)));
  let calls = 0;
  const actions = createEditorActions({
    store,
    client: {
      postJSON: () => {
        calls += 1;
        return pendingResponse.promise;
      },
      getState: async () => snapshot(0)
    }
  });

  const first = actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });
  const second = await actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });

  assert.equal(second.ok, false);
  assert.equal(second.skipped, true);
  assert.ok(second.error instanceof Error);
  assert.equal(calls, 1);

  pendingResponse.resolve(snapshot(1));
  assert.equal((await first).ok, true);
});

test("failed mutation preserves confirmed state, clears preview, and records retry", async () => {
  const confirmed = snapshot(2);
  let state = createInitialEditorState(confirmed);
  state = setPreviewTerm(state, "age", { ...termPayload(), y: [1.8] });
  const store = createEditorStore(state);
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw new Error("network down"); },
      getState: async () => { throw new Error("still offline"); }
    }
  });
  const payload = { term: "age" };

  const result = await actions.executeStateMutation({ name: "drag", path: "/drag", payload });

  assert.equal(result.ok, false);
  assert.equal(result.error.message, "network down");
  assert.strictEqual(store.getState().remote.snapshot, confirmed);
  assert.equal(store.getState().view.preview, null);
  assert.deepEqual(store.getState().request.mutation, {
    status: "error", operation: "drag", error: "network down"
  });
  assert.deepEqual(store.getState().request.recovery, {
    message: "network down",
    retry: { name: "drag", path: "/drag", payload }
  });
});

test("mutation retry descriptors snapshot caller-owned payloads", async () => {
  const pendingResponse = deferred();
  const confirmed = snapshot(2);
  const store = createEditorStore(createInitialEditorState(confirmed));
  /** @type {Record<string, unknown>[]} */
  const requests = [];
  const actions = createEditorActions({
    store,
    client: {
      postJSON: (_path, payload) => {
        requests.push(payload);
        return pendingResponse.promise;
      },
      getState: async () => { throw new Error("offline"); }
    }
  });
  const payload = { operation: "drag", point: { index: 3 } };

  const pending = actions.executeStateMutation({ name: "drag", path: "/drag", payload });
  payload.operation = "mutated";
  payload.point.index = 99;
  pendingResponse.reject(new Error("network down"));
  await pending;

  assert.deepEqual(requests[0], { operation: "drag", point: { index: 3 } });
  const retry = store.getState().request.recovery?.retry;
  assert.ok(retry);
  assert.deepEqual(retry.payload, {
    operation: "drag", point: { index: 3 }
  });
});

test("failed mutation commits a recovered authoritative snapshot", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  const recovered = snapshot(3);
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw new Error("response lost"); },
      getState: async () => recovered
    }
  });

  const result = await actions.executeStateMutation({ name: "drag", path: "/drag", payload: {} });

  assert.equal(result.ok, false);
  assert.strictEqual(store.getState().remote.snapshot, recovered);
  assert.equal(store.getState().request.recovery?.message, "response lost");
});

test("failed recovery never rewinds a newer authoritative commit", async () => {
  const pendingResponse = deferred();
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  const newer = snapshot(3);
  const actions = createEditorActions({
    store,
    client: {
      postJSON: () => pendingResponse.promise,
      getState: async () => { throw new Error("offline"); }
    }
  });

  const pending = actions.executeStateMutation({ name: "drag", path: "/drag", payload: {} });
  store.update((state) => commitRemote(state, newer));
  pendingResponse.reject(new Error("response lost"));
  await pending;

  assert.strictEqual(store.getState().remote.snapshot, newer);
  assert.equal(store.getState().request.recovery?.message, "response lost");
});

test("late evidence cannot replace a newer revision", async () => {
  const pendingEvidence = deferred();
  const store = createEditorStore(createInitialEditorState(snapshot(3)));
  const actions = createEditorActions({
    store,
    client: { postJSON: () => pendingEvidence.promise, getState: async () => snapshot(3) }
  });

  const pending = actions.refreshEvidence("metrics", "/metrics", {});
  store.update((state) => commitRemote(state, snapshot(4)));
  pendingEvidence.resolve({ available: true });

  assert.equal(await pending, false);
  assert.equal(store.getState().request.evidence.metrics.payload, null);
  assert.equal(store.getState().request.evidence.metrics.status, "stale");
  assert.equal(store.getState().request.evidence.metrics.revision, 3);
});

test("only the latest same-revision evidence response is accepted", async () => {
  const firstResponse = deferred();
  const secondResponse = deferred();
  const responses = [firstResponse, secondResponse];
  const store = createEditorStore(createInitialEditorState(snapshot(3)));
  const actions = createEditorActions({
    store,
    client: {
      postJSON: () => {
        const response = responses.shift();
        assert.ok(response);
        return response.promise;
      },
      getState: async () => snapshot(3)
    }
  });

  const first = actions.refreshEvidence("metrics", "/metrics", { request: 1 });
  const second = actions.refreshEvidence("metrics", "/metrics", { request: 2 });
  secondResponse.resolve({ value: "latest" });
  assert.equal(await second, true);
  firstResponse.resolve({ value: "old" });
  assert.equal(await first, false);
  assert.deepEqual(store.getState().request.evidence.metrics.payload, { value: "latest" });
});

test("evidence errors retain payload and can retry from their descriptor", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(5)));
  store.update((state) => ({
    ...state,
    request: {
      ...state.request,
      evidence: {
        ...state.request.evidence,
        metrics: { ...state.request.evidence.metrics, payload: { value: "old" } }
      }
    }
  }));
  let calls = 0;
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => {
        calls += 1;
        if (calls === 1) throw new Error("metrics offline");
        return { value: "new" };
      },
      getState: async () => snapshot(5)
    }
  });

  assert.equal(await actions.refreshEvidence("metrics", "/metrics", { split: "train" }), false);
  assert.equal(store.getState().request.evidence.metrics.status, "error");
  assert.deepEqual(store.getState().request.evidence.metrics.payload, { value: "old" });
  assert.equal(store.getState().request.evidence.metrics.error, "metrics offline");
  assert.deepEqual(store.getState().request.evidence.metrics.retry, {
    path: "/metrics", payload: { split: "train" }
  });

  assert.equal(await actions.retryEvidence("metrics"), true);
  assert.equal(store.getState().request.evidence.metrics.status, "current");
  assert.deepEqual(store.getState().request.evidence.metrics.payload, { value: "new" });
  assert.equal(store.getState().request.evidence.metrics.retry, null);
});

test("evidence retry descriptors snapshot caller-owned payloads", async () => {
  const pendingResponse = deferred();
  const store = createEditorStore(createInitialEditorState(snapshot(5)));
  /** @type {Record<string, unknown>[]} */
  const requests = [];
  const actions = createEditorActions({
    store,
    client: {
      postJSON: (_path, payload) => {
        requests.push(payload);
        return pendingResponse.promise;
      },
      getState: async () => snapshot(5)
    }
  });
  const payload = { split: "train", options: { weighted: true } };

  const pending = actions.refreshEvidence("metrics", "/metrics", payload);
  payload.split = "test";
  payload.options.weighted = false;
  pendingResponse.reject(new Error("metrics offline"));
  await pending;

  assert.deepEqual(requests[0], { split: "train", options: { weighted: true } });
  assert.deepEqual(store.getState().request.evidence.metrics.retry?.payload, {
    split: "train", options: { weighted: true }
  });
});

test("stale evidence failures do not overwrite the newer revision", async () => {
  const pendingEvidence = deferred();
  const store = createEditorStore(createInitialEditorState(snapshot(6)));
  const actions = createEditorActions({
    store,
    client: { postJSON: () => pendingEvidence.promise, getState: async () => snapshot(6) }
  });

  const pending = actions.refreshEvidence("summary", "/summary", {});
  store.update((state) => commitRemote(state, snapshot(7)));
  pendingEvidence.reject(new Error("late failure"));

  assert.equal(await pending, false);
  assert.equal(store.getState().request.evidence.summary.error, null);
  assert.equal(store.getState().request.evidence.summary.status, "stale");
  assert.equal(store.getState().request.evidence.summary.revision, 6);
});

test("non-Error rejections use message properties and safe fallbacks", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(1)));
  let calls = 0;
  const revoked = Proxy.revocable({}, {});
  revoked.revoke();
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => {
        calls += 1;
        if (calls === 1) throw { message: "plain failure" };
        throw revoked.proxy;
      },
      getState: async () => { throw new Error("offline"); }
    }
  });

  const first = await actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });
  assert.equal(first.ok, false);
  if (first.ok) assert.fail("plain rejection unexpectedly succeeded");
  assert.equal(first.error.message, "plain failure");

  actions.dismissRecovery();
  const second = await actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });
  assert.equal(second.ok, false);
  if (second.ok) assert.fail("hostile rejection unexpectedly succeeded");
  assert.equal(second.error.message, "Editor request failed.");
});

test("mutation retry replays its stored descriptor", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(1)));
  let calls = 0;
  /** @type {Array<{path:string, payload:Record<string, unknown>}>} */
  const requests = [];
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async (path, payload) => {
        calls += 1;
        requests.push({ path, payload });
        if (calls === 1) throw new Error("once");
        return snapshot(2);
      },
      getState: async () => { throw new Error("offline"); }
    }
  });

  const payload = { operation: "shift_up" };
  await actions.executeStateMutation({ name: "shift", path: "/op", payload });
  assert.equal(store.getState().request.mutation.status, "error");

  assert.equal((await actions.retryMutation()).ok, true);
  assert.equal(store.getState().remote.snapshot?.model_revision, 2);
  assert.equal(store.getState().request.recovery, null);
  assert.deepEqual(requests, [
    { path: "/op", payload },
    { path: "/op", payload }
  ]);
});

test("dismiss and missing retries are deterministic", async () => {
  const store = createEditorStore(createInitialEditorState(snapshot(1)));
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw new Error("offline"); },
      getState: async () => { throw new Error("offline"); }
    }
  });

  const missingMutation = await actions.retryMutation();
  assert.equal(missingMutation.ok, false);
  if (missingMutation.ok) assert.fail("missing retry unexpectedly succeeded");
  assert.equal(missingMutation.skipped, true);
  assert.match(missingMutation.error.message, /No failed/);
  assert.equal(await actions.retryEvidence("report"), false);

  await actions.executeStateMutation({ name: "shift", path: "/op", payload: {} });
  assert.equal(store.getState().request.mutation.status, "error");
  actions.dismissRecovery();
  assert.equal(store.getState().request.recovery, null);
  assert.deepEqual(store.getState().request.mutation, {
    status: "idle", operation: null, error: null
  });
});

test("action module exposes only the controller factory and exact methods", () => {
  const store = createEditorStore(createInitialEditorState(snapshot(0)));
  const actions = createEditorActions({
    store,
    client: { postJSON: async () => snapshot(0), getState: async () => snapshot(0) }
  });

  assert.deepEqual(Object.keys(actionsModule), ["createEditorActions"]);
  assert.deepEqual(Object.keys(actions).sort(), [
    "dismissRecovery",
    "executeStateMutation",
    "initialize",
    "patchView",
    "refreshEvidence",
    "retryEvidence",
    "retryMutation"
  ]);
});
