import assert from "node:assert/strict";
import test from "node:test";

import { createEmptyEvidenceState } from "../../src/superglm/editor/app/api/contracts.js";
import { MUTATION, isHeld, moveMutation } from "../../src/superglm/editor/app/state/mutation.js";

/** @typedef {import('../../src/superglm/editor/app/api/contracts.js').EditorRequestState} EditorRequestState */
/** @typedef {import('../../src/superglm/editor/app/api/contracts.js').MutationRequestState} MutationRequestState */
/** @typedef {import('../../src/superglm/editor/app/state/mutation.js').MutationEvent} MutationEvent */

const retry = { name: "drag", path: "/drag", payload: { term: "age" } };

/** @type {Record<MutationRequestState['status'], MutationRequestState>} */
const MUTATIONS = {
  idle: { status: "idle", operation: null, error: null },
  running: { status: "running", operation: "refit", error: null, blocking: true },
  error: { status: "error", operation: "drag", error: "network down" }
};

/** @type {Record<MutationEvent['type'], MutationEvent>} */
const EVENTS = {
  hold: { type: "hold", operation: "collapse levels", blocking: false },
  pass: { type: "pass" },
  settle: { type: "settle" },
  fail: { type: "fail", operation: "drag", error: "network down", message: "network down", retry },
  dismiss: { type: "dismiss" },
  notice: { type: "notice", message: "Nothing to undo." }
};

/**
 * A request state as the editor can hold it: a failed change always shows its alert.
 * @param {MutationRequestState['status']} status
 * @param {EditorRequestState['recovery']} [recovery]
 * @returns {EditorRequestState}
 */
function request(status, recovery = status === "error" ? { message: "network down", retry } : null) {
  return {
    mutation: MUTATIONS[status],
    evidence: {
      metrics: createEmptyEvidenceState(),
      summary: createEmptyEvidenceState(),
      report: createEmptyEvidenceState()
    },
    recovery,
    nextSequence: 1
  };
}

test("the table names every state and event of the editor's hold", () => {
  assert.deepEqual([...MUTATION.states].sort(), Object.keys(MUTATIONS).sort());
  assert.deepEqual([...MUTATION.events].sort(), Object.keys(EVENTS).sort());
  assert.equal(MUTATION.initial, "idle");
});

test("every move lands where the table says, and a move it does not list changes nothing", () => {
  for (const status of MUTATION.states) {
    for (const type of MUTATION.events) {
      const before = request(status, { message: "an earlier alert", retry: null });
      const after = moveMutation(before, EVENTS[type]);
      const target = MUTATION.next(status, type);
      if (target === null) {
        assert.strictEqual(after, before, `${status} takes no ${type}`);
      } else {
        assert.equal(after.mutation.status, target, `${status} --${type}--> ${target}`);
        assert.strictEqual(after.evidence, before.evidence);
      }
    }
  }
});

test("each state carries only what belongs to it", () => {
  for (const status of MUTATION.states) {
    for (const type of MUTATION.events) {
      const { mutation, recovery } = moveMutation(request(status), EVENTS[type]);
      if (mutation.status === "idle") {
        assert.deepEqual(mutation, { status: "idle", operation: null, error: null });
        assert.equal(recovery?.retry ?? null, null, "only a failed change offers Retry");
      } else if (mutation.status === "running") {
        assert.equal(typeof mutation.operation, "string");
        assert.equal(mutation.error, null);
      } else {
        assert.equal(typeof mutation.operation, "string");
        assert.equal(typeof mutation.error, "string");
        assert.notEqual(recovery, null, "a failed change shows its alert");
      }
    }
  }
});

test("a running change takes no second change and ends only by its own answer", () => {
  const running = request("running");

  assert.equal(isHeld(running), true);
  assert.equal(isHeld(request("idle")), false);
  assert.equal(isHeld(request("error")), false);
  assert.strictEqual(moveMutation(running, EVENTS.hold), running);
  assert.strictEqual(moveMutation(running, EVENTS.pass), running);
  assert.equal(moveMutation(running, EVENTS.dismiss).mutation.status, "running");
  assert.equal(moveMutation(running, EVENTS.notice).mutation.status, "running");
  assert.equal(moveMutation(running, EVENTS.settle).mutation.status, "idle");
  assert.equal(moveMutation(running, EVENTS.fail).mutation.status, "error");
});

test("a hold names its overlay, and carries a detail only when given one", () => {
  const plain = moveMutation(request("idle"), EVENTS.hold);
  const detailed = moveMutation(request("error", { message: "network down", retry }), {
    type: "hold", operation: "Fitting free levels", blocking: true, detail: "age" });

  assert.deepEqual(plain.mutation, {
    status: "running", operation: "collapse levels", error: null, blocking: false
  });
  assert.deepEqual(detailed.mutation, {
    status: "running", operation: "Fitting free levels", error: null, blocking: true, detail: "age"
  });
  assert.equal(detailed.recovery, null);
});

test("a failure keeps its alert until the next change or Dismiss", () => {
  const failed = moveMutation(request("running"), EVENTS.fail);

  assert.deepEqual(failed.mutation, { status: "error", operation: "drag", error: "network down" });
  assert.deepEqual(failed.recovery, { message: "network down", retry });
  // A notice replaces the alert's sentence and its Retry, not the failure.
  const noticed = moveMutation(failed, EVENTS.notice);
  assert.equal(noticed.mutation.status, "error");
  assert.deepEqual(noticed.recovery, { message: "Nothing to undo.", retry: null });
  const dismissed = moveMutation(noticed, EVENTS.dismiss);
  assert.deepEqual(dismissed.mutation, { status: "idle", operation: null, error: null });
  assert.equal(dismissed.recovery, null);
  assert.equal(moveMutation(failed, EVENTS.pass).recovery, null);
});

test("a settled change may leave a sentence in the alert, with no Retry", () => {
  const settled = moveMutation(request("running"), { type: "settle", notice: "Refresh incomplete." });

  assert.deepEqual(settled.mutation, { status: "idle", operation: null, error: null });
  assert.deepEqual(settled.recovery, { message: "Refresh incomplete.", retry: null });
});

test("Dismiss with nothing to dismiss changes nothing", () => {
  const idle = request("idle");
  const running = request("running");

  assert.strictEqual(moveMutation(idle, EVENTS.dismiss), idle);
  assert.strictEqual(moveMutation(running, EVENTS.dismiss), running);
  const noticed = request("idle", { message: "Nothing to undo.", retry: null });
  const dismissed = moveMutation(noticed, EVENTS.dismiss);
  assert.equal(dismissed.recovery, null);
  assert.strictEqual(dismissed.mutation, noticed.mutation);
});
