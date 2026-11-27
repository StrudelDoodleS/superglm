// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { CONTROL_HELP, HELP_SECTIONS } from "../../src/superglm/editor/app/views/help_content.js";
import {
  assessWindowUrl,
  coalesce,
  createWindowSync,
  readAssessWindow,
  sessionKey,
  startedJobs,
  syncChannelName,
} from "../../src/superglm/editor/app/views/window_sync.js";

const SNAPSHOT = Object.freeze({
  model_revision: 0,
  in_force: "m0",
  pending: [],
  jobs_running: [],
  final_fit: { available: false, stale: false },
  selection: {},
  state_generation: 1,
});

// Channels of one name hear each other and never themselves, as BroadcastChannel does.
function fakeChannelPair() {
  const open = new Set();
  const posted = [];
  class Channel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      open.add(this);
    }

    postMessage(data) {
      posted.push(data);
      for (const other of open) {
        if (other !== this && other.name === this.name) other.onmessage?.({ data });
      }
    }

    close() {
      open.delete(this);
    }
  }
  return { Channel, posted };
}

test("an Assess window keeps the page's address and token and opens on its view", () => {
  assert.equal(assessWindowUrl("http://127.0.0.1:8765/?token=abc&test=1", "cv"), "/?token=abc&test=1&assess=cv");
  assert.equal(assessWindowUrl("http://127.0.0.1:8765/?token=abc&assess=cv", "final"), "/?token=abc&assess=final");
  assert.equal(readAssessWindow("?token=abc&assess=cv"), "cv");
  assert.equal(readAssessWindow("?assess=assess"), "assess");
  assert.equal(readAssessWindow("?assess=editor"), null);
  assert.equal(readAssessWindow("?token=abc"), null);
});

test("windows of one editor share a channel, and another editor's windows do not hear it", () => {
  assert.equal(syncChannelName("http://127.0.0.1:8765/?token=abc&test=1"), "superglm-editor:8765:abc");
  assert.notEqual(syncChannelName("http://127.0.0.1:8765/?test=1"), syncChannelName("http://127.0.0.1:8766/?test=1"));
});

test("the session key moves with the model, the waiting list and jobs, not a selection", () => {
  const base = sessionKey(SNAPSHOT);
  assert.equal(sessionKey({ ...SNAPSHOT, selection: { age: [1, 2] }, state_generation: 9 }), base);
  for (const changed of [{ model_revision: 1 }, { in_force: "x" }, { pending: [{ id: "p1" }] },
    { jobs_running: [{ job_id: "j" }] }, { final_fit: { available: true, stale: false } },
    { final_fit: { available: false, stale: true } }]) {
    assert.notEqual(sessionKey({ ...SNAPSHOT, ...changed }), base);
  }
  assert.notEqual(sessionKey(null), base);
});

test("a window tells the others and hears them", () => {
  const { Channel, posted } = fakeChannelPair();
  const heard = [];
  const a = createWindowSync({ name: "n", onMessage: () => {}, Channel });
  createWindowSync({ name: "n", onMessage: (m) => heard.push(m), Channel });
  createWindowSync({ name: "other", onMessage: (m) => heard.push(m), Channel });
  a.announce({ type: "superglm.changed", key: "k", job: true });
  assert.deepEqual(heard, [{ type: "superglm.changed", key: "k", job: true }]);
  assert.equal(posted.length, 1);
  // Anything else on the channel is not a message of ours.
  new Channel("n").postMessage({ type: "other", key: "k" });
  assert.equal(heard.length, 1);
  a.close();
  a.announce({ type: "superglm.changed", key: "k2", job: false });
  assert.equal(heard.length, 1);
  assert.doesNotThrow(() => createWindowSync({ name: "n", onMessage: () => {}, Channel: undefined }).announce(heard[0]));
});

test("a re-read asked for while one runs is run once more after it, not once per ask", async () => {
  let runs = 0;
  let release;
  const reread = coalesce(async () => {
    runs += 1;
    if (runs === 1) await new Promise((resolve) => { release = resolve; });
  });
  const first = reread();
  await reread();
  await reread();
  assert.equal(runs, 1);
  release();
  await first;
  assert.equal(runs, 2);
  await reread();
  assert.equal(runs, 3);
});

test("a job is news once: when it starts, not on each poll, nor when the session listed it", () => {
  const known = new Set(["listed"]);
  const running = (job_id) => ({ job_id, status: "running" });
  assert.deepEqual(startedJobs({ cv: running("c1"), final_fit: null }, known), ["c1"]);
  assert.deepEqual(startedJobs({ cv: running("c1"), final_fit: running("listed") }, known), []);
  assert.deepEqual(startedJobs({ cv: { job_id: "c2", status: "done" }, final_fit: null }, known), []);
});

test("Help names the second window and says why a notebook cell does not offer it", () => {
  assert.deepEqual(CONTROL_HELP.new_window, {
    title: "Open in new window",
    body: "Show this view in a second browser window, for another monitor. It follows every change made here.",
  });
  const assess = HELP_SECTIONS.find((section) => section.title === "Assess");
  assert.ok(assess.items.includes(
    "Open in new window: Show this view in a second browser window, for another monitor. It follows every change made here.",
  ));
  assert.ok(assess.items.includes(
    "Inside a notebook cell it is not offered: the page reaches Python through the notebook, which a second window cannot.",
  ));
});
