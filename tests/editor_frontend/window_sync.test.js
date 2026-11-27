// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";

import { REPORT_NAMES } from "../../src/superglm/editor/app/api/contracts.js";
import { CONTROL_HELP, HELP_SECTIONS } from "../../src/superglm/editor/app/views/help_content.js";
import {
  assessWindowUrl,
  coalesce,
  createSessionPeer,
  createWindowSync,
  readAssessWindow,
  sessionKey,
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

test("the page's head script marks an Assess window for exactly the views readAssessWindow takes", () => {
  const page = readFileSync(new URL("../../src/superglm/editor/app/index.html", import.meta.url), "utf8");
  const views = JSON.parse(page.match(/const ASSESS_WINDOW_VIEWS = (\[[^\]]*\]);/)[1]);
  assert.deepEqual(views, ["assess", ...REPORT_NAMES]);
  for (const view of views) assert.equal(readAssessWindow(`?assess=${view}`), view);
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
  const message = { type: "superglm.changed", key: "k", job: { id: "j", phase: "started" } };
  a.announce(message);
  assert.deepEqual(heard, [message]);
  assert.equal(posted.length, 1);
  // Anything else on the channel is not a message of ours.
  const stranger = new Channel("n");
  stranger.postMessage({ type: "other", key: "k" });
  stranger.postMessage({ type: "superglm.changed", key: "k", job: true });
  stranger.postMessage({ type: "superglm.changed", key: "k", job: { id: "j", phase: "later" } });
  assert.equal(heard.length, 1);
  a.close();
  a.announce({ type: "superglm.changed", key: "k2" });
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

// Channels that deliver later, as BroadcastChannel does; `flush` delivers everything sent.
function laterChannels() {
  const open = new Set();
  const queue = [];
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
        if (other !== this && other.name === this.name) queue.push(() => other.onmessage?.({ data }));
      }
    }

    close() {
      open.delete(this);
    }
  }
  const flush = async () => {
    while (queue.length) {
      await new Promise((resolve) => { setImmediate(resolve); });
      queue.shift()();
    }
  };
  return { Channel, posted, flush };
}

// The session both windows read: a re-read takes its key, and a window whose key
// moved tells the others, as main.js's store subscription does. A window's own
// polling hands a job's start or end to `sees`, as main.js's Cross-validation hooks do.
function sessionWindow(Channel, server) {
  const win = { key: server.key(), reads: 0, lapses: 0, settles: 0 };
  win.peer = createSessionPeer({
    name: "editor",
    Channel,
    key: () => win.key,
    onJobNews: () => { win.lapses += 1; },
    reread: () => {
      win.reads += 1;
      const before = win.key;
      win.key = server.key();
      if (win.key !== before) win.peer.tell();
    },
  });
  win.sees = (id, phase) => {
    const notice = { id, phase };
    if (!win.peer.isNews(notice)) return;
    if (phase === "settled") win.settles += 1;
    win.peer.tell(notice);
  };
  win.count = () => {
    const counts = [win.reads, win.lapses, win.settles];
    win.reads = 0;
    win.lapses = 0;
    win.settles = 0;
    return counts;
  };
  return win;
}

test("one job costs each window at most one re-read when it starts and one when it ends", async () => {
  const { Channel, flush } = laterChannels();
  const server = { running: [], key() { return `3|${this.running.join()}`; } };
  const main = sessionWindow(Channel, server);
  const popup = sessionWindow(Channel, server);

  // Run CV in the main window; the popup hears of it, and both then poll it.
  server.running = ["cv-1"];
  main.sees("cv-1", "started");
  await flush();
  popup.sees("cv-1", "started");
  main.sees("cv-1", "started");
  await flush();
  const started = [main.count(), popup.count()];
  assert.deepEqual(started[1], [1, 1, 0]);
  assert.ok(started[0][0] <= 1, `main read ${started[0][0]} times as the job started`);

  // Both pollers see it end: the first handles it and tells; the other has handled it then.
  server.running = [];
  main.sees("cv-1", "settled");
  await flush();
  popup.sees("cv-1", "settled");
  main.sees("cv-1", "settled");
  await flush();
  const settled = [main.count(), popup.count()];
  assert.deepEqual(settled[1], [1, 1, 0]);
  assert.equal(settled[0][2], 1);
  assert.ok(settled[0][0] <= 1, `main read ${settled[0][0]} times as the job ended`);
  assert.equal(main.key, popup.key);
});

test("a page's hello makes every other window lapse its job reports and re-read once", async () => {
  const { Channel, posted, flush } = laterChannels();
  const server = { key: () => "3|" };
  const open = sessionWindow(Channel, server);
  const loaded = sessionWindow(Channel, server);
  loaded.peer.hello();
  await flush();
  assert.deepEqual(open.count(), [1, 1, 0]);
  assert.deepEqual(loaded.count(), [0, 0, 0]);
  assert.equal(posted.filter((message) => message.hello).length, 1);
  assert.equal(posted.length, 1);
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
