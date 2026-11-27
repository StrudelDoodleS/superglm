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

// A run per call of `next()`: each run waits until the test finishes it, and
// resolves to its number, or rejects when the test says so.
function gatedRuns() {
  const runs = [];
  const run = () => new Promise((resolve, reject) => {
    runs.push({ number: runs.length + 1, resolve, reject });
  });
  const tick = () => new Promise((resolve) => { setImmediate(resolve); });
  const finish = async (number, error = null) => {
    await tick();
    const entry = runs[number - 1];
    if (error) entry.reject(error);
    else entry.resolve(entry.number);
    await tick();
  };
  return { runs, run, finish, tick };
}

test("asks made while a run goes share one trailing run", async () => {
  const { runs, run, finish, tick } = gatedRuns();
  const reread = coalesce(run);
  const first = reread();
  await tick();
  const during = [reread(), reread(), reread(), reread(), reread()];
  assert.ok(during.every((promise) => promise === during[0]));
  assert.equal(runs.length, 1);
  await finish(1);
  assert.equal(runs.length, 2);
  await finish(2);
  assert.deepEqual(await Promise.all([first, ...during]), [1, 2, 2, 2, 2, 2]);
  assert.equal(runs.length, 2);
});

test("an ask a run's completion makes cannot hold back an ask already waiting", async () => {
  const { runs, run, finish, tick } = gatedRuns();
  const reread = coalesce(run);
  // Each run's completion asks again, as a caller that re-reads on every answer would.
  let again = 5;
  const askOnCompletion = (promise) => promise.then(() => {
    if (again-- > 0) askOnCompletion(reread());
  });
  askOnCompletion(reread());
  await tick();
  const waiting = reread();
  let got = null;
  waiting.then((value) => { got = value; });
  await finish(1);
  await finish(2);
  assert.equal(got, 2);
  for (let number = 3; number <= runs.length; number += 1) await finish(number);
});

test("a run that fails fails only the asks it covers", async () => {
  const { run, finish, tick } = gatedRuns();
  const reread = coalesce(run);
  const first = assert.rejects(reread(), /first failed/);
  await tick();
  const second = reread();
  await finish(1, new Error("first failed"));
  await first;
  const third = assert.rejects(reread(), /third failed/);
  await finish(2);
  assert.equal(await second, 2);
  await finish(3, new Error("third failed"));
  await third;
  const fourth = reread();
  await finish(4);
  assert.equal(await fourth, 4);
});

test("an ask never gets the outcome of a run that started before it", async () => {
  const { runs, run, finish, tick } = gatedRuns();
  const reread = coalesce(run);
  const asked = [];
  const ask = () => {
    const started = runs.length;
    asked.push(reread().then((number) => [started, number]));
  };
  ask();
  await tick();
  ask();
  ask();
  await finish(1);
  ask();
  await finish(2);
  await finish(3);
  for (const [started, number] of await Promise.all(asked)) {
    assert.ok(number > started, `an ask made after ${started} runs got run ${number}`);
  }
});

// One editor's windows over a channel that delivers later, beside a Python session
// whose /state answers wait for `run`, which delivers every message before it lets
// the next answer through. Each window wires the production peer and coalescer as
// main.js does: a taken snapshot whose key moved is told to the others, as main.js's
// store subscription does, and `settle(kind)` stands in for actions.settleJob.
function editorWorld() {
  const open = new Set();
  const deliveries = [];
  const answers = [];
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
        if (other !== this && other.name === this.name) deliveries.push(() => other.onmessage?.({ data }));
      }
    }

    close() {
      open.delete(this);
    }
  }
  const python = {
    running: [],
    finalFit: false,
    failures: 0,
    key() { return `3|${this.running.join()}|${this.finalFit}`; },
  };

  function editorWindow() {
    const win = { key: "", reads: 0, lapses: 0, asks: [], shown: "cv" };
    // GET /state: whether it succeeded, once `run` lets its answer through.
    win.read = () => {
      win.reads += 1;
      return new Promise((resolve) => {
        answers.push(() => {
          if (python.failures > 0) {
            python.failures -= 1;
            resolve(false);
            return;
          }
          const before = win.key;
          win.key = python.key();
          if (before && before !== win.key) win.peer.tell();
          resolve(true);
        });
      });
    };
    win.peer = createSessionPeer({
      name: "editor",
      Channel,
      key: () => win.key,
      onJobNews: () => { win.lapses += 1; },
      reread: coalesce(win.read),
    });
    // A Final fit re-reads Python, and a taken re-read has lapsed every report and asked
    // for the open one; otherwise the job's reports lapse here and the open one is asked.
    win.settle = (kind) => async () => {
      if (kind === "final_fit" && (await win.read())) {
        win.asks.push(win.shown);
        return;
      }
      win.lapses += 1;
      win.asks.push(win.shown);
    };
    win.count = () => {
      const counts = { reads: win.reads, lapses: win.lapses, asks: win.asks };
      Object.assign(win, { reads: 0, lapses: 0, asks: [] });
      return counts;
    };
    return win;
  }

  // Deliver and answer until nothing moves; a storm of messages fails the test.
  async function run(limit = 200) {
    for (let step = 0; ; step += 1) {
      assert.ok(step < limit, "the windows never went quiet");
      await new Promise((resolve) => { setImmediate(resolve); });
      const next = deliveries.shift() ?? answers.shift();
      if (!next) return;
      next();
    }
  }

  // A page loads: its first snapshot, which it does not tell, then its hello.
  async function load(win, onFirstSnapshot = () => {}) {
    const loading = win.read().then(() => {
      onFirstSnapshot();
      win.peer.hello();
    });
    await run();
    await loading;
    await run();
  }

  return { python, posted, editorWindow, run, load };
}

test("a Final fit's end, its key told before its notice, costs each window at most two reads and then quiet", async () => {
  const world = editorWorld();
  const main = world.editorWindow();
  const popup = world.editorWindow();
  await world.load(main);
  await world.load(popup);
  const started = { id: "ff-1", phase: "started" };
  world.python.running = ["ff-1"];
  const startedHere = main.peer.sawJob(started);
  await world.run();
  await startedHere;
  await popup.peer.sawJob(started);
  await world.run();
  main.count();
  popup.count();

  // The main window's poll sees the end first; its settlement re-read moves its key,
  // which it tells before the notice, so the popup has a read held when the notice comes.
  const settled = { id: "ff-1", phase: "settled" };
  world.python.running = [];
  world.python.finalFit = true;
  const endedHere = main.peer.sawJob(settled, main.settle("final_fit"));
  await world.run();
  await endedHere;
  const endedThere = popup.peer.sawJob(settled, popup.settle("final_fit"));
  await world.run();
  await endedThere;
  await world.run();

  const [fromMain, fromPopup] = [main.count(), popup.count()];
  assert.ok(fromMain.reads <= 2 && fromPopup.reads <= 2, JSON.stringify([fromMain, fromPopup]));
  assert.deepEqual(fromMain.asks, ["cv"]);
  assert.equal(fromPopup.lapses, 1);
  assert.equal(main.key, popup.key);
  await world.run();
  assert.deepEqual([main.reads, popup.reads], [0, 0]);
});

test("a Final fit's end both windows poll at once costs each at most three reads, then the windows agree and go quiet", async () => {
  const world = editorWorld();
  const main = world.editorWindow();
  const popup = world.editorWindow();
  await world.load(main);
  await world.load(popup);
  const started = { id: "ff-1", phase: "started" };
  world.python.running = ["ff-1"];
  const startedHere = main.peer.sawJob(started);
  await world.run();
  await startedHere;
  await popup.peer.sawJob(started);
  await world.run();
  main.count();
  popup.count();

  // Both polls see the end first, so both start their settlement re-read. The main
  // window's is answered first: its moved key and its notice reach the popup while the
  // popup still holds the key from before the end.
  const settled = { id: "ff-1", phase: "settled" };
  world.python.running = [];
  world.python.finalFit = true;
  const ended = [
    main.peer.sawJob(settled, main.settle("final_fit")),
    popup.peer.sawJob(settled, popup.settle("final_fit")),
  ];
  await world.run();
  await Promise.all(ended);
  await world.run();

  const [fromMain, fromPopup] = [main.count(), popup.count()];
  assert.ok(fromMain.reads <= 3 && fromPopup.reads <= 3, JSON.stringify([fromMain, fromPopup]));
  assert.deepEqual([fromMain.asks, fromPopup.asks], [["cv"], ["cv"]]);
  assert.equal(main.key, popup.key);
  await world.run();
  assert.deepEqual([main.reads, popup.reads], [0, 0]);
});

test("a popup opened while the main window's job runs costs the main window at most two reads and then quiet", async () => {
  const world = editorWorld();
  const main = world.editorWindow();
  await world.load(main);
  const started = { id: "cv-1", phase: "started" };
  world.python.running = ["cv-1"];
  // Run CV started here: the job is known, but the main window's snapshot does not list it.
  await main.peer.sawJob(started);
  await world.run();
  main.count();

  const popup = world.editorWindow();
  // The popup's first snapshot lists the job, and watching it tells its start.
  await world.load(popup, () => { void popup.peer.sawJob(started); });
  await world.run();

  const [fromMain, fromPopup] = [main.count(), popup.count()];
  assert.ok(fromMain.reads <= 2, JSON.stringify(fromMain));
  assert.ok(fromPopup.reads <= 2, JSON.stringify(fromPopup));
  assert.equal(main.key, popup.key);
  await world.run();
  assert.deepEqual([main.reads, popup.reads], [0, 0]);
});

test("a window whose re-read for a peer's notice failed settles the job when its own poll sees the end", async () => {
  const world = editorWorld();
  const main = world.editorWindow();
  const popup = world.editorWindow();
  await world.load(main);
  await world.load(popup);
  world.python.running = ["cv-1"];
  await main.peer.sawJob({ id: "cv-1", phase: "started" });
  await world.run();
  main.count();
  popup.count();

  const settled = { id: "cv-1", phase: "settled" };
  world.python.running = [];
  world.python.failures = 1;
  const endedHere = main.peer.sawJob(settled, main.settle("cv"));
  await world.run();
  await endedHere;
  assert.deepEqual(popup.count(), { reads: 1, lapses: 1, asks: [] });

  await popup.peer.sawJob(settled, popup.settle("cv"));
  await world.run();
  assert.deepEqual(popup.count(), { reads: 0, lapses: 1, asks: ["cv"] });
});

test("a page's hello makes every other window lapse its job reports and re-read once", async () => {
  const world = editorWorld();
  const open = world.editorWindow();
  await world.load(open);
  open.count();
  const loaded = world.editorWindow();
  await world.load(loaded);
  assert.deepEqual(open.count(), { reads: 1, lapses: 1, asks: [] });
  assert.deepEqual(loaded.count(), { reads: 1, lapses: 0, asks: [] });  // its own first snapshot
  assert.equal(world.posted.filter((message) => message.hello).length, 2);
  assert.equal(world.posted.length, 2);
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
