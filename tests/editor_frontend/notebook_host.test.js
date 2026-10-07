import assert from "node:assert/strict";
import test from "node:test";

import { createEditorClient } from "../../src/superglm/editor/app/api/client.js";
import {
  MAX_REQUEST_BYTES,
  REQUEST,
  RESPONSE,
  createMessageFetch,
  linkModules
} from "../../src/superglm/editor/app/api/notebook_host.js";

const realm = { Response, Headers, Blob };

function recordingModel() {
  /** @type {any[]} */
  const sent = [];
  return { sent, send: (/** @type {unknown} */ content) => sent.push(content) };
}

/** @param {string} text */
function bytes(text) {
  return new DataView(new TextEncoder().encode(text).buffer);
}

test("a request goes to Python as one message with its id, method, url, headers and body", () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  transport.fetch("/op", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: '{"operation":"reset"}'
  });
  transport.fetch(new URL("http://kernel/download_export?format=xlsx"));
  const [first, second] = model.sent.map((message) => message.id);
  assert.match(first, /^[0-9a-z]+-0$/);
  assert.equal(second, first.replace(/-0$/, "-1"));
  assert.deepEqual(model.sent, [
    {
      type: REQUEST,
      id: first,
      method: "POST",
      url: "/op",
      headers: [["content-type", "application/json"]],
      body: '{"operation":"reset"}'
    },
    { type: REQUEST, id: second, method: "GET", url: "/download_export?format=xlsx", headers: [], body: null }
  ]);
});

test("views of one widget never take each other's replies", async () => {
  // Python's replies reach every view; a rebuilt page's new transport also
  // overlaps the old one's requests still in flight.
  const model = recordingModel();
  const state = createMessageFetch(model, realm);
  const download = createMessageFetch(model, realm);
  const stateResponse = state.fetch("/state");
  const downloadResponse = download.fetch("/download_export?format=joblib");
  const [stateId, downloadId] = model.sent.map((message) => message.id);
  assert.notEqual(stateId, downloadId);
  const reply = { type: RESPONSE, id: stateId, status: 200, headers: [], part: 0, parts: 1 };
  for (const view of [state, download]) view.receive(reply, [bytes('{"terms":{}}')]);
  assert.equal(await (await stateResponse).text(), '{"terms":{}}');
  assert.equal(download.pending.size, 1);
  for (const view of [state, download]) {
    view.receive({ ...reply, id: downloadId }, [bytes("model bytes")]);
  }
  assert.equal(await (await downloadResponse).text(), "model bytes");
});

test("a reply in parts resolves once every part has arrived, in part order", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  const response = transport.fetch("/state");
  const reply = { type: RESPONSE, id: model.sent[0].id, status: 200, headers: [["x-superglm-validation", "train"]], parts: 3 };
  transport.receive({ ...reply, part: 2 }, [bytes("c")]);
  transport.receive({ ...reply, part: 0 }, [bytes("a")]);
  assert.equal(transport.pending.size, 1);
  transport.receive({ ...reply, part: 1 }, [bytes("b")]);
  const resolved = await response;
  assert.equal(resolved.status, 200);
  assert.equal(resolved.headers.get("x-superglm-validation"), "train");
  assert.equal(await resolved.text(), "abc");
  assert.equal(transport.pending.size, 0);
});

test("an error status resolves as a Response the client reads as an error", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  const client = createEditorClient({ fetchImpl: transport.fetch });
  const failed = client.postJSON("/op", { operation: "nope" });
  transport.receive(
    { type: RESPONSE, id: model.sent[0].id, status: 400, headers: [["content-type", "application/json"]], part: 0, parts: 1 },
    [bytes('{"error":"Unknown editor operation"}')]
  );
  await assert.rejects(failed, { name: "EditorAPIError", status: 400, message: "Unknown editor operation" });
});

test("a no-content reply has no body, and other messages are ignored", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  const response = transport.fetch("/favicon.ico");
  const id = model.sent[0].id;
  transport.receive({ type: "something else", id });
  transport.receive({ type: RESPONSE, id: `${id}9`, status: 200, headers: [], part: 0, parts: 1 }, [bytes("x")]);
  transport.receive({ type: RESPONSE, id, status: 204, headers: [], part: 0, parts: 1 }, [bytes("")]);
  const resolved = await response;
  assert.equal(resolved.status, 204);
  assert.equal(resolved.body, null);
});

test("a body that is not text is refused before anything is sent", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  await assert.rejects(transport.fetch("/op", { method: "POST", body: new Blob(["x"]) }), /text request bodies/);
  assert.deepEqual(model.sent, []);
});

test("a transport knows its own request ids and no other's", () => {
  const model = recordingModel();
  const mine = createMessageFetch(model, realm);
  const other = createMessageFetch(model, realm);
  mine.fetch("/op", { method: "POST", body: "{}" });
  const id = model.sent[0].id;
  assert.equal(mine.owns(id), true);
  assert.equal(other.owns(id), false);
  assert.equal(mine.owns(0), false);
});

test("a request whose message would pass the widget-message limit is refused as Python refuses", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  const bytes = (/** @type {unknown} */ message) =>
    new TextEncoder().encode(JSON.stringify(message)).length;
  // The message's own envelope, measured on an empty body; ids -0 to -9 are one length.
  transport.fetch("/note", { method: "POST", body: "" });
  const envelope = bytes(model.sent[0]);
  transport.fetch("/note", { method: "POST", body: "a".repeat(MAX_REQUEST_BYTES - envelope) });
  assert.equal(bytes(model.sent[1]), MAX_REQUEST_BYTES);

  const client = createEditorClient({ fetchImpl: transport.fetch });
  const over = client.requestJSON("/note", {
    method: "POST",
    body: "a".repeat(MAX_REQUEST_BYTES - envelope + 1)
  });
  await assert.rejects(over, {
    name: "EditorAPIError",
    status: 413,
    message: /too large to send from a notebook cell: its request is 4\.0 MB/
  });
  // A body escaped again inside the message counts at its size as sent:
  // a backslash measured as one byte travels as two.
  const escaped = transport.fetch("/note", {
    method: "POST",
    body: "\\".repeat(MAX_REQUEST_BYTES / 2 + 1)
  });
  assert.equal((await escaped).status, 413);
  assert.equal(model.sent.length, 2);
});

test("closing fails every request still waiting", async () => {
  const transport = createMessageFetch(recordingModel(), realm);
  const waiting = transport.fetch("/job_status");
  transport.close();
  await assert.rejects(waiting, /closed/);
  assert.equal(transport.pending.size, 0);
});

test("modules link to the URLs of the modules before them", () => {
  /** @type {string[]} */
  const made = [];
  const urls = linkModules(
    [
      { path: "format.js", source: "export const f = 1;" },
      { path: "main.js", source: `import { f } from "superglm-module:format.js"; import 'superglm-module:format.js';` }
    ],
    (source) => {
      made.push(source);
      return `blob:${made.length}`;
    }
  );
  assert.deepEqual(urls, { "format.js": "blob:1", "main.js": "blob:2" });
  assert.equal(made[1], `import { f } from "blob:1"; import 'blob:1';`);
  assert.throws(
    () => linkModules([{ path: "main.js", source: `import "superglm-module:later.js";` }], () => "blob:x"),
    /imports later\.js before it is loaded/
  );
});

test("the client sends through the notebook host's fetch when the page has one", async () => {
  /** @type {string[]} */
  const urls = [];
  const host = {
    kind: "notebook",
    fetch: async (/** @type {RequestInfo|URL} */ url) => {
      urls.push(String(url));
      return new Response("{}", { status: 200 });
    }
  };
  Object.assign(globalThis, { superglmEditorHost: host });
  try {
    await createEditorClient().getState();
  } finally {
    delete (/** @type {any} */ (globalThis)).superglmEditorHost;
  }
  assert.deepEqual(urls, ["/state"]);
});
