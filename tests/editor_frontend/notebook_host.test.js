import assert from "node:assert/strict";
import test from "node:test";

import { createEditorClient } from "../../src/superglm/editor/app/api/client.js";
import {
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
  assert.deepEqual(model.sent, [
    {
      type: REQUEST,
      id: 0,
      method: "POST",
      url: "/op",
      headers: [["content-type", "application/json"]],
      body: '{"operation":"reset"}'
    },
    { type: REQUEST, id: 1, method: "GET", url: "/download_export?format=xlsx", headers: [], body: null }
  ]);
});

test("a reply in parts resolves once every part has arrived, in part order", async () => {
  const model = recordingModel();
  const transport = createMessageFetch(model, realm);
  const response = transport.fetch("/state");
  const reply = { type: RESPONSE, id: 0, status: 200, headers: [["x-superglm-validation", "train"]], parts: 3 };
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
    { type: RESPONSE, id: 0, status: 400, headers: [["content-type", "application/json"]], part: 0, parts: 1 },
    [bytes('{"error":"Unknown editor operation"}')]
  );
  await assert.rejects(failed, { name: "EditorAPIError", status: 400, message: "Unknown editor operation" });
});

test("a no-content reply has no body, and other messages are ignored", async () => {
  const transport = createMessageFetch(recordingModel(), realm);
  const response = transport.fetch("/favicon.ico");
  transport.receive({ type: "something else", id: 0 });
  transport.receive({ type: RESPONSE, id: 99, status: 200, headers: [], part: 0, parts: 1 }, [bytes("x")]);
  transport.receive({ type: RESPONSE, id: 0, status: 204, headers: [], part: 0, parts: 1 }, [bytes("")]);
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
