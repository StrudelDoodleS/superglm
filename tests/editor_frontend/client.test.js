import assert from "node:assert/strict";
import test from "node:test";

import * as facade from "../../src/superglm/editor/app/api.js";
import * as clientModule from "../../src/superglm/editor/app/api/client.js";

const {
  EditorAPIError,
  createEditorClient,
  editorClient
} = clientModule;

test("client attaches the widget token and preserves request headers", async () => {
  /** @type {{url:string, options:RequestInit}} */
  const request = { url: "", options: {} };
  const client = createEditorClient({
    token: "secret",
    fetchImpl: async (url, options) => {
      request.url = String(url);
      request.options = options ?? {};
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    }
  });

  assert.deepEqual(await client.postJSON("/op", { operation: "reset" }), { ok: true });
  const headers = new Headers(request.options.headers);
  assert.equal(request.url, "/op");
  assert.equal(request.options.method, "POST");
  assert.equal(headers.get("X-SuperGLM-Editor-Token"), "secret");
  assert.equal(headers.get("Content-Type"), "application/json");
  assert.equal(request.options.body, JSON.stringify({ operation: "reset" }));

  await client.requestJSON("/custom", { headers: { "X-Trace": "kept" } });
  assert.equal(new Headers(request.options.headers).get("X-Trace"), "kept");
});

test("client getState requests the authoritative state route", async () => {
  /** @type {string[]} */
  const urls = [];
  const client = createEditorClient({
    fetchImpl: async (url) => {
      urls.push(String(url));
      return new Response(JSON.stringify({ model_revision: 3 }), { status: 200 });
    }
  });

  assert.deepEqual(await client.getState(), { model_revision: 3 });
  assert.deepEqual(urls, ["/state"]);
});

test("client preserves status and parsed payload on API errors", async () => {
  const payload = { error: "bad edit", field: "age" };
  const client = createEditorClient({
    fetchImpl: async () => new Response(JSON.stringify(payload), { status: 400 })
  });

  await assert.rejects(
    () => client.getState(),
    (error) => {
      assert.ok(error instanceof EditorAPIError);
      assert.equal(error.name, "EditorAPIError");
      assert.equal(error.status, 400);
      assert.equal(error.message, "bad edit");
      assert.deepEqual(error.payload, payload);
      return true;
    }
  );
});

test("invalid error JSON still becomes a structured API error", async () => {
  const client = createEditorClient({
    fetchImpl: async () => new Response("not json", {
      status: 503,
      statusText: "Unavailable"
    })
  });

  await assert.rejects(
    () => client.requestJSON("/broken"),
    (error) => error instanceof EditorAPIError
      && error.status === 503
      && error.payload === null
      && error.message === "Unavailable"
  );
});

test("requestBlob returns responses and preserves JSON error details", async () => {
  const success = new Response(new Blob(["model"]), {
    status: 200,
    headers: { "Content-Disposition": "attachment; filename=model.joblib" }
  });
  let response = success;
  const client = createEditorClient({ fetchImpl: async () => response });

  assert.strictEqual(await client.requestBlob("/download"), success);

  response = new Response(JSON.stringify({ error: "cannot save", detail: "disk" }), {
    status: 500,
    statusText: "Server Error"
  });
  await assert.rejects(
    () => client.requestBlob("/download"),
    (error) => {
      if (!(error instanceof EditorAPIError)) return false;
      const payload = error.payload;
      return error.status === 500
        && error.message === "cannot save"
        && payload !== null
        && typeof payload === "object"
        && "detail" in payload
        && payload.detail === "disk";
    }
  );
});

test("client and compatibility facade expose only the supported surface", () => {
  assert.ok(editorClient);
  assert.deepEqual(Object.keys(clientModule).sort(), [
    "EditorAPIError",
    "createEditorClient",
    "editorClient",
    "postJSON",
    "requestBlob",
    "requestJSON"
  ]);
  assert.deepEqual(Object.keys(facade).sort(), [
    "EditorAPIError",
    "editorClient",
    "postJSON",
    "requestBlob",
    "requestJSON"
  ]);
});

test("client job calls post to the job routes", async () => {
  /** @type {Array<{url:string, body:unknown}>} */
  const requests = [];
  const client = createEditorClient({
    fetchImpl: async (url, options) => {
      requests.push({ url: String(url), body: JSON.parse(String(options?.body)) });
      return new Response(JSON.stringify({ job_id: "cv-1", status: "running" }), { status: 200 });
    }
  });

  await client.jobStart("cv");
  await client.jobStatus("cv-1", true);
  await client.jobCancel("cv-1");

  assert.deepEqual(requests, [
    { url: "/job_start", body: { kind: "cv" } },
    { url: "/job_status", body: { job_id: "cv-1", wait: true } },
    { url: "/job_cancel", body: { job_id: "cv-1" } }
  ]);
});

const KNOWN_STATE = "X-SuperGLM-Editor-Known-State";

/** @param {unknown[]} replies answered in order @returns the client and what it sent */
function replyingClient(replies) {
  /** @type {Array<{path:string, known:string|null}>} */
  const sent = [];
  const client = createEditorClient({
    fetchImpl: async (url, options) => {
      sent.push({ path: String(url), known: new Headers(options?.headers).get(KNOWN_STATE) });
      return new Response(JSON.stringify(replies.shift()), { status: 200 });
    }
  });
  return { client, sent };
}

/** @param {number} generation @param {Record<string, unknown>} terms @param {Record<string, string>} versions */
function cutState(generation, terms, versions) {
  return { state_generation: generation, selected_term: "a", terms, term_versions: versions };
}

test("a state reply that leaves a term out takes it from the client's copy, the same object", async () => {
  const { client, sent } = replyingClient([
    cutState(1, { a: { y: [1] }, b: { y: [2] } }, { a: "a1", b: "b1" }),
    cutState(2, { b: { y: [3] } }, { a: "a1", b: "b2" })
  ]);

  const first = /** @type {any} */ (await client.getState());
  const second = /** @type {any} */ (await client.getState());

  assert.deepEqual(sent.map(({ known }) => known), ["none", "1"]);
  assert.strictEqual(second.terms.a, first.terms.a);
  assert.deepEqual(second.terms.b, { y: [3] });
  assert.deepEqual(Object.keys(second.terms), ["a", "b"]);
});

test("a state inside a reply merges the same way and the rest of the reply is kept", async () => {
  const { client } = replyingClient([
    cutState(1, { a: { y: [1] }, b: { y: [2] } }, { a: "a1", b: "b1" }),
    { ok: true, summary: "kept", state: cutState(2, {}, { a: "a1", b: "b1" }) }
  ]);

  const first = /** @type {any} */ (await client.getState());
  const reply = /** @type {any} */ (await client.postJSON("/note", {}));

  assert.equal(reply.summary, "kept");
  assert.strictEqual(reply.state.terms.b, first.terms.b);
});

test("a term the copy holds at another version asks for every term", async () => {
  const { client, sent } = replyingClient([
    cutState(1, { a: { y: [1] }, b: { y: [2] } }, { a: "a1", b: "b1" }),
    cutState(3, {}, { a: "a2", b: "b1" }),
    cutState(4, { a: { y: [9] }, b: { y: [2] } }, { a: "a2", b: "b1" })
  ]);

  await client.getState();
  const healed = /** @type {any} */ (await client.postJSON("/op", {}));

  assert.deepEqual(sent, [
    { path: "/state", known: "none" },
    { path: "/op", known: "1" },
    { path: "/state", known: "none" }
  ]);
  assert.equal(healed.state_generation, 4);
  assert.deepEqual(healed.terms.a, { y: [9] });
});

test("an older reply arriving late does not move the copy back", async () => {
  const { client, sent } = replyingClient([
    cutState(5, { a: { y: [5] } }, { a: "a5" }),
    cutState(4, { a: { y: [4] } }, { a: "a4" }),
    cutState(6, {}, { a: "a5" })
  ]);

  await client.getState();
  const late = /** @type {any} */ (await client.getState());
  const next = /** @type {any} */ (await client.getState());

  assert.deepEqual(late.terms.a, { y: [4] });
  assert.deepEqual(sent.map(({ known }) => known), ["none", "5", "5"]);
  assert.deepEqual(next.terms.a, { y: [5] });
});

test("a reply that names no versions passes through unchanged", async () => {
  const plain = { state_generation: 1, terms: { a: { y: [1] } } };
  const { client } = replyingClient([plain, { ok: true }]);

  assert.deepEqual(await client.getState(), plain);
  assert.deepEqual(await client.postJSON("/select", {}), { ok: true });
});
