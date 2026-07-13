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
