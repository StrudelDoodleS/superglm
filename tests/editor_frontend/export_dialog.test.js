// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { bindExportDialog } from "../../src/superglm/editor/app/views/export_dialog.js";

class FakeElement {
  constructor(value = "") {
    this.value = value;
    this.checked = false;
    this.disabled = false;
    this.textContent = "";
    this.listeners = new Map();
  }

  addEventListener(name, listener) {
    const listeners = this.listeners.get(name) ?? new Set();
    listeners.add(listener);
    this.listeners.set(name, listeners);
  }

  removeEventListener(name, listener) {
    this.listeners.get(name)?.delete(listener);
  }

  listenerCount(name) {
    return this.listeners.get(name)?.size ?? 0;
  }

  async emit(name) {
    const event = { target: this, preventDefault() {} };
    await Promise.all(
      [...(this.listeners.get(name) ?? [])].map((listener) => listener(event)),
    );
  }
}

class FakeDialog extends FakeElement {
  constructor() {
    super();
    this.open = false;
    this.showCalls = 0;
    this.closeCalls = 0;
  }

  showModal() {
    this.open = true;
    this.showCalls += 1;
  }

  close() {
    this.open = false;
    this.closeCalls += 1;
  }
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function blobResponse({ disposition = "", validation = "" } = {}) {
  const headers = new Map();
  if (disposition) headers.set("content-disposition", disposition);
  if (validation) headers.set("x-superglm-validation", validation);
  return {
    headers: {
      get(name) {
        return headers.get(name.toLowerCase()) ?? null;
      },
    },
    async blob() {
      return new Blob(["artifact"]);
    },
  };
}

function exportFixture() {
  const action = new FakeElement();
  const dialog = new FakeDialog();
  const close = new FakeElement();
  const joblib = new FakeElement("joblib");
  const xlsx = new FakeElement("xlsx");
  joblib.checked = true;
  const filename = new FakeElement("superglm_edited_model.joblib");
  const directory = new FakeElement(".");
  const download = new FakeElement();
  const saveToKernel = new FakeElement();
  const openDirectory = new FakeElement();
  const status = new FakeElement();
  const saved = [];
  const posts = [];
  const blobPaths = [];
  const client = {
    async requestBlob(path) {
      blobPaths.push(path);
      return blobResponse();
    },
    async postJSON(path, payload) {
      posts.push({ path, payload });
      if (path === "/save_directory") return { path: "/tmp/superglm" };
      if (path === "/open_directory") return { path: payload.path };
      return { path: `${payload.directory}/${payload.filename}` };
    },
  };
  const nodes = {
    action,
    dialog,
    close,
    formatInputs: [joblib, xlsx],
    filename,
    directory,
    download,
    saveToKernel,
    openDirectory,
    status,
  };
  return {
    client,
    nodes,
    action,
    dialog,
    close,
    joblib,
    xlsx,
    filename,
    directory,
    download,
    saveToKernel,
    openDirectory,
    status,
    saved,
    posts,
    blobPaths,
    context: {
      client,
      nodes,
      async saveBlobToFile(blob, name, metadata) {
        saved.push({ blob, filename: name, metadata });
        return `Downloaded ${name}`;
      },
    },
  };
}

test("Python model download uses the export route, server filename, and validation status", async () => {
  const fixture = exportFixture();
  fixture.filename.value = "edited model.joblib";
  fixture.client.requestBlob = async (path) => {
    fixture.blobPaths.push(path);
    return blobResponse({
      disposition: 'attachment; filename="confirmed-model.joblib"',
      validation: "artifact+predictions",
    });
  };
  const binding = bindExportDialog(fixture.context);

  await fixture.download.emit("click");

  assert.deepEqual(fixture.blobPaths, [
    "/download_export?format=joblib&filename=edited%20model.joblib",
  ]);
  assert.equal(fixture.saved.length, 1);
  assert.equal(fixture.saved[0].filename, "confirmed-model.joblib");
  assert.equal(fixture.saved[0].metadata.description, "Python model");
  assert.deepEqual(fixture.saved[0].metadata.accept, {
    "application/octet-stream": [".joblib"],
  });
  assert.match(fixture.status.textContent, /validated.*predictions/i);
  binding.destroy();
});

test("Excel selection applies its default and posts the kernel export payload", async () => {
  const fixture = exportFixture();
  const binding = bindExportDialog(fixture.context);
  fixture.joblib.checked = false;
  fixture.xlsx.checked = true;

  await fixture.xlsx.emit("change");
  await fixture.saveToKernel.emit("click");

  assert.equal(fixture.filename.value, "superglm_rating_tables.xlsx");
  assert.deepEqual(fixture.posts, [{
    path: "/export_file",
    payload: {
      format: "xlsx",
      directory: ".",
      filename: "superglm_rating_tables.xlsx",
    },
  }]);
  assert.equal(fixture.status.textContent, "Saved ./superglm_rating_tables.xlsx");
  assert.doesNotMatch(fixture.status.textContent, /validated/i);
  binding.destroy();
});

test("format changes preserve a custom filename but replace blank and prior defaults", async () => {
  const fixture = exportFixture();
  const binding = bindExportDialog(fixture.context);

  fixture.filename.value = "analyst-output.bin";
  fixture.joblib.checked = false;
  fixture.xlsx.checked = true;
  await fixture.xlsx.emit("change");
  assert.equal(fixture.filename.value, "analyst-output.bin");

  fixture.filename.value = "";
  fixture.xlsx.checked = false;
  fixture.joblib.checked = true;
  await fixture.joblib.emit("change");
  assert.equal(fixture.filename.value, "superglm_edited_model.joblib");

  fixture.xlsx.checked = true;
  fixture.joblib.checked = false;
  await fixture.xlsx.emit("change");
  assert.equal(fixture.filename.value, "superglm_rating_tables.xlsx");
  binding.destroy();
});

test("one pending export suppresses duplicate clicks and restores both actions after failure", async () => {
  const fixture = exportFixture();
  const request = deferred();
  let requests = 0;
  fixture.client.requestBlob = async () => {
    requests += 1;
    return request.promise;
  };
  const binding = bindExportDialog(fixture.context);

  const first = fixture.download.emit("click");
  assert.equal(fixture.download.disabled, true);
  assert.equal(fixture.saveToKernel.disabled, true);
  await fixture.saveToKernel.emit("click");
  await fixture.download.emit("click");
  assert.equal(requests, 1);

  request.reject(new Error("training data required"));
  await first;
  assert.equal(fixture.download.disabled, false);
  assert.equal(fixture.saveToKernel.disabled, false);
  assert.equal(fixture.status.textContent, "training data required");
  binding.destroy();
});

test("dialog open initializes the kernel directory, folder open is optional, and close is owned", async () => {
  const fixture = exportFixture();
  fixture.directory.value = "";
  const binding = bindExportDialog(fixture.context);

  await fixture.action.emit("click");
  assert.equal(fixture.dialog.open, true);
  assert.equal(fixture.dialog.showCalls, 1);
  assert.equal(fixture.directory.value, "/tmp/superglm");
  assert.deepEqual(fixture.posts[0], {
    path: "/save_directory",
    payload: { path: "" },
  });

  await fixture.openDirectory.emit("click");
  assert.deepEqual(fixture.posts[1], {
    path: "/open_directory",
    payload: { path: "/tmp/superglm" },
  });
  assert.equal(fixture.status.textContent, "Opened /tmp/superglm");

  await fixture.close.emit("click");
  assert.equal(fixture.dialog.open, false);
  assert.equal(fixture.dialog.closeCalls, 1);
  binding.destroy();
});

test("cancelled browser save is a normal status and destroy removes every listener", async () => {
  const fixture = exportFixture();
  fixture.context.saveBlobToFile = async () => null;
  const binding = bindExportDialog(fixture.context);

  await fixture.download.emit("click");
  assert.equal(fixture.status.textContent, "Download cancelled.");

  binding.destroy();
  for (const [node, event] of [
    [fixture.action, "click"],
    [fixture.close, "click"],
    [fixture.joblib, "change"],
    [fixture.xlsx, "change"],
    [fixture.download, "click"],
    [fixture.saveToKernel, "click"],
    [fixture.openDirectory, "click"],
  ]) {
    assert.equal(node.listenerCount(event), 0);
  }
  const priorRequests = fixture.blobPaths.length;
  await fixture.download.emit("click");
  assert.equal(fixture.blobPaths.length, priorRequests);
});

test("the dialog says how many waiting changes the export leaves out", async () => {
  const fixture = exportFixture();
  const pendingNote = new FakeElement();
  let waiting = 2;
  const binding = bindExportDialog({
    ...fixture.context,
    nodes: { ...fixture.nodes, pendingNote },
    pendingCount: () => waiting,
  });

  await fixture.action.emit("click");
  assert.equal(pendingNote.hidden, false);
  assert.equal(
    pendingNote.textContent,
    "2 waiting changes are not included. The export is the last refit.",
  );

  fixture.dialog.close();
  waiting = 1;
  await fixture.action.emit("click");
  assert.equal(
    pendingNote.textContent,
    "1 waiting change is not included. The export is the last refit.",
  );

  fixture.dialog.close();
  waiting = 0;
  await fixture.action.emit("click");
  assert.equal(pendingNote.hidden, true);
  assert.equal(pendingNote.textContent, "");
  binding.destroy();
});

test("the Final fit model option follows availability and downloads the final export", async () => {
  const fixture = exportFixture();
  const final = new FakeElement("final");
  fixture.nodes.formatInputs.push(final);
  let available = false;
  const binding = bindExportDialog({ ...fixture.context, finalFitAvailable: () => available });

  await fixture.action.emit("click");
  assert.equal(final.disabled, true);

  available = true;
  fixture.dialog.open = false;
  await fixture.action.emit("click");
  assert.equal(final.disabled, false);
  fixture.joblib.checked = false;
  final.checked = true;
  await final.emit("change");
  assert.equal(fixture.filename.value, "superglm_final_model.joblib");
  await fixture.download.emit("click");
  assert.deepEqual(fixture.blobPaths, [
    "/download_export?format=final&filename=superglm_final_model.joblib",
  ]);

  // Once the model changes, the stale final fit is no longer offered.
  available = false;
  fixture.dialog.open = false;
  await fixture.action.emit("click");
  assert.deepEqual([final.disabled, final.checked, fixture.joblib.checked], [true, false, true]);
  assert.equal(fixture.filename.value, "superglm_edited_model.joblib");
  binding.destroy();
});

test("Structure (JSON) downloads and saves the structure file with its own default name", async () => {
  const fixture = exportFixture();
  const structure = new FakeElement("structure");
  fixture.nodes.formatInputs.push(structure);
  const pendingNote = new FakeElement();
  const binding = bindExportDialog({
    ...fixture.context,
    nodes: { ...fixture.nodes, pendingNote },
    pendingCount: () => 1,
  });

  await fixture.action.emit("click");
  // The structure is the last refit too, so the waiting-changes note stands.
  assert.equal(
    pendingNote.textContent,
    "1 waiting change is not included. The export is the last refit.",
  );
  fixture.joblib.checked = false;
  structure.checked = true;
  await structure.emit("change");
  assert.equal(fixture.filename.value, "superglm_structure.json");

  await fixture.download.emit("click");
  assert.deepEqual(fixture.blobPaths, [
    "/download_export?format=structure&filename=superglm_structure.json",
  ]);
  assert.equal(fixture.saved[0].metadata.description, "Structure (JSON)");
  assert.deepEqual(fixture.saved[0].metadata.accept, { "application/json": [".json"] });
  assert.equal(fixture.status.textContent, "Downloaded superglm_structure.json");

  await fixture.saveToKernel.emit("click");
  assert.deepEqual(fixture.posts.at(-1), {
    path: "/export_file",
    payload: { format: "structure", directory: ".", filename: "superglm_structure.json" },
  });
  assert.equal(fixture.status.textContent, "Saved ./superglm_structure.json");
  binding.destroy();
});
