// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  bindAppBar,
  renderAppBar,
  revertAvailable,
} from "../../src/superglm/editor/app/views/app_bar.js";

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.disabled = false;
    this.isContentEditable = false;
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

  emit(name, properties = {}) {
    const event = {
      target: this,
      key: "",
      ctrlKey: false,
      metaKey: false,
      shiftKey: false,
      altKey: false,
      defaultPrevented: false,
      ...properties,
      preventDefault() {
        this.defaultPrevented = true;
      },
    };
    for (const listener of this.listeners.get(name) ?? []) listener(event);
    return event;
  }

  querySelectorAll() {
    return [];
  }

  querySelector(selector) {
    return selector === "dialog[open]" ? this.openDialog : null;
  }

  closest() {
    return null;
  }
}

class FakeButton extends FakeElement {
  constructor() {
    super("button");
  }
}

test("global undo and redo shortcuts pause while any native dialog is open", (t) => {
  const originalDocument = globalThis.document;
  const originalElement = globalThis.Element;
  const originalHTMLElement = globalThis.HTMLElement;
  const originalButton = globalThis.HTMLButtonElement;
  const documentHub = new FakeElement("document");
  globalThis.document = documentHub;
  globalThis.Element = FakeElement;
  globalThis.HTMLElement = FakeElement;
  globalThis.HTMLButtonElement = FakeButton;
  t.after(() => {
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
    if (originalElement === undefined) delete globalThis.Element;
    else globalThis.Element = originalElement;
    if (originalHTMLElement === undefined) delete globalThis.HTMLElement;
    else globalThis.HTMLElement = originalHTMLElement;
    if (originalButton === undefined) delete globalThis.HTMLButtonElement;
    else globalThis.HTMLButtonElement = originalButton;
  });

  const root = new FakeElement("nav");
  const undoButton = new FakeButton();
  const redoButton = new FakeButton();
  let undoCalls = 0;
  let redoCalls = 0;
  const binding = bindAppBar({
    root,
    undoButton,
    redoButton,
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    onView: () => {},
    onUndo: () => { undoCalls += 1; },
    onRedo: () => { redoCalls += 1; },
    onRevert: () => {},
    onRefresh: () => {},
  });

  documentHub.openDialog = new FakeElement("dialog");
  const blockedUndo = documentHub.emit("keydown", { key: "z", ctrlKey: true });
  const blockedRedo = documentHub.emit("keydown", { key: "y", metaKey: true });
  assert.deepEqual([undoCalls, redoCalls], [0, 0]);
  assert.equal(blockedUndo.defaultPrevented, false);
  assert.equal(blockedRedo.defaultPrevented, false);

  documentHub.openDialog = null;
  const undo = documentHub.emit("keydown", { key: "z", ctrlKey: true });
  const redo = documentHub.emit("keydown", { key: "y", metaKey: true });
  assert.deepEqual([undoCalls, redoCalls], [1, 1]);
  assert.equal(undo.defaultPrevented, true);
  assert.equal(redo.defaultPrevented, true);

  binding.destroy();
});

test("Refresh is disabled while busy and Revert only when something can be reverted", () => {
  const root = new FakeElement("nav");
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
  };
  const render = (overrides) => renderAppBar({
    root,
    activeView: "editor",
    ...buttons,
    undoLabel: null,
    redoLabel: null,
    canRevert: false,
    busy: false,
    ...overrides,
  });

  render({});
  assert.deepEqual(
    [buttons.refreshButton.disabled, buttons.revertButton.disabled],
    [false, true],
  );
  render({ busy: true });
  assert.equal(buttons.refreshButton.disabled, true);
  render({ canRevert: true });
  assert.deepEqual(
    [buttons.refreshButton.disabled, buttons.revertButton.disabled],
    [false, false],
  );
});

test("Undo and Redo follow the snapshot and name what they would take", () => {
  const root = new FakeElement("nav");
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
  };
  const render = (undoLabel, redoLabel) => {
    renderAppBar({
      root, activeView: "editor", ...buttons, undoLabel, redoLabel, canRevert: false, busy: false,
    });
    const { undoButton, redoButton } = buttons;
    return [undoButton.disabled, undoButton.dataset.popoverBody,
      redoButton.disabled, redoButton.dataset.popoverBody];
  };

  assert.deepEqual(render("Line 30–45 in age", null),
    [false, "Undo: Line 30–45 in age", true, "Nothing to redo."]);
  assert.deepEqual(render(null, "shift age"),
    [true, "Nothing to undo.", false, "Redo: shift age"]);
});

test("Revert is available whenever anything differs from the opened model", () => {
  const snapshot = (overrides) => ({
    timeline: [{ kind: "marker" }],
    undo_redo: { undo: null, redo: null },
    in_force_is_original: true,
    ...overrides,
  });
  const marker = { kind: "marker" };
  const edit = (redo) => ({ kind: "edit", label: "shift age", redo });
  const step = (redo) => ({ kind: "structural", label: "Line 30–45 in age", redo });
  assert.equal(revertAvailable(snapshot({})), false);
  assert.equal(revertAvailable(snapshot({ timeline: [edit(false), marker] })), true);
  assert.equal(revertAvailable(snapshot({ timeline: [step(false), edit(false), marker] })), true);
  // An undone edit or step differs from nothing in force, and Revert would
  // only push a step that changes nothing.
  assert.equal(revertAvailable(snapshot({ timeline: [marker, edit(true)] })), false);
  // Edits before a step were set aside by it, so they are not live.
  assert.equal(revertAvailable(snapshot({ timeline: [edit(false), step(false), marker] })), false);
  assert.equal(
    revertAvailable(snapshot({ undo_redo: { undo: "revert to original model", redo: "x" } })),
    false,
  );
  // A structural step or a distribution re-profile puts another model in force.
  assert.equal(revertAvailable(snapshot({ in_force_is_original: false })), true);
});
