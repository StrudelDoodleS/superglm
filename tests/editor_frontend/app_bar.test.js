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
    this.hidden = false;
    this.textContent = "";
    this.isContentEditable = false;
    this.listeners = new Map();
    this.attributes = new Map();
    this.classes = new Set();
    this.classList = {
      toggle: (name, force) => (force ? this.classes.add(name) : this.classes.delete(name)),
      contains: (name) => this.classes.has(name),
    };
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
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
    refitButton: new FakeButton(),
    onView: () => {},
    onUndo: () => { undoCalls += 1; },
    onRedo: () => { redoCalls += 1; },
    onRevert: () => {},
    onRefresh: () => {},
    onRefit: () => {},
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
    refitButton: new FakeButton(),
    refitCount: new FakeElement("span"),
  };
  const render = (overrides) => renderAppBar({
    root,
    activeView: "editor",
    ...buttons,
    undoLabel: null,
    redoLabel: null,
    canRevert: false,
    busy: false,
    pendingCount: 0,
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
    refitButton: new FakeButton(),
    refitCount: new FakeElement("span"),
  };
  const render = (undoLabel, redoLabel) => {
    renderAppBar({
      root, activeView: "editor", ...buttons, undoLabel, redoLabel, canRevert: false, busy: false,
      pendingCount: 0,
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
  // A change waiting for Refit is something Revert takes back too; an undone one is not.
  const waiting = {
    kind: "pending", status: "waiting", label: "collapse B10 + B11 in brand", redo: false,
  };
  assert.equal(revertAvailable(snapshot({ timeline: [waiting, marker] })), true);
  assert.equal(revertAvailable(snapshot({ timeline: [marker, { ...waiting, redo: true }] })), false);
  // A structural step or a distribution re-profile puts another model in force.
  assert.equal(revertAvailable(snapshot({ in_force_is_original: false })), true);
});

function installDocument(t) {
  const saved = ["document", "Element", "HTMLElement", "HTMLButtonElement"].map(
    (name) => [name, globalThis[name]],
  );
  const documentHub = new FakeElement("document");
  globalThis.document = documentHub;
  globalThis.Element = FakeElement;
  globalThis.HTMLElement = FakeElement;
  globalThis.HTMLButtonElement = FakeButton;
  t.after(() => {
    for (const [name, value] of saved) {
      if (value === undefined) delete globalThis[name];
      else globalThis[name] = value;
    }
  });
  return documentHub;
}

test("Refit shows the waiting count, and only a count enables it", () => {
  const root = new FakeElement("nav");
  const buttons = {
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton: new FakeButton(),
    refitCount: new FakeElement("span"),
  };
  const render = (pendingCount, busy = false) => renderAppBar({
    root, activeView: "editor", ...buttons, undoLabel: null, redoLabel: null,
    canRevert: false, busy, pendingCount,
  });
  const { refitButton, refitCount } = buttons;

  render(0);
  assert.deepEqual(
    [refitButton.disabled, refitCount.hidden, refitButton.classList.contains("has-pending")],
    [true, true, false],
  );
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, nothing waiting");
  render(2);
  assert.deepEqual(
    [refitButton.disabled, refitCount.hidden, refitCount.textContent,
      refitButton.classList.contains("has-pending")],
    [false, false, "2", true],
  );
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, 2 changes waiting");
  assert.equal(
    refitButton.dataset.popoverBody,
    "Apply 2 changes in one fit. Hand edits on terms whose structure did not change are kept.",
  );
  render(1);
  assert.equal(refitButton.getAttribute("aria-label"), "Refit, 1 change waiting");
  render(1, true);
  assert.equal(refitButton.disabled, true);
});

test("R refits what is waiting, except while typing, with a modifier, in a dialog, or with nothing waiting", (t) => {
  const documentHub = installDocument(t);
  const root = new FakeElement("nav");
  const refitButton = new FakeButton();
  let refits = 0;
  const binding = bindAppBar({
    root,
    undoButton: new FakeButton(),
    redoButton: new FakeButton(),
    revertButton: new FakeButton(),
    refreshButton: new FakeButton(),
    refitButton,
    onView: () => {},
    onUndo: () => {},
    onRedo: () => {},
    onRevert: () => {},
    onRefresh: () => {},
    onRefit: () => { refits += 1; },
  });

  const pressed = documentHub.emit("keydown", { key: "r" });
  assert.equal(refits, 1);
  assert.equal(pressed.defaultPrevented, true);
  documentHub.emit("keydown", { key: "R", shiftKey: true });
  assert.equal(refits, 2);

  // Reload stays the browser's; typing and dialogs keep their keys.
  documentHub.emit("keydown", { key: "r", ctrlKey: true });
  documentHub.emit("keydown", { key: "r", metaKey: true });
  documentHub.emit("keydown", { key: "r", target: new FakeElement("input") });
  documentHub.openDialog = new FakeElement("dialog");
  documentHub.emit("keydown", { key: "r" });
  documentHub.openDialog = null;
  refitButton.disabled = true;
  documentHub.emit("keydown", { key: "r" });
  assert.equal(refits, 2);

  refitButton.disabled = false;
  refitButton.emit("click");
  assert.equal(refits, 3);
  binding.destroy();
  documentHub.emit("keydown", { key: "r" });
  refitButton.emit("click");
  assert.equal(refits, 3);
});
