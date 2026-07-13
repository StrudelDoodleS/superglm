// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { bindAppBar } from "../../src/superglm/editor/app/views/app_bar.js";

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
    onView: () => {},
    onUndo: () => { undoCalls += 1; },
    onRedo: () => { redoCalls += 1; },
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
