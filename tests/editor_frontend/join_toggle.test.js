// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  bindJoinToggle,
  effectiveShapeJoin,
  readShapeJoin,
  renderJoinToggle,
  storeShapeJoin,
} from "../../src/superglm/editor/app/views/join_toggle.js";

class FakeButton {
  constructor(join) {
    this.dataset = { join };
    this.attributes = new Map();
    this.tabIndex = -1;
    this.focused = false;
  }

  closest(selector) {
    return selector === "[data-join]" ? this : null;
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  focus() {
    this.focused = true;
  }
}

class FakeRoot {
  constructor(buttons) {
    this.buttons = buttons;
    this.listeners = new Map();
  }

  querySelectorAll(selector) {
    return selector === "[data-join]" ? this.buttons : [];
  }

  contains(node) {
    return this.buttons.includes(node);
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type) {
    this.listeners.delete(type);
  }

  emit(type, event) {
    this.listeners.get(type)?.(event);
  }
}

function fixture() {
  const buttons = [new FakeButton("tangent"), new FakeButton("kink")];
  const root = new FakeRoot(buttons);
  const chosen = [];
  const binding = bindJoinToggle(root, { onChange: (join) => chosen.push(join) });
  return { buttons, root, chosen, binding };
}

test("the toggle reports a click on a join and steps between them with the arrow keys", () => {
  const { buttons, root, chosen, binding } = fixture();
  globalThis.HTMLButtonElement = FakeButton;
  globalThis.Element = FakeButton;
  try {
    root.emit("click", { target: buttons[1] });
    root.emit("click", { target: { closest: () => null } });
    let prevented = 0;
    root.emit("keydown", { target: buttons[0], key: "ArrowRight", preventDefault: () => prevented++ });
    root.emit("keydown", { target: buttons[1], key: "ArrowRight", preventDefault: () => prevented++ });
    root.emit("keydown", { target: buttons[0], key: "Tab", preventDefault: () => prevented++ });
    assert.deepEqual(chosen, ["kink", "kink", "tangent"]);
    assert.equal(prevented, 2);
    assert.equal(buttons[1].focused && buttons[0].focused, true);
    binding.destroy();
    assert.equal(root.listeners.size, 0);
  } finally {
    delete globalThis.HTMLButtonElement;
    delete globalThis.Element;
  }
});

test("rendering marks the chosen join checked and makes it the group's tab stop", () => {
  const { buttons, root } = fixture();
  globalThis.HTMLButtonElement = FakeButton;
  try {
    renderJoinToggle(root, "kink");
    assert.deepEqual(
      buttons.map((button) => [button.getAttribute("aria-checked"), button.tabIndex]),
      [["false", -1], ["true", 0]],
    );
    renderJoinToggle(root, "tangent");
    assert.deepEqual(buttons.map((button) => button.tabIndex), [0, -1]);
  } finally {
    delete globalThis.HTMLButtonElement;
  }
});

test("the join is remembered in storage and is Tangent without one or with storage blocked", () => {
  const store = new Map();
  const storage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, value),
  };
  assert.equal(readShapeJoin(storage), "tangent");
  storeShapeJoin("kink", storage);
  assert.deepEqual([...store.entries()], [["superglm.editor.shapeJoin", "kink"]]);
  assert.equal(readShapeJoin(storage), "kink");
  store.set("superglm.editor.shapeJoin", "smooth");
  assert.equal(readShapeJoin(storage), "tangent");

  const blocked = {
    getItem() { throw new Error("storage disabled"); },
    setItem() { throw new Error("storage disabled"); },
  };
  assert.equal(readShapeJoin(blocked), "tangent");
  assert.doesNotThrow(() => storeShapeJoin("kink", blocked));
});

test("a term that cannot take the chosen join gets the first it can, and the choice is kept", () => {
  assert.equal(effectiveShapeJoin("tangent", undefined), "tangent");
  assert.equal(effectiveShapeJoin("tangent", ["tangent", "kink"]), "tangent");
  assert.equal(effectiveShapeJoin("tangent", ["kink"]), "kink");
  assert.equal(effectiveShapeJoin("kink", ["kink"]), "kink");
});

test("a join the term cannot take is disabled, says why, and ignores clicks and arrows", () => {
  const { buttons, root, chosen, binding } = fixture();
  globalThis.HTMLButtonElement = FakeButton;
  globalThis.Element = FakeButton;
  try {
    buttons[0].dataset.popoverBody = "The curve leaves the shape along its slope.";
    renderJoinToggle(root, "kink", ["kink"], "A degree-1 spline cannot join a range along its tangent.");
    assert.deepEqual(
      buttons.map((button) => [button.getAttribute("aria-disabled"), button.getAttribute("aria-checked")]),
      [["true", "false"], ["false", "true"]],
    );
    assert.equal(buttons[0].dataset.popoverBody, "A degree-1 spline cannot join a range along its tangent.");
    root.emit("click", { target: buttons[0] });
    root.emit("keydown", { target: buttons[1], key: "ArrowLeft", preventDefault: () => {} });
    assert.deepEqual(chosen, ["kink"]);

    renderJoinToggle(root, "tangent", ["tangent", "kink"], null);
    assert.equal(buttons[0].getAttribute("aria-disabled"), "false");
    assert.equal(buttons[0].dataset.popoverBody, "The curve leaves the shape along its slope.");
    binding.destroy();
  } finally {
    delete globalThis.HTMLButtonElement;
    delete globalThis.Element;
  }
});
