// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { bindToolRail, renderToolRail } from "../../src/superglm/editor/app/views/tool_rail.js";

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  toggle(value, active) {
    if (active) this.values.add(value);
    else this.values.delete(value);
  }

  contains(value) {
    return this.values.has(value);
  }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.disabled = false;
    this.isContentEditable = false;
  }

  closest(selector) {
    return selector === "[data-tool]" && this.dataset.tool ? this : null;
  }
}

class FakeButton extends FakeElement {
  constructor(tool) {
    super("button");
    this.dataset.tool = tool;
    this.attributes = new Map();
    this.classList = new FakeClassList();
    this.tabIndex = -1;
    this.focused = false;
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

class FakeEventHub extends FakeElement {
  constructor(buttons = []) {
    super("nav");
    this.buttons = buttons;
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

  emit(name, properties) {
    const event = {
      defaultPrevented: false,
      ctrlKey: false,
      metaKey: false,
      altKey: false,
      ...properties,
      preventDefault() {
        this.defaultPrevented = true;
      },
    };
    for (const listener of this.listeners.get(name) ?? []) listener(event);
    return event;
  }

  querySelectorAll(selector) {
    return selector === '[role="radio"]' ? this.buttons.filter((button) => button.dataset.tool !== "help") : [];
  }

  querySelector(selector) {
    const match = selector.match(/^\[data-tool="(.+)"\]$/);
    return match ? this.buttons.find((button) => button.dataset.tool === match[1]) ?? null : null;
  }

  contains(element) {
    return this.buttons.includes(element);
  }
}

globalThis.Element = FakeElement;
globalThis.HTMLElement = FakeElement;
globalThis.HTMLButtonElement = FakeButton;

test("tool rail owns exclusive semantics, shortcuts, roving focus, and cleanup", () => {
  const buttons = ["select", "move", "zoom", "handles", "help"].map(
    (tool) => new FakeButton(tool),
  );
  const root = new FakeEventHub(buttons);
  const shortcuts = new FakeEventHub();
  const modes = [];
  let helpCount = 0;
  const binding = bindToolRail({
    root,
    shortcutRoot: shortcuts,
    onMode: (mode) => modes.push(mode),
    onHelp: () => helpCount += 1,
  });

  renderToolRail(root, { mode: "select", handlesAvailable: false });
  assert.equal(buttons[0].getAttribute("aria-checked"), "true");
  assert.equal(buttons[0].tabIndex, 0);
  assert.equal(buttons[3].disabled, true);

  root.emit("click", { target: buttons[1] });
  assert.deepEqual(modes, ["move"]);
  root.emit("keydown", { target: buttons[1], key: "ArrowDown" });
  assert.equal(buttons[2].focused, true);
  assert.deepEqual(modes, ["move", "zoom"]);

  shortcuts.emit("keydown", { target: root, key: "v" });
  shortcuts.emit("keydown", { target: root, key: "?" });
  assert.deepEqual(modes, ["move", "zoom", "select"]);
  assert.equal(helpCount, 1);

  binding.destroy();
  root.emit("click", { target: buttons[1] });
  shortcuts.emit("keydown", { target: root, key: "?" });
  assert.deepEqual(modes, ["move", "zoom", "select"]);
  assert.equal(helpCount, 1);
});

test("unavailable Handles falls back to the sole enabled Select radio", () => {
  const buttons = ["select", "move", "zoom", "handles", "help"].map(
    (tool) => new FakeButton(tool),
  );
  const root = new FakeEventHub(buttons);

  renderToolRail(root, { mode: "handles", handlesAvailable: false });

  assert.equal(buttons[0].disabled, false);
  assert.equal(buttons[0].getAttribute("aria-checked"), "true");
  assert.equal(buttons[0].tabIndex, 0);
  assert.equal(buttons[3].disabled, true);
  assert.equal(buttons[3].getAttribute("aria-checked"), "false");
  assert.equal(buttons[3].tabIndex, -1);
  assert.deepEqual(
    buttons.slice(0, 4).filter((button) => !button.disabled && button.tabIndex === 0),
    [buttons[0]],
  );
});
