// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { ACTIVE_CLASS, CHART_FOCUS, bindChartFocus } from "../../src/superglm/editor/app/views/chart_focus.js";

/** An element with a class list, a parent, and a box while `shown`. */
function element(parent = null) {
  const classes = new Set();
  const node = {
    parent,
    shown: true,
    isConnected: true,
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
    },
    contains(other) {
      for (let at = other; at; at = at.parent) if (at === node) return true;
      return false;
    },
    getClientRects: () => (node.shown ? [{}] : []),
  };
  return node;
}

/** A document that hands each listener the events dispatched on it. */
function fakeDocument() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) { listeners.set(type, [...(listeners.get(type) ?? []), listener]); },
    removeEventListener(type, listener) { listeners.set(type, (listeners.get(type) ?? []).filter((l) => l !== listener)); },
    dispatch(type, init) {
      const event = { type, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...init };
      for (const listener of listeners.get(type) ?? []) listener(event);
      return event;
    },
    count: () => [...listeners.values()].reduce((sum, list) => sum + list.length, 0),
  };
}

function setup() {
  const doc = fakeDocument();
  const focus = bindChartFocus(doc);
  const page = element();
  const editor = element(page);
  const point = element(editor);
  const check = element(page);
  const home = element(page);
  const outside = element(page);
  focus.register(editor);
  focus.register(check, [home]);
  const bordered = () => [editor, check].filter((chart) => chart.classList.contains(ACTIVE_CLASS));
  return { doc, focus, editor, point, check, home, outside, bordered };
}

test("the wheel focus table: a chart is activated, another takes over, and anything else lets go", () => {
  assert.deepEqual([...CHART_FOCUS.states], ["idle", "active"]);
  assert.equal(CHART_FOCUS.initial, "idle");
  const walked = CHART_FOCUS.states.flatMap((state) => CHART_FOCUS.events.map((event) => [state, event, CHART_FOCUS.next(state, event)]));
  assert.deepEqual(walked, [
    ["idle", "activate", "active"],
    ["idle", "deactivate", null],
    ["active", "activate", "active"],
    ["active", "deactivate", "idle"],
  ]);
});

test("a press inside a chart activates it without taking the press, and one outside lets it go", () => {
  const { doc, focus, editor, point, check, home, outside, bordered } = setup();
  assert.deepEqual(bordered(), []);
  const press = doc.dispatch("pointerdown", { target: point });
  assert.equal(press.defaultPrevented, false);
  assert.ok(focus.isActive(editor));
  assert.deepEqual(bordered(), [editor]);
  // At most one chart is active: the check takes over, from its plot or from its Home button.
  doc.dispatch("pointerdown", { target: check });
  assert.deepEqual(bordered(), [check]);
  assert.ok(!focus.isActive(editor));
  doc.dispatch("pointerdown", { target: point });
  doc.dispatch("pointerdown", { target: home });
  assert.deepEqual(bordered(), [check]);
  doc.dispatch("pointerdown", { target: outside });
  assert.deepEqual(bordered(), []);
  assert.ok(!focus.isActive(check));
});

test("Escape lets the active chart go, and focusing a chart activates it", () => {
  const { doc, focus, editor, point, check, bordered } = setup();
  doc.dispatch("focusin", { target: point });
  assert.deepEqual(bordered(), [editor]);
  doc.dispatch("keydown", { key: "a" });
  assert.ok(focus.isActive(editor));
  const escape = doc.dispatch("keydown", { key: "Escape" });
  assert.equal(escape.defaultPrevented, false);
  assert.deepEqual(bordered(), []);
  doc.dispatch("focusin", { target: check });
  assert.ok(focus.isActive(check));
});

test("a chart that leaves the page is no longer active", () => {
  const { doc, focus, editor, point, check, bordered } = setup();
  doc.dispatch("pointerdown", { target: point });
  editor.shown = false;
  focus.sync();
  assert.deepEqual(bordered(), []);
  assert.ok(!focus.isActive(editor));
  editor.shown = true;
  assert.ok(!focus.isActive(editor), "coming back does not activate it");
  doc.dispatch("pointerdown", { target: check });
  check.isConnected = false;
  assert.ok(!focus.isActive(check));
  assert.deepEqual(bordered(), []);
  focus.destroy();
  assert.equal(doc.count(), 0);
});
