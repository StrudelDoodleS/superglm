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

/** A target that hands each listener the events dispatched on it. */
function fakeTarget() {
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

/** A document, and its window. */
function fakeDocument() {
  return Object.assign(fakeTarget(), { defaultView: fakeTarget() });
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
  assert.equal(doc.defaultView.count(), 0);
});

test("the page's window losing focus lets the wheel go, as a press in the notebook around the editor does", () => {
  const { doc, focus, editor, point, bordered } = setup();
  const view = doc.defaultView;
  doc.dispatch("pointerdown", { target: point });
  // An element's blur reaching the window is not the window's.
  view.dispatch("blur", { target: point });
  assert.ok(focus.isActive(editor));
  view.dispatch("blur", { target: view });
  assert.ok(!focus.isActive(editor));
  assert.deepEqual(bordered(), []);
});

test("a chart drawn in another's place lets the wheel go, and the same chart drawn again keeps it", () => {
  const { doc, focus, check, bordered } = setup();
  focus.drawn(check, "check_qq");
  doc.dispatch("pointerdown", { target: check });
  focus.drawn(check, "check_qq");
  assert.ok(focus.isActive(check), "a redraw of the same chart keeps the wheel");
  assert.deepEqual(bordered(), [check]);
  // Another check opened in the same frame, by the keyboard: nothing was pressed outside it.
  focus.drawn(check, "check_pit");
  assert.ok(!focus.isActive(check));
  assert.deepEqual(bordered(), []);
  // Drawn as none, as a check closed is, it lets go too.
  doc.dispatch("focusin", { target: check });
  focus.drawn(check, null);
  assert.ok(!focus.isActive(check));
});
