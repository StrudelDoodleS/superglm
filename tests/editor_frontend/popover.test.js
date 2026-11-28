// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  POPOVER,
  PopoverDelay,
  TOOLTIP_SHOW_DELAY_MS,
  TOOLTIP_SWITCH_WINDOW_MS,
  bindPopovers,
} from "../../src/superglm/editor/app/views/popover.js";

function fakeScheduler() {
  /** @type {Map<number, {next:() => void, delay:number}>} */
  const timers = new Map();
  let nextId = 1;

  return {
    /** @param {() => void} next @param {number} delay */
    setTimer(next, delay) {
      const id = nextId;
      nextId += 1;
      timers.set(id, { next, delay });
      return id;
    },
    /** @param {number} id */
    clearTimer(id) {
      timers.delete(id);
    },
    /** Run the timers set for ``delay``. */
    flush(delay = TOOLTIP_SHOW_DELAY_MS) {
      for (const [id, timer] of [...timers]) {
        if (timer.delay !== delay) continue;
        timers.delete(id);
        timer.next();
      }
    },
  };
}

test("pointer disclosure waits 350ms and pointer dismissal is immediate", () => {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });

  delay.pointerEnter();
  assert.deepEqual(events, []);
  scheduler.flush();
  assert.deepEqual(events, ["show"]);
  delay.pointerLeave();
  assert.deepEqual(events, ["show", "hide"]);
});

test("keyboard focus shows immediately and Escape hides", () => {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });

  delay.focus();
  delay.escape();
  assert.deepEqual(events, ["show", "hide"]);
});

function delayed() {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });
  return { scheduler, events, delay };
}

test("moving from an open tooltip to a neighbour swaps at once, with no second wait", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.pointerLeave();
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide", "show"]);
});

test("after the switch window has passed the next tooltip waits again", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.pointerLeave();
  scheduler.flush(TOOLTIP_SWITCH_WINDOW_MS);
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide"]);
  scheduler.flush();
  assert.deepEqual(events, ["show", "hide", "show"]);
});

test("closing a tooltip on purpose makes the next one wait", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.escape();
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide"]);
});

test("a tooltip that never opened does not make the next one instant", () => {
  const { events, delay } = delayed();
  delay.pointerEnter();
  delay.pointerLeave();
  delay.pointerEnter();
  assert.deepEqual(events, []);
});

// The popover owner on a minimal element tree: attributes, `data-` attributes as `dataset`, a
// parent chain to the page's root, and listeners on the root, where every event is sent.
class FakeEvent {
  constructor(type, init = {}) { Object.assign(this, init); this.type = type; }
}
class FakePointerEvent extends FakeEvent {}
class FakeFocusEvent extends FakeEvent {}
class FakeKeyboardEvent extends FakeEvent {}
const camelToData = (name) => `data-${name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;
class FakeElement {
  constructor(attrs = {}) {
    this.attrs = new Map(Object.entries(attrs));
    this.children = []; this.parent = null; this.listeners = new Map(); this.style = {}; this.text = "";
    this.focusVisible = false; this.isRoot = false;
    this.rect = { left: 100, top: 100, right: 140, bottom: 120, width: 40, height: 20 };
    this.dataset = new Proxy({}, {
      get: (_, key) => (typeof key === "string" ? this.attrs.get(camelToData(key)) : undefined),
      set: (_, key, value) => { this.attrs.set(camelToData(String(key)), String(value)); return true; },
      deleteProperty: (_, key) => { this.attrs.delete(camelToData(String(key))); return true; },
    });
  }
  get id() { return this.attrs.get("id") ?? ""; }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(value) { if (value) this.attrs.set("hidden", ""); else this.attrs.delete("hidden"); }
  get textContent() { return this.text; }
  set textContent(value) { this.text = String(value); }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  hasAttribute(name) { return this.attrs.has(name); }
  removeAttribute(name) { this.attrs.delete(name); }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; }
  get isConnected() { for (let node = this; node; node = node.parent) if (node.isRoot) return true; return false; }
  contains(other) { for (let node = other; node; node = node.parent) if (node === this) return true; return false; }
  matches(selector) {
    if (selector === ":focus-visible") return this.focusVisible;
    return selector.split(",").some((part) => {
      const name = /^\s*\[([^\]=]+)\]\s*$/.exec(part)?.[1];
      return name !== undefined && this.attrs.has(name);
    });
  }
  closest(selector) { for (let node = this; node; node = node.parent) if (node.matches(selector)) return node; return null; }
  querySelector(selector) {
    for (const child of this.children) {
      if (child.matches(selector)) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  getBoundingClientRect() { return this.rect; }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter((l) => l !== listener)); }
  send(event) { for (const listener of this.listeners.get(event.type) ?? []) listener(event); }
}

/** The page's root, its popover, and the popover owner bound to them, its timers fired by hand. */
function popoverPage() {
  const scheduler = fakeScheduler();
  Object.assign(globalThis, {
    Element: FakeElement, HTMLElement: FakeElement, Node: FakeElement,
    PointerEvent: FakePointerEvent, FocusEvent: FakeFocusEvent, KeyboardEvent: FakeKeyboardEvent,
    window: { setTimeout: scheduler.setTimer, clearTimeout: scheduler.clearTimer, innerWidth: 1000, innerHeight: 800 },
  });
  const root = new FakeElement();
  root.isRoot = true;
  const popover = new FakeElement({ id: "uiPopover", hidden: "" });
  const heading = new FakeElement({ "data-popover-heading": "" });
  const body = new FakeElement({ "data-popover-description": "" });
  popover.append(heading, body);
  root.append(popover);
  const owner = bindPopovers({ root, popover });
  const hover = (anchor) => { root.send(new FakePointerEvent("pointerover", { target: anchor })); scheduler.flush(); };
  return { scheduler, root, popover, body, owner, hover };
}

const WAITING = "Waiting for the edited model's answer; the comparison is ready once it has loaded.";

/** The Compare switch, disabled with `body` its reason. */
function compareButton(body) {
  return new FakeElement({
    id: "checkCompare", role: "switch", "aria-disabled": "true", "data-help-control": "compare_original",
    "data-popover-title": "Compare with the fit", "data-popover-body": body,
  });
}

test("a shown popover reads its anchor again when the anchor's help is updated under it", () => {
  const { root, popover, body, owner, hover } = popoverPage();
  const button = compareButton(WAITING);
  root.append(button);
  hover(button);
  assert.equal(popover.hidden, false);
  assert.equal(body.textContent, WAITING);
  // The check lands and the switch turns live, under a pointer that did not move.
  button.setAttribute("aria-disabled", "false");
  button.dataset.popoverBody = "Draws the model as fitted in grey beside the model as edited in blue, on every check. "
    + "It changes only what is drawn.";
  button.rect = { ...button.rect, left: 300, right: 340 };
  owner.anchorUpdated();
  assert.equal(popover.hidden, false);
  assert.match(body.textContent, /^Draws the model as fitted/);
  assert.equal(button.getAttribute("aria-describedby"), "uiPopover");
});

test("help whose anchor left the page, or was hidden, closes when the anchor is updated", () => {
  const { root, popover, owner, hover } = popoverPage();
  const note = new FakeElement({ "data-popover-title": "Parameters taken as exact", "data-popover-body": "Simulated." });
  root.append(note);
  hover(note);
  assert.equal(popover.hidden, false);
  note.remove();
  owner.anchorUpdated();
  assert.equal(popover.hidden, true);

  const panel = new FakeElement();
  const button = compareButton(WAITING);
  panel.append(button);
  root.append(panel);
  hover(button);
  assert.equal(popover.hidden, false);
  panel.hidden = true;
  owner.anchorUpdated();
  assert.equal(popover.hidden, true);
  assert.equal(button.getAttribute("aria-describedby"), null);
});

test("help waiting its delay shows its anchor's newest text, and an update shows nothing early", () => {
  const { scheduler, root, popover, body, owner } = popoverPage();
  const button = compareButton(WAITING);
  root.append(button);
  root.send(new FakePointerEvent("pointerover", { target: button }));
  button.dataset.popoverBody = "No check shown has the model as fitted to draw beside the edit.";
  owner.anchorUpdated();
  assert.equal(popover.hidden, true);
  scheduler.flush();
  assert.equal(body.textContent, "No check shown has the model as fitted to draw beside the edit.");
});

// Each event of the popover's table, sent to a delay held in the state it names.
const POPOVER_EVENTS = {
  enter: (delay) => delay.pointerEnter(),
  focus: (delay) => delay.focus(),
  leave: (delay) => delay.pointerLeave(),
  escape: (delay) => delay.escape(),
  due: (_delay, scheduler) => scheduler.flush(TOOLTIP_SHOW_DELAY_MS),
  lapse: (_delay, scheduler) => scheduler.flush(TOOLTIP_SWITCH_WINDOW_MS),
  update: (delay) => delay.update(),
};
const INTO = {
  closed: () => {},
  pending: (delay) => delay.pointerEnter(),
  open: (delay) => delay.focus(),
  recent: (delay) => { delay.focus(); delay.pointerLeave(); },
};

test("the popover table: every state and event, each move landing where it says", () => {
  assert.deepEqual([...POPOVER.states], ["closed", "pending", "open", "recent"]);
  assert.deepEqual([...POPOVER.events].sort(), Object.keys(POPOVER_EVENTS).sort());
  for (const state of POPOVER.states) {
    for (const event of POPOVER.events) {
      const { scheduler, events, delay } = delayed();
      INTO[state](delay);
      assert.equal(delay.state, state);
      events.length = 0;
      POPOVER_EVENTS[event](delay, scheduler);
      const target = POPOVER.next(state, event) ?? state;
      assert.equal(delay.state, target, `${state} --${event}--> ${target}`);
      assert.equal(delay.visible, target === "open");
      // Help is written on reaching open, and again on an update there; it is taken down on leaving it.
      const wrote = (target === "open" && (state !== "open" || event === "update")) ? ["show"] : [];
      assert.deepEqual(events, state === "open" && target !== "open" ? ["hide"] : wrote, `${state} --${event}`);
    }
  }
});
