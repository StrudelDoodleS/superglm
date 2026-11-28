// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { REPORT_VIEWS } from "../../src/superglm/editor/app/api/contracts.js";
import { LIST_STOP, assessListMarkup, bindAssessList, stepEntry } from "../../src/superglm/editor/app/views/assess_list.js";

test("the list marks the open entry and keeps one tab stop", () => {
  const html = assessListMarkup("diagnose");
  assert.equal((html.match(/aria-current="true"/g) ?? []).length, 1);
  assert.match(html, /data-assess-entry="diagnose"[^>]*aria-current="true"[^>]*tabindex="0"/);
  assert.equal((html.match(/tabindex="0"/g) ?? []).length, 1);
  assert.match(html, /data-popover-title="Smoothing search"/);
});

test("arrow keys, Home and End step through the entries", () => {
  const ids = ["assess", ...REPORT_VIEWS];
  assert.equal(stepEntry(ids, "assess", "ArrowDown"), "validation");
  assert.equal(stepEntry(ids, "assess", "ArrowUp"), "assess");
  assert.equal(stepEntry(ids, "cv", "End"), "diagnose");
  assert.equal(stepEntry(ids, "cv", "Home"), "assess");
  assert.equal(stepEntry(ids, "cv", "a"), null);
});

test("the list hides checks the model does not offer and marks flagged ones", () => {
  const markup = assessListMarkup("assess", { offered: ["check_qq"], flagged: ["check_qq"] });
  assert.match(markup, /data-assess-entry="check_qq"[^>]*data-flagged="true"/);
  assert.match(markup, /data-assess-entry="check_pit"[^>]*hidden/);
  assert.match(markup, /data-assess-group="calibration"[^>]*hidden/);
  assert.match(markup, /data-assess-entry="check_qq"[^>]*aria-label="Q-Q plot, flagged"/);
  assert.doesNotMatch(markup, /data-assess-group="distribution"[^>]*hidden/);
  assert.doesNotMatch(markup, /data-assess-entry="check_qq"[^>]*hidden/);
  // A row neither hidden nor flagged keeps its own name.
  assert.match(markup, /data-assess-entry="validation"[^>]*data-flagged="false"[^>]*aria-label="Validation"/);
  const none = assessListMarkup("assess");
  assert.equal((none.match(/ hidden/g) ?? []).length, 6);
});

// The list on a minimal tree of its rows and groups, which records the row that took focus.
class ListNode {
  constructor(data) {
    this.dataset = { ...data }; this.attrs = new Map(); this.tabIndex = -1; this.children = []; this.focusedRows = null;
  }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(value) { if (value) this.attrs.set("hidden", ""); else this.attrs.delete("hidden"); }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); }
  closest(selector) { return selector === "[data-assess-entry]" && this.dataset.assessEntry ? this : null; }
  querySelector(selector) {
    return selector === "[data-assess-entry]:not([hidden])" ? this.children.find((row) => !row.hidden) ?? null : null;
  }
  focus() { this.focusedRows.push(this.dataset.assessEntry); }
}

function fakeList() {
  globalThis.Element = ListNode;
  globalThis.HTMLElement = ListNode;
  const focused = [];
  const ids = ["assess", "validation", "cv", "final", "diagnose", "check_qq", "check_pit", "check_ae", "check_calibration"];
  const rows = ids.map((id) => Object.assign(new ListNode({ assessEntry: id }), { focusedRows: focused }));
  const groups = [["reports", 1, 5], ["distribution", 5, 7], ["calibration", 7, 9]].map(([key, from, to]) =>
    Object.assign(new ListNode({ assessGroup: key }), { children: rows.slice(from, to) }));
  const listeners = new Map();
  const root = {
    querySelector: (selector) => (selector === "[data-assess-entry]" ? rows[0] : null),
    querySelectorAll: (selector) => (selector === "[data-assess-entry]" ? rows : selector === "[data-assess-group]" ? groups : []),
    contains: (node) => rows.includes(node),
    addEventListener: (type, listener) => listeners.set(type, listener),
    removeEventListener: (type) => listeners.delete(type),
  };
  const list = bindAssessList(root, { onOpen: () => {} });
  const row = (id) => rows.find((each) => each.dataset.assessEntry === id);
  const key = (id, name) => listeners.get("keydown")({ key: name, target: row(id), preventDefault() {} });
  const stop = () => rows.filter((each) => each.tabIndex === 0).map((each) => each.dataset.assessEntry);
  return { list, row, key, stop, focused };
}

const ALL = { offered: ["check_qq", "check_pit", "check_ae", "check_calibration"], flagged: [] };

test("a row the arrows reached keeps the tab stop while a check's flag lands", () => {
  const { list, row, key, stop, focused } = fakeList();
  list.render("check_qq", ALL);
  assert.deepEqual(stop(), ["check_qq"]);
  key("check_qq", "ArrowUp");
  assert.deepEqual(focused, ["diagnose"]);
  assert.deepEqual(stop(), ["diagnose"]);
  // The Q-Q answer flags its check: the flag and the open entry are drawn, the tab stop stays.
  list.render("check_qq", { ...ALL, flagged: ["check_qq"] });
  assert.deepEqual(stop(), ["diagnose"]);
  assert.equal(row("check_qq").dataset.flagged, "true");
  assert.equal(row("check_qq").getAttribute("aria-current"), "true");
  assert.equal(row("diagnose").getAttribute("aria-current"), null);
  // Opening an entry gives it the tab stop.
  list.render("diagnose", ALL);
  assert.deepEqual(stop(), ["diagnose"]);
  key("diagnose", "ArrowDown");
  list.render("diagnose", ALL);
  assert.deepEqual(stop(), ["check_qq"]);
});

test("an offer that hides the row holding the tab stop gives it back to the open entry", () => {
  const { list, key, stop } = fakeList();
  list.render("check_qq", ALL);
  key("check_qq", "ArrowDown");
  assert.deepEqual(stop(), ["check_pit"]);
  list.render("check_qq", { offered: ["check_qq"], flagged: [] });
  assert.deepEqual(stop(), ["check_qq"]);
  // An open check the model does not offer has no row: the Overview holds the stop.
  list.render("check_pit", { offered: ["check_qq"], flagged: [] });
  assert.deepEqual(stop(), ["assess"]);
});

test("the list's tab stop table: every state and event, each move landing where it says", () => {
  assert.deepEqual([...LIST_STOP.states], ["current", "roving"]);
  assert.deepEqual([...LIST_STOP.events].sort(), ["hide", "open", "step"]);
  // Held in each state, then sent each event.
  const into = { current: () => {}, roving: (key) => key("check_qq", "ArrowDown") };
  const send = {
    step: (list, key) => key("check_qq", "ArrowDown"),
    open: (list) => list.render("check_pit", ALL),
    hide: (list) => list.render("check_qq", { offered: ["check_qq"], flagged: [] }),
  };
  for (const state of LIST_STOP.states) {
    for (const event of LIST_STOP.events) {
      const { list, key, stop } = fakeList();
      list.render("check_qq", ALL);
      into[state](key);
      send[event](list, key);
      const target = LIST_STOP.next(state, event) ?? state;
      assert.equal(list.stop().state, target, `${state} --${event}--> ${target}`);
      assert.equal(stop().length, 1);
    }
  }
});
