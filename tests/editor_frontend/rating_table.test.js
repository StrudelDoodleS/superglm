// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  RATING_TABLE_FAILED,
  bindTermViewToggle,
  formatRatingCell,
  ratingTableModel,
  renderTermViewToggle,
} from "../../src/superglm/editor/app/views/rating_table.js";

test("cells read as the workbook's number formats show them", () => {
  assert.equal(formatRatingCell(1.0069302040629162, "0.000000"), "1.006930");
  assert.equal(formatRatingCell(12345.678, "#,##0.00"), "12,345.68");
  assert.equal(formatRatingCell(-0.012345678901234567, "0.000000000000"), "-0.012345678901");
  assert.equal(formatRatingCell(0.000123456789012345678, "0.00000000000000E+00"), "1.23456789012346E-04");
  assert.equal(formatRatingCell(4, null), "4");
  assert.equal(formatRatingCell("[18.0, 20.0)", null), "[18.0, 20.0)");
  assert.equal(formatRatingCell(null, "0.000000"), "");
});

test("an available block becomes a header and formatted rows", () => {
  const model = ratingTableModel({
    term: "band",
    available: true,
    reason: null,
    columns: ["band", "Relativity", "Weight"],
    rows: [["low", 1, 210.5], ["high", 1.25, 1530]],
    formats: [null, "0.000000", "#,##0.00"],
    note: null,
    model_revision: 3,
  });

  assert.equal(model.message, null);
  assert.deepEqual(model.header, ["band", "Relativity", "Weight"]);
  assert.deepEqual(model.body, [["low", "1.000000", "210.50"], ["high", "1.250000", "1,530.00"]]);
  assert.deepEqual(model.numeric, [false, true, true]);
});

test("a refused table shows its fixed reason", () => {
  const model = ratingTableModel({
    term: "x", available: false, reason: "No table.", columns: [], rows: [], formats: [],
    note: null, model_revision: 0,
  });
  assert.equal(model.message, "No table.");
  assert.equal(ratingTableModel({ ...model, available: false, reason: null }).message, RATING_TABLE_FAILED);
});

class FakeButton {
  constructor(view) {
    this.dataset = { termView: view };
    this.attributes = new Map();
    this.tabIndex = -1;
    this.classes = new Set();
    this.classList = { toggle: (name, on) => (on ? this.classes.add(name) : this.classes.delete(name)) };
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  closest() { return this; }
  focus() { this.focused = true; }
}

globalThis.Element = FakeButton;
globalThis.HTMLElement = FakeButton;
globalThis.HTMLButtonElement = FakeButton;

test("the switch marks one view and reports clicks and arrows", () => {
  const buttons = [new FakeButton("chart"), new FakeButton("table")];
  const listeners = new Map();
  const root = {
    querySelectorAll: () => buttons,
    querySelector: (selector) => {
      if (selector.includes('aria-checked="true"')) {
        return buttons.find((button) => button.getAttribute("aria-checked") === "true") ?? null;
      }
      return buttons.find((button) => selector.includes(`"${button.dataset.termView}"`)) ?? null;
    },
    contains: (node) => buttons.includes(node),
    addEventListener: (name, listener) => listeners.set(name, listener),
    removeEventListener: (name) => listeners.delete(name),
  };
  const views = [];
  bindTermViewToggle(root, { onChange: (view) => views.push(view) });

  renderTermViewToggle(root, "table");
  assert.equal(buttons[1].getAttribute("aria-checked"), "true");
  assert.equal(buttons[1].tabIndex, 0);
  assert.equal(buttons[0].getAttribute("aria-checked"), "false");

  listeners.get("click")({ target: buttons[0] });
  listeners.get("keydown")({ key: "ArrowRight", preventDefault() {} });
  assert.deepEqual(views, ["chart", "chart"]);
  assert.equal(buttons[0].focused, true);
});
