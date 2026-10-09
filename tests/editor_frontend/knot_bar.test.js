// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { AT_LEAST_ONE, tooManyKnots } from "../../src/superglm/editor/app/knots.js";
import {
  bindKnotBar,
  renderKnotBar,
  renderKnotChip,
  renderKnotStatus,
} from "../../src/superglm/editor/app/views/knot_bar.js";

class FakeNode {
  constructor(tagName = "span", doc = null) {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.hidden = false;
    this.className = "";
    this.children = [];
    this.text = "";
    this.value = "";
    this.attributes = new Map();
    this.listeners = new Map();
    this.ownerDocument = doc ?? { activeElement: null, createElement: (tag) => new FakeNode(tag) };
  }

  get textContent() {
    return this.children.length
      ? this.children.map((child) => (typeof child === "string" ? child : child.textContent)).join("")
      : this.text;
  }

  set textContent(value) {
    this.children = [];
    this.text = String(value);
  }

  replaceChildren(...nodes) {
    this.text = "";
    this.children = nodes;
  }

  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  removeEventListener(name) { this.listeners.delete(name); }
  emit(name) { this.listeners.get(name)?.({}); }
}

function nodes() {
  const doc = { activeElement: null, createElement: (tag) => new FakeNode(tag, doc) };
  const make = (tag) => new FakeNode(tag, doc);
  return {
    doc,
    root: make("div"), fewer: make("button"), count: make("output"), more: make("button"),
    rule: make("select"), hand: make("option"), alphaWrap: make("label"), alpha: make("input"),
    reset: make("button"),
  };
}

const LEVELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"];

function orderedTerm(knots = {}, pending = null) {
  return {
    kind: "ordered categorical",
    term_type: "ordered categorical",
    x: LEVELS.map((_, i) => i),
    levels: LEVELS,
    shape: { available: true, reason: null, ranges: [], support: null, specials: [] },
    knots: {
      available: true, reason: null, positions: [2.5], count: 1, strategy: "uniform",
      alpha: 0.2, from_editor: false, lo: 0, hi: 5, min_gap: 0.1, max_count: 5,
      resettable: false, ...knots,
    },
    pending,
  };
}

test("the controls show only in Knots mode, with the shown count, rule and alpha", () => {
  const bar = nodes();
  renderKnotBar(bar, orderedTerm(), false);
  assert.equal(bar.root.hidden, true);

  const tempered = orderedTerm({ strategy: "quantile_tempered", alpha: 0.4 });
  renderKnotBar(bar, tempered, true);
  assert.equal(bar.root.hidden, false);
  assert.equal(bar.count.textContent, "1");
  assert.equal(bar.rule.value, "quantile_tempered");
  assert.equal(bar.hand.hidden, true);
  assert.equal(bar.alphaWrap.hidden, false);
  assert.equal(bar.alpha.value, "0.4");

  // Positions placed by hand show as the disabled Hand choice, without alpha.
  const byHand = orderedTerm({}, {
    knots: { positions: [1.5, 3.2], count: 2, strategy: "explicit", alpha: 0.2 },
  });
  renderKnotBar(bar, byHand, true);
  assert.equal(bar.count.textContent, "2");
  assert.equal(bar.rule.value, "explicit");
  assert.equal(bar.hand.hidden, false);
  assert.equal(bar.alphaWrap.hidden, true);
});

test("the stepper's buttons stay hoverable at their limits and say why; Reset only when it would change", () => {
  const bar = nodes();
  renderKnotBar(bar, orderedTerm(), true);
  assert.equal(bar.fewer.getAttribute("aria-disabled"), "true");
  assert.equal(bar.fewer.dataset.popoverBody, AT_LEAST_ONE);
  assert.equal(bar.more.getAttribute("aria-disabled"), "false");
  assert.equal(bar.more.dataset.popoverBody, "2 knots, all re-placed by even spacing.");
  assert.equal(bar.reset.getAttribute("aria-disabled"), "true");

  renderKnotBar(bar, orderedTerm({ positions: [0.5, 1.5, 2.5, 3.5, 4.5], count: 5, resettable: true }), true);
  assert.equal(bar.more.getAttribute("aria-disabled"), "true");
  assert.equal(bar.more.dataset.popoverTitle, "One knot more");
  assert.equal(bar.more.dataset.popoverBody, tooManyKnots(5));
  assert.equal(bar.reset.getAttribute("aria-disabled"), "false");
});

test("each control stages one knot change, and one that cannot act says why instead", async () => {
  const bar = nodes();
  let term = orderedTerm({ positions: [1, 2, 3], count: 3, resettable: true });
  const changes = [];
  const refusals = [];
  let settled = 0;
  bindKnotBar(bar, {
    term: () => term,
    onChange: async (params) => { changes.push(params); },
    onRefuse: (message) => refusals.push(message),
    onSettled: () => { settled += 1; },
  });
  renderKnotBar(bar, term, true);

  bar.more.emit("click");
  bar.fewer.emit("click");
  bar.rule.value = "quantile_tempered";
  bar.rule.emit("change");
  bar.alpha.value = "1.7";
  bar.alpha.emit("change");
  bar.reset.emit("click");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(changes, [
    { count: 4, strategy: "uniform" },
    { count: 2, strategy: "uniform" },
    { count: 3, strategy: "quantile_tempered", alpha: 0.2 },
    { count: 3, strategy: "quantile_tempered", alpha: 1 },
    { reset: true },
  ]);
  assert.equal(bar.alpha.value, "1");
  assert.equal(settled, 5);

  term = orderedTerm({ positions: [0.5, 1.5, 2.5, 3.5, 4.5], count: 5 });
  renderKnotBar(bar, term, true);
  bar.more.emit("click");
  bar.reset.emit("click");
  assert.deepEqual(refusals, [tooManyKnots(5), "The knots are the ones declared in code."]);
  assert.equal(changes.length, 5);
});

test("the chip names the knots in every mode and takes the waiting tint", () => {
  const chip = new FakeNode();
  renderKnotChip(chip, orderedTerm());
  assert.deepEqual([chip.hidden, chip.textContent, chip.dataset.waiting], [false, "1 knot · even spacing", "false"]);
  renderKnotChip(chip, orderedTerm({}, {
    knots: { positions: [1, 2.5], count: 2, strategy: "explicit", alpha: 0.2 },
  }));
  assert.deepEqual([chip.textContent, chip.dataset.waiting], ["2 knots · placed by hand", "true"]);
  renderKnotChip(chip, { kind: "categorical", knots: null });
  assert.equal(chip.hidden, true);
});

test("in Knots mode the status line explains the gestures, or why the last did nothing", () => {
  const status = new FakeNode("div");
  renderKnotStatus(status, {});
  assert.equal(
    status.textContent,
    "Knots. Drag to move · click the axis to add · drag below the axis to remove · arrow keys nudge"
      + " the selected knot, Delete removes it",
  );
  assert.equal(status.children.find((child) => child.tagName === "KBD").textContent, "Delete");
  renderKnotStatus(status, { pendingCount: 1, message: AT_LEAST_ONE });
  assert.equal(status.children[0].className, "status-waiting");
  assert.equal(status.textContent, `1 change waiting for refit · ${AT_LEAST_ONE}`);
});
