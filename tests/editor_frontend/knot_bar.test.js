// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { AT_LEAST_ONE, tooManyKnots } from "../../src/superglm/editor/app/knots.js";
import {
  SHRINK_BODY,
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
  const option = (value) => Object.assign(make("option"), { value, disabled: false });
  const hand = Object.assign(option("explicit"), { disabled: true });
  const rule = make("select");
  rule.options = [
    ...["uniform", "quantile", "quantile_rows", "quantile_tempered"].map(option), hand,
  ];
  const kind = make("select");
  kind.options = ["ps", "bs", "cr", "ns", "cr_cardinal"].map(option);
  return {
    doc,
    root: make("div"), fewer: make("button"), count: make("output"), more: make("button"),
    rule, hand, alphaWrap: make("label"), alpha: make("input"), reset: make("button"),
    kind, shrink: make("button"),
  };
}

/** The rule choices on offer: shown, and not disabled. */
function offered(bar) {
  return bar.rule.options.filter((option) => !option.hidden && !option.disabled)
    .map((option) => option.value);
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
      resettable: false, kind: "ps", select: false, kinds: ["ps", "bs", "cr", "ns"],
      select_available: true, select_reason: null, ...knots,
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

const EVEN_ONLY = "'age_band' is a natural spline (kind=\"ns\"), whose penalty needs evenly spaced knots; "
  + 'change its count here, or declare it with kind="cr" or kind="ps" to place its knots freely.';

test("a term that takes evenly spaced knots only is offered even spacing alone", async () => {
  const bar = nodes();
  renderKnotBar(bar, orderedTerm({ positions: [1, 2, 3], count: 3 }), true);
  assert.deepEqual(offered(bar), ["uniform", "quantile", "quantile_rows", "quantile_tempered"]);

  // In force by a rule it can no longer take: that rule stays named, but cannot be chosen.
  let term = orderedTerm({
    positions: [1, 2, 3], count: 3, strategy: "quantile", even_only: EVEN_ONLY, resettable: true,
  });
  renderKnotBar(bar, term, true);
  assert.deepEqual(offered(bar), ["uniform"]);
  const quantile = bar.rule.options[1];
  assert.deepEqual([quantile.hidden, quantile.disabled], [false, true]);
  assert.equal(bar.rule.options[3].hidden, true);
  // The count re-places evenly, and Reset still works.
  assert.equal(bar.more.dataset.popoverBody, "4 knots, all re-placed by even spacing.");
  const changes = [];
  bindKnotBar(bar, {
    term: () => term,
    onChange: async (params) => { changes.push(params); },
    onRefuse: () => {},
    onSettled: () => {},
  });
  bar.more.emit("click");
  bar.reset.emit("click");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(changes, [{ count: 4, strategy: "uniform" }, { reset: true }]);
  term = orderedTerm({ positions: [1, 2, 3], count: 3, even_only: EVEN_ONLY });
  renderKnotBar(bar, term, true);
  assert.deepEqual(offered(bar), ["uniform"]);
  assert.equal(bar.rule.options[1].hidden, true);
});

/** The kind choices on offer: shown, and not disabled. */
function kindsOffered(bar) {
  return bar.kind.options.filter((option) => !option.hidden && !option.disabled)
    .map((option) => option.value);
}

test("Kind offers the four kinds and shows the waiting one; a cardinal spline is named, not offered", () => {
  const bar = nodes();
  renderKnotBar(bar, orderedTerm(), true);
  assert.equal(bar.kind.value, "ps");
  assert.deepEqual(kindsOffered(bar), ["ps", "bs", "cr", "ns"]);
  assert.equal(bar.kind.options[4].hidden, true);
  assert.equal(bar.kind.dataset.waiting, "false");

  // A waiting change shows its kind, in the waiting tint; Shrink keeps its own state.
  renderKnotBar(bar, orderedTerm({}, { basis: { kind: "cr", select: false } }), true);
  assert.equal(bar.kind.value, "cr");
  assert.equal(bar.kind.dataset.waiting, "true");
  assert.equal(bar.shrink.dataset.waiting, "false");

  // Declared in code as a cardinal spline: named in force, but it cannot be chosen.
  renderKnotBar(bar, orderedTerm({ kind: "cr_cardinal" }), true);
  assert.equal(bar.kind.value, "cr_cardinal");
  const cardinal = bar.kind.options[4];
  assert.deepEqual([cardinal.hidden, cardinal.disabled], [false, true]);
  assert.deepEqual(kindsOffered(bar), ["ps", "bs", "cr", "ns"]);
});

const NO_SHRINK = "A natural spline cannot take shrinkage. To shrink 'age_band', choose another "
  + "kind; to make it a natural spline, turn Shrink off first.";

test("Shrink is pressed while on, and says why it cannot change instead of staging", () => {
  const bar = nodes();
  renderKnotBar(bar, orderedTerm(), true);
  assert.equal(bar.shrink.getAttribute("aria-pressed"), "false");
  assert.equal(bar.shrink.getAttribute("aria-disabled"), "false");
  assert.equal(bar.shrink.dataset.popoverTitle, "Shrink");
  assert.equal(bar.shrink.dataset.popoverBody, SHRINK_BODY);

  renderKnotBar(bar, orderedTerm({}, { basis: { kind: "ps", select: true } }), true);
  assert.equal(bar.shrink.getAttribute("aria-pressed"), "true");
  assert.equal(bar.shrink.dataset.waiting, "true");

  const natural = orderedTerm({
    kind: "ns", select_available: false, select_reason: NO_SHRINK,
  });
  renderKnotBar(bar, natural, true);
  assert.equal(bar.shrink.getAttribute("aria-disabled"), "true");
  assert.equal(bar.shrink.dataset.popoverBody, NO_SHRINK);
});

test("Kind and Shrink each stage one basis change, and an unchanged kind stages nothing", async () => {
  const bar = nodes();
  let term = orderedTerm({ select: true });
  const changes = [];
  const basis = [];
  const refusals = [];
  let settled = 0;
  bindKnotBar(bar, {
    term: () => term,
    onChange: async (params) => { changes.push(params); },
    onBasis: async (params) => { basis.push(params); },
    onRefuse: (message) => refusals.push(message),
    onSettled: () => { settled += 1; },
  });
  renderKnotBar(bar, term, true);

  bar.kind.value = "cr";
  bar.kind.emit("change");
  bar.kind.value = "ps";
  bar.kind.emit("change");
  bar.shrink.emit("click");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(basis, [{ kind: "cr" }, { select: false }]);
  assert.deepEqual(changes, []);
  assert.equal(settled, 3);

  term = orderedTerm({ kind: "ns", select_available: false, select_reason: NO_SHRINK });
  renderKnotBar(bar, term, true);
  bar.shrink.emit("click");
  assert.deepEqual(refusals, [NO_SHRINK]);
  assert.equal(basis.length, 2);
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
  renderKnotStatus(status, { message: AT_LEAST_ONE });
  assert.equal(status.textContent, AT_LEAST_ONE);
  renderKnotStatus(status, { evenOnly: true });
  assert.equal(
    status.textContent,
    "Knots. This term takes evenly spaced knots only; change their count above the chart.",
  );
});
