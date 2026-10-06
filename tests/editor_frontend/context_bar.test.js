// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  renderContextBar,
  renderNewLevelsControl,
  waitingLabel,
} from "../../src/superglm/editor/app/views/context_bar.js";
import { HELP_SECTIONS, STRUCTURE_HELP } from "../../src/superglm/editor/app/views/help_content.js";

class FakeNode {
  constructor(tagName = "span") {
    this.tagName = tagName.toUpperCase();
    this.dataset = {};
    this.hidden = false;
    this.className = "";
    this.children = [];
    this.text = "";
    this.ownerDocument = { createElement: (tag) => new FakeNode(tag) };
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
}

const TERM = {
  kind: "categorical",
  term_type: "categorical",
  effective_df: 4,
  n_points: 10,
  reference: null,
  impact: { weighted_mean_relativity: 1.2, selected_weight_share: 0.25 },
};

function render(context) {
  const nodes = {
    nameNode: new FakeNode(),
    kindNode: new FakeNode(),
    edfNode: new FakeNode(),
    referenceNode: new FakeNode(),
    statusNode: new FakeNode(),
  };
  renderContextBar(nodes, { name: "territory", term: TERM, ...context });
  return nodes.statusNode;
}

test("the EDF chip gives three significant figures, as the feature list and the inspector do", () => {
  const chips = [10, 5, 11.3, 4.2137].map((effective_df) => {
    const nodes = {
      kindNode: new FakeNode(),
      edfNode: new FakeNode(),
      referenceNode: new FakeNode(),
      statusNode: new FakeNode(),
    };
    renderContextBar(nodes, { name: "territory", term: { ...TERM, effective_df }, selectionSize: 0 });
    return nodes.edfNode.textContent;
  });
  assert.deepEqual(chips, ["EDF 10.0", "EDF 5.00", "EDF 11.3", "EDF 4.21"]);
});

test("with nothing waiting the status line is the selection sentence", () => {
  const status = render({ selectionSize: 2 });
  assert.equal(
    status.textContent,
    "2 of 10 selected · average edit relativity 1.2x · selected exposure 25%",
  );
  assert.equal(status.dataset.term, "territory");
});

test("changes waiting lead the status line, then the last-refit note or the selection", () => {
  const idle = render({ selectionSize: 0, pendingCount: 2 });
  assert.equal(idle.children[0].className, "status-waiting");
  assert.equal(idle.children[0].textContent, "2 changes waiting for refit");
  assert.equal(
    idle.textContent,
    "2 changes waiting for refit · the curve and metrics are from the last refit",
  );

  const selecting = render({ selectionSize: 3, pendingCount: 1 });
  assert.equal(
    selecting.textContent,
    "1 change waiting for refit · 3 of 10 selected · average edit relativity 1.2x · selected exposure 25%",
  );
  assert.equal(selecting.dataset.term, "territory");
  assert.equal(waitingLabel(1), "1 change waiting for refit");
});

test("a Shift-click span reads as its range on the status line", () => {
  const range = { lo: "T02", hi: "T08" };
  assert.equal(
    render({ selectionSize: 7, range }).textContent,
    "Range T02 – T08 · 7 of 10 points · selected exposure 25%",
  );
  assert.equal(
    render({ selectionSize: 7, range, pendingCount: 1 }).textContent,
    "1 change waiting for refit · Range T02 – T08 · 7 of 10 points · selected exposure 25%",
  );
  // Any other selection keeps the selection sentence.
  assert.equal(
    render({ selectionSize: 7, range: null }).textContent,
    "7 of 10 selected · average edit relativity 1.2x · selected exposure 25%",
  );
});

/** Plain nodes, as the reference chip needs nothing more. */
function nodes() {
  const node = () => ({ textContent: "", hidden: false, dataset: {} });
  return { kindNode: node(), edfNode: node(), referenceNode: node(), statusNode: node() };
}

test("a waiting reference change shows in the reference chip", () => {
  const n = nodes();
  renderContextBar(n, {
    name: "VehBrand",
    term: {
      kind: "categorical", term_type: "categorical", effective_df: 10, n_points: 11,
      reference: { level: "B2", policy: "kept" },
      pending: { groups: null, ranges: [], reference: "B10 + B11" },
    },
    selectionSize: 0,
  });
  assert.equal(n.referenceNode.textContent, "reference B10 + B11 · waiting");
  assert.equal(n.referenceNode.dataset.waiting, "true");
});

test("without a waiting reference the chip shows the fitted one", () => {
  const n = nodes();
  renderContextBar(n, {
    name: "VehBrand",
    term: { kind: "categorical", effective_df: 10, n_points: 11,
      reference: { level: "B2", policy: "kept" } },
    selectionSize: 0,
  });
  assert.equal(n.referenceNode.textContent, "reference B2 · kept");
  assert.equal(n.referenceNode.dataset.waiting, "false");
});

/** The "New levels →" label and its select, with the DOM calls the control makes. */
function newLevelsNodes() {
  const doc = { createElement: (tag) => ({ tagName: tag.toUpperCase(), value: "", textContent: "" }) };
  const select = {
    ownerDocument: doc,
    options: [],
    value: "",
    disabled: false,
    dataset: {},
    replaceChildren(...options) {
      this.options = options;
      this.rebuilt = (this.rebuilt ?? 0) + 1;
    },
  };
  return { wrap: { hidden: true, dataset: {} }, select };
}

const UNSEEN = {
  policy: "Other",
  choices: [
    { value: "error", label: "Refuse" },
    { value: "base", label: "Reference" },
    { value: "Other", label: "Other" },
  ],
  reason: null,
};

test("New levels lists Refuse, Reference and each group, and shows the choice in force", () => {
  const n = newLevelsNodes();
  renderNewLevelsControl(n, { ...TERM, unseen: UNSEEN });
  assert.equal(n.wrap.hidden, false);
  assert.deepEqual(
    n.select.options.map((option) => [option.value, option.textContent]),
    [["error", "Refuse"], ["base", "Reference"], ["Other", "Other"]],
  );
  assert.equal(n.select.value, "Other");
  assert.equal(n.select.disabled, false);
  // Its hover popover is its Help entry, which Help lists under Model structure.
  assert.equal(n.wrap.dataset.popoverTitle, STRUCTURE_HELP.new_levels.title);
  assert.equal(n.wrap.dataset.popoverBody, STRUCTURE_HELP.new_levels.body);
  assert.equal(STRUCTURE_HELP.new_levels.title, "New levels →");
  const structure = HELP_SECTIONS.find((section) => section.title === "Model structure");
  assert.ok(structure.keys.includes("new_levels"));
  // The same choices again keep the open list as it is; a new group rebuilds it.
  renderNewLevelsControl(n, { ...TERM, unseen: { ...UNSEEN, policy: "base" } });
  assert.equal(n.select.rebuilt, 1);
  assert.equal(n.select.value, "base");
  const grown = [...UNSEEN.choices, { value: "B1+B2", label: "B1+B2" }];
  renderNewLevelsControl(n, { ...TERM, unseen: { ...UNSEEN, choices: grown } });
  assert.equal(n.select.rebuilt, 2);
  assert.equal(n.select.options.length, 4);
});

test("New levels is absent on spline and ordered terms", () => {
  for (const term of [
    { ...TERM, kind: "numeric", term_type: "spline", unseen: null },
    { ...TERM, term_type: "ordered categorical", unseen: null },
    { ...TERM },
  ]) {
    const n = newLevelsNodes();
    n.wrap.hidden = false;
    renderNewLevelsControl(n, term);
    assert.equal(n.wrap.hidden, true, term.term_type);
  }
});

test("New levels is disabled, with the reason in its popover, while the choice cannot be made", () => {
  const n = newLevelsNodes();
  const reason = "Refit or undo the waiting changes to 'territory' before choosing where its new levels go.";
  renderNewLevelsControl(n, { ...TERM, unseen: { ...UNSEEN, reason } });
  assert.equal(n.wrap.hidden, false);
  assert.equal(n.select.disabled, true);
  assert.equal(n.wrap.dataset.popoverBody, reason);
  renderNewLevelsControl(n, { ...TERM, unseen: UNSEEN });
  assert.equal(n.select.disabled, false);
  assert.equal(n.wrap.dataset.popoverBody, STRUCTURE_HELP.new_levels.body);
});
