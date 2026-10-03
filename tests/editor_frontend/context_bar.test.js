// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  renderContextBar,
  waitingLabel,
} from "../../src/superglm/editor/app/views/context_bar.js";

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
