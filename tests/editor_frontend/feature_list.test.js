// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  bindFeatureList,
  filterFeatures,
  readFeatureListOpen,
  renderFeatureList,
  storeFeatureListOpen,
} from "../../src/superglm/editor/app/views/feature_list.js";

// A small DOM: enough tree, selectors and focus bookkeeping for the list.
class FakeDocument {
  constructor() {
    this.activeElement = null;
  }

  createElement(tagName) {
    return new FakeElement(tagName, this);
  }
}

class FakeElement {
  constructor(tagName, ownerDocument) {
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this.dataset = {};
    this.attributes = new Map();
    this.children = [];
    this.parentNode = null;
    this.listeners = new Map();
    this.textContent = "";
    this.className = "";
    this.tabIndex = -1;
    this.value = "";
    this.scrolls = [];
    this.isContentEditable = false;
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  append(...nodes) {
    for (const node of nodes) {
      node.parentNode = this;
      this.children.push(node);
    }
  }

  appendChild(node) {
    this.append(node);
    return node;
  }

  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }

  contains(node) {
    return node === this || this.children.some((child) => child.contains(node));
  }

  descendants() {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }

  matches(selector) {
    if (selector === "[data-term]") return this.dataset.term !== undefined;
    if (selector === '[data-term][tabindex="0"]') {
      return this.dataset.term !== undefined && this.tabIndex === 0;
    }
    if (selector === '[aria-current="true"]') return this.getAttribute("aria-current") === "true";
    throw new Error(`fake DOM cannot match ${selector}`);
  }

  querySelectorAll(selector) {
    return this.descendants().filter((node) => node.matches(selector));
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  closest(selector) {
    if (this.matches(selector)) return this;
    return this.parentNode ? this.parentNode.closest(selector) : null;
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  scrollIntoView(options) {
    this.scrolls.push(options);
  }

  addEventListener(name, listener) {
    const listeners = this.listeners.get(name) ?? new Set();
    listeners.add(listener);
    this.listeners.set(name, listeners);
  }

  removeEventListener(name, listener) {
    this.listeners.get(name)?.delete(listener);
  }

  // Bubbles from this node to the root, like the real event path.
  emit(name, properties = {}) {
    const event = {
      target: this,
      key: "",
      defaultPrevented: false,
      ...properties,
      preventDefault() {
        this.defaultPrevented = true;
      },
    };
    for (let node = this; node; node = node.parentNode) {
      for (const listener of node.listeners.get(name) ?? []) listener(event);
    }
    return event;
  }
}

globalThis.Element = FakeElement;
globalThis.HTMLElement = FakeElement;

const TERMS = {
  age: { kind: "spline", term_type: "spline", effective_df: 4.2137 },
  mileage: { kind: "spline", term_type: "spline", effective_df: 2.5 },
  region: { kind: "categorical", term_type: "categorical", effective_df: 3 },
  territory: { kind: "categorical", term_type: "categorical", effective_df: null },
};
const GROUPS = [["spline", ["age", "mileage"]], ["categorical", ["region", "territory"]]];

function fixture() {
  const doc = new FakeDocument();
  const nodes = {
    root: doc.createElement("nav"),
    search: doc.createElement("input"),
    rows: doc.createElement("div"),
    toggle: doc.createElement("button"),
    strip: doc.createElement("span"),
  };
  nodes.root.append(nodes.search, nodes.toggle, nodes.strip, nodes.rows);
  const calls = { selected: [], queries: [], toggles: 0 };
  const binding = bindFeatureList(nodes, {
    onSelect: (name) => calls.selected.push(name),
    onQuery: (query) => calls.queries.push(query),
    onToggle: () => { calls.toggles += 1; },
  });
  const render = (overrides = {}) => renderFeatureList(nodes, {
    groups: GROUPS,
    terms: TERMS,
    activeTerm: "mileage",
    query: "",
    open: true,
    ...overrides,
  });
  const rows = () => nodes.rows.querySelectorAll("[data-term]");
  return { doc, nodes, calls, binding, render, rows };
}

test("empty query keeps every feature", () => {
  const groups = [["Smooth", ["age", "mileage"]], ["Categorical", ["region", "territory"]]];
  assert.deepEqual(filterFeatures(groups, ""), groups);
  assert.deepEqual(filterFeatures(groups, "   "), groups);
});

test("query filters case-insensitively on both sides and drops empty groups", () => {
  const groups = [["Smooth", ["age", "Mileage"]], ["Categorical", ["region", "Territory"]]];
  assert.deepEqual(filterFeatures(groups, "TER"), [["Categorical", ["Territory"]]]);
  assert.deepEqual(filterFeatures(groups, "mile"), [["Smooth", ["Mileage"]]]);
  assert.deepEqual(filterFeatures(groups, "e"), [
    ["Smooth", ["age", "Mileage"]],
    ["Categorical", ["region", "Territory"]],
  ]);
  assert.deepEqual(filterFeatures(groups, "zzz"), []);
});

test("rows carry name, kind and EDF under group headings; the active row is current, the tab stop, and scrolled into view", () => {
  const { nodes, render, rows } = fixture();
  render();

  const sections = nodes.rows.children;
  assert.deepEqual(sections.map((section) => section.children[0].textContent), ["spline", "categorical"]);
  assert.deepEqual(
    rows().map((row) => row.children.map((part) => part.textContent)),
    [
      ["age", "spline", "EDF 4.21"],
      ["mileage", "spline", "EDF 2.5"],
      ["region", "categorical", "EDF 3"],
      ["territory", "categorical", "EDF —"],
    ],
  );
  assert.deepEqual(rows().map((row) => row.getAttribute("aria-current")), [null, "true", null, null]);
  assert.deepEqual(rows().map((row) => row.tabIndex), [-1, 0, -1, -1]);
  assert.deepEqual(rows()[1].scrolls, [{ block: "nearest" }]);
  assert.deepEqual(rows()[0].scrolls, []);
  assert.equal(nodes.strip.textContent, "mileage");
  assert.equal(nodes.root.dataset.open, "true");
  assert.equal(nodes.toggle.getAttribute("aria-expanded"), "true");
});

test("a query narrows the rows, moves the tab stop to the first match when the active feature is hidden, and says when nothing matches", () => {
  const { nodes, render, rows } = fixture();
  render({ query: "R" });
  assert.deepEqual(rows().map((row) => row.dataset.term), ["region", "territory"]);
  assert.deepEqual(rows().map((row) => row.tabIndex), [0, -1]);
  assert.deepEqual(rows().map((row) => row.getAttribute("aria-current")), [null, null]);

  render({ query: "age" });
  assert.deepEqual(rows().map((row) => [row.dataset.term, row.tabIndex]), [["age", -1], ["mileage", 0]]);

  render({ query: "nothing here" });
  assert.deepEqual(rows(), []);
  assert.deepEqual(nodes.rows.children.map((node) => node.textContent), ["No features match."]);

  render({ groups: [], terms: {}, activeTerm: "" });
  assert.deepEqual(nodes.rows.children, []);
});

test("arrow keys step focus through the visible rows, back to the search box from the top, and into the list from the search box", () => {
  const { doc, nodes, render, rows } = fixture();
  render();
  const [age, mileage, region, territory] = rows();

  mileage.focus();
  const down = mileage.emit("keydown", { key: "ArrowDown" });
  assert.equal(doc.activeElement, region);
  assert.equal(down.defaultPrevented, true);
  region.emit("keydown", { key: "End" });
  assert.equal(doc.activeElement, territory);
  territory.emit("keydown", { key: "ArrowDown" });
  assert.equal(doc.activeElement, territory);
  territory.emit("keydown", { key: "Home" });
  assert.equal(doc.activeElement, age);
  age.emit("keydown", { key: "ArrowUp" });
  assert.equal(doc.activeElement, nodes.search);

  const enter = nodes.search.emit("keydown", { key: "ArrowDown" });
  assert.equal(doc.activeElement, mileage);
  assert.equal(enter.defaultPrevented, true);

  const ignored = mileage.emit("keydown", { key: "ArrowRight" });
  assert.equal(ignored.defaultPrevented, false);
  assert.equal(doc.activeElement, mileage);
});

test("a click inside a row selects its feature; Enter in the search box opens the first match only with a query; Escape clears the query", () => {
  const { nodes, calls, render, rows } = fixture();
  render();

  rows()[2].children[0].emit("click");
  assert.deepEqual(calls.selected, ["region"]);
  nodes.rows.emit("click");
  assert.deepEqual(calls.selected, ["region"]);

  nodes.search.value = "";
  const emptyEnter = nodes.search.emit("keydown", { key: "Enter" });
  assert.deepEqual(calls.selected, ["region"]);
  assert.equal(emptyEnter.defaultPrevented, false);

  nodes.search.value = "ter";
  nodes.search.emit("input");
  assert.deepEqual(calls.queries, ["ter"]);
  render({ query: "ter" });
  const enter = nodes.search.emit("keydown", { key: "Enter" });
  assert.deepEqual(calls.selected, ["region", "territory"]);
  assert.equal(enter.defaultPrevented, true);

  const escape = nodes.search.emit("keydown", { key: "Escape" });
  assert.equal(nodes.search.value, "");
  assert.deepEqual(calls.queries, ["ter", ""]);
  assert.equal(escape.defaultPrevented, true);
  const idleEscape = nodes.search.emit("keydown", { key: "Escape" });
  assert.equal(idleEscape.defaultPrevented, false);
  assert.deepEqual(calls.queries, ["ter", ""]);
});

test("the collapsed strip names the active feature and the toggle reports each click", () => {
  const { nodes, calls, render } = fixture();
  render({ open: false, activeTerm: "region" });
  assert.equal(nodes.root.dataset.open, "false");
  assert.equal(nodes.toggle.getAttribute("aria-expanded"), "false");
  assert.equal(nodes.strip.textContent, "region");

  nodes.toggle.emit("click");
  nodes.toggle.emit("click");
  assert.equal(calls.toggles, 2);
});

test("the collapse preference is stored and read back, and storage that throws defaults to open", () => {
  const store = new Map();
  const storage = {
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => store.set(key, value),
  };
  assert.equal(readFeatureListOpen(storage), true);
  storeFeatureListOpen(false, storage);
  assert.deepEqual([...store.entries()], [["superglm.editor.featureList", "collapsed"]]);
  assert.equal(readFeatureListOpen(storage), false);
  storeFeatureListOpen(true, storage);
  assert.equal(readFeatureListOpen(storage), true);

  const blocked = {
    getItem() { throw new Error("storage disabled"); },
    setItem() { throw new Error("storage disabled"); },
  };
  assert.equal(readFeatureListOpen(blocked), true);
  assert.doesNotThrow(() => storeFeatureListOpen(false, blocked));
});

test("a focused row keeps focus across a re-render and loses it only when filtered out", () => {
  const { doc, nodes, render, rows } = fixture();
  render();
  rows()[2].focus();
  render({ activeTerm: "region" });
  assert.equal(doc.activeElement, rows()[2]);
  assert.equal(doc.activeElement.dataset.term, "region");

  render({ query: "age" });
  assert.equal(doc.activeElement.dataset.term, "region");
  assert.equal(nodes.rows.contains(doc.activeElement), false);
});

test("destroy detaches every listener", () => {
  const { nodes, calls, binding, render, rows } = fixture();
  render();
  binding.destroy();
  rows()[0].emit("click");
  nodes.search.value = "a";
  nodes.search.emit("input");
  nodes.toggle.emit("click");
  assert.deepEqual(calls, { selected: [], queries: [], toggles: 0 });
});
