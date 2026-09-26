// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { renderHistory } from "../../src/superglm/editor/app/history.js";

/** A history frame that records the one scroll the render asks of it. */
function frame() {
  const node = { innerHTML: "", scrolled: [] };
  node.querySelector = (selector) =>
    node.innerHTML.includes(`class="${selector.slice(1)}"`)
      ? { scrollIntoView: (options) => node.scrolled.push([selector, options]) }
      : null;
  return node;
}

const rowClasses = (node) =>
  [...node.innerHTML.matchAll(/<li class="([^"]+)"/g)].map((match) => match[1]);
const rowLabels = (node) =>
  [...node.innerHTML.matchAll(/class="history-label">([^<]*)</g)].map((match) => match[1]);
const rowMarks = (node) =>
  [...node.innerHTML.matchAll(/class="history-(hash|chip)"/g)].map((match) => match[1]);

test("the timeline renders in order with the marker, muted redo entries and step chips", () => {
  const node = frame();
  renderHistory(
    [
      { kind: "edit", label: "shift age", hash: "a1b2c3d", n_points: 3, params: { amount: 0.05 }, redo: false },
      { kind: "structural", label: "Line 30–45 in age", redo: false },
      { kind: "marker" },
      { kind: "edit", label: "smooth <age>", hash: "e4f5a6b", n_points: 12, params: {}, redo: true },
      { kind: "structural", label: "collapse T01 + T03 in territory", redo: true },
    ],
    node,
  );

  assert.deepEqual(rowClasses(node), [
    "history-item edit",
    "history-item structural",
    "history-now",
    "history-item edit redo",
    "history-item structural redo",
  ]);
  assert.deepEqual(rowLabels(node), [
    "shift age",
    "Line 30–45 in age",
    "smooth &lt;age&gt;",
    "collapse T01 + T03 in territory",
  ]);
  // Edits carry their hash; steps carry the chip instead.
  assert.deepEqual(rowMarks(node), ["hash", "chip", "hash", "chip"]);
  assert.match(node.innerHTML, /a1b2c3d/);
  assert.match(node.innerHTML, /3 points · amount=0\.05/);
  // The marker is kept in view each time the list is drawn.
  assert.deepEqual(node.scrolled, [[".history-now", { block: "nearest" }]]);
});

test("a timeline holding only the marker, or none at all, says nothing happened yet", () => {
  for (const timeline of [[{ kind: "marker" }], undefined]) {
    const node = frame();
    renderHistory(timeline, node);
    assert.equal(node.innerHTML, '<div class="history-empty">Nothing yet.</div>');
    assert.deepEqual(node.scrolled, []);
  }
});
