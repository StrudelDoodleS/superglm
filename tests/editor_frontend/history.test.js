// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { renderHistory } from "../../src/superglm/editor/app/history.js";

const at = (hours, minutes) => new Date(2026, 9, 3, hours, minutes).getTime() / 1000;

/** The pane's sections in order, each with its rows' labels, top to bottom. */
function sections(node) {
  return node.innerHTML.split("<section ").slice(1).map((part) => [
    part.match(/class="history-section (\w+)"/)[1],
    [...part.matchAll(/class="history-label">([^<]*)</g)].map((match) => match[1]),
  ]);
}

/** One row's markup, found by its step id. */
function row(node, id) {
  const marker = node.innerHTML.indexOf(`data-step-id="${id}"`);
  const start = node.innerHTML.lastIndexOf("<li", marker);
  return node.innerHTML.slice(start, node.innerHTML.indexOf("</li>", marker));
}

const TIMELINE = [
  { kind: "edit", status: "edit", id: "0a1b2c3", time: at(14, 1), note: null,
    label: "Shift 3 – 5", term: "curve", operation: "shift", redo: false },
  { kind: "structural", status: "applied", id: "1b2c3d4", time: at(14, 3),
    note: "Young-driver tail is noise", label: "Line 18 – 26", term: "DrivAge",
    operation: "shape_range", redo: false },
  { kind: "pending", status: "waiting", id: "2c3d4e5", time: at(14, 5), note: null,
    label: "Collapse B10 + B11", term: "VehBrand", operation: "collapse", redo: false },
  { kind: "marker" },
  { kind: "pending", status: "waiting", id: "3d4e5f6", time: at(14, 6), note: "<b>thin</b>",
    label: "Collapse B13 + B14", term: "VehBrand", operation: "collapse", redo: true },
];

test("waiting changes sit above the applied ones, newest first, and the undone ones below", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  assert.deepEqual(sections(node), [
    ["waiting", ["Collapse B10 + B11"]],
    ["applied", ["Line 18 – 26", "Shift 3 – 5"]],
    ["undone", ["Collapse B13 + B14"]],
  ]);
  assert.match(node.innerHTML, /Notes are saved with the exported Python model\./);
});

test("Undo takes the newest step, whichever section it sits in", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  assert.equal(node.innerHTML.match(/history-undo-chip/g).length, 1);
  assert.match(row(node, "2c3d4e5"), /Undo takes this/);

  // An edit made after the waiting change is what Undo takes next.
  const editedLast = [
    ...TIMELINE.slice(0, 3),
    { ...TIMELINE[0], id: "4e5f6a7", time: at(14, 7), label: "Smooth 62.7 – 85" },
    { kind: "marker" },
  ];
  renderHistory(editedLast, node);
  assert.match(row(node, "4e5f6a7"), /Undo takes this/);
  assert.doesNotMatch(row(node, "2c3d4e5"), /Undo takes this/);
});

test("each step shows its id, time, term and kind, its note escaped, and a pencil", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE, node);
  const waiting = row(node, "2c3d4e5");
  assert.match(waiting, /class="history-item waiting"/);
  assert.match(waiting, /<code class="history-id">2c3d4e5<\/code>/);
  assert.ok(waiting.includes(
    `<time datetime="${new Date(at(14, 5) * 1000).toISOString()}">14:05</time>`,
  ));
  assert.match(waiting, /class="history-meta">VehBrand · collapse</);
  assert.match(waiting, /aria-label="Add a note"/);

  const applied = row(node, "1b2c3d4");
  assert.match(applied, /class="history-meta">DrivAge · shape</);
  assert.match(applied, /aria-label="Edit note"/);
  assert.match(applied, /<span>Young-driver tail is noise<\/span>/);
  assert.match(row(node, "0a1b2c3"), /class="history-meta">curve · edit</);

  const undone = row(node, "3d4e5f6");
  assert.match(undone, /class="history-item waiting redo"/);
  assert.match(undone, /&lt;b&gt;thin&lt;\/b&gt;/);
});

test("a timeline from before step ids still lists its edits and steps, with no pencil", () => {
  const node = { innerHTML: "" };
  renderHistory([
    { kind: "edit", label: "shift age", hash: "a1b2c3d", n_points: 3, redo: false },
    { kind: "structural", label: "Line 30–45 in age", redo: false },
    { kind: "marker" },
  ], node);
  assert.deepEqual(sections(node), [["applied", ["Line 30–45 in age", "shift age"]]]);
  assert.match(node.innerHTML, /<code class="history-id">a1b2c3d<\/code>/);
  assert.doesNotMatch(node.innerHTML, /history-note-edit/);
});

test("a timeline holding only the marker, or none at all, says nothing happened yet", () => {
  for (const timeline of [[{ kind: "marker" }], undefined]) {
    const node = { innerHTML: "" };
    renderHistory(timeline, node);
    assert.equal(node.innerHTML, '<div class="history-empty">Nothing yet.</div>');
  }
});
