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
    ["applied", ["Line 18 – 26", "Shift 3 – 5", "Opened model"]],
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
  assert.deepEqual(sections(node), [["applied", ["Line 30–45 in age", "Shift age", "Opened model"]]]);
  assert.match(node.innerHTML, /<code class="history-id">a1b2c3d<\/code>/);
  assert.doesNotMatch(node.innerHTML, /history-note-edit/);
});

// Entries as the state payload sends them. The backend's labels name the term
// ("collapse B10 + B11 in VehBrand") for the Undo popover; a waiting or
// applied change carries its parameters by label.
const BACKEND = [
  { kind: "edit", status: "edit", id: "a000001", time: at(13, 58), note: null,
    label: "shift DrivAge", term: "DrivAge", operation: "shift", n_points: 3, params: {},
    redo: false },
  { kind: "pending", status: "applied", id: "a000002", time: at(14, 0), note: null,
    label: "collapse <B1> + B11 in VehBrand", term: "VehBrand", operation: "collapse",
    params: { levels: ["<B1>", "B11"], group_label: null }, redo: false },
  { kind: "pending", status: "applied", id: "a000003", time: at(14, 1), note: null,
    label: "Line 18–26 in DrivAge", term: "DrivAge", operation: "shape",
    params: { lo: 18, hi: 26, degree: 1, join: "tangent" }, redo: false },
  { kind: "structural", status: "applied", id: "a000004", time: at(14, 2), note: null,
    label: "Refit · 2 changes", term: null, operation: "refit_pending", redo: false },
  { kind: "structural", status: "applied", id: "a000005", time: at(14, 2), note: null,
    label: "Hand edits carried over: DrivAge, VehAge", term: null, operation: "carry_edits",
    redo: false },
  { kind: "pending", status: "waiting", id: "a000006", time: at(14, 3), note: null,
    label: "ungroup B10 in VehBrand", term: "VehBrand", operation: "ungroup",
    params: { levels: ["B10"] }, redo: false },
  { kind: "pending", status: "waiting", id: "a000007", time: at(14, 4), note: null,
    label: "set reference of VehBrand to B2", term: "VehBrand", operation: "set_reference",
    params: { level: "B2" }, redo: false },
  { kind: "marker" },
];

test("each step reads as a message built from what it did, without its term", () => {
  const node = { innerHTML: "" };
  renderHistory(BACKEND, node);
  assert.deepEqual(sections(node), [
    ["waiting", ["Set reference B2", "Ungroup B10"]],
    ["applied", [
      "Hand edits carried over: DrivAge, VehAge",
      "Refit · 2 changes",
      "Line 18 – 26",
      "Collapse &lt;B1&gt; + B11",
      "Shift DrivAge",
      "Opened model",
    ]],
  ]);
  // The meta line is what names the term.
  assert.match(row(node, "a000002"), /class="history-meta">VehBrand · collapse</);
});

test("an edit names its action and the stretch of axis it changed, as the axis prints it", () => {
  const node = { innerHTML: "" };
  const edit = (id, operation, params, label) => ({
    kind: "edit", status: "edit", id, time: at(14, 0), note: null, label,
    term: "DrivAge", operation, n_points: 4, params, redo: false,
  });
  renderHistory([
    edit("c000001", "smooth", { strength: 1, lo: 62.68, hi: 85 }, "smooth DrivAge"),
    edit("c000002", "isotonic", { direction: "increasing", lo: 18, hi: 30 }, "isotonic DrivAge"),
    edit("c000003", "isotonic", { direction: "decreasing", lo: 18, hi: 30 }, "isotonic DrivAge"),
    edit("c000004", "shift", { delta: -0.05, lo: "B10", hi: "B10" }, "shift VehBrand"),
    edit("c000005", "linear_interpolate", { strength: 0.5, lo: 0.0177, hi: 2.5 }, "linear interpolate DrivAge"),
    edit("c000006", "set_values", { lo: 40.3, hi: 51.5 }, "set values DrivAge"),
    // A handle move, or an edit from before ranges were kept, keeps its label.
    edit("c000007", "control_point", { handle_index: 2, log_effect: 0.1, x: 30 }, "control point DrivAge"),
    { kind: "marker" },
  ], node);
  assert.deepEqual(sections(node), [["applied", [
    "Control point DrivAge",
    "Set values 40.3 – 51.5",
    "Straighten 0.0177 – 2.5",
    "Decrease B10",
    "Make decreasing 18 – 30",
    "Make increasing 18 – 30",
    "Smooth 62.7 – 85",
    "Opened model",
  ]]]);
  assert.match(row(node, "c000001"), /class="history-meta">DrivAge · edit</);
});

test("a change refitted at once reads the same as a staged one", () => {
  // Refit after every change lists the change once, as its step, without params.
  const node = { innerHTML: "" };
  renderHistory([
    { kind: "structural", status: "applied", id: "b000001", label: "collapse B10 + B11 in VehBrand",
      term: "VehBrand", operation: "collapse_levels", redo: false },
    { kind: "structural", status: "applied", id: "b000002", label: "ungroup B10, B11 in VehBrand",
      term: "VehBrand", operation: "ungroup_levels", redo: false },
    { kind: "structural", status: "applied", id: "b000003", label: "set reference of VehBrand to B2",
      term: "VehBrand", operation: "set_reference", redo: false },
    { kind: "structural", status: "applied", id: "b000004", label: "Cubic 18.5–26 in DrivAge",
      term: "DrivAge", operation: "shape_range", redo: false },
    { kind: "structural", status: "applied", id: "b000005", label: "revert to original model",
      term: null, operation: "revert_to_original", redo: false },
    { kind: "marker" },
  ], node);
  assert.deepEqual(sections(node), [["applied", [
    "Revert to original model",
    "Cubic 18.5 – 26",
    "Set reference B2",
    "Ungroup B10, B11",
    "Collapse B10 + B11",
    "Opened model",
  ]]]);
});

test("the history ends at its root, the opened model, below every applied step", () => {
  const node = { innerHTML: "" };
  renderHistory(TIMELINE.slice(0, 4), node);
  const last = node.innerHTML.slice(node.innerHTML.lastIndexOf("<li"));
  assert.match(last, /^<li class="history-item root">/);
  assert.match(last, /class="history-label">Opened model</);
  // The state payload has no id or time for the session's start, so the root
  // shows none, and has no note to write.
  assert.doesNotMatch(last, /data-step-id|history-id|<time|history-note-edit|Undo takes this/);

  // With nothing applied yet, the root is the applied list on its own.
  renderHistory([TIMELINE[2], { kind: "marker" }], node);
  assert.deepEqual(sections(node), [
    ["waiting", ["Collapse B10 + B11"]],
    ["applied", ["Opened model"]],
  ]);
});

test("a timeline holding only the marker, or none at all, says nothing happened yet", () => {
  for (const timeline of [[{ kind: "marker" }], undefined]) {
    const node = { innerHTML: "" };
    renderHistory(timeline, node);
    assert.equal(node.innerHTML, '<div class="history-empty">Nothing yet.</div>');
  }
});
