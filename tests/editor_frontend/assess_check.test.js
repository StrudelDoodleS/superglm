// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { CHECK_REPORTS } from "../../src/superglm/editor/app/api/contracts.js";
import { countText } from "../../src/superglm/editor/app/reports.js";
import { fmt, fmtCount } from "../../src/superglm/editor/app/format.js";
import { createInitialEditorState } from "../../src/superglm/editor/app/state/store.js";
import { nearestMark } from "../../src/superglm/editor/app/chart/check_plots.js";
import {
  bindCheckHover,
  checkFace,
  compareSwitch,
  flaggedChecks,
  renderCheckView,
  splitLine
} from "../../src/superglm/editor/app/views/assess_check.js";
import { CHECK_FIXTURE as FIXTURE, withCurrentChecks } from "./check_evidence.js";
import { neutralCopy } from "./neutral_copy.js";

function snapshot() {
  return {
    model_revision: 0, selected_term: "x",
    terms: { x: { kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1], previous_y: null,
      levels: null, n_points: 1, controls: null, group_display: null, impact: {},
      shape: { available: true, reason: null, ranges: [], support: null, specials: [] } } },
    selection: { x: [0] }, undo_redo: { undo: null, redo: null }, timeline: [{ kind: "marker" }],
    in_force_is_original: true, checks: [...CHECK_REPORTS]
  };
}

/** The page's evidence with every check of `reports` current. */
function evidenceFrom(reports) {
  return withCurrentChecks(createInitialEditorState(snapshot()).request.evidence, reports);
}

test("each card face counts what left its band and names one figure", () => {
  const calm = FIXTURE.calm;
  const qq = calm.check_qq.models.edited;
  assert.deepEqual(checkFace("check_qq", calm.check_qq, "current"),
    { value: `${countText(qq.outside)} of ${countText(qq.n)} points outside the band`, note: "Validation, 200 rows", flagged: qq.outside > 0 });
  const pit = calm.check_pit.models.edited;
  assert.equal(checkFace("check_pit", calm.check_pit, "current").note,
    `Bins ${Math.min(...pit.counts)} to ${Math.max(...pit.counts)} against ${pit.band_lower} to ${pit.band_upper}`);
  assert.equal(checkFace("check_qq", FIXTURE.wrong.check_qq, "current").flagged, true);
});

test("the four faces read their counts, the overall ratio and the 99th percentile's share", () => {
  const wrong = FIXTURE.wrong;
  const qq = wrong.check_qq.models.edited;
  // The flag counts all n, not the points sent to draw.
  assert.ok(qq.index.length < qq.n);
  assert.deepEqual(checkFace("check_qq", wrong.check_qq, "current"),
    { value: `${countText(qq.outside)} of 1,000 points outside the band`, note: "Validation, 1,000 rows", flagged: true });
  const pit = wrong.check_pit.models.edited;
  assert.deepEqual(checkFace("check_pit", wrong.check_pit, "current").value, `${pit.outside} of 20 bins outside the band`);
  const ae = FIXTURE.calm.check_ae.models.edited;
  assert.deepEqual(checkFace("check_ae", FIXTURE.calm.check_ae, "current"),
    { value: `0 of ${ae.cells} groups outside their interval`, note: `Overall ${ae.overall.ratio.toFixed(2)}`, flagged: false });
  const q = wrong.check_calibration.models.edited.quantiles;
  const share = q.realised_exceedance[q.p.indexOf(0.99)];
  assert.deepEqual(checkFace("check_calibration", wrong.check_calibration, "current"), {
    value: `${wrong.check_calibration.models.edited.outside} of 11 quantiles outside the band`,
    note: `99th percentile exceeded by ${(100 * share).toFixed(1)}%`,
    flagged: true
  });
  // Edited by hand: the face reads the model as edited, not the one fitted beside it.
  const edited = FIXTURE.edited.check_ae.models.edited;
  assert.equal(checkFace("check_ae", FIXTURE.edited.check_ae, "current").note, `Overall ${edited.overall.ratio.toFixed(2)}`);
  const off = structuredClone(FIXTURE.calm.check_ae);
  off.models.edited.overall.ratio = null;
  assert.equal(checkFace("check_ae", off, "current").note, "Overall —");
});

test("a check with no answer for this model says why, with no flag", () => {
  const off = { ...FIXTURE.calm.check_ae, available: false, reason: "R", models: { edited: { available: false, reason: "R", flagged: false, outside: 0, cells: 0 }, original: null } };
  assert.deepEqual(checkFace("check_ae", off, "current"), { value: "—", note: "R", flagged: false });
  assert.deepEqual(checkFace("check_ae", null, "updating"), { value: "—", note: "Loading check...", flagged: false });
  assert.deepEqual(checkFace("check_ae", null, "error"), { value: "—", note: "Check unavailable.", flagged: false });
  // Unavailable for this model alone: the model's own reason, and a flag it carries does not show.
  const body = { ...FIXTURE.wrong.check_qq, models: { edited: { available: false, reason: "Only one row.", flagged: true, outside: 0, cells: 0 }, original: null } };
  assert.deepEqual(checkFace("check_qq", body, "current"), { value: "—", note: "Only one row.", flagged: false });
});

test("every face is in neutral words", () => {
  for (const [name, reports] of Object.entries(FIXTURE)) {
    for (const report of CHECK_REPORTS) {
      const face = checkFace(report, reports[report], "current");
      assert.deepEqual(neutralCopy(`${face.value} ${face.note}`), [], `${name} ${report}`);
    }
  }
});

test("the flagged checks are the offered ones whose model as edited left its band", () => {
  const wrong = evidenceFrom(FIXTURE.wrong);
  assert.deepEqual(flaggedChecks(wrong, CHECK_REPORTS), CHECK_REPORTS.filter((name) => FIXTURE.wrong[name].models.edited.flagged));
  assert.deepEqual(flaggedChecks(wrong, ["check_ae", "check_qq"]), ["check_qq"]);
  assert.deepEqual(flaggedChecks(evidenceFrom(FIXTURE.calm), CHECK_REPORTS), []);
  assert.deepEqual(flaggedChecks({ ...wrong, check_qq: { ...wrong.check_qq, payload: null } }, ["check_qq"]), []);
});

/** The snapshot of a model edited by hand on `x`. */
function editedSnapshot() {
  const held = snapshot();
  return { ...held, terms: { x: { ...held.terms.x, edited: true } } };
}

test("Compare with the fit is live only when an offered check compares an edit", () => {
  const calm = compareSwitch(evidenceFrom(FIXTURE.calm), CHECK_REPORTS, true, snapshot());
  assert.deepEqual(calm, { live: false, checked: false, body: "The model is as fitted; there is no edit to compare." });
  const edited = evidenceFrom(FIXTURE.edited);
  assert.deepEqual(compareSwitch(edited, CHECK_REPORTS, false, editedSnapshot()), {
    live: true, checked: false,
    body: "Draws the model as fitted in grey beside the model as edited in blue, on every check. It changes only what is drawn."
  });
  assert.equal(compareSwitch(edited, CHECK_REPORTS, true, editedSnapshot()).checked, true);
  // A model a structural step replaced differs from the fit with no hand edit.
  assert.equal(compareSwitch(edited, CHECK_REPORTS, true, { ...snapshot(), in_force_is_original: false }).live, true);
  // A model offering no checks has nothing to compare, whatever the evidence holds.
  assert.equal(compareSwitch(edited, [], true, editedSnapshot()).live, false);
  assert.deepEqual(neutralCopy(compareSwitch(edited, CHECK_REPORTS, true, editedSnapshot()).body), []);
});

test("after Undo, Compare is off while the checks still hold the edited model's payloads", () => {
  // The model in force is the fit again; the evidence kept from the edit still draws the fit.
  const kept = evidenceFrom(FIXTURE.edited);
  assert.deepEqual(compareSwitch(kept, CHECK_REPORTS, true, snapshot()),
    { live: false, checked: false, body: "The model is as fitted; there is no edit to compare." });
  assert.equal(compareSwitch(kept, CHECK_REPORTS, true, null).live, false);
});

const CHAR_PX = 7;
const LINE_PX = 16;
const ASCENT_PX = 12;
/** A text node's lines: its tspans, or its own text. */
const textLines = (node) => (node.children.length ? node.children.map((child) => child.textContent) : [node.text]);

// The large view and its hover, on a minimal element tree that records every attribute, text and
// markup write, so a test can say which nodes a gesture touched.
class FakeNode {
  constructor(tag, log, attrs = {}) {
    this.tag = tag; this.log = log; this.attrs = new Map(Object.entries(attrs));
    this.children = []; this.parent = null; this.text = ""; this.html = ""; this.listeners = new Map();
    this.dataset = {}; this.rect = { left: 0, top: 0, width: 0, height: 0 };
    this.scrollTop = 0; this.scrollLeft = 0; this.clientWidth = 0; this.clientHeight = 0;
    this.ownerDocument = { createElementNS: (_ns, name) => new FakeNode(name, log) };
  }
  write(what) { this.log.push({ node: this, what }); }
  setAttribute(name, value) { this.write(name); this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  hasAttribute(name) { return this.attrs.has(name); }
  removeAttribute(name) { this.write(name); this.attrs.delete(name); }
  toggleAttribute(name, force) {
    const on = force ?? !this.attrs.has(name);
    if (on) this.setAttribute(name, ""); else this.removeAttribute(name);
    return on;
  }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(value) { this.toggleAttribute("hidden", Boolean(value)); }
  get textContent() { return this.text + this.children.map((child) => child.textContent).join(""); }
  set textContent(value) { this.write("text"); this.text = String(value); this.children = []; }
  get innerHTML() { return this.html; }
  set innerHTML(markup) {
    this.write("html"); this.html = markup; this.text = ""; this.children = [];
    const size = /<svg class="check-plot" viewBox="[^"]*" width="([\d.]+)" height="([\d.]+)"/.exec(markup);
    if (!size) return;
    const svg = new FakeNode("svg", this.log, { class: "check-plot" });
    svg.rect = { left: this.rect.left - this.scrollLeft, top: this.rect.top - this.scrollTop, width: Number(size[1]), height: Number(size[2]) };
    const readout = new FakeNode("g", this.log, { class: "check-readout", hidden: "" });
    svg.adopt(readout); this.adopt(svg);
  }
  adopt(child) { child.parent = this; this.children.push(child); }
  append(...nodes) { this.write("children"); for (const node of nodes) this.adopt(node); }
  matches(selector) {
    if (selector.startsWith(".")) return (this.attrs.get("class") ?? "").split(/\s+/).includes(selector.slice(1));
    return this.tag === selector;
  }
  closest(selector) {
    for (let node = this; node; node = node.parent) if (node.matches(selector)) return node;
    return null;
  }
  querySelector(selector) {
    for (const child of this.children) {
      if (child.matches(selector)) return child;
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }
  getBoundingClientRect() { return this.rect; }
  // Text is 7 px a character, its lines 16 px apart with the first baseline 12 px down.
  getComputedTextLength() { return this.textContent.length * CHAR_PX; }
  getBBox() {
    const lines = textLines(this);
    return { x: 0, y: -ASCENT_PX, width: Math.max(...lines.map((line) => line.length)) * CHAR_PX, height: lines.length * LINE_PX };
  }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter((l) => l !== listener)); }
  dispatch(type, event = {}) {
    const full = { preventDefault() { this.defaultPrevented = true; }, ...event };
    for (const listener of this.listeners.get(type) ?? []) listener(full);
    return full;
  }
}

function fakeFrame({ width = 820, height = 520 } = {}) {
  const log = [];
  const node = (tag) => new FakeNode(tag, log);
  const nodes = { title: node("h2"), split: node("span"), legend: node("div"), frame: node("div") };
  nodes.frame.clientWidth = width; nodes.frame.clientHeight = height;
  nodes.frame.rect = { left: 0, top: 0, width, height };
  const frames = [];
  globalThis.requestAnimationFrame = (callback) => frames.push(callback);
  globalThis.cancelAnimationFrame = () => {};
  const readout = () => nodes.frame.querySelector(".check-readout");
  return {
    nodes,
    recordWrites() { const from = log.length; return () => log.slice(from); },
    pointerMove(x, y) { nodes.frame.dispatch("pointermove", { clientX: x, clientY: y }); },
    pointerLeave() { nodes.frame.dispatch("pointerleave"); },
    key(key) { return nodes.frame.dispatch("keydown", { key }); },
    flushFrame() { for (const callback of frames.splice(0)) callback(0); },
    pendingFrames: () => frames.length,
    readout,
    readoutShown: () => !readout().hidden,
    readoutText: () => readout().querySelector("text")?.textContent ?? "",
  };
}

const VIEW = { compare: false, width: 820, height: 520 };

test("hover moves only the readout", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_qq", FIXTURE.calm.check_qq, VIEW);
  const writes = dom.recordWrites();
  const hover = bindCheckHover(dom.nodes.frame, () => scene);
  for (const x of [100, 300, 500]) dom.pointerMove(x, 200), dom.flushFrame();
  assert.ok(writes().length > 0);
  assert.ok(writes().every((write) => write.node.closest(".check-readout")), "a hover wrote outside the readout");
  assert.ok(dom.readoutShown());
  assert.equal(dom.readoutText(), scene.marks[nearestMark(scene.marks, 500, 200)].text);
  // Many moves in one frame read out once, at the last of them.
  for (const x of [120, 340, 560]) dom.pointerMove(x, 260);
  assert.equal(dom.pendingFrames(), 1);
  dom.flushFrame();
  assert.equal(dom.readoutText(), scene.marks[nearestMark(scene.marks, 560, 260)].text);
  dom.pointerLeave();
  assert.ok(!dom.readoutShown());
  hover.destroy();
  dom.pointerMove(100, 100); dom.flushFrame();
  assert.ok(!dom.readoutShown());
});

test("hover finds the nearest point by x and y, so a lower A/E panel reads its own point", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, VIEW);
  bindCheckHover(dom.nodes.frame, () => scene);
  // A point whose nearest point by x alone, 3 px to its right, is in another panel.
  const target = scene.marks.findIndex((mark, i) => nearestMark(scene.marks, mark.x + 3) !== i
    && nearestMark(scene.marks, mark.x + 3, mark.y) === i);
  assert.ok(target >= 0, "the fixture has no point x alone would miss");
  assert.ok(scene.marks.some((mark) => mark.y !== scene.marks[target].y));
  const { x, y } = scene.marks[target];
  dom.pointerMove(x + 3, y); dom.flushFrame();
  assert.equal(dom.readoutText(), scene.marks[target].text);
});

test("hover reads through the frame's scroll", () => {
  const dom = fakeFrame({ height: 200 });
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, { ...VIEW, height: 200 });
  assert.ok(scene.height > 200, "A/E keeps its panels' height and grows past the frame");
  bindCheckHover(dom.nodes.frame, () => scene);
  // Scrolled 150 px down, the svg's top sits 150 px above the frame's.
  dom.nodes.frame.querySelector("svg").rect.top = -150;
  const lowest = scene.marks.reduce((best, mark) => (mark.y > best.y ? mark : best));
  dom.pointerMove(lowest.x, lowest.y - 150); dom.flushFrame();
  assert.equal(dom.readoutText(), lowest.text);
});

test("the arrow keys step through the points while the plot has focus", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_pit", FIXTURE.calm.check_pit, VIEW);
  assert.equal(dom.nodes.frame.getAttribute("tabindex"), "0");
  assert.equal(dom.nodes.frame.getAttribute("role"), "img");
  assert.equal(dom.nodes.frame.getAttribute("aria-label"), "PIT histogram");
  bindCheckHover(dom.nodes.frame, () => scene);
  assert.ok(dom.key("ArrowRight").defaultPrevented);
  assert.equal(dom.readoutText(), scene.marks[0].text);
  dom.key("ArrowRight");
  assert.equal(dom.readoutText(), scene.marks[1].text);
  dom.key("ArrowLeft"); dom.key("ArrowLeft");
  assert.equal(dom.readoutText(), scene.marks[0].text);
  dom.key("End");
  assert.equal(dom.readoutText(), scene.marks.at(-1).text);
  dom.key("Escape");
  assert.ok(!dom.readoutShown());
  assert.equal(dom.key("a").defaultPrevented, undefined);
});

test("the view is headed by the check's label, with its split and rows beside it", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_qq", FIXTURE.calm.check_qq, VIEW);
  assert.equal(dom.nodes.title.textContent, "Q-Q plot");
  assert.equal(dom.nodes.split.innerHTML, "Validation, 200 rows");
  const points = scene.layers.filter((l) => l.role === "edited" || l.role === "outside")
    .reduce((sum, l) => sum + (l.d.match(/M/g) ?? []).length, 0);
  assert.equal(points, FIXTURE.calm.check_qq.models.edited.index.length);
  assert.ok(dom.nodes.frame.innerHTML.startsWith('<svg class="check-plot"'));
  assert.equal(dom.nodes.legend.hidden, true);
  assert.deepEqual(splitLine(FIXTURE.wrong.check_qq, false), { text: "Validation, 1,000 rows", level: null, note: null });
  assert.deepEqual(splitLine({ ...FIXTURE.calm.check_qq, split: null }, false), { text: "", level: null, note: null });
});

test("a held-out split whose level was not simulated says its parameters are taken as exact", () => {
  const payload = structuredClone(FIXTURE.calm.check_pit);
  const note = "The band treats the fitted parameters as exact, because these rows could not be simulated.";
  Object.assign(payload.models.edited, { level_source: "known", n_draws: 0, level_note: note });
  assert.deepEqual(splitLine(payload, false), { text: "Validation, 200 rows", level: "parameters taken as exact", note });
  // With no fit beside it, compare draws the edit alone, and the line names no model.
  assert.deepEqual(splitLine(payload, true).level, "parameters taken as exact");
  const dom = fakeFrame();
  renderCheckView(dom.nodes, "check_pit", payload, VIEW);
  const html = dom.nodes.split.innerHTML;
  assert.ok(html.startsWith("Validation, 200 rows · <span"), html);
  assert.match(html, />parameters taken as exact<\/span>$/);
  assert.match(html, new RegExp(`data-popover-body="${note}"`));
  assert.match(html, /data-popover-title="[^"]+"/);
  assert.deepEqual(neutralCopy(html), []);
});

test("compare draws the fit and names both models; a check with no answer says why", () => {
  const dom = fakeFrame();
  const compared = renderCheckView(dom.nodes, "check_ae", FIXTURE.edited.check_ae, { ...VIEW, compare: true });
  assert.ok(compared.layers.some((l) => l.role === "original"));
  assert.equal(dom.nodes.legend.hidden, false);
  renderCheckView(dom.nodes, "check_ae", FIXTURE.edited.check_ae, VIEW);
  assert.equal(dom.nodes.legend.hidden, true);
  // On a fit with no edit there is no fit to draw beside it, compare or not.
  renderCheckView(dom.nodes, "check_ae", FIXTURE.calm.check_ae, { ...VIEW, compare: true });
  assert.equal(dom.nodes.legend.hidden, true);
  const off = { ...FIXTURE.calm.check_ae, available: false, reason: "Needs at least two rows.", models: { edited: { available: false, reason: "Needs at least two rows.", flagged: false, outside: 0, cells: 0 }, original: null } };
  assert.equal(renderCheckView(dom.nodes, "check_ae", off, VIEW), null);
  assert.equal(dom.nodes.frame.innerHTML, '<p class="check-reason">Needs at least two rows.</p>');
  assert.equal(dom.nodes.frame.getAttribute("role"), null);
  assert.equal(dom.nodes.frame.getAttribute("tabindex"), null);
  assert.equal(renderCheckView(dom.nodes, "check_ae", null, VIEW), null);
  assert.equal(dom.nodes.frame.innerHTML, "");
  assert.equal(dom.nodes.title.textContent, "Actual over expected");
});

test("compared, the split line names the one model whose band took its parameters as exact", () => {
  const note = "The band treats the fitted parameters as exact, because these rows could not be simulated.";
  /** The edited fixture's PIT check, with the models in `known` keeping the known level. */
  const fellBack = (...known) => {
    const payload = structuredClone(FIXTURE.edited.check_pit);
    for (const role of known) Object.assign(payload.models[role], { level_source: "known", n_draws: 0, level_note: note });
    return payload;
  };
  const level = (payload, compare) => splitLine(payload, compare).level;
  // The edit alone fell back: named once the fit is drawn beside it.
  assert.equal(level(fellBack("edited"), true), "parameters taken as exact (as edited)");
  assert.equal(level(fellBack("edited"), false), "parameters taken as exact");
  // The fit alone fell back: named while it is drawn, and nothing to say while it is not.
  assert.equal(level(fellBack("original"), true), "parameters taken as exact (as fitted)");
  assert.deepEqual(splitLine(fellBack("original"), true).note, note);
  assert.equal(level(fellBack("original"), false), null);
  // Both fell back: the line holds for both.
  assert.equal(level(fellBack("edited", "original"), true), "parameters taken as exact");
  assert.equal(level(fellBack(), true), null);
  const dom = fakeFrame();
  renderCheckView(dom.nodes, "check_pit", fellBack("original"), { ...VIEW, compare: true });
  const html = dom.nodes.split.innerHTML;
  assert.match(html, />parameters taken as exact \(as fitted\)<\/span>$/);
  assert.match(html, new RegExp(`data-popover-body="${note}"`));
});

test("compared, the marker gives each drawn model's reason, and only drawn models count", () => {
  const edited = "The band treats the fitted parameters as exact, because these rows could not be simulated.";
  const fitted = "The band treats the fitted parameters as exact, because the fit could not be simulated.";
  const payload = structuredClone(FIXTURE.edited.check_pit);
  Object.assign(payload.models.edited, { level_source: "known", n_draws: 0, level_note: edited });
  Object.assign(payload.models.original, { level_source: "known", n_draws: 0, level_note: fitted });
  // Different reasons: both, each named.
  assert.deepEqual(splitLine(payload, true),
    { text: "Validation, 200 rows", level: "parameters taken as exact", note: `As edited: ${edited} As fitted: ${fitted}` });
  // Without the fit drawn, only the edit's reason.
  assert.equal(splitLine(payload, false).note, edited);
  // The same reason: given once.
  payload.models.original.level_note = edited;
  assert.equal(splitLine(payload, true).note, edited);
  // An unavailable edit draws nothing, so the fit's fallback is not disclosed either.
  const off = structuredClone(payload);
  off.models.edited = { available: false, reason: "Needs at least two rows.", flagged: false, outside: 0, cells: 0 };
  assert.deepEqual(splitLine(off, true), { text: "Validation, 200 rows", level: null, note: null });
  const dom = fakeFrame();
  renderCheckView(dom.nodes, "check_pit", off, { ...VIEW, compare: true });
  assert.equal(dom.nodes.split.innerHTML, "Validation, 200 rows");
  assert.equal(dom.nodes.frame.innerHTML, '<p class="check-reason">Needs at least two rows.</p>');
});

/** Where the readout's text sits in the scene, by the fake's text metrics, and its lines. */
function readoutBox(dom) {
  const group = dom.readout();
  const [tx, ty] = group.getAttribute("transform").match(/-?[\d.]+/g).map(Number);
  const text = group.querySelector("text");
  const lines = textLines(text);
  const left = tx + Number(text.getAttribute("x"));
  const top = ty + Number(text.getAttribute("y")) - ASCENT_PX;
  for (const line of text.children) assert.equal(line.getAttribute("x"), text.getAttribute("x"));
  return { left, right: left + Math.max(...lines.map((line) => line.length)) * CHAR_PX, top, bottom: top + lines.length * LINE_PX, lines };
}

/** The wrong fit's A/E check with group `c` of `g` renamed `name`. */
function namedGroup(name) {
  const payload = structuredClone(FIXTURE.wrong.check_ae);
  const table = payload.models.edited.tables.find((t) => t.covariate === "g");
  table.levels = table.levels.map((level) => (level === "c" ? name : level));
  return payload;
}

test("a readout keeps to the side of its point with room, inside the frame", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_ae", namedGroup("a longer group name"), VIEW);
  bindCheckHover(dom.nodes.frame, () => scene);
  for (const index of [0, scene.marks.length - 1]) {
    const mark = scene.marks[index];
    dom.pointerMove(mark.x, mark.y); dom.flushFrame();
    const box = readoutBox(dom);
    assert.deepEqual(box.lines, [mark.text]);
    assert.ok(box.left >= 0 && box.right <= VIEW.width, `${box.left} to ${box.right}`);
    // Beside the point, not over it.
    assert.ok(box.left > mark.x || box.right < mark.x, `${mark.x}: ${box.left} to ${box.right}`);
  }
});

test("a readout too long for the frame breaks between its values and stays inside", () => {
  const name = "a group name of exactly forty characters";
  assert.equal(name.length, 40);
  const width = 400;
  const dom = fakeFrame({ width });
  const scene = renderCheckView(dom.nodes, "check_ae", namedGroup(name), { ...VIEW, width });
  bindCheckHover(dom.nodes.frame, () => scene);
  const mark = scene.marks.find((m) => m.text.includes(name));
  assert.ok(mark.text.length * CHAR_PX > width, "the readout fits on one line");
  dom.pointerMove(mark.x, mark.y); dom.flushFrame();
  const box = readoutBox(dom);
  assert.ok(box.lines.length > 1);
  assert.equal(box.lines.join(""), mark.text);
  assert.equal(dom.readoutText(), mark.text);
  assert.ok(box.left >= 0 && box.right <= width, `${box.left} to ${box.right}`);
  assert.ok(box.top >= 0 && box.bottom <= scene.height, `${box.top} to ${box.bottom}`);
});

test("a group name over 40 characters is shortened in the readout, which keeps every value and fits", () => {
  const name = "q".repeat(2100);
  const frame = { width: 600, height: 350 };
  const dom = fakeFrame(frame);
  const payload = namedGroup(name);
  const scene = renderCheckView(dom.nodes, "check_ae", payload, { ...VIEW, ...frame });
  bindCheckHover(dom.nodes.frame, () => scene);
  const table = payload.models.edited.tables.find((t) => t.covariate === "g");
  const i = table.levels.indexOf(name);
  const mark = scene.marks.find((m) => m.text.startsWith(`g ${"q".repeat(39)}…`));
  assert.ok(mark, "no readout names the group by its first 39 characters");
  dom.pointerMove(mark.x, mark.y); dom.flushFrame();
  const text = dom.readoutText();
  assert.equal(text, `g ${"q".repeat(39)}… · ${fmt(table.ratio[i])} · interval ${fmt(table.lower[i])}`
    + ` to ${fmt(table.upper[i])} · ${fmtCount(table.n[i])} rows`);
  const box = readoutBox(dom);
  assert.ok(box.left >= 0 && box.right <= frame.width && box.top >= 0 && box.bottom <= frame.height,
    `${box.left} to ${box.right}, ${box.top} to ${box.bottom}`);
  // The hover's accessible text and the panel's own label keep the whole name.
  assert.equal(dom.readout().querySelector("text").getAttribute("aria-label"), text.replace(`${"q".repeat(39)}…`, name));
  const first = structuredClone(payload);
  first.models.edited.tables.find((t) => t.covariate === "g").levels[0] = name;
  assert.ok(renderCheckView(fakeFrame(frame).nodes, "check_ae", first, { ...VIEW, ...frame }).labels.some((label) => label.text === name));
  // A name of 40 characters is kept whole.
  const forty = "r".repeat(40);
  assert.ok(renderCheckView(fakeFrame().nodes, "check_ae", namedGroup(forty), VIEW).marks.some((m) => m.text.startsWith(`g ${forty} · `)));
});

test("a readout stays inside the frame's visible part when it is scrolled", () => {
  const dom = fakeFrame({ height: 200 });
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, { ...VIEW, height: 200 });
  bindCheckHover(dom.nodes.frame, () => scene);
  // Scrolled 80 px down, the frame shows the scene from 80 to 280 px.
  dom.nodes.frame.querySelector("svg").rect.top = -80;
  // The highest point in view, too near the visible top for its readout to sit above it.
  const mark = scene.marks.filter((m) => m.y >= 80 && m.y < 95).sort((a, b) => a.y - b.y)[0];
  assert.ok(mark, "no point sits just inside the visible top");
  dom.pointerMove(mark.x, mark.y - 80); dom.flushFrame();
  assert.equal(dom.readoutText(), mark.text);
  const box = readoutBox(dom);
  assert.ok(box.top >= 80 && box.bottom <= 280, `${box.top} to ${box.bottom}`);
});

test("a readout is placed again on the same point after the frame scrolls", () => {
  const dom = fakeFrame({ height: 200 });
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, { ...VIEW, height: 200 });
  bindCheckHover(dom.nodes.frame, () => scene);
  const mark = scene.marks.filter((m) => m.y >= 80 && m.y < 95).sort((a, b) => a.y - b.y)[0];
  assert.ok(mark, "no point sits just inside the visible top");
  // Unscrolled, the readout sits above the point.
  dom.pointerMove(mark.x, mark.y); dom.flushFrame();
  assert.ok(readoutBox(dom).bottom <= mark.y);
  // Scrolled 80 px down, the same point is read out inside the part now in view.
  dom.nodes.frame.scrollTop = 80;
  dom.nodes.frame.querySelector("svg").rect.top = -80;
  dom.pointerMove(mark.x, mark.y - 80); dom.flushFrame();
  assert.equal(dom.readoutText(), mark.text);
  const box = readoutBox(dom);
  assert.ok(box.top >= 80 && box.bottom <= 280, `${box.top} to ${box.bottom}`);
});
