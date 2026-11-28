// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { CHECK_REPORTS } from "../../src/superglm/editor/app/api/contracts.js";
import { countText } from "../../src/superglm/editor/app/reports.js";
import { fmt, fmtCount } from "../../src/superglm/editor/app/format.js";
import { createInitialEditorState } from "../../src/superglm/editor/app/state/store.js";
import { checkScene, nearestMark } from "../../src/superglm/editor/app/chart/check_plots.js";
import {
  CHECK_READOUT,
  COMPARE_CONTROL,
  bindCheckHover,
  checkFace,
  compareSwitch,
  flaggedChecks,
  redrawCheckPlot,
  renderCheckView,
  splitLine
} from "../../src/superglm/editor/app/views/assess_check.js";
import { CHECK_FIXTURE as FIXTURE, withCurrentChecks } from "./check_evidence.js";
import { ASCENT_PX, CHAR_PX, LINE_PX, fakeFrame, textLines } from "./check_frame.js";
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

const COMPARES = "Draws the model as fitted in grey beside the model as edited in blue, on every check. "
  + "It changes only what is drawn.";

test("Compare with the fit is ready only when an offered check draws the fit beside an edit", () => {
  const calm = compareSwitch(evidenceFrom(FIXTURE.calm), CHECK_REPORTS, true, snapshot());
  assert.deepEqual(calm,
    { state: "asFitted", disabled: true, checked: false, body: "The model is as fitted; there is no edit to compare." });
  const edited = evidenceFrom(FIXTURE.edited);
  assert.deepEqual(compareSwitch(edited, CHECK_REPORTS, false, editedSnapshot()),
    { state: "ready", disabled: false, checked: false, body: COMPARES });
  assert.equal(compareSwitch(edited, CHECK_REPORTS, true, editedSnapshot()).checked, true);
  // A model a structural step replaced differs from the fit with no hand edit.
  assert.equal(compareSwitch(edited, CHECK_REPORTS, true, { ...snapshot(), in_force_is_original: false }).state, "ready");
  // A model offering no checks hides the switch, whatever the evidence holds.
  assert.equal(compareSwitch(edited, [], true, editedSnapshot()).state, "hidden");
  for (const evidence of [edited, evidenceFrom(FIXTURE.calm)]) {
    assert.deepEqual(neutralCopy(compareSwitch(evidence, CHECK_REPORTS, true, editedSnapshot()).body), []);
  }
});

test("after Undo, Compare is off while the checks still hold the edited model's payloads", () => {
  // The model in force is the fit again; the evidence kept from the edit still draws the fit.
  const kept = evidenceFrom(FIXTURE.edited);
  assert.deepEqual(compareSwitch(kept, CHECK_REPORTS, true, snapshot()),
    { state: "asFitted", disabled: true, checked: false, body: "The model is as fitted; there is no edit to compare." });
  assert.equal(compareSwitch(kept, CHECK_REPORTS, true, null).state, "asFitted");
});

test("right after an edit, Compare waits for the edited model's check instead of saying there is no edit", () => {
  // The check holds its answer for the model as fitted while it is asked again for the edit.
  const asked = Object.fromEntries(Object.entries(evidenceFrom(FIXTURE.calm)).map(([name, evidence]) =>
    [name, CHECK_REPORTS.includes(name) ? { ...evidence, status: "updating" } : evidence]));
  const waiting = compareSwitch(asked, ["check_qq"], true, editedSnapshot());
  assert.equal(waiting.state, "waitingForComparison");
  assert.equal(waiting.disabled, true);
  assert.equal(waiting.checked, false);
  assert.notEqual(waiting.body, "The model is as fitted; there is no edit to compare.");
  // A check never answered yet waits too; one lapsed and not yet asked again as well.
  const empty = createInitialEditorState(snapshot()).request.evidence;
  assert.equal(compareSwitch(empty, ["check_qq"], true, editedSnapshot()).state, "waitingForComparison");
  const lapsed = { ...asked, check_qq: { ...asked.check_qq, status: "stale", error: null } };
  assert.equal(compareSwitch(lapsed, ["check_qq"], true, editedSnapshot()).state, "waitingForComparison");
});

test("Compare says there is no comparison when the edit's check failed or the fit has no answer", () => {
  const calm = evidenceFrom(FIXTURE.calm);
  const failed = { ...calm, check_qq: { ...calm.check_qq, status: "stale", error: "offline" } };
  const unavailable = compareSwitch(failed, ["check_qq"], true, editedSnapshot());
  assert.equal(unavailable.state, "comparisonUnavailable");
  assert.equal(unavailable.disabled, true);
  // `compares_edit` alone is no comparison: the fitted model must have an answer to draw.
  const edited = evidenceFrom(FIXTURE.edited);
  const qq = FIXTURE.edited.check_qq;
  const noFit = { ...qq, models: { ...qq.models, original: { ...qq.models.original, available: false, reason: "No rows." } } };
  const answered = { ...edited, check_qq: { ...edited.check_qq, payload: noFit } };
  assert.equal(compareSwitch(answered, ["check_qq"], true, editedSnapshot()).state, "comparisonUnavailable");
  const bodies = new Set(["hidden", "asFitted", "waitingForComparison", "comparisonUnavailable", "ready"]
    .map((state) => COMPARE_CONTROL[state].body));
  assert.equal(bodies.size, 4);
  for (const [state, control] of Object.entries(COMPARE_CONTROL)) {
    assert.equal(control.disabled, state !== "ready", state);
  }
});

const VIEW = { compare: false, highlight: true, width: 820, height: 520 };

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

test("the readout table: hidden, pointed, stepped by the keys, or suspended through a gesture", () => {
  assert.deepEqual([...CHECK_READOUT.states], ["hidden", "pointed", "keyboard", "suspended"]);
  assert.equal(CHECK_READOUT.initial, "hidden");
  const moves = Object.fromEntries(CHECK_READOUT.states.map((state) => [state,
    Object.fromEntries(CHECK_READOUT.events.flatMap((event) => {
      const next = CHECK_READOUT.next(state, event);
      return next === null ? [] : [[event, next]];
    }))]));
  assert.deepEqual(moves, {
    hidden: { point: "pointed", key: "keyboard", suspend: "suspended" },
    pointed: { point: "pointed", key: "keyboard", leave: "hidden", suspend: "suspended", redraw: "pointed", scroll: "pointed", close: "hidden" },
    keyboard: { point: "pointed", key: "keyboard", leave: "hidden", suspend: "suspended", redraw: "hidden", scroll: "keyboard", close: "hidden" },
    suspended: { point: "suspended", leave: "suspended", redraw: "suspended", rest: "hidden", resume: "pointed", close: "hidden" },
  });
});

test("the readout waits through a gesture: a read queued before it, one shown, and the keys", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_qq", FIXTURE.calm.check_qq, VIEW);
  const hover = bindCheckHover(dom.nodes.frame, () => scene);
  dom.pointerMove(300, 200);
  hover.suspend();
  dom.flushFrame();
  assert.ok(!dom.readoutShown(), "a read queued before the gesture began is dropped");
  dom.pointerMove(320, 210); dom.flushFrame();
  assert.ok(!dom.readoutShown());
  assert.ok(dom.key("ArrowRight").defaultPrevented);
  assert.ok(!dom.readoutShown(), "the keys read out nothing while a gesture runs");
  // At rest it reads out where the pointer is.
  hover.rest(); dom.flushFrame();
  assert.equal(dom.readoutText(), scene.marks[nearestMark(scene.marks, 320, 210)].text);
  // One shown is hidden at once when a gesture starts, and stays hidden at rest once the pointer left.
  hover.suspend();
  assert.ok(!dom.readoutShown());
  dom.pointerLeave();
  hover.rest(); dom.flushFrame();
  assert.ok(!dom.readoutShown());
  // Closed, the check's readout forgets the pointer: nothing reads out until it moves again.
  dom.pointerMove(300, 200); dom.flushFrame();
  hover.close();
  hover.redraw(); dom.flushFrame();
  assert.ok(!dom.readoutShown());
});

test("a scroll of the frame reads out again where the pointer is, in the next frame", () => {
  const dom = fakeFrame({ height: 200 });
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, { ...VIEW, height: 200 });
  bindCheckHover(dom.nodes.frame, () => scene);
  // A point in view whose nearest point, once the frame scrolls 80 px under the pointer, is another.
  const index = scene.marks.findIndex((m) => m.y > 20 && m.y < 180 && nearestMark(scene.marks, m.x, m.y + 80) !== nearestMark(scene.marks, m.x, m.y));
  assert.ok(index >= 0, "no point changes under the pointer with an 80 px scroll");
  const { x, y } = scene.marks[index];
  dom.pointerMove(x, y); dom.flushFrame();
  assert.equal(dom.readoutText(), scene.marks[nearestMark(scene.marks, x, y)].text);
  dom.scrollTo(80);
  assert.equal(dom.pendingFrames(), 1);
  dom.flushFrame();
  assert.equal(dom.readoutText(), scene.marks[nearestMark(scene.marks, x, y + 80)].text);
});

test("a scroll keeps a stepped point's readout in the view, and lets it go once its point leaves", () => {
  const dom = fakeFrame({ height: 200 });
  const scene = renderCheckView(dom.nodes, "check_ae", FIXTURE.wrong.check_ae, { ...VIEW, height: 200 });
  bindCheckHover(dom.nodes.frame, () => scene);
  dom.key("Home");
  const mark = scene.marks[scene.steps[0]];
  assert.ok(mark.y < 120, "the first point read sits near the top");
  dom.scrollTo(Math.max(0, mark.y - 150));
  dom.flushFrame();
  assert.equal(dom.readoutText(), mark.text);
  dom.scrollTo(mark.y + 20);
  dom.flushFrame();
  assert.ok(!dom.readoutShown(), "its point left the view");
});

test("the arrow keys step through one A/E panel's groups, left to right, before the next panel's", () => {
  const dom = fakeFrame();
  const payload = FIXTURE.wrong.check_ae;
  const scene = renderCheckView(dom.nodes, "check_ae", payload, VIEW);
  bindCheckHover(dom.nodes.frame, () => scene);
  const read = [];
  for (const _ of scene.marks) {
    dom.key("ArrowRight");
    read.push(scene.marks.find((m) => m.text === dom.readoutText()));
  }
  assert.equal(new Set(read).size, scene.marks.length, "every point is read once");
  const tables = payload.models.edited.tables;
  const panel = (mark) => tables.findIndex((t) => mark.text.startsWith(`${t.covariate} `));
  for (const [i, mark] of read.entries()) {
    if (i === 0) continue;
    const [was, now] = [panel(read[i - 1]), panel(mark)];
    assert.ok(now > was || (now === was && mark.x > read[i - 1].x), `${read[i - 1].text} then ${mark.text}`);
  }
});

test("a step by the keys is said in full to a screen reader, which the plot's role lets reach its readout", () => {
  const name = "q".repeat(60);
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_ae", namedGroup(name), VIEW);
  // An img's children are presentational; the plot takes its own keys.
  assert.equal(dom.nodes.frame.getAttribute("role"), "application");
  assert.equal(dom.nodes.frame.getAttribute("aria-roledescription"), "chart");
  bindCheckHover(dom.nodes.frame, () => scene, { live: dom.nodes.live });
  const long = scene.steps.findIndex((i) => scene.marks[i].aria?.includes(name));
  assert.ok(long > 0, "no readout shortens the long name");
  dom.key("Home");
  assert.equal(dom.nodes.live.textContent, scene.marks[scene.steps[0]].text);
  for (let i = 0; i < long; i++) dom.key("ArrowRight");
  assert.equal(dom.nodes.live.textContent, scene.marks[scene.steps[long]].aria);
  // A hover is not said; leaving the keys' point clears what was.
  dom.pointerMove(10, 10); dom.flushFrame();
  assert.equal(dom.nodes.live.textContent, "");
  // A frame with no plot has no role.
  renderCheckView(dom.nodes, "check_ae", null, VIEW);
  assert.equal(dom.nodes.frame.getAttribute("role"), null);
  assert.equal(dom.nodes.frame.getAttribute("aria-roledescription"), null);
});

test("the view is headed by the check's label, with its split and rows beside it", () => {
  const dom = fakeFrame();
  const scene = renderCheckView(dom.nodes, "check_qq", FIXTURE.calm.check_qq, VIEW);
  assert.equal(dom.nodes.title.textContent, "Q-Q plot");
  // The Help icon beside the title names the check whose entry it shows.
  assert.equal(dom.nodes.help.dataset.helpCheck, "check_qq");
  assert.equal(dom.nodes.help.getAttribute("aria-label"), "About the Q-Q plot");
  renderCheckView(dom.nodes, "check_ae", FIXTURE.calm.check_ae, VIEW);
  assert.equal(dom.nodes.help.dataset.helpCheck, "check_ae");
  renderCheckView(dom.nodes, "check_qq", FIXTURE.calm.check_qq, VIEW);
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

test("compared, a fitted body that draws nothing is not named in the split line", () => {
  const note = "The band treats the fitted parameters as exact, because the fit could not be simulated.";
  const payload = structuredClone(FIXTURE.edited.check_ae);
  Object.assign(payload.models.original, { level_source: "known", n_draws: 0, level_note: note });
  const dom = fakeFrame();
  // Drawn, the fit's fallback is named.
  renderCheckView(dom.nodes, "check_ae", payload, { ...VIEW, compare: true });
  assert.match(dom.nodes.split.innerHTML, />parameters taken as exact \(as fitted\)<\/span>$/);
  // With no finite interval the fit draws nothing, so there is no fitted series to mark.
  for (const table of payload.models.original.tables) table.lower = table.lower.map(() => null);
  const scene = renderCheckView(dom.nodes, "check_ae", payload, { ...VIEW, compare: true });
  assert.ok(!scene.layers.some((l) => l.role === "original"));
  assert.equal(dom.nodes.legend.hidden, true);
  assert.doesNotMatch(dom.nodes.split.innerHTML, /as fitted|check-level/);
  // The edit's own fallback is still given, naming no model.
  Object.assign(payload.models.edited, { level_source: "known", n_draws: 0, level_note: note });
  renderCheckView(dom.nodes, "check_ae", payload, { ...VIEW, compare: true });
  assert.match(dom.nodes.split.innerHTML, />parameters taken as exact<\/span>$/);
});

test("compared, the legend and split line stay as they are whatever a zoom shows of the fit", () => {
  const note = "The band treats the fitted parameters as exact, because the fit could not be simulated.";
  const payload = structuredClone(FIXTURE.edited.check_qq);
  Object.assign(payload.models.original, { level_source: "known", n_draws: 0, level_note: note });
  const options = { ...VIEW, compare: true };
  // A corner of the plot off the diagonal, where no fitted point lies.
  const { base } = checkScene("check_qq", payload, VIEW, options).zoom;
  const w = base.x[1] - base.x[0];
  const corner = { x: [base.x[0], base.x[0] + 0.05 * w], y: [base.y[1] - 0.05 * w, base.y[1]] };
  const dom = fakeFrame();
  const zoomed = renderCheckView(dom.nodes, "check_qq", payload, { ...options, domain: corner });
  assert.ok(!zoomed.layers.some((l) => l.role === "original"));
  assert.equal(dom.nodes.legend.hidden, false);
  assert.match(dom.nodes.split.innerHTML, />parameters taken as exact \(as fitted\)<\/span>$/);
  // Shown whole again, the fitted points return under the same legend.
  const whole = redrawCheckPlot(dom.nodes.frame, "check_qq", payload, { ...options, domain: null });
  assert.ok(whole.layers.some((l) => l.role === "original"));
  assert.equal(dom.nodes.legend.hidden, false);
  // And the full render, whole, agrees with the zoomed one.
  renderCheckView(dom.nodes, "check_qq", payload, options);
  assert.equal(dom.nodes.legend.hidden, false);
  assert.match(dom.nodes.split.innerHTML, />parameters taken as exact \(as fitted\)<\/span>$/);
});

test("a zoom's redraw writes the plot's svg in place, as its scene draws it", () => {
  const payload = FIXTURE.wrong.check_qq;
  const dom = fakeFrame();
  const whole = renderCheckView(dom.nodes, "check_qq", payload, VIEW);
  const svg = dom.nodes.frame.querySelector("svg");
  const { base } = whole.zoom;
  const domains = [
    { x: [base.x[0], base.x[0] + 0.3 * (base.x[1] - base.x[0])], y: [base.y[0], base.y[0] + 0.2 * (base.y[1] - base.y[0])] },
    { x: [base.x[1] - 0.05 * (base.x[1] - base.x[0]), base.x[1]], y: [base.y[0], base.y[0] + 0.05 * (base.y[1] - base.y[0])] },
    null,
  ];
  const round = (v) => String(Math.round(v * 100) / 100);
  for (const domain of domains) {
    const scene = redrawCheckPlot(dom.nodes.frame, "check_qq", payload, { ...VIEW, domain });
    assert.equal(dom.nodes.frame.querySelector("svg"), svg, "the svg is kept");
    assert.equal(dom.nodes.frame.html.length > 0, true);
    const kinds = svg.children.map((node) => node.tag);
    // The paths, then the labels, then the readout, as sceneMarkup lays them out.
    assert.deepEqual(kinds, [...scene.layers.map(() => "path"), ...scene.labels.map(() => "text"), "g"]);
    const paths = svg.children.filter((node) => node.tag === "path");
    assert.deepEqual(paths.map((p) => [p.getAttribute("class"), p.getAttribute("d")]),
      scene.layers.map((l) => [`check-${l.role}`, l.d]));
    const texts = svg.children.filter((node) => node.tag === "text");
    assert.deepEqual(texts.map((t) => [t.getAttribute("class"), t.getAttribute("x"), t.getAttribute("y"), t.getAttribute("text-anchor"), t.textContent]),
      scene.labels.map((l) => [l.axis ? "check-title" : "check-tick", round(l.x), round(l.y), l.anchor, l.text]));
    assert.equal(dom.readout().hidden, true);
  }
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

test("a readout counts a group name by character and never splits one", () => {
  const whole = `${"q".repeat(38)}😀A`;
  assert.equal(Array.from(whole).length, 40);
  const longer = `${"q".repeat(38)}😀AB`;
  assert.equal(Array.from(longer).length, 41);
  for (const [name, shown] of [[whole, whole], [longer, `${"q".repeat(38)}😀…`]]) {
    const dom = fakeFrame();
    const scene = renderCheckView(dom.nodes, "check_ae", namedGroup(name), VIEW);
    const mark = scene.marks.find((m) => m.text.startsWith(`g ${"q".repeat(38)}`));
    assert.ok(mark, "no readout names the group");
    assert.ok(mark.text.startsWith(`g ${shown} · `), mark.text);
    assert.equal(Array.from(shown).length, 40);
    // No character is cut in half: the text holds no lone surrogate.
    assert.ok(mark.text.isWellFormed(), "the readout splits a character");
  }
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
