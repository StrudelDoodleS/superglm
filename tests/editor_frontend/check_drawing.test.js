// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { checkScene, nearestMark } from "../../src/superglm/editor/app/chart/check_plots.js";
import { ZOOM_IN, zoomAbout } from "../../src/superglm/editor/app/chart/check_zoom.js";
import { ACTIVE_CLASS, bindChartFocus } from "../../src/superglm/editor/app/views/chart_focus.js";
import { CHECK_DRAWING, bindCheckDrawing } from "../../src/superglm/editor/app/views/check_drawing.js";
import { CHECK_FIXTURE as FIXTURE } from "./check_evidence.js";
import { fakeDocument, fakeFrame } from "./check_frame.js";

const VIEW = { compare: false, highlight: true };

/**
 * The open check as main.js mounts it: drawn from the wrong fit's payloads, its zoom kept per
 * check and drawn again in place at once, as the store's subscription does, under the page's
 * wheel focus. The frame sits at the page's origin, so client px are scene px.
 */
function mounted({ width = 820, height = 520 } = {}) {
  const dom = fakeFrame({ width, height });
  const { frame } = dom.nodes;
  const doc = fakeDocument();
  const focus = bindChartFocus(doc);
  focus.register(frame);
  const page = { open: null, zooms: {}, published: [] };
  const drawing = bindCheckDrawing(dom.nodes, {
    focus,
    onZoom(domain) {
      page.published.push(domain);
      page.zooms = { ...page.zooms, [page.open]: domain };
      drawing.zoom(page.open, domain);
    },
  });
  /** Draw `report` afresh, as an evidence reply, a switch or a resize to `size` does. */
  const draw = (report = page.open, size = { width, height }, payload = FIXTURE.wrong[report]) => {
    page.open = report;
    drawing.draw(report, payload, { ...VIEW, ...size, domain: page.zooms[report] ?? null });
  };
  /** `report` drawn zoomed in about the middle of its plot. */
  const zoomed = (report) => {
    const whole = checkScene(report, FIXTURE.wrong[report], { width, height }, VIEW).zoom;
    page.zooms[report] = zoomAbout(whole, { x: width / 2, y: height / 2 }, ZOOM_IN ** 4);
    draw(report);
  };
  return { dom, frame, doc, focus, drawing, page, draw, zoomed, scene: () => drawing.current().scene ?? null };
}

test("the drawing table: a message or a plot drawn from anywhere, a plot zoomed in place, and closed", () => {
  assert.equal(CHECK_DRAWING.initial, "closed");
  const walked = CHECK_DRAWING.states.flatMap((state) => CHECK_DRAWING.events.map((event) => [state, event, CHECK_DRAWING.next(state, event)]));
  assert.deepEqual(walked, [
    ["closed", "message", "message"],
    ["closed", "plot", "plot"],
    ["closed", "close", null],
    ["closed", "zoom", null],
    ["message", "message", "message"],
    ["message", "plot", "plot"],
    ["message", "close", "closed"],
    ["message", "zoom", null],
    ["plot", "message", "message"],
    ["plot", "plot", "plot"],
    ["plot", "close", "closed"],
    ["plot", "zoom", "plot"],
  ]);
});

test("what the frame holds is one record: closed, a message, the plot, zoomed in place, closed again", () => {
  const m = mounted();
  assert.deepEqual(m.drawing.current(), { state: "closed" });
  assert.equal(m.drawing.size(), null);
  m.drawing.draw("check_qq", null, { ...VIEW, width: 820, height: 520 });
  assert.deepEqual(m.drawing.current(), { state: "message", report: "check_qq", size: "820x520" });
  assert.equal(m.drawing.zoom("check_qq", null), false, "a message has no plot to zoom");
  m.draw("check_qq");
  const plot = m.drawing.current();
  assert.equal(plot.state, "plot");
  assert.equal(plot.size, "820x520");
  const svg = m.frame.querySelector("svg");
  const domain = zoomAbout(plot.scene.zoom, { x: 400, y: 260 }, ZOOM_IN);
  assert.equal(m.drawing.zoom("check_qq", domain), true);
  assert.equal(m.frame.querySelector("svg"), svg, "a zoom draws the same svg in place");
  assert.equal(m.drawing.current().options.domain, domain);
  assert.equal(m.drawing.zoom("check_pit", null), false, "no plot of another check is drawn");
  // The same check drawn again reads out at once where the pointer is; another one forgets it.
  m.dom.pointerMove(300, 200);
  m.dom.flushFrame();
  m.draw("check_qq");
  assert.ok(m.dom.readoutShown());
  m.draw("check_pit");
  m.dom.flushFrame();
  assert.ok(!m.dom.readoutShown(), "nothing reads out on the new check until the pointer moves");
  m.drawing.close();
  assert.deepEqual(m.drawing.current(), { state: "closed" });
  assert.equal(m.drawing.zoom("check_qq", null), false);
  m.drawing.destroy();
});

test("a full redraw ends a box drag on the plot it replaced: its box, capture and release go with it", () => {
  const m = mounted();
  m.draw("check_qq");
  const svg = m.frame.querySelector("svg");
  m.dom.pointerDown(300, 200, { shiftKey: true });
  m.dom.pointerMove(500, 300);
  assert.ok(svg.querySelector(".check-zoom-box"));
  assert.ok(m.frame.hasPointerCapture(1));
  // An evidence reply draws the check afresh while the drag is still held.
  m.draw("check_qq");
  assert.equal(svg.querySelector(".check-zoom-box"), null, "the box is off the old plot");
  assert.equal(m.frame.querySelector(".check-zoom-box"), null);
  assert.ok(!m.frame.hasPointerCapture(1));
  m.dom.pointerMove(520, 320);
  m.dom.pointerUp(520, 320);
  m.dom.flushFrame();
  assert.deepEqual(m.page.published, [], "the release zooms to no box");
  m.drawing.destroy();
});

test("a resize during a pan ends it: no pan mixes the old plot's geometry with the new one's", () => {
  const m = mounted();
  m.zoomed("check_qq");
  const before = m.page.zooms.check_qq;
  m.dom.pointerDown(400, 260);
  assert.equal(m.frame.dataset.panning, "true");
  m.dom.pointerMove(420, 250);
  m.draw("check_qq", { width: 700, height: 440 });
  assert.equal(m.frame.dataset.panning, undefined);
  m.dom.flushFrame();
  m.dom.pointerMove(460, 250);
  m.dom.pointerUp(460, 250);
  m.dom.flushFrame();
  assert.deepEqual(m.page.published, []);
  assert.equal(m.page.zooms.check_qq, before);
  m.drawing.destroy();
});

test("a zoom drawn in place is no replacement: the pan making it goes on", () => {
  const m = mounted();
  m.zoomed("check_qq");
  m.dom.pointerDown(400, 260);
  m.dom.pointerMove(420, 250);
  m.dom.flushFrame();
  assert.equal(m.page.published.length, 1);
  assert.equal(m.frame.dataset.panning, "true");
  m.dom.pointerMove(440, 250);
  m.dom.pointerUp(440, 250);
  m.dom.flushFrame();
  assert.equal(m.page.published.length, 2);
  m.drawing.destroy();
});

test("the readout waits for a drag on the drawn check: a read queued before its press, and one shown", () => {
  const m = mounted();
  m.zoomed("check_qq");
  m.dom.pointerMove(300, 200);
  m.dom.pointerDown(300, 200);
  m.dom.flushFrame();
  assert.ok(!m.dom.readoutShown(), "the read queued before the press waits for the drag");
  m.dom.pointerUp(300, 200);
  m.dom.flushFrame();
  const scene = m.scene();
  assert.equal(m.dom.readoutText(), scene.marks[nearestMark(scene.marks, 300, 200)].text);
  // A press that begins a pan hides a readout shown, before any move.
  m.dom.pointerDown(300, 200);
  assert.ok(!m.dom.readoutShown());
  m.dom.pointerUp(300, 200);
  m.drawing.destroy();
});

test("a wheel turned on one check holds nothing on the next, whose own wheel the readout waits for", () => {
  const m = mounted();
  m.draw("check_qq");
  m.dom.wheel(400, 260);
  m.dom.flushFrame();
  assert.equal(m.page.published.length, 1);
  m.draw("check_pit");
  m.dom.pointerMove(300, 200);
  m.dom.flushFrame();
  assert.ok(m.dom.readoutShown(), "the new check reads out at once");
  m.dom.wheel(300, 200);
  assert.ok(!m.dom.readoutShown(), "its own wheel suspends the readout");
  m.drawing.destroy();
});

test("opening another check lets the wheel go, and the same check drawn again keeps it", () => {
  const m = mounted();
  m.draw("check_qq");
  m.doc.dispatch("pointerdown", { target: m.frame });
  assert.ok(m.focus.isActive(m.frame));
  m.draw("check_qq");
  m.draw("check_qq", { width: 700, height: 440 });
  assert.ok(m.focus.isActive(m.frame), "a redraw of the same check keeps the wheel");
  assert.ok(m.frame.classList.contains(ACTIVE_CLASS));
  // Opened by the keyboard from the list: no press reaches the page.
  m.draw("check_pit");
  assert.ok(!m.focus.isActive(m.frame));
  assert.ok(!m.frame.classList.contains(ACTIVE_CLASS));
  m.doc.dispatch("pointerdown", { target: m.frame });
  m.drawing.close();
  assert.ok(!m.focus.isActive(m.frame), "a closed check lets it go");
  m.drawing.destroy();
});

test("a frame whose plot runs past an edge says which, as it scrolls and as it is drawn again", () => {
  const m = mounted({ height: 200 });
  m.draw("check_ae");
  assert.ok(m.scene().height > 200);
  assert.equal(m.frame.dataset.overflow, "below");
  m.dom.scrollTo(80);
  assert.equal(m.frame.dataset.overflow, "above below");
  m.dom.scrollTo(m.frame.scrollHeight - 200);
  assert.equal(m.frame.dataset.overflow, "above");
  // A plot that fits the frame says nothing, nor does a message.
  m.draw("check_qq", { width: 820, height: 200 });
  assert.equal(m.frame.dataset.overflow, undefined);
  m.draw("check_ae", { width: 820, height: 200 });
  assert.equal(m.frame.dataset.overflow, "below");
  m.drawing.draw("check_ae", null, { ...VIEW, width: 820, height: 200 });
  assert.equal(m.frame.dataset.overflow, undefined);
  m.drawing.destroy();
});
