// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { checkScene } from "../../src/superglm/editor/app/chart/check_plots.js";
import { ZOOM_IN, panBy, zoomAbout, zoomToBox } from "../../src/superglm/editor/app/chart/check_zoom.js";
import { CHECK_DRAG, CHECK_WHEEL, bindCheckZoom } from "../../src/superglm/editor/app/views/check_gestures.js";
import { CHECK_FIXTURE as FIXTURE } from "./check_evidence.js";

const BOX = { width: 820, height: 520 };

/** A listener table standing in for an element. */
class FakeTarget {
  constructor() { this.listeners = new Map(); }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]); }
  removeEventListener(type, listener) { this.listeners.set(type, (this.listeners.get(type) ?? []).filter((l) => l !== listener)); }
  dispatch(type, event = {}) {
    const full = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, ...event };
    for (const listener of this.listeners.get(type) ?? []) listener(full);
    return full;
  }
}

/**
 * The check view as main.js wires it: the open check, its zoom kept per check, and the scene
 * drawn again, at once, whenever a zoom is published. The svg sits at the page's origin at the
 * scene's size, so client px are scene px. The frame holds the pointers it captured, and the
 * svg the boxes appended to it until they are removed.
 */
function checkView(open = "check_qq", { onStart, onRest } = {}) {
  const frames = [];
  globalThis.requestAnimationFrame = (callback) => frames.push(callback);
  globalThis.cancelAnimationFrame = (id) => { frames[id - 1] = null; };
  const boxes = new Set();
  const svg = {
    getBoundingClientRect: () => ({ left: 0, top: 0, width: BOX.width, height: BOX.height }),
    append: (node) => boxes.add(node),
  };
  const captured = new Set();
  const frame = Object.assign(new FakeTarget(), {
    scrollHeight: BOX.height, clientHeight: BOX.height, dataset: {},
    querySelector: (selector) => (selector === "svg" ? svg : null),
    setPointerCapture: (id) => captured.add(id),
    hasPointerCapture: (id) => captured.has(id),
    releasePointerCapture: (id) => captured.delete(id),
    ownerDocument: { createElementNS: () => { const node = { setAttribute() {}, remove: () => boxes.delete(node) }; return node; } },
  });
  const view = { open, zooms: {}, published: [] };
  const draw = () => {
    view.scene = checkScene(view.open, FIXTURE.wrong[view.open], BOX,
      { compare: false, highlight: false, domain: view.zooms[view.open] ?? null });
  };
  draw();
  const gestures = bindCheckZoom(frame, {
    getScene: () => view.scene,
    isActive: () => true,
    onZoom: (domain) => {
      view.published.push(domain);
      view.zooms = { ...view.zooms, [view.open]: domain };
      if (domain === null) delete view.zooms[view.open];
      draw();
    },
    onStart,
    onRest,
  });
  return {
    view, gestures, draw, frame, captured, boxes,
    zoomTo(domain) { view.zooms[view.open] = domain; draw(); },
    openCheck(name) { view.open = name; draw(); },
    wheel: (x, y, deltaY = -100) => frame.dispatch("wheel", { deltaY, deltaMode: 0, ctrlKey: true, clientX: x, clientY: y }),
    down: (x, y, shiftKey = false, pointerId = 1) =>
      frame.dispatch("pointerdown", { button: 0, pointerId, clientX: x, clientY: y, shiftKey }),
    move: (x, y, pointerId = 1) => frame.dispatch("pointermove", { pointerId, clientX: x, clientY: y }),
    up: (x, y, pointerId = 1) => frame.dispatch("pointerup", { pointerId, clientX: x, clientY: y }),
    lose: (type, pointerId = 1) => frame.dispatch(type, { pointerId }),
    key: (key) => frame.dispatch("keydown", { key }),
    dblclick: () => frame.dispatch("dblclick"),
    flushFrame() { for (const callback of frames.splice(0)) callback?.(0); },
  };
}

/** A zoom frame `zoom` at `domain`, or whole for null. */
const at = (zoom, domain) => ({ ...zoom, domain: domain ?? zoom.base });

test("a pan and a wheel turned in one frame publish one domain, the two composed", () => {
  const page = checkView();
  const whole = page.view.scene.zoom;
  page.zoomTo(zoomAbout(whole, { x: 400, y: 260 }, ZOOM_IN ** 4));
  const zoomed = page.view.scene.zoom;
  page.down(400, 260);
  page.move(440, 240);
  page.wheel(500, 200);
  assert.deepEqual(page.view.published, []);
  page.flushFrame();
  assert.equal(page.view.published.length, 1);
  const panned = at(zoomed, panBy(zoomed, { x: 400, y: 260 }, { x: 440, y: 240 }));
  assert.deepEqual(page.view.published[0], zoomAbout(panned, { x: 500, y: 200 }, ZOOM_IN));
  // The drag pans on from the domain that frame drew, so the wheel's zoom is kept.
  page.move(460, 240);
  page.flushFrame();
  assert.equal(page.view.published.length, 2);
  const width = (d) => d.x[1] - d.x[0];
  assert.ok(Math.abs(width(page.view.published[1]) - width(page.view.published[0])) < 1e-12);
});

/** Every state of `machine` against every event it takes anywhere, with where each leads. */
const walk = (machine) => machine.states.flatMap((state) => machine.events.map((event) => [state, event, machine.next(state, event)]));

test("the drag table: a pan or a box begins from idle, is released or dropped back there, and nothing else moves", () => {
  assert.deepEqual([...CHECK_DRAG.states], ["idle", "panning", "boxing"]);
  assert.equal(CHECK_DRAG.initial, "idle");
  assert.deepEqual(walk(CHECK_DRAG), [
    ["idle", "pan", "panning"],
    ["idle", "box", "boxing"],
    ["idle", "release", null],
    ["idle", "drop", null],
    ["panning", "pan", null],
    ["panning", "box", null],
    ["panning", "release", "idle"],
    ["panning", "drop", "idle"],
    ["boxing", "pan", null],
    ["boxing", "box", null],
    ["boxing", "release", "idle"],
    ["boxing", "drop", "idle"],
  ]);
});

test("the wheel table: a turn sets it turning, and it comes to rest or is stopped", () => {
  assert.equal(CHECK_WHEEL.initial, "still");
  assert.deepEqual(walk(CHECK_WHEEL), [
    ["still", "turn", "turning"],
    ["still", "rest", null],
    ["still", "stop", null],
    ["turning", "turn", "turning"],
    ["turning", "rest", "still"],
    ["turning", "stop", "still"],
  ]);
});

test("a drag belongs to the pointer that began it: another pointer's move, release, cancel or capture loss moves nothing", () => {
  const page = checkView();
  page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
  const zoomed = page.view.scene.zoom;
  page.down(400, 260);
  // A second contact's press starts nothing, and its move pans nothing.
  page.down(300, 200, false, 2);
  page.move(500, 200, 2);
  page.flushFrame();
  assert.deepEqual(page.view.published, []);
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    page.frame.dispatch(type, { pointerId: 2, clientX: 300, clientY: 200 });
    assert.equal(page.gestures.busy(), true, type);
    assert.equal(page.frame.dataset.panning, "true", type);
    assert.ok(page.captured.has(1), type);
  }
  page.move(440, 240);
  page.up(450, 240);
  page.flushFrame();
  assert.deepEqual(page.view.published, [panBy(zoomed, { x: 400, y: 260 }, { x: 450, y: 240 })]);
  assert.equal(page.gestures.busy(), false);
});

test("cancel ends a drag: its box comes off the plot, its cursor and capture go, and its release moves nothing", () => {
  for (const shiftKey of [false, true]) {
    const page = checkView();
    page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
    page.down(400, 260, shiftKey);
    page.move(460, 300);
    assert.ok(page.captured.has(1));
    assert.equal(page.boxes.size, shiftKey ? 1 : 0);
    page.gestures.cancel();
    assert.equal(page.boxes.size, 0, "the box is off the plot");
    assert.ok(!page.captured.has(1), "the capture is let go");
    assert.equal(page.frame.dataset.panning, undefined);
    page.move(480, 300);
    page.up(480, 300);
    page.flushFrame();
    assert.deepEqual(page.view.published, [], `shift ${shiftKey}`);
  }
});

test("cancel stops a turning wheel at once, and onStart and onRest bracket every gesture", () => {
  const said = [];
  const page = checkView("check_qq", { onStart: () => said.push("start"), onRest: () => said.push("rest") });
  page.wheel(400, 260);
  page.wheel(400, 260);
  assert.equal(page.gestures.busy(), true);
  page.gestures.cancel();
  assert.equal(page.gestures.busy(), false, "a cancelled wheel is still at once, not after its rest");
  assert.deepEqual(said, ["start", "rest"]);
  // A drag: busy from its press to its release, and a wheel turned during it starts nothing more.
  page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
  page.down(400, 260);
  page.wheel(400, 260);
  page.up(420, 260);
  assert.deepEqual(said, ["start", "rest", "start"], "the wheel still turns after the release");
  page.gestures.cancel();
  assert.deepEqual(said, ["start", "rest", "start", "rest"]);
  // A press that begins no drag, on a check shown whole, is no gesture.
  page.gestures.reset();
  page.flushFrame();
  page.down(400, 260);
  page.up(400, 260);
  assert.deepEqual(said, ["start", "rest", "start", "rest"]);
  page.gestures.destroy();
});

test("a press during a pan starts no box: the pan goes on and its release publishes it", () => {
  const page = checkView();
  page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
  const before = page.view.scene.zoom;
  page.down(400, 260);
  const second = page.down(300, 200, true);
  assert.equal(second.defaultPrevented, false);
  page.move(440, 260);
  page.up(440, 260);
  page.flushFrame();
  assert.deepEqual(page.view.published, [panBy(before, { x: 400, y: 260 }, { x: 440, y: 260 })]);
  assert.equal(page.gestures.busy(), false);
});

test("letting go of a pan or a box publishes in the next frame, never at once", () => {
  const page = checkView();
  page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
  page.down(400, 260);
  page.move(420, 250);
  page.flushFrame();
  assert.equal(page.view.published.length, 1);
  const moved = page.view.scene.zoom;
  page.move(430, 250);
  page.up(450, 250);
  assert.equal(page.view.published.length, 1);
  page.flushFrame();
  assert.equal(page.view.published.length, 2);
  assert.ok(page.view.published[1].x[0] < moved.domain.x[0]);
  // A box: nothing until the frame, then the box's domain.
  const before = page.view.scene.zoom;
  page.down(300, 200, true);
  page.move(500, 300);
  page.up(500, 300);
  assert.equal(page.view.published.length, 2);
  page.flushFrame();
  assert.equal(page.view.published.length, 3);
  assert.deepEqual(page.view.published[2], zoomToBox(before, { x: 300, y: 200 }, { x: 500, y: 300 }));
});

test("a pan cancelled or losing its pointer moves nothing, and a release's own capture loss keeps it", () => {
  for (const type of ["pointercancel", "lostpointercapture"]) {
    const page = checkView();
    page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
    const zoomed = page.view.zooms.check_qq;
    page.down(400, 260);
    page.move(440, 240);
    page.lose(type);
    page.flushFrame();
    assert.deepEqual(page.view.published, [], type);
    assert.deepEqual(page.view.zooms.check_qq, zoomed, type);
    assert.equal(page.gestures.busy(), false, type);
  }
  // A browser releases capture after every pointerup; the release's pan is still drawn.
  const page = checkView();
  page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 4));
  const zoomed = page.view.scene.zoom;
  page.down(400, 260);
  page.move(440, 240);
  page.up(450, 240);
  page.lose("lostpointercapture");
  page.flushFrame();
  assert.deepEqual(page.view.published, [panBy(zoomed, { x: 400, y: 260 }, { x: 450, y: 240 })]);
});

test("a queued wheel belongs to its check: opening another, Home, a double-click or 0 drops it", () => {
  // Opened another check before the frame: neither check's domain moves.
  let page = checkView("check_qq");
  page.wheel(400, 260);
  page.openCheck("check_pit");
  page.flushFrame();
  assert.deepEqual(page.view.zooms, {});
  assert.deepEqual(page.view.published, []);
  // Home before the frame: the check is whole, published once, and stays whole.
  for (const reset of [(p) => p.gestures.reset(), (p) => p.dblclick(), (p) => p.key("0")]) {
    page = checkView("check_qq");
    page.zoomTo(zoomAbout(page.view.scene.zoom, { x: 400, y: 260 }, ZOOM_IN ** 2));
    page.wheel(400, 260);
    reset(page);
    page.flushFrame();
    assert.deepEqual(page.view.published, [null]);
    assert.deepEqual(page.view.zooms, {});
  }
  // The split changing drops it too: main.js cancels, then clears every zoom.
  page = checkView("check_qq");
  page.wheel(400, 260);
  page.gestures.cancel();
  page.flushFrame();
  assert.deepEqual(page.view.published, []);
});

test("the keys' zoom waits for the frame and joins the wheel's", () => {
  const page = checkView("check_pit");
  page.key("+");
  page.key("+");
  page.wheel(400, 260);
  assert.deepEqual(page.view.published, []);
  page.flushFrame();
  assert.equal(page.view.published.length, 1);
  const width = page.view.published[0].y[1] - page.view.published[0].y[0];
  const whole = page.view.scene.zoom.base;
  assert.ok(Math.abs(width - (whole.y[1] - whole.y[0]) * ZOOM_IN ** 3) < 1e-9);
});
