// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { checkScene } from "../../src/superglm/editor/app/chart/check_plots.js";
import { ZOOM_IN, panBy, zoomAbout, zoomToBox } from "../../src/superglm/editor/app/chart/check_zoom.js";
import { bindCheckZoom } from "../../src/superglm/editor/app/views/check_gestures.js";
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
 * scene's size, so client px are scene px.
 */
function checkView(open = "check_qq") {
  const frames = [];
  globalThis.requestAnimationFrame = (callback) => frames.push(callback);
  globalThis.cancelAnimationFrame = (id) => { frames[id - 1] = null; };
  const svg = { getBoundingClientRect: () => ({ left: 0, top: 0, width: BOX.width, height: BOX.height }), append() {} };
  const frame = Object.assign(new FakeTarget(), {
    scrollHeight: BOX.height, clientHeight: BOX.height, dataset: {},
    querySelector: (selector) => (selector === "svg" ? svg : null),
    setPointerCapture() {},
    ownerDocument: { createElementNS: () => ({ setAttribute() {}, remove() {} }) },
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
  });
  return {
    view, gestures, draw,
    zoomTo(domain) { view.zooms[view.open] = domain; draw(); },
    openCheck(name) { view.open = name; draw(); },
    wheel: (x, y, deltaY = -100) => frame.dispatch("wheel", { deltaY, deltaMode: 0, ctrlKey: true, clientX: x, clientY: y }),
    down: (x, y, shiftKey = false) => frame.dispatch("pointerdown", { button: 0, pointerId: 1, clientX: x, clientY: y, shiftKey }),
    move: (x, y) => frame.dispatch("pointermove", { clientX: x, clientY: y }),
    up: (x, y) => frame.dispatch("pointerup", { clientX: x, clientY: y }),
    lose: (type) => frame.dispatch(type, { pointerId: 1 }),
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
