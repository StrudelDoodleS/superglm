// @ts-check
/**
 * Zoom and pan on a check drawn large, with the Editor chart's gestures. On the chart the wheel
 * works on (see chart_focus.js), the wheel zooms about the pointer, unless the frame is taller
 * than the view, which the wheel then scrolls; Ctrl or Cmd with the wheel, or a pinch, zooms any
 * check at any time. Off it, a plain wheel is the page's: an inactive frame does not scroll
 * (assess.css), so the browser hands that wheel on to whatever scrolls around it. Shift and drag
 * zooms to a box, a drag on a zoomed check pans it, and a double-click, Home or the 0 key shows
 * it whole; + and - zoom about the middle while it has focus.
 *
 * Every gesture queues its input for the next animation frame, bound to the scene it was given
 * on; that frame composes all of it into one domain and hands it to `onZoom` once, and the view
 * draws the check again at it, asking Python for nothing. Input queued on a scene no longer on
 * screen (another check opened, the check drawn afresh) is dropped, and `reset` and `cancel` drop
 * it too. While the wheel turns, and through a drag, the gesture is busy and the hover readout
 * waits; `onRest` says when it may come back.
 */

import { ZOOM_IN, ZOOM_OUT, isWhole, panBy, stepZoom, zoomAbout, zoomToBox } from "../chart/check_zoom.js";

/** @typedef {import('../chart/check_plots.js').Scene} Scene */
/** @typedef {import('../chart/check_plots.js').Domain} Domain */
/** @typedef {import('../chart/check_plots.js').ZoomFrame} ZoomFrame */
/** @typedef {import('../chart/check_zoom.js').Point} Point */
/**
 * @typedef {Object} CheckZoomOptions
 * @property {() => Scene|null} getScene the scene on screen
 * @property {() => boolean} isActive whether the wheel works on this frame
 * @property {(domain: Domain|null) => void} onZoom draw the check on screen at `domain`, or whole
 *   for null
 * @property {() => void} [onRest] the wheel has come to rest, or a drag has ended
 */
/**
 * The input of one animation frame, on the scene it was given on: a drag's latest pan from where
 * it began, a box let go, the wheel's ticks as one factor about where the last one was, the keys'
 * steps, and whether a drag ended.
 * @typedef {{scene: Scene, pan: {zoom: ZoomFrame, from: Point, to: Point}|null,
 *   box: {from: Point, to: Point}|null, factor: number, at: Point, step: number, rest: boolean}} Queued
 */
/**
 * A drag under way: a pan, from `from` on the zoom frame `zoom`, last at `last`; or a box from
 * `start`, drawn as `node`.
 * @typedef {{kind: 'pan', zoom: ZoomFrame, from: Point, last: Point}
 *   | {kind: 'box', start: Point, node: SVGRectElement}} Drag
 */

const SVG_NS = "http://www.w3.org/2000/svg";
/** @type {Readonly<Record<string, number>>} */
const KEY_ZOOM = Object.freeze({ "+": ZOOM_IN, "=": ZOOM_IN, "-": ZOOM_OUT, _: ZOOM_OUT });
// How long after its last tick a wheel counts as at rest, as on the Editor chart.
const WHEEL_REST_MS = 160;

/**
 * Bind zoom and pan to the check in `frame`.
 * @param {HTMLElement} frame @param {CheckZoomOptions} options
 * @returns {{busy(): boolean, reset(): void, cancel(): void, destroy(): void}} `busy` while the
 *   wheel turns or a drag runs; `reset` shows the check whole; `cancel` drops what is queued
 */
export function bindCheckZoom(frame, { getScene, isActive, onZoom, onRest }) {
  let frameId = 0;
  /** @type {Queued|null} */
  let queued = null;
  /** @type {ReturnType<typeof setTimeout>|0} */
  let turning = 0;
  /** @type {Drag|null} */
  let drag = null;

  /** The pointer in the scene's px: the svg's box moves with the frame's scroll. @param {MouseEvent} event @returns {Point|null} */
  function scenePoint(event) {
    const scene = getScene();
    const svg = frame.querySelector("svg");
    const rect = svg?.getBoundingClientRect();
    if (!scene || !rect || !(rect.width > 0 && rect.height > 0)) return null;
    return { x: ((event.clientX - rect.left) * scene.width) / rect.width, y: ((event.clientY - rect.top) * scene.height) / rect.height };
  }

  /**
   * The input queued for the next frame on `scene`, asking for that frame; what was queued on
   * another scene is dropped.
   * @param {Scene} scene @returns {Queued}
   */
  function queue(scene) {
    if (queued?.scene !== scene) queued = { scene, pan: null, box: null, factor: 1, at: { x: 0, y: 0 }, step: 1, rest: false };
    if (!frameId) frameId = requestAnimationFrame(onFrame);
    return queued;
  }

  // The frame's one redraw: what was queued, composed in order (the pan, a box, the wheel, the
  // keys), each read at the domain the one before gave, and published once.
  function onFrame() {
    frameId = 0;
    const input = queued;
    queued = null;
    const scene = getScene();
    if (!input || !scene || scene !== input.scene || !scene.zoom) return;
    let zoom = scene.zoom;
    /** @param {Domain|null} domain */
    const move = (domain) => {
      zoom = { ...zoom, domain: domain ?? zoom.base };
    };
    if (input.pan) move(panBy(input.pan.zoom, input.pan.from, input.pan.to));
    if (input.box) move(zoomToBox(zoom, input.box.from, input.box.to));
    if (input.factor !== 1) move(zoomAbout(zoom, input.at, input.factor));
    if (input.step !== 1) move(stepZoom(zoom, input.step));
    if (input.pan || input.box || input.factor !== 1 || input.step !== 1) onZoom(isWhole(zoom.domain, zoom.base) ? null : zoom.domain);
    // A drag under way pans on from what this frame drew, so a zoom made during it is kept.
    if (drag?.kind === "pan" && (input.box || input.factor !== 1 || input.step !== 1)) {
      drag.zoom = getScene()?.zoom ?? drag.zoom;
      drag.from = drag.last;
    }
    if (input.rest) onRest?.();
  }

  /** The drag ended; `rest` says the readout may come back now rather than after a frame. @param {boolean} rest */
  function endDrag(rest) {
    if (!drag) return;
    if (drag.kind === "box") drag.node.remove();
    drag = null;
    delete frame.dataset.panning;
    if (rest) onRest?.();
  }

  // What is queued, and any drag, dropped.
  function cancel() {
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0;
    queued = null;
    endDrag(false);
  }

  // The check whole again, at once, and nothing queued left to move it.
  function reset() {
    cancel();
    onZoom(null);
  }

  /** @param {WheelEvent} event */
  function onWheel(event) {
    const scene = getScene();
    if (!scene?.zoom || event.deltaY === 0) return;
    // A plain wheel zooms the active check, unless its frame is taller than the view, which
    // the wheel then scrolls; off the active chart it is left to the page.
    if (!(event.ctrlKey || event.metaKey) && (!isActive() || frame.scrollHeight > frame.clientHeight)) return;
    const point = scenePoint(event);
    if (!point) return;
    event.preventDefault();
    const input = queue(scene);
    input.factor *= event.deltaY < 0 ? ZOOM_IN : ZOOM_OUT;
    input.at = point;
    clearTimeout(turning);
    turning = setTimeout(() => {
      turning = 0;
      onRest?.();
    }, WHEEL_REST_MS);
  }

  /** @param {Point} to */
  function drawBox(to) {
    if (drag?.kind !== "box") return;
    const { start, node } = drag;
    node.setAttribute("x", String(Math.min(start.x, to.x)));
    node.setAttribute("y", String(Math.min(start.y, to.y)));
    node.setAttribute("width", String(Math.abs(to.x - start.x)));
    node.setAttribute("height", String(Math.abs(to.y - start.y)));
  }

  /** @param {PointerEvent} event */
  function onPointerDown(event) {
    const zoom = getScene()?.zoom;
    const point = event.button === 0 ? scenePoint(event) : null;
    if (!zoom || !point) return;
    if (event.shiftKey) {
      event.preventDefault();
      const node = /** @type {SVGRectElement} */ (frame.ownerDocument.createElementNS(SVG_NS, "rect"));
      node.setAttribute("class", "check-zoom-box");
      frame.querySelector("svg")?.append(node);
      drag = { kind: "box", start: point, node };
      drawBox(point);
    } else if (!isWhole(zoom.domain, zoom.base)) {
      // The press is left to focus the plot; the plot's text cannot be selected (assess.css).
      drag = { kind: "pan", zoom, from: point, last: point };
      frame.dataset.panning = "true";
    } else {
      return;
    }
    frame.setPointerCapture?.(event.pointerId);
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    if (!drag) return;
    const point = scenePoint(event);
    const scene = getScene();
    if (!point || !scene) return;
    if (drag.kind === "box") {
      drawBox(point);
      return;
    }
    drag.last = point;
    queue(scene).pan = { zoom: drag.zoom, from: drag.from, to: point };
  }

  /** @param {PointerEvent} event */
  function onPointerUp(event) {
    const point = scenePoint(event);
    const scene = getScene();
    if (!drag || !point || !scene) {
      endDrag(true);
      return;
    }
    // Let go, the drag's last move is drawn in the next frame with anything else queued.
    const input = queue(scene);
    if (drag.kind === "box") input.box = { from: drag.start, to: point };
    else input.pan = { zoom: drag.zoom, from: drag.from, to: point };
    input.rest = true;
    endDrag(false);
  }

  // A drag the browser took away (a cancel, or capture lost before a release) moves nothing more:
  // its queued pan goes with it. After a release the drag has already ended, and its pan stays.
  const onDragLost = () => {
    if (drag?.kind === "pan" && queued) queued.pan = null;
    endDrag(true);
  };

  function onDoubleClick() {
    if (getScene()?.zoom) reset();
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const scene = getScene();
    if (!scene?.zoom || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "0") {
      event.preventDefault();
      reset();
      return;
    }
    const step = KEY_ZOOM[event.key];
    if (step === undefined) return;
    event.preventDefault();
    queue(scene).step *= step;
  }

  const wheelOptions = { passive: false };
  frame.addEventListener("wheel", onWheel, wheelOptions);
  frame.addEventListener("pointerdown", onPointerDown);
  frame.addEventListener("pointermove", onPointerMove);
  frame.addEventListener("pointerup", onPointerUp);
  frame.addEventListener("pointercancel", onDragLost);
  frame.addEventListener("lostpointercapture", onDragLost);
  frame.addEventListener("dblclick", onDoubleClick);
  frame.addEventListener("keydown", onKeyDown);
  return {
    busy: () => Boolean(drag || turning),
    reset,
    cancel,
    destroy() {
      cancel();
      clearTimeout(turning);
      turning = 0;
      frame.removeEventListener("wheel", onWheel);
      frame.removeEventListener("pointerdown", onPointerDown);
      frame.removeEventListener("pointermove", onPointerMove);
      frame.removeEventListener("pointerup", onPointerUp);
      frame.removeEventListener("pointercancel", onDragLost);
      frame.removeEventListener("lostpointercapture", onDragLost);
      frame.removeEventListener("dblclick", onDoubleClick);
      frame.removeEventListener("keydown", onKeyDown);
    },
  };
}
