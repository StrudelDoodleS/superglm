// @ts-check
/**
 * Zoom and pan on a check drawn large, with the Editor chart's gestures. On the chart the wheel
 * works on (see chart_focus.js), the wheel zooms about the pointer, unless the frame is taller
 * than the view, which the wheel then scrolls; Ctrl or Cmd with the wheel, or a pinch, zooms any
 * check at any time. Off it, a plain wheel scrolls the page, the frame included. Shift and drag
 * zooms to a box, a drag on a zoomed check pans it, and a double-click, Home or the 0 key shows
 * it whole; + and - zoom about the middle while it has focus. Every gesture asks for one
 * animation frame, which hands the new domain to `onZoom` once; the view draws the check again
 * at it, and nothing asks Python.
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
 * @property {(domain: Domain|null) => void} onZoom draw the check at `domain`, or whole for null
 */

const SVG_NS = "http://www.w3.org/2000/svg";
/** @type {Readonly<Record<string, number>>} */
const KEY_ZOOM = Object.freeze({ "+": ZOOM_IN, "=": ZOOM_IN, "-": ZOOM_OUT, _: ZOOM_OUT });
// A wheel's lines and pages, in px, for a frame that passes a plain wheel to the page.
const WHEEL_LINE = 16;

/** @param {WheelEvent} event @param {Element} page @returns {number} the wheel's travel in px */
const wheelPx = (event, page) => event.deltaY * (event.deltaMode === 1 ? WHEEL_LINE : event.deltaMode === 2 ? page.clientHeight : 1);

/**
 * What scrolls around `node`: its nearest scrolling ancestor, or the page's own scroller.
 * @param {HTMLElement} node @returns {Element|null}
 */
function scrollerAbove(node) {
  for (let at = node.parentElement; at; at = at.parentElement) {
    if (at.scrollHeight > at.clientHeight && /auto|scroll/.test(getComputedStyle(at).overflowY)) return at;
  }
  const page = node.ownerDocument.scrollingElement;
  return page && page.scrollHeight > page.clientHeight ? page : null;
}

/**
 * Bind zoom and pan to the check in `frame`.
 * @param {HTMLElement} frame @param {CheckZoomOptions} options
 * @returns {{busy(): boolean, destroy(): void}} `busy` while a drag pans or draws a box
 */
export function bindCheckZoom(frame, { getScene, isActive, onZoom }) {
  let pending = 0;
  let factor = 1;
  /** @type {Point} */
  let wheelAt = { x: 0, y: 0 };
  /** @type {{zoom: ZoomFrame, from: Point, to: Point|null}|null} */
  let pan = null;
  /** @type {{start: Point, node: SVGRectElement|null}|null} */
  let box = null;

  /** @returns {ZoomFrame|null} */
  const zoomOf = () => getScene()?.zoom ?? null;

  /** The pointer in the scene's px: the svg's box moves with the frame's scroll. @param {MouseEvent} event @returns {Point|null} */
  function scenePoint(event) {
    const scene = getScene();
    const svg = frame.querySelector("svg");
    const rect = svg?.getBoundingClientRect();
    if (!scene || !rect || !(rect.width > 0 && rect.height > 0)) return null;
    return { x: ((event.clientX - rect.left) * scene.width) / rect.width, y: ((event.clientY - rect.top) * scene.height) / rect.height };
  }

  // The frame's one redraw: the latest pan, then the wheel's ticks since the last frame as one zoom.
  function onFrame() {
    pending = 0;
    if (pan?.to) {
      onZoom(panBy(pan.zoom, pan.from, pan.to));
      pan.to = null;
    }
    const zoom = zoomOf();
    if (factor !== 1 && zoom) onZoom(zoomAbout(zoom, wheelAt, factor));
    factor = 1;
  }

  const schedule = () => {
    if (!pending) pending = requestAnimationFrame(onFrame);
  };

  /** @param {WheelEvent} event */
  function onWheel(event) {
    if (!zoomOf() || event.deltaY === 0) return;
    if (!(event.ctrlKey || event.metaKey)) {
      const scrolls = frame.scrollHeight > frame.clientHeight;
      // The active check zooms, or scrolls its own frame when that is taller than the view.
      if (isActive()) {
        if (scrolls) return;
      } else {
        // Off the active chart a plain wheel is the page's, even over a frame that scrolls.
        if (scrolls) {
          event.preventDefault();
          const page = scrollerAbove(frame);
          page?.scrollBy({ top: wheelPx(event, page) });
        }
        return;
      }
    }
    const point = scenePoint(event);
    if (!point) return;
    event.preventDefault();
    factor *= event.deltaY < 0 ? ZOOM_IN : ZOOM_OUT;
    wheelAt = point;
    schedule();
  }

  /** @param {PointerEvent} event */
  function onPointerDown(event) {
    const zoom = zoomOf();
    const point = event.button === 0 ? scenePoint(event) : null;
    if (!zoom || !point) return;
    if (event.shiftKey) {
      event.preventDefault();
      const node = /** @type {SVGRectElement} */ (frame.ownerDocument.createElementNS(SVG_NS, "rect"));
      node.setAttribute("class", "check-zoom-box");
      frame.querySelector("svg")?.append(node);
      box = { start: point, node };
      drawBox(point);
    } else if (!isWhole(zoom.domain, zoom.base)) {
      // The press is left to focus the plot; the plot's text cannot be selected (assess.css).
      pan = { zoom, from: point, to: null };
      frame.dataset.panning = "true";
    } else {
      return;
    }
    frame.setPointerCapture?.(event.pointerId);
  }

  /** @param {Point} to */
  function drawBox(to) {
    if (!box?.node) return;
    const { start, node } = box;
    node.setAttribute("x", String(Math.min(start.x, to.x)));
    node.setAttribute("y", String(Math.min(start.y, to.y)));
    node.setAttribute("width", String(Math.abs(to.x - start.x)));
    node.setAttribute("height", String(Math.abs(to.y - start.y)));
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    if (!box && !pan) return;
    const point = scenePoint(event);
    if (!point) return;
    if (box) drawBox(point);
    if (pan) {
      pan.to = point;
      schedule();
    }
  }

  function endDrag() {
    box?.node?.remove();
    box = null;
    pan = null;
    delete frame.dataset.panning;
  }

  /** @param {PointerEvent} event */
  function onPointerUp(event) {
    const zoom = zoomOf();
    const point = scenePoint(event);
    if (box && zoom && point) onZoom(zoomToBox(zoom, box.start, point));
    if (pan && point) {
      pan.to = point;
      if (pending) {
        cancelAnimationFrame(pending);
        pending = 0;
      }
      onFrame();
    }
    endDrag();
  }

  function onDoubleClick() {
    if (zoomOf()) onZoom(null);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const zoom = zoomOf();
    if (!zoom || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "0") {
      event.preventDefault();
      onZoom(null);
      return;
    }
    const step = KEY_ZOOM[event.key];
    if (step === undefined) return;
    event.preventDefault();
    onZoom(stepZoom(zoom, step));
  }

  const wheelOptions = { passive: false };
  frame.addEventListener("wheel", onWheel, wheelOptions);
  frame.addEventListener("pointerdown", onPointerDown);
  frame.addEventListener("pointermove", onPointerMove);
  frame.addEventListener("pointerup", onPointerUp);
  frame.addEventListener("pointercancel", endDrag);
  frame.addEventListener("lostpointercapture", endDrag);
  frame.addEventListener("dblclick", onDoubleClick);
  frame.addEventListener("keydown", onKeyDown);
  return {
    busy: () => Boolean(box || pan),
    destroy() {
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      endDrag();
      frame.removeEventListener("wheel", onWheel, wheelOptions);
      frame.removeEventListener("pointerdown", onPointerDown);
      frame.removeEventListener("pointermove", onPointerMove);
      frame.removeEventListener("pointerup", onPointerUp);
      frame.removeEventListener("pointercancel", endDrag);
      frame.removeEventListener("lostpointercapture", endDrag);
      frame.removeEventListener("dblclick", onDoubleClick);
      frame.removeEventListener("keydown", onKeyDown);
    },
  };
}
