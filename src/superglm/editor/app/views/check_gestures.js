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
 * screen is dropped. A drag belongs to the pointer that began it: another pointer's moves,
 * release, cancel or capture loss move nothing. The check drawn afresh, closed or shown whole
 * (`cancel`, `reset`) ends every gesture: its queued input, its drag with the drag's box, cursor
 * and pointer capture, and the wheel's turn. While the wheel turns, and through a drag, the
 * gesture is busy; `onStart` says when it becomes busy and `onRest` when it is no longer.
 */

import { ZOOM_IN, ZOOM_OUT, isWhole, panBy, stepZoom, zoomAbout, zoomToBox } from "../chart/check_zoom.js";
import { defineMachine } from "../state/machine.js";

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
 * @property {() => void} [onStart] a gesture has begun: the wheel turns or a drag runs
 * @property {() => void} [onRest] no gesture runs any longer: the wheel is at rest and no drag runs
 */
/**
 * The input of one animation frame, on the scene it was given on: a drag's latest pan from where
 * it began, a box let go, the wheel's ticks as one factor about where the last one was, and the
 * keys' steps.
 * @typedef {{scene: Scene, pan: {zoom: ZoomFrame, from: Point, to: Point}|null,
 *   box: {from: Point, to: Point}|null, factor: number, at: Point, step: number}} Queued
 */
/**
 * A drag under way, by the pointer `pointerId`: a pan, from `from` on the zoom frame `zoom`, last
 * at `last`; or a box from `start`, drawn as `node`.
 * @typedef {{kind: 'pan', pointerId: number, zoom: ZoomFrame, from: Point, last: Point}
 *   | {kind: 'box', pointerId: number, start: Point, node: SVGRectElement}} Drag
 */

/**
 * A drag on a check: none, a pan, or a box. A press during a drag starts nothing. A drag is
 * released by its pointer, its last move drawn, or dropped, drawing nothing more: cancelled or
 * its capture lost, or its check drawn afresh, closed or shown whole.
 */
export const CHECK_DRAG = defineMachine({
  name: "check drag",
  initial: "idle",
  moves: {
    idle: { pan: "panning", box: "boxing" },
    panning: { release: "idle", drop: "idle" },
    boxing: { release: "idle", drop: "idle" },
  },
});

/**
 * The wheel on a check: still, or turning until it has been at rest for `WHEEL_REST_MS`; it
 * stops at once when its check is drawn afresh, closed or shown whole.
 */
export const CHECK_WHEEL = defineMachine({
  name: "check wheel",
  initial: "still",
  moves: {
    still: { turn: "turning" },
    turning: { turn: "turning", rest: "still", stop: "still" },
  },
});

/** @typedef {'idle'|'panning'|'boxing'} DragState */
/** @typedef {'still'|'turning'} WheelState */

const SVG_NS = "http://www.w3.org/2000/svg";
/** @type {Readonly<Record<string, number>>} */
const KEY_ZOOM = Object.freeze({ "+": ZOOM_IN, "=": ZOOM_IN, "-": ZOOM_OUT, _: ZOOM_OUT });
// How long after its last tick a wheel counts as at rest, as on the Editor chart.
const WHEEL_REST_MS = 160;

/**
 * Bind zoom and pan to the check in `frame`.
 * @param {HTMLElement} frame @param {CheckZoomOptions} options
 * @returns {{busy(): boolean, reset(): void, cancel(): void, destroy(): void}} `busy` while the
 *   wheel turns or a drag runs; `reset` shows the check whole; `cancel` ends every gesture, as
 *   the check drawn afresh or closed does
 */
export function bindCheckZoom(frame, { getScene, isActive, onZoom, onStart, onRest }) {
  let frameId = 0;
  /** @type {Queued|null} */
  let queued = null;
  /** @type {DragState} */
  let dragState = CHECK_DRAG.initial;
  /** @type {Drag|null} what the drag under way carries; null while idle */
  let drag = null;
  /** @type {WheelState} */
  let wheelState = CHECK_WHEEL.initial;
  /** @type {ReturnType<typeof setTimeout>|0} the turning wheel's rest timer; 0 while still */
  let turning = 0;

  const busy = () => dragState !== "idle" || wheelState !== "still";

  /** Say so when a move made the gesture busy, or left it at rest. @param {boolean} was busy before it */
  function announce(was) {
    const now = busy();
    if (now !== was) (now ? onStart : onRest)?.();
  }

  /**
   * The one writer of the drag: `pan` or `box` begins one, carrying `payload`, with its pointer
   * captured, and `release` or `drop` lets it go, taking its box off the plot and its capture
   * back. Its cursor and its box are this function's alone. An event the state does not take
   * changes nothing.
   * @param {'pan'|'box'|'release'|'drop'} event @param {Drag|null} payload
   */
  function moveDrag(event, payload) {
    const next = CHECK_DRAG.next(dragState, event);
    if (next === null) return;
    const was = busy();
    if (drag) {
      if (drag.kind === "box") drag.node.remove();
      if (frame.hasPointerCapture?.(drag.pointerId)) frame.releasePointerCapture(drag.pointerId);
    }
    if (next === "panning") frame.dataset.panning = "true";
    else delete frame.dataset.panning;
    dragState = next;
    drag = payload;
    if (drag) {
      if (drag.kind === "box") frame.querySelector("svg")?.append(drag.node);
      frame.setPointerCapture?.(drag.pointerId);
    }
    announce(was);
  }

  /**
   * The one writer of the wheel: each `turn` sets its rest timer afresh, and `rest`, when the
   * timer fires, or `stop` clears it.
   * @param {'turn'|'rest'|'stop'} event
   */
  function moveWheel(event) {
    const next = CHECK_WHEEL.next(wheelState, event);
    if (next === null) return;
    const was = busy();
    clearTimeout(turning);
    turning = next === "turning" ? setTimeout(() => moveWheel("rest"), WHEEL_REST_MS) : 0;
    wheelState = next;
    announce(was);
  }

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
    if (queued?.scene !== scene) queued = { scene, pan: null, box: null, factor: 1, at: { x: 0, y: 0 }, step: 1 };
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
  }

  // Every gesture ended: what is queued dropped, any drag dropped, and the wheel stopped.
  function cancel() {
    if (frameId) cancelAnimationFrame(frameId);
    frameId = 0;
    queued = null;
    moveDrag("drop", null);
    moveWheel("stop");
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
    moveWheel("turn");
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
    const begin = event.shiftKey ? "box" : "pan";
    if (!zoom || !point || CHECK_DRAG.next(dragState, begin) === null) return;
    const { pointerId } = event;
    if (begin === "box") {
      event.preventDefault();
      const node = /** @type {SVGRectElement} */ (frame.ownerDocument.createElementNS(SVG_NS, "rect"));
      node.setAttribute("class", "check-zoom-box");
      moveDrag("box", { kind: "box", pointerId, start: point, node });
      drawBox(point);
    } else if (!isWhole(zoom.domain, zoom.base)) {
      // The press is left to focus the plot; the plot's text cannot be selected (assess.css).
      moveDrag("pan", { kind: "pan", pointerId, zoom, from: point, last: point });
    }
  }

  /** @param {PointerEvent} event whether it is the drag's own pointer's */
  const ofDrag = (event) => drag !== null && event.pointerId === drag.pointerId;

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    if (!drag || !ofDrag(event)) return;
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
    if (!drag || !ofDrag(event)) return;
    const point = scenePoint(event);
    const scene = getScene();
    // Let go, the drag's last move is drawn in the next frame with anything else queued.
    if (point && scene) {
      const input = queue(scene);
      if (drag.kind === "box") input.box = { from: drag.start, to: point };
      else input.pan = { zoom: drag.zoom, from: drag.from, to: point };
    }
    moveDrag("release", null);
  }

  // A drag the browser took away (a cancel, or capture lost before a release) moves nothing more:
  // its queued pan goes with it. After a release the drag has already ended, and its pan stays.
  /** @param {PointerEvent} event */
  const onDragLost = (event) => {
    if (!drag || !ofDrag(event)) return;
    if (drag.kind === "pan" && queued) queued.pan = null;
    moveDrag("drop", null);
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
    busy,
    reset,
    cancel,
    destroy() {
      cancel();
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
