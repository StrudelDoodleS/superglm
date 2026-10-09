// @ts-check
// Knots mode's gestures on the chart. Drag a knot along the x-axis to move it
// (it may pass its neighbours, and one dropped too close to another settles on
// the nearest free spot), drag it below the axis to remove it, click the band
// along the axis to add one, click a knot to select it; the arrow keys nudge
// the selected knot, Delete or Backspace removes it and Escape lets it go. On
// a term that takes evenly spaced knots only, none of these move a knot: the
// status line says why, and the count above the chart still works.
// A gesture lives here while it runs and redraws only the knot layer; the
// finished change goes to Python as one structural change.

import {
  addOutcome,
  addSpot,
  clampKnot,
  dropOutcome,
  moveOutcome,
  nudgeKnot,
  removeOutcome
} from "./knots.js";
import { inRemoveZone, knotAt, knotIndex, knotX, onKnotBand } from "./chart/knot_marks.js";

/** @typedef {import('./api/contracts.js').KnotParams} KnotParams */
/** @typedef {import('./chart/knot_marks.js').KnotFrame} KnotFrame */
/** @typedef {import('./chart/knot_marks.js').KnotUi} KnotUi */
/** @typedef {import('./knots.js').KnotOutcome} KnotOutcome */
/**
 * The chart's svg as chart.js leaves it: the knot frame it drew last.
 * @typedef {Pick<SVGSVGElement, "addEventListener"|"removeEventListener"|"setPointerCapture"
 *   |"getScreenCTM"|"getBoundingClientRect"|"focus">
 *   & {viewBox:{baseVal:{x:number, y:number, width:number, height:number}},
 *   createSVGPoint?:()=>DOMPoint, _knotFrame?:KnotFrame|null}} KnotSvg
 */

// A press that moves no further than this, in svg px, is a click, as in interactions.js.
const CLICK_SLOP = 3;
/** @type {Readonly<Record<string, -1|1>>} */
const ARROWS = Object.freeze({ ArrowLeft: -1, ArrowRight: 1 });

/**
 * Bind Knots mode's gestures to the chart.
 * @param {object} context
 * @param {KnotSvg} context.svg
 * @param {()=>boolean} context.active Knots mode is on for a term whose knots can change
 * @param {(params:KnotParams)=>unknown} context.onChange stages the change; resolves truthy
 *   once it is staged
 * @param {()=>void} context.onStatus the status line's sentence changed
 * @param {(ui:Readonly<KnotUi>)=>void} context.redraw redraws the knot layer
 */
export function bindKnotGestures({ svg, active, onChange, onStatus, redraw }) {
  /** @type {KnotUi} */
  const ui = { selected: null, drag: null, hover: null, pending: null };
  /** @type {{x:number, y:number}|null} where a drag started */
  let dragStart = null;
  /** @type {{start:{x:number, y:number}, cancelled:boolean}|null} a press on the band */
  let press = null;
  /** @type {string|null} */
  let message = null;

  /** @returns {KnotFrame|null} */
  function frame() {
    const current = svg._knotFrame ?? null;
    return active() && current?.editing ? current : null;
  }

  function draw() {
    redraw(ui);
  }

  /** @param {string|null} next */
  function say(next) {
    if (next === message) return;
    message = next;
    onStatus();
  }

  /**
   * Stage a finished gesture's change and select the knot it leaves; if the
   * change is not staged, the selection goes back.
   * @param {KnotOutcome|null} outcome
   */
  function apply(outcome) {
    if (outcome === null) return;
    if ("refusal" in outcome) {
      say(outcome.refusal);
      return;
    }
    const before = ui.selected;
    ui.selected = outcome.select;
    // Drawn where the gesture left them until the change is answered, so a
    // dropped knot does not flash back to its old place on the way.
    const positions = "positions" in outcome.params ? outcome.params.positions : null;
    ui.pending = positions ? [...positions].sort((a, b) => a - b) : null;
    draw();
    void Promise.resolve(onChange(outcome.params)).then((staged) => {
      ui.pending = null;
      if (!staged && ui.selected === outcome.select) ui.selected = before;
      draw();
    });
  }

  /** @param {PointerEvent} event */
  function onPointerDown(event) {
    const current = frame();
    if (!current || event.button !== 0 || event.shiftKey || event.ctrlKey || event.metaKey) return;
    const point = svgPoint(svg, event);
    say(null);
    const index = knotAt(current, point);
    const evenOnly = current.axis.evenOnly;
    if (evenOnly && (index !== null || onKnotBand(current, point))) {
      event.preventDefault();
      say(evenOnly);
      return;
    }
    if (index !== null) {
      const x = current.positions[index];
      ui.selected = x;
      ui.drag = { from: x, x, moved: false, remove: false };
      ui.hover = null;
      dragStart = point;
    } else if (onKnotBand(current, point)) {
      press = { start: point, cancelled: false };
    } else {
      if (ui.selected !== null) {
        ui.selected = null;
        draw();
      }
      return;
    }
    event.preventDefault();
    svg.setPointerCapture(event.pointerId);
    svg.focus({ preventScroll: true });
    draw();
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    const current = frame();
    if (!current) return;
    const point = svgPoint(svg, event);
    const drag = ui.drag;
    if (drag && dragStart) {
      drag.moved = drag.moved || movedPastSlop(dragStart, point);
      if (!drag.moved) return;
      drag.x = clampKnot(knotX(current, point.x), current.axis);
      drag.remove = inRemoveZone(current, point);
      draw();
      return;
    }
    if (press) {
      press.cancelled = press.cancelled || movedPastSlop(press.start, point);
      return;
    }
    const hover = !current.axis.evenOnly && onKnotBand(current, point) &&
      knotAt(current, point) === null
      ? addSpot(current.positions, knotX(current, point.x), current.axis)
      : null;
    if (hover === ui.hover) return;
    ui.hover = hover;
    draw();
  }

  /** @param {PointerEvent} event */
  function onPointerUp(event) {
    const current = frame();
    const drag = ui.drag;
    const pressed = press;
    ui.drag = null;
    dragStart = null;
    press = null;
    if (!current) return;
    if (drag) {
      const index = knotIndex(current, drag.from);
      draw();
      if (index !== null) apply(dropOutcome(current.positions, index, drag, current.axis));
      return;
    }
    if (pressed && !pressed.cancelled && !movedPastSlop(pressed.start, svgPoint(svg, event))) {
      const spot = addSpot(current.positions, knotX(current, pressed.start.x), current.axis);
      ui.hover = null;
      apply(addOutcome(current.positions, spot, current.axis));
    }
  }

  function onCancel() {
    if (!ui.drag && !press) return;
    ui.drag = null;
    dragStart = null;
    press = null;
    draw();
  }

  function onPointerLeave() {
    if (ui.drag || ui.hover === null) return;
    ui.hover = null;
    draw();
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const current = frame();
    if (!current || event.altKey || event.ctrlKey || event.metaKey || ui.drag) return;
    const index = knotIndex(current, ui.selected);
    const direction = Object.hasOwn(ARROWS, event.key) ? ARROWS[event.key] : 0;
    if (event.key === "Escape") {
      if (index === null) return;
      event.preventDefault();
      ui.selected = null;
      draw();
    } else if (current.axis.evenOnly && (direction !== 0 || event.key === "Delete" ||
      event.key === "Backspace")) {
      event.preventDefault();
      say(current.axis.evenOnly);
    } else if (direction !== 0 && current.positions.length) {
      event.preventDefault();
      say(null);
      if (index === null) {
        // With none selected, the first arrow picks the knot at that end.
        ui.selected = current.positions[direction > 0 ? 0 : current.positions.length - 1];
        draw();
        return;
      }
      const steps = event.shiftKey ? 10 : 1;
      const x = nudgeKnot(current.positions, index, direction, steps, current.axis);
      if (x !== null) apply(moveOutcome(current.positions, index, x));
    } else if ((event.key === "Delete" || event.key === "Backspace") && index !== null) {
      event.preventDefault();
      apply(removeOutcome(current.positions, index));
    }
  }

  svg.addEventListener("pointerdown", onPointerDown);
  svg.addEventListener("pointermove", onPointerMove);
  svg.addEventListener("pointerup", onPointerUp);
  svg.addEventListener("pointercancel", onCancel);
  svg.addEventListener("lostpointercapture", onCancel);
  svg.addEventListener("pointerleave", onPointerLeave);
  svg.addEventListener("keydown", onKeyDown);

  return Object.freeze({
    /** The gesture in progress, for the chart's full redraw. @returns {Readonly<KnotUi>} */
    ui: () => ui,
    /** The status line's sentence for the last refused gesture, if any. */
    message: () => message,
    /** Say why a toolbar control did nothing. @param {string|null} next */
    say,
    /** Forget the gesture and the selection, as a new term or mode does. */
    reset() {
      ui.selected = null;
      ui.drag = null;
      ui.hover = null;
      dragStart = null;
      press = null;
      message = null;
    },
    destroy() {
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointermove", onPointerMove);
      svg.removeEventListener("pointerup", onPointerUp);
      svg.removeEventListener("pointercancel", onCancel);
      svg.removeEventListener("lostpointercapture", onCancel);
      svg.removeEventListener("pointerleave", onPointerLeave);
      svg.removeEventListener("keydown", onKeyDown);
    }
  });
}

/** @param {{x:number, y:number}} start @param {{x:number, y:number}} point */
function movedPastSlop(start, point) {
  return Math.abs(point.x - start.x) > CLICK_SLOP || Math.abs(point.y - start.y) > CLICK_SLOP;
}

/**
 * A pointer event's place in svg coordinates, through the svg's screen
 * matrix where there is one, as interactions.js reads it.
 * @param {KnotSvg} svg @param {{clientX:number, clientY:number}} event
 */
function svgPoint(svg, event) {
  const matrix = svg.getScreenCTM();
  if (matrix && svg.createSVGPoint) {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const mapped = point.matrixTransform(matrix.inverse());
    return { x: mapped.x, y: mapped.y };
  }
  const box = svg.getBoundingClientRect();
  const viewBox = svg.viewBox.baseVal;
  return {
    x: viewBox.x + (event.clientX - box.left) * viewBox.width / Math.max(box.width, 1e-12),
    y: viewBox.y + (event.clientY - box.top) * viewBox.height / Math.max(box.height, 1e-12)
  };
}
