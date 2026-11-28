// @ts-check
/**
 * The check open large, as mounted in its frame: closed, a message in place of the plot (a check
 * not yet answered, or with no answer for this model), or the plot. One renderer draws it, and is
 * the only writer of what it holds. It is what the frame's other owners hear from: a full draw
 * replaces the plot's svg, which ends every gesture on the one it replaced (check_gestures.js);
 * the readout hears that the same check was drawn again, or that it was closed or another opened
 * (assess_check.js); and the wheel focus hears which check the frame now draws (chart_focus.js).
 * A zoom draws the same plot again in place, which keeps the gesture making it.
 *
 * A frame whose plot runs past its foot, or was scrolled down, says so: `data-overflow` names
 * the edges with more of the plot beyond them, which an inactive frame, one the wheel does not
 * scroll, fades (assess.css).
 */

import { defineMachine } from "../state/machine.js";
import { bindCheckHover, redrawCheckPlot, renderCheckView } from "./assess_check.js";
import { bindCheckZoom } from "./check_gestures.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../chart/check_plots.js').Scene} Scene */
/** @typedef {import('../chart/check_plots.js').Domain} Domain */
/** @typedef {import('./assess_check.js').CheckNodes} CheckNodes */
/** @typedef {import('./chart_focus.js').ChartFocus} ChartFocus */
/** @typedef {{compare: boolean, highlight: boolean, domain?: Domain|null, width: number, height: number}} DrawOptions */
/**
 * What the frame holds: nothing; `report`'s message, drawn at `size`; or `report`'s plot of
 * `payload`, drawn at `size` with `options`, whose scene the hover and the gestures read.
 * @typedef {{state: 'closed'}
 *   | {state: 'message', report: CheckReport, size: string}
 *   | {state: 'plot', report: CheckReport, size: string, payload: CheckPayload, options: DrawOptions,
 *     scene: Scene}} Drawing
 */

/**
 * The drawing's lifecycle: a full draw, of a message or a plot, from any state; a zoom, in place,
 * of a plot alone; and closing what is open.
 */
export const CHECK_DRAWING = defineMachine({
  name: "check drawing",
  initial: "closed",
  moves: {
    closed: { message: "message", plot: "plot" },
    message: { message: "message", plot: "plot", close: "closed" },
    plot: { message: "message", plot: "plot", zoom: "plot", close: "closed" },
  },
});

/** @type {Readonly<Drawing>} */
const CLOSED = Object.freeze({ state: "closed" });

/**
 * Mount the open check in `nodes.frame`, with its hover readout and its zoom and pan.
 * @param {CheckNodes} nodes
 * @param {{focus: Pick<ChartFocus, 'isActive'|'drawn'>, onZoom: (domain: Domain|null) => void}} options
 *   `focus` the page's wheel focus; `onZoom` keep the open check's zoom at `domain`, which then
 *   comes back to `zoom`
 * @returns {{draw(report: CheckReport, payload: CheckPayload|null, options: DrawOptions): void,
 *   zoom(report: CheckReport, domain: Domain|null): boolean, close(): void, size(): string|null,
 *   current(): Readonly<Drawing>, reset(): void, cancel(): void, destroy(): void}} `draw` draws
 *   `report` afresh; `zoom` draws its plot again at `domain`, and is false where no plot of
 *   `report` is drawn to zoom; `size` is the size drawn at; `current` what the frame holds;
 *   `reset` shows the check whole; `cancel` ends every gesture
 */
export function bindCheckDrawing(nodes, { focus, onZoom }) {
  const { frame } = nodes;
  /** @type {Drawing} */
  let drawing = CLOSED;
  const scene = () => (drawing.state === "plot" ? drawing.scene : null);
  const readout = bindCheckHover(frame, scene, { live: nodes.live ?? null });
  const gestures = bindCheckZoom(frame, {
    getScene: scene,
    isActive: () => focus.isActive(frame),
    onZoom,
    onStart: readout.suspend,
    onRest: readout.rest,
  });

  /**
   * The one writer of the drawing: `event` moves it to `value`, when the table takes `event` to
   * `value`'s state, and tells the frame's other owners. A full draw or a close ends every gesture
   * first; then another check, or none, is a close to the readout and a new chart to the focus,
   * and the same one drawn again a redraw to the readout.
   * @param {'message'|'plot'|'zoom'|'close'} event @param {Drawing} value
   */
  function move(event, value) {
    if (CHECK_DRAWING.next(drawing.state, event) !== value.state) return;
    const before = drawing;
    drawing = value;
    const report = value.state === "closed" ? null : value.report;
    if (event !== "zoom") {
      gestures.cancel();
      markOverflow();
      focus.drawn(frame, report);
    }
    if (before.state !== "closed" && before.report === report) readout.redraw();
    else readout.close();
  }

  // The edges of the frame with more of the plot beyond them, on the frame. Its one writer.
  function markOverflow() {
    const above = drawing.state === "plot" && frame.scrollTop > 0;
    const below = drawing.state === "plot" && frame.scrollTop + frame.clientHeight < frame.scrollHeight - 1;
    const value = above && below ? "above below" : above ? "above" : below ? "below" : "";
    if ((frame.dataset.overflow ?? "") === value) return;
    if (value) frame.dataset.overflow = value;
    else delete frame.dataset.overflow;
  }

  frame.addEventListener("scroll", markOverflow, { passive: true });
  return {
    draw(report, payload, options) {
      const drawn = renderCheckView(nodes, report, payload, options);
      const size = `${options.width}x${options.height}`;
      if (drawn && payload) move("plot", { state: "plot", report, size, payload, options, scene: drawn });
      else move("message", { state: "message", report, size });
    },
    zoom(report, domain) {
      if (drawing.state !== "plot" || drawing.report !== report) return false;
      if ((drawing.options.domain ?? null) === domain) return true;
      const options = { ...drawing.options, domain };
      const drawn = redrawCheckPlot(frame, report, drawing.payload, options);
      move("zoom", { ...drawing, options, scene: drawn });
      return true;
    },
    close: () => move("close", CLOSED),
    size: () => (drawing.state === "closed" ? null : drawing.size),
    current: () => drawing,
    reset: gestures.reset,
    cancel: gestures.cancel,
    destroy() {
      frame.removeEventListener("scroll", markOverflow);
      gestures.destroy();
      readout.destroy();
    },
  };
}
