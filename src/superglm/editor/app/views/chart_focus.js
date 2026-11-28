// @ts-check
/**
 * Which chart the mouse wheel works on, as in a notebook's cell outputs: none, until a press
 * inside a chart, or focus reaching it, makes it the active one. A press anywhere else, Escape,
 * or the chart leaving the page lets it go, and the wheel scrolls the page again. At most one
 * chart is active, and it carries `is-wheel-active`, which draws its slight border. The press
 * itself is left to the chart, so selecting and hovering work as before.
 */

import { defineMachine } from "../state/machine.js";

export const ACTIVE_CLASS = "is-wheel-active";

/** The wheel focus: idle, or one chart active; activating another moves it there. */
export const CHART_FOCUS = defineMachine({
  name: "chart wheel focus",
  initial: "idle",
  moves: {
    idle: { activate: "active" },
    active: { activate: "active", deactivate: "idle" },
  },
});

/** @typedef {'idle'|'active'} FocusState */
/**
 * @typedef {Object} ChartFocus
 * @property {(chart: Element, parts?: Element[]) => () => void} register a chart, and the
 *   controls outside it that belong to it (its Home button), which a press on activates it too
 * @property {(chart: Element) => boolean} isActive whether the wheel works on `chart`
 * @property {() => void} sync let go of an active chart that has left the page
 * @property {() => void} destroy
 */

/** @param {Element} chart whether it is on the page and drawn */
const onPage = (chart) => chart.isConnected && chart.getClientRects().length > 0;

/**
 * Watch `doc` for the presses, focus and Escape that move the wheel focus.
 * @param {Pick<Document, 'addEventListener'|'removeEventListener'>} doc
 * @returns {ChartFocus}
 */
export function bindChartFocus(doc) {
  /** @type {FocusState} */
  let state = CHART_FOCUS.initial;
  /** @type {Element|null} */
  let active = null;
  /** @type {Map<Element, Element[]>} */
  const charts = new Map();

  /**
   * The one writer of the wheel focus.
   * @param {'activate'|'deactivate'} event @param {Element|null} chart
   */
  function move(event, chart) {
    const next = CHART_FOCUS.next(state, event);
    if (next === null) return;
    const now = next === "active" ? chart : null;
    if (active && active !== now) active.classList.remove(ACTIVE_CLASS);
    state = next;
    active = now;
    active?.classList.add(ACTIVE_CLASS);
  }

  /** @param {EventTarget|null} target @param {boolean} withParts @returns {Element|null} */
  function chartOf(target, withParts) {
    const node = /** @type {Node|null} */ (target);
    if (!node) return null;
    for (const [chart, parts] of charts) {
      if (chart.contains(node) || (withParts && parts.some((part) => part.contains(node)))) return chart;
    }
    return null;
  }

  /** @param {Event} event */
  function onPointerDown(event) {
    const chart = chartOf(event.target, true);
    move(chart ? "activate" : "deactivate", chart);
  }

  /** @param {Event} event */
  function onFocusIn(event) {
    const chart = chartOf(event.target, false);
    if (chart) move("activate", chart);
  }

  /** @param {Event} event */
  function onKeyDown(event) {
    if (/** @type {KeyboardEvent} */ (event).key === "Escape") move("deactivate", null);
  }

  function sync() {
    if (active && !onPage(active)) move("deactivate", null);
  }

  doc.addEventListener("pointerdown", onPointerDown, true);
  doc.addEventListener("focusin", onFocusIn);
  doc.addEventListener("keydown", onKeyDown);
  return {
    register(chart, parts = []) {
      charts.set(chart, parts);
      return () => {
        if (active === chart) move("deactivate", null);
        charts.delete(chart);
      };
    },
    isActive(chart) {
      sync();
      return active === chart;
    },
    sync,
    destroy() {
      move("deactivate", null);
      doc.removeEventListener("pointerdown", onPointerDown, true);
      doc.removeEventListener("focusin", onFocusIn);
      doc.removeEventListener("keydown", onKeyDown);
    },
  };
}
