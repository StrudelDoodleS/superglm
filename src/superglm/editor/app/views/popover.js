// @ts-check

import { defineMachine } from "../state/machine.js";
import { helpForElement } from "./help_content.js";

export const TOOLTIP_SHOW_DELAY_MS = 350;
// Reaching another target this soon after a tooltip closes shows its tooltip at once.
export const TOOLTIP_SWITCH_WINDOW_MS = 300;

/** @typedef {(callback: () => void, delay: number) => number} SetTimer */
/** @typedef {(timer: number) => void} ClearTimer */
/** @typedef {HTMLElement|SVGElement} PopoverTarget */

/**
 * A popover's lifecycle: `closed`; `pending`, its show delay running; `open`; and `recent`, closed
 * a moment ago, so that reaching another target shows its help at once. `due` is the show delay
 * ending and `lapse` the switch window; `update` is its anchor written under it, which reads an
 * open popover's help again.
 */
export const POPOVER = defineMachine({
  name: "popover",
  initial: "closed",
  moves: {
    closed: { enter: "pending", focus: "open" },
    pending: { enter: "pending", focus: "open", due: "open", leave: "closed", escape: "closed" },
    open: { enter: "open", focus: "open", update: "open", leave: "recent", escape: "closed" },
    recent: { enter: "open", focus: "open", lapse: "closed", escape: "closed" },
  },
});

export class PopoverDelay {
  /**
   * @param {object} options
   * @param {SetTimer} options.setTimer
   * @param {ClearTimer} options.clearTimer
   * @param {() => void} options.onShow write the help, at opening and at each update while open
   * @param {() => void} options.onHide
   */
  constructor({ setTimer, clearTimer, onShow, onHide }) {
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.onShow = onShow;
    this.onHide = onHide;
    /** @type {'closed'|'pending'|'open'|'recent'} */
    this.state = POPOVER.initial;
    // The timer of `pending` (the show delay) or of `recent` (the switch window).
    /** @type {number | null} */
    this.timer = null;
  }

  get visible() {
    return this.state === "open";
  }

  pointerEnter() {
    this.move("enter");
  }

  pointerLeave() {
    this.move("leave");
  }

  focus() {
    this.move("focus");
  }

  /** Close on purpose: the next tooltip waits its delay. */
  escape() {
    this.move("escape");
  }

  /** The anchor was written under the help: an open popover reads it again. */
  update() {
    this.move("update");
  }

  /**
   * The one writer of the state, its timer and the help shown: `event` makes its move in the
   * table, if it is one. The state's timer ends with it, and a pending popover entered again
   * waits its delay afresh.
   * @param {'enter'|'focus'|'leave'|'escape'|'due'|'lapse'|'update'} event
   */
  move(event) {
    const from = this.state;
    const to = POPOVER.next(from, event);
    if (to === null) return;
    if (this.timer !== null && (to !== from || event === "enter")) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    this.state = to;
    if (to === "pending" && this.timer === null) {
      this.timer = this.setTimer(() => { this.timer = null; this.move("due"); }, TOOLTIP_SHOW_DELAY_MS);
    }
    if (to === "recent") {
      this.timer = this.setTimer(() => { this.timer = null; this.move("lapse"); }, TOOLTIP_SWITCH_WINDOW_MS);
    }
    if (to === "open" && (from !== "open" || event === "update")) this.onShow();
    if (from === "open" && to !== "open") this.onHide();
  }
}

/**
 * Bind one delegated popover controller for the editor document.
 *
 * @param {object} options
 * @param {Document | HTMLElement} options.root
 * @param {HTMLElement} options.popover
 */
export function bindPopovers({ root, popover }) {
  /** @type {PopoverTarget | null} */
  let target = null;
  /** @type {PopoverTarget | null} */
  let hoveredTarget = null;
  /** @type {PopoverTarget | null} */
  let focusedTarget = null;

  /** @param {string} selector @returns {HTMLElement} */
  function requirePopoverPart(selector) {
    const node = popover.querySelector(selector);
    if (!(node instanceof HTMLElement)) {
      throw new Error(`Popover markup requires ${selector}`);
    }
    return node;
  }
  const titleNode = requirePopoverPart("[data-popover-heading]");
  const bodyNode = requirePopoverPart("[data-popover-description]");

  const delay = new PopoverDelay({
    setTimer: window.setTimeout.bind(window),
    clearTimer: window.clearTimeout.bind(window),
    onShow: show,
    onHide: hide,
  });

  /** @param {EventTarget | null} node @returns {PopoverTarget | null} */
  function candidate(node) {
    if (!(node instanceof Element)) return null;
    const closest = node.closest(
      "[data-tool], [data-help-operation], [data-help-control], [data-help-check], [data-popover-title]",
    );
    if (closest instanceof HTMLElement) return closest;
    return typeof SVGElement !== "undefined" && closest instanceof SVGElement ? closest : null;
  }

  /** @param {boolean} immediate */
  function syncTarget(immediate) {
    const next = focusedTarget || hoveredTarget;
    if (next !== target) {
      delay.pointerLeave();
      if (target) target.removeAttribute("aria-describedby");
      target = next;
    }

    if (!target || !helpForElement(target)) {
      delay.pointerLeave();
    } else if (delay.visible) {
      return;
    } else if (immediate || focusedTarget === target) {
      delay.focus();
    } else {
      delay.pointerEnter();
    }
  }

  function show() {
    if (!target?.isConnected) return;
    const help = helpForElement(target);
    if (!help) return;

    titleNode.textContent = help.title;
    bodyNode.textContent = help.body;
    popover.hidden = false;
    target.setAttribute("aria-describedby", popover.id);

    const anchor = target.getBoundingClientRect();
    const box = popover.getBoundingClientRect();
    // Tooltips of one row of buttons share an edge, so moving along it only changes the text.
    const group = target.closest("[data-popover-group]");
    const wanted = group
      ? group.getBoundingClientRect().right - box.width
      : anchor.left + anchor.width / 2 - box.width / 2;
    const left = Math.max(8, Math.min(window.innerWidth - box.width - 8, wanted));
    const below = anchor.bottom + 8;
    const top =
      below + box.height <= window.innerHeight - 8
        ? below
        : anchor.top - box.height - 8;
    popover.style.left = `${left}px`;
    popover.style.top = `${Math.max(8, top)}px`;
  }

  function hide() {
    if (target) target.removeAttribute("aria-describedby");
    popover.hidden = true;
  }

  /** @param {Event} event */
  function onPointerOver(event) {
    // A target re-rendered away sends no pointerout, and its tooltip would stay.
    if (hoveredTarget && !hoveredTarget.isConnected) {
      hoveredTarget = null;
      syncTarget(false);
    }
    const next = candidate(event.target);
    if (!next || next === hoveredTarget) return;
    hoveredTarget = next;
    syncTarget(false);
  }

  /** @param {Event} event */
  function onPointerOut(event) {
    if (!hoveredTarget) return;
    const related = event instanceof PointerEvent ? event.relatedTarget : null;
    if (related instanceof Node && hoveredTarget.contains(related)) return;
    // Between the buttons of a row the tooltip stays until the next one takes over.
    if (related instanceof Node && hoveredTarget.closest("[data-popover-group]")?.contains(related)) {
      return;
    }
    hoveredTarget = null;
    syncTarget(false);
  }

  /** @param {Event} event */
  function onFocusIn(event) {
    const next = candidate(event.target);
    if (!next) return;
    // Focus shows a tooltip only where the browser marks it focus-visible (the keyboard);
    // focus a click moved somewhere shows nothing, and the pointer's own hover still does.
    if (event.target instanceof Element && !event.target.matches(":focus-visible")) return;
    focusedTarget = next;
    syncTarget(true);
  }

  /** @param {Event} event */
  function onFocusOut(event) {
    if (!focusedTarget) return;
    const related = event instanceof FocusEvent ? event.relatedTarget : null;
    if (related instanceof Node && focusedTarget.contains(related)) return;
    focusedTarget = null;
    syncTarget(false);
  }

  function dismiss() {
    delay.escape();
    if (target) target.removeAttribute("aria-describedby");
    target = null;
    hoveredTarget = null;
    focusedTarget = null;
  }

  /** @param {Event} event */
  function onKeyDown(event) {
    if (!(event instanceof KeyboardEvent) || event.key !== "Escape") return;
    dismiss();
  }

  /** @param {Event} event */
  function onClick(event) {
    if (!candidate(event.target)) return;
    dismiss();
  }

  root.addEventListener("pointerover", onPointerOver);
  root.addEventListener("pointerout", onPointerOut);
  root.addEventListener("focusin", onFocusIn);
  root.addEventListener("focusout", onFocusOut);
  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKeyDown);

  /** Whether `node` is drawn: on the page, and neither it nor an ancestor hidden. @param {PopoverTarget} node */
  const drawn = (node) => node.isConnected && !node.closest("[hidden]");

  /**
   * A renderer wrote over anchors: help open on an anchor still drawn reads it again and is
   * placed again beside it, and help whose anchor left the page, or was hidden, closes.
   */
  function anchorUpdated() {
    if (!target) return;
    if (drawn(target)) {
      if (helpForElement(target)) delay.update();
      else syncTarget(false);
      return;
    }
    if (hoveredTarget === target) hoveredTarget = null;
    if (focusedTarget === target) focusedTarget = null;
    syncTarget(false);
  }

  return Object.freeze({
    close: dismiss,
    anchorUpdated,
    isOpen: () => !popover.hidden,
    destroy: () => {
      dismiss();
      root.removeEventListener("pointerover", onPointerOver);
      root.removeEventListener("pointerout", onPointerOut);
      root.removeEventListener("focusin", onFocusIn);
      root.removeEventListener("focusout", onFocusOut);
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}
