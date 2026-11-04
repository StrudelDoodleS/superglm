// @ts-check

import { helpForElement } from "./help_content.js";

export const TOOLTIP_SHOW_DELAY_MS = 350;
// Reaching another target this soon after a tooltip closes shows its tooltip at once.
export const TOOLTIP_SWITCH_WINDOW_MS = 300;

/** @typedef {(callback: () => void, delay: number) => number} SetTimer */
/** @typedef {(timer: number) => void} ClearTimer */
/** @typedef {HTMLElement|SVGElement} PopoverTarget */

export class PopoverDelay {
  /**
   * @param {object} options
   * @param {SetTimer} options.setTimer
   * @param {ClearTimer} options.clearTimer
   * @param {() => void} options.onShow
   * @param {() => void} options.onHide
   */
  constructor({ setTimer, clearTimer, onShow, onHide }) {
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.onShow = onShow;
    this.onHide = onHide;
    /** @type {number | null} */
    this.timer = null;
    /** @type {number | null} */
    this.switchTimer = null;
    this.visible = false;
  }

  pointerEnter() {
    this.cancel();
    if (this.switchTimer !== null) {
      this.show();
      return;
    }
    this.timer = this.setTimer(() => this.show(), TOOLTIP_SHOW_DELAY_MS);
  }

  pointerLeave() {
    this.cancel();
    this.hide();
  }

  focus() {
    this.cancel();
    this.show();
  }

  /** Close on purpose: the next tooltip waits its delay. */
  escape() {
    this.cancel();
    this.hide();
    this.endSwitch();
  }

  show() {
    this.timer = null;
    this.endSwitch();
    if (this.visible) return;
    this.visible = true;
    this.onShow();
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.onHide();
    this.switchTimer = this.setTimer(() => {
      this.switchTimer = null;
    }, TOOLTIP_SWITCH_WINDOW_MS);
  }

  endSwitch() {
    if (this.switchTimer === null) return;
    this.clearTimer(this.switchTimer);
    this.switchTimer = null;
  }

  cancel() {
    if (this.timer === null) return;
    this.clearTimer(this.timer);
    this.timer = null;
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
      "[data-tool], [data-help-operation], [data-help-control], [data-popover-title]",
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

  return Object.freeze({
    close: dismiss,
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
