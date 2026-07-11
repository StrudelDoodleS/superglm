// @ts-check

import { helpForElement } from "./help_content.js";

export const TOOLTIP_SHOW_DELAY_MS = 350;

/** @typedef {(callback: () => void, delay: number) => number} SetTimer */
/** @typedef {(timer: number) => void} ClearTimer */

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
    this.visible = false;
  }

  pointerEnter() {
    this.cancel();
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

  escape() {
    this.cancel();
    this.hide();
  }

  show() {
    this.timer = null;
    if (this.visible) return;
    this.visible = true;
    this.onShow();
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    this.onHide();
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
  /** @type {HTMLElement | null} */
  let target = null;

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

  /** @param {EventTarget | null} node @returns {HTMLElement | null} */
  function candidate(node) {
    if (!(node instanceof Element)) return null;
    const closest = node.closest(
      "[data-tool], [data-help-operation], [data-popover-title]",
    );
    return closest instanceof HTMLElement ? closest : null;
  }

  /** @param {HTMLElement | null} next @param {boolean} immediate */
  function setTarget(next, immediate) {
    if (next !== target) {
      delay.pointerLeave();
      if (target) target.removeAttribute("aria-describedby");
      target = next;
    }

    if (!target || !helpForElement(target)) {
      delay.pointerLeave();
    } else if (immediate) {
      delay.focus();
    } else {
      delay.pointerEnter();
    }
  }

  function show() {
    if (!target) return;
    const help = helpForElement(target);
    if (!help) return;

    titleNode.textContent = help.title;
    bodyNode.textContent = help.body;
    popover.hidden = false;
    target.setAttribute("aria-describedby", popover.id);

    const anchor = target.getBoundingClientRect();
    const box = popover.getBoundingClientRect();
    const left = Math.max(
      8,
      Math.min(
        window.innerWidth - box.width - 8,
        anchor.left + anchor.width / 2 - box.width / 2,
      ),
    );
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
    const next = candidate(event.target);
    if (next && next !== target) setTarget(next, false);
  }

  /** @param {Event} event */
  function onPointerOut(event) {
    if (!target) return;
    const related = event instanceof PointerEvent ? event.relatedTarget : null;
    if (related instanceof Node && target.contains(related)) return;
    delay.pointerLeave();
    target = null;
  }

  /** @param {Event} event */
  function onFocusIn(event) {
    const next = candidate(event.target);
    if (next) setTarget(next, true);
  }

  /** @param {Event} event */
  function onFocusOut(event) {
    if (!target) return;
    const related = event instanceof FocusEvent ? event.relatedTarget : null;
    if (related instanceof Node && target.contains(related)) return;
    delay.pointerLeave();
    target = null;
  }

  /** @param {Event} event */
  function onKeyDown(event) {
    if (!(event instanceof KeyboardEvent) || event.key !== "Escape") return;
    delay.escape();
  }

  root.addEventListener("pointerover", onPointerOver);
  root.addEventListener("pointerout", onPointerOut);
  root.addEventListener("focusin", onFocusIn);
  root.addEventListener("focusout", onFocusOut);
  root.addEventListener("keydown", onKeyDown);

  return Object.freeze({
    close: () => delay.escape(),
    isOpen: () => !popover.hidden,
    destroy: () => {
      delay.escape();
      root.removeEventListener("pointerover", onPointerOver);
      root.removeEventListener("pointerout", onPointerOut);
      root.removeEventListener("focusin", onFocusIn);
      root.removeEventListener("focusout", onFocusOut);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}
