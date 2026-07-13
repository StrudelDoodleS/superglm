import assert from "node:assert/strict";
import test from "node:test";

import {
  PopoverDelay,
  TOOLTIP_SHOW_DELAY_MS,
} from "../../src/superglm/editor/app/views/popover.js";

function fakeScheduler() {
  /** @type {(() => void) | null} */
  let callback = null;

  return {
    /** @param {() => void} next @param {number} delay */
    setTimer(next, delay) {
      callback = next;
      assert.equal(delay, TOOLTIP_SHOW_DELAY_MS);
      return 1;
    },
    clearTimer() {
      callback = null;
    },
    flush() {
      const next = callback;
      callback = null;
      if (next) next();
    },
  };
}

test("pointer disclosure waits 350ms and pointer dismissal is immediate", () => {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });

  delay.pointerEnter();
  assert.deepEqual(events, []);
  scheduler.flush();
  assert.deepEqual(events, ["show"]);
  delay.pointerLeave();
  assert.deepEqual(events, ["show", "hide"]);
});

test("keyboard focus shows immediately and Escape hides", () => {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });

  delay.focus();
  delay.escape();
  assert.deepEqual(events, ["show", "hide"]);
});
