import assert from "node:assert/strict";
import test from "node:test";

import {
  PopoverDelay,
  TOOLTIP_SHOW_DELAY_MS,
  TOOLTIP_SWITCH_WINDOW_MS,
} from "../../src/superglm/editor/app/views/popover.js";

function fakeScheduler() {
  /** @type {Map<number, {next:() => void, delay:number}>} */
  const timers = new Map();
  let nextId = 1;

  return {
    /** @param {() => void} next @param {number} delay */
    setTimer(next, delay) {
      const id = nextId;
      nextId += 1;
      timers.set(id, { next, delay });
      return id;
    },
    /** @param {number} id */
    clearTimer(id) {
      timers.delete(id);
    },
    /** Run the timers set for ``delay``. */
    flush(delay = TOOLTIP_SHOW_DELAY_MS) {
      for (const [id, timer] of [...timers]) {
        if (timer.delay !== delay) continue;
        timers.delete(id);
        timer.next();
      }
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

function delayed() {
  const scheduler = fakeScheduler();
  /** @type {string[]} */
  const events = [];
  const delay = new PopoverDelay({
    setTimer: scheduler.setTimer,
    clearTimer: scheduler.clearTimer,
    onShow: () => events.push("show"),
    onHide: () => events.push("hide"),
  });
  return { scheduler, events, delay };
}

test("moving from an open tooltip to a neighbour swaps at once, with no second wait", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.pointerLeave();
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide", "show"]);
});

test("after the switch window has passed the next tooltip waits again", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.pointerLeave();
  scheduler.flush(TOOLTIP_SWITCH_WINDOW_MS);
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide"]);
  scheduler.flush();
  assert.deepEqual(events, ["show", "hide", "show"]);
});

test("closing a tooltip on purpose makes the next one wait", () => {
  const { scheduler, events, delay } = delayed();
  delay.pointerEnter();
  scheduler.flush();
  delay.escape();
  delay.pointerEnter();
  assert.deepEqual(events, ["show", "hide"]);
});

test("a tooltip that never opened does not make the next one instant", () => {
  const { events, delay } = delayed();
  delay.pointerEnter();
  delay.pointerLeave();
  delay.pointerEnter();
  assert.deepEqual(events, []);
});
