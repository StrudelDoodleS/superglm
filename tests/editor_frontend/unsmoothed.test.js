import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  UNSMOOTHED_BUSY,
  UNSMOOTHED_HELP,
  unsmoothedEntry,
  unsmoothedRange,
  unsmoothedRuns,
  unsmoothedSeries,
  unsmoothedToggle,
  withUnsmoothed
} from "../../src/superglm/editor/app/unsmoothed.js";
import { HELP_SECTIONS } from "../../src/superglm/editor/app/views/help_content.js";

const REFUSAL = "With its smoothing off, the curve of 'age' is not determined.";

/**
 * A spline's line on its own grid, with the note the fit gave it.
 * @param {Record<string, unknown>} [overrides]
 * @returns {import("../../src/superglm/editor/app/api/contracts.js").UnsmoothedLine}
 */
function splineLine(overrides = {}) {
  return /** @type {any} */ ({
    term: "age", x: [18, 30, 50], y: [1.2, 0.9, 1.05], note: "The refit stopped early.",
    fit_token: 4, ...overrides
  });
}

/** @param {string} status @param {Record<string, unknown>} [overrides] */
function entry(status, overrides = {}) {
  return /** @type {any} */ ({ fit_token: 4, status, line: null, reason: null, ...overrides });
}

const spline = /** @type {any} */ ({ unsmoothed: true });

test("the toggle shows on a smoothed term only, pressed while the choice is on", () => {
  assert.equal(unsmoothedToggle(false, /** @type {any} */ ({ unsmoothed: false }), null).hidden, true);
  assert.equal(unsmoothedToggle(true, undefined, null).hidden, true);
  assert.deepEqual(unsmoothedToggle(false, spline, null), {
    hidden: false, pressed: false, busy: false, disabled: false, body: UNSMOOTHED_HELP
  });
  assert.equal(unsmoothedToggle(true, spline, null).pressed, true);
});

test("the toggle is busy while its fit runs and off, saying why, where the fit was refused", () => {
  assert.deepEqual(unsmoothedToggle(true, spline, entry("running")), {
    hidden: false, pressed: true, busy: true, disabled: false, body: UNSMOOTHED_BUSY
  });
  assert.deepEqual(unsmoothedToggle(true, spline, entry("refused", { reason: REFUSAL })), {
    hidden: false, pressed: true, busy: false, disabled: true, body: REFUSAL
  });
  // A failed request is said, but the toggle stays on hand to try again.
  const failed = unsmoothedToggle(true, spline, entry("failed", { reason: "HTTP 502" }));
  assert.deepEqual([failed.disabled, failed.body], [false, "HTTP 502"]);
  // A line's note joins the hover text.
  const ready = unsmoothedToggle(true, spline, entry("ready", { line: splineLine() }));
  assert.equal(ready.body, `${UNSMOOTHED_HELP} The refit stopped early.`);
  // Turned off, the toggle is plain whatever the fit did.
  assert.deepEqual(unsmoothedToggle(false, spline, entry("refused", { reason: REFUSAL })), {
    hidden: false, pressed: false, busy: false, disabled: false, body: UNSMOOTHED_HELP
  });
});

test("an entry belongs to its fit, and a slow answer for an older fit does not replace it", () => {
  const entries = { band: entry("ready", { fit_token: 5 }) };
  assert.equal(unsmoothedEntry(entries, "band", 5), entries.band);
  assert.equal(unsmoothedEntry(entries, "band", 6), null);
  assert.equal(unsmoothedEntry(entries, "age", 5), null);
  assert.equal(withUnsmoothed(entries, "band", entry("ready", { fit_token: 4 })), entries);
  const newer = withUnsmoothed(entries, "band", entry("running", { fit_token: 6 }));
  assert.equal(newer.band.fit_token, 6);
  assert.equal(entries.band.fit_token, 5);
});

test("a spline's line is its own grid, a value that is not finite a gap in it", () => {
  assert.deepEqual(unsmoothedSeries(splineLine()), { x: [18, 30, 50], y: [1.2, 0.9, 1.05] });
  assert.deepEqual(unsmoothedSeries(splineLine({ y: [1.2, Infinity, 1.05] })), {
    x: [18, 30, 50], y: [1.2, null, 1.05]
  });
  assert.equal(unsmoothedSeries(null), null);
});

test("the line breaks at a gap, and a level alone between two gaps is a point", () => {
  assert.deepEqual(unsmoothedRuns({ x: [0, 1, 2, 3, 4], y: [0.8, 1, null, 1.1, null] }), [
    { x: [0, 1], y: [0.8, 1] },
    { x: [3], y: [1.1] }
  ]);
  assert.deepEqual(unsmoothedRuns({ x: [0, 1], y: [null, null] }), []);
});

test("the line takes part in the range up to the curve's own range again on each side", () => {
  // Within reach, it is one more series.
  assert.deepEqual(unsmoothedRange(0.75, 1.25, [0.5, null, 1.5]), [0.5, 1.5]);
  // A wild unsmoothed spline stops at the reach and runs off the plot.
  assert.deepEqual(unsmoothedRange(0.75, 1.25, [0.01, 40]), [0.25, 1.75]);
  // A flat curve still leaves the line 0.1 of room.
  const [low, high] = unsmoothedRange(1, 1, [0.5, 3]);
  assert.ok(Math.abs(low - 0.9) < 1e-12 && Math.abs(high - 1.1) < 1e-12);
  assert.deepEqual(unsmoothedRange(0.8, 1.2, [null]), [0.8, 1.2]);
});

test("the toggle's hover text and Help say the same", () => {
  const html = readFileSync(
    new URL("../../src/superglm/editor/app/index.html", import.meta.url), "utf8"
  );
  assert.ok(html.includes(`data-popover-body="${UNSMOOTHED_HELP}"`));
  const section = HELP_SECTIONS.find((candidate) => candidate.title === "Unsmoothed line");
  assert.ok(section?.items?.some((item) => item.endsWith(UNSMOOTHED_HELP)));
});
