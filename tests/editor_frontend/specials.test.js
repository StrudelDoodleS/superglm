import assert from "node:assert/strict";
import test from "node:test";

import {
  DECLARED_SPECIAL,
  REFERENCE_STAYS,
  freeLevelMarks,
  freeLevelsShown,
  specialActions,
  waitingSpecials
} from "../../src/superglm/editor/app/specials.js";

/**
 * An ordered term with bands A to E, E special and returnable, Z declared special.
 * @param {Record<string, unknown>} [overrides]
 * @returns {import("../../src/superglm/editor/app/api/contracts.js").TermPayload}
 */
function ordered(overrides = {}) {
  return /** @type {any} */ ({
    term_type: "ordered categorical",
    levels: ["A", "B", "C", "D", "E", "Z"],
    reference: { level: "B", policy: "most_exposed" },
    shape: { available: true, reason: null, ranges: [], support: null, specials: ["E", "Z"], returnable: ["E"] },
    pending: { groups: null, ranges: [], reference: null, specials: null },
    ...overrides
  });
}

test("Make special shows for levels on the curve, Back on the curve for special ones", () => {
  const term = ordered();
  assert.deepEqual(specialActions(term, ["C", "D"]).make, { visible: true, enabled: true, reason: null });
  assert.equal(specialActions(term, ["C", "D"]).back.visible, false);
  assert.deepEqual(specialActions(term, ["E"]).back, { visible: true, enabled: true, reason: null });
  assert.equal(specialActions(term, ["E"]).make.visible, false);
  // A mixed selection takes neither.
  const mixed = specialActions(term, ["C", "E"]);
  assert.deepEqual([mixed.make.visible, mixed.back.visible], [false, false]);
});

test("each action says why it is disabled: the reference, or a level the code declares", () => {
  const term = ordered();
  assert.deepEqual(specialActions(term, ["B", "C"]).make, {
    visible: true, enabled: false, reason: REFERENCE_STAYS
  });
  assert.deepEqual(specialActions(term, ["E", "Z"]).back, {
    visible: true, enabled: false, reason: DECLARED_SPECIAL
  });
  // A waiting reference change decides: B can go once C is to be the reference.
  const moved = ordered({ pending: { groups: null, ranges: [], reference: "C", specials: null } });
  assert.deepEqual(specialActions(moved, ["B"]).make, { visible: true, enabled: true, reason: null });
  assert.deepEqual(specialActions(moved, ["C"]).make, {
    visible: true, enabled: false, reason: REFERENCE_STAYS
  });
  // Neither shows on a categorical term, or with nothing selected.
  const categorical = specialActions(ordered({ term_type: "categorical" }), ["C"]);
  assert.deepEqual([categorical.make.visible, categorical.back.visible], [false, false]);
  assert.equal(specialActions(term, []).make.visible, false);
});

test("waiting specials are the levels a waiting change takes off the curve or puts back", () => {
  assert.deepEqual(waitingSpecials(ordered()), { freed: [], returned: [] });
  const waiting = ordered({ pending: { groups: null, ranges: [], reference: null, specials: ["C", "Z"] } });
  assert.deepEqual(waitingSpecials(waiting), { freed: ["C"], returned: ["E"] });
});

test("free-level marks sit at each compared level's point, a collapsed group's once", () => {
  const free = {
    term: "band", levels: ["A", "B", "C", "D"], y: [0.9, 1, 1.3, 1.1],
    curve: [0.95, 1, 1.1, 1.1],
    lower: [0.8, 1, 1.2, 1], upper: [1, 1, 1.4, 1.2], flagged: ["C"],
    confidence: 0.95, z: 2.6, shrunk: false, fit_token: 4, notice: null
  };
  const expanded = freeLevelMarks(free, { x: [0, 1, 2, 3, 4, 5], levels: ["A", "B", "C", "D", "E", "Z"] });
  assert.deepEqual(expanded.map((mark) => [mark.level, mark.x, mark.flagged, mark.curve]), [
    ["A", 0, false, 0.95], ["B", 1, false, 1], ["C", 2, true, 1.1], ["D", 3, false, 1.1]
  ]);
  // In the Collapsed display C and D are one group point, which takes one mark.
  const collapsed = freeLevelMarks(free, {
    x: [0, 1, 2, 3, 4], levels: ["A", "B", "C+D", "E", "Z"], displayIsCollapsed: true,
    displaySourceLevels: [["A"], ["B"], ["C", "D"], ["E"], ["Z"]]
  });
  assert.deepEqual(collapsed.map((mark) => [mark.level, mark.x]), [["A", 0], ["B", 1], ["C", 2]]);
  assert.deepEqual(freeLevelMarks(null, { x: [0], levels: ["A"] }), []);
  // A comparison is shown for its own term and fit only.
  assert.equal(freeLevelsShown(free, "band", 4), true);
  assert.equal(freeLevelsShown(free, "band", 5), false);
  assert.equal(freeLevelsShown(free, "area", 4), false);
});
