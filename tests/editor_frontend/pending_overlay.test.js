// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  pendingGroupMarks,
  waitingBracketText,
} from "../../src/superglm/editor/app/chart/pending_overlay.js";

const LEVELS = ["T01", "T02", "T03", "T04", "T05"];
const expanded = { x: LEVELS.map((_, i) => i), displayToSourceIndices: LEVELS.map((_, i) => [i]) };

function waiting(groups, levelGroups = []) {
  return {
    levels: LEVELS.slice(),
    level_groups: levelGroups,
    pending: { groups, ranges: [], reference: null },
  };
}

test("each waiting group takes its members' points, in axis order, with the next palette slots", () => {
  const term = waiting({ "T04+T05": ["T05", "T04"], "T02+T03": ["T02", "T03"] });
  assert.deepEqual(pendingGroupMarks(term, expanded), [
    { label: "T02+T03", members: ["T02", "T03"], display: [1, 2], slot: 0 },
    { label: "T04+T05", members: ["T04", "T05"], display: [3, 4], slot: 1 },
  ]);
});

test("a group the fitted term already has is not waiting, and slots follow the fitted groups", () => {
  const term = waiting(
    { "T01+T02": ["T01", "T02"], "T03+T04": ["T03", "T04"] },
    [{ label: "T01+T02", indices: [1, 0] }],
  );
  assert.deepEqual(pendingGroupMarks(term, expanded), [
    { label: "T03+T04", members: ["T03", "T04"], display: [2, 3], slot: 1 },
  ]);
});

test("drawn collapsed, a waiting group maps through the display's source levels", () => {
  const collapsed = { x: [0, 1, 2, 3], displayToSourceIndices: [[0], [1, 2], [3], [4]] };
  const term = waiting({ "T03+T04": ["T03", "T04"] }, [{ label: "T02+T03", indices: [1, 2] }]);
  assert.deepEqual(pendingGroupMarks(term, collapsed)[0].display, [1, 2]);
});

test("levels that look like numbers match as labels, and unknown or lone members draw nothing", () => {
  const term = {
    levels: ["1", "2", "10"],
    pending: { groups: { "1+10": [1, 10], solo: ["2", "99"] }, ranges: [], reference: null },
  };
  const axis = { x: [0, 1, 2], displayToSourceIndices: [[0], [1], [2]] };
  assert.deepEqual(pendingGroupMarks(term, axis), [
    { label: "1+10", members: ["1", "10"], display: [0, 2], slot: 0 },
  ]);
});

test("a term with nothing waiting, or without levels, draws no group", () => {
  assert.deepEqual(pendingGroupMarks({ levels: LEVELS, pending: null }, expanded), []);
  assert.deepEqual(pendingGroupMarks({ levels: LEVELS }, expanded), []);
  const numeric = { levels: null, pending: { groups: { a: ["x", "y"] }, ranges: [], reference: null } };
  assert.deepEqual(pendingGroupMarks(numeric, expanded), []);
});

test("the bracket names up to three members and counts more", () => {
  assert.equal(waitingBracketText({ members: ["B10", "B11"] }), "B10 + B11 · waiting");
  assert.equal(waitingBracketText({ members: ["A", "B", "C"] }), "A + B + C · waiting");
  assert.equal(waitingBracketText({ members: ["A", "B", "C", "D", "E"] }), "5 levels · waiting");
});
