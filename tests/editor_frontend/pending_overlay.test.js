// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  pendingGroupMarks,
  pendingUngroupMarks,
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

test("a fitted group ungrouped in full is marked as leaving, in the group's own slot", () => {
  const term = waiting({ "T01+T05": ["T01", "T05"] }, [
    { label: "T01+T05", indices: [0, 4] },
    { label: "T02+T03", indices: [2, 1] },
  ]);
  assert.deepEqual(pendingGroupMarks(term, expanded), []);
  assert.deepEqual(pendingUngroupMarks(term, expanded), [
    { leaves: "T02+T03", members: ["T02", "T03"], display: [1, 2], slot: 1 },
  ]);
  // An empty mapping: the waiting change regroups the term and leaves no group.
  const dissolved = waiting({}, [{ label: "T02+T03", indices: [1, 2] }]);
  assert.deepEqual(pendingUngroupMarks(dissolved, expanded), [
    { leaves: "T02+T03", members: ["T02", "T03"], display: [1, 2], slot: 0 },
  ]);
});

test("after a partial ungroup the rest of the group is not waiting; the level that leaves is", () => {
  const term = waiting({ "T02+T03": ["T02", "T03"] }, [{ label: "T02+T03+T04", indices: [1, 2, 3] }]);
  assert.deepEqual(pendingGroupMarks(term, expanded), []);
  assert.deepEqual(pendingUngroupMarks(term, expanded), [
    { leaves: "T02+T03+T04", members: ["T04"], display: [3], slot: 0 },
  ]);
});

test("a level pulled into a new waiting group goes with it; the one left alone is leaving", () => {
  const term = waiting({ "T03+T04": ["T03", "T04"] }, [{ label: "T02+T03", indices: [1, 2] }]);
  assert.deepEqual(pendingGroupMarks(term, expanded), [
    { label: "T03+T04", members: ["T03", "T04"], display: [2, 3], slot: 1 },
  ]);
  assert.deepEqual(pendingUngroupMarks(term, expanded), [
    { leaves: "T02+T03", members: ["T02"], display: [1], slot: 0 },
  ]);
});

test("nothing is leaving while no regrouping waits, and drawn collapsed a leaver maps to its group's point", () => {
  const fitted = [{ label: "T02+T03", indices: [1, 2] }];
  assert.deepEqual(pendingUngroupMarks(waiting(null, fitted), expanded), []);
  assert.deepEqual(pendingUngroupMarks({ levels: LEVELS, level_groups: fitted }, expanded), []);
  const collapsed = { x: [0, 1, 2, 3], displayToSourceIndices: [[0], [1, 2], [3], [4]] };
  assert.deepEqual(pendingUngroupMarks(waiting({}, fitted), collapsed)[0].display, [1]);
});

test("an ungroup's bracket says the levels are ungrouped, listed with commas", () => {
  assert.equal(waitingBracketText({ members: ["T02", "T03"], leaves: "T02+T03" }), "T02, T03 ungrouped · waiting");
  assert.equal(waitingBracketText({ members: ["T04"], leaves: "T02+T03+T04" }), "T04 ungrouped · waiting");
  assert.equal(
    waitingBracketText({ members: ["A", "B", "C", "D"], leaves: "A+B+C+D" }),
    "4 levels ungrouped · waiting",
  );
});
