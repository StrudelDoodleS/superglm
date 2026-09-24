// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_POLYNOMIAL_DEGREE,
  MAX_SEGMENT_DEGREE,
  addBreak,
  breakX,
  cycleDegree,
  degreeName,
  draftProblem,
  formHint,
  initialDraft,
  isTransformable,
  maxSegmentDegree,
  moveBreak,
  removeBreak,
  setPolynomialDegree,
  snapBreak,
  stepBreak,
  transformPayload
} from "../../src/superglm/editor/app/breaks.js";

const BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"];

function orderedTerm(piecewise = null) {
  return {
    term_type: "ordered categorical",
    levels: BANDS.slice(),
    x: BANDS.map((_, i) => i),
    transform: { axis: BANDS.slice(), piecewise }
  };
}

function numericTerm() {
  const x = Array.from({ length: 201 }, (_, i) => i * 0.05);
  return { term_type: "spline", levels: null, x, transform: { axis: null, piecewise: null } };
}

function yearsTerm() {
  const x = Array.from({ length: 201 }, (_, i) => 1990 + i * 0.15);
  return { term_type: "spline", levels: null, x, transform: { axis: null, piecewise: null } };
}

function draft(breaks, degrees, form = "piecewise") {
  return { form, breaks, degrees, degree: 3 };
}

test("snapBreak on an ordered term picks the nearest interior band", () => {
  const term = orderedTerm();
  assert.equal(snapBreak(term, 2.4), "B3");
  assert.equal(snapBreak(term, 0.1), "B2");
  assert.equal(snapBreak(term, 7.0), "B7");
  assert.equal(snapBreak(term, -3), "B2");
});

test("snapBreak on a numeric term rounds to three significant figures of the span, inside it", () => {
  const term = numericTerm();
  assert.equal(snapBreak(term, 3.14159), 3.1);
  assert.equal(snapBreak(term, -1), null);
  assert.equal(snapBreak(term, 0), null);
  assert.equal(snapBreak(term, 10), null);
  assert.equal(snapBreak(term, 9.999), null);
});

test("breakX maps a band to its axis position and a value to itself", () => {
  assert.equal(breakX(orderedTerm(), "B4"), 3);
  assert.equal(breakX(numericTerm(), 2.5), 2.5);
});

test("addBreak keeps breaks sorted, splits the segment and copies its degree within the cap", () => {
  const term = orderedTerm();
  let current = initialDraft(term);
  assert.deepEqual(current, { form: "piecewise", breaks: [], degrees: [1], degree: 3 });
  current = addBreak(term, current, "B5");
  assert.deepEqual(current.breaks, ["B5"]);
  assert.deepEqual(current.degrees, [1, 1]);
  current = addBreak(term, current, "B3");
  assert.deepEqual(current.breaks, ["B3", "B5"]);
  assert.deepEqual(current.degrees, [1, 1, 1]);
  const quadraticLast = { ...current, degrees: [1, 1, 2] };
  const split = addBreak(term, quadraticLast, "B7");
  assert.deepEqual(split.breaks, ["B3", "B5", "B7"]);
  // B7 to B8 spans one band, so that half can only be linear.
  assert.deepEqual(split.degrees, [1, 1, 2, 1]);
  const flatFirst = { ...current, degrees: [0, 1, 1] };
  const splitFlat = addBreak(term, flatFirst, "B2");
  assert.deepEqual(splitFlat.breaks, ["B2", "B3", "B5"]);
  assert.deepEqual(splitFlat.degrees, [0, 1, 1, 1]);
  assert.strictEqual(addBreak(term, current, "B3"), current);
  assert.strictEqual(addBreak(term, current, null), current);
});

test("addBreak on a numeric term sorts by value", () => {
  const term = numericTerm();
  let current = addBreak(term, initialDraft(term), 6.5);
  current = addBreak(term, current, 2.25);
  assert.deepEqual(current.breaks, [2.25, 6.5]);
  assert.deepEqual(current.degrees, [1, 1, 1]);
});

test("moveBreak never crosses or lands on a neighbour", () => {
  const term = orderedTerm();
  const current = draft(["B3", "B5"], [1, 1, 1]);
  assert.strictEqual(moveBreak(term, current, 0, "B5"), current);
  assert.strictEqual(moveBreak(term, current, 0, "B6"), current);
  assert.strictEqual(moveBreak(term, current, 1, "B3"), current);
  assert.strictEqual(moveBreak(term, current, 1, null), current);
  assert.deepEqual(moveBreak(term, current, 0, "B4").breaks, ["B4", "B5"]);
  assert.deepEqual(moveBreak(term, current, 1, "B7").breaks, ["B3", "B7"]);
  assert.deepEqual(moveBreak(term, current, 0, "B2").breaks, ["B2", "B5"]);
  const numeric = numericTerm();
  const values = draft([2, 5], [1, 1, 1]);
  assert.strictEqual(moveBreak(numeric, values, 0, 5), values);
  assert.strictEqual(moveBreak(numeric, values, 0, 6), values);
  assert.deepEqual(moveBreak(numeric, values, 0, 3).breaks, [3, 5]);
});

test("stepBreak moves one band or one grid step", () => {
  const term = orderedTerm();
  assert.equal(stepBreak(term, "B3", 1), "B4");
  assert.equal(stepBreak(term, "B3", -1), "B2");
  assert.equal(stepBreak(term, "B2", -1), null);
  assert.equal(stepBreak(term, "B7", 1), null);
  const numeric = numericTerm();
  assert.equal(stepBreak(numeric, 3, 1), 3.1);
  assert.equal(stepBreak(numeric, 0.05, -1), null);
});

test("an offset numeric axis keeps a grid step between breaks, not only round values", () => {
  const years = yearsTerm();
  assert.equal(snapBreak(years, 2003.14159), 2003.1);
  assert.equal(snapBreak(years, 2003), 2003);
  assert.equal(stepBreak(years, 2003, 1), 2003.1);
  assert.equal(stepBreak(years, 2003.1, -1), 2003);
  assert.equal(snapBreak({ ...years, x: [10000, 40000] }, 12345), 12300);
});

test("moving a break lowers a segment degree its new span can't hold", () => {
  const moved = moveBreak(orderedTerm(), draft(["B4"], [3, 3]), 0, "B2");
  assert.deepEqual(moved.breaks, ["B2"]);
  assert.deepEqual(moved.degrees, [1, 3]);
});

test("removeBreak merges the two segments at the higher degree", () => {
  const term = orderedTerm();
  const current = draft(["B3", "B5"], [0, 2, 1]);
  const removed = removeBreak(term, current, 0);
  assert.deepEqual(removed.breaks, ["B5"]);
  assert.deepEqual(removed.degrees, [2, 1]);
  const last = removeBreak(term, current, 1);
  assert.deepEqual(last.breaks, ["B3"]);
  assert.deepEqual(last.degrees, [0, 2]);
});

test("segment degree caps at min(3, span) on bands and 1 on a numeric axis", () => {
  const term = orderedTerm();
  assert.equal(MAX_SEGMENT_DEGREE, 3);
  assert.equal(maxSegmentDegree(term, draft(["B4"], [1, 1]), 0), 3);
  assert.equal(maxSegmentDegree(term, draft(["B4"], [1, 1]), 1), 3);
  assert.equal(maxSegmentDegree(term, draft(["B2"], [1, 1]), 0), 1);
  assert.equal(maxSegmentDegree(term, draft(["B3"], [1, 1]), 0), 2);
  assert.equal(maxSegmentDegree(term, draft(["B3", "B5"], [1, 1, 1]), 1), 2);
  assert.equal(maxSegmentDegree(numericTerm(), draft([4], [1, 1]), 0), 1);
});

test("cycleDegree cycles 1, 2, 3, 0 within the segment cap", () => {
  const term = orderedTerm();
  let current = draft(["B4"], [1, 1]);
  const seen = [];
  for (let i = 0; i < 4; i++) {
    current = cycleDegree(term, current, 0);
    seen.push(current.degrees[0]);
  }
  assert.deepEqual(seen, [2, 3, 0, 1]);
  assert.deepEqual(current.degrees, [1, 1]);
  const narrow = cycleDegree(term, draft(["B2"], [1, 1]), 0);
  assert.deepEqual(narrow.degrees, [0, 1]);
  assert.deepEqual(cycleDegree(term, narrow, 0).degrees, [1, 1]);
});

test("setPolynomialDegree stays within 1 to 5", () => {
  assert.equal(MAX_POLYNOMIAL_DEGREE, 5);
  const current = draft([], [1], "polynomial");
  assert.equal(setPolynomialDegree(current, 4).degree, 4);
  assert.equal(setPolynomialDegree(current, 9).degree, 5);
  assert.equal(setPolynomialDegree(current, 0).degree, 1);
});

test("draftProblem mirrors the library's piecewise rules with fixed messages", () => {
  const term = orderedTerm();
  assert.equal(draftProblem(term, draft([], [1])), "Add at least one break.");
  assert.equal(draftProblem(term, draft([], [1], "spline")), "Add at least one break.");
  assert.equal(draftProblem(term, draft([], [1], "polynomial")), null);
  assert.equal(
    draftProblem(term, draft(["B4"], [0, 0])),
    "Give at least one segment a degree: all-flat is a constant."
  );
  assert.equal(
    draftProblem(term, draft(["B3", "B5"], [0, 0, 1])),
    "Two flat segments in a row make one plateau: remove the break between them."
  );
  assert.equal(draftProblem(term, draft(["B3", "B5"], [0, 1, 0])), null);
  assert.equal(draftProblem(term, draft(["B4"], [1, 1])), null);
  assert.equal(draftProblem(numericTerm(), draft([4], [1, 1])), null);
});

test("hints and degree names are the stated strings", () => {
  assert.equal(formHint(draft([], [1])), "Click the plot to add a break.");
  assert.equal(formHint(draft([], [1], "spline")), "Knots sit at the breaks.");
  assert.equal(formHint(draft([], [1], "polynomial")), "One polynomial across the whole axis.");
  assert.deepEqual([0, 1, 2, 3].map(degreeName), ["flat", "linear", "quadratic", "cubic"]);
});

test("transformPayload sends degrees only for an ordered piecewise draft", () => {
  const term = orderedTerm();
  assert.deepEqual(transformPayload("age", term, draft(["B3", "B5"], [0, 2, 1])), {
    term: "age",
    form: "piecewise",
    breaks: ["B3", "B5"],
    degrees: [0, 2, 1]
  });
  assert.deepEqual(transformPayload("mileage", numericTerm(), draft([2.5], [1, 1])), {
    term: "mileage",
    form: "piecewise",
    breaks: [2.5]
  });
  assert.deepEqual(transformPayload("age", term, draft(["B3"], [1, 1], "spline")), {
    term: "age",
    form: "spline",
    breaks: ["B3"]
  });
  const polynomial = { ...draft(["B3"], [1, 1], "polynomial"), degree: 2 };
  assert.deepEqual(transformPayload("age", term, polynomial), {
    term: "age",
    form: "polynomial",
    breaks: [],
    degree: 2
  });
});

test("initialDraft loads a term's current piecewise breaks and degrees", () => {
  const piecewise = { breaks: ["B3", "B6"], degrees: [0, 2, 1] };
  const loaded = initialDraft(orderedTerm(piecewise));
  assert.deepEqual(loaded, {
    form: "piecewise",
    breaks: ["B3", "B6"],
    degrees: [0, 2, 1],
    degree: 3
  });
  assert.notStrictEqual(loaded.breaks, piecewise.breaks);
  assert.notStrictEqual(loaded.degrees, piecewise.degrees);
});

test("isTransformable follows the transform payload", () => {
  assert.equal(isTransformable(orderedTerm()), true);
  assert.equal(isTransformable(numericTerm()), true);
  assert.equal(isTransformable({ term_type: "categorical", transform: null }), false);
  assert.equal(isTransformable(null), false);
});
