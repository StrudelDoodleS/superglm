// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasParameters,
  parameterGroups,
} from "../../src/superglm/editor/app/views/feature_groups.js";

const term = (parameter, label, role, bare) => ({
  kind: "spline",
  term_type: "spline",
  parameter,
  parameter_label: label,
  parameter_role: role,
  term_label: bare,
});

const LSS_TERMS = {
  "mean:age": term("mean", "Average", "average", "age"),
  "scale:age": term("scale", "Spread", "spread", "age"),
  "mean:region": term("mean", "Average", "average", "region"),
  "scale:region": term("scale", "Spread", "spread", "region"),
};

test("a payload with parameter roles has parameters; a SuperGLM payload has none", () => {
  assert.equal(hasParameters(LSS_TERMS), true);
  assert.equal(hasParameters({ age: { kind: "spline", term_type: "spline" } }), false);
  assert.equal(hasParameters({}), false);
});

test("groups follow the payload, each parameter first where it first appears", () => {
  assert.deepEqual(parameterGroups(LSS_TERMS), [
    { key: "mean", label: "Average", role: "average", terms: ["mean:age", "mean:region"] },
    {
      key: "scale", label: "Spread", role: "spread",
      terms: ["scale:age", "scale:region"],
    },
  ]);
});

test("a spread that is listed first stays first", () => {
  const terms = {
    "scale:age": LSS_TERMS["scale:age"],
    "mean:age": LSS_TERMS["mean:age"],
  };
  assert.deepEqual(parameterGroups(terms).map((group) => group.key), ["scale", "mean"]);
});

test("generic wording and a frequency model name their own parameters", () => {
  const generic = {
    "mean:x": term("mean", "Average", "average", "x"),
    "scale:x": term("scale", "Spread", "spread", "x"),
  };
  assert.deepEqual(parameterGroups(generic).map((group) => group.label), ["Average", "Spread"]);
  const counts = {
    "mean:x": term("mean", "Average", "average", "x"),
    "theta:x": term("theta", "Spread", "spread", "x"),
  };
  assert.deepEqual(
    parameterGroups(counts).map((group) => [group.label, group.role]),
    [["Average", "average"], ["Spread", "spread"]],
  );
});

test("a term without parameter fields in a mixed payload is left out of every group", () => {
  const groups = parameterGroups({ ...LSS_TERMS, stray: { kind: "spline" } });
  assert.equal(groups.flatMap((group) => group.terms).includes("stray"), false);
});
