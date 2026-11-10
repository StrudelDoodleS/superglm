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
  "mean:age": term("mean", "Mean", "average", "age"),
  "scale:age": term("scale", "CV", "spread", "age"),
  "mean:region": term("mean", "Mean", "average", "region"),
  "scale:region": term("scale", "CV", "spread", "region"),
};

test("a payload with parameter roles has parameters; a SuperGLM payload has none", () => {
  assert.equal(hasParameters(LSS_TERMS), true);
  assert.equal(hasParameters({ age: { kind: "spline", term_type: "spline" } }), false);
  assert.equal(hasParameters({}), false);
});

test("groups follow the payload, each parameter first where it first appears", () => {
  assert.deepEqual(parameterGroups(LSS_TERMS), [
    { key: "mean", label: "Mean", role: "average", terms: ["mean:age", "mean:region"] },
    {
      key: "scale", label: "CV", role: "spread",
      terms: ["scale:age", "scale:region"],
    },
  ]);
});

test("a CV group that is listed first stays first", () => {
  const terms = {
    "scale:age": LSS_TERMS["scale:age"],
    "mean:age": LSS_TERMS["mean:age"],
  };
  assert.deepEqual(parameterGroups(terms).map((group) => group.key), ["scale", "mean"]);
});

test("generic wording and a frequency model name their own parameters", () => {
  const generic = {
    "mean:x": term("mean", "Mean", "average", "x"),
    "scale:x": term("scale", "CV", "spread", "x"),
  };
  assert.deepEqual(parameterGroups(generic).map((group) => group.label), ["Mean", "CV"]);
  const counts = {
    "mean:x": term("mean", "Mean", "average", "x"),
    "theta:x": term("theta", "Overdispersion", "spread", "x"),
  };
  assert.deepEqual(
    parameterGroups(counts).map((group) => [group.label, group.role]),
    [["Mean", "average"], ["Overdispersion", "spread"]],
  );
});

test("a term without parameter fields in a mixed payload is left out of every group", () => {
  const groups = parameterGroups({ ...LSS_TERMS, stray: { kind: "spline" } });
  assert.equal(groups.flatMap((group) => group.terms).includes("stray"), false);
});
