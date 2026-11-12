// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  canShowContributions,
  fromDisplay,
  noEffect,
  showsRelativity,
  reachableDisplay,
  shiftDisplay,
  toDisplay,
} from "../../src/superglm/editor/app/chart/display.js";
import { shiftedCurve } from "../../src/superglm/editor/app/chart/ordered_spline.js";
import { fmt, fmtCount } from "../../src/superglm/editor/app/format.js";

const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/display_transforms.json", import.meta.url), "utf8")
);
const EPS = Number.EPSILON;

const THETA = { kind: "inverse_theta" };
const ADDITIVE = { kind: "additive", link: "identity" };

test("relativity is exp/log with the 1e-12 floor, bit for bit", () => {
  for (const e of [-3, -0.2, 0, 0.7, 4]) assert.equal(toDisplay(undefined, e), Math.exp(e));
  assert.equal(fromDisplay(undefined, 0), Math.log(1e-12));
  for (const v of [0.4, 1, 2.5]) assert.equal(fromDisplay(undefined, v), Math.log(v));
  const spec = { kind: "relativity" };
  assert.equal(toDisplay(spec, 0.7), Math.exp(0.7));
  assert.equal(fromDisplay(spec, -1), Math.log(1e-12));
});

test("an overdispersion term is the relativity of 1/theta, exp(-effect), bit for bit", () => {
  for (const e of [-3, -0.2, 0, 0.7, 4]) assert.equal(toDisplay(THETA, e), Math.exp(-e));
  for (const v of [0.4, 1, 2.5]) assert.equal(fromDisplay(THETA, v), -Math.log(v));
  assert.equal(fromDisplay(THETA, 0), -Math.log(1e-12));
});

test("every transform matches the Python fixture", () => {
  for (const [kind, row] of Object.entries(FIXTURE)) {
    const spec = { kind };
    row.effects.forEach((e, i) => {
      const py = row.display[i];
      assert.ok(Math.abs(toDisplay(spec, e) - py) <= 64 * EPS * Math.max(1, Math.abs(py)),
        `${kind} toDisplay(${e})`);
    });
    row.probes.forEach((v, i) => {
      const py = row.effects_from_probes[i];
      assert.ok(Math.abs(fromDisplay(spec, v) - py) <= 64 * EPS * Math.max(1, Math.abs(py)),
        `${kind} fromDisplay(${v})`);
    });
  }
});

test("a value at or below zero clamps to the smallest relativity, never to its mirror", () => {
  for (const spec of [{ kind: "relativity" }, THETA]) {
    const floor = fromDisplay(spec, 0);
    assert.ok(Number.isFinite(floor), spec.kind);
    for (const v of [-3, -1, -0.3, -1e-9]) assert.equal(fromDisplay(spec, v), floor, `${spec.kind} ${v}`);
    for (const v of [-3, -1, -0.3]) assert.equal(reachableDisplay(spec, v), 1e-12, `${spec.kind} ${v}`);
    assert.ok(toDisplay(spec, fromDisplay(spec, -1)) > 0);
  }
});

test("additive maps are identity and unclamped", () => {
  const row = FIXTURE.additive;
  row.effects.forEach((e, i) => assert.equal(toDisplay(ADDITIVE, e), row.display[i], `toDisplay(${e})`));
  row.probes.forEach((v, i) => assert.equal(fromDisplay(ADDITIVE, v), row.effects_from_probes[i], `fromDisplay(${v})`));
  for (const v of [-3, -1e-9, 0, 2.5]) assert.equal(reachableDisplay(ADDITIVE, v), v);
  assert.equal(shiftDisplay(ADDITIVE, -2, 0.5, -3), -3.5);
  const moved = shiftedCurve([-1, 0, 2], [1, 0.5, 0], -4, ADDITIVE);
  assert.deepEqual(moved, [-5, -2, 2]);
});

test("no effect is 1 on a multiplicative curve and 0 on an additive one", () => {
  assert.equal(noEffect(undefined), 1);
  assert.equal(noEffect({ kind: "relativity", link: "floored_log" }), 1);
  assert.equal(noEffect(THETA), 1);
  assert.equal(noEffect(ADDITIVE), 0);
  for (const spec of [undefined, THETA, ADDITIVE]) assert.equal(toDisplay(spec, 0), noEffect(spec));
});

test("a reachable value is one the transform can return", () => {
  assert.equal(reachableDisplay(undefined, -2), 1e-12);
  assert.equal(reachableDisplay(undefined, 0.3), 0.3);
  assert.equal(reachableDisplay(THETA, 0.3), 0.3);
});

test("shifting a display value moves its effect by the weighted delta", () => {
  assert.equal(shiftDisplay(undefined, 2, 0.5, 0.3), 2 * Math.exp(0.5 * 0.3));
  assert.equal(shiftDisplay(THETA, 2, 0.5, 0.3), 2 * Math.exp(-0.5 * 0.3));
  for (const [kind, row] of Object.entries(FIXTURE)) {
    const spec = { kind };
    const zero = row.effects.indexOf(0);
    assert.ok(zero >= 0, `${kind} fixture has a zero effect`);
    const base = row.display[zero];
    const shifted = shiftDisplay(spec, base, 1, 0.25);
    const expected = toDisplay(spec, 0.25);
    assert.ok(Math.abs(shifted - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), kind);
  }
});

test("an overdispersion term has no Contrib or Build overlay", () => {
  const controls = { basis: [[1, 0]], build_basis: [[1, 0]] };
  const glm = { controls };
  assert.equal(canShowContributions(glm), true);
  assert.equal(canShowContributions({ ...glm, display: { kind: "relativity" } }), true);
  assert.equal(canShowContributions({ ...glm, display: THETA }), false);
  assert.equal(canShowContributions({ controls: { basis: [] , build_basis: [] } }), false);
  assert.equal(canShowContributions(null), false);
});

test("a handle drag shifts an overdispersion curve on the effect scale", () => {
  const base = [0.4, 0.55, 0.9];
  const row = [0, 0.5, 1];
  const delta = 0.3;
  const moved = shiftedCurve(base, row, delta, THETA);
  base.forEach((b, i) => {
    const expected = toDisplay(THETA, fromDisplay(THETA, b) + row[i] * delta);
    assert.ok(Math.abs(moved[i] - expected) <= 64 * EPS * Math.max(1, expected), `index ${i}`);
  });
});

test("only a relativity curve gets the overlays that exponentiate basis terms", () => {
  assert.equal(showsRelativity(undefined), true);
  assert.equal(showsRelativity({}), true);
  assert.equal(showsRelativity({ display: { kind: "relativity" } }), true);
  assert.equal(showsRelativity({ display: THETA }), false);
  assert.equal(showsRelativity({ display: ADDITIVE }), false);
  assert.equal(canShowContributions({ controls: { basis: [[1]], build_basis: [[1]] }, display: ADDITIVE }), false);
});

test("fmtCount groups thousands and drops stray decimals, and reads small numbers as fmt does", () => {
  assert.equal(fmtCount(126036.5), "126,037");
  assert.equal(fmtCount(3386), "3,386");
  assert.equal(fmtCount(104402.5), "104,403");
  assert.equal(fmtCount(0), "0");
  assert.equal(fmtCount(255.6), "256");
  assert.equal(fmtCount(42.5), fmt(42.5));
  assert.equal(fmtCount(0.25), fmt(0.25));
  assert.equal(fmtCount(Number.NaN), "");
});
