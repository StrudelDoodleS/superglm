// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  canShowContributions,
  fromDisplay,
  formatDisplay,
  showsRelativity,
  reachableDisplay,
  shiftDisplay,
  toDisplay,
} from "../../src/superglm/editor/app/chart/display.js";
import { shiftedCurve } from "../../src/superglm/editor/app/chart/ordered_spline.js";
import { fmt } from "../../src/superglm/editor/app/format.js";

const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/display_transforms.json", import.meta.url), "utf8")
);
const EPS = Number.EPSILON;

test("relativity is exp/log with the 1e-12 floor, bit for bit", () => {
  for (const e of [-3, -0.2, 0, 0.7, 4]) assert.equal(toDisplay(undefined, e), Math.exp(e));
  assert.equal(fromDisplay(undefined, 0), Math.log(1e-12));
  for (const v of [0.4, 1, 2.5]) assert.equal(fromDisplay(undefined, v), Math.log(v));
  const spec = { kind: "relativity", reference_eta: 0, floor: 0 };
  assert.equal(toDisplay(spec, 0.7), Math.exp(0.7));
  assert.equal(fromDisplay(spec, -1), Math.log(1e-12));
});

test("every transform matches the Python fixture", () => {
  for (const [kind, row] of Object.entries(FIXTURE)) {
    const spec = { kind, reference_eta: row.reference_eta, floor: row.floor };
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

test("out-of-range spreads clamp to finite effects", () => {
  const lognormal = { kind: "lognormal_cv", reference_eta: -0.7, floor: 0.05 };
  const theta = { kind: "inverse_theta", reference_eta: 1, floor: 0 };
  assert.ok(Number.isFinite(fromDisplay(lognormal, 0)));
  assert.ok(Number.isFinite(fromDisplay(theta, -1)));
  assert.ok(Number.isFinite(fromDisplay(theta, 0)));
  assert.ok(toDisplay(theta, fromDisplay(theta, -1)) > 0);
  assert.ok(toDisplay(lognormal, fromDisplay(lognormal, 0)) >= 0);
});

test("a reachable value is one the transform can return", () => {
  assert.equal(reachableDisplay(undefined, -2), 1e-12);
  assert.equal(reachableDisplay(undefined, 0.3), 0.3);
  const theta = { kind: "inverse_theta", reference_eta: 1, floor: 0 };
  assert.ok(reachableDisplay(theta, -1) > 0);
  const lognormal = { kind: "lognormal_cv", reference_eta: -0.7, floor: 0.05 };
  const lowest = reachableDisplay(lognormal, 0);
  assert.ok(lowest > 0);
  assert.ok(lowest <= reachableDisplay(lognormal, 0.5));
  assert.ok(Math.abs(reachableDisplay(lognormal, 0.5) - 0.5) <= 64 * EPS);
});

test("shifting a display value moves its effect by the weighted delta", () => {
  assert.equal(shiftDisplay(undefined, 2, 0.5, 0.3), 2 * Math.exp(0.5 * 0.3));
  for (const [kind, row] of Object.entries(FIXTURE)) {
    const spec = { kind, reference_eta: row.reference_eta, floor: row.floor };
    const zero = row.effects.indexOf(0);
    assert.ok(zero >= 0, `${kind} fixture has a zero effect`);
    const base = row.display[zero];
    const shifted = shiftDisplay(spec, base, 1, 0.25);
    const expected = toDisplay(spec, 0.25);
    assert.ok(Math.abs(shifted - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), kind);
  }
});

test("formatDisplay renders relativity as the chart does today", () => {
  for (const v of [0.0004, 0.05, 0.5, 1, 2.2, 13.7, 120]) {
    assert.equal(formatDisplay({}, v), fmt(v));
    assert.equal(formatDisplay({ y_format: "relativity" }, v), fmt(v));
  }
  assert.equal(formatDisplay({ y_format: "times" }, 2.2), "2.2×");
  assert.equal(formatDisplay({ y_format: "plain" }, 0.456), fmt(0.456));
  assert.equal(formatDisplay({ y_format: "plain" }, Number.NaN), "");
});

test("a spread term has no Contrib or Build overlay", () => {
  const controls = { basis: [[1, 0]], build_basis: [[1, 0]] };
  const glm = { controls };
  assert.equal(canShowContributions(glm), true);
  assert.equal(canShowContributions({ ...glm, display: { kind: "relativity", reference_eta: 0, floor: 0 } }), true);
  assert.equal(
    canShowContributions({ ...glm, display: { kind: "gamma_cv", reference_eta: -0.6, floor: 0 } }),
    false
  );
  assert.equal(canShowContributions({ controls: { basis: [] , build_basis: [] } }), false);
  assert.equal(canShowContributions(null), false);
});

test("a handle drag shifts a spread curve on the effect scale", () => {
  const spec = { kind: "gamma_cv", reference_eta: -0.6, floor: 0 };
  const base = [0.4, 0.55, 0.9];
  const row = [0, 0.5, 1];
  const delta = 0.3;
  const moved = shiftedCurve(base, row, delta, spec);
  base.forEach((b, i) => {
    const expected = toDisplay(spec, fromDisplay(spec, b) + row[i] * delta);
    assert.ok(Math.abs(moved[i] - expected) <= 64 * EPS * Math.max(1, expected), `index ${i}`);
  });
});

test("only a relativity curve gets the overlays that exponentiate basis terms", () => {
  assert.equal(showsRelativity(undefined), true);
  assert.equal(showsRelativity({}), true);
  assert.equal(showsRelativity({ display: { kind: "relativity", reference_eta: 0, floor: 0 } }), true);
  for (const kind of ["lognormal_cv", "gamma_cv", "inverse_theta"]) {
    assert.equal(showsRelativity({ display: { kind, reference_eta: 0, floor: 0 } }), false, kind);
  }
});
