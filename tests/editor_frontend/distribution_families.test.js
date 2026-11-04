// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  GAMMA_CV_MAX,
  PROBABILITY_MIN,
  WILSON_HILFERTY_SHAPE,
  lawOf,
} from "../../src/superglm/editor/app/distribution/families.js";
import { gammaP, lnGamma } from "../../src/superglm/editor/app/distribution/math.js";

const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/distribution_laws.json", import.meta.url), "utf8")
);
const EPS = Number.EPSILON;
const ULPS = 1e4;
const LOG = { kind: "log" };
const FLOORED = { kind: "floored_log", floor: 0.01 };
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const SMALLEST_NORMAL = 2 ** -1022;
// cdf and sf of one point are a pair (p, 1 - p), or two halves of one erfc value, so their sum
// differs from 1 by the rounding of the complement and of the sum: 2u = eps.
const SUM_TO_ONE = EPS;

const lawOfRow = (row) => lawOf(row.family, row.eta, row.links);
const labelOf = (row) => `${row.family} eta=${JSON.stringify(row.eta)}`;
const shapeOf = (row) => Math.exp(-2 * row.eta[1]);
const gammaRows = FIXTURE.rows.filter((row) => row.family === "gamma");

/**
 * Simpson's rule on [a, b] with n panels, and its error bound (b - a) max|f(4)| h^4 / 180, where
 * h^4 f(4) is estimated by the fourth difference of the samples.
 */
function simpson(f, a, b, n = 4000) {
  const h = (b - a) / n;
  const values = Array.from({ length: n + 1 }, (_, i) => f(a + i * h));
  let sum = values[0] + values[n];
  for (let i = 1; i < n; i++) sum += values[i] * (i % 2 ? 4 : 2);
  let fourth = 0;
  for (let i = 2; i <= n - 2; i++) {
    const difference = values[i - 2] - 4 * values[i - 1] + 6 * values[i] - 4 * values[i + 1];
    fourth = Math.max(fourth, Math.abs(difference + values[i + 2]));
  }
  return { value: (sum * h) / 3, bound: ((b - a) * fourth) / 180 };
}

/**
 * The mass of the density of log10 Y over [lo, hi], and how far it can be from the exact value:
 * Simpson's own error bound, plus 1e4 eps times the condition number of the mass, which is 1 for
 * the sum itself and |l| f(l) at each limit for the relative rounding of a limit.
 */
function massOver(law, lo, hi) {
  const f = (l) => 10 ** law.log10Density(l);
  const { value, bound } = simpson(f, lo, hi);
  const limits = Math.abs(lo) * f(lo) + Math.abs(hi) * f(hi);
  return { mass: value, tolerance: bound + ULPS * EPS * (1 + limits) };
}

test("the fixture's shape limit is the module's, and the rows lie on the right side of it", () => {
  assert.equal(FIXTURE.exact_shape_limit, WILSON_HILFERTY_SHAPE);
  for (const row of gammaRows) assert.ok(shapeOf(row) <= WILSON_HILFERTY_SHAPE, labelOf(row));
  for (const row of FIXTURE.large_shape_rows) {
    assert.ok(shapeOf(row) > WILSON_HILFERTY_SHAPE, labelOf(row));
  }
  assert.ok(Math.max(...gammaRows.map(shapeOf)) > 0.98 * WILSON_HILFERTY_SHAPE);
});

test("cdf, sf and quantile match the families within 1e4 eps max(1, kappa)", (t) => {
  const worst = { cdf: 0, sf: 0, quantile: 0 };
  for (const row of FIXTURE.rows) {
    const law = lawOfRow(row);
    const evaluate = {
      cdf: (r) => law.cdf(r.y),
      sf: (r) => law.sf(r.y),
      quantile: (r) => law.quantile(r.p),
    };
    for (const name of Object.keys(evaluate)) {
      for (const r of row[name]) {
        const got = evaluate[name](r);
        const bound = ULPS * EPS * Math.max(1, r.kappa) * Math.abs(r.value);
        const err = Math.abs(got - r.value);
        assert.ok(err <= bound, `${name} ${labelOf(row)} ${JSON.stringify(r)}: js ${got}`);
        worst[name] = Math.max(worst[name], err / bound);
      }
    }
  }
  t.diagnostic(`worst error / bound: ${JSON.stringify(worst)} over ${FIXTURE.rows.length} laws`);
});

test("mean is default_prediction to 64 eps; mode follows its condition number", () => {
  for (const row of [...FIXTURE.rows, ...FIXTURE.large_shape_rows]) {
    const law = lawOfRow(row);
    const where = labelOf(row);
    assert.ok(Math.abs(law.mean - row.mean.value) <= 64 * EPS * row.mean.value, `mean ${where}`);
    const bound = 64 * EPS * row.mode.kappa * row.mode.value;
    assert.ok(Math.abs(law.mode - row.mode.value) <= bound, `mode ${where}: ${law.mode}`);
  }
});

test("the median is quantile(0.5) to the bit, at every shape", () => {
  for (const row of [...FIXTURE.rows, ...FIXTURE.large_shape_rows]) {
    const law = lawOfRow(row);
    assert.equal(law.median, law.quantile(0.5), labelOf(row));
  }
});

test("log10Density matches 50-digit values within 1e4 eps (|value| + condition)", (t) => {
  let worst = 0;
  for (const row of [...FIXTURE.rows, ...FIXTURE.large_shape_rows]) {
    const law = lawOfRow(row);
    for (const r of row.log10_density) {
      const bound = ULPS * EPS * (Math.abs(r.value) + r.condition);
      const err = Math.abs(law.log10Density(r.l) - r.value);
      assert.ok(err <= bound, `${labelOf(row)} ${JSON.stringify(r)}: error ${err} > ${bound}`);
      worst = Math.max(worst, err / bound);
    }
  }
  t.diagnostic(`worst error / bound = ${worst.toExponential(2)}`);
});

test("the density of log10 Y holds the mass the quantile function says it does", () => {
  // Between the 1% and 99% quantiles the density integrates to 0.98.
  for (const row of FIXTURE.rows) {
    const law = lawOfRow(row);
    const lo = Math.log10(law.quantile(0.01));
    const hi = Math.log10(law.quantile(0.99));
    const { mass, tolerance } = massOver(law, lo, hi);
    assert.ok(Math.abs(mass - 0.98) <= tolerance, `${labelOf(row)}: mass ${mass} +- ${tolerance}`);
  }
  // Past the shape limit, 12 standard deviations around the mean hold all of it.
  for (const row of FIXTURE.large_shape_rows) {
    const law = lawOfRow(row);
    const cv = Math.exp(row.eta[1]);
    const lo = Math.log10(law.mean * (1 - 12 * cv));
    const hi = Math.log10(law.mean * (1 + 12 * cv));
    const { mass, tolerance } = massOver(law, lo, hi);
    assert.ok(Math.abs(mass - 1) <= tolerance, `${labelOf(row)}: mass ${mass} +- ${tolerance}`);
  }
});

test("the density of Y peaks at the mode, or falls from 0 when the mode is 0", () => {
  /** log10 of the density of Y itself: the density of log10 Y over y ln 10. */
  const logDensity = (law, y) => law.log10Density(Math.log10(y)) - Math.log10(y);
  for (const row of [...FIXTURE.rows, ...FIXTURE.large_shape_rows]) {
    const law = lawOfRow(row);
    if (law.mode > 0) {
      const peak = logDensity(law, law.mode);
      for (const factor of [0.95, 0.99, 1.01, 1.05]) {
        assert.ok(logDensity(law, law.mode * factor) < peak, `${labelOf(row)} x${factor}`);
      }
    } else {
      const heights = [1e-6, 1e-3, 1].map((f) => logDensity(law, f * law.mean));
      assert.ok(heights[0] > heights[1] && heights[1] > heights[2], labelOf(row));
    }
  }
});

test("above the shape limit the gamma is the approximation, and it converges as the shape grows", () => {
  // Measured only, no published bound: the sup of |cdf - P| over z in [-6, 6] is c / k, so going
  // from 1.01e4 to 4e4 cuts it by 1.01e4 / 4e4 = 0.2525 (measured 0.253). A shifted or mis-scaled
  // transform has an error of order k^(-1/2) or 1, and cuts it by 0.5 or 1.
  const mean = 1000;
  const sup = (shape) => {
    const cv = 1 / Math.sqrt(shape);
    const law = lawOf("gamma", [Math.log(mean), Math.log(cv)], [LOG, LOG]);
    let worst = 0;
    let exactPathWouldFail = false;
    for (let z = -6; z <= 6; z += 0.1) {
      const y = mean * (1 + z * cv);
      const x = (y / mean) * shape;
      const exact = gammaP(shape, x);
      const kappa = Math.exp(shape * Math.log(x) - x - lnGamma(shape)) / exact;
      const err = Math.abs(law.cdf(y) - exact);
      worst = Math.max(worst, err);
      exactPathWouldFail ||= err > ULPS * EPS * Math.max(1, kappa) * exact;
    }
    assert.ok(exactPathWouldFail, `shape ${shape} is on the exact path`);
    return worst;
  };
  const [near, far] = [sup(1.01e4), sup(4e4)];
  // 20% allows for the O(k^(-3/2)) term beyond c / k and for the z grid that samples the sup.
  const ratio = 1.2 * (1.01e4 / 4e4);
  assert.ok(far <= ratio * near, `error ${near} at 1.01e4 and ${far} at 4e4`);
});

test("above the shape limit the gamma keeps the properties of a law", () => {
  const mean = 2500;
  const probabilities = [1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-12];
  for (const shape of [1.01e4, 4e4, 1e5, 1e7, 1e12]) {
    const cv = 1 / Math.sqrt(shape);
    const law = lawOf("gamma", [Math.log(mean), Math.log(cv)], [LOG, LOG]);
    const where = `shape ${shape}`;
    assert.equal(law.mean, Math.exp(Math.log(mean)));
    assert.ok(Math.abs(law.mode - mean * (1 - cv * cv)) <= 64 * EPS * law.mode, where);

    // Monotone, and the two tails sum to one.
    let previousCdf = -1;
    let previousSf = 2;
    for (let z = -9; z <= 9; z += 0.25) {
      const y = mean * (1 + z * cv);
      const [c, s] = [law.cdf(y), law.sf(y)];
      assert.ok(c >= previousCdf && s <= previousSf, `${where} z=${z}`);
      assert.ok(Math.abs(c + s - 1) <= SUM_TO_ONE, `${where} z=${z}: cdf + sf = ${c + s}`);
      [previousCdf, previousSf] = [c, s];
    }

    // The quantile inverts the tail it is the quantile of, within the conditioning of the tail.
    for (const p of probabilities) {
      const y = law.quantile(p);
      const target = p <= 0.5 ? p : 1 - p;
      const tail = p <= 0.5 ? law.cdf(y) : law.sf(y);
      const z = (3 * (Math.cbrt(y / mean) - 1 + (cv * cv) / 9)) / cv;
      const kappa = (Math.exp(-0.5 * z * z) / SQRT_2PI) * (Math.cbrt(y / mean) / cv) / target;
      const bound = ULPS * EPS * Math.max(1, kappa) * target;
      assert.ok(Math.abs(tail - target) <= bound, `${where} p=${p}: tail ${tail}`);
      assert.ok(Number.isFinite(y) && y > 0, `${where} p=${p}`);
    }

    // And the other way, on the half where the distribution function is accurate in relative terms.
    for (const p of [1e-6, 0.01, 0.1, 0.3]) {
      const y = law.quantile(p);
      const back = law.quantile(law.cdf(y));
      const z = (3 * (Math.cbrt(y / mean) - 1 + (cv * cv) / 9)) / cv;
      const kappaQ = (p * SQRT_2PI * cv) / (Math.exp(-0.5 * z * z) * Math.cbrt(y / mean));
      const bound = ULPS * EPS * Math.max(1, kappaQ) * y;
      assert.ok(Math.abs(back - y) <= bound, `${where} p=${p}: ${back} vs ${y}`);
    }
    assert.equal(law.median, law.quantile(0.5), where);
    assert.equal(law.cdf(0), 0);
    assert.equal(law.sf(0), 1);
    assert.equal(law.cdf(Infinity), 1);
    assert.equal(law.sf(Infinity), 0);
  }
});

test("a floored log link adds its floor to exp(eta)", () => {
  const law = lawOf("lognormal_mean", [0, -50], [LOG, { kind: "floored_log", floor: 0.5 }]);
  assert.equal(law.median, Math.exp(-0.125));
  assert.equal(law.mode, Math.exp(-0.125 - 0.25));
  const plain = lawOf("lognormal_mean", [0, Math.log(0.5)], [LOG, LOG]);
  assert.equal(plain.median, Math.exp(-0.125));
});

test("no point inside the documented domain makes a law throw or return NaN", () => {
  const probabilities = [PROBABILITY_MIN, 1e-6, 0.01, 0.5, 0.99, 1 - 1e-6, 1 - PROBABILITY_MIN];
  // The gamma from CV = 1.4e-11 (shape 5e21) up to GAMMA_CV_MAX; the log-normal from the scale
  // floor up to the scale 314 and the scale whose CV is GAMMA_CV_MAX.
  const spreads = (from, to, edge) => {
    const grid = [];
    for (let eta = from; eta < to; eta += 0.25) grid.push(eta);
    return [...grid, edge];
  };
  const etaMax = Math.log(GAMMA_CV_MAX);
  const scaleAtCvMax = Math.sqrt(Math.log1p(GAMMA_CV_MAX ** 2));
  const cases = [
    ["gamma", [LOG, LOG], spreads(-25, etaMax, etaMax)],
    ["lognormal_mean", [LOG, FLOORED], spreads(-30, 6, Math.log(scaleAtCvMax - FLOORED.floor))],
  ];
  let checked = 0;
  let underflowed = 0;
  for (const [family, links, grid] of cases) {
    for (const eta0 of [-3, 0, 4, 9, 14]) {
      for (const eta1 of grid) {
        const law = lawOf(family, [eta0, eta1], links);
        const where = `${family} eta=[${eta0}, ${eta1}]`;
        for (const value of [law.mean, law.median, law.mode]) {
          assert.ok(value >= 0 && value < Infinity, `${where}: location ${value}`);
        }
        for (const p of probabilities) {
          const y = law.quantile(p);
          assert.ok(y >= 0 && y < Infinity, `${where} p=${p}: quantile ${y}`);
          const [c, s] = [law.cdf(y), law.sf(y)];
          assert.ok(c >= 0 && c <= 1 && s >= 0 && s <= 1, `${where} p=${p}: cdf ${c} sf ${s}`);
          assert.ok(Math.abs(c + s - 1) <= SUM_TO_ONE, `${where} p=${p}: cdf + sf = ${c + s}`);

          // The quantile inverts the tail it is the quantile of, to 1e4 eps max(1, x f / tail).
          // Where the true quantile is below the smallest float64 it is returned as 0 or a
          // subnormal and cannot invert anything: that excuses the round trip only when the law
          // itself puts at least p below the smallest normal float64.
          if (y < SMALLEST_NORMAL) {
            assert.ok(law.cdf(SMALLEST_NORMAL) >= p, `${where} p=${p}: quantile ${y} too small`);
            underflowed++;
            continue;
          }
          const target = p <= 0.5 ? p : 1 - p;
          const tail = p <= 0.5 ? c : s;
          const xDensity = 10 ** law.log10Density(Math.log10(y)) / Math.LN10;
          const bound = ULPS * EPS * Math.max(1, xDensity / target) * target;
          assert.ok(Math.abs(tail - target) <= bound, `${where} p=${p}: tail ${tail} for ${y}`);
          checked++;
        }
        for (const l of [-8, -1, 0, 2, 5, 9]) {
          assert.ok(!Number.isNaN(law.log10Density(l)), `${where} l=${l}`);
        }
        assert.equal(law.log10Density(Infinity), -Infinity);
        assert.equal(law.log10Density(-Infinity), -Infinity);
      }
    }
  }
  assert.ok(checked > 0 && underflowed > 0, `round trips ${checked}, underflowed ${underflowed}`);
});

test("outside the documented domain lawOf and quantile throw a RangeError that names the limit", () => {
  const etaMax = Math.log(GAMMA_CV_MAX);
  const cvLimit = { name: "RangeError", message: new RegExp(`at most ${GAMMA_CV_MAX} `) };
  const range = `\\[${PROBABILITY_MIN}, 1 - ${PROBABILITY_MIN}\\]`;
  const probabilityLimit = { name: "RangeError", message: new RegExp(range) };

  // At the CV limit a law is built; just above it, and at a CV where the gamma quantile of
  // math.js stops converging (shape 1.7e-9), it is refused.
  assert.ok(lawOf("gamma", [5, etaMax], [LOG, LOG]).quantile(1 - PROBABILITY_MIN) > 0);
  assert.throws(() => lawOf("gamma", [5, etaMax + 1e-9], [LOG, LOG]), cvLimit);
  assert.throws(() => lawOf("gamma", [5, 10.1], [LOG, LOG]), cvLimit);

  // The quantile takes both ends of the probability range and nothing past them.
  const laws = {
    "gamma, exact": lawOf("gamma", [5, -3.5], [LOG, LOG]),
    "gamma, approximate": lawOf("gamma", [5, -5.5], [LOG, LOG]),
    "log-normal": lawOf("lognormal_mean", [5, -1], [LOG, FLOORED]),
  };
  for (const [name, law] of Object.entries(laws)) {
    for (const p of [PROBABILITY_MIN, 1 - PROBABILITY_MIN]) {
      assert.ok(law.quantile(p) > 0, `${name} p=${p}`);
    }
    for (const p of [PROBABILITY_MIN / 2, 1e-300, 1 - PROBABILITY_MIN / 2]) {
      assert.throws(() => law.quantile(p), probabilityLimit, `${name} p=${p}`);
    }
  }
});

test("invalid arguments throw RangeError and never return NaN", () => {
  const good = lawOf("gamma", [5, -1], [LOG, LOG]);
  const lognormal = lawOf("lognormal_mean", [5, -1], [LOG, FLOORED]);
  const bad = [
    () => lawOf("poisson", [0, 0], [LOG, LOG]),
    () => lawOf("gamma", [0], [LOG, LOG]),
    () => lawOf("gamma", [0, 0, 0], [LOG, LOG]),
    () => lawOf("gamma", [NaN, 0], [LOG, LOG]),
    () => lawOf("gamma", [0, Infinity], [LOG, LOG]),
    () => lawOf("gamma", [0, 0], [LOG]),
    () => lawOf("gamma", [0, 0], [FLOORED, LOG]),
    () => lawOf("gamma", [0, 0], [LOG, FLOORED]),
    () => lawOf("gamma", [800, 0], [LOG, LOG]),
    () => lawOf("gamma", [0, -400], [LOG, LOG]),
    () => lawOf("gamma", [0, 400], [LOG, LOG]),
    () => lawOf("gamma", [705, Math.log(GAMMA_CV_MAX)], [LOG, LOG]),
    () => lawOf("lognormal_mean", [0, 0], [LOG, { kind: "floored_log", floor: -1 }]),
    () => lawOf("lognormal_mean", [0, 0], [LOG, { kind: "floored_log", floor: NaN }]),
    () => lawOf("lognormal_mean", [0, 0], [LOG, { kind: "identity" }]),
    () => lawOf("lognormal_mean", [0, 400], [LOG, FLOORED]),
    () => lawOf("lognormal_mean", [800, 0], [LOG, FLOORED]),
    ...[good, lognormal].flatMap((law) => [
      () => law.cdf(NaN),
      () => law.sf(NaN),
      () => law.quantile(0),
      () => law.quantile(1),
      () => law.quantile(NaN),
      () => law.log10Density(NaN),
    ]),
  ];
  for (const f of bad) assert.throws(f, RangeError, String(f));
});
