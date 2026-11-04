// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  erfc,
  gammaP,
  gammaQ,
  gammaQuantile,
  lnGamma,
  nbPmf,
  nbSf,
  normCdf,
  normQuantile,
  normSf,
} from "../../src/superglm/editor/app/distribution/math.js";

const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/distribution_math.json", import.meta.url), "utf8")
);
const EPS = Number.EPSILON;
const ULPS = 1e4;

/** Checks every fixture row against |js - scipy| <= 1e4 eps max(1, kappa) |scipy|. */
function checkRows(t, name, evaluate) {
  let worst = 0;
  for (const row of FIXTURE[name]) {
    const got = evaluate(row);
    const bound = ULPS * EPS * Math.max(1, row.kappa ?? 1) * Math.abs(row.value);
    const err = Math.abs(got - row.value);
    assert.ok(err <= bound, `${name} ${JSON.stringify(row)}: js ${got}, error ${err} > ${bound}`);
    if (bound > 0) worst = Math.max(worst, err / bound);
    else assert.equal(err, 0);
  }
  const rows = FIXTURE[name].length;
  t.diagnostic(`${name}: worst error / bound = ${worst.toExponential(2)} over ${rows} rows`);
}

test("erfc matches scipy over the three Cody intervals and both signs", (t) => {
  checkRows(t, "erfc", (r) => erfc(r.x));
});

test("normCdf and normSf keep each tail's relative accuracy", (t) => {
  checkRows(t, "norm_cdf", (r) => normCdf(r.z));
  checkRows(t, "norm_sf", (r) => normSf(r.z));
});

test("normQuantile matches scipy from 1e-12 to 1 - 1e-12", (t) => {
  checkRows(t, "norm_quantile", (r) => normQuantile(r.p));
});

test("lnGamma matches scipy across the upward recurrence and the Stirling series", (t) => {
  checkRows(t, "lngamma", (r) => lnGamma(r.a));
});

test("gammaP and gammaQ match scipy in both tails for every shape", (t) => {
  checkRows(t, "gamma_p", (r) => gammaP(r.a, r.x));
  checkRows(t, "gamma_q", (r) => gammaQ(r.a, r.x));
});

test("gammaQuantile matches scipy in both tails for every shape", (t) => {
  checkRows(t, "gamma_quantile", (r) => gammaQuantile(r.a, r.p));
});

test("negative binomial pmf and survival match scipy", (t) => {
  checkRows(t, "nb_pmf", (r) => nbPmf(r.k, r.mean, r.theta));
  checkRows(t, "nb_sf", (r) => nbSf(r.k, r.mean, r.theta));
});

test("the two tails of erfc are exact reflections", () => {
  for (const x of [0.05, 0.4999, 0.5, 1.3, 3.9, 4.0, 7.5]) {
    assert.equal(erfc(-x), 2 - erfc(x));
  }
  assert.equal(erfc(0), 1);
  assert.equal(erfc(40), 0);
  assert.equal(erfc(-40), 2);
  assert.equal(normQuantile(0.5), 0);
  assert.equal(normCdf(Infinity), 1);
  assert.equal(normSf(Infinity), 0);
});

test("the erfc tail keeps a few units of roundoff where exp(-x^2) alone loses x^2 of them", () => {
  // erfc(x) to 25 digits from mpmath. exp(-x^2) from the rounded square would be off by x^2 u.
  const exact = [
    [5.37, 3.093965988445734240195e-14],
    [12.71, 3.077757136066643313078e-72],
    [19.83, 4.746904376020047025281e-173],
    [26.37, 2.146113603942126487155e-304],
  ];
  for (const [x, value] of exact) {
    assert.ok(Math.abs(erfc(x) - value) <= 4 * EPS * value, `erfc(${x})`);
  }
});

test("normQuantile reaches the far tail, past AS 241's break point at r = 5", () => {
  // The quantiles of 1e-30, 1e-100, 1e-300 and 2^-1074 to 22 digits from mpmath.
  const exact = [
    [1e-30, -11.46402468844361572698],
    [1e-100, -21.27345356096532429512],
    [1e-300, -37.04709629936119923722],
    [2 ** -1074, -38.46740561714434625078],
  ];
  for (const [p, z] of exact) {
    assert.ok(Math.abs(normQuantile(p) - z) <= 4 * EPS * Math.abs(z), `p=${p}`);
  }
  assert.equal(normQuantile(0.625), -normQuantile(0.375));
});

test("quantiles invert the distribution functions they belong to", () => {
  for (const p of [1e-12, 1e-6, 0.3, 0.5, 0.9, 1 - 1e-9]) {
    const z = normQuantile(p);
    const target = p <= 0.5 ? p : 1 - p;
    const got = p <= 0.5 ? normCdf(z) : normSf(z);
    const kappa = (Math.abs(z) * Math.exp(-0.5 * z * z)) / Math.sqrt(2 * Math.PI) / target;
    assert.ok(Math.abs(got - target) <= ULPS * EPS * Math.max(1, kappa) * target, `p=${p}`);
  }
  for (const a of [0.04, 0.5, 1, 7, 250]) {
    for (const p of [1e-9, 1e-3, 0.5, 0.97, 1 - 1e-9]) {
      const x = gammaQuantile(a, p);
      const q = 1 - p;
      const tail = p <= 0.5 ? gammaP(a, x) : gammaQ(a, x);
      const target = p <= 0.5 ? p : q;
      const kappa = Math.exp(a * Math.log(x) - x - lnGamma(a)) / tail;
      const bound = ULPS * EPS * Math.max(1, kappa) * target;
      assert.ok(Math.abs(tail - target) <= bound, `a=${a} p=${p}`);
    }
  }
});

test("a negative binomial law is a probability distribution", () => {
  for (const [mean, theta] of [[0.3, 3], [2, 0.5], [0.05, 50]]) {
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += nbPmf(k, mean, theta);
    assert.ok(Math.abs(1 - sum - nbSf(4, mean, theta)) <= 8 * EPS);
    assert.equal(nbSf(0, mean, theta), 1);
  }
  assert.equal(nbPmf(0, 0, 3), 1);
  assert.equal(nbPmf(2, 0, 3), 0);
});

test("invalid arguments throw RangeError and never return NaN", () => {
  const bad = [
    () => normQuantile(0),
    () => normQuantile(1),
    () => normQuantile(-0.1),
    () => normQuantile(NaN),
    () => erfc(NaN),
    () => normCdf(NaN),
    () => lnGamma(0),
    () => lnGamma(-2),
    () => lnGamma(NaN),
    () => gammaP(0, 1),
    () => gammaP(-1, 1),
    () => gammaP(2, -0.5),
    () => gammaQ(2, NaN),
    () => gammaQuantile(0, 0.5),
    () => gammaQuantile(2, 0),
    () => gammaQuantile(2, 1),
    () => nbPmf(-1, 1, 1),
    () => nbPmf(1.5, 1, 1),
    () => nbPmf(1, -1, 1),
    () => nbPmf(1, 1, 0),
    () => nbSf(1, 1, -2),
  ];
  for (const f of bad) assert.throws(f, RangeError, String(f));
});

test("the series and continued fraction throw at the iteration cap instead of guessing", () => {
  // Near x = a the series needs about 8.5 sqrt(a) terms and the fraction about 0.6 sqrt(a).
  assert.throws(() => gammaP(1e6, 1e6), RangeError);
  assert.throws(() => gammaQ(1e8, 1e8 + 2), RangeError);
});

test("the gamma functions take their limits at x = 0 and x = Infinity", () => {
  assert.equal(gammaP(3, 0), 0);
  assert.equal(gammaQ(3, 0), 1);
  assert.equal(gammaP(3, Infinity), 1);
  assert.equal(gammaQ(3, Infinity), 0);
});
