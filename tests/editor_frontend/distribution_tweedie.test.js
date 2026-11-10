// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { PROBABILITY_MIN, lawOf } from "../../src/superglm/editor/app/distribution/families.js";
import { gammaP, gammaQ, lnGamma } from "../../src/superglm/editor/app/distribution/math.js";
import {
  TWEEDIE_RATE_MAX,
  TWEEDIE_SHAPE_MAX,
} from "../../src/superglm/editor/app/distribution/tweedie.js";

// Reference values for the compound Poisson-gamma law of a Tweedie power between one and two: the
// Python family's cdf and quantile (Poisson tail tolerance 1e-12), and the same sums in 40-digit
// arithmetic. The fixture's `produced_by` says how they were made.
const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/tweedie_reference.json", import.meta.url), "utf8")
);
const U = 2 ** -53;
const LOG = { kind: "log" };
const WALLS = { kind: "bounded_logit", lower: 1.05, upper: 1.95 };
const LINKS = [LOG, LOG, WALLS];
// The Python family leaves out at most this much Poisson mass at each end of its window, so its
// distribution function is at most twice that far from the truth.
const PYTHON_TAIL = 2e-12;
const SMALLEST_SURVIVAL = 1e-30;

const etaOf = (c) => [Math.log(c.mean), Math.log(c.dispersion), Math.log((c.power - 1.05) / (1.95 - c.power))];
const lawOfCase = (c) => lawOf("tweedie", etaOf(c), LINKS);

// A term of the sums is exp(-lambda + n ln lambda - ln n!) times a gamma factor, so its exponent
// carries a rounding error of about u (lambda + n |ln lambda| + ln n!) = 2 u lambda ln lambda at
// n = lambda; the sums add positive terms, so their relative error is that. 40 u covers the
// gamma factors, whose own relative error is a few units of roundoff times their conditioning.
const sumTolerance = (rate) => U * (40 + 2 * rate * Math.log(rate + Math.E));

/**
 * The size of the exponent a gamma tail Q(n a, x) forms, a ln x - x - ln Gamma(a), at the order n
 * that dominates the sum at y (the Poisson mode, or j_y = y^(2-p) / ((2-p) phi) where larger):
 * its rounding is u times this, times a small constant.
 */
function gammaExponentSize(c, y) {
  const alpha = (2 - c.power) / (c.power - 1);
  const scale = c.dispersion * (c.power - 1) * c.mean ** (c.power - 1);
  const jY = y ** (2 - c.power) / ((2 - c.power) * c.dispersion);
  const a = Math.max(1, Math.round(Math.max(c.rate, jY))) * alpha;
  const x = y / scale;
  return x + a * (Math.abs(Math.log(x)) + Math.log(a + 2));
}

/**
 * The law is built from ln mean, ln dispersion and the power's logit, and forms the rate and the
 * claim scale from them in float64; each is rounded to u times the sum of the magnitudes that went
 * into it, and the power to u absolute, which is u / (p - 1) or u / (2 - p) relative in the claim
 * shape. A probability moves by its conditioning kappa = y f(y) / S times that.
 */
function parameterSize(c) {
  const logScale = Math.log(c.dispersion) + Math.log(c.power - 1) + (c.power - 1) * Math.log(c.mean);
  return (
    Math.abs(Math.log(c.mean)) + Math.abs(Math.log(c.dispersion)) + Math.abs(Math.log(c.rate)) +
    Math.abs(logScale) + 1 / Math.min(c.power - 1, 2 - c.power)
  );
}

/** Composite Simpson's rule on [a, b] with n (even) panels. */
function simpson(f, a, b, n) {
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i += 1) sum += f(a + i * h) * (i % 2 ? 4 : 2);
  return (sum * h) / 3;
}

/** The mass above zero and the mean of the part above zero, from the density of log10 Y. */
function integrals(law, lo, hi, n) {
  const density = (l) => 10 ** law.log10Density(l);
  return {
    mass: simpson(density, lo, hi, n),
    meanAbove: simpson((l) => density(l) * 10 ** l, lo, hi, n),
  };
}

const closeRelative = (actual, expected, tolerance, label) =>
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.abs(expected), `${label}: ${actual} vs ${expected}`);

test("the chance of exactly zero and the mean agree with the references", () => {
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    const tolerance = sumTolerance(c.rate);
    closeRelative(law.zero, c.zero_mp, tolerance, `${c.name} zero`);
    // The Python family's cdf at zero is its chance of zero.
    closeRelative(law.zero, c.zero_py, tolerance, `${c.name} zero (python)`);
    assert.equal(law.cdf(0), law.zero, c.name);
    assert.equal(law.cdf(-1), 0, c.name);
    assert.equal(law.sf(-1), 1, c.name);
    closeRelative(law.sf(0), -Math.expm1(-c.rate), tolerance, `${c.name} sf(0)`);
    closeRelative(law.mean, c.mean, 4 * U, `${c.name} mean`);
  }
});

test("the distribution function agrees with the Python family and the 40-digit sum", () => {
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    for (const row of c.amounts) {
      const f = law.cdf(row.y);
      const label = `${c.name} y=${row.y}`;
      assert.ok(Math.abs(f - row.cdf_mp) <= sumTolerance(c.rate), `${label}: ${f} vs ${row.cdf_mp}`);
      assert.ok(Math.abs(f - row.cdf_py) <= PYTHON_TAIL + sumTolerance(c.rate), `${label}: ${f} vs ${row.cdf_py}`);
    }
  }
});

test("the survival function keeps its relative accuracy down to 1e-30", () => {
  let smallest = 1;
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    for (const row of c.amounts) {
      const s = law.sf(row.y);
      const label = `${c.name} y=${row.y}`;
      if (row.sf_mp >= SMALLEST_SURVIVAL) {
        // The gamma tails Q are evaluated to relative accuracy up to the rounding of their exponent,
        // and the law's parameters to that of their formation, moved by the survival's conditioning.
        const kappa = 10 ** law.log10Density(Math.log10(row.y)) / Math.LN10 / row.sf_mp;
        const exponent = gammaExponentSize(c, row.y) + parameterSize(c) * (1 + kappa);
        closeRelative(s, row.sf_mp, sumTolerance(c.rate) + 16 * U * exponent, label);
        smallest = Math.min(smallest, row.sf_mp);
      } else {
        // Below it the sum leaves out Poisson mass of 2^-53 * 1e-30 at most, in absolute terms.
        assert.ok(Math.abs(s - row.sf_mp) <= 2 * U * SMALLEST_SURVIVAL, `${label}: ${s} vs ${row.sf_mp}`);
      }
    }
  }
  // The references reach well into the upper tail, below the 1 in 10,000 point.
  assert.ok(smallest < 1e-25, `smallest survival tested ${smallest}`);
});

/**
 * The rate, claim shape and claim scale of the compound Poisson sum, from the link-scale
 * predictors with the same operations the law uses, so that two sums over the same parameters
 * differ only in how many terms they keep.
 */
function claimParameters([eta0, eta1, eta2], lower = 1.05, upper = 1.95) {
  const e = Math.exp(-Math.abs(eta2));
  const small = e / (1 + e);
  const sigma = eta2 >= 0 ? 1 - small : small;
  const complement = eta2 >= 0 ? small : 1 - small;
  const jump = lower - 1 + (upper - lower) * sigma;
  const tail = 2 - upper + (upper - lower) * complement;
  return {
    rate: Math.exp(tail * eta0 - eta1 - Math.log(tail)),
    shape: tail / jump,
    scale: Math.exp(eta1 + Math.log(jump) + jump * eta0),
  };
}

/** Neumaier's compensated sum: its error is a unit roundoff, not one per term. */
function compensatedSum(values) {
  let sum = 0;
  let carry = 0;
  for (const x of values) {
    const next = sum + x;
    carry += Math.abs(sum) >= Math.abs(x) ? sum - next + x : x - next + sum;
    sum = next;
  }
  return sum + carry;
}

test("with few claims the window leaves out no more than a unit roundoff of the sum", () => {
  // Every order from 1 to well past the window, with the law's own terms and a compensated sum.
  // Below a rate of 4 the law adds at most a couple of dozen terms, so adding them one after the
  // other costs at most 24 u; what it leaves out is 2^-53 of the sum. A dropped term is worth far
  // more: the first order carries 15% of the mass at a rate of 3.
  for (const c of FIXTURE.cases.filter((row) => row.rate < 4)) {
    const { rate, shape, scale } = claimParameters(etaOf(c));
    const law = lawOfCase(c);
    const weight = (n) => Math.exp(-rate + n * Math.log(rate) - lnGamma(n + 1));
    const orders = Array.from({ length: Math.ceil(rate + 20 * Math.sqrt(rate) + 50) }, (_, k) => k + 1);
    for (const row of c.amounts.filter((r) => r.sf_mp >= SMALLEST_SURVIVAL)) {
      const below = compensatedSum(orders.map((n) => weight(n) * gammaP(n * shape, row.y / scale)));
      const above = compensatedSum(orders.map((n) => weight(n) * gammaQ(n * shape, row.y / scale)));
      const label = `${c.name} y=${row.y}`;
      closeRelative(law.cdf(row.y), law.zero + below, 24 * U, `${label} cdf`);
      closeRelative(law.sf(row.y), above, 24 * U, `${label} sf`);
    }
  }
});

test("the quantile inverts the distribution function and agrees with the Python family", () => {
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    const tolerance = sumTolerance(c.rate);
    for (const [p, expected] of c.quantiles) {
      const y = law.quantile(p);
      const label = `${c.name} p=${p}`;
      // dF / d ln y at the answer is y f(y) = f_L / ln 10, with f_L the density of log10 Y.
      const slope = 10 ** law.log10Density(Math.log10(expected)) / Math.LN10;
      // The search stops when its step in ln y is below 1e-12, which moves F by slope * 1e-12.
      assert.ok(Math.abs(law.cdf(y) - p) <= 2 * tolerance + 2e-12 * slope, `${label}: cdf(quantile) = ${law.cdf(y)}`);
      // Python's cdf is off by its tail tolerance, which moves its quantile by that over the slope.
      const allowed = (PYTHON_TAIL + 2 * tolerance) / slope + 1e-11;
      assert.ok(Math.abs(y - expected) <= allowed * expected, `${label}: ${y} vs ${expected}`);
    }
  }
});

test("the density of log10 Y agrees with the 40-digit sum", () => {
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    const alpha = (2 - c.power) / (c.power - 1);
    const scale = c.dispersion * (c.power - 1) * c.mean ** (c.power - 1);
    for (const row of c.amounts) {
      const l = Math.log10(row.y);
      const h = 1e-5 * Math.max(1, Math.abs(l));
      // The conditioning of the density in l: it is read at l = log10(y), rounded to a double.
      const slope = Math.abs(law.log10Density(l + h) - law.log10Density(l - h)) / (2 * h);
      // The exponent of the dominant term n, where j_y is within one of the peak.
      const n = Math.max(1, Math.round(row.y ** (2 - c.power) / ((2 - c.power) * c.dispersion)));
      const s = Math.log(row.y / scale);
      const size = c.rate * (1 + Math.abs(Math.log(c.rate))) + n * (Math.abs(Math.log(c.rate)) + Math.log(n + 1))
        + n * alpha * (Math.abs(s) + Math.log(n * alpha + 2) + Math.abs(Math.log(row.y))) + row.y / scale;
      const tolerance = (32 * U * size) / Math.LN10 + 8 * U + 8 * U * slope * Math.abs(l);
      const got = law.log10Density(l);
      assert.ok(Math.abs(got - row.log10_density_mp) <= tolerance, `${c.name} l=${l}: ${got} vs ${row.log10_density_mp}`);
    }
  }
});

test("the distribution function is nondecreasing, reaches one, and agrees with the survival function", () => {
  for (const c of FIXTURE.cases) {
    const law = lawOfCase(c);
    const tolerance = 2 * sumTolerance(c.rate);
    let previous = law.zero;
    for (let k = 0; k <= 400; k += 1) {
      const y = c.mean * 10 ** (-8 + (11 * k) / 400);
      const f = law.cdf(y);
      assert.ok(f >= previous - tolerance && f <= 1, `${c.name} y=${y}: ${f} after ${previous}`);
      // Both are accurate where neither is a difference of nearly equal numbers.
      assert.ok(Math.abs(f + law.sf(y) - 1) <= 2 * tolerance, `${c.name} y=${y}: cdf + sf = ${f + law.sf(y)}`);
      previous = Math.max(previous, f);
    }
    assert.ok(law.cdf(1e6 * c.mean * 100) >= 1 - tolerance, c.name);
    assert.equal(law.cdf(Infinity), 1);
    assert.equal(law.sf(Infinity), 0);
  }
});

test("quantiles at or below the atom are zero, and the median and mode say so", () => {
  const law = lawOf("tweedie", [Math.log(50), Math.log(275.6), 0], LINKS);
  assert.ok(Math.abs(law.zero - 0.95) < 1e-3);
  for (const p of [PROBABILITY_MIN, 0.5, 0.9, law.zero * (1 - 1e-9), law.zero]) assert.equal(law.quantile(p), 0, `p=${p}`);
  assert.ok(law.quantile(law.zero + 1e-6) > 0);
  assert.equal(law.median, 0);
  assert.equal(law.mode, 0);
  // With the atom below a half the median is positive and the mode is the density's peak.
  const mostly = lawOf("tweedie", [Math.log(1000), Math.log(1000 ** 0.5 / (-Math.log(0.05) * 0.5)), 0], LINKS);
  assert.ok(mostly.zero < 0.5 && mostly.median > 0);
  assert.equal(mostly.median, mostly.quantile(0.5));
});

test("the mode is the highest peak of the density of the part above zero, or zero when it only falls", () => {
  // [mean, chance of zero, power]: a unimodal law, a lattice-like one near the lower wall, a power
  // above 1.5 whose claim shape is below 1 but whose rate is large, and one that only falls from zero.
  const peaks = [[1000, 0.05, 1.5], [400, 0.02, 1.3], [300, 0.05, 1.06], [100, Math.exp(-69), 1.9]];
  for (const [mean, zero, power] of peaks) {
    const rate = -Math.log(zero);
    const eta = [Math.log(mean), Math.log(mean ** (2 - power) / (rate * (2 - power))), Math.log((power - 1.05) / (1.95 - power))];
    const law = lawOf("tweedie", eta, LINKS);
    // ln f(y) = ln f_L - ln y - ln ln 10; scan a fine grid of log10 amounts for its highest point.
    const lnDensity = (l) => law.log10Density(l) * Math.LN10 - l * Math.LN10;
    let best = -Infinity;
    let at = 0;
    for (let k = 0; k <= 20000; k += 1) {
      const l = Math.log10(mean) - 3 + (4.5 * k) / 20000;
      if (lnDensity(l) > best) [best, at] = [lnDensity(l), l];
    }
    assert.ok(Math.abs(Math.log10(law.mode) - at) <= 2.5e-4, `${mean} ${power}: ${law.mode} vs ${10 ** at}`);
    const m = Math.log10(law.mode);
    assert.ok(lnDensity(m) >= lnDensity(m - 1e-3) && lnDensity(m) >= lnDensity(m + 1e-3));
  }
  const eta = [Math.log(400), Math.log(400 ** 0.3 / (-Math.log(0.2) * 0.3)), Math.log(0.65 / 0.25)];
  assert.equal(lawOf("tweedie", eta, LINKS).mode, 0);
});

test("the density of log10 Y integrates to the chance above zero and the mean to the mean", () => {
  // [mean, chance of zero, power]; the range runs from the 1e-11 quantile of the part above zero
  // to its 1 - 1e-11 quantile, so at most 2e-11 of its mass lies outside it.
  const laws = [[50, 0.95, 1.5], [300, 0.5, 1.06], [300, 0.5, 1.94], [1000, 0.05, 1.5], [100, Math.exp(-40), 1.5]];
  for (const [mean, zero, power] of laws) {
    const rate = -Math.log(zero);
    const eta = [Math.log(mean), Math.log(mean ** (2 - power) / (rate * (2 - power))), Math.log((power - 1.05) / (1.95 - power))];
    const law = lawOf("tweedie", eta, LINKS);
    const lo = Math.log10(law.quantile(law.zero + 1e-11 * (1 - law.zero)));
    const hi = Math.log10(law.quantile(1 - 1e-11));
    const { mass, meanAbove } = integrals(law, lo, hi, 200000);
    const label = `${mean} ${zero} ${power}`;
    assert.ok(Math.abs(mass - (1 - law.zero)) <= 1e-9, `${label}: mass ${mass} vs ${1 - law.zero}`);
    // The atom adds nothing to the mean, so the part above zero carries all of it.
    assert.ok(Math.abs(meanAbove - mean) <= 1e-8 * mean, `${label}: mean ${meanAbove}`);
  }
});

test("the power is read between its walls: eta 0 is the middle, large eta the walls", () => {
  const mean = 4;
  const dispersion = 0.5;
  /** The chance of zero from the model's own formula, exp(-mean^(2-p) / (dispersion (2-p))). */
  const zeroAt = (p) => Math.exp(-(mean ** (2 - p)) / (dispersion * (2 - p)));
  const at = (eta2, links = LINKS) => lawOf("tweedie", [Math.log(mean), Math.log(dispersion), eta2], links);
  closeRelative(at(0).zero, zeroAt(1.5), 1e-14, "middle");
  closeRelative(at(40).zero, zeroAt(1.95), 1e-14, "upper wall");
  closeRelative(at(-40).zero, zeroAt(1.05), 1e-14, "lower wall");
  const narrow = [LOG, LOG, { kind: "bounded_logit", lower: 1.2, upper: 1.8 }];
  closeRelative(at(0, narrow).zero, zeroAt(1.5), 1e-14, "middle of other walls");
  closeRelative(at(40, narrow).zero, zeroAt(1.8), 1e-14, "upper of other walls");
  closeRelative(at(-40, narrow).zero, zeroAt(1.2), 1e-14, "lower of other walls");
  // A quarter of the way up the logit: p = lower + (upper - lower) / (1 + exp(-eta)).
  closeRelative(at(Math.log(3)).zero, zeroAt(1.05 + 0.9 * 0.75), 1e-14, "three to one");
});

test("the other families carry no atom", () => {
  assert.equal(lawOf("gamma", [Math.log(100), Math.log(0.5)], [LOG, LOG]).zero, 0);
  assert.equal(lawOf("lognormal_mean", [Math.log(100), -0.5], [LOG, LOG]).zero, 0);
});

test("inputs the law cannot compute with throw a RangeError", () => {
  const ok = [Math.log(100), Math.log(5), 0];
  assert.doesNotThrow(() => lawOf("tweedie", ok, LINKS));
  const refusals = [
    ["two predictors", () => lawOf("tweedie", ok.slice(0, 2), LINKS)],
    ["four predictors", () => lawOf("tweedie", [...ok, 0], LINKS)],
    ["a NaN predictor", () => lawOf("tweedie", [ok[0], Number.NaN, 0], LINKS)],
    ["an infinite predictor", () => lawOf("tweedie", [ok[0], ok[1], Infinity], LINKS)],
    ["two links", () => lawOf("tweedie", ok, [LOG, LOG])],
    ["a log link on the power", () => lawOf("tweedie", ok, [LOG, LOG, LOG])],
    ["a floored link on the mean", () => lawOf("tweedie", ok, [{ kind: "floored_log", floor: 0.1 }, LOG, WALLS])],
    ["a lower wall of one", () => lawOf("tweedie", ok, [LOG, LOG, { ...WALLS, lower: 1 }])],
    ["an upper wall of two", () => lawOf("tweedie", ok, [LOG, LOG, { ...WALLS, upper: 2 }])],
    ["walls the wrong way round", () => lawOf("tweedie", ok, [LOG, LOG, { ...WALLS, lower: 1.9, upper: 1.2 }])],
    ["a rate above the cap", () => lawOf("tweedie", [Math.log(100), Math.log(5e-3), 0], LINKS)],
    ["a rate that underflows", () => lawOf("tweedie", [Math.log(1e-3), Math.log(1e300), 0], LINKS)],
    ["a mean beyond float64", () => lawOf("tweedie", [800, 0, 0], LINKS)],
    ["claims whose gamma shape is beyond math.js", () => lawOf("tweedie", [Math.log(100), Math.log(0.05), -40], LINKS)],
  ];
  for (const [what, call] of refusals) assert.throws(call, RangeError, what);
  const law = lawOf("tweedie", ok, LINKS);
  for (const call of [() => law.cdf(Number.NaN), () => law.sf(Number.NaN), () => law.log10Density(Number.NaN)]) {
    assert.throws(call, RangeError);
  }
  for (const p of [0, 1, PROBABILITY_MIN / 2, 1 - PROBABILITY_MIN / 2, Number.NaN]) {
    assert.throws(() => law.quantile(p), RangeError, `p=${p}`);
  }
  assert.throws(() => lawOf("poisson", ok, LINKS), RangeError);
});

test("laws up to the rate cap are computed", () => {
  // At the cap the law is nearly normal: a mean of 100 with rate 1900 at power 1.5 has CV 0.03.
  const rate = 0.95 * TWEEDIE_RATE_MAX;
  const eta = [Math.log(100), Math.log(100 ** 0.5 / (rate * 0.5)), 0];
  const law = lawOf("tweedie", eta, LINKS);
  assert.ok(Math.abs(law.cdf(100) - 0.5) < 0.02);
  assert.ok(Math.abs(law.cdf(law.quantile(0.9999)) - 0.9999) < 1e-9);
  assert.ok(TWEEDIE_SHAPE_MAX > 0);
});
