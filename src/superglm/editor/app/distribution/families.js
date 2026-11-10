// @ts-check
/**
 * Predicted laws of the continuous severity families, rebuilt in the browser from link-scale
 * predictors so that the distribution panel can redraw while a curve is dragged.
 *
 *  - lognormal_mean: log Y ~ N(mu, s^2), the mean E[Y] under a log link and the scale s under
 *    the shifted log link s = floor + exp(eta). The location is mu = ln(mean) - s^2 / 2 and the
 *    mode exp(mu - s^2).
 *  - gamma: mean and coefficient of variation CV under log links. The shape is k = 1 / CV^2, the
 *    scale theta = mean CV^2, and the mode (k - 1) theta for k >= 1, else 0.
 *  - tweedie: a power between one and two, a compound Poisson sum of gamma claims with a chance of
 *    exactly zero; its method, sources and domain are in tweedie.js.
 *
 * Sources
 *  - Wilson, E. B. & Hilferty, M. M. (1931), "The distribution of chi-square", PNAS 17(12):
 *    684-688: (X / mean)^(1/3) is close to N(1 - 1 / (9k), 1 / (9k)) for X ~ Gamma(k). Above
 *    WILSON_HILFERTY_SHAPE the distribution function, survival function and quantile use it,
 *    because the incomplete gamma series and continued fraction of math.js stop converging in
 *    their iteration cap from a shape of about 5e4 (CV below 0.0045).
 *  - Abramowitz & Stegun, Handbook of Mathematical Functions (1964), 6.1.40: the Stirling series
 *    for ln Gamma, which keeps the density at a shape of 10 or more free of the cancellation
 *    between k ln k and ln Gamma(k).
 *
 * Accuracy. The log-normal and the gamma at shape up to WILSON_HILFERTY_SHAPE are as accurate as
 * math.js: a few units of roundoff times the condition number of the quantity. Above that shape
 * the gamma distribution function, survival function and quantile carry the Wilson-Hilferty
 * approximation error, which this module does not bound. Canal (2005), "A normal approximation
 * for the chi-square distribution", Comput. Statist. Data Anal. 48(4): 803-808, tabulates the
 * error of the cube-root approximation for 1 to 1000 degrees of freedom (shape up to 500), and
 * Brown (1974), "Error analysis of some normal approximations to the chi-square distribution",
 * J. Academy of Marketing Science 2(3): 447-454, tabulates it per degrees of freedom (only its
 * abstract was read, so its range is unverified); no explicit bound for shapes above 1e4 was
 * found. The mean, mode and log-density stay exact at every shape. The approximation keeps the
 * properties of a law: distribution function and survival function are monotone and sum to one,
 * and the quantile inverts them.
 *
 * Domain, and what throws. Inside it no method throws or returns NaN:
 *  - eta is two finite numbers and the links are those of the family;
 *  - the mean, shape, CV^2 and scale theta of the gamma, and the mean and s^2 of the log-normal,
 *    are positive and finite in float64;
 *  - the gamma CV is at most GAMMA_CV_MAX;
 *  - `quantile` takes p in [PROBABILITY_MIN, 1 - PROBABILITY_MIN].
 * Outside it, `lawOf` throws a RangeError for a gamma CV above GAMMA_CV_MAX or a parameter outside
 * float64, `quantile` throws a RangeError naming the probability range, and `cdf`, `sf` and
 * `log10Density` throw a RangeError for a NaN argument.
 *
 * The CV limit is where math.js stays accurate. At a small shape it forms the upper tail of the
 * incomplete gamma function as 1 - P, which carries an absolute error of a few eps; from a CV of
 * about 26 that error exceeds 1e4 eps times the condition number of a tail near 3e-4 (checked on
 * a dense grid of CV and p), and its quantile stops converging below a shape of about 1e-9.
 *
 * Float64 limit. At a small gamma shape the true quantile can be smaller than the smallest float64.
 * At CV = GAMMA_CV_MAX the 10% quantile is of order 1e-400 times the mean, so `quantile` returns 0
 * or a subnormal, `cdf` of it is 0 rather than p, and the round trip cdf(quantile(p)) = p holds
 * only where the quantile is a normal positive number.
 */

import { tweedieLaw } from "./tweedie.js";
import {
  LN_SQRT_2PI,
  STIRLING_MIN,
  gammaP,
  gammaQ,
  gammaQuantile,
  lnGamma,
  normCdf,
  normQuantile,
  normSf,
  stirlingSeries,
  wilsonHilfertyRoot,
} from "./math.js";

/**
 * A link's inverse: exp(eta) for "log", floor + exp(eta) for "floored_log", and for
 * "bounded_logit" lower + (upper - lower) / (1 + exp(-eta)).
 * @typedef {{kind: "log"} | {kind: "floored_log", floor: number} |
 *   {kind: "bounded_logit", lower: number, upper: number}} LinkSpec
 */

/**
 * @typedef {object} Law
 * @property {number} mean the mean of the whole law, the chance of exactly zero included
 * @property {number} zero the chance of exactly zero: 0 for the continuous families
 * @property {number} median the 0.5 quantile; 0 where `zero` is at least a half
 * @property {number} mode the most common amount: the peak of the density, or 0 where the density
 *   falls from zero or `zero` is at least a half
 * @property {(y: number) => number} cdf P(Y <= y), which is `zero` at y = 0
 * @property {(y: number) => number} sf P(Y > y), accurate in the upper tail
 * @property {(p: number) => number} quantile the y with P(Y <= y) = p, p in [PROBABILITY_MIN, 1 - PROBABILITY_MIN]
 * @property {(l: number) => number} log10Density the density of log10 Y at l, on a log10 scale
 */

/** Above this gamma shape (CV below 0.01) the cdf, sf and quantile use Wilson-Hilferty. */
export const WILSON_HILFERTY_SHAPE = 1e4;
/** The largest gamma CV a law is built for; the dragged CV is clamped to it. */
export const GAMMA_CV_MAX = 20;
/** `quantile` takes p from this to 1 minus this. */
export const PROBABILITY_MIN = 1e-12;

/**
 * The median and the mean of a law's amounts above zero: the part its density draws, and so the
 * marks a drawn curve carries. A law with no chance of exactly zero gives its own.
 * @param {Law} law
 */
export const medianAboveZero = (law) =>
  law.zero > 0 ? law.quantile(law.zero + 0.5 * (1 - law.zero)) : law.median;
/** @param {Law} law */
export const meanAboveZero = (law) => law.mean / (1 - law.zero);

const ETA_CV_MAX = Math.log(GAMMA_CV_MAX);
const LN_LN10 = Math.log(Math.LN10);
// Below this |w| the series for exp(w) - 1 - w converges to the roundoff in a dozen terms and
// the difference of the two terms would lose digits.
const SERIES_BELOW = 0.1;

/** @param {string} what @param {unknown} got @param {string} rule @returns {never} */
function refuse(what, got, rule) {
  const shown = JSON.stringify(got, (_, v) =>
    typeof v === "number" && !Number.isFinite(v) ? String(v) : v
  );
  throw new RangeError(`lawOf: ${what} must be ${rule}, got ${shown}`);
}

/** @param {number} value @param {string} what */
function requireNumber(value, what) {
  if (Number.isNaN(value)) refuse(what, value, "a number");
}

/**
 * @param {number} p
 * @returns {number} p, when it lies in [PROBABILITY_MIN, 1 - PROBABILITY_MIN]
 */
function requireQuantileLevel(p) {
  if (!(p >= PROBABILITY_MIN && p <= 1 - PROBABILITY_MIN)) {
    refuse("p", p, `in [${PROBABILITY_MIN}, 1 - ${PROBABILITY_MIN}]`);
  }
  return p;
}

/**
 * exp(w) - 1 - w without cancellation near 0.
 * @param {number} w
 */
function expm1MinusX(w) {
  if (Math.abs(w) >= SERIES_BELOW) return Math.expm1(w) - w;
  let term = (w * w) / 2;
  let sum = term;
  for (let n = 3; Math.abs(term) > Math.abs(sum) * Number.EPSILON; n++) {
    term *= w / n;
    sum += term;
  }
  return sum;
}

/**
 * @param {readonly LinkSpec[]} links @param {number} index @param {readonly string[]} allowed
 * @param {string} what
 * @returns {number} the floor of the link at `index`: 0 for a plain log link
 */
function linkFloor(links, index, allowed, what) {
  const link = links[index];
  if (!link || !allowed.includes(link.kind)) {
    refuse(what, link, `a link of kind ${allowed.map((a) => `"${a}"`).join(" or ")}`);
  }
  if (link.kind === "log") return 0;
  // The allowed kinds above leave only floored_log, the one link here that has a floor.
  const { floor } = /** @type {{kind: "floored_log", floor: number}} */ (link);
  if (!(Number.isFinite(floor) && floor >= 0)) {
    refuse(`${what} floor`, floor, "a finite number of zero or more");
  }
  return floor;
}

/**
 * @param {readonly number[]} eta @param {readonly LinkSpec[]} links
 * @param {readonly string[]} spreadKinds link kinds the spread parameter may use
 * @returns {{eta0: number, eta1: number, floor: number}}
 */
function readInputs(eta, links, spreadKinds) {
  if (!Array.isArray(eta) || eta.length !== 2 || !eta.every(Number.isFinite)) {
    refuse("eta", eta, "two finite link-scale numbers [mean, CV]");
  }
  if (!Array.isArray(links) || links.length !== 2) {
    refuse("links", links, "two link specs [mean, CV]");
  }
  linkFloor(links, 0, ["log"], "the mean link");
  const floor = linkFloor(links, 1, spreadKinds, "the CV link");
  return { eta0: eta[0], eta1: eta[1], floor };
}

/**
 * @param {number} eta0 @param {number} eta1 @param {number} floor
 * @returns {Law}
 */
function logNormalLaw(eta0, eta1, floor) {
  const scale = floor + Math.exp(eta1);
  const mean = Math.exp(eta0);
  // mu = ln(mean) - s^2 / 2, with ln(mean) = eta0 from the log link.
  const mu = eta0 - (scale * scale) / 2;
  if (!(scale > 0 && Number.isFinite(scale * scale) && Number.isFinite(mean) && mean > 0)) {
    refuse("the log-normal mean and scale", [mean, scale], "positive and within float64");
  }
  /** @param {number} y */
  const standardised = (y) => (Math.log(y) - mu) / scale;
  /** @param {number} p */
  const quantile = (p) => Math.exp(mu + scale * normQuantile(requireQuantileLevel(p)));
  return {
    mean,
    zero: 0,
    median: quantile(0.5),
    mode: Math.exp(mu - scale * scale),
    cdf(y) {
      requireNumber(y, "y");
      return y > 0 ? normCdf(standardised(y)) : 0;
    },
    sf(y) {
      requireNumber(y, "y");
      return y > 0 ? normSf(standardised(y)) : 1;
    },
    quantile,
    log10Density(l) {
      requireNumber(l, "l");
      if (!Number.isFinite(l)) return -Infinity;
      // f_L(l) = ln 10 * phi(z) / s with z = (l ln 10 - mu) / s.
      const z = (l * Math.LN10 - mu) / scale;
      return (LN_LN10 - Math.log(scale) - LN_SQRT_2PI - (z * z) / 2) / Math.LN10;
    },
  };
}

/**
 * @param {number} eta0 @param {number} eta1
 * @returns {Law}
 */
function gammaLaw(eta0, eta1) {
  if (eta1 > ETA_CV_MAX) {
    refuse("the gamma CV", Math.exp(eta1), `at most ${GAMMA_CV_MAX} (GAMMA_CV_MAX)`);
  }
  const mean = Math.exp(eta0);
  const shape = Math.exp(-2 * eta1);
  const cv2 = Math.exp(2 * eta1);
  const theta = mean / shape;
  if (![mean, shape, cv2, theta].every((v) => v > 0 && Number.isFinite(v))) {
    const got = [mean, Math.exp(eta1), theta];
    refuse("the gamma mean, CV and scale", got, "positive and within float64");
  }
  const approximate = shape > WILSON_HILFERTY_SHAPE;
  const cv = Math.exp(eta1);
  /** ln(y / theta) for theta = mean CV^2. */
  const logScaled = (/** @type {number} */ l) => l * Math.LN10 - (eta0 + 2 * eta1);

  // Wilson-Hilferty: ((y / mean)^(1/3) - (1 - CV^2 / 9)) / (CV / 3) is standard normal.
  /** @param {number} y */
  const standardised = (y) =>
    (3 * (Math.expm1(Math.log1p((y - mean) / mean) / 3) + cv2 / 9)) / cv;
  /** @param {number} y */
  const scaled = (y) => (y / mean) * shape;
  /** @param {number} p */
  const quantile = (p) => {
    requireQuantileLevel(p);
    if (!approximate) return theta * gammaQuantile(shape, p);
    const base = wilsonHilfertyRoot(shape, normQuantile(p));
    return mean * base * base * base;
  };

  // Constant for the law, and the fan asks for its density once per pixel row.
  const belowStirling = shape < STIRLING_MIN;
  const shapeTerm = belowStirling ? lnGamma(shape) : stirlingSeries(shape);

  /** ln(y f_Y(y)) at log10 y = l; from STIRLING_MIN on it avoids the k ln k cancellation. */
  const logMass = (/** @type {number} */ l) => {
    if (belowStirling) {
      const scaledLog = logScaled(l);
      return shape * scaledLog - Math.exp(scaledLog) - shapeTerm;
    }
    // k (ln t - t + 1) + (ln k) / 2 - ln sqrt(2 pi) - tail, with t = y / mean, w = ln t,
    // (ln k) / 2 = -eta1, and tail the Stirling series of ln Gamma(k) past (k - 1/2) ln k - k.
    const w = l * Math.LN10 - eta0;
    return -shape * expm1MinusX(w) - eta1 - LN_SQRT_2PI - shapeTerm;
  };

  return {
    mean,
    zero: 0,
    median: quantile(0.5),
    mode: shape >= 1 ? (mean * (shape - 1)) / shape : 0,
    cdf(y) {
      requireNumber(y, "y");
      if (!(y > 0)) return 0;
      return approximate ? normCdf(standardised(y)) : gammaP(shape, scaled(y));
    },
    sf(y) {
      requireNumber(y, "y");
      if (!(y > 0)) return 1;
      return approximate ? normSf(standardised(y)) : gammaQ(shape, scaled(y));
    },
    quantile,
    log10Density(l) {
      requireNumber(l, "l");
      if (!Number.isFinite(l)) return -Infinity;
      return (LN_LN10 + logMass(l)) / Math.LN10;
    },
  };
}

/**
 * The predicted distribution of one row.
 * @param {"lognormal_mean" | "gamma" | "tweedie"} family
 * @param {readonly number[]} eta link-scale predictors [mean, CV]; the CV is the
 *   log-normal scale s or the gamma coefficient of variation; for "tweedie" three, [mean,
 *   dispersion, power]
 * @param {readonly LinkSpec[]} links the families' links for the predictors
 * @returns {Law}
 */
export function lawOf(family, eta, links) {
  if (family === "lognormal_mean") {
    const { eta0, eta1, floor } = readInputs(eta, links, ["log", "floored_log"]);
    return logNormalLaw(eta0, eta1, floor);
  }
  if (family === "gamma") {
    const { eta0, eta1 } = readInputs(eta, links, ["log"]);
    return gammaLaw(eta0, eta1);
  }
  if (family === "tweedie") {
    return tweedieLaw(eta, links, { refuse, requireNumber, requireQuantileLevel });
  }
  return refuse("family", family, '"lognormal_mean", "gamma" or "tweedie"');
}
