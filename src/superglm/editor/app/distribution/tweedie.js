// @ts-check
/**
 * The predicted law of one Tweedie row with a power between one and two, rebuilt in the browser
 * from the link-scale predictors [ln mean, ln dispersion, bounded-logit power] so that the
 * distribution panel can redraw while a curve is dragged.
 *
 * The law. For 1 < p < 2 a Tweedie variable with mean mu, dispersion phi and power p is a compound
 * Poisson sum Y = X_1 + ... + X_N of gamma claims, N ~ Poisson(lambda), with
 *    lambda = mu^(2-p) / (phi (2-p)),  claim shape a = (2-p) / (p-1),  claim scale g = phi (p-1) mu^(p-1)
 * (Dunn & Smyth 2005, section 3, which cites Jorgensen 1997, p. 141 for the representation; the
 * book itself was not read). Y is exactly zero with probability exp(-lambda), and given N = n >= 1
 * it is Gamma(n a, g). Conditioning on N gives the other functions as Poisson-weighted sums, with
 * b_n = exp(-lambda) lambda^n / n! (this derivation is ours; the density series is theirs):
 *    P(Y <= y) = exp(-lambda) + sum_{n>=1} b_n P(n a, y / g)        P the regularised lower gamma
 *    P(Y >  y) = sum_{n>=1} b_n Q(n a, y / g)                       Q = 1 - P, kept as its own sum
 *    y f(y)    = sum_{n>=1} b_n (y/g)^(n a) exp(-y/g) / Gamma(n a)  the density of the positive part
 *
 * Which terms are summed. Dunn & Smyth 2005, section 4: the terms W_j of the density series are
 * those above with j = n, ln W_j is concave in j (a linear term minus ln Gamma(j + 1) minus
 * ln Gamma(a j), and ln Gamma is convex), its maximum is within one of j_max = y^(2-p) / ((2-p) phi),
 * and the terms fall faster than geometrically either side of it. Their stopping rule is a term
 * below exp(-37) of the largest; here the sum stops when the exact geometric bound on the terms left
 * out, t_k r / (1 - r) for the last ratio r of consecutive terms, is below 2^-53 of the sum, which
 * holds for any concave ln t_n. So the density is a sum of positive terms with relative truncation
 * error of one unit roundoff. Their Table 1 gives the number of terms: it grows without bound as p
 * nears 2, phi nears 0 or y grows, and full machine accuracy is out of reach only for p > 2.
 *
 * The distribution and survival functions use the same window idea on the Poisson weights alone.
 * Every gamma factor lies in [0, 1], so the terms left out are at most the Poisson mass left out,
 * which log-concavity of b_n bounds the same way (b_{n+1} / (1 - lambda / (n + 2)) above n, and
 * b_{n-1} / (1 - (n - 1) / lambda) below it). The window grows from the Poisson mode until the
 * mass left out at each end is below 2^-53 of max(the sum so far, SUM_FLOOR), so the sum has
 * relative truncation error 2^-53 for sums above 1e-30 and absolute error 1e-46 below it. The
 * survival function takes its terms from Q, which math.js evaluates to relative accuracy, so it
 * keeps its relative accuracy in the upper tail.
 *
 * What the method assumes and where the editor sits. It assumes 1 < p < 2 (the compound Poisson
 * regime; the editor's walls are 1.05 and 1.95 and refuse anything else), a finite window of terms
 * (the rate and the claim shapes below), and gamma functions at shapes up to TWEEDIE_SHAPE_MAX.
 * Both walls are inside it: at p = 1.95 the claims have shape 0.0526 and the sum needs terms out to
 * where lambda is large, and at p = 1.05 the shape is 19 and the law is nearly a lattice (Dunn &
 * Smyth, section 8: modes near n a g when sqrt(lambda / a) is below about 0.5), which the density
 * draws as it is. The coefficient of variation is CV = 1 / sqrt(lambda (2-p)), so a row with a CV
 * of 1 or more has lambda below about 20, and a chance of zero of 1 in 10,000 is lambda = 9.2; the
 * domain below is far wider than that.
 *
 * Accuracy. Each term carries a relative error of a few units of roundoff times its exponent's
 * size, about 2 lambda ln(lambda) u (u = 2^-53): 3e-14 at lambda = 100 and about 2e-12 at the cap.
 * The distribution function, survival function and density agree with the same sums in 40-digit
 * arithmetic to 4e-13 relative or better up to lambda = 300, the largest tested, and the Python
 * family's series agrees with the distribution function within its own Poisson tail tolerance of
 * 1e-12. `quantile` inverts the distribution function by Newton's iteration in ln y, to a step of
 * 1e-12, so the probability it inverts is met to about 1e-12 times the density's slope.
 *
 * Domain, and what throws. `tweedieLaw` throws a RangeError unless
 *  - eta is three finite numbers and the links are log, log and a bounded logit with
 *    1 < lower < upper < 2;
 *  - the mean, dispersion, rate lambda and claim scale are positive and within float64, with lambda
 *    between 1e-300 and TWEEDIE_RATE_MAX;
 *  - the claim shape a times the largest Poisson order a window can reach,
 *    lambda + 20 sqrt(lambda) + 50, is at most TWEEDIE_SHAPE_MAX.
 * Inside it `cdf`, `sf` and `log10Density` throw only for a NaN argument, and `quantile` throws for
 * a probability outside the range the other laws take. Float64 only: `log10Density` returns
 * -Infinity where j_max is above WALK_MAX (the density is below exp(-7e5) there), and the amount
 * where the quantile would be below the smallest float64 is returned as 0 or a subnormal.
 */

import { gammaP, gammaQ, gammaQuantile, lnGamma } from "./math.js";

/** @typedef {import('./families.js').Law} Law */
/**
 * What this module borrows from families.js, whose guards it shares and which imports it.
 * @typedef {object} Guards
 * @property {(what: string, got: unknown, rule: string) => never} refuse
 * @property {(value: number, what: string) => void} requireNumber
 * @property {(p: number) => number} requireQuantileLevel
 */

/** The largest compound Poisson rate (mean claim count) a law is built for. */
export const TWEEDIE_RATE_MAX = 2000;
/** The largest gamma shape a window of terms may reach; math.js is accurate to about 5e4. */
export const TWEEDIE_SHAPE_MAX = 3e4;

const LN10 = Math.LN10;
const LN_LN10 = Math.log(Math.LN10);
const TAIL = 2 ** -53;
const SUM_FLOOR = 1e-30;
const WALK_MAX = 200000;
const TABLE_MAX = 4096;
const RATE_MIN_LN = Math.log(1e-300);
const LN_LIMIT = 700;
const EXP_MAX = 709;
const LN_TINY = -745;
const NEWTON_CAP = 100;
const MAX_LOG_STEP = 20;
// The density's peak search: golden section to this relative width, and the smallest rise over the
// bracket's ends that counts as a peak.
const GOLDEN = (Math.sqrt(5) - 1) / 2;
const PEAK_WIDTH = 1e-8;
const PEAK_RISE = 1e-6;

/**
 * ln Gamma at the integers n + 1 and at n times the claim shape: the sums over claim counts need
 * the same values again and again, as neighbouring evaluations share their windows, and ln Gamma
 * costs more than a lookup. A table belongs to one law, grows on demand to TABLE_MAX entries, and
 * never needs invalidating because the law's claim shape never changes.
 * @param {number} shape
 */
function lnGammaTables(shape) {
  let capacity = 64;
  let facts = new Float64Array(capacity);
  let shapes = new Float64Array(capacity);
  let filled = 0;
  /** @param {number} n */
  const grow = (n) => {
    if (n >= capacity) {
      while (capacity <= n) capacity *= 2;
      const nextFacts = new Float64Array(capacity);
      const nextShapes = new Float64Array(capacity);
      nextFacts.set(facts);
      nextShapes.set(shapes);
      facts = nextFacts;
      shapes = nextShapes;
    }
    for (let k = filled + 1; k <= n; k++) {
      facts[k] = lnGamma(k + 1);
      shapes[k] = lnGamma(k * shape);
    }
    filled = n;
  };
  return {
    /** ln n! @param {number} n */
    lnFactorial(n) {
      if (n > TABLE_MAX) return lnGamma(n + 1);
      if (n > filled) grow(n);
      return facts[n];
    },
    /** ln Gamma(n a) @param {number} n */
    lnClaims(n) {
      if (n > TABLE_MAX) return lnGamma(n * shape);
      if (n > filled) grow(n);
      return shapes[n];
    },
  };
}

/**
 * The power as p - 1 and 2 - p, each formed without cancellation near its wall:
 * p = lower + (upper - lower) / (1 + exp(-eta)).
 * @param {number} eta @param {number} lower @param {number} upper
 * @returns {{power: number, jump: number, tail: number}}
 */
function powerParts(eta, lower, upper) {
  const e = Math.exp(-Math.abs(eta));
  const small = e / (1 + e);
  const sigma = eta >= 0 ? 1 - small : small;
  const complement = eta >= 0 ? small : 1 - small;
  const span = upper - lower;
  return {
    power: lower + span * sigma,
    jump: lower - 1 + span * sigma,
    tail: 2 - upper + span * complement,
  };
}

/**
 * @param {readonly number[]} eta @param {readonly unknown[]} links @param {Guards} guards
 * @returns {{eta0: number, eta1: number, lower: number, upper: number, eta2: number}}
 */
function readInputs(eta, links, guards) {
  const { refuse } = guards;
  if (!Array.isArray(eta) || eta.length !== 3 || !eta.every(Number.isFinite)) {
    refuse("eta", eta, "three finite link-scale numbers [mean, dispersion, power]");
  }
  if (!Array.isArray(links) || links.length !== 3) {
    refuse("links", links, "three link specs [mean, dispersion, power]");
  }
  const [mean, dispersion, wall] = /** @type {any[]} */ (links);
  if (mean?.kind !== "log") refuse("the mean link", mean, 'a link of kind "log"');
  if (dispersion?.kind !== "log") refuse("the dispersion link", dispersion, 'a link of kind "log"');
  if (wall?.kind !== "bounded_logit") {
    refuse("the power link", wall, 'a link of kind "bounded_logit"');
  }
  if (!(wall.lower > 1 && wall.upper < 2 && wall.lower < wall.upper)) {
    refuse("the power link's walls", [wall.lower, wall.upper], "numbers with 1 < lower < upper < 2");
  }
  return { eta0: eta[0], eta1: eta[1], eta2: eta[2], lower: wall.lower, upper: wall.upper };
}

/**
 * The predicted distribution of one Tweedie row.
 * @param {readonly number[]} eta link-scale predictors [mean, dispersion, power]
 * @param {readonly unknown[]} links the links for [mean, dispersion, power]
 * @param {Guards} guards
 * @returns {Law}
 */
export function tweedieLaw(eta, links, guards) {
  const { refuse, requireNumber, requireQuantileLevel } = guards;
  const { eta0, eta1, eta2, lower, upper } = readInputs(eta, links, guards);
  const { power, jump, tail } = powerParts(eta2, lower, upper);
  const mean = Math.exp(eta0);
  const logRate = tail * eta0 - eta1 - Math.log(tail);
  const logScale = eta1 + Math.log(jump) + jump * eta0;
  const rate = Math.exp(logRate);
  const shape = tail / jump;
  const top = rate + 20 * Math.sqrt(rate) + 50;
  const inside =
    Number.isFinite(mean) && mean > 0 && Number.isFinite(rate) &&
    logRate > RATE_MIN_LN && rate <= TWEEDIE_RATE_MAX &&
    Math.abs(logScale) < LN_LIMIT && top * shape <= TWEEDIE_SHAPE_MAX;
  if (!inside) {
    const got = { mean, rate, claimShape: shape, claimScale: Math.exp(logScale) };
    refuse(
      "the Tweedie mean, dispersion and power",
      got,
      `within the browser's range: a claim rate between 1e-300 and ${TWEEDIE_RATE_MAX}, claim shape times its largest term at most ${TWEEDIE_SHAPE_MAX}, and finite float64 values`
    );
  }
  const scale = Math.exp(logScale);
  const zero = Math.exp(-rate);
  const positive = -Math.expm1(-rate);
  const tables = lnGammaTables(shape);
  const poissonMode = Math.max(1, Math.floor(rate));

  /** ln b_n, the Poisson log-probability of n claims. @param {number} n */
  const lnWeight = (n) => -rate + n * logRate - tables.lnFactorial(n);
  /** @param {number} n */
  const weight = (n) => Math.exp(lnWeight(n));

  /**
   * sum_{n>=1} b_n c(n) for gamma factors c in [0, 1], the window grown from the Poisson mode until
   * the Poisson mass left out at each end is below TAIL of the sum.
   * @param {(n: number) => number} c
   */
  function mixture(c) {
    let low = poissonMode;
    let high = poissonMode;
    let sum = weight(low) * c(low);
    for (;;) {
      const limit = TAIL * Math.max(sum, SUM_FLOOR);
      let grew = false;
      if (weight(high + 1) / (1 - rate / (high + 2)) > limit) {
        high += 1;
        sum += weight(high) * c(high);
        grew = true;
      }
      if (low > 1 && weight(low - 1) / (1 - (low - 1) / rate) > limit) {
        low -= 1;
        sum += weight(low) * c(low);
        grew = true;
      }
      if (!grew) return sum;
      if (high > top || high - low > WALK_MAX) {
        throw new RangeError(`tweedieLaw: the sum over claim counts ran past ${Math.round(top)} terms`);
      }
    }
  }

  /** ln of the sum over n of the nth term of y f(y), at ln y = u. @param {number} u */
  function logMass(u) {
    const s = u - logScale;
    if (s > EXP_MAX) return -Infinity;
    const x = Math.exp(s);
    /** @param {number} n */
    const term = (n) => lnWeight(n) + n * shape * s - x - tables.lnClaims(n);
    const guessLn = tail * u - Math.log(tail) - eta1;
    // From j_max above WALK_MAX the density is below exp(-7e5): the Poisson weight at 100 times the
    // rate (the cap is 2000, so j_max is always at least that far out) is already that small.
    if (guessLn > Math.log(WALK_MAX)) return -Infinity;
    let n = Math.max(1, Math.round(Math.exp(guessLn)));
    let peak = term(n);
    // The terms are log-concave in n, so walking uphill from the guess reaches the exact maximum.
    for (let up = term(n + 1); up > peak; up = term(n + 1)) {
      n += 1;
      peak = up;
    }
    for (let down = n > 1 ? term(n - 1) : -Infinity; down > peak; down = n > 1 ? term(n - 1) : -Infinity) {
      n -= 1;
      peak = down;
    }
    let sum = 1;
    for (const step of [1, -1]) {
      let previous = 0;
      for (let k = n + step; k >= 1; k += step) {
        const relative = term(k) - peak;
        const ratio = Math.exp(relative - previous);
        const added = Math.exp(relative);
        sum += added;
        previous = relative;
        if (ratio < 1 && (added * ratio) / (1 - ratio) <= TAIL * sum) break;
        if (Math.abs(k - n) > WALK_MAX) throw new RangeError("tweedieLaw: density sum did not close");
      }
    }
    return peak + Math.log(sum);
  }

  /** ln of the density of the positive part at y. @param {number} y */
  const logDensity = (y) => logMass(Math.log(y)) - Math.log(y);

  /** P(Y <= y) less the atom, for y > 0. @param {number} y */
  const cdfAbove = (y) => mixture((n) => gammaP(n * shape, y / scale));
  /** @param {number} y */
  const sfAbove = (y) => Math.min(positive, mixture((n) => gammaQ(n * shape, y / scale)));

  /**
   * ln y to start the quantile search from: the quantile of the gamma with the positive part's mean
   * and variance, at the same level of the positive part.
   * @param {number} level the level in (0, 1) of the positive part, for the upper tail the level
   *   left of it; NaN when no gamma can start it
   */
  function startFor(level) {
    const second = Math.exp(eta1 + power * eta0) + mean * mean;
    const m = mean / positive;
    const v = second / positive - m * m;
    if (level > 0 && level < 1 && v > 0 && Number.isFinite(v) && Number.isFinite(m)) {
      try {
        const u = Math.log((v / m) * gammaQuantile((m * m) / v, level));
        if (Number.isFinite(u)) return Math.min(EXP_MAX, Math.max(LN_TINY, u));
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
      }
    }
    return Math.min(EXP_MAX, Math.max(LN_TINY, Math.log(mean)));
  }

  /** @param {number} p */
  function quantile(p) {
    requireQuantileLevel(p);
    if (p <= zero) return 0;
    const upperTail = p > 0.5;
    const target = upperTail ? 1 - p : p - zero;
    const level = upperTail ? 1 - (1 - p) / positive : (p - zero) / positive;
    // g is increasing in u = ln y and zero at the quantile; its slope is y f(y) = exp(logMass(u)).
    /** @param {number} u */
    const g = (u) => (upperTail ? target - sfAbove(Math.exp(u)) : cdfAbove(Math.exp(u)) - target);
    let u = startFor(level);
    let lo = LN_TINY;
    let hi = Infinity;
    let previous = Infinity;
    for (let iteration = 0; iteration < NEWTON_CAP; iteration++) {
      const residual = g(u);
      if (residual === 0) return Math.exp(u);
      if (residual < 0) lo = Math.max(lo, u);
      else hi = Math.min(hi, u);
      const slope = Math.exp(logMass(u));
      const step = slope > 0 ? -residual / slope : residual < 0 ? Infinity : -Infinity;
      const size = Math.abs(step);
      if (size <= 1e-12 || (size < 1e-8 && size >= previous)) {
        return Math.exp(Math.min(EXP_MAX, Math.max(LN_TINY, u + step)));
      }
      previous = size;
      let next = u + Math.max(-MAX_LOG_STEP, Math.min(MAX_LOG_STEP, step));
      if (!(next > lo && next < hi)) next = Number.isFinite(hi) ? 0.5 * (lo + hi) : u + 1;
      if (next <= LN_TINY) return Math.exp(LN_TINY);
      u = next;
    }
    throw new RangeError(`tweedieLaw: no quantile convergence for p = ${p} in ${NEWTON_CAP} iterations`);
  }

  /**
   * The highest peak of the positive part's density near the Poisson mode, or 0 when the density
   * has none there (it falls from zero). The peak of the claim count n sits at (n a - 1) g with
   * spread sqrt(n a) g; the mixture's is inside four spreads of the count nearest where the
   * Poisson weight over sqrt(n) is largest, or of the next.
   * @returns {number}
   */
  function densityMode() {
    let best = 0;
    let bestLn = -Infinity;
    const first = Math.max(1, Math.floor(rate - 0.5));
    for (const n of [first, first + 1]) {
      if (!(n * shape > 1)) continue;
      const centre = scale * (n * shape - 1);
      const spread = scale * Math.sqrt(n * shape);
      const hiEnd = centre + 4 * spread;
      const loEnd = Math.max(centre - 4 * spread, hiEnd * 1e-9);
      const ends = Math.max(logDensity(loEnd), logDensity(hiEnd));
      let a = loEnd;
      let b = hiEnd;
      let c = b - GOLDEN * (b - a);
      let d = a + GOLDEN * (b - a);
      let fc = logDensity(c);
      let fd = logDensity(d);
      while (b - a > PEAK_WIDTH * hiEnd) {
        if (fc > fd) {
          b = d;
          d = c;
          fd = fc;
          c = b - GOLDEN * (b - a);
          fc = logDensity(c);
        } else {
          a = c;
          c = d;
          fc = fd;
          d = a + GOLDEN * (b - a);
          fd = logDensity(d);
        }
      }
      const y = 0.5 * (a + b);
      const value = logDensity(y);
      if (value > ends + PEAK_RISE && value > bestLn) {
        best = y;
        bestLn = value;
      }
    }
    return best;
  }

  /** @type {number | undefined} */
  let medianValue;
  /** @type {number | undefined} */
  let modeValue;
  return {
    mean,
    zero,
    get median() {
      if (medianValue === undefined) medianValue = quantile(0.5);
      return medianValue;
    },
    get mode() {
      if (modeValue === undefined) modeValue = zero >= 0.5 ? 0 : densityMode();
      return modeValue;
    },
    cdf(y) {
      requireNumber(y, "y");
      if (y < 0) return 0;
      if (y === 0) return zero;
      if (y === Infinity) return 1;
      return Math.min(1, zero + cdfAbove(y));
    },
    sf(y) {
      requireNumber(y, "y");
      if (y < 0) return 1;
      if (y === 0) return positive;
      if (y === Infinity) return 0;
      return sfAbove(y);
    },
    quantile,
    log10Density(l) {
      requireNumber(l, "l");
      if (!Number.isFinite(l)) return -Infinity;
      return (LN_LN10 + logMass(l * LN10)) / LN10;
    },
  };
}
