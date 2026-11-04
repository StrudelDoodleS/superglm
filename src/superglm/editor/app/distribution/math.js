// @ts-check
/**
 * Special functions for the distribution panel: the normal tail and quantile, ln Gamma, the
 * regularised incomplete gamma functions P and Q with the gamma quantile, and negative-binomial
 * probabilities. Each follows a published method; every coefficient table is transcribed from the
 * paper that prints it.
 *
 * Sources
 *  - Cody, W. J. (1969), "Rational Chebyshev approximations for the error function",
 *    Math. Comp. 23(107): 631-637. Table II (n = 4: erf(x), |x| <= 0.5) is ERF_P and ERF_Q;
 *    Table III (n = 8: erfc(x), 0.46875 <= x <= 4) is ERFC_P and ERFC_Q; Table IV (n = 5:
 *    erfc(x), x >= 4) is ERFC_TAIL_P and ERFC_TAIL_Q. Table I gives their maximal relative errors
 *    as 10^-19.50, 10^-18.59 and 10^-18.24.
 *  - Wichura, M. J. (1988), "Algorithm AS 241: The percentage points of the normal distribution",
 *    Applied Statistics 37(3): 477-484. The PPND16 coefficients A0..A7 and B1..B7 (PPND_A,
 *    PPND_B), C0..C7 and D1..D7 (PPND_C, PPND_D), E0..E7 and F1..F7 (PPND_E, PPND_F), with the
 *    break points 0.425 and 5 and the constants 0.180625 and 1.6. The sums of the mantissas equal
 *    the paper's printed hash sums AB, CD and EF.
 *  - Abramowitz & Stegun, Handbook of Mathematical Functions (1964), 6.1.40: the Stirling series
 *    for ln Gamma, with the Bernoulli numbers B2n / (2n (2n - 1)) as its coefficients.
 *  - Gil, Segura & Temme (2012), "Efficient and accurate algorithms for the computation and
 *    inversion of the incomplete gamma function ratios", SIAM J. Sci. Comput. 34(6): A2965-A2981:
 *    the power series (2.11) for P, Legendre's continued fraction (2.25) for Q, the lower bound
 *    (3.8) that makes Newton's iteration for a < 1 start safely, and the asymptotic tail start of
 *    section 3.2.
 *  - Wilson & Hilferty (1931), "The distribution of chi-square", PNAS 17(12): 684-688: the
 *    cube-root normal approximation that starts the gamma quantile for a >= 1.
 *  - Lentz (1976), Appl. Opt. 15(3): 668-671, and Thompson & Barnett (1986), J. Comput. Phys.
 *    64(2): 490-509: the modified Lentz evaluation of a continued fraction.
 *
 * Accuracy. Every function returns a value within a small multiple of the float64 unit roundoff
 * times the condition number of the quantity it computes. Invalid arguments throw a RangeError,
 * as does an iteration that fails to converge; no function returns NaN.
 */

const EPS = Number.EPSILON;
const ITERATION_CAP = 2000;
const NEWTON_CAP = 100;
// ln Gamma(a) is reduced to a >= STIRLING_MIN by recurrence, where the series below is exact to
// the float64 roundoff with its last term (B14 / (14 * 13)) z^-13.
export const STIRLING_MIN = 10;
export const LN_SQRT_2PI = 0.9189385332046727418;
const INV_SQRT_PI = 0.5641895835477562869;
const TINY = 1e-300;
// erfc(x) is below the smallest subnormal from here up.
const ERFC_UNDERFLOW = 27.3;
// A Newton step in ln x beyond this means the iterate is far from the root; the bracket takes over.
const MAX_LOG_STEP = 20;

const ERF_P = [
  3.209377589138469472562e3,
  3.774852376853020208137e2,
  1.138641541510501556495e2,
  3.161123743870565596947e0,
  1.857777061846031526730e-1,
];

const ERF_Q = [
  2.844236833439170622273e3,
  1.282616526077372275645e3,
  2.440246379344441733056e2,
  2.360129095234412093499e1,
  1,
];

const ERFC_P = [
  1.23033935479799725272e3,
  2.05107837782607146532e3,
  1.71204761263407058314e3,
  8.81952221241769090411e2,
  2.98635138197400131132e2,
  6.61191906371416294775e1,
  8.88314979438837594118e0,
  5.64188496988670089180e-1,
  2.15311535474403846343e-8,
];

const ERFC_Q = [
  1.23033935480374942043e3,
  3.43936767414372163696e3,
  4.36261909014324715820e3,
  3.29079923573345962678e3,
  1.62138957456669018874e3,
  5.37181101862009857509e2,
  1.17693950891312499305e2,
  1.57449261107098347253e1,
  1,
];

const ERFC_TAIL_P = [
  -6.58749161529837803157e-4,
  -1.60837851487422766278e-2,
  -1.25781726111229246204e-1,
  -3.60344899949804439429e-1,
  -3.05326634961232344035e-1,
  -1.63153871373020978498e-2,
];

const ERFC_TAIL_Q = [
  2.33520497626869185443e-3,
  6.05183413124413191178e-2,
  5.27905102951428412248e-1,
  1.87295284992346047209e0,
  2.56852019228982242072e0,
  1,
];

const PPND_A = [
  3.3871328727963666080e0,
  1.3314166789178437745e2,
  1.9715909503065514427e3,
  1.3731693765509461125e4,
  4.5921953931549871457e4,
  6.7265770927008700853e4,
  3.3430575583588128105e4,
  2.5090809287301226727e3,
];

const PPND_B = [
  1,
  4.2313330701600911252e1,
  6.8718700749205790830e2,
  5.3941960214247511077e3,
  2.1213794301586595867e4,
  3.9307895800092710610e4,
  2.8729085735721942674e4,
  5.2264952788528545610e3,
];

const PPND_C = [
  1.42343711074968357734e0,
  4.63033784615654529590e0,
  5.76949722146069140550e0,
  3.64784832476320460504e0,
  1.27045825245236838258e0,
  2.41780725177450611770e-1,
  2.27238449892691845833e-2,
  7.74545014278341407640e-4,
];

const PPND_D = [
  1,
  2.05319162663775882187e0,
  1.67638483018380384940e0,
  6.89767334985100004550e-1,
  1.48103976427480074590e-1,
  1.51986665636164571966e-2,
  5.47593808499534494600e-4,
  1.05075007164441684324e-9,
];

const PPND_E = [
  6.65790464350110377720e0,
  5.46378491116411436990e0,
  1.78482653991729133580e0,
  2.96560571828504891230e-1,
  2.65321895265761230930e-2,
  1.24266094738807843860e-3,
  2.71155556874348757815e-5,
  2.01033439929228813265e-7,
];

const PPND_F = [
  1,
  5.99832206555887937690e-1,
  1.36929880922735805310e-1,
  1.48753612908506148525e-2,
  7.86869131145613259100e-4,
  1.84631831751005468180e-5,
  1.42151175831644588870e-7,
  2.04426310338993978564e-15,
];

/**
 * @param {string} where @param {string} name @param {number} value @param {string} rule
 * @returns {never}
 */
function refuse(where, name, value, rule) {
  throw new RangeError(`${where}: ${name} must be ${rule}, got ${value}`);
}

/** @param {number} a @param {string} where */
function requireShape(a, where) {
  if (!(a > 0 && a < Infinity)) refuse(where, "a", a, "a positive finite number");
}

/** @param {number} p @param {string} where */
function requireProbability(p, where) {
  if (!(p > 0 && p < 1)) refuse(where, "p", p, "strictly between 0 and 1");
}

/**
 * Horner evaluation, coefficients in increasing order.
 * @param {readonly number[]} c @param {number} x
 */
function polynomial(c, x) {
  let r = 0;
  for (let i = c.length - 1; i >= 0; i--) r = r * x + c[i];
  return r;
}

/**
 * exp(-y^2) with the relative error of one exponential: y^2 = t^2 + (y - t)(y + t), where t is y
 * rounded to 1/16, so t^2 is exact and the correction term is small.
 * @param {number} y
 */
function expNegSquare(y) {
  const t = Math.round(y * 16) / 16;
  return Math.exp(-t * t) * Math.exp(-(y - t) * (y + t));
}

/**
 * Complementary error function, 2/sqrt(pi) times the integral of exp(-t^2) from x to infinity
 * (Cody 1969). |x| < 0.5 is 1 - erf(x) from the erf approximation, 0.5 <= |x| < 4 the erfc
 * approximation with its exponential factored out, |x| >= 4 the asymptotic form in 1/x^2.
 * @param {number} x
 * @returns {number}
 */
export function erfc(x) {
  if (Number.isNaN(x)) refuse("erfc", "x", x, "a number");
  const y = Math.abs(x);
  if (y < 0.5) {
    const z = y * y;
    return 1 - (x * polynomial(ERF_P, z)) / polynomial(ERF_Q, z);
  }
  let upper;
  if (y >= ERFC_UNDERFLOW) upper = 0;
  else if (y < 4) upper = expNegSquare(y) * (polynomial(ERFC_P, y) / polynomial(ERFC_Q, y));
  else {
    const z = 1 / (y * y);
    const correction = (z * polynomial(ERFC_TAIL_P, z)) / polynomial(ERFC_TAIL_Q, z);
    upper = (expNegSquare(y) / y) * (INV_SQRT_PI + correction);
  }
  return x < 0 ? 2 - upper : upper;
}

/**
 * Standard normal distribution function; the lower tail keeps its relative accuracy.
 * @param {number} z
 * @returns {number}
 */
export function normCdf(z) {
  if (Number.isNaN(z)) refuse("normCdf", "z", z, "a number");
  return 0.5 * erfc(-z * Math.SQRT1_2);
}

/**
 * Standard normal survival function; the upper tail keeps its relative accuracy.
 * @param {number} z
 * @returns {number}
 */
export function normSf(z) {
  if (Number.isNaN(z)) refuse("normSf", "z", z, "a number");
  return 0.5 * erfc(z * Math.SQRT1_2);
}

/**
 * Standard normal quantile, Wichura's PPND16 (accurate to about one part in 1e16).
 * @param {number} p probability, strictly between 0 and 1
 * @returns {number}
 */
export function normQuantile(p) {
  requireProbability(p, "normQuantile");
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    return (q * polynomial(PPND_A, r)) / polynomial(PPND_B, r);
  }
  const tail = Math.sqrt(-Math.log(q < 0 ? p : 1 - p));
  let z;
  if (tail <= 5) {
    const r = tail - 1.6;
    z = polynomial(PPND_C, r) / polynomial(PPND_D, r);
  } else {
    const r = tail - 5;
    z = polynomial(PPND_E, r) / polynomial(PPND_F, r);
  }
  return q < 0 ? -z : z;
}

/**
 * ln Gamma(z) - [(z - 1/2) ln z - z + ln sqrt(2 pi)] for z >= STIRLING_MIN: the series of
 * Abramowitz & Stegun 6.1.40, whose size is 1 / (12 z).
 * @param {number} z
 */
export function stirlingSeries(z) {
  const w = 1 / (z * z);
  return (
    (1 / 12 +
      w *
        (-1 / 360 +
          w * (1 / 1260 + w * (-1 / 1680 + w * (1 / 1188 + w * (-691 / 360360 + w / 156)))))) /
    z
  );
}

/**
 * ln Gamma(z) for z >= STIRLING_MIN: Stirling's series (Abramowitz & Stegun 6.1.40).
 * @param {number} z
 */
function stirling(z) {
  return (z - 0.5) * Math.log(z) - z + LN_SQRT_2PI + stirlingSeries(z);
}

/**
 * ln Gamma(a) for a > 0: the Stirling series after the upward recurrence
 * Gamma(a) = Gamma(a + n) / (a (a + 1) ... (a + n - 1)) to a + n >= 10.
 * @param {number} a
 * @returns {number}
 */
export function lnGamma(a) {
  requireShape(a, "lnGamma");
  if (a >= STIRLING_MIN) return stirling(a);
  // Each factor a + k is rounded once from the exact sum; accumulating z += 1 would round a itself.
  let product = 1;
  let steps = 0;
  while (a + steps < STIRLING_MIN) product *= a + steps++;
  return stirling(a + steps) - Math.log(product);
}

/**
 * The factor x^a exp(-x) / Gamma(a) in P and Q, computed in log space.
 * @param {number} a @param {number} x
 */
function gammaPrefactor(a, x) {
  return Math.exp(a * Math.log(x) - x - lnGamma(a));
}

/**
 * P(a, x) by the power series sum x^n / (a)_{n+1} (Gil, Segura & Temme 2.11); its terms fall
 * from the first when x < a + 1.
 * @param {number} a @param {number} x
 */
function lowerSeries(a, x) {
  let term = 1 / a;
  let sum = term;
  for (let n = 1; n <= ITERATION_CAP; n++) {
    term *= x / (a + n);
    sum += term;
    if (term <= sum * EPS) return sum * gammaPrefactor(a, x);
  }
  throw new RangeError(
    `gammaP: the series for a = ${a}, x = ${x} did not converge in ${ITERATION_CAP} terms`
  );
}

/**
 * Q(a, x) by Legendre's continued fraction
 * 1 / (x + 1 - a - 1 (1 - a) / (x + 3 - a - 2 (2 - a) / (x + 5 - a - ...))),
 * evaluated by the modified Lentz method; it converges for x >= a + 1.
 * @param {number} a @param {number} x
 */
function upperFraction(a, x) {
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= ITERATION_CAP; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const ratio = d * c;
    h *= ratio;
    if (Math.abs(ratio - 1) <= EPS) return h * gammaPrefactor(a, x);
  }
  throw new RangeError(
    `gammaQ: the continued fraction for a = ${a}, x = ${x} did not converge ` +
      `in ${ITERATION_CAP} iterations`
  );
}

/**
 * [P(a, x), Q(a, x)]: the series for x < a + 1 and the continued fraction beyond, each giving the
 * smaller-error tail directly and the other by subtraction from 1.
 * @param {number} a @param {number} x
 * @returns {[number, number]}
 */
function gammaPair(a, x) {
  if (x === 0) return [0, 1];
  if (x === Infinity) return [1, 0];
  if (x < a + 1) {
    const p = lowerSeries(a, x);
    return [p, 1 - p];
  }
  const q = upperFraction(a, x);
  return [1 - q, q];
}

/** @param {number} a @param {number} x @param {string} where */
function requireGammaArguments(a, x, where) {
  requireShape(a, where);
  if (!(x >= 0)) refuse(where, "x", x, "a number of zero or more");
}

/**
 * Regularised lower incomplete gamma function P(a, x) = Gamma(a, 0..x) / Gamma(a).
 * @param {number} a shape, positive @param {number} x zero or more
 * @returns {number}
 */
export function gammaP(a, x) {
  requireGammaArguments(a, x, "gammaP");
  return gammaPair(a, x)[0];
}

/**
 * Regularised upper incomplete gamma function Q(a, x) = 1 - P(a, x).
 * @param {number} a shape, positive @param {number} x zero or more
 * @returns {number}
 */
export function gammaQ(a, x) {
  requireGammaArguments(a, x, "gammaQ");
  return gammaPair(a, x)[1];
}

/**
 * The Wilson-Hilferty cube root (x / a)^(1/3) of the gamma quantile at the standard normal z:
 * 1 - 1 / (9a) + z / (3 sqrt(a)), for shape a.
 * @param {number} a shape @param {number} z standard normal quantile
 */
export function wilsonHilfertyRoot(a, z) {
  return 1 - 1 / (9 * a) + z / (3 * Math.sqrt(a));
}

/**
 * ln x0 to start Newton's iteration for the gamma quantile, and the lower bound ln x_l <= ln x.
 * x_l = (p Gamma(a + 1))^(1/a) bounds the root from below for every a (Gil, Segura & Temme 3.8)
 * and is close to it in the lower tail. For a >= 1 the Wilson-Hilferty cube root starts the
 * iteration; in the upper tail with a < 1 the solution of x^a exp(-x) = q Gamma(a) (section 3.2).
 * @param {number} a @param {number} p @param {number} target the smaller tail probability
 * @param {number} lowerBound ln x_l
 */
function quantileStart(a, p, target, lowerBound) {
  if (a >= 1) {
    const base = wilsonHilfertyRoot(a, normQuantile(p));
    return base > 0 ? Math.max(lowerBound, Math.log(a) + 3 * Math.log(base)) : lowerBound;
  }
  if (p <= 0.5) return lowerBound;
  const level = -Math.log(target) - lnGamma(a);
  if (!(level > 1)) return lowerBound;
  let x = level;
  for (let i = 0; i < 3; i++) x = level + a * Math.log(x);
  return Math.max(lowerBound, Math.log(x));
}

/**
 * The x with P(a, x) = p: Newton's iteration in u = ln x on P (or on Q when p > 0.5, solving
 * Q = 1 - p, which is exact there), kept inside a bracket that bisection takes over when a step
 * leaves it. The derivative is dP/du = x^a exp(-x) / Gamma(a). It stops when the step in u is at
 * most 4 eps, or when steps stop shrinking at a rounding floor of P itself.
 * @param {number} a shape, positive @param {number} p probability, strictly between 0 and 1
 * @returns {number}
 */
export function gammaQuantile(a, p) {
  requireShape(a, "gammaQuantile");
  requireProbability(p, "gammaQuantile");
  const upper = p > 0.5;
  const target = upper ? 1 - p : p;
  const lowerBound = (Math.log(p) + lnGamma(a + 1)) / a;
  // Below this x^a / Gamma(a + 1) is P to the full precision, and x is not representable anyway.
  if (lowerBound < -700) return Math.exp(lowerBound);
  const lnGammaA = lnGamma(a);
  let u = quantileStart(a, p, target, lowerBound);
  let lo = lowerBound;
  let hi = Infinity;
  let previous = Infinity;
  for (let iteration = 0; iteration < NEWTON_CAP; iteration++) {
    const x = Math.exp(u);
    const [lower, tail] = gammaPair(a, x);
    const residual = upper ? target - tail : lower - target;
    if (residual === 0) return x;
    if (residual < 0) lo = Math.max(lo, u);
    else hi = Math.min(hi, u);
    const slope = Math.exp(a * u - x - lnGammaA);
    const step = slope > 0 ? -residual / slope : residual < 0 ? Infinity : -Infinity;
    const size = Math.abs(step);
    if (size <= 4 * EPS || (size < 1e-6 && size >= previous)) return Math.exp(u + step);
    previous = size;
    let next = u + Math.max(-MAX_LOG_STEP, Math.min(MAX_LOG_STEP, step));
    if (!(next > lo && next < hi)) next = Number.isFinite(hi) ? 0.5 * (lo + hi) : u + 1;
    u = next;
  }
  throw new RangeError(
    `gammaQuantile: no convergence for a = ${a}, p = ${p} in ${NEWTON_CAP} iterations`
  );
}

/** @param {number} k @param {number} mean @param {number} theta @param {string} where */
function requireNegativeBinomial(k, mean, theta, where) {
  if (!(Number.isInteger(k) && k >= 0)) refuse(where, "k", k, "a whole number of zero or more");
  if (!(mean >= 0 && mean < Infinity)) {
    refuse(where, "mean", mean, "a finite number of zero or more");
  }
  if (!(theta > 0 && theta < Infinity)) refuse(where, "theta", theta, "a positive finite number");
}

/**
 * Negative binomial probabilities by the recurrence
 * p_{j+1} = p_j (j + theta) / (j + 1) * mean / (theta + mean),
 * from p_0 = (theta / (theta + mean))^theta.
 * @param {number} k count @param {number} mean @param {number} theta
 * @param {(j: number, p: number) => void} visit called with each j < k and p_j
 * @returns {number} p_k
 */
function negativeBinomialWalk(k, mean, theta, visit) {
  const ratio = mean / (theta + mean);
  let p = Math.exp(theta * Math.log1p(-ratio));
  for (let j = 0; j < k; j++) {
    visit(j, p);
    p *= ((j + theta) / (j + 1)) * ratio;
  }
  return p;
}

/**
 * P(Y = k) for a negative binomial with the given mean and size theta, whose variance is
 * mean + mean^2 / theta.
 * @param {number} k @param {number} mean @param {number} theta
 * @returns {number}
 */
export function nbPmf(k, mean, theta) {
  requireNegativeBinomial(k, mean, theta, "nbPmf");
  return negativeBinomialWalk(k, mean, theta, () => {});
}

/**
 * P(Y >= k) = 1 - sum_{j<k} P(Y = j) for the same law.
 * @param {number} k @param {number} mean @param {number} theta
 * @returns {number}
 */
export function nbSf(k, mean, theta) {
  requireNegativeBinomial(k, mean, theta, "nbSf");
  let below = 0;
  negativeBinomialWalk(k, mean, theta, (_, p) => {
    below += p;
  });
  return Math.max(0, 1 - below);
}
