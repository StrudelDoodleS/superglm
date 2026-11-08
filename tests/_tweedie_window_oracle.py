"""Independent high precision compound-law references, for tiny fixtures only."""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np


def gamma_window(lower, upper, shape, log_scale):
    scale = mp.exp(log_scale)
    lo = max(mp.mpf(lower), 0) / scale
    hi = mp.mpf(upper) / scale
    return mp.gammainc(shape, lo, hi) / mp.gamma(shape)


def window(lower, upper, mean, dispersion, power, weight=1, terms=100):
    mean, dispersion, power, weight = map(mp.mpf, (mean, dispersion, power, weight))
    rate = weight * mean ** (2 - power) / (dispersion * (2 - power))
    shape = (2 - power) / (power - 1)
    scale = dispersion * (power - 1) * mean ** (power - 1) / weight
    if upper <= lower or upper < 0:
        return mp.mpf(0)
    atom = mp.exp(-rate) if lower < 0 <= upper else mp.mpf(0)
    if upper == 0:
        return atom
    if upper == mp.inf and lower <= 0:
        return mp.mpf(1) if lower < 0 else -mp.expm1(-rate)
    return atom + mp.fsum(
        mp.exp(-rate)
        * rate**j
        / mp.factorial(j)
        * gamma_window(lower, upper, j * shape, mp.log(scale))
        for j in range(1, terms + 1)
    )


def density(y, mean, dispersion, power, weight=1, terms=100):
    mean, dispersion, power, weight = map(mp.mpf, (mean, dispersion, power, weight))
    rate = weight * mean ** (2 - power) / (dispersion * (2 - power))
    if y == 0:
        return mp.exp(-rate)
    a = (2 - power) / (power - 1)
    b = dispersion * (power - 1) * mean ** (power - 1) / weight
    return mp.fsum(
        mp.exp(-rate)
        * rate**j
        / mp.factorial(j)
        * mp.exp(-mp.mpf(y) / b)
        * mp.mpf(y) ** (j * a - 1)
        / (mp.gamma(j * a) * b ** (j * a))
        for j in range(1, terms + 1)
    )


def log_jet(function, point):
    """High precision differences of the independent probability.

    At 60 digits and h=1e-12, second-difference rounding is below 1e-35;
    truncation is O(1e-24) times the fourth derivative on these tiny fixtures.
    This avoids mpmath.diff's much more expensive internal precision lift.
    No production derivative, range or recurrence participates.
    """
    with mp.workdps(60):
        point = tuple(map(mp.mpf, point))
        k = len(point)

        def f(*args):
            return mp.log(function(*args))

        value = f(*point)
        step = mp.mpf("1e-12")

        def at(i, sign, j=None, other=0):
            moved = list(point)
            moved[i] += sign * step
            if j is not None:
                moved[j] += other * step
            return f(*moved)

        plus, minus = [at(i, 1) for i in range(k)], [at(i, -1) for i in range(k)]
        score = [float((plus[i] - minus[i]) / (2 * step)) for i in range(k)]
        hessian = []
        for i, j in combinations_with_replacement(range(k), 2):
            if i == j:
                entry = (plus[i] - 2 * value + minus[i]) / step**2
            else:
                entry = (at(i, 1, j, 1) - at(i, 1, j, -1) - at(i, -1, j, 1) + at(i, -1, j, -1)) / (
                    4 * step**2
                )
            hessian.append(float(entry))
    return np.array([float(value)]), np.array([score]), np.array([hessian])


def scipy_rows(y, lower, upper, side, theta, weights, semantics):
    """Long, fixed independent mixture for ordinary-domain fit references.

    Evaluates 240 terms using SciPy's gamma functions, with no production
    range, stopping, endpoint or derivative routines. The test fixtures have
    rate below 30; a Chernoff tail at 240 is below 1e-120.
    """
    from scipy import special

    theta = np.broadcast_to(theta, (len(y), 3))
    mu, phi, p = theta.T
    w = weights if semantics == "prior" else np.ones(len(y))
    mass = weights if semantics == "frequency" else np.ones(len(y))
    rate = w * mu ** (2 - p) / (phi * (2 - p))
    a = (2 - p) / (p - 1)
    b = phi * (p - 1) * mu ** (p - 1) / w
    j = np.arange(1.0, 241.0)[:, None]
    shape = j * a
    log_mass = j * np.log(rate) - rate - special.gammaln(j + 1)

    def probability(lo, hi):
        lo, hi = np.maximum(lo, 0) / b, np.maximum(hi, 0) / b
        cdf = special.gammainc(shape, hi) - special.gammainc(shape, lo)
        sf = special.gammaincc(shape, lo) - special.gammaincc(shape, hi)
        gamma = np.where(lo >= shape, sf, cdf)
        return np.sum(np.exp(log_mass) * gamma, axis=0)

    def win(lo, hi):
        return probability(lo, hi) + np.where((lo < 0) & (hi >= 0), np.exp(-rate), 0)

    positive_y = np.maximum(y, 1e-100)
    log_density = special.logsumexp(
        log_mass
        + (shape - 1) * np.log(positive_y)
        - positive_y / b
        - special.gammaln(shape)
        - shape * np.log(b),
        axis=0,
    )
    numerator = np.where(y == 0, -rate, log_density)
    censored = side != 0
    if np.any(censored):
        lo = np.where(side == 1, y, lower)
        hi = np.where(side == -1, y, upper)
        numerator[censored] = np.log(win(lo, hi))[censored]
    return mass * (numerator - np.log(win(lower, upper)))


def fit_fixture(semantics, n=96):
    """Independent compound sampling, mixed bounds, both censors and exact zeros."""
    rng = np.random.default_rng(947)
    x = np.linspace(-1, 1, n)
    weights = rng.integers(1, 3, n).astype(float)
    w = weights if semantics == "prior" else np.ones(n)
    mu, phi, p = 1.8, 1.4, 1.5
    counts = rng.poisson(w * mu ** (2 - p) / (phi * (2 - p)))
    y = np.zeros(n)
    present = counts > 0
    y[present] = rng.gamma(
        counts[present] * (2 - p) / (p - 1), phi * (p - 1) * mu ** (p - 1) / w[present]
    )
    lower = np.where(np.arange(n) % 5 == 0, 0.1, -np.inf)
    upper = np.where(np.arange(n) % 3 == 0, 5.0, np.inf)
    keep = (y > lower) & (y <= upper)
    y, weights, x, lower, upper = (v[keep] for v in (y, weights, x, lower, upper))
    side = np.zeros(len(y), dtype=int)
    side[(y > 0) & (y < 0.4)] = -1
    side[y > 3.0] = 1
    y = np.where(side == -1, 0.4, np.where(side == 1, 3.0, y))
    return x, y, lower, upper, side, weights


def gamma_iteration_observer(a, x, upper, order, along):
    """UNTIMED work instrumentation: replay unchanged primitive recurrences.

    Small-shape's public helper returns bool, so its Python equivalent is
    traced only to read its loop counter. Numerical oracle functions above
    do not use this observer. Production files/dependency remain unchanged.
    """
    import sys

    from superglm.distributional.kernels import _incomplete_gamma as engine

    scratch = np.empty(engine.WORK_SIZE)
    count = 0
    n = (order + 1) * (order + 2) // 2
    series = x <= a + max(1.0, np.sqrt(a))
    if series and a < engine.SMALL_SHAPE:

        def trace(frame, event, arg):
            nonlocal count
            if frame.f_code is engine._small_shape_log_q.py_func.__code__:
                count = max(count, frame.f_locals.get("n_term", 0))
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            ok = engine._small_shape_log_q.py_func(a, x, order, n, scratch, along)
        finally:
            sys.settrace(previous)
        if not ok or upper or scratch[engine._LOG] <= -np.log(2.0):
            return count
    steps = (
        engine._series_log_sum(a, x, order, n, scratch, along)
        if series
        else engine._continued_fraction_log_h(a, x, n, scratch, along)
    )
    return count + (steps if steps >= 0 else engine.MAX_TERMS)
