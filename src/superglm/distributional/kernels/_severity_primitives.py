"""Stable scalar primitives the declared severity kernels call (Numba).

The generated kernels in ``kernels._declared`` are products of these and of
``math`` functions; their SymPy derivative rules and mpmath references live in
``tools/generate_severity_families.py``, a dev-only tool.  Each primitive takes
the singular or cancelling part out analytically and evaluates the regular
remainder directly, so no generated derivative subtracts two nearly equal
quantities that a primitive resolves:

- ``softplus(z) = log(1 + e^z)`` and ``expit(z) = 1/(1 + e^-z)``, split at
  ``z = 0`` so no exponential overflows;
- ``log1mexp(t) = log(1 - e^-t)`` for ``t > 0``, by ``log(-expm1(-t))`` up to
  ``log 2`` and ``log1p(-e^-t)`` above (Maechler, *Accurately computing
  log(1 - exp(-|a|))*, Rmpfr vignette, 2012);
- ``invexpm1(t) = 1/(e^t - 1)``, written ``e^-t/(1 - e^-t)`` above ``t = 1`` so
  it underflows gradually instead of through an overflowing ``expm1``;
- ``gdev(y, m) = y/m - 1 - log(y/m)``, half the gamma unit deviance, which is
  Loader's ``bd0(m, y)/m``, by his series in ``v = (m - y)/(m + y)`` (Loader,
  *Fast and accurate computation of binomial probabilities*, 2000) for
  ``y/m`` in ``(1/2, 2)``, where ``y - m`` is exact, and directly outside,
  where ``|y/m - 1| <= 3.3 gdev``: relative-accurate for exact inputs;
- ``gnorm(a) = a log a - a - log Gamma(a)``, by Stirling's series for
  ``a >= 10`` (DLMF 5.11.1);
- ``psires_0123(a)``: ``psi^(n)(a)`` less the ``n``-th derivative of
  ``log a``, ``n = 0..3``, by the asymptotic series for ``a >= 10`` (DLMF
  5.11.2 and 5.15.8, differentiated) and from ``polygamma_0123`` below;
- ``sres_0123(a)``: the scaled residuals ``a^(n+1) psires(n, a)``, ``n = 0..3``,
  bounded at both ends where ``psires`` grows like ``a^-(n+1)``: the origin's
  zeta series below ``a = 1/4``, the asymptotic series from ``a = 10``, and
  ``psires_0123`` between (``GammaLS``'s ``A`` and ``J`` are ``n = 0, 1``);
- ``polygamma_0123(x)``: digamma to pentagamma by the shift recurrence to
  ``x >= 16`` and the asymptotic series through ``B_14``; the Tweedie series
  kernel and the incomplete-gamma engine call this one copy.

Error form, for exact float64 inputs ``x_i`` (issue #564, stage 1):
``|err| <= c u (sum_i |x_i df/dx_i| + |f|)``, the input and output rounding of
a backward-stable evaluation, finite at the roots of ``gdev`` (``y = m``) and
``gnorm`` (``a`` near 6.45).  Below ``a = 10`` the direct ``gnorm`` and
``psires`` forms carry an absolute allowance of the same shape over their
terms.  ``tests/test_severity_primitives.py`` states ``c`` from the error
analysis and holds every primitive to it against mpmath.

Numba's on-disk cache keys a compiled caller on its own file only, so the
generator stamps a digest of this file into every generated module, and a
digest of the primitives each other kernel module imports into that module's
``_CALLEE_DIGEST``.
"""

from __future__ import annotations

import math

from numba import njit
from scipy import special

_LOG_TWO = math.log(2.0)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_MIN_NORMAL = 2.2250738585072014e-308
# |v| < 1/3 is y/m in (1/2, 2); there v^2 < 1/9 and 17 terms leave under u of the sum.
_GDEV_SERIES = 1.0 / 3.0
_ODD_RECIPROCALS = tuple(1.0 / (2 * j + 1) for j in range(1, 18))
# The asymptotic branches of gnorm and psires_0123 start here.
_ASYMPTOTIC = 10.0
# Bernoulli numbers B_2k, k = 1..15, as exact integer ratios: at a = 10 the
# fifteenth term of every series below is under 1e-17 of its sum.
_BERNOULLI = (
    (1, 6),
    (-1, 30),
    (1, 42),
    (-1, 30),
    (5, 66),
    (-691, 2730),
    (7, 6),
    (-3617, 510),
    (43867, 798),
    (-174611, 330),
    (854513, 138),
    (-236364091, 2730),
    (8553103, 6),
    (-23749461029, 870),
    (8615841276005, 14322),
)
_K = range(1, len(_BERNOULLI) + 1)
# Coefficients of x^k, x = 1/a^2, each one correctly rounded integer quotient.
_GNORM = tuple(p / (q * 2 * k * (2 * k - 1)) for k, (p, q) in zip(_K, _BERNOULLI, strict=True))
_PSI0 = tuple(p / (q * 2 * k) for k, (p, q) in zip(_K, _BERNOULLI, strict=True))
_PSI1 = tuple(p / q for p, q in _BERNOULLI)
_PSI2 = tuple((2 * k + 1) * p / q for k, (p, q) in zip(_K, _BERNOULLI, strict=True))
_PSI3 = tuple((2 * k + 1) * (2 * k + 2) * p / q for k, (p, q) in zip(_K, _BERNOULLI, strict=True))
# a^(n+1) psires(n, a) = -(-1)^n n!/2 - (-1)^n sum_k B_2k (2k)(2k+1)..(2k+n-1)/(2k) a^(1-2k),
# as coefficients of x^k (x = 1/a^2) times a.
_SRES = tuple(
    tuple(
        -((-1) ** n) * p * math.prod(range(2 * k, 2 * k + n)) / (q * 2 * k)
        for k, (p, q) in zip(_K, _BERNOULLI, strict=True)
    )
    for n in range(4)
)
# Below a = 1/4: psi(a) = -1/a - gamma + sum_{k>=2} (-1)^k zeta(k) a^(k-1); 60 terms leave
# under 1e-19 of the sum for every n <= 3.
_ORIGIN = 0.25
_ZETA = tuple(float(z) for z in special.zeta(range(2, 62), 1.0))
_EULER = 0.5772156649015329
# Above x = 16 the polygamma series through B_14 leave a relative remainder below 1e-16.
_POLYGAMMA_FROM = 16.0
_POLYGAMMA_BERNOULLI = tuple(p / q for p, q in _BERNOULLI[:7])


@njit(cache=True)
def softplus(z):
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


@njit(cache=True)
def expit(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@njit(cache=True)
def log1mexp(t):
    if t <= _LOG_TWO:
        return math.log(-math.expm1(-t))
    return math.log1p(-math.exp(-t))


@njit(cache=True)
def invexpm1(t):
    if t > 1.0:
        e = math.exp(-t)
        return e / (1.0 - e)
    return 1.0 / math.expm1(t)


@njit(cache=True)
def gdev(y, m):
    d = m - y
    v = d / (m + y)
    if abs(v) < _GDEV_SERIES:
        # bd0(m, y)/m = (m - y) v/m + 2 sum_j v^(2j+1)/(2j+1).
        total = d * v / m
        term = 2.0 * v
        v2 = v * v
        for coefficient in _ODD_RECIPROCALS:
            term *= v2
            step = term * coefficient
            if total + step == total:
                break
            total += step
        return total
    r = y / m
    log_r = math.log(r) if r >= _MIN_NORMAL else math.log(y) - math.log(m)
    return (r - 1.0) - log_r


@njit(cache=True)
def _series(coefficients, x):
    """``sum_k c_k x^k``, ``k = 1..len(c)``, by Horner's rule."""
    total = 0.0
    for index in range(len(coefficients) - 1, -1, -1):
        total = (total + coefficients[index]) * x
    return total


@njit(cache=True)
def gnorm(a):
    if a >= _ASYMPTOTIC:
        inverse = 1.0 / a
        return 0.5 * math.log(a) - _HALF_LOG_TWO_PI - a * _series(_GNORM, inverse * inverse)
    return a * math.log(a) - a - math.lgamma(a)


@njit(cache=True)
def polygamma_0123(value):
    """Digamma, trigamma, tetragamma and pentagamma of a finite positive scalar.

    The recurrence lifts the argument to at least 16, where the asymptotic
    expansions through ``B_14`` (DLMF 5.11.2 and 5.15.8, differentiated) leave
    a relative remainder below ``1e-16`` in every channel.
    """
    shifted_digamma = 0.0
    shifted_trigamma = 0.0
    shifted_tetragamma = 0.0
    shifted_pentagamma = 0.0
    x = value
    while x < _POLYGAMMA_FROM:
        inverse = 1.0 / x
        inverse2 = inverse * inverse
        shifted_digamma -= inverse
        shifted_trigamma += inverse2
        shifted_tetragamma -= 2.0 * inverse2 * inverse
        shifted_pentagamma += 6.0 * inverse2 * inverse2
        x += 1.0
    inverse = 1.0 / x
    i2 = inverse * inverse
    digamma = 0.0
    trigamma = 0.0
    tetragamma = 0.0
    pentagamma = 0.0
    power = i2
    for k in range(len(_POLYGAMMA_BERNOULLI)):
        b = _POLYGAMMA_BERNOULLI[k]
        order = 2.0 * (k + 1)
        digamma -= b / order * power
        trigamma += b * power * inverse
        tetragamma -= (order + 1.0) * b * power * i2
        pentagamma += (order + 1.0) * (order + 2.0) * b * power * i2 * inverse
        power *= i2
    digamma += shifted_digamma + math.log(x) - 0.5 * inverse
    trigamma += shifted_trigamma + inverse + 0.5 * i2
    tetragamma += shifted_tetragamma - i2 - i2 * inverse
    pentagamma += shifted_pentagamma + 2.0 * i2 * inverse + 3.0 * i2 * i2
    return digamma, trigamma, tetragamma, pentagamma


@njit(cache=True)
def psires_0123(a):
    """``psi^(n)(a) - d^n log(a)/da^n`` for ``n = 0, 1, 2, 3``."""
    inverse = 1.0 / a
    x = inverse * inverse
    if a >= _ASYMPTOTIC:
        return (
            -0.5 * inverse - _series(_PSI0, x),
            0.5 * x + inverse * _series(_PSI1, x),
            -x * inverse - x * _series(_PSI2, x),
            3.0 * x * x + x * inverse * _series(_PSI3, x),
        )
    p0, p1, p2, p3 = polygamma_0123(a)
    return p0 - math.log(a), p1 - inverse, p2 + x, p3 - 2.0 * x * inverse


@njit(cache=True)
def sres_0123(a):
    """``a^(n+1) (psi^(n)(a) - d^n log(a)/da^n)`` for ``n = 0, 1, 2, 3``."""
    if a < _ORIGIN:
        s0 = -1.0 - a * (_EULER + math.log(a))
        s1 = 1.0 - a
        s2 = -2.0 + a
        s3 = 6.0 - 2.0 * a
        power = a
        for index in range(len(_ZETA)):
            k = index + 2.0
            power *= a
            term = _ZETA[index] * power if index % 2 == 0 else -_ZETA[index] * power
            s0 += term
            s1 += (k - 1.0) * term
            s2 += (k - 1.0) * (k - 2.0) * term
            s3 += (k - 1.0) * (k - 2.0) * (k - 3.0) * term
        return s0, s1, s2, s3
    if a >= _ASYMPTOTIC:
        x = 1.0 / (a * a)
        return (
            -0.5 + a * _series(_SRES[0], x),
            0.5 + a * _series(_SRES[1], x),
            -1.0 + a * _series(_SRES[2], x),
            3.0 + a * _series(_SRES[3], x),
        )
    r0, r1, r2, r3 = psires_0123(a)
    a2 = a * a
    return a * r0, a2 * r1, a2 * a * r2, a2 * a2 * r3


__all__ = [
    "expit",
    "gdev",
    "gnorm",
    "invexpm1",
    "log1mexp",
    "polygamma_0123",
    "psires_0123",
    "softplus",
    "sres_0123",
]
