"""Stable scalar primitives for generated severity kernels: SymPy rule, numba body, mpmath oracle.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

A family declaration composes these; the generator differentiates through their
``fdiff`` rules, so every derivative is again a product of primitives and no
generated expression subtracts two nearly equal quantities that a primitive
already evaluates without cancellation.  Each primitive is implemented once,
here, and is the only place its tails are handled.

- ``softplus(z) = log(1 + e^z)``; ``d/dz = expit(z)``.
- ``expit(z) = 1/(1 + e^-z)``; ``d/dz = expit(z) expit(-z)``.
- ``log1mexp(t) = log(1 - e^-t)`` for ``t > 0`` (Maechler 2012); ``d/dt = invexpm1(t)``.
- ``invexpm1(t) = 1/(e^t - 1)``; ``d/dt = -invexpm1(t) (1 + invexpm1(t))``.
- ``dev(t) = t - log(1 + t)`` for ``t > -1``; ``d/dt = t/(1 + t)``.
- ``gdev(y, m) = y/m - 1 - log(y/m)``, the gamma unit deviance over two; its
  partials are ``(y - m)/(y m)`` and ``(m - y)/m^2``, so no derivative forms
  ``1 + t`` from a rounded ``t``.
- ``gnorm(a) = a log a - a - log Gamma(a)``; ``d/da = -psires(0, a)``.
- ``psires(n, a)``: ``psi^(n)(a)`` less the ``n``-th derivative of ``log a``;
  ``d/da = psires(n + 1, a)``.
- SymPy's own ``loggamma`` and ``polygamma`` map to ``math.lgamma``, scipy's
  ``psi`` (through its Cython entry point) and :func:`polygamma` below.
"""

from __future__ import annotations

import ctypes
import math

import mpmath as mp
import sympy as sp
from numba import njit
from numba.extending import get_cython_function_address
from sympy.core.function import ArgumentIndexError

# ---------------------------------------------------------------- SymPy rules


class softplus(sp.Function):
    def fdiff(self, argindex=1):
        return expit(self.args[0])


class expit(sp.Function):
    def fdiff(self, argindex=1):
        z = self.args[0]
        return expit(z) * expit(-z)


class log1mexp(sp.Function):
    def fdiff(self, argindex=1):
        return invexpm1(self.args[0])


class invexpm1(sp.Function):
    def fdiff(self, argindex=1):
        r = invexpm1(self.args[0])
        return -r * (1 + r)


class dev(sp.Function):
    def fdiff(self, argindex=1):
        t = self.args[0]
        return t / (1 + t)


class gdev(sp.Function):
    nargs = 2

    def fdiff(self, argindex=1):
        y, m = self.args
        if argindex == 1:
            return (y - m) / (y * m)
        return (m - y) / m**2


class gnorm(sp.Function):
    def fdiff(self, argindex=1):
        return -psires(0, self.args[0])


class psires(sp.Function):
    nargs = 2

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        n, a = self.args
        return psires(n + 1, a)


PRIMITIVES = (softplus, expit, log1mexp, invexpm1, dev, gdev, gnorm, psires)

# ---------------------------------------------------------------- numba bodies

_psi = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1psi")
)
_LN2 = math.log(2.0)
# Bernoulli numbers B_2k for k = 1..10.
_B = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
)
_ASYMPTOTIC = 10.0


@njit(cache=True)
def nb_softplus(z):
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


@njit(cache=True)
def nb_expit(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@njit(cache=True)
def nb_log1mexp(t):
    if t <= _LN2:
        return math.log(-math.expm1(-t))
    return math.log1p(-math.exp(-t))


@njit(cache=True)
def nb_invexpm1(t):
    return 1.0 / math.expm1(t)


@njit(cache=True)
def nb_dev(t):
    if abs(t) < 0.25:
        total = 0.0
        power = t * t
        for n in range(2, 40):
            term = power / n
            total += term if n % 2 == 0 else -term
            power *= t
        return total
    return t - math.log1p(t)


@njit(cache=True)
def nb_gdev(y, m):
    t = (y - m) / m
    if abs(t) < 0.25:
        return nb_dev(t)
    r = y / m
    return (r - 1.0) - math.log(r)


@njit(cache=True)
def _factorial(n):
    out = 1.0
    for k in range(2, n + 1):
        out *= k
    return out


@njit(cache=True)
def _polygamma_asymptotic_tail(n, a):
    """``psi^(n)(a)`` less its leading ``(-1)^(n+1) (n-1)!/a^n`` term, for ``n >= 1``, ``a >= 10``."""
    sign = 1.0 if n % 2 == 1 else -1.0
    total = _factorial(n) / (2.0 * a ** (n + 1))
    for k in range(1, 11):
        total += _B[k - 1] * _factorial(2 * k + n - 1) / (_factorial(2 * k) * a ** (2 * k + n))
    return sign * total


@njit(cache=True)
def nb_polygamma(n, a):
    """``psi^(n)(a)`` for ``n >= 1`` and ``a > 0``: upward recurrence, then the asymptotic series."""
    sign = 1.0 if n % 2 == 1 else -1.0
    shift = 0.0
    x = a
    while x < _ASYMPTOTIC:
        shift += 1.0 / x ** (n + 1)
        x += 1.0
    leading = sign * _factorial(n - 1) / x**n
    return leading + _polygamma_asymptotic_tail(n, x) + sign * _factorial(n) * shift


@njit(cache=True)
def nb_digamma(a):
    return _psi(a)


@njit(cache=True)
def nb_psires(n, a):
    if n == 0:
        if a >= _ASYMPTOTIC:
            inverse_square = 1.0 / (a * a)
            total = 0.0
            power = inverse_square
            for k in range(1, 11):
                total += _B[k - 1] / (2.0 * k) * power
                power *= inverse_square
            return -0.5 / a - total
        return _psi(a) - math.log(a)
    sign = 1.0 if n % 2 == 1 else -1.0
    if a >= _ASYMPTOTIC:
        return _polygamma_asymptotic_tail(n, a)
    return nb_polygamma(n, a) - sign * _factorial(n - 1) / a**n


@njit(cache=True)
def nb_gnorm(a):
    if a >= _ASYMPTOTIC:
        inverse = 1.0 / a
        inverse_square = inverse * inverse
        total = 0.0
        power = inverse
        for k in range(1, 11):
            total += _B[k - 1] / (2.0 * k * (2.0 * k - 1.0)) * power
            power *= inverse_square
        return 0.5 * math.log(a) - 0.5 * math.log(2.0 * math.pi) - total
    return a * math.log(a) - a - math.lgamma(a)


# ---------------------------------------------------------------- mpmath oracles


def _mp_psires(n, a):
    n = int(n)
    if n == 0:
        return mp.digamma(a) - mp.log(a)
    return mp.polygamma(n, a) - (-1) ** (n - 1) * mp.factorial(n - 1) / a**n


MPMATH = {
    "softplus": lambda z: mp.log1p(mp.exp(z)),
    "expit": lambda z: 1 / (1 + mp.exp(-z)),
    "log1mexp": lambda t: mp.log(-mp.expm1(-t)),
    "invexpm1": lambda t: 1 / mp.expm1(t),
    "dev": lambda t: t - mp.log1p(t),
    "gdev": lambda y, m: y / m - 1 - mp.log(y / m),
    "gnorm": lambda a: a * mp.log(a) - a - mp.loggamma(a),
    "psires": _mp_psires,
    "loggamma": mp.loggamma,
    "polygamma": lambda n, a: mp.polygamma(int(n), a),
    "log1p": mp.log1p,
    "expm1": mp.expm1,
}
