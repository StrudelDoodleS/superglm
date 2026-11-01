"""Fixture declarations, a test-only engine and the mpmath oracle for the severity generator.

The declarations here are test fixtures, not public families: Weibull (scale
and mean forms), Burr XII, log-logistic and Lomax, each with log density, log
survival and log CDF, written in log space with the generator's primitives.
``LSE`` is a two-argument engine with closed-form partials (plan interface
I2): ``log(e^u + e^v)`` on the upper tail and ``log(e^u - e^v)``, defined for
``u > v``, on the lower.

:class:`Oracle` evaluates a generated expression at one float64 point in
mpmath and, with it, the expression's rounding-error magnitude ``A``: the
primitives' error bounds composed through the expression tree exactly as the
generated code evaluates it (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., 2002, sections 3.1-3.3, the running error bound in
which every operation contributes ``u`` times its result and propagates its
operands' errors through its partials).  Every subtraction, the generated
``1 - t(1 + r)`` of the log CDFs included, is charged ``u`` times its terms,
which is issue #564's stage-2 model.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from numba import njit
from scipy import special, stats

from superglm.distributional.kernels._severity_primitives import (
    expit,
    invexpm1,
    log1mexp,
    softplus,
)

ROOT = Path(__file__).resolve().parents[1]
U = 2.0**-53


def load_generator():
    """``tools/generate_severity_families.py`` as a module (``tools`` is not a package)."""
    name = "generate_severity_families"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "tools" / "generate_severity_families.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = load_generator()
log = sp.log


def build(declaration, directory: Path):
    """Generate, write and import one declaration's kernels (Numba cache off), with its derivation."""
    derived = G.derive(declaration)
    text = G.render(derived, G.PRIMITIVES.read_text()).replace("cache=True", "cache=False")
    path = directory / f"declared_{declaration.name}.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"declared_{declaration.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, derived


# ---------------------------------------------------------------- declarations


def _scale_z(y, scale, shape):
    return shape * log(y / scale)


def _mean_z(y, mean, shape):
    return shape * (log(y / mean) + sp.loggamma(1 + 1 / shape))


def _sympy(function):
    return getattr(G, function.__name__)


log1mexp_, softplus_ = _sympy(log1mexp), _sympy(softplus)


def _weibull(name, z):
    return G.Declaration(
        name=name,
        parameters=(("mean" if "mean" in name else "scale"), "shape"),
        log_density=lambda y, s, k: log(k) - log(y) + z(y, s, k) - sp.exp(z(y, s, k)),
        log_survival=lambda y, s, k: -sp.exp(z(y, s, k)),
        log_cdf=lambda y, s, k: log1mexp_(sp.exp(z(y, s, k))),
    )


WEIBULL = _weibull("weibull", _scale_z)
WEIBULL_MEAN = _weibull("weibull_mean", _mean_z)
BURR12 = G.Declaration(
    name="burr12",
    parameters=("scale", "c", "k"),
    log_density=lambda y, s, c, k: (
        log(c) + log(k) - log(y) + _scale_z(y, s, c) - (k + 1) * softplus_(_scale_z(y, s, c))
    ),
    log_survival=lambda y, s, c, k: -k * softplus_(_scale_z(y, s, c)),
    log_cdf=lambda y, s, c, k: log1mexp_(k * softplus_(_scale_z(y, s, c))),
)
LOG_LOGISTIC = G.Declaration(
    name="log_logistic",
    parameters=("scale", "shape"),
    log_density=lambda y, s, k: (
        log(k) - log(y) + _scale_z(y, s, k) - 2 * softplus_(_scale_z(y, s, k))
    ),
    log_survival=lambda y, s, k: -softplus_(_scale_z(y, s, k)),
    log_cdf=lambda y, s, k: -softplus_(-_scale_z(y, s, k)),
)
# Lomax is Burr XII at c = 1: softplus(log(y/scale)) is log1p(y/scale) with bounded derivatives.
LOMAX = G.Declaration(
    name="lomax",
    parameters=("scale", "shape"),
    log_density=lambda y, s, k: log(k) - log(y) + log(y / s) - (k + 1) * softplus_(log(y / s)),
    log_survival=lambda y, s, k: -k * softplus_(log(y / s)),
    log_cdf=lambda y, s, k: log1mexp_(k * softplus_(log(y / s))),
)
# The textbook form of the gamma: the oracle must reject it.
NAIVE_GAMMA = G.Declaration(
    name="naive_gamma",
    parameters=("mean", "cv"),
    log_density=lambda y, m, cv: (
        cv**-2 * log(cv**-2)
        - sp.loggamma(cv**-2)
        + (cv**-2 - 1) * log(y)
        - cv**-2 * log(m)
        - cv**-2 * y / m
    ),
)

FAMILIES = (WEIBULL, WEIBULL_MEAN, BURR12, LOG_LOGISTIC, LOMAX)

SCIPY = {
    "weibull": lambda s, k: stats.weibull_min(c=k, scale=s),
    "weibull_mean": lambda m, k: stats.weibull_min(c=k, scale=m / special.gamma(1.0 + 1.0 / k)),
    "burr12": lambda s, c, k: stats.burr12(c=c, d=k, scale=s),
    "log_logistic": lambda s, k: stats.fisk(c=k, scale=s),
    "lomax": lambda s, k: stats.lomax(c=k, scale=s),
    "gamma": lambda m, cv, w=1.0: stats.gamma(a=w / cv**2, scale=m * cv**2 / w),
    "naive_gamma": lambda m, cv: stats.gamma(a=cv**-2, scale=m * cv**2),
}
CELLS = {
    "weibull": [(1000.0, k) for k in (0.25, 0.7, 1.5, 6.0)],
    "weibull_mean": [(1000.0, k) for k in (0.25, 0.7, 1.5, 6.0)],
    "burr12": [(1000.0, c, k) for c, k in ((0.5, 0.8), (2.0, 1.5), (6.0, 0.3), (1.2, 8.0))],
    "log_logistic": [(1000.0, k) for k in (0.6, 1.5, 4.0, 12.0)],
    "lomax": [(1000.0, k) for k in (0.3, 1.1, 3.0, 25.0)],
    "gamma": [(1000.0, cv) for cv in (0.02, 0.3, 1.2, 5.0)],
    "naive_gamma": [(1000.0, cv) for cv in (0.02, 0.3, 1.2, 5.0)],
}
BODY = (1e-3, 0.1, 0.5, 0.9, 0.999)
TAILS = (1e-12, 1e-30, 1e-100)


def responses(law) -> list[tuple[str, float]]:
    """Body quantiles, then left (``F``) and right (``S``) tail points down to 1e-100."""
    points = [(f"p={p:g}", float(law.ppf(p))) for p in BODY]
    points += [(f"F={p:g}", float(law.ppf(p))) for p in TAILS]
    points += [(f"S={p:g}", float(law.isf(p))) for p in TAILS]
    return [(label, y) for label, y in points if math.isfinite(y) and y > 0.0]


# ---------------------------------------------------------------- the test-only engine


@njit(cache=False)
def lse_engine(u, v, upper, order, out):
    """``log(e^u + e^v)`` (upper) or ``log(e^u - e^v)``, ``u > v`` (lower), and its partials."""
    s = u - v
    if upper:
        p, q = expit(s), expit(-s)
        out[0] = v + softplus(s)
        first = (p, q)
        # softplus^(k)(s), k = 1..4, in p = expit(s), q = expit(-s).
        g = (p, p * q, p * q * (q - p), p * q * (1.0 - 6.0 * p * q))
    else:
        if not s > 0.0:
            out[:] = np.nan
            return
        r = invexpm1(s)
        out[0] = u + log1mexp(s)
        first = (1.0 + r, -r)
        # log1mexp^(k)(s), k = 1..4, in r = 1/(e^s - 1).
        g = (
            r,
            -r * (1.0 + r),
            r * (1.0 + r) * (1.0 + 2.0 * r),
            -r * (1.0 + r) * (1.0 + 6.0 * r * (1.0 + r)),
        )
    if order >= 1:
        out[1] = first[0]
        out[2] = first[1]
    for k in range(2, order + 1):
        base = k * (k + 1) // 2
        for b in range(k + 1):
            out[base + b] = g[k - 1] if b % 2 == 0 else -g[k - 1]


LSE = G.Engine(
    name="lse_engine",
    module="tests._severity_fixtures",
    reference=lambda u, v, upper: mp.log(mp.exp(u) + mp.exp(v) if upper else mp.exp(u) - mp.exp(v)),
)


def _engine_arguments(y, a, b):
    return a * log(y), log(b) - y / a


def _engine_declaration(name, upper_tail, lower_tail):
    return G.Declaration(
        name=name,
        parameters=("a", "b"),
        log_density=lambda y, a, b: log(a) - log(y) - b * y,
        log_survival=lambda y, a, b: upper_tail(*_engine_arguments(y, a, b)),
        log_cdf=lambda y, a, b: lower_tail(*_engine_arguments(y, a, b)),
    )


ENGINE_LAW = _engine_declaration(
    "engine_law", lambda u, v: LSE(u, v, upper=True), lambda u, v: LSE(u, v, upper=False)
)
CLOSED_LAW = _engine_declaration(
    "closed_law", lambda u, v: v + softplus_(u - v), lambda u, v: u + log1mexp_(u - v)
)


# ---------------------------------------------------------------- the oracle

# The primitives' stage-1 constant (tests/test_severity_primitives.py) and
# libm's one-ulp bound, in u; lgamma is allowed two ulps.
PRIMITIVE_C = 16.0
LIBM_C = 2.0
RELATIVE = ("softplus", "expit", "log1mexp", "invexpm1", "gdev", "ratio")


def own_error(name: str, args, value, partials):
    """The error a function node adds itself, in u: its stage-1 bound for exact inputs."""
    if name in ("log", "exp", "log1p", "expm1"):
        return LIBM_C * abs(value)
    if name in RELATIVE:
        return PRIMITIVE_C * abs(value)
    backward = abs(value) + sum(
        abs(x * d) for x, d in zip(args, partials, strict=True) if d is not None
    )
    if name == "loggamma":
        return 2 * LIBM_C * backward
    if name in ("gnorm", "psires", "sres") and args[-1] < 10:
        backward += primitive_allowance(name, args)
    return PRIMITIVE_C * backward


def primitive_allowance(name: str, args):
    """The absolute allowance below ``a = 10`` (issue #564): the direct form's terms.

    ``sres`` runs the direct form only from ``a = 1/4``, below which its series
    is relatively accurate.
    """
    a = args[-1]
    if name == "gnorm":
        return abs(a * mp.log(a)) + a + abs(mp.loggamma(a))
    n = int(args[0])
    if name == "sres" and a < 0.25:
        return 0
    leading = mp.log(a) if n == 0 else (-1) ** (n - 1) * mp.factorial(n - 1) / a**n
    terms = abs(mp.polygamma(n, a)) + abs(leading)
    return terms * a ** (n + 1) if name == "sres" else terms


class Oracle:
    """mpmath value and rounding-error magnitude (in u) of expressions at one float64 point."""

    def __init__(self, point: dict):
        self.point = {symbol: mp.mpf(float(value)) for symbol, value in point.items()}
        self.values: dict = {}
        self.scales: dict = {}

    def value(self, e):
        if e in self.values:
            return self.values[e]
        if e.is_Symbol:
            result = self.point[e]
        elif e.is_Number:
            result = mp.mpf(e.p) / e.q if e.is_Rational else mp.mpf(float(e))
        elif e.is_Add:
            result = mp.fsum(self.value(a) for a in e.args)
        elif e.is_Mul:
            result = mp.fprod(self.value(a) for a in e.args)
        elif e.is_Pow:
            result = self.value(e.base) ** self.value(e.exp)
        else:
            result = self._function(e)(*(self.value(a) for a in e.args))
        self.values[e] = result
        return result

    @staticmethod
    def _function(e):
        name = type(e).__name__
        if hasattr(type(e), "engine"):
            return G.engine_mpmath(type(e))
        if name in G.MPMATH:
            return G.MPMATH[name]
        return getattr(mp, name)

    def scale(self, e):
        """``A(e)``: ``|fl(e) - e| <= u A(e)`` to first order in ``u``."""
        if e in self.scales:
            return self.scales[e]
        if e.is_Symbol or e.is_Integer:
            result = mp.mpf(0)
        elif e.is_Rational:
            result = abs(self.value(e))
        elif e.is_Add:
            terms = [abs(self.value(a)) for a in e.args]
            result = mp.fsum(self.scale(a) for a in e.args) + (len(e.args) - 1) * mp.fsum(terms)
        elif e.is_Mul:
            values = [self.value(a) for a in e.args]
            propagated = mp.fsum(
                self.scale(a) * abs(mp.fprod(values[:j] + values[j + 1 :]))
                for j, a in enumerate(e.args)
            )
            result = propagated + (len(e.args) - 1) * abs(self.value(e))
        elif e.is_Pow and e.exp.is_Integer:
            k = int(e.exp)
            base = self.value(e.base)
            propagated = abs(k * base ** (k - 1)) * self.scale(e.base)
            result = propagated + (abs(k) - 1 + (k < 0)) * abs(self.value(e))
        elif e.is_Function:
            args = [self.value(a) for a in e.args]
            skip = 1 if type(e).__name__ in ("polygamma", "psires", "sres") else 0
            partials = [None] * skip + [
                self.value(e.fdiff(i + 1)) for i in range(skip, len(e.args))
            ]
            propagated = mp.fsum(
                abs(d) * self.scale(a)
                for a, d in zip(e.args, partials, strict=True)
                if d is not None
            )
            result = propagated + own_error(type(e).__name__, args, self.value(e), partials)
        elif e.is_Pow:
            base, exponent, value = self.value(e.base), self.value(e.exp), self.value(e)
            result = (
                abs(exponent * base ** (exponent - 1)) * self.scale(e.base)
                + abs(value * mp.log(base)) * self.scale(e.exp)
                + LIBM_C * abs(value)
            )
        else:
            raise NotImplementedError(f"no rounding model for {e.func}")
        self.scales[e] = result
        return result


__all__ = [
    "BODY",
    "CELLS",
    "CLOSED_LAW",
    "ENGINE_LAW",
    "FAMILIES",
    "G",
    "LSE",
    "NAIVE_GAMMA",
    "Oracle",
    "SCIPY",
    "TAILS",
    "U",
    "build",
    "lse_engine",
    "responses",
]
