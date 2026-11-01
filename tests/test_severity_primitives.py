"""The declared severity kernels' primitives against mpmath (issue #564, stage 1).

Each primitive ``f`` of exact float64 inputs ``x_i`` is held to

    |err| <= C u E_f(x) + 2**-1074,

with ``u = 2**-53``, the subnormal spacing covering gradual underflow, and

- ``E_f = |f|`` (relative accuracy) for ``softplus``, ``expit``, ``log1mexp``,
  ``invexpm1`` and ``gdev``.  This implies issue #564's
  ``sum_i |x_i df/dx_i| + |f|``, which stays finite at ``gdev``'s root
  ``y = m``, where ``gdev`` is exactly 0;
- ``E_f = sum_i |x_i df/dx_i| + |f|`` for ``gnorm``, ``psires_0123``,
  ``sres_0123`` and ``polygamma_0123``.  Where the direct forms run (below
  ``a = 10``, and for ``sres`` from ``a = 1/4``) ``E`` adds the allowance
  ``a |log a| + a + |lgamma a|`` (``gnorm``) or
  ``|psi^(n) a| + |d^n log a / da^n|`` (``psires``; times ``a^(n+1)`` for
  ``sres``).

``C = 16`` is from the error analysis, with libm's ``exp``, ``log``,
``log1p``, ``expm1`` at one ulp (``2u``) and ``lgamma`` at two:

- ``softplus``, ``expit``, ``log1mexp``, ``invexpm1``: two or three libm
  calls and roundings, at most ``6u |f|``.
- ``gdev``: Loader's series on ``y/m`` in ``(1/2, 2)`` costs about ``6u |f|``.
  Outside it, ``(r - 1) - log r`` with ``r = fl(y/m)`` costs at most
  ``u(4|t| + 3|f|)`` with ``|t| = |y/m - 1| <= 3.3 |f|``, about ``13u |f|``.
- ``gnorm`` from ``a = 10``: ``0.5 log a - log(2 pi)/2 - S(a)`` costs about
  ``u(4|f| + 2.8)``, ``5.1u`` of ``E`` at ``a = 10``.  Below 10:
  ``a log a - a - lgamma a`` costs at most ``4u`` of its allowance.
- ``psires_0123`` and ``polygamma_0123``: the shift recurrence's 16 terms
  bound the polygammas within ``16u`` of their sums.  ``sres_0123`` adds one
  power of ``a`` to ``psires`` in the direct range.  Elsewhere it sums the
  zeta or Bernoulli series onto a leading constant that dominates the sum,
  a few ``u`` relative.

The measured worst values of ``err / (u E)`` on this grid (x86-64 Linux,
glibc, AMD EPYC 9645) are pinned in ``_MEASURED``.  The test asserts ``C``,
never the measurement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from superglm.distributional.kernels import _severity_primitives as prims  # noqa: E402

U = 2.0**-53
C = 16.0
TINY = 2.0**-1074
mp.mp.dps = 50

_Z = sorted(
    {
        *np.linspace(-40.0, 40.0, 161),
        *(s * 10.0**e for s in (-1, 1) for e in np.linspace(-300, 2.85, 40)),
        -745.0,
        709.0,
    }
)
_T = sorted(
    {
        *np.geomspace(1e-300, 700.0, 120),
        math.log(2.0),
        np.nextafter(math.log(2.0), 1.0),
        1.0,
        np.nextafter(1.0, 0.0),
    }
)
_RATIOS = sorted(
    {
        1.0,
        *(1.0 + s * 10.0**e for s in (-1, 1) for e in np.linspace(-16, -0.4, 30)),
        0.5,
        np.nextafter(0.5, 0.0),
        2.0,
        np.nextafter(2.0, 3.0),
        *np.geomspace(1e-300, 1e300, 61),
    }
)
_A = sorted(
    {
        *np.geomspace(1e-3, 1e6, 120),
        *np.linspace(6.3, 6.6, 31),
        9.99,
        np.nextafter(10.0, 0.0),
        10.0,
        10.01,
        1.4616321449683622,
        2500.0,
    }
)


def _mp_psires(n, a):
    leading = mp.log(a) if n == 0 else (-1) ** (n - 1) * mp.factorial(n - 1) / a**n
    return mp.polygamma(n, a) - leading


def _ratio(got, exact, scale) -> float:
    error = abs(mp.mpf(float(got)) - exact)
    return float(error / (U * scale)) if error > TINY else 0.0


def _relative_cases():
    """``(name, compiled, inputs, mpmath reference)`` for the relatively accurate primitives."""
    m = 1e5
    yield "softplus", prims.softplus, [(z,) for z in _Z], lambda z: mp.log1p(mp.exp(z))
    yield "expit", prims.expit, [(z,) for z in _Z], lambda z: 1 / (1 + mp.exp(-z))
    yield (
        "log1mexp",
        prims.log1mexp,
        [(t,) for t in _T],
        lambda t: mp.log(-mp.expm1(-t)) if t < 1 else mp.log1p(-mp.exp(-t)),
    )
    yield "invexpm1", prims.invexpm1, [(t,) for t in _T], lambda t: 1 / mp.expm1(t)
    gdev_inputs = [(float(r * m), m) for r in _RATIOS if 0.0 < r * m < 1e308]
    gdev_inputs += [(float(r), 1.0) for r in _RATIOS]
    # Near the top of the range, where ``m + y`` overflows.
    gdev_inputs += [(float(r * 1e308), 1e308) for r in _RATIOS if r < 1.7]
    yield "gdev", prims.gdev, gdev_inputs, lambda y, m: y / m - 1 - mp.log(y / m)
    yield "logratio", prims.logratio, gdev_inputs, lambda y, m: mp.log(y / m)


def _relative_worst(name, function, inputs, reference) -> float:
    worst = 0.0
    for x in inputs:
        exact = reference(*(mp.mpf(v) for v in x))
        got = function(*x)
        assert abs(mp.mpf(float(got)) - exact) <= C * U * abs(exact) + TINY, (name, x, got)
        worst = max(worst, _ratio(got, exact, abs(exact)) if abs(exact) > TINY else 0.0)
    return worst


def _gamma_scale(name, n, a):
    """``E`` of ``gnorm`` (``n = None``) or ``psires(n, .)`` at ``a``, and the exact value."""
    a = mp.mpf(a)
    if name == "gnorm":
        value = a * mp.log(a) - a - mp.loggamma(a)
        scale = abs(a * _mp_psires(0, a)) + abs(value)
        if a < 10:
            scale += abs(a * mp.log(a)) + a + abs(mp.loggamma(a))
        return value, scale
    if name in ("psires", "sres"):
        power = a ** (n + 1) if name == "sres" else 1
        value = power * _mp_psires(n, a)
        # a d/da of a^(n+1) psires(n, a) is (n + 1) sres(n) + sres(n + 1).
        derivative = (
            (n + 1) * value + a ** (n + 2) * _mp_psires(n + 1, a)
            if name == "sres"
            else a * _mp_psires(n + 1, a)
        )
        scale = abs(derivative) + abs(value)
        if a < 10 and not (name == "sres" and a < 0.25):
            scale += power * (abs(mp.polygamma(n, a)) + abs(_mp_psires(n, a) - mp.polygamma(n, a)))
        return value, scale
    value = mp.polygamma(n, a)
    return value, abs(a * mp.polygamma(n + 1, a)) + abs(value)


def _gamma_worst(gnorm, psires_0123, sres_0123, *, below=None) -> dict[str, float]:
    """Worst ``err / (u E)`` of every gamma-function channel, over ``a`` in ``below`` or the grid."""
    worst: dict[str, float] = {}
    for a in _A if below is None else below:
        channels = [("gnorm", None, gnorm(a))]
        channels += [("psires", n, v) for n, v in enumerate(psires_0123(a))]
        channels += [("sres", n, v) for n, v in enumerate(sres_0123(a))]
        channels += [("polygamma", n, v) for n, v in enumerate(prims.polygamma_0123(a))]
        for name, n, got in channels:
            exact, scale = _gamma_scale(name, n, a)
            key = name if n is None else f"{name}{n}"
            worst[key] = max(worst.get(key, 0.0), _ratio(got, exact, scale))
    return worst


# Measured worst err / (u E) on this grid, recorded beside C (which the tests assert).
_MEASURED = {
    "softplus": 1.34,
    "expit": 2.68,
    "log1mexp": 0.88,
    "invexpm1": 0.88,
    "gdev": 2.83,
    "gnorm": 2.38,
    "psires0": 4.48,
    "psires1": 1.19,
    "psires2": 1.76,
    "psires3": 1.23,
    "sres0": 6.30,
    "sres1": 3.41,
    "sres2": 3.26,
    "sres3": 3.32,
    "polygamma0": 3.62,
    "polygamma1": 1.57,
    "polygamma2": 2.11,
    "polygamma3": 1.54,
}


@pytest.mark.parametrize("case", list(_relative_cases()), ids=lambda case: case[0])
def test_elementary_primitives_are_relatively_accurate(case) -> None:
    assert _relative_worst(*case) <= C


def test_gamma_function_primitives_hold_the_backward_error_form() -> None:
    worst = _gamma_worst(prims.gnorm, prims.psires_0123, prims.sres_0123)
    assert set(worst) == set(_MEASURED) - {"softplus", "expit", "log1mexp", "invexpm1", "gdev"}
    assert max(worst.values()) <= C, worst


def test_gdev_is_exactly_zero_at_its_root() -> None:
    for m in (1e-300, 0.37, 1.0, 1e5, 1e300):
        assert prims.gdev(m, m) == 0.0


def test_the_direct_forms_above_ten_fail_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation check: the direct forms in place of the ``a >= 10`` branches fail at ``a = 2,500``."""
    monkeypatch.setattr(prims, "_ASYMPTOTIC", math.inf)
    # sres_0123 reaches the direct form through psires_0123, whose compiled switch is frozen.
    monkeypatch.setattr(prims, "psires_0123", prims.psires_0123.py_func)
    worst = _gamma_worst(
        prims.gnorm.py_func, prims.psires_0123, prims.sres_0123.py_func, below=[2500.0]
    )
    # Measured at a = 2,500, err / (u E): gnorm 4.4e3; psires0-3 1.9e4, 8.4e2, 2.9e2, 2.0e2;
    # sres0-3 3.8e4, 2.5e3, 1.2e3, 9.9e2.
    for key in ("gnorm", *(f"{name}{n}" for name in ("psires", "sres") for n in range(4))):
        assert worst[key] > C, (key, worst)
