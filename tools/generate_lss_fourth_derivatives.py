"""Generate the LSS families' fourth-derivative closed forms with SymPy.

    uv run python tools/generate_lss_fourth_derivatives.py           # rewrite the module
    uv run python tools/generate_lss_fourth_derivatives.py --check   # exit 1 if it is stale

The output, ``src/superglm/distributional/kernels/_fourth_generated.py``, is
plain arithmetic on its arguments: no SymPy, so each function runs on float64
arrays or on scalars inside Numba.  Every special function a family needs
(series, Stirling remainders, Mills ratios, Tweedie cumulants) is an argument
computed by the family kernel's stable routines; this module holds only what
the chain and product rules produce, plus the generalized gamma series table
(coefficients and ``np.exp`` closed forms).  The generator also stamps a
digest of its output into ``_tweedie_numba.py`` (see ``stamped_kernel``).

Floor-stable forms.  A link ``sigma = floor + e^b`` makes the predictor
derivatives polynomials in ``u = e^b/sigma`` whose constant terms cancel as
the floor goes to zero (``1 - u`` is ``floor/sigma``).  Each such polynomial is
homogenised with ``c = floor/sigma``, writing ``u^i`` as ``u^i (u + c)^(d - i)``,
which is exact because ``u + c = 1`` and leaves no constant to cancel; the
same device writes the NB2 polynomials in ``q = mu/(mu + theta)`` and
``p = theta/(mu + theta)``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import sys
from pathlib import Path

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "superglm"
    / "distributional"
    / "kernels"
    / "_fourth_generated.py"
)


def packed(k_parameters: int, order: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations_with_replacement(range(k_parameters), order))


def partials(expression, operators, order: int) -> list:
    """Packed ``order``-th derivatives, ``operators[k]`` differentiating along axis ``k``."""
    cache = {(): expression}

    def derivative(key):
        if key not in cache:
            cache[key] = sp.expand(operators[key[-1]](derivative(key[:-1])))
        return cache[key]

    return [derivative(key) for key in packed(len(operators), order)]


def sign_split(expression, sign):
    """``E(+1)``, ``E(-1)`` merged as ``(E+ + E-)/2 + sign (E+ - E-)/2``, exact for ``sign = +-1``."""
    plus = sp.expand(expression.subs(sign, 1))
    minus = sp.expand(expression.subs(sign, -1))
    return sp.expand((plus + minus) / 2) + sign * sp.expand((plus - minus) / 2)


def homogenise(expression, u, c):
    """Rewrite the polynomial in ``u`` so it has no constant: ``u^i -> u^i (u + c)^(d - i)``.

    Exact under ``u + c = 1``; ``expression`` must be polynomial in ``u`` over
    the other symbols (a common denominator is factored out first).
    """
    numerator, denominator = sp.fraction(sp.together(expression))
    polynomial = sp.Poly(sp.expand(numerator), u)
    degree = polynomial.degree()
    rewritten = sum(
        coefficient * u**power * (u + c) ** (degree - power)
        for (power,), coefficient in polynomial.terms()
    )
    return sp.factor(sp.expand(rewritten)) / denominator


class _Printer(PythonCodePrinter):
    def _print_Rational(self, expr):
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Integer(self, expr):
        return f"{int(expr)}.0"

    def _print_Pow(self, expr, rational=False):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer and exponent > 0:
            factors = " * ".join([self.parenthesize(base, 100)] * int(exponent))
            return f"({factors})"
        if exponent.is_Integer and exponent < 0:
            factors = " * ".join([self.parenthesize(base, 100)] * int(-exponent))
            return f"(1.0 / ({factors}))"
        return super()._print_Pow(expr, rational=rational)


_PRINTER = _Printer({"fully_qualified_modules": False})


def emit(name: str, arguments: list, outputs: list, doc: str) -> str:
    """One function returning a tuple of the expressions, common subexpressions shared."""
    for expression in outputs:
        bad = [atom for atom in sp.preorder_traversal(expression) if isinstance(atom, sp.Function)]
        if bad:
            raise ValueError(f"{name}: generated code may only do arithmetic, found {bad[0]}")
    replacements, reduced = sp.cse(
        outputs, symbols=sp.numbered_symbols("t_"), optimizations="basic"
    )
    lines = [f"def {name}({', '.join(str(argument) for argument in arguments)}):"]
    lines.append('    """' + doc.strip().replace("\n", "\n    ") + '\n    """')
    for symbol, value in replacements:
        lines.append(f"    {symbol} = {_PRINTER.doprint(value)}")
    rendered = ", ".join(_PRINTER.doprint(expression) for expression in reduced)
    lines.append(f"    return ({rendered},)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- two-piece location


def two_piece_location():
    """Third and fourth derivatives of ``-log sigma - u^2/2`` within one piece."""
    t, mu, eps = sp.symbols("t mu eps", real=True)
    sigma, s = sp.symbols("sigma s", positive=True)
    u, e = sp.symbols("u e", real=True)
    density = -sp.log(sigma) - (t - mu) ** 2 / (2 * sigma**2 * (1 + e * eps) ** 2)
    operators = [lambda f: sp.diff(f, mu), lambda f: sp.diff(f, sigma), lambda f: sp.diff(f, eps)]
    outputs = []
    for order in (3, 4):
        for expression in partials(density, operators, order):
            value = expression.subs(t, mu + u * sigma * (1 + e * eps)).subs(eps, (s - 1) / e)
            outputs.append(sp.factor(sign_split(sp.simplify(value), e)))
    inverse_sigma, inverse_s = sp.symbols("inverse_sigma inverse_s", positive=True)
    outputs = [
        sp.expand(value.subs({sigma: 1 / inverse_sigma, s: 1 / inverse_s})) for value in outputs
    ]
    return emit(
        "two_piece_location_third_fourth",
        [u, inverse_sigma, inverse_s, e],
        outputs,
        """Unit-mass natural third (10) then fourth (15) derivatives in ``(mu, sigma, eps)``.

Within the piece of sign ``e`` (``s = 1 + e eps``) the row is
``-log sigma - u^2/2`` with ``u = (t - mu)/(sigma s)``; every derivative is a
monomial in ``u``, ``1/sigma`` and ``1/s``.  Packed in
``combinations_with_replacement`` order.""",
    )


# ---------------------------------------------------------------- floor-log-link rows


def _floor_log_rows(density, substitution, name: str, doc: str, extra_arguments: list):
    """Predictor fourth derivatives of a row under ``a`` and ``sigma = floor + e^b``.

    ``density(a, kappa, floor)`` is the row with ``kappa = e^b``, so the
    ``b``-derivative is ``kappa d/dkappa``; ``substitution(a, kappa, floor)``
    maps the response symbol to the residual after differentiating.  The
    result is written in ``u = kappa/sigma`` and ``c = floor/sigma``,
    homogenised.
    """
    a = sp.Symbol("a", real=True)
    kappa, floor, sigma = sp.symbols("kappa floor sigma", positive=True)
    u, c = sp.symbols("u c", positive=True)
    operators = [lambda f: sp.diff(f, a), lambda f: kappa * sp.diff(f, kappa)]
    outputs = []
    for expression in partials(density(a, kappa, floor), operators, 4):
        value = sp.expand(expression.subs(substitution(a, kappa, floor)))
        value = sp.together(value.subs(floor, sigma - kappa))
        value = sp.together(sp.expand(value.subs(kappa, u * sigma)))
        outputs.append(homogenise(value, u, c))
    return emit(name, [*extra_arguments, sigma, u, c], outputs, doc)


def gaussian_predictor():
    r, normalizer, residual, y = sp.symbols("r normalizer residual y", real=True)
    return _floor_log_rows(
        lambda a, kappa, floor: (
            -normalizer * sp.log(floor + kappa)
            - residual * (y - a) ** 2 / (2 * (floor + kappa) ** 2)
        ),
        lambda a, kappa, floor: {y: r + a},
        "gaussian_predictor_fourth",
        """Gaussian ``(location, log(scale - floor))`` fourth derivatives, packed ``aaaa..bbbb``.

The row is ``-N log sigma - R r^2/(2 sigma^2)`` with ``r = y - a``, ``N`` and
``R`` the normalizer and residual masses; ``u = (sigma - floor)/sigma`` and
``c = floor/sigma`` (``u + c = 1``) carry the floor.""",
        [r, normalizer, residual],
    )


def log_normal_mean_predictor():
    r, mass, log_y = sp.symbols("r mass log_y", real=True)
    return _floor_log_rows(
        lambda a, kappa, floor: (
            -mass * sp.log(floor + kappa)
            - mass * (log_y - a + (floor + kappa) ** 2 / 2) ** 2 / (2 * (floor + kappa) ** 2)
        ),
        lambda a, kappa, floor: {log_y: r + a - (floor + kappa) ** 2 / 2},
        "log_normal_mean_predictor_fourth",
        """Log-normal mean-form ``(log m, log(sigma - floor))`` fourth derivatives, packed.

The row is ``-m log sigma - m r^2/(2 sigma^2)`` with ``mu = a - sigma^2/2`` and
the residual ``r = log y - mu`` an argument; ``u``, ``c`` as for the Gaussian.""",
        [r, mass],
    )


# ---------------------------------------------------------------- NB2, generalized Pareto, Gamma


def negative_binomial_log():
    """``l_aaaa, l_aaab, l_aabb, l_abbb`` in ``(a, b) = (log mean, log theta)``.

    These four never touch ``lgamma(y + theta) - lgamma(theta)``, which
    depends on ``b`` alone; ``l_bbbb`` is the kernel's series or recurrence.
    """
    y, theta, q, p = sp.symbols("y theta q p", positive=True)
    mean, size, total = sp.symbols("mean size total", positive=True)
    density = y * sp.log(mean) + size * sp.log(size) - (y + size) * sp.log(mean + size)
    operators = [lambda f: mean * sp.diff(f, mean), lambda f: size * sp.diff(f, size)]
    outputs = []
    for expression in partials(density, operators, 4)[:4]:
        value = sp.together(expression.subs({mean: q * total, size: p * total}))
        value = sp.expand(sp.simplify(value)).subs(total, theta / p)
        outputs.append(homogenise(sp.together(sp.expand(value)), q, p))
    return emit(
        "negative_binomial_log_fourth_closed",
        [y, theta, q, p],
        outputs,
        """Unit-mass ``(l_aaaa, l_aaab, l_aabb, l_abbb)`` of NB2 in ``(log mean, log theta)``.

``q = mean/(mean + theta)`` and ``p = theta/(mean + theta)``; the
polynomials are homogenised in ``(q, p)``.""",
    )


def generalized_pareto():
    """``T_pppp, T_pppx, T_ppxx, T_pxxx`` in ``(psi, xi)`` through ``v`` and ``a``.

    Every derivative with a ``psi`` in it is rational (``d l/d psi = (v - a)/psi``)
    and, with ``xi v = 1 - a``, a polynomial in ``v = t/(1 + z)`` and
    ``a = 1/(1 + z)`` over a power of ``psi``; both stay bounded on the
    support, in the positive tail as well.
    """
    y, psi, xi = sp.symbols("y psi xi", positive=True)
    v, a = sp.symbols("v a", positive=True)
    density = -sp.log(psi) - (1 + 1 / xi) * sp.log(1 + xi * y / psi)
    operators = [lambda f: sp.diff(f, psi), lambda f: sp.diff(f, xi)]
    outputs = []
    for expression in partials(density, operators, 4)[:4]:
        value = sp.together(sp.simplify(expression))
        value = value.subs(y, psi * v / a).subs(xi, (1 - a) / v)
        outputs.append(sp.factor(sp.simplify(value)))
    return emit(
        "generalized_pareto_fourth_closed",
        [v, a, psi],
        outputs,
        """Unit-mass ``(T_pppp, T_pppx, T_ppxx, T_pxxx)`` of the generalized Pareto row.

``v = t/(1 + z)`` and ``a = 1/(1 + z)`` with ``t = y/psi``, ``z = xi t``; the
pure-shape ``T_xxxx`` is the kernel's series or tail form.""",
    )


def gamma_log():
    """Gamma ``(log mean, log scale)`` fourth derivatives; checked here, emitted compact.

    With shape ``a = W e^(-2 beta)`` and ``z = y e^(-alpha)`` the row is
    ``-a dev(z) + nu(log a)``, ``dev(z) = z - 1 - log z``; ``nu' = -A``,
    ``nu'' = -(A + J)``, ``nu''' = -(A + J + Jd)``, ``nu'''' = -(A + J + Jd + Jdd)``
    in the residuals ``A = a (psi(a) - log a)``, ``J = a (a psi'(a) - 1)`` and
    their log-shape derivatives ``Jd = a J'(a)``, ``Jdd = a Jd'(a)``.
    """
    alpha, beta, y, lam0 = sp.symbols("alpha beta y lambda0", real=True)
    lam = sp.Symbol("lam", real=True)
    nu_values = sp.symbols("nu0:5", real=True)
    shape = sp.exp(lam)
    z = y * sp.exp(-alpha)
    # nu(lam) as a Taylor polynomial in lam around lam = lambda0 - 2 beta is all
    # four derivatives need; differentiate in (alpha, lam) with d/d beta = -2 d/d lam.
    nu = sum(value * (lam - lam0) ** k / sp.factorial(k) for k, value in enumerate(nu_values))
    density = -shape * (z - 1 - (sp.log(y) - alpha)) + nu
    operators = [lambda f: sp.diff(f, alpha), lambda f: -2 * sp.diff(f, lam)]
    az, at, ad, big_a, j, jd, jdd = sp.symbols("az at ad A J Jd Jdd", real=True)
    nu_of = {
        nu_values[1]: -big_a,
        nu_values[2]: -(big_a + j),
        nu_values[3]: -(big_a + j + jd),
        nu_values[4]: -(big_a + j + jd + jdd),
    }
    expected = [-az, -2 * az, -4 * az, -8 * at, -16 * (ad + big_a + j + jd + jdd)]
    for expression, target in zip(partials(density, operators, 4), expected, strict=True):
        value = sp.expand(expression.subs(lam, lam0).subs(nu_of))
        reference = target.subs(
            {
                az: sp.exp(lam0) * z,
                at: sp.exp(lam0) * (z - 1),
                ad: sp.exp(lam0) * (z - 1 - (sp.log(y) - alpha)),
            }
        )
        if sp.simplify(value - sp.expand(reference)) != 0:
            raise AssertionError(f"Gamma fourth derivative mismatch: {value} vs {target}")
    return emit(
        "gamma_log_fourth",
        [az, at, ad, big_a, j, jd, jdd],
        expected,
        """Unit-mass Gamma fourth derivatives in ``(log mean, log scale)``, packed ``aaaa..bbbb``.

``az = a y/mu``, ``at = a (y/mu - 1)``, ``ad = a dev(y/mu)``; ``A``, ``J``,
``Jd``, ``Jdd`` the shape residuals (see the generator).""",
    )


# ---------------------------------------------------------------- two-piece mean loading


def two_piece_piece_derivatives():
    """Derivatives of one piece of ``log K``, ``log(1 + e eps) + G(sigma (1 + e eps))``.

    ``G`` is ``x^2/2 + log Phi(+-x)``; its derivatives ``G1..G4`` at
    ``x = sigma (1 + e eps)`` are arguments, written here as a quartic Taylor
    polynomial so the chain rule carries them.  Orders one to four, packed in
    ``(sigma, eps)``: 2 + 3 + 4 + 5 values.
    """
    sigma, eps, e, x = sp.symbols("sigma eps e x", real=True)
    s = sp.Symbol("s", positive=True)
    g = sp.symbols("G1:5", real=True)
    x0 = sp.Symbol("x0", real=True)
    taylor = sum(g[k - 1] * (x - x0) ** k / sp.factorial(k) for k in range(1, 5))
    piece = sp.log(1 + e * eps) + taylor.subs(x, sigma * (1 + e * eps))
    operators = [lambda f: sp.diff(f, sigma), lambda f: sp.diff(f, eps)]
    outputs = []
    for order in (1, 2, 3, 4):
        for expression in partials(piece, operators, order):
            value = expression.subs(x0, sigma * (1 + e * eps))
            value = sp.expand(value.subs(eps, (s - 1) / e))
            outputs.append(sp.factor(sign_split(sp.simplify(value), e)))
    return emit(
        "two_piece_piece_loading",
        [sigma, s, e, *g],
        outputs,
        """One piece's ``log K`` derivatives of orders one to four in ``(sigma, eps)``.

The piece of sign ``e`` (``s = 1 + e eps``) is ``log s + G(sigma s)`` with
``G1..G4`` the derivatives of ``G`` at ``sigma s``; 14 values, packed by order.""",
    )


def two_piece_mills():
    """``G1..G4`` from the inverse Mills ratio ``r`` of each piece.

    Left: ``G1 = a - r`` with ``r = phi(a)/Phi(-a)`` and ``r' = r (r - a)``;
    right: ``G1 = a + r`` with ``r = phi(a)/Phi(a)`` and ``r' = -r (a + r)``.
    """
    a, r = sp.symbols("a r", real=True)
    pieces = []
    for side, first, slope in (
        ("left", a - r, r * (r - a)),
        ("right", a + r, -r * (a + r)),
    ):
        values = [first]
        for _ in range(3):
            current = values[-1]
            values.append(sp.factor(sp.expand(sp.diff(current, a) + sp.diff(current, r) * slope)))
        pieces.append(
            emit(
                f"two_piece_{side}_mills",
                [a, r],
                values,
                f"""``(G1, G2, G3, G4)`` of the {side} piece from its inverse Mills ratio ``r``.""",
            )
        )
    return "\n\n".join(pieces)


# ---------------------------------------------------------------- generalized gamma


def _gg_power_split(a: int, b: int) -> tuple[int, int, int]:
    """``G_ab = w^pw Q^pq M_ab(u)``: the powers that keep ``M_ab`` a series from ``u^0``."""
    return max(0, 2 - a + b), max(0, a - b - 2), max(a, b + 2)


def generalized_gamma_location():
    """Natural third and fourth derivatives of the Prentice row in ``(mu, sigma, Q)``.

    The row is ``-log sigma + h(Q) + G(w, Q)`` with ``w = (log y - mu)/sigma``,
    ``h(Q) = -S(Q^-2)`` (``S`` the Stirling remainder) and
    ``G(w, Q) = -(e^(Q w) - 1 - Q w)/Q^2 = -sum_n Q^(n-2) w^n/n!``.  ``G`` and
    ``h`` enter as quartic Taylor polynomials, so the chain rule through ``w``
    carries their partials; each partial ``G_ab = d^a_w d^b_Q G`` is
    ``w^pw Q^pq M_ab(u)`` with ``u = Q w`` and ``M_ab`` a power series from
    ``u^0`` (``generalized_gamma_series``), and ``h3``, ``h4`` are the shape
    term's derivatives ``f'(Q)``, ``f''(Q)``.
    """
    mu, sigma, q, log_y = sp.symbols("mu sigma Q log_y", real=True)
    w0, q0, w = sp.symbols("w0 Q0 w", real=True)
    h = sp.symbols("h1:5", real=True)
    g_symbols = {
        (i, j): sp.Symbol(f"G{i}{j}", real=True)
        for i in range(5)
        for j in range(5)
        if 1 <= i + j <= 4
    }
    big_w = (log_y - mu) / sigma
    taylor_g = sum(
        symbol * (big_w - w0) ** i * (q - q0) ** j / (sp.factorial(i) * sp.factorial(j))
        for (i, j), symbol in g_symbols.items()
    )
    taylor_h = sum(h[k - 1] * (q - q0) ** k / sp.factorial(k) for k in range(1, 5))
    density = -sp.log(sigma) + taylor_h + taylor_g
    operators = [lambda f: sp.diff(f, mu), lambda f: sp.diff(f, sigma), lambda f: sp.diff(f, q)]
    m_symbols = {key: sp.Symbol(f"M{key[0]}{key[1]}", real=True) for key in g_symbols}
    replace = {}
    for (i, j), symbol in g_symbols.items():
        pw, pq, _ = _gg_power_split(i, j)
        replace[symbol] = w**pw * q**pq * m_symbols[(i, j)]
    outputs = []
    used = set()
    for order in (3, 4):
        for expression in partials(density, operators, order):
            value = expression.subs(log_y, w0 * sigma + mu).subs(q0, q)
            value = sp.expand(value.subs(w0, w).subs(replace))
            used |= {key for key, symbol in m_symbols.items() if value.has(symbol)}
            outputs.append(sp.factor(value))
    arguments = [w, q, sigma, h[2], h[3], *(m_symbols[key] for key in sorted(used))]
    text = emit(
        "generalized_gamma_location_third_fourth",
        arguments,
        outputs,
        """Unit-mass natural third (10) then fourth (15) derivatives in ``(mu, sigma, Q)``.

``w = (log y - mu)/sigma``; ``h3 = f'(Q)``, ``h4 = f''(Q)`` the shape term's
derivatives; ``Mab`` the series of ``generalized_gamma_series`` at
``u = Q w``.  Packed in ``combinations_with_replacement`` order.""",
    )
    return text, sorted(used)


def generalized_gamma_series(keys):
    """``M_ab(u)`` as a power series (coefficients) and its direct closed form.

    ``M_ab(u) = -sum_{n >= n0} (n)_a (n - 2)_b u^(n - n0)/n!`` with falling
    factorials and ``n0 = max(a, b + 2)``, so ``G_ab = w^pw Q^pq M_ab``.  The
    direct form follows from ``G = -(e^(Q w) - 1 - Q w)/Q^2`` at ``w = 1``,
    ``Q = u`` (or ``Q = 1``, ``w = u`` when ``pq > 0``).  The series runs where
    the direct form would cancel, inside a radius chosen here so that the
    direct form would lose more than the series (``_choose_radius``).
    """
    u = sp.Symbol("u", real=True)
    q, w = sp.symbols("Q w", real=True)
    g = -(sp.exp(q * w) - 1 - q * w) / q**2
    blocks = []
    table = []
    for a, b in keys:
        pw, pq, n0 = _gg_power_split(a, b)
        partial = sp.diff(g, w, a, q, b) if b else sp.diff(g, w, a)
        if pq == 0:
            direct = sp.simplify(partial.subs(w, 1).subs(q, u))
        else:
            direct = sp.simplify((partial / q**pq).subs(q, 1).subs(w, u))
        terms = 60
        coefficients = []
        for n in range(n0, n0 + terms):
            falling_a = sp.ff(n, a)
            falling_b = sp.ff(n - 2, b)
            coefficients.append(-sp.Rational(falling_a * falling_b, sp.factorial(n)))
        series = sum(c * u**k for k, c in enumerate(coefficients[:20]))
        check = sp.N((series - direct).subs(u, sp.Rational(1, 10)), 30)
        if abs(check) > 1e-25:
            raise AssertionError(f"M{a}{b}: series and direct disagree by {check}")
        printed = _NUMPY_PRINTER.doprint(direct)
        radius, kept = _choose_radius(direct, coefficients, printed)
        name = f"M{a}{b}"
        table.append(
            f"    {name!r}: ({radius!r}, ({', '.join(repr(c) for c in kept)},), "
            f"lambda u: {printed}),"
        )
    blocks.append(
        "#: ``name: (radius, series coefficients, direct form)`` for "
        "``generalized_gamma_location_third_fourth``\n"
        "GENERALIZED_GAMMA_SERIES = {\n" + "\n".join(table) + "\n}\n"
    )
    return "\n".join(blocks)


def _choose_radius(direct, coefficients, printed):
    """Series radius minimising the predicted float64 error over ``|u| <= 8``.

    Each form's error is predicted, in units of the unit roundoff, by its
    condition number at 40 digits: the absolute-term scale of the direct
    form's numerator over its value, and ``sum |c_m u^m|`` over the value for
    the series.  The radius picks the series inside and the direct form
    outside so that the larger of the two over the grid is smallest; being
    exact arithmetic, the choice is the same on every platform.
    """
    del printed
    u = sp.Symbol("u", real=True)
    numerator, denominator = sp.fraction(sp.together(direct))
    terms = sp.Add.make_args(sp.expand(numerator))
    grid = [sp.Rational(sign * step, 16) for step in range(1, 129) for sign in (1, -1)]
    direct_condition, series_scale, values = [], [], []
    for point in grid:
        value = abs(sp.N(direct.subs(u, point), 40))
        absolute = sum(abs(sp.N(term.subs(u, point), 40)) for term in terms)
        absolute /= abs(sp.N(denominator.subs(u, point), 40))
        values.append(value)
        direct_condition.append(float(absolute / value) if value else float("inf"))
        series_scale.append(
            float(sum(abs(c) * abs(point) ** m for m, c in enumerate(coefficients)) / value)
            if value
            else float("inf")
        )
    best = None
    for radius in [step / 4 for step in range(0, 17)]:
        error = max(
            series_scale[index] if abs(point) < radius else direct_condition[index]
            for index, point in enumerate(grid)
        )
        if best is None or error < best[0]:
            best = (error, radius)
    radius = best[1]
    kept = []
    for c in coefficients:
        kept.append(float(c))
        if radius == 0.0 or abs(float(c)) * radius ** len(kept) < 1e-20 * abs(
            float(coefficients[0])
        ):
            break
    return radius, tuple(kept)


class _NumpyPrinter(_Printer):
    def _print_exp(self, expr):
        return f"np.exp({self._print(expr.args[0])})"


_NUMPY_PRINTER = _NumpyPrinter({"fully_qualified_modules": False})


def generalized_gamma_loading():
    """Fourth derivatives of ``log C(sigma, Q)``: the ``Phi`` part and the direct Stirling part.

    ``log C = sigma^2 Phi(v) - log1p(v)/2 + S(x) - S(k)``, ``v = sigma Q``,
    ``x = (1 + v)/Q^2``, ``k = 1/Q^2``.  ``Phi`` enters through ``Phi1..Phi4``
    at ``v``; the Stirling difference's direct route through ``S1x..S4x`` and
    ``S1k..S4k``.  (The kernel's series route for ``k, x >= 30`` is the
    Bernoulli sum differentiated term by term.)
    """
    sigma, q, x = sp.symbols("sigma Q x", real=True)
    v0, x0, k0 = sp.symbols("v0 x0 k0", real=True)
    phi = sp.symbols("Phi1:5", real=True)
    sx = sp.symbols("S1x S2x S3x S4x", real=True)
    sk = sp.symbols("S1k S2k S3k S4k", real=True)
    v = sigma * q
    taylor_phi = sum(phi[k - 1] * (v - v0) ** k / sp.factorial(k) for k in range(1, 5))
    xx = (1 + v) / q**2
    kk = 1 / q**2
    taylor_sx = sum(sx[k - 1] * (xx - x0) ** k / sp.factorial(k) for k in range(1, 5))
    taylor_sk = sum(sk[k - 1] * (kk - k0) ** k / sp.factorial(k) for k in range(1, 5))
    operators = [lambda f: sp.diff(f, sigma), lambda f: sp.diff(f, q)]
    one = sp.Symbol("one", positive=True)
    part_phi = sigma**2 * taylor_phi - sp.log(1 + v) / 2
    phi_outputs = []
    for expression in partials(part_phi, operators, 4):
        value = sp.expand(expression.subs(v0, v)).subs(1 + v, one)
        phi_outputs.append(sp.factor(value))
    stirling_outputs = []
    for expression in partials(taylor_sx - taylor_sk, operators, 4):
        value = expression.subs({x0: xx, k0: kk})
        stirling_outputs.append(sp.expand(value))
    del x
    return "\n\n".join(
        [
            emit(
                "generalized_gamma_loading_phi_fourth",
                [sigma, q, one, *phi],
                phi_outputs,
                """Fourth derivatives of ``sigma^2 Phi(sigma Q) - log1p(sigma Q)/2`` in ``(sigma, Q)``.

``one = 1 + sigma Q``; ``Phi1..Phi4`` the derivatives of
``Phi(v) = ((1 + v) log1p(v) - v)/v^2`` at ``v = sigma Q``.  Packed ``ssss..QQQQ``.""",
            ),
            emit(
                "generalized_gamma_loading_stirling_fourth",
                [sigma, q, *sx, *sk],
                stirling_outputs,
                """Fourth derivatives of ``S(x) - S(k)`` in ``(sigma, Q)``, ``x = (1 + sigma Q)/Q^2``.

``S1x..S4x`` and ``S1k..S4k`` the Stirling remainder's derivatives at ``x``
and at ``k = 1/Q^2``; the direct route, used where ``k`` or ``x`` is below 30.""",
            ),
        ]
    )


def generalized_gamma_shape_term():
    """``h''''(Q)`` for ``h(Q) = -S(Q^-2)``, the direct route in ``S1..S4`` at ``k = Q^-2``."""
    q, k0 = sp.symbols("Q k0", real=True)
    s = sp.symbols("S1:5", real=True)
    taylor = sum(s[n - 1] * (1 / q**2 - k0) ** n / sp.factorial(n) for n in range(1, 5))
    value = sp.expand(sp.diff(-taylor, q, 4).subs(k0, 1 / q**2))
    return emit(
        "generalized_gamma_shape_term_fourth",
        [q, *s],
        [sp.factor(value)],
        """``d^4/dQ^4 [-S(Q^-2)]``, the shape term's ``f''(Q)``, from ``S1..S4`` at ``Q^-2``.""",
    )


# ---------------------------------------------------------------- Tweedie


def _total_derivative(expression, rules):
    """Differentiate an expression in symbols whose derivatives ``rules`` gives."""
    return sp.expand(sum(sp.diff(expression, symbol) * rate for symbol, rate in rules.items()))


def tweedie_closed():
    """Every derivative of the closed part in ``(mu, rho, p)``, ``rho = log phi``.

    With ``r = p - 1``, ``s = 2 - p``, ``F = e^-rho``, ``A = y mu^-r``,
    ``C = mu^s``, ``M = 1/mu`` and ``L = log mu``, the positive row's closed
    part is ``-F (A/r + C/s)`` (its series part depends on ``(rho, p)``
    alone) and the zero row is ``-F C/s``.  Derivatives follow the rules
    ``d/dmu = M d/dL``, ``dA/dL = -r A``, ``dC/dL = s C``, ``dM/dL = -M``,
    ``dA/dp = -L A``, ``dC/dp = -L C``, ``dr/dp = 1``, ``ds/dp = -1``,
    ``dF/drho = -F``; all 34 derivatives of orders one to four, packed by order.
    """
    big_f, big_a, big_c, big_m, big_l, r, s, y = sp.symbols("F A C M L r s y", real=True)
    rules = [
        {big_a: -r * big_a * big_m, big_c: s * big_c * big_m, big_m: -big_m * big_m, big_l: big_m},
        {big_f: -big_f},
        {big_a: -big_l * big_a, big_c: -big_l * big_c, r: 1, s: -1},
    ]
    operators = [lambda f, rule=rule: _total_derivative(f, rule) for rule in rules]
    pieces = []
    for name, row in (
        ("tweedie_positive_closed", -big_f * (big_a / r + big_c / s)),
        ("tweedie_zero_closed", -big_f * big_c / s),
    ):
        outputs = []
        for order in (1, 2, 3, 4):
            outputs.extend(sp.factor(value) for value in partials(row, operators, order))
        pieces.append(
            emit(
                name,
                [big_f, big_a, big_c, big_m, big_l, r, s],
                outputs,
                """Closed-part derivatives of orders one to four in ``(mu, rho, p)``, packed by order.

``F = e^-rho``, ``A = y mu^(1-p)``, ``C = mu^(2-p)``, ``M = 1/mu``,
``L = log mu``, ``r = p - 1``, ``s = 2 - p``: 3 + 6 + 10 + 15 values.""",
            )
        )
    del y
    return "\n\n".join(pieces)


def tweedie_series_terms():
    """``zeta`` and one series term's ``p``-derivatives through fourth order.

    ``zeta = alpha (log y - log r) - log s - rho/r`` with ``alpha = s/r`` and
    ``s = 1 - r`` is ``-rho - (s/r) d - log s`` in ``d = log r - log y + rho``,
    whose ``p``-derivative is ``1/r``.  Term ``j`` is
    ``j zeta - lgamma(j + 1) - lgamma(alpha j)``; with ``d alpha/dp = -1/r^2``
    its ``p``-derivatives carry the polygamma values ``P0..P3`` at ``alpha j``.
    """
    r, d, j = sp.symbols("r d j", positive=True)
    s = sp.Symbol("s", positive=True)
    zeta = sp.symbols("zeta1:5", real=True)
    polygamma = sp.symbols("P0:4", real=True)
    current = -(1 - r) / r * d - sp.log(1 - r)
    zeta_outputs = []
    for _ in range(4):
        current = sp.diff(current, r) + sp.diff(current, d) / r
        zeta_outputs.append(sp.factor(sp.simplify(current)).subs(r - 1, -s).subs(1 - r, s))
    first = j * zeta[0] + polygamma[0] * j / r**2
    rules = {r: 1, **{polygamma[k]: -polygamma[k + 1] * j / r**2 for k in range(3)}}
    rules.update({zeta[k]: zeta[k + 1] for k in range(3)})
    channels = [first]
    for _ in range(3):
        channels.append(sp.factor(_total_derivative(channels[-1], rules)))
    return "\n\n".join(
        [
            emit(
                "tweedie_zeta_derivatives",
                [r, s, d],
                zeta_outputs,
                """``d^k zeta/dp^k``, ``k = 1..4``, with ``d = log r - log y + rho``, ``s = 1 - r``.""",
            ),
            emit(
                "tweedie_term_derivatives",
                [j, r, *zeta, *polygamma],
                channels,
                """``(q_p, q_pp, q_ppp, q_pppp)`` of series term ``j``.

``zeta1..zeta4`` the ``p``-derivatives of ``zeta``; ``P0..P3`` digamma to
pentagamma at ``alpha j``.""",
            ),
        ]
    )


def tweedie_cumulants():
    """The series part ``W(rho, p) = log sum_j e^(L_j)`` through fourth order.

    Derivatives of a log-sum are joint cumulants of the term derivatives over
    the set partitions of the indices (``W_A = sum_pi kappa(L_B : B in pi)``,
    Hardy 2006, section 6).  Here ``L_rho = -j/r``, ``L_rho p = j/r^2``,
    ``L_rho pp = -2j/r^3``, ``L_rho ppp = 6j/r^4``, ``L_p = Y``, ``L_pp = Z``,
    ``L_ppp = V``, ``L_pppp = U`` and pure-``rho`` derivatives beyond the first
    vanish.  Joint cumulants are written in the central moments ``c_...`` and
    means ``m_...`` the kernel accumulates.
    """
    r = sp.Symbol("r", positive=True)
    variables = ("j", "Y", "Z", "V", "U")
    derivative_of = {
        (0,): {"j": -1 / r},
        (1,): {"Y": 1},
        (0, 0): {},
        (0, 1): {"j": 1 / r**2},
        (1, 1): {"Z": 1},
        (0, 0, 0): {},
        (0, 0, 1): {},
        (0, 1, 1): {"j": -2 / r**3},
        (1, 1, 1): {"V": 1},
        (0, 0, 0, 0): {},
        (0, 0, 0, 1): {},
        (0, 0, 1, 1): {},
        (0, 1, 1, 1): {"j": 6 / r**4},
        (1, 1, 1, 1): {"U": 1},
    }
    used = set()

    def moment(names):
        key = "".join(sorted(names, key=variables.index))
        used.add(key)
        return sp.Symbol(("m_" if len(names) == 1 else "c_") + key, real=True)

    def cumulant(names):
        if len(names) == 1:
            return moment(names)
        if len(names) in (2, 3):
            return moment(names)
        a, b, c, d = names
        return (
            moment(names)
            - moment((a, b)) * moment((c, d))
            - moment((a, c)) * moment((b, d))
            - moment((a, d)) * moment((b, c))
        )

    outputs = []
    for order in (1, 2, 3, 4):
        for target in packed(2, order):
            total = 0
            for partition in _set_partition_list(list(range(order))):
                blocks = [
                    tuple(sorted(target[position] for position in block)) for block in partition
                ]
                factors = [derivative_of[block] for block in blocks]
                if any(not factor for factor in factors):
                    continue
                for choice in itertools.product(*(factor.items() for factor in factors)):
                    coefficient = sp.Mul(*(rate for _, rate in choice))
                    total += coefficient * cumulant([name for name, _ in choice])
            outputs.append(sp.factor(sp.expand(total)))
    arguments = [r] + [
        sp.Symbol(("m_" if len(key) == 1 else "c_") + key, real=True)
        for key in sorted(used, key=lambda k: (len(k), [variables.index(c) for c in k]))
    ]
    return emit(
        "tweedie_series_cumulants",
        arguments,
        outputs,
        """``W``'s derivatives of orders one to four in ``(rho, p)``, packed by order (2+3+4+5).

``m_X`` the series-weighted mean of ``X`` and ``c_XY..`` the central moments
about those means, over ``j``, ``Y = q_p``, ``Z = q_pp``, ``V = q_ppp``,
``U = q_pppp``.""",
    )


def _set_partition_list(positions):
    if not positions:
        return [[]]
    first, rest = positions[0], positions[1:]
    result = []
    for partition in _set_partition_list(rest):
        for index in range(len(partition)):
            result.append(partition[:index] + [[first, *partition[index]]] + partition[index + 1 :])
        result.append([[first], *partition])
    return result


def render() -> str:
    pieces = [
        two_piece_location(),
        gaussian_predictor(),
        log_normal_mean_predictor(),
        negative_binomial_log(),
        generalized_pareto(),
        gamma_log(),
        two_piece_piece_derivatives(),
        two_piece_mills(),
    ]
    gg_text, gg_keys = generalized_gamma_location()
    pieces += [
        gg_text,
        generalized_gamma_loading(),
        generalized_gamma_shape_term(),
        tweedie_closed(),
        tweedie_series_terms(),
        tweedie_cumulants(),
    ]
    header = (
        '"""Generated by ``tools/generate_lss_fourth_derivatives.py``; do not edit by hand.\n\n'
        "Plain arithmetic on the arguments (float64 arrays or Numba scalars), and the\n"
        "generalized gamma series table; the generator's docstrings say how each block\n"
        "is derived.\n"
        '"""\n\n# ruff: noqa\n# fmt: off\n\nimport numpy as np\n\n'
    )
    names = [
        line.split("(", 1)[0].removeprefix("def ")
        for piece in pieces
        for line in piece.splitlines()
        if line.startswith("def ")
    ]
    names.append("GENERALIZED_GAMMA_SERIES")
    footer = "\n__all__ = [\n" + "".join(f'    "{name}",\n' for name in sorted(names)) + "]\n"
    tables = generalized_gamma_series(gg_keys)
    return header + "\n\n".join(pieces) + "\n\n" + tables + footer


TWEEDIE_KERNEL = OUTPUT.with_name("_tweedie_numba.py")
_DIGEST_LINE = re.compile(r'^_FOURTH_GENERATED_DIGEST = "[^"]*"$', re.MULTILINE)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stamped_kernel(text: str) -> str:
    """The Tweedie kernel with its digest of the generated module brought up to date.

    The kernel compiles generated functions under Numba's on-disk cache, which
    keys a caller on its own file only; the digest line changes that file
    whenever the generated module changes.
    """
    kernel = TWEEDIE_KERNEL.read_text()
    if not _DIGEST_LINE.search(kernel):
        raise RuntimeError(f"{TWEEDIE_KERNEL} has no _FOURTH_GENERATED_DIGEST line")
    return _DIGEST_LINE.sub(f'_FOURTH_GENERATED_DIGEST = "{digest(text)}"', kernel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the module is stale")
    arguments = parser.parse_args()
    text = render()
    kernel = stamped_kernel(text)
    if arguments.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != text or TWEEDIE_KERNEL.read_text() != kernel:
            print(f"{OUTPUT} is stale; rerun the generator", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    TWEEDIE_KERNEL.write_text(kernel)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
