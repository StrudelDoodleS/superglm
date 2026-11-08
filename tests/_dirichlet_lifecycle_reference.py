"""Small SciPy references in natural-alpha or raw natural-spline coordinates.

No SuperLSS family, design, penalty or optimizer is used here. CubicSpline
interpolates knot values; two-point Gauss quadrature integrates the product
of its piecewise-linear second derivatives exactly (up to float64 rounding).
"""

import numpy as np
from scipy import interpolate, linalg, optimize, special


def cardinal(x, knots):
    spline = interpolate.CubicSpline(knots, np.eye(len(knots)), bc_type="natural")
    points, weights = np.polynomial.legendre.leggauss(2)
    penalty = np.zeros((len(knots), len(knots)))
    for lo, hi in zip(knots[:-1], knots[1:]):
        half = (hi - lo) / 2
        second = spline((hi + lo) / 2 + half * points, 2)
        penalty += second.T @ (second * (weights * half)[:, None])
    # The public cr penalty uses knot-interval coordinates u=x/h, so
    # integral (d²f/du²)² du = h³ integral (d²f/dx²)² dx.
    spacing = (knots[-1] - knots[0]) / (len(knots) - 1)
    return spline(x), penalty * spacing**3


def objective(y, designs, penalties, offsets, weights):
    ends = np.cumsum([0, *(d.shape[1] for d in designs)])
    slices = [slice(a, b) for a, b in zip(ends[:-1], ends[1:])]
    penalty = linalg.block_diag(*penalties)
    logy = np.log(y / y.sum(axis=1, keepdims=True))

    def evaluate(beta):
        eta = np.column_stack([d @ beta[s] for d, s in zip(designs, slices)]) + offsets
        alpha = np.exp(eta)
        total = alpha.sum(axis=1)
        ll = special.gammaln(total) - special.gammaln(alpha).sum(axis=1)
        ll += ((alpha - 1) * logy).sum(axis=1)
        score = alpha * (special.digamma(total)[:, None] - special.digamma(alpha) + logy)
        gradient = np.concatenate([-d.T @ (weights * score[:, j]) for j, d in enumerate(designs)])
        hessian = np.empty(penalty.shape)
        for j, (dj, sj) in enumerate(zip(designs, slices)):
            for k, (dk, sk) in enumerate(zip(designs, slices)):
                h = alpha[:, j] * alpha[:, k] * special.polygamma(1, total)
                if j == k:
                    h += score[:, j] - alpha[:, j] ** 2 * special.polygamma(1, alpha[:, j])
                hessian[sj, sk] = -dj.T @ ((weights * h)[:, None] * dk)
        return (
            -weights @ ll + 0.5 * beta @ penalty @ beta,
            gradient + penalty @ beta,
            hessian + penalty,
        )

    return evaluate


def stationary(evaluate, width):
    result = optimize.minimize(
        lambda b: evaluate(b)[0],
        np.zeros(width),
        jac=lambda b: evaluate(b)[1],
        hess=lambda b: evaluate(b)[2],
        method="trust-exact",
        options={"gtol": 1e-9},
    )
    beta = result.x.copy()
    for _ in range(3):
        _, gradient, hessian = evaluate(beta)
        beta -= np.linalg.solve(hessian, gradient)
    assert np.linalg.norm(evaluate(beta)[1], ord=np.inf) < 2e-8
    return beta


def natural_constant(y, weights):
    logy = np.log(y / y.sum(axis=1, keepdims=True))
    mass = weights.sum()
    logs = weights @ logy

    def evaluate(alpha):
        value = (
            -mass * (special.gammaln(alpha.sum()) - special.gammaln(alpha).sum())
            - (alpha - 1) @ logs
        )
        gradient = -mass * (special.digamma(alpha.sum()) - special.digamma(alpha)) - logs
        hessian = mass * (np.diag(special.polygamma(1, alpha)) - special.polygamma(1, alpha.sum()))
        return value, gradient, hessian

    result = optimize.minimize(
        lambda a: evaluate(a)[:2],
        np.ones(y.shape[1]),
        jac=True,
        bounds=[(1e-8, None)] * y.shape[1],
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-9},
    )
    alpha = result.x.copy()
    for _ in range(3):
        _, gradient, hessian = evaluate(alpha)
        alpha -= np.linalg.solve(hessian, gradient)
    assert (alpha > 0).all()
    assert np.linalg.norm(evaluate(alpha)[1], ord=np.inf) < 2e-8
    return alpha, evaluate


def spline_profile(y, x, offsets, log_lambdas):
    """Independent profiled negative LAML and coefficients, up to a constant."""
    pairs = [
        cardinal(x, k)
        for k in (np.array([-1, -0.5, 0, 0.5, 1]), np.array([-1, -0.6, -0.2, 0.2, 0.6, 1]))
    ]
    designs = [pairs[0][0], pairs[1][0], np.ones((len(x), 1))]
    penalties = [np.exp(rho) * pair[1] for rho, pair in zip(log_lambdas, pairs)]
    ref = objective(y, designs, [*penalties, np.zeros((1, 1))], offsets, np.ones(len(x)))
    beta = stationary(ref, 12)
    value, _, hessian = ref(beta)
    # Each natural cubic penalty has exactly two null directions (constant/line).
    logdet_s = sum(np.log(np.linalg.eigvalsh(p)[2:]).sum() for p in penalties)
    laml = value + 0.5 * np.linalg.slogdet(hessian)[1] - 0.5 * logdet_s
    return laml, beta, hessian, designs
