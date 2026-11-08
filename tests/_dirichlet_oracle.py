"""Independent normalized-density references; no production kernel imports."""

from itertools import combinations_with_replacement

import mpmath as mp
import numpy as np
from scipy import optimize, special


def mp_row(response, alpha):
    """Differentiate the ordinary log-gamma density at exact binary inputs."""
    with mp.workdps(100):
        x = list(map(mp.mpf, map(float, response)))
        y = [v / mp.fsum(x) for v in x]
        a = list(map(mp.mpf, map(float, alpha)))
        total = mp.fsum(a)
        value = mp.loggamma(total) - mp.fsum(map(mp.loggamma, a))
        value += mp.fsum((v - 1) * mp.log(z) for v, z in zip(a, y, strict=True))
        score = [mp.digamma(total) - mp.digamma(v) + mp.log(z) for v, z in zip(a, y)]
        hessian = [
            mp.polygamma(1, total) - (mp.polygamma(1, a[j]) if j == k else 0)
            for j, k in combinations_with_replacement(range(len(a)), 2)
        ]
        return float(value), np.array(score, float), np.array(hessian, float)


def mp_curvature_direction(response, eta, first, second=None, *, identity=False):
    """Differentiate the linked Hessian with independent mpmath variables."""
    with mp.workdps(70):
        x = list(map(mp.mpf, map(float, response)))
        logy = [mp.log(v / mp.fsum(x)) for v in x]
        eta = list(map(mp.mpf, map(float, eta)))
        first = list(map(mp.mpf, map(float, first)))
        second = [mp.mpf(0)] * len(eta) if second is None else list(map(mp.mpf, map(float, second)))

        def hessian(t, u, j, k):
            z = [e + t * v + u * w for e, v, w in zip(eta, first, second)]
            a = z if identity else list(map(mp.exp, z))
            total = mp.fsum(a)
            h = mp.polygamma(1, total) - (mp.polygamma(1, a[j]) if j == k else 0)
            if not identity:
                h *= a[j] * a[k]
                if j == k:
                    h += a[j] * (mp.digamma(total) - mp.digamma(a[j]) + logy[j])
            return -h

        order = (1, 0) if all(v == 0 for v in second) else (1, 1)
        return np.array(
            [
                float(mp.diff(lambda t, u: hessian(t, u, j, k), (0, 0), order))
                for j, k in combinations_with_replacement(range(len(eta)), 2)
            ]
        )


def fit_reference(y, design, weights):
    """Small ordinary-domain optimizer, independent density/gradient/Hessian."""
    n, k = y.shape
    p = design.shape[1]
    logy = np.log(y / y.sum(axis=1, keepdims=True))

    def objective(beta):
        alpha = np.exp(design @ beta.reshape(k, p).T)
        total = alpha.sum(axis=1)
        value = special.gammaln(total) - special.gammaln(alpha).sum(axis=1)
        value += ((alpha - 1) * logy).sum(axis=1)
        score = alpha * (special.digamma(total)[:, None] - special.digamma(alpha) + logy)
        gradient = -(score * weights[:, None]).T @ design
        blocks = np.empty((k, p, k, p))
        for j in range(k):
            for column in range(k):
                h = alpha[:, j] * alpha[:, column] * special.polygamma(1, total)
                if j == column:
                    h -= alpha[:, j] ** 2 * special.polygamma(1, alpha[:, j])
                    h += score[:, j]
                blocks[j, :, column, :] = -design.T @ ((h * weights)[:, None] * design)
        return -float(weights @ value), gradient.ravel(), blocks.reshape(k * p, k * p)

    start = np.zeros(k * p)
    result = optimize.minimize(
        lambda b: objective(b)[0],
        start,
        jac=lambda b: objective(b)[1],
        hess=lambda b: objective(b)[2],
        method="trust-exact",
        options={"gtol": 2e-9},
    )
    # Polish with Newton: optimizer flags are not stationarity evidence.
    beta = result.x.copy()
    for _ in range(4):
        _, score, hessian = objective(beta)
        beta -= np.linalg.solve(hessian, score)
    assert np.linalg.norm(objective(beta)[1], ord=np.inf) < 2e-8
    return beta, objective
